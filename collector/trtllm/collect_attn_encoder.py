# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT-LLM encoder (non-causal) attention collector for multimodal / omni-modal models.

Builds a single TRT-LLM torch-flow attention layer running in encoder mode:

- ``kv_cache_manager=None``                        -> single-pass, no KV cache
- ``attention_mask=PredefinedAttentionMask.FULL``  -> non-causal (padding mask)
- ``pos_embd_params=None``                         -> RoPE applied outside FMHA

Quant: bf16 only. TRT-LLM upstream does not support fp8 on the encoder path.
"""

__compat__ = "trtllm>=1.3.0rc5"

import os

import tensorrt_llm
import torch
from tensorrt_llm._torch.attention_backend import TrtllmAttentionMetadata
from tensorrt_llm._torch.attention_backend.interface import (
    AttentionRuntimeFeatures,
    PredefinedAttentionMask,
)
from tensorrt_llm._torch.attention_backend.utils import create_attention
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig

from collector.case_generator import get_attention_encoder_shape_sweeps
from collector.helper import benchmark_with_power, get_sm_version, log_perf
from collector.planner.schemas.attention import build_encoder_attention_cases
from collector.registry_types import PerfFile


def get_encoder_attention_test_cases():
    return build_encoder_attention_cases(
        "trtllm",
        get_attention_encoder_shape_sweeps("trtllm"),
        sm_version=get_sm_version(),
        framework_version=tensorrt_llm.__version__,
    )


def run_encoder_attention_torch(
    batch_size,
    seq_len,
    num_heads,
    head_dim,
    *,
    perf_filename,
    device="cuda:0",
):
    device = torch.device(device)
    torch.set_default_device(device)
    torch.cuda.set_device(device)

    os.environ["TRTLLM_ENABLE_XQA_JIT"] = "0"

    quant_config = QuantConfig(quant_algo=None, kv_cache_quant_algo=None, group_size=128)

    # pos_embd_params=None: production ViT applies RoPE outside the FMHA kernel.
    attn = create_attention(
        backend_name="TRTLLM",
        layer_idx=0,
        num_heads=num_heads,
        head_dim=head_dim,
        num_kv_heads=num_heads,  # MHA
        pos_embd_params=None,
        quant_config=quant_config,
        is_mla_enable=False,
    )

    mapping = Mapping(world_size=1, rank=0, tp_size=1)
    total_num_tokens = seq_len * batch_size
    input_seq_lens = [seq_len] * batch_size
    request_ids = list(range(batch_size))

    # No KV cache: encoder is single-pass.
    attn_metadata = TrtllmAttentionMetadata(
        max_num_requests=batch_size,
        max_num_tokens=total_num_tokens,
        # _max_seq_len_storage must be set explicitly when kv_cache_manager=None.
        _max_seq_len_storage=seq_len,
        kv_cache_manager=None,
        mapping=mapping,
        enable_flash_mla=False,
        seq_lens=torch.tensor(input_seq_lens, dtype=torch.int32, device="cpu"),
        num_contexts=batch_size,
        position_ids=None,
        kv_cache_params=KVCacheParams(use_cache=False),
        cross=None,
        request_ids=request_ids,
        prompt_lens=input_seq_lens,
        runtime_features=AttentionRuntimeFeatures(
            chunked_prefill=False, cache_reuse=False, has_speculative_draft_tokens=False
        ),
        all_rank_num_tokens=None,
        workspace=torch.tensor([], device=device, dtype=torch.int8),
    )
    attn_metadata.prepare()

    q = torch.randn([total_num_tokens, num_heads * head_dim]).bfloat16().to(device)
    kv = torch.randn([total_num_tokens, 2 * num_heads * head_dim]).bfloat16().to(device)
    input_qkv = torch.concat([q, kv], dim=-1)

    def kernel_func():
        attn.forward(
            input_qkv,
            None,
            None,
            attn_metadata,
            attention_mask=PredefinedAttentionMask.FULL,
        )

    with benchmark_with_power(
        device=device,
        kernel_func=kernel_func,
        num_warmups=10,
        num_runs=6,
        repeat_n=1,
    ) as results:
        pass

    latency = results["latency_ms"]
    log_perf(
        item_list=[
            {
                "batch_size": batch_size,
                "isl": seq_len,
                "num_heads": num_heads,
                "head_dim": head_dim,
                "attn_dtype": "bfloat16",
                "latency": latency,
            }
        ],
        framework="TRTLLM",
        version=tensorrt_llm.__version__,
        device_name=torch.cuda.get_device_name(device),
        op_name="encoder_attention",
        kernel_source="torch_flow",
        perf_filename=perf_filename,
        power_stats=results["power_stats"],
    )


if __name__ == "__main__":
    test_cases = get_encoder_attention_test_cases()
    for test_case in test_cases:
        try:
            run_encoder_attention_torch(*test_case, perf_filename=PerfFile.ENCODER_ATTENTION)
        except Exception as e:
            print(f"[encoder_attention] case {test_case} failed: {type(e).__name__}: {e}")
