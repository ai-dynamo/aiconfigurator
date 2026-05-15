# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Encoder (non-causal) attention collector for multimodal / omni-modal models.

Covers ViT-style vision encoders, audio encoders, and any other bidirectional
encoder path: full N^2, MHA, no KV cache. Uses the same TRT-LLM path that
Qwen2-VL ViT uses in production (see
``tensorrt_llm/_torch/models/modeling_qwen2vl.py``):
  - ``kv_cache_manager=None`` -> prepare() sets use_cache=False, kv_cache_block_offsets=None
  - ``attention_mask=PredefinedAttentionMask.FULL`` -> mMaskType=padding (non-causal)
"""

import os

import tensorrt_llm
import torch
from tensorrt_llm._torch.attention_backend import TrtllmAttentionMetadata
from tensorrt_llm._torch.attention_backend.interface import (
    AttentionRuntimeFeatures,
    PositionalEmbeddingParams,
    PredefinedAttentionMask,
    RopeParams,
)
from tensorrt_llm._torch.attention_backend.utils import create_attention
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig

from collector.helper import benchmark_with_power, log_perf
from collector.registry_types import PerfFile


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

    pos_embd_params = PositionalEmbeddingParams(type=PositionEmbeddingType.rope_gpt_neox, rope=RopeParams(dim=head_dim))
    quant_config = QuantConfig(quant_algo=None, kv_cache_quant_algo=None, group_size=128)

    attn = create_attention(
        backend_name="TRTLLM",
        layer_idx=0,
        num_heads=num_heads,
        head_dim=head_dim,
        num_kv_heads=num_heads,  # MHA
        pos_embd_params=pos_embd_params,
        quant_config=quant_config,
        is_mla_enable=False,
    )

    mapping = Mapping(world_size=1, rank=0, tp_size=1)
    total_num_tokens = seq_len * batch_size
    input_seq_lens = [seq_len] * batch_size
    request_ids = list(range(batch_size))

    # No KV cache: encoder is single-pass. This mirrors Qwen2-VL ViT
    # (modeling_qwen2vl.py:708-717) which sets kv_cache_manager=None.
    attn_metadata = TrtllmAttentionMetadata(
        max_num_requests=batch_size,
        max_num_tokens=total_num_tokens,
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

    # Warmup once
    kernel_func()

    with benchmark_with_power(
        device=device, kernel_func=kernel_func, num_warmups=10, num_runs=6, repeat_n=1,
    ) as results:
        pass

    latency = results["latency_ms"]
    log_perf(
        item_list=[
            {
                "batch_size": batch_size,
                "seqlen": seq_len,
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


def get_encoder_attention_test_cases():
    """Encoder matrix: MHA, bf16, non-causal."""
    b_list = [1, 2, 4, 8, 16, 32, 64]
    s_list = [256, 512, 1024, 2048, 4096, 8192, 16384]
    n_list = [12, 16, 24, 32]
    head_dim_list = [72, 80, 128]

    test_cases = []
    for head_dim in head_dim_list:
        for n in n_list:
            for s in s_list:
                for b in b_list:
                    if 4 * b * s * n * head_dim * 2 >= 2**31:
                        continue
                    test_cases.append([b, s, n, head_dim])
    return test_cases


if __name__ == "__main__":
    for test_case in get_encoder_attention_test_cases()[:5]:
        print(f"Running encoder attention test case: {test_case}")
        run_encoder_attention_torch(*test_case, perf_filename=PerfFile.ENCODER_ATTENTION)
