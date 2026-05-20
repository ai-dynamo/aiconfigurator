# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Encoder (non-causal) attention collector for multimodal / omni-modal models.

Covers ViT-style vision encoders, audio encoders, and any other bidirectional
encoder path: full N^2, MHA, no KV cache. Uses the same TRT-LLM path that
Qwen2-VL ViT uses in production (see
``tensorrt_llm/_torch/models/modeling_qwen2vl.py``):
  - ``kv_cache_manager=None`` -> prepare() sets use_cache=False, kv_cache_block_offsets=None
  - ``attention_mask=PredefinedAttentionMask.FULL`` -> mMaskType=padding (non-causal)
  - ``pos_embd_params=None`` -> FMHA kernel does NOT do fused RoPE; production ViT
    applies RoPE outside the attention op via ``apply_rotary_pos_emb_vision`` (see
    ``Qwen2_5_VLVisionAttention.__init__`` in modeling_qwen2vl.py:518). The
    out-of-kernel RoPE cost is modeled separately by ``EncoderAttention.query()``
    using ``partial_rotary_factor``.
"""

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

    quant_config = QuantConfig(quant_algo=None, kv_cache_quant_algo=None, group_size=128)

    # pos_embd_params=None mirrors Qwen2_5_VLVisionAttention (modeling_qwen2vl.py:518).
    # Production ViT applies RoPE outside the FMHA kernel; if we passed a non-None
    # pos_embd_params here the FMHA kernel would perform a *fused* RoPE inside, the
    # collected latency would double-count RoPE against EncoderAttention.query()'s
    # partial_rotary_factor term.
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

    # No KV cache: encoder is single-pass. This mirrors Qwen2-VL ViT
    # (modeling_qwen2vl.py:708-717) which sets kv_cache_manager=None.
    attn_metadata = TrtllmAttentionMetadata(
        max_num_requests=batch_size,
        max_num_tokens=total_num_tokens,
        # kv_cache_manager=None requires max_seq_len to be set explicitly
        # (trtllm.py:813-830: "If the attention is no cache, max_seq_len
        # should be set manually by user").
        max_seq_len=seq_len,
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
    b_list = [1, 2, 4, 8, 16, 32, 64]
    s_list = [13, 16, 26, 32, 52, 64, 104, 128, 192, 256, 400, 512, 576,
              1024, 1296, 1500, 1536, 2048, 2304, 3072, 3136, 4096, 5184,
              6144, 6400, 7744, 8192, 9216, 10240, 10816, 12288, 12544,
              14400, 16384, 24576, 32768, 49152, 65536]

    n_list = [12, 16, 20, 24, 32]

    head_dim_list = [64, 72, 80, 88, 96, 128, 160]

    test_cases = []
    for head_dim in head_dim_list:
        for n in sorted(n_list, reverse=True):
            for s in sorted(s_list, reverse=True):
                for b in sorted(b_list, reverse=True):
                    if 4 * b * s * n * head_dim * 2 >= 2**31:
                        continue
                    test_cases.append([b, s, n, head_dim])
    return test_cases


if __name__ == "__main__":
    for test_case in get_encoder_attention_test_cases()[:5]:
        print(f"Running encoder attention test case: {test_case}")
        run_encoder_attention_torch(*test_case, perf_filename=PerfFile.ENCODER_ATTENTION)
