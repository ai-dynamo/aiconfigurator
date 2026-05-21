# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Encoder (non-causal) attention collector for multimodal encoders.

Directly invokes the ViT wrappers used by vLLM's ``MMEncoderAttention`` path
(``vit_flash_attn_wrapper`` / ``vit_triton_attn_wrapper`` / ``vit_torch_sdpa_wrapper``).
"""

__compat__ = "vllm>=0.21.0"

import torch
from vllm.model_executor.models.vision import get_vit_attn_backend
from vllm.v1.attention.backends.fa_utils import get_flash_attn_version
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.ops.vit_attn_wrappers import (
    vit_flash_attn_wrapper,
    vit_torch_sdpa_wrapper,
    vit_triton_attn_wrapper,
)
from vllm.version import __version__ as vllm_version

from collector.helper import benchmark_with_power, log_perf


def run_encoder_attention_torch(
    batch_size,
    seq_len,
    num_heads,
    head_dim,
    *,
    perf_filename,
    device="cuda:0",
):
    torch.cuda.set_device(device)
    dtype = torch.bfloat16
    scale = 1.0 / (head_dim**0.5)

    # Same selector MMEncoderAttention uses (cuda.py:get_supported_vit_attn_backends).
    backend = get_vit_attn_backend(head_size=head_dim, dtype=dtype)

    # ViT wrappers expect (B, S, N, D); internal einops rearrange handles varlen.
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype, device=device)

    # Pre-generate cu_seqlens so the FA/Triton wrappers skip their internal ``torch.arange`` fallback 
    cu_seqlens = torch.arange(
        0, (batch_size + 1) * seq_len, step=seq_len,
        dtype=torch.int32, device=device,
    )

    if backend == AttentionBackendEnum.FLASH_ATTN:
        fa_version = get_flash_attn_version(head_size=head_dim)
        run = lambda: vit_flash_attn_wrapper(
            q, k, v,
            batch_size=batch_size,
            is_rocm_aiter=False,
            fa_version=fa_version,
            scale=scale,
            cu_seqlens=cu_seqlens,
        )
    elif backend == AttentionBackendEnum.TRITON_ATTN:
        run = lambda: vit_triton_attn_wrapper(
            q, k, v, batch_size=batch_size, scale=scale, cu_seqlens=cu_seqlens,
        )
    elif backend == AttentionBackendEnum.TORCH_SDPA:
        run = lambda: vit_torch_sdpa_wrapper(q, k, v, scale=scale)
    else:
        # FlashInfer ViT needs cu_seqlens padding + workspace; not on the ViT default path.
        raise NotImplementedError(f"ViT backend {backend} not supported by collector")

    with benchmark_with_power(
        device=device, kernel_func=run, num_warmups=3, num_runs=6, repeat_n=1,
    ) as results:
        pass

    latency = results["latency_ms"]
    print(f"encoder attn latency: {latency}")

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
        framework="VLLM",
        version=vllm_version,
        device_name=torch.cuda.get_device_name(device),
        op_name="encoder_attention",
        kernel_source=f"vllm_vit_{backend.name}".lower(),
        perf_filename=perf_filename,
        power_stats=results["power_stats"],
    )


def get_encoder_attention_test_cases(if_unit_test=False):
    if if_unit_test:
        return [[1, 256, 16, 72]]

    b_list = [1, 2, 4, 8, 16, 32, 64]
    s_list = [13, 16, 26, 32, 52, 64, 104, 128, 192, 256, 400, 512, 576,
              1024, 1296, 1500, 1536, 2048, 2304, 3072, 3136, 4096, 5184,
              6144, 6400, 7744, 8192, 9216, 10240, 10816, 12288, 12544,
              14400, 16384, 24576, 32768, 49152, 65536]

    n_list = [2, 4, 5, 8, 10, 16, 20]

    head_dim_list = [64, 72, 80, 88, 96, 128]

    test_cases = []
    for head_dim in head_dim_list:
        for n in sorted(n_list, reverse=True):
            for s in sorted(s_list, reverse=True):
                for b in sorted(b_list, reverse=True):
                    # Workload token budget guard (max 128K tokens)
                    if b * s > 131072:
                        continue
                    test_cases.append([b, s, n, head_dim])
    return test_cases


if __name__ == "__main__":
    from collector.registry_types import PerfFile

    for test_case in get_encoder_attention_test_cases()[:5]:
        print(f"Running encoder attention test case: {test_case}")
        run_encoder_attention_torch(*test_case, perf_filename=PerfFile.ENCODER_ATTENTION)
