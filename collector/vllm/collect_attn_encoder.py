# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM encoder (non-causal) attention collector for multimodal encoders.

Directly invokes the ViT wrappers used by vLLM's ``MMEncoderAttention`` path
(``vit_flash_attn_wrapper`` / ``vit_triton_attn_wrapper`` / ``vit_torch_sdpa_wrapper``).

Quant: bf16 only. vLLM upstream supports fp8 ViT FMHA via FLASHINFER;
enabling that path here is left for the future.
"""

__compat__ = "vllm==0.24.0"

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

from collector.case_generator import get_attention_encoder_head_configs, get_attention_encoder_shape_sweeps
from collector.helper import benchmark_with_power, log_perf


def _int_list(values):
    return [int(value) for value in values]


def get_encoder_attention_test_cases():
    test_cases = []

    for shape_sweep in get_attention_encoder_shape_sweeps("vllm"):
        batch_sizes = _int_list(shape_sweep["batch_sizes"])
        sequence_lengths = _int_list(shape_sweep["sequence_lengths"])

        for head_config in get_attention_encoder_head_configs(shape_sweep):
            n = head_config.num_heads
            head_dim = head_config.head_dim
            for s in sorted(sequence_lengths):
                for b in sorted(batch_sizes):
                    # Workload token budget (128K) + 32-bit indexing safety.
                    if b * s > 131072:
                        continue
                    if 4 * b * s * n * head_dim * 2 >= 2**31:
                        continue
                    test_cases.append([b, s, n, head_dim])

    return test_cases


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

    # Pre-generate cu_seqlens so the FA/Triton wrappers skip their internal ``torch.arange`` fallback.
    cu_seqlens = torch.arange(
        0,
        (batch_size + 1) * seq_len,
        step=seq_len,
        dtype=torch.int32,
        device=device,
    )

    if backend == AttentionBackendEnum.FLASH_ATTN:
        fa_version = get_flash_attn_version(head_size=head_dim)

        def run():
            vit_flash_attn_wrapper(
                q,
                k,
                v,
                batch_size=batch_size,
                is_rocm_aiter=False,
                fa_version=fa_version,
                scale=scale,
                cu_seqlens=cu_seqlens,
            )
    elif backend == AttentionBackendEnum.TRITON_ATTN:

        def run():
            vit_triton_attn_wrapper(
                q,
                k,
                v,
                batch_size=batch_size,
                scale=scale,
                cu_seqlens=cu_seqlens,
            )
    elif backend == AttentionBackendEnum.TORCH_SDPA:

        def run():
            vit_torch_sdpa_wrapper(
                q,
                k,
                v,
                scale=scale,
            )
    else:
        # FlashInfer ViT needs cu_seqlens padding + workspace; not on the ViT default path.
        raise NotImplementedError(f"ViT backend {backend} not supported by collector")

    with benchmark_with_power(
        device=device,
        kernel_func=run,
        num_warmups=3,
        num_runs=6,
        repeat_n=1,
    ) as results:
        pass

    latency = results["latency_ms"]
    print(f"encoder attn latency: {latency}")

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
        framework="VLLM",
        version=vllm_version,
        device_name=torch.cuda.get_device_name(device),
        op_name="encoder_attention",
        kernel_source=f"vllm_vit_{backend.name}".lower(),
        perf_filename=perf_filename,
        power_stats=results["power_stats"],
    )


if __name__ == "__main__":
    from collector.registry_types import PerfFile

    test_cases = get_encoder_attention_test_cases()
    for test_case in test_cases:
        try:
            run_encoder_attention_torch(*test_case, perf_filename=PerfFile.ENCODER_ATTENTION)
        except Exception as e:
            print(f"[encoder_attention] case {test_case} failed: {type(e).__name__}: {e}")
