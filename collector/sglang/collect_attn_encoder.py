# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang encoder (non-causal) attention collector for multimodal / omni-modal models.

SM dispatch mirrors ``VisionAttention._determine_attention_backend``:

- CUDA SM == 90 (Hopper)     -> ``flash_attn_varlen_func``           (FA3)
- CUDA SM == 100 (Blackwell) -> ``flash_attn_varlen_func(ver=4)``    (FA4)
- other CUDA (SM<90, SM120)  -> ``context_attention_fwd``            (Triton)

Quant: bf16 only. SGLang upstream does not support fp8 ViT FMHA.
"""

__compat__ = "sglang==0.5.14"

from typing import NamedTuple

import pkg_resources
import torch

from collector.case_generator import get_attention_encoder_head_configs, get_attention_encoder_shape_sweeps
from collector.helper import benchmark_with_power, get_sm_version, log_perf


class Timing(NamedTuple):
    mean: float


def _int_list(values):
    return [int(value) for value in values]


def get_encoder_attention_test_cases():
    test_cases = []

    for shape_sweep in get_attention_encoder_shape_sweeps("sglang"):
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


def _build_kernel_runner(
    device: torch.device,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
):
    """Return ``(run_iter, backend_tag)`` for the given device and shape."""
    if device.type != "cuda":
        raise RuntimeError(f"encoder attention collector requires CUDA device, got {device}")

    total_tokens = batch_size * seq_len
    q = torch.randn(total_tokens, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(total_tokens, num_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(total_tokens, num_heads, head_dim, device=device, dtype=dtype)

    # Uniform batch: cu_seqlens = [0, s, 2s, ..., b*s], max_seqlen = s
    cu_seqlens = torch.arange(
        0,
        (batch_size + 1) * seq_len,
        step=seq_len,
        dtype=torch.int32,
        device=device,
    )
    max_seqlen = seq_len
    softmax_scale = head_dim**-0.5

    sm = get_sm_version()  # 90 / 100 / 120 / ...

    if sm == 90:
        # Matches VisionFlash3Attention.forward.
        from sglang.jit_kernel.flash_attention import flash_attn_varlen_func

        def run_iter():
            flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                softmax_scale=softmax_scale,
                window_size=(-1, -1),
            )

        return run_iter, "flash_attention_v3"

    if sm == 100:
        # Matches VisionFlash4Attention.forward.
        from sglang.jit_kernel.flash_attention import flash_attn_varlen_func

        def run_iter():
            flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                softmax_scale=softmax_scale,
                ver=4,
            )

        return run_iter, "flash_attention_v4"

    # SM<90 or SM>100: Triton path matching VisionTritonAttention.forward.
    from sglang.srt.layers.attention.triton_ops.prefill_attention import (
        context_attention_fwd,
    )

    seq_lens = torch.full(
        (batch_size,),
        seq_len,
        dtype=torch.int32,
        device=device,
    )
    output = torch.empty_like(q)

    def run_iter():
        context_attention_fwd(
            q,
            k,
            v,
            output,
            cu_seqlens,
            seq_lens,
            max_seqlen,
            is_causal=False,
            sm_scale=softmax_scale,
        )

    return run_iter, "triton"


def run_encoder_attention_torch(
    batch_size,
    seq_len,
    num_heads,
    head_dim,
    *,
    perf_filename,
    device="cuda:0",
):
    torch_device = torch.device(device)
    torch.cuda.set_device(device)

    run_iter, backend_tag = _build_kernel_runner(
        device=torch_device,
        batch_size=batch_size,
        seq_len=seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        dtype=torch.bfloat16,
    )

    with benchmark_with_power(
        device=torch_device,
        kernel_func=run_iter,
        num_warmups=3,
        num_runs=20,
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
        framework="SGLang",
        version=pkg_resources.get_distribution("sglang").version,
        device_name=torch.cuda.get_device_name(device),
        op_name="encoder_attention",
        kernel_source=backend_tag,
        perf_filename=perf_filename,
        power_stats=results["power_stats"],
    )

    return Timing(latency * 1e-3)


if __name__ == "__main__":
    from collector.registry_types import PerfFile

    for test_case in get_encoder_attention_test_cases():
        try:
            run_encoder_attention_torch(*test_case, perf_filename=PerfFile.ENCODER_ATTENTION)
        except Exception as e:
            print(f"[encoder_attention] case {test_case} failed: {type(e).__name__}: {e}")
