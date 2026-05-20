# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Encoder (non-causal) attention collector for multimodal / omni-modal models.

Covers ViT-style vision encoders, audio encoders, and any other bidirectional
encoder path: full N^2, MHA, no KV cache. We call the **same underlying kernels**
that sglang's production ``VisionAttention`` wrapper dispatches to per-SM, but
skip the wrapper's ``seq_lens.max().item()`` host sync (vision.py:435, 489, 369)
so the call can be captured into a CUDA graph — matching how the existing LLM
collectors (``collect_attn.py``) measure attention latency. Eager-mode timing
would include wrapper host overhead and would not be comparable to LLM data.

SM dispatch mirrors ``VisionAttention._determine_attention_backend``
(vision.py:934-975):

- CUDA SM == 90 (Hopper)     -> ``flash_attn_varlen_func``           (FA3)
- CUDA SM == 100 (Blackwell) -> ``flash_attn_varlen_func(ver=4)``    (FA4)
- other CUDA (SM<90, SM120)  -> ``context_attention_fwd``            (Triton)

For our uniform-shape sweep ``max_seqlen == seq_len`` is a known python int,
so the wrapper's host sync is unnecessary — the kernel arguments are byte-
identical to what production passes.
"""

__compat__ = "sglang>=0.5.10rc0"

from typing import NamedTuple

import pkg_resources
import torch

from collector.helper import benchmark_with_power, get_sm_version, log_perf


class Timing(NamedTuple):
    mean: float


def _build_kernel_runner(
    device: torch.device,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
):
    """Return ``(run_iter, backend_tag)`` for the given device and shape.

    ``run_iter`` calls the same kernel that the production sglang
    ``VisionAttention`` wrapper would dispatch on this SM, with arguments
    matched to vision.py:430/447/370/491 (non-cuda-graph branch). Inputs are
    closed over so ``run_iter()`` is CUDA-graph capturable (no host syncs).
    """
    if device.type != "cuda":
        raise RuntimeError(
            f"encoder attention collector requires CUDA device, got {device}"
        )

    total_tokens = batch_size * seq_len
    q = torch.randn(total_tokens, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(total_tokens, num_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(total_tokens, num_heads, head_dim, device=device, dtype=dtype)

    # Uniform batch: cu_seqlens = [0, s, 2s, ..., b*s], max_seqlen = s (python int)
    cu_seqlens = torch.arange(
        0, (batch_size + 1) * seq_len,
        step=seq_len, dtype=torch.int32, device=device,
    )
    max_seqlen = seq_len
    softmax_scale = head_dim ** -0.5

    sm = get_sm_version()  # 90 / 100 / 120 / ...

    if sm == 90:
        # Matches VisionFlash3Attention.forward (vision.py:437-447) non-graph branch.
        from sglang.jit_kernel.flash_attention import flash_attn_varlen_func

        def run_iter():
            flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                softmax_scale=softmax_scale,
                window_size=(-1, -1),
            )

        return run_iter, "flash_attention_v3"

    if sm == 100:
        # Matches VisionFlash4Attention.forward (vision.py:491-501).
        from sglang.jit_kernel.flash_attention import flash_attn_varlen_func

        def run_iter():
            flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                softmax_scale=softmax_scale,
                ver=4,
            )

        return run_iter, "flash_attention_v4"

    # SM<90 or SM>100 (e.g. SM80 A100, SM120 RTX 5090): Triton path,
    # matching VisionTritonAttention.forward (vision.py:362-381) non-graph branch.
    from sglang.srt.layers.attention.triton_ops.prefill_attention import (
        context_attention_fwd,
    )

    seq_lens = torch.full(
        (batch_size,), seq_len, dtype=torch.int32, device=device,
    )
    output = torch.empty_like(q)

    def run_iter():
        context_attention_fwd(
            q, k, v, output,
            cu_seqlens, seq_lens, max_seqlen,
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

    # Use benchmark_with_power context manager
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
                "seqlen": seq_len,
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
    from collector.registry_types import PerfFile

    for test_case in get_encoder_attention_test_cases()[:5]:
        print(f"Running encoder attention test case: {test_case}")
        run_encoder_attention_torch(*test_case, perf_filename=PerfFile.ENCODER_ATTENTION)
