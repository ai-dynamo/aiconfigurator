# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vision (ViT-style) attention collector for multimodal models.

Non-causal, MHA, no KV cache. Calls SGLang's vision ``flash_attn_varlen_func``
directly with ``causal=False``, matching the path used by ``VisionFlash3Attention``
(see ``sglang/srt/layers/attention/vision.py:385-449``).  No paged KV cache pool
is created — vision is single-pass.
"""

__compat__ = "sglang>=0.5.10rc0"

from typing import NamedTuple

import pkg_resources
import torch

from collector.helper import benchmark_with_power, log_perf


class Timing(NamedTuple):
    mean: float


def run_vision_attention_torch(
    batch_size,
    seq_len,
    num_heads,
    head_dim,
    *,
    perf_filename,
    device="cuda:0",
):
    from sglang.jit_kernel.flash_attention import flash_attn_varlen_func

    torch_device = torch.device(device)
    torch.cuda.set_device(device)

    total_tokens = batch_size * seq_len
    dtype = torch.bfloat16
    q = torch.randn(total_tokens, num_heads, head_dim, device=torch_device, dtype=dtype)
    k = torch.randn(total_tokens, num_heads, head_dim, device=torch_device, dtype=dtype)
    v = torch.randn(total_tokens, num_heads, head_dim, device=torch_device, dtype=dtype)

    cu_seqlens = torch.arange(
        0, (batch_size + 1) * seq_len, step=seq_len, dtype=torch.int32, device=torch_device
    )

    def run_iter():
        flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=seq_len,
            max_seqlen_k=seq_len,
            causal=False,
        )

    with benchmark_with_power(
        device=torch_device, kernel_func=run_iter, num_warmups=3, num_runs=20, repeat_n=1,
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
        op_name="vision_attention",
        kernel_source="sglang_vision_fa",
        perf_filename=perf_filename,
        power_stats=results["power_stats"],
    )

    return Timing(latency * 1e-3)


def get_vision_attention_test_cases():
    """ViT-only matrix: MHA, bf16, non-causal."""
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
    from collector.registry_types import PerfFile

    for test_case in get_vision_attention_test_cases()[:5]:
        print(f"Running vision attention test case: {test_case}")
        run_vision_attention_torch(*test_case, perf_filename=PerfFile.VISION_ATTENTION)
