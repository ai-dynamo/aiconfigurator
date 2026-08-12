# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM MLA generation BMM micro-collector.

Benchmarks the batched-matrix-multiply helpers used around MLA generation
pre/post processing (the ``W_UK_T`` absorb and ``W_UV`` projection). On
NVIDIA, vLLM runs BOTH as plain bf16 ``torch.bmm`` — the fp8 BMM variants
are ROCm/AITER-gated (``rocm_aiter_ops.triton_fp8_bmm``):

- stock MLA: mla_attention.py:864 (``torch.bmm(mqa_q_nope, W_UK_T, ...)``)
  and :1152 (``torch.bmm(x, self.W_UV, ...)``) @ 0.24-family builds;
- Kimi-K3 custom MLA: models/kimi_k3/nvidia/mla.py:569 (BMM1 absorb) and
  :401 (W_UV projection) @ the kimi-k3 preview — same torch.bmm.

The lane therefore collects the ``bfloat16`` dtype only; an fp8 row would
label a kernel NVIDIA vLLM never dispatches. Until this collector existed,
vLLM MLA-BMM queries fell back to the trtllm tables (the SDK's
"low-fidelity fallback rows for mla_bmm_perf" warning).
"""

__compat__ = "vllm==0.24.0"

import pkg_resources
import torch

from collector.case_generator import get_mla_bmm_case_specs
from collector.helper import benchmark_with_power, log_perf

# NVIDIA vLLM dispatches these BMMs in bf16 only (module docstring citations);
# fp8 sweep entries would be mislabeled rows, not coverage.
_SUPPORTED_DTYPES = {"bfloat16"}


def _get_mla_bmm_test_cases(op_name: str):
    return [
        [case.num_tokens, case.num_heads, case.dtype, case.num_warmups, case.num_runs]
        for case in get_mla_bmm_case_specs("vllm", op_name)
        if case.dtype in _SUPPORTED_DTYPES
    ]


def get_mla_gen_pre_test_cases():
    return _get_mla_bmm_test_cases("mla_bmm_gen_pre")


def get_mla_gen_post_test_cases():
    return _get_mla_bmm_test_cases("mla_bmm_gen_post")


def _log_row(*, op_name, dtype, num_tokens, num_heads, results, perf_filename, device):
    if not log_perf(
        item_list=[
            {
                "bmm_dtype": dtype,
                "num_tokens": num_tokens,
                "num_heads": num_heads,
                "latency": results["latency_ms"],
            }
        ],
        framework="VLLM",
        version=pkg_resources.get_distribution("vllm").version,
        device_name=torch.cuda.get_device_name(device),
        op_name=op_name,
        kernel_source="vllm_torch_bmm",
        perf_filename=perf_filename,
        power_stats=results["power_stats"],
    ):
        raise RuntimeError(f"Failed to persist vLLM MLA BMM performance row to {perf_filename}")


def run_mla_gen_pre(num_tokens, num_heads, dtype, num_warmups, num_runs, *, perf_filename, device="cuda:0"):
    torch.cuda.set_device(device)
    torch.set_default_device(device)
    assert dtype == "bfloat16", "NVIDIA vLLM MLA pre-BMM is bf16 torch.bmm only"

    qk_nope_head_dim = 128
    kv_lora_rank = 512

    # BMM1 absorb: (N, B, P) x (N, P, L) — mla_attention.py:864 / kimi-k3
    # mla.py:569 shapes.
    q_nope = torch.randn((num_tokens, num_heads, qk_nope_head_dim), device=device, dtype=torch.bfloat16)
    w_uk_t = torch.randn((num_heads, qk_nope_head_dim, kv_lora_rank), device=device, dtype=torch.bfloat16)
    out = torch.empty((num_heads, num_tokens, kv_lora_rank), device=device, dtype=torch.bfloat16)

    def kernel_func():
        torch.bmm(q_nope.transpose(0, 1), w_uk_t, out=out)

    with benchmark_with_power(
        device=device,
        kernel_func=kernel_func,
        num_warmups=num_warmups,
        num_runs=num_runs,
        repeat_n=1,
    ) as results:
        pass

    _log_row(
        op_name="mla_gen_pre",
        dtype=dtype,
        num_tokens=num_tokens,
        num_heads=num_heads,
        results=results,
        perf_filename=perf_filename,
        device=device,
    )


def run_mla_gen_post(num_tokens, num_heads, dtype, num_warmups, num_runs, *, perf_filename, device="cuda:0"):
    torch.cuda.set_device(device)
    torch.set_default_device(device)
    assert dtype == "bfloat16", "NVIDIA vLLM MLA post-BMM is bf16 torch.bmm only"

    kv_lora_rank = 512
    v_head_dim = 128

    # W_UV projection: (N, B, L) x (N, L, V) with a transposed-view output —
    # mla_attention.py:1152 / kimi-k3 mla.py:401 shapes.
    attn_output = torch.randn((num_tokens, num_heads, kv_lora_rank), device=device, dtype=torch.bfloat16)
    w_uv = torch.randn((num_heads, kv_lora_rank, v_head_dim), device=device, dtype=torch.bfloat16)
    attn_bmm_output = torch.empty((num_tokens, num_heads, v_head_dim), device=device, dtype=torch.bfloat16)

    def kernel_func():
        torch.bmm(
            attn_output.transpose(0, 1),
            w_uv,
            out=attn_bmm_output.transpose(0, 1),
        )

    with benchmark_with_power(
        device=device,
        kernel_func=kernel_func,
        num_warmups=num_warmups,
        num_runs=num_runs,
        repeat_n=1,
    ) as results:
        pass

    _log_row(
        op_name="mla_gen_post",
        dtype=dtype,
        num_tokens=num_tokens,
        num_heads=num_heads,
        results=results,
        perf_filename=perf_filename,
        device=device,
    )
