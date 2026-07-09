# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V3.2 DSA sparse sub-kernel collector for SGLang.

Thin wrapper over ``glm5_dsa_sparse_modules``. DeepSeek-V3.2 and GLM-5 share the
SAME DSA prefill kernels (deep_gemm.fp8_mqa_logits + fast_topk_v2); shapes/dims
are read DYNAMICALLY from the model config, so the GLM-5 worker is reused
verbatim — only the architecture tag (``DeepseekV32ForCausalLM``) and output
filenames (``dsv32_*``) differ, passed as parameters.

DSV3.2 only needs ``mqa`` + ``topk`` for the CP delta in
``ContextDSAModule._query_cp`` (``dsa_attn`` is not used by the delta).
"""

from __future__ import annotations

from collector.sglang.glm5_dsa_sparse_modules import (
    _dsa_context_derived_shapes,
    _dsa_generation_derived_shapes,
    run_glm5_dsa_sparse_kernel_worker,
)

__all__ = [
    "get_dsv32_mqa_test_cases",
    "get_dsv32_topk_test_cases",
    "run_dsv32_dsa_sparse_kernel_worker",
]

DSV32_DEFAULT_MODEL = "deepseek-ai/DeepSeek-V3.2-Exp"
DSV32_ARCHITECTURE = "DeepseekV32ForCausalLM"
KERNEL_TO_OP_NAME = {
    "mqa": "dsv32_mqa_logits_module",
    "topk": "dsv32_topk_module",
}


def run_dsv32_dsa_sparse_kernel_worker(model_path, kernel, bs_only, *, perf_filename, device="cuda:0"):
    """Reuse the GLM-5 worker with DeepSeek-V3.2's architecture tag + dsv32_* names."""
    return run_glm5_dsa_sparse_kernel_worker(
        model_path,
        kernel,
        bs_only,
        perf_filename=perf_filename,
        device=device,
        architecture=DSV32_ARCHITECTURE,
        op_name_map=KERNEL_TO_OP_NAME,
        label="dsv32",
    )


def _dsv32_sparse_kernel_cases(kernel):
    cases = []
    for m in [DSV32_DEFAULT_MODEL]:
        ctx = _dsa_context_derived_shapes(m)
        dec = _dsa_generation_derived_shapes(m)
        bss = sorted({b for (_p, _i, b) in ctx} | {b for (_p, _i, b) in dec})
        cases.extend([m, kernel, b] for b in bss)
    return cases


def get_dsv32_mqa_test_cases():
    return _dsv32_sparse_kernel_cases("mqa")


def get_dsv32_topk_test_cases():
    return _dsv32_sparse_kernel_cases("topk")
