# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Mamba2 SSM Collector for AIConfigurator.

This collector benchmarks the core Mamba2 operations (Conv1D + SSM combined):

Context (prefill) phase:
    - causal_conv1d_fn: Applies causal 1D convolution over the sequence
    - mamba_chunk_scan_combined: SSM scan using chunked algorithm

Generation (decode) phase:
    - causal_conv1d_update: Updates conv state and produces output for single token
    - selective_state_update: Updates SSM state and produces output for single token

The in_proj and out_proj GEMMs are standard linear layers that can be modeled
using the existing GEMM infrastructure. This collector focuses on the unique
Conv1D + SSM operations that are specific to Mamba2.

Mamba2 Layer Flow:
    in_proj (GEMM) → Conv1D → SSM Scan/Update → out_proj (GEMM)
    ^^^^^^^^^^^^^    ^^^^^^^^^^^^^^^^^^^^^^^    ^^^^^^^^^^^^^^^^
    Use GEMM model       Benchmarked here       Use GEMM model

Usage:
    python collect_mamba2.py

Output:
    mamba2_perf.txt - Performance data for Mamba2 Conv1D+SSM operations
"""

import gc
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # keep these imports to satisfy static analysis (ruff F821).
    from tensorrt_llm._torch.modules.mamba.causal_conv1d import causal_conv1d_fn, causal_conv1d_update
    from tensorrt_llm._torch.modules.mamba.selective_state_update import selective_state_update
    from tensorrt_llm._torch.modules.mamba.ssd_combined import mamba_chunk_scan_combined
import torch
from einops import repeat

try:
    from common_test_cases import get_common_mamba2_test_cases

    from helper import (
        EXIT_CODE_RESTART,
        benchmark_with_power,
        get_sm_version,
        log_perf,
    )
except ModuleNotFoundError:
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from common_test_cases import get_common_mamba2_test_cases

    from helper import (
        EXIT_CODE_RESTART,
        benchmark_with_power,
        get_sm_version,
        log_perf,
    )

aic_debug = int(os.getenv("aic_mamba2_debug", "0"))  # noqa: SIM112
# Use cached inputs (same data each iteration) instead of randomized inputs
# Set AIC_MAMBA2_CACHED_INPUTS=1 to enable cached mode (saves memory but may be overly optimistic)
aic_cached_inputs = int(os.getenv("AIC_MAMBA2_CACHED_INPUTS", "0"))


def get_mamba2_test_cases():
    """
    Generate test cases for Mamba2 SSM benchmarking.

    Returns a list of test case configurations for both context (prefill)
    and generation (decode) phases.

    Context phase uses static batching (batch_size x seq_len) for P/D disaggregated serving.
    Generation phase uses batch_size only (1 token per request).
    """
    test_cases = []

    # Get common test cases from centralized definition
    for common_case in get_common_mamba2_test_cases():
        if common_case.phase == "context":
            # Context phase: sweep batch_size x seq_len
            test_cases.append(
                [
                    common_case.phase,
                    common_case.d_model,
                    common_case.d_state,
                    common_case.d_conv,
                    common_case.nheads,
                    common_case.head_dim,
                    common_case.n_groups,
                    common_case.chunk_size,
                    common_case.batch_size_list,
                    common_case.seq_len_list,
                    common_case.model_name,
                    "mamba2_perf.txt",
                ]
            )
        else:
            # Generation phase: sweep batch_size only
            test_cases.append(
                [
                    common_case.phase,
                    common_case.d_model,
                    common_case.d_state,
                    common_case.d_conv,
                    common_case.nheads,
                    common_case.head_dim,
                    common_case.n_groups,
                    common_case.chunk_size,
                    common_case.batch_size_list,
                    None,  # seq_len_list not used for generation
                    common_case.model_name,
                    "mamba2_perf.txt",
                ]
            )

    return test_cases


def run_mamba2_context_benchmark(
    d_model: int,
    d_state: int,
    d_conv: int,
    nheads: int,
    head_dim: int,
    n_groups: int,
    chunk_size: int,
    batch_size_list: list[int],
    seq_len_list: list[int],
    model_name: str,
    perf_filename: str,
    device: str = "cuda:0",
):
    """
    Benchmark Mamba2 SSM for context (prefill) phase.

    Uses static batching (batch_size x seq_len) for P/D disaggregated serving.

    This benchmarks:
    1. causal_conv1d_fn - Conv1D for context phase
    2. mamba_chunk_scan_combined - SSM scan

    Together these represent the core compute of a Mamba2 layer
    (excluding in_proj/out_proj GEMMs which use existing GEMM model).
    """
    device = torch.device(device)
    torch.cuda.set_device(device)
    torch.set_default_device(device)

    dtype = torch.bfloat16

    # Derived dimensions
    d_inner = nheads * head_dim
    conv_dim = d_inner + 2 * n_groups * d_state

    if aic_debug == 1:
        print(
            f"Mamba2 Context: d_model={d_model}, d_inner={d_inner}, "
            f"nheads={nheads}, head_dim={head_dim}, d_state={d_state}, conv_dim={conv_dim}"
        )

    # Create conv1d weights
    conv_weight = torch.randn(conv_dim, d_conv, dtype=dtype, device=device)
    conv_bias = torch.randn(conv_dim, dtype=dtype, device=device)

    # SSM parameters (uppercase A, B, C, D are standard SSM notation)
    A = -torch.rand(nheads, device=device) - 1.0  # noqa: N806
    D = torch.randn(nheads, device=device)  # noqa: N806
    dt_bias = torch.rand(nheads, device=device) - 4.0

    # Sweep over batch_size x seq_len combinations
    for batch_size in batch_size_list:
        for seq_len in seq_len_list:
            if aic_debug == 1:
                print(f"  Benchmarking batch_size={batch_size}, seq_len={seq_len}")

            try:
                num_warmups = 3
                num_runs = 10
                total_iters = num_warmups + num_runs

                # Conv state cache: (batch, dim, width-1) - updated in-place by causal_conv1d_fn
                conv_state = torch.randn(batch_size, conv_dim, d_conv - 1, dtype=dtype, device=device)

                # Common log data for both kernels
                common_log_data = {
                    "phase": "context",
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "num_tokens": batch_size * seq_len,
                    "d_model": d_model,
                    "d_state": d_state,
                    "d_conv": d_conv,
                    "nheads": nheads,
                    "head_dim": head_dim,
                    "n_groups": n_groups,
                    "chunk_size": chunk_size,
                    "model_name": model_name,
                }

                if aic_cached_inputs:
                    # Cached mode: use same inputs for all iterations (saves memory)
                    xbc_input = torch.randn(batch_size, conv_dim, seq_len, dtype=dtype, device=device)
                    x = torch.randn(batch_size, seq_len, nheads, head_dim, dtype=dtype, device=device)
                    dt = torch.randn(batch_size, seq_len, nheads, dtype=dtype, device=device)
                    B = torch.randn(batch_size, seq_len, n_groups, d_state, dtype=dtype, device=device)  # noqa: N806
                    C = torch.randn(batch_size, seq_len, n_groups, d_state, dtype=dtype, device=device)  # noqa: N806

                    # --- Benchmark causal_conv1d_fn ---
                    torch.cuda.synchronize()
                    causal_conv1d_fn(xbc_input, conv_weight, conv_bias, activation="silu", conv_states=conv_state)
                    torch.cuda.synchronize()

                    def run_conv1d(_xbc=xbc_input, _conv_state=conv_state):
                        causal_conv1d_fn(_xbc, conv_weight, conv_bias, activation="silu", conv_states=_conv_state)

                    with benchmark_with_power(
                        device=device,
                        kernel_func=run_conv1d,
                        num_warmups=num_warmups,
                        num_runs=num_runs,
                        repeat_n=1,
                        allow_graph_fail=True,
                    ) as results:
                        log_perf(
                            item_list=[{**common_log_data, "latency": results["latency_ms"]}],
                            framework="TRTLLM",
                            version=tensorrt_llm.__version__,
                            device_name=torch.cuda.get_device_name(device),
                            op_name="mamba2",
                            kernel_source="causal_conv1d_fn",
                            perf_filename=perf_filename,
                            power_stats=results["power_stats"],
                        )

                    # --- Benchmark mamba_chunk_scan_combined ---
                    torch.cuda.synchronize()
                    mamba_chunk_scan_combined(
                        x,
                        dt,
                        A,
                        B,
                        C,
                        chunk_size=chunk_size,
                        D=D,
                        z=None,
                        dt_bias=dt_bias,
                        dt_softplus=True,
                        return_final_states=True,
                    )
                    torch.cuda.synchronize()

                    def run_ssm_scan(_x=x, _dt=dt, _b=B, _c=C):
                        mamba_chunk_scan_combined(
                            _x,
                            _dt,
                            A,
                            _b,
                            _c,
                            chunk_size=chunk_size,
                            D=D,
                            z=None,
                            dt_bias=dt_bias,
                            dt_softplus=True,
                            return_final_states=True,
                        )

                    with benchmark_with_power(
                        device=device,
                        kernel_func=run_ssm_scan,
                        num_warmups=num_warmups,
                        num_runs=num_runs,
                        repeat_n=1,
                        allow_graph_fail=True,
                    ) as results:
                        log_perf(
                            item_list=[{**common_log_data, "latency": results["latency_ms"]}],
                            framework="TRTLLM",
                            version=tensorrt_llm.__version__,
                            device_name=torch.cuda.get_device_name(device),
                            op_name="mamba2",
                            kernel_source="mamba_chunk_scan_combined",
                            perf_filename=perf_filename,
                            power_stats=results["power_stats"],
                        )
                else:
                    # Randomized mode (default): pre-generate pool of random inputs
                    # to avoid L2 cache effects while excluding tensor creation from timing
                    input_pool = _make_input_pool(
                        {
                            "xbc": (batch_size, conv_dim, seq_len),
                            "x": (batch_size, seq_len, nheads, head_dim),
                            "dt": (batch_size, seq_len, nheads),
                            "B": (batch_size, seq_len, n_groups, d_state),
                            "C": (batch_size, seq_len, n_groups, d_state),
                        },
                        total_iters,
                        dtype,
                        device,
                    )

                    # Warmup with first set of inputs
                    torch.cuda.synchronize()
                    causal_conv1d_fn(
                        input_pool["xbc"][0], conv_weight, conv_bias, activation="silu", conv_states=conv_state
                    )
                    torch.cuda.synchronize()

                    conv1d_iter_idx = [0]

                    def run_conv1d(_pool=input_pool, _conv_state=conv_state, _idx=conv1d_iter_idx):
                        idx = _idx[0] % total_iters
                        _idx[0] += 1
                        causal_conv1d_fn(
                            _pool["xbc"][idx], conv_weight, conv_bias, activation="silu", conv_states=_conv_state
                        )

                    with benchmark_with_power(
                        device=device,
                        kernel_func=run_conv1d,
                        num_warmups=num_warmups,
                        num_runs=num_runs,
                        repeat_n=1,
                        allow_graph_fail=True,
                    ) as results:
                        log_perf(
                            item_list=[{**common_log_data, "latency": results["latency_ms"]}],
                            framework="TRTLLM",
                            version=tensorrt_llm.__version__,
                            device_name=torch.cuda.get_device_name(device),
                            op_name="mamba2",
                            kernel_source="causal_conv1d_fn",
                            perf_filename=perf_filename,
                            power_stats=results["power_stats"],
                        )

                    # --- Benchmark mamba_chunk_scan_combined ---
                    torch.cuda.synchronize()
                    mamba_chunk_scan_combined(
                        input_pool["x"][0],
                        input_pool["dt"][0],
                        A,
                        input_pool["B"][0],
                        input_pool["C"][0],
                        chunk_size=chunk_size,
                        D=D,
                        z=None,
                        dt_bias=dt_bias,
                        dt_softplus=True,
                        return_final_states=True,
                    )
                    torch.cuda.synchronize()

                    ssm_iter_idx = [0]

                    def run_ssm_scan(_pool=input_pool, _idx=ssm_iter_idx):
                        idx = _idx[0] % total_iters
                        _idx[0] += 1

                        mamba_chunk_scan_combined(
                            _pool["x"][idx],
                            _pool["dt"][idx],
                            A,
                            _pool["B"][idx],
                            _pool["C"][idx],
                            chunk_size=chunk_size,
                            D=D,
                            z=None,
                            dt_bias=dt_bias,
                            dt_softplus=True,
                            return_final_states=True,
                        )

                    with benchmark_with_power(
                        device=device,
                        kernel_func=run_ssm_scan,
                        num_warmups=num_warmups,
                        num_runs=num_runs,
                        repeat_n=1,
                        allow_graph_fail=True,
                    ) as results:
                        log_perf(
                            item_list=[{**common_log_data, "latency": results["latency_ms"]}],
                            framework="TRTLLM",
                            version=tensorrt_llm.__version__,
                            device_name=torch.cuda.get_device_name(device),
                            op_name="mamba2",
                            kernel_source="mamba_chunk_scan_combined",
                            perf_filename=perf_filename,
                            power_stats=results["power_stats"],
                        )

                # Cleanup
                if aic_cached_inputs:
                    del x, dt, B, C, xbc_input, conv_state
                else:
                    del input_pool, conv_state
                gc.collect()
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"  Error at batch_size={batch_size}, seq_len={seq_len}: {e}")
                continue


def run_mamba2_generation_benchmark(
    d_model: int,
    d_state: int,
    d_conv: int,
    nheads: int,
    head_dim: int,
    n_groups: int,
    chunk_size: int,
    batch_size_list: list[int],
    model_name: str,
    perf_filename: str,
    device: str = "cuda:0",
):
    """
    Benchmark Mamba2 SSM for generation (decode) phase.

    This benchmarks:
    1. causal_conv1d_update - Conv1D state update for decode
    2. selective_state_update - SSM state update

    Together these represent the core compute of a Mamba2 layer during decode
    (excluding in_proj/out_proj GEMMs which use existing GEMM model).
    """
    device = torch.device(device)
    torch.cuda.set_device(device)
    torch.set_default_device(device)

    dtype = torch.bfloat16

    # Derived dimensions
    d_inner = nheads * head_dim
    conv_dim = d_inner + 2 * n_groups * d_state

    if aic_debug == 1:
        print(
            f"Mamba2 Generation: d_model={d_model}, d_inner={d_inner}, "
            f"nheads={nheads}, head_dim={head_dim}, d_state={d_state}, conv_dim={conv_dim}"
        )

    # Conv1d weights
    conv_weight = torch.randn(conv_dim, d_conv, dtype=dtype, device=device)
    conv_bias = torch.randn(conv_dim, dtype=dtype, device=device)

    # SSM parameters - need to be expanded for generation (uppercase is standard SSM notation)
    a_base = -torch.rand(nheads, device=device) - 1.0
    d_base = torch.randn(nheads, device=device)
    dt_bias_base = torch.rand(nheads, device=device) - 4.0

    # Expand for generation (selective_state_update expects different shapes)
    A = repeat(a_base, "h -> h p n", p=head_dim, n=d_state).to(dtype=torch.float32)  # noqa: N806
    D = repeat(d_base, "h -> h p", p=head_dim)  # noqa: N806
    dt_bias = repeat(dt_bias_base, "h -> h p", p=head_dim)

    for batch_size in batch_size_list:
        if aic_debug == 1:
            print(f"  Benchmarking batch_size={batch_size}")

        try:
            num_warmups = 3
            num_runs = 10
            total_iters = num_warmups + num_runs
            # Conv1d state: [batch, dim, width-1] where width = d_conv
            conv_state = torch.randn(batch_size, conv_dim, d_conv - 1, dtype=dtype, device=device)

            # SSM state: [batch, nheads, head_dim, d_state]
            ssm_state = torch.randn(batch_size, nheads, head_dim, d_state, dtype=dtype, device=device)

            if aic_cached_inputs:
                # Common log data for both kernels
                common_log_data = {
                    "phase": "generation",
                    "batch_size": batch_size,
                    "seq_len": 1,
                    "num_tokens": batch_size,
                    "d_model": d_model,
                    "d_state": d_state,
                    "d_conv": d_conv,
                    "nheads": nheads,
                    "head_dim": head_dim,
                    "n_groups": n_groups,
                    "chunk_size": chunk_size,
                    "model_name": model_name,
                }
                # Cached mode: use same inputs for all iterations (saves memory)
                xbc_input = torch.randn(batch_size, conv_dim, dtype=dtype, device=device)
                x = torch.randn(batch_size, nheads, head_dim, dtype=dtype, device=device)
                dt = torch.randn(batch_size, nheads, head_dim, dtype=dtype, device=device)
                B = torch.randn(batch_size, n_groups, d_state, dtype=dtype, device=device)  # noqa: N806
                C = torch.randn(batch_size, n_groups, d_state, dtype=dtype, device=device)  # noqa: N806

                # --- Benchmark causal_conv1d_update ---
                torch.cuda.synchronize()
                causal_conv1d_update(xbc_input, conv_state, conv_weight, conv_bias, activation="silu")
                torch.cuda.synchronize()

                def run_conv1d_update(_xbc=xbc_input, _conv_state=conv_state):
                    causal_conv1d_update(_xbc, _conv_state, conv_weight, conv_bias, activation="silu")

                with benchmark_with_power(
                    device=device,
                    kernel_func=run_conv1d_update,
                    num_warmups=num_warmups,
                    num_runs=num_runs,
                    repeat_n=1,
                    allow_graph_fail=True,
                ) as results:
                    log_perf(
                        item_list=[{**common_log_data, "latency": results["latency_ms"]}],
                        framework="TRTLLM",
                        version=tensorrt_llm.__version__,
                        device_name=torch.cuda.get_device_name(device),
                        op_name="mamba2",
                        kernel_source="causal_conv1d_update",
                        perf_filename=perf_filename,
                        power_stats=results["power_stats"],
                    )

                # --- Benchmark selective_state_update ---
                torch.cuda.synchronize()
                selective_state_update(
                    ssm_state,
                    x,
                    dt,
                    A,
                    B,
                    C,
                    D,
                    z=None,
                    dt_bias=dt_bias,
                    dt_softplus=True,
                )
                torch.cuda.synchronize()

                def run_state_update(_ssm_state=ssm_state, _x=x, _dt=dt, _b=B, _c=C):
                    selective_state_update(
                        _ssm_state,
                        _x,
                        _dt,
                        A,
                        _b,
                        _c,
                        D,
                        z=None,
                        dt_bias=dt_bias,
                        dt_softplus=True,
                    )

                with benchmark_with_power(
                    device=device,
                    kernel_func=run_state_update,
                    num_warmups=num_warmups,
                    num_runs=num_runs,
                    repeat_n=1,
                    allow_graph_fail=True,
                ) as results:
                    log_perf(
                        item_list=[{**common_log_data, "latency": results["latency_ms"]}],
                        framework="TRTLLM",
                        version=tensorrt_llm.__version__,
                        device_name=torch.cuda.get_device_name(device),
                        op_name="mamba2",
                        kernel_source="selective_state_update",
                        perf_filename=perf_filename,
                        power_stats=results["power_stats"],
                    )
            else:
                # Common log data for both kernels
                common_log_data = {
                    "phase": "generation",
                    "batch_size": batch_size,
                    "seq_len": 1,
                    "num_tokens": batch_size,
                    "d_model": d_model,
                    "d_state": d_state,
                    "d_conv": d_conv,
                    "nheads": nheads,
                    "head_dim": head_dim,
                    "n_groups": n_groups,
                    "chunk_size": chunk_size,
                    "model_name": model_name,
                }
                # Randomized mode (default): pre-generate pool of random inputs
                input_pool = _make_input_pool(
                    {
                        "xbc": (batch_size, conv_dim),
                        "x": (batch_size, nheads, head_dim),
                        "dt": (batch_size, nheads, head_dim),
                        "B": (batch_size, n_groups, d_state),
                        "C": (batch_size, n_groups, d_state),
                    },
                    total_iters,
                    dtype,
                    device,
                )

                # --- Benchmark causal_conv1d_update ---
                torch.cuda.synchronize()
                causal_conv1d_update(input_pool["xbc"][0], conv_state, conv_weight, conv_bias, activation="silu")
                torch.cuda.synchronize()

                conv1d_iter_idx = [0]

                def run_conv1d_update(_pool=input_pool, _conv_state=conv_state, _idx=conv1d_iter_idx):
                    idx = _idx[0] % total_iters
                    _idx[0] += 1
                    causal_conv1d_update(_pool["xbc"][idx], _conv_state, conv_weight, conv_bias, activation="silu")

                with benchmark_with_power(
                    device=device,
                    kernel_func=run_conv1d_update,
                    num_warmups=num_warmups,
                    num_runs=num_runs,
                    repeat_n=1,
                    allow_graph_fail=True,
                ) as results:
                    log_perf(
                        item_list=[{**common_log_data, "latency": results["latency_ms"]}],
                        framework="TRTLLM",
                        version=tensorrt_llm.__version__,
                        device_name=torch.cuda.get_device_name(device),
                        op_name="mamba2",
                        kernel_source="causal_conv1d_update",
                        perf_filename=perf_filename,
                        power_stats=results["power_stats"],
                    )

                # --- Benchmark selective_state_update ---
                torch.cuda.synchronize()
                selective_state_update(
                    ssm_state,
                    input_pool["x"][0],
                    input_pool["dt"][0],
                    A,
                    input_pool["B"][0],
                    input_pool["C"][0],
                    D,
                    z=None,
                    dt_bias=dt_bias,
                    dt_softplus=True,
                )
                torch.cuda.synchronize()

                ssm_iter_idx = [0]

                def run_state_update(_pool=input_pool, _ssm_state=ssm_state, _idx=ssm_iter_idx):
                    idx = _idx[0] % total_iters
                    _idx[0] += 1
                    selective_state_update(
                        _ssm_state,
                        _pool["x"][idx],
                        _pool["dt"][idx],
                        A,
                        _pool["B"][idx],
                        _pool["C"][idx],
                        D,
                        z=None,
                        dt_bias=dt_bias,
                        dt_softplus=True,
                    )

                with benchmark_with_power(
                    device=device,
                    kernel_func=run_state_update,
                    num_warmups=num_warmups,
                    num_runs=num_runs,
                    repeat_n=1,
                    allow_graph_fail=True,
                ) as results:
                    log_perf(
                        item_list=[{**common_log_data, "latency": results["latency_ms"]}],
                        framework="TRTLLM",
                        version=tensorrt_llm.__version__,
                        device_name=torch.cuda.get_device_name(device),
                        op_name="mamba2",
                        kernel_source="selective_state_update",
                        perf_filename=perf_filename,
                        power_stats=results["power_stats"],
                    )

            # Cleanup
            if aic_cached_inputs:
                del ssm_state, conv_state, x, dt, B, C, xbc_input
            else:
                del ssm_state, conv_state, input_pool
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  Error at batch_size={batch_size}: {e}")
            continue


def _make_input_pool(
    shapes: dict[str, tuple[int, ...]],
    count: int,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, list[torch.Tensor]]:
    """Pre-generate a pool of random input tensors for randomized benchmarking."""
    return {
        name: [torch.randn(*shape, dtype=dtype, device=device) for _ in range(count)] for name, shape in shapes.items()
    }


def run_mamba2_torch(
    phase: str,
    d_model: int,
    d_state: int,
    d_conv: int,
    nheads: int,
    head_dim: int,
    n_groups: int,
    chunk_size: int,
    batch_size_list: list[int],
    seq_len_list: list[int] | None,
    model_name: str,
    perf_filename: str,
    device: str = "cuda:0",
):
    """
    Main entry point for Mamba2 benchmarking.

    Routes to appropriate benchmark function based on phase.

    Args:
        phase: "context" or "generation"
        d_model: Hidden size
        d_state: SSM state dimension
        d_conv: Conv1d kernel size
        nheads: Number of Mamba heads
        head_dim: Dimension per head
        n_groups: Number of groups for B, C matrices
        chunk_size: Chunk size for SSM scan
        batch_size_list: List of batch sizes to sweep
        seq_len_list: List of sequence lengths (context only, None for generation)
        model_name: Model configuration name
        perf_filename: Output performance file
        device: CUDA device string
    """
    # Suppress both Python-level print() and C-level printf() during TRT-LLM import
    import contextlib

    with (
        open(os.devnull, "w") as _devnull_file,
        contextlib.redirect_stdout(_devnull_file),
        contextlib.redirect_stderr(_devnull_file),
    ):
        import tensorrt_llm
        from tensorrt_llm._torch.modules.mamba.causal_conv1d import causal_conv1d_fn, causal_conv1d_update
        from tensorrt_llm._torch.modules.mamba.selective_state_update import selective_state_update
        from tensorrt_llm._torch.modules.mamba.ssd_combined import mamba_chunk_scan_combined

    # Expose imported symbols as module globals for benchmark functions
    globals().update(
        {
            "tensorrt_llm": tensorrt_llm,
            "selective_state_update": selective_state_update,
            "mamba_chunk_scan_combined": mamba_chunk_scan_combined,
            "causal_conv1d_fn": causal_conv1d_fn,
            "causal_conv1d_update": causal_conv1d_update,
        }
    )

    if phase == "context":
        run_mamba2_context_benchmark(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            nheads=nheads,
            head_dim=head_dim,
            n_groups=n_groups,
            chunk_size=chunk_size,
            batch_size_list=batch_size_list,
            seq_len_list=seq_len_list,
            model_name=model_name,
            perf_filename=perf_filename,
            device=device,
        )
    elif phase == "generation":
        run_mamba2_generation_benchmark(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            nheads=nheads,
            head_dim=head_dim,
            n_groups=n_groups,
            chunk_size=chunk_size,
            batch_size_list=batch_size_list,
            model_name=model_name,
            perf_filename=perf_filename,
            device=device,
        )
    else:
        raise ValueError(f"Unknown phase: {phase}")

    # Exit worker process to ensure clean GPU state
    import sys

    sys.exit(EXIT_CODE_RESTART)


if __name__ == "__main__":
    import tensorrt_llm

    print(f"Mamba2 Collector - TensorRT-LLM {tensorrt_llm.__version__}")
    print(f"SM Version: {get_sm_version()}")
    print(f"Device: {torch.cuda.get_device_name()}")
    print()

    test_cases = get_mamba2_test_cases()
    print(f"Total test cases: {len(test_cases)}")

    for i, test_case in enumerate(test_cases):
        (
            phase,
            d_model,
            d_state,
            d_conv,
            nheads,
            head_dim,
            n_groups,
            chunk_size,
            batch_size_list,
            seq_len_list,
            model_name,
            perf_filename,
        ) = test_case

        print(f"\n[{i + 1}/{len(test_cases)}] {model_name} - {phase}")
        print(f"  d_model={d_model}, nheads={nheads}, head_dim={head_dim}, d_state={d_state}, n_groups={n_groups}")

        if phase == "context":
            print(f"  batch_sizes={batch_size_list}")
            print(f"  seq_lens={seq_len_list}")
        else:
            print(f"  batch_sizes={batch_size_list}")

        run_mamba2_torch(
            phase=phase,
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            nheads=nheads,
            head_dim=head_dim,
            n_groups=n_groups,
            chunk_size=chunk_size,
            batch_size_list=batch_size_list,
            seq_len_list=seq_len_list,
            model_name=model_name,
            perf_filename=perf_filename,
        )
