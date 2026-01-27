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
    ^^^^^^^^^^^^^^^^         ^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^
    Use GEMM model          Benchmarked here     Use GEMM model

Usage:
    python collect_mamba2.py

Output:
    mamba2_perf.txt - Performance data for Mamba2 Conv1D+SSM operations
"""

import gc
import os

import tensorrt_llm
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

# Import Mamba2 kernels from TensorRT-LLM
from tensorrt_llm._torch.modules.mamba.causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from tensorrt_llm._torch.modules.mamba.selective_state_update import selective_state_update
from tensorrt_llm._torch.modules.mamba.ssd_combined import mamba_chunk_scan_combined

aic_debug = int(os.getenv("aic_mamba2_debug", "0"))  # noqa: SIM112


def get_mamba2_test_cases():
    """
    Generate test cases for Mamba2 SSM benchmarking.

    Returns a list of test case configurations for both context (prefill)
    and generation (decode) phases.
    """
    test_cases = []

    # Get common test cases from centralized definition
    for common_case in get_common_mamba2_test_cases():
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
                common_case.num_tokens_list if common_case.phase == "context" else common_case.batch_size_list,
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
    num_tokens_list: list[int],
    model_name: str,
    perf_filename: str,
    device: str = "cuda:0",
):
    """
    Benchmark Mamba2 SSM for context (prefill) phase.

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

    for num_tokens in num_tokens_list:
        if aic_debug == 1:
            print(f"  Benchmarking num_tokens={num_tokens}")

        try:
            # Input tensor for conv1d: [num_tokens, conv_dim]
            # This is the output of in_proj GEMM
            xbc_input = torch.randn(num_tokens, conv_dim, dtype=dtype, device=device)

            # After conv1d, we split xbc into x, B, C and reshape for SSM
            # For benchmarking, we prepare the SSM inputs directly
            # x: [batch=1, seqlen, nheads, head_dim]
            # B, C: [batch=1, seqlen, n_groups, d_state]
            x = torch.randn(1, num_tokens, nheads, head_dim, dtype=dtype, device=device)
            dt = torch.randn(1, num_tokens, nheads, dtype=dtype, device=device)
            B = torch.randn(1, num_tokens, n_groups, d_state, dtype=dtype, device=device)  # noqa: N806
            C = torch.randn(1, num_tokens, n_groups, d_state, dtype=dtype, device=device)  # noqa: N806

            # Pre-allocate outputs
            conv_out = torch.empty_like(xbc_input)
            ssm_out = torch.empty_like(x)

            # Warmup
            torch.cuda.synchronize()
            causal_conv1d_fn(xbc_input, conv_weight, conv_bias, activation="silu")
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
                out=ssm_out,
            )
            torch.cuda.synchronize()

            # Define benchmark function - Conv1D + SSM scan combined
            # Capture loop variables as default args to satisfy ruff F821
            def run_conv1d_and_ssm_scan(_xbc=xbc_input, _x=x, _dt=dt, _b=B, _c=C, _out=ssm_out):
                # Step 1: Causal Conv1D
                causal_conv1d_fn(_xbc, conv_weight, conv_bias, activation="silu")
                # Step 2: SSM Scan
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
                    out=_out,
                )

            # Benchmark with power measurement
            with benchmark_with_power(
                device=device,
                kernel_func=run_conv1d_and_ssm_scan,
                num_warmups=3,
                num_runs=10,
                repeat_n=1,
                allow_graph_fail=True,
            ) as results:
                latency = results["latency_ms"]
                power_stats = results["power_stats"]

            # Log performance
            log_perf(
                item_list=[
                    {
                        "phase": "context",
                        "num_tokens": num_tokens,
                        "d_model": d_model,
                        "d_state": d_state,
                        "d_conv": d_conv,
                        "nheads": nheads,
                        "head_dim": head_dim,
                        "n_groups": n_groups,
                        "chunk_size": chunk_size,
                        "model_name": model_name,
                        "latency": latency,
                    }
                ],
                framework="TRTLLM",
                version=tensorrt_llm.__version__,
                device_name=torch.cuda.get_device_name(device),
                op_name="mamba2",
                kernel_source="conv1d_ssm_combined",
                perf_filename=perf_filename,
                power_stats=power_stats,
            )

            # Cleanup
            del x, dt, B, C, ssm_out, xbc_input, conv_out
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  Error at num_tokens={num_tokens}: {e}")
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
            # Conv1d state: [batch, conv_dim, d_conv]
            conv_state = torch.randn(batch_size, conv_dim, d_conv, dtype=dtype, device=device)

            # SSM state: [batch, nheads, head_dim, d_state]
            ssm_state = torch.randn(batch_size, nheads, head_dim, d_state, dtype=dtype, device=device)

            # Input for single token from in_proj: [batch, conv_dim]
            xbc_input = torch.randn(batch_size, conv_dim, dtype=dtype, device=device)

            # SSM inputs for single token: [batch, nheads, head_dim]
            x = torch.randn(batch_size, nheads, head_dim, dtype=dtype, device=device)
            dt = torch.randn(batch_size, nheads, head_dim, dtype=dtype, device=device)
            B = torch.randn(batch_size, n_groups, d_state, dtype=dtype, device=device)  # noqa: N806
            C = torch.randn(batch_size, n_groups, d_state, dtype=dtype, device=device)  # noqa: N806

            # Pre-allocate output
            y = torch.empty_like(x)

            # Warmup
            torch.cuda.synchronize()
            causal_conv1d_update(xbc_input, conv_state, conv_weight, conv_bias, activation="silu")
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
                out=y,
            )
            torch.cuda.synchronize()

            # Define benchmark function - Conv1D update + SSM state update
            # Capture loop variables as default args to satisfy ruff F821
            def run_conv1d_update_and_state_update(
                _xbc=xbc_input, _conv_state=conv_state, _ssm_state=ssm_state, _x=x, _dt=dt, _b=B, _c=C, _y=y
            ):
                # Step 1: Causal Conv1D update (processes single token, updates conv_state)
                causal_conv1d_update(_xbc, _conv_state, conv_weight, conv_bias, activation="silu")
                # Step 2: SSM state update
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
                    out=_y,
                )

            # Benchmark with power measurement
            with benchmark_with_power(
                device=device,
                kernel_func=run_conv1d_update_and_state_update,
                num_warmups=3,
                num_runs=10,
                repeat_n=1,
                allow_graph_fail=True,
            ) as results:
                latency = results["latency_ms"]
                power_stats = results["power_stats"]

            # Log performance
            log_perf(
                item_list=[
                    {
                        "phase": "generation",
                        "batch_size": batch_size,
                        "d_model": d_model,
                        "d_state": d_state,
                        "d_conv": d_conv,
                        "nheads": nheads,
                        "head_dim": head_dim,
                        "n_groups": n_groups,
                        "chunk_size": chunk_size,
                        "model_name": model_name,
                        "latency": latency,
                    }
                ],
                framework="TRTLLM",
                version=tensorrt_llm.__version__,
                device_name=torch.cuda.get_device_name(device),
                op_name="mamba2",
                kernel_source="conv1d_ssm_combined",
                perf_filename=perf_filename,
                power_stats=power_stats,
            )

            # Cleanup
            del ssm_state, conv_state, x, dt, B, C, y, xbc_input
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  Error at batch_size={batch_size}: {e}")
            continue


def run_mamba2_torch(
    phase: str,
    d_model: int,
    d_state: int,
    d_conv: int,
    nheads: int,
    head_dim: int,
    n_groups: int,
    chunk_size: int,
    tokens_or_batch_list: list[int],
    model_name: str,
    perf_filename: str,
    device: str = "cuda:0",
):
    """
    Main entry point for Mamba2 benchmarking.

    Routes to appropriate benchmark function based on phase.
    """
    if phase == "context":
        run_mamba2_context_benchmark(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            nheads=nheads,
            head_dim=head_dim,
            n_groups=n_groups,
            chunk_size=chunk_size,
            num_tokens_list=tokens_or_batch_list,
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
            batch_size_list=tokens_or_batch_list,
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
            tokens_or_batch_list,
            model_name,
            perf_filename,
        ) = test_case

        print(f"\n[{i + 1}/{len(test_cases)}] {model_name} - {phase}")
        print(f"  d_model={d_model}, nheads={nheads}, head_dim={head_dim}, d_state={d_state}, n_groups={n_groups}")

        run_mamba2_torch(
            phase=phase,
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            nheads=nheads,
            head_dim=head_dim,
            n_groups=n_groups,
            chunk_size=chunk_size,
            tokens_or_batch_list=tokens_or_batch_list,
            model_name=model_name,
            perf_filename=perf_filename,
        )
