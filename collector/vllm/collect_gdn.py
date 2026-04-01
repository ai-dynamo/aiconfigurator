# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

__compat__ = "vllm>=0.17.0"

"""
GDN (Gated DeltaNet) Collector for AIConfigurator.

This collector benchmarks the core GDN operations used by Qwen3.5
linear_attention layers (chunk_gated_delta_rule + causal_conv1d):

Context (prefill) phase:
    - causal_conv1d_fn: Applies causal 1D convolution over the sequence (key channels)
    - chunk_gated_delta_rule: GDN scan over (Q, K, V, beta) using chunked algorithm

Generation (decode) phase:
    - causal_conv1d_update: Updates conv state for single token (key channels)
    - fused_sigmoid_gating_delta_rule_update: GDN state update for single token

The in_proj and out_proj GEMMs are standard linear layers modeled by the existing
GEMM infrastructure. This collector focuses on the unique GDN operations.

GDN Layer Flow:
    in_proj (GEMM) → Conv1D (keys) → GDN Scan/Update → out_proj (GEMM)
    ^^^^^^^^^^^^^    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^    ^^^^^^^^^^^^^^^^
    Use GEMM model          Benchmarked here            Use GEMM model

Usage:
    python collect_gdn.py

Output:
    gdn_perf.txt - Performance data for GDN Conv1D + scan operations
"""

import gc
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Keep these imports to satisfy static analysis.
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from fla.ops.gated_delta_rule.recurrent import fused_sigmoid_gating_delta_rule_update

import torch

try:
    from collector.common_test_cases import get_common_gdn_test_cases
    from collector.helper import (
        EXIT_CODE_RESTART,
        benchmark_with_power,
        get_sm_version,
        log_perf,
    )
except ModuleNotFoundError:
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from common_test_cases import get_common_gdn_test_cases

    from helper import (
        EXIT_CODE_RESTART,
        benchmark_with_power,
        get_sm_version,
        log_perf,
    )

aic_debug = int(os.getenv("aic_gdn_debug", "0"))  # noqa: SIM112
# Use cached inputs (same data each iteration) instead of randomized inputs
aic_cached_inputs = int(os.getenv("AIC_GDN_CACHED_INPUTS", "0"))


def get_gdn_test_cases():
    """
    Generate test cases for GDN kernel benchmarking.

    Returns a list of test case configurations for both context (prefill)
    and generation (decode) phases.
    """
    test_cases = []

    for common_case in get_common_gdn_test_cases():
        if common_case.phase == "context":
            test_cases.append(
                [
                    common_case.phase,
                    common_case.d_model,
                    common_case.d_conv,
                    common_case.num_k_heads,
                    common_case.head_k_dim,
                    common_case.num_v_heads,
                    common_case.head_v_dim,
                    common_case.batch_size_list,
                    common_case.seq_len_list,
                    common_case.model_name,
                    "gdn_perf.txt",
                ]
            )
        else:
            test_cases.append(
                [
                    common_case.phase,
                    common_case.d_model,
                    common_case.d_conv,
                    common_case.num_k_heads,
                    common_case.head_k_dim,
                    common_case.num_v_heads,
                    common_case.head_v_dim,
                    common_case.batch_size_list,
                    None,  # seq_len_list not used for generation
                    common_case.model_name,
                    "gdn_perf.txt",
                ]
            )

    return test_cases


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


def run_gdn_context_benchmark(
    d_model: int,
    d_conv: int,
    num_k_heads: int,
    head_k_dim: int,
    num_v_heads: int,
    head_v_dim: int,
    batch_size_list: list[int],
    seq_len_list: list[int],
    model_name: str,
    perf_filename: str,
    device: str = "cuda:0",
):
    """
    Benchmark GDN operations for context (prefill) phase.

    Benchmarks:
    1. causal_conv1d_fn  — Conv1D over key channels
    2. chunk_gated_delta_rule — GDN scan (Q, K, V, beta)
    """
    device = torch.device(device)
    torch.cuda.set_device(device)
    torch.set_default_device(device)

    dtype = torch.bfloat16

    # Conv1d is applied to key channels only
    conv_channels = num_k_heads * head_k_dim

    if aic_debug:
        print(
            f"GDN Context: d_model={d_model}, conv_channels={conv_channels}, "
            f"num_k_heads={num_k_heads}, head_k_dim={head_k_dim}, "
            f"num_v_heads={num_v_heads}, head_v_dim={head_v_dim}"
        )

    # Conv1d weights: (channels, kernel_size)
    conv_weight = torch.randn(conv_channels, d_conv, dtype=dtype, device=device)
    conv_bias = torch.randn(conv_channels, dtype=dtype, device=device)

    for batch_size in batch_size_list:
        for seq_len in seq_len_list:
            if aic_debug:
                print(f"  Benchmarking batch_size={batch_size}, seq_len={seq_len}")

            try:
                num_warmups = 3
                num_runs = 10
                total_iters = num_warmups + num_runs

                common_log_data = {
                    "phase": "context",
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "num_tokens": batch_size * seq_len,
                    "d_model": d_model,
                    "d_conv": d_conv,
                    "num_k_heads": num_k_heads,
                    "head_k_dim": head_k_dim,
                    "num_v_heads": num_v_heads,
                    "head_v_dim": head_v_dim,
                    "model_name": model_name,
                }

                # Conv state: (batch, channels, d_conv - 1)
                conv_state = torch.randn(batch_size, conv_channels, d_conv - 1, dtype=dtype, device=device)

                if aic_cached_inputs:
                    # Cached mode: same inputs every iteration
                    k_input = torch.randn(batch_size, conv_channels, seq_len, dtype=dtype, device=device)
                    q = torch.randn(batch_size, seq_len, num_k_heads, head_k_dim, dtype=dtype, device=device)
                    k = torch.randn(batch_size, seq_len, num_k_heads, head_k_dim, dtype=dtype, device=device)
                    v = torch.randn(batch_size, seq_len, num_v_heads, head_v_dim, dtype=dtype, device=device)
                    beta = torch.sigmoid(torch.randn(batch_size, seq_len, num_k_heads, dtype=dtype, device=device))

                    # --- Benchmark causal_conv1d_fn ---
                    torch.cuda.synchronize()
                    causal_conv1d_fn(k_input, conv_weight, conv_bias, activation="silu", conv_states=conv_state)
                    torch.cuda.synchronize()

                    def run_conv1d(_k=k_input, _cs=conv_state):
                        causal_conv1d_fn(_k, conv_weight, conv_bias, activation="silu", conv_states=_cs)

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
                            framework="vLLM",
                            version=vllm_version,
                            device_name=torch.cuda.get_device_name(device),
                            op_name="gdn",
                            kernel_source="causal_conv1d_fn",
                            perf_filename=perf_filename,
                            power_stats=results["power_stats"],
                        )

                    # --- Benchmark chunk_gated_delta_rule ---
                    torch.cuda.synchronize()
                    chunk_gated_delta_rule(q, k, v, beta)
                    torch.cuda.synchronize()

                    def run_gdn_scan(_q=q, _k=k, _v=v, _beta=beta):
                        chunk_gated_delta_rule(_q, _k, _v, _beta)

                    with benchmark_with_power(
                        device=device,
                        kernel_func=run_gdn_scan,
                        num_warmups=num_warmups,
                        num_runs=num_runs,
                        repeat_n=1,
                        allow_graph_fail=True,
                    ) as results:
                        log_perf(
                            item_list=[{**common_log_data, "latency": results["latency_ms"]}],
                            framework="vLLM",
                            version=vllm_version,
                            device_name=torch.cuda.get_device_name(device),
                            op_name="gdn",
                            kernel_source="chunk_gated_delta_rule",
                            perf_filename=perf_filename,
                            power_stats=results["power_stats"],
                        )

                else:
                    # Randomized mode: pre-generate pool of inputs
                    input_pool = _make_input_pool(
                        {
                            "k_input": (batch_size, conv_channels, seq_len),
                            "q": (batch_size, seq_len, num_k_heads, head_k_dim),
                            "k": (batch_size, seq_len, num_k_heads, head_k_dim),
                            "v": (batch_size, seq_len, num_v_heads, head_v_dim),
                            "beta": (batch_size, seq_len, num_k_heads),
                        },
                        total_iters,
                        dtype,
                        device,
                    )
                    # Clamp beta to (0, 1) via sigmoid
                    for i in range(total_iters):
                        input_pool["beta"][i] = torch.sigmoid(input_pool["beta"][i])

                    # --- Benchmark causal_conv1d_fn ---
                    torch.cuda.synchronize()
                    causal_conv1d_fn(
                        input_pool["k_input"][0], conv_weight, conv_bias, activation="silu", conv_states=conv_state
                    )
                    torch.cuda.synchronize()

                    conv1d_iter_idx = [0]

                    def run_conv1d(_pool=input_pool, _cs=conv_state, _idx=conv1d_iter_idx):
                        idx = _idx[0] % total_iters
                        _idx[0] += 1
                        causal_conv1d_fn(
                            _pool["k_input"][idx], conv_weight, conv_bias, activation="silu", conv_states=_cs
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
                            framework="vLLM",
                            version=vllm_version,
                            device_name=torch.cuda.get_device_name(device),
                            op_name="gdn",
                            kernel_source="causal_conv1d_fn",
                            perf_filename=perf_filename,
                            power_stats=results["power_stats"],
                        )

                    # --- Benchmark chunk_gated_delta_rule ---
                    torch.cuda.synchronize()
                    chunk_gated_delta_rule(
                        input_pool["q"][0], input_pool["k"][0], input_pool["v"][0], input_pool["beta"][0]
                    )
                    torch.cuda.synchronize()

                    gdn_scan_iter_idx = [0]

                    def run_gdn_scan(_pool=input_pool, _idx=gdn_scan_iter_idx):
                        idx = _idx[0] % total_iters
                        _idx[0] += 1
                        chunk_gated_delta_rule(_pool["q"][idx], _pool["k"][idx], _pool["v"][idx], _pool["beta"][idx])

                    with benchmark_with_power(
                        device=device,
                        kernel_func=run_gdn_scan,
                        num_warmups=num_warmups,
                        num_runs=num_runs,
                        repeat_n=1,
                        allow_graph_fail=True,
                    ) as results:
                        log_perf(
                            item_list=[{**common_log_data, "latency": results["latency_ms"]}],
                            framework="vLLM",
                            version=vllm_version,
                            device_name=torch.cuda.get_device_name(device),
                            op_name="gdn",
                            kernel_source="chunk_gated_delta_rule",
                            perf_filename=perf_filename,
                            power_stats=results["power_stats"],
                        )

                # Cleanup
                if aic_cached_inputs:
                    del k_input, q, k, v, beta, conv_state
                else:
                    del input_pool, conv_state
                gc.collect()
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"  Error at batch_size={batch_size}, seq_len={seq_len}: {e}")
                continue


def run_gdn_generation_benchmark(
    d_model: int,
    d_conv: int,
    num_k_heads: int,
    head_k_dim: int,
    num_v_heads: int,
    head_v_dim: int,
    batch_size_list: list[int],
    model_name: str,
    perf_filename: str,
    device: str = "cuda:0",
):
    """
    Benchmark GDN operations for generation (decode) phase.

    Benchmarks:
    1. causal_conv1d_update — Conv1D state update for single token (key channels)
    2. fused_sigmoid_gating_delta_rule_update — GDN state update for single token
    """
    device = torch.device(device)
    torch.cuda.set_device(device)
    torch.set_default_device(device)

    dtype = torch.bfloat16

    # Conv1d is applied to key channels only
    conv_channels = num_k_heads * head_k_dim
    # GDN state: outer product of k and v heads per key head
    # state_size: (batch, num_k_heads, head_k_dim, head_v_dim)

    if aic_debug:
        print(
            f"GDN Generation: d_model={d_model}, conv_channels={conv_channels}, "
            f"num_k_heads={num_k_heads}, head_k_dim={head_k_dim}, "
            f"num_v_heads={num_v_heads}, head_v_dim={head_v_dim}"
        )

    # Conv1d weights: (channels, kernel_size)
    conv_weight = torch.randn(conv_channels, d_conv, dtype=dtype, device=device)
    conv_bias = torch.randn(conv_channels, dtype=dtype, device=device)

    for batch_size in batch_size_list:
        if aic_debug:
            print(f"  Benchmarking batch_size={batch_size}")

        try:
            num_warmups = 3
            num_runs = 10
            total_iters = num_warmups + num_runs

            # Conv state: (batch, channels, d_conv - 1)
            conv_state = torch.randn(batch_size, conv_channels, d_conv - 1, dtype=dtype, device=device)
            # GDN state: (batch, num_k_heads, head_k_dim, head_v_dim)
            gdn_state = torch.randn(batch_size, num_k_heads, head_k_dim, head_v_dim, dtype=torch.float32, device=device)

            common_log_data = {
                "phase": "generation",
                "batch_size": batch_size,
                "seq_len": 1,
                "num_tokens": batch_size,
                "d_model": d_model,
                "d_conv": d_conv,
                "num_k_heads": num_k_heads,
                "head_k_dim": head_k_dim,
                "num_v_heads": num_v_heads,
                "head_v_dim": head_v_dim,
                "model_name": model_name,
            }

            if aic_cached_inputs:
                # Cached mode: same inputs every iteration
                k_input = torch.randn(batch_size, conv_channels, dtype=dtype, device=device)
                q = torch.randn(batch_size, num_k_heads, head_k_dim, dtype=dtype, device=device)
                k = torch.randn(batch_size, num_k_heads, head_k_dim, dtype=dtype, device=device)
                v = torch.randn(batch_size, num_v_heads, head_v_dim, dtype=dtype, device=device)
                beta = torch.sigmoid(torch.randn(batch_size, num_k_heads, dtype=dtype, device=device))

                # --- Benchmark causal_conv1d_update ---
                torch.cuda.synchronize()
                causal_conv1d_update(k_input, conv_state, conv_weight, conv_bias, activation="silu")
                torch.cuda.synchronize()

                def run_conv1d_update(_k=k_input, _cs=conv_state):
                    causal_conv1d_update(_k, _cs, conv_weight, conv_bias, activation="silu")

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
                        framework="vLLM",
                        version=vllm_version,
                        device_name=torch.cuda.get_device_name(device),
                        op_name="gdn",
                        kernel_source="causal_conv1d_update",
                        perf_filename=perf_filename,
                        power_stats=results["power_stats"],
                    )

                # --- Benchmark fused_sigmoid_gating_delta_rule_update ---
                torch.cuda.synchronize()
                fused_sigmoid_gating_delta_rule_update(q, k, v, beta, gdn_state)
                torch.cuda.synchronize()

                def run_gdn_update(_q=q, _k=k, _v=v, _beta=beta, _state=gdn_state):
                    fused_sigmoid_gating_delta_rule_update(_q, _k, _v, _beta, _state)

                with benchmark_with_power(
                    device=device,
                    kernel_func=run_gdn_update,
                    num_warmups=num_warmups,
                    num_runs=num_runs,
                    repeat_n=1,
                    allow_graph_fail=True,
                ) as results:
                    log_perf(
                        item_list=[{**common_log_data, "latency": results["latency_ms"]}],
                        framework="vLLM",
                        version=vllm_version,
                        device_name=torch.cuda.get_device_name(device),
                        op_name="gdn",
                        kernel_source="fused_sigmoid_gating_delta_rule_update",
                        perf_filename=perf_filename,
                        power_stats=results["power_stats"],
                    )

            else:
                # Randomized mode: pre-generate pool of inputs
                input_pool = _make_input_pool(
                    {
                        "k_input": (batch_size, conv_channels),
                        "q": (batch_size, num_k_heads, head_k_dim),
                        "k": (batch_size, num_k_heads, head_k_dim),
                        "v": (batch_size, num_v_heads, head_v_dim),
                        "beta": (batch_size, num_k_heads),
                    },
                    total_iters,
                    dtype,
                    device,
                )
                # Clamp beta to (0, 1) via sigmoid
                for i in range(total_iters):
                    input_pool["beta"][i] = torch.sigmoid(input_pool["beta"][i])

                # --- Benchmark causal_conv1d_update ---
                torch.cuda.synchronize()
                causal_conv1d_update(input_pool["k_input"][0], conv_state, conv_weight, conv_bias, activation="silu")
                torch.cuda.synchronize()

                conv1d_iter_idx = [0]

                def run_conv1d_update(_pool=input_pool, _cs=conv_state, _idx=conv1d_iter_idx):
                    idx = _idx[0] % total_iters
                    _idx[0] += 1
                    causal_conv1d_update(_pool["k_input"][idx], _cs, conv_weight, conv_bias, activation="silu")

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
                        framework="vLLM",
                        version=vllm_version,
                        device_name=torch.cuda.get_device_name(device),
                        op_name="gdn",
                        kernel_source="causal_conv1d_update",
                        perf_filename=perf_filename,
                        power_stats=results["power_stats"],
                    )

                # --- Benchmark fused_sigmoid_gating_delta_rule_update ---
                torch.cuda.synchronize()
                fused_sigmoid_gating_delta_rule_update(
                    input_pool["q"][0], input_pool["k"][0], input_pool["v"][0], input_pool["beta"][0], gdn_state
                )
                torch.cuda.synchronize()

                gdn_iter_idx = [0]

                def run_gdn_update(_pool=input_pool, _state=gdn_state, _idx=gdn_iter_idx):
                    idx = _idx[0] % total_iters
                    _idx[0] += 1
                    fused_sigmoid_gating_delta_rule_update(
                        _pool["q"][idx], _pool["k"][idx], _pool["v"][idx], _pool["beta"][idx], _state
                    )

                with benchmark_with_power(
                    device=device,
                    kernel_func=run_gdn_update,
                    num_warmups=num_warmups,
                    num_runs=num_runs,
                    repeat_n=1,
                    allow_graph_fail=True,
                ) as results:
                    log_perf(
                        item_list=[{**common_log_data, "latency": results["latency_ms"]}],
                        framework="vLLM",
                        version=vllm_version,
                        device_name=torch.cuda.get_device_name(device),
                        op_name="gdn",
                        kernel_source="fused_sigmoid_gating_delta_rule_update",
                        perf_filename=perf_filename,
                        power_stats=results["power_stats"],
                    )

            # Cleanup
            if aic_cached_inputs:
                del k_input, q, k, v, beta, conv_state, gdn_state
            else:
                del input_pool, conv_state, gdn_state
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  Error at batch_size={batch_size}: {e}")
            continue


def run_gdn_torch(
    phase: str,
    d_model: int,
    d_conv: int,
    num_k_heads: int,
    head_k_dim: int,
    num_v_heads: int,
    head_v_dim: int,
    batch_size_list: list[int],
    seq_len_list: list[int] | None,
    model_name: str,
    perf_filename: str,
    device: str = "cuda:0",
):
    """
    Main entry point for GDN benchmarking.

    Routes to appropriate benchmark function based on phase.
    Imports GDN kernels from FLA (via vLLM's dependency) at runtime.
    """
    import contextlib

    with (
        open(os.devnull, "w") as _devnull_file,
        contextlib.redirect_stdout(_devnull_file),
        contextlib.redirect_stderr(_devnull_file),
    ):
        import vllm
        from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        from fla.ops.gated_delta_rule.recurrent import fused_sigmoid_gating_delta_rule_update

    from vllm.version import __version__ as vllm_version_str

    globals().update(
        {
            "vllm": vllm,
            "vllm_version": vllm_version_str,
            "causal_conv1d_fn": causal_conv1d_fn,
            "causal_conv1d_update": causal_conv1d_update,
            "chunk_gated_delta_rule": chunk_gated_delta_rule,
            "fused_sigmoid_gating_delta_rule_update": fused_sigmoid_gating_delta_rule_update,
        }
    )

    if phase == "context":
        run_gdn_context_benchmark(
            d_model=d_model,
            d_conv=d_conv,
            num_k_heads=num_k_heads,
            head_k_dim=head_k_dim,
            num_v_heads=num_v_heads,
            head_v_dim=head_v_dim,
            batch_size_list=batch_size_list,
            seq_len_list=seq_len_list,
            model_name=model_name,
            perf_filename=perf_filename,
            device=device,
        )
    elif phase == "generation":
        run_gdn_generation_benchmark(
            d_model=d_model,
            d_conv=d_conv,
            num_k_heads=num_k_heads,
            head_k_dim=head_k_dim,
            num_v_heads=num_v_heads,
            head_v_dim=head_v_dim,
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
    from vllm.version import __version__ as vllm_version

    print(f"GDN Collector - vLLM {vllm_version}")
    print(f"SM Version: {get_sm_version()}")
    print(f"Device: {torch.cuda.get_device_name()}")
    print()

    test_cases = get_gdn_test_cases()
    print(f"Total test cases: {len(test_cases)}")

    for i, test_case in enumerate(test_cases):
        (
            phase,
            d_model,
            d_conv,
            num_k_heads,
            head_k_dim,
            num_v_heads,
            head_v_dim,
            batch_size_list,
            seq_len_list,
            model_name,
            perf_filename,
        ) = test_case

        print(f"\n[{i + 1}/{len(test_cases)}] {model_name} - {phase}")
        print(
            f"  d_model={d_model}, num_k_heads={num_k_heads}, head_k_dim={head_k_dim}, "
            f"num_v_heads={num_v_heads}, head_v_dim={head_v_dim}, d_conv={d_conv}"
        )

        if phase == "context":
            print(f"  batch_sizes={batch_size_list}")
            print(f"  seq_lens={seq_len_list}")
        else:
            print(f"  batch_sizes={batch_size_list}")

        run_gdn_torch(
            phase=phase,
            d_model=d_model,
            d_conv=d_conv,
            num_k_heads=num_k_heads,
            head_k_dim=head_k_dim,
            num_v_heads=num_v_heads,
            head_v_dim=head_v_dim,
            batch_size_list=batch_size_list,
            seq_len_list=seq_len_list,
            model_name=model_name,
            perf_filename=perf_filename,
        )
