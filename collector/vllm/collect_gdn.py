# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

__compat__ = "vllm>=0.17.0"

"""
GDN (Gated DeltaNet) Collector for AIConfigurator.

This collector benchmarks the core GDN operations used by Qwen3.5
linear_attention layers using vLLM's vendored FLA Triton kernels.

vLLM also has a FlashInfer CUTLASS prefill path on SM90+; this collector
benchmarks the Triton FLA path which is the fallback on all architectures.

Context (prefill) phase:
    - causal_conv1d_fn: Applies causal 1D convolution over the sequence (key channels)
    - chunk_gated_delta_rule: GDN scan over (Q, K, V, g, beta) using chunked algorithm

Generation (decode) phase:
    - causal_conv1d_update: Updates conv state for single token (key channels)
    - fused_recurrent_gated_delta_rule: GDN state update for single token

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
    from vllm.model_executor.layers.fla.ops import (
        chunk_gated_delta_rule,
        fused_recurrent_gated_delta_rule,
    )
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn, causal_conv1d_update

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
    2. chunk_gated_delta_rule — GDN scan (Q, K, V, g, beta) via vLLM's vendored FLA
    """
    device = torch.device(device)
    torch.cuda.set_device(device)
    torch.set_default_device(device)

    dtype = torch.bfloat16

    conv_channels = num_k_heads * head_k_dim

    if aic_debug:
        print(
            f"GDN Context: d_model={d_model}, conv_channels={conv_channels}, "
            f"num_k_heads={num_k_heads}, head_k_dim={head_k_dim}, "
            f"num_v_heads={num_v_heads}, head_v_dim={head_v_dim}"
        )

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

                conv_state = torch.randn(batch_size, conv_channels, d_conv - 1, dtype=dtype, device=device)

                if aic_cached_inputs:
                    k_input = torch.randn(conv_channels, batch_size * seq_len, dtype=dtype, device=device)
                    q = torch.randn(batch_size, seq_len, num_k_heads, head_k_dim, dtype=dtype, device=device)
                    k = torch.randn(batch_size, seq_len, num_k_heads, head_k_dim, dtype=dtype, device=device)
                    v = torch.randn(batch_size, seq_len, num_v_heads, head_v_dim, dtype=dtype, device=device)
                    g = torch.nn.functional.logsigmoid(
                        torch.randn(batch_size, seq_len, num_v_heads, dtype=dtype, device=device)
                    )
                    beta = torch.sigmoid(torch.randn(batch_size, seq_len, num_v_heads, dtype=dtype, device=device))
                    query_start_loc = torch.arange(
                        0, batch_size * seq_len + 1, seq_len, dtype=torch.int32, device=device
                    )
                    cache_indices = torch.arange(batch_size, dtype=torch.int32, device=device)
                    has_initial_state = torch.ones(batch_size, dtype=torch.bool, device=device)

                    # --- Benchmark causal_conv1d_fn ---
                    torch.cuda.synchronize()
                    causal_conv1d_fn(
                        k_input,
                        conv_weight,
                        conv_bias,
                        conv_state,
                        query_start_loc,
                        cache_indices=cache_indices,
                        has_initial_state=has_initial_state,
                        activation="silu",
                    )
                    torch.cuda.synchronize()

                    def run_conv1d(
                        _k=k_input, _cs=conv_state, _qsl=query_start_loc, _ci=cache_indices, _hi=has_initial_state
                    ):
                        causal_conv1d_fn(
                            _k,
                            conv_weight,
                            conv_bias,
                            _cs,
                            _qsl,
                            cache_indices=_ci,
                            has_initial_state=_hi,
                            activation="silu",
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
                    chunk_gated_delta_rule(q, k, v, g, beta)
                    torch.cuda.synchronize()

                    def run_gdn_scan(_q=q, _k=k, _v=v, _g=g, _beta=beta):
                        chunk_gated_delta_rule(_q, _k, _v, _g, _beta)

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
                    input_pool = _make_input_pool(
                        {
                            "k_input": (conv_channels, batch_size * seq_len),
                            "q": (batch_size, seq_len, num_k_heads, head_k_dim),
                            "k": (batch_size, seq_len, num_k_heads, head_k_dim),
                            "v": (batch_size, seq_len, num_v_heads, head_v_dim),
                            "g": (batch_size, seq_len, num_v_heads),
                            "beta": (batch_size, seq_len, num_v_heads),
                        },
                        total_iters,
                        dtype,
                        device,
                    )
                    for i in range(total_iters):
                        input_pool["g"][i] = torch.nn.functional.logsigmoid(input_pool["g"][i])
                        input_pool["beta"][i] = torch.sigmoid(input_pool["beta"][i])
                    query_start_loc = torch.arange(
                        0, batch_size * seq_len + 1, seq_len, dtype=torch.int32, device=device
                    )
                    cache_indices = torch.arange(batch_size, dtype=torch.int32, device=device)
                    has_initial_state = torch.ones(batch_size, dtype=torch.bool, device=device)

                    # --- Benchmark causal_conv1d_fn ---
                    torch.cuda.synchronize()
                    causal_conv1d_fn(
                        input_pool["k_input"][0],
                        conv_weight,
                        conv_bias,
                        conv_state,
                        query_start_loc,
                        cache_indices=cache_indices,
                        has_initial_state=has_initial_state,
                        activation="silu",
                    )
                    torch.cuda.synchronize()

                    conv1d_iter_idx = [0]

                    def run_conv1d(
                        _pool=input_pool,
                        _cs=conv_state,
                        _idx=conv1d_iter_idx,
                        _qsl=query_start_loc,
                        _ci=cache_indices,
                        _hi=has_initial_state,
                    ):
                        idx = _idx[0] % total_iters
                        _idx[0] += 1
                        causal_conv1d_fn(
                            _pool["k_input"][idx],
                            conv_weight,
                            conv_bias,
                            _cs,
                            _qsl,
                            cache_indices=_ci,
                            has_initial_state=_hi,
                            activation="silu",
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
                        input_pool["q"][0],
                        input_pool["k"][0],
                        input_pool["v"][0],
                        input_pool["g"][0],
                        input_pool["beta"][0],
                    )
                    torch.cuda.synchronize()

                    gdn_scan_iter_idx = [0]

                    def run_gdn_scan(_pool=input_pool, _idx=gdn_scan_iter_idx):
                        idx = _idx[0] % total_iters
                        _idx[0] += 1
                        chunk_gated_delta_rule(
                            _pool["q"][idx],
                            _pool["k"][idx],
                            _pool["v"][idx],
                            _pool["g"][idx],
                            _pool["beta"][idx],
                        )

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
                    del k_input, q, k, v, g, beta, conv_state
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
    2. fused_recurrent_gated_delta_rule — GDN state update for single token
    """
    device = torch.device(device)
    torch.cuda.set_device(device)
    torch.set_default_device(device)

    dtype = torch.bfloat16

    conv_channels = num_k_heads * head_k_dim

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

            conv_state = torch.randn(batch_size, conv_channels, d_conv - 1, dtype=dtype, device=device)
            # vLLM GDN state layout: [batch, num_v_heads, head_v_dim, head_k_dim]
            gdn_state = torch.randn(batch_size, num_v_heads, head_v_dim, head_k_dim, dtype=dtype, device=device)

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
                k_input = torch.randn(batch_size, conv_channels, dtype=dtype, device=device)
                q = torch.randn(batch_size, 1, num_k_heads, head_k_dim, dtype=dtype, device=device)
                k = torch.randn(batch_size, 1, num_k_heads, head_k_dim, dtype=dtype, device=device)
                v = torch.randn(batch_size, 1, num_v_heads, head_v_dim, dtype=dtype, device=device)
                g = torch.nn.functional.logsigmoid(torch.randn(batch_size, 1, num_v_heads, dtype=dtype, device=device))
                beta = torch.sigmoid(torch.randn(batch_size, 1, num_v_heads, dtype=dtype, device=device))
                conv_state_indices = torch.arange(batch_size, dtype=torch.int32, device=device)

                # --- Benchmark causal_conv1d_update ---
                torch.cuda.synchronize()
                causal_conv1d_update(
                    k_input,
                    conv_state,
                    conv_weight,
                    conv_bias,
                    activation="silu",
                    conv_state_indices=conv_state_indices,
                )
                torch.cuda.synchronize()

                def run_conv1d_update(_k=k_input, _cs=conv_state, _csi=conv_state_indices):
                    causal_conv1d_update(
                        _k,
                        _cs,
                        conv_weight,
                        conv_bias,
                        activation="silu",
                        conv_state_indices=_csi,
                    )

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

                # --- Benchmark fused_recurrent_gated_delta_rule ---
                torch.cuda.synchronize()
                fused_recurrent_gated_delta_rule(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    initial_state=gdn_state,
                    inplace_final_state=True,
                )
                torch.cuda.synchronize()

                def run_gdn_update(_q=q, _k=k, _v=v, _g=g, _beta=beta, _state=gdn_state):
                    fused_recurrent_gated_delta_rule(
                        _q,
                        _k,
                        _v,
                        _g,
                        _beta,
                        initial_state=_state,
                        inplace_final_state=True,
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
                        kernel_source="fused_recurrent_gated_delta_rule",
                        perf_filename=perf_filename,
                        power_stats=results["power_stats"],
                    )

            else:
                input_pool = _make_input_pool(
                    {
                        "k_input": (batch_size, conv_channels),
                        "q": (batch_size, 1, num_k_heads, head_k_dim),
                        "k": (batch_size, 1, num_k_heads, head_k_dim),
                        "v": (batch_size, 1, num_v_heads, head_v_dim),
                        "g": (batch_size, 1, num_v_heads),
                        "beta": (batch_size, 1, num_v_heads),
                    },
                    total_iters,
                    dtype,
                    device,
                )
                for i in range(total_iters):
                    input_pool["g"][i] = torch.nn.functional.logsigmoid(input_pool["g"][i])
                    input_pool["beta"][i] = torch.sigmoid(input_pool["beta"][i])

                conv_state_indices = torch.arange(batch_size, dtype=torch.int32, device=device)

                # --- Benchmark causal_conv1d_update ---
                torch.cuda.synchronize()
                causal_conv1d_update(
                    input_pool["k_input"][0],
                    conv_state,
                    conv_weight,
                    conv_bias,
                    activation="silu",
                    conv_state_indices=conv_state_indices,
                )
                torch.cuda.synchronize()

                conv1d_iter_idx = [0]

                def run_conv1d_update(_pool=input_pool, _cs=conv_state, _idx=conv1d_iter_idx, _csi=conv_state_indices):
                    idx = _idx[0] % total_iters
                    _idx[0] += 1
                    causal_conv1d_update(
                        _pool["k_input"][idx],
                        _cs,
                        conv_weight,
                        conv_bias,
                        activation="silu",
                        conv_state_indices=_csi,
                    )

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

                # --- Benchmark fused_recurrent_gated_delta_rule ---
                torch.cuda.synchronize()
                fused_recurrent_gated_delta_rule(
                    input_pool["q"][0],
                    input_pool["k"][0],
                    input_pool["v"][0],
                    input_pool["g"][0],
                    input_pool["beta"][0],
                    initial_state=gdn_state,
                    inplace_final_state=True,
                )
                torch.cuda.synchronize()

                gdn_iter_idx = [0]

                def run_gdn_update(_pool=input_pool, _state=gdn_state, _idx=gdn_iter_idx):
                    idx = _idx[0] % total_iters
                    _idx[0] += 1
                    fused_recurrent_gated_delta_rule(
                        _pool["q"][idx],
                        _pool["k"][idx],
                        _pool["v"][idx],
                        _pool["g"][idx],
                        _pool["beta"][idx],
                        initial_state=_state,
                        inplace_final_state=True,
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
                        kernel_source="fused_recurrent_gated_delta_rule",
                        perf_filename=perf_filename,
                        power_stats=results["power_stats"],
                    )

            # Cleanup
            if aic_cached_inputs:
                del k_input, q, k, v, g, beta, conv_state, gdn_state
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
    *,
    perf_filename: str,
    device: str = "cuda:0",
):
    """
    Main entry point for GDN benchmarking.

    Routes to appropriate benchmark function based on phase.
    Imports GDN kernels from vLLM's vendored FLA copy at runtime.
    """
    import contextlib

    with (
        open(os.devnull, "w") as _devnull_file,
        contextlib.redirect_stdout(_devnull_file),
        contextlib.redirect_stderr(_devnull_file),
    ):
        import vllm
        from vllm.model_executor.layers.fla.ops import (
            chunk_gated_delta_rule,
            fused_recurrent_gated_delta_rule,
        )
        from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn, causal_conv1d_update

    from vllm.version import __version__ as vllm_version_str

    globals().update(
        {
            "vllm": vllm,
            "vllm_version": vllm_version_str,
            "causal_conv1d_fn": causal_conv1d_fn,
            "causal_conv1d_update": causal_conv1d_update,
            "chunk_gated_delta_rule": chunk_gated_delta_rule,
            "fused_recurrent_gated_delta_rule": fused_recurrent_gated_delta_rule,
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

    # Return EXIT_CODE_RESTART to signal that a process restart would be
    # desirable for GPU memory cleanup.  collect.py's orchestrator previously
    # relied on this function calling sys.exit(EXIT_CODE_RESTART) directly,
    # which killed the worker process after each task so the OS reclaimed GPU
    # memory before the next task started.  That also prevented the __main__
    # for-loop from completing more than one case when run standalone.
    #
    # The sys.exit has been moved outside the loop in __main__ so that all
    # test cases run in sequence.  When invoked via collect.py the worker
    # process no longer restarts between GDN tasks; if GPU OOM is observed in
    # that path, restoring per-task process recycling here would fix it.
    return EXIT_CODE_RESTART


if __name__ == "__main__":
    import sys

    from vllm.version import __version__ as vllm_version

    from collector.registry_types import PerfFile

    print(f"GDN Collector - vLLM {vllm_version}")
    print(f"SM Version: {get_sm_version()}")
    print(f"Device: {torch.cuda.get_device_name()}")
    print()

    test_cases = get_gdn_test_cases()
    print(f"Total test cases: {len(test_cases)}")

    last_exit_code = 0
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

        last_exit_code = run_gdn_torch(
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
            perf_filename=PerfFile.GDN,
        )

    sys.exit(last_exit_code)
