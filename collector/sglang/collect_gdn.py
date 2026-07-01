# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
GDN (Gated DeltaNet) Collector for AIConfigurator — SGLang backend.

SGLang ships its own vendored Triton FLA implementation for GDN ops in
Qwen3.5 linear_attention layers. Profiling the SGLang-bundled kernels
ensures the collected data reflects SGLang's actual runtime performance.

Context (prefill) phase:
    - causal_conv1d_fn: Causal 1D convolution over packed Q+K+V channels
    - chunk_gated_delta_rule: GDN chunked scan (Q, K, V, g, beta)

Generation (decode) phase:
    - causal_conv1d_update: Single-step conv state update
    - fused_recurrent_gated_delta_rule_packed_decode: Packed GDN recurrence

The in_proj and out_proj GEMMs are standard linear layers modeled by the
existing GEMM infrastructure. This collector focuses on the unique GDN ops.

GDN Layer Flow:
    in_proj (GEMM) → Conv1D (packed QKV) → GDN Scan/Update → out_proj (GEMM)
    ^^^^^^^^^^^^^    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^    ^^^^^^^^^^^^^^^^
    Use GEMM model          Benchmarked here            Use GEMM model

Usage:
    python collect_gdn.py

Output:
    gdn_perf.txt - Performance data for GDN Conv1D + scan operations
"""

__compat__ = "sglang==0.5.14"

import gc
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.layers.attention.fla.chunk import chunk_gated_delta_rule
    from sglang.srt.layers.attention.fla.fused_recurrent import (
        fused_recurrent_gated_delta_rule_packed_decode,
    )
    from sglang.srt.layers.attention.mamba.causal_conv1d import causal_conv1d_fn, causal_conv1d_update

import torch

try:
    from collector.case_generator import get_common_gdn_test_cases
    from collector.helper import (
        EXIT_CODE_RESTART,
        benchmark_with_power,
        get_sm_version,
        log_perf,
    )
except ModuleNotFoundError:
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from case_generator import get_common_gdn_test_cases

    from helper import (
        EXIT_CODE_RESTART,
        benchmark_with_power,
        get_sm_version,
        log_perf,
    )

aic_debug = int(os.getenv("aic_gdn_debug", "0"))  # noqa: SIM112
MAX_GDN_CONTEXT_TOKENS = 262_144
MAX_GDN_CONTEXT_VALUE_ELEMENTS = 1_073_741_824


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
    sglang_version: str,
    device: str = "cuda:0",
):
    """
    Benchmark GDN operations for context (prefill) phase using SGLang's Triton FLA kernels.

    Benchmarks:
    1. causal_conv1d_fn  — Conv1D over packed Q+K+V channels
    2. chunk_gated_delta_rule — GDN scan (Q, K, V, g, beta) via SGLang's vendored FLA
    """
    device = torch.device(device)
    torch.cuda.set_device(device)
    torch.set_default_device(device)

    dtype = torch.bfloat16

    qk_dim = num_k_heads * head_k_dim
    value_dim = num_v_heads * head_v_dim
    conv_channels = 2 * qk_dim + value_dim

    if aic_debug:
        print(
            f"GDN Context: d_model={d_model}, num_k_heads={num_k_heads}, head_k_dim={head_k_dim}, "
            f"num_v_heads={num_v_heads}, head_v_dim={head_v_dim}, d_conv={d_conv}"
        )

    conv_weight = torch.randn(conv_channels, d_conv, dtype=dtype, device=device)
    successful_points = 0
    failed_points = 0
    skipped_points = 0

    for batch_size in batch_size_list:
        for seq_len in seq_len_list:
            total_tokens = batch_size * seq_len
            value_elements = total_tokens * value_dim
            if total_tokens > MAX_GDN_CONTEXT_TOKENS or value_elements > MAX_GDN_CONTEXT_VALUE_ELEMENTS:
                skipped_points += 1
                if aic_debug:
                    print(
                        f"  Skipping batch_size={batch_size}, seq_len={seq_len}: "
                        f"total_tokens={total_tokens}, value_elements={value_elements} exceed "
                        "the SGLang 0.5.14 GDN context limits "
                        f"({MAX_GDN_CONTEXT_TOKENS} tokens, "
                        f"{MAX_GDN_CONTEXT_VALUE_ELEMENTS} value elements)"
                    )
                continue
            if aic_debug:
                print(f"  Benchmarking batch_size={batch_size}, seq_len={seq_len}")

            beta = conv_input = conv_state = cu_seqlens = g = has_initial_state = None
            k = mixed_qkv = q = recurrent_state = seq_lens_cpu = state_indices = v = None
            try:
                num_warmups = 3
                num_runs = 10
                cu_seqlens = torch.arange(
                    0,
                    total_tokens + 1,
                    seq_len,
                    dtype=torch.int32,
                    device=device,
                )
                seq_lens_cpu = [seq_len] * batch_size
                state_indices = torch.arange(batch_size, dtype=torch.int32, device=device)
                has_initial_state = torch.zeros(batch_size, dtype=torch.bool, device=device)
                conv_state = torch.zeros(
                    batch_size,
                    conv_channels,
                    d_conv - 1,
                    dtype=dtype,
                    device=device,
                )
                recurrent_state = torch.zeros(
                    batch_size,
                    num_v_heads,
                    head_v_dim,
                    head_k_dim,
                    dtype=torch.float32,
                    device=device,
                )

                # SGLang flattens continuous-batching requests, then
                # transposes packed QKV before the varlen convolution.
                mixed_qkv = torch.randn(total_tokens, conv_channels, dtype=dtype, device=device)
                conv_input = mixed_qkv.transpose(0, 1)
                q = torch.randn(1, total_tokens, num_k_heads, head_k_dim, dtype=dtype, device=device)
                k = torch.randn(1, total_tokens, num_k_heads, head_k_dim, dtype=dtype, device=device)
                v = torch.randn(1, total_tokens, num_v_heads, head_v_dim, dtype=dtype, device=device)
                g = torch.nn.functional.logsigmoid(
                    torch.randn(1, total_tokens, num_v_heads, dtype=torch.float32, device=device)
                )
                beta = torch.sigmoid(torch.randn(1, total_tokens, num_v_heads, dtype=torch.float32, device=device))

                common_log_data = {
                    "phase": "context",
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "num_tokens": total_tokens,
                    "d_model": d_model,
                    "d_conv": d_conv,
                    "num_k_heads": num_k_heads,
                    "head_k_dim": head_k_dim,
                    "num_v_heads": num_v_heads,
                    "head_v_dim": head_v_dim,
                    "model_name": model_name,
                }

                def run_conv1d():
                    causal_conv1d_fn(
                        conv_input,
                        conv_weight,
                        None,
                        query_start_loc=cu_seqlens,
                        cache_indices=state_indices,
                        has_initial_state=has_initial_state,
                        conv_states=conv_state,
                        activation="silu",
                        seq_lens_cpu=seq_lens_cpu,
                    )

                with benchmark_with_power(
                    device=device,
                    kernel_func=run_conv1d,
                    num_warmups=num_warmups,
                    num_runs=num_runs,
                    repeat_n=1,
                    allow_graph_fail=True,
                ) as results:
                    if not log_perf(
                        item_list=[{**common_log_data, "latency": results["latency_ms"]}],
                        framework="SGLang",
                        version=sglang_version,
                        device_name=torch.cuda.get_device_name(device),
                        op_name="gdn",
                        kernel_source="causal_conv1d_fn",
                        perf_filename=perf_filename,
                        power_stats=results["power_stats"],
                    ):
                        raise RuntimeError(f"failed to persist SGLang GDN context row to {perf_filename}")

                def run_gdn_scan():
                    chunk_gated_delta_rule(
                        q,
                        k,
                        v,
                        g,
                        beta,
                        initial_state=recurrent_state,
                        initial_state_indices=state_indices,
                        cu_seqlens=cu_seqlens,
                        head_first=False,
                        use_qk_l2norm_in_kernel=True,
                    )

                with benchmark_with_power(
                    device=device,
                    kernel_func=run_gdn_scan,
                    num_warmups=num_warmups,
                    num_runs=num_runs,
                    repeat_n=1,
                    allow_graph_fail=True,
                ) as results:
                    if not log_perf(
                        item_list=[{**common_log_data, "latency": results["latency_ms"]}],
                        framework="SGLang",
                        version=sglang_version,
                        device_name=torch.cuda.get_device_name(device),
                        op_name="gdn",
                        kernel_source="chunk_gated_delta_rule",
                        perf_filename=perf_filename,
                        power_stats=results["power_stats"],
                    ):
                        raise RuntimeError(f"failed to persist SGLang GDN context row to {perf_filename}")
                successful_points += 1

            except Exception as e:
                failed_points += 1
                print(f"  Error at batch_size={batch_size}, seq_len={seq_len}: {e}")
                continue
            finally:
                beta = conv_input = conv_state = cu_seqlens = g = has_initial_state = None
                k = mixed_qkv = q = recurrent_state = seq_lens_cpu = state_indices = v = None
                cleanup_errors = []
                for cleanup_name, cleanup_fn in (
                    ("gc.collect", gc.collect),
                    ("torch.cuda.empty_cache", torch.cuda.empty_cache),
                ):
                    try:
                        cleanup_fn()
                    except Exception as cleanup_error:
                        cleanup_errors.append(f"{cleanup_name}: {type(cleanup_error).__name__}: {cleanup_error}")
                if cleanup_errors:
                    raise RuntimeError(f"SGLang GDN context cleanup failed: {'; '.join(cleanup_errors)}")

    summary = f"ok={successful_points} error={failed_points} skip={skipped_points}"
    print(f"GDN context summary: {summary}")
    if failed_points or (successful_points == 0 and skipped_points == 0):
        raise RuntimeError(f"SGLang GDN context collection failed strict completeness: {summary}")


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
    sglang_version: str,
    device: str = "cuda:0",
):
    """
    Benchmark GDN operations for generation (decode) phase using SGLang's Triton FLA kernels.

    Benchmarks:
    1. causal_conv1d_update  — Single-step conv state update
    2. fused_recurrent_gated_delta_rule_packed_decode — Packed GDN recurrence
    """
    device = torch.device(device)
    torch.cuda.set_device(device)
    torch.set_default_device(device)

    dtype = torch.bfloat16
    qk_dim = num_k_heads * head_k_dim
    value_dim = num_v_heads * head_v_dim
    conv_channels = 2 * qk_dim + value_dim

    if aic_debug:
        print(
            f"GDN Generation: d_model={d_model}, num_k_heads={num_k_heads}, head_k_dim={head_k_dim}, "
            f"num_v_heads={num_v_heads}, head_v_dim={head_v_dim}, d_conv={d_conv}"
        )

    conv_weight = torch.randn(conv_channels, d_conv, dtype=dtype, device=device)
    successful_points = 0
    failed_points = 0

    for batch_size in batch_size_list:
        if aic_debug:
            print(f"  Benchmarking batch_size={batch_size}")

        a = a_log = b = conv_state = dt_bias = mixed_qkv = None
        output = recurrent_state = state_indices = None
        try:
            num_warmups = 3
            num_runs = 10
            mixed_qkv = torch.randn(batch_size, conv_channels, dtype=dtype, device=device)
            conv_state = torch.zeros(
                batch_size,
                conv_channels,
                d_conv - 1,
                dtype=dtype,
                device=device,
            )
            state_indices = torch.arange(batch_size, dtype=torch.int32, device=device)
            recurrent_state = torch.zeros(
                batch_size,
                num_v_heads,
                head_v_dim,
                head_k_dim,
                dtype=torch.float32,
                device=device,
            )
            a = torch.randn(batch_size, num_v_heads, dtype=dtype, device=device)
            b = torch.randn(batch_size, num_v_heads, dtype=dtype, device=device)
            a_log = torch.zeros(num_v_heads, dtype=torch.float32, device=device)
            dt_bias = torch.ones(num_v_heads, dtype=torch.float32, device=device)
            output = torch.empty(
                batch_size,
                1,
                num_v_heads,
                head_v_dim,
                dtype=dtype,
                device=device,
            )

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

            def run_conv1d_update():
                causal_conv1d_update(
                    mixed_qkv,
                    conv_state,
                    conv_weight,
                    None,
                    activation="silu",
                    conv_state_indices=state_indices,
                )

            with benchmark_with_power(
                device=device,
                kernel_func=run_conv1d_update,
                num_warmups=num_warmups,
                num_runs=num_runs,
                repeat_n=1,
                allow_graph_fail=True,
            ) as results:
                if not log_perf(
                    item_list=[{**common_log_data, "latency": results["latency_ms"]}],
                    framework="SGLang",
                    version=sglang_version,
                    device_name=torch.cuda.get_device_name(device),
                    op_name="gdn",
                    kernel_source="causal_conv1d_update",
                    perf_filename=perf_filename,
                    power_stats=results["power_stats"],
                ):
                    raise RuntimeError(f"failed to persist SGLang GDN generation row to {perf_filename}")

            def run_gdn_update():
                fused_recurrent_gated_delta_rule_packed_decode(
                    mixed_qkv=mixed_qkv,
                    a=a,
                    b=b,
                    A_log=a_log,
                    dt_bias=dt_bias,
                    scale=head_k_dim**-0.5,
                    initial_state=recurrent_state,
                    out=output,
                    ssm_state_indices=state_indices,
                    use_qk_l2norm_in_kernel=True,
                )

            with benchmark_with_power(
                device=device,
                kernel_func=run_gdn_update,
                num_warmups=num_warmups,
                num_runs=num_runs,
                repeat_n=1,
                allow_graph_fail=True,
            ) as results:
                if not log_perf(
                    item_list=[{**common_log_data, "latency": results["latency_ms"]}],
                    framework="SGLang",
                    version=sglang_version,
                    device_name=torch.cuda.get_device_name(device),
                    op_name="gdn",
                    kernel_source="fused_recurrent_gated_delta_rule_packed_decode",
                    perf_filename=perf_filename,
                    power_stats=results["power_stats"],
                ):
                    raise RuntimeError(f"failed to persist SGLang GDN generation row to {perf_filename}")
            successful_points += 1

        except Exception as e:
            failed_points += 1
            print(f"  Error at batch_size={batch_size}: {e}")
            continue
        finally:
            a = a_log = b = conv_state = dt_bias = mixed_qkv = None
            output = recurrent_state = state_indices = None
            cleanup_errors = []
            for cleanup_name, cleanup_fn in (
                ("gc.collect", gc.collect),
                ("torch.cuda.empty_cache", torch.cuda.empty_cache),
            ):
                try:
                    cleanup_fn()
                except Exception as cleanup_error:
                    cleanup_errors.append(f"{cleanup_name}: {type(cleanup_error).__name__}: {cleanup_error}")
            if cleanup_errors:
                raise RuntimeError(f"SGLang GDN generation cleanup failed: {'; '.join(cleanup_errors)}")

    summary = f"ok={successful_points} error={failed_points} skip=0"
    print(f"GDN generation summary: {summary}")
    if failed_points or successful_points == 0:
        raise RuntimeError(f"SGLang GDN generation collection failed strict completeness: {summary}")


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
    Main entry point for GDN benchmarking using SGLang's Triton FLA kernels.

    Routes to appropriate benchmark function based on phase.
    Imports the target SGLang kernels at runtime.
    """
    import contextlib

    with (
        open(os.devnull, "w") as _devnull_file,
        contextlib.redirect_stdout(_devnull_file),
        contextlib.redirect_stderr(_devnull_file),
    ):
        from sglang.srt.layers.attention.fla.chunk import chunk_gated_delta_rule
        from sglang.srt.layers.attention.fla.fused_recurrent import (
            fused_recurrent_gated_delta_rule_packed_decode,
        )
        from sglang.srt.layers.attention.mamba.causal_conv1d import causal_conv1d_fn, causal_conv1d_update

    from importlib.metadata import version as _get_version

    sglang_version = _get_version("sglang")

    globals().update(
        {
            "causal_conv1d_fn": causal_conv1d_fn,
            "causal_conv1d_update": causal_conv1d_update,
            "chunk_gated_delta_rule": chunk_gated_delta_rule,
            "fused_recurrent_gated_delta_rule_packed_decode": fused_recurrent_gated_delta_rule_packed_decode,
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
            sglang_version=sglang_version,
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
            sglang_version=sglang_version,
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
    from importlib.metadata import version as _get_ver

    from collector.registry_types import PerfFile

    print(f"GDN Collector - SGLang {_get_ver('sglang')}")
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
