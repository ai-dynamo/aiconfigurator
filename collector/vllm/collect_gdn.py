# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM 0.24.0 Qwen3.5 Gated DeltaNet collector.

The collector times the two production core operations separately. Prefill
uses vLLM's ``ChunkGatedDeltaRule`` so its default resolver selects the real
FlashInfer or Triton/FLA backend for the current GPU. Decode uses vLLM's packed
recurrent kernel. Both phases use the production packed-QKV convolution width.
Input/output projection GEMMs remain covered by the GEMM collector.
"""

__compat__ = "vllm==0.24.0"

import gc
import os
from types import SimpleNamespace

import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fla.ops import fused_recurrent_gated_delta_rule_packed_decode
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import ChunkGatedDeltaRule
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from vllm.version import __version__ as vllm_version

from collector.case_generator import get_common_gdn_test_cases
from collector.helper import EXIT_CODE_RESTART, benchmark_with_power, get_sm_version, log_perf

aic_debug = int(os.getenv("aic_gdn_debug", "0"))  # noqa: SIM112


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
    device: str = "cuda:0",
):
    """Benchmark the production packed-QKV convolution and prefill GDN op."""
    device = torch.device(device)
    torch.cuda.set_device(device)
    torch.set_default_device(device)

    dtype = torch.bfloat16
    key_dim = num_k_heads * head_k_dim
    value_dim = num_v_heads * head_v_dim
    conv_dim = 2 * key_dim + value_dim

    # ChunkGatedDeltaRule owns vLLM's production backend resolver. The only
    # model field consulted by the default resolver is linear_key_head_dim.
    vllm_config = VllmConfig()
    vllm_config.model_config = SimpleNamespace(hf_text_config=SimpleNamespace(linear_key_head_dim=head_k_dim))
    with set_current_vllm_config(vllm_config):
        chunk_gdn = ChunkGatedDeltaRule()
    prefill_backend = chunk_gdn.gdn_prefill_backend

    if aic_debug:
        print(
            f"GDN Context: d_model={d_model}, conv_dim={conv_dim}, "
            f"num_k_heads={num_k_heads}, head_k_dim={head_k_dim}, "
            f"num_v_heads={num_v_heads}, head_v_dim={head_v_dim}, "
            f"prefill_backend={prefill_backend}"
        )

    conv_weight = torch.randn(conv_dim, d_conv, dtype=dtype, device=device)
    # Qwen3.5 constructs this depthwise convolution with bias=False.
    conv_bias = None

    for batch_size in batch_size_list:
        for seq_len in seq_len_list:
            if aic_debug:
                print(f"  Benchmarking batch_size={batch_size}, seq_len={seq_len}")

            num_warmups = 3
            num_runs = 10
            num_tokens = batch_size * seq_len
            common_log_data = {
                "phase": "context",
                "batch_size": batch_size,
                "seq_len": seq_len,
                "num_tokens": num_tokens,
                "d_model": d_model,
                "d_conv": d_conv,
                "num_k_heads": num_k_heads,
                "head_k_dim": head_k_dim,
                "num_v_heads": num_v_heads,
                "head_v_dim": head_v_dim,
                "model_name": model_name,
            }
            query_start_loc = torch.arange(
                0,
                num_tokens + 1,
                seq_len,
                dtype=torch.int32,
                device=device,
            )

            # Production prefill applies the depthwise convolution to packed
            # QKV, not only to K. Fresh prompts have no initial conv state.
            conv_input = torch.randn(num_tokens, conv_dim, dtype=dtype, device=device).transpose(0, 1)
            conv_state = torch.zeros(
                batch_size + 1,
                conv_dim,
                d_conv - 1,
                dtype=dtype,
                device=device,
            )
            # State slot zero is vLLM's null block and is intentionally never
            # assigned to a live request.
            cache_indices = torch.arange(1, batch_size + 1, dtype=torch.int32, device=device)
            has_initial_state = torch.zeros(batch_size, dtype=torch.bool, device=device)

            def run_conv1d(_conv_input=conv_input, _conv_state=conv_state):
                causal_conv1d_fn(
                    _conv_input,
                    conv_weight,
                    conv_bias,
                    _conv_state,
                    query_start_loc,
                    cache_indices=cache_indices,
                    has_initial_state=has_initial_state,
                    activation="silu",
                )

            run_conv1d()
            torch.cuda.synchronize()
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

            # Release the packed convolution input before allocating Q/K/V for
            # the scan; the largest full-collection points otherwise hold both
            # large representations at once.
            del run_conv1d, conv_input, conv_state
            gc.collect()
            torch.cuda.empty_cache()

            # fused_post_conv_prep produces these packed, batch-flattened
            # tensors in production and performs Q/K normalization itself.
            q = torch.nn.functional.normalize(
                torch.randn(
                    1,
                    num_tokens,
                    num_k_heads,
                    head_k_dim,
                    dtype=dtype,
                    device=device,
                ),
                dim=-1,
            )
            k = torch.nn.functional.normalize(
                torch.randn(
                    1,
                    num_tokens,
                    num_k_heads,
                    head_k_dim,
                    dtype=dtype,
                    device=device,
                ),
                dim=-1,
            )
            v = torch.randn(1, num_tokens, num_v_heads, head_v_dim, dtype=dtype, device=device)
            g = torch.nn.functional.logsigmoid(
                torch.randn(
                    1,
                    num_tokens,
                    num_v_heads,
                    dtype=torch.float32,
                    device=device,
                )
            )
            beta = torch.sigmoid(
                torch.randn(
                    1,
                    num_tokens,
                    num_v_heads,
                    dtype=torch.float32,
                    device=device,
                )
            )
            gdn_state = torch.zeros(
                batch_size,
                num_v_heads,
                head_v_dim,
                head_k_dim,
                dtype=dtype,
                device=device,
            )

            def run_gdn_scan(_q=q, _k=k, _v=v, _g=g, _beta=beta, _state=gdn_state):
                chunk_gdn(
                    q=_q,
                    k=_k,
                    v=_v,
                    g=_g,
                    beta=_beta,
                    initial_state=_state,
                    output_final_state=True,
                    cu_seqlens=query_start_loc,
                    use_qk_l2norm_in_kernel=False,
                )

            run_gdn_scan()
            torch.cuda.synchronize()
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
                    kernel_source=f"chunk_gated_delta_rule_{prefill_backend}",
                    perf_filename=perf_filename,
                    power_stats=results["power_stats"],
                )

            del run_gdn_scan, q, k, v, g, beta, gdn_state
            gc.collect()
            torch.cuda.empty_cache()


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
    """Benchmark packed-QKV convolution and packed recurrent decode."""
    device = torch.device(device)
    torch.cuda.set_device(device)
    torch.set_default_device(device)

    dtype = torch.bfloat16
    key_dim = num_k_heads * head_k_dim
    value_dim = num_v_heads * head_v_dim
    conv_dim = 2 * key_dim + value_dim

    if aic_debug:
        print(
            f"GDN Generation: d_model={d_model}, conv_dim={conv_dim}, "
            f"num_k_heads={num_k_heads}, head_k_dim={head_k_dim}, "
            f"num_v_heads={num_v_heads}, head_v_dim={head_v_dim}"
        )

    conv_weight = torch.randn(conv_dim, d_conv, dtype=dtype, device=device)
    conv_bias = None
    a_log = torch.zeros(num_v_heads, dtype=torch.float32, device=device)
    dt_bias = torch.zeros(num_v_heads, dtype=dtype, device=device)

    for batch_size in batch_size_list:
        if aic_debug:
            print(f"  Benchmarking batch_size={batch_size}")

        num_warmups = 3
        num_runs = 10
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

        conv_input = torch.randn(batch_size, conv_dim, dtype=dtype, device=device)
        conv_state = torch.randn(
            batch_size + 1,
            conv_dim,
            d_conv - 1,
            dtype=dtype,
            device=device,
        )
        state_indices = torch.arange(1, batch_size + 1, dtype=torch.int32, device=device)

        def run_conv1d_update(_conv_input=conv_input, _conv_state=conv_state):
            return causal_conv1d_update(
                _conv_input,
                _conv_state,
                conv_weight,
                conv_bias,
                activation="silu",
                conv_state_indices=state_indices,
                validate_data=True,
            )

        mixed_qkv = run_conv1d_update()
        torch.cuda.synchronize()
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

        del run_conv1d_update, conv_input, conv_state
        gc.collect()
        torch.cuda.empty_cache()

        # vLLM 0.24.0's packed recurrent kernel launches grid-y as
        # batch_size * num_v_heads; CUDA limits grid-y to 65,535. Keep the
        # valid convolution measurement above, then surface the unsupported
        # recurrent point to Collector V2 instead of silently dropping it.
        if batch_size * num_v_heads > 65_535:
            raise RuntimeError(
                "vLLM 0.24.0 packed recurrent GDN exceeds CUDA grid-y limit: "
                f"batch_size={batch_size}, num_v_heads={num_v_heads}, "
                f"grid_y={batch_size * num_v_heads} > 65535"
            )

        # The production non-spec decode fast path feeds packed convolution
        # output directly to this recurrent kernel.
        a = torch.randn(batch_size, num_v_heads, dtype=dtype, device=device)
        b = torch.randn(batch_size, num_v_heads, dtype=dtype, device=device)
        gdn_state = torch.randn(
            batch_size + 1,
            num_v_heads,
            head_v_dim,
            head_k_dim,
            dtype=dtype,
            device=device,
        )
        out = torch.empty(
            batch_size,
            1,
            num_v_heads,
            head_v_dim,
            dtype=dtype,
            device=device,
        )

        def run_gdn_update(
            _mixed_qkv=mixed_qkv,
            _a=a,
            _b=b,
            _state=gdn_state,
            _out=out,
        ):
            fused_recurrent_gated_delta_rule_packed_decode(
                mixed_qkv=_mixed_qkv,
                a=_a,
                b=_b,
                A_log=a_log,
                dt_bias=dt_bias,
                scale=head_k_dim**-0.5,
                initial_state=_state,
                out=_out,
                ssm_state_indices=state_indices,
                use_qk_l2norm_in_kernel=True,
            )

        run_gdn_update()
        torch.cuda.synchronize()
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
                kernel_source="fused_recurrent_gated_delta_rule_packed_decode",
                perf_filename=perf_filename,
                power_stats=results["power_stats"],
            )

        del run_gdn_update, mixed_qkv, a, b, gdn_state, out
        gc.collect()
        torch.cuda.empty_cache()


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
    """Route one collector-v2 GDN case to its phase implementation."""
    if phase == "context":
        if seq_len_list is None:
            raise ValueError("context GDN cases require seq_len_list")
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
