# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang large-EP MoE expert-compute collector (op ``moe_ep``).

Benchmarks the DeepEP MoE expert compute (``DeepEPMoE.run_moe_core``) through a
minimal SGLang engine setup, simulating an EP world of ``moe_ep_size`` ranks on
one GPU by loading only the rank-local expert shard. The module owns rank-local
model runner construction, warmup/measurement and emission of the unified
``moe_expert_compute_perf`` rows (one table, ``inference_phase`` column) consumed by
``aiconfigurator_core.sdk.operations.moe_comm.load_moe_expert_compute_data``.

Shapes are DECLARED: every benchmarked geometry comes from the
``model_case_values.moe`` rows marked ``wideep: true`` crossed with the
``cases/base_ops/moe.yaml`` expert-parallel grid. The live HF config is read
only to assert that the loaded checkpoint agrees with the declaration.
"""

import functools
import json
import logging
import os
import sys

import numpy as np
import torch
import torch.distributed as dist
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.entrypoints.engine import _set_envs_and_config
from sglang.srt.layers.moe import initialize_moe_config
from sglang.srt.layers.moe.token_dispatcher.deepep import (
    DeepEPLLDispatchOutput,
    DeepEPNormalDispatchOutput,
)
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import (
    configure_logger,
    get_bool_env_var,
    set_gpu_proc_affinity,
    suppress_other_loggers,
)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
COLLECTOR_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
if COLLECTOR_ROOT not in sys.path:
    sys.path.append(COLLECTOR_ROOT)

try:
    from helper import _resolve_local_model_path, log_perf, power_law_deepep_decode, power_law_deepep_prefill
except ModuleNotFoundError:
    sys.path.append(COLLECTOR_ROOT)
    from helper import _resolve_local_model_path, log_perf, power_law_deepep_decode, power_law_deepep_prefill
from importlib.metadata import version as get_version
from math import ceil as _ceil

#: The only quantization this collector benchmarks: the DeepEP MoE path is
#: driven with fp8 activations and ``moe_runner_backend="deep_gemm"`` (see
#: ``run_moe_benchmark``), i.e. block-scaled FP8. Cases for models whose
#: declared sglang ``allowed_modes`` exclude it are not generated — a
#: declaration-layer decision, never a relabelled invocation.
MOE_EP_QUANT_MODE = "fp8_block"

#: ``kernel_source`` ground truth: the benchmark invokes
#: ``moe_layer.experts.run_moe_core`` on sglang's ``DeepEPMoE`` module, which
#: the consumer keys as ``deepep_moe``
#: (``sdk/operations/moe_comm.py::_SGLANG_ADAPTED_KERNEL_SOURCES``). The legacy
#: wideep tables spelled the same module ``deepepmoe``; the unified table uses
#: the consumer spelling so new-schema rows are queryable.
MOE_EP_KERNEL_SOURCE = "deepep_moe"

#: Written into the ``op_name`` prefix column of every row; the context /
#: generation split lives in the ``inference_phase`` payload column.
MOE_EP_OP_NAME = "moe_ep"


class MoeEpDeclarationMismatchError(RuntimeError):
    """The loaded checkpoint disagrees with the declared model_case_values row.

    Declared shapes are what gets persisted, so a mismatch would label the row
    with geometry that was not benchmarked. Fail the case instead.
    """


class MoeEpBenchmarkError(RuntimeError):
    """One queued moe_ep benchmark case failed to execute.

    ``layer_permissions.md``: a queued case is executed or raises. The executor
    records this with its case parameters in ``errors_<module>.json``.
    """


def _measure_power_enabled() -> bool:
    """Whether this run measures power (mirrors ``helper._parse_bool_env``).

    Both phase writers gate their power columns on this single per-run flag, so
    one ``moe_expert_compute_perf`` file always has one column set (``helper.log_perf``
    writes the CSV header from the first row it sees).
    """
    value = os.environ.get("COLLECTOR_MEASURE_POWER")
    return False if value is None else value.lower() in ("true", "1", "yes")


def _power_columns(power_stats) -> dict | None:
    """D7: emit power only where the bench measures it, and never as 0.0.

    Returns ``None`` (no power columns at all) when the run does not measure
    power; otherwise the measured stats, or NaN when the sampler produced no
    samples — an unknown, which is not the same fact as "idle at zero watts".
    """
    if not _measure_power_enabled():
        return None
    if power_stats and power_stats.get("power") is not None:
        return power_stats
    return {"power": float("nan"), "power_limit": float("nan")}


def _power_device():
    """The concrete CUDA device the NVML sampler needs (``device.index``)."""
    return torch.device("cuda", torch.cuda.current_device())


def _moe_expert_compute_perf_path(output_path, perf_filename) -> str:
    """Resolve the registry-provided perf filename into the run's output dir.

    ``perf_filename`` is ``PerfFile.MOE_EXPERT_COMPUTE`` as bound by ``collect.py``; no
    collector-local filename literals.
    """
    directory = output_path if output_path is not None else os.getcwd()
    return os.path.join(directory, str(perf_filename))


def _build_moe_ep_row(
    *,
    moe_dtype: str,
    distribution: str,
    inference_phase: str,
    num_tokens: int,
    hidden_size: int,
    inter_size: int,
    topk: int,
    num_experts: int,
    num_slots: int,
    moe_tp_size: int,
    moe_ep_size: int,
    latency_ms: float,
) -> dict:
    """Build one unified ``moe_expert_compute_perf`` payload row.

    Column set and order are the consumer contract
    (``sdk/operations/moe_comm.py::load_moe_expert_compute_data``); the
    ``framework/version/device/op_name/kernel_source`` prefix is added by
    ``helper.log_perf``. ``latency`` is milliseconds — the loader stores the
    column raw, unlike the microsecond-collected a2a table.
    """
    return {
        "moe_dtype": moe_dtype,
        "distribution": distribution,
        "inference_phase": inference_phase,
        "num_tokens": int(num_tokens),
        "hidden_size": int(hidden_size),
        "inter_size": int(inter_size),
        "topk": int(topk),
        "num_experts": int(num_experts),
        # sglang has no EPLB redundant-expert axis today: every routed expert
        # owns exactly one slot. The redundancy axis stays trtllm-side until a
        # declared case axis exists here.
        "num_slots": int(num_slots),
        "moe_tp_size": int(moe_tp_size),
        "moe_ep_size": int(moe_ep_size),
        "latency": float(latency_ms),
    }


def _is_scale_ue8m0() -> bool:
    """Check if deep_gemm uses ue8m0 scale format (Blackwell GPUs)."""
    try:
        from sglang.srt.layers.deep_gemm_wrapper import configurer as deep_gemm_wrapper

        return getattr(deep_gemm_wrapper, "DEEPGEMM_SCALE_UE8M0", False)
    except ImportError:
        return False


def _make_scale_tensor(num_tokens: int, hidden_size: int, device) -> torch.Tensor:
    """Create a scale tensor matching the format expected by run_moe_core.

    On Blackwell (ue8m0), scales are packed as int32 with 4 scales per element.
    On older GPUs, scales are float32 with one per 128-element block.
    """
    scale_dim = hidden_size // 128
    if _is_scale_ue8m0():
        return torch.ones(
            num_tokens,
            _ceil(scale_dim / 4),
            device=device,
            dtype=torch.int32,
        )
    return torch.ones(num_tokens, scale_dim, device=device, dtype=torch.float32)


def _selected_moe_model_id() -> str:
    # collect.py sets COLLECTOR_MODEL_PATH from --model-path / case_plan.model_path.
    return (
        os.environ.get("COLLECTOR_MODEL_PATH")
        or os.environ.get("MOE_MODEL_PATH")
        or os.environ.get("DEEPSEEK_MODEL_PATH")
        or "deepseek-ai/DeepSeek-V3"
    )


@functools.cache
def _resolve_moe_model_path(model_id: str) -> str:
    return _resolve_local_model_path(model_id)


def _get_moe_model_path() -> str:
    """Resolve MoE model path lazily so that ``collect.py``'s registry import
    of this module does not trigger tempdir / JSON I/O on every invocation.

    Cached per model id for the lifetime of the process; subprocess workers each get a
    fresh interpreter and re-resolve on first call (which converges on the
    same deterministic tempdir built by ``helper._resolve_local_model_path``).
    """
    return _resolve_moe_model_path(_selected_moe_model_id())


def _case_model_path(model_name: str) -> str:
    """The model artifact for one declared case.

    ``collect.py`` sets ``COLLECTOR_MODEL_PATH`` only for single-model plans
    (``--model-path``), where it points at the operator's artifact for the one
    model the plan expanded — it wins. A full plan carries cases from every
    declared wideep model, so each case resolves its OWN declared
    ``model_name``: the loaded checkpoint's config is the consistency oracle
    for the declaration asserts in :func:`run_moe`, and loading a default
    model for another model's case would fail them (or worse, benchmark the
    wrong geometry).
    """
    return _resolve_moe_model_path(os.environ.get("COLLECTOR_MODEL_PATH") or model_name)


def _sorted_phase_cases(test_cases):
    """Sort phase cases on the non-token key axis first (D5: sorted emission).

    Rows land in the perf file grouped by distribution, ascending in tokens
    within each group, so the persisted table is deterministic regardless of
    how the case list was built.
    """
    return sorted(
        test_cases,
        key=lambda case: (
            case["distributed"],
            case["power_law_alpha"] if case.get("power_law_alpha") is not None else -1.0,
            case["num_tokens"],
        ),
    )


def get_moe_prefill_test_cases(rank, *, topk, num_experts):
    """Get test cases for MoE prefill phase including distribution and alpha.

    Returns a list of dicts with keys: 'num_tokens', 'distributed', 'power_law_alpha'.
    For uniform distribution, 'power_law_alpha' is None.

    The uniform variant only exists where its synthetic workload is non-empty:
    ``num_token * topk * ep // num_experts`` tokens per local expert must be
    positive, or the rank under measurement receives nothing (pure declared
    arithmetic, so it is resolved HERE, at generation time — the benchmark
    loop runs this list exactly and treats an empty workload as an invariant
    breach). Power-law variants sample their own per-expert counts and keep
    every token point.
    """
    test_cases = []
    num_tokens = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    power_law_alphas = [0.6, 0.8, 1.01, 1.02, 1.2]

    dropped_small_workload = []
    dropped_oversized = []
    dropped_empty_uniform = []
    for num_token in sorted(num_tokens):
        if num_token * 8 < 128:
            dropped_small_workload.append(num_token)
            continue
        if num_token * rank > 256 * 2048:
            dropped_oversized.append(num_token)
            continue
        # Uniform
        if int(num_token * topk * rank // num_experts) <= 0:
            dropped_empty_uniform.append(num_token)
        else:
            test_cases.append({"num_tokens": num_token, "distributed": "uniform", "power_law_alpha": None})
        # Power-law variants
        for alpha in power_law_alphas:
            test_cases.append(
                {
                    "num_tokens": num_token,
                    "distributed": "power_law",
                    "power_law_alpha": alpha,
                }
            )

    if dropped_small_workload:
        print(
            f"moe_ep context: dropped {len(dropped_small_workload)} token points "
            f"{dropped_small_workload} below the minimum measurable workload "
            "(num_token * 8 < 128)"
        )
    if dropped_oversized:
        print(
            f"moe_ep context: dropped {len(dropped_oversized)} token points "
            f"{dropped_oversized} above the per-rank token budget "
            f"(num_token * ep={rank} > 256 * 2048)"
        )
    if dropped_empty_uniform:
        print(
            f"moe_ep context: dropped {len(dropped_empty_uniform)} uniform token points "
            f"{dropped_empty_uniform} at generation (num_token * topk={topk} * ep={rank} // "
            f"num_experts={num_experts} yields 0 tokens per local expert); power-law variants kept"
        )
    return _sorted_phase_cases(test_cases)


def get_moe_decode_test_cases():
    """Get test cases for MoE decode phase including distribution and alpha.

    Returns a list of dicts with keys: 'num_tokens', 'distributed', 'power_law_alpha'.
    For uniform distribution, 'power_law_alpha' is None.
    """
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    power_law_alphas = [0.6, 0.8, 1.01, 1.02, 1.2]
    test_cases = []
    # Uniform cases
    for bs in batch_sizes:
        test_cases.append(
            {
                "num_tokens": bs,
                "distributed": "uniform",
                "power_law_alpha": None,
            }
        )
    # Power-law cases
    for bs in batch_sizes:
        for alpha in power_law_alphas:
            test_cases.append(
                {
                    "num_tokens": bs,
                    "distributed": "power_law",
                    "power_law_alpha": alpha,
                }
            )
    return _sorted_phase_cases(test_cases)


def load_model_with_dummy_weights(server_args, port_args, tp_rank):
    """Load model with dummy weights and limited layers for MoE testing"""
    suppress_other_loggers()
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None

    if server_args.load_format == "dummy":
        existing_override = {}
        if server_args.json_model_override_args:
            existing_override = json.loads(server_args.json_model_override_args)

        existing_override["num_hidden_layers"] = 4
        server_args.json_model_override_args = json.dumps(existing_override)

    model_config = ModelConfig.from_server_args(server_args)
    rank_print(f"Loading model with {model_config.num_hidden_layers} layers")
    rank_print("Will test MoE module from layer 3 (4th layer, 0-indexed)")

    model_runner = ModelRunner(
        model_config=model_config,
        mem_fraction_static=server_args.mem_fraction_static,
        gpu_id=tp_rank,
        tp_rank=tp_rank,
        tp_size=server_args.tp_size,
        pp_rank=0,
        pp_size=1,
        moe_ep_rank=tp_rank,
        moe_ep_size=server_args.ep_size,
        nccl_port=port_args.nccl_port,
        server_args=server_args,
    )

    rank_print("Model loaded successfully.")

    if server_args.tp_size > 1:
        dist.barrier()

    return model_runner


def benchmark_moe_layer_prefill(
    model_runner,
    server_args,
    port_args,
    num_warmup,
    num_iterations,
    test_layer,
    rank_print,
    device,
    tp_rank,
    prefill_test_cases,
    moe_layer,
    num_local_experts,
    simulated_ep_size,
    output_path,
    perf_filename,
    model_hidden_size,
    model_inter_size,
    model_total_experts,
    model_topk,
    num_slots,
    moe_dtype,
):
    """Benchmark MoE layer in the context (prefill) phase.

    Args:
        num_local_experts: Number of experts on this GPU (= declared num_experts // moe_ep_size)
        simulated_ep_size: The EP size being simulated (declared moe_ep_size)
        perf_filename: Registry-provided ``PerfFile.MOE_EXPERT_COMPUTE`` filename
        model_hidden_size: Declared hidden_size
        model_inter_size: Declared per-expert inter_size
        model_total_experts: Declared total expert count
        model_topk: Declared top-k — the value persisted in the row (the live
            ``moe_layer.topk`` read is the subject of run_moe's assert, not the
            row's source)
        num_slots: Declared expert slots (== model_total_experts on sglang)
        moe_dtype: Declared quantization label for the persisted row
    """

    for case in prefill_test_cases:
        try:
            # Backward compatible: old format was just an int
            if isinstance(case, dict):
                num_token = case["num_tokens"]
                distributed = case.get("distributed", "uniform")
                power_law_alpha = case.get("power_law_alpha", 0.8) if distributed == "power_law" else None
            else:
                num_token = int(case)
                distributed = "uniform"
                power_law_alpha = None

            model_runner.req_to_token_pool.clear()
            model_runner.token_to_kv_pool_allocator.clear()

            # Fake dispatch outputs with random data
            hidden_states_per_token_iter = torch.randn(
                int(num_token * simulated_ep_size),
                model_runner.model.config.hidden_size,
                dtype=torch.bfloat16,
                device=device,
            )

            if hidden_states_per_token_iter.shape[1] % 128 != 0:
                pad_size = 128 - (hidden_states_per_token_iter.shape[1] % 128)
                hidden_states_per_token_iter = torch.nn.functional.pad(hidden_states_per_token_iter, (0, pad_size))

            hidden_states_fp8_tensor_iter = hidden_states_per_token_iter.to(torch.float8_e4m3fn)
            scale_tensor_iter = _make_scale_tensor(
                hidden_states_per_token_iter.shape[0],
                hidden_states_per_token_iter.shape[1],
                hidden_states_per_token_iter.device,
            )

            num_tokens_iter = hidden_states_per_token_iter.shape[0]
            topk = moe_layer.topk.topk_config.top_k
            topk_idx_iter = torch.full((num_tokens_iter, topk), -1, device=device, dtype=torch.int32)
            topk_weights_iter = torch.zeros((num_tokens_iter, topk), device=device, dtype=torch.float32)

            if distributed == "uniform":
                tokens_per_local_expert = int(num_token * topk * simulated_ep_size // model_total_experts)
                rank_print(f"tokens_per_local_expert: {tokens_per_local_expert}")
                if tokens_per_local_expert <= 0:
                    # get_moe_prefill_test_cases resolves this pure declared
                    # arithmetic at generation time, so an empty uniform
                    # workload here means the manifest and the runtime have
                    # drifted apart. Raise rather than skip: a queued point
                    # either runs or fails classified.
                    raise MoeEpBenchmarkError(
                        f"moe_ep context: uniform num_tokens={num_token} reached the benchmark with "
                        f"0 tokens per local expert (topk={topk}, ep={simulated_ep_size}, "
                        f"num_experts={model_total_experts}); the generation-time manifest should "
                        "have excluded it"
                    )
                num_recv = [tokens_per_local_expert] * num_local_experts

                total_valid_positions = sum(num_recv)
                expert_indices_list = []
                for expert_id in range(num_local_experts):
                    expert_indices_list.extend([expert_id] * tokens_per_local_expert)

                expert_indices_tensor = torch.tensor(expert_indices_list, device=device, dtype=torch.int32)
                shuffled_indices = torch.randperm(len(expert_indices_tensor), device=device)
                expert_indices_tensor = expert_indices_tensor[shuffled_indices]

                positions_per_row = total_valid_positions // num_tokens_iter
                extra_positions = total_valid_positions % num_tokens_iter

                valid_positions_count = 0
                for i in range(num_tokens_iter):
                    current_row_positions = positions_per_row + (1 if i < extra_positions else 0)
                    for j in range(current_row_positions):
                        if valid_positions_count < total_valid_positions:
                            topk_idx_iter[i, j % topk] = expert_indices_tensor[valid_positions_count]
                            valid_positions_count += 1
                        else:
                            break

                # Uniform weights across used columns
                for i in range(num_tokens_iter):
                    used_mask = topk_idx_iter[i] != -1
                    if used_mask.any():
                        topk_weights_iter[i, used_mask] = 1.0 / topk

            elif distributed == "power_law":
                # Use power_law_deepep_prefill to generate router logits for local experts
                # Generate multiple samples to avoid outliers from a single sampling
                power_law_samples = []
                for _ in range(5):
                    topk_idx_sample, topk_weights_sample, num_recv_tensor = power_law_deepep_prefill(
                        num_tokens_iter,
                        num_local_experts * simulated_ep_size,
                        topk,
                        simulated_ep_size,
                        power_law_alpha if power_law_alpha is not None else 0.8,
                    )
                    topk_idx_sample = topk_idx_sample.to(device).contiguous()
                    topk_weights_sample = topk_weights_sample.to(device).contiguous()
                    topk_weights_sample = torch.nan_to_num(topk_weights_sample, nan=0.0, posinf=0.0, neginf=0.0)
                    num_recv = num_recv_tensor.tolist()
                    power_law_samples.append((topk_idx_sample, topk_weights_sample, num_recv))

            else:
                raise ValueError(f"Unsupported distributed mode: {distributed}")

            # For uniform distribution, create a single-element list for unified processing
            if distributed == "uniform":
                # Safety clamp for weights
                topk_weights_iter = torch.nan_to_num(topk_weights_iter, nan=0.0, posinf=0.0, neginf=0.0)
                power_law_samples = [(topk_idx_iter, topk_weights_iter, num_recv)]

            # Warmup
            for _ in range(num_warmup):
                for topk_idx_sample, topk_weights_sample, num_recv_sample in power_law_samples:
                    hidden_states_fp8_tensor_iter = hidden_states_per_token_iter.to(torch.float8_e4m3fn)
                    scale_tensor_iter = _make_scale_tensor(
                        hidden_states_per_token_iter.shape[0],
                        hidden_states_per_token_iter.shape[1],
                        hidden_states_per_token_iter.device,
                    )
                    dispatch_output = DeepEPNormalDispatchOutput(
                        hidden_states=hidden_states_fp8_tensor_iter,
                        hidden_states_scale=scale_tensor_iter,
                        topk_ids=topk_idx_sample.clone(),
                        topk_weights=topk_weights_sample.clone(),
                        num_recv_tokens_per_expert=num_recv_sample,
                    )
                    _ = moe_layer.experts.run_moe_core(dispatch_output)

            torch.get_device_module(device).synchronize()
            torch.cuda.empty_cache()

            gemm_latencies = []

            # Power is sampled by a background NVML thread around the existing
            # event timing (D7) — the measurement method is unchanged.
            from helper import power_monitoring_only

            with power_monitoring_only(_power_device()) as power_monitor:
                for i in range(num_iterations):
                    for topk_idx_sample, topk_weights_sample, num_recv_sample in power_law_samples:
                        hidden_states_fp8_tensor_iter = hidden_states_per_token_iter.to(torch.float8_e4m3fn)
                        scale_tensor_iter = _make_scale_tensor(
                            hidden_states_per_token_iter.shape[0],
                            hidden_states_per_token_iter.shape[1],
                            hidden_states_per_token_iter.device,
                        )
                        dispatch_output = DeepEPNormalDispatchOutput(
                            hidden_states=hidden_states_fp8_tensor_iter,
                            hidden_states_scale=scale_tensor_iter,
                            topk_ids=topk_idx_sample.clone(),
                            topk_weights=topk_weights_sample.clone(),
                            num_recv_tokens_per_expert=num_recv_sample,
                        )
                        torch.get_device_module(device).synchronize()
                        start_event = torch.cuda.Event(enable_timing=True)
                        end_event = torch.cuda.Event(enable_timing=True)
                        start_event.record()

                        _ = moe_layer.experts.run_moe_core(dispatch_output)

                        end_event.record()
                        end_event.synchronize()
                        latency_ms = start_event.elapsed_time(end_event)
                        if i > 2:
                            gemm_latencies.append(latency_ms)
                power_stats = power_monitor.stop_sampling() if power_monitor is not None else None

            torch.cuda.empty_cache()

            avg_latency_ms = np.mean(gemm_latencies)

            if tp_rank == 0:
                rank_print("DeepEP MoE GEMM Results (Prefill):")
                rank_print(f"  Average latency: {avg_latency_ms:.3f}ms")
                distribution_str = f"power_law_{power_law_alpha}" if distributed == "power_law" else distributed
                # log_perf reports write failures by returning False — fail
                # closed, or the executor checkpoints this case as passed
                # with rows missing.
                if not log_perf(
                    item_list=[
                        _build_moe_ep_row(
                            moe_dtype=moe_dtype,
                            distribution=distribution_str,
                            inference_phase="context",
                            num_tokens=num_token * simulated_ep_size,
                            hidden_size=model_hidden_size,
                            inter_size=model_inter_size,
                            topk=model_topk,
                            num_experts=model_total_experts,
                            num_slots=num_slots,
                            moe_tp_size=1,
                            moe_ep_size=simulated_ep_size,
                            latency_ms=avg_latency_ms,
                        )
                    ],
                    framework="SGLang",
                    version=get_version("sglang"),
                    device_name=torch.cuda.get_device_name(server_args.device),
                    op_name=MOE_EP_OP_NAME,
                    kernel_source=MOE_EP_KERNEL_SOURCE,
                    perf_filename=_moe_expert_compute_perf_path(output_path, perf_filename),
                    power_stats=_power_columns(power_stats),
                ):
                    raise MoeEpBenchmarkError(
                        f"helper.log_perf failed to persist the measured context row "
                        f"(num_tokens={num_token * simulated_ep_size}, ep={simulated_ep_size})"
                    )
            del (
                hidden_states_per_token_iter,
                hidden_states_fp8_tensor_iter,
                scale_tensor_iter,
                topk_idx_iter,
                topk_weights_iter,
                num_recv,
                dispatch_output,
            )
            torch.cuda.empty_cache()

        except Exception as e:
            # Execute-or-raise: the failure is classified and recorded by the
            # executor with this case's parameters, never swallowed. A CUDA
            # fault additionally resets the worker process (collect.py
            # _is_cuda_fatal), so no cache cleanup is attempted here.
            raise MoeEpBenchmarkError(
                f"moe_ep context case failed (num_tokens={case}, "
                f"moe_ep_size={simulated_ep_size}, num_experts={model_total_experts}): {e}"
            ) from e


def benchmark_moe_layer_decode(
    model_runner,
    server_args,
    port_args,
    num_warmup,
    num_iterations,
    test_layer,
    rank_print,
    device,
    tp_rank,
    decode_test_cases,
    moe_layer,
    num_local_experts,
    simulated_ep_size,
    output_path,
    perf_filename,
    model_hidden_size,
    model_inter_size,
    model_total_experts,
    model_topk,
    num_slots,
    moe_dtype,
):
    """Benchmark MoE layer in the generation (decode) phase.

    Argument semantics are identical to :func:`benchmark_moe_layer_prefill`;
    both phases write the same unified ``moe_expert_compute_perf`` table and differ only in
    the ``inference_phase`` column.
    """
    model_runner.req_to_token_pool.clear()
    model_runner.token_to_kv_pool_allocator.clear()
    top_k = moe_layer.topk.topk_config.top_k

    for case in decode_test_cases:
        try:
            num_token = case["num_tokens"]
            distributed = case["distributed"]
            power_law_alpha = case.get("power_law_alpha", 0.8) if distributed == "power_law" else None
            # FIXME(kernel-limit): DeepEP's low-latency path sizes its dispatch
            # buffers from `num_max_dispatch_tokens_per_rank` (the deep_ep
            # `Buffer.low_latency_dispatch` / `get_low_latency_rdma_size_hint`
            # API contract), so a rank may not receive more than
            # num_max_dispatch_tokens_per_rank * num_ranks tokens for any one
            # expert. 128 is the value this benchmark pins for the synthetic
            # buffers below; it is the claimed DeepEP limit at the wideep pin
            # (sglang 0.5.10 + DeepEP, framework_manifest.yaml wideep_sglang)
            # and is UNVERIFIED against the framework source — sglang is not
            # vendored here, so no file:line citation is possible from this
            # repo. Re-check on the next wideep version bump: either probe the
            # buffer for its real capacity or replace this with a guard that
            # raises citing the verified source line.
            num_max_dispatch_tokens_per_rank = 128

            # The declared decode batch grid tops out at 128 == the buffer
            # bound above, so no decode case can exceed it. Raise rather than
            # skip if that invariant ever breaks: an oversized batch would
            # index past the synthetic dispatch buffers allocated below.
            if num_token > num_max_dispatch_tokens_per_rank:
                raise MoeEpBenchmarkError(
                    f"moe_ep generation: num_tokens={num_token} exceeds the DeepEP low-latency "
                    f"dispatch bound num_max_dispatch_tokens_per_rank={num_max_dispatch_tokens_per_rank}; "
                    "the decode sweep and the buffer bound have drifted apart"
                )

            hidden_size = model_runner.model.config.hidden_size

            if hidden_size % 128 != 0:
                pad_size = 128 - (hidden_size % 128)
                hidden_size += pad_size

            hidden_states = torch.randn(
                num_local_experts,
                num_max_dispatch_tokens_per_rank * simulated_ep_size,
                hidden_size,
                dtype=torch.bfloat16,
                device="cuda",
            )

            scale_hidden_size = hidden_size // 128
            scale_tensor = torch.ones(
                num_local_experts,
                num_max_dispatch_tokens_per_rank * simulated_ep_size,
                scale_hidden_size,
                device=hidden_states.device,
                dtype=torch.float32,
            )
            hidden_states_fp8_tensor = hidden_states.to(torch.float8_e4m3fn)

            masked_m = torch.zeros(num_local_experts, device=device, dtype=torch.int32)

            # support two distributed mode: power_law and uniform
            if distributed == "power_law":
                masked_m_list = [
                    power_law_deepep_decode(
                        num_token * simulated_ep_size,
                        num_local_experts * simulated_ep_size,
                        top_k,
                        simulated_ep_size,
                        power_law_alpha,
                    )
                    .to(masked_m.dtype)
                    .to(torch.device(device))
                    for _ in range(5)
                ]
            elif distributed == "uniform":
                # Total experts = model_total_experts, simulated_ep_size = model_total_experts / num_local_experts
                base_tokens_per_expert = int(num_token * top_k) * simulated_ep_size // model_total_experts
                if base_tokens_per_expert == 0:
                    # Each expert that receives tokens gets exactly 1 token
                    # Number of experts with tokens on this card = total_calls / simulated_ep_size
                    # = (num_token * top_k * num_rank) / num_rank = num_token * top_k
                    masked_m[: int(num_token * top_k)] = 1
                else:
                    masked_m[:] = base_tokens_per_expert
                masked_m_list = [masked_m]
            else:
                raise ValueError(f"Unsupported distributed mode: {distributed}")
            max_masked_m = int(torch.stack([mm.max() for mm in masked_m_list]).max().item())
            if max_masked_m > hidden_states.shape[1]:
                # Same FIXME(kernel-limit) bound as above, expressed per
                # expert: this sampled skew routes more tokens to one expert
                # than the synthetic buffer holds. Raise rather than skip —
                # a declared token point that cannot run fails classified
                # (a rerun re-samples), it does not silently thin the curve.
                raise MoeEpBenchmarkError(
                    f"moe_ep generation: num_tokens={num_token} drew a routing sample with "
                    f"max masked_m {max_masked_m} exceeding the dispatch buffer "
                    f"{hidden_states.shape[1]} (num_max_dispatch_tokens_per_rank * ep)"
                )
            scale_tensor = torch.ones(
                num_local_experts,
                num_max_dispatch_tokens_per_rank * simulated_ep_size,
                scale_hidden_size,
                device=hidden_states.device,
                dtype=torch.float32,
            )
            hidden_states_fp8_tensor = hidden_states.to(torch.float8_e4m3fn)

            topk_idx_empty = torch.empty(0, device=device, dtype=torch.int32)
            topk_weights_empty = torch.empty(0, device=device, dtype=torch.float32)

            torch.get_device_module(device).synchronize()
            torch.cuda.empty_cache()

            for _ in range(num_warmup):
                dispatch_output_list = []
                for masked_m in masked_m_list:
                    hidden_states_fp8_tensor_copy = hidden_states_fp8_tensor.clone()
                    scale_tensor_copy = scale_tensor.clone()

                    output = DeepEPLLDispatchOutput(
                        hidden_states=hidden_states_fp8_tensor_copy,
                        hidden_states_scale=scale_tensor_copy,
                        topk_ids=topk_idx_empty,
                        topk_weights=topk_weights_empty,
                        masked_m=masked_m,
                        expected_m=int(torch.ceil(masked_m.float().mean()).item()),
                    )
                    dispatch_output_list.append(output)

                for dispatch_output in dispatch_output_list:
                    _ = moe_layer.experts.run_moe_core(dispatch_output)

            torch.get_device_module(device).synchronize()
            torch.cuda.empty_cache()

            # Use benchmark_with_power for timing
            from helper import benchmark_with_power

            # Pre-compute expected_m values outside of kernel_func to avoid .item() during CUDA graph capture
            expected_m_list = [int(torch.ceil(masked_m_item.float().mean()).item()) for masked_m_item in masked_m_list]

            # Pre-clone masked_m tensors (they won't be disposed by run_moe_core)
            masked_m_clones = [m.clone() for m in masked_m_list]

            # Pre-create enough tensor copies to avoid clone() inside kernel_func
            # run_moe_core disposes hidden_states and hidden_states_scale via dispose_tensor()
            # Estimate: kernel_func called ~4 times (warmup 3 + capture 1) in graph mode
            # Each call iterates len(masked_m_list) times (max 5 for power_law)
            # Total: 4 * 5 = 20 tensor sets needed, use 50 for safety
            num_masked_m = len(masked_m_list)
            num_kernel_calls = 20  # Conservative estimate for kernel_func invocations
            num_tensor_sets = num_kernel_calls * num_masked_m

            hidden_states_copies = []
            scale_copies = []
            for _ in range(num_tensor_sets):
                hidden_states_copies.append(
                    torch.randn(
                        num_local_experts,
                        num_max_dispatch_tokens_per_rank * simulated_ep_size,
                        hidden_size,
                        dtype=torch.bfloat16,
                        device=device,
                    ).to(torch.float8_e4m3fn)
                )
                scale_copies.append(
                    torch.ones(
                        num_local_experts,
                        num_max_dispatch_tokens_per_rank * simulated_ep_size,
                        scale_hidden_size,
                        device=device,
                        dtype=torch.float32,
                    )
                )

            # Use a mutable container to track tensor index across all run_moe_core calls
            tensor_idx = [0]

            def kernel_func():
                for masked_m_clone, expected_m_val in zip(masked_m_clones, expected_m_list, strict=True):
                    idx = tensor_idx[0] % num_tensor_sets
                    tensor_idx[0] += 1
                    dispatch_output = DeepEPLLDispatchOutput(
                        hidden_states=hidden_states_copies[idx],
                        hidden_states_scale=scale_copies[idx],
                        topk_ids=torch.empty(0, device=device, dtype=torch.int32),
                        topk_weights=torch.empty(0, device=device, dtype=torch.float32),
                        masked_m=masked_m_clone,
                        expected_m=expected_m_val,
                    )
                    _ = moe_layer.experts.run_moe_core(dispatch_output)

            with benchmark_with_power(
                device=_power_device(),
                kernel_func=kernel_func,
                num_warmups=3,
                num_runs=num_iterations,
                repeat_n=1,
            ) as results:
                pass

            avg_latency_ms = results["latency_ms"] / len(masked_m_list)
            power_stats = results["power_stats"]

            if tp_rank == 0:
                rank_print("DeepEP MoE GEMM Results (Decode) - CUDA Graph Enabled:")
                rank_print(f"  Average latency: {avg_latency_ms:.3f}ms")
                distribution_str = f"power_law_{power_law_alpha}" if distributed == "power_law" else distributed
                # Fail closed on a reported write failure, mirroring the
                # context loop above.
                if not log_perf(
                    item_list=[
                        _build_moe_ep_row(
                            moe_dtype=moe_dtype,
                            distribution=distribution_str,
                            inference_phase="generation",
                            num_tokens=num_token * simulated_ep_size,
                            hidden_size=model_hidden_size,
                            inter_size=model_inter_size,
                            topk=model_topk,
                            num_experts=model_total_experts,
                            num_slots=num_slots,
                            moe_tp_size=1,
                            moe_ep_size=simulated_ep_size,
                            latency_ms=avg_latency_ms,
                        )
                    ],
                    framework="SGLang",
                    version=get_version("sglang"),
                    device_name=torch.cuda.get_device_name(server_args.device),
                    op_name=MOE_EP_OP_NAME,
                    kernel_source=MOE_EP_KERNEL_SOURCE,
                    perf_filename=_moe_expert_compute_perf_path(output_path, perf_filename),
                    power_stats=_power_columns(power_stats),
                ):
                    raise MoeEpBenchmarkError(
                        f"helper.log_perf failed to persist the measured generation row "
                        f"(num_tokens={num_token * simulated_ep_size}, ep={simulated_ep_size})"
                    )
            del hidden_states, hidden_states_fp8_tensor, scale_tensor, dispatch_output_list
            torch.cuda.empty_cache()

        except Exception as e:
            # Execute-or-raise, same doctrine as the context loop above.
            raise MoeEpBenchmarkError(
                f"moe_ep generation case failed (case={case}, "
                f"moe_ep_size={simulated_ep_size}, num_experts={model_total_experts}): {e}"
            ) from e


def _assert_declared(field: str, declared, live) -> None:
    """Consistency assert: the loaded checkpoint must match the declared row.

    The persisted row carries the DECLARED geometry, so a divergence means the
    case would be labelled with a shape that was not benchmarked. Raise instead
    (``case_authoring.md``: unresolvable declarations fail loudly).
    """
    if live is None:
        raise MoeEpDeclarationMismatchError(
            f"moe_ep: could not read {field} from the loaded checkpoint to verify the declared value {declared!r}"
        )
    if int(declared) != int(live):
        raise MoeEpDeclarationMismatchError(
            f"moe_ep: declared {field}={declared} but the loaded checkpoint reports {live}; "
            "update collector/cases/models/<architecture>_cases.yaml or the model path"
        )


def run_moe(
    server_args,
    port_args,
    num_warmup,
    num_iterations,
    test_layer,
    num_experts,
    tp_rank,
    output_path,
    perf_filename,
    moe_ep_size,
    model_topk,
    model_hidden_size,
    model_inter_size,
    model_total_experts,
    num_slots,
    moe_dtype,
):
    """Run the complete moe_ep benchmark for one declared (model, EP) case.

    ``num_experts`` is the rank-local expert count the model is loaded with;
    every other shape argument is the DECLARED value that will be persisted.
    """

    if get_bool_env_var("SGLANG_SET_CPU_AFFINITY"):
        set_gpu_proc_affinity(server_args.tp_size, server_args.nnodes, tp_rank)

    configure_logger(server_args, prefix=f" TP{tp_rank}")

    # Initialize MoE config in subprocess (required for DeepEP + DeepGEMM backend)
    _set_envs_and_config(server_args)
    initialize_moe_config(server_args)

    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None

    rank_print(f"\n{'=' * 60}")
    rank_print(f"Testing MoE Layer {test_layer}")
    rank_print(f"{'=' * 60}")

    rank_print(f"\n{'=' * 50}")
    rank_print(f"Testing with {num_experts} experts")
    rank_print(f"{'=' * 50}")

    # Read the ORIGINAL model config BEFORE applying the expert override:
    # it is the consistency oracle for the declared shapes, never their
    # source.
    original_json_override = server_args.json_model_override_args
    original_model_config = ModelConfig.from_server_args(server_args)
    original_hf_config = original_model_config.hf_config
    # Per-expert MLP intermediate size. The HF field name varies:
    #   moe_intermediate_size  - DeepSeek-V3, Qwen3-MoE, GPT-OSS
    #   intermediate_size      - MiniMax-M2 (Mixtral-style; no separate dense MLP)
    live_inter_size = getattr(original_hf_config, "moe_intermediate_size", None) or getattr(
        original_hf_config, "intermediate_size", None
    )
    # Total expert count. The HF field name varies:
    #   n_routed_experts   - DeepSeek-V3
    #   num_experts        - Qwen3-MoE, GPT-OSS
    #   num_local_experts  - MiniMax-M2 (Mixtral-style)
    live_total_experts = (
        getattr(original_hf_config, "n_routed_experts", None)
        or getattr(original_hf_config, "num_experts", None)
        or getattr(original_hf_config, "num_local_experts", None)
    )
    _assert_declared("hidden_size", model_hidden_size, getattr(original_hf_config, "hidden_size", None))
    _assert_declared("inter_size", model_inter_size, live_inter_size)
    _assert_declared("num_experts", model_total_experts, live_total_experts)
    rank_print(
        f"Declared model shape confirmed against the checkpoint: hidden_size={model_hidden_size}, "
        f"inter_size={model_inter_size}, total_experts={model_total_experts}"
    )

    # Now apply override to load model with reduced experts.
    # The HF expert-count field varies across model families; override
    # all known names so this works for DeepSeek / Qwen / MiniMax.
    server_args.json_model_override_args = json.dumps(
        {
            "num_hidden_layers": 4,
            "n_routed_experts": num_experts,  # DeepSeek-V3
            "num_experts": num_experts,  # Qwen3-MoE, GPT-OSS
            "num_local_experts": num_experts,  # MiniMax-M2 (Mixtral-style)
        }
    )

    model_runner = load_model_with_dummy_weights(server_args, port_args, tp_rank)

    # MoE submodule attribute name differs across sglang models:
    #   .mlp                 - DeepSeek-V2/V3, Qwen2/3-MoE, GPT-OSS
    #   .block_sparse_moe    - MiniMax-M2 (HF Mixtral-style), Mixtral
    decoder_layer = model_runner.model.model.layers[test_layer]
    moe_layer = None
    for attr in ("mlp", "block_sparse_moe"):
        candidate = getattr(decoder_layer, attr, None)
        # Require an `experts` submodule whose `run_moe_core` is callable: this
        # is what the benchmark actually invokes below, and it filters out
        # non-MoE MLP layers (e.g. DeepSeek's leading dense layers).
        experts = getattr(candidate, "experts", None) if candidate is not None else None
        if experts is not None and callable(getattr(experts, "run_moe_core", None)):
            moe_layer = candidate
            break
    if moe_layer is None:
        raise AttributeError(
            f"Could not find MoE submodule on {type(decoder_layer).__name__}; "
            "tried .mlp / .block_sparse_moe. "
            "Add the attribute name used by this model to the probe list."
        )
    # Supports DeepSeek-V3 and Qwen3 MoE
    if hasattr(moe_layer, "config") and hasattr(moe_layer.config, "n_routed_experts"):
        # DeepSeek-V3 style
        actual_num_experts = moe_layer.config.n_routed_experts
    elif hasattr(moe_layer, "experts") and hasattr(moe_layer.experts, "num_experts"):
        # Qwen3 MoE style - from experts submodule
        actual_num_experts = moe_layer.experts.num_experts
    elif hasattr(moe_layer, "num_experts"):
        # Direct attribute (deepep mode)
        actual_num_experts = moe_layer.num_experts
    else:
        # Fall back to hf_config; probe the same three field names as the
        # pre-load probe at the top of this function, since the loaded
        # config has had all three overridden to the simulated count.
        hf_config = model_runner.model_config.hf_config
        actual_num_experts = (
            getattr(hf_config, "n_routed_experts", None)
            or getattr(hf_config, "num_experts", None)
            or getattr(hf_config, "num_local_experts", None)
        )
        if actual_num_experts is None:
            raise AttributeError(
                f"Could not determine expert count from {type(moe_layer).__name__} "
                "or hf_config; tried .config.n_routed_experts / "
                ".experts.num_experts / .num_experts on the MoE layer and "
                "n_routed_experts / num_experts / num_local_experts on hf_config."
            )

    _assert_declared("topk", model_topk, getattr(getattr(moe_layer.topk, "topk_config", None), "top_k", None))

    rank_print(f"Loaded model with {actual_num_experts} local experts (simulating {model_total_experts} total)")

    server_args.json_model_override_args = original_json_override

    # The declared case decides the EP world; the loaded shard must match it,
    # otherwise the row would claim an EP size that was not benchmarked.
    _assert_declared("num_local_experts", num_experts, actual_num_experts)
    num_local_experts = actual_num_experts  # With ep_size=1, all experts are local
    simulated_ep_size = moe_ep_size
    if num_local_experts * simulated_ep_size != model_total_experts:
        raise MoeEpDeclarationMismatchError(
            f"moe_ep: declared moe_ep_size={simulated_ep_size} x local experts={num_local_experts} "
            f"!= declared num_experts={model_total_experts}"
        )
    rank_print(
        f"Simulating EP size: {simulated_ep_size} "
        f"(num_local_experts={num_local_experts}, total_experts={model_total_experts})"
    )

    prefill_test_cases = get_moe_prefill_test_cases(simulated_ep_size, topk=model_topk, num_experts=model_total_experts)
    rank_print(f"Testing {len(prefill_test_cases)} prefill configurations...")

    # Use deepep_mode="normal" for prefill
    server_args.deepep_mode = "normal"
    benchmark_moe_layer_prefill(
        model_runner,
        server_args,
        port_args,
        num_warmup,
        num_iterations,
        test_layer,
        rank_print,
        server_args.device,
        tp_rank,
        prefill_test_cases,
        moe_layer,
        num_local_experts,
        simulated_ep_size,
        output_path,
        perf_filename,
        model_hidden_size=model_hidden_size,
        model_inter_size=model_inter_size,
        model_total_experts=model_total_experts,
        model_topk=model_topk,
        num_slots=num_slots,
        moe_dtype=moe_dtype,
    )

    decode_test_cases = get_moe_decode_test_cases()
    rank_print(f"Testing {len(decode_test_cases)} decode configurations...")
    # Use deepep_mode="low_latency" for decode
    server_args.deepep_mode = "low_latency"
    benchmark_moe_layer_decode(
        model_runner,
        server_args,
        port_args,
        num_warmup,
        num_iterations,
        test_layer,
        rank_print,
        server_args.device,
        tp_rank,
        decode_test_cases,
        moe_layer,
        num_local_experts,
        simulated_ep_size,
        output_path,
        perf_filename,
        model_hidden_size=model_hidden_size,
        model_inter_size=model_inter_size,
        model_total_experts=model_total_experts,
        model_topk=model_topk,
        num_slots=num_slots,
        moe_dtype=moe_dtype,
    )

    del model_runner, moe_layer
    torch.cuda.empty_cache()

    rank_print(f"\n{'=' * 60}")
    rank_print("BENCHMARK COMPLETED SUCCESSFULLY")
    rank_print(f"{'=' * 60}")


# ============================================================================
# Functions for collect.py framework (trtllm style: direct params, not index)
# ============================================================================


def get_moe_ep_test_cases():
    """Declared large-EP MoE compute cases.

    Expands the ``model_case_values.moe`` rows marked ``wideep: true`` against
    the shared ``cases/base_ops/moe.yaml`` expert-parallel grid — the same
    recipe source the trtllm wideep compute collector uses — instead of
    deriving the sweep from a live HF ``n_routed_experts`` read.

    Filters, all in declared homes:

    * ``wideep: true`` on the model's moe row — the op is declared for the model.
    * ``tp == 1 and ep > 1`` — the large-EP identity of this table; TP-sharded
      MoE is the stock ``moe`` op's business.
    * declared sglang quantization policy must allow :data:`MOE_EP_QUANT_MODE`,
      the only precision this benchmark actually runs.

    One benchmark invocation per ``(model, ep)``: this collector simulates the
    EP world on a single GPU and sweeps token counts and expert distributions
    internally, so the base grid's ``gpu_counts`` and
    ``token_expert_distributions`` axes collapse here. Emitted sorted (D5).

    Returns:
        list[list]: ``[num_local_experts, moe_ep_size, hidden_size, inter_size,
        topk, num_experts, num_slots, moe_dtype, model_name]`` per case — the
        positional argument list ``collect.py`` hands to :func:`run_moe_ep`.
        ``model_name`` keeps the case's model identity: the subprocess loads
        THAT checkpoint (:func:`_case_model_path`), and two models that happen
        to share every shape argument stay distinct tasks.
    """
    try:
        # collect.py puts COLLECTOR_ROOT on sys.path (see the module header).
        from case_generator import (
            get_common_moe_test_cases,
            is_wideep_moe_model,
            moe_model_allows_quantization,
        )
    except ModuleNotFoundError:
        from collector.case_generator import (
            get_common_moe_test_cases,
            is_wideep_moe_model,
            moe_model_allows_quantization,
        )

    recipes = get_common_moe_test_cases(backend="sglang")
    dropped_not_declared = 0
    dropped_not_ep = 0
    dropped_quant = 0
    cases: dict[tuple[str, int], list] = {}

    for recipe in recipes:
        if not is_wideep_moe_model(recipe.model_name):
            dropped_not_declared += 1
            continue
        if recipe.tp != 1 or recipe.ep <= 1:
            dropped_not_ep += 1
            continue
        if not moe_model_allows_quantization("sglang", recipe.model_name, MOE_EP_QUANT_MODE):
            dropped_quant += 1
            continue
        num_experts = int(recipe.num_experts)
        ep_size = int(recipe.ep)
        # get_common_moe_test_cases already enforces num_experts % ep == 0.
        cases.setdefault(
            (recipe.model_name, ep_size),
            [
                num_experts // ep_size,
                ep_size,
                int(recipe.hidden_size),
                int(recipe.inter_size),
                int(recipe.topk),
                num_experts,
                num_experts,  # num_slots: no EPLB redundancy axis on sglang
                MOE_EP_QUANT_MODE,
                recipe.model_name,
            ],
        )

    print(
        f"moe_ep: {len(cases)} cases from {len(recipes)} moe recipes "
        f"(dropped: {dropped_not_declared} not declared wideep, "
        f"{dropped_not_ep} not tp=1/ep>1, {dropped_quant} quant not allowed; "
        f"{len(recipes) - dropped_not_declared - dropped_not_ep - dropped_quant - len(cases)} "
        f"deduplicated onto (model, ep))"
    )
    return [cases[key] for key in sorted(cases)]


def run_moe_benchmark(
    num_local_experts,
    moe_ep_size,
    hidden_size,
    inter_size,
    topk,
    num_experts,
    num_slots,
    moe_dtype,
    model_name,
    gpu_id,
    output_path,
    perf_filename,
):
    """Run one moe_ep case — called in a subprocess with CUDA_VISIBLE_DEVICES set.

    All initialization that must happen after CUDA_VISIBLE_DEVICES is set lives
    here. Every shape argument is the DECLARED value from the case plan, and
    ``model_name`` is the declared model whose checkpoint this case loads.
    """
    # In subprocess, always use cuda:0 since CUDA_VISIBLE_DEVICES isolates the GPU
    torch.cuda.set_device("cuda:0")

    server_port = 30000 + gpu_id * 100
    server_args = ServerArgs(
        model_path=_case_model_path(model_name),
        dtype="auto",
        device="cuda",
        load_format="dummy",
        tp_size=1,
        trust_remote_code=True,
        mem_fraction_static=0.3,
        moe_a2a_backend="deepep",
        moe_runner_backend="deep_gemm",
        deepep_mode="auto",
        ep_size=1,
        node_rank=0,
        host="localhost",
        port=server_port,
        cuda_graph_max_bs=4,
        disable_cuda_graph=True,
    )

    logging.basicConfig(level=getattr(logging, server_args.log_level.upper()), format="%(message)s")
    _set_envs_and_config(server_args)

    # PortArgs.init_new() must be called in subprocess for proper isolation
    port_args = PortArgs.init_new(server_args)

    print(f"\n{'=' * 60}")
    print(
        f"moe_ep: model={model_name}, local_experts={num_local_experts}, moe_ep_size={moe_ep_size}, "
        f"num_experts={num_experts}, moe_dtype={moe_dtype}, GPU={gpu_id}"
    )
    print(f"{'=' * 60}")

    run_moe(
        server_args,
        port_args,
        3,
        10,
        3,
        num_local_experts,
        0,
        output_path,
        perf_filename,
        moe_ep_size=moe_ep_size,
        model_topk=topk,
        model_hidden_size=hidden_size,
        model_inter_size=inter_size,
        model_total_experts=num_experts,
        num_slots=num_slots,
        moe_dtype=moe_dtype,
    )

    torch.cuda.empty_cache()
    print(f"Completed local_experts={num_local_experts} (EP size {moe_ep_size})")


def _run_moe_subprocess(case_args, gpu_id, output_path, perf_filename):
    """Run one moe_ep case in a subprocess with CUDA_VISIBLE_DEVICES isolation."""
    import subprocess
    import sys

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    code = f'''
import sys
sys.path.insert(0, "{THIS_DIR}")
sys.path.insert(0, "{COLLECTOR_ROOT}")
from collect_deepep_moe import run_moe_benchmark
run_moe_benchmark(*{case_args!r}, {gpu_id}, {output_path!r}, {str(perf_filename)!r})
'''

    proc = subprocess.Popen(
        [sys.executable, "-c", code],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=THIS_DIR,
    )

    try:
        stdout, _ = proc.communicate(timeout=600)  # 10 min timeout per moe_ep case
        if stdout:
            print(stdout.decode("utf-8", errors="replace"))
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        raise MoeEpBenchmarkError(f"moe_ep subprocess timed out for case {case_args}") from None

    if proc.returncode != 0:
        raise MoeEpBenchmarkError(f"moe_ep subprocess failed with exit code {proc.returncode} for case {case_args}")


def run_moe_ep(
    num_local_experts,
    moe_ep_size,
    hidden_size,
    inter_size,
    topk,
    num_experts,
    num_slots,
    moe_dtype,
    model_name,
    *,
    perf_filename,
    device="cuda:0",
):
    """Run one declared large-EP MoE compute case.

    Compatible with the collect.py framework — uses a subprocess for GPU
    isolation. ``perf_filename`` is ``PerfFile.MOE_EXPERT_COMPUTE``, bound by collect.py
    from the registry OpEntry.
    """
    device_str = str(device) if not isinstance(device, str) else device
    gpu_id = int(device_str.split(":")[-1]) if ":" in device_str else 0

    print("\n" + "=" * 60)
    print(f"moe_ep: model={model_name}, local_experts={num_local_experts}, moe_ep_size={moe_ep_size}, GPU={gpu_id}")
    print("=" * 60)

    # Resolve output_path from cwd so perf files land in the collector
    # framework's result directory (consistent with collect_moe.py behavior).
    _run_moe_subprocess(
        [
            num_local_experts,
            moe_ep_size,
            hidden_size,
            inter_size,
            topk,
            num_experts,
            num_slots,
            moe_dtype,
            model_name,
        ],
        gpu_id,
        os.getcwd(),
        perf_filename,
    )


if __name__ == "__main__":
    import argparse

    from registry_types import PerfFile

    parser = argparse.ArgumentParser(description="SGLang large-EP DeepEP MoE compute benchmark")
    parser.add_argument("--output-path", default=None, help="Output directory for perf files")
    args = parser.parse_args()

    print(f"Model path: {_get_moe_model_path()}")

    for test_case in get_moe_ep_test_cases():
        run_moe_ep(*test_case, perf_filename=PerfFile.MOE_EXPERT_COMPUTE)

    print("\n" + "=" * 60)
    print("SCRIPT COMPLETED SUCCESSFULLY")
    print("=" * 60)
