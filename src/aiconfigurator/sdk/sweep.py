# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Search / sweep functions for finding feasible worker configurations under SLA.

Two entry points:

- :func:`sweep_agg` — sweep parallel x batch x ctx_tokens for an aggregated
  IFB worker; filter by SLA; return a feasible-candidate DataFrame.
- :func:`sweep_disagg` — sweep prefill_parallel x decode_parallel x
  batches x num_workers with rate matching; return a feasible-candidate DataFrame.

Both functions own the entire search loop themselves and call
``predict.predict_*`` for per-point evaluation.  They replace the
``InferenceSession.find_best_*`` / ``DisaggInferenceSession.find_best_*``
search paths and the ``pareto_analysis.agg_pareto`` / ``disagg_pareto``
proxies.

Note on "Pareto": these functions return the SLA-feasible candidate set,
NOT a Pareto frontier.  The Pareto frontier is a downstream view computed
in :mod:`aiconfigurator.sdk.picking` (``get_pareto_front``) for plotting.
Selecting the best config under SLA is done by sorting + group-by on this
candidate set, not by traversing the frontier.

Output DataFrame schema is ``common.ColumnsAgg`` for agg and
``common.ColumnsDisagg`` for disagg, so downstream picking in
:mod:`aiconfigurator.sdk.picking` works without change.
"""

from __future__ import annotations

import copy
import functools
import logging
from typing import Any

import numpy as np
import pandas as pd

from aiconfigurator.sdk import common, config
from aiconfigurator.sdk.backends.base_backend import BaseBackend
from aiconfigurator.sdk.backends.factory import get_backend
from aiconfigurator.sdk.errors import (
    InsufficientMemoryError,
    KVCacheCapacityError,
    NoFeasibleConfigError,
)
from aiconfigurator.sdk.models import get_model
from aiconfigurator.sdk.perf_database import PerfDatabase
from aiconfigurator.sdk.predict import predict_agg_worker, predict_disagg_worker
from aiconfigurator.sdk.utils import enumerate_ttft_tpot_constraints

logger = logging.getLogger(__name__)

# Empirical degradation factors used in disagg rate matching.  Sourced from
# the same values as :data:`aiconfigurator.sdk.picking._RATE_MATCHING_*`
# (locked in via parity test; do not change without updating picking.py too).
_RATE_MATCH_PREFILL_DEGRADATION = 0.9
_RATE_MATCH_DECODE_DEGRADATION = 0.92

# TTFT pre-correction for queueing under concurrency, sourced from
# picking._AUTOSCALE_TTFT_CORRECTION_FACTOR (locked by integration parity test).
_AUTOSCALE_TTFT_CORRECTION_FACTOR = 1.8

# Disagg search shape constants (mirror inference_session.py module-level).
_DECODE_FILTER_RATIO_MIN = 0.0
_DECODE_FILTER_RATIO_MAX = 1.0
_MAX_DECODE_WORKERS_PER_CATEGORY = 16
_MAX_PREFILL_WORKERS = 32

# Default decode batch-size schedule for disagg worker enumeration.
_DEFAULT_DECODE_BATCH_SCHEDULE: list[int] = (
    list(range(1, 16, 1)) + list(range(16, 32, 2)) + list(range(32, 128, 4)) + list(range(128, 512, 8)) + [512]
)

# Default batch-size schedule used by sweep_agg.  Mirrors the schedule in
# the legacy ``backend.find_best_agg_result_under_constraints`` so results
# stay byte-identical.
_DEFAULT_AGG_BATCH_SCHEDULE: list[int] = (
    list(range(1, 16, 1))
    + list(range(16, 32, 4))
    + list(range(32, 64, 8))
    + list(range(64, 256, 16))
    + list(range(256, 512, 32))
    + list(range(512, 1024, 256))
    + [1024]
)


# ---------------------------------------------------------------------------
# Rate matching (disagg post-processing, inlined for sweep's internal use)
# ---------------------------------------------------------------------------


def _rate_match_dict(
    prefill_summary_dict: dict,
    prefill_num_worker: int,
    decode_summary_dict: dict,
    decode_num_worker: int,
    prefill_degradation: float = _RATE_MATCH_PREFILL_DEGRADATION,
    decode_degradation: float = _RATE_MATCH_DECODE_DEGRADATION,
) -> dict:
    """Compose per-worker prefill+decode metrics into one disagg row.

    Output schema matches ``common.ColumnsDisagg``.  This is the same
    arithmetic as ``picking._build_disagg_summary_dict``; the parity test
    in ``tests/unit/sdk/sweep/test_rate_match_parity.py`` guards against
    drift.  See picking.py for the original implementation.
    """
    p = prefill_summary_dict
    d = decode_summary_dict
    osl = p["osl"]

    seq_s = min(
        p["seq/s"] * prefill_num_worker * prefill_degradation,
        d["seq/s"] * decode_num_worker * decode_degradation,
    )
    prefill_gpus = p["pp"] * p["tp"] * p["dp"]
    decode_gpus = d["pp"] * d["tp"] * d["dp"]
    num_total_gpus = prefill_gpus * prefill_num_worker + decode_gpus * decode_num_worker
    seq_s_gpu = seq_s / num_total_gpus if num_total_gpus > 0 else 0.0
    tokens_s = seq_s * osl
    tokens_s_gpu = tokens_s / num_total_gpus if num_total_gpus > 0 else 0.0
    encoder_latency = float(p.get("encoder_latency", 0.0))
    encoder_memory = float(p.get("encoder_memory", 0.0))
    # static_ctx ttft already includes colocated encoder latency.
    request_latency = p["ttft"] + d["tpot"] * max(osl - 1, 0)

    # Weighted average power across prefill and decode phases.
    ttft = p["ttft"]
    tpot = d["tpot"]
    decode_time = tpot * max(osl - 1, 0)
    total_time = ttft + decode_time
    prefill_power = p.get("power_w", 0.0)
    decode_power = d.get("power_w", 0.0)
    disagg_power_avg = (prefill_power * ttft + decode_power * decode_time) / total_time if total_time > 0 else 0.0

    return {
        "model": p["model"],
        "isl": p["isl"],
        "osl": osl,
        "prefix": p["prefix"],
        "concurrency": d["concurrency"] * decode_num_worker,
        "request_rate": seq_s,
        "(p)bs": p["bs"],
        "(p)global_bs": p["global_bs"],
        "(p)workers": prefill_num_worker,
        "(d)bs": d["bs"],
        "(d)global_bs": d["global_bs"],
        "(d)workers": decode_num_worker,
        "ttft": ttft,
        "tpot": tpot,
        "request_latency": request_latency,
        "encoder_latency": encoder_latency,
        "seq/s": seq_s,
        "seq/s/gpu": seq_s_gpu,
        "tokens/s": tokens_s,
        "tokens/s/gpu": tokens_s_gpu,
        "tokens/s/user": d["tokens/s/user"],
        "(p)seq/s/worker": p["seq/s"],
        "(d)seq/s/worker": d["seq/s"],
        "num_total_gpus": num_total_gpus,
        "(p)tp": p["tp"],
        "(p)pp": p["pp"],
        "(p)dp": p["dp"],
        "(p)moe_tp": p["moe_tp"],
        "(p)moe_ep": p["moe_ep"],
        "(p)cp": p.get("cp", 1),
        "(p)parallel": p["parallel"],
        "(p)gemm": p["gemm"],
        "(p)kvcache": p["kvcache"],
        "(p)fmha": p["fmha"],
        "(p)moe": p["moe"],
        "(p)comm": p["comm"],
        "(p)memory": p["memory"],
        "(p)backend": p.get("backend", ""),
        "(p)version": p.get("version", ""),
        "(p)system": p.get("system", ""),
        "(d)tp": d["tp"],
        "(d)pp": d["pp"],
        "(d)dp": d["dp"],
        "(d)moe_tp": d["moe_tp"],
        "(d)moe_ep": d["moe_ep"],
        "(d)parallel": d["parallel"],
        "(d)gemm": d["gemm"],
        "(d)kvcache": d["kvcache"],
        "(d)fmha": d["fmha"],
        "(d)moe": d["moe"],
        "(d)comm": d["comm"],
        "(d)memory": d["memory"],
        "(d)backend": d.get("backend", ""),
        "(d)version": d.get("version", ""),
        "(d)system": d.get("system", ""),
        # Encoder is colocated with prefill for VL; text-only models leave these
        # visibility fields at zero/empty.
        "(e)workers": 0,
        "(e)tp": 0,
        "(e)pp": 0,
        "(e)parallel": "",
        "(e)memory": encoder_memory,
        "power_w": disagg_power_avg,
        # Queueing (pass-calendar) columns — mirrors
        # picking._build_disagg_summary_dict (see rationale there); guarded
        # by the rate-match parity test.
        "ttft_steady_mean": ttft,
        "ttft_steady_p50": ttft,
        "ttft_steady_p90": ttft,
        "ttft_steady_p99": ttft,
        "ttft_transient_mean": ttft,
        "ttft_transient_max": ttft,
        "itl_mean": tpot,
        "itl_p50": tpot,
        "itl_p99": tpot,
    }


# ---------------------------------------------------------------------------
# Agg sweep
# ---------------------------------------------------------------------------


def _agg_ctx_tokens_list(isl: int, ctx_stride: int, enable_chunked_prefill: bool) -> list[int]:
    """Mirror of ``base_backend._get_ctx_tokens_list_for_agg_sweep``.

    Inlined here so sweep.py does not depend on a private helper on
    BaseBackend.  Algorithm is identical; locked by parity tests.
    """
    max_normal_ctx_tokens = 8192
    max_ctx_tokens_multiple_of_isl = 2
    max_ctx_tokens_small_search_steps = 16
    max_ctx_tokens_search_steps = 8

    max_ctx_tokens = max(max_normal_ctx_tokens, isl * max_ctx_tokens_multiple_of_isl)
    ctx_stride = max(ctx_stride, max_normal_ctx_tokens // max_ctx_tokens_small_search_steps)
    ctx_stride_large = max(
        1024,
        ctx_stride,
        max_ctx_tokens // max_ctx_tokens_search_steps,
    )

    if not enable_chunked_prefill:
        new_ctx_stride = max(isl, ctx_stride)
        new_ctx_stride_large = int(np.ceil(ctx_stride_large / isl) * isl)
        ctx_stride = new_ctx_stride
        ctx_stride_large = new_ctx_stride_large

    ctx_tokens_list: list[int] = []
    ctx_tokens = 0
    while True:
        if ctx_tokens < max_normal_ctx_tokens:
            ctx_tokens += ctx_stride
        else:
            ctx_tokens += ctx_stride_large
        if ctx_tokens > max_ctx_tokens:
            break
        ctx_tokens_list.append(ctx_tokens)

    for i in range(1, max_ctx_tokens_multiple_of_isl + 1):
        v = isl * i
        if v not in ctx_tokens_list:
            ctx_tokens_list.append(v)
    ctx_tokens_list.sort()
    return ctx_tokens_list


def _sweep_one_parallel_agg(
    *,
    model: Any,
    backend: BaseBackend,
    database: PerfDatabase,
    runtime_config: config.RuntimeConfig,
    top_k: int,
    max_batch_size: int,
    ctx_stride: int,
    enable_chunked_prefill: bool,
    free_gpu_memory_fraction: float | None,
    max_seq_len: int | None,
    predictor: Any = None,
) -> tuple[pd.DataFrame, bool, bool]:
    """Sweep batch_size x ctx_tokens for one fixed parallel choice.

    Caller is responsible for constructing ``model`` and ``backend`` and
    reusing them across multiple tpot iterations so the backend's internal
    ``_agg_cache`` survives — recreating the backend per tpot would force
    a full recomputation per tpot, ~80x slowdown for an 80-element tpot
    sweep.

    Returns ``(rows_df, saw_model_fit, saw_memory_fit)``.  Logic faithfully
    reproduces the body of the legacy
    ``backend.find_best_agg_result_under_constraints``; parity is enforced by
    the integration test.
    """
    isl = runtime_config.isl
    osl = runtime_config.osl
    ttft_target = runtime_config.ttft
    tpot_target = runtime_config.tpot

    b_list = [b for b in _DEFAULT_AGG_BATCH_SCHEDULE if b <= max_batch_size]
    ctx_tokens_list = _agg_ctx_tokens_list(isl, ctx_stride, enable_chunked_prefill)

    results_dict_list: list[dict] = []
    results_per_ops_source: list[dict | None] = []
    capped_b: list[int] = []
    saw_model_fit = False
    saw_memory_fit = False

    for b in b_list:
        for ctx_tokens in ctx_tokens_list:
            # batch / ctx_tokens balance guards (legacy semantics)
            if b - np.ceil(ctx_tokens / isl) < 0:
                break
            if b > 1 and (b - np.ceil(ctx_tokens / isl) < 1):
                break

            # Skip equivalent gen_tokens slices to avoid recomputing the same point.
            balance_score = isl * b / ctx_tokens / osl
            if balance_score > 1:
                gen_tokens = b // balance_score
                if gen_tokens > 1 and gen_tokens in capped_b:
                    continue
                capped_b.append(gen_tokens)

            # Deep-copy the full runtime_config (mirrors the disagg path below) so
            # every field is preserved per batch point. Explicit field-by-field
            # construction silently dropped multimodal fields (image_height/width,
            # num_images_per_request, num_image_tokens), zeroing the image encoder
            # workload in agg while disagg stayed correct (NVBug 6401839).
            point_rt = copy.deepcopy(runtime_config)
            point_rt.batch_size = b

            backend_kwargs: dict[str, Any] = {}
            if max_seq_len is not None:
                backend_kwargs["max_seq_len"] = max_seq_len
            if free_gpu_memory_fraction is not None:
                backend_kwargs["free_gpu_memory_fraction"] = free_gpu_memory_fraction

            summary = predict_agg_worker(
                model=model,
                backend=backend,
                database=database,
                runtime_config=point_rt,
                ctx_tokens=ctx_tokens,
                predictor=predictor,
                **backend_kwargs,
            )

            model_oom = summary.check_oom()
            kv_cache_oom = summary.check_kv_cache_oom()
            saw_model_fit |= not model_oom
            saw_memory_fit |= not model_oom and not kv_cache_oom
            if model_oom or kv_cache_oom:
                break  # ctx_tokens monotonic → larger will also OOM
            result_dict = summary.get_result_dict()
            if result_dict and result_dict["tpot"] <= tpot_target and result_dict["ttft"] <= ttft_target:
                results_dict_list.append(result_dict)
                results_per_ops_source.append(summary.get_per_ops_source())

    if not results_dict_list:
        return pd.DataFrame(columns=common.ColumnsAgg), saw_model_fit, saw_memory_fit

    df = pd.DataFrame(results_dict_list, columns=common.ColumnsAgg).round(3)
    df["_per_ops_source"] = results_per_ops_source
    df = df.sort_values(by="seq/s", ascending=False).round(3)
    if top_k > 0:
        df = df.head(top_k)
    return df, saw_model_fit, saw_memory_fit


def sweep_agg(
    *,
    model_path: str,
    runtime_config: config.RuntimeConfig,
    database: PerfDatabase,
    backend_name: str,
    model_config: config.ModelConfig,
    parallel_config_list: list[list[int]] | list[tuple[int, int, int, int, int, int]],
    top_k: int = 10,
    max_batch_size: int = 512,
    ctx_stride: int = 512,
    enable_chunked_prefill: bool = False,
    free_gpu_memory_fraction: float | None = None,
    max_seq_len: int | None = None,
    predictor: Any = None,
) -> pd.DataFrame:
    """Sweep parallel x batch x ctx_tokens for agg; return feasible-candidate DataFrame.

    Replaces ``pareto_analysis.agg_pareto`` -> ``InferenceSession.find_best_agg``
    -> ``backend.find_best_agg_result_under_constraints``.  Output schema is
    ``common.ColumnsAgg``, sorted by ``tokens/s/gpu`` descending.  This is
    the SLA-feasible candidate set; Pareto frontier is a downstream view in
    ``aiconfigurator.sdk.picking`` (used for plotting only — config selection
    works directly on this candidate set).

    Per-tpot sweeping (``runtime_config.tpot`` may be a list) and
    request-latency-derived constraints are handled here as in the legacy
    proxy.

    Args:
        model_path: HuggingFace model path or local path.
        runtime_config: Base runtime config.  ``tpot`` may be a list to
            sweep multiple latency targets; ``request_latency`` triggers
            enumeration of (ttft, tpot) pairs that satisfy it.
        database: Loaded perf database for (system, backend, version).
        backend_name: Backend name ("trtllm", "vllm", "sglang").
        model_config: Base model config; tp/pp/dp/moe_tp/moe_ep are
            overwritten per parallel candidate during the sweep.
        parallel_config_list: List of (tp, pp, dp, moe_tp, moe_ep, cp) tuples
            to enumerate.
        top_k: Per-(parallel, tpot) top-K rows to keep before concat.
        max_batch_size: Upper bound on batch size sweep.
        ctx_stride: Stride for ctx_tokens sweep.
        enable_chunked_prefill: When False, ctx_tokens snaps to multiples of isl.
        free_gpu_memory_fraction: TRT-LLM-only KV cache fraction.
        max_seq_len: TRT-LLM-only per-slot KV cache budget.

    Returns:
        Deduped, sorted feasible-candidate DataFrame with schema ``common.ColumnsAgg``.

    Raises:
        InsufficientMemoryError: When the model does not fit in any config.
        KVCacheCapacityError: When the model fits but the KV cache does not.
        NoFeasibleConfigError: When SLA cannot be satisfied at any point.
        RuntimeError: When no results are produced and a configuration raises.
    """
    results_df = pd.DataFrame(columns=common.ColumnsAgg)
    exceptions: list[Exception] = []
    saw_model_fit = False
    saw_memory_fit = False

    for parallel_config in parallel_config_list:
        tp_size, pp_size, dp_size, moe_tp_size, moe_ep_size, cp_size = parallel_config
        logger.debug(
            "sweep_agg: parallel tp=%s pp=%s dp=%s moe_tp=%s moe_ep=%s cp=%s",
            tp_size,
            pp_size,
            dp_size,
            moe_tp_size,
            moe_ep_size,
            cp_size,
        )
        try:
            point_model_config = copy.deepcopy(model_config)
            point_model_config.tp_size = tp_size
            point_model_config.pp_size = pp_size
            point_model_config.moe_tp_size = moe_tp_size
            point_model_config.moe_ep_size = moe_ep_size
            point_model_config.attention_dp_size = dp_size
            point_model_config.cp_size = cp_size

            # Build backend + model ONCE per parallel choice so the backend's
            # internal _agg_cache survives across the tpot sweep below.
            # Recreating per (parallel, tpot) destroys the cache and causes
            # an ~80x slowdown for a wide tpot list.
            backend = get_backend(backend_name)
            model = get_model(
                model_path=model_path,
                model_config=point_model_config,
                backend_name=backend_name,
            )

            runtime_configs_to_evaluate: list[config.RuntimeConfig] = []
            if runtime_config.request_latency is not None and runtime_config.request_latency > 0:
                pairs = enumerate_ttft_tpot_constraints(
                    runtime_config.osl, runtime_config.request_latency, runtime_config.ttft
                )
                if not pairs:
                    logger.debug(
                        "sweep_agg: no (ttft, tpot) pairs for request_latency=%s",
                        runtime_config.request_latency,
                    )
                    continue
                for ttft_c, tpot_c in pairs:
                    rt = copy.deepcopy(runtime_config)
                    rt.ttft = ttft_c
                    rt.tpot = tpot_c
                    runtime_configs_to_evaluate.append(rt)
            else:
                tpot_list = runtime_config.tpot if isinstance(runtime_config.tpot, list) else [runtime_config.tpot]
                for tpot_v in tpot_list:
                    rt = copy.deepcopy(runtime_config)
                    rt.tpot = tpot_v
                    runtime_configs_to_evaluate.append(rt)

            if not runtime_configs_to_evaluate:
                continue

            for point_rt in runtime_configs_to_evaluate:
                point_df, point_saw_model_fit, point_saw_memory_fit = _sweep_one_parallel_agg(
                    model=model,
                    backend=backend,
                    database=database,
                    runtime_config=point_rt,
                    top_k=top_k,
                    max_batch_size=max_batch_size,
                    ctx_stride=ctx_stride,
                    enable_chunked_prefill=enable_chunked_prefill,
                    free_gpu_memory_fraction=free_gpu_memory_fraction,
                    max_seq_len=max_seq_len,
                    predictor=predictor,
                )
                saw_model_fit |= point_saw_model_fit
                saw_memory_fit |= point_saw_memory_fit
                if len(point_df) == 0:
                    continue
                if len(results_df) == 0:
                    results_df = point_df
                else:
                    results_df = pd.concat([results_df, point_df], axis=0, ignore_index=True)
        except Exception as exc:
            logger.info(
                "sweep_agg: error at tp=%s pp=%s dp=%s moe_tp=%s moe_ep=%s, skipping",
                tp_size,
                pp_size,
                dp_size,
                moe_tp_size,
                moe_ep_size,
            )
            exceptions.append(exc)
            continue

    if not results_df.empty:
        dedupe_cols = [c for c in results_df.columns if c != "_per_ops_source"]
        results_df = results_df.drop_duplicates(subset=dedupe_cols, ignore_index=True)
        results_df = results_df.sort_values(by="tokens/s/gpu", ascending=False).reset_index(drop=True)
        return results_df

    if exceptions:
        raise RuntimeError(
            f"sweep_agg: no results for any parallel configuration. Last exception: {exceptions[-1]}"
        ) from exceptions[-1]
    if not saw_model_fit:
        raise InsufficientMemoryError(
            "sweep_agg: no results — model does not fit in GPU memory for any parallel config. "
            "Try increasing --total-gpus, using a quantized model, or a system with more VRAM per GPU."
        )
    if not saw_memory_fit:
        raise KVCacheCapacityError(
            "sweep_agg: no results — requested batch_size exceeds KV cache capacity for all configs. "
            "Try reducing batch_size, increasing free_gpu_memory_fraction, or a system with more VRAM."
        )
    raise NoFeasibleConfigError(
        "sweep_agg: no parallel configuration met TTFT/TPOT or request-latency constraints. "
        "Try relaxing --ttft / --tpot / --request-latency."
    )


# ---------------------------------------------------------------------------
# Disagg sweep
# ---------------------------------------------------------------------------


def _get_disagg_worker_candidates(
    *,
    model_path: str,
    model_config: config.ModelConfig,
    parallel_config_list: list[tuple[int, int, int, int, int, int]] | list[list[int]],
    b_list: list[int] | range,
    runtime_config: config.RuntimeConfig,
    role: str,
    database: PerfDatabase,
    backend_name: str,
    latency_correction: float,
    predictor: Any = None,
) -> pd.DataFrame:
    """Enumerate (parallel, batch_size) worker candidates for a disagg role.

    Returns a DataFrame in ``common.ColumnsStatic`` schema, one row per
    (parallel, batch_size) that fits in memory.  Replaces the body of
    ``DisaggInferenceSession.get_worker_candidates``.
    """
    backend = get_backend(backend_name)
    summary_df = pd.DataFrame(columns=common.ColumnsStatic)
    exceptions: list[Exception] = []
    all_configs_oom = True

    for parallel_config in parallel_config_list:
        tp_size, pp_size, dp_size, moe_tp_size, moe_ep_size, cp_size = parallel_config
        logger.debug(
            "sweep_disagg/%s: candidate parallel tp=%s pp=%s dp=%s moe_tp=%s moe_ep=%s cp=%s",
            role,
            tp_size,
            pp_size,
            dp_size,
            moe_tp_size,
            moe_ep_size,
            cp_size,
        )
        try:
            point_mc = copy.deepcopy(model_config)
            point_mc.tp_size = tp_size
            point_mc.pp_size = pp_size
            point_mc.moe_tp_size = moe_tp_size
            point_mc.moe_ep_size = moe_ep_size
            point_mc.attention_dp_size = dp_size
            point_mc.cp_size = cp_size

            model = get_model(model_path=model_path, model_config=point_mc, backend_name=backend_name)

            for b in b_list:
                point_rt = copy.deepcopy(runtime_config)
                point_rt.batch_size = b
                summary = predict_disagg_worker(
                    model=model,
                    backend=backend,
                    database=database,
                    runtime_config=point_rt,
                    role=role,  # type: ignore[arg-type]
                    latency_correction=latency_correction,
                    predictor=predictor,
                )
                if not summary.check_oom():
                    all_configs_oom = False
                    summary_df = pd.concat(
                        [summary_df, summary.get_summary_df()],
                        axis=0,
                        ignore_index=True,
                    )
                else:
                    break
        except Exception as e:
            logger.warning(
                "sweep_disagg/%s: error at parallel tp=%s pp=%s dp=%s moe_tp=%s moe_ep=%s; skipping. err=%s",
                role,
                tp_size,
                pp_size,
                dp_size,
                moe_tp_size,
                moe_ep_size,
                e,
            )
            exceptions.append(e)
            continue

    if summary_df.empty:
        if exceptions:
            raise RuntimeError(
                f"sweep_disagg/{role}: no results for any parallel config. Last exception: {exceptions[-1]}"
            ) from exceptions[-1]
        if all_configs_oom:
            raise InsufficientMemoryError(
                f"sweep_disagg/{role}: no results — model does not fit in GPU memory for any parallel config. "
                "Try increasing GPU budget, using a quantized model, or a system with more VRAM per GPU."
            )
        raise NoFeasibleConfigError(
            f"sweep_disagg/{role}: no parallel configuration met TTFT/TPOT or request-latency constraints."
        )
    return summary_df


def _find_best_disagg_under_constraint(
    *,
    ttft_target: float,
    tpot_target: float,
    prefill_summary_df: pd.DataFrame,
    decode_summary_df: pd.DataFrame,
    return_top_k: int,
    num_gpu_set: set[int],
    prefill_num_worker_list: list[int],
    decode_num_worker_list: list[int],
    max_prefill_gpus: int | None,
    max_decode_gpus: int | None,
    require_same_tp: bool,
    prefill_degradation: float,
    decode_degradation: float,
    match_workers: Any,
    autoscale_ttft_correction_factor: float = _AUTOSCALE_TTFT_CORRECTION_FACTOR,
) -> pd.DataFrame | None:
    """For one (ttft, tpot) pair, filter + rate-match + pick best per decode parallel.

    Mirrors ``_find_best_result_under_constraints`` in
    DisaggInferenceSession.find_best_disagg_result_under_constraints.

    ``match_workers`` is supplied by the caller (``sweep_disagg``) so its
    ``lru_cache`` is shared across all (ttft, tpot) pairs -- its result is
    independent of the target, so a per-pair cache would recompute identical
    matches.
    """

    p_corrected = prefill_summary_df.assign(ttft=prefill_summary_df["ttft"] * autoscale_ttft_correction_factor)
    p_candidates = p_corrected[p_corrected["ttft"] < ttft_target]
    if len(p_candidates) == 0:
        logger.debug("sweep_disagg: no prefill candidates meet ttft<%sms", ttft_target)
        return None
    p_candidates = (
        p_candidates.sort_values(by=["seq/s/gpu", "global_bs"], ascending=[False, True])
        .reset_index(drop=True)
        .head(_MAX_PREFILL_WORKERS)
    )

    d_candidates = decode_summary_df[
        (decode_summary_df["tpot"] < tpot_target * _DECODE_FILTER_RATIO_MAX)
        & (decode_summary_df["tpot"] > tpot_target * _DECODE_FILTER_RATIO_MIN)
    ].copy()
    if len(d_candidates) == 0:
        logger.debug("sweep_disagg: no decode candidates meet tpot<%sms", tpot_target)
        return None

    all_category_results: list[dict] = []
    p_records = p_candidates.to_dict("records")

    for parallel_value, parallel_group in d_candidates.groupby("parallel"):
        group_sorted = (
            parallel_group.sort_values(by=["seq/s/gpu"], ascending=[False])
            .reset_index(drop=True)
            .head(_MAX_DECODE_WORKERS_PER_CATEGORY)
        )
        decode_records = group_sorted.to_dict("records")
        category_results: list[dict] = []
        for d_worker in decode_records:
            d_throughput = float(d_worker["seq/s"])
            d_gpus = d_worker["num_total_gpus"]
            for p_worker in p_records:
                if require_same_tp and p_worker["tp"] != d_worker["tp"]:
                    continue
                p_throughput = float(p_worker["seq/s"])
                p_gpus = p_worker["num_total_gpus"]
                p_num, d_num = match_workers(
                    prefill_throughput=p_throughput,
                    prefill_gpus=p_gpus,
                    decode_throughput=d_throughput,
                    decode_gpus=d_gpus,
                    prefill_deg=prefill_degradation,
                    decode_deg=decode_degradation,
                )
                if p_num == -1 or d_num == -1:
                    continue
                disagg_dict = _rate_match_dict(
                    p_worker,
                    p_num,
                    d_worker,
                    d_num,
                    prefill_degradation=prefill_degradation,
                    decode_degradation=decode_degradation,
                )
                category_results.append(disagg_dict)
        if category_results:
            best = max(category_results, key=lambda x: (x["tokens/s/gpu"], -x["num_total_gpus"]))
            all_category_results.append(best)
        else:
            logger.debug("sweep_disagg: no matched result for decode parallel %s", parallel_value)

    if not all_category_results:
        logger.debug("sweep_disagg: no disagg summary after constraints")
        return None

    df = pd.DataFrame(all_category_results, columns=common.ColumnsDisagg).round(3)
    df = df.sort_values(by=["tokens/s/gpu"], ascending=[False]).head(return_top_k).reset_index(drop=True)
    return df


def sweep_disagg(
    *,
    model_path: str,
    runtime_config: config.RuntimeConfig,
    prefill_database: PerfDatabase,
    prefill_backend_name: str,
    prefill_model_config: config.ModelConfig,
    prefill_parallel_config_list: list[tuple[int, int, int, int, int, int]] | list[list[int]],
    prefill_latency_correction: float,
    decode_database: PerfDatabase,
    decode_backend_name: str,
    decode_model_config: config.ModelConfig,
    decode_parallel_config_list: list[tuple[int, int, int, int, int, int]] | list[list[int]],
    decode_latency_correction: float,
    prefill_max_num_tokens: int = 16384,
    decode_max_num_tokens: int = 512,
    prefill_num_worker_list: list[int] | None = None,
    decode_num_worker_list: list[int] | None = None,
    num_gpu_list: list[int] | None = None,
    max_prefill_gpus: int | None = None,
    max_decode_gpus: int | None = None,
    require_same_tp: bool = False,
    autoscale: bool = False,
    target_tpot: float | None = None,
    rate_matching_prefill_degradation: float | None = None,
    rate_matching_decode_degradation: float | None = None,
    autoscale_ttft_correction_factor: float | None = None,
    predictor: Any = None,
) -> pd.DataFrame:
    """Sweep prefill_parallel x decode_parallel x batches x workers with rate matching.

    Replaces ``pareto_analysis.disagg_pareto`` ->
    ``DisaggInferenceSession.find_best_disagg_result_under_constraints``.
    Output schema is ``common.ColumnsDisagg``, sorted by ``tokens/s/gpu``.

    The two databases / backends are accepted independently to support
    hetero-disagg (prefill and decode on different systems).

    Returns:
        DataFrame (possibly empty) with schema ``common.ColumnsDisagg``.

    Raises:
        ValueError: invalid GPU bounds.
        RuntimeError: no feasible worker candidates.
        NoFeasibleConfigError: no point satisfies the SLA.
    """
    if max_prefill_gpus is not None and max_prefill_gpus <= 0:
        raise ValueError(f"max_prefill_gpus must be > 0, got {max_prefill_gpus}")
    if max_decode_gpus is not None and max_decode_gpus <= 0:
        raise ValueError(f"max_decode_gpus must be > 0, got {max_decode_gpus}")

    p_deg = (
        rate_matching_prefill_degradation
        if rate_matching_prefill_degradation is not None
        else _RATE_MATCH_PREFILL_DEGRADATION
    )
    d_deg = (
        rate_matching_decode_degradation
        if rate_matching_decode_degradation is not None
        else _RATE_MATCH_DECODE_DEGRADATION
    )
    ttft_corr = (
        autoscale_ttft_correction_factor
        if autoscale_ttft_correction_factor is not None
        else _AUTOSCALE_TTFT_CORRECTION_FACTOR
    )
    p_num_workers = prefill_num_worker_list or []
    d_num_workers = decode_num_worker_list or []
    if not p_num_workers or not d_num_workers:
        raise ValueError(
            "sweep_disagg requires non-empty prefill_num_worker_list and decode_num_worker_list. "
            "Empty lists silently produce zero results because the rate-matching inner loop "
            "iterates over them.  Pass an explicit range (e.g. list(range(1, 33))) or omit the "
            "argument entirely to let Task fill in defaults."
        )
    num_gpu_set: set[int] = set(num_gpu_list) if num_gpu_list else set()

    if decode_max_num_tokens < 1:
        logger.warning("decode_max_num_tokens < 1, clamping to 1")
        decode_max_num_tokens = 1
    if decode_max_num_tokens > max(_DEFAULT_DECODE_BATCH_SCHEDULE):
        decode_batch_range: list[int] | range = _DEFAULT_DECODE_BATCH_SCHEDULE + [decode_max_num_tokens]
    else:
        decode_batch_range = [b for b in _DEFAULT_DECODE_BATCH_SCHEDULE if b <= decode_max_num_tokens]

    if prefill_max_num_tokens < runtime_config.isl:
        logger.warning("prefill_max_num_tokens < runtime_config.isl, clamping to isl")
        prefill_max_num_tokens = runtime_config.isl
    max_prefill_batch_size = prefill_max_num_tokens // runtime_config.isl
    prefill_batch_range = range(1, max_prefill_batch_size + 1)

    prefill_summary_df = _get_disagg_worker_candidates(
        model_path=model_path,
        model_config=prefill_model_config,
        parallel_config_list=prefill_parallel_config_list,
        b_list=prefill_batch_range,
        runtime_config=runtime_config,
        role="prefill",
        database=prefill_database,
        backend_name=prefill_backend_name,
        latency_correction=prefill_latency_correction,
        predictor=predictor,
    )
    decode_summary_df = _get_disagg_worker_candidates(
        model_path=model_path,
        model_config=decode_model_config,
        parallel_config_list=decode_parallel_config_list,
        b_list=decode_batch_range,
        runtime_config=runtime_config,
        role="decode",
        database=decode_database,
        backend_name=decode_backend_name,
        latency_correction=decode_latency_correction,
        predictor=predictor,
    )

    if len(prefill_summary_df) == 0 or len(decode_summary_df) == 0:
        logger.debug("sweep_disagg: no prefill or decode worker candidates")
        return pd.DataFrame(columns=common.ColumnsDisagg)

    if autoscale:
        from aiconfigurator.sdk.picking import pick_autoscale

        target_ttft_v = runtime_config.ttft
        if target_tpot is None:
            tpot_values = runtime_config.tpot if isinstance(runtime_config.tpot, list) else [runtime_config.tpot]
            target_tpot_v = max(tpot_values)
        else:
            target_tpot_v = target_tpot
        result = pick_autoscale(
            prefill_df=prefill_summary_df,
            decode_df=decode_summary_df,
            target_ttft=target_ttft_v,
            target_tpot=target_tpot_v,
            top_n=5,
            ttft_correction_factor=ttft_corr,
        )
        df = result["best_config_df"]
        if df is None or df.empty:
            return pd.DataFrame(columns=common.ColumnsDisagg)
        return df

    constraint_pairs: list[tuple[float, float]] = []
    if runtime_config.request_latency is not None and runtime_config.request_latency > 0:
        constraint_pairs = enumerate_ttft_tpot_constraints(
            runtime_config.osl,
            runtime_config.request_latency,
            runtime_config.ttft,
        )
        if not constraint_pairs:
            logger.debug(
                "sweep_disagg: no (ttft, tpot) pairs for request_latency=%s",
                runtime_config.request_latency,
            )
    else:
        tpot_values = runtime_config.tpot if isinstance(runtime_config.tpot, list) else [runtime_config.tpot]
        constraint_pairs = [(runtime_config.ttft, tpot) for tpot in tpot_values]

    # Worker-count rate matching depends only on per-worker throughput/GPUs and the
    # (constant) worker-count lists + GPU budget -- NOT on the (ttft, tpot) target.
    # Define it once here so the lru_cache is shared across every constraint pair.
    # Nesting it inside _find_best_disagg_under_constraint rebuilt the cache per pair
    # and recomputed identical matches (the dominant cost of the disagg sweep).
    @functools.lru_cache(maxsize=8192)
    def _match_workers(
        prefill_throughput: float,
        prefill_gpus: int,
        decode_throughput: float,
        decode_gpus: int,
        prefill_deg: float,
        decode_deg: float,
    ) -> tuple[int, int]:
        prefill_opt, decode_opt = -1, -1
        throughput_per_gpu_max = 0.0
        for d_num in d_num_workers:
            for p_num in p_num_workers:
                num_gpu = prefill_gpus * p_num + decode_gpus * d_num
                if num_gpu_set and num_gpu not in num_gpu_set:
                    continue
                if max_prefill_gpus is not None and max_decode_gpus is not None:
                    if prefill_gpus * p_num > max_prefill_gpus:
                        continue
                    if decode_gpus * d_num > max_decode_gpus:
                        continue
                p_corrected = prefill_throughput * p_num * prefill_deg
                d_corrected = decode_throughput * d_num * decode_deg
                tpg = min(p_corrected, d_corrected) / num_gpu
                if tpg > throughput_per_gpu_max:
                    throughput_per_gpu_max = tpg
                    prefill_opt, decode_opt = p_num, d_num
        return prefill_opt, decode_opt

    disagg_df = pd.DataFrame(columns=common.ColumnsDisagg)
    for ttft_c, tpot_c in constraint_pairs:
        logger.debug("sweep_disagg: finding best for ttft=%sms tpot=%sms", ttft_c, tpot_c)
        partial = _find_best_disagg_under_constraint(
            ttft_target=ttft_c,
            tpot_target=tpot_c,
            prefill_summary_df=prefill_summary_df,
            decode_summary_df=decode_summary_df,
            return_top_k=5,
            num_gpu_set=num_gpu_set,
            prefill_num_worker_list=p_num_workers,
            decode_num_worker_list=d_num_workers,
            max_prefill_gpus=max_prefill_gpus,
            max_decode_gpus=max_decode_gpus,
            require_same_tp=require_same_tp,
            prefill_degradation=p_deg,
            decode_degradation=d_deg,
            match_workers=_match_workers,
            autoscale_ttft_correction_factor=ttft_corr,
        )
        if partial is not None:
            disagg_df = pd.concat([disagg_df, partial], axis=0, ignore_index=True)

    if len(disagg_df) == 0:
        logger.debug("sweep_disagg: no disagg result satisfies any constraint")
        return pd.DataFrame(columns=common.ColumnsDisagg)

    return (
        disagg_df.drop_duplicates(ignore_index=True)
        .sort_values(by="tokens/s/gpu", ascending=False)
        .reset_index(drop=True)
    )
