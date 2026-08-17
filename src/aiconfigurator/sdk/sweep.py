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
import dataclasses
import functools
import logging
import math
from collections.abc import Callable
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
    PerfDataNotAvailableError,
)
from aiconfigurator.sdk.models import get_model
from aiconfigurator.sdk.models.vit_ops import EncoderOnlyModel, build_encoder_ops
from aiconfigurator.sdk.perf_database import PerfDatabase, has_perf_data_not_available_cause
from aiconfigurator.sdk.performance_result import MOE_COMM_FALLBACKS_COLUMN, merge_moe_comm_fallbacks
from aiconfigurator.sdk.picking import parallel_dim, worker_gpus
from aiconfigurator.sdk.predict import predict_agg_worker, predict_disagg_worker
from aiconfigurator.sdk.speculative import SpeculativeDecodingProfile
from aiconfigurator.sdk.utils import enumerate_ttft_tpot_constraints, get_model_config_from_model_path

logger = logging.getLogger(__name__)

# Empirical degradation factors used in disagg rate matching.  Sourced from
# the same values as :data:`aiconfigurator.sdk.picking._RATE_MATCHING_*`
# (locked in via parity test; do not change without updating picking.py too).
_RATE_MATCH_PREFILL_DEGRADATION = 0.9
_RATE_MATCH_DECODE_DEGRADATION = 0.92
# EPD: the encode pool takes the same class of loss as prefill.
_RATE_MATCH_ENCODER_DEGRADATION = 0.9

# TTFT pre-correction for queueing under concurrency, sourced from
# picking._AUTOSCALE_TTFT_CORRECTION_FACTOR (locked by integration parity test).
_AUTOSCALE_TTFT_CORRECTION_FACTOR = 1.8

# Disagg search shape constants (mirror inference_session.py module-level).
_DECODE_FILTER_RATIO_MIN = 0.0
_DECODE_FILTER_RATIO_MAX = 1.0
_MAX_DECODE_WORKERS_PER_CATEGORY = 16
_MAX_PREFILL_WORKERS = 32
# EPD encode-pool size sweep bound, mirroring the P/D worker-list bounds.
_MAX_ENCODE_WORKERS = 32

# Default decode batch-size schedule for disagg worker enumeration.
_DEFAULT_DECODE_BATCH_SCHEDULE: list[int] = (
    list(range(1, 16, 1)) + list(range(16, 32, 2)) + list(range(32, 128, 4)) + list(range(128, 512, 8)) + [512]
)

# Default EPD encode-worker search space; batches are capped at SGLang's
# default (SGLANG_ENCODER_MAX_BATCH_SIZE = 8) — explicit candidates above
# the cap would claim throughput the reference deployment cannot form.
_DEFAULT_ENCODER_TP_LIST: list[int] = [1, 2, 4, 8]
_DEFAULT_ENCODER_BATCH_SCHEDULE: list[int] = [1, 2, 4, 8]
_MAX_ENCODER_BATCH = 8

# E+agg: agg-worker replicas explored per rate-matched cell (mirrors the
# disagg worker lists, range(1, 33)).
_MAX_AGG_WORKERS_EPD = 32

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


def _preferred_sweep_exception(exceptions: list[Exception]) -> Exception:
    """Keep a structured data miss visible across skipped parallel configs.

    Sweeps deliberately skip invalid parallel choices. If no choice produces a
    result, the last skipped error can therefore be an unrelated TP-divisibility
    assertion even though valid choices reached a structured performance-data
    miss. Prefer that miss so callers such as the support-matrix generator can
    make the intended SILICON-to-HYBRID decision. Otherwise retain the legacy
    last-exception behavior.
    """
    return next(
        (error for error in exceptions if has_perf_data_not_available_cause(error)),
        exceptions[-1],
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
    prefill_gpus = worker_gpus(p)
    decode_gpus = worker_gpus(d)
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
        "(p)cp": parallel_dim(p.get("cp")),
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
        # Colocated-encoder visibility fields; EPD overwrites them via
        # _overlay_encoder_stage.
        "(e)workers": 0,
        "(e)tp": 0,
        "(e)pp": 0,
        "(e)bs": 0,
        "(e)parallel": "",
        "(e)memory": encoder_memory,
        "power_w": disagg_power_avg,
        MOE_COMM_FALLBACKS_COLUMN: merge_moe_comm_fallbacks(
            p.get(MOE_COMM_FALLBACKS_COLUMN),
            d.get(MOE_COMM_FALLBACKS_COLUMN),
        ),
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
    speculative_profile: SpeculativeDecodingProfile | None = None,
) -> tuple[pd.DataFrame, bool, bool]:
    """Sweep batch_size x ctx_tokens for one fixed parallel choice.

    Caller is responsible for constructing ``model`` and ``backend`` and
    reusing them across multiple tpot iterations so the backend's internal
    ``_agg_cache`` survives — recreating the backend per tpot would force
    a full recomputation per tpot, ~80x slowdown for an 80-element tpot
    sweep.

    Returns ``(rows_df, saw_model_fit, saw_memory_fit, perf_misses)``.  Logic faithfully
    reproduces the body of the legacy
    ``backend.find_best_agg_result_under_constraints``; parity is enforced by
    the integration test.
    """
    # Vision tokens occupy the prefill context, so the ctx-token grid and the
    # batch/ctx feasibility guards below run on the effective ISL -- same as
    # the legacy find_best_agg_result_under_constraints (its isl_eff) and as
    # run_agg's internal accounting.
    isl = runtime_config.isl + BaseBackend._visual_context_tokens(model, runtime_config)
    osl = runtime_config.osl
    ttft_target = runtime_config.ttft
    tpot_target = runtime_config.tpot

    b_list = [b for b in _DEFAULT_AGG_BATCH_SCHEDULE if b <= max_batch_size]
    ctx_tokens_list = _agg_ctx_tokens_list(isl, ctx_stride, enable_chunked_prefill)

    # The capped-gen dedup below assumes the non-speculative schedule
    # (decode_iterations == osl). With speculative progress the boundary is
    # fractional and backend schedulers (e.g. vLLM's b - ceil(ctx/isl) decode
    # requests) still distinguish batches this generic key would merge, so the
    # heuristic is disabled and every guard-passing point is evaluated.
    _progress = speculative_profile.tokens_per_iteration if speculative_profile else 1.0
    dedup_gen_slices = _progress == 1.0

    results_dict_list: list[dict] = []
    results_per_ops_source: list[dict | None] = []
    results_moe_comm_fallbacks: list[tuple] = []
    capped_b: list[int] = []
    saw_model_fit = False
    saw_memory_fit = False
    perf_misses = 0

    for b in b_list:
        for ctx_tokens in ctx_tokens_list:
            # batch / ctx_tokens balance guards (legacy semantics)
            if b - np.ceil(ctx_tokens / isl) < 0:
                break
            if b > 1 and (b - np.ceil(ctx_tokens / isl) < 1):
                break

            # Skip equivalent gen_tokens slices to avoid recomputing the same point.
            if dedup_gen_slices:
                balance_score = isl * b / ctx_tokens / osl
                if balance_score > 1:
                    gen_tokens = b // balance_score
                    if gen_tokens > 1 and gen_tokens in capped_b:
                        continue
                    capped_b.append(gen_tokens)

            # dataclasses.replace shallow-copies all fields (including multimodal
            # fields like image_height/width that field-by-field construction
            # silently dropped -- NVBug 6401839) and overrides only the named
            # ones. Safe because the sweep never mutates list-valued fields.
            point_rt = dataclasses.replace(runtime_config, batch_size=b)

            backend_kwargs: dict[str, Any] = {}
            if max_seq_len is not None:
                backend_kwargs["max_seq_len"] = max_seq_len
            if free_gpu_memory_fraction is not None:
                backend_kwargs["free_gpu_memory_fraction"] = free_gpu_memory_fraction

            try:
                summary = predict_agg_worker(
                    model=model,
                    backend=backend,
                    database=database,
                    runtime_config=point_rt,
                    ctx_tokens=ctx_tokens,
                    predictor=predictor,
                    speculative_profile=speculative_profile,
                    **backend_kwargs,
                )
            except PerfDataNotAvailableError:
                # This batch/ctx point is unanswerable (e.g. an FPM query
                # beyond the collected domain, which never extrapolates);
                # the point is infeasible, not the whole parallel config.
                perf_misses += 1
                continue

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
                results_moe_comm_fallbacks.append(merge_moe_comm_fallbacks(summary.get_moe_comm_fallbacks()))

    if not results_dict_list:
        return pd.DataFrame(columns=common.ColumnsAgg), saw_model_fit, saw_memory_fit, perf_misses

    df = pd.DataFrame(results_dict_list, columns=common.ColumnsAgg).round(3)
    df["_per_ops_source"] = results_per_ops_source
    df[MOE_COMM_FALLBACKS_COLUMN] = results_moe_comm_fallbacks
    df = df.sort_values(by="seq/s", ascending=False).round(3)
    if top_k > 0:
        df = df.head(top_k)
    return df, saw_model_fit, saw_memory_fit, perf_misses


def _point_model_config(model_config, parallel_config) -> config.ModelConfig:
    """The ModelConfig for one parallel point.

    ``model_config`` is either a template (deep-copied, as always) or a builder
    called with the point's ``(tp, pp, dp, moe_tp, moe_ep, cp)`` tuple. The
    builder form exists because some ModelConfig fields are decided per
    parallel config rather than per task: ``moe_comm_backend`` -- large EP is
    offered for exactly the tuples the perf data covers (``Task.
    _resolve_moe_comm_backend``) -- is the first of them. Either way the caller
    then overwrites the parallelism fields on the returned object.
    """
    if callable(model_config):
        return model_config(tuple(parallel_config))
    return copy.deepcopy(model_config)


def _language_only_config(model_config):
    """Language-only variant of a template OR per-point builder (both forms
    ``_point_model_config`` accepts)."""
    if callable(model_config):
        return lambda parallel: dataclasses.replace(model_config(parallel), language_only=True)
    return dataclasses.replace(model_config, language_only=True)


def sweep_agg(
    *,
    model_path: str,
    runtime_config: config.RuntimeConfig,
    database: PerfDatabase,
    backend_name: str,
    model_config: config.ModelConfig | Callable[..., config.ModelConfig],
    parallel_config_list: list[list[int]] | list[tuple[int, int, int, int, int, int]],
    top_k: int = 10,
    max_batch_size: int = 512,
    ctx_stride: int = 512,
    enable_chunked_prefill: bool = False,
    free_gpu_memory_fraction: float | None = None,
    max_seq_len: int | None = None,
    enable_epd: bool = False,
    encoder_tp_list: list[int] | None = None,
    encoder_batch_list: list[int] | None = None,
    max_encoder_workers: int | None = None,
    encoder_latency_correction: float = 1.0,
    encoder_database: PerfDatabase | None = None,
    rate_matching_encoder_degradation: float | None = None,
    num_gpu_list: list[int] | None = None,
    predictor: Any = None,
    speculative_profile: SpeculativeDecodingProfile | None = None,
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

    ``enable_epd`` switches VL agg into E+agg: the vision encoder runs on
    dedicated encode workers while the agg workers become language-only, and
    each row is a rate-matched cell of ``(a)workers`` agg plus ``(e)workers``
    encode workers with the encode batch latency added to TTFT — rows then
    carry ``common.ColumnsAggEpd`` instead of ``common.ColumnsAgg``.
    ``num_gpu_list`` is the allowed per-cell GPU counts (as in
    ``sweep_disagg``); it is unused outside EPD.

    Args:
        model_path: HuggingFace model path or local path.
        runtime_config: Base runtime config.  ``tpot`` may be a list to
            sweep multiple latency targets; ``request_latency`` triggers
            enumeration of (ttft, tpot) pairs that satisfy it.
        database: Loaded perf database for (system, backend, version).
        backend_name: Backend name ("trtllm", "vllm", "sglang").
        model_config: Base model config; tp/pp/dp/moe_tp/moe_ep are
            overwritten per parallel candidate during the sweep. May instead be
            a per-point builder taking the parallel tuple (see
            ``_point_model_config``).
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
    perf_misses = 0

    e_deg = (
        rate_matching_encoder_degradation
        if rate_matching_encoder_degradation is not None
        else _RATE_MATCH_ENCODER_DEGRADATION
    )
    if not (math.isfinite(e_deg) and e_deg > 0):
        raise ValueError(f"rate_matching_encoder_degradation must be a positive finite number, got {e_deg!r}.")
    encoder_candidates: list[dict] | None = None
    if enable_epd:
        # E+agg: enumerate the encode pool once; hetero encoder may use its
        # own database (defaults to the agg side).
        encoder_candidates = _get_encoder_worker_candidates(
            model_path=model_path,
            tp_list=encoder_tp_list,
            b_list=encoder_batch_list,
            runtime_config=runtime_config,
            database=encoder_database or database,
            backend_name=backend_name,
            latency_correction=encoder_latency_correction,
        )
        model_config = _language_only_config(model_config)
    # Per-cell GPU budget for the E+agg rate matching (sweep_disagg semantics).
    epd_num_gpu_set: set[int] = set(num_gpu_list) if num_gpu_list else set()

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
            point_model_config = dataclasses.replace(
                _point_model_config(model_config, parallel_config),
                tp_size=tp_size,
                pp_size=pp_size,
                moe_tp_size=moe_tp_size,
                moe_ep_size=moe_ep_size,
                attention_dp_size=dp_size,
                cp_size=cp_size,
            )

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
                    runtime_configs_to_evaluate.append(dataclasses.replace(runtime_config, ttft=ttft_c, tpot=tpot_c))
            else:
                tpot_list = runtime_config.tpot if isinstance(runtime_config.tpot, list) else [runtime_config.tpot]
                for tpot_v in tpot_list:
                    runtime_configs_to_evaluate.append(dataclasses.replace(runtime_config, tpot=tpot_v))

            if not runtime_configs_to_evaluate:
                continue

            for point_rt in runtime_configs_to_evaluate:
                point_df, point_saw_model_fit, point_saw_memory_fit, point_perf_misses = _sweep_one_parallel_agg(
                    model=model,
                    backend=backend,
                    database=database,
                    runtime_config=point_rt,
                    # EPD defers the top_k cut to the encoder pairing (a cut
                    # here could shadow the only pairable rows).
                    top_k=0 if encoder_candidates is not None else top_k,
                    max_batch_size=max_batch_size,
                    ctx_stride=ctx_stride,
                    enable_chunked_prefill=enable_chunked_prefill,
                    free_gpu_memory_fraction=free_gpu_memory_fraction,
                    max_seq_len=max_seq_len,
                    predictor=predictor,
                    speculative_profile=speculative_profile,
                )
                saw_model_fit |= point_saw_model_fit
                saw_memory_fit |= point_saw_memory_fit
                perf_misses += point_perf_misses
                if encoder_candidates is not None and len(point_df) > 0:
                    # The per-point ttft filter ran on the language-only ttft;
                    # pair exactly here.
                    point_df = _rate_match_agg_epd(
                        point_df,
                        encoder_candidates,
                        ttft_target=point_rt.ttft,
                        num_gpu_set=epd_num_gpu_set,
                        top_k=top_k,
                        max_encoder_workers=max_encoder_workers,
                        encoder_degradation=e_deg,
                    )
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
        dedupe_cols = [c for c in results_df.columns if c not in {"_per_ops_source", MOE_COMM_FALLBACKS_COLUMN}]
        results_df = results_df.drop_duplicates(subset=dedupe_cols, ignore_index=True)
        results_df = results_df.sort_values(by="tokens/s/gpu", ascending=False).reset_index(drop=True)
        return results_df

    if exceptions:
        terminal_error = _preferred_sweep_exception(exceptions)
        raise RuntimeError(
            f"sweep_agg: no results for any parallel configuration. Selected exception: {terminal_error}"
        ) from terminal_error
    if not saw_model_fit:
        if perf_misses:
            raise NoFeasibleConfigError(
                f"sweep_agg: no results — {perf_misses} batch point(s) had no answerable perf data "
                "(e.g. FPM queries outside the collected domain, or no collected cell matches the "
                "model identity/quant modes). Check the collected cells against the resolved quant "
                "configuration, or use forward_model='op_level'."
            )
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
    model_config: config.ModelConfig | Callable[..., config.ModelConfig],
    parallel_config_list: list[tuple[int, int, int, int, int, int]] | list[list[int]],
    b_list: list[int] | range,
    runtime_config: config.RuntimeConfig,
    role: str,
    database: PerfDatabase,
    backend_name: str,
    latency_correction: float,
    predictor: Any = None,
    speculative_profile: SpeculativeDecodingProfile | None = None,
    free_gpu_memory_fraction: float | None = None,
) -> pd.DataFrame:
    """Enumerate (parallel, batch_size) worker candidates for a disagg role.

    Returns a DataFrame in ``common.ColumnsStatic`` schema, one row per
    (parallel, batch_size) that fits in memory.  Replaces the body of
    ``DisaggInferenceSession.get_worker_candidates``.
    """
    backend = get_backend(backend_name)
    result_rows: list[pd.DataFrame] = []
    exceptions: list[Exception] = []
    all_configs_oom = True
    perf_misses = 0

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
            point_mc = dataclasses.replace(
                _point_model_config(model_config, parallel_config),
                tp_size=tp_size,
                pp_size=pp_size,
                moe_tp_size=moe_tp_size,
                moe_ep_size=moe_ep_size,
                attention_dp_size=dp_size,
                cp_size=cp_size,
            )

            model = get_model(model_path=model_path, model_config=point_mc, backend_name=backend_name)

            for b in b_list:
                point_rt = dataclasses.replace(runtime_config, batch_size=b)
                try:
                    summary = predict_disagg_worker(
                        model=model,
                        backend=backend,
                        database=database,
                        runtime_config=point_rt,
                        role=role,  # type: ignore[arg-type]
                        latency_correction=latency_correction,
                        predictor=predictor,
                        speculative_profile=speculative_profile,
                        free_gpu_memory_fraction=free_gpu_memory_fraction,
                    )
                except PerfDataNotAvailableError:
                    # Unanswerable batch point (e.g. FPM out-of-domain):
                    # skip the point, keep the parallel config.
                    perf_misses += 1
                    continue
                if not summary.check_oom() and not summary.check_kv_cache_oom():
                    all_configs_oom = False
                    summary_df = summary.get_summary_df().copy()
                    summary_df[MOE_COMM_FALLBACKS_COLUMN] = [
                        merge_moe_comm_fallbacks(summary.get_moe_comm_fallbacks())
                    ] * len(summary_df)
                    result_rows.append(summary_df)
                else:
                    # Larger b will always OOM. check_kv_cache_oom covers the
                    # fraction-based budget (e.g. vLLM only manages
                    # gpu_memory_utilization of total memory): a worker whose
                    # KV for batch b cannot actually be allocated must not
                    # enter the candidate pool, or the search selects
                    # deployments whose projected concurrency is physically
                    # unreachable (#1396).
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

    if not result_rows:
        if exceptions:
            terminal_error = _preferred_sweep_exception(exceptions)
            raise RuntimeError(
                f"sweep_disagg/{role}: no results for any parallel config. Selected exception: {terminal_error}"
            ) from terminal_error
        if all_configs_oom:
            if perf_misses:
                raise NoFeasibleConfigError(
                    f"sweep_disagg/{role}: no results — {perf_misses} batch point(s) had no answerable "
                    "perf data (e.g. FPM queries outside the collected domain, or no collected cell "
                    "matches the model identity/quant modes). Check the collected cells against the "
                    "resolved quant configuration, or use forward_model='op_level'."
                )
            raise InsufficientMemoryError(
                f"sweep_disagg/{role}: no results — model does not fit in GPU memory for any parallel config. "
                "Try increasing GPU budget, using a quantized model, or a system with more VRAM per GPU."
            )
        raise NoFeasibleConfigError(
            f"sweep_disagg/{role}: no parallel configuration met TTFT/TPOT or request-latency constraints."
        )
    return pd.concat(result_rows, axis=0, ignore_index=True)


# ---------------------------------------------------------------------------
# EPD (encoder disaggregation) helpers
# ---------------------------------------------------------------------------


def _get_encoder_worker_candidates(
    *,
    model_path: str,
    tp_list: list[int] | None,
    b_list: list[int] | None,
    runtime_config: config.RuntimeConfig,
    database: PerfDatabase,
    backend_name: str,
    latency_correction: float,
) -> list[dict]:
    """Enumerate (tp, batch) encode-worker candidates for EPD.

    An encode (E) worker runs only the vision encoder (ViT + projector),
    mirroring an encoder-only instance that loads no LLM weights (e.g.
    SGLang ``--encoder-only``), so only ViT-side rules constrain its tp.
    The ViT is weight-sharded over the worker's tp (encoder DP off, the
    engines' encoder-instance default); the DP layout's benefit is expressed
    by sweeping multiple small-tp workers instead.  A worker encodes one
    batch of ``b`` requests in ``encoder_latency`` ms (throughput
    ``b / encoder_latency``); batch points that do not improve throughput
    or exceed the encoder system's GPU memory are dropped.

    Returns row dicts with keys
    ``encoder_latency / seq/s / num_total_gpus / tp / bs / memory /
    power_w / power_coverage``.
    """
    backend = get_backend(backend_name)
    enc_cfg = get_model_config_from_model_path(model_path).get("extra_params")
    if not isinstance(enc_cfg, common.VisionEncoderConfig):
        # Not a VL model -> EPD cannot apply (config error, not a type bug).
        raise ValueError(  # noqa: TRY004
            f"EPD (encoder disaggregation) requested but model {model_path!r} has no vision encoder."
        )
    if BaseBackend._visual_context_tokens_from_encoder_config(enc_cfg, runtime_config) <= 0:
        raise ValueError(
            "EPD (encoder disaggregation) requested but the workload has no image input; "
            "set image_height/image_width (or num_image_tokens) and num_images_per_request."
        )
    rows: list[dict] = []
    saw_oom = False
    # Same OOM gate as the P/D candidates (set_memory_and_check_oom), charging
    # the same framework overhead as _get_memory_usage (nccl_mem + other_mem).
    mem_capacity_gib = database.system_spec["gpu"]["mem_capacity"] / (1 << 30)
    misc_spec = database.system_spec["misc"]
    other_mem = misc_spec["other_mem"] * (1.0 + backend.OTHERS_OVERHEAD_FRAC)
    # The dominance filter below relies on ascending batch order.
    b_schedule = sorted(set(b_list or _DEFAULT_ENCODER_BATCH_SCHEDULE))
    if b_schedule[-1] > _MAX_ENCODER_BATCH:
        raise ValueError(
            f"encoder batch candidates {[b for b in b_schedule if b > _MAX_ENCODER_BATCH]} exceed "
            f"the supported maximum {_MAX_ENCODER_BATCH} (SGLang's SGLANG_ENCODER_MAX_BATCH_SIZE "
            "default); modeling a raised deployment cap needs an explicit knob."
        )
    for etp in sorted(set(tp_list or _DEFAULT_ENCODER_TP_LIST)):
        if min(etp, 8) not in misc_spec["nccl_mem"]:
            logger.warning(
                "EPD encoder: tp=%s skipped: no comm data for this world size on this system (supported: %s)",
                etp,
                sorted(misc_spec["nccl_mem"]),
            )
            continue
        overhead_gib = (misc_spec["nccl_mem"][min(etp, 8)] + other_mem) / (1 << 30)
        try:
            encoder_ops = build_encoder_ops(enc_cfg, etp, enable_encoder_dp=False)
        except ValueError as e:
            # ViT-side constraint: heads / FFN width not divisible by this tp.
            logger.debug("EPD encoder: tp=%s rejected: %s", etp, e)
            continue
        model = EncoderOnlyModel(
            encoder_ops=encoder_ops,
            encoder_config=enc_cfg,
            # The ModelConfig slice the encoder phase reads (DP off).
            config=config.ModelConfig(tp_size=etp, enable_encoder_dp=False),
        )
        best_seq_s = 0.0
        for b in b_schedule:
            latency, power_w, memory, power_coverage = backend.run_encoder_static(
                model, database, runtime_config, b, latency_correction_scale=latency_correction
            )
            if memory.get("total", 0.0) + overhead_gib >= mem_capacity_gib:
                # Memory grows with batch: larger batches also OOM.
                saw_oom = True
                break
            if not (latency > 0 and math.isfinite(latency)):
                raise ValueError(
                    f"EPD encoder: invalid batch latency ({latency}) at tp={etp} bs={b}; "
                    "the encoder perf data or latency correction is invalid."
                )
            seq_s = b * 1000.0 / latency
            if seq_s <= best_seq_s:
                continue
            best_seq_s = seq_s
            rows.append(
                {
                    "encoder_latency": round(latency, 3),
                    "seq/s": seq_s,
                    "num_total_gpus": etp,
                    "tp": etp,
                    "bs": b,
                    "memory": round(memory.get("total", 0.0), 3),
                    "power_w": power_w,
                    "power_coverage": power_coverage,
                }
            )
    if not rows:
        if saw_oom:
            raise InsufficientMemoryError(
                "EPD encoder: the encode worker does not fit in GPU memory for any (tp, batch)."
            )
        raise NoFeasibleConfigError(
            "EPD encoder: no encode-worker candidate for any encoder tp "
            "(tp must divide the ViT geometry and have comm data on this system; see warnings)."
        )
    return rows


def _epd_e_num_candidates(
    lm_required: float,
    lm_gpus: int,
    encoder_capacity: float,
    encoder_gpus: int,
    num_gpu_set: set[int],
    bound: int,
):
    """Encode-pool sizes that can contribute a cell — exactly what a flat
    ``1..bound`` sweep would keep: under a per-replica budget only
    grid-landing counts pass the membership filter (derived here); without
    one, counts past the first non-binding size can never win the per-GPU
    argmax.  Ascending order keeps the flat sweep's tie behavior."""
    if num_gpu_set:
        counts = (
            (total - lm_gpus) // encoder_gpus
            for total in num_gpu_set
            if total > lm_gpus and (total - lm_gpus) % encoder_gpus == 0
        )
        return sorted(e for e in counts if e <= bound)
    return range(1, min(math.ceil(lm_required / encoder_capacity), bound) + 1)


def _degraded_encoder_capacity(seq_s: float, degradation: float, e_num: int = 1) -> float:
    """Degraded encode-pool capacity.  Keeps the ((seq/s x deg) x workers)
    association order shared by the matchers and the overlay cap, so the
    capped value is bit-identical to the capacity the argmax used."""
    return seq_s * degradation * e_num


def _overlay_encoder_stage(
    disagg_dict: dict,
    encoder_worker: dict,
    encoder_num_worker: int,
    prefill_power: float = 0.0,
    decode_power: float = 0.0,
    ttft_scale: float = 1.0,
    encoder_degradation: float = _RATE_MATCH_ENCODER_DEGRADATION,
) -> dict:
    """Overlay the encode stage onto a rate-matched P/D or agg row (EPD).

    Encode -> prefill is sequential per request, so the encode batch latency
    adds to TTFT and request latency.  ``ttft_scale`` mirrors each baseline:
    disagg corrects the encode stage like its prefill ttft (x1.8), agg adds
    it raw (run_agg adds the inline encoder outside its queueing factor).
    When the degraded encode pool capacity is below the row throughput,
    ``seq/s`` is capped and the rate-derived columns rescale with it;
    ``power_w`` (and ``power_coverage``, when the row carries it) is
    re-weighted over the encode + prefill + decode timeline.
    """
    row = dict(disagg_dict)
    encoder_latency = encoder_worker["encoder_latency"]
    encoder_ttft_share = encoder_latency * ttft_scale
    prefill_ttft = row["ttft"]
    decode_time = row["tpot"] * max(row["osl"] - 1, 0)
    total_time = encoder_ttft_share + prefill_ttft + decode_time
    if total_time > 0:
        row["power_w"] = (
            encoder_worker.get("power_w", 0.0) * encoder_ttft_share
            + prefill_power * prefill_ttft
            + decode_power * decode_time
        ) / total_time
        if "power_coverage" in row:
            row["power_coverage"] = (
                encoder_worker.get("power_coverage", 0.0) * encoder_ttft_share
                + row["power_coverage"] * (prefill_ttft + decode_time)
            ) / total_time
        elif not encoder_worker.get("power_coverage", 0.0) >= 1.0:
            # Sweep rows carry no coverage channel to express a partially
            # covered encoder, so anything short of full encoder energy data
            # fails closed to the no-data sentinel instead of silently
            # understating the blended power.
            row["power_w"] = 0.0
    row["encoder_latency"] = encoder_latency
    row["ttft"] = prefill_ttft + encoder_ttft_share
    row["request_latency"] = row["request_latency"] + encoder_ttft_share
    encoder_capacity = _degraded_encoder_capacity(encoder_worker["seq/s"], encoder_degradation, encoder_num_worker)
    if 0 < encoder_capacity < row["seq/s"]:
        row["tokens/s"] = row["tokens/s"] * (encoder_capacity / row["seq/s"])
        row["seq/s"] = encoder_capacity
        row["request_rate"] = encoder_capacity
    num_total_gpus = row["num_total_gpus"] + encoder_worker["num_total_gpus"] * encoder_num_worker
    row["seq/s/gpu"] = row["seq/s"] / num_total_gpus
    row["tokens/s/gpu"] = row["tokens/s"] / num_total_gpus
    row["num_total_gpus"] = num_total_gpus
    row["(e)workers"] = encoder_num_worker
    row["(e)tp"] = encoder_worker["tp"]
    row["(e)pp"] = 1
    row["(e)bs"] = encoder_worker["bs"]
    row["(e)parallel"] = f"tp{encoder_worker['tp']}"
    row["(e)memory"] = encoder_worker["memory"]
    return row


def _rate_match_agg_epd(
    agg_df: pd.DataFrame,
    encoder_records: list[dict],
    *,
    ttft_target: float,
    num_gpu_set: set[int] | None = None,
    top_k: int = 0,
    max_encoder_workers: int | None = None,
    encoder_degradation: float = _RATE_MATCH_ENCODER_DEGRADATION,
) -> pd.DataFrame:
    """Rate-match the encode pool against language-only agg workers (E+agg).

    For each encode-worker choice, pair the agg rows whose encode latency
    still fits the TTFT budget -- best throughput first, up to ``top_k``
    rows per choice (0 = uncapped); the caller hands over the full
    SLA-feasible row set so a pre-cut cannot shadow the pairable rows.
    Each pairing sweeps cells of ``a`` agg + ``e`` encode workers (cell
    throughput = min of the two pools, applied by the overlay); cells whose
    GPU total is not in ``num_gpu_set`` are skipped, and the nondominated
    (cell gpus, cell rate) frontier is emitted — one row per surviving cell
    size — so the cluster-aware picker sees every packing alternative.
    Returned rows are per-cell (``common.ColumnsAggEpd`` plus passthrough
    columns), so the downstream replicas logic scales whole cells.
    """
    records = agg_df.sort_values(by="seq/s", ascending=False).to_dict("records")
    e_workers_bound = max_encoder_workers or _MAX_ENCODE_WORKERS
    rows: list[dict] = []
    for enc_worker in encoder_records:
        encoder_capacity = _degraded_encoder_capacity(float(enc_worker["seq/s"]), encoder_degradation)
        paired = 0
        for r in records:
            if enc_worker["encoder_latency"] + r["ttft"] >= ttft_target:
                continue
            rate_one = float(r["seq/s"])
            gpus_one = int(r["num_total_gpus"])
            if rate_one <= 0:
                continue
            # Downstream picking packs whole cells (floor(total_gpus /
            # cell_gpus)), so a single per-cell-efficiency argmax could
            # discard the packed winner; keep the best rate per cell size
            # and emit the nondominated (gpus, rate) frontier.
            best_per_gpus: dict[int, tuple[float, int, int]] = {}
            for a_num in range(1, _MAX_AGG_WORKERS_EPD + 1):
                agg_rate = rate_one * a_num
                for e_num in _epd_e_num_candidates(
                    agg_rate,
                    gpus_one * a_num,
                    encoder_capacity,
                    enc_worker["num_total_gpus"],
                    num_gpu_set or set(),
                    e_workers_bound,
                ):
                    num_gpu = gpus_one * a_num + enc_worker["num_total_gpus"] * e_num
                    if num_gpu_set and num_gpu not in num_gpu_set:
                        continue
                    cell_rate = min(agg_rate, encoder_capacity * e_num)
                    incumbent = best_per_gpus.get(num_gpu)
                    if incumbent is None or cell_rate > incumbent[0]:
                        best_per_gpus[num_gpu] = (cell_rate, a_num, e_num)
            if not best_per_gpus:
                continue
            kept: list[tuple[int, float]] = []
            for num_gpu in sorted(best_per_gpus):
                cell_rate, a_num, e_num = best_per_gpus[num_gpu]
                # Dominated at every cluster size: no rate gain over a
                # smaller cell, or a proportional repeat of one.
                if kept and cell_rate <= kept[-1][1]:
                    continue
                if any(num_gpu % gpus == 0 and cell_rate <= (num_gpu // gpus) * rate for gpus, rate in kept):
                    continue
                kept.append((num_gpu, cell_rate))
                cell = dict(r)
                cell["(a)workers"] = a_num
                cell["seq/s"] = rate_one * a_num
                cell["tokens/s"] = float(r["tokens/s"]) * a_num
                cell["request_rate"] = cell["seq/s"]
                cell["concurrency"] = r["concurrency"] * a_num
                cell["num_total_gpus"] = gpus_one * a_num
                # Cell rate columns are the uncapped agg-side capacity; the
                # overlay applies the min() with the encode pool.
                rows.append(
                    _overlay_encoder_stage(
                        cell,
                        enc_worker,
                        e_num,
                        prefill_power=r.get("power_w", 0.0),
                        decode_power=r.get("power_w", 0.0),
                        encoder_degradation=encoder_degradation,
                    )
                )
            paired += 1
            if top_k and paired >= top_k:
                break
    columns = common.ColumnsAggEpd + [c for c in agg_df.columns if c not in common.ColumnsAggEpd]
    return pd.DataFrame(rows, columns=columns)


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
    require_same_tp: bool | Callable[[dict, dict], bool],
    prefill_degradation: float,
    decode_degradation: float,
    match_workers: Any,
    autoscale_ttft_correction_factor: float = _AUTOSCALE_TTFT_CORRECTION_FACTOR,
    encoder_records: list[dict] | None = None,
    encoder_degradation: float = _RATE_MATCH_ENCODER_DEGRADATION,
) -> pd.DataFrame | None:
    """For one (ttft, tpot) pair, filter + rate-match + pick best per decode parallel.

    Mirrors ``_find_best_result_under_constraints`` in
    DisaggInferenceSession.find_best_disagg_result_under_constraints.

    ``match_workers`` is supplied by the caller (``sweep_disagg``) so its
    cache is shared across all (ttft, tpot) pairs -- its result is
    independent of the target, so a per-pair cache would recompute identical
    matches.

    ``require_same_tp`` may be a predicate over the (prefill row, decode row)
    pair instead of one bool: the constraint it models (sglang KV transfer
    layout) is lifted per pair by a large-EP side, which is a property of the
    pair's parallel configs, not of the task.

    When ``encoder_records`` is given (EPD), each encode-worker choice spends
    its batch latency out of the TTFT budget before the prefill filter and
    joins the worker rate matching as the third pool.
    """
    requires_same_tp = (
        require_same_tp if callable(require_same_tp) else (lambda _p, _d, _flag=bool(require_same_tp): _flag)
    )

    p_corrected = prefill_summary_df.assign(ttft=prefill_summary_df["ttft"] * autoscale_ttft_correction_factor)

    def _prefill_records(ttft_budget: float) -> list[dict]:
        candidates = p_corrected[p_corrected["ttft"] < ttft_budget]
        if len(candidates) == 0:
            return []
        return (
            candidates.sort_values(by=["seq/s/gpu", "global_bs"], ascending=[False, True])
            .reset_index(drop=True)
            .head(_MAX_PREFILL_WORKERS)
            .to_dict("records")
        )

    # Each encode choice spends its corrected (x1.8) latency out of the TTFT
    # budget -- same correction as prefill, mirroring the inline PD baseline
    # whose whole E+P ttft is corrected.  [None] = plain PD, full budget.
    encoder_choices: list[dict | None] = [None]
    if encoder_records:
        encoder_choices = [
            e for e in encoder_records if e["encoder_latency"] * autoscale_ttft_correction_factor < ttft_target
        ]
    p_records_per_choice = [
        _prefill_records(
            ttft_target - (enc_worker["encoder_latency"] * autoscale_ttft_correction_factor if enc_worker else 0.0)
        )
        for enc_worker in encoder_choices
    ]
    if not any(p_records_per_choice):
        logger.debug("sweep_disagg: no prefill candidates meet ttft<%sms", ttft_target)
        return None

    d_candidates = decode_summary_df[
        (decode_summary_df["tpot"] < tpot_target * _DECODE_FILTER_RATIO_MAX)
        & (decode_summary_df["tpot"] > tpot_target * _DECODE_FILTER_RATIO_MIN)
    ].copy()
    if len(d_candidates) == 0:
        logger.debug("sweep_disagg: no decode candidates meet tpot<%sms", tpot_target)
        return None

    all_category_results: list[dict] = []

    for parallel_value, parallel_group in d_candidates.groupby("parallel"):
        group_sorted = (
            parallel_group.sort_values(by=["seq/s/gpu"], ascending=[False])
            .reset_index(drop=True)
            .head(_MAX_DECODE_WORKERS_PER_CATEGORY)
        )
        decode_records = group_sorted.to_dict("records")
        category_results: list[dict] = []
        for enc_worker, p_records in zip(encoder_choices, p_records_per_choice, strict=True):
            for d_worker in decode_records:
                d_throughput = float(d_worker["seq/s"])
                d_gpus = d_worker["num_total_gpus"]
                for p_worker in p_records:
                    if p_worker["tp"] != d_worker["tp"] and requires_same_tp(p_worker, d_worker):
                        continue
                    p_throughput = float(p_worker["seq/s"])
                    p_gpus = p_worker["num_total_gpus"]
                    p_num, d_num, e_num = match_workers(
                        prefill_throughput=p_throughput,
                        prefill_gpus=p_gpus,
                        decode_throughput=d_throughput,
                        decode_gpus=d_gpus,
                        prefill_deg=prefill_degradation,
                        decode_deg=decode_degradation,
                        encoder_throughput=float(enc_worker["seq/s"]) if enc_worker else 0.0,
                        encoder_gpus=enc_worker["num_total_gpus"] if enc_worker else 0,
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
                    if enc_worker is not None:
                        disagg_dict = _overlay_encoder_stage(
                            disagg_dict,
                            enc_worker,
                            e_num,
                            prefill_power=p_worker.get("power_w", 0.0),
                            decode_power=d_worker.get("power_w", 0.0),
                            ttft_scale=autoscale_ttft_correction_factor,
                            encoder_degradation=encoder_degradation,
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

    df = pd.DataFrame(all_category_results, columns=[*common.ColumnsDisagg, MOE_COMM_FALLBACKS_COLUMN]).round(3)
    df = df.sort_values(by=["tokens/s/gpu"], ascending=[False]).head(return_top_k).reset_index(drop=True)
    return df


def sweep_disagg(
    *,
    model_path: str,
    runtime_config: config.RuntimeConfig,
    prefill_database: PerfDatabase,
    prefill_backend_name: str,
    prefill_model_config: config.ModelConfig | Callable[..., config.ModelConfig],
    prefill_parallel_config_list: list[tuple[int, int, int, int, int, int]] | list[list[int]],
    prefill_latency_correction: float,
    decode_database: PerfDatabase,
    decode_backend_name: str,
    decode_model_config: config.ModelConfig | Callable[..., config.ModelConfig],
    decode_parallel_config_list: list[tuple[int, int, int, int, int, int]] | list[list[int]],
    decode_latency_correction: float,
    prefill_max_num_tokens: int = 16384,
    decode_max_num_tokens: int = 512,
    prefill_num_worker_list: list[int] | None = None,
    decode_num_worker_list: list[int] | None = None,
    num_gpu_list: list[int] | None = None,
    max_prefill_gpus: int | None = None,
    max_decode_gpus: int | None = None,
    require_same_tp: bool | Callable[[dict, dict], bool] = False,
    autoscale: bool = False,
    target_tpot: float | None = None,
    rate_matching_prefill_degradation: float | None = None,
    rate_matching_decode_degradation: float | None = None,
    rate_matching_encoder_degradation: float | None = None,
    autoscale_ttft_correction_factor: float | None = None,
    enable_epd: bool = False,
    encoder_tp_list: list[int] | None = None,
    encoder_batch_list: list[int] | None = None,
    max_encoder_workers: int | None = None,
    encoder_latency_correction: float = 1.0,
    encoder_database: PerfDatabase | None = None,
    predictor: Any = None,
    speculative_profile: SpeculativeDecodingProfile | None = None,
    free_gpu_memory_fraction: float | None = None,
) -> pd.DataFrame:
    """Sweep prefill_parallel x decode_parallel x batches x workers with rate matching.

    Replaces ``pareto_analysis.disagg_pareto`` ->
    ``DisaggInferenceSession.find_best_disagg_result_under_constraints``.
    Output schema is ``common.ColumnsDisagg``, sorted by ``tokens/s/gpu``.

    The two databases / backends are accepted independently to support
    hetero-disagg (prefill and decode on different systems).

    ``enable_epd`` switches VL disagg into EPD: the vision encoder runs on
    dedicated encode workers, prefill workers become language-only, TTFT
    gains the encode batch latency, and the encode pool joins the worker
    rate matching (``(e)*`` columns in the output).

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
    if enable_epd and autoscale:
        raise ValueError("EPD (enable_epd) is not supported with autoscale.")

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
    e_deg = (
        rate_matching_encoder_degradation
        if rate_matching_encoder_degradation is not None
        else _RATE_MATCH_ENCODER_DEGRADATION
    )
    for deg_name, deg_value in (("prefill", p_deg), ("decode", d_deg), ("encoder", e_deg)):
        if not (math.isfinite(deg_value) and deg_value > 0):
            raise ValueError(
                f"rate_matching_{deg_name}_degradation must be a positive finite number, got {deg_value!r}."
            )
    ttft_corr = (
        autoscale_ttft_correction_factor
        if autoscale_ttft_correction_factor is not None
        else _AUTOSCALE_TTFT_CORRECTION_FACTOR
    )
    p_num_workers = prefill_num_worker_list or []
    d_num_workers = decode_num_worker_list or []
    e_workers_bound = max_encoder_workers or _MAX_ENCODE_WORKERS
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

    # The token budget divides by the effective ISL (text + vision context
    # tokens); Task._prefill_effective_isl builds the budget the same way,
    # so the caller's batch intent round-trips.
    prefill_effective_isl = BaseBackend.effective_prefill_isl(model_path, runtime_config)
    if prefill_max_num_tokens < prefill_effective_isl:
        logger.warning("prefill_max_num_tokens < effective prefill ISL, clamping to effective ISL")
        prefill_max_num_tokens = prefill_effective_isl
    max_prefill_batch_size = prefill_max_num_tokens // prefill_effective_isl
    prefill_batch_range = range(1, max_prefill_batch_size + 1)

    encoder_candidates: list[dict] | None = None
    if enable_epd:
        # Hetero encoder: encoder_database defaults to the prefill side.
        encoder_candidates = _get_encoder_worker_candidates(
            model_path=model_path,
            tp_list=encoder_tp_list,
            b_list=encoder_batch_list,
            runtime_config=runtime_config,
            database=encoder_database or prefill_database,
            backend_name=prefill_backend_name,
            latency_correction=encoder_latency_correction,
        )
        # EPD prefill workers are language-only (vision tokens stay in context).
        prefill_model_config = _language_only_config(prefill_model_config)

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
        speculative_profile=speculative_profile,
        free_gpu_memory_fraction=free_gpu_memory_fraction,
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
        speculative_profile=speculative_profile,
        free_gpu_memory_fraction=free_gpu_memory_fraction,
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
            prefill_degradation_factor=p_deg,
            decode_degradation_factor=d_deg,
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

    # Worker-count matching is independent of the (ttft, tpot) target, so the
    # cache is defined once here and shared across constraint pairs (the
    # matching is the dominant cost of the disagg sweep).  Unbounded: the
    # (p, d[, e]) key cross-product can exceed a default lru size.
    @functools.cache
    def _match_workers(
        prefill_throughput: float,
        prefill_gpus: int,
        decode_throughput: float,
        decode_gpus: int,
        prefill_deg: float,
        decode_deg: float,
        encoder_throughput: float = 0.0,
        encoder_gpus: int = 0,
    ) -> tuple[int, int, int]:
        """Pick (p_num, d_num, e_num) maximizing throughput per GPU.

        The encode pool is the optional third rate-matched pool (EPD): the
        achieved throughput is the min of the degraded pool capacities.
        ``encoder_throughput = 0`` (plain PD) returns ``e_num = 0``.
        """
        prefill_opt, decode_opt, encoder_opt = -1, -1, -1
        throughput_per_gpu_max = 0.0
        encoder_capacity = _degraded_encoder_capacity(encoder_throughput, e_deg)
        for d_num in d_num_workers:
            for p_num in p_num_workers:
                if max_prefill_gpus is not None and max_decode_gpus is not None:
                    if prefill_gpus * p_num > max_prefill_gpus:
                        continue
                    if decode_gpus * d_num > max_decode_gpus:
                        continue
                p_corrected = prefill_throughput * p_num * prefill_deg
                d_corrected = decode_throughput * d_num * decode_deg
                pd_required = min(p_corrected, d_corrected)
                if encoder_capacity > 0:
                    e_num_candidates = _epd_e_num_candidates(
                        pd_required,
                        prefill_gpus * p_num + decode_gpus * d_num,
                        encoder_capacity,
                        encoder_gpus,
                        num_gpu_set,
                        e_workers_bound,
                    )
                else:
                    e_num_candidates = (0,)
                for e_num in e_num_candidates:
                    required = pd_required if e_num == 0 else min(pd_required, encoder_capacity * e_num)
                    num_gpu = prefill_gpus * p_num + decode_gpus * d_num + encoder_gpus * e_num
                    if num_gpu_set and num_gpu not in num_gpu_set:
                        continue
                    tpg = required / num_gpu
                    if tpg > throughput_per_gpu_max:
                        throughput_per_gpu_max = tpg
                        prefill_opt, decode_opt, encoder_opt = p_num, d_num, e_num
        return prefill_opt, decode_opt, encoder_opt

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
            encoder_records=encoder_candidates,
            encoder_degradation=e_deg,
        )
        if partial is not None:
            disagg_df = pd.concat([disagg_df, partial], axis=0, ignore_index=True)

    if len(disagg_df) == 0:
        logger.debug("sweep_disagg: no disagg result satisfies any constraint")
        return pd.DataFrame(columns=common.ColumnsDisagg)

    dedupe_cols = [
        column for column in disagg_df.columns if column not in {"_per_ops_source", MOE_COMM_FALLBACKS_COLUMN}
    ]
    return (
        disagg_df.drop_duplicates(subset=dedupe_cols, ignore_index=True)
        .sort_values(by="tokens/s/gpu", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# AFD sweep
# ---------------------------------------------------------------------------


def sweep_afd(
    *,
    model_path: str,
    runtime_config: config.RuntimeConfig,
    database: PerfDatabase,
    backend_name: str,
    model_config: config.ModelConfig,
    afd_parallel_config_list: list[tuple[int, int, int, int, int, str]],
    gpus_per_node: int,
    total_gpus: int | None = None,
    combined_with_pd: bool = True,
    comm_overhead_factor: float = 1.0,
    boundary_on_attn: bool = True,
    router_on_attn: bool = False,
    f_latency_scale: float = 1.0,
    comm_hiding_tolerance: float = 0.1,
    total_batch_size: int | None = None,
    max_a_batch_size: int = 1024,
    target_ttft: float | None = None,
    free_gpu_memory_fraction: float | None = None,
    max_seq_len: int | None = None,
    # Hetero A/F pools; each defaults to the shared database / backend /
    # gpus_per_node, so omitting them keeps the homogeneous behavior.
    a_database: PerfDatabase | None = None,
    a_backend_name: str | None = None,
    a_system_name: str | None = None,
    a_gpus_per_node: int | None = None,
    f_database: PerfDatabase | None = None,
    f_backend_name: str | None = None,
    f_system_name: str | None = None,
    f_gpus_per_node: int | None = None,
    # combined-with-PD prefill options
    prefill_database: PerfDatabase | None = None,
    prefill_backend_name: str | None = None,
    prefill_model_config: config.ModelConfig | None = None,
    prefill_parallel_config_list: list | None = None,
    prefill_batch_size_list: list[int] | None = None,
    prefill_system_name: str | None = None,
    prefill_backend_version: str | None = None,
    prefill_gpus_per_node: int | None = None,
    prefill_max_candidates: int = 256,
    prefill_candidate_overflow: str = "error",
    max_prefill_gpus: int | None = None,
    max_prefill_workers: int | None = None,
    # calibration
    prefill_degradation: float | None = None,
    decode_degradation: float | None = None,
    ttft_correction_factor: float | None = None,
    decode_latency_correction: float = 1.0,
) -> pd.DataFrame:
    """Sweep AFD candidate topologies; return feasible-candidate DataFrame.

    Thin wrapper around :func:`pareto_analysis.afd_pareto` that matches the
    interface pattern of :func:`sweep_agg` / :func:`sweep_disagg`.

    Returns:
        DataFrame with :data:`common.ColumnsAFD` schema sorted by
        ``tokens/s/gpu`` descending.

    Raises:
        NoFeasibleConfigError: When no candidate satisfies the SLA.
    """
    from aiconfigurator.sdk.pareto_analysis import (
        _AFD_DECODE_DEGRADATION,
        _AFD_PREFILL_DEGRADATION,
        _AFD_TTFT_CORRECTION_FACTOR,
        afd_pareto,
    )

    if not afd_parallel_config_list:
        raise NoFeasibleConfigError("sweep_afd: empty afd_parallel_config_list — no AFD topologies to evaluate.")

    result_df = afd_pareto(
        model_path=model_path,
        runtime_config=runtime_config,
        database=database,
        backend_name=backend_name,
        afd_parallel_config_list=afd_parallel_config_list,
        gpus_per_node=gpus_per_node,
        model_config=model_config,
        total_gpus=total_gpus,
        combined_with_pd=combined_with_pd,
        comm_overhead_factor=comm_overhead_factor,
        boundary_on_attn=boundary_on_attn,
        router_on_attn=router_on_attn,
        f_latency_scale=f_latency_scale,
        comm_hiding_tolerance=comm_hiding_tolerance,
        total_batch_size=total_batch_size,
        max_a_batch_size=max_a_batch_size,
        target_ttft=target_ttft,
        free_gpu_memory_fraction=free_gpu_memory_fraction,
        max_seq_len=max_seq_len,
        a_database=a_database,
        a_backend_name=a_backend_name,
        a_system_name=a_system_name,
        a_gpus_per_node=a_gpus_per_node,
        f_database=f_database,
        f_backend_name=f_backend_name,
        f_system_name=f_system_name,
        f_gpus_per_node=f_gpus_per_node,
        prefill_database=prefill_database,
        prefill_backend_name=prefill_backend_name,
        prefill_model_config=prefill_model_config,
        prefill_parallel_config_list=prefill_parallel_config_list,
        prefill_batch_size_list=prefill_batch_size_list,
        prefill_system_name=prefill_system_name,
        prefill_backend_version=prefill_backend_version,
        prefill_gpus_per_node=prefill_gpus_per_node,
        prefill_max_candidates=prefill_max_candidates,
        prefill_candidate_overflow=prefill_candidate_overflow,
        max_prefill_gpus=max_prefill_gpus,
        max_prefill_workers=max_prefill_workers,
        prefill_degradation=prefill_degradation or _AFD_PREFILL_DEGRADATION,
        decode_degradation=decode_degradation or _AFD_DECODE_DEGRADATION,
        ttft_correction_factor=ttft_correction_factor or _AFD_TTFT_CORRECTION_FACTOR,
        decode_latency_correction=decode_latency_correction,
    )
    return result_df
