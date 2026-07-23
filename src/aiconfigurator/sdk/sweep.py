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
import math
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
from aiconfigurator.sdk.models.vit_ops import EncoderOnlyModel, build_encoder_ops
from aiconfigurator.sdk.perf_database import PerfDatabase
from aiconfigurator.sdk.predict import predict_agg_worker, predict_disagg_worker
from aiconfigurator.sdk.utils import enumerate_ttft_tpot_constraints, get_model_config_from_model_path

logger = logging.getLogger(__name__)

# Empirical degradation factors used in disagg rate matching.  Sourced from
# the same values as :data:`aiconfigurator.sdk.picking._RATE_MATCHING_*`
# (locked in via parity test; do not change without updating picking.py too).
_RATE_MATCH_PREFILL_DEGRADATION = 0.9
_RATE_MATCH_DECODE_DEGRADATION = 0.92
# EPD: the encode pool suffers the same pipeline-bubble class of loss as
# prefill (imperfect batch packing under bursty arrivals).
_RATE_MATCH_ENCODER_DEGRADATION = 0.9

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

# Default EPD encode-worker search space.  TP mirrors a real encode
# instance's --tp-size choices.  The batch schedule models cross-request
# batching by the encode instance's greedy scheduler and is capped at
# SGLang's default (SGLANG_ENCODER_MAX_BATCH_SIZE = 8) -- larger batches
# do not exist in the deployed system, so sweeping them would credit the
# encode pool with unreachable throughput.
_DEFAULT_ENCODER_TP_LIST: list[int] = [1, 2, 4, 8]
_DEFAULT_ENCODER_BATCH_SCHEDULE: list[int] = [1, 2, 4, 8]

# E+agg: agg-worker replicas explored per rate-matched cell.  Mirrors the
# disagg worker lists (range(1, 33)); the cell only exists to express the
# integer agg:encode worker ratio.
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


def vl_effective_isl(model_path: str, runtime_config: config.RuntimeConfig) -> int:
    """Per-request effective ISL: text ISL plus vision context tokens.

    Single source of truth for the VL token accounting shared by ``Task``
    (which builds token budgets as ``batch x effective_isl``) and
    ``sweep_disagg`` (which divides the budget by the same value to recover
    the batch range) -- deriving it in two places would silently break that
    round trip.  Falls back to the plain text ISL when the model config
    cannot be resolved or the model / workload has no vision input.
    """
    try:
        enc_cfg = get_model_config_from_model_path(model_path).get("extra_params")
    except Exception:
        logger.debug("Could not resolve model config for VL effective ISL; using text ISL", exc_info=True)
        return runtime_config.isl
    return runtime_config.isl + BaseBackend._visual_context_tokens_from_encoder_config(enc_cfg, runtime_config)


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
        # visibility fields at zero/empty.  EPD overlays them via
        # _overlay_encoder_stage.
        "(e)workers": 0,
        "(e)tp": 0,
        "(e)pp": 0,
        "(e)bs": 0,
        "(e)parallel": "",
        "(e)memory": encoder_memory,
        "power_w": disagg_power_avg,
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
    enable_epd: bool = False,
    encoder_tp_list: list[int] | None = None,
    encoder_batch_list: list[int] | None = None,
    encoder_latency_correction: float = 1.0,
    encoder_database: PerfDatabase | None = None,
    num_gpu_list: list[int] | None = None,
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

    ``enable_epd`` switches VL agg into E+agg: the vision encoder runs on
    dedicated encode workers (same candidate space as EPD disagg,
    ``encoder_tp_list x encoder_batch_list``) while prefill+decode stay
    aggregated.  The agg worker becomes language-only (vision tokens stay in
    context, no ViT hosted); each row is then a rate-matched cell of
    ``(a)workers`` agg workers plus ``(e)workers`` encode workers, with the
    encode batch latency added to TTFT.  ``num_gpu_list`` is the allowed
    per-cell (per-replica) GPU counts for those cells, exactly as in
    ``sweep_disagg``; it is unused outside EPD (a plain agg row is a single
    worker whose GPU count is already gated by the parallel candidates).
    Default (``enable_epd=False``) keeps the encoder inline (colocated): its
    latency is part of the agg worker's TTFT and its weights part of the
    worker's memory.

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

    encoder_candidates: list[dict] | None = None
    if enable_epd:
        # E+agg: enumerate the encode-worker pool once (independent of the
        # agg parallel choice), then sweep language-only agg workers.  The
        # encode pool may live on its own system / GPU type (hetero
        # encoder, ``encoder_database``); it defaults to the agg side.
        encoder_candidates = _get_encoder_worker_candidates(
            model_path=model_path,
            tp_list=encoder_tp_list,
            b_list=encoder_batch_list,
            runtime_config=runtime_config,
            database=encoder_database or database,
            backend_name=backend_name,
            latency_correction=encoder_latency_correction,
        )
        model_config = copy.deepcopy(model_config)
        model_config.language_only = True
    # Per-cell (per-replica) GPU budget for the E+agg rate matching, same
    # semantics as sweep_disagg's num_gpu_list.
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
                    # EPD defers the top_k cut to the encoder pairing: cutting
                    # the language-only rows here would let high-throughput
                    # rows without encode headroom shadow the pairable ones.
                    top_k=0 if encoder_candidates is not None else top_k,
                    max_batch_size=max_batch_size,
                    ctx_stride=ctx_stride,
                    enable_chunked_prefill=enable_chunked_prefill,
                    free_gpu_memory_fraction=free_gpu_memory_fraction,
                    max_seq_len=max_seq_len,
                    predictor=predictor,
                )
                saw_model_fit |= point_saw_model_fit
                saw_memory_fit |= point_saw_memory_fit
                if encoder_candidates is not None and len(point_df) > 0:
                    # E+agg: the per-point ttft filter above ran on the
                    # language-only ttft (a superset — adding the encode
                    # latency only tightens it); pair exactly here.
                    point_df = _rate_match_agg_epd(
                        point_df,
                        encoder_candidates,
                        ttft_target=point_rt.ttft,
                        num_gpu_set=epd_num_gpu_set,
                        top_k=top_k,
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

    An encode (E) worker runs only the vision encoder (ViT + projector) with
    its own tensor parallelism, mirroring an encoder-only instance that loads
    no LLM weights (e.g. SGLang ``--encoder-only``).  The worker model is
    built directly from the encoder config: constructing the full LLM here
    would waste work and, worse, let LLM-side parallelism rules (KV-head
    divisibility, MoE width identities) reject a tp the ViT itself supports.
    ``tp_list`` / ``b_list`` fall back to :data:`_DEFAULT_ENCODER_TP_LIST` /
    :data:`_DEFAULT_ENCODER_BATCH_SCHEDULE` when None.
    A worker encodes one batch of ``b`` requests (each with
    ``num_images_per_request`` images) in ``encoder_latency`` ms, so its
    throughput is ``b / encoder_latency``.  Per tp, batch points that do not
    improve throughput over a smaller batch are dominated (same rate at
    worse latency) and dropped.

    Returns row dicts with keys
    ``encoder_latency / seq/s / num_total_gpus / tp / bs / memory``.
    """
    backend = get_backend(backend_name)
    enc_cfg = get_model_config_from_model_path(model_path).get("extra_params")
    if not isinstance(enc_cfg, common.VisionEncoderConfig):
        # Config error, not a typing mistake: "no VisionEncoderConfig" means
        # the model is not a VL model, so EPD cannot apply to it.
        raise ValueError(  # noqa: TRY004
            f"EPD (encoder disaggregation) requested but model {model_path!r} has no vision encoder."
        )
    if BaseBackend._visual_context_tokens_from_encoder_config(enc_cfg, runtime_config) <= 0:
        raise ValueError(
            "EPD (encoder disaggregation) requested but the workload has no image input; "
            "set image_height/image_width (or num_image_tokens) and num_images_per_request."
        )
    rows: list[dict] = []
    # The dominance filter below relies on ascending batch order.
    b_schedule = sorted(set(b_list or _DEFAULT_ENCODER_BATCH_SCHEDULE))
    for etp in sorted(set(tp_list or _DEFAULT_ENCODER_TP_LIST)):
        try:
            encoder_ops = build_encoder_ops(enc_cfg, etp)
        except ValueError as e:
            # ViT-side constraint: heads / FFN width not divisible by this tp.
            logger.debug("EPD encoder: tp=%s rejected: %s", etp, e)
            continue
        model = EncoderOnlyModel(encoder_ops=encoder_ops, encoder_config=enc_cfg)
        best_seq_s = 0.0
        for b in b_schedule:
            latency, power_w, memory = backend.run_encoder_static(
                model, database, runtime_config, b, latency_correction_scale=latency_correction
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
                }
            )
    if not rows:
        raise NoFeasibleConfigError("EPD encoder: no encode-worker candidate for any encoder tp.")
    return rows


def _overlay_encoder_stage(
    disagg_dict: dict,
    encoder_worker: dict,
    encoder_num_worker: int,
    prefill_power: float = 0.0,
    decode_power: float = 0.0,
    ttft_scale: float = 1.0,
) -> dict:
    """Overlay the encode stage onto a rate-matched P/D or agg row (EPD).

    Encode -> prefill is sequential per request (prefill starts only after
    the full image embedding arrives), so the encode batch latency adds to
    TTFT and request latency.  ``ttft_scale`` keeps the queueing correction
    symmetric with the non-EPD baseline: the disagg path passes the
    autoscale TTFT correction factor (its inline-encoder prefill ttft is
    corrected wholesale, so the disaggregated encode stage must be too),
    while the agg path keeps 1.0 (run_agg adds the inline encoder outside
    its queueing factor).  The ``encoder_latency`` column stays the raw
    stage latency, like the inline rows.  Three-pool rate matching: when
    the encode pool's degraded capacity is below the row throughput (the
    matcher may pick an encode-bound cell), ``seq/s`` is capped at that
    capacity and the rate-derived columns rescale with it; per-request
    latency columns are unaffected (the batch latency does not depend on
    the pool size).  ``power_w`` is re-weighted over the three-phase
    timeline (encode + prefill + decode); the prefill phase reuses the
    colocated worker's average power as an approximation.
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
    row["encoder_latency"] = encoder_latency
    row["ttft"] = prefill_ttft + encoder_ttft_share
    row["request_latency"] = row["request_latency"] + encoder_ttft_share
    # Same association order as the matchers ((seq/s x deg) x workers) so
    # the capped value is bit-identical to the capacity the argmax used.
    encoder_capacity = encoder_worker["seq/s"] * _RATE_MATCH_ENCODER_DEGRADATION * encoder_num_worker
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
) -> pd.DataFrame:
    """Rate-match the encode pool against language-only agg workers (E+agg).

    The agg counterpart of the disagg encoder rate matching + ``_overlay_encoder_stage``:
    for each encode-worker choice, pair the agg rows whose encode latency
    still fits the TTFT budget -- best throughput first, up to ``top_k``
    rows per choice (0 = uncapped), mirroring the per-choice
    ``_prefill_records`` re-filter in the disagg path.  The caller hands
    over the full SLA-feasible row set: a top_k cut on the language-only
    rows *before* this pairing would let high-throughput rows without
    encode headroom shadow the (possibly only) pairable rows.  For each
    pairing, sweep cells of ``a`` agg workers + ``e`` encode workers
    -- ``e`` up to the first count that no longer binds (larger pools are
    dominated), smaller counts making the encode pool the rate-matched
    bottleneck (cell throughput = min of the two pools, applied by the
    overlay).  ``num_gpu_set`` is the per-replica GPU budget, exactly as in
    the disagg matching: cells whose total GPU count is not allowed are
    skipped *before* the argmax, so a feasible small cell is never shadowed
    by an infeasible better-amortized large one.  Among feasible cells the
    throughput-per-GPU argmax wins; ties keep the smaller cell.  The
    returned rows are per-cell — ``(a)workers`` agg workers — so the
    downstream replicas logic scales whole cells.
    """
    records = agg_df.sort_values(by="seq/s", ascending=False).to_dict("records")
    rows: list[dict] = []
    for enc in encoder_records:
        encoder_capacity = float(enc["seq/s"]) * _RATE_MATCH_ENCODER_DEGRADATION
        paired = 0
        for r in records:
            if enc["encoder_latency"] + r["ttft"] >= ttft_target:
                continue
            rate_one = float(r["seq/s"])
            gpus_one = int(r["num_total_gpus"])
            if rate_one <= 0:
                continue
            best: tuple[tuple[float, int], int, int] | None = None
            for a_num in range(1, _MAX_AGG_WORKERS_EPD + 1):
                agg_rate = rate_one * a_num
                for e_num in range(1, max(1, math.ceil(agg_rate / encoder_capacity)) + 1):
                    num_gpu = gpus_one * a_num + enc["num_total_gpus"] * e_num
                    if num_gpu_set and num_gpu not in num_gpu_set:
                        continue
                    cell_rate = min(agg_rate, encoder_capacity * e_num)
                    key = (cell_rate / num_gpu, -num_gpu)
                    if best is None or key > best[0]:
                        best = (key, a_num, e_num)
            if best is None:
                continue
            _, a_num, e_num = best
            cell = dict(r)
            cell["(a)workers"] = a_num
            cell["seq/s"] = rate_one * a_num
            cell["tokens/s"] = float(r["tokens/s"]) * a_num
            cell["request_rate"] = cell["seq/s"]
            cell["concurrency"] = r["concurrency"] * a_num
            cell["num_total_gpus"] = gpus_one * a_num
            # cell rate columns are the uncapped agg-side capacity; the
            # overlay applies the min() with the encode pool and recomputes
            # the per-GPU metrics with the encode GPUs included.  The agg
            # worker covers both the prefill and decode phases of the power
            # timeline.
            rows.append(
                _overlay_encoder_stage(
                    cell,
                    enc,
                    e_num,
                    prefill_power=r.get("power_w", 0.0),
                    decode_power=r.get("power_w", 0.0),
                )
            )
            paired += 1
            if top_k and paired >= top_k:
                break
    if not rows:
        return pd.DataFrame(columns=list(agg_df.columns))
    return pd.DataFrame(rows)


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
    encoder_records: list[dict] | None = None,
) -> pd.DataFrame | None:
    """For one (ttft, tpot) pair, filter + rate-match + pick best per decode parallel.

    Mirrors ``_find_best_result_under_constraints`` in
    DisaggInferenceSession.find_best_disagg_result_under_constraints.

    ``match_workers`` is supplied by the caller (``sweep_disagg``) so its
    cache is shared across all (ttft, tpot) pairs -- its result is
    independent of the target, so a per-pair cache would recompute identical
    matches.

    When ``encoder_records`` is given (EPD), each encode-worker choice spends
    its batch latency out of the TTFT budget before the prefill filter and
    joins the worker rate matching as the third pool (``match_workers``
    encoder arguments); the per-decode-parallel best is picked across encode
    choices as well.
    """

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

    # EPD: encode -> prefill is sequential per request, so each encode choice
    # consumes its latency from the TTFT budget -- under the same queueing
    # correction as the prefill stage (the inline PD baseline corrects its
    # whole E+P ttft, so correcting only P here would systematically favor
    # EPD by the uncorrected encode share).  Plain PD is the single
    # no-encoder choice with the full budget.
    encoder_choices: list[dict | None] = [None]
    if encoder_records:
        encoder_choices = [
            e for e in encoder_records if e["encoder_latency"] * autoscale_ttft_correction_factor < ttft_target
        ]
    p_records_per_choice = [
        _prefill_records(ttft_target - (enc["encoder_latency"] * autoscale_ttft_correction_factor if enc else 0.0))
        for enc in encoder_choices
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
        for enc, p_records in zip(encoder_choices, p_records_per_choice, strict=True):
            for d_worker in decode_records:
                d_throughput = float(d_worker["seq/s"])
                d_gpus = d_worker["num_total_gpus"]
                for p_worker in p_records:
                    if require_same_tp and p_worker["tp"] != d_worker["tp"]:
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
                        encoder_throughput=float(enc["seq/s"]) if enc else 0.0,
                        encoder_gpus=enc["num_total_gpus"] if enc else 0,
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
                    if enc is not None:
                        disagg_dict = _overlay_encoder_stage(
                            disagg_dict,
                            enc,
                            e_num,
                            prefill_power=p_worker.get("power_w", 0.0),
                            decode_power=d_worker.get("power_w", 0.0),
                            ttft_scale=autoscale_ttft_correction_factor,
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
    enable_epd: bool = False,
    encoder_tp_list: list[int] | None = None,
    encoder_batch_list: list[int] | None = None,
    encoder_latency_correction: float = 1.0,
    encoder_database: PerfDatabase | None = None,
    predictor: Any = None,
) -> pd.DataFrame:
    """Sweep prefill_parallel x decode_parallel x batches x workers with rate matching.

    Replaces ``pareto_analysis.disagg_pareto`` ->
    ``DisaggInferenceSession.find_best_disagg_result_under_constraints``.
    Output schema is ``common.ColumnsDisagg``, sorted by ``tokens/s/gpu``.

    The two databases / backends are accepted independently to support
    hetero-disagg (prefill and decode on different systems).

    ``enable_epd`` switches VL disagg into EPD: the vision encoder runs on
    dedicated encode workers, enumerated over
    ``encoder_tp_list x encoder_batch_list`` (defaults
    :data:`_DEFAULT_ENCODER_TP_LIST` / :data:`_DEFAULT_ENCODER_BATCH_SCHEDULE`)
    on the prefill database/backend.  Prefill workers become language-only
    (vision tokens stay in context, no ViT hosted), TTFT gains the encode
    batch latency, and the encode pool joins the worker rate matching
    (``(e)*`` columns in the output).

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

    # Vision tokens occupy the prefill context, so the per-request cost that
    # divides the token budget is the effective ISL (mirrors the legacy
    # DisaggInferenceSession and the agg sweep above).  Task builds the
    # budget with the same helper, so the caller's batch intent round-trips.
    prefill_effective_isl = vl_effective_isl(model_path, runtime_config)
    if prefill_max_num_tokens < prefill_effective_isl:
        logger.warning("prefill_max_num_tokens < effective prefill ISL, clamping to effective ISL")
        prefill_max_num_tokens = prefill_effective_isl
    max_prefill_batch_size = prefill_max_num_tokens // prefill_effective_isl
    prefill_batch_range = range(1, max_prefill_batch_size + 1)

    encoder_candidates: list[dict] | None = None
    if enable_epd:
        # The encode pool may live on its own system / GPU type (hetero
        # encoder, ``encoder_database``); it defaults to the prefill side.
        encoder_candidates = _get_encoder_worker_candidates(
            model_path=model_path,
            tp_list=encoder_tp_list,
            b_list=encoder_batch_list,
            runtime_config=runtime_config,
            database=encoder_database or prefill_database,
            backend_name=prefill_backend_name,
            latency_correction=encoder_latency_correction,
        )
        # EPD prefill workers are language-only (e.g. SGLang --language-only):
        # vision tokens stay in their context, but they never host the ViT.
        prefill_model_config = copy.deepcopy(prefill_model_config)
        prefill_model_config.language_only = True

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
    # Define it once here so the cache is shared across every constraint pair.
    # Nesting it inside _find_best_disagg_under_constraint rebuilt the cache per pair
    # and recomputed identical matches (the dominant cost of the disagg sweep).
    # Unbounded cache: the key space is the (p, d[, e]) candidate cross-product,
    # which can exceed a default lru size but stays small in memory.
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

        The encode pool is the optional third rate-matched pool (EPD),
        fully symmetric with P and D: its size is swept like theirs and the
        achieved throughput is the min of the three degraded pool
        capacities -- the encode pool may itself be the bottleneck, running
        at its degradation headroom exactly like a binding P or D pool.
        ``encoder_throughput = 0`` (plain PD) drops the encoder terms
        entirely and returns ``e_num = 0``.
        """
        prefill_opt, decode_opt, encoder_opt = -1, -1, -1
        throughput_per_gpu_max = 0.0
        encoder_capacity = encoder_throughput * _RATE_MATCH_ENCODER_DEGRADATION
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
                    # Sweep the encode pool up to the first size that no
                    # longer binds; larger pools are dominated (same
                    # throughput, more GPUs).  Smaller sizes trade capped
                    # throughput for GPUs -- and under a per-replica GPU
                    # budget they are sometimes the only feasible cells.
                    e_num_candidates = range(1, max(1, math.ceil(pd_required / encoder_capacity)) + 1)
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
