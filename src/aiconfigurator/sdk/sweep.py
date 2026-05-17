# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Search / sweep functions for finding Pareto-optimal worker configurations.

Two entry points:

- :func:`sweep_agg` — sweep parallel x batch x ctx_tokens for an aggregated
  IFB worker; filter by SLA; return a Pareto DataFrame.
- :func:`sweep_disagg` — sweep prefill_parallel x decode_parallel x
  batches x num_workers with rate matching; return a Pareto DataFrame.

Both functions own the entire search loop themselves and call
``predict.predict_*`` for per-point evaluation.  They replace the
``InferenceSession.find_best_*`` / ``DisaggInferenceSession.find_best_*``
search paths and the ``pareto_analysis.agg_pareto`` / ``disagg_pareto``
proxies.

Output DataFrame schema is ``common.ColumnsAgg`` for agg and
``common.ColumnsDisagg`` for disagg, so downstream picking in
:mod:`aiconfigurator.sdk.picking` works without change.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np
import pandas as pd

from aiconfigurator.sdk import common, config
from aiconfigurator.sdk.backends.factory import get_backend
from aiconfigurator.sdk.errors import NoFeasibleConfigError
from aiconfigurator.sdk.models import get_model
from aiconfigurator.sdk.perf_database import PerfDatabase
from aiconfigurator.sdk.predict import predict_agg_worker
from aiconfigurator.sdk.utils import enumerate_ttft_tpot_constraints

logger = logging.getLogger(__name__)

# Empirical degradation factors used in disagg rate matching.  Sourced from
# the same values as :data:`aiconfigurator.sdk.picking._RATE_MATCHING_*`
# (locked in via parity test; do not change without updating picking.py too).
_RATE_MATCH_PREFILL_DEGRADATION = 0.9
_RATE_MATCH_DECODE_DEGRADATION = 0.92

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
    model_path: str,
    backend_name: str,
    database: PerfDatabase,
    model_config: config.ModelConfig,
    runtime_config: config.RuntimeConfig,
    top_k: int,
    max_batch_size: int,
    ctx_stride: int,
    enable_chunked_prefill: bool,
    free_gpu_memory_fraction: float | None,
    max_seq_len: int | None,
) -> tuple[pd.DataFrame, bool]:
    """Sweep batch_size x ctx_tokens for one fixed parallel choice.

    Returns ``(rows_df, all_oom)``.  Logic faithfully reproduces the
    body of the legacy ``backend.find_best_agg_result_under_constraints``;
    parity is enforced by the integration test.
    """
    backend = get_backend(backend_name)
    model = get_model(model_path=model_path, model_config=model_config, backend_name=backend_name)

    isl = runtime_config.isl
    osl = runtime_config.osl
    ttft_target = runtime_config.ttft
    tpot_target = runtime_config.tpot
    prefix = runtime_config.prefix

    b_list = [b for b in _DEFAULT_AGG_BATCH_SCHEDULE if b <= max_batch_size]
    ctx_tokens_list = _agg_ctx_tokens_list(isl, ctx_stride, enable_chunked_prefill)

    results_dict_list: list[dict] = []
    results_per_ops_source: list[dict | None] = []
    capped_b: list[int] = []
    all_oom = True

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

            point_rt = config.RuntimeConfig(
                batch_size=b,
                isl=isl,
                osl=osl,
                prefix=prefix,
                seq_imbalance_correction_scale=runtime_config.seq_imbalance_correction_scale,
                engine_step_backend=runtime_config.engine_step_backend,
            )

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
                **backend_kwargs,
            )

            if summary.check_oom() or summary.check_kv_cache_oom():
                break  # ctx_tokens monotonic → larger will also OOM
            all_oom = False
            result_dict = summary.get_result_dict()
            if result_dict and result_dict["tpot"] <= tpot_target and result_dict["ttft"] <= ttft_target:
                results_dict_list.append(result_dict)
                results_per_ops_source.append(summary.get_per_ops_source())

    if not results_dict_list:
        return pd.DataFrame(columns=common.ColumnsAgg), all_oom

    df = pd.DataFrame(results_dict_list, columns=common.ColumnsAgg).round(3)
    df["_per_ops_source"] = results_per_ops_source
    df = df.sort_values(by="seq/s", ascending=False).round(3)
    if top_k > 0:
        df = df.head(top_k)
    return df, all_oom


def sweep_agg(
    *,
    model_path: str,
    runtime_config: config.RuntimeConfig,
    database: PerfDatabase,
    backend_name: str,
    model_config: config.ModelConfig,
    parallel_config_list: list[list[int]] | list[tuple[int, int, int, int, int]],
    top_k: int = 10,
    max_batch_size: int = 512,
    ctx_stride: int = 512,
    enable_chunked_prefill: bool = False,
    free_gpu_memory_fraction: float | None = None,
    max_seq_len: int | None = None,
) -> pd.DataFrame:
    """Sweep parallel x batch x ctx_tokens for agg; return Pareto DataFrame.

    Replaces ``pareto_analysis.agg_pareto`` -> ``InferenceSession.find_best_agg``
    -> ``backend.find_best_agg_result_under_constraints``.  Output schema is
    ``common.ColumnsAgg``, sorted by ``tokens/s/gpu`` descending.

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
        parallel_config_list: List of (tp, pp, dp, moe_tp, moe_ep) tuples
            to enumerate.
        top_k: Per-(parallel, tpot) top-K rows to keep before concat.
        max_batch_size: Upper bound on batch size sweep.
        ctx_stride: Stride for ctx_tokens sweep.
        enable_chunked_prefill: When False, ctx_tokens snaps to multiples of isl.
        free_gpu_memory_fraction: TRT-LLM-only KV cache fraction.
        max_seq_len: TRT-LLM-only per-slot KV cache budget.

    Returns:
        Deduped, sorted Pareto DataFrame with schema ``common.ColumnsAgg``.

    Raises:
        RuntimeError: When all configs OOM or no point meets SLA.
        NoFeasibleConfigError: When SLA cannot be satisfied at any point.
    """
    results_df = pd.DataFrame(columns=common.ColumnsAgg)
    exceptions: list[Exception] = []
    all_configs_oom = True
    all_kv_cache_oom = True

    for parallel_config in parallel_config_list:
        tp_size, pp_size, dp_size, moe_tp_size, moe_ep_size = parallel_config
        logger.debug(
            "sweep_agg: parallel tp=%s pp=%s dp=%s moe_tp=%s moe_ep=%s",
            tp_size,
            pp_size,
            dp_size,
            moe_tp_size,
            moe_ep_size,
        )
        try:
            point_model_config = copy.deepcopy(model_config)
            point_model_config.tp_size = tp_size
            point_model_config.pp_size = pp_size
            point_model_config.moe_tp_size = moe_tp_size
            point_model_config.moe_ep_size = moe_ep_size
            point_model_config.attention_dp_size = dp_size

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
                point_df, all_oom = _sweep_one_parallel_agg(
                    model_path=model_path,
                    backend_name=backend_name,
                    database=database,
                    model_config=point_model_config,
                    runtime_config=point_rt,
                    top_k=top_k,
                    max_batch_size=max_batch_size,
                    ctx_stride=ctx_stride,
                    enable_chunked_prefill=enable_chunked_prefill,
                    free_gpu_memory_fraction=free_gpu_memory_fraction,
                    max_seq_len=max_seq_len,
                )
                if not all_oom:
                    all_configs_oom = False
                # KV cache OOM is detected per-summary inside; we conservatively
                # mark "not all KV cache OOM" whenever we produced any row.
                if len(point_df) > 0:
                    all_kv_cache_oom = False
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
    if all_configs_oom:
        raise RuntimeError(
            "sweep_agg: no results — model does not fit in GPU memory for any parallel config. "
            "Try increasing --total-gpus, using a quantized model, or a system with more VRAM per GPU."
        )
    if all_kv_cache_oom:
        raise RuntimeError(
            "sweep_agg: no results — requested batch_size exceeds KV cache capacity for all configs. "
            "Try reducing batch_size, increasing free_gpu_memory_fraction, or a system with more VRAM."
        )
    raise NoFeasibleConfigError(
        "sweep_agg: no parallel configuration met TTFT/TPOT or request-latency constraints. "
        "Try relaxing --ttft / --tpot / --request-latency."
    )


# ---------------------------------------------------------------------------
# Disagg sweep — DEFERRED to Pass 2
# ---------------------------------------------------------------------------


def sweep_disagg(*args, **kwargs):  # pragma: no cover - placeholder
    """Sweep prefill_parallel x decode_parallel x batches x workers with rate matching.

    NOT YET IMPLEMENTED.  Pass 1 of the sweep.py refactor covers agg only.
    Pass 2 will inline the 334-line disagg search logic currently in
    ``DisaggInferenceSession.find_best_disagg_result_under_constraints``.

    Until then, callers needing disagg should keep using the legacy path
    (``DisaggInferenceSession`` is still in the tree and used by webapp).
    """
    raise NotImplementedError(
        "sweep_disagg is not implemented yet — Pass 2 of the sweep refactor. Use pareto_analysis.disagg_pareto for now."
    )
