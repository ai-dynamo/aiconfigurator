# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import logging
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotext

from aiconfigurator.logging_utils import use_plain_cli_output
from aiconfigurator.sdk import config
from aiconfigurator.sdk.backends.factory import get_backend
from aiconfigurator.sdk.common import ColumnsAFD, ColumnsAgg
from aiconfigurator.sdk.errors import NoFeasibleConfigError
from aiconfigurator.sdk.inference_session import (
    AFDInferenceSession,
    DisaggInferenceSession,
    InferenceSession,
)
from aiconfigurator.sdk.models import check_is_moe, get_model, resolve_context_fmha_compat
from aiconfigurator.sdk.perf_database import PerfDatabase
from aiconfigurator.sdk.utils import enumerate_ttft_tpot_constraints, strip_unicode_to_ascii

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# AFD rate-matching calibration constants
# ---------------------------------------------------------------------------

# AFD rate-matching degradation factors.
# Prefill: identical to disagg — same static-pool pipeline-bubble overhead.
_AFD_PREFILL_DEGRADATION = 0.9
# Decode: A/F ping-pong pipeline bubble + microbatch scheduling overhead.
# Slightly less than disagg's 0.92 (decode-slot under-saturation) because
# AFD decode has a different bottleneck — A↔F transfer overlap efficiency.
_AFD_DECODE_DEGRADATION = 0.95
# TTFT correction for concurrent prefill queueing (same pool structure as disagg).
_AFD_TTFT_CORRECTION_FACTOR = 1.8
_AFD_PREFILL_BATCH_SIZE_LIST = [1, 2, 4, 8, 16, 32]
_AFD_PREFILL_MAX_CANDIDATES = 256
_AFD_PREFILL_CANDIDATE_OVERFLOW = "error"
_AFD_LOW_LATENCY_BATCH_SIZE_LIST = [1, 2, 4]


def _normalize_positive_int_list(name: str, values: list[int] | tuple[int, ...] | None) -> list[int]:
    if values is None:
        return []
    if not isinstance(values, (list, tuple)):
        raise TypeError(f"{name} must be a list of positive integers.")
    normalized: list[int] = []
    for value in values:
        if isinstance(value, bool) or int(value) != value:
            raise ValueError(f"{name} entries must be positive integers, got {value!r}.")
        int_value = int(value)
        if int_value < 1:
            raise ValueError(f"{name} entries must be positive integers, got {value!r}.")
        normalized.append(int_value)
    return list(dict.fromkeys(normalized))


def agg_pareto(
    model_path: str,
    runtime_config: config.RuntimeConfig,
    database: PerfDatabase,
    backend_name: str,
    model_config: config.ModelConfig,
    parallel_config_list: list[list[int]],
    enable_chunked_prefill: bool = False,
    free_gpu_memory_fraction: float | None = None,
    max_seq_len: int | None = None,
) -> pd.DataFrame:
    """
    Find Pareto front for agg.
    We will first enumerate all the parallel configurations and then find the Pareto front for
    each parallel configuration.

    Args:
        model_path: name of the model
        runtime_config: runtime config. tpot is a list of tpot values to search over or a single
            tpot value
        database: database
        backend_name: name of the backend
        model_config: model config
        parallel_config_list: list of parallel configurations
        enable_chunked_prefill: whether the inference framework will have chunked prefill enabled.
            Affects the context tokens sweep granularity. Default is False.

    Returns:
        results_df: dataframe of the results
    """

    # agg is agg server, the loop over parallel is outside here.
    results_df = pd.DataFrame(columns=ColumnsAgg)
    exceptions = []
    all_configs_oom = True
    all_kv_cache_oom = True
    for parallel_config in parallel_config_list:
        tp_size, pp_size, dp_size, moe_tp_size, moe_ep_size, cp_size = parallel_config
        logger.debug(
            f"Getting candidate workers with parallel config: tp={tp_size}, pp={pp_size}, "
            f"dp={dp_size}, moe_tp={moe_tp_size}, moe_ep={moe_ep_size}, cp={cp_size}"
        )

        try:
            overwritten_model_config = copy.deepcopy(model_config)
            overwritten_model_config.pp_size = pp_size
            overwritten_model_config.tp_size = tp_size
            overwritten_model_config.moe_tp_size = moe_tp_size
            overwritten_model_config.moe_ep_size = moe_ep_size
            overwritten_model_config.attention_dp_size = dp_size
            overwritten_model_config.cp_size = cp_size
            model = get_model(
                model_path=model_path,
                model_config=overwritten_model_config,
                backend_name=backend_name,
            )
            backend = get_backend(backend_name)
            sess = InferenceSession(model=model, database=database, backend=backend)

            runtime_configs_to_evaluate: list[config.RuntimeConfig] = []
            if runtime_config.request_latency is not None and runtime_config.request_latency > 0:
                ttft_tpot_constraints = enumerate_ttft_tpot_constraints(
                    runtime_config.osl, runtime_config.request_latency, runtime_config.ttft
                )
                if not ttft_tpot_constraints:
                    logger.debug(
                        "No ttft/tpot constraints derived for request_latency=%s", runtime_config.request_latency
                    )
                    continue
                logger.debug(
                    "Enumerated %d ttft/tpot constraint pairs for request_latency=%sms",
                    len(ttft_tpot_constraints),
                    runtime_config.request_latency,
                )
                for ttft_constraint, tpot_constraint in ttft_tpot_constraints:
                    overwritten_runtime_config = copy.deepcopy(runtime_config)
                    overwritten_runtime_config.ttft = ttft_constraint
                    overwritten_runtime_config.tpot = tpot_constraint
                    runtime_configs_to_evaluate.append(overwritten_runtime_config)
            else:
                tpot_list = runtime_config.tpot if isinstance(runtime_config.tpot, list) else [runtime_config.tpot]
                for tpot in tpot_list:
                    overwritten_runtime_config = copy.deepcopy(runtime_config)
                    overwritten_runtime_config.tpot = tpot
                    runtime_configs_to_evaluate.append(overwritten_runtime_config)

            if not runtime_configs_to_evaluate:
                continue

            for overwritten_runtime_config in runtime_configs_to_evaluate:
                summary = sess.find_best_agg_result_under_constraints(
                    runtime_config=overwritten_runtime_config,
                    top_k=10,
                    max_batch_size=512,
                    ctx_stride=512,
                    enable_chunked_prefill=enable_chunked_prefill,
                    free_gpu_memory_fraction=free_gpu_memory_fraction,
                    max_seq_len=max_seq_len,
                )
                if not summary.check_oom():
                    all_configs_oom = False
                if not summary.check_kv_cache_oom():
                    all_kv_cache_oom = False
                result_df = summary.get_summary_df()
                if len(result_df) == 0:
                    logger.debug(
                        "No result found for constraints ttft=%s, tpot=%s, request_latency=%s in agg pareto.",
                        overwritten_runtime_config.ttft,
                        overwritten_runtime_config.tpot,
                        overwritten_runtime_config.request_latency,
                    )
                    continue
                if len(results_df) == 0:
                    results_df = result_df
                else:
                    results_df = pd.concat([results_df, result_df], axis=0, ignore_index=True)
        except Exception as e:
            logger.info(
                "Error getting candidate workers with parallel config: tp=%s, pp=%s, dp=%s, "
                "moe_tp=%s, moe_ep=%s, skip this combination",
                tp_size,
                pp_size,
                dp_size,
                moe_tp_size,
                moe_ep_size,
            )
            exceptions.append(e)
            continue

    if not results_df.empty:
        # Dedupe on numeric/identifier columns only; _per_ops_source holds dicts
        # which are unhashable and would break drop_duplicates with default subset.
        dedupe_cols = [c for c in results_df.columns if c != "_per_ops_source"]
        results_df = results_df.drop_duplicates(subset=dedupe_cols, ignore_index=True)
        results_df = results_df.sort_values(by="tokens/s/gpu", ascending=False).reset_index(drop=True)
    else:
        if exceptions:
            raise RuntimeError(
                f"No results found for any parallel configuration. Showing last exception: {exceptions[-1]}"
            ) from exceptions[-1]
        if all_configs_oom:
            raise RuntimeError(
                "No results found: the model does not fit in GPU memory for any parallel "
                "configuration. Try increasing --total-gpus, using a quantized model, or "
                "using a system with more VRAM per GPU."
            )
        if all_kv_cache_oom:
            raise RuntimeError(
                "No results found: the requested batch size exceeds KV cache capacity for all "
                "parallel configurations. Try reducing --batch-size, increasing "
                "--free-gpu-memory-fraction, or using a system with more VRAM per GPU."
            )
        raise NoFeasibleConfigError(
            "No results found for any parallel configuration. No configuration satisfied the "
            "TTFT/TPOT or request-latency constraints. Try relaxing --ttft, --tpot, or "
            "--request_latency (e.g., higher ttft/tpot or higher request_latency)."
        )

    return results_df


def disagg_pareto(
    model_path: str,
    runtime_config: config.RuntimeConfig,
    prefill_database: PerfDatabase,
    prefill_backend_name: str,
    prefill_model_config: config.ModelConfig,
    prefill_parallel_config_list: list[list[int]],
    prefill_latency_correction_scale: float,
    decode_database: PerfDatabase,
    decode_backend_name: str,
    decode_model_config: config.ModelConfig,
    decode_parallel_config_list: list[list[int]],
    decode_latency_correction_scale: float,
    **kwargs,
) -> pd.DataFrame:
    """
    Find Pareto front for Disaggregated Inference.
    This is a proxy function calls into
    DisaggInferenceSession.find_best    _disagg_result_under_constraints.

    Args:
        model_path: name of the model
        runtime_config: runtime config
        prefill_database: prefill database
        prefill_backend_name: prefill backend name
        prefill_model_config: prefill model config
        prefill_parallel_config_list: prefill parallel config list
        prefill_latency_correction_scale: prefill latency correction scale
        decode_database: decode database
        decode_backend_name: decode backend name
        decode_model_config: decode model config
        decode_parallel_config_list: decode parallel config list
        decode_latency_correction_scale: decode latency correction scale
        **kwargs: other arguments
        prefill_max_num_tokens: max number of tokens for prefill worker, in kwargs
        decode_max_num_tokens: max number of tokens for decode worker, in kwargs
        num_gpu_list: list of number of gpus in a disagg replica composed of xPyD, in kwargs
        max_num_gpu: max number of gpus in a disagg replica composed of xPyD, in kwargs
        prefill_num_worker_list: list of number of prefill workers in a disagg replica composed of
            xPyD, x_list, in kwargs
        prefill_max_num_worker: max number of prefill workers in a disagg replica composed of xPyD,
            x_max, in kwargs
        decode_num_worker_list: list of number of decode workers in a disagg replica composed of
            xPyD, y_list, in kwargs
        decode_max_num_worker: max number of decode workers in a disagg replica composed of xPyD,
            y_max, in kwargs

    Returns:
        results_df: dataframe of the results
    """

    def get_working_list(working_list, max_constraint):
        """
        Get working list based on max constraint. a helper function
        """
        if working_list is not None:
            if max_constraint is not None:
                working_list = [i for i in working_list if i <= max_constraint]
                logger.debug(f"{working_list} constrained by max_constraint: {max_constraint}")
            else:
                logger.debug(f"{working_list}")
        else:
            if max_constraint is not None:
                working_list = list(range(1, max_constraint + 1))
                logger.debug(f"{working_list} constrained by max_constraint: {max_constraint}")
            else:
                logger.debug(f"no constraint on {working_list}")
        return working_list

    prefill_backend = get_backend(prefill_backend_name)
    decode_backend = get_backend(decode_backend_name)

    encoder_database = kwargs.get("encoder_database")
    encoder_backend_name = kwargs.get("encoder_backend_name")
    encoder_backend = get_backend(encoder_backend_name) if encoder_backend_name else None

    disagg_sess = DisaggInferenceSession(
        prefill_database,
        prefill_backend,
        decode_database,
        decode_backend,
        encoder_database=encoder_database,
        encoder_backend=encoder_backend,
    )
    disagg_sess.set_latency_correction_scales(prefill_latency_correction_scale, decode_latency_correction_scale)

    # None means we use internally tuned default values for rate-matching degradation factors.
    rate_matching_prefill = kwargs.pop("rate_matching_prefill_degradation_factor", None)
    rate_matching_decode = kwargs.pop("rate_matching_decode_degradation_factor", None)
    if rate_matching_prefill is not None or rate_matching_decode is not None:
        kw = {}
        if rate_matching_prefill is not None:
            kw["prefill_degradation_factor"] = rate_matching_prefill
        if rate_matching_decode is not None:
            kw["decode_degradation_factor"] = rate_matching_decode
        disagg_sess.set_rate_matching_degradation_factors(**kw)

    prefill_max_num_tokens = kwargs.get("prefill_max_num_tokens", 16384)
    decode_max_num_tokens = kwargs.get("decode_max_num_tokens", 512)
    logger.debug(f"prefill_max_num_tokens: {prefill_max_num_tokens}, decode_max_num_tokens: {decode_max_num_tokens}")

    # num gpu constraint for the whole system
    num_gpu_list = kwargs.get("num_gpu_list")
    max_num_gpu = kwargs.get("max_num_gpu")
    logger.debug(f"num_gpu_list: {num_gpu_list}, max_num_gpu: {max_num_gpu}")
    num_gpu_list = get_working_list(num_gpu_list, max_num_gpu)

    # prefill worker constraint
    prefill_num_worker_list = kwargs.get("prefill_num_worker_list")
    prefill_max_num_worker = kwargs.get("prefill_max_num_worker")
    logger.debug(
        f"prefill_num_worker_list: {prefill_num_worker_list}, prefill_max_num_worker: {prefill_max_num_worker}"
    )
    prefill_num_worker_list = get_working_list(prefill_num_worker_list, prefill_max_num_worker)

    # decode worker constraint
    decode_num_worker_list = kwargs.get("decode_num_worker_list")
    decode_max_num_worker = kwargs.get("decode_max_num_worker")
    logger.debug(f"decode_num_worker_list: {decode_num_worker_list}, decode_max_num_worker: {decode_max_num_worker}")
    decode_num_worker_list = get_working_list(decode_num_worker_list, decode_max_num_worker)

    max_prefill_gpus = kwargs.get("max_prefill_gpus")
    max_decode_gpus = kwargs.get("max_decode_gpus")
    require_same_tp = kwargs.get("require_same_tp", False)
    autoscale = kwargs.get("autoscale", False)
    target_tpot = kwargs.get("target_tpot")

    summary = disagg_sess.find_best_disagg_result_under_constraints(
        model_path=model_path,
        runtime_config=runtime_config,
        prefill_model_config=prefill_model_config,
        prefill_parallel_config_list=prefill_parallel_config_list,
        prefill_max_num_tokens=prefill_max_num_tokens,
        prefill_num_worker_list=prefill_num_worker_list,
        decode_model_config=decode_model_config,
        decode_parallel_config_list=decode_parallel_config_list,
        decode_max_num_tokens=decode_max_num_tokens,
        decode_num_worker_list=decode_num_worker_list,
        num_gpu_list=num_gpu_list,
        max_prefill_gpus=max_prefill_gpus,
        max_decode_gpus=max_decode_gpus,
        require_same_tp=require_same_tp,
        autoscale=autoscale,
        target_tpot=target_tpot,
    )

    return summary.get_summary_df()


def _enumerate_afd_prefill_options(
    model_path: str,
    runtime_config: config.RuntimeConfig,
    database: PerfDatabase,
    backend_name: str,
    gpus_per_node: int,
    base_model_config: config.ModelConfig | None = None,
    quant_modes: dict | None = None,
    *,
    prefill_database: PerfDatabase | None = None,
    prefill_backend_name: str | None = None,
    prefill_model_config: config.ModelConfig | None = None,
    prefill_parallel_config_list: list[tuple[int, int, int, int, int]] | None = None,
    prefill_batch_size_list: list[int] | tuple[int, ...] | None = None,
    prefill_system_name: str | None = None,
    prefill_backend_version: str | None = None,
    total_gpus: int | None = None,
    max_prefill_gpus: int | None = None,
    max_candidates: int = _AFD_PREFILL_MAX_CANDIDATES,
    candidate_overflow: str = _AFD_PREFILL_CANDIDATE_OVERFLOW,
) -> list[dict]:
    """Estimate static prefill worker options for AFD combined-with-PD sweeps.

    AFD models the decode phase; the prefill phase runs on a separate
    static pool. The static prefill estimate depends on the prefill
    worker's own parallelism and batch size, so this search is computed
    once and shared across all AFD decode candidates.

    ``base_model_config`` carries the resolved model/backend semantics
    from the owning task. ``prefill_model_config`` overrides it when
    AFD is configured with a separate prefill worker config. ``quant_modes``
    is kept for compatibility with older direct callers.

    Returns a list of dicts with prefill parallelism, per-worker GPU
    count, batch size, TTFT, ``seq_s`` (per-worker), memory and power.
    """
    is_moe = check_is_moe(model_path)
    effective_database = prefill_database or database
    effective_backend_name = prefill_backend_name or backend_name
    backend = get_backend(effective_backend_name)
    quant_modes = dict(quant_modes or {})
    base_model_config = copy.deepcopy(
        prefill_model_config if prefill_model_config is not None else base_model_config
    ) if (prefill_model_config is not None or base_model_config is not None) else config.ModelConfig()
    for key, value in quant_modes.items():
        if value is not None:
            setattr(base_model_config, key, value)

    batch_size_list = _normalize_positive_int_list(
        "prefill_batch_size_list",
        prefill_batch_size_list or _AFD_PREFILL_BATCH_SIZE_LIST,
    )
    if not batch_size_list:
        batch_size_list = list(_AFD_PREFILL_BATCH_SIZE_LIST)

    if prefill_parallel_config_list is None:
        tp_candidates = sorted({tp for tp in (1, 2, 4, 8) if tp <= max(gpus_per_node, 1)})
        prefill_parallel_config_list = [
            (tp, 1, 1, 1 if is_moe else tp, tp if is_moe else 1) for tp in tp_candidates
        ]

    if max_candidates < 1:
        raise ValueError(f"afd_config.prefill_search.max_candidates must be >= 1, got {max_candidates}.")
    if candidate_overflow not in {"error", "truncate"}:
        raise ValueError("afd_config.prefill_search.candidate_overflow must be 'error' or 'truncate'.")

    candidates: list[tuple[int, int, int, int, int, int, int, int]] = []
    for parallel_config in prefill_parallel_config_list:
        values = [int(value) for value in parallel_config]
        if len(values) == 5:
            tp, pp, dp, moe_tp, moe_ep = values
            cp = 1
        elif len(values) == 6:
            tp, pp, dp, moe_tp, moe_ep, cp = values
        else:
            raise ValueError(
                "afd_config.prefill_search.parallel_config_list entries must contain "
                f"5 or 6 integers, got {len(values)}: {parallel_config!r}."
            )
        num_gpus = tp * pp * dp * cp
        if total_gpus is not None and total_gpus > 0 and num_gpus > total_gpus:
            continue
        if max_prefill_gpus is not None and max_prefill_gpus > 0 and num_gpus > max_prefill_gpus:
            continue
        for batch_size in batch_size_list:
            candidates.append((tp, pp, dp, moe_tp, moe_ep, cp, num_gpus, batch_size))

    if len(candidates) > max_candidates:
        message = (
            f"AFD static prefill search produced {len(candidates)} candidates, exceeding "
            f"afd_config.prefill_search.max_candidates={max_candidates}."
        )
        if candidate_overflow == "truncate":
            logger.warning("%s Truncating deterministically to the first %d candidates.", message, max_candidates)
            candidates = candidates[:max_candidates]
        else:
            raise ValueError(f"{message} Reduce afd_config.prefill_search or set candidate_overflow='truncate'.")

    options: list[dict] = []
    for tp, pp, dp, moe_tp, moe_ep, cp, num_gpus, batch_size in candidates:
        try:
            model_config = copy.deepcopy(base_model_config)
            model_config.tp_size = tp
            model_config.pp_size = pp
            model_config.attention_dp_size = dp
            model_config.moe_tp_size = moe_tp
            model_config.moe_ep_size = moe_ep
            model_config.cp_size = cp
            # AFD static prefill pool runs context attention (static_ctx); align with
            # disagg prefill (cli/api.py) by downgrading fp8 fmha -> bfloat16 for
            # DeepSeek-V3/Kimi so sglang context_mla (which lacks the fp8 slice) is hit.
            resolve_context_fmha_compat(model_config, model_path, is_context_role=True)
            model = get_model(model_path, model_config, effective_backend_name)
            sess = InferenceSession(model=model, database=effective_database, backend=backend)
            prefill_runtime_config = copy.deepcopy(runtime_config)
            prefill_runtime_config.batch_size = batch_size
            summary = sess.run_static(runtime_config=prefill_runtime_config, mode="static_ctx")
            if summary.check_oom():
                continue
            result_dict = summary.get_result_dict()
            ttft = float(result_dict.get("ttft", 0.0) or 0.0)
            seq_s = float(result_dict.get("seq/s", 0.0) or 0.0)
            if ttft <= 0.0 or seq_s <= 0.0:
                continue
            option_num_gpus = int(result_dict.get("num_total_gpus", num_gpus) or num_gpus)
            if total_gpus is not None and total_gpus > 0 and option_num_gpus > total_gpus:
                continue
            if max_prefill_gpus is not None and max_prefill_gpus > 0 and option_num_gpus > max_prefill_gpus:
                continue
            options.append(
                {
                    "tp": tp,
                    "pp": pp,
                    "dp": dp,
                    "moe_tp": moe_tp,
                    "moe_ep": moe_ep,
                    "cp": cp,
                    "batch_size": batch_size,
                    "num_gpus": option_num_gpus,
                    "ttft": ttft,
                    "seq_s": seq_s,
                    "memory": float(result_dict.get("memory", 0.0) or 0.0),
                    "power_w": float(result_dict.get("power_w", 0.0) or 0.0),
                    "system": result_dict.get("system", prefill_system_name),
                    "backend": result_dict.get("backend", effective_backend_name),
                    "version": result_dict.get("version", prefill_backend_version),
                }
            )
        except Exception:
            logger.debug(
                "AFD prefill option tp=%d pp=%d dp=%d moe_tp=%d ep=%d cp=%d bs=%d failed, skipping",
                tp,
                pp,
                dp,
                moe_tp,
                moe_ep,
                cp,
                batch_size,
                exc_info=True,
            )
            continue
    return options


def _combine_afd_row_with_static_prefill(
    row: dict,
    prefill_options: list[dict],
    *,
    target_ttft: float | None = None,
    target_request_latency: float | None = None,
    total_gpus: int | None = None,
    max_prefill_gpus: int | None = None,
    max_prefill_workers: int | None = None,
    prefill_degradation: float = _AFD_PREFILL_DEGRADATION,
    decode_degradation: float = _AFD_DECODE_DEGRADATION,
    ttft_correction_factor: float = _AFD_TTFT_CORRECTION_FACTOR,
) -> dict | None:
    """Merge an AFD decode row with a rate-matched static prefill pool.

    Picks the feasible prefill option that minimizes the prefill GPU count
    needed to keep up with the AFD decode rate, then merges TTFT/TPOT,
    throughput, GPU budget, and memory into a single combined row.  Applies
    degradation factors and TTFT correction symmetrically with the disagg
    rate-matching path.

    Returns ``None`` when no prefill option satisfies the hard TTFT/request
    latency constraints.
    """
    if not prefill_options:
        return None

    decode_seq_s = float(row.get("seq/s", 0.0) or 0.0)
    if decode_seq_s <= 0.0:
        return None

    osl = int(row.get("osl", 1) or 1)
    tpot = float(row.get("tpot", 0.0) or 0.0)
    decode_time = tpot * max(osl - 1, 0)
    decode_gpus = int(row.get("num_total_gpus", 0) or 0)
    best_key = None
    best_option = None
    best_workers = 0
    for option in prefill_options:
        # Apply prefill degradation to per-worker throughput for rate-matching
        effective_per_worker = option["seq_s"] * prefill_degradation
        if effective_per_worker <= 0.0:
            continue
        num_workers = max(1, math.ceil(decode_seq_s * decode_degradation / effective_per_worker))
        if max_prefill_workers is not None and max_prefill_workers > 0 and num_workers > max_prefill_workers:
            continue
        prefill_gpus = num_workers * option["num_gpus"]
        if max_prefill_gpus is not None and max_prefill_gpus > 0 and prefill_gpus > max_prefill_gpus:
            continue
        num_total_gpus = decode_gpus + prefill_gpus
        if total_gpus is not None and total_gpus > 0 and num_total_gpus > total_gpus:
            continue
        # TTFT check uses corrected TTFT (concurrent queueing)
        corrected_ttft = option["ttft"] * ttft_correction_factor
        if target_ttft is not None and target_ttft > 0 and corrected_ttft > target_ttft:
            continue
        request_latency = corrected_ttft + decode_time
        if (
            target_request_latency is not None
            and target_request_latency > 0
            and request_latency > target_request_latency
        ):
            continue
        key = (num_total_gpus, prefill_gpus, corrected_ttft)
        if best_key is None or key < best_key:
            best_key = key
            best_option = option
            best_workers = num_workers

    if best_option is None:
        return None

    ttft = best_option["ttft"] * ttft_correction_factor
    # Rate-matched throughput with degradation factors
    effective_prefill_seq_s = best_workers * best_option["seq_s"] * prefill_degradation
    effective_decode_seq_s = decode_seq_s * decode_degradation
    seq_s = min(effective_decode_seq_s, effective_prefill_seq_s)
    tokens_s = seq_s * osl
    request_latency = ttft + decode_time
    num_total_gpus = decode_gpus + best_workers * best_option["num_gpus"]
    decode_power = float(row.get("power_w", 0.0) or 0.0)
    power_w = (
        (best_option["power_w"] * ttft + decode_power * decode_time) / request_latency if request_latency > 0.0 else 0.0
    )

    combined = dict(row)
    combined.update(
        {
            "ttft": round(ttft, 3),
            "tpot": round(tpot, 3),
            "request_latency": round(request_latency, 3),
            "seq/s": round(seq_s, 3),
            "request_rate": round(seq_s, 3),
            "tokens/s": round(tokens_s, 2),
            "tokens/s/gpu": round(tokens_s / num_total_gpus, 2) if num_total_gpus > 0 else 0.0,
            "tokens/s/user": round(1000.0 / tpot, 2) if tpot > 0.0 else 0.0,
            "num_total_gpus": num_total_gpus,
            "memory": round(max(float(row.get("memory", 0.0) or 0.0), best_option["memory"]), 2),
            "power_w": round(power_w, 3),
            "combined_with_pd": True,
            "(p)workers": best_workers,
            "(p)tp": best_option["tp"],
            "(p)pp": best_option.get("pp", 1),
            "(p)dp": best_option.get("dp", 1),
            "(p)moe_tp": best_option.get("moe_tp", best_option["tp"]),
            "(p)ep": best_option.get("moe_ep", 1),
            "(p)bs": best_option.get("batch_size", 1),
            "(p)num_gpus": best_option["num_gpus"],
            "(p)system": best_option.get("system"),
            "(p)backend": best_option.get("backend"),
            "(p)version": best_option.get("version"),
            "(p)impl": "static_ctx",
            "(d)impl": "afd",
        }
    )
    return combined


def _analytical_max_batch_size(
    backend,
    model,
    database: PerfDatabase,
    partition_ops,
    *,
    isl: int,
    osl: int,
    prefix: int,
    max_seq_len: int | None,
    include_kvcache: bool,
    kvcache_multiplier: int = 1,
    free_gpu_memory_fraction: float | None = None,
    align_to: int = 8,
) -> int:
    """Compute the maximum batch_size that fits in GPU HBM analytically.

    The memory model is linear in ``batch_size``:
    ``total = fixed + batch_size * marginal``.  This function samples
    ``get_partition_memory_usage`` at two reference points to extract
    the linear coefficients, then solves for the largest ``batch_size``
    satisfying both the absolute HBM capacity and the KV-cache fraction
    budget.  The result is aligned down to ``align_to``.

    Using two-point sampling (rather than replicating the formula) keeps
    the function automatically compatible with backend-specific overrides
    (SGLang overhead fracs, MoE workspace, MTP correction, etc.).
    """
    effective_max_seq_len = max_seq_len if max_seq_len is not None else isl + osl

    def _sample(bs: int) -> dict[str, float]:
        return backend.get_partition_memory_usage(
            model,
            database,
            partition_ops=partition_ops,
            batch_size=bs,
            beam_width=1,
            isl=isl,
            osl=osl,
            num_tokens=bs,
            prefix=prefix,
            max_seq_len=effective_max_seq_len,
            include_kvcache=include_kvcache,
            kvcache_multiplier=kvcache_multiplier,
        )

    _BS_LO, _BS_HI = 128, 256
    try:
        mem_lo = _sample(_BS_LO)
        mem_hi = _sample(_BS_HI)
    except Exception:
        logger.debug("_analytical_max_batch_size: sampling failed", exc_info=True)
        return 0

    delta_bs = _BS_HI - _BS_LO
    marginal_gib = (mem_hi["total"] - mem_lo["total"]) / delta_bs
    fixed_gib = mem_lo["total"] - marginal_gib * _BS_LO

    gpu_cap_gib = database.system_spec["gpu"]["mem_capacity"] / (1 << 30)

    if marginal_gib <= 0:
        max_bs = 10000
    else:
        max_bs = int((gpu_cap_gib - fixed_gib) / marginal_gib)

    if free_gpu_memory_fraction is not None and include_kvcache:
        reserved, tolerance = backend.get_kv_cache_memory_check_params()
        frac = free_gpu_memory_fraction * (1.0 - reserved) * (1.0 - tolerance)
        kv_marginal = (mem_hi["kvcache"] - mem_lo["kvcache"]) / delta_bs
        act_marginal = marginal_gib - kv_marginal
        denom = kv_marginal + frac * act_marginal
        if denom > 0:
            max_bs_kv = int(frac * (gpu_cap_gib - fixed_gib) / denom)
            max_bs = min(max_bs, max_bs_kv)

    max_bs = max(max_bs, 0)
    if align_to > 1:
        max_bs = (max_bs // align_to) * align_to
    return max_bs


def _afd_total_batch_capacity(
    max_micro_batch_size: int,
    num_microbatches: int,
    *,
    align_to: int = 1,
) -> int:
    """Convert an AFD per-execution microbatch capacity to total A batch."""
    max_total_batch_size = max(int(max_micro_batch_size), 0) * max(int(num_microbatches or 1), 1)
    if align_to > 1:
        max_total_batch_size = (max_total_batch_size // align_to) * align_to
    return max_total_batch_size


def _derive_a_batch_size(
    model_path: str,
    a_model_config: config.ModelConfig,
    backend: object,
    database: PerfDatabase,
    *,
    num_microbatches: int,
    boundary_on_attn: bool,
    isl: int,
    osl: int,
    prefix: int,
    max_seq_len: int | None,
    free_gpu_memory_fraction: float | None,
) -> tuple[int, object, object]:
    """Derive ``a_batch_size`` from A-pool KV-cache capacity.

    Analytically computes the maximum per-execution microbatch whose A-pool
    memory fits within the GPU HBM budget (both absolute capacity and KV-cache
    fraction constraints), then converts it to the total in-flight batch over
    all microbatches. The total result is aligned down to a multiple of 8,
    capped at 256, and floored at 32.

    Falls back to 32 when the analytical result is below 32; the caller's
    per-candidate OOM check will then skip that topology naturally.

    Returns (batch_size, a_model, a_partition) so the caller can reuse
    the already-constructed model and partition for the balance-ratio probe.
    """
    from aiconfigurator.sdk.afd_partition import build_afd_ops_partition

    a_model = get_model(model_path, a_model_config, backend.name.value)
    a_partition = build_afd_ops_partition(a_model, phase="generation", boundary_on_attn=boundary_on_attn)

    kvcache_multiplier = max(int(num_microbatches or 1), 1)

    max_micro_bs = _analytical_max_batch_size(
        backend,
        a_model,
        database,
        a_partition.attn_ops,
        isl=isl,
        osl=osl,
        prefix=prefix,
        max_seq_len=max_seq_len,
        include_kvcache=True,
        kvcache_multiplier=kvcache_multiplier,
        free_gpu_memory_fraction=free_gpu_memory_fraction,
        align_to=1,
    )
    max_total_bs = _afd_total_batch_capacity(max_micro_bs, kvcache_multiplier, align_to=8)

    return max(min(max_total_bs, 256), 32), a_model, a_partition


_AFD_BALANCE_RATIO_THRESHOLD = 0.3


def _quick_balance_ratio(
    a_ops,
    f_ops,
    database: PerfDatabase,
    *,
    batch_size: int,
    seq_len: int,
    runtime_config,
    a_model,
    f_model,
) -> float:
    """Single-point latency probe to estimate A/F balance ratio cheaply."""
    kwargs_base = {
        "batch_size": batch_size,
        "beam_width": 1,
        "s": seq_len,
        "prefix": runtime_config.prefix,
        "gen_seq_imbalance_correction_scale": runtime_config.gen_seq_imbalance_correction_scale,
    }
    t_a = sum(
        float(op.query(database, x=batch_size, model_name=getattr(a_model, "model_name", ""), **kwargs_base))
        for op in a_ops
    )
    t_f = sum(
        float(op.query(database, x=batch_size, model_name=getattr(f_model, "model_name", ""), **kwargs_base))
        for op in f_ops
    )
    return min(t_a, t_f) / max(t_a, t_f, 1e-9)


def _afd_runtime_configs_for_sla(runtime_config: config.RuntimeConfig) -> list[config.RuntimeConfig]:
    runtime_configs: list[config.RuntimeConfig] = []
    if runtime_config.request_latency is not None and runtime_config.request_latency > 0:
        ttft_tpot_constraints = enumerate_ttft_tpot_constraints(
            runtime_config.osl,
            runtime_config.request_latency,
            runtime_config.ttft,
        )
        for ttft_constraint, tpot_constraint in ttft_tpot_constraints:
            overwritten_runtime_config = copy.deepcopy(runtime_config)
            overwritten_runtime_config.ttft = ttft_constraint
            overwritten_runtime_config.tpot = tpot_constraint
            runtime_configs.append(overwritten_runtime_config)
        return runtime_configs

    tpot_values = runtime_config.tpot if isinstance(runtime_config.tpot, list) else [runtime_config.tpot]
    for tpot in tpot_values:
        overwritten_runtime_config = copy.deepcopy(runtime_config)
        overwritten_runtime_config.tpot = tpot
        runtime_configs.append(overwritten_runtime_config)
    return runtime_configs


def _finite_float(value, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _recompute_afd_decode_rate_fields(row: dict) -> None:
    tpot = _finite_float(row.get("tpot"), 0.0)
    ttft = _finite_float(row.get("ttft"), 0.0)
    osl = max(int(_finite_float(row.get("osl"), 1.0)), 1)
    b_total = _finite_float(row.get("b_total", row.get("concurrency", 0.0)), 0.0)
    total_gpus = _finite_float(row.get("num_total_gpus"), 0.0)

    tokens_per_s = b_total / (tpot / 1000.0) if tpot > 0.0 and b_total > 0.0 else 0.0
    seq_per_s = tokens_per_s / osl if tokens_per_s > 0.0 else 0.0

    row["request_latency"] = round(ttft + tpot * max(osl - 1, 0), 3)
    row["tokens/s"] = round(tokens_per_s, 2)
    row["seq/s"] = round(seq_per_s, 3)
    row["request_rate"] = round(seq_per_s, 3)
    row["tokens/s/gpu"] = round(tokens_per_s / total_gpus, 2) if total_gpus > 0.0 else 0.0
    row["tokens/s/user"] = round(1000.0 / tpot, 2) if tpot > 0.0 else 0.0


def _apply_afd_decode_latency_correction(row: dict, decode_latency_correction: float) -> None:
    if decode_latency_correction == 1.0:
        return
    tpot = _finite_float(row.get("tpot"), 0.0) * decode_latency_correction
    row["tpot"] = round(tpot, 3)
    if "decode_t_step" in row:
        row["decode_t_step"] = round(tpot, 3)
    if row.get("phase") == "decode" and "t_step" in row:
        row["t_step"] = round(tpot, 3)
    _recompute_afd_decode_rate_fields(row)


def _afd_sla_rejection_reason(
    row: dict,
    runtime_config: config.RuntimeConfig,
    *,
    require_ttft: bool,
) -> str | None:
    target_tpot = runtime_config.tpot
    if target_tpot is not None and float(row.get("tpot", 0.0) or 0.0) > float(target_tpot):
        return "tpot"

    target_ttft = runtime_config.ttft
    if (
        require_ttft
        and target_ttft is not None
        and target_ttft > 0
        and float(row.get("ttft", 0.0) or 0.0) > float(target_ttft)
    ):
        return "ttft"

    target_request_latency = runtime_config.request_latency
    if (
        target_request_latency is not None
        and target_request_latency > 0
        and float(row.get("request_latency", 0.0) or 0.0) > float(target_request_latency)
    ):
        return "request_latency"

    return None


def _format_afd_rejection_summary(rejection_counts: dict[str, int]) -> str:
    return ", ".join(f"{key}={value}" for key, value in rejection_counts.items() if value)


def afd_pareto(
    model_path: str,
    runtime_config: config.RuntimeConfig,
    database: PerfDatabase,
    backend_name: str,
    afd_parallel_config_list: list[tuple[int, int, int, int, int, str]],
    gpus_per_node: int,
    *,
    model_config: config.ModelConfig | None = None,
    total_gpus: int | None = None,
    total_batch_size: int | None = None,
    combined_with_pd: bool = True,
    comm_overhead_factor: float = 1.0,
    boundary_on_attn: bool = True,
    target_ttft: float | None = None,
    free_gpu_memory_fraction: float | None = None,
    max_seq_len: int | None = None,
    quant_modes: dict | None = None,
    prefill_database: PerfDatabase | None = None,
    prefill_backend_name: str | None = None,
    prefill_model_config: config.ModelConfig | None = None,
    prefill_parallel_config_list: list[tuple[int, int, int, int, int]] | None = None,
    prefill_batch_size_list: list[int] | None = None,
    prefill_system_name: str | None = None,
    prefill_backend_version: str | None = None,
    prefill_max_candidates: int = _AFD_PREFILL_MAX_CANDIDATES,
    prefill_candidate_overflow: str = _AFD_PREFILL_CANDIDATE_OVERFLOW,
    max_prefill_gpus: int | None = None,
    max_prefill_workers: int | None = None,
    prefill_degradation: float = _AFD_PREFILL_DEGRADATION,
    decode_degradation: float = _AFD_DECODE_DEGRADATION,
    ttft_correction_factor: float = _AFD_TTFT_CORRECTION_FACTOR,
    decode_latency_correction: float = 1.0,
) -> pd.DataFrame:
    """Sweep AFD candidate topologies and collect per-candidate estimates.

    Each candidate is a ``(n_a_nodes, n_f_nodes, tp_a, f_moe_ep_size,
    num_microbatches, pipeline_model)`` tuple (see
    ``task.build_afd_parallel_lists``).  ``a_batch_size`` is derived
    per-candidate from the A-pool KV-cache capacity via
    :func:`_derive_a_batch_size` instead of being enumerated unless
    ``total_batch_size`` is provided.  Fixed total batch mode requires
    exact divisibility by the candidate's A-worker count and evaluates
    that one exact batch size without lowering it.  For each candidate
    the AFD decode phase is estimated via
    :class:`AFDInferenceSession`; when ``combined_with_pd`` is set, the
    result is merged with a rate-matched static prefill pool so the row
    carries end-to-end TTFT + TPOT and the full GPU budget.

    ``model_config`` carries the fully resolved model/backend semantics
    from the owning task. A/F candidates and the static prefill pool
    deep-copy it and only override candidate parallelism fields.
    ``quant_modes`` is kept for compatibility with older direct callers.

    OOM candidates and per-candidate failures are skipped. Returns a
    DataFrame with :data:`common.ColumnsAFD` columns sorted by
    ``tokens/s/gpu``.
    """
    from aiconfigurator.sdk.afd_partition import build_afd_ops_partition
    from aiconfigurator.sdk.config import AFDConfig

    fixed_total_batch_size = None
    if total_batch_size is not None:
        if isinstance(total_batch_size, bool) or not isinstance(total_batch_size, int):
            raise ValueError(f"total_batch_size must be a positive integer, got {total_batch_size!r}.")
        fixed_total_batch_size = total_batch_size
        if fixed_total_batch_size < 1:
            raise ValueError(f"total_batch_size must be a positive integer, got {total_batch_size!r}.")

    backend = get_backend(backend_name)
    quant_modes = dict(quant_modes or {})
    base_model_config = copy.deepcopy(model_config) if model_config is not None else config.ModelConfig()
    for key, value in quant_modes.items():
        if value is not None:
            setattr(base_model_config, key, value)

    prefill_options: list[dict] = []
    if combined_with_pd:
        prefill_options = _enumerate_afd_prefill_options(
            model_path=model_path,
            runtime_config=runtime_config,
            database=database,
            backend_name=backend_name,
            gpus_per_node=gpus_per_node,
            base_model_config=base_model_config,
            quant_modes=quant_modes,
            prefill_database=prefill_database,
            prefill_backend_name=prefill_backend_name,
            prefill_model_config=prefill_model_config,
            prefill_parallel_config_list=prefill_parallel_config_list,
            prefill_batch_size_list=prefill_batch_size_list,
            prefill_system_name=prefill_system_name,
            prefill_backend_version=prefill_backend_version,
            total_gpus=total_gpus,
            max_prefill_gpus=max_prefill_gpus,
            max_candidates=prefill_max_candidates,
            candidate_overflow=prefill_candidate_overflow,
        )
        if not prefill_options:
            raise NoFeasibleConfigError(
                "AFD combined_with_pd=True requires at least one feasible static prefill option. "
                "No static prefill option satisfied the estimate, so no decode-only AFD row was returned."
            )

    base_runtime_config = copy.deepcopy(runtime_config)
    if target_ttft is not None:
        base_runtime_config.ttft = target_ttft
    runtime_configs_to_evaluate = _afd_runtime_configs_for_sla(base_runtime_config)
    if not runtime_configs_to_evaluate:
        raise NoFeasibleConfigError(
            "No AFD SLA constraint pairs could be derived for "
            f"request_latency={runtime_config.request_latency}."
        )

    rows: list[dict] = []
    rejection_counts = {
        "oom": 0,
        "fixed_batch": 0,
        "tpot": 0,
        "ttft": 0,
        "request_latency": 0,
        "prefill_combine": 0,
        "gpu_budget": 0,
        "balance": 0,
        "low_batch_oom": 0,
    }
    exceptions: list[Exception] = []
    # Track topologies that OOMed at a given microbatch count; higher counts
    # use a larger KV-cache multiplier and can be pruned for the same topology.
    _oom_at_mb: dict[tuple[int, int, int, int], int] = {}
    for eval_runtime_config in runtime_configs_to_evaluate:
        target_tpot = eval_runtime_config.tpot
        for candidate in afd_parallel_config_list:
            n_a_nodes, n_f_nodes, tp_a, f_moe_ep_size, num_microbatches, pipeline_model = candidate
            topo_key = (int(n_a_nodes), int(n_f_nodes), int(tp_a), int(f_moe_ep_size))
            oom_mb = _oom_at_mb.get(topo_key)
            if oom_mb is not None and int(num_microbatches) >= oom_mb:
                rejection_counts["oom"] += 1
                continue

            try:
                # --- Topology-level checks (independent of a_batch_size) ---
                tp_f = int(n_f_nodes) * int(gpus_per_node)
                if int(f_moe_ep_size) <= 0 or tp_f % int(f_moe_ep_size) != 0:
                    continue
                f_moe_tp = tp_f // int(f_moe_ep_size)
                n_a_workers = (int(n_a_nodes) * int(gpus_per_node)) // int(tp_a)

                a_model_config = copy.deepcopy(base_model_config)
                a_model_config.tp_size = int(tp_a)
                a_model_config.pp_size = 1
                a_model_config.moe_tp_size = int(tp_a)
                a_model_config.moe_ep_size = 1
                a_model_config.attention_dp_size = 1

                f_model_config = copy.deepcopy(base_model_config)
                f_model_config.tp_size = tp_f
                f_model_config.pp_size = 1
                f_model_config.moe_tp_size = f_moe_tp
                f_model_config.moe_ep_size = int(f_moe_ep_size)
                f_model_config.attention_dp_size = 1

                # --- A-Worker: analytical max batch_size ---
                if fixed_total_batch_size is None:
                    derived_bs, a_model, a_partition = _derive_a_batch_size(
                        model_path,
                        a_model_config,
                        backend,
                        database,
                        num_microbatches=num_microbatches,
                        boundary_on_attn=boundary_on_attn,
                        isl=eval_runtime_config.isl,
                        osl=eval_runtime_config.osl,
                        prefix=eval_runtime_config.prefix or 0,
                        max_seq_len=max_seq_len,
                        free_gpu_memory_fraction=free_gpu_memory_fraction,
                    )
                else:
                    if n_a_workers <= 0 or fixed_total_batch_size % n_a_workers != 0:
                        rejection_counts["fixed_batch"] += 1
                        logger.debug(
                            "AFD candidate a%dxf%d tp_a=%d ep=%d mb=%d pipe=%s: "
                            "total_batch_size=%d is not exactly divisible by n_a_workers=%d, skipping",
                            n_a_nodes,
                            n_f_nodes,
                            tp_a,
                            f_moe_ep_size,
                            num_microbatches,
                            pipeline_model,
                            fixed_total_batch_size,
                            n_a_workers,
                        )
                        continue
                    fixed_a_batch_size = fixed_total_batch_size // n_a_workers
                    if fixed_a_batch_size < 1:
                        rejection_counts["fixed_batch"] += 1
                        logger.debug(
                            "AFD candidate a%dxf%d tp_a=%d ep=%d mb=%d pipe=%s: "
                            "derived fixed a_batch_size=%d < 1, skipping",
                            n_a_nodes,
                            n_f_nodes,
                            tp_a,
                            f_moe_ep_size,
                            num_microbatches,
                            pipeline_model,
                            fixed_a_batch_size,
                        )
                        continue
                    a_model = get_model(model_path, a_model_config, backend.name.value)
                    a_partition = build_afd_ops_partition(
                        a_model,
                        phase="generation",
                        boundary_on_attn=boundary_on_attn,
                    )
                    max_bs_a_micro = _analytical_max_batch_size(
                        backend,
                        a_model,
                        database,
                        a_partition.attn_ops,
                        isl=eval_runtime_config.isl,
                        osl=eval_runtime_config.osl,
                        prefix=eval_runtime_config.prefix or 0,
                        max_seq_len=max_seq_len,
                        include_kvcache=True,
                        kvcache_multiplier=max(int(num_microbatches or 1), 1),
                        free_gpu_memory_fraction=free_gpu_memory_fraction,
                        align_to=1,
                    )
                    derived_bs = _afd_total_batch_capacity(
                        max_bs_a_micro,
                        num_microbatches,
                    )

                # --- F-Worker: analytical max batch_size ---
                f_model = get_model(model_path, f_model_config, backend.name.value)
                f_partition = build_afd_ops_partition(
                    f_model,
                    phase="generation",
                    boundary_on_attn=boundary_on_attn,
                )

                max_bs_f_micro = _analytical_max_batch_size(
                    backend,
                    f_model,
                    database,
                    f_partition.ffn_ops,
                    isl=eval_runtime_config.isl,
                    osl=eval_runtime_config.osl,
                    prefix=eval_runtime_config.prefix or 0,
                    max_seq_len=max_seq_len,
                    include_kvcache=False,
                    kvcache_multiplier=1,
                    free_gpu_memory_fraction=None,
                    align_to=1,
                )
                nm = max(int(num_microbatches or 1), 1)
                if n_a_workers > 0 and max_bs_f_micro > 0:
                    max_bs_f = _afd_total_batch_capacity(
                        max_bs_f_micro // n_a_workers,
                        nm,
                    )
                else:
                    max_bs_f = 0

                candidate_rows: list[dict] = []
                if fixed_total_batch_size is not None:
                    a_batch_size = fixed_a_batch_size
                    if derived_bs < a_batch_size or max_bs_f < a_batch_size:
                        rejection_counts["oom"] += 1
                        logger.debug(
                            "AFD candidate a%dxf%d tp_a=%d ep=%d mb=%d pipe=%s: "
                            "fixed a_batch_size=%d exceeds analytical capacity (A=%d, F=%d), skipping",
                            n_a_nodes,
                            n_f_nodes,
                            tp_a,
                            f_moe_ep_size,
                            num_microbatches,
                            pipeline_model,
                            a_batch_size,
                            derived_bs,
                            max_bs_f,
                        )
                        continue

                    afd_config = AFDConfig(
                        n_a_nodes=int(n_a_nodes),
                        n_f_nodes=int(n_f_nodes),
                        gpus_per_node=int(gpus_per_node),
                        tp_a=int(tp_a),
                        f_moe_ep_size=int(f_moe_ep_size),
                        a_batch_size=a_batch_size,
                        num_microbatches=int(num_microbatches),
                        pipeline_model=str(pipeline_model),
                        comm_overhead_factor=float(comm_overhead_factor),
                        phase="decode",
                        combined_with_pd=bool(combined_with_pd),
                        boundary_on_attn=bool(boundary_on_attn),
                    )
                    candidate_runtime_config = copy.deepcopy(eval_runtime_config)
                    candidate_runtime_config.batch_size = afd_config.n_a_workers * afd_config.a_batch_size

                    session = AFDInferenceSession(
                        model_path=model_path,
                        a_model_config=a_model_config,
                        f_model_config=f_model_config,
                        database=database,
                        backend=backend,
                        afd_config=afd_config,
                    )
                    summary = session.run_afd(
                        candidate_runtime_config,
                        phase="decode",
                        free_gpu_memory_fraction=free_gpu_memory_fraction,
                        max_seq_len=max_seq_len,
                    )
                    if summary.check_oom():
                        rejection_counts["oom"] += 1
                        continue

                    best_row = dict(summary.get_result_dict())
                    _apply_afd_decode_latency_correction(best_row, decode_latency_correction)
                    row_tpot = float(best_row.get("tpot", 0.0) or 0.0)
                    if target_tpot is not None and row_tpot > float(target_tpot):
                        rejection_counts["tpot"] += 1
                        logger.debug(
                            "AFD candidate a%dxf%d tp_a=%d ep=%d mb=%d pipe=%s fixed bs=%d "
                            "TPOT=%.1fms > %.1fms, skipping",
                            n_a_nodes,
                            n_f_nodes,
                            tp_a,
                            f_moe_ep_size,
                            num_microbatches,
                            pipeline_model,
                            a_batch_size,
                            row_tpot,
                            target_tpot,
                        )
                        best_row = None
                    if best_row is not None:
                        candidate_rows.append(best_row)
                else:
                    best_row = None
                    candidate_rejected_by_tpot = False

                    raw_combined_max_bs = min(derived_bs, max_bs_f) if max_bs_f > 0 else derived_bs
                    combined_max_bs = (raw_combined_max_bs // 8) * 8
                    low_latency_batch_sizes = [
                        batch_size
                        for batch_size in _AFD_LOW_LATENCY_BATCH_SIZE_LIST
                        if batch_size <= raw_combined_max_bs
                    ]
                    max_probe_batch_size = max([combined_max_bs, *low_latency_batch_sizes], default=0)
                    if max_probe_batch_size < 1:
                        rejection_counts["oom"] += 1
                        if topo_key not in _oom_at_mb:
                            _oom_at_mb[topo_key] = int(num_microbatches)
                        logger.debug(
                            "AFD candidate a%dxf%d tp_a=%d ep=%d mb=%d pipe=%s: "
                            "analytical max_bs=%d (A=%d, F=%d) < 1, skipping",
                            n_a_nodes,
                            n_f_nodes,
                            tp_a,
                            f_moe_ep_size,
                            num_microbatches,
                            pipeline_model,
                            raw_combined_max_bs,
                            derived_bs,
                            max_bs_f,
                        )
                        continue

                    # --- Quick balance_ratio pre-filter ---
                    probe_bs = min(32, max_probe_batch_size)
                    probe_s = eval_runtime_config.isl + (eval_runtime_config.osl // 2)
                    quick_ratio = _quick_balance_ratio(
                        a_partition.attn_ops,
                        f_partition.ffn_ops,
                        database,
                        batch_size=probe_bs,
                        seq_len=probe_s,
                        runtime_config=eval_runtime_config,
                        a_model=a_model,
                        f_model=f_model,
                    )
                    if quick_ratio < _AFD_BALANCE_RATIO_THRESHOLD:
                        rejection_counts["balance"] += 1
                        logger.debug(
                            "AFD candidate a%dxf%d tp_a=%d ep=%d mb=%d pipe=%s: "
                            "balance_ratio=%.3f < %.1f, pruned",
                            n_a_nodes,
                            n_f_nodes,
                            tp_a,
                            f_moe_ep_size,
                            num_microbatches,
                            pipeline_model,
                            quick_ratio,
                            _AFD_BALANCE_RATIO_THRESHOLD,
                        )
                        continue

                    def _evaluate_auto_batch(a_batch_size: int) -> tuple[dict | None, str | None]:
                        afd_config = AFDConfig(
                            n_a_nodes=int(n_a_nodes),
                            n_f_nodes=int(n_f_nodes),
                            gpus_per_node=int(gpus_per_node),
                            tp_a=int(tp_a),
                            f_moe_ep_size=int(f_moe_ep_size),
                            a_batch_size=a_batch_size,
                            num_microbatches=int(num_microbatches),
                            pipeline_model=str(pipeline_model),
                            comm_overhead_factor=float(comm_overhead_factor),
                            phase="decode",
                            combined_with_pd=bool(combined_with_pd),
                            boundary_on_attn=bool(boundary_on_attn),
                        )

                        candidate_runtime_config = copy.deepcopy(eval_runtime_config)
                        candidate_runtime_config.batch_size = afd_config.n_a_workers * afd_config.a_batch_size

                        session = AFDInferenceSession(
                            model_path=model_path,
                            a_model_config=a_model_config,
                            f_model_config=f_model_config,
                            database=database,
                            backend=backend,
                            afd_config=afd_config,
                        )
                        summary = session.run_afd(
                            candidate_runtime_config,
                            phase="decode",
                            free_gpu_memory_fraction=free_gpu_memory_fraction,
                            max_seq_len=max_seq_len,
                        )

                        if summary.check_oom():
                            return None, "oom"

                        row = dict(summary.get_result_dict())
                        _apply_afd_decode_latency_correction(row, decode_latency_correction)

                        row_tpot = float(row.get("tpot", 0.0) or 0.0)
                        if target_tpot is not None and row_tpot > float(target_tpot):
                            return row, "tpot"
                        return row, None

                    # Binary search for the largest batch size (aligned to 8) that
                    # satisfies TPOT SLA.  TPOT is monotonically increasing with bs,
                    # so binary search finds the optimum in ~5 run_afd() calls instead
                    # of the old halving approach which could skip valid intermediate
                    # values.
                    if combined_max_bs >= 32:
                        bs_align = 8
                        bs_min = 32
                        lo = bs_min
                        hi = combined_max_bs

                        while lo <= hi:
                            mid = ((lo + hi) // 2 // bs_align) * bs_align
                            if mid < bs_min:
                                mid = bs_min
                            a_batch_size = mid

                            row, rejection_reason = _evaluate_auto_batch(a_batch_size)
                            if rejection_reason == "oom":
                                hi = mid - bs_align
                                continue

                            if rejection_reason == "tpot":
                                rejection_counts["tpot"] += 1
                                candidate_rejected_by_tpot = True
                                hi = mid - bs_align
                                row_tpot = float(row.get("tpot", 0.0) or 0.0) if row is not None else 0.0
                                logger.debug(
                                    "AFD candidate a%dxf%d tp_a=%d ep=%d mb=%d pipe=%s bs=%d "
                                    "TPOT=%.1fms > %.1fms, searching lower",
                                    n_a_nodes,
                                    n_f_nodes,
                                    tp_a,
                                    f_moe_ep_size,
                                    num_microbatches,
                                    pipeline_model,
                                    a_batch_size,
                                    row_tpot,
                                    target_tpot,
                                )
                                continue

                            # Passed OOM + TPOT — record and search higher for a better bs.
                            best_row = row
                            lo = mid + bs_align

                    if best_row is not None:
                        candidate_rows.append(best_row)
                    elif combined_max_bs >= 32 and not candidate_rejected_by_tpot:
                        rejection_counts["oom"] += 1

                    existing_batch_sizes = {
                        int(row.get("(a)bs", 0) or 0)
                        for row in candidate_rows
                    }
                    for a_batch_size in low_latency_batch_sizes:
                        if a_batch_size in existing_batch_sizes:
                            continue
                        row, rejection_reason = _evaluate_auto_batch(a_batch_size)
                        if rejection_reason == "oom":
                            rejection_counts["low_batch_oom"] += 1
                            continue
                        if rejection_reason == "tpot":
                            rejection_counts["tpot"] += 1
                            continue
                        if row is not None:
                            candidate_rows.append(row)
                            existing_batch_sizes.add(a_batch_size)

                    if not candidate_rows and combined_max_bs < 32 and not low_latency_batch_sizes:
                        rejection_counts["oom"] += 1

                accepted_candidate = False
                for candidate_row in candidate_rows:
                    best_row = candidate_row
                    if combined_with_pd and prefill_options:
                        combined = _combine_afd_row_with_static_prefill(
                            best_row,
                            prefill_options,
                            target_ttft=eval_runtime_config.ttft,
                            target_request_latency=eval_runtime_config.request_latency,
                            total_gpus=total_gpus,
                            max_prefill_gpus=max_prefill_gpus,
                            max_prefill_workers=max_prefill_workers,
                            prefill_degradation=prefill_degradation,
                            decode_degradation=decode_degradation,
                            ttft_correction_factor=ttft_correction_factor,
                        )
                        if combined is not None:
                            best_row = combined
                        else:
                            rejection_counts["prefill_combine"] += 1
                            best_row = None

                    if best_row is not None:
                        sla_reason = _afd_sla_rejection_reason(
                            best_row,
                            eval_runtime_config,
                            require_ttft=bool(combined_with_pd),
                        )
                        if sla_reason is not None:
                            rejection_counts[sla_reason] += 1
                            best_row = None

                    if (
                        best_row is not None
                        and total_gpus is not None
                        and total_gpus > 0
                        and int(best_row.get("num_total_gpus", 0)) > total_gpus
                    ):
                        rejection_counts["gpu_budget"] += 1
                        best_row = None

                    if best_row is not None:
                        rows.append(best_row)
                        accepted_candidate = True

                if not accepted_candidate:
                    logger.debug(
                        "AFD candidate a%dxf%d tp_a=%d ep=%d mb=%d pipe=%s: "
                        "no batch size satisfied OOM+SLA, skipping",
                        n_a_nodes,
                        n_f_nodes,
                        tp_a,
                        f_moe_ep_size,
                        num_microbatches,
                        pipeline_model,
                    )
            except Exception as e:
                logger.debug(
                    "AFD candidate a%sxf%s tp_a=%s ep=%s mb=%s pipe=%s failed, skipping",
                    n_a_nodes,
                    n_f_nodes,
                    tp_a,
                    f_moe_ep_size,
                    num_microbatches,
                    pipeline_model,
                    exc_info=True,
                )
                exceptions.append(e)
                continue

    if not rows:
        rejection_summary = _format_afd_rejection_summary(rejection_counts) or "none"
        if exceptions and not any(rejection_counts.values()):
            raise RuntimeError(
                f"No AFD results found for any candidate topology. Showing last exception: {exceptions[-1]}"
            ) from exceptions[-1]
        if fixed_total_batch_size is not None:
            raise NoFeasibleConfigError(
                f"No AFD results found for total_batch_size={fixed_total_batch_size}: no candidate topology "
                "could use the fixed batch exactly while satisfying memory and SLA constraints. "
                f"Rejections: {rejection_summary}."
            )
        raise NoFeasibleConfigError(
            "No AFD results found for any candidate topology. No configuration satisfied "
            "the memory, GPU budget, or SLA constraints. "
            f"Rejections: {rejection_summary}."
        )

    results_df = pd.DataFrame(rows, columns=ColumnsAFD)
    dedupe_cols = [c for c in results_df.columns if c != "_per_ops_source"]
    results_df = results_df.drop_duplicates(subset=dedupe_cols, ignore_index=True)
    results_df = results_df.sort_values(by="tokens/s/gpu", ascending=False).reset_index(drop=True)
    return results_df


def get_pareto_front(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    maximize_x: bool = True,
    maximize_y: bool = True,
) -> pd.DataFrame:
    """
    Get Pareto front from raw data points.

    Args:
        df: Source dataframe.
        x_col: Column name for x axis.
        y_col: Column name for y axis.
        maximize_x: Treat larger values on x axis as better if True, else minimize.
        maximize_y: Treat larger values on y axis as better if True, else minimize.
    """
    if df is None:
        return pd.DataFrame()
    if df.empty:
        return df.iloc[0:0].copy()
    if x_col not in df.columns or y_col not in df.columns:
        return pd.DataFrame(columns=[x_col, y_col])

    working = df[[x_col, y_col]].replace([np.inf, -np.inf], np.nan)
    valid_mask = working.notna().all(axis=1)
    if not valid_mask.any():
        return df.iloc[0:0].copy()

    df = df.loc[valid_mask].sort_values(by=x_col)

    def is_pareto(costs: np.ndarray) -> np.ndarray:
        is_better = np.ones(costs.shape[0], dtype=bool)
        for i, c in enumerate(costs):
            if is_better[i]:
                # Keep any point with a lower cost
                is_better[is_better] = np.any(costs[is_better] > c, axis=1)  # Remove dominated points
                is_better[i] = True  # And keep self
        return is_better

    working = df[[x_col, y_col]].copy()
    if not maximize_x:
        working[x_col] = -working[x_col]
    if not maximize_y:
        working[y_col] = -working[y_col]

    # Convert DataFrame columns to numpy array
    costs = working[[x_col, y_col]].values
    is_pareto_front = is_pareto(costs)

    # Plot Pareto front
    pareto_front = df[is_pareto_front]
    return pareto_front.sort_values(by=x_col).reset_index(drop=True)


def draw_pareto(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    ax: plt.Axes,
    color: str,
    label: str,
    *,
    maximize_x: bool = True,
    maximize_y: bool = True,
) -> None:
    """
    Draw Pareto front to plot.
    """
    df = df.sort_values(by=x_col)

    # Plot Pareto front
    pareto_front = get_pareto_front(df, x_col, y_col, maximize_x=maximize_x, maximize_y=maximize_y)
    ax.plot(pareto_front[x_col], pareto_front[y_col], color=color, label=label)
    ax.scatter(pareto_front[x_col], pareto_front[y_col], color=color)

    # Add labels and title
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.legend()


def draw_pareto_to_string(
    title: str,
    series: list[dict],
    *,
    highlight: dict | None = None,
    x_label: str = "tokens/s/user",
    y_label: str = "tokens/s/gpu_cluster",
) -> str:
    """Render one or more Pareto series as ASCII plot text.

    Args:
        title: Plot title prefix.
        series: List of dictionaries describing the series to plot. Expected keys:
            - "df": pandas DataFrame containing the Pareto frontier.
            - "label": Series label (default: "series-{index}").
            - "color": plotext color (RGB tuple or name).
            - "marker": plotext marker (default: "dot").
        highlight: Optional dictionary describing a highlighted point set. Accepts
            keys "df", "label", "color", "marker" similar to ``series``.
    """

    plotext.plot_size(80, 30)
    plotext.theme("clear")

    palette = [
        (144, 238, 144),  # light green
        (200, 200, 200),  # gray
        (135, 206, 235),  # sky blue
        (255, 182, 193),  # light pink
        (255, 160, 122),  # light salmon
        (221, 160, 221),  # plum
    ]
    markers = ["dot", "fdot", "hdot", "ldot", "sdot", "xdot"]

    y_max = 0.0
    x_max = 0.0
    x_min = math.inf

    for idx, entry in enumerate(series):
        df = entry.get("df")
        if df is None or df.empty:
            continue
        color = entry.get("color") or palette[idx % len(palette)]
        marker = entry.get("marker") or markers[idx % len(markers)]
        label = entry.get("label") or f"series-{idx + 1}"
        plotext.plot(
            df[x_label],
            df[y_label],
            label=label,
            color=color,
            marker=marker,
        )
        y_max = max(df[y_label].max(), y_max)
        x_max = max(df[x_label].max(), x_max)
        x_min = min(df[x_label].min(), x_min)

    if highlight is not None:
        highlight_df = highlight.get("df")
        if highlight_df is not None and not highlight_df.empty:
            color = highlight.get("color") or (255, 215, 0)  # gold
            marker = highlight.get("marker") or "xdot"
            label = highlight.get("label") or "Best"
            plotext.plot(
                highlight_df[x_label],
                highlight_df[y_label],
                label=label,
                color=color,
                marker=marker,
            )
            y_max = max(highlight_df[y_label].max(), y_max)
            x_max = max(highlight_df[x_label].max(), x_max)
            x_min = min(highlight_df[x_label].min(), x_min)

    plotext.title(f"{title}: {y_label} vs {x_label}")
    plotext.xlabel(x_label)
    plotext.ylabel(y_label)
    plotext.grid(False)

    if y_max > 0.0 and x_max > 0.0:
        y_max = ((y_max * 1.2) + 49) // 50 * 50
        x_limit = ((x_max * 1.1) + 19) // 20 * 20
        cap = 300.0
        has_points_within_cap = x_min <= cap
        effective_x_max = min(x_limit, cap) if has_points_within_cap else x_limit
        plotext.ylim(0.0, y_max)
        plotext.xlim(0.0, effective_x_max)

    try:
        buf = plotext.build()
        # Strip ANSI escapes and Unicode box-drawing / block characters
        # so piped output (e.g. `| cat -v`) is readable pure ASCII.
        if use_plain_cli_output():
            buf = strip_unicode_to_ascii(buf)
    except Exception:
        logger.exception("failed to build plotext")
        buf = ""
    plotext.clear_data()
    return buf


def _get_best_configs_under_constraint(
    total_gpus: int,
    pareto_df: pd.DataFrame,
    target_value: float,
    constraint_col: str,
    top_n: int = 1,
    group_by: str | None = None,
    *,
    secondary_sort_col: str | None = None,
    secondary_sort_ascending: bool = False,
) -> pd.DataFrame:
    """Generic helper to rank configs under a scalar constraint."""
    if pareto_df is None or pareto_df.empty:
        return pd.DataFrame()

    if target_value is None:
        logger.info("No target value provided for constraint column '%s'.", constraint_col)
        return pd.DataFrame()

    if constraint_col not in pareto_df.columns or "tokens/s/gpu" not in pareto_df.columns:
        logger.warning(
            "Pareto DataFrame for constraint evaluation is missing '%s' or 'tokens/s/gpu' columns.",
            constraint_col,
        )
        return pd.DataFrame()

    candidate_configs = pareto_df[pareto_df[constraint_col] <= target_value].copy()

    if top_n < 1:
        logger.error("top_n is less than 1")
        return pd.DataFrame()

    if candidate_configs.empty:
        # No config meets the constraint strictly -- fall back to closest matches.
        logger.info(
            "No config found with %s <= %s. Returning top-%d closest configs.",
            constraint_col,
            target_value,
            top_n,
        )
        candidate_configs = pareto_df.copy()
        candidate_configs["_sla_exceeded"] = True
    else:
        candidate_configs["_sla_exceeded"] = False

    # compute achieved cluster-scale tokens/s/gpu
    candidate_configs["tokens/s/gpu_cluster"] = (
        candidate_configs["tokens/s/gpu"]
        * (total_gpus // candidate_configs["num_total_gpus"])
        * candidate_configs["num_total_gpus"]
        / total_gpus
    )
    candidate_configs.replace([np.inf, -np.inf], np.nan, inplace=True)
    finite_value_cols = [constraint_col, "tokens/s/gpu_cluster"]
    invalid_value_mask = candidate_configs[finite_value_cols].isna().any(axis=1)
    if invalid_value_mask.any():
        logger.info(
            "Dropping %d Pareto configs with non-finite %s or tokens/s/gpu_cluster values.",
            int(invalid_value_mask.sum()),
            constraint_col,
        )
        candidate_configs = candidate_configs.loc[~invalid_value_mask].copy()

    if candidate_configs.empty:
        return pd.DataFrame()

    if group_by is not None and group_by in candidate_configs.columns:
        top_indexes = candidate_configs.groupby(group_by)["tokens/s/gpu_cluster"].idxmax().dropna()
        if top_indexes.empty:
            return pd.DataFrame()
        candidate_configs = candidate_configs.loc[top_indexes]

    if candidate_configs["_sla_exceeded"].all():
        # All configs exceed the SLA -- sort by closest to target
        sort_columns = [constraint_col]
        sort_ascending = [True]
    else:
        sort_columns = ["tokens/s/gpu_cluster"]
        sort_ascending = [False]
        if secondary_sort_col and secondary_sort_col in candidate_configs.columns:
            sort_columns.append(secondary_sort_col)
            sort_ascending.append(secondary_sort_ascending)

    candidate_configs = (
        candidate_configs.sort_values(by=sort_columns, ascending=sort_ascending).head(top_n).reset_index(drop=True)
    )
    candidate_configs.drop(columns=["_sla_exceeded"], inplace=True)
    return candidate_configs


def get_best_configs_under_tpot_constraint(
    total_gpus: int,
    pareto_df: pd.DataFrame,
    target_tpot: float,
    top_n: int = 1,
    group_by: str | None = None,
) -> pd.DataFrame:
    """TPOT specific convenience wrapper."""
    return _get_best_configs_under_constraint(
        total_gpus=total_gpus,
        pareto_df=pareto_df,
        target_value=target_tpot,
        constraint_col="tpot",
        top_n=top_n,
        group_by=group_by,
        secondary_sort_col="tokens/s/user",
        secondary_sort_ascending=False,
    )


def get_best_configs_under_request_latency_constraint(
    total_gpus: int,
    pareto_df: pd.DataFrame,
    target_request_latency: float,
    top_n: int = 1,
    group_by: str | None = None,
) -> pd.DataFrame:
    """Request-latency specific wrapper."""
    return _get_best_configs_under_constraint(
        total_gpus=total_gpus,
        pareto_df=pareto_df,
        target_value=target_request_latency,
        constraint_col="request_latency",
        top_n=top_n,
        group_by=group_by,
        secondary_sort_col="request_latency",
        secondary_sort_ascending=True,
    )


def get_best_configs_for_target_load(
    pareto_df: pd.DataFrame,
    constraint_col: str,
    constraint_value: float,
    target_request_rate: float | None = None,
    target_concurrency: float | None = None,
    top_n: int = 5,
    max_total_gpus: int | None = None,
    group_by: str | None = None,
) -> pd.DataFrame:
    """Find configs that serve a target load with minimum GPUs under SLA.

    Exactly one of ``target_request_rate`` or ``target_concurrency`` must be
    provided.  For each candidate config the number of replicas needed to
    meet the load target is computed from the per-replica ``seq/s`` or
    ``concurrency`` column, then ``total_gpus_needed`` is derived.  Results
    are ranked by ``total_gpus_needed`` ascending (fewer GPUs is better).

    Args:
        pareto_df: DataFrame from the sweep (agg or disagg).
        constraint_col: SLA column to filter on (``"tpot"`` or
            ``"request_latency"``).
        constraint_value: Maximum allowed value for *constraint_col*.
        target_request_rate: Target system request rate in req/s.
        target_concurrency: Target number of concurrent requests.
        top_n: Number of top configurations to return.
        max_total_gpus: Optional upper bound on total GPUs.
        group_by: Optional column to group-by before ranking (takes best
            per group, same semantics as
            :func:`_get_best_configs_under_constraint`).

    Returns:
        DataFrame of top-N configs with added columns
        ``replicas_needed``, ``total_gpus_needed`` and
        ``tokens/s/gpu_cluster``.
    """
    if pareto_df is None or pareto_df.empty:
        return pd.DataFrame()

    has_rate = target_request_rate is not None
    has_conc = target_concurrency is not None
    if has_rate == has_conc:
        raise ValueError("Exactly one of target_request_rate or target_concurrency must be provided.")

    if constraint_col not in pareto_df.columns:
        logger.warning("Pareto DataFrame is missing constraint column '%s'.", constraint_col)
        return pd.DataFrame()

    # 1. Filter by SLA constraint (fall back to closest if none meet it)
    candidates = pareto_df[pareto_df[constraint_col] <= constraint_value].copy()
    if candidates.empty:
        logger.info(
            "No config found with %s <= %s for load-match. Returning top-%d closest configs.",
            constraint_col,
            constraint_value,
            top_n,
        )
        candidates = pareto_df.copy()
        candidates["_sla_exceeded"] = True
    else:
        candidates["_sla_exceeded"] = False

    # 2. Compute replicas needed per-row
    if has_rate:
        load_col = "seq/s"
        if load_col not in candidates.columns:
            logger.warning("Pareto DataFrame is missing '%s' column for load-match.", load_col)
            return pd.DataFrame()
        candidates["replicas_needed"] = np.ceil(target_request_rate / candidates[load_col]).astype(int)
    else:
        load_col = "concurrency"
        if load_col not in candidates.columns:
            logger.warning("Pareto DataFrame is missing '%s' column for load-match.", load_col)
            return pd.DataFrame()
        candidates["replicas_needed"] = np.ceil(target_concurrency / candidates[load_col]).astype(int)

    candidates["replicas_needed"] = candidates["replicas_needed"].clip(lower=1)

    # 3. Total GPUs needed
    candidates["total_gpus_needed"] = candidates["replicas_needed"] * candidates["num_total_gpus"]

    # 4. tokens/s/gpu_cluster = per-replica efficiency (no cluster-wide scaling)
    candidates["tokens/s/gpu_cluster"] = candidates["tokens/s/gpu"]

    # 5. GPU ceiling: prefer configs that fit; if none do, warn and cap to max_total_gpus
    gpu_capped = False
    if max_total_gpus is not None and not candidates["_sla_exceeded"].all():
        fits = candidates[candidates["total_gpus_needed"] <= max_total_gpus]
        if fits.empty:
            logger.warning(
                "Target load requires more GPUs than available (%d). "
                "Returning best config scaled to use all %d GPUs. "
                "The target load may NOT be fully served.",
                max_total_gpus,
                max_total_gpus,
            )
            # Cap replicas to what fits in the GPU budget and keep going.
            # Switch to maximizing throughput since all configs exceed budget.
            # Save uncapped replicas to compute load_served_pct.
            uncapped_replicas = candidates["replicas_needed"].copy()
            candidates["replicas_needed"] = (max_total_gpus // candidates["num_total_gpus"]).clip(lower=1).astype(int)
            candidates["total_gpus_needed"] = candidates["replicas_needed"] * candidates["num_total_gpus"]
            # Percentage of target load that the capped deployment can serve
            candidates["load_served_pct"] = ((candidates["replicas_needed"] / uncapped_replicas) * 100).round(1)
            # Recompute cluster throughput with capped replicas
            candidates["tokens/s/gpu_cluster"] = (
                candidates["tokens/s/gpu"]
                * candidates["replicas_needed"]
                * candidates["num_total_gpus"]
                / max_total_gpus
            )
            gpu_capped = True
        else:
            candidates = fits

    # 6. Group-by (take best per group)
    if group_by is not None and group_by in candidates.columns:
        if candidates["_sla_exceeded"].all():
            top_indexes = candidates.groupby(group_by)[constraint_col].idxmin()
        elif gpu_capped:
            top_indexes = candidates.groupby(group_by)["tokens/s/gpu_cluster"].idxmax()
        else:
            top_indexes = candidates.groupby(group_by)["total_gpus_needed"].idxmin()
        candidates = candidates.loc[top_indexes]

    # 7. Rank
    if candidates["_sla_exceeded"].all():
        # Sort by closest to SLA target
        sort_cols = [constraint_col]
        sort_asc = [True]
    elif gpu_capped:
        # GPU budget exceeded: maximize throughput with available GPUs
        sort_cols = ["tokens/s/gpu_cluster"]
        sort_asc = [False]
    else:
        # Normal load-match: fewest GPUs first, tiebreak by higher tokens/s/gpu
        sort_cols = ["total_gpus_needed", "tokens/s/gpu"]
        sort_asc = [True, False]

    candidates = candidates.sort_values(by=sort_cols, ascending=sort_asc).head(top_n).reset_index(drop=True)
    candidates.drop(columns=["_sla_exceeded"], inplace=True, errors="ignore")
    return candidates
