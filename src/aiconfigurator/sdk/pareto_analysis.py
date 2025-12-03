# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import logging
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotext

from aiconfigurator.sdk import config
from aiconfigurator.sdk.backends.factory import get_backend
from aiconfigurator.sdk.common import ColumnsAgg
from aiconfigurator.sdk.inference_session import DisaggInferenceSession, InferenceSession
from aiconfigurator.sdk.models import get_model
from aiconfigurator.sdk.perf_database import PerfDatabase
from aiconfigurator.sdk.utils import enumerate_ttft_tpot_constraints

logger = logging.getLogger(__name__)


def agg_pareto(
    model_name: str,
    runtime_config: config.RuntimeConfig,
    database: PerfDatabase,
    backend_name: str,
    model_config: config.ModelConfig,
    parallel_config_list: list[list[int]],
) -> pd.DataFrame:
    """
    Find Pareto front for agg.
    We will first enumerate all the parallel configurations and then find the Pareto front for
    each parallel configuration.

    Args:
        model_name: name of the model
        runtime_config: runtime config. tpot is a list of tpot values to search over or a single
            tpot value
        database: database
        backend_name: name of the backend
        model_config: model config
        parallel_config_list: list of parallel configurations

    Returns:
        results_df: dataframe of the results
    """

    # agg is agg server, the loop over parallel is outside here.
    results_df = pd.DataFrame(columns=ColumnsAgg)
    exceptions = []
    for parallel_config in parallel_config_list:
        tp_size, pp_size, dp_size, moe_tp_size, moe_ep_size = parallel_config
        logger.debug(
            f"Getting candidate workers with parallel config: tp={tp_size}, pp={pp_size}, "
            f"dp={dp_size}, moe_tp={moe_tp_size}, moe_ep={moe_ep_size}"
        )

        try:
            overwritten_model_config = copy.deepcopy(model_config)
            overwritten_model_config.pp_size = pp_size
            overwritten_model_config.tp_size = tp_size
            overwritten_model_config.moe_tp_size = moe_tp_size
            overwritten_model_config.moe_ep_size = moe_ep_size
            overwritten_model_config.attention_dp_size = dp_size
            model = get_model(
                model_name=model_name,
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
                )
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
        results_df = results_df.drop_duplicates(ignore_index=True)
        results_df = results_df.sort_values(by="tokens/s/gpu", ascending=False).reset_index(drop=True)
    else:
        raise RuntimeError(
            f"No results found for any parallel configuration. Showing last exception: {exceptions[-1]}"
        ) from exceptions[-1]

    return results_df


def disagg_pareto(
    model_name: str,
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
        model_name: name of the model
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

    disagg_sess = DisaggInferenceSession(prefill_database, prefill_backend, decode_database, decode_backend)
    disagg_sess.set_latency_correction_scales(prefill_latency_correction_scale, decode_latency_correction_scale)

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

    summary = disagg_sess.find_best_disagg_result_under_constraints(
        model_name=model_name,
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
    )

    return summary.get_summary_df()


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
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.sort_values(by=x_col)

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

    if not candidate_configs.empty:
        # compute achieved cluster-scale tokens/s/gpu
        candidate_configs["tokens/s/gpu_cluster"] = (
            candidate_configs["tokens/s/gpu"]
            * (total_gpus // candidate_configs["num_total_gpus"])
            * candidate_configs["num_total_gpus"]
            / total_gpus
        )
        if group_by is not None:
            top_indexes = candidate_configs.groupby(group_by)["tokens/s/gpu_cluster"].idxmax()
            candidate_configs = candidate_configs.loc[top_indexes]
        sort_columns = ["tokens/s/gpu_cluster"]
        sort_ascending = [False]
        if secondary_sort_col and secondary_sort_col in candidate_configs.columns:
            sort_columns.append(secondary_sort_col)
            sort_ascending.append(secondary_sort_ascending)
        candidate_configs = (
            candidate_configs.sort_values(by=sort_columns, ascending=sort_ascending).head(top_n).reset_index(drop=True)
        )
        return candidate_configs
    else:
        # No config meets constraint
        # Optionally, one could return the one closest to target_tpot if no strict candidates exist.
        # For now, return empty if no config meets the criteria.
        logger.info("No config found on Pareto front with %s <= %s.", constraint_col, target_value)
        return pd.DataFrame()


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
