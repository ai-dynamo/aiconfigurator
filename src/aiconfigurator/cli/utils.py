# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

import pandas as pd

from aiconfigurator.sdk.pareto_analysis import (
    get_best_configs_under_request_latency_constraint,
    get_best_configs_under_tpot_constraint,
    get_pareto_front,
)
from aiconfigurator.sdk.task import TaskConfig

logger = logging.getLogger(__name__)


def process_experiment_result(task_config: TaskConfig, result: dict[str, pd.DataFrame], top_n: int = 5) -> tuple:
    """
    Process the result of a single experiment.
    Args:
        task_config: TaskConfig object for the experiment.
        result: Dictionary containing the pareto_df result of the experiment.
        top_n: Number of top configurations to return.
    Returns:
        tuple:
            - best_config_df: Best configuration dataframe.
            - best_throughput: Best throughput.
            - pareto_frontier_df: Pareto frontier dataframe.
            - x_axis_col: X-axis column name.
    """
    pareto_df = result["pareto_df"]
    runtime_cfg = task_config.config.runtime_config
    target_tpot = runtime_cfg.tpot
    target_request_latency = runtime_cfg.request_latency
    use_request_latency = target_request_latency is not None and target_request_latency > 0
    total_gpus = getattr(task_config, "total_gpus", None) or 0

    # Compute tokens/s/gpu_cluster for pareto_df
    if pareto_df is not None and not pareto_df.empty:
        pareto_df["tokens/s/gpu_cluster"] = (
            pareto_df["tokens/s/gpu"]
            * (total_gpus // pareto_df["num_total_gpus"])
            * pareto_df["num_total_gpus"]
            / total_gpus
        )
        x_axis_col = "request_latency" if use_request_latency else "tokens/s/user"
        pareto_frontier_df = get_pareto_front(
            pareto_df,
            x_axis_col,
            "tokens/s/gpu_cluster",
            maximize_x=not use_request_latency,
            maximize_y=True,
        )
    else:
        pareto_frontier_df = pd.DataFrame()
        x_axis_col = "request_latency" if use_request_latency else "tokens/s/user"

    group_by_key = "(d)parallel" if task_config.serving_mode == "disagg" else "parallel"
    if use_request_latency:
        best_config_df = get_best_configs_under_request_latency_constraint(
            total_gpus=total_gpus,
            pareto_df=pareto_df,
            target_request_latency=target_request_latency,
            top_n=top_n,
            group_by=group_by_key,
        )
    else:
        best_config_df = get_best_configs_under_tpot_constraint(  # based on all data points
            total_gpus=total_gpus,
            pareto_df=pareto_df,
            target_tpot=target_tpot,
            top_n=top_n,
            group_by=group_by_key,
        )

    if not best_config_df.empty:
        best_throughput = best_config_df["tokens/s/gpu_cluster"].values[0]
    else:
        best_throughput = 0.0

    return best_config_df, best_throughput, pareto_frontier_df, x_axis_col


def _merge_into_top_n(
    exps: list[str],
    task_configs: dict[str, TaskConfig],
    best_configs: dict[str, pd.DataFrame],
    pareto_fronts: dict[str, pd.DataFrame],
    pareto_x_axis: dict[str, str],
    top_n: int = 5,
) -> tuple:
    """Merge the best configs and pareto fronts into top N."""
    best_configs_dfs = []
    pareto_dfs = []
    for exp_name in exps:
        backend_name = task_configs[exp_name].backend_name
        df = best_configs[exp_name].copy()
        if not df.empty:
            df["backend"] = backend_name
            best_configs_dfs.append(df)

        pf = pareto_fronts.get(exp_name)
        if pf is not None and not pf.empty:
            pf_copy = pf.copy()
            pf_copy["backend"] = backend_name
            pareto_dfs.append(pf_copy)

    # Merge all best configs and take top N
    if best_configs_dfs:
        df_best_configs = pd.concat(best_configs_dfs, ignore_index=True)
        df_best_configs = df_best_configs.sort_values("tokens/s/gpu_cluster", ascending=False).head(top_n)
    else:
        df_best_configs = pd.DataFrame()
    best_throughput = df_best_configs["tokens/s/gpu_cluster"].values[0] if not df_best_configs.empty else 0.0

    df_merged_pareto_front = x_col = None
    # Merge pareto fronts for plotting and recompute Pareto frontier
    if pareto_dfs:
        df_combined_pareto = pd.concat(pareto_dfs, ignore_index=True)
        x_col = pareto_x_axis.get(exps[0], "tokens/s/user")
        df_merged_pareto_front = get_pareto_front(
            df_combined_pareto,
            x_col,
            "tokens/s/gpu_cluster",
            maximize_x=(x_col != "request_latency"),
            maximize_y=True,
        )

    return df_best_configs, best_throughput, df_merged_pareto_front, x_col


def merge_experiment_results_by_mode(
    task_configs: dict[str, TaskConfig],
    best_configs: dict[str, pd.DataFrame],
    pareto_fronts: dict[str, pd.DataFrame],
    pareto_x_axis: dict[str, str],
    top_n: int = 5,
) -> tuple[dict[str, pd.DataFrame], dict[str, float], dict[str, pd.DataFrame], dict[str, str]]:
    """
    Merge results from multiple experiments into Top N agg and disagg.
    For example, when backend="any", we have 6 experiments: agg_trtllm, agg_vllm, agg_sglang,
    disagg_trtllm, disagg_vllm, disagg_sglang. This function merges them into 2:
    agg (with top N from all backends) and disagg (with top N from all backends).

    Args:
        results: Dictionary containing the results of the experiments.
        task_configs: Dictionary containing the task configs of the experiments.
        best_configs: Dictionary containing the best configs of the experiments.
        best_throughputs: Dictionary containing the best throughputs of the experiments.
        pareto_fronts: Dictionary containing the pareto fronts of the experiments.
        pareto_x_axis: Dictionary containing the pareto x-axis of the experiments.
        top_n: Number of top configurations to return.

    Returns:
        tuple:
            - best_configs: Dictionary containing the best configs of the merged experiments.
            - best_throughputs: Dictionary containing the best throughputs of the merged experiments.
            - pareto_fronts: Dictionary containing the pareto fronts of the merged experiments.
            - pareto_x_axis: Dictionary containing the pareto x-axis of the merged experiments.
            - task_configs: Dictionary containing the task configs of the merged experiments.
    """
    agg_exps = [name for name, task_config in task_configs.items() if task_config.serving_mode == "agg"]
    disagg_exps = [name for name, task_config in task_configs.items() if task_config.serving_mode == "disagg"]

    merged_best_configs = {}
    merged_best_throughputs = {}
    merged_pareto_fronts = {}
    merged_pareto_x_axis = {}

    agg_merged = _merge_into_top_n(agg_exps, task_configs, best_configs, pareto_fronts, pareto_x_axis, top_n)
    disagg_merged = _merge_into_top_n(disagg_exps, task_configs, best_configs, pareto_fronts, pareto_x_axis, top_n)

    merged_best_configs["agg"] = agg_merged[0]
    merged_best_throughputs["agg"] = agg_merged[1]
    merged_pareto_fronts["agg"] = agg_merged[2]
    merged_pareto_x_axis["agg"] = agg_merged[3]
    merged_best_configs["disagg"] = disagg_merged[0]
    merged_best_throughputs["disagg"] = disagg_merged[1]
    merged_pareto_fronts["disagg"] = disagg_merged[2]
    merged_pareto_x_axis["disagg"] = disagg_merged[3]

    return merged_best_configs, merged_best_throughputs, merged_pareto_fronts, merged_pareto_x_axis
