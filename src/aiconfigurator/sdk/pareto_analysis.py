# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiconfigurator.sdk.inference_session import InferenceSession, DisaggInferenceSession
import pandas as pd
import numpy as np
from aiconfigurator.sdk.models import get_model
from aiconfigurator.sdk.perf_database import get_database, PerfDatabase, get_latest_database_version
from aiconfigurator.sdk.common import ColumnsAgg
from aiconfigurator.sdk.backends.factory import get_backend
import matplotlib.pyplot as plt
import plotext
import copy
from typing import Optional
from aiconfigurator.sdk import config
from aiconfigurator.sdk import common
import logging
import traceback
import logging
import traceback
from scipy.interpolate import interp1d
from aiconfigurator.sdk.utils import safe_mkdir
logger = logging.getLogger(__name__)


def enumerate_parallel_config(num_gpu_list: list[int], 
                              tp_list: list[int], 
                              pp_list: list[int], 
                              dp_list: list[int]=[1], 
                              moe_tp_list: list[int]=[1], 
                              moe_ep_list: list[int]=[1], 
                              is_moe: bool=False,
                              backend: common.BackendName=common.BackendName.trtllm) -> list[list[int]]:
    """
    Enumerate parallel configurations based on parallel list. This is a helper function for agg_pareto and disagg_pareto to define search space.

    Args:
        num_gpu_list: list of number of gpus, this is used to filter out invalid parallel configurations
        tp_list: list of tensor parallel sizes
        pp_list: list of pipeline parallel sizes
        dp_list: list of data parallel sizes
        moe_tp_list: list of moe tensor parallel sizes
        moe_ep_list: list of moe expert parallel sizes
        is_moe: whether to use moe
        backend: backend name enum. Important for moe parallel enumeration as different backends have different moe parallel support.
    Returns:
        parallel_config_list: list of parallel configurations
    """
    parallel_config_list = []
    for tp in tp_list:
        for pp in pp_list:
            if is_moe:
                for dp in dp_list:
                    for moe_tp in moe_tp_list:
                        for moe_ep in moe_ep_list:
                            if dp*tp*pp in num_gpu_list and dp*tp == moe_tp*moe_ep: # check num gpu and width
                                # backend specific filters
                                if backend == common.BackendName.trtllm: # trtllm as trtllm don't supports attn tp > 1
                                    if dp > 1 and tp > 1:
                                        continue
                                elif backend == common.BackendName.sglang: # sglang doesn't support moe tp and moe ep > 1 at the same time for now
                                    if moe_tp > 1:
                                        continue
                                parallel_config_list.append([tp, pp, dp, moe_tp, moe_ep])
            else:
                if tp*pp in num_gpu_list:
                    parallel_config_list.append([tp, pp, 1, 1, 1])
    
    for parallel_config in parallel_config_list:
        tp, pp, dp, moe_tp, moe_ep = parallel_config
        logger.info(f"Enumerated parallel config: tp={tp}, pp={pp}, dp={dp}, moe_tp={moe_tp}, moe_ep={moe_ep}")

    return parallel_config_list

def agg_pareto(model_name: str,
               runtime_config: config.RuntimeConfig, 
               database: PerfDatabase,
               backend_name: str,
               model_config: config.ModelConfig,
               parallel_config_list: list[list[int]]) -> pd.DataFrame:
    """
    Find Pareto front for agg.
    We will first enumerate all the parallel configurations and then find the Pareto front for each parallel configuration.

    Args:
        model_name: name of the model
        runtime_config: runtime config. tpot is a list of tpot values to search over or a single tpot value
        database: database
        backend_name: name of the backend
        model_config: model config
        parallel_config_list: list of parallel configurations
    
    Returns:
        results_df: dataframe of the results
    """
    
    tpot_list = runtime_config.tpot if isinstance(runtime_config.tpot, list) else [runtime_config.tpot]

    # agg is agg server, the loop over parallel is outside here.
    results_df = pd.DataFrame(columns=ColumnsAgg)
    for parallel_config in parallel_config_list:
        tp_size, pp_size, dp_size, moe_tp_size, moe_ep_size = parallel_config
        logger.debug(f"Getting candidate workers with parallel config: tp={tp_size}, pp={pp_size}, dp={dp_size}, moe_tp={moe_tp_size}, moe_ep={moe_ep_size}")
        
        try:
            overwritten_model_config = copy.deepcopy(model_config)
            overwritten_model_config.pp_size = pp_size
            overwritten_model_config.tp_size = tp_size
            overwritten_model_config.moe_tp_size = moe_tp_size
            overwritten_model_config.moe_ep_size = moe_ep_size
            overwritten_model_config.attention_dp_size = dp_size
            model = get_model(model_name=model_name, model_config=overwritten_model_config, backend_name=backend_name)
            backend = get_backend(backend_name)
            sess = InferenceSession(model=model, database=database, backend=backend)
            for tpot in tpot_list:
                overwritten_runtime_config = copy.deepcopy(runtime_config)
                overwritten_runtime_config.tpot = tpot
                summary = sess.find_best_agg_result_under_constraints(runtime_config=overwritten_runtime_config,
                                                        top_k=10, max_batch_size=512, ctx_stride=512)
                result_df = summary.get_summary_df()
                if (len(result_df) == 0):
                    logger.debug(f"No result found for tpot {tpot}ms in agg pareto.")
                    continue
                if len(results_df) == 0:
                    results_df = result_df
                else:
                    results_df = pd.concat([results_df, result_df], axis=0, ignore_index=True)
        except Exception as e:
            logger.error(f"Error getting candidate workers with parallel config: tp={tp_size}, pp={pp_size}, dp={dp_size}, moe_tp={moe_tp_size}, moe_ep={moe_ep_size}, skip this combination: {traceback.format_exc()}")
            continue

    results_df = results_df.sort_values(by='tokens/s/gpu', ascending=False).reset_index(drop=True)

    return results_df

def disagg_pareto(model_name: str,
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
                  **kwargs) -> pd.DataFrame:
    """
    Find Pareto front for Disaggregated Inference.
    This is a proxy function calls into DisaggInferenceSession.find_best_disagg_result_under_constraints.

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
        prefill_num_worker_list: list of number of prefill workers in a disagg replica composed of xPyD, x_list, in kwargs
        prefill_max_num_worker: max number of prefill workers in a disagg replica composed of xPyD, x_max, in kwargs
        decode_num_worker_list: list of number of decode workers in a disagg replica composed of xPyD, y_list, in kwargs
        decode_max_num_worker: max number of decode workers in a disagg replica composed of xPyD, y_max, in kwargs
    
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
                working_list = list(range(1, max_constraint+1))
                logger.debug(f"{working_list} constrained by max_constraint: {max_constraint}")
            else:
                logger.debug(f"no constraint on {working_list}")
        return working_list
    
    prefill_backend = get_backend(prefill_backend_name)
    decode_backend = get_backend(decode_backend_name)

    disagg_sess = DisaggInferenceSession(prefill_database, prefill_backend, decode_database, decode_backend)
    disagg_sess.set_latency_correction_scales(prefill_latency_correction_scale, decode_latency_correction_scale)

    prefill_max_num_tokens = kwargs.get('prefill_max_num_tokens', 16384)
    decode_max_num_tokens = kwargs.get('decode_max_num_tokens', 512)
    logger.debug(f"prefill_max_num_tokens: {prefill_max_num_tokens}, decode_max_num_tokens: {decode_max_num_tokens}")

    # num gpu constraint for the whole system
    num_gpu_list = kwargs.get('num_gpu_list', None)
    max_num_gpu = kwargs.get('max_num_gpu', None)
    logger.debug(f"num_gpu_list: {num_gpu_list}, max_num_gpu: {max_num_gpu}")
    num_gpu_list = get_working_list(num_gpu_list, max_num_gpu)

    # prefill worker constraint
    prefill_num_worker_list = kwargs.get('prefill_num_worker_list', None)
    prefill_max_num_worker = kwargs.get('prefill_max_num_worker', None)
    logger.debug(f"prefill_num_worker_list: {prefill_num_worker_list}, prefill_max_num_worker: {prefill_max_num_worker}")
    prefill_num_worker_list = get_working_list(prefill_num_worker_list, prefill_max_num_worker)
    
    # decode worker constraint
    decode_num_worker_list = kwargs.get('decode_num_worker_list', None)
    decode_max_num_worker = kwargs.get('decode_max_num_worker', None)
    logger.debug(f"decode_num_worker_list: {decode_num_worker_list}, decode_max_num_worker: {decode_max_num_worker}")
    decode_num_worker_list = get_working_list(decode_num_worker_list, decode_max_num_worker)

    summary = disagg_sess.find_best_disagg_result_under_constraints(model_name=model_name,
                                                                    runtime_config=runtime_config,
                                                                    prefill_model_config=prefill_model_config,
                                                                    prefill_parallel_config_list=prefill_parallel_config_list,
                                                                    prefill_max_num_tokens=prefill_max_num_tokens,
                                                                    prefill_num_worker_list=prefill_num_worker_list,
                                                                    decode_model_config=decode_model_config,
                                                                    decode_parallel_config_list=decode_parallel_config_list,
                                                                    decode_max_num_tokens=decode_max_num_tokens,
                                                                    decode_num_worker_list=decode_num_worker_list,
                                                                    num_gpu_list=num_gpu_list)

    return summary.get_summary_df()


def get_pareto_front(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    """
    Get Pareto front from raw data points.
    """
    df = df.sort_values(by=x_col)
    def is_pareto(costs: np.ndarray) -> np.ndarray:
        is_better = np.ones(costs.shape[0], dtype=bool)
        for i, c in enumerate(costs):
            if is_better[i]:
                # Keep any point with a lower cost
                is_better[is_better] = np.any(costs[is_better]>c, axis=1)  # Remove dominated points
                is_better[i] = True  # And keep self
        return is_better

    # Convert DataFrame columns to numpy array
    costs = df[[x_col, y_col]].values
    is_pareto_front = is_pareto(costs)

    # Plot Pareto front
    pareto_front = df[is_pareto_front]
    return pareto_front

def draw_pareto(df: pd.DataFrame, x_col: str, y_col: str, ax: plt.Axes, color: str, label: str) -> None:
    """
    Draw Pareto front to plot.
    """
    df = df.sort_values(by=x_col)

    # Plot Pareto front
    pareto_front = get_pareto_front(df, x_col, y_col)
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
    highlight: Optional[dict] = None,
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
    markers = ["dot", "fdot", "hdot", "ldot", "sdot", "x"]

    y_max = 0.0
    x_max = 0.0

    for idx, entry in enumerate(series):
        df = entry.get("df")
        if df is None or df.empty:
            continue
        color = entry.get("color") or palette[idx % len(palette)]
        marker = entry.get("marker") or markers[idx % len(markers)]
        label = entry.get("label") or f"series-{idx+1}"
        plotext.plot(
            df['tokens/s/user'],
            df['tokens/s/gpu'],
            label=label,
            color=color,
            marker=marker,
        )
        y_max = max(df['tokens/s/gpu'].max(), y_max)
        x_max = max(df['tokens/s/user'].max(), x_max)

    if highlight is not None:
        highlight_df = highlight.get("df")
        if highlight_df is not None and not highlight_df.empty:
            color = highlight.get("color") or (255, 215, 0)  # gold
            marker = highlight.get("marker") or "x"
            label = highlight.get("label") or "Best"
            plotext.plot(
                highlight_df['tokens/s/user'],
                highlight_df['tokens/s/gpu'],
                label=label,
                color=color,
                marker=marker,
            )
            y_max = max(highlight_df['tokens/s/gpu'].max(), y_max)
            x_max = max(highlight_df['tokens/s/user'].max(), x_max)

    plotext.title(f"{title}: tokens/s/gpu vs tokens/s/user")
    plotext.xlabel("tokens/s/user")
    plotext.ylabel("tokens/s/gpu")
    plotext.grid(False)

    if y_max > 0.0 and x_max > 0.0:
        y_max = ((y_max * 1.2) + 49) // 50 * 50
        x_max = ((x_max * 1.1) + 19) // 20 * 20
        x_max = min(x_max, 300)
        plotext.ylim(0.0, y_max)
        plotext.xlim(0.0, x_max)

    try:
        buf = plotext.build()
    except Exception as e:
        logger.error(f"failed to build plotext: {e}")
        buf = ""
    plotext.clear_data()
    return buf

def interpolate_throughput_at_tpot(df: Optional[pd.DataFrame], target_tpot: float) -> float:
    """
    Interpolates the throughput at a given TPOT. This is more for reference by reading the pareto frontier.
    Args:
        df: The DataFrame containing the throughput data.
        target_tpot: The target TPOT in ms.
    Returns:
        The interpolated throughput at the target TPOT.
    """
    if df is None or df.empty:
        return 0.0
    
    target_tps_user = 1000.0/target_tpot
    
    # Filter out points where tpot is not available or invalid
    df_filtered = df.dropna(subset=['tokens/s/user', 'tokens/s/gpu'])
    if df_filtered.empty or len(df_filtered) < 2:
        # Not enough points to interpolate, try to find closest or return 0
        if not df_filtered.empty:
                # Fallback: find the point with tpot closest to target_tps_user
            closest_idx = (df_filtered['tokens/s/user'] - target_tps_user).abs().idxmin()
            return df_filtered.loc[closest_idx, 'tokens/s/gpu']
        return 0.0

    # Sort by tokens/s/user for interpolation
    df_sorted = df_filtered.sort_values(by='tokens/s/user')
    
    # Create interpolation functions
    # If target_tpot is outside the range, interp1d will extrapolate or error depending on fill_value
    # Using fill_value="extrapolate" can be risky.
    # It's often better to clamp to the nearest value if outside the range.
    min_tps_user, max_tps_user = df_sorted['tokens/s/user'].min(), df_sorted['tokens/s/user'].max()

    if target_tps_user < min_tps_user:
        return df_sorted.iloc[0]['tokens/s/gpu'] # Closest value at smallest tokens/s/user
    if target_tps_user > max_tps_user:
        return 0.0 # cannot meet the target tps_user
        
    interp_func = interp1d(df_sorted['tokens/s/user'], df_sorted['tokens/s/gpu'], kind='linear', fill_value="extrapolate")
    
    interpolated_throughput = float(interp_func(target_tps_user))
    return max(0.0, interpolated_throughput) # Ensure non-negative throughput

def get_best_configs_under_tpot_constraint(
    total_gpus: int,
    pareto_df: pd.DataFrame, 
    target_tpot: float,
    top_n: int = 1,
    group_by: Optional[str] = None,
) -> pd.DataFrame:
    """
    Finds the best actual config from a Pareto DataFrame
    that meets the target_tpot constraint (tpot <= target_tpot)
    and maximizes 'tokens/s/gpu'.
    Args:
        pareto_df: The Pareto DataFrame.
        target_tpot: The target TPOT in ms.
    Returns:
        A DataFrame containing the best config that meets the target_tpot constraint.
    """
    if pareto_df is None or pareto_df.empty:
        return pd.DataFrame()

    # Ensure 'tpot' and 'tokens/s/gpu' columns exist
    if 'tpot' not in pareto_df.columns or 'tokens/s/gpu' not in pareto_df.columns:
        logger.warning("Pareto DataFrame for _get_best_configs_under_tpot_constraint is missing 'tpot' or 'tokens/s/gpu' columns.")
        return pd.DataFrame()

    candidate_configs = pareto_df[pareto_df['tpot'] <= target_tpot].copy()

    if top_n < 1:
        logger.error(f"top_n is less than 1")
        return pd.DataFrame()
    
    if not candidate_configs.empty:
        # compute achieved cluster-scale tokens/s/gpu
        candidate_configs['tokens/s/gpu_cluster'] = candidate_configs['tokens/s/gpu'] * \
            (total_gpus // candidate_configs['num_total_gpus']) * candidate_configs['num_total_gpus'] / total_gpus
        if group_by is not None:
            top_indexes = candidate_configs.groupby(group_by)['tokens/s/gpu_cluster'].idxmax()
            candidate_configs = candidate_configs.loc[top_indexes]
        candidate_configs = candidate_configs.sort_values(by='tokens/s/gpu_cluster', ascending=False).head(top_n).reset_index(drop=True)
        logger.debug(f"actual replica-level throughputs: {candidate_configs['tokens/s/gpu'].iloc[0]:.2f} vs. actual cluster-level throughputs: {candidate_configs['tokens/s/gpu_cluster'].iloc[0]:.2f}")        
        return candidate_configs
    else:
        # No config meets tpot <= target_tpot.
        # Optionally, one could return the one closest to target_tpot if no strict candidates exist.
        # For now, return empty if no config meets the criteria.
        logger.info(f"No config found on Pareto front with TPOT <= {target_tpot}ms.")
        return pd.DataFrame()
