# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiconfigurator.sdk.inference_session import InferenceSession, DisaggInferenceSession
import pandas as pd
import numpy as np
from aiconfigurator.sdk.models import get_model
from aiconfigurator.sdk.perf_database import get_database, PerfDatabase
from aiconfigurator.sdk.common import ColumnsIFB, BackendName
import matplotlib.pyplot as plt
import copy
from aiconfigurator.sdk import config
import os, random
from aiconfigurator.sdk import common
from aiconfigurator.sdk.backends.factory import get_backend
import logging
import traceback

logger = logging.getLogger(__name__)


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

def enumerate_parallel_config(num_gpu_list: list[int], 
                              tp_list: list[int], 
                              pp_list: list[int], 
                              dp_list: list[int]=[1], 
                              moe_tp_list: list[int]=[1], 
                              moe_ep_list: list[int]=[1], 
                              is_moe: bool=False,
                              backend: BackendName=BackendName.trtllm) -> list[list[int]]:
    """
    Enumerate parallel configurations based on parallel list. This is a helper function for ifb_pareto and disagg_pareto to define search space.

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
                                if backend == BackendName.trtllm: # trtllm as trtllm don't supports attn tp > 1
                                    if dp > 1 and tp > 1:
                                        continue
                                elif backend == BackendName.sglang: # sglang doesn't support moe tp and moe ep > 1 at the same time for now
                                    if moe_tp > 1 and moe_ep > 1:
                                        continue
                                parallel_config_list.append([tp, pp, dp, moe_tp, moe_ep])
            else:
                if tp*pp in num_gpu_list:
                    parallel_config_list.append([tp, pp, 1, 1, 1])

    return parallel_config_list

def ifb_pareto(model_name: str,
               runtime_config: config.RuntimeConfig, 
               database: PerfDatabase,
               backend_name: str,
               model_config: config.ModelConfig,
               parallel_config_list: list[list[int]]) -> pd.DataFrame:
    """
    Find Pareto front for IFB.
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

    # ifb is agg server, the loop over parallel is outside here.
    results_df = pd.DataFrame(columns=ColumnsIFB)
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
            model = get_model(model_name=model_name, model_config=overwritten_model_config)
            backend = get_backend(backend_name)
            sess = InferenceSession(model=model, database=database, backend=backend)
            for tpot in tpot_list:
                overwritten_runtime_config = copy.deepcopy(runtime_config)
                overwritten_runtime_config.tpot = tpot
                summary = sess.find_best_ifb_result_under_constraints(runtime_config=overwritten_runtime_config,
                                                        top_k=10, max_batch_size=512, ctx_stride=512)
                result_df = summary.get_summary_df()
                if (len(result_df) == 0):
                    logger.debug(f"No result found for tpot {tpot}ms in ifb pareto.")
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
                  prefill_correction_scale: float,
                  decode_database: PerfDatabase, 
                  decode_backend_name: str, 
                  decode_model_config: config.ModelConfig, 
                  decode_parallel_config_list: list[list[int]], 
                  decode_correction_scale: float,
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
        prefill_correction_scale: prefill correction scale
        decode_database: decode database
        decode_backend_name: decode backend name
        decode_model_config: decode model config
        decode_parallel_config_list: decode parallel config list
        decode_correction_scale: decode correction scale
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
    disagg_sess.set_correction_scales(prefill_correction_scale, decode_correction_scale)

    prefill_max_num_tokens = kwargs.get('prefill_max_num_tokens', 16384)
    decode_max_num_tokens = kwargs.get('decode_max_num_tokens', 512)

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

def compare_results(label_results_dict: dict[str, pd.DataFrame], title: str, write_results: bool=True, show_results: bool=True) -> None:
    """
    Compare results of different methods.
    """
    color_list = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    # store results
    if write_results:
        dir_name = f'results/{title}_{random.randint(0,1000000)}'
        logger.info('saving results to ', dir_name)
        os.makedirs(dir_name, exist_ok=True)
        for label, results_df in label_results_dict.items():
            results_df.to_csv(f'{dir_name}/{label}.csv', index=False)

    fig, ax = plt.subplots(1,1, figsize=(8,5))
    plt.title(title)
    for i, (label, results_df) in enumerate(label_results_dict.items()):
        draw_pareto(results_df, 'tokens/s/user', 'tokens/s/gpu', ax, color_list[i%len(color_list)], label)

    if write_results:
        plt.savefig(f'{dir_name}/{title}.png')
    if show_results:
        plt.show()
    return


if __name__ == '__main__':
    version = '0.20.0'
    backend_name = 'trtllm'
    model_name = 'DEEPSEEK_V3'
    max_worker_list = list(range(1,33,1))
    model_config = config.ModelConfig(gemm_quant_mode=common.GEMMQuantMode.fp8_ootb,
                                      kvcache_quant_mode=common.KVCacheQuantMode.float16,
                                      fmha_quant_mode=common.FMHAQuantMode.float16,
                                      moe_quant_mode=common.MoEQuantMode.fp8_block)
    runtime_config = config.RuntimeConfig(isl=1024, osl=1024, ttft=1000, tpot=list(range(4,20,1))+list(range(20,200,5)))

    h200_database = get_database(system='h200_sxm', backend=backend_name, version=version)
    # moe
    ifb_parallel_config_list = prefill_parallel_config_list = decode_parallel_config_list = enumerate_parallel_config(num_gpu_list=[8], # deepseek needs at least 4 gpus
                                                         tp_list=[1,8],
                                                         pp_list=[1],
                                                         moe_tp_list=[1],
                                                         moe_ep_list=[8],
                                                         dp_list=[1,8],
                                                         is_moe=True,
                                                         backend=common.BackendName(backend_name))
    
    ifb_df_h200 = ifb_pareto(model_name=model_name, runtime_config=runtime_config, 
                             database=h200_database, backend_name=backend_name, model_config=model_config, 
                             parallel_config_list=ifb_parallel_config_list)

    disagg_df_h200 = disagg_pareto(model_name=model_name, runtime_config=runtime_config,
                                   prefill_database=h200_database, prefill_backend_name=backend_name, prefill_model_config=model_config, 
                                   prefill_parallel_config_list=prefill_parallel_config_list, prefill_num_worker_list=max_worker_list,
                                   prefill_correction_scale=1.0,
                                   decode_database=h200_database, decode_backend_name=backend_name, decode_model_config=model_config, 
                                   decode_parallel_config_list=decode_parallel_config_list, decode_num_worker_list=max_worker_list,
                                   decode_correction_scale=1.0)
                                  

    compare_results({'DISAGG_h200':disagg_df_h200, 'IFB_h200':ifb_df_h200},
                      f'{version}_{model_name}_isl{runtime_config.isl}_osl{runtime_config.osl}_ttft{runtime_config.ttft}_{model_config.gemm_quant_mode.name}_{model_config.kvcache_quant_mode.name}')