# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiconfigurator.sdk.inference_session import InferenceSession, DisaggInferenceSession
import pandas as pd
import numpy as np
from aiconfigurator.sdk.models import get_model
from aiconfigurator.sdk.perf_database import get_database, PerfDatabase, get_latest_database_version
from aiconfigurator.sdk.common import ColumnsAgg, BackendName
from aiconfigurator.sdk.backends.factory import get_backend
from aiconfigurator.sdk.models import check_is_moe
import matplotlib.pyplot as plt
import plotext
import copy
from typing import Optional, Any, Union
from aiconfigurator.sdk import config
from aiconfigurator.sdk import common
import logging
import traceback
from munch import Munch, DefaultMunch
import logging
import traceback
from scipy.interpolate import interp1d
logger = logging.getLogger(__name__)


def enumerate_parallel_config(num_gpu_list: list[int], 
                              tp_list: list[int], 
                              pp_list: list[int], 
                              dp_list: list[int]=[1], 
                              moe_tp_list: list[int]=[1], 
                              moe_ep_list: list[int]=[1], 
                              is_moe: bool=False,
                              backend: BackendName=BackendName.trtllm) -> list[list[int]]:
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
            model = get_model(model_name=model_name, model_config=overwritten_model_config)
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


class TaskConfig:

    def _populate_agg_config(self):
        self.config.worker_config = Munch()
        self.config.worker_config.system_name = self.system_name
        self.config.worker_config.backend_name = self.backend_name
        if self.backend_version is None:
            self.config.worker_config.backend_version = get_latest_database_version(self.config.worker_config.system_name, self.config.worker_config.backend_name)
        else:
            self.config.worker_config.backend_version = self.backend_version

        is_moe = check_is_moe(self.config.model_name)
        should_enable_pp = False # TODO, add logic to verify if pp is needed on pcie-based system. However pp it not well aligned yet.

        self.config.is_moe = is_moe
        # default parallel config
        self.config.worker_config.num_gpu_per_worker = [1, 2, 4, 8]
        self.config.worker_config.tp_list = [1, 2, 4, 8]
        self.config.worker_config.pp_list = [1, 2, 4, 8] if should_enable_pp else [1]
        self.config.worker_config.dp_list = [1, 2, 4, 8] if is_moe else [1]
        self.config.worker_config.moe_tp_list = [1]
        self.config.worker_config.moe_ep_list = [1, 2, 4, 8] if is_moe else [1]

        if not is_moe:
            if self.config.worker_config.system_name == 'gb200_sxm':
                self.config.worker_config.num_gpu_per_worker = [1, 2, 4, 8, 16]
                self.config.worker_config.tp_list = [1, 2, 4, 8, 16]
                self.config.worker_config.pp_list = [1]
        else:
            if self.enable_wide_ep:
                self.config.worker_config.num_gpu_per_worker = [1, 2, 4, 8, 16, 32]
                self.config.worker_config.tp_list = [1, 2, 4, 8]
                self.config.worker_config.pp_list = [1, 2, 4, 8, 16, 32] if should_enable_pp else [1]
                self.config.worker_config.dp_list = [1, 2, 4, 8, 16, 32]
                self.config.worker_config.moe_tp_list = [1]
                self.config.worker_config.moe_ep_list = [1, 2, 4, 8, 16, 32]
                
        # quantization config
        self.config.worker_config.gemm_quant_mode = "fp8_block"
        self.config.worker_config.moe_quant_mode = "fp8_block"
        self.config.worker_config.kvcache_quant_mode = "fp8"
        self.config.worker_config.fmha_quant_mode = "float16" if self.config.model_name in ['DEEPSEEK_V3', 'KIMI_K2'] else "fp8"
        self.config.worker_config.comm_quant_mode = "half"

        database = get_database(system=self.config.worker_config.system_name, backend=self.config.worker_config.backend_name, version=self.config.worker_config.backend_version)
        if database.system_spec['gpu']['sm_version'] >= 100:
            self.config.worker_config.gemm_quant_mode = "nvfp4"
            self.config.worker_config.moe_quant_mode = "nvfp4"
        elif database.system_spec['gpu']['sm_version'] < 89:
            self.config.worker_config.gemm_quant_mode = "float16"
            self.config.worker_config.moe_quant_mode = "float16"
            self.config.worker_config.kvcache_quant_mode = "float16"

        if self.use_specific_quant_mode is not None:
            # fp8_tensor, float16.
            if self.use_specific_quant_mode != 'w4afp8': # w4afp8 is only for moe
                self.config.worker_config.gemm_quant_mode = self.use_specific_quant_mode
            self.config.worker_config.moe_quant_mode = self.use_specific_quant_mode
        
        logger.info(f"Task {self.task_name}: Runtime config: {self.config.runtime_config}")
        logger.info(f"Task {self.task_name}: Worker config: {self.config.worker_config}")

    def _populate_disagg_config(self):
        self.config.prefill_worker_config = Munch()
        self.config.decode_worker_config = Munch()
        self.config.replica_config = Munch()
        self.config.advanced_tuning_config = Munch()

        self.config.prefill_worker_config.system_name = self.system_name
        self.config.prefill_worker_config.backend_name = self.backend_name
        self.config.decode_worker_config.system_name = self.decode_system_name if self.decode_system_name is not None else self.system_name
        self.config.decode_worker_config.backend_name = self.backend_name
        if self.backend_version is None:
            self.config.prefill_worker_config.backend_version = get_latest_database_version(self.config.prefill_worker_config.system_name, self.config.prefill_worker_config.backend_name)
            self.config.decode_worker_config.backend_version = get_latest_database_version(self.config.decode_worker_config.system_name, self.config.decode_worker_config.backend_name)
        else:
            self.config.prefill_worker_config.backend_version = self.backend_version
            self.config.decode_worker_config.backend_version = self.backend_version

        is_moe = check_is_moe(self.config.model_name)
        should_enable_pp = False # TODO, add logic to verify if pp is needed on pcie-based system. However pp it not well aligned yet.

        self.config.is_moe = is_moe
    
        # default parallel config
        self.config.prefill_worker_config.num_gpu_per_worker = [1, 2, 4, 8]
        self.config.prefill_worker_config.tp_list = [1, 2, 4, 8]
        self.config.prefill_worker_config.pp_list = [1, 2, 4, 8] if should_enable_pp else [1]
        self.config.prefill_worker_config.dp_list = [1] # we disable prefill attn dp for empirical reason
        self.config.prefill_worker_config.moe_tp_list = [1]
        self.config.prefill_worker_config.moe_ep_list = [1, 2, 4, 8] if is_moe else [1]
        self.config.decode_worker_config.num_gpu_per_worker = [1, 2, 4, 8]
        self.config.decode_worker_config.tp_list = [1, 2, 4, 8]
        self.config.decode_worker_config.pp_list = [1, 2, 4, 8] if should_enable_pp else [1]
        self.config.decode_worker_config.dp_list = [1, 2, 4, 8] if is_moe else [1]
        self.config.decode_worker_config.moe_tp_list = [1]
        self.config.decode_worker_config.moe_ep_list = [1, 2, 4, 8] if is_moe else [1]

        if not is_moe:
            if self.config.prefill_worker_config.system_name == 'gb200_sxm':
                self.config.prefill_worker_config.num_gpu_per_worker = [1, 2, 4, 8, 16]
                self.config.prefill_worker_config.tp_list = [1, 2, 4, 8, 16]
                self.config.prefill_worker_config.pp_list = [1]
            if self.config.decode_worker_config.system_name == 'gb200_sxm':
                self.config.decode_worker_config.num_gpu_per_worker = [1, 2, 4, 8, 16]
                self.config.decode_worker_config.tp_list = [1, 2, 4, 8, 16]
                self.config.decode_worker_config.pp_list = [1]
        else:
            if self.enable_wide_ep:
                self.config.prefill_worker_config.num_gpu_per_worker = [1, 2, 4, 8, 16]
                self.config.prefill_worker_config.tp_list = [1, 2, 4, 8]
                self.config.prefill_worker_config.pp_list = [1, 2, 4, 8, 16] if should_enable_pp else [1]
                self.config.prefill_worker_config.dp_list = [1, 2, 4]
                self.config.prefill_worker_config.moe_tp_list = [1]
                self.config.prefill_worker_config.moe_ep_list = [1, 2, 4, 8, 16]
                
                self.config.decode_worker_config.num_gpu_per_worker = [1, 2, 4, 8, 16, 32, 64]
                self.config.decode_worker_config.tp_list = [1, 2, 4, 8]
                self.config.decode_worker_config.pp_list = [1, 2, 4, 8, 16, 32, 64] if should_enable_pp else [1]
                self.config.decode_worker_config.dp_list = [1, 2, 4, 8, 16, 32, 64]
                self.config.decode_worker_config.moe_tp_list = [1]
                self.config.decode_worker_config.moe_ep_list = [1, 2, 4, 8, 16, 32, 64]
        
        # quantization config
        self.config.prefill_worker_config.gemm_quant_mode = "fp8_block"
        self.config.prefill_worker_config.moe_quant_mode = "fp8_block"
        self.config.prefill_worker_config.kvcache_quant_mode = "fp8"
        self.config.prefill_worker_config.fmha_quant_mode = "float16" if self.config.model_name in ['DEEPSEEK_V3', 'KIMI_K2'] else "fp8"
        self.config.prefill_worker_config.comm_quant_mode = "half"
        
        database = get_database(system=self.config.prefill_worker_config.system_name, backend=self.config.prefill_worker_config.backend_name, version=self.config.prefill_worker_config.backend_version)
        if database.system_spec['gpu']['sm_version'] >= 100:
            self.config.prefill_worker_config.gemm_quant_mode = "nvfp4"
            self.config.prefill_worker_config.moe_quant_mode = "nvfp4"
        elif database.system_spec['gpu']['sm_version'] < 89:
            self.config.prefill_worker_config.gemm_quant_mode = "float16"
            self.config.prefill_worker_config.moe_quant_mode = "float16"
            self.config.prefill_worker_config.kvcache_quant_mode = "float16"
        
        if self.use_specific_quant_mode is not None:
            # fp8_tensor, float16.
            if self.use_specific_quant_mode != 'w4afp8': # w4afp8 is only for moe
                self.config.prefill_worker_config.gemm_quant_mode = self.use_specific_quant_mode
            self.config.prefill_worker_config.moe_quant_mode = self.use_specific_quant_mode

        self.config.decode_worker_config.gemm_quant_mode = "fp8_block"
        self.config.decode_worker_config.moe_quant_mode = "fp8_block"
        self.config.decode_worker_config.kvcache_quant_mode = "fp8"
        self.config.decode_worker_config.fmha_quant_mode = "float16"
        self.config.decode_worker_config.comm_quant_mode = "half"
        
        database = get_database(system=self.config.decode_worker_config.system_name, backend=self.config.decode_worker_config.backend_name, version=self.config.decode_worker_config.backend_version)
        if database.system_spec['gpu']['sm_version'] >= 100:
            self.config.decode_worker_config.gemm_quant_mode = "nvfp4"
            self.config.decode_worker_config.moe_quant_mode = "nvfp4"
        elif database.system_spec['gpu']['sm_version'] < 89:
            self.config.decode_worker_config.gemm_quant_mode = "float16"
            self.config.decode_worker_config.moe_quant_mode = "float16"
            self.config.decode_worker_config.kvcache_quant_mode = "float16"
        
        if self.use_specific_quant_mode is not None:
            # fp8_tensor, float16.
            if self.use_specific_quant_mode != 'w4afp8': # w4afp8 is only for moe
                self.config.decode_worker_config.gemm_quant_mode = self.use_specific_quant_mode
            self.config.decode_worker_config.moe_quant_mode = self.use_specific_quant_mode
        
        # replica config
        self.config.replica_config.num_gpu_per_replica = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128]
        self.config.replica_config.max_gpu_per_replica = 128
        self.config.replica_config.max_prefill_worker = 32
        self.config.replica_config.max_decode_worker = 32
        if self.enable_wide_ep:
            self.config.replica_config.num_gpu_per_replica = None
            self.config.replica_config.max_gpu_per_replica = 512
            self.config.replica_config.max_prefill_worker = 32
            self.config.replica_config.max_decode_worker = 32
        
        # advanced tuning config
        self.config.advanced_tuning_config.prefill_correction_scale = 0.9
        self.config.advanced_tuning_config.decode_correction_scale = 0.92
        self.config.advanced_tuning_config.prefill_max_batch_size = 1
        self.config.advanced_tuning_config.decode_max_batch_size = 512

        logger.info(f"Task {self.task_name}: Runtime config: {self.config.runtime_config}")
        logger.info(f"Task {self.task_name}: Prefill worker config: {self.config.prefill_worker_config}")
        logger.info(f"Task {self.task_name}: Decode worker config: {self.config.decode_worker_config}")
        logger.info(f"Task {self.task_name}: Replica config: {self.config.replica_config}")
        logger.info(f"Task {self.task_name}: Advanced tuning config: {self.config.advanced_tuning_config}")

    def _overwrite_from_yaml_config(self, yaml_config: Union[dict, DefaultMunch]):
        # Convert to dict for easier manipulation, then convert back to Munch at the end
        target_dict = self.config.toDict()
        source_dict = copy.deepcopy(yaml_config)
        
        def _recursive_update(target, source, path=""):
            """Recursively update target dict with source dict values"""
            for key, value in source.items():
                current_path = f"{path}.{key}" if path else key
                
                # Check if key exists in target
                if key not in target:
                    logger.warning(f"Key '{current_path}' does not exist in self.config, adding it anyway")
                
                # If both are dicts, recurse
                if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                    _recursive_update(target[key], value, current_path)
                else:
                    # Direct assignment
                    target[key] = value
        
        _recursive_update(target_dict, source_dict)
        
        # Convert back to DefaultMunch
        self.config = DefaultMunch.fromDict(target_dict, DefaultMunch)
    
    def _convert_worker_config_to_enum(self, worker_config: Union[dict, DefaultMunch]):
        worker_config.gemm_quant_mode = common.GEMMQuantMode[worker_config.gemm_quant_mode]
        worker_config.moe_quant_mode = common.MoEQuantMode[worker_config.moe_quant_mode]
        worker_config.kvcache_quant_mode = common.KVCacheQuantMode[worker_config.kvcache_quant_mode]
        worker_config.fmha_quant_mode = common.FMHAQuantMode[worker_config.fmha_quant_mode]
        worker_config.comm_quant_mode = common.CommQuantMode[worker_config.comm_quant_mode]

    def __init__(self, 
                 serving_mode: str,
                 model_name: str,
                 system_name: str = 'h200_sxm',
                 decode_system_name: Optional[str] = None,
                 backend_name: str = 'trtllm',
                 backend_version: Optional[str] = None,
                 use_specific_quant_mode: Optional[str] = None, # fp8, fp8_block, float16, w4afp8, nvfp4
                 isl: int = 4000,
                 osl: int = 1000,
                 ttft: float = 1000,
                 tpot: float = 50,
                 enable_wide_ep: bool = False,
                 yaml_config: Optional[dict] = None):

        # non-config part
        self.task_name = f"{serving_mode}_{model_name}_{system_name}_{decode_system_name}_{backend_name}_{backend_version}_{isl}_{osl}_{ttft}_{tpot}"        
        self.system_name = system_name
        self.decode_system_name = decode_system_name
        self.backend_name = backend_name
        self.backend_version = backend_version
        self.use_specific_quant_mode = use_specific_quant_mode        
        self.enable_wide_ep = enable_wide_ep

        # common config part - use recursive DefaultMunch to ensure independent objects
        self.config = Munch()
        self.config.serving_mode = serving_mode
        self.config.model_name = model_name
        self.config.task_name = self.task_name
        self.config.nextn = 0 if model_name not in ['DEEPSEEK_V3', 'KIMI_K2'] else 1
        self.config.nextn_accept_rates = [0.85,0,0,0,0]
        self.config.runtime_config = Munch()
        self.config.runtime_config.isl = isl
        self.config.runtime_config.osl = osl
        self.config.runtime_config.ttft = ttft
        self.config.runtime_config.tpot = tpot

        if serving_mode == 'agg':
            self._populate_agg_config()
        elif serving_mode == 'disagg':
            self._populate_disagg_config()
        else:
            raise ValueError(f"Invalid serving mode: {serving_mode}")
        
        if yaml_config is not None:
            self._overwrite_from_yaml_config(yaml_config=yaml_config)

        if serving_mode == 'agg':
            self._convert_worker_config_to_enum(self.config.worker_config)
        elif serving_mode == 'disagg':
            self._convert_worker_config_to_enum(self.config.prefill_worker_config)
            self._convert_worker_config_to_enum(self.config.decode_worker_config)

class TaskRunner:
    def run_agg(self, task_config: DefaultMunch) -> Optional[pd.DataFrame]:
        logger.info(f"Task {task_config.task_name}: Setting up runtime config")
        runtime_config = config.RuntimeConfig(isl=task_config.runtime_config.isl,
                                              osl=task_config.runtime_config.osl, 
                                              ttft=task_config.runtime_config.ttft, 
                                              tpot=list(range(1,20,1))+list(range(20,300,5)))
        logger.info(f"Task {task_config.task_name}: Setting up database")
        try:
            database = copy.deepcopy(get_database(system=task_config.worker_config.system_name, 
                                                backend=task_config.worker_config.backend_name, 
                                                version=task_config.worker_config.backend_version))
        except Exception as e:
            logger.error(f"Error getting database for {task_config.worker_config.system_name} {task_config.worker_config.backend_name} {task_config.worker_config.backend_version}: {e}")
            return None
        logger.info(f"Task {task_config.task_name}: Setting up model config")
        model_config = config.ModelConfig(gemm_quant_mode=task_config.worker_config.gemm_quant_mode,
                                          kvcache_quant_mode=task_config.worker_config.kvcache_quant_mode,
                                          fmha_quant_mode=task_config.worker_config.fmha_quant_mode,
                                          moe_quant_mode=task_config.worker_config.moe_quant_mode,
                                          comm_quant_mode=task_config.worker_config.comm_quant_mode,
                                          nextn=task_config.nextn,
                                          nextn_accept_rates=task_config.nextn_accept_rates)
        logger.info(f"Task {task_config.task_name}: Enumerating parallel config")
        try:
            parallel_config_list = enumerate_parallel_config(
                num_gpu_list=task_config.worker_config.num_gpu_per_worker,
                tp_list=task_config.worker_config.tp_list,
                pp_list=task_config.worker_config.pp_list,
                dp_list=task_config.worker_config.dp_list,
                moe_tp_list=task_config.worker_config.moe_tp_list,
                moe_ep_list=task_config.worker_config.moe_ep_list,
                is_moe=check_is_moe(task_config.model_name),
                backend=common.BackendName(task_config.worker_config.backend_name)
            )
        except Exception as e:
            logger.error(f"Error enumerating parallel config for {task_config.worker_config.system_name} {task_config.worker_config.backend_name} {task_config.worker_config.backend_version}: {e}")
            return None
        logger.info(f"Task {task_config.task_name}: Running agg pareto")
        return agg_pareto(model_name=task_config.model_name,
               runtime_config=runtime_config, 
               database=database,
               backend_name=task_config.worker_config.backend_name,
               model_config=model_config,
               parallel_config_list=parallel_config_list)

    def run_disagg(self, task_config: DefaultMunch) -> Optional[pd.DataFrame]:
        logger.info(f"Task {task_config.task_name}: Setting up runtime config")
        runtime_config = config.RuntimeConfig(isl=task_config.runtime_config.isl,
                                              osl=task_config.runtime_config.osl, 
                                              ttft=task_config.runtime_config.ttft, 
                                              tpot=list(range(1,20,1))+list(range(20,300,5)))

        # prefill                                              
        logger.info(f"Task {task_config.task_name}: Setting up prefill database")
        try:
            prefill_database = copy.deepcopy(get_database(system=task_config.prefill_worker_config.system_name, 
                                                        backend=task_config.prefill_worker_config.backend_name, 
                                                        version=task_config.prefill_worker_config.backend_version))
        except Exception as e:
            logger.error(f"Error getting prefill database for {task_config.prefill_worker_config.system_name} {task_config.prefill_worker_config.backend_name} {task_config.prefill_worker_config.backend_version}: {e}")
            return None
        logger.info(f"Task {task_config.task_name}: Setting up prefill model config")
        prefill_model_config = config.ModelConfig(gemm_quant_mode=task_config.prefill_worker_config.gemm_quant_mode,
                                                  kvcache_quant_mode=task_config.prefill_worker_config.kvcache_quant_mode,
                                                  fmha_quant_mode=task_config.prefill_worker_config.fmha_quant_mode,
                                                  moe_quant_mode=task_config.prefill_worker_config.moe_quant_mode,
                                                  comm_quant_mode=task_config.prefill_worker_config.comm_quant_mode,
                                                  nextn=task_config.nextn,
                                                  nextn_accept_rates=task_config.nextn_accept_rates)

        logger.info(f"Task {task_config.task_name}: Enumerating prefill parallel config")
        try:
            prefill_parallel_config_list = enumerate_parallel_config(
                num_gpu_list=task_config.prefill_worker_config.num_gpu_per_worker,
                tp_list=task_config.prefill_worker_config.tp_list,
                pp_list=task_config.prefill_worker_config.pp_list,
                dp_list=task_config.prefill_worker_config.dp_list,
                moe_tp_list=task_config.prefill_worker_config.moe_tp_list,
                moe_ep_list=task_config.prefill_worker_config.moe_ep_list,
                is_moe=check_is_moe(task_config.model_name),
                backend=common.BackendName(task_config.prefill_worker_config.backend_name)
            )
        except Exception as e:
            logger.error(f"Error enumerating prefill parallel config for {task_config.prefill_worker_config.system_name} {task_config.prefill_worker_config.backend_name} {task_config.prefill_worker_config.backend_version}: {e}")
            return None

        # decode
        logger.info(f"Task {task_config.task_name}: Setting up decode database")
        try:
            decode_database = copy.deepcopy(get_database(system=task_config.decode_worker_config.system_name, 
                                                        backend=task_config.decode_worker_config.backend_name, 
                                                        version=task_config.decode_worker_config.backend_version))
        except Exception as e:
            logger.error(f"Error getting decode database for {task_config.decode_worker_config.system_name} {task_config.decode_worker_config.backend_name} {task_config.decode_worker_config.backend_version}: {e}")
            return None
        logger.info(f"Task {task_config.task_name}: Setting up decode model config")
        decode_model_config = config.ModelConfig(gemm_quant_mode=task_config.decode_worker_config.gemm_quant_mode,
                                                  kvcache_quant_mode=task_config.decode_worker_config.kvcache_quant_mode,
                                                  fmha_quant_mode=task_config.decode_worker_config.fmha_quant_mode,
                                                  moe_quant_mode=task_config.decode_worker_config.moe_quant_mode,
                                                  comm_quant_mode=task_config.decode_worker_config.comm_quant_mode,
                                                  nextn=task_config.nextn,
                                                  nextn_accept_rates=task_config.nextn_accept_rates)
        
        logger.info(f"Task {task_config.task_name}: Enumerating decode parallel config")
        try:
            decode_parallel_config_list = enumerate_parallel_config(
                num_gpu_list=task_config.decode_worker_config.num_gpu_per_worker,
                tp_list=task_config.decode_worker_config.tp_list,
                pp_list=task_config.decode_worker_config.pp_list,
                dp_list=task_config.decode_worker_config.dp_list,
                moe_tp_list=task_config.decode_worker_config.moe_tp_list,
                moe_ep_list=task_config.decode_worker_config.moe_ep_list,
                is_moe=check_is_moe(task_config.model_name),
                backend=common.BackendName(task_config.decode_worker_config.backend_name)
            )
        except Exception as e:
            logger.error(f"Error enumerating decode parallel config for {task_config.decode_worker_config.system_name} {task_config.decode_worker_config.backend_name} {task_config.decode_worker_config.backend_version}: {e}")
            return None
        
        logger.info(f"Task {task_config.task_name}: Running disagg pareto")
        return disagg_pareto(
                            model_name=task_config.model_name,
                            runtime_config=runtime_config,
                            prefill_database=prefill_database,
                            prefill_backend_name=task_config.prefill_worker_config.backend_name,
                            prefill_model_config=prefill_model_config,
                            prefill_parallel_config_list=prefill_parallel_config_list,
                            decode_database=decode_database,
                            decode_backend_name=task_config.decode_worker_config.backend_name,
                            decode_model_config=decode_model_config,
                            decode_parallel_config_list=decode_parallel_config_list,
                            num_gpu_list=task_config.replica_config.num_gpu_per_replica,
                            max_num_gpu=task_config.replica_config.max_gpu_per_replica,
                            prefill_max_num_worker=task_config.replica_config.max_prefill_worker,
                            decode_max_num_worker=task_config.replica_config.max_decode_worker,
                            prefill_max_batch_size=task_config.advanced_tuning_config.prefill_max_batch_size,
                            decode_max_batch_size=task_config.advanced_tuning_config.decode_max_batch_size,
                            prefill_correction_scale=task_config.advanced_tuning_config.prefill_correction_scale,                    
                            decode_correction_scale=task_config.advanced_tuning_config.decode_correction_scale
                            )

    def run(self, task_config: TaskConfig) -> Optional[pd.DataFrame]:
        task_name = task_config.task_name
        serving_mode = task_config.config.serving_mode
        # run
        logger.info(f"Starting Pareto Analysis for {task_name} in {serving_mode} mode...")
        try:
            if serving_mode == 'agg':
                df = self.run_agg(task_config.config)
            elif serving_mode == 'disagg':
                df = self.run_disagg(task_config.config)
            else:
                raise ValueError(f"Invalid serving mode: {serving_mode}")
        except Exception as e:
            logger.error(f"Error running pareto analysis for {task_name} in {serving_mode} mode: {e}")
            df = None

        if df is None:
            logger.warning(f"No result found for {task_name} in {serving_mode} mode.")
        
        return df


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

def draw_pareto_to_string(title: str,
                            best_config_df: Optional[pd.DataFrame],
                            disagg_pareto_df: Optional[pd.DataFrame],
                            agg_pareto_df: Optional[pd.DataFrame]) -> str:
    """
    Draw Pareto front to string.
    """
    plotext.plot_size(80, 30)
    plotext.theme("clear")
    if disagg_pareto_df is not None and not disagg_pareto_df.empty:
        plotext.plot(
            disagg_pareto_df['tokens/s/user'],
            disagg_pareto_df['tokens/s/gpu'],
            label='Disagg',
            color=(144, 238, 144),  # light green
            marker='d'
        )
    if agg_pareto_df is not None and not agg_pareto_df.empty:
        plotext.plot(
            agg_pareto_df['tokens/s/user'],
            agg_pareto_df['tokens/s/gpu'],
            label='Agg',
            color=(200, 200, 200),  # gray
            marker='a'
        )

    if best_config_df is not None and not best_config_df.empty:
        plotext.plot(
            best_config_df['tokens/s/user'],
            best_config_df['tokens/s/gpu'],
            label='Best',
            color=(255, 215, 0),  # gold
            marker='X'
        )

    plotext.title(f"{title}: tokens/s/gpu vs tokens/s/user")
    plotext.xlabel("tokens/s/user")
    plotext.ylabel("tokens/s/gpu")
    plotext.grid(False)

    y_min = 0.0
    y_max = 0.0
    x_min = 0.0
    x_max = 0.0
    if disagg_pareto_df is not None and not disagg_pareto_df.empty:
        y_max = max(disagg_pareto_df['tokens/s/gpu'].max(), y_max)
        x_max = max(disagg_pareto_df['tokens/s/user'].max(), x_max)
    if agg_pareto_df is not None and not agg_pareto_df.empty:
        y_max = max(agg_pareto_df['tokens/s/gpu'].max(), y_max)
        x_max = max(agg_pareto_df['tokens/s/user'].max(), x_max)
    y_max = y_max * 1.2
    y_max = ((y_max + 49) // 50) * 50
    x_max = x_max * 1.1
    x_max = ((x_max + 19) // 20) * 20
    x_max = min(x_max, 300)
    if y_max > 0.0 and x_max > 0.0:
        plotext.ylim(y_min, y_max)
        plotext.xlim(x_min, x_max)

    buf = plotext.build()
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

def get_best_config_under_tpot_constraint(
    total_gpus: int,
    pareto_df: pd.DataFrame, 
    target_tpot: float
) -> pd.DataFrame:
    """
    Finds the best actual config from a Pareto frontier DataFrame
    that meets the target_tpot constraint (tpot <= target_tpot)
    and maximizes 'tokens/s/gpu'.
    Args:
        pareto_df: The Pareto frontier DataFrame.
        target_tpot: The target TPOT in ms.
    Returns:
        A DataFrame containing the best config that meets the target_tpot constraint.
    """
    if pareto_df is None or pareto_df.empty:
        return pd.DataFrame()

    # Ensure 'tpot' and 'tokens/s/gpu' columns exist
    if 'tpot' not in pareto_df.columns or 'tokens/s/gpu' not in pareto_df.columns:
        logger.warning("Pareto DataFrame for _get_best_config_under_tpot_constraint is missing 'tpot' or 'tokens/s/gpu' columns.")
        return pd.DataFrame()

    candidate_configs = pareto_df[pareto_df['tpot'] <= target_tpot].copy()
    
    if not candidate_configs.empty:
        # compute achieved cluster-scale tokens/s/gpu
        candidate_configs['tokens/s/gpu_cluster'] = candidate_configs['tokens/s/gpu'] * \
            (total_gpus // candidate_configs['num_total_gpus']) * candidate_configs['num_total_gpus'] / total_gpus
        candidate_configs = candidate_configs.sort_values(by='tokens/s/gpu_cluster', ascending=False)
        logger.debug(f"actual replica-level throughputs: {candidate_configs['tokens/s/gpu'].iloc[0]:.2f} vs. actual cluster-level throughputs: {candidate_configs['tokens/s/gpu_cluster'].iloc[0]:.2f}")        
        return candidate_configs.head(1)
    else:
        # No config meets tpot <= target_tpot.
        # Optionally, one could return the one closest to target_tpot if no strict candidates exist.
        # For now, return empty if no config meets the criteria.
        logger.info(f"No config found on Pareto front with TPOT <= {target_tpot}ms.")
        return pd.DataFrame()

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    
    model_name = 'DEEPSEEK_V3'
    isl = 4000
    osl = 1000
    ttft = 1000
    tpot = 50

    agg_task_config = TaskConfig(serving_mode='agg', 
                                 model_name=model_name, 
                                 system_name='h200_sxm',
                                 isl=isl,
                                 osl=osl,
                                 ttft=ttft,
                                 tpot=tpot,
                                 use_specific_quant_mode='w4afp8')
    disagg_task_config = TaskConfig(serving_mode='disagg', 
                                    model_name=model_name, 
                                    system_name='h200_sxm', 
                                    decode_system_name='h100_sxm', 
                                    isl=isl,
                                    osl=osl,
                                    ttft=ttft,
                                    tpot=tpot,
                                    use_specific_quant_mode='w4afp8')
    task_runner = TaskRunner()
    agg_df = task_runner.run(agg_task_config)
    disagg_df = task_runner.run(disagg_task_config)
    print(agg_df)
    print(disagg_df)

