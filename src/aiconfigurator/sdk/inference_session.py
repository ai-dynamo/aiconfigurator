# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiconfigurator.sdk import models, perf_database, config, common
from aiconfigurator.sdk.backends.base_backend import BaseBackend
import copy
import pandas as pd
import numpy as np
from collections import defaultdict
from aiconfigurator.sdk.inference_summary import InferenceSummary
import warnings
import logging
from typing import List, Optional, Tuple, Dict, Any
import traceback
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)

class InferenceSession(object):
    """
    InferenceSession holds the model and database to run inference loop

    Attributes:
        model (models.BaseModel): the model to run inference
        database (perf_database.PerfDatabase): the database to run inference
        backend (backend.Backend): the backend to run inference
    
    Methods:
        run_static (static, static_ctx, static_gen): to support static batching and disagg, returns details of a static run
        run_agg (static, static_ctx, static_gen): run agg inference, returns summary of the perf result with given agg config and runtime config (concurrency)
        find_best_agg_result_under_constraints (static, static_ctx, static_gen):
            find the best agg result under constraints, returns summary 
            which contains all the possible agg config and perf that matchs SLA.
    """
    def __init__(self, model:models.BaseModel, 
                 database:perf_database.PerfDatabase, 
                 backend:BaseBackend) -> None:
        """
        Initialize the InferenceSession
        """
        self._model = model
        self._database = database
        self._backend = backend

    def run_static(self, runtime_config:config.RuntimeConfig, mode:str, stride:int=32, latency_correction_scale:float=1.0) -> InferenceSummary :
        """
        Run static inference

        Args:
            runtime_config (RuntimeConfig): the runtime config
            mode (str): the mode to run inference, static, static_ctx, static_gen
            stride (int): the stride is used to accelerate the estimation, for a give osl, will only computes the i, i+stride, i+2*stride, ...
                step, default is 32.

        Returns:
            InferenceSummary: the summary of the inference result
        """
        return self._backend.run_static(self._model, self._database, runtime_config, mode, stride, latency_correction_scale)

    def run_agg(self, runtime_config:config.RuntimeConfig, **kwargs) -> InferenceSummary:
        """
        Run agg inference

        Args:
            runtime_config (RuntimeConfig): the runtime config
            **kwargs: other arguments to run agg, depends on the backend specific design

        Returns:
            InferenceSummary: the summary of the inference result
        """
        return self._backend.run_agg(self._model, self._database, runtime_config, **kwargs)
    
    # Optimization
    def find_best_agg_result_under_constraints(self, runtime_config:config.RuntimeConfig, **kwargs) -> InferenceSummary:
        """
        Find the best agg result under constraints

        Args:
            runtime_config (RuntimeConfig): the runtime config
            **kwargs: other arguments to find the best agg result under constraints, depends on the backend specific design

        Returns:
            InferenceSummary: the summary of the inference result, contains all the possible agg config and perf that matchs SLA.
        """
        return self._backend.find_best_agg_result_under_constraints(self._model, self._database, runtime_config, **kwargs)

    
class DisaggInferenceSession(object):
    '''
    Disaggregated inference session
    Run prefill and generation separately, with different models (parallel and precision config can be different) and databases
    0. init func only takes database and backend, model is passed in run_disagg
    1. run_disagg, given model, database and backend, given everything fixed ((max)batchsize and num_workers) , return the perf result of the system
    2. find_best_disagg_result_under_constraints, given database and backend, sweep batchsize and model parallel to match SLA,
      sweep workers to get best system perf/gpu if allowed. Return config (parallel, batchsize and num_workers) and perf.
    3. TODO, should consider kvcache model in future
    Disagg is more like a post processing step to do rate matching, that's why it's a DiaggInferenceSession instread of using InferenceSession.

    Attributes:
        prefill_database (perf_database.PerfDatabase): the database to run prefill
        prefill_backend (backend.Backend): the backend to run prefill
        decode_database (perf_database.PerfDatabase): the database to run decode
        decode_backend (backend.Backend): the backend to run decode

    Methods:
        run_disagg (model_name, runtime_config, prefill_model_config, prefill_batch_size, 
                    prefill_num_worker, decode_model_config, decode_batch_size, decode_num_worker)
            run disagg with given prefill/decode worker info
        find_best_disagg_result_under_constraints (model_name,runtime_config, prefill_model_config, 
                    prefill_parallel_config_list, prefill_max_num_tokens, prefill_num_worker_list, 
                    decode_model_config, decode_parallel_config_list, decode_max_num_tokens, 
                    decode_num_worker_list, num_gpu_list)
            find the best disagg result under constraints
        set_latency_correction_scales (prefill_latency_correction_scale, decode_latency_correction_scale):
            set the correction scales for better alignment with real system
    '''
    def __init__(self, 
                 prefill_database:perf_database.PerfDatabase, 
                 prefill_backend:BaseBackend,
                 decode_database:perf_database.PerfDatabase, 
                 decode_backend:BaseBackend) -> None:
        """
        Initialize the DisaggInferenceSession
        """
        self._prefill_database = prefill_database
        self._prefill_backend = prefill_backend
        self._decode_database = decode_database
        self._decode_backend = decode_backend

        # allow user to set correction scales for better alignment with real system
        # now the corection scales are used to correct the latency, not throughput, corrected latency = latency * correction_scale
        self._prefill_latency_correction_scale = 1.0
        self._decode_latency_correction_scale = 1.0

        # comes from pipeline bubble, especially when benchmarking with concurrency
        self._RATE_MATCHING_PREFILL_DEGRADATION_FACTOR = 0.9
        # comes from not saturating the batchsize slot of decode worker
        self._RATE_MATCHING_DECODE_DEGRADATION_FACTOR = 0.92

    def set_latency_correction_scales(self, prefill_latency_correction_scale:float, decode_latency_correction_scale:float):
        """
        Set the correction scales for better alignment with real system
        """
        self._prefill_latency_correction_scale = prefill_latency_correction_scale
        self._decode_latency_correction_scale = decode_latency_correction_scale

    def _get_disagg_summary_df(self, prefill_summary_df:pd.DataFrame, 
                               prefill_num_worker:int, 
                               decode_summary_df:pd.DataFrame, 
                               decode_num_worker:int) -> pd.DataFrame:
        """
        Get the disagg summary df based on prefill and decode summary df
        """
        seq_s = min(prefill_summary_df['seq/s']*prefill_num_worker*self._RATE_MATCHING_PREFILL_DEGRADATION_FACTOR, 
        decode_summary_df['seq/s']*decode_num_worker*self._RATE_MATCHING_DECODE_DEGRADATION_FACTOR)
        prefill_gpus = prefill_summary_df['pp']*prefill_summary_df['tp']*prefill_summary_df['dp']
        decode_gpus = decode_summary_df['pp']*decode_summary_df['tp']*decode_summary_df['dp']
        seq_s_gpu = seq_s / (prefill_gpus * prefill_num_worker + decode_gpus * decode_num_worker)

        model_name = prefill_summary_df['model']
        isl = prefill_summary_df['isl']
        osl = prefill_summary_df['osl']
        concurrency = decode_summary_df['concurrency']*decode_num_worker # this is not exact matching. You can use this concurrency to benchmark the system.
        request_rate = seq_s
        p_bs = prefill_summary_df['bs']
        p_global_bs = prefill_summary_df['global_bs']
        p_workers = prefill_num_worker
        d_bs = decode_summary_df['bs']
        d_global_bs = decode_summary_df['global_bs']
        d_workers = decode_num_worker
        ttft = prefill_summary_df['ttft']
        tpot = decode_summary_df['tpot']
        tokens_s = seq_s * osl
        tokens_s_gpu = tokens_s / (prefill_gpus * prefill_num_worker + decode_gpus * decode_num_worker)
        tokens_s_user = decode_summary_df['tokens/s/user']
        p_seq_s_worker = prefill_summary_df['seq/s']
        d_seq_s_worker = decode_summary_df['seq/s']
        num_total_gpus = prefill_gpus * prefill_num_worker + decode_gpus * decode_num_worker        
        p_tp = prefill_summary_df['tp']
        p_pp = prefill_summary_df['pp']
        p_dp = prefill_summary_df['dp']
        p_moe_tp = prefill_summary_df['moe_tp']
        p_moe_ep = prefill_summary_df['moe_ep']
        p_parallel = prefill_summary_df['parallel']
        p_gemm = prefill_summary_df['gemm']
        p_kvcache = prefill_summary_df['kvcache']
        p_fmha = prefill_summary_df['fmha']
        p_moe = prefill_summary_df['moe']
        p_comm = prefill_summary_df['comm']
        p_memory = prefill_summary_df['memory']
        p_backend = prefill_summary_df['backend']
        p_version = prefill_summary_df['version']
        p_system = prefill_summary_df['system']
        d_tp = decode_summary_df['tp']
        d_pp = decode_summary_df['pp']
        d_dp = decode_summary_df['dp']
        d_moe_tp = decode_summary_df['moe_tp']
        d_moe_ep = decode_summary_df['moe_ep']
        d_parallel = decode_summary_df['parallel']
        d_gemm = decode_summary_df['gemm']
        d_kvcache = decode_summary_df['kvcache']
        d_fmha = decode_summary_df['fmha']
        d_moe = decode_summary_df['moe']
        d_comm = decode_summary_df['comm']
        d_memory = decode_summary_df['memory']
        d_backend = decode_summary_df['backend']
        d_version = decode_summary_df['version']
        d_system = decode_summary_df['system']
                
        return pd.DataFrame([[model_name, isl, osl, 
                              concurrency, request_rate, p_bs, p_global_bs, p_workers, d_bs, d_global_bs, d_workers, 
                              ttft, tpot, seq_s, seq_s_gpu, tokens_s, tokens_s_gpu, tokens_s_user, p_seq_s_worker, d_seq_s_worker, 
                              num_total_gpus,
                              p_tp, p_pp, p_dp, p_moe_tp, p_moe_ep, p_parallel, 
                              p_gemm, p_kvcache, p_fmha, p_moe, p_comm, p_memory, 
                              p_backend, p_version, p_system, 
                              d_tp, d_pp, d_dp, d_moe_tp, d_moe_ep, d_parallel, 
                              d_gemm, d_kvcache, d_fmha, d_moe, d_comm, d_memory, 
                              d_backend, d_version, d_system]], columns=common.ColumnsDisagg).round(3)

    def run_disagg(self, 
                   model_name : str, 
                   runtime_config : config.RuntimeConfig, 
                   prefill_model_config : config.ModelConfig, 
                   prefill_batch_size : int, 
                   prefill_num_worker : int, 
                   decode_model_config : config.ModelConfig, 
                   decode_batch_size : int, 
                   decode_num_worker : int) -> InferenceSummary:
        '''
        Run disagg with given prefill/decode worker info

        Args:
            model_name (str): the model name
            runtime_config (RuntimeConfig): the runtime config
            prefill_model_config (ModelConfig): the prefill model config
            prefill_batch_size (int): the prefill batch size
            prefill_num_worker (int): the number of prefill workers
            decode_model_config (ModelConfig): the decode model config
            decode_batch_size (int): the decode batch size
            decode_num_worker (int): the number of decode workers

        Returns:
            InferenceSummary: the summary of the inference result
        '''
        prefill_model = models.get_model(model_name, prefill_model_config, self._prefill_backend.name.value)
        decode_model = models.get_model(model_name, decode_model_config, self._decode_backend.name.value)
        prefill_sess = InferenceSession(model=prefill_model,
                                        database=self._prefill_database,
                                        backend=self._prefill_backend)
        decode_sess = InferenceSession(model=decode_model,
                                        database=self._decode_database,
                                        backend=self._decode_backend)
                
        prefill_runtime_config = copy.deepcopy(runtime_config)
        prefill_runtime_config.batch_size = prefill_batch_size
        prefill_summary = prefill_sess.run_static(mode='static_ctx', runtime_config=prefill_runtime_config)
        decode_runtime_config = copy.deepcopy(runtime_config)
        decode_runtime_config.batch_size = decode_batch_size
        decode_summary = decode_sess.run_static(mode='static_gen', runtime_config=decode_runtime_config)
        disagg_summary_df = self._get_disagg_summary_df(prefill_summary.get_summary_df(), prefill_num_worker, decode_summary.get_summary_df(), decode_num_worker)

        disagg_summary = InferenceSummary(runtime_config=runtime_config)
        disagg_summary.set_summary_df(disagg_summary_df)
        return disagg_summary
    
    # optimization
    def find_best_disagg_result_under_constraints(self, 
                                                 model_name : str, 
                                                 runtime_config : config.RuntimeConfig, 
                                                 prefill_model_config : config.ModelConfig, 
                                                 prefill_parallel_config_list : List[Tuple[int, int, int, int, int]], 
                                                 prefill_max_num_tokens : int, 
                                                 prefill_num_worker_list : List[int], 
                                                 decode_model_config : config.ModelConfig, 
                                                 decode_parallel_config_list : List[Tuple[int, int, int, int, int]], 
                                                 decode_max_num_tokens : int, 
                                                 decode_num_worker_list : List[int],
                                                 num_gpu_list : Optional[List[int]]) -> Optional[InferenceSummary]:
        '''
        Run disagg with given constraints
        1. get all summary df, which matches the constraints
        2. find best config under constraints, call match scales to get the best scale
        3. call a func to get disagg_summary_df (this is shared by run_disgg func)
        4. return summary
        5. several empirical values:
            - 0.7 is the threshold to filter decode workers, because the performance of decode workers is much lower than prefill workers
            - 5 is the top k to return for drawing pareto frontier of each tpot

        Args:
            model_name (str): the model name
            runtime_config (RuntimeConfig): the runtime config
            prefill_model_config (ModelConfig): the prefill model config
            prefill_parallel_config_list (List[Tuple[int, int, int, int, int]]): the prefill parallel config list
            prefill_max_num_tokens (int): the prefill max num tokens
            prefill_num_worker_list (List[int]): the prefill num worker list
            decode_model_config (ModelConfig): the decode model config
            decode_parallel_config_list (List[Tuple[int, int, int, int, int]]): the decode parallel config list
            decode_max_num_tokens (int): the decode max num tokens
            decode_num_worker_list (List[int]): the decode num worker list
            num_gpu_list (Optional[List[int]]): the num gpu list

        Returns:
            Optional[InferenceSummary]: the summary of the inference result, contains all the possible disagg config and perf that matchs SLA.
        '''

        def _match_workers(prefill_throughput:float, 
                            prefill_gpus:int, 
                            decode_throughput:float, 
                            decode_gpus:int, 
                            prefill_num_worker_list:List[int], 
                            decode_num_worker_list:List[int], 
                            num_gpu_list:Optional[List[int]],
                            rate_matching_prefill_degradation_factor:float,
                            rate_matching_decode_degradation_factor:float) -> Tuple[int, int]:
            """
            Match the prefill and decode workers, return the best prefill and decode num worker
            """
            prefill_opt_num_worker, decode_opt_num_worker = -1, -1
            throughput_per_gpu_max = 0
            for decode_num_worker in decode_num_worker_list:
                for prefill_num_worker in prefill_num_worker_list:
                    num_gpu = prefill_gpus*prefill_num_worker + decode_gpus*decode_num_worker
                    if num_gpu_list is not None and num_gpu not in num_gpu_list:
                        continue

                    prefill_throughput_corrected = prefill_throughput * prefill_num_worker * rate_matching_prefill_degradation_factor
                    decode_throughput_corrected = decode_throughput * decode_num_worker * rate_matching_decode_degradation_factor

                    # criteria 1, try to make prefill_throughput larger than decode_throughput
                    #   other wise, decode bs cannot be achieved and decode throughput cannot be achieved as well.
                    #if prefill_throughput < decode_throughput:
                    #    continue

                    # criteria 2, try to make the throughput per gpu larger
                    throughput_per_gpu = min(prefill_throughput_corrected, decode_throughput_corrected) / num_gpu

                    if  throughput_per_gpu > throughput_per_gpu_max:
                        throughput_per_gpu_max = throughput_per_gpu
                        prefill_opt_num_worker, decode_opt_num_worker = prefill_num_worker, decode_num_worker

            return prefill_opt_num_worker, decode_opt_num_worker


        def _get_summary_df(model_config:config.ModelConfig, 
                           parallel_config_list:List[Tuple[int, int, int, int, int]], 
                           b_list:List[int], 
                           runtime_config:config.RuntimeConfig, 
                           mode:str,
                           latency_correction_scale:float = 1.0) -> pd.DataFrame:
            """
            Get all worker candidates based on give search space
            """
            summary_df = pd.DataFrame(columns=common.ColumnsStatic)

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
                    model = models.get_model(model_name=model_name, model_config=overwritten_model_config, backend_name=self._prefill_backend.name.value)
                    if mode == 'static_ctx':
                        sess = InferenceSession(model=model, database=self._prefill_database, backend=self._prefill_backend)
                    else:
                        sess = InferenceSession(model=model, database=self._decode_database, backend=self._decode_backend)

                    for b in b_list:
                        overwritten_runtime_config = copy.deepcopy(runtime_config)
                        overwritten_runtime_config.batch_size = b
                        summary = sess.run_static(mode=mode, runtime_config=overwritten_runtime_config, latency_correction_scale=latency_correction_scale)
                        if not summary.check_oom():
                            summary_df = pd.concat([summary_df, summary.get_summary_df()], axis=0, ignore_index=True)
                        else: # larger b will always OOM
                            break
                except Exception as e:
                    logger.error(f"Error getting candidate workers with parallel config: tp={tp_size}, pp={pp_size}, dp={dp_size}, moe_tp={moe_tp_size}, moe_ep={moe_ep_size} skip this combination: {traceback.format_exc()}")
                    continue
            return summary_df
        

        def _find_best_result_under_constraints_with_diversity(ttft:float, 
                                               tpot:float, 
                                               prefill_summary_df:pd.DataFrame, 
                                               decode_summary_df:pd.DataFrame, 
                                               return_top_k:int, 
                                               num_gpu_list:Optional[List[int]],
                                               rate_matching_prefill_degradation_factor:float,
                                               rate_matching_decode_degradation_factor:float) -> InferenceSummary: 
            """
            Find the best result under constraints with diversity
            """

            # 1. different from find_best_result_under_constraints, we need to find the best result with diversity for decode.
            #   diversity means we should categorize the decode summary df into different categories based on parallelism
            #   (we can use the parallel key in the df).
            #   do the rate matching and sort the result by category - throughput.
            # 2. for prefill, follow two rules: high throughput, if at same level, choose the one with small batchsize.
            #   add one func for correct ttft (we have some fomula, just leave it blank for now)
            # 3. prefill/decode correction are already applied to workers. 
            #   Additioanl correction can be a degradation factor for the final result during the rate matching process.
            # 4. rate matching. The prefill throughput should be 1.x larger than the decode throughput. 1.x is an empirical value.
            #   Default is 1.1.

            DECODE_FILTER_RATIO_MIN = 0.0
            DECODE_FILTER_RATIO_MAX = 1.0
            MAX_DECODE_WORKERS_PER_CATEGORY = 16
            MAX_PREFILL_WORKERS = 32

            # only ttft will be corrected here, other latency and throughput will not be corrected.
            # concurrency / num_prefill_workers = local_concurrency(lc); N x concurrency requests
            # formula = (lc * (lc+1) / 2 + lc * (N-1) )/lc/N
            # if we use N=10, it's lc/20+0.95. assume lc can be 15-20, 1.8 is a reasonable correction factor.
            # as we need to get the lc after rate matching, we cannot get the exact value now. 
            # Let's make it simple to do pre-correction instead of post-correction.
            correction_factor = 1.8  # let's make it simple for now.
            prefill_candidates = prefill_summary_df.assign(
                                            ttft=prefill_summary_df['ttft'] * correction_factor)

            prefill_candidates = prefill_candidates[prefill_candidates['context_latency'] < ttft]
            if len(prefill_candidates) == 0:
                logger.debug(f"No prefill worker candidates found for ttft {ttft}ms.")
                return None
            prefill_candidates = prefill_candidates.sort_values(by=['seq/s/gpu', 'global_bs'], ascending=[False, True]).reset_index(drop=True).head(MAX_PREFILL_WORKERS)

            decode_candidates = decode_summary_df[(decode_summary_df['tpot'] < tpot*DECODE_FILTER_RATIO_MAX) & (decode_summary_df['tpot'] > tpot * DECODE_FILTER_RATIO_MIN)].copy()
            if len(decode_candidates) == 0:
                logger.debug(f"No decode worker candidates found for tpot {tpot}ms.")
                return None

            disagg_summary_df_list: List[pd.DataFrame] = []
            for parallel_value, parallel_group in decode_candidates.groupby('parallel'):
                parallel_group_sorted = parallel_group.sort_values(by=['seq/s/gpu'], ascending=[False]).reset_index(drop=True).head(MAX_DECODE_WORKERS_PER_CATEGORY)
                category_results: List[pd.DataFrame] = []
                for _, decode_worker in parallel_group_sorted.iterrows():
                    decode_throughput = float(decode_worker['seq/s'])
                    decode_gpus = decode_worker['num_total_gpus']
                    for _, prefill_worker in prefill_candidates.iterrows():
                        prefill_throughput = float(prefill_worker['seq/s'])
                        prefill_gpus = prefill_worker['num_total_gpus']
                        prefill_num_worker, decode_num_worker = _match_workers(
                            prefill_throughput=prefill_throughput,
                            prefill_gpus=prefill_gpus,
                            decode_throughput=decode_throughput,
                            decode_gpus=decode_gpus,
                            prefill_num_worker_list=prefill_num_worker_list,
                            decode_num_worker_list=decode_num_worker_list,
                            num_gpu_list=num_gpu_list,
                            rate_matching_prefill_degradation_factor=rate_matching_prefill_degradation_factor,
                            rate_matching_decode_degradation_factor=rate_matching_decode_degradation_factor
                        )
                        if prefill_num_worker == -1 or decode_num_worker == -1:
                            continue

                        disagg_df = self._get_disagg_summary_df(prefill_worker, prefill_num_worker, decode_worker, decode_num_worker)
                        category_results.append(disagg_df)

                if category_results:
                    category_df = pd.concat(category_results, ignore_index=True)
                    # only return the best one for each category
                    category_df = category_df.sort_values(by=['tokens/s/gpu', 'num_total_gpus'], ascending=[False, True]).head(1).reset_index(drop=True)
                    disagg_summary_df_list.append(category_df)
                else:
                    logger.debug(f"No matched result for decode parallel {parallel_value} under diversity constraints.")

            if not disagg_summary_df_list:
                logger.debug("No disagg summary found after applying diversity constraints.")
                return None

            disagg_summary_df = pd.concat(disagg_summary_df_list, ignore_index=True)
            disagg_summary_df = disagg_summary_df.sort_values(by=['tokens/s/gpu'], ascending=[False]).head(return_top_k).reset_index(drop=True)
            return disagg_summary_df

        def _find_best_result_under_constraints(ttft:float, 
                                               tpot:float, 
                                               prefill_summary_df:pd.DataFrame, 
                                               decode_summary_df:pd.DataFrame, 
                                               return_top_k:int, 
                                               num_gpu_list:Optional[List[int]]) -> InferenceSummary:
            """
            Find the best result under constraints
            """
            MAX_NUM_DECODE_worker_CANDIDATES = 64
            MAX_NUM_PREFILL_worker_CANDIDATES = 32
            # 0.7 is empirical value, filter out some workers to improve searching speed
            decode_worker_candidates = decode_summary_df[(decode_summary_df['tpot']<tpot) & (decode_summary_df['tpot']>tpot*0.7)]
            if len(decode_worker_candidates) == 0:
                logger.debug(f"No decode worker candidates found for tpot {tpot}ms.")
                return None
            decode_worker_candidates = decode_worker_candidates.sort_values(by=['seq/s/gpu'], ascending=False).reset_index(drop=True).head(MAX_NUM_DECODE_worker_CANDIDATES)
            # don't filter prefill worker candidates, because prefill worker candidates are much fewer
            prefill_worker_candidates = prefill_summary_df[prefill_summary_df['context_latency']<ttft]
            if len(prefill_worker_candidates) == 0:
                logger.debug(f"No prefill worker candidates found for ttft {ttft}ms.")
                return None
            prefill_worker_candidates = prefill_worker_candidates.sort_values(by=['seq/s/gpu'], ascending=False).reset_index(drop=True).head(MAX_NUM_PREFILL_worker_CANDIDATES)
        
            logger.debug(f"num decode worker candidates: {len(decode_worker_candidates)} num prefill worker candidates: {len(prefill_worker_candidates)}")

            disagg_summary_df = pd.DataFrame(columns=common.ColumnsDisagg)
            # this can be used to reduce the search space, disabled for now due to no strong demand to reduce the searching time.
            logger.debug(f"pmax = {decode_worker_candidates['seq/s'].max()/prefill_worker_candidates['seq/s'].min()}")
            logger.debug(f"dmax = {prefill_worker_candidates['seq/s'].max()/decode_worker_candidates['seq/s'].min()}")
            for prefill_index,  prefill_worker in prefill_worker_candidates.iterrows():
                for decode_index, decode_worker in decode_worker_candidates.iterrows():
                    prefill_throughput = prefill_worker['seq/s']
                    decode_throughput = decode_worker['seq/s']
                    prefill_gpus = prefill_worker['pp']*prefill_worker['tp']*prefill_worker['dp']
                    decode_gpus = decode_worker['pp']*decode_worker['tp']*decode_worker['dp']
                    prefill_num_worker, decode_num_worker = _match_workers(prefill_throughput, prefill_gpus, decode_throughput, decode_gpus, prefill_num_worker_list, decode_num_worker_list, num_gpu_list)
                    if prefill_num_worker == -1 or decode_num_worker == -1:
                        continue
                    disagg_summary_df = pd.concat([disagg_summary_df, self._get_disagg_summary_df(prefill_worker, prefill_num_worker, decode_worker, decode_num_worker)], axis=0, ignore_index=True)
            if len(disagg_summary_df) == 0:
                logger.debug(f"No disagg summary df found for tpot {tpot}ms.")
                return None

            filtered_disagg_summary_df = disagg_summary_df.sort_values(by=['tokens/s/gpu', 'num_total_gpus'], ascending=[False, True]).head(return_top_k).reset_index(drop=True)
            return filtered_disagg_summary_df

        # start, get all possible p/d servers
        if decode_max_num_tokens < 1:
            logger.warning(f"decode_max_num_tokens is less than 1, set to 1")
            decode_max_num_tokens = 1
        decode_batch_size_list_default = list(range(1,16,1))+list(range(16,32,2))+list(range(32,128,4))+list(range(128,512,8))+[512]
        if decode_max_num_tokens > max(decode_batch_size_list_default):
            decode_batch_size_range =  decode_batch_size_list_default + [decode_max_num_tokens]
        else:
            decode_batch_size_range = [i for i in decode_batch_size_list_default if i <= decode_max_num_tokens]

        if prefill_max_num_tokens < runtime_config.isl:
            logger.warning(f"prefill_max_num_tokens is less than runtime_config.isl, set to runtime_config.isl")
            prefill_max_num_tokens = runtime_config.isl
        
        max_prefill_batch_size = prefill_max_num_tokens // runtime_config.isl
        prefill_batch_size_range = range(1, max_prefill_batch_size+1)

        # initialize disagg summary
        disagg_summary = InferenceSummary(runtime_config=runtime_config)
        disagg_summary_df = pd.DataFrame(columns=common.ColumnsDisagg)
        disagg_summary.set_summary_df(disagg_summary_df)
        
        # find prefill and decode workers
        prefill_summary_df = _get_summary_df(
            prefill_model_config, 
            prefill_parallel_config_list, 
            prefill_batch_size_range, 
            runtime_config, 
            'static_ctx',
            latency_correction_scale=self._prefill_latency_correction_scale,
            )
        decode_summary_df = _get_summary_df(
            decode_model_config, 
            decode_parallel_config_list, 
            decode_batch_size_range, 
            runtime_config, 
            'static_gen',
            latency_correction_scale=self._decode_latency_correction_scale
            )
        if len(prefill_summary_df) == 0 or len(decode_summary_df) == 0:
            logger.debug(f"No prefill or decode workers found for {model_name} with given configs.")
            return disagg_summary
        
        # find best result under constraints
        ttft = runtime_config.ttft
        tpot_list = runtime_config.tpot if isinstance(runtime_config.tpot, list) else [runtime_config.tpot]
        for tpot in tpot_list:
            logger.debug(f"Finding best result under constraints for tpot={tpot}ms...")
            filtered_disagg_summary_df = _find_best_result_under_constraints_with_diversity(
                ttft=ttft, 
                tpot=tpot, 
                prefill_summary_df=prefill_summary_df, 
                decode_summary_df=decode_summary_df, 
                return_top_k=5, 
                num_gpu_list=num_gpu_list, 
                rate_matching_prefill_degradation_factor=self._RATE_MATCHING_PREFILL_DEGRADATION_FACTOR, 
                rate_matching_decode_degradation_factor=self._RATE_MATCHING_DECODE_DEGRADATION_FACTOR
                )
            if filtered_disagg_summary_df is not None:
                disagg_summary_df = pd.concat([disagg_summary_df, filtered_disagg_summary_df], axis=0, ignore_index=True)
        if len(disagg_summary_df) == 0:
            logger.debug(f"No disagg result found for {model_name} with given constraints.")
            return disagg_summary

        # set final disagg summary
        disagg_summary.set_summary_df(disagg_summary_df)
        return disagg_summary