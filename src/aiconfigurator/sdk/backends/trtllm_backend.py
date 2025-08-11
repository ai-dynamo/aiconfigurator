# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiconfigurator.sdk.backends.base_backend import BaseBackend
from aiconfigurator.sdk import common
from aiconfigurator.sdk.models import BaseModel, get_model_family
from aiconfigurator.sdk.perf_database import PerfDatabase
from aiconfigurator.sdk.config import RuntimeConfig
from aiconfigurator.sdk.inference_summary import InferenceSummary
import numpy as np
import pandas as pd
from collections import defaultdict
import logging
logger = logging.getLogger(__name__)

class TRTLLMBackend(BaseBackend):
    """
    TRTLLM backend.
    """
    def __init__(self,):
        super().__init__()
        self._ifb_cache = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))

    def run_ifb(self, 
                model: BaseModel, 
                database: PerfDatabase, 
                runtime_config: RuntimeConfig, 
                **kwargs) -> InferenceSummary:
        """
        Run the IFB inference.
        """
        isl = runtime_config.isl
        osl = runtime_config.osl
        b = runtime_config.batch_size
        ctx_tokens = kwargs.get('ctx_tokens', None)
        assert ctx_tokens is not None, 'ctx_tokens is required'
        balance_score = isl * b / ctx_tokens / osl

        try:
            summary = self._ifb_cache[isl][osl][b][ctx_tokens]
        except KeyError:
            # we would like to calculate num_mix_steps and num_genonly_steps based on isl, osl, b, ctx_tokens
            # within osl steps, need to finish all the ctx tokens
            steps_to_finish_ctx = np.ceil(isl * b / ctx_tokens)
            num_mix_steps = num_genonly_steps = 0
            num_mix_steps_for_tpot_calc = 0 # this is a correction for tpot calc only.
            if b > 1:
                if steps_to_finish_ctx >= osl:
                    num_mix_steps = steps_to_finish_ctx
                    num_mix_ctx_tokens = ctx_tokens
                    num_mix_gen_tokens = max(1, b//(steps_to_finish_ctx/osl))
                    num_genonly_steps = 0
                    num_genonly_tokens = 0
                    num_mix_steps_for_tpot_calc = num_mix_steps
                else:
                    # 3-step is an empirical correction for pipelining requests where new requests cannot be enqueued immediately after last request's exit
                    num_mix_steps = steps_to_finish_ctx
                    num_mix_ctx_tokens = ctx_tokens
                    num_mix_gen_tokens = b - np.ceil(ctx_tokens/isl) # the error check is outside
                    assert num_mix_gen_tokens >= 1, f'num_mix_gen_tokens: {num_mix_gen_tokens}, b: {b}, ctx_tokens: {ctx_tokens}, isl: {isl}'
                    num_genonly_steps = osl - num_mix_steps
                    num_genonly_tokens = b
                    num_mix_steps_for_tpot_calc = max(1, num_mix_steps - 3)
            elif b == 1:
            # special case for b=1
                num_mix_steps = 1
                num_mix_ctx_tokens = ctx_tokens
                num_mix_gen_tokens = 0
                num_genonly_steps = osl-1
                num_genonly_tokens = 1
                num_mix_steps_for_tpot_calc = 0

            def _get_mix_step_latency(model: BaseModel, database: PerfDatabase, ctx_tokens: int, gen_tokens: int, isl: int, osl: int) -> float:
                num_tokens = ctx_tokens + gen_tokens
                summary = self.run_static(model, database, RuntimeConfig(batch_size=1, beam_width=1, isl=num_tokens, osl=1), mode='static_ctx')
                latency_dict = summary.get_context_latency_dict()
                non_attention_latency = 0.
                #TODO, fix for DS. DS has different ops for attn in ctx and gen. 
                for layer_name, latency in latency_dict.items():
                    if layer_name != 'context_attention':
                        non_attention_latency += latency

                # second pass to get ctx attn, split full isl over num_steps(=np.ceil(isl/ctx_tokens)), average the ctx attn latency
                num_tokens = isl
                summary = self.run_static(model, database, RuntimeConfig(batch_size=1, beam_width=1, isl=num_tokens, osl=1), mode='static_ctx')
                latency_dict = summary.get_context_latency_dict()
                ctx_attention_latency = latency_dict['context_attention'] / (np.ceil(isl/ctx_tokens))
            
                # third pass to get generation attn. use isl+osl//2 for avg generation attn latency.
                if gen_tokens > 0:
                    num_tokens = gen_tokens
                    summary = self.run_static(model, database, RuntimeConfig(batch_size=num_tokens, beam_width=1, isl=isl+osl//2, osl=2), mode='static_gen')
                    latency_dict = summary.get_generation_latency_dict()
                    gen_attention_latency = latency_dict['generation_attention']
                else:
                    gen_attention_latency = 0.

                return non_attention_latency + ctx_attention_latency + gen_attention_latency
                
            def _get_genonly_step_latency(model: BaseModel, database: PerfDatabase, gen_tokens: int, isl: int, osl: int) -> float:
                if gen_tokens <= 0:
                    return 0.
                num_tokens = gen_tokens
                summary = self.run_static(model, database, RuntimeConfig(batch_size=num_tokens, beam_width=1, isl=isl+osl//2, osl=2), mode='static_gen')
                latency_dict = summary.get_generation_latency_dict()
                genonly_step_latency = 0.
                for layer_name, latency in latency_dict.items():
                    genonly_step_latency += latency
                
                return genonly_step_latency

            mix_step_latency = _get_mix_step_latency(model, database, num_mix_ctx_tokens, num_mix_gen_tokens, isl, osl)
            genonly_step_latency = _get_genonly_step_latency(model, database, num_genonly_tokens, isl, osl)

            ttft = mix_step_latency * np.ceil(isl/ctx_tokens)
            tpot = (mix_step_latency * num_mix_steps_for_tpot_calc + genonly_step_latency * num_genonly_steps) / (num_mix_steps_for_tpot_calc + num_genonly_steps)
            output_throughput = 1000 / (num_mix_steps*mix_step_latency + num_genonly_steps*genonly_step_latency) * b * (osl-1)
            logger.debug(f'ctx_tokens: {ctx_tokens}, b: {b}, osl: {osl}, isl: {isl}, num_mix_steps: {num_mix_steps}, num_genonly_steps: {num_genonly_steps}, num_mix_ctx_tokens: {num_mix_ctx_tokens}, num_mix_gen_tokens: {num_mix_gen_tokens}, num_genonly_tokens: {num_genonly_tokens}')
            logger.debug(f'mix_step_latency: {mix_step_latency}, genonly_step_latency: {genonly_step_latency}')
            logger.debug(f'ttft: {ttft}, tpot: {tpot}, output_throughput: {output_throughput}')

            num_ctx_requests = np.ceil(ctx_tokens/isl)
            num_gen_requests = b - num_ctx_requests
            if b == 1:
                num_ctx_requests = 1
                num_gen_requests = 1

            # correct output_throughput and concurrency for attention dp (global batch)
            scale_factor = model.config.pp_size * model.config.attention_dp_size
            output_throughput = output_throughput * scale_factor
            concurrency = b * scale_factor

            request_rate = output_throughput / (osl - 1)
            if b > 1:
                num_tokens = num_gen_requests+ctx_tokens # will not be corrected by balance score when it's larger than 1.0 in order to indicate what's happening
            else:
                num_tokens = ctx_tokens
            memory = self._get_memory_usage(model, database, b, 1, isl, osl, num_tokens)
            tp = model.config.tp_size
            pp = model.config.pp_size
            dp = model.config.attention_dp_size
            moe_tp = model.config.moe_tp_size
            moe_ep = model.config.moe_ep_size
            tokens_s_gpu = output_throughput/pp/tp/dp
            tokens_s_user = 1000/tpot
            seq_s = request_rate
            seq_s_gpu = seq_s/pp/tp/dp
            tokens_s = output_throughput
            num_total_gpus = tp*pp*dp
            parallel = f'tp{tp}pp{pp}dp{dp}etp{moe_tp}ep{moe_ep}'
            gemm = model.config.gemm_quant_mode.name
            kvcache = model.config.kvcache_quant_mode.name
            fmha = model.config.fmha_quant_mode.name
            moe = model.config.moe_quant_mode.name
            comm = model.config.comm_quant_mode.name
            mem = memory['total']
            
            result = pd.DataFrame(columns=common.ColumnsIFB, 
                                  data=[[model.model_name, isl, osl, \
                                         concurrency, request_rate, b, b*model.config.attention_dp_size, \
                                         ttft, tpot, seq_s, seq_s_gpu, tokens_s, tokens_s_gpu, tokens_s_user, \
                                         num_total_gpus, \
                                         tp, pp, dp, moe_tp, moe_ep, parallel, \
                                         gemm, kvcache, fmha, moe, comm, \
                                         mem, \
                                         balance_score, num_ctx_requests, num_gen_requests, num_tokens, ctx_tokens, num_gen_requests, \
                                         database.backend, database.version, database.system]]).round(3)
            summary = InferenceSummary(RuntimeConfig(isl=isl, osl=osl))
            summary.set_memory_and_check_oom(memory, database.system_spec['gpu']['mem_capacity'])
            summary.set_summary_df(result)
        
            # caching
            self._ifb_cache[isl][osl][b][ctx_tokens] = summary

        return summary

    def find_best_ifb_result_under_constraints(self, 
                                               model: BaseModel, 
                                               database: PerfDatabase, 
                                               runtime_config: RuntimeConfig, 
                                               **kwargs) -> InferenceSummary:
        """
        Find the best IFB result under constraints.

        Args:
            model: the model to be tested
            database: the database to be tested
            runtime_config: the runtime configuration
            top_k: the number of best results to return
            max_batch_size: the maximum batch size to test
            ctx_stride: the stride of ctx tokens to test, it will impact the time to run the test.

        Returns:
            A summary of the best IFB result under constraints.
        """
        isl = runtime_config.isl
        osl = runtime_config.osl
        ttft = runtime_config.ttft
        tpot = runtime_config.tpot

        top_k = kwargs.get('top_k', 1)
        max_batch_size = kwargs.get('max_batch_size', 512)
        ctx_stride = kwargs.get('ctx_stride', 512)

        MAX_NORMAL_CTX_TOKENS = 8192
        MAX_CTX_TOKENS_MULTIPLE_OF_ISL = 2
        MAX_CTX_TOKENS_SEARCH_STEPS = 8 # for ctx stride large for faster sweeping

        max_ctx_tokens = max(MAX_NORMAL_CTX_TOKENS, isl*MAX_CTX_TOKENS_MULTIPLE_OF_ISL)
        ctx_stride_large = max(1024, ctx_stride, max_ctx_tokens//MAX_CTX_TOKENS_SEARCH_STEPS) # if ctx tokens is already larger than 2048, we need to increase ctx_stride for faster sweeping
        # when b is larger than 1024, the result is not good as the data collection is not enough to cover this.
        b_list_default = list(range(1,16,1))+list(range(16,32,4))+list(range(32,64,8))+list(range(64,256,16))+list(range(256,512,32))+list(range(512,1024,256))+[1024]
    
        # sweep for batch_size and ctx_tokens
        # ctx_tokens will have a step of ctx_stride. When it's larger than 8192, we will increase the step to ctx_stride_large.
        # outer_loop is over batch_size dimention, from 1 to max_batch_size
        # inner_loop is over ctx_tokens dimention, from 0 to max_ctx_tokens where it's max(8192, 4*isl)
        # during the loop, as b, ctx_tokens and system memory are monotonic, we can break the inner loop when the system is oom.
        b_list = [b for b in b_list_default if b <= max_batch_size]
        # prepare ctx_tokens_list
        ctx_tokens_list = []
        ctx_tokens = 0
        while True:
            ctx_tokens = (ctx_tokens + ctx_stride) if ctx_tokens < MAX_NORMAL_CTX_TOKENS else (ctx_tokens + ctx_stride_large)
            if ctx_tokens > max_ctx_tokens:
                break
            ctx_tokens_list.append(ctx_tokens)
        # add those just match the multiple of isl
        for i in range(1,MAX_CTX_TOKENS_MULTIPLE_OF_ISL+1):
            ctx_tokens = isl * i
            if ctx_tokens not in ctx_tokens_list:
                ctx_tokens_list.append(ctx_tokens)
        ctx_tokens_list.sort()
        
        results_df = pd.DataFrame(columns=common.ColumnsIFB)
        capped_b = []
        for b in b_list:
            for ctx_tokens in ctx_tokens_list:
                if (b - np.ceil(ctx_tokens/isl) < 0): # allow b==1
                    break

                if b > 1 and (b - np.ceil(ctx_tokens/isl) < 1): # general case, to ensure there's at least one gen req
                    break

                # filter out repeated records for balance score correction
                balance_score = isl * b / ctx_tokens / osl 
                if balance_score > 1:
                    gen_tokens = b // balance_score
                    if gen_tokens > 1 and gen_tokens in capped_b:
                        continue
                    else:
                        capped_b.append(gen_tokens)

                summary = self.run_ifb(model=model, database=database, runtime_config=RuntimeConfig(batch_size=b, isl=isl, osl=osl), ctx_tokens=ctx_tokens)

                if summary.check_oom():
                    break # larger ctx tokens will cause oom
                if summary.get_summary_df().loc[0,'tpot'] <= tpot and summary.get_summary_df().loc[0,'ttft'] <= ttft:
                    if len(results_df) == 0:
                        results_df = summary.get_summary_df()
                    else:
                        results_df = pd.concat([results_df, summary.get_summary_df()], axis=0, ignore_index=True)

        sorted_results_df = results_df.sort_values(by='seq/s', ascending=False).round(3)
        if top_k > 0:
            sorted_results_df = sorted_results_df.head(top_k)
        
        summary = InferenceSummary(runtime_config)
        summary.set_summary_df(sorted_results_df)
        return summary
    
    def _get_memory_usage(self, 
                          model: BaseModel, 
                          database: PerfDatabase, 
                          batch_size: int, 
                          beam_width: int, 
                          isl: int, 
                          osl: int, 
                          num_tokens: int = 0) -> dict[str, float]:
        """
        Get the memory usage of the backend.
        """
        weights, activations, kvcache = 0., 0., 0.
        for op in model.context_ops:
            weights += op.get_weights()
        
        # count weights on a single GPU
        weights /= model.config.pp_size
        
        h = model._num_heads*model._head_size
        if num_tokens == 0:
            num_tokens = isl*batch_size
        
        # ==== this below section is backend specific ====
        if get_model_family(model.model_name) == 'GPT':
            c_dict = {1:10, 2:6, 4:5, 8:5}
            activations = 2*num_tokens*h*c_dict[min(model.config.tp_size, 8)]
            activations = max(activations, 70*1024*1024) # minimum act
        elif get_model_family(model.model_name) == 'LLAMA':
            c_dict = {1:11, 2:6.5, 4:5, 8:5}
            activations = 2*num_tokens*h*c_dict[min(model.config.tp_size, 8)]
            activations = max(activations, 70*1024*1024) # minimum act
        elif get_model_family(model.model_name) == 'MOE':
            c_dict = {1:22, 2:13, 4:10, 8:10}
            activations = 2*num_tokens*h*c_dict[min(model.config.tp_size, 8)]
            activations = max(activations, 70*1024*1024) # minimum act
        elif get_model_family(model.model_name) == 'DEEPSEEK':
            c_dict = {1:22, 2:13, 4:10, 8:10}
            activations = 2*num_tokens*h*c_dict[min(model.config.tp_size, 8)]
            # moe workspace, 128 for block scale, float for 4bytes
            activations += num_tokens * h * model.config.attention_dp_size * model._num_experts * model._topk \
                /model.config.moe_ep_size / 128 * 4 # still an improvement opportunity in trtllm to achieve this.
            # nextn correction for ds only, MTP
            if model.config.nextn > 0:
                activations = activations * (model.config.nextn+1)
            activations = max(activations, 70*1024*1024) # minimum act
        else:
            c_dict = {1:10, 2:6, 4:5, 8:5}  # 4+6/TP, fp8 will have relatively low act, but ignore here. need more experiments
            activations = 2*num_tokens*h*c_dict[min(model.config.tp_size, 8)]
            activations = max(activations, 70*1024*1024) # minimum act
        # ==== this above section is backend specific ====

        if get_model_family(model.model_name) == 'DEEPSEEK':
            kvcache_per_token = model._num_layers*576
        else:
            num_kv_heads_per_GPU = (model._num_kv_heads+model.config.tp_size-1)//model.config.tp_size
            kvcache_per_token = num_kv_heads_per_GPU*model._head_size*model._num_layers*2
        # should not be divided by pp_size as you need to hold all kvcache for stages.
        kvcache = (batch_size*isl+batch_size*beam_width*osl)*model.config.kvcache_quant_mode.value.memory*kvcache_per_token
        #if 'DEEPSEEK' in model.model_name or 'MOE' in model.model_name:
        #    kvcache = kvcache * model.config.attention_dp_size # this is incorrect. tp will duplicate the kvcache while attn_dp will not.

        # starting from 2.22
        nccl_mem = database.system_spec['misc']['nccl_mem'][min(model.config.tp_size, 8)]

        # cuda, cublas, etc.
        others_mem = database.system_spec['misc']['other_mem']

        OneGiB = 1<<30
        return {'total':(weights+activations+kvcache+nccl_mem+others_mem)/OneGiB,
                'weights':weights/OneGiB,
                'activations':activations/OneGiB,
                'kvcache':kvcache/OneGiB,
                'nccl':nccl_mem/OneGiB,
                'others':others_mem/OneGiB}
    

