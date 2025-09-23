# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiconfigurator.sdk.backends.base_backend import BaseBackend
from aiconfigurator.sdk import common
from aiconfigurator.sdk.config import RuntimeConfig
from aiconfigurator.sdk.inference_summary import InferenceSummary
import numpy as np
import pandas as pd
import copy
from collections import defaultdict
from aiconfigurator.sdk.models import BaseModel
from aiconfigurator.sdk.perf_database import PerfDatabase

class SGLANGBackend(BaseBackend):
    """
    SGLANG backend.
    """
    def __init__(self):
        super().__init__()

    def run_disagg(self, 
                model: BaseModel, 
                database: PerfDatabase, 
                runtime_config: RuntimeConfig, 
                mode: str, 
                stride: int = 32) -> InferenceSummary:
        """
        Run the disaggregated inference with different batch sizes for prefill and decode.
        
        Args:
            model: The model to run inference on
            database: The performance database
            runtime_config: Runtime configuration with batch_size for prefill and decode_bs for decode
            mode: Must be 'disagg' for disaggregated inference
            stride: Stride for generation acceleration (default: 32)
            
        Returns:
            InferenceSummary: Summary of the disaggregated inference result
        """
        def _run_context(batch_size: int, isl: int) -> dict[str, float]:
            context_latency_dict = defaultdict(float)

            for op in model.context_ops:
                #query latency and store the latency
                x = batch_size*isl if 'logits_gemm' not in op._name else batch_size
                latency = op.query(database, x=x, batch_size=batch_size, beam_width=1, s=isl)
                context_latency_dict[op._name] += latency

            return context_latency_dict

        def _run_generation(batch_size: int, beam_width: int, isl: int, osl: int, stride: int) -> dict[str, float]:
            # mtp/speculative decoding correction
            batch_size = batch_size*(model._nextn+1)

            latencies = []
            cached_latency_dict = None
            for i in range(osl-1):

                if i%stride != 0:
                    latencies.append(copy.deepcopy(cached_latency_dict))
                    continue

                latency_dict = defaultdict(float)
                for op in model.generation_ops:
                    latency = op.query(database, x=batch_size*beam_width, batch_size=batch_size, beam_width=beam_width, s=isl+i+1, is_context=False)
                    latency_dict[op._name] += latency
                cached_latency_dict = latency_dict

                latencies.append(latency_dict)
        
            generation_latency_dict = {}
            if len(latencies) > 0:
                for key in latencies[0].keys():
                    generation_latency_dict[key] = 0.0
                    for latency_dict in latencies:
                        generation_latency_dict[key] += latency_dict[key]
            
            return generation_latency_dict

        summary = InferenceSummary(runtime_config)
        prefill_batch_size = runtime_config.batch_size
        decode_batch_size = runtime_config.decode_bs if runtime_config.decode_bs is not None else runtime_config.batch_size
        beam_width, isl, osl = runtime_config.beam_width, runtime_config.isl, runtime_config.osl
        
        # For disaggregated mode, run both prefill and decode with different batch sizes
        context_latency_dict = _run_context(prefill_batch_size, isl)
        generation_latency_dict = _run_generation(decode_batch_size, beam_width, isl, osl, stride)
        # Use the larger batch size for memory calculation to be conservative
        max_batch_size = max(prefill_batch_size, decode_batch_size)
        memory = self._get_memory_usage(model, database, max_batch_size, beam_width, isl, osl)

        context_latency, generation_latency = 0.0, 0.0
        for op, op_latency in context_latency_dict.items():
            context_latency += op_latency
        for op, op_latency in generation_latency_dict.items():
            generation_latency += op_latency

        # For disaggregated mode, use prefill batch size for overall calculation
        bs = prefill_batch_size
            
        global_bs = bs * model.config.attention_dp_size
        concurrency = global_bs
        latency = context_latency + generation_latency
        request_rate = 0.0
        ttft = context_latency
        tpot = 0.0 if osl <= 1 else generation_latency / (osl-1)
        seq_s = 0.0 if latency == 0.0 else global_bs / latency * 1000 * model.config.pp_size
        seq_s_gpu = seq_s / model.config.tp_size / model.config.pp_size / model.config.attention_dp_size
        tokens_s = seq_s * osl  # For disaggregated mode, include all tokens
        tokens_s_gpu = tokens_s / model.config.tp_size / model.config.pp_size / model.config.attention_dp_size
        tokens_s_user = 0.0 if tpot == 0.0 else 1000.0 / tpot
        tp = model.config.tp_size
        pp = model.config.pp_size
        dp = model.config.attention_dp_size
        moe_tp = model.config.moe_tp_size
        moe_ep = model.config.moe_ep_size
        num_total_gpus = tp*pp*dp
        parallel = f'tp{tp}pp{pp}dp{dp}etp{moe_tp}ep{moe_ep}'
        gemm = model.config.gemm_quant_mode.name
        kvcache = model.config.kvcache_quant_mode.name
        fmha = model.config.fmha_quant_mode.name
        moe = model.config.moe_quant_mode.name
        comm = model.config.comm_quant_mode.name
        mem = memory['total']
        
        
        data = [[model.model_name, isl, osl, \
                 concurrency, request_rate, bs, global_bs, \
                 ttft, tpot, seq_s, seq_s_gpu, tokens_s, tokens_s_gpu, tokens_s_user, latency, context_latency, generation_latency, \
                 num_total_gpus, \
                 tp, pp, dp, moe_tp, moe_ep, parallel, \
                 gemm, kvcache, fmha, moe, comm, \
                 mem,
                 database.backend, database.version, database.system]]

        summary_df = pd.DataFrame(data, columns=common.ColumnsStatic).round(3)

        summary.set_context_latency_dict(context_latency_dict)
        summary.set_generation_latency_dict(generation_latency_dict)
        summary.set_memory_and_check_oom(memory, database.system_spec['gpu']['mem_capacity'])
        summary.set_summary_df(summary_df)

        return summary

    def run_ifb(self, 
                model: BaseModel, 
                database: PerfDatabase, 
                runtime_config: RuntimeConfig, 
                **kwargs) -> InferenceSummary:
        pass

    def find_best_ifb_result_under_constraints(self, 
                                               model: BaseModel, 
                                               database: PerfDatabase, 
                                               runtime_config: RuntimeConfig, 
                                               **kwargs) -> InferenceSummary:
        pass
    
    def _get_memory_usage(self, 
                          model: BaseModel, 
                          database: PerfDatabase, 
                          batch_size: int, 
                          beam_width: int, 
                          isl: int, 
                          osl: int, 
                          num_tokens: int = 0) -> dict[str, float]:
        """
        Get the memory usage of the SGLANG backend.
        
        SGLANG backend typically has different memory characteristics compared to TRTLLM:
        - Generally higher activation memory due to Python overhead and dynamic execution
        - May have different KV cache management strategies
        - Communication patterns may differ from NCCL-based systems
        """
        weights, activations, kvcache = 0., 0., 0.
        
        # Calculate weights memory - same as TRTLLM
        for op in model.context_ops:
            weights += op.get_weights()
        
        # Count weights on a single GPU
        weights /= model.config.pp_size
        
        h = model._num_heads * model._head_size
        if num_tokens == 0:
            num_tokens = isl * batch_size
        
        # ==== SGLANG backend specific memory calculations ====
        # SGLANG typically has higher activation memory due to Python overhead
        # and dynamic execution patterns
        if 'GPT' in model.model_name:
            # SGLANG overhead: increase coefficients by 20-30% compared to TRTLLM
            c_dict = {1: 13, 2: 8, 4: 6.5, 8: 6.5}  # Increased from TRTLLM values
            activations = 2 * num_tokens * h * c_dict[min(model.config.tp_size, 8)]
            activations = max(activations, 90 * 1024 * 1024)  # Higher minimum for SGLANG
        elif 'LLAMA' in model.model_name:
            c_dict = {1: 14, 2: 8.5, 4: 6.5, 8: 6.5}  # Increased from TRTLLM values
            activations = 2 * num_tokens * h * c_dict[min(model.config.tp_size, 8)]
            activations = max(activations, 90 * 1024 * 1024)  # Higher minimum for SGLANG
        elif 'MOE' in model.model_name:
            c_dict = {1: 28, 2: 17, 4: 13, 8: 13}  # Increased from TRTLLM values
            activations = 2 * num_tokens * h * c_dict[min(model.config.tp_size, 8)]
            activations = max(activations, 90 * 1024 * 1024)  # Higher minimum for SGLANG
        elif 'DEEPSEEK' in model.model_name:
            c_dict = {1: 28, 2: 17, 4: 13, 8: 13}  # Increased from TRTLLM values
            activations = 2 * num_tokens * h * c_dict[min(model.config.tp_size, 8)]
            # MOE workspace - SGLANG may have different memory management
            activations += num_tokens * h * model.config.attention_dp_size * model._num_experts * model._topk \
                / model.config.moe_ep_size / 128 * 4
            # NextN correction for DeepSeek
            if model.config.nextn > 0:
                activations = activations * (model.config.nextn + 1)
            activations = max(activations, 90 * 1024 * 1024)  # Higher minimum for SGLANG
        else:
            # Default case - increased coefficients for SGLANG
            c_dict = {1: 13, 2: 8, 4: 6.5, 8: 6.5}  # Increased from TRTLLM values
            activations = 2 * num_tokens * h * c_dict[min(model.config.tp_size, 8)]
            activations = max(activations, 90 * 1024 * 1024)  # Higher minimum for SGLANG
        
        # Add SGLANG-specific overhead for dynamic execution and Python runtime
        sglang_overhead = activations * 0.15  # 15% additional overhead for SGLANG
        activations += sglang_overhead
        
        # ==== KV Cache calculation - SGLANG specific ====
        if 'DEEPSEEK' in model.model_name:
            kvcache_per_token = model._num_layers * 576
        else:
            num_kv_heads_per_GPU = (model._num_kv_heads + model.config.tp_size - 1) // model.config.tp_size
            kvcache_per_token = num_kv_heads_per_GPU * model._head_size * model._num_layers * 2
        
        # SGLANG KV cache - may have different quantization or management
        # Use the same calculation as TRTLLM for now, but could be adjusted based on SGLANG specifics
        kvcache = (batch_size * isl + batch_size * beam_width * osl) * model.config.kvcache_quant_mode.value.memory * kvcache_per_token
        
        # ==== Communication and system memory ====
        # SGLANG may use different communication patterns than NCCL
        # For now, use the same NCCL memory calculation but could be adjusted
        nccl_mem = database.system_spec['misc']['nccl_mem'][min(model.config.tp_size, 8)]
        
        # System memory - SGLANG may have different overhead
        others_mem = database.system_spec['misc']['other_mem']
        
        # Add SGLANG-specific system overhead
        sglang_system_overhead = others_mem * 0.2  # 20% additional system overhead for SGLANG
        others_mem += sglang_system_overhead
        
        OneGiB = 1 << 30
        return {
            'total': (weights + activations + kvcache + nccl_mem + others_mem) / OneGiB,
            'weights': weights / OneGiB,
            'activations': activations / OneGiB,
            'kvcache': kvcache / OneGiB,
            'nccl': nccl_mem / OneGiB,
            'others': others_mem / OneGiB
        }



