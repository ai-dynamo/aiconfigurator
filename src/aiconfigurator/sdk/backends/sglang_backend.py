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
        self.name = common.BackendName.sglang

    def run_agg(self, 
                model: BaseModel, 
                database: PerfDatabase, 
                runtime_config: RuntimeConfig, 
                **kwargs) -> InferenceSummary:
        pass

    def find_best_agg_result_under_constraints(self, 
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
            c_dict = {1: 13, 2: 8, 4: 6.5, 8: 6.5}  
            activations = 2 * num_tokens * h * c_dict[min(model.config.tp_size, 8)]
            activations = max(activations, 90 * 1024 * 1024)  # Higher minimum for SGLANG
        elif 'LLAMA' in model.model_name:
            c_dict = {1: 14, 2: 8.5, 4: 6.5, 8: 6.5} 
            activations = 2 * num_tokens * h * c_dict[min(model.config.tp_size, 8)]
            activations = max(activations, 90 * 1024 * 1024)  # Higher minimum for SGLANG
        elif 'MOE' in model.model_name:
            c_dict = {1: 28, 2: 17, 4: 13, 8: 13} 
            activations = 2 * num_tokens * h * c_dict[min(model.config.tp_size, 8)]
            activations = max(activations, 90 * 1024 * 1024)  # Higher minimum for SGLANG
        elif 'DEEPSEEK' in model.model_name:
            c_dict = {1: 28, 2: 17, 4: 13, 8: 13}
            activations = 2 * num_tokens * h * c_dict[min(model.config.tp_size, 8)]
            activations += num_tokens * h * model.config.attention_dp_size * model._num_experts * model._topk \
                / model.config.moe_ep_size / 128 * 4
            if model.config.nextn > 0:
                activations = activations * (model.config.nextn + 1)
            activations = max(activations, 90 * 1024 * 1024)  # Higher minimum for SGLANG
        else:
            # Default case - increased coefficients for SGLANG
            c_dict = {1: 13, 2: 8, 4: 6.5, 8: 6.5} 
            activations = 2 * num_tokens * h * c_dict[min(model.config.tp_size, 8)]
            activations = max(activations, 90 * 1024 * 1024)  # Higher minimum for SGLANG

        sglang_overhead = activations * 0.15  # 15% additional overhead for SGLANG
        activations += sglang_overhead
        
        # ==== KV Cache calculation - SGLANG specific ====
        if 'DEEPSEEK' in model.model_name:
            kvcache_per_token = model._num_layers * 576
        else:
            num_kv_heads_per_GPU = (model._num_kv_heads + model.config.tp_size - 1) // model.config.tp_size
            kvcache_per_token = num_kv_heads_per_GPU * model._head_size * model._num_layers * 2
        
        kvcache = (batch_size * isl + batch_size * beam_width * osl) * model.config.kvcache_quant_mode.value.memory * kvcache_per_token
        
        # ==== Communication and system memory ====
        nccl_mem = database.system_spec['misc']['nccl_mem'][min(model.config.tp_size, 8)]
        
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



