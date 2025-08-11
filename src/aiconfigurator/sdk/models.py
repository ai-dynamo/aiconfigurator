# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import aiconfigurator.sdk.operations as ops
from aiconfigurator.sdk import common
from aiconfigurator.sdk import config
import logging

from typing import Union, List
from dataclasses import dataclass 

logger = logging.getLogger(__name__)

def get_model(model_name: str, model_config: config.ModelConfig) -> BaseModel:
    """
    Get model.
    """
    assert(model_name in common.SupportedModels), f"unsupport model {model_name}"
    model_family,l,n,n_kv,d,hidden,inter,vocab,context,topk,num_experts,moe_inter_size, extra_params = common.SupportedModels[model_name]
    assert(model_family in common.ModelFamily), f"model is not in ModelFamily(GPT, LLAMA, MOE, DEEPSEEK, NEMOTRONNAS)"

    if model_config.overwrite_num_layers > 0:
        l = model_config.overwrite_num_layers

    if model_family == 'GPT':
        model = GPTModel(model_name, model_family, l, n, n_kv, d, \
                    hidden, inter, vocab, context, \
                    model_config)
    elif model_family == 'LLAMA':
        model = LLAMAModel(model_name, model_family, l, n, n_kv, d, \
                    hidden, inter, vocab, context, \
                    model_config)
    elif model_family == 'MOE':
        model = MOEModel(topk, num_experts, moe_inter_size, \
                         model_name, model_family, l, n, n_kv, d, \
                    hidden, inter, vocab, context, \
                    model_config)
    elif model_family == 'DEEPSEEK':
        model = DeepSeekModel(topk, num_experts, moe_inter_size, \
                         model_name, model_family, l, n, n_kv, d, \
                    hidden, inter, vocab, context, \
                    model_config)
    elif model_family == 'NEMOTRONNAS':
        model = NemotronNas(model_name, model_family, l, n, n_kv, d, \
                            hidden, inter, vocab, context, \
                            model_config)
        model.context_ops = extra_params
        model.generation_ops = extra_params

    return model

def get_model_family(model_name: str) -> str:
    """
    Get model family.
    """
    assert(model_name in common.SupportedModels), f"unsupport model {model_name}"
    model_family,l,n,n_kv,d,hidden,inter,vocab,context,topk,num_experts,moe_inter_size, extra_params = common.SupportedModels[model_name]
    return model_family

def check_is_moe(model_name: str) -> bool:
    """
    Check if the model is a MoE model.
    """
    return get_model_family(model_name) == 'MOE' or get_model_family(model_name) == 'DEEPSEEK'

def calc_expectation(nextn: int, nextn_accept_rates: list[float]) -> float:
    """
    Calculate expectation for mtp
    """
    prob = 1.
    if nextn == 0:
        return 0.0
    
    for i in range(nextn):
        prob *= nextn_accept_rates[i]
    if nextn > 1:
        return prob + calc_expectation(nextn-1, nextn_accept_rates)
    else:
        return prob

class BaseModel(object):
    """
    Base model class.
    """
    def __init__(self, model_name: str, model_family: str, num_layers: int, num_heads: int, num_kv_heads: int, head_size: int, 
                 hidden_size: int, inter_size: int, vocab_size: int, context_length: int, model_config: config.ModelConfig) -> None:
        
        self.model_name = model_name
        self.model_family = model_family
        self.config = model_config
        self.context_ops = []
        self.generation_ops = []
        
        # internal only
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._num_kv_heads = num_kv_heads
        self._head_size = head_size
        self._hidden_size = hidden_size
        self._inter_size = inter_size
        self._vocab_size = vocab_size
        self._context_length = context_length
        self._num_kv_heads_per_GPU = (self._num_kv_heads+model_config.tp_size-1)//model_config.tp_size

        if self._num_layers % model_config.pp_size != 0:
            logger.warning(f"num_layers {self._num_layers} is not divisible by pp_size {model_config.pp_size}. \
                           this will introduce additional rounding error. Currently we're nothing to correct this.")

        assert(self._num_heads % model_config.tp_size == 0 and self._num_heads // model_config.tp_size >= 4), \
            f"num_heads {self._num_heads} should be divisible by tp_size {model_config.tp_size} and the division result should be >= 4"

        self._nextn = model_config.nextn
        self._nextn_accept_rates = model_config.nextn_accept_rates

class GPTModel(BaseModel):
    """
    GPT series uses this model impl.
    Some rules to follow,
    Due to implementation, attn layer name needs to be context_attention or generation_attention, exact match is required. Same for logits_gemm.
    Other than DS V3, all other models don't support mtp
    """
    def __init__(self, *args) -> None:
        super().__init__(*args)
        assert self._nextn == 0, 'Only DS V3 supports mtp'

        h = self._hidden_size
        tp_size = self.config.tp_size
        pp_size = self.config.pp_size
        num_kv_heads_per_GPU = self._num_kv_heads_per_GPU
        gemm_quant_mode = self.config.gemm_quant_mode
        kvcache_quant_mode = self.config.kvcache_quant_mode
        fmha_quant_mode = self.config.fmha_quant_mode

        self.context_ops.extend([ops.Embedding(f'context_embedding', 1, self._vocab_size, h, 0.3),
                                ops.ElementWise(f'context_add_norm_1', self._num_layers, 2*h, 2*h, 0.8),
                                ops.GEMM(f'context_qkv_gemm', self._num_layers, self._num_heads*self._head_size//tp_size + self._head_size*num_kv_heads_per_GPU*2, h, gemm_quant_mode), 
                                ops.ContextAttention(f'context_attention', self._num_layers, self._num_heads//tp_size, num_kv_heads_per_GPU, kvcache_quant_mode, fmha_quant_mode),
                                ops.GEMM(f'context_proj_gemm', self._num_layers, h, self._num_heads*self._head_size//tp_size, gemm_quant_mode),
                                ops.ElementWise(f'context_add_norm_2', self._num_layers, 2*h, 2*h, 0.8),
                                ops.GEMM(f'context_ffn1_gemm', self._num_layers, self._inter_size//tp_size, h, gemm_quant_mode),
                                ops.ElementWise(f'context_act', self._num_layers, self._inter_size//tp_size, self._inter_size//tp_size, 0.8),
                                ops.GEMM(f'context_ffn2_gemm', self._num_layers, h, self._inter_size//tp_size, gemm_quant_mode),
                                ops.GEMM(f'context_logits_gemm', 1, self._vocab_size//tp_size, h, common.GEMMQuantMode.float16)])

        self.generation_ops.extend([ops.Embedding(f'generation_embedding', 1, self._vocab_size, h, 0.3),
                                ops.ElementWise(f'generation_add_norm_1', self._num_layers, 2*h, 2*h, 0.8),
                                ops.GEMM(f'generation_qkv_gemm', self._num_layers, self._num_heads*self._head_size//tp_size+self._head_size*num_kv_heads_per_GPU*2, h, gemm_quant_mode), 
                                ops.GenerationAttention(f'generation_attention', self._num_layers, self._num_heads//tp_size, num_kv_heads_per_GPU, kvcache_quant_mode),
                                ops.GEMM(f'generation_proj_gemm', self._num_layers, h, self._num_heads*self._head_size//tp_size, gemm_quant_mode),
                                ops.ElementWise(f'generation_add_norm_2', self._num_layers, 2*h, 2*h, 0.8),
                                ops.GEMM(f'generation_ffn1_gemm', self._num_layers, self._inter_size//tp_size, h, gemm_quant_mode),
                                ops.ElementWise(f'generation_act', self._num_layers, self._inter_size//tp_size, self._inter_size//tp_size, 0.8),
                                ops.GEMM(f'generation_ffn2_gemm', self._num_layers, h, self._inter_size//tp_size, gemm_quant_mode),
                                ops.GEMM(f'generation_logits_gemm', 1, self._vocab_size//tp_size, h, common.GEMMQuantMode.float16)])
        
        # when tp_size=0, the comm part will be 0
        self.context_ops.append(ops.AllReduce('context_ar_1', self._num_layers, h, tp_size))
        self.context_ops.append(ops.AllReduce('context_ar_2', self._num_layers, h, tp_size))
        self.generation_ops.append(ops.AllReduce('generation_ar_1', self._num_layers, h, tp_size))
        self.generation_ops.append(ops.AllReduce('generation_ar_2', self._num_layers, h, tp_size))

        # pp
        pp_scale_factor = pp_size-1
        self.context_ops.append(ops.P2P('context_p2p', pp_scale_factor, h, pp_size))
        self.generation_ops.append(ops.P2P('generation_p2p', pp_scale_factor, h, pp_size))

class LLAMAModel(BaseModel):
    """
    LLAMA series uses this model impl. Other variants without large difference can use this as well, e.g., only positional embedding or activation is different.
    Some rules to follow,
    Due to implementation, attn layer name needs to be context_attention or generation_attention, exact match is required. Same for logits_gemm.
    Other than DS V3, all other models don't support mtp
    """
    def __init__(self, *args) -> None:
        super().__init__(*args)
        assert self._nextn == 0, 'Only DS V3 supports mtp'

        h = self._hidden_size
        tp_size = self.config.tp_size
        pp_size = self.config.pp_size
        num_kv_heads_per_GPU = self._num_kv_heads_per_GPU
        gemm_quant_mode = self.config.gemm_quant_mode
        kvcache_quant_mode = self.config.kvcache_quant_mode
        fmha_quant_mode = self.config.fmha_quant_mode


        self.context_ops.extend([ops.Embedding(f'context_embedding', 1, self._vocab_size, h, 0.3),
                                ops.ElementWise(f'context_add_norm_1', self._num_layers, 2*h, 2*h, 0.8),
                                ops.GEMM(f'context_qkv_gemm', self._num_layers, self._num_heads*self._head_size//tp_size+self._head_size*num_kv_heads_per_GPU*2, h, gemm_quant_mode), 
                                ops.ContextAttention(f'context_attention', self._num_layers, self._num_heads//tp_size, num_kv_heads_per_GPU, kvcache_quant_mode, fmha_quant_mode),
                                ops.GEMM(f'context_proj_gemm', self._num_layers, h, self._num_heads*self._head_size//tp_size, gemm_quant_mode),
                                ops.ElementWise(f'context_add_norm_2', self._num_layers, 2*h, 2*h, 0.8),
                                ops.GEMM(f'context_gate_ffn1_gemm', self._num_layers, 2*self._inter_size//tp_size, h, gemm_quant_mode),
                                ops.ElementWise(f'context_act_gate', self._num_layers, 2*self._inter_size//tp_size, self._inter_size//tp_size, 0.8),
                                ops.GEMM(f'context_ffn2_gemm', self._num_layers, h, self._inter_size//tp_size, gemm_quant_mode),
                                ops.GEMM(f'context_logits_gemm', 1, self._vocab_size//tp_size, h, common.GEMMQuantMode.float16)])

        self.generation_ops.extend([ops.Embedding(f'generation_embedding', 1, self._vocab_size, h, 0.3),
                                ops.ElementWise(f'generation_add_nrom_1', self._num_layers, 2*h, 2*h, 0.8),
                                ops.GEMM(f'generation_qkv_gemm', self._num_layers, self._num_heads*self._head_size//tp_size+self._head_size*num_kv_heads_per_GPU*2, h, gemm_quant_mode), 
                                ops.GenerationAttention(f'generation_attention', self._num_layers, self._num_heads//tp_size, num_kv_heads_per_GPU, kvcache_quant_mode),
                                ops.GEMM(f'generation_proj_gemm', self._num_layers, h, self._num_heads*self._head_size//tp_size, gemm_quant_mode),
                                ops.ElementWise(f'generation_add_norm_2', self._num_layers, 2*h, 2*h, 0.8),
                                ops.GEMM(f'generation_gate_ffn1_gemm', self._num_layers, 2*self._inter_size//tp_size, h, gemm_quant_mode),
                                ops.ElementWise(f'generation_act_gate', self._num_layers, 2*self._inter_size//tp_size, self._inter_size//tp_size, 0.8),
                                ops.GEMM(f'generation_ffn2_gemm', self._num_layers, h, self._inter_size//tp_size, gemm_quant_mode),
                                ops.GEMM(f'generation_logits_gemm', 1, self._vocab_size//tp_size, h, common.GEMMQuantMode.float16)])
        
        # when tp_message_size=0, the comm part will be 0
        self.context_ops.append(ops.AllReduce('context_ar_1', self._num_layers, h, tp_size))
        self.context_ops.append(ops.AllReduce('context_ar_2', self._num_layers, h, tp_size))
        self.generation_ops.append(ops.AllReduce('generation_ar_1', self._num_layers, h, tp_size))
        self.generation_ops.append(ops.AllReduce('generation_ar_2', self._num_layers, h, tp_size))

        # pp
        pp_scale_factor = pp_size-1
        self.context_ops.append(ops.P2P('context_p2p', pp_scale_factor, h, pp_size))
        self.generation_ops.append(ops.P2P('generation_p2p', pp_scale_factor, h, pp_size))

# mostly for mixtral models
class MOEModel(BaseModel):
    """
    Traditional MoE models uses this model impl: Mixtral, LLAMA4_MOE, etc.
    Some rules to follow,
    Due to implementation, attn layer name needs to be context_attention or generation_attention, exact match is required. Same for logits_gemm.
    Other than DS V3, all other models don't support mtp
    TODO: redesign shared moe part.
    """
    def __init__(self, topk: int, num_experts: int, moe_inter_size: int, *args) -> None:
        super().__init__(*args)
        assert self._nextn == 0, 'Only DS V3 supports mtp'

        # make sure the paralel width is same
        assert(self.config.tp_size * self.config.attention_dp_size == self.config.moe_tp_size * self.config.moe_ep_size), \
            f"tp_size ({self.config.tp_size}) * attention_dp_size ({self.config.attention_dp_size}) should be equal to moe_tp_size ({self.config.moe_tp_size}) * moe_ep_size ({self.config.moe_ep_size})"
        
        assert(num_experts >= self.config.moe_ep_size), f"ep size cannot be larger than num_experts {num_experts}"
        assert(self.config.tp_size * self.config.attention_dp_size <= 256), f"moe ep size {self.config.moe_ep_size} * moe tp size {self.config.moe_tp_size} should not be larger than 256"

        self._topk = topk

        self._topk = topk
        self._num_experts = num_experts
        self._moe_inter_size = moe_inter_size

        moe_quant_mode = self.config.moe_quant_mode

        h = self._hidden_size
        tp_size = self.config.tp_size
        moe_tp_size = self.config.moe_tp_size
        moe_ep_size = self.config.moe_ep_size
        attention_dp_size = self.config.attention_dp_size
        pp_size = self.config.pp_size
        num_kv_heads_per_GPU = self._num_kv_heads_per_GPU
        gemm_quant_mode = self.config.gemm_quant_mode
        kvcache_quant_mode = self.config.kvcache_quant_mode
        fmha_quant_mode = self.config.fmha_quant_mode
        workload_distribution = self.config.workload_distribution

        self.context_ops.extend([ops.Embedding(f'context_embedding', 1, self._vocab_size, h, 0.3),
                                ops.ElementWise(f'context_add_norm_1', self._num_layers, 2*h, 2*h, 0.8),
                                ops.GEMM(f'context_qkv_gemm', self._num_layers, self._num_heads*self._head_size//tp_size+self._head_size*num_kv_heads_per_GPU*2, h, gemm_quant_mode),
                                ops.ContextAttention(f'context_attention', self._num_layers, self._num_heads//tp_size, num_kv_heads_per_GPU, kvcache_quant_mode, fmha_quant_mode),
                                ops.GEMM(f'context_proj_gemm', self._num_layers, h, self._num_heads*self._head_size//tp_size, gemm_quant_mode),
                                ops.ElementWise(f'context_add_norm_2', self._num_layers, 2*h, 2*h, 0.8)])

        #router, only take it into account when num_experts >= 128
        if self._num_experts >= 128:
            self.context_ops.extend([
                            ops.GEMM(f'context_router_gemm', self._num_layers, self._num_experts, h, common.GEMMQuantMode.float16)
                            ])

        # dispatch tokens to experts, moe calc and get tokens back
        self.context_ops.extend([
                                ops.MoEDispatch(f'context_moe_pre_dispatch', self._num_layers, h, self._topk, self._num_experts, moe_tp_size, moe_ep_size, attention_dp_size, True),
                                ops.MoE(f'context_moe', self._num_layers, h, self._moe_inter_size, self._topk, self._num_experts, moe_tp_size, moe_ep_size, moe_quant_mode, workload_distribution, attention_dp_size),
                                ops.MoEDispatch(f'context_moe_post_dispatch', self._num_layers, h, self._topk, self._num_experts, moe_tp_size, moe_ep_size, attention_dp_size, False)])
        
        self.context_ops.extend([ops.GEMM(f'context_logits_gemm', 1, self._vocab_size//tp_size, h, common.GEMMQuantMode.float16)])

        self.generation_ops.extend([ops.Embedding(f'generation_embedding', 1, self._vocab_size, h, 0.3),
                                ops.ElementWise(f'generation_add_norm_1', self._num_layers, 2*h, 2*h, 0.8),
                                ops.GEMM(f'generation_qkv_gemm', self._num_layers, self._num_heads*self._head_size//tp_size+self._head_size*num_kv_heads_per_GPU*2, h, gemm_quant_mode),
                                ops.GenerationAttention(f'generation_attention', self._num_layers, self._num_heads//tp_size, num_kv_heads_per_GPU, kvcache_quant_mode),
                                ops.GEMM(f'generation_proj_gemm', self._num_layers, h, self._num_heads*self._head_size//tp_size, gemm_quant_mode),
                                ops.ElementWise(f'generation_add_norm_2', self._num_layers, 2*h, 2*h, 0.8)])

        #router, only take it into account when num_experts >= 128
        if self._num_experts >= 128:
            self.generation_ops.extend([
                            ops.GEMM(f'generation_router_gemm', self._num_layers, self._num_experts, h, common.GEMMQuantMode.float16)
                            ])

        # dispatch tokens to experts, moe calc and get tokens back
        self.generation_ops.extend([
                                ops.MoEDispatch(f'generation_moe_pre_dispatch', self._num_layers, h, self._topk, self._num_experts, moe_tp_size, moe_ep_size, attention_dp_size, True),
                                ops.MoE(f'generation_moe', self._num_layers, h, self._moe_inter_size, self._topk, self._num_experts, moe_tp_size, moe_ep_size, moe_quant_mode, workload_distribution, attention_dp_size),
                                ops.MoEDispatch(f'generation_moe_post_dispatch', self._num_layers, h, self._topk, self._num_experts, moe_tp_size, moe_ep_size, attention_dp_size, False)
                                ])
        # logits gemm
        self.generation_ops.extend([ops.GEMM(f'generation_logits_gemm', 1, self._vocab_size//tp_size, h, common.GEMMQuantMode.float16)])

        # # # when tp_size=0, the comm part will be 0
        # self.context_ops.append(ops.AllReduce('context_ar_1', self._num_layers, h, tp_size))
        # self.context_ops.append(ops.AllReduce('context_ar_2', self._num_layers, h, tp_size))
        # self.generation_ops.append(ops.AllReduce('generation_ar_1', self._num_layers, h, tp_size))
        # self.generation_ops.append(ops.AllReduce('generation_ar_2', self._num_layers, h, tp_size))

        # pp
        pp_scale_factor = pp_size-1
        self.context_ops.append(ops.P2P('context_p2p', pp_scale_factor, h, pp_size))
        self.generation_ops.append(ops.P2P('generation_p2p', pp_scale_factor, h, pp_size))


class DeepSeekModel(BaseModel):
    """
    DeepSeek V3/R1 uses this model impl.
    """
    def __init__(self, topk: int, num_experts: int, moe_inter_size: int, *args) -> None:
        super().__init__(*args)

        # make sure the paralel width is same
        assert(self.config.tp_size * self.config.attention_dp_size == self.config.moe_tp_size * self.config.moe_ep_size), \
            f"tp_size ({self.config.tp_size}) * attention_dp_size ({self.config.attention_dp_size}) should be equal to moe_tp_size ({self.config.moe_tp_size}) * moe_ep_size ({self.config.moe_ep_size})"
        
        assert(num_experts >= self.config.moe_ep_size), f"ep size cannot be larger than num_experts {num_experts}"
        assert(self.config.tp_size * self.config.attention_dp_size <= 256), f"moe ep size {self.config.moe_ep_size} * moe tp size {self.config.moe_tp_size} should not be larger than 256"

        self._topk = topk
        self._num_experts = num_experts
        self._moe_inter_size = moe_inter_size

         # used to scale the tpot to reflect mtp effect: 
         # 1. mtp will reduce the overall time by expected_tokens_per_step 
         # 2. mtp module introduces nextn new transformer layers+linear layers (we ignore the linear layers for now)
         # 3. special correction in ifb step due to we leveraging ctx phase for gen tokens non-attn part
         # meanwhile, needs to scale the actual bs of generation by nextn, this is covered in inferencesession
        self._mtp_scale_factor = 1./(1+calc_expectation(self._nextn, self._nextn_accept_rates))*(self._nextn+self._num_layers)/self._num_layers

        gemm_quant_mode = self.config.gemm_quant_mode
        moe_quant_mode = self.config.moe_quant_mode

        mla_bmm_quant_mode = common.GEMMQuantMode.fp8 if gemm_quant_mode != common.GEMMQuantMode.float16 else common.GEMMQuantMode.float16

        h = self._hidden_size # 7168
        tp_size = self.config.tp_size
        moe_tp_size = self.config.moe_tp_size
        moe_ep_size = self.config.moe_ep_size
        attention_dp_size = self.config.attention_dp_size
        pp_size = self.config.pp_size
        num_kv_heads_per_GPU = self._num_kv_heads_per_GPU

        kvcache_quant_mode = self.config.kvcache_quant_mode
        fmha_quant_mode = self.config.fmha_quant_mode
        workload_distribution = self.config.workload_distribution

        self.context_ops.extend([ops.Embedding(f'context_embedding', 1, self._vocab_size, h, 0.3),
                                ops.ElementWise(f'context_add_norm_1', self._num_layers, 2*h, 2*h, 0.8),
                                ops.GEMM(f'context_downscale_gemm', self._num_layers, 2112, h, gemm_quant_mode), # on every gpu, fused_a
                                ops.GEMM(f'context_q_b_proj_gemm', self._num_layers, 24576//tp_size, 1536, gemm_quant_mode),
                                ops.GEMM(f'context_kv_b_proj_gemm', self._num_layers, 32768//tp_size, 512, gemm_quant_mode), # ifb ctx attn part
                                ops.ContextMLA(f'context_attention', self._num_layers, tp_size, kvcache_quant_mode, fmha_quant_mode), # ifb ctx attn part
                                ops.GEMM(f'context_proj_gemm', self._num_layers, h, 128*128//tp_size, gemm_quant_mode), # ifb ctx attn part
                                ops.ElementWise(f'context_add_norm_2', self._num_layers, 2*h, 2*h, 0.8)])

        # shared moe
        self.context_ops.extend([
                                ops.GEMM(f'context_shared_gate_gemm', self._num_layers, self._moe_inter_size//tp_size, h, gemm_quant_mode),
                                ops.GEMM(f'context_shared_ffn1_gemm', self._num_layers, self._moe_inter_size//tp_size, h, gemm_quant_mode),
                                ops.ElementWise(f'context_shared_act_gate', self._num_layers, 2*self._moe_inter_size//tp_size, self._moe_inter_size//tp_size, 0.8),
                                ops.GEMM(f'context_shared_ffn2_gemm', self._num_layers, h, self._moe_inter_size//tp_size, gemm_quant_mode)
                                ])
        
        # router gemm, num_experts is large enough, cannot be ignored anymore.
        self.context_ops.extend([
                                ops.GEMM(f'context_router_gemm', self._num_layers, self._num_experts, h, common.GEMMQuantMode.float16)
                                ])

        # dispatch tokens to experts, pre-dispatch
        self.context_ops.extend([
                                ops.MoEDispatch(f'context_moe_pre_dispatch', self._num_layers, h, self._topk, self._num_experts, moe_tp_size, moe_ep_size, attention_dp_size, True)
                                ])
        
        # moe part
        self.context_ops.extend([ops.MoE(f'context_moe', self._num_layers, h, self._moe_inter_size, self._topk, self._num_experts, moe_tp_size, moe_ep_size, moe_quant_mode, workload_distribution, attention_dp_size)
                                ])

        # dispatch tokens to experts, post-dispatch
        self.context_ops.extend([
                                ops.MoEDispatch(f'context_moe_post_dispatch', self._num_layers, h, self._topk, self._num_experts, moe_tp_size, moe_ep_size, attention_dp_size, False)
                                ])

        self.context_ops.extend([ops.GEMM(f'context_logits_gemm', 1, self._vocab_size//tp_size, h, common.GEMMQuantMode.float16)])
        #####generation part, only generation part is scaled by mtp_scale_factor
        self.generation_ops.extend([ops.Embedding(f'generation_embedding', 1*self._mtp_scale_factor, self._vocab_size, h, 0.3),
                                ops.ElementWise(f'generation_add_norm_1', self._num_layers*self._mtp_scale_factor, 2*h, 2*h, 0.8),
                                ops.GEMM(f'generation_downscale_gemm', self._num_layers*self._mtp_scale_factor, 2112, h, gemm_quant_mode), # on every gpu
                                ops.GEMM(f'generation_q_b_proj_gemm', self._num_layers*self._mtp_scale_factor, 24576//tp_size, 1536, gemm_quant_mode),
                                ops.MLABmm(f'generation_bmm_pre', self._num_layers*self._mtp_scale_factor, self._num_heads//tp_size, mla_bmm_quant_mode, if_pre=True), # ifb gen attn part
                                ops.GenerationMLA(f'generation_attention', self._num_layers*self._mtp_scale_factor, tp_size, kvcache_quant_mode), # ifb gen attn part
                                ops.MLABmm(f'generation_bmm_post', self._num_layers*self._mtp_scale_factor, self._num_heads//tp_size, mla_bmm_quant_mode, if_pre=False), # ifb gen attn part
                                ops.GEMM(f'generation_proj_gemm', self._num_layers*self._mtp_scale_factor, h, h//tp_size, gemm_quant_mode),
                                ops.ElementWise(f'generation_add_norm_2', self._num_layers*self._mtp_scale_factor, 2*h, 2*h, 0.8)])

        # shared moe
        self.generation_ops.extend([
                                ops.GEMM(f'generation_shared_gate_gemm', self._num_layers*self._mtp_scale_factor, self._moe_inter_size//tp_size, h, gemm_quant_mode),
                                ops.GEMM(f'generation_shared_ffn1_gemm', self._num_layers*self._mtp_scale_factor, self._moe_inter_size//tp_size, h, gemm_quant_mode),
                                ops.ElementWise(f'generation_shared_act_gate', self._num_layers*self._mtp_scale_factor, 2*self._moe_inter_size//tp_size, self._moe_inter_size//tp_size, 0.8),
                                ops.GEMM(f'generation_shared_ffn2_gemm', self._num_layers*self._mtp_scale_factor, h, self._moe_inter_size//tp_size, gemm_quant_mode)
                                ])     
        
        # router gemm, num_experts is large enough, cannot be ignored anymore.
        self.generation_ops.extend([
                                ops.GEMM(f'generation_router_gemm', self._num_layers*self._mtp_scale_factor, self._num_experts, h, common.GEMMQuantMode.float16)
                                ])

        # dispatch tokens to experts, pre-dispatch
        self.generation_ops.extend([
                                ops.MoEDispatch(f'generation_moe_pre_dispatch', self._num_layers*self._mtp_scale_factor, h, self._topk, self._num_experts, moe_tp_size, moe_ep_size, attention_dp_size, True)
                                ])
   
        # moe part
        self.generation_ops.extend([ops.MoE(f'generation_moe', self._num_layers*self._mtp_scale_factor, h, self._moe_inter_size, self._topk, self._num_experts, moe_tp_size, moe_ep_size, moe_quant_mode, workload_distribution, attention_dp_size),
                                ])

        # dispatch tokens to experts, post-dispatch
        self.generation_ops.extend([
                                ops.MoEDispatch(f'generation_moe_post_dispatch', self._num_layers*self._mtp_scale_factor, h, self._topk, self._num_experts, moe_tp_size, moe_ep_size, attention_dp_size, False)
                                ])

        self.generation_ops.extend([ops.GEMM(f'generation_logits_gemm', 1*self._mtp_scale_factor, self._vocab_size//tp_size, h, common.GEMMQuantMode.float16)])

        # when tp_size=0, the comm part will be 0
        # self.context_ops.append(ops.AllReduce('context_ar_1', self._num_layers, h, tp_size))
        # self.context_ops.append(ops.AllReduce('context_ar_2', self._num_layers, h, tp_size))
        # self.generation_ops.append(ops.AllReduce('generation_ar_1', self._num_layers*self._mtp_scale_factor, h, tp_size))
        # self.generation_ops.append(ops.AllReduce('generation_ar_2', self._num_layers*self._mtp_scale_factor, h, tp_size))

        # pp
        pp_scale_factor = pp_size-1
        self.context_ops.append(ops.P2P('context_p2p', pp_scale_factor*self._mtp_scale_factor, h, pp_size))
        self.generation_ops.append(ops.P2P('generation_p2p', pp_scale_factor*self._mtp_scale_factor, h, pp_size))

        # TODO
        # a lot of quantization ops

class NemotronNas(BaseModel):
    """
    NemotronNas model implementation with configurable block architectures.
    
    This model supports flexible transformer architectures where each block can have
    different configurations for attention and feed-forward network components.
    The model does not support multi-token prediction (mtp).

    refer to "PUZZLE: DISTILLATION-BASED NAS FOR INFERENCE-OPTIMIZED LLMS"(
    https://arxiv.org/pdf/2411.19146) for the details of creaing this type of 
    models
    """

    def __init__(self, *args):
        """
        Initialize NemotronNas model with configurable transformer blocks.
        
        Args:
            *args: Arguments passed to BaseModel constructor including:
                - model_name (str): Name of the model
                - model_family (str): Model family (should be "NEMOTRONNAS")
                - num_layers (int): Number of transformer layers
                - num_heads (int): Number of attention heads
                - num_kv_heads (int): Number of key-value heads (0 for this model, will set using block_configs)
                - head_size (int): Size of each attention head
                - hidden_size (int): Hidden dimension size
                - inter_size (int): Intermediate size (0 for this model, will set using block_configs)
                - vocab_size (int): Vocabulary size
                - context_length (int): Maximum context length
                - model_config (ModelConfig): Model configuration object    
        Raises:
            AssertionError: If model configuration specifies mtp (nextn != 0), as only DS V3 supports mtp
        """
        super().__init__(*args)

        assert self._nextn == 0, 'Only DS V3 supports mtp'

    @property
    def context_ops(self):
        """
        Get the context(prefill) processing operations pipeline.
        
        Returns:
            List[ops.Operation]: List of operations for processing context 
            sequences, including: 
                - embedding, 
                - attention blocks, 
                - FFN blocks, 
                - P2P communication,
                - all reduce communication 
                - logits computation.
        """
        return self._context_ops

    @context_ops.setter
    def context_ops(self, puzzle_block_configs: List[common.BlockConfig]):
        """
        Set the context(prefill) processing operations pipeline based on block configurations.
        
        Constructs a pipeline of operations for processing input context by creating operations
        for each configured transformer block. The pipeline includes embedding lookup, 
        transformer blocks (with optional attention and FFN components), pipeline parallel
        communication, and final logits computation.
        
        Args:
            puzzle_block_configs (List[BlockConfig]): List of block configurations where each
                BlockConfig specifies:
                - num_inst (int): Number of instances of this block type
                - attn_no_op (bool): Whether to skip attention operations for this block
                - attn_n_heads_in_group (int): Number of attention heads in group (used if attn_no_op is False)
                - ffn_no_op (bool): Whether to skip FFN operations for this block
                - ffn_ffn_mult (float): FFN size multiplier relative to hidden size (used if ffn_no_op is False)

        """
        self._context_ops = []
        if puzzle_block_configs:
            h = self._hidden_size
            tp_size = self.config.tp_size
            pp_size = self.config.pp_size
            gemm_quant_mode = self.config.gemm_quant_mode
            kvcache_quant_mode = self.config.kvcache_quant_mode
            fmha_quant_mode = self.config.fmha_quant_mode
            pp_scale_factor = pp_size-1
            self._context_ops.append(ops.Embedding(f'context_embedding', 1, self._vocab_size, h, 0.3))
            for b in puzzle_block_configs:
                count = b.num_inst   
                if not b.attn_no_op:
                    num_kv_heads = self._num_heads // b.attn_n_heads_in_group
                    num_kv_heads_per_GPU = (num_kv_heads + tp_size - 1) // tp_size
                    self._context_ops.extend([ops.ElementWise(f'context_add_norm_1', count, 2*h, 2*h, 0.8),
                                            ops.GEMM(f'context_qkv_gemm', count, self._num_heads*self._head_size//tp_size+self._head_size*num_kv_heads_per_GPU*2, h, gemm_quant_mode), 
                                            ops.ContextAttention(f'context_attention', count, self._num_heads//tp_size, num_kv_heads_per_GPU, kvcache_quant_mode, fmha_quant_mode),
                                            ops.GEMM(f'context_proj_gemm', count, h, self._num_heads*self._head_size//tp_size, gemm_quant_mode),
                                            ops.AllReduce('context_ar_1', count, h, tp_size)])
                if not b.ffn_no_op:
                    inter_size = self._ffn_mult_to_intermediate_size(b.ffn_ffn_mult)
                    self._context_ops.extend([ops.ElementWise(f'context_add_norm_2', count, 2*h, 2*h, 0.8),
                                            ops.GEMM(f'context_gate_ffn1_gemm', count, 2*inter_size//tp_size, h, gemm_quant_mode),
                                            ops.ElementWise(f'context_act_gate', count, 2*inter_size//tp_size, inter_size//tp_size, 0.8),
                                            ops.GEMM(f'context_ffn2_gemm', count, h, inter_size//tp_size, gemm_quant_mode),
                                            ops.AllReduce('context_ar_2', count, h, tp_size)])
            self._context_ops.append(ops.P2P('context_p2p', pp_scale_factor, h, pp_size))
            self._context_ops.append(ops.GEMM(f'context_logits_gemm', 1, self._vocab_size//tp_size, h, common.GEMMQuantMode.float16))

    @property
    def generation_ops(self):
        """
        Get the generation (decoding) operations pipeline.
        
        Returns:
            List[ops.Operation]: List of operations for the decoding phase 
            including: 
                - embedding, 
                - attention blocks, 
                - FFN blocks, 
                - P2P communication,
                - all reduce communication 
                - logits computation.
        """
        return self._generation_ops 

    @generation_ops.setter
    def generation_ops(self, puzzle_block_configs: List[common.BlockConfig]):
        """
        Set the generation (decoding) operations pipeline based on block configurations.
        
        Constructs a pipeline of operations for autoregressive generation by creating operations
        for each configured transformer block. Similar to context_ops but uses generation-specific
        attention operations that support KV-cache for efficient autoregressive decoding.
        
        Args:
            puzzle_block_configs (List[BlockConfig]): List of block configurations where each
                BlockConfig specifies:
                - num_inst (int): Number of instances of this block type
                - attn_no_op (bool): Whether to skip attention operations for this block
                - attn_n_heads_in_group (int): Number of attention heads in group (used if attn_no_op is False)
                - ffn_no_op (bool): Whether to skip FFN operations for this block
                - ffn_ffn_mult (float): FFN size multiplier relative to hidden size (used if ffn_no_op is False)
        """
        self._generation_ops = []
        if puzzle_block_configs:
            h = self._hidden_size
            tp_size = self.config.tp_size
            pp_size = self.config.pp_size
            gemm_quant_mode = self.config.gemm_quant_mode
            kvcache_quant_mode = self.config.kvcache_quant_mode
            fmha_quant_mode = self.config.fmha_quant_mode
            pp_scale_factor = pp_size-1
            self._generation_ops.append(ops.Embedding(f'generation_embedding', 1, self._vocab_size, h, 0.3))
            for b in puzzle_block_configs:
                count = b.num_inst 
                if not b.attn_no_op:
                    num_kv_heads = self._num_heads // b.attn_n_heads_in_group
                    num_kv_heads_per_GPU = (num_kv_heads + tp_size - 1) // tp_size
                    self._generation_ops.extend([ops.ElementWise(f'generation_add_nrom_1', count, 2*h, 2*h, 0.8),
                                                ops.GEMM(f'generation_qkv_gemm', count, self._num_heads*self._head_size//tp_size+self._head_size*num_kv_heads_per_GPU*2, h, gemm_quant_mode), 
                                                ops.GenerationAttention(f'generation_attention', count, self._num_heads//tp_size, num_kv_heads_per_GPU, kvcache_quant_mode),
                                                ops.GEMM(f'generation_proj_gemm', count, h, self._num_heads*self._head_size//tp_size, gemm_quant_mode),
                                                ops.AllReduce('generation_ar_1', count, h, tp_size)])
                if not b.ffn_no_op:
                    inter_size = self._ffn_mult_to_intermediate_size(b.ffn_ffn_mult)
                    self._generation_ops.extend([ops.ElementWise(f'generation_add_norm_2', count, 2*h, 2*h, 0.8),
                                                ops.GEMM(f'generation_gate_ffn1_gemm', count, 2*inter_size//tp_size, h, gemm_quant_mode),
                                                ops.ElementWise(f'generation_act_gate', count, 2*inter_size//tp_size, inter_size//tp_size, 0.8),
                                                ops.GEMM(f'generation_ffn2_gemm', count, h, inter_size//tp_size, gemm_quant_mode),
                                                ops.AllReduce('generation_ar_2', count, h, tp_size)])
            self._generation_ops.append(ops.P2P('generation_p2p', pp_scale_factor, h, pp_size))
            self._generation_ops.append(ops.GEMM(f'generation_logits_gemm', 1, self._vocab_size//tp_size, h, common.GEMMQuantMode.float16))
    
    
    def _ffn_mult_to_intermediate_size(self, ffn_mult: float) -> int:
        """
        Rule used to convert ffn_mult into the intermediate size of the ffn GEMM 
        
        Args:
            ffn_mult (float): FFN size multiplier relative to hidden size
        """
        # conversion codes adopted from 
        # https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1/blob/main/modeling_decilm.py
        inter_size = int(2 * ffn_mult * self._hidden_size / 3)
        if inter_size % 256 == 0:
            return inter_size
        return inter_size + 256 - (inter_size % 256)
       
if __name__ == '__main__':
    # TODO, move to unit tests
    model = get_model('DEEPSEEK_V3', config.ModelConfig(
        tp_size=1,
        pp_size=1,
        moe_tp_size=1,
        moe_ep_size=1,
        attention_dp_size=1,
        gemm_quant_mode=common.GEMMQuantMode.fp8_ootb,
        kvcache_quant_mode=common.KVCacheQuantMode.fp8,
        fmha_quant_mode=common.FMHAQuantMode.fp8,
        moe_quant_mode=common.MoEQuantMode.w4afp8,
        nextn=2,
        nextn_accept_rates=[0.5, 0.5],
    ))
    print(model.context_ops)
    print(model.generation_ops)
    print(model.config)
