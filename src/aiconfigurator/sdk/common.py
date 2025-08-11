# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from collections import namedtuple
from dataclasses import dataclass

@dataclass(frozen=True)
class BlockConfig:
    """
    Configuration for a single transformer block in NemotronNas.
    
    Attributes:
        attn_n_heads_in_group (int): Number of attention heads in the group for this block
        attn_no_op (bool): If True, skip attention operations for this block
        ffn_ffn_mult (float): Multiplier for FFN intermediate size relative to hidden size
        ffn_no_op (bool): If True, skip FFN operations for this block
        num_inst (int): number of ocurrances of the given block
    """
    attn_n_heads_in_group: int = 8
    attn_no_op: bool = False
    ffn_ffn_mult: float = 3.5
    ffn_no_op: bool = False
    num_inst: int = 0

"""
Supported models
    model name: model_family,l,n,n_kv,d,hidden_size,inter_size,vocab,context,topk,num_experts,moe_inter_size,extra_params
"""
SupportedModels = {
                'GPT_7B':['GPT',32,32,32,128,32*128,32*128*4,50527,2048, 0, 0, 0, None],
                'GPT_13B':['GPT',40,40,40,128,40*128,40*128*4,50527,2048, 0, 0, 0, None],
                'GPT_30B':['GPT',48,56,56,128,56*128,56*128*4,50527,2048, 0, 0, 0, None],
                'GPT_66B':['GPT',64,72,72,128,72*128,72*128*4,50527,2048, 0, 0, 0, None],
                'GPT_175B':['GPT',96,96,96,128,96*128,96*128*4,50527,2048, 0, 0, 0, None],
                'LLAMA2_7B':['LLAMA',32,32,32,128,32*128,11008,32000,2048, 0, 0, 0, None],
                'LLAMA2_13B':['LLAMA',40,40,40,128,40*128,13824,32000,4096, 0, 0, 0, None],
                'LLAMA2_70B':['LLAMA',80,64,8,128,64*128,28672,32000,4096, 0, 0, 0, None],
                'LLAMA3.1_8B':['LLAMA',32,32,8,128,32*128,14336,128256,131072, 0, 0, 0, None],
                'LLAMA3.1_70B':['LLAMA',80,64,8,128,64*128,28672,128256,131072, 0, 0, 0, None],
                'LLAMA3.1_405B':['LLAMA',126,128,8,128,128*128,53248,128256,131072, 0, 0, 0, None],
                'MOE_Mixtral8x7B':['MOE', 32,32,8,128,32*128,14336,32000,32768,2,8,14336, None],
                'MOE_Mixtral8x22B':['MOE', 56,48,8,128,48*128,16384,32000,65536,2,8,16384, None],
                #'MOE_GPT_1.8T':['MOE', 120,120,1,128,30720,50247,4096,2,16,0, None],
                #'MOE_GPT_1.8T_FineGrained':['MOE', 120,120,1,128,3840,50247,4096,16,128,0, None],
                #'MOE_Deepseek_16B_Base':['MOE', 28,16,16,128,2816,102400,4096,6,64,1408, None],
                #'MOE_Deepseek_V2':['MOE', 60,128,128,40,3072,102400,163840,6,160,1536, None],#using MLA, not standard attention
                'DEEPSEEK_V3':['DEEPSEEK', 61,128,128,56,128*56,18432,129280,4096,8,256,2048, None],#using MLA, not standard attention, 3 of 61 are dense layers using intersize 18432, others using 2048
                'KIMI_K2':['DEEPSEEK', 61,128,128,56,128*56,18432,163840,131072, 8, 384, 2048, None],                
                #'MOE_Qwen1.5_A2.7B':['MOE', 24,16,16,128,5632,151936,32768,4,60,1408, None],
                #'MOE_Qwen2_57B_A14B':['MOE', 28,28,4,128,20480,151936,32768,8,64,2560, None]
                'QWEN2.5_1.5B':['LLAMA', 28,12,2,128,12*128,8960,151936,131072, 0, 0, 0, None],
                'QWEN2.5_7B':['LLAMA', 28,28,4,128,28*128,18944,152064,131072, 0, 0, 0, None],
                'QWEN2.5_32B':['LLAMA', 64,40,8,128,40*128,27648,152064,32768, 0, 0, 0, None],
                'QWEN2.5_72B':['LLAMA', 80,64,8,128,64*128,29568,152064,32768, 0, 0, 0, None],
                'QWEN3_32B':['LLAMA', 64,64,8,128,5120,25600,151936,40960, 0, 0, 0, None], # qwen3 is not using hiddensize=headdim*numheads.
                'QWEN3_235B':['MOE', 94,64,4,128,4096,12288,151936,40960, 8, 128, 1536, None],
                'QWEN3_480B':['MOE', 62,96,8,128,6144,8192,151936,262144,8,160,2560, None],
                'Nemotron_super_v1.1':['NEMOTRONNAS', 80, 64, 0, 128, 8192, 0, 128256, 131072, 0, 0, 0, 
                                                    [
                                                        BlockConfig(8, False, 5.25, False, 48),
                                                        BlockConfig(None, True, 1.3125, False, 10), 
                                                        BlockConfig(None, True, 1.0, False, 8), 
                                                        BlockConfig(None, True, 0.5, False, 6),
                                                        BlockConfig(None, True, 2.625, False, 5),
                                                        BlockConfig(8, False, 2.625, False, 1), 
                                                        BlockConfig(None, True, 3.28125, False, 1),
                                                        BlockConfig(None, True, 5.25, False, 1)
                                                    ]
                                                    ]
              }

"""
Model family for model definition
"""
ModelFamily = {'GPT', 'LLAMA', 'MOE', 'DEEPSEEK', 'NEMOTRONNAS'}

"""
All reduce strategy for trtllm custom allreduce
"""
AllReduceStrategy = {'NCCL', 'ONESHOT', 'TWOSHOT', 'AUTO'}

"""
Columns for static inference summary dataframe
"""
ColumnsStatic = ['model', 'isl', 'osl', 
                 'concurrency', 'request_rate', 'bs', 'global_bs',
                 'ttft', 'tpot', 'seq/s', 'seq/s/gpu', 'tokens/s', 'tokens/s/gpu', 'tokens/s/user', 'latency', 'context_latency', 'generation_latency', 
                 'num_total_gpus',
                 'tp', 'pp', 'dp', 'moe_tp', 'moe_ep', 'parallel',
                 'gemm', 'kvcache', 'fmha', 'moe', 'comm',
                 'memory', 
                 'backend', 'version', 'system']

"""
Columns for IFB inference summary dataframe
"""
ColumnsIFB = ['model', 'isl', 'osl', 
              'concurrency', 'request_rate', 'bs', 'global_bs',
              'ttft', 'tpot', 'seq/s', 'seq/s/gpu', 'tokens/s', 'tokens/s/gpu', 'tokens/s/user', 
              'num_total_gpus',
              'tp', 'pp', 'dp', 'moe_tp', 'moe_ep', 'parallel',
              'gemm', 'kvcache', 'fmha', 'moe', 'comm',
              'memory', 
              'balance_score','num_ctx_reqs','num_gen_reqs','num_tokens','ctx_tokens','gen_tokens', # ifb specific
              'backend', 'version', 'system']             

"""
Columns for disaggregated inference summary dataframe
"""
ColumnsDisagg = ['model', 'isl', 'osl', 
                 'concurrency', 'request_rate', '(p)bs', '(p)global_bs', '(p)workers', '(d)bs', '(d)global_bs', '(d)workers',
                 'ttft', 'tpot', 'seq/s', 'seq/s/gpu', 'tokens/s', 'tokens/s/gpu', 'tokens/s/user', '(p)seq/s/worker', '(d)seq/s/worker',
                 'num_total_gpus',
                 '(p)tp', '(p)pp', '(p)dp', '(p)moe_tp', '(p)moe_ep', '(p)parallel', 
                 '(p)gemm', '(p)kvcache', '(p)fmha', '(p)moe', '(p)comm',
                 '(p)memory',
                 '(p)backend', '(p)version', '(p)system',
                 '(d)tp', '(d)pp', '(d)dp', '(d)moe_tp', '(d)moe_ep', '(d)parallel',
                 '(d)gemm', '(d)kvcache', '(d)fmha', '(d)moe', '(d)comm',
                 '(d)memory',
                 '(d)backend', '(d)version', '(d)system']

class SOLMode(Enum):
    """
    SOL mode for database.
    """
    NON_SOL = 0
    SOL = 1
    SOL_FULL = 2

class BackendName(Enum):
    """
    Backend name for inference.
    """
    trtllm = 'trtllm'
    sglang = 'sglang'
    vllm = 'vllm'

class PerfDataFilename(Enum):
    """
    Perf data filename for database to load.
    """
    gemm = 'gemm_perf.txt'
    nccl = 'nccl_perf.txt'
    generation_attention = 'generation_attention_perf.txt'
    context_attention = 'context_attention_perf.txt'
    context_mla = 'context_mla_perf.txt'
    generation_mla = 'generation_mla_perf.txt'
    mla_bmm = 'mla_bmm_perf.txt'
    moe = 'moe_perf.txt'
    custom_allreduce = 'custom_allreduce_perf.txt'

QuantMapping = namedtuple('QuantMapping', ['memory', 'compute', 'name'])

class GEMMQuantMode(Enum):
    """
    GEMM quant mode.
    """
    float16 = QuantMapping(2, 1, 'float16') # w16a16
    int8_wo = QuantMapping(1, 1, 'int8_wo') # w8a16
    int4_wo = QuantMapping(0.5, 1, 'int4_wo') # w4a16
    fp8 = QuantMapping(1, 2, 'fp8') #w8fp8
    sq = QuantMapping(1, 2, 'sq') # w8int8
    fp8_block = QuantMapping(1, 2, 'fp8_block') # specific for trtllm torch ds fp8
    fp8_ootb = QuantMapping(1, 2, 'fp8_ootb') # in future, should deprecate this mode as it's specific for trtllm trt backend
    nvfp4 = QuantMapping(0.5, 4, 'nvfp4') # nvfp4 on blackwell

class MoEQuantMode(Enum):
    """
    MoE quant mode.
    """
    float16 = QuantMapping(2, 1, 'float16') # w16a16
    fp8 = QuantMapping(1, 2, 'fp8') # w8fp8
    int4_wo = QuantMapping(0.5, 1, 'int4_wo') # w4a16
    fp8_block = QuantMapping(1, 2, 'fp8_block') # specific for trtllm torch ds fp8
    w4afp8 = QuantMapping(0.5, 2, 'w4afp8') # specific for trtllm torch ds w4a8
    nvfp4 = QuantMapping(0.5, 4, 'nvfp4') # nvfp4 on blackwell

class FMHAQuantMode(Enum):
    """
    FMHA quant mode.
    """
    float16 = QuantMapping(0, 1, 'float16')
    fp8 = QuantMapping(0, 2, 'fp8')

class KVCacheQuantMode(Enum):
    """
    KVCache quant mode.
    """
    float16 = QuantMapping(2, 0, 'float16')
    int8 = QuantMapping(1, 0, 'int8')
    fp8 = QuantMapping(1, 0, 'fp8')

class CommQuantMode(Enum):
    """
    Comm quant mode.
    """
    half = QuantMapping(2, 0, 'half')
    int8 = QuantMapping(1, 0, 'int8')
    fp8 = QuantMapping(1, 0, 'fp8')
