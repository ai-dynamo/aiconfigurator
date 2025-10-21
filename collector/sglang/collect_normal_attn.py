# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from helper import getSMVersion, log_perf
from collections import namedtuple
from functools import partial
import math
import os
from typing import NamedTuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from helper import log_perf
Timing = NamedTuple('timing', [('mean', float)])
from sgl_kernel.flash_attn import flash_attn_with_kvcache as flash_attn_func_v3

import math
import random

from triton.testing import do_bench

import torch
import triton
import pkg_resources


DISABLE_BACKWARD = os.getenv("FLASH_ATTENTION_DISABLE_BACKWARD", "FALSE") == "TRUE"

def get_context_attention_test_cases():
    test_cases = []
    b_list = [1,2,4,8,16,32,64,128,256]
    s_list = [16,32,64,128,256,512,1024,1536,2048,3072,4096,6144,8192,10240,12288,16384,262144]
    n_list = [4,8,12,16,24,32,40,48,64,96]
    n_kv_list = [0,1,2,4,8]
    for n in sorted(n_list, reverse=True):
        for s in sorted(s_list, reverse=True):
            for b in sorted(b_list, reverse=True):
                for n_kv in n_kv_list:
                    if n_kv != 0:
                        if n_kv >= n or n%n_kv != 0:
                            continue
                    num_kv_heads = n_kv if n_kv !=0 else n

                    if num_kv_heads == n:
                        if b*s > 65536 or b >128:
                            continue
                    else:
                        if b*s > 131072:
                            continue
                    if b*s*num_kv_heads*128*2 >= 2147483647:
                        continue

                    #print(f'collecting heads: {n} kv_heads: {num_kv_heads} seq: {s} batchsize: {b}')
                    # use fp8 kv cache, fp8 context fmha, is_context_phase. in torch flow, int8 kvcache is not supported yet.
                    # fp16 kv cache, fp16 context fmha, is_context_phase
                    test_cases.append([b, s, n, num_kv_heads, 128, False, False, True, 'context_attention_perf.txt'])
                    test_cases.append([b, s, n, num_kv_heads, 128, True, False, True, 'context_attention_perf.txt'])
                    test_cases.append([b, s, n, num_kv_heads, 128, True, True, True, 'context_attention_perf.txt'])

    return test_cases

def get_generation_attention_test_cases():
    test_cases = []

    # generation
    isl = 1
    b_list = [1,2,4,8,16,32,64,128,256,512,1024,2048,]
    b_list_xqa = [1,2,4,8,16,32,64,128,256,512,1024,2048]
    # the i-th token to record. 1 for context phase. mapping to osl definition
    s_list = [2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072]
    # full n {4, 5, 7, 8, 9, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40, 48, 56, 72, 96}
    n_list = [4,8,12,16,24,32,40,48,64]
    n_list_xqa = [4,8,16,32,64,96,128]
    n_kv_list = [1,2,4,8]

    # MHA
    max_bsn = 8192*1024 #2*1024*1024*1024/128/2 INT32MAX/128/2
    max_bsn_largeb = max_bsn//2
    for n in sorted(n_list, reverse=True):
        b_s_dict = {}
        s_b_dict = {}        
        for s in s_list:
            max_b = max_bsn // s // n # b*s*n*byte <= max_bsn
            for b in b_list:
                if b > max_b:
                    break
                if s not in s_b_dict.keys():
                    s_b_dict[s] = {b}
                else:
                    s_b_dict[s].add(b)
        for s, b_set in s_b_dict.items():
            if len(b_set) < 4:
                continue
            for b in b_set:
                if b not in b_s_dict.keys():
                    b_s_dict[b] = {s-1}
                b_s_dict[b].add(s-1)

        for b, s_list_limited in b_s_dict.items():
            target_s_list = sorted(s_list_limited)
            if b >= 256:
                target_s_list = target_s_list[:-1]
            #print(f'collecting MHA heads: {n} batchsize: {b}  steps: {s_list_limited}')
            # fp8 kv cache, fp8 context fmha, is_context_phase
            for s in target_s_list:
                test_cases.append([b, s, n, n, 128, False, False, False, 'generation_attention_perf.txt'])
                test_cases.append([b, s, n, n, 128, True, False, False, 'generation_attention_perf.txt'])

    # XQA
    max_bsn = 8192*1024*2 #2*1024*1024*1024/128/2
    for n in sorted(n_list_xqa, reverse=True):
        b_s_dict = {}
        s_b_dict = {}        
        for s in s_list:
            max_b = max_bsn // s // n
            for b in b_list:
                if b > max_b:
                    break
                if s not in s_b_dict.keys():
                    s_b_dict[s] = {b}
                else:
                    s_b_dict[s].add(b)
        for s, b_set in s_b_dict.items():
            if len(b_set) < 4:
                continue
            for b in b_set:
                if b not in b_s_dict.keys():
                    b_s_dict[b] = {s-1}
                b_s_dict[b].add(s-1)

        for b, s_list_limited in b_s_dict.items():
            target_s_list = sorted(s_list_limited)
            if b >= 256:
                target_s_list = target_s_list[:-1]
            for n_kv in n_kv_list:
                if n_kv >= n:
                    continue
                
                # fp8 kv cache, fp8 context fmha, is_context_phase
                for s in target_s_list:
                    test_cases.append([b, s, n, n_kv, 128, False, False, False, 'generation_attention_perf.txt'])
                    test_cases.append([b, s, n, n_kv, 128, True, False, False, 'generation_attention_perf.txt'])
    return test_cases

def time_fwd(func, *args, repeats=30, verbose=True, desc="", **kwargs):
    return Timing(do_bench(lambda: func(*args, **kwargs), warmup=3, rep=repeats) * 1e-3)

def run_attention_torch(batch_size,
                        input_len,
                        num_heads,
                        num_key_value_heads, # keep same as num_heads for MHA
                        head_dim,
                        use_fp8_kv_cache,
                        use_fp8_context_fmha,
                        is_context_phase,                        
                        perf_filename,
                        device='cuda:0'):
    if use_fp8_context_fmha:
        assert use_fp8_kv_cache, f"If you want to use fp8 context fmha, kv cache must be fp8"
    kvtype = torch.float8_e4m3fn if use_fp8_kv_cache else torch.bfloat16

    seqlen = input_len
    seqlen_q = input_len if is_context_phase else 1
    headdim = head_dim
    headdim_v = head_dim
    q = torch.randn(batch_size, seqlen_q, num_heads, headdim, device=device, dtype=torch.bfloat16, requires_grad=True)
    if is_context_phase:
        k = torch.randn(batch_size, seqlen, num_key_value_heads, headdim, device=device, dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(batch_size, seqlen, num_key_value_heads, headdim_v, device=device, dtype=torch.bfloat16, requires_grad=True)
        k_cache = torch.zeros_like(k, device=device, dtype=torch.bfloat16)
        v_cache = torch.zeros_like(v, device=device, dtype=torch.bfloat16)
        k, v, k_cache, v_cache = [x.detach().to(kvtype).requires_grad_() for x in [k, v, k_cache, v_cache]]
        cache_seqlens = torch.tensor([0 for _ in range(batch_size)], dtype=torch.int32, device=device)
    else:
        k_cache = torch.randn(batch_size, seqlen, num_key_value_heads, headdim, device=device, dtype=torch.bfloat16, requires_grad=True)
        v_cache = torch.randn(batch_size, seqlen, num_key_value_heads, headdim_v, device=device, dtype=torch.bfloat16, requires_grad=True)
        k_cache, v_cache = [x.detach().to(kvtype).requires_grad_() for x in [k_cache, v_cache]]
        k, v, cache_seqlens = None, None, None
    
    def float16attn_fp8kvcache(q, k_cache, v_cache, k, v, **kwargs):
        k_cache = k_cache.to(torch.bfloat16)
        v_cache = v_cache.to(torch.bfloat16)
        k = None if k is None else k.to(torch.bfloat16)
        v = None if v is None else v.to(torch.bfloat16)
        flash_attn_func_v3(q, k_cache, v_cache, k, v, **kwargs)

    if use_fp8_context_fmha:
        q = q.to(kvtype)
        m1 = time_fwd(flash_attn_func_v3, q, k_cache, v_cache, k, v, cache_seqlens=cache_seqlens, repeats=10, verbose=True, desc='Fav3')
    elif use_fp8_kv_cache:
        m1 = time_fwd(float16attn_fp8kvcache, q, k_cache, v_cache, k, v, cache_seqlens=cache_seqlens, repeats=10, verbose=True, desc='Fav3')
    else:
        m1 = time_fwd(flash_attn_func_v3, q, k_cache, v_cache, k, v, cache_seqlens=cache_seqlens, repeats=10, verbose=True, desc='Fav3')

    latency = m1.mean * 1e3

    if is_context_phase:
        isl = input_len
        step = 0
        op_name = 'context_attention'
    else:
        isl = 1
        step = input_len
        op_name = 'generation_attention'

    log_perf(item_list=[{ 
                'batch_size': batch_size,
                'isl': isl,
                'num_heads': num_heads,
                'num_key_value_heads': num_key_value_heads,
                'head_dim': 128,
                'beam_width': 1,
                'attn_dtype': 'fp8' if use_fp8_context_fmha else 'float16',
                'kv_cache_dtype': 'fp8' if use_fp8_kv_cache else "float16",                     
                'step': step, 
                'latency': latency
                }], 
        framework='SGLang', 
        version=pkg_resources.get_distribution('sglang').version, 
        device_name=torch.cuda.get_device_name(device), 
        op_name=op_name, 
        kernel_source='default', 
        perf_filename=perf_filename)
