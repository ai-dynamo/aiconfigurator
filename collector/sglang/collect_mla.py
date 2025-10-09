# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from collections import namedtuple
from functools import partial
import math
import os
from typing import NamedTuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

import time
# cudnn = None

from helper import log_perf
Timing = NamedTuple('timing', [('mean', float)])
from sgl_kernel.flash_attn import flash_attn_varlen_func as flash_attn_varlen_func_v3

from triton.testing import do_bench

import math
import random

import torch
import triton
import pkg_resources

# pip install flashinfer-python
from flash_mla import flash_mla_with_kvcache, get_mla_metadata

DISABLE_BACKWARD = os.getenv("FLASH_ATTENTION_DISABLE_BACKWARD", "FALSE") == "TRUE"

def get_context_mla_test_cases():
    dtype_list = [torch.bfloat16, torch.float8_e4m3fn]
    test_cases = []
    n_list = [64, 128]
    b_list = [1,2,4,8,16,32,64,128,256]
    s_list = [16,32,64,128,256,512,1024,1536,2048,3072,4096,6144,8192,10240,12288,16384]
    for n in n_list:
        for b in b_list:
            for s in s_list:
                for dtype in dtype_list:
                    for tp_size in [1,2,4,8,16,32,64]:
                        if b*s > 32768:
                            continue
                        # (input_len, batch_size, output_len, kv_cache_dtype, world_size, tp_size, tokens_per_block, warming_up, test_ite, is_context_phase)
                        test_cases.append([s,b,1,dtype,n,tp_size,tp_size,64,10,6,True,"context_mla_perf.txt"])
    return test_cases

def get_generation_mla_test_cases():
    dtype_list = [torch.bfloat16, torch.float8_e4m3fn]
    test_cases = []
    n_list = [64, 128]
    for n in n_list:
        for b in [1,2,4,8,16,32,64,128,256,512,1024]:
            for s in [2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072]: # [target token s] is equivelant to [in: s-1, step=1]
                for dtype in dtype_list:
                    for tp_size in [1,2,4,8,16,32,64]:
                        if b*s > 1024*4096*2*2:
                            continue
                        # (input_len, batch_size, output_len, kv_cache_dtype, world_size, tp_size, tokens_per_block, warming_up, test_ite, is_context_phase)
                        test_cases.append([s-1,b,1,dtype,n,tp_size,tp_size,64,10,6,False,"generation_mla_perf.txt"])
    return test_cases


def time_fwd(func, *args, repeats=30, verbose=True, desc="", **kwargs):
    return Timing(do_bench(lambda: func(*args, **kwargs), warmup=3, rep=repeats) * 1e-3)

@torch.inference_mode()
def run_flash_mla(q, block_table, blocked_k, descale_q, descale_k, max_seqlen_pad, block_size, b, s_q, cache_seqlens, h_q, h_kv, d, dv, causal, dtype):
    for i in range(b):
        blocked_k.view(b, max_seqlen_pad, h_kv, d)[i, cache_seqlens[i].item():] = float("nan")
    blocked_v = blocked_k[..., :dv]

    tile_scheduler_metadata, num_splits = get_mla_metadata(cache_seqlens, s_q * h_q // h_kv, h_kv)

    def flash_mla():
        if dtype == torch.bfloat16:
            return flash_mla_with_kvcache(
                q, blocked_k, block_table, cache_seqlens, dv,
                tile_scheduler_metadata, num_splits, causal=causal,
            )
        else:
            return flash_mla_with_kvcache(
                q, blocked_k, block_table, cache_seqlens, dv,
                tile_scheduler_metadata, num_splits, causal=causal,
                descale_q=descale_q, descale_k=descale_k
            )

    out_flash, lse_flash = flash_mla()
    t = triton.testing.do_bench(flash_mla)
    return out_flash, lse_flash, t


def compare_a(target, b, s_q, cache_seqlens, h_q, h_kv, d, dv, causal, dtype, device):
    #print(f"{target}: {b=}, {s_q=}, mean_seqlens={cache_seqlens.float().mean()}, {h_q=}, {h_kv=}, {d=}, {dv=}, {causal=}, {dtype=}")
    #torch.set_default_dtype(dtype)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(device)
    torch.cuda.set_device(device)
    torch.manual_seed(0)
    random.seed(0)
    target_func = run_flash_mla
    
    total_seqlens = cache_seqlens.sum().item()
    mean_seqlens = cache_seqlens.float().mean().int().item()
    max_seqlen = cache_seqlens.max().item()
    max_seqlen_pad = triton.cdiv(max_seqlen, 256) * 256
    # print(f"{total_seqlens=}, {mean_seqlens=}, {max_seqlen=}")

    q = torch.randn(b, s_q, h_q, d).to(dtype)
    block_size = 64
    block_table = torch.arange(b * max_seqlen_pad // block_size, dtype=torch.int32).view(b, max_seqlen_pad // block_size)
    blocked_k = torch.randn(block_table.numel(), block_size, h_kv, d).to(dtype)

    descale_q = torch.ones((1), dtype=torch.float32)
    descale_k = torch.ones((1), dtype=torch.float32)
    
    out_b, lse_b, perf_b = target_func(q, block_table, blocked_k, descale_q, descale_k, max_seqlen_pad, block_size, b, s_q, cache_seqlens, h_q, h_kv, d, dv, causal, dtype)

    FLOPS = s_q * total_seqlens * h_q * (d + dv) * 2
    bytes = (total_seqlens * h_kv * d + b * s_q * h_q * d + b * s_q * h_q * dv) * (torch.finfo(dtype).bits // 8)
    #print(f"perf {target}: {perf_b:.3f} ms, {FLOPS / 10 ** 9 / perf_b:.0f} TFLOPS, {bytes / 10 ** 6 / perf_b:.0f} GB/s")
    return perf_b

def run_mla(input_len, batch_size, output_len, kv_cache_dtype, num_heads, world_size, tp_size, tokens_per_block, warming_up, test_ite, is_context_phase, perf_filename, device='cuda:0'):
    torch.cuda.set_device(device)

    assert kv_cache_dtype in [torch.bfloat16, torch.float8_e4m3fn], "only support torch.bfloat16 (because of flash mla)"
    assert num_heads % tp_size == 0, "num_heads != N * tp_size"
    num_heads = int(num_heads / tp_size)

    if is_context_phase:
        seqlen = input_len
        seqlen_q = input_len
        headdim = 128+64
        headdim_v = 128
        q = torch.randn(batch_size, seqlen_q, num_heads, headdim, device=device, dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(batch_size, seqlen, num_heads, headdim, device=device, dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(batch_size, seqlen, num_heads, headdim_v, device=device, dtype=torch.bfloat16, requires_grad=True)
        q, k, v = [x.detach().to(kv_cache_dtype).requires_grad_() for x in [q, k, v]]

        q_unpad, k_unpad, v_unpad = [rearrange(x.detach(), "b s h d -> (b s) h d").requires_grad_() for x in [q, k, v]]
        cu_seqlens_q = torch.arange(batch_size + 1, device=device, dtype=torch.int32) * seqlen_q
        cu_seqlens_k = torch.arange(batch_size + 1, device=device, dtype=torch.int32) * seqlen
        
        m1 = time_fwd(
            flash_attn_varlen_func_v3,
            q_unpad,
            k_unpad,
            v_unpad,
            cu_seqlens_q,
            cu_seqlens_k,
            seqlen_q,
            seqlen,
            causal=True,
            window_size=(-1, -1),
            softcap=0.0,
            num_splits=0,
            pack_gqa=None,
            repeats=10,
            verbose=True,
            desc='Fav3'
        )
        latency = m1.mean * 1e3
    else:
        latency = compare_a(
                'flash_mla',
                batch_size,
                1,
                torch.tensor([input_len for _ in range(batch_size)], dtype=torch.int32, device=device),
                num_heads,
                1,
                512+64,
                512,
                True,
                kv_cache_dtype,
                device)

    if is_context_phase:
        isl = input_len
        step = 0
    else:
        isl = 1
        step = input_len

    if kv_cache_dtype == torch.bfloat16:
        str_type = 'float16'
    else:
        str_type = 'fp8'

    log_perf(item_list=[{ 
                        'mla_dtype': 'float16',
                        'kv_cache_dtype': str_type, 
                        'num_heads': num_heads,
                        'batch_size': batch_size,
                        'isl': isl,
                        'tp_size': tp_size,
                        'step': step, 
                        'latency': latency
                        }], 
            framework='SGLang', 
            version=pkg_resources.get_distribution('sglang').version, 
            device_name=torch.cuda.get_device_name(device), 
            op_name=f'mla_{"context" if is_context_phase else "generation"}', 
            kernel_source='default', 
            perf_filename=perf_filename)
    
