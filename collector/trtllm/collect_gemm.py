# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from cuda import cuda
import torch
import tensorrt_llm
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig
import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils
import math
from helper import getSMVersion, log_perf

def get_gemm_test_cases():
    x_list = [1,2,4,8,16,32,48,64,80,96,128,160,192,256,384,512,768,1024,2048,4096,8192]
    nk_list = [32,64,128,256,512,768,1024,1536,2048,2560,3072,3584,4096,5120,6144,7168,8192,10240,12288]
    nk_list_ext = [16384,65536] # for coverage and interp purpose
    gemm_list = ['float16']
    if getSMVersion() > 86:
        gemm_list += ['fp8']
        if getSMVersion() < 100:
            gemm_list += ['fp8_block']
    if getSMVersion() >= 100:
        gemm_list += ['nvfp4']

    test_cases=[]
    for gemm_type in gemm_list:
        # x_list_orig+add+ext  <==> nk_list+ext
        for x in sorted(x_list, reverse=True):
            for n in sorted(nk_list+nk_list_ext, reverse=True):
                for k in sorted(nk_list+nk_list_ext, reverse=True):
                    if n*k == 65536*65536:
                        continue
                    if gemm_type == 'nvfp4' or gemm_type == 'fp8_block':
                        if n < 128 or k < 128:
                            continue
                    test_cases.append([gemm_type,x,n,k,'gemm_perf.txt'])

    return test_cases


def run_gemm(gemm_type, m, n, k, perf_filename, device='cuda:0'):
    torch.cuda.set_device(device)
    dtype = torch.bfloat16
    x = torch.randn((m, k), dtype=dtype).to(torch.device(device))

    if gemm_type == 'fp8':
        qc = QuantConfig(quant_algo=QuantAlgo.FP8)
    elif gemm_type == 'fp8_block':
        group_size = 128
        qc = QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES, group_size=group_size)
    elif gemm_type == 'nvfp4':
        group_size = 128
        qc = QuantConfig(quant_algo=QuantAlgo.NVFP4, group_size=group_size)
    else:
        qc = None

    repeat_n = 5 # to reduce impact of L2 cache hit
    op_list = []
    for i in range(repeat_n):
        gemm = Linear(
            k,
            n,
            bias=False,
            dtype=dtype,
            quant_config=qc,
        )

        if gemm_type == 'fp8':
            weights = {'weight': torch.randn((n, k), dtype=torch.bfloat16, device=torch.device(device)).to(dtype=torch.float8_e4m3fn),
                       'weight_scale': torch.randn(1, dtype=torch.float32, device=torch.device(device))}
        elif gemm_type == 'fp8_block':
            weights = {'weight': torch.randn((n, k), dtype=torch.bfloat16, device=torch.device(device)).to(dtype=torch.float8_e4m3fn),
                       'weight_scale': torch.randn((math.ceil(n/group_size), math.ceil(k/group_size)), dtype=torch.float32, device=torch.device(device))}
        elif gemm_type == 'nvfp4':
            # From trtllm test case
            x_sf_global = (448 * 6) / x.abs().max().float()
            w = torch.randn((n, k), dtype=torch.float16, device=torch.device(device))
            w_sf_global = (448 * 6) / w.abs().max().float()
            w_fp4, w_sf_block = torch.ops.trtllm.fp4_quantize(w, w_sf_global, 16, False)
            w_sf_block_unswizzled = (
                    torch.ops.trtllm.nvfp4_block_scale_interleave_reverse(
                        w_sf_block.cpu().view(k, -1)))
            weights = {'weight': w_fp4.cpu(),
                       'weight_scale': w_sf_block_unswizzled.view(torch.float8_e4m3fn),
                       'weight_scale_2': 1.0 / w_sf_global.cpu(),
                       'input_scale': 1.0 / x_sf_global.cpu()}
        else:
            weights = {'weight': torch.randn((n, k), dtype=torch.bfloat16, device=torch.device(device))}

        gemm.load_weights([weights])
        gemm.to(torch.device(device))
        gemm.forward(x) # dry run to init        
        op_list.append(gemm)

    num_warmups = 3
    num_runs = 6

    # capture
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for op in op_list:
            op.forward(x)
    # warmup
    for i in range(num_warmups):
        g.replay()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for i in range(num_runs):
        g.replay()
    end_event.record()
    torch.cuda.synchronize()
    latency = start_event.elapsed_time(end_event)/num_runs/len(op_list)

    log_perf(item_list=[{ 
                'gemm_dtype': gemm_type,
                'm': m,
                'n': n,
                'k': k,
                'latency': latency
                }], 
    framework='TRTLLM', 
    version=tensorrt_llm.__version__, 
    device_name=torch.cuda.get_device_name(device), 
    op_name='gemm', 
    kernel_source='torch_flow', 
    perf_filename=perf_filename)