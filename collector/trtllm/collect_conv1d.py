# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import os
from cuda import cuda
import torch
import tensorrt_llm
from tensorrt_llm._torch.modules.mamba.causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from helper import log_perf

def get_conv1d_fn_test_cases():
    """
    Generate test cases for Conv1DFn operations.
    
    Test parameters:
    - batch_size: batch size
    - isl: sequence length
    - conv_kernel_size: size of the convolution kernel
    - conv_dim: dimension of the convolution
    - tp_size: attention tensor parallel size
    """
    b_list = [1,2,4,8,16,32,64,128,256,512,1024,2048]
    s_list = [2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072]
    tp_sizes = [1, 2, 4, 8]
    conv_dims = [64, 128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096]
    kernel_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]    

    test_cases = []
    for batch_size in b_list:
        for isl in s_list:
            for tp_size in tp_sizes:
                for conv_dim in conv_dims:
                    for kernel_size in kernel_sizes:
                        test_cases.append([batch_size, isl, kernel_size, conv_dim, tp_size, 'conv1d_fn_perf.txt'])

    return test_cases


def run_conv1d_fn(batch_size, isl, conv_kernel_size, conv_dim, tp_size, perf_filename, device='cuda:0'):
    """
    Run Conv1DFn performance benchmarking.
    
    Args:
        batch_size: Batch size
        isl: Sequence length
        conv_kernel_size: Size of the convolution kernel
        conv_dim: Dimension of the convolution
        tp_size: Attention tensor parallel size
        perf_filename: Output file for performance results
        device: CUDA device to use
    """
    dtype = torch.bfloat16
    mixed_qkv = torch.randn((batch_size * isl, conv_dim // tp_size), dtype=dtype).to(torch.device(device))
    conv1d_weights = torch.randn((conv_dim // tp_size, conv_kernel_size), dtype=dtype, device=device)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        mixed_qkv_trans = mixed_qkv.transpose(0, 1)
        # TODO: measure optional arguments
        causal_conv1d_fn(
            mixed_qkv_trans,
            conv1d_weights,
        ).transpose(0, 1)

    num_warmups = 3
    num_runs = 6
    
    # warmup
    for _ in range(num_warmups):
        g.replay()

    # measure
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(num_runs):
        g.replay()
    end_event.record()
    torch.cuda.synchronize()
    latency = start_event.elapsed_time(end_event)/num_runs

    log_perf(
        item_list=[{ 
            'batch_size': batch_size,
            'isl': isl,
            'conv_kernel_size': conv_kernel_size,
            'conv_dim': conv_dim,
            'tp_size': tp_size,
            'latency': latency
        }], 
        framework='TRTLLM', 
        version=tensorrt_llm.__version__, 
        device_name=torch.cuda.get_device_name(device), 
        op_name='conv1d_fn', 
        kernel_source='default', 
        perf_filename=perf_filename
    )

def get_conv1d_update_test_cases():
    """
    Generate test cases for Conv1DUpdate operations.
    
    Test parameters:
    - batch_size: batch size
    - isl: sequence length
    - conv_kernel_size: size of the convolution kernel
    - conv_dim: dimension of the convolution
    - tp_size: attention tensor parallel size
    """
    b_list = [1,2,4,8,16,32,64,128,256,512,1024,2048]
    s_list = [2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072]
    tp_sizes = [1, 2, 4, 8]
    conv_dims = [1,2,4,8,16,32]
    kernel_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]    

    test_cases = []
    for batch_size in b_list:
        for isl in s_list:
            for tp_size in tp_sizes:
                for conv_dim in conv_dims:
                    for kernel_size in kernel_sizes:
                        test_cases.append([batch_size, isl, kernel_size, conv_dim, tp_size, 'conv1d_update_perf.txt'])

    return test_cases


def run_conv1d_update(batch_size, isl, conv_kernel_size, conv_dim, tp_size, perf_filename, device='cuda:0'):
    """
    Run Conv1DUpdate performance benchmarking.
    
    Args:
        batch_size: Batch size
        isl: Sequence length
        conv_kernel_size: Size of the convolution kernel
        conv_dim: Dimension of the convolution
        tp_size: Attention tensor parallel size
        perf_filename: Output file for performance results
        device: CUDA device to use
    """
    dtype = torch.bfloat16
    mixed_qkv = torch.randn((batch_size * isl, conv_dim // tp_size), dtype=dtype).to(torch.device(device))
    conv1d_weights = torch.randn((conv_dim // tp_size, conv_kernel_size), dtype=dtype, device=device)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        # TODO: measure optional arguments
        causal_conv1d_update(
            mixed_qkv,
            conv1d_weights,
        )

    num_warmups = 3
    num_runs = 6
    
    # warmup
    for _ in range(num_warmups):
        g.replay()

    # measure
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(num_runs):
        g.replay()
    end_event.record()
    torch.cuda.synchronize()
    latency = start_event.elapsed_time(end_event)/num_runs

    log_perf(
        item_list=[{ 
            'batch_size': batch_size,
            'isl': isl,
            'conv_kernel_size': conv_kernel_size,
            'conv_dim': conv_dim,
            'tp_size': tp_size,
            'latency': latency
        }], 
        framework='TRTLLM', 
        version=tensorrt_llm.__version__, 
        device_name=torch.cuda.get_device_name(device), 
        op_name='conv1d_fn', 
        kernel_source='default', 
        perf_filename=perf_filename
    )
