# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from cuda import cuda
import torch
import torch.nn as nn
import tensorrt_llm
import math
from helper import getSMVersion, log_perf

def get_conv1d_test_cases():
    """
    Generate test cases for Conv1D operations.
    
    Test parameters:
    - batch_size: batch size (analogous to 'm' in GEMM)
    - in_channels: number of input channels
    - out_channels: number of output channels  
    - kernel_size: size of the convolution kernel
    - seq_length: sequence length
    """
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    channel_sizes = [64, 128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096]
    kernel_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    seq_lengths = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    
    conv1d_types = ['float16']
    if getSMVersion() > 86:
        conv1d_types += ['fp8']
        if getSMVersion() < 100:
            conv1d_types += ['fp8_block']
    if getSMVersion() >= 100:
        conv1d_types += ['nvfp4']

    test_cases = []
    for conv_type in conv1d_types:
        # Generate test cases with various combinations
        for batch_size in batch_sizes:
            for in_ch in channel_sizes:
                for out_ch in channel_sizes:
                    for kernel_size in kernel_sizes:
                        for seq_len in seq_lengths:
                            # Skip extremely large cases
                            if batch_size * in_ch * seq_len > 16777216:
                                continue
                            if conv_type == 'nvfp4' or conv_type == 'fp8_block':
                                if in_ch < 128 or out_ch < 128:
                                    continue
                            test_cases.append([conv_type, batch_size, in_ch, out_ch, kernel_size, seq_len, 'conv1d_perf.txt'])

    return test_cases


def run_conv1d(conv_type, batch_size, in_channels, out_channels, kernel_size, seq_length, perf_filename, device='cuda:0'):
    """
    Run Conv1D performance benchmarking.
    
    Args:
        conv_type: Type of convolution ('float16', 'fp8', 'fp8_block', 'nvfp4')
        batch_size: Batch size
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolution kernel
        seq_length: Sequence length
        perf_filename: Output file for performance results
        device: CUDA device to use
    """
    device = torch.device(device)
    torch.cuda.set_device(device)
    torch.set_default_device(device)

    # For now, we focus on float16/bfloat16 benchmarking
    # Quantized Conv1D support can be added when available in TensorRT-LLM
    if conv_type != 'float16':
        # Skip non-float16 for now as Conv1D quantization in TensorRT-LLM needs verification
        return
    
    dtype = torch.bfloat16
    # Conv1D expects input shape: (batch_size, in_channels, seq_length)
    x = torch.randn((batch_size, in_channels, seq_length), dtype=dtype).to(torch.device(device))

    repeat_n = 5  # to reduce impact of L2 cache hit
    op_list = []
    
    for i in range(repeat_n):
        # Use PyTorch's native Conv1d
        conv1d = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=kernel_size//2,  # 'same' padding
            bias=False,
            dtype=dtype,
        )
        
        # Initialize weights randomly
        conv1d.weight.data = torch.randn((out_channels, in_channels, kernel_size), dtype=dtype, device=device)
        
        conv1d.to(torch.device(device))
        conv1d(x)  # dry run to init        
        op_list.append(conv1d)

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

    log_perf(
        item_list=[{ 
            'conv_dtype': conv_type,
            'batch_size': batch_size,
            'in_channels': in_channels,
            'out_channels': out_channels,
            'kernel_size': kernel_size,
            'seq_length': seq_length,
            'latency': latency
        }], 
        framework='TRTLLM', 
        version=tensorrt_llm.__version__, 
        device_name=torch.cuda.get_device_name(device), 
        op_name='conv1d', 
        kernel_source='torch_flow', 
        perf_filename=perf_filename
    )
