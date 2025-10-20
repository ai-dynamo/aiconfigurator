# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import os
from cuda import cuda
import torch
import tensorrt_llm
from tensorrt_llm._torch.modules.fla.chunk import chunk_gated_delta_rule
from tensorrt_llm._torch.modules.fla.fused_sigmoid_gating_recurrent import fused_sigmoid_gating_delta_rule_update
from helper import log_perf

def get_chunk_gated_delta_rule_test_cases():
    """
    Generate test cases for chunk_gated_delta_rule() operations.
    
    Test parameters:
    - num_heads: number of heads
    - head_k_dim: dimension of the key heads
    - head_v_dim: dimension of the value heads
    - num_value_heads: number of value heads
    - isl: sequence length
    """
    num_heads_list = [1,2,4,8,16,32,64,128,256,512,1024,2048]
    head_k_dim_list = [1,2,4,8,16,32,64,128]
    head_v_dim_list = [1,2,4,8,16,32,64,128]
    num_value_heads_list = [1,2,4,8,16,32,64,128]
    isl_list = [2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072]

    test_cases = []
    for num_heads in num_heads_list:
            for head_k_dim in head_k_dim_list:
                for head_v_dim in head_v_dim_list:
                    for num_value_heads in num_value_heads_list:
                        for isl in isl_list:
                            test_cases.append([num_heads, head_k_dim, head_v_dim, num_value_heads, isl, 'chunk_gated_delta_rule_perf.txt'])

    return test_cases


def run_chunk_gated_delta_rule(num_heads, head_k_dim, head_v_dim, num_value_heads, isl, perf_filename, device='cuda:0'):
    """
    Run chunk_gated_delta_rule() performance benchmarking.

    Args:
        num_heads: Number of heads
        head_k_dim: Dimension of the key heads
        head_v_dim: Dimension of the value heads
        num_value_heads: Number of value heads
        isl: Sequence length
        perf_filename: Output file for performance results
        device: CUDA device to use
    """
    # NOTICE: ignored fused_gdn_gating operation
    dtype = torch.bfloat16
    q = torch.randn((1, isl, num_heads, head_k_dim), dtype=dtype).to(torch.device(device))
    k = torch.randn((1, isl, num_heads, head_v_dim), dtype=dtype).to(torch.device(device))
    v = torch.randn((1, isl, num_value_heads, head_v_dim), dtype=dtype).to(torch.device(device))
    gate = torch.randn((1, isl, num_heads), dtype=dtype).to(torch.device(device))
    beta = torch.randn((1, isl, num_heads), dtype=dtype).to(torch.device(device))

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        chunk_gated_delta_rule(q, k, v, gate, beta)

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
            'num_heads': num_heads,
            'head_k_dim': head_k_dim,
            'head_v_dim': head_v_dim,
            'num_value_heads': num_value_heads,
            'isl': isl,
            'latency': latency
        }], 
        framework='TRTLLM', 
        version=tensorrt_llm.__version__, 
        device_name=torch.cuda.get_device_name(device), 
        op_name='chunk_gated_delta_rule', 
        kernel_source='default', 
        perf_filename=perf_filename
    )

def get_gated_delta_rule_update_test_cases():
    """
    Generate test cases for Conv1DUpdate operations.
    
    Test parameters:
    - batch_size: batch size
    - isl: sequence length
    - num_heads: number of heads
    - head_k_dim: dimension of the key heads
    - head_v_dim: dimension of the value heads
    - num_value_heads: number of value heads
    - max_batch_size: maximum batch size
    """
    b_list = [1,2,4,8,16,32,64,128,256,512,1024,2048]
    s_list = [2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072]
    num_heads_list = [1,2,4,8,16,32,64,128,256,512,1024,2048]
    head_k_dim_list = [1,2,4,8,16,32,64,128]
    head_v_dim_list = [1,2,4,8,16,32,64,128]
    num_value_heads_list = [1,2,4,8,16,32,64,128]
    max_batch_size_list = [1,2,4,8,16,32,64,128,256,512,1024,2048]

    test_cases = []
    for batch_size in b_list:
        for isl in s_list:
            for num_heads in num_heads_list:
                for head_k_dim in head_k_dim_list:
                    for head_v_dim in head_v_dim_list:
                        for num_value_heads in num_value_heads_list:
                            for max_batch_size in max_batch_size_list:
                                test_cases.append([batch_size, isl, num_heads, head_k_dim, head_v_dim, num_value_heads, max_batch_size, 'gated_delta_rule_update_perf.txt'])

    return test_cases


def run_gated_delta_rule_update(batch_size, isl, num_heads, head_k_dim, head_v_dim, num_value_heads, max_batch_size, perf_filename, device='cuda:0'):
    """
    Run fused_sigmoid_gating_delta_rule_update() performance benchmarking.
    
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
    A_log = torch.randn((num_heads * num_value_heads), dtype=dtype).to(torch.device(device))
    dt_bias = torch.randn((num_heads * num_value_heads), dtype=dtype).to(torch.device(device))
    q = torch.randn((batch_size, isl, num_heads, head_k_dim), dtype=dtype).to(torch.device(device))
    k = torch.randn((batch_size, isl, num_heads, head_k_dim), dtype=dtype).to(torch.device(device))
    v = torch.randn((batch_size, isl, num_value_heads, head_v_dim), dtype=dtype).to(torch.device(device))
    a = torch.randn((batch_size * isl, num_heads * num_value_heads), dtype=dtype).to(torch.device(device))
    b = torch.randn((batch_size, isl, num_heads * num_value_heads), dtype=dtype).to(torch.device(device))
    initial_state_source = torch.randn((max_batch_size, num_heads * num_value_heads, head_k_dim, head_v_dim), dtype=dtype).to(torch.device(device))
    initial_state_indices = torch.randn((batch_size), dtype=dtype).to(torch.device(device))
    softplus_beta = 1.0
    softplus_threshold = 20.0

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        # TODO: measure optional arguments
        fused_sigmoid_gating_delta_rule_update(
            A_log =A_log,
            dt_bias = dt_bias,
            q = q,
            k = k,
            v = v,
            a = a,
            b = b,
            initial_state_source = initial_state_source,
            initial_state_indices = initial_state_indices,
            softplus_beta = softplus_beta,
            softplus_threshold = softplus_threshold,
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
            'num_heads': num_heads,
            'head_k_dim': head_k_dim,
            'head_v_dim': head_v_dim,
            'num_value_heads': num_value_heads,
            'max_batch_size': max_batch_size,
            'latency': latency
        }], 
        framework='TRTLLM', 
        version=tensorrt_llm.__version__, 
        device_name=torch.cuda.get_device_name(device), 
        op_name='gated_delta_rule_update', 
        kernel_source='default', 
        perf_filename=perf_filename
    )
