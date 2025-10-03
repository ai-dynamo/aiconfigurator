# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from argparse import ArgumentParser
import os
import subprocess
import torch
from helper import log_perf

def NCCL_benchmark(dtype: str,
                   NCCL_op: str = "all_gather",
                   test_range: str = "10,10000000,1000",
                   num_gpus: int = 8):
    NCCL_test_bin = ''
    if NCCL_op == "all_gather":
        NCCL_test_bin = 'all_gather_perf'
    elif NCCL_op == "alltoall":
        NCCL_test_bin = 'alltoall_perf'
    elif NCCL_op == "reduce_scatter":
        NCCL_test_bin = 'reduce_scatter_perf'
    elif NCCL_op == "all_reduce":
        NCCL_test_bin = "all_reduce_perf"
    assert(NCCL_test_bin != '')
    
    min_size, max_size, ratio = [int(i) for i in test_range.split(",")]
    size = min_size

    major, minor, patch = torch.cuda.nccl.version()
    NCCL_version = f'{major}.{minor}.{patch}'

    bytes_per_element = 2 if dtype == 'half' else 1
    
    while size < max_size:
        inner_loop = 100 if size <= 16777216 else 60
        cmd_args = [NCCL_test_bin, '-b', str(size), '-e', str(size), '-t', str(num_gpus), '-d', dtype, '-w', '40', '-a', '1', '-n', str(inner_loop), '-c', '0']
        result = subprocess.run(cmd_args, capture_output=True, text=True)
        print_lines = result.stdout.split('\n')
        for index_line in range(len(print_lines)):
            if 'time' in print_lines[index_line]:
                break
        latency = float(print_lines[index_line + 2].split()[5])*1e-3 # us to ms
        
        print(NCCL_test_bin, f"{size=}, {latency=}")
        log_perf(item_list=[{ 
                    'nccl_dtype': dtype,
                    'num_gpus': num_gpus,
                    'message_size': size//bytes_per_element,
                    'latency': latency
                    }], 
        framework='TRTLLM', 
        version=NCCL_version, 
        device_name=torch.cuda.get_device_name(), 
        op_name=NCCL_op, 
        kernel_source='NCCL', 
        perf_filename='nccl_perf.txt')

        size *= ratio


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--NCCL_op",
        "-NCCL",
        default="all_gather",
        choices=['all_gather', 'alltoall', 'reduce_scatter', 'all_reduce'],
        help="NCCL OP: all_gather, alltoall, reduce_scatter, all_reduce")
    parser.add_argument(
        "--dtype", 
        "-t", 
        default="half",
        choices=['half', 'int8'],
        help="NCCL OP data type")
    parser.add_argument(
        "--range",
        "-r",
        default="512,536870913,2",  # 512B to 512MB
        help="min_size,max_size,multiplicative_ratio")
    parser.add_argument("--num_gpus", "-n", default=8, type=int)
    args = parser.parse_args()

    NCCL_benchmark(args.dtype, args.NCCL_op, args.range, args.num_gpus)

