# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import os

import torch
try:
    from cuda import cudart
except:
    from cuda.bindings import runtime as cudart
from tensorrt_llm import Mapping
from tensorrt_llm._torch.distributed import (
    AllReduce,
    AllReduceFusionOp,
    AllReduceParams,
)
from tensorrt_llm.functional import AllReduceStrategy


class NCCLProfiler:
    def __init__(self, perf_filename="/path/to/custom_ar.txt", prefix=""):
        self._prefix = prefix
        self._latency = 0.0
        self._layer_name = ""
        self._perf_filename = perf_filename

    def report_layer_time(self, layer_name, ms):
        self._layer_name = layer_name
        self._latency = ms

    def write_to_file(self):
        with open(self._perf_filename, "a") as f:
            f.write(self._prefix + f",{self._layer_name},{self._latency}\n")


world_size = int(os.environ["SLURM_NTASKS"])
rank = int(os.environ["RANK"])
gpus_per_node = int(os.environ["SLURM_NTASKS_PER_NODE"])
local_rank = int(os.environ["SLURM_LOCALID"])

# print(world_size, rank, gpus_per_node, local_rank)
torch.cuda.set_device(local_rank)
cudart.cudaSetDevice(local_rank)
mapping = Mapping(world_size=world_size, rank=rank, gpus_per_node=gpus_per_node, tp_size=world_size)

all_reduce_params = AllReduceParams(
    strategy=AllReduceStrategy.AUTO,
    fusion_op=AllReduceFusionOp.NONE,
    residual=None,
    norm_weight=None,
    scale=None,
    bias=None,
    eps=1e-6,
)

min_size = 128
max_size = 4300000000


def get_input_shape_and_comm_size(size, token_dim=4096):
    if size <= token_dim:
        return [1, size]
    else:
        num_token = size // token_dim
        return [num_token, token_dim]


repeat_n = 5
num_warmups = 3
num_runs = 6

while min_size < max_size:
    input_shape = get_input_shape_and_comm_size(min_size)

    input_tensor = torch.ones(input_shape, dtype=torch.bfloat16, device="cuda")
    # print(input_tensor)

    # to reduce impact of L2 cache hit
    op_list = []
    for i in range(repeat_n):
        allreduce = AllReduce(mapping=mapping).cuda()
        output = allreduce(input_tensor, all_reduce_params=all_reduce_params)  # dry run to init
        op_list.append(allreduce)
        # print(output)

    # capture
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for op in op_list:
            op(input_tensor, all_reduce_params=all_reduce_params)

    # warmup
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    for i in range(num_warmups):
        g.replay()
    torch.cuda.synchronize()
    start_event.record()
    for i in range(num_runs):
        g.replay()
    end_event.record()
    torch.cuda.synchronize()

    latency = start_event.elapsed_time(end_event) / num_runs / repeat_n
    print(latency)

    if rank == 0 and local_rank == 0:
        profiler = NCCLProfiler(prefix=f'half,{world_size},{min_size},"ALLREDUCE_AUTO"')
        profiler.report_layer_time("all_reduce", latency)
        profiler.write_to_file()
    min_size *= 2
