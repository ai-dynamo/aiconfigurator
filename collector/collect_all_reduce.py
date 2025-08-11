# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from argparse import ArgumentParser

# isort: off
import torch
# isort: on
from cuda import cuda, cudart
from polygraphy.backend.trt import CreateConfig, EngineFromNetwork

import tensorrt_llm as tllm
from tensorrt_llm import Mapping, Tensor
from tensorrt_llm._utils import OMPI_COMM_TYPE_HOST, mpi_comm
from tensorrt_llm.functional import (AllReduceParams, AllReduceStrategy, allreduce)
from tensorrt_llm.plugin.plugin import current_all_reduce_helper,CustomAllReduceHelper
import tensorrt as trt
from statistics import mean

class AllReduceProfiler(trt.IProfiler):
    def __init__(self, perf_filename='custom_allreduce_perf.txt', prefix='', target_layers=[0,1,2]):
        trt.IProfiler.__init__(self)
        self._counter = 0
        self._prefix = prefix
        self._target_layers = target_layers
        self._latencies = []
        self._layer_names = []
        self._perf_filename = perf_filename

    def report_layer_time(self, layer_name, ms):
        
        if self._counter in self._target_layers:
            self._layer_names.append(layer_name)
            self._latencies.append(ms)
        self._counter += 1
    
    def write_to_file(self):
        #return

        num_cases = len(self._latencies)
        assert(num_cases > 2)
        latencies = sorted(self._latencies)
        latencies = latencies[1:-1]
        latency = mean(latencies)
        with open(self._perf_filename, 'a') as f:
            f.write(self._prefix + f',{self._layer_names[-1]},{latency}\n')


def allreduce_benchmark(dtype: str,
                        test_range: str = "10,10000000,10",
                        no_header: bool = False):
    tllm.logger.set_level('error')
    world_size = tllm.mpi_world_size()
    rank = tllm.mpi_rank()
    local_comm = mpi_comm().Split_type(split_type=OMPI_COMM_TYPE_HOST)
    local_rank = local_comm.Get_rank()
    gpus_per_node = local_comm.Get_size()

    torch.cuda.set_device(local_rank)
    cudart.cudaSetDevice(local_rank)

    mapping = Mapping(world_size=world_size, rank=rank, gpus_per_node=gpus_per_node, tp_size=world_size)

    if world_size == 1:
        raise RuntimeError("Benchmark must run with mpi_world_size > 1")

    torch_dtype = tllm._utils.str_dtype_to_torch(dtype)
    min_size, max_size, ratio = [int(i) for i in test_range.split(",")]

    size = min_size
    dtype_size = torch.finfo(torch_dtype).bits // 8
    while size < max_size:
        inner_loop = 100 if size <= 16777216 else 15

        input = torch.ones(size, dtype=torch_dtype, device="cuda")

        for strategy in [
                AllReduceStrategy.AUTO,
                AllReduceStrategy.NCCL,
                AllReduceStrategy.ONESHOT,
                AllReduceStrategy.TWOSHOT,
        ]:
            builder = tllm.Builder()
            builder.strongly_typed = False
            net = builder.create_network()
            net.plugin_config.set_nccl_plugin(dtype)
            _buffers, workspace = current_all_reduce_helper(
            ).allocate_workspace(mapping, CustomAllReduceHelper.max_workspace_size_auto(
                    mapping.tp_size))
            

            with tllm.net_guard(net):
                network = tllm.default_trtnet()

                x = Tensor(name='x',
                           shape=input.shape,
                           dtype=tllm.str_dtype_to_trt(dtype))

                current_all_reduce_helper().set_workspace_tensor(mapping)

                current = x
                for _ in range(inner_loop):
                    current = allreduce(current, mapping.tp_group, all_reduce_params=AllReduceParams(strategy=strategy))
                output = current.trt_tensor

                network.mark_output(output)
                output.name = 'output'
                output.dtype = tllm.str_dtype_to_trt(dtype)

            build_engine = EngineFromNetwork(
                (builder.trt_builder, net.trt_network),
                config=CreateConfig(
                    fp16=(dtype == 'float16'),
                    bf16=(dtype == 'bfloat16'),
                    precision_constraints='obey',
                ))

            output = torch.zeros_like(input)

            stream = torch.cuda.current_stream()
            feed_dict = {'x': input, 'all_reduce_workspace': workspace}

            session = tllm.runtime.Session.from_engine(build_engine())
            if mapping.rank == 0:
                session._context.profiler = AllReduceProfiler(prefix=f'{dtype},{world_size},{size},{strategy.name}',target_layers=[i for i in range(inner_loop-12,inner_loop-5)])
            _, start = cuda.cuEventCreate(0)
            _, stop = cuda.cuEventCreate(0)

            tllm.mpi_barrier()

            cuda.cuEventRecord(start, stream.cuda_stream)
            session.run(inputs=feed_dict,
                        outputs={"output": output},
                        stream=stream.cuda_stream)
            cuda.cuEventRecord(stop, stream.cuda_stream)
            torch.cuda.synchronize()
            _, ms = cuda.cuEventElapsedTime(start, stop)

            if mapping.rank == 0:
                print(f"{size=}, {strategy=}, {ms=}")
                session._context.profiler.write_to_file()
        size *= ratio
        if mapping.rank == 0:
            print("")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dtype", "-t", default="float16")
    parser.add_argument(
        "--range",
        "-r",
        default="2048,536870913,2",  # 256 to 256M
        help="min_size,max_size,multiplicative_ratio")
    parser.add_argument("--no-header", action="store_true")
    args = parser.parse_args()

    allreduce_benchmark(args.dtype, args.range, args.no_header)

