# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
from argparse import ArgumentParser

import torch

from helper import log_perf


def nccl_benchmark(
    dtype: str,
    nccl_op: str = "all_gather",
    test_range: str = "10,10000000,1000",
    num_gpus: int = 8,
    power_monitor=None,
    measure_power=False,
):
    """
    Run NCCL benchmark with optional power measurement.

    Args:
        dtype: Data type ('half' or 'int8')
        nccl_op: NCCL operation type
        test_range: Size range (min,max,ratio)
        num_gpus: Number of GPUs
        power_monitor: NVMLPowerMonitor instance (optional)
        measure_power: Whether to measure power consumption
    """
    nccl_test_bin = ""
    if nccl_op == "all_gather":
        nccl_test_bin = "all_gather_perf"
    elif nccl_op == "alltoall":
        nccl_test_bin = "alltoall_perf"
    elif nccl_op == "reduce_scatter":
        nccl_test_bin = "reduce_scatter_perf"
    elif nccl_op == "all_reduce":
        nccl_test_bin = "all_reduce_perf"
    assert nccl_test_bin != ""

    min_size, max_size, ratio = [int(i) for i in test_range.split(",")]
    size = min_size

    major, minor, patch = torch.cuda.nccl.version()
    nccl_version = f"{major}.{minor}.{patch}"

    bytes_per_element = 2 if dtype == "half" else 1

    while size < max_size:
        if size <= 1048576:
            inner_loop = 1000
        elif size <= 16777216:
            inner_loop = 500
        else:
            inner_loop = 100

        cmd_args = [
            nccl_test_bin,
            "-b",
            str(size),
            "-e",
            str(size),
            "-t",
            str(num_gpus),
            "-d",
            dtype,
            "-w",
            "40",
            "-a",
            "1",
            "-n",
            str(inner_loop),
            "-c",
            "0",
        ]

        # Power measurement for communication operations
        if measure_power and power_monitor is not None:
            power_monitor.begin_window("nccl", sync_execution=False)
            result = subprocess.run(cmd_args, capture_output=True, text=True)
            measurement = power_monitor.end_window("nccl", sync_execution=False)
            power = sum(measurement.gpu_energy[i] for i in range(num_gpus)) / measurement.time
        else:
            result = subprocess.run(cmd_args, capture_output=True, text=True)
            power = None

        print_lines = result.stdout.split("\n")
        for index_line in range(len(print_lines)):
            if "time" in print_lines[index_line]:
                break
        latency = float(print_lines[index_line + 2].split()[5]) * 1e-3  # us to ms

        # Build result item
        item = {
            "nccl_dtype": dtype,
            "num_gpus": num_gpus,
            "message_size": size // bytes_per_element,
            "latency": latency,
        }

        if power is not None:
            item["power"] = power

        print(nccl_test_bin, f"{size=}, {latency=}, {power=}")
        
        perf_filename = "nccl_perf.txt"
        
        log_perf(
            item_list=[item],
            framework="TRTLLM",
            version=nccl_version,
            device_name=torch.cuda.get_device_name(),
            op_name=nccl_op,
            kernel_source="NCCL",
            perf_filename=perf_filename,
        )

        size *= ratio


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--nccl_op",
        "-NCCL",
        default="all_gather",
        choices=["all_gather", "alltoall", "reduce_scatter", "all_reduce"],
        help="NCCL OP: all_gather, alltoall, reduce_scatter, all_reduce",
    )
    parser.add_argument("--dtype", "-t", default="half", choices=["half", "int8"], help="NCCL OP data type")
    parser.add_argument(
        "--range",
        "-r",
        default="512,536870913,2",  # 512B to 512MB
        help="min_size,max_size,multiplicative_ratio",
    )
    parser.add_argument("--num_gpus", "-n", default=8, type=int)
    parser.add_argument(
        "--measure_power",
        action="store_true",
        default=False,
        help="Enable power measurement during NCCL benchmark",
    )
    args = parser.parse_args()

    power_monitor = None
    if args.measure_power:
        from nvml_power_monitor import NVMLPowerMonitor

        power_monitor = NVMLPowerMonitor(gpu_indices=list(range(args.num_gpus)))

    nccl_benchmark(args.dtype, args.nccl_op, args.range, args.num_gpus, power_monitor, args.measure_power)
