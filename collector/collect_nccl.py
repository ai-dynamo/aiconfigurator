# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
from argparse import ArgumentParser

import torch

from helper import PowerMonitor, log_perf


def nccl_benchmark(
    dtype: str,
    nccl_op: str = "all_gather",
    test_range: str = "10,10000000,1000",
    num_gpus: int = 8,
    measure_power: bool = False,
):
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

    # Initialize power monitoring if enabled
    power_monitor = None
    if measure_power:
        # Use GPU 0 for power monitoring (representative in multi-GPU scenarios)
        power_monitor = PowerMonitor(device_id=0)
        if not power_monitor._init_handle():
            print("Warning: Failed to initialize power monitoring, continuing without power measurement")
            power_monitor = None

    while size < max_size:
        inner_loop = 100 if size <= 16777216 else 60
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

        # Start power monitoring before benchmark
        power_stats = None
        if power_monitor:
            power_monitor.start_sampling()

        result = subprocess.run(cmd_args, capture_output=True, text=True)

        # Stop power monitoring after benchmark
        if power_monitor:
            power_stats = power_monitor.stop_sampling()

        print_lines = result.stdout.split("\n")
        for index_line in range(len(print_lines)):
            if "time" in print_lines[index_line]:
                break
        latency = float(print_lines[index_line + 2].split()[5]) * 1e-3  # us to ms

        print(nccl_test_bin, f"{size=}, {latency=}")
        if power_stats:
            print(f"  Power: {power_stats['power']:.2f}W (limit: {power_stats['power_limit']:.2f}W)")

        log_perf(
            item_list=[
                {
                    "nccl_dtype": dtype,
                    "num_gpus": num_gpus,
                    "message_size": size // bytes_per_element,
                    "latency": latency,
                }
            ],
            framework="TRTLLM",
            version=nccl_version,
            device_name=torch.cuda.get_device_name(),
            op_name=nccl_op,
            kernel_source="NCCL",
            perf_filename="nccl_perf.txt",
            power_stats=power_stats,
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
        help="Enable power monitoring during NCCL benchmark execution (samples at 100ms intervals)",
    )
    parser.add_argument(
        "--power_test_duration_sec",
        type=float,
        default=1.0,
        help="Minimum duration for benchmark runs when power measurement is enabled (default: 1.0s). "
        "Note: NCCL tests already run long enough, so this is informational only.",
    )
    args = parser.parse_args()

    nccl_benchmark(args.dtype, args.nccl_op, args.range, args.num_gpus, args.measure_power)
