# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Collect oneCCL communication performance data for XPU (Intel GPU).

This script uses the oneCCL benchmark binary (compiled from oneCCL examples)
to measure collective communication latencies on Intel XPU devices.
It produces nccl_perf.txt compatible output for use in projection models.

Prerequisites:
  - oneCCL benchmark binary installed at /usr/local/bin/oneccl_benchmark
    (compile from /opt/intel/oneapi/ccl/*/share/doc/ccl/examples/benchmark/)
  - Intel MPI (mpirun) available on PATH
  - Intel GPU (XPU) devices available

Usage:
  python collect_oneccl_xpu.py --oneccl_op all_gather --dtype half --num_gpus 4
  python collect_oneccl_xpu.py --oneccl_op reduce_scatter --dtype half --num_gpus 4
  python collect_oneccl_xpu.py --oneccl_op all_reduce --dtype half --num_gpus 4
  python collect_oneccl_xpu.py --oneccl_op alltoall --dtype half --num_gpus 4
"""

import csv
import os
import subprocess
import tempfile
from argparse import ArgumentParser

from helper import log_perf

# Mapping from our op names to oneCCL benchmark collective names
OP_NAME_TO_CCL = {
    "all_gather": "allgather",
    "alltoall": "alltoall",
    "reduce_scatter": "reduce_scatter",
    "all_reduce": "allreduce",
}

# Mapping from our dtype names to oneCCL benchmark dtype names
DTYPE_TO_CCL = {
    "half": "float16",
    "int8": "int8",
}

BYTES_PER_ELEMENT = {
    "half": 2,
    "int8": 1,
}

BENCHMARK_BIN = "oneccl_benchmark"
BENCHMARK_BIN_FALLBACK = "/usr/local/bin/oneccl_benchmark"


def find_benchmark_binary():
    """Locate the oneCCL benchmark binary."""
    result = subprocess.run(["which", BENCHMARK_BIN], capture_output=True, text=True)
    if result.returncode == 0:
        return BENCHMARK_BIN
    if os.path.exists(BENCHMARK_BIN_FALLBACK):
        return BENCHMARK_BIN_FALLBACK
    raise FileNotFoundError(
        "oneCCL benchmark binary not found. "
        "Compile from /opt/intel/oneapi/ccl/*/share/doc/ccl/examples/benchmark/ "
        "and install to /usr/local/bin/oneccl_benchmark"
    )


def get_oneccl_version():
    """Get installed oneCCL version string."""
    try:
        result = subprocess.run(
            ["dpkg-query", "-W", "-f", "${Version}", "intel-oneapi-ccl-devel"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown_version"


def get_device_name():
    """Get the Intel GPU device name via sycl-ls or fallback.

    Parses sycl-ls output like:
      [level_zero:gpu][level_zero:0] ..., Intel(R) Graphics [0xe211] 20.1.0 [...]
    Extracts: "Intel(R) Graphics [0xe211]"
    """
    try:
        result = subprocess.run(["sycl-ls"], capture_output=True, text=True)
        for line in result.stdout.split("\n"):
            if "level_zero:gpu" in line and "Intel" in line:
                # Format: ..., Intel(R) Graphics [0xHEXID] VERSION [DRIVER]
                # Extract the part after the last comma, before the version number
                parts = line.split(",")
                if len(parts) >= 2:
                    device_part = parts[-1].strip()
                    # Find "Intel..." up to the PCI ID bracket
                    # e.g. "Intel(R) Graphics [0xe211] 20.1.0 [1.6.33578+15]"
                    # We want "Intel(R) Graphics [0xe211]"
                    import re

                    match = re.search(r"(Intel\S*\s+Graphics\s+\[0x[0-9a-fA-F]+\])", device_part)
                    if match:
                        return match.group(1)
                    # Fallback: return everything before the driver version bracket
                    match = re.search(r"(Intel.*?)\s+\d+\.\d+", device_part)
                    if match:
                        return match.group(1).strip()
    except Exception:
        pass
    return "Intel GPU"


def generate_elem_counts(test_range, bytes_per_element):
    """Generate element counts from byte-based test range.

    Args:
        test_range: "min_bytes,max_bytes,ratio" string
        bytes_per_element: bytes per data element

    Returns:
        list of element counts
    """
    try:
        min_size, max_size, ratio = [int(i) for i in test_range.split(",")]
    except ValueError as exc:
        raise ValueError("--range must be 'min_bytes,max_bytes,multiplicative_ratio' with integer values") from exc

    if min_size <= 0 or max_size <= min_size:
        raise ValueError("--range must satisfy 0 < min_bytes < max_bytes")
    if ratio <= 1:
        raise ValueError("--range multiplicative_ratio must be > 1")

    counts = []
    size = min_size
    while size < max_size:
        elem_count = max(1, size // bytes_per_element)
        counts.append(elem_count)
        size *= ratio
    return counts


def oneccl_benchmark(
    dtype: str,
    oneccl_op: str = "all_gather",
    test_range: str = "512,536870913,2",
    num_gpus: int = 2,
    iters: int = 100,
    warmup_iters: int = 20,
):
    """Run oneCCL benchmark and log results.

    Runs the compiled oneCCL benchmark binary via mpirun to measure
    collective communication latency on Intel XPU devices.
    """
    benchmark_bin = find_benchmark_binary()
    ccl_coll = OP_NAME_TO_CCL[oneccl_op]
    ccl_dtype = DTYPE_TO_CCL[dtype]
    bytes_per_elem = BYTES_PER_ELEMENT[dtype]

    version = get_oneccl_version()
    device_name = get_device_name()

    # Generate element counts from byte-range specification
    elem_counts = generate_elem_counts(test_range, bytes_per_elem)
    if not elem_counts:
        print("No element counts generated from range.")
        return

    print(f"Running oneCCL benchmark: op={oneccl_op}, dtype={dtype}, num_gpus={num_gpus}, {len(elem_counts)} sizes")

    # Run in batches to avoid too-long command lines
    batch_size = 50
    for batch_start in range(0, len(elem_counts), batch_size):
        batch = elem_counts[batch_start : batch_start + batch_size]
        elem_counts_str = ",".join(str(c) for c in batch)

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            csv_path = f.name

        try:
            env = os.environ.copy()
            env["CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK"] = "0"
            env["FI_PROVIDER"] = "tcp"
            env["I_MPI_OFI_PROVIDER"] = "tcp"

            cmd = [
                "mpirun",
                "-n",
                str(num_gpus),
                benchmark_bin,
                "--backend",
                "sycl",
                "--coll",
                ccl_coll,
                "--dtype",
                ccl_dtype,
                "--elem_counts",
                elem_counts_str,
                "--iters",
                str(iters),
                "--warmup_iters",
                str(warmup_iters),
                "--csv_filepath",
                csv_path,
            ]

            print(
                f"  Batch {batch_start // batch_size + 1}: "
                f"sizes {batch[0] * bytes_per_elem}-{batch[-1] * bytes_per_elem} bytes"
            )

            try:
                result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
            except subprocess.TimeoutExpired:
                print(f"  Timeout: batch did not complete within 300s (op={oneccl_op} may not be supported). Skipping.")
                continue

            if result.returncode != 0:
                print(f"  Error (exit {result.returncode}):")
                print(f"  stderr: {result.stderr[:500]}")
                continue

            # Parse CSV output from oneCCL benchmark
            if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
                print("  Warning: no CSV output produced")
                continue

            with open(csv_path, encoding="utf-8") as csvf:
                reader = csv.DictReader(csvf)
                items = []
                for row in reader:
                    latency_us = float(row["t_avg[usec]"])
                    latency_ms = latency_us * 1e-3
                    msg_bytes = int(row["message_size"])
                    msg_elements = msg_bytes // bytes_per_elem

                    print(f"    {ccl_coll}: {msg_bytes} bytes ({msg_elements} elements), latency={latency_ms:.6f} ms")

                    items.append(
                        {
                            "nccl_dtype": dtype,
                            "num_gpus": num_gpus,
                            "message_size": msg_elements,
                            "latency": latency_ms,
                        }
                    )

                if items:
                    log_perf(
                        item_list=items,
                        framework="VLLM",
                        version=version,
                        device_name=device_name,
                        op_name=oneccl_op,
                        kernel_source="oneCCL",
                        perf_filename="oneccl_perf.txt",
                    )
        finally:
            if os.path.exists(csv_path):
                os.unlink(csv_path)

    print("Done. Results appended to oneccl_perf.txt")


if __name__ == "__main__":
    parser = ArgumentParser(description="Collect oneCCL communication performance data for XPU")
    parser.add_argument(
        "--oneccl_op",
        "-O",
        default="all_gather",
        choices=["all_gather", "alltoall", "reduce_scatter", "all_reduce"],
        help="oneCCL operation to benchmark",
    )
    parser.add_argument(
        "--dtype",
        "-t",
        default="half",
        choices=["half", "int8"],
        help="Data type for the collective operation",
    )
    parser.add_argument(
        "--range",
        "-r",
        default="512,536870913,2",  # 512B to 512MB, multiply by 2
        help="min_bytes,max_bytes,multiplicative_ratio",
    )
    parser.add_argument("--num_gpus", "-n", default=2, type=int, help="Number of GPUs (MPI ranks)")
    parser.add_argument("--iters", "-i", default=100, type=int, help="Benchmark iterations per size")
    parser.add_argument("--warmup_iters", "-w", default=20, type=int, help="Warmup iterations per size")
    args = parser.parse_args()

    oneccl_benchmark(
        dtype=args.dtype,
        oneccl_op=args.oneccl_op,
        test_range=args.range,
        num_gpus=args.num_gpus,
        iters=args.iters,
        warmup_iters=args.warmup_iters,
    )
