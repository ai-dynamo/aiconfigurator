#!/usr/bin/env python3
"""Parse nsys SQLite export to extract allreduce kernel durations per message size.

Usage:
    # Step 1: Run collector under nsys
    nsys profile -o allreduce_bench --capture-range=none \
        mpirun -np 4 --allow-run-as-root \
        python collect_all_reduce.py --backend trtllm --use-nsys

    # Step 2: Export to SQLite
    nsys export --type=sqlite allreduce_bench.nsys-rep

    # Step 3: Parse and generate perf file
    python parse_nsys_allreduce.py allreduce_bench.sqlite \
        --perf-filename custom_allreduce_perf.txt \
        --framework TRTLLM --version 1.3.0rc12 --device "NVIDIA GB300"
"""

import argparse
import csv
import os
import sqlite3
import sys


def parse_nsys_sqlite(sqlite_path):
    """Extract allreduce kernel durations grouped by NVTX message size markers.

    Returns:
        dict: {message_size: [duration_ms, ...]}
    """
    conn = sqlite3.connect(sqlite_path)

    # Get NVTX ranges with "allreduce_size_" prefix
    # NVTX push/pop ranges are in the NVTX_EVENTS table
    nvtx_query = """
        SELECT start, end, text
        FROM NVTX_EVENTS
        WHERE text LIKE 'allreduce_size_%'
        ORDER BY start
    """
    try:
        nvtx_ranges = conn.execute(nvtx_query).fetchall()
    except sqlite3.OperationalError:
        # Try alternative table name
        nvtx_query = """
            SELECT start, end, text
            FROM StringIds s
            JOIN NVTX_EVENTS e ON e.textId = s.id
            WHERE s.value LIKE 'allreduce_size_%'
            ORDER BY start
        """
        try:
            nvtx_ranges = conn.execute(nvtx_query).fetchall()
        except sqlite3.OperationalError:
            print("Could not find NVTX events. Available tables:")
            tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            for t in tables:
                print(f"  {t[0]}")
            conn.close()
            sys.exit(1)

    if not nvtx_ranges:
        # Try joining with StringIds table (common nsys format)
        nvtx_query = """
            SELECT e.start, e.end, s.value as text
            FROM NVTX_EVENTS e
            JOIN StringIds s ON e.textId = s.id
            WHERE s.value LIKE 'allreduce_size_%'
            ORDER BY e.start
        """
        nvtx_ranges = conn.execute(nvtx_query).fetchall()

    if not nvtx_ranges:
        print("No NVTX ranges found with 'allreduce_size_' prefix.")
        print("Make sure the collector was run with --use-nsys flag.")
        conn.close()
        sys.exit(1)

    print(f"Found {len(nvtx_ranges)} NVTX size ranges")

    # Get all GPU kernel executions that match allreduce
    kernel_query = """
        SELECT start, end, duration,
               demangledName, shortName
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        WHERE demangledName LIKE '%allreduce%'
           OR shortName LIKE '%allreduce%'
        ORDER BY start
    """
    try:
        kernels = conn.execute(kernel_query).fetchall()
    except sqlite3.OperationalError:
        # Try alternative column names
        kernel_query = """
            SELECT start, end, (end - start) as duration,
                   demangledName
            FROM CUPTI_ACTIVITY_KIND_KERNEL
            WHERE demangledName LIKE '%allreduce%'
            ORDER BY start
        """
        kernels = conn.execute(kernel_query).fetchall()

    conn.close()

    if not kernels:
        print("No allreduce kernels found in the trace.")
        sys.exit(1)

    print(f"Found {len(kernels)} allreduce kernel instances")

    # Match kernels to NVTX ranges
    results = {}
    for nvtx_start, nvtx_end, text in nvtx_ranges:
        size = int(text.replace("allreduce_size_", ""))
        durations = []
        for k in kernels:
            k_start, k_end = k[0], k[1]
            k_duration_ns = k[2]
            # Kernel falls within NVTX range
            if k_start >= nvtx_start and k_end <= nvtx_end:
                durations.append(k_duration_ns / 1e6)  # ns to ms
        results[size] = durations

    return results


def write_perf_file(results, perf_filename, framework, version, device, dtype, num_gpus, implementation):
    """Write results in the same CSV format as the collector."""
    file_exists = os.path.exists(perf_filename)

    with open(perf_filename, "a", newline="") as f:
        fieldnames = [
            "framework",
            "version",
            "device",
            "op_name",
            "kernel_source",
            "allreduce_dtype",
            "num_gpus",
            "message_size",
            "latency",
            "implementation",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for size in sorted(results.keys()):
            durations = results[size]
            if not durations:
                print(f"  Size {size}: no kernels found, skipping")
                continue
            avg_latency = sum(durations) / len(durations)
            median_latency = sorted(durations)[len(durations) // 2]
            print(f"  Size {size}: {len(durations)} kernels, avg={avg_latency:.4f}ms, median={median_latency:.4f}ms")
            writer.writerow(
                {
                    "framework": framework,
                    "version": version,
                    "device": device,
                    "op_name": "all_reduce",
                    "kernel_source": framework,
                    "allreduce_dtype": dtype,
                    "num_gpus": num_gpus,
                    "message_size": size,
                    "latency": median_latency,
                    "implementation": implementation,
                }
            )

    print(f"\nResults written to {perf_filename}")


def main():
    parser = argparse.ArgumentParser(description="Parse nsys SQLite export for allreduce kernel durations")
    parser.add_argument("sqlite_path", help="Path to nsys SQLite export (.sqlite)")
    parser.add_argument(
        "--perf-filename", "-f", default="custom_allreduce_perf_nsys.txt", help="Output performance file"
    )
    parser.add_argument("--framework", default="TRTLLM")
    parser.add_argument("--version", default="unknown")
    parser.add_argument("--device", default="unknown")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--implementation", default="trtllm")

    args = parser.parse_args()

    if not os.path.exists(args.sqlite_path):
        print(f"Error: {args.sqlite_path} not found")
        sys.exit(1)

    print(f"Parsing {args.sqlite_path}...")
    results = parse_nsys_sqlite(args.sqlite_path)

    write_perf_file(
        results,
        args.perf_filename,
        args.framework,
        args.version,
        args.device,
        args.dtype,
        args.num_gpus,
        args.implementation,
    )


if __name__ == "__main__":
    main()
