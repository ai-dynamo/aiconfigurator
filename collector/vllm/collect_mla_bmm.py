# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
vLLM MLA BMM Performance Collector for Hopper (SM90).

Collects performance data for MLA (Multi-Head Latent Attention) BMM operations
with BF16 and FP8 precision.

MLA BMM Operations:
- mla_gen_pre: Q_nope @ K_b_proj^T -> latent space
  Shape: (num_tokens, num_heads, 128) @ (num_heads, 512, 128)^T -> (num_tokens, num_heads, 512)
- mla_gen_post: attn_output @ V_b_proj^T -> output
  Shape: (num_tokens, num_heads, 512) @ (num_heads, 128, 512)^T -> (num_tokens, num_heads, 128)

Usage:
    python collect_mla_bmm.py --device cuda:0

Output:
    mla_bmm_perf.txt - CSV file with MLA BMM performance data
"""

import argparse
import os
from typing import List, Tuple

import torch
from vllm.version import __version__ as vllm_version

# Add parent directory for helper import
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from collector.helper import benchmark_with_power, get_sm_version, log_perf


# MLA fixed dimensions (DeepSeek V3/R1 style)
QK_NOPE_HEAD_DIM = 128
KV_LORA_RANK = 512
V_HEAD_DIM = 128


def get_mla_gen_pre_test_cases() -> List[Tuple[int, int, str, int, int, str]]:
    """
    Generate test cases for mla_gen_pre BMM.

    Returns:
        List of (num_tokens, num_heads, dtype, num_warmups, num_runs, perf_filename) tuples
    """
    test_cases = []

    # Generation token counts (batch sizes for decode)
    gen_num_tokens = [
        1, 2, 4, 8, 16, 32, 48, 64, 80, 96,
        128, 160, 192, 256, 320, 384, 512, 768,
        1024, 1536, 2048, 3072, 4096, 6144, 8192,
    ]

    # Number of attention heads (after TP split)
    num_heads = [128, 64, 32, 16, 8, 4, 2, 1]

    # Precision modes
    dtype_list = ["float16"]
    sm = get_sm_version()
    if sm >= 90:
        # FP8 requires Hopper (SM90) or later
        dtype_list += ["fp8"]

    for num_tokens in gen_num_tokens:
        for num_head in num_heads:
            for dtype in dtype_list:
                test_cases.append([num_tokens, num_head, dtype, 2, 10, "mla_bmm_perf.txt"])

    return test_cases


def get_mla_gen_post_test_cases() -> List[Tuple[int, int, str, int, int, str]]:
    """
    Generate test cases for mla_gen_post BMM.

    Returns:
        List of (num_tokens, num_heads, dtype, num_warmups, num_runs, perf_filename) tuples
    """
    test_cases = []

    # Context token counts (can be larger)
    ctx_num_tokens = [
        1, 2, 4, 8, 16, 32, 48, 64, 80, 96,
        128, 160, 192, 256, 320, 384, 512, 768,
        1024, 1536, 2048, 3072, 4096, 6144, 8192,
        12288, 16384, 20480,
    ]

    num_heads = [128, 64, 32, 16, 8, 4, 2, 1]

    dtype_list = ["float16"]
    sm = get_sm_version()
    if sm >= 90:
        dtype_list += ["fp8"]

    for num_tokens in ctx_num_tokens:
        for num_head in num_heads:
            for dtype in dtype_list:
                test_cases.append([num_tokens, num_head, dtype, 2, 10, "mla_bmm_perf.txt"])

    return test_cases


def run_mla_gen_pre(
    num_tokens: int,
    num_heads: int,
    dtype: str,
    num_warmups: int,
    num_runs: int,
    perf_filename: str,
    device: str = "cuda:0",
) -> None:
    """
    Run mla_gen_pre BMM benchmark.

    Operation: Q_nope @ K_b_proj^T
    Shape: (num_tokens, num_heads, 128) @ (num_heads, 512, 128)^T -> (num_tokens, num_heads, 512)

    This is the first BMM in MLA generation, projecting query to latent space.
    """
    torch.cuda.set_device(device)
    torch.set_default_device(device)

    if dtype == "float16":
        # BF16 path
        q_nope = torch.randn(
            [num_tokens, num_heads, QK_NOPE_HEAD_DIM],
            dtype=torch.bfloat16,
            device=device,
        )
        k_b_proj = torch.randn(
            [num_heads, KV_LORA_RANK, QK_NOPE_HEAD_DIM],
            dtype=torch.bfloat16,
            device=device,
        )
        # Transpose for BMM: (num_heads, 128, 512)
        k_b_proj_t = k_b_proj.transpose(1, 2)

        def kernel_func():
            # Q_nope: (num_tokens, num_heads, 128) -> (num_heads, num_tokens, 128)
            # K_b_proj^T: (num_heads, 128, 512)
            # Output: (num_heads, num_tokens, 512)
            torch.bmm(q_nope.transpose(0, 1), k_b_proj_t)

    elif dtype == "fp8":
        # FP8 path for Hopper
        q_nope = torch.randn(
            [num_tokens, num_heads, QK_NOPE_HEAD_DIM],
            dtype=torch.bfloat16,
            device=device,
        )
        # Quantize weight to FP8
        k_b_proj = torch.randn(
            [num_heads, KV_LORA_RANK, QK_NOPE_HEAD_DIM],
            dtype=torch.bfloat16,
            device=device,
        ).to(torch.float8_e4m3fn)
        k_b_proj_t = k_b_proj.transpose(1, 2)

        # For FP8 BMM, we need to quantize activations on-the-fly
        # vLLM uses per-tensor or block-wise quantization
        def kernel_func():
            # Quantize activation to FP8
            q_nope_fp8 = q_nope.to(torch.float8_e4m3fn)
            # BMM in FP32 then convert back
            # Note: Real vLLM kernel would use FP8 TC
            q_nope_t = q_nope_fp8.transpose(0, 1).float()
            torch.bmm(q_nope_t, k_b_proj_t.float())
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    # Warmup (already done in benchmark_with_power, but we need to init tensors)
    _ = kernel_func()
    torch.cuda.synchronize()

    # Benchmark with power measurement
    with benchmark_with_power(
        device=device,
        kernel_func=kernel_func,
        num_warmups=num_warmups,
        num_runs=num_runs,
        repeat_n=1,
    ) as results:
        pass

    # Log performance data
    log_perf(
        item_list=[
            {
                "bmm_dtype": dtype,
                "num_tokens": num_tokens,
                "num_heads": num_heads,
                "latency": results["latency_ms"],
            }
        ],
        framework="VLLM",
        version=vllm_version,
        device_name=torch.cuda.get_device_name(device),
        op_name="mla_gen_pre",
        kernel_source="default",
        perf_filename=perf_filename,
        power_stats=results.get("power_stats"),
    )

    print(f"mla_gen_pre {dtype} tokens={num_tokens} heads={num_heads}: {results['latency_ms']:.4f} ms")


def run_mla_gen_post(
    num_tokens: int,
    num_heads: int,
    dtype: str,
    num_warmups: int,
    num_runs: int,
    perf_filename: str,
    device: str = "cuda:0",
) -> None:
    """
    Run mla_gen_post BMM benchmark.

    Operation: attn_output @ V_b_proj^T
    Shape: (num_tokens, num_heads, 512) @ (num_heads, 128, 512)^T -> (num_tokens, num_heads, 128)

    This is the second BMM in MLA generation, projecting from latent space to output.
    """
    torch.cuda.set_device(device)
    torch.set_default_device(device)

    if dtype == "float16":
        # BF16 path
        attn_output = torch.randn(
            [num_tokens, num_heads, KV_LORA_RANK],
            dtype=torch.bfloat16,
            device=device,
        )
        v_b_proj = torch.randn(
            [num_heads, V_HEAD_DIM, KV_LORA_RANK],
            dtype=torch.bfloat16,
            device=device,
        )
        # Transpose for BMM: (num_heads, 512, 128)
        v_b_proj_t = v_b_proj.transpose(1, 2)

        def kernel_func():
            # attn_output: (num_tokens, num_heads, 512) -> (num_heads, num_tokens, 512)
            # V_b_proj^T: (num_heads, 512, 128)
            # Output: (num_heads, num_tokens, 128)
            torch.bmm(attn_output.transpose(0, 1), v_b_proj_t)

    elif dtype == "fp8":
        # FP8 path for Hopper
        attn_output = torch.randn(
            [num_tokens, num_heads, KV_LORA_RANK],
            dtype=torch.bfloat16,
            device=device,
        )
        v_b_proj = torch.randn(
            [num_heads, V_HEAD_DIM, KV_LORA_RANK],
            dtype=torch.bfloat16,
            device=device,
        ).to(torch.float8_e4m3fn)
        v_b_proj_t = v_b_proj.transpose(1, 2)

        def kernel_func():
            attn_output_fp8 = attn_output.to(torch.float8_e4m3fn)
            attn_output_t = attn_output_fp8.transpose(0, 1).float()
            torch.bmm(attn_output_t, v_b_proj_t.float())
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    # Warmup
    _ = kernel_func()
    torch.cuda.synchronize()

    # Benchmark with power measurement
    with benchmark_with_power(
        device=device,
        kernel_func=kernel_func,
        num_warmups=num_warmups,
        num_runs=num_runs,
        repeat_n=1,
    ) as results:
        pass

    # Log performance data
    log_perf(
        item_list=[
            {
                "bmm_dtype": dtype,
                "num_tokens": num_tokens,
                "num_heads": num_heads,
                "latency": results["latency_ms"],
            }
        ],
        framework="VLLM",
        version=vllm_version,
        device_name=torch.cuda.get_device_name(device),
        op_name="mla_gen_post",
        kernel_source="default",
        perf_filename=perf_filename,
        power_stats=results.get("power_stats"),
    )

    print(f"mla_gen_post {dtype} tokens={num_tokens} heads={num_heads}: {results['latency_ms']:.4f} ms")


def main():
    parser = argparse.ArgumentParser(description="vLLM MLA BMM Collector for Hopper")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device")
    parser.add_argument("--output_dir", type=str, default=".", help="Output directory")
    parser.add_argument("--num_warmups", type=int, default=2, help="Number of warmup iterations")
    parser.add_argument("--num_runs", type=int, default=10, help="Number of benchmark iterations")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of test cases (for quick testing)",
    )
    parser.add_argument(
        "--op",
        type=str,
        choices=["pre", "post", "both"],
        default="both",
        help="Which operation to benchmark",
    )
    args = parser.parse_args()

    # Check SM version for FP8 support
    sm = get_sm_version()
    print(f"Detected SM version: {sm}")

    if sm < 90:
        print("Warning: FP8 requires SM90+ (Hopper). Only BF16 will be benchmarked.")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    perf_filename = os.path.join(args.output_dir, "mla_bmm_perf.txt")

    # Run benchmarks
    if args.op in ["pre", "both"]:
        test_cases = get_mla_gen_pre_test_cases()
        if args.limit:
            test_cases = test_cases[: args.limit]
        print(f"\nRunning {len(test_cases)} mla_gen_pre test cases...")
        for i, test_case in enumerate(test_cases):
            print(f"[{i + 1}/{len(test_cases)}] ", end="")
            test_case[-1] = perf_filename  # Update output path
            run_mla_gen_pre(*test_case, device=args.device)

    if args.op in ["post", "both"]:
        test_cases = get_mla_gen_post_test_cases()
        if args.limit:
            test_cases = test_cases[: args.limit]
        print(f"\nRunning {len(test_cases)} mla_gen_post test cases...")
        for i, test_case in enumerate(test_cases):
            print(f"[{i + 1}/{len(test_cases)}] ", end="")
            test_case[-1] = perf_filename  # Update output path
            run_mla_gen_post(*test_case, device=args.device)

    print(f"\nResults saved to {perf_filename}")


if __name__ == "__main__":
    main()
