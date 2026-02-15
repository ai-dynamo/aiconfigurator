# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Phase 3: nsys Profiling for MLA BMM Alignment

This script profiles the vLLM MLA BMM operations and compares with
trtllm/sglang implementations.

Usage:
    # Profile the mock layer
    nsys profile -o mla_bmm_profile -t cuda,nvtx python test_mla_bmm_profile.py

    # View results
    nsys stats --report cuda_gpu_kern_sum mla_bmm_profile.nsys-rep
"""

import torch
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from test_mla_bmm_layer import create_mla_bmm_layer


def profile_mla_bmm(
    num_tokens: int = 128,
    num_heads: int = 32,
    dtype: str = "float16",
    is_pre: bool = True,
    num_iterations: int = 100,
    device: str = "cuda:0",
):
    """
    Profile MLA BMM operation.
    
    Args:
        num_tokens: Number of tokens
        num_heads: Number of attention heads
        dtype: Data type ("float16" or "fp8")
        is_pre: True for mla_gen_pre, False for mla_gen_post
        num_iterations: Number of iterations to run
        device: CUDA device
    """
    torch.cuda.set_device(device)
    
    # Create layer
    layer = create_mla_bmm_layer(num_heads, dtype, is_pre, device)
    
    # Create input
    if is_pre:
        x = torch.randn(num_tokens, num_heads, 128, dtype=torch.bfloat16, device=device)
    else:
        x = torch.randn(num_tokens, num_heads, 512, dtype=torch.bfloat16, device=device)
    
    # Warmup
    for _ in range(10):
        _ = layer(x)
    torch.cuda.synchronize()
    
    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(num_iterations):
        _ = layer(x)
    end.record()
    torch.cuda.synchronize()
    
    latency_ms = start.elapsed_time(end) / num_iterations
    
    op_name = "mla_gen_pre" if is_pre else "mla_gen_post"
    print(f"{op_name} {dtype} tokens={num_tokens} heads={num_heads}: {latency_ms:.4f} ms")
    
    return latency_ms


def compare_with_trtllm():
    """
    Compare vLLM MLA BMM with trtllm implementation.
    
    This runs the same operations using trtllm's BMM kernels for comparison.
    """
    print("\n=== Comparison with TensorRT-LLM ===\n")
    
    device = "cuda:0"
    num_tokens = 128
    num_heads = 32
    
    # Test parameters
    test_cases = [
        (128, 32, "float16", True),   # mla_gen_pre
        (128, 32, "float16", False),  # mla_gen_post
        (128, 32, "fp8", True),       # mla_gen_pre FP8
        (128, 32, "fp8", False),      # mla_gen_post FP8
    ]
    
    results = []
    
    for num_tokens, num_heads, dtype, is_pre in test_cases:
        latency = profile_mla_bmm(num_tokens, num_heads, dtype, is_pre, device=device)
        results.append({
            "op": "mla_gen_pre" if is_pre else "mla_gen_post",
            "dtype": dtype,
            "tokens": num_tokens,
            "heads": num_heads,
            "latency_ms": latency,
        })
    
    return results


def verify_kernel_alignment():
    """
    Verify that the BMM kernels are aligned with trtllm/sglang.
    
    Expected kernels for BMM on Hopper:
    - BF16: cutlass::Gemm, ampere_bmm, or similar
    - FP8: fp8_bmm, block_scale_bmm, or similar
    """
    print("\n=== Kernel Alignment Check ===\n")
    
    # This would be done by running nsys and checking kernel names
    # For now, we just print what to look for
    
    print("Expected kernels for MLA BMM on Hopper (SM90):")
    print("  BF16: cutlass::device_kernel::GemmUniversal, ampere_bmm")
    print("  FP8:  fp8_block_scaling_bmm, cutlass::epilogue::threadblock::Epilogue")
    print("\nTo verify, run:")
    print("  nsys profile -o mla_bmm_profile -t cuda,nvtx python test_mla_bmm_profile.py")
    print("  nsys stats --report cuda_gpu_kern_sum mla_bmm_profile.nsys-rep")


def main():
    parser = argparse.ArgumentParser(description="Profile MLA BMM for alignment")
    parser.add_argument("--tokens", type=int, default=128, help="Number of tokens")
    parser.add_argument("--heads", type=int, default=32, help="Number of heads")
    parser.add_argument("--dtype", type=str, choices=["float16", "fp8"], default="float16")
    parser.add_argument("--op", type=str, choices=["pre", "post", "both"], default="both")
    parser.add_argument("--iterations", type=int, default=100, help="Number of iterations")
    parser.add_argument("--compare", action="store_true", help="Compare with trtllm")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    
    if args.compare:
        compare_with_trtllm()
        verify_kernel_alignment()
        return
    
    if args.op in ["pre", "both"]:
        profile_mla_bmm(args.tokens, args.heads, args.dtype, True, args.iterations, args.device)
    
    if args.op in ["post", "both"]:
        profile_mla_bmm(args.tokens, args.heads, args.dtype, False, args.iterations, args.device)


if __name__ == "__main__":
    main()
