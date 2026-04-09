# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Internode EP dispatch/combine benchmark using MORI (MoRI-EP).

Usage (multi-node):
  # Node 0:
  export MASTER_ADDR=<ip> WORLD_SIZE=2 RANK=0 MASTER_PORT=40303
  python test_internode_mori.py |& tee moriep_node_2_mode_normal.log

  # Node 1:
  export MASTER_ADDR=<ip> WORLD_SIZE=2 RANK=1 MASTER_PORT=40303
  python test_internode_mori.py
"""
import argparse
import os
import time

import mori
import torch
import torch.distributed as dist

from utils import init_dist

os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "6G")

# Kernel-type presets: (EpDispatchCombineKernelType, block_num, rdma_block_num, warp_num_per_block)
_KERNEL_CONFIGS = {
    "internode_v1": (
        mori.ops.EpDispatchCombineKernelType.InterNodeV1,
        96,   # block_num
        64,   # rdma_block_num
        8,    # warp_num_per_block
    ),
    "internode_v1_ll": (
        mori.ops.EpDispatchCombineKernelType.InterNodeV1LL,
        256,  # block_num
        128,  # rdma_block_num
        8,    # warp_num_per_block
    ),
}


def _make_config(
    rank,
    world_size,
    kernel_type_str,
    data_type,
    hidden_dim,
    max_num_inp_token_per_rank,
    num_experts_per_rank,
    num_experts_per_token,
    num_local_ranks,
    scale_dim=0,
    scale_type_size=1,
):
    kernel_type, block_num, rdma_block_num, warp_num_per_block = _KERNEL_CONFIGS[kernel_type_str]
    return mori.ops.EpDispatchCombineConfig(
        data_type=data_type,
        rank=rank,
        world_size=world_size,
        hidden_dim=hidden_dim,
        scale_dim=scale_dim,
        scale_type_size=scale_type_size,
        max_num_inp_token_per_rank=max_num_inp_token_per_rank,
        num_experts_per_rank=num_experts_per_rank,
        num_experts_per_token=num_experts_per_token,
        max_token_type_size=2,
        block_num=block_num,
        rdma_block_num=rdma_block_num,
        warp_num_per_block=warp_num_per_block,
        kernel_type=kernel_type,
        gpu_per_node=num_local_ranks,
        num_qp_per_pe=2,
    )


def _gen_test_data(num_tokens, hidden, num_topk, num_experts, rank, world_size, device):
    """Generate synthetic dispatch/combine test data."""
    total_experts = num_experts
    num_experts_per_rank = num_experts // world_size

    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device=device)
    topk_idx = torch.stack([
        torch.randperm(total_experts, device=device)[:num_topk]
        for _ in range(num_tokens)
    ]).to(torch.int32)
    topk_weights = torch.softmax(
        torch.randn(num_tokens, num_topk, device=device, dtype=torch.float32), dim=-1
    )
    # No FP8 scale for BF16 dispatch
    scale = None

    return x, topk_weights, scale, topk_idx


def _bench_with_cuda_events(fn, num_warmup=10, num_iters=10, graph_replay_iters=10):
    """Benchmark a callable using CUDA graph capture and timed replays.

    Returns average latency in microseconds.
    """
    # Warmup
    for _ in range(num_warmup):
        fn()
    torch.cuda.synchronize()

    # Capture CUDA graph
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()

    # Timed replays
    latencies = []
    for _ in range(num_iters):
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(graph_replay_iters)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(graph_replay_iters)]

        for j in range(graph_replay_iters):
            start_events[j].record()
            graph.replay()
            end_events[j].record()

        torch.cuda.synchronize()
        avg_ms = sum(s.elapsed_time(e) for s, e in zip(start_events, end_events)) / graph_replay_iters
        latencies.append(avg_ms * 1000)  # convert ms -> us

    return min(latencies)


# noinspection PyShadowingNames
def test_main(
    args: argparse.Namespace,
    local_rank: int,
    num_local_ranks: int,
    num_ranks: int,
    num_nodes: int,
    rank: int,
    group: dist.ProcessGroup,
):
    num_tokens, hidden = args.num_tokens, args.hidden
    num_topk, num_experts = args.num_topk, args.num_experts
    num_experts_per_rank = num_experts // num_ranks
    device = torch.device("cuda", local_rank)
    kernel_type_str = "internode_v1_ll" if args.test_ll else "internode_v1"

    assert num_experts % num_ranks == 0 and num_local_ranks == 8
    if local_rank == 0:
        print(
            f"[config] num_tokens={num_tokens}, hidden={hidden}, "
            f"num_topk={num_topk}, num_experts={num_experts}",
            flush=True,
        )

    # Create mori config and op
    config = _make_config(
        rank=rank,
        world_size=num_ranks,
        kernel_type_str=kernel_type_str,
        data_type=torch.bfloat16,
        hidden_dim=hidden,
        max_num_inp_token_per_rank=num_tokens,
        num_experts_per_rank=num_experts_per_rank,
        num_experts_per_token=num_topk,
        num_local_ranks=num_local_ranks,
    )
    op = mori.ops.EpDispatchCombineOp(config)

    # Generate test data
    x, topk_weights, scale, topk_idx = _gen_test_data(
        num_tokens, hidden, num_topk, num_experts, rank, num_ranks, device
    )

    dist.barrier(group=group)
    time.sleep(1)

    # --- Correctness check ---
    dispatch_output, dispatch_weights, dispatch_scale, dispatch_indices, dispatch_recv_num_token = op.dispatch(
        x, topk_weights, scale, topk_idx
    )
    torch.cuda.synchronize()
    dist.barrier(group=group)

    total_recv = dispatch_recv_num_token[0].item() if dispatch_recv_num_token.numel() > 0 else dispatch_output.size(0)

    combine_output, _ = op.combine(dispatch_output, None, dispatch_indices)
    torch.cuda.synchronize()
    dist.barrier(group=group)

    if local_rank == 0:
        print(
            f"[correctness] dispatch recv tokens: {total_recv}, "
            f"combine output shape: {combine_output.shape}",
            flush=True,
        )

    # --- Benchmark dispatch ---
    # Re-run dispatch to get fresh handle for benchmarking
    dispatch_output, dispatch_weights, dispatch_scale, dispatch_indices, dispatch_recv_num_token = op.dispatch(
        x, topk_weights, scale, topk_idx
    )
    torch.cuda.synchronize()
    dist.barrier(group=group)

    total_recv = dispatch_recv_num_token[0].item() if dispatch_recv_num_token.numel() > 0 else dispatch_output.size(0)
    dispatch_bytes = total_recv * hidden * x.element_size()

    dispatch_latency_us = _bench_with_cuda_events(
        lambda: op.dispatch(x, topk_weights, scale, topk_idx)
    )

    # Gather dispatch latency from all ranks
    dispatch_lat_tensor = torch.tensor([dispatch_latency_us], device=device)
    dispatch_lat_list = [torch.zeros(1, device=device) for _ in range(num_ranks)]
    dist.all_gather(dispatch_lat_list, dispatch_lat_tensor, group=group)
    max_dispatch_lat_us = max(t.item() for t in dispatch_lat_list)

    dispatch_bw_gbps = dispatch_bytes / 1e9 / (max_dispatch_lat_us / 1e6) if max_dispatch_lat_us > 0 else 0

    if local_rank == 0:
        print(
            f"[tuning] Best dispatch (BF16): SMs 0, NVL chunk 0, RDMA chunk 0, "
            f"transmit: {max_dispatch_lat_us:.2f} us, notify: 0.00 us, "
            f"BW: {dispatch_bw_gbps:.2f} GB/s (RDMA), {dispatch_bw_gbps:.2f} GB/s (NVL)",
            flush=True,
        )
        print("", flush=True)

    # --- Benchmark FP8 dispatch ---
    try:
        fp8_dtype = torch.float8_e4m3fnuz
        x_fp8 = x.to(fp8_dtype)
        scale_dim = hidden // 128
        fp8_scale = torch.ones((num_tokens, scale_dim), dtype=torch.float32, device=device)

        fp8_config = _make_config(
            rank=rank,
            world_size=num_ranks,
            kernel_type_str=kernel_type_str,
            data_type=fp8_dtype,
            hidden_dim=hidden,
            max_num_inp_token_per_rank=num_tokens,
            num_experts_per_rank=num_experts_per_rank,
            num_experts_per_token=num_topk,
            num_local_ranks=num_local_ranks,
            scale_dim=scale_dim,
            scale_type_size=torch.float32.itemsize,
        )
        fp8_op = mori.ops.EpDispatchCombineOp(fp8_config)

        fp8_dispatch_out, _, _, _, fp8_recv_num = fp8_op.dispatch(
            x_fp8, topk_weights, fp8_scale, topk_idx
        )
        torch.cuda.synchronize()
        dist.barrier(group=group)

        fp8_total_recv = fp8_recv_num[0].item() if fp8_recv_num.numel() > 0 else fp8_dispatch_out.size(0)
        fp8_dispatch_bytes = fp8_total_recv * hidden * x_fp8.element_size()

        fp8_dispatch_latency_us = _bench_with_cuda_events(
            lambda: fp8_op.dispatch(x_fp8, topk_weights, fp8_scale, topk_idx)
        )

        fp8_lat_tensor = torch.tensor([fp8_dispatch_latency_us], device=device)
        fp8_lat_list = [torch.zeros(1, device=device) for _ in range(num_ranks)]
        dist.all_gather(fp8_lat_list, fp8_lat_tensor, group=group)
        max_fp8_lat_us = max(t.item() for t in fp8_lat_list)

        fp8_dispatch_bw = fp8_dispatch_bytes / 1e9 / (max_fp8_lat_us / 1e6) if max_fp8_lat_us > 0 else 0

        if local_rank == 0:
            print(
                f"[tuning] Best dispatch (FP8): SMs 0, NVL chunk 0, RDMA chunk 0, "
                f"transmit: {max_fp8_lat_us:.2f} us, notify: 0.00 us, "
                f"BW: {fp8_dispatch_bw:.2f} GB/s (RDMA), {fp8_dispatch_bw:.2f} GB/s (NVL)",
                flush=True,
            )
            print("", flush=True)
    except Exception as e:
        if local_rank == 0:
            print(f"[info] FP8 dispatch benchmark skipped: {e}", flush=True)

    # --- Benchmark combine ---
    combine_bytes = total_recv * hidden * 2  # BF16
    combine_latency_us = _bench_with_cuda_events(
        lambda: op.combine(dispatch_output, None, dispatch_indices)
    )

    combine_lat_tensor = torch.tensor([combine_latency_us], device=device)
    combine_lat_list = [torch.zeros(1, device=device) for _ in range(num_ranks)]
    dist.all_gather(combine_lat_list, combine_lat_tensor, group=group)
    max_combine_lat_us = max(t.item() for t in combine_lat_list)

    combine_bw_gbps = combine_bytes / 1e9 / (max_combine_lat_us / 1e6) if max_combine_lat_us > 0 else 0

    if local_rank == 0:
        print(
            f"[tuning] Best combine: SMs 0, NVL chunk 0, RDMA chunk 0, "
            f"transmit: {max_combine_lat_us:.2f} us, notify: 0.00 us, "
            f"BW: {combine_bw_gbps:.2f} GB/s (RDMA), {combine_bw_gbps:.2f} GB/s (NVL)",
            flush=True,
        )
        print("", flush=True)


# noinspection PyShadowingNames
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    num_nodes = int(os.getenv("WORLD_SIZE", 1))
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)

    # Initialize mori shmem using the default process group
    mori.shmem.shmem_torch_process_group_init("default")

    torch.manual_seed(rank)

    tokens = [
        1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128,
        160, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072,
    ]

    for num_tokens in tokens:
        args.num_tokens = num_tokens
        test_main(
            args,
            local_rank,
            num_local_ranks,
            num_ranks,
            num_nodes,
            rank,
            group,
        )
        if local_rank == 0:
            print("", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test internode EP kernels (MORI)")
    parser.add_argument("--num-processes", type=int, default=8, help="Number of processes to spawn (default: 8)")
    parser.add_argument("--num-tokens", type=int, default=4096, help="Number of tokens (default: 4096)")
    parser.add_argument("--hidden", type=int, default=7168, help="Hidden dimension size (default: 7168)")
    parser.add_argument("--num-topk", type=int, default=8, help="Number of top-k experts (default: 8)")
    parser.add_argument("--num-experts", type=int, default=256, help="Number of experts (default: 256)")
    parser.add_argument(
        "--test-ll",
        action="store_true",
        help="Use InterNodeV1LL (low-latency) kernel type instead of InterNodeV1",
    )
    args = parser.parse_args()

    num_processes = args.num_processes
    torch.multiprocessing.spawn(test_loop, args=(num_processes, args), nprocs=num_processes)
