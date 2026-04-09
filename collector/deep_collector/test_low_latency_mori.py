# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Low-latency EP dispatch/combine benchmark using MORI (MoRI-EP AsyncLL).

Usage (single node):
  python test_low_latency_mori.py |& tee moriep_node_1_mode_ll.log

Usage (multi-node):
  # Node 0:
  export MASTER_ADDR=<ip> WORLD_SIZE=2 RANK=0 MASTER_PORT=40303
  python test_low_latency_mori.py |& tee moriep_node_2_mode_ll.log

  # Node 1:
  export MASTER_ADDR=<ip> WORLD_SIZE=2 RANK=1 MASTER_PORT=40303
  python test_low_latency_mori.py
"""
import argparse
import os
import random
import time

import mori
import torch
import torch.distributed as dist

from utils import bench, calc_diff, init_dist

os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "6G")


def _make_config(
    rank,
    world_size,
    data_type,
    hidden_dim,
    max_num_inp_token_per_rank,
    num_experts_per_rank,
    num_experts_per_token,
    scale_dim=0,
    scale_type_size=1,
):
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
        block_num=64,
        warp_num_per_block=8,
        kernel_type=mori.ops.EpDispatchCombineKernelType.AsyncLL,
    )


def test_main(
    num_tokens: int,
    hidden: int,
    num_experts: int,
    num_topk: int,
    rank: int,
    num_ranks: int,
    group: dist.ProcessGroup,
    op: mori.ops.EpDispatchCombineOp,
    seed: int = 0,
):
    torch.manual_seed(seed + rank)
    random.seed(seed + rank)
    device = torch.device("cuda", rank % 8)

    assert num_experts % num_ranks == 0
    num_local_experts = num_experts // num_ranks

    # Generate test data
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device=device)
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device=device).abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=True)[1].to(torch.int32)
    topk_weights = torch.randn((num_tokens, num_topk), dtype=torch.float32, device=device).abs()

    # Randomly mask some positions
    for _ in range(min(10, num_tokens)):
        topk_idx[random.randint(0, num_tokens - 1), random.randint(0, num_topk - 1)] = -1

    scale = None

    # --- Correctness check: dispatch_send + dispatch_recv ---
    dispatch_output, dispatch_weights, dispatch_scale, dispatch_indices, dispatch_recv_num_token = op.dispatch_send(
        x, topk_weights, scale, topk_idx
    )
    op.dispatch_recv()
    torch.cuda.synchronize()
    dist.barrier(group=group)

    total_recv = dispatch_recv_num_token[0].item() if dispatch_recv_num_token.numel() > 0 else dispatch_output.size(0)

    # --- Correctness check: combine_send + combine_recv ---
    combine_output, _ = op.combine_send(dispatch_output, None, topk_idx)
    op.combine_recv()
    torch.cuda.synchronize()
    dist.barrier(group=group)

    if rank == 0:
        print(
            f"[correctness] dispatch recv tokens: {total_recv}, "
            f"combine output shape: {combine_output.shape}",
            flush=True,
        )

    # --- Bandwidth calculation ---
    # FP8 dispatch: hidden + hidden/128*4 (scales) + 16 (metadata) per selection
    # BF16 combine: hidden*2 per selection
    num_fp8_bytes = hidden + hidden // 128 * 4 + 16
    num_bf16_bytes = hidden * 2
    num_dispatch_comm_bytes, num_combine_comm_bytes = 0, 0
    for i in range(num_tokens):
        num_selections = (topk_idx[i] != -1).sum().item()
        num_dispatch_comm_bytes += num_fp8_bytes * num_selections
        num_combine_comm_bytes += num_bf16_bytes * num_selections

    # --- Benchmark: dispatch + combine e2e ---
    def run_dispatch_combine():
        d_out, d_w, d_s, d_idx, d_recv = op.dispatch_send(x, topk_weights, scale, topk_idx)
        op.dispatch_recv()
        c_out, _ = op.combine_send(d_out, None, topk_idx)
        op.combine_recv()

    avg_t, min_t, max_t = bench(run_dispatch_combine, num_warmups=30, num_tests=30)

    # --- Benchmark: dispatch only ---
    def run_dispatch_only():
        op.dispatch_send(x, topk_weights, scale, topk_idx)
        op.dispatch_recv()

    dispatch_avg_t, dispatch_min_t, _ = bench(run_dispatch_only, num_warmups=30, num_tests=30)

    # --- Benchmark: combine only (re-run dispatch first for fresh state) ---
    d_out, d_w, d_s, d_idx, d_recv = op.dispatch_send(x, topk_weights, scale, topk_idx)
    op.dispatch_recv()
    torch.cuda.synchronize()
    dist.barrier(group=group)

    def run_combine_only():
        op.combine_send(d_out, None, topk_idx)
        op.combine_recv()

    combine_avg_t, combine_min_t, _ = bench(run_combine_only, num_warmups=30, num_tests=30)

    dispatch_bw = num_dispatch_comm_bytes / 1e9 / dispatch_avg_t if dispatch_avg_t > 0 else 0
    combine_bw = num_combine_comm_bytes / 1e9 / combine_avg_t if combine_avg_t > 0 else 0

    if rank == 0:
        # Output format compatible with extract_data.py ll log parser
        print(
            f"[rank {rank}] num_tokens={num_tokens}, hidden={hidden}, "
            f"num_experts={num_experts}, num_topk={num_topk}, "
            f"return_recv_hook=False Dispatch bandwidth: "
            f"{dispatch_bw:.2f} GB/s, "
            f"avg_t={dispatch_avg_t * 1e6:.2f} us | "
            f"Combine bandwidth: {combine_bw:.2f} GB/s, "
            f"avg_t={combine_avg_t * 1e6:.2f} us",
            flush=True,
        )

    # --- Benchmark: split send/recv timing ---
    # Dispatch send timing
    def run_dispatch_send():
        op.dispatch_send(x, topk_weights, scale, topk_idx)

    dispatch_send_avg, _, _ = bench(run_dispatch_send, num_warmups=30, num_tests=30)

    def run_dispatch_recv():
        op.dispatch_recv()

    # Need to call dispatch_send first
    op.dispatch_send(x, topk_weights, scale, topk_idx)
    torch.cuda.synchronize()
    dispatch_recv_avg, _, _ = bench(run_dispatch_recv, num_warmups=0, num_tests=1)

    # Combine send/recv timing
    d_out2, _, _, _, _ = op.dispatch_send(x, topk_weights, scale, topk_idx)
    op.dispatch_recv()
    torch.cuda.synchronize()
    dist.barrier(group=group)

    def run_combine_send():
        op.combine_send(d_out2, None, topk_idx)

    combine_send_avg, _, _ = bench(run_combine_send, num_warmups=30, num_tests=30)

    op.combine_send(d_out2, None, topk_idx)
    torch.cuda.synchronize()

    def run_combine_recv():
        op.combine_recv()

    combine_recv_avg, _, _ = bench(run_combine_recv, num_warmups=0, num_tests=1)

    if rank == 0:
        print(
            f"[rank {rank}] num_tokens={num_tokens}, hidden={hidden}, "
            f"num_experts={num_experts}, num_topk={num_topk}, "
            f"return_recv_hook=True Dispatch send/recv time: "
            f"{dispatch_send_avg * 1e6:.2f} + {dispatch_recv_avg * 1e6:.2f} us | "
            f"Combine send/recv time: {combine_send_avg * 1e6:.2f} + "
            f"{combine_recv_avg * 1e6:.2f} us",
            flush=True,
        )


# noinspection PyShadowingNames
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)

    # Initialize mori shmem
    mori.shmem.shmem_torch_process_group_init("default")

    num_tokens, hidden = args.num_tokens, args.hidden
    num_topk, num_experts = args.num_topk, args.num_experts
    num_experts_per_rank = num_experts // num_ranks

    config = _make_config(
        rank=rank,
        world_size=num_ranks,
        data_type=torch.bfloat16,
        hidden_dim=hidden,
        max_num_inp_token_per_rank=num_tokens,
        num_experts_per_rank=num_experts_per_rank,
        num_experts_per_token=num_topk,
    )
    op = mori.ops.EpDispatchCombineOp(config)

    if local_rank == 0:
        print(
            f"Allocating mori AsyncLL op: world_size={num_ranks}, "
            f"num_experts_per_rank={num_experts_per_rank}",
            flush=True,
        )

    ll_tokens = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 160, 256]

    for num_tok in ll_tokens:
        # Recreate op for each token count (max_num_inp_token_per_rank changes)
        ll_config = _make_config(
            rank=rank,
            world_size=num_ranks,
            data_type=torch.bfloat16,
            hidden_dim=hidden,
            max_num_inp_token_per_rank=num_tok,
            num_experts_per_rank=num_experts_per_rank,
            num_experts_per_token=num_topk,
        )
        ll_op = mori.ops.EpDispatchCombineOp(ll_config)

        test_main(
            num_tok,
            hidden,
            num_experts,
            num_topk,
            rank,
            num_ranks,
            group,
            ll_op,
            seed=1,
        )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test low-latency EP kernels (MORI AsyncLL)")
    parser.add_argument("--num-processes", type=int, default=8, help="Number of processes to spawn (default: 8)")
    parser.add_argument("--num-tokens", type=int, default=128, help="Number of tokens (default: 128)")
    parser.add_argument("--hidden", type=int, default=7168, help="Hidden dimension size (default: 7168)")
    parser.add_argument("--num-topk", type=int, default=8, help="Number of top-k experts (default: 8)")
    parser.add_argument("--num-experts", type=int, default=256, help="Number of experts (default: 256)")
    args = parser.parse_args()

    num_processes = args.num_processes
    torch.multiprocessing.spawn(test_loop, args=(num_processes, args), nprocs=num_processes)
