# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Node allocation utilities.

This module provides a simple bin-packing heuristic to place prefill/decode
workers on nodes given per-worker GPU count.
"""
from typing import List, Dict
GPU_PER_NODE = 8 # TODO: Support other situations.

def allocate_disagg_nodes(p_worker: int, p_gpu: int, d_worker: int, d_gpu: int,
                          gpu_per_node: int = GPU_PER_NODE) -> List[Dict[str, int]]:
    """Greedy placement of workers on nodes."""
    nodes = []
    # prefill
    for _ in range(p_worker):
        placed = False
        for n in nodes:
            if n['used'] + p_gpu <= gpu_per_node:
                n['p_worker'] += 1
                n['used'] += p_gpu
                placed = True
                break
        if not placed:
            nodes.append({'p_worker': 1, 'd_worker': 0, 'used': p_gpu})
    # decode
    for _ in range(d_worker):
        placed = False
        for n in nodes:
            if n['used'] + d_gpu <= gpu_per_node:
                n['d_worker'] += 1
                n['used'] += d_gpu
                placed = True
                break
        if not placed:
            nodes.append({'p_worker': 0, 'd_worker': 1, 'used': d_gpu})
    return [{'p_worker': n['p_worker'], 'd_worker': n['d_worker']} for n in nodes]
