# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end tests for cli_recommend, covering the GPU-budget escalation path
needed for large models that exceed a single node's memory.
"""

import pytest

from aiconfigurator.cli.api import cli_recommend

pytestmark = [pytest.mark.e2e, pytest.mark.build]


def test_recommend_multi_node_moe_returns_results():
    """cli_recommend returns a multi-node config for Kimi-K3 on B200.

    Kimi-K3 (large MoE) does not fit in a single B200 node (8x 192 GB).
    cli_recommend must return at least one feasible configuration with
    num_total_gpus > 8.  Uses real silicon data.
    """
    result = cli_recommend(
        model_path="moonshotai/Kimi-K3",
        system="b200_sxm",
        backend="vllm",
        isl=4000,
        osl=1000,
        target_concurrency=16,
        database_mode="SILICON",
    )

    assert result.chosen_exp is not None
    best = result.best_configs.get(result.chosen_exp)
    assert best is not None and not best.empty, "Expected at least one recommended config"

    top = best.iloc[0]
    # Model requires more than one B200 node (8 GPUs) to fit in memory.
    assert top["num_total_gpus"] > 8, f"Expected multi-node config (>8 GPUs), got {top['num_total_gpus']}"
    # At least one parallelism dimension must exceed single-node capacity,
    # confirming TP/EP candidates were actually scaled during escalation.
    # agg rows use plain tp/moe_tp/moe_ep; disagg rows use (p)tp/(d)tp etc.
    # Check both name sets so the assertion holds regardless of which wins.
    parallel_dims = [
        top.get("tp", 1),
        top.get("moe_tp", 1),
        top.get("moe_ep", 1),
        top.get("(p)tp", 1),
        top.get("(d)tp", 1),
        top.get("(p)moe_ep", 1),
        top.get("(d)moe_ep", 1),
    ]
    assert max(parallel_dims) > 8
    # Sanity: predicted latencies are positive and finite.
    assert top["ttft"] > 0
    assert top["tpot"] > 0


def test_recommend_single_node_dense_model():
    """cli_recommend finds configs for a small dense model within one node."""
    result = cli_recommend(
        model_path="meta-llama/Meta-Llama-3.1-8B",
        system="h200_sxm",
        backend="vllm",
        isl=4000,
        osl=1000,
        target_concurrency=32,
        database_mode="HYBRID",
    )

    assert result.chosen_exp is not None
    best = result.best_configs.get(result.chosen_exp)
    assert best is not None and not best.empty
    # An 8B model should fit comfortably within a single H200 node.
    assert best.iloc[0]["num_total_gpus"] <= 8
