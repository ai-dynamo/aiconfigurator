# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end tests for cli_recommend, covering the GPU-budget escalation path
needed for large models that exceed a single node's memory.
"""

import pytest

from aiconfigurator.cli.api import cli_recommend

pytestmark = pytest.mark.e2e


def test_recommend_multi_node_moe_returns_results():
    """cli_recommend finds valid configs for Kimi-K3 on H200.

    Kimi-K3 (2.8T MoE) does not fit in a single H200 node (8x 80 GB).
    The recommend escalation must scale TP/EP candidates beyond the single-node
    limit and return at least one feasible configuration.
    """
    result = cli_recommend(
        model_path="moonshotai/Kimi-K3",
        system="h200_sxm",
        backend="vllm",
        isl=4000,
        osl=1000,
        target_concurrency=16,
        database_mode="HYBRID",
    )

    assert result.chosen_exp is not None
    best = result.best_configs.get(result.chosen_exp)
    assert best is not None and not best.empty, "Expected at least one recommended config"

    top = best.iloc[0]
    # Model requires more than one H200 node (8 GPUs) to fit in memory.
    assert top["num_total_gpus"] > 8, f"Expected multi-node config (>8 GPUs), got {top['num_total_gpus']}"
    # At least one parallelism dimension must exceed single-node capacity,
    # confirming TP/EP candidates were actually scaled during escalation.
    assert max(top["tp"], top["moe_tp"], top["moe_ep"]) > 8
    # Sanity: predicted latencies are positive and finite.
    assert top["ttft"] > 0
    assert top["tpot"] > 0


def test_recommend_single_node_dense_model():
    """cli_recommend finds configs for a small dense model within one node."""
    result = cli_recommend(
        model_path="meta-llama/Llama-3.1-8B-Instruct",
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
