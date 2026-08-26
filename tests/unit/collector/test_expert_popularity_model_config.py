# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

from collector.expert_popularity.model_config import (
    resolve_routing_dimensions,
    validate_declared_routing_dimensions,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (
            {
                "architectures": ["DeepseekV2ForCausalLM"],
                "num_hidden_layers": 27,
                "n_routed_experts": 64,
                "num_experts_per_tok": 6,
                "first_k_dense_replace": 1,
                "moe_layer_freq": 1,
            },
            (27, 64, 6, tuple(range(1, 27))),
        ),
        (
            {
                "architectures": ["Qwen2MoeForCausalLM"],
                "num_hidden_layers": 24,
                "num_experts": 60,
                "num_experts_per_tok": 4,
                "decoder_sparse_step": 1,
            },
            (24, 60, 4, tuple(range(24))),
        ),
        (
            {
                "architectures": ["GptOssForCausalLM"],
                "num_hidden_layers": 24,
                "num_local_experts": 32,
                "experts_per_token": 4,
            },
            (24, 32, 4, tuple(range(24))),
        ),
        (
            {
                "architectures": ["MiniMaxM2ForCausalLM"],
                "num_hidden_layers": 62,
                "num_local_experts": 256,
                "num_experts_per_tok": 8,
            },
            (62, 256, 8, tuple(range(62))),
        ),
        (
            {
                "architectures": ["KimiK25ForConditionalGeneration"],
                "text_config": {
                    "num_hidden_layers": 61,
                    "n_routed_experts": 384,
                    "num_experts_per_tok": 8,
                    "first_k_dense_replace": 1,
                    "moe_layer_freq": 1,
                },
            },
            (61, 384, 8, tuple(range(1, 61))),
        ),
        (
            {
                "architectures": ["DeepseekV4ForCausalLM"],
                "num_hidden_layers": 43,
                "n_routed_experts": 256,
                "num_experts_per_tok": 6,
            },
            (43, 256, 6, tuple(range(43))),
        ),
    ],
)
def test_resolve_supported_routing_dimensions(config: dict, expected: tuple):
    resolved = resolve_routing_dimensions(config)

    assert (resolved.num_layers, resolved.num_experts, resolved.top_k, resolved.moe_layer_ids) == expected


def test_declared_dimensions_must_match_checkpoint_config():
    config = {
        "architectures": ["Qwen3MoeForCausalLM"],
        "num_hidden_layers": 48,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "decoder_sparse_step": 1,
        "mlp_only_layers": [],
    }

    with pytest.raises(ValueError, match="do not match"):
        validate_declared_routing_dimensions(
            config,
            num_layers=48,
            num_experts=64,
            top_k=8,
            moe_layer_ids=list(range(48)),
        )


def test_unknown_layer_placement_fails_closed():
    with pytest.raises(ValueError, match="unsupported MoE layer-placement"):
        resolve_routing_dimensions(
            {
                "architectures": ["UnknownMoeForCausalLM"],
                "num_hidden_layers": 4,
                "num_experts": 8,
                "num_experts_per_tok": 2,
            }
        )


def test_multinode_job_normalizes_tp_replicated_recorder_counts():
    job = (Path(__file__).parents[3] / "collector" / "expert_popularity" / "slurm" / "multinode.sbatch").read_text(
        encoding="utf-8"
    )

    assert '--replication-factor "$TP_SIZE"' in job
    assert ': "${CAMPAIGN_ROOT:?submit with CAMPAIGN_ROOT}"' in job
    assert ': "${IMAGE_SQSH:?submit with IMAGE_SQSH}"' in job
    assert ': "${HF_CACHE:?submit with HF_CACHE}"' in job
    assert "TP_SIZE=$((NNODES * GPUS_PER_NODE))" in job
    assert 'CANONICAL_MODEL_ID="${CANONICAL_MODEL_ID:-$MODEL_ID}"' in job
    assert '--checkpoint-model-id "$MODEL_ID"' in job
    assert '--checkpoint-model-revision "$MODEL_REVISION"' in job
    assert "ROUTING_EQUIVALENCE_EVIDENCE_B64" in job
    assert '--routing-equivalence-evidence-json "$ROUTING_EQUIVALENCE_EVIDENCE_JSON"' in job

    node_runner = (
        Path(__file__).parents[3] / "collector" / "expert_popularity" / "slurm" / "run_sglang_multinode_node.sh"
    ).read_text(encoding="utf-8")
    assert '"SGLANG_DSV4_FP4_EXPERTS"' in node_runner
    assert '"SGLANG_OPT_FP8_WO_A_GEMM"' in node_runner


def test_slurm_launchers_do_not_embed_site_identity():
    slurm_dir = Path(__file__).parents[3] / "collector" / "expert_popularity" / "slurm"
    forbidden = (
        "#SBATCH --account=",
        "#SBATCH --partition=",
        "#SBATCH --constraint=",
        "#SBATCH --nodelist=",
        "#SBATCH --switches=",
        "#SBATCH --segment=",
        "/home/",
        "/scratch/",
        "NCCL_SOCKET_IFNAME=",
        "NCCL_IB_HCA=",
        "UCX_NET_DEVICES=",
    )
    launchers = sorted((*slurm_dir.glob("*.sbatch"), *slurm_dir.glob("*.sh")))
    assert {path.name for path in launchers} == {
        "multinode.sbatch",
        "run_sglang_container.sh",
        "run_sglang_multinode_node.sh",
        "single_gpu.sbatch",
        "stage_model.sbatch",
        "stage_sglang_image.sbatch",
    }
    for path in launchers:
        text = path.read_text(encoding="utf-8")
        assert not [value for value in forbidden if value in text], path
