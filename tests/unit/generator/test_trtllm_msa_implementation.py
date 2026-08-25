# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MiniMax-M3 x TRT-LLM SM100-family deployments must prescribe the msa
(fmha_sm100) sparse-attention implementation: rc23 serving defaults to the
Triton reference, while the shipped SM100/103 perf tables are collected with
implementation="msa" — the emitted engine config makes deployments run
exactly the configuration the data represents (PR #1507 review 4969690316).
Non-MSA models and non-SM100 systems must NOT carry the block."""

from __future__ import annotations

import pytest
import yaml

from aiconfigurator.generator.api import generate_backend_artifacts

pytestmark = pytest.mark.unit

_BACKEND_VERSION = "1.3.0rc23"


def _params(model_cfg: dict) -> dict:
    return {
        "ServiceConfig": {
            "model_path": "MiniMaxAI/MiniMax-M3",
            "served_model_path": "MiniMaxAI/MiniMax-M3",
            "served_model_name": "MiniMax-M3",
            "include_frontend": True,
        },
        "K8sConfig": {"name_prefix": "test", "k8s_namespace": "default"},
        "DynConfig": {"mode": "agg"},
        "WorkerConfig": {"agg_workers": 1, "agg_gpus_per_worker": 8},
        "NodeConfig": {"num_gpus_per_node": 8},
        "SlaConfig": {"isl": 1024, "osl": 256},
        "ModelConfig": {"is_moe": True, "prefix": 0, **model_cfg},
        "BenchConfig": {},
        "params": {
            "agg": {
                "tensor_parallel_size": 8,
                "pipeline_parallel_size": 1,
                "data_parallel_size": 1,
                "moe_tensor_parallel_size": 1,
                "moe_expert_parallel_size": 8,
                "max_batch_size": 32,
                "max_num_tokens": 4096,
                "max_seq_len": 4096,
            },
        },
    }


def _engine_yaml(artifacts) -> dict:
    for name, content in artifacts.items():
        if "extra_engine_args" in name and "agg" in name:
            return yaml.safe_load(content)
    raise AssertionError(f"no agg engine args artifact in {sorted(artifacts)}")


def test_msa_implementation_rendered_when_bridge_sets_it():
    artifacts = generate_backend_artifacts(
        backend="trtllm",
        backend_version=_BACKEND_VERSION,
        params=_params({"msa_sparse_implementation": "msa"}),
    )
    engine = _engine_yaml(artifacts)
    assert engine.get("sparse_attention_config") == {"implementation": "msa"}


def test_no_sparse_block_without_the_bridge_field():
    artifacts = generate_backend_artifacts(
        backend="trtllm",
        backend_version=_BACKEND_VERSION,
        params=_params({}),
    )
    engine = _engine_yaml(artifacts)
    assert "sparse_attention_config" not in engine


class _FakeTask:
    def __init__(self, backend="trtllm", model="MiniMaxAI/MiniMax-M3", system="b200_sxm"):
        self.primary_backend_name = backend
        self.primary_model_path = model
        self.primary_system_name = system


@pytest.mark.parametrize(
    ("backend", "model", "system", "expected"),
    [
        ("trtllm", "MiniMaxAI/MiniMax-M3", "b200_sxm", "msa"),
        # The NVFP4 bundle carries the raw hub VL-wrapper architecture; the
        # prescription must cover both artifact forms of the same model.
        ("trtllm", "nvidia/MiniMax-M3-NVFP4", "b200_sxm", "msa"),
        ("trtllm", "MiniMaxAI/MiniMax-M3", "gb300", "msa"),
        ("trtllm", "MiniMaxAI/MiniMax-M3", "h200_sxm", None),  # SM90: Triton default
        ("trtllm", "MiniMaxAI/MiniMax-M3", "rtx_pro_6000_server", None),  # SM120
        ("trtllm", "deepseek-ai/DeepSeek-V3", "b200_sxm", None),  # non-MSA arch
        ("vllm", "MiniMaxAI/MiniMax-M3", "b200_sxm", None),  # trtllm-only knob
    ],
)
def test_bridge_prescribes_msa_only_for_m3_on_the_sm100_family(backend, model, system, expected):
    from aiconfigurator.generator.module_bridge import _msa_sparse_implementation

    assert _msa_sparse_implementation(_FakeTask(backend, model, system)) == expected


def test_naive_generator_carries_the_msa_prescription():
    """The naive entry point (`cli generate` path) must emit the same
    MiniMax-M3/SM100-family prescription as the optimized path — otherwise a
    naive deployment runs the TRT-LLM default the perf rows do not represent
    (review 4972622548 item 1)."""
    from aiconfigurator.generator.naive import build_naive_generator_params

    params = build_naive_generator_params(
        model_name="MiniMaxAI/MiniMax-M3",
        total_gpus=8,
        system_name="b200_sxm",
        backend_name="trtllm",
        mode="agg",
    )
    assert params["ModelConfig"]["msa_sparse_implementation"] == "msa"

    off_family = build_naive_generator_params(
        model_name="MiniMaxAI/MiniMax-M3",
        total_gpus=8,
        system_name="h200_sxm",
        backend_name="trtllm",
        mode="agg",
    )
    assert "msa_sparse_implementation" not in off_family["ModelConfig"]
