# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TRT-LLM disagg MTP: BOTH prefill and decode engines need speculative_config.

TRT-LLM sizes the KV cache with ``get_num_spec_layers(spec_config)`` extra
MTP-layer pools; a ctx (prefill) engine without ``speculative_config`` has a
mismatched KV layout vs the gen (decode) engine. The official disaggregated
benchmark config carries the block on both sides.
"""

from __future__ import annotations

import copy

import pytest
import yaml

from aiconfigurator.generator.api import generate_backend_artifacts

pytestmark = pytest.mark.unit

_BACKEND_VERSION = "1.3.0rc14"

_PARAMS = {
    "ServiceConfig": {
        "model_path": "deepseek-ai/DeepSeek-V3",
        "served_model_path": "deepseek-ai/DeepSeek-V3",
        "served_model_name": "DeepSeek-V3",
        "include_frontend": True,
    },
    "K8sConfig": {"name_prefix": "test", "k8s_namespace": "default"},
    "DynConfig": {"mode": "disagg"},
    "WorkerConfig": {
        "agg_workers": 0,
        "prefill_workers": 1,
        "decode_workers": 1,
        "prefill_gpus_per_worker": 8,
        "decode_gpus_per_worker": 8,
    },
    "NodeConfig": {"num_gpus_per_node": 8},
    "SlaConfig": {"isl": 1024, "osl": 256},
    "ModelConfig": {"is_moe": True, "prefix": 0, "nextn": 1},
    "BenchConfig": {},
    "params": {
        "prefill": {
            "tensor_parallel_size": 8,
            "pipeline_parallel_size": 1,
            "data_parallel_size": 1,
            "moe_tensor_parallel_size": 1,
            "moe_expert_parallel_size": 8,
            "max_batch_size": 4,
            "max_num_tokens": 4096,
            "max_seq_len": 4096,
        },
        "decode": {
            "tensor_parallel_size": 8,
            "pipeline_parallel_size": 1,
            "data_parallel_size": 1,
            "moe_tensor_parallel_size": 1,
            "moe_expert_parallel_size": 8,
            "max_batch_size": 64,
            "max_num_tokens": 512,
            "max_seq_len": 4096,
        },
    },
}


def _render(nextn: int):
    params = copy.deepcopy(_PARAMS)
    params["ModelConfig"]["nextn"] = nextn
    return generate_backend_artifacts(
        params,
        "trtllm",
        backend_version=_BACKEND_VERSION,
        deployment_target="dynamo-j2",
    )


def test_disagg_mtp_spec_config_on_both_roles():
    artifacts = _render(nextn=1)
    for role in ("prefill", "decode"):
        engine = yaml.safe_load(artifacts[f"extra_engine_args_{role}.yaml"])
        spec = engine.get("speculative_config")
        assert spec is not None, f"{role} engine is missing speculative_config"
        assert spec["decoding_type"] == "MTP"
        assert spec["num_nextn_predict_layers"] == 1


def test_disagg_mtp_preserves_exact_nextn_value():
    """num_nextn_predict_layers must carry the exact draft length (no cap)."""
    artifacts = _render(nextn=6)
    for role in ("prefill", "decode"):
        engine = yaml.safe_load(artifacts[f"extra_engine_args_{role}.yaml"])
        assert engine["speculative_config"]["num_nextn_predict_layers"] == 6


def test_disagg_nextn_zero_emits_no_spec_config():
    artifacts = _render(nextn=0)
    for role in ("prefill", "decode"):
        engine = yaml.safe_load(artifacts[f"extra_engine_args_{role}.yaml"])
        assert "speculative_config" not in (engine or {}), f"{role} engine unexpectedly has speculative_config"
