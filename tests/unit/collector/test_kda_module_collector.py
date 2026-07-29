# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the module-level KDA collector's GPU-free parts:
the doctored shrunk-config recipe (the collector's structural contract with
the kimi-k3 branch model code) and the declared-grid reuse."""

import json
import sys
from unittest.mock import MagicMock

import pytest

_saved_mock = sys.modules.get("torch")
_restore_mock = isinstance(_saved_mock, MagicMock)
if _restore_mock:
    sys.modules.pop("torch")

try:
    import torch as _real_torch  # noqa: F401
except ImportError:
    if _restore_mock:
        sys.modules["torch"] = _saved_mock
    pytest.skip("real torch required to import the collector module", allow_module_level=True)

try:
    from collector.registry_types import PerfFile
    from collector.sglang.collect_kda_module import (
        K3_SHARD_HEADS,
        _base_grid,
        _doctor_model_config,
        _geometry_from_config,
    )
finally:
    if _restore_mock:
        sys.modules["torch"] = _saved_mock


def _k3_like_config() -> dict:
    return {
        "architectures": ["KimiK3ForConditionalGeneration"],
        "text_config": {
            "hidden_size": 7168,
            "num_hidden_layers": 93,
            "num_attention_heads": 96,
            "num_key_value_heads": 96,
            "num_experts": 896,
            "num_expert_group": 1,
            "linear_attn_config": {
                "num_heads": 96,
                "head_dim": 128,
                "short_conv_kernel_size": 4,
                "use_full_rank_gate": True,
                "kda_layers": [1, 2, 3, 5],
                "full_attn_layers": [4, 8],
            },
        },
    }


@pytest.fixture
def model_src(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "config.json").write_text(json.dumps(_k3_like_config()))
    (src / "tokenizer_config.json").write_text("{}")
    # weight artifacts must NOT be copied (dummy load needs none)
    (src / "model-00001.safetensors").write_bytes(b"weights")
    (src / "model.safetensors.index.json").write_text("{}")
    return src


@pytest.mark.unit
def test_doctored_config_realizes_per_rank_shard(model_src, tmp_path):
    dst = tmp_path / "dst"
    _doctor_model_config(str(model_src), str(dst), kda_heads=12)
    cfg = json.loads((dst / "config.json").read_text())
    tc = cfg["text_config"]

    # 4 layers, 1-indexed lists: is_kda_layer(idx) == (idx+1) in kda_layers
    # (srt/configs/kimi_linear.py:156-159), so layers 0..2 are KDA, 3 is MLA.
    assert tc["num_hidden_layers"] == 4
    assert tc["linear_attn_config"]["kda_layers"] == [1, 2, 3]
    assert tc["linear_attn_config"]["full_attn_layers"] == [4]
    assert tc["linear_attn_config"]["num_heads"] == 12
    # MLA heads are pinned to the runnable shard regardless of KDA geometry
    # (trtllm_mla decode rejects the full 96 q-heads at tp1).
    assert tc["num_attention_heads"] == 12
    assert tc["num_key_value_heads"] == 12
    # MoE shrink is legal only because num_expert_group == 1 in the source.
    assert tc["num_experts"] == 64
    # untouched structural fields survive
    assert tc["hidden_size"] == 7168
    assert tc["linear_attn_config"]["use_full_rank_gate"] is True


@pytest.mark.unit
def test_doctored_dir_excludes_weight_artifacts(model_src, tmp_path):
    dst = tmp_path / "dst"
    _doctor_model_config(str(model_src), str(dst), kda_heads=96)
    names = {p.name for p in dst.iterdir()}
    assert "config.json" in names
    assert "tokenizer_config.json" in names
    assert "model-00001.safetensors" not in names
    assert "model.safetensors.index.json" not in names


@pytest.mark.unit
def test_geometry_row_fields_come_from_doctored_config(model_src, tmp_path):
    dst = tmp_path / "dst"
    _doctor_model_config(str(model_src), str(dst), kda_heads=24)
    geometry = _geometry_from_config(str(dst))
    assert geometry == {
        "d_model": 7168,
        "d_conv": 4,
        "num_k_heads": 24,
        "head_k_dim": 128,
        "num_v_heads": 24,
        "head_v_dim": 128,
    }


@pytest.mark.unit
def test_grid_comes_from_declared_kda_base_yaml():
    grid = _base_grid()
    # the module lane sweeps the SAME declared grid as the kernel lane so
    # module rows interpolate on identical support
    assert grid["generation_batch_sizes"][0] == 1
    assert 1024 in grid["generation_batch_sizes"]
    assert 32768 in grid["context_sequence_lengths"]
    assert set(grid["context_batch_sizes"]) >= {1, 64}


@pytest.mark.unit
def test_shard_set_covers_k3_tp_ladder():
    assert K3_SHARD_HEADS == (12, 24, 48, 96)  # 96 / tp for tp in (8,4,2,1)


@pytest.mark.unit
def test_output_defaults_to_registered_perf_file():
    assert PerfFile.LINEAR_ATTN_MODULE.value == "linear_attn_module_perf.txt"
