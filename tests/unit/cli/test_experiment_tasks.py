# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""build_experiment_tasks builds v2 Tasks from (legacy or flat) experiment YAML.

Legacy V1 experiment dicts (``mode`` / nested ``config`` / ``profiles``) are
auto-converted to the flat V2 schema by ``Task.from_yaml``; these tests check
that top-level fields survive the conversion onto the flat ``Task``.
"""

import pytest

from aiconfigurator.cli.main import build_experiment_tasks

pytestmark = pytest.mark.unit


def test_build_experiment_preserves_top_level_runtime_fields():
    tasks = build_experiment_tasks(
        config={
            "exps": ["exp_agg_trtllm"],
            "exp_agg_trtllm": {
                "mode": "patch",  # legacy marker -> auto-converted
                "serving_mode": "agg",
                "model_path": "Qwen/Qwen3-32B",
                "total_gpus": 8,
                "system_name": "h200_sxm",
                "backend_name": "trtllm",
                "isl": 8000,
                "osl": 1000,
                "prefix": 5600,
                "ttft": 1000.0,
                "tpot": 20.0,
                "request_latency": 25000.0,
            },
        }
    )

    task = tasks["exp_agg_trtllm"]
    assert task.serving_mode == "agg"
    assert task.isl == 8000
    assert task.osl == 1000
    assert task.prefix == 5600
    assert task.ttft == 1000.0
    assert task.tpot == 20.0
    assert task.request_latency == 25000.0


def test_build_experiment_accepts_flat_v2_disagg():
    """Flat-V2 disagg has no top-level model_path/system_name (only prefill_*/decode_*);
    the pre-flight skip check must not drop it (regression: V1-centric check skipped it)."""
    tasks = build_experiment_tasks(
        config={
            "exps": ["exp_disagg"],
            "exp_disagg": {
                "serving_mode": "disagg",
                "prefill_model_path": "Qwen/Qwen3-32B-FP8",
                "prefill_system_name": "h200_sxm",
                "decode_model_path": "Qwen/Qwen3-32B-FP8",
                "decode_system_name": "h200_sxm",
                "total_gpus": 16,
            },
        }
    )
    assert "exp_disagg" in tasks
    assert tasks["exp_disagg"].serving_mode == "disagg"


def test_build_experiment_keeps_prefix_at_top_level():
    tasks = build_experiment_tasks(
        config={
            "exps": ["exp_agg"],
            "exp_agg": {
                "mode": "patch",
                "serving_mode": "agg",
                "model_path": "nvidia/Kimi-K2.5-NVFP4",
                "total_gpus": 8,
                "system_name": "b200_sxm",
                "backend_name": "trtllm",
                "database_mode": "HYBRID",
                "isl": 4000,
                "osl": 1000,
                "prefix": 1000,
            },
        }
    )

    task = tasks["exp_agg"]
    assert task.prefix == 1000
    assert task.database_mode == "HYBRID"


def test_build_experiment_forwards_top_level_moe_backend():
    tasks = build_experiment_tasks(
        config={
            "exps": ["exp_megamoe"],
            "exp_megamoe": {
                "mode": "patch",
                "serving_mode": "agg",
                "model_path": "deepseek-ai/DeepSeek-V4-Pro",
                "total_gpus": 32,
                "system_name": "gb200",
                "backend_name": "sglang",
                "backend_version": "0.5.10",
                "database_mode": "HYBRID",
                "moe_backend": "megamoe",
            },
        }
    )

    task = tasks["exp_megamoe"]
    assert task.serving_mode == "agg"
    assert task.moe_backend == "megamoe"


def test_build_experiment_preflight_uses_per_role_backend_version_for_flat_v2_disagg(monkeypatch):
    """Flat-v2 disagg has no top-level backend_version; the early backend-version check must
    read the per-role prefill_*/decode_* fields (regression: top-level-only check skipped them)."""
    import aiconfigurator.cli.main as cli_main

    calls: list[tuple] = []
    monkeypatch.setattr(
        cli_main,
        "_ensure_backend_version_available",
        lambda system, backend, version: calls.append((system, backend, version)),
    )
    build_experiment_tasks(
        config={
            "exps": ["exp_disagg"],
            "exp_disagg": {
                "serving_mode": "disagg",
                "prefill_model_path": "Qwen/Qwen3-32B-FP8",
                "prefill_system_name": "h200_sxm",
                "prefill_backend_name": "sglang",
                "prefill_backend_version": "0.5.10",
                "decode_model_path": "Qwen/Qwen3-32B-FP8",
                "decode_system_name": "h200_sxm",
                "decode_backend_name": "sglang",
                "decode_backend_version": "0.5.10",
                "total_gpus": 16,
            },
        }
    )
    # The preflight runs before Task construction, so the per-role version is validated
    # even if downstream construction differs; top-level-only code would have recorded nothing.
    assert ("h200_sxm", "sglang", "0.5.10") in calls
