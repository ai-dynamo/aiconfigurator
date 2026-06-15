# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model/hardware resolution via AIConfigurator (weights -> memory-fit floor).

Skipped unless ``aiconfigurator`` is importable. Uses models whose configs
resolve without HF auth (DeepSeek-V3, Qwen3-32B)."""

import pytest

pytest.importorskip("aiconfigurator")

import spica.model_hw as mh_mod  # noqa: E402
from spica.kv_estimate import NoPerfDatabase  # noqa: E402
from spica.model_hw import (  # noqa: E402
    NoViableParallelConfig,
    parallel_configs_for,
    resolve_model_hardware,
)

DEEPSEEK = "deepseek-ai/DeepSeek-V3"
QWEN = "Qwen/Qwen3-32B"


def test_resolve_deepseek_is_moe_mla_with_fit_floor():
    mh = resolve_model_hardware(DEEPSEEK, "h200_sxm", gpu_budget=32, backend="trtllm")
    assert mh.is_moe and mh.mla and mh.enable_wideep and mh.fits
    # ~1.3 TB of weights -> needs many GPUs per worker; power of 2.
    assert mh.min_gpus_per_worker >= 8
    assert mh.min_gpus_per_worker & (mh.min_gpus_per_worker - 1) == 0


def test_resolve_dense_qwen():
    mh = resolve_model_hardware(QWEN, "h200_sxm", gpu_budget=8, backend="trtllm")
    assert not mh.is_moe
    assert not mh.mla
    assert mh.min_gpus_per_worker == 1
    assert mh.fits


def test_agg_configs_respect_memory_fit_floor():
    mh = resolve_model_hardware(DEEPSEEK, "h200_sxm", gpu_budget=32, backend="trtllm")
    cfgs = parallel_configs_for(DEEPSEEK, "h200_sxm", gpu_budget=32, deployment_mode="agg", backend="trtllm")
    assert cfgs
    assert all(c.shape.gpus_per_worker >= mh.min_gpus_per_worker for c in cfgs)
    assert any(c.shape.moe_tp > 1 for c in cfgs)  # pure expert-TP scanned for all MoE
    assert all(c.total_gpus <= 32 for c in cfgs)


def test_disagg_configs_each_role_above_floor():
    mh = resolve_model_hardware(DEEPSEEK, "h200_sxm", gpu_budget=32, backend="trtllm")
    cfgs = parallel_configs_for(DEEPSEEK, "h200_sxm", gpu_budget=32, deployment_mode="disagg", backend="trtllm")
    assert cfgs
    for c in cfgs:
        assert c.prefill.shape.gpus_per_worker >= mh.min_gpus_per_worker
        assert c.decode.shape.gpus_per_worker >= mh.min_gpus_per_worker
        assert c.total_gpus <= 32


def test_model_too_big_for_budget_raises():
    with pytest.raises(NoViableParallelConfig):
        parallel_configs_for(DEEPSEEK, "h200_sxm", gpu_budget=8, deployment_mode="agg", backend="trtllm")


# --- KV-cache validity path (max_seq_len given) ---


def test_kv_filter_keeps_only_feasible_shapes(monkeypatch):
    # Pretend only workers with >= 4 GPUs hold a sequence (KV estimate stubbed).
    def fake_feasible(shapes, **kwargs):
        return {s: 100_000 for s in dict.fromkeys(shapes) if s.gpus_per_worker >= 4}

    monkeypatch.setattr(mh_mod, "feasible_shape_tokens", fake_feasible)
    cfgs = parallel_configs_for(
        DEEPSEEK, "gb200", gpu_budget=16, deployment_mode="agg", backend="trtllm", max_seq_len=8192
    )
    assert cfgs
    assert all(c.shape.gpus_per_worker >= 4 for c in cfgs)  # weight floor bypassed; KV decides
    assert all(c.total_gpus <= 16 for c in cfgs)


def test_kv_filter_disagg_requires_both_roles_feasible(monkeypatch):
    def fake_feasible(shapes, **kwargs):
        return {s: 100_000 for s in dict.fromkeys(shapes) if s.gpus_per_worker >= 4}

    monkeypatch.setattr(mh_mod, "feasible_shape_tokens", fake_feasible)
    cfgs = parallel_configs_for(
        DEEPSEEK, "gb200", gpu_budget=16, deployment_mode="disagg", backend="trtllm", max_seq_len=8192
    )
    assert cfgs
    for c in cfgs:
        assert c.prefill.shape.gpus_per_worker >= 4
        assert c.decode.shape.gpus_per_worker >= 4


def test_kv_filter_no_feasible_shape_raises(monkeypatch):
    monkeypatch.setattr(mh_mod, "feasible_shape_tokens", lambda shapes, **kwargs: {})
    with pytest.raises(NoViableParallelConfig, match="KV-cache estimate"):
        parallel_configs_for(
            DEEPSEEK, "gb200", gpu_budget=16, deployment_mode="agg", backend="trtllm", max_seq_len=8192
        )


def test_kv_path_end_to_end_deepseek_gb200():
    # Real native estimate; skipped without the gb200 perf DB / model build.
    try:
        cfgs = parallel_configs_for(
            DEEPSEEK, "gb200", gpu_budget=16, deployment_mode="agg", backend="trtllm", max_seq_len=8192
        )
    except NoPerfDatabase:
        pytest.skip("no gb200/trtllm perf DB")
    except ValueError as exc:
        if "unsupported model/backend/GPU" in str(exc):
            pytest.skip(f"native KV build unavailable: {exc}")
        raise
    assert cfgs
    # DeepSeek-V3 OOMs at 2 GPUs/worker; smallest feasible worker is >= 4 GPUs.
    assert all(c.shape.gpus_per_worker >= 4 for c in cfgs)
    assert all(c.total_gpus <= 16 for c in cfgs)
