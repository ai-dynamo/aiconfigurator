# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model/hardware resolution via AIConfigurator (weights -> memory-fit floor).

Skipped unless ``aiconfigurator`` is importable. Uses models whose configs
resolve without HF auth (DeepSeek-V3, Qwen3-32B)."""

import pytest

pytest.importorskip("aiconfigurator")

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
    assert all(c.shape.moe_tp == 1 for c in cfgs)  # MLA -> only TEP/DEP
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
