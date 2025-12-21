# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from aiconfigurator.sdk.perf_database import get_database
from aiconfigurator.sdk.task import TaskConfig, TaskConfigFactory

REPO_ROOT = Path(__file__).resolve().parents[2]
SYSTEMS_DIR = str(REPO_ROOT / "src" / "aiconfigurator" / "systems")


@pytest.mark.parametrize("model_name", ["QWEN3_32B"])
def test_l40s_sglang_default_gemm_quant_does_not_use_fp8_block(model_name: str) -> None:
    db = get_database(system="l40s", backend="sglang", version="0.5.5.post3", systems_dir=SYSTEMS_DIR)
    assert db is not None, "Expected l40s/sglang/0.5.5.post3 database to load in test environment"

    gemm, moe, kvcache, fmha, comm = TaskConfigFactory._get_quant_mode(
        model_name=model_name, backend="sglang", database=db, use_specific_quant_mode=None
    )
    assert gemm != "fp8_block"
    assert gemm in {"fp8", "float16", "int8_wo"}


def test_l40s_sglang_forced_fp8_block_gemm_fails_fast_with_actionable_error() -> None:
    with pytest.raises(ValueError) as excinfo:
        TaskConfig(
            serving_mode="agg",
            model_name="QWEN3_32B",
            system_name="l40s",
            backend_name="sglang",
            backend_version="0.5.5.post3",
            total_gpus=8,
            yaml_config={
                "mode": "patch",
                "config": {"worker_config": {"gemm_quant_mode": "fp8_block"}},
            },
        )

    msg = str(excinfo.value)
    assert "Unsupported gemm quant mode" in msg
    assert "Supported gemm modes" in msg
