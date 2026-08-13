# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

from aiconfigurator.sdk.utils import HuggingFaceDownloadError
from tools.support_matrix import support_matrix
from tools.support_matrix.support_matrix import (
    SupportMatrix,
    TestConstraints,
    _get_support_matrix_image_size,
    _require_nonzero_encoder_result,
    _support_matrix_row_command,
)

pytestmark = pytest.mark.unit

SCOUT = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
MAVERICK = "meta-llama/Llama-4-Maverick-17B-128E-Instruct"
CONSTRAINTS = TestConstraints(total_gpus=32, isl=256, osl=256, prefix=128, ttft=2000.0, tpot=50.0)


@pytest.mark.parametrize("model_id", [SCOUT, MAVERICK])
def test_llama4_matrix_uses_checkpoint_image_size(model_id):
    assert _get_support_matrix_image_size(model_id) == 336


def test_llama4_matrix_guard_does_not_change_qwen3vl_workloads():
    assert _get_support_matrix_image_size("Qwen/Qwen3-VL-32B-Instruct") == 0


@pytest.mark.parametrize("mode", ["agg", "disagg"])
@pytest.mark.parametrize("model_id", [SCOUT, MAVERICK])
def test_llama4_matrix_tasks_enable_nonzero_image_work(mode, model_id):
    task = SupportMatrix._create_task(
        mode=mode,
        model=model_id,
        system="h200_sxm",
        backend="trtllm",
        version="1.3.0rc20",
        constraints=CONSTRAINTS,
        engine_step_backend=None,
    )

    assert task.image_height == 336
    assert task.image_width == 336
    assert task.num_images_per_request == 1


def test_llama4_replay_command_cannot_skip_encoder():
    command = _support_matrix_row_command(
        model=SCOUT,
        system="h200_sxm",
        backend="trtllm",
        version="1.3.0rc20",
        mode="agg",
        constraints=CONSTRAINTS,
    )

    assert "--image-height 336" in command
    assert "--image-width 336" in command
    assert "--num-images 1" in command


@pytest.mark.parametrize("model_id", [SCOUT, MAVERICK])
def test_matrix_rejects_skipped_llama4_encoder(model_id):
    with pytest.raises(RuntimeError, match="produced no nonzero encoder work"):
        _require_nonzero_encoder_result(model_id, pd.DataFrame({"encoder_latency": [0.0]}))


@pytest.mark.parametrize("model_id", [SCOUT, MAVERICK])
def test_matrix_accepts_nonzero_llama4_encoder(model_id):
    _require_nonzero_encoder_result(model_id, pd.DataFrame({"encoder_latency": [1.0]}))


@pytest.mark.parametrize("model_id", [SCOUT, MAVERICK])
def test_matrix_rejects_missing_encoder_latency_column(model_id):
    with pytest.raises(RuntimeError, match="produced no nonzero encoder work"):
        _require_nonzero_encoder_result(model_id, pd.DataFrame({"ttft": [1.0]}))


def test_live_matrix_paths_surface_metadata_failures_but_explicit_legacy_rendering_remains_tolerant(monkeypatch):
    def _boom(_model_path):
        raise HuggingFaceDownloadError("offline")

    monkeypatch.setattr(support_matrix, "_get_model_info", _boom)

    with pytest.raises(HuggingFaceDownloadError, match="offline"):
        _require_nonzero_encoder_result(SCOUT, pd.DataFrame({"encoder_latency": [0.0]}))

    with pytest.raises(HuggingFaceDownloadError, match="offline"):
        SupportMatrix._create_task(
            mode="agg",
            model=SCOUT,
            system="h200_sxm",
            backend="trtllm",
            version="1.3.0rc20",
            constraints=CONSTRAINTS,
            engine_step_backend=None,
        )

    with pytest.raises(HuggingFaceDownloadError, match="offline"):
        _support_matrix_row_command(
            model=SCOUT,
            system="h200_sxm",
            backend="trtllm",
            version="1.3.0rc20",
            mode="agg",
            constraints=CONSTRAINTS,
        )

    command = _support_matrix_row_command(
        model="arbitrary/legacy-model",
        system="h200_sxm",
        backend="trtllm",
        version="1.3.0rc20",
        mode="agg",
        constraints=CONSTRAINTS,
        image_size=0,
    )
    assert "--image-height" not in command
