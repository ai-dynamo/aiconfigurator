# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from aiconfigurator.sdk import common
from tools.support_matrix import support_matrix as support_matrix_module
from tools.support_matrix.compare_support_matrix import check_csv_sanity, read_csv
from tools.support_matrix.support_matrix import (
    STATUS_FAIL,
    STATUS_HYBRID_PASS,
    STATUS_PASS,
    ModelMetadataResolutionError,
    SupportMatrix,
    TestConstraints,
    _get_matrix_visual_workload,
    _resolve_test_constraints,
    _support_matrix_row_command,
)

pytestmark = pytest.mark.unit


def _result(system: str) -> tuple[str, str, str, str, str, str, str, None]:
    """Build a minimal passing support-matrix result row for a system."""
    return ("test/model", "TestArchitecture", system, "trtllm", "1.0.0", "agg", STATUS_PASS, None)


def test_save_results_to_csv_writes_manifest_in_display_order(tmp_path):
    """Split support-matrix output should preserve product-priority file order."""
    results = [
        _result("b60"),
        _result("a100_sxm"),
        _result("l40s"),
        _result("h100_sxm"),
        _result("h200_sxm"),
        _result("rtx_pro_6000_server"),
        _result("gb300"),
        _result("b300_sxm"),
        _result("gb200"),
        _result("b200_sxm"),
    ]

    support_matrix = SupportMatrix.__new__(SupportMatrix)
    support_matrix.save_results_to_csv(results, str(tmp_path))

    with (tmp_path / "index.json").open() as f:
        manifest = json.load(f)

    assert manifest["files"] == [
        "b200_sxm.csv",
        "gb200.csv",
        "b300_sxm.csv",
        "gb300.csv",
        "rtx_pro_6000_server.csv",
        "h200_sxm.csv",
        "h100_sxm.csv",
        "l40s.csv",
        "a100_sxm.csv",
        "b60.csv",
    ]


def test_save_results_to_csv_upgrades_legacy_rows_to_valid_current_schema(tmp_path):
    silicon_command = (
        "uv run aiconfigurator cli default --model-path test/model-9col --total-gpus 8 "
        "--system b200_sxm --backend trtllm --backend-version 1.0.0 "
        "--database-mode SILICON --isl 256 --osl 256 --prefix 0 --ttft 2000 --tpot 50 --top-n 1 --no-color"
    )
    results = [
        # Legacy 8-column PASS: command and Source are both upgraded.
        ("test/model-8col", "TestArchitecture", "b200_sxm", "trtllm", "1.0.0", "agg", STATUS_PASS, None),
        # Legacy 9-column PASS: preserve its command and infer Source=silicon.
        (
            "test/model-9col",
            "TestArchitecture",
            "b200_sxm",
            "trtllm",
            "1.0.0",
            "agg",
            STATUS_PASS,
            None,
            silicon_command,
        ),
        # Non-pass legacy rows remain source-less in the current schema.
        (
            "test/model-fail",
            "TestArchitecture",
            "b200_sxm",
            "trtllm",
            "1.0.0",
            "agg",
            STATUS_FAIL,
            "expected failure",
            silicon_command.replace("test/model-9col", "test/model-fail"),
        ),
    ]
    output_file = tmp_path / "support.csv"
    support_matrix = SupportMatrix.__new__(SupportMatrix)

    support_matrix.save_results_to_csv(results, str(output_file))
    header, rows = read_csv(str(output_file))

    assert [row[9] for row in rows] == ["silicon", "silicon", ""]
    assert rows[1][8] == silicon_command
    assert check_csv_sanity(header, rows) == []


def test_save_results_to_csv_rejects_legacy_hybrid_pass_without_source(tmp_path):
    row = (
        "test/hybrid",
        "TestArchitecture",
        "b200_sxm",
        "trtllm",
        "1.0.0",
        "agg",
        STATUS_HYBRID_PASS,
        None,
        "uv run aiconfigurator cli default --database-mode HYBRID",
    )
    support_matrix = SupportMatrix.__new__(SupportMatrix)

    with pytest.raises(ValueError, match="cannot be upgraded without an explicit empirical Source"):
        support_matrix.save_results_to_csv([row], str(tmp_path / "support.csv"))


def test_task_uses_silicon_database_mode(monkeypatch):
    captured_kwargs = {}

    class FakeTask:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    monkeypatch.setattr(support_matrix_module, "Task", FakeTask)

    SupportMatrix._create_task(
        mode="agg",
        model="Qwen/Qwen3-0.6B",
        system="b200_sxm",
        backend="sglang",
        version="0.5.12",
        constraints=TestConstraints(total_gpus=4, isl=256, osl=256, prefix=128, ttft=1500.0, tpot=50.0),
        engine_step_backend=None,
    )

    assert captured_kwargs["database_mode"] == common.DatabaseMode.SILICON.name


def test_matrix_derives_gemma4_fixed_budget_image_workload():
    _get_matrix_visual_workload.cache_clear()

    assert _get_matrix_visual_workload("google/gemma-4-26B-A4B") == (672, 960, 1)
    assert _get_matrix_visual_workload("Qwen/Qwen3-VL-32B-Instruct") == (448, 448, 1)
    assert _get_matrix_visual_workload("Qwen/Qwen3-32B") is None


def test_matrix_model_metadata_failure_is_fatal(monkeypatch):
    def fail_metadata(_model):
        raise ValueError("bad model metadata")

    monkeypatch.setattr(support_matrix_module, "_get_test_constraints", fail_metadata)

    with pytest.raises(ModelMetadataResolutionError, match="test/bad-model"):
        _resolve_test_constraints("test/bad-model")


def test_parallel_model_metadata_failure_aborts_without_retry(monkeypatch):
    class MetadataFailureFuture:
        def result(self):
            raise ModelMetadataResolutionError("bad model metadata")

        def cancel(self):
            return True

    class FakeExecutor:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def submit(self, _fn, _combo):
            return MetadataFailureFuture()

    monkeypatch.setattr(support_matrix_module, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(support_matrix_module, "as_completed", lambda futures: list(futures))
    matrix = SupportMatrix.__new__(SupportMatrix)
    combo = ("test/bad-model", "b200_sxm", "trtllm", "1.0")

    with pytest.raises(ModelMetadataResolutionError, match="bad model metadata"):
        matrix._run_parallel_combinations([combo], max_workers=1, pbar=None)


def test_matrix_image_constraints_reach_task_and_replay_command(monkeypatch):
    captured_kwargs = {}

    class FakeTask:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    monkeypatch.setattr(support_matrix_module, "Task", FakeTask)
    constraints = TestConstraints(
        total_gpus=32,
        isl=256,
        osl=256,
        prefix=128,
        ttft=2000.0,
        tpot=50.0,
        image_height=672,
        image_width=960,
        num_images=1,
    )

    SupportMatrix._create_task(
        mode="disagg",
        model="google/gemma-4-26B-A4B",
        system="b200_sxm",
        backend="trtllm",
        version="1.3.0rc20",
        constraints=constraints,
        engine_step_backend=None,
    )
    command = _support_matrix_row_command(
        model="google/gemma-4-26B-A4B",
        system="b200_sxm",
        backend="trtllm",
        version="1.3.0rc20",
        mode="disagg",
        constraints=constraints,
    )

    assert captured_kwargs["image_height"] == 672
    assert captured_kwargs["image_width"] == 960
    assert captured_kwargs["num_images_per_request"] == 1
    assert "--image-height 672 --image-width 960 --num-images 1" in command
