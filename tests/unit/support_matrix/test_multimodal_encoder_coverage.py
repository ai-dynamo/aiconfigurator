# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

from aiconfigurator.sdk import common
from tools.support_matrix import support_matrix as support_matrix_module
from tools.support_matrix.compare_support_matrix import check_csv_sanity, read_csv
from tools.support_matrix.support_matrix import (
    STATUS_FAIL,
    STATUS_PASS,
    SUPPORT_MATRIX_HEADER,
    SUPPORT_MATRIX_IMAGE_WORKLOAD,
    SupportMatrix,
    TestConstraints,
    _get_encoder_coverage,
    _image_workload_csv_values,
)

pytestmark = pytest.mark.unit

QWEN3_VL_MODELS = (
    "Qwen/Qwen3-VL-2B-Instruct",
    "Qwen/Qwen3-VL-4B-Instruct",
    "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "Qwen/Qwen3-VL-32B-Instruct",
    "Qwen/Qwen3-VL-32B-Thinking",
    "Qwen/Qwen3-VL-235B-A22B-Instruct",
)


def test_qwen3_vl_acceptance_list_covers_the_default_roster():
    roster_variants = {model for model in common.SupportMatrixHFModels if model.startswith("Qwen/Qwen3-VL-")}

    assert roster_variants == set(QWEN3_VL_MODELS)


def _b200_system_spec() -> dict:
    return {"gpu": {"sm_version": 100, "fp8_tc_flops": 1, "fp4_tc_flops": 1}}


def _patch_constraints(monkeypatch) -> None:
    monkeypatch.setattr(
        support_matrix_module,
        "_get_test_constraints",
        lambda _model: TestConstraints(total_gpus=4, isl=256, osl=256, prefix=128, ttft=1500.0, tpot=50.0),
    )


def _encoder_df(mode: str, *, latency: float = 1.25, memory: float = 0.5) -> pd.DataFrame:
    memory_column = "encoder_memory" if mode == "agg" else "(e)memory"
    return pd.DataFrame({"encoder_latency": [latency, latency], memory_column: [memory, memory]})


@pytest.mark.parametrize("model", QWEN3_VL_MODELS)
def test_all_default_qwen3_vl_variants_have_encoder_implementations_and_image_commands(monkeypatch, model):
    calls = []

    def fake_run_mode(**kwargs):
        calls.append(kwargs)
        return _encoder_df(kwargs["mode"])

    monkeypatch.setattr(SupportMatrix, "_run_mode", staticmethod(fake_run_mode))
    _patch_constraints(monkeypatch)

    statuses, errors, commands, sources = SupportMatrix.run_single_test(
        model=model,
        system="b200_sxm",
        backend="vllm",
        version="0.24.0",
        system_spec=_b200_system_spec(),
        modes_to_test=("agg",),
        include_commands=True,
    )

    coverage = _get_encoder_coverage(model)
    assert coverage.checkpoint_declares_encoder
    assert coverage.aic_encoder_implemented
    assert statuses == {"agg": STATUS_PASS}
    assert errors == {"agg": None}
    assert sources == {"agg": "silicon"}
    assert len(calls) == 1
    assert calls[0]["image_workload"] == SUPPORT_MATRIX_IMAGE_WORKLOAD
    assert "--image-height 1024" in commands["agg"]
    assert "--image-width 1024" in commands["agg"]
    assert "--num-images 1" in commands["agg"]


@pytest.mark.parametrize(
    "model",
    [
        "Qwen/Qwen3-VL-8B-Instruct",
        "Qwen/Qwen3-VL-30B-A3B-Instruct",
    ],
)
def test_dense_and_moe_qwen3_vl_cover_agg_and_disagg_without_text_only_duplicate(monkeypatch, model):
    calls = []

    def fake_run_mode(**kwargs):
        calls.append(kwargs)
        return _encoder_df(kwargs["mode"])

    monkeypatch.setattr(SupportMatrix, "_run_mode", staticmethod(fake_run_mode))
    _patch_constraints(monkeypatch)

    statuses, errors = SupportMatrix.run_single_test(
        model=model,
        system="b200_sxm",
        backend="vllm",
        version="0.24.0",
        system_spec=_b200_system_spec(),
    )

    assert statuses == {"agg": STATUS_PASS, "disagg": STATUS_PASS}
    assert errors == {"agg": None, "disagg": None}
    assert [call["mode"] for call in calls] == ["agg", "disagg"]
    assert all(call["image_workload"] == SUPPORT_MATRIX_IMAGE_WORKLOAD for call in calls)


@pytest.mark.parametrize(
    ("mode", "latency", "memory", "expected_field"),
    [
        ("agg", 0.0, 0.5, "encoder_latency"),
        ("disagg", 1.25, 0.0, "(e)memory"),
    ],
)
def test_encoder_model_cannot_pass_with_zero_encoder_evidence(monkeypatch, mode, latency, memory, expected_field):
    calls = []

    def fake_run_mode(**kwargs):
        calls.append(kwargs)
        return _encoder_df(kwargs["mode"], latency=latency, memory=memory)

    monkeypatch.setattr(SupportMatrix, "_run_mode", staticmethod(fake_run_mode))
    _patch_constraints(monkeypatch)

    statuses, errors = SupportMatrix.run_single_test(
        model="Qwen/Qwen3-VL-8B-Instruct",
        system="b200_sxm",
        backend="vllm",
        version="0.24.0",
        system_spec=_b200_system_spec(),
        modes_to_test=(mode,),
    )

    assert statuses == {mode: STATUS_FAIL}
    assert "ENCODER_NOT_EXERCISED" in errors[mode]
    assert expected_field in errors[mode]
    assert len(calls) == 1


@pytest.mark.parametrize(
    "model",
    [
        "Qwen/Qwen3.5-27B",
        "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "stepfun-ai/Step-3.7-Flash",
    ],
)
def test_declared_but_unimplemented_encoder_is_explicit_failure(monkeypatch, model):
    def fail_if_run(**_kwargs):
        pytest.fail("an unsupported encoder must fail before running the text backbone")

    monkeypatch.setattr(SupportMatrix, "_run_mode", staticmethod(fail_if_run))
    _patch_constraints(monkeypatch)

    statuses, errors, commands, _sources = SupportMatrix.run_single_test(
        model=model,
        system="b200_sxm",
        backend="vllm",
        version="0.24.0",
        system_spec=_b200_system_spec(),
        include_commands=True,
    )

    coverage = _get_encoder_coverage(model)
    assert coverage.checkpoint_declares_encoder
    assert not coverage.aic_encoder_implemented
    assert statuses == {"agg": STATUS_FAIL, "disagg": STATUS_FAIL}
    assert all(error.startswith("ENCODER_UNSUPPORTED:") for error in errors.values())
    for mode, command in commands.items():
        assert "tools/support_matrix/generate_support_matrix.py" in command
        assert f"--mode {mode}" in command
        assert "--no-save" in command
        assert "--expect-status FAIL" in command
        assert "--expect-error-prefix ENCODER_UNSUPPORTED:" in command
        assert "aiconfigurator cli default" not in command


def test_encoder_unsupported_row_persists_a_valid_preflight_replay_command(monkeypatch):
    monkeypatch.setattr(
        SupportMatrix,
        "_run_mode",
        staticmethod(lambda **_kwargs: pytest.fail("unsupported encoder must not run")),
    )
    _patch_constraints(monkeypatch)

    statuses, errors, commands, _sources = SupportMatrix.run_single_test(
        model="Qwen/Qwen3.5-27B",
        system="b200_sxm",
        backend="vllm",
        version="0.24.0",
        modes_to_test=("agg",),
        include_commands=True,
    )
    row = [
        "Qwen/Qwen3.5-27B",
        "Qwen3_5ForConditionalGeneration",
        "b200_sxm",
        "vllm",
        "0.24.0",
        "agg",
        statuses["agg"],
        errors["agg"],
        commands["agg"],
        "",
        "",
        "",
        "",
    ]

    assert check_csv_sanity(SUPPORT_MATRIX_HEADER, [row]) == []


@pytest.mark.parametrize(
    ("status", "command", "expected_error"),
    [
        (
            STATUS_FAIL,
            "uv run aiconfigurator cli default --database-mode SILICON",
            "must invoke tools/support_matrix/generate_support_matrix.py",
        ),
        (
            "HW_INCOMPATIBLE",
            (
                "uv run python tools/support_matrix/generate_support_matrix.py "
                "--model Qwen/Qwen3.5-27B --system b200_sxm --backend vllm "
                "--backend-version 0.24.0 --mode agg --no-save --expect-status FAIL "
                "--expect-error-prefix ENCODER_UNSUPPORTED:"
            ),
            "ENCODER_UNSUPPORTED rows must use status FAIL",
        ),
    ],
)
def test_encoder_unsupported_row_requires_failure_preflight_contract(status, command, expected_error):
    row = [
        "Qwen/Qwen3.5-27B",
        "Qwen3_5ForConditionalGeneration",
        "b200_sxm",
        "vllm",
        "0.24.0",
        "agg",
        status,
        "ENCODER_UNSUPPORTED: no AIC encoder implementation",
        command,
        "",
        "",
        "",
        "",
    ]

    errors = check_csv_sanity(SUPPORT_MATRIX_HEADER, [row])

    assert any(expected_error in error for error in errors)


def test_text_only_model_keeps_existing_workload_and_command(monkeypatch):
    calls = []

    def fake_run_mode(**kwargs):
        calls.append(kwargs)
        return pd.DataFrame({"request_rate": [1.0]})

    monkeypatch.setattr(SupportMatrix, "_run_mode", staticmethod(fake_run_mode))
    _patch_constraints(monkeypatch)

    statuses, _errors, commands, _sources = SupportMatrix.run_single_test(
        model="Qwen/Qwen3-8B",
        system="b200_sxm",
        backend="vllm",
        version="0.24.0",
        system_spec=_b200_system_spec(),
        modes_to_test=("agg",),
        include_commands=True,
    )

    assert statuses == {"agg": STATUS_PASS}
    assert calls[0]["image_workload"] is None
    assert "--image-height" not in commands["agg"]
    assert "--image-width" not in commands["agg"]
    assert "--num-images" not in commands["agg"]
    assert _image_workload_csv_values("Qwen/Qwen3-8B") == ("", "", "")


def test_image_workload_metadata_is_persisted_and_matches_command(tmp_path):
    image_values = _image_workload_csv_values("Qwen/Qwen3-VL-8B-Instruct")
    command = (
        "uv run aiconfigurator cli default --model-path Qwen/Qwen3-VL-8B-Instruct "
        "--database-mode SILICON --image-height 1024 --image-width 1024 --num-images 1"
    )
    row = (
        "Qwen/Qwen3-VL-8B-Instruct",
        "Qwen3VLForConditionalGeneration",
        "b200_sxm",
        "vllm",
        "0.24.0",
        "agg",
        STATUS_PASS,
        None,
        command,
        "silicon",
        *image_values,
    )
    output_file = tmp_path / "support.csv"

    SupportMatrix.__new__(SupportMatrix).save_results_to_csv([row], str(output_file))
    header, rows = read_csv(str(output_file))

    assert header == SUPPORT_MATRIX_HEADER
    assert rows[0][10:13] == ["1024", "1024", "1"]
    assert check_csv_sanity(header, rows) == []


def test_image_command_without_persisted_workload_metadata_is_invalid():
    row = [
        "Qwen/Qwen3-VL-8B-Instruct",
        "Qwen3VLForConditionalGeneration",
        "b200_sxm",
        "vllm",
        "0.24.0",
        "agg",
        STATUS_PASS,
        "",
        (
            "uv run aiconfigurator cli default --database-mode SILICON "
            "--image-height 1024 --image-width 1024 --num-images 1"
        ),
        "silicon",
        "",
        "",
        "",
    ]

    errors = check_csv_sanity(SUPPORT_MATRIX_HEADER, [row])

    assert any("image arguments but image workload metadata is empty" in error for error in errors)


def test_encoder_pass_without_command_or_metadata_evidence_is_invalid():
    row = [
        "Qwen/Qwen3-VL-8B-Instruct",
        "Qwen3VLForConditionalGeneration",
        "b200_sxm",
        "vllm",
        "0.24.0",
        "agg",
        STATUS_PASS,
        "",
        "uv run aiconfigurator cli default --database-mode SILICON",
        "silicon",
        "",
        "",
        "",
    ]

    errors = check_csv_sanity(SUPPORT_MATRIX_HEADER, [row])

    assert any("encoder PASS requires persisted image workload metadata" in error for error in errors)


def test_encoder_pass_with_noncanonical_image_workload_is_invalid():
    row = [
        "Qwen/Qwen3-VL-8B-Instruct",
        "Qwen3VLForConditionalGeneration",
        "b200_sxm",
        "vllm",
        "0.24.0",
        "agg",
        STATUS_PASS,
        "",
        (
            "uv run aiconfigurator cli default --database-mode SILICON "
            "--image-height 448 --image-width 448 --num-images 1"
        ),
        "silicon",
        "448",
        "448",
        "1",
    ]

    errors = check_csv_sanity(SUPPORT_MATRIX_HEADER, [row])

    assert any("must match the canonical encoder workload ('1024', '1024', '1')" in error for error in errors)


def test_text_only_model_cannot_persist_image_workload_metadata():
    row = [
        "Qwen/Qwen3-8B",
        "Qwen3ForCausalLM",
        "b200_sxm",
        "vllm",
        "0.24.0",
        "agg",
        STATUS_PASS,
        "",
        (
            "uv run aiconfigurator cli default --database-mode SILICON "
            "--image-height 1024 --image-width 1024 --num-images 1"
        ),
        "silicon",
        "1024",
        "1024",
        "1",
    ]

    errors = check_csv_sanity(SUPPORT_MATRIX_HEADER, [row])

    assert any("model has no AIC encoder workload" in error for error in errors)


@pytest.mark.parametrize("status", [STATUS_PASS, STATUS_FAIL])
def test_legacy_encoder_row_cannot_be_upgraded_without_image_evidence(tmp_path, status):
    row = (
        "Qwen/Qwen3-VL-8B-Instruct",
        "Qwen3VLForConditionalGeneration",
        "b200_sxm",
        "vllm",
        "0.24.0",
        "agg",
        status,
        None if status == STATUS_PASS else "encoder execution failed",
        "uv run aiconfigurator cli default --database-mode SILICON",
        "silicon" if status == STATUS_PASS else "",
    )

    with pytest.raises(ValueError, match="Legacy multimodal rows cannot be upgraded"):
        SupportMatrix.__new__(SupportMatrix).save_results_to_csv([row], str(tmp_path / "support.csv"))
