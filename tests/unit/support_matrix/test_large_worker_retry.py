# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import shlex

import pandas as pd
import pytest

from tools.support_matrix import generate_support_matrix
from tools.support_matrix import support_matrix as support_matrix_module
from tools.support_matrix.support_matrix import (
    STATUS_FAIL,
    STATUS_PASS,
    EncoderCoverage,
    ModelMetadataError,
    SupportMatrix,
    TestConstraints,
)

pytestmark = pytest.mark.unit


def test_run_single_test_uses_plain_default_task_for_large_moe(monkeypatch):
    calls: list[dict] = []

    def fake_run_mode(**kwargs):
        calls.append(kwargs)
        return pd.DataFrame({"tokens/s/gpu": [1.0]})

    monkeypatch.setattr(SupportMatrix, "_run_mode", staticmethod(fake_run_mode))
    monkeypatch.setattr(
        support_matrix_module,
        "_get_test_constraints",
        lambda _model: TestConstraints(total_gpus=128, isl=256, osl=256, prefix=128, ttft=2_000_000, tpot=50_000),
    )

    statuses, errors = SupportMatrix.run_single_test(
        model="zai-org/GLM-5",
        system="b200_sxm",
        backend="sglang",
        version="0.5.10",
        system_spec={"gpu": {"sm_version": 100, "fp8_tc_flops": 1, "fp4_tc_flops": 1}},
    )

    assert statuses == {"agg": STATUS_PASS, "disagg": STATUS_PASS}
    assert errors == {"agg": None, "disagg": None}
    assert len(calls) == 2
    assert all("yaml_config" not in call for call in calls)


def test_run_single_test_reports_plain_large_moe_replay_command(monkeypatch):
    calls: list[dict] = []

    def fake_run_mode(**kwargs):
        calls.append(kwargs)
        return pd.DataFrame({"tokens/s/gpu": [1.0]})

    monkeypatch.setattr(SupportMatrix, "_run_mode", staticmethod(fake_run_mode))
    monkeypatch.setattr(
        support_matrix_module,
        "_get_test_constraints",
        lambda _model: TestConstraints(total_gpus=128, isl=256, osl=256, prefix=128, ttft=2_000_000, tpot=50_000),
    )

    statuses, errors, commands, _provenance = SupportMatrix.run_single_test(
        model="zai-org/GLM-5",
        system="b200_sxm",
        backend="sglang",
        version="0.5.10",
        modes_to_test=("agg",),
        include_commands=True,
        system_spec={"gpu": {"sm_version": 100, "fp8_tc_flops": 1, "fp4_tc_flops": 1}},
    )

    assert statuses == {"agg": STATUS_PASS}
    assert errors == {"agg": None}
    assert len(calls) == 1
    assert "yaml_config" not in calls[0]
    command_parts = shlex.split(commands["agg"])
    assert "--config-yaml" not in command_parts
    assert "--config-yaml-inline" not in command_parts


def test_run_single_test_can_return_row_replay_commands(monkeypatch):
    def fake_run_mode(**_kwargs):
        return pd.DataFrame({"tokens/s/gpu": [1.0]})

    monkeypatch.setattr(SupportMatrix, "_run_mode", staticmethod(fake_run_mode))
    monkeypatch.setattr(
        support_matrix_module,
        "_get_test_constraints",
        lambda _model: TestConstraints(total_gpus=32, isl=256, osl=256, prefix=128, ttft=2000.0, tpot=50.0),
    )

    statuses, errors, commands, _provenance = SupportMatrix.run_single_test(
        model="zai-org/GLM-5",
        system="b200_sxm",
        backend="sglang",
        version="0.5.10",
        modes_to_test=("agg",),
        include_commands=True,
        system_spec={"gpu": {"sm_version": 100, "fp8_tc_flops": 1, "fp4_tc_flops": 1}},
    )

    assert statuses == {"agg": STATUS_PASS}
    assert errors == {"agg": None}
    assert commands == {
        "agg": (
            "uv run aiconfigurator cli default --model-path zai-org/GLM-5 --total-gpus 32 "
            "--system b200_sxm --backend sglang --backend-version 0.5.10 "
            "--database-mode SILICON --isl 256 --osl 256 --prefix 128 --ttft 2000.0 "
            "--tpot 50.0 --top-n 1 --no-color"
        )
    }


def test_generate_support_matrix_reports_empty_filter(monkeypatch, capsys):
    class EmptySupportMatrix:
        def __init__(self, **_kwargs):
            pass

        def generate_combinations(self):
            return [("Qwen/Qwen3-32B", "h200_sxm", "trtllm", "1.3.0rc10")]

        def test_support_matrix(self, **_kwargs):
            raise AssertionError("empty filters should stop before matrix execution")

    monkeypatch.setattr(generate_support_matrix, "SupportMatrix", EmptySupportMatrix)
    monkeypatch.setattr(
        "sys.argv",
        ["generate_support_matrix.py", "--model", "missing/model", "--no-save"],
    )

    with pytest.raises(SystemExit) as exc:
        generate_support_matrix.main()

    assert exc.value.code == 2
    assert "No support-matrix combinations matched the provided filters." in capsys.readouterr().err


def test_generate_support_matrix_replay_asserts_status_and_error_prefix(monkeypatch, capsys):
    class ReplaySupportMatrix:
        def __init__(self, **_kwargs):
            pass

        def generate_combinations(self):
            return [("Qwen/Qwen3.5-27B", "b200_sxm", "vllm", "0.24.0")]

        def test_support_matrix(self, **_kwargs):
            return [
                (
                    "Qwen/Qwen3.5-27B",
                    "Qwen3_5ForConditionalGeneration",
                    "b200_sxm",
                    "vllm",
                    "0.24.0",
                    "agg",
                    STATUS_FAIL,
                    "ENCODER_UNSUPPORTED: no AIC encoder implementation",
                    "replay",
                    "",
                    "",
                    "",
                    "",
                )
            ]

    monkeypatch.setattr(generate_support_matrix, "SupportMatrix", ReplaySupportMatrix)
    monkeypatch.setattr(
        "sys.argv",
        [
            "generate_support_matrix.py",
            "--model",
            "Qwen/Qwen3.5-27B",
            "--system",
            "b200_sxm",
            "--backend",
            "vllm",
            "--backend-version",
            "0.24.0",
            "--mode",
            "agg",
            "--no-save",
            "--expect-status",
            STATUS_FAIL,
            "--expect-error-prefix",
            "ENCODER_UNSUPPORTED:",
        ],
    )

    generate_support_matrix.main()

    assert "Observed support-matrix result:" in capsys.readouterr().out


def test_metadata_failure_aborts_sequential_retry(monkeypatch):
    combo = ("example/model", "b200_sxm", "vllm", "0.24.0")
    matrix = object.__new__(SupportMatrix)
    matrix.compare_engine_step_backends = False
    matrix.engine_step_comparison_rtol = 0.05
    matrix.engine_step_comparison_atol = 1e-3
    matrix.engine_step_frontier_rtol = 0.75
    matrix.engine_step_frontier_atol = 1e-3
    monkeypatch.setattr(matrix, "get_architecture", lambda _model: "ExampleForCausalLM")
    monkeypatch.setattr(
        matrix,
        "_run_parallel_combinations",
        lambda *_args, **_kwargs: ([], {combo}),
    )
    monkeypatch.setattr(
        support_matrix_module,
        "_get_test_constraints",
        lambda _model: TestConstraints(4, 256, 256, 128, 1500.0, 50.0),
    )
    monkeypatch.setattr(
        support_matrix_module,
        "_get_encoder_coverage",
        lambda _model: EncoderCoverage(False, False, "ExampleForCausalLM"),
    )
    monkeypatch.setattr(
        SupportMatrix,
        "run_single_test",
        staticmethod(lambda **_kwargs: (_ for _ in ()).throw(ModelMetadataError("metadata unavailable"))),
    )
    monkeypatch.setattr(support_matrix_module.perf_database, "unload_database", lambda *_args: None)

    with pytest.raises(ModelMetadataError, match="metadata unavailable"):
        matrix.test_support_matrix(max_workers=1, combinations=[combo], modes_to_test=("agg",))


def test_metadata_failure_from_parallel_worker_is_not_retried(monkeypatch):
    combo = ("example/model", "b200_sxm", "vllm", "0.24.0")

    class FakeProcess:
        terminated = False

        def is_alive(self):
            return True

        def terminate(self):
            self.terminated = True

    process = FakeProcess()

    class FailedFuture:
        def result(self):
            raise ModelMetadataError("metadata unavailable")

        def cancel(self):
            return True

    class FakeExecutor:
        def __init__(self):
            self._processes = {1: process}
            self.shutdown_calls = []

        def submit(self, *_args):
            return FailedFuture()

        def shutdown(self, *, wait, cancel_futures=False):
            self.shutdown_calls.append((wait, cancel_futures))

    class FakeProgress:
        def update(self, _count):
            raise AssertionError("fatal metadata failures must not be counted as completed or retried")

    matrix = object.__new__(SupportMatrix)
    executor = FakeExecutor()
    monkeypatch.setattr(support_matrix_module, "ProcessPoolExecutor", lambda **_kwargs: executor)
    monkeypatch.setattr(support_matrix_module, "as_completed", lambda futures: futures)

    with pytest.raises(ModelMetadataError, match="metadata unavailable"):
        matrix._run_parallel_combinations([combo], max_workers=1, pbar=FakeProgress())

    assert process.terminated
    assert executor.shutdown_calls == [(True, True)]
