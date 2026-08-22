# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import aiconfigurator.cli.api as cli_api
import aiconfigurator.cli.main as cli_main
from aiconfigurator.cli.api import EstimateResult
from aiconfigurator.sdk.errors import EmpiricalNotImplementedError, PerfDataNotAvailableError

pytestmark = pytest.mark.unit


def _estimate_result(*, database_mode: str) -> EstimateResult:
    latency_scale = 0.5 if database_mode == "SOL" else 1.0
    return EstimateResult(
        ttft=10.0 * latency_scale,
        tpot=2.0 * latency_scale,
        power_w=None,
        isl=128,
        osl=16,
        batch_size=1,
        ctx_tokens=128,
        tp_size=1,
        pp_size=1,
        model_path="Qwen/Qwen3-32B",
        system_name="test-system",
        backend_name="test-backend",
        backend_version="test-version",
        raw={
            "ttft": 10.0 * latency_scale,
            "tpot": 2.0 * latency_scale,
            "request_latency": 40.0 * latency_scale,
            "tokens/s": 100.0,
            "tokens/s/gpu": 100.0,
            "tokens/s/user": 50.0,
            "seq/s": 5.0,
            "concurrency": 2.0,
            "memory": 12.0,
        },
        mode="agg",
        per_ops_data={"mix_step": {"context_attention": 10.0 * latency_scale}},
        per_ops_source={"mix_step": {"context_attention": "silicon"}},
    )


def _estimate_args(cli_parser, *, detail: str = "time"):
    return cli_parser.parse_args(
        [
            "estimate",
            "--model-path",
            "Qwen/Qwen3-32B",
            "--system",
            "test-system",
            "--detail",
            detail,
            "--no-color",
        ]
    )


@pytest.mark.parametrize("detail", ["time", "all"])
@pytest.mark.parametrize("error_type", [PerfDataNotAvailableError, EmpiricalNotImplementedError])
def test_detail_reports_unavailable_sol_comparison_without_failing(
    cli_parser,
    monkeypatch,
    capsys,
    detail: str,
    error_type: type[RuntimeError],
) -> None:
    def estimate(**kwargs):
        if kwargs["database_mode"] == "SOL":
            raise error_type("DeepEP SOL data is unavailable for EP32")
        return _estimate_result(database_mode=kwargs["database_mode"])

    monkeypatch.setattr(cli_api, "cli_estimate", estimate)
    monkeypatch.setattr(cli_main.perf_database, "set_systems_paths", lambda _paths: None)
    args = _estimate_args(cli_parser, detail=detail)

    cli_main.main(args)

    output = capsys.readouterr().out
    assert "Performance Estimate (agg)" in output
    assert f"Detailed Breakdown ({detail})" in output
    assert "SOL comparison unavailable: DeepEP SOL data is unavailable for EP32" in output
    assert "context_attention" in output


def test_time_detail_keeps_sol_comparison_when_available(cli_parser, monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli_api,
        "cli_estimate",
        lambda **kwargs: _estimate_result(database_mode=kwargs["database_mode"]),
    )
    monkeypatch.setattr(cli_main.perf_database, "set_systems_paths", lambda _paths: None)
    args = _estimate_args(cli_parser)

    cli_main.main(args)

    output = capsys.readouterr().out
    assert "SOL comparison unavailable" not in output
    assert "SOL%" in output
    assert "Mix Step (total = 10.000 ms, SOL = 5.000 ms, SOL% = 50.0%)" in output


def test_time_detail_does_not_mask_unexpected_sol_failure(cli_parser, monkeypatch) -> None:
    def estimate(**kwargs):
        if kwargs["database_mode"] == "SOL":
            raise RuntimeError("unexpected SOL bug")
        return _estimate_result(database_mode=kwargs["database_mode"])

    monkeypatch.setattr(cli_api, "cli_estimate", estimate)
    monkeypatch.setattr(cli_main.perf_database, "set_systems_paths", lambda _paths: None)
    args = _estimate_args(cli_parser)

    with pytest.raises(RuntimeError, match="unexpected SOL bug"):
        cli_main.main(args)


def test_time_detail_does_not_downgrade_invalid_primary_estimate(cli_parser, monkeypatch, capsys) -> None:
    def invalid_estimate(**_kwargs):
        raise ValueError("invalid primary configuration")

    monkeypatch.setattr(cli_api, "cli_estimate", invalid_estimate)
    monkeypatch.setattr(cli_main.perf_database, "set_systems_paths", lambda _paths: None)
    args = _estimate_args(cli_parser)

    with pytest.raises(SystemExit, match="invalid primary configuration"):
        cli_main.main(args)

    output = capsys.readouterr().out
    assert "Performance Estimate" not in output
    assert "SOL comparison unavailable" not in output


def test_time_detail_does_not_downgrade_sol_validation_error(cli_parser, monkeypatch, capsys) -> None:
    def estimate(**kwargs):
        if kwargs["database_mode"] == "SOL":
            raise ValueError("invalid SOL configuration")
        return _estimate_result(database_mode=kwargs["database_mode"])

    monkeypatch.setattr(cli_api, "cli_estimate", estimate)
    monkeypatch.setattr(cli_main.perf_database, "set_systems_paths", lambda _paths: None)
    args = _estimate_args(cli_parser)

    with pytest.raises(SystemExit, match="invalid SOL configuration"):
        cli_main.main(args)

    output = capsys.readouterr().out
    assert "SOL comparison unavailable" not in output
