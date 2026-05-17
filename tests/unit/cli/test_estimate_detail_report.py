# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiconfigurator.cli.api import EstimateResult
from aiconfigurator.cli.estimate_detail_report import format_estimate_detail_report
from aiconfigurator.sdk.config import RuntimeConfig
from aiconfigurator.sdk.inference_summary import InferenceSummary

pytestmark = pytest.mark.unit


def _estimate_result(
    *,
    mode: str,
    raw: dict,
    summary: InferenceSummary | None = None,
    per_ops_data: dict | None = None,
    per_ops_source: dict | None = None,
) -> EstimateResult:
    return EstimateResult(
        ttft=float(raw.get("ttft", 0.0) or 0.0),
        tpot=float(raw.get("tpot", 0.0) or 0.0),
        power_w=float(raw.get("power_w", 0.0) or 0.0),
        isl=128,
        osl=16,
        batch_size=1,
        ctx_tokens=128,
        tp_size=1,
        pp_size=1,
        model_path="test-model",
        system_name="test-system",
        backend_name="test-backend",
        backend_version="test-version",
        raw=raw,
        mode=mode,
        summary=summary,
        per_ops_data=per_ops_data,
        per_ops_source=per_ops_source,
    )


def _static_summary(context: dict[str, float], generation: dict[str, float]) -> InferenceSummary:
    summary = InferenceSummary(RuntimeConfig(batch_size=1, isl=128, osl=16))
    summary.set_context_latency_dict(context)
    summary.set_generation_latency_dict(generation)
    summary.set_context_source_dict(dict.fromkeys(context, "silicon"))
    summary.set_generation_source_dict(dict.fromkeys(generation, "empirical"))
    return summary


def test_format_estimate_detail_report_static_pairs_sol_summary() -> None:
    result = _estimate_result(
        mode="static",
        raw={"ttft": 10.0, "tpot": 2.0, "request_latency": 40.0},
        summary=_static_summary({"context_qkv_gemm": 10.0}, {"generation_qkv_gemm": 30.0}),
    )
    sol_result = _estimate_result(
        mode="static",
        raw={"ttft": 5.0, "tpot": 1.0, "request_latency": 20.0},
        summary=_static_summary({"context_qkv_gemm": 5.0}, {"generation_qkv_gemm": 15.0}),
    )

    report = format_estimate_detail_report(result, sol_result, detail="time", width=100)

    assert "latency" in report
    assert "SOL%" in report
    assert "request latency" in report
    assert "40.000 ms" in report
    assert "20.000 ms" in report
    assert "Context phase (total = 10.000 ms, SOL = 5.000 ms, SOL% = 50.0%)" in report
    assert "context_qkv_gemm" in report


@pytest.mark.parametrize(
    ("mode", "per_ops_data", "sol_per_ops_data", "expected"),
    [
        (
            "agg",
            {
                "mix_step": {"context_attention": 20.0},
                "genonly_step": {"generation_attention": 10.0},
                "scheduling": {"num_mix_steps": 2.0, "num_genonly_steps": 3.0},
            },
            {"mix_step": {"context_attention": 10.0}, "genonly_step": {"generation_attention": 5.0}},
            (
                "Scheduling: 2 mix steps + 3 gen-only steps",
                "Mix Step (total = 20.000 ms, SOL = 10.000 ms, SOL% = 50.0%)",
            ),
        ),
        (
            "disagg",
            {"prefill": {"context_attention": 40.0}, "decode": {"generation_attention": 8.0}},
            {"prefill": {"context_attention": 10.0}, "decode": {"generation_attention": 4.0}},
            (
                "Prefill (static_ctx) (total = 40.000 ms, SOL = 10.000 ms, SOL% = 25.0%)",
                "Decode (static_gen) (total = 8.000 ms, SOL = 4.000 ms, SOL% = 50.0%)",
            ),
        ),
    ],
)
def test_format_estimate_detail_report_uses_per_ops_data(
    mode: str,
    per_ops_data: dict,
    sol_per_ops_data: dict,
    expected: tuple[str, ...],
) -> None:
    result = _estimate_result(
        mode=mode,
        raw={"ttft": 100.0, "tpot": 10.0, "request_latency": 250.0},
        per_ops_data=per_ops_data,
        per_ops_source={"mix_step": {"context_attention": "silicon"}},
    )
    sol_result = _estimate_result(
        mode=mode,
        raw={"ttft": 50.0, "tpot": 5.0, "request_latency": 125.0},
        per_ops_data=sol_per_ops_data,
    )

    report = format_estimate_detail_report(result, sol_result, detail="time", width=100)

    assert "latency" in report
    assert "SOL%" in report
    for line in expected:
        assert line in report


def test_format_estimate_detail_report_uses_raw_per_ops_source() -> None:
    result = _estimate_result(
        mode="disagg",
        raw={"ttft": 100.0, "tpot": 10.0, "request_latency": 250.0},
        per_ops_source={
            "prefill": {"context_attention": "silicon"},
            "decode": {"generation_attention": "empirical"},
        },
    )

    report = format_estimate_detail_report(result, detail="source")

    assert "Data Source Breakdown (per-op)" in report
    assert "Prefill (static_ctx)" in report
    assert "context_attention" in report
    assert "silicon" in report
    assert "Decode (static_gen)" in report
    assert "generation_attention" in report
    assert "empirical" in report
