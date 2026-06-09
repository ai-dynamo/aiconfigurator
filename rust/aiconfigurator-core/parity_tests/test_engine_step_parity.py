# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke parity checks for Python SDK engine-step latency versus Rust core.

These tests are expected to xfail while the Rust implementation is still
catching up; the xfail message records the measured drift for each surface.
"""

from __future__ import annotations

import contextlib
import io
import json
from dataclasses import dataclass

import pytest

from aiconfigurator.cli.api import cli_estimate
from aiconfigurator.sdk import config, perf_database, rust_engine_step
from aiconfigurator.sdk.models import get_model

pytestmark = pytest.mark.integration


@dataclass(frozen=True)
class EngineStepParityCase:
    model_path: str
    system_name: str = "b200_sxm"
    backend_name: str = "vllm"
    backend_version: str = "0.19.0"
    batch_size: int = 1
    isl: int = 1024
    osl: int = 2
    prefix: int = 0
    tp_size: int = 8
    pp_size: int = 1
    attention_dp_size: int = 1
    moe_tp_size: int = 1
    moe_ep_size: int = 8
    agg_batch_size: int = 2
    agg_ctx_tokens: int | None = None
    disagg_prefill_batch_size: int = 1
    disagg_prefill_num_workers: int = 1
    disagg_decode_batch_size: int = 4
    disagg_decode_num_workers: int = 1


SMOKE_CASES = [
    pytest.param(
        EngineStepParityCase(model_path="MiniMaxAI/MiniMax-M2.5"),
        id="minimax-m25-b200-vllm-019-isl1024-osl2",
    ),
    pytest.param(
        EngineStepParityCase(model_path="moonshotai/Kimi-K2.5"),
        id="kimi-k25-b200-vllm-019-isl1024-osl2",
    ),
    pytest.param(
        EngineStepParityCase(
            model_path="MiniMaxAI/MiniMax-M2.5",
            batch_size=2,
            isl=2048,
            osl=5,
            prefix=256,
        ),
        id="minimax-m25-b200-vllm-019-sampled-prefix",
    ),
]

PARITY_RTOL = 0.01


def _quiet_call(func, *args, **kwargs):
    """Keep interpolation loader chatter out of parity test output."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return func(*args, **kwargs)


def _static_metrics(
    case: EngineStepParityCase,
    *,
    engine_step_backend: str,
    osl: int | None = None,
) -> dict[str, float]:
    kwargs = {
        "model_path": case.model_path,
        "system_name": case.system_name,
        "backend_name": case.backend_name,
        "backend_version": case.backend_version,
        "batch_size": case.batch_size,
        "isl": case.isl,
        "osl": case.osl if osl is None else osl,
        "prefix": case.prefix,
        "tp_size": case.tp_size,
        "pp_size": case.pp_size,
        "attention_dp_size": case.attention_dp_size,
        "moe_tp_size": case.moe_tp_size,
        "moe_ep_size": case.moe_ep_size,
        "stride": 1,
        "engine_step_backend": engine_step_backend,
    }
    ctx = _quiet_call(cli_estimate, mode="static_ctx", **kwargs).summary.get_summary_df().iloc[0]
    gen = _quiet_call(cli_estimate, mode="static_gen", **kwargs).summary.get_summary_df().iloc[0]
    context_ms = float(ctx["context_latency"])
    generation_ms = float(gen["generation_latency"])
    return {
        "context_ms": context_ms,
        "generation_ms": generation_ms,
        "total_ms": context_ms + generation_ms,
    }


def _agg_metrics(case: EngineStepParityCase, *, engine_step_backend: str) -> dict[str, float]:
    result = _quiet_call(
        cli_estimate,
        mode="agg",
        model_path=case.model_path,
        system_name=case.system_name,
        backend_name=case.backend_name,
        backend_version=case.backend_version,
        batch_size=case.agg_batch_size,
        ctx_tokens=case.agg_ctx_tokens or case.isl,
        isl=case.isl,
        osl=case.osl,
        prefix=case.prefix,
        tp_size=case.tp_size,
        pp_size=case.pp_size,
        attention_dp_size=case.attention_dp_size,
        moe_tp_size=case.moe_tp_size,
        moe_ep_size=case.moe_ep_size,
        engine_step_backend=engine_step_backend,
    )
    return {
        "ttft_ms": float(result.ttft),
        "tpot_ms": float(result.tpot),
        "request_latency_ms": float(result.request_latency),
    }


def _disagg_metrics(case: EngineStepParityCase, *, engine_step_backend: str) -> dict[str, float]:
    result = _quiet_call(
        cli_estimate,
        mode="disagg",
        model_path=case.model_path,
        system_name=case.system_name,
        backend_name=case.backend_name,
        backend_version=case.backend_version,
        isl=case.isl,
        osl=case.osl,
        prefix=case.prefix,
        tp_size=case.tp_size,
        pp_size=case.pp_size,
        attention_dp_size=case.attention_dp_size,
        moe_tp_size=case.moe_tp_size,
        moe_ep_size=case.moe_ep_size,
        prefill_batch_size=case.disagg_prefill_batch_size,
        prefill_num_workers=case.disagg_prefill_num_workers,
        decode_batch_size=case.disagg_decode_batch_size,
        decode_num_workers=case.disagg_decode_num_workers,
        engine_step_backend=engine_step_backend,
    )
    return {
        "ttft_ms": float(result.ttft),
        "tpot_ms": float(result.tpot),
        "request_latency_ms": float(result.request_latency),
    }


def _direct_mixed_step_fpm_ms(case: EngineStepParityCase) -> float:
    database = _quiet_call(perf_database.get_database, case.system_name, case.backend_name, case.backend_version)
    if database is None:
        raise RuntimeError(
            f"failed to load perf database for {case.system_name}/{case.backend_name}/{case.backend_version}"
        )

    model_config = config.ModelConfig(
        tp_size=case.tp_size,
        pp_size=case.pp_size,
        attention_dp_size=case.attention_dp_size,
        moe_tp_size=case.moe_tp_size,
        moe_ep_size=case.moe_ep_size,
    )
    model = _quiet_call(get_model, case.model_path, model_config, case.backend_name)
    estimator = rust_engine_step.RustEngineStepEstimator(
        json.loads(rust_engine_step._engine_config_json(model, database))
    )
    metrics = {
        "version": 1,
        "scheduled_requests": {
            "num_prefill_requests": case.batch_size,
            "sum_prefill_tokens": case.batch_size * max(case.isl - case.prefix, 0),
            "sum_prefill_kv_tokens": case.batch_size * case.prefix,
            "num_decode_requests": case.batch_size,
            "sum_decode_kv_tokens": case.batch_size * case.isl,
        },
    }
    try:
        return estimator.forward_pass_time_ms(rust_engine_step._metrics_by_attention_dp_rank(model, metrics))
    finally:
        estimator.close()


def _parity_mismatch_reason(comparisons: dict[str, tuple[float, float]]) -> str | None:
    rows = []
    has_mismatch = False
    metric_width = max([len("metric"), *(len(name) for name in comparisons)])
    for name, (python_value, rust_value) in comparisons.items():
        allowed = max(abs(python_value) * PARITY_RTOL, 1e-9)
        delta = rust_value - python_value
        delta_pct = delta / abs(python_value) * 100 if python_value else float("inf")
        status = "drift" if abs(delta) > allowed else "ok"
        has_mismatch = has_mismatch or status == "drift"
        rows.append(
            f"{name:<{metric_width}} {python_value:>10.3f} {rust_value:>10.3f} "
            f"{delta:>10.3f} {delta_pct:>9.2f}% {PARITY_RTOL * 100:>9.2f}% {status:>6}"
        )
    if not has_mismatch:
        return None
    return "\n".join(
        [
            "parity drift (expected)",
            f"{'metric':<{metric_width}} {'python_ms':>10} {'rust_ms':>10} "
            f"{'delta_ms':>10} {'delta_pct':>10} {'tolerance':>10} {'status':>6}",
            *rows,
        ]
    )


def _static_comparison_metrics(case: EngineStepParityCase) -> dict[str, tuple[float, float]]:
    python_metrics = _static_metrics(case, engine_step_backend="python")
    rust_metrics = _static_metrics(case, engine_step_backend="rust")
    return {
        "static_ctx": (python_metrics["context_ms"], rust_metrics["context_ms"]),
        "static_gen": (python_metrics["generation_ms"], rust_metrics["generation_ms"]),
        "static_total": (python_metrics["total_ms"], rust_metrics["total_ms"]),
    }


def _mixed_step_comparison_metrics(case: EngineStepParityCase) -> dict[str, tuple[float, float]]:
    python_metrics = _static_metrics(case, engine_step_backend="python", osl=2)
    return {
        "mixed_step": (python_metrics["total_ms"], _direct_mixed_step_fpm_ms(case)),
    }


def _agg_comparison_metrics(case: EngineStepParityCase) -> dict[str, tuple[float, float]]:
    python_metrics = _agg_metrics(case, engine_step_backend="python")
    rust_metrics = _agg_metrics(case, engine_step_backend="rust")
    return {
        "agg_ttft": (python_metrics["ttft_ms"], rust_metrics["ttft_ms"]),
        "agg_tpot": (python_metrics["tpot_ms"], rust_metrics["tpot_ms"]),
        "agg_request": (python_metrics["request_latency_ms"], rust_metrics["request_latency_ms"]),
    }


def _disagg_comparison_metrics(case: EngineStepParityCase) -> dict[str, tuple[float, float]]:
    python_metrics = _disagg_metrics(case, engine_step_backend="python")
    rust_metrics = _disagg_metrics(case, engine_step_backend="rust")
    return {
        "disagg_ttft": (python_metrics["ttft_ms"], rust_metrics["ttft_ms"]),
        "disagg_tpot": (python_metrics["tpot_ms"], rust_metrics["tpot_ms"]),
        "disagg_request": (python_metrics["request_latency_ms"], rust_metrics["request_latency_ms"]),
    }


def _prepare_rust_core(monkeypatch: pytest.MonkeyPatch) -> None:
    if not rust_engine_step.is_rust_core_available():
        pytest.skip(
            "Rust core PyO3 extension is required "
            "(install with `pip install -e \".[rust]\"` or `maturin develop --release`)"
        )

    rust_engine_step._import_rust_core.cache_clear()
    rust_engine_step._cached_estimator.cache_clear()


class TestRustEngineStepStaticParity:
    @pytest.mark.parametrize("case", SMOKE_CASES)
    def test_smoke_parity(
        self,
        case: EngineStepParityCase,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _prepare_rust_core(monkeypatch)

        reason = _parity_mismatch_reason(_static_comparison_metrics(case))
        if reason:
            pytest.xfail(reason)


class TestRustEngineStepMixedStepParity:
    @pytest.mark.parametrize("case", SMOKE_CASES)
    def test_smoke_parity(
        self,
        case: EngineStepParityCase,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _prepare_rust_core(monkeypatch)

        reason = _parity_mismatch_reason(_mixed_step_comparison_metrics(case))
        if reason:
            pytest.xfail(reason)


class TestRustEngineStepAggParity:
    @pytest.mark.parametrize("case", SMOKE_CASES)
    def test_smoke_parity(
        self,
        case: EngineStepParityCase,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _prepare_rust_core(monkeypatch)

        reason = _parity_mismatch_reason(_agg_comparison_metrics(case))
        if reason:
            pytest.xfail(reason)


class TestRustEngineStepDisaggParity:
    @pytest.mark.parametrize("case", SMOKE_CASES)
    def test_smoke_parity(
        self,
        case: EngineStepParityCase,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _prepare_rust_core(monkeypatch)

        reason = _parity_mismatch_reason(_disagg_comparison_metrics(case))
        if reason:
            pytest.xfail(reason)
