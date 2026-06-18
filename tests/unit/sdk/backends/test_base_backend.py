# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.backends.base_backend import BaseBackend
from aiconfigurator.sdk.config import ModelConfig, RuntimeConfig

pytestmark = pytest.mark.unit


class _LatencyResult:
    def __init__(self, latency_ms: float, energy_wms: float) -> None:
        self._latency_ms = latency_ms
        self.energy = energy_wms

    def __float__(self) -> float:
        return self._latency_ms


class _StaticOp:
    def __init__(self, name: str, latency_ms: float, energy_wms: float) -> None:
        self._name = name
        self._latency_ms = latency_ms
        self._energy_wms = energy_wms

    def query(self, *args, **kwargs) -> _LatencyResult:
        return _LatencyResult(self._latency_ms, self._energy_wms)


class _TestBackend(BaseBackend):
    def run_agg(self, model, database, runtime_config, **kwargs):
        return super().run_agg(model, database, runtime_config, **kwargs)

    def find_best_agg_result_under_constraints(self, model, database, runtime_config, **kwargs):
        raise NotImplementedError

    def _get_memory_usage(
        self,
        model,
        database,
        batch_size,
        beam_width,
        isl,
        osl,
        num_tokens=0,
        prefix=0,
        encoder_memory=None,
    ) -> dict[str, float]:
        return {"total": 1.0}


@pytest.fixture
def backend() -> BaseBackend:
    return _TestBackend()


@pytest.fixture
def database():
    return SimpleNamespace(
        backend="test-backend",
        version="test-version",
        system="test-system",
        system_spec={"gpu": {"mem_capacity": 80 * (1 << 30)}},
    )


@pytest.fixture
def model():
    model = MagicMock()
    model.model_path = "test-model"
    model.model_name = "test-model"
    model._nextn = 0
    model.encoder_ops = []
    model.context_ops = [
        _StaticOp("context_attention", latency_ms=11.0, energy_wms=110.0),
        _StaticOp("logits_gemm", latency_ms=3.0, energy_wms=30.0),
    ]
    model.generation_ops = [
        _StaticOp("generation_attention", latency_ms=2.0, energy_wms=20.0),
        _StaticOp("generation_mlp", latency_ms=1.0, energy_wms=10.0),
    ]
    model.config = ModelConfig(
        tp_size=1,
        pp_size=1,
        attention_dp_size=1,
        moe_tp_size=1,
        moe_ep_size=1,
        gemm_quant_mode=common.GEMMQuantMode.bfloat16,
        moe_quant_mode=common.MoEQuantMode.bfloat16,
        kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
        fmha_quant_mode=common.FMHAQuantMode.bfloat16,
        comm_quant_mode=common.CommQuantMode.half,
    )
    return model


@pytest.fixture
def runtime_config() -> RuntimeConfig:
    return RuntimeConfig(batch_size=2, beam_width=1, isl=8, osl=5, prefix=2)


@pytest.mark.parametrize("mode", ["static", "static_ctx", "static_gen"])
@pytest.mark.parametrize("latency_correction_scale", [1.0, 1.25])
def test_run_static_latency_only_matches_run_static_latency(
    backend: BaseBackend,
    model,
    database,
    runtime_config: RuntimeConfig,
    mode: str,
    latency_correction_scale: float,
) -> None:
    summary = backend.run_static(
        model,
        database,
        runtime_config,
        mode=mode,
        stride=2,
        latency_correction_scale=latency_correction_scale,
    )
    latency_only = backend.run_static_latency_only(
        model,
        database,
        runtime_config,
        mode=mode,
        stride=2,
        latency_correction_scale=latency_correction_scale,
    )

    summary_latency = sum(summary.get_context_latency_dict().values()) + sum(
        summary.get_generation_latency_dict().values()
    )
    request_latency = float(summary.get_summary_df().iloc[0]["request_latency"])

    assert latency_only == pytest.approx(summary_latency)
    assert latency_only == pytest.approx(request_latency, abs=1e-3)


def test_run_static_can_route_to_rust_engine_step_backend(
    monkeypatch,
    backend: BaseBackend,
    model,
    database,
) -> None:
    from aiconfigurator.sdk.backends import base_backend as base_backend_module

    calls = []

    def _fake_rust_breakdown(model_arg, database_arg, runtime_config_arg, mode_arg, stride_arg, scale_arg):
        calls.append((model_arg, database_arg, runtime_config_arg, mode_arg, stride_arg, scale_arg))
        return (
            {"rust_engine_step_context": 7.0},
            {"rust_engine_step_generation": 3.0},
            {"rust_engine_step_context": "rust"},
            {"rust_engine_step_generation": "rust"},
        )

    monkeypatch.setattr(
        base_backend_module,
        "estimate_static_latency_breakdown_with_rust",
        _fake_rust_breakdown,
    )

    summary = backend.run_static(
        model,
        database,
        RuntimeConfig(batch_size=2, beam_width=1, isl=8, osl=5, prefix=2, engine_step_backend="rust"),
        mode="static",
        stride=2,
        latency_correction_scale=1.25,
    )

    assert len(calls) == 1
    assert calls[0][3:] == ("static", 2, 1.25)
    assert summary.get_context_latency_dict() == {"rust_engine_step_context": 7.0}
    assert summary.get_generation_latency_dict() == {"rust_engine_step_generation": 3.0}
    assert summary.get_context_energy_wms_dict() == {"rust_engine_step_context": 0.0}
    assert summary.get_generation_energy_wms_dict() == {"rust_engine_step_generation": 0.0}
    assert summary.get_context_source_dict() == {"rust_engine_step_context": "rust"}
    assert summary.get_generation_source_dict() == {"rust_engine_step_generation": "rust"}


def test_run_agg_ttft_legacy_correction_is_default(
    monkeypatch,
    backend: BaseBackend,
    model,
    database,
) -> None:
    runtime_config = RuntimeConfig(batch_size=2, beam_width=1, isl=8, osl=5, prefix=2)

    monkeypatch.setattr(
        backend,
        "_run_encoder_phase",
        lambda *_args, **_kwargs: ({}, {}, {}, 0),
    )
    monkeypatch.setattr(
        backend,
        "_get_encoder_component_memory_for_runtime",
        lambda *_args, **_kwargs: {"total": 0.0},
    )
    monkeypatch.setattr(
        backend,
        "_get_mix_step_latency",
        lambda *_args, **_kwargs: (10.0, 0.0, {}, {}),
    )
    monkeypatch.setattr(
        backend,
        "_get_genonly_step_latency",
        lambda *_args, **_kwargs: (5.0, 0.0, {}, {}),
    )

    summary = backend.run_agg(model, database, runtime_config, ctx_tokens=4)
    result = summary.get_result_dict()

    # Expected:
    # llm_ttft = ceil(isl/ctx_tokens) * mix_step = ceil(8/4)*10 = 20ms
    # correction_factor = min(2 + (steps_to_finish_ctx - 3)/2/10, 4)
    # steps_to_finish_ctx = ceil(isl*b/ctx_tokens) = ceil(8*2/4) = 4
    # correction_factor = 2.05
    # ttft = 20 * 2.05 = 41ms
    assert result["ttft"] == pytest.approx(41.0)


def test_run_agg_ttft_md1_fixed_interval_model(
    monkeypatch,
    backend: BaseBackend,
    model,
    database,
) -> None:
    runtime_config = RuntimeConfig(batch_size=2, beam_width=1, isl=8, osl=5, prefix=2)

    monkeypatch.setattr(
        backend,
        "_run_encoder_phase",
        lambda *_args, **_kwargs: ({}, {}, {}, 0),
    )
    monkeypatch.setattr(
        backend,
        "_get_encoder_component_memory_for_runtime",
        lambda *_args, **_kwargs: {"total": 0.0},
    )
    monkeypatch.setattr(
        backend,
        "_get_mix_step_latency",
        lambda *_args, **_kwargs: (10.0, 0.0, {}, {}),
    )
    monkeypatch.setattr(
        backend,
        "_get_genonly_step_latency",
        lambda *_args, **_kwargs: (5.0, 0.0, {}, {}),
    )

    summary = backend.run_agg(
        model,
        database,
        runtime_config,
        ctx_tokens=4,
        ttft_queue_model="md1",
        ttft_wait_base_steps=0.0,
        ttft_request_interval_ms=25.0,
        ttft_wait_queue_scale=1.0,
        ttft_wait_max_queue_steps=10.0,
    )
    result = summary.get_result_dict()

    # Under same scheduling as above:
    # prefill_service_time = mix_step / (ctx_tokens/isl) = 10 / (4/8) = 20ms
    # request interval = 25ms -> lambda=40/s, mu=50/s, rho=0.8
    # M/D/1: Wq = rho/(2*(1-rho))*service = 40ms -> 4 queue steps
    # ttft = llm_ttft(20) + 4*10 = 60ms
    assert result["ttft"] == pytest.approx(60.0)


@pytest.mark.parametrize("request_interval_ms", [None, 0.0, 1000.0])
def test_run_agg_ttft_md1_defaults_to_one_request_per_second(
    monkeypatch,
    backend: BaseBackend,
    model,
    database,
    request_interval_ms,
) -> None:
    runtime_config = RuntimeConfig(batch_size=2, beam_width=1, isl=8, osl=5, prefix=2)

    monkeypatch.setattr(
        backend,
        "_run_encoder_phase",
        lambda *_args, **_kwargs: ({}, {}, {}, 0),
    )
    monkeypatch.setattr(
        backend,
        "_get_encoder_component_memory_for_runtime",
        lambda *_args, **_kwargs: {"total": 0.0},
    )
    monkeypatch.setattr(
        backend,
        "_get_mix_step_latency",
        lambda *_args, **_kwargs: (10.0, 0.0, {}, {}),
    )
    monkeypatch.setattr(
        backend,
        "_get_genonly_step_latency",
        lambda *_args, **_kwargs: (5.0, 0.0, {}, {}),
    )

    kwargs = {
        "ctx_tokens": 4,
        "ttft_queue_model": "md1",
        "ttft_wait_base_steps": 0.0,
        "ttft_wait_queue_scale": 1.0,
        "ttft_wait_max_queue_steps": 10.0,
    }
    if request_interval_ms is not None:
        kwargs["ttft_request_interval_ms"] = request_interval_ms

    summary = backend.run_agg(model, database, runtime_config, **kwargs)
    result = summary.get_result_dict()

    # Public default is 0.0; md1 normalizes omitted or zero interval to the
    # internal 1000ms / 1 req/s fallback, not to this candidate's predicted
    # steady-state throughput.
    # prefill_service_time = 20ms, lambda=1/s, mu=50/s, rho=0.02
    # M/D/1: Wq = 0.02/(2*(1-0.02))*20ms = 0.2040816327ms
    assert result["ttft"] == pytest.approx(20.2040816327)


def test_run_agg_ttft_dd1_has_no_queue_wait_below_capacity(
    monkeypatch,
    backend: BaseBackend,
    model,
    database,
) -> None:
    runtime_config = RuntimeConfig(batch_size=2, beam_width=1, isl=8, osl=5, prefix=2)

    monkeypatch.setattr(
        backend,
        "_run_encoder_phase",
        lambda *_args, **_kwargs: ({}, {}, {}, 0),
    )
    monkeypatch.setattr(
        backend,
        "_get_encoder_component_memory_for_runtime",
        lambda *_args, **_kwargs: {"total": 0.0},
    )
    monkeypatch.setattr(
        backend,
        "_get_mix_step_latency",
        lambda *_args, **_kwargs: (10.0, 0.0, {}, {}),
    )
    monkeypatch.setattr(
        backend,
        "_get_genonly_step_latency",
        lambda *_args, **_kwargs: (5.0, 0.0, {}, {}),
    )

    summary = backend.run_agg(
        model,
        database,
        runtime_config,
        ctx_tokens=4,
        ttft_queue_model="dd1",
        ttft_wait_base_steps=0.0,
        ttft_request_interval_ms=25.0,
        ttft_wait_max_queue_steps=10.0,
    )
    result = summary.get_result_dict()

    # D/D/1 deterministic arrivals: service_time=20ms < interval=25ms,
    # so a no-backlog stream does not accumulate queue waiting.
    assert result["ttft"] == pytest.approx(20.0)


def test_run_agg_ttft_dd1_caps_queue_wait_at_capacity(
    monkeypatch,
    backend: BaseBackend,
    model,
    database,
) -> None:
    runtime_config = RuntimeConfig(batch_size=2, beam_width=1, isl=8, osl=5, prefix=2)

    monkeypatch.setattr(
        backend,
        "_run_encoder_phase",
        lambda *_args, **_kwargs: ({}, {}, {}, 0),
    )
    monkeypatch.setattr(
        backend,
        "_get_encoder_component_memory_for_runtime",
        lambda *_args, **_kwargs: {"total": 0.0},
    )
    monkeypatch.setattr(
        backend,
        "_get_mix_step_latency",
        lambda *_args, **_kwargs: (10.0, 0.0, {}, {}),
    )
    monkeypatch.setattr(
        backend,
        "_get_genonly_step_latency",
        lambda *_args, **_kwargs: (5.0, 0.0, {}, {}),
    )

    summary = backend.run_agg(
        model,
        database,
        runtime_config,
        ctx_tokens=4,
        ttft_queue_model="dd1",
        ttft_wait_base_steps=0.0,
        ttft_request_interval_ms=20.0,
        ttft_wait_max_queue_steps=10.0,
    )
    result = summary.get_result_dict()

    # D/D/1 at capacity has no finite steady-state slack in this coarse model;
    # cap the queue wait instead of reporting an optimistic zero.
    assert result["ttft"] == pytest.approx(120.0)
