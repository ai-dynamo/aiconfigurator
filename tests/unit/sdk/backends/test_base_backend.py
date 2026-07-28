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


class _EnergyAccessTrapResult:
    def __init__(self, latency_ms: float) -> None:
        self._latency_ms = latency_ms

    def __float__(self) -> float:
        return self._latency_ms

    @property
    def energy(self) -> float:
        raise AssertionError("latency-only execution must not access energy")


class _EnergyAccessTrapOp:
    def __init__(self, name: str, latency_ms: float = 1.0) -> None:
        self._name = name
        self._latency_ms = latency_ms

    def query(self, *args, **kwargs) -> _EnergyAccessTrapResult:
        return _EnergyAccessTrapResult(self._latency_ms)


class _TestBackend(BaseBackend):
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
    # Python phase runners supply energy (Rust tracks latency only).
    # Context: (110.0 + 30.0) * scale=1.25 = 175.0
    # Generation: 4 steps * (20.0 + 10.0) * scale=1.25 = 150.0
    assert summary.get_context_energy_wms_dict() == {"rust_engine_step_context": pytest.approx(175.0)}
    assert summary.get_generation_energy_wms_dict() == {"rust_engine_step_generation": pytest.approx(150.0)}
    assert summary.get_context_source_dict() == {"rust_engine_step_context": "rust"}
    assert summary.get_generation_source_dict() == {"rust_engine_step_generation": "rust"}


def test_run_agg_with_osl_one_does_not_divide_by_zero(
    backend: BaseBackend,
    model,
    database,
    monkeypatch,
) -> None:
    """Regression: osl=1 (no-decode) must not raise and tokens/s/user must be 0.0."""
    monkeypatch.setattr(
        backend,
        "_get_mix_step_latency",
        lambda *args, **kwargs: (1.0, 1.0, {}, {}),
    )
    monkeypatch.setattr(
        backend,
        "_get_genonly_step_latency",
        lambda *args, **kwargs: (0.0, 0.0, {}, {}),
    )
    monkeypatch.setattr(
        backend,
        "_get_memory_usage",
        lambda *args, **kwargs: {"total": 1.0},
    )

    summary = backend.run_agg(
        model,
        database,
        RuntimeConfig(batch_size=2, beam_width=1, isl=8, osl=1, prefix=2),
        ctx_tokens=8,
    )

    row = summary.get_summary_df().iloc[0]
    assert row["tpot"] > 0.0
    assert row["tokens/s/user"] == 0.0


def test_mix_step_efficiency_base_default_is_one(backend: BaseBackend) -> None:
    assert backend._mix_step_efficiency(ctx_tokens=4096, gen_tokens=16) == 1.0
    assert backend._mix_step_efficiency(ctx_tokens=4096, gen_tokens=0) == 1.0
    assert backend._mix_step_efficiency(ctx_tokens=0, gen_tokens=0) == 1.0


def test_run_static_latency_only_skips_python_phase_runners_for_rust_path(
    monkeypatch,
    backend: BaseBackend,
    model,
    database,
) -> None:
    """include_energy=False must not invoke _run_context_phase or _run_generation_phase."""
    from aiconfigurator.sdk.backends import base_backend as base_backend_module

    monkeypatch.setattr(
        base_backend_module,
        "estimate_static_latency_breakdown_with_rust",
        lambda *args, **kwargs: (
            {"rust_engine_step_context": 7.0},
            {"rust_engine_step_generation": 3.0},
            {"rust_engine_step_context": "rust"},
            {"rust_engine_step_generation": "rust"},
        ),
    )

    ctx_phase = MagicMock(wraps=backend._run_context_phase)
    gen_phase = MagicMock(wraps=backend._run_generation_phase)
    monkeypatch.setattr(backend, "_run_context_phase", ctx_phase)
    monkeypatch.setattr(backend, "_run_generation_phase", gen_phase)

    backend.run_static_latency_only(
        model,
        database,
        RuntimeConfig(batch_size=2, beam_width=1, isl=8, osl=5, prefix=2, engine_step_backend="rust"),
        mode="static",
        stride=2,
    )

    ctx_phase.assert_not_called()
    gen_phase.assert_not_called()


def test_run_static_latency_only_does_not_access_energy_in_python_fallback(
    monkeypatch,
    backend: BaseBackend,
    model,
    database,
) -> None:
    """Latency-only Python execution must skip encoder, context, and generation energy."""
    from aiconfigurator.sdk.backends import base_backend as base_backend_module

    monkeypatch.setattr(base_backend_module, "should_use_rust_engine_step", lambda *args, **kwargs: False)
    model.encoder_ops = [_EnergyAccessTrapOp("encoder_attention")]
    model.context_ops = [_EnergyAccessTrapOp("context_attention")]
    model.generation_ops = [_EnergyAccessTrapOp("generation_attention")]
    model.encoder_config = common.VisionEncoderConfig(
        depth=1,
        hidden_size=64,
        num_heads=1,
        intermediate_size=128,
        patch_size=16,
        temporal_patch_size=2,
        spatial_merge_size=2,
        out_hidden_size=64,
    )

    latency = backend.run_static_latency_only(
        model,
        database,
        RuntimeConfig(
            batch_size=1,
            beam_width=1,
            isl=8,
            osl=5,
            image_height=32,
            image_width=32,
            num_images_per_request=1,
        ),
        mode="static",
        stride=2,
    )

    assert latency == pytest.approx(6.0)
