# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.backends.base_backend import BaseBackend
from aiconfigurator.sdk.config import ModelConfig, RuntimeConfig
from aiconfigurator.sdk.step_estimate import MixedStepInput, StepEstimate

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


def test_run_agg_with_osl_one_does_not_divide_by_zero(
    backend: BaseBackend,
    model,
    database,
    monkeypatch,
) -> None:
    """Regression: osl=1 (no-decode) must not raise and tokens/s/user must be 0.0."""
    monkeypatch.setattr(
        backend,
        "run_mixed",
        lambda *args, **kwargs: StepEstimate(latency_ms=1.0, energy_wms=1.0),
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


def test_run_mixed_returns_components_and_counts_speculative_query_tokens(
    backend: BaseBackend,
    model,
    database,
) -> None:
    calls: list[dict] = []

    class _RecordingOp(_StaticOp):
        def query(self, *args, **kwargs) -> _LatencyResult:
            calls.append(kwargs)
            return super().query(*args, **kwargs)

    model._nextn = 2
    model.context_ops = [
        _StaticOp("context_attention", latency_ms=11.0, energy_wms=110.0),
        _RecordingOp("context_mlp", latency_ms=3.0, energy_wms=30.0),
    ]

    estimate = backend.run_mixed(
        model,
        database,
        RuntimeConfig(isl=8, osl=5, prefix=2),
        MixedStepInput(
            context_tokens=8,
            num_decode_requests=2,
        ),
    )

    assert estimate.num_decode_requests == 2
    assert estimate.num_decode_query_tokens == 6
    assert estimate.latency_ms == pytest.approx(sum(estimate.component_latency_ms.values()))
    assert set(estimate.component_latency_ms) == {
        "shared_non_attention",
        "context_attention",
        "decode_attention",
    }
    # Six new prefill tokens plus two requests verifying three target tokens each.
    assert calls[0]["x"] == 12


def test_run_mixed_rust_path_returns_the_same_structured_contract(
    monkeypatch,
    backend: BaseBackend,
    model,
    database,
) -> None:
    from aiconfigurator.sdk.backends import base_backend as base_backend_module

    model._nextn = 2
    monkeypatch.setattr(base_backend_module, "should_use_rust_engine_step", lambda *args: True)
    monkeypatch.setattr(
        base_backend_module,
        "estimate_mixed_step_breakdown_with_rust",
        lambda *args, **kwargs: {
            "total": 8.5,
            "shared_non_attention": 5.0,
            "context_attention": 2.0,
            "decode_attention": 1.5,
        },
    )

    estimate = backend.run_mixed(
        model,
        database,
        RuntimeConfig(isl=8, osl=5, engine_step_backend="rust"),
        MixedStepInput(context_tokens=8, num_decode_requests=2),
    )

    assert estimate.latency_ms == 8.5
    assert estimate.component_latency_ms == {
        "shared_non_attention": 5.0,
        "context_attention": 2.0,
        "decode_attention": 1.5,
    }
    assert estimate.num_decode_requests == 2
    assert estimate.num_decode_query_tokens == 6


def test_run_agg_applies_speculative_progress_in_scheduler(
    monkeypatch,
    backend: BaseBackend,
    model,
    database,
) -> None:
    model._nextn = 1
    seen_steps: list[MixedStepInput] = []

    def _run_mixed(*args, **kwargs):
        step = args[-1]
        seen_steps.append(step)
        return StepEstimate(
            latency_ms=10.0,
            energy_wms=100.0,
            component_latency_ms={"shared_non_attention": 10.0},
            component_energy_wms={"shared_non_attention": 100.0},
            num_decode_requests=step.num_decode_requests,
            num_decode_query_tokens=step.num_decode_requests * 2,
        )

    monkeypatch.setattr(backend, "run_mixed", _run_mixed)
    monkeypatch.setattr(
        backend,
        "_get_genonly_step_latency",
        lambda *args, **kwargs: (5.0, 50.0, {"decode": 5.0}, {"decode": "silicon"}),
    )

    summary = backend.run_agg(
        model,
        database,
        RuntimeConfig(batch_size=2, beam_width=1, isl=8, osl=5),
        ctx_tokens=8,
        decode_tokens_per_iteration=2.0,
    )
    row = summary.get_summary_df().iloc[0]
    scheduling = summary.get_step_estimates()["scheduling"]

    assert seen_steps[0].num_decode_requests == 1
    assert scheduling["decode_tokens_per_iteration"] == 2.0
    assert scheduling["decode_iterations"] == 3.0
    assert scheduling["num_mix_steps"] == 2.0
    assert scheduling["num_genonly_steps"] == 1.0
    assert row["tpot"] == pytest.approx(4.167)
    assert row["tokens/s"] == pytest.approx(320.0)


def test_run_agg_records_progress_only_when_explicitly_supplied(
    monkeypatch,
    backend: BaseBackend,
    model,
    database,
) -> None:
    """An omitted decode_tokens_per_iteration must leave no scheduler-progress
    marker on the summary: its presence is what tells the upper-layer
    projection to stand down, so recording the 1.0 default would silently
    disable the legacy post-hoc projection flow."""
    monkeypatch.setattr(
        backend,
        "run_mixed",
        lambda *args, **kwargs: StepEstimate(latency_ms=10.0, energy_wms=100.0),
    )
    monkeypatch.setattr(
        backend,
        "_get_genonly_step_latency",
        lambda *args, **kwargs: (5.0, 50.0, {"decode": 5.0}, {"decode": "silicon"}),
    )

    summary = backend.run_agg(
        model,
        database,
        RuntimeConfig(batch_size=2, beam_width=1, isl=8, osl=5),
        ctx_tokens=8,
    )
    assert "decode_tokens_per_iteration" not in summary.get_step_estimates()["scheduling"]

    explicit = backend.run_agg(
        model,
        database,
        RuntimeConfig(batch_size=2, beam_width=1, isl=8, osl=5),
        ctx_tokens=8,
        decode_tokens_per_iteration=1.0,
    )
    assert explicit.get_step_estimates()["scheduling"]["decode_tokens_per_iteration"] == 1.0
    # The two calls schedule identically but carry different projection
    # eligibility; they must not have shared a cache entry.
    assert explicit is not summary


@pytest.mark.parametrize("progress", [0.0, 3.0, float("inf"), float("nan")])
def test_run_agg_rejects_invalid_speculative_progress(
    progress,
    backend: BaseBackend,
    model,
    database,
) -> None:
    model._nextn = 1

    with pytest.raises(ValueError, match="decode_tokens_per_iteration"):
        backend.run_agg(
            model,
            database,
            RuntimeConfig(batch_size=2, beam_width=1, isl=8, osl=5),
            ctx_tokens=8,
            decode_tokens_per_iteration=progress,
        )


def _vision_encoder_config() -> common.VisionEncoderConfig:
    return common.VisionEncoderConfig(
        depth=1,
        hidden_size=8,
        num_heads=1,
        intermediate_size=8,
        patch_size=14,
        temporal_patch_size=1,
        spatial_merge_size=2,
        out_hidden_size=8,
    )


def test_run_mixed_derives_effective_multimodal_isl_for_direct_calls(
    backend: BaseBackend,
    model,
    database,
) -> None:
    """Regression: a direct run_mixed call with a plain (text-only isl) config
    must query attention at the image-augmented effective isl, exactly as
    run_static / run_agg model it. Text isl=8 plus 16 visual tokens gives
    effective isl 24: context attention runs as (batch=ceil(24/24)=1, s=24)
    and decode attention at kv length 24 + osl//2, not the text-only shapes
    (batch=3, s=8) / s=11."""
    model.encoder_config = _vision_encoder_config()

    ctx_attn_calls: list[dict] = []
    gen_attn_calls: list[dict] = []

    class _RecordingCtxAttn(_StaticOp):
        def query(self, *args, **kwargs) -> _LatencyResult:
            ctx_attn_calls.append(kwargs)
            return super().query(*args, **kwargs)

    class _RecordingGenAttn(_StaticOp):
        def query(self, *args, **kwargs) -> _LatencyResult:
            gen_attn_calls.append(kwargs)
            return super().query(*args, **kwargs)

    model.context_ops = [
        _RecordingCtxAttn("context_attention", latency_ms=11.0, energy_wms=110.0),
        _StaticOp("context_mlp", latency_ms=3.0, energy_wms=30.0),
    ]
    model.generation_ops = [
        _RecordingGenAttn("generation_attention", latency_ms=2.0, energy_wms=20.0),
    ]

    backend.run_mixed(
        model,
        database,
        RuntimeConfig(isl=8, osl=6, num_images_per_request=1, num_image_tokens=16),
        MixedStepInput(context_tokens=24, num_decode_requests=1),
    )

    # Pass 2 (context attention): batch = ceil(ctx_tokens / effective_isl).
    pass2 = ctx_attn_calls[-1]
    assert pass2["batch_size"] == 1
    assert pass2["s"] == 24
    # Pass 3 (decode attention): kv length = effective_isl + osl // 2 (+1 for
    # the first decode step inside the static generation phase).
    pass3 = gen_attn_calls[-1]
    assert pass3["s"] == 24 + 6 // 2 + 1


def test_run_agg_does_not_double_count_visual_tokens_in_run_mixed(
    monkeypatch,
    backend: BaseBackend,
    model,
    database,
) -> None:
    """run_mixed owns the visual-token adjustment, so run_agg must hand it the
    unmodified runtime_config (a pre-adjusted copy would count images twice),
    while the genonly step keeps receiving run_agg's effective isl."""
    model.encoder_config = _vision_encoder_config()

    mixed_configs: list[RuntimeConfig] = []

    def _run_mixed(model_arg, database_arg, runtime_config_arg, step):
        mixed_configs.append(runtime_config_arg)
        return StepEstimate(latency_ms=1.0, energy_wms=1.0)

    genonly_isl: list[int] = []

    def _genonly(model_arg, database_arg, runtime_config_arg, num_tokens, isl, osl):
        genonly_isl.append(isl)
        return (1.0, 1.0, {}, {})

    monkeypatch.setattr(backend, "run_mixed", _run_mixed)
    monkeypatch.setattr(backend, "_get_genonly_step_latency", _genonly)

    runtime_config = RuntimeConfig(
        batch_size=2,
        beam_width=1,
        isl=8,
        osl=5,
        num_images_per_request=1,
        num_image_tokens=16,
    )
    backend.run_agg(model, database, runtime_config, ctx_tokens=8)

    assert mixed_configs[0] is runtime_config
    assert genonly_isl == [8 + 16]


def test_mixed_step_requires_context_tokens() -> None:
    with pytest.raises(ValueError, match="context_tokens"):
        MixedStepInput(context_tokens=0, num_decode_requests=1)


def test_mix_step_efficiency_base_default_is_one(backend: BaseBackend) -> None:
    assert backend._mix_step_efficiency(ctx_tokens=4096, gen_tokens=16) == 1.0
    assert backend._mix_step_efficiency(ctx_tokens=4096, gen_tokens=0) == 1.0
    assert backend._mix_step_efficiency(ctx_tokens=0, gen_tokens=0) == 1.0
