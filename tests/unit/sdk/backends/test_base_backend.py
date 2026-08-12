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
        # 6-tuple: (ctx latency, gen latency, ctx energy, gen energy, ctx source, gen source)
        return (
            {"context_qkv_gemm": 4.0, "context_attention": 3.0},
            {"generation_qkv_gemm": 2.0, "generation_attention": 1.0},
            {"context_qkv_gemm": 40.0, "context_attention": 8.0},
            {"generation_qkv_gemm": 12.0, "generation_attention": 5.0},
            {"context_qkv_gemm": "silicon", "context_attention": "empirical"},
            {"generation_qkv_gemm": "silicon", "generation_attention": "mixed"},
        )

    monkeypatch.setattr(
        base_backend_module,
        "estimate_static_latency_breakdown_with_rust",
        _fake_rust_breakdown,
    )

    def _phase_runner_trap(*args, **kwargs):
        raise AssertionError("the rust path must never re-run the Python phase runners (not even for energy)")

    monkeypatch.setattr(backend, "_run_context_phase", _phase_runner_trap)
    monkeypatch.setattr(backend, "_run_generation_phase", _phase_runner_trap)

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
    # Real op-name keys folded from the compiled engine's per-op results.
    assert summary.get_context_latency_dict() == {"context_qkv_gemm": 4.0, "context_attention": 3.0}
    assert summary.get_generation_latency_dict() == {"generation_qkv_gemm": 2.0, "generation_attention": 1.0}
    # Per-op energies are real and DISTINCT per key — not one phase total
    # smeared onto every key, and not recomputed by a Python double-eval.
    assert summary.get_context_energy_wms_dict() == {"context_qkv_gemm": 40.0, "context_attention": 8.0}
    assert summary.get_generation_energy_wms_dict() == {"generation_qkv_gemm": 12.0, "generation_attention": 5.0}
    assert summary.get_context_source_dict() == {"context_qkv_gemm": "silicon", "context_attention": "empirical"}
    assert summary.get_generation_source_dict() == {"generation_qkv_gemm": "silicon", "generation_attention": "mixed"}


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
    """The rust branch maps the bridge's rich dict 1:1 onto ``StepEstimate``:
    real energy, per-component splits, and the Python branch's per-op keys
    (raw non-attention names plus the two literal attention keys) — the
    synthetic "rust_engine_step_mixed" key no longer exists."""
    from aiconfigurator.sdk.backends import base_backend as base_backend_module

    model._nextn = 2
    monkeypatch.setattr(base_backend_module, "should_use_rust_engine_step", lambda *args: True)
    components = {
        "latency_ms": 8.5,
        "energy_wms": 85.0,
        "component_latency_ms": {"shared_non_attention": 5.0, "context_attention": 2.0, "decode_attention": 1.5},
        "component_energy_wms": {"shared_non_attention": 50.0, "context_attention": 20.0, "decode_attention": 15.0},
        "per_op_latency_ms": {"context_mlp": 5.0, "context_attention (scaled)": 2.0, "generation_attention": 1.5},
        "per_op_source": {
            "context_mlp": "silicon",
            "context_attention (scaled)": "empirical",
            "generation_attention": "silicon",
        },
    }
    monkeypatch.setattr(
        base_backend_module,
        "estimate_mixed_step_breakdown_with_rust",
        lambda *args, **kwargs: components,
    )

    estimate = backend.run_mixed(
        model,
        database,
        RuntimeConfig(isl=8, osl=5, engine_step_backend="rust"),
        MixedStepInput(context_tokens=8, num_decode_requests=2),
    )

    assert estimate.latency_ms == 8.5
    assert estimate.energy_wms == 85.0  # real energy, not a 0.0 placeholder
    assert estimate.component_latency_ms == components["component_latency_ms"]
    assert estimate.component_energy_wms == components["component_energy_wms"]
    assert estimate.per_op_latency_ms == components["per_op_latency_ms"]
    assert "context_attention (scaled)" in estimate.per_op_latency_ms
    assert estimate.per_op_source == components["per_op_source"]
    assert estimate.context_tokens == 8
    assert estimate.num_decode_requests == 2
    assert estimate.num_decode_query_tokens == 6


def test_get_genonly_step_latency_rust_path_returns_decode_breakdown_verbatim(
    monkeypatch,
    backend: BaseBackend,
    model,
    database,
) -> None:
    """The rust branch returns ``estimate_decode_step_breakdown_with_rust``
    verbatim — real op names, real per-op energy (no synthetic
    "rust_engine_step_generation" key) — without running the Python static
    step."""
    from aiconfigurator.sdk.backends import base_backend as base_backend_module

    calls = []
    breakdown = (
        3.0,
        24.0,
        {"generation_attention": 2.0, "generation_mlp": 1.0},
        {"generation_attention": "silicon", "generation_mlp": "empirical"},
    )

    def _fake_decode_breakdown(model_arg, database_arg, **kwargs):
        calls.append(kwargs)
        return breakdown

    monkeypatch.setattr(base_backend_module, "estimate_decode_step_breakdown_with_rust", _fake_decode_breakdown)

    def _python_step_trap(*args, **kwargs):
        raise AssertionError("the rust path must not run the Python static step")

    monkeypatch.setattr(backend, "run_static", _python_step_trap)

    result = backend._get_genonly_step_latency(
        model,
        database,
        RuntimeConfig(isl=8, osl=6, gen_seq_imbalance_correction_scale=1.25, engine_step_backend="rust"),
        gen_tokens=4,
        isl=8,
        osl=6,
    )

    assert result == breakdown
    assert calls == [{"gen_tokens": 4, "isl": 8, "osl": 6, "gen_seq_imbalance_correction_scale": 1.25}]


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


def test_run_static_latency_only_skips_python_phase_runners_for_rust_path(
    monkeypatch,
    backend: BaseBackend,
    model,
    database,
) -> None:
    """include_energy=False must not invoke _run_context_phase or
    _run_generation_phase, and must zero the energy dicts while keeping their
    key sets identical to the latency dicts."""
    from aiconfigurator.sdk.backends import base_backend as base_backend_module

    monkeypatch.setattr(
        base_backend_module,
        "estimate_static_latency_breakdown_with_rust",
        lambda *args, **kwargs: (
            {"context_qkv_gemm": 4.0, "context_attention": 3.0},
            {"generation_qkv_gemm": 2.0, "generation_attention": 1.0},
            {"context_qkv_gemm": 40.0, "context_attention": 8.0},
            {"generation_qkv_gemm": 12.0, "generation_attention": 5.0},
            {"context_qkv_gemm": "silicon", "context_attention": "empirical"},
            {"generation_qkv_gemm": "silicon", "generation_attention": "mixed"},
        ),
    )

    ctx_phase = MagicMock(wraps=backend._run_context_phase)
    gen_phase = MagicMock(wraps=backend._run_generation_phase)
    monkeypatch.setattr(backend, "_run_context_phase", ctx_phase)
    monkeypatch.setattr(backend, "_run_generation_phase", gen_phase)

    runtime_config = RuntimeConfig(batch_size=2, beam_width=1, isl=8, osl=5, prefix=2, engine_step_backend="rust")
    latency = backend.run_static_latency_only(
        model,
        database,
        runtime_config,
        mode="static",
        stride=2,
    )
    assert latency == pytest.approx(10.0)

    (
        context_latency,
        context_energy,
        generation_latency,
        generation_energy,
        _,
        _,
    ) = backend._run_static_breakdown(model, database, runtime_config, mode="static", stride=2, include_energy=False)
    # Latency-only callers must not observe energy, but the key sets stay
    # paired with the latency dicts (the power coverage gate pairs by name).
    assert context_energy == {"context_qkv_gemm": 0.0, "context_attention": 0.0}
    assert generation_energy == {"generation_qkv_gemm": 0.0, "generation_attention": 0.0}
    assert context_energy.keys() == context_latency.keys()
    assert generation_energy.keys() == generation_latency.keys()

    # The old rust-path energy compensation re-ran the Python phases "for
    # energy only"; that double evaluation is gone on every rust-path call.
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
