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
        mtp_scaled_tokens=None,
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
    model.forward_model = "op_level"
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


class TestMTPActivationMemoryScaling:
    """MTP activation scaling applies only to the decode-token share."""

    @staticmethod
    def _model():
        return SimpleNamespace(
            context_ops=[SimpleNamespace(get_weights=lambda: 0.0)],
            config=ModelConfig(
                tp_size=1,
                pp_size=1,
                attention_dp_size=1,
                moe_tp_size=1,
                moe_ep_size=1,
            ),
            _num_heads=32,
            _head_size=128,
            _num_experts=0,
            model_family="test",
            get_kvcache_bytes_per_sequence=lambda _seq_len: 0.0,
            _cp_kv_memory_divisor=lambda: 1,
        )

    @staticmethod
    def _database():
        return SimpleNamespace(system_spec={"misc": {"nccl_mem": {1: 0}, "other_mem": 0}})

    def _activations(self, *, nextn: int, num_tokens: int, mtp_scaled_tokens: int | None) -> float:
        model = self._model()
        model.config.nextn = nextn
        return BaseBackend()._get_memory_usage(
            model,
            self._database(),
            batch_size=4,
            beam_width=1,
            isl=65_536,
            osl=400,
            num_tokens=num_tokens,
            mtp_scaled_tokens=mtp_scaled_tokens,
        )["activations"]

    def test_mixed_step_scales_only_decode_share(self):
        num_tokens = 65_536 + 3
        base = self._activations(nextn=0, num_tokens=num_tokens, mtp_scaled_tokens=3)
        spec = self._activations(nextn=3, num_tokens=num_tokens, mtp_scaled_tokens=3)
        assert spec / base == pytest.approx((65_536 + 3 * 4) / num_tokens, rel=1e-6)

    def test_decode_only_step_keeps_full_multiplier(self):
        base = self._activations(nextn=0, num_tokens=512, mtp_scaled_tokens=None)
        spec = self._activations(nextn=3, num_tokens=512, mtp_scaled_tokens=None)
        assert spec / base == pytest.approx(4.0, rel=1e-6)

    def test_prefill_only_step_does_not_scale(self):
        base = self._activations(nextn=0, num_tokens=65_536, mtp_scaled_tokens=0)
        spec = self._activations(nextn=3, num_tokens=65_536, mtp_scaled_tokens=0)
        assert spec == pytest.approx(base, rel=1e-9)


@pytest.mark.parametrize("mode", ["static", "static_ctx", "static_gen"])
@pytest.mark.parametrize("latency_correction_scale", [1.0, 1.25])
def test_run_static_latency_only_matches_run_static_latency(
    monkeypatch,
    backend: BaseBackend,
    model,
    database,
    mode: str,
    latency_correction_scale: float,
) -> None:
    from aiconfigurator.sdk.backends import base_backend as base_backend_module

    def _fake_rust_breakdown(model_arg, database_arg, runtime_config_arg, mode_arg, stride_arg, scale_arg):
        # Mode- and scale-aware like the real bridge: the engine applies the
        # flat correction scale itself and only fills the phases the mode ran.
        ctx = {"context_attention": 11.0 * scale_arg, "logits_gemm": 3.0 * scale_arg}
        gen = {"generation_attention": 2.0 * scale_arg, "generation_mlp": 1.0 * scale_arg}
        if mode_arg == "static_ctx":
            gen = {}
        elif mode_arg == "static_gen":
            ctx = {}
        return (
            ctx,
            gen,
            {name: latency * 10.0 for name, latency in ctx.items()},
            {name: latency * 10.0 for name, latency in gen.items()},
            dict.fromkeys(ctx, "silicon"),
            dict.fromkeys(gen, "silicon"),
        )

    monkeypatch.setattr(base_backend_module, "estimate_static_latency_breakdown_with_rust", _fake_rust_breakdown)
    runtime_config = RuntimeConfig(batch_size=2, beam_width=1, isl=8, osl=5, prefix=2, engine_step_backend="rust")

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
        RuntimeConfig(batch_size=2, beam_width=1, isl=8, osl=1, prefix=2, engine_step_backend="rust"),
        ctx_tokens=8,
    )

    row = summary.get_summary_df().iloc[0]
    assert row["tpot"] > 0.0
    assert row["tokens/s/user"] == 0.0


@pytest.mark.parametrize(
    ("osl", "expected_memory_calls", "expected_activations"),
    [
        (1, [(8, 0)], 0.068359375),
        (5, [(8, 0), (1, None)], 0.2734375),
    ],
)
def test_run_agg_b1_uses_scheduled_activation_peak(
    monkeypatch,
    backend: BaseBackend,
    model,
    database,
    osl: int,
    expected_memory_calls: list[tuple[int, int | None]],
    expected_activations: float,
) -> None:
    """A b=1 agg run retains a decode peak only when decode is scheduled."""
    model._nextn = 3
    model.config.nextn = 3
    model.context_ops = [SimpleNamespace(get_weights=lambda: 0.0)]
    model._num_heads = 32
    model._head_size = 128
    model._num_experts = 0
    model.model_family = "test"
    model.get_kvcache_bytes_per_sequence = lambda _seq_len: 0.0
    model._cp_kv_memory_divisor = lambda: 1
    database.system_spec["misc"] = {"nccl_mem": {1: 0}, "other_mem": 0}
    monkeypatch.setattr(
        backend,
        "run_mixed",
        lambda *args, **kwargs: StepEstimate(latency_ms=1.0, energy_wms=1.0),
    )
    monkeypatch.setattr(
        backend,
        "_get_genonly_step_latency",
        lambda *args, **kwargs: (1.0, 1.0, {"decode": 1.0}, {"decode": "silicon"}),
    )

    memory_calls: list[dict] = []

    def _get_memory_usage(*args, **kwargs):
        memory_calls.append(kwargs)
        return BaseBackend._get_memory_usage(backend, *args, **kwargs)

    monkeypatch.setattr(backend, "_get_memory_usage", _get_memory_usage)

    summary = backend.run_agg(
        model,
        database,
        RuntimeConfig(batch_size=1, beam_width=1, isl=8, osl=osl, engine_step_backend="rust"),
        ctx_tokens=8,
    )

    assert [(call["num_tokens"], call["mtp_scaled_tokens"]) for call in memory_calls] == expected_memory_calls
    assert summary.get_memory()["activations"] == pytest.approx(expected_activations)


def test_run_mixed_returns_components_and_counts_speculative_query_tokens(
    monkeypatch,
    backend: BaseBackend,
    model,
    database,
) -> None:
    from aiconfigurator.sdk.backends import base_backend as base_backend_module

    calls: list[dict] = []

    def _fake_mixed_breakdown(model_arg, database_arg, **kwargs):
        calls.append(kwargs)
        return {
            "latency_ms": 8.5,
            "energy_wms": 85.0,
            "component_latency_ms": {"shared_non_attention": 5.0, "context_attention": 2.0, "decode_attention": 1.5},
            "component_energy_wms": {"shared_non_attention": 50.0, "context_attention": 20.0, "decode_attention": 15.0},
            "per_op_latency_ms": {"context_mlp": 5.0},
            "per_op_source": {"context_mlp": "silicon"},
        }

    monkeypatch.setattr(base_backend_module, "estimate_mixed_step_breakdown_with_rust", _fake_mixed_breakdown)
    model._nextn = 2

    estimate = backend.run_mixed(
        model,
        database,
        RuntimeConfig(isl=8, osl=5, prefix=2, engine_step_backend="rust"),
        MixedStepInput(
            context_tokens=8,
            num_decode_requests=2,
        ),
    )

    assert estimate.num_decode_requests == 2
    # Every decode request verifies one target token plus the two scheduled
    # drafts (nextn=2); scheduling metadata is computed before the bridge call.
    assert estimate.num_decode_query_tokens == 6
    assert estimate.latency_ms == pytest.approx(sum(estimate.component_latency_ms.values()))
    assert set(estimate.component_latency_ms) == {
        "shared_non_attention",
        "context_attention",
        "decode_attention",
    }
    # The bridge receives the RAW decode-request count; the engine applies the
    # (nextn + 1) scaling internally, exactly like the deleted Python passes.
    assert calls[0]["gen_tokens"] == 2
    assert calls[0]["ctx_tokens"] == 8


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
        RuntimeConfig(batch_size=2, beam_width=1, isl=8, osl=5, engine_step_backend="rust"),
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
        RuntimeConfig(batch_size=2, beam_width=1, isl=8, osl=5, engine_step_backend="rust"),
        ctx_tokens=8,
    )
    assert "decode_tokens_per_iteration" not in summary.get_step_estimates()["scheduling"]

    explicit = backend.run_agg(
        model,
        database,
        RuntimeConfig(batch_size=2, beam_width=1, isl=8, osl=5, engine_step_backend="rust"),
        ctx_tokens=8,
        decode_tokens_per_iteration=1.0,
    )
    assert explicit.get_step_estimates()["scheduling"]["decode_tokens_per_iteration"] == 1.0
    # The two calls schedule identically but carry different projection
    # eligibility; they must not have shared a cache entry.
    assert explicit is not summary


@pytest.mark.parametrize(
    ("engine_step_backend", "error", "match"),
    [
        ("auto", ValueError, "unknown engine_step_backend 'auto'"),
        (None, TypeError, "compiled engine is the only aggregate engine-step executor"),
    ],
)
def test_run_agg_validates_backend_before_cache_hit(
    monkeypatch,
    backend: BaseBackend,
    model,
    database,
    engine_step_backend: str | None,
    error: type[Exception],
    match: str,
) -> None:
    """A cached Rust summary must not bypass request/database validation."""
    monkeypatch.delenv("AICONFIGURATOR_ENGINE_STEP_BACKEND", raising=False)
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

    cached = backend.run_agg(
        model,
        database,
        RuntimeConfig(batch_size=2, beam_width=1, isl=8, osl=5, engine_step_backend="rust"),
        ctx_tokens=8,
    )
    assert (
        backend.run_agg(
            model,
            database,
            RuntimeConfig(batch_size=2, beam_width=1, isl=8, osl=5, engine_step_backend="rust"),
            ctx_tokens=8,
        )
        is cached
    )

    with pytest.raises(error, match=match):
        backend.run_agg(
            model,
            database,
            RuntimeConfig(batch_size=2, beam_width=1, isl=8, osl=5, engine_step_backend=engine_step_backend),
            ctx_tokens=8,
        )


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
    monkeypatch,
    backend: BaseBackend,
    model,
    database,
) -> None:
    """Regression: a direct run_mixed call with a plain (text-only isl) config
    must hand the bridge the image-augmented effective isl, exactly as
    run_static / run_agg model it. Text isl=8 plus 16 visual tokens gives
    effective isl 24 — the visual adjustment happens before the bridge call,
    not inside the engine."""
    from aiconfigurator.sdk.backends import base_backend as base_backend_module

    model.encoder_config = _vision_encoder_config()

    calls: list[dict] = []

    def _fake_mixed_breakdown(model_arg, database_arg, **kwargs):
        calls.append(kwargs)
        return {
            "latency_ms": 1.0,
            "energy_wms": 1.0,
            "component_latency_ms": {"shared_non_attention": 1.0},
            "component_energy_wms": {"shared_non_attention": 1.0},
            "per_op_latency_ms": {},
            "per_op_source": {},
        }

    monkeypatch.setattr(base_backend_module, "estimate_mixed_step_breakdown_with_rust", _fake_mixed_breakdown)

    backend.run_mixed(
        model,
        database,
        RuntimeConfig(isl=8, osl=6, num_images_per_request=1, num_image_tokens=16, engine_step_backend="rust"),
        MixedStepInput(context_tokens=24, num_decode_requests=1),
    )

    assert calls[-1]["isl"] == 24
    assert calls[-1]["ctx_tokens"] == 24


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
        engine_step_backend="rust",
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


def test_run_static_latency_only_zeroes_energy_with_paired_keys(
    monkeypatch,
    backend: BaseBackend,
    model,
    database,
) -> None:
    """include_energy=False must zero the energy dicts while keeping their
    key sets identical to the latency dicts (the power coverage gate pairs
    latency and energy by name)."""
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


def test_step_requires_a_real_perf_database(
    backend: BaseBackend,
    model,
    database,
) -> None:
    """The compiled engine resolves perf data from disk by identity, so a
    step over a synthetic database is a hard TypeError now that the Python
    step branch is gone (op-level and FPM models alike)."""
    with pytest.raises(TypeError, match="on-disk identity"):
        backend.run_static(
            model,
            database,
            RuntimeConfig(batch_size=2, beam_width=1, isl=8, osl=5, prefix=2),
            mode="static",
            stride=2,
        )
    with pytest.raises(TypeError, match="on-disk identity"):
        backend.run_mixed(
            model,
            database,
            RuntimeConfig(isl=8, osl=5),
            MixedStepInput(context_tokens=8, num_decode_requests=1),
        )
    with pytest.raises(TypeError, match="on-disk identity"):
        backend._get_genonly_step_latency(
            model,
            database,
            RuntimeConfig(isl=8, osl=5),
            gen_tokens=2,
            isl=8,
            osl=5,
        )
