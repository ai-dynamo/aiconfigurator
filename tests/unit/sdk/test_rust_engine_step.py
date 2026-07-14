# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from aiconfigurator.sdk import common, rust_engine_step
from aiconfigurator.sdk.config import ModelConfig, RuntimeConfig

pytestmark = pytest.mark.unit


def test_should_use_rust_engine_step_supports_runtime_config_and_env(monkeypatch) -> None:
    monkeypatch.setenv("AICONFIGURATOR_ENGINE_STEP_BACKEND", "rust")

    assert rust_engine_step.should_use_rust_engine_step(RuntimeConfig())
    assert rust_engine_step.should_use_rust_engine_step(RuntimeConfig(engine_step_backend="rust"))
    assert not rust_engine_step.should_use_rust_engine_step(RuntimeConfig(engine_step_backend="python"))


def _dense_model() -> SimpleNamespace:
    return SimpleNamespace(
        model_path="Test/Dense",
        architecture="LlamaForCausalLM",
        _context_length=4096,
        _nextn=0,
        config=ModelConfig(
            tp_size=1,
            pp_size=1,
            attention_dp_size=2,
            moe_tp_size=1,
            moe_ep_size=1,
            gemm_quant_mode=common.GEMMQuantMode.bfloat16,
            moe_quant_mode=common.MoEQuantMode.bfloat16,
            kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
            fmha_quant_mode=common.FMHAQuantMode.bfloat16,
        ),
    )


def test_static_latency_breakdown_routes_through_engine_handle(monkeypatch) -> None:
    """The static helper maps ``RuntimeConfig`` onto ``EngineHandle.run_static``
    and collapses the scalar phase totals into the synthetic breakdown dicts,
    applying ``latency_correction_scale`` afterwards."""
    calls = []

    class _FakeHandle:
        def run_static(self, **kwargs):
            calls.append(kwargs)
            # (context_ms, generation_ms, total_ms)
            return (10.0, 6.0, 16.0)

    monkeypatch.setattr(rust_engine_step, "_cached_engine_handle", lambda model, database: _FakeHandle())

    model = _dense_model()
    database = SimpleNamespace(system="test_sxm", backend="vllm", version="1.0.0")

    context_latency, generation_latency, context_source, generation_source = (
        rust_engine_step.estimate_static_latency_breakdown_with_rust(
            model,
            database,
            RuntimeConfig(batch_size=2, beam_width=1, isl=8, osl=4, prefix=2),
            mode="static",
            stride=2,
            latency_correction_scale=1.5,
        )
    )

    assert context_latency == {"rust_engine_step_context": 15.0}
    assert generation_latency == {"rust_engine_step_generation": 9.0}
    assert context_source == {"rust_engine_step_context": "rust"}
    assert generation_source == {"rust_engine_step_generation": "rust"}

    # The runtime config is forwarded verbatim (the Rust engine performs the
    # stride quadrature + (nextn+1) scaling internally).
    assert calls[0]["batch_size"] == 2
    assert calls[0]["isl"] == 8
    assert calls[0]["osl"] == 4
    assert calls[0]["prefix"] == 2
    assert calls[0]["mode"] == "static"
    assert calls[0]["stride"] == 2


def test_mixed_and_decode_helpers_pass_raw_step_args(monkeypatch) -> None:
    """The mixed / decode helpers pass raw step args straight to the handle;
    the Rust engine owns the FPM packing."""
    mixed_calls = []
    decode_calls = []

    class _FakeHandle:
        def mixed_step_latency(self, *args, **kwargs):
            mixed_calls.append((args, kwargs))
            return 8.5

        def decode_step_latency(self, *args, **kwargs):
            decode_calls.append((args, kwargs))
            return 9.5

    monkeypatch.setattr(rust_engine_step, "_cached_engine_handle", lambda model, database: _FakeHandle())

    model = _dense_model()
    database = SimpleNamespace(system="test_sxm", backend="vllm", version="1.0.0")

    mixed_ms = rust_engine_step.estimate_mixed_step_latency_with_rust(
        model,
        database,
        ctx_tokens=384,
        gen_tokens=7,
        isl=256,
        osl=256,
        prefix=128,
    )
    decode_ms = rust_engine_step.estimate_decode_step_latency_with_rust(
        model,
        database,
        gen_tokens=7,
        isl=256,
        osl=256,
    )

    assert mixed_ms == 8.5
    assert decode_ms == 9.5
    # Raw step args pass through positionally; the runtime imbalance scales
    # ride as kwargs (default 1.0 when the caller doesn't set them).
    assert mixed_calls == [
        (
            (384, 7, 256, 256, 128),
            {"seq_imbalance_correction_scale": 1.0, "gen_seq_imbalance_correction_scale": 1.0},
        )
    ]
    assert decode_calls == [((7, 256, 256), {"gen_seq_imbalance_correction_scale": 1.0})]


def test_engine_config_json_preserves_moe_specific_quant_mode() -> None:
    model = SimpleNamespace(
        model_path="Test/Moe",
        architecture="GptOssForCausalLM",
        config=ModelConfig(
            tp_size=1,
            pp_size=1,
            attention_dp_size=1,
            moe_tp_size=1,
            moe_ep_size=1,
            gemm_quant_mode=common.GEMMQuantMode.bfloat16,
            moe_quant_mode=common.MoEQuantMode.w4a16_mxfp4,
            kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
            fmha_quant_mode=common.FMHAQuantMode.bfloat16,
        ),
    )
    database = SimpleNamespace(system="test_sxm", backend="vllm", version="1.0.0")

    config = json.loads(rust_engine_step._engine_config_json(model, database))

    assert config["weight_dtype"] == "bfloat16"
    assert config["moe_dtype"] == "w4a16_mxfp4"


def test_configure_data_roots_passes_systems_path_through(tmp_path, monkeypatch) -> None:
    """Rust reads parquet directly, so the wrapper just hands its
    ``AICONFIGURATOR_SYSTEMS_PATH`` through unchanged to the Rust crate."""
    systems_root = tmp_path / "systems"
    systems_root.mkdir(parents=True)
    monkeypatch.setenv("AICONFIGURATOR_SYSTEMS_PATH", str(systems_root))
    rust_engine_step._configure_default_data_roots()
    assert Path(os.environ["AICONFIGURATOR_SYSTEMS_PATH"]) == systems_root


# ---- ForwardPassPerfModel wrapper (PR #1152, re-platformed onto the PyO3 core) ----


def test_normalize_tuning_iterations_handles_convenience_forms() -> None:
    """The wrapper normalizes single FPM / single-iteration / nested inputs to
    the canonical nested-list shape before marshalling to the Rust core."""
    single = {"version": 1}
    assert rust_engine_step._normalize_tuning_iterations(single) == [[single]]
    # A flat list of FPM dicts is one iteration's per-rank list.
    assert rust_engine_step._normalize_tuning_iterations([single, single]) == [[single, single]]
    # An already-nested list passes through.
    nested = [[single], [single]]
    assert rust_engine_step._normalize_tuning_iterations(nested) == nested
    assert rust_engine_step._normalize_tuning_iterations([]) == []


def test_forward_pass_perf_model_regression_marshalling(monkeypatch) -> None:
    """The wrapper marshals FPM dicts to JSON and unwraps the Rust results,
    without needing a native engine (regression-only fake inner)."""
    calls = {"estimate": [], "tune": [], "diag": 0}

    class _FakeInner:
        def estimate_forward_pass_time_ms(self, fpm_json):
            calls["estimate"].append(json.loads(fpm_json))
            return None

        def tune_with_fpms(self, iterations_json):
            calls["tune"].append(json.loads(iterations_json))

        def diagnostics(self):
            calls["diag"] += 1
            return json.dumps({"source": "fallback_regression", "readiness": "insufficient_data"})

        def min_correction_factor(self):
            return None

        def max_correction_factor(self):
            return None

        def avg_correction_factor(self):
            return None

    model = rust_engine_step.RustForwardPassPerfModel(_FakeInner())

    single_fpm = {"version": 1, "scheduled_requests": {"num_prefill_requests": 1, "sum_prefill_tokens": 10}}
    assert model.estimate_forward_pass_time_ms(single_fpm) is None
    # The single dict is marshalled verbatim (the Rust core accepts a bare obj).
    assert calls["estimate"][0] == single_fpm

    model.tune_with_fpms(single_fpm)
    assert calls["tune"][0] == [[single_fpm]]

    model.tune_with_fpms([single_fpm, single_fpm])
    assert calls["tune"][1] == [[single_fpm, single_fpm]]

    assert model.diagnostics()["source"] == "fallback_regression"
    assert model.get_min_correction_factor() is None


@pytest.mark.integration
def test_forward_pass_perf_model_native_end_to_end() -> None:
    """End-to-end native forward-pass model over a real fixture.

    Builds a native model via ``compile_engine`` (crossing into the Rust core),
    estimates a prefill iteration, then tunes with an observation engineered to
    drive the correction factor to exactly 2.0 off the model's own native
    estimate. Requires the compiled ``aiconfigurator_core`` extension.
    """
    pytest.importorskip("aiconfigurator_core")
    from aiconfigurator.sdk.rust_engine_step import RustForwardPassPerfModel

    config = {
        "schema_version": 1,
        "model_name": "Qwen/Qwen3-32B",
        "system_name": "h200_sxm",
        "backend": "trtllm",
        "backend_version": "1.3.0rc10",
        "tp_size": 4,
        "pp_size": 1,
        "moe_tp_size": None,
        "moe_ep_size": None,
        "attention_dp_size": 1,
        "weight_dtype": None,
        "moe_dtype": None,
        "activation_dtype": None,
        "kv_cache_dtype": None,
        "kv_block_size": None,
        "nextn": None,
        "nextn_accept_rates": None,
        "extra": {},
    }
    model = RustForwardPassPerfModel.from_native(config, {"min_observations": 2})

    prefill = [
        {
            "version": 1,
            "scheduled_requests": {
                "num_prefill_requests": 2,
                "sum_prefill_tokens": 2048,
                "sum_prefill_kv_tokens": 0,
            },
        }
    ]
    native_ms = model.estimate_forward_pass_time_ms(prefill)
    assert native_ms is not None and native_ms > 0.0

    assert model.get_min_correction_factor() is None
    assert model.diagnostics()["source"] == "aic"

    obs = [
        {
            "version": 1,
            "wall_time": native_ms * 2.0 / 1000.0,
            "scheduled_requests": {
                "num_prefill_requests": 2,
                "sum_prefill_tokens": 2048,
                "sum_prefill_kv_tokens": 0,
            },
        }
    ]
    model.tune_with_fpms([obs, obs])

    corrected = model.estimate_forward_pass_time_ms(prefill)
    assert corrected == pytest.approx(native_ms * 2.0)
    assert model.get_min_correction_factor() == pytest.approx(2.0)
    assert model.diagnostics()["source"] == "aic_with_correction"


@pytest.mark.integration
def test_forward_pass_perf_model_best_available_falls_back_on_bad_config() -> None:
    """``best_available`` falls back to regression when the native engine cannot
    be compiled (an unknown model), recording the reason in diagnostics."""
    pytest.importorskip("aiconfigurator_core")
    from aiconfigurator.sdk.rust_engine_step import RustForwardPassPerfModel

    config = {
        "schema_version": 1,
        "model_name": "this/model-does-not-exist-xyz",
        "system_name": "h200_sxm",
        "backend": "trtllm",
        "backend_version": "1.3.0rc10",
        "tp_size": 1,
        "pp_size": 1,
        "moe_tp_size": None,
        "moe_ep_size": None,
        "attention_dp_size": 1,
        "weight_dtype": None,
        "moe_dtype": None,
        "activation_dtype": None,
        "kv_cache_dtype": None,
        "kv_block_size": None,
        "nextn": None,
        "nextn_accept_rates": None,
        "extra": {},
    }
    model = RustForwardPassPerfModel.best_available(config, {"min_observations": 2})
    diag = model.diagnostics()
    assert diag["source"] == "fallback_regression"
    assert diag["last_warning"] is not None


def test_sparse_cp_ops_emit_cp_fields_in_spec():
    """Sparse-attention CP is now PORTED to the compiled engine (dsa + dsv4
    _query_cp compositions), so the specs carry cp_size (+ window_size for
    dsv4 HCA) instead of refusing compilation. Both engines compute when the
    sparse tables exist and fail loud identically when they don't -- logical
    parity does not wait for data."""

    class _Dsv4Op:
        _name = "context_attention"
        _scale_factor = 1.0
        _compress_ratio = 4
        _num_heads = 64
        _native_heads = 64
        _tp_size = 1
        _cp_size = 2
        _window_size = 2048
        # Structural dims the emitter forwards for the Rust-side SOL
        # (real ops always carry these via _BaseDeepSeekV4AttentionModule).
        _hidden_size = 7168
        _q_lora_rank = 1536
        _o_lora_rank = 1024
        _head_dim = 512
        _rope_head_dim = 64
        _index_n_heads = 64
        _index_head_dim = 128
        _index_topk = 1024
        _o_groups = 16
        from aiconfigurator.sdk import common

        _kvcache_quant_mode = common.KVCacheQuantMode.fp8
        _kv_cache_dtype = None
        _fmha_quant_mode = common.FMHAQuantMode.bfloat16
        _gemm_quant_mode = common.GEMMQuantMode.fp8_block

    # exercised via the dict builder directly to avoid registry wiring
    from aiconfigurator.sdk.engine import _dsv4_module

    spec = _dsv4_module(_Dsv4Op(), architecture="DeepseekV4ForCausalLM")
    assert spec["cp_size"] == 2
    assert spec["window_size"] == 2048


def test_engine_config_json_identity_disambiguates_collapsed_quant_modes():
    """Two models differing only in a wire-collapsed dtype (sq vs int8_wo both
    -> "int8") or an identity-omitted ModelConfig field (moe_backend) must get
    DISTINCT handle-cache keys — sharing one cached handle silently returns
    the other model's latencies."""
    from aiconfigurator.sdk import common

    def _model(gemm_mode, moe_backend=None):
        cfg = SimpleNamespace(
            tp_size=8,
            pp_size=1,
            moe_tp_size=1,
            moe_ep_size=8,
            attention_dp_size=1,
            cp_size=None,
            gemm_quant_mode=gemm_mode,
            moe_quant_mode=None,
            fmha_quant_mode=None,
            kvcache_quant_mode=None,
            comm_quant_mode=None,
            moe_backend=moe_backend,
            attention_backend=None,
            enable_wideep=False,
            enable_eplb=False,
            wideep_num_slots=None,
            cp_style=None,
            workload_distribution=None,
            overwrite_num_layers=None,
            sms=None,
        )
        return SimpleNamespace(model_path="test/model", architecture=None, config=cfg, _nextn=None)

    database = SimpleNamespace(system="test_sxm", backend="vllm", version="1.0.0")
    key_sq = rust_engine_step._engine_config_json(_model(common.GEMMQuantMode.sq), database)
    key_int8 = rust_engine_step._engine_config_json(_model(common.GEMMQuantMode.int8_wo), database)
    assert key_sq != key_int8, "sq and int8_wo must not alias one cached handle"

    key_deepep = rust_engine_step._engine_config_json(
        _model(common.GEMMQuantMode.sq, moe_backend="deepep_moe"), database
    )
    assert key_sq != key_deepep, "moe_backend must participate in the cache identity"


def test_op_conversion_error_falls_back_to_python_step(monkeypatch):
    """An OpConversionError (op graph not expressible in Rust) must be
    surfaced as RustEngineUnsupportedError, cached per engine identity, and
    caught by the base_backend gates (fallback to the Python step) — NOT
    crash the sweep."""
    import pytest

    from aiconfigurator.sdk.engine import OpConversionError
    from aiconfigurator.sdk.rust_engine_step import RustEngineUnsupportedError

    calls = {"n": 0}

    def _raise_conversion(*args, **kwargs):
        calls["n"] += 1
        raise OpConversionError("unsupported op: ContextMSAModule")

    monkeypatch.setattr("aiconfigurator.sdk.engine.build_engine_spec_json", _raise_conversion)
    rust_engine_step._engine_handle_cache_clear()

    model = _dense_model()
    database = SimpleNamespace(system="test_sxm", backend="vllm", version="1.0.0")

    with pytest.raises(RustEngineUnsupportedError):
        rust_engine_step._cached_engine_handle(model, database)
    # Second call re-raises from the cache without recompiling.
    with pytest.raises(RustEngineUnsupportedError):
        rust_engine_step._cached_engine_handle(model, database)
    assert calls["n"] == 1, "compile failure must be memoized per engine identity"
    rust_engine_step._engine_handle_cache_clear()


def test_wideep_mla_spec_emits_per_rank_heads_not_tp():
    """The WideEP MLA table axis is per-rank heads; the Python query converts
    ``num_heads = 128 // tp_size`` (mla.py). The spec emitter must apply the
    same conversion — emitting raw tp_size makes Rust query the wrong table
    slice (tp=8 would read the heads=8 extrapolation instead of heads=16)."""
    from aiconfigurator.sdk import common
    from aiconfigurator.sdk.engine import _wideep_context_mla, _wideep_generation_mla

    class _WideEpOp:
        _name = "context_attention"
        _scale_factor = 1.0
        _tp_size = 8
        _cp_size = 1
        _kvcache_quant_mode = common.KVCacheQuantMode.fp8
        _fmha_quant_mode = common.FMHAQuantMode.fp8_block
        _attn_backend = "flashinfer"

    ctx_spec = _wideep_context_mla(_WideEpOp())
    gen_spec = _wideep_generation_mla(_WideEpOp())
    assert ctx_spec["num_heads"] == 16  # 128 // 8, NOT tp_size=8
    assert gen_spec["num_heads"] == 16


def test_non_silicon_database_mode_falls_back_to_python_step():
    """The compiled engine is SILICON-only (no util_empirical layer); HYBRID /
    EMPIRICAL databases must stay on the Python step so both backends give
    the SAME answer (parity by delegation) instead of the rust side failing
    configs Python fills in empirically."""
    from enum import Enum

    from aiconfigurator.sdk.config import RuntimeConfig
    from aiconfigurator.sdk.rust_engine_step import should_use_rust_engine_step

    class _Mode(Enum):
        SILICON = "SILICON"
        HYBRID = "HYBRID"

    class _DB:
        def __init__(self, mode):
            self._mode = mode

        def get_default_database_mode(self):
            return self._mode

    rc = RuntimeConfig(engine_step_backend="rust")
    assert should_use_rust_engine_step(rc, _DB(_Mode.SILICON))
    assert not should_use_rust_engine_step(rc, _DB(_Mode.HYBRID))
    assert should_use_rust_engine_step(rc)  # no database context -> unchanged
