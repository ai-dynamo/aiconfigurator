# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import pytest

from aiconfigurator.sdk import common, rust_engine_step
from aiconfigurator.sdk.config import ModelConfig, RuntimeConfig
from aiconfigurator.sdk.performance_result import MoECommFallback

pytestmark = pytest.mark.unit

_CONTEXT_FALLBACK_PAYLOAD = ("context", "deepep_ht", 32, 8, 8, 1)
_GENERATION_FALLBACK_PAYLOAD = ("generation", "deepep_ll", 32, 8, 8, 1)
_CONTEXT_FALLBACK = MoECommFallback(*_CONTEXT_FALLBACK_PAYLOAD)
_GENERATION_FALLBACK = MoECommFallback(*_GENERATION_FALLBACK_PAYLOAD)


def test_should_use_rust_engine_step_supports_runtime_config_and_env(monkeypatch) -> None:
    monkeypatch.setenv("AICONFIGURATOR_ENGINE_STEP_BACKEND", "rust")

    assert rust_engine_step.should_use_rust_engine_step(RuntimeConfig())
    assert rust_engine_step.should_use_rust_engine_step(RuntimeConfig(engine_step_backend="rust"))


def test_engine_step_backend_defaults_to_rust(monkeypatch) -> None:
    """With nothing requested, the compiled engine is the default for a real
    database (one the compiled engine could re-load from disk)."""
    monkeypatch.delenv("AICONFIGURATOR_ENGINE_STEP_BACKEND", raising=False)
    database = _real_database()

    assert rust_engine_step.should_use_rust_engine_step(RuntimeConfig(), database)
    # Unknown values fail closed instead of silently picking an engine.
    with pytest.raises(ValueError, match="engine_step_backend"):
        rust_engine_step.should_use_rust_engine_step(RuntimeConfig(engine_step_backend="auto"), database)


def test_python_backend_value_is_removed(monkeypatch) -> None:
    """The deprecated ``"python"`` no-op completed its one-release window
    (deprecation-cleanup PR): the value now fails closed like any other
    unknown backend, from both the config and the environment."""
    monkeypatch.delenv("AICONFIGURATOR_ENGINE_STEP_BACKEND", raising=False)
    database = _real_database()

    with pytest.raises(ValueError, match=r"unknown engine_step_backend 'python'"):
        rust_engine_step.should_use_rust_engine_step(RuntimeConfig(engine_step_backend="python"), database)

    monkeypatch.setenv("AICONFIGURATOR_ENGINE_STEP_BACKEND", "python")
    with pytest.raises(ValueError, match=r"unknown engine_step_backend 'python'"):
        rust_engine_step.should_use_rust_engine_step(RuntimeConfig(), database)


def _real_database():
    """A real ``PerfDatabase`` instance (loader bypassed): default routing
    requires the real type — synthetic database doubles delegate to the
    Python step."""
    from aiconfigurator_core.sdk.perf_database import PerfDatabase

    database = PerfDatabase.__new__(PerfDatabase)
    database.system = "routing_probe"
    database.backend = "vllm"
    database.version = "1.0.0"
    database._default_database_mode = common.DatabaseMode.SILICON
    return database


def test_default_routing_rust_routes_power_carrying_databases(monkeypatch) -> None:
    """Per-op energy crosses the FFI now: power-carrying PerfDatabases rust-
    route by default and the filesystem power probe is gone entirely — default
    routing never scans the data tree for power columns."""
    monkeypatch.delenv("AICONFIGURATOR_ENGINE_STEP_BACKEND", raising=False)

    # Contract pin: the power carve-out and its probe machinery are deleted.
    assert not hasattr(rust_engine_step, "_database_has_power_data")
    assert not hasattr(rust_engine_step, "_scan_for_power_columns")
    assert not hasattr(rust_engine_step, "_POWER_DATA_CACHE")

    # A real SILICON-mode PerfDatabase rust-routes by default, power data or not.
    assert rust_engine_step.should_use_rust_engine_step(RuntimeConfig(), _real_database())
    assert rust_engine_step.should_use_rust_engine_step(RuntimeConfig(engine_step_backend="rust"), _real_database())
    # Synthetic database doubles have no on-disk identity the compiled engine
    # could resolve: default routing delegates them to the Python step.
    assert not rust_engine_step.should_use_rust_engine_step(
        RuntimeConfig(), SimpleNamespace(system="mock", backend="vllm", version="1.0.0")
    )


@pytest.fixture
def _handle_cache_harness(monkeypatch):
    """Drive ``_cached_engine_handle`` end-to-end with the compile tail
    stubbed at its import sources: models compile to sentinel handles (or an
    ``OpConversionError`` when ``model.fail``), so cache-hit recency, negative
    entries, and eviction are exercised through the production code path."""
    import aiconfigurator_core
    from aiconfigurator_core.sdk import engine as core_engine

    monkeypatch.setattr(rust_engine_step, "_ENGINE_HANDLE_CACHE", OrderedDict())
    monkeypatch.setattr(rust_engine_step, "_ENGINE_HANDLE_CACHE_MAX", 2)
    monkeypatch.setattr(rust_engine_step, "_configure_default_data_roots", lambda: None)
    monkeypatch.setattr(rust_engine_step, "_engine_config_json", lambda model, database: model.key)

    compiles: list[str] = []

    def fake_build(model, **kwargs):
        if getattr(model, "fail", False):
            raise core_engine.OpConversionError(f"cannot express {model.key}")
        compiles.append(model.key)
        return "{}"

    class _FakeHandle:
        def __init__(self, spec_bytes, systems_path=None) -> None:
            self.spec_bytes = spec_bytes

    monkeypatch.setattr(core_engine, "build_engine_spec_json", fake_build)
    monkeypatch.setattr(core_engine, "EngineHandle", _FakeHandle)
    monkeypatch.setattr(aiconfigurator_core, "engine_spec_bincode_from_json", lambda spec: b"")

    def model(key: str, *, fail: bool = False) -> SimpleNamespace:
        return SimpleNamespace(key=key, model_path=key, _nextn=None, fail=fail)

    database = SimpleNamespace(system="test-system", backend="vllm", version="1.0.0")
    return model, database, compiles


def test_cached_engine_handle_hits_and_evicts_least_recently_used(_handle_cache_harness) -> None:
    """Cache-hit recency and LRU eviction, through ``_cached_engine_handle``:
    a hit refreshes recency, so inserting past the cap evicts the least
    recently USED identity (recompiled on re-visit), not insertion order."""
    model, database, compiles = _handle_cache_harness

    m_a, m_b, m_c = model("a"), model("b"), model("c")
    handle_a = rust_engine_step._cached_engine_handle(m_a, database)
    rust_engine_step._cached_engine_handle(m_b, database)
    assert rust_engine_step._cached_engine_handle(m_a, database) is handle_a  # hit refreshes "a"
    assert compiles == ["a", "b"]

    rust_engine_step._cached_engine_handle(m_c, database)  # cap 2: evicts "b"
    rust_engine_step._cached_engine_handle(m_a, database)  # still cached
    rust_engine_step._cached_engine_handle(m_b, database)  # evicted -> recompiles
    assert compiles == ["a", "b", "c", "b"]


def test_cached_engine_handle_negative_entries_raise_fresh_errors(_handle_cache_harness) -> None:
    """An unsupported graph is remembered without re-walking it, but each hit
    raises a FRESH ``RustEngineUnsupportedError`` — caching the raised
    instance would pin model/database via ``__cause__`` and grow its traceback
    on every re-raise."""
    model, database, compiles = _handle_cache_harness

    bad = model("bad", fail=True)
    with pytest.raises(rust_engine_step.RustEngineUnsupportedError) as first:
        rust_engine_step._cached_engine_handle(bad, database)
    with pytest.raises(rust_engine_step.RustEngineUnsupportedError) as second:
        rust_engine_step._cached_engine_handle(bad, database)

    assert str(first.value) == str(second.value) == "cannot express bad"
    assert first.value is not second.value
    assert second.value.__cause__ is None  # cache hit: no pinned compile context
    assert compiles == []  # the op graph was walked once, never re-walked


def test_clear_all_op_caches_drops_engine_handles(_handle_cache_harness) -> None:
    from aiconfigurator.sdk.operations import clear_all_op_caches

    model, database, compiles = _handle_cache_harness
    rust_engine_step._cached_engine_handle(model("a"), database)
    assert rust_engine_step._ENGINE_HANDLE_CACHE

    clear_all_op_caches()
    assert not rust_engine_step._ENGINE_HANDLE_CACHE


def test_cached_engine_handle_mirrors_database_systems_root(_handle_cache_harness, monkeypatch) -> None:
    """The compiled engine must resolve the system yaml from the root the
    paired database actually matched (multi-root ``--systems-paths``), not the
    process-wide env default; the env is only the fallback for duck-typed
    databases without a ``systems_root``."""
    from aiconfigurator_core.sdk import engine as core_engine

    model, database, compiles = _handle_cache_harness
    captured: list = []

    def capturing_build(model, **kwargs):
        captured.append(kwargs["systems_path"])
        return "{}"

    monkeypatch.setattr(core_engine, "build_engine_spec_json", capturing_build)
    monkeypatch.setenv("AICONFIGURATOR_SYSTEMS_PATH", "/env/root")

    database.systems_root = "/custom/root"
    rust_engine_step._cached_engine_handle(model("mirrored"), database)
    assert captured[-1] == "/custom/root"

    del database.systems_root
    rust_engine_step._cached_engine_handle(model("fallback"), database)
    assert captured[-1] == "/env/root"


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


def test_fold_per_op_accumulates_duplicate_names_and_merges_sources() -> None:
    """Duplicate op names fold with ``+=`` (latency AND energy); agreeing
    sources keep their tag while mismatched sources merge to ``"mixed"``;
    ``scale`` multiplies latency and energy per entry. The three dicts share
    one key set (the power-coverage gate pairs latency and energy by name)."""
    entries = [
        ("gemm", 2.0, 10.0, "silicon"),
        ("gemm", 3.0, 5.0, "empirical"),
        ("attention", 1.0, 0.5, "silicon"),
        ("attention", 1.0, 0.5, "silicon"),
    ]

    latency, energy, source, fallbacks = rust_engine_step._fold_per_op(entries)
    assert latency == {"gemm": 5.0, "attention": 2.0}
    assert energy == {"gemm": 15.0, "attention": 1.0}
    assert source == {"gemm": "mixed", "attention": "silicon"}
    assert fallbacks == ()
    assert latency.keys() == energy.keys() == source.keys()

    scaled_latency, scaled_energy, _, _ = rust_engine_step._fold_per_op(entries, scale=2.0)
    assert scaled_latency == {"gemm": 10.0, "attention": 4.0}
    assert scaled_energy == {"gemm": 30.0, "attention": 2.0}


def test_fold_per_op_preserves_every_ordered_fallback_record() -> None:
    entries = [
        (
            "moe_dispatch",
            2.0,
            0.0,
            "estimated",
            [_GENERATION_FALLBACK_PAYLOAD, _CONTEXT_FALLBACK_PAYLOAD, _GENERATION_FALLBACK_PAYLOAD],
        ),
        ("attention", 1.0, 0.0, "silicon", None),
    ]

    _, _, _, fallbacks = rust_engine_step._fold_per_op(entries)

    assert fallbacks == (_CONTEXT_FALLBACK, _GENERATION_FALLBACK)


def test_fold_per_op_reads_inline_first_fallback_metadata() -> None:
    entries = [
        (
            "moe_dispatch",
            2.0,
            0.0,
            "estimated",
            (
                _GENERATION_FALLBACK_PAYLOAD,
                [_CONTEXT_FALLBACK_PAYLOAD, _GENERATION_FALLBACK_PAYLOAD],
            ),
        )
    ]

    _, _, _, fallbacks = rust_engine_step._fold_per_op(entries)

    assert fallbacks == (_CONTEXT_FALLBACK, _GENERATION_FALLBACK)


def test_static_latency_breakdown_routes_through_engine_handle(monkeypatch) -> None:
    """The static helper maps ``RuntimeConfig`` onto
    ``EngineHandle._run_static_per_op_with_metadata`` and folds the per-op
    entries into real-name latency / energy / source dicts, applying
    ``latency_correction_scale`` per key to latency AND energy."""
    calls = []

    class _FakeHandle:
        def _run_static_per_op_with_metadata(self, **kwargs):
            calls.append(kwargs)
            return (
                [
                    ("context_qkv_gemm", 4.0, 40.0, "silicon"),
                    ("context_moe_dispatch", 6.0, 0.0, "estimated", _CONTEXT_FALLBACK_PAYLOAD),
                ],
                [("generation_moe_dispatch", 6.0, 12.0, "estimated", _GENERATION_FALLBACK_PAYLOAD)],
            )

        def last_provenance(self):
            return None  # pure-silicon answer

    monkeypatch.setattr(rust_engine_step, "_cached_engine_handle", lambda model, database: _FakeHandle())

    model = _dense_model()
    database = SimpleNamespace(system="test_sxm", backend="vllm", version="1.0.0")

    (
        context_latency,
        generation_latency,
        context_energy,
        generation_energy,
        context_source,
        generation_source,
        fallbacks,
    ) = rust_engine_step.estimate_static_latency_breakdown_with_rust(
        model,
        database,
        RuntimeConfig(batch_size=2, beam_width=1, isl=8, osl=4, prefix=2),
        mode="static",
        stride=2,
        latency_correction_scale=1.5,
    )

    # Real op names, latency AND energy scaled by 1.5 per key.
    assert context_latency == {"context_qkv_gemm": 6.0, "context_moe_dispatch": 9.0}
    assert generation_latency == {"generation_moe_dispatch": 9.0}
    assert context_energy == {"context_qkv_gemm": 60.0, "context_moe_dispatch": 0.0}
    assert generation_energy == {"generation_moe_dispatch": 18.0}
    # Real provenance tags cross the FFI (no synthetic "rust" source).
    assert context_source == {"context_qkv_gemm": "silicon", "context_moe_dispatch": "estimated"}
    assert generation_source == {"generation_moe_dispatch": "estimated"}
    assert fallbacks == (_CONTEXT_FALLBACK, _GENERATION_FALLBACK)

    # The runtime config is forwarded verbatim (the Rust engine performs the
    # stride quadrature + (nextn+1) scaling internally).
    assert calls[0]["batch_size"] == 2
    assert calls[0]["isl"] == 8
    assert calls[0]["osl"] == 4
    assert calls[0]["prefix"] == 2
    assert calls[0]["beam_width"] == 1
    assert calls[0]["seq_imbalance_correction_scale"] == 1.0
    assert calls[0]["gen_seq_imbalance_correction_scale"] == 1.0
    assert calls[0]["mode"] == "static"
    assert calls[0]["stride"] == 2


def test_mixed_and_decode_helpers_pass_raw_step_args(monkeypatch) -> None:
    """The mixed / decode helpers pass raw step args straight to the handle;
    the Rust engine owns the FPM packing."""
    mixed_calls = []
    breakdown_calls = []
    decode_calls = []

    class _FakeHandle:
        def mixed_step_latency(self, *args, **kwargs):
            mixed_calls.append((args, kwargs))
            return 8.5

        def _mixed_step_breakdown_per_op_with_metadata(self, *args, **kwargs):
            breakdown_calls.append((args, kwargs))
            return (
                [("context_mlp", 5.0, 50.0, "estimated", _CONTEXT_FALLBACK_PAYLOAD)],
                [("context_attention", 2.0, 20.0, "silicon")],
                [("generation_attention", 1.5, 15.0, "silicon")],
            )

        def decode_step_latency(self, *args, **kwargs):
            decode_calls.append((args, kwargs))
            return 9.5

        def last_provenance(self):
            return None  # pure-silicon answer

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

    assert rust_engine_step.estimate_mixed_step_breakdown_with_rust(
        model,
        database,
        ctx_tokens=384,
        gen_tokens=7,
        isl=256,
        osl=256,
        prefix=128,
    ) == {
        "latency_ms": 8.5,
        "energy_wms": 85.0,
        "component_latency_ms": {"shared_non_attention": 5.0, "context_attention": 2.0, "decode_attention": 1.5},
        "component_energy_wms": {"shared_non_attention": 50.0, "context_attention": 20.0, "decode_attention": 15.0},
        "per_op_latency_ms": {"context_mlp": 5.0, "context_attention (scaled)": 2.0, "generation_attention": 1.5},
        "per_op_source": {
            "context_mlp": "estimated",
            "context_attention (scaled)": "silicon",
            "generation_attention": "silicon",
        },
        "moe_comm_fallbacks": (_CONTEXT_FALLBACK,),
    }
    assert breakdown_calls == [
        (
            (384, 7, 256, 256, 128),
            {"seq_imbalance_correction_scale": 1.0, "gen_seq_imbalance_correction_scale": 1.0},
        )
    ]


def test_mixed_step_breakdown_bridge_shapes_missing_attention_passes(monkeypatch) -> None:
    """The mixed bridge folds the three per-op passes into the Python branch's
    ``StepEstimate`` shape: raw non-attention names plus the two literal
    attention keys. A missing pass (empty decode list here) folds to 0.0 under
    the Python branch's default ``"silicon"`` source."""

    class _FakeHandle:
        def _mixed_step_breakdown_per_op_with_metadata(self, *args, **kwargs):
            return (
                [("qkv_gemm", 3.0, 30.0, "silicon"), ("mlp", 2.0, 0.0, "empirical")],
                [("context_attention", 4.0, 8.0, "empirical")],
                [],  # no decode requests scheduled: pass 3 is absent
            )

        def last_provenance(self):
            return None

    monkeypatch.setattr(rust_engine_step, "_cached_engine_handle", lambda model, database: _FakeHandle())

    components = rust_engine_step.estimate_mixed_step_breakdown_with_rust(
        _dense_model(),
        SimpleNamespace(system="test_sxm", backend="vllm", version="1.0.0"),
        ctx_tokens=384,
        gen_tokens=0,
        isl=256,
        osl=256,
        prefix=0,
    )

    assert components["per_op_latency_ms"] == {
        "qkv_gemm": 3.0,
        "mlp": 2.0,
        "context_attention (scaled)": 4.0,
        "generation_attention": 0.0,
    }
    assert components["per_op_source"] == {
        "qkv_gemm": "silicon",
        "mlp": "empirical",
        "context_attention (scaled)": "empirical",
        "generation_attention": "silicon",
    }
    assert components["component_latency_ms"] == {
        "shared_non_attention": 5.0,
        "context_attention": 4.0,
        "decode_attention": 0.0,
    }
    assert components["component_energy_wms"] == {
        "shared_non_attention": 30.0,
        "context_attention": 8.0,
        "decode_attention": 0.0,
    }
    assert components["latency_ms"] == 9.0
    assert components["energy_wms"] == 38.0


def test_decode_step_breakdown_folds_per_op_entries(monkeypatch) -> None:
    """The decode bridge returns latency, energy, per-op data, and executed
    fallback provenance from ``EngineHandle.decode_step_per_op``."""
    calls = []

    class _FakeHandle:
        def _decode_step_per_op_with_metadata(self, *args, **kwargs):
            calls.append((args, kwargs))
            return [
                ("generation_attention", 2.0, 20.0, "silicon"),
                ("generation_mlp", 1.0, 4.0, "estimated", _GENERATION_FALLBACK_PAYLOAD),
            ]

        def last_provenance(self):
            return None

    monkeypatch.setattr(rust_engine_step, "_cached_engine_handle", lambda model, database: _FakeHandle())

    latency_ms, energy_wms, per_op_latency, per_op_source, fallbacks = (
        rust_engine_step.estimate_decode_step_breakdown_with_rust(
            _dense_model(),
            SimpleNamespace(system="test_sxm", backend="vllm", version="1.0.0"),
            gen_tokens=7,
            isl=256,
            osl=256,
        )
    )

    assert latency_ms == 3.0
    assert energy_wms == 24.0
    assert per_op_latency == {"generation_attention": 2.0, "generation_mlp": 1.0}
    assert per_op_source == {"generation_attention": "silicon", "generation_mlp": "estimated"}
    assert fallbacks == (_GENERATION_FALLBACK,)
    assert calls == [((7, 256, 256), {"gen_seq_imbalance_correction_scale": 1.0})]


def test_evaluate_op_helpers_forward_args_and_return_entries_verbatim(monkeypatch) -> None:
    """The thin op-list evaluation FFI: indices / ops JSON and the resolved
    shape kwargs pass straight to the handle and the raw entry list comes back
    verbatim (the A/F-partition orchestration stays in Python)."""
    entries = [("context_qkv_gemm", 1.0, 2.0, "silicon"), ("context_mlp", 0.5, 1.0, "empirical")]
    calls = []

    class _FakeHandle:
        def evaluate_context_ops(self, indices, **kwargs):
            calls.append(("context", indices, kwargs))
            return entries

        def evaluate_generation_ops(self, indices, **kwargs):
            calls.append(("generation", indices, kwargs))
            return entries

        def evaluate_ops_json(self, ops_json, **kwargs):
            calls.append(("json", ops_json, kwargs))
            return entries

        def last_provenance(self):
            return None

    monkeypatch.setattr(rust_engine_step, "_cached_engine_handle", lambda model, database: _FakeHandle())

    model = _dense_model()
    database = SimpleNamespace(system="test_sxm", backend="vllm", version="1.0.0")

    result = rust_engine_step.evaluate_context_ops_with_rust(
        model,
        database,
        indices=(0, 2, 3),
        batch_size=4,
        s=128,
        prefix=16,
        seq_imbalance_correction_scale=0.0,  # explicit 0.0 must NOT clobber to 1.0
    )
    assert result is entries
    assert calls[0] == (
        "context",
        [0, 2, 3],
        {"batch_size": 4, "s": 128, "prefix": 16, "seq_imbalance_correction_scale": 0.0, "x": None},
    )

    result = rust_engine_step.evaluate_generation_ops_with_rust(
        model, database, indices=[5, 1], batch_size=2, s=64, x=77
    )
    assert result is entries
    assert calls[1] == (
        "generation",
        [5, 1],
        {"batch_size": 2, "s": 64, "gen_seq_imbalance_correction_scale": 1.0, "prefix": 0, "x": 77},
    )

    ops_json = '[{"op": "gemm"}]'
    result = rust_engine_step.evaluate_ops_json_with_rust(
        model, database, ops_json=ops_json, is_context=True, batch_size=3, s=32
    )
    assert result is entries
    assert calls[2] == (
        "json",
        ops_json,
        {"is_context": True, "batch_size": 3, "s": 32, "prefix": 0, "imbalance_correction_scale": 1.0, "x": None},
    )


def test_rust_provenance_tier_forwarded_into_python_capture(monkeypatch) -> None:
    """The engine-step helpers forward the compiled engine's per-call
    empirical provenance tier into Python's ``capture_provenance`` (used by
    the support matrix to label HYBRID_PASS rows). Silicon answers
    (``last_provenance() is None`` or ``"silicon"``) record nothing."""
    from aiconfigurator.sdk.operations import util_empirical

    class _FakeHandle:
        def __init__(self, tier):
            self._tier = tier

        def _run_static_per_op_with_metadata(self, **kwargs):
            return ([("context_attention", 10.0, 0.0, "silicon")], [("generation_attention", 6.0, 0.0, "silicon")])

        def mixed_step_latency(self, *args, **kwargs):
            return 8.5

        def decode_step_latency(self, *args, **kwargs):
            return 9.5

        def last_provenance(self):
            return self._tier

    model = _dense_model()
    database = SimpleNamespace(system="test_sxm", backend="vllm", version="1.0.0")
    runtime_config = RuntimeConfig(batch_size=1, beam_width=1, isl=8, osl=4)

    def run_all_helpers(tier):
        monkeypatch.setattr(rust_engine_step, "_cached_engine_handle", lambda model, database: _FakeHandle(tier))
        rust_engine_step.estimate_static_latency_breakdown_with_rust(
            model, database, runtime_config, mode="static", stride=2, latency_correction_scale=1.0
        )
        rust_engine_step.estimate_mixed_step_latency_with_rust(
            model, database, ctx_tokens=8, gen_tokens=1, isl=8, osl=4, prefix=0
        )
        rust_engine_step.estimate_decode_step_latency_with_rust(model, database, gen_tokens=1, isl=8, osl=4)

    # A rust-routed HYBRID rescue records its tier (all three step surfaces).
    with util_empirical.capture_provenance() as tags:
        run_all_helpers("xop")
    assert tags == {"xop"}
    assert util_empirical.worst_provenance(tags) == "xop"

    # Pure-silicon runs record nothing -> worst_provenance stays "silicon".
    for silicon_tier in (None, "silicon"):
        with util_empirical.capture_provenance() as tags:
            run_all_helpers(silicon_tier)
        assert tags == set()
        assert util_empirical.worst_provenance(tags) == "silicon"

    # Outside a capture, forwarding is a no-op (note_provenance no-ops).
    run_all_helpers("xshape")


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


def test_engine_config_json_preserves_w4a16_nvfp4_weight_and_moe_profiles() -> None:
    model = SimpleNamespace(
        model_path="Test/W4A16Nvfp4",
        architecture="Qwen3ForCausalLM",
        config=ModelConfig(
            tp_size=1,
            pp_size=1,
            attention_dp_size=1,
            moe_tp_size=None,
            moe_ep_size=None,
            gemm_quant_mode=common.GEMMQuantMode.w4a16_nvfp4,
            moe_quant_mode=common.MoEQuantMode.w4a16_nvfp4,
            kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
            fmha_quant_mode=common.FMHAQuantMode.bfloat16,
        ),
    )
    database = SimpleNamespace(system="test_sxm", backend="vllm", version="1.0.0")

    config = json.loads(rust_engine_step._engine_config_json(model, database))

    assert config["weight_dtype"] == "w4a16_nvfp4"
    assert config["moe_dtype"] == "w4a16_nvfp4"


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
def test_nemotron_super_fp8_native_estimation_uses_packaged_moe_data() -> None:
    """Issue #1522: the exact deployed MoE key must estimate successfully."""
    pytest.importorskip("aiconfigurator_core")
    from aiconfigurator_core.sdk.rust_engine_step import RustForwardPassPerfModel

    model = RustForwardPassPerfModel.from_native(
        {
            "schema_version": 1,
            "model_name": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8",
            "system_name": "h100_sxm",
            "backend": "vllm",
            "backend_version": "0.24.0",
            "tp_size": 4,
            "pp_size": 1,
            "attention_dp_size": 1,
            "cp_size": None,
            "moe_tp_size": 1,
            "moe_ep_size": 4,
            "weight_dtype": "fp8",
            "activation_dtype": None,
            "moe_dtype": None,
            "kv_cache_dtype": None,
            "kv_block_size": 16,
            "nextn": None,
            "nextn_accept_rates": None,
            "extra": {},
        },
        {
            "bucket_count": 16,
            "max_batch_size": 512,
            "max_kv_tokens": 2_000_000,
            "max_num_tokens": 8192,
            "max_observations": 1024,
            "min_observations": 5,
        },
    )

    estimate_ms = model.estimate_forward_pass_time_ms(
        {
            "version": 1,
            "worker_id": "repro",
            "dp_rank": 0,
            "counter_id": 0,
            "scheduled_requests": {
                "num_prefill_requests": 1,
                "sum_prefill_tokens": 4224,
                "var_prefill_length": 0.0,
                "sum_prefill_kv_tokens": 0,
                "num_decode_requests": 0,
                "sum_decode_kv_tokens": 0,
                "var_decode_kv_tokens": 0.0,
            },
            "queued_requests": {
                "num_prefill_requests": 0,
                "sum_prefill_tokens": 0,
                "var_prefill_length": 0.0,
                "num_decode_requests": 0,
                "sum_decode_kv_tokens": 0,
                "var_decode_kv_tokens": 0.0,
            },
        }
    )

    assert model.diagnostics()["readiness"] == "ready"
    assert estimate_ms is not None and estimate_ms > 0.0


@pytest.mark.integration
def test_forward_pass_perf_model_native_default_directional_bounds_end_to_end() -> None:
    """End-to-end native forward-pass model with default bounds over a real fixture.

    Builds a native model via ``compile_engine`` (crossing into the Rust core),
    estimates a prefill iteration, then drives one correction bucket through
    its slower ceiling, faster floor, and recovery between them. Requires the
    compiled ``aiconfigurator_core`` extension.
    """
    pytest.importorskip("aiconfigurator_core")
    from aiconfigurator.sdk.rust_engine_step import RustForwardPassPerfModel

    config = {
        "schema_version": 1,
        "model_name": "Qwen/Qwen3-32B",
        "system_name": "h200_sxm",
        "backend": "trtllm",
        "backend_version": "1.3.0rc20",
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
        "extra": {},
    }
    model = RustForwardPassPerfModel.from_native(
        config,
        {
            "min_observations": 2,
            "max_observations": 2,
        },
    )

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
            "wall_time": native_ms * 8.0 / 1000.0,
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
    assert model.get_max_correction_factor() == pytest.approx(2.0)
    assert model.diagnostics()["source"] == "aic_with_correction"

    faster_obs = [
        {
            "version": 1,
            "wall_time": native_ms * 0.1 / 1000.0,
            "scheduled_requests": {
                "num_prefill_requests": 2,
                "sum_prefill_tokens": 2048,
                "sum_prefill_kv_tokens": 0,
            },
        }
    ]
    model.tune_with_fpms([faster_obs])
    assert model.estimate_forward_pass_time_ms(prefill) == pytest.approx(native_ms * 1.25)

    model.tune_with_fpms([faster_obs])
    assert model.estimate_forward_pass_time_ms(prefill) == pytest.approx(native_ms * 0.5)

    recovery_obs = [
        {
            "version": 1,
            "wall_time": native_ms * 1.25 / 1000.0,
            "scheduled_requests": {
                "num_prefill_requests": 2,
                "sum_prefill_tokens": 2048,
                "sum_prefill_kv_tokens": 0,
            },
        }
    ]
    model.tune_with_fpms([recovery_obs, recovery_obs])
    assert model.estimate_forward_pass_time_ms(prefill) == pytest.approx(native_ms * 1.25)


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
        "backend_version": "1.3.0rc20",
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
    parity does not wait for data. Since the pyo3 op unification the op
    serializes ITSELF (`_spec_json`); the pin now reads the real wire."""
    from aiconfigurator.sdk import common
    from aiconfigurator.sdk.operations import ContextDeepSeekV4AttentionModule

    op = ContextDeepSeekV4AttentionModule(
        "context_attention",
        1.0,
        64,  # num_heads (per-rank)
        64,  # native_heads
        1,  # tp_size
        7168,  # hidden_size
        1536,  # q_lora_rank
        1024,  # o_lora_rank
        512,  # head_dim
        64,  # rope_head_dim
        64,  # index_n_heads
        128,  # index_head_dim
        1024,  # index_topk
        2048,  # window_size
        4,  # compress_ratio -> Csa
        16,  # o_groups (rank-local)
        common.KVCacheQuantMode.fp8,
        common.FMHAQuantMode.bfloat16,
        common.GEMMQuantMode.fp8_block,
        architecture="DeepseekV4ForCausalLM",
        cp_size=2,
    )
    spec = json.loads(op._spec_json())["Dsv4Context"]
    assert spec["cp_size"] == 2
    assert spec["window_size"] == 2048
    assert spec["attn_kind"] == "Csa"


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
            # enable_wideep dropped from the fixture: the deprecated flag left
            # the engine identity (constant False; moe_comm_backend +
            # num_gpus_per_node carry the regime).
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


def test_engine_config_json_identity_includes_database_policy():
    """Two views of the SAME on-disk identity that differ only in the
    shared-layer or strict-provenance policy must get DISTINCT handle-cache
    keys: ``build_engine_spec_json`` bakes the policy-dependent
    ``perf_db_sources`` into the compiled handle, so aliasing them makes the
    reuse-aware behavior call-order-dependent (whichever view warms the cache
    answers — or fails — for the other)."""
    from aiconfigurator.sdk import common

    def _model():
        cfg = SimpleNamespace(
            tp_size=1,
            pp_size=1,
            moe_tp_size=1,
            moe_ep_size=8,
            attention_dp_size=1,
            cp_size=8,
            gemm_quant_mode=common.GEMMQuantMode.fp8_block,
            moe_quant_mode=None,
            fmha_quant_mode=None,
            kvcache_quant_mode=None,
            comm_quant_mode=None,
            moe_backend=None,
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

    def _view(*, shared_layer: bool, strict_provenance: bool):
        return SimpleNamespace(
            system="test_sxm",
            backend="sglang",
            version="0.5.12",
            enable_shared_layer=shared_layer,
            strict_provenance=strict_provenance,
        )

    base = rust_engine_step._engine_config_json(_model(), _view(shared_layer=False, strict_provenance=False))
    shared_on = rust_engine_step._engine_config_json(_model(), _view(shared_layer=True, strict_provenance=False))
    strict_on = rust_engine_step._engine_config_json(_model(), _view(shared_layer=False, strict_provenance=True))
    assert base != shared_on, "enable_shared_layer must participate in the cache identity"
    assert base != strict_on, "strict_provenance must participate in the cache identity"
    assert shared_on != strict_on


def test_op_conversion_error_raises_typed_and_memoized(monkeypatch):
    """An OpConversionError (op graph not expressible in Rust) must be
    surfaced as RustEngineUnsupportedError and cached per engine identity so
    a sweep does not re-walk a known-unconvertible op graph. (The AFD op-list
    path catches it; the engine-step surfaces propagate it — the opspec
    coverage tripwire keeps it unreachable for shipped op-level models.)"""
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
    """The WideEP MLA table axis is per-rank heads; the retired Python query
    converted ``num_heads = 128 // tp_size`` (mla.py). The Rust constructor
    must apply the same conversion — emitting raw tp_size makes the engine
    query the wrong table slice (tp=8 would read the heads=8 extrapolation
    instead of heads=16)."""
    from aiconfigurator.sdk import common
    from aiconfigurator.sdk.operations import WideEPContextMLA, WideEPGenerationMLA

    ctx = WideEPContextMLA("context_attention", 1.0, 8, common.KVCacheQuantMode.fp8, common.FMHAQuantMode.fp8_block)
    gen = WideEPGenerationMLA(
        "generation_attention", 1.0, 8, common.KVCacheQuantMode.fp8, common.FMHAQuantMode.fp8_block
    )
    ctx_spec = json.loads(ctx._spec_json())["WideEpContextMla"]
    gen_spec = json.loads(gen._spec_json())["WideEpGenerationMla"]
    assert ctx_spec["num_heads"] == 16  # 128 // 8, NOT tp_size=8
    assert gen_spec["num_heads"] == 16


# ---- Large-EP op graphs: native compilation (AIC-1601, PR 2.5) ----

_SYSTEMS_DATA_ROOT = Path(__file__).resolve().parents[3] / "aic-core/src/aiconfigurator_core/systems/data"


def test_large_ep_opspec_key_sets_match_the_rust_structs():
    """Tripwire against silent-default drift. The Rust crate does NOT set
    ``deny_unknown_fields``: a misspelled key in an emitted opspec dict would
    silently fall back to the struct's ``#[serde(default)]`` value with no
    error anywhere. The emitted key sets must therefore EQUAL the Rust struct
    field sets exactly — source of truth:
    ``rust/aiconfigurator-core/src/operators/moe_a2a.rs::MoeAllToAllOp`` and
    ``rust/aiconfigurator-core/src/operators/moe_expert_compute.rs::MoeExpertComputeOp``."""
    from aiconfigurator.sdk.operations import MoEAllToAll, MoEExpertCompute

    a2a = MoEAllToAll(
        "context_moe_dispatch",
        58.0,
        phase="dispatch",
        comm_backend="deepep_ht",
        comm_dtype="default",
        hidden_size=7168,
        topk=8,
        num_experts=256,
        moe_ep_size=32,
        node_num=4,
        sms=20,
        attention_tp_size=1,
    )
    a2a_spec = json.loads(a2a._spec_json())
    assert set(a2a_spec) == {"MoeAllToAll"}
    # rust `MoeAllToAllOp` fields (operators/moe_a2a.rs), verbatim.
    assert frozenset(a2a_spec["MoeAllToAll"]) == frozenset(
        {
            "name",
            "scale_factor",
            "phase",
            "comm_backend",
            "comm_dtype",
            "hidden_size",
            "topk",
            "num_experts",
            "moe_ep_size",
            "node_num",
            "sms",
            "attention_tp_size",
        }
    )

    ep = MoEExpertCompute(
        "context_moe",
        58.0,
        hidden_size=7168,
        inter_size=2048,
        topk=8,
        num_experts=256,
        moe_ep_size=32,
        quant_mode=common.MoEQuantMode.fp8_block,
        workload_distribution="power_law_1.01",
        attention_dp_size=32,
        inference_phase="context",
        num_slots=None,
        kernel_source=None,
        is_gated=True,
        enable_eplb=True,
    )
    ep_spec = json.loads(ep._spec_json())
    assert set(ep_spec) == {"MoeExpertCompute"}
    # rust `MoeExpertComputeOp` fields (operators/moe_expert_compute.rs), verbatim.
    assert frozenset(ep_spec["MoeExpertCompute"]) == frozenset(
        {
            "name",
            "scale_factor",
            "hidden_size",
            "inter_size",
            "topk",
            "num_experts",
            "moe_ep_size",
            "quant_mode",
            "workload_distribution",
            "attention_dp_size",
            "inference_phase",
            "num_slots",
            "kernel_source",
            "is_gated",
            "enable_eplb",
        }
    )
    # Wire formats the Rust serde impls expect: quant_mode is the snake_case
    # ``MoEQuantMode`` member name; an unpinned kernel_source crosses as null
    # (the Rust op ports all five auto-resolution legs and resolves at query
    # time); the Python ctor already resolved num_slots=None -> num_experts.
    fields = ep_spec["MoeExpertCompute"]
    assert fields["quant_mode"] == "fp8_block"
    assert fields["kernel_source"] is None
    assert fields["num_slots"] == 256
    assert fields["is_gated"] is True and fields["enable_eplb"] is True


def _h200_sglang_wideep_paths() -> list[str]:
    from aiconfigurator.sdk.operations.base import resolve_op_data_path

    return [
        resolve_op_data_path(str(_SYSTEMS_DATA_ROOT / "h200_sxm"), "sglang", "0.5.6.post2", filename)
        for filename in (
            "wideep_deepep_normal_perf.parquet",
            "wideep_deepep_ll_perf.parquet",
            "wideep_context_moe_perf.parquet",
            "wideep_generation_moe_perf.parquet",
            "wideep_context_mla_perf.parquet",
            "wideep_generation_mla_perf.parquet",
        )
    ]


@pytest.mark.skipif(
    not all(os.path.exists(p) for p in _h200_sglang_wideep_paths()),
    reason="shipped h200_sxm sglang wideEP parquets not present",
)
def test_large_ep_op_graph_compiles_natively(caplog):
    """AIC-1601 (PR 2.5): the large-EP ops (MoEAllToAll / MoEExpertCompute) now have
    Rust op constructors and wire mirrors, so a large-EP model compiles
    into the Rust engine natively — the documented Python-step fallback this
    test used to pin is retired. A rust-routed static run must answer with
    the scalar engine-step keys and match the Python step on the same
    config."""
    import logging
    import math

    from aiconfigurator.sdk.backends.factory import get_backend
    from aiconfigurator.sdk.engine import build_engine_spec_json
    from aiconfigurator.sdk.models import get_model
    from aiconfigurator.sdk.perf_database import get_database

    # A shipped-data large-EP config: DeepSeek-R1 EP32 on h200/sglang, the
    # per-phase comm backends + node width the enumerator would set, and the
    # legacy wideEP quant set (fp8_block MLA slices, fp8 KV cache).
    cfg = ModelConfig(
        tp_size=1,
        pp_size=1,
        attention_dp_size=32,
        moe_tp_size=1,
        moe_ep_size=32,
        gemm_quant_mode=common.GEMMQuantMode.fp8_block,
        moe_quant_mode=common.MoEQuantMode.fp8_block,
        kvcache_quant_mode=common.KVCacheQuantMode.fp8,
        fmha_quant_mode=common.FMHAQuantMode.fp8_block,
        moe_comm_backend={"context": "deepep_ht", "generation": "deepep_ll"},
        num_gpus_per_node=8,
    )
    model = get_model("deepseek-ai/DeepSeek-R1", cfg, "sglang")
    # Current slot: the wideEP tables backfill from their 0.5.6.post2/0.5.9/
    # 0.5.10/0.5.12 sole-source dirs while gemm/attention resolve on the
    # primary — the production large-EP query shape.
    database = get_database("h200_sxm", "sglang", "0.5.14")

    # (1) The op graph compiles into an EngineSpec carrying the tagged
    # large-EP variants, with the per-phase comm backends the config set.
    spec = json.loads(
        build_engine_spec_json(
            model,
            model_path="deepseek-ai/DeepSeek-R1",
            system="h200_sxm",
            backend="sglang",
            backend_version="0.5.14",
            kv_block_size=None,
            systems_path=None,
            nextn=0,
            database=database,
        )
    )
    for phase_ops, comm_backend in ((spec["context_ops"], "deepep_ht"), (spec["generation_ops"], "deepep_ll")):
        a2a_fields = [op["MoeAllToAll"] for op in phase_ops if "MoeAllToAll" in op]
        ep_fields = [op["MoeExpertCompute"] for op in phase_ops if "MoeExpertCompute" in op]
        assert a2a_fields and ep_fields, "the compiled spec must carry the large-EP variants"
        assert {fields["comm_backend"] for fields in a2a_fields} == {comm_backend}
        # Production graphs never pin a kernel: it crosses as null and the
        # Rust op auto-resolves per backend at query time.
        assert all(fields["kernel_source"] is None for fields in ep_fields)

    rust_engine_step._engine_handle_cache_clear()
    try:
        # (2) The engine-step wrapper compiles a live handle — no
        # RustEngineUnsupportedError.
        assert rust_engine_step._cached_engine_handle(model, database) is not None

        # (3) End to end through the backend gate: the rust-routed run_static
        # answers natively — no python-step fallback warning, and (since the
        # per-op FFI, #1496) the breakdown carries per-op keys like the
        # Python step's, including the large-EP ops priced by the Rust
        # engine.
        backend = get_backend("sglang")
        runtime_config = RuntimeConfig(batch_size=1, beam_width=1, isl=1024, osl=32, engine_step_backend="rust")
        rust_engine_step._python_step_fallback_reset()
        with caplog.at_level(logging.WARNING):
            summary = backend.run_static(model, database, runtime_config, mode="static", stride=32)
        assert not any("using the python step" in record.message for record in caplog.records)

        context_latency = summary.get_context_latency_dict()
        generation_latency = summary.get_generation_latency_dict()
        for phase_latency in (context_latency, generation_latency):
            assert phase_latency, "the rust step must report a per-op breakdown"
            assert any("moe_dispatch" in name or "moe_combine" in name for name in phase_latency), (
                "the large-EP comm ops must be priced by the rust engine",
                sorted(phase_latency),
            )
            for name, value in phase_latency.items():
                assert math.isfinite(value) and value >= 0.0, name
        rust_context = sum(context_latency.values())
        rust_generation = sum(generation_latency.values())
        assert rust_context > 0.0 and rust_generation > 0.0

        # (Step 4 of the original test — "parity with the Python step" via
        # engine_step_backend="python" — retired with the value: since the
        # PR-3 no-op it had become a vacuous rust-vs-rust self-comparison,
        # and the value itself is gone after the deprecation window. The
        # per-op oracle parity for this graph lives in parity_tests/.)
    finally:
        rust_engine_step._engine_handle_cache_clear()


@pytest.mark.integration
def test_shipped_gb200_ep32_node8_reports_executed_fallback_to_api_and_cli(cli_parser, monkeypatch, caplog):
    """The production-shaped EP32/node8 request must identify the EP8/node1
    silicon rows the Rust query actually used, while keeping generic per-op
    source tags as ``estimated``."""
    import logging

    from aiconfigurator.cli.api import cli_estimate
    from aiconfigurator.cli.estimate_detail_report import format_estimate_detail_report
    from aiconfigurator.cli.main import _run_estimate_mode
    from aiconfigurator.sdk.models import get_model
    from aiconfigurator.sdk.perf_database import get_database

    rust_engine_step._engine_handle_cache_clear()
    try:
        captured_results = []

        def _capture_real_estimate(**kwargs):
            result = cli_estimate(**kwargs)
            captured_results.append(result)
            return result

        monkeypatch.setattr("aiconfigurator.cli.api.cli_estimate", _capture_real_estimate)
        args = cli_parser.parse_args(
            [
                "estimate",
                "--model-path",
                "deepseek-ai/DeepSeek-R1",
                "--system",
                "gb200",
                "--backend",
                "sglang",
                "--backend-version",
                "0.5.16",
                "--estimate-mode",
                "static",
                "--database-mode",
                "SILICON",
                "--isl",
                "1024",
                "--osl",
                "32",
                "--batch-size",
                "1",
                "--tp-size",
                "1",
                "--pp-size",
                "1",
                "--attention-dp-size",
                "32",
                "--moe-tp-size",
                "1",
                "--moe-ep-size",
                "32",
                "--gemm-quant-mode",
                "fp8_block",
                "--moe-quant-mode",
                "fp8_block",
                "--kvcache-quant-mode",
                "fp8",
                "--fmha-quant-mode",
                "fp8_block",
            ]
        )
        with caplog.at_level(logging.WARNING, logger="aiconfigurator.cli.main"):
            _run_estimate_mode(args)
        result = captured_results[0]

        expected = (_CONTEXT_FALLBACK, _GENERATION_FALLBACK)
        assert result.moe_comm_fallbacks == expected
        assert result.summary is not None
        assert result.summary.get_moe_comm_fallbacks() == expected
        assert result.summary.get_context_source_dict()["context_moe_dispatch"] == "estimated"
        assert result.summary.get_generation_source_dict()["generation_moe_dispatch"] == "estimated"
        assert (
            "Estimated MoE communication latency used fallback silicon data: "
            "context/deepep_ht: requested EP32/node8; using EP8/node1 silicon data." in caplog.messages
        )

        config = ModelConfig(
            tp_size=1,
            pp_size=1,
            attention_dp_size=32,
            moe_tp_size=1,
            moe_ep_size=32,
            gemm_quant_mode=common.GEMMQuantMode.fp8_block,
            moe_quant_mode=common.MoEQuantMode.fp8_block,
            kvcache_quant_mode=common.KVCacheQuantMode.fp8,
            fmha_quant_mode=common.FMHAQuantMode.fp8_block,
            moe_comm_backend={"context": "deepep_ht", "generation": "deepep_ll"},
            num_gpus_per_node=4,
        )
        model = get_model("deepseek-ai/DeepSeek-R1", config, "sglang")
        database = get_database("gb200", "sglang", "0.5.16")
        handle = rust_engine_step._cached_engine_handle(model, database)

        def _fallback_payloads(entries):
            payloads = []
            for entry in entries:
                metadata = entry[4]
                if metadata is None:
                    continue
                assert isinstance(metadata, tuple) and len(metadata) == 2
                first, additional = metadata
                assert isinstance(first, tuple) and isinstance(additional, list)
                payloads.append(first)
                payloads.extend(additional)
            return payloads

        public_context, public_generation = handle.run_static_per_op(
            batch_size=1,
            isl=1024,
            osl=32,
            mode="static",
            stride=32,
        )
        metadata_context, metadata_generation = handle._run_static_per_op_with_metadata(
            batch_size=1,
            isl=1024,
            osl=32,
            mode="static",
            stride=32,
        )
        assert all(len(entry) == 4 for entry in (*public_context, *public_generation))
        assert all(len(entry) == 5 for entry in (*metadata_context, *metadata_generation))
        assert {payload[0] for payload in _fallback_payloads(metadata_context)} == {"context"}
        assert {payload[0] for payload in _fallback_payloads(metadata_generation)} == {"generation"}

        public_shared, _, _ = handle.mixed_step_breakdown_per_op(1024, 1, 1024, 32)
        metadata_shared, metadata_context_attention, metadata_decode_attention = (
            handle._mixed_step_breakdown_per_op_with_metadata(1024, 1, 1024, 32)
        )
        assert all(len(entry) == 4 for entry in public_shared)
        assert {payload[0] for payload in _fallback_payloads(metadata_shared)} == {"context"}
        assert not any(entry[4] is not None for entry in (*metadata_context_attention, *metadata_decode_attention))

        public_decode = handle.decode_step_per_op(1, 1024, 32)
        metadata_decode = handle._decode_step_per_op_with_metadata(1, 1024, 32)
        assert all(len(entry) == 4 for entry in public_decode)
        assert {payload[0] for payload in _fallback_payloads(metadata_decode)} == {"generation"}

        detail = format_estimate_detail_report(result, detail="source")
        assert "context/deepep_ht: requested EP32/node8; using EP8/node1 silicon data" in detail
        assert "generation/deepep_ll: requested EP32/node8; using EP8/node1 silicon data" in detail

    finally:
        rust_engine_step._engine_handle_cache_clear()


def test_every_selectable_database_mode_routes_to_rust():
    """The compiled engine answers every selectable database mode — SILICON,
    the util-space empirical layer (HYBRID / EMPIRICAL), and SOL (also ported
    to Rust). The former mode-based delegation is gone: SOL_FULL, the only
    non-rust name, is a Python-side PER-CALL diagnostic that mode entry
    refuses to activate as a default mode, so the gate no longer inspects the
    database mode at all."""
    from enum import Enum

    from aiconfigurator.sdk.config import RuntimeConfig
    from aiconfigurator.sdk.rust_engine_step import should_use_rust_engine_step

    class _Mode(Enum):
        SILICON = "SILICON"
        HYBRID = "HYBRID"
        EMPIRICAL = "EMPIRICAL"
        SOL = "SOL"

    class _DB:
        def __init__(self, mode):
            self._mode = mode

        def get_default_database_mode(self):
            return self._mode

    rc = RuntimeConfig(engine_step_backend="rust")
    assert should_use_rust_engine_step(rc, _DB(_Mode.SILICON))
    assert should_use_rust_engine_step(rc, _DB(_Mode.HYBRID))
    assert should_use_rust_engine_step(rc, _DB(_Mode.EMPIRICAL))
    assert should_use_rust_engine_step(rc, _DB(_Mode.SOL))
    assert should_use_rust_engine_step(rc)  # no database context -> unchanged


@pytest.mark.unit
def test_rust_perf_db_misses_translate_to_perf_data_not_available():
    """The PyO3 boundary collapses every Rust error into ValueError; the
    perf-DB miss class (prefix "perf database error: ") must re-surface as
    PerfDataNotAvailableError so sweep.py can mark the point unanswerable
    instead of aborting the whole parallel config. Other ValueErrors pass
    through untouched."""
    from aiconfigurator.sdk.errors import PerfDataNotAvailableError
    from aiconfigurator_core.sdk.rust_engine_step import _reraise_engine_error

    miss = ValueError(
        "perf database error: FPM decode query total_kv_read_tokens=4013448 is outside the collected domain"
    )
    with pytest.raises(PerfDataNotAvailableError):
        _reraise_engine_error(miss)

    genuine = ValueError("invalid engine config: isl must be greater than 0")
    with pytest.raises(ValueError) as excinfo:
        _reraise_engine_error(genuine)
    # PerfDataNotAvailableError subclasses RuntimeError, so pytest.raises
    # (ValueError) alone can never catch a translated error — assert the
    # exact object passed through untouched instead.
    assert excinfo.value is genuine


@pytest.mark.unit
def test_engine_handle_cache_key_distinguishes_raw_quant_identity():
    """FPM cell identity keys on the five RAW quant enum names, including
    comm_quant_mode; the collapsed DataType strings under-key it (fp8 vs
    fp8_ootb -> "fp8", no comm axis at all)."""
    from types import SimpleNamespace

    from aiconfigurator_core.sdk.rust_engine_step import _engine_config_json

    def make(comm, gemm):
        config = SimpleNamespace(
            tp_size=4,
            pp_size=1,
            moe_tp_size=1,
            moe_ep_size=4,
            attention_dp_size=1,
            cp_size=None,
            gemm_quant_mode=SimpleNamespace(name=gemm, value=None),
            moe_quant_mode=SimpleNamespace(name="nvfp4", value=None),
            fmha_quant_mode=SimpleNamespace(name="bfloat16", value=None),
            comm_quant_mode=SimpleNamespace(name=comm, value=None),
            kvcache_quant_mode=SimpleNamespace(name="fp8", value=None),
        )
        model = SimpleNamespace(
            config=config,
            model_path="org/model-a",
            architecture="X",
            forward_model="fpm",
            _nextn=None,
            _nextn_accepted=None,
        )
        database = SimpleNamespace(system="b200_sxm", backend="vllm", version="0.25.1", systems_root="/tmp/x")
        return _engine_config_json(model, database)

    assert make("half", "fp8") != make("int8", "fp8")
    assert make("half", "fp8") != make("half", "fp8_ootb")
    assert make("half", "fp8") == make("half", "fp8")


def test_python_step_fallback_telemetry_counts_and_warns_once(caplog) -> None:
    import logging

    from aiconfigurator_core.sdk import rust_engine_step as res

    res._python_step_fallback_reset()
    try:
        with caplog.at_level(logging.DEBUG, logger="aiconfigurator_core.sdk.rust_engine_step"):
            res.note_python_step_fallback("non_perf_database", "SimpleNamespace")
            res.note_python_step_fallback("non_perf_database", "SimpleNamespace")
            res.note_python_step_fallback("unsupported_op_graph:afd", "no OpSpec conversion")
        assert res.python_step_fallback_counts() == {
            "non_perf_database": 2,
            "unsupported_op_graph:afd": 1,
        }
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_records) == 2  # warn-once per distinct reason
    finally:
        res._python_step_fallback_reset()


def test_default_config_wideep_mla_spec_survives_the_real_bincode_decode():
    """A DEFAULT `ModelConfig` must produce a spec Rust can decode.

    `ModelConfig.attention_backend` is None ("no lane override", AIC-1715) while
    `PyWideEPContextMLA`/`PyWideEPGenerationMLA` type `attn_backend` as a
    non-optional `&str` (pyo3 raises a `TypeError` at construction on `None` —
    even louder than the retired dict-builder's `null`-serializes-then-bincode-
    aborts failure mode). Model-building code (`DeepSeekModel._build_*_ops`)
    pre-bakes the default (`config.attention_backend or "flashinfer"`) at the
    construction call site for exactly this reason. sglang WideEP DeepSeek is
    exactly the config that routes to the compiled engine, so this must
    round-trip through the REAL extension — a monkeypatched stub is
    structurally blind to payload types.
    """
    import aiconfigurator_core
    from aiconfigurator.sdk import common, engine
    from aiconfigurator.sdk import config as sdk_config
    from aiconfigurator.sdk.operations.mla import WideEPContextMLA, WideEPGenerationMLA

    cfg = sdk_config.ModelConfig(
        tp_size=8,
        gemm_quant_mode=common.GEMMQuantMode.fp8_block,
        moe_quant_mode=common.MoEQuantMode.fp8_block,
        kvcache_quant_mode=common.KVCacheQuantMode.fp8,
        fmha_quant_mode=common.FMHAQuantMode.fp8_block,
        moe_tp_size=1,
        moe_ep_size=8,
        enable_wideep=True,
    )
    assert cfg.attention_backend is None, "the regression under test needs the no-override default"

    # Mirrors DeepSeekModel._build_*_ops: pre-bake the same default the
    # constructor's own pyo3 signature carries (`or "flashinfer"`), since
    # `config.attention_backend` is None here by construction.
    attn_backend = cfg.attention_backend or "flashinfer"
    ctx = WideEPContextMLA("context_attention", 61, 8, cfg.kvcache_quant_mode, cfg.fmha_quant_mode, attn_backend)
    gen = WideEPGenerationMLA(
        "generation_attention", 61, 8, cfg.kvcache_quant_mode, cfg.fmha_quant_mode, attn_backend
    )
    model = SimpleNamespace(config=cfg, architecture="DeepseekV3ForCausalLM", context_ops=[ctx], generation_ops=[gen])

    spec_json = engine.build_engine_spec_json(
        model,
        model_path="deepseek-ai/DeepSeek-V3",
        system="gb200",
        backend="sglang",
        backend_version="0.5.14",
        kv_block_size=None,
        systems_path=None,
        nextn=0,
        database=None,
    )
    spec = json.loads(spec_json)
    assert spec["context_ops"][0]["WideEpContextMla"]["attn_backend"] == "flashinfer"
    assert spec["generation_ops"][0]["WideEpGenerationMla"]["attn_backend"] == "flashinfer"

    # The JSON -> bincode conversion is NOT schema-gated (the version check lives
    # in `from_bincode`), so this exercises field validation on today's tree.
    assert len(aiconfigurator_core.engine_spec_bincode_from_json(spec_json)) > 0
