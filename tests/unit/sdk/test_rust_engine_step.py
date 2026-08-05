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

pytestmark = pytest.mark.unit


def test_should_use_rust_engine_step_supports_runtime_config_and_env(monkeypatch) -> None:
    monkeypatch.setenv("AICONFIGURATOR_ENGINE_STEP_BACKEND", "rust")

    assert rust_engine_step.should_use_rust_engine_step(RuntimeConfig())
    assert rust_engine_step.should_use_rust_engine_step(RuntimeConfig(engine_step_backend="rust"))
    assert not rust_engine_step.should_use_rust_engine_step(RuntimeConfig(engine_step_backend="python"))


def test_engine_step_backend_defaults_to_rust(monkeypatch) -> None:
    """With nothing requested, the compiled engine is the default for a real
    database (one the compiled engine could re-load from disk)."""
    monkeypatch.delenv("AICONFIGURATOR_ENGINE_STEP_BACKEND", raising=False)
    database = _real_database()

    assert rust_engine_step.should_use_rust_engine_step(RuntimeConfig(), database)
    assert not rust_engine_step.should_use_rust_engine_step(RuntimeConfig(engine_step_backend="python"), database)
    # Unknown values keep their historical meaning: not rust.
    assert not rust_engine_step.should_use_rust_engine_step(RuntimeConfig(engine_step_backend="auto"), database)


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

    latency, energy, source = rust_engine_step._fold_per_op(entries)
    assert latency == {"gemm": 5.0, "attention": 2.0}
    assert energy == {"gemm": 15.0, "attention": 1.0}
    assert source == {"gemm": "mixed", "attention": "silicon"}
    assert latency.keys() == energy.keys() == source.keys()

    scaled_latency, scaled_energy, _ = rust_engine_step._fold_per_op(entries, scale=2.0)
    assert scaled_latency == {"gemm": 10.0, "attention": 4.0}
    assert scaled_energy == {"gemm": 30.0, "attention": 2.0}


def test_static_latency_breakdown_routes_through_engine_handle(monkeypatch) -> None:
    """The static helper maps ``RuntimeConfig`` onto
    ``EngineHandle.run_static_per_op`` and folds the per-op entries into
    real-name latency / energy / source dicts, applying
    ``latency_correction_scale`` per key to latency AND energy."""
    calls = []

    class _FakeHandle:
        def run_static_per_op(self, **kwargs):
            calls.append(kwargs)
            # (context entries, generation entries): (name, latency_ms, energy_wms, source)
            return (
                [("context_qkv_gemm", 4.0, 40.0, "silicon"), ("context_attention", 6.0, 0.0, "empirical")],
                [("generation_qkv_gemm", 6.0, 12.0, "silicon")],
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
    ) = rust_engine_step.estimate_static_latency_breakdown_with_rust(
        model,
        database,
        RuntimeConfig(batch_size=2, beam_width=1, isl=8, osl=4, prefix=2),
        mode="static",
        stride=2,
        latency_correction_scale=1.5,
    )

    # Real op names, latency AND energy scaled by 1.5 per key.
    assert context_latency == {"context_qkv_gemm": 6.0, "context_attention": 9.0}
    assert generation_latency == {"generation_qkv_gemm": 9.0}
    assert context_energy == {"context_qkv_gemm": 60.0, "context_attention": 0.0}
    assert generation_energy == {"generation_qkv_gemm": 18.0}
    # Real provenance tags cross the FFI (no synthetic "rust" source).
    assert context_source == {"context_qkv_gemm": "silicon", "context_attention": "empirical"}
    assert generation_source == {"generation_qkv_gemm": "silicon"}

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

        def mixed_step_breakdown_per_op(self, *args, **kwargs):
            breakdown_calls.append((args, kwargs))
            return (
                [("context_mlp", 5.0, 50.0, "silicon")],
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
            "context_mlp": "silicon",
            "context_attention (scaled)": "silicon",
            "generation_attention": "silicon",
        },
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
        def mixed_step_breakdown_per_op(self, *args, **kwargs):
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
    """The decode bridge returns ``(latency_ms, energy_wms, per_op_latency,
    per_op_source)`` folded from ``EngineHandle.decode_step_per_op`` — the
    exact shape ``_get_genonly_step_latency`` produces on the Python step."""
    calls = []

    class _FakeHandle:
        def decode_step_per_op(self, *args, **kwargs):
            calls.append((args, kwargs))
            return [
                ("generation_attention", 2.0, 20.0, "silicon"),
                ("generation_mlp", 1.0, 4.0, "empirical"),
            ]

        def last_provenance(self):
            return None

    monkeypatch.setattr(rust_engine_step, "_cached_engine_handle", lambda model, database: _FakeHandle())

    latency_ms, energy_wms, per_op_latency, per_op_source = rust_engine_step.estimate_decode_step_breakdown_with_rust(
        _dense_model(),
        SimpleNamespace(system="test_sxm", backend="vllm", version="1.0.0"),
        gen_tokens=7,
        isl=256,
        osl=256,
    )

    assert latency_ms == 3.0
    assert energy_wms == 24.0
    assert per_op_latency == {"generation_attention": 2.0, "generation_mlp": 1.0}
    assert per_op_source == {"generation_attention": "silicon", "generation_mlp": "empirical"}
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

        def run_static_per_op(self, **kwargs):
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


def test_every_selectable_database_mode_routes_to_rust():
    """The compiled engine answers every selectable database mode — SILICON,
    the util-space empirical layer (HYBRID / EMPIRICAL), and SOL (ported with
    the SOL_FULL retirement) — so none of them delegates to the Python step.
    Only the retired SOL_FULL name (which mode entry refuses to activate)
    still trips the vestigial gate."""
    from enum import Enum

    from aiconfigurator.sdk.config import RuntimeConfig
    from aiconfigurator.sdk.rust_engine_step import should_use_rust_engine_step

    class _Mode(Enum):
        SILICON = "SILICON"
        HYBRID = "HYBRID"
        EMPIRICAL = "EMPIRICAL"
        SOL = "SOL"
        SOL_FULL = "SOL_FULL"

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
    assert not should_use_rust_engine_step(rc, _DB(_Mode.SOL_FULL))
    assert should_use_rust_engine_step(rc)  # no database context -> unchanged


def test_python_step_fallback_telemetry_counts_and_warns_once(caplog) -> None:
    import logging

    from aiconfigurator_core.sdk import rust_engine_step as res

    res._python_step_fallback_reset()
    try:
        with caplog.at_level(logging.DEBUG, logger="aiconfigurator_core.sdk.rust_engine_step"):
            res.note_python_step_fallback("explicit_python", "python")
            res.note_python_step_fallback("explicit_python", "python")
            res.note_python_step_fallback("database_mode", "SOL_FULL")
        assert res.python_step_fallback_counts() == {"explicit_python": 2, "database_mode": 1}
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_records) == 2  # warn-once per distinct reason
    finally:
        res._python_step_fallback_reset()
