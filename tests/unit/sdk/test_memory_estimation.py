# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the pure-Python KV-cache memory estimate in ``sdk.memory``.

Exercises the naive-fallback math (MLA-aware per-token KV, rough weight estimate,
80%-of-post-weight reservation, default constants) and the OfFree/OfTotal native
budget formulas. ``sdk.memory`` imports the compiled ``aiconfigurator_core``
extension at module top, so these tests are skipped (``pytest.importorskip``)
when it is not built. The native budget math is tested with a synthetic breakdown
(no perf DB / model build), and the routing in ``estimate_kv_cache`` is driven by
monkeypatching ``KVCacheEstimator.from_request`` (to raise -> fallback, or to
return a ``KVCacheEstimator`` carrying a synthetic breakdown).

The fixture-derived magic numbers (DeepSeek-V3 -> 70_272 bytes/token, Llama-3.1-70B
-> 327_680) match the values the prior Rust naive tests asserted, so this is a
checkable parity assertion for the port.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

# ``sdk.memory`` imports the compiled ``aiconfigurator_core`` extension at module
# top, so these pure-Python tests require it to be importable. Skip them when it
# is not built rather than stubbing ``sys.modules`` — a stub would leak to other
# test modules in the same xdist worker and break tests that use the real
# extension (e.g. the FPM pyclass).
pytest.importorskip("aiconfigurator_core")

from aiconfigurator.sdk import memory

_GIB = 1 << 30


# --------------------------------------------------------------------------- #
# Memory-fraction validation (runs before any model build).
# --------------------------------------------------------------------------- #


def test_fraction_validation_rejects_wrong_kind_per_backend():
    with pytest.raises(ValueError, match="incompatible memory fraction"):
        memory._validate_memory_fraction("trtllm", "of_total", 0.9)
    with pytest.raises(ValueError, match="incompatible memory fraction"):
        memory._validate_memory_fraction("vllm", "of_free", 0.9)
    with pytest.raises(ValueError, match="incompatible memory fraction"):
        memory._validate_memory_fraction("sglang", "of_free", 0.9)
    # Correct pairings pass.
    memory._validate_memory_fraction("trtllm", "of_free", 0.9)
    memory._validate_memory_fraction("vllm", "of_total", 0.85)
    memory._validate_memory_fraction("sglang", "of_total", 0.85)


def test_fraction_validation_rejects_out_of_range():
    with pytest.raises(ValueError):
        memory._validate_memory_fraction("trtllm", "of_free", 1.5)
    with pytest.raises(ValueError):
        memory._validate_memory_fraction("vllm", "of_total", -0.1)


def test_naive_reservation_validation_rejects_out_of_range():
    # Boundaries are valid; finite values outside [0, 1] (and non-finite) are not.
    memory._validate_naive_reservation(0.0)
    memory._validate_naive_reservation(1.0)
    memory._validate_naive_reservation(0.8)
    for bad in (-0.1, 1.5, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="naive_kv_reservation"):
            memory._validate_naive_reservation(bad)


def test_estimate_num_gpu_blocks_rejects_non_positive_or_non_integer_block_size():
    # Caught up front (before any model build), so no perf DB / fixture is needed.
    # A positive non-integer (e.g. 0.5 -> int() == 0) must be rejected rather than
    # reaching the floor-divide and raising ZeroDivisionError.
    common = dict(
        max_num_tokens=8192,
        max_batch_size=256,
        memory_fraction_kind="of_free",
        memory_fraction_value=0.9,
    )
    for bad in (0, -1, 0.5, 1.5):
        with pytest.raises(ValueError, match="scheduler_block_size"):
            memory.estimate_num_gpu_blocks(
                "Qwen/Qwen3-32B",
                "h200_sxm",
                "trtllm",
                scheduler_block_size=bad,
                **common,
            )


# --------------------------------------------------------------------------- #
# Native budget math: OfFree (TRT-LLM) and OfTotal (vLLM/SGLang).
# --------------------------------------------------------------------------- #


def _breakdown(non_kv: float, kv_per_token: float, capacity: float) -> dict[str, float]:
    return {
        "weights_bytes": non_kv * 0.7,
        "activations_bytes": non_kv * 0.2,
        "runtime_overhead_bytes": non_kv * 0.07,
        "comm_overhead_bytes": non_kv * 0.03,
        "non_kv_bytes": non_kv,
        "kv_size_per_token_bytes": kv_per_token,
        "gpu_memory_capacity_bytes": capacity,
    }


def _native_estimate(breakdown, **kwargs):
    """Run the native budget math over a (synthetic) breakdown via KVCacheEstimator."""
    return memory.KVCacheEstimator(breakdown).estimate(**kwargs)


def _naive_estimate(
    model_path,
    *,
    tp_size=1,
    pp_size=1,
    gpu_memory_capacity_bytes_override,
    naive_kv_reservation,
    allow_hf_config_download=False,
):
    """Build a NaiveKVCacheEstimator and run its estimate (mirrors the old helper)."""
    return memory.NaiveKVCacheEstimator.from_model_path(
        model_path, tp_size=tp_size, pp_size=pp_size, allow_hf_config_download=allow_hf_config_download
    ).estimate(capacity=gpu_memory_capacity_bytes_override, naive_kv_reservation=naive_kv_reservation)


def test_of_free_budget_formula():
    capacity, non_kv, kv_per_token, f = 141.0 * _GIB, 60.0 * _GIB, 327_680.0, 0.9
    out = _native_estimate(
        _breakdown(non_kv, kv_per_token, capacity),
        is_of_free=True,
        fraction=f,
        gpu_memory_capacity_bytes_override=None,
    )
    ref_kv = (capacity - non_kv) * f
    assert out["total_kv_size_bytes"] == int(ref_kv)
    assert out["total_kv_size_tokens"] == int(ref_kv / kv_per_token)
    assert out["source"] == "native"
    assert out["memory_breakdown"] is not None


def test_of_total_budget_formula_differs_from_of_free():
    capacity, non_kv, kv_per_token, f = 180.0 * _GIB, 70.0 * _GIB, 131_072.0, 0.9
    bd = _breakdown(non_kv, kv_per_token, capacity)
    of_total = _native_estimate(bd, is_of_free=False, fraction=f, gpu_memory_capacity_bytes_override=None)
    of_free = _native_estimate(bd, is_of_free=True, fraction=f, gpu_memory_capacity_bytes_override=None)
    assert of_total["total_kv_size_bytes"] == int(capacity * f - non_kv)
    # OfFree gives a strictly larger budget for f < 1 (the whole point of the split).
    assert of_free["total_kv_size_bytes"] > of_total["total_kv_size_bytes"]


def test_native_no_kv_budget_raises():
    capacity, non_kv = 100.0 * _GIB, 95.0 * _GIB
    with pytest.raises(ValueError, match="no KV budget"):
        _native_estimate(
            _breakdown(non_kv, 100_000.0, capacity),
            is_of_free=False,
            fraction=0.9,
            gpu_memory_capacity_bytes_override=None,
        )


def test_native_zero_per_token_raises():
    with pytest.raises(ValueError, match="kv_size_per_token_bytes"):
        _native_estimate(
            _breakdown(10.0 * _GIB, 0.0, 100.0 * _GIB),
            is_of_free=True,
            fraction=0.9,
            gpu_memory_capacity_bytes_override=None,
        )


def test_breakdown_applies_nextn_to_model_config(monkeypatch):
    # KVCacheEstimator.from_request must push nextn/MTP onto the ModelConfig before
    # get_model, so _get_memory_usage scales activation memory for speculative
    # decoding instead of treating the request as nextn=0.
    captured = {}

    def _fake_get_model(model_path, model_config, backend):
        captured["nextn"] = model_config.nextn
        captured["has_nextn_accepted"] = hasattr(model_config, "nextn_accepted")

        class _StubModel:
            def get_kvcache_bytes_per_sequence(self, seq_len):
                return 100.0 * seq_len

            def get_kvcache_max_tokens(self, budget):
                return int(budget // 100)

        return _StubModel()

    class _StubBackend:
        def _get_memory_usage(self, *a, **k):
            return {"weights": 1.0, "activations": 1.0, "others": 1.0, "nccl": 1.0, "kvcache": 0.0}

    class _StubDB:
        def __init__(self):
            self.system_spec = {"gpu": {"mem_capacity": 100 * _GIB}}

    monkeypatch.setattr(memory, "get_model", _fake_get_model)
    monkeypatch.setattr(memory, "get_backend", lambda backend: _StubBackend())

    def _fake_get_database(*args, **kwargs):
        captured["systems_paths"] = kwargs.get("systems_paths")
        return _StubDB()

    monkeypatch.setattr(memory.perf_database, "get_database", _fake_get_database)

    memory.KVCacheEstimator.from_request(
        "Qwen/Qwen3-32B",
        "h200_sxm",
        "trtllm",
        max_num_tokens=8192,
        max_batch_size=256,
        nextn=2,
        systems_path="/tmp/aic-core-systems",
    )
    assert captured["nextn"] == 2
    assert captured["has_nextn_accepted"] is False
    assert captured["systems_paths"] == "/tmp/aic-core-systems"


def test_breakdown_accepts_nextn_without_acceptance_field(monkeypatch):
    # Capacity math consumes only the core-owned draft depth.
    captured = {}

    def _fake_get_model(model_path, model_config, backend):
        captured["nextn"] = model_config.nextn
        captured["has_nextn_accepted"] = hasattr(model_config, "nextn_accepted")

        class _StubModel:
            def get_kvcache_bytes_per_sequence(self, seq_len):
                return 100.0 * seq_len

            def get_kvcache_max_tokens(self, budget):
                return int(budget // 100)

        return _StubModel()

    class _StubBackend:
        def _get_memory_usage(self, *a, **k):
            return {"weights": 1.0, "activations": 1.0, "others": 1.0, "nccl": 1.0, "kvcache": 0.0}

    class _StubDB:
        def __init__(self):
            self.system_spec = {"gpu": {"mem_capacity": 100 * _GIB}}

    monkeypatch.setattr(memory, "get_model", _fake_get_model)
    monkeypatch.setattr(memory, "get_backend", lambda backend: _StubBackend())
    monkeypatch.setattr(memory.perf_database, "get_database", lambda *a, **k: _StubDB())

    memory.KVCacheEstimator.from_request(
        "Qwen/Qwen3-32B",
        "h200_sxm",
        "trtllm",
        max_num_tokens=8192,
        max_batch_size=256,
        nextn=2,
    )
    assert captured["nextn"] == 2
    assert captured["has_nextn_accepted"] is False


def test_native_capacity_override_wins():
    capacity, non_kv, kv_per_token, f = 141.0 * _GIB, 60.0 * _GIB, 327_680.0, 0.9
    override = 200 * _GIB
    out = _native_estimate(
        _breakdown(non_kv, kv_per_token, capacity),
        is_of_free=True,
        fraction=f,
        gpu_memory_capacity_bytes_override=override,
    )
    assert out["total_gpu_capacity_bytes"] == override
    assert out["total_kv_size_bytes"] == int((override - non_kv) * f)


# --------------------------------------------------------------------------- #
# NaiveKVCacheEstimator: per-token KV (standard GQA/MHA vs MLA), weight estimate,
# config->geometry parsing, sliding-window detection, and the byte-budget ->
# token-count inverse with auto LINEAR/HYBRID mode selection.
# --------------------------------------------------------------------------- #


def _estimator(geometry, *, dtype_bytes=2, tp_size=1, pp_size=1):
    return memory.NaiveKVCacheEstimator(geometry, dtype_bytes=dtype_bytes, tp_size=tp_size, pp_size=pp_size)


def test_naive_kv_per_token_mla():
    # DeepSeek-V3-style: layers=61, kv_lora_rank=512, qk_rope_head_dim=64, bf16:
    #   61 * (512 + 64) * 2 = 70_272 bytes/token. The latent is NOT sharded by TP.
    geom = {"layers": 61, "kv_lora_rank": 512, "qk_rope_head_dim": 64}
    assert _estimator(geom, tp_size=1).kv_bytes_per_token() == 70_272
    assert _estimator(geom, tp_size=8).kv_bytes_per_token() == 70_272


def test_naive_kv_per_token_mla_fails_closed_without_latent_width():
    # MLA detected (kv_lora_rank present) but qk_rope_head_dim missing -> the
    # cached latent width is unclear, so fail closed (None -> caller raises) rather
    # than guessing the latent width.
    assert _estimator({"layers": 61, "kv_lora_rank": 512}).kv_bytes_per_token() is None


def test_naive_kv_per_token_standard():
    # Llama-3.1-70B-style: layers=80, num_kv_heads=8, head_dim=128, bf16:
    #   TP=1: 2 * 8 * 128 * 80 * 2 = 327_680.
    geom = {"layers": 80, "num_kv_heads": 8, "head_dim": 128}
    assert _estimator(geom, tp_size=1).kv_bytes_per_token() == 327_680
    # TP=4: kv_heads_per_rank = ceil(8/4) = 2 -> quarter.
    assert _estimator(geom, tp_size=4).kv_bytes_per_token() == 2 * 2 * 128 * 80 * 2
    # TP=16: kv_heads_per_rank = ceil(8/16) = 1.
    assert _estimator(geom, tp_size=16).kv_bytes_per_token() == 2 * 1 * 128 * 80 * 2


def test_naive_geometry_from_raw_unsupported_arch():
    # Unsupported architecture -> raw-key extraction (parsed=None). head_dim is
    # derived from hidden_size / num_attention_heads when absent.
    hf_config = {
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "hidden_size": 4096,
        "vocab_size": 128_000,
        "intermediate_size": 11_008,
    }
    geom = memory.NaiveKVCacheEstimator._geometry(hf_config, None)
    assert geom["head_dim"] == 4096 // 32  # 128
    # Standard branch: 2 * ceil(8/2) * 128 * 32 * 2 (bf16).
    assert _estimator(geom, tp_size=2).kv_bytes_per_token() == 2 * 4 * 128 * 32 * 2
    assert _estimator(geom).weight_bytes() is not None
    # vocab / intermediate_size are required: drop either and the weight is None.
    assert _estimator({**geom, "vocab": None}).weight_bytes() is None
    assert _estimator({**geom, "inter": None}).weight_bytes() is None


def test_naive_kv_per_token_empty_geometry_is_none():
    assert _estimator({}).kv_bytes_per_token() is None
    assert _estimator({}).weight_bytes() is None


def test_dtype_bytes():
    dtype_bytes = memory.NaiveKVCacheEstimator._dtype_bytes
    assert dtype_bytes({"torch_dtype": "bfloat16"}) == 2
    assert dtype_bytes({"torch_dtype": "float32"}) == 4
    assert dtype_bytes({"torch_dtype": "fp8_e4m3"}) == 1
    assert dtype_bytes({}) == 2  # default bf16


# --------------------------------------------------------------------------- #
# NaiveKVCacheEstimator hybrid sliding-window support: layer-layout detection +
# the piecewise byte-budget -> token-count inverse, so the fallback does not
# assume linear KV growth for SWA models.
# --------------------------------------------------------------------------- #


def test_naive_swa_layout_from_layer_types():
    # Explicit per-layer list (Gemma-3/4 style): count sliding vs the rest.
    hf_config = {"sliding_window": 1024, "layer_types": ["sliding_attention"] * 5 + ["full_attention"]}
    assert memory.NaiveKVCacheEstimator._swa_layout(hf_config) == (1024, 5, 1)


def test_naive_swa_layout_from_pattern():
    # sliding_window + sliding_window_pattern=6: one global layer every 6 layers.
    hf_config = {"sliding_window": 4096, "num_hidden_layers": 30, "sliding_window_pattern": 6}
    assert memory.NaiveKVCacheEstimator._swa_layout(hf_config) == (4096, 25, 5)


def test_naive_swa_layout_unresolved_returns_none():
    # A bare window with no per-layer signal is too model-specific to split.
    assert memory.NaiveKVCacheEstimator._swa_layout({"sliding_window": 4096}) == (None, None, None)
    assert memory.NaiveKVCacheEstimator._swa_layout({}) == (None, None, None)


def test_naive_hybrid_mode_caps_swa_past_window():
    # 25 SWA + 5 global layers, window 1024, 8 KV heads, head_dim 128, bf16, tp=1.
    geom = {
        "sliding_window": 1024,
        "num_swa_layers": 25,
        "num_global_layers": 5,
        "num_kv_heads": 8,
        "head_dim": 128,
    }
    est = _estimator(geom)
    assert est.mode is memory.NaiveKVCacheMode.HYBRID
    hybrid = memory.NaiveKVCacheMode.HYBRID
    per_layer = 2 * 8 * 128 * 2  # 4096
    rate = (25 + 5) * per_layer  # every layer grows up to the window
    global_const = 5 * per_layer
    assert est.get_kvcache_max_tokens(rate * 512, mode=hybrid) == 512  # within the window: linear
    assert est.get_kvcache_max_tokens(rate * 1024, mode=hybrid) == 1024  # at the boundary
    # Past the window only the 5 global layers grow.
    budget = rate * 1024 + global_const * 3
    assert est.get_kvcache_max_tokens(budget, mode=hybrid) == 1027
    # mode auto-resolves to HYBRID here, so it gives the same answer as forcing it.
    assert est.get_kvcache_max_tokens(budget) == 1027
    # Beats the linear seq_len=1 extrapolation, which over-charges the SWA layers.
    assert est.get_kvcache_max_tokens(budget, mode=hybrid) > budget // rate


def test_naive_mode_linear_for_standard_and_mla():
    # No sliding-window layout -> LINEAR.
    assert _estimator({"num_kv_heads": 8, "head_dim": 128}).mode is memory.NaiveKVCacheMode.LINEAR
    # MLA latent cache is out of scope for the hybrid heuristic -> LINEAR.
    mla_hybrid = {
        "sliding_window": 1024,
        "num_swa_layers": 5,
        "num_global_layers": 1,
        "kv_lora_rank": 512,
        "num_kv_heads": 8,
        "head_dim": 128,
    }
    assert _estimator(mla_hybrid).mode is memory.NaiveKVCacheMode.LINEAR


# --------------------------------------------------------------------------- #
# Naive fallback end-to-end (post-weight reservation + defaults), driven through
# the SDK's existing config loaders + `_parse_hf_config_json` (cached fixtures).
# --------------------------------------------------------------------------- #


# A synthetic config for an architecture AIC does not support, carrying the full
# geometry the naive weight + KV formulas require.
_RAW_UNSUPPORTED = {
    "architectures": ["FooBarForCausalLM"],
    "num_hidden_layers": 48,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "hidden_size": 4096,
    "vocab_size": 128_000,
    "intermediate_size": 14_336,
    "torch_dtype": "bfloat16",
}


def test_naive_fallback_mla_fixture():
    # `deepseek-ai/DeepSeek-V3` is cached in model_configs (DefaultHFModels), so
    # `NaiveKVCacheEstimator._load_config` loads it without a download; the
    # supported arch goes through `_parse_hf_config_json` -> 70_272 bytes/token.
    capacity = 180 * _GIB
    out = _naive_estimate(
        "deepseek-ai/DeepSeek-V3",
        tp_size=16,
        pp_size=1,
        gpu_memory_capacity_bytes_override=capacity,
        naive_kv_reservation=memory._DEFAULT_NAIVE_KV_RESERVATION,
        allow_hf_config_download=False,
    )
    assert out["source"] == "naive_fallback"
    assert out["memory_breakdown"] is None
    assert out["kv_size_per_token_bytes"] == 70_272  # config-driven MLA latent


def test_naive_fallback_unsupported_arch_uses_raw_read(monkeypatch):
    # Unsupported architecture: `_parse_hf_config_json` raises, so the estimator
    # falls to the raw-key read. This drives the real parser (FooBar arch is not in
    # ARCHITECTURE_TO_MODEL_FAMILY) through the except branch into
    # `NaiveKVCacheEstimator._geometry(hf_config, None)`.
    monkeypatch.setattr(memory.NaiveKVCacheEstimator, "_load_config", lambda *a, **k: dict(_RAW_UNSUPPORTED))
    out = _naive_estimate(
        "foo/bar-unknown-arch",
        tp_size=1,
        pp_size=1,
        gpu_memory_capacity_bytes_override=200 * _GIB,
        naive_kv_reservation=memory._DEFAULT_NAIVE_KV_RESERVATION,
        allow_hf_config_download=False,
    )
    assert out["source"] == "naive_fallback"
    # Raw-read standard branch: layers=48, n_kv=8, head_dim=4096/32=128, bf16.
    assert out["kv_size_per_token_bytes"] == 2 * 8 * 128 * 48 * 2  # 196608


def test_naive_fallback_raises_without_config():
    # HF id that is neither local nor pre-cached, with download disabled -> no
    # config -> raise rather than guess with a placeholder constant.
    with pytest.raises(ValueError, match="insufficient model metadata"):
        _naive_estimate(
            "definitely/not-a-real-model-xyz",
            tp_size=1,
            pp_size=1,
            gpu_memory_capacity_bytes_override=200 * _GIB,
            naive_kv_reservation=memory._DEFAULT_NAIVE_KV_RESERVATION,
            allow_hf_config_download=False,
        )


def test_naive_fallback_raises_on_missing_weight_metadata(monkeypatch):
    # Config present but lacking weight-estimate fields (no vocab_size /
    # intermediate_size) -> raise rather than fabricate a placeholder weight.
    raw = {k: v for k, v in _RAW_UNSUPPORTED.items() if k not in ("vocab_size", "intermediate_size")}
    monkeypatch.setattr(memory.NaiveKVCacheEstimator, "_load_config", lambda *a, **k: raw)
    with pytest.raises(ValueError, match="weight-estimate fields"):
        _naive_estimate(
            "foo/bar-unknown-arch",
            tp_size=1,
            pp_size=1,
            gpu_memory_capacity_bytes_override=200 * _GIB,
            naive_kv_reservation=memory._DEFAULT_NAIVE_KV_RESERVATION,
            allow_hf_config_download=False,
        )


def test_naive_fallback_honors_reservation_param(monkeypatch):
    # The reservation fraction is caller-adjustable (default 0.80); a 0.5 reserve
    # halves the post-weight KV budget. Uses a full synthetic config so the weight
    # estimate is available (no placeholder fallback any more).
    capacity = 200 * _GIB
    monkeypatch.setattr(memory.NaiveKVCacheEstimator, "_load_config", lambda *a, **k: dict(_RAW_UNSUPPORTED))
    weight = float(
        memory.NaiveKVCacheEstimator(
            memory.NaiveKVCacheEstimator._geometry(dict(_RAW_UNSUPPORTED), None),
            dtype_bytes=memory.NaiveKVCacheEstimator._dtype_bytes(_RAW_UNSUPPORTED),
            tp_size=1,
            pp_size=1,
        ).weight_bytes()
    )
    post_weight = capacity - weight
    out = _naive_estimate(
        "foo/bar-unknown-arch",
        tp_size=1,
        pp_size=1,
        gpu_memory_capacity_bytes_override=capacity,
        naive_kv_reservation=0.5,
        allow_hf_config_download=False,
    )
    assert out["total_kv_size_bytes"] == int(post_weight * 0.5)


def test_naive_fallback_requires_capacity_override():
    # Use a cached model so from_model_path succeeds; the capacity check inside
    # .estimate() is what must raise.
    with pytest.raises(ValueError, match="gpu_memory_capacity_bytes_override"):
        _naive_estimate(
            "deepseek-ai/DeepSeek-V3",
            tp_size=1,
            pp_size=1,
            gpu_memory_capacity_bytes_override=None,
            naive_kv_reservation=memory._DEFAULT_NAIVE_KV_RESERVATION,
            allow_hf_config_download=False,
        )


# An unsupported hybrid sliding-window config: 25 SWA + 5 global layers, window
# 1024. The naive fallback should cap the SWA layers at the window rather than
# extrapolate the seq_len=1 per-token rate forever.
_RAW_HYBRID = {
    "architectures": ["FooSwaForCausalLM"],
    "num_hidden_layers": 30,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "hidden_size": 2048,
    "vocab_size": 256_000,
    "intermediate_size": 8192,
    "sliding_window": 1024,
    "layer_types": (["sliding_attention"] * 5 + ["full_attention"]) * 5,
    "torch_dtype": "bfloat16",
}


def test_naive_fallback_hybrid_beats_linear_capacity(monkeypatch):
    # With a large budget the token count runs well past the window, so the
    # window-capped curve fits strictly more tokens than dividing by the full
    # per-token rate would. kv_size_per_token_bytes (the reported seq=1 rate) is
    # unchanged -- only the capacity inversion is curve-aware.
    monkeypatch.setattr(memory.NaiveKVCacheEstimator, "_load_config", lambda *a, **k: dict(_RAW_HYBRID))
    out = _naive_estimate(
        "foo/swa-unknown-arch",
        tp_size=1,
        pp_size=1,
        gpu_memory_capacity_bytes_override=200 * _GIB,
        naive_kv_reservation=memory._DEFAULT_NAIVE_KV_RESERVATION,
        allow_hf_config_download=False,
    )
    assert out["source"] == "naive_fallback"
    linear_tokens = out["total_kv_size_bytes"] // out["kv_size_per_token_bytes"]
    assert out["total_kv_size_tokens"] > linear_tokens


# --------------------------------------------------------------------------- #
# estimate_kv_cache: routing between native and naive fallback.
# --------------------------------------------------------------------------- #


def test_estimate_kv_cache_falls_back_when_breakdown_raises(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("perf DB not available")

    monkeypatch.setattr(memory.KVCacheEstimator, "from_request", classmethod(_boom))
    monkeypatch.setattr(memory.NaiveKVCacheEstimator, "_load_config", lambda *a, **k: dict(_RAW_UNSUPPORTED))
    out = memory.estimate_kv_cache(
        "foo/bar-unknown-arch",
        "h200_sxm",
        "trtllm",
        max_num_tokens=8192,
        max_batch_size=256,
        memory_fraction_kind="of_free",
        memory_fraction_value=0.9,
        gpu_memory_capacity_bytes_override=200 * _GIB,
        allow_naive_fallback=True,
    )
    assert out["source"] == "naive_fallback"
    assert out["tolerance_adjusted"] is None


def test_estimate_kv_cache_propagates_when_fallback_disabled(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("perf DB not available")

    monkeypatch.setattr(memory.KVCacheEstimator, "from_request", classmethod(_boom))
    with pytest.raises(ValueError, match="unsupported model/backend/GPU"):
        memory.estimate_kv_cache(
            "definitely/not-a-real-model-xyz",
            "h200_sxm",
            "trtllm",
            max_num_tokens=8192,
            max_batch_size=256,
            memory_fraction_kind="of_free",
            memory_fraction_value=0.9,
            gpu_memory_capacity_bytes_override=200 * _GIB,
            allow_naive_fallback=False,
        )


def test_estimate_kv_cache_nextn_reaches_breakdown(monkeypatch):
    # Acceptance is not part of the aic-core memory API.
    reached = {"n": 0}

    def _spy(*args, **kwargs):
        reached["n"] += 1
        reached["nextn"] = kwargs.get("nextn")
        raise RuntimeError("stop here")

    monkeypatch.setattr(memory.KVCacheEstimator, "from_request", classmethod(_spy))

    with pytest.raises(ValueError, match="unsupported model/backend/GPU"):
        memory.estimate_kv_cache(
            "Qwen/Qwen3-32B",
            "h200_sxm",
            "trtllm",
            max_num_tokens=8192,
            max_batch_size=256,
            memory_fraction_kind="of_free",
            memory_fraction_value=0.9,
            nextn=2,
            allow_naive_fallback=False,
        )
    assert reached["n"] == 1
    assert reached["nextn"] == 2


def test_estimate_kv_cache_rejects_bad_fraction_before_breakdown(monkeypatch):
    # Fraction validation must run before any breakdown attempt: a TRT-LLM +
    # of_total request fails even though the breakdown would also have raised.
    called = {"n": 0}

    def _spy(*args, **kwargs):
        called["n"] += 1
        raise RuntimeError("should not be reached")

    monkeypatch.setattr(memory.KVCacheEstimator, "from_request", classmethod(_spy))
    with pytest.raises(ValueError, match="incompatible memory fraction"):
        memory.estimate_kv_cache(
            "Qwen/Qwen3-32B",
            "h200_sxm",
            "trtllm",
            max_num_tokens=8192,
            max_batch_size=256,
            memory_fraction_kind="of_total",
            memory_fraction_value=0.9,
            allow_naive_fallback=True,
        )
    assert called["n"] == 0


def test_estimate_kv_cache_native_uses_synthetic_breakdown(monkeypatch):
    bd = _breakdown(60.0 * _GIB, 327_680.0, 141.0 * _GIB)
    monkeypatch.setattr(memory.KVCacheEstimator, "from_request", classmethod(lambda cls, *a, **k: cls(bd)))
    out = memory.estimate_kv_cache(
        "Qwen/Qwen3-32B",
        "h200_sxm",
        "trtllm",
        max_num_tokens=8192,
        max_batch_size=256,
        memory_fraction_kind="of_free",
        memory_fraction_value=0.9,
    )
    assert out["source"] == "native"
    assert out["total_kv_size_bytes"] == int((141.0 * _GIB - 60.0 * _GIB) * 0.9)
    assert out["tolerance_adjusted"] is None
    # The internal KV-curve inverse callable must not leak into the result dict
    # (it would not survive the PyO3 boundary).
    assert "tokens_from_kv_bytes" not in out


# --------------------------------------------------------------------------- #
# tolerance_fraction: validation (up front) + application (native & naive).
#
# This math moved out of Rust `apply_tolerance` into the Python
# `estimate_kv_cache`; the assertions reproduce the formula the Rust tests
# previously asserted (`adj_bytes = floor(raw * (1 - t))`, tokens floor-divided),
# so this is the parity guarantee that the port preserves the old numbers.
# --------------------------------------------------------------------------- #


def test_tolerance_applied_to_native_estimate(monkeypatch):
    bd = _breakdown(60.0 * _GIB, 327_680.0, 141.0 * _GIB)
    monkeypatch.setattr(memory.KVCacheEstimator, "from_request", classmethod(lambda cls, *a, **k: cls(bd)))
    out = memory.estimate_kv_cache(
        "Qwen/Qwen3-32B",
        "h200_sxm",
        "trtllm",
        max_num_tokens=8192,
        max_batch_size=256,
        memory_fraction_kind="of_free",
        memory_fraction_value=0.9,
        tolerance_fraction=0.05,
    )
    adj = out["tolerance_adjusted"]
    assert adj is not None
    assert adj["tolerance_fraction"] == pytest.approx(0.05)
    # adj bytes = floor(raw * 0.95); tokens recomputed by floor-divide.
    expected_bytes = int(out["total_kv_size_bytes"] * 0.95)
    assert adj["total_kv_size_bytes"] == expected_bytes
    assert adj["total_kv_size_tokens"] == expected_bytes // out["kv_size_per_token_bytes"]
    # Raw fields are left untouched.
    assert out["total_kv_size_bytes"] == int((141.0 * _GIB - 60.0 * _GIB) * 0.9)


def test_tolerance_applied_to_naive_fallback(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("perf DB not available")

    monkeypatch.setattr(memory.KVCacheEstimator, "from_request", classmethod(_boom))
    monkeypatch.setattr(memory.NaiveKVCacheEstimator, "_load_config", lambda *a, **k: dict(_RAW_UNSUPPORTED))
    out = memory.estimate_kv_cache(
        "foo/bar-unknown-arch",
        "h200_sxm",
        "trtllm",
        max_num_tokens=8192,
        max_batch_size=256,
        memory_fraction_kind="of_free",
        memory_fraction_value=0.9,
        gpu_memory_capacity_bytes_override=200 * _GIB,
        tolerance_fraction=0.05,
        allow_naive_fallback=True,
    )
    assert out["source"] == "naive_fallback"
    adj = out["tolerance_adjusted"]
    assert adj is not None
    expected_bytes = int(out["total_kv_size_bytes"] * 0.95)
    assert adj["total_kv_size_bytes"] == expected_bytes
    assert adj["total_kv_size_tokens"] == expected_bytes // out["kv_size_per_token_bytes"]


def test_tolerance_out_of_range_rejected_before_breakdown(monkeypatch):
    # Tolerance validation must run before any breakdown attempt: a bad tolerance
    # fails even though the breakdown would have succeeded (spy never called).
    called = {"n": 0}

    def _spy(*args, **kwargs):
        called["n"] += 1
        raise RuntimeError("should not be reached")

    monkeypatch.setattr(memory.KVCacheEstimator, "from_request", classmethod(_spy))
    for bad in (1.0, 1.5, -0.1, float("nan")):
        with pytest.raises(ValueError, match="tolerance_fraction"):
            memory.estimate_kv_cache(
                "Qwen/Qwen3-32B",
                "h200_sxm",
                "trtllm",
                max_num_tokens=8192,
                max_batch_size=256,
                memory_fraction_kind="of_free",
                memory_fraction_value=0.9,
                tolerance_fraction=bad,
                allow_naive_fallback=True,
            )
    assert called["n"] == 0


def test_tolerance_zero_is_noop_margin(monkeypatch):
    # t = 0.0 is accepted (half-open [0, 1)): adjusted bytes == raw bytes.
    bd = _breakdown(60.0 * _GIB, 327_680.0, 141.0 * _GIB)
    monkeypatch.setattr(memory.KVCacheEstimator, "from_request", classmethod(lambda cls, *a, **k: cls(bd)))
    out = memory.estimate_kv_cache(
        "Qwen/Qwen3-32B",
        "h200_sxm",
        "trtllm",
        max_num_tokens=8192,
        max_batch_size=256,
        memory_fraction_kind="of_free",
        memory_fraction_value=0.9,
        tolerance_fraction=0.0,
    )
    adj = out["tolerance_adjusted"]
    assert adj is not None
    assert adj["total_kv_size_bytes"] == out["total_kv_size_bytes"]
