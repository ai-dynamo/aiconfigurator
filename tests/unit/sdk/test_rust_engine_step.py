# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
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


def test_static_latency_breakdown_maps_runtime_config_to_fpm(monkeypatch) -> None:
    calls = []

    class _FakeEstimator:
        def forward_pass_time_ms(self, metrics):
            calls.append(metrics)
            scheduled = metrics[0]["scheduled_requests"]
            if scheduled.get("num_prefill_requests", 0):
                return 10.0
            return 2.0

    monkeypatch.setattr(rust_engine_step, "_cached_estimator", lambda _: _FakeEstimator())

    model = SimpleNamespace(
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
    assert [len(call) for call in calls] == [2, 2, 2]
    assert calls[0][0]["scheduled_requests"] == {
        "num_prefill_requests": 2,
        "sum_prefill_tokens": 12,
        "sum_prefill_kv_tokens": 4,
    }
    assert calls[1][0]["scheduled_requests"] == {
        "num_decode_requests": 2,
        "sum_decode_kv_tokens": 16,
    }
    assert calls[2][0]["scheduled_requests"] == {
        "num_decode_requests": 2,
        "sum_decode_kv_tokens": 20,
    }


def test_mixed_and_decode_helpers_map_to_fpm(monkeypatch) -> None:
    calls = []

    class _FakeEstimator:
        def forward_pass_time_ms(self, metrics):
            calls.append(metrics)
            return 7.5 + len(calls)

    monkeypatch.setattr(rust_engine_step, "_cached_estimator", lambda _: _FakeEstimator())

    model = SimpleNamespace(
        model_path="Test/Dense",
        architecture="LlamaForCausalLM",
        _context_length=4096,
        config=ModelConfig(
            tp_size=1,
            pp_size=1,
            attention_dp_size=1,
            gemm_quant_mode=common.GEMMQuantMode.bfloat16,
            moe_quant_mode=common.MoEQuantMode.bfloat16,
            kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
            fmha_quant_mode=common.FMHAQuantMode.bfloat16,
        ),
    )
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
    assert calls[0][0]["scheduled_requests"] == {
        "num_prefill_requests": 2,
        "sum_prefill_tokens": 384,
        "sum_prefill_kv_tokens": 256,
        "num_decode_requests": 7,
        "sum_decode_kv_tokens": 2688,
    }
    assert calls[1][0]["scheduled_requests"] == {
        "num_decode_requests": 7,
        "sum_decode_kv_tokens": 2688,
    }


def test_is_rust_core_available_handles_missing_extension(monkeypatch) -> None:
    """When the PyO3 extension can't be imported, the helper returns False instead of raising."""
    rust_engine_step._import_rust_core.cache_clear()

    def _raise(*args, **kwargs):
        raise ImportError("simulated missing extension")

    monkeypatch.setattr("builtins.__import__", _raise)
    try:
        assert rust_engine_step.is_rust_core_available() is False
    finally:
        rust_engine_step._import_rust_core.cache_clear()


def test_forward_pass_perf_model_wraps_pyo3_extension(monkeypatch) -> None:
    """The Python wrapper serializes inputs to JSON and decodes the PyO3 outputs."""
    recorded: dict = {}

    class _FakeInner:
        def __init__(self, kind):
            recorded["kind"] = kind

        def estimate_forward_pass_time_ms(self, metrics_json):
            recorded["estimate_metrics"] = metrics_json
            return 42.0

        def tune_with_fpms(self, iterations_json):
            recorded.setdefault("tuned", []).append(iterations_json)

        def diagnostics_json(self):
            return json.dumps({"source": "aic_with_correction"})

        def min_correction_factor(self):
            return 1.5

        def max_correction_factor(self):
            return 2.5

        def avg_correction_factor(self):
            return 2.0

    class _FakePerfModelClass:
        @staticmethod
        def from_native(config_json, options_json=None):
            recorded["from_native"] = (config_json, options_json)
            return _FakeInner("native")

        @staticmethod
        def best_available(config_json, options_json=None):
            recorded["best_available"] = (config_json, options_json)
            return _FakeInner("best")

        @staticmethod
        def from_regression(options_json=None):
            recorded["from_regression"] = options_json
            return _FakeInner("regression")

    monkeypatch.setattr(rust_engine_step, "_perf_model_class", lambda: _FakePerfModelClass)
    monkeypatch.setattr(rust_engine_step, "_configure_default_data_roots", lambda config=None: None)

    model = rust_engine_step.RustForwardPassPerfModel.best_available(
        {"config": True},
        {"min_observations": 2},
    )
    assert json.loads(recorded["best_available"][0]) == {"config": True}
    assert json.loads(recorded["best_available"][1]) == {"min_observations": 2}

    assert model.estimate_forward_pass_time_ms({"version": 1}) == 42.0
    assert json.loads(recorded["estimate_metrics"]) == {"version": 1}

    # The Python wrapper passes the raw iteration shape; Rust normalizes it.
    model.tune_with_fpms([[{"version": 1}], [{"version": 1}]])
    assert json.loads(recorded["tuned"][-1]) == [[{"version": 1}], [{"version": 1}]]

    assert model.diagnostics() == {"source": "aic_with_correction"}
    assert model.get_min_correction_factor() == 1.5
    assert model.get_max_correction_factor() == 2.5
    assert model.get_avg_correction_factor() == 2.0

    rust_engine_step.RustForwardPassPerfModel.from_native({"config": True}).close()
    assert json.loads(recorded["from_native"][0]) == {"config": True}
    assert recorded["from_native"][1] is None

    rust_engine_step.RustForwardPassPerfModel.from_regression({"min_observations": 2}).close()
    assert json.loads(recorded["from_regression"]) == {"min_observations": 2}
    model.close()


def test_forward_pass_perf_model_returns_none_before_tuning(monkeypatch) -> None:
    """A regression model not yet tuned returns ``None`` (PyO3 ``Option`` -> ``None``)."""

    class _FakeInner:
        def estimate_forward_pass_time_ms(self, metrics_json):
            return None

    class _FakePerfModelClass:
        @staticmethod
        def from_regression(options_json=None):
            return _FakeInner()

    monkeypatch.setattr(rust_engine_step, "_perf_model_class", lambda: _FakePerfModelClass)
    monkeypatch.setattr(rust_engine_step, "_configure_default_data_roots", lambda config=None: None)

    model = rust_engine_step.RustForwardPassPerfModel.from_regression()
    assert model.estimate_forward_pass_time_ms({"version": 1}) is None


@pytest.mark.skipif(
    not rust_engine_step.is_rust_core_available(),
    reason="Rust core PyO3 extension is not importable "
           "(install with `pip install -e \".[rust]\"` or `maturin develop --release`)"
)
def test_real_rust_core_returns_expected_latency(tmp_path, monkeypatch) -> None:
    systems_root = tmp_path / "systems"
    data_root = systems_root / "data" / "test_sxm" / "vllm" / "1.0.0"
    model_configs_root = tmp_path / "model_configs"
    data_root.mkdir(parents=True)
    model_configs_root.mkdir()

    (systems_root / "test_sxm.yaml").write_text("data_dir: data/test_sxm\n")
    (model_configs_root / "Test--Dense_config.json").write_text(
        """{
  "architectures": ["LlamaForCausalLM"],
  "model_type": "llama",
  "num_hidden_layers": 2,
  "num_attention_heads": 4,
  "num_key_value_heads": 2,
  "head_dim": 8,
  "hidden_size": 32,
  "intermediate_size": 64,
  "vocab_size": 160
}
"""
    )
    (data_root / "gemm_perf.txt").write_text(
        "gemm_dtype,m,n,k,latency\n"
        "bfloat16,20,64,32,1.0\n"
        "bfloat16,20,32,32,2.0\n"
        "bfloat16,20,128,32,3.0\n"
        "bfloat16,20,32,64,4.0\n"
        "bfloat16,2,160,32,0.5\n"
    )
    (data_root / "context_attention_perf.txt").write_text(
        "attn_dtype,kv_cache_dtype,batch_size,isl,num_heads,num_key_value_heads,head_dim,latency\n"
        "bfloat16,bfloat16,2,10,4,2,8,5.0\n"
    )
    (data_root / "generation_attention_perf.txt").write_text(
        "attn_dtype,kv_cache_dtype,batch_size,isl,num_heads,num_key_value_heads,head_dim,step,latency\n"
        "bfloat16,bfloat16,2,16,4,2,8,1,0.7\n"
    )

    monkeypatch.setenv("AICONFIGURATOR_SYSTEMS_PATH", str(systems_root))
    monkeypatch.setenv("AICONFIGURATOR_MODEL_CONFIGS_PATH", str(model_configs_root))
    rust_engine_step._import_rust_core.cache_clear()

    estimator = rust_engine_step.RustEngineStepEstimator(
        {
            "schema_version": 1,
            "model_name": "Test/Dense",
            "model_arch": None,
            "system_name": "test_sxm",
            "backend": "vllm",
            "backend_version": "1.0.0",
            "tp_size": 1,
            "pp_size": 1,
            "moe_tp_size": None,
            "moe_ep_size": None,
            "attention_dp_size": None,
            "weight_dtype": "bfloat16",
            "activation_dtype": "bfloat16",
            "kv_cache_dtype": "bfloat16",
            "kv_block_size": None,
            "extra": {},
        }
    )

    latency_ms = estimator.forward_pass_time_ms(
        [
            {
                "version": 1,
                "scheduled_requests": {
                    "num_prefill_requests": 2,
                    "sum_prefill_tokens": 20,
                    "sum_prefill_kv_tokens": 0,
                },
            },
        ]
    )

    assert latency_ms == pytest.approx(30.5)


@pytest.mark.skipif(
    not rust_engine_step.is_rust_core_available(),
    reason="Rust core PyO3 extension is not importable "
           "(install with `pip install -e \".[rust]\"` or `maturin develop --release`)"
)
def test_real_forward_pass_perf_model_regression(monkeypatch) -> None:
    """A real regression-only FPM model needs no perf data and reports diagnostics."""
    monkeypatch.setattr(rust_engine_step, "_configure_default_data_roots", lambda config=None: None)
    rust_engine_step._import_rust_core.cache_clear()

    model = rust_engine_step.RustForwardPassPerfModel.from_regression()
    metrics = [
        {
            "version": 1,
            "scheduled_requests": {"num_decode_requests": 1, "sum_decode_kv_tokens": 128},
        }
    ]
    # Untuned regression models return None until enough observations exist.
    assert model.estimate_forward_pass_time_ms(metrics) is None

    diagnostics = model.diagnostics()
    assert "source" in diagnostics
    assert "readiness" in diagnostics
