# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ctypes
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from aiconfigurator.sdk import common, rust_engine_step
from aiconfigurator.sdk.config import ModelConfig, RuntimeConfig

pytestmark = pytest.mark.unit


def test_should_use_rust_engine_step_supports_runtime_config_and_env(monkeypatch) -> None:
    monkeypatch.setenv("AICONFIGURATOR_ENGINE_STEP_BACKEND", "rust")

    assert rust_engine_step.should_use_rust_engine_step(RuntimeConfig())
    assert rust_engine_step.should_use_rust_engine_step(RuntimeConfig(engine_step_backend="rust"))
    assert not rust_engine_step.should_use_rust_engine_step(RuntimeConfig(engine_step_backend="python"))


def test_autobuild_uses_release_profile(tmp_path, monkeypatch) -> None:
    crate_root = tmp_path / "rust" / "aiconfigurator-core"
    crate_root.mkdir(parents=True)
    (crate_root / "Cargo.toml").write_text('[package]\nname = "aiconfigurator-core"\nversion = "0.0.0"\n')

    monkeypatch.setattr(rust_engine_step, "_crate_root", lambda: crate_root)
    monkeypatch.setattr(rust_engine_step.shutil, "which", lambda name: "/usr/bin/cargo")

    commands = []

    def fake_run(command, check):
        commands.append(command)
        library_path = crate_root / "target" / "release" / rust_engine_step._library_name()
        library_path.parent.mkdir(parents=True)
        library_path.touch()

    monkeypatch.setattr(rust_engine_step.subprocess, "run", fake_run)

    library_path = rust_engine_step._build_rust_core()

    assert library_path == crate_root / "target" / "release" / rust_engine_step._library_name()
    assert commands == [
        [
            "cargo",
            "build",
            "--release",
            "--manifest-path",
            str(crate_root / "Cargo.toml"),
        ]
    ]


def test_find_library_can_ignore_debug_artifacts(tmp_path, monkeypatch) -> None:
    crate_root = tmp_path / "rust" / "aiconfigurator-core"
    debug_library_path = crate_root / "target" / "debug" / rust_engine_step._library_name()
    debug_library_path.parent.mkdir(parents=True)
    debug_library_path.touch()

    monkeypatch.delenv("AICONFIGURATOR_RUST_CORE_LIB", raising=False)
    monkeypatch.setattr(rust_engine_step, "_crate_root", lambda: crate_root)

    assert rust_engine_step._find_library(include_debug=False) is None
    assert rust_engine_step._find_library(include_debug=True) == debug_library_path


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
    # `sum_prefill_tokens` is the NEW prefill-token count (matches the
    # static-path `_prefill_metrics` convention: cached prefix tokens are
    # carried separately by `sum_prefill_kv_tokens`). For these inputs:
    #   num_prefill_requests = ceil(384 / 256) = 2
    #   cached_total = prefix(128) * num_prefill_requests(2) = 256
    #   new_prefill_tokens = ctx_tokens(384) - cached_total(256) = 128
    assert calls[0][0]["scheduled_requests"] == {
        "num_prefill_requests": 2,
        "sum_prefill_tokens": 128,
        "sum_prefill_kv_tokens": 256,
        "num_decode_requests": 7,
        "sum_decode_kv_tokens": 2688,
    }
    assert calls[1][0]["scheduled_requests"] == {
        "num_decode_requests": 7,
        "sum_decode_kv_tokens": 2688,
    }


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
    """Now that Rust reads parquet directly, the overlay is gone: the
    wrapper just hands its `AICONFIGURATOR_SYSTEMS_PATH` through unchanged
    to the Rust crate."""
    systems_root = tmp_path / "systems"
    systems_root.mkdir(parents=True)
    monkeypatch.setenv("AICONFIGURATOR_SYSTEMS_PATH", str(systems_root))
    rust_engine_step._configure_default_data_roots()
    assert Path(os.environ["AICONFIGURATOR_SYSTEMS_PATH"]) == systems_root


def test_engine_step_estimator_can_load_old_core_without_forward_pass_perf_symbols(tmp_path, monkeypatch) -> None:
    rust_engine_step._load_library.cache_clear()
    calls = []

    class _FakeFunc:
        def __init__(self, name, callback):
            self.name = name
            self.callback = callback
            self.argtypes = None
            self.restype = None

        def __call__(self, *args):
            calls.append(self.name)
            return self.callback(*args)

    class _OldCoreLib:
        def __init__(self):
            self.aic_engine_step_estimator_new = _FakeFunc("estimator_new", self._new_estimator)
            self.aic_engine_step_forward_pass_time_ms = _FakeFunc("forward_pass_time_ms", self._estimate)
            self.aic_engine_step_estimator_free = _FakeFunc("estimator_free", lambda handle: None)
            self.aic_engine_step_string_free = _FakeFunc("string_free", lambda message: None)

        def _new_estimator(self, config_json, out_handle):
            assert json.loads(config_json) == {"config": True}
            ctypes.cast(out_handle, ctypes.POINTER(ctypes.c_void_p))[0] = ctypes.c_void_p(123)
            return None

        def _estimate(self, handle, metrics_json, out_ms):
            assert handle.value == 123
            assert json.loads(metrics_json) == {"version": 1}
            ctypes.cast(out_ms, ctypes.POINTER(ctypes.c_double))[0] = ctypes.c_double(7.0)
            return None

    library_path = tmp_path / rust_engine_step._library_name()
    library_path.touch()
    fake_lib = _OldCoreLib()
    monkeypatch.setattr(rust_engine_step, "_find_library", lambda include_debug: library_path)
    monkeypatch.setattr(rust_engine_step.ctypes, "CDLL", lambda path: fake_lib)

    estimator = rust_engine_step.RustEngineStepEstimator({"config": True})
    assert estimator.forward_pass_time_ms({"version": 1}) == 7.0
    estimator.close()

    assert calls == ["estimator_new", "forward_pass_time_ms", "estimator_free"]
    with pytest.raises(rust_engine_step.RustCoreUnavailableError, match="ForwardPassPerfModel API"):
        rust_engine_step.RustForwardPassPerfModel.from_regression()
    rust_engine_step._load_library.cache_clear()


def test_forward_pass_perf_model_fake_library_wrapper(monkeypatch) -> None:
    calls = []
    buffers = []

    class _FakeFunc:
        def __init__(self, name, callback):
            self.name = name
            self.callback = callback
            self.argtypes = None
            self.restype = None

        def __call__(self, *args):
            calls.append(self.name)
            return self.callback(*args)

    class _FakeLib:
        def __init__(self):
            self.aic_forward_pass_perf_model_best_available = _FakeFunc("best_available", self._new_model)
            self.aic_forward_pass_perf_model_from_native = _FakeFunc("from_native", self._new_model)
            self.aic_forward_pass_perf_model_from_regression = _FakeFunc("from_regression", self._new_regression)
            self.aic_forward_pass_perf_model_estimate_forward_pass_time_ms = _FakeFunc("estimate", self._estimate)
            self.aic_forward_pass_perf_model_tune_with_fpms = _FakeFunc("tune", self._tune)
            self.aic_forward_pass_perf_model_diagnostics_json = _FakeFunc("diagnostics", self._diagnostics)
            self.aic_forward_pass_perf_model_min_correction_factor = _FakeFunc("min_factor", self._factor(1.5))
            self.aic_forward_pass_perf_model_max_correction_factor = _FakeFunc("max_factor", self._factor(2.5))
            self.aic_forward_pass_perf_model_avg_correction_factor = _FakeFunc("avg_factor", self._factor(2.0))
            self.aic_forward_pass_perf_model_free = _FakeFunc("free", lambda handle: None)
            self.aic_engine_step_string_free = _FakeFunc("string_free", lambda message: None)
            self.tuned_iterations = None

        def _write_handle(self, out_handle):
            ctypes.cast(out_handle, ctypes.POINTER(ctypes.c_void_p))[0] = ctypes.c_void_p(123)

        def _new_model(self, config_json, options_json, out_handle):
            assert json.loads(config_json) == {"config": True}
            if options_json is not None:
                assert json.loads(options_json) == {"min_observations": 2}
            self._write_handle(out_handle)
            return None

        def _new_regression(self, options_json, out_handle):
            assert json.loads(options_json) == {"min_observations": 2}
            self._write_handle(out_handle)
            return None

        def _estimate(self, handle, metrics_json, out_ms, out_has_value):
            assert handle.value == 123
            assert json.loads(metrics_json) == {"version": 1}
            ctypes.cast(out_ms, ctypes.POINTER(ctypes.c_double))[0] = ctypes.c_double(42.0)
            ctypes.cast(out_has_value, ctypes.POINTER(ctypes.c_bool))[0] = ctypes.c_bool(True)
            return None

        def _tune(self, handle, iterations_json):
            assert handle.value == 123
            self.tuned_iterations = json.loads(iterations_json)
            return None

        def _diagnostics(self, handle, out_json):
            payload = json.dumps({"source": "aic_with_correction"}).encode()
            buffer = ctypes.create_string_buffer(payload)
            buffers.append(buffer)
            ctypes.cast(out_json, ctypes.POINTER(ctypes.c_void_p))[0] = ctypes.cast(buffer, ctypes.c_void_p)
            return None

        def _factor(self, value):
            def callback(handle, out_value, out_has_value):
                ctypes.cast(out_value, ctypes.POINTER(ctypes.c_double))[0] = ctypes.c_double(value)
                ctypes.cast(out_has_value, ctypes.POINTER(ctypes.c_bool))[0] = ctypes.c_bool(True)
                return None

            return callback

    fake_lib = _FakeLib()
    monkeypatch.setattr(rust_engine_step, "_load_library", lambda autobuild: fake_lib)

    model = rust_engine_step.RustForwardPassPerfModel.best_available(
        {"config": True},
        {"min_observations": 2},
    )

    assert model.estimate_forward_pass_time_ms({"version": 1}) == 42.0
    model.tune_with_fpms({"version": 1})
    assert fake_lib.tuned_iterations == [[{"version": 1}]]
    model.tune_with_fpms([{"version": 1}, {"version": 1}])
    assert fake_lib.tuned_iterations == [[{"version": 1}, {"version": 1}]]
    model.tune_with_fpms([[{"version": 1}], [{"version": 1}]])
    assert fake_lib.tuned_iterations == [[{"version": 1}], [{"version": 1}]]
    assert model.diagnostics() == {"source": "aic_with_correction"}
    assert model.get_min_correction_factor() == 1.5
    assert model.get_max_correction_factor() == 2.5
    assert model.get_avg_correction_factor() == 2.0

    rust_engine_step.RustForwardPassPerfModel.from_native(
        {"config": True},
        {"min_observations": 2},
    ).close()
    rust_engine_step.RustForwardPassPerfModel.from_regression({"min_observations": 2}).close()
    model.close()

    assert "best_available" in calls
    assert "from_native" in calls
    assert "from_regression" in calls


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is required to build the Rust core shared library")
def test_ctypes_wrapper_calls_real_rust_core(tmp_path, monkeypatch) -> None:
    systems_root = tmp_path / "systems"
    data_root = systems_root / "data" / "test_sxm" / "vllm" / "1.0.0"
    model_configs_root = tmp_path / "model_configs"
    data_root.mkdir(parents=True)
    model_configs_root.mkdir()

    (systems_root / "test_sxm.yaml").write_text(
        "data_dir: data/test_sxm\n"
        "gpu:\n"
        "  mem_bw: 1000000000000000000000000000000\n"
        "  mem_bw_empirical_scaling_factor: 1.0\n"
        "  mem_empirical_constant_latency: 0.0\n"
        "node:\n"
        "  num_gpus_per_node: 8\n"
        "  inter_node_bw: 1000000000000000000000000000000\n"
        "  intra_node_bw: 1000000000000000000000000000000\n"
        "  p2p_latency: 0.0\n"
    )
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
    # The Rust crate reads parquet directly (no CSV fallback); the synthetic
    # fixtures here mirror the row schemas of the production *_perf.parquet
    # files. Integer columns are INT64 in the real perf-DB; encode them that
    # way here so PerfReader's u32-narrowing path is exercised.
    pq.write_table(
        pa.table(
            {
                "gemm_dtype": ["bfloat16", "bfloat16", "bfloat16", "bfloat16", "bfloat16"],
                "m": pa.array([20, 20, 20, 20, 2], type=pa.int64()),
                "n": pa.array([64, 32, 128, 32, 160], type=pa.int64()),
                "k": pa.array([32, 32, 32, 64, 32], type=pa.int64()),
                "latency": [1.0, 2.0, 3.0, 4.0, 0.5],
            }
        ),
        data_root / "gemm_perf.parquet",
        compression="zstd",
    )
    pq.write_table(
        pa.table(
            {
                "attn_dtype": ["bfloat16"],
                "kv_cache_dtype": ["bfloat16"],
                "batch_size": pa.array([2], type=pa.int64()),
                "isl": pa.array([10], type=pa.int64()),
                "num_heads": pa.array([4], type=pa.int64()),
                "num_key_value_heads": pa.array([2], type=pa.int64()),
                "head_dim": pa.array([8], type=pa.int64()),
                "latency": [5.0],
            }
        ),
        data_root / "context_attention_perf.parquet",
        compression="zstd",
    )
    pq.write_table(
        pa.table(
            {
                "attn_dtype": ["bfloat16"],
                "kv_cache_dtype": ["bfloat16"],
                "batch_size": pa.array([2], type=pa.int64()),
                "isl": pa.array([16], type=pa.int64()),
                "num_heads": pa.array([4], type=pa.int64()),
                "num_key_value_heads": pa.array([2], type=pa.int64()),
                "head_dim": pa.array([8], type=pa.int64()),
                "step": pa.array([1], type=pa.int64()),
                "latency": [0.7],
            }
        ),
        data_root / "generation_attention_perf.parquet",
        compression="zstd",
    )

    monkeypatch.setenv("AICONFIGURATOR_SYSTEMS_PATH", str(systems_root))
    monkeypatch.setenv("AICONFIGURATOR_MODEL_CONFIGS_PATH", str(model_configs_root))
    monkeypatch.setenv("AICONFIGURATOR_RUST_CORE_AUTOBUILD", "1")
    rust_engine_step._load_library.cache_clear()

    config = {
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
    estimator = rust_engine_step.RustEngineStepEstimator(config)

    metrics = [
        {
            "version": 1,
            "scheduled_requests": {
                "num_prefill_requests": 2,
                "sum_prefill_tokens": 20,
                "sum_prefill_kv_tokens": 0,
            },
        },
    ]
    latency_ms = estimator.forward_pass_time_ms(metrics)

    assert latency_ms == pytest.approx(30.5)

    regression = rust_engine_step.RustForwardPassPerfModel.from_regression({"min_observations": 2})
    assert regression.estimate_forward_pass_time_ms(metrics) is None
    regression.tune_with_fpms(
        [
            [
                {
                    "version": 1,
                    "wall_time": 0.010,
                    "scheduled_requests": {
                        "num_prefill_requests": 1,
                        "sum_prefill_tokens": 10,
                    },
                }
            ],
            [
                {
                    "version": 1,
                    "wall_time": 0.020,
                    "scheduled_requests": {
                        "num_prefill_requests": 1,
                        "sum_prefill_tokens": 20,
                    },
                }
            ],
        ]
    )
    assert regression.estimate_forward_pass_time_ms(
        {
            "version": 1,
            "scheduled_requests": {
                "num_prefill_requests": 1,
                "sum_prefill_tokens": 30,
            },
        }
    ) == pytest.approx(30.0)
    assert regression.get_min_correction_factor() is None

    tuned = rust_engine_step.RustForwardPassPerfModel.best_available(config, {"min_observations": 2})
    assert tuned.estimate_forward_pass_time_ms(metrics) == pytest.approx(30.5)
    tuned.tune_with_fpms(
        [
            [
                {
                    **metrics[0],
                    "wall_time": 0.061,
                }
            ],
            [
                {
                    **metrics[0],
                    "wall_time": 0.061,
                }
            ],
        ]
    )
    assert tuned.estimate_forward_pass_time_ms(metrics) == pytest.approx(61.0)
    assert tuned.get_min_correction_factor() == pytest.approx(2.0)
    assert tuned.get_max_correction_factor() == pytest.approx(2.0)
    assert tuned.get_avg_correction_factor() == pytest.approx(2.0)
    assert tuned.diagnostics()["source"] == "aic_with_correction"
