# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import shutil
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


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is required to build the Rust core shared library")
def test_ctypes_wrapper_calls_real_rust_core(tmp_path, monkeypatch) -> None:
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
    monkeypatch.setenv("AICONFIGURATOR_RUST_CORE_AUTOBUILD", "1")
    rust_engine_step._load_library.cache_clear()

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
