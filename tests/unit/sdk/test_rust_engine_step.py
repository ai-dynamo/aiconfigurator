# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

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


def test_configure_data_roots_materializes_parquet_perf_for_rust(tmp_path, monkeypatch) -> None:
    systems_root = tmp_path / "systems"
    data_root = systems_root / "data" / "test_sxm" / "vllm" / "1.0.0"
    data_root.mkdir(parents=True)
    (systems_root / "test_sxm.yaml").write_text(
        "data_dir: data/test_sxm\n"
        "gpu:\n"
        "  mem_bw: 1\n"
        "node:\n"
        "  num_gpus_per_node: 8\n"
        "  inter_node_bw: 1\n"
        "  intra_node_bw: 1\n"
        "misc:\n"
        "  nccl_version: '2.27.3'\n"
    )
    pq.write_table(
        pa.table(
            {
                "gemm_dtype": ["bfloat16"],
                "m": [1],
                "n": [2],
                "k": [3],
                "latency": [4.0],
            }
        ),
        data_root / "gemm_perf.parquet",
    )

    monkeypatch.setenv("AICONFIGURATOR_SYSTEMS_PATH", str(systems_root))
    monkeypatch.delenv("AICONFIGURATOR_RUST_SYSTEMS_SOURCE_PATH", raising=False)
    rust_engine_step._configure_default_data_roots(
        {
            "system_name": "test_sxm",
            "backend": "vllm",
            "backend_version": "1.0.0",
        }
    )

    rust_systems_root = Path(os.environ["AICONFIGURATOR_SYSTEMS_PATH"])
    materialized = rust_systems_root / "data" / "test_sxm" / "vllm" / "1.0.0" / "gemm_perf.txt"

    assert rust_systems_root != systems_root
    assert (rust_systems_root / "test_sxm.yaml").is_file()
    assert materialized.read_text().splitlines()[0] == '"gemm_dtype","m","n","k","latency"'
    assert os.environ["AICONFIGURATOR_RUST_SYSTEMS_SOURCE_PATH"] == str(systems_root)


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
