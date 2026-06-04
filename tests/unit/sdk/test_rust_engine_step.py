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
        def mixed_step_latency(self, *args):
            mixed_calls.append(args)
            return 8.5

        def decode_step_latency(self, *args):
            decode_calls.append(args)
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
    assert mixed_calls == [(384, 7, 256, 256, 128)]
    assert decode_calls == [(7, 256, 256)]


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
