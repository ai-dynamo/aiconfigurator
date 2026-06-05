# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiconfigurator.sdk.backends import base_backend
from aiconfigurator.sdk.backends.vllm_backend import VLLMBackend
from aiconfigurator.sdk.config import RuntimeConfig

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(("use_layerwise", "expected_backend"), [(False, "rust"), (True, "python")])
def test_vllm_layerwise_forces_python_static_engine_step(monkeypatch, use_layerwise, expected_backend) -> None:
    from aiconfigurator.sdk.backends import vllm_backend

    captured_backends = []
    monkeypatch.setattr(vllm_backend, "_USE_LAYERWISE", use_layerwise)

    def _fake_run_static_breakdown(
        self,
        model,
        database,
        runtime_config,
        mode,
        stride=32,
        latency_correction_scale=1.0,
        img_ctx_tokens=0,
    ):
        del self, model, database, mode, stride, latency_correction_scale, img_ctx_tokens
        captured_backends.append(runtime_config.engine_step_backend)
        return {}, {}, {}, {}, {}, {}

    monkeypatch.setattr(base_backend.BaseBackend, "_run_static_breakdown", _fake_run_static_breakdown)

    VLLMBackend()._run_static_breakdown(
        model=object(),
        database=object(),
        runtime_config=RuntimeConfig(engine_step_backend="rust"),
        mode="static_gen",
    )

    assert captured_backends == [expected_backend]


@pytest.mark.parametrize(("use_layerwise", "expected_backend"), [(False, "rust"), (True, "python")])
def test_vllm_layerwise_forces_python_agg_engine_step(monkeypatch, use_layerwise, expected_backend) -> None:
    from aiconfigurator.sdk.backends import vllm_backend

    captured_backends = []
    expected_summary = object()
    monkeypatch.setattr(vllm_backend, "_USE_LAYERWISE", use_layerwise)

    def _fake_run_agg(self, model, database, runtime_config, **kwargs):
        del self, model, database, kwargs
        captured_backends.append(runtime_config.engine_step_backend)
        return expected_summary

    monkeypatch.setattr(base_backend.BaseBackend, "run_agg", _fake_run_agg)

    summary = VLLMBackend().run_agg(
        model=object(),
        database=object(),
        runtime_config=RuntimeConfig(engine_step_backend="rust"),
        ctx_tokens=8,
    )

    assert summary is expected_summary
    assert captured_backends == [expected_backend]


def test_vllm_layerwise_mixed_step_uses_whole_layer_tables(monkeypatch) -> None:
    from aiconfigurator.sdk.backends import vllm_backend

    monkeypatch.setattr(vllm_backend, "_USE_LAYERWISE", True)

    class _Config:
        tp_size = 2
        pp_size = 1

    class _Model:
        model_path = "Qwen/Qwen3-32B"
        config = _Config()
        _num_layers = 4
        _hidden_size = 5120

    class _Database:
        def query_layerwise(self, model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache=0):
            assert model == "Qwen/Qwen3-32B"
            assert tp_size == 2
            if phase == "CTX" and batch_size == 1 and seq_len == 10 and seq_len_kv_cache == 0:
                return 1.0
            if phase == "CTX" and batch_size == 1 and seq_len == 2 and seq_len_kv_cache == 0:
                return 0.3
            if phase == "GEN" and batch_size == 2 and seq_len == 1032:
                return 0.7
            raise AssertionError((phase, batch_size, seq_len, seq_len_kv_cache))

        def query_custom_allreduce(self, quant_mode, tp_size, size):
            del quant_mode
            assert tp_size == 2
            assert size in {10 * 5120}
            return 0.1

    latency_ms, energy_wms, per_ops, sources = VLLMBackend()._get_mix_step_latency(
        _Model(),
        _Database(),
        RuntimeConfig(),
        ctx_tokens=8,
        gen_tokens=2,
        isl=1024,
        osl=16,
        prefix=0,
    )

    assert latency_ms == pytest.approx((1.0 * 4) + (0.1 * 2 * 4) + ((0.7 - 0.3) * 4))
    assert energy_wms == 0.0
    assert per_ops == {
        "mixed_layerwise_context_combined": 4.0,
        "mixed_layerwise_context_tp_allreduce": pytest.approx(0.8),
        "mixed_layerwise_decode_delta": pytest.approx(1.6),
    }
    assert set(sources) == set(per_ops)


def test_vllm_layerwise_context_uses_chunked_prefill_kv(monkeypatch) -> None:
    from aiconfigurator.sdk.backends import vllm_backend

    monkeypatch.setattr(vllm_backend, "_USE_LAYERWISE", True)

    class _Config:
        tp_size = 2
        pp_size = 1

    class _Model:
        model_path = "Qwen/Qwen3-32B"
        config = _Config()
        _num_layers = 4

    class _Database:
        def query_layerwise(self, model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache=0):
            assert model == "Qwen/Qwen3-32B"
            assert phase == "CTX"
            assert tp_size == 2
            assert batch_size == 1
            if seq_len == 8192 and seq_len_kv_cache == 0:
                return 1.0
            if seq_len == 8192 and seq_len_kv_cache == 8192:
                return 1.5
            raise AssertionError((seq_len, seq_len_kv_cache))

    latency, _energy, _sources = VLLMBackend()._run_context_phase(
        _Model(),
        _Database(),
        RuntimeConfig(),
        batch_size=1,
        isl=16384,
        prefix=0,
    )

    assert latency["context_layerwise"] == pytest.approx((1.0 + 1.5) * 4)
