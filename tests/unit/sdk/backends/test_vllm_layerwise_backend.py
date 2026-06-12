# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.backends import base_backend
from aiconfigurator.sdk.backends.vllm_backend import VLLMBackend
from aiconfigurator.sdk.config import RuntimeConfig
from aiconfigurator.sdk.operations.layerwise import load_layerwise_data
from aiconfigurator.sdk.perf_database import PerfDataNotAvailableError
from aiconfigurator.sdk.performance_result import PerformanceResult

pytestmark = pytest.mark.unit


def test_layerwise_loader_merges_duplicate_representative_shapes(tmp_path) -> None:
    path = tmp_path / "layerwise_perf.csv"
    path.write_text(
        "\n".join(
            [
                "model,phase,attn_tp,batch_size,new_tokens,past_kv,latency_ms,"
                "rms_latency_ms,rms_kernel_count,layer_type,measured_layer_count,layer_multiplier,includes_moe",
                "Qwen/Qwen3.6-35B-A3B,GEN,1,8,1,4096,0.10,0.01,2,linear_attention_moe,1,30,true",
                "Qwen/Qwen3.6-35B-A3B,GEN,1,8,1,4096,0.20,0.02,3,full_attention_moe,1,10,true",
            ]
        )
        + "\n"
    )

    data = load_layerwise_data(str(path))
    detail = data["qwen/qwen3.6-35b-a3b"]["GEN"][1][8][4096]

    assert detail["latency"] == pytest.approx((0.10 * 30) + (0.20 * 10))
    assert detail["rms_latency"] == pytest.approx((0.01 * 30) + (0.02 * 10))
    assert detail["rms_kernel_count"] == 5
    assert detail["includes_moe"] is True
    assert detail["layer_type"] == "combined"
    assert detail["measured_layer_count"] == 1.0
    assert detail["layer_multiplier"] == 1.0
    assert [component["layer_type"] for component in detail["components"]] == [
        "linear_attention_moe",
        "full_attention_moe",
    ]


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


def test_vllm_layerwise_mixed_step_uses_context_prefix(monkeypatch) -> None:
    from aiconfigurator.sdk.backends import vllm_backend

    monkeypatch.setattr(vllm_backend, "_USE_LAYERWISE", True)

    class _Config:
        tp_size = 1
        pp_size = 1
        moe_tp_size = 1
        moe_ep_size = 1

    class _Model:
        model_path = "Qwen/Qwen3-32B"
        config = _Config()
        _num_layers = 4
        _hidden_size = 5120
        _nextn = 0

    class _Database:
        def query_layerwise(self, model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache=0):
            assert model == "Qwen/Qwen3-32B"
            assert tp_size == 1
            if phase == "CTX" and batch_size == 1 and seq_len == 10 and seq_len_kv_cache == 4096:
                return 1.5
            if phase == "CTX" and batch_size == 1 and seq_len == 2 and seq_len_kv_cache == 0:
                return 0.3
            if phase == "GEN" and batch_size == 2 and seq_len == 1032:
                return 0.7
            raise AssertionError((phase, batch_size, seq_len, seq_len_kv_cache))

    latency_ms, energy_wms, per_ops, sources = VLLMBackend()._get_mix_step_latency(
        _Model(),
        _Database(),
        RuntimeConfig(),
        ctx_tokens=8,
        gen_tokens=2,
        isl=1024,
        osl=16,
        prefix=4096,
    )

    assert latency_ms == pytest.approx((1.5 * 4) + ((0.7 - 0.3) * 4))
    assert energy_wms == 0.0
    assert per_ops == {
        "mixed_layerwise_context_combined": 6.0,
        "mixed_layerwise_context_tp_allreduce": 0.0,
        "mixed_layerwise_decode_delta": pytest.approx(1.6),
    }
    assert set(sources) == set(per_ops)


def test_vllm_layerwise_dense_mixed_scheduler_context_covers_small_decode_delta(monkeypatch) -> None:
    from aiconfigurator.sdk.backends import vllm_backend

    monkeypatch.setattr(vllm_backend, "_USE_LAYERWISE", True)

    class _Config:
        tp_size = 1
        pp_size = 1
        moe_tp_size = 1
        moe_ep_size = 1

    class _Model:
        model_path = "Qwen/Qwen3-32B"
        config = _Config()
        _num_layers = 4
        _hidden_size = 5120
        _nextn = 0

    class _Database:
        def query_layerwise_detail(self, model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache=0):
            assert model == "Qwen/Qwen3-32B"
            assert tp_size == 1
            assert batch_size == 1
            assert phase == "CTX"
            assert (seq_len, seq_len_kv_cache) == (8192, 0)
            return {
                "latency": 100.0,
                "energy": 0.0,
                "latency_source": "schedule_to_update",
                "physical_gpus": 1,
            }

        def query_layerwise(self, *args, **kwargs):
            raise AssertionError(f"dense scheduler mixed row should not query a GEN decode delta: {args}")

    latency_ms, energy_wms, per_ops, sources = VLLMBackend()._get_mix_step_latency(
        _Model(),
        _Database(),
        RuntimeConfig(),
        ctx_tokens=8190,
        gen_tokens=2,
        isl=4096,
        osl=2,
        prefix=0,
    )

    assert latency_ms == pytest.approx(100.0 * 4)
    assert energy_wms == 0.0
    assert per_ops == {
        "mixed_layerwise_context_combined": 400.0,
        "mixed_layerwise_context_tp_allreduce": 0.0,
        "mixed_layerwise_decode_delta": 0.0,
    }
    assert set(sources) == set(per_ops)


def test_vllm_layerwise_mixed_prefers_scheduler_context_over_prefix_kernel_row(monkeypatch) -> None:
    from aiconfigurator.sdk.backends import vllm_backend

    monkeypatch.setattr(vllm_backend, "_USE_LAYERWISE", True)

    class _Config:
        tp_size = 1
        pp_size = 1

    class _Model:
        model_path = "Qwen/Qwen3-32B"
        config = _Config()
        _num_layers = 4
        _hidden_size = 5120

    class _Database:
        def query_layerwise_detail(self, model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache=0):
            assert model == "Qwen/Qwen3-32B"
            assert phase == "CTX"
            assert tp_size == 1
            assert batch_size == 1
            assert seq_len == 8192
            if seq_len_kv_cache == 1365:
                return {"latency": 0.1, "energy": 0.0}
            if seq_len_kv_cache == 0:
                return {
                    "latency": 10.0,
                    "energy": 0.0,
                    "latency_source": "schedule_to_update",
                }
            raise AssertionError(seq_len_kv_cache)

    latency_ms, _energy_wms, per_ops, _sources = VLLMBackend()._get_mix_step_latency(
        _Model(),
        _Database(),
        RuntimeConfig(),
        ctx_tokens=8192,
        gen_tokens=0,
        isl=4096,
        osl=1,
        prefix=1365,
    )

    assert latency_ms == pytest.approx(40.0)
    assert per_ops["mixed_layerwise_context_combined"] == pytest.approx(40.0)


def test_vllm_layerwise_mixed_prefers_large_fresh_context_surface_without_source(monkeypatch) -> None:
    from aiconfigurator.sdk.backends import vllm_backend

    monkeypatch.setattr(vllm_backend, "_USE_LAYERWISE", True)

    class _Config:
        tp_size = 1
        pp_size = 1

    class _Model:
        model_path = "Qwen/Qwen3-32B"
        config = _Config()
        _num_layers = 4
        _hidden_size = 5120

    class _Database:
        def query_layerwise_detail(self, model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache=0):
            assert model == "Qwen/Qwen3-32B"
            assert phase == "CTX"
            assert tp_size == 1
            assert batch_size == 1
            assert seq_len == 4099
            if seq_len_kv_cache == 2048:
                return {"latency": 0.1, "energy": 0.0}
            if seq_len_kv_cache == 0:
                return {"latency": 10.0, "energy": 0.0}
            raise AssertionError(seq_len_kv_cache)

    latency_ms, _energy_wms, per_ops, _sources = VLLMBackend()._get_mix_step_latency(
        _Model(),
        _Database(),
        RuntimeConfig(),
        ctx_tokens=4099,
        gen_tokens=0,
        isl=4096,
        osl=1,
        prefix=2048,
    )

    assert latency_ms == pytest.approx(40.0)
    assert per_ops["mixed_layerwise_context_combined"] == pytest.approx(40.0)


def test_vllm_layerwise_mixed_keeps_decode_delta_when_tiny_context_subtraction_missing(monkeypatch) -> None:
    from aiconfigurator.sdk.backends import vllm_backend

    monkeypatch.setattr(vllm_backend, "_USE_LAYERWISE", True)

    class _Config:
        tp_size = 1
        pp_size = 1

    class _Model:
        model_path = "Qwen/Qwen3-32B"
        config = _Config()
        _num_layers = 4
        _hidden_size = 5120

    class _Database:
        def query_layerwise_detail(self, model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache=0):
            assert model == "Qwen/Qwen3-32B"
            assert tp_size == 1
            if phase == "CTX" and batch_size == 1 and seq_len == 5 and seq_len_kv_cache == 4094:
                return {"latency": 0.1, "energy": 0.0}
            if phase == "CTX" and batch_size == 1 and seq_len == 3 and seq_len_kv_cache == 0:
                raise ValueError("no tiny context row")
            if phase == "GEN" and batch_size == 3 and seq_len == 4096:
                return {"latency": 1.0, "energy": 0.0}
            if phase == "GEN" and batch_size == 3 and seq_len == 4097:
                raise ValueError("nearest decode fallback")
            raise AssertionError((phase, batch_size, seq_len, seq_len_kv_cache))

    latency_ms, _energy_wms, per_ops, _sources = VLLMBackend()._get_mix_step_latency(
        _Model(),
        _Database(),
        RuntimeConfig(),
        ctx_tokens=2,
        gen_tokens=3,
        isl=4097,
        osl=1,
        prefix=4094,
    )

    assert latency_ms == pytest.approx((0.1 * 4) + (1.0 * 4))
    assert per_ops["mixed_layerwise_decode_delta"] == pytest.approx(4.0)


def test_vllm_layerwise_mixed_uses_prefix_context_chunk_envelope(monkeypatch) -> None:
    from aiconfigurator.sdk.backends import vllm_backend

    monkeypatch.setattr(vllm_backend, "_USE_LAYERWISE", True)

    class _Config:
        tp_size = 2
        pp_size = 1
        moe_tp_size = 1
        moe_ep_size = 2

    class _Model:
        model_path = "Qwen/Qwen3.6-35B-A3B"
        config = _Config()
        _num_layers = 4
        _hidden_size = 5120
        _num_experts = 256

    class _Database:
        def query_layerwise_detail(self, model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache=0):
            assert model == "Qwen/Qwen3.6-35B-A3B"
            assert phase == "CTX"
            assert tp_size == 2
            assert batch_size == 1
            if seq_len == 931 and seq_len_kv_cache == 3168:
                raise ValueError("combined mixed context row was not collected")
            if seq_len == 928 and seq_len_kv_cache == 3168:
                return {
                    "latency": 10.0,
                    "energy": 0.0,
                    "latency_source": "schedule_to_update",
                    "physical_gpus": 2,
                    "includes_moe": True,
                }
            raise AssertionError((phase, batch_size, seq_len, seq_len_kv_cache))

        def query_custom_allreduce(self, quant_mode, tp_size, size):
            raise AssertionError("physical scheduler envelope already covers TP")

    latency_ms, _energy_wms, per_ops, _sources = VLLMBackend()._get_mix_step_latency(
        _Model(),
        _Database(),
        RuntimeConfig(),
        ctx_tokens=928,
        gen_tokens=3,
        isl=4096,
        osl=1,
        prefix=3168,
    )

    assert latency_ms == pytest.approx(40.0)
    assert per_ops == {
        "mixed_layerwise_context_combined": 40.0,
        "mixed_layerwise_context_tp_allreduce": 0.0,
        "mixed_layerwise_decode_delta": 0.0,
    }


def test_vllm_layerwise_mixed_moe_ep_envelope_skips_small_decode_delta(monkeypatch) -> None:
    from aiconfigurator.sdk.backends import vllm_backend

    monkeypatch.setattr(vllm_backend, "_USE_LAYERWISE", True)

    class _Config:
        tp_size = 2
        pp_size = 1
        moe_tp_size = 1
        moe_ep_size = 2

    class _Model:
        model_path = "Qwen/Qwen3.6-35B-A3B"
        config = _Config()
        _num_layers = 4
        _hidden_size = 5120
        _num_experts = 256

    class _Database:
        def query_layerwise_detail(self, model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache=0):
            assert model == "Qwen/Qwen3.6-35B-A3B"
            assert tp_size == 2
            if phase == "CTX" and batch_size == 1 and seq_len == 931 and seq_len_kv_cache == 3168:
                raise ValueError("combined mixed context row was not collected")
            if phase == "CTX" and batch_size == 1 and seq_len == 928 and seq_len_kv_cache == 3168:
                return {
                    "latency": 10.0,
                    "energy": 0.0,
                    "latency_source": "schedule_to_update",
                    "physical_gpus": 1,
                    "includes_moe": True,
                }
            if phase == "GEN":
                raise AssertionError("small decode work should be covered by the MoE scheduler envelope")
            raise AssertionError((phase, batch_size, seq_len, seq_len_kv_cache))

        def query_custom_allreduce(self, quant_mode, tp_size, size):
            raise AssertionError("MoE EP scheduler envelope already covers this mixed-step TP tail")

    latency_ms, _energy_wms, per_ops, _sources = VLLMBackend()._get_mix_step_latency(
        _Model(),
        _Database(),
        RuntimeConfig(),
        ctx_tokens=928,
        gen_tokens=3,
        isl=4096,
        osl=1,
        prefix=3168,
    )

    assert latency_ms == pytest.approx(40.0)
    assert per_ops == {
        "mixed_layerwise_context_combined": 40.0,
        "mixed_layerwise_context_tp_allreduce": 0.0,
        "mixed_layerwise_decode_delta": 0.0,
    }


def test_vllm_layerwise_mixed_ep_high_decode_adds_second_decode_step(monkeypatch) -> None:
    from aiconfigurator.sdk.backends import vllm_backend

    monkeypatch.setattr(vllm_backend, "_USE_LAYERWISE", True)

    class _Config:
        tp_size = 4
        pp_size = 1
        moe_tp_size = 1
        moe_ep_size = 4

    class _Model:
        model_path = "Qwen/Qwen3.6-35B-A3B"
        config = _Config()
        _num_layers = 4
        _hidden_size = 2048
        _num_experts = 256
        _topk = 8

    class _Database:
        def query_layerwise_detail(self, model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache=0):
            assert model == "Qwen/Qwen3.6-35B-A3B"
            assert tp_size == 4
            if phase == "CTX" and batch_size == 1 and seq_len == 3568 and seq_len_kv_cache == 1452:
                return {
                    "latency": 10.0,
                    "energy": 0.0,
                    "latency_source": "schedule_to_update",
                    "physical_gpus": 1,
                    "includes_moe": True,
                }
            if phase == "GEN" and batch_size == 9 and seq_len == 4096:
                return {"latency": 1.0, "energy": 0.0, "includes_moe": True}
            raise AssertionError((phase, batch_size, seq_len, seq_len_kv_cache))

    latency_ms, _energy_wms, per_ops, _sources = VLLMBackend()._get_mix_step_latency(
        _Model(),
        _Database(),
        RuntimeConfig(),
        ctx_tokens=3568,
        gen_tokens=9,
        isl=4096,
        osl=1,
        prefix=1452,
    )

    assert latency_ms == pytest.approx(40.0 + 8.0)
    assert per_ops == {
        "mixed_layerwise_context_combined": 40.0,
        "mixed_layerwise_context_tp_allreduce": 0.0,
        "mixed_layerwise_decode_delta": 8.0,
    }


def test_vllm_layerwise_mixed_ep_short_context_high_decode_adds_extra_decode_slices(monkeypatch) -> None:
    from aiconfigurator.sdk.backends import vllm_backend

    monkeypatch.setattr(vllm_backend, "_USE_LAYERWISE", True)

    class _Config:
        tp_size = 4
        pp_size = 1
        moe_tp_size = 1
        moe_ep_size = 4

    class _Model:
        model_path = "Qwen/Qwen3.6-35B-A3B"
        config = _Config()
        _num_layers = 4
        _hidden_size = 2048
        _num_experts = 256
        _topk = 8

    class _Database:
        def query_layerwise_detail(self, model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache=0):
            assert model == "Qwen/Qwen3.6-35B-A3B"
            assert tp_size == 4
            if phase == "CTX" and batch_size == 1 and seq_len == 156 and seq_len_kv_cache == 0:
                return {
                    "latency": 10.0,
                    "energy": 0.0,
                    "latency_source": "schedule_to_update",
                    "physical_gpus": 1,
                    "includes_moe": True,
                }
            if phase == "GEN" and batch_size == 15 and seq_len == 4096:
                return {"latency": 1.0, "energy": 0.0, "includes_moe": True}
            raise AssertionError((phase, batch_size, seq_len, seq_len_kv_cache))

    latency_ms, _energy_wms, per_ops, _sources = VLLMBackend()._get_mix_step_latency(
        _Model(),
        _Database(),
        RuntimeConfig(),
        ctx_tokens=141,
        gen_tokens=15,
        isl=4096,
        osl=1,
        prefix=0,
    )

    assert latency_ms == pytest.approx(40.0 + 24.0)
    assert per_ops == {
        "mixed_layerwise_context_combined": 40.0,
        "mixed_layerwise_context_tp_allreduce": 0.0,
        "mixed_layerwise_decode_delta": 24.0,
    }


def test_vllm_layerwise_mixed_ep_fresh_full_prefill_high_decode_uses_context_envelope(monkeypatch) -> None:
    from aiconfigurator.sdk.backends import vllm_backend

    monkeypatch.setattr(vllm_backend, "_USE_LAYERWISE", True)

    class _Config:
        tp_size = 4
        pp_size = 1
        moe_tp_size = 1
        moe_ep_size = 4

    class _Model:
        model_path = "Qwen/Qwen3.6-35B-A3B"
        config = _Config()
        _num_layers = 4
        _hidden_size = 2048
        _num_experts = 256
        _topk = 8

    class _Database:
        def query_layerwise_detail(self, model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache=0):
            assert model == "Qwen/Qwen3.6-35B-A3B"
            assert tp_size == 4
            if phase == "CTX" and batch_size == 1 and seq_len == 6015 and seq_len_kv_cache == 0:
                return {
                    "latency": 10.0,
                    "energy": 0.0,
                    "latency_source": "schedule_to_update",
                    "physical_gpus": 1,
                    "includes_moe": True,
                    "max_num_batched_tokens": 7920,
                }
            if phase == "GEN" and batch_size == 15 and seq_len == 4096:
                raise AssertionError("full fresh prefill envelope should cover high decode")
            raise AssertionError((phase, batch_size, seq_len, seq_len_kv_cache))

    latency_ms, _energy_wms, per_ops, _sources = VLLMBackend()._get_mix_step_latency(
        _Model(),
        _Database(),
        RuntimeConfig(),
        ctx_tokens=6000,
        gen_tokens=15,
        isl=4096,
        osl=1,
        prefix=0,
    )

    assert latency_ms == pytest.approx(40.0)
    assert per_ops == {
        "mixed_layerwise_context_combined": 40.0,
        "mixed_layerwise_context_tp_allreduce": 0.0,
        "mixed_layerwise_decode_delta": 0.0,
    }


def test_vllm_layerwise_mixed_ep_high_decode_uses_saturated_chunk_floor(monkeypatch) -> None:
    from aiconfigurator.sdk.backends import vllm_backend

    monkeypatch.setattr(vllm_backend, "_USE_LAYERWISE", True)

    class _Config:
        tp_size = 8
        pp_size = 1
        moe_tp_size = 1
        moe_ep_size = 8

    class _Model:
        model_path = "Qwen/Qwen3.6-35B-A3B"
        config = _Config()
        _num_layers = 4
        _hidden_size = 2048
        _num_experts = 256
        _topk = 8

    class _Database:
        def query_layerwise_detail(self, model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache=0):
            assert model == "Qwen/Qwen3.6-35B-A3B"
            assert tp_size == 8
            if phase == "CTX" and batch_size == 1 and seq_len == 559 and seq_len_kv_cache == 0:
                return {
                    "latency": 5.0,
                    "energy": 0.0,
                    "latency_source": "schedule_to_update",
                    "physical_gpus": 1,
                    "includes_moe": True,
                    "max_num_batched_tokens": 8192,
                }
            if phase == "CTX" and batch_size == 1 and seq_len == 8192 and seq_len_kv_cache == 0:
                return {
                    "latency": 15.0,
                    "energy": 0.0,
                    "latency_source": "schedule_to_update",
                    "physical_gpus": 1,
                    "includes_moe": True,
                    "max_num_batched_tokens": 8192,
                }
            if phase == "GEN" and batch_size == 15 and seq_len == 4096:
                return {"latency": 1.0, "energy": 0.0, "includes_moe": True}
            raise AssertionError((phase, batch_size, seq_len, seq_len_kv_cache))

    latency_ms, _energy_wms, per_ops, _sources = VLLMBackend()._get_mix_step_latency(
        _Model(),
        _Database(),
        RuntimeConfig(vllm_max_num_batched_tokens=8192),
        ctx_tokens=544,
        gen_tokens=15,
        isl=4096,
        osl=1,
        prefix=0,
    )

    assert latency_ms == pytest.approx(60.0)
    assert per_ops == {
        "mixed_layerwise_context_combined": 20.0,
        "mixed_layerwise_context_tp_allreduce": 0.0,
        "mixed_layerwise_decode_delta": 8.0,
        "mixed_layerwise_ep_high_decode_floor": 60.0,
    }


def test_vllm_layerwise_generation_adds_tp_allreduce(monkeypatch) -> None:
    from aiconfigurator.sdk.backends import vllm_backend

    monkeypatch.setattr(vllm_backend, "_USE_LAYERWISE", True)

    class _Config:
        tp_size = 8
        pp_size = 1

    class _Model:
        model_path = "Qwen/Qwen3-32B"
        config = _Config()
        _num_layers = 4
        _hidden_size = 5120
        _nextn = 0

    class _Database:
        def query_layerwise(self, model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache=0):
            assert model == "Qwen/Qwen3-32B"
            assert phase == "GEN"
            assert tp_size == 8
            assert batch_size == 16
            assert seq_len_kv_cache == 0
            if seq_len == 1024:
                return 1.0
            if seq_len == 1056:
                return 1.5
            raise AssertionError((phase, batch_size, seq_len, seq_len_kv_cache))

        def query_custom_allreduce(self, quant_mode, tp_size, size):
            del quant_mode
            assert tp_size == 8
            assert size == 16 * 5120
            return 0.1

        def query_allreduce_rms(self, quant_mode, tp_size, size, hidden_size):
            del quant_mode
            assert tp_size == 8
            assert size == 16 * 5120
            assert hidden_size == 5120
            return 0.07

    latency_ms, energy_wms, sources = VLLMBackend()._run_generation_phase(
        _Model(),
        _Database(),
        RuntimeConfig(),
        batch_size=16,
        beam_width=1,
        isl=1024,
        osl=65,
        stride=32,
    )

    expected_layerwise = (1.0 * 4 * 32) + (1.5 * 4 * 32)
    expected_allreduce = ((0.1 + 0.07) * 4 * 32) + ((0.1 + 0.07) * 4 * 32)
    assert latency_ms["generation_layerwise"] == pytest.approx(expected_layerwise)
    assert latency_ms["generation_tp_allreduce"] == pytest.approx(expected_allreduce)
    assert energy_wms["generation_layerwise"] == 0.0
    assert energy_wms["generation_tp_allreduce"] == 0.0
    assert sources == {
        "generation_layerwise": "silicon",
        "generation_tp_allreduce": "silicon",
    }


def test_vllm_layerwise_generation_uses_fused_allreduce_rms_as_second_comm_slot(monkeypatch) -> None:
    from aiconfigurator.sdk.backends import vllm_backend

    monkeypatch.setattr(vllm_backend, "_USE_LAYERWISE", True)

    class _Config:
        tp_size = 8
        pp_size = 1

    class _Model:
        model_path = "Qwen/Qwen3-32B"
        config = _Config()
        _num_layers = 4
        _hidden_size = 5120
        _nextn = 0

    class _Database:
        def query_layerwise_detail(self, model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache=0):
            assert model == "Qwen/Qwen3-32B"
            assert phase == "GEN"
            assert tp_size == 8
            assert batch_size == 16
            assert seq_len_kv_cache == 0
            assert seq_len == 1024
            return {"latency": 1.0, "energy": 0.0, "rms_latency": 0.1}

        def query_allreduce_rms(self, quant_mode, tp_size, size, hidden_size):
            del quant_mode
            assert tp_size == 8
            assert size == 16 * 5120
            assert hidden_size == 5120
            return 0.2

        def query_custom_allreduce(self, quant_mode, tp_size, size):
            del quant_mode
            assert tp_size == 8
            assert size == 16 * 5120
            return 0.3

    latency_ms, energy_wms, sources = VLLMBackend()._run_generation_phase(
        _Model(),
        _Database(),
        RuntimeConfig(),
        batch_size=16,
        beam_width=1,
        isl=1024,
        osl=2,
        stride=32,
    )

    assert latency_ms["generation_layerwise"] == pytest.approx(1.0 * 4)
    assert latency_ms["generation_tp_allreduce"] == pytest.approx((0.3 + 0.2) * 4)
    assert "generation_tp_allreduce_rms" not in latency_ms
    assert sum(latency_ms.values()) == pytest.approx(6.0)
    assert energy_wms["generation_layerwise"] == 0.0
    assert energy_wms["generation_tp_allreduce"] == 0.0
    assert sources == {
        "generation_layerwise": "silicon",
        "generation_tp_allreduce": "silicon",
    }


def test_vllm_layerwise_context_adds_moe_compute(monkeypatch) -> None:
    from aiconfigurator.sdk.backends import vllm_backend

    monkeypatch.setattr(vllm_backend, "_USE_LAYERWISE", True)

    class _Config:
        tp_size = 8
        pp_size = 1
        attention_dp_size = 1
        moe_tp_size = 1
        moe_ep_size = 8
        moe_quant_mode = common.MoEQuantMode.nvfp4
        workload_distribution = "power_law"
        moe_backend = None
        enable_eplb = False

    class _Model:
        model_path = "openai/gpt-oss-120b"
        config = _Config()
        _num_layers = 4
        _hidden_size = 2880
        _topk = 4
        _num_experts = 128
        _moe_inter_size = 2880
        _power_law_alpha = 1.2

    class _Database:
        def query_layerwise(self, model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache=0):
            assert (model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache) == (
                "openai/gpt-oss-120b",
                "CTX",
                8,
                1,
                128,
                0,
            )
            return 0.25

        def query_custom_allreduce(self, quant_mode, tp_size, size):
            del quant_mode
            assert (tp_size, size) == (8, 128 * 2880)
            return 0.0

        def query_nccl(self, dtype, num_gpus, operation, message_size):
            assert dtype == common.CommQuantMode.half
            assert (num_gpus, operation, message_size) == (8, "alltoall", 128 * 2880 * 4 // 8)
            return 0.1

        def query_moe(self, **kwargs):
            assert kwargs == {
                "num_tokens": 128,
                "hidden_size": 2880,
                "inter_size": 2880,
                "topk": 4,
                "num_experts": 128,
                "moe_tp_size": 1,
                "moe_ep_size": 8,
                "quant_mode": common.MoEQuantMode.nvfp4,
                "workload_distribution": "power_law_1.2",
                "is_context": True,
                "moe_backend": None,
                "is_gated": True,
                "enable_eplb": False,
            }
            return PerformanceResult(0.5, energy=2.0, source="empirical")

        def query_gemm(self, m, n, k, quant_mode, database_mode=None):
            assert (m, n, k, quant_mode, database_mode) == (
                128,
                128,
                2880,
                common.GEMMQuantMode.bfloat16,
                common.DatabaseMode.HYBRID,
            )
            return PerformanceResult(0.05, energy=0.2, source="empirical")

    latency, energy, sources = VLLMBackend()._run_context_phase(
        _Model(),
        _Database(),
        RuntimeConfig(),
        batch_size=1,
        isl=128,
        prefix=0,
    )

    assert latency["context_layerwise"] == pytest.approx(1.0)
    assert latency["context_moe"] == pytest.approx(2.0)
    assert latency["context_moe_router"] == pytest.approx(0.2)
    assert latency["context_moe_ep_alltoall"] == pytest.approx(0.4)
    assert energy["context_moe"] == pytest.approx(8.0)
    assert energy["context_moe_router"] == pytest.approx(0.8)
    assert sources["context_moe"] == "empirical"
    assert sources["context_moe_router"] == "empirical"
    assert sources["context_moe_ep_alltoall"] == "silicon"


def test_vllm_layerwise_qwen_noop_prefers_sampled_zipf_moe_when_available(monkeypatch) -> None:
    monkeypatch.setattr(
        VLLMBackend,
        "_layerwise_moe_distribution_available",
        lambda self, model, database, distribution, *, token_count: distribution == "sampled_zipf_0.8"
        and token_count <= 16,
    )

    class _Config:
        tp_size = 1
        pp_size = 1
        attention_dp_size = 1
        moe_tp_size = 1
        moe_ep_size = 1
        moe_quant_mode = common.MoEQuantMode.bfloat16
        workload_distribution = "power_law"
        moe_backend = None
        enable_eplb = False

    class _Model:
        model_path = "Qwen/Qwen3.6-35B-A3B"
        config = _Config()
        _hidden_size = 2048
        _topk = 8
        _num_experts = 256
        _moe_inter_size = 512
        _shared_expert_inter_size = 512
        _power_law_alpha = 1.2

    class _Database:
        def query_moe(self, **kwargs):
            assert kwargs["workload_distribution"] == "sampled_zipf_0.8"
            assert kwargs["num_tokens"] == 8
            assert kwargs["hidden_size"] == 2048
            assert kwargs["inter_size"] == 512
            assert kwargs["topk"] == 8
            assert kwargs["num_experts"] == 256
            assert kwargs["moe_tp_size"] == 1
            assert kwargs["moe_ep_size"] == 1
            return PerformanceResult(0.05, energy=0.2, source="empirical")

    latency, energy, source = VLLMBackend()._layerwise_moe_compute(
        _Model(),
        _Database(),
        token_count=8,
        num_layers=40,
        is_context=False,
    )

    assert latency == pytest.approx(2.0)
    assert energy == pytest.approx(8.0)
    assert source == "empirical"

    assert (
        VLLMBackend()._layerwise_moe_workload_distribution(
            _Model(),
            _Database(),
            token_count=32,
        )
        == "power_law"
    )


def test_vllm_layerwise_deepseek_noop_keeps_power_law_moe(monkeypatch) -> None:
    monkeypatch.setattr(
        VLLMBackend,
        "_layerwise_moe_distribution_available",
        lambda self, model, database, distribution, *, token_count: distribution == "sampled_zipf_0.8",
    )

    class _Config:
        tp_size = 1
        pp_size = 1
        attention_dp_size = 1
        moe_tp_size = 1
        moe_ep_size = 2
        moe_quant_mode = common.MoEQuantMode.w4a8_mxfp4_mxfp8
        workload_distribution = "power_law"
        moe_backend = None
        enable_eplb = False

    class _Model:
        model_path = "deepseek-ai/DeepSeek-V4-Flash"
        config = _Config()
        _hidden_size = 4096
        _topk = 6
        _num_experts = 256
        _moe_inter_size = 2048
        _shared_expert_inter_size = 2048
        _power_law_alpha = 1.2

    class _Database:
        def query_moe(self, **kwargs):
            assert kwargs["workload_distribution"] == "power_law_1.2"
            assert kwargs["num_tokens"] == 1
            return PerformanceResult(0.03, energy=0.2, source="silicon")

    latency, energy, source = VLLMBackend()._layerwise_moe_compute(
        _Model(),
        _Database(),
        token_count=1,
        num_layers=43,
        is_context=False,
    )

    assert latency == pytest.approx(1.29)
    assert energy == pytest.approx(8.6)
    assert source == "silicon"


def test_vllm_layerwise_shared_expert_addback_includes_gate_and_output_mul() -> None:
    class _Config:
        tp_size = 2
        gemm_quant_mode = common.GEMMQuantMode.bfloat16

    class _Model:
        config = _Config()
        _hidden_size = 2048
        _shared_expert_inter_size = 512

    class _Database:
        def __init__(self):
            self.gemm_shapes = []
            self.mem_sizes = []

        def query_gemm(self, m, n, k, quant_mode, database_mode=None):
            self.gemm_shapes.append((m, n, k, quant_mode, database_mode))
            return PerformanceResult(0.1, energy=0.2, source="silicon")

        def query_mem_op(self, size, database_mode=None):
            self.mem_sizes.append((size, database_mode))
            return PerformanceResult(0.01, energy=0.02, source="silicon")

    database = _Database()
    latency, energy, source = VLLMBackend()._layerwise_moe_shared_expert_compute(
        _Model(),
        database,
        token_count=128,
        num_layers=4,
    )

    assert database.gemm_shapes == [
        (128, 1, 2048, common.GEMMQuantMode.bfloat16, common.DatabaseMode.HYBRID),
        (128, 256, 2048, common.GEMMQuantMode.bfloat16, common.DatabaseMode.HYBRID),
        (128, 2048, 256, common.GEMMQuantMode.bfloat16, common.DatabaseMode.HYBRID),
    ]
    assert database.mem_sizes == [
        (128 * 256 * 4, common.DatabaseMode.HYBRID),
        (128 * 2048 * 4, common.DatabaseMode.HYBRID),
    ]
    assert latency == pytest.approx((0.1 * 3 + 0.01 * 2) * 4)
    assert energy == pytest.approx((0.2 * 3 + 0.02 * 2) * 4)
    assert source == "silicon"


def test_vllm_layerwise_deepseek_v4_noop_moe_addback_overlaps_shared_expert(monkeypatch) -> None:
    backend = VLLMBackend()

    class _Model:
        model_path = "deepseek-ai/DeepSeek-V4-Flash"

    monkeypatch.setattr(
        backend,
        "_layerwise_qwen_module_moe_compute",
        lambda *args, **kwargs: (0.0, 0.0, "silicon"),
    )
    monkeypatch.setattr(
        backend,
        "_layerwise_moe_compute",
        lambda *args, **kwargs: (1.25, 0.1, "silicon"),
    )
    monkeypatch.setattr(
        backend,
        "_layerwise_moe_router_compute",
        lambda *args, **kwargs: (0.75, 0.2, "silicon"),
    )
    monkeypatch.setattr(
        backend,
        "_layerwise_moe_shared_expert_compute",
        lambda *args, **kwargs: (1.5, 0.3, "silicon"),
    )

    moe, router, shared, bundled = backend._layerwise_noop_moe_addback(
        _Model(),
        database=object(),
        token_count=1,
        num_layers=43,
        is_context=False,
    )

    assert moe[0] == pytest.approx(2.0)
    assert moe[1] == pytest.approx(0.3)
    assert moe[2] == "mixed"
    assert router == (0.0, 0.0, "silicon")
    assert shared == (0.0, 0.0, "silicon")
    assert bundled is True


def test_vllm_layerwise_context_skips_moe_compute_when_layer_includes_moe(monkeypatch) -> None:
    from aiconfigurator.sdk.backends import vllm_backend

    monkeypatch.setattr(vllm_backend, "_USE_LAYERWISE", True)

    class _Config:
        tp_size = 1
        pp_size = 1

    class _Model:
        model_path = "openai/gpt-oss-120b"
        config = _Config()
        _num_layers = 4
        _hidden_size = 2880
        _topk = 4
        _num_experts = 128
        _moe_inter_size = 2880

    class _Database:
        def query_layerwise_detail(self, model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache=0):
            assert (model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache) == (
                "openai/gpt-oss-120b",
                "CTX",
                1,
                1,
                128,
                0,
            )
            return {"latency": 0.25, "energy": 0.0, "rms_latency": 0.0, "includes_moe": True}

        def query_moe(self, **kwargs):
            raise AssertionError(f"MoE should not be queried for full-MoE layer rows: {kwargs}")

    latency, energy, sources = VLLMBackend()._run_context_phase(
        _Model(),
        _Database(),
        RuntimeConfig(),
        batch_size=1,
        isl=128,
        prefix=0,
    )

    assert latency == {"context_layerwise": pytest.approx(1.0), "context_tp_allreduce": 0.0}
    assert energy == {"context_layerwise": 0.0, "context_tp_allreduce": 0.0}
    assert sources == {"context_layerwise": "silicon", "context_tp_allreduce": "silicon"}


def test_vllm_layerwise_context_envelope_rows_add_explicit_allreduce(monkeypatch) -> None:
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
        def query_layerwise_detail(self, model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache=0):
            assert (model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache) == (
                "Qwen/Qwen3-32B",
                "CTX",
                2,
                1,
                128,
                0,
            )
            return {
                "latency": 17.0,
                "energy": 0.0,
                "rms_latency": 0.0,
                "measured_layer_count": 4,
                "layer_multiplier": 4,
                "latency_source": "schedule_to_update",
                "physical_gpus": 1,
            }

        def query_custom_allreduce(self, quant_mode, tp_size, size, execution_mode=None):
            del quant_mode
            assert tp_size == 2
            assert size == 128 * 5120
            assert execution_mode == "eager"
            return 0.25

    latency, energy, sources = VLLMBackend()._run_context_phase(
        _Model(),
        _Database(),
        RuntimeConfig(),
        batch_size=1,
        isl=128,
        prefix=0,
    )

    assert latency == {"context_layerwise": pytest.approx(17.0), "context_tp_allreduce": pytest.approx(2.0)}
    assert energy == {"context_layerwise": 0.0, "context_tp_allreduce": 0.0}
    assert sources == {"context_layerwise": "silicon", "context_tp_allreduce": "silicon"}


def test_vllm_layerwise_context_adds_moe_tp_allreduce_for_moe_envelope_rows(monkeypatch) -> None:
    from aiconfigurator.sdk.backends import vllm_backend

    monkeypatch.setattr(vllm_backend, "_USE_LAYERWISE", True)

    class _Config:
        tp_size = 2
        pp_size = 1
        moe_tp_size = 2

    class _Model:
        model_path = "Qwen/Qwen3.6-35B-A3B"
        config = _Config()
        _num_layers = 4
        _hidden_size = 2560

    class _Database:
        def query_layerwise_detail(self, model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache=0):
            assert (model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache) == (
                "Qwen/Qwen3.6-35B-A3B",
                "CTX",
                2,
                1,
                128,
                0,
            )
            return {
                "latency": 17.0,
                "energy": 0.0,
                "rms_latency": 0.0,
                "includes_moe": True,
                "measured_layer_count": 4,
                "layer_multiplier": 4,
                "latency_source": "worker_wall",
            }

        def query_custom_allreduce(self, quant_mode, tp_size, size, execution_mode=None):
            del quant_mode
            assert tp_size == 2
            assert size == 128 * 2560
            assert execution_mode == "eager"
            return 0.25

    latency, energy, sources = VLLMBackend()._run_context_phase(
        _Model(),
        _Database(),
        RuntimeConfig(),
        batch_size=1,
        isl=128,
        prefix=0,
    )

    assert latency["context_layerwise"] == pytest.approx(17.0)
    assert latency["context_tp_allreduce"] == pytest.approx(2.0)
    assert latency["context_moe_tp_allreduce"] == pytest.approx(1.0)
    assert energy["context_moe_tp_allreduce"] == 0.0
    assert sources["context_moe_tp_allreduce"] == "silicon"


def test_vllm_layerwise_deepseek_scheduler_moe_context_skips_dense_tp_allreduce(monkeypatch) -> None:
    from aiconfigurator.sdk.backends import vllm_backend

    monkeypatch.setattr(vllm_backend, "_USE_LAYERWISE", True)

    class _Config:
        tp_size = 4
        pp_size = 1
        moe_tp_size = 1
        moe_ep_size = 4

    class _Model:
        model_path = "deepseek-ai/DeepSeek-V4-Flash"
        config = _Config()
        _num_layers = 4
        _hidden_size = 2048
        _topk = 8

    class _Database:
        def query_layerwise_detail(self, model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache=0):
            assert (model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache) == (
                "deepseek-ai/DeepSeek-V4-Flash",
                "CTX",
                4,
                1,
                128,
                0,
            )
            return {
                "latency": 17.0,
                "energy": 0.0,
                "rms_latency": 0.0,
                "includes_moe": True,
                "measured_layer_count": 4,
                "layer_multiplier": 4,
                "latency_source": "schedule_to_update",
                "physical_gpus": 1,
            }

        def query_custom_allreduce(self, *args, **kwargs):
            raise AssertionError((args, kwargs))

        def query_nccl(self, dtype, num_gpus, operation, message_size):
            assert dtype == common.CommQuantMode.half
            assert (num_gpus, operation, message_size) == (4, "alltoall", 128 * 2048 * 8 // 4)
            return 0.1

    latency, energy, sources = VLLMBackend()._run_context_phase(
        _Model(),
        _Database(),
        RuntimeConfig(),
        batch_size=1,
        isl=128,
        prefix=0,
    )

    assert latency["context_layerwise"] == pytest.approx(17.0)
    assert latency["context_tp_allreduce"] == 0.0
    assert latency["context_moe_ep_alltoall"] == pytest.approx(0.4)
    assert energy["context_moe_ep_alltoall"] == 0.0
    assert sources["context_moe_ep_alltoall"] == "silicon"


def test_vllm_layerwise_context_adds_moe_tp_allreduce_for_noop_rows(monkeypatch) -> None:
    from aiconfigurator.sdk.backends import vllm_backend

    monkeypatch.setattr(vllm_backend, "_USE_LAYERWISE", True)

    class _Config:
        tp_size = 2
        pp_size = 1
        attention_dp_size = 1
        moe_tp_size = 2
        moe_ep_size = 1
        moe_quant_mode = common.MoEQuantMode.bfloat16
        workload_distribution = "power_law"
        moe_backend = None
        enable_eplb = False
        gemm_quant_mode = common.GEMMQuantMode.bfloat16

    class _Model:
        model_path = "Qwen/Qwen3.6-35B-A3B"
        config = _Config()
        _num_layers = 4
        _hidden_size = 2048
        _topk = 8
        _num_experts = 256
        _moe_inter_size = 512
        _power_law_alpha = 1.2

    class _Database:
        def query_layerwise_detail(self, model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache=0):
            assert (model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache) == (
                "Qwen/Qwen3.6-35B-A3B",
                "CTX",
                2,
                1,
                128,
                0,
            )
            return {
                "latency": 10.0,
                "energy": 0.0,
                "rms_latency": 0.0,
                "includes_moe": False,
                "measured_layer_count": 4,
                "layer_multiplier": 4,
                "latency_source": "schedule_to_update",
                "physical_gpus": 1,
            }

        def query_custom_allreduce(self, quant_mode, tp_size, size, execution_mode=None):
            del quant_mode
            assert tp_size == 2
            assert size == 128 * 2048
            assert execution_mode == "eager"
            return 0.25

        def query_moe(self, **kwargs):
            assert kwargs["moe_tp_size"] == 2
            return PerformanceResult(0.5, energy=1.0, source="silicon")

        def query_gemm(self, *args, **kwargs):
            del args, kwargs
            return PerformanceResult(0.0, energy=0.0, source="silicon")

    latency, energy, sources = VLLMBackend()._run_context_phase(
        _Model(),
        _Database(),
        RuntimeConfig(),
        batch_size=1,
        isl=128,
        prefix=0,
    )

    assert latency["context_layerwise"] == pytest.approx(10.0)
    assert latency["context_tp_allreduce"] == pytest.approx(2.0)
    assert latency["context_moe"] == pytest.approx(2.0)
    assert latency["context_moe_tp_allreduce"] == pytest.approx(1.0)
    assert energy["context_moe"] == pytest.approx(4.0)
    assert sources["context_moe_tp_allreduce"] == "silicon"


def test_vllm_layerwise_context_prefers_qwen_module_moe_addback(monkeypatch) -> None:
    from aiconfigurator.sdk.backends import vllm_backend

    monkeypatch.setattr(vllm_backend, "_USE_LAYERWISE", True)

    def _module_addback(self, model, database, token_count, num_layers, is_context):
        del self, database
        assert model.model_path == "Qwen/Qwen3.6-35B-A3B"
        assert token_count == 128
        assert num_layers == 4
        assert is_context is True
        return 3.0, 0.6, "silicon"

    monkeypatch.setattr(VLLMBackend, "_layerwise_qwen_module_moe_compute", _module_addback)

    class _Config:
        tp_size = 2
        pp_size = 1
        attention_dp_size = 1
        moe_tp_size = 2
        moe_ep_size = 1
        moe_quant_mode = common.MoEQuantMode.bfloat16
        workload_distribution = "power_law"
        moe_backend = None
        enable_eplb = False
        gemm_quant_mode = common.GEMMQuantMode.bfloat16

    class _Model:
        model_path = "Qwen/Qwen3.6-35B-A3B"
        config = _Config()
        _num_layers = 4
        _hidden_size = 2048
        _topk = 8
        _num_experts = 256
        _moe_inter_size = 512
        _shared_expert_inter_size = 512
        _power_law_alpha = 1.2

    class _Database:
        def query_layerwise_detail(self, model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache=0):
            assert (model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache) == (
                "Qwen/Qwen3.6-35B-A3B",
                "CTX",
                2,
                1,
                128,
                0,
            )
            return {
                "latency": 10.0,
                "energy": 0.0,
                "rms_latency": 0.0,
                "includes_moe": False,
                "measured_layer_count": 4,
                "layer_multiplier": 4,
                "latency_source": "schedule_to_update",
                "physical_gpus": 1,
            }

        def query_custom_allreduce(self, quant_mode, tp_size, size, execution_mode=None):
            del quant_mode
            assert tp_size == 2
            assert size == 128 * 2048
            assert execution_mode == "eager"
            return 0.25

        def query_moe(self, **kwargs):
            raise AssertionError(f"bundled Qwen module MoE row should replace routed MoE query: {kwargs}")

        def query_gemm(self, *args, **kwargs):
            raise AssertionError("bundled Qwen module MoE row should replace router/shared GEMMs")

    latency, energy, sources = VLLMBackend()._run_context_phase(
        _Model(),
        _Database(),
        RuntimeConfig(),
        batch_size=1,
        isl=128,
        prefix=0,
    )

    assert latency["context_layerwise"] == pytest.approx(10.0)
    assert latency["context_tp_allreduce"] == pytest.approx(2.0)
    assert latency["context_moe"] == pytest.approx(3.0)
    assert "context_moe_tp_allreduce" not in latency
    assert "context_moe_router" not in latency
    assert "context_moe_shared_expert" not in latency
    assert energy["context_moe"] == pytest.approx(0.6)
    assert sources["context_moe"] == "silicon"


def test_vllm_layerwise_context_adds_moe_ep_alltoall_for_single_gpu_moe_rows(monkeypatch) -> None:
    from aiconfigurator.sdk.backends import vllm_backend

    monkeypatch.setattr(vllm_backend, "_USE_LAYERWISE", True)

    class _Config:
        tp_size = 4
        pp_size = 1
        moe_tp_size = 1
        moe_ep_size = 4

    class _Model:
        model_path = "Qwen/Qwen3.6-35B-A3B"
        config = _Config()
        _num_layers = 4
        _hidden_size = 2048
        _topk = 8

    class _Database:
        def query_layerwise_detail(self, model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache=0):
            assert (model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache) == (
                "Qwen/Qwen3.6-35B-A3B",
                "CTX",
                4,
                1,
                128,
                0,
            )
            return {
                "latency": 17.0,
                "energy": 0.0,
                "rms_latency": 0.0,
                "includes_moe": True,
                "measured_layer_count": 4,
                "layer_multiplier": 4,
                "latency_source": "schedule_to_update",
                "physical_gpus": 1,
            }

        def query_custom_allreduce(self, quant_mode, tp_size, size, execution_mode=None):
            del quant_mode
            assert tp_size == 4
            assert size == 128 * 2048
            assert execution_mode == "eager"
            return 0.25

        def query_nccl(self, dtype, num_gpus, operation, message_size):
            assert dtype == common.CommQuantMode.half
            assert (num_gpus, operation, message_size) == (4, "alltoall", 128 * 2048 * 8 // 4)
            return 0.1

    latency, energy, sources = VLLMBackend()._run_context_phase(
        _Model(),
        _Database(),
        RuntimeConfig(),
        batch_size=1,
        isl=128,
        prefix=0,
    )

    assert latency["context_layerwise"] == pytest.approx(17.0)
    assert latency["context_tp_allreduce"] == pytest.approx(2.0)
    assert latency["context_moe_ep_alltoall"] == pytest.approx(0.4)
    assert energy["context_moe_ep_alltoall"] == 0.0
    assert sources["context_moe_ep_alltoall"] == "silicon"


def test_vllm_layerwise_context_chunks_noop_moe_addback(monkeypatch) -> None:
    from aiconfigurator.sdk.backends import vllm_backend

    monkeypatch.setattr(vllm_backend, "_USE_LAYERWISE", True)

    class _Config:
        tp_size = 1
        pp_size = 1
        attention_dp_size = 1
        moe_tp_size = 1
        moe_ep_size = 1
        gemm_quant_mode = common.GEMMQuantMode.bfloat16
        moe_quant_mode = common.MoEQuantMode.w4a8_mxfp4_mxfp8
        workload_distribution = "power_law"
        moe_backend = None
        enable_eplb = False

    class _Model:
        model_path = "deepseek-ai/DeepSeek-V4-Flash"
        config = _Config()
        _num_layers = 2
        _hidden_size = 4096
        _topk = 6
        _num_experts = 256
        _moe_inter_size = 2048
        _nextn = 0

    class _Database:
        def __init__(self):
            self.layerwise_queries = []
            self.moe_token_counts = []
            self.router_token_counts = []

        def query_layerwise_detail(self, model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache=0):
            query = (model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache)
            self.layerwise_queries.append(query)
            assert query in {
                ("deepseek-ai/DeepSeek-V4-Flash", "CTX", 1, 1, 2048, 0),
                ("deepseek-ai/DeepSeek-V4-Flash", "CTX", 1, 1, 2048, 2048),
            }
            return {
                "latency": 3.0,
                "energy": 0.0,
                "rms_latency": 0.0,
                "includes_moe": False,
                "measured_layer_count": 2,
                "layer_multiplier": 2,
                "latency_source": "schedule_to_update",
                "physical_gpus": 1,
            }

        def query_moe(self, **kwargs):
            self.moe_token_counts.append(kwargs["num_tokens"])
            assert kwargs["is_context"] is True
            assert kwargs["workload_distribution"] == "power_law_1.2"
            return PerformanceResult(0.5, energy=2.0, source="empirical")

        def query_gemm(self, m, n, k, quant_mode, database_mode=None):
            self.router_token_counts.append(m)
            assert (m, n, k, quant_mode, database_mode) == (
                2048,
                256,
                4096,
                common.GEMMQuantMode.bfloat16,
                common.DatabaseMode.HYBRID,
            )
            return PerformanceResult(0.1, energy=0.4, source="empirical")

    database = _Database()
    latency, energy, sources = VLLMBackend()._run_context_phase(
        _Model(),
        database,
        RuntimeConfig(vllm_max_num_batched_tokens=2048),
        batch_size=1,
        isl=4096,
        prefix=0,
    )

    assert database.layerwise_queries == [
        ("deepseek-ai/DeepSeek-V4-Flash", "CTX", 1, 1, 2048, 0),
        ("deepseek-ai/DeepSeek-V4-Flash", "CTX", 1, 1, 2048, 2048),
    ]
    assert database.moe_token_counts == [2048, 2048]
    assert database.router_token_counts == [2048, 2048]
    assert latency["context_layerwise"] == pytest.approx(6.0)
    assert latency["context_moe"] == pytest.approx(2.4)
    assert energy["context_moe"] == pytest.approx(9.6)
    assert sources["context_moe"] == "mixed"


def test_vllm_deepseek_context_only_sums_shared_noop_addback(monkeypatch) -> None:
    class _Config:
        tp_size = 1
        pp_size = 1

    class _Model:
        model_path = "deepseek-ai/DeepSeek-V4-Flash"
        config = _Config()

    backend = VLLMBackend()
    monkeypatch.setattr(
        backend,
        "_layerwise_qwen_module_moe_compute",
        lambda *args, **kwargs: (0.0, 0.0, "silicon"),
    )
    monkeypatch.setattr(
        backend,
        "_layerwise_moe_compute",
        lambda *args, **kwargs: (10.0, 1.0, "moe"),
    )
    monkeypatch.setattr(
        backend,
        "_layerwise_moe_router_compute",
        lambda *args, **kwargs: (2.0, 0.2, "router"),
    )
    monkeypatch.setattr(
        backend,
        "_layerwise_moe_shared_expert_compute",
        lambda *args, **kwargs: (3.0, 0.3, "shared"),
    )

    standalone_context = backend._layerwise_noop_moe_addback(
        _Model(),
        object(),
        token_count=128,
        num_layers=1,
        is_context=True,
        deepseek_context_sum_shared=True,
    )
    mixed_context = backend._layerwise_noop_moe_addback(
        _Model(),
        object(),
        token_count=128,
        num_layers=1,
        is_context=True,
    )
    decode = backend._layerwise_noop_moe_addback(
        _Model(),
        object(),
        token_count=1,
        num_layers=1,
        is_context=False,
        deepseek_context_sum_shared=True,
    )

    assert standalone_context[0][0] == pytest.approx(15.0)
    assert standalone_context[0][1] == pytest.approx(1.5)
    assert standalone_context[0][2] == "mixed"
    assert mixed_context[0][0] == pytest.approx(12.0)
    assert mixed_context[0][1] == pytest.approx(1.2)
    assert mixed_context[0][2] == "mixed"
    assert decode[0][0] == pytest.approx(12.0)
    assert decode[0][1] == pytest.approx(1.2)
    assert decode[0][2] == "mixed"


def test_vllm_layerwise_context_uses_structural_moe_components(monkeypatch) -> None:
    from aiconfigurator.sdk.backends import vllm_backend

    monkeypatch.setattr(vllm_backend, "_USE_LAYERWISE", True)

    class _Config:
        tp_size = 1
        pp_size = 1

    class _Model:
        model_path = "Qwen/Qwen3.6-35B-A3B"
        config = _Config()
        _num_layers = 4
        _hidden_size = 2880
        _topk = 8
        _num_experts = 128
        _moe_inter_size = 768

    class _Database:
        def query_layerwise_detail(self, model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache=0):
            assert (model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache) == (
                "Qwen/Qwen3.6-35B-A3B",
                "CTX",
                1,
                1,
                128,
                0,
            )
            return {
                "latency": 22.0,
                "energy": 0.0,
                "rms_latency": 0.0,
                "includes_moe": True,
                "components": [
                    {
                        "latency": 10.0,
                        "measured_layer_count": 4,
                        "layer_multiplier": 4,
                        "includes_moe": False,
                    },
                    {
                        "latency": 1.0,
                        "measured_layer_count": 1,
                        "layer_multiplier": 4,
                        "includes_moe": False,
                    },
                    {
                        "latency": 2.0,
                        "measured_layer_count": 1,
                        "layer_multiplier": 4,
                        "includes_moe": True,
                    },
                ],
            }

        def query_moe(self, **kwargs):
            raise AssertionError(f"MoE op-level fallback should not be used for structural components: {kwargs}")

    latency, energy, sources = VLLMBackend()._run_context_phase(
        _Model(),
        _Database(),
        RuntimeConfig(),
        batch_size=1,
        isl=128,
        prefix=0,
    )

    assert latency == {"context_layerwise": pytest.approx(14.0), "context_tp_allreduce": 0.0}
    assert energy == {"context_layerwise": 0.0, "context_tp_allreduce": 0.0}
    assert sources == {"context_layerwise": "silicon", "context_tp_allreduce": "silicon"}


def test_vllm_layerwise_structural_moe_components_choose_one_latency_source(monkeypatch) -> None:
    from aiconfigurator.sdk.backends import vllm_backend

    monkeypatch.setattr(vllm_backend, "_USE_LAYERWISE", True)

    class _Config:
        tp_size = 1
        pp_size = 1

    class _Model:
        model_path = "Qwen/Qwen3.6-35B-A3B"
        config = _Config()
        _num_layers = 4
        _hidden_size = 2880
        _topk = 8
        _num_experts = 128
        _moe_inter_size = 768

    class _Database:
        def query_layerwise_detail(self, model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache=0):
            del model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache
            return {
                "latency": 999.0,
                "energy": 0.0,
                "rms_latency": 0.0,
                "includes_moe": True,
                "components": [
                    {
                        "latency": 10.0,
                        "measured_layer_count": 4,
                        "layer_multiplier": 4,
                        "includes_moe": False,
                        "latency_source": "schedule_to_update",
                    },
                    {
                        "latency": 1.0,
                        "measured_layer_count": 1,
                        "layer_multiplier": 4,
                        "includes_moe": False,
                        "latency_source": "span",
                    },
                    {
                        "latency": 4.0,
                        "measured_layer_count": 1,
                        "layer_multiplier": 4,
                        "includes_moe": True,
                        "latency_source": "span",
                    },
                    {
                        "latency": 1.0,
                        "measured_layer_count": 1,
                        "layer_multiplier": 4,
                        "includes_moe": False,
                        "latency_source": "gpu",
                    },
                    {
                        "latency": 2.0,
                        "measured_layer_count": 1,
                        "layer_multiplier": 4,
                        "includes_moe": True,
                        "latency_source": "gpu",
                    },
                ],
            }

        def query_moe(self, **kwargs):
            raise AssertionError(f"MoE op-level fallback should not be used for structural components: {kwargs}")

    latency, energy, sources = VLLMBackend()._run_context_phase(
        _Model(),
        _Database(),
        RuntimeConfig(),
        batch_size=1,
        isl=128,
        prefix=0,
    )

    assert latency == {"context_layerwise": pytest.approx(14.0), "context_tp_allreduce": 0.0}
    assert energy == {"context_layerwise": 0.0, "context_tp_allreduce": 0.0}
    assert sources == {"context_layerwise": "silicon", "context_tp_allreduce": "silicon"}


def test_vllm_layerwise_structural_moe_components_infer_legacy_delta_source(monkeypatch) -> None:
    from aiconfigurator.sdk.backends import vllm_backend

    monkeypatch.setattr(vllm_backend, "_USE_LAYERWISE", True)

    class _Config:
        tp_size = 1
        pp_size = 1

    class _Model:
        model_path = "Qwen/Qwen3.6-35B-A3B"
        config = _Config()
        _num_layers = 4
        _hidden_size = 2880
        _topk = 8
        _num_experts = 128
        _moe_inter_size = 768

    class _Database:
        def query_layerwise_detail(self, model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache=0):
            del model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache
            return {
                "latency": 999.0,
                "energy": 0.0,
                "rms_latency": 0.0,
                "includes_moe": True,
                "components": [
                    {
                        "latency": 10.0,
                        "measured_layer_count": 4,
                        "layer_multiplier": 4,
                        "includes_moe": False,
                        "latency_source": "schedule_to_update",
                    },
                    {
                        "latency": 1.0,
                        "measured_layer_count": 1,
                        "layer_multiplier": 4,
                        "includes_moe": False,
                        "latency_source": "span",
                    },
                    {
                        "latency": 4.0,
                        "measured_layer_count": 1,
                        "layer_multiplier": 4,
                        "includes_moe": True,
                    },
                ],
            }

        def query_moe(self, **kwargs):
            raise AssertionError(f"MoE op-level fallback should not be used for structural components: {kwargs}")

    latency, energy, sources = VLLMBackend()._run_context_phase(
        _Model(),
        _Database(),
        RuntimeConfig(),
        batch_size=1,
        isl=128,
        prefix=0,
    )

    assert latency == {"context_layerwise": pytest.approx(22.0), "context_tp_allreduce": 0.0}
    assert energy == {"context_layerwise": 0.0, "context_tp_allreduce": 0.0}
    assert sources == {"context_layerwise": "silicon", "context_tp_allreduce": "silicon"}


def test_vllm_layerwise_context_scales_representative_metadata(monkeypatch) -> None:
    from aiconfigurator.sdk.backends import vllm_backend

    monkeypatch.setattr(vllm_backend, "_USE_LAYERWISE", True)

    class _Config:
        tp_size = 1
        pp_size = 1

    class _Model:
        model_path = "Qwen/Qwen3-32B"
        config = _Config()
        _num_layers = 64
        _hidden_size = 5120

    class _Database:
        def query_layerwise_detail(self, model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache=0):
            assert (model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache) == (
                "Qwen/Qwen3-32B",
                "CTX",
                1,
                1,
                128,
                0,
            )
            return {
                "latency": 4.0,
                "energy": 0.0,
                "rms_latency": 0.0,
                "layer_type": "dense",
                "measured_layer_count": 16,
                "layer_multiplier": 64,
            }

    latency, energy, sources = VLLMBackend()._run_context_phase(
        _Model(),
        _Database(),
        RuntimeConfig(),
        batch_size=1,
        isl=128,
        prefix=0,
    )

    assert latency == {"context_layerwise": pytest.approx(16.0), "context_tp_allreduce": 0.0}
    assert energy == {"context_layerwise": 0.0, "context_tp_allreduce": 0.0}
    assert sources == {"context_layerwise": "silicon", "context_tp_allreduce": "silicon"}


def test_vllm_layerwise_generation_adds_moe_compute(monkeypatch) -> None:
    from aiconfigurator.sdk.backends import vllm_backend

    monkeypatch.setattr(vllm_backend, "_USE_LAYERWISE", True)

    class _Config:
        tp_size = 8
        pp_size = 1
        attention_dp_size = 1
        moe_tp_size = 1
        moe_ep_size = 8
        moe_quant_mode = common.MoEQuantMode.nvfp4
        workload_distribution = "balanced"
        moe_backend = None
        enable_eplb = False

    class _Model:
        model_path = "openai/gpt-oss-120b"
        config = _Config()
        _num_layers = 4
        _hidden_size = 2880
        _topk = 4
        _num_experts = 128
        _moe_inter_size = 2880
        _nextn = 0

    class _Database:
        def query_layerwise_detail(self, model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache=0):
            assert (model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache) == (
                "openai/gpt-oss-120b",
                "GEN",
                8,
                16,
                1024,
                0,
            )
            return {"latency": 1.0, "energy": 0.0, "rms_latency": 0.0}

        def query_nccl(self, dtype, num_gpus, operation, message_size):
            assert dtype == common.CommQuantMode.half
            assert (num_gpus, operation, message_size) == (8, "alltoall", 16 * 2880 * 4 // 8)
            return 0.1

        def query_moe(self, **kwargs):
            assert kwargs == {
                "num_tokens": 16,
                "hidden_size": 2880,
                "inter_size": 2880,
                "topk": 4,
                "num_experts": 128,
                "moe_tp_size": 1,
                "moe_ep_size": 8,
                "quant_mode": common.MoEQuantMode.nvfp4,
                "workload_distribution": "balanced",
                "is_context": False,
                "moe_backend": None,
                "is_gated": True,
                "enable_eplb": False,
            }
            return PerformanceResult(0.5, energy=2.0, source="empirical")

        def query_gemm(self, m, n, k, quant_mode, database_mode=None):
            assert (m, n, k, quant_mode, database_mode) == (
                16,
                128,
                2880,
                common.GEMMQuantMode.bfloat16,
                common.DatabaseMode.HYBRID,
            )
            return PerformanceResult(0.05, energy=0.2, source="empirical")

    latency, energy, sources = VLLMBackend()._run_generation_phase(
        _Model(),
        _Database(),
        RuntimeConfig(),
        batch_size=16,
        beam_width=1,
        isl=1024,
        osl=3,
        stride=32,
    )

    assert latency["generation_layerwise"] == pytest.approx(8.0)
    assert latency["generation_moe"] == pytest.approx(4.0)
    assert latency["generation_moe_router"] == pytest.approx(0.4)
    assert latency["generation_moe_ep_alltoall"] == pytest.approx(1.6)
    assert energy["generation_moe"] == pytest.approx(16.0)
    assert energy["generation_moe_router"] == pytest.approx(1.6)
    assert sources["generation_moe"] == "empirical"
    assert sources["generation_moe_router"] == "empirical"
    assert sources["generation_moe_ep_alltoall"] == "silicon"


def test_vllm_layerwise_generation_skips_explicit_ep_alltoall_for_bundled_deepseek_moe(monkeypatch) -> None:
    from aiconfigurator.sdk.backends import vllm_backend

    monkeypatch.setattr(vllm_backend, "_USE_LAYERWISE", True)

    class _Config:
        tp_size = 1
        pp_size = 1
        attention_dp_size = 1
        moe_tp_size = 1
        moe_ep_size = 2
        gemm_quant_mode = common.GEMMQuantMode.bfloat16
        moe_quant_mode = common.MoEQuantMode.w4a8_mxfp4_mxfp8
        workload_distribution = "power_law"
        moe_backend = None
        enable_eplb = False

    class _Model:
        model_path = "deepseek-ai/DeepSeek-V4-Flash"
        config = _Config()
        _num_layers = 2
        _hidden_size = 4096
        _topk = 6
        _num_experts = 256
        _moe_inter_size = 2048
        _nextn = 0

    class _Database:
        def query_layerwise_detail(self, model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache=0):
            assert (model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache) == (
                "deepseek-ai/DeepSeek-V4-Flash",
                "GEN",
                1,
                1,
                4096,
                0,
            )
            return {
                "latency": 1.0,
                "energy": 0.0,
                "rms_latency": 0.0,
                "includes_moe": False,
                "measured_layer_count": 1,
                "layer_multiplier": 2,
            }

        def query_moe(self, **kwargs):
            assert kwargs == {
                "num_tokens": 1,
                "hidden_size": 4096,
                "inter_size": 2048,
                "topk": 6,
                "num_experts": 256,
                "moe_tp_size": 1,
                "moe_ep_size": 2,
                "quant_mode": common.MoEQuantMode.w4a8_mxfp4_mxfp8,
                "workload_distribution": "power_law_1.2",
                "is_context": False,
                "moe_backend": None,
                "is_gated": True,
                "enable_eplb": False,
            }
            return PerformanceResult(0.5, energy=2.0, source="empirical")

        def query_gemm(self, m, n, k, quant_mode, database_mode=None):
            assert (m, n, k, quant_mode, database_mode) == (
                1,
                256,
                4096,
                common.GEMMQuantMode.bfloat16,
                common.DatabaseMode.HYBRID,
            )
            return PerformanceResult(0.05, energy=0.2, source="empirical")

        def query_nccl(self, *args, **kwargs):
            raise AssertionError("bundled DeepSeek-V4 no-op MoE addback should not add explicit EP all-to-all")

    latency, energy, sources = VLLMBackend()._run_generation_phase(
        _Model(),
        _Database(),
        RuntimeConfig(),
        batch_size=1,
        beam_width=1,
        isl=4096,
        osl=2,
        stride=32,
    )

    assert latency["generation_layerwise"] == pytest.approx(2.0)
    assert latency["generation_moe"] == pytest.approx((0.5 + 0.05) * 2)
    assert "generation_moe_ep_alltoall" not in latency
    assert energy["generation_moe"] == pytest.approx((2.0 + 0.2) * 2)
    assert sources["generation_moe"] == "mixed"


def test_vllm_layerwise_generation_skips_moe_compute_when_layer_includes_moe(monkeypatch) -> None:
    from aiconfigurator.sdk.backends import vllm_backend

    monkeypatch.setattr(vllm_backend, "_USE_LAYERWISE", True)

    class _Config:
        tp_size = 1
        pp_size = 1

    class _Model:
        model_path = "openai/gpt-oss-120b"
        config = _Config()
        _num_layers = 4
        _hidden_size = 2880
        _topk = 4
        _num_experts = 128
        _moe_inter_size = 2880
        _nextn = 0

    class _Database:
        def query_layerwise_detail(self, model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache=0):
            assert (model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache) == (
                "openai/gpt-oss-120b",
                "GEN",
                1,
                16,
                1024,
                0,
            )
            return {"latency": 1.0, "energy": 0.0, "rms_latency": 0.0, "includes_moe": True}

        def query_moe(self, **kwargs):
            raise AssertionError(f"MoE should not be queried for full-MoE layer rows: {kwargs}")

    latency, energy, sources = VLLMBackend()._run_generation_phase(
        _Model(),
        _Database(),
        RuntimeConfig(),
        batch_size=16,
        beam_width=1,
        isl=1024,
        osl=3,
        stride=32,
    )

    assert latency == {"generation_layerwise": pytest.approx(8.0)}
    assert energy == {"generation_layerwise": 0.0}
    assert sources == {"generation_layerwise": "silicon"}


def test_vllm_layerwise_generation_does_not_add_moe_tp_allreduce_for_full_moe_rows(monkeypatch) -> None:
    from aiconfigurator.sdk.backends import vllm_backend

    monkeypatch.setattr(vllm_backend, "_USE_LAYERWISE", True)

    class _Config:
        tp_size = 2
        pp_size = 1
        moe_tp_size = 2

    class _Model:
        model_path = "Qwen/Qwen3.6-35B-A3B"
        config = _Config()
        _num_layers = 4
        _hidden_size = 2880
        _nextn = 0

    class _Database:
        def query_layerwise_detail(self, model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache=0):
            del seq_len_kv_cache
            assert (model, phase, tp_size, batch_size, seq_len) == (
                "Qwen/Qwen3.6-35B-A3B",
                "GEN",
                2,
                8,
                4096,
            )
            return {
                "latency": 1.0,
                "energy": 0.0,
                "rms_latency": 0.0,
                "includes_moe": True,
                "measured_layer_count": 4,
                "layer_multiplier": 4,
            }

        def query_custom_allreduce(self, quant_mode, tp_size, size):
            del quant_mode
            assert tp_size == 2
            assert size == 8 * 2880
            return 0.25

        def query_moe(self, **kwargs):
            raise AssertionError(f"MoE should not be queried for full-MoE layer rows: {kwargs}")

    latency, energy, sources = VLLMBackend()._run_generation_phase(
        _Model(),
        _Database(),
        RuntimeConfig(),
        batch_size=8,
        beam_width=1,
        isl=4096,
        osl=3,
        stride=32,
    )

    assert latency["generation_layerwise"] == pytest.approx(2.0)
    assert latency["generation_tp_allreduce"] == pytest.approx(0.25 * 2 * 4 * 2)
    assert "generation_moe_tp_allreduce" not in latency
    assert "generation_moe_tp_allreduce" not in energy
    assert "generation_moe_tp_allreduce" not in sources


def test_vllm_layerwise_generation_uses_one_tp_collective_for_pure_ep_moe_rows(monkeypatch) -> None:
    from aiconfigurator.sdk.backends import vllm_backend

    monkeypatch.setattr(vllm_backend, "_USE_LAYERWISE", True)

    class _Config:
        tp_size = 4
        pp_size = 1
        moe_tp_size = 1
        moe_ep_size = 4

    class _Model:
        model_path = "Qwen/Qwen3.6-35B-A3B"
        config = _Config()
        _num_layers = 40
        _hidden_size = 2048
        _topk = 8
        _nextn = 0

    class _Database:
        def query_layerwise_detail(self, model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache=0):
            del seq_len_kv_cache
            assert (model, phase, tp_size, batch_size, seq_len) == (
                "Qwen/Qwen3.6-35B-A3B",
                "GEN",
                4,
                1,
                1024,
            )
            return {
                "latency": 3.25,
                "energy": 0.0,
                "rms_latency": 0.0,
                "includes_moe": True,
                "measured_layer_count": 40,
                "layer_multiplier": 40,
                "physical_gpus": 1,
            }

        def query_custom_allreduce(self, quant_mode, tp_size, size):
            del quant_mode
            assert tp_size == 4
            assert size == 2048
            return 0.25

        def query_allreduce_rms(self, quant_mode, tp_size, size, hidden_size):
            del quant_mode
            assert tp_size == 4
            assert size == 2048
            assert hidden_size == 2048
            return 0.125

        def query_nccl(self, *args, **kwargs):
            raise AssertionError("pure-EP layerwise decode rows should not add exposed EP all-to-all")

    latency, energy, sources = VLLMBackend()._run_generation_phase(
        _Model(),
        _Database(),
        RuntimeConfig(),
        batch_size=1,
        beam_width=1,
        isl=1024,
        osl=2,
        stride=32,
    )

    assert latency["generation_layerwise"] == pytest.approx(3.25)
    assert latency["generation_tp_allreduce"] == pytest.approx(0.125 * 40)
    assert "generation_moe_ep_alltoall" not in latency
    assert energy["generation_tp_allreduce"] == 0.0
    assert sources["generation_tp_allreduce"] == "silicon"


@pytest.mark.parametrize("gen_error", [PerfDataNotAvailableError("missing GEN KV row"), ValueError("sparse GEN KV")])
def test_vllm_layerwise_mixed_step_falls_back_when_gen_kv_missing(monkeypatch, gen_error) -> None:
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
            if phase == "GEN":
                raise gen_error
            raise AssertionError((phase, batch_size, seq_len, seq_len_kv_cache))

        def query_custom_allreduce(self, quant_mode, tp_size, size):
            del quant_mode
            assert tp_size == 2
            assert size == 10 * 5120
            return 0.1

    latency_ms, _energy_wms, per_ops, _sources = VLLMBackend()._get_mix_step_latency(
        _Model(),
        _Database(),
        RuntimeConfig(),
        ctx_tokens=8,
        gen_tokens=2,
        isl=2048,
        osl=16,
        prefix=0,
    )

    assert latency_ms == pytest.approx((1.0 * 4) + (0.1 * 2 * 4))
    assert per_ops == {
        "mixed_layerwise_context_combined": 4.0,
        "mixed_layerwise_context_tp_allreduce": pytest.approx(0.8),
        "mixed_layerwise_decode_delta": 0.0,
    }


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
        RuntimeConfig(vllm_max_num_batched_tokens=8192),
        batch_size=1,
        isl=16384,
        prefix=0,
    )

    assert latency["context_layerwise"] == pytest.approx((1.0 + 1.5) * 4)


def test_vllm_layerwise_context_falls_back_to_direct_long_row(monkeypatch) -> None:
    from aiconfigurator.sdk.backends import vllm_backend

    monkeypatch.setattr(vllm_backend, "_USE_LAYERWISE", True)

    class _Config:
        tp_size = 1
        pp_size = 1

    class _Model:
        model_path = "Qwen/Qwen3-32B"
        config = _Config()
        _num_layers = 4

    class _Database:
        def query_layerwise(self, model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache=0):
            assert model == "Qwen/Qwen3-32B"
            assert phase == "CTX"
            assert tp_size == 1
            assert batch_size == 1
            if seq_len == 8192 and seq_len_kv_cache == 0:
                return 1.0
            if seq_len == 8192 and seq_len_kv_cache == 8192:
                raise KeyError("sparse CTX KV grid")
            if seq_len == 16384 and seq_len_kv_cache == 0:
                return 2.25
            raise AssertionError((seq_len, seq_len_kv_cache))

    latency, _energy, _sources = VLLMBackend()._run_context_phase(
        _Model(),
        _Database(),
        RuntimeConfig(vllm_max_num_batched_tokens=8192),
        batch_size=1,
        isl=16384,
        prefix=0,
    )

    assert latency["context_layerwise"] == pytest.approx(2.25 * 4)


def test_vllm_layerwise_deepseek_context_composes_sparse_prefix_chunks(monkeypatch) -> None:
    from aiconfigurator.sdk.backends import vllm_backend

    monkeypatch.setattr(vllm_backend, "_USE_LAYERWISE", True)

    class _Config:
        tp_size = 2
        pp_size = 1

    class _Model:
        model_path = "deepseek-ai/DeepSeek-V4-Flash"
        config = _Config()
        _num_layers = 4

    class _Database:
        def query_layerwise(self, model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache=0):
            assert model == "deepseek-ai/DeepSeek-V4-Flash"
            assert phase == "CTX"
            assert tp_size == 2
            assert batch_size == 1
            if seq_len == 8192 and seq_len_kv_cache == 0:
                return 1.0
            if seq_len == 8192 and seq_len_kv_cache == 8192:
                raise KeyError("sparse CTX KV grid")
            if seq_len == 16384 and seq_len_kv_cache == 0:
                return 2.25
            raise AssertionError((seq_len, seq_len_kv_cache))

    latency, _energy, _sources = VLLMBackend()._run_context_phase(
        _Model(),
        _Database(),
        RuntimeConfig(vllm_max_num_batched_tokens=8192),
        batch_size=1,
        isl=16384,
        prefix=0,
    )

    assert latency["context_layerwise"] == pytest.approx(2.0 * 4)
