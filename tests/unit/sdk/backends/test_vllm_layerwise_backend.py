# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.backends import base_backend
from aiconfigurator.sdk.backends.vllm_backend import VLLMBackend
from aiconfigurator.sdk.config import RuntimeConfig
from aiconfigurator.sdk.operations.layerwise import load_layerwise_data
from aiconfigurator.sdk.performance_result import PerformanceResult

pytestmark = pytest.mark.unit


class _Config:
    tp_size = 1
    pp_size = 1
    moe_tp_size = 1
    moe_ep_size = 1
    attention_dp_size = 1
    comm_quant_mode = common.CommQuantMode.half
    moe_quant_mode = common.MoEQuantMode.bfloat16
    workload_distribution = "power_law"
    moe_backend = None


class _Model:
    model_path = "Qwen/Qwen3-32B"
    config = _Config()
    _num_layers = 4
    _hidden_size = 8
    _intermediate_size = 16
    _num_experts = 8
    _topk = 0
    _nextn = 0


def _detail(latency: float, *, layers: int = 4, includes_moe: bool = False, mode: str = "dense") -> dict:
    return {
        "latency": latency,
        "energy": 0.0,
        "measured_layer_count": 1.0,
        "layer_multiplier": float(layers),
        "includes_moe": includes_moe,
        "moe_weight_mode": mode,
        "latency_source": "span",
        "physical_gpus": 1.0,
    }


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

    detail = load_layerwise_data(str(path))["qwen/qwen3.6-35b-a3b"]["GEN"][1][8][4096]

    assert detail["latency"] == pytest.approx((0.10 * 30) + (0.20 * 10))
    assert detail["rms_latency"] == pytest.approx((0.01 * 30) + (0.02 * 10))
    assert detail["rms_kernel_count"] == 5
    assert detail["includes_moe"] is True
    assert detail["layer_type"] == "combined"
    assert detail["measured_layer_count"] == 1.0
    assert detail["layer_multiplier"] == 1.0


def test_layerwise_loader_merges_scheduler_envelopes_by_max(tmp_path) -> None:
    path = tmp_path / "layerwise_perf.csv"
    path.write_text(
        "\n".join(
            [
                "model,phase,attn_tp,batch_size,new_tokens,past_kv,latency_ms,latency_source,"
                "layer_type,measured_layer_count,layer_multiplier,includes_moe",
                "Qwen/Qwen3.6-35B-A3B,GEN,1,1,1,4096,3.70,schedule_to_update,"
                "linear_attention_moe,1,1,false",
                "Qwen/Qwen3.6-35B-A3B,GEN,1,1,1,4096,3.45,schedule_to_update,"
                "full_attention_moe,1,1,false",
            ]
        )
        + "\n"
    )

    detail = load_layerwise_data(str(path))["qwen/qwen3.6-35b-a3b"]["GEN"][1][1][4096]

    assert detail["latency"] == pytest.approx(3.70)
    assert detail["latency_source"] == "schedule_to_update"
    assert detail["includes_moe"] is False
    assert detail["layer_type"] == "combined"
    assert detail["seq_len_q"] == 1.0
    assert detail["seq_len_kv_cache"] == 4096.0


def test_vllm_layerwise_composed_context_row_is_not_scheduler_like() -> None:
    backend = VLLMBackend()

    assert backend._layerwise_scheduler_like_detail(
        {"latency_source": "schedule_to_update", "max_num_batched_tokens": 2048, "seq_len_q": 2048}
    )
    assert not backend._layerwise_scheduler_like_detail(
        {"latency_source": "schedule_to_update", "max_num_batched_tokens": 2048, "seq_len_q": 4096}
    )
    assert not backend._layerwise_scheduler_like_detail({"latency_source": "span"})


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


def test_query_layerwise_detail_supports_legacy_latency_query() -> None:
    class _Database:
        def query_layerwise(self, *args):
            assert args == ("Qwen/Qwen3-32B", "GEN", 1, 2, 1, 4096)
            return PerformanceResult(2.5, energy=1.25, source="silicon")

    detail = VLLMBackend()._query_layerwise_detail(
        _Database(),
        "Qwen/Qwen3-32B",
        "GEN",
        1,
        2,
        1,
        4096,
    )

    assert detail["latency"] == pytest.approx(2.5)
    assert detail["energy"] == pytest.approx(1.25)
    assert detail["latency_source"] == "silicon"


def test_context_step_scales_layerwise_row_and_adds_structural_tp_allreduce(monkeypatch) -> None:
    from aiconfigurator.sdk.backends import vllm_backend

    class _ConfigTp2(_Config):
        tp_size = 2

    class _ModelTp2(_Model):
        config = _ConfigTp2()

    class _Database:
        def query_layerwise_detail(self, *args, **kwargs):
            assert args == ("Qwen/Qwen3-32B", "CTX", 2, 1, 8, 0)
            assert kwargs["moe_tp_size"] == 1
            assert kwargs["moe_ep_size"] == 1
            return _detail(1.0)

        def query_custom_allreduce(self, quant_mode, tp_size, size, database_mode=None, execution_mode=None):
            del quant_mode, database_mode, execution_mode
            assert tp_size == 2
            assert size == 8 * 8
            return PerformanceResult(0.5)

    monkeypatch.setattr(vllm_backend, "_USE_LAYERWISE", True)

    latency, energy, sources = VLLMBackend()._get_context_step_latency(
        _ModelTp2(),
        _Database(),
        RuntimeConfig(),
        ctx_tokens=8,
    )

    assert latency == {
        "context_layerwise": pytest.approx(4.0),
        "context_tp_allreduce": pytest.approx(4.0),
    }
    assert energy["context_layerwise"] == 0.0
    assert sources["context_layerwise"] == "silicon"


def test_context_layer_detail_chunks_long_context_rows() -> None:
    calls = []

    class _Database:
        def query_layerwise_detail(self, model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache, **kwargs):
            del model, phase, tp_size, batch_size, kwargs
            calls.append((seq_len, seq_len_kv_cache))
            return _detail(float(seq_len), layers=1)

    detail = VLLMBackend()._layerwise_context_layer_detail(
        _Database(),
        "Qwen/Qwen3-32B",
        tp_size=1,
        batch_size=1,
        seq_len=10,
        prefix=2,
        max_num_batched_tokens=4,
    )

    assert calls == [(4, 2), (4, 6), (2, 10)]
    assert detail["latency"] == pytest.approx(10.0)


def test_context_noop_moe_mode_is_selected_from_layerwise_rows() -> None:
    class _Database:
        def query_layerwise_detail(self, *args, **kwargs):
            assert args == ("custom/moe-model", "CTX", 2, 1, 2048, 0)
            assert kwargs["moe_weight_mode"] == "noop"
            assert kwargs["max_num_batched_tokens"] == 2048
            assert kwargs["moe_tp_size"] == 1
            assert kwargs["moe_ep_size"] == 4
            return _detail(1.0, mode="noop")

    mode = VLLMBackend()._layerwise_context_noop_moe_weight_mode(
        _Database(),
        model_name="custom/moe-model",
        tp_size=2,
        batch_size=1,
        seq_len=4096,
        runtime_config=RuntimeConfig(vllm_max_num_batched_tokens=2048),
        moe_tp_size=1,
        moe_ep_size=4,
    )

    assert mode == "noop"


def test_decode_step_scales_layerwise_row_and_adds_structural_tp_allreduce(monkeypatch) -> None:
    from aiconfigurator.sdk.backends import vllm_backend

    class _ConfigTp2(_Config):
        tp_size = 2

    class _ModelTp2(_Model):
        config = _ConfigTp2()

    class _Database:
        def query_layerwise_detail(self, *args, **kwargs):
            assert args == ("Qwen/Qwen3-32B", "GEN", 2, 3, 4096, 0)
            assert kwargs["moe_tp_size"] == 1
            assert kwargs["moe_ep_size"] == 1
            return _detail(2.0)

        def query_custom_allreduce(self, quant_mode, tp_size, size, database_mode=None, execution_mode=None):
            del quant_mode, database_mode, execution_mode
            assert tp_size == 2
            assert size == 3 * 8
            return PerformanceResult(0.25)

    monkeypatch.setattr(vllm_backend, "_USE_LAYERWISE", True)

    latency, energy, sources = VLLMBackend()._get_decode_step_latency(
        _ModelTp2(),
        _Database(),
        RuntimeConfig(),
        batch_size=3,
        past_kv=4096,
    )

    assert latency == {
        "generation_layerwise": pytest.approx(8.0),
        "generation_tp_allreduce": pytest.approx(1.0),
    }
    assert energy["generation_layerwise"] == 0.0
    assert sources["generation_layerwise"] == "silicon"


def test_noop_moe_addback_queries_structural_moe_table() -> None:
    class _MoeModel(_Model):
        _topk = 2

    class _Database:
        def query_moe(self, **kwargs):
            assert kwargs["num_tokens"] == 5
            assert kwargs["topk"] == 2
            assert kwargs["workload_distribution"] == "sampled_zipf_0.8"
            return PerformanceResult(1.25, energy=0.5, source="silicon")

    moe, router, shared, bundled = VLLMBackend()._layerwise_noop_moe_addback(
        _MoeModel(),
        _Database(),
        token_count=5,
        num_layers=3,
        is_context=False,
    )

    assert moe == (pytest.approx(3.75), pytest.approx(1.5), "silicon")
    assert router == (0.0, 0.0, "silicon")
    assert shared == (0.0, 0.0, "silicon")
    assert bundled is False


def test_mixed_step_is_mechanical_context_plus_decode(monkeypatch) -> None:
    from aiconfigurator.sdk.backends import vllm_backend

    backend = VLLMBackend()
    monkeypatch.setattr(vllm_backend, "_USE_LAYERWISE", True)

    def _context_step(model, database, runtime_config, *, ctx_tokens, ctx_kv_tokens=0, ctx_requests=1):
        del model, database, runtime_config
        assert (ctx_tokens, ctx_kv_tokens, ctx_requests) == (9, 3, 3)
        return (
            {"context_layerwise": 5.0, "context_tp_allreduce": 0.75},
            {"context_layerwise": 1.0, "context_tp_allreduce": 0.0},
            {"context_layerwise": "silicon", "context_tp_allreduce": "silicon"},
        )

    def _decode_step(model, database, runtime_config, *, batch_size, past_kv):
        del model, database, runtime_config
        assert (batch_size, past_kv) == (4, 12)
        return (
            {"generation_layerwise": 2.25, "generation_moe": 0.5},
            {"generation_layerwise": 0.25, "generation_moe": 0.5},
            {"generation_layerwise": "silicon", "generation_moe": "silicon"},
        )

    monkeypatch.setattr(backend, "_get_context_step_latency", _context_step)
    monkeypatch.setattr(backend, "_get_decode_step_latency", _decode_step)

    latency_ms, energy_wms, per_ops, sources = backend._get_mix_step_latency(
        _Model(),
        object(),
        RuntimeConfig(),
        ctx_tokens=9,
        gen_tokens=4,
        isl=8,
        osl=8,
        prefix=1,
        ctx_requests=3,
    )

    assert latency_ms == pytest.approx(8.5)
    assert energy_wms == pytest.approx(1.75)
    assert per_ops == {
        "mixed_layerwise_context_combined": 5.0,
        "mixed_layerwise_context_tp_allreduce": 0.75,
        "mixed_layerwise_decode_delta": 2.75,
    }
    assert sources == {
        "mixed_layerwise_context_combined": "silicon",
        "mixed_layerwise_context_tp_allreduce": "silicon",
        "mixed_layerwise_decode_delta": "silicon",
    }


def test_calibration_helpers_are_not_part_of_backend_surface() -> None:
    deleted_helpers = {
        "_layerwise_dense_mixed_decode_tail_slices",
        "_layerwise_dense_mixed_context_envelope_multiplier",
        "_layerwise_noop_context_continuation_floor_ms",
        "_layerwise_qwen_noop_scheduler_residual_ms",
        "_layerwise_smoothed_deepseek_generation_detail",
        "_layerwise_smoothed_noop_small_context_detail",
        "_layerwise_deepseek_large_context_moe_overhead_ms",
        "_layerwise_deepseek_tp1_small_context_overhead_ms",
        "_deepseek_context_moe_weight_mode",
        "_deepseek_context_moe_distribution_override",
        "_layerwise_noop_moe_addback_for_context",
    }

    assert not any(hasattr(VLLMBackend, name) for name in deleted_helpers)
