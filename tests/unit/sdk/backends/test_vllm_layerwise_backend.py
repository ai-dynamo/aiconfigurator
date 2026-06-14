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

    data = load_layerwise_data(str(path))
    detail = data["qwen/qwen3.6-35b-a3b"]["GEN"][1][1][4096]

    assert detail["latency"] == pytest.approx(3.70)
    assert detail["latency_source"] == "schedule_to_update"
    assert detail["includes_moe"] is False
    assert detail["layer_type"] == "combined"
    assert detail["seq_len_q"] == 1.0
    assert detail["seq_len_kv_cache"] == 4096.0
    assert [component["layer_type"] for component in detail["components"]] == [
        "linear_attention_moe",
        "full_attention_moe",
    ]

def test_vllm_layerwise_composed_context_row_is_not_scheduler_like() -> None:
    backend = VLLMBackend()

    assert backend._layerwise_scheduler_like_detail(
        {
            "latency_source": "schedule_to_update",
            "max_num_batched_tokens": 2048,
            "seq_len_q": 2048,
        }
    )
    assert not backend._layerwise_scheduler_like_detail(
        {
            "latency_source": "schedule_to_update",
            "max_num_batched_tokens": 2048,
            "seq_len_q": 4096,
        }
    )
    assert not backend._layerwise_scheduler_like_detail(
        {
            "latency_source": "schedule_to_update",
            "max_num_batched_tokens": 2048,
            "query_seq_len_q": 7664,
            "components": [
                {
                    "latency_source": "schedule_to_update",
                    "max_num_batched_tokens": 2048,
                    "seq_len_q": 1,
                }
            ],
        }
    )
    assert backend._layerwise_scheduler_like_detail(
        {
            "latency_source": "schedule_to_update",
            "max_num_batched_tokens": 2048,
            "seq_len_q": 1,
            "seq_len_kv_cache": 4096,
            "query_seq_len_q": 4096,
        }
    )


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
        def query_layerwise_detail(self, model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache=0, **kwargs):
            del kwargs
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


def test_vllm_layerwise_deepseek_no_ep_mixed_uses_context_floor(monkeypatch) -> None:
    from aiconfigurator.sdk.backends import vllm_backend

    monkeypatch.setattr(vllm_backend, "_USE_LAYERWISE", True)

    class _Config:
        tp_size = 4
        pp_size = 1
        attention_dp_size = 1
        moe_tp_size = 4
        moe_ep_size = 1
        moe_quant_mode = common.MoEQuantMode.w4a8_mxfp4_mxfp8
        workload_distribution = "power_law"
        moe_backend = None
        enable_eplb = False
        gemm_quant_mode = common.GEMMQuantMode.bfloat16

    class _Model:
        model_path = "deepseek-ai/DeepSeek-V4-Flash"
        config = _Config()
        _num_layers = 1
        _hidden_size = 2048
        _topk = 6
        _num_experts = 256
        _moe_inter_size = 512
        _shared_expert_inter_size = 0
        _nextn = 0

    class _Database:
        def query_layerwise_detail(
            self,
            model,
            phase,
            tp_size,
            batch_size,
            seq_len,
            seq_len_kv_cache=0,
            **kwargs,
        ):
            assert model == "deepseek-ai/DeepSeek-V4-Flash"
            assert phase == "CTX"
            assert tp_size == 4
            assert batch_size == 1
            assert seq_len_kv_cache == 0
            if seq_len == 4097:
                return {
                    "latency": 50.0,
                    "energy": 0.0,
                    "latency_source": "schedule_to_update",
                    "includes_moe": False,
                    "moe_weight_mode": "noop",
                    "measured_layer_count": 1,
                    "layer_multiplier": 1,
                }
            if seq_len == 4096:
                return {
                    "latency": 100.0,
                    "energy": 0.0,
                    "latency_source": "schedule_to_update",
                    "includes_moe": False,
                    "moe_weight_mode": "noop",
                    "measured_layer_count": 1,
                    "layer_multiplier": 1,
                }
            raise AssertionError((phase, batch_size, seq_len, seq_len_kv_cache))

        def query_custom_allreduce(self, *args, **kwargs):
            return 0.0

    backend = VLLMBackend()
    monkeypatch.setattr(
        backend,
        "_layerwise_noop_moe_addback_for_context",
        lambda *args, **kwargs: ((0.0, 0.0, "silicon"), (0.0, 0.0, "silicon"), (0.0, 0.0, "silicon"), True),
    )
    latency_ms, _energy_wms, per_ops, _sources = backend._get_mix_step_latency(
        _Model(),
        _Database(),
        RuntimeConfig(),
        ctx_tokens=4096,
        gen_tokens=1,
        isl=4096,
        osl=1,
        prefix=0,
    )

    assert latency_ms == pytest.approx(105.0)
    assert per_ops["mixed_layerwise_context_floor"] == pytest.approx(105.0)


def test_vllm_layerwise_deepseek_compressed_small_continuation_skips_decode_tail(monkeypatch) -> None:
    from aiconfigurator.sdk.backends import vllm_backend

    monkeypatch.setattr(vllm_backend, "_USE_LAYERWISE", True)

    class _Extra:
        compress_ratios = (0, 4, 128)
        sliding_window = 128

    class _Config:
        tp_size = 1
        pp_size = 1
        attention_dp_size = 1
        moe_tp_size = 1
        moe_ep_size = 1

    class _Model:
        model_path = "deepseek-ai/DeepSeek-V4-Flash"
        config = _Config()
        extra_params = _Extra()
        _num_layers = 1
        _hidden_size = 2048
        _topk = 6
        _num_experts = 256
        _moe_inter_size = 512
        _shared_expert_inter_size = 2048

    class _Database:
        def query_layerwise_detail(
            self,
            model,
            phase,
            tp_size,
            batch_size,
            seq_len,
            seq_len_kv_cache=0,
            **kwargs,
        ):
            assert model == "deepseek-ai/DeepSeek-V4-Flash"
            assert phase == "CTX"
            assert tp_size == 1
            assert batch_size == 1
            assert seq_len in {3, 2048}
            return {
                "latency": 88.0 if seq_len == 2048 else 0.0,
                "energy": 0.0,
                "latency_source": "schedule_to_update",
                "includes_moe": False,
                "moe_weight_mode": "noop",
                "measured_layer_count": 1,
                "layer_multiplier": 1,
            }

        def query_custom_allreduce(self, *args, **kwargs):
            return 0.0

    backend = VLLMBackend()
    monkeypatch.setattr(
        backend,
        "_layerwise_noop_moe_addback",
        lambda *args, **kwargs: ((0.0, 0.0, "silicon"), (0.0, 0.0, "silicon"), (0.0, 0.0, "silicon"), True),
    )
    monkeypatch.setattr(
        backend,
        "_get_context_step_latency",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("small continuation should not use floor")),
    )

    latency_ms, _energy_wms, per_ops, _sources = backend._get_mix_step_latency(
        _Model(),
        _Database(),
        RuntimeConfig(vllm_max_num_batched_tokens=2048),
        ctx_tokens=4097,
        ctx_requests=2,
        gen_tokens=2,
        isl=4096,
        osl=1,
        prefix=2048,
    )

    assert latency_ms == pytest.approx(88.0)
    assert per_ops == {
        "mixed_layerwise_context_combined": 88.0,
        "mixed_layerwise_context_tp_allreduce": 0.0,
        "mixed_layerwise_decode_delta": 0.0,
    }


def test_vllm_layerwise_deepseek_compressed_large_continuation_caps_decode_tail(monkeypatch) -> None:
    from aiconfigurator.sdk.backends import vllm_backend

    monkeypatch.setattr(vllm_backend, "_USE_LAYERWISE", True)

    class _Extra:
        compress_ratios = (0, 4, 128)
        sliding_window = 128

    class _Config:
        tp_size = 1
        pp_size = 1
        attention_dp_size = 1
        moe_tp_size = 1
        moe_ep_size = 1

    class _Model:
        model_path = "deepseek-ai/DeepSeek-V4-Flash"
        config = _Config()
        extra_params = _Extra()
        _num_layers = 1
        _hidden_size = 2048
        _topk = 6
        _num_experts = 256
        _moe_inter_size = 512
        _shared_expert_inter_size = 2048

    class _Database:
        def query_layerwise_detail(
            self,
            model,
            phase,
            tp_size,
            batch_size,
            seq_len,
            seq_len_kv_cache=0,
            **kwargs,
        ):
            assert model == "deepseek-ai/DeepSeek-V4-Flash"
            assert tp_size == 1
            if phase == "CTX":
                assert batch_size == 1
                assert seq_len == 2048 or 1 <= seq_len < 2048
                return {
                    "latency": 100.0 if seq_len == 2048 else 5.0,
                    "energy": 0.0,
                    "latency_source": "schedule_to_update",
                    "includes_moe": False,
                    "moe_weight_mode": "noop",
                    "measured_layer_count": 1,
                    "layer_multiplier": 1,
                }
            if phase == "GEN":
                assert batch_size == 4
                assert seq_len == 4096
                return {
                    "latency": 80.0,
                    "energy": 0.0,
                    "latency_source": "schedule_to_update",
                    "includes_moe": False,
                    "moe_weight_mode": "noop",
                    "measured_layer_count": 1,
                    "layer_multiplier": 1,
                }
            raise AssertionError((phase, batch_size, seq_len, seq_len_kv_cache))

        def query_custom_allreduce(self, *args, **kwargs):
            return 0.0

    backend = VLLMBackend()
    monkeypatch.setattr(
        backend,
        "_layerwise_noop_moe_addback",
        lambda *args, **kwargs: ((0.0, 0.0, "silicon"), (0.0, 0.0, "silicon"), (0.0, 0.0, "silicon"), True),
    )
    monkeypatch.setattr(
        backend,
        "_get_context_step_latency",
        lambda *args, **kwargs: ({"context_layerwise": 0.0}, {}, {}),
    )

    latency_ms, _energy_wms, per_ops, _sources = backend._get_mix_step_latency(
        _Model(),
        _Database(),
        RuntimeConfig(vllm_max_num_batched_tokens=2048),
        ctx_tokens=8188,
        ctx_requests=3,
        gen_tokens=4,
        isl=4096,
        osl=1,
        prefix=1364,
    )

    assert latency_ms == pytest.approx(305.0)
    assert per_ops == {
        "mixed_layerwise_context_combined": 205.0,
        "mixed_layerwise_context_tp_allreduce": 0.0,
        "mixed_layerwise_decode_delta": 100.0,
    }


def test_vllm_layerwise_deepseek_ep_long_prefix_uses_context_envelope(monkeypatch) -> None:
    from aiconfigurator.sdk.backends import vllm_backend

    monkeypatch.setattr(vllm_backend, "_USE_LAYERWISE", True)

    class _Extra:
        compress_ratios = (0, 4, 128)
        sliding_window = 128

    class _Config:
        tp_size = 1
        pp_size = 1
        attention_dp_size = 1
        moe_tp_size = 1
        moe_ep_size = 4

    class _Model:
        model_path = "deepseek-ai/DeepSeek-V4-Flash"
        config = _Config()
        extra_params = _Extra()
        _num_layers = 1
        _hidden_size = 2048
        _topk = 8
        _num_experts = 256
        _moe_inter_size = 512
        _shared_expert_inter_size = 2048

    class _Database:
        def query_layerwise_detail(
            self,
            model,
            phase,
            tp_size,
            batch_size,
            seq_len,
            seq_len_kv_cache=0,
            **kwargs,
        ):
            assert model == "deepseek-ai/DeepSeek-V4-Flash"
            assert tp_size == 1
            if phase == "GEN":
                raise AssertionError("long-prefix compressed context should cover the decode tail")
            if (
                phase == "CTX"
                and batch_size == 1
                and (seq_len, seq_len_kv_cache) in {(2048, 8192), (2048, 10240), (818, 12288)}
            ):
                return {
                    "latency": 44.0 if seq_len == 2048 else 46.0,
                    "energy": 0.0,
                    "latency_source": "schedule_to_update",
                    "includes_moe": False,
                    "moe_weight_mode": "noop",
                    "max_num_batched_tokens": 2048,
                    "measured_layer_count": 1,
                    "layer_multiplier": 1,
                }
            raise AssertionError((phase, batch_size, seq_len, seq_len_kv_cache))

        def query_custom_allreduce(self, *args, **kwargs):
            return 0.0

    backend = VLLMBackend()
    monkeypatch.setattr(
        backend,
        "_layerwise_noop_moe_addback",
        lambda *args, **kwargs: ((0.0, 0.0, "silicon"), (0.0, 0.0, "silicon"), (0.0, 0.0, "silicon"), True),
    )

    latency_ms, _energy_wms, per_ops, _sources = backend._get_mix_step_latency(
        _Model(),
        _Database(),
        RuntimeConfig(vllm_max_num_batched_tokens=2048),
        ctx_tokens=4906,
        gen_tokens=8,
        isl=5977,
        osl=1,
        prefix=8192,
    )

    assert latency_ms == pytest.approx(134.0)
    assert per_ops == {
        "mixed_layerwise_context_combined": 134.0,
        "mixed_layerwise_context_tp_allreduce": 0.0,
        "mixed_layerwise_decode_delta": 0.0,
    }


def test_vllm_layerwise_deepseek_ep_mid_prefix_uses_context_envelope(monkeypatch) -> None:
    from aiconfigurator.sdk.backends import vllm_backend

    monkeypatch.setattr(vllm_backend, "_USE_LAYERWISE", True)

    class _Extra:
        compress_ratios = (0, 4, 128)
        sliding_window = 128

    class _Config:
        tp_size = 1
        pp_size = 1
        attention_dp_size = 1
        moe_tp_size = 1
        moe_ep_size = 4

    class _Model:
        model_path = "deepseek-ai/DeepSeek-V4-Flash"
        config = _Config()
        extra_params = _Extra()
        _num_layers = 1
        _hidden_size = 2048
        _topk = 8
        _num_experts = 256
        _moe_inter_size = 512
        _shared_expert_inter_size = 2048

    class _Database:
        def query_layerwise_detail(
            self,
            model,
            phase,
            tp_size,
            batch_size,
            seq_len,
            seq_len_kv_cache=0,
            **kwargs,
        ):
            assert model == "deepseek-ai/DeepSeek-V4-Flash"
            assert tp_size == 1
            if phase == "GEN":
                raise AssertionError("mid-prefix compressed context should cover the decode tail")
            if phase == "CTX" and batch_size == 1:
                return {
                    "latency": 123.0,
                    "energy": 0.0,
                    "latency_source": "schedule_to_update",
                    "includes_moe": False,
                    "moe_weight_mode": "noop",
                    "max_num_batched_tokens": 2048,
                    "measured_layer_count": 1,
                    "layer_multiplier": 1,
                }
            raise AssertionError((phase, batch_size, seq_len, seq_len_kv_cache))

        def query_custom_allreduce(self, *args, **kwargs):
            return 0.0

    backend = VLLMBackend()
    monkeypatch.setattr(
        backend,
        "_layerwise_noop_moe_addback",
        lambda *args, **kwargs: ((0.0, 0.0, "silicon"), (0.0, 0.0, "silicon"), (0.0, 0.0, "silicon"), True),
    )

    latency_ms, _energy_wms, per_ops, _sources = backend._get_mix_step_latency(
        _Model(),
        _Database(),
        RuntimeConfig(vllm_max_num_batched_tokens=2048),
        ctx_tokens=4520,
        gen_tokens=7,
        isl=5972,
        osl=1,
        prefix=3388,
    )

    assert latency_ms == pytest.approx(369.0)
    assert per_ops == {
        "mixed_layerwise_context_combined": 369.0,
        "mixed_layerwise_context_tp_allreduce": 0.0,
        "mixed_layerwise_decode_delta": 0.0,
    }


def test_vllm_layerwise_deepseek_high_kv_small_continuation_adds_decode_tail(monkeypatch) -> None:
    from aiconfigurator.sdk.backends import vllm_backend

    monkeypatch.setattr(vllm_backend, "_USE_LAYERWISE", True)

    class _Extra:
        compress_ratios = (0, 4, 128)
        sliding_window = 128

    class _Config:
        tp_size = 1
        pp_size = 1
        attention_dp_size = 1
        moe_tp_size = 1
        moe_ep_size = 4

    class _Model:
        model_path = "deepseek-ai/DeepSeek-V4-Flash"
        config = _Config()
        extra_params = _Extra()
        _num_layers = 1
        _hidden_size = 2048
        _topk = 8
        _num_experts = 256
        _moe_inter_size = 512
        _shared_expert_inter_size = 2048

    class _Database:
        def query_layerwise_detail(
            self,
            model,
            phase,
            tp_size,
            batch_size,
            seq_len,
            seq_len_kv_cache=0,
            **kwargs,
        ):
            assert model == "deepseek-ai/DeepSeek-V4-Flash"
            assert tp_size == 1
            if phase == "CTX" and batch_size == 1 and seq_len == 1538 and seq_len_kv_cache == 8192:
                return {
                    "latency": 44.0,
                    "energy": 0.0,
                    "latency_source": "schedule_to_update",
                    "includes_moe": False,
                    "moe_weight_mode": "noop",
                    "max_num_batched_tokens": 2048,
                    "measured_layer_count": 1,
                    "layer_multiplier": 1,
                }
            if phase == "GEN" and batch_size == 4:
                return {
                    "latency": 35.0,
                    "energy": 0.0,
                    "latency_source": "schedule_to_update",
                    "measured_layer_count": 1,
                    "layer_multiplier": 1,
                }
            raise AssertionError((phase, batch_size, seq_len, seq_len_kv_cache))

        def query_custom_allreduce(self, *args, **kwargs):
            return 0.0

    backend = VLLMBackend()
    monkeypatch.setattr(
        backend,
        "_layerwise_noop_moe_addback",
        lambda *args, **kwargs: ((0.0, 0.0, "silicon"), (0.0, 0.0, "silicon"), (0.0, 0.0, "silicon"), True),
    )

    latency_ms, _energy_wms, per_ops, _sources = backend._get_mix_step_latency(
        _Model(),
        _Database(),
        RuntimeConfig(vllm_max_num_batched_tokens=2048),
        ctx_tokens=1530,
        gen_tokens=8,
        isl=5939,
        osl=1,
        prefix=8192,
    )

    assert latency_ms == pytest.approx(114.0)
    assert per_ops == {
        "mixed_layerwise_context_combined": 44.0,
        "mixed_layerwise_context_tp_allreduce": 0.0,
        "mixed_layerwise_decode_delta": 70.0,
    }


def test_vllm_layerwise_mixed_noop_moe_continuation_uses_saturated_context_floor(monkeypatch) -> None:
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
        _topk = 8

    class _Database:
        def query_layerwise_detail(self, model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache=0):
            assert model == "Qwen/Qwen3.6-35B-A3B"
            assert tp_size == 2
            if phase == "CTX" and batch_size == 1 and seq_len == 931 and seq_len_kv_cache in {0, 3168}:
                return {
                    "latency": 5.0,
                    "energy": 0.0,
                    "latency_source": "schedule_to_update",
                    "includes_moe": False,
                }
            if phase == "CTX" and batch_size == 1 and seq_len == 4096 and seq_len_kv_cache == 0:
                return {
                    "latency": 12.0,
                    "energy": 0.0,
                    "latency_source": "schedule_to_update",
                    "includes_moe": False,
                }
            if phase == "GEN" and batch_size == 3 and seq_len == 4096:
                return {
                    "latency": 2.0,
                    "energy": 0.0,
                    "latency_source": "schedule_to_update",
                    "includes_moe": False,
                }
            raise AssertionError((phase, batch_size, seq_len, seq_len_kv_cache))

        def query_custom_allreduce(self, quant_mode, tp_size, size):
            raise AssertionError("no-op MoE scheduler envelope already covers this mixed-step TP tail")

    latency_ms, _energy_wms, per_ops, _sources = VLLMBackend()._get_mix_step_latency(
        _Model(),
        _Database(),
        RuntimeConfig(vllm_max_num_batched_tokens=2048),
        ctx_tokens=928,
        gen_tokens=3,
        isl=4096,
        osl=1,
        prefix=3168,
    )

    assert latency_ms == pytest.approx(55.2)
    assert per_ops == {
        "mixed_layerwise_context_combined": pytest.approx(43.2),
        "mixed_layerwise_context_tp_allreduce": 0.0,
        "mixed_layerwise_decode_delta": pytest.approx(12.0),
    }


def test_vllm_layerwise_mixed_moe_multi_context_requests_use_sqrt_effective_context_shape(monkeypatch) -> None:
    from aiconfigurator.sdk.backends import vllm_backend

    monkeypatch.setattr(vllm_backend, "_USE_LAYERWISE", True)

    queried_context_chunks = []

    class _Config:
        tp_size = 2
        pp_size = 1
        moe_tp_size = 2
        moe_ep_size = 1

    class _Model:
        model_path = "Qwen/Qwen3.6-35B-A3B"
        config = _Config()
        _num_layers = 4
        _hidden_size = 2048
        _num_experts = 256
        _topk = 8

    class _Database:
        def query_layerwise_detail(
            self,
            model,
            phase,
            tp_size,
            batch_size,
            seq_len,
            seq_len_kv_cache=0,
            **kwargs,
        ):
            assert model == "Qwen/Qwen3.6-35B-A3B"
            assert tp_size == 2
            if phase == "CTX":
                queried_context_chunks.append((seq_len, seq_len_kv_cache))
                assert batch_size == 1
                if seq_len == 2048 and seq_len_kv_cache in {0, 2464}:
                    return {
                        "latency": 8.0,
                        "energy": 0.0,
                        "latency_source": "schedule_to_update",
                        "includes_moe": True,
                        "measured_layer_count": 1,
                        "layer_multiplier": 4,
                    }
                if seq_len == 245 and seq_len_kv_cache in {2048, 4512}:
                    return {
                        "latency": 1.0,
                        "energy": 0.0,
                        "latency_source": "schedule_to_update",
                        "includes_moe": True,
                        "measured_layer_count": 1,
                        "layer_multiplier": 4,
                    }
            if phase == "GEN" and batch_size == 1 and seq_len == 4097:
                return {
                    "latency": 2.0,
                    "energy": 0.0,
                    "latency_source": "schedule_to_update",
                    "includes_moe": False,
                    "moe_weight_mode": "noop",
                    "measured_layer_count": 1,
                    "layer_multiplier": 4,
                }
            raise AssertionError((phase, batch_size, seq_len, seq_len_kv_cache))

        def query_custom_allreduce(self, *args, **kwargs):
            raise AssertionError("scheduler-like MoE context should cover mixed TP allreduce")

    latency_ms, _energy_wms, per_ops, _sources = VLLMBackend()._get_mix_step_latency(
        _Model(),
        _Database(),
        RuntimeConfig(vllm_max_num_batched_tokens=2048),
        ctx_tokens=3970,
        ctx_requests=3,
        gen_tokens=1,
        isl=4097,
        osl=1,
        prefix=2464,
    )

    assert (3970, 2464) not in queried_context_chunks
    assert (1324, 2464) not in queried_context_chunks
    assert (1325, 2464) not in queried_context_chunks
    assert (2293, 2464) not in queried_context_chunks
    assert (2048, 2464) in queried_context_chunks
    assert (245, 4512) in queried_context_chunks
    assert latency_ms == pytest.approx(44.0)
    assert per_ops["mixed_layerwise_context_combined"] == pytest.approx(36.0)
    assert per_ops["mixed_layerwise_decode_delta"] == pytest.approx(8.0)


def test_vllm_layerwise_mixed_noop_moe_multi_context_high_decode_uses_floor(monkeypatch) -> None:
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
        def query_layerwise_detail(
            self,
            model,
            phase,
            tp_size,
            batch_size,
            seq_len,
            seq_len_kv_cache=0,
            **kwargs,
        ):
            assert model == "Qwen/Qwen3.6-35B-A3B"
            assert tp_size == 4
            if phase == "CTX" and batch_size == 1:
                if seq_len == 1516 and seq_len_kv_cache in {0, 1584}:
                    return {
                        "latency": 5.0,
                        "energy": 0.0,
                        "latency_source": "schedule_to_update",
                        "includes_moe": False,
                        "moe_weight_mode": "noop",
                        "measured_layer_count": 1,
                        "layer_multiplier": 4,
                        "max_num_batched_tokens": 2048,
                        "query_seq_len_q": seq_len,
                    }
                if seq_len == 4096 and seq_len_kv_cache == 0:
                    return {
                        "latency": 12.0,
                        "energy": 0.0,
                        "latency_source": "schedule_to_update",
                        "includes_moe": False,
                        "moe_weight_mode": "noop",
                        "measured_layer_count": 1,
                        "layer_multiplier": 4,
                        "max_num_batched_tokens": 2048,
                        "query_seq_len_q": seq_len,
                    }
            if phase == "GEN" and batch_size == 4 and seq_len == 4096:
                return {
                    "latency": 2.0,
                    "energy": 0.0,
                    "latency_source": "schedule_to_update",
                    "includes_moe": False,
                    "moe_weight_mode": "noop",
                    "measured_layer_count": 1,
                    "layer_multiplier": 4,
                }
            raise AssertionError((phase, batch_size, seq_len, seq_len_kv_cache))

        def query_custom_allreduce(self, *args, **kwargs):
            raise AssertionError("scheduler-like MoE context should cover mixed TP allreduce")

    latency_ms, _energy_wms, per_ops, _sources = VLLMBackend()._get_mix_step_latency(
        _Model(),
        _Database(),
        RuntimeConfig(vllm_max_num_batched_tokens=2048),
        ctx_tokens=3008,
        ctx_requests=4,
        gen_tokens=12,
        isl=4096,
        osl=1,
        prefix=1584,
    )

    assert latency_ms == pytest.approx(40.8)
    assert per_ops["mixed_layerwise_context_combined"] == pytest.approx(20.0)
    assert per_ops["mixed_layerwise_decode_delta"] == pytest.approx(10.0)
    assert per_ops["mixed_layerwise_ep_high_decode_floor"] == pytest.approx(40.8)


def test_vllm_layerwise_mixed_noop_moe_multi_context_low_decode_uses_continuation_floor(monkeypatch) -> None:
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
        _shared_expert_inter_size = 0

    class _Database:
        def query_layerwise_detail(
            self,
            model,
            phase,
            tp_size,
            batch_size,
            seq_len,
            seq_len_kv_cache=0,
            **kwargs,
        ):
            assert model == "Qwen/Qwen3.6-35B-A3B"
            assert tp_size == 4
            if phase == "CTX" and batch_size == 1:
                if seq_len == 2048 and seq_len_kv_cache in {0, 1024, 2048}:
                    return {
                        "latency": 5.0,
                        "energy": 0.0,
                        "latency_source": "schedule_to_update",
                        "includes_moe": False,
                        "moe_weight_mode": "noop",
                        "measured_layer_count": 1,
                        "layer_multiplier": 4,
                        "max_num_batched_tokens": 2048,
                        "query_seq_len_q": seq_len,
                    }
                if seq_len == 4096 and seq_len_kv_cache == 0:
                    return {
                        "latency": 10.0,
                        "energy": 0.0,
                        "latency_source": "schedule_to_update",
                        "includes_moe": False,
                        "moe_weight_mode": "noop",
                        "measured_layer_count": 1,
                        "layer_multiplier": 4,
                        "max_num_batched_tokens": 2048,
                    }
            if phase == "GEN" and batch_size == 1 and seq_len == 4096:
                return {
                    "latency": 2.0,
                    "energy": 0.0,
                    "latency_source": "schedule_to_update",
                    "includes_moe": False,
                    "moe_weight_mode": "noop",
                    "measured_layer_count": 1,
                    "layer_multiplier": 4,
                }
            raise AssertionError((phase, batch_size, seq_len, seq_len_kv_cache))

        def query_custom_allreduce(self, *args, **kwargs):
            raise AssertionError("scheduler-like MoE context should cover mixed TP allreduce")

    latency_ms, _energy_wms, per_ops, _sources = VLLMBackend()._get_mix_step_latency(
        _Model(),
        _Database(),
        RuntimeConfig(vllm_max_num_batched_tokens=2048),
        ctx_tokens=4094,
        ctx_requests=4,
        gen_tokens=1,
        isl=4096,
        osl=1,
        prefix=1024,
    )

    assert latency_ms == pytest.approx(54.8)
    assert per_ops["mixed_layerwise_context_combined"] == pytest.approx(36.0)
    assert per_ops["mixed_layerwise_decode_delta"] == pytest.approx(10.0)
    assert per_ops["mixed_layerwise_ep_high_decode_floor"] == pytest.approx(54.8)


def test_vllm_layerwise_mixed_dense_scheduler_context_uses_overlapped_tp_tail(monkeypatch) -> None:
    from aiconfigurator.sdk.backends import vllm_backend

    monkeypatch.setattr(vllm_backend, "_USE_LAYERWISE", True)

    class _Config:
        tp_size = 8
        pp_size = 1

    class _Model:
        model_path = "Qwen/Qwen3-32B"
        config = _Config()
        _num_layers = 4
        _hidden_size = 1024
        _topk = 0
        _num_experts = 0

    class _Database:
        def query_layerwise_detail(self, model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache=0, **kwargs):
            assert (model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache) == (
                "Qwen/Qwen3-32B",
                "CTX",
                8,
                1,
                4097,
                0,
            )
            return {
                "latency": 10.0,
                "energy": 0.0,
                "latency_source": "schedule_to_update",
                "includes_moe": False,
                "measured_layer_count": 1,
                "layer_multiplier": 4,
                "max_num_batched_tokens": 8192,
                "seq_len_q": 4097,
            }

        def query_custom_allreduce(self, *args, **kwargs):
            return 1.0

    latency_ms, _energy_wms, per_ops, _sources = VLLMBackend()._get_mix_step_latency(
        _Model(),
        _Database(),
        RuntimeConfig(),
        ctx_tokens=4096,
        gen_tokens=1,
        isl=4096,
        osl=1,
        prefix=0,
    )

    assert latency_ms == pytest.approx(42.0)
    assert per_ops["mixed_layerwise_context_combined"] == pytest.approx(40.0)
    assert per_ops["mixed_layerwise_context_tp_allreduce"] == pytest.approx(2.0)
    assert per_ops["mixed_layerwise_decode_delta"] == pytest.approx(0.0)


def test_vllm_layerwise_mixed_dense_multi_context_requests_use_aggregate_context_chunks(monkeypatch) -> None:
    from aiconfigurator.sdk.backends import vllm_backend

    monkeypatch.setattr(vllm_backend, "_USE_LAYERWISE", True)

    queried_context_chunks = []

    class _Config:
        tp_size = 1
        pp_size = 1
        moe_tp_size = 1
        moe_ep_size = 1

    class _Model:
        model_path = "Qwen/Qwen3-32B"
        config = _Config()
        _num_layers = 4
        _hidden_size = 2048
        _num_experts = 0
        _topk = 0

    class _Database:
        def query_layerwise_detail(
            self,
            model,
            phase,
            tp_size,
            batch_size,
            seq_len,
            seq_len_kv_cache=0,
            **kwargs,
        ):
            assert model == "Qwen/Qwen3-32B"
            assert tp_size == 1
            if phase == "CTX":
                queried_context_chunks.append((seq_len, seq_len_kv_cache))
                assert batch_size == 1
                if seq_len == 2048 and seq_len_kv_cache == 2464:
                    return {
                        "latency": 5.0,
                        "energy": 0.0,
                        "latency_source": "schedule_to_update",
                        "includes_moe": False,
                        "measured_layer_count": 1,
                        "layer_multiplier": 4,
                    }
                if seq_len == 1923 and seq_len_kv_cache == 4512:
                    return {
                        "latency": 4.0,
                        "energy": 0.0,
                        "latency_source": "schedule_to_update",
                        "includes_moe": False,
                        "measured_layer_count": 1,
                        "layer_multiplier": 4,
                    }
            raise AssertionError((phase, batch_size, seq_len, seq_len_kv_cache))

        def query_custom_allreduce(self, *args, **kwargs):
            raise AssertionError("tp=1 dense mixed case should not query allreduce")

    latency_ms, _energy_wms, per_ops, _sources = VLLMBackend()._get_mix_step_latency(
        _Model(),
        _Database(),
        RuntimeConfig(vllm_max_num_batched_tokens=2048),
        ctx_tokens=3970,
        ctx_requests=3,
        gen_tokens=1,
        isl=4097,
        osl=1,
        prefix=2464,
    )

    assert (2048, 2464) in queried_context_chunks
    assert (1923, 4512) in queried_context_chunks
    assert latency_ms == pytest.approx(32.4)
    assert per_ops["mixed_layerwise_context_combined"] == pytest.approx(32.4)
    assert per_ops["mixed_layerwise_decode_delta"] == pytest.approx(0.0)


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


def test_vllm_layerwise_mixed_ep_tiny_fresh_prefill_hides_decode_tail(monkeypatch) -> None:
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
        _shared_expert_inter_size = 0

    class _Database:
        def query_layerwise_detail(self, model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache=0, **kwargs):
            assert model == "Qwen/Qwen3.6-35B-A3B"
            assert tp_size == 4
            if phase == "CTX" and batch_size == 1 and seq_len == 131 and seq_len_kv_cache == 0:
                return {
                    "latency": 12.0,
                    "energy": 0.0,
                    "latency_source": "schedule_to_update",
                    "measured_layer_count": 4,
                    "layer_multiplier": 4,
                    "moe_weight_mode": "noop",
                    "max_num_batched_tokens": 2048,
                }
            raise AssertionError((phase, batch_size, seq_len, seq_len_kv_cache))

    latency_ms, _energy_wms, per_ops, _sources = VLLMBackend()._get_mix_step_latency(
        _Model(),
        _Database(),
        RuntimeConfig(vllm_max_num_batched_tokens=2048),
        ctx_tokens=100,
        gen_tokens=31,
        isl=4096,
        osl=1,
        prefix=0,
    )

    assert latency_ms == pytest.approx(12.0)
    assert per_ops == {
        "mixed_layerwise_context_combined": 12.0,
        "mixed_layerwise_context_tp_allreduce": 0.0,
        "mixed_layerwise_decode_delta": 0.0,
    }


def test_vllm_layerwise_mixed_ep_tiny_fresh_prefill_skips_saturated_floor(monkeypatch) -> None:
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
        _shared_expert_inter_size = 0

    class _Database:
        def query_layerwise_detail(self, model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache=0, **kwargs):
            assert model == "Qwen/Qwen3.6-35B-A3B"
            assert tp_size == 4
            if phase == "CTX" and batch_size == 1 and seq_len == 144 and seq_len_kv_cache == 0:
                return {
                    "latency": 12.0,
                    "energy": 0.0,
                    "latency_source": "schedule_to_update",
                    "measured_layer_count": 4,
                    "layer_multiplier": 4,
                    "includes_moe": True,
                    "max_num_batched_tokens": 2048,
                }
            if phase == "CTX" and batch_size == 1 and seq_len == 2048 and seq_len_kv_cache == 0:
                return {
                    "latency": 100.0,
                    "energy": 0.0,
                    "latency_source": "schedule_to_update",
                    "measured_layer_count": 4,
                    "layer_multiplier": 4,
                    "includes_moe": True,
                    "max_num_batched_tokens": 2048,
                }
            raise AssertionError((phase, batch_size, seq_len, seq_len_kv_cache))

    latency_ms, _energy_wms, per_ops, _sources = VLLMBackend()._get_mix_step_latency(
        _Model(),
        _Database(),
        RuntimeConfig(vllm_max_num_batched_tokens=2048),
        ctx_tokens=128,
        gen_tokens=16,
        isl=4096,
        osl=1,
        prefix=0,
    )

    assert latency_ms == pytest.approx(12.0)
    assert per_ops == {
        "mixed_layerwise_context_combined": 12.0,
        "mixed_layerwise_context_tp_allreduce": 0.0,
        "mixed_layerwise_decode_delta": 0.0,
    }


def test_vllm_layerwise_mixed_ep_tiny_fresh_prefill_skips_noop_moe_addback(monkeypatch) -> None:
    from aiconfigurator.sdk.backends import vllm_backend

    monkeypatch.setattr(vllm_backend, "_USE_LAYERWISE", True)

    class _Config:
        tp_size = 1
        pp_size = 1
        moe_tp_size = 1
        moe_ep_size = 4

    class _Extra:
        compress_ratios = (0, 4, 128)
        sliding_window = 128

    class _Model:
        model_path = "deepseek-ai/DeepSeek-V4-Flash"
        config = _Config()
        extra_params = _Extra()
        _num_layers = 1
        _hidden_size = 2048
        _num_experts = 256
        _topk = 6
        _shared_expert_inter_size = 2048

    class _Database:
        def query_layerwise_detail(self, model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache=0, **kwargs):
            assert model == "deepseek-ai/DeepSeek-V4-Flash"
            assert tp_size == 1
            if phase == "CTX" and batch_size == 1 and seq_len == 120 and seq_len_kv_cache == 0:
                return {
                    "latency": 27.0,
                    "energy": 0.0,
                    "latency_source": "schedule_to_update",
                    "measured_layer_count": 1,
                    "layer_multiplier": 1,
                    "includes_moe": False,
                    "moe_weight_mode": "noop",
                    "max_num_batched_tokens": 2048,
                }
            raise AssertionError((phase, batch_size, seq_len, seq_len_kv_cache))

        def query_custom_allreduce(self, *args, **kwargs):
            return 0.0

    backend = VLLMBackend()
    monkeypatch.setattr(
        backend,
        "_layerwise_noop_moe_addback",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("tiny fresh mixed row should use the scheduler envelope")
        ),
    )

    latency_ms, _energy_wms, per_ops, _sources = backend._get_mix_step_latency(
        _Model(),
        _Database(),
        RuntimeConfig(vllm_max_num_batched_tokens=2048),
        ctx_tokens=112,
        gen_tokens=8,
        isl=3132,
        osl=1,
        prefix=0,
    )

    assert latency_ms == pytest.approx(27.0 * 0.75)
    assert per_ops == {
        "mixed_layerwise_context_combined": pytest.approx(27.0 * 0.75),
        "mixed_layerwise_context_tp_allreduce": 0.0,
        "mixed_layerwise_decode_delta": 0.0,
    }


def test_vllm_layerwise_qwen_low_tp_small_fresh_moe_mixed_uses_saturated_floor(monkeypatch) -> None:
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
        _num_layers = 1
        _hidden_size = 2048
        _num_experts = 256
        _topk = 8

    class _Database:
        def query_layerwise_detail(self, model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache=0, **kwargs):
            assert model == "Qwen/Qwen3.6-35B-A3B"
            assert tp_size == 2
            assert phase == "CTX"
            assert batch_size == 1
            if seq_len == 673 and seq_len_kv_cache == 0:
                return {
                    "latency": 16.0,
                    "energy": 0.0,
                    "latency_source": "schedule_to_update",
                    "measured_layer_count": 1,
                    "layer_multiplier": 1,
                    "includes_moe": False,
                    "moe_weight_mode": "noop",
                    "max_num_batched_tokens": 2048,
                }
            if seq_len == 4096 and seq_len_kv_cache == 0:
                return {
                    "latency": 80.0,
                    "energy": 0.0,
                    "latency_source": "schedule_to_update",
                    "measured_layer_count": 1,
                    "layer_multiplier": 1,
                    "includes_moe": False,
                    "moe_weight_mode": "noop",
                    "max_num_batched_tokens": 2048,
                    "query_seq_len_q": 4096,
                }
            raise AssertionError((phase, batch_size, seq_len, seq_len_kv_cache))

        def query_custom_allreduce(self, *args, **kwargs):
            return 0.0

    backend = VLLMBackend()
    monkeypatch.setattr(
        backend,
        "_layerwise_noop_moe_addback",
        lambda *args, **kwargs: ((0.0, 0.0, "silicon"), (0.0, 0.0, "silicon"), (0.0, 0.0, "silicon"), False),
    )

    latency_ms, _energy_wms, per_ops, _sources = backend._get_mix_step_latency(
        _Model(),
        _Database(),
        RuntimeConfig(vllm_max_num_batched_tokens=2048),
        ctx_tokens=642,
        gen_tokens=31,
        isl=4352,
        osl=1,
        prefix=0,
    )

    assert latency_ms == pytest.approx(40.0)
    assert per_ops == {
        "mixed_layerwise_context_combined": 16.0,
        "mixed_layerwise_context_tp_allreduce": 0.0,
        "mixed_layerwise_decode_delta": 0.0,
        "mixed_layerwise_ep_high_decode_floor": 40.0,
    }


def test_vllm_layerwise_qwen_low_tp_fresh_moe_floor_uses_saturated_scheduler_row(monkeypatch) -> None:
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
        _num_layers = 1
        _hidden_size = 2048
        _num_experts = 256
        _topk = 8

    class _Database:
        def query_layerwise_detail(self, model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache=0, **kwargs):
            assert model == "Qwen/Qwen3.6-35B-A3B"
            assert tp_size == 2
            if phase == "CTX" and seq_len == 673 and seq_len_kv_cache == 0:
                assert batch_size == 1
                return {
                    "latency": 16.0,
                    "energy": 0.0,
                    "measured_layer_count": 1,
                    "layer_multiplier": 1,
                    "includes_moe": False,
                    "moe_weight_mode": "noop",
                }
            if phase == "CTX" and seq_len == 31 and seq_len_kv_cache == 0:
                assert batch_size == 1
                return {
                    "latency": 1.0,
                    "energy": 0.0,
                    "measured_layer_count": 1,
                    "layer_multiplier": 1,
                    "includes_moe": False,
                    "moe_weight_mode": "noop",
                }
            if phase == "CTX" and seq_len == 4096 and seq_len_kv_cache == 0:
                assert batch_size == 1
                return {
                    "latency": 80.0,
                    "energy": 0.0,
                    "latency_source": "schedule_to_update",
                    "measured_layer_count": 1,
                    "layer_multiplier": 1,
                    "includes_moe": False,
                    "moe_weight_mode": "noop",
                    "max_num_batched_tokens": 2048,
                    "query_seq_len_q": 4096,
                }
            if phase == "GEN" and batch_size == 31:
                return {
                    "latency": 1.0,
                    "energy": 0.0,
                    "measured_layer_count": 1,
                    "layer_multiplier": 1,
                    "includes_moe": False,
                }
            raise AssertionError((phase, batch_size, seq_len, seq_len_kv_cache))

        def query_custom_allreduce(self, *args, **kwargs):
            return 0.0

    backend = VLLMBackend()
    monkeypatch.setattr(
        backend,
        "_layerwise_noop_moe_addback",
        lambda *args, **kwargs: ((0.0, 0.0, "silicon"), (0.0, 0.0, "silicon"), (0.0, 0.0, "silicon"), False),
    )

    latency_ms, _energy_wms, per_ops, _sources = backend._get_mix_step_latency(
        _Model(),
        _Database(),
        RuntimeConfig(vllm_max_num_batched_tokens=2048),
        ctx_tokens=642,
        gen_tokens=31,
        isl=4352,
        osl=1,
        prefix=0,
    )

    assert latency_ms == pytest.approx(40.0)
    assert per_ops["mixed_layerwise_context_combined"] == pytest.approx(16.0)
    assert per_ops["mixed_layerwise_decode_delta"] == pytest.approx(0.0)
    assert per_ops["mixed_layerwise_ep_high_decode_floor"] == pytest.approx(40.0)


def test_vllm_layerwise_qwen_tp1_large_continuation_uses_partial_floor_for_low_kv(monkeypatch) -> None:
    from aiconfigurator.sdk.backends import vllm_backend

    monkeypatch.setattr(vllm_backend, "_USE_LAYERWISE", True)

    class _Config:
        tp_size = 1
        pp_size = 1
        moe_tp_size = 1
        moe_ep_size = 1

    class _Model:
        model_path = "Qwen/Qwen3.6-35B-A3B"
        config = _Config()
        _num_layers = 1
        _hidden_size = 2048
        _num_experts = 256
        _topk = 8
        _shared_expert_inter_size = 0

    def _ctx_detail(latency: float, seq_len: int = 4096) -> dict[str, object]:
        return {
            "latency": latency,
            "energy": 0.0,
            "latency_source": "schedule_to_update",
            "measured_layer_count": 1,
            "layer_multiplier": 1,
            "includes_moe": False,
            "moe_weight_mode": "noop",
            "physical_gpus": 1,
            "max_num_batched_tokens": 8192,
            "query_seq_len_q": min(seq_len, 8192),
        }

    backend = VLLMBackend()

    def context_for_runtime(_database, _model_name, _tp_size, _batch_size, seq_len, _prefix, *_args, **_kwargs):
        if seq_len == 4096:
            return _ctx_detail(80.0, seq_len)
        return _ctx_detail(20.0, seq_len)

    def context_step_detail(_database, _model_name, _tp_size, _batch_size, seq_len, _prefix, **_kwargs):
        if seq_len != 4096:
            raise AssertionError(seq_len)
        return _ctx_detail(80.0, seq_len)

    def query_layerwise_detail(_database, _model_name, phase, _tp_size, batch_size, _seq_len, *_args, **_kwargs):
        if phase != "GEN" or batch_size != 1:
            raise AssertionError((phase, batch_size))
        return {
            "latency": 5.0,
            "energy": 0.0,
            "measured_layer_count": 1,
            "layer_multiplier": 1,
            "includes_moe": False,
        }

    monkeypatch.setattr(backend, "_layerwise_context_detail_for_runtime", context_for_runtime)
    monkeypatch.setattr(backend, "_layerwise_context_step_detail", context_step_detail)
    monkeypatch.setattr(backend, "_query_layerwise_detail", query_layerwise_detail)
    monkeypatch.setattr(backend, "_layerwise_tp_allreduce_ms", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(
        backend,
        "_layerwise_noop_moe_addback",
        lambda *args, **kwargs: ((0.0, 0.0, "silicon"), (0.0, 0.0, "silicon"), (0.0, 0.0, "silicon"), False),
    )

    latency_ms, _energy_wms, per_ops, _sources = backend._get_mix_step_latency(
        _Model(),
        object(),
        RuntimeConfig(vllm_max_num_batched_tokens=2048),
        ctx_tokens=8191,
        ctx_requests=25,
        gen_tokens=1,
        isl=1025,
        osl=1,
        prefix=132,
    )

    assert latency_ms == pytest.approx(78.0)
    assert per_ops["mixed_layerwise_context_combined"] == pytest.approx(20.0)
    assert per_ops["mixed_layerwise_decode_delta"] == pytest.approx(10.0)
    assert per_ops["mixed_layerwise_ep_high_decode_floor"] == pytest.approx(78.0)

    monkeypatch.setattr(
        backend,
        "_layerwise_noop_moe_addback",
        lambda *args, **kwargs: ((12.0, 0.0, "silicon"), (1.0, 0.0, "silicon"), (0.0, 0.0, "silicon"), False),
    )
    latency_ms, _energy_wms, per_ops, _sources = backend._get_mix_step_latency(
        _Model(),
        object(),
        RuntimeConfig(vllm_max_num_batched_tokens=2048),
        ctx_tokens=8191,
        ctx_requests=25,
        gen_tokens=1,
        isl=1025,
        osl=1,
        prefix=132,
    )

    assert latency_ms == pytest.approx(78.0)
    assert per_ops["mixed_layerwise_ep_high_decode_floor"] == pytest.approx(78.0)
    assert "mixed_moe" not in per_ops
    assert "mixed_moe_router" not in per_ops


def test_vllm_layerwise_qwen_ep4_small_continuation_uses_high_ep_floor(monkeypatch) -> None:
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
        _num_layers = 1
        _hidden_size = 2048
        _num_experts = 256
        _topk = 8
        _shared_expert_inter_size = 0

    def _ctx_detail(latency: float, seq_len: int) -> dict[str, object]:
        return {
            "latency": latency,
            "energy": 0.0,
            "latency_source": "schedule_to_update",
            "measured_layer_count": 1,
            "layer_multiplier": 1,
            "includes_moe": False,
            "moe_weight_mode": "noop",
            "physical_gpus": 1,
            "max_num_batched_tokens": 2048,
            "query_seq_len_q": seq_len,
        }

    backend = VLLMBackend()

    def context_for_runtime(_database, _model_name, _tp_size, _batch_size, seq_len, _prefix, *_args, **_kwargs):
        return _ctx_detail(20.0, seq_len)

    def context_step_detail(_database, _model_name, _tp_size, _batch_size, seq_len, _prefix, **_kwargs):
        if seq_len != 4096:
            raise AssertionError(seq_len)
        return _ctx_detail(80.0, seq_len)

    def query_layerwise_detail(_database, _model_name, phase, _tp_size, batch_size, _seq_len, *_args, **_kwargs):
        if phase != "GEN" or batch_size != 30:
            raise AssertionError((phase, batch_size))
        return {
            "latency": 5.0,
            "energy": 0.0,
            "measured_layer_count": 1,
            "layer_multiplier": 1,
            "includes_moe": False,
        }

    monkeypatch.setattr(backend, "_layerwise_context_detail_for_runtime", context_for_runtime)
    monkeypatch.setattr(backend, "_layerwise_context_step_detail", context_step_detail)
    monkeypatch.setattr(backend, "_query_layerwise_detail", query_layerwise_detail)
    monkeypatch.setattr(backend, "_layerwise_tp_allreduce_ms", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(
        backend,
        "_layerwise_noop_moe_addback",
        lambda *args, **kwargs: ((0.0, 0.0, "silicon"), (0.0, 0.0, "silicon"), (0.0, 0.0, "silicon"), False),
    )

    latency_ms, _energy_wms, per_ops, _sources = backend._get_mix_step_latency(
        _Model(),
        object(),
        RuntimeConfig(vllm_max_num_batched_tokens=2048),
        ctx_tokens=864,
        ctx_requests=1,
        gen_tokens=30,
        isl=4937,
        osl=1,
        prefix=2640,
    )

    assert latency_ms == pytest.approx(60.0)
    assert per_ops["mixed_layerwise_context_combined"] == pytest.approx(20.0)
    assert per_ops["mixed_layerwise_ep_high_decode_floor"] == pytest.approx(60.0)


def test_vllm_layerwise_qwen_moetp_small_continuation_uses_decode_pressure_floor(monkeypatch) -> None:
    from aiconfigurator.sdk.backends import vllm_backend

    monkeypatch.setattr(vllm_backend, "_USE_LAYERWISE", True)

    class _Config:
        tp_size = 2
        pp_size = 1
        moe_tp_size = 2
        moe_ep_size = 1

    class _Model:
        model_path = "Qwen/Qwen3.6-35B-A3B"
        config = _Config()
        _num_layers = 1
        _hidden_size = 2048
        _num_experts = 256
        _topk = 8
        _shared_expert_inter_size = 0

    def _ctx_detail(latency: float, seq_len: int) -> dict[str, object]:
        return {
            "latency": latency,
            "energy": 0.0,
            "latency_source": "schedule_to_update",
            "measured_layer_count": 1,
            "layer_multiplier": 1,
            "includes_moe": False,
            "moe_weight_mode": "noop",
            "max_num_batched_tokens": 2048,
            "query_seq_len_q": seq_len,
        }

    backend = VLLMBackend()

    def context_for_runtime(_database, _model_name, _tp_size, _batch_size, seq_len, _prefix, *_args, **_kwargs):
        return _ctx_detail(20.0, seq_len)

    def context_step_detail(_database, _model_name, _tp_size, _batch_size, seq_len, _prefix, **_kwargs):
        if seq_len != 4096:
            raise AssertionError(seq_len)
        return _ctx_detail(80.0, seq_len)

    def query_layerwise_detail(_database, _model_name, phase, _tp_size, batch_size, _seq_len, *_args, **_kwargs):
        if phase != "GEN" or batch_size != 31:
            raise AssertionError((phase, batch_size))
        return {
            "latency": 5.0,
            "energy": 0.0,
            "measured_layer_count": 1,
            "layer_multiplier": 1,
            "includes_moe": False,
        }

    monkeypatch.setattr(backend, "_layerwise_context_detail_for_runtime", context_for_runtime)
    monkeypatch.setattr(backend, "_layerwise_context_step_detail", context_step_detail)
    monkeypatch.setattr(backend, "_query_layerwise_detail", query_layerwise_detail)
    monkeypatch.setattr(backend, "_layerwise_tp_allreduce_ms", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(
        backend,
        "_layerwise_noop_moe_addback",
        lambda *args, **kwargs: ((0.0, 0.0, "silicon"), (0.0, 0.0, "silicon"), (0.0, 0.0, "silicon"), False),
    )

    latency_ms, _energy_wms, per_ops, _sources = backend._get_mix_step_latency(
        _Model(),
        object(),
        RuntimeConfig(vllm_max_num_batched_tokens=2048),
        ctx_tokens=467,
        ctx_requests=1,
        gen_tokens=31,
        isl=7329,
        osl=1,
        prefix=0,
    )

    assert latency_ms == pytest.approx(20.0)
    assert "mixed_layerwise_ep_high_decode_floor" not in per_ops

    latency_ms, _energy_wms, per_ops, _sources = backend._get_mix_step_latency(
        _Model(),
        object(),
        RuntimeConfig(vllm_max_num_batched_tokens=2048),
        ctx_tokens=483,
        ctx_requests=1,
        gen_tokens=31,
        isl=6590,
        osl=1,
        prefix=2112,
    )

    assert latency_ms == pytest.approx(42.0)
    assert per_ops["mixed_layerwise_ep_high_decode_floor"] == pytest.approx(42.0)

    monkeypatch.setattr(
        backend,
        "_layerwise_noop_moe_addback",
        lambda *args, **kwargs: ((6.0, 0.0, "silicon"), (1.0, 0.0, "silicon"), (0.0, 0.0, "silicon"), False),
    )
    latency_ms, _energy_wms, per_ops, _sources = backend._get_mix_step_latency(
        _Model(),
        object(),
        RuntimeConfig(vllm_max_num_batched_tokens=2048),
        ctx_tokens=498,
        ctx_requests=1,
        gen_tokens=31,
        isl=7418,
        osl=1,
        prefix=0,
    )

    assert latency_ms == pytest.approx(40.0)
    assert per_ops["mixed_layerwise_ep_high_decode_floor"] == pytest.approx(40.0)
    assert "mixed_moe" not in per_ops
    assert "mixed_moe_router" not in per_ops


def test_vllm_layerwise_qwen_moetp_large_multi_request_lifts_floor(monkeypatch) -> None:
    from aiconfigurator.sdk.backends import vllm_backend

    monkeypatch.setattr(vllm_backend, "_USE_LAYERWISE", True)

    class _Config:
        tp_size = 2
        pp_size = 1
        moe_tp_size = 2
        moe_ep_size = 1

    class _Model:
        model_path = "Qwen/Qwen3.6-35B-A3B"
        config = _Config()
        _num_layers = 1
        _hidden_size = 2048
        _num_experts = 256
        _topk = 8
        _shared_expert_inter_size = 0

    def _ctx_detail(latency: float, seq_len: int) -> dict[str, object]:
        return {
            "latency": latency,
            "energy": 0.0,
            "latency_source": "schedule_to_update",
            "measured_layer_count": 1,
            "layer_multiplier": 1,
            "includes_moe": False,
            "moe_weight_mode": "noop",
            "physical_gpus": 1,
            "max_num_batched_tokens": 2048,
            "query_seq_len_q": min(seq_len, 2048),
        }

    backend = VLLMBackend()

    def context_for_runtime(_database, _model_name, _tp_size, _batch_size, seq_len, _prefix, *_args, **_kwargs):
        if seq_len != 4096:
            return _ctx_detail(20.0, seq_len)
        return _ctx_detail(40.0, seq_len)

    def context_step_detail(_database, _model_name, _tp_size, _batch_size, seq_len, _prefix, **_kwargs):
        if seq_len != 16384:
            raise AssertionError(seq_len)
        return _ctx_detail(100.0, seq_len)

    def query_layerwise_detail(_database, _model_name, phase, _tp_size, batch_size, _seq_len, *_args, **_kwargs):
        if phase != "GEN" or batch_size not in {4, 23}:
            raise AssertionError((phase, batch_size))
        return {
            "latency": 4.0,
            "energy": 0.0,
            "measured_layer_count": 1,
            "layer_multiplier": 1,
            "includes_moe": False,
        }

    monkeypatch.setattr(backend, "_layerwise_context_detail_for_runtime", context_for_runtime)
    monkeypatch.setattr(backend, "_layerwise_context_step_detail", context_step_detail)
    monkeypatch.setattr(backend, "_query_layerwise_detail", query_layerwise_detail)
    monkeypatch.setattr(backend, "_layerwise_tp_allreduce_ms", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(
        backend,
        "_layerwise_noop_moe_addback",
        lambda *args, **kwargs: ((0.0, 0.0, "silicon"), (0.0, 0.0, "silicon"), (0.0, 0.0, "silicon"), False),
    )

    latency_ms, _energy_wms, per_ops, _sources = backend._get_mix_step_latency(
        _Model(),
        object(),
        RuntimeConfig(vllm_max_num_batched_tokens=8192),
        ctx_tokens=7296,
        ctx_requests=5,
        gen_tokens=23,
        isl=2518,
        osl=1,
        prefix=211,
    )

    assert latency_ms == pytest.approx(100.0 * 0.53)
    assert per_ops["mixed_layerwise_ep_high_decode_floor"] == pytest.approx(100.0 * 0.53)


def test_vllm_layerwise_qwen_tp2_tiny_fresh_ep_moe_skips_saturated_floor(monkeypatch) -> None:
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
        _num_layers = 1
        _hidden_size = 2048
        _num_experts = 256
        _topk = 8

    class _Database:
        def query_layerwise_detail(self, model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache=0, **kwargs):
            assert model == "Qwen/Qwen3.6-35B-A3B"
            assert tp_size == 2
            assert phase == "CTX"
            assert batch_size == 1
            if seq_len == 131 and seq_len_kv_cache == 0:
                return {
                    "latency": 16.0,
                    "energy": 0.0,
                    "latency_source": "schedule_to_update",
                    "measured_layer_count": 1,
                    "layer_multiplier": 1,
                    "includes_moe": False,
                    "moe_weight_mode": "noop",
                    "max_num_batched_tokens": 2048,
                }
            if seq_len == 4096 and seq_len_kv_cache == 0:
                return {
                    "latency": 80.0,
                    "energy": 0.0,
                    "latency_source": "schedule_to_update",
                    "measured_layer_count": 1,
                    "layer_multiplier": 1,
                    "includes_moe": False,
                    "moe_weight_mode": "noop",
                    "max_num_batched_tokens": 2048,
                    "query_seq_len_q": 4096,
                }
            raise AssertionError((phase, batch_size, seq_len, seq_len_kv_cache))

        def query_custom_allreduce(self, *args, **kwargs):
            return 0.0

    backend = VLLMBackend()
    monkeypatch.setattr(
        backend,
        "_layerwise_noop_moe_addback",
        lambda *args, **kwargs: ((0.0, 0.0, "silicon"), (0.0, 0.0, "silicon"), (0.0, 0.0, "silicon"), False),
    )

    latency_ms, _energy_wms, per_ops, _sources = backend._get_mix_step_latency(
        _Model(),
        _Database(),
        RuntimeConfig(vllm_max_num_batched_tokens=2048),
        ctx_tokens=100,
        gen_tokens=31,
        isl=4096,
        osl=1,
        prefix=0,
    )

    assert latency_ms == pytest.approx(16.0)
    assert per_ops == {
        "mixed_layerwise_context_combined": 16.0,
        "mixed_layerwise_context_tp_allreduce": 0.0,
        "mixed_layerwise_decode_delta": 0.0,
    }


def test_vllm_layerwise_qwen_tp2_ep_small_continuation_uses_saturated_floor(monkeypatch) -> None:
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
        _num_layers = 1
        _hidden_size = 2048
        _num_experts = 256
        _topk = 8
        _shared_expert_inter_size = 0

    class _Database:
        def query_layerwise_detail(self, model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache=0, **kwargs):
            assert model == "Qwen/Qwen3.6-35B-A3B"
            assert tp_size == 2
            if phase == "CTX" and batch_size == 1:
                if (seq_len, seq_len_kv_cache) in {(644, 4224), (31, 4868)}:
                    return {
                        "latency": 16.0 if seq_len == 644 else 1.0,
                        "energy": 0.0,
                        "latency_source": "schedule_to_update",
                        "measured_layer_count": 1,
                        "layer_multiplier": 1,
                        "includes_moe": False,
                        "moe_weight_mode": "noop",
                        "max_num_batched_tokens": 2048,
                    }
                if seq_len == 4096 and seq_len_kv_cache == 0:
                    return {
                        "latency": 80.0,
                        "energy": 0.0,
                        "latency_source": "schedule_to_update",
                        "measured_layer_count": 1,
                        "layer_multiplier": 1,
                        "includes_moe": False,
                        "moe_weight_mode": "noop",
                        "max_num_batched_tokens": 2048,
                        "query_seq_len_q": 4096,
                    }
            if phase == "GEN" and batch_size == 4:
                return {
                    "latency": 1.0,
                    "energy": 0.0,
                    "measured_layer_count": 1,
                    "layer_multiplier": 1,
                    "includes_moe": False,
                }
            raise AssertionError((phase, batch_size, seq_len, seq_len_kv_cache))

        def query_custom_allreduce(self, *args, **kwargs):
            return 0.0

    backend = VLLMBackend()
    monkeypatch.setattr(
        backend,
        "_layerwise_noop_moe_addback",
        lambda *args, **kwargs: ((0.0, 0.0, "silicon"), (0.0, 0.0, "silicon"), (0.0, 0.0, "silicon"), False),
    )

    latency_ms, _energy_wms, per_ops, _sources = backend._get_mix_step_latency(
        _Model(),
        _Database(),
        RuntimeConfig(vllm_max_num_batched_tokens=2048),
        ctx_tokens=613,
        gen_tokens=31,
        isl=5450,
        osl=1,
        prefix=4224,
    )

    assert latency_ms == pytest.approx(40.0)
    assert per_ops["mixed_layerwise_context_combined"] == pytest.approx(16.0)
    assert per_ops["mixed_layerwise_decode_delta"] == pytest.approx(1.5)
    assert per_ops["mixed_layerwise_ep_high_decode_floor"] == pytest.approx(40.0)


def test_vllm_layerwise_mixed_ep_high_decode_fresh_prefill_skips_noop_moe_addback(monkeypatch) -> None:
    from aiconfigurator.sdk.backends import vllm_backend

    monkeypatch.setattr(vllm_backend, "_USE_LAYERWISE", True)

    class _Config:
        tp_size = 1
        pp_size = 1
        moe_tp_size = 1
        moe_ep_size = 4

    class _Extra:
        compress_ratios = (0, 4, 128)
        sliding_window = 128

    class _Model:
        model_path = "deepseek-ai/DeepSeek-V4-Flash"
        config = _Config()
        extra_params = _Extra()
        _num_layers = 1
        _hidden_size = 2048
        _num_experts = 256
        _topk = 8
        _shared_expert_inter_size = 2048

    class _Database:
        def query_layerwise_detail(self, model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache=0, **kwargs):
            assert model == "deepseek-ai/DeepSeek-V4-Flash"
            assert tp_size == 1
            if (
                phase == "CTX"
                and batch_size == 1
                and (seq_len, seq_len_kv_cache) in {(2048, 0), (2048, 2048), (8, 4096)}
            ):
                return {
                    "latency": 0.0 if seq_len == 8 else 56.0,
                    "energy": 0.0,
                    "latency_source": "schedule_to_update",
                    "measured_layer_count": 1,
                    "layer_multiplier": 1,
                    "includes_moe": False,
                    "moe_weight_mode": "noop",
                    "max_num_batched_tokens": 2048,
                }
            raise AssertionError((phase, batch_size, seq_len, seq_len_kv_cache))

        def query_custom_allreduce(self, *args, **kwargs):
            return 0.0

    backend = VLLMBackend()
    monkeypatch.setattr(
        backend,
        "_layerwise_noop_moe_addback",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("high-decode fresh EP mixed row should use the scheduler envelope")
        ),
    )

    latency_ms, _energy_wms, per_ops, _sources = backend._get_mix_step_latency(
        _Model(),
        _Database(),
        RuntimeConfig(vllm_max_num_batched_tokens=2048),
        ctx_tokens=4096,
        gen_tokens=8,
        isl=3946,
        osl=1,
        prefix=0,
    )

    assert latency_ms == pytest.approx(112.0)
    assert per_ops == {
        "mixed_layerwise_context_combined": 112.0,
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
        lambda self, model, database, distribution, *, token_count: distribution == "power_law_1.2",
    )

    class _Config:
        tp_size = 1
        pp_size = 1
        attention_dp_size = 1
        moe_tp_size = 1
        moe_ep_size = 2
        moe_quant_mode = common.MoEQuantMode.w4a8_mxfp4_mxfp8
        workload_distribution = "sampled_zipf_1.2"
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


def test_vllm_layerwise_context_dense_tp1_medium_scheduler_adds_small_overhead(monkeypatch) -> None:
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
            assert (model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache) == (
                "Qwen/Qwen3-32B",
                "CTX",
                1,
                1,
                1024,
                0,
            )
            return {
                "latency": 46.0,
                "energy": 0.0,
                "rms_latency": 0.0,
                "measured_layer_count": 4,
                "layer_multiplier": 4,
                "latency_source": "schedule_to_update",
                "physical_gpus": 1,
                "moe_weight_mode": "dense",
            }

    latency, energy, sources = VLLMBackend()._run_context_phase(
        _Model(),
        _Database(),
        RuntimeConfig(),
        batch_size=1,
        isl=1024,
        prefix=0,
    )

    assert latency == {
        "context_layerwise": pytest.approx(46.0),
        "context_tp_allreduce": 0.0,
        "context_scheduler_overhead": pytest.approx(2.0),
    }
    assert energy["context_scheduler_overhead"] == 0.0
    assert sources["context_scheduler_overhead"] == "silicon"


def test_vllm_layerwise_context_uses_geomean_rms_tail_for_tiny_dense_tp_allreduce(monkeypatch) -> None:
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

        def query_allreduce_rms(self, quant_mode, tp_size, size, hidden_size):
            del quant_mode
            assert tp_size == 2
            assert size == 128 * 5120
            assert hidden_size == 5120
            return 0.125

    latency, _, _ = VLLMBackend()._run_context_phase(
        _Model(),
        _Database(),
        RuntimeConfig(),
        batch_size=1,
        isl=128,
        prefix=0,
    )

    assert latency["context_layerwise"] == pytest.approx(17.0)
    assert latency["context_tp_allreduce"] == pytest.approx(1.0)


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


def test_vllm_layerwise_deepseek_high_ep_noop_context_uses_larger_envelope_floor(monkeypatch) -> None:
    from aiconfigurator.sdk.backends import vllm_backend

    monkeypatch.setattr(vllm_backend, "_USE_LAYERWISE", True)

    class _Config:
        tp_size = 8
        pp_size = 1
        attention_dp_size = 1
        moe_tp_size = 1
        moe_ep_size = 8
        moe_quant_mode = common.MoEQuantMode.w4a8_mxfp4_mxfp8
        workload_distribution = "power_law"
        moe_backend = None
        enable_eplb = False
        gemm_quant_mode = common.GEMMQuantMode.bfloat16

    class _Model:
        model_path = "deepseek-ai/DeepSeek-V4-Flash"
        config = _Config()
        _num_layers = 1
        _hidden_size = 2048
        _topk = 8
        _num_experts = 256
        _moe_inter_size = 512
        _nextn = 0

    class _Database:
        def __init__(self):
            self.layerwise_queries = []

        def query_layerwise_detail(
            self,
            model,
            phase,
            tp_size,
            batch_size,
            seq_len,
            seq_len_kv_cache=0,
            **kwargs,
        ):
            self.layerwise_queries.append((phase, seq_len, seq_len_kv_cache, kwargs.get("moe_ep_size")))
            assert model == "deepseek-ai/DeepSeek-V4-Flash"
            assert phase == "CTX"
            assert tp_size == 8
            assert batch_size == 1
            assert seq_len_kv_cache == 0
            latency_by_tokens = {128: 10.0, 1024: 40.0}
            assert seq_len in latency_by_tokens
            return {
                "latency": latency_by_tokens[seq_len],
                "energy": 0.0,
                "rms_latency": 0.0,
                "includes_moe": False,
                "moe_weight_mode": "noop",
                "measured_layer_count": 1,
                "layer_multiplier": 1,
                "latency_source": "schedule_to_update",
                "physical_gpus": 1,
            }

        def query_custom_allreduce(self, *args, **kwargs):
            return 0.0

    backend = VLLMBackend()
    monkeypatch.setattr(
        backend,
        "_layerwise_noop_moe_addback_for_context",
        lambda *args, **kwargs: ((0.0, 0.0, "silicon"), (0.0, 0.0, "silicon"), (0.0, 0.0, "silicon"), True),
    )
    database = _Database()

    latency, _energy, _sources = backend._run_context_phase(
        _Model(),
        database,
        RuntimeConfig(),
        batch_size=1,
        isl=128,
        prefix=0,
    )

    assert latency["context_layerwise"] == pytest.approx(34.0)
    assert database.layerwise_queries == [
        ("CTX", 128, 0, 8),
        ("CTX", 1024, 0, 8),
        ("CTX", 1024, 4096, 8),
    ]


def test_vllm_layerwise_noop_context_moetp_initial_floor_uses_lower_fraction() -> None:
    class _Database:
        def query_layerwise_detail(
            self,
            model,
            phase,
            tp_size,
            batch_size,
            seq_len,
            seq_len_kv_cache=0,
            **kwargs,
        ):
            assert (model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache) == (
                "Qwen/Qwen3.6-35B-A3B",
                "CTX",
                2,
                1,
                1024,
                4096,
            )
            assert kwargs["moe_tp_size"] == 2
            assert kwargs["moe_ep_size"] == 1
            return {
                "latency": 40.0,
                "latency_source": "schedule_to_update",
                "moe_weight_mode": "noop",
                "measured_layer_count": 1,
                "layer_multiplier": 1,
            }

    current_detail = {
        "latency_source": "schedule_to_update",
        "moe_weight_mode": "noop",
    }

    latency = VLLMBackend()._layerwise_noop_context_continuation_floor_ms(
        _Database(),
        "Qwen/Qwen3.6-35B-A3B",
        2,
        1,
        1024,
        0,
        RuntimeConfig(vllm_max_num_batched_tokens=2048),
        current_detail,
        20.0,
        1,
        moe_tp_size=2,
        moe_ep_size=1,
    )

    assert latency == pytest.approx(31.2)


def test_vllm_layerwise_deepseek_ep4_noop_context_uses_large_context_floor(monkeypatch) -> None:
    from aiconfigurator.sdk.backends import vllm_backend

    monkeypatch.setattr(vllm_backend, "_USE_LAYERWISE", True)

    class _Config:
        tp_size = 4
        pp_size = 1
        attention_dp_size = 1
        moe_tp_size = 1
        moe_ep_size = 4
        moe_quant_mode = common.MoEQuantMode.w4a8_mxfp4_mxfp8
        workload_distribution = "power_law"
        moe_backend = None
        enable_eplb = False
        gemm_quant_mode = common.GEMMQuantMode.bfloat16

    class _Model:
        model_path = "deepseek-ai/DeepSeek-V4-Flash"
        config = _Config()
        _num_layers = 1
        _hidden_size = 2048
        _topk = 8
        _num_experts = 256
        _moe_inter_size = 512
        _nextn = 0

    class _Database:
        def __init__(self):
            self.layerwise_queries = []

        def query_layerwise_detail(
            self,
            model,
            phase,
            tp_size,
            batch_size,
            seq_len,
            seq_len_kv_cache=0,
            **kwargs,
        ):
            self.layerwise_queries.append((phase, seq_len, seq_len_kv_cache, kwargs.get("moe_ep_size")))
            assert model == "deepseek-ai/DeepSeek-V4-Flash"
            assert phase == "CTX"
            assert tp_size == 4
            assert batch_size == 1
            assert seq_len_kv_cache == 0
            latency_by_tokens = {1024: 40.0, 4096: 120.0}
            assert seq_len in latency_by_tokens
            return {
                "latency": latency_by_tokens[seq_len],
                "energy": 0.0,
                "rms_latency": 0.0,
                "includes_moe": False,
                "moe_weight_mode": "noop",
                "measured_layer_count": 1,
                "layer_multiplier": 1,
                "latency_source": "schedule_to_update",
                "physical_gpus": 1,
            }

        def query_custom_allreduce(self, *args, **kwargs):
            return 0.0

    backend = VLLMBackend()
    monkeypatch.setattr(
        backend,
        "_layerwise_noop_moe_addback_for_context",
        lambda *args, **kwargs: ((0.0, 0.0, "silicon"), (0.0, 0.0, "silicon"), (0.0, 0.0, "silicon"), True),
    )
    database = _Database()

    latency, _energy, _sources = backend._run_context_phase(
        _Model(),
        database,
        RuntimeConfig(),
        batch_size=1,
        isl=1024,
        prefix=0,
    )

    assert latency["context_layerwise"] == pytest.approx(90.0)
    assert database.layerwise_queries == [
        ("CTX", 1024, 0, 4),
        ("CTX", 1024, 4096, 4),
        ("CTX", 4096, 0, 4),
        ("CTX", 4096, 4096, 4),
    ]


def test_vllm_layerwise_deepseek_ep1_mid_context_uses_large_context_floor(monkeypatch) -> None:
    from aiconfigurator.sdk.backends import vllm_backend

    monkeypatch.setattr(vllm_backend, "_USE_LAYERWISE", True)

    class _Config:
        tp_size = 4
        pp_size = 1
        attention_dp_size = 1
        moe_tp_size = 4
        moe_ep_size = 1
        moe_quant_mode = common.MoEQuantMode.w4a8_mxfp4_mxfp8
        workload_distribution = "power_law"
        moe_backend = None
        enable_eplb = False
        gemm_quant_mode = common.GEMMQuantMode.bfloat16

    class _Model:
        model_path = "deepseek-ai/DeepSeek-V4-Flash"
        config = _Config()
        _num_layers = 1
        _hidden_size = 2048
        _topk = 8
        _num_experts = 256
        _moe_inter_size = 512
        _nextn = 0

    class _Database:
        def __init__(self):
            self.layerwise_queries = []

        def query_layerwise_detail(
            self,
            model,
            phase,
            tp_size,
            batch_size,
            seq_len,
            seq_len_kv_cache=0,
            **kwargs,
        ):
            self.layerwise_queries.append((phase, seq_len, seq_len_kv_cache, kwargs.get("moe_ep_size")))
            assert model == "deepseek-ai/DeepSeek-V4-Flash"
            assert phase == "CTX"
            assert tp_size == 4
            assert batch_size == 1
            assert seq_len_kv_cache == 0
            latency_by_tokens = {1024: 40.0, 4096: 120.0}
            assert seq_len in latency_by_tokens
            return {
                "latency": latency_by_tokens[seq_len],
                "energy": 0.0,
                "rms_latency": 0.0,
                "includes_moe": False,
                "moe_weight_mode": "noop",
                "measured_layer_count": 1,
                "layer_multiplier": 1,
                "latency_source": "schedule_to_update",
                "physical_gpus": 1,
            }

        def query_custom_allreduce(self, *args, **kwargs):
            return 0.0

    backend = VLLMBackend()
    monkeypatch.setattr(
        backend,
        "_layerwise_noop_moe_addback_for_context",
        lambda *args, **kwargs: ((0.0, 0.0, "silicon"), (0.0, 0.0, "silicon"), (0.0, 0.0, "silicon"), True),
    )
    database = _Database()

    latency, _energy, _sources = backend._run_context_phase(
        _Model(),
        database,
        RuntimeConfig(),
        batch_size=1,
        isl=1024,
        prefix=0,
    )

    assert latency["context_layerwise"] == pytest.approx(78.0)
    assert database.layerwise_queries == [
        ("CTX", 1024, 0, 1),
        ("CTX", 1024, 4096, 1),
        ("CTX", 4096, 0, 1),
        ("CTX", 4096, 4096, 1),
    ]


def test_vllm_layerwise_deepseek_high_ep_context_addback_uses_direct_shape(monkeypatch) -> None:
    class _Config:
        moe_ep_size = 4

    class _Model:
        model_path = "deepseek-ai/DeepSeek-V4-Flash"
        config = _Config()

    backend = VLLMBackend()
    calls = []

    def _fake_addback(*args, **kwargs):
        calls.append(kwargs["token_count"])
        return (
            (float(kwargs["token_count"]), 0.0, "silicon"),
            (0.0, 0.0, "silicon"),
            (0.0, 0.0, "silicon"),
            True,
        )

    monkeypatch.setattr(backend, "_layerwise_noop_moe_addback", _fake_addback)

    pieces = backend._layerwise_noop_moe_addback_for_context(
        _Model(),
        object(),
        token_count=4096,
        num_layers=1,
        runtime_config=RuntimeConfig(vllm_max_num_batched_tokens=2048),
        deepseek_context_sum_shared=True,
    )

    assert pieces[0][0] == pytest.approx(4096.0)
    assert calls == [4096]


def test_vllm_layerwise_deepseek_ep1_low_moetp_context_addback_chunks_at_1024(monkeypatch) -> None:
    class _Config:
        moe_ep_size = 1
        moe_tp_size = 2

    class _Model:
        model_path = "deepseek-ai/DeepSeek-V4-Flash"
        config = _Config()

    backend = VLLMBackend()
    calls = []

    def _fake_addback(*args, **kwargs):
        calls.append(kwargs["token_count"])
        return (
            (float(kwargs["token_count"]), 0.0, "silicon"),
            (0.0, 0.0, "silicon"),
            (0.0, 0.0, "silicon"),
            True,
        )

    monkeypatch.setattr(backend, "_layerwise_noop_moe_addback", _fake_addback)

    pieces = backend._layerwise_noop_moe_addback_for_context(
        _Model(),
        object(),
        token_count=4096,
        num_layers=1,
        runtime_config=RuntimeConfig(vllm_max_num_batched_tokens=2048),
        deepseek_context_sum_shared=True,
    )

    assert pieces[0][0] == pytest.approx(4096.0)
    assert calls == [1024, 1024, 1024, 1024]


def test_vllm_layerwise_deepseek_large_context_moe_overhead_is_scoped() -> None:
    class _Config:
        moe_tp_size = 4
        moe_ep_size = 1

    class _Model:
        model_path = "deepseek-ai/DeepSeek-V4-Flash"
        config = _Config()

    backend = VLLMBackend()

    assert backend._layerwise_deepseek_large_context_moe_overhead_ms(
        _Model(),
        ctx_tokens=4096,
        moe_ms=40.0,
    ) == pytest.approx(14.0)

    backend._layerwise_deepseek_context_floor_active = True
    assert backend._layerwise_deepseek_large_context_moe_overhead_ms(
        _Model(),
        ctx_tokens=4096,
        moe_ms=40.0,
    ) == 0.0
    backend._layerwise_deepseek_context_floor_active = False

    backend._layerwise_mixed_context_floor_active = True
    assert backend._layerwise_deepseek_large_context_moe_overhead_ms(
        _Model(),
        ctx_tokens=4096,
        moe_ms=40.0,
    ) == 0.0
    backend._layerwise_mixed_context_floor_active = False

    assert backend._layerwise_deepseek_large_context_moe_overhead_ms(
        _Model(),
        ctx_tokens=1024,
        moe_ms=40.0,
    ) == 0.0

    _Config.moe_tp_size = 2
    _Config.moe_ep_size = 1
    assert backend._layerwise_deepseek_large_context_moe_overhead_ms(
        _Model(),
        ctx_tokens=4096,
        moe_ms=40.0,
    ) == 0.0

    _Config.moe_tp_size = 1
    _Config.moe_ep_size = 4
    assert backend._layerwise_deepseek_large_context_moe_overhead_ms(
        _Model(),
        ctx_tokens=4096,
        moe_ms=40.0,
    ) == 0.0

    _Config.moe_tp_size = 1
    _Config.moe_ep_size = 2
    assert backend._layerwise_deepseek_large_context_moe_overhead_ms(
        _Model(),
        ctx_tokens=4096,
        moe_ms=40.0,
    ) == pytest.approx(14.0)


def test_vllm_layerwise_deepseek_decode_ep_moe_overhead_is_scoped() -> None:
    class _Config:
        moe_ep_size = 2

    class _Model:
        model_path = "deepseek-ai/DeepSeek-V4-Flash"
        config = _Config()

    backend = VLLMBackend()

    assert backend._layerwise_deepseek_decode_ep_moe_overhead_ms(
        _Model(),
        moe_ms=2.0,
    ) == pytest.approx(1.0)

    _Config.moe_ep_size = 1
    assert backend._layerwise_deepseek_decode_ep_moe_overhead_ms(
        _Model(),
        moe_ms=2.0,
    ) == 0.0


def test_vllm_layerwise_deepseek_generation_smooths_isolated_tp_outlier(monkeypatch) -> None:
    backend = VLLMBackend()
    current = {
        "latency": 9.5,
        "latency_source": "schedule_to_update",
        "moe_weight_mode": "noop",
        "measured_layer_count": 1,
        "layer_multiplier": 1,
    }
    neighbors = {
        2: {
            "latency": 7.8,
            "latency_source": "schedule_to_update",
            "moe_weight_mode": "noop",
            "measured_layer_count": 1,
            "layer_multiplier": 1,
        },
        8: {
            "latency": 7.7,
            "latency_source": "schedule_to_update",
            "moe_weight_mode": "noop",
            "measured_layer_count": 1,
            "layer_multiplier": 1,
        },
    }

    def _fake_query(database, model_name, phase, tp_size, batch_size, seq_len, **kwargs):
        del database, model_name, phase, batch_size, seq_len, kwargs
        return neighbors[tp_size]

    monkeypatch.setattr(backend, "_query_layerwise_detail", _fake_query)

    smoothed = backend._layerwise_smoothed_deepseek_generation_detail(
        object(),
        "deepseek-ai/DeepSeek-V4-Flash",
        tp_size=4,
        batch_size=1,
        past_kv=4096,
        num_layers=1,
        layer_detail=current,
        moe_tp_size=4,
        moe_ep_size=1,
    )

    assert smoothed["latency"] == pytest.approx(7.75)
    assert smoothed["diagnostic_smoothed_from_latency"] == pytest.approx(9.5)


def test_vllm_layerwise_noop_small_context_smooths_parallelism_outlier(monkeypatch) -> None:
    backend = VLLMBackend()
    current = {
        "latency": 20.0,
        "latency_source": "schedule_to_update",
        "moe_weight_mode": "noop",
        "measured_layer_count": 1,
        "layer_multiplier": 1,
    }
    neighbor = {
        "latency": 16.0,
        "latency_source": "schedule_to_update",
        "moe_weight_mode": "noop",
        "measured_layer_count": 1,
        "layer_multiplier": 1,
    }

    def _fake_context_step_detail(*args, **kwargs):
        del args, kwargs
        return neighbor

    monkeypatch.setattr(backend, "_layerwise_context_step_detail", _fake_context_step_detail)

    smoothed = backend._layerwise_smoothed_noop_small_context_detail(
        object(),
        "Qwen/Qwen3.6-35B-A3B",
        tp_size=8,
        batch_size=1,
        seq_len=128,
        prefix=0,
        num_layers=1,
        layer_detail=current,
        moe_tp_size=1,
        moe_ep_size=8,
    )

    assert smoothed["latency"] == pytest.approx(16.0)
    assert smoothed["diagnostic_smoothed_from_latency"] == pytest.approx(20.0)


def test_vllm_layerwise_deepseek_tp1_small_context_overhead_is_scoped() -> None:
    class _Config:
        tp_size = 1

    class _Model:
        model_path = "deepseek-ai/DeepSeek-V4-Flash"
        config = _Config()
        _shared_expert_inter_size = 2048

    detail = {
        "latency_source": "schedule_to_update",
        "moe_weight_mode": "noop",
    }
    backend = VLLMBackend()

    assert backend._layerwise_deepseek_tp1_small_context_overhead_ms(
        _Model(),
        ctx_tokens=128,
        ctx_kv_tokens=0,
        ctx_requests=1,
        layer_detail=detail,
        layer_ms=28.0,
        layer_includes_moe=False,
    ) == pytest.approx(3.64)

    _Config.tp_size = 2
    assert backend._layerwise_deepseek_tp1_small_context_overhead_ms(
        _Model(),
        ctx_tokens=128,
        ctx_kv_tokens=0,
        ctx_requests=1,
        layer_detail=detail,
        layer_ms=28.0,
        layer_includes_moe=False,
    ) == 0.0


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


def test_vllm_layerwise_context_interpolated_noop_envelope_skips_moe_addback(monkeypatch) -> None:
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
                928,
                3168,
            )
            return {
                "latency": 10.0,
                "energy": 0.0,
                "rms_latency": 0.0,
                "includes_moe": False,
                "measured_layer_count": 1,
                "layer_multiplier": 1,
                "latency_source": "schedule_to_update",
                "physical_gpus": 1,
                "components": [
                    {
                        "latency": 4.0,
                        "includes_moe": False,
                        "moe_weight_mode": "noop",
                        "measured_layer_count": 4,
                        "layer_multiplier": 4,
                        "latency_source": "schedule_to_update",
                    }
                ],
            }

        def query_custom_allreduce(self, quant_mode, tp_size, size, execution_mode=None):
            del quant_mode
            assert tp_size == 2
            assert size == 928 * 2048
            assert execution_mode == "eager"
            return 0.25

        def query_moe(self, **kwargs):
            raise AssertionError(f"full no-op scheduler envelope should not add MoE compute: {kwargs}")

        def query_gemm(self, *args, **kwargs):
            raise AssertionError("full no-op scheduler envelope should not add router/shared GEMMs")

    latency, _energy, _sources = VLLMBackend()._run_context_phase(
        _Model(),
        _Database(),
        RuntimeConfig(),
        batch_size=1,
        isl=4096,
        prefix=3168,
    )

    assert latency["context_layerwise"] == pytest.approx(10.0)
    assert latency["context_tp_allreduce"] == pytest.approx(2.0)
    assert "context_moe" not in latency
    assert "context_moe_router" not in latency
    assert "context_moe_tp_allreduce" not in latency


def test_vllm_layerwise_context_noop_continuation_uses_long_prefix_floor(monkeypatch) -> None:
    from aiconfigurator.sdk.backends import vllm_backend

    monkeypatch.setattr(vllm_backend, "_USE_LAYERWISE", True)

    class _Config:
        tp_size = 4
        pp_size = 1
        attention_dp_size = 1
        moe_tp_size = 1
        moe_ep_size = 4
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
        def query_layerwise_detail(self, model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache=0, **kwargs):
            del kwargs
            assert (model, phase, tp_size, batch_size, seq_len) == (
                "Qwen/Qwen3.6-35B-A3B",
                "CTX",
                4,
                1,
                496,
            )
            if seq_len_kv_cache == 528:
                return {
                    "latency": 5.0,
                    "energy": 0.0,
                    "includes_moe": False,
                    "measured_layer_count": 1,
                    "layer_multiplier": 1,
                    "latency_source": "schedule_to_update",
                    "moe_weight_mode": "noop",
                }
            if seq_len_kv_cache == 4096:
                return {
                    "latency": 12.0,
                    "energy": 0.0,
                    "includes_moe": False,
                    "measured_layer_count": 1,
                    "layer_multiplier": 1,
                    "latency_source": "schedule_to_update",
                    "moe_weight_mode": "noop",
                }
            raise AssertionError(seq_len_kv_cache)

        def query_custom_allreduce(self, quant_mode, tp_size, size, execution_mode=None):
            del quant_mode
            assert tp_size == 4
            assert size == 496 * 2048
            assert execution_mode == "eager"
            return 0.1

        def query_moe(self, **kwargs):
            raise AssertionError(f"full no-op scheduler envelope should not add MoE compute: {kwargs}")

        def query_gemm(self, *args, **kwargs):
            raise AssertionError("full no-op scheduler envelope should not add router/shared GEMMs")

    latency, _energy, _sources = VLLMBackend()._run_context_phase(
        _Model(),
        _Database(),
        RuntimeConfig(vllm_max_num_batched_tokens=2048),
        batch_size=1,
        isl=1024,
        prefix=528,
    )

    assert latency["context_layerwise"] == pytest.approx(10.8)
    assert latency["context_tp_allreduce"] == pytest.approx(0.8)
    assert "context_moe" not in latency


def test_vllm_layerwise_context_noop_fresh_medium_prefill_uses_long_prefix_floor(monkeypatch) -> None:
    from aiconfigurator.sdk.backends import vllm_backend

    monkeypatch.setattr(vllm_backend, "_USE_LAYERWISE", True)

    def _no_moe_addback(self, *args, **kwargs):
        del self, args, kwargs
        zero = (0.0, 0.0, "silicon")
        return zero, zero, zero, False

    monkeypatch.setattr(VLLMBackend, "_layerwise_noop_moe_addback_for_context", _no_moe_addback)

    class _Config:
        tp_size = 2
        pp_size = 1
        attention_dp_size = 1
        moe_tp_size = 1
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
        def query_layerwise_detail(self, model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache=0, **kwargs):
            del kwargs
            assert (model, phase, tp_size, batch_size, seq_len) == (
                "Qwen/Qwen3.6-35B-A3B",
                "CTX",
                2,
                1,
                1024,
            )
            if seq_len_kv_cache == 0:
                return {
                    "latency": 5.0,
                    "energy": 0.0,
                    "includes_moe": False,
                    "measured_layer_count": 1,
                    "layer_multiplier": 1,
                    "latency_source": "schedule_to_update",
                    "moe_weight_mode": "noop",
                }
            if seq_len_kv_cache == 4096:
                return {
                    "latency": 12.0,
                    "energy": 0.0,
                    "includes_moe": False,
                    "measured_layer_count": 1,
                    "layer_multiplier": 1,
                    "latency_source": "schedule_to_update",
                    "moe_weight_mode": "noop",
                }
            raise AssertionError(seq_len_kv_cache)

        def query_custom_allreduce(self, quant_mode, tp_size, size, execution_mode=None):
            del quant_mode, size
            assert tp_size == 2
            assert execution_mode == "eager"
            return 0.0

    latency, _energy, _sources = VLLMBackend()._run_context_phase(
        _Model(),
        _Database(),
        RuntimeConfig(vllm_max_num_batched_tokens=2048),
        batch_size=1,
        isl=1024,
        prefix=0,
    )

    assert latency["context_layerwise"] == pytest.approx(10.8)
    assert "context_moe" not in latency


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


def test_vllm_layerwise_deepseek_low_moetp_context_step_chunks_noop_moe_addback(monkeypatch) -> None:
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

        def query_layerwise_detail(self, model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache=0, **kwargs):
            query = (model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache)
            self.layerwise_queries.append(query)
            assert query in {
                ("deepseek-ai/DeepSeek-V4-Flash", "CTX", 1, 1, 2048, 0),
                ("deepseek-ai/DeepSeek-V4-Flash", "CTX", 1, 1, 2048, 2048),
            }
            return {
                "latency": 1.5,
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
                1024,
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
    assert database.moe_token_counts == [1024, 1024, 1024, 1024]
    assert database.router_token_counts == [1024, 1024, 1024, 1024]
    assert latency["context_layerwise"] == pytest.approx(3.0)
    assert latency["context_moe"] == pytest.approx(4.8)
    assert energy["context_moe"] == pytest.approx(19.2)
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


def test_vllm_layerwise_decode_full_depth_scheduler_noop_moe_skips_addback(monkeypatch) -> None:
    from aiconfigurator.sdk.backends import vllm_backend

    monkeypatch.setattr(vllm_backend, "_USE_LAYERWISE", True)

    class _Config:
        tp_size = 1
        pp_size = 1
        attention_dp_size = 1
        moe_tp_size = 1
        moe_ep_size = 1
        moe_quant_mode = common.MoEQuantMode.nvfp4
        workload_distribution = "balanced"
        moe_backend = None
        enable_eplb = False

    class _Model:
        model_path = "Qwen/Qwen3.6-35B-A3B"
        config = _Config()
        _num_layers = 4
        _hidden_size = 2880
        _topk = 8
        _num_experts = 128
        _moe_inter_size = 2880
        _nextn = 0

    class _Database:
        def query_layerwise_detail(self, model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache=0, **kwargs):
            assert (model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache) == (
                "Qwen/Qwen3.6-35B-A3B",
                "GEN",
                1,
                1,
                4096,
                0,
            )
            assert kwargs == {
                "moe_weight_mode": None,
                "max_num_batched_tokens": None,
                "moe_tp_size": 1,
                "moe_ep_size": 1,
            }
            return {
                "latency": 5.0,
                "energy": 0.0,
                "rms_latency": 0.0,
                "latency_source": "schedule_to_update",
                "measured_layer_count": 4,
                "layer_multiplier": 4,
            }

        def query_moe(self, **kwargs):
            raise AssertionError(f"full-depth scheduler envelope should not add MoE compute: {kwargs}")

        def query_gemm(self, *args, **kwargs):
            raise AssertionError("full-depth scheduler envelope should not add router GEMM")

    latency, energy, sources = VLLMBackend()._get_decode_step_latency(
        _Model(),
        _Database(),
        RuntimeConfig(),
        batch_size=1,
        past_kv=4096,
    )

    assert latency == {"generation_layerwise": pytest.approx(5.0)}
    assert energy["generation_layerwise"] == 0.0
    assert sources == {"generation_layerwise": "silicon"}


def test_vllm_layerwise_decode_dense_scheduler_skips_tp_allreduce(monkeypatch) -> None:
    from aiconfigurator.sdk.backends import vllm_backend

    monkeypatch.setattr(vllm_backend, "_USE_LAYERWISE", True)

    class _Config:
        tp_size = 8
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
        def query_layerwise_detail(self, model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache=0, **kwargs):
            assert (model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache) == (
                "Qwen/Qwen3-32B",
                "GEN",
                8,
                1,
                4096,
                0,
            )
            assert kwargs == {
                "moe_weight_mode": None,
                "max_num_batched_tokens": None,
                "moe_tp_size": 1,
                "moe_ep_size": 1,
            }
            return {
                "latency": 5.0,
                "energy": 0.0,
                "rms_latency": 0.0,
                "latency_source": "schedule_to_update",
                "measured_layer_count": 1,
                "layer_multiplier": 1,
                "physical_gpus": 1,
                "moe_weight_mode": "dense",
            }

        def query_custom_allreduce(self, quant_mode, tp_size, size):
            raise AssertionError("dense scheduler-envelope decode should not add standalone TP allreduce")

        def query_allreduce_rms(self, quant_mode, tp_size, size, hidden_size):
            raise AssertionError("dense scheduler-envelope decode should not add standalone allreduce+RMS")

    latency, energy, sources = VLLMBackend()._get_decode_step_latency(
        _Model(),
        _Database(),
        RuntimeConfig(),
        batch_size=1,
        past_kv=4096,
    )

    assert latency == {"generation_layerwise": pytest.approx(5.0)}
    assert energy["generation_layerwise"] == 0.0
    assert sources == {"generation_layerwise": "silicon"}


def test_vllm_layerwise_decode_noop_moe_scheduler_adds_tp_allreduce_for_simulated_tp(monkeypatch) -> None:
    from aiconfigurator.sdk.backends import vllm_backend

    monkeypatch.setattr(vllm_backend, "_USE_LAYERWISE", True)

    class _Config:
        tp_size = 4
        pp_size = 1
        attention_dp_size = 1
        moe_tp_size = 4
        moe_ep_size = 1
        moe_quant_mode = common.MoEQuantMode.nvfp4
        workload_distribution = "balanced"
        moe_backend = None
        enable_eplb = False

    class _Model:
        model_path = "Qwen/Qwen3.6-35B-A3B"
        config = _Config()
        _num_layers = 4
        _hidden_size = 2880
        _topk = 8
        _num_experts = 128
        _moe_inter_size = 2880
        _nextn = 0

    class _Database:
        def query_layerwise_detail(self, model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache=0, **kwargs):
            assert (model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache) == (
                "Qwen/Qwen3.6-35B-A3B",
                "GEN",
                4,
                1,
                4096,
                0,
            )
            assert kwargs == {
                "moe_weight_mode": None,
                "max_num_batched_tokens": None,
                "moe_tp_size": 4,
                "moe_ep_size": 1,
            }
            return {
                "latency": 5.0,
                "energy": 0.0,
                "rms_latency": 0.0,
                "latency_source": "schedule_to_update",
                "measured_layer_count": 4,
                "layer_multiplier": 4,
                "physical_gpus": 1,
                "moe_weight_mode": "noop",
            }

        def query_custom_allreduce(self, quant_mode, tp_size, size):
            del quant_mode
            assert tp_size == 4
            assert size == 2880
            return 0.3

        def query_allreduce_rms(self, quant_mode, tp_size, size, hidden_size):
            del quant_mode
            assert tp_size == 4
            assert size == 2880
            assert hidden_size == 2880
            return 0.2

        def query_moe(self, **kwargs):
            raise AssertionError(f"full-depth scheduler envelope should not add MoE compute: {kwargs}")

        def query_gemm(self, *args, **kwargs):
            raise AssertionError("full-depth scheduler envelope should not add router GEMM")

    latency, energy, sources = VLLMBackend()._get_decode_step_latency(
        _Model(),
        _Database(),
        RuntimeConfig(),
        batch_size=1,
        past_kv=4096,
    )

    assert latency == {
        "generation_layerwise": pytest.approx(5.0),
        "generation_moe_scheduler_residual": pytest.approx(1.9),
        "generation_tp_allreduce": pytest.approx((0.3 + 0.2) * 4),
    }
    assert energy["generation_layerwise"] == 0.0
    assert energy["generation_moe_scheduler_residual"] == 0.0
    assert energy["generation_tp_allreduce"] == 0.0
    assert sources == {
        "generation_layerwise": "silicon",
        "generation_moe_scheduler_residual": "silicon",
        "generation_tp_allreduce": "silicon",
    }


def test_vllm_layerwise_decode_qwen_ep_noop_scheduler_adds_visible_residual(monkeypatch) -> None:
    from aiconfigurator.sdk.backends import vllm_backend

    monkeypatch.setattr(vllm_backend, "_USE_LAYERWISE", True)

    class _Config:
        tp_size = 2
        pp_size = 1
        attention_dp_size = 1
        moe_tp_size = 1
        moe_ep_size = 4
        moe_quant_mode = common.MoEQuantMode.bfloat16
        workload_distribution = "balanced"
        moe_backend = None
        enable_eplb = False

    class _Model:
        model_path = "Qwen/Qwen3.6-35B-A3B"
        config = _Config()
        _num_layers = 4
        _hidden_size = 2048
        _topk = 8
        _num_experts = 256
        _moe_inter_size = 512
        _shared_expert_inter_size = 0
        _nextn = 0

    class _Database:
        def query_layerwise_detail(self, model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache=0, **kwargs):
            assert (model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache) == (
                "Qwen/Qwen3.6-35B-A3B",
                "GEN",
                2,
                1,
                4096,
                0,
            )
            return {
                "latency": 5.0,
                "energy": 0.0,
                "rms_latency": 0.0,
                "latency_source": "schedule_to_update",
                "measured_layer_count": 4,
                "layer_multiplier": 4,
                "physical_gpus": 1,
                "moe_weight_mode": "noop",
            }

        def query_custom_allreduce(self, quant_mode, tp_size, size):
            del quant_mode
            assert tp_size == 2
            assert size == 2048
            return 0.3

        def query_allreduce_rms(self, quant_mode, tp_size, size, hidden_size):
            del quant_mode
            assert tp_size == 2
            assert size == 2048
            assert hidden_size == 2048
            return 0.2

        def query_moe(self, **kwargs):
            raise AssertionError(f"Qwen no-op scheduler residual should not query MoE table: {kwargs}")

    latency, energy, sources = VLLMBackend()._get_decode_step_latency(
        _Model(),
        _Database(),
        RuntimeConfig(),
        batch_size=1,
        past_kv=4096,
    )

    assert latency["generation_layerwise"] == pytest.approx(5.0)
    assert latency["generation_tp_allreduce"] == pytest.approx((0.3 + 0.2) * 4)
    assert latency["generation_moe_scheduler_residual"] == pytest.approx(1.9)
    assert energy["generation_tp_allreduce"] == 0.0
    assert energy["generation_moe_scheduler_residual"] == 0.0
    assert sources["generation_tp_allreduce"] == "silicon"
    assert sources["generation_moe_scheduler_residual"] == "silicon"

    _Config.moe_tp_size = 4
    _Config.moe_ep_size = 1
    latency, _energy, _sources = VLLMBackend()._get_decode_step_latency(
        _Model(),
        _Database(),
        RuntimeConfig(),
        batch_size=1,
        past_kv=4096,
    )
    assert latency["generation_moe_scheduler_residual"] == pytest.approx(1.9)

    _Config.moe_tp_size = 1
    _Config.moe_ep_size = 8
    latency, _energy, _sources = VLLMBackend()._get_decode_step_latency(
        _Model(),
        _Database(),
        RuntimeConfig(),
        batch_size=1,
        past_kv=4096,
    )
    assert latency["generation_moe_scheduler_residual"] == pytest.approx(1.9)


def test_vllm_layerwise_context_qwen_ep_noop_scheduler_adds_visible_residual(monkeypatch) -> None:
    from aiconfigurator.sdk.backends import vllm_backend

    monkeypatch.setattr(vllm_backend, "_USE_LAYERWISE", True)

    class _Config:
        tp_size = 2
        pp_size = 1
        attention_dp_size = 1
        moe_tp_size = 1
        moe_ep_size = 2
        moe_quant_mode = common.MoEQuantMode.bfloat16
        workload_distribution = "balanced"
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
        _shared_expert_inter_size = 0

    class _Database:
        def query_layerwise_detail(self, model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache=0, **kwargs):
            assert (model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache) == (
                "Qwen/Qwen3.6-35B-A3B",
                "CTX",
                2,
                1,
                1024,
                0,
            )
            return {
                "latency": 5.0,
                "energy": 0.0,
                "rms_latency": 0.0,
                "latency_source": "schedule_to_update",
                "measured_layer_count": 4,
                "layer_multiplier": 4,
                "physical_gpus": 1,
                "moe_weight_mode": "noop",
            }

        def query_custom_allreduce(self, *args, **kwargs):
            return 0.0

        def query_moe(self, **kwargs):
            raise PerfDataNotAvailableError("no Qwen EP context MoE op row")

        def query_gemm(self, *args, **kwargs):
            raise PerfDataNotAvailableError("no Qwen EP router op row")

        def query_mem_op(self, *args, **kwargs):
            raise PerfDataNotAvailableError("no Qwen EP shared op row")

    latency, energy, sources = VLLMBackend()._run_context_phase(
        _Model(),
        _Database(),
        RuntimeConfig(),
        batch_size=1,
        isl=1024,
        prefix=0,
    )

    assert latency["context_layerwise"] == pytest.approx(5.0)
    assert latency["context_moe_scheduler_residual"] == pytest.approx(1.0)
    assert energy["context_moe_scheduler_residual"] == 0.0
    assert sources["context_moe_scheduler_residual"] == "silicon"


def test_vllm_layerwise_decode_deepseek_noop_scheduler_adds_moe_back(monkeypatch) -> None:
    from aiconfigurator.sdk.backends import vllm_backend

    monkeypatch.setattr(vllm_backend, "_USE_LAYERWISE", True)

    class _Config:
        tp_size = 1
        pp_size = 1
        attention_dp_size = 1
        moe_tp_size = 1
        moe_ep_size = 1
        moe_quant_mode = common.MoEQuantMode.w4a8_mxfp4_mxfp8
        workload_distribution = "power_law"
        moe_backend = None
        enable_eplb = False

    class _Model:
        model_path = "deepseek-ai/DeepSeek-V4-Flash"
        config = _Config()
        _num_layers = 4
        _hidden_size = 4096
        _topk = 6
        _num_experts = 256
        _moe_inter_size = 2048
        _shared_expert_inter_size = 2048
        _power_law_alpha = 1.2
        _nextn = 0

    class _Database:
        def query_layerwise_detail(self, model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache=0, **kwargs):
            assert (model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache) == (
                "deepseek-ai/DeepSeek-V4-Flash",
                "GEN",
                1,
                1,
                4096,
                0,
            )
            assert kwargs == {
                "moe_weight_mode": None,
                "max_num_batched_tokens": None,
                "moe_tp_size": 1,
                "moe_ep_size": 1,
            }
            return {
                "latency": 5.0,
                "energy": 0.0,
                "latency_source": "schedule_to_update",
                "measured_layer_count": 4,
                "layer_multiplier": 4,
                "physical_gpus": 1,
                "moe_weight_mode": "noop",
            }

        def query_moe(self, **kwargs):
            assert kwargs["num_tokens"] == 1
            assert kwargs["workload_distribution"] == "power_law_1.2"
            return PerformanceResult(0.25, energy=0.5, source="moe")

        def query_gemm(self, m, n, k, quant_mode, database_mode=None):
            assert m == 1
            return PerformanceResult(0.05, energy=0.1, source="gemm")

        def query_mem_op(self, *args, **kwargs):
            return PerformanceResult(0.0, energy=0.0, source="mem")

    latency, energy, sources = VLLMBackend()._get_decode_step_latency(
        _Model(),
        _Database(),
        RuntimeConfig(),
        batch_size=1,
        past_kv=4096,
    )

    assert latency["generation_layerwise"] == pytest.approx(5.0)
    assert latency["generation_moe"] == pytest.approx(1.2)
    assert "generation_moe_router" not in latency
    assert "generation_moe_shared_expert" not in latency
    assert energy["generation_moe"] == pytest.approx(2.4)
    assert sources["generation_moe"] == "mixed"


def test_vllm_layerwise_decode_deepseek_noop_scheduler_moetp_keeps_addback_bundled(monkeypatch) -> None:
    from aiconfigurator.sdk.backends import vllm_backend

    monkeypatch.setattr(vllm_backend, "_USE_LAYERWISE", True)

    class _Config:
        tp_size = 2
        pp_size = 1
        attention_dp_size = 1
        moe_tp_size = 2
        moe_ep_size = 1
        moe_quant_mode = common.MoEQuantMode.w4a8_mxfp4_mxfp8
        workload_distribution = "power_law"
        moe_backend = None
        enable_eplb = False

    class _Model:
        model_path = "deepseek-ai/DeepSeek-V4-Flash"
        config = _Config()
        _num_layers = 4
        _hidden_size = 4096
        _topk = 6
        _num_experts = 256
        _moe_inter_size = 2048
        _shared_expert_inter_size = 2048
        _power_law_alpha = 1.2
        _nextn = 0

    class _Database:
        def query_layerwise_detail(self, model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache=0, **kwargs):
            assert (model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache) == (
                "deepseek-ai/DeepSeek-V4-Flash",
                "GEN",
                2,
                1,
                4096,
                0,
            )
            assert kwargs == {
                "moe_weight_mode": None,
                "max_num_batched_tokens": None,
                "moe_tp_size": 2,
                "moe_ep_size": 1,
            }
            return {
                "latency": 5.0,
                "energy": 0.0,
                "latency_source": "schedule_to_update",
                "measured_layer_count": 4,
                "layer_multiplier": 4,
                "physical_gpus": 1,
                "moe_weight_mode": "noop",
                "components": [{"moe_tp_size": 2.0, "moe_ep_size": 1.0}],
            }

        def query_moe(self, **kwargs):
            raise AssertionError(f"MoE addback should stay bundled for MoE-TP rows: {kwargs}")

        def query_gemm(self, *args, **kwargs):
            raise AssertionError("MoE addback should stay bundled for MoE-TP rows")

        def query_mem_op(self, *args, **kwargs):
            raise AssertionError("MoE addback should stay bundled for MoE-TP rows")

    latency, energy, sources = VLLMBackend()._get_decode_step_latency(
        _Model(),
        _Database(),
        RuntimeConfig(),
        batch_size=1,
        past_kv=4096,
    )

    assert latency == {"generation_layerwise": pytest.approx(5.0)}
    assert energy == {"generation_layerwise": 0.0}
    assert sources == {"generation_layerwise": "silicon"}


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


def test_vllm_layerwise_context_step_uses_chunked_rows(monkeypatch) -> None:
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
                return 1.25
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


def test_vllm_layerwise_deepseek_context_step_uses_prefix0_chunks_when_kv_sparse(monkeypatch) -> None:
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

    assert latency["context_layerwise"] == pytest.approx(1.0 * 2 * 4)
