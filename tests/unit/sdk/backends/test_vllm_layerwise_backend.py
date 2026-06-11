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
    assert latency["context_moe_ep_alltoall"] == pytest.approx(0.4)
    assert energy["context_moe"] == pytest.approx(8.0)
    assert sources["context_moe"] == "empirical"
    assert sources["context_moe_ep_alltoall"] == "silicon"


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


def test_vllm_layerwise_context_envelope_rows_skip_explicit_allreduce(monkeypatch) -> None:
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
                "physical_gpus": 2,
            }

        def query_custom_allreduce(self, *args, **kwargs):
            raise AssertionError((args, kwargs))

    latency, energy, sources = VLLMBackend()._run_context_phase(
        _Model(),
        _Database(),
        RuntimeConfig(),
        batch_size=1,
        isl=128,
        prefix=0,
    )

    assert latency == {"context_layerwise": pytest.approx(17.0), "context_tp_allreduce": 0.0}
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
    assert latency["generation_moe_ep_alltoall"] == pytest.approx(1.6)
    assert energy["generation_moe"] == pytest.approx(16.0)
    assert sources["generation_moe"] == "empirical"
    assert sources["generation_moe_ep_alltoall"] == "silicon"


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


def test_vllm_layerwise_generation_adds_moe_tp_allreduce_for_moe_rows(monkeypatch) -> None:
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
    assert latency["generation_moe_tp_allreduce"] == pytest.approx(0.25 * 4 * 2)
    assert energy["generation_moe_tp_allreduce"] == 0.0
    assert sources["generation_moe_tp_allreduce"] == "silicon"


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
    assert latency["generation_tp_allreduce"] == pytest.approx(0.25 * 40)
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
        RuntimeConfig(),
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
                raise ValueError("sparse CTX KV grid")
            if seq_len == 16384 and seq_len_kv_cache == 0:
                return 2.25
            raise AssertionError((seq_len, seq_len_kv_cache))

    latency, _energy, _sources = VLLMBackend()._run_context_phase(
        _Model(),
        _Database(),
        RuntimeConfig(),
        batch_size=1,
        isl=16384,
        prefix=0,
    )

    assert latency["context_layerwise"] == pytest.approx(2.25 * 4)
