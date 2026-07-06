# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
from types import SimpleNamespace
from typing import ClassVar

import pandas as pd
import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk import pareto_analysis as pa
from aiconfigurator.sdk.config import ModelConfig, RuntimeConfig
from aiconfigurator.sdk.errors import NoFeasibleConfigError

pytestmark = pytest.mark.unit


class _FakeDatabase:
    system_spec: ClassVar[dict] = {"node": {"num_gpus_per_node": 8}}


class _FakeSummary:
    def check_oom(self):
        return False

    def get_result_dict(self):
        return {"ttft": 10.0, "seq/s": 20.0, "memory": 30.0, "power_w": 40.0}

    def get_summary_df(self):
        return pd.DataFrame([{"tokens/s/gpu": 1.0}])



def _patch_afd_pareto_fixed_batch_dependencies(
    monkeypatch,
    *,
    max_batch_size: int = 1024,
    f_max_batch_size: int | None = None,
    session_oom: bool = False,
    tpot: float | None = 10.0,
):
    captured = {"sessions": [], "runtime_configs": [], "max_batch_calls": [], "derive_called": False}

    class FakeBackend:
        name = SimpleNamespace(value="trtllm")

        def get_partition_memory_usage(self, *_args, **_kwargs):
            return {"total": 1.0, "kvcache": 0.1}

        def get_kv_cache_memory_check_params(self):
            return 0.0, 0.0

    class FakeSummary:
        def __init__(self, runtime_config, afd_config):
            self._runtime_config = copy.deepcopy(runtime_config)
            self._afd_config = copy.deepcopy(afd_config)

        def check_oom(self):
            return session_oom

        def get_result_dict(self):
            b_total = self._runtime_config.batch_size
            row_tpot = float(self._runtime_config.tpot if tpot is None else tpot)
            return {
                "model": "Qwen/Qwen3-32B",
                "phase": "decode",
                "isl": self._runtime_config.isl,
                "osl": self._runtime_config.osl,
                "(a)nodes": self._afd_config.n_a_nodes,
                "(a)tp": self._afd_config.tp_a,
                "(a)bs": self._afd_config.a_batch_size,
                "(a)workers": self._afd_config.n_a_workers,
                "(f)nodes": self._afd_config.n_f_nodes,
                "(f)tp": self._afd_config.tp_f,
                "(f)ep": self._afd_config.f_moe_ep_size,
                "(f)workers": self._afd_config.n_f_workers,
                "ttft": 0.0,
                "tpot": row_tpot,
                "request_latency": row_tpot * max(self._runtime_config.osl - 1, 1),
                "seq/s": 1.0,
                "request_rate": 1.0,
                "tokens/s": float(self._runtime_config.osl),
                "tokens/s/gpu": 1.0,
                "tokens/s/user": float(self._runtime_config.osl),
                "concurrency": b_total,
                "parallel": "a1n-tp2+f1n-ep1",
                "num_total_gpus": 16,
                "memory": 1.0,
                "power_w": 0.0,
                "backend": "trtllm",
                "version": "test-version",
                "system": "h200_sxm",
            }

    class FakeAFDInferenceSession:
        def __init__(self, *, afd_config, **_kwargs):
            captured["sessions"].append(copy.deepcopy(afd_config))

        def run_afd(self, runtime_config, **_kwargs):
            captured["runtime_configs"].append(copy.deepcopy(runtime_config))
            return FakeSummary(runtime_config, captured["sessions"][-1])

    def fake_analytical_max_batch_size(*_args, include_kvcache, **_kwargs):
        captured["max_batch_calls"].append(include_kvcache)
        if not include_kvcache and f_max_batch_size is not None:
            return f_max_batch_size
        return max_batch_size

    def fake_derive_a_batch_size(*_args, **_kwargs):
        captured["derive_called"] = True
        return max_batch_size, object(), SimpleNamespace(attn_ops=[])

    monkeypatch.setattr(pa, "get_backend", lambda _name: FakeBackend())
    monkeypatch.setattr(pa, "get_model", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(pa, "AFDInferenceSession", FakeAFDInferenceSession)
    monkeypatch.setattr(pa, "_analytical_max_batch_size", fake_analytical_max_batch_size)
    monkeypatch.setattr(pa, "_derive_a_batch_size", fake_derive_a_batch_size)
    monkeypatch.setattr(pa, "_quick_balance_ratio", lambda *_args, **_kwargs: 1.0)
    monkeypatch.setattr(
        "aiconfigurator.sdk.afd_partition.build_afd_ops_partition",
        lambda *_args, **_kwargs: SimpleNamespace(attn_ops=[], ffn_ops=[]),
    )
    return captured


def test_afd_pareto_fixed_total_batch_uses_exact_a_batch(monkeypatch):
    captured = _patch_afd_pareto_fixed_batch_dependencies(monkeypatch, max_batch_size=1024)

    df = pa.afd_pareto(
        model_path="Qwen/Qwen3-32B",
        runtime_config=RuntimeConfig(isl=128, osl=32, tpot=25.0),
        database=_FakeDatabase(),
        backend_name="trtllm",
        afd_parallel_config_list=[(1, 1, 2, 1, 3, "optimistic")],
        gpus_per_node=8,
        combined_with_pd=False,
        total_batch_size=256,
    )

    assert not df.empty
    assert len(captured["sessions"]) == 1
    assert captured["sessions"][0].a_batch_size == 64
    assert captured["runtime_configs"][0].batch_size == 256
    assert captured["derive_called"] is False


def test_afd_pareto_request_latency_enumerates_constraints_and_filters_final_rows(monkeypatch):
    captured = _patch_afd_pareto_fixed_batch_dependencies(monkeypatch, max_batch_size=1024, tpot=None)
    monkeypatch.setattr(
        pa, "enumerate_ttft_tpot_constraints",
        lambda _osl, _request_latency, _ttft: [(100.0, 5.0), (100.0, 20.0)],
    )

    df = pa.afd_pareto(
        model_path="Qwen/Qwen3-32B",
        runtime_config=RuntimeConfig(isl=128, osl=10, ttft=100.0, tpot=99.0, request_latency=150.0),
        database=_FakeDatabase(),
        backend_name="trtllm",
        afd_parallel_config_list=[(1, 1, 2, 1, 3, "optimistic")],
        gpus_per_node=8,
        combined_with_pd=False,
        total_batch_size=256,
    )

    assert [runtime_config.tpot for runtime_config in captured["runtime_configs"]] == [5.0, 20.0]
    assert df["tpot"].tolist() == [5.0]
    assert (df["request_latency"] <= 150.0).all()


def test_afd_pareto_no_feasible_sla_reports_rejection_summary(monkeypatch):
    _patch_afd_pareto_fixed_batch_dependencies(monkeypatch, max_batch_size=1024, tpot=None)
    monkeypatch.setattr(pa, "enumerate_ttft_tpot_constraints", lambda _osl, _request_latency, _ttft: [(100.0, 20.0)])

    with pytest.raises(NoFeasibleConfigError, match=r"Rejections: .*request_latency"):
        pa.afd_pareto(
            model_path="Qwen/Qwen3-32B",
            runtime_config=RuntimeConfig(isl=128, osl=10, ttft=100.0, tpot=99.0, request_latency=150.0),
            database=_FakeDatabase(),
            backend_name="trtllm",
            afd_parallel_config_list=[(1, 1, 2, 1, 3, "optimistic")],
            gpus_per_node=8,
            combined_with_pd=False,
            total_batch_size=256,
        )


@pytest.mark.parametrize("total_batch_size", [0, True, 256.0])
def test_afd_pareto_fixed_total_batch_rejects_non_positive_integer(total_batch_size):
    with pytest.raises(ValueError, match="total_batch_size must be a positive integer"):
        pa.afd_pareto(
            model_path="Qwen/Qwen3-32B",
            runtime_config=RuntimeConfig(isl=128, osl=32),
            database=_FakeDatabase(),
            backend_name="trtllm",
            afd_parallel_config_list=[(1, 1, 2, 1, 3, "optimistic")],
            gpus_per_node=8,
            combined_with_pd=False,
            total_batch_size=total_batch_size,
        )


@pytest.mark.parametrize("total_batch_size", [3, 10])
def test_afd_pareto_fixed_total_batch_requires_exact_divisibility(monkeypatch, total_batch_size):
    captured = _patch_afd_pareto_fixed_batch_dependencies(monkeypatch, max_batch_size=1024)

    with pytest.raises(NoFeasibleConfigError, match=f"total_batch_size={total_batch_size}"):
        pa.afd_pareto(
            model_path="Qwen/Qwen3-32B",
            runtime_config=RuntimeConfig(isl=128, osl=32, tpot=25.0),
            database=_FakeDatabase(),
            backend_name="trtllm",
            afd_parallel_config_list=[(1, 1, 2, 1, 3, "optimistic")],
            gpus_per_node=8,
            combined_with_pd=False,
            total_batch_size=total_batch_size,
        )

    assert captured["sessions"] == []


def test_afd_pareto_fixed_total_batch_checks_memory_capacity(monkeypatch):
    captured = _patch_afd_pareto_fixed_batch_dependencies(monkeypatch, max_batch_size=32)

    with pytest.raises(NoFeasibleConfigError, match="total_batch_size=256"):
        pa.afd_pareto(
            model_path="Qwen/Qwen3-32B",
            runtime_config=RuntimeConfig(isl=128, osl=32, tpot=25.0),
            database=_FakeDatabase(),
            backend_name="trtllm",
            afd_parallel_config_list=[(1, 1, 2, 1, 3, "optimistic")],
            gpus_per_node=8,
            combined_with_pd=False,
            total_batch_size=256,
        )

    assert captured["sessions"] == []


def test_afd_pareto_without_fixed_total_batch_keeps_auto_derivation(monkeypatch):
    captured = _patch_afd_pareto_fixed_batch_dependencies(
        monkeypatch,
        max_batch_size=64,
        f_max_batch_size=1024,
    )

    df = pa.afd_pareto(
        model_path="Qwen/Qwen3-32B",
        runtime_config=RuntimeConfig(isl=128, osl=32),
        database=_FakeDatabase(),
        backend_name="trtllm",
        afd_parallel_config_list=[(1, 1, 2, 1, 3, "optimistic")],
        gpus_per_node=8,
        combined_with_pd=False,
    )

    assert not df.empty
    assert captured["derive_called"] is True
    assert len(captured["sessions"]) > 1
    evaluated_batch_sizes = [afd_config.a_batch_size for afd_config in captured["sessions"]]
    assert {1, 2, 4, 64}.issubset(evaluated_batch_sizes)
    assert {1, 2, 4, 64}.issubset(set(df["(a)bs"]))


def test_afd_pareto_combined_with_pd_requires_static_prefill_options(monkeypatch):
    monkeypatch.setattr(pa, "get_backend", lambda _name: object())
    monkeypatch.setattr(pa, "_enumerate_afd_prefill_options", lambda **_kwargs: [])

    with pytest.raises(NoFeasibleConfigError, match="combined_with_pd=True requires"):
        pa.afd_pareto(
            model_path="Qwen/Qwen3-32B",
            runtime_config=RuntimeConfig(isl=128, osl=32, tpot=25.0),
            database=_FakeDatabase(),
            backend_name="trtllm",
            afd_parallel_config_list=[(1, 1, 2, 2, 3, "optimistic")],
            gpus_per_node=8,
            combined_with_pd=True,
        )


def test_static_prefill_options_inherit_base_model_config(monkeypatch):
    captured_model_configs = []

    class FakeInferenceSession:
        def __init__(self, **_kwargs):
            pass

        def run_static(self, runtime_config, mode):
            assert mode == "static_ctx"
            assert runtime_config.batch_size == 1
            return _FakeSummary()

    def fake_get_model(_model_path, model_config, _backend_name):
        captured_model_configs.append(copy.deepcopy(model_config))
        return object()

    monkeypatch.setattr(pa, "check_is_moe", lambda _model_path: True)
    monkeypatch.setattr(pa, "get_backend", lambda _name: object())
    monkeypatch.setattr(pa, "get_model", fake_get_model)
    monkeypatch.setattr(pa, "InferenceSession", FakeInferenceSession)

    base_model_config = ModelConfig(
        nextn=2,
        nextn_accept_rates=[0.7, 0.2, 0.0, 0.0, 0.0],
        moe_backend="deepep_moe",
        attention_backend="fa3",
        enable_wideep=True,
        enable_eplb=True,
        workload_distribution="worker_distribution",
        gemm_quant_mode=common.GEMMQuantMode.fp8,
    )

    options = pa._enumerate_afd_prefill_options(
        model_path="Qwen/Qwen3-MoE",
        runtime_config=RuntimeConfig(isl=128, osl=32),
        database=_FakeDatabase(),
        backend_name="trtllm",
        gpus_per_node=4,
        base_model_config=base_model_config,
    )

    assert options
    assert captured_model_configs
    for model_config in captured_model_configs:
        assert model_config.nextn == 2
        assert model_config.nextn_accept_rates == [0.7, 0.2, 0.0, 0.0, 0.0]
        assert model_config.moe_backend == "deepep_moe"
        assert model_config.attention_backend == "fa3"
        assert model_config.enable_wideep is True
        assert model_config.enable_eplb is True
        assert model_config.workload_distribution == "worker_distribution"
        assert model_config.gemm_quant_mode == common.GEMMQuantMode.fp8
        assert model_config.pp_size == 1
        assert model_config.attention_dp_size == 1
        assert model_config.moe_tp_size == 1
        assert model_config.moe_ep_size == model_config.tp_size
