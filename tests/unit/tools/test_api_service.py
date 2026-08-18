# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the REST API service (tools/api_service/app.py).

These tests mock the SDK functions to test API layer logic without
requiring the Rust native extension or performance databases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from tools.api_service.app import app

client = TestClient(app)


def _sdk_available() -> bool:
    try:
        from aiconfigurator_core.sdk.common import SupportedSystems
        return len(SupportedSystems) > 0
    except Exception:
        return False


# ─── Fixtures ────────────────────────────────────────────────────────────────

VALID_RECOMMEND_BODY = {
    "model_path": "Qwen/Qwen3-32B",
    "system": "h200_sxm",
    "target_concurrency": 32,
    "isl": 4000,
    "osl": 1000,
    "ttft": 2000.0,
    "tpot": 30.0,
    "backend": "vllm",
    "database_mode": "HYBRID",
    "top_n": 2,
}

VALID_MEMORY_BODY = {
    "model_path": "Qwen/Qwen3-32B",
    "system": "h200_sxm",
    "backend": "vllm",
    "backend_version": "0.24.0",
    "max_num_tokens": 8192,
    "max_batch_size": 128,
    "tp_size": 2,
    "pp_size": 1,
    "memory_fraction_kind": "of_total",
    "memory_fraction_value": 0.9,
}

MOCK_COLUMNS_AGG = [
    "model", "isl", "osl", "prefix", "concurrency", "request_rate",
    "bs", "global_bs", "ttft", "tpot", "request_latency",
    "encoder_latency", "encoder_memory",
    "seq/s", "seq/s/gpu", "tokens/s", "tokens/s/gpu", "tokens/s/user",
    "num_total_gpus", "tp", "pp", "dp", "moe_tp", "moe_ep", "cp",
    "parallel", "gemm", "kvcache", "fmha", "moe", "comm",
    "memory", "balance_score",
    "num_ctx_reqs", "num_gen_reqs", "num_tokens", "ctx_tokens", "gen_tokens",
    "backend", "version", "system", "power_w",
]

MOCK_ROW = {
    "model": "Qwen/Qwen3-32B",
    "isl": np.int64(4000), "osl": np.int64(1000), "prefix": np.int64(0),
    "concurrency": np.int64(48), "request_rate": 1.681,
    "bs": np.int64(48), "global_bs": np.int64(48),
    "ttft": 471.378, "tpot": 28.118, "request_latency": 28561.14,
    "encoder_latency": np.int64(0), "encoder_memory": 0.0,
    "seq/s": 1.681, "seq/s/gpu": 0.84,
    "tokens/s": 1678.925, "tokens/s/gpu": 839.462, "tokens/s/user": 35.565,
    "num_total_gpus": np.int64(2),
    "tp": np.int64(2), "pp": np.int64(1), "dp": np.int64(1),
    "moe_tp": np.int64(1), "moe_ep": np.int64(1), "cp": np.int64(1),
    "parallel": "tp2pp1dp1etp1ep1",
    "gemm": "bfloat16", "kvcache": "bfloat16", "fmha": "bfloat16",
    "moe": "bfloat16", "comm": "half",
    "memory": 64.044, "balance_score": 0.048,
    "num_ctx_reqs": 1.0, "num_gen_reqs": 47.0,
    "num_tokens": 4047.0, "ctx_tokens": np.int64(4000), "gen_tokens": 47.0,
    "backend": "vllm", "version": "0.24.0", "system": "h200_sxm", "power_w": 0.0,
    "total_gpus_needed": np.int64(2), "replicas_needed": np.int64(1),
}

MOCK_KV_CACHE_RESULT = {
    "total_gpu_capacity_bytes": 151397597184,
    "total_kv_size_bytes": 98507266457,
    "kv_size_per_token_bytes": 131072,
    "total_kv_size_tokens": 751550,
    "source": "native",
    "memory_breakdown": {
        "weights_bytes": 32761446400,
        "activations_bytes": 532480000,
        "runtime_overhead_bytes": 3758096384,
        "comm_overhead_bytes": 358612992,
    },
    "tolerance_adjusted": None,
}

MOCK_SYSTEM_SPEC = {
    "gpu": {
        "mem_bw": 4800000000000,
        "mem_capacity": 151397597184,
        "bfloat16_tc_flops": 989000000000000,
        "fp8_tc_flops": 1978000000000000,
        "power": 700,
        "sm_version": 90,
    },
    "node": {
        "num_gpus_per_node": 8,
    },
}


@dataclass
class MockCLIResult:
    chosen_exp: str = "agg"
    best_configs: dict = field(default_factory=dict)
    pareto_fronts: dict = field(default_factory=dict)
    best_throughputs: dict = field(default_factory=dict)
    tasks: dict = field(default_factory=dict)
    best_latencies: dict = field(default_factory=dict)
    raw_results: dict = field(default_factory=dict)
    outcomes: dict = field(default_factory=dict)


def make_mock_cli_result(rows=None):
    if rows is None:
        rows = [MOCK_ROW]
    df = pd.DataFrame(rows)
    result = MockCLIResult()
    result.best_configs = {"agg": df}
    return result


# ─── /recommend tests ────────────────────────────────────────────────────────


class TestRecommend:

    @patch("tools.api_service.app.cli_recommend")
    def test_success(self, mock_recommend):
        mock_recommend.return_value = make_mock_cli_result()
        resp = client.post("/recommend", json=VALID_RECOMMEND_BODY)
        assert resp.status_code == 200
        data = resp.json()
        assert "configs" in data
        assert "chosen_mode" in data
        assert data["chosen_mode"] == "agg"
        assert len(data["configs"]) >= 1

    @patch("tools.api_service.app.cli_recommend")
    def test_response_fields(self, mock_recommend):
        mock_recommend.return_value = make_mock_cli_result()
        resp = client.post("/recommend", json=VALID_RECOMMEND_BODY)
        cfg = resp.json()["configs"][0]
        assert cfg["tp"] == 2
        assert cfg["pp"] == 1
        assert cfg["dp"] == 1
        assert cfg["total_gpus_needed"] == 2
        assert cfg["replicas_needed"] == 1
        assert cfg["num_total_gpus"] == 2
        assert cfg["ttft"] == 471.378
        assert cfg["tpot"] == 28.118
        assert cfg["tokens_per_second"] == 1678.925
        assert cfg["tokens_per_second_per_gpu"] == 839.462
        assert cfg["memory"] == 64.044
        assert cfg["model"] == "Qwen/Qwen3-32B"
        assert cfg["system"] == "h200_sxm"
        assert cfg["backend"] == "vllm"
        assert cfg["backend_version"] == "0.24.0"
        assert cfg["gemm"] == "bfloat16"

    @patch("tools.api_service.app.cli_recommend")
    def test_no_serving_config_by_default(self, mock_recommend):
        mock_recommend.return_value = make_mock_cli_result()
        resp = client.post("/recommend", json=VALID_RECOMMEND_BODY)
        cfg = resp.json()["configs"][0]
        assert cfg["serving_config"] is None
        assert cfg["memory_breakdown"] is None

    @patch("tools.api_service.app.estimate_kv_cache")
    @patch("tools.api_service.app.cli_recommend")
    def test_include_config(self, mock_recommend, mock_kv):
        mock_recommend.return_value = make_mock_cli_result()
        resp = client.post("/recommend?include=config", json=VALID_RECOMMEND_BODY)
        cfg = resp.json()["configs"][0]
        sc = cfg["serving_config"]
        assert sc is not None
        assert sc["backend"] == "vllm"
        assert sc["tensor_parallel_size"] == 2
        assert sc["max_model_len"] == 5000
        assert sc["gpu_memory_utilization"] == 0.9
        assert isinstance(sc["enable_chunked_prefill"], bool)
        assert isinstance(sc["enable_prefix_caching"], bool)
        assert cfg["memory_breakdown"] is None

    @patch("tools.api_service.app.estimate_kv_cache")
    @patch("tools.api_service.app.cli_recommend")
    def test_include_memory(self, mock_recommend, mock_kv):
        mock_recommend.return_value = make_mock_cli_result()
        mock_kv.return_value = MOCK_KV_CACHE_RESULT
        resp = client.post("/recommend?include=memory", json=VALID_RECOMMEND_BODY)
        cfg = resp.json()["configs"][0]
        mb = cfg["memory_breakdown"]
        assert mb is not None
        assert mb["weights_bytes"] == 32761446400
        assert mb["activations_bytes"] == 532480000
        assert mb["kv_cache_bytes"] == 98507266457
        assert cfg["serving_config"] is None

    @patch("tools.api_service.app.estimate_kv_cache")
    @patch("tools.api_service.app.cli_recommend")
    def test_include_config_and_memory(self, mock_recommend, mock_kv):
        mock_recommend.return_value = make_mock_cli_result()
        mock_kv.return_value = MOCK_KV_CACHE_RESULT
        resp = client.post("/recommend?include=config,memory", json=VALID_RECOMMEND_BODY)
        cfg = resp.json()["configs"][0]
        assert cfg["serving_config"] is not None
        assert cfg["memory_breakdown"] is not None

    @patch("tools.api_service.app.cli_recommend")
    def test_top_n_limits_results(self, mock_recommend):
        rows = [MOCK_ROW.copy() for _ in range(5)]
        mock_recommend.return_value = make_mock_cli_result(rows)
        body = {**VALID_RECOMMEND_BODY, "top_n": 2}
        resp = client.post("/recommend", json=body)
        assert len(resp.json()["configs"]) == 2

    def test_requires_exactly_one_target(self):
        body = {**VALID_RECOMMEND_BODY}
        del body["target_concurrency"]
        resp = client.post("/recommend", json=body)
        assert resp.status_code == 422

    def test_rejects_both_targets(self):
        body = {**VALID_RECOMMEND_BODY, "target_request_rate": 10}
        resp = client.post("/recommend", json=body)
        assert resp.status_code == 422

    def test_accepts_target_request_rate(self):
        body = {**VALID_RECOMMEND_BODY}
        del body["target_concurrency"]
        body["target_request_rate"] = 10.0
        with patch("tools.api_service.app.cli_recommend") as mock:
            mock.return_value = make_mock_cli_result()
            resp = client.post("/recommend", json=body)
            assert resp.status_code == 200
            call_kwargs = mock.call_args.kwargs
            assert call_kwargs["target_request_rate"] == 10.0
            assert call_kwargs["target_concurrency"] is None

    def test_requires_model_path(self):
        body = {**VALID_RECOMMEND_BODY}
        del body["model_path"]
        resp = client.post("/recommend", json=body)
        assert resp.status_code == 422

    def test_requires_system(self):
        body = {**VALID_RECOMMEND_BODY}
        del body["system"]
        resp = client.post("/recommend", json=body)
        assert resp.status_code == 422

    @patch("tools.api_service.app.cli_recommend")
    def test_no_config_found_returns_422(self, mock_recommend):
        result = MockCLIResult()
        result.best_configs = {"agg": pd.DataFrame()}
        mock_recommend.return_value = result
        resp = client.post("/recommend", json=VALID_RECOMMEND_BODY)
        assert resp.status_code == 422
        assert "No configuration" in resp.json()["detail"]

    @patch("tools.api_service.app.cli_recommend")
    def test_value_error_returns_422(self, mock_recommend):
        mock_recommend.side_effect = ValueError("bad input")
        resp = client.post("/recommend", json=VALID_RECOMMEND_BODY)
        assert resp.status_code == 422
        assert "bad input" in resp.json()["detail"]

    @patch("tools.api_service.app.cli_recommend")
    def test_unexpected_error_returns_500(self, mock_recommend):
        mock_recommend.side_effect = RuntimeError("internal failure")
        resp = client.post("/recommend", json=VALID_RECOMMEND_BODY)
        assert resp.status_code == 500

    @patch("tools.api_service.app.cli_recommend")
    def test_defaults_applied(self, mock_recommend):
        mock_recommend.return_value = make_mock_cli_result()
        body = {"model_path": "Qwen/Qwen3-32B", "system": "h200_sxm", "target_concurrency": 32}
        resp = client.post("/recommend", json=body)
        assert resp.status_code == 200
        call_kwargs = mock_recommend.call_args.kwargs
        assert call_kwargs["backend"] == "vllm"
        assert call_kwargs["isl"] == 4000
        assert call_kwargs["osl"] == 1000
        assert call_kwargs["ttft"] == 2000.0
        assert call_kwargs["tpot"] == 30.0
        assert call_kwargs["database_mode"] == "HYBRID"
        assert call_kwargs["top_n"] == 5

    @patch("tools.api_service.app.cli_recommend")
    def test_null_fields_get_request_defaults(self, mock_recommend):
        row = {k: None for k in MOCK_ROW}
        row["model"] = "Qwen/Qwen3-32B"
        row["ttft"] = 500.0
        row["tpot"] = 25.0
        row["concurrency"] = np.int64(32)
        row["tokens/s"] = 1000.0
        row["total_gpus_needed"] = np.int64(4)
        row["replicas_needed"] = np.int64(2)
        row["num_total_gpus"] = np.int64(2)
        mock_recommend.return_value = make_mock_cli_result([row])
        resp = client.post("/recommend", json=VALID_RECOMMEND_BODY)
        cfg = resp.json()["configs"][0]
        assert cfg["system"] == "h200_sxm"
        assert cfg["backend"] == "vllm"

    @patch("tools.api_service.app.estimate_kv_cache")
    @patch("tools.api_service.app.cli_recommend")
    def test_memory_breakdown_failure_returns_null(self, mock_recommend, mock_kv):
        mock_recommend.return_value = make_mock_cli_result()
        mock_kv.side_effect = ValueError("unsupported")
        resp = client.post("/recommend?include=memory", json=VALID_RECOMMEND_BODY)
        assert resp.status_code == 200
        assert resp.json()["configs"][0]["memory_breakdown"] is None


# ─── /recommend mode tests ────────────────────────────────────────────────────


class TestRecommendMode:

    @patch("tools.api_service.app.load_system_spec")
    @patch("tools.api_service.app._execute_and_wrap_result")
    @patch("tools.api_service.app._build_recommend_tasks")
    @patch("tools.api_service.app.build_default_tasks")
    def test_quick_mode_returns_single_config(self, mock_build, mock_reco_tasks, mock_execute, mock_spec):
        mock_spec.return_value = MOCK_SYSTEM_SPEC
        mock_build.return_value = {"agg_vllm": MagicMock(serving_mode="agg")}
        mock_reco_tasks.return_value = {"agg_vllm": MagicMock()}
        mock_execute.return_value = make_mock_cli_result()
        resp = client.post("/recommend?mode=quick", json=VALID_RECOMMEND_BODY)
        assert resp.status_code == 200
        assert len(resp.json()["configs"]) == 1
        call_kwargs = mock_execute.call_args
        assert call_kwargs.kwargs.get("top_n") == 1

    @patch("tools.api_service.app.load_system_spec")
    @patch("tools.api_service.app._execute_and_wrap_result")
    @patch("tools.api_service.app._build_recommend_tasks")
    @patch("tools.api_service.app.build_default_tasks")
    def test_quick_mode_filters_disagg_tasks(self, mock_build, mock_reco_tasks, mock_execute, mock_spec):
        mock_spec.return_value = MOCK_SYSTEM_SPEC
        mock_build.return_value = {
            "agg_vllm": MagicMock(serving_mode="agg"),
            "disagg_vllm": MagicMock(serving_mode="disagg"),
        }
        mock_reco_tasks.return_value = {"agg_vllm": MagicMock()}
        mock_execute.return_value = make_mock_cli_result()
        resp = client.post("/recommend?mode=quick", json=VALID_RECOMMEND_BODY)
        assert resp.status_code == 200
        built_tasks = mock_reco_tasks.call_args[0][0]
        assert all("disagg" not in k for k in built_tasks)

    @patch("tools.api_service.app.cli_recommend")
    def test_default_mode_unchanged(self, mock_recommend):
        mock_recommend.return_value = make_mock_cli_result()
        resp = client.post("/recommend", json=VALID_RECOMMEND_BODY)
        assert resp.status_code == 200
        assert resp.json()["chosen_mode"] == "agg"

    @patch("tools.api_service.app.estimate_kv_cache")
    @patch("tools.api_service.app.load_system_spec")
    @patch("tools.api_service.app._execute_and_wrap_result")
    @patch("tools.api_service.app._build_recommend_tasks")
    @patch("tools.api_service.app.build_default_tasks")
    def test_quick_with_include_config_and_memory(self, mock_build, mock_reco_tasks, mock_execute, mock_spec, mock_kv):
        mock_spec.return_value = MOCK_SYSTEM_SPEC
        mock_build.return_value = {"agg_vllm": MagicMock(serving_mode="agg")}
        mock_reco_tasks.return_value = {"agg_vllm": MagicMock()}
        mock_execute.return_value = make_mock_cli_result()
        mock_kv.return_value = MOCK_KV_CACHE_RESULT
        resp = client.post("/recommend?mode=quick&include=config,memory", json=VALID_RECOMMEND_BODY)
        assert resp.status_code == 200
        cfg = resp.json()["configs"][0]
        assert cfg["serving_config"] is not None
        assert cfg["serving_config"]["tensor_parallel_size"] >= 1
        assert cfg["memory_breakdown"] is not None
        assert cfg["memory_breakdown"]["weights_bytes"] > 0


# ─── /memory tests ───────────────────────────────────────────────────────────


class TestMemory:

    @patch("tools.api_service.app.estimate_kv_cache")
    def test_success(self, mock_kv):
        mock_kv.return_value = MOCK_KV_CACHE_RESULT
        resp = client.post("/memory", json=VALID_MEMORY_BODY)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_gpu_capacity_bytes"] == 151397597184
        assert data["total_kv_size_bytes"] == 98507266457
        assert data["kv_size_per_token_bytes"] == 131072
        assert data["total_kv_size_tokens"] == 751550
        assert data["source"] == "native"

    @patch("tools.api_service.app.estimate_kv_cache")
    def test_memory_breakdown(self, mock_kv):
        mock_kv.return_value = MOCK_KV_CACHE_RESULT
        resp = client.post("/memory", json=VALID_MEMORY_BODY)
        mb = resp.json()["memory_breakdown"]
        assert mb["weights_bytes"] == 32761446400
        assert mb["activations_bytes"] == 532480000
        assert mb["runtime_overhead_bytes"] == 3758096384
        assert mb["comm_overhead_bytes"] == 358612992
        assert mb["kv_cache_bytes"] == 98507266457

    @patch("tools.api_service.app.estimate_kv_cache")
    def test_value_error_returns_422(self, mock_kv):
        mock_kv.side_effect = ValueError("unsupported model/backend/GPU for KV-cache estimation")
        resp = client.post("/memory", json=VALID_MEMORY_BODY)
        assert resp.status_code == 422
        assert "No performance data" in resp.json()["detail"]

    @patch("tools.api_service.app.estimate_kv_cache")
    def test_unexpected_error_returns_500(self, mock_kv):
        mock_kv.side_effect = RuntimeError("crash")
        resp = client.post("/memory", json=VALID_MEMORY_BODY)
        assert resp.status_code == 500

    def test_requires_model_path(self):
        body = {**VALID_MEMORY_BODY}
        del body["model_path"]
        resp = client.post("/memory", json=body)
        assert resp.status_code == 422

    def test_requires_system(self):
        body = {**VALID_MEMORY_BODY}
        del body["system"]
        resp = client.post("/memory", json=body)
        assert resp.status_code == 422

    @patch("tools.api_service.app.estimate_kv_cache")
    def test_defaults_applied(self, mock_kv):
        mock_kv.return_value = MOCK_KV_CACHE_RESULT
        body = {"model_path": "Qwen/Qwen3-32B", "system": "h200_sxm"}
        resp = client.post("/memory", json=body)
        assert resp.status_code == 200
        call_kwargs = mock_kv.call_args.kwargs
        assert call_kwargs["backend"] == "vllm"
        assert call_kwargs["max_num_tokens"] == 8192
        assert call_kwargs["max_batch_size"] == 128
        assert call_kwargs["tp_size"] == 1
        assert call_kwargs["memory_fraction_kind"] == "of_total"
        assert call_kwargs["memory_fraction_value"] == 1.0


# ─── /models tests ───────────────────────────────────────────────────────────


class TestModels:

    @patch("tools.api_service.app.get_default_models")
    def test_returns_sorted_list(self, mock_models):
        mock_models.return_value = {"Zeta/Z-1B", "Alpha/A-7B", "Meta/M-70B"}
        resp = client.get("/models")
        assert resp.status_code == 200
        models = resp.json()["models"]
        assert models == ["Alpha/A-7B", "Meta/M-70B", "Zeta/Z-1B"]

    @patch("tools.api_service.app.get_default_models")
    def test_empty_set(self, mock_models):
        mock_models.return_value = set()
        resp = client.get("/models")
        assert resp.status_code == 200
        assert resp.json()["models"] == []


# ─── /systems tests ──────────────────────────────────────────────────────────


class TestSystems:

    @patch("tools.api_service.app.SupportedSystems", {"h200_sxm", "a100_sxm"})
    def test_returns_sorted_objects(self):
        resp = client.get("/systems")
        assert resp.status_code == 200
        systems = resp.json()["systems"]
        assert len(systems) == 2
        assert systems[0]["id"] == "a100_sxm"
        assert systems[1]["id"] == "h200_sxm"
        assert "name" in systems[0]
        assert "vendor" not in systems[0]

    @patch("tools.api_service.app.load_system_spec")
    @patch("tools.api_service.app.SupportedSystems", {"h200_sxm"})
    def test_include_specs(self, mock_spec):
        mock_spec.return_value = MOCK_SYSTEM_SPEC
        resp = client.get("/systems?include=specs")
        assert resp.status_code == 200
        sys = resp.json()["systems"][0]
        assert sys["id"] == "h200_sxm"
        assert sys["name"] == "NVIDIA H200 SXM"
        assert sys["vendor"] == "nvidia"
        assert sys["architecture"] == "hopper"
        assert sys["memory_bytes"] == 151397597184
        assert sys["tdp_watts"] == 700.0
        assert sys["gpus_per_node"] == 8

    @patch("tools.api_service.app.SupportedSystems", {"h200_sxm"})
    def test_no_include_omits_specs(self):
        resp = client.get("/systems")
        sys = resp.json()["systems"][0]
        assert "vendor" not in sys
        assert "memory_bytes" not in sys

    @patch("tools.api_service.app.load_system_spec")
    @patch("tools.api_service.app.SupportedSystems", {"h200_sxm"})
    def test_spec_failure_still_returns_entry(self, mock_spec):
        mock_spec.side_effect = FileNotFoundError("missing yaml")
        resp = client.get("/systems?include=specs")
        assert resp.status_code == 200
        sys = resp.json()["systems"][0]
        assert sys["id"] == "h200_sxm"
        assert "vendor" not in sys


# ─── Integration tests (require SDK) ─────────────────────────────────────────


@pytest.mark.skipif(
    not _sdk_available(),
    reason="aiconfigurator SDK not installed or missing perf data",
)
class TestIntegration:

    def test_recommend_real(self):
        resp = client.post("/recommend", json={
            "model_path": "Qwen/Qwen3-32B",
            "system": "h200_sxm",
            "target_concurrency": 32,
            "top_n": 1,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["configs"]) == 1
        cfg = data["configs"][0]
        assert cfg["total_gpus_needed"] >= 1
        assert cfg["tp"] >= 1
        assert cfg["ttft"] > 0
        assert cfg["tokens_per_second"] > 0

    def test_recommend_with_include(self):
        resp = client.post("/recommend?include=config,memory", json={
            "model_path": "Qwen/Qwen3-32B",
            "system": "h200_sxm",
            "target_concurrency": 32,
            "top_n": 1,
        })
        assert resp.status_code == 200
        cfg = resp.json()["configs"][0]
        assert cfg["serving_config"] is not None
        assert cfg["serving_config"]["tensor_parallel_size"] >= 1

    def test_memory_real(self):
        resp = client.post("/memory", json={
            "model_path": "Qwen/Qwen3-32B",
            "system": "h200_sxm",
            "backend": "vllm",
            "backend_version": "0.24.0",
            "tp_size": 2,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_kv_size_bytes"] > 0
        assert data["memory_breakdown"]["weights_bytes"] > 0

    def test_models_real(self):
        resp = client.get("/models")
        assert resp.status_code == 200
        models = resp.json()["models"]
        assert len(models) > 0
        assert all(isinstance(m, str) for m in models)

    def test_systems_real(self):
        resp = client.get("/systems?include=specs")
        assert resp.status_code == 200
        systems = resp.json()["systems"]
        assert len(systems) > 0
        assert all(s["memory_bytes"] > 0 for s in systems)

    def test_estimate_real(self):
        resp = client.post("/estimate", json={
            "model_path": "Qwen/Qwen3-32B",
            "system": "h200_sxm",
            "backend": "vllm",
            "tp_size": 2,
            "batch_size": 48,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["ttft"] > 0
        assert data["tpot"] > 0
        assert data["tokens_per_second"] > 0

    def test_estimate_with_include_real(self):
        resp = client.post("/estimate?include=config,memory", json={
            "model_path": "Qwen/Qwen3-32B",
            "system": "h200_sxm",
            "backend": "vllm",
            "backend_version": "0.24.0",
            "tp_size": 2,
            "batch_size": 48,
        })
        assert resp.status_code == 200
        cfg = resp.json()
        assert cfg["serving_config"] is not None
        assert cfg["serving_config"]["tensor_parallel_size"] == 2


# ─── /recommend disagg tests ──────────────────────────────────────────────────


class TestRecommendDisagg:

    @patch("tools.api_service.app.cli_recommend")
    def test_disagg_result_has_prefill_decode_configs(self, mock_recommend):
        disagg_row = {
            "model": "Qwen/Qwen3-32B", "isl": 1000, "osl": 150,
            "concurrency": 36, "ttft": 180.157, "tpot": 24.675,
            "tokens/s": 1348.785, "tokens/s/gpu": 674.392,
            "num_total_gpus": 2, "total_gpus_needed": 6, "replicas_needed": 3,
            "request_rate": 8.992, "request_latency": 3856.732,
            "encoder_latency": 0.0, "encoder_memory": 0.0,
            "seq/s": 8.992, "seq/s/gpu": 4.496,
            "tokens/s/user": 40.527, "power_w": 0.0,
            "(p)tp": np.int64(1), "(p)pp": np.int64(1), "(p)dp": np.int64(1),
            "(p)workers": np.int64(1), "(p)memory": 64.9,
            "(p)gemm": "bfloat16", "(p)kvcache": "bfloat16",
            "(p)fmha": "bfloat16", "(p)moe": "bfloat16", "(p)comm": "half",
            "(p)version": "0.24.0",
            "(d)tp": np.int64(1), "(d)pp": np.int64(1), "(d)dp": np.int64(1),
            "(d)workers": np.int64(1), "(d)memory": 64.9,
            "(d)gemm": "bfloat16", "(d)version": "0.24.0",
        }
        result = make_mock_cli_result([disagg_row])
        result.chosen_exp = "disagg_vllm"
        result.best_configs = {"disagg_vllm": pd.DataFrame([disagg_row])}
        mock_recommend.return_value = result

        resp = client.post("/recommend", json=VALID_RECOMMEND_BODY)
        assert resp.status_code == 200
        data = resp.json()
        assert "disagg" in data["chosen_mode"]
        cfg = data["configs"][0]
        assert cfg["tp"] is None
        assert cfg["prefill_config"] is not None
        assert cfg["decode_config"] is not None
        assert cfg["prefill_config"]["tp"] == 1
        assert cfg["decode_config"]["tp"] == 1
        assert cfg["total_gpus_needed"] == 6
        assert cfg["memory"] == pytest.approx(129.8)

    @patch("tools.api_service.app.cli_recommend")
    def test_agg_result_has_no_prefill_decode(self, mock_recommend):
        mock_recommend.return_value = make_mock_cli_result()
        resp = client.post("/recommend", json=VALID_RECOMMEND_BODY)
        assert resp.status_code == 200
        cfg = resp.json()["configs"][0]
        assert cfg["tp"] == 2
        assert cfg["prefill_config"] is None
        assert cfg["decode_config"] is None


# ─── /estimate tests ──────────────────────────────────────────────────────────

MOCK_ESTIMATE_RESULT_RAW = {
    "model": "Qwen/Qwen3-32B", "isl": 4000, "osl": 1000,
    "ttft": 471.378, "tpot": 28.118, "request_latency": 28561.14,
    "bs": np.int64(128), "global_bs": np.int64(128),
    "tokens/s": 1678.925, "tokens/s/gpu": 839.462, "tokens/s/user": 35.565,
    "num_total_gpus": np.int64(2),
    "tp": np.int64(2), "pp": np.int64(1), "dp": np.int64(1),
    "memory": 64.044,
    "backend": "vllm", "version": "0.24.0", "system": "h200_sxm",
    "gemm": "bfloat16", "kvcache": "bfloat16",
    "power_w": 0.0,
}

VALID_ESTIMATE_BODY = {
    "model_path": "Qwen/Qwen3-32B",
    "system": "h200_sxm",
    "backend": "vllm",
    "isl": 4000,
    "osl": 1000,
    "tp_size": 2,
    "batch_size": 128,
}


def make_mock_estimate_result():
    mock = MagicMock()
    mock.ttft = 471.378
    mock.tpot = 28.118
    mock.power_w = 0.0
    mock.raw = MOCK_ESTIMATE_RESULT_RAW
    return mock


class TestEstimate:

    @patch("tools.api_service.app.cli_estimate")
    def test_success(self, mock_estimate):
        mock_estimate.return_value = make_mock_estimate_result()
        resp = client.post("/estimate", json=VALID_ESTIMATE_BODY)
        assert resp.status_code == 200
        data = resp.json()
        assert data["ttft"] == pytest.approx(471.378)
        assert data["tpot"] == pytest.approx(28.118)
        assert data["tokens_per_second"] == pytest.approx(1678.925)
        assert data["tp"] == 2
        assert data["serving_config"] is None
        assert data["memory_breakdown"] is None

    @patch("tools.api_service.app.cli_estimate")
    def test_include_config(self, mock_estimate):
        mock_estimate.return_value = make_mock_estimate_result()
        resp = client.post("/estimate?include=config", json=VALID_ESTIMATE_BODY)
        assert resp.status_code == 200
        sc = resp.json()["serving_config"]
        assert sc is not None
        assert sc["tensor_parallel_size"] == 2

    @patch("tools.api_service.app.estimate_kv_cache")
    @patch("tools.api_service.app.cli_estimate")
    def test_include_memory(self, mock_estimate, mock_kv):
        mock_estimate.return_value = make_mock_estimate_result()
        mock_kv.return_value = MOCK_KV_CACHE_RESULT
        resp = client.post("/estimate?include=memory", json=VALID_ESTIMATE_BODY)
        assert resp.status_code == 200
        mb = resp.json()["memory_breakdown"]
        assert mb is not None
        assert mb["weights_bytes"] == 32761446400

    @patch("tools.api_service.app.estimate_kv_cache")
    @patch("tools.api_service.app.cli_estimate")
    def test_include_config_and_memory(self, mock_estimate, mock_kv):
        mock_estimate.return_value = make_mock_estimate_result()
        mock_kv.return_value = MOCK_KV_CACHE_RESULT
        resp = client.post("/estimate?include=config,memory", json=VALID_ESTIMATE_BODY)
        assert resp.status_code == 200
        assert resp.json()["serving_config"] is not None
        assert resp.json()["memory_breakdown"] is not None

    @patch("tools.api_service.app.cli_estimate")
    def test_calls_sdk_with_correct_params(self, mock_estimate):
        mock_estimate.return_value = make_mock_estimate_result()
        client.post("/estimate", json=VALID_ESTIMATE_BODY)
        kwargs = mock_estimate.call_args.kwargs
        assert kwargs["system_name"] == "h200_sxm"
        assert kwargs["backend_name"] == "vllm"
        assert kwargs["tp_size"] == 2
        assert kwargs["batch_size"] == 128

    def test_requires_model_path(self):
        body = {**VALID_ESTIMATE_BODY}
        del body["model_path"]
        resp = client.post("/estimate", json=body)
        assert resp.status_code == 422

    def test_requires_system(self):
        body = {**VALID_ESTIMATE_BODY}
        del body["system"]
        resp = client.post("/estimate", json=body)
        assert resp.status_code == 422

    @patch("tools.api_service.app.cli_estimate")
    def test_value_error_returns_422(self, mock_estimate):
        mock_estimate.side_effect = ValueError("unsupported model/backend/GPU for estimation")
        resp = client.post("/estimate", json=VALID_ESTIMATE_BODY)
        assert resp.status_code == 422

    @patch("tools.api_service.app.cli_estimate")
    def test_unexpected_error_returns_500(self, mock_estimate):
        mock_estimate.side_effect = RuntimeError("crash")
        resp = client.post("/estimate", json=VALID_ESTIMATE_BODY)
        assert resp.status_code == 500

    @patch("tools.api_service.app.cli_estimate")
    def test_model_config_accepted(self, mock_estimate):
        mock_estimate.return_value = make_mock_estimate_result()
        body = {**VALID_ESTIMATE_BODY, "model_config": {"hidden_size": 8192, "architectures": ["LlamaForCausalLM"]}}
        resp = client.post("/estimate", json=body)
        assert resp.status_code == 200

    @patch("tools.api_service.app.cli_estimate")
    def test_inclusive_tpot(self, mock_estimate):
        mock_estimate.return_value = make_mock_estimate_result()
        body = {**VALID_ESTIMATE_BODY, "inclusive_tpot": True}
        resp = client.post("/estimate", json=body)
        assert resp.status_code == 200
        data = resp.json()
        # inclusive = (ttft + tpot * (osl - 1)) / osl
        # = (471.378 + 28.118 * 999) / 1000
        expected = (471.378 + 28.118 * (1000 - 1)) / 1000
        assert data["tpot"] == pytest.approx(expected)
        assert data["ttft"] == pytest.approx(471.378)

    @patch("tools.api_service.app.cli_estimate")
    def test_inclusive_tpot_default_false(self, mock_estimate):
        mock_estimate.return_value = make_mock_estimate_result()
        resp = client.post("/estimate", json=VALID_ESTIMATE_BODY)
        assert resp.json()["tpot"] == pytest.approx(28.118)


# ─── model_config passthrough tests ──────────────────────────────────────────

class TestModelConfigPassthrough:

    @patch("tools.api_service.app.cli_recommend")
    def test_recommend_accepts_model_config(self, mock_recommend):
        mock_recommend.return_value = make_mock_cli_result()
        body = {**VALID_RECOMMEND_BODY, "model_config": {"hidden_size": 8192, "architectures": ["LlamaForCausalLM"]}}
        resp = client.post("/recommend", json=body)
        assert resp.status_code == 200

    @patch("tools.api_service.app.estimate_kv_cache")
    def test_memory_accepts_model_config(self, mock_kv):
        mock_kv.return_value = MOCK_KV_CACHE_RESULT
        body = {**VALID_MEMORY_BODY, "model_config": {"hidden_size": 8192, "architectures": ["LlamaForCausalLM"]}}
        resp = client.post("/memory", json=body)
        assert resp.status_code == 200
