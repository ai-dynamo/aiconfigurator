# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for sflow input parsing (JSON and CSV formats)."""

import json
import os

import pytest

from aiconfigurator.cli.sflow_input import (
    _build_worker_params,
    load_sflow_input,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def disagg_json_data():
    """Sample disagg JSON input matching the real-world format."""
    return {
        "E8": {
            "model": "deepseek-ai/DeepSeek-R1",
            "isl": 8192,
            "osl": 1024,
            "ttft": 431.384,
            "tpot": 24.918,
            "concurrency": 1984,
            "num_total_gpus": 48,
            "prefill": {
                "bs": 1,
                "workers": 8,
                "tp": 1,
                "pp": 1,
                "dp": 4,
                "moe_tp": 1,
                "moe_ep": 4,
                "gemm": "nvfp4",
                "kvcache": "fp8",
                "fmha": "fp8",
                "moe": "nvfp4",
                "version": "1.2.0rc6",
                "system": "gb200",
            },
            "decode": {
                "bs": 124,
                "workers": 1,
                "tp": 1,
                "pp": 1,
                "dp": 16,
                "moe_tp": 1,
                "moe_ep": 16,
                "gemm": "nvfp4",
                "kvcache": "fp8",
                "fmha": "fp8",
                "moe": "nvfp4",
                "version": "1.2.0rc6",
                "system": "gb200",
            },
        }
    }


@pytest.fixture
def agg_json_data():
    """Sample agg JSON input with explicit 'agg' key."""
    return {
        "config1": {
            "model": "Qwen/Qwen3-32B",
            "backend": "trtllm",
            "isl": 2000,
            "osl": 500,
            "ttft": 80427.6,
            "tpot": 168.9,
            "concurrency": 288,
            "num_total_gpus": 2,
            "agg": {
                "bs": 144,
                "workers": 1,
                "tp": 2,
                "pp": 1,
                "dp": 1,
                "moe_tp": 0,
                "moe_ep": 0,
                "version": "1.3.0rc1",
                "system": "gb200",
            },
        }
    }


def _write_json(data, tmp_path):
    path = os.path.join(str(tmp_path), "input.json")
    with open(path, "w") as f:
        json.dump(data, f)
    return path


def _write_csv(content, tmp_path):
    path = os.path.join(str(tmp_path), "input.csv")
    with open(path, "w") as f:
        f.write(content)
    return path


# ---------------------------------------------------------------------------
# JSON disagg tests
# ---------------------------------------------------------------------------


class TestJsonDisaggParsing:
    def test_basic_disagg(self, tmp_path, disagg_json_data):
        path = _write_json(disagg_json_data, tmp_path)
        configs = load_sflow_input(path)

        assert len(configs) == 1
        cfg = configs[0]
        assert cfg.name == "E8"
        assert cfg.serving_mode == "disagg"
        assert cfg.model == "deepseek-ai/DeepSeek-R1"
        assert cfg.system == "gb200"
        assert cfg.isl == 8192
        assert cfg.osl == 1024
        assert cfg.ttft == 431.384
        assert cfg.tpot == 24.918
        assert cfg.concurrency == 1984
        assert cfg.num_total_gpus == 48
        assert cfg.backend_version == "1.2.0rc6"

    def test_disagg_prefill_params(self, tmp_path, disagg_json_data):
        path = _write_json(disagg_json_data, tmp_path)
        cfg = load_sflow_input(path)[0]

        assert cfg.prefill_workers == 8
        p = cfg.prefill_params
        assert p["tensor_parallel_size"] == 1
        assert p["pipeline_parallel_size"] == 1
        assert p["data_parallel_size"] == 4
        assert p["moe_tensor_parallel_size"] == 1
        assert p["moe_expert_parallel_size"] == 4
        assert p["max_batch_size"] == 1
        assert p["gemm_quant_mode"] == "nvfp4"
        assert p["kvcache_quant_mode"] == "fp8"
        assert p["kv_cache_dtype"] == "fp8"
        assert p["fmha_quant_mode"] == "fp8"
        assert p["moe_quant_mode"] == "nvfp4"
        assert p["gpus_per_worker"] == 4  # tp * pp * dp

    def test_disagg_decode_params(self, tmp_path, disagg_json_data):
        path = _write_json(disagg_json_data, tmp_path)
        cfg = load_sflow_input(path)[0]

        assert cfg.decode_workers == 1
        d = cfg.decode_params
        assert d["tensor_parallel_size"] == 1
        assert d["data_parallel_size"] == 16
        assert d["max_batch_size"] == 124
        assert d["gpus_per_worker"] == 16

    def test_disagg_no_agg_params(self, tmp_path, disagg_json_data):
        path = _write_json(disagg_json_data, tmp_path)
        cfg = load_sflow_input(path)[0]
        assert cfg.agg_params is None
        assert cfg.agg_workers == 0

    def test_multiple_configs(self, tmp_path, disagg_json_data):
        data = dict(disagg_json_data)
        data["E12"] = dict(disagg_json_data["E8"])
        data["E12"]["concurrency"] = 2048
        path = _write_json(data, tmp_path)
        configs = load_sflow_input(path)
        assert len(configs) == 2
        names = {c.name for c in configs}
        assert names == {"E8", "E12"}


# ---------------------------------------------------------------------------
# JSON agg tests
# ---------------------------------------------------------------------------


class TestJsonAggParsing:
    def test_basic_agg_with_key(self, tmp_path, agg_json_data):
        path = _write_json(agg_json_data, tmp_path)
        configs = load_sflow_input(path)

        assert len(configs) == 1
        cfg = configs[0]
        assert cfg.name == "config1"
        assert cfg.serving_mode == "agg"
        assert cfg.model == "Qwen/Qwen3-32B"
        assert cfg.backend == "trtllm"
        assert cfg.system == "gb200"
        assert cfg.backend_version == "1.3.0rc1"

    def test_agg_params(self, tmp_path, agg_json_data):
        path = _write_json(agg_json_data, tmp_path)
        cfg = load_sflow_input(path)[0]

        assert cfg.agg_workers == 1
        a = cfg.agg_params
        assert a["tensor_parallel_size"] == 2
        assert a["pipeline_parallel_size"] == 1
        assert a["max_batch_size"] == 144
        assert a["gpus_per_worker"] == 2

    def test_agg_no_disagg_params(self, tmp_path, agg_json_data):
        path = _write_json(agg_json_data, tmp_path)
        cfg = load_sflow_input(path)[0]
        assert cfg.prefill_params is None
        assert cfg.decode_params is None

    def test_default_backend(self, tmp_path, agg_json_data):
        """Backend defaults to trtllm when not specified."""
        del agg_json_data["config1"]["backend"]
        path = _write_json(agg_json_data, tmp_path)
        cfg = load_sflow_input(path, backend_default="sglang")[0]
        assert cfg.backend == "sglang"


# ---------------------------------------------------------------------------
# CSV agg tests
# ---------------------------------------------------------------------------


class TestCsvAggParsing:
    def test_basic_agg_csv(self, tmp_path):
        csv = "mode,model,isl,osl,tp,pp,bs,system,backend_version\nagg,Qwen/Qwen3-32B,2000,500,2,1,144,gb200,1.3.0rc1\n"
        path = _write_csv(csv, tmp_path)
        configs = load_sflow_input(path)

        assert len(configs) == 1
        cfg = configs[0]
        assert cfg.name == "row_0"
        assert cfg.serving_mode == "agg"
        assert cfg.model == "Qwen/Qwen3-32B"
        assert cfg.isl == 2000
        assert cfg.osl == 500
        a = cfg.agg_params
        assert a["tensor_parallel_size"] == 2
        assert a["max_batch_size"] == 144

    def test_csv_column_aliases(self, tmp_path):
        """Test that backend_name is aliased to backend, silicon_ttft_avg to ttft."""
        csv = (
            "mode,model,isl,osl,tp,pp,bs,system,backend_name,silicon_ttft_avg,silicon_tpot_avg\n"
            "agg,Qwen/Qwen3-32B,2000,500,2,1,144,gb200,trtllm,100.5,10.2\n"
        )
        path = _write_csv(csv, tmp_path)
        cfg = load_sflow_input(path)[0]
        assert cfg.backend == "trtllm"
        assert cfg.ttft == 100.5
        assert cfg.tpot == 10.2

    def test_csv_default_mode_is_agg(self, tmp_path):
        """Rows without mode column default to agg."""
        csv = "model,isl,osl,tp,pp,bs,system\nQwen/Qwen3-32B,2000,500,2,1,144,gb200\n"
        path = _write_csv(csv, tmp_path)
        cfg = load_sflow_input(path)[0]
        assert cfg.serving_mode == "agg"

    def test_csv_name_column(self, tmp_path):
        csv = "name,model,isl,osl,tp,pp,bs,system\nmyconfig,Qwen/Qwen3-32B,2000,500,2,1,144,gb200\n"
        path = _write_csv(csv, tmp_path)
        assert load_sflow_input(path)[0].name == "myconfig"

    def test_csv_multiple_rows(self, tmp_path):
        csv = (
            "model,isl,osl,tp,pp,bs,system\n"
            "Qwen/Qwen3-32B,2000,500,2,1,144,gb200\n"
            "Qwen/Qwen3-32B,4000,1000,4,1,64,h200_sxm\n"
        )
        path = _write_csv(csv, tmp_path)
        configs = load_sflow_input(path)
        assert len(configs) == 2


# ---------------------------------------------------------------------------
# CSV disagg tests
# ---------------------------------------------------------------------------


class TestCsvDisaggParsing:
    def test_disagg_csv_with_prefixed_columns(self, tmp_path):
        csv = (
            "mode,model,isl,osl,system,"
            "prefill_tp,prefill_pp,prefill_dp,prefill_bs,prefill_workers,prefill_gemm,prefill_kvcache,"
            "decode_tp,decode_pp,decode_dp,decode_bs,decode_workers,decode_gemm,decode_kvcache\n"
            "disagg,deepseek-ai/DeepSeek-R1,8192,1024,gb200,"
            "1,1,4,1,8,nvfp4,fp8,"
            "1,1,16,124,1,nvfp4,fp8\n"
        )
        path = _write_csv(csv, tmp_path)
        configs = load_sflow_input(path)

        assert len(configs) == 1
        cfg = configs[0]
        assert cfg.serving_mode == "disagg"
        assert cfg.prefill_workers == 8
        assert cfg.decode_workers == 1
        assert cfg.prefill_params["data_parallel_size"] == 4
        assert cfg.decode_params["data_parallel_size"] == 16
        assert cfg.decode_params["max_batch_size"] == 124


# ---------------------------------------------------------------------------
# _build_worker_params tests
# ---------------------------------------------------------------------------


class TestBuildWorkerParams:
    def test_basic_mapping(self):
        d = {"tp": 2, "pp": 1, "dp": 4, "moe_tp": 1, "moe_ep": 4, "bs": 32}
        params = _build_worker_params(d)
        assert params["tensor_parallel_size"] == 2
        assert params["pipeline_parallel_size"] == 1
        assert params["data_parallel_size"] == 4
        assert params["gpus_per_worker"] == 8  # 2 * 1 * 4
        assert params["moe_tensor_parallel_size"] == 1
        assert params["moe_expert_parallel_size"] == 4
        assert params["max_batch_size"] == 32

    def test_quant_modes(self):
        d = {"tp": 1, "pp": 1, "dp": 1, "bs": 1, "gemm": "nvfp4", "kvcache": "fp8", "fmha": "fp8", "moe": "nvfp4"}
        params = _build_worker_params(d)
        assert params["gemm_quant_mode"] == "nvfp4"
        assert params["kvcache_quant_mode"] == "fp8"
        assert params["kv_cache_dtype"] == "fp8"
        assert params["fmha_quant_mode"] == "fp8"
        assert params["moe_quant_mode"] == "nvfp4"

    def test_null_quant_modes_excluded(self):
        d = {"tp": 1, "pp": 1, "dp": 1, "bs": 1, "gemm": None, "kvcache": None}
        params = _build_worker_params(d)
        assert "gemm_quant_mode" not in params
        assert "kvcache_quant_mode" not in params

    def test_defaults_for_missing_fields(self):
        params = _build_worker_params({})
        assert params["tensor_parallel_size"] == 1
        assert params["pipeline_parallel_size"] == 1
        assert params["data_parallel_size"] == 1
        assert params["max_batch_size"] == 1
        assert params["gpus_per_worker"] == 1


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_sflow_input("/nonexistent/file.json")

    def test_unsupported_extension(self, tmp_path):
        path = os.path.join(str(tmp_path), "input.txt")
        with open(path, "w") as f:
            f.write("hello")
        with pytest.raises(ValueError, match="Unsupported input file extension"):
            load_sflow_input(path)

    def test_invalid_json_structure(self, tmp_path):
        path = os.path.join(str(tmp_path), "input.json")
        with open(path, "w") as f:
            json.dump([1, 2, 3], f)
        with pytest.raises(ValueError, match="top-level object"):
            load_sflow_input(path)
