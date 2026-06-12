# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import csv
from pathlib import Path

from aiconfigurator.sdk.perf_database import PerfDatabase

pytestmark = pytest.mark.unit


def test_query_layerwise_loads_csv_data(tmp_path):
    systems_root = tmp_path / "systems"
    data_dir = systems_root / "data" / "test_system" / "vllm" / "0.20.1"
    data_dir.mkdir(parents=True)
    (systems_root / "test_system.yaml").write_text(
        """
data_dir: data/test_system
gpu:
  mem_capacity: 1
node:
  num_gpus_per_node: 1
misc:
  nccl_mem: {1: 0}
  other_mem: 0
"""
    )
    (data_dir / "layerwise_perf.csv").write_text(
        "\n".join(
            [
                "framework,framework_version,system,model,phase,tp_size,batch_size,seq_len_q,seq_len_kv_cache,"
                "latency_ms,rms_latency_ms,rms_kernel_count,includes_moe,layer_type,layer_index,"
                "measured_layer_count,layer_multiplier",
                "vLLM,0.20.1,test,Qwen/Qwen3-32B,CTX,1,1,8192,0,7.00",
                "vLLM,0.20.1,test,Qwen/Qwen3-32B,CTX,1,1,8192,8192,9.00",
                "vLLM,0.20.1,test,Qwen/Qwen3-32B,GEN,1,4,1,1024,0.25,0.03,3,true,moe,0,1,64",
                "",
            ]
        )
    )
    db = PerfDatabase("test_system", "vllm", "0.20.1", systems_root=str(systems_root))

    assert float(db.query_layerwise("qwen/qwen3-32b", "GEN", 1, 4, 1024)) == pytest.approx(0.25)
    assert float(db.query_layerwise("qwen/qwen3-32b", "CTX", 1, 1, 8192)) == pytest.approx(7.00)
    assert float(db.query_layerwise("qwen/qwen3-32b", "CTX", 1, 1, 8192, seq_len_kv_cache=8192)) == pytest.approx(9.00)
    detail = db.query_layerwise_detail("qwen/qwen3-32b", "GEN", 1, 4, 1024)
    assert detail["latency"] == pytest.approx(0.25)
    assert detail["rms_latency"] == pytest.approx(0.03)
    assert detail["rms_kernel_count"] == pytest.approx(3)
    assert detail["includes_moe"] is True
    assert detail["layer_type"] == "moe"
    assert detail["measured_layer_count"] == pytest.approx(1)
    assert detail["layer_multiplier"] == pytest.approx(64)


def test_query_layerwise_detail_selects_moe_weight_mode(tmp_path):
    systems_root = tmp_path / "systems"
    data_dir = systems_root / "data" / "test_system" / "vllm" / "0.20.1"
    data_dir.mkdir(parents=True)
    (systems_root / "test_system.yaml").write_text(
        """
data_dir: data/test_system
gpu:
  mem_capacity: 1
node:
  num_gpus_per_node: 1
misc:
  nccl_mem: {1: 0}
  other_mem: 0
"""
    )
    (data_dir / "layerwise_perf.csv").write_text(
        "\n".join(
            [
                "framework,framework_version,system,model,phase,tp_size,batch_size,seq_len_q,seq_len_kv_cache,"
                "latency_ms,rms_latency_ms,rms_kernel_count,includes_moe,moe_weight_mode,layer_type,layer_index,"
                "measured_layer_count,layer_multiplier",
                "vLLM,0.20.1,test,deepseek-ai/DeepSeek-V4-Flash,CTX,2,1,128,0,21.0,0,0,false,noop,moe,0,43,43",
                "vLLM,0.20.1,test,deepseek-ai/DeepSeek-V4-Flash,CTX,2,1,128,0,27.0,0,0,true,dummy,moe,0,43,43",
                "",
            ]
        )
    )
    db = PerfDatabase("test_system", "vllm", "0.20.1", systems_root=str(systems_root))

    default_detail = db.query_layerwise_detail("deepseek-ai/DeepSeek-V4-Flash", "CTX", 2, 1, 128)
    noop_detail = db.query_layerwise_detail(
        "deepseek-ai/DeepSeek-V4-Flash",
        "CTX",
        2,
        1,
        128,
        moe_weight_mode="noop",
    )
    dummy_detail = db.query_layerwise_detail(
        "deepseek-ai/DeepSeek-V4-Flash",
        "CTX",
        2,
        1,
        128,
        moe_weight_mode="dummy",
    )

    assert default_detail["latency"] == pytest.approx(27.0)
    assert default_detail["moe_weight_mode"] == "dummy"
    assert noop_detail["latency"] == pytest.approx(21.0)
    assert noop_detail["moe_weight_mode"] == "noop"
    assert dummy_detail["latency"] == pytest.approx(27.0)
    assert dummy_detail["moe_weight_mode"] == "dummy"


def test_layerwise_loader_rejects_multi_gpu_physical_rows(tmp_path):
    systems_root = tmp_path / "systems"
    data_dir = systems_root / "data" / "test_system" / "vllm" / "0.20.1"
    data_dir.mkdir(parents=True)
    (systems_root / "test_system.yaml").write_text(
        """
data_dir: data/test_system
gpu:
  mem_capacity: 1
node:
  num_gpus_per_node: 1
misc:
  nccl_mem: {1: 0}
  other_mem: 0
"""
    )
    (data_dir / "layerwise_perf.csv").write_text(
        "\n".join(
            [
                "framework,framework_version,system,model,phase,tp_size,batch_size,seq_len_q,"
                "seq_len_kv_cache,latency_ms,physical_gpus",
                "vLLM,0.20.1,test,Qwen/Qwen3-32B,CTX,2,1,128,0,7.00,2",
                "",
            ]
        )
    )

    db = PerfDatabase("test_system", "vllm", "0.20.1", systems_root=str(systems_root))

    with pytest.raises(ValueError, match="one physical GPU per worker"):
        db.query_layerwise("qwen/qwen3-32b", "CTX", 2, 1, 128)


def test_installed_layerwise_data_uses_one_physical_gpu_per_worker():
    repo_root = Path(__file__).resolve().parents[4]
    layerwise_files = list((repo_root / "src" / "aiconfigurator" / "systems" / "data").rglob("layerwise_perf.csv"))
    violations: list[str] = []
    for path in layerwise_files:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "physical_gpus" not in reader.fieldnames:
                continue
            for line_number, row in enumerate(reader, start=2):
                value = row.get("physical_gpus")
                if value in (None, ""):
                    continue
                if float(value) > 1:
                    violations.append(f"{path}:{line_number}: physical_gpus={value}")

    assert violations == []
