# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
from pathlib import Path

import pytest

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


def test_query_layerwise_detail_selects_max_num_batched_tokens(tmp_path):
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
                "latency_ms,latency_source,max_num_batched_tokens",
                "vLLM,0.20.1,test,Qwen/Qwen3-32B,CTX,8,1,128,0,15.0,schedule_to_update,",
                "vLLM,0.20.1,test,Qwen/Qwen3-32B,CTX,8,1,128,0,9.0,schedule_to_update,2048",
                "",
            ]
        )
    )
    db = PerfDatabase("test_system", "vllm", "0.20.1", systems_root=str(systems_root))

    default_detail = db.query_layerwise_detail("qwen/qwen3-32b", "CTX", 8, 1, 128)
    chunked_detail = db.query_layerwise_detail(
        "qwen/qwen3-32b",
        "CTX",
        8,
        1,
        128,
        max_num_batched_tokens=2048,
    )

    assert default_detail["latency"] == pytest.approx(15.0)
    assert "max_num_batched_tokens" not in default_detail
    assert chunked_detail["latency"] == pytest.approx(9.0)
    assert chunked_detail["max_num_batched_tokens"] == pytest.approx(2048)


def test_query_layerwise_detail_selects_max_num_seqs_for_decode(tmp_path):
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
                "latency_ms,latency_source,max_num_seqs",
                "vLLM,0.20.1,test,Qwen/Qwen3-32B,GEN,4,1,1,4096,2.3,schedule_to_update,256",
                "vLLM,0.20.1,test,Qwen/Qwen3-32B,GEN,4,1,1,4096,3.6,schedule_to_update,128",
                "",
            ]
        )
    )
    db = PerfDatabase("test_system", "vllm", "0.20.1", systems_root=str(systems_root))

    default_detail = db.query_layerwise_detail("qwen/qwen3-32b", "GEN", 4, 1, 4096)
    maxseq_detail = db.query_layerwise_detail(
        "qwen/qwen3-32b",
        "GEN",
        4,
        1,
        4096,
        max_num_seqs=128,
    )

    assert default_detail["latency"] == pytest.approx(2.3)
    assert default_detail["max_num_seqs"] == pytest.approx(256)
    assert maxseq_detail["latency"] == pytest.approx(3.6)
    assert maxseq_detail["max_num_seqs"] == pytest.approx(128)


def test_query_layerwise_detail_smooths_generation_grid_before_interpolation(tmp_path):
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
                "latency_ms,latency_source",
                "vLLM,0.20.1,test,Qwen/Qwen3-32B,GEN,4,8,1,4096,10.0,schedule_to_update",
                "vLLM,0.20.1,test,Qwen/Qwen3-32B,GEN,4,16,1,4096,80.0,schedule_to_update",
                "vLLM,0.20.1,test,Qwen/Qwen3-32B,GEN,4,32,1,4096,11.0,schedule_to_update",
                "",
            ]
        )
    )
    db = PerfDatabase("test_system", "vllm", "0.20.1", systems_root=str(systems_root))

    exact_detail = db.query_layerwise_detail("qwen/qwen3-32b", "GEN", 4, 16, 4096)
    interpolated_detail = db.query_layerwise_detail("qwen/qwen3-32b", "GEN", 4, 13, 4096)

    assert exact_detail["latency"] == pytest.approx(10.5)
    assert interpolated_detail["latency"] == pytest.approx(10.3125)


def test_query_layerwise_detail_selects_context_batch_size_when_available(tmp_path):
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
                "framework,framework_version,system,model,phase,attn_tp,moe_tp,ep,batch_size,new_tokens,past_kv,"
                "latency_ms,latency_source,max_num_batched_tokens,moe_weight_mode",
                "vLLM,0.20.1,test,Qwen/Qwen3.6-35B-A3B,ctx,2,2,1,1,2640,0,14.0,schedule_to_update,2048,noop",
                "vLLM,0.20.1,test,Qwen/Qwen3.6-35B-A3B,ctx,2,2,1,3,2640,0,45.0,schedule_to_update,2048,noop",
                "",
            ]
        )
    )
    db = PerfDatabase("test_system", "vllm", "0.20.1", systems_root=str(systems_root))

    single_detail = db.query_layerwise_detail(
        "Qwen/Qwen3.6-35B-A3B",
        "CTX",
        2,
        1,
        2640,
        moe_weight_mode="noop",
        max_num_batched_tokens=2048,
        moe_tp_size=2,
        moe_ep_size=1,
    )
    batch_detail = db.query_layerwise_detail(
        "Qwen/Qwen3.6-35B-A3B",
        "CTX",
        2,
        3,
        2640,
        moe_weight_mode="noop",
        max_num_batched_tokens=2048,
        moe_tp_size=2,
        moe_ep_size=1,
    )

    assert single_detail["latency"] == pytest.approx(14.0)
    assert batch_detail["latency"] == pytest.approx(45.0)


def test_query_layerwise_detail_selects_moe_parallelism(tmp_path):
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
                "framework,framework_version,system,model,phase,attn_tp,moe_tp,ep,batch_size,new_tokens,past_kv,"
                "latency_ms,latency_source,max_num_batched_tokens,moe_weight_mode",
                "vLLM,0.20.1,test,Qwen/Qwen3.6-35B-A3B,ctx,2,2,1,1,128,0,14.0,schedule_to_update,8192,noop",
                "vLLM,0.20.1,test,Qwen/Qwen3.6-35B-A3B,ctx,2,1,2,1,128,0,22.0,schedule_to_update,8192,noop",
                "",
            ]
        )
    )
    db = PerfDatabase("test_system", "vllm", "0.20.1", systems_root=str(systems_root))

    merged_detail = db.query_layerwise_detail(
        "Qwen/Qwen3.6-35B-A3B",
        "CTX",
        2,
        1,
        128,
        moe_weight_mode="noop",
        max_num_batched_tokens=8192,
    )
    tp_only_detail = db.query_layerwise_detail(
        "Qwen/Qwen3.6-35B-A3B",
        "CTX",
        2,
        1,
        128,
        moe_weight_mode="noop",
        max_num_batched_tokens=8192,
        moe_tp_size=2,
        moe_ep_size=1,
    )
    ep_detail = db.query_layerwise_detail(
        "Qwen/Qwen3.6-35B-A3B",
        "CTX",
        2,
        1,
        128,
        moe_weight_mode="noop",
        max_num_batched_tokens=8192,
        moe_tp_size=1,
        moe_ep_size=2,
    )

    assert merged_detail["latency"] == pytest.approx(22.0)
    assert tp_only_detail["latency"] == pytest.approx(14.0)
    assert ep_detail["latency"] == pytest.approx(22.0)


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
