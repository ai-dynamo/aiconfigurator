# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
                "framework,framework_version,system,model,phase,tp_size,batch_size,seq_len_q,seq_len_kv_cache,latency_ms",
                "vLLM,0.20.1,test,Qwen/Qwen3-32B,CTX,1,1,8192,0,7.00",
                "vLLM,0.20.1,test,Qwen/Qwen3-32B,CTX,1,1,8192,8192,9.00",
                "vLLM,0.20.1,test,Qwen/Qwen3-32B,GEN,1,4,1,1024,0.25",
                "",
            ]
        )
    )

    db = PerfDatabase("test_system", "vllm", "0.20.1", systems_root=str(systems_root))

    assert float(db.query_layerwise("qwen/qwen3-32b", "GEN", 1, 4, 1024)) == pytest.approx(0.25)
    assert float(db.query_layerwise("qwen/qwen3-32b", "CTX", 1, 1, 8192)) == pytest.approx(7.00)
    assert float(db.query_layerwise("qwen/qwen3-32b", "CTX", 1, 1, 8192, seq_len_kv_cache=8192)) == pytest.approx(9.00)
