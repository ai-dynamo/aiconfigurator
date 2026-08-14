# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Two-sided schema contract for the retained ``moe_a2a`` table."""

import ast
from pathlib import Path

import pandas as pd
import pytest

from aiconfigurator_core.sdk.operations.moe_comm import load_moe_a2a_data

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
MOE_A2A_HEADER = (
    "framework,version,device,op_name,kernel_source,"
    "comm_backend,phase,comm_dtype,ep_size,node_num,hidden_size,topk,num_experts,"
    "num_tokens,sms,transmit_us,notify_us,latency"
)
_TWIN_PINS = {
    "tests/unit/collector/test_collect_moe_a2a.py": MOE_A2A_HEADER,
    "tests/unit/collector/test_collect_trtllm_alltoall.py": MOE_A2A_HEADER,
}


def test_collector_twins_pin_the_same_header():
    for relative_path, header in _TWIN_PINS.items():
        source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        module = ast.parse(source)
        assignment = next(
            node
            for node in module.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "MOE_A2A_HEADER" for target in node.targets)
        )
        assert ast.literal_eval(assignment.value) == header


def test_moe_a2a_header_row_loads_with_us_converted_to_ms(tmp_path):
    row = {
        "framework": "SGLang",
        "version": "0.5.12",
        "device": "NVIDIA H200",
        "op_name": "moe_a2a",
        "kernel_source": "deepep",
        "comm_backend": "deepep_ht",
        "phase": "dispatch",
        "comm_dtype": "default",
        "ep_size": 8,
        "node_num": 2,
        "hidden_size": 7168,
        "topk": 8,
        "num_experts": 256,
        "num_tokens": 4096,
        "sms": 24,
        "transmit_us": 700.0,
        "notify_us": 150.0,
        "latency": 850.0,
    }
    columns = MOE_A2A_HEADER.split(",")
    path = tmp_path / "moe_a2a_perf.parquet"
    pd.DataFrame([[row[column] for column in columns]], columns=columns).to_parquet(path, index=False)

    data = load_moe_a2a_data([(str(path), None)])
    leaf = data["deepep_ht"]["dispatch"]["default"][8][2][7168][8][256][24][4096]
    assert leaf["latency"] == pytest.approx(0.850)
