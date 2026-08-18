# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DSA module data-loading tests.

The query-side behaviour this file used to pin (silicon backend/prefix-axis
routing, topk+1 crossings, SOL formulas and monotonicity, empirical raw-head
grids, sequence-overflow util holds, typed error surfaces) retired to the
compiled engine with #1357 PR-5; it is anchored by
tests/cross_package/test_query_shim_baseline.py and the frozen parity
goldens. The loaders and their table structure stay Python-owned and tested
here.
"""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.operations.dsa import (
    DEFAULT_DSA_ARCHITECTURE,
    load_context_dsa_module_data,
    load_generation_dsa_module_data,
)

pytestmark = pytest.mark.unit

GLM5_ARCHITECTURE = "GlmMoeDsaForCausalLM"


def _dsa_value(latency: float) -> dict[str, float]:
    return {"latency": latency, "power": 10.0, "energy": latency * 10.0}


class TestContextDSALoaders:
    def test_glm5_context_loader_requires_step_column(self, tmp_path):
        data_path = tmp_path / "dsa_context_module_perf.txt"
        data_path.write_text(
            "architecture,gemm_type,mla_dtype,kv_cache_dtype,num_heads,batch_size,isl,latency\n"
            f"{GLM5_ARCHITECTURE},bfloat16,bfloat16,bfloat16,32,1,256,10.0\n"
        )

        with pytest.raises(ValueError, match="requires a non-empty step column"):
            load_context_dsa_module_data(str(data_path))

    def test_glm5_context_loader_accepts_numeric_zero_step(self, tmp_path):
        data_path = tmp_path / "dsa_context_module_perf.parquet"
        table = pa.table(
            {
                "architecture": [GLM5_ARCHITECTURE],
                "gemm_type": ["bfloat16"],
                "mla_dtype": ["bfloat16"],
                "kv_cache_dtype": ["bfloat16"],
                "num_heads": [32],
                "batch_size": [1],
                "isl": [256],
                "step": [0],
                "latency": [10.0],
            }
        )
        pq.write_table(table, data_path)

        data = load_context_dsa_module_data(str(data_path))

        value = data[common.FMHAQuantMode.bfloat16][common.KVCacheQuantMode.bfloat16][common.GEMMQuantMode.bfloat16][
            GLM5_ARCHITECTURE
        ]["flashmla_kv"][32][0][256][1]
        assert value["latency"] == pytest.approx(10.0)

    def test_default_context_loader_treats_whitespace_step_as_missing(self, tmp_path):
        data_path = tmp_path / "dsa_context_module_perf.txt"
        data_path.write_text(
            "architecture,gemm_type,mla_dtype,kv_cache_dtype,num_heads,batch_size,isl,step,latency\n"
            f"{DEFAULT_DSA_ARCHITECTURE},bfloat16,bfloat16,bfloat16,32,1,256,  ,10.0\n"
        )

        data = load_context_dsa_module_data(str(data_path))

        value = data[common.FMHAQuantMode.bfloat16][common.KVCacheQuantMode.bfloat16][common.GEMMQuantMode.bfloat16][
            DEFAULT_DSA_ARCHITECTURE
        ]["flashmla_kv"][32][0][256][1]
        assert value["latency"] == pytest.approx(10.0)

    def test_context_loader_keeps_first_source_on_coordinate_conflict(self, tmp_path):
        header = (
            "architecture,kernel_source,gemm_type,mla_dtype,kv_cache_dtype,"
            "num_heads,batch_size,isl,step,latency,power\n"
        )
        active = tmp_path / "active_context.txt"
        fallback = tmp_path / "fallback_context.txt"
        active.write_text(
            header
            + f"{DEFAULT_DSA_ARCHITECTURE},default,bfloat16,bfloat16,bfloat16,32,1,256,0,7.0,10.0\n"
            + f"{DEFAULT_DSA_ARCHITECTURE},default,bfloat16,bfloat16,bfloat16,32,1,256,0,10.0,10.0\n"
        )
        fallback.write_text(
            header
            + f"{DEFAULT_DSA_ARCHITECTURE},default,bfloat16,bfloat16,bfloat16,32,1,256,0,99.0,10.0\n"
            + f"{DEFAULT_DSA_ARCHITECTURE},default,bfloat16,bfloat16,bfloat16,32,2,512,0,20.0,10.0\n"
        )

        data = load_context_dsa_module_data([(str(active), None), (str(fallback), {"default"})])
        head_data = data[common.FMHAQuantMode.bfloat16][common.KVCacheQuantMode.bfloat16][
            common.GEMMQuantMode.bfloat16
        ][DEFAULT_DSA_ARCHITECTURE]["flashmla_kv"][32][0]

        assert head_data[256][1] == _dsa_value(10.0)
        assert head_data[512][2] == _dsa_value(20.0)


class TestGenerationDSALoaders:
    def test_generation_loader_indexes_total_decode_length(self, tmp_path):
        data_path = tmp_path / "dsa_generation_module_perf.parquet"
        table = pa.table(
            {
                "architecture": [DEFAULT_DSA_ARCHITECTURE],
                "gemm_type": ["bfloat16"],
                "mla_dtype": ["bfloat16"],
                "kv_cache_dtype": ["bfloat16"],
                "num_heads": [32],
                "batch_size": [1],
                "isl": [1],
                "tp_size": [1],
                "step": [149],
                "latency": [20.0],
                "power": [10.0],
            }
        )
        pq.write_table(table, data_path)

        data = load_generation_dsa_module_data(str(data_path))

        assert data[common.KVCacheQuantMode.bfloat16][common.GEMMQuantMode.bfloat16][DEFAULT_DSA_ARCHITECTURE][
            "flashmla_kv"
        ][32][1][150] == _dsa_value(20.0)

    def test_generation_loader_keeps_first_source_on_total_sequence_conflict(self, tmp_path):
        header = (
            "architecture,kernel_source,gemm_type,mla_dtype,kv_cache_dtype,"
            "num_heads,batch_size,isl,step,latency,power\n"
        )
        active = tmp_path / "active_generation.txt"
        fallback = tmp_path / "fallback_generation.txt"
        active.write_text(
            header
            + f"{DEFAULT_DSA_ARCHITECTURE},default,bfloat16,bfloat16,bfloat16,32,1,1,149,7.0,10.0\n"
            + f"{DEFAULT_DSA_ARCHITECTURE},default,bfloat16,bfloat16,bfloat16,32,1,2,148,10.0,10.0\n"
        )
        fallback.write_text(
            header
            # Different isl/step decomposition, same indexed total sequence.
            + f"{DEFAULT_DSA_ARCHITECTURE},default,bfloat16,bfloat16,bfloat16,32,1,2,148,99.0,10.0\n"
            + f"{DEFAULT_DSA_ARCHITECTURE},default,bfloat16,bfloat16,bfloat16,32,2,1,150,20.0,10.0\n"
        )

        data = load_generation_dsa_module_data([(str(active), None), (str(fallback), {"default"})])
        head_data = data[common.KVCacheQuantMode.bfloat16][common.GEMMQuantMode.bfloat16][DEFAULT_DSA_ARCHITECTURE][
            "flashmla_kv"
        ][32]

        assert head_data[1][150] == _dsa_value(10.0)
        assert head_data[2][151] == _dsa_value(20.0)


# NOTE(#1357 PR-5 review): the GLM-5 CP sparse cluster (_query_cp /
# _load_glm5_sparse / _lookup_2d) was production-orphaned Python latency
# math and is deleted; its tests retired with it (single-oracle rule).
