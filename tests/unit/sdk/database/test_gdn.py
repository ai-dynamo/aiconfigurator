# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from aiconfigurator.sdk.operations.mamba import GDNKernel, load_gdn_data
from aiconfigurator.sdk.performance_result import PerformanceResult


class _LoadedGdnData(dict):
    loaded = True


class _FakeDatabase:
    def __init__(self, data):
        self._gdn_data = data
        self._extracted_metrics_cache = {}
        self.system_spec = {"gpu": {"mem_bw": 1_000.0}}

    @staticmethod
    def _interp_pr(latency, *, energy=0.0):
        return PerformanceResult(latency, energy=energy, source="silicon")


def _query_generation(database, *, kernel_source: str, num_v_heads: int = 32):
    return GDNKernel._query_gdn_table(
        database,
        phase="generation",
        kernel_source=kernel_source,
        batch_size=1,
        seq_len=None,
        d_model=4096,
        num_k_heads=16,
        head_k_dim=128,
        num_v_heads=num_v_heads,
        head_v_dim=128,
        d_conv=4,
    )


@pytest.mark.parametrize(
    ("stored_kernel_source", "queried_kernel_source"),
    [
        (
            "fused_recurrent_gated_delta_rule",
            "fused_sigmoid_gating_delta_rule_update",
        ),
        (
            "fused_sigmoid_gating_delta_rule_update",
            "fused_recurrent_gated_delta_rule",
        ),
    ],
)
def test_gdn_kernel_aliases_share_one_canonical_parquet_key(
    tmp_path,
    monkeypatch,
    stored_kernel_source,
    queried_kernel_source,
):
    perf_file = tmp_path / "gdn_perf.txt"
    perf_file.write_text(
        "framework,version,device,op_name,kernel_source,phase,batch_size,seq_len,num_tokens,"
        "d_model,d_conv,num_k_heads,head_k_dim,num_v_heads,head_v_dim,model_name,latency\n"
        f"SGLang,0.5.10,GPU,gdn,{stored_kernel_source},generation,1,1,1,"
        "4096,4,16,128,32,128,Qwen/Qwen3.5-9B,1.25\n",
        encoding="utf-8",
    )
    database = _FakeDatabase(_LoadedGdnData(load_gdn_data(str(perf_file))))
    monkeypatch.setattr(GDNKernel, "load_data", classmethod(lambda cls, db: None))

    result = _query_generation(
        database,
        kernel_source=queried_kernel_source,
    )

    assert float(result) == pytest.approx(1.25)
    assert result.source == "silicon"


def test_gdn_conv_sol_counts_only_key_channels(monkeypatch):
    unloaded = _LoadedGdnData()
    unloaded.loaded = False
    database = _FakeDatabase(unloaded)
    monkeypatch.setattr(GDNKernel, "load_data", classmethod(lambda cls, db: None))

    small_v = _query_generation(database, kernel_source="causal_conv1d_update", num_v_heads=16)
    large_v = _query_generation(database, kernel_source="causal_conv1d_update", num_v_heads=64)

    key_channels = 16 * 128
    expected = (key_channels * (4 + 1) * 2 + key_channels * 2) / 1_000.0 * 1_000.0
    assert float(small_v) == pytest.approx(expected)
    assert float(large_v) == pytest.approx(expected)
    assert small_v.source == "sol"
