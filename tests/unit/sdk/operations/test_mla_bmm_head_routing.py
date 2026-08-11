# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact-head-first routing for the mla_bmm tables.

Every dataset carries the DeepSeek power-of-two head slices; exact rows for
non-pow2 shards (Kimi-K3's 96/48/24/12) exist only where re-collected (b200
sglang today). ``MLABmm._query_mla_bmm_table`` must use the exact slice at
scale 1.0 when it has rows and otherwise reroute to the next-pow2 slice
scaled linearly by the head ratio. The Rust twin lives in
``operators/mla.rs::resolve_bmm_slice_heads``.
"""

import pytest

from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk.operations.mla import MLABmm
from aiconfigurator_core.sdk.performance_result import PerformanceResult

pytestmark = pytest.mark.unit

BF16 = common.GEMMQuantMode.bfloat16
TOKENS = (1, 16, 64, 256)


class _LoadedTable(dict):
    loaded = True

    def raise_if_not_loaded(self):
        pass


class _StubDatabase:
    def __init__(self, heads_to_latency):
        self.system_spec = {"gpu": {"bfloat16_tc_flops": 2e15, "mem_bw": 8e12}}
        self._mla_bmm_data = _LoadedTable(
            {
                BF16: {
                    "mla_gen_pre": {
                        heads: {t: {"latency": latency, "power": 0.0, "energy": 2.0 * latency} for t in TOKENS}
                        for heads, latency in heads_to_latency.items()
                    }
                }
            }
        )

    @staticmethod
    def _interp_pr(latency, energy=0.0):
        return PerformanceResult(latency, energy=energy, source="silicon")

    def _query_silicon_or_hybrid(self, get_silicon, get_empirical, database_mode, error_msg):
        return get_silicon()


@pytest.fixture(autouse=True)
def _no_disk_load(monkeypatch):
    monkeypatch.setattr(MLABmm, "load_data", classmethod(lambda cls, database: None))


def _query(db, num_heads):
    return MLABmm._query_mla_bmm_table(db, 64, num_heads, BF16, if_pre=True, database_mode=common.DatabaseMode.SILICON)


def test_exact_head_rows_win_at_scale_one():
    # 96-head exact rows present: used as-is; the pow2 slice must not leak in.
    db = _StubDatabase({96: 0.5, 128: 999.0})
    result = _query(db, 96)
    assert float(result) == pytest.approx(0.5)
    assert result.source == "silicon"


def test_missing_exact_reroutes_to_next_pow2_with_head_ratio():
    db = _StubDatabase({128: 0.8})
    result = _query(db, 96)
    assert float(result) == pytest.approx(0.8 * 96 / 128)
    assert result.energy == pytest.approx(2.0 * 0.8 * 96 / 128)
    assert result.source == "silicon"
