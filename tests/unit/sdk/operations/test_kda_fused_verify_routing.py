# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KDA verify-phase kernel routing on fused-CuTeDSL datasets.

SM100 sglang serves DSPARK target-verify through the fused CuTeDSL kernel
(``fused_kda_decode_mtp_dspark``) — one row per verify step covering BOTH the
conv update and the chain-verify recurrence — so b200_sxm-style datasets carry
no Triton verify rows. ``KDAKernel._query_kda_table`` must route the
recurrence op onto the fused table and fold the conv op to zero, while
Triton-verify datasets (Hopper-style) and the vLLM physical kernels stay
untouched. The Rust twin lives in ``operators/mamba.rs::KdaOp::query``.
"""

import pytest

from aiconfigurator_core.sdk.operations.mamba import KDAKernel
from aiconfigurator_core.sdk.performance_result import PerformanceResult

pytestmark = pytest.mark.unit

MODEL_KEY = (7168, 48, 128, 48, 128, 4)
SHARD = dict(
    d_model=7168,
    num_k_heads=48,
    head_k_dim=128,
    num_v_heads=48,
    head_v_dim=128,
    d_conv=4,
)


class _LoadedTable(dict):
    loaded = True


class _StubDatabase:
    def __init__(self, kda_data):
        self.system_spec = {"gpu": {"mem_bw": 8000}}
        self._kda_data = _LoadedTable(kda_data)

    @staticmethod
    def _interp_pr(latency, energy=0.0):
        return PerformanceResult(latency, energy=energy, source="silicon")


def _verify_grid(latency):
    # Exact grid hits at (batch, draft) so interpolation is the identity.
    entry = {"latency": latency, "power": 0.0, "energy": 0.0}
    return {MODEL_KEY: {b: dict.fromkeys((2, 4, 8), entry) for b in (1, 4, 16, 64)}}


def _query(db, kernel_source):
    return KDAKernel._query_kda_table(
        db,
        phase="verify",
        kernel_source=kernel_source,
        batch_size=16,
        seq_len=4,
        **SHARD,
    )


@pytest.fixture(autouse=True)
def _no_disk_load(monkeypatch):
    monkeypatch.setattr(KDAKernel, "load_data", classmethod(lambda cls, database: None))


def test_fused_dataset_routes_recurrence_and_folds_conv_to_zero():
    db = _StubDatabase({"fused_kda_decode_mtp_dspark": {"verify": _verify_grid(0.5)}})
    recurrence = _query(db, "fused_sigmoid_gating_delta_rule_update")
    assert float(recurrence) == pytest.approx(0.5)
    assert recurrence.source == "silicon"
    conv = _query(db, "causal_conv1d_update")
    assert float(conv) == 0.0
    assert conv.source == "silicon"


def test_triton_dataset_keeps_physical_kernels():
    db = _StubDatabase(
        {
            "fused_sigmoid_gating_delta_rule_update": {"verify": _verify_grid(0.3)},
            "causal_conv1d_update": {"verify": _verify_grid(0.1)},
        }
    )
    assert float(_query(db, "fused_sigmoid_gating_delta_rule_update")) == pytest.approx(0.3)
    assert float(_query(db, "causal_conv1d_update")) == pytest.approx(0.1)


def test_vllm_verify_kernel_is_never_rerouted():
    # A fused sglang table must not capture the vLLM chain-verify kernel.
    db = _StubDatabase({"fused_kda_decode_mtp_dspark": {"verify": _verify_grid(0.5)}})
    result = _query(db, "fused_recurrent_kda")
    assert result.source == "sol"


def _generation_grid(latency):
    entry = {"latency": latency, "power": 0.0, "energy": 0.0}
    return {MODEL_KEY: dict.fromkeys((1, 4, 16, 64), entry)}


def _query_gen(db, kernel_source, shard=None):
    return KDAKernel._query_kda_table(
        db,
        phase="generation",
        kernel_source=kernel_source,
        batch_size=16,
        seq_len=None,
        **(shard or SHARD),
    )


def test_fused_decode_shard_routes_per_model_key():
    # The 12-head TP8 shard ships one fused generation row and no Triton pair;
    # other shards keep the Triton rows. Routing is per model key and must win
    # over the nearest-shard fallback.
    other_key = (7168, 24, 128, 24, 128, 4)
    other_shard = dict(SHARD, num_k_heads=24, num_v_heads=24)
    db = _StubDatabase(
        {
            "kda_fused_decode": {"generation": _generation_grid(0.2)},
            "fused_recurrent_kda_packed_decode": {"generation": {other_key: _generation_grid(0.4)[MODEL_KEY]}},
            "causal_conv1d_update": {"generation": {other_key: _generation_grid(0.1)[MODEL_KEY]}},
        }
    )
    fused = _query_gen(db, "fused_recurrent_kda_packed_decode")
    assert float(fused) == pytest.approx(0.2) and fused.source == "silicon"
    conv = _query_gen(db, "causal_conv1d_update")
    assert float(conv) == 0.0 and conv.source == "silicon"
    # The 24-head shard still reads its own Triton rows.
    assert float(_query_gen(db, "fused_recurrent_kda_packed_decode", other_shard)) == pytest.approx(0.4)
    assert float(_query_gen(db, "causal_conv1d_update", other_shard)) == pytest.approx(0.1)


def test_fused_decode_sol_is_conv_plus_packed_recurrence():
    empty = _StubDatabase({})
    empty._kda_data = {}
    fused = _query_gen(empty, "kda_fused_decode")
    conv = _query_gen(empty, "causal_conv1d_update")
    recurrence = _query_gen(empty, "fused_recurrent_kda_packed_decode")
    assert fused.source == "sol"
    assert float(fused) == pytest.approx(float(conv) + float(recurrence))


def test_fused_sol_byte_model_is_the_sum_of_conv_and_recurrence():
    empty = _StubDatabase({})
    empty._kda_data = {}  # not loaded -> pure SOL path
    fused = _query(empty, "fused_kda_decode_mtp_dspark")
    conv = _query(empty, "causal_conv1d_update")
    recurrence = _query(empty, "fused_sigmoid_gating_delta_rule_update")
    assert fused.source == "sol"
    assert float(fused) == pytest.approx(float(conv) + float(recurrence))
