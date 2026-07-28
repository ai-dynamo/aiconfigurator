# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KDA verify-phase kernel routing on fused-CuTeDSL datasets.

SM100 sglang serves DSPARK target-verify through the fused CuTeDSL kernel
(``fused_kda_decode_mtp_dspark``) — one row per verify step covering BOTH the
conv update and the chain-verify recurrence — so b200_sxm-style datasets carry
no Triton verify rows. ``KDAKernel._query_kda_table`` must route the
recurrence op onto the fused table and fold the conv op to zero, while
Triton-verify datasets (h20_3e-style) and the vLLM physical kernels stay
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
    system_spec = {"gpu": {"mem_bw": 8000}}

    def __init__(self, kda_data):
        self._kda_data = _LoadedTable(kda_data)

    @staticmethod
    def _interp_pr(latency, energy=0.0):
        return PerformanceResult(latency, energy=energy, source="silicon")


def _verify_grid(latency):
    # Exact grid hits at (batch, draft) so interpolation is the identity.
    entry = {"latency": latency, "power": 0.0, "energy": 0.0}
    return {MODEL_KEY: {b: {d: entry for d in (2, 4, 8)} for b in (1, 4, 16, 64)}}


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


def test_fused_sol_byte_model_is_the_sum_of_conv_and_recurrence():
    empty = _StubDatabase({})
    empty._kda_data = {}  # not loaded -> pure SOL path
    fused = _query(empty, "fused_kda_decode_mtp_dspark")
    conv = _query(empty, "causal_conv1d_update")
    recurrence = _query(empty, "fused_sigmoid_gating_delta_rule_update")
    assert fused.source == "sol"
    assert float(fused) == pytest.approx(float(conv) + float(recurrence))
