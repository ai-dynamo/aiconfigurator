# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the queueing (pass-calendar) model.

Structural assertions only — accuracy against the DES oracle is covered by
tools/queueing_oracle/validate_formula.py (run manually / in slow CI since it
executes the discrete-event simulation).
"""

import math

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.queueing import (
    CALENDARS,
    DisaggSpec,
    Distribution,
    EngineSpec,
    WorkloadSpec,
    estimate_disagg,
    evaluate_closed_loop,
    kv_handoff_ms,
    static_report,
)
from aiconfigurator.sdk.queueing.closed_form import (
    QUEUEING_COLUMNS,
    operating_point_columns,
    static_degenerate_columns,
)


class SyntheticTiming:
    """Deterministic timing: prefill linear in tokens, decode in batch+ctx."""

    def prefill_ms(self, batch_size, mean_isl, mean_prefix):
        tokens = batch_size * max(0, mean_isl - mean_prefix)
        return 10.0 + 0.02 * tokens

    def decode_ms(self, batch_size, context_len):
        return max(1.0, 2.0 + 0.05 * batch_size + 0.001 * context_len)


TIMING = SyntheticTiming()


class TestDistribution:
    def test_mean_and_quantiles(self):
        d = Distribution()
        d.add(10.0, 9.0)
        d.add(100.0, 1.0)
        assert d.mean == pytest.approx(19.0)
        assert d.p50 == 10.0
        assert d.p99 == 100.0
        assert d.maximum == 100.0

    def test_empty(self):
        d = Distribution()
        assert math.isnan(d.mean)
        assert math.isnan(d.p50)


class TestWorkloadSpec:
    def test_requires_exactly_one_arrival_spec(self):
        with pytest.raises(ValueError):
            WorkloadSpec(isl=100, osl=10)
        with pytest.raises(ValueError):
            WorkloadSpec(isl=100, osl=10, concurrency=4, request_rate=1.0)

    def test_effective_isl(self):
        wl = WorkloadSpec(isl=100, osl=10, prefix=90, concurrency=1)
        assert wl.effective_isl == 10


class TestClosedLoopEvaluator:
    def test_steady_state_shape(self):
        wl = WorkloadSpec(isl=2048, osl=64, concurrency=8, num_requests=200)
        rep = evaluate_closed_loop(wl, EngineSpec(), TIMING, backend="vllm")
        # steady TTFT at least covers one own prefill chunk
        own = TIMING.prefill_ms(1, 2048, 0)
        assert rep.ttft_steady.mean >= own * 0.5
        # transient staircase strictly dominates steady state
        assert rep.ttft_transient.mean > rep.ttft_steady.mean
        assert rep.ttft_transient.maximum >= rep.ttft_transient.mean
        # ITL is bimodal: p99 (mix pass) well above p50 (gen-only pass)
        assert rep.itl.p99 > rep.itl.p50
        assert rep.throughput_rps > 0
        # blended mean(N) sits between steady and transient
        assert rep.ttft_steady.mean <= rep.ttft_mean_n <= rep.ttft_transient.mean

    def test_mean_n_monotone_in_n(self):
        eng = EngineSpec()
        means = []
        for n in (64, 256, 2048):
            wl = WorkloadSpec(isl=2048, osl=64, concurrency=8, num_requests=n)
            means.append(evaluate_closed_loop(wl, eng, TIMING).ttft_mean_n)
        # transient weight shrinks with N -> blended mean decreases
        assert means[0] > means[1] > means[2]

    def test_prefix_reduces_ttft(self):
        eng = EngineSpec()
        base = evaluate_closed_loop(WorkloadSpec(isl=4096, osl=32, concurrency=4), eng, TIMING)
        cached = evaluate_closed_loop(WorkloadSpec(isl=4096, osl=32, prefix=3072, concurrency=4), eng, TIMING)
        assert cached.ttft_steady.mean < base.ttft_steady.mean

    def test_sglang_itl_spike_is_whole_prefill_batch(self):
        wl = WorkloadSpec(isl=4096, osl=64, concurrency=8)
        eng = EngineSpec(max_num_batched_tokens=8192)
        vllm = evaluate_closed_loop(wl, eng, TIMING, backend="vllm")
        sglang = evaluate_closed_loop(wl, eng, TIMING, backend="sglang")
        # alternating calendar: decode stalls behind dedicated prefill
        # batches, so the ITL tail cannot be milder than the fused calendar's
        assert sglang.itl.p99 >= vllm.itl.p99 * 0.9
        assert sglang.itl.p99 > sglang.itl.p50

    def test_trtllm_guaranteed_no_evict_caps_concurrency(self):
        wl = WorkloadSpec(isl=2048, osl=64, concurrency=64)
        eng = EngineSpec(guaranteed_no_evict=True, kv_capacity_tokens=4 * (2048 + 64))
        cap = CALENDARS["trtllm"].admission_cap(wl, eng)
        assert cap == 4

    def test_open_loop_rejected_by_evaluator(self):
        wl = WorkloadSpec(isl=128, osl=8, request_rate=5.0)
        with pytest.raises(ValueError):
            evaluate_closed_loop(wl, EngineSpec(), TIMING)


class TestDisagg:
    SPEC = DisaggSpec(
        num_prefill_workers=1, num_decode_workers=1, kv_transfer_bandwidth_gbps=64.0, kv_bytes_per_token=131072
    )

    def test_handoff_formula(self):
        wl = WorkloadSpec(isl=4096, osl=16, concurrency=4)
        # mocker common/utils.rs: isl * bytes/token / bandwidth
        assert kv_handoff_ms(wl, self.SPEC) == pytest.approx(4096 * 131072 / 64e9 * 1000.0)

    def test_tandem_decomposition(self):
        wl = WorkloadSpec(isl=2048, osl=32, concurrency=8)
        rep = estimate_disagg(wl, EngineSpec(), EngineSpec(), TIMING, self.SPEC)
        assert rep.mode == "disagg"
        assert rep.kv_transfer_ms > 0
        # TTFT covers at least prefill service + handoff
        assert rep.ttft_steady.mean >= (TIMING.prefill_ms(1, 2048, 0) + rep.kv_transfer_ms) * 0.9
        # decode stage has no prefill interference: tight ITL
        assert rep.itl.p99 <= rep.itl.p50 * 1.5

    def test_more_prefill_workers_reduce_ttft_tail(self):
        wl = WorkloadSpec(isl=4096, osl=32, concurrency=16)
        one = estimate_disagg(wl, EngineSpec(), EngineSpec(), TIMING, DisaggSpec(1, 2))
        four = estimate_disagg(wl, EngineSpec(), EngineSpec(), TIMING, DisaggSpec(4, 2))
        assert four.ttft_steady.mean <= one.ttft_steady.mean * 1.01


class TestStaticDegenerate:
    def test_all_metrics_collapse(self):
        rep = static_report(context_latency_ms=123.0, gen_step_latency_ms=7.0, osl=32)
        assert rep.ttft_steady.mean == rep.ttft_steady.p99 == 123.0
        assert rep.ttft_transient.mean == 123.0
        assert rep.itl.p50 == rep.itl.p99 == rep.tpot.mean == 7.0

    def test_static_columns_equal_legacy_scalars(self):
        cols = static_degenerate_columns(123.0, 7.0)
        assert all(cols[k] == 123.0 for k in cols if k.startswith("ttft"))
        assert all(cols[k] == 7.0 for k in cols if k.startswith("itl"))


class TestOperatingPointColumns:
    def test_arithmetic_only_and_sane(self):
        cols = operating_point_columns(
            isl=4096,
            osl=256,
            batch_size=32,
            ctx_tokens=8192,
            mix_step_ms=180.0,
            genonly_step_ms=12.0,
            prefill_step_ms=170.0,
            num_mix_steps=16,
            num_genonly_steps=240,
        )
        assert set(cols) == set(QUEUEING_COLUMNS)
        # own service = ceil(4096/8192)=1 mix pass; residual adds < one pass
        assert 180.0 <= cols["ttft_steady_mean"] <= 360.0
        assert cols["ttft_transient_max"] == math.ceil(32 * 4096 / 8192) * 180.0
        assert cols["itl_p50"] == 12.0
        assert cols["itl_p99"] == 180.0
        assert cols["ttft_steady_p99"] >= cols["ttft_steady_p50"]

    def test_columns_registered_in_all_schemas(self):
        for schema in (common.ColumnsAgg, common.ColumnsStatic, common.ColumnsDisagg):
            for col in QUEUEING_COLUMNS:
                assert col in schema
