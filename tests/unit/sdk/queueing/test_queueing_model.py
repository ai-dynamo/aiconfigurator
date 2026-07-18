# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the queueing (pass-calendar) model.

Structural assertions only — accuracy validation methodology and recorded
results are documented in docs/design/queueing_model.md §5.
"""

import math

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.queueing import (
    CALENDARS,
    Distribution,
    EngineSpec,
    WorkloadSpec,
    evaluate_closed_loop,
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


class TestSlaFunnel:
    """Funnel semantics with a stubbed evaluator (no perf DB needed)."""

    def _df(self):
        import pandas as pd

        from aiconfigurator.sdk import common

        rows = []
        for i, (bs, hi, lo) in enumerate([(8, 100.0, 50.0), (16, 300.0, 80.0), (32, 900.0, 700.0)]):
            r = dict.fromkeys(common.ColumnsAgg, 0)
            r.update(
                {
                    "isl": 1024,
                    "osl": 32,
                    "prefix": 0,
                    "bs": bs,
                    "ctx_tokens": 4096,
                    "seq/s": 100 - i,
                    "encoder_latency": 0.0,
                    "ttft_steady_p99_lo": lo,
                    "ttft_steady_p99_hi": hi,
                    "queueing_tier": "screening",
                }
            )
            rows.append(r)
        return pd.DataFrame(rows)

    def test_funnel_decides_straddlers_with_evaluator(self, monkeypatch):
        from aiconfigurator.sdk.queueing import refine as refine_mod

        def fake_eval(wl, eng, timing, backend="vllm", **kw):
            d = Distribution()
            # bs=16 straddler resolves feasible; bs=32 infeasible
            d.add(150.0 if wl.concurrency == 16 else 500.0)
            itl = Distribution()
            itl.add(5.0)
            from aiconfigurator.sdk.queueing.spec import QueueingReport

            return QueueingReport(
                ttft_steady=d, ttft_transient=d, itl=itl, tpot=itl, throughput_rps=1.0, output_tokens_per_s=1.0, e2e=d
            )

        monkeypatch.setattr(refine_mod, "evaluate_closed_loop", fake_eval)
        monkeypatch.setattr(refine_mod, "DatabaseTimingModel", lambda *a, **k: object())

        class _B:  # bare backend stand-in for cache attachment
            pass

        class _Db:
            backend = "vllm"

        df = self._df()
        out = refine_mod.apply_sla_funnel(
            df, model=object(), database=_Db(), backend=_B(), constraints={"ttft": (200.0, 0.5)}, top_k=3
        )
        # bs=8: hi=100 <= 200 -> certain pass, stays screening
        # bs=16: straddler, refined 150 <= 200 -> kept, quantitative
        # bs=32: lo=700 would have been screened out upstream; here hi>200
        #        -> refined 500 > 200 -> dropped
        assert set(out["bs"]) == {8, 16}
        tiers = dict(zip(out["bs"], out["queueing_tier"], strict=True))
        assert tiers[8] == "screening"
        assert tiers[16] == "quantitative"

    def test_unsupported_percentile_rejected(self):
        from aiconfigurator.sdk.queueing import refine as refine_mod

        with pytest.raises(ValueError):
            refine_mod.apply_sla_funnel(
                self._df(), model=None, database=None, backend=None, constraints={"ttft": (200.0, 0.87)}
            )

    def test_percentile_defaults_in_runtime_config(self):
        from aiconfigurator.sdk.config import RuntimeConfig

        rc = RuntimeConfig(batch_size=1, isl=8, osl=8)
        assert rc.ttft_percentile == 0.5
        assert rc.itl_percentile == 0.99


class TestBracketAndE2E:
    def test_bracket_bounds_the_quantiles(self):
        cols = operating_point_columns(
            isl=1024,
            osl=64,
            batch_size=16,
            ctx_tokens=8176,
            mix_step_ms=90.0,
            genonly_step_ms=8.0,
            prefill_step_ms=82.0,
            num_mix_steps=2,
            num_genonly_steps=62,
        )
        assert cols["ttft_steady_p99_lo"] <= cols["ttft_steady_p50"]
        assert cols["ttft_steady_p99_hi"] >= cols["ttft_steady_p99"]
        assert cols["queueing_tier"] == "screening"

    def test_evaluator_reports_e2e_distribution(self):
        wl = WorkloadSpec(isl=1024, osl=16, concurrency=4)
        rep = evaluate_closed_loop(wl, EngineSpec(), TIMING)
        assert rep.e2e.values, "e2e distribution should be populated"
        assert rep.e2e.mean > rep.ttft_steady.mean


class TestMultimodalRefine:
    def test_encoder_latency_shifts_ttft(self, monkeypatch):
        import pandas as pd

        from aiconfigurator.sdk import common
        from aiconfigurator.sdk.queueing import refine as refine_mod
        from aiconfigurator.sdk.queueing.spec import QueueingReport

        def fake_eval(wl, eng, timing, backend="vllm", **kw):
            # visual tokens must have joined the prefill length
            assert wl.isl == 1024 + 128
            d = Distribution()
            d.add(100.0)
            itl = Distribution()
            itl.add(5.0)
            return QueueingReport(
                ttft_steady=d, ttft_transient=d, itl=itl, tpot=itl, throughput_rps=1.0, output_tokens_per_s=1.0, e2e=d
            )

        monkeypatch.setattr(refine_mod, "evaluate_closed_loop", fake_eval)
        monkeypatch.setattr(refine_mod, "DatabaseTimingModel", lambda *a, **k: object())

        class _Backend:
            def _visual_context_tokens(self, model, runtime_config):
                return 128

        class _Db:
            backend = "vllm"

        row = dict.fromkeys(common.ColumnsAgg, 0)
        row.update(
            {
                "isl": 1024,
                "osl": 32,
                "prefix": 0,
                "bs": 8,
                "ctx_tokens": 4096,
                "seq/s": 1.0,
                "encoder_latency": 50.0,
                "queueing_tier": "screening",
            }
        )
        df = pd.DataFrame([row])
        reports = refine_mod.refine_rows(
            df, [0], model=object(), database=_Db(), backend=_Backend(), runtime_config=object()
        )
        assert 0 in reports
        # encoder latency shifts the TTFT distribution additively
        assert df.at[0, "ttft_steady_p50"] == pytest.approx(150.0)
        assert df.at[0, "queueing_tier"] == "quantitative"

    def test_multimodal_without_runtime_config_skipped(self, monkeypatch):
        import pandas as pd

        from aiconfigurator.sdk import common
        from aiconfigurator.sdk.queueing import refine as refine_mod

        monkeypatch.setattr(refine_mod, "DatabaseTimingModel", lambda *a, **k: object())

        class _Db:
            backend = "vllm"

        row = dict.fromkeys(common.ColumnsAgg, 0)
        row.update(
            {
                "isl": 1024,
                "osl": 32,
                "prefix": 0,
                "bs": 8,
                "ctx_tokens": 4096,
                "seq/s": 1.0,
                "encoder_latency": 50.0,
                "queueing_tier": "screening",
            }
        )

        class _Backend:
            pass

        df = pd.DataFrame([row])
        reports = refine_mod.refine_rows(df, [0], model=object(), database=_Db(), backend=_Backend())
        assert reports == {}
        assert df.at[0, "queueing_tier"] == "screening"
