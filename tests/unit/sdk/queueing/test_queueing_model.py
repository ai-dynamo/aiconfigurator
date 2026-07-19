# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the queueing (pass-calendar) model.

Structural assertions only — accuracy validation methodology and recorded
results are documented in docs/design/queueing_model.md §5.
"""

import math
import typing

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

# CI gates on `-m "unit or build"` — unmarked tests never run there
pytestmark = pytest.mark.unit


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
        # the alternating structure is the mixed-chunk-OFF mode
        alt_eng = EngineSpec(max_num_batched_tokens=8192, enable_mixed_chunk=False)
        vllm = evaluate_closed_loop(wl, eng, TIMING, backend="vllm")
        sglang = evaluate_closed_loop(wl, alt_eng, TIMING, backend="sglang")
        # alternating calendar: decode stalls behind dedicated prefill
        # batches, so the ITL tail cannot be milder than the fused calendar's
        assert sglang.itl.p99 >= vllm.itl.p99 * 0.9
        assert sglang.itl.p99 > sglang.itl.p50

    def test_sglang_mixed_chunk_is_default_and_differs_from_alternating(self):
        # AIC's generator deploys SGLang agg with enable_mixed_chunk=true
        # (rule_plugin/sglang.rule) — the calendar defaults must match the
        # deployed engine, not the dedicated-prefill-batch structure
        assert EngineSpec().enable_mixed_chunk is True
        wl = WorkloadSpec(isl=4096, osl=64, concurrency=8)
        mixed = evaluate_closed_loop(wl, EngineSpec(max_num_batched_tokens=8192), TIMING, backend="sglang")
        alt = evaluate_closed_loop(
            wl, EngineSpec(max_num_batched_tokens=8192, enable_mixed_chunk=False), TIMING, backend="sglang"
        )
        assert mixed.itl.p99 > mixed.itl.p50  # chunk-bearing passes still spike
        assert (mixed.ttft_steady.mean, mixed.itl.p99) != (alt.ttft_steady.mean, alt.itl.p99)

    def test_fused_c1_ttft_excludes_decode_row(self):
        # a prefill completer's first token is sampled off the final chunk's
        # logits in the same fused pass — it is not an extra decode row
        wl = WorkloadSpec(isl=2048, osl=16, concurrency=1)
        rep = evaluate_closed_loop(wl, EngineSpec(), TIMING)
        assert rep.ttft_steady.mean == pytest.approx(TIMING.prefill_ms(1, 2048, 0))

    def test_mixed_pass_hook_preferred_over_sum(self):
        """A timing model exposing mixed_pass_ms prices genuinely mixed
        passes through it (fused batch, shared cost paid once); pure-decode
        and pure-prefill passes keep the dedicated estimators."""

        class HookTiming(SyntheticTiming):
            def __init__(self):
                self.mixed_calls = []

            def mixed_pass_ms(self, ctx_tokens, gen_tokens, isl, osl, prefix):
                self.mixed_calls.append((ctx_tokens, gen_tokens, isl, osl, prefix))
                # cheaper than the sum by construction: shared cost once
                return 0.5 * (self.prefill_ms(1, ctx_tokens, 0) + self.decode_ms(gen_tokens, isl))

        wl = WorkloadSpec(isl=2048, osl=32, concurrency=8)
        eng = EngineSpec(max_num_batched_tokens=4096)
        hook = HookTiming()
        rep_hook = evaluate_closed_loop(wl, eng, hook, backend="vllm")
        rep_sum = evaluate_closed_loop(wl, eng, SyntheticTiming(), backend="vllm")
        assert hook.mixed_calls, "mixed passes must route through the hook"
        # every recorded call is a genuinely mixed pass with workload shape
        for ctx_tokens, gen_tokens, isl, osl, prefix in hook.mixed_calls:
            assert ctx_tokens > 0 and gen_tokens > 0
            assert (isl, osl, prefix) == (2048, 32, 0)
        # cheaper mixed passes -> strictly better steady TTFT than the sum
        assert rep_hook.ttft_steady.mean < rep_sum.ttft_steady.mean

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


class TestScreeningAdditiveStages:
    """Encoder / dispatch-overhead stages and prefix effective-length —
    the additive terms the legacy `ttft` carries must also reach the
    screening columns, or percentile screens drift from what deploys."""

    _KW: typing.ClassVar[dict] = dict(
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

    def test_encoder_and_dispatch_shift_ttft_only(self):
        base = operating_point_columns(**self._KW)
        shifted = operating_point_columns(**self._KW, encoder_ms=50.0, dispatch_overhead_ms=5.0)
        for col in QUEUEING_COLUMNS:
            if col == "queueing_tier":
                continue
            if col.startswith("ttft"):
                assert shifted[col] == pytest.approx(base[col] + 55.0), col
            else:
                assert shifted[col] == base[col], col

    def test_prefix_uses_effective_prompt_length(self):
        kw = dict(
            isl=8192,
            osl=64,
            batch_size=8,
            ctx_tokens=2048,
            mix_step_ms=100.0,
            genonly_step_ms=10.0,
            prefill_step_ms=95.0,
            num_mix_steps=32,
            num_genonly_steps=32,
        )
        cold = operating_point_columns(**kw)
        cached = operating_point_columns(**kw, prefix=6144)
        # cached tokens do not consume the token budget: 8192-6144=2048
        # effective tokens -> 1 own chunk instead of 4
        assert cached["ttft_steady_mean"] < cold["ttft_steady_mean"]
        assert cached["ttft_transient_max"] == pytest.approx(math.ceil(8 * 2048 / 2048) * 100.0)
        assert cold["ttft_transient_max"] == pytest.approx(math.ceil(8 * 8192 / 2048) * 100.0)

    def test_prefix_bracket_lower_bound_holds_against_evaluator(self):
        """Regression: chunk counts must use effective isl, or the funnel's
        wide-keep bound (`ttft_steady_p99_lo`) can exceed the evaluator's
        p99 for prefix rows and falsely reject feasible candidates."""
        wl = WorkloadSpec(isl=4096, osl=32, prefix=3072, concurrency=8)
        ctx_tokens = 2048
        eng = EngineSpec(max_num_batched_tokens=ctx_tokens + wl.concurrency)
        rep = evaluate_closed_loop(wl, eng, TIMING)
        t_gen = TIMING.decode_ms(8, 4096 + 16)
        t_mix = TIMING.prefill_ms(1, 4096, 3072) + t_gen
        cols = operating_point_columns(
            isl=4096,
            osl=32,
            batch_size=8,
            ctx_tokens=ctx_tokens,
            mix_step_ms=t_mix,
            genonly_step_ms=t_gen,
            prefill_step_ms=t_mix,
            num_mix_steps=16,
            num_genonly_steps=16,
            prefix=3072,
        )
        assert cols["ttft_steady_p99_lo"] <= rep.ttft_steady.p99


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


class TestDisaggTandem:
    """Tandem-recursion structural semantics (accuracy is gated by the
    oracle's disagg families — see tools/queueing_oracle)."""

    def _spec(self, **kw):
        from aiconfigurator.sdk.queueing import DisaggSpec

        base = dict(
            num_prefill_workers=1,
            num_decode_workers=1,
            kv_bytes_per_token=100_000,
            egress_bytes_per_s=1e9,
            ingress_bytes_per_s=1e9,
            bw_efficiency=1.0,
        )
        base.update(kw)
        return DisaggSpec(**base)

    def test_first_token_prefill_side_and_handoff_in_first_gap(self):
        from aiconfigurator.sdk.queueing import evaluate_disagg

        wl = WorkloadSpec(isl=1024, osl=8, concurrency=4)
        rep = evaluate_disagg(wl, EngineSpec(), EngineSpec(), TIMING, TIMING, self._spec())
        handoff_ms = 1024 * 100_000 / 1e9 * 1000.0  # 102.4 solo
        assert rep.mode == "disagg"
        # the measured mean transfer is at least the solo time (fan-in can
        # only slow it down), and the handoff lands in the ITL tail — the
        # first gap — not in TTFT
        assert rep.kv_transfer_ms >= handoff_ms * 0.999
        assert rep.itl.maximum >= handoff_ms
        # TTFT is prefill-side: no transfer term (solo prefill of this
        # shape is ~30ms; give queueing headroom but stay below handoff)
        assert rep.ttft_steady.mean < handoff_ms

    def test_osl1_completes_without_decode_stage(self):
        from aiconfigurator.sdk.queueing import evaluate_disagg

        wl = WorkloadSpec(isl=512, osl=1, concurrency=2)
        rep = evaluate_disagg(wl, EngineSpec(), EngineSpec(), TIMING, TIMING, self._spec())
        assert rep.throughput_rps > 0
        assert not rep.itl.values  # single-token requests have no gaps

    def test_mixed_phases_is_phase_robust_mixture(self):
        from aiconfigurator.sdk.queueing import evaluate_disagg_mixed

        wl = WorkloadSpec(isl=1024, osl=8, concurrency=4)
        rep = evaluate_disagg_mixed(wl, EngineSpec(), EngineSpec(), TIMING, TIMING, self._spec(), phases=3)
        assert rep.mode == "disagg"
        assert rep.ttft_steady.values and rep.itl.values
        assert rep.throughput_rps > 0

    def test_open_loop_rejected(self):
        from aiconfigurator.sdk.queueing import evaluate_disagg

        wl = WorkloadSpec(isl=128, osl=8, request_rate=5.0)
        with pytest.raises(ValueError):
            evaluate_disagg(wl, EngineSpec(), EngineSpec(), TIMING, TIMING, self._spec())


class TestDisaggReportUpgrade:
    def test_composed_rows_upgrade_to_quantitative(self, monkeypatch):
        import pandas as pd

        from aiconfigurator.sdk import common as sdk_common
        from aiconfigurator_core.sdk.queueing import refine as refine_mod
        from aiconfigurator_core.sdk.queueing import timing as timing_mod

        class _FakeModel:
            def get_kvcache_bytes_per_sequence(self, seq_len):
                return 1000.0 * seq_len

        class _FakeDb:
            system_spec: typing.ClassVar[dict] = {"node": {"inter_node_bw": 50e9, "intra_node_bw": 450e9}}

        monkeypatch.setattr(refine_mod, "_rebuild_stage", lambda *a, **k: (_FakeModel(), _FakeDb(), object()))
        monkeypatch.setattr(timing_mod, "DatabaseTimingModel", lambda m, d, b: TIMING)

        row = dict.fromkeys(sdk_common.ColumnsDisagg, 0.0)
        row.update(
            {
                "model": "test/model",
                "isl": 1024,
                "osl": 8,
                "prefix": 0,
                "concurrency": 4,
                "encoder_latency": 0.0,
                "(p)bs": 2,
                "(p)workers": 1,
                "(d)bs": 8,
                "(d)workers": 1,
                "(p)backend": "vllm",
                "(d)backend": "vllm",
                "queueing_tier": "composed",
            }
        )
        df = pd.DataFrame([row])
        out = refine_mod.refine_report_rows(df)
        assert out.at[0, "queueing_tier"] == "quantitative"
        assert out.at[0, "ttft_steady_p50"] > 0
        assert out.at[0, "itl_p99"] > 0

    def test_multimodal_composed_rows_stay_visible(self, monkeypatch):
        import pandas as pd

        from aiconfigurator.sdk import common as sdk_common
        from aiconfigurator_core.sdk.queueing import refine as refine_mod

        monkeypatch.setattr(
            refine_mod, "_rebuild_stage", lambda *a, **k: pytest.fail("must not rebuild for multimodal rows")
        )
        row = dict.fromkeys(sdk_common.ColumnsDisagg, 0.0)
        row.update(
            {"model": "m", "isl": 64, "osl": 8, "concurrency": 2, "encoder_latency": 5.0, "queueing_tier": "composed"}
        )
        df = pd.DataFrame([row])
        out = refine_mod.refine_report_rows(df)
        assert out.at[0, "queueing_tier"] == "composed"


class TestDatabaseTimingMixedPass:
    def test_delegates_to_mix_step_runner_with_efficiency(self):
        from aiconfigurator.sdk.queueing.timing import DatabaseTimingModel

        calls = []

        class _Backend:
            def _get_mix_step_latency(self, model, database, runtime_config, ctx_tokens, gen_tokens, isl, osl, prefix):
                calls.append((ctx_tokens, gen_tokens, isl, osl, prefix))
                return 100.0, 0.0, {}, {}

            def _mix_step_efficiency(self, ctx_tokens, gen_tokens):
                return 0.8

        timing = DatabaseTimingModel(model=object(), database=object(), backend=_Backend())
        # 4096 is grain-aligned -> passed through unchanged
        assert timing.mixed_pass_ms(4096, 7, 2048, 32, 0) == pytest.approx(80.0)
        assert calls == [(4096, 7, 2048, 32, 0)]
        # second call hits the cache — the runner is not consulted again
        assert timing.mixed_pass_ms(4096, 7, 2048, 32, 0) == pytest.approx(80.0)
        assert len(calls) == 1


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
