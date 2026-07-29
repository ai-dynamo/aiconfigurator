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

    def test_rejects_out_of_range_parameters(self):
        with pytest.raises(ValueError):
            WorkloadSpec(isl=100, osl=10, prefix=101, concurrency=4)
        with pytest.raises(ValueError):
            WorkloadSpec(isl=100, osl=10, concurrency=0)
        with pytest.raises(ValueError):
            WorkloadSpec(isl=100, osl=10, request_rate=0.0)
        with pytest.raises(ValueError):
            WorkloadSpec(isl=100, osl=10, concurrency=4, num_requests=0)


class TestEngineSpecValidation:
    def test_rejects_non_positive_limits(self):
        with pytest.raises(ValueError):
            EngineSpec(max_num_batched_tokens=0)
        with pytest.raises(ValueError):
            EngineSpec(max_num_seqs=0)

    def test_guaranteed_no_evict_requires_kv_capacity(self):
        with pytest.raises(ValueError):
            EngineSpec(guaranteed_no_evict=True)
        EngineSpec(guaranteed_no_evict=True, kv_capacity_tokens=1024)  # valid


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

    def test_osl_leq_1_reports_zero_itl(self):
        # run_agg emits tpot = 0.0 for no-decode points; itl_* must agree
        cols = operating_point_columns(
            isl=4096,
            osl=1,
            batch_size=8,
            ctx_tokens=8192,
            mix_step_ms=180.0,
            genonly_step_ms=12.0,
            prefill_step_ms=170.0,
            num_mix_steps=16,
            num_genonly_steps=0,
        )
        assert cols["itl_mean"] == 0.0
        assert cols["itl_p50"] == 0.0
        assert cols["itl_p99"] == 0.0

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

    def test_task_percentile_semantics_are_presence_activated(self):
        """One rule for CLI and exp YAML: setting any percentile field (or
        an itl target) activates percentile filtering; sla_refine is the
        only explicit boolean and never changes elimination semantics."""
        from aiconfigurator.sdk.task_v2 import Task

        model = "Qwen/Qwen3-32B"  # resolved from the local model_configs cache
        assert Task(model_path=model).sla_percentile is False
        assert Task(model_path=model, ttft_percentile=0.99).sla_percentile is True
        task = Task(model_path=model, itl=100.0)
        assert task.sla_percentile is True
        rc = task.build_runtime_config()
        # unset percentiles fall back to evaluation-time defaults
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

    def test_spec_rejects_invalid_workers_and_cap(self):
        from aiconfigurator.sdk.queueing import DisaggSpec

        with pytest.raises(ValueError):
            DisaggSpec(num_prefill_workers=0, num_decode_workers=1)
        with pytest.raises(ValueError):
            DisaggSpec(num_prefill_workers=1, num_decode_workers=0)
        with pytest.raises(ValueError):
            DisaggSpec(num_prefill_workers=1, num_decode_workers=1, prefill_inflight_cap=0)

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

    def test_rejects_kv_pressure_inputs(self):
        """The tandem models no KV admission gate and no hold-until-transfer
        accounting — engines carrying KV-pressure knobs are rejected loudly
        (same honesty contract as the agg calendars) instead of silently
        returning optimistic numbers."""
        from aiconfigurator.sdk.queueing import evaluate_disagg

        wl = WorkloadSpec(isl=1024, osl=8, concurrency=2)
        gne = EngineSpec(guaranteed_no_evict=True, kv_capacity_tokens=8192)
        with pytest.raises(ValueError, match="KV-pressure"):
            evaluate_disagg(wl, gne, EngineSpec(), TIMING, TIMING, self._spec())
        with pytest.raises(ValueError, match="KV-pressure"):
            evaluate_disagg(wl, EngineSpec(), EngineSpec(kv_capacity_tokens=8192), TIMING, TIMING, self._spec())

    def test_variable_shape_parity_and_desync(self):
        """Degenerate quantile streams reproduce the fixed-shape recursion
        exactly; genuinely heterogeneous isl at the same mean desynchronizes
        the cohort and strictly lowers steady TTFT (the measured effect:
        h20e trtllm tp4, isl cv=0.25 -> TTFT 1.8s -> 0.66s)."""
        from aiconfigurator.sdk.queueing import evaluate_closed_loop

        eng = EngineSpec(max_num_batched_tokens=8192, enable_chunked_prefill=False)
        wl_fixed = WorkloadSpec(isl=4096, osl=8, concurrency=16)
        wl_degen = WorkloadSpec(isl=4096, osl=8, concurrency=16, isl_quantiles=(4096,) * 4, osl_quantiles=(8,) * 4)
        rep_f = evaluate_closed_loop(wl_fixed, eng, TIMING, backend="vllm")
        rep_d = evaluate_closed_loop(wl_degen, eng, TIMING, backend="vllm")
        assert rep_d.ttft_steady.mean == pytest.approx(rep_f.ttft_steady.mean, abs=1e-9)
        assert rep_d.throughput_rps == pytest.approx(rep_f.throughput_rps, rel=1e-12)

        wl_var = WorkloadSpec(isl=4096, osl=8, concurrency=16, isl_quantiles=(1024, 3072, 5120, 7168))  # same mean 4096
        rep_v = evaluate_closed_loop(wl_var, eng, TIMING, backend="vllm")
        assert rep_v.ttft_steady.mean < rep_f.ttft_steady.mean
        # throughput stays in the same regime (measured: <10% shift)
        assert rep_v.throughput_rps == pytest.approx(rep_f.throughput_rps, rel=0.35)

    def test_variable_shape_validation(self):
        with pytest.raises(ValueError):
            WorkloadSpec(isl=1024, osl=16, concurrency=2, isl_quantiles=())
        with pytest.raises(ValueError):
            WorkloadSpec(isl=1024, osl=16, concurrency=2, isl_quantiles=(0, 512))
        with pytest.raises(ValueError):
            WorkloadSpec(isl=1024, osl=16, prefix=512, concurrency=2, isl_quantiles=(256, 2048))

    def test_workload_fidelity_contract(self):
        """Reports declare the input tier they consumed — including the
        disagg tandem, which consumes W0-W3 like the agg calendars."""
        from aiconfigurator.sdk.queueing import (
            evaluate_closed_loop,
            evaluate_disagg,
            evaluate_open_loop,
            static_report,
        )
        from aiconfigurator_core.sdk.queueing.spec import workload_fidelity

        eng = EngineSpec(max_num_batched_tokens=8192, enable_chunked_prefill=False)
        w0 = WorkloadSpec(isl=1024, osl=8, concurrency=2)
        w1 = WorkloadSpec(isl=1024, osl=8, request_rate=2.0)
        w2 = WorkloadSpec(isl=1024, osl=8, concurrency=2, isl_quantiles=(512, 1536))
        assert workload_fidelity(w0) == "W0(closed-loop, fixed-shape)"
        assert workload_fidelity(w1) == "W1(open-loop, fixed-shape)"
        assert workload_fidelity(w2) == "W2(closed-loop, shape-marginals)"
        assert evaluate_closed_loop(w0, eng, TIMING).workload_fidelity.startswith("W0")
        assert evaluate_open_loop(w1, eng, TIMING).workload_fidelity.startswith("W1")
        assert evaluate_closed_loop(w2, eng, TIMING).workload_fidelity.startswith("W2")
        assert static_report(100.0, 5.0, osl=8).workload_fidelity.startswith("W0")
        de = EngineSpec()
        assert (
            evaluate_disagg(w0, de, de, TIMING, TIMING, self._spec()).workload_fidelity.startswith("W0")
        )
        assert (
            evaluate_disagg(w1, de, de, TIMING, TIMING, self._spec()).workload_fidelity.startswith("W1")
        )
        assert (
            evaluate_disagg(w2, de, de, TIMING, TIMING, self._spec()).workload_fidelity.startswith("W2")
        )
        w3 = WorkloadSpec(isl=1024, osl=8, concurrency=2, shape_tuples=((512, 0, 8), (1536, 256, 8)))
        assert (
            evaluate_disagg(w3, de, de, TIMING, TIMING, self._spec()).workload_fidelity.startswith("W3")
        )

    def test_w3_joint_shapes_prefix_and_empirical_arrivals(self):
        """Joint (isl, prefix, osl) strata carry per-request prefix hits
        (more cached prefix -> strictly less prefill work -> lower TTFT) and
        empirical inter-arrival strata express batched arrivals (zeros)."""
        from aiconfigurator.sdk.queueing import evaluate_closed_loop, evaluate_open_loop
        from aiconfigurator_core.sdk.queueing.spec import (
            stratified_shape_tuples,
            workload_fidelity,
        )

        eng = EngineSpec(max_num_batched_tokens=8192, enable_chunked_prefill=False)
        # C=1, single tuple: TTFT is exactly one prefill of the EFFECTIVE
        # prompt (isl - prefix) priced at (isl, prefix) — the per-slot prefix
        # reaches the timing call unmangled
        solo = WorkloadSpec(isl=4096, osl=8, concurrency=1, shape_tuples=((4096, 3072, 8),))
        rep_solo = evaluate_closed_loop(solo, eng, TIMING)
        assert rep_solo.ttft_steady.mean == pytest.approx(TIMING.prefill_ms(1, 4096, 3072))
        cold = WorkloadSpec(isl=4096, osl=8, concurrency=8, shape_tuples=((2048, 0, 8), (6144, 0, 8)))
        warm = WorkloadSpec(isl=4096, osl=8, concurrency=8, shape_tuples=((2048, 1024, 8), (6144, 3072, 8)))
        assert workload_fidelity(warm) == "W3(closed-loop, joint-shapes)"
        rep_cold = evaluate_closed_loop(cold, eng, TIMING)
        rep_warm = evaluate_closed_loop(warm, eng, TIMING)
        # strictly less prefill work per request -> throughput cannot drop
        assert rep_warm.throughput_rps >= rep_cold.throughput_rps
        assert rep_warm.workload_fidelity.startswith("W3")

        # batched arrivals: 3 of 4 inter-arrival strata are zero
        wl = WorkloadSpec(isl=1024, osl=8, request_rate=4.0, arrival_quantiles=(0.0, 0.0, 0.0, 1000.0))
        rep = evaluate_open_loop(wl, eng, TIMING)
        assert rep.throughput_rps == pytest.approx(4.0, rel=0.1)
        assert "empirical-arrivals" in rep.workload_fidelity
        # deterministic ordering helper
        st_ = stratified_shape_tuples([(100, 0, 10), (200, 50, 20), (300, 0, 5)], k=2)
        assert len(st_) == 2 and all(len(t) == 3 for t in st_)

    def test_w3_validation(self):
        with pytest.raises(ValueError):  # prefix >= isl
            WorkloadSpec(isl=1024, osl=8, concurrency=2, shape_tuples=((512, 512, 8),))
        with pytest.raises(ValueError):  # exclusive with marginals
            WorkloadSpec(isl=1024, osl=8, concurrency=2, shape_tuples=((512, 0, 8),), isl_quantiles=(512,))
        with pytest.raises(ValueError):  # arrivals need open loop
            WorkloadSpec(isl=1024, osl=8, concurrency=2, arrival_quantiles=(1.0,))
        from aiconfigurator.sdk.queueing import evaluate_disagg

        with pytest.raises(ValueError):  # trace replay is an open-loop mode
            evaluate_disagg(
                WorkloadSpec(isl=1024, osl=8, concurrency=2),
                EngineSpec(),
                EngineSpec(),
                TIMING,
                TIMING,
                self._spec(),
                arrival_trace=[(0.0, 512, 0, 8)],
            )
        with pytest.raises(ValueError):  # stagger shapes the closed-loop burst only
            evaluate_disagg(
                WorkloadSpec(isl=1024, osl=8, request_rate=2.0),
                EngineSpec(),
                EngineSpec(),
                TIMING,
                TIMING,
                self._spec(),
                initial_stagger_ms=5.0,
            )

    def test_open_loop_exact_trace_replay(self):
        """arrival_trace evaluates a verbatim (arrival, isl, prefix, osl)
        sequence — pairing and ordering preserved. Deterministic, and the
        ordering is load-bearing: front-loading the heavy requests must not
        produce the same TTFT distribution as back-loading them."""
        from aiconfigurator.sdk.queueing import evaluate_open_loop

        eng = EngineSpec(max_num_batched_tokens=4096, enable_chunked_prefill=True)
        heavy = [(0.0, 4096, 0, 8)] * 6
        light = [(0.0, 256, 0, 8)] * 6
        mk = lambda seq: [(i * 10.0, a, b, c) for i, (_, a, b, c) in enumerate(seq)]
        wl = WorkloadSpec(isl=2176, osl=8, request_rate=5.0)
        r1 = evaluate_open_loop(wl, eng, TIMING, arrival_trace=mk(heavy + light), warmup_requests=0)
        r2 = evaluate_open_loop(wl, eng, TIMING, arrival_trace=mk(light + heavy), warmup_requests=0)
        again = evaluate_open_loop(wl, eng, TIMING, arrival_trace=mk(heavy + light), warmup_requests=0)
        assert r1.ttft_steady.mean == again.ttft_steady.mean  # deterministic
        # light-first lets the small requests finish before the heavy queue forms
        assert r2.ttft_steady.quantile(0.5) < r1.ttft_steady.quantile(0.5)

    def test_open_loop_rate_tracking_and_queueing(self):
        """Open loop: throughput tracks the arrival rate below capacity;
        TTFT at low utilization is ~one solo prefill, and it strictly grows
        as the rate approaches capacity (queue wait the closed loop cannot
        represent). Deterministic: identical inputs -> identical outputs."""
        from aiconfigurator.sdk.queueing import evaluate_open_loop

        eng = EngineSpec(max_num_batched_tokens=8192, enable_chunked_prefill=False)
        low = evaluate_open_loop(WorkloadSpec(isl=4096, osl=8, request_rate=2.0), eng, TIMING)
        high = evaluate_open_loop(WorkloadSpec(isl=4096, osl=8, request_rate=9.5), eng, TIMING)
        assert low.throughput_rps == pytest.approx(2.0, rel=0.05)
        solo = TIMING.prefill_ms(1, 4096, 0)
        assert low.ttft_steady.mean < solo * 2.5
        assert high.ttft_steady.mean > low.ttft_steady.mean
        again = evaluate_open_loop(WorkloadSpec(isl=4096, osl=8, request_rate=2.0), eng, TIMING)
        assert again.ttft_steady.mean == low.ttft_steady.mean

    def test_open_loop_requires_rate_and_diverges_beyond_capacity(self):
        from aiconfigurator.sdk.queueing import evaluate_open_loop

        eng = EngineSpec(max_num_batched_tokens=8192, enable_chunked_prefill=False)
        with pytest.raises(ValueError):
            evaluate_open_loop(WorkloadSpec(isl=4096, osl=8, concurrency=4), eng, TIMING)
        with pytest.raises(RuntimeError):
            # far beyond capacity: the waiting queue must diverge, not hang
            evaluate_open_loop(
                WorkloadSpec(isl=4096, osl=8, request_rate=1000.0),
                eng,
                TIMING,
                warmup_requests=512,
                window_requests=2048,
            )

    def test_turnaround_delays_replacement_visibility(self):
        """Replacements become visible to the prefill pool only after the
        client turnaround (same semantics as the agg calendar): the cycle
        lengthens by ~eps per generation, so throughput strictly drops,
        while the TTFT origin stays at the completion instant."""
        from aiconfigurator.sdk.queueing import evaluate_disagg

        base = WorkloadSpec(isl=1024, osl=8, concurrency=1)
        delayed = WorkloadSpec(isl=1024, osl=8, concurrency=1, turnaround_ms=500.0)
        rep0 = evaluate_disagg(base, EngineSpec(), EngineSpec(), TIMING, TIMING, self._spec())
        rep1 = evaluate_disagg(delayed, EngineSpec(), EngineSpec(), TIMING, TIMING, self._spec())
        # C=1: the cycle stretches by ~eps, so 1/X grows by ~eps
        cycle0 = 1000.0 / rep0.throughput_rps
        cycle1 = 1000.0 / rep1.throughput_rps
        assert 400.0 < cycle1 - cycle0 < 600.0
        # TTFT origin is the dispatch instant, so the visibility delay lands
        # in TTFT — the same convention as the agg calendar (and as a real
        # client, which times TTFT from its own send)
        assert 400.0 < rep1.ttft_steady.mean - rep0.ttft_steady.mean < 600.0


class TestDisaggWorkloadTiers:
    """W1-W3 coverage of the tandem: per-request shapes/prefix/transfer
    bytes, open-loop arrivals, exact trace replay. The shape and arrival
    streams are shared with the agg calendars (spec._shape_drawer /
    spec._interarrival_stream), so identical workload inputs reach both
    evaluators."""

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

    def _fast_fabric(self):
        # ~1 ms/handoff: keeps the transfer off the critical path so rate
        # sweeps exercise prefill/decode queueing, not the NIC
        return self._spec(egress_bytes_per_s=100e9, ingress_bytes_per_s=100e9)

    def test_degenerate_quantiles_reproduce_fixed_shape(self):
        """Degenerate shape streams must reproduce the fixed-shape tandem
        recursion exactly (the same parity contract as the agg calendar)."""
        from aiconfigurator.sdk.queueing import evaluate_disagg

        eng = EngineSpec()
        fixed = WorkloadSpec(isl=1024, osl=8, concurrency=4)
        degen = WorkloadSpec(isl=1024, osl=8, concurrency=4, isl_quantiles=(1024,) * 4, osl_quantiles=(8,) * 4)
        rf = evaluate_disagg(fixed, eng, eng, TIMING, TIMING, self._spec())
        rd = evaluate_disagg(degen, eng, eng, TIMING, TIMING, self._spec())
        assert rd.ttft_steady.mean == pytest.approx(rf.ttft_steady.mean, abs=1e-9)
        assert rd.throughput_rps == pytest.approx(rf.throughput_rps, rel=1e-12)
        assert rd.kv_transfer_ms == pytest.approx(rf.kv_transfer_ms, abs=1e-9)

    def test_per_request_prefix_reaches_prefill_timing_and_transfer_moves_full_context(self):
        """C=1 single joint stratum: steady TTFT is exactly one prefill of
        the EFFECTIVE prompt priced at (isl, prefix), and the KV handoff
        moves the FULL context (cached prefix saves prefill compute, not
        transfer bytes — the decode pool holds no copy of the prefix)."""
        from aiconfigurator.sdk.queueing import evaluate_disagg

        wl = WorkloadSpec(isl=4096, osl=4, concurrency=1, shape_tuples=((4096, 3072, 4),))
        rep = evaluate_disagg(wl, EngineSpec(), EngineSpec(), TIMING, TIMING, self._spec())
        assert rep.ttft_steady.mean == pytest.approx(TIMING.prefill_ms(1, 4096, 3072))
        assert rep.kv_transfer_ms == pytest.approx(4096 * 100_000 / 1e9 * 1000.0, rel=1e-6)

    def test_transfer_bytes_follow_per_request_isl(self):
        """Heterogeneous shapes at C=1: each handoff prices its OWN isl
        (solo transfers 51.2 / 204.8 ms alternating), not the workload's
        nominal mean — the nominal isl is deliberately set off the tuple
        mean so scalar pricing (old behavior: 99.9 ms for every request)
        cannot masquerade as the per-request answer."""
        from aiconfigurator.sdk.queueing import evaluate_disagg

        wl = WorkloadSpec(isl=999, osl=4, concurrency=1, shape_tuples=((512, 0, 4), (2048, 0, 4)))
        rep = evaluate_disagg(wl, EngineSpec(), EngineSpec(), TIMING, TIMING, self._spec())
        # near-even alternation: the mean sits by 128 ms, far from 99.9
        assert 115.0 < rep.kv_transfer_ms < 141.0
        # the big stratum's handoff is visible in the first ITL gap
        assert rep.itl.maximum >= 204.8 * 0.999

    def test_open_loop_rate_tracking_queueing_and_determinism(self):
        """Open loop: throughput tracks the arrival rate below capacity,
        steady TTFT strictly grows with utilization (queue wait the closed
        loop cannot represent), and identical inputs reproduce bit-equal
        outputs (zero-RNG contract)."""
        from aiconfigurator.sdk.queueing import evaluate_disagg

        eng = EngineSpec()
        spec = self._fast_fabric()
        low = evaluate_disagg(WorkloadSpec(isl=1024, osl=8, request_rate=2.0), eng, eng, TIMING, TIMING, spec)
        high = evaluate_disagg(WorkloadSpec(isl=1024, osl=8, request_rate=25.0), eng, eng, TIMING, TIMING, spec)
        assert low.mode == "disagg"
        assert low.throughput_rps == pytest.approx(2.0, rel=0.05)
        assert high.throughput_rps == pytest.approx(25.0, rel=0.05)
        assert high.ttft_steady.mean > low.ttft_steady.mean
        again = evaluate_disagg(WorkloadSpec(isl=1024, osl=8, request_rate=2.0), eng, eng, TIMING, TIMING, spec)
        assert again.ttft_steady.mean == low.ttft_steady.mean
        assert again.kv_transfer_ms == low.kv_transfer_ms

    def test_open_loop_backlog_diverges_beyond_capacity(self):
        from aiconfigurator.sdk.queueing import evaluate_disagg

        eng = EngineSpec(max_num_batched_tokens=8192)
        with pytest.raises(RuntimeError, match="diverged"):
            evaluate_disagg(
                WorkloadSpec(isl=4096, osl=8, request_rate=1000.0),
                eng,
                eng,
                TIMING,
                TIMING,
                self._fast_fabric(),
                warmup_requests=512,
                window_requests=2048,
            )

    def test_trace_replay_ordering_per_request_and_handoff(self):
        """arrival_trace evaluates a verbatim (arrival, isl, prefix, osl)
        sequence: ordering is load-bearing (front-loading the heavy
        requests differs from back-loading), per-request diagnostics come
        back in trace order with the KV-handoff duration attached."""
        from aiconfigurator.sdk.queueing import evaluate_disagg

        eng = EngineSpec(max_num_batched_tokens=4096)
        heavy = [(0.0, 4096, 0, 8)] * 6
        light = [(0.0, 256, 0, 8)] * 6
        mk = lambda seq: [(i * 10.0, a, b, c) for i, (_, a, b, c) in enumerate(seq)]
        wl = WorkloadSpec(isl=2176, osl=8, request_rate=5.0)
        kw = dict(warmup_requests=0)
        r1 = evaluate_disagg(wl, eng, eng, TIMING, TIMING, self._spec(), arrival_trace=mk(heavy + light), **kw)
        r2 = evaluate_disagg(wl, eng, eng, TIMING, TIMING, self._spec(), arrival_trace=mk(light + heavy), **kw)
        again = evaluate_disagg(wl, eng, eng, TIMING, TIMING, self._spec(), arrival_trace=mk(heavy + light), **kw)
        assert r1.ttft_steady.mean == again.ttft_steady.mean  # deterministic
        # light-first lets the small requests finish before the heavy queue forms
        assert r2.ttft_steady.quantile(0.5) < r1.ttft_steady.quantile(0.5)
        assert r1.per_request is not None and len(r1.per_request) == 12
        assert [p["isl"] for p in r1.per_request] == [4096] * 6 + [256] * 6  # trace order
        for p in r1.per_request:
            assert p["ttft_ms"] is not None and p["e2e_ms"] is not None
            # every request decodes (osl 8), so every request paid a handoff
            # at least as long as its solo transfer time
            assert p["xfer_ms"] >= p["isl"] * 100_000 / 1e9 * 1000.0 * 0.999
            assert p["e2e_ms"] > p["ttft_ms"]

    def test_handoff_flow_placement(self):
        """Serving-flow split: default (DES-idealized) streams the first
        token prefill-side and the handoff lands in gap 1; decode-attach
        (handoff_in_ttft=True — TRT-LLM native disagg / dynamo, per the
        slow-link A/B) absorbs the transfer into TTFT with gap 1 clean.
        Same total timeline, different placement."""
        from aiconfigurator.sdk.queueing import evaluate_disagg

        trace = [(0.0, 2048, 0, 4)]
        wl = WorkloadSpec(isl=2048, osl=4, request_rate=1.0)
        kw = dict(arrival_trace=trace, warmup_requests=0)
        eng = EngineSpec()
        xfer = 2048 * 100_000 / 1e9 * 1000.0  # 204.8 ms solo
        gap1 = evaluate_disagg(wl, eng, eng, TIMING, TIMING, self._spec(), **kw)
        ttft_flow = evaluate_disagg(wl, eng, eng, TIMING, TIMING, self._spec(handoff_in_ttft=True), **kw)
        prefill = TIMING.prefill_ms(1, 2048, 0)
        assert gap1.per_request[0]["ttft_ms"] == pytest.approx(prefill)
        assert gap1.itl.maximum >= xfer * 0.999
        assert ttft_flow.per_request[0]["ttft_ms"] == pytest.approx(prefill + xfer, rel=1e-3)
        assert ttft_flow.itl.maximum < xfer  # gap 1 is a clean decode gap
        # placement moves time between TTFT and gap 1, not the total
        assert ttft_flow.per_request[0]["e2e_ms"] == pytest.approx(gap1.per_request[0]["e2e_ms"], rel=1e-3)

    def test_chunked_off_prefill_stops_admission_at_whole_prompts(self):
        """enable_chunked_prefill=False on the prefill engine: a prompt that
        no longer fits the remaining budget is NOT split (the agg
        FusedCalendar rule; TRT-LLM disagg ctx workers deploy chunked-off).
        With budget 8192 and prompts 6144+4096, chunked-off runs them as two
        solo passes; chunked-on co-schedules the second prompt's head chunk
        into the first pass, so the FIRST request's pass (and TTFT) longer."""
        from aiconfigurator.sdk.queueing import evaluate_disagg

        trace = [(0.0, 6144, 0, 4), (0.0, 4096, 0, 4)]
        wl = WorkloadSpec(isl=5120, osl=4, request_rate=1.0)
        kw = dict(arrival_trace=trace, warmup_requests=0)
        eng_off = EngineSpec(max_num_batched_tokens=8192, enable_chunked_prefill=False)
        eng_on = EngineSpec(max_num_batched_tokens=8192, enable_chunked_prefill=True)
        r_off = evaluate_disagg(wl, eng_off, eng_off, TIMING, TIMING, self._fast_fabric(), **kw)
        r_on = evaluate_disagg(wl, eng_on, eng_on, TIMING, TIMING, self._fast_fabric(), **kw)
        t0_off = r_off.per_request[0]["ttft_ms"]
        t1_off = r_off.per_request[1]["ttft_ms"]
        assert t0_off == pytest.approx(TIMING.prefill_ms(1, 6144, 0))
        assert t1_off == pytest.approx(TIMING.prefill_ms(1, 6144, 0) + TIMING.prefill_ms(1, 4096, 0))
        assert r_on.per_request[0]["ttft_ms"] > t0_off  # head chunk rides along

    def test_mixed_open_loop_is_single_evaluation_passthrough(self):
        """Open-loop workloads have no initial-cohort phase to mix over:
        evaluate_disagg_mixed returns the single evaluation (including
        trace-replay diagnostics)."""
        from aiconfigurator.sdk.queueing import evaluate_disagg_mixed

        eng = EngineSpec()
        rep = evaluate_disagg_mixed(
            WorkloadSpec(isl=1024, osl=8, request_rate=2.0), eng, eng, TIMING, TIMING, self._fast_fabric()
        )
        assert rep.mode == "disagg"
        assert rep.throughput_rps == pytest.approx(2.0, rel=0.05)
        trace = [(i * 100.0, 512, 0, 4) for i in range(8)]
        rep_tr = evaluate_disagg_mixed(
            WorkloadSpec(isl=512, osl=4, request_rate=10.0),
            eng,
            eng,
            TIMING,
            TIMING,
            self._fast_fabric(),
            arrival_trace=trace,
            warmup_requests=0,
        )
        assert rep_tr.per_request is not None and len(rep_tr.per_request) == 8


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


class TestArrivalPlaneAndChunkPricing:
    """ingest_us_per_token (client-dispatch -> scheduler-arrival mapping) and
    the mixed-hook structural validity split (whole-prompt vs mid-prompt
    chunks). Provenance: h20e trtllm 1.3.0rc20 probes — a 23k and a 1k prompt
    dispatched together always serve small-first (flip at 60-100 ms lag), and
    a queued small IS co-scheduled into the head's last-chunk leftover."""

    def _trace_rep(self, ingest, cap=1):
        from aiconfigurator.sdk.queueing import evaluate_open_loop

        # big dispatched first, small at the same instant
        trace = [(0.0, 4000, 0, 4), (0.0, 500, 0, 4)]
        wl = WorkloadSpec(isl=2048, osl=4, request_rate=1.0, ingest_us_per_token=ingest)
        eng = EngineSpec(max_num_batched_tokens=4096, max_num_seqs=cap, enable_chunked_prefill=True)
        return evaluate_open_loop(wl, eng, SyntheticTiming(), backend="vllm", warmup_requests=0, arrival_trace=trace)

    def test_same_instant_burst_keeps_dispatch_order_without_ingest(self):
        rep = self._trace_rep(ingest=0.0)
        big, small = rep.per_request
        assert (big["isl"], small["isl"]) == (4000, 500)  # trace order preserved
        assert small["ttft_ms"] > big["ttft_ms"]  # FCFS by dispatch: small waits out big

    def test_ingest_slope_reorders_same_instant_burst_shortest_first(self):
        rep = self._trace_rep(ingest=3.6)
        big, small = rep.per_request
        assert (big["isl"], small["isl"]) == (4000, 500)  # output stays trace-ordered
        assert small["ttft_ms"] < big["ttft_ms"]  # scheduler arrival: small ingests first

    def test_negative_ingest_rejected(self):
        with pytest.raises(ValueError):
            WorkloadSpec(isl=128, osl=8, concurrency=1, ingest_us_per_token=-1.0)

    def test_mixed_hook_only_prices_whole_prompt_passes(self):
        class RecordingHook(SyntheticTiming):
            def __init__(self):
                self.mixed_calls = []

            def mixed_pass_ms(self, ctx_tokens, gen_tokens, isl, osl, prefix):
                self.mixed_calls.append(ctx_tokens)
                return 0.5 * (self.prefill_ms(1, ctx_tokens, 0) + self.decode_ms(gen_tokens, isl))

        eng = EngineSpec(max_num_batched_tokens=4096, max_num_seqs=8, enable_chunked_prefill=True)
        # whole-prompt regime: isl fits the budget -> hook stays active
        whole = RecordingHook()
        evaluate_closed_loop(WorkloadSpec(isl=2048, osl=32, concurrency=8), eng, whole, backend="vllm")
        assert whole.mixed_calls, "single-pass prompts must keep pricing via the hook"
        # chunked regime: every prompt spans passes -> per-pass past state
        # exists, so mixed passes must fall back to the prefill+decode sum
        chunked = RecordingHook()
        evaluate_closed_loop(WorkloadSpec(isl=6000, osl=32, concurrency=8), eng, chunked, backend="vllm")
        assert not chunked.mixed_calls, "mid-prompt chunks are outside the hook's regime"


class TestKvPressureHonesty:
    """KV-pressure semantics are modeled ONLY as the GUARANTEED_NO_EVICT
    admission gate (trtllm). vLLM preempts-and-recomputes and SGLang
    retracts — different dynamics; accepting kv inputs and ignoring them
    would silently return optimistic numbers, so calendars reject loudly
    (same contract as unconsumed workload-fidelity tiers)."""

    def _eng(self, **kw):
        return EngineSpec(max_num_batched_tokens=4096, max_num_seqs=8, **kw)

    def test_vllm_rejects_kv_capacity(self):
        wl = WorkloadSpec(isl=1024, osl=16, concurrency=4)
        with pytest.raises(ValueError, match="KV-pressure"):
            evaluate_closed_loop(wl, self._eng(kv_capacity_tokens=65536), TIMING, backend="vllm")

    def test_sglang_rejects_kv_capacity(self):
        wl = WorkloadSpec(isl=1024, osl=16, concurrency=4)
        with pytest.raises(ValueError, match="KV-pressure"):
            evaluate_closed_loop(wl, self._eng(kv_capacity_tokens=65536), TIMING, backend="sglang")

    def test_trtllm_without_gne_rejects_kv_capacity(self):
        wl = WorkloadSpec(isl=1024, osl=16, concurrency=4)
        with pytest.raises(ValueError, match="MAX_UTILIZATION"):
            evaluate_closed_loop(wl, self._eng(kv_capacity_tokens=65536), TIMING, backend="trtllm")

    def test_trtllm_gne_still_consumes_kv(self):
        wl = WorkloadSpec(isl=2048, osl=64, concurrency=64)
        eng = self._eng(guaranteed_no_evict=True, kv_capacity_tokens=4 * (2048 + 64))
        assert CALENDARS["trtllm"].admission_cap(wl, eng) == 4
