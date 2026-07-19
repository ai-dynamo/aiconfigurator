# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CI hook for the DES oracle validation gate (tools/queueing_oracle).

Running the gate here makes scheduler-semantics drift (design doc §6) a
failing check instead of a stale doc claim: any change to sdk/queueing that
diverges from the oracle's vLLM-anchored scheduling semantics fails the
9-family battery. The whole gate is stdlib + sdk.queueing and runs in ~1s.
"""

import importlib
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_ORACLE_DIR = Path(__file__).parents[4] / "tools" / "queueing_oracle"
_ORACLE_MODULES = ("vllm_sim", "workload", "metrics", "sysspec", "validate_formula")


@pytest.fixture(scope="module")
def oracle():
    sys.path.insert(0, str(_ORACLE_DIR))
    try:
        yield {name: importlib.import_module(name) for name in _ORACLE_MODULES}
    finally:
        sys.path.remove(str(_ORACLE_DIR))
        for name in _ORACLE_MODULES:
            sys.modules.pop(name, None)


def test_validation_gate_passes(oracle):
    """The full 9-family gate: evaluator tier within tolerance vs the DES."""
    assert oracle["validate_formula"].main() == 0


def _perf(oracle):
    return oracle["vllm_sim"].CallbackPerfModel(
        lambda b, t, p: 10.0 + 0.02 * b * t,
        lambda b, c: 2.0 + 0.05 * b + 0.001 * c,
    )


# 1024 tokens x 100_000 B/token over 1 GB/s at efficiency 1.0 = 102.4 ms
_SOLO_XFER_MS = 1024 * 100_000 / 1e9 * 1000.0


def _spec(oracle, eff=1.0):
    return oracle["vllm_sim"].TransferSpec(
        kv_bytes_per_token=100_000, egress_bytes_per_s=1e9, ingress_bytes_per_s=1e9, bw_efficiency=eff
    )


class TestTransferFabric:
    """Max-min fair sharing math: transfers are not independent."""

    def _fabric(self, oracle, eff=1.0):
        vs = oracle["vllm_sim"]
        return vs.TransferFabric(_spec(oracle, eff))

    def test_fan_in_shares_ingress(self, oracle):
        f = self._fabric(oracle)
        f.submit(0, 0, 1e8, 0.0, "a")
        f.submit(1, 0, 1e8, 0.0, "b")  # two sources, one destination
        assert f.next_completion_ms() == pytest.approx(200.0)
        assert {p for _, p in f.pop_completed(200.0)} == {"a", "b"}

    def test_fan_out_shares_egress(self, oracle):
        f = self._fabric(oracle)
        f.submit(0, 0, 1e8, 0.0, "a")
        f.submit(0, 1, 1e8, 0.0, "b")  # one source, two destinations
        assert f.next_completion_ms() == pytest.approx(200.0)

    def test_disjoint_pairs_run_at_full_rate(self, oracle):
        f = self._fabric(oracle)
        f.submit(0, 0, 1e8, 0.0, "a")
        f.submit(1, 1, 1e8, 0.0, "b")
        assert f.next_completion_ms() == pytest.approx(100.0)

    def test_staggered_flow_frees_bandwidth_on_completion(self, oracle):
        f = self._fabric(oracle)
        f.submit(0, 0, 1e8, 0.0, "a")
        f.submit(1, 0, 1e8, 50.0, "b")  # joins after a already moved 50 MB
        # a: 50 MB left at half rate -> done at 150; b then runs solo
        assert f.pop_completed(150.0) == [(150.0, "a")]
        assert f.next_completion_ms() == pytest.approx(200.0)

    def test_future_submit_does_not_swallow_completions(self, oracle):
        # a caller may submit with a FUTURE timestamp (a pass computed in
        # one loop iteration timestamps its outputs at the pass end); the
        # piecewise advance must still complete in-flight flows at their
        # true times instead of dragging them to the submit time
        f = self._fabric(oracle)
        f.submit(0, 0, 1e8, 0.0, "a")  # solo -> done at 100
        f.submit(1, 1, 1e8, 300.0, "b")  # future submit past a's completion
        done = f.pop_completed(300.0)
        assert done == [(pytest.approx(100.0), "a")]

    def test_bw_efficiency_derates_line_rate(self, oracle):
        f = self._fabric(oracle, eff=0.5)
        f.submit(0, 0, 1e8, 0.0, "a")
        assert f.next_completion_ms() == pytest.approx(200.0)


class TestSysSpecWiring:
    def test_transfer_spec_from_system_dict(self, oracle):
        node = {"node": {"inter_node_bw": 50e9, "intra_node_bw": 450e9}}
        spec = oracle["sysspec"].transfer_spec_from_system(node, kv_bytes_per_token=137_000)
        assert spec.egress_bytes_per_s == 50e9
        assert spec.bw_efficiency == 0.8
        intra = oracle["sysspec"].transfer_spec_from_system(node, kv_bytes_per_token=137_000, cross_node=False)
        assert intra.ingress_bytes_per_s == 450e9

    def test_transfer_spec_from_real_system_yaml(self, oracle):
        spec = oracle["sysspec"].transfer_spec_from_system("h200_sxm", kv_bytes_per_token=1)
        assert spec.egress_bytes_per_s > 0
        assert spec.ingress_bytes_per_s == spec.egress_bytes_per_s


class TestDisaggFlow:
    """Disagg serving-flow semantics: first token from the prefill worker,
    KV handoff in the first ITL gap, transfer contention computed."""

    def test_first_token_from_prefill_and_handoff_in_first_gap(self, oracle):
        vs = oracle["vllm_sim"]
        reqs = oracle["workload"].synthetic(request_count=8, isl=1024, osl=8, block_size=64)
        vs.DisaggSimulator(
            1,
            1,
            vs.EngineArgs(worker_type="prefill"),
            vs.EngineArgs(worker_type="decode"),
            _perf(oracle),
            concurrency=2,
            transfer=_spec(oracle),
        ).run(reqs)

        solo_prefill_ms = 10.0 + 0.02 * 1024
        for r in reqs:
            assert len(r.token_times) == 8
            assert r.completed_ms >= 0
            # TTFT is prefill-side (queueing + prefill compute); the KV
            # handoff defers token 2, not token 1
            assert r.token_times[0] - r.dispatch_ms >= solo_prefill_ms
            assert r.token_times[1] - r.token_times[0] >= _SOLO_XFER_MS

    def test_fan_in_slows_concurrent_handoffs(self, oracle):
        vs = oracle["vllm_sim"]
        wl = oracle["workload"]

        # two prefill workers complete simultaneously and fan in on one
        # decode worker: both transfers run at half rate
        reqs = wl.synthetic(request_count=2, isl=1024, osl=4, block_size=64)
        vs.DisaggSimulator(
            2,
            1,
            vs.EngineArgs(worker_type="prefill"),
            vs.EngineArgs(worker_type="decode"),
            _perf(oracle),
            concurrency=2,
            transfer=_spec(oracle),
        ).run(reqs)
        for r in reqs:
            assert r.token_times[1] - r.token_times[0] >= 2 * _SOLO_XFER_MS

        # baseline: a single request's handoff runs at full rate
        solo = wl.synthetic(request_count=1, isl=1024, osl=4, block_size=64)
        vs.DisaggSimulator(
            1,
            1,
            vs.EngineArgs(worker_type="prefill"),
            vs.EngineArgs(worker_type="decode"),
            _perf(oracle),
            concurrency=1,
            transfer=_spec(oracle),
        ).run(solo)
        assert solo[0].token_times[1] - solo[0].token_times[0] < 1.5 * _SOLO_XFER_MS

    def test_osl1_completes_on_the_prefill_worker(self, oracle):
        vs = oracle["vllm_sim"]
        reqs = oracle["workload"].synthetic(request_count=4, isl=512, osl=1, block_size=64)
        vs.DisaggSimulator(
            1,
            1,
            vs.EngineArgs(worker_type="prefill"),
            vs.EngineArgs(worker_type="decode"),
            _perf(oracle),
            concurrency=2,
            transfer=_spec(oracle),
        ).run(reqs)
        for r in reqs:
            assert len(r.token_times) == 1
            assert r.completed_ms >= 0
