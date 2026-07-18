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
_ORACLE_MODULES = ("vllm_sim", "workload", "metrics", "validate_formula")


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


class TestDisaggFlow:
    """Disagg serving-flow semantics: first token from the prefill worker,
    KV handoff in the first ITL gap, decode continues the same sequence."""

    def _perf(self, oracle):
        return oracle["vllm_sim"].CallbackPerfModel(
            lambda b, t, p: 10.0 + 0.02 * b * t,
            lambda b, c: 2.0 + 0.05 * b + 0.001 * c,
        )

    def test_first_token_from_prefill_and_handoff_in_first_gap(self, oracle):
        vs = oracle["vllm_sim"]
        prefill_args = vs.EngineArgs(worker_type="prefill", kv_transfer_bandwidth_gbps=1.0, kv_bytes_per_token=100_000)
        decode_args = vs.EngineArgs(worker_type="decode")
        reqs = oracle["workload"].synthetic(request_count=8, isl=1024, osl=8, block_size=64)
        vs.DisaggSimulator(1, 1, prefill_args, decode_args, self._perf(oracle), concurrency=2).run(reqs)

        handoff_ms = 1024 * 100_000 / 1e9 * 1000.0
        solo_prefill_ms = 10.0 + 0.02 * 1024
        for r in reqs:
            assert len(r.token_times) == 8
            assert r.completed_ms >= 0
            # TTFT is prefill-side (queueing + prefill compute); the KV
            # handoff defers token 2, not token 1
            assert r.token_times[0] - r.dispatch_ms >= solo_prefill_ms
            assert r.token_times[1] - r.token_times[0] >= handoff_ms

    def test_osl1_completes_on_the_prefill_worker(self, oracle):
        vs = oracle["vllm_sim"]
        prefill_args = vs.EngineArgs(worker_type="prefill", kv_transfer_bandwidth_gbps=1.0, kv_bytes_per_token=100_000)
        decode_args = vs.EngineArgs(worker_type="decode")
        reqs = oracle["workload"].synthetic(request_count=4, isl=512, osl=1, block_size=64)
        vs.DisaggSimulator(1, 1, prefill_args, decode_args, self._perf(oracle), concurrency=2).run(reqs)
        for r in reqs:
            assert len(r.token_times) == 1
            assert r.completed_ms >= 0
