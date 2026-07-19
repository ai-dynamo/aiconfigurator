# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Engine-step performance guard -- the perf analog of ``test_engine_step_parity``.

Asserts the compiled Rust engine-step stays at least a floor multiple as fast as
the pure-Python step, per case. Python and Rust are timed back-to-back on the
same host, so the reported *speedup ratio* is far more comparable across
machines than absolute wall-clock -- most machine-speed variance divides out
(the ratio can still shift somewhat across architectures). That makes it safe to
run as a blocking gate on shared CI runners where absolute wall-clock is noisy.

It exists to catch algorithmic regressions in the Rust hot path -- e.g. a
per-query ``SiteIndex::resolve`` that sorts every collected GEMM site
(``O(n log n)``) instead of selecting the nearest handful (``O(n)``). That class
of bug once pushed the Rust step to 0.15-0.78x of Python (see
``benchmarks.md``); any floor between today's margins and 1.0 trips it.

Every floor is ``>= 1.0``: the gate encodes the project goal that Rust must be
*at least as fast as* Python on each guarded case -- a sub-1.0 floor would license
Rust being slower, defeating the migration. Floors sit below the measured margin
(see ``../docs/perf-speedup-report.md``) with headroom for runner noise:
- ``nemotron-nas`` -- large graph, wide stable margin (~1.9-2.3x) -> 1.5x floor,
  which also catches partial regressions that still leave Rust >1x.
- the small (~20 us) graphs sit near the fixed-FFI-tax floor (~1.1-1.5x) -> 1.0x,
  the goal line: Rust must not lose to Python, with no sub-1 slack. These have
  the thinnest margin and are the most machine-dependent, so ``ITERATIONS`` is
  set high for a stable p50; if x86 CI shows a benign sub-1.0 dip, prefer raising
  iterations over dropping the floor below 1.0.
"""

from __future__ import annotations

import os
import sys

import pytest

# ``parity_tests`` has no ``__init__.py`` so pytest already prepends this dir to
# ``sys.path``; do it explicitly too so the module imports under any importmode.
sys.path.insert(0, os.path.dirname(__file__))

from benchmark_engine_step import CASES, _run_case

pytestmark = pytest.mark.integration

WARMUP = 20
ITERATIONS = 100

# Minimum Rust-vs-Python p50 speedup per case (applied to both phases). All
# >= 1.0 by design (see the module docstring). Derived from the per-case minimum
# margin in ../docs/perf-speedup-report.md, discounted ~15% for runner variance and
# clamped to 1.0.
MIN_SPEEDUP = {
    "nemotron-nas-49b": 1.5,
    "deepseek-v3": 1.0,
    "minimax-m25": 1.0,
}


@pytest.mark.parametrize("case_name", sorted(MIN_SPEEDUP))
def test_engine_step_not_slower_than_python(case_name: str) -> None:
    floor = MIN_SPEEDUP[case_name]
    result = _run_case(
        CASES[case_name],
        warmup=WARMUP,
        iterations=ITERATIONS,
        suppress_output=True,
        cache_mode="hot",
    )

    header = (
        f"{case_name}: Rust-vs-Python p50 speedup floor {floor:.2f}x "
        f"(warmup={WARMUP}, iterations={ITERATIONS})\n"
        f"{'phase':<12}{'python_p50_us':>15}{'rust_p50_us':>13}"
        f"{'speedup':>10}{'floor':>8}{'status':>8}"
    )
    rows = []
    failures = []
    for phase, vals in result["phases"].items():
        py_p50 = vals["python"]["call_p50_us"]
        rust_p50 = vals["rust"]["call_p50_us"]
        speedup = vals["rust"]["speedup_vs_python_p50"]
        ok = speedup >= floor
        if not ok:
            failures.append(phase)
        rows.append(
            f"{phase:<12}{py_p50:>15.2f}{rust_p50:>13.2f}{speedup:>9.2f}x{floor:>7.2f}x{'OK' if ok else 'SLOW':>8}"
        )

    report = "\n".join([header, *rows])
    assert not failures, f"Rust engine-step below the {floor:.2f}x speedup floor on {', '.join(failures)}:\n{report}"
