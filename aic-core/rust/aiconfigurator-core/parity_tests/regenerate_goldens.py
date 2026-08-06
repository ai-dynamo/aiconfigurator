# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regenerate the parity golden fixtures from the FROZEN Python engine.

Dedup-plan Gate 2 (`docs/python-dedup-plan.md`): the parity suites compare the
live Rust engine against golden fixtures captured from the Python latency
path while that path is still alive. This script is the only writer of
``parity_tests/goldens/``:

- ``engine_step.json``   — every (case, surface) pair in
  ``test_engine_step_parity.ENGINE_STEP_GOLDEN_MATRIX`` (the script defines
  no case of its own; the test module is the single source of truth).
- ``compile_engine.json`` — the compile-engine subset references (static /
  mixed / decode per subset case, plus the chunked-prefill, imbalance-scale,
  and WideEP references) from ``test_compile_engine_parity``.
- ``per_op.json``        — the Python summary per-op dicts (latency + energy
  + source) for static ctx/gen of the 10-case compile-engine subset; these
  anchor the per-op op-list FFI.

Run from the repository root::

    .venv/bin/python aic-core/rust/aiconfigurator-core/parity_tests/regenerate_goldens.py

Contract: the output is byte-reproducible — floats serialize at full repr
precision, keys are sorted, and the header carries no wall-clock timestamps
(only the git HEAD of the capture). Run it twice and diff to prove a capture
is clean. Regenerate deliberately (review the diff!) when a case list or a
compared metric changes, never as a way to silence a parity failure.
"""

from __future__ import annotations

import os

# Thread caps: pinned before any numpy/pandas import so the capture is
# byte-reproducible across hosts and runs (BLAS/rayon thread counts must not
# influence the frozen reference).
THREAD_CAP_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "RAYON_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
os.environ.update(THREAD_CAP_ENV)

# Pinned so golden capture freezes the PYTHON engine: since the rust default
# flip (PR-1) an unpinned run routes the step to the compiled engine, which
# would capture rust-vs-rust goldens and void the differential oracle. The
# env var backstops every internal RuntimeConfig construction; the imported
# `_python_*` helpers additionally pin it per-config. (Same rationale as
# tools/prediction_regression_gate/collect_static.py.)
ENGINE_STEP_BACKEND = "python"
os.environ["AICONFIGURATOR_ENGINE_STEP_BACKEND"] = ENGINE_STEP_BACKEND

import json
import math
import subprocess
import sys
import time
from pathlib import Path

_PARITY_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_PARITY_DIR))

import test_compile_engine_parity as compile_parity
import test_engine_step_parity as engine_parity

GOLDEN_DIR = _PARITY_DIR / "goldens"
COMMAND = "python aic-core/rust/aiconfigurator-core/parity_tests/regenerate_goldens.py"


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=_PARITY_DIR, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _header(**counts: int) -> dict:
    # Deliberately NO wall-clock timestamp: two captures of the same tree
    # must be byte-identical; the HEAD sha is the only capture identity.
    return {
        "command": COMMAND,
        "engine_step_backend": ENGINE_STEP_BACKEND,
        "git_describe": _git("describe", "--always", "--dirty"),
        "git_head": _git("rev-parse", "HEAD"),
        "thread_caps": THREAD_CAP_ENV,
        **counts,
    }


def _require_finite(value: float, context: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"non-finite golden value for {context}: {value!r}")
    return value


def _encode_metrics(metrics: dict) -> dict:
    """One golden record per (case, surface): ``{"error": kind}`` when every
    metric raised with a single kind (the single-call-site agg/disagg
    wrapping), else ``{"values": {metric: float | {"error": kind}}}``."""
    sentinel = engine_parity._ErrorSentinel
    kinds = {value.kind for value in metrics.values() if isinstance(value, sentinel)}
    if len(kinds) == 1 and all(isinstance(value, sentinel) for value in metrics.values()):
        return {"error": kinds.pop()}
    return {
        "values": {
            name: ({"error": value.kind} if isinstance(value, sentinel) else _require_finite(value, name))
            for name, value in metrics.items()
        }
    }


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def capture_engine_step() -> dict:
    cases: dict[str, dict] = {}
    surface_count = 0
    for params, surfaces in engine_parity.ENGINE_STEP_GOLDEN_MATRIX:
        for param in params:
            (case,) = param.values
            # Per-case cache hygiene, mirroring `_prepare_rust_core` in the
            # tests (a no-op for this python-pinned capture, kept so capture
            # and test setup stay symmetrical).
            engine_parity.rust_engine_step._engine_handle_cache_clear()
            entry = cases.setdefault(param.id, {})
            for surface in surfaces:
                started = time.monotonic()
                metrics = engine_parity._surface_metrics(case, surface, engine_step_backend=ENGINE_STEP_BACKEND)
                entry[surface] = _encode_metrics(metrics)
                surface_count += 1
                _log(f"[engine_step {surface_count:>3}] {param.id} :: {surface} ({time.monotonic() - started:.1f}s)")
    return {"header": _header(case_count=len(cases), surface_count=surface_count), "cases": cases}


def capture_compile_engine() -> dict:
    references: dict[str, float] = {}
    for param in compile_parity._SUBSET_CASES:
        (case,) = param.values
        started = time.monotonic()
        references[f"{param.id}::static_ctx"] = compile_parity._python_static(case, "static_ctx", 1)
        references[f"{param.id}::static_gen"] = compile_parity._python_static(case, "static_gen", 1)
        references[f"{param.id}::mixed_step"] = compile_parity._python_mixed(case)
        references[f"{param.id}::decode_step"] = compile_parity._python_decode(case)
        _log(f"[compile_engine] {param.id} ({time.monotonic() - started:.1f}s)")
    for name, capture in (
        ("chunked_prefill", compile_parity._python_chunked_prefill_references),
        ("imbalance_scale", compile_parity._python_imbalance_scale_references),
        ("wideep_sglang", compile_parity._python_wideep_sglang_references),
        ("wideep_trtllm", compile_parity._python_wideep_trtllm_references),
    ):
        started = time.monotonic()
        references.update(capture())
        _log(f"[compile_engine] {name} ({time.monotonic() - started:.1f}s)")
    references = {key: _require_finite(value, key) for key, value in references.items()}
    return {"header": _header(reference_count=len(references)), "references": references}


def capture_per_op() -> dict:
    cases: dict[str, dict] = {}
    for param in compile_parity._SUBSET_CASES:
        (case,) = param.values
        started = time.monotonic()
        reference = compile_parity._python_static_per_op_reference(case)
        for phase, ops in reference.items():
            for op_name, record in ops.items():
                _require_finite(record["latency_ms"], f"{param.id}::{phase}::{op_name}::latency_ms")
                _require_finite(record["energy_wms"], f"{param.id}::{phase}::{op_name}::energy_wms")
        cases[param.id] = reference
        _log(f"[per_op] {param.id} ({time.monotonic() - started:.1f}s)")
    return {"header": _header(case_count=len(cases)), "cases": cases}


def _write(filename: str, payload: dict) -> None:
    path = GOLDEN_DIR / filename
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    _log(f"wrote {path} ({path.stat().st_size:,} bytes)")


def main() -> int:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    _write("engine_step.json", capture_engine_step())
    _write("compile_engine.json", capture_compile_engine())
    _write("per_op.json", capture_per_op())
    _log(f"golden capture complete in {time.monotonic() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
