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

Integrity guards (these fixtures are Gate 2's frozen deletion anchor):

- The starting tree must be clean apart from ``parity_tests/goldens/``
  itself (the script's own output, overwritten wholesale — a leftover
  partial capture cannot influence a new one). Any other tracked-file
  modification aborts before a single case is evaluated, so a header can
  never record an unidentifiable source state.
- All three payloads are captured in full BEFORE any file is written. A
  mid-capture failure (e.g. ``pytest.skip`` from a missing WideEP data set)
  therefore leaves the committed goldens byte-untouched instead of a
  partially rewritten mix.
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


# Repo-root-relative prefix of this script's own output directory —
# porcelain paths are always root-relative regardless of cwd.
_GOLDEN_REL_PREFIX = "aic-core/rust/aiconfigurator-core/parity_tests/goldens/"


def _dirty_paths(porcelain: str) -> list[str]:
    """Tracked-file modifications that would make the capture state
    unidentifiable: every ``git status --porcelain`` entry except untracked
    files (which do not affect ``git describe --dirty`` or the capture) and
    the goldens directory itself (this script's output — about to be
    rewritten wholesale, so its pre-state is not a capture input)."""
    dirty: list[str] = []
    for line in porcelain.splitlines():
        if not line.strip() or line.startswith("??"):
            continue
        # Rename entries carry both sides: "R  old -> new".
        paths = line[3:].split(" -> ")
        if all(path.strip().strip('"').startswith(_GOLDEN_REL_PREFIX) for path in paths):
            continue
        dirty.append(line[3:].strip())
    return dirty


def _require_clean_tree() -> None:
    """Abort a capture that starts from a dirty tree (goldens excluded, see
    `_dirty_paths`) BEFORE any evaluation or write."""
    try:
        porcelain = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=_PARITY_DIR, text=True, stderr=subprocess.DEVNULL
        )
    except Exception:
        return  # not a git checkout — capture identity is "unknown" anyway
    dirty = _dirty_paths(porcelain)
    if dirty:
        raise RuntimeError(
            "golden capture requires a clean tree (only parity_tests/goldens/ "
            "may differ): commit or stash these paths first: " + ", ".join(sorted(dirty))
        )


# Captured ONCE at import, before any write. NO `--dirty` suffix: the
# clean-tree guard rejects every tracked modification except goldens-dir
# output from a previous run, so a dirty flag could only ever echo this
# script's own artifacts — omitting it keeps the documented run-twice-and-
# diff workflow byte-idempotent (the second run re-stamps identical headers).
_GIT_STATE = {
    "git_describe": _git("describe", "--always"),
    "git_head": _git("rev-parse", "HEAD"),
}


def _header(**counts: int) -> dict:
    # Deliberately NO wall-clock timestamp: two captures of the same tree
    # must be byte-identical; the HEAD sha is the only capture identity.
    return {
        "command": COMMAND,
        "engine_step_backend": ENGINE_STEP_BACKEND,
        **_GIT_STATE,
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


# The capture helpers are shared with the pytest suites, where a missing perf
# database surfaces as ``pytest.skip`` — a ``Skipped`` outcome that subclasses
# BaseException and would otherwise leak out of this script as a cryptic
# traceback. Caught explicitly below and converted into a clear error.
try:
    from _pytest.outcomes import Skipped as _PytestSkipped
except Exception:  # pragma: no cover — pytest is a hard dependency here
    _PytestSkipped = ()  # type: ignore[assignment]


def main() -> int:
    _require_clean_tree()
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    # Capture ALL payloads before writing ANY tracked fixture: a late
    # failure must leave the committed goldens byte-untouched.
    try:
        payloads = {
            "engine_step.json": capture_engine_step(),
            "compile_engine.json": capture_compile_engine(),
            "per_op.json": capture_per_op(),
        }
    except _PytestSkipped as exc:
        raise RuntimeError(
            "golden capture aborted BEFORE any write — a capture helper "
            f"skipped ({exc}). Regeneration requires the FULL systems data "
            "set (e.g. the WideEP databases); the committed goldens are "
            "untouched."
        ) from exc
    for filename, payload in payloads.items():
        _write(filename, payload)
    _log(f"golden capture complete in {time.monotonic() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
