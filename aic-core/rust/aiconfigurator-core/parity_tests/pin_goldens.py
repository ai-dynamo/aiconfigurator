# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pin golden records from the LIVE rust engine (post-freeze workflow).

The Python engine-step path is gone (dedup-plan Gate 3), so the Python-era
golden records in ``parity_tests/goldens/`` can never be recaptured — they
are the frozen reference. This script is the post-freeze maintenance tool:

- **Default (append-only)**: compute and add records ONLY for (case, surface)
  pairs / reference keys / per-op cases that the test matrices declare but the
  fixtures lack — i.e. cases added AFTER the freeze. Existing records are
  never touched.
- ``--refresh KEY ...``: recompute the named existing records from the live
  rust engine. For a deliberate, reviewed rust-side modeling change; the
  golden diff in the PR is the review artifact.
- ``--refresh-all``: recompute every record. The full-diff analog of the
  retired python-era recapture — use with the same discipline (never to
  silence an unexplained parity failure: a red Rust-vs-golden test means the
  engine drifted from the frozen reference).

Every record this script writes or rewrites is provenance-marked in the
file's top-level ``post_freeze_pins`` map (key -> the git HEAD of the pin),
so python-era frozen values and rust-pinned values stay distinguishable.

Run from the repository root::

    .venv/bin/python aic-core/rust/aiconfigurator-core/parity_tests/pin_goldens.py

Contract carried over from the retired capture script: byte-reproducible
output (thread caps pinned, sorted keys, full-repr floats, no wall-clock
timestamps), a clean-tree guard before any evaluation, and all payloads
computed in full before any file is written.
"""

from __future__ import annotations

import os

# Thread caps: pinned before any numpy/pandas import so a pin run is
# byte-reproducible across hosts and runs.
THREAD_CAP_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "RAYON_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
os.environ.update(THREAD_CAP_ENV)

# Pin the LIVE engine explicitly: the compiled rust engine is the only step
# executor, and an ambient env override must not leak deprecation warnings
# (or a future value) into a pin run.
os.environ["AICONFIGURATOR_ENGINE_STEP_BACKEND"] = "rust"

import argparse
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
PIN_MAP_KEY = "post_freeze_pins"

# The reused suite helpers surface a missing perf database as ``pytest.skip``
# — a ``Skipped`` outcome that subclasses BaseException and would leak out of
# this script as a cryptic traceback. Caught in ``main`` and converted into a
# clear before-any-write abort.
try:
    from _pytest.outcomes import Skipped as _PytestSkipped
except Exception:  # pragma: no cover — pytest is a hard dependency here
    _PytestSkipped = ()  # type: ignore[assignment]


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=_PARITY_DIR, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


_GOLDEN_REL_PREFIX = "aic-core/rust/aiconfigurator-core/parity_tests/goldens/"


def _dirty_paths(porcelain: str) -> list[str]:
    """Every porcelain entry except the goldens themselves (this script's
    output). Untracked files count as dirty too — unlike `git describe`,
    a pin's VALUE can depend on them (an untracked parquet under
    systems/data changes what the live engine loads), so a clean header
    must mean a fully committed input state."""
    dirty: list[str] = []
    for line in porcelain.splitlines():
        if not line.strip():
            continue
        paths = line[3:].split(" -> ")
        if all(path.strip().strip('"').startswith(_GOLDEN_REL_PREFIX) for path in paths):
            continue
        dirty.append(line[3:].strip())
    return dirty


def _require_clean_tree() -> None:
    try:
        porcelain = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=_PARITY_DIR, text=True, stderr=subprocess.DEVNULL
        )
    except Exception:
        return  # not a git checkout — pin identity is "unknown" anyway
    dirty = _dirty_paths(porcelain)
    if dirty:
        raise RuntimeError(
            "golden pinning requires a clean tree (only parity_tests/goldens/ "
            "may differ): commit or stash these paths first: " + ", ".join(sorted(dirty))
        )


_GIT_HEAD = _git("rev-parse", "HEAD")


def _require_finite(value: float, context: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"non-finite golden value for {context}: {value!r}")
    return value


def _encode_metrics(metrics: dict) -> dict:
    """Same record encoding as the frozen fixtures: ``{"error": kind}`` when
    every metric raised with a single kind, else ``{"values": ...}``."""
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


def _load(filename: str) -> dict:
    path = GOLDEN_DIR / filename
    if not path.is_file():
        raise RuntimeError(f"missing golden fixture {path} — the frozen fixtures must exist to pin against")
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Live-rust computations, mirroring each suite's live side exactly.
# ---------------------------------------------------------------------------


def _engine_step_record(case, surface: str) -> dict:
    metrics = engine_parity._surface_metrics(case, surface)
    return _encode_metrics(metrics)


def _compile_case_references(case) -> dict[str, float]:
    handle = compile_parity._compile_handle(case)
    osl = max(case.osl, 2)
    ctx, gen, _total = handle.run_static(
        batch_size=case.batch_size, isl=case.isl, osl=osl, prefix=case.prefix, stride=1
    )
    return {
        "static_ctx": float(ctx),
        "static_gen": float(gen),
        "mixed_step": float(handle.mixed_step_latency(case.isl, case.batch_size, case.isl, osl, case.prefix)),
        "decode_step": float(handle.decode_step_latency(case.batch_size, case.isl, osl)),
    }


def _compile_scenario_references() -> dict[str, float]:
    """The fixed-scenario reference keys (chunked prefill, imbalance scales,
    WideEP), mirroring the corresponding test bodies."""
    references: dict[str, float] = {}

    chunk_case = compile_parity._SUBSET_BY_ID[compile_parity._CHUNKED_PREFILL_CASE_ID].values[0]
    chunk_handle = compile_parity._compile_handle(chunk_case)
    for ctx_tokens, gen_tokens, isl, osl, prefix in compile_parity._CHUNKED_PREFILL_SHAPES:
        key = compile_parity._chunked_prefill_key(ctx_tokens, gen_tokens, isl, osl, prefix)
        references[key] = float(chunk_handle.mixed_step_latency(ctx_tokens, gen_tokens, isl, osl, prefix))

    imb_case = compile_parity._SUBSET_BY_ID[compile_parity._IMBALANCE_CASE_ID].values[0]
    imb_handle = compile_parity._compile_handle(imb_case)
    imb_osl = max(imb_case.osl, 2)
    imb_ctx, imb_gen, _ = imb_handle.run_static(
        batch_size=imb_case.batch_size,
        isl=imb_case.isl,
        osl=imb_osl,
        prefix=imb_case.prefix,
        seq_imbalance_correction_scale=compile_parity._IMBALANCE_CTX_SCALE,
        gen_seq_imbalance_correction_scale=compile_parity._IMBALANCE_GEN_SCALE,
        stride=1,
    )
    references["imbalance_scale::static_ctx"] = float(imb_ctx)
    references["imbalance_scale::static_gen"] = float(imb_gen)
    references["imbalance_scale::mixed_step"] = float(
        imb_handle.mixed_step_latency(
            imb_case.isl,
            imb_case.batch_size,
            imb_case.isl,
            imb_osl,
            imb_case.prefix,
            seq_imbalance_correction_scale=compile_parity._IMBALANCE_CTX_SCALE,
            gen_seq_imbalance_correction_scale=compile_parity._IMBALANCE_GEN_SCALE,
        )
    )
    references["imbalance_scale::decode_step"] = float(
        imb_handle.decode_step_latency(
            imb_case.batch_size,
            imb_case.isl,
            imb_osl,
            gen_seq_imbalance_correction_scale=compile_parity._IMBALANCE_GEN_SCALE,
        )
    )

    _m, _b, _d, sglang_spec = compile_parity._build_wideep_sglang()
    sglang_handle = compile_parity._handle_from_spec_json(sglang_spec)
    ws_ctx, ws_gen, _ = sglang_handle.run_static(batch_size=1, isl=1024, osl=4, prefix=0, stride=1)
    references["wideep_sglang::static_ctx"] = float(ws_ctx)
    references["wideep_sglang::static_gen"] = float(ws_gen)
    references["wideep_sglang::mixed_step"] = float(sglang_handle.mixed_step_latency(1024, 2, 1024, 4, 0))
    references["wideep_sglang::decode_step"] = float(sglang_handle.decode_step_latency(2, 1024, 4))

    _m, _b, _d, trtllm_spec = compile_parity._build_wideep_trtllm()
    trtllm_handle = compile_parity._handle_from_spec_json(trtllm_spec)
    wt_ctx, wt_gen, _ = trtllm_handle.run_static(batch_size=1, isl=1024, osl=4, prefix=0, stride=1)
    references["wideep_trtllm::static_ctx"] = float(wt_ctx)
    references["wideep_trtllm::static_gen"] = float(wt_gen)

    _m, _b, _d, trtllm_rc20_spec = compile_parity._build_wideep_trtllm("1.3.0rc20")
    trtllm_rc20_handle = compile_parity._handle_from_spec_json(trtllm_rc20_spec)
    wt20_ctx, wt20_gen, _ = trtllm_rc20_handle.run_static(batch_size=1, isl=1024, osl=4, prefix=0, stride=1)
    references["wideep_trtllm_rc20_fallback::static_ctx"] = float(wt20_ctx)
    references["wideep_trtllm_rc20_fallback::static_gen"] = float(wt20_gen)

    missing = [key for key in _SCENARIO_KEYS if key not in references]
    assert not missing, f"scenario compute drifted from _SCENARIO_KEYS: {missing}"
    return references


def _per_op_record(case) -> dict:
    handle = compile_parity._compile_handle(case)
    ctx_entries, gen_entries = handle.run_static_per_op(
        batch_size=case.batch_size, isl=case.isl, osl=max(case.osl, 2), prefix=case.prefix, stride=1
    )

    def pack(entries) -> dict:
        folded = compile_parity._fold_per_op_entries(entries)
        return {
            name: {
                "latency_ms": _require_finite(latency, f"{name}::latency_ms"),
                "energy_wms": _require_finite(energy, f"{name}::energy_wms"),
                "source": str(source),
            }
            for name, (latency, energy, source) in folded.items()
        }

    return {"context": pack(ctx_entries), "generation": pack(gen_entries)}


# ---------------------------------------------------------------------------
# Expected key populations (from the same single-source-of-truth matrices the
# tests parametrize over) and the pin planner.
# ---------------------------------------------------------------------------


def _engine_step_population() -> dict[tuple[str, str], object]:
    population: dict[tuple[str, str], object] = {}
    for params, surfaces in engine_parity.ENGINE_STEP_GOLDEN_MATRIX:
        for param in params:
            (case,) = param.values
            for surface in surfaces:
                population[(param.id, surface)] = case
    return population


# Single source for the fixed-scenario reference keys: consumed by BOTH the
# population (planner) and `_compile_scenario_references` (compute), so a
# key can never exist in one and not the other. Keep in lockstep with the
# corresponding test classes in test_compile_engine_parity.py.
_SCENARIO_KEYS = (
    "imbalance_scale::static_ctx",
    "imbalance_scale::static_gen",
    "imbalance_scale::mixed_step",
    "imbalance_scale::decode_step",
    "wideep_sglang::static_ctx",
    "wideep_sglang::static_gen",
    "wideep_sglang::mixed_step",
    "wideep_sglang::decode_step",
    "wideep_trtllm::static_ctx",
    "wideep_trtllm::static_gen",
    "wideep_trtllm_rc20_fallback::static_ctx",
    "wideep_trtllm_rc20_fallback::static_gen",
)


def _compile_reference_population() -> dict[str, object]:
    population: dict[str, object] = {}
    for param in compile_parity._SUBSET_CASES:
        (case,) = param.values
        for metric in ("static_ctx", "static_gen", "mixed_step", "decode_step"):
            population[f"{param.id}::{metric}"] = ("case", case)
    for ctx_tokens, gen_tokens, isl, osl, prefix in compile_parity._CHUNKED_PREFILL_SHAPES:
        population[compile_parity._chunked_prefill_key(ctx_tokens, gen_tokens, isl, osl, prefix)] = ("scenario", None)
    for key in _SCENARIO_KEYS:
        population[key] = ("scenario", None)
    return population


def _plan(existing: set, population: set, refresh: set[str], refresh_all: bool, encode) -> list:
    """Resolve which keys to (re)compute: missing ones always; existing ones
    only when named by ``--refresh`` (or ``--refresh-all``). ``refresh`` keys
    were validated against the full key universe up front in ``main`` —
    BEFORE any live-engine evaluation — so a typo aborts in milliseconds
    instead of after the compute loops."""
    missing = population - existing
    named = population if refresh_all else {key for key in population if encode(key) in refresh}
    return sorted(missing | (named & existing), key=encode)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--refresh",
        nargs="*",
        default=[],
        metavar="KEY",
        help="existing record keys to recompute from the live rust engine "
        "(engine_step: 'case_id::surface'; compile_engine: the reference key; per_op: the case id)",
    )
    parser.add_argument(
        "--refresh-all",
        action="store_true",
        help="recompute EVERY record from the live rust engine (full-diff review workflow)",
    )
    args = parser.parse_args(argv)
    refresh = set(args.refresh)

    _require_clean_tree()
    started = time.monotonic()

    engine_step = _load("engine_step.json")
    compile_engine = _load("compile_engine.json")
    per_op = _load("per_op.json")

    pinned: dict[str, list[str]] = {"engine_step.json": [], "compile_engine.json": [], "per_op.json": []}

    # Build every key population FIRST and reject unknown --refresh names
    # BEFORE any live-engine evaluation (with or without --refresh-all): a
    # typo must abort in milliseconds, not after the compute loops.
    population = _engine_step_population()
    ref_population = _compile_reference_population()
    per_op_population = {param.id: param.values[0] for param in compile_parity._SUBSET_CASES}
    encode = lambda key: f"{key[0]}::{key[1]}"
    key_universe = {encode(key) for key in population} | set(ref_population) | set(per_op_population)
    unmatched = refresh - key_universe
    if unmatched:
        raise SystemExit(f"--refresh names keys outside every test matrix: {sorted(unmatched)}")

    # Compute ALL records before writing ANY file (a mid-run failure leaves
    # the committed goldens byte-untouched). The reused test helpers surface
    # a missing dataset as pytest.skip — a BaseException that would
    # otherwise escape as a cryptic traceback, so convert it into a clear
    # abort (same guard the retired regenerate_goldens.py carried).
    try:
        # --- engine_step.json ---------------------------------------------
        existing = {(case_id, surface) for case_id, surfaces in engine_step["cases"].items() for surface in surfaces}
        last_cleared_case_id = None
        for case_id, surface in _plan(existing, set(population), refresh, args.refresh_all, encode):
            case = population[(case_id, surface)]
            if case_id != last_cleared_case_id:
                # One clear per CASE (the plan is case-sorted): the surfaces
                # of one case share the engine handle instead of redoing the
                # compile + Rust perf-DB load four times.
                engine_parity.rust_engine_step._engine_handle_cache_clear()
                last_cleared_case_id = case_id
            record_started = time.monotonic()
            engine_step["cases"].setdefault(case_id, {})[surface] = _engine_step_record(case, surface)
            pinned["engine_step.json"].append(f"{case_id}::{surface}")
            _log(f"[engine_step] pinned {case_id} :: {surface} ({time.monotonic() - record_started:.1f}s)")

        # --- compile_engine.json --------------------------------------------
        references = compile_engine["references"]
        todo = _plan(set(references), set(ref_population), refresh, args.refresh_all, str)
        case_cache: dict[str, dict[str, float]] = {}
        scenario_refs: dict[str, float] | None = None
        for key in todo:
            kind, case = ref_population[key]
            if kind == "case":
                case_id, metric = key.rsplit("::", 1)
                values = case_cache.get(case_id)
                if values is None:
                    values = case_cache[case_id] = _compile_case_references(case)
                references[key] = _require_finite(values[metric], key)
            else:
                if scenario_refs is None:
                    scenario_refs = _compile_scenario_references()
                references[key] = _require_finite(scenario_refs[key], key)
            pinned["compile_engine.json"].append(key)
            _log(f"[compile_engine] pinned {key}")

        # --- per_op.json ----------------------------------------------------
        for case_id in _plan(set(per_op["cases"]), set(per_op_population), refresh, args.refresh_all, str):
            record_started = time.monotonic()
            per_op["cases"][case_id] = _per_op_record(per_op_population[case_id])
            pinned["per_op.json"].append(case_id)
            _log(f"[per_op] pinned {case_id} ({time.monotonic() - record_started:.1f}s)")
    except _PytestSkipped as exc:
        raise RuntimeError(
            "golden pinning aborted BEFORE any write — a reused test helper "
            f"skipped ({exc}). Pinning these records requires the FULL systems "
            "data set (e.g. the WideEP databases); the committed goldens are "
            "untouched."
        ) from exc

    total = sum(len(keys) for keys in pinned.values())
    if total == 0:
        _log("nothing to pin — every matrix record already exists (use --refresh/--refresh-all to recompute)")
        return 0

    # Provenance: every pinned key maps to the HEAD it was computed at.
    for filename, payload in (
        ("engine_step.json", engine_step),
        ("compile_engine.json", compile_engine),
        ("per_op.json", per_op),
    ):
        if not pinned[filename]:
            continue
        pin_map = payload.setdefault(PIN_MAP_KEY, {})
        for key in pinned[filename]:
            pin_map[key] = {"engine": "rust", "git_head": _GIT_HEAD}
        path = GOLDEN_DIR / filename
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
        _log(f"wrote {path} ({len(pinned[filename])} pinned records)")

    _log(f"pinned {total} records in {time.monotonic() - started:.0f}s — review the golden diff before committing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
