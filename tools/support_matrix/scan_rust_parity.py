#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Full support-matrix Rust-vs-Python parity scan.

Iterates every PASS row in `aic-core/src/aiconfigurator_core/systems/support_matrix/*.csv`
and verifies the Rust engine-step backend matches the Python reference within
tolerance. Results land in a single SQLite file so the run is resumable
across crashes, SIGINTs, or multi-day wall-clock budgets.

See the support-matrix scan design notes in the project docs for the full
design rationale.
"""

import argparse
import csv
import logging
import os
import signal
import sqlite3
import subprocess
import sys
import time
import traceback
from concurrent.futures import (
    BrokenExecutor,
    ProcessPoolExecutor,
    as_completed,
)
from concurrent.futures import (
    TimeoutError as FutureTimeoutError,
)
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

from tools.support_matrix.support_matrix import (
    DEFAULT_ENGINE_STEP_COMPARISON_ATOL,
    SupportMatrix,
    TestConstraints,
    _compare_pareto_dfs,
    _get_test_constraints,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_BASELINE_DIR = _REPO_ROOT / "aic-core" / "src" / "aiconfigurator_core" / "systems" / "support_matrix"
DEFAULT_DB_PATH = _REPO_ROOT / "aic-core" / "rust" / "aiconfigurator-core" / "parity_tests" / "scan.sqlite"

SCAN_MODE_PROBE = "probe_only"
SCAN_MODE_PARETO = "pareto_only"
SCAN_MODE_BOTH = "both"
ALL_SCAN_MODES = (SCAN_MODE_PROBE, SCAN_MODE_PARETO, SCAN_MODE_BOTH)

# Tight tolerances per the scan plan (<1% strict per-row, 5% envelope fallback).
PROBE_RTOL = 0.01
PROBE_ATOL = 1e-3  # ms
PARETO_STRICT_RTOL = 0.01
PARETO_STRICT_ATOL = DEFAULT_ENGINE_STEP_COMPARISON_ATOL
PARETO_ENVELOPE_RTOL = 0.05
PARETO_ENVELOPE_ATOL = DEFAULT_ENGINE_STEP_COMPARISON_ATOL

PROBE_STATUS_PASS = "PASS"
PROBE_STATUS_DRIFT = "DRIFT"
PROBE_STATUS_PY_ERROR = "PY_ERROR_ONLY"
PROBE_STATUS_RUST_ERROR = "RUST_ERROR_ONLY"
PROBE_STATUS_BOTH_ERROR = "BOTH_ERROR_PASS"
PROBE_STATUS_SKIPPED = "SKIPPED"

PARETO_STATUS_STRICT_PASS = "STRICT_PASS"
PARETO_STATUS_ENVELOPE_PASS = "ENVELOPE_PASS"
PARETO_STATUS_DRIFT = "DRIFT"
PARETO_STATUS_REGRESSION = "REGRESSION"
PARETO_STATUS_ERROR = "ERROR"

# Default per-entry wall-clock budget (a wide MoE disagg sweep can spike a few
# minutes; 15 min should comfortably cover the tail).
DEFAULT_PER_ENTRY_TIMEOUT_SEC = 900

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS entries (
    entry_key       TEXT PRIMARY KEY,
    model           TEXT NOT NULL,
    architecture    TEXT NOT NULL,
    system          TEXT NOT NULL,
    backend         TEXT NOT NULL,
    version         TEXT NOT NULL,
    mode            TEXT NOT NULL,
    baseline_status TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS probe_results (
    entry_key       TEXT NOT NULL REFERENCES entries(entry_key),
    probe_shape     TEXT NOT NULL,
    python_ttft_ms  REAL,
    python_tpot_ms  REAL,
    rust_ttft_ms    REAL,
    rust_tpot_ms    REAL,
    ttft_drift_pct  REAL,
    tpot_drift_pct  REAL,
    python_err      TEXT,
    rust_err        TEXT,
    status          TEXT NOT NULL,
    duration_ms     REAL,
    completed_at    TEXT,
    PRIMARY KEY (entry_key, probe_shape)
);

CREATE TABLE IF NOT EXISTS pareto_results (
    entry_key             TEXT PRIMARY KEY REFERENCES entries(entry_key),
    python_status         TEXT,
    rust_status           TEXT,
    strict_max_drift_pct  REAL,
    frontier_envelope_pct REAL,
    comparison_outcome    TEXT NOT NULL,
    error_msg             TEXT,
    duration_ms           REAL,
    completed_at          TEXT
);

CREATE TABLE IF NOT EXISTS run_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_entries_system  ON entries(system);
CREATE INDEX IF NOT EXISTS idx_entries_backend ON entries(backend);
CREATE INDEX IF NOT EXISTS idx_probe_status    ON probe_results(status);
CREATE INDEX IF NOT EXISTS idx_pareto_outcome  ON pareto_results(comparison_outcome);
"""

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Entry:
    model: str
    architecture: str
    system: str
    backend: str
    version: str
    mode: str
    baseline_status: str

    @property
    def key(self) -> str:
        return f"{self.model}|{self.system}|{self.backend}|{self.version}|{self.mode}"


@dataclass
class ProbeRecord:
    entry_key: str
    probe_shape: str
    python_ttft_ms: float | None
    python_tpot_ms: float | None
    rust_ttft_ms: float | None
    rust_tpot_ms: float | None
    ttft_drift_pct: float | None
    tpot_drift_pct: float | None
    python_err: str | None
    rust_err: str | None
    status: str
    duration_ms: float
    completed_at: str


@dataclass
class ParetoRecord:
    entry_key: str
    python_status: str | None
    rust_status: str | None
    strict_max_drift_pct: float | None
    frontier_envelope_pct: float | None
    comparison_outcome: str
    error_msg: str | None
    duration_ms: float
    completed_at: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _drift_pct(python_value: float | None, rust_value: float | None) -> float | None:
    """Symmetric relative drift: (rust - python) / max(|python|, |rust|, eps)."""
    if python_value is None or rust_value is None:
        return None
    denom = max(abs(python_value), abs(rust_value), 1e-9)
    return (rust_value - python_value) / denom * 100.0


def _shorten(msg: str, limit: int = 400) -> str:
    msg = " ".join(msg.split())
    if len(msg) <= limit:
        return msg
    return msg[: limit - 1] + "…"


def _format_exc(exc: BaseException) -> str:
    return _shorten(f"{type(exc).__name__}: {exc}")


def _git_head_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------


def load_entries(baseline_dir: Path, *, status_filter: tuple[str, ...] = ("PASS",)) -> list[Entry]:
    """Read every per-system support-matrix CSV and return matching rows."""
    entries: list[Entry] = []
    for csv_path in sorted(baseline_dir.glob("*.csv")):
        with csv_path.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                status = row.get("Status", "").strip()
                if status_filter and status not in status_filter:
                    continue
                entries.append(
                    Entry(
                        model=row["HuggingFaceID"].strip(),
                        architecture=row["Architecture"].strip(),
                        system=row["System"].strip(),
                        backend=row["Backend"].strip(),
                        version=row["Version"].strip(),
                        mode=row["Mode"].strip(),
                        baseline_status=status,
                    )
                )
    return entries


# ---------------------------------------------------------------------------
# SQLite IO
# ---------------------------------------------------------------------------


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, isolation_level=None, timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(_connect(db_path)) as conn:
        for stmt in _SCHEMA_SQL.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                conn.execute(stmt)


def seed_entries(db_path: Path, entries: list[Entry]) -> int:
    """Insert new entries; return count newly inserted."""
    inserted = 0
    with closing(_connect(db_path)) as conn:
        conn.execute("BEGIN")
        for entry in entries:
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO entries
                    (entry_key, model, architecture, system, backend, version, mode, baseline_status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.key,
                    entry.model,
                    entry.architecture,
                    entry.system,
                    entry.backend,
                    entry.version,
                    entry.mode,
                    entry.baseline_status,
                ),
            )
            inserted += cur.rowcount
        conn.execute("COMMIT")
    return inserted


def write_run_meta(db_path: Path, items: dict[str, str]) -> None:
    with closing(_connect(db_path)) as conn:
        conn.execute("BEGIN")
        for key, value in items.items():
            conn.execute(
                "INSERT OR REPLACE INTO run_meta (key, value) VALUES (?, ?)",
                (key, value),
            )
        conn.execute("COMMIT")


def read_run_meta(db_path: Path, key: str) -> str | None:
    with closing(_connect(db_path)) as conn:
        row = conn.execute("SELECT value FROM run_meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None


def pending_entries_for_probe(db_path: Path) -> set[str]:
    with closing(_connect(db_path)) as conn:
        done = {
            row[0]
            for row in conn.execute(
                "SELECT entry_key FROM probe_results WHERE status <> ?",
                ("PENDING",),
            )
        }
        all_keys = {row[0] for row in conn.execute("SELECT entry_key FROM entries")}
    return all_keys - done


def pending_entries_for_pareto(db_path: Path) -> set[str]:
    with closing(_connect(db_path)) as conn:
        done = {
            row[0]
            for row in conn.execute(
                "SELECT entry_key FROM pareto_results WHERE comparison_outcome <> ?",
                ("PENDING",),
            )
        }
        all_keys = {row[0] for row in conn.execute("SELECT entry_key FROM entries")}
    return all_keys - done


def write_probe_record(conn: sqlite3.Connection, record: ProbeRecord) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO probe_results
            (entry_key, probe_shape, python_ttft_ms, python_tpot_ms,
             rust_ttft_ms, rust_tpot_ms, ttft_drift_pct, tpot_drift_pct,
             python_err, rust_err, status, duration_ms, completed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.entry_key,
            record.probe_shape,
            record.python_ttft_ms,
            record.python_tpot_ms,
            record.rust_ttft_ms,
            record.rust_tpot_ms,
            record.ttft_drift_pct,
            record.tpot_drift_pct,
            record.python_err,
            record.rust_err,
            record.status,
            record.duration_ms,
            record.completed_at,
        ),
    )


def write_pareto_record(conn: sqlite3.Connection, record: ParetoRecord) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO pareto_results
            (entry_key, python_status, rust_status, strict_max_drift_pct,
             frontier_envelope_pct, comparison_outcome, error_msg,
             duration_ms, completed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.entry_key,
            record.python_status,
            record.rust_status,
            record.strict_max_drift_pct,
            record.frontier_envelope_pct,
            record.comparison_outcome,
            record.error_msg,
            record.duration_ms,
            record.completed_at,
        ),
    )


# ---------------------------------------------------------------------------
# Probe layer (cli_estimate Python vs Rust)
# ---------------------------------------------------------------------------


def _probe_parallelism(total_gpus: int) -> tuple[int, int, int]:
    """Pick (tp, moe_tp, moe_ep) for a single-shot probe.

    Strategy: cap TP at 8 and run pure-TP (moe_tp=1, moe_ep=tp) which is the
    most common valid config. Larger entries (>8 GPU) still probe at tp=8.
    """
    tp = min(8, max(1, total_gpus))
    return tp, 1, tp


def _run_probe(
    entry: Entry, constraints: TestConstraints, backend_label: str
) -> tuple[float | None, float | None, str | None]:
    """Run one cli_estimate call and return (ttft, tpot, err)."""
    # Lazy-import inside the worker to keep startup quick when a worker
    # is created just for the probe layer.
    from aiconfigurator.cli.api import cli_estimate

    tp, moe_tp, moe_ep = _probe_parallelism(constraints.total_gpus)
    common_kwargs = dict(
        model_path=entry.model,
        system_name=entry.system,
        backend_name=entry.backend,
        backend_version=entry.version,
        isl=constraints.isl,
        osl=constraints.osl,
        prefix=constraints.prefix,
        tp_size=tp,
        moe_tp_size=moe_tp,
        moe_ep_size=moe_ep,
        engine_step_backend=backend_label,
    )

    try:
        if entry.mode == "agg":
            result = cli_estimate(mode="agg", batch_size=16, **common_kwargs)
        elif entry.mode == "disagg":
            result = cli_estimate(
                mode="disagg",
                prefill_batch_size=4,
                prefill_num_workers=1,
                decode_batch_size=16,
                decode_num_workers=1,
                **common_kwargs,
            )
        else:
            return None, None, f"unsupported mode {entry.mode!r}"
    except Exception as exc:
        return None, None, _format_exc(exc)

    return float(result.ttft), float(result.tpot), None


def probe_entry(entry: Entry) -> ProbeRecord:
    """Probe one entry: cli_estimate(python) vs cli_estimate(rust)."""
    started = time.monotonic()
    constraints = _get_test_constraints(entry.model)
    shape_str = (
        f"isl={constraints.isl},osl={constraints.osl},prefix={constraints.prefix},total_gpus={constraints.total_gpus}"
    )

    python_ttft, python_tpot, python_err = _run_probe(entry, constraints, backend_label="python")
    rust_ttft, rust_tpot, rust_err = _run_probe(entry, constraints, backend_label="rust")

    ttft_drift = _drift_pct(python_ttft, rust_ttft)
    tpot_drift = _drift_pct(python_tpot, rust_tpot)

    status = _classify_probe(
        python_err=python_err,
        rust_err=rust_err,
        ttft_drift=ttft_drift,
        tpot_drift=tpot_drift,
    )
    duration_ms = (time.monotonic() - started) * 1000.0
    return ProbeRecord(
        entry_key=entry.key,
        probe_shape=shape_str,
        python_ttft_ms=python_ttft,
        python_tpot_ms=python_tpot,
        rust_ttft_ms=rust_ttft,
        rust_tpot_ms=rust_tpot,
        ttft_drift_pct=ttft_drift,
        tpot_drift_pct=tpot_drift,
        python_err=python_err,
        rust_err=rust_err,
        status=status,
        duration_ms=duration_ms,
        completed_at=_now_iso(),
    )


def _classify_probe(
    *,
    python_err: str | None,
    rust_err: str | None,
    ttft_drift: float | None,
    tpot_drift: float | None,
) -> str:
    if python_err and rust_err:
        return PROBE_STATUS_BOTH_ERROR
    if python_err and not rust_err:
        return PROBE_STATUS_PY_ERROR
    if rust_err and not python_err:
        return PROBE_STATUS_RUST_ERROR
    # Both succeeded; check drift.
    rtol_pct = PROBE_RTOL * 100.0
    if ttft_drift is not None and abs(ttft_drift) > rtol_pct:
        return PROBE_STATUS_DRIFT
    if tpot_drift is not None and abs(tpot_drift) > rtol_pct:
        return PROBE_STATUS_DRIFT
    return PROBE_STATUS_PASS


# ---------------------------------------------------------------------------
# Pareto layer (cli_default Python vs Rust via existing SupportMatrix)
# ---------------------------------------------------------------------------


# Lazy per-process SupportMatrix singleton; built on first use to avoid
# duplicating the model + database discovery across every entry.
_worker_matrix: SupportMatrix | None = None
_worker_rust_autobuild_set = False


def _get_worker_matrix() -> SupportMatrix:
    global _worker_matrix, _worker_rust_autobuild_set
    if not _worker_rust_autobuild_set:
        os.environ.setdefault("AICONFIGURATOR_RUST_CORE_AUTOBUILD", "1")
        _worker_rust_autobuild_set = True
    if _worker_matrix is None:
        _worker_matrix = SupportMatrix(
            compare_engine_step_backends=True,
            engine_step_comparison_rtol=PARETO_STRICT_RTOL,
            engine_step_comparison_atol=PARETO_STRICT_ATOL,
            engine_step_frontier_rtol=PARETO_ENVELOPE_RTOL,
            engine_step_frontier_atol=PARETO_ENVELOPE_ATOL,
        )
    return _worker_matrix


def pareto_entry(entry: Entry) -> ParetoRecord:
    """Run cli_default twice (python + rust) and compare via _compare_pareto_dfs."""
    started = time.monotonic()
    _get_worker_matrix()  # lazy-init the per-process databases on first call
    constraints = _get_test_constraints(entry.model)

    python_df = None
    rust_df = None
    python_err: str | None = None
    rust_err: str | None = None
    try:
        python_df = SupportMatrix._run_mode(
            mode=entry.mode,
            model=entry.model,
            system=entry.system,
            backend=entry.backend,
            version=entry.version,
            constraints=constraints,
            engine_step_backend="python",
        )
    except Exception as exc:
        python_err = _format_exc(exc)

    try:
        rust_df = SupportMatrix._run_mode(
            mode=entry.mode,
            model=entry.model,
            system=entry.system,
            backend=entry.backend,
            version=entry.version,
            constraints=constraints,
            engine_step_backend="rust",
        )
    except Exception as exc:
        rust_err = _format_exc(exc)

    duration_ms = (time.monotonic() - started) * 1000.0
    completed_at = _now_iso()

    python_ok = python_err is None and python_df is not None and not python_df.empty
    rust_ok = rust_err is None and rust_df is not None and not rust_df.empty

    # Regression: baseline says PASS but Python or Rust failed this run.
    if not python_ok:
        outcome = PARETO_STATUS_ERROR
        return ParetoRecord(
            entry_key=entry.key,
            python_status="ERROR" if python_err else "EMPTY",
            rust_status="ERROR" if rust_err else ("EMPTY" if not rust_ok else "PASS"),
            strict_max_drift_pct=None,
            frontier_envelope_pct=None,
            comparison_outcome=outcome,
            error_msg=python_err or "Python pareto_df empty",
            duration_ms=duration_ms,
            completed_at=completed_at,
        )

    if not rust_ok:
        return ParetoRecord(
            entry_key=entry.key,
            python_status="PASS",
            rust_status="ERROR" if rust_err else "EMPTY",
            strict_max_drift_pct=None,
            frontier_envelope_pct=None,
            comparison_outcome=PARETO_STATUS_REGRESSION,
            error_msg=rust_err or "Rust pareto_df empty",
            duration_ms=duration_ms,
            completed_at=completed_at,
        )

    mismatch = _compare_pareto_dfs(
        python_df,
        rust_df,
        rtol=PARETO_STRICT_RTOL,
        atol=PARETO_STRICT_ATOL,
        frontier_rtol=PARETO_ENVELOPE_RTOL,
        frontier_atol=PARETO_ENVELOPE_ATOL,
    )

    if mismatch is None:
        outcome = PARETO_STATUS_STRICT_PASS
        return ParetoRecord(
            entry_key=entry.key,
            python_status="PASS",
            rust_status="PASS",
            strict_max_drift_pct=0.0,
            frontier_envelope_pct=None,
            comparison_outcome=outcome,
            error_msg=None,
            duration_ms=duration_ms,
            completed_at=completed_at,
        )

    # Strict check failed; see whether the relaxed envelope check would pass.
    # Disable the strict per-row comparison by setting its rtol very loose so
    # _compare_pareto_dfs only fails when the envelope-fallback path also
    # fails. This is a clean re-classification (STRICT_PASS / ENVELOPE_PASS
    # / DRIFT) without parsing the mismatch string.
    envelope_only = _compare_pareto_dfs(
        python_df,
        rust_df,
        rtol=1.0,
        atol=1.0,
        frontier_rtol=PARETO_ENVELOPE_RTOL,
        frontier_atol=PARETO_ENVELOPE_ATOL,
    )
    outcome = PARETO_STATUS_ENVELOPE_PASS if envelope_only is None else PARETO_STATUS_DRIFT

    return ParetoRecord(
        entry_key=entry.key,
        python_status="PASS",
        rust_status="PASS",
        strict_max_drift_pct=None,
        frontier_envelope_pct=None,
        comparison_outcome=outcome,
        error_msg=_shorten(mismatch),
        duration_ms=duration_ms,
        completed_at=completed_at,
    )


# ---------------------------------------------------------------------------
# Worker dispatch
# ---------------------------------------------------------------------------


def _worker_init() -> None:
    # Quiet noisy stdlib loggers in worker processes; keep WARN+.
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    # SIGINT in workers: let parent handle the shutdown gracefully via
    # ProcessPoolExecutor.shutdown(cancel_futures=True).
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _run_one_entry(
    entry_payload: tuple[Entry, str],
) -> tuple[Entry, ProbeRecord | None, ParetoRecord | None, str | None]:
    """Top-level worker function -- must be importable for pickling."""
    entry, scan_mode = entry_payload
    probe: ProbeRecord | None = None
    pareto: ParetoRecord | None = None
    err: str | None = None
    try:
        if scan_mode in (SCAN_MODE_PROBE, SCAN_MODE_BOTH):
            probe = probe_entry(entry)
        if scan_mode in (SCAN_MODE_PARETO, SCAN_MODE_BOTH):
            pareto = pareto_entry(entry)
    except Exception as exc:
        err = _format_exc(exc) + "\n" + _shorten(traceback.format_exc(), limit=600)
    return entry, probe, pareto, err


# ---------------------------------------------------------------------------
# Scan driver
# ---------------------------------------------------------------------------


def cmd_scan(args: argparse.Namespace) -> int:
    db_path = Path(args.db_path)
    baseline_dir = Path(args.baseline_dir)

    if not baseline_dir.exists():
        print(f"baseline-dir does not exist: {baseline_dir}", file=sys.stderr)
        return 2

    print(f"Scan mode: {args.scan_mode}")
    print(f"SQLite:   {db_path}")
    print(f"Workers:  {args.workers}")

    init_db(db_path)

    head_sha = _git_head_sha()
    prior_sha = read_run_meta(db_path, "commit_sha")
    if prior_sha and prior_sha != head_sha and not args.continue_across_commits:
        print(
            f"Refusing to mix results: scan.sqlite was written at commit {prior_sha},\n"
            f"current HEAD is {head_sha}.\n"
            f"Re-run with --continue-across-commits to acknowledge, or use --reset to start over.",
            file=sys.stderr,
        )
        return 3
    write_run_meta(
        db_path,
        {
            "commit_sha": head_sha,
            "scan_mode": args.scan_mode,
            "last_started_at": _now_iso(),
            "workers": str(args.workers),
        },
    )

    entries = load_entries(baseline_dir, status_filter=("PASS",))
    if not entries:
        print(f"No PASS entries found under {baseline_dir}", file=sys.stderr)
        return 4
    new_count = seed_entries(db_path, entries)
    print(f"Loaded {len(entries)} PASS entries ({new_count} newly seeded).")

    work = _select_work(db_path, entries, args.scan_mode, limit=args.limit)
    if not work:
        print("Nothing to do -- every entry already has a completed result.")
        return 0
    print(f"Pending: {len(work)} entries")

    return _run_pool(
        db_path=db_path,
        work=work,
        scan_mode=args.scan_mode,
        workers=args.workers,
        per_entry_timeout=args.per_entry_timeout_sec,
        status_interval=args.status_interval_sec,
        max_tasks_per_child=args.max_tasks_per_child,
    )


def _select_work(db_path: Path, entries: list[Entry], scan_mode: str, *, limit: int | None) -> list[Entry]:
    by_key = {e.key: e for e in entries}
    if scan_mode == SCAN_MODE_PROBE:
        pending = pending_entries_for_probe(db_path)
    elif scan_mode == SCAN_MODE_PARETO:
        pending = pending_entries_for_pareto(db_path)
    else:
        pending = pending_entries_for_probe(db_path) | pending_entries_for_pareto(db_path)
    work = [by_key[k] for k in sorted(pending) if k in by_key]
    if limit is not None:
        work = work[:limit]
    return work


def _run_pool(
    *,
    db_path: Path,
    work: list[Entry],
    scan_mode: str,
    workers: int,
    per_entry_timeout: int,
    status_interval: int,
    max_tasks_per_child: int | None = None,
) -> int:
    counters = {"PASS": 0, "DRIFT": 0, "REGRESSION": 0, "ERROR": 0, "TIMEOUT": 0, "OTHER": 0}
    completed = 0
    total = len(work)
    started_at = time.monotonic()
    last_status = started_at

    # WAL allows multiple readers but a single writer ought to own commits.
    conn = _connect(db_path)
    try:
        # max_tasks_per_child recycles each worker process after N entries so the
        # per-process SupportMatrix / perf-DB cache (see _worker_matrix) is freed
        # instead of growing unbounded -> avoids OOM on long full-matrix runs.
        # macOS/Windows default to the "spawn" start method, which this requires
        # (it is incompatible with "fork"); None keeps workers alive for the whole run.
        pool_kwargs: dict = {"max_workers": workers, "initializer": _worker_init}
        if max_tasks_per_child and max_tasks_per_child > 0:
            pool_kwargs["max_tasks_per_child"] = max_tasks_per_child
        with ProcessPoolExecutor(**pool_kwargs) as pool:
            future_to_entry: dict = {pool.submit(_run_one_entry, (entry, scan_mode)): entry for entry in work}
            try:
                while future_to_entry:
                    done_iter = as_completed(future_to_entry, timeout=status_interval)
                    progressed = False
                    try:
                        for future in done_iter:
                            entry = future_to_entry.pop(future)
                            progressed = True
                            try:
                                _, probe, pareto, err = future.result(timeout=per_entry_timeout)
                            except FutureTimeoutError:
                                counters["TIMEOUT"] += 1
                                _write_timeout(conn, entry, scan_mode, per_entry_timeout)
                                completed += 1
                                continue
                            except BrokenExecutor as exc:
                                print(f"Worker pool broken: {exc}", file=sys.stderr)
                                raise
                            if err:
                                counters["ERROR"] += 1
                                _write_top_level_error(conn, entry, scan_mode, err)
                            else:
                                if probe is not None:
                                    write_probe_record(conn, probe)
                                    counters[_bucket_probe(probe.status)] = (
                                        counters.get(_bucket_probe(probe.status), 0) + 1
                                    )
                                if pareto is not None:
                                    write_pareto_record(conn, pareto)
                                    counters[_bucket_pareto(pareto.comparison_outcome)] = (
                                        counters.get(_bucket_pareto(pareto.comparison_outcome), 0) + 1
                                    )
                            completed += 1
                            if time.monotonic() - last_status > status_interval:
                                _print_status(completed, total, counters, started_at)
                                last_status = time.monotonic()
                    except FutureTimeoutError:
                        pass

                    if not progressed and not future_to_entry:
                        break
                    if time.monotonic() - last_status > status_interval:
                        _print_status(completed, total, counters, started_at)
                        last_status = time.monotonic()
            except KeyboardInterrupt:
                print("\nSIGINT received; cancelling pending work and persisting completed results...", file=sys.stderr)
                pool.shutdown(wait=True, cancel_futures=True)
                _print_status(completed, total, counters, started_at)
                return 130
    finally:
        conn.close()

    _print_status(completed, total, counters, started_at, final=True)
    failures = counters.get("DRIFT", 0) + counters.get("REGRESSION", 0)
    return 1 if failures else 0


def _bucket_probe(status: str) -> str:
    if status in (PROBE_STATUS_PASS, PROBE_STATUS_BOTH_ERROR):
        return "PASS"
    if status == PROBE_STATUS_DRIFT:
        return "DRIFT"
    if status in (PROBE_STATUS_RUST_ERROR,):
        return "REGRESSION"
    if status == PROBE_STATUS_PY_ERROR:
        return "ERROR"
    return "OTHER"


def _bucket_pareto(outcome: str) -> str:
    if outcome in (PARETO_STATUS_STRICT_PASS, PARETO_STATUS_ENVELOPE_PASS):
        return "PASS"
    if outcome == PARETO_STATUS_DRIFT:
        return "DRIFT"
    if outcome == PARETO_STATUS_REGRESSION:
        return "REGRESSION"
    if outcome == PARETO_STATUS_ERROR:
        return "ERROR"
    return "OTHER"


def _write_timeout(conn: sqlite3.Connection, entry: Entry, scan_mode: str, per_entry_timeout: int) -> None:
    msg = f"per-entry timeout after {per_entry_timeout}s"
    completed_at = _now_iso()
    if scan_mode in (SCAN_MODE_PROBE, SCAN_MODE_BOTH):
        write_probe_record(
            conn,
            ProbeRecord(
                entry_key=entry.key,
                probe_shape="timeout",
                python_ttft_ms=None,
                python_tpot_ms=None,
                rust_ttft_ms=None,
                rust_tpot_ms=None,
                ttft_drift_pct=None,
                tpot_drift_pct=None,
                python_err=None,
                rust_err=None,
                status="TIMEOUT",
                duration_ms=per_entry_timeout * 1000.0,
                completed_at=completed_at,
            ),
        )
    if scan_mode in (SCAN_MODE_PARETO, SCAN_MODE_BOTH):
        write_pareto_record(
            conn,
            ParetoRecord(
                entry_key=entry.key,
                python_status=None,
                rust_status=None,
                strict_max_drift_pct=None,
                frontier_envelope_pct=None,
                comparison_outcome="TIMEOUT",
                error_msg=msg,
                duration_ms=per_entry_timeout * 1000.0,
                completed_at=completed_at,
            ),
        )


def _write_top_level_error(conn: sqlite3.Connection, entry: Entry, scan_mode: str, err: str) -> None:
    completed_at = _now_iso()
    if scan_mode in (SCAN_MODE_PROBE, SCAN_MODE_BOTH):
        write_probe_record(
            conn,
            ProbeRecord(
                entry_key=entry.key,
                probe_shape="worker_error",
                python_ttft_ms=None,
                python_tpot_ms=None,
                rust_ttft_ms=None,
                rust_tpot_ms=None,
                ttft_drift_pct=None,
                tpot_drift_pct=None,
                python_err=err,
                rust_err=err,
                status="WORKER_ERROR",
                duration_ms=0.0,
                completed_at=completed_at,
            ),
        )
    if scan_mode in (SCAN_MODE_PARETO, SCAN_MODE_BOTH):
        write_pareto_record(
            conn,
            ParetoRecord(
                entry_key=entry.key,
                python_status=None,
                rust_status=None,
                strict_max_drift_pct=None,
                frontier_envelope_pct=None,
                comparison_outcome="WORKER_ERROR",
                error_msg=err,
                duration_ms=0.0,
                completed_at=completed_at,
            ),
        )


def _print_status(
    completed: int, total: int, counters: dict[str, int], started_at: float, *, final: bool = False
) -> None:
    elapsed = time.monotonic() - started_at
    eta = (elapsed / completed) * (total - completed) if completed else 0
    pct = (completed / total * 100) if total else 0
    line = (
        f"[{int(elapsed):>6}s] {completed}/{total} ({pct:.1f}%) "
        f"PASS={counters['PASS']} DRIFT={counters['DRIFT']} "
        f"REGRESSION={counters['REGRESSION']} ERROR={counters['ERROR']} "
        f"TIMEOUT={counters['TIMEOUT']} ETA={int(eta)}s"
    )
    del final  # final flag is reserved for a future "summary at end" mode
    print(line, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Report subcommand
# ---------------------------------------------------------------------------


def cmd_report(args: argparse.Namespace) -> int:
    db_path = Path(args.db_path)
    if not db_path.exists():
        print(f"No SQLite at {db_path}", file=sys.stderr)
        return 2

    with closing(_connect(db_path)) as conn:
        total_entries = conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
        probe_total = conn.execute("SELECT COUNT(*) FROM probe_results").fetchone()[0]
        pareto_total = conn.execute("SELECT COUNT(*) FROM pareto_results").fetchone()[0]

        probe_by_status = dict(conn.execute("SELECT status, COUNT(*) FROM probe_results GROUP BY status"))
        pareto_by_outcome = dict(
            conn.execute("SELECT comparison_outcome, COUNT(*) FROM pareto_results GROUP BY comparison_outcome")
        )

        per_system_probe = list(
            conn.execute(
                """
                SELECT e.system,
                       COUNT(p.entry_key) AS probed,
                       SUM(CASE WHEN p.status IN ('PASS','BOTH_ERROR_PASS') THEN 1 ELSE 0 END) AS passed,
                       SUM(CASE WHEN p.status = 'DRIFT' THEN 1 ELSE 0 END) AS drifted,
                       SUM(CASE WHEN p.status = 'RUST_ERROR_ONLY' THEN 1 ELSE 0 END) AS regressed
                FROM entries e LEFT JOIN probe_results p ON e.entry_key = p.entry_key
                GROUP BY e.system
                ORDER BY e.system
                """
            )
        )

        top_drifts = list(
            conn.execute(
                """
                SELECT entry_key, ttft_drift_pct, tpot_drift_pct, status
                FROM probe_results
                WHERE status = 'DRIFT'
                ORDER BY ABS(COALESCE(tpot_drift_pct, 0)) DESC
                LIMIT ?
                """,
                (args.top,),
            )
        )

        regressions = list(
            conn.execute(
                """
                SELECT entry_key, status, COALESCE(rust_err, '') AS rust_err
                FROM probe_results
                WHERE status = 'RUST_ERROR_ONLY'
                ORDER BY entry_key
                LIMIT 100
                """
            )
        )

        pareto_drifts = list(
            conn.execute(
                """
                SELECT entry_key, comparison_outcome, COALESCE(error_msg, '') AS msg
                FROM pareto_results
                WHERE comparison_outcome IN ('DRIFT', 'REGRESSION', 'ERROR', 'WORKER_ERROR', 'TIMEOUT')
                ORDER BY entry_key
                LIMIT ?
                """,
                (args.top,),
            )
        )

    print(f"Total entries: {total_entries}")
    print(f"Probe results: {probe_total} -- {probe_by_status}")
    print(f"Pareto results: {pareto_total} -- {pareto_by_outcome}")
    print()

    if per_system_probe:
        print("Per-system probe summary:")
        print(f"  {'system':<24} {'probed':>7} {'pass':>7} {'drift':>7} {'reg':>7}")
        for row in per_system_probe:
            system, probed, passed, drifted, regressed = row
            print(f"  {system:<24} {probed or 0:>7} {passed or 0:>7} {drifted or 0:>7} {regressed or 0:>7}")
        print()

    if top_drifts:
        print(f"Top {len(top_drifts)} probe drifts:")
        for entry_key, ttft_drift, tpot_drift, status in top_drifts:
            print(f"  {entry_key}: ttft={ttft_drift or 0:+.3f}% tpot={tpot_drift or 0:+.3f}% [{status}]")
        print()

    if regressions:
        print(f"Probe regressions (Rust-only error, baseline PASS): {len(regressions)}")
        for entry_key, status, rust_err in regressions[:20]:
            print(f"  {entry_key}: {rust_err}")
        if len(regressions) > 20:
            print(f"  ... and {len(regressions) - 20} more")
        print()

    if pareto_drifts:
        print(f"Pareto drift/regression rows (top {len(pareto_drifts)}):")
        for entry_key, outcome, msg in pareto_drifts:
            print(f"  [{outcome}] {entry_key}: {msg}")
        print()

    if args.csv:
        _emit_csv(db_path, Path(args.csv))
        print(f"Wrote {args.csv}")

    return 0


def _emit_csv(db_path: Path, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(_connect(db_path)) as conn:
        rows = list(
            conn.execute(
                """
                SELECT
                    e.model, e.architecture, e.system, e.backend, e.version, e.mode,
                    e.baseline_status,
                    p.status        AS probe_status,
                    p.ttft_drift_pct,
                    p.tpot_drift_pct,
                    p.python_err    AS probe_python_err,
                    p.rust_err      AS probe_rust_err,
                    pr.comparison_outcome AS pareto_outcome,
                    pr.python_status AS pareto_python_status,
                    pr.rust_status   AS pareto_rust_status,
                    pr.error_msg     AS pareto_error_msg
                FROM entries e
                LEFT JOIN probe_results p ON e.entry_key = p.entry_key
                LEFT JOIN pareto_results pr ON e.entry_key = pr.entry_key
                ORDER BY e.system, e.backend, e.version, e.model, e.mode
                """
            )
        )
        columns = [d[0] for d in conn.execute("SELECT * FROM entries LIMIT 0").description]
        del columns
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "model",
                "architecture",
                "system",
                "backend",
                "version",
                "mode",
                "baseline_status",
                "probe_status",
                "ttft_drift_pct",
                "tpot_drift_pct",
                "probe_python_err",
                "probe_rust_err",
                "pareto_outcome",
                "pareto_python_status",
                "pareto_rust_status",
                "pareto_error_msg",
            ]
        )
        for row in rows:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


def cmd_reset(args: argparse.Namespace) -> int:
    db_path = Path(args.db_path)
    if not db_path.exists():
        print(f"No SQLite at {db_path}; nothing to reset.")
        return 0
    if not args.yes:
        print(
            f"Refusing to reset {db_path} without --yes. This will delete all probe + pareto results from this scan.",
            file=sys.stderr,
        )
        return 1
    with closing(_connect(db_path)) as conn:
        conn.execute("BEGIN")
        conn.execute("DELETE FROM probe_results")
        conn.execute("DELETE FROM pareto_results")
        conn.execute("DELETE FROM run_meta")
        conn.execute("COMMIT")
    print(f"Reset complete. {db_path} preserved with empty result tables.")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scan_rust_parity.py",
        description="Full Rust-vs-Python parity scan over the published support matrix.",
    )
    parser.add_argument(
        "--db-path",
        default=str(DEFAULT_DB_PATH),
        help=f"SQLite checkpoint file (default: {DEFAULT_DB_PATH}).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    scan = sub.add_parser("scan", help="Run the scan (resumable).")
    scan.add_argument("--baseline-dir", default=str(DEFAULT_BASELINE_DIR))
    scan.add_argument("--scan-mode", choices=ALL_SCAN_MODES, default=SCAN_MODE_PARETO)
    scan.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    scan.add_argument("--limit", type=int, default=None, help="Cap pending entries (smoke testing).")
    scan.add_argument("--per-entry-timeout-sec", type=int, default=DEFAULT_PER_ENTRY_TIMEOUT_SEC)
    scan.add_argument("--status-interval-sec", type=int, default=30)
    scan.add_argument(
        "--max-tasks-per-child",
        type=int,
        default=0,
        help=(
            "Recycle each worker process after this many entries to bound memory "
            "(the per-process perf-DB cache grows otherwise). 0 = never recycle "
            "(workers live for the whole run). STRONGLY PREFER 0: a finite value "
            "deterministically deadlocks this homogeneous workload (all workers hit "
            "the W*N recycle boundary at once -> ProcessPoolExecutor recycle hang; "
            "see runbook parity-scan-runbook.md §4.0). To bound memory, "
            "recycle at the process boundary with --limit shards instead, keeping "
            "--max-tasks-per-child 0. Requires the 'spawn' start method."
        ),
    )
    scan.add_argument(
        "--continue-across-commits",
        action="store_true",
        help="Acknowledge that the SQLite has results from a different HEAD sha.",
    )
    scan.set_defaults(func=cmd_scan)

    report = sub.add_parser("report", help="Print summary; optionally dump per-row CSV.")
    report.add_argument("--top", type=int, default=20, help="Top-N drift / regression rows to print.")
    report.add_argument("--csv", default=None, help="Write per-row CSV to this path.")
    report.set_defaults(func=cmd_report)

    reset = sub.add_parser("reset", help="Wipe all result rows (preserves entries).")
    reset.add_argument("--yes", action="store_true", help="Confirm the reset.")
    reset.set_defaults(func=cmd_reset)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
