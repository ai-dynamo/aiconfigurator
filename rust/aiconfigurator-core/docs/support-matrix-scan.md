# Full Support Matrix Scan Plan

Goal: before shipping the Rust engine-step migration, exhaustively verify
that the Rust core matches Python across every `(model, system, backend,
version, mode)` tuple that the official support matrix currently
reports as `PASS`, with no regression in `PASS` status and engine-step
output drift within tolerance.

This scan is expected to run for hours-to-days. It MUST be
resumable.

## Scope

### Entries

- Source: the 10 per-system CSVs under
  `src/aiconfigurator/systems/support_matrix/{a100_sxm,b200_sxm,b300_sxm,
  b60,gb200,gb300,h100_sxm,h200_sxm,l40s,rtx_pro_6000_server}.csv`.
- Filter: `Status == PASS` rows only. Across all 10 systems that is
  **2,016 entries** (each row is one `(model, system, backend, version,
  mode)` tuple — `mode` is already split per row).
- Skip: `FAIL`, `HW_INCOMPATIBLE`, `FRAMEWORK_INCOMPATIBLE`. Those rows
  do not run today and cannot drift. The status-regression check below
  separately catches the case where a previously-`PASS` row becomes
  `FAIL` on the Rust path.

### What "matches" means per entry

Two layers, both required to pass:

1. **Engine-step probe** (fast, tight tolerance).
   For each entry, call `cli_estimate(engine_step_backend="python")`
   and `cli_estimate(engine_step_backend="rust")` on a fixed probe
   shape derived from the same test constraints the support-matrix
   generator already uses (`isl=256, osl=256, prefix=128`, parallelism
   from per-model size class). Compare `ttft` and `tpot`:
   - **rtol ≤ 1%, atol = 1e-3 ms.**
   - Symmetric error contract: both engines raising = pass; only one
     raising = fail.

2. **`cli_default` Pareto comparison** (slow, layered tolerance).
   For the same entry, run `cli_default(engine_step_backend="python")`
   and `cli_default(engine_step_backend="rust")`. Compare the
   resulting `pareto_df` via the existing `_compare_pareto_dfs`:
   - **Strict per-row Pareto metric: rtol ≤ 1%, atol = 1e-3.** This
     fires when Python and Rust pick the same frontier rows.
   - **Frontier envelope fallback: rtol ≤ 5%, atol = 1e-3.** This
     fires when row selection differs. Tightening it below 5% is not
     feasible — Pareto-row selection is discrete, so engine-step drift
     well below 1% can still shuffle which configuration each backend
     picks. The relaxed envelope check still bounds the user-visible
     end-to-end outcome.

The probe-layer <1% tolerance is the hard correctness signal; the
`cli_default` layer is the end-to-end UX signal.

### Status-regression check

Independent of drift: any entry that is `PASS` in the baseline CSV but
errors on Rust today is a regression. The runner records both:

- `python_status`: PASS / FAIL / ERROR (sanity check; must stay `PASS`
  for in-scope entries).
- `rust_status`: PASS / FAIL / ERROR.

A row with `python_status=PASS` and `rust_status!=PASS` is a regression
even if no drift comparison was possible.

## Storage / checkpointing

Single SQLite file: `rust/aiconfigurator-core/parity_tests/scan.sqlite`.

### Schema

```sql
CREATE TABLE entries (
    entry_key       TEXT PRIMARY KEY,           -- model|system|backend|version|mode
    model           TEXT NOT NULL,
    architecture    TEXT NOT NULL,
    system          TEXT NOT NULL,
    backend         TEXT NOT NULL,
    version         TEXT NOT NULL,
    mode            TEXT NOT NULL,              -- agg | disagg
    baseline_status TEXT NOT NULL               -- always PASS for in-scope
);

CREATE TABLE probe_results (
    entry_key       TEXT NOT NULL REFERENCES entries(entry_key),
    probe_shape     TEXT NOT NULL,              -- "isl=256,osl=256,prefix=128,bs=auto"
    python_ttft_ms  REAL,
    python_tpot_ms  REAL,
    rust_ttft_ms    REAL,
    rust_tpot_ms    REAL,
    ttft_drift_pct  REAL,                       -- (rust - python) / python * 100
    tpot_drift_pct  REAL,
    python_err      TEXT,                       -- non-null if Python raised
    rust_err        TEXT,                       -- non-null if Rust raised
    status          TEXT NOT NULL,              -- PASS | DRIFT | PY_ERROR_ONLY | RUST_ERROR_ONLY | BOTH_ERROR_PASS | ...
    duration_ms     REAL,
    completed_at    TEXT,                       -- ISO-8601
    PRIMARY KEY (entry_key, probe_shape)
);

CREATE TABLE pareto_results (
    entry_key                TEXT PRIMARY KEY REFERENCES entries(entry_key),
    python_status            TEXT,              -- PASS / FAIL / ERROR
    rust_status              TEXT,
    strict_max_drift_pct     REAL,              -- max rtol seen in strict per-row comparison
    frontier_envelope_pct    REAL,              -- rtol seen in envelope fallback (NULL if strict matched)
    comparison_outcome       TEXT NOT NULL,     -- STRICT_PASS | ENVELOPE_PASS | DRIFT | REGRESSION | SKIPPED
    error_msg                TEXT,
    duration_ms              REAL,
    completed_at             TEXT
);

CREATE TABLE run_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
-- e.g. ("commit_sha", "<HEAD>"), ("started_at", ...), ("python_version", ...),
-- ("aic_version", ...), ("scan_mode", "probe_only" | "pareto_only" | "both")
```

### Resume semantics

- On startup, the runner reads `entries` (populating it from the
  baseline CSVs on first run).
- For each entry, it skips `probe_results` rows that already exist
  with a non-`PENDING` status. Same for `pareto_results`.
- A SIGINT or crash mid-entry leaves the entry without a result row —
  the next run re-attempts it. (Worker processes write their result
  row atomically at the end of the entry; partial state is not
  persisted.)
- Resumption is idempotent across code changes only if `commit_sha` in
  `run_meta` matches `HEAD`. If it differs, the runner aborts with a
  message explaining that the user must either explicitly `--continue`
  (acknowledging cross-commit results), `--reset`, or use a new
  SQLite path.

### Concurrency

- N worker processes (default: `os.cpu_count() // 2` to leave headroom
  for the Python+Rust per-process load).
- A single coordinator process owns the SQLite connection in
  `journal_mode=WAL`, `synchronous=NORMAL`. Workers send their
  per-entry result over a `multiprocessing.Queue`; the coordinator
  batches commits every ~1 s or every 50 rows.
- Per-entry timeout: 15 minutes wall-clock (probe + `cli_default`).
  Entries that time out are recorded as `status=TIMEOUT` and re-tried
  on a subsequent run.

## Runner

New CLI: `tools/support_matrix/scan_rust_parity.py`.

```bash
uv run python tools/support_matrix/scan_rust_parity.py \
    --baseline-dir src/aiconfigurator/systems/support_matrix \
    --db-path     rust/aiconfigurator-core/parity_tests/scan.sqlite \
    --workers     16 \
    --scan-mode   both \
    --probe-rtol  0.01 \
    --probe-atol  0.001 \
    --pareto-strict-rtol 0.01 \
    --pareto-envelope-rtol 0.05 \
    --per-entry-timeout-sec 900
```

`--scan-mode {probe_only,pareto_only,both}` so a fast probe-only scan
can run first (filters obvious regressions in ~3–5 hours), and a
`--scan-mode pareto_only` follow-up can run unattended on entries
the probe accepted.

Status output: a periodic stderr line (every 30 s) of
`done / pending / errored / drifted` counts; full results in
SQLite.

## Wall-clock estimate

Assuming ~8 s per probe entry and ~60 s per `cli_default` entry, on
16 workers:

- Probe-only: 2,016 / 16 × 8 s ≈ **17 minutes**.
- `cli_default`-only: 2,016 / 16 × 60 s ≈ **2.1 hours**.
- Combined (`both`): ≈ **2.3 hours** if workers can interleave.

Add 20–40% headroom for `cli_default` outliers (large MoE sweeps,
disagg mode), tail timeouts, and SQLite write coalescing. Plan budget:
**4–8 hours wall-clock** on a 16-worker host. Days-long runs only
happen if `--workers 1` or a smaller host. The checkpoint design
covers both.

## Reporting

After the scan finishes (or at any point — the SQLite is queryable
live):

- `scan_rust_parity.py report --db-path scan.sqlite` prints:
  - Total entries / probed / pareto-checked / regressed / drifted.
  - Per-system summary (one row per system CSV).
  - Top-N drift entries (sorted by `tpot_drift_pct` desc).
  - Regression list (`python_status=PASS, rust_status!=PASS`).
- Also emits `scan_results.csv` mirroring the existing
  support-matrix CSV shape, plus drift columns, so the team can diff
  against the baseline `support_matrix/` CSVs row-for-row.

## Failure budgets

The scan is considered "ship-ready" when:

1. **Zero status regressions**: no row where `python_status=PASS,
   rust_status!=PASS`.
2. **Probe drift ≤ 1%** on every in-scope entry. Exceptions must be
   documented in this file with a root cause (data-grid edge,
   table-extrapolation, deliberate divergence, etc.).
3. **Pareto comparison** for every entry resolves as `STRICT_PASS` or
   `ENVELOPE_PASS`. `DRIFT` entries must each be triaged before ship.
4. **No `TIMEOUT` entries** persist after a final clean re-run.

Any `DRIFT` or `REGRESSION` rows must be triaged before merge, the
same way the existing parity smoke suite is triaged today.

## Implementation work-items

| # | Item | Notes |
| --- | --- | --- |
| W1 | New runner module `tools/support_matrix/scan_rust_parity.py` | CLI, SQLite IO, worker pool, signal handling. |
| W2 | Probe driver: per-entry `cli_estimate` Python+Rust + drift compute | Reuses `cli_estimate` already exposed via `aiconfigurator.cli.api`. |
| W3 | Pareto driver: reuse existing `SupportMatrix.run_single_test` with `compare_engine_step_backends=True` | Adjust tolerances to <1% strict / 5% envelope. |
| W4 | SQLite schema + migrations | Single file; one-time creation. |
| W5 | Resume + commit-sha guard | Refuse to mix results across commits unless `--continue`. |
| W6 | Reporting subcommand | `report`, `report --csv`, `report --drift-only`. |
| W7 | CI hook (optional) | `--scan-mode probe_only --workers 8 --top-n 50` smoke for PRs. Full scan stays manual. |

W1–W6 are required pre-ship. W7 is a nice-to-have follow-up.

## Stop conditions

Stop and re-plan if:

- The probe scan returns a regression class (>50 rows with the same
  failure mode). Likely a structural bug in the Rust crate — fix
  before continuing the scan.
- `cli_default` runtime per entry exceeds 5× the estimate above on
  the host of choice; revisit the parallelism strategy or shard by
  system.
- SQLite contention causes the runner to spend >10% of wall-clock on
  writes. Switch to per-system shard files and a post-run merge.
- Resume after a code change shows results that contradict the
  pre-change run on identical entries — surface the suspect rows and
  triage before continuing.
