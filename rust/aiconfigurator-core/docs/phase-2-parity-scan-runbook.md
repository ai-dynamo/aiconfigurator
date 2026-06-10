<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Phase 2 Parity Scan — Cloud Execution Runbook

**Audience:** an autonomous agent (or engineer) running the full Rust↔Python
parity scan on a **large-RAM cloud host**. This document is self-contained —
you do not need any prior conversation context.

## 1. Goal

Before deprecating the duplicated Python latency engine (Phase 2), we must
prove the **Rust** engine-step core matches the **Python** SDK across the
entire published support matrix. This scan runs both engines on every
`(model, system, backend, version, mode)` tuple the matrix reports as `PASS`
and records drift.

**Deliverable:** a completed `scan.sqlite` + a `report.csv`, with:
- `REGRESSION == 0` (no entry that is `PASS` in Python but errors in Rust),
- the `STRICT_PASS` count and the full `DRIFT` list (each triaged).

These feed a coverage/alignment showcase doc back in the main repo.

## 2. Host requirements (READ FIRST)

> **Do NOT run this on a <48GB machine.** Each worker process loads a full
> per-`(model,system,backend,version)` perf DB and caches it. A 36GB laptop
> swap-thrashes to death within minutes (observed: swap pinned at 18/18GB,
> workers hang). That is *why* this runbook exists.

| Resource | Minimum | Recommended |
|---|---|---|
| RAM | 48 GB | **64–128 GB** |
| vCPU | 8 | 16–32 |
| Disk | 30 GB (repo + perf DB via git-lfs + Rust target) | 50 GB |
| Network | git-lfs pull + first-time HF config fetches | — |

No GPU needed — this is a pure CPU perf-model scan.

## 3. Environment setup

```bash
# 3.1 Toolchains
#   - Python 3.10+ (3.12 recommended), uv, git-lfs, and a Rust toolchain.
curl -LsSf https://astral.sh/uv/install.sh | sh          # uv (if absent)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y  # cargo/rustc
source "$HOME/.cargo/env"
git lfs install

# 3.2 Clone + pin the commit you want to certify.
git clone https://github.com/ai-dynamo/aiconfigurator.git
cd aiconfigurator
# Pin a specific commit so results are reproducible & comparable. Record it.
# (Use the main HEAD at scan time, or the exact sha under test.)
git rev-parse HEAD

# 3.3 Perf databases (REQUIRED — the scan is meaningless without them).
git lfs pull

# 3.4 Install (maturin build-backend compiles the Rust core during install,
#     ~30–60s cold; needs cargo on PATH).
uv run pip install -e ".[dev]"

# 3.5 Verify both halves import.
uv run python -c "import aiconfigurator_core; import aiconfigurator; print('core+sdk OK')"
```

If `import aiconfigurator_core` fails, the Rust build did not run — confirm
`cargo --version` works and re-run step 3.4.

## 4. The scan — two phases

The runner is `tools/support_matrix/scan_rust_parity.py`. It is **resumable**:
results are checkpointed per-entry into the SQLite file, and re-running skips
completed rows. Run **probe-only first** (fast, catches regressions/large
drift), triage, then run the slow **pareto** phase.

### 4.0 Memory safety (the flag that prevents OOM)

`--max-tasks-per-child N` recycles each worker process after `N` entries, so
the per-process perf-DB cache is freed instead of growing unbounded. **Always
set it.** Tune workers to RAM (each worker ≈ a few GB once warm):

| Host RAM | Probe phase | Pareto phase (heavier) |
|---|---|---|
| 48 GB | `--workers 4 --max-tasks-per-child 25` | `--workers 3 --max-tasks-per-child 15` |
| 64 GB | `--workers 8 --max-tasks-per-child 25` | `--workers 6 --max-tasks-per-child 20` |
| 128 GB | `--workers 16 --max-tasks-per-child 50` | `--workers 12 --max-tasks-per-child 25` |

If you ever see workers hang with no progress in the status line, or
`BrokenExecutor` in the log: you are OOM/swapping — halve `--workers` and
lower `--max-tasks-per-child`, then re-run (it resumes).

### 4.1 Probe-only phase (fast)

```bash
DB=rust/aiconfigurator-core/parity_tests/scan.sqlite

uv run python tools/support_matrix/scan_rust_parity.py \
    --db-path "$DB" \
    scan --scan-mode probe_only \
    --workers 8 --max-tasks-per-child 25
```

- ~2,158 entries. On a 16-vCPU/64GB host expect well under an hour.
- Per-entry probe shape: `isl=256, osl=256, prefix=128`, parallelism by
  model size class. Compares `ttft`/`tpot`: **pass if rtol ≤ 1%** (atol 1e-3 ms).
- Tolerances are baked-in constants in the runner (not CLI flags).

### 4.2 Pareto phase (slow, end-to-end)

After probe triage, run the `cli_default` Pareto comparison on the **same DB**
(it fills the `pareto_results` table; probe rows are untouched):

```bash
uv run python tools/support_matrix/scan_rust_parity.py \
    --db-path "$DB" \
    scan --scan-mode pareto_only \
    --workers 6 --max-tasks-per-child 20
```

- Hours-scale. Per-entry timeout default 900s; timeouts are recorded and
  retried on re-run.
- Pareto verdicts: `STRICT_PASS` (per-row rtol ≤ 1%) / `ENVELOPE_PASS`
  (frontier rtol ≤ 5% when row-selection differs) / `DRIFT` / `REGRESSION`.

### 4.3 Resume / commit guard

- Re-running the same command resumes from the checkpoint automatically.
- The runner stores `commit_sha` in `run_meta` and **refuses to mix results
  across commits**. If you intentionally continue on a different commit, add
  `--continue-across-commits`. To start clean: `scan_rust_parity.py
  --db-path "$DB" reset --yes` (preserves the seeded entries, wipes results).

## 5. Monitoring

The runner prints a status line to stderr every 30s:
`[<elapsed>] done/total PASS=.. DRIFT=.. REGRESSION=.. ERROR=.. TIMEOUT=..`.

Query progress live (SQLite is WAL, safe to read mid-run):

```bash
sqlite3 "$DB" "SELECT COUNT(*) FROM probe_results;"
sqlite3 "$DB" "SELECT status, COUNT(*) FROM probe_results GROUP BY status;"
sqlite3 "$DB" "SELECT comparison_outcome, COUNT(*) FROM pareto_results GROUP BY comparison_outcome;"

# Healthy = the done count keeps climbing. If it stalls for minutes while the
# process is alive, you are swapping (see 4.0).
```

## 6. Triage

1. **Regressions (hard fail).** Any `python_status=PASS, rust_status!=PASS`.
   Must reach **0**. List them:
   ```bash
   sqlite3 "$DB" "SELECT e.model,e.system,e.backend,e.version,e.mode,p.error_msg
     FROM entries e JOIN pareto_results p USING(entry_key)
     WHERE p.comparison_outcome='REGRESSION';"
   ```
2. **Probe drift > 1%.** Inspect each:
   ```bash
   sqlite3 "$DB" "SELECT e.model,e.system,e.backend,e.mode,
     round(pr.ttft_drift_pct,2), round(pr.tpot_drift_pct,2)
     FROM entries e JOIN probe_results pr USING(entry_key)
     WHERE pr.status='DRIFT' ORDER BY abs(pr.tpot_drift_pct) DESC;"
   ```
   **Known watch item:** `deepseek-ai/DeepSeek-V4-Pro` (+ `-Flash`) on
   `b200_sxm/sglang` showed very large drift (-70% tpot) in a pre-fix probe.
   Confirm whether it is resolved on the scanned commit; if still large, flag
   it with the per-op breakdown rather than silently accepting.
3. **Pareto `DRIFT`.** Each `DRIFT` row needs a one-line root cause or an
   explicit "accepted, known" note (e.g. discrete frontier-knee disagreement,
   bs=1 endpoint noise). Historical clusters for reference: NCCL/OneCCL perf-DB
   path selection, scan-comparator bs=1 false positives, frontier-pick ties.

## 7. Completion criteria

The scan is ship-ready when:
1. `REGRESSION == 0`.
2. Every probe entry drift ≤ 1% (documented exceptions only).
3. Every pareto entry resolves `STRICT_PASS` or `ENVELOPE_PASS`; remaining
   `DRIFT` rows each triaged.
4. No `TIMEOUT` rows persist after a final clean re-run.

## 8. Deliverables (hand back)

```bash
# Summary + per-row CSV.
uv run python tools/support_matrix/scan_rust_parity.py --db-path "$DB" report --top 50
uv run python tools/support_matrix/scan_rust_parity.py --db-path "$DB" report --csv scan_results.csv
```

Hand back **either**:
- the `scan.sqlite` file itself (preferred — fully queryable), **or**
- `scan_results.csv` + the `report --top 50` text + the three GROUP BY counts
  from §5.

Also report: the exact `commit_sha` scanned, host RAM/vCPU, wall-clock, and the
worker/recycle settings used.

## 9. Gotchas checklist

- [ ] `git lfs pull` actually fetched the perf DBs (not LFS pointer stubs).
- [ ] `import aiconfigurator_core` succeeds (Rust core built).
- [ ] `--max-tasks-per-child` set on **both** phases (OOM guard).
- [ ] Workers sized to RAM; if status stalls → swapping → reduce and re-run.
- [ ] `commit_sha` recorded; don't mix commits without `--continue-across-commits`.
- [ ] `HF_HOME` set if the host needs a writable HF cache for config fetches.
- [ ] Don't run on the 36GB laptop — that's what sent this scan to the cloud.

## 10. Background context (optional reading)

- Scan design + schema + historical outcome (2026-06-01: 1906 STRICT_PASS,
  16 DRIFT, 0 REGRESSION over 2016 entries): `phase1/support-matrix-scan.md`.
- Why Phase 2 needs this (flip Rust to default, then delete Python latency
  path): `phase-2-python-dedup-plan.md`.
- Note: the matrix has since grown to ~2,158 entries (new DeepSeek-V4, Qwen3-VL,
  GLM-5, more backend versions), so a fresh full scan on current HEAD is
  required — the 2016-entry baseline is stale.
