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

### 4.0 Memory safety: recycle at the PROCESS boundary, not in the pool

Long-lived workers leak RSS even though the perf DB is tiny (~76 MB total):
the pareto `cli_default` sweep produces large pandas/numpy intermediates, and
multi-threaded BLAS/rayon code inflates **glibc malloc arenas** that are never
returned to the OS while the process lives. On a long pareto run this climbs
until it can OOM or even shut the host down. So you DO need to recycle — but
do it the right way.

> **WARNING — do NOT use a finite `--max-tasks-per-child`.** This workload is
> homogeneous, so all `W` workers reach the `W × N` task boundary at the same
> instant and try to recycle simultaneously, triggering a CPython
> `ProcessPoolExecutor` recycle **deadlock** (main thread parks in
> `futex_do_wait`, all workers gone, frozen indefinitely). It is deterministic
> and memory-independent (observed at 5 GB used / 0 swap / load 0.07). Lowering
> `--workers` does NOT fix it — it just moves the boundary. **Always pass
> `--max-tasks-per-child 0`** (the documented "never recycle in-pool" mode).

Recycle at the **process boundary** instead: run the scan in `--limit N`
shards. Each shard runs a fresh process that exits after `N` entries, returning
**all** memory to the OS — worker arenas, main-process accumulation, everything
— which is more thorough than in-pool recycling and cannot deadlock. The DB is
checkpointed, so the next shard resumes from where the last stopped.

Also cap library threads — 16 workers already saturate the cores, so per-worker
BLAS/rayon threads are pure oversubscription and the main arena-bloat driver:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       RAYON_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
```

Tune workers + shard size to RAM (always with `--max-tasks-per-child 0`):

| Host RAM | Probe (one shot) | Pareto (`--limit` shard loop) |
|---|---|---|
| 48 GB | `--workers 4` | `--workers 4 --limit 100` |
| 64 GB | `--workers 8` | `--workers 8 --limit 150` |
| 128 GB | `--workers 16` | `--workers 12 --limit 200` |

If a shard's RSS still climbs toward the limit, lower `--limit` (recycle more
often) and/or `--workers`. See §5 for how to tell a deadlock-hang from real
memory pressure.

### 4.1 Probe-only phase (fast)

Probe is light (single-point `cli_estimate`), so it runs in one shot — no
shard loop needed. Set the thread caps from §4.0 first.

```bash
DB=aic-core/rust/aiconfigurator-core/parity_tests/scan.sqlite
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       RAYON_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

uv run python tools/support_matrix/scan_rust_parity.py \
    --db-path "$DB" \
    scan --scan-mode probe_only \
    --workers 16 --max-tasks-per-child 0
```

- ~2,158 entries. On a 16-vCPU/64GB host expect well under an hour.
- Per-entry probe shape: `isl=256, osl=256, prefix=128`, parallelism by
  model size class. Compares `ttft`/`tpot`: **pass if rtol ≤ 1%** (atol 1e-3 ms).
- Tolerances are baked-in constants in the runner (not CLI flags).

### 4.2 Pareto phase (slow, end-to-end)

After probe triage, run the `cli_default` Pareto comparison on the **same DB**
(it fills the `pareto_results` table; probe rows are untouched). This is the
heavy phase — run it as a `--limit` shard loop so each process exits and frees
memory between shards (§4.0). `--limit` caps *pending* entries, so each shard
picks up where the last stopped:

```bash
# Thread caps from §4.0 must already be exported.
prev=-1
while :; do
  uv run python tools/support_matrix/scan_rust_parity.py \
      --db-path "$DB" \
      scan --scan-mode pareto_only \
      --workers 12 --max-tasks-per-child 0 --limit 200
  cur=$(sqlite3 "$DB" "SELECT COUNT(*) FROM pareto_results;")
  echo "pareto rows so far: $cur"
  # Stop when a shard adds no new rows (no pending left, or only runner-skipped
  # entries remain). Robust — does not depend on the runner's internal pending
  # definition.
  [ "$cur" -eq "$prev" ] && { echo "no new progress -> done"; break; }
  prev=$cur
done
```

- Hours-scale. Per-entry timeout default 900s; timeouts are recorded and
  retried on re-run. ~2,158 / 200 ≈ 11 shards; the ~15s per-shard startup
  (import + Rust core load) is negligible.
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

# Healthy = the done count keeps climbing.
```

Watch total worker RSS to catch real memory pressure before it shuts the host
down:

```bash
watch -n30 'ps -o rss= -p $(pgrep -f scan_rust_parity | tr "\n" "," | sed "s/,$//") | awk "{s+=\$1} END {print s/1048576 \" GB RSS\"}"'
```

**Stall vs. memory pressure** — two different failures, do not confuse them:
- **Deadlock** (done count frozen, process alive, **low** RSS / no swap): you
  used a finite `--max-tasks-per-child` and hit the recycle deadlock (§4.0).
  Fix: `--max-tasks-per-child 0` + the shard loop. Re-running resumes.
- **Memory pressure** (RSS climbing toward host RAM, swap growing): lower
  `--limit` and/or `--workers`, then resume.

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
- [ ] `--max-tasks-per-child 0` on both phases (a finite value DEADLOCKS — §4.0).
- [ ] Library thread caps exported (`OMP_NUM_THREADS=1` etc.) before each phase.
- [ ] Pareto run as a `--limit` shard loop (process-boundary recycle, §4.2).
- [ ] Workers/`--limit` sized to RAM; distinguish deadlock-hang from swap (§5).
- [ ] `commit_sha` recorded; don't mix commits without `--continue-across-commits`.
- [ ] `HF_HOME` set if the host needs a writable HF cache for config fetches.
- [ ] Don't run on the 36GB laptop — that's what sent this scan to the cloud.

## 10. Background context (optional reading)

- Completed scan result — the deliverable this runbook produces
  (last full run 2026-06-16, commit `048c3a7f`: gate CLOSED, 0 REGRESSION,
  DRIFT list triaged over ~2,158 entries):
  `parity-scan-report.md`.
- Why Phase 2 needs this (flip Rust to default, then delete the Python latency
  path): `python-dedup-plan.md`.
- Re-run this runbook on the current HEAD whenever the support matrix grows or
  the Rust hot path changes; the runner refuses to mix results across commits
  (§4.3), so record the `commit_sha` under test.
