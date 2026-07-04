<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Prediction Regression Gate — old-vs-new run_static sampling + scheduling defense

**Status:** implemented on `feat/regression-gate-v2` (SDK knob, tier-1/tier-2
collectors, comparison report, `workflow_dispatch` CI workflow). Rollout is
two PRs (§5): this PR ships the harness without touching PR CI; a follow-up
PR enables the `pull_request` trigger and demotes the old accuracy job.

## 1. Problem

The PR-time accuracy regression job (`accuracy-regression-testing.yml`) has
three structural problems:

1. **It never gates.** The threshold check is `continue-on-error: true` by
   design, and has to be: correcting one modeling error can move the
   *aggregate* MAPE away from silicon because previously-cancelling errors
   stop cancelling. A check that cannot block is a 31-minute FYI.
2. **Slow, and the cost is the workload, not the tooling.** Measured on a
   recent run (33 min total): checkout + installs ≈ 3.5 min (the old-revision
   venv itself is only ~1.4 min), then **~14.5 min per side** running 5,222
   silicon rows through the full `cli_estimate` pipeline, twice. The per-PR
   cost grows with every model/system added to the silicon sample.
3. **Coverage is sample-shaped, not support-shaped.** Coverage exists only
   where silicon was measured (6 systems, 15 models, versions frozen at
   measurement time). A modeling regression for `l40s`, `rtx_pro_6000_server`,
   `b60`, any NVFP4-only model, or any combo without silicon rows is invisible
   at PR time. "Can this combo run at all" (support regressions) is only
   sparsely covered.

Key observation: **PR regression testing and silicon accuracy tracking are
different jobs.** A PR gate answers "did this change alter predictions, and
where?" — that needs *self-comparison* over the support surface, not silicon
ground truth. Silicon MAPE ("are we accurate?") belongs in a scheduled job
whose output is a tracked metric, not a merge gate.

## 2. Design

Keep V1's **old-vs-new architecture** (collect on the PR base revision and on
the PR head, diff the two) and split the signal into three orthogonal layers:

```text
Layer 1  (accuracy)     silicon anchor points, predicted vs MEASURED
         axes: ~35 curated real e2e measurements (silicon_refs.csv)
         oracle: silicon ground truth; report-only at PR time, and the
                 nightly accuracy-tracking workload
         cost: ~2 min per side

Layer 2  (e2e self-comparison)   agg/disagg scheduling defense
         axes: 16 curated cli_estimate configs, one per scheduling feature
         oracle: the other revision; OK -> not-OK blocks
         cost: ~30 s per side

Layer 3  (wide & flat)  run_static prefill/decode point sampling
         axes: system x backend x version x model x parallelism x quant
         oracle: the other revision; OK -> not-OK blocks
         cost: ~6.6 ms dense / ~35 ms MoE-MLA per point (vs ~100 ms+ per
               cli_estimate row) -> the whole support surface fits in minutes
```

Layers 2 and 3 answer "did this PR change predictions" (self-comparison, no
silicon needed — that is what makes the full support surface affordable).
Layer 1 answers "how far from real hardware", per point: it reuses V1's
committed silicon sample but as a small curated anchor set, so a PR that moves
a point's accuracy shows the old -> new error movement in its report instead
of an opaque aggregate MAPE.

Both sides run **their own copy** of the harness (V1's pattern), so a PR that
changes the grid or the config list sees that change attributed to itself as
added/removed rows — no cross-revision API coupling.

### 2.1 Why old-vs-new instead of committed baselines

The first iteration of this design committed the tier-1 snapshot as per-combo
baseline CSVs (106k rows, ~7 MB) and diffed PR CI against them. It was
byte-deterministic and fast — and went stale in two days: two merges to main
(a new bundled model, a DeepSeek FMHA fix) invalidated the baselines of every
open PR, starting with this one (1,836-row drift on its own CI). Committed
baselines make every prediction-affecting merge a tax on all open PRs
(rebase + regenerate + review a four-digit-line CSV diff).

Old-vs-new has no state to go stale: the oracle is recomputed from the PR's
actual base every run. The price is collecting twice; the workload is cheap
enough that the two sides run as parallel CI jobs, so wall clock stays at
single-side cost (§4).

### 2.2 Tier 1 — static op-surface sampling

For every `(system, backend, version)` combo with real local data, run a fixed
point grid through `InferenceSession.run_static_latency_only` (no IFB, no rate
matching, no Pareto search) over every bundled model x parallelism layout x
quant variant (`tools/prediction_regression_gate/grid.py` is the single home of this
policy). Per point, record a three-state outcome:

| field | meaning |
|---|---|
| `status` | `OK` \| `DATA_MISS` (`PerfDataNotAvailableError`) \| `INVALID` (validation/other error, exception type recorded) |
| `value_ms` | run_static scalar result (6-decimal round) |

**Database discipline:** `database_mode=SILICON`, and **shared layer OFF**
(the SDK knob in §3). Every combo's result then depends only on (code, that
combo's own data files); sibling inheritance can't mask a per-version data
hole. `engine_step_backend` is pinned to `python` so the in-flight Rust
default flip is a deliberate, visible change.

### 2.3 Tier 2 — scheduling defense

Defends what tier 1 deliberately bypasses: IFB batching/token budgeting
(`run_agg`), disagg rate matching + worker balancing, speculative decoding
acceptance math, prefix caching, VL encoder path, memory/OOM checks.

16 curated configs (`tools/prediction_regression_gate/tier2_configs.yaml`), roughly one
per scheduling feature x {agg, disagg}, including an OOM-boundary case whose
*expected* outcome is an exception — the memory model is also code. Each
side snapshots `id,status,ttft_ms,tpot_ms` via `run_tier2.py`.

### 2.4 Gate policy

`report.py` diffs the two snapshots and classifies:

| category | meaning | gate |
|---|---|---|
| `REGRESSION` | OK -> DATA_MISS / INVALID (a working combo stopped working) | **blocking** |
| `DRIFT` | OK -> OK, relative change > rtol (default 1e-4) | report |
| `GAIN` | DATA_MISS / INVALID -> OK | report |
| `STATUS_CHANGE` | non-OK status or error type changed | report |
| `ROWS_ADDED/REMOVED` | grid, model list, or config list changed | report |

Only "was working, stopped working" blocks: it is objective, attributable,
and almost never intentional. Everything else lands in a categorized report
(CI step summary + `drift_report.csv` artifact) for the reviewer — an
intentional modeling change needs no acknowledgment ritual, because there is
no baseline to refresh; the report itself is the review artifact. This also
finally gives the "可以跑/不可以跑" regression a hard gate, which V1 never had.

### 2.5 The accuracy layer (run_silicon)

`make_silicon_refs.py` curates ~35 anchor points from
`src/aiconfigurator/systems/silicon_sample.csv` (real e2e measurements with
provenance; a 2026-04 dump of the internal benchmark DB): up to 2 configs per
(system, backend, mode) group, verified to predict OK at curation time.
`run_silicon.py` predicts each anchor (measured backend_version when local
data has it, else latest-data fallback — V1 semantics, skew recorded) and
snapshots predicted vs measured with per-metric relative error.

At PR time this renders as a report-only section: per-point error, and — with
both sides present — exactly which points' accuracy a PR moved and by how
much. Run standalone on a schedule, the same snapshot is the nightly accuracy
trend. Refreshing the anchors after a new silicon dump is: update
silicon_sample.csv, re-run make_silicon_refs.py, review the diff.

### 2.6 What stays out of the PR gate

- **Aggregate silicon MAPE over the full 5,222-row sample** stays in
  `accuracy-regression-testing.yml`, demoted to a scheduled/nightly tracked
  metric by the follow-up PR (it is already `continue-on-error`). Longer term
  the nightly can converge onto `run_silicon.py` with a larger anchor set and
  V1's two-venv machinery retires.
- The daily full support-matrix workflow is unchanged.

## 3. Required SDK change (small, additive)

Shared-layer control is currently derived from `database_mode` only
(`perf_database._shared_layer_enabled`); SILICON always inherits sibling rows.
Add an explicit override, default preserving today's behavior:

```python
get_database(...) / get_database_view(..., shared_layer: bool | None = None)
# None -> derive from database_mode (today's behavior)
# False -> SILICON with own-version data only  (tier-1 harness uses this)
```

`PerfDatabase.__init__` stores it; `_build_op_sources` already branches on
`self.enable_shared_layer`, so the knob is ~10 lines + tests
(`tests/unit/sdk/test_perf_database_shared_layer.py`).

## 4. Measured cost

Single-side collection, 28 combos / 106k rows:

| environment | tier-1 | tier-2 |
|---|---|---|
| Apple-silicon dev box, `--jobs 8` | ~200 s | ~28 s |
| CI `ubuntu-latest` (4 vCPU), `--jobs $(nproc)` | ~9 min | ~1 min |

Determinism: repeated runs byte-identical at 6-decimal rounding — including
across platforms (a macOS collection reproduced a CI Linux collection exactly,
0 differing rows in 106k).

CI shape: `old` and `new` collection run as parallel matrix jobs; the report
job is seconds. Wall clock ≈ setup (~4 min) + one collection (~10 min)
≈ **~13-14 min end to end**, at 2× compute — vs ~31 min for the old job.
Growth is linear and cheap: +1 model ≈ +0.12 s x #combos / workers per side.

## 5. Rollout: two PRs

1. **PR 1 (this PR, inert):** SDK knob, `tools/prediction_regression_gate/` (grid,
   collectors, compare, report), unit tests, and `prediction-regression-gate.yml` with
   `workflow_dispatch` only. Nothing in PR CI changes. After merge, validate
   by dispatching the workflow manually (inputs default to old=main,
   new=dispatched ref); the ref-resolution step already handles the
   `pull_request` event.
2. **PR 2 (trigger-only):** add the `pull_request` trigger to
   `prediction-regression-gate.yml`; in the same PR, demote `accuracy-regression-testing`
   to `schedule:` and remove the dead `accuracy-regression-comment.yml`.
   The PR gates itself end-to-end (the workflow runs on its own merge ref),
   and reverting it restores the status quo without touching the harness.

The old side of the very first comparisons (bases predating PR 1) has no
harness; the collect job uploads an empty snapshot and `report.py` degrades
to new-side statistics with exit 0.

## 6. Implementation map

| piece | file(s) |
|---|---|
| SDK shared-layer knob | `src/aiconfigurator/sdk/perf_database.py` + `tests/unit/sdk/test_perf_database_shared_layer.py` |
| Tier-1 grid policy | `tools/prediction_regression_gate/grid.py` — 9 shape points, 3 parallelism layouts per family, quant default+fp8 (+nvfp4 on Blackwell-family systems) |
| Tier-1 collector | `tools/prediction_regression_gate/collect_static.py` — multiprocess over combos, `--output-dir`/`--systems`/`--backends`/`--versions latest\|all` |
| Tier-2 collector | `tools/prediction_regression_gate/run_tier2.py` + `tier2_configs.yaml` (16 configs) |
| Accuracy layer | `tools/accuracy_tracking/make_silicon_refs.py` (curation), `silicon_refs.csv` (~35 anchors with provenance), `run_silicon.py` (predicted vs measured snapshot) |
| Comparison + report | `tools/prediction_regression_gate/compare.py` (pure CSV diff), `tools/prediction_regression_gate/report.py` (markdown summary incl. accuracy section, drift_report.csv, exit code) |
| Unit tests | `tests/unit/tools/test_prediction_regression_gate_compare.py`, `tests/unit/tools/test_prediction_regression_gate_report.py` |
| CI | `.github/workflows/prediction-regression-gate.yml` — refs -> collect (matrix old/new) -> report; `workflow_dispatch` only until PR 2 |

Local usage (compare your branch against main):

```bash
git worktree add /tmp/aic-old main
(cd /tmp/aic-old && python tools/prediction_regression_gate/collect_static.py --output-dir /tmp/gate-old --jobs 8 \
                 && python tools/prediction_regression_gate/run_tier2.py --output-dir /tmp/gate-old)
python tools/prediction_regression_gate/collect_static.py --output-dir /tmp/gate-new --jobs 8
python tools/prediction_regression_gate/run_tier2.py --output-dir /tmp/gate-new
python tools/prediction_regression_gate/report.py --old /tmp/gate-old --new /tmp/gate-new
```

## 7. Open questions / risks

- **2× compute per PR.** Accepted: it buys statelessness. If runner budget
  becomes a concern, the `versions latest` profile can be narrowed, or the
  old-side snapshot cached by base SHA (same artifact reused across pushes
  to an unchanged base).
- **Quant axis enumeration**: derive per-model quant variants from
  `supported_quant_mode` x model-compat rather than the current hand list;
  start with default+fp8(+nvfp4) and extend later.
- **Parallelism policy**: fixed per family in `grid.py`. A policy invalid for
  a future model shows as INVALID from day one (visible, not silent).
- **Drift is report-only.** A PR that silently degrades accuracy everywhere
  but keeps everything runnable will pass the gate; the report makes it
  loudly visible, and the nightly silicon-MAPE track catches the accuracy
  dimension. If review discipline proves insufficient, a drift-count
  threshold can be added to `BLOCKING_CATEGORIES` later.
