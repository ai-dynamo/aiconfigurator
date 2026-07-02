<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# PR Regression Test V2 — run_static sampling + scheduling defense

**Status:** implemented on `worktree-regression-test-redesign` (SDK knob, tier-1
collector + gate, tier-2 configs + gate, baselines, CI workflows). See §5 for
the file map.
**Feasibility scripts:** `tools/accuracy_regression_testing/bench_run_static_feasibility.py`,
`tools/accuracy_regression_testing/bench_tier1_sweep.py` (research artifacts, disposable).

## 1. Problem

The PR-time accuracy regression job (`accuracy-regression-testing.yml`) has three
structural problems:

1. **Coupled signal.** Every sample runs the full agg/disagg pipeline
   (`cli_estimate`). A TTFT delta can come from an op-level modeling change, a
   scheduling change (IFB, rate matching), a data change, or shared-layer
   inheritance from a *different* backend version. One number, four causes —
   the gate is un-attributable, which is why it runs `continue-on-error` today.
2. **Slow and quadratic-ish.** 5,222 silicon rows × full pipeline × **two
   revisions** (old side includes a from-scratch venv + editable install)
   ≈ 30 min per PR today, and it grows with every model/system/version added
   to the sample. The e2e support-matrix smoke (`test_pr_support_matrix.py`)
   caps itself at 4 hand-picked cases for the same reason.
3. **Coverage is sample-shaped, not support-shaped.** Coverage exists only where
   silicon was measured (6 systems, 15 models, versions frozen at measurement
   time). A modeling regression for `l40s`, `rtx_pro_6000_server`, `b60`, any
   NVFP4-only model, or any combo without silicon rows is invisible at PR time.
   "Can this combo run at all" (support regressions) is only sparsely covered.

Key observation: **PR regression testing and silicon accuracy tracking are
different jobs.** A PR gate answers "did this change alter predictions, and
where?" — that needs *self-comparison* over the support surface, not silicon
ground truth. Silicon MAPE ("are we accurate?") belongs in a scheduled job
whose output is a tracked metric, not a merge gate.

## 2. Design

Two tiers, orthogonal axes:

```text
Tier 1  (wide & flat)   run_static prefill/decode point sampling
        axes: system x backend x version x model x quant     [the growing axes]
        oracle: committed baseline (golden) per combo
        gate:  status transitions (OK -> MISS/INVALID) + latency drift

Tier 2  (narrow & deep)  agg/disagg scheduling defense
        axes: ~15-25 curated e2e configs, one per scheduling feature
        oracle: committed golden TTFT/TPOT (+ concurrency curve points)
        gate:  drift beyond tolerance
```

### 2.1 Tier 1 — static op-surface sampling

For every `(model, system, backend, version[, quant])` in scope, run a **fixed
point grid** through `InferenceSession.run_static` (no IFB, no rate matching,
no Pareto search):

- prefill points: `static_ctx`, `bs=1`, `isl ∈ {1024, 8192}` (osl=8)
- decode points: `static_gen`, `(bs, isl) ∈ {(1,1024), (32,1024), (128,1024), (32,8192)}`, osl=256
- fixed parallelism policy per model family (e.g. dense `tp=4`; MoE `tp=4 = moe_tp1 × moe_ep4`;
  large-MLA family gets its known-good layout). The policy is part of the
  harness, versioned with the baseline. The point of fixing it: tier 1 must
  never exercise the *search* code path, only the *evaluation* path.

Per point, record a three-state outcome plus values:

| field | meaning |
|---|---|
| `status` | `OK` \| `DATA_MISS` (`PerfDataNotAvailableError`) \| `INVALID` (validation/other error, exception type recorded) |
| `ctx_latency_ms` / `step_latency_ms` | run_static scalar result (6-decimal round) |

**Database discipline:**

- `database_mode=SILICON`, and **shared layer OFF** (see §3 — needs a small SDK
  knob). Every combo's result then depends only on (code, that combo's own data
  files). A data drop for version N never shifts version M's baseline; sibling
  inheritance can't mask a per-version data hole.
- Version axis = versions that have real data dirs (marker-only dirs are
  excluded by definition once the shared layer is off — they have nothing to
  test).

**Oracle = committed baseline CSV** (one file per `system/backend/version`,
~58 models × 6-8 points ≈ a few hundred rows each, ~100 KB total class of
artifact). PR CI runs the **new revision only** and diffs:

- `OK -> DATA_MISS/INVALID`: **hard fail** (a supported combo stopped working —
  this is the "可以跑/不可以跑" regression).
- `DATA_MISS/INVALID -> OK`: coverage gained; non-fatal, reported, and the PR
  must refresh the baseline (enforced by the diff check itself).
- `OK -> OK` with `|Δ| > tolerance` (default: relative 1e-4 as "changed",
  per-partition report; blocking threshold configurable per partition, e.g.
  fail if any point moves > 2% without a baseline update in the same PR):
  intentional modeling changes ship the baseline diff **in the same PR** — the
  diff becomes the review artifact, replacing today's opaque aggregate-MAPE
  comment.

This makes the old-revision run (and its venv install) unnecessary: the
baseline *is* the old revision, reviewed at rest.

### 2.2 Tier 2 — scheduling defense

Defends what tier 1 deliberately bypasses: IFB batching/token budgeting
(`run_agg`), disagg rate matching + worker balancing (picking), speculative
decoding acceptance math, prefix caching, VL encoder path, memory/OOM checks.

Curated config list, roughly one per scheduling feature × {agg, disagg}:

- dense agg IFB at 3 concurrencies (curve shape, not one point)
- MoE agg with attention-DP
- disagg xPyD rate-matching case (prefill/decode worker asymmetry)
- MTP/nextn on a DeepSeek-family model; prefix>0 case; VL case; AFD case
- 1-2 OOM-boundary cases (expected INVALID — the memory model is also code)

~15-25 `cli_estimate` calls, golden-compared like tier 1. These do **not**
iterate the tier-1 axes: scheduling code is combo-generic by design, so one
representative model per relevant family suffices. Cost is seconds (warm
`cli_estimate` ≈ 0.1 s).

### 2.3 What stays out of the PR gate

- **Silicon accuracy (MAPE)** moves to the scheduled/nightly track — same
  `silicon_sample.csv`, same compare script, but as a *tracked metric* with
  trend reporting. It stops pretending to be a merge gate (it already is
  `continue-on-error`).
- The daily full support-matrix workflow is unchanged (it validates the
  shipped matrix CSVs; tier 1 is finer-grained and per-version but does not
  produce the user-facing matrix).

## 3. Required SDK change (small, additive)

Shared-layer control is currently derived from `database_mode` only
(`perf_database._shared_layer_enabled`); SILICON always inherits sibling rows.
Add an explicit override, default preserving today's behavior:

```python
get_database(...) / get_database_view(..., shared_layer: bool | None = None)
# None -> derive from database_mode (today's behavior)
# False -> SILICON with own-version data only  (tier-1 harness uses this)
```

Plumbing: `PerfDatabase.__init__` stores it; `_build_op_sources` already
branches on `self.enable_shared_layer` (perf_database.py:1619), so the knob is
~10 lines + tests. The feasibility sweep simulated it by patching
`_shared_layer_enabled` and behaves as expected (§4).

## 4. Measured feasibility (Apple-silicon dev box, single process)

From `bench_run_static_feasibility.py` / `bench_tier1_sweep.py`:

| cost | measured |
|---|---|
| `import aiconfigurator` | ~1.3 s |
| DB template + first query warmup per (system, backend, version) | ~2-4 s (parquet lazy-load) |
| model+session build | ~0.1 ms |
| marginal `run_static` point (session reuse) | ~6.6 ms dense / ~35 ms MoE-MLA |
| **58 bundled models × 6 points, one combo** | **7.0 s** (shared on), 5.9 s (shared off) |
| warm full `cli_estimate` agg (today's per-row) | ~100 ms + cold DB amortization |

Determinism: repeated runs across processes byte-identical at 6-decimal
rounding.

Shared-layer-off sanity: models whose active-version data is inherited flip to
`DATA_MISS` (e.g. GLM-5 on h200/trtllm/1.3.0rc10: all-ok → 6/6 miss;
DeepSeek-R1: 2/6 ctx-MLA misses) — exactly the per-version signal the tier-1
baseline should pin, and invisible today.

**Scope arithmetic** (current repo data): 28 engine (system, backend) combos,
~50 version dirs with real data, 58 bundled models (offline-buildable),
6-8 points each:

```text
tier 1 full sweep ≈ 50 combos × (~4 s warmup + 58 × ~0.12 s) ≈ 9 min single-process
                  ≈ ~70-120 s with ProcessPoolExecutor(8) (workers partition combos,
                    one DB load each — same pattern as tools/support_matrix)
tier 2            ≈ ~20 × ~1 s ≈ 30 s
CI job wall clock ≈ setup (checkout+LFS+pip ~2-3 min, unchanged) + ~2-3 min test
                  ≈ ~5 min total vs ~30 min today; no old-revision venv at all
```

Growth model: +1 model ≈ +0.12 s × #combos ÷ workers (~1 s); +1 version dir
≈ +10 s ÷ workers. Linear, with per-unit cost ~100× cheaper than today's
per-sample cost — the axis explosion the current design can't absorb is
absorbable here. A quant axis (×2-3 on applicable models) stays within budget.

## 5. Implementation map (all landed on this branch)

| piece | file(s) |
|---|---|
| SDK shared-layer knob | `sdk/perf_database.py` (`shared_layer: bool \| None` on `PerfDatabase` / `get_database` / `get_database_view`; overridden templates cached separately) + 4 tests in `tests/unit/sdk/test_perf_database_shared_layer.py` |
| Tier-1 grid policy | `tools/regression_v2/grid.py` — 9 shape points (4 prefill + 5 decode, isl up to 32k), 3 parallelism layouts per family (dense tp1/4/8; MoE tp4·ep4 / tp8·ep8 / adp8-wide-EP), quant default+fp8 (+nvfp4 on Blackwell-family systems) |
| Tier-1 collector | `tools/regression_v2/collect_static_baseline.py` — multiprocessing over combos, `--update` / `--output-dir` / `--versions latest\|all`, `run_static_latency_only`, `engine_step_backend` pinned to `python` |
| Tier-1 gate | `tools/regression_v2/compare.py` (pure CSV diff, 8 unit tests in `tests/unit/tools/test_regression_v2_compare.py`) + `tests/regression_v2/test_static_regression.py` (per-combo parametrized, categorized failures, report artifact via `AIC_REGV2_REPORT`) |
| Tier-2 configs + gate | `tools/regression_v2/tier2_configs.yaml` (17 configs: IFB concurrency curve, long-isl budget, MoE adp wide-EP, disagg rate-matching dense+MoE+sglang, MTP nextn1/2, prefix, VL encoder, vllm/sglang agg, OOM boundary) + `tests/regression_v2/test_scheduling_regression.py` (`AIC_REGV2_UPDATE_GOLDEN=1` refresh) |
| Baselines | `tests/baselines/regression_v2/<system>/<backend>/<version>.csv` (28 combos, 106,488 rows, ~7 MB) + `tier2_golden.csv` (17 rows) |
| CI | `.github/workflows/regression-v2.yml` (PR gate: collect + compare + artifacts + step summary); `accuracy-regression-testing.yml` demoted to nightly `schedule:` vs latest release tag; dead `accuracy-regression-comment.yml` removed |
| pytest | `regression_v2` marker registered in `pytest.ini`; excluded from `-m "unit or build"` CI |

Measured on this branch: full tier-1 collection 28 combos / 106,488 rows /
**~68 s** at `--jobs 8` (statuses: 54,683 OK / 51,795 DATA_MISS / 10 INVALID);
tier-1 compare < 1 s; tier-2 live run ~25 s. Baselines byte-identical across
repeated runs and across process topologies (pool vs single process).

Rollout: land with the job **advisory** for one week of PRs (noise expected to
be zero given determinism), then flip to blocking. The old job's demotion to
nightly ships in the same change.

## 6. Open questions / risks

- **Quant axis enumeration**: derive per-model quant variants from
  `supported_quant_mode` × model-compat (`_apply_model_quant_defaults`) rather
  than hand lists — needs a small resolver; start with per-model default quant
  (matches bundled `*_config.json` set, incl. FP8/NVFP4 variants) and add the
  cross product later.
- **Parallelism policy**: fixed per family in the harness. Risk: a policy
  invalid for a future model shows as INVALID from day one (visible, not
  silent) — acceptable; policy lives next to the baseline so changing it
  regenerates baselines explicitly.
- **Platform float drift** (macOS dev vs Linux CI): baselines are generated in
  CI; comparisons use rounded values + relative tolerance, so cross-platform
  regeneration is safe but should stay CI-canonical.
- **Rust engine-step default flip** (phase-2 plan in flight): pin
  `engine_step_backend` explicitly in the harness so the flip is one deliberate
  baseline regeneration, not ambient drift.
- **Baseline churn on data drops**: intended — a data PR's baseline diff shows
  exactly which combos' predictions moved, which is information today's
  pipeline hides behind shared-layer inheritance.
