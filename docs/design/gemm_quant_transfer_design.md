# GEMM quant-transfer tiers + validate-gate cleanup — design

Status: IMPLEMENTED (see §11 for as-built resolutions of the open items and
confirmed-in-review semantic decisions; §3/§5/§6 below are the original
proposal and remain accurate except where §11 notes a delta)
Branch: `worktree-gemm-xprofile-borrow`
Relation: prerequisite mechanism for PR #1392 (nvfp4_wo); adds **no new quant**.
Companion recipe: `docs/design/how_to_add_a_gemm_quant_mode.md`.

## 1. Goal

Give GEMM the same empirical transfer ladder MoE already has, so a quant with
no collected GEMM data can be estimated in HYBRID/EMPIRICAL via util borrowing
instead of raising, and clean the `task_v2` validate gate so it mirrors —
never exceeds — the query ladder's admission policy.

Acceptance: adding a future data-less quant requires only (a) one
`GEMMQuantMode` enum line and (b) one `_GEMM_QUANT_UTIL_LEVEL` line, and
HYBRID estimate works end-to-end while SILICON still rejects early.

Out of scope (unchanged): `nvfp4_wo` or any new enum member, `cli/api.py`
resolve_* call sites, `tools/support_matrix/` FP4 gate, `collector/`,
`generator/`, WideEP alltoall dispatch path.

## 2. Current state and gap

`operations/moe.py::get_empirical` walks: own-shape grid → XSHAPE → XQUANT →
XPROFILE (util-level-ratio rescale), each gated by
`database.transfer_policy`.

`operations/gemm.py::_query_gemm_table::get_empirical` (gemm.py:446-465) only
builds the query quant's own depth-3 `(m, n, k)` util grid via
`util_empirical.grid_for`. A quant with zero rows in `gemm_perf.parquet`
yields `grid=None` → `EmpiricalNotImplementedError`, regardless of policy.
Rust `operators/gemm.rs::gemm_empirical` mirrors that exactly.

The validate gate (`task_v2.py:1268-1406`) checks GEMM with **no**
`profile_transfer`, and its `_profile_reachable` only models XQUANT
(same-profile). It also carries a dead string alias
`"nvfp4_wo": "bfloat16"` (task_v2.py:1372) pre-planted by #1395 — no such
enum member exists, and normalize-to-bf16 was the review-rejected approach.

## 3. GEMM ladder design — tier by tier

Ladder inserted in `get_empirical` after the own-grid miss
(`grid is None or not grid.samples`), mirroring moe.py's structure and
respecting `database.transfer_policy`:

| Tier | Decision | Reason |
|---|---|---|
| own-grid | keep as-is | unchanged fast path; provenance `empirical` |
| XSHAPE | **omitted** | GEMM's own grid is already depth-3 over *all* collected `(m, n, k)` of the query quant with per-axis clamping + 2-NN in normalized log space — any own-quant sample anywhere already feeds it. A separate "nearest collected shape within the query quant" tier would select from exactly the same sample set and can never fire when the own grid is empty. (MoE needs XSHAPE because its own grid is sliced per categorical shape `(topk, experts, hidden, inter, tp, ep)`; GEMM has no categorical shape axes.) |
| XQUANT | **added** | same `(memory, compute)` profile ⇒ GEMM SOL is *numerically identical* (profile fully determines `sol_math`/`sol_mem` for a given shape), so a sibling's util grid transfers with no rescale. Real coverage wins today: `sq`/`fp8_ootb`/`fp8_block` ↔ `fp8` (all `(1,2)`) on systems where only one was collected. Provenance `xquant`, `util_scale=1`. |
| XPROFILE | **added** (the core deliverable) | no same-profile data at all → borrow the nearest-profile collected quant's full `(m, n, k)` util grid, built with the **reference** quant's own SOL, rescaled by `e(query)/e(ref)` from `_GEMM_QUANT_UTIL_LEVEL`. Provenance `xprofile`. Last resort, lowest confidence. |
| XOP | **omitted** | no meaningful donor op for a dense GEMM; MoE's per-expert grouped GEMM has different launch/kernel structure. Nothing enables it; the policy token stays inert for GEMM. |

Mechanics shared with MoE (reuse, don't fork):

- Python: candidates built as `util_empirical.ReferenceCandidate`; grids via
  `util_empirical.grid_from_reference` (cache keys `("gemm_xquant", ...)` /
  `("gemm_xprofile", ...)` including quant, ref-quant and policy in
  `selection_key`). One candidate per sibling quant; its `node` is that
  quant's whole `m→n→k` table, `features` are the profile pair
  `(memory, compute)`, `sol_fn` bound per the ReferenceCandidate contract
  (query quant's SOL for XQUANT — numerically equal anyway; ref quant's SOL
  for XPROFILE).
- Candidate ordering: XQUANT candidates all tie on features →
  `_nearest_candidate`'s stable argmin picks the **first-seen table quant**
  (dict insertion = file row order), identical to MoE's tie-break convention.
  XPROFILE loops nearest-profile-first via a `_xprofile_gemm_quants` helper
  with the same `|Δmemory| + |Δcompute|` distance as `_xprofile_moe_quants`.
- The ladder operates on the **table quant** after
  `_normalize_gemm_quant_mode_for_table` (fp8_static→fp8), exactly like the
  own-grid path does today; `fp8_static` therefore inherits fp8's ladder
  for the base GEMM term.
- `compute_scale` / `scale_matrix` tables get **no** transfer ladder: they
  are only consumed by `fp8_static` (which has data wherever it's admitted;
  the SILICON early-reject for missing overhead tables at task_v2.py:1315
  stays). A future data-less quant never touches them.

## 4. `_GEMM_QUANT_UTIL_LEVEL` — seed method and values

Same contract as `_MOE_QUANT_UTIL_LEVEL` (moe.py:87): keyed by
`(memory, compute)` profile, single scalar per profile (the per-component
split was already validated as untrustworthy for MoE — same reasoning holds:
only the *ratio* is consumed), levels relative and tunable, `[data]` /
`[inferred]` annotations per row.

Derivation (offline, script committed under the PR as the method-of-record in
the table's comment): for every `(system, backend, version)` gemm table,
`util = SOL/measured` per row, level = median util over the clearly
compute-bound region (`m ≥ 1024, n ≥ 2048, k ≥ 2048`) — the region where the
systematic kernel-efficiency offset dominates and launch-overhead noise is
excluded. Cross-stack check on b200/h200/h100 × trtllm/vllm/sglang:

| profile | measured medians (min–max across 6 stacks) | seed |
|---|---|---|
| (2, 1) bf16 | 0.55 – 0.79 | 0.70 `[data]` |
| (1, 2) fp8-family | 0.28 – 0.55 | 0.45 `[data]` |
| (0.5625, 4) nvfp4 | 0.21 – 0.36 | 0.30 `[data]` |
| (0.5, 4) w4a4 | — | 0.30 `[inferred ≈ nvfp4]` |
| (1, 1) w8a16 | — | 0.55 `[inferred]` |
| (0.5, 1) w4a16 | — | 0.45 `[inferred: fused-dequant weight-only runs below the bf16 compute roofline it shares; Marlin-class]` |
| (0.5, 2) w4a8 | — | 0.35 `[inferred]` |
| default | — | 0.45 |

The bf16/fp8 ratio is 1.38–2.00 across the six stacks (±20% around ~1.6) —
looser than MoE's ~10% but stable enough for a last-resort tier. LOO on the
mechanism (predict a collected quant from its nearest-profile sibling at
shared shapes, `m ≥ 64`): the practically relevant direction
`nvfp4 ← fp8` gives 22–33% MAPE (vs 24–33% unscaled — levels near-tie there);
`fp8 ← bf16` gives ~50–60% scaled vs ~60–80% raw. Final seeds re-derived at
implementation time with this script and recorded in the comment, MoE-style.

## 5. Rust mirror

Same-PR, point-for-point parity:

- `operators/gemm.rs`: `gemm_empirical` gains the ladder
  (own grid → xquant → xprofile), reusing `util_empirical.rs`'s
  `UtilGrid.reference_provenance`, `nearest_candidate_index`, and
  `ProvenanceTier::XProfile` (already present). `GEMM_QUANT_UTIL_LEVEL`
  const + `gemm_quant_util_level()` mirroring the Python table verbatim.
- **Ordering gotcha**: `GemmGrids.by_quant` is a `BTreeMap` (alphabetical),
  but Python iterates the gemm dict in file-row (first-seen) order and the
  XQUANT tie-break depends on it. The loader must additionally record
  first-seen quant order (`quant_order: Vec<String>`), and a new
  `GemmTable::available_quants()` returns that order — the exact analogue of
  `MoeTable::available_quants` ("first-seen (file row) order").
- Grid cache keys: `gemm_xquant:{...}` / `gemm_xprofile:{...}` carrying
  quant, ref-quant, and `policy_fingerprint` (moe.rs:122 pattern) so
  differently-policied lookups cannot alias.
- Pre-push: `cargo test --workspace --all-features` (default args miss the
  public-api pins and pyo3-gated tests).

## 6. Validate-gate cleanup (`task_v2._check_role_against_db`)

Target invariant: **the gate mirrors the ladder — admits exactly what the
resolved policy + DB contents make reachable at query time, and never more.**

1. `_check("gemm", gemm_quant_mode)` → `profile_transfer=True` (GEMM now has
   the transfer machinery, so the relaxation MoE already gets applies).
2. Reachability extends beyond XQUANT:
   - `xquant_enabled` (existing, task_v2.py:1342): non-SILICON mode AND
     `TransferKind.XQUANT ∈ resolve_transfer_policy(...)` AND
     `_profile_reachable` (a same-profile supported quant exists).
   - **new** `xprofile_enabled`: non-SILICON mode AND
     `TransferKind.XPROFILE ∈ resolve_transfer_policy(...)` AND the query
     quant's `(memory, compute)` profile is listed in the op's util-level
     table AND the DB has ≥1 supported quant of a *different* profile
     (any collected quant is a viable nearest-profile reference).
   - The profile-listed condition is exposed as a small public helper per op
     (e.g. `gemm.xprofile_util_level_known(mode)` /
     `moe.xprofile_util_level_known(mode)`) so the gate doesn't reach into
     private tables. Note the one deliberate delta vs. runtime: the ladder
     falls back to a default level for unlisted profiles, but the gate
     requires a listed profile — this enforces the "enum line + level line"
     recipe instead of silently admitting a quant nobody calibrated.
     (Flagged here because the stated goal is "not stricter than the
     ladder"; this is the single, intentional exception.)
   - Applies to `gemm`, `moe`, and `wideep_*_moe` `_check` calls (the MoE
     runtime ladder already has XPROFILE; today's gate is stricter than the
     MoE ladder — that asymmetry is part of what this cleans).
3. `validation_aliases` (task_v2.py:1372): drop `"nvfp4_wo": "bfloat16"`
   (dead — no such enum member, and the normalize-to-bf16 design was
   rejected in #1392 review; profile-reachability is its replacement).
   **Keep** `"w4a16_mxfp4_cutlass": "w4a16_mxfp4"` — verified real: moe.py's
   query path normalizes `w4a16_mxfp4_cutlass` for table lookup
   (moe.py:2491), so the alias mirrors an actual normalize.
4. SILICON behavior unchanged: both flags require non-SILICON mode, so
   data-less quants still fail fast in SILICON.

## 7. Behavior changes (intentional, to be stated in the PR)

- HYBRID/EMPIRICAL estimates for *existing* quants on systems where they have
  no data change from "raise `EmpiricalNotImplementedError`" to "xprofile
  estimate" under the default (`aggressive`) policy — e.g. nvfp4 GEMM on
  Hopper. This is the mechanism working as intended; previously-computable
  values (silicon, own-grid empirical, covered hybrid) are bit-unchanged —
  the ladder only extends behind the existing miss.
- Parity ladder-miss case
  `minimax-m25-nvfp4-h200-vllm-019-hybrid-miss` +
  `test_hybrid_ladder_miss_raises_typed_empirical_miss`: after the GEMM
  ladder lands, the raise moves from the first NVFP4 GEMM op to the NVFP4
  MoE op (whose h200/vllm tables have no reference slice at that tp/ep, per
  the case's own comment) — error symmetry must be re-verified and the case
  comments updated, **not deleted** (main is the error-symmetric version).
  The SILICON typed-miss case (`test_silicon_data_gap_raises_typed_perf_data_miss`)
  is untouched by construction.
- Support-matrix HYBRID tier labels may improve for previously-failing
  configs (provenance `xprofile` now reachable for dense models). No
  support-matrix code changes.

## 8. Tests

1. **Mechanism acceptance (Python)**: fixture DB whose gemm table carries
   only `bfloat16`; query an existing dataless quant (`int4_wo`, profile
   `(0.5, 1)` — present in the enum, absent from every fixture) →
   HYBRID returns a finite latency with provenance `xprofile` and
   `util_scale = e(0.5,1)/e(2,1)`; SILICON raises
   `PerfDataNotAvailableError`; policy `balanced` (no XPROFILE) still raises
   `EmpiricalNotImplementedError`. This is the "synthetic new quant"
   equivalent — the mechanism keys only off profile + level line, so an
   existing enum member with no data exercises exactly the add-a-quant path.
2. **XQUANT tier (Python)**: fixture with `fp8` data only; query `sq` →
   provenance `xquant`, latency equals the fp8-grid reconstruction with the
   query's own SOL (no rescale).
3. **Gate tests**: gemm quant admitted under HYBRID+aggressive when only a
   different-profile quant is in `supported_quant_mode`; rejected under
   SILICON; rejected under `transfer_policy="balanced"`; rejected when the
   profile is not in the level table. Alias-removal regression: a fake
   `"nvfp4_wo"` string no longer validates.
4. **Rust parity**: unit oracles in `gemm.rs` tests pinning Python-probed
   values for one xquant and one xprofile query (same fixture pattern as
   `moe_xprofile_transfer_matches_python_oracle`, moe.rs:937), incl. the
   first-seen-order tie-break; engine-step parity gains a
   dense-model xprofile case; existing ladder-miss/error-symmetry cases kept
   (§7).
5. **Zero-change regression**: full unit suite + golden comparison; goldens
   must not move (all golden configs resolve at silicon/own-grid tiers).

## 9. Docs deliverable

`docs/design/how_to_add_a_gemm_quant_mode.md` (one page, the #1392 handoff):
enum line → level-table line (both languages) → what the gate now admits →
how to verify (the §8.1 test as template) → when to graduate from XPROFILE
borrow to collected data.

## 10. Open items to confirm before coding

1. Gate/ladder delta in §6.2 (gate requires a *listed* profile; ladder has a
   default): keep as specified (recommended — enforces the recipe), or relax
   the gate to mirror the default too?
2. Seed statistic: median over the compute-bound region (`m≥1024, n,k≥2048`)
   as above; alternative is per-stack tables (rejected: MoE precedent is one
   global table, and only ratios are consumed).
3. XQUANT first-seen-order tie-break (MoE convention, needs the Rust
   `quant_order` addition) vs. a fixed enum-order tie-break (no loader
   change, but diverges from MoE's established semantics). Design assumes
   the former.

## 11. As-built resolutions (post-review, implemented)

1. **§10.1–3 resolved as recommended**: gate requires a listed profile
   (recipe enforcement, the one intentional gate-stricter delta); global
   level table seeded from compute-bound medians; first-seen-order
   tie-break with the Rust `GemmGrids.quant_order` addition.
2. **Semantics unified — one primitive, derived labels (review decision).**
   `xshape`/`xquant`/`xprofile` are not three mechanisms but confidence
   labels over ONE borrow primitive
   (`util_empirical.quant_transfer_grid`): correction is
   `e(profile_q)/e(profile_ref)`, identically 1 within a profile. MoE was
   refactored onto the primitive with zero behavior change (pinned by the
   existing unit suite and Rust parity oracles); GEMM is the second
   consumer. GEMM's xshape relation class is structurally EMPTY (the
   own-quant depth-3 grid subsumes cross-shape), encoded as an empty
   candidate list, not an omitted code path.
3. **Compute-first XPROFILE ordering for GEMM** (review requirement: a
   weight-only quant must borrow float16/bfloat16 data, without any
   quant-specific code). GEMM orders cross-profile references
   lexicographically by `(|Δcompute|, |Δmemory|)`
   (`xprofile_quant_order(prefer_same_compute=True)`): the compute factor
   (activation precision) determines the dense kernel's compute family, so
   e.g. `(0.5625, 1)` deterministically borrows bfloat16 instead of
   tie-breaking into fp8 under plain L1 (both at distance 1.4375). MoE
   keeps its historical L1 metric — its level ratios were validated under
   that ordering and existing selections must not reshuffle.
4. **§7 parity-case outcome, stronger than predicted**: on
   h200/vllm/0.19.0 the MiniMax-M2.5-NVFP4 HYBRID case now COMPUTES on both
   sides under the default policy (the MoE ladder always had a reachable
   reference there; the old raise came from the GEMM op, and the old test
   comment misattributed it to MoE). The case was re-labelled
   `...-hybrid-xprofile` (value parity) and the error-symmetry contract
   moved to a new `...-hybrid-balanced-miss` case + the FFI typed-miss test,
   both pinned with `transfer_policy="balanced"` (no same-profile sibling
   for (0.5625, 4) ⇒ the miss is stable by construction).
5. **Level seeds shipped as proposed** (bf16 0.70, fp8-family 0.45, nvfp4
   0.30, inferred rows per §4), with method + LOO documented on the Python
   table and mirrored in the Rust const.
