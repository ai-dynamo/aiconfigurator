# GEMM quant transfer + validate-gate mirror (as built)

PR #1455 — prerequisite mechanism for #1392 (nvfp4_wo); adds **no new quant**.
Recipe for adding one: `how_to_add_a_gemm_quant_mode.md`.

## Goal

A quant with no collected GEMM data becomes estimable in HYBRID/EMPIRICAL by
borrowing a collected quant's util curve (as MoE already could), and the
`task_v2` validate gate admits exactly what the resolved transfer policy + DB
contents make reachable at query time — never more. Acceptance: one enum line
+ one util-level line per new quant.

## Semantics: one primitive, derived labels

`util_empirical.quant_transfer_grid` is the single borrow mechanism:

    latency(query) = SOL_query / (util_ref × e(profile_query)/e(profile_ref))

The correction ratio is identically 1 within a `(memory, compute)` profile,
so `xshape` / `xquant` / `xprofile` are **confidence labels derived from the
selected reference's relation to the query**, not three mechanisms:

| label | relation | correction | trust source |
|---|---|---|---|
| `xshape` | same quant, other slice | 1 | kernel identity |
| `xquant` | same profile | 1 (structural) | SOL-coefficient identity — no calibrated constant |
| `xprofile` | cross profile | level ratio | calibrated per-profile util LEVEL; last resort |

Reference preference is lexicographic (relation rank, profile distance,
slice-feature distance), implemented as the historical tier flow so MoE's
established selection semantics stay bit-identical (MoE was refactored onto
the primitive; pinned by its unit suite and Rust parity oracles). Policy
tokens (`TransferKind`) act as relation admission filters — `balanced` stops
at the zero-calibration relations.

## GEMM specifics

- **`xshape` is structurally empty**: the own-quant grid is already depth-3
  over every collected `(m, n, k)` (GEMM has no categorical slice axes), so
  there is no "other slice of the same quant" to borrow. Encoded as an empty
  candidate list, not an omitted code path.
- **`xquant`**: first same-profile sibling in file (first-seen) order — one
  whole-table candidate per sibling with constant features, so the pooled
  nearest-feature selection degrades to file order. Same SOL, no rescale.
- **`xprofile` ordering is compute-first**: `(|Δcompute|, |Δmemory|)`
  lexicographic (`xprofile_quant_order(prefer_same_compute=True)`). The
  compute factor (activation precision) determines the dense kernel's
  compute family — a weight-only quant runs bf16 MMA after fused dequant —
  so e.g. `(0.5625, 1)` deterministically borrows **bfloat16**; under plain
  L1 it ties bf16/fp8 at 1.4375 and the reference would be a file-order
  lottery. **MoE keeps its historical L1 metric**: its level ratios were
  validated under that ordering and existing selections must not reshuffle.
- `fp8_static` enters the ladder as its table quant `fp8` (existing
  normalize); the `compute_scale`/`scale_matrix` overhead tables get no
  ladder (only `fp8_static` consumes them, and it requires data anyway).
- Level table `_GEMM_QUANT_UTIL_LEVEL` (`operations/gemm.py`, Rust mirror in
  `operators/gemm.rs`): values, derivation method (compute-bound-region
  medians across b200/h200/h100 × trtllm/vllm/sglang) and LOO evidence
  (nvfp4←fp8 22–33% MAPE; bf16/fp8 ratio stable to ~±20%) are documented on
  the table itself — the code comment is the method-of-record.

## Validate gate (`task_v2._check_role_against_db`)

`gemm` (and `moe`/`wideep_*_moe`) `_check` with `profile_transfer=True`:

- `xquant_enabled` = non-SILICON mode ∧ `XQUANT` ∈ resolved policy ∧ a
  same-profile supported quant exists.
- `xprofile_enabled` = non-SILICON mode ∧ `XPROFILE` ∈ resolved policy ∧ the
  query profile is **listed** in the op's level table ∧ ≥1 supported quant
  of a different profile. The listed-profile condition is the one deliberate
  way the gate is stricter than the ladder (which has a default level): it
  enforces the enum-line + level-line recipe instead of silently admitting a
  quant nobody calibrated.
- **`fp8_static` is excluded from transfer admission**: it is a composite
  mode (fp8 base minus `compute_scale`/`scale_matrix`), and the overhead
  tables have no ladder — profile-reachability could admit it wherever plain
  fp8 data exists while `query_compute_scale` still dies at run time. Its
  admission stays purely data-driven (in `supported_quant_mode` iff all
  three tables exist).
- Dead `"nvfp4_wo": "bfloat16"` validation alias removed (no such enum
  member; normalize-to-bf16 was reviewed out of #1392 — it substitutes the
  reference's SOL for the query's, losing the w4 memory-side benefit). The
  real `w4a16_mxfp4_cutlass → w4a16_mxfp4` alias (a query-time normalize,
  moe.py) is kept. SILICON behavior unchanged.

## Rust parity notes

- `GemmGrids.by_quant` is a `BTreeMap` (alphabetical); ladder tie-breaks are
  pinned to Python's dict-insertion (file row) order, so the loader records
  `quant_order` and `GemmTable::available_quants()` exposes it — the
  analogue of `MoeTable::available_quants`.
- xquant takes the FIRST same-profile sibling and does not retry on an empty
  grid (parity with Python's single pooled selection); xprofile loops until
  a non-empty grid.
- Oracle tests in `operators/gemm.rs` pin Python values at 1e-9, including a
  case where compute-first and L1/file-order orderings disagree.

## Behavior change (intentional)

HYBRID/EMPIRICAL queries for existing quants with no data on a system change
from raising `EmpiricalNotImplementedError` to a policy-gated estimate under
the default policy (e.g. nvfp4 GEMM on Hopper). Previously-computable paths
are bit-unchanged; restrictive policies still reject. Parity-case fallout:
`minimax-m25-nvfp4-h200` HYBRID now computes on both sides — the old raise
came from the GEMM op (the old comment misattributed it to MoE) — so that
case became a value-parity `…-hybrid-xprofile` case and the error-symmetry
contract moved to `…-hybrid-balanced-miss` + the FFI typed-miss test, pinned
with `transfer_policy="balanced"` (no same-profile sibling for `(0.5625, 4)`
⇒ the miss is stable by construction).

## Out of scope

No new quant enum members; no `cli/api.py` resolve_* changes; no
`tools/support_matrix/` FP4-gate changes; no `collector/` or `generator/`
changes; WideEP alltoall dispatch unchanged. A possible follow-up (not this
PR): route the borrowed slice through the perf_interp v2 engine instead of
the deliberately simple `UtilGrid` 2-NN, unifying within-slice resolution —
changes every empirical estimate's numerics, so it needs its own LOO
evidence and golden re-pinning.
