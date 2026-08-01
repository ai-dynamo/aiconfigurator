# How to add a GEMM quant mode (without collected data)

The handoff recipe for PR #1392's `nvfp4_wo` and any successor. Mechanism
rationale lives in `gemm_quant_transfer_design.md`; this page is only the
steps.

## The two lines

1. **Enum line** — `GEMMQuantMode` in
   `aic-core/src/aiconfigurator_core/sdk/common.py`, with
   `QuantMapping(memory, compute, name)`:
   - `memory` = weight bytes per element **including scales**
     (nvfp4-style 4-bit + 1 fp8 scale per 16 = `9/16`).
   - `compute` = the precision the MMA actually runs in (1 = bf16, 2 = fp8,
     4 = fp4). **A weight-only quant computes in bf16 ⇒ `compute=1`** —
     this field decides which data is borrowed (compute-first ordering ⇒
     weight-only borrows bfloat16), so get it right.

   Rust mirror, same PR: the `GemmQuantMode` variant in
   `rust/aiconfigurator-core/src/common/enums.rs` and the name-parse arm in
   `gemm_quant_by_name` (`src/perf_database/gemm.rs`).

2. **Util-level line** — only if the `(memory, compute)` profile is not yet
   listed: one row in `_GEMM_QUANT_UTIL_LEVEL`
   (`operations/gemm.py`; derivation method on the table) and its Rust
   mirror `GEMM_QUANT_UTIL_LEVEL` (`operators/gemm.rs`). If unsure, match
   the structurally nearest `[inferred]` row — only ratios are consumed.
   **The validate gate requires the profile to be listed**, so this line is
   not optional for a new profile.

No table normalization, no `validation_aliases`, no query-layer changes.
HYBRID/EMPIRICAL then estimates via the transfer ladder (SOL stays your
quant's own — the w4 memory-side benefit is preserved); SILICON still
rejects until real data exists.

## Verify

- Template test:
  `tests/unit/sdk/database/test_gemm_quant_transfer.py::test_xprofile_weight_only_borrows_bf16_not_fp8`
  (`int4_wo` is the data-less stand-in; clone for your quant).
  `test_level_table_covers_every_gemm_quant_profile` will fail loudly if
  the level line is missing.
- Rust parity: extend `gemm_quant_transfer_ladder_matches_python_oracles`
  (`operators/gemm.rs`) with a Python-probed oracle (generation snippet in
  its doc comment).
- Before pushing: `cargo test --workspace --all-features` +
  `parity_tests/test_engine_step_parity.py`.

## Graduating to collected data

The ladder only fires behind an own-data miss: once `gemm_perf.parquet`
carries rows for your quant they win automatically — no code change, delete
nothing. Provenance moves `xprofile → silicon` by itself.

## Out of the mechanism's scope

Checkpoint-name → quant resolution (`cli/api.py` `resolve_*`) and the
support-matrix FP4 gate are separate decisions with their own owners; this
recipe only makes the quant *estimable*.
