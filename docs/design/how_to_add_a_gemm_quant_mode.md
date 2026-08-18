# How to add a GEMM or MoE quant mode (without collected data)

The handoff recipe for PR #1392's `nvfp4_wo` and any successor. Mechanism
rationale lives in `gemm_quant_transfer_design.md`; this page is only the
steps.

As of PR #1392, **MoE follows the same two-line recipe as GEMM**: add the
enum variant and one util-level row; the transfer ladder handles the rest.
No normalization functions, no validation aliases, no query-layer changes in
either operation family.

## The two lines (applies to both GEMM and MoE)

1. **Enum line** — `GEMMQuantMode` *and* `MoEQuantMode` in
   `aic-core/src/aiconfigurator_core/sdk/common.py`, with
   `QuantMapping(memory, compute, name, compute_dtype)`:
   - `memory` = weight bytes per element **including scales**
     (nvfp4-style 4-bit + 1 fp8 scale per 16 = `9/16`).
   - `compute` = the precision the MMA actually runs in (1 = bf16, 2 = fp8,
     4 = fp4). **A weight-only quant computes in bf16 ⇒ `compute=1`** —
     this field decides which data is borrowed (compute-first ordering ⇒
     weight-only borrows bfloat16), so get it right.

   Rust mirrors, same PR:
   - `GemmQuantMode` / `MoeQuantMode` variants in
     `aic-core/rust/aiconfigurator-core/src/common/enums.rs`
   - Name-parse arm in `gemm_quant_by_name`
     (`aic-core/rust/aiconfigurator-core/src/perf_database/gemm.rs`)
     (MoE uses `#[serde(rename_all = "snake_case")]` so no explicit parser needed)
   - Add the variant to `ALL_MOE_QUANTS` in
     `aic-core/rust/aiconfigurator-core/src/operators/moe.rs`

2. **Util-level line** — one row in each table if the `(memory, compute)`
   profile is not yet listed:
   - GEMM: `_GEMM_QUANT_UTIL_LEVEL`
     (`aic-core/src/aiconfigurator_core/sdk/operations/gemm.py`) + Rust mirror
     `GEMM_QUANT_UTIL_LEVEL`
     (`aic-core/rust/aiconfigurator-core/src/operators/gemm.rs`)
   - MoE: `_MOE_QUANT_UTIL_LEVEL`
     (`aic-core/src/aiconfigurator_core/sdk/operations/moe.py`) + Rust mirror
     `MOE_QUANT_UTIL_LEVEL`
     (`aic-core/rust/aiconfigurator-core/src/operators/moe.rs`)

   If unsure of the level, match the structurally nearest `[inferred]` row —
   only ratios are consumed. **The validate gate requires the profile to be
   listed in both tables**, so these lines are not optional for a new profile.

No table normalization, no `validation_aliases`, no query-layer changes.
HYBRID/EMPIRICAL then estimates via the transfer ladder (SOL stays your
quant's own — the memory-side benefit is preserved); SILICON still rejects
until real data exists.

## Verify

- GEMM template test:
  `tests/unit/sdk/database/test_gemm_quant_transfer.py::test_xprofile_weight_only_borrows_bf16_not_fp8`
  (`int4_wo` is the data-less stand-in; clone for your quant).
  `test_level_table_covers_every_gemm_quant_profile` will fail loudly if
  the GEMM level line is missing.
- MoE validate test: add a `test_validate_<quant>_ladder_admitted_in_hybrid`
  (see `test_validate_nvfp4_wo_ladder_admitted_in_hybrid` in
  `tests/unit/sdk/task_v2/test_task_config.py` as a template).
- Rust parity:
  - GEMM: extend `gemm_quant_transfer_ladder_matches_python_oracles`
    (`aic-core/rust/aiconfigurator-core/src/operators/gemm.rs`) with a
    Python-probed oracle.
  - MoE: add a `moe_<quant>_ladder_matches_python_oracle` test in
    `aic-core/rust/aiconfigurator-core/src/operators/moe.rs`
    (see `moe_nvfp4_wo_ladder_matches_python_oracle`).
- Before pushing: `cargo test --workspace --all-features` +
  `parity_tests/test_engine_step_parity.py`.

## Graduating to collected data

The ladder only fires behind an own-data miss: once `gemm_perf.parquet` or
`moe_perf.parquet` carries rows for your quant they win automatically — no
code change, delete nothing. Provenance moves `xprofile → silicon` by itself.

## Out of the mechanism's scope

Checkpoint-name → quant resolution (`cli/api.py` `resolve_*`), the
non-Blackwell remap (`resolve_nvfp4_for_system`), and the support-matrix
hardware gate are separate decisions with their own owners; this recipe only
makes the quant *estimable*.
