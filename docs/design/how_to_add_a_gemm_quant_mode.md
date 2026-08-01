# How to add a GEMM quant mode (without collected data)

Audience: anyone adding a quant mode the perf DB has no data for — the
handoff recipe for PR #1392's `nvfp4_wo` and any successor. The transfer
mechanism (this repo's "quant-transfer primitive") makes a data-less quant
estimable in HYBRID/EMPIRICAL by borrowing a collected quant's util curve;
you only declare *what the quant is*, never *whose table it should read*.

## The two lines

1. **Enum line** — `GEMMQuantMode` in
   `aic-core/src/aiconfigurator_core/sdk/common.py`, with the correct
   `QuantMapping(memory, compute, name)`:
   - `memory` = weight bytes per element **including scales** (nvfp4-style
     4-bit + 1 fp8 scale per 16 = `9/16`).
   - `compute` = the precision the MMA actually runs in: 1 = bf16, 2 = fp8,
     4 = fp4. **A weight-only quant computes in bf16 ⇒ `compute=1`**, no
     matter how the weights are stored. This field decides which data the
     mechanism borrows (see below) — get it right.

   Rust mirror (same PR): the `GemmQuantMode` variant + mapping in
   `aic-core/rust/aiconfigurator-core/src/common/enums.rs`, and the
   name-parse arm in `gemm_quant_by_name`
   (`src/perf_database/gemm.rs`).

2. **Util-level line** — only if the `(memory, compute)` profile is not yet
   listed: one row in `_GEMM_QUANT_UTIL_LEVEL`
   (`aic-core/src/aiconfigurator_core/sdk/operations/gemm.py`) and its Rust
   mirror `GEMM_QUANT_UTIL_LEVEL` (`src/operators/gemm.rs`). The value is
   the profile's typical achieved util (SOL/measured) in the compute-bound
   region; derivation method and LOO evidence are documented on the Python
   table. If unsure, match the structurally nearest `[inferred]` row — only
   ratios are consumed. **The validate gate requires the profile to be
   listed** (it deliberately refuses the runtime default fallback), so this
   line is not optional for a new profile.

That's it. No table normalization, no `validation_aliases`, no query-layer
changes.

## What you get, and from where

- **HYBRID / EMPIRICAL**: `latency = SOL(your quant) / (util_ref ×
  e(your profile)/e(ref profile))`, provenance `xprofile` (or `xquant`
  unscaled, if a same-profile quant is collected). SOL uses YOUR profile —
  a weight-only quant keeps its w4 HBM benefit on the memory-bound end.
  This is why normalizing to another quant's table (the approach reviewed
  out of #1392) is wrong: it substitutes the reference's SOL for yours.
- **Reference selection is compute-first**: nearest profile by
  `(|Δcompute|, |Δmemory|)`. A weight-only quant (`compute=1`) therefore
  borrows **bfloat16** data — never tie-breaks into fp8 — because it runs
  the bf16 MMA kernel family. `nvfp4_wo (0.5625, 1)` on Hopper borrows
  bf16's util curve rescaled by `e(0.5-ish,1)/e(2,1)`.
- **SILICON**: still rejects early (`_validate_database_quant_modes`) — no
  data is no data; the gate admits the quant only in non-SILICON modes with
  `TransferKind.XPROFILE` in the resolved transfer policy.

## Verify

- Template test:
  `tests/unit/sdk/database/test_gemm_quant_transfer.py::test_xprofile_weight_only_borrows_bf16_not_fp8`
  (uses `int4_wo` as the data-less stand-in; clone it for your quant, or
  just extend `test_level_table_covers_every_gemm_quant_profile`).
- Rust parity: extend
  `gemm_quant_transfer_ladder_matches_python_oracles`
  (`src/operators/gemm.rs`) with a Python-probed oracle for your quant
  (generation snippet in the test doc comment).
- Before pushing: `cargo test --workspace --all-features` (default args
  miss the public-api pins and pyo3-gated tests) + the parity suite
  `parity_tests/test_engine_step_parity.py`.

## Graduating to collected data

The ladder only fires behind an own-data miss: the day
`gemm_perf.parquet` carries rows for your quant, they win automatically —
no code change, delete nothing. Collect on the systems that matter and the
provenance moves `xprofile → silicon` by itself.

## Out of the mechanism's scope

Checkpoint-name → quant resolution (`cli/api.py` `resolve_*`) and the
support-matrix FP4 gate are separate decisions with their own owners; this
recipe only makes the quant *estimable*.
