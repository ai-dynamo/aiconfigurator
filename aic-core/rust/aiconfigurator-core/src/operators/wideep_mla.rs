// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SGLang WideEP MLA operators (context + generation).
//!
//! Apple-to-apple port of `aiconfigurator.sdk.operations.mla.{WideEPContextMLA,
//! WideEPGenerationMLA}`. These are SGLang-only ops used by the WideEP
//! DeepSeek variant — Python loads the tables lazily and errors at query
//! time when the backend isn't `sglang`. The Rust perf-database layer
//! delegates the table miss to the operator's per-call SOL fallback,
//! matching the legacy MLA / DSA contract.
//!
//! The two ops carry the same configuration (num_heads, quant modes,
//! attention backend), but the Python signatures differ slightly: context
//! takes a `prefix` parameter so the operator can apply
//! `prefix_correction = (full_s^2 - prefix^2) / full_s^2`. Generation has
//! no prefix concept.

use serde::{Deserialize, Serialize};
use crate::common::enums::{DatabaseMode, FmhaQuantMode, KvCacheQuantMode};
use crate::common::error::AicError;
use crate::operators::base::{PerformanceResult, Source};
use crate::operators::util_empirical::{self, UtilGrid};
use crate::perf_database::wideep_mla::{
    wideep_context_mla_sol_ms, wideep_generation_mla_sol_ms,
};
use crate::perf_database::PerfDatabase;

fn prefix_correction(full_s: u32, prefix: u32) -> f64 {
    if full_s == 0 {
        return 0.0;
    }
    let f = full_s as f64;
    let p = prefix as f64;
    (f * f - p * p) / (f * f)
}

/// Python whitelists the attention backend INSIDE `get_silicon` only
/// (`mla.py:1191-1192` generation, `:1459-1460` context:
/// `if attn_backend not in {"flashinfer", "fa3"}: raise ValueError`), AFTER
/// the table's `raise_if_not_loaded()`. The EMPIRICAL path slices whatever
/// backend is requested (`mla.py:1147, 1410`) — e.g. b200's `trtllm_mla`
/// wideep slices calibrate fine there — so the check must not run for
/// EMPIRICAL or for the HYBRID fallback. The error is a Python `ValueError`,
/// deliberately NOT a missing-data signal: under HYBRID it propagates
/// instead of triggering the empirical fallback.
fn check_attn_backend(attn_backend: &str) -> Result<(), AicError> {
    if attn_backend != "flashinfer" && attn_backend != "fa3" {
        return Err(AicError::InvalidEngineConfig(format!(
            "Unsupported attention backend: {attn_backend}"
        )));
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WideEpContextMlaOp {
    pub name: String,
    pub scale_factor: f64,
    pub num_heads: u32,
    pub kv_cache_dtype: KvCacheQuantMode,
    pub fmha_quant_mode: FmhaQuantMode,
    /// Mirrors Python's `attn_backend` argument: `"flashinfer"` (default)
    /// or `"fa3"`. The CSV's `kernel_source` column carries this value.
    pub attn_backend: String,
    /// Context-parallel factor (Python `WideEPContextMLA._cp_size`). When
    /// `>1`, prefill MLA is modeled as SGLang AllGather rank-0's two zigzag
    /// chunks: `ctx(c, prefix) + ctx(c, prefix + isl - c)` with
    /// `c = ceil(isl / 2cp)`, mirroring `mla.py:1505-1510` and
    /// `operators/mla.rs::ContextMlaOp`. Absent in pre-CP specs -> 1.
    #[serde(default = "crate::operators::gemm::default_seq_split")]
    pub cp_size: u32,
}

impl WideEpContextMlaOp {
    pub fn new(
        name: impl Into<String>,
        num_heads: u32,
        kv_cache_dtype: KvCacheQuantMode,
        fmha_quant_mode: FmhaQuantMode,
    ) -> Self {
        Self {
            name: name.into(),
            scale_factor: 1.0,
            num_heads,
            kv_cache_dtype,
            fmha_quant_mode,
            attn_backend: "flashinfer".to_string(),
            cp_size: 1,
        }
    }

    pub fn query(
        &self,
        db: &PerfDatabase,
        batch_size: u32,
        isl: u32,
        prefix: u32,
    ) -> Result<PerformanceResult, AicError> {
        // ctx(s, pfx): the un-sharded wideep context-MLA query for a sequence
        // chunk of length `s` at prefix `pfx` (mode dispatch handles prefix
        // correction for silicon and the prefix-aware SOL for empirical).
        let ctx = |s: u32, pfx: u32| -> Result<(f64, Source), AicError> {
            query_wideep_context_mla_table(
                db,
                batch_size,
                s,
                pfx,
                self.num_heads,
                self.kv_cache_dtype,
                self.fmha_quant_mode,
                &self.attn_backend,
            )
        };
        // Context parallelism (SGLang AllGather / zigzag): model rank 0's two
        // balanced chunks, c = ceil(isl / 2cp). Mirrors Python
        // `WideEPContextMLA.query` (mla.py:1505-1510) and
        // `operators/mla.rs::ContextMlaOp`.
        let (latency, source) = if self.cp_size > 1 {
            let c = isl.div_ceil(2 * self.cp_size).max(1);
            let (first, first_source) = ctx(c, prefix)?;
            let (second, second_source) = ctx(c, prefix + isl - c)?;
            (first + second, first_source.combine(second_source))
        } else {
            ctx(isl, prefix)?
        };
        Ok(PerformanceResult::new(latency, source)
            .clamp_non_negative()
            .scaled(self.scale_factor))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn h200_sglang_db() -> PerfDatabase {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("src/aiconfigurator_core/systems");
        PerfDatabase::load(&root, "h200_sxm", "sglang", "0.5.10").expect("db loads")
    }

    fn op(cp_size: u32) -> WideEpContextMlaOp {
        let mut op = WideEpContextMlaOp::new(
            "wideep_ctx_mla",
            128,
            KvCacheQuantMode::Fp8,
            FmhaQuantMode::Fp8Block,
        );
        // h200 sglang 0.5.10 wideep table carries kernel_source in
        // {flashinfer, fa3} — the only backends the Python query whitelists.
        op.attn_backend = "flashinfer".to_string();
        op.cp_size = cp_size;
        op
    }

    /// CP zigzag mirrors Python `WideEPContextMLA.query` (mla.py:1505-1510):
    /// cp>1 models SGLang AllGather rank-0's two chunks
    /// `ctx(c, prefix) + ctx(c, prefix + isl - c)` with `c = ceil(isl / 2cp)`;
    /// cp=1 stays the single full-length query.
    #[test]
    fn cp_zigzag_composition() {
        let db = h200_sglang_db();
        let (b, isl, prefix) = (4u32, 4096u32, 0u32);

        // cp=1 unchanged: exactly the raw single-chunk table query (prefix=0
        // means prefix_correction = 1).
        let baseline = op(1).query(&db, b, isl, prefix).expect("cp=1 query");
        let raw = db
            .wideep_mla
            .query_context(
                b,
                isl + prefix,
                128,
                KvCacheQuantMode::Fp8,
                FmhaQuantMode::Fp8Block,
                "flashinfer",
            )
            .expect("raw table query");
        assert!(
            (baseline.latency_ms - raw).abs() < 1e-12,
            "cp=1 must remain the plain query: {} vs {}",
            baseline.latency_ms,
            raw
        );

        // cp=2 equals the two-chunk sum.
        let cp = 2u32;
        let c = isl.div_ceil(2 * cp).max(1);
        let chunked = op(cp).query(&db, b, isl, prefix).expect("cp=2 query");
        let chunk1 = op(1).query(&db, b, c, prefix).expect("chunk1 query");
        let chunk2 = op(1)
            .query(&db, b, c, prefix + isl - c)
            .expect("chunk2 query");
        assert!(
            (chunked.latency_ms - (chunk1.latency_ms + chunk2.latency_ms)).abs() < 1e-9,
            "cp=2 ({}) must equal two-chunk sum ({} + {})",
            chunked.latency_ms,
            chunk1.latency_ms,
            chunk2.latency_ms
        );
        assert!(
            chunked.latency_ms > 0.0 && chunked.latency_ms < baseline.latency_ms,
            "rank-0 CP work ({}) must be positive and below the full prefill ({})",
            chunked.latency_ms,
            baseline.latency_ms
        );
    }


    fn assert_close(got: f64, expected: f64, what: &str) {
        assert!(
            (got - expected).abs() < 1e-9,
            "{what}: expected {expected}, got {got}"
        );
    }

    /// Oracle values generated from the Python reference on the same data:
    /// `WideEPContextMLA._query_wideep_context_mla_table(db, b, s, prefix,
    /// tp_size, kv=fp8, fmha=fp8_block, attention_backend, database_mode=
    /// EMPIRICAL)` on h200_sxm/sglang/0.5.10 (`get_database(...,
    /// shared_layer=False)`). `num_heads` here is Python's `128 // tp_size`.
    /// Regenerate if the shipped table or the util math changes.
    #[test]
    fn wideep_context_mla_empirical_matches_python_oracles() {
        let mut db = h200_sglang_db();
        db.database_mode = crate::common::enums::DatabaseMode::Empirical;
        let cases: &[(u32, u32, u32, u32, &str, f64)] = &[
            // tp=8 (n=16), prefix > 0: the query SOL carries prefix natively
            (2, 3000, 1000, 16, "flashinfer", 0.7957964702937849),
            // tp=1 (n=128), exact collected hit
            (4, 4096, 0, 128, "flashinfer", 9.6274),
            // fa3 kernel slice, exact collected hit
            (4, 4096, 0, 128, "fa3", 9.7168),
        ];
        for &(b, s, prefix, n, backend, expected) in cases {
            let (latency, source) = query_wideep_context_mla_table(
                &db,
                b,
                s,
                prefix,
                n,
                KvCacheQuantMode::Fp8,
                FmhaQuantMode::Fp8Block,
                backend,
            )
            .expect("empirical query");
            assert_close(
                latency,
                expected,
                &format!("wideep_ctx_mla(b={b}, s={s}, pfx={prefix}, n={n}, {backend})"),
            );
            assert_eq!(source, Source::Empirical);
        }
        // Python capture: {"empirical"} (own-slice util, mla.py:1431).
        assert_eq!(
            db.worst_provenance(),
            util_empirical::ProvenanceTier::Empirical
        );

        // HYBRID on a kv slice with no data (bfloat16; h200's wideep tables
        // are fp8-KV only) -> terminal EmpiricalNotImplemented miss.
        db.database_mode = crate::common::enums::DatabaseMode::Hybrid;
        let result = query_wideep_context_mla_table(
            &db,
            4,
            4096,
            0,
            128,
            KvCacheQuantMode::Bfloat16,
            FmhaQuantMode::Fp8Block,
            "flashinfer",
        );
        assert!(matches!(result, Err(AicError::EmpiricalNotImplemented(_))), "got {result:?}");
    }

    /// Python oracle: `WideEPGenerationMLA._query_wideep_generation_mla_table`
    /// in EMPIRICAL mode on h200_sxm/sglang/0.5.10 (shared_layer=False). The
    /// caller's fmha label is inert (derived from the fp8 KV cache), verified
    /// on the Python side: fp8_block and bfloat16 labels give the same value.
    #[test]
    fn wideep_generation_mla_empirical_matches_python_oracles() {
        let mut db = h200_sglang_db();
        db.database_mode = crate::common::enums::DatabaseMode::Empirical;
        let cases: &[(u32, u32, u32, &str, f64)] = &[
            // tp=8 (n=16), off-grid (b, s)
            (3, 5000, 16, "flashinfer", 0.0509930128230621),
            // tp=1 (n=128), exact collected hit
            (1, 4096, 128, "flashinfer", 0.1017),
            // fa3 kernel slice, off-grid seq
            (2, 9000, 128, "fa3", 0.12152241047815426),
        ];
        for &(b, s, n, backend, expected) in cases {
            let (latency, source) = query_wideep_generation_mla_table(
                &db,
                b,
                s,
                n,
                KvCacheQuantMode::Fp8,
                backend,
            )
            .expect("empirical query");
            assert_close(
                latency,
                expected,
                &format!("wideep_gen_mla(b={b}, s={s}, n={n}, {backend})"),
            );
            assert_eq!(source, Source::Empirical);
        }
        // Python capture: {"empirical"} (own-slice util, mla.py:1162).
        assert_eq!(
            db.worst_provenance(),
            util_empirical::ProvenanceTier::Empirical
        );

        // HYBRID on a kv slice with no data (bfloat16) -> terminal miss.
        db.database_mode = crate::common::enums::DatabaseMode::Hybrid;
        let result =
            query_wideep_generation_mla_table(&db, 1, 4096, 128, KvCacheQuantMode::Bfloat16, "flashinfer");
        assert!(matches!(result, Err(AicError::EmpiricalNotImplemented(_))), "got {result:?}");
    }

    /// Python whitelists attn_backend in {flashinfer, fa3} inside
    /// `get_silicon` only (mla.py:1191-1192, 1459-1460) — SILICON (the
    /// db default here) errors on an out-of-whitelist backend even when a
    /// matching kernel_source slice exists in the data, and under HYBRID
    /// with a loaded table the ValueError propagates (a config error, not
    /// a missing-data fallback trigger).
    #[test]
    fn attn_backend_whitelist_mirrors_python() {
        let db = h200_sglang_db();
        let mut bad = op(1);
        bad.attn_backend = "trtllm_mla".to_string();
        // Pin the WHITELIST error specifically: h200 has no trtllm_mla slice,
        // so a plain `.is_err()` would also pass on a mere lookup miss
        // (`PerfDatabase`) if the whitelist were removed.
        let ctx = bad.query(&db, 1, 1024, 0);
        assert!(
            matches!(ctx, Err(AicError::InvalidEngineConfig(_))),
            "context whitelist must reject trtllm_mla, got {ctx:?}"
        );

        let mut bad_gen = WideEpGenerationMlaOp::new(
            "wideep_gen_mla",
            128,
            KvCacheQuantMode::Fp8,
            FmhaQuantMode::Fp8Block,
        );
        bad_gen.attn_backend = "trtllm_mla".to_string();
        let gen = bad_gen.query(&db, 1, 1024);
        assert!(
            matches!(gen, Err(AicError::InvalidEngineConfig(_))),
            "generation whitelist must reject trtllm_mla, got {gen:?}"
        );

        // fa3 stays allowed (h200 carries an fa3 slice).
        let mut fa3 = op(1);
        fa3.attn_backend = "fa3".to_string();
        assert!(fa3.query(&db, 1, 1024, 0).is_ok());

        // HYBRID + loaded table + bad backend: the whitelist error is NOT a
        // missing-data signal, so it propagates instead of falling back to
        // the empirical estimate (Python: plain ValueError is not in
        // `_MISSING_SILICON_DATA_EXCEPTIONS`).
        let mut hybrid = h200_sglang_db();
        hybrid.database_mode = crate::common::enums::DatabaseMode::Hybrid;
        let mut bad = op(1);
        bad.attn_backend = "trtllm_mla".to_string();
        let result = bad.query(&hybrid, 1, 1024, 0);
        assert!(
            matches!(result, Err(AicError::InvalidEngineConfig(_))),
            "HYBRID must propagate the whitelist error, got {result:?}"
        );
    }

    /// EMPIRICAL mode never runs the whitelist: it slices whatever backend
    /// is requested (mla.py:1147, 1410) — b200's wideep MLA tables are
    /// `trtllm_mla`-only and calibrate fine. Oracles:
    ///
    /// ```text
    /// db = perf_database.get_database_view("b200_sxm", "sglang", "0.5.10",
    ///     allow_missing_data=True, database_mode="EMPIRICAL", shared_layer=False)
    /// float(WideEPContextMLA._query_wideep_context_mla_table(db, b, s,
    ///     prefix, tp_size, KVCacheQuantMode.fp8, FMHAQuantMode.fp8_block,
    ///     "trtllm_mla", DatabaseMode.EMPIRICAL))   # + generation variant
    /// ```
    #[test]
    fn empirical_estimates_from_non_whitelisted_backend_slice() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("src/aiconfigurator_core/systems");
        let mut db = PerfDatabase::load(&root, "b200_sxm", "sglang", "0.5.10").expect("db loads");
        db.database_mode = crate::common::enums::DatabaseMode::Empirical;

        // (b, s, prefix, num_heads = 128 // tp_size, expected)
        let ctx_cases: &[(u32, u32, u32, u32, f64)] = &[
            (4, 4096, 0, 128, 5.1478),
            (2, 3000, 1000, 16, 0.5668068670651026),
        ];
        for &(b, s, prefix, n, expected) in ctx_cases {
            let (latency, source) = query_wideep_context_mla_table(
                &db,
                b,
                s,
                prefix,
                n,
                KvCacheQuantMode::Fp8,
                FmhaQuantMode::Fp8Block,
                "trtllm_mla",
            )
            .expect("trtllm_mla slice estimates");
            assert_close(latency, expected, &format!("trtllm_mla ctx(b={b}, s={s}, pfx={prefix})"));
            assert_eq!(source, Source::Empirical);
        }

        let (latency, source) = query_wideep_generation_mla_table(
            &db,
            3,
            5000,
            16,
            KvCacheQuantMode::Fp8,
            "trtllm_mla",
        )
        .expect("trtllm_mla gen slice estimates");
        assert_close(latency, 0.056915046282426024, "trtllm_mla gen(b=3, s=5000)");
        assert_eq!(source, Source::Empirical);
    }

    /// HYBRID with the table NOT loaded (vLLM DBs ship no wideep MLA files):
    /// the load probe is the typed miss that precedes the whitelist — the
    /// fallback runs the empirical path with the requested backend and
    /// surfaces the terminal EmpiricalNotImplemented (never the whitelist
    /// config error). Mirrors Python get_silicon's `raise_if_not_loaded()`
    /// -> whitelist ordering (mla.py:1449-1461).
    #[test]
    fn hybrid_unloaded_table_falls_to_empirical_before_whitelist() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("src/aiconfigurator_core/systems");
        let mut db = PerfDatabase::load(&root, "b200_sxm", "vllm", "0.19.0").expect("db loads");
        db.database_mode = crate::common::enums::DatabaseMode::Hybrid;
        let result = query_wideep_context_mla_table(
            &db,
            1,
            1024,
            0,
            128,
            KvCacheQuantMode::Fp8,
            FmhaQuantMode::Fp8Block,
            "trtllm_mla",
        );
        assert!(
            matches!(result, Err(AicError::EmpiricalNotImplemented(_))),
            "expected the typed empirical miss, got {result:?}"
        );
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WideEpGenerationMlaOp {
    pub name: String,
    pub scale_factor: f64,
    pub num_heads: u32,
    pub kv_cache_dtype: KvCacheQuantMode,
    /// Python `WideEPGenerationMLA` stores `_fmha_quant_mode` even though
    /// the generation perf-DB nesting doesn't key by it; carried here to
    /// keep the struct shape close to the Python class.
    pub fmha_quant_mode: FmhaQuantMode,
    pub attn_backend: String,
}

impl WideEpGenerationMlaOp {
    pub fn new(
        name: impl Into<String>,
        num_heads: u32,
        kv_cache_dtype: KvCacheQuantMode,
        fmha_quant_mode: FmhaQuantMode,
    ) -> Self {
        Self {
            name: name.into(),
            scale_factor: 1.0,
            num_heads,
            kv_cache_dtype,
            fmha_quant_mode,
            attn_backend: "flashinfer".to_string(),
        }
    }

    pub fn query(
        &self,
        db: &PerfDatabase,
        batch_size: u32,
        s: u32,
    ) -> Result<PerformanceResult, AicError> {
        let (latency, source) = query_wideep_generation_mla_table(
            db,
            batch_size,
            s,
            self.num_heads,
            self.kv_cache_dtype,
            &self.attn_backend,
        )?;
        Ok(PerformanceResult::new(latency, source)
            .clamp_non_negative()
            .scaled(self.scale_factor))
    }
}

// ---------------------------------------------------------------------------
// Database-mode dispatch, mirroring the Python `_query_wideep_*_table`
// classmethods (`operations/mla.py`): SILICON queries the table; HYBRID
// converts a typed silicon miss into the util-space empirical estimate;
// EMPIRICAL always estimates. The SOL diagnostic modes never reach the
// compiled engine. Python's classmethods take `tp_size` and use
// `num_head = 128 // tp_size` everywhere (query coordinate AND query SOL);
// these dispatches take that `num_heads` value directly, matching the op
// struct / table surface.
// ---------------------------------------------------------------------------

/// Sample-coordinate head mapping for the util-empirical grids. Python's
/// `get_empirical` sol_fn maps a collected `num_heads` coordinate via
/// `tp = round(128 / c[0])` (banker's rounding) then `128 // tp` — unlike
/// the silicon sol_fn, which floors both divisions.
fn wideep_sample_num_head(n: f64) -> f64 {
    let tp_size = (128.0 / n).round_ties_even();
    (128.0 / tp_size).floor()
}

/// WideEP context MLA latency (prefix correction applied on the silicon
/// branch) under the database's query mode.
#[allow(clippy::too_many_arguments)]
fn query_wideep_context_mla_table(
    db: &PerfDatabase,
    b: u32,
    s: u32,
    prefix: u32,
    num_heads: u32,
    kv_quant: KvCacheQuantMode,
    fmha_quant: FmhaQuantMode,
    attn_backend: &str,
) -> Result<(f64, Source), AicError> {
    let silicon = || -> Result<f64, AicError> {
        // Python get_silicon order (mla.py:1449-1461): raise_if_not_loaded
        // (typed miss -> HYBRID may still estimate) BEFORE the attn-backend
        // whitelist ValueError (propagates, never falls back).
        db.wideep_mla.ensure_context_loaded()?;
        check_attn_backend(attn_backend)?;
        let full_s = s + prefix;
        let raw = db.wideep_mla.query_context(
            b,
            full_s,
            num_heads,
            kv_quant,
            fmha_quant,
            attn_backend,
        )?;
        Ok(raw * prefix_correction(full_s, prefix))
    };
    match db.database_mode {
        DatabaseMode::Empirical => Ok((
            wideep_context_mla_empirical(db, b, s, prefix, num_heads, kv_quant, fmha_quant, attn_backend)?,
            Source::Empirical,
        )),
        DatabaseMode::Hybrid => match silicon() {
            Ok(latency) => Ok((latency, Source::Silicon)),
            Err(err) if err.is_missing_perf_data() => Ok((
                wideep_context_mla_empirical(
                    db, b, s, prefix, num_heads, kv_quant, fmha_quant, attn_backend,
                )?,
                Source::Empirical,
            )),
            Err(err) => Err(err),
        },
        _ => Ok((silicon()?, Source::Silicon)),
    }
}

/// `SOL(query)/util` over the `(attn_backend, fmha, kv)` slice's own
/// `(num_heads, s, b)` grid. Mirrors
/// `WideEPContextMLA._query_wideep_context_mla_table::get_empirical`
/// (depth 3; samples are prefix=0, the query SOL carries prefix natively;
/// the kv quant keys the slice only — the context SOL never reads it).
#[allow(clippy::too_many_arguments)]
fn wideep_context_mla_empirical(
    db: &PerfDatabase,
    b: u32,
    s: u32,
    prefix: u32,
    num_heads: u32,
    kv_quant: KvCacheQuantMode,
    fmha_quant: FmhaQuantMode,
    attn_backend: &str,
) -> Result<f64, AicError> {
    let spec = &db.system_spec;
    // c = (num_heads, full_s, b); tp = round(128 / c[0]) per Python.
    let sol = |c: &[f64]| {
        wideep_context_mla_sol_ms(spec, fmha_quant, wideep_sample_num_head(c[0]), c[1], 0.0, c[2])
    };
    let key = format!(
        "wideep_ctx_mla:{attn_backend}:{}:{}",
        fmha_quant.name(),
        kv_quant.name()
    );
    let grid = db.util_grids.get_or_try_build(&key, || {
        match db.wideep_mla.context_points(attn_backend, kv_quant, fmha_quant) {
            Ok(points) => Ok(Some(UtilGrid::new(util_empirical::build_samples(points, sol)))),
            // Typed coverage miss -> no grid (estimate() raises the
            // empirical miss); schema/load errors propagate.
            Err(err) if err.is_missing_perf_data() => Ok(None),
            Err(err) => Err(err),
        }
    })?;
    // Query SOL uses the op's own head count (= Python's `128 // tp_size`).
    let sol_query = wideep_context_mla_sol_ms(
        spec,
        fmha_quant,
        num_heads as f64,
        s as f64,
        prefix as f64,
        b as f64,
    );
    let query = [num_heads as f64, (s + prefix) as f64, b as f64];
    let (latency, _) = util_empirical::estimate(sol_query, &query, grid.as_deref(), 1.0)?;
    // Own-slice util fired (Python mla.py:1431 estimate()'s default tier).
    db.note_provenance(util_empirical::ProvenanceTier::Empirical);
    Ok(latency)
}

/// WideEP generation MLA latency under the database's query mode. The fmha
/// mode is derived from the kv-cache dtype (fp8 KV -> fp8, else bfloat16),
/// exactly like the top of Python's
/// `_query_wideep_generation_mla_table` — the caller's fmha label is inert
/// for generation.
fn query_wideep_generation_mla_table(
    db: &PerfDatabase,
    b: u32,
    s: u32,
    num_heads: u32,
    kv_quant: KvCacheQuantMode,
    attn_backend: &str,
) -> Result<(f64, Source), AicError> {
    let silicon = || -> Result<f64, AicError> {
        // Python get_silicon order (mla.py:1188-1192): raise_if_not_loaded
        // (typed miss -> HYBRID may still estimate) BEFORE the attn-backend
        // whitelist ValueError (propagates, never falls back).
        db.wideep_mla.ensure_generation_loaded()?;
        check_attn_backend(attn_backend)?;
        db.wideep_mla
            .query_generation(b, s, num_heads, kv_quant, attn_backend)
    };
    match db.database_mode {
        DatabaseMode::Empirical => Ok((
            wideep_generation_mla_empirical(db, b, s, num_heads, kv_quant, attn_backend)?,
            Source::Empirical,
        )),
        DatabaseMode::Hybrid => match silicon() {
            Ok(latency) => Ok((latency, Source::Silicon)),
            Err(err) if err.is_missing_perf_data() => Ok((
                wideep_generation_mla_empirical(db, b, s, num_heads, kv_quant, attn_backend)?,
                Source::Empirical,
            )),
            Err(err) => Err(err),
        },
        _ => Ok((silicon()?, Source::Silicon)),
    }
}

/// `SOL(query)/util` over the `(attn_backend, kv)` slice's own
/// `(num_heads, b, s)` grid. Mirrors
/// `WideEPGenerationMLA._query_wideep_generation_mla_table::get_empirical`.
fn wideep_generation_mla_empirical(
    db: &PerfDatabase,
    b: u32,
    s: u32,
    num_heads: u32,
    kv_quant: KvCacheQuantMode,
    attn_backend: &str,
) -> Result<f64, AicError> {
    // Decode compute dtype follows the kv-cache dtype (Python overrides the
    // passed fmha label before get_sol is defined).
    let fmha_quant = if kv_quant == KvCacheQuantMode::Fp8 {
        FmhaQuantMode::Fp8
    } else {
        FmhaQuantMode::Bfloat16
    };
    let spec = &db.system_spec;
    // c = (num_heads, b, s); tp = round(128 / c[0]) per Python.
    let sol = |c: &[f64]| {
        wideep_generation_mla_sol_ms(spec, fmha_quant, wideep_sample_num_head(c[0]), c[1], c[2])
    };
    let key = format!("wideep_gen_mla:{attn_backend}:{}", kv_quant.name());
    let grid = db.util_grids.get_or_try_build(&key, || {
        match db.wideep_mla.generation_points(attn_backend, kv_quant) {
            Ok(points) => Ok(Some(UtilGrid::new(util_empirical::build_samples(points, sol)))),
            Err(err) if err.is_missing_perf_data() => Ok(None),
            Err(err) => Err(err),
        }
    })?;
    // Query SOL uses the op's own head count (= Python's `128 // tp_size`).
    let sol_query = wideep_generation_mla_sol_ms(spec, fmha_quant, num_heads as f64, b as f64, s as f64);
    let query = [num_heads as f64, b as f64, s as f64];
    let (latency, _) = util_empirical::estimate(sol_query, &query, grid.as_deref(), 1.0)?;
    // Own-slice util fired (Python mla.py:1162 estimate()'s default tier).
    db.note_provenance(util_empirical::ProvenanceTier::Empirical);
    Ok(latency)
}
