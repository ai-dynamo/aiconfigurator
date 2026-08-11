// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! MLA operators: op-level context/generation, module-level
//! context/generation, and MLA BMM (pre/post).
//!
//! Mirrors `aiconfigurator.sdk.operations.mla.{ContextMLA, GenerationMLA,
//! MLAModule, MLABmm}`. Op-level paths apply Python's prefix-correction
//! multiplier inside the mode dispatch (silicon branch only — the empirical
//! branch's SOL carries prefix natively, exactly like Python's
//! `get_empirical`); module-level paths do the same since the perf-DB layer
//! returns raw table values. MLA BMM has a quant-mode fallback to bfloat16
//! as part of slice selection (silicon inside the perf-DB query, empirical
//! before grid construction).

use crate::common::enums::{DatabaseMode, FmhaQuantMode, GemmQuantMode, KvCacheQuantMode};
use crate::common::error::AicError;
use crate::operators::base::{PerformanceResult, Source};
use crate::operators::util_empirical::{self, UtilGrid};
use crate::perf_database::attention::generation_attn_flops;
use crate::perf_database::gemm::quant_tc_flops;
use crate::perf_database::mla::{
    context_mla_sol_ms, context_mla_sol_prefix_ms, generation_mla_module_sol_ms,
    generation_mla_sol_ms, mla_bmm_sol_ms,
};
use crate::perf_database::PerfDatabase;
use serde::{Deserialize, Serialize};

fn prefix_correction(full_s: u32, prefix: u32) -> f64 {
    if full_s == 0 {
        return 0.0;
    }
    let f = full_s as f64;
    let p = prefix as f64;
    (f * f - p * p) / (f * f)
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ContextMlaOp {
    pub name: String,
    pub scale_factor: f64,
    pub num_heads: u32,
    pub kv_cache_dtype: KvCacheQuantMode,
    pub fmha_quant_mode: FmhaQuantMode,
    /// Context-parallel factor (Python `ContextMLA._cp_size`). When `>1`,
    /// prefill MLA is modeled as SGLang AllGather rank-0's two zigzag chunks:
    /// `ctx(c, prefix) + ctx(c, prefix + isl - c)` with `c = ceil(isl / 2cp)`,
    /// mirroring `operators/attention.rs::ContextAttentionOp`. Absent in
    /// pre-CP specs -> 1 (no sharding).
    #[serde(default = "crate::operators::gemm::default_seq_split")]
    pub cp_size: u32,
}

impl ContextMlaOp {
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
        // ctx(s, pfx): the un-sharded context-MLA query for a sequence chunk of
        // length `s` at prefix `pfx` (mode dispatch handles prefix correction
        // for silicon and the prefix-aware SOL for empirical).
        let ctx = |s: u32, pfx: u32| -> Result<PerformanceResult, AicError> {
            query_context_mla_table(
                db,
                batch_size,
                s,
                pfx,
                self.num_heads,
                self.kv_cache_dtype,
                self.fmha_quant_mode,
            )
        };
        // Context parallelism (SGLang AllGather / zigzag): model rank 0's two
        // balanced chunks, c = ceil(isl / 2cp). Mirrors Python
        // `ContextMLA.query` and `operators/attention.rs::ContextAttentionOp`.
        // Latency and energy both sum across the chunks (Python `__add__`).
        let result = if self.cp_size > 1 {
            let c = isl.div_ceil(2 * self.cp_size).max(1);
            ctx(c, prefix)?.plus(ctx(c, prefix + isl - c)?)
        } else {
            ctx(isl, prefix)?
        };
        Ok(result.clamp_non_negative().scaled(self.scale_factor))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GenerationMlaOp {
    pub name: String,
    pub scale_factor: f64,
    pub num_heads: u32,
    pub kv_cache_dtype: KvCacheQuantMode,
}

impl GenerationMlaOp {
    pub fn new(name: impl Into<String>, num_heads: u32, kv_cache_dtype: KvCacheQuantMode) -> Self {
        Self {
            name: name.into(),
            scale_factor: 1.0,
            num_heads,
            kv_cache_dtype,
        }
    }

    pub fn query(
        &self,
        db: &PerfDatabase,
        batch_size: u32,
        s: u32,
    ) -> Result<PerformanceResult, AicError> {
        let result =
            query_generation_mla_table(db, batch_size, s, self.num_heads, self.kv_cache_dtype)?;
        Ok(result.clamp_non_negative().scaled(self.scale_factor))
    }
}

/// Module-level MLA operator (context + generation in one struct since
/// they share config-time fields).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MlaModuleOp {
    pub name: String,
    pub scale_factor: f64,
    pub num_heads: u32,
    pub kv_cache_dtype: KvCacheQuantMode,
    pub fmha_quant_mode: FmhaQuantMode,
    pub gemm_quant_mode: GemmQuantMode,
    /// Model-native identity for the `[native][local]` module table (#1458).
    /// `None` = legacy single-native resolution; `serde(default)` covers JSON
    /// specs predating the field. NO `skip_serializing_if`: bincode decodes
    /// positionally, so an omitted field would desync every op decoded after
    /// it — the layout change is gated by the ENGINE_SPEC_SCHEMA_VERSION bump.
    #[serde(default)]
    pub native_num_heads: Option<u32>,
}

impl MlaModuleOp {
    pub fn new(
        name: impl Into<String>,
        num_heads: u32,
        kv_cache_dtype: KvCacheQuantMode,
        fmha_quant_mode: FmhaQuantMode,
        gemm_quant_mode: GemmQuantMode,
    ) -> Self {
        Self {
            name: name.into(),
            scale_factor: 1.0,
            num_heads,
            kv_cache_dtype,
            fmha_quant_mode,
            gemm_quant_mode,
            native_num_heads: None,
        }
    }

    pub fn query_context(
        &self,
        db: &PerfDatabase,
        batch_size: u32,
        isl: u32,
        prefix: u32,
    ) -> Result<PerformanceResult, AicError> {
        let result = query_context_mla_module_table(
            db,
            batch_size,
            isl,
            prefix,
            self.num_heads,
            self.kv_cache_dtype,
            self.fmha_quant_mode,
            self.gemm_quant_mode,
            self.native_num_heads,
        )?;
        Ok(result.clamp_non_negative().scaled(self.scale_factor))
    }

    pub fn query_generation(
        &self,
        db: &PerfDatabase,
        batch_size: u32,
        s: u32,
    ) -> Result<PerformanceResult, AicError> {
        // No fmha arg: the generation module table has no fmha axis (decode
        // compute dtype follows the kv-cache dtype).
        let result = query_generation_mla_module_table(
            db,
            batch_size,
            s,
            self.num_heads,
            self.kv_cache_dtype,
            self.gemm_quant_mode,
            self.native_num_heads,
        )?;
        Ok(result.clamp_non_negative().scaled(self.scale_factor))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MlaBmmOp {
    pub name: String,
    pub scale_factor: f64,
    pub num_heads: u32,
    pub quant_mode: GemmQuantMode,
    pub is_pre: bool,
}

impl MlaBmmOp {
    pub fn new(
        name: impl Into<String>,
        num_heads: u32,
        quant_mode: GemmQuantMode,
        is_pre: bool,
    ) -> Self {
        Self {
            name: name.into(),
            scale_factor: 1.0,
            num_heads,
            quant_mode,
            is_pre,
        }
    }

    pub fn query(&self, db: &PerfDatabase, num_tokens: u32) -> Result<PerformanceResult, AicError> {
        let result =
            query_mla_bmm_table(db, num_tokens, self.num_heads, self.quant_mode, self.is_pre)?;
        Ok(result.clamp_non_negative().scaled(self.scale_factor))
    }
}

// ---------------------------------------------------------------------------
// Database-mode dispatch, mirroring the Python `_query_*_table` classmethods
// (`operations/mla.py`): SILICON queries the table; HYBRID converts a typed
// silicon miss into the util-space empirical estimate; EMPIRICAL always
// estimates; SOL (and the retired SOL_FULL alias) returns the pure
// speed-of-light roofline with `Source::Sol` and zero energy.
// ---------------------------------------------------------------------------

/// Op-level context MLA latency + energy (prefix correction applied to BOTH
/// on the silicon branch — Python mla.py `get_silicon`) under the database's
/// query mode. Empirical estimates carry no energy.
fn query_context_mla_table(
    db: &PerfDatabase,
    b: u32,
    s: u32,
    prefix: u32,
    num_heads: u32,
    kv_quant: KvCacheQuantMode,
    fmha_quant: FmhaQuantMode,
) -> Result<PerformanceResult, AicError> {
    let silicon = || -> Result<PerformanceResult, AicError> {
        let full_s = s + prefix;
        let raw = db
            .mla
            .query_context(b, full_s, num_heads, kv_quant, fmha_quant)?;
        let correction = prefix_correction(full_s, prefix);
        Ok(PerformanceResult::with_energy(
            raw.latency * correction,
            raw.energy * correction,
            Source::Silicon,
        ))
    };
    match db.database_mode {
        // Python `_query_context_mla_table`: `get_sol(b, s, prefix, num_heads,
        // kvcache_quant_mode, fmha_quant_mode)[0]` — prefix lives inside the
        // SOL formula, so no prefix correction applies here.
        DatabaseMode::Sol | DatabaseMode::SolFull => {
            let attn_flops = quant_tc_flops(&db.system_spec, fmha_quant.mapping())?;
            Ok(PerformanceResult::new(
                context_mla_sol_prefix_ms(
                    &db.system_spec,
                    kv_quant,
                    num_heads as f64,
                    s as f64,
                    prefix as f64,
                    b as f64,
                    attn_flops,
                ),
                Source::Sol,
            ))
        }
        DatabaseMode::Empirical => Ok(PerformanceResult::new(
            context_mla_empirical(db, b, s, prefix, num_heads, kv_quant, fmha_quant)?,
            Source::Empirical,
        )),
        DatabaseMode::Hybrid => match silicon() {
            Ok(result) => Ok(result),
            Err(err) if err.is_missing_perf_data() => Ok(PerformanceResult::new(
                context_mla_empirical(db, b, s, prefix, num_heads, kv_quant, fmha_quant)?,
                Source::Empirical,
            )),
            Err(err) => Err(err),
        },
        _ => silicon(),
    }
}

/// `SOL(query)/util` over the (fmha, kv) slice's own `(num_heads, s, b)`
/// grid. Mirrors Python `ContextMLA._query_context_mla_table::get_empirical`
/// (depth 3; samples are prefix=0, the query SOL carries prefix natively).
fn context_mla_empirical(
    db: &PerfDatabase,
    b: u32,
    s: u32,
    prefix: u32,
    num_heads: u32,
    kv_quant: KvCacheQuantMode,
    fmha_quant: FmhaQuantMode,
) -> Result<f64, AicError> {
    let spec = &db.system_spec;
    let attn_flops = quant_tc_flops(spec, fmha_quant.mapping())?;
    // c = (num_heads, full_s, b), prefix = 0 for collected samples.
    let sol = |c: &[f64]| context_mla_sol_ms(spec, kv_quant, c[0], c[1], c[2], attn_flops);
    let key = format!("ctx_mla:{}:{}", fmha_quant.name(), kv_quant.name());
    let grid = db.util_grids.get_or_try_build(&key, || {
        match db.mla.context_points(kv_quant, fmha_quant) {
            Ok(points) => Ok(Some(UtilGrid::new(util_empirical::build_samples(
                points, sol,
            )))),
            // Typed coverage miss -> no grid (estimate() raises the
            // empirical miss); schema/load errors propagate.
            Err(err) if err.is_missing_perf_data() => Ok(None),
            Err(err) => Err(err),
        }
    })?;
    let sol_query = context_mla_sol_prefix_ms(
        spec,
        kv_quant,
        num_heads as f64,
        s as f64,
        prefix as f64,
        b as f64,
        attn_flops,
    );
    let query = [num_heads as f64, (s + prefix) as f64, b as f64];
    let (latency, _) = util_empirical::estimate(sol_query, &query, grid.as_deref(), 1.0)?;
    // Own-shape util fired (Python mla.py, estimate()'s default tier).
    db.note_provenance(util_empirical::ProvenanceTier::Empirical);
    Ok(latency)
}

/// Op-level generation MLA latency + energy under the database's query
/// mode. Empirical estimates carry no energy.
fn query_generation_mla_table(
    db: &PerfDatabase,
    b: u32,
    s: u32,
    num_heads: u32,
    kv_quant: KvCacheQuantMode,
) -> Result<PerformanceResult, AicError> {
    let silicon = |v: crate::perf_database::perf_interp::LeafValue| {
        PerformanceResult::with_energy(v.latency, v.energy, Source::Silicon)
    };
    match db.database_mode {
        // Python `_query_generation_mla_table`: `get_sol(b, s, num_heads,
        // kvcache_quant_mode)[0]` — flops implied by the kv-cache dtype.
        DatabaseMode::Sol | DatabaseMode::SolFull => {
            let attn_flops = generation_attn_flops(&db.system_spec, kv_quant)?;
            Ok(PerformanceResult::new(
                generation_mla_sol_ms(
                    &db.system_spec,
                    kv_quant,
                    num_heads as f64,
                    b as f64,
                    s as f64,
                    attn_flops,
                ),
                Source::Sol,
            ))
        }
        DatabaseMode::Empirical => Ok(PerformanceResult::new(
            generation_mla_empirical(db, b, s, num_heads, kv_quant)?,
            Source::Empirical,
        )),
        DatabaseMode::Hybrid => match db.mla.query_generation(b, s, num_heads, kv_quant) {
            Ok(value) => Ok(silicon(value)),
            Err(err) if err.is_missing_perf_data() => Ok(PerformanceResult::new(
                generation_mla_empirical(db, b, s, num_heads, kv_quant)?,
                Source::Empirical,
            )),
            Err(err) => Err(err),
        },
        _ => Ok(silicon(db.mla.query_generation(b, s, num_heads, kv_quant)?)),
    }
}

/// `SOL(query)/util` over the kv slice's own `(num_heads, b, s)` grid.
/// Mirrors `GenerationMLA._query_generation_mla_table::get_empirical`.
fn generation_mla_empirical(
    db: &PerfDatabase,
    b: u32,
    s: u32,
    num_heads: u32,
    kv_quant: KvCacheQuantMode,
) -> Result<f64, AicError> {
    let spec = &db.system_spec;
    let attn_flops = generation_attn_flops(spec, kv_quant)?;
    // c = (num_heads, b, s).
    let sol = |c: &[f64]| generation_mla_sol_ms(spec, kv_quant, c[0], c[1], c[2], attn_flops);
    let key = format!("gen_mla:{}", kv_quant.name());
    let grid =
        db.util_grids
            .get_or_try_build(&key, || match db.mla.generation_points(kv_quant) {
                Ok(points) => Ok(Some(UtilGrid::new(util_empirical::build_samples(
                    points, sol,
                )))),
                Err(err) if err.is_missing_perf_data() => Ok(None),
                Err(err) => Err(err),
            })?;
    let query = [num_heads as f64, b as f64, s as f64];
    let (latency, _) = util_empirical::estimate(sol(&query), &query, grid.as_deref(), 1.0)?;
    // Own-shape util fired (Python mla.py, estimate()'s default tier).
    db.note_provenance(util_empirical::ProvenanceTier::Empirical);
    Ok(latency)
}

/// Head slice the BMM table queries run against, with the linear scale to
/// apply to that slice's result. Exact-head-first: return the requested
/// head count at scale 1.0 when the table has rows for it, else the next
/// power of two — the DeepSeek grid every dataset carries (exact rows for
/// non-pow2 shards, e.g. Kimi-K3's 96/48/24/12, exist only where
/// re-collected). A missing table/slice also resolves to the pow2 fallback
/// so downstream misses keep the legacy error shape. Python twin:
/// `MLABmm._resolve_slice_heads`.
fn resolve_bmm_slice_heads(
    db: &PerfDatabase,
    num_heads: u32,
    quant: GemmQuantMode,
    is_pre: bool,
) -> Result<(u32, f64), AicError> {
    let pow2 = num_heads.next_power_of_two();
    if pow2 == num_heads {
        return Ok((num_heads, 1.0));
    }
    let has_exact = match db.mla.bmm_selected_quant(quant) {
        Ok(selected) => db.mla.bmm_has_heads(selected, is_pre, num_heads)?,
        Err(err) if err.is_missing_perf_data() => false,
        Err(err) => return Err(err),
    };
    if has_exact {
        Ok((num_heads, 1.0))
    } else {
        Ok((pow2, f64::from(num_heads) / f64::from(pow2)))
    }
}

/// MLA BMM (pre/post) latency + energy under the database's query mode.
/// The head_scale multiplies latency AND energy on the silicon branch
/// (Python: `_interp_pr(lat * head_scale, energy=energy * head_scale)`);
/// empirical estimates carry no energy.
fn query_mla_bmm_table(
    db: &PerfDatabase,
    num_tokens: u32,
    num_heads: u32,
    quant: GemmQuantMode,
    is_pre: bool,
) -> Result<PerformanceResult, AicError> {
    // Python `_query_mla_bmm_table` dispatches the SOL modes BEFORE the head
    // routing — the SOL is exactly linear in num_heads and touches no table
    // (`get_sol(num_tokens, num_heads, quant_mode, if_pre)[0]`, if_pre unused).
    if matches!(db.database_mode, DatabaseMode::Sol | DatabaseMode::SolFull) {
        let spec = &db.system_spec;
        let bmm_flops = quant_tc_flops(spec, quant.mapping())?;
        return Ok(PerformanceResult::new(
            mla_bmm_sol_ms(spec, quant, num_heads as f64, num_tokens as f64, bmm_flops),
            Source::Sol,
        ));
    }
    // Exact-head-first routing with a data-presence fallback: query the
    // exact head slice at scale 1.0 when it has rows, otherwise the
    // next-pow2 DeepSeek slice scaled linearly by the head ratio (BMM is
    // per-head batched; reproduces the legacy count-ratio modeling for
    // Kimi-K3's 96-family shards). Python twin:
    // `MLABmm._query_mla_bmm_table`.
    let (num_heads, head_scale) = resolve_bmm_slice_heads(db, num_heads, quant, is_pre)?;
    let silicon = |v: crate::perf_database::perf_interp::LeafValue| {
        PerformanceResult::with_energy(
            v.latency * head_scale,
            v.energy * head_scale,
            Source::Silicon,
        )
    };
    match db.database_mode {
        DatabaseMode::Empirical => Ok(PerformanceResult::new(
            mla_bmm_empirical(db, num_tokens, num_heads, quant, is_pre)? * head_scale,
            Source::Empirical,
        )),
        DatabaseMode::Hybrid => match db.mla.query_bmm(num_tokens, num_heads, quant, is_pre) {
            Ok(value) => Ok(silicon(value)),
            Err(err) if err.is_missing_perf_data() => Ok(PerformanceResult::new(
                mla_bmm_empirical(db, num_tokens, num_heads, quant, is_pre)? * head_scale,
                Source::Empirical,
            )),
            Err(err) => Err(err),
        },
        _ => Ok(silicon(
            db.mla.query_bmm(num_tokens, num_heads, quant, is_pre)?,
        )),
    }
}

/// `SOL(query)/util` over the 1-D `num_tokens` curve of the selected
/// `(quant, op_name, num_heads)` slice. Mirrors
/// `MLABmm._query_mla_bmm_table::get_empirical`: slice selection falls back
/// to the bfloat16 quant when the requested quant has no BMM data at all
/// (BEFORE estimate()); the SOL keeps using the REQUESTED quant either way.
fn mla_bmm_empirical(
    db: &PerfDatabase,
    num_tokens: u32,
    num_heads: u32,
    quant: GemmQuantMode,
    is_pre: bool,
) -> Result<f64, AicError> {
    let spec = &db.system_spec;
    let bmm_flops = quant_tc_flops(spec, quant.mapping())?;
    // c = (num_tokens,); the SOL is bound to the REQUESTED quant.
    let sol = |c: &[f64]| mla_bmm_sol_ms(spec, quant, num_heads as f64, c[0], bmm_flops);
    let op_name = if is_pre {
        "mla_gen_pre"
    } else {
        "mla_gen_post"
    };
    // Slice selection first (Python: `qm = quant if quant in wrapper else
    // bfloat16`); a typed miss here means the whole BMM table is absent.
    let grid = match db.mla.bmm_selected_quant(quant) {
        Ok(selected) => {
            // The key carries the REQUESTED and the ACTUALLY-selected quant:
            // on the bfloat16 fallback the samples still get the requested
            // quant's SOL, so grids from the same slice under different
            // requested quants must not alias (Python keys on the requested
            // quant plus the concrete node identity).
            let key = format!(
                "mla_bmm:{}:{}:{}:{}",
                quant.name(),
                selected.name(),
                op_name,
                num_heads
            );
            db.util_grids.get_or_try_build(&key, || {
                match db.mla.bmm_points(selected, is_pre, num_heads) {
                    Ok(points) => Ok(Some(UtilGrid::new(util_empirical::build_samples(
                        points, sol,
                    )))),
                    Err(err) if err.is_missing_perf_data() => Ok(None),
                    Err(err) => Err(err),
                }
            })?
        }
        Err(err) if err.is_missing_perf_data() => None,
        Err(err) => return Err(err),
    };
    let query = [num_tokens as f64];
    let (latency, _) = util_empirical::estimate(sol(&query), &query, grid.as_deref(), 1.0)?;
    // Own-shape util fired (Python mla.py, estimate()'s default tier).
    db.note_provenance(util_empirical::ProvenanceTier::Empirical);
    Ok(latency)
}

/// Module-level context MLA latency (prefix correction on the silicon
/// branch) under the database's query mode.
#[allow(clippy::too_many_arguments)]
fn query_context_mla_module_table(
    db: &PerfDatabase,
    b: u32,
    s: u32,
    prefix: u32,
    num_heads: u32,
    kv_quant: KvCacheQuantMode,
    fmha_quant: FmhaQuantMode,
    gemm_quant: GemmQuantMode,
    native_heads: Option<u32>,
) -> Result<PerformanceResult, AicError> {
    let silicon = || -> Result<PerformanceResult, AicError> {
        let full_s = s + prefix;
        let raw = db.mla.query_context_module(
            b,
            full_s,
            num_heads,
            kv_quant,
            fmha_quant,
            gemm_quant,
            native_heads,
        )?;
        let correction = prefix_correction(full_s, prefix);
        Ok(PerformanceResult::with_energy(
            raw.latency * correction,
            raw.energy * correction,
            Source::Silicon,
        ))
    };
    match db.database_mode {
        // Python `_query_context_mla_module_table`: same SOL model as the
        // op-level context MLA — `get_sol(b, s, prefix, num_heads,
        // kvcache_quant_mode, fmha_quant_mode)[0]` (gemm quant / native heads
        // only select the empirical slice; they never enter the SOL).
        DatabaseMode::Sol | DatabaseMode::SolFull => {
            let attn_flops = quant_tc_flops(&db.system_spec, fmha_quant.mapping())?;
            Ok(PerformanceResult::new(
                context_mla_sol_prefix_ms(
                    &db.system_spec,
                    kv_quant,
                    num_heads as f64,
                    s as f64,
                    prefix as f64,
                    b as f64,
                    attn_flops,
                ),
                Source::Sol,
            ))
        }
        DatabaseMode::Empirical => Ok(PerformanceResult::new(
            context_mla_module_empirical(
                db,
                b,
                s,
                prefix,
                num_heads,
                kv_quant,
                fmha_quant,
                gemm_quant,
                native_heads,
            )?,
            Source::Empirical,
        )),
        DatabaseMode::Hybrid => match silicon() {
            Ok(result) => Ok(result),
            Err(err) if err.is_missing_perf_data() => Ok(PerformanceResult::new(
                context_mla_module_empirical(
                    db,
                    b,
                    s,
                    prefix,
                    num_heads,
                    kv_quant,
                    fmha_quant,
                    gemm_quant,
                    native_heads,
                )?,
                Source::Empirical,
            )),
            Err(err) => Err(err),
        },
        _ => silicon(),
    }
}

/// Mirrors `MLAModule._query_context_mla_module_table::get_empirical`: same
/// SOL as the op-level context MLA (the gemm quant only selects the slice),
/// over the module's own `(num_heads, s, b)` grid.
#[allow(clippy::too_many_arguments)]
fn context_mla_module_empirical(
    db: &PerfDatabase,
    b: u32,
    s: u32,
    prefix: u32,
    num_heads: u32,
    kv_quant: KvCacheQuantMode,
    fmha_quant: FmhaQuantMode,
    gemm_quant: GemmQuantMode,
    native_heads: Option<u32>,
) -> Result<f64, AicError> {
    let spec = &db.system_spec;
    let attn_flops = quant_tc_flops(spec, fmha_quant.mapping())?;
    // c = (num_heads, full_s, b), prefix = 0 for collected samples.
    let sol = |c: &[f64]| context_mla_sol_ms(spec, kv_quant, c[0], c[1], c[2], attn_flops);
    // Native in the cache key: distinct native buckets are distinct grids (#1458).
    let key = format!(
        "ctx_mla_mod:{}:{}:{}:{:?}",
        fmha_quant.name(),
        kv_quant.name(),
        gemm_quant.name(),
        native_heads
    );
    let grid = db.util_grids.get_or_try_build(&key, || {
        match db
            .mla
            .context_module_points(kv_quant, fmha_quant, gemm_quant, native_heads)
        {
            Ok(points) => Ok(Some(UtilGrid::new(util_empirical::build_samples(
                points, sol,
            )))),
            Err(err) if err.is_missing_perf_data() => Ok(None),
            Err(err) => Err(err),
        }
    })?;
    let sol_query = context_mla_sol_prefix_ms(
        spec,
        kv_quant,
        num_heads as f64,
        s as f64,
        prefix as f64,
        b as f64,
        attn_flops,
    );
    let query = [num_heads as f64, (s + prefix) as f64, b as f64];
    let (latency, _) = util_empirical::estimate(sol_query, &query, grid.as_deref(), 1.0)?;
    // Own-shape util fired (Python mla.py, estimate()'s default tier).
    db.note_provenance(util_empirical::ProvenanceTier::Empirical);
    Ok(latency)
}

/// Module-level generation MLA latency + energy under the database's query
/// mode. Empirical estimates carry no energy.
fn query_generation_mla_module_table(
    db: &PerfDatabase,
    b: u32,
    s: u32,
    num_heads: u32,
    kv_quant: KvCacheQuantMode,
    gemm_quant: GemmQuantMode,
    native_heads: Option<u32>,
) -> Result<PerformanceResult, AicError> {
    let silicon = |v: crate::perf_database::perf_interp::LeafValue| {
        PerformanceResult::with_energy(v.latency, v.energy, Source::Silicon)
    };
    match db.database_mode {
        // Python `_query_generation_mla_module_table`: `get_sol(b, s,
        // num_heads, kv_cache_dtype)[0]` — MLA attention SOL plus the BMM
        // pre+post SOL folded into the same math/mem terms before the max.
        DatabaseMode::Sol | DatabaseMode::SolFull => {
            let spec = &db.system_spec;
            let attn_flops = generation_attn_flops(spec, kv_quant)?;
            let bmm_flops = quant_tc_flops(spec, gemm_quant.mapping())?;
            Ok(PerformanceResult::new(
                generation_mla_module_sol_ms(
                    spec,
                    kv_quant,
                    gemm_quant,
                    num_heads as f64,
                    b as f64,
                    s as f64,
                    attn_flops,
                    bmm_flops,
                ),
                Source::Sol,
            ))
        }
        DatabaseMode::Empirical => Ok(PerformanceResult::new(
            generation_mla_module_empirical(
                db,
                b,
                s,
                num_heads,
                kv_quant,
                gemm_quant,
                native_heads,
            )?,
            Source::Empirical,
        )),
        DatabaseMode::Hybrid => {
            match db.mla.query_generation_module(
                b,
                s,
                num_heads,
                kv_quant,
                gemm_quant,
                native_heads,
            ) {
                Ok(value) => Ok(silicon(value)),
                Err(err) if err.is_missing_perf_data() => Ok(PerformanceResult::new(
                    generation_mla_module_empirical(
                        db,
                        b,
                        s,
                        num_heads,
                        kv_quant,
                        gemm_quant,
                        native_heads,
                    )?,
                    Source::Empirical,
                )),
                Err(err) => Err(err),
            }
        }
        _ => Ok(silicon(db.mla.query_generation_module(
            b,
            s,
            num_heads,
            kv_quant,
            gemm_quant,
            native_heads,
        )?)),
    }
}

/// Mirrors `MLAModule._query_generation_mla_module_table::get_empirical`:
/// generation MLA SOL + BMM pre/post terms (the module SOL closes over the
/// gemm quant), over the module's own `(num_heads, b, s)` grid.
fn generation_mla_module_empirical(
    db: &PerfDatabase,
    b: u32,
    s: u32,
    num_heads: u32,
    kv_quant: KvCacheQuantMode,
    gemm_quant: GemmQuantMode,
    native_heads: Option<u32>,
) -> Result<f64, AicError> {
    let spec = &db.system_spec;
    let attn_flops = generation_attn_flops(spec, kv_quant)?;
    let bmm_flops = quant_tc_flops(spec, gemm_quant.mapping())?;
    // c = (num_heads, b, s).
    let sol = |c: &[f64]| {
        generation_mla_module_sol_ms(
            spec, kv_quant, gemm_quant, c[0], c[1], c[2], attn_flops, bmm_flops,
        )
    };
    let key = format!(
        "gen_mla_mod:{}:{}:{:?}",
        kv_quant.name(),
        gemm_quant.name(),
        native_heads
    );
    let grid = db.util_grids.get_or_try_build(&key, || {
        match db
            .mla
            .generation_module_points(kv_quant, gemm_quant, native_heads)
        {
            Ok(points) => Ok(Some(UtilGrid::new(util_empirical::build_samples(
                points, sol,
            )))),
            Err(err) if err.is_missing_perf_data() => Ok(None),
            Err(err) => Err(err),
        }
    })?;
    let query = [num_heads as f64, b as f64, s as f64];
    let (latency, _) = util_empirical::estimate(sol(&query), &query, grid.as_deref(), 1.0)?;
    // Own-shape util fired (Python mla.py, estimate()'s default tier).
    db.note_provenance(util_empirical::ProvenanceTier::Empirical);
    Ok(latency)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    const REPO_ROOT_HINT: &str = env!("CARGO_MANIFEST_DIR");

    fn b200_vllm_db() -> PerfDatabase {
        let systems_root = PathBuf::from(REPO_ROOT_HINT)
            .join("../..")
            .join("src/aiconfigurator_core/systems");
        PerfDatabase::load(&systems_root, "b200_sxm", "vllm", "0.19.0").expect("db must load")
    }

    #[test]
    fn mla_module_context_smoke() {
        let db = b200_vllm_db();
        let op = MlaModuleOp::new(
            "ctx_mod",
            128,
            KvCacheQuantMode::Bfloat16,
            FmhaQuantMode::Bfloat16,
            GemmQuantMode::Bfloat16,
        );
        // Exact-hit row latency=0.1351, prefix=0 means prefix_correction=1.0.
        let result = op.query_context(&db, 1, 1, 0).expect("query must succeed");
        assert!(
            (result.latency_ms - 0.1351).abs() < 1e-6,
            "expected recorded module latency, got {}",
            result.latency_ms
        );
    }

    #[test]
    fn mla_op_context_absent_on_vllm_b200() {
        let db = b200_vllm_db();
        let op = ContextMlaOp::new(
            "ctx_op",
            128,
            KvCacheQuantMode::Bfloat16,
            FmhaQuantMode::Bfloat16,
        );
        // vLLM b200 only ships module-level MLA — op-level should error.
        let err = op.query(&db, 1, 1024, 0).unwrap_err();
        match err {
            AicError::Io { .. } | AicError::PerfDatabase(_) => {}
            other => panic!("unexpected error: {other:?}"),
        }
    }

    fn gb200_trtllm_db() -> PerfDatabase {
        let systems_root = PathBuf::from(REPO_ROOT_HINT)
            .join("../..")
            .join("src/aiconfigurator_core/systems");
        PerfDatabase::load(&systems_root, "gb200", "trtllm", "1.3.0rc10").expect("db must load")
    }

    fn assert_close(got: f64, expected: f64, what: &str) {
        assert!(
            (got - expected).abs() < 1e-9,
            "{what}: expected {expected}, got {got}"
        );
    }

    /// Oracle values generated from the Python reference on the same data:
    /// `ContextMLA._query_context_mla_table(db, b, s, prefix, num_heads,
    /// kv, fmha, database_mode=EMPIRICAL)` on gb200/trtllm/1.3.0rc10
    /// (`get_database(..., shared_layer=False)`, matching this single-primary
    /// loader). Regenerate if the shipped table or the util math changes.
    #[test]
    fn context_mla_empirical_matches_python_oracles() {
        let mut db = gb200_trtllm_db();
        db.database_mode = DatabaseMode::Empirical;
        let cases: &[(u32, u32, u32, u32, f64)] = &[
            // off-grid seq
            (4, 5000, 0, 128, 3.5591050930761825),
            // prefix > 0: the query SOL carries prefix natively
            (2, 3000, 1024, 16, 0.17932261401789712),
            // exact collected hit: util reconstruction returns the measured value
            (4, 4096, 0, 128, 2.4523092905680337),
        ];
        for &(b, s, prefix, n, expected) in cases {
            let __r = query_context_mla_table(
                &db,
                b,
                s,
                prefix,
                n,
                KvCacheQuantMode::Bfloat16,
                FmhaQuantMode::Bfloat16,
            )
            .expect("empirical query");
            let (latency, source) = (__r.latency_ms, __r.source);
            assert_close(
                latency,
                expected,
                &format!("ctx_mla(b={b}, s={s}, pfx={prefix}, n={n})"),
            );
            assert_eq!(source, Source::Empirical);
        }
    }

    /// HYBRID on a slice with NO collected data (fmha=fp8 on gb200/trtllm,
    /// whose context MLA table is bfloat16-only) must surface the terminal
    /// EmpiricalNotImplemented miss (mirrors the Python contract).
    #[test]
    fn context_mla_hybrid_missing_slice_raises_empirical_not_implemented() {
        let mut db = gb200_trtllm_db();
        db.database_mode = DatabaseMode::Hybrid;
        let result = query_context_mla_table(
            &db,
            4,
            4096,
            0,
            128,
            KvCacheQuantMode::Bfloat16,
            FmhaQuantMode::Fp8,
        );
        assert!(
            matches!(result, Err(AicError::EmpiricalNotImplemented(_))),
            "got {result:?}"
        );
    }

    /// Python oracle: `GenerationMLA._query_generation_mla_table` in
    /// EMPIRICAL mode on gb200/trtllm/1.3.0rc10 (shared_layer=False).
    #[test]
    fn generation_mla_empirical_matches_python_oracles() {
        let mut db = gb200_trtllm_db();
        db.database_mode = DatabaseMode::Empirical;
        let cases: &[(u32, u32, u32, f64)] = &[
            // off-grid (b, s)
            (7, 9000, 128, 0.02734810076798607),
            // exact collected hit
            (1, 4096, 128, 0.02057066683967908),
        ];
        for &(b, s, n, expected) in cases {
            let __r = query_generation_mla_table(&db, b, s, n, KvCacheQuantMode::Bfloat16)
                .expect("empirical query");
            let (latency, source) = (__r.latency_ms, __r.source);
            assert_close(latency, expected, &format!("gen_mla(b={b}, s={s}, n={n})"));
            assert_eq!(source, Source::Empirical);
        }

        // HYBRID with a kv dtype that has no table (int8) -> terminal miss.
        db.database_mode = DatabaseMode::Hybrid;
        let result = query_generation_mla_table(&db, 1, 4096, 128, KvCacheQuantMode::Int8);
        assert!(
            matches!(result, Err(AicError::EmpiricalNotImplemented(_))),
            "got {result:?}"
        );
    }

    /// Python oracle: `MLABmm._query_mla_bmm_table` in EMPIRICAL mode on
    /// gb200/trtllm/1.3.0rc10 (shared_layer=False). The fp8 cases exercise
    /// the bfloat16 slice fallback (gb200's BMM table is bfloat16-only) with
    /// the SOL still bound to the REQUESTED fp8 quant.
    #[test]
    fn mla_bmm_empirical_matches_python_oracles() {
        let mut db = gb200_trtllm_db();
        db.database_mode = DatabaseMode::Empirical;
        let cases: &[(u32, u32, GemmQuantMode, bool, f64)] = &[
            // off-grid tokens on the requested bf16 slice
            (
                100,
                128,
                GemmQuantMode::Bfloat16,
                true,
                0.008883413307229573,
            ),
            // exact collected hit
            (
                256,
                128,
                GemmQuantMode::Bfloat16,
                true,
                0.010847999900579452,
            ),
            // fp8 requested -> bfloat16 fallback slice, fp8 SOL
            (20000, 128, GemmQuantMode::Fp8, true, 0.5326748099591996),
            // fallback on the post BMM at another head count
            (777, 64, GemmQuantMode::Fp8, false, 0.010838556565365292),
        ];
        for &(t, n, quant, is_pre, expected) in cases {
            let __r = query_mla_bmm_table(&db, t, n, quant, is_pre).expect("empirical query");
            let (latency, source) = (__r.latency_ms, __r.source);
            assert_close(
                latency,
                expected,
                &format!("mla_bmm(t={t}, n={n}, {quant:?}, pre={is_pre})"),
            );
            assert_eq!(source, Source::Empirical);
        }

        // HYBRID at a head count whose exact AND next-pow2 slices are both
        // absent (gb200 tops out at 128 heads) -> terminal miss.
        db.database_mode = DatabaseMode::Hybrid;
        let result = query_mla_bmm_table(&db, 64, 130, GemmQuantMode::Bfloat16, true);
        assert!(
            matches!(result, Err(AicError::EmpiricalNotImplemented(_))),
            "got {result:?}"
        );
    }

    /// Exact-head-first routing: a non-pow2 head count without exact rows
    /// (gb200's BMM table carries pow2 heads only) reroutes to the
    /// next-pow2 slice scaled linearly by the head ratio. Python twin:
    /// `tests/unit/sdk/operations/test_mla_bmm_head_routing.py`.
    #[test]
    fn mla_bmm_non_pow2_heads_reroute_to_next_pow2_slice() {
        let db = gb200_trtllm_db();
        let __r = query_mla_bmm_table(&db, 64, 7, GemmQuantMode::Bfloat16, true).expect("reroute");
        let (lat7, source) = (__r.latency_ms, __r.source);
        let __r = query_mla_bmm_table(&db, 64, 8, GemmQuantMode::Bfloat16, true).expect("exact");
        let lat8 = __r.latency_ms;
        assert_eq!(source, Source::Silicon);
        assert_close(lat7, lat8 * 7.0 / 8.0, "mla_bmm 7 -> 8-head slice reroute");
    }

    /// Python oracle: `MLAModule._query_context_mla_module_table` in
    /// EMPIRICAL mode on b200_sxm/vllm/0.19.0 (shared_layer=False).
    #[test]
    fn context_mla_module_empirical_matches_python_oracles() {
        let mut db = b200_vllm_db();
        db.database_mode = DatabaseMode::Empirical;
        type Case = (
            u32,
            u32,
            u32,
            u32,
            FmhaQuantMode,
            KvCacheQuantMode,
            GemmQuantMode,
            f64,
        );
        let cases: &[Case] = &[
            // off-grid seq, bf16^3 slice
            (
                2,
                5000,
                0,
                128,
                FmhaQuantMode::Bfloat16,
                KvCacheQuantMode::Bfloat16,
                GemmQuantMode::Bfloat16,
                5.030970266382403,
            ),
            // prefix > 0
            (
                1,
                2000,
                2048,
                16,
                FmhaQuantMode::Bfloat16,
                KvCacheQuantMode::Bfloat16,
                GemmQuantMode::Bfloat16,
                0.3328645307516498,
            ),
            // exact collected hit
            (
                1,
                1,
                0,
                128,
                FmhaQuantMode::Bfloat16,
                KvCacheQuantMode::Bfloat16,
                GemmQuantMode::Bfloat16,
                0.1351,
            ),
            // fp8 fmha/kv with fp8_block gemm slice
            (
                2,
                5000,
                0,
                128,
                FmhaQuantMode::Fp8,
                KvCacheQuantMode::Fp8,
                GemmQuantMode::Fp8Block,
                4.68971474347924,
            ),
        ];
        for &(b, s, prefix, n, fmha, kv, gemm, expected) in cases {
            let __r = query_context_mla_module_table(&db, b, s, prefix, n, kv, fmha, gemm, None)
                .expect("empirical query");
            let (latency, source) = (__r.latency_ms, __r.source);
            assert_close(
                latency,
                expected,
                &format!(
                    "ctx_mla_mod(b={b}, s={s}, pfx={prefix}, n={n}, {fmha:?}, {kv:?}, {gemm:?})"
                ),
            );
            assert_eq!(source, Source::Empirical);
        }

        // HYBRID with a gemm quant slice that has no data (fp8) -> miss.
        db.database_mode = DatabaseMode::Hybrid;
        let result = query_context_mla_module_table(
            &db,
            2,
            5000,
            0,
            128,
            KvCacheQuantMode::Bfloat16,
            FmhaQuantMode::Bfloat16,
            GemmQuantMode::Fp8,
            None,
        );
        assert!(
            matches!(result, Err(AicError::EmpiricalNotImplemented(_))),
            "got {result:?}"
        );
    }

    /// Python oracle: `MLAModule._query_generation_mla_module_table` in
    /// EMPIRICAL mode on b200_sxm/vllm/0.19.0 (shared_layer=False). The
    /// fp8/fp8_block case exercises the module SOL's dependence on the gemm
    /// quant (the BMM terms close over it).
    #[test]
    fn generation_mla_module_empirical_matches_python_oracles() {
        let mut db = b200_vllm_db();
        db.database_mode = DatabaseMode::Empirical;
        let cases: &[(u32, u32, u32, KvCacheQuantMode, GemmQuantMode, f64)] = &[
            (
                8,
                3000,
                128,
                KvCacheQuantMode::Bfloat16,
                GemmQuantMode::Bfloat16,
                0.16175364801657854,
            ),
            // exact collected hit (s = isl + step)
            (
                1,
                4097,
                128,
                KvCacheQuantMode::Bfloat16,
                GemmQuantMode::Bfloat16,
                0.1497,
            ),
            (
                8,
                3000,
                16,
                KvCacheQuantMode::Fp8,
                GemmQuantMode::Fp8Block,
                0.1146692546817411,
            ),
        ];
        for &(b, s, n, kv, gemm, expected) in cases {
            let __r = query_generation_mla_module_table(&db, b, s, n, kv, gemm, None)
                .expect("empirical query");
            let (latency, source) = (__r.latency_ms, __r.source);
            assert_close(
                latency,
                expected,
                &format!("gen_mla_mod(b={b}, s={s}, n={n}, {kv:?}, {gemm:?})"),
            );
            assert_eq!(source, Source::Empirical);
        }

        // HYBRID with a gemm quant slice that has no data (fp8) -> miss.
        db.database_mode = DatabaseMode::Hybrid;
        let result = query_generation_mla_module_table(
            &db,
            8,
            3000,
            128,
            KvCacheQuantMode::Bfloat16,
            GemmQuantMode::Fp8,
            None,
        );
        assert!(
            matches!(result, Err(AicError::EmpiricalNotImplemented(_))),
            "got {result:?}"
        );
    }

    /// SOL mode returns the pure roofline tagged `Source::Sol` for the MLA
    /// dispatches — context (prefix inside the formula, no prefix
    /// correction), generation (kv-implied flops) and BMM (dispatched BEFORE
    /// the head routing, exactly linear in the requested heads).
    #[test]
    fn mla_sol_mode_returns_roofline_with_sol_source() {
        let mut db = b200_vllm_db();
        db.database_mode = DatabaseMode::Sol;
        let spec = db.system_spec.clone();

        let ctx = ContextMlaOp::new(
            "ctx_mla",
            128,
            KvCacheQuantMode::Bfloat16,
            FmhaQuantMode::Bfloat16,
        );
        let result = ctx.query(&db, 2, 1024, 512).expect("ctx sol");
        let attn_flops = quant_tc_flops(&spec, FmhaQuantMode::Bfloat16.mapping()).unwrap();
        let expected = context_mla_sol_prefix_ms(
            &spec,
            KvCacheQuantMode::Bfloat16,
            128.0,
            1024.0,
            512.0,
            2.0,
            attn_flops,
        );
        assert_eq!(result.latency_ms, expected);
        assert_eq!(result.source, Source::Sol);
        assert_eq!(result.energy_wms, 0.0);

        let gen = GenerationMlaOp::new("gen_mla", 128, KvCacheQuantMode::Bfloat16);
        let result = gen.query(&db, 8, 1024).expect("gen sol");
        let gen_flops = generation_attn_flops(&spec, KvCacheQuantMode::Bfloat16).unwrap();
        let expected = generation_mla_sol_ms(
            &spec,
            KvCacheQuantMode::Bfloat16,
            128.0,
            8.0,
            1024.0,
            gen_flops,
        );
        assert_eq!(result.latency_ms, expected);
        assert_eq!(result.source, Source::Sol);

        // BMM with a non-pow2 head count: the SOL never routes to a slice
        // (head_scale must not apply) and never touches the table.
        let bmm = MlaBmmOp::new("bmm", 96, GemmQuantMode::Bfloat16, true);
        let result = bmm.query(&db, 64).expect("bmm sol");
        let bmm_flops = quant_tc_flops(&spec, GemmQuantMode::Bfloat16.mapping()).unwrap();
        let expected = mla_bmm_sol_ms(&spec, GemmQuantMode::Bfloat16, 96.0, 64.0, bmm_flops);
        assert_eq!(result.latency_ms, expected);
        assert_eq!(result.source, Source::Sol);
    }
}
