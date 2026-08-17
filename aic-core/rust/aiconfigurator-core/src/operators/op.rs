// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `Op` enum: unified typed dispatch for every operator family.
//!
//! Mirrors Python's `model.context_ops` / `model.generation_ops` /
//! `model.encoder_ops` lists — each entry is one typed op carrying its
//! config-time parameters. Session code iterates the list and calls
//! `op.query(db, runtime)` exactly the way Python's `_run_context_phase` /
//! `_run_generation_phase` iterate `for op in model.context_ops:
//! op.query(database, **runtime_kwargs)`.
//!
//! Module-level ops with separate context/generation queries (MLA module,
//! DSA, DSV4) get one variant per phase so a single `query` method handles
//! dispatch.

use serde::{Deserialize, Serialize};

use crate::common::error::AicError;
use crate::operators::{
    ContextAttentionOp, ContextMlaOp, CustomAllReduceOp, DsaModuleOp, Dsv4MegaMoeOp, Dsv4ModuleOp,
    ElementwiseOp, EmbeddingOp, EncoderAttentionOp, FpmForwardOp, GdnOp, GemmOp,
    GenerationAttentionOp, GenerationMlaOp, KdaOp, Mamba2Op, MhcModuleOp, MlaBmmOp, MlaModuleOp,
    MoEDispatchOp, MoeAllToAllOp, MoeExpertComputeOp, MoeOp, MsaModuleOp, NcclOp, P2POp,
    PerformanceResult, Source, VisionEncoderOp, WideEpContextMlaOp, WideEpGenerationMlaOp,
};
use crate::perf_database::PerfDatabase;

/// Runtime context passed to every op's `query`.
///
/// Mirrors Python's `**kwargs` payload to `op.query(database, ...)`. Each
/// op variant extracts the fields it needs; non-applicable fields are
/// safely ignored.
#[derive(Clone, Copy, Debug)]
pub struct RuntimeContext {
    /// Per-rank batch size for attention queries (context: prefill batch;
    /// generation: decode batch).
    pub batch_size: u32,
    /// Beam width (1 for static / agg / disagg; >1 for beam-search modes,
    /// which are not currently exercised by the engine-step path).
    pub beam_width: u32,
    /// Sequence length passed to attention queries. For context phase this
    /// is `effective_isl = isl - prefix`. For generation phase this is the
    /// current `isl + step + 1` decode position.
    pub s: u32,
    /// Prefix length already in KV cache (context phase only; 0 otherwise).
    pub prefix: u32,
    /// Total per-rank token count for compute-bound ops (GEMM, Embedding,
    /// Elementwise, MoE, MoE dispatch, comm). Python passes this as `x` to
    /// `op.query`. For context: `batch_size * effective_isl`. For
    /// generation: `batch_size * beam_width`.
    pub num_tokens: u32,
    /// Sequence-imbalance correction multiplier for context attention.
    pub seq_imbalance_correction_scale: f64,
    /// Sequence-imbalance correction multiplier for generation attention.
    pub gen_seq_imbalance_correction_scale: f64,
    /// Number of vision-encoder tokens per image (encoder phase only).
    pub num_image_tokens: u32,
}

impl Default for RuntimeContext {
    fn default() -> Self {
        Self {
            batch_size: 1,
            beam_width: 1,
            s: 1,
            prefix: 0,
            num_tokens: 1,
            seq_imbalance_correction_scale: 1.0,
            gen_seq_imbalance_correction_scale: 1.0,
            num_image_tokens: 0,
        }
    }
}

/// Typed operator. One variant per Python `operations` family.
///
/// Module-level ops with separate context/generation queries become
/// distinct variants so dispatch is unambiguous.
///
/// Serializes as the wire-format op for [`crate::engine::spec::EngineSpec`]
/// (re-exported there as `OpSpec`). All config-time fields are plain
/// serializable data, so the enum and its recursive `Overlap`/`Fallback`
/// children round-trip through bincode.
///
/// `Op::Vision` is part of the shared session path and derives serde with
/// the rest, but it is **never emitted into a compiled `EngineSpec`**:
/// `compile_engine` decomposes the vision encoder into its child
/// `Gemm`/`EncoderAttention`/`Elementwise` ops instead.
/// Production specs therefore never contain a `Vision` variant.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Op {
    Gemm(GemmOp),
    Embedding(EmbeddingOp),
    Elementwise(ElementwiseOp),
    ContextAttention(ContextAttentionOp),
    GenerationAttention(GenerationAttentionOp),
    EncoderAttention(EncoderAttentionOp),
    ContextMla(ContextMlaOp),
    GenerationMla(GenerationMlaOp),
    MlaModuleContext(MlaModuleOp),
    MlaModuleGeneration(MlaModuleOp),
    MlaBmm(MlaBmmOp),
    Moe(MoeOp),
    MoeDispatch(MoEDispatchOp),
    CustomAllReduce(CustomAllReduceOp),
    Nccl(NcclOp),
    P2P(P2POp),
    Vision(VisionEncoderOp),
    DsaContext(DsaModuleOp),
    DsaGeneration(DsaModuleOp),
    /// MiniMax Sparse Attention (MSA) context module — no silicon data;
    /// answers only under HYBRID/EMPIRICAL via cross-op DSA util transfer.
    MsaContext(MsaModuleOp),
    /// MSA generation module (`s` = total KV length).
    MsaGeneration(MsaModuleOp),
    Dsv4Context(Dsv4ModuleOp),
    Dsv4Generation(Dsv4ModuleOp),
    Mhc(MhcModuleOp),
    Mamba2(Mamba2Op),
    Gdn(GdnOp),
    /// SGLang WideEP context MLA — replaces `ContextMlaOp` in the
    /// `WideEPDeepSeekModel` variant. SGLang-only perf data.
    WideEpContextMla(WideEpContextMlaOp),
    /// SGLang WideEP generation MLA — replaces `GenerationMlaOp` in the
    /// `WideEPDeepSeekModel` variant.
    WideEpGenerationMla(WideEpGenerationMlaOp),
    /// Two op groups that execute in parallel on different CUDA streams.
    /// Mirrors Python `aiconfigurator.sdk.operations.overlap.OverlapOp`:
    /// `latency = max(sum(group_a), sum(group_b))`.
    Overlap(OverlapOp),
    /// Try a primary op; on perf-DB miss, fall back to summing a list of
    /// granular ops. Mirrors Python
    /// `aiconfigurator.sdk.operations.overlap.FallbackOp`: supports the
    /// transitional state where some systems have module-level profiling
    /// data and others still ship per-kernel granular data.
    Fallback(FallbackOp),
    /// SGLang DeepSeek-V4 MegaMoE routed module (Python
    /// `DeepSeekV4MegaMoEModule`): one variant for both phases — the op's
    /// `is_context` field selects the phase inside the unified table.
    /// Measured-SILICON-only; see `operators/dsv4.rs::Dsv4MegaMoeOp`.
    ///
    /// APPENDED after `Fallback` on purpose: bincode enum indices are
    /// positional, so appending does not shift existing variants and
    /// `ENGINE_SPEC_SCHEMA_VERSION` stays unchanged. Do NOT insert new
    /// variants mid-enum.
    Dsv4MegaMoe(Dsv4MegaMoeOp),
    /// Kimi Delta Attention (KDA) kernel for Kimi-K3 linear_attention
    /// layers — Python `KDAKernel` (a `GDNKernel` subclass with a distinct
    /// `kda_perf` table, an fp32-state SOL byte model, a "verify" phase and
    /// a `draft_tokens` field). APPENDED at the end (see the bincode note on
    /// `Dsv4MegaMoe`); the new serialized variant bumped
    /// `ENGINE_SPEC_SCHEMA_VERSION` to 5 (renumbered to 6 at its merge).
    Kda(KdaOp),
    /// Whole-model forward pass (Python `forward_model="fpm"`): with the FPM
    /// rewrite each phase op list is exactly one of these, answering from the
    /// collected `fpm_forward_perf` cells instead of a granular composition.
    /// NOT related to the `crate::fpm` (ForwardPassPerfModel) module.
    /// APPENDED at the end (see the bincode note on `Dsv4MegaMoe`); claimed
    /// `ENGINE_SPEC_SCHEMA_VERSION` 5 concurrently with #1460/#1435 and was
    /// renumbered to 9 across the intervening wire-format landings.
    FpmForward(FpmForwardOp),
    /// Unified large-EP MoE all-to-all comm phase (Python
    /// `operations.moe_comm.MoEAllToAll`) — one variant serves every backend
    /// and every phase; the op's `phase` / `comm_backend` fields select the
    /// slice. Measured-SILICON-only; see `operators/moe_a2a.rs`.
    ///
    /// APPENDED after `FpmForward` — same positional-index rule as above.
    MoeAllToAll(MoeAllToAllOp),
    /// Unified large-EP MoE expert compute (Python
    /// `operations.moe_comm.MoEExpertCompute`) — one variant for both inference phases;
    /// the op's `inference_phase` field selects the slice.
    /// Measured-SILICON-only; see `operators/moe_expert_compute.rs`.
    MoeExpertCompute(MoeExpertComputeOp),
}

/// Inline-defined here (rather than a sibling module under `operators/`)
/// because the variants of an overlap group are themselves `Op` values —
/// the definition is cyclic with `Op` and the implementation is small.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OverlapOp {
    pub name: String,
    pub group_a: Vec<Op>,
    pub group_b: Vec<Op>,
}

impl OverlapOp {
    pub fn new(name: impl Into<String>, group_a: Vec<Op>, group_b: Vec<Op>) -> Self {
        Self {
            name: name.into(),
            group_a,
            group_b,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FallbackOp {
    pub name: String,
    /// Try this first. On `AicError::PerfDatabase`-class failure (missing
    /// file / missing data point), the fallback chain is used instead.
    pub primary: Box<Op>,
    pub fallback: Vec<Op>,
}

impl FallbackOp {
    pub fn new(name: impl Into<String>, primary: Op, fallback: Vec<Op>) -> Self {
        Self {
            name: name.into(),
            primary: Box::new(primary),
            fallback,
        }
    }
}

impl Op {
    /// Stable op name (Python `op._name`). Used by session code to filter
    /// (e.g. context-attention exclusion in mix-step composition) and for
    /// debugging.
    pub fn name(&self) -> &str {
        match self {
            Op::Gemm(o) => &o.name,
            Op::Embedding(o) => &o.name,
            Op::Elementwise(o) => &o.name,
            Op::ContextAttention(o) => &o.name,
            Op::GenerationAttention(o) => &o.name,
            Op::EncoderAttention(o) => &o.name,
            Op::ContextMla(o) => &o.name,
            Op::GenerationMla(o) => &o.name,
            Op::MlaModuleContext(o) => &o.name,
            Op::MlaModuleGeneration(o) => &o.name,
            Op::MlaBmm(o) => &o.name,
            Op::Moe(o) => &o.name,
            Op::MoeDispatch(o) => &o.name,
            Op::CustomAllReduce(o) => &o.name,
            Op::Nccl(o) => &o.name,
            Op::P2P(o) => &o.name,
            Op::Vision(o) => &o.name,
            Op::DsaContext(o) => &o.name,
            Op::DsaGeneration(o) => &o.name,
            Op::MsaContext(o) => &o.name,
            Op::MsaGeneration(o) => &o.name,
            Op::Dsv4Context(o) => &o.name,
            Op::Dsv4Generation(o) => &o.name,
            Op::Mhc(o) => &o.name,
            Op::Mamba2(o) => &o.name,
            Op::Gdn(o) => &o.name,
            Op::WideEpContextMla(o) => &o.name,
            Op::WideEpGenerationMla(o) => &o.name,
            Op::FpmForward(o) => &o.name,
            Op::Overlap(o) => &o.name,
            Op::Fallback(o) => &o.name,
            Op::Dsv4MegaMoe(o) => &o.name,
            Op::Kda(o) => &o.name,
            Op::MoeAllToAll(o) => &o.name,
            Op::MoeExpertCompute(o) => &o.name,
        }
    }

    /// True if this op's name matches Python's mix-step filter for the
    /// context-attention bucket. Python uses literal string equality on
    /// `"context_attention"` — that's the LLAMA / MOE attention op name.
    /// Models with module-level attention (e.g. Kimi's
    /// `context_mla_module`) have names that don't match this filter, so
    /// they're treated as non-attention in the mix-step composition
    /// (matching Python's intent: the module already represents the full
    /// fused attention+projection work and shouldn't be re-decomposed).
    pub fn is_context_attention(&self) -> bool {
        self.name() == "context_attention"
    }

    /// True if this op's name matches Python's mix-step filter for the
    /// generation-attention bucket (`"generation_attention"`).
    pub fn is_generation_attention(&self) -> bool {
        self.name() == "generation_attention"
    }

    /// Identifies the logits projection GEMM by name. Python special-cases
    /// `logits_gemm` in `_run_context_phase` to use `x=batch_size` instead
    /// of `x=batch_size * effective_isl`.
    pub fn is_logits_gemm(&self) -> bool {
        matches!(self, Op::Gemm(_)) && self.name().contains("logits_gemm")
    }

    /// Query this op with the given runtime. Returns the scaled latency
    /// from the underlying op's `query` method.
    pub fn query(
        &self,
        db: &PerfDatabase,
        ctx: &RuntimeContext,
    ) -> Result<PerformanceResult, AicError> {
        match self {
            Op::Gemm(op) => op.query(db, ctx.num_tokens, None),
            Op::Embedding(op) => op.query(db, ctx.num_tokens),
            Op::Elementwise(op) => op.query(db, ctx.num_tokens),
            Op::ContextAttention(op) => op.query(
                db,
                ctx.batch_size,
                ctx.s,
                ctx.prefix,
                ctx.seq_imbalance_correction_scale,
            ),
            Op::GenerationAttention(op) => op.query(
                db,
                ctx.batch_size,
                ctx.s,
                ctx.gen_seq_imbalance_correction_scale,
            ),
            Op::EncoderAttention(op) => op.query(db, ctx.batch_size, ctx.s),
            Op::ContextMla(op) => op.query(db, ctx.batch_size, ctx.s, ctx.prefix),
            Op::GenerationMla(op) => op.query(db, ctx.batch_size, ctx.s),
            Op::MlaModuleContext(op) => op.query_context(db, ctx.batch_size, ctx.s, ctx.prefix),
            Op::MlaModuleGeneration(op) => op.query_generation(db, ctx.batch_size, ctx.s),
            // Python's `MLABmm.query` uses `batch_size` as the BMM table's
            // tokens-axis index (the table's `num_tokens` column equals the
            // op's per-request count, which is `batch_size`). Pass
            // `ctx.batch_size`, not `ctx.num_tokens`.
            Op::MlaBmm(op) => op.query(db, ctx.batch_size),
            Op::Moe(op) => op.query(db, ctx.num_tokens),
            Op::MoeDispatch(op) => op.query(db, ctx.num_tokens),
            Op::CustomAllReduce(op) => op.query(db, ctx.num_tokens),
            Op::Nccl(op) => op.query(db, ctx.num_tokens),
            Op::P2P(op) => op.query(db, ctx.num_tokens),
            Op::Vision(op) => op.query(db, ctx.num_image_tokens),
            Op::DsaContext(op) => op.query_context(db, ctx.batch_size, ctx.s, ctx.prefix),
            Op::DsaGeneration(op) => op.query_generation(db, ctx.batch_size, ctx.s),
            Op::MsaContext(op) => op.query_context(db, ctx.batch_size, ctx.s, ctx.prefix),
            Op::MsaGeneration(op) => op.query_generation(db, ctx.batch_size, ctx.s),
            Op::Dsv4Context(op) => op.query_context(db, ctx.batch_size, ctx.s, ctx.prefix),
            Op::Dsv4Generation(op) => op.query_generation(db, ctx.batch_size, ctx.s),
            Op::Mhc(op) => op.query(db, ctx.num_tokens),
            Op::Mamba2(op) => op.query(db, ctx.batch_size, ctx.s),
            Op::Gdn(op) => op.query(db, ctx.batch_size, ctx.s),
            Op::WideEpContextMla(op) => op.query(db, ctx.batch_size, ctx.s, ctx.prefix),
            Op::WideEpGenerationMla(op) => op.query(db, ctx.batch_size, ctx.s),
            // Whole-model op: consumes batch_size/s/prefix/beam_width from the
            // context (num_tokens is ignored, mirroring Python's kwargs use).
            Op::FpmForward(op) => op.query(db, ctx),
            Op::Overlap(op) => {
                // Mirrors Python `OverlapOp.query`: each group accumulates
                // through `PerformanceResult` addition from a zero/empirical
                // seed (`total_a = PerformanceResult(0.0, energy=0.0,
                // source="empirical"); total_a += ...`), then latency =
                // max(group_a_total, group_b_total) while ENERGY = group_a +
                // group_b (both groups consume power even though they overlap
                // in time). The source is `(total_a + total_b).source` — the
                // seeds and `plus`'s zero-identity rule make zero-valued
                // members (empty groups, nested empty composites, zero-cost
                // legs) source-NEUTRAL instead of poisoning the tag to Mixed.
                let mut total_a = PerformanceResult::new(0.0, Source::Empirical);
                for inner in &op.group_a {
                    total_a = total_a.plus(inner.query(db, ctx)?);
                }
                let mut total_b = PerformanceResult::new(0.0, Source::Empirical);
                for inner in &op.group_b {
                    total_b = total_b.plus(inner.query(db, ctx)?);
                }
                let merged = total_a.plus(total_b);
                Ok(PerformanceResult::with_energy(
                    total_a.latency_ms.max(total_b.latency_ms),
                    total_a.energy_wms + total_b.energy_wms,
                    merged.source,
                )
                .clamp_non_negative())
            }
            Op::Fallback(op) => {
                // Mirrors Python `FallbackOp.query`: try the primary; on
                // perf-DB-class failure, sum the fallback chain instead.
                // (Python additionally caches `primary_unavailable=True` to
                // skip subsequent retries — we don't bother here because the
                // hot-path penalty is one `OnceLock::get` per call on a
                // populated path and one retry on a missing one.)
                //
                // Under HYBRID the primary is evaluated against a SILICON
                // view (Python swaps in `_get_configured_database_view(db,
                // SILICON, transfer_policy)`): a missing module table must
                // fall to the granular fallback chain, not be hybrid-
                // estimated at module level. The fallback ops then run
                // against the ORIGINAL (hybrid) database.
                let silicon_db;
                let primary_db: &PerfDatabase =
                    if db.database_mode == crate::common::enums::DatabaseMode::Hybrid {
                        silicon_db = db.silicon_view();
                        &silicon_db
                    } else {
                        db
                    };
                match op.primary.query(primary_db, ctx) {
                    // Primary result passes through verbatim — its energy
                    // rides along (Python returns `self._primary.query(...)`).
                    Ok(r) => Ok(r),
                    Err(AicError::PerfDatabase(_)) | Err(AicError::Io { .. }) => {
                        // Fallback chain: Python sums PerformanceResults
                        // from a zero/empirical seed (`total =
                        // PerformanceResult(0.0, energy=0.0,
                        // source="empirical"); total += op.query(...)`), so
                        // latency AND energy both accumulate and an empty (or
                        // all-zero) chain keeps the empirical seed tag via
                        // `plus`'s zero-identity rule.
                        let mut total = PerformanceResult::new(0.0, Source::Empirical);
                        for inner in &op.fallback {
                            total = total.plus(inner.query(db, ctx)?);
                        }
                        // `with_energy` (sol: None) keeps the pre-existing
                        // composed-result behavior: only the SOURCE semantics
                        // change in this fix, not the SOL decomposition.
                        Ok(PerformanceResult::with_energy(
                            total.latency_ms,
                            total.energy_wms,
                            total.source,
                        )
                        .clamp_non_negative())
                    }
                    Err(other) => Err(other),
                }
            }
            // Rank-LOCAL token count, like Moe/MoeDispatch (Python passes the
            // same `x`); the megamoe table is indexed by local-rank tokens
            // and the op must NOT re-multiply by attention_dp_size.
            Op::Dsv4MegaMoe(op) => op.query(db, ctx.num_tokens),
            // Like Gdn: the op derives its phase coordinates internally
            // (verify divides the (nextn+1)-scaled batch by draft_tokens).
            Op::Kda(op) => op.query(db, ctx.batch_size, ctx.s),
            // Both large-EP ops take Python's `x` (moe_comm.py:657, :1291) —
            // the same per-rank token count every other compute/comm op gets.
            // The per-op token rescaling (`// attention_tp_size` for the comm
            // side, `* attention_dp_size` for the compute side) happens INSIDE
            // each `query`, exactly where Python does it.
            Op::MoeAllToAll(op) => op.query(db, ctx.num_tokens),
            Op::MoeExpertCompute(op) => op.query(db, ctx.num_tokens),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::enums::GemmQuantMode;
    use crate::operators::gemm::GemmOp;
    use crate::perf_database::energy_test_fixtures::{write_parquet, Col};
    use crate::perf_database::PerfDatabase;

    /// Minimal systems root with ONE bf16 GEMM row: an exact-hit silicon
    /// leaf at `num_tokens=128`, and a guaranteed typed data miss for any
    /// fp8 query (no fp8 table exists).
    fn one_row_gemm_db() -> (tempfile::TempDir, PerfDatabase) {
        let tmp = tempfile::tempdir().expect("tmpdir");
        let data = crate::perf_database::energy_test_fixtures::write_energy_systems_root(tmp.path());
        write_parquet(
            &data.join("gemm_perf.parquet"),
            &[
                Col::Str("gemm_dtype", vec!["bfloat16"]),
                Col::I64("m", vec![128]),
                Col::I64("n", vec![1024]),
                Col::I64("k", vec![1024]),
                Col::F64("latency", vec![1.0]),
                Col::F64("power", vec![100.0]),
            ],
        );
        let db = PerfDatabase::load(tmp.path(), "testsys", "vllm", "1.0").expect("db must load");
        (tmp, db)
    }

    fn silicon_leaf() -> Op {
        Op::Gemm(GemmOp::new("hit", 1024, 1024, GemmQuantMode::Bfloat16))
    }

    fn missing_leaf() -> Op {
        Op::Gemm(GemmOp::new("miss", 1024, 1024, GemmQuantMode::Fp8))
    }

    fn ctx() -> RuntimeContext {
        RuntimeContext {
            num_tokens: 128,
            ..RuntimeContext::default()
        }
    }

    fn empty_overlap(name: &str) -> Op {
        Op::Overlap(OverlapOp::new(name, vec![], vec![]))
    }

    // Zero-valued composite provenance oracle (review #1552 round 4): the
    // legacy Python accumulators start from `PerformanceResult(0.0,
    // energy=0.0, source="empirical")` and `__add__` treats a (0.0, 0.0)
    // operand as a source-neutral identity, so zero-valued members must
    // never poison a composite's tag to Mixed and empty composition must
    // report `empirical`.

    #[test]
    fn empty_overlap_source_is_empirical() {
        let (_tmp, db) = one_row_gemm_db();
        let r = empty_overlap("e").query(&db, &ctx()).expect("query");
        assert_eq!(r.latency_ms, 0.0);
        assert_eq!(r.energy_wms, 0.0);
        assert_eq!(r.source, Source::Empirical);
    }

    #[test]
    fn half_empty_overlap_keeps_leaf_source() {
        let (_tmp, db) = one_row_gemm_db();
        let op = Op::Overlap(OverlapOp::new("half", vec![silicon_leaf()], vec![]));
        let r = op.query(&db, &ctx()).expect("query");
        assert!((r.latency_ms - 1.0).abs() < 1e-12);
        assert_eq!(r.source, Source::Silicon);
    }

    #[test]
    fn nested_zero_overlap_is_source_neutral_same_group() {
        let (_tmp, db) = one_row_gemm_db();
        let op = Op::Overlap(OverlapOp::new(
            "same",
            vec![silicon_leaf(), empty_overlap("nested")],
            vec![],
        ));
        let r = op.query(&db, &ctx()).expect("query");
        assert!((r.latency_ms - 1.0).abs() < 1e-12);
        assert_eq!(r.source, Source::Silicon, "zero-valued nested composite must be source-neutral");
    }

    #[test]
    fn nested_zero_overlap_is_source_neutral_opposite_group() {
        let (_tmp, db) = one_row_gemm_db();
        let op = Op::Overlap(OverlapOp::new(
            "opp",
            vec![silicon_leaf()],
            vec![empty_overlap("nested")],
        ));
        let r = op.query(&db, &ctx()).expect("query");
        assert!((r.latency_ms - 1.0).abs() < 1e-12);
        assert_eq!(r.source, Source::Silicon, "zero-valued opposite group must be source-neutral");
    }

    #[test]
    fn failed_primary_empty_fallback_is_empirical() {
        let (_tmp, db) = one_row_gemm_db();
        let op = Op::Fallback(FallbackOp::new("fb", missing_leaf(), vec![]));
        let r = op.query(&db, &ctx()).expect("query");
        assert_eq!(r.latency_ms, 0.0);
        assert_eq!(r.energy_wms, 0.0);
        assert_eq!(r.source, Source::Empirical);
    }

    #[test]
    fn failed_primary_fallback_chain_keeps_leaf_source() {
        let (_tmp, db) = one_row_gemm_db();
        let op = Op::Fallback(FallbackOp::new("fb", missing_leaf(), vec![silicon_leaf()]));
        let r = op.query(&db, &ctx()).expect("query");
        assert!((r.latency_ms - 1.0).abs() < 1e-12);
        assert_eq!(r.source, Source::Silicon);
    }
}

