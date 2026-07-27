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
    ContextAttentionOp, ContextMlaOp, CustomAllReduceOp, DsaModuleOp, Dsv4MegaMoeOp,
    Dsv4ModuleOp,
    ElementwiseOp, EmbeddingOp, EncoderAttentionOp, GdnOp, GemmOp, GenerationAttentionOp,
    GenerationMlaOp, Mamba2Op, MhcModuleOp, MlaBmmOp, MlaModuleOp, MoEDispatchOp, MoeOp,
    MsaModuleOp, TrtllmWideEpMoEDispatchOp,
    NcclOp, P2POp, PerformanceResult, Source, VisionEncoderOp, WideEpContextMlaOp,
    WideEpGenerationMlaOp, WideEpMoeOp,
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
    /// TensorRT-LLM WideEP MoE compute. Used by the
    /// `TrtllmWideEPDeepSeekModel` variant; dispatch / combine cost is
    /// modeled separately by `WideEpMoeDispatch`.
    WideEpMoe(WideEpMoeOp),
    /// TensorRT-LLM WideEP All2All dispatch (prepare+dispatch / combine).
    /// Mirrors Python `TrtLLMWideEPMoEDispatch` — a direct `Operation`
    /// subclass, NOT a `MoEDispatch` flavor.
    WideEpMoeDispatch(TrtllmWideEpMoEDispatchOp),
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
            Op::WideEpMoe(o) => &o.name,
            Op::WideEpMoeDispatch(o) => &o.name,
            Op::Overlap(o) => &o.name,
            Op::Fallback(o) => &o.name,
            Op::Dsv4MegaMoe(o) => &o.name,
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
            Op::WideEpMoe(op) => op.query(db, ctx.num_tokens),
            Op::WideEpMoeDispatch(op) => op.query(db, ctx.num_tokens),
            Op::Overlap(op) => {
                // Mirrors Python `OverlapOp.query`: each group is summed
                // independently, then `max(group_a_total, group_b_total)` is
                // returned. Source tag follows the additive combine rule.
                let mut total_a = 0.0_f64;
                let mut source_a: Option<Source> = None;
                for inner in &op.group_a {
                    let r = inner.query(db, ctx)?;
                    total_a += r.latency_ms;
                    source_a = Some(match source_a {
                        None => r.source,
                        Some(prev) => prev.combine(r.source),
                    });
                }
                let mut total_b = 0.0_f64;
                let mut source_b: Option<Source> = None;
                for inner in &op.group_b {
                    let r = inner.query(db, ctx)?;
                    total_b += r.latency_ms;
                    source_b = Some(match source_b {
                        None => r.source,
                        Some(prev) => prev.combine(r.source),
                    });
                }
                let source = match (source_a, source_b) {
                    (Some(a), Some(b)) => a.combine(b),
                    (Some(s), None) | (None, Some(s)) => s,
                    (None, None) => Source::Silicon,
                };
                Ok(PerformanceResult::new(total_a.max(total_b), source).clamp_non_negative())
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
                    Ok(r) => Ok(r),
                    Err(AicError::PerfDatabase(_)) | Err(AicError::Io { .. }) => {
                        let mut total = 0.0_f64;
                        let mut source: Option<Source> = None;
                        for inner in &op.fallback {
                            let r = inner.query(db, ctx)?;
                            total += r.latency_ms;
                            source = Some(match source {
                                None => r.source,
                                Some(prev) => prev.combine(r.source),
                            });
                        }
                        Ok(PerformanceResult::new(total, source.unwrap_or(Source::Silicon))
                            .clamp_non_negative())
                    }
                    Err(other) => Err(other),
                }
            }
            // Rank-LOCAL token count, like Moe/MoeDispatch (Python passes the
            // same `x`); the megamoe table is indexed by local-rank tokens
            // and the op must NOT re-multiply by attention_dp_size.
            Op::Dsv4MegaMoe(op) => op.query(db, ctx.num_tokens),
        }
    }
}

