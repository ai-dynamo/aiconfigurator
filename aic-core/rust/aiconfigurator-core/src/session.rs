// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-phase op-list execution primitives.
//!
//! Mirrors `aiconfigurator.sdk.backends.base_backend`: iterates a context /
//! generation op list to compute per-phase latency, and composes the mix-step
//! latency exactly the way Python's `_get_mix_step_latency` does — one combined
//! non-attention pass plus per-phase attention. The compiled
//! [`crate::engine::Engine`] drives these free functions over the precompiled
//! op lists from an [`crate::engine::spec::EngineSpec`].

use crate::common::error::AicError;
use crate::operators::{Op, RuntimeContext};
use crate::perf_database::PerfDatabase;

/// Component latencies for one mixed prefill/decode forward pass.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct MixedStepBreakdown {
    pub(crate) shared_non_attention_ms: f64,
    pub(crate) context_attention_ms: f64,
    pub(crate) decode_attention_ms: f64,
}

impl MixedStepBreakdown {
    pub(crate) fn total_ms(self) -> f64 {
        self.shared_non_attention_ms + self.context_attention_ms + self.decode_attention_ms
    }
}

/// Python `_run_context_phase` (`base_backend.py:144`) — one full pass over
/// the context op list. A free function so the compiled
/// [`crate::engine::Engine`] iterates one canonical body over its op list.
///
/// `effective_isl = isl - prefix` is the caller's responsibility (Python
/// computes it in `_run_context_phase` and `_run_static_breakdown`); this
/// fn takes the already-effective ISL and does NOT validate it (the Engine
/// caller performs Python's `effective_isl > 0` check before calling).
pub(crate) fn run_context_ops(
    ops: &[Op],
    db: &PerfDatabase,
    batch_size: u32,
    effective_isl: u32,
    prefix: u32,
) -> Result<f64, AicError> {
    let mut total = 0.0_f64;
    for op in ops {
        let x = if op.is_logits_gemm() {
            batch_size
        } else {
            batch_size * effective_isl
        };
        let ctx = RuntimeContext {
            batch_size,
            beam_width: 1,
            s: effective_isl,
            prefix,
            num_tokens: x,
            seq_imbalance_correction_scale: 1.0,
            gen_seq_imbalance_correction_scale: 1.0,
            num_image_tokens: 0,
        };
        total += op.query(db, &ctx)?.latency_ms;
    }
    Ok(total)
}

/// Python `_run_generation_phase` inner loop body for a single decode step
/// (`base_backend.py:185`). A free function shared by the compiled
/// [`crate::engine::Engine`]. Computes one step's latency at decode position
/// `kv_seq_tokens` (= Python `s = isl + i + 1`); the stride quadrature
/// (`for i in range(0, osl-1, stride)` × `repeat_count`) and the `(nextn + 1)`
/// decode-batch multiplier are applied by the caller (the Engine).
pub(crate) fn run_generation_ops_step(
    ops: &[Op],
    db: &PerfDatabase,
    batch_size: u32,
    kv_seq_tokens: u32,
) -> Result<f64, AicError> {
    let mut total = 0.0_f64;
    for op in ops {
        let ctx = RuntimeContext {
            batch_size,
            beam_width: 1,
            s: kv_seq_tokens,
            prefix: 0,
            num_tokens: batch_size,
            seq_imbalance_correction_scale: 1.0,
            gen_seq_imbalance_correction_scale: 1.0,
            num_image_tokens: 0,
        };
        total += op.query(db, &ctx)?.latency_ms;
    }
    Ok(total)
}

/// Python `_get_mix_step_latency` (Rust-shaped) — the three-pass mix-step
/// composition for the agg path's chunked-prefill + decode step. A free
/// function driven by the compiled [`crate::engine::Engine`]. The passes filter
/// differently from the full-pass loops — pass 1 skips `context_attention`,
/// pass 2 keeps only `context_attention`, pass 3 keeps only
/// `generation_attention` — so it does NOT route through [`run_context_ops`] /
/// [`run_generation_ops_step`].
///
/// Algorithm (mirrors Python):
/// 1. Combined non-attention pass: iterate `context_ops`, **skip**
///    context-attention ops, with `batch=1`, `isl=ctx_tokens+gen_tokens`,
///    `prefix=combined_prefix`.
/// 2. Context attention contribution: re-run only context-attention ops with
///    the prefill batch shape (`batch = ceil(ctx_tokens/isl)`).
/// 3. Decode attention contribution: iterate `generation_ops`, only
///    generation-attention ops, with the decode batch shape.
#[allow(clippy::too_many_arguments)]
pub(crate) fn get_mix_step_breakdown(
    context_ops: &[Op],
    generation_ops: &[Op],
    db: &PerfDatabase,
    ctx_tokens: u32,
    gen_tokens: u32,
    new_tokens_per_prefill_req: u32,
    prefix_per_req: u32,
    combined_prefix: u32,
    kv_per_decode_req: u32,
    decode_batch: u32,
) -> Result<MixedStepBreakdown, AicError> {
    // ---- Pass 1: combined non-attention work (batch=1, isl=ctx+gen) ----
    // Python: `run_static` is called with `isl = num_tokens_combined`
    // and `prefix = prefix * floor(ctx_tokens / isl)`, which makes
    // `effective_isl = isl - prefix = ctx_new + gen` (same as
    // `(ctx_tokens + gen_tokens)` here, since `ctx_tokens` is already
    // the NEW prefill count). The op queries get `s = effective_isl`
    // and **`prefix = combined_prefix`** — not zero.
    //
    // The non-zero prefix matters for attention ops that travel
    // through pass 1 because they live inside a composite Op (e.g.
    // DSv3's `FallbackOp("context_mla_block")` wraps `MlaModule`).
    // The pass-1 filter is `op.is_context_attention()`, which only
    // matches the name `"context_attention"`; an MLA module under a
    // different name is included in pass 1 and needs the prefix to
    // produce the same `(full_s, prefix_correction)` Python applies.
    // For non-attention ops (GEMM, MoE, etc.) the prefix field is
    // ignored, so threading it through is harmless.
    let effective_isl_combined = (ctx_tokens + gen_tokens).max(1);
    let mut shared_non_attention_ms = 0.0_f64;
    for op in context_ops {
        if op.is_context_attention() {
            continue;
        }
        let x = if op.is_logits_gemm() {
            1
        } else {
            effective_isl_combined
        };
        let ctx = RuntimeContext {
            batch_size: 1,
            beam_width: 1,
            s: effective_isl_combined,
            prefix: combined_prefix,
            num_tokens: x,
            seq_imbalance_correction_scale: 1.0,
            gen_seq_imbalance_correction_scale: 1.0,
            num_image_tokens: 0,
        };
        shared_non_attention_ms += op.query(db, &ctx)?.latency_ms;
    }

    // ---- Pass 2: context attention with prefill batch ----
    // Python's pass 2 simulates one prefill request's full attention:
    //   batch_size = ceil(ctx_tokens / isl)
    //   query context_attention with isl = full per-req new tokens,
    //   prefix = prefix_per_req, then divide by scale = ceil(isl/ctx_tokens).
    //
    // The FPM only carries this chunk's metrics. For the common
    // un-chunked path (chunk == one request's new tokens) we treat
    // `new_tokens_per_prefill_req` as the per-request ISL and the
    // ceil/scale cancel to 1; chunked-prefill callers should encode
    // their full ISL through the packing if they need exact parity
    // with Python's chunked path.
    let isl_eff_pass2 = new_tokens_per_prefill_req.max(1);
    let ctx_attn_batch = ((ctx_tokens + isl_eff_pass2 - 1) / isl_eff_pass2).max(1);
    let scale_factor = ((isl_eff_pass2 + ctx_tokens - 1) / ctx_tokens.max(1)).max(1) as f64;
    let mut context_attention_ms = 0.0_f64;
    for op in context_ops {
        if !op.is_context_attention() {
            continue;
        }
        let ctx = RuntimeContext {
            batch_size: ctx_attn_batch,
            beam_width: 1,
            s: isl_eff_pass2,
            prefix: prefix_per_req,
            num_tokens: ctx_tokens,
            seq_imbalance_correction_scale: 1.0,
            gen_seq_imbalance_correction_scale: 1.0,
            num_image_tokens: 0,
        };
        context_attention_ms += op.query(db, &ctx)?.latency_ms / scale_factor;
    }

    // ---- Pass 3: generation attention with decode batch ----
    // Mirror Python `_get_mix_step_latency`, which only adds the overlapped
    // decode-attention term `if gen_tokens > 0`. When this mixed step carries no
    // decode requests (e.g. the prefill-only first step that drives `ttft` at low
    // batch), Python contributes zero — so we must skip the pass entirely rather
    // than query at the `decode_batch.max(1)` floor, which would add a spurious
    // batch-1 generation_attention and inflate the step latency.
    let mut decode_attention_ms = 0.0_f64;
    if decode_batch > 0 {
        for op in generation_ops {
            if !op.is_generation_attention() {
                continue;
            }
            let ctx = RuntimeContext {
                batch_size: decode_batch,
                beam_width: 1,
                s: kv_per_decode_req.max(1),
                prefix: 0,
                num_tokens: decode_batch,
                seq_imbalance_correction_scale: 1.0,
                gen_seq_imbalance_correction_scale: 1.0,
                num_image_tokens: 0,
            };
            decode_attention_ms += op.query(db, &ctx)?.latency_ms;
        }
    }

    Ok(MixedStepBreakdown {
        shared_non_attention_ms,
        context_attention_ms,
        decode_attention_ms,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn get_mix_step_ops(
    context_ops: &[Op],
    generation_ops: &[Op],
    db: &PerfDatabase,
    ctx_tokens: u32,
    gen_tokens: u32,
    new_tokens_per_prefill_req: u32,
    prefix_per_req: u32,
    combined_prefix: u32,
    kv_per_decode_req: u32,
    decode_batch: u32,
) -> Result<f64, AicError> {
    Ok(get_mix_step_breakdown(
        context_ops,
        generation_ops,
        db,
        ctx_tokens,
        gen_tokens,
        new_tokens_per_prefill_req,
        prefix_per_req,
        combined_prefix,
        kv_per_decode_req,
        decode_batch,
    )?
    .total_ms())
}
