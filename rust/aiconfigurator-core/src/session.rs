// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Inference session driver.
//!
//! Mirrors `aiconfigurator.sdk.backends.base_backend`: iterates
//! `model.context_ops` / `model.generation_ops` to compute per-phase
//! latency, and composes the mix-step latency exactly the way Python's
//! `_get_mix_step_latency` does — one combined non-attention pass plus
//! per-phase attention. The FFI driver in `lib.rs` dispatches to these
//! methods based on which fields are populated in the FPM input.

use crate::common::enums::{
    BackendKind, CommQuantMode, FmhaQuantMode, GemmQuantMode, KvCacheQuantMode, MoeQuantMode,
};
use crate::common::error::AicError;
use crate::common::system_spec::SystemSpec;
use crate::models::base::{DtypeConfig, Model, ModelConfig, ModelSpec, ParallelConfig};
use crate::models::config_loader::HfModelConfig;
use crate::models::factory::build_model;
use crate::operators::{Op, RuntimeContext};
use crate::perf_database::PerfDatabase;
use crate::ForwardPassMetrics;

/// Configured estimator: model + perf database.
pub struct SessionEstimator {
    pub model: Model,
    pub db: PerfDatabase,
    pub backend: BackendKind,
}

impl std::fmt::Debug for SessionEstimator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionEstimator")
            .field("family", &self.model.config.spec.family)
            .field("architecture", &self.model.config.spec.architecture)
            .field("backend", &self.backend)
            .finish_non_exhaustive()
    }
}

impl SessionEstimator {
    pub fn build(
        hf: HfModelConfig,
        system_spec: SystemSpec,
        db: PerfDatabase,
        backend: BackendKind,
        parallel: ParallelConfig,
        dtypes: DtypeConfig,
    ) -> Result<Self, AicError> {
        Self::build_with_options(
            hf,
            system_spec,
            db,
            backend,
            parallel,
            dtypes,
            Default::default(),
            false,
            Default::default(),
        )
    }

    /// Build with WideEP routing knobs + MTP speculative decoding params.
    /// The standard `build` defaults all three to disabled, which keeps
    /// every existing call site on the standard model-builder path.
    #[allow(clippy::too_many_arguments)]
    pub fn build_with_options(
        hf: HfModelConfig,
        system_spec: SystemSpec,
        db: PerfDatabase,
        backend: BackendKind,
        parallel: ParallelConfig,
        dtypes: DtypeConfig,
        wideep_mode: crate::models::base::WideEpMode,
        enable_eplb: bool,
        mtp: crate::models::base::MtpConfig,
    ) -> Result<Self, AicError> {
        let spec: ModelSpec = hf.into();
        let config = ModelConfig {
            spec,
            parallel,
            dtypes,
            system_spec,
            backend,
            wideep_mode,
            enable_eplb,
            mtp,
        };
        let model = build_model(config)?;
        Ok(Self { model, db, backend })
    }

    /// Compute one forward-pass latency from a list of per-rank FPM entries.
    pub fn forward_pass_time_ms(
        &self,
        metrics_by_rank: &[ForwardPassMetrics],
    ) -> Result<f64, AicError> {
        if metrics_by_rank.is_empty() {
            return Err(AicError::InvalidForwardPassMetrics(
                "at least one attention-DP rank metric required".to_string(),
            ));
        }
        let mut max_latency = 0.0_f64;
        for metrics in metrics_by_rank {
            let rank_latency = self.rank_latency_ms(metrics)?;
            if rank_latency > max_latency {
                max_latency = rank_latency;
            }
        }
        Ok(max_latency)
    }

    fn rank_latency_ms(&self, metrics: &ForwardPassMetrics) -> Result<f64, AicError> {
        let sched = &metrics.scheduled_requests;
        let has_prefill = sched.num_prefill_requests > 0;
        let has_decode = sched.num_decode_requests > 0;

        if has_prefill && has_decode {
            // Mix step (continuous batching): compose like Python's
            // `_get_mix_step_latency`.
            //
            // The FPM construction in `estimate_mixed_step_latency_with_rust`
            // sets `sum_prefill_kv_tokens = prefix_per_req * num_prefill_requests`
            // which is exactly Python's `prefix * floor(ctx_tokens / isl)` value
            // used for the pass-1 combined non-attention call. We pass it
            // through unchanged.
            let n_prefill = sched.num_prefill_requests.max(1);
            let new_tokens_per_req = sched.sum_prefill_tokens / n_prefill;
            let prefix_per_req = sched.sum_prefill_kv_tokens / n_prefill;
            let n_decode = sched.num_decode_requests.max(1);
            let kv_per_req = sched.sum_decode_kv_tokens / n_decode;
            let ctx_tokens = sched.sum_prefill_tokens;
            let gen_tokens = sched.num_decode_requests;
            return self.get_mix_step_latency_ms(
                ctx_tokens,
                gen_tokens,
                new_tokens_per_req.max(1),
                prefix_per_req,
                sched.sum_prefill_kv_tokens,
                kv_per_req,
                n_decode,
            );
        }

        let mut total = 0.0_f64;

        if has_prefill {
            let n_prefill = sched.num_prefill_requests.max(1);
            let new_tokens_per_req = sched.sum_prefill_tokens / n_prefill;
            let prefix_per_req = sched.sum_prefill_kv_tokens / n_prefill;
            total += self.run_context_phase(n_prefill, new_tokens_per_req, prefix_per_req)?;
        }

        if has_decode {
            let n_decode = sched.num_decode_requests.max(1);
            let kv_per_req = sched.sum_decode_kv_tokens / n_decode;
            total += self.run_generation_phase(n_decode, kv_per_req)?;
        }

        Ok(total)
    }

    /// Python `_run_context_phase` 1:1: iterate context_ops, sum latencies.
    pub fn run_context_phase(
        &self,
        batch_size: u32,
        effective_isl: u32,
        prefix: u32,
    ) -> Result<f64, AicError> {
        let mut total = 0.0_f64;
        for op in &self.model.context_ops {
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
            total += op.query(&self.db, &ctx)?.latency_ms;
        }
        Ok(total)
    }

    /// Python `_run_generation_phase` for a single decode step (stride=osl).
    /// We compute one step's latency given the current decode position; the
    /// caller can multiply by osl to integrate over the decode trajectory.
    pub fn run_generation_phase(
        &self,
        batch_size: u32,
        kv_seq_tokens: u32,
    ) -> Result<f64, AicError> {
        let mut total = 0.0_f64;
        for op in &self.model.generation_ops {
            let x = if op.is_logits_gemm() {
                batch_size
            } else {
                batch_size
            };
            let ctx = RuntimeContext {
                batch_size,
                beam_width: 1,
                s: kv_seq_tokens,
                prefix: 0,
                num_tokens: x,
                seq_imbalance_correction_scale: 1.0,
                gen_seq_imbalance_correction_scale: 1.0,
                num_image_tokens: 0,
            };
            total += op.query(&self.db, &ctx)?.latency_ms;
        }
        Ok(total)
    }

    /// Python `_get_mix_step_latency` (Rust-shaped) — for the agg path's
    /// chunked-prefill + decode step.
    ///
    /// Algorithm (mirrors Python):
    /// 1. Combined non-attention pass: iterate `context_ops`, **skip**
    ///    context-attention ops, with `batch=1`, `isl=ctx_tokens+gen_tokens`,
    ///    `prefix=prefix_per_req * floor(ctx_tokens/isl_per_req)`.
    /// 2. Context attention contribution: re-run only context-attention
    ///    ops with the prefill batch shape (`batch = ceil(ctx_tokens/isl)`).
    /// 3. Decode attention contribution: iterate `generation_ops`, only
    ///    generation-attention ops, with the decode batch shape.
    pub fn get_mix_step_latency_ms(
        &self,
        ctx_tokens: u32,
        gen_tokens: u32,
        new_tokens_per_prefill_req: u32,
        prefix_per_req: u32,
        combined_prefix: u32,
        kv_per_decode_req: u32,
        decode_batch: u32,
    ) -> Result<f64, AicError> {
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
        let mut total = 0.0_f64;
        for op in &self.model.context_ops {
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
            total += op.query(&self.db, &ctx)?.latency_ms;
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
        // their full ISL through the FPM constructor if they need exact
        // parity with Python's chunked path.
        let isl_eff_pass2 = new_tokens_per_prefill_req.max(1);
        let ctx_attn_batch = ((ctx_tokens + isl_eff_pass2 - 1) / isl_eff_pass2).max(1);
        let scale_factor = ((isl_eff_pass2 + ctx_tokens - 1) / ctx_tokens.max(1)).max(1) as f64;
        for op in &self.model.context_ops {
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
            total += op.query(&self.db, &ctx)?.latency_ms / scale_factor;
        }

        // ---- Pass 3: generation attention with decode batch ----
        for op in &self.model.generation_ops {
            if !op.is_generation_attention() {
                continue;
            }
            let ctx = RuntimeContext {
                batch_size: decode_batch.max(1),
                beam_width: 1,
                s: kv_per_decode_req.max(1),
                prefix: 0,
                num_tokens: decode_batch.max(1),
                seq_imbalance_correction_scale: 1.0,
                gen_seq_imbalance_correction_scale: 1.0,
                num_image_tokens: 0,
            };
            total += op.query(&self.db, &ctx)?.latency_ms;
        }

        Ok(total)
    }

    pub fn clear_runtime_caches(&self) {
        // OnceLock-backed tables stay populated for the estimator's
        // lifetime by design; reconstructing the estimator clears them.
    }

    /// Per-op context-phase breakdown for debugging / parity drilldown.
    pub fn debug_context_breakdown(
        &self,
        batch_size: u32,
        effective_isl: u32,
        prefix: u32,
    ) -> Result<Vec<(String, f64)>, AicError> {
        let mut out = Vec::with_capacity(self.model.context_ops.len());
        for op in &self.model.context_ops {
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
            let r = op.query(&self.db, &ctx)?;
            out.push((op.name().to_string(), r.latency_ms));
        }
        Ok(out)
    }

    /// Per-op generation-phase breakdown for debugging / parity drilldown.
    pub fn debug_generation_breakdown(
        &self,
        batch_size: u32,
        kv_seq_tokens: u32,
    ) -> Result<Vec<(String, f64)>, AicError> {
        let mut out = Vec::with_capacity(self.model.generation_ops.len());
        for op in &self.model.generation_ops {
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
            let r = op.query(&self.db, &ctx)?;
            out.push((op.name().to_string(), r.latency_ms));
        }
        Ok(out)
    }
}

/// Map legacy `crate::DataType` (FFI input) to the new per-op-family
/// quant enums.
pub fn dtypes_from_legacy(
    weight: Option<&crate::DataType>,
    moe: Option<&crate::DataType>,
    activation: Option<&crate::DataType>,
    kv_cache: Option<&crate::DataType>,
) -> DtypeConfig {
    DtypeConfig {
        gemm_quant: weight
            .map(legacy_to_gemm_quant)
            .unwrap_or(GemmQuantMode::Bfloat16),
        moe_quant: moe.map(legacy_to_moe_quant).unwrap_or(MoeQuantMode::Bfloat16),
        fmha_quant: activation
            .map(legacy_to_fmha_quant)
            .unwrap_or(FmhaQuantMode::Bfloat16),
        kv_cache_quant: kv_cache
            .map(legacy_to_kv_quant)
            .unwrap_or(KvCacheQuantMode::Bfloat16),
        comm_quant: CommQuantMode::Half,
    }
}

fn legacy_to_gemm_quant(d: &crate::DataType) -> GemmQuantMode {
    use crate::DataType::*;
    match d {
        Bfloat16 | Float16 => GemmQuantMode::Bfloat16,
        Fp8 => GemmQuantMode::Fp8,
        Fp8Static => GemmQuantMode::Fp8Static,
        Fp8Block => GemmQuantMode::Fp8Block,
        Nvfp4 => GemmQuantMode::Nvfp4,
        Int8 => GemmQuantMode::Int8Wo,
        Int4 | W4afp8 | W4a16Mxfp4 | W4a8Mxfp4Mxfp8 => GemmQuantMode::Int4Wo,
    }
}

fn legacy_to_moe_quant(d: &crate::DataType) -> MoeQuantMode {
    use crate::DataType::*;
    match d {
        Bfloat16 | Float16 => MoeQuantMode::Bfloat16,
        Fp8 | Fp8Static => MoeQuantMode::Fp8,
        Fp8Block => MoeQuantMode::Fp8Block,
        Nvfp4 => MoeQuantMode::Nvfp4,
        Int4 | Int8 => MoeQuantMode::Int4Wo,
        W4afp8 => MoeQuantMode::W4afp8,
        W4a16Mxfp4 => MoeQuantMode::W4a16Mxfp4,
        W4a8Mxfp4Mxfp8 => MoeQuantMode::W4a8Mxfp4Mxfp8,
    }
}

fn legacy_to_fmha_quant(d: &crate::DataType) -> FmhaQuantMode {
    use crate::DataType::*;
    match d {
        Fp8 | Fp8Static | Nvfp4 => FmhaQuantMode::Fp8,
        Fp8Block => FmhaQuantMode::Fp8Block,
        _ => FmhaQuantMode::Bfloat16,
    }
}

fn legacy_to_kv_quant(d: &crate::DataType) -> KvCacheQuantMode {
    use crate::DataType::*;
    match d {
        Fp8 | Fp8Static | Fp8Block | Nvfp4 => KvCacheQuantMode::Fp8,
        Int8 => KvCacheQuantMode::Int8,
        _ => KvCacheQuantMode::Bfloat16,
    }
}

pub fn backend_from_legacy(legacy: &crate::BackendKind) -> BackendKind {
    use crate::BackendKind as L;
    match legacy {
        L::Trtllm => BackendKind::Trtllm,
        L::Sglang => BackendKind::Sglang,
        L::Vllm => BackendKind::Vllm,
    }
}

#[cfg(test)]
mod debug_tests {
    use super::*;
    use crate::common::system_spec::SystemSpec;
    use crate::models::config_loader;
    use std::path::PathBuf;

    fn make_estimator(model_path: &str, dtypes: DtypeConfig) -> SessionEstimator {
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let systems_root = manifest.join("../../src/aiconfigurator/systems");
        let model_configs_root = manifest.join("../../src/aiconfigurator/model_configs");
        let hf = config_loader::load(model_path, &model_configs_root).unwrap();
        let system_spec = SystemSpec::load(&systems_root.join("b200_sxm.yaml")).unwrap();
        let db = PerfDatabase::load(&systems_root, "b200_sxm", "vllm", "0.19.0").unwrap();
        let parallel = ParallelConfig {
            tp_size: 8,
            pp_size: 1,
            attention_dp_size: 1,
            moe_tp_size: 1,
            moe_ep_size: 8,
        };
        SessionEstimator::build(hf, system_spec, db, BackendKind::Vllm, parallel, dtypes).unwrap()
    }

    fn fp8_block_dtypes() -> DtypeConfig {
        DtypeConfig {
            gemm_quant: GemmQuantMode::Fp8Block,
            moe_quant: MoeQuantMode::Fp8Block,
            fmha_quant: FmhaQuantMode::Bfloat16,
            kv_cache_quant: KvCacheQuantMode::Fp8,
            comm_quant: CommQuantMode::Half,
        }
    }

    fn kimi_vllm_dtypes() -> DtypeConfig {
        DtypeConfig {
            gemm_quant: GemmQuantMode::Bfloat16,
            moe_quant: MoeQuantMode::Int4Wo,
            fmha_quant: FmhaQuantMode::Bfloat16,
            kv_cache_quant: KvCacheQuantMode::Bfloat16,
            comm_quant: CommQuantMode::Half,
        }
    }

    #[test]
    fn debug_minimax_context_breakdown() {
        let est = make_estimator("MiniMaxAI/MiniMax-M2.5", fp8_block_dtypes());
        let breakdown = est.debug_context_breakdown(1, 1024, 0).unwrap();
        println!("\n=== MiniMax static_ctx breakdown ===");
        let mut total = 0.0;
        for (name, lat) in &breakdown {
            println!("  {name}: {lat:.4}ms");
            total += lat;
        }
        println!("  TOTAL: {total:.4}ms");
    }

    #[test]
    fn debug_kimi_context_breakdown() {
        let est = make_estimator("moonshotai/Kimi-K2.5", kimi_vllm_dtypes());
        let breakdown = est.debug_context_breakdown(1, 1024, 0).unwrap();
        println!("\n=== Kimi static_ctx breakdown ===");
        let mut total = 0.0;
        for (name, lat) in &breakdown {
            println!("  {name}: {lat:.4}ms");
            total += lat;
        }
        println!("  TOTAL: {total:.4}ms");
    }

    #[test]
    fn debug_kimi_generation_breakdown() {
        let est = make_estimator("moonshotai/Kimi-K2.5", kimi_vllm_dtypes());
        let breakdown = est.debug_generation_breakdown(1, 1025).unwrap();
        println!("\n=== Kimi static_gen breakdown ===");
        let mut total = 0.0;
        for (name, lat) in &breakdown {
            println!("  {name}: {lat:.4}ms");
            total += lat;
        }
        println!("  TOTAL: {total:.4}ms");
    }
}
