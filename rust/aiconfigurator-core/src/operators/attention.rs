// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Attention operators: context, generation, encoder.
//!
//! Mirrors `aiconfigurator.sdk.operations.attention.{ContextAttention,
//! GenerationAttention, EncoderAttention}`. Each holds its config-time
//! attention shape (n, n_kv, head_size, window_size, quant modes) and
//! wraps the raw `AttentionTable` query with:
//!
//! - prefix correction `(full_s² − prefix²) / full_s²` for context paths
//! - fused-op extras for context: qk_norm (optional), apply_rope, kv_write
//!   via the analytic `mem_op` formula
//! - 1.1× correction factor on the extras (matches Python)
//! - `seq_imbalance_correction_scale` / `gen_seq_imbalance_correction_scale`
//!   multiplier for unbalanced sequence distributions
//! - `scale_factor` scaling at the end

use serde::{Deserialize, Serialize};
use crate::common::enums::{FmhaQuantMode, KvCacheQuantMode};
use crate::common::error::AicError;
use crate::common::system_spec::SystemSpec;
use crate::operators::base::{PerformanceResult, Source};
use crate::perf_database::PerfDatabase;

/// Analytic memory-op latency (ms). Matches Python's
/// `PerfDatabase.query_mem_op` empirical path (the only path used for
/// SILICON queries — there's no perf table for raw memory ops).
pub(crate) fn mem_op_latency_ms(spec: &SystemSpec, mem_bytes: f64) -> f64 {
    let mem_bw = spec.gpu.mem_bw.max(1.0);
    let scaling = spec.gpu.mem_bw_empirical_scaling_factor.max(1e-9);
    let constant = spec.gpu.mem_empirical_constant_latency;
    (mem_bytes / (mem_bw * scaling) + constant) * 1000.0
}

fn prefix_correction(full_s: u32, prefix: u32) -> f64 {
    if full_s == 0 {
        return 0.0;
    }
    let f = full_s as f64;
    let p = prefix as f64;
    (f * f - p * p) / (f * f)
}

// ---------------------------------------------------------------------------
// Context attention
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ContextAttentionOp {
    pub name: String,
    pub scale_factor: f64,
    pub n: u32,
    pub n_kv: u32,
    pub head_size: u32,
    pub window_size: u32,
    pub kv_cache_dtype: KvCacheQuantMode,
    pub fmha_quant_mode: FmhaQuantMode,
    pub use_qk_norm: bool,
    /// Context-parallel factor (Python's `_cp_size`, = `cp_size`). When `>1`,
    /// prefill FMHA is modeled as rank-0's two zigzag chunks:
    /// `ctx(c, prefix) + ctx(c, prefix + isl - c)` with `c = ceil(isl / 2cp)`.
    /// Defaults to 1 (no CP). The fused rope/kv_write/qk_norm extras are still
    /// added once, not per chunk.
    #[serde(default = "crate::operators::gemm::default_seq_split")]
    pub cp_size: u32,
}

impl ContextAttentionOp {
    pub fn new(
        name: impl Into<String>,
        n: u32,
        n_kv: u32,
        head_size: u32,
        kv_cache_dtype: KvCacheQuantMode,
        fmha_quant_mode: FmhaQuantMode,
    ) -> Self {
        Self {
            name: name.into(),
            scale_factor: 1.0,
            n,
            n_kv,
            head_size,
            window_size: 0,
            kv_cache_dtype,
            fmha_quant_mode,
            use_qk_norm: false,
            cp_size: 1,
        }
    }

    pub fn query(
        &self,
        db: &PerfDatabase,
        batch_size: u32,
        isl: u32,
        prefix: u32,
        seq_imbalance_correction_scale: f64,
    ) -> Result<PerformanceResult, AicError> {
        // Mirror Python's `ContextAttention._ctx(s, pfx)`: query the table at
        // the full sequence `s + pfx` and apply the prefix correction for
        // `pfx`. The Rust table is keyed by the full sequence length.
        let ctx = |s: u32, pfx: u32| -> Result<f64, AicError> {
            let full_s = s + pfx;
            let table = db.attention.query_context(
                batch_size,
                full_s,
                self.n,
                self.n_kv,
                self.head_size,
                self.window_size,
                self.kv_cache_dtype,
                self.fmha_quant_mode,
            )?;
            Ok(table * prefix_correction(full_s, pfx))
        };

        // Context parallelism (SGLang AllGather / zigzag): model rank 0's two
        // balanced chunks. c = ceil(isl / 2cp); rank 0 owns chunk 0 (prefix
        // unchanged) and chunk 2cp-1 (attends almost the full sequence). Only
        // the FMHA table term is split; the fused extras below are added once.
        let mut latency = if self.cp_size > 1 {
            let c = isl.div_ceil(2 * self.cp_size).max(1);
            ctx(c, prefix)? + ctx(c, prefix + isl - c)?
        } else {
            ctx(isl, prefix)?
        };

        // Fused-op extras (qk_norm optional, rope + kv_write mandatory).
        let q_num = (self.n * self.head_size) as f64;
        let k_num = (self.n_kv * self.head_size) as f64;
        let v_num = (self.n_kv * self.head_size) as f64;
        let spec = &db.system_spec;

        let mut extra = 0.0;
        if self.use_qk_norm {
            let qk_norm = 2.0 * mem_op_latency_ms(spec, q_num * 2.0)
                + 2.0 * mem_op_latency_ms(spec, k_num * 2.0);
            extra += qk_norm * 2.0; // elementwise before norm
        }
        let apply_rope = 2.0 * mem_op_latency_ms(spec, q_num * 2.0 + k_num * 2.0);
        let kv_write = mem_op_latency_ms(spec, k_num * self.fmha_quant_mode.mapping().memory)
            + mem_op_latency_ms(spec, v_num * self.fmha_quant_mode.mapping().memory);
        extra += apply_rope + kv_write;

        latency += extra * 1.1; // Python's correction factor for the fused extras.

        if seq_imbalance_correction_scale != 1.0 {
            latency *= seq_imbalance_correction_scale;
        }

        Ok(PerformanceResult::new(latency, Source::Silicon)
            .clamp_non_negative()
            .scaled(self.scale_factor))
    }
}

// ---------------------------------------------------------------------------
// Generation attention
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GenerationAttentionOp {
    pub name: String,
    pub scale_factor: f64,
    pub n: u32,
    pub n_kv: u32,
    pub head_size: u32,
    pub window_size: u32,
    pub kv_cache_dtype: KvCacheQuantMode,
}

impl GenerationAttentionOp {
    pub fn new(
        name: impl Into<String>,
        n: u32,
        n_kv: u32,
        head_size: u32,
        kv_cache_dtype: KvCacheQuantMode,
    ) -> Self {
        Self {
            name: name.into(),
            scale_factor: 1.0,
            n,
            n_kv,
            head_size,
            window_size: 0,
            kv_cache_dtype,
        }
    }

    pub fn query(
        &self,
        db: &PerfDatabase,
        batch_size: u32,
        kv_seq_tokens: u32,
        gen_seq_imbalance_correction_scale: f64,
    ) -> Result<PerformanceResult, AicError> {
        let table_latency = db.attention.query_generation(
            batch_size,
            kv_seq_tokens,
            self.n,
            self.n_kv,
            self.head_size,
            self.window_size,
            self.kv_cache_dtype,
        )?;
        let mut latency = table_latency;
        if gen_seq_imbalance_correction_scale != 1.0 {
            latency *= gen_seq_imbalance_correction_scale;
        }
        Ok(PerformanceResult::new(latency, Source::Silicon)
            .clamp_non_negative()
            .scaled(self.scale_factor))
    }
}

// ---------------------------------------------------------------------------
// Encoder attention (non-causal; vision encoder path)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EncoderAttentionOp {
    pub name: String,
    pub scale_factor: f64,
    pub n: u32,
    pub head_size: u32,
    pub fmha_quant_mode: FmhaQuantMode,
    /// Partial-RoPE fraction (Python `_partial_rotary_factor`): 1.0 = full
    /// rotation, 0.5 = half head_dim rotated (Qwen3-VL), 0.0 = no RoPE.
    /// Adds `factor * 2 * mem_op(Q+K bytes) * 1.1` on top of the table
    /// latency. Defaults to 0.0 for pre-field opspecs.
    #[serde(default)]
    pub partial_rotary_factor: f64,
}

impl EncoderAttentionOp {
    pub fn new(
        name: impl Into<String>,
        n: u32,
        head_size: u32,
        fmha_quant_mode: FmhaQuantMode,
    ) -> Self {
        Self {
            name: name.into(),
            scale_factor: 1.0,
            n,
            head_size,
            fmha_quant_mode,
            partial_rotary_factor: 0.0,
        }
    }

    pub fn query(
        &self,
        db: &PerfDatabase,
        batch_size: u32,
        s: u32,
    ) -> Result<PerformanceResult, AicError> {
        let mut latency = db.attention.query_encoder(
            batch_size,
            s,
            self.n,
            self.head_size,
            self.fmha_quant_mode,
        )?;
        // Partial RoPE extra (Python `EncoderAttention.query`,
        // operations/attention.py): Q + K bytes (bf16) over all tokens,
        // rotated fractionally, with the 1.1 correction factor.
        if self.partial_rotary_factor > 0.0 {
            let qk_num = (self.n as u64) * (self.head_size as u64); // MHA: q == k
            let qk_bytes = 2 * (qk_num * 2) * (batch_size as u64) * (s as u64);
            let apply_rope =
                self.partial_rotary_factor * 2.0 * mem_op_latency_ms(&db.system_spec, qk_bytes as f64);
            latency += apply_rope * 1.1;
        }
        Ok(PerformanceResult::new(latency, Source::Silicon)
            .clamp_non_negative()
            .scaled(self.scale_factor))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    const REPO_ROOT_HINT: &str = env!("CARGO_MANIFEST_DIR");

    fn b200_vllm_db() -> PerfDatabase {
        let systems_root = PathBuf::from(REPO_ROOT_HINT)
            .join("../..")
            .join("src/aiconfigurator/systems");
        PerfDatabase::load(&systems_root, "b200_sxm", "vllm", "0.19.0").expect("db must load")
    }

    #[test]
    fn context_attention_smoke() {
        let db = b200_vllm_db();
        let op = ContextAttentionOp::new(
            "ctx",
            64,
            1,
            128,
            KvCacheQuantMode::Fp8,
            FmhaQuantMode::Bfloat16,
        );
        // prefix=0 means prefix_correction=1.0, table latency is consumed full.
        let result = op
            .query(&db, 8, 16384, 0, 1.0)
            .expect("context attention query must succeed");
        // Table value at exact hit is 19.82; mem_op extras add ~0-1ms on top.
        assert!(result.latency_ms > 19.0 && result.latency_ms < 30.0);
        assert_eq!(result.source, Source::Silicon);
    }

    #[test]
    fn context_attention_prefix_correction_shrinks_latency() {
        let db = b200_vllm_db();
        let op = ContextAttentionOp::new(
            "ctx",
            64,
            1,
            128,
            KvCacheQuantMode::Fp8,
            FmhaQuantMode::Bfloat16,
        );
        // prefix=8192 -> prefix_correction = (16384^2 - 8192^2)/16384^2 = 0.75
        let with_prefix = op
            .query(&db, 8, 8192, 8192, 1.0)
            .expect("query must succeed")
            .latency_ms;
        let no_prefix = op.query(&db, 8, 16384, 0, 1.0).expect("query must succeed").latency_ms;
        assert!(
            with_prefix < no_prefix,
            "prefix correction must shrink latency: {with_prefix} vs {no_prefix}"
        );
    }

    #[test]
    fn generation_attention_smoke() {
        let db = b200_vllm_db();
        let op = GenerationAttentionOp::new("gen", 64, 4, 128, KvCacheQuantMode::Fp8);
        // b=32 isl+step=2 n=64 n_kv=4. The query averages 5 interp samples
        // over s ∈ [1, 2] (s_samples = [1,1,1,1,2]) on the densified grid,
        // matching Python's `_query_generation_attention_table`. Verified
        // against `PerfDatabase.query_generation_attention` (= 0.0086442669).
        let result = op
            .query(&db, 32, 2, 1.0)
            .expect("gen attention query must succeed");
        assert!(
            // Python v2 engine value (raw table, no densified lattice); the
            // pre-perf_interp expectation was 0.008644266923268636.
            (result.latency_ms - 0.008451361751014535).abs() < 1e-9,
            "expected 5-sample-averaged gen latency, got {}",
            result.latency_ms
        );
    }

    #[test]
    fn mem_op_latency_uses_empirical_formula() {
        let db = b200_vllm_db();
        // Formula check against the LIVE spec values (hardcoding mem_bw went
        // stale when PR #1246 corrected b200 to 7.7 TB/s):
        // latency = (bytes / (mem_bw * scaling) + constant) * 1000.
        let spec = &db.system_spec;
        let latency = mem_op_latency_ms(spec, 1_000_000.0);
        let expected = (1_000_000.0_f64
            / (spec.gpu.mem_bw * spec.gpu.mem_bw_empirical_scaling_factor)
            + spec.gpu.mem_empirical_constant_latency)
            * 1000.0;
        assert!((latency - expected).abs() < 1e-12);
    }
}
