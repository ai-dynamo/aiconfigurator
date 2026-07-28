// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Mamba2, Gated Delta Network (GDN) and Kimi Delta Attention (KDA)
//! operators for hybrid state-space models (Nemotron-H, Qwen3.5, Kimi-K3).
//!
//! Each wraps `db.state_space.query_*` with `scale_factor` + `clamp`.

use serde::{Deserialize, Serialize};
use crate::common::error::AicError;
use crate::operators::base::{PerformanceResult, Source};
use crate::perf_database::PerfDatabase;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Mamba2Op {
    pub name: String,
    pub scale_factor: f64,
    /// Kernel routine name. Distinguishes per-phase Mamba2 variants
    /// (`causal_conv1d_fn` for context, `causal_conv1d_update` for
    /// generation, etc.).
    pub kernel_source: String,
    pub phase: String, // "context" | "generation" (matches Python; SOL branch keys on phase == "context")
    pub d_model: u32,
    pub d_state: u32,
    pub d_conv: u32,
    pub nheads: u32,
    pub head_dim: u32,
    pub n_groups: u32,
    pub chunk_size: u32,
}

impl Mamba2Op {
    pub fn query(
        &self,
        db: &PerfDatabase,
        batch_size: u32,
        seq_len: u32,
    ) -> Result<PerformanceResult, AicError> {
        // Mirrors Python `Mamba2Kernel.query`: silicon-first, SOL fallback
        // on perf-DB miss. The op's arg-style SOL is threaded into the table
        // query so the perf_interp engine can util-hold beyond-range queries
        // (mirroring how Python passes `get_sol` into the engine record).
        match db.state_space.query_mamba2(
            &self.kernel_source,
            &self.phase,
            batch_size,
            seq_len,
            self.d_model,
            self.d_state,
            self.d_conv,
            self.nheads,
            self.head_dim,
            self.n_groups,
            self.chunk_size,
            &|b, s| self.sol_latency_ms(db, b, s),
        ) {
            Ok(latency) => Ok(PerformanceResult::new(latency, Source::Silicon)
                .clamp_non_negative()
                .scaled(self.scale_factor)),
            Err(AicError::PerfDatabase(_)) => {
                let latency = self.sol_latency_ms(db, batch_size as f64, seq_len as f64);
                Ok(PerformanceResult::new(latency, Source::Sol)
                    .clamp_non_negative()
                    .scaled(self.scale_factor))
            }
            Err(other) => Err(other),
        }
    }

    /// Mirrors Python `Mamba2Kernel.get_sol()`. `f64` coordinates because the
    /// perf_interp engine evaluates SOL at blended/snapped anchor points.
    fn sol_latency_ms(&self, db: &PerfDatabase, batch_size: f64, seq_len: f64) -> f64 {
        let nheads = self.nheads as f64;
        let head_dim = self.head_dim as f64;
        let n_groups = self.n_groups as f64;
        let d_state = self.d_state as f64;
        let d_conv = self.d_conv as f64;
        let d_inner = nheads * head_dim;
        let conv_dim = d_inner + 2.0 * n_groups * d_state;
        let x = if self.phase == "context" && seq_len > 0.0 {
            batch_size * seq_len
        } else {
            batch_size
        };
        let total_bytes = match self.kernel_source.as_str() {
            "causal_conv1d_fn" | "causal_conv1d_update" => {
                x * conv_dim * (d_conv + 1.0) * 2.0 + x * conv_dim * 2.0
            }
            _ => {
                // SSM kernels (`mamba_chunk_scan_combined` etc.).
                x * (d_inner + n_groups * d_state * 2.0 + nheads) * 2.0 + x * d_inner * 2.0
            }
        };
        let mem_bw = db.system_spec.gpu.mem_bw.max(1.0);
        total_bytes / mem_bw * 1000.0
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GdnOp {
    pub name: String,
    pub scale_factor: f64,
    /// GDN kernel name. Context uses `causal_conv1d_fn` and
    /// `chunk_gated_delta_rule`; generation uses `causal_conv1d_update`
    /// and `fused_sigmoid_gating_delta_rule_update`.
    pub kernel_source: String,
    pub phase: String, // "context" | "generation" (matches Python; SOL branch keys on phase == "context")
    pub d_model: u32,
    pub d_conv: u32,
    pub num_k_heads: u32,
    pub head_k_dim: u32,
    pub num_v_heads: u32,
    pub head_v_dim: u32,
}

impl GdnOp {
    pub fn query(
        &self,
        db: &PerfDatabase,
        batch_size: u32,
        seq_len: u32,
    ) -> Result<PerformanceResult, AicError> {
        // Mirrors Python `GDNKernel.query`: try the silicon table; on a
        // `PerfDataNotAvailableError`-class miss (the perf DB doesn't ship
        // every kernel/phase slice), fall back to a per-kernel SOL formula.
        // The op's arg-style SOL is threaded into the table query for
        // beyond-range util-hold (mirrors Python's engine record sol_fn).
        match db.state_space.query_gdn(
            &self.kernel_source,
            &self.phase,
            batch_size,
            seq_len,
            self.d_model,
            self.d_conv,
            self.num_k_heads,
            self.head_k_dim,
            self.num_v_heads,
            self.head_v_dim,
            &|b, s| self.sol_latency_ms(db, b, s),
        ) {
            Ok(latency) => Ok(PerformanceResult::new(latency, Source::Silicon)
                .clamp_non_negative()
                .scaled(self.scale_factor)),
            Err(AicError::PerfDatabase(_)) => {
                let latency = self.sol_latency_ms(db, batch_size as f64, seq_len as f64);
                Ok(PerformanceResult::new(latency, Source::Sol)
                    .clamp_non_negative()
                    .scaled(self.scale_factor))
            }
            Err(other) => Err(other),
        }
    }

    /// Mirrors Python `GDNKernel.get_sol()`. Per-kernel byte-count formula
    /// divided by GPU memory bandwidth. `f64` coordinates because the
    /// perf_interp engine evaluates SOL at blended/snapped anchor points.
    fn sol_latency_ms(&self, db: &PerfDatabase, batch_size: f64, seq_len: f64) -> f64 {
        let x = if self.phase == "context" && seq_len > 0.0 {
            batch_size * seq_len
        } else {
            batch_size
        };
        let bs = batch_size;
        let nk = self.num_k_heads as f64;
        let hk = self.head_k_dim as f64;
        let nv = self.num_v_heads as f64;
        let hv = self.head_v_dim as f64;
        let conv_channels = nk * hk + nv * hv;
        let d_conv = self.d_conv as f64;
        let d_model = self.d_model as f64;
        let state_size = nv * hk * hv;
        let chunk_size = 64.0_f64;
        // Python: `(s // chunk_size) if s else 0` — floor division.
        let num_chunks = if seq_len > 0.0 {
            (seq_len / 64.0).floor()
        } else {
            0.0
        };
        let h_chunks_bytes = num_chunks * state_size * 2.0 * bs;
        let _ = chunk_size; // reserved for clarity / future use

        let (read_bytes, write_bytes) = match self.kernel_source.as_str() {
            "causal_conv1d_fn" | "causal_conv1d_update" => (
                x * conv_channels * (d_conv + 1.0) * 2.0,
                x * conv_channels * 2.0,
            ),
            "chunk_gated_delta_rule" => (
                x * (nk * hk + nv * hv) * 2.0 + state_size * 2.0 * bs + h_chunks_bytes,
                x * nv * hv * 2.0 + state_size * 2.0 * bs + h_chunks_bytes,
            ),
            "fused_sigmoid_gating_delta_rule_update" => (
                x * (nk * hk + nv * hv) * 2.0 + state_size * 2.0 * bs,
                x * nv * hv * 2.0 + state_size * 2.0 * bs,
            ),
            _ => (x * d_model * 2.0, x * d_model * 2.0),
        };
        let mem_bw = db.system_spec.gpu.mem_bw.max(1.0);
        (read_bytes + write_bytes) / mem_bw * 1000.0
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct KdaOp {
    pub name: String,
    pub scale_factor: f64,
    /// KDA kernel name. Context uses `causal_conv1d_fn_qkv3` and
    /// `chunk_kda`; generation uses `causal_conv1d_update` and
    /// `fused_recurrent_kda_packed_decode`; verify uses
    /// `causal_conv1d_update` and `fused_sigmoid_gating_delta_rule_update`.
    pub kernel_source: String,
    /// "context" | "generation" | "verify" (matches Python; the SOL byte
    /// model and the verify batch division key on this phase).
    pub phase: String,
    pub d_model: u32,
    pub d_conv: u32,
    pub num_k_heads: u32,
    pub head_k_dim: u32,
    pub num_v_heads: u32,
    pub head_v_dim: u32,
    /// Verify-phase token width (speculative block size + 1). Python sends
    /// `op._draft_tokens`; 0 outside verify.
    pub draft_tokens: i64,
}

impl KdaOp {
    pub fn query(
        &self,
        db: &PerfDatabase,
        batch_size: u32,
        seq_len: u32,
    ) -> Result<PerformanceResult, AicError> {
        // Mirrors Python `KDAKernel.query`: silicon-first, SOL fallback on a
        // perf-DB miss, with the op's arg-style SOL threaded into the table
        // query for beyond-range util-hold (same shape as GdnOp above).
        // Verify batching is normalized ONCE here — both the table lookup and
        // the SOL fallback see the adjusted `(batch, seq)` coordinates.
        let (batch_size, seq_len) = self.effective_coords(batch_size, seq_len);
        match db.state_space.query_kda(
            &self.kernel_source,
            &self.phase,
            batch_size,
            seq_len,
            self.d_model,
            self.d_conv,
            self.num_k_heads,
            self.head_k_dim,
            self.num_v_heads,
            self.head_v_dim,
            &|b, s| self.sol_latency_ms(db, b, s),
        ) {
            Ok(latency) => Ok(PerformanceResult::new(latency, Source::Silicon)
                .clamp_non_negative()
                .scaled(self.scale_factor)),
            Err(AicError::PerfDatabase(_)) => {
                let latency = self.sol_latency_ms(db, batch_size as f64, seq_len as f64);
                Ok(PerformanceResult::new(latency, Source::Sol)
                    .clamp_non_negative()
                    .scaled(self.scale_factor))
            }
            Err(other) => Err(other),
        }
    }

    /// Phase-normalized `(batch, seq)` for the table lookup and the SOL
    /// fallback. Mirrors Python `KDAKernel.query`:
    /// - context: pass-through.
    /// - verify: `seq_len := draft_tokens`; the backend scales the generation
    ///   batch by `(nextn + 1)` to model the verify token width, and the
    ///   verify table is keyed by `(requests, draft_tokens)`, so divide the
    ///   scaled batch back down: `batch := max(1, round(batch / draft))`.
    /// - generation: `seq := 0` (Python passes `seq_len=None`; generation
    ///   SOL formulas treat it as absent).
    fn effective_coords(&self, batch_size: u32, seq_len: u32) -> (u32, u32) {
        match self.phase.as_str() {
            "context" => (batch_size, seq_len),
            "verify" => {
                let draft = self.draft_tokens.clamp(0, u32::MAX as i64) as u32;
                let batch = if self.draft_tokens > 0 {
                    // Python `round()` is banker's rounding — keep ties-to-even.
                    (batch_size as f64 / self.draft_tokens as f64)
                        .round_ties_even()
                        .max(1.0) as u32
                } else {
                    batch_size
                };
                (batch, draft)
            }
            _ => (batch_size, 0),
        }
    }

    /// Mirrors Python `KDAKernel._query_kda_table::get_sol`. `f64`
    /// coordinates because the perf_interp engine evaluates SOL at
    /// blended/snapped anchor points.
    fn sol_latency_ms(&self, db: &PerfDatabase, batch_size: f64, seq_len: f64) -> f64 {
        let mem_bw = db.system_spec.gpu.mem_bw.max(1.0);
        self.sol_total_bytes(batch_size, seq_len) / mem_bw * 1000.0
    }

    /// Memory-bound byte model (read + write). Pure so the formulas are unit
    /// testable without a `PerfDatabase`.
    fn sol_total_bytes(&self, batch_size: f64, seq_len: f64) -> f64 {
        let b = batch_size;
        let s = seq_len;
        let x = if (self.phase == "context" || self.phase == "verify") && s > 0.0 {
            b * s
        } else {
            b
        };
        let proj_size = self.num_v_heads as f64 * self.head_v_dim as f64;
        // fp32 delta-rule state: [num_v_heads, head_k_dim, head_v_dim].
        let state_bytes =
            self.num_v_heads as f64 * self.head_k_dim as f64 * self.head_v_dim as f64 * 4.0;
        let d_conv = self.d_conv as f64;
        let d_model = self.d_model as f64;
        // SOL-only aliasing of the vLLM physical kernels onto the canonical
        // byte models (Python `_query_kda_table`; the TABLE lookup keeps the
        // physical name — no alias map on load, unlike GDN).
        let kernel = match self.kernel_source.as_str() {
            // vLLM prefill cores: same chunked-scan byte model as chunk_kda.
            "chunk_kda_with_fused_gate" | "flashkda_fwd" => "chunk_kda",
            // vLLM fused decode = conv update + recurrence + gated norm.
            "fused_kda_decode" => "fused_recurrent_kda_packed_decode",
            // vLLM chain-verify kernel: same traffic as the sglang verify op.
            "fused_recurrent_kda" => "fused_sigmoid_gating_delta_rule_update",
            other => other,
        };
        let (read_bytes, write_bytes) = match kernel {
            // Three sequential convs, each over proj_size channels.
            "causal_conv1d_fn_qkv3" => (
                3.0 * (x * proj_size * (d_conv + 1.0) * 2.0),
                3.0 * (x * proj_size * 2.0),
            ),
            "causal_conv1d_update" => {
                let conv_channels = 3.0 * proj_size;
                (
                    x * conv_channels * (d_conv + 1.0) * 2.0,
                    x * conv_channels * 2.0,
                )
            }
            "chunk_kda" => {
                // Chunked delta-rule scan; per-chunk states h [B, NT, H, K, V]
                // round-trip through global memory (fp32 state).
                // Python: `(s // chunk_size) if s else 0` — floor division.
                let num_chunks = if s > 0.0 { (s / 64.0).floor() } else { 0.0 };
                let h_chunks_bytes = num_chunks * state_bytes * b;
                // q/k/v plus the per-K gate g are all x * proj_size wide.
                (
                    x * 4.0 * proj_size * 2.0 + state_bytes * b + h_chunks_bytes,
                    x * proj_size * 2.0 + state_bytes * b + h_chunks_bytes,
                )
            }
            "fused_recurrent_kda_packed_decode" => (
                x * (3.0 * proj_size + proj_size) * 2.0 + state_bytes * b,
                x * proj_size * 2.0 + state_bytes * b,
            ),
            // Verify: reads committed state once per request, writes one
            // intermediate fp32 state per draft token.
            "fused_sigmoid_gating_delta_rule_update" => (
                x * 4.0 * proj_size * 2.0 + state_bytes * b,
                x * proj_size * 2.0 + state_bytes * x,
            ),
            _ => (x * d_model * 2.0, x * d_model * 2.0),
        };
        read_bytes + write_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kda_op(kernel_source: &str, phase: &str, draft_tokens: i64) -> KdaOp {
        KdaOp {
            name: "kda".into(),
            scale_factor: 1.0,
            kernel_source: kernel_source.into(),
            phase: phase.into(),
            d_model: 4096,
            d_conv: 4,
            num_k_heads: 16,
            head_k_dim: 128,
            num_v_heads: 16,
            head_v_dim: 128,
            draft_tokens,
        }
    }

    #[test]
    fn kda_verify_divides_scaled_batch_by_draft_tokens() {
        let op = kda_op("fused_sigmoid_gating_delta_rule_update", "verify", 4);
        // The backend scales the generation batch by (nextn + 1) = draft.
        assert_eq!(op.effective_coords(16, 1), (4, 4));
        // Python `max(1, round(...))` floor.
        assert_eq!(op.effective_coords(1, 1), (1, 4));
        // Python round() is banker's rounding: 10/4 = 2.5 -> 2, not 3.
        assert_eq!(op.effective_coords(10, 1), (2, 4));
        // draft_tokens == 0: batch untouched, seq collapses to 0 (SOL x = b).
        let op0 = kda_op("fused_sigmoid_gating_delta_rule_update", "verify", 0);
        assert_eq!(op0.effective_coords(16, 1), (16, 0));
        // Generation drops the runtime seq (Python passes seq_len=None).
        let gen = kda_op("fused_recurrent_kda_packed_decode", "generation", 0);
        assert_eq!(gen.effective_coords(8, 4096), (8, 0));
    }

    #[test]
    fn kda_sol_byte_model_matches_python_formulas() {
        // Shape: proj_size = 16*128 = 2048; state_bytes = 16*128*128*4 = 1048576.
        let proj = 2048.0;
        let state = 1_048_576.0;

        // Verify recurrence at (b=2, s=4): x = 8.
        // read = 8*4*2048*2 + state*2; write = 8*2048*2 + state*8.
        let op = kda_op("fused_sigmoid_gating_delta_rule_update", "verify", 4);
        let x = 8.0;
        let expected =
            (x * 4.0 * proj * 2.0 + state * 2.0) + (x * proj * 2.0 + state * x);
        assert_eq!(op.sol_total_bytes(2.0, 4.0), expected);

        // vLLM prefill core aliases onto the chunk_kda byte model
        // (b=1, s=128 -> x=128, num_chunks=2).
        let op = kda_op("flashkda_fwd", "context", 0);
        let x = 128.0;
        let h_chunks = 2.0 * state * 1.0;
        let expected = (x * 4.0 * proj * 2.0 + state + h_chunks)
            + (x * proj * 2.0 + state + h_chunks);
        assert_eq!(op.sol_total_bytes(1.0, 128.0), expected);

        // Packed conv update (generation, x = b): 3P channels.
        let op = kda_op("causal_conv1d_update", "generation", 0);
        let ch = 3.0 * proj;
        let expected = 8.0 * ch * 5.0 * 2.0 + 8.0 * ch * 2.0;
        assert_eq!(op.sol_total_bytes(8.0, 0.0), expected);

        // qkv3 context conv: three sequential convs over proj channels.
        let op = kda_op("causal_conv1d_fn_qkv3", "context", 0);
        let x = 2.0 * 64.0;
        let expected = 3.0 * (x * proj * 5.0 * 2.0) + 3.0 * (x * proj * 2.0);
        assert_eq!(op.sol_total_bytes(2.0, 64.0), expected);

        // Unknown kernel: d_model round-trip.
        let op = kda_op("mystery_kernel", "generation", 0);
        assert_eq!(op.sol_total_bytes(4.0, 0.0), 4.0 * 4096.0 * 2.0 * 2.0);
    }
}
