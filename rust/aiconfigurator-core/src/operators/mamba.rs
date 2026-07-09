// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Mamba2 and Gated Delta Network (GDN) operators for hybrid state-space
//! models (Nemotron-H et al.).
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
