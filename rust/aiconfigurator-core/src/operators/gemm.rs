// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! GEMM operator.
//!
//! Mirrors `aiconfigurator.sdk.operations.gemm.GEMM`. Config-time fields
//! (`n`, `k`, quant mode, scale_factor, scale_num_tokens) are set once when
//! the model graph is built; the query takes only `x` (the M dimension /
//! number of tokens) and routes through the GEMM perf table.
//!
//! For `fp8_static` quant mode, subtracts `compute_scale` overhead from the
//! GEMM latency (and additionally subtracts `scale_matrix` when the input
//! is also low-precision). Post-subtraction latency is floored at the GEMM's
//! own SOL roofline — mirroring Python's `max(latency_floor, latency)` where
//! `latency_floor = query_gemm(..., DatabaseMode.SOL)` — not at 0.

use serde::{Deserialize, Serialize};
use crate::common::enums::GemmQuantMode;
use crate::common::error::AicError;
use crate::operators::base::{PerformanceResult, Source};
use crate::perf_database::PerfDatabase;

/// GEMM operation: a dense matrix multiply of shape `M=x, N=n, K=k`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GemmOp {
    pub name: String,
    pub scale_factor: f64,
    pub n: u32,
    pub k: u32,
    pub quant_mode: GemmQuantMode,
    pub scale_num_tokens: u32,
    pub low_precision_input: bool,
    /// Context-parallel sequence-shard factor (Python's `_seq_split`, = `cp_size`).
    /// The per-rank token count is `ceil(m / seq_split)`. Defaults to 1 (no CP).
    #[serde(default = "default_seq_split")]
    pub seq_split: u32,
}

pub(crate) fn default_seq_split() -> u32 {
    1
}

impl GemmOp {
    /// Convenience constructor for the most common case (no token scaling,
    /// standard input precision).
    pub fn new(name: impl Into<String>, n: u32, k: u32, quant_mode: GemmQuantMode) -> Self {
        Self {
            name: name.into(),
            scale_factor: 1.0,
            n,
            k,
            quant_mode,
            scale_num_tokens: 1,
            low_precision_input: false,
            seq_split: 1,
        }
    }

    /// Query GEMM latency for the given `x` (M / number of tokens) and an
    /// optional quant-mode override.
    ///
    /// Mirrors Python's `GEMM.query`:
    /// 1. `m = x // scale_num_tokens`
    /// 2. Query GEMM table at `(m, n, k, quant_mode)`.
    /// 3. For `fp8_static`: subtract `compute_scale(m, k)` (and
    ///    `scale_matrix(m, k)` when low-precision-input).
    /// 4. Clamp to `>= 0`, scale by `scale_factor`.
    pub fn query(
        &self,
        db: &PerfDatabase,
        x: u32,
        quant_override: Option<GemmQuantMode>,
    ) -> Result<PerformanceResult, AicError> {
        let m = x / self.scale_num_tokens.max(1);
        // CP: per-rank token count = ceil(m / seq_split) (busiest rank).
        let m = m.div_ceil(self.seq_split.max(1));
        let quant = quant_override.unwrap_or(self.quant_mode);

        let mut latency = db.gemm.query(quant, m, self.n, self.k)?;
        let mut source = Source::Silicon;
        let mut latency_floor = 0.0_f64;

        if quant == GemmQuantMode::Fp8Static {
            let cs_latency = db.gemm.query_compute_scale(quant, m, self.k)?;
            latency -= cs_latency;

            if self.low_precision_input {
                let sm_latency = db.gemm.query_scale_matrix(quant, m, self.k)?;
                latency -= sm_latency;
            }
            // Python (`operations/gemm.py`): the subtraction leaves a path
            // that still contains the GEMM; independently interpolated
            // component tables can cross, but that path cannot be faster than
            // the GEMM's own roofline. Floor at the SOL (NOT at 0), and tag
            // the result "estimated" (fp8_static is modeled from dynamic FP8
            // plus overhead tables, not measured directly).
            latency_floor = crate::perf_database::gemm::gemm_sol_latency_ms(
                &db.system_spec,
                quant,
                m as f64,
                self.n as f64,
                self.k as f64,
            );
            source = Source::Estimated;
        }

        Ok(PerformanceResult::new(latency.max(latency_floor), source)
            .clamp_non_negative()
            .scaled(self.scale_factor))
    }

    /// Per-tensor weight count in bytes, matching Python's
    /// `GEMM.get_weights()`: `n * k * quant_mode.value.memory *
    /// scale_factor`. The `scale_factor` multiplier mirrors the way model
    /// builders count weights once at construction and let the op replicate
    /// per layer (e.g. `scale_factor = num_hidden_layers` for body GEMMs).
    pub fn weights_bytes(&self) -> f64 {
        (self.n as f64) * (self.k as f64) * self.quant_mode.mapping().memory * self.scale_factor
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
    fn gemm_op_query_exact_hit_matches_table() {
        let db = b200_vllm_db();
        // bf16 GEMM at m=32768 n=65536 k=16384 -> latency=41.5967
        let op = GemmOp::new("test", 65536, 16384, GemmQuantMode::Bfloat16);
        let result = op.query(&db, 32768, None).expect("query must succeed");
        assert!(
            (result.latency_ms - 41.59673055013021).abs() < 1e-9,
            "expected recorded latency, got {}",
            result.latency_ms
        );
        assert_eq!(result.source, Source::Silicon);
    }

    #[test]
    fn gemm_op_scale_factor_multiplies_latency() {
        let db = b200_vllm_db();
        let op = GemmOp {
            name: "scaled".to_string(),
            scale_factor: 0.5,
            n: 65536,
            k: 16384,
            quant_mode: GemmQuantMode::Bfloat16,
            scale_num_tokens: 1,
            low_precision_input: false,
            seq_split: 1,
        };
        let result = op.query(&db, 32768, None).expect("query must succeed");
        assert!(
            (result.latency_ms - 41.59673055013021 * 0.5).abs() < 1e-9,
            "scale_factor must be applied to latency: got {}",
            result.latency_ms
        );
    }

    #[test]
    fn gemm_op_scale_num_tokens_divides_x() {
        let db = b200_vllm_db();
        // scale_num_tokens=2 means x=65536 should query at m=32768.
        let op = GemmOp {
            name: "halved".to_string(),
            scale_factor: 1.0,
            n: 65536,
            k: 16384,
            quant_mode: GemmQuantMode::Bfloat16,
            scale_num_tokens: 2,
            low_precision_input: false,
            seq_split: 1,
        };
        let result = op.query(&db, 65536, None).expect("query must succeed");
        assert!(
            (result.latency_ms - 41.59673055013021).abs() < 1e-9,
            "scale_num_tokens must divide x: got {}",
            result.latency_ms
        );
    }

    #[test]
    fn gemm_op_quant_override_routes_to_different_quant() {
        let db = b200_vllm_db();
        let op = GemmOp::new("default-bf16", 65536, 16384, GemmQuantMode::Bfloat16);

        // Override to nvfp4 at the same shape -> latency 20.5387 (recorded).
        let result = op
            .query(&db, 32768, Some(GemmQuantMode::Nvfp4))
            .expect("override query must succeed");
        assert!(
            (result.latency_ms - 20.538665771484375).abs() < 1e-9,
            "quant override must change the lookup: got {}",
            result.latency_ms
        );
    }

    #[test]
    fn gemm_op_weights_bytes_matches_python_formula() {
        let op = GemmOp::new("w", 1024, 4096, GemmQuantMode::Bfloat16);
        // bfloat16 memory factor is 2.0; weights = 1024 * 4096 * 2.0.
        assert_eq!(op.weights_bytes(), 1024.0 * 4096.0 * 2.0);

        let fp8_op = GemmOp::new("w-fp8", 1024, 4096, GemmQuantMode::Fp8);
        // fp8 memory factor is 1.0.
        assert_eq!(fp8_op.weights_bytes(), 1024.0 * 4096.0 * 1.0);
    }
}
