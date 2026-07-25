// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! MiniMax Sparse Attention (MSA) module ops for MiniMax-M3.
//!
//! Mirrors `aiconfigurator.sdk.operations.msa`. MSA is structurally a GQA
//! version of DSA: an indexer scores KV *blocks* (block_size tokens each),
//! the top-k blocks are selected, and full attention runs over only the
//! selected tokens.
//!
//! There is NO collected MSA silicon data. These ops therefore answer only
//! under HYBRID / EMPIRICAL: the SOL is analytic (same three-group split as
//! DSA/DSV4 — GEMM projections, FP8 indexer, sparse attention) and the
//! empirical value is a CROSS-OP (XOP) transfer from DSA's measured
//! utilisation at the same workload, scaled by the manual `dsa_scale_k`
//! level-alignment hook: `latency = SOL_msa / (util_dsa * k)`. SILICON mode
//! raises the perf-data miss; a policy with XOP disabled raises the terminal
//! empirical miss (there is nothing else to fall back on).

use serde::{Deserialize, Serialize};

use crate::common::enums::{
    DatabaseMode, FmhaQuantMode, GemmQuantMode, KvCacheQuantMode, TransferKind,
};
use crate::common::error::AicError;
use crate::common::system_spec::SystemSpec;
use crate::operators::base::{PerformanceResult, Source};
use crate::operators::dsa::DsaModuleOp;
use crate::perf_database::dsa::{dsa_context_sol_ms, dsa_dims, dsa_generation_sol_ms};
use crate::perf_database::gemm::tc_flops_for_compute;
use crate::perf_database::PerfDatabase;

/// One MSA module block (context or generation — the phase is chosen by the
/// `Op` variant). Field-for-field mirror of Python `_BaseMSAModule`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MsaModuleOp {
    pub name: String,
    pub scale_factor: f64,
    /// Local (per-rank) query heads.
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub hidden_size: u32,
    pub head_dim: u32,
    pub v_head_dim: u32,
    pub index_n_heads: u32,
    pub index_head_dim: u32,
    pub index_topk: u32,
    pub block_size: u32,
    pub kv_cache_dtype: KvCacheQuantMode,
    pub fmha_quant_mode: FmhaQuantMode,
    pub gemm_quant_mode: GemmQuantMode,
    /// The DSA architecture whose measured utilisation is borrowed (XOP).
    pub dsa_architecture: String,
    /// Manual cross-op level-alignment scale `k` (`latency = SOL/(util*k)`).
    pub dsa_scale_k: f64,
}

impl MsaModuleOp {
    /// Context (prefill) query. Mirrors `ContextMSAModule.query`.
    pub fn query_context(
        &self,
        db: &PerfDatabase,
        batch_size: u32,
        s: u32,
        prefix: u32,
    ) -> Result<PerformanceResult, AicError> {
        let sol = self.sol_ms(db, batch_size, s, prefix, true);
        match db.database_mode {
            DatabaseMode::Sol | DatabaseMode::SolFull => {
                Ok(PerformanceResult::new(sol * self.scale_factor, Source::Sol))
            }
            DatabaseMode::Silicon => Err(AicError::PerfDatabase(
                "MSA has no silicon data; use HYBRID or EMPIRICAL.".to_string(),
            )),
            DatabaseMode::Hybrid | DatabaseMode::Empirical => {
                if !db.transfer_policy.contains(TransferKind::XOp) {
                    return Err(AicError::EmpiricalNotImplemented(
                        "MSA context: cross-op transfer (xop) is disabled by the transfer policy \
                         and MSA has no own silicon data."
                            .to_string(),
                    ));
                }
                let util = self.dsa_context_util(db, batch_size, s, prefix);
                match util {
                    Some(util) if util > 0.0 => {
                        let latency = sol / (util * self.dsa_scale_k);
                        // Cross-op transfer from DSA (Python msa.py:297 "xop").
                        db.note_provenance(crate::operators::util_empirical::ProvenanceTier::XOp);
                        Ok(PerformanceResult::new(
                            latency * self.scale_factor,
                            Source::Empirical,
                        ))
                    }
                    _ => Err(AicError::EmpiricalNotImplemented(format!(
                        "MSA context: no DSA util to transfer from (arch={}, b={batch_size}, \
                         s={s}); collect DSA data or set msa_dsa_scale_k against an available \
                         quant.",
                        self.dsa_architecture
                    ))),
                }
            }
        }
    }

    /// Generation (decode) query; `s` is the total KV length. Mirrors
    /// `GenerationMSAModule.query`.
    pub fn query_generation(
        &self,
        db: &PerfDatabase,
        batch_size: u32,
        s: u32,
    ) -> Result<PerformanceResult, AicError> {
        let sol = self.sol_ms(db, batch_size, s, 0, false);
        match db.database_mode {
            DatabaseMode::Sol | DatabaseMode::SolFull => {
                Ok(PerformanceResult::new(sol * self.scale_factor, Source::Sol))
            }
            DatabaseMode::Silicon => Err(AicError::PerfDatabase(
                "MSA has no silicon data; use HYBRID or EMPIRICAL.".to_string(),
            )),
            DatabaseMode::Hybrid | DatabaseMode::Empirical => {
                if !db.transfer_policy.contains(TransferKind::XOp) {
                    return Err(AicError::EmpiricalNotImplemented(
                        "MSA generation: cross-op transfer (xop) is disabled by the transfer \
                         policy and MSA has no own silicon data."
                            .to_string(),
                    ));
                }
                let util = self.dsa_generation_util(db, batch_size, s);
                match util {
                    Some(util) if util > 0.0 => {
                        let latency = sol / (util * self.dsa_scale_k);
                        // Cross-op transfer from DSA (Python msa.py:335 "xop").
                        db.note_provenance(crate::operators::util_empirical::ProvenanceTier::XOp);
                        Ok(PerformanceResult::new(
                            latency * self.scale_factor,
                            Source::Empirical,
                        ))
                    }
                    _ => Err(AicError::EmpiricalNotImplemented(format!(
                        "MSA generation: no DSA util to transfer from (arch={}, b={batch_size}, \
                         s={s}); collect DSA data or set msa_dsa_scale_k against an available \
                         quant.",
                        self.dsa_architecture
                    ))),
                }
            }
        }
    }

    fn sol_ms(&self, db: &PerfDatabase, b: u32, s: u32, prefix: u32, is_context: bool) -> f64 {
        msa_attention_sol_ms(
            &db.system_spec,
            is_context,
            b as i128,
            s as i128,
            prefix as i128,
            self.num_heads as i128,
            self.num_kv_heads as i128,
            self.hidden_size as i128,
            self.head_dim as i128,
            self.v_head_dim as i128,
            self.index_n_heads as i128,
            self.index_head_dim as i128,
            self.index_topk as i128,
            self.block_size as i128,
            self.kv_cache_dtype,
            self.fmha_quant_mode,
            self.gemm_quant_mode,
        )
    }

    /// DSA's measured utilisation (SOL / silicon) at the same context
    /// workload, or `None`. Mirrors Python `_dsa_context_util`: the SOL comes
    /// from the analytic DSA formula (`database_mode=SOL`), the silicon value
    /// from a SILICON-view probe of the DSA module table; ANY failure means
    /// no transfer source.
    fn dsa_context_util(&self, db: &PerfDatabase, b: u32, s: u32, prefix: u32) -> Option<f64> {
        let dims = dsa_dims(&self.dsa_architecture);
        let sol = dsa_context_sol_ms(
            &db.system_spec,
            dims,
            dims.index_topk,
            self.kv_cache_dtype,
            self.fmha_quant_mode,
            self.gemm_quant_mode,
            b as i64,
            s as i64,
            prefix as i64,
            self.num_heads as i64,
            // MSA borrows the FULL DSA layer (Python's probe never sets
            // skip_indexer).
            false,
        );
        let probe = self.dsa_probe(dims.index_topk);
        let silicon = probe
            .query_context(&db.silicon_view(), b, s, prefix)
            .ok()?
            .latency_ms;
        if sol > 0.0 && silicon > 0.0 {
            Some(sol / silicon)
        } else {
            None
        }
    }

    /// DSA's measured utilisation at the same decode workload, or `None`.
    /// Mirrors Python `_dsa_generation_util`.
    fn dsa_generation_util(&self, db: &PerfDatabase, b: u32, s: u32) -> Option<f64> {
        let dims = dsa_dims(&self.dsa_architecture);
        let sol = dsa_generation_sol_ms(
            &db.system_spec,
            dims,
            self.kv_cache_dtype,
            self.gemm_quant_mode,
            b as i64,
            s as i64,
            self.num_heads as i64,
        );
        let probe = self.dsa_probe(dims.index_topk);
        let silicon = probe
            .query_generation(&db.silicon_view(), b, s)
            .ok()?
            .latency_ms;
        if sol > 0.0 && silicon > 0.0 {
            Some(sol / silicon)
        } else {
            None
        }
    }

    fn dsa_probe(&self, index_topk: i64) -> DsaModuleOp {
        DsaModuleOp::new(
            format!("{}_dsa_probe", self.name),
            self.num_heads,
            self.kv_cache_dtype,
            self.fmha_quant_mode,
            self.gemm_quant_mode,
            self.dsa_architecture.clone(),
            index_topk as u32,
        )
    }
}

/// SOL for one MSA block. Verbatim port of Python `_msa_attention_sol`
/// (`operations/msa.py`): GQA projections + per-block FP8 indexer + sparse
/// attention over the top-k selected tokens, integer pair counts in i128.
#[allow(clippy::too_many_arguments)]
fn msa_attention_sol_ms(
    spec: &SystemSpec,
    is_context: bool,
    b: i128,
    s: i128,
    prefix: i128,
    num_heads: i128,
    num_kv_heads: i128,
    hidden_size: i128,
    head_dim: i128,
    v_head_dim: i128,
    index_n_heads: i128,
    index_head_dim: i128,
    index_topk: i128,
    block_size: i128,
    kv_quant: KvCacheQuantMode,
    fmha_quant: FmhaQuantMode,
    gemm_quant: GemmQuantMode,
) -> f64 {
    let qk_head_dim = head_dim;
    let tokens = if is_context { b * s } else { b };
    // context: full prefill of `s` new tokens on top of `prefix` cached.
    // generation: 1 query token, kv_len = s - 1 cached.
    let full_s = if is_context { prefix + s } else { s };
    let kv_len = if is_context { full_s } else { (s - 1).max(0) };

    // ── GEMM group (Q / GQA-KV / O / indexer-Q projections) ──────────────
    let gemm_ops = 2 * tokens * hidden_size * (num_heads * qk_head_dim)
        + 2 * tokens * hidden_size * (2 * num_kv_heads * head_dim)
        + 2 * tokens * (num_heads * v_head_dim) * hidden_size
        + 2 * tokens * hidden_size * (index_n_heads * index_head_dim);

    // ── sparse attention: top-k saturated causal (query, kv) pair count ──
    let (pairs, score_len) = if is_context {
        let pairs = if full_s <= index_topk {
            b * (full_s * (full_s + 1) - prefix * (prefix + 1)) / 2
        } else if prefix >= index_topk {
            tokens * index_topk
        } else {
            let ramp = b * (index_topk * (index_topk + 1) - prefix * (prefix + 1)) / 2;
            let sat = b * (full_s - index_topk) * index_topk;
            ramp + sat
        };
        (pairs, full_s)
    } else {
        (tokens * kv_len.min(index_topk), kv_len)
    };
    let effective_kv = if is_context {
        full_s.min(index_topk)
    } else {
        kv_len.min(index_topk)
    };
    let attention_ops = 2 * num_heads * (qk_head_dim + v_head_dim) * pairs; // QK^T + AV

    // ── indexer: per-block scoring (block_size tokens per block), FP8 ────
    let num_blocks = if score_len > index_topk {
        (score_len + block_size - 1) / block_size
    } else {
        0
    };
    let indexer_ops = 2 * tokens * index_n_heads * index_head_dim * num_blocks;

    // ── memory ───────────────────────────────────────────────────────────
    let gemm_weight_elems = hidden_size * num_heads * qk_head_dim
        + hidden_size * 2 * num_kv_heads * head_dim
        + num_heads * v_head_dim * hidden_size
        + hidden_size * index_n_heads * index_head_dim;
    let gemm_weight_bytes = gemm_weight_elems as f64 * gemm_quant.mapping().memory;
    let kv_cache_bytes = (b * num_kv_heads * effective_kv * (qk_head_dim + v_head_dim)) as f64
        * kv_quant.mapping().memory;
    // FP8 index keys, per block (1 byte per element).
    let indexer_cache_bytes = (b * num_blocks * index_n_heads * index_head_dim) as f64;
    let q_io_bytes = (tokens * num_heads * qk_head_dim) as f64 * fmha_quant.mapping().memory * 2.0;
    let total_mem = gemm_weight_bytes + kv_cache_bytes + indexer_cache_bytes + q_io_bytes;

    let gemm_flops = tc_flops_for_compute(spec, gemm_quant.mapping().compute);
    // Python passes `common.FMHAQuantMode.fp8` (compute factor 2).
    let fp8_flops = tc_flops_for_compute(spec, 2.0);
    let attn_flops = tc_flops_for_compute(spec, fmha_quant.mapping().compute);

    let sol_math = (gemm_ops as f64 / gemm_flops
        + indexer_ops as f64 / fp8_flops
        + attention_ops as f64 / attn_flops)
        * 1000.0;
    let sol_mem = total_mem / spec.gpu.mem_bw * 1000.0;
    sol_math.max(sol_mem)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::enums::TransferPolicy;
    use std::path::PathBuf;

    const REPO_ROOT_HINT: &str = env!("CARGO_MANIFEST_DIR");

    fn db(backend: &str, version: &str) -> PerfDatabase {
        let systems_root = PathBuf::from(REPO_ROOT_HINT)
            .join("../..")
            .join("src/aiconfigurator_core/systems");
        let mut db = PerfDatabase::load(&systems_root, "b200_sxm", backend, version).expect("db must load");
        db.database_mode = DatabaseMode::Hybrid;
        db
    }

    /// MiniMax-M3 MSA block at tp=8 (local heads 8, GQA kv heads 1), all
    /// bfloat16 — the shapes the M3 model builder emits.
    fn msa_op() -> MsaModuleOp {
        MsaModuleOp {
            name: "msa".to_string(),
            scale_factor: 1.0,
            num_heads: 8,
            num_kv_heads: 1,
            hidden_size: 6144,
            head_dim: 128,
            v_head_dim: 128,
            index_n_heads: 4,
            index_head_dim: 128,
            index_topk: 2048,
            block_size: 128,
            kv_cache_dtype: KvCacheQuantMode::Bfloat16,
            fmha_quant_mode: FmhaQuantMode::Bfloat16,
            gemm_quant_mode: GemmQuantMode::Bfloat16,
            dsa_architecture: "GlmMoeDsaForCausalLM".to_string(),
            dsa_scale_k: 1.0,
        }
    }

    fn approx(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-9 * b.abs().max(1.0), "expected {b}, got {a}");
    }

    /// Oracle values from the Python reference (`ContextMSAModule` /
    /// `GenerationMSAModule` with `scale_factor=1.0` on a HYBRID
    /// `shared_layer=False` view). Regenerate if the DSA tables or the
    /// util math change.
    #[test]
    fn msa_xop_transfer_matches_python_oracles() {
        for (backend, version, anchors) in [
            (
                "sglang",
                "0.5.14",
                [
                    (1u32, 1024u32, 0u32, true, 0.15945479750992286),
                    (2, 3000, 512, true, 1.0573128907623435),
                    (8, 1025, 0, false, 0.03198051529058935),
                    (4, 7777, 0, false, 0.03568683502696848),
                ],
            ),
            (
                "vllm",
                "0.19.0",
                [
                    (1, 1024, 0, true, 0.44046553899838176),
                    (2, 3000, 512, true, 12.825449653876566),
                    (8, 1025, 0, false, 0.07392686165512992),
                    (4, 7777, 0, false, 0.0756313776883858),
                ],
            ),
        ] {
            let db = db(backend, version);
            let op = msa_op();
            for (b, s, prefix, is_context, expected) in anchors {
                db.reset_provenance();
                let result = if is_context {
                    op.query_context(&db, b, s, prefix)
                } else {
                    op.query_generation(&db, b, s)
                }
                .unwrap_or_else(|e| panic!("{backend} b={b} s={s}: {e}"));
                approx(result.latency_ms, expected);
                assert_eq!(result.source, Source::Empirical);
                // The xop tier must be recorded on the db accumulator
                // (Python msa.py:297/335 note_provenance("xop")).
                assert_eq!(
                    db.worst_provenance(),
                    crate::operators::util_empirical::ProvenanceTier::XOp
                );
            }
        }
    }

    /// XOP disabled ("balanced" preset) -> the terminal empirical miss; and
    /// SILICON mode -> the perf-data miss ("MSA has no silicon data").
    #[test]
    fn msa_policy_and_silicon_contracts() {
        let mut hybrid = db("vllm", "0.19.0");
        hybrid.transfer_policy = TransferPolicy {
            xshape: true,
            xquant: true,
            xprofile: false,
            xop: false,
        };
        let op = msa_op();
        assert!(matches!(
            op.query_context(&hybrid, 1, 1024, 0),
            Err(AicError::EmpiricalNotImplemented(_))
        ));
        assert!(matches!(
            op.query_generation(&hybrid, 8, 1025),
            Err(AicError::EmpiricalNotImplemented(_))
        ));

        let mut silicon = db("vllm", "0.19.0");
        silicon.database_mode = DatabaseMode::Silicon;
        assert!(matches!(
            op.query_context(&silicon, 1, 1024, 0),
            Err(AicError::PerfDatabase(_))
        ));
    }
}
