// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DSA (Dynamic Sparse Attention) module operator.
//!
//! Mirrors `aiconfigurator.sdk.operations.dsa.ContextDSAModule` /
//! `GenerationDSAModule`. The context lookup evaluates at `isl` (the
//! new-token count) on the raw 4-axis `[heads][prefix][seq][batch]` grid via
//! the perf_interp v2 engine (see `perf_database::dsa::query_context`).
//!
//! `index_topk` is the top-k boundary (per-architecture; 2048 for both
//! DeepSeek-V3.2 and GLM-5). It is plumbed from the Python op-spec emitter.

use serde::{Deserialize, Serialize};
use crate::common::enums::{FmhaQuantMode, GemmQuantMode, KvCacheQuantMode};
use crate::common::error::AicError;
use crate::operators::base::{PerformanceResult, Source};
use crate::operators::communication::NcclOp;
use crate::perf_database::dsa::{
    bs_slice, dsa_dims, dsa_sparse_file_prefix, lookup_2d, DsaSparseTables,
};
use crate::perf_database::PerfDatabase;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DsaModuleOp {
    pub name: String,
    pub scale_factor: f64,
    pub num_heads: u32,
    pub kv_cache_dtype: KvCacheQuantMode,
    pub fmha_quant_mode: FmhaQuantMode,
    pub gemm_quant_mode: GemmQuantMode,
    pub architecture: String,
    /// Top-k boundary for the sparse-attention regime split. Sourced from
    /// `DSA_MODEL_DIMS[architecture]["index_topk"]` on the Python side.
    pub index_topk: u32,
    /// Context-parallel size (Python `ContextDSAModule._cp_size`). When > 1
    /// the context query runs the GLM-5/DSA sparse-CP prefill composition
    /// (Python `_query_cp`); 1 (the default) keeps the plain 4-axis lookup.
    /// NOT yet emitted by the Python opspec — `engine.py::_reject_cp` still
    /// guards CP specs; the guard and this field's emission flip atomically
    /// once BOTH the dsa and dsv4 Rust CP paths land.
    #[serde(default = "default_cp_size")]
    pub cp_size: u32,
    /// GLM-5.2 shared-index amortization weight (Python `_full_frac`): the
    /// exact fraction of indexer-computing layers. Per-layer cost is
    /// `full_frac*full + (1-full_frac)*skip` using the directly-collected
    /// skip-indexer table. 1.0 (DeepSeek-V3.2 / GLM-5, and pre-field opspecs)
    /// keeps the pure-full path — the skip table is never touched.
    #[serde(default = "default_full_frac")]
    pub full_frac: f64,
}

fn default_cp_size() -> u32 {
    1
}

fn default_full_frac() -> f64 {
    1.0
}

impl DsaModuleOp {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: impl Into<String>,
        num_heads: u32,
        kv_cache_dtype: KvCacheQuantMode,
        fmha_quant_mode: FmhaQuantMode,
        gemm_quant_mode: GemmQuantMode,
        architecture: impl Into<String>,
        index_topk: u32,
    ) -> Self {
        Self {
            name: name.into(),
            scale_factor: 1.0,
            num_heads,
            kv_cache_dtype,
            fmha_quant_mode,
            gemm_quant_mode,
            architecture: architecture.into(),
            index_topk,
            cp_size: 1,
            full_frac: 1.0,
        }
    }

    pub fn query_context(
        &self,
        db: &PerfDatabase,
        batch_size: u32,
        isl: u32,
        prefix: u32,
    ) -> Result<PerformanceResult, AicError> {
        let w = self.full_frac;
        // CP (round-robin sequence split) prefill takes the sparse-delta
        // composition path (Python `ContextDSAModule.query` -> `_query_cp`
        // when `_cp_size > 1`). GLM-5.2 amortizes full/skip on the CP path
        // too (both carry the same scale_factor, so the weighted sum of the
        // already-scaled results is exact — Python `_amortize`).
        if self.cp_size > 1 {
            let full = self.query_context_cp(db, batch_size, isl, prefix, false)?;
            if w >= 1.0 {
                return Ok(full);
            }
            let skip = self.query_context_cp(db, batch_size, isl, prefix, true)?;
            return Ok(PerformanceResult::new(
                w * full.latency_ms + (1.0 - w) * skip.latency_ms,
                full.source,
            ));
        }
        // Query at `isl` (new-token count) for the exact `prefix` slice — NOT
        // `isl + prefix`. The perf-DB layer resolves one 4-axis RAW grid via
        // the perf_interp v2 engine; there is no multiplicative prefix
        // correction (it had no Python counterpart and under-counted context
        // latency ~75%). `dsa_backend="trtllm"` mirrors Python's non-CP
        // default (`_query_context_dsa_module_table(dsa_backend="trtllm")`).
        let q = |skip_indexer: bool| {
            db.dsa.query_context(
                &db.system_spec,
                batch_size,
                isl,
                self.num_heads,
                self.kv_cache_dtype,
                self.fmha_quant_mode,
                self.gemm_quant_mode,
                &self.architecture,
                prefix,
                self.index_topk,
                "trtllm",
                skip_indexer,
            )
        };
        let full = q(false)?;
        let latency = if w >= 1.0 {
            full
        } else {
            // GLM-5.2 shared-index amortization (Python ContextDSAModule.query).
            w * full + (1.0 - w) * q(true)?
        };
        Ok(PerformanceResult::new(latency, Source::Silicon)
            .clamp_non_negative()
            .scaled(self.scale_factor))
    }

    /// Context-Parallel (CP) prefill — GLM-5/DSA sparse composition.
    ///
    /// Wires the real data dependencies and delegates to [`Self::query_cp_with`]
    /// (the verbatim mirror of Python `ContextDSAModule._query_cp`,
    /// `skip_indexer=False` — the Rust table loads Python's full slice, so
    /// this matches the full-layer-only opspec path):
    /// - base = the existing 4-axis engine query at `(b, per_card, prefix)`
    ///   with `dsa_backend="flashmla_kv"`, exactly like Python's `_query_cp`
    ///   base query;
    /// - AG = `db.query_nccl(half, cp, "all_gather", elems)` via [`NcclOp`],
    ///   which mirrors Python's fan-out cap + multi-node bandwidth scaling.
    fn query_context_cp(
        &self,
        db: &PerfDatabase,
        batch_size: u32,
        isl: u32,
        prefix: u32,
        skip_indexer: bool,
    ) -> Result<PerformanceResult, AicError> {
        let sparse = db.dsa.load_cp_sparse(&self.architecture, self.num_heads)?;
        let mut base = |per_card: u32| {
            db.dsa.query_context(
                &db.system_spec,
                batch_size,
                per_card,
                self.num_heads,
                self.kv_cache_dtype,
                self.fmha_quant_mode,
                self.gemm_quant_mode,
                &self.architecture,
                prefix,
                self.index_topk,
                // Python `_query_cp` queries the CP base on the flashmla_kv
                // slice (the kernel used under CP).
                "flashmla_kv",
                skip_indexer,
            )
        };
        let mut ag = |elems: u64| {
            NcclOp::new(
                format!("{}_cp_all_gather", self.name),
                1.0,
                elems as f64,
                self.cp_size,
                "all_gather",
            )
            .query(db, 1)
            .map(|r| r.latency_ms)
        };
        self.query_cp_with(&sparse, batch_size, isl, prefix, skip_indexer, &mut base, &mut ag)
    }

    /// CP (round-robin split) per-layer DSA composition. Verbatim mirror of
    /// Python `ContextDSAModule._query_cp` (2026-06-11 strategy):
    ///
    /// ```text
    /// result = dsa(isl/cp, prefix)
    ///        + [mqa(isl, prefix)/cp       - mqa(isl/cp, prefix)]
    ///        + [topk_last(isl, prefix)/cp - topk_flat(isl/cp, prefix)]
    ///        + AG_KV + AG_LSE
    /// ```
    ///
    /// `base(per_card)` supplies the per-card monolithic dsa_module latency;
    /// `ag(elems)` the all-gather latency for an element volume (bf16). Both
    /// are injected so the composition is unit-testable against the same
    /// synthetic inputs as the Python test
    /// (`tests/unit/sdk/test_cp_dsa_modeling.py::test_query_cp_composition`).
    fn query_cp_with(
        &self,
        sparse: &DsaSparseTables,
        b: u32,
        isl: u32,
        prefix: u32,
        skip_indexer: bool,
        base: &mut dyn FnMut(u32) -> Result<f64, AicError>,
        ag: &mut dyn FnMut(u64) -> Result<f64, AicError>,
    ) -> Result<PerformanceResult, AicError> {
        let cp = self.cp_size;
        let per_card = isl.div_ceil(cp).max(1); // ceil: critical path = busiest CP rank
        let file_prefix = dsa_sparse_file_prefix(&self.architecture);
        // Fail fast: CP DSA modeling REQUIRES the sparse mqa/topk tables for
        // the mqa/topk_last deltas. `lookup_2d` clamps isl + interpolates
        // step, so an empty grid below means the table is absent entirely
        // (parquet not collected) — degrading silently to dsa_base would hide
        // that. Message shape mirrors Python's fail-loud contract.
        // skip_indexer layers carry NO indexer -> no mqa/topk deltas needed,
        // so don't require the sparse tables for them (Python dsa.py:835-837).
        let missing: Vec<&str> = if skip_indexer {
            Vec::new()
        } else {
            [
                ("mqa", &sparse.mqa),
                ("topk_last", &sparse.topk_last),
                ("topk_flat", &sparse.topk_flat),
            ]
            .into_iter()
            .filter(|(_, grid)| grid.is_empty())
            .map(|(name, _)| name)
            .collect()
        };
        if !missing.is_empty() {
            return Err(AicError::PerfDatabase(format!(
                "DSA CP modeling needs sparse tables ['{}'] for {} (num_heads={}); \
                 collect {file_prefix}_mqa_logits/{file_prefix}_topk first.",
                missing.join("', '"),
                self.architecture,
                self.num_heads
            )));
        }
        // Base: per-card monolithic dsa_module at (per_card, prefix).
        let dsa_base = base(per_card)?;
        // Look the sparse sub-kernels up at the REAL batch b (the bs slice
        // carries the measured bs=b latency), so the delta matches dsa_base
        // (queried at b) WITHOUT an external x b linearity assumption.
        let mqa_tab = bs_slice(&sparse.mqa, b);
        let tl_tab = bs_slice(&sparse.topk_last, b);
        let tf_tab = bs_slice(&sparse.topk_flat, b);
        let empty = std::collections::BTreeMap::new();
        let mqa_tab = mqa_tab.unwrap_or(&empty);
        let tl_tab = tl_tab.unwrap_or(&empty);
        let tf_tab = tf_tab.unwrap_or(&empty);
        let mqa_full = lookup_2d(mqa_tab, isl, prefix)?;
        let mqa_perc = lookup_2d(mqa_tab, per_card, prefix)?;
        let tl_full = lookup_2d(tl_tab, isl, prefix)?;
        let tf_perc = lookup_2d(tf_tab, per_card, prefix)?;
        let mut latency = dsa_base;
        // skip layers reuse a sibling's topk index: no per-layer mqa/topk, so
        // no full/cp deltas — just the per-card skip base + the attention
        // all-gathers (Python dsa.py:871-876).
        if !skip_indexer {
            if let (Some(mqa_full), Some(mqa_perc), Some(tl_full), Some(tf_perc)) =
                (mqa_full, mqa_perc, tl_full, tf_perc)
            {
                let delta_mqa = mqa_full / f64::from(cp) - mqa_perc;
                let delta_topk = tl_full / f64::from(cp) - tf_perc;
                latency += delta_mqa + delta_topk;
            }
        }
        // CP attention all-gathers, per current-chunk tokens (isl, not
        // isl+prefix; prefix KV is already replicated), bf16 (see the Python
        // comment block for the sglang instrumentation provenance):
        //   ag_kv  = DSA indexer key      -> b * isl * index_head_dim  (=128)
        //   ag_lse = compressed KV latent -> b * isl * (kv_lora + rope) (=576)
        // (The hidden_states AG/RS is the MoE token dispatch, modeled by the
        // MoE dispatch ops, not here.)
        let dims = dsa_dims(&self.architecture);
        let tokens = u64::from(b) * u64::from(isl);
        // A skip-indexer (reuse) layer never runs the per-layer indexer, so it
        // does not all-gather the DSA indexer key — only the MLA
        // compressed-KV/LSE gather remains (Python dsa.py:895-902).
        let ag_kv = if skip_indexer {
            0.0
        } else {
            ag(tokens * dims.index_head_dim as u64)?
        };
        let ag_lse = ag(tokens * (dims.kv_lora_rank + dims.qk_rope_head_dim) as u64)?;
        latency += ag_kv + ag_lse;
        Ok(PerformanceResult::new(latency, Source::Estimated).scaled(self.scale_factor))
    }

    pub fn query_generation(
        &self,
        db: &PerfDatabase,
        batch_size: u32,
        s: u32,
    ) -> Result<PerformanceResult, AicError> {
        // `dsa_backend="trtllm"` mirrors Python's generation default
        // (`_query_generation_dsa_module_table(dsa_backend="trtllm")`).
        let q = |skip_indexer: bool| {
            db.dsa.query_generation(
                &db.system_spec,
                batch_size,
                s,
                self.num_heads,
                self.kv_cache_dtype,
                self.fmha_quant_mode,
                self.gemm_quant_mode,
                &self.architecture,
                "trtllm",
                skip_indexer,
            )
        };
        let w = self.full_frac;
        let full = q(false)?;
        let latency = if w >= 1.0 {
            full
        } else {
            // GLM-5.2 shared-index amortization (Python GenerationDSAModule.query).
            w * full + (1.0 - w) * q(true)?
        };
        Ok(PerformanceResult::new(latency, Source::Silicon)
            .clamp_non_negative()
            .scaled(self.scale_factor))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perf_database::dsa::SparseGrid;
    use std::path::PathBuf;

    fn glm_cp_op(cp_size: u32) -> DsaModuleOp {
        DsaModuleOp {
            name: "context_dsa".into(),
            scale_factor: 1.0,
            num_heads: 64,
            kv_cache_dtype: KvCacheQuantMode::Bfloat16,
            fmha_quant_mode: FmhaQuantMode::Bfloat16,
            gemm_quant_mode: GemmQuantMode::Bfloat16,
            architecture: "GlmMoeDsaForCausalLM".into(),
            index_topk: 2048,
            cp_size,
            full_frac: 1.0,
        }
    }

    fn grid(rows: &[(u32, u32, u32, f64)]) -> SparseGrid {
        let mut g = SparseGrid::new();
        for &(bs, isl, step, lat) in rows {
            g.entry(bs).or_default().insert((isl, step), lat);
        }
        g
    }

    /// Composition parity with Python
    /// `tests/unit/sdk/test_cp_dsa_modeling.py::test_query_cp_composition`,
    /// same synthetic inputs (cp=8, isl=16384, prefix=0, b=1; base 4300,
    /// each AG 50):
    ///
    ///   delta_mqa  = mqa_full/cp - mqa_perc  = 1600/8 - 25  = 175
    ///   delta_topk = tl_full/cp  - tf_perc   = 800/8  - 100 = 0
    ///   latency    = 4300 + 175 + 0 + ag_kv 50 + ag_lse 50  = 4575
    ///
    /// AG volumes: indexer key isl*128, compressed latent isl*(512 + 64).
    #[test]
    fn cp_composition_matches_python_synthetic() {
        let (cp, isl, prefix) = (8u32, 16384u32, 0u32);
        let per_card = isl.div_ceil(cp); // 2048
        let sparse = DsaSparseTables {
            mqa: grid(&[(1, isl, 0, 1600.0), (1, per_card, 0, 25.0)]),
            topk_last: grid(&[(1, isl, 0, 800.0), (1, per_card, 0, 190.0)]),
            topk_flat: grid(&[(1, per_card, 0, 100.0)]),
            dsa_attn: SparseGrid::new(),
        };
        let op = glm_cp_op(cp);
        let mut base_calls: Vec<u32> = Vec::new();
        let mut ag_volumes: Vec<u64> = Vec::new();
        let res = op
            .query_cp_with(
                &sparse,
                1,
                isl,
                prefix,
                false,
                &mut |per_card| {
                    base_calls.push(per_card);
                    Ok(4300.0) // per-card monolithic base
                },
                &mut |elems| {
                    ag_volumes.push(elems);
                    Ok(50.0) // each AG
                },
            )
            .expect("CP composition must succeed");
        assert_eq!(res.latency_ms, 4575.0);
        assert_eq!(res.source, Source::Estimated);
        assert_eq!(base_calls, vec![per_card]); // base queried at isl/cp
        ag_volumes.sort_unstable();
        let mut expected = vec![u64::from(isl) * 128, u64::from(isl) * (512 + 64)];
        expected.sort_unstable();
        assert_eq!(ag_volumes, expected);
    }

    /// Mirrors Python `test_query_cp_raises_when_isl_beyond_grid`: the
    /// composition must propagate `lookup_2d`'s fail-loud (no silent
    /// under-estimate) when isl exceeds the collected sparse grid.
    #[test]
    fn cp_propagates_lookup_fail_loud_beyond_grid() {
        let sparse = DsaSparseTables {
            mqa: grid(&[(1, 16384, 0, 1600.0), (1, 4096, 0, 25.0)]), // grid caps at 16384
            topk_last: grid(&[(1, 16384, 0, 800.0), (1, 4096, 0, 190.0)]),
            topk_flat: grid(&[(1, 4096, 0, 100.0)]),
            dsa_attn: SparseGrid::new(),
        };
        let op = glm_cp_op(8);
        let err = op
            .query_cp_with(
                &sparse,
                1,
                32768, // > grid max 16384
                0,
                false,
                &mut |_| Ok(4300.0),
                &mut |_| Ok(50.0),
            )
            .unwrap_err();
        assert!(
            err.to_string().contains("exceeds the collected"),
            "unexpected message: {err}"
        );
    }

    /// Fail-loud symmetry with Python's absent-sparse-table contract: the
    /// glm5_* / dsv32_* sparse parquets ship nowhere, so a CP context query
    /// against a real perf DB must error, naming the missing tables and the
    /// files to collect — never degrade silently to the per-card base.
    #[test]
    fn cp_missing_sparse_tables_fail_loud() {
        let systems_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("src/aiconfigurator/systems");
        let db = PerfDatabase::load(&systems_root, "b200_sxm", "vllm", "0.19.0")
            .expect("b200_sxm/vllm/0.19.0 must load");
        let op = glm_cp_op(8);
        let err = op.query_context(&db, 1, 16384, 0).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("DSA CP modeling needs sparse tables")
                && msg.contains("['mqa', 'topk_last', 'topk_flat']")
                && msg.contains("GlmMoeDsaForCausalLM")
                && msg.contains("num_heads=64")
                && msg.contains("collect glm5_mqa_logits/glm5_topk first."),
            "unexpected message: {msg}"
        );
    }

    /// `cp_size` is absent from every opspec the Python emitter produces
    /// today (`engine.py::_reject_cp` still guards CP) — it must default to
    /// 1 so existing specs keep the plain non-CP lookup.
    #[test]
    fn cp_size_defaults_to_one_in_serde() {
        let mut v = serde_json::to_value(glm_cp_op(3)).expect("serialize");
        v.as_object_mut().expect("object").remove("cp_size");
        let de: DsaModuleOp = serde_json::from_value(v).expect("deserialize");
        assert_eq!(de.cp_size, 1);
    }
}
