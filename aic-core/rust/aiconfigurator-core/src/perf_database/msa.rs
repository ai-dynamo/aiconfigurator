// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! MSA (MiniMax Sparse Attention) module perf tables.
//!
//! Two parquet files: `msa_context_module_perf.parquet` and
//! `msa_generation_module_perf.parquet`, collected with the exact DSA-module
//! row schema (architecture always `MiniMaxM3ForCausalLM`, kernel_source
//! `"default"`, op_name `msa_context_module` / `msa_generation_module` — never
//! a skip_indexer tag). Loading therefore reuses the DSA parquet loader
//! verbatim (Python `operations/msa.py` delegates to the DSA loaders the same
//! way), including the kernel_source -> dsa_backend bucketing that queries
//! resolve with `select_dsa_backend`'s fallback chain.
//!
//! Queries resolve on the RAW grids through the shared `perf_interp` v2
//! engine, mirroring Python `ContextMSAModule._query_context_msa_module_table`
//! / `GenerationMSAModule._query_generation_msa_module_table`:
//! - context: 4-axis Grid RAW `[num_heads][prefix][seq][batch]`;
//! - generation: 3-axis Grid RAW `[num_heads][batch][seq = isl + step]`.
//!
//! The analytic MSA SOL is injected by the operator (`operators/msa.rs`) — the
//! MSA structural dims live on the op, not on a per-architecture dims table.
//!
//! Python's generation loader keys `[kv][gemm][arch]` with NO mla_dtype axis;
//! the shared Rust loader retains it in `DsaKey`, so the generation node cache
//! here is built with the fmha component blanked (merged across mla_dtype
//! values — uniform in collected files) and queried the same way.

use std::path::PathBuf;
use std::sync::OnceLock;

use super::dsa::{
    build_context_nodes, build_generation_nodes, clone_err, load_dsa_parquet, missing,
    select_dsa_backend, DsaGrids, DsaKey, NodeCache,
};
use super::perf_interp::{self, OpInterpConfig};
use super::resolve_op_sources;
use crate::common::enums::{FmhaQuantMode, GemmQuantMode, KvCacheQuantMode};
use crate::common::error::AicError;
use crate::config::{PerfDbSources, PerfSource};

pub struct MsaTable {
    data_root: PathBuf,
    /// Ordered, priority-sorted sources (shared-layer aware; see [`PerfSource`]).
    context_sources: Vec<PerfSource>,
    generation_sources: Vec<PerfSource>,
    context: OnceLock<Result<DsaGrids, AicError>>,
    generation: OnceLock<Result<DsaGrids, AicError>>,
    context_nodes: OnceLock<Result<NodeCache, AicError>>,
    generation_nodes: OnceLock<Result<NodeCache, AicError>>,
}

impl MsaTable {
    /// Construct with shared-layer sources resolved from `perf_db_sources`
    /// (Python-supplied); each file falls back to its primary
    /// `data_root/<basename>` when absent from the map. No I/O.
    pub fn with_sources(data_root: PathBuf, perf_db_sources: &PerfDbSources) -> Self {
        let context_sources =
            resolve_op_sources(perf_db_sources, "msa_context_module_perf.parquet", &data_root);
        let generation_sources = resolve_op_sources(
            perf_db_sources,
            "msa_generation_module_perf.parquet",
            &data_root,
        );
        Self {
            data_root,
            context_sources,
            generation_sources,
            context: OnceLock::new(),
            generation: OnceLock::new(),
            context_nodes: OnceLock::new(),
            generation_nodes: OnceLock::new(),
        }
    }

    /// Context-MSA module latency: one 4-axis Grid RAW engine query on the raw
    /// `[num_heads][prefix][seq][batch]` table, evaluated at `isl` (the
    /// new-token count). `sol` is the operator's analytic MSA SOL over the
    /// engine coordinates `(num_heads, prefix, seq, batch)`.
    #[allow(clippy::too_many_arguments)]
    pub fn query_context(
        &self,
        b: u32,
        isl: u32,
        prefix: u32,
        num_heads: u32,
        kv_quant: KvCacheQuantMode,
        fmha_quant: FmhaQuantMode,
        gemm_quant: GemmQuantMode,
        architecture: &str,
        sol: &dyn Fn(&[f64]) -> f64,
    ) -> Result<f64, AicError> {
        let nodes = self.load_context_nodes()?;
        let key = DsaKey {
            architecture: architecture.to_string(),
            fmha_quant: fmha_quant.name().to_string(),
            kv_quant: kv_quant.name().to_string(),
            gemm_quant: gemm_quant.name().to_string(),
        };
        // MSA rows carry kernel_source="default"; the shared loader buckets
        // them under flashmla_kv (both buckets for bf16 KV) and the request
        // for "trtllm" descends the same fallback chain Python's
        // `_select_dsa_backend(node, "trtllm")` uses.
        let node = nodes
            .by_keys
            .get(&key)
            .and_then(|by_backend| select_dsa_backend(by_backend, "trtllm"))
            .ok_or_else(|| missing("context MSA module", &self.data_root, format!("{key:?}")))?;
        let cfg = OpInterpConfig::grid(&["num_heads", "prefix", "seq_len", "batch"], sol);
        perf_interp::query(
            &cfg,
            node,
            &[num_heads as f64, prefix as f64, isl as f64, b as f64],
        )
    }

    /// Generation-MSA module latency: one 3-axis Grid RAW engine query with
    /// the generation axis order `(num_heads, batch, seq)`, `seq = isl + step`
    /// collapsed at load time exactly like the DSA generation table. `sol` is
    /// the operator's decode SOL over those coordinates.
    #[allow(clippy::too_many_arguments)]
    pub fn query_generation(
        &self,
        b: u32,
        sequence_tokens: u32,
        num_heads: u32,
        kv_quant: KvCacheQuantMode,
        gemm_quant: GemmQuantMode,
        architecture: &str,
        sol: &dyn Fn(&[f64]) -> f64,
    ) -> Result<f64, AicError> {
        let nodes = self.load_generation_nodes()?;
        // Python's generation table has no mla_dtype axis; the node cache is
        // merged across it with a blanked fmha component (see module docs).
        let key = DsaKey {
            architecture: architecture.to_string(),
            fmha_quant: String::new(),
            kv_quant: kv_quant.name().to_string(),
            gemm_quant: gemm_quant.name().to_string(),
        };
        let node = nodes
            .by_keys
            .get(&key)
            .and_then(|by_backend| select_dsa_backend(by_backend, "trtllm"))
            .ok_or_else(|| missing("generation MSA module", &self.data_root, format!("{key:?}")))?;
        let cfg = OpInterpConfig::grid(&["num_heads", "batch", "seq_len"], sol);
        perf_interp::query(
            &cfg,
            node,
            &[num_heads as f64, b as f64, sequence_tokens as f64],
        )
    }

    fn load_context_nodes(&self) -> Result<&NodeCache, AicError> {
        let cell = self.context_nodes.get_or_init(|| {
            let grids = self.load_context()?;
            Ok(build_context_nodes(grids))
        });
        cell.as_ref().map_err(clone_err)
    }

    fn load_generation_nodes(&self) -> Result<&NodeCache, AicError> {
        let cell = self.generation_nodes.get_or_init(|| {
            let grids = self.load_generation()?;
            Ok(build_generation_nodes(&merge_generation_fmha(grids)))
        });
        cell.as_ref().map_err(clone_err)
    }

    fn load_context(&self) -> Result<&DsaGrids, AicError> {
        let cell = self
            .context
            .get_or_init(|| load_dsa_parquet(&self.context_sources, false, false));
        cell.as_ref().map_err(clone_err)
    }

    fn load_generation(&self) -> Result<&DsaGrids, AicError> {
        // Collapse (isl, step) -> seq at LOAD time like the DSA generation
        // table; MSA decode rows are collected as isl=1, step=kv_len.
        let cell = self
            .generation
            .get_or_init(|| load_dsa_parquet(&self.generation_sources, true, false));
        cell.as_ref().map_err(clone_err)
    }

    /// Test-only synthetic-table injection (no parquet writer in the test
    /// deps). Panics if the lazily-loaded cells were already initialised.
    #[cfg(test)]
    pub(crate) fn inject_for_test(&self, context: DsaGrids, generation: DsaGrids) {
        assert!(self.context.set(Ok(context)).is_ok(), "context grids already loaded");
        assert!(
            self.generation.set(Ok(generation)).is_ok(),
            "generation grids already loaded"
        );
    }
}

/// Merge a generation `DsaGrids` across the fmha (`mla_dtype`) key component
/// into a single blanked-fmha key per (arch, kv, gemm), mirroring Python's
/// fmha-less generation nesting. Collected MSA files carry one uniform
/// mla_dtype, so the deterministic BTreeMap-order overwrite on a duplicate
/// coordinate never fires in practice.
fn merge_generation_fmha(grids: &DsaGrids) -> DsaGrids {
    let mut by_keys: std::collections::BTreeMap<DsaKey, _> = std::collections::BTreeMap::new();
    for (key, by_backend) in &grids.by_keys {
        let merged_key = DsaKey {
            architecture: key.architecture.clone(),
            fmha_quant: String::new(),
            kv_quant: key.kv_quant.clone(),
            gemm_quant: key.gemm_quant.clone(),
        };
        let backends: &mut std::collections::BTreeMap<String, _> =
            by_keys.entry(merged_key).or_default();
        for (backend, by_heads) in by_backend {
            let dest = backends.entry(backend.clone()).or_default();
            merge_head_grid(dest, by_heads);
        }
    }
    DsaGrids { by_keys }
}

fn merge_head_grid(dest: &mut super::dsa::DsaHeadGrid, src: &super::dsa::DsaHeadGrid) {
    for (&n, by_step) in src {
        let dest_step = dest.entry(n).or_default();
        for (&step, by_isl) in by_step {
            let dest_isl = dest_step.entry(step).or_default();
            for (&isl, by_batch) in by_isl {
                let dest_batch = dest_isl.entry(isl).or_default();
                for (&bb, &lat) in by_batch {
                    dest_batch.insert(bb, lat);
                }
            }
        }
    }
}
