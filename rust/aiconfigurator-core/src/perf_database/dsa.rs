// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DSA (DeepSeek-V3.2 Dynamic Sparse Attention) module perf tables.
//!
//! Two parquet files: `dsa_context_module_perf.parquet` and
//! `dsa_generation_module_perf.parquet`. Both share columns: model,
//! architecture, op_name, kernel_source, mla_dtype, kv_cache_dtype,
//! gemm_type, num_heads, batch_size, isl, tp_size, step, latency. Each file
//! loads from an ordered, shared-layer-aware source list (see [`PerfSource`]).
//!
//! Data is nested by (architecture, mla_dtype, kv_cache_dtype, gemm_type)
//! → dsa_backend → num_heads → step → isl → batch_size → latency. The
//! `step` axis is the "prefix value" (past-KV length). Mirroring Python
//! `load_context_dsa_module_data` / `load_generation_dsa_module_data`:
//! - full vs GLM-5.2 skip-indexer rows share ONE file, split by `op_name`
//!   (`*_skip_indexer` suffix); the tables here keep the FULL rows (the
//!   default slice every Rust query path consumes — Python's op passes
//!   `skip_indexer=False` on the paths ported here).
//! - `dsa_backend` bucket(s) are derived per row mirroring Python
//!   `_dsa_kernel_source_buckets` (bf16-KV rows back BOTH buckets; fp8 rows
//!   bucket by executed-kernel name); queries select a backend slice
//!   with Python `_select_dsa_backend`'s fallback chain.
//!
//! Queries resolve on the RAW grids through the shared `perf_interp` v2
//! engine, mirroring Python `operations/dsa.py`:
//! - context: 4-axis Grid RAW `[num_heads][prefix][seq][batch]` — the
//!   topk-piecewise dispatch and the DSv4 robust-lookup / batch-scaling
//!   layers were DELETED in v2 (plain linear bracket crossing over the topk
//!   knee measured fine, +1.0% signed), and out-of-range (incl. prefix) is
//!   util-hold with the regime-aware analytic SOL.
//! - generation: 3-axis Grid RAW `[num_heads][batch][seq]` where
//!   `seq = isl + step` (Python `generation_grid_config` axis order:
//!   batch BEFORE seq).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use crate::common::enums::{FmhaQuantMode, GemmQuantMode, KvCacheQuantMode};
use crate::common::error::AicError;
use crate::common::system_spec::SystemSpec;
use crate::config::{PerfDbSources, PerfSource};
use super::perf_interp::{self, Node, OpInterpConfig};
use super::{kernel_source_ok, resolve_op_sources};
use crate::perf_database::parquet_loader::PerfReader;

pub struct DsaTable {
    data_root: PathBuf,
    /// Ordered, priority-sorted sources for the context/generation DSA perf
    /// files (shared-layer aware; see [`PerfSource`]). Single-primary,
    /// no-filter by default (`DsaTable::new`).
    context_sources: Vec<PerfSource>,
    generation_sources: Vec<PerfSource>,
    context: OnceLock<Result<DsaGrids, AicError>>,
    generation: OnceLock<Result<DsaGrids, AicError>>,
    /// GLM-5.2 skip-indexer (reuse-layer) tables — same files, rows tagged by
    /// `op_name` `*_skip_indexer`. Loaded lazily; empty when the parquet
    /// carries no skip rows (DeepSeek-V3.2 / GLM-5), in which case the skip
    /// query fails loud exactly like Python's `None` slot.
    context_skip: OnceLock<Result<DsaGrids, AicError>>,
    generation_skip: OnceLock<Result<DsaGrids, AicError>>,
    context_skip_nodes: OnceLock<Result<NodeCache, AicError>>,
    generation_skip_nodes: OnceLock<Result<NodeCache, AicError>>,
    /// Engine-ready per-`DsaKey` context tables with the raw shape
    /// `[num_heads][step][isl][batch]`, built once from the loaded grids.
    context_nodes: OnceLock<Result<NodeCache, AicError>>,
    /// Engine-ready per-`DsaKey` generation tables with the Python v2 axis
    /// order `[num_heads][batch][seq = isl + step]`, built once at load.
    generation_nodes: OnceLock<Result<NodeCache, AicError>>,
    /// Shared-layer source map retained for the lazily-resolved CP sparse
    /// sub-kernel files (`<prefix>_{mqa_logits,topk,dsa_attn}_module_perf.parquet`,
    /// prefix = glm5 / dsv32 by architecture). Python reads these primary-only
    /// (`_load_glm5_sparse` -> `pd.read_parquet(data_dir/<fn>)`); the default
    /// (empty-map) resolution here is exactly that single primary file.
    perf_db_sources: PerfDbSources,
    /// CP sparse sub-kernel tables keyed by (file_prefix, num_heads) — the
    /// Rust mirror of Python `ContextDSAModule._glm5_sparse_cache`.
    sparse: Mutex<BTreeMap<(String, u32), Arc<DsaSparseTables>>>,
}

/// One CP sparse sub-kernel grid: `bs -> {(isl, step) -> latency_ms}`.
/// Mirrors the Python `_load_glm5_sparse` `out2d` entries (bs-keyed so
/// `_query_cp` looks the deltas up at the REAL measured batch).
pub type SparseGrid = BTreeMap<u32, BTreeMap<(u32, u32), f64>>;

/// DSA sparse sub-kernel tables (mqa / topk / dsa_attn) for the CP prefill
/// composition. An absent parquet leaves its grid empty — the operator's
/// missing-tables check fails loud, mirroring Python (`_read` -> None -> `{}`).
#[derive(Debug, Default, PartialEq)]
pub struct DsaSparseTables {
    pub mqa: SparseGrid,
    pub topk_last: SparseGrid,
    pub topk_flat: SparseGrid,
    /// Optional (not used by the CP delta; DSV3.2 only collects mqa + topk).
    pub dsa_attn: SparseGrid,
}

/// DSA sparse sub-kernel data-file prefix per architecture. Mirrors Python
/// `operations.dsa._dsa_sparse_file_prefix` (defaults to glm5 for back-compat).
pub fn dsa_sparse_file_prefix(architecture: &str) -> &'static str {
    match architecture {
        "DeepseekV32ForCausalLM" => "dsv32",
        _ => "glm5",
    }
}

struct NodeCache {
    /// (arch, fmha, kv, gemm) → dsa_backend → engine table.
    by_keys: BTreeMap<DsaKey, BTreeMap<String, Node>>,
}

/// num_heads → step → isl → batch → latency.
pub type DsaHeadGrid = BTreeMap<u32, BTreeMap<u32, BTreeMap<u32, BTreeMap<u32, f64>>>>;

/// (arch, fmha, kv, gemm) → dsa_backend → num_heads → step → isl → batch →
/// latency. The `dsa_backend` level mirrors Python's
/// `...[architecture][dsa_backend][num_heads]...` nesting.
pub struct DsaGrids {
    pub by_keys: BTreeMap<DsaKey, BTreeMap<String, DsaHeadGrid>>,
}

/// Pick the per-backend slice from a backend-keyed map. Mirror of Python
/// `_select_dsa_backend`: requested backend, else `flashmla_kv`, else
/// `trtllm`, else the first populated slice (single-backend parquets still
/// resolve). The derived backend values are only ever `trtllm` /
/// `flashmla_kv`, so the final arm is a defensive mirror of Python's
/// `next(iter(arch_dict.values()))`.
fn select_dsa_backend<'a, T>(
    by_backend: &'a BTreeMap<String, T>,
    dsa_backend: &str,
) -> Option<&'a T> {
    by_backend
        .get(dsa_backend)
        .or_else(|| by_backend.get("flashmla_kv"))
        .or_else(|| by_backend.get("trtllm"))
        .or_else(|| by_backend.values().next())
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct DsaKey {
    pub architecture: String,
    pub fmha_quant: String,
    pub kv_quant: String,
    pub gemm_quant: String,
}

/// Per-architecture structural dims. Mirrors Python
/// `operations.dsa.DSA_MODEL_DIMS`; unknown architectures fall back to the
/// DeepSeek-V3.2 dims (Python `DSA_MODEL_DIMS.get(arch, DSA_MODEL_DIMS[DEFAULT])`).
pub(crate) struct DsaDims {
    pub(crate) hidden_size: i64,
    pub(crate) q_lora_rank: i64,
    pub(crate) kv_lora_rank: i64,
    pub(crate) qk_nope_head_dim: i64,
    pub(crate) qk_rope_head_dim: i64,
    pub(crate) v_head_dim: i64,
    pub(crate) index_topk: i64,
    pub(crate) index_head_dim: i64,
    pub(crate) index_n_heads: i64,
}

const DSV32_DIMS: DsaDims = DsaDims {
    hidden_size: 7168,
    q_lora_rank: 1536,
    kv_lora_rank: 512,
    qk_nope_head_dim: 128,
    qk_rope_head_dim: 64,
    v_head_dim: 128,
    index_topk: 2048,
    index_head_dim: 128,
    index_n_heads: 64,
};

const GLM_MOE_DSA_DIMS: DsaDims = DsaDims {
    hidden_size: 6144,
    q_lora_rank: 2048,
    qk_nope_head_dim: 192,
    kv_lora_rank: 512,
    qk_rope_head_dim: 64,
    v_head_dim: 256,
    index_topk: 2048,
    index_head_dim: 128,
    index_n_heads: 32,
};

pub(crate) fn dsa_dims(architecture: &str) -> &'static DsaDims {
    match architecture {
        "GlmMoeDsaForCausalLM" => &GLM_MOE_DSA_DIMS,
        _ => &DSV32_DIMS,
    }
}

impl DsaTable {
    /// Construct an empty table for the given data directory. No I/O. Each
    /// perf file is sourced solely from `data_root/<basename>` with no
    /// `kernel_source` filter (pre-shared-layer behaviour).
    pub fn new(data_root: PathBuf) -> Self {
        Self::with_sources(data_root, &PerfDbSources::default())
    }

    /// Construct with shared-layer (sibling/cross-version) sources resolved from
    /// `perf_db_sources` (Python-supplied). Each DSA file falls back to its
    /// primary `data_root/<basename>` when absent from the map. No I/O.
    pub fn with_sources(data_root: PathBuf, perf_db_sources: &PerfDbSources) -> Self {
        let context_sources =
            resolve_op_sources(perf_db_sources, "dsa_context_module_perf.parquet", &data_root);
        let generation_sources = resolve_op_sources(
            perf_db_sources,
            "dsa_generation_module_perf.parquet",
            &data_root,
        );
        Self {
            data_root,
            context_sources,
            generation_sources,
            context: OnceLock::new(),
            generation: OnceLock::new(),
            context_skip: OnceLock::new(),
            generation_skip: OnceLock::new(),
            context_skip_nodes: OnceLock::new(),
            generation_skip_nodes: OnceLock::new(),
            context_nodes: OnceLock::new(),
            generation_nodes: OnceLock::new(),
            perf_db_sources: perf_db_sources.clone(),
            sparse: Mutex::new(BTreeMap::new()),
        }
    }

    /// Load the DSA CP sparse sub-kernel tables (mqa / topk / dsa_attn) for
    /// `architecture` filtered to `num_heads`. Mirror of Python
    /// `ContextDSAModule._load_glm5_sparse`: architecture selects the file
    /// prefix (glm5 / dsv32); a missing parquet leaves its grid empty (the
    /// operator's missing-tables check then fails loud); within one file the
    /// last row wins for duplicate coordinates (pandas `iterrows` overwrite).
    /// Results are cached per (file_prefix, num_heads).
    pub fn load_cp_sparse(
        &self,
        architecture: &str,
        num_heads: u32,
    ) -> Result<Arc<DsaSparseTables>, AicError> {
        let file_prefix = dsa_sparse_file_prefix(architecture);
        let key = (file_prefix.to_string(), num_heads);
        if let Some(tables) = self.sparse.lock().expect("dsa sparse cache poisoned").get(&key) {
            return Ok(Arc::clone(tables));
        }
        let mut tables = DsaSparseTables::default();
        // mqa logits: every row goes to `mqa`.
        load_sparse_parquet(
            &resolve_op_sources(
                &self.perf_db_sources,
                &format!("{file_prefix}_mqa_logits_module_perf.parquet"),
                &self.data_root,
            ),
            num_heads,
            |_| SparseKind::Mqa,
            &mut tables,
        )?;
        // topk: split by `score_mode` — "flat" -> topk_flat, else topk_last
        // (Python: `"topk_flat" if str(r.get("score_mode","")) == "flat" else "topk_last"`).
        load_sparse_parquet(
            &resolve_op_sources(
                &self.perf_db_sources,
                &format!("{file_prefix}_topk_module_perf.parquet"),
                &self.data_root,
            ),
            num_heads,
            |score_mode| {
                if score_mode == Some("flat") {
                    SparseKind::TopkFlat
                } else {
                    SparseKind::TopkLast
                }
            },
            &mut tables,
        )?;
        // dsa_attn: optional, not used by the CP delta.
        load_sparse_parquet(
            &resolve_op_sources(
                &self.perf_db_sources,
                &format!("{file_prefix}_dsa_attn_module_perf.parquet"),
                &self.data_root,
            ),
            num_heads,
            |_| SparseKind::DsaAttn,
            &mut tables,
        )?;
        let arc = Arc::new(tables);
        Ok(Arc::clone(
            self.sparse
                .lock()
                .expect("dsa sparse cache poisoned")
                .entry(key)
                .or_insert(arc),
        ))
    }

    /// Context-DSA module latency for the sparse-attention block.
    ///
    /// Mirrors Python `ContextDSAModule._query_context_dsa_module_table`
    /// (SILICON path): one 4-axis Grid RAW engine query on the raw
    /// `[num_heads][prefix][seq][batch]` table, evaluated at `isl` (the
    /// new-token count), NOT `isl + prefix`. `index_topk` feeds the analytic
    /// SOL (indexer on/off regime + sparse-KV pair count); the top-k
    /// piecewise interpolation layer that used to consume it was deleted in
    /// v2 alongside the robust-lookup/batch-scaling fallbacks.
    ///
    /// `dsa_backend` selects the per-backend slice with Python
    /// `_select_dsa_backend`'s fallback chain. Python's op passes `"trtllm"`
    /// on the plain path and `"flashmla_kv"` for the CP base.
    #[allow(clippy::too_many_arguments)]
    pub fn query_context(
        &self,
        spec: &SystemSpec,
        b: u32,
        isl: u32,
        num_heads: u32,
        kv_quant: KvCacheQuantMode,
        fmha_quant: FmhaQuantMode,
        gemm_quant: GemmQuantMode,
        architecture: &str,
        prefix: u32,
        index_topk: u32,
        dsa_backend: &str,
        skip_indexer: bool,
    ) -> Result<f64, AicError> {
        // `skip_indexer=true` reads the GLM-5.2 reuse-layer table (rows tagged
        // `*_skip_indexer` in the same parquet) and zeroes the indexer terms
        // in the SOL — mirroring Python `_query_context_dsa_module_table`.
        let nodes = if skip_indexer {
            self.load_context_skip_nodes()?
        } else {
            self.load_context_nodes()?
        };
        let key = DsaKey {
            architecture: architecture.to_string(),
            fmha_quant: fmha_quant.name().to_string(),
            kv_quant: kv_quant.name().to_string(),
            gemm_quant: gemm_quant.name().to_string(),
        };
        let node = nodes
            .by_keys
            .get(&key)
            .and_then(|by_backend| select_dsa_backend(by_backend, dsa_backend))
            .ok_or_else(|| {
                AicError::PerfDatabase(format!("context DSA module data missing for {key:?}"))
            })?;

        let dims = dsa_dims(architecture);
        let topk = index_topk as i64;
        // Engine coordinates are (num_heads, prefix, seq, batch); the SOL is
        // Python's get_sol(b, s, prefix, num_heads, ...) re-ordered to match.
        let sol = move |c: &[f64]| {
            dsa_context_sol_ms(
                spec,
                dims,
                topk,
                kv_quant,
                fmha_quant,
                gemm_quant,
                c[3] as i64, // b
                c[2] as i64, // s
                c[1] as i64, // prefix
                c[0] as i64, // num_heads
                skip_indexer,
            )
        };
        let cfg = OpInterpConfig::grid(&["num_heads", "prefix", "seq_len", "batch"], &sol);
        perf_interp::query(
            &cfg,
            node,
            &[num_heads as f64, prefix as f64, isl as f64, b as f64],
        )
    }

    /// Raw generation-DSA module latency. `sequence_tokens = isl + step`
    /// from the CSV. Mirrors Python `GenerationDSAModule` (SILICON path):
    /// one 3-axis Grid RAW engine query with Python's
    /// `generation_grid_config` axis order `(num_heads, batch, seq)` —
    /// batch is the MIDDLE axis (Python's generation loader nests
    /// `[num_heads][b][s]`), the derived cache here matches that order.
    /// Out-of-range seq/batch is util-hold on the decode SOL.
    ///
    /// `dsa_backend` selects the per-backend slice (Python's op passes the
    /// table function default, `"trtllm"`).
    #[allow(clippy::too_many_arguments)]
    pub fn query_generation(
        &self,
        spec: &SystemSpec,
        b: u32,
        sequence_tokens: u32,
        num_heads: u32,
        kv_quant: KvCacheQuantMode,
        fmha_quant: FmhaQuantMode,
        gemm_quant: GemmQuantMode,
        architecture: &str,
        dsa_backend: &str,
        skip_indexer: bool,
    ) -> Result<f64, AicError> {
        // `skip_indexer=true` reads the GLM-5.2 reuse-layer generation table.
        // The generation SOL is skip-independent (Python's generation get_sol
        // has no skip branch) — only the table slice differs.
        let nodes = if skip_indexer {
            self.load_generation_skip_nodes()?
        } else {
            self.load_generation_nodes()?
        };
        // NOTE: Python's generation table is keyed (kv, gemm, arch) only —
        // no mla_dtype axis. The Rust key retains `fmha_quant` from the
        // parquet `mla_dtype` column (uniformly `bfloat16` in collected
        // generation files today), so callers must pass the collected value.
        let key = DsaKey {
            architecture: architecture.to_string(),
            fmha_quant: fmha_quant.name().to_string(),
            kv_quant: kv_quant.name().to_string(),
            gemm_quant: gemm_quant.name().to_string(),
        };
        let node = nodes
            .by_keys
            .get(&key)
            .and_then(|by_backend| select_dsa_backend(by_backend, dsa_backend))
            .ok_or_else(|| missing("generation DSA module", &self.data_root, format!("{key:?}")))?;

        let dims = dsa_dims(architecture);
        // Engine coordinates are (num_heads, batch, seq); the SOL is
        // Python's get_sol(b, s, num_heads, kv_cache_dtype) re-ordered.
        let sol = move |c: &[f64]| {
            dsa_generation_sol_ms(
                spec,
                dims,
                kv_quant,
                gemm_quant,
                c[1] as i64, // b
                c[2] as i64, // s
                c[0] as i64, // num_heads
            )
        };
        let cfg = OpInterpConfig::grid(&["num_heads", "batch", "seq_len"], &sol);
        perf_interp::query(
            &cfg,
            node,
            &[num_heads as f64, b as f64, sequence_tokens as f64],
        )
    }

    fn load_context(&self) -> Result<&DsaGrids, AicError> {
        let cell = self
            .context
            .get_or_init(|| load_dsa_parquet(&self.context_sources, false, false));
        cell.as_ref().map_err(clone_err)
    }

    fn load_context_skip(&self) -> Result<&DsaGrids, AicError> {
        let cell = self
            .context_skip
            .get_or_init(|| load_dsa_parquet(&self.context_sources, false, true));
        cell.as_ref().map_err(clone_err)
    }

    fn load_generation(&self) -> Result<&DsaGrids, AicError> {
        // Collapse (isl, step) -> seq at LOAD time, in file-row order, so
        // same-seq ties resolve exactly like Python's generation loader
        // (last file row wins within a source; first source wins across).
        let cell = self
            .generation
            .get_or_init(|| load_dsa_parquet(&self.generation_sources, true, false));
        cell.as_ref().map_err(clone_err)
    }

    fn load_generation_skip(&self) -> Result<&DsaGrids, AicError> {
        // Same load-time (isl, step) -> seq collapse as the full table.
        let cell = self
            .generation_skip
            .get_or_init(|| load_dsa_parquet(&self.generation_sources, true, true));
        cell.as_ref().map_err(clone_err)
    }

    fn load_context_nodes(&self) -> Result<&NodeCache, AicError> {
        let cell = self.context_nodes.get_or_init(|| {
            let grids = self.load_context()?;
            Ok(build_context_nodes(grids))
        });
        cell.as_ref().map_err(clone_err)
    }

    fn load_context_skip_nodes(&self) -> Result<&NodeCache, AicError> {
        let cell = self.context_skip_nodes.get_or_init(|| {
            let grids = self.load_context_skip()?;
            Ok(build_context_nodes(grids))
        });
        cell.as_ref().map_err(clone_err)
    }

    fn load_generation_nodes(&self) -> Result<&NodeCache, AicError> {
        let cell = self.generation_nodes.get_or_init(|| {
            let grids = self.load_generation()?;
            Ok(build_generation_nodes(grids))
        });
        cell.as_ref().map_err(clone_err)
    }

    fn load_generation_skip_nodes(&self) -> Result<&NodeCache, AicError> {
        let cell = self.generation_skip_nodes.get_or_init(|| {
            let grids = self.load_generation_skip()?;
            Ok(build_generation_nodes(grids))
        });
        cell.as_ref().map_err(clone_err)
    }
}

/// Materialise the per-`(DsaKey, dsa_backend)` engine table for
/// `query_context` with the raw nesting `[num_heads][step][isl][batch]` (the
/// 4-axis grid Python v2 resolves on).
fn build_context_nodes(grids: &DsaGrids) -> NodeCache {
    let mut by_keys: BTreeMap<DsaKey, BTreeMap<String, Node>> = BTreeMap::new();
    for (key, by_backend) in &grids.by_keys {
        let backends = by_keys.entry(key.clone()).or_default();
        for (backend, by_heads) in by_backend {
            let node = backends
                .entry(backend.clone())
                .or_insert_with(Node::branch);
            for (&n, by_step) in by_heads {
                for (&step, by_isl) in by_step {
                    for (&isl, by_batch) in by_isl {
                        for (&bb, &lat) in by_batch {
                            node.insert(&[n, step, isl, bb], lat);
                        }
                    }
                }
            }
        }
    }
    NodeCache { by_keys }
}

/// Materialise the per-`(DsaKey, dsa_backend)` engine table for
/// `query_generation` with the Python v2 generation nesting
/// `[num_heads][batch][seq = isl + step]`.
/// The (isl, step) -> seq collapse happens at LOAD time in file-row order
/// (`load_dsa_parquet` with `collapse_isl_step_to_seq`), mirroring Python's
/// per-row overwrite; the grids reaching here carry `step = 0, isl = seq`,
/// so `seq = isl + step` below is the identity and no tie remains to break.
fn build_generation_nodes(grids: &DsaGrids) -> NodeCache {
    let mut by_keys: BTreeMap<DsaKey, BTreeMap<String, Node>> = BTreeMap::new();
    for (key, by_backend) in &grids.by_keys {
        let backends = by_keys.entry(key.clone()).or_default();
        for (backend, by_heads) in by_backend {
            let node = backends
                .entry(backend.clone())
                .or_insert_with(Node::branch);
            for (&n, by_step) in by_heads {
                for (&step, by_isl) in by_step {
                    for (&isl, by_batch) in by_isl {
                        let seq = isl + step;
                        for (&bb, &lat) in by_batch {
                            node.insert(&[n, bb, seq], lat);
                        }
                    }
                }
            }
        }
    }
    NodeCache { by_keys }
}

// ---------------------------------------------------------------------------
// Analytic SOLs — verbatim ports of Python `operations/dsa.py` get_sol
// ---------------------------------------------------------------------------

/// Python `GEMM._get_quant_tc_flops`: compute factor 1 -> bf16 TC flops,
/// 2 -> fp8, 4 -> fp4; fall back to `bf16 * factor` when the spec entry is
/// missing.
fn tc_flops(spec: &SystemSpec, compute_factor: f64) -> f64 {
    let bf16 = spec.gpu.bfloat16_tc_flops.unwrap_or(0.0);
    let direct = match compute_factor as u32 {
        1 => spec.gpu.bfloat16_tc_flops,
        2 => spec.gpu.fp8_tc_flops,
        4 => spec.gpu.fp4_tc_flops,
        _ => None,
    };
    direct.unwrap_or(bf16 * compute_factor)
}

/// Python `common.indexer_cache_entry_bytes`: FP8 indexer KV entry with one
/// 4-byte scale per 128 values.
fn indexer_cache_entry_bytes(index_head_dim: i64) -> i64 {
    index_head_dim + ((index_head_dim + 127) / 128) * 4
}

/// Context DSA analytic roofline. Verbatim port of Python
/// `ContextDSAModule._query_context_dsa_module_table::get_sol` with
/// `skip_indexer=False` (the Rust table has no GLM-5.2 skip-indexer split;
/// collected files carry only full rows).
///
/// Ops split into a GEMM group (gemm_quant), the always-FP8 indexer-logits
/// group (active only when `full_s > index_topk`), and the sparse-MLA
/// attention group (fmha_quant) whose exact KV pair count is
/// `sum_{i=0..s-1} min(prefix+i+1, index_topk)`.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
fn dsa_context_sol_ms(
    spec: &SystemSpec,
    dims: &DsaDims,
    index_topk: i64,
    kv_quant: KvCacheQuantMode,
    fmha_quant: FmhaQuantMode,
    gemm_quant: GemmQuantMode,
    b: i64,
    s: i64,
    prefix: i64,
    num_heads: i64,
    skip_indexer: bool,
) -> f64 {
    let (hidden, q_lora, kv_lora) = (dims.hidden_size, dims.q_lora_rank, dims.kv_lora_rank);
    let (inh, ihd) = (dims.index_n_heads, dims.index_head_dim);
    let qk_head_dim = dims.qk_nope_head_dim + dims.qk_rope_head_dim;
    let attn_head_dim = kv_lora + dims.qk_rope_head_dim;
    let v_dim = dims.v_head_dim;

    let (b, s, prefix, num_heads) = (b as i128, s as i128, prefix as i128, num_heads as i128);
    let (hidden, q_lora, kv_lora) = (hidden as i128, q_lora as i128, kv_lora as i128);
    let (inh, ihd, topk) = (inh as i128, ihd as i128, index_topk as i128);
    let (qk_head_dim, attn_head_dim, v_dim) = (qk_head_dim as i128, attn_head_dim as i128, v_dim as i128);
    let (qk_nope, qk_rope) = (dims.qk_nope_head_dim as i128, dims.qk_rope_head_dim as i128);

    let full_s = s + prefix;
    let tokens = b * s;

    // ── Compute (FLOPs) ─────────────────────────────────────────
    let proj_out = q_lora + kv_lora + qk_rope + ihd;
    let gemm_group_ops = 2 * tokens * hidden * proj_out
        + 2 * tokens * q_lora * (num_heads * qk_head_dim)
        + 2 * tokens * q_lora * (inh * ihd)
        + 2 * tokens * hidden * inh
        + 2 * tokens * (num_heads * v_dim) * hidden
        + 2 * num_heads * tokens * qk_nope * kv_lora
        + 2 * num_heads * tokens * kv_lora * v_dim;

    // Indexer logits group: always FP8; off when the full sequence fits the
    // top-k window (regime split). A skip-indexer (reuse) layer never runs
    // the per-layer indexer, so the group is zero regardless of full_s
    // (Python operations/dsa.py get_sol).
    let indexer_logits_ops = if skip_indexer || full_s <= topk {
        0
    } else {
        2 * tokens * inh * ihd * full_s
    };

    // Sparse MLA attention group. Exact KV pair count:
    // sum_{i=0..s-1} min(prefix+i+1, topk).
    let effective_kv = full_s.min(topk);
    let total_kv_pairs = if full_s <= topk {
        b * (full_s * (full_s + 1) - prefix * (prefix + 1)) / 2
    } else if prefix >= topk {
        tokens * topk
    } else {
        let ramp = b * (topk * (topk + 1) - prefix * (prefix + 1)) / 2;
        let sat = b * (full_s - topk) * topk;
        ramp + sat
    };
    let sparse_attn_ops = 2 * num_heads * (attn_head_dim + kv_lora) * total_kv_pairs;

    // ── Memory (bytes) ──────────────────────────────────────────
    let gemm_weight_elems = hidden * proj_out
        + q_lora * num_heads * qk_head_dim
        + q_lora * inh * ihd
        + hidden * inh
        + num_heads * v_dim * hidden;
    let gemm_weight_bytes = gemm_weight_elems as f64 * gemm_quant.mapping().memory;

    let kv_cache_bytes =
        (b * num_heads * effective_kv * attn_head_dim) as f64 * kv_quant.mapping().memory;
    // Skip layers never store the index-K cache (the indexer never runs).
    let indexer_cache_bytes = if skip_indexer || full_s <= topk {
        0.0
    } else {
        (b * full_s * indexer_cache_entry_bytes(dims.index_head_dim) as i128) as f64
    };
    let q_io_bytes = (tokens * num_heads * qk_head_dim) as f64 * fmha_quant.mapping().memory * 2.0;

    let total_mem = gemm_weight_bytes + kv_cache_bytes + indexer_cache_bytes + q_io_bytes;

    // ── SOL ─────────────────────────────────────────────────────
    let gemm_flops = tc_flops(spec, gemm_quant.mapping().compute);
    let indexer_fp8_flops = tc_flops(spec, FmhaQuantMode::Fp8.mapping().compute);
    let attn_flops = tc_flops(spec, fmha_quant.mapping().compute);

    let sol_math = (gemm_group_ops as f64 / gemm_flops
        + indexer_logits_ops as f64 / indexer_fp8_flops
        + sparse_attn_ops as f64 / attn_flops)
        * 1000.0;
    let sol_mem = total_mem / spec.gpu.mem_bw * 1000.0;
    sol_math.max(sol_mem)
}

/// Generation DSA analytic roofline. Verbatim port of Python
/// `GenerationDSAModule._query_generation_dsa_module_table::get_sol`
/// (1 token per request; the attention group is hardcoded bfloat16 in
/// Python — `fmha_mode = FMHAQuantMode.bfloat16` — so no fmha arg here).
fn dsa_generation_sol_ms(
    spec: &SystemSpec,
    dims: &DsaDims,
    kv_quant: KvCacheQuantMode,
    gemm_quant: GemmQuantMode,
    b: i64,
    s: i64,
    num_heads: i64,
) -> f64 {
    let (b, s, num_heads) = (b as i128, s as i128, num_heads as i128);
    let (hidden, q_lora, kv_lora) = (
        dims.hidden_size as i128,
        dims.q_lora_rank as i128,
        dims.kv_lora_rank as i128,
    );
    let (inh, ihd, topk) = (
        dims.index_n_heads as i128,
        dims.index_head_dim as i128,
        dims.index_topk as i128,
    );
    let (qk_nope, qk_rope, v_dim) = (
        dims.qk_nope_head_dim as i128,
        dims.qk_rope_head_dim as i128,
        dims.v_head_dim as i128,
    );
    let qk_head_dim = qk_nope + qk_rope;
    let attn_head_dim = kv_lora + qk_rope;

    let tokens = b;
    let proj_out = q_lora + kv_lora + qk_rope + ihd;
    let effective_kv = s.min(topk);

    let gemm_group_ops = 2 * tokens * hidden * proj_out
        + 2 * tokens * q_lora * num_heads * qk_head_dim
        + 2 * tokens * q_lora * inh * ihd
        + 2 * tokens * hidden * inh
        + 2 * tokens * num_heads * v_dim * hidden
        + 2 * num_heads * tokens * qk_nope * kv_lora
        + 2 * num_heads * tokens * kv_lora * v_dim;

    let indexer_logits_ops = 2 * tokens * inh * ihd * s;
    let sparse_attn_ops = 2 * tokens * num_heads * (attn_head_dim + kv_lora) * effective_kv;

    let gemm_weight_elems = hidden * proj_out
        + q_lora * num_heads * qk_head_dim
        + q_lora * inh * ihd
        + hidden * inh
        + num_heads * v_dim * hidden;
    let gemm_weight_bytes = gemm_weight_elems as f64 * gemm_quant.mapping().memory;
    let indexer_cache_bytes =
        (b * s * indexer_cache_entry_bytes(dims.index_head_dim) as i128) as f64;
    let kv_cache_bytes = (b * effective_kv * attn_head_dim) as f64 * kv_quant.mapping().memory;
    let total_mem = gemm_weight_bytes + indexer_cache_bytes + kv_cache_bytes;

    let gemm_flops = tc_flops(spec, gemm_quant.mapping().compute);
    let indexer_fp8_flops = tc_flops(spec, FmhaQuantMode::Fp8.mapping().compute);
    let attn_flops = tc_flops(spec, FmhaQuantMode::Bfloat16.mapping().compute);

    let sol_math = (gemm_group_ops as f64 / gemm_flops
        + indexer_logits_ops as f64 / indexer_fp8_flops
        + sparse_attn_ops as f64 / attn_flops)
        * 1000.0;
    let sol_mem = total_mem / spec.gpu.mem_bw * 1000.0;
    sol_math.max(sol_mem)
}

/// Load a DSA module table from an ordered, priority-sorted source list, with
/// Python's two-phase duplicate policy (`operations/dsa.py:1461-1502`):
///
/// 1. **Within a source: last row wins.** Each source is collapsed into a
///    per-source coordinate map with plain overwrite, mirroring Python's
///    "Preserve legacy last-row-wins behavior within each source"
///    (`source_values[coordinate] = value`).
/// 2. **Across sources: first source wins.** Sources are priority-ordered
///    (active stack first, shared fallbacks later); a coordinate already
///    populated by an earlier source is skipped (`seen_coordinates` guard in
///    Python, `or_insert` here).
///
/// Row selection and keying mirror the same Python loaders:
/// - **op_name split** (Python `op_kind`): rows whose `op_name` contains
///   `skip_indexer` (the GLM-5.2 reuse-layer table sharing this file) are
///   DROPPED — this table is Python's `op_kind="full"` slice, the one every
///   ported query path consumes (`skip_indexer=False`). A missing `op_name`
///   column keeps every row (Python `row.get("op_name") or ""`).
/// - **dsa_backend keying**: derived per row from `kernel_source`
///   (`"trtllm" if "trtllm" in ks else "flashmla_kv"`; missing column →
///   `flashmla_kv`) and made a nesting level of the grid, so trtllm and
///   flashmla_kv measurements of the same shape never collapse into one.
///
/// Missing files are skipped (a sibling declared in the manifest need not
/// exist for every system); an error is returned only when no source yields
/// rows. Shared by both the context and generation DSA files.
/// `collapse_isl_step_to_seq` selects the generation-table coordinate:
/// Python's `load_generation_dsa_module_data` keys rows by `s = isl + step`
/// ("Total decode length is the canonical coordinate even if two rows
/// decompose it into different isl/step pairs"), so both the within-source
/// last-row-wins overwrite AND the cross-source first-wins guard operate at
/// the COLLAPSED coordinate, and the winner among same-`s` rows is the last
/// row in FILE order — not any sorted (step, isl) order. Collapsing here,
/// before the phase-1 insert, reproduces both. The collapsed grid stores
/// `step = 0, isl = s` in the shared `DsaHeadGrid` shape. Context keeps the
/// raw `(step, isl)` coordinate (`load_context_dsa_module_data` keys on it
/// exactly).
fn load_dsa_parquet(
    sources: &[PerfSource],
    collapse_isl_step_to_seq: bool,
    want_skip_rows: bool,
) -> Result<DsaGrids, AicError> {
    let mut by_keys: BTreeMap<DsaKey, BTreeMap<String, DsaHeadGrid>> = BTreeMap::new();
    let mut any_source = false;
    for source in sources {
        let path = source.path();
        if !path.exists() {
            continue;
        }
        any_source = true;
        let reader = PerfReader::open(path)?;
        let arch_col = reader.col("architecture")?;
        let mla_dtype_col = reader.col("mla_dtype")?;
        let kv_cache_dtype_col = reader.col("kv_cache_dtype")?;
        let gemm_type_col = reader.col("gemm_type")?;
        let num_heads_col = reader.col("num_heads")?;
        let batch_size_col = reader.col("batch_size")?;
        let isl_col = reader.col("isl")?;
        let step_col = reader.col("step")?;
        let latency_col = reader.col("latency")?;
        // Optional columns: legacy files may lack them (Python `row.get`).
        let op_name_col = reader.col_optional("op_name");
        let ks_col = reader.col_optional("kernel_source");
        // Phase 1: collapse this source with last-row-wins (plain insert).
        let mut source_values: BTreeMap<(DsaKey, String, u32, u32, u32, u32), f64> =
            BTreeMap::new();
        for row in reader.rows()? {
            let row = row?;
            if !kernel_source_ok(source.kernel_sources(), ks_col, &row)? {
                continue;
            }
            // Full vs skip-indexer share one file, split by op_name (Python:
            // `if ("skip_indexer" in (row.get("op_name") or "")) != (op_kind
            // == "skip"): continue`).
            let is_skip_row = row
                .str_optional(op_name_col)?
                .unwrap_or("")
                .contains("skip_indexer");
            if is_skip_row != want_skip_rows {
                continue;
            }
            let key = DsaKey {
                architecture: row.str_owned(arch_col)?,
                fmha_quant: row.str_owned(mla_dtype_col)?,
                kv_quant: row.str_owned(kv_cache_dtype_col)?,
                gemm_quant: row.str_owned(gemm_type_col)?,
            };
            // Configured-backend bucket(s) for the row — mirrors Python
            // `_dsa_kernel_source_buckets` (operations/dsa.py). The
            // trtllm/flashmla_kv split is an FP8-KV serving rule; with a BF16
            // KV cache there is exactly ONE real execution path per shape, so
            // every bf16 row backs BOTH buckets (a bare substring test split
            // one measured sweep across the buckets and left the default
            // query bucket with nothing beyond 2048 tokens). FP8 rows bucket
            // by executed-kernel name; dense ragged prefill is selected by
            // SHAPE (isl <= 2048) under either configured backend, so it
            // backs both. Legacy (pre-0.5.14) names keep the old substring
            // rule.
            let ks_name = row.str_optional(ks_col)?.unwrap_or("").to_string();
            let buckets: &[&str] = if key.kv_quant == "bfloat16" {
                &["trtllm", "flashmla_kv"]
            } else {
                match ks_name.as_str() {
                    "sglang_dsa_indexer_trtllm" | "sglang_dsa_skip_indexer_trtllm" => &["trtllm"],
                    "sglang_dsa_indexer_flashmla_sparse"
                    | "sglang_dsa_skip_indexer_flashmla_sparse" => &["flashmla_kv"],
                    "sglang_dsa_dense_mha_trtllm_ragged" => &["trtllm", "flashmla_kv"],
                    _ if ks_name.contains("trtllm") => &["trtllm"],
                    _ => &["flashmla_kv"],
                }
            };
            let (step, isl) = if collapse_isl_step_to_seq {
                // Generation: canonical coordinate is s = isl + step (see fn
                // doc); same-`s` rows overwrite in file order below.
                (0, row.u32(isl_col)? + row.u32(step_col)?)
            } else {
                (row.u32(step_col)?, row.u32(isl_col)?)
            };
            let num_heads = row.u32(num_heads_col)?;
            let batch_size = row.u32(batch_size_col)?;
            let latency = row.f64(latency_col)?;
            for dsa_backend in buckets {
                source_values.insert(
                    (key.clone(), dsa_backend.to_string(), num_heads, step, isl, batch_size),
                    latency,
                );
            }
        }
        // Phase 2: merge with first-source-wins (earlier source outranks).
        for ((key, dsa_backend, num_heads, step, isl, batch_size), latency) in source_values {
            by_keys
                .entry(key)
                .or_default()
                .entry(dsa_backend)
                .or_default()
                .entry(num_heads)
                .or_default()
                .entry(step)
                .or_default()
                .entry(isl)
                .or_default()
                .entry(batch_size)
                .or_insert(latency);
        }
    }
    if !any_source || by_keys.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "no DSA module rows loaded from {} source(s) (first: {})",
            sources.len(),
            sources.first().map(|s| s.path().display().to_string()).unwrap_or_default()
        )));
    }
    Ok(DsaGrids { by_keys })
}

/// Destination grid selector for one CP sparse parquet row, decided from the
/// row's `score_mode` (present only in the topk file).
enum SparseKind {
    Mqa,
    TopkLast,
    TopkFlat,
    DsaAttn,
}

/// Load one CP sparse sub-kernel parquet into `tables`. Mirrors Python
/// `_load_glm5_sparse`'s per-file body:
/// - missing file => no-op (grid stays empty; fail-loud happens at the
///   composition's missing-tables check);
/// - `num_heads` filter only when the column exists
///   (`df[df["num_heads"] == num_heads] if "num_heads" in df else df`);
/// - within a source, LAST row wins for duplicate `(bs, isl, step)`
///   coordinates (pandas `iterrows` overwrite); across shared-layer sources
///   the earlier (higher-priority) source wins, like every other multi-source
///   loader here (Python reads primary-only, which the default single-source
///   resolution reproduces exactly).
fn load_sparse_parquet(
    sources: &[PerfSource],
    num_heads: u32,
    kind_of: impl Fn(Option<&str>) -> SparseKind,
    tables: &mut DsaSparseTables,
) -> Result<(), AicError> {
    for source in sources {
        let path = source.path();
        if !path.exists() {
            continue;
        }
        let reader = PerfReader::open(path)?;
        let batch_size_col = reader.col("batch_size")?;
        let isl_col = reader.col("isl")?;
        let step_col = reader.col("step")?;
        let latency_col = reader.col("latency")?;
        let num_heads_col = reader.col_optional("num_heads");
        let score_mode_col = reader.col_optional("score_mode");
        let ks_col = reader.col_optional("kernel_source");
        // Phase 1: collapse this source with last-row-wins (plain insert).
        let mut source_values: BTreeMap<(u8, u32, u32, u32), f64> = BTreeMap::new();
        for row in reader.rows()? {
            let row = row?;
            if !kernel_source_ok(source.kernel_sources(), ks_col, &row)? {
                continue;
            }
            if let Some(col) = num_heads_col {
                if row.u32(col)? != num_heads {
                    continue;
                }
            }
            let kind = kind_of(row.str_optional(score_mode_col)?);
            source_values.insert(
                (
                    kind as u8,
                    row.u32(batch_size_col)?,
                    row.u32(isl_col)?,
                    row.u32(step_col)?,
                ),
                row.f64(latency_col)?,
            );
        }
        // Phase 2: merge with first-source-wins (earlier source outranks).
        for ((kind, bs, isl, step), latency) in source_values {
            let grid = match kind {
                k if k == SparseKind::Mqa as u8 => &mut tables.mqa,
                k if k == SparseKind::TopkLast as u8 => &mut tables.topk_last,
                k if k == SparseKind::TopkFlat as u8 => &mut tables.topk_flat,
                _ => &mut tables.dsa_attn,
            };
            grid.entry(bs)
                .or_default()
                .entry((isl, step))
                .or_insert(latency);
        }
    }
    Ok(())
}

/// Pick the collected-batch slice nearest to `b` from a bs-keyed sparse grid.
/// Mirror of Python `ContextDSAModule._bs_slice`: exact match when collected
/// (the common case), otherwise the nearest collected batch. `None` for an
/// empty grid (Python returns `{}`; `lookup_2d` on it yields `None`). On an
/// exact-distance tie the SMALLER collected batch wins (ascending iteration,
/// strict `<`; Python's tie order follows file row order and is not defined).
pub fn bs_slice(by_bs: &SparseGrid, b: u32) -> Option<&BTreeMap<(u32, u32), f64>> {
    if let Some(exact) = by_bs.get(&b) {
        return Some(exact);
    }
    let mut best: Option<(u64, &BTreeMap<(u32, u32), f64>)> = None;
    for (&bs, grid) in by_bs {
        let dist = (i64::from(bs) - i64::from(b)).unsigned_abs();
        if best.map_or(true, |(d, _)| dist < d) {
            best = Some((dist, grid));
        }
    }
    best.map(|(_, grid)| grid)
}

/// Lookup a `{(isl, step) -> latency}` sparse grid at a fixed isl (exact grid
/// value, else nearest collected isl), linear interp on step (clamped at the
/// collected step range). Mirror of Python `ContextDSAModule._lookup_2d`,
/// including the fail-loud contract: an `isl` beyond the collected grid RAISES
/// (mqa/topk scale super-linearly with isl — clamping would silently
/// under-estimate). Empty table => `Ok(None)`.
pub fn lookup_2d(
    table: &BTreeMap<(u32, u32), f64>,
    isl: u32,
    step: u32,
) -> Result<Option<f64>, AicError> {
    if table.is_empty() {
        return Ok(None);
    }
    let max_isl = table.keys().map(|&(i, _)| i).max().expect("non-empty");
    if isl > max_isl {
        return Err(AicError::PerfDatabase(format!(
            "DSA CP: isl={isl} exceeds the collected sparse-kernel grid \
             (max isl={max_isl}); mqa/topk scale super-linearly with isl, so \
             clamping the isl axis would silently under-estimate. Re-collect with \
             AIC_CHUNKED_PREFILL_SIZE >= {isl} \
             (docs/CONTEXT_PARALLEL_DSA_MODELING.md \u{a7}9.1)."
        )));
    }
    // Nearest collected isl; ties pick the smaller (Python `min` over the
    // sorted isl list keeps the first == smaller candidate).
    let mut use_isl = None;
    for &(i, _) in table.keys() {
        let dist = (i64::from(i) - i64::from(isl)).unsigned_abs();
        if use_isl.map_or(true, |(d, _)| dist < d) {
            use_isl = Some((dist, i));
        }
    }
    let use_isl = use_isl.expect("non-empty").1;
    if let Some(&exact) = table.get(&(use_isl, step)) {
        return Ok(Some(exact));
    }
    let steps: Vec<u32> = table
        .range((use_isl, u32::MIN)..=(use_isl, u32::MAX))
        .map(|(&(_, st), _)| st)
        .collect();
    let Some((&first, &last)) = steps.first().zip(steps.last()) else {
        return Ok(None);
    };
    let lo = steps.iter().rev().find(|&&st| st <= step).copied().unwrap_or(first);
    let hi = steps.iter().find(|&&st| st >= step).copied().unwrap_or(last);
    if lo == hi {
        return Ok(Some(table[&(use_isl, lo)]));
    }
    let a = table[&(use_isl, lo)];
    let b = table[&(use_isl, hi)];
    Ok(Some(
        a + (b - a) * f64::from(step - lo) / f64::from(hi - lo),
    ))
}

fn missing(table: &str, data_root: &Path, descriptor: String) -> AicError {
    AicError::PerfDatabase(format!(
        "{table} data missing for {descriptor} at {}",
        data_root.display()
    ))
}

fn clone_err(err: &AicError) -> AicError {
    AicError::PerfDatabase(err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    const REPO_ROOT_HINT: &str = env!("CARGO_MANIFEST_DIR");

    fn b200_vllm_data_root() -> PathBuf {
        PathBuf::from(REPO_ROOT_HINT)
            .join("../..")
            .join("src/aiconfigurator/systems/data/b200_sxm/vllm/0.19.0")
    }

    fn b200_sxm_spec() -> SystemSpec {
        let systems_yaml = PathBuf::from(REPO_ROOT_HINT)
            .join("../..")
            .join("src/aiconfigurator/systems/b200_sxm.yaml");
        SystemSpec::load(&systems_yaml).expect("b200_sxm.yaml must parse")
    }

    const INDEX_TOPK: u32 = 2048;

    fn approx_rel(got: f64, want: f64) {
        assert!(
            ((got - want) / want).abs() < 1e-9,
            "rust {got} vs python {want}"
        );
    }

    #[test]
    fn dsa_context_module_exact_hit() {
        // First row of dsa_context_module_perf.txt:
        // arch=DeepseekV32ForCausalLM mla=bfloat16 kv=bfloat16 gemm=bfloat16
        // n=128 b=1 isl=1 step=0 latency=1.0972. Exact 4-axis hit — the
        // engine returns the measured leaf verbatim.
        // NOTE(shared-layer merge): oracle generated pre-shared-layer; regenerate if this fails.
        let table = DsaTable::new(b200_vllm_data_root());
        let spec = b200_sxm_spec();
        let latency = table
            .query_context(
                &spec,
                1,
                1,
                128,
                KvCacheQuantMode::Bfloat16,
                FmhaQuantMode::Bfloat16,
                GemmQuantMode::Bfloat16,
                "DeepseekV32ForCausalLM",
                0,
                INDEX_TOPK,
                "trtllm",
                false,
            )
            .expect("DSA context query must succeed");
        assert!(
            (latency - 1.0972).abs() < 1e-6,
            "expected recorded latency, got {latency}"
        );
    }

    /// Within-file duplicate policy: LAST row wins (Python two-phase loader,
    /// `operations/dsa.py:1461-1502`). The real b300_sxm/vllm/0.19.0
    /// `dsa_context_module_perf.parquet` carries 16k+ within-file duplicate
    /// coordinates; at (arch=DeepseekV32ForCausalLM, bf16/bf16/bf16, n=128,
    /// step=0, isl=8192, b=1) the first occurrence records 7.7643 and the
    /// last records 7.7560. Python oracle (7.756) generated with:
    ///
    /// ```text
    /// PYTHONPATH=src python3 -c "
    /// from aiconfigurator.sdk.perf_database import get_database
    /// from aiconfigurator.sdk import common
    /// db = get_database('b300_sxm', 'vllm', '0.19.0')
    /// r = db.query_context_dsa_module(
    ///     b=1, s=8192, num_heads=128,
    ///     kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
    ///     fmha_quant_mode=common.FMHAQuantMode.bfloat16,
    ///     gemm_quant_mode=common.GEMMQuantMode.bfloat16,
    ///     database_mode=common.DatabaseMode.SILICON,
    ///     prefix=0, architecture='DeepseekV32ForCausalLM',
    ///     dsa_backend='flashmla_kv')
    /// print(float(r))"
    /// ```
    ///
    /// The pre-fix per-row `or_insert` (first-row-wins) returned 7.7643 here.
    #[test]
    fn dsa_within_file_duplicates_last_row_wins() {
        let data_root = PathBuf::from(REPO_ROOT_HINT)
            .join("../..")
            .join("src/aiconfigurator/systems/data/b300_sxm/vllm/0.19.0");
        let systems_yaml = PathBuf::from(REPO_ROOT_HINT)
            .join("../..")
            .join("src/aiconfigurator/systems/b300_sxm.yaml");
        let spec = SystemSpec::load(&systems_yaml).expect("b300_sxm.yaml must parse");
        let table = DsaTable::new(data_root);
        let latency = table
            .query_context(
                &spec,
                1,
                8192,
                128,
                KvCacheQuantMode::Bfloat16,
                FmhaQuantMode::Bfloat16,
                GemmQuantMode::Bfloat16,
                "DeepseekV32ForCausalLM",
                0,
                INDEX_TOPK,
                "trtllm",
                false,
            )
            .expect("DSA context query must succeed");
        approx_rel(latency, 7.756);
    }

    #[test]
    fn dsa_unknown_architecture_errors() {
        let table = DsaTable::new(b200_vllm_data_root());
        let spec = b200_sxm_spec();
        let err = table
            .query_context(
                &spec,
                1,
                1024,
                128,
                KvCacheQuantMode::Bfloat16,
                FmhaQuantMode::Bfloat16,
                GemmQuantMode::Bfloat16,
                "NotAnArchitecture",
                0,
                INDEX_TOPK,
                "trtllm",
                false,
            )
            .unwrap_err();
        assert!(matches!(err, AicError::PerfDatabase(_)));
    }

    /// Cross-language parity with the Python v2 engine on the real
    /// b200_sxm/vllm/0.19.0 tables. Oracle values generated with
    /// `PYTHONPATH=src python3` via
    /// `PerfDatabase.query_context_dsa_module(..., DatabaseMode.SILICON)`
    /// (shared layer off so both sides read the same single parquet):
    /// exact hit / interior seq / interior batch / interior prefix (GLM) /
    /// seq util-hold / prefix util-hold.
    // NOTE(shared-layer merge): oracle generated pre-shared-layer; regenerate if this fails.
    #[test]
    fn dsa_context_matches_python_v2_engine() {
        let table = DsaTable::new(b200_vllm_data_root());
        let spec = b200_sxm_spec();
        let q = |b: u32, s: u32, prefix: u32, heads: u32, arch: &str| {
            table
                .query_context(
                    &spec,
                    b,
                    s,
                    heads,
                    KvCacheQuantMode::Bfloat16,
                    FmhaQuantMode::Bfloat16,
                    GemmQuantMode::Bfloat16,
                    arch,
                    prefix,
                    INDEX_TOPK,
                    "trtllm",
                    false,
                )
                .unwrap()
        };
        let dsv32 = "DeepseekV32ForCausalLM";
        let glm = "GlmMoeDsaForCausalLM";
        // exact 4-axis hit
        approx_rel(q(4, 2048, 0, 128, dsv32), 7.6471);
        // interior seq blend (2048 < 2560 < 3072)
        approx_rel(q(2, 2560, 0, 128, dsv32), 4.9806);
        // interior batch blend (2 < 3 < 4)
        approx_rel(q(3, 1024, 0, 128, dsv32), 3.0913);
        // interior prefix blend on the GLM step axis (0 < 64 < 128)
        approx_rel(q(1, 128, 64, 16, glm), 1.2492999999999999);
        // seq util-hold beyond the 32768 frontier (validates the context SOL)
        approx_rel(q(1, 65536, 0, 128, dsv32), 89.56218926395842);
        // prefix util-hold beyond the 128 step frontier
        approx_rel(q(1, 2048, 4096, 128, dsv32), 3.2580009866421995);
    }

    // ------------------------------------------------------------------
    // CP sparse sub-kernel tables (GLM-5/DSA sparse-CP prefill model)
    // ------------------------------------------------------------------

    fn grid(rows: &[(u32, u32, u32, f64)]) -> SparseGrid {
        let mut g = SparseGrid::new();
        for &(bs, isl, step, lat) in rows {
            g.entry(bs).or_default().insert((isl, step), lat);
        }
        g
    }

    /// Mirrors Python
    /// `tests/unit/sdk/test_cp_dsa_modeling.py::test_lookup_2d_exact_and_step_interp`.
    #[test]
    fn cp_lookup_2d_exact_step_interp_clamp_and_empty() {
        let t = grid(&[(1, 4096, 0, 100.0), (1, 4096, 1024, 200.0), (1, 8192, 0, 400.0)]);
        let t = &t[&1];
        assert_eq!(lookup_2d(t, 4096, 0).unwrap(), Some(100.0)); // exact grid point
        assert_eq!(lookup_2d(t, 4096, 512).unwrap(), Some(150.0)); // step interp
        assert_eq!(lookup_2d(t, 4096, 4096).unwrap(), Some(200.0)); // step clamp to max
        assert_eq!(lookup_2d(&BTreeMap::new(), 4096, 0).unwrap(), None); // empty table
    }

    /// Mirrors Python `test_lookup_2d_fails_loud_on_out_of_grid_isl`: isl
    /// beyond the collected grid must RAISE, not silently clamp.
    #[test]
    fn cp_lookup_2d_fails_loud_on_out_of_grid_isl() {
        let t = grid(&[(1, 4096, 0, 100.0), (1, 8192, 0, 400.0)]);
        let err = lookup_2d(&t[&1], 16384, 0).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("DSA CP: isl=16384 exceeds the collected sparse-kernel grid")
                && msg.contains("max isl=8192")
                && msg.contains("AIC_CHUNKED_PREFILL_SIZE >= 16384"),
            "unexpected message: {msg}"
        );
    }

    /// Python `_bs_slice`: exact match, else nearest collected batch; empty -> {}.
    #[test]
    fn cp_bs_slice_exact_nearest_and_empty() {
        let g = grid(&[(1, 2048, 0, 25.0), (8, 2048, 0, 90.0)]);
        assert_eq!(bs_slice(&g, 8).unwrap()[&(2048, 0)], 90.0); // exact
        assert_eq!(bs_slice(&g, 6).unwrap()[&(2048, 0)], 90.0); // nearest: |6-8| < |6-1|
        assert_eq!(bs_slice(&g, 3).unwrap()[&(2048, 0)], 25.0); // nearest: |3-1| < |3-8|
        assert!(bs_slice(&SparseGrid::new(), 1).is_none()); // empty grid
    }

    /// Absent sparse parquets => empty tables (Python `_read` -> None ->
    /// grids stay `{}`); the operator's missing-tables check fails loud on top.
    #[test]
    fn cp_sparse_absent_files_load_empty() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        let table = DsaTable::new(tmp.path().to_path_buf());
        let sparse = table
            .load_cp_sparse("GlmMoeDsaForCausalLM", 64)
            .expect("absent files are not a load error");
        assert_eq!(*sparse, DsaSparseTables::default());
    }

    /// Write one synthetic CP sparse parquet with the collector's column set
    /// (`num_heads, batch_size, isl, step, latency[, score_mode]`).
    fn write_sparse_parquet(path: &Path, with_score_mode: bool, rows: &[(i64, i64, i64, i64, f64, &str)]) {
        use parquet::data_type::{ByteArray, ByteArrayType, DoubleType, Int64Type};
        use parquet::file::properties::WriterProperties;
        use parquet::file::writer::SerializedFileWriter;
        use parquet::schema::parser::parse_message_type;

        let schema = if with_score_mode {
            "message schema {
                REQUIRED INT64 num_heads;
                REQUIRED INT64 batch_size;
                REQUIRED INT64 isl;
                REQUIRED INT64 step;
                REQUIRED DOUBLE latency;
                REQUIRED BINARY score_mode (UTF8);
            }"
        } else {
            "message schema {
                REQUIRED INT64 num_heads;
                REQUIRED INT64 batch_size;
                REQUIRED INT64 isl;
                REQUIRED INT64 step;
                REQUIRED DOUBLE latency;
            }"
        };
        let schema = Arc::new(parse_message_type(schema).expect("schema must parse"));
        let file = std::fs::File::create(path).expect("create parquet");
        let mut writer =
            SerializedFileWriter::new(file, schema, Arc::new(WriterProperties::builder().build()))
                .expect("writer");
        let mut rg = writer.next_row_group().expect("row group");
        let int_cols: [Vec<i64>; 4] = [
            rows.iter().map(|r| r.0).collect(),
            rows.iter().map(|r| r.1).collect(),
            rows.iter().map(|r| r.2).collect(),
            rows.iter().map(|r| r.3).collect(),
        ];
        for values in &int_cols {
            let mut col = rg.next_column().expect("next col").expect("int col");
            col.typed::<Int64Type>().write_batch(values, None, None).expect("write ints");
            col.close().expect("close col");
        }
        let latencies: Vec<f64> = rows.iter().map(|r| r.4).collect();
        let mut col = rg.next_column().expect("next col").expect("latency col");
        col.typed::<DoubleType>().write_batch(&latencies, None, None).expect("write latency");
        col.close().expect("close col");
        if with_score_mode {
            let modes: Vec<ByteArray> = rows.iter().map(|r| ByteArray::from(r.5)).collect();
            let mut col = rg.next_column().expect("next col").expect("score col");
            col.typed::<ByteArrayType>().write_batch(&modes, None, None).expect("write score");
            col.close().expect("close col");
        }
        rg.close().expect("close row group");
        writer.close().expect("close writer");
    }

    /// Loader parity with Python `ContextDSAModule._load_glm5_sparse` on the
    /// same synthetic rows: num_heads filter, last-row-wins within a file,
    /// score_mode "flat" -> topk_flat (else topk_last), bs-keyed grids,
    /// absent dsa_attn file -> empty grid. Expectation generated with:
    ///
    /// ```text
    /// PYTHONPATH=src python3 -c "
    /// import tempfile, os, types
    /// import pandas as pd
    /// from aiconfigurator.sdk.operations.dsa import ContextDSAModule
    /// tmp = tempfile.mkdtemp()
    /// data_dir = os.path.join(tmp, 'data', 'vllm', '1.0'); os.makedirs(data_dir)
    /// pd.DataFrame([
    ///     dict(num_heads=64, batch_size=1, isl=2048, step=0, latency=25.0),
    ///     dict(num_heads=64, batch_size=1, isl=16384, step=0, latency=1600.0),
    ///     dict(num_heads=64, batch_size=1, isl=16384, step=0, latency=1601.5),
    ///     dict(num_heads=32, batch_size=1, isl=2048, step=0, latency=99.0),
    ///     dict(num_heads=64, batch_size=2, isl=2048, step=128, latency=30.0),
    /// ]).to_parquet(os.path.join(data_dir, 'glm5_mqa_logits_module_perf.parquet'))
    /// pd.DataFrame([
    ///     dict(num_heads=64, batch_size=1, isl=16384, step=0, latency=800.0, score_mode='top_last'),
    ///     dict(num_heads=64, batch_size=1, isl=2048, step=0, latency=190.0, score_mode='top_last'),
    ///     dict(num_heads=64, batch_size=1, isl=2048, step=0, latency=100.0, score_mode='flat'),
    /// ]).to_parquet(os.path.join(data_dir, 'glm5_topk_module_perf.parquet'))
    /// db = types.SimpleNamespace(systems_root=tmp, system_spec={'data_dir': 'data'},
    ///     system='testsys', backend='vllm', version='1.0', enable_shared_layer=False)
    /// print(ContextDSAModule._load_glm5_sparse(db, 'GlmMoeDsaForCausalLM', 64)['_2d'])"
    /// # -> mqa       = {1: {(2048, 0): 25.0, (16384, 0): 1601.5}, 2: {(2048, 128): 30.0}}
    /// #    topk_last = {1: {(16384, 0): 800.0, (2048, 0): 190.0}}
    /// #    topk_flat = {1: {(2048, 0): 100.0}}
    /// #    dsa_attn  = {}
    /// ```
    #[test]
    fn cp_sparse_loader_matches_python_loader() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        write_sparse_parquet(
            &tmp.path().join("glm5_mqa_logits_module_perf.parquet"),
            false,
            &[
                (64, 1, 2048, 0, 25.0, ""),
                (64, 1, 16384, 0, 1600.0, ""),
                (64, 1, 16384, 0, 1601.5, ""), // duplicate coordinate: LAST row wins
                (32, 1, 2048, 0, 99.0, ""),    // filtered out (num_heads != 64)
                (64, 2, 2048, 128, 30.0, ""),  // second bs slice
            ],
        );
        write_sparse_parquet(
            &tmp.path().join("glm5_topk_module_perf.parquet"),
            true,
            &[
                (64, 1, 16384, 0, 800.0, "top_last"),
                (64, 1, 2048, 0, 190.0, "top_last"),
                (64, 1, 2048, 0, 100.0, "flat"),
            ],
        );
        let table = DsaTable::new(tmp.path().to_path_buf());
        let sparse = table
            .load_cp_sparse("GlmMoeDsaForCausalLM", 64)
            .expect("sparse tables must load");
        assert_eq!(
            sparse.mqa,
            grid(&[(1, 2048, 0, 25.0), (1, 16384, 0, 1601.5), (2, 2048, 128, 30.0)])
        );
        assert_eq!(sparse.topk_last, grid(&[(1, 16384, 0, 800.0), (1, 2048, 0, 190.0)]));
        assert_eq!(sparse.topk_flat, grid(&[(1, 2048, 0, 100.0)]));
        assert!(sparse.dsa_attn.is_empty());
    }

    /// Generation parity: exact / interior seq / interior batch / seq
    /// util-hold against Python
    /// `PerfDatabase.query_generation_dsa_module(..., DatabaseMode.SILICON)`.
    // NOTE(shared-layer merge): oracle generated pre-shared-layer; regenerate if this fails.
    #[test]
    fn dsa_generation_matches_python_v2_engine() {
        let table = DsaTable::new(b200_vllm_data_root());
        let spec = b200_sxm_spec();
        let q = |b: u32, s: u32| {
            table
                .query_generation(
                    &spec,
                    b,
                    s,
                    128,
                    KvCacheQuantMode::Bfloat16,
                    FmhaQuantMode::Bfloat16,
                    GemmQuantMode::Bfloat16,
                    "DeepseekV32ForCausalLM",
                    "trtllm",
                    false,
                )
                .unwrap()
        };
        // exact hit
        approx_rel(q(16, 4097), 0.2698);
        // interior seq blend (2049 < 3000 < 4097)
        approx_rel(q(16, 3000), 0.261390380859375);
        // interior batch blend (16 < 24 < 32)
        approx_rel(q(24, 4097), 0.27545);
        // seq util-hold beyond the frontier (validates the decode SOL)
        approx_rel(q(16, 300000), 0.5461828075504237);
    }

    /// Write one synthetic DSA module parquet with the collector's column set
    /// (`op_name, kernel_source, architecture, mla_dtype, kv_cache_dtype,
    /// gemm_type, num_heads, batch_size, isl, step, latency`). Row tuple:
    /// `(op_name, kernel_source, latency)`; the shape coordinate is fixed at
    /// (DeepseekV32ForCausalLM, bf16/bf16/bf16, n=128, b=1, isl=1024, step=0).
    fn write_dsa_module_parquet(path: &Path, rows: &[(&str, &str, f64)]) {
        let rows_kv: Vec<(&str, &str, &str, i64, f64)> =
            rows.iter().map(|r| (r.0, r.1, "bfloat16", 1024, r.2)).collect();
        write_dsa_module_parquet_rows(path, &rows_kv)
    }

    /// Row tuple: `(op_name, kernel_source, kv_cache_dtype, isl, latency)`.
    fn write_dsa_module_parquet_rows(path: &Path, rows: &[(&str, &str, &str, i64, f64)]) {
        use parquet::data_type::{ByteArray, ByteArrayType, DoubleType, Int64Type};
        use parquet::file::properties::WriterProperties;
        use parquet::file::writer::SerializedFileWriter;
        use parquet::schema::parser::parse_message_type;

        let schema = "message schema {
            REQUIRED BINARY op_name (UTF8);
            REQUIRED BINARY kernel_source (UTF8);
            REQUIRED BINARY architecture (UTF8);
            REQUIRED BINARY mla_dtype (UTF8);
            REQUIRED BINARY kv_cache_dtype (UTF8);
            REQUIRED BINARY gemm_type (UTF8);
            REQUIRED INT64 num_heads;
            REQUIRED INT64 batch_size;
            REQUIRED INT64 isl;
            REQUIRED INT64 step;
            REQUIRED DOUBLE latency;
        }";
        let schema = Arc::new(parse_message_type(schema).expect("schema must parse"));
        let file = std::fs::File::create(path).expect("create parquet");
        let mut writer =
            SerializedFileWriter::new(file, schema, Arc::new(WriterProperties::builder().build()))
                .expect("writer");
        let mut rg = writer.next_row_group().expect("row group");
        let str_cols: [Vec<ByteArray>; 6] = [
            rows.iter().map(|r| ByteArray::from(r.0)).collect(),
            rows.iter().map(|r| ByteArray::from(r.1)).collect(),
            rows.iter().map(|_| ByteArray::from("DeepseekV32ForCausalLM")).collect(),
            rows.iter().map(|_| ByteArray::from("bfloat16")).collect(),
            rows.iter().map(|r| ByteArray::from(r.2)).collect(),
            rows.iter().map(|_| ByteArray::from("bfloat16")).collect(),
        ];
        for values in &str_cols {
            let mut col = rg.next_column().expect("next col").expect("str col");
            col.typed::<ByteArrayType>().write_batch(values, None, None).expect("write str");
            col.close().expect("close col");
        }
        let int_cols: [Vec<i64>; 4] = [
            rows.iter().map(|_| 128).collect(), // num_heads
            rows.iter().map(|_| 1).collect(),   // batch_size
            rows.iter().map(|r| r.3).collect(), // isl
            rows.iter().map(|_| 0).collect(),   // step
        ];
        for values in &int_cols {
            let mut col = rg.next_column().expect("next col").expect("int col");
            col.typed::<Int64Type>().write_batch(values, None, None).expect("write ints");
            col.close().expect("close col");
        }
        let latencies: Vec<f64> = rows.iter().map(|r| r.4).collect();
        let mut col = rg.next_column().expect("next col").expect("latency col");
        col.typed::<DoubleType>().write_batch(&latencies, None, None).expect("write latency");
        col.close().expect("close col");
        rg.close().expect("close row group");
        writer.close().expect("close writer");
    }

    /// Item 2: one file carrying full + GLM-5.2 skip-indexer rows AND two
    /// kernel_source backends at the SAME coordinate must NOT blend. Python
    /// (`load_context_dsa_module_data`) splits by op_name and keys by the
    /// derived dsa_backend; its query picks the requested backend slice with
    /// the `_select_dsa_backend` fallback chain. Oracle generated with:
    ///
    /// ```text
    /// PYTHONPATH=src python3 -c "
    /// import pandas as pd, tempfile, os
    /// from aiconfigurator.sdk.operations.dsa import load_context_dsa_module_data, _select_dsa_backend
    /// from aiconfigurator.sdk import common
    /// tmp = tempfile.mkdtemp(); f = os.path.join(tmp, 'dsa_context_module_perf.parquet')
    /// base = dict(architecture='DeepseekV32ForCausalLM', mla_dtype='bfloat16',
    ///             kv_cache_dtype='bfloat16', gemm_type='bfloat16',
    ///             num_heads=128, batch_size=1, isl=1024, step=0)
    /// pd.DataFrame([
    ///  dict(op_name='dsa_context_module', kernel_source='trtllm_gen', latency=1.0, **base),
    ///  dict(op_name='dsa_context_module_skip_indexer', kernel_source='trtllm_gen', latency=9.0, **base),
    ///  dict(op_name='dsa_context_module', kernel_source='default', latency=5.0, **base),
    /// ]).to_parquet(f)
    /// full = load_context_dsa_module_data(f, op_kind='full')
    /// node = full[common.FMHAQuantMode.bfloat16][common.KVCacheQuantMode.bfloat16][common.GEMMQuantMode.bfloat16]['DeepseekV32ForCausalLM']
    /// print(_select_dsa_backend(node, 'trtllm')[128][0][1024][1])       # {'latency': 5.0, ...} (bf16 rows back both buckets)
    /// print(_select_dsa_backend(node, 'flashmla_kv')[128][0][1024][1])  # {'latency': 5.0, ...}"
    /// ```
    ///
    /// The old loader (no op_name filter, no backend keying) collapsed all
    /// three rows into one cell with last-row-wins and answered 5.0 for both
    /// backends (and would answer 9.0 if the skip row came last).
    #[test]
    fn dsa_full_and_skip_indexer_rows_do_not_blend_and_backend_slices_split() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        write_dsa_module_parquet(
            &tmp.path().join("dsa_context_module_perf.parquet"),
            &[
                ("dsa_context_module", "trtllm_gen", 1.0),
                ("dsa_context_module_skip_indexer", "trtllm_gen", 9.0),
                ("dsa_context_module", "default", 5.0),
            ],
        );
        let table = DsaTable::new(tmp.path().to_path_buf());
        let spec = b200_sxm_spec();
        let q = |dsa_backend: &str| {
            table
                .query_context(
                    &spec,
                    1,
                    1024,
                    128,
                    KvCacheQuantMode::Bfloat16,
                    FmhaQuantMode::Bfloat16,
                    GemmQuantMode::Bfloat16,
                    "DeepseekV32ForCausalLM",
                    0,
                    INDEX_TOPK,
                    dsa_backend,
                    false,
                )
                .expect("query must succeed")
        };
        // BF16-KV rows back BOTH buckets (mirrors Python
        // `_dsa_kernel_source_buckets`): both full rows land in each bucket
        // at the same coordinate, so last-row-wins answers 5.0 everywhere.
        // The skip row (9.0) still never blends in.
        assert_eq!(q("trtllm"), 5.0);
        assert_eq!(q("flashmla_kv"), 5.0);
    }

    /// FP8-KV rows bucket by executed-kernel name (the serving FP8-KV
    /// sub-backend selector): indexer_trtllm -> trtllm only,
    /// indexer_flashmla_sparse -> flashmla_kv only, dense ragged prefill
    /// (shape-dispatched) -> both.
    #[test]
    fn dsa_fp8_rows_bucket_by_executed_kernel_name() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        write_dsa_module_parquet_rows(
            &tmp.path().join("dsa_context_module_perf.parquet"),
            &[
                ("dsa_context_module", "sglang_dsa_indexer_trtllm", "fp8", 4096, 1.0),
                ("dsa_context_module", "sglang_dsa_indexer_flashmla_sparse", "fp8", 4096, 5.0),
                ("dsa_context_module", "sglang_dsa_dense_mha_trtllm_ragged", "fp8", 1024, 7.0),
            ],
        );
        let table = DsaTable::new(tmp.path().to_path_buf());
        let spec = b200_sxm_spec();
        let q = |dsa_backend: &str, isl: u32| {
            table
                .query_context(
                    &spec,
                    1,
                    isl,
                    128,
                    KvCacheQuantMode::Fp8,
                    FmhaQuantMode::Bfloat16,
                    GemmQuantMode::Bfloat16,
                    "DeepseekV32ForCausalLM",
                    0,
                    INDEX_TOPK,
                    dsa_backend,
                    false,
                )
                .expect("query must succeed")
        };
        assert_eq!(q("trtllm", 4096), 1.0);
        assert_eq!(q("flashmla_kv", 4096), 5.0);
        // Dense prefill rows back both buckets.
        assert_eq!(q("trtllm", 1024), 7.0);
        assert_eq!(q("flashmla_kv", 1024), 7.0);
    }

    /// Write one synthetic DSA GENERATION parquet with the collector's
    /// column set. Row tuple: `(isl, step, latency)`; the shape coordinate
    /// is fixed at (DeepseekV32ForCausalLM, bf16 mla/kv/gemm, n=128, b=1,
    /// op_name="dsa_generation_module", kernel_source="default").
    fn write_dsa_generation_parquet(path: &Path, rows: &[(i64, i64, f64)]) {
        use parquet::data_type::{ByteArray, ByteArrayType, DoubleType, Int64Type};
        use parquet::file::properties::WriterProperties;
        use parquet::file::writer::SerializedFileWriter;
        use parquet::schema::parser::parse_message_type;

        let schema = "message schema {
            REQUIRED BINARY op_name (UTF8);
            REQUIRED BINARY kernel_source (UTF8);
            REQUIRED BINARY architecture (UTF8);
            REQUIRED BINARY mla_dtype (UTF8);
            REQUIRED BINARY kv_cache_dtype (UTF8);
            REQUIRED BINARY gemm_type (UTF8);
            REQUIRED INT64 num_heads;
            REQUIRED INT64 batch_size;
            REQUIRED INT64 isl;
            REQUIRED INT64 step;
            REQUIRED DOUBLE latency;
        }";
        let schema = Arc::new(parse_message_type(schema).expect("schema must parse"));
        let file = std::fs::File::create(path).expect("create parquet");
        let mut writer =
            SerializedFileWriter::new(file, schema, Arc::new(WriterProperties::builder().build()))
                .expect("writer");
        let mut rg = writer.next_row_group().expect("row group");
        let str_cols: [Vec<ByteArray>; 6] = [
            rows.iter().map(|_| ByteArray::from("dsa_generation_module")).collect(),
            rows.iter().map(|_| ByteArray::from("default")).collect(),
            rows.iter().map(|_| ByteArray::from("DeepseekV32ForCausalLM")).collect(),
            rows.iter().map(|_| ByteArray::from("bfloat16")).collect(),
            rows.iter().map(|_| ByteArray::from("bfloat16")).collect(),
            rows.iter().map(|_| ByteArray::from("bfloat16")).collect(),
        ];
        for values in &str_cols {
            let mut col = rg.next_column().expect("next col").expect("str col");
            col.typed::<ByteArrayType>().write_batch(values, None, None).expect("write str");
            col.close().expect("close col");
        }
        let int_cols: [Vec<i64>; 4] = [
            rows.iter().map(|_| 128).collect(), // num_heads
            rows.iter().map(|_| 1).collect(),   // batch_size
            rows.iter().map(|r| r.0).collect(), // isl
            rows.iter().map(|r| r.1).collect(), // step
        ];
        for values in &int_cols {
            let mut col = rg.next_column().expect("next col").expect("int col");
            col.typed::<Int64Type>().write_batch(values, None, None).expect("write ints");
            col.close().expect("close col");
        }
        let latencies: Vec<f64> = rows.iter().map(|r| r.2).collect();
        let mut col = rg.next_column().expect("next col").expect("latency col");
        col.typed::<DoubleType>().write_batch(&latencies, None, None).expect("write latency");
        col.close().expect("close col");
        rg.close().expect("close row group");
        writer.close().expect("close writer");
    }

    /// Issue #1333 item 4.7-3: two generation rows whose different
    /// (isl, step) decompositions collapse to the SAME s_total = isl + step
    /// must resolve to Python's winner — the LAST row in FILE order (plain
    /// per-source dict overwrite in `load_generation_dsa_module_data`) — not
    /// the largest-step row. The old Rust collapse iterated the
    /// `[step][isl]` BTreeMaps ascending inside `build_generation_nodes`,
    /// so the step=20 row always wrote last and won with 111.0 regardless
    /// of file order. Oracle:
    ///
    /// ```text
    /// PYTHONPATH=src python3 -c "
    /// import pandas as pd, tempfile, os
    /// from aiconfigurator.sdk.operations.dsa import load_generation_dsa_module_data
    /// from aiconfigurator.sdk import common
    /// tmp = tempfile.mkdtemp(); f = os.path.join(tmp, 'dsa_generation_module_perf.parquet')
    /// base = dict(architecture='DeepseekV32ForCausalLM', mla_dtype='bfloat16',
    ///             kv_cache_dtype='bfloat16', gemm_type='bfloat16', num_heads=128,
    ///             batch_size=1, op_name='dsa_generation_module', kernel_source='default')
    /// pd.DataFrame([
    ///  dict(isl=80, step=20, latency=111.0, **base),
    ///  dict(isl=90, step=10, latency=222.0, **base),
    /// ]).to_parquet(f)
    /// d = load_generation_dsa_module_data(f)
    /// print(d[common.KVCacheQuantMode.bfloat16][common.GEMMQuantMode.bfloat16]
    ///        ['DeepseekV32ForCausalLM']['flashmla_kv'][128][1][100])
    /// # -> {'latency': 222.0, 'power': 0.0, 'energy': 0.0}"
    /// ```
    #[test]
    fn dsa_generation_same_seq_ties_resolve_last_file_row() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        write_dsa_generation_parquet(
            &tmp.path().join("dsa_generation_module_perf.parquet"),
            &[(80, 20, 111.0), (90, 10, 222.0)],
        );
        let table = DsaTable::new(tmp.path().to_path_buf());
        let spec = b200_sxm_spec();
        let got = table
            .query_generation(
                &spec,
                1,   // b
                100, // seq = isl + step
                128, // num_heads
                KvCacheQuantMode::Bfloat16,
                FmhaQuantMode::Bfloat16,
                GemmQuantMode::Bfloat16,
                "DeepseekV32ForCausalLM",
                "flashmla_kv",
                false,
            )
            .expect("query must succeed");
        assert_eq!(got, 222.0);
    }

    /// Item 2 (fallback): a single-backend file must resolve for ANY
    /// requested backend via Python's `_select_dsa_backend` chain
    /// (requested -> flashmla_kv -> trtllm -> first).
    #[test]
    fn dsa_backend_fallback_resolves_single_backend_files() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        write_dsa_module_parquet(
            &tmp.path().join("dsa_context_module_perf.parquet"),
            &[("dsa_context_module", "trtllm_gen", 3.5)],
        );
        let table = DsaTable::new(tmp.path().to_path_buf());
        let spec = b200_sxm_spec();
        let got = table
            .query_context(
                &spec,
                1,
                1024,
                128,
                KvCacheQuantMode::Bfloat16,
                FmhaQuantMode::Bfloat16,
                GemmQuantMode::Bfloat16,
                "DeepseekV32ForCausalLM",
                0,
                INDEX_TOPK,
                "flashmla_kv", // absent: falls back to the trtllm slice
                false,
            )
            .expect("query must succeed");
        assert_eq!(got, 3.5);
    }
}
