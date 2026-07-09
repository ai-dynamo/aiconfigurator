// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DSA (DeepSeek-V3.2 Dynamic Sparse Attention) module perf tables.
//!
//! Two parquet files: `dsa_context_module_perf.parquet` and
//! `dsa_generation_module_perf.parquet`. Both share columns: model,
//! architecture, mla_dtype, kv_cache_dtype, gemm_type, num_heads,
//! batch_size, isl, tp_size, step, latency.
//!
//! Data is nested by (architecture, mla_dtype, kv_cache_dtype, gemm_type)
//! → num_heads → step → isl → batch_size → latency. The `step` axis is the
//! "prefix value" (past-KV length).
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
use std::sync::OnceLock;

use crate::common::enums::{FmhaQuantMode, GemmQuantMode, KvCacheQuantMode};
use crate::common::error::AicError;
use crate::common::system_spec::SystemSpec;
use super::perf_interp::{self, Node, OpInterpConfig};
use crate::perf_database::parquet_loader::PerfReader;

pub struct DsaTable {
    data_root: PathBuf,
    context: OnceLock<Result<DsaGrids, AicError>>,
    generation: OnceLock<Result<DsaGrids, AicError>>,
    /// Engine-ready per-`DsaKey` context tables with the raw shape
    /// `[num_heads][step][isl][batch]`, built once from the loaded grids.
    context_nodes: OnceLock<Result<NodeCache, AicError>>,
    /// Engine-ready per-`DsaKey` generation tables with the Python v2 axis
    /// order `[num_heads][batch][seq = isl + step]`, built once at load.
    generation_nodes: OnceLock<Result<NodeCache, AicError>>,
}

struct NodeCache {
    by_keys: BTreeMap<DsaKey, Node>,
}

/// (arch, fmha, kv, gemm) → num_heads → step → isl → batch → latency.
pub struct DsaGrids {
    pub by_keys: BTreeMap<DsaKey, BTreeMap<u32, BTreeMap<u32, BTreeMap<u32, BTreeMap<u32, f64>>>>>,
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
struct DsaDims {
    hidden_size: i64,
    q_lora_rank: i64,
    kv_lora_rank: i64,
    qk_nope_head_dim: i64,
    qk_rope_head_dim: i64,
    v_head_dim: i64,
    index_topk: i64,
    index_head_dim: i64,
    index_n_heads: i64,
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

fn dsa_dims(architecture: &str) -> &'static DsaDims {
    match architecture {
        "GlmMoeDsaForCausalLM" => &GLM_MOE_DSA_DIMS,
        _ => &DSV32_DIMS,
    }
}

impl DsaTable {
    pub fn new(data_root: PathBuf) -> Self {
        Self {
            data_root,
            context: OnceLock::new(),
            generation: OnceLock::new(),
            context_nodes: OnceLock::new(),
            generation_nodes: OnceLock::new(),
        }
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
    ) -> Result<f64, AicError> {
        let nodes = self.load_context_nodes()?;
        let key = DsaKey {
            architecture: architecture.to_string(),
            fmha_quant: fmha_quant.name().to_string(),
            kv_quant: kv_quant.name().to_string(),
            gemm_quant: gemm_quant.name().to_string(),
        };
        let node = nodes.by_keys.get(&key).ok_or_else(|| {
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
    ) -> Result<f64, AicError> {
        let nodes = self.load_generation_nodes()?;
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
            .get_or_init(|| load_dsa_parquet(&self.data_root.join("dsa_context_module_perf.parquet")));
        cell.as_ref().map_err(clone_err)
    }

    fn load_generation(&self) -> Result<&DsaGrids, AicError> {
        let cell = self
            .generation
            .get_or_init(|| load_dsa_parquet(&self.data_root.join("dsa_generation_module_perf.parquet")));
        cell.as_ref().map_err(clone_err)
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
            Ok(build_generation_nodes(grids))
        });
        cell.as_ref().map_err(clone_err)
    }
}

/// Materialise the per-`DsaKey` engine table for `query_context` with the
/// raw nesting `[num_heads][step][isl][batch]` (the 4-axis grid Python v2
/// resolves on).
fn build_context_nodes(grids: &DsaGrids) -> NodeCache {
    let mut by_keys: BTreeMap<DsaKey, Node> = BTreeMap::new();
    for (key, by_heads) in &grids.by_keys {
        let node = by_keys.entry(key.clone()).or_insert_with(Node::branch);
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
    NodeCache { by_keys }
}

/// Materialise the per-`DsaKey` engine table for `query_generation` with the
/// Python v2 generation nesting `[num_heads][batch][seq = isl + step]`.
/// If multiple (step, isl) pairs map to the same seq the last write in
/// BTreeMap-sorted order wins (collected files carry no such collisions;
/// Python's per-row overwrite is file-order last-wins).
fn build_generation_nodes(grids: &DsaGrids) -> NodeCache {
    let mut by_keys: BTreeMap<DsaKey, Node> = BTreeMap::new();
    for (key, by_heads) in &grids.by_keys {
        let node = by_keys.entry(key.clone()).or_insert_with(Node::branch);
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
    // top-k window (regime split).
    let indexer_logits_ops = if full_s <= topk {
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
    let indexer_cache_bytes = if full_s <= topk {
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

fn load_dsa_parquet(path: &Path) -> Result<DsaGrids, AicError> {
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

    let mut by_keys: BTreeMap<DsaKey, BTreeMap<u32, BTreeMap<u32, BTreeMap<u32, BTreeMap<u32, f64>>>>> =
        BTreeMap::new();
    for row in reader.rows()? {
        let row = row?;
        let key = DsaKey {
            architecture: row.str_owned(arch_col)?,
            fmha_quant: row.str_owned(mla_dtype_col)?,
            kv_quant: row.str_owned(kv_cache_dtype_col)?,
            gemm_quant: row.str_owned(gemm_type_col)?,
        };
        // First-wins parity with Python `load_dsa_module_data`.
        by_keys
            .entry(key)
            .or_default()
            .entry(row.u32(num_heads_col)?)
            .or_default()
            .entry(row.u32(step_col)?)
            .or_default()
            .entry(row.u32(isl_col)?)
            .or_default()
            .entry(row.u32(batch_size_col)?)
            .or_insert(row.f64(latency_col)?);
    }
    if by_keys.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "no DSA module rows loaded from {}",
            path.display()
        )));
    }
    Ok(DsaGrids { by_keys })
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
            )
            .expect("DSA context query must succeed");
        assert!(
            (latency - 1.0972).abs() < 1e-6,
            "expected recorded latency, got {latency}"
        );
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

    /// Generation parity: exact / interior seq / interior batch / seq
    /// util-hold against Python
    /// `PerfDatabase.query_generation_dsa_module(..., DatabaseMode.SILICON)`.
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
}
