// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! TRT-LLM all-to-all perf table for distributed MoE dispatch.
//!
//! One parquet table, `trtllm_alltoall_perf.parquet` (a subset of the MoE
//! perf columns), keyed by `[kernel_source][op_name][quant][num_nodes]
//! [hidden_size][topk][num_experts][moe_ep_size]` -> `{num_tokens ->
//! latency}`.
//!
//! The loader is lazy. Token curves resolve on the shared perf_interp v2
//! engine (Grid, RAW lerp in range; beyond the collected range the boundary
//! util is held with `k_tail=1` and a LINEAR num_tokens proxy SOL carries the
//! growth). Python's SOL (`_query_alltoall_table.get_sol`) is
//! `const * num_tokens` per slice, so the linear proxy is ratio-equivalent.
//!
//! The sibling wideEP MoE-compute and SGLang DeepEP dispatch tables that used
//! to live here retired with AIC-1601; large-EP is served by
//! `perf_database::moe_ep`.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::OnceLock;

use crate::common::enums::MoeQuantMode;
use crate::common::error::AicError;
use crate::common::system_spec::SystemSpec;
use crate::config::{PerfDbSources, PerfSource};
use super::{kernel_source_ok, resolve_op_sources};
use super::moe::query_token_curve;
use crate::perf_database::parquet_loader::PerfReader;

pub struct TrtllmAlltoallTable {
    data_root: PathBuf,
    /// Ordered, priority-sorted sources for `trtllm_alltoall_perf.parquet`
    /// (shared-layer aware; see [`PerfSource`]). Single-primary, no-filter by
    /// default (`TrtllmAlltoallTable::new`).
    alltoall_sources: Vec<PerfSource>,
    trtllm_alltoall: OnceLock<Result<AlltoallGrids, AicError>>,
}

/// TRT-LLM alltoall grids. Keying mirrors Python `load_trtllm_alltoall_data`
/// exactly: `[kernel_source][op_name][quant][num_nodes][hidden_size][topk]
/// [num_experts][moe_ep_size][num_tokens]`. Note the table has NO
/// `distribution` axis (the parquet column is ignored, as in Python) and NO
/// `inter_size`/`moe_tp_size`.
struct AlltoallGrids {
    by_keys: BTreeMap<AlltoallKey, BTreeMap<u32, f64>>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct AlltoallKey {
    kernel_source: String,
    op_name: String,
    quant: String,
    num_nodes: u32,
    hidden_size: u32,
    topk: u32,
    num_experts: u32,
    moe_ep_size: u32,
}

impl TrtllmAlltoallTable {
    /// Construct an empty table for the given data directory. No I/O. The
    /// perf file is sourced solely from `data_root/<basename>` with no
    /// `kernel_source` filter (pre-shared-layer behaviour).
    pub fn new(data_root: PathBuf) -> Self {
        Self::with_sources(data_root, &PerfDbSources::default())
    }

    /// Construct with shared-layer (sibling/cross-version) sources resolved from
    /// `perf_db_sources` (Python-supplied). The perf file falls back to its
    /// primary `data_root/<basename>` when absent from the map. No I/O.
    pub fn with_sources(data_root: PathBuf, perf_db_sources: &PerfDbSources) -> Self {
        let alltoall_sources =
            resolve_op_sources(perf_db_sources, "trtllm_alltoall_perf.parquet", &data_root);
        Self { data_root, alltoall_sources, trtllm_alltoall: OnceLock::new() }
    }

    /// TRT-LLM alltoall latency for one phase op. Mirrors Python
    /// `TrtLLMWideEPMoEDispatch._query_alltoall_table` (SILICON path):
    ///
    /// 1. `node_num` defaults to `1 if ep < 4 else ep // 4` (the Python
    ///    default when the caller doesn't pass one — no Rust caller does);
    /// 2. `op_name` must be one of the four collected phases;
    /// 3. the kernel is auto-selected from the system architecture + MoE
    ///    backend ([`select_alltoall_kernel`]); `NotEnabled` short-circuits
    ///    to 0.0 (a dense-fallback config has no alltoall cost);
    /// 4. `fp8_block` reuses the `fp8` tables (behavioral mode,
    ///    `_normalize_quant_mode_for_table`);
    /// 5. the 1-D token curve resolves RAW-lerp in range; beyond it the
    ///    boundary util holds on the linear token proxy, ratio-identical to
    ///    Python's per-slice alltoall SOL (`const * num_tokens`).
    #[allow(clippy::too_many_arguments)]
    pub fn query_trtllm_alltoall(
        &self,
        spec: &SystemSpec,
        op_name: &str,
        num_tokens: u32,
        hidden_size: u32,
        topk: u32,
        num_experts: u32,
        moe_ep_size: u32,
        quant: MoeQuantMode,
        moe_backend: Option<&str>,
    ) -> Result<f64, AicError> {
        const VALID_OP_NAMES: [&str; 4] = [
            "alltoall_prepare",
            "alltoall_dispatch",
            "alltoall_combine",
            "alltoall_combine_low_precision",
        ];
        if !VALID_OP_NAMES.contains(&op_name) {
            return Err(AicError::PerfDatabase(format!(
                "Invalid op_name '{op_name}'. Must be one of {VALID_OP_NAMES:?}"
            )));
        }
        let kernel_source = select_alltoall_kernel(spec, moe_ep_size, topk, moe_backend);
        if kernel_source == "NotEnabled" {
            return Ok(0.0);
        }
        let node_num = if moe_ep_size < 4 { 1 } else { moe_ep_size / 4 };
        // fp8_block reuses the fp8 alltoall tables (Python
        // `_normalize_quant_mode_for_table`); the table key is the only
        // consumer — the (linear-proxy) SOL is quant-independent.
        let table_quant = if quant == MoeQuantMode::Fp8Block {
            MoeQuantMode::Fp8
        } else {
            quant
        };
        let grids = self.load_trtllm_alltoall()?;
        let key = AlltoallKey {
            kernel_source: kernel_source.to_string(),
            op_name: op_name.to_string(),
            quant: table_quant.name().to_string(),
            num_nodes: node_num,
            hidden_size,
            topk,
            num_experts,
            moe_ep_size,
        };
        let by_tokens = grids.by_keys.get(&key).ok_or_else(|| {
            AicError::PerfDatabase(format!(
                "trtllm alltoall data missing for {key:?} at {}",
                self.data_root.display()
            ))
        })?;
        query_token_curve(by_tokens, num_tokens as f64, &|t| t)
    }

    /// Collected `(num_tokens, latency)` points of one resolved alltoall
    /// slice — the operator-layer util-calibration input (Python
    /// `_query_alltoall_table.get_empirical` builds its depth-1 grid over the
    /// same slice the silicon lookup resolves). The caller has already done
    /// kernel selection / quant normalization / the node_num default; a
    /// missing key or empty curve is a typed miss.
    #[allow(clippy::too_many_arguments)]
    pub fn alltoall_slice_points(
        &self,
        kernel_source: &str,
        op_name: &str,
        table_quant: MoeQuantMode,
        node_num: u32,
        hidden_size: u32,
        topk: u32,
        num_experts: u32,
        moe_ep_size: u32,
    ) -> Result<Vec<(u32, f64)>, AicError> {
        let grids = self.load_trtllm_alltoall()?;
        let key = AlltoallKey {
            kernel_source: kernel_source.to_string(),
            op_name: op_name.to_string(),
            quant: table_quant.name().to_string(),
            num_nodes: node_num,
            hidden_size,
            topk,
            num_experts,
            moe_ep_size,
        };
        let by_tokens = grids.by_keys.get(&key).ok_or_else(|| {
            AicError::PerfDatabase(format!(
                "trtllm alltoall data missing for {key:?} at {}",
                self.data_root.display()
            ))
        })?;
        if by_tokens.is_empty() {
            return Err(AicError::PerfDatabase(format!(
                "trtllm alltoall data empty for {key:?} at {}",
                self.data_root.display()
            )));
        }
        Ok(by_tokens.iter().map(|(&t, &lat)| (t, lat)).collect())
    }

    fn load_trtllm_alltoall(&self) -> Result<&AlltoallGrids, AicError> {
        let cell = self
            .trtllm_alltoall
            .get_or_init(|| load_alltoall_parquet(&self.alltoall_sources));
        cell.as_ref().map_err(clone_err)
    }
}

/// Auto-select the TRT-LLM All2All kernel. Verbatim port of Python
/// `TrtLLMWideEPMoEDispatch._select_alltoall_kernel` (operations/moe.py),
/// aligned with TensorRT-LLM's per-backend `select_alltoall_method_type`:
///
/// - `DEEPGEMM` / `CUTE_DSL` MoE backends never use alltoall;
/// - WideEP: MNNVL (SM >= 100) -> `NVLinkTwoSided`; else DeepEP when feasible
///   (`ep > 1 && topk <= 8`): inter-node (`ep > num_gpus_per_node`) ->
///   `DeepEP`, intra-node -> `DeepEPLowLatency`; else `NotEnabled`;
/// - non-WideEP (Cutlass/TRTLLM): MNNVL -> `NVLinkOneSided`, else `NotEnabled`.
///
/// Python additionally warns when the preferred kernel is absent from the
/// loaded table but still returns it (the downstream slice miss surfaces the
/// error) — behavior-identical, so the warning is not replicated.
pub(crate) fn select_alltoall_kernel(
    spec: &SystemSpec,
    moe_ep_size: u32,
    topk: u32,
    moe_backend: Option<&str>,
) -> &'static str {
    if let Some(backend) = moe_backend {
        let upper = backend.to_uppercase();
        if upper == "DEEPGEMM" || upper == "CUTE_DSL" {
            return "NotEnabled";
        }
    }
    let supports_mnnvl = spec.gpu.sm_version.unwrap_or(0) >= 100;
    let is_wideep = moe_backend.map(|b| b.to_uppercase() == "WIDEEP").unwrap_or(false);
    if is_wideep {
        if supports_mnnvl {
            return "NVLinkTwoSided";
        }
        let deepep_feasible = moe_ep_size > 1 && topk <= 8;
        let is_inter_node = moe_ep_size > spec.node.num_gpus_per_node;
        if deepep_feasible && is_inter_node {
            "DeepEP"
        } else if deepep_feasible {
            "DeepEPLowLatency"
        } else {
            "NotEnabled"
        }
    } else if supports_mnnvl {
        "NVLinkOneSided"
    } else {
        "NotEnabled"
    }
}

/// Load the TRT-LLM alltoall table from an ordered source list. Mirrors
/// Python `load_trtllm_alltoall_data` exactly:
///
/// - key `[kernel_source][op_name][quant][num_nodes][hidden][topk]
///   [num_experts][moe_ep_size]` -> `{num_tokens -> latency}`;
/// - `kernel_source` defaults to `"NVLinkTwoSided"` when the column is
///   absent; `num_nodes` defaults to `max(1, moe_ep_size // 4)` (GB200 NVL4);
/// - the `distribution` column is IGNORED (Python never reads it);
/// - duplicates resolve FIRST-wins (Python `load_trtllm_alltoall_data`
///   guards with the standard skip-on-key-conflict idiom since #1423 —
///   shared-layer contract, design §6.1).
fn load_alltoall_parquet(sources: &[PerfSource]) -> Result<AlltoallGrids, AicError> {
    let mut by_keys: BTreeMap<AlltoallKey, BTreeMap<u32, f64>> = BTreeMap::new();
    let mut any_source = false;
    for source in sources {
        let path = source.path();
        if !path.exists() {
            continue;
        }
        any_source = true;
        let reader = PerfReader::open(path)?;
        let op_name_col = reader.col("op_name")?;
        let moe_dtype_col = reader.col("moe_dtype")?;
        let num_tokens_col = reader.col("num_tokens")?;
        let hidden_size_col = reader.col("hidden_size")?;
        let topk_col = reader.col("topk")?;
        let num_experts_col = reader.col("num_experts")?;
        let moe_ep_size_col = reader.col("moe_ep_size")?;
        let latency_col = reader.col("latency")?;
        let ks_col = reader.col_optional("kernel_source");
        let num_nodes_col = reader.col_optional("num_nodes");
        for row in reader.rows()? {
            let row = row?;
            if !kernel_source_ok(source.kernel_sources(), ks_col, &row)? {
                continue;
            }
            let moe_ep_size = row.u32(moe_ep_size_col)?;
            let key = AlltoallKey {
                kernel_source: row
                    .str_optional(ks_col)?
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| "NVLinkTwoSided".to_string()),
                op_name: row.str_owned(op_name_col)?,
                quant: row.str_owned(moe_dtype_col)?,
                num_nodes: row
                    .u32_optional(num_nodes_col)?
                    .unwrap_or_else(|| (moe_ep_size / 4).max(1)),
                hidden_size: row.u32(hidden_size_col)?,
                topk: row.u32(topk_col)?,
                num_experts: row.u32(num_experts_col)?,
                moe_ep_size,
            };
            by_keys
                .entry(key)
                .or_default()
                .entry(row.u32(num_tokens_col)?)
                .or_insert(row.f64(latency_col)?);
        }
    }
    if !any_source || by_keys.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "no TRT-LLM alltoall rows loaded from {} source(s) (first: {})",
            sources.len(),
            sources.first().map(|s| s.path().display().to_string()).unwrap_or_default()
        )));
    }
    Ok(AlltoallGrids { by_keys })
}

fn clone_err(err: &AicError) -> AicError {
    AicError::PerfDatabase(err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gb200_spec() -> SystemSpec {
        let yaml = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../src/aiconfigurator_core/systems/gb200.yaml");
        SystemSpec::load(&yaml).expect("gb200.yaml must parse")
    }

    fn gb200_trtllm_table() -> TrtllmAlltoallTable {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../src/aiconfigurator_core/systems/data/gb200/trtllm/1.3.0rc10");
        TrtllmAlltoallTable::new(root)
    }

    /// Kernel auto-selection mirrors Python `_select_alltoall_kernel`:
    /// gb200 (sm 100): non-WideEP -> NVLinkOneSided, WideEP -> NVLinkTwoSided,
    /// DEEPGEMM/CUTE_DSL -> NotEnabled (query returns 0.0).
    #[test]
    fn alltoall_kernel_selection_matches_python() {
        let spec = gb200_spec();
        assert_eq!(select_alltoall_kernel(&spec, 4, 8, None), "NVLinkOneSided");
        assert_eq!(select_alltoall_kernel(&spec, 4, 8, Some("WIDEEP")), "NVLinkTwoSided");
        assert_eq!(select_alltoall_kernel(&spec, 4, 8, Some("DeepGemm")), "NotEnabled");
        assert_eq!(select_alltoall_kernel(&spec, 4, 8, Some("cute_dsl")), "NotEnabled");
        let table = gb200_trtllm_table();
        let zero = table
            .query_trtllm_alltoall(&spec, "alltoall_dispatch", 1, 7168, 8, 256, 4, MoeQuantMode::Fp8, Some("DEEPGEMM"))
            .expect("NotEnabled short-circuits");
        assert_eq!(zero, 0.0);
    }

    /// Exact-hit anchors from gb200/trtllm/1.3.0rc10/trtllm_alltoall_perf.parquet.
    /// The pre-fix loader collapsed kernel_source/op_name/num_nodes (1,556 of
    /// 2,096 rows collided) and the query keyed distribution="uniform" (data is
    /// "balanced") — these anchors fail on both bugs.
    #[test]
    fn alltoall_exact_hits_match_parquet_rows() {
        let spec = gb200_spec();
        let table = gb200_trtllm_table();
        // WideEP -> NVLinkTwoSided; fp8 dispatch row (ep=4 -> node_num=1).
        let dispatch = table
            .query_trtllm_alltoall(&spec, "alltoall_dispatch", 1, 7168, 8, 256, 4, MoeQuantMode::Fp8, Some("WIDEEP"))
            .expect("dispatch row");
        assert!((dispatch - 0.011_372_800_171_375_274).abs() < 1e-12, "got {dispatch}");
        // Same slice, combine phase: distinct value proves op_name keys the table.
        let combine = table
            .query_trtllm_alltoall(&spec, "alltoall_combine", 1, 7168, 8, 256, 4, MoeQuantMode::Fp8, Some("WIDEEP"))
            .expect("combine row");
        assert!((combine - 0.012_921_600_043_773_651).abs() < 1e-12, "got {combine}");
        // fp8_block reuses the fp8 tables (Python `_normalize_quant_mode_for_table`).
        let block = table
            .query_trtllm_alltoall(&spec, "alltoall_dispatch", 1, 7168, 8, 256, 4, MoeQuantMode::Fp8Block, Some("WIDEEP"))
            .expect("fp8_block reroutes to fp8");
        assert_eq!(block, dispatch);
        // Non-WideEP -> NVLinkOneSided (nvfp4-only slice, ep=2 -> node_num=1).
        let one_sided = table
            .query_trtllm_alltoall(&spec, "alltoall_dispatch", 1, 7168, 8, 256, 2, MoeQuantMode::Nvfp4, None)
            .expect("one-sided row");
        assert!((one_sided - 0.012_895_999_848_842_621).abs() < 1e-12, "got {one_sided}");
    }

    #[test]
    fn alltoall_loader_smoke() {
        // No TRT-LLM alltoall data exists on vLLM b200 (TRT-LLM territory).
        // Loader must surface a clean typed error, not a panic.
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../src/aiconfigurator_core/systems/data/b200_sxm/vllm/0.19.0");
        let spec = gb200_spec();
        let table = TrtllmAlltoallTable::new(root);
        let err = table
            .query_trtllm_alltoall(
                &spec,
                "alltoall_dispatch",
                64,
                7168,
                8,
                256,
                4,
                MoeQuantMode::Fp8,
                Some("WIDEEP"),
            )
            .unwrap_err();
        match err {
            AicError::Io { .. } | AicError::PerfDatabase(_) => {}
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
