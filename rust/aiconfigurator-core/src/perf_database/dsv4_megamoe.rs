// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DeepSeek-V4 MegaMoE routed-module perf table.
//!
//! `dsv4_megamoe_module_perf.parquet` — the measured SGLang/DeepGEMM MegaMoE
//! routed path (prepared hidden states + top-k tensors -> SGLang pre-dispatch
//! -> `deep_gemm.fp8_fp4_mega_moe` -> routed output scaling). Gate/top-k and
//! shared experts are modeled outside this table.
//!
//! Port of Python `operations/dsv4.py::load_dsv4_megamoe_module_data` +
//! `DeepSeekV4MegaMoEModule._query_megamoe_table`.
//!
//! ## Loading (mirrors the Python loader exactly)
//!
//! - SINGLE primary file only: Python's `load_data` passes one unified path
//!   (`_read_filtered_rows(<str>)`), NOT the shared-layer source list — the
//!   loader even `raise`s `TypeError` on a list. No sibling/cross-version
//!   inheritance and no `kernel_source` row filter here.
//! - Row invariants (any violation fails the WHOLE load, mirroring the
//!   Python `ValueError`s): `used_cuda_graph` must be true,
//!   `includes_gate_topk` must be false, `includes_routed_scale` must be
//!   true; `phase` must be `context` | `generation`.
//! - Keying: `[phase][kernel_source][kernel_dtype][moe_dtype][pre_dispatch]`
//!   `[source_policy][distribution][topk][num_experts]`
//!   `[num_fused_shared_experts][hidden_size][inter_size][moe_tp_size]`
//!   `[moe_ep_size][num_tokens]`. Duplicate leaves are a load ERROR
//!   (Python `_put_nested`), not last-wins.
//! - `moe_dtype` must name a valid `MoEQuantMode` member (Python
//!   `common.MoEQuantMode[row["moe_dtype"]]` KeyErrors otherwise).
//!
//! ## Query (mirrors `_query_megamoe_table`)
//!
//! Strict measured-only table: exact key walk (typed miss on any absent
//! level), then a 1-D `num_tokens` Grid resolution with a LINEAR token proxy
//! SOL (`sol_fn = lambda t: float(t)` — routed-expert work scales ~linearly
//! with tokens at fixed topk/experts/hidden, and util-hold only needs the
//! SOL RATIO). The SILICON/HYBRID-only mode contract lives on the operator
//! (`operators/dsv4.rs::Dsv4MegaMoeOp`), matching Python's split. Latency
//! only — the Rust engine does not track energy.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::OnceLock;

use crate::common::enums::MoeQuantMode;
use crate::common::error::AicError;
use super::moe::query_token_curve;
use crate::perf_database::parquet_loader::PerfReader;

pub struct Dsv4MegaMoeTable {
    /// The single unified perf file (see the module note: the Python loader
    /// reads ONE primary path, never the shared-layer source list).
    primary_path: PathBuf,
    module: OnceLock<Result<Dsv4MegaMoeGrids, AicError>>,
}

struct Dsv4MegaMoeGrids {
    by_keys: BTreeMap<Dsv4MegaMoeKey, BTreeMap<u32, f64>>,
}

/// Full table key (every level of the Python nested dict except the trailing
/// `num_tokens` axis). `quant` holds the `MoeQuantMode` name — Python keys
/// the enum member, whose name is exactly this string.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Dsv4MegaMoeKey {
    phase: String,
    kernel_source: String,
    kernel_dtype: String,
    quant: String,
    pre_dispatch: String,
    source_policy: String,
    distribution: String,
    topk: u32,
    num_experts: u32,
    num_fused_shared_experts: u32,
    hidden_size: u32,
    inter_size: u32,
    moe_tp_size: u32,
    moe_ep_size: u32,
}

/// Python `common.MoEQuantMode[name]` (member-name lookup). The Rust serde
/// names are the same snake_case strings, so a serde round-trip is an exact
/// mirror of the Python KeyError contract.
fn moe_dtype_from_name(name: &str) -> Option<MoeQuantMode> {
    serde_json::from_value(serde_json::Value::String(name.to_string())).ok()
}

impl Dsv4MegaMoeTable {
    /// Construct for the given data directory. No I/O.
    pub fn new(data_root: PathBuf) -> Self {
        Self {
            primary_path: data_root.join("dsv4_megamoe_module_perf.parquet"),
            module: OnceLock::new(),
        }
    }

    /// Query the measured MegaMoE routed-module latency (ms) at `num_tokens`
    /// (rank-LOCAL token count — callers must NOT pre-multiply by
    /// attention_dp_size). Mirrors the table body of Python
    /// `_query_megamoe_table`: exact key walk, then the 1-axis Grid engine
    /// with the linear token-proxy SOL.
    #[allow(clippy::too_many_arguments)]
    pub fn query_module(
        &self,
        num_tokens: u32,
        hidden_size: u32,
        inter_size: u32,
        topk: u32,
        num_experts: u32,
        moe_tp_size: u32,
        moe_ep_size: u32,
        quant: MoeQuantMode,
        workload_distribution: &str,
        is_context: bool,
        source_policy: &str,
        pre_dispatch: &str,
        num_fused_shared_experts: u32,
        kernel_source: &str,
        kernel_dtype: &str,
    ) -> Result<f64, AicError> {
        let grids = self.load_module()?;
        let phase = if is_context { "context" } else { "generation" };
        let key = Dsv4MegaMoeKey {
            phase: phase.to_string(),
            kernel_source: kernel_source.to_string(),
            kernel_dtype: kernel_dtype.to_string(),
            quant: quant.name().to_string(),
            pre_dispatch: pre_dispatch.to_string(),
            source_policy: source_policy.to_string(),
            distribution: workload_distribution.to_string(),
            topk,
            num_experts,
            num_fused_shared_experts,
            hidden_size,
            inter_size,
            moe_tp_size,
            moe_ep_size,
        };
        let curve = grids.by_keys.get(&key).ok_or_else(|| {
            // Python's KeyError -> PerfDataNotAvailableError message.
            AicError::PerfDatabase(format!(
                "No DSv4 MegaMoE {phase} module data for kernel_source={kernel_source:?}, \
                 kernel_dtype={kernel_dtype:?}, quant_mode={}, pre_dispatch={pre_dispatch:?}, \
                 source_policy={source_policy:?}, workload_distribution={workload_distribution:?}, \
                 topk={topk}, num_experts={num_experts}, \
                 num_fused_shared_experts={num_fused_shared_experts}, hidden_size={hidden_size}, \
                 inter_size={inter_size}, moe_tp_size={moe_tp_size}, moe_ep_size={moe_ep_size}.",
                quant.name()
            ))
        })?;
        // Python: OpInterpConfig(axes=("num_tokens",), resolver=Grid(),
        // sol_fn=lambda t: float(t)) — in-range RAW lerp, boundary util-hold
        // beyond the collected range with the linear token proxy.
        query_token_curve(curve, f64::from(num_tokens), &|t| t)
    }

    fn load_module(&self) -> Result<&Dsv4MegaMoeGrids, AicError> {
        let cell = self
            .module
            .get_or_init(|| load_module_parquet(&self.primary_path));
        cell.as_ref().map_err(clone_err)
    }
}

/// Load the unified MegaMoE module parquet. Mirrors Python
/// `load_dsv4_megamoe_module_data` (see the module doc): required columns,
/// bool invariants, phase validation, duplicate-leaf ERROR. A missing file is
/// a typed miss (Python: `LoadedOpData.raise_if_not_loaded` ->
/// `PerfDataNotAvailableError`).
fn load_module_parquet(path: &PathBuf) -> Result<Dsv4MegaMoeGrids, AicError> {
    if !path.exists() {
        return Err(AicError::PerfDatabase(format!(
            "DSv4 MegaMoE module data not loaded: perf file not found at {}. This combination \
             of model, system, backend, and backend version is not supported by AIC in SILICON \
             mode.",
            path.display()
        )));
    }
    let reader = PerfReader::open(path)?;
    // Required columns (a missing column fails the load, matching Python's
    // KeyError / phase ValueError contract).
    let phase_col = reader.col("phase")?;
    let kernel_dtype_col = reader.col("kernel_dtype")?;
    let moe_dtype_col = reader.col("moe_dtype")?;
    let pre_dispatch_col = reader.col("pre_dispatch")?;
    let source_policy_col = reader.col("source_policy")?;
    let distribution_col = reader.col("distribution")?;
    let topk_col = reader.col("topk")?;
    let num_experts_col = reader.col("num_experts")?;
    let hidden_size_col = reader.col("hidden_size")?;
    let inter_size_col = reader.col("inter_size")?;
    let moe_ep_size_col = reader.col("moe_ep_size")?;
    let num_tokens_col = reader.col("num_tokens")?;
    let latency_col = reader.col("latency")?;
    // Python reads `row["routed_scaling_factor"]` unconditionally into the
    // leaf metadata (KeyError if absent), even though the query never
    // consumes it — require the column for load parity.
    let routed_scaling_col = reader.col("routed_scaling_factor")?;
    // Bool invariants: Python's defaults for absent columns are chosen so
    // that a missing column ALSO fails the invariant (used_cuda_graph
    // default None -> false != true; includes_gate_topk default "true" !=
    // false; includes_routed_scale default None -> false != true). Required
    // columns mirror that fail-on-absence exactly.
    let used_cuda_graph_col = reader.col("used_cuda_graph")?;
    let includes_gate_topk_col = reader.col("includes_gate_topk")?;
    let includes_routed_scale_col = reader.col("includes_routed_scale")?;
    // Optional columns with Python defaults.
    let kernel_source_col = reader.col_optional("kernel_source");
    let num_fused_shared_col = reader.col_optional("num_fused_shared_experts");
    let moe_tp_size_col = reader.col_optional("moe_tp_size");

    let mut by_keys: BTreeMap<Dsv4MegaMoeKey, BTreeMap<u32, f64>> = BTreeMap::new();
    for row in reader.rows()? {
        let row = row?;
        for (col, expected, error) in [
            (used_cuda_graph_col, true, "DSv4 MegaMoE perf row was not collected with CUDA Graph"),
            (
                includes_gate_topk_col,
                false,
                "DSv4 MegaMoE perf row includes gate/top-k outside the supported boundary",
            ),
            (
                includes_routed_scale_col,
                true,
                "DSv4 MegaMoE perf row does not include SGLang routed output scaling",
            ),
        ] {
            if row.bool(col)? != expected {
                return Err(AicError::PerfDatabase(format!("{error}: {}", path.display())));
            }
        }
        let phase = row.str_owned(phase_col)?;
        if phase != "context" && phase != "generation" {
            return Err(AicError::PerfDatabase(format!(
                "DSv4 MegaMoE perf row has unsupported phase={phase:?}: {}",
                path.display()
            )));
        }
        let moe_dtype = row.str_owned(moe_dtype_col)?;
        let Some(quant) = moe_dtype_from_name(&moe_dtype) else {
            return Err(AicError::PerfDatabase(format!(
                "DSv4 MegaMoE perf row has unknown moe_dtype={moe_dtype:?} at {}",
                path.display()
            )));
        };
        // routed_scaling_factor: read for the required-column/parse contract
        // only (the query never consumes it; energy is not modeled in Rust).
        let _ = row.f64(routed_scaling_col)?;
        let key = Dsv4MegaMoeKey {
            phase,
            kernel_source: row
                .str_optional(kernel_source_col)?
                .map(str::to_string)
                .unwrap_or_else(|| "deepgemm_megamoe".to_string()),
            kernel_dtype: row.str_owned(kernel_dtype_col)?,
            quant: quant.name().to_string(),
            pre_dispatch: row.str_owned(pre_dispatch_col)?,
            source_policy: row.str_owned(source_policy_col)?,
            distribution: row.str_owned(distribution_col)?,
            topk: row.u32(topk_col)?,
            num_experts: row.u32(num_experts_col)?,
            num_fused_shared_experts: row.u32_optional(num_fused_shared_col)?.unwrap_or(0),
            hidden_size: row.u32(hidden_size_col)?,
            inter_size: row.u32(inter_size_col)?,
            moe_tp_size: row.u32_optional(moe_tp_size_col)?.unwrap_or(1),
            moe_ep_size: row.u32(moe_ep_size_col)?,
        };
        let num_tokens = row.u32(num_tokens_col)?;
        let latency = row.f64(latency_col)?;
        // Python `_put_nested`: a duplicate leaf is a load ERROR, not
        // last-wins.
        if by_keys
            .entry(key.clone())
            .or_default()
            .insert(num_tokens, latency)
            .is_some()
        {
            return Err(AicError::PerfDatabase(format!(
                "duplicate DSv4 MegaMoE data row for {} {key:?} num_tokens={num_tokens}",
                path.display()
            )));
        }
    }
    Ok(Dsv4MegaMoeGrids { by_keys })
}

fn clone_err(err: &AicError) -> AicError {
    AicError::PerfDatabase(err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moe_dtype_names_mirror_python_member_lookup() {
        assert_eq!(
            moe_dtype_from_name("w4a8_mxfp4_mxfp8"),
            Some(MoeQuantMode::W4a8Mxfp4Mxfp8)
        );
        assert_eq!(moe_dtype_from_name("fp8_block"), Some(MoeQuantMode::Fp8Block));
        assert_eq!(moe_dtype_from_name("not_a_dtype"), None);
    }

    /// Missing perf file is a typed miss (Python
    /// `LoadedOpData.raise_if_not_loaded` -> `PerfDataNotAvailableError`).
    #[test]
    fn missing_file_is_typed_miss() {
        let table = Dsv4MegaMoeTable::new(PathBuf::from("/nonexistent/dir"));
        let err = table
            .query_module(
                1024,
                7168,
                3072,
                6,
                384,
                1,
                8,
                MoeQuantMode::W4a8Mxfp4Mxfp8,
                "balanced",
                true,
                "random",
                "sglang_jit",
                0,
                "deepgemm_megamoe",
                "fp8_fp4",
            )
            .unwrap_err();
        assert!(err.is_missing_perf_data(), "got {err:?}");
        assert!(err.to_string().contains("DSv4 MegaMoE module data not loaded"));
    }
}
