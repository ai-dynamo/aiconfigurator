// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! TensorRT-LLM WideEP MoE *compute* perf table.
//!
//! `wideep_moe_perf.txt` (Python `PerfDataFilename.wideep_moe_compute`).
//! Pure-compute kernel timing (no All2All) for the WideEP execution
//! path. The dispatch / combine cost is modeled separately by the
//! `wideep` (DeepEP / TRT-LLM All2All) table.
//!
//! CSV columns: framework, version, device, op_name, kernel_source,
//! moe_dtype, moe_kernel, num_tokens, dp_num_tokens, rank0_num_tokens,
//! hidden_size, inter_size, topk, num_experts, num_slots, moe_tp_size,
//! moe_ep_size, distribution, simulation_mode, latency.
//!
//! Loader nesting mirrors Python's
//! `data[kernel_source][quant][distribution][topk][num_experts][hidden]`
//! `[inter][num_slots][moe_tp_size][moe_ep_size][num_tokens] = latency`.
//! At query time the leaf `num_tokens` axis is 1-D interpolated.
//!
//! `kernel_source` identifies the WideEP MoE compute kernel:
//!   - `moe_torch_flow` (Cutlass; default for SM < 100)
//!   - `deepgemm` (SM >= 100 with fp8_block)
//! `distribution` carries the workload-distribution string used by the
//! `MoEModel`/`MoeOp`, e.g. `power_law_1.01` or `power_law_1.01_eplb`
//! (the `_eplb` suffix selects the Expert Parallel Load Balancer
//! variants used by the TRT-LLM WideEP path).

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::OnceLock;

use crate::common::enums::MoeQuantMode;
use crate::common::error::AicError;
use crate::config::{PerfDbSources, PerfSource};
use super::{kernel_source_ok, resolve_op_sources};
use super::moe::query_token_curve;
use crate::perf_database::parquet_loader::PerfReader;

pub struct WideEpMoeTable {
    data_root: PathBuf,
    /// Ordered, priority-sorted sources for the WideEP MoE compute perf file
    /// (shared-layer aware; see [`PerfSource`]). Single-primary, no-filter by
    /// default (`WideEpMoeTable::new`).
    wideep_moe_sources: Vec<PerfSource>,
    compute: OnceLock<Result<WideEpMoeGrids, AicError>>,
}

/// `(kernel_source, quant, distribution, topk, num_experts, hidden, inter,
///   num_slots, moe_tp, moe_ep)` -> `num_tokens -> latency`.
pub struct WideEpMoeGrids {
    pub by_keys: BTreeMap<WideEpMoeKey, BTreeMap<u32, f64>>,
    /// First-seen (file row order) distribution per `(kernel_source, quant)`.
    /// Python's fallback takes `list(quant_data.keys())[0]` — dict INSERTION
    /// order, i.e. the distribution of the first row loaded for that
    /// `(kernel, quant)` — which differs from sorted order on real shards
    /// (e.g. gb200/h100 wideep files start with `power_law_1.01_eplb`).
    pub first_distribution: BTreeMap<(String, String), String>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct WideEpMoeKey {
    pub kernel_source: String,
    pub quant: String,
    pub distribution: String,
    pub topk: u32,
    pub num_experts: u32,
    pub hidden_size: u32,
    pub inter_size: u32,
    pub num_slots: u32,
    pub moe_tp_size: u32,
    pub moe_ep_size: u32,
}

/// `kernel_source` defaults to `"moe_torch_flow"` when null, matching Python's
/// `load_wideep_moe_compute_data` behavior.
const DEFAULT_KERNEL_SOURCE: &str = "moe_torch_flow";

impl WideEpMoeTable {
    /// Construct an empty table for the given data directory. No I/O. The perf
    /// file is sourced solely from `data_root/wideep_moe_perf.parquet` with no
    /// `kernel_source` filter (pre-shared-layer behaviour).
    pub fn new(data_root: PathBuf) -> Self {
        Self::with_sources(data_root, &PerfDbSources::default())
    }

    /// Construct with shared-layer (sibling/cross-version) sources resolved from
    /// `perf_db_sources` (Python-supplied). Falls back to the primary
    /// `data_root/wideep_moe_perf.parquet` when absent from the map. No I/O.
    pub fn with_sources(data_root: PathBuf, perf_db_sources: &PerfDbSources) -> Self {
        let wideep_moe_sources =
            resolve_op_sources(perf_db_sources, "wideep_moe_perf.parquet", &data_root);
        Self {
            data_root,
            wideep_moe_sources,
            compute: OnceLock::new(),
        }
    }

    /// Query WideEP MoE compute latency at `num_tokens` along the
    /// `(kernel_source, quant, distribution, topk, num_experts, hidden,
    /// inter, num_slots, moe_tp_size, moe_ep_size)` key. The token curve
    /// rides the perf_interp v2 engine (1-axis Grid, RAW lerp in range,
    /// boundary util-hold beyond it), mirroring the silicon body of Python's
    /// `TrtLLMWideEPMoE._query_compute_table`. `kernel_source` must be the
    /// RESOLVED kernel (the operator mirrors `_select_kernel`, including its
    /// data-availability fallback via [`Self::available_kernels`]); the
    /// distribution falls back level-wise exactly like Python's
    /// `get_silicon` (first distribution under the matched `(kernel, quant)`
    /// when the exact string is absent — see [`Self::slice_points`]).
    ///
    /// `sol` is the WideEP MoE roofline (num_slots-aware) the beyond-range
    /// util-hold anchors on — Python threads the same closure through
    /// `OpInterpConfig(sol_fn=...)`. In-range lerp never consults it.
    #[allow(clippy::too_many_arguments)]
    pub fn query_compute(
        &self,
        num_tokens: u32,
        hidden_size: u32,
        inter_size: u32,
        topk: u32,
        num_experts: u32,
        num_slots: u32,
        moe_tp_size: u32,
        moe_ep_size: u32,
        quant: MoeQuantMode,
        distribution: &str,
        kernel_source: &str,
        sol: &dyn Fn(f64) -> f64,
    ) -> Result<f64, AicError> {
        let by_tokens = self.resolve_slice(
            kernel_source,
            quant.name(),
            distribution,
            topk,
            num_experts,
            hidden_size,
            inter_size,
            num_slots,
            moe_tp_size,
            moe_ep_size,
        )?;
        query_token_curve(by_tokens, num_tokens as f64, sol)
    }

    /// Own-slice `num_tokens -> latency_ms` points, after the level-wise
    /// distribution fallback. Typed `AicError::PerfDatabase` miss when the
    /// slice is absent or empty — mirrors the `_slice` closure of Python
    /// `_query_compute_table.get_empirical_from_sol`
    /// (`util_empirical.require_data_slice` semantics).
    #[allow(clippy::too_many_arguments)]
    pub fn slice_points(
        &self,
        kernel_source: &str,
        quant: MoeQuantMode,
        distribution: &str,
        topk: u32,
        num_experts: u32,
        hidden_size: u32,
        inter_size: u32,
        num_slots: u32,
        moe_tp_size: u32,
        moe_ep_size: u32,
    ) -> Result<Vec<(u32, f64)>, AicError> {
        let by_tokens = self.resolve_slice(
            kernel_source,
            quant.name(),
            distribution,
            topk,
            num_experts,
            hidden_size,
            inter_size,
            num_slots,
            moe_tp_size,
            moe_ep_size,
        )?;
        Ok(by_tokens.iter().map(|(&t, &lat)| (t, lat)).collect())
    }

    /// Distinct kernel names present in the loaded table, in sorted
    /// (BTreeMap) order; EMPTY when the table failed to load with a typed
    /// data miss. Mirrors the `if database._wideep_moe_compute_data:` guard
    /// of Python `_select_kernel` (an unloaded/empty table is falsy there —
    /// the caller then keeps its architecture-preferred kernel). NOTE:
    /// Python yields dict insertion (file row) order; sorted order is
    /// observable only when several kernels coexist and the preferred one is
    /// absent.
    pub fn available_kernels(&self) -> Result<Vec<String>, AicError> {
        let grids = match self.load_compute() {
            Ok(grids) => grids,
            Err(err) if err.is_missing_perf_data() => return Ok(Vec::new()),
            Err(err) => return Err(err),
        };
        let mut names: Vec<String> = Vec::new();
        for key in grids.by_keys.keys() {
            // `WideEpMoeKey` sorts by kernel_source first, so consecutive
            // dedup suffices.
            if names.last().map(String::as_str) != Some(key.kernel_source.as_str()) {
                names.push(key.kernel_source.clone());
            }
        }
        Ok(names)
    }

    /// Level-wise slice resolution mirroring Python's nested-dict walk:
    /// `require_data_slice(data, kernel)` -> `require_data_slice(kd, quant)`
    /// -> distribution fallback (`workload if present else first available
    /// under (kernel, quant)`) -> exact remaining coordinate. Each level
    /// misses with a typed `AicError::PerfDatabase`. The Python fallback
    /// takes the FIRST distribution in dict-insertion (file row) order —
    /// served here from the load-time `first_distribution` map — and then
    /// requires the full shape under it (a shape present only under a later
    /// distribution still misses).
    #[allow(clippy::too_many_arguments)]
    fn resolve_slice(
        &self,
        kernel_source: &str,
        quant_name: &str,
        distribution: &str,
        topk: u32,
        num_experts: u32,
        hidden_size: u32,
        inter_size: u32,
        num_slots: u32,
        moe_tp_size: u32,
        moe_ep_size: u32,
    ) -> Result<&BTreeMap<u32, f64>, AicError> {
        let grids = self.load_compute()?;
        let mut kernel_seen = false;
        let mut quant_seen = false;
        let mut requested_distribution_seen = false;
        for key in grids.by_keys.keys() {
            if key.kernel_source != kernel_source {
                continue;
            }
            kernel_seen = true;
            if key.quant != quant_name {
                continue;
            }
            quant_seen = true;
            if key.distribution == distribution {
                requested_distribution_seen = true;
                break;
            }
        }
        if !kernel_seen {
            return Err(AicError::PerfDatabase(format!(
                "WideEP MoE compute data missing for kernel_source={kernel_source:?} at {}",
                self.data_root.display()
            )));
        }
        if !quant_seen {
            return Err(AicError::PerfDatabase(format!(
                "WideEP MoE compute data missing for kernel_source={kernel_source:?} \
                 quant={quant_name:?} at {}",
                self.data_root.display()
            )));
        }
        let dist = if requested_distribution_seen {
            distribution
        } else {
            grids
                .first_distribution
                .get(&(kernel_source.to_string(), quant_name.to_string()))
                .expect("quant_seen implies a recorded first distribution")
        };
        let key = WideEpMoeKey {
            kernel_source: kernel_source.to_string(),
            quant: quant_name.to_string(),
            distribution: dist.to_string(),
            topk,
            num_experts,
            hidden_size,
            inter_size,
            num_slots,
            moe_tp_size,
            moe_ep_size,
        };
        grids
            .by_keys
            .get(&key)
            .filter(|curve| !curve.is_empty())
            .ok_or_else(|| {
                AicError::PerfDatabase(format!(
                    "WideEP MoE compute data missing for {key:?} at {}",
                    self.data_root.display()
                ))
            })
    }

    fn load_compute(&self) -> Result<&WideEpMoeGrids, AicError> {
        let cell = self
            .compute
            .get_or_init(|| load_compute_parquet(&self.wideep_moe_sources));
        cell.as_ref().map_err(clone_err)
    }
}

/// Load the WideEP MoE compute table from an ordered, priority-sorted source
/// list. Sources are read in order; the first source containing a key wins
/// (`or_insert`). Missing files are skipped (a sibling declared in the manifest
/// need not exist for every system); an error is returned only when no source
/// yields rows.
fn load_compute_parquet(sources: &[PerfSource]) -> Result<WideEpMoeGrids, AicError> {
    let mut by_keys: BTreeMap<WideEpMoeKey, BTreeMap<u32, f64>> = BTreeMap::new();
    let mut first_distribution: BTreeMap<(String, String), String> = BTreeMap::new();
    let mut any_source = false;
    for source in sources {
        let path = source.path();
        if !path.exists() {
            continue;
        }
        any_source = true;
        let reader = PerfReader::open(path)?;
        let kernel_source_col = reader.col_optional("kernel_source");
        let moe_dtype_col = reader.col("moe_dtype")?;
        let num_tokens_col = reader.col("num_tokens")?;
        let hidden_size_col = reader.col("hidden_size")?;
        let inter_size_col = reader.col("inter_size")?;
        let topk_col = reader.col("topk")?;
        let num_experts_col = reader.col("num_experts")?;
        let num_slots_col = reader.col("num_slots")?;
        let moe_tp_size_col = reader.col("moe_tp_size")?;
        let moe_ep_size_col = reader.col("moe_ep_size")?;
        let distribution_col = reader.col("distribution")?;
        let latency_col = reader.col("latency")?;

        for row in reader.rows()? {
            let row = row?;
            if !kernel_source_ok(source.kernel_sources(), kernel_source_col, &row)? {
                continue;
            }
            // `kernel_source` is optional/nullable in the perf file; default to
            // "moe_torch_flow" when absent, matching Python's loader.
            let kernel_source = row
                .str_optional(kernel_source_col)?
                .map(|s| s.to_string())
                .unwrap_or_else(|| DEFAULT_KERNEL_SOURCE.to_string());
            let key = WideEpMoeKey {
                kernel_source,
                quant: row.str_owned(moe_dtype_col)?,
                distribution: row.str_owned(distribution_col)?,
                topk: row.u32(topk_col)?,
                num_experts: row.u32(num_experts_col)?,
                hidden_size: row.u32(hidden_size_col)?,
                inter_size: row.u32(inter_size_col)?,
                num_slots: row.u32(num_slots_col)?,
                moe_tp_size: row.u32(moe_tp_size_col)?,
                moe_ep_size: row.u32(moe_ep_size_col)?,
            };
            // First-seen (file order) distribution per (kernel, quant) — the
            // distribution-fallback anchor (Python dict insertion order).
            first_distribution
                .entry((key.kernel_source.clone(), key.quant.clone()))
                .or_insert_with(|| key.distribution.clone());
            // Last-wins parity with Python `load_wideep_moe_compute_data`
            // (moe.py): it direct-assigns per coordinate with no
            // `try/except KeyError` guard, so a later row overwrites an
            // earlier one — both within a file and across the concatenated
            // shared-layer sources (`_read_filtered_rows` appends in source
            // order and the loader assigns in row order). Real shards carry
            // duplicate keys with differing latencies (e.g. 270 keys in
            // rtx_pro_6000_server/trtllm/1.3.0rc10), so first-wins here was a
            // live numeric divergence, same class as the MLA-module fix in
            // `mla.rs::load_module_parquet`.
            by_keys
                .entry(key)
                .or_default()
                .insert(row.u32(num_tokens_col)?, row.f64(latency_col)?);
        }
    }
    if !any_source || by_keys.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "no WideEP MoE compute rows loaded from {} source(s) (first: {})",
            sources.len(),
            sources.first().map(|s| s.path().display().to_string()).unwrap_or_default()
        )));
    }
    Ok(WideEpMoeGrids { by_keys, first_distribution })
}

fn clone_err(err: &AicError) -> AicError {
    AicError::PerfDatabase(err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    const REPO_ROOT_HINT: &str = env!("CARGO_MANIFEST_DIR");

    fn b200_trtllm_data_root() -> PathBuf {
        PathBuf::from(REPO_ROOT_HINT)
            .join("../..")
            .join("src/aiconfigurator_core/systems/data/b200_sxm/trtllm/1.3.0rc10")
    }

    #[test]
    fn wideep_moe_compute_exact_hit() {
        // First row of b200_sxm/trtllm/1.3.0rc10/wideep_moe_perf.txt:
        // kernel=wideep_compute_cutlass moe_dtype=nvfp4 num_tokens=1
        // hidden=6144 inter=2048 topk=8 num_experts=256 num_slots=256
        // moe_tp=1 moe_ep=2 distribution=power_law_1.01 latency=0.08600...
        let table = WideEpMoeTable::new(b200_trtllm_data_root());
        let latency = table
            .query_compute(
                1,
                6144,
                2048,
                8,
                256,
                256,
                1,
                2,
                MoeQuantMode::Nvfp4,
                "power_law_1.01",
                "wideep_compute_cutlass",
                &|t| t,
            )
            .expect("WideEP MoE compute query must succeed");
        assert!(
            (latency - 0.086_009_597_778_320_32).abs() < 1e-6,
            "expected recorded latency, got {latency}"
        );
        assert_eq!(
            table.available_kernels().expect("kernels list"),
            vec!["wideep_compute_cutlass".to_string()]
        );
    }

    #[test]
    fn wideep_moe_duplicate_key_last_row_wins() {
        // rtx_pro_6000_server/trtllm/1.3.0rc10/wideep_moe_perf.parquet carries
        // 270 duplicate coordinates with differing latencies. Python's
        // `load_wideep_moe_compute_data` direct-assigns (last row wins); this
        // key's first occurrence is 0.11578880548477173, its last is
        // 0.11609599590301514 — the loaded grid must hold the LAST value.
        let root = PathBuf::from(REPO_ROOT_HINT)
            .join("../..")
            .join("src/aiconfigurator_core/systems/data/rtx_pro_6000_server/trtllm/1.3.0rc10");
        let table = WideEpMoeTable::new(root);
        let latency = table
            .query_compute(
                1,
                7168,
                2048,
                8,
                384,
                384,
                1,
                2,
                MoeQuantMode::Nvfp4,
                "power_law_1.01",
                "wideep_compute_cutlass",
                &|t| t,
            )
            .expect("WideEP MoE compute query must succeed");
        assert!(
            (latency - 0.116_095_995_903_015_14).abs() < 1e-9,
            "duplicate key must resolve last-wins (Python parity): got {latency}, \
             first-wins would give 0.11578880548477173"
        );
    }
}
