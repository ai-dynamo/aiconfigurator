// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Basic MoE perf table.
//!
//! Mirrors the raw SILICON-path layout of
//! `aiconfigurator_core.sdk.operations.moe.MoE._query_moe_table`:
//!
//! `moe_data[quant][distribution][topk][num_experts][hidden][inter][moe_tp][moe_ep]`
//! returns a `{num_tokens -> latency_ms}` dict.
//!
//! Query is 1-D linear interpolation along `num_tokens` (extrapolation
//! allowed). Any SOL-anchored overflow estimation for queries beyond the
//! largest recorded token count is the model layer's responsibility; this
//! raw table query simply 1-D-interpolates with extrapolation when needed.
//!
//! `workload_distribution` falls back to `"uniform"` when the requested
//! variant is absent for the given quant, matching Python's behavior.
//!
//! WideEP / DeepEP / TRT-LLM all-to-all variants live in
//! `perf_database::wideep`, `wideep_mla`, and `wideep_moe`.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::OnceLock;

use crate::common::enums::MoeQuantMode;
use crate::common::error::AicError;
use crate::config::{PerfDbSources, PerfSource};
use super::{kernel_source_ok, resolve_op_sources};
use super::interpolation::{interp_1d, nearest_neighbors};
use crate::perf_database::parquet_loader::PerfReader;

pub struct MoeTable {
    data_root: PathBuf,
    /// Ordered, priority-sorted sources for the MoE perf file (shared-layer
    /// aware; see [`PerfSource`]). Single-primary, no-filter by default
    /// (`MoeTable::new`).
    moe_sources: Vec<PerfSource>,
    moe: OnceLock<Result<LoadedMoeGrids, AicError>>,
}

/// Two parallel grids split by `kernel_source`. Mirrors Python's split in
/// `aiconfigurator_core.sdk.operations.moe.MoE.load_data`, where rows tagged
/// `kernel_source == "moe_torch_flow_min_latency"` route to a separate
/// accumulator that the TRT-LLM SILICON path probes first for small-token
/// nvfp4 gated MoE queries.
struct LoadedMoeGrids {
    default: MoeGrids,
    low_latency: MoeGrids,
}

struct MoeGrids {
    by_keys: BTreeMap<MoeKey, BTreeMap<u32, f64>>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct MoeKey {
    quant: String,
    distribution: String,
    topk: u32,
    num_experts: u32,
    hidden_size: u32,
    inter_size: u32,
    moe_tp_size: u32,
    moe_ep_size: u32,
}

impl MoeTable {
    /// Construct an empty table for the given data directory. No I/O. The MoE
    /// perf file is sourced solely from `data_root/moe_perf.parquet` with no
    /// `kernel_source` filter (pre-shared-layer behaviour).
    pub fn new(data_root: PathBuf) -> Self {
        Self::with_sources(data_root, &PerfDbSources::default())
    }

    /// Construct with shared-layer (sibling/cross-version) sources resolved from
    /// `perf_db_sources` (Python-supplied). The MoE file falls back to its
    /// primary `data_root/moe_perf.parquet` when absent from the map. No I/O.
    pub fn with_sources(data_root: PathBuf, perf_db_sources: &PerfDbSources) -> Self {
        let moe_sources = resolve_op_sources(perf_db_sources, "moe_perf.parquet", &data_root);
        Self {
            data_root,
            moe_sources,
            moe: OnceLock::new(),
        }
    }

    /// Raw MoE latency in ms by 1-D interpolation along `num_tokens`.
    ///
    /// Falls back to the `"uniform"` distribution if the requested
    /// distribution is absent for the given quant mode. Extrapolates beyond
    /// the largest recorded `num_tokens` via linear extension; operator
    /// layer is responsible for SOL-anchored utilization correction for
    /// overflow queries.
    pub fn query(
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
    ) -> Result<f64, AicError> {
        let loaded = self.load()?;
        let grids = &loaded.default;
        let quant_name = quant.name();

        let dist = self.resolve_distribution(grids, quant_name, workload_distribution);
        let key = MoeKey {
            quant: quant_name.to_string(),
            distribution: dist,
            topk,
            num_experts,
            hidden_size,
            inter_size,
            moe_tp_size,
            moe_ep_size,
        };
        let by_tokens = grids.by_keys.get(&key).ok_or_else(|| {
            AicError::PerfDatabase(format!(
                "MoE data missing for {key:?} at {}",
                self.data_root.display()
            ))
        })?;

        // Exact hit.
        if let Some(&latency) = by_tokens.get(&num_tokens) {
            return Ok(latency);
        }
        // 1-D interpolation (extrapolation allowed).
        let token_keys: Vec<u32> = by_tokens.keys().copied().collect();
        if token_keys.is_empty() {
            return Err(AicError::PerfDatabase(format!(
                "MoE data has no token points for {key:?} at {}",
                self.data_root.display()
            )));
        }
        let (lo, hi) = nearest_neighbors(num_tokens, &token_keys, false)?;
        Ok(interp_1d(
            lo as f64,
            hi as f64,
            by_tokens[&lo],
            by_tokens[&hi],
            num_tokens as f64,
        ))
    }

    /// Largest recorded `num_tokens` for a given (quant, distribution,
    /// topology) tuple. Used by the operator layer to decide whether to
    /// invoke SOL-anchored overflow estimation.
    pub fn max_token_point(
        &self,
        hidden_size: u32,
        inter_size: u32,
        topk: u32,
        num_experts: u32,
        moe_tp_size: u32,
        moe_ep_size: u32,
        quant: MoeQuantMode,
        workload_distribution: &str,
    ) -> Result<Option<u32>, AicError> {
        let loaded = self.load()?;
        let grids = &loaded.default;
        let quant_name = quant.name();
        let dist = self.resolve_distribution(grids, quant_name, workload_distribution);
        let key = MoeKey {
            quant: quant_name.to_string(),
            distribution: dist,
            topk,
            num_experts,
            hidden_size,
            inter_size,
            moe_tp_size,
            moe_ep_size,
        };
        Ok(grids
            .by_keys
            .get(&key)
            .and_then(|m| m.keys().next_back().copied()))
    }

    /// Probe the TRT-LLM low-latency NVFP4 MoE kernel table.
    ///
    /// Returns `Ok(Some(latency_ms))` when the loaded `low_latency` grid
    /// contains a matching `(quant, distribution-after-uniform-fallback,
    /// topk, num_experts, hidden, inter, moe_tp, moe_ep)` entry, and
    /// `Ok(None)` when the shape is absent — the caller should then fall
    /// through to `query()` (the default grid).
    ///
    /// Mirrors Python's small-token nvfp4 gated-MoE branch in
    /// `MoE._query_moe_table`: the low-latency table is consulted with a
    /// try/except KeyError that falls back to `_moe_data` on miss.
    pub fn query_low_latency(
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
    ) -> Result<Option<f64>, AicError> {
        let loaded = self.load()?;
        let grids = &loaded.low_latency;
        if grids.by_keys.is_empty() {
            return Ok(None);
        }
        let quant_name = quant.name();
        let dist = self.resolve_distribution(grids, quant_name, workload_distribution);
        let key = MoeKey {
            quant: quant_name.to_string(),
            distribution: dist,
            topk,
            num_experts,
            hidden_size,
            inter_size,
            moe_tp_size,
            moe_ep_size,
        };
        let Some(by_tokens) = grids.by_keys.get(&key) else {
            return Ok(None);
        };
        if by_tokens.is_empty() {
            return Ok(None);
        }
        if let Some(&latency) = by_tokens.get(&num_tokens) {
            return Ok(Some(latency));
        }
        let token_keys: Vec<u32> = by_tokens.keys().copied().collect();
        let (lo, hi) = nearest_neighbors(num_tokens, &token_keys, false)?;
        Ok(Some(interp_1d(
            lo as f64,
            hi as f64,
            by_tokens[&lo],
            by_tokens[&hi],
            num_tokens as f64,
        )))
    }

    /// `true` iff the loaded low-latency grid has any rows.
    ///
    /// Older perf-DB versions predate the `kernel_source` column, so the
    /// low-latency accumulator stays empty and the small-token nvfp4 gate
    /// is short-circuited at the operator layer.
    pub fn low_latency_available(&self) -> Result<bool, AicError> {
        let loaded = self.load()?;
        Ok(!loaded.low_latency.by_keys.is_empty())
    }

    /// Mirrors Python's:
    /// `dist = workload if workload in moe_data[quant] else "uniform"`
    fn resolve_distribution(
        &self,
        grids: &MoeGrids,
        quant: &str,
        workload_distribution: &str,
    ) -> String {
        // Check whether any key with this (quant, distribution) exists.
        let requested_exists = grids
            .by_keys
            .keys()
            .any(|k| k.quant == quant && k.distribution == workload_distribution);
        if requested_exists {
            workload_distribution.to_string()
        } else {
            "uniform".to_string()
        }
    }

    fn load(&self) -> Result<&LoadedMoeGrids, AicError> {
        let cell = self
            .moe
            .get_or_init(|| load_moe_parquet(&self.moe_sources));
        cell.as_ref().map_err(clone_err)
    }
}

/// Load the MoE table from an ordered, priority-sorted source list. Sources are
/// read in order; the first source containing a `(shape, num_tokens)` tuple wins
/// (`or_insert`), mirroring Python's `_read_filtered_rows` concatenation +
/// `load_moe_data` skip-on-key-conflict. Missing files are skipped (a sibling
/// declared in the manifest need not exist for every system); an error is
/// returned only when no source yields rows.
fn load_moe_parquet(sources: &[PerfSource]) -> Result<LoadedMoeGrids, AicError> {
    let mut default_keys: BTreeMap<MoeKey, BTreeMap<u32, f64>> = BTreeMap::new();
    let mut low_latency_keys: BTreeMap<MoeKey, BTreeMap<u32, f64>> = BTreeMap::new();
    let mut any_source = false;
    for source in sources {
        let path = source.path();
        if !path.exists() {
            continue;
        }
        any_source = true;
        let reader = PerfReader::open(path)?;
        let moe_dtype_col = reader.col("moe_dtype")?;
        let num_tokens_col = reader.col("num_tokens")?;
        let hidden_size_col = reader.col("hidden_size")?;
        let inter_size_col = reader.col("inter_size")?;
        let topk_col = reader.col("topk")?;
        let num_experts_col = reader.col("num_experts")?;
        let moe_tp_size_col = reader.col("moe_tp_size")?;
        let moe_ep_size_col = reader.col("moe_ep_size")?;
        let distribution_col = reader.col("distribution")?;
        let latency_col = reader.col("latency")?;
        // Optional in older perf-DB versions; when absent every row falls into
        // the `default` grid (matching the pre-split behavior). The same column
        // gates the per-source shared-layer `kernel_source` allowlist.
        let kernel_source_col = reader.col_optional("kernel_source");
        for row in reader.rows()? {
            let row = row?;
            if !kernel_source_ok(source.kernel_sources(), kernel_source_col, &row)? {
                continue;
            }
            let key = MoeKey {
                quant: row.str_owned(moe_dtype_col)?,
                distribution: row.str_owned(distribution_col)?,
                topk: row.u32(topk_col)?,
                num_experts: row.u32(num_experts_col)?,
                hidden_size: row.u32(hidden_size_col)?,
                inter_size: row.u32(inter_size_col)?,
                moe_tp_size: row.u32(moe_tp_size_col)?,
                moe_ep_size: row.u32(moe_ep_size_col)?,
            };
            let kernel_source = row.str_optional(kernel_source_col)?.unwrap_or("");
            let target = if kernel_source == "moe_torch_flow_min_latency" {
                &mut low_latency_keys
            } else {
                &mut default_keys
            };
            // Python's `load_moe_data` wraps the leaf insert in a try/except KeyError
            // and skips on conflict, i.e. it keeps the FIRST occurrence of each
            // (shape, num_tokens) tuple. Some perf files contain duplicate rows
            // (same kernel_source, same shape) — preserving first-wins parity here,
            // extended across shared-layer sources (earlier source wins).
            target
                .entry(key)
                .or_default()
                .entry(row.u32(num_tokens_col)?)
                .or_insert(row.f64(latency_col)?);
        }
    }
    if !any_source || (default_keys.is_empty() && low_latency_keys.is_empty()) {
        return Err(AicError::PerfDatabase(format!(
            "no rows loaded from {} source(s) (first: {})",
            sources.len(),
            sources.first().map(|s| s.path().display().to_string()).unwrap_or_default()
        )));
    }
    Ok(LoadedMoeGrids {
        default: MoeGrids { by_keys: default_keys },
        low_latency: MoeGrids { by_keys: low_latency_keys },
    })
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
            .join("src/aiconfigurator_core/systems/data/b200_sxm/vllm/0.19.0")
    }

    #[test]
    fn moe_table_loads_b200_vllm() {
        let table = MoeTable::new(b200_vllm_data_root());
        let _ = table.load().expect("moe_perf.parquet must load");
    }

    #[test]
    fn moe_distribution_falls_back_to_uniform() {
        // Pick any common smoke shape; non-existent distribution should
        // fall back without erroring.
        let table = MoeTable::new(b200_vllm_data_root());
        // Use a shape that's likely covered by vLLM b200 data; if not,
        // the error should be about the topology key, not about
        // missing distribution.
        let result = table.query(
            1024,
            4096,
            2048,
            2,
            128,
            1,
            8,
            MoeQuantMode::Bfloat16,
            "nonexistent_distribution",
        );
        // Either succeeds (uniform fallback found a match) or errors
        // with a topology mismatch — but not a distribution-specific
        // error.
        match result {
            Ok(latency) => assert!(latency > 0.0),
            Err(AicError::PerfDatabase(msg)) => {
                assert!(
                    !msg.contains("nonexistent_distribution"),
                    "expected uniform fallback, not literal distribution name in error: {msg}"
                );
            }
            Err(other) => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn moe_lazy_loads_once() {
        let table = MoeTable::new(b200_vllm_data_root());
        // Load twice; cached path should produce same outcome.
        let r1 = table.load();
        let r2 = table.load();
        assert_eq!(r1.is_ok(), r2.is_ok());
    }

    fn b200_trtllm_data_root() -> PathBuf {
        PathBuf::from(REPO_ROOT_HINT)
            .join("../..")
            .join("src/aiconfigurator_core/systems/data/b200_sxm/trtllm/1.2.0rc5")
    }

    #[test]
    fn moe_low_latency_grid_split_on_b200_trtllm() {
        // b200 trtllm 1.2.0rc5 perf-DB carries `moe_torch_flow_min_latency`
        // rows; they must land in the low_latency grid, not the default
        // one. vLLM/SGLang DBs lack the column entirely → low_latency
        // empty → `low_latency_available()` returns false.
        let table = MoeTable::new(b200_trtllm_data_root());
        let available = table
            .low_latency_available()
            .expect("moe_perf.parquet must load");
        assert!(
            available,
            "expected b200/trtllm/1.2.0rc5 to carry moe_torch_flow_min_latency rows"
        );

        let vllm = MoeTable::new(b200_vllm_data_root());
        let vllm_available = vllm
            .low_latency_available()
            .expect("vllm moe_perf.parquet must load");
        assert!(
            !vllm_available,
            "vLLM perf DB lacks kernel_source column → low_latency should be empty"
        );
    }
}
