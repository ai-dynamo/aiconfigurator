// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Basic MoE perf table.
//!
//! Mirrors the raw SILICON-path layout of
//! `aiconfigurator.sdk.operations.moe.MoE._query_moe_table`:
//!
//! `moe_data[quant][distribution][topk][num_experts][hidden][inter][moe_tp][moe_ep]`
//! returns a `{num_tokens -> latency_ms}` dict.
//!
//! Resolution mirrors Python v2's `_resolve_tokens`: the token curve rides
//! the shared `perf_interp` engine (1-axis Grid, RAW lerp in range; beyond
//! the collected range the boundary util is held with `k_tail=1` and the
//! caller-supplied MoE roofline SOL carries the growth — unclamped util,
//! exactly like Python which deleted the hand-rolled overflow estimator).
//! The SOL closure comes from the operator layer (`operators/moe.rs`),
//! which owns the roofline math.
//!
//! Singleton-underflow contract (Python `_require_moe_token_points`): a
//! curve with a single token point queried BELOW that point is a structured
//! miss — one large-token row cannot define the low-token launch floor.
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
use super::perf_interp::{self, Node, OpInterpConfig};
use crate::perf_database::parquet_loader::PerfReader;

/// Resolve a 1-axis `num_tokens -> latency_ms` curve on the perf_interp v2
/// engine: exact hit / RAW lerp in range; beyond the collected range the
/// engine holds the boundary util (`k_tail=1`) and lets `sol` carry the
/// growth. Shared by the MoE / WideEP / mHC token-curve families.
pub(crate) fn query_token_curve(
    curve: &BTreeMap<u32, f64>,
    num_tokens: f64,
    sol: &dyn Fn(f64) -> f64,
) -> Result<f64, AicError> {
    let mut node = Node::branch();
    for (&t, &lat) in curve {
        node.insert(&[t], lat);
    }
    let sol_slice = |c: &[f64]| sol(c[0]);
    let cfg = OpInterpConfig::grid(&["num_tokens"], &sol_slice);
    perf_interp::query(&cfg, &node, &[num_tokens])
}

/// Python `_require_moe_token_points`: a singleton curve queried below its
/// only measured point is a structured miss (it cannot define the low-token
/// launch-overhead regime). Multi-point underflow and singleton overflow go
/// to the engine's util-hold unchanged.
pub(crate) fn singleton_underflow(curve: &BTreeMap<u32, f64>, num_tokens: u32) -> Option<u32> {
    if curve.len() == 1 {
        let &only = curve.keys().next().expect("len checked");
        if num_tokens < only {
            return Some(only);
        }
    }
    None
}

pub struct MoeTable {
    data_root: PathBuf,
    /// Ordered, priority-sorted sources for the MoE perf file (shared-layer
    /// aware; see [`PerfSource`]). Single-primary, no-filter by default
    /// (`MoeTable::new`).
    moe_sources: Vec<PerfSource>,
    moe: OnceLock<Result<LoadedMoeGrids, AicError>>,
}

/// Two parallel grids split by `kernel_source`. Mirrors Python's split in
/// `aiconfigurator.sdk.operations.moe.MoE.load_data`, where rows tagged
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

    /// Raw MoE latency in ms via the perf_interp v2 engine (1-axis token
    /// curve): exact hit / RAW lerp in range; beyond the collected range the
    /// boundary util is held (`k_tail=1`, unclamped) and `sol` — the
    /// operator layer's MoE roofline — carries the growth. Mirrors Python
    /// `MoE._query_moe_table._resolve_tokens`.
    ///
    /// Falls back to the `"uniform"` distribution if the requested
    /// distribution is absent for the given quant mode. A singleton curve
    /// queried below its only point is a structured miss (Python
    /// `_require_moe_token_points`).
    #[allow(clippy::too_many_arguments)]
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
        sol: &dyn Fn(f64) -> f64,
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
        if by_tokens.is_empty() {
            return Err(AicError::PerfDatabase(format!(
                "MoE data has no token points for {key:?} at {}",
                self.data_root.display()
            )));
        }
        if let Some(only) = singleton_underflow(by_tokens, num_tokens) {
            return Err(AicError::PerfDatabase(format!(
                "MoE silicon token underflow has only one measured point; cannot infer \
                 low-token latency from a singleton. num_tokens={num_tokens}, \
                 measured_token={only}, key={key:?}"
            )));
        }
        query_token_curve(by_tokens, num_tokens as f64, sol)
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
    /// try/except that falls back to `_moe_data` when the SHAPE is absent
    /// (`Ok(None)` here). A singleton-underflow on a present shape is an
    /// `Err` (structured miss), not a fallback — in Python the guard fires
    /// inside `_resolve_tokens`, after the ll table has been selected.
    #[allow(clippy::too_many_arguments)]
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
        sol: &dyn Fn(f64) -> f64,
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
        if let Some(only) = singleton_underflow(by_tokens, num_tokens) {
            return Err(AicError::PerfDatabase(format!(
                "MoE low-latency token underflow has only one measured point; cannot infer \
                 low-token latency from a singleton. num_tokens={num_tokens}, \
                 measured_token={only}, key={key:?}"
            )));
        }
        query_token_curve(by_tokens, num_tokens as f64, sol).map(Some)
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
            let kernel_source = row.str_optional(kernel_source_col)?.unwrap_or("").to_string();
            // Kernel-specific mxfp4 remaps (mirror Python `load_moe_data`):
            // the collector logs two distinct kernels under one `moe_dtype`;
            // route them to dedicated quant modes so DeepSeek-V4 modeling can
            // select the right one per GPU generation.
            //  - Blackwell trtllm-gen MXFP4xMXFP8:
            //    w4a8_mxfp4_mxfp8 + sglang_mxfp4_flashinfer_trtllm_moe
            //      -> w4a8_mxfp4_mxfp8_trtllm
            //  - Hopper flashinfer cutlass SM90 mixed-GEMM:
            //    w4a16_mxfp4 + sglang_flashinfer_cutlass_moe
            //      -> w4a16_mxfp4_cutlass
            let raw_quant = row.str_owned(moe_dtype_col)?;
            let quant = match (raw_quant.as_str(), kernel_source.as_str()) {
                ("w4a8_mxfp4_mxfp8", "sglang_mxfp4_flashinfer_trtllm_moe") => {
                    "w4a8_mxfp4_mxfp8_trtllm".to_string()
                }
                ("w4a16_mxfp4", "sglang_flashinfer_cutlass_moe") => {
                    "w4a16_mxfp4_cutlass".to_string()
                }
                _ => raw_quant,
            };
            let key = MoeKey {
                quant,
                distribution: row.str_owned(distribution_col)?,
                topk: row.u32(topk_col)?,
                num_experts: row.u32(num_experts_col)?,
                hidden_size: row.u32(hidden_size_col)?,
                inter_size: row.u32(inter_size_col)?,
                moe_tp_size: row.u32(moe_tp_size_col)?,
                moe_ep_size: row.u32(moe_ep_size_col)?,
            };
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
            .join("src/aiconfigurator/systems/data/b200_sxm/vllm/0.19.0")
    }

    #[test]
    fn moe_table_loads_b200_vllm() {
        let table = MoeTable::new(b200_vllm_data_root());
        let _ = table.load().expect("moe_perf.parquet must load");
    }

    /// Linear token proxy — fine for key-selection tests where only the
    /// resolution path (not the extrapolated value) matters.
    fn proxy_sol(t: f64) -> f64 {
        t
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
            &proxy_sol,
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
            .join("src/aiconfigurator/systems/data/b200_sxm/trtllm/1.2.0rc5")
    }

    /// Cross-language parity with the Python v2 engine. Expected values from:
    ///
    /// ```text
    /// PYTHONPATH=src python3 -c "
    /// from aiconfigurator.sdk.perf_database import PerfDatabase
    /// from aiconfigurator.sdk import common
    /// db = PerfDatabase('b200_sxm','vllm','0.19.0',
    ///                   systems_root='src/aiconfigurator/systems', database_mode='SOL')
    /// for nt in [384, 4096, 7]:
    ///     r = db.query_moe(num_tokens=nt, hidden_size=5120, inter_size=8192, topk=1,
    ///                      num_experts=16, moe_tp_size=1, moe_ep_size=1,
    ///                      quant_mode=common.MoEQuantMode.bfloat16,
    ///                      workload_distribution='power_law_1.01',
    ///                      database_mode=common.DatabaseMode.SILICON)
    ///     print(nt, repr(float(r)))"
    /// ```
    ///
    /// (`database_mode='SOL'` at construction disables the shared layer so
    /// Python loads exactly the same primary parquet the Rust table reads.)
    /// The collected token curve is {128, 256, 512, 1024}: nt=384 is an
    /// interior RAW lerp; nt=4096 / nt=7 are beyond-range util-holds where
    /// the MoE roofline SOL carries the growth (weight-load-dominated regime
    /// — a raw linear extrapolation would give ~2.0 ms at nt=4096, the
    /// roofline hold gives ~0.97 ms).
    // NOTE(shared-layer merge): oracle generated pre-shared-layer; regenerate if
    // this fails. `MoeTable::new` resolves to the single primary source with no
    // kernel_source filter, so no shared rows should join this curve.
    #[test]
    fn moe_query_matches_python_v2_engine() {
        use crate::common::system_spec::SystemSpec;

        let systems_yaml = PathBuf::from(REPO_ROOT_HINT)
            .join("../..")
            .join("src/aiconfigurator/systems/b200_sxm.yaml");
        let spec = SystemSpec::load(&systems_yaml).expect("b200_sxm.yaml must parse");

        // The MoE roofline exactly as the operator layer passes it
        // (`operators/moe.rs::sol_latency_ms`, gated => num_gemms = 3),
        // mirroring Python `MoE._query_moe_table.get_sol` incl. its integer
        // floor divisions.
        let quant = MoeQuantMode::Bfloat16;
        let (h, inter, topk, ne, ep, tp) = (5120u64, 8192u64, 1u64, 16u64, 1u64, 1u64);
        let sol = |t: f64| -> f64 {
            let num_gemms = 3u64;
            let total_tokens = t.round() as u64 * topk;
            let ops = total_tokens * h * inter * num_gemms * 2 / ep / tp;
            let mem_bytes_int = total_tokens / ep * h * 2
                + total_tokens / ep * inter * num_gemms / tp
                + h * inter * num_gemms / tp * std::cmp::min(ne / ep, total_tokens / ep);
            let mem_bytes = (mem_bytes_int as f64) * quant.mapping().memory;
            let tc_flops = spec.gpu.bfloat16_tc_flops.unwrap_or(1.0);
            let sol_math = (ops as f64) / (tc_flops * quant.mapping().compute) * 1000.0;
            let sol_mem = mem_bytes / spec.gpu.mem_bw * 1000.0;
            sol_math.max(sol_mem)
        };

        let table = MoeTable::new(b200_vllm_data_root());
        let cases: &[(u32, f64)] = &[
            (384, 0.707481598854065),
            (4096, 0.9657080651305716),
            (7, 0.2885776182085593),
        ];
        for &(nt, expected) in cases {
            let got = table
                .query(nt, 5120, 8192, 1, 16, 1, 1, quant, "power_law_1.01", &sol)
                .expect("query must succeed");
            assert!(
                ((got - expected) / expected).abs() < 1e-9,
                "nt={nt}: rust {got} vs python {expected}"
            );
        }
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
