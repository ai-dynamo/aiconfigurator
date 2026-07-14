// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! WideEP / DeepEP / TRT-LLM all-to-all perf tables for distributed MoE.
//!
//! Five CSVs span two shape families:
//!
//! 1. MoE-compute layout (same columns as `moe_perf.txt`):
//!    - `wideep_context_moe_perf.txt`
//!    - `wideep_generation_moe_perf.txt`
//!    - `wideep_moe_perf.txt` (TRT-LLM WideEP MoE compute; extra columns
//!      handled by tolerant deserialization)
//!    - `trtllm_alltoall_perf.txt` (TRT-LLM alltoall dispatch; subset of
//!      MoE columns)
//!
//! 2. DeepEP dispatch layout (separate notify/transmit latencies):
//!    - `wideep_deepep_normal_perf.txt`
//!    - `wideep_deepep_ll_perf.txt`
//!
//! All loaders are lazy. Token curves resolve on the shared perf_interp v2
//! engine (Grid, RAW lerp in range; beyond the collected range the
//! boundary util is held with `k_tail=1` and a LINEAR num_tokens proxy SOL
//! carries the growth):
//!
//! - DeepEP dispatch: exactly Python's wiring (`moe.py::_query_wideep_
//!   deepep_{normal,ll}_table` pass `sol_fn=lambda ..., t: float(t)` —
//!   dispatch bytes scale ~linearly with tokens, so a linear proxy is
//!   ratio-equivalent to any bandwidth roofline). DeepEP-normal keys the
//!   `dispatch_sms` axis and resolves a 2-axis `(sms, num_tokens)` Grid
//!   (off-grid sms snaps to the nearest collected value); the LL table is a
//!   1-axis token curve. Python interpolates the SUMMED per-row latency;
//!   here the engine runs per `DispatchPoint` field, which is numerically
//!   identical under a linear proxy (lerp, snap, and `q/b`-scaling all
//!   distribute over the sum; zero fields use the zero-boundary
//!   convention).
//! - TRT-LLM alltoall: Python's SOL (`_query_alltoall_table.get_sol`) is
//!   `const * num_tokens` per slice, so the proxy is ratio-equivalent
//!   there too.
//! - The three MoE-compute variants (context / generation / trtllm-wideep,
//!   currently caller-less in Rust): Python anchors beyond-range holds on
//!   the MoE roofline (via `_resolve_tokens` / `_query_compute_table`),
//!   which this table layer cannot compute. The linear proxy only affects
//!   beyond-range holds; in-range lerp matches Python exactly. A future
//!   operator-layer caller should thread the roofline through like
//!   `moe.rs::MoeTable::query` does.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use crate::common::enums::MoeQuantMode;
use crate::common::error::AicError;
use crate::common::system_spec::SystemSpec;
use crate::config::{PerfDbSources, PerfSource};
use super::perf_interp::{self, Node, OpInterpConfig};
use super::{kernel_source_ok, resolve_op_sources};
use super::moe::{query_token_curve, singleton_underflow};
use crate::perf_database::parquet_loader::PerfReader;

pub struct WideEpTable {
    data_root: PathBuf,
    /// Ordered, priority-sorted sources per distinct perf-file basename
    /// (shared-layer aware; see [`PerfSource`]). Single-primary, no-filter by
    /// default (`WideEpTable::new`).
    context_moe_sources: Vec<PerfSource>,
    generation_moe_sources: Vec<PerfSource>,
    moe_sources: Vec<PerfSource>,
    alltoall_sources: Vec<PerfSource>,
    deepep_normal_sources: Vec<PerfSource>,
    deepep_ll_sources: Vec<PerfSource>,
    context_moe: OnceLock<Result<MoeGrids, AicError>>,
    generation_moe: OnceLock<Result<MoeGrids, AicError>>,
    trtllm_wideep_moe: OnceLock<Result<MoeGrids, AicError>>,
    trtllm_alltoall: OnceLock<Result<AlltoallGrids, AicError>>,
    deepep_normal: OnceLock<Result<NormalDispatchGrids, AicError>>,
    deepep_ll: OnceLock<Result<DispatchGrids, AicError>>,
}

struct MoeGrids {
    by_keys: BTreeMap<MoeKey, BTreeMap<u32, f64>>,
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

struct DispatchGrids {
    by_keys: BTreeMap<DispatchKey, BTreeMap<u32, DispatchPoint>>,
}

/// DeepEP-normal grids carry the `dispatch_sms` level Python keys by
/// (`moe.py::load_wideep_deepep_normal_data`:
/// `[node_num][hidden_size][topk][num_experts][dispatch_sms][num_token]`).
struct NormalDispatchGrids {
    by_keys: BTreeMap<DispatchKey, BTreeMap<u32, BTreeMap<u32, DispatchPoint>>>,
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

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct DispatchKey {
    node_num: u32,
    hidden_size: u32,
    num_topk: u32,
    num_experts: u32,
}

/// Dispatch-style latency point. DeepEP normal CSV reports separate
/// notify/transmit times for dispatch and combine; DeepEP LL reports
/// average latencies and bandwidths.
#[derive(Clone, Copy, Debug, Default)]
pub struct DispatchPoint {
    pub dispatch_transmit_us: f64,
    pub dispatch_notify_us: f64,
    pub combine_transmit_us: f64,
    pub combine_notify_us: f64,
    /// LL-only: combine average latency (us).
    pub combine_avg_t_us: f64,
    /// LL-only: dispatch average latency (us).
    pub dispatch_avg_t_us: f64,
}

impl WideEpTable {
    /// Construct an empty table for the given data directory. No I/O. Each
    /// perf file is sourced solely from `data_root/<basename>` with no
    /// `kernel_source` filter (pre-shared-layer behaviour).
    pub fn new(data_root: PathBuf) -> Self {
        Self::with_sources(data_root, &PerfDbSources::default())
    }

    /// Construct with shared-layer (sibling/cross-version) sources resolved from
    /// `perf_db_sources` (Python-supplied). Each perf file falls back to its
    /// primary `data_root/<basename>` when absent from the map. No I/O.
    pub fn with_sources(data_root: PathBuf, perf_db_sources: &PerfDbSources) -> Self {
        let context_moe_sources =
            resolve_op_sources(perf_db_sources, "wideep_context_moe_perf.parquet", &data_root);
        let generation_moe_sources = resolve_op_sources(
            perf_db_sources,
            "wideep_generation_moe_perf.parquet",
            &data_root,
        );
        let moe_sources = resolve_op_sources(perf_db_sources, "wideep_moe_perf.parquet", &data_root);
        let alltoall_sources =
            resolve_op_sources(perf_db_sources, "trtllm_alltoall_perf.parquet", &data_root);
        let deepep_normal_sources = resolve_op_sources(
            perf_db_sources,
            "wideep_deepep_normal_perf.parquet",
            &data_root,
        );
        let deepep_ll_sources =
            resolve_op_sources(perf_db_sources, "wideep_deepep_ll_perf.parquet", &data_root);
        Self {
            data_root,
            context_moe_sources,
            generation_moe_sources,
            moe_sources,
            alltoall_sources,
            deepep_normal_sources,
            deepep_ll_sources,
            context_moe: OnceLock::new(),
            generation_moe: OnceLock::new(),
            trtllm_wideep_moe: OnceLock::new(),
            trtllm_alltoall: OnceLock::new(),
            deepep_normal: OnceLock::new(),
            deepep_ll: OnceLock::new(),
        }
    }

    /// WideEP context-phase MoE compute latency (ms).
    ///
    /// Python routes this table through `_resolve_tokens`, so the
    /// singleton-underflow guard applies (structured miss).
    #[allow(clippy::too_many_arguments)]
    pub fn query_context_moe(
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
        let grids = self.load_context_moe()?;
        query_moe(
            grids,
            num_tokens,
            hidden_size,
            inter_size,
            topk,
            num_experts,
            moe_tp_size,
            moe_ep_size,
            quant,
            workload_distribution,
            &self.data_root,
            true,
            sol,
        )
    }

    /// WideEP generation-phase MoE compute latency (ms). Singleton-underflow
    /// guard applies (see `query_context_moe`).
    #[allow(clippy::too_many_arguments)]
    pub fn query_generation_moe(
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
        let grids = self.load_generation_moe()?;
        query_moe(
            grids,
            num_tokens,
            hidden_size,
            inter_size,
            topk,
            num_experts,
            moe_tp_size,
            moe_ep_size,
            quant,
            workload_distribution,
            &self.data_root,
            true,
            sol,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn query_trtllm_wideep_moe(
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
        let grids = self.load_trtllm_wideep_moe()?;
        // Caller-less duplicate of `wideep_moe.rs::WideEpMoeTable` (the live
        // TRT-LLM WideEP compute table); kept on the linear token proxy.
        query_moe(
            grids,
            num_tokens,
            hidden_size,
            inter_size,
            topk,
            num_experts,
            moe_tp_size,
            moe_ep_size,
            quant,
            workload_distribution,
            &self.data_root,
            false,
            &|t| t,
        )
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

    /// DeepEP normal-mode dispatch point.
    ///
    /// Mirrors Python `_query_wideep_deepep_normal_table` (silicon path):
    /// `node_num == 1 && sms == 20` resolves the sm=20 slice as a plain 1-D
    /// token curve; anything else resolves a 2-axis `(sms, num_tokens)` Grid
    /// where an off-grid `sms` snaps to the nearest collected value (the SOL
    /// is the linear token proxy, constant in sms — no data supports an sms
    /// scaling story yet).
    pub fn query_deepep_normal(
        &self,
        node_num: u32,
        hidden_size: u32,
        num_tokens: u32,
        num_topk: u32,
        num_experts: u32,
        sms: u32,
    ) -> Result<DispatchPoint, AicError> {
        let grids = self.load_deepep_normal()?;
        let key = DispatchKey {
            node_num,
            hidden_size,
            num_topk,
            num_experts,
        };
        let by_sms = grids.by_keys.get(&key).ok_or_else(|| {
            AicError::PerfDatabase(format!(
                "dispatch data missing for {key:?} at {}",
                self.data_root.display()
            ))
        })?;
        if node_num == 1 && sms == 20 {
            // Python: only sm=20 is collected for node_num==1 today; the
            // sms bucket is indexed directly and the token curve rides the
            // 1-axis engine.
            let by_tokens = by_sms.get(&20).ok_or_else(|| {
                AicError::PerfDatabase(format!(
                    "dispatch data missing for {key:?} (sms=20) at {}",
                    self.data_root.display()
                ))
            })?;
            if let Some(point) = by_tokens.get(&num_tokens) {
                return Ok(*point);
            }
            return Ok(DispatchPoint {
                dispatch_transmit_us: dispatch_field(by_tokens, |p| p.dispatch_transmit_us, num_tokens)?,
                dispatch_notify_us: dispatch_field(by_tokens, |p| p.dispatch_notify_us, num_tokens)?,
                combine_transmit_us: dispatch_field(by_tokens, |p| p.combine_transmit_us, num_tokens)?,
                combine_notify_us: dispatch_field(by_tokens, |p| p.combine_notify_us, num_tokens)?,
                combine_avg_t_us: dispatch_field(by_tokens, |p| p.combine_avg_t_us, num_tokens)?,
                dispatch_avg_t_us: dispatch_field(by_tokens, |p| p.dispatch_avg_t_us, num_tokens)?,
            });
        }
        if let Some(point) = by_sms.get(&sms).and_then(|slice| slice.get(&num_tokens)) {
            return Ok(*point);
        }
        Ok(DispatchPoint {
            dispatch_transmit_us: dispatch_field_2d(by_sms, |p| p.dispatch_transmit_us, sms, num_tokens)?,
            dispatch_notify_us: dispatch_field_2d(by_sms, |p| p.dispatch_notify_us, sms, num_tokens)?,
            combine_transmit_us: dispatch_field_2d(by_sms, |p| p.combine_transmit_us, sms, num_tokens)?,
            combine_notify_us: dispatch_field_2d(by_sms, |p| p.combine_notify_us, sms, num_tokens)?,
            combine_avg_t_us: dispatch_field_2d(by_sms, |p| p.combine_avg_t_us, sms, num_tokens)?,
            dispatch_avg_t_us: dispatch_field_2d(by_sms, |p| p.dispatch_avg_t_us, sms, num_tokens)?,
        })
    }

    /// DeepEP low-latency dispatch point.
    pub fn query_deepep_ll(
        &self,
        node_num: u32,
        hidden_size: u32,
        num_tokens: u32,
        num_topk: u32,
        num_experts: u32,
    ) -> Result<DispatchPoint, AicError> {
        let grids = self.load_deepep_ll()?;
        dispatch_lookup(grids, node_num, hidden_size, num_tokens, num_topk, num_experts, &self.data_root)
    }

    fn load_context_moe(&self) -> Result<&MoeGrids, AicError> {
        let cell = self
            .context_moe
            .get_or_init(|| load_moe_parquet(&self.context_moe_sources));
        cell.as_ref().map_err(clone_err)
    }
    fn load_generation_moe(&self) -> Result<&MoeGrids, AicError> {
        let cell = self
            .generation_moe
            .get_or_init(|| load_moe_parquet(&self.generation_moe_sources));
        cell.as_ref().map_err(clone_err)
    }
    fn load_trtllm_wideep_moe(&self) -> Result<&MoeGrids, AicError> {
        let cell = self
            .trtllm_wideep_moe
            .get_or_init(|| load_moe_parquet(&self.moe_sources));
        cell.as_ref().map_err(clone_err)
    }
    fn load_trtllm_alltoall(&self) -> Result<&AlltoallGrids, AicError> {
        let cell = self
            .trtllm_alltoall
            .get_or_init(|| load_alltoall_parquet(&self.alltoall_sources));
        cell.as_ref().map_err(clone_err)
    }
    fn load_deepep_normal(&self) -> Result<&NormalDispatchGrids, AicError> {
        let cell = self
            .deepep_normal
            .get_or_init(|| load_deepep_normal_parquet(&self.deepep_normal_sources));
        cell.as_ref().map_err(clone_err)
    }
    fn load_deepep_ll(&self) -> Result<&DispatchGrids, AicError> {
        let cell = self
            .deepep_ll
            .get_or_init(|| load_deepep_ll_parquet(&self.deepep_ll_sources));
        cell.as_ref().map_err(clone_err)
    }
}

#[allow(clippy::too_many_arguments)]
fn query_moe(
    grids: &MoeGrids,
    num_tokens: u32,
    hidden_size: u32,
    inter_size: u32,
    topk: u32,
    num_experts: u32,
    moe_tp_size: u32,
    moe_ep_size: u32,
    quant: MoeQuantMode,
    workload_distribution: &str,
    data_root: &Path,
    guard_singleton_underflow: bool,
    sol: &dyn Fn(f64) -> f64,
) -> Result<f64, AicError> {
    let quant_name = quant.name();
    let requested_exists = grids
        .by_keys
        .keys()
        .any(|k| k.quant == quant_name && k.distribution == workload_distribution);
    let distribution = if requested_exists {
        workload_distribution.to_string()
    } else {
        "uniform".to_string()
    };
    let key = MoeKey {
        quant: quant_name.to_string(),
        distribution,
        topk,
        num_experts,
        hidden_size,
        inter_size,
        moe_tp_size,
        moe_ep_size,
    };
    let by_tokens = grids
        .by_keys
        .get(&key)
        .ok_or_else(|| AicError::PerfDatabase(format!("MoE data missing for {key:?} at {}", data_root.display())))?;
    if by_tokens.is_empty() {
        return Err(AicError::PerfDatabase("MoE table has no token points".to_string()));
    }
    // Python's `_resolve_tokens` singleton-underflow guard applies only to
    // the paths that route through it (sglang deepep context/generation MoE
    // compute); `_query_alltoall_table` / `_query_compute_table` query the
    // engine directly and let a singleton curve util-hold instead.
    if guard_singleton_underflow {
        if let Some(only) = singleton_underflow(by_tokens, num_tokens) {
            return Err(AicError::PerfDatabase(format!(
                "MoE silicon token underflow has only one measured point; cannot infer \
                 low-token latency from a singleton. num_tokens={num_tokens}, \
                 measured_token={only}, key={key:?}"
            )));
        }
    }
    query_token_curve(by_tokens, num_tokens as f64, sol)
}

/// Resolve one `DispatchPoint` field's token curve on the engine, with the
/// zero-boundary convention: a field whose boundary anchor is 0 (unused in
/// this table variant — e.g. the four normal-mode fields of an LL table)
/// contributes 0 beyond the range instead of a spurious "no positive-util
/// anchor" miss. Under the linear proxy this is numerically identical to
/// Python's util-hold on the SUMMED curve (`q/b * 0 = 0`).
fn dispatch_field(
    by_tokens: &BTreeMap<u32, DispatchPoint>,
    field: fn(&DispatchPoint) -> f64,
    num_tokens: u32,
) -> Result<f64, AicError> {
    let (&first, _) = by_tokens.iter().next().expect("caller checked non-empty");
    let (&last, _) = by_tokens.iter().next_back().expect("caller checked non-empty");
    if num_tokens < first || num_tokens > last {
        let anchor = if num_tokens < first { &by_tokens[&first] } else { &by_tokens[&last] };
        if field(anchor) == 0.0 {
            return Ok(0.0);
        }
    }
    let curve: BTreeMap<u32, f64> = by_tokens.iter().map(|(&t, p)| (t, field(p))).collect();
    query_token_curve(&curve, num_tokens as f64, &|t| t)
}

/// Resolve one `DispatchPoint` field on the DeepEP-normal 2-axis
/// `(sms, num_tokens)` grid via the shared engine (Python's
/// `OpInterpConfig(axes=("sms","num_tokens"), resolver=Grid(),
/// sol_fn=lambda _sm, t: float(t))`): in-range coordinates bracket+blend, an
/// off-grid `sms` snaps to the nearest collected value, and beyond-range
/// tokens util-hold on the linear proxy. The zero-boundary convention (see
/// [`dispatch_field`]) is applied via the hold anchor: when the query resolves
/// through `grid_hold` and the anchor's field is 0, the field contributes 0
/// instead of a spurious "no positive-util anchor" miss — numerically
/// identical to Python's util-hold on the SUMMED curve.
fn dispatch_field_2d(
    by_sms: &BTreeMap<u32, BTreeMap<u32, DispatchPoint>>,
    field: fn(&DispatchPoint) -> f64,
    sms: u32,
    num_tokens: u32,
) -> Result<f64, AicError> {
    if let Some(anchor) = normal_hold_anchor(by_sms, sms, num_tokens) {
        if field(anchor) == 0.0 {
            return Ok(0.0);
        }
    }
    let mut node = Node::branch();
    for (&sm, by_tokens) in by_sms {
        for (&t, point) in by_tokens {
            node.insert(&[sm, t], field(point));
        }
    }
    let sol = |c: &[f64]| c[1];
    let cfg = OpInterpConfig::grid(&["sms", "num_tokens"], &sol);
    perf_interp::query(&cfg, &node, &[sms as f64, num_tokens as f64])
}

/// Predict the engine's `grid_hold` anchor for a `(sms, num_tokens)` query on
/// a DeepEP-normal slice; `None` when the query resolves in-range (exact hit,
/// token lerp, sms lerp, or single-survivor). Mirrors `grid_hold`'s
/// snap-then-tail walk with `k_tail = 1`: sms snaps to the nearest collected
/// key (tie -> smaller), tokens anchor at the boundary key when beyond the
/// slice's range and at the nearest key otherwise.
fn normal_hold_anchor<'a>(
    by_sms: &'a BTreeMap<u32, BTreeMap<u32, DispatchPoint>>,
    sms: u32,
    num_tokens: u32,
) -> Option<&'a DispatchPoint> {
    if by_sms.is_empty() {
        return None;
    }
    let covered = |slice: &BTreeMap<u32, DispatchPoint>| -> bool {
        let (&first, _) = slice.iter().next().expect("loader never stores empty slices");
        let (&last, _) = slice.iter().next_back().expect("loader never stores empty slices");
        first <= num_tokens && num_tokens <= last
    };
    let nearest = |keys: &mut dyn Iterator<Item = u32>, c: u32| -> u32 {
        keys.min_by(|a, b| {
            let da = (f64::from(*a) - f64::from(c)).abs();
            let db = (f64::from(*b) - f64::from(c)).abs();
            da.total_cmp(&db)
        })
        .expect("non-empty")
    };
    let anchor_in = |slice: &'a BTreeMap<u32, DispatchPoint>| -> &'a DispatchPoint {
        let (&first, _) = slice.iter().next().expect("non-empty");
        let (&last, _) = slice.iter().next_back().expect("non-empty");
        if num_tokens > last {
            &slice[&last]
        } else if num_tokens < first {
            &slice[&first]
        } else {
            // tokens in range but an OUTER (sms) axis was snapped -> the
            // engine holds at the NEAREST token key, not a lerp.
            let key = nearest(&mut slice.keys().copied(), num_tokens);
            &slice[&key]
        }
    };
    if let Some(slice) = by_sms.get(&sms) {
        // Exact sms key collapses that level; tokens in range resolve
        // in-slice (exact/lerp) -> no hold.
        if covered(slice) {
            return None;
        }
        return Some(anchor_in(slice));
    }
    let (&min_sms, _) = by_sms.iter().next().expect("non-empty");
    let (&max_sms, _) = by_sms.iter().next_back().expect("non-empty");
    if sms > min_sms && sms < max_sms {
        // Bracketed sms: any covering bracket slice resolves in-range
        // (two-sided lerp or single-survivor) -> no hold.
        let (_, lo_slice) = by_sms.range(..sms).next_back().expect("bracketed");
        let (_, hi_slice) = by_sms.range(sms..).next().expect("bracketed");
        if covered(lo_slice) || covered(hi_slice) {
            return None;
        }
    }
    let snapped = nearest(&mut by_sms.keys().copied(), sms);
    Some(anchor_in(&by_sms[&snapped]))
}

fn dispatch_lookup(
    grids: &DispatchGrids,
    node_num: u32,
    hidden_size: u32,
    num_tokens: u32,
    num_topk: u32,
    num_experts: u32,
    data_root: &Path,
) -> Result<DispatchPoint, AicError> {
    let key = DispatchKey {
        node_num,
        hidden_size,
        num_topk,
        num_experts,
    };
    let by_tokens = grids.by_keys.get(&key).ok_or_else(|| {
        AicError::PerfDatabase(format!("dispatch data missing for {key:?} at {}", data_root.display()))
    })?;
    if let Some(point) = by_tokens.get(&num_tokens) {
        return Ok(*point);
    }
    if by_tokens.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "dispatch data has no token points for {key:?} at {}",
            data_root.display()
        )));
    }
    Ok(DispatchPoint {
        dispatch_transmit_us: dispatch_field(by_tokens, |p| p.dispatch_transmit_us, num_tokens)?,
        dispatch_notify_us: dispatch_field(by_tokens, |p| p.dispatch_notify_us, num_tokens)?,
        combine_transmit_us: dispatch_field(by_tokens, |p| p.combine_transmit_us, num_tokens)?,
        combine_notify_us: dispatch_field(by_tokens, |p| p.combine_notify_us, num_tokens)?,
        combine_avg_t_us: dispatch_field(by_tokens, |p| p.combine_avg_t_us, num_tokens)?,
        dispatch_avg_t_us: dispatch_field(by_tokens, |p| p.dispatch_avg_t_us, num_tokens)?,
    })
}

/// Load a MoE-shape table from an ordered, priority-sorted source list.
/// Sources are read in order and duplicates resolve LAST-wins, mirroring
/// Python `load_wideep_context_moe_data` / `load_wideep_generation_moe_data`
/// / `load_wideep_moe_compute_data`, which direct-assign per coordinate with
/// no `try/except KeyError` guard — both within a file and across the
/// concatenated shared-layer rows. Missing files are skipped; an error is
/// returned only when no source yields rows. Reused for the
/// context/generation/wideep-moe parquets (alltoall has its own loader).
fn load_moe_parquet(sources: &[PerfSource]) -> Result<MoeGrids, AicError> {
    let mut by_keys: BTreeMap<MoeKey, BTreeMap<u32, f64>> = BTreeMap::new();
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
        // `inter_size` / `moe_tp_size` are absent in `trtllm_alltoall_perf.parquet`;
        // shared with the wideep_*_moe parquets which carry them. Optional lookup
        // mirrors the prior `Option<u32>` deserialization plus `unwrap_or` default.
        let inter_size_col = reader.col_optional("inter_size");
        let topk_col = reader.col("topk")?;
        let num_experts_col = reader.col("num_experts")?;
        let moe_tp_size_col = reader.col_optional("moe_tp_size");
        let moe_ep_size_col = reader.col("moe_ep_size")?;
        let distribution_col = reader.col("distribution")?;
        let latency_col = reader.col("latency")?;
        let ks_col = reader.col_optional("kernel_source");
        for row in reader.rows()? {
            let row = row?;
            if !kernel_source_ok(source.kernel_sources(), ks_col, &row)? {
                continue;
            }
            let key = MoeKey {
                quant: row.str_owned(moe_dtype_col)?,
                distribution: row.str_owned(distribution_col)?,
                topk: row.u32(topk_col)?,
                num_experts: row.u32(num_experts_col)?,
                hidden_size: row.u32(hidden_size_col)?,
                inter_size: row.u32_optional(inter_size_col)?.unwrap_or(0),
                moe_tp_size: row.u32_optional(moe_tp_size_col)?.unwrap_or(1),
                moe_ep_size: row.u32(moe_ep_size_col)?,
            };
            // Last-wins parity with Python `load_wideep_*_moe_data`
            // (direct-assign, no skip-on-conflict guard).
            by_keys
                .entry(key)
                .or_default()
                .insert(row.u32(num_tokens_col)?, row.f64(latency_col)?);
        }
    }
    if !any_source || by_keys.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "no MoE-shape rows loaded from {} source(s) (first: {})",
            sources.len(),
            sources.first().map(|s| s.path().display().to_string()).unwrap_or_default()
        )));
    }
    Ok(MoeGrids { by_keys })
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
/// - duplicates resolve LAST-wins (Python direct-assigns per coordinate over
///   the concatenated shared-layer rows).
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
                .insert(row.u32(num_tokens_col)?, row.f64(latency_col)?);
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

/// Load the DeepEP-normal dispatch table from an ordered source list. Missing
/// files are skipped; an error is returned only when no source yields rows.
///
/// Keying and merge mirror Python `load_wideep_deepep_normal_data` exactly:
/// the coordinate INCLUDES `dispatch_sms` (a required CSV column — Python
/// does `int(row["dispatch_sms"])`), and duplicates resolve FIRST-wins
/// (`if num_token in ...: skip`) over the concatenated source rows
/// (`_read_filtered_rows` flattens sources in priority order, so an earlier
/// source also outranks later siblings). Plain per-row `or_insert` reproduces
/// both rules; there is no within-file last-wins here.
fn load_deepep_normal_parquet(sources: &[PerfSource]) -> Result<NormalDispatchGrids, AicError> {
    let mut by_keys: BTreeMap<DispatchKey, BTreeMap<u32, BTreeMap<u32, DispatchPoint>>> =
        BTreeMap::new();
    let mut any_source = false;
    for source in sources {
        let path = source.path();
        if !path.exists() {
            continue;
        }
        any_source = true;
        let reader = PerfReader::open(path)?;
        let node_num_col = reader.col("node_num")?;
        let hidden_size_col = reader.col("hidden_size")?;
        let num_token_col = reader.col("num_token")?;
        let num_topk_col = reader.col("num_topk")?;
        let num_experts_col = reader.col("num_experts")?;
        let dispatch_sms_col = reader.col("dispatch_sms")?;
        let dispatch_transmit_us_col = reader.col_optional("dispatch_transmit_us");
        let dispatch_notify_us_col = reader.col_optional("dispatch_notify_us");
        let combine_transmit_us_col = reader.col_optional("combine_transmit_us");
        let combine_notify_us_col = reader.col_optional("combine_notify_us");
        let ks_col = reader.col_optional("kernel_source");
        for row in reader.rows()? {
            let row = row?;
            if !kernel_source_ok(source.kernel_sources(), ks_col, &row)? {
                continue;
            }
            let key = DispatchKey {
                node_num: row.u32(node_num_col)?,
                hidden_size: row.u32(hidden_size_col)?,
                num_topk: row.u32(num_topk_col)?,
                num_experts: row.u32(num_experts_col)?,
            };
            // First-wins on the FULL coordinate incl. dispatch_sms (Python
            // skip-on-conflict).
            by_keys
                .entry(key)
                .or_default()
                .entry(row.u32(dispatch_sms_col)?)
                .or_default()
                .entry(row.u32(num_token_col)?)
                .or_insert(DispatchPoint {
                    dispatch_transmit_us: row.f64_optional(dispatch_transmit_us_col)?.unwrap_or(0.0),
                    dispatch_notify_us: row.f64_optional(dispatch_notify_us_col)?.unwrap_or(0.0),
                    combine_transmit_us: row.f64_optional(combine_transmit_us_col)?.unwrap_or(0.0),
                    combine_notify_us: row.f64_optional(combine_notify_us_col)?.unwrap_or(0.0),
                    combine_avg_t_us: 0.0,
                    dispatch_avg_t_us: 0.0,
                });
        }
    }
    if !any_source || by_keys.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "no DeepEP-normal rows loaded from {} source(s) (first: {})",
            sources.len(),
            sources.first().map(|s| s.path().display().to_string()).unwrap_or_default()
        )));
    }
    Ok(NormalDispatchGrids { by_keys })
}

/// Load the DeepEP low-latency dispatch table from an ordered source list.
/// Same first-wins skip-on-conflict merge as [`load_deepep_normal_parquet`]
/// (Python `load_wideep_deepep_ll_data`: `if num_token in ...: skip`, over
/// `_read_filtered_rows`' priority-ordered concatenation) and the same
/// missing-file-skip semantics. The LL coordinate has no sms level.
fn load_deepep_ll_parquet(sources: &[PerfSource]) -> Result<DispatchGrids, AicError> {
    let mut by_keys: BTreeMap<DispatchKey, BTreeMap<u32, DispatchPoint>> = BTreeMap::new();
    let mut any_source = false;
    for source in sources {
        let path = source.path();
        if !path.exists() {
            continue;
        }
        any_source = true;
        let reader = PerfReader::open(path)?;
        let node_num_col = reader.col("node_num")?;
        let hidden_size_col = reader.col("hidden_size")?;
        let num_token_col = reader.col("num_token")?;
        let num_topk_col = reader.col("num_topk")?;
        let num_experts_col = reader.col("num_experts")?;
        let combine_avg_t_us_col = reader.col_optional("combine_avg_t_us");
        let dispatch_avg_t_us_col = reader.col_optional("dispatch_avg_t_us");
        let ks_col = reader.col_optional("kernel_source");
        for row in reader.rows()? {
            let row = row?;
            if !kernel_source_ok(source.kernel_sources(), ks_col, &row)? {
                continue;
            }
            let key = DispatchKey {
                node_num: row.u32(node_num_col)?,
                hidden_size: row.u32(hidden_size_col)?,
                num_topk: row.u32(num_topk_col)?,
                num_experts: row.u32(num_experts_col)?,
            };
            // First-wins on the full coordinate (Python skip-on-conflict).
            by_keys.entry(key).or_default().entry(row.u32(num_token_col)?).or_insert(
                DispatchPoint {
                    dispatch_transmit_us: 0.0,
                    dispatch_notify_us: 0.0,
                    combine_transmit_us: 0.0,
                    combine_notify_us: 0.0,
                    combine_avg_t_us: row.f64_optional(combine_avg_t_us_col)?.unwrap_or(0.0),
                    dispatch_avg_t_us: row.f64_optional(dispatch_avg_t_us_col)?.unwrap_or(0.0),
                },
            );
        }
    }
    if !any_source || by_keys.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "no DeepEP-LL rows loaded from {} source(s) (first: {})",
            sources.len(),
            sources.first().map(|s| s.path().display().to_string()).unwrap_or_default()
        )));
    }
    Ok(DispatchGrids { by_keys })
}

fn clone_err(err: &AicError) -> AicError {
    AicError::PerfDatabase(err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Cross-language parity with the Python v2 engine. Expected values from:
    ///
    /// ```text
    /// PYTHONPATH=src python3 -c "
    /// from aiconfigurator.sdk.perf_database import PerfDatabase
    /// from aiconfigurator.sdk import common
    /// db = PerfDatabase('h100_sxm','sglang','0.5.6.post2',
    ///                   systems_root='src/aiconfigurator/systems', database_mode='SOL')
    /// for nt in [20, 4096]:
    ///     r = db.query_wideep_deepep_ll(node_num=1, num_tokens=nt, num_experts=256,
    ///                                   topk=8, hidden_size=7168,
    ///                                   database_mode=common.DatabaseMode.SILICON)
    ///     print(nt, repr(float(r)))"
    /// ```
    ///
    /// Python interpolates the SUMMED (dispatch + combine) curve and returns
    /// ms; the Rust table resolves each `DispatchPoint` field separately in
    /// us. Under the shared linear-proxy SOL the two are numerically
    /// identical (lerp and boundary util-hold both distribute over the sum),
    /// so `(dispatch_avg + combine_avg) / 1000` must match. nt=20 is an
    /// interior lerp (collected 16 / 24); nt=4096 is a beyond-max util-hold
    /// (collected max 2048).
    // NOTE(shared-layer merge): oracle generated pre-shared-layer; regenerate if
    // this fails. `WideEpTable::new` resolves to single primary sources with no
    // kernel_source filter, so no shared rows should join this curve.
    #[test]
    fn deepep_ll_matches_python_v2_engine() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../src/aiconfigurator/systems/data/h100_sxm/sglang/0.5.6.post2");
        let table = WideEpTable::new(root);
        let cases: &[(u32, f64)] = &[(20, 0.03853445), (4096, 2.980605)];
        for &(nt, expected_ms) in cases {
            let point = table
                .query_deepep_ll(1, 7168, nt, 8, 256)
                .expect("query must succeed");
            let got_ms = (point.dispatch_avg_t_us + point.combine_avg_t_us) / 1000.0;
            assert!(
                ((got_ms - expected_ms) / expected_ms).abs() < 1e-9,
                "nt={nt}: rust {got_ms} vs python {expected_ms}"
            );
            // The four normal-mode fields are absent from LL tables; the
            // zero-boundary convention must keep them at 0, not error.
            assert_eq!(point.dispatch_transmit_us, 0.0);
            assert_eq!(point.combine_notify_us, 0.0);
        }
    }

    /// Write one synthetic DeepEP-normal parquet with the collector's column
    /// set. Row tuple: `(node_num, dispatch_sms, num_token,
    /// dispatch_transmit_us)`; the other latency fields are fixed so the
    /// zero-boundary convention is exercised (`combine_notify_us = 0`).
    /// Shape coordinate fixed at (hidden=7168, topk=8, experts=256).
    fn write_deepep_normal_parquet(path: &std::path::Path, rows: &[(i64, i64, i64, f64)]) {
        use parquet::data_type::{DoubleType, Int64Type};
        use parquet::file::properties::WriterProperties;
        use parquet::file::writer::SerializedFileWriter;
        use parquet::schema::parser::parse_message_type;
        use std::sync::Arc;

        let schema = "message schema {
            REQUIRED INT64 node_num;
            REQUIRED INT64 hidden_size;
            REQUIRED INT64 num_token;
            REQUIRED INT64 num_topk;
            REQUIRED INT64 num_experts;
            REQUIRED INT64 dispatch_sms;
            REQUIRED DOUBLE dispatch_transmit_us;
            REQUIRED DOUBLE dispatch_notify_us;
            REQUIRED DOUBLE combine_transmit_us;
            REQUIRED DOUBLE combine_notify_us;
        }";
        let schema = Arc::new(parse_message_type(schema).expect("schema must parse"));
        let file = std::fs::File::create(path).expect("create parquet");
        let mut writer =
            SerializedFileWriter::new(file, schema, Arc::new(WriterProperties::builder().build()))
                .expect("writer");
        let mut rg = writer.next_row_group().expect("row group");
        let int_cols: [Vec<i64>; 6] = [
            rows.iter().map(|r| r.0).collect(),                // node_num
            rows.iter().map(|_| 7168).collect(),               // hidden_size
            rows.iter().map(|r| r.2).collect(),                // num_token
            rows.iter().map(|_| 8).collect(),                  // num_topk
            rows.iter().map(|_| 256).collect(),                // num_experts
            rows.iter().map(|r| r.1).collect(),                // dispatch_sms
        ];
        for values in &int_cols {
            let mut col = rg.next_column().expect("next col").expect("int col");
            col.typed::<Int64Type>().write_batch(values, None, None).expect("write ints");
            col.close().expect("close col");
        }
        let f64_cols: [Vec<f64>; 4] = [
            rows.iter().map(|r| r.3).collect(),                // dispatch_transmit_us
            rows.iter().map(|_| 1.0).collect(),                // dispatch_notify_us
            rows.iter().map(|_| 2.0).collect(),                // combine_transmit_us
            rows.iter().map(|_| 0.0).collect(),                // combine_notify_us (zero-boundary)
        ];
        for values in &f64_cols {
            let mut col = rg.next_column().expect("next col").expect("f64 col");
            col.typed::<DoubleType>().write_batch(values, None, None).expect("write f64");
            col.close().expect("close col");
        }
        rg.close().expect("close row group");
        writer.close().expect("close writer");
    }

    /// Item 3: the DeepEP-normal table is keyed by `dispatch_sms` (Python
    /// `[node][hidden][topk][experts][dispatch_sms][num_token]`) and queried
    /// with nearest-snap sms semantics. The old Rust `DispatchKey` lacked the
    /// sms level and last-wins-collapsed the rows, answering the LAST sms row
    /// for every query. Python oracle (verified against
    /// `perf_interp.query(OpInterpConfig(axes=("sms","num_tokens"),
    /// resolver=Grid(), sol_fn=lambda _sm, t: float(t)), data, sms, 64)` on
    /// the same synthetic dict): sms=16 -> 100 (own row), sms=32 -> 500 (own
    /// row), sms=24 -> 300 (sms lerp), sms=12 -> 100 (snap below), sms=40 ->
    /// 500 (snap above).
    #[test]
    fn deepep_normal_keys_by_dispatch_sms_and_snaps_off_grid() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        write_deepep_normal_parquet(
            &tmp.path().join("wideep_deepep_normal_perf.parquet"),
            &[(2, 16, 64, 100.0), (2, 32, 64, 500.0)],
        );
        let table = WideEpTable::new(tmp.path().to_path_buf());
        let q = |sms: u32| {
            table
                .query_deepep_normal(2, 7168, 64, 8, 256, sms)
                .expect("query must succeed")
        };
        // Each collected sms answers from its OWN row (the old collapsed key
        // returned 500 for both).
        assert_eq!(q(16).dispatch_transmit_us, 100.0);
        assert_eq!(q(32).dispatch_transmit_us, 500.0);
        // Off-grid sms: interior lerp / nearest snap (Python Grid semantics).
        assert_eq!(q(24).dispatch_transmit_us, 300.0);
        assert_eq!(q(12).dispatch_transmit_us, 100.0);
        assert_eq!(q(40).dispatch_transmit_us, 500.0);
        // The LL-only fields stay 0 through every path (zero-boundary
        // convention on the snap/hold paths, plain lerp of zeros in range).
        assert_eq!(q(24).combine_avg_t_us, 0.0);
        assert_eq!(q(12).combine_avg_t_us, 0.0);
        // combine_notify_us is measured-zero in this fixture: snap/hold paths
        // must yield 0, not a "no positive-util anchor" miss.
        assert_eq!(q(12).combine_notify_us, 0.0);
    }

    /// Item 3 (token axis under the sms grid): beyond-range tokens util-hold
    /// on the linear proxy inside the resolved sms slice; an off-grid sms
    /// with in-range tokens holds at the NEAREST token key (not a lerp),
    /// mirroring `_grid_hold`'s outer-axis-snapped tail. Python oracle from
    /// the same `perf_interp` config on `{16: {64: 100, 128: 200}}`:
    /// (sms=16, nt=256) -> 400 (= 200 * 256/128); (sms=12, nt=96) -> 150
    /// (= 100 * 96/64; tie |96-64| == |96-128| keeps the smaller key).
    #[test]
    fn deepep_normal_token_hold_and_outer_snap_match_python() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        write_deepep_normal_parquet(
            &tmp.path().join("wideep_deepep_normal_perf.parquet"),
            &[(2, 16, 64, 100.0), (2, 16, 128, 200.0)],
        );
        let table = WideEpTable::new(tmp.path().to_path_buf());
        let hold = table
            .query_deepep_normal(2, 7168, 256, 8, 256, 16)
            .expect("query must succeed");
        assert!((hold.dispatch_transmit_us - 400.0).abs() < 1e-9);
        let snapped = table
            .query_deepep_normal(2, 7168, 96, 8, 256, 12)
            .expect("query must succeed");
        assert!((snapped.dispatch_transmit_us - 150.0).abs() < 1e-9);
    }

    /// Item 3 (sm=20 fast path): `node_num == 1 && sms == 20` resolves the
    /// sm=20 slice as a plain 1-D token curve (Python's "only collect sm=20
    /// for now" branch) — interior tokens lerp, and rows at other sms values
    /// must not leak in.
    #[test]
    fn deepep_normal_node1_sms20_uses_dedicated_slice() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        write_deepep_normal_parquet(
            &tmp.path().join("wideep_deepep_normal_perf.parquet"),
            &[(1, 20, 64, 100.0), (1, 20, 128, 200.0), (1, 40, 64, 900.0)],
        );
        let table = WideEpTable::new(tmp.path().to_path_buf());
        let exact = table
            .query_deepep_normal(1, 7168, 64, 8, 256, 20)
            .expect("query must succeed");
        assert_eq!(exact.dispatch_transmit_us, 100.0);
        let lerp = table
            .query_deepep_normal(1, 7168, 96, 8, 256, 20)
            .expect("query must succeed");
        assert!((lerp.dispatch_transmit_us - 150.0).abs() < 1e-9);
    }

    /// Duplicate coordinates (same sms + token) resolve FIRST-wins, mirroring
    /// Python's skip-on-conflict (`if num_token in ...`).
    #[test]
    fn deepep_normal_duplicate_rows_first_wins() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        write_deepep_normal_parquet(
            &tmp.path().join("wideep_deepep_normal_perf.parquet"),
            &[(2, 16, 64, 100.0), (2, 16, 64, 999.0)],
        );
        let table = WideEpTable::new(tmp.path().to_path_buf());
        let point = table
            .query_deepep_normal(2, 7168, 64, 8, 256, 16)
            .expect("query must succeed");
        assert_eq!(point.dispatch_transmit_us, 100.0);
    }

    fn gb200_spec() -> SystemSpec {
        let yaml = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../src/aiconfigurator/systems/gb200.yaml");
        SystemSpec::load(&yaml).expect("gb200.yaml must parse")
    }

    fn gb200_trtllm_table() -> WideEpTable {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../src/aiconfigurator/systems/data/gb200/trtllm/1.3.0rc10");
        WideEpTable::new(root)
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
    fn wideep_loaders_smoke() {
        // None of the WideEP/DeepEP data exists on vLLM b200 (TRT-LLM/SGLang
        // territory). Loader must surface clean errors.
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../src/aiconfigurator/systems/data/b200_sxm/vllm/0.19.0");
        let table = WideEpTable::new(root);
        let err = table
            .query_context_moe(
                1024,
                4096,
                2048,
                2,
                128,
                1,
                8,
                MoeQuantMode::Bfloat16,
                "uniform",
                &|t| t,
            )
            .unwrap_err();
        match err {
            AicError::Io { .. } | AicError::PerfDatabase(_) => {}
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
