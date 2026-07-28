// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Communication operators: custom allreduce, NCCL collectives, P2P.
//!
//! Mirrors `aiconfigurator.sdk.operations.communication.{CustomAllReduce,
//! NCCL, P2P}`, including the database-mode dispatch of
//! `_query_custom_allreduce_table` / `_query_nccl_table`. This is where the
//! topology-aware scaling lives:
//!
//! - `CustomAllReduceOp`: caps `tp_size` to `num_gpus_per_node` before
//!   the table lookup, then scales by `(tp-1)/tp * (per_node)/(per_node-1)
//!   * intra_bw/p2p_bw` when the actual fan-out exceeds the node.
//! - `NcclOp`: caps `num_gpus` to the table's max recorded fan-out, then
//!   scales by `(num_gpus-1)/num_gpus * max/(max-1) * max_bw/req_bw`.
//! - `P2POp`: pure analytic formula — `(bytes / inter_node_bw +
//!   p2p_latency) * 1000`. No CSV, no mode dispatch (analytic in every
//!   mode, like Python's `_query_p2p_table`).

use serde::{Deserialize, Serialize};
use crate::common::enums::{CommQuantMode, DatabaseMode};
use crate::common::error::AicError;
use crate::common::system_spec::SystemSpec;
use crate::operators::base::{PerformanceResult, Source};
use crate::operators::util_empirical::{self, UtilGrid};
use crate::perf_database::PerfDatabase;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CustomAllReduceOp {
    pub name: String,
    pub scale_factor: f64,
    pub hidden_size: u32,
    pub tp_size: u32,
    pub quant: CommQuantMode,
    /// CP sequence-shard factor (Python's `_seq_split`, = `cp_size`): the
    /// per-rank payload is `ceil(num_tokens / seq_split)`. Defaults to 1.
    #[serde(default = "crate::operators::gemm::default_seq_split")]
    pub seq_split: u32,
}

impl CustomAllReduceOp {
    pub fn new(
        name: impl Into<String>,
        scale_factor: f64,
        hidden_size: u32,
        tp_size: u32,
    ) -> Self {
        Self {
            name: name.into(),
            scale_factor,
            hidden_size,
            tp_size,
            quant: CommQuantMode::Half,
            seq_split: 1,
        }
    }

    /// Query for `num_tokens` of activation. Python's
    /// `CustomAllReduce.query` computes `size = x * self._h` (element
    /// count, not bytes) and passes it directly to
    /// `query_custom_allreduce`. Mirror that here — the underlying table
    /// is keyed by element count, with dtype implicit in the quant mode.
    /// Node-fan-out capping, beyond-node bandwidth scaling and the GB200
    /// NVL72 -> NCCL reroute all live in the DB-level `_scaled` query
    /// (mirroring Python's `_query_custom_allreduce_table` funnel).
    pub fn query(
        &self,
        db: &PerfDatabase,
        num_tokens: u32,
    ) -> Result<PerformanceResult, AicError> {
        if self.tp_size <= 1 {
            // No-op short-circuit: tp_size=1 has no allreduce. Python tags
            // it "empirical" so EMPIRICAL/SOL breakdowns don't report a
            // spurious silicon leakage.
            return Ok(PerformanceResult::new(0.0, Source::Empirical));
        }
        let per_rank_tokens = num_tokens.div_ceil(self.seq_split.max(1)); // CP: busiest rank
        let message_size = (per_rank_tokens as f64) * (self.hidden_size as f64);
        let (latency, source) =
            query_custom_allreduce_table(db, self.quant, self.tp_size, message_size)?;
        Ok(PerformanceResult::new(latency, source)
            .clamp_non_negative()
            .scaled(self.scale_factor))
    }
}

// ---------------------------------------------------------------------------
// Database-mode dispatch, mirroring the Python `_query_*_table` classmethods
// (`operations/communication.py`): SILICON queries the table (+ topology
// scaling); HYBRID converts a typed silicon miss into the util-space
// empirical estimate; EMPIRICAL always estimates. The SOL diagnostic modes
// never reach the compiled engine (the routing gate delegates them to the
// Python step).
// ---------------------------------------------------------------------------

/// Custom-allreduce latency for `(quant, tp_size, size)` under the
/// database's query mode. `size` is an element count (see
/// [`CustomAllReduceOp::query`]).
fn query_custom_allreduce_table(
    db: &PerfDatabase,
    quant: CommQuantMode,
    tp_size: u32,
    size: f64,
) -> Result<(f64, Source), AicError> {
    // GB200 NVL72 reroute — inside get_silicon only (Python
    // `communication.py:188-190`): custom AR is only collected up to tp4
    // there, so (SILICON and HYBRID) queries reroute to the MODE-AWARE
    // `database.query_nccl` dispatch. Under HYBRID an NCCL silicon miss
    // therefore falls to the NCCL empirical (never the custom-AR
    // empirical), and a terminal NCCL EmpiricalNotImplemented propagates
    // (Python's outer catch only handles the missing-silicon-data class).
    // EMPIRICAL mode never reaches get_silicon, so it never reroutes.
    // Handled here rather than via the DB-internal reroute in
    // `query_custom_allreduce_scaled`, which is silicon-only; in SILICON
    // mode both routes evaluate the identical `query_nccl_scaled` call.
    if db.system_spec.node.num_gpus_per_node == 72
        && tp_size > 4
        && db.database_mode != DatabaseMode::Empirical
    {
        return query_nccl_table(db, quant, tp_size, "all_reduce", size);
    }
    // The silicon path is the DB-level `_scaled` query (fan-out capping +
    // beyond-range bandwidth correction inside), mirroring Python
    // `_query_custom_allreduce_table.get_silicon`.
    let silicon =
        |db: &PerfDatabase| db.communication.query_custom_allreduce_scaled(&db.system_spec, quant, tp_size, size);
    match db.database_mode {
        DatabaseMode::Empirical => Ok((
            custom_allreduce_empirical(db, quant, tp_size, size)?,
            Source::Empirical,
        )),
        DatabaseMode::Hybrid => match silicon(db) {
            Ok(latency) => Ok((latency, Source::Silicon)),
            Err(err) if err.is_missing_perf_data() => Ok((
                custom_allreduce_empirical(db, quant, tp_size, size)?,
                Source::Empirical,
            )),
            Err(err) => Err(err),
        },
        _ => Ok((silicon(db)?, Source::Silicon)),
    }
}

/// Ring-allreduce SOL in ms. Mirrors Python
/// `_query_custom_allreduce_table.get_sol`: assume ring allreduce, ignore
/// constant latency, assume bfloat16 (`size` elements x 2 bytes).
fn custom_allreduce_sol_ms(spec: &SystemSpec, tp_size: u32, size: f64) -> f64 {
    if tp_size == 1 {
        return 0.0;
    }
    let p2p_bw = p2p_bandwidth(spec, tp_size);
    let tp = tp_size as f64;
    2.0 * size * 2.0 / tp * (tp - 1.0) / p2p_bw * 1000.0
}

/// `SOL(query)/util` over the collected custom-allreduce size curve.
/// Mirrors Python `_query_custom_allreduce_table.get_empirical`: SOL uses
/// the real `tp_size`; the util grid is built from the effective
/// (node-capped) tp slice, so the SOL ratio carries any multi-node
/// bandwidth scaling. Rank-count overflow borrows the node-boundary util
/// slice regardless of the transfer policy — Python's documented
/// compatibility exception (`xshape` provenance, TODO #1260).
fn custom_allreduce_empirical(
    db: &PerfDatabase,
    quant: CommQuantMode,
    tp_size: u32,
    size: f64,
) -> Result<f64, AicError> {
    let spec = &db.system_spec;
    let sol_q = custom_allreduce_sol_ms(spec, tp_size, size);
    if tp_size <= 1 || sol_q <= 0.0 {
        // No communication for a single rank -> 0/SOL, not a data gap.
        return Ok(sol_q);
    }
    let eff = tp_size.min(spec.node.num_gpus_per_node);
    let sol = |c: &[f64]| custom_allreduce_sol_ms(spec, eff, c[0]);
    let key = format!("custom_allreduce:{}:{eff}", quant.name());
    let grid = db.util_grids.get_or_try_build(&key, || {
        match db.communication.custom_allreduce_points(quant, eff) {
            Ok(points) => Ok(Some(UtilGrid::new(util_empirical::build_samples(points, sol)))),
            // Typed coverage miss -> no grid (estimate() raises the
            // empirical miss); schema/load errors propagate.
            Err(err) if err.is_missing_perf_data() => Ok(None),
            Err(err) => Err(err),
        }
    })?;
    let query = [size];
    let (latency, _) = util_empirical::estimate(sol_q, &query, grid.as_deref(), 1.0)?;
    // Rank-count overflow borrowed the node-boundary tp slice -> "xshape";
    // otherwise own-slice "empirical" (Python communication.py:167-168).
    db.note_provenance(if eff != tp_size {
        util_empirical::ProvenanceTier::XShape
    } else {
        util_empirical::ProvenanceTier::Empirical
    });
    Ok(latency)
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NcclOp {
    pub name: String,
    pub scale_factor: f64,
    /// Elements moved per token (Python's `_num_elements_per_token`). This is a
    /// float, not an integer: the CP KV all-gather sizes it as
    /// `kvcache_bytes_per_token / comm_bytes`, which can be fractional.
    pub hidden_size: f64,
    pub num_gpus: u32,
    pub dtype: CommQuantMode,
    pub operation: String,
    /// CP sequence-shard factor (Python's `_seq_split`, = `cp_size`): the
    /// per-rank payload is `ceil(num_tokens / seq_split)`. Defaults to 1.
    /// Note the CP KV all-gather (`context_cp_all_gather`) itself keeps
    /// `seq_split=1` (it moves the full per-token KV), so this is per-op.
    #[serde(default = "crate::operators::gemm::default_seq_split")]
    pub seq_split: u32,
}

impl NcclOp {
    pub fn new(
        name: impl Into<String>,
        scale_factor: f64,
        hidden_size: f64,
        num_gpus: u32,
        operation: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            scale_factor,
            hidden_size,
            num_gpus,
            dtype: CommQuantMode::Half,
            operation: operation.into(),
            seq_split: 1,
        }
    }

    pub fn query(
        &self,
        db: &PerfDatabase,
        num_tokens: u32,
    ) -> Result<PerformanceResult, AicError> {
        if self.num_gpus <= 1 {
            // No communication for a single rank. Python has no op-level
            // short-circuit, but every mode branch returns 0 tagged
            // "empirical" for num_gpus <= 1 (silicon's `num_gpus == 1`
            // early return / empirical's `sol_q = 0` guard).
            return Ok(PerformanceResult::new(0.0, Source::Empirical));
        }
        let per_rank_tokens = num_tokens.div_ceil(self.seq_split.max(1)); // CP: busiest rank
        // Python: message_size = ceil(x/seq_split) * num_elements_per_token —
        // kept as a FLOAT (fractional element counts are real: the gemma4 CP
        // KV all-gather sizes per-token elements as kv_bytes / comm_bytes).
        let message_size = (per_rank_tokens as f64) * self.hidden_size;
        let (latency, source) =
            query_nccl_table(db, self.dtype, self.num_gpus, &self.operation, message_size)?;
        Ok(PerformanceResult::new(latency, source)
            .clamp_non_negative()
            .scaled(self.scale_factor))
    }
}

/// NCCL collective latency for `(dtype, num_gpus, operation,
/// message_size)` under the database's query mode.
fn query_nccl_table(
    db: &PerfDatabase,
    dtype: CommQuantMode,
    num_gpus: u32,
    operation: &str,
    message_size: f64,
) -> Result<(f64, Source), AicError> {
    match db.database_mode {
        DatabaseMode::Empirical => Ok((
            nccl_empirical(db, dtype, num_gpus, operation, message_size)?,
            Source::Empirical,
        )),
        DatabaseMode::Hybrid => match db
            .communication
            .query_nccl_scaled(&db.system_spec, dtype, operation, num_gpus, message_size)
        {
            Ok(latency) => Ok((latency, Source::Silicon)),
            Err(err) if err.is_missing_perf_data() => Ok((
                nccl_empirical(db, dtype, num_gpus, operation, message_size)?,
                Source::Empirical,
            )),
            Err(err) => Err(err),
        },
        _ => Ok((
            db.communication
                .query_nccl_scaled(&db.system_spec, dtype, operation, num_gpus, message_size)?,
            Source::Silicon,
        )),
    }
}

/// Per-collective SOL in ms. Mirrors Python `_query_nccl_table.get_sol`:
/// one-directional ring traffic for gather/scatter-style collectives, 2x
/// for all_reduce, and 0 for unknown collectives (which the empirical
/// path then returns unchanged, not as a data gap).
fn nccl_sol_ms(
    spec: &SystemSpec,
    dtype: CommQuantMode,
    num_gpus: u32,
    operation: &str,
    message_size: f64,
) -> f64 {
    let p2p_bw = p2p_bandwidth(spec, num_gpus);
    let n = num_gpus as f64;
    let mem = dtype.mapping().memory;
    match operation {
        "all_gather" | "alltoall" | "reduce_scatter" => {
            mem * message_size * (n - 1.0) / n / p2p_bw * 1000.0
        }
        "all_reduce" => 2.0 * mem * message_size * (n - 1.0) / n / p2p_bw * 1000.0,
        _ => 0.0,
    }
}

/// `SOL(query)/util` over the collected NCCL size curve for this
/// `(dtype, operation, num_gpus)` slice. Mirrors Python
/// `_query_nccl_table.get_empirical`: the grid is built from the available
/// (capped) num_gpus slice; SOL uses the real `num_gpus` so the SOL ratio
/// carries scaling beyond the largest collected fan-out (`xshape`
/// borrowing regardless of transfer policy, TODO #1260). A source with no
/// NCCL/OneCCL data loaded at all, or no `(dtype, operation)` bucket, is
/// the terminal empirical miss.
fn nccl_empirical(
    db: &PerfDatabase,
    dtype: CommQuantMode,
    num_gpus: u32,
    operation: &str,
    message_size: f64,
) -> Result<f64, AicError> {
    let spec = &db.system_spec;
    let sol_q = nccl_sol_ms(spec, dtype, num_gpus, operation, message_size);
    if num_gpus <= 1 || sol_q <= 0.0 {
        // No communication for a single rank -> 0, not a data gap.
        return Ok(sol_q);
    }
    let max_collected = match db.communication.nccl_empirical_max_num_gpus(dtype, operation) {
        Ok(max) => max,
        Err(err) if err.is_missing_perf_data() => {
            return Err(AicError::EmpiricalNotImplemented(format!(
                "No NCCL data for operation '{operation}' ({}, num_gpus={num_gpus}).",
                dtype.name()
            )));
        }
        Err(err) => return Err(err),
    };
    let eff = num_gpus.min(max_collected);
    let sol = |c: &[f64]| nccl_sol_ms(spec, dtype, eff, operation, c[0]);
    let key = format!("nccl:{}:{operation}:{eff}", dtype.name());
    let grid = db.util_grids.get_or_try_build(&key, || {
        match db.communication.nccl_empirical_points(dtype, operation, eff) {
            Ok(points) => Ok(Some(UtilGrid::new(util_empirical::build_samples(points, sol)))),
            // Typed coverage miss (e.g. an uncollected intermediate gpu
            // count) -> no grid; estimate() raises the empirical miss.
            Err(err) if err.is_missing_perf_data() => Ok(None),
            Err(err) => Err(err),
        }
    })?;
    let query = [message_size];
    let (latency, _) = util_empirical::estimate(sol_q, &query, grid.as_deref(), 1.0)?;
    // Fan-out beyond the largest collected num_gpus borrowed the boundary
    // slice -> "xshape"; otherwise own-slice "empirical" (Python
    // communication.py:439-440).
    db.note_provenance(if eff != num_gpus {
        util_empirical::ProvenanceTier::XShape
    } else {
        util_empirical::ProvenanceTier::Empirical
    });
    Ok(latency)
}

/// Pure analytic P2P latency — no CSV.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct P2POp {
    pub name: String,
    pub scale_factor: f64,
    pub pp_size: u32,
    pub hidden_size: u32,
    /// CP sequence-shard factor (Python's `_seq_split`, = `cp_size`): the
    /// per-rank payload is `ceil(x / seq_split)`. Defaults to 1.
    #[serde(default = "crate::operators::gemm::default_seq_split")]
    pub seq_split: u32,
}

impl P2POp {
    pub fn new(name: impl Into<String>, pp_size: u32, hidden_size: u32) -> Self {
        Self {
            name: name.into(),
            scale_factor: 1.0,
            pp_size,
            hidden_size,
            seq_split: 1,
        }
    }

    pub fn query(&self, db: &PerfDatabase, x: u32) -> Result<PerformanceResult, AicError> {
        if self.pp_size <= 1 {
            return Ok(PerformanceResult::zero());
        }
        let spec = &db.system_spec;
        let per_rank_tokens = x.div_ceil(self.seq_split.max(1)); // CP: busiest rank
        let bytes = (per_rank_tokens as f64) * (self.hidden_size as f64) * 2.0;
        let inter_bw = spec.node.inter_node_bw.max(1.0);
        let latency = (bytes / inter_bw + spec.node.p2p_latency) * 1000.0;
        Ok(PerformanceResult::new(latency, Source::Empirical)
            .clamp_non_negative()
            .scaled(self.scale_factor))
    }
}

fn p2p_bandwidth(spec: &SystemSpec, num_gpus: u32) -> f64 {
    spec.get_p2p_bandwidth(num_gpus)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perf_database::CommunicationTable;
    use std::path::PathBuf;

    const REPO_ROOT_HINT: &str = env!("CARGO_MANIFEST_DIR");

    fn b200_vllm_db() -> PerfDatabase {
        let systems_root = PathBuf::from(REPO_ROOT_HINT)
            .join("../..")
            .join("src/aiconfigurator_core/systems");
        PerfDatabase::load(&systems_root, "b200_sxm", "vllm", "0.19.0").expect("db must load")
    }

    /// Oracle values generated from the Python reference on the same data:
    /// `CustomAllReduce._query_custom_allreduce_table(db, half, tp, size,
    /// database_mode=EMPIRICAL)` on b200_sxm/vllm/0.19.0 (collected tp
    /// slices: {2, 4, 8}; sizes 128..512Mi elements). Regenerate if the
    /// shipped table or the util-empirical math changes.
    #[test]
    fn custom_allreduce_empirical_matches_python_oracles() {
        let mut db = b200_vllm_db();
        db.database_mode = DatabaseMode::Empirical;
        use util_empirical::ProvenanceTier;
        let cases = [
            // off-grid size on a collected tp slice
            (8u32, 300_000u64, 0.0210877421663319, ProvenanceTier::Empirical),
            // exact collected hit: util reconstruction returns the measured value
            (8, 1024, 0.007716159820556641, ProvenanceTier::Empirical),
            // rank overflow: tp=16 on an 8-GPU node borrows the tp=8 boundary
            // util; SOL(tp=16, inter-node bw) carries the multi-node scaling.
            // Python capture: {"xshape"} (communication.py:167).
            (16, 524_288, 0.1607595384120941, ProvenanceTier::XShape),
            (2, 777_777, 0.019098769751423886, ProvenanceTier::Empirical),
            (4, 65_536, 0.008213120102882384, ProvenanceTier::Empirical),
        ];
        for (tp, size, expected, tier) in cases {
            db.reset_provenance();
            let (latency, source) =
                query_custom_allreduce_table(&db, CommQuantMode::Half, tp, size as f64)
                    .expect("empirical query");
            assert!(
                (latency - expected).abs() < 1e-9,
                "(tp={tp}, size={size}): expected {expected}, got {latency}"
            );
            assert_eq!(source, Source::Empirical);
            assert_eq!(db.worst_provenance(), tier, "(tp={tp}, size={size}): wrong tier");
        }
    }

    /// An uncollected tp slice (tp=5: eff=5 < 8, no data) is the terminal
    /// EmpiricalNotImplemented miss in both EMPIRICAL and HYBRID modes,
    /// never a fabricated value (mirrors the Python contract).
    #[test]
    fn custom_allreduce_missing_tp_raises_empirical_not_implemented() {
        for mode in [DatabaseMode::Empirical, DatabaseMode::Hybrid] {
            let mut db = b200_vllm_db();
            db.database_mode = mode;
            let result = query_custom_allreduce_table(&db, CommQuantMode::Half, 5, 1024 as f64);
            assert!(
                matches!(result, Err(AicError::EmpiricalNotImplemented(_))),
                "{mode:?}: got {result:?}"
            );
        }
    }

    /// HYBRID prefers silicon whenever the table covers the slice. Oracles:
    /// `CustomAllReduce._query_custom_allreduce_table(..., database_mode=HYBRID)`
    /// (source reported as `silicon` by Python for both).
    #[test]
    fn custom_allreduce_hybrid_prefers_silicon() {
        let mut db = b200_vllm_db();
        db.database_mode = DatabaseMode::Hybrid;
        let cases = [
            (8u32, 300_000u64, 0.024691462737973777),
            // rank overflow still resolves on silicon (tp=8 slice + scale)
            (16, 524_288, 0.16075953841209412),
        ];
        for (tp, size, expected) in cases {
            let (latency, source) =
                query_custom_allreduce_table(&db, CommQuantMode::Half, tp, size as f64)
                    .expect("hybrid query");
            assert!(
                (latency - expected).abs() < 1e-9,
                "(tp={tp}, size={size}): expected {expected}, got {latency}"
            );
            assert_eq!(source, Source::Silicon);
        }
    }

    /// Oracle values generated from the Python reference:
    /// `NCCL._query_nccl_table(db, half, num_gpus, op, msg,
    /// database_mode=EMPIRICAL)` on b200_sxm/vllm/0.19.0 (nccl 2.27.3
    /// data; collected num_gpus {2, 4, 8}; sizes 256..256Mi).
    #[test]
    fn nccl_empirical_matches_python_oracles() {
        let mut db = b200_vllm_db();
        db.database_mode = DatabaseMode::Empirical;
        use util_empirical::ProvenanceTier;
        let cases = [
            // off-grid message sizes on collected fan-outs
            (8u32, "all_reduce", 300_000u64, 0.027067167868039803, ProvenanceTier::Empirical),
            (4, "all_gather", 10_000_000, 0.05871496201240777, ProvenanceTier::Empirical),
            // gpu-count overflow: 32 > max collected 8 borrows the boundary
            // util; SOL(32, inter-node bw) carries the scaling.
            // Python capture: {"xshape"} (communication.py:439).
            (32, "all_reduce", 1_048_576, 0.31676464285714284, ProvenanceTier::XShape),
            // below-min size: boundary util-hold
            (8, "reduce_scatter", 999, 0.015899137756552793, ProvenanceTier::Empirical),
            // exact collected hit
            (2, "alltoall", 4096, 0.009470000000000001, ProvenanceTier::Empirical),
        ];
        for (num_gpus, op, msg, expected, tier) in cases {
            db.reset_provenance();
            let (latency, source) = query_nccl_table(&db, CommQuantMode::Half, num_gpus, op, msg as f64)
                .expect("empirical query");
            assert!(
                (latency - expected).abs() < 1e-9,
                "({num_gpus}, {op}, {msg}): expected {expected}, got {latency}"
            );
            assert_eq!(source, Source::Empirical);
            assert_eq!(db.worst_provenance(), tier, "({num_gpus}, {op}, {msg}): wrong tier");
        }
    }

    /// An uncollected intermediate gpu count (num_gpus=3: eff=3, no slice)
    /// is the terminal EmpiricalNotImplemented miss in EMPIRICAL and
    /// HYBRID modes.
    #[test]
    fn nccl_missing_gpu_count_raises_empirical_not_implemented() {
        for mode in [DatabaseMode::Empirical, DatabaseMode::Hybrid] {
            let mut db = b200_vllm_db();
            db.database_mode = mode;
            let result = query_nccl_table(&db, CommQuantMode::Half, 3, "all_reduce", 1024 as f64);
            assert!(
                matches!(result, Err(AicError::EmpiricalNotImplemented(_))),
                "{mode:?}: got {result:?}"
            );
        }
    }

    /// Unknown collectives SOL to 0 and return it unchanged in EMPIRICAL
    /// mode (Python `get_empirical`'s `sol_q <= 0` early return — a no-op,
    /// not a data gap). Same for a single rank.
    #[test]
    fn nccl_empirical_zero_sol_returns_zero_not_a_gap() {
        let mut db = b200_vllm_db();
        db.database_mode = DatabaseMode::Empirical;
        let (latency, source) =
            query_nccl_table(&db, CommQuantMode::Half, 8, "broadcast", 1024 as f64).expect("no-op query");
        assert_eq!(latency, 0.0);
        assert_eq!(source, Source::Empirical);
        let (latency, _) =
            query_nccl_table(&db, CommQuantMode::Half, 1, "all_reduce", 4096 as f64).expect("single rank");
        assert_eq!(latency, 0.0);
        // tp=1 allreduce likewise returns 0 in EMPIRICAL mode.
        let (latency, _) = query_custom_allreduce_table(&db, CommQuantMode::Half, 1, 4096 as f64)
            .expect("single rank allreduce");
        assert_eq!(latency, 0.0);
    }

    /// GB200 NVL72 reroute (num_gpus_per_node == 72 && tp > 4): Python's
    /// get_silicon returns the MODE-AWARE `database.query_nccl(...)`
    /// (communication.py:188-190), so under HYBRID an NCCL miss falls to the
    /// NCCL empirical — never the custom-AR empirical — and EMPIRICAL mode
    /// never reroutes at all. No shipped spec has 72 GPUs per node, so the
    /// spec is patched like the Python oracle run:
    ///
    /// ```text
    /// db = perf_database.get_database_view("b200_sxm", "vllm", "0.19.0",
    ///     allow_missing_data=True, database_mode=..., shared_layer=False)
    /// db.system_spec = copy.deepcopy(db.system_spec)
    /// db.system_spec["node"]["num_gpus_per_node"] = 72
    /// CustomAllReduce._query_custom_allreduce_table(db, half, tp, size, mode)
    /// ```
    #[test]
    fn custom_allreduce_nvl72_reroute_is_mode_aware() {
        let nvl72_db = |mode: DatabaseMode| {
            let mut db = b200_vllm_db();
            db.tables_mut().system_spec.node.num_gpus_per_node = 72;
            db.database_mode = mode;
            db
        };

        // HYBRID with NCCL data present: the reroute answers from the NCCL
        // silicon table (tp=16 exercises the beyond-max fan-out scaling).
        let db = nvl72_db(DatabaseMode::Hybrid);
        for (tp, size, expected) in [
            (8u32, 300_000u64, 0.02807729736328125),
            (16, 524_288, 0.031017857142857142),
        ] {
            let (latency, source) =
                query_custom_allreduce_table(&db, CommQuantMode::Half, tp, size as f64)
                    .expect("hybrid reroute");
            assert!(
                (latency - expected).abs() < 1e-9,
                "(tp={tp}, size={size}): expected {expected}, got {latency}"
            );
            assert_eq!(source, Source::Silicon);
        }

        // SILICON mode is unchanged: the operator-level reroute evaluates the
        // identical `query_nccl_scaled` the DB-internal reroute did.
        let db = nvl72_db(DatabaseMode::Silicon);
        let (latency, source) =
            query_custom_allreduce_table(&db, CommQuantMode::Half, 8, 300_000.0)
                .expect("silicon reroute");
        let scaled = db
            .communication
            .query_custom_allreduce_scaled(&db.system_spec, CommQuantMode::Half, 8, 300_000.0)
            .expect("db-level reroute");
        assert!(
            (latency - scaled).abs() < 1e-12,
            "operator ({latency}) and DB-level ({scaled}) reroutes must agree"
        );
        assert_eq!(source, Source::Silicon);

        // EMPIRICAL never reaches get_silicon, hence never reroutes: the
        // value stays the custom-AR empirical estimate (same as the
        // unpatched-spec oracle at tp=8/300k).
        let db = nvl72_db(DatabaseMode::Empirical);
        let (latency, source) =
            query_custom_allreduce_table(&db, CommQuantMode::Half, 8, 300_000.0)
                .expect("empirical never reroutes");
        assert!(
            (latency - 0.0210877421663319).abs() < 1e-9,
            "expected the custom-AR empirical value, got {latency}"
        );
        assert_eq!(source, Source::Empirical);

        // HYBRID with NO NCCL data at all: the nested NCCL empirical raises
        // the terminal EmpiricalNotImplemented, which PROPAGATES (Python:
        // "No NCCL data to estimate all_reduce (half, num_gpus=8)." — not in
        // `_MISSING_SILICON_DATA_EXCEPTIONS`, so the outer catch never runs
        // the custom-AR empirical). The pre-fix DB-internal reroute wrongly
        // fell back to the custom-AR empirical here.
        let mut db = b200_vllm_db();
        {
            let tables = db.tables_mut();
            tables.system_spec.node.num_gpus_per_node = 72;
            tables.communication =
                CommunicationTable::new(tables.data_root.clone(), None, None);
        }
        db.database_mode = DatabaseMode::Hybrid;
        let result = query_custom_allreduce_table(&db, CommQuantMode::Half, 8, 300_000.0);
        assert!(
            matches!(result, Err(AicError::EmpiricalNotImplemented(_))),
            "expected the NCCL empirical miss to propagate, got {result:?}"
        );
    }

    /// HYBRID prefers silicon whenever the table covers the slice. Oracles:
    /// `NCCL._query_nccl_table(..., database_mode=HYBRID)`.
    #[test]
    fn nccl_hybrid_prefers_silicon() {
        let mut db = b200_vllm_db();
        db.database_mode = DatabaseMode::Hybrid;
        let cases = [
            (8u32, "all_reduce", 300_000u64, 0.02807729736328125),
            (32, "all_reduce", 1_048_576, 0.3167646428571429),
        ];
        for (num_gpus, op, msg, expected) in cases {
            let (latency, source) = query_nccl_table(&db, CommQuantMode::Half, num_gpus, op, msg as f64)
                .expect("hybrid query");
            assert!(
                (latency - expected).abs() < 1e-9,
                "({num_gpus}, {op}, {msg}): expected {expected}, got {latency}"
            );
            assert_eq!(source, Source::Silicon);
        }
    }
}
