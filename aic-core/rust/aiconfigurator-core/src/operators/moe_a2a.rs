// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unified large-EP MoE all-to-all comm operator.
//!
//! Rust port of Python `sdk/operations/moe_comm.py::MoEAllToAll` — the OP
//! layer over the `perf_database/moe_a2a.rs` table (`_query_a2a_table`'s
//! silicon body lives there; everything the op adds is here):
//!
//! - `attention_tp_size` divides the token key before the lookup
//!   (moe_comm.py:661) — plain floor division, legacy fidelity with
//!   `MoEDispatch`'s `num_tokens // self._scale_num_tokens`. There is NO
//!   `max(1, ...)` guard: 0 tokens is a reachable query point.
//! - comm ops see PER-RANK tokens: unlike [`super::moe_expert_compute::MoeExpertComputeOp`] this op
//!   NEVER multiplies by `attention_dp_size` (moe_comm.py:657, :473-474).
//! - backend/phase validation against the `MOE_A2A_BACKENDS` registry
//!   (`_validate_a2a_request`, moe_comm.py:419-424) — Python raises
//!   `ValueError`, so this is a hard config error, NOT a data miss.
//! - the mode gate: SOL/SOL_FULL/EMPIRICAL have no estimation tier for this
//!   family (moe_comm.py:610-614) and HYBRID's empirical leg is the same
//!   raise (:639-643). See [`MoeAllToAllOp::query`].
//!
//! `scale_factor` multiplies the resolved latency, exactly like every sibling
//! op (moe_comm.py:674-678).

use serde::{Deserialize, Serialize};

use crate::common::enums::DatabaseMode;
use crate::common::error::AicError;
use crate::operators::base::{MoeCommFallback, PerformanceResult, Source};
use crate::perf_database::PerfDatabase;

/// The `MOE_A2A_BACKENDS` registry knowledge the operator layer needs:
/// backend name -> the comm phases that backend DECLARES.
///
/// SOURCE OF TRUTH is Python `moe_comm.py:92-119`; only the name and
/// `comm_phases` columns are mirrored here. The framework applicability,
/// `min_sm` and `max_topk` feasibility rules stay in Python — enumerating
/// feasible large-EP configurations is the model builder's job, not the
/// engine's.
///
/// The per-backend `comm_phases` ARE part of [`validate_a2a_request`]
/// (matches Python `_validate_a2a_request` after the CodeRabbit follow-up on
/// #1442): asking `deepep_ht` for `"prepare"` — a phase it does not declare —
/// is a config `ValueError`, failed where the intent is expressed rather than
/// surfacing later as a data miss. Pinned by
/// `undeclared_phase_is_a_config_error_not_a_data_miss`.
const MOE_A2A_BACKENDS: [(&str, &[&str]); 8] = [
    ("deepep_ht", &["dispatch", "combine"]),
    ("deepep_ll", &["dispatch", "combine"]),
    ("deepep_v2_context", &["dispatch", "combine"]),
    ("deepep_v2_generation", &["dispatch", "combine"]),
    ("trtllm_deepep_ht", &["dispatch", "combine"]),
    ("trtllm_deepep_ll", &["dispatch", "combine"]),
    ("nvlink_two_sided", &["prepare", "dispatch", "combine"]),
    ("nvlink_one_sided", &["dispatch", "combine"]),
];

/// Python `_A2A_PHASES` (moe_comm.py:416) — the phase names any backend may
/// be asked for.
const A2A_PHASES: [&str; 3] = ["prepare", "dispatch", "combine"];

/// Python `_validate_a2a_request` (moe_comm.py:419-424): an unknown backend
/// or an unknown phase is a `ValueError` — a programming error, deliberately
/// NOT a missing-data signal, so it must not trigger HYBRID/`FallbackOp`
/// handling. [`AicError::InvalidEngineConfig`] is the crate's ValueError
/// equivalent (same choice as `moe_dispatch.rs`'s `op_name` guard) and is the
/// one config-error variant `fpm::model::can_fallback_to_regression` refuses
/// to swallow.
fn validate_a2a_request(comm_backend: &str, phase: &str) -> Result<(), AicError> {
    if !MOE_A2A_BACKENDS
        .iter()
        .any(|(name, _)| *name == comm_backend)
    {
        let known: Vec<&str> = MOE_A2A_BACKENDS.iter().map(|(name, _)| *name).collect();
        return Err(AicError::InvalidEngineConfig(format!(
            "Invalid comm_backend '{comm_backend}'. Must be one of {known:?}"
        )));
    }
    if !A2A_PHASES.contains(&phase) {
        return Err(AicError::InvalidEngineConfig(format!(
            "Invalid phase '{phase}'. Must be one of {A2A_PHASES:?}"
        )));
    }
    let supported = MOE_A2A_BACKENDS
        .iter()
        .find(|(name, _)| *name == comm_backend)
        .map(|(_, phases)| *phases)
        .expect("backend membership checked above");
    if !supported.contains(&phase) {
        return Err(AicError::InvalidEngineConfig(format!(
            "comm_backend '{comm_backend}' does not implement phase '{phase}'; supported: {supported:?}"
        )));
    }
    Ok(())
}

fn default_comm_dtype() -> String {
    "default".to_string()
}

/// One comm phase of the unified large-EP all-to-all (Python
/// `MoEAllToAll`; one op instance per phase).
///
/// Field order is the wire format (bincode serializes struct fields
/// positionally) and matches the Python `_to_opspec` contract. `#[serde(default)]`
/// is applied exactly where the Python ctor has a default
/// (moe_comm.py:497-499), so an opspec that omits those keys queries the same
/// point Python would.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MoeAllToAllOp {
    pub name: String,
    pub scale_factor: f64,
    /// Comm phase: `"prepare"` / `"dispatch"` / `"combine"`.
    pub phase: String,
    /// Registry key, see [`MOE_A2A_BACKENDS`].
    pub comm_backend: String,
    /// Payload dtype of THIS phase (dispatch and combine may differ). The
    /// table resolves it exact-first, then the `fp8_block` -> `fp8`
    /// behavioral alias, then the sole collected dtype. Python default
    /// `"default"` (moe_comm.py:497) — the legacy DeepEP key.
    #[serde(default = "default_comm_dtype")]
    pub comm_dtype: String,
    pub hidden_size: u32,
    pub topk: u32,
    pub num_experts: u32,
    pub moe_ep_size: u32,
    pub node_num: u32,
    /// SM budget for the HT (DeepEP-normal) path; 0 for the LL/NVLink rows
    /// that carry no SM axis. Python default 0 (moe_comm.py:498).
    #[serde(default)]
    pub sms: u32,
    /// Attention-TP token-stream shard factor. Python default 1
    /// (moe_comm.py:499).
    #[serde(default = "crate::operators::gemm::default_seq_split")]
    pub attention_tp_size: u32,
}

impl MoeAllToAllOp {
    /// Query one all-to-all phase at the PER-RANK token count `num_tokens`
    /// (Python `query`'s `x`).
    ///
    /// Mirrors moe_comm.py:656-678 in source order:
    /// 1. `num_tokens = num_tokens // self._attention_tp_size` (:661) — plain
    ///    floor division, no `max(1, ...)` guard. The `.max(1)` below applies
    ///    to the DIVISOR only (panic guard for a malformed spec; Python would
    ///    `ZeroDivisionError`), never to the quotient — same convention as
    ///    `moe_dispatch.rs`'s `scale_num_tokens`.
    /// 2. `_validate_a2a_request` (:600, reached through `query_moe_a2a`).
    /// 3. the mode gate (:610-614) — see below.
    /// 4. the table lookup (:616-637) and `PerformanceResult(float(result) *
    ///    scale_factor, source=...)` (:674-678); the silicon path tags
    ///    `source="silicon"` (`_interp_pr`, perf_database.py:2510-2516).
    ///
    /// MODE SEMANTICS (silicon-only family):
    /// - SILICON — table query; a miss propagates as the typed data miss.
    /// - HYBRID — table query, and a MISSING-DATA miss converts to
    ///   [`AicError::EmpiricalNotImplemented`], because Python's HYBRID leg
    ///   calls `get_empirical()` which raises `EmpiricalNotImplementedError`
    ///   (:639-643). Non-data errors propagate unchanged.
    /// - SOL / SOL_FULL / EMPIRICAL — `EmpiricalNotImplementedError` BEFORE
    ///   any table access (:610-614). Rust's `DatabaseMode::Sol` /
    ///   `DatabaseMode::SolFull` / `DatabaseMode::Empirical` mirror those
    ///   three one-to-one, and the catch-all arm covers all of them.
    pub fn query(&self, db: &PerfDatabase, num_tokens: u32) -> Result<PerformanceResult, AicError> {
        // moe_comm.py:661 — attention TP shards the token stream ahead of the
        // A2A. Never scaled by attention_dp_size (:657).
        let tokens = num_tokens / self.attention_tp_size.max(1);
        // moe_comm.py:600 — validation precedes the mode gate (:610), so an
        // invalid backend/phase is a ValueError even under SOL/EMPIRICAL.
        validate_a2a_request(&self.comm_backend, &self.phase)?;
        match db.database_mode {
            DatabaseMode::Silicon | DatabaseMode::Hybrid => {}
            mode => {
                return Err(AicError::EmpiricalNotImplemented(format!(
                    "{mode:?} mode is not available for moe_a2a {}/{}: silicon data required \
                     (estimation tier is a planned follow-up).",
                    self.comm_backend, self.phase
                )))
            }
        }
        let exact_shape = db.moe_a2a.has_shape(
            &self.comm_backend,
            &self.phase,
            &self.comm_dtype,
            self.moe_ep_size,
            self.node_num,
            self.hidden_size,
            self.topk,
            self.num_experts,
        )?;
        let use_node1_fallback = db.backend == "sglang"
            && matches!(self.comm_backend.as_str(), "deepep_ht" | "deepep_ll")
            && self.node_num > 1
            && !exact_shape;
        let (lookup_ep_size, lookup_node_num, source) = if use_node1_fallback {
            // Legacy DeepEP tables have no EP axis. Their unified adapter
            // stores node-1 rows at EP=8; PR #1314 intentionally lets those
            // same-shape measurements represent an unmeasured multi-node
            // scale and marks the optimistic result as estimated.
            (
                crate::perf_database::moe_a2a::legacy_deepep_ep_size(1),
                1,
                Source::Estimated,
            )
        } else {
            (self.moe_ep_size, self.node_num, Source::Silicon)
        };
        let latency = self
            .silicon_latency_at(db, tokens, lookup_ep_size, lookup_node_num)
            .map_err(|err| {
                if db.database_mode == DatabaseMode::Hybrid && err.is_missing_perf_data() {
                    // moe_comm.py:639-643 — HYBRID's empirical fallback for this
                    // family is itself the typed not-implemented raise.
                    AicError::EmpiricalNotImplemented(format!(
                    "HYBRID empirical fallback is not available for moe_a2a {}/{}: silicon data \
                     required (estimation tier is a planned follow-up). Silicon miss: {err}",
                    self.comm_backend, self.phase
                ))
                } else {
                    err
                }
            })?;
        // Python: `PerformanceResult(float(result) * scale, source=...)` — no
        // clamp (nothing is subtracted on this path).
        let result = PerformanceResult::new(latency, source).scaled(self.scale_factor);
        if use_node1_fallback {
            let comm_backend = match self.comm_backend.as_str() {
                "deepep_ht" => "deepep_ht",
                "deepep_ll" => "deepep_ll",
                _ => unreachable!("node-1 fallback is restricted to SGLang DeepEP"),
            };
            Ok(result.with_moe_comm_fallback(MoeCommFallback {
                comm_backend,
                requested_ep_size: self.moe_ep_size,
                requested_node_num: self.node_num,
                measurement_ep_size: lookup_ep_size,
                measurement_node_num: lookup_node_num,
            }))
        } else {
            Ok(result)
        }
    }

    fn silicon_latency_at(
        &self,
        db: &PerfDatabase,
        tokens: u32,
        ep_size: u32,
        node_num: u32,
    ) -> Result<f64, AicError> {
        db.moe_a2a.query(
            &self.comm_backend,
            &self.phase,
            &self.comm_dtype,
            ep_size,
            node_num,
            self.hidden_size,
            self.topk,
            self.num_experts,
            tokens,
            self.sms,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::enums::TransferPolicy;
    use crate::operators::{FallbackOp, Op, OverlapOp, RuntimeContext};
    use crate::perf_database::MoeA2aTable;
    use parquet::data_type::{ByteArray, ByteArrayType, DoubleType, Int64Type};
    use parquet::file::properties::WriterProperties;
    use parquet::file::writer::{SerializedFileWriter, SerializedRowGroupWriter};
    use parquet::schema::parser::parse_message_type;
    use std::collections::BTreeMap;
    use std::fs::File;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    fn systems_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../src/aiconfigurator_core/systems")
    }

    fn write_column<T: parquet::data_type::DataType>(
        rg: &mut SerializedRowGroupWriter<'_, File>,
        values: &[T::T],
    ) {
        let mut col = rg.next_column().unwrap().unwrap();
        col.typed::<T>().write_batch(values, None, None).unwrap();
        col.close().unwrap();
    }

    /// One synthetic new-schema `moe_a2a_perf.parquet` row. Shape is pinned
    /// at (ep=16, node=2, hidden=7168, topk=8, experts=256); the axes under
    /// test are backend/phase/tokens.
    struct A2aRow {
        comm_backend: &'static str,
        phase: &'static str,
        ep_size: i64,
        node_num: i64,
        sms: i64,
        num_tokens: i64,
        /// MICROseconds — the new-schema collector unit the loader /1000s.
        latency_us: f64,
    }

    fn a2a_row(
        comm_backend: &'static str,
        phase: &'static str,
        num_tokens: i64,
        latency_us: f64,
    ) -> A2aRow {
        A2aRow {
            comm_backend,
            phase,
            ep_size: 16,
            node_num: 2,
            sms: 0,
            num_tokens,
            latency_us,
        }
    }

    fn a2a_row_at(
        comm_backend: &'static str,
        phase: &'static str,
        ep_size: i64,
        node_num: i64,
        sms: i64,
        num_tokens: i64,
        latency_us: f64,
    ) -> A2aRow {
        A2aRow {
            comm_backend,
            phase,
            ep_size,
            node_num,
            sms,
            num_tokens,
            latency_us,
        }
    }

    fn write_moe_a2a_parquet(path: &Path, rows: &[A2aRow]) {
        let schema = Arc::new(
            parse_message_type(
                "message a2a {
                    REQUIRED BYTE_ARRAY comm_backend (UTF8);
                    REQUIRED BYTE_ARRAY phase (UTF8);
                    REQUIRED BYTE_ARRAY comm_dtype (UTF8);
                    REQUIRED INT64 ep_size;
                    REQUIRED INT64 node_num;
                    REQUIRED INT64 hidden_size;
                    REQUIRED INT64 topk;
                    REQUIRED INT64 num_experts;
                    REQUIRED INT64 sms;
                    REQUIRED INT64 num_tokens;
                    REQUIRED DOUBLE latency;
                }",
            )
            .unwrap(),
        );
        let file = File::create(path).unwrap();
        let mut writer =
            SerializedFileWriter::new(file, schema, Arc::new(WriterProperties::builder().build()))
                .unwrap();
        let mut rg = writer.next_row_group().unwrap();
        let n = rows.len();
        write_column::<ByteArrayType>(
            &mut rg,
            &rows
                .iter()
                .map(|r| ByteArray::from(r.comm_backend))
                .collect::<Vec<_>>(),
        );
        write_column::<ByteArrayType>(
            &mut rg,
            &rows
                .iter()
                .map(|r| ByteArray::from(r.phase))
                .collect::<Vec<_>>(),
        );
        write_column::<ByteArrayType>(&mut rg, &vec![ByteArray::from("default"); n]);
        write_column::<Int64Type>(&mut rg, &rows.iter().map(|r| r.ep_size).collect::<Vec<_>>());
        write_column::<Int64Type>(
            &mut rg,
            &rows.iter().map(|r| r.node_num).collect::<Vec<_>>(),
        );
        write_column::<Int64Type>(&mut rg, &vec![7168_i64; n]);
        write_column::<Int64Type>(&mut rg, &vec![8_i64; n]);
        write_column::<Int64Type>(&mut rg, &vec![256_i64; n]);
        write_column::<Int64Type>(&mut rg, &rows.iter().map(|r| r.sms).collect::<Vec<_>>());
        write_column::<Int64Type>(
            &mut rg,
            &rows.iter().map(|r| r.num_tokens).collect::<Vec<_>>(),
        );
        write_column::<DoubleType>(
            &mut rg,
            &rows.iter().map(|r| r.latency_us).collect::<Vec<_>>(),
        );
        rg.close().unwrap();
        writer.close().unwrap();
    }

    /// A real database (for the system spec / data-root plumbing) with the
    /// moe_a2a table swapped for a synthetic one — the `moe_dispatch.rs`
    /// pattern. Latencies are chosen so every assertion below is an exact
    /// collected point.
    fn synthetic_db(mode: DatabaseMode) -> (tempfile::TempDir, PerfDatabase) {
        let tmp = tempfile::tempdir().expect("tmpdir");
        write_moe_a2a_parquet(
            &tmp.path().join("moe_a2a_perf.parquet"),
            &[
                // dispatch curve: token -> us
                a2a_row("deepep_ht", "dispatch", 31, 310.0),
                a2a_row("deepep_ht", "dispatch", 32, 320.0),
                a2a_row("deepep_ht", "dispatch", 63, 630.0),
                a2a_row("deepep_ht", "dispatch", 64, 640.0),
                // combine curve on the SAME shape, different values, so a
                // phase mix-up is observable.
                a2a_row("deepep_ht", "combine", 31, 3100.0),
                a2a_row("deepep_ht", "combine", 63, 6300.0),
                a2a_row("deepep_ht", "combine", 64, 6400.0),
                // an LL (generation) backend carrying the same token points.
                a2a_row("deepep_ll", "dispatch", 63, 63000.0),
                a2a_row("deepep_ll", "dispatch", 64, 64000.0),
                // Legacy node-1 DeepEP substitute (the unified adapter stores
                // these rows at EP=8 regardless of the requested target EP).
                a2a_row_at("deepep_ht", "dispatch", 8, 1, 20, 64, 1280.0),
                a2a_row_at("deepep_ht", "combine", 8, 1, 20, 64, 2560.0),
                a2a_row_at("deepep_ll", "dispatch", 8, 1, 0, 64, 12800.0),
            ],
        );
        let mut db = PerfDatabase::load(&systems_root(), "h200_sxm", "sglang", "0.5.6.post2")
            .expect("h200_sxm/sglang/0.5.6.post2 must load")
            .with_mode(mode, TransferPolicy::ALL);
        db.tables_mut().moe_a2a = MoeA2aTable::new(tmp.path().to_path_buf());
        (tmp, db)
    }

    fn op(phase: &str, comm_backend: &str, attention_tp_size: u32) -> MoeAllToAllOp {
        MoeAllToAllOp {
            name: format!("moe_{phase}"),
            scale_factor: 1.0,
            phase: phase.to_string(),
            comm_backend: comm_backend.to_string(),
            comm_dtype: "default".into(),
            hidden_size: 7168,
            topk: 8,
            num_experts: 256,
            moe_ep_size: 16,
            node_num: 2,
            sms: 0,
            attention_tp_size,
        }
    }

    // -----------------------------------------------------------------
    // Token-key arithmetic (moe_comm.py:661)
    // -----------------------------------------------------------------

    /// The context (`deepep_ht`) op divides by attention_tp_size; the
    /// generation (`deepep_ll`) op the model builder emits carries
    /// `attention_tp_size = 1` (blocks/moe.py:565 — deepep AND context), so
    /// the SAME `x` reaches the table undivided. Both ops are constructed
    /// here so the divide/no-divide split is one comparison.
    #[test]
    fn tp2_context_divides_and_tp1_generation_does_not() {
        let (_tmp, db) = synthetic_db(DatabaseMode::Silicon);
        // x = 128, tp = 2 -> 64 tokens -> 640us.
        let ctx = op("dispatch", "deepep_ht", 2)
            .query(&db, 128)
            .expect("context dispatch");
        assert!((ctx.latency_ms - 0.640).abs() < 1e-12, "got {ctx:?}");
        // x = 64, tp = 1 -> 64 tokens on the LL curve -> 64000us.
        let generation = op("dispatch", "deepep_ll", 1)
            .query(&db, 64)
            .expect("generation dispatch");
        assert!(
            (generation.latency_ms - 64.0).abs() < 1e-12,
            "got {generation:?}"
        );
        // A tp=2 generation op would halve the key: 32 tokens is UNCOLLECTED
        // on the LL curve (points 63/64), so the boundary-hold value differs
        // from the 64-token point — the divide is observable either way.
        let halved = op("dispatch", "deepep_ll", 2)
            .query(&db, 64)
            .expect("tp2 generation");
        assert!(
            (halved.latency_ms - 64.0).abs() > 1e-6,
            "tp=2 must move the token key, got {halved:?}"
        );
    }

    /// Plain floor division, no rounding and no `max(1, ...)`: x=63, tp=2
    /// keys 31, not 32 (the two adjacent collected points differ).
    #[test]
    fn floor_division_is_exact_at_odd_token_counts() {
        let (_tmp, db) = synthetic_db(DatabaseMode::Silicon);
        let got = op("dispatch", "deepep_ht", 2)
            .query(&db, 63)
            .expect("odd x");
        assert!(
            (got.latency_ms - 0.310).abs() < 1e-12,
            "x=63, tp=2 must key 31 (0.310 ms), got {got:?}"
        );
        // x=64 keys 32 -> the neighbouring point.
        let up = op("dispatch", "deepep_ht", 2)
            .query(&db, 64)
            .expect("even x");
        assert!((up.latency_ms - 0.320).abs() < 1e-12, "got {up:?}");
    }

    /// No `max(1, ...)` guard (moe_comm.py:660): `x < attention_tp_size` keys
    /// ZERO tokens, where the linear token proxy SOL is 0 and the engine has
    /// nothing to anchor a below-range hold on — a typed data miss.
    ///
    /// The pin is the CONTRAST with the guarded variant: keying 1 (what
    /// `max(1, ...)` would produce) resolves fine through the below-range
    /// hold, so a guard would silently turn this miss into a number.
    ///
    /// Python agrees on shipped data — `MoEAllToAll(..., attention_tp_size=4)`
    /// against h200_sxm/sglang/0.5.6.post2 `deepep_ht/dispatch` (ep=8,
    /// node=1, hidden=7168, topk=8, experts=256, sms=20) raises
    /// `PerfDataNotAvailableError` at `x=3` and `x=0`, and returns 0.03728 at
    /// `x=4` (token 1, a collected point there).
    #[test]
    fn zero_token_key_is_reachable_and_not_clamped_to_one() {
        let (_tmp, db) = synthetic_db(DatabaseMode::Silicon);
        let missed = op("dispatch", "deepep_ht", 4).query(&db, 3);
        assert!(
            matches!(missed, Err(AicError::PerfDatabase(_))),
            "0 tokens must stay a typed miss, got {missed:?}"
        );
        let guarded = op("dispatch", "deepep_ht", 4)
            .query(&db, 4)
            .expect("token 1 resolves through the below-range hold");
        assert!(
            guarded.latency_ms > 0.0,
            "a max(1, ...) guard would have returned {} for x=3",
            guarded.latency_ms
        );
    }

    /// `scale_factor` multiplies the resolved latency (moe_comm.py:675).
    #[test]
    fn scale_factor_multiplies_the_resolved_latency() {
        let (_tmp, db) = synthetic_db(DatabaseMode::Silicon);
        let mut scaled = op("combine", "deepep_ht", 1);
        scaled.scale_factor = 61.0;
        let got = scaled.query(&db, 64).expect("combine");
        assert!((got.latency_ms - 6.400 * 61.0).abs() < 1e-12, "got {got:?}");
        assert_eq!(got.source, Source::Silicon);
        assert!(got.moe_comm_fallbacks.is_empty());
    }

    #[test]
    fn sglang_deepep_node1_substitutes_for_missing_multi_node_scale_as_estimated() {
        let (_tmp, db) = synthetic_db(DatabaseMode::Silicon);
        let mut fallback = op("dispatch", "deepep_ht", 1);
        fallback.moe_ep_size = 128;
        fallback.node_num = 32;
        fallback.sms = 20;
        let got = fallback.query(&db, 64).expect("node-1 fallback");
        assert!((got.latency_ms - 1.280).abs() < 1e-12, "got {got:?}");
        assert_eq!(got.source, Source::Estimated);
        assert_eq!(
            got.moe_comm_fallbacks.iter().copied().collect::<Vec<_>>(),
            vec![MoeCommFallback {
                comm_backend: "deepep_ht",
                requested_ep_size: 128,
                requested_node_num: 32,
                measurement_ep_size: 8,
                measurement_node_num: 1,
            }]
        );
    }

    #[test]
    fn exact_requested_topology_wins_over_available_node1_donor() {
        let (_tmp, db) = synthetic_db(DatabaseMode::Silicon);
        // `op` requests EP16/node2, and `synthetic_db` also contains the
        // legacy EP8/node1 donor for this same shape. The exact 640us point
        // must win over the donor's 1280us point.
        let got = op("dispatch", "deepep_ht", 1)
            .query(&db, 64)
            .expect("exact requested topology");
        assert!((got.latency_ms - 0.640).abs() < 1e-12, "got {got:?}");
        assert_eq!(got.source, Source::Silicon);
        assert!(got.moe_comm_fallbacks.is_empty());
    }

    #[test]
    fn missing_node1_donor_remains_a_data_error() {
        let (_tmp, db) = synthetic_db(DatabaseMode::Silicon);
        let mut missing = op("dispatch", "deepep_ht", 1);
        missing.moe_ep_size = 128;
        missing.node_num = 32;
        missing.hidden_size = 9999;
        missing.sms = 20;

        let result = missing.query(&db, 64);

        assert!(
            matches!(result, Err(AicError::PerfDatabase(_))),
            "an unavailable donor must remain a typed data miss, got {result:?}"
        );
    }

    #[test]
    fn overlap_and_fallback_composites_preserve_every_executed_a2a_substitution() {
        let (_tmp, db) = synthetic_db(DatabaseMode::Silicon);
        let mut context = op("dispatch", "deepep_ht", 1);
        context.moe_ep_size = 128;
        context.node_num = 32;
        context.sms = 20;
        let mut generation = op("dispatch", "deepep_ll", 1);
        generation.moe_ep_size = 128;
        generation.node_num = 32;

        let overlap = Op::Overlap(OverlapOp::new(
            "a2a_overlap",
            vec![Op::MoeAllToAll(context.clone())],
            vec![Op::MoeAllToAll(generation.clone())],
        ));
        let overlap_result = overlap
            .query(
                &db,
                &RuntimeContext {
                    num_tokens: 64,
                    ..RuntimeContext::default()
                },
            )
            .expect("overlap query");
        assert_eq!(
            overlap_result
                .moe_comm_fallbacks
                .iter()
                .map(|fallback| fallback.comm_backend)
                .collect::<Vec<_>>(),
            vec!["deepep_ht", "deepep_ll"]
        );

        let mut missing_primary = context.clone();
        missing_primary.hidden_size = 9999;
        let fallback = Op::Fallback(FallbackOp::new(
            "a2a_fallback",
            Op::MoeAllToAll(missing_primary),
            vec![Op::MoeAllToAll(context), Op::MoeAllToAll(generation)],
        ));
        let fallback_result = fallback
            .query(
                &db,
                &RuntimeContext {
                    num_tokens: 64,
                    ..RuntimeContext::default()
                },
            )
            .expect("fallback query");
        assert_eq!(
            fallback_result
                .moe_comm_fallbacks
                .iter()
                .map(|record| record.comm_backend)
                .collect::<Vec<_>>(),
            vec!["deepep_ht", "deepep_ll"]
        );
    }

    // -----------------------------------------------------------------
    // Registry validation vs data miss (_validate_a2a_request)
    // -----------------------------------------------------------------

    /// An unknown backend or phase is Python's `ValueError`
    /// (moe_comm.py:421-424) -> [`AicError::InvalidEngineConfig`], which is
    /// NOT `is_missing_perf_data` and therefore never converts into a HYBRID
    /// estimate or a `FallbackOp` chain.
    #[test]
    fn unknown_backend_and_phase_are_config_errors_not_data_misses() {
        let (_tmp, db) = synthetic_db(DatabaseMode::Silicon);
        let bad_backend = op("dispatch", "deepep_xl", 1).query(&db, 64);
        assert!(
            matches!(&bad_backend, Err(AicError::InvalidEngineConfig(msg)) if msg.contains("comm_backend")),
            "got {bad_backend:?}"
        );
        assert!(
            !bad_backend.unwrap_err().is_missing_perf_data(),
            "a registry ValueError must not be a missing-data signal"
        );
        let bad_phase = op("scatter", "deepep_ht", 1).query(&db, 64);
        assert!(
            matches!(&bad_phase, Err(AicError::InvalidEngineConfig(msg)) if msg.contains("phase")),
            "got {bad_phase:?}"
        );
    }

    /// The distinction the ledgered T4 carry is about: a KNOWN backend on a
    /// shape the table does not carry is an ordinary typed data miss
    /// ([`AicError::PerfDatabase`]), not a config error.
    #[test]
    fn uncollected_shape_on_a_known_backend_is_a_data_miss() {
        let (_tmp, db) = synthetic_db(DatabaseMode::Silicon);
        let mut missing = op("dispatch", "deepep_ht", 1);
        missing.hidden_size = 9999;
        let result = missing.query(&db, 64);
        assert!(
            matches!(result, Err(AicError::PerfDatabase(_))),
            "got {result:?}"
        );
    }

    /// Python now validates the phase against the backend's own
    /// `comm_phases` (the CodeRabbit follow-up on #1442): `deepep_ht` does
    /// not declare `"prepare"` (see [`MOE_A2A_BACKENDS`]), so the request is
    /// a config error at the validation boundary, never a data miss.
    #[test]
    fn undeclared_phase_is_a_config_error_not_a_data_miss() {
        let (_tmp, db) = synthetic_db(DatabaseMode::Silicon);
        let result = op("prepare", "deepep_ht", 1).query(&db, 64);
        assert!(
            matches!(result, Err(AicError::InvalidEngineConfig(_))),
            "an undeclared phase must be a config error, got {result:?}"
        );
        // ... and the registry does declare it for the NVLink two-sided
        // backend, which is the only reason the phase name exists.
        let declared: &[&str] = MOE_A2A_BACKENDS
            .iter()
            .find(|(name, _)| *name == "nvlink_two_sided")
            .map(|(_, phases)| *phases)
            .expect("registry entry");
        assert!(declared.contains(&"prepare"));
    }

    /// Every phase any backend declares must be a member of the global
    /// validation tuple — the invariant that makes the two const tables
    /// consistent with `moe_comm.py`.
    #[test]
    fn declared_comm_phases_are_a_subset_of_the_validated_phases() {
        for (backend, phases) in MOE_A2A_BACKENDS {
            for phase in phases {
                assert!(
                    A2A_PHASES.contains(phase),
                    "{backend} declares an unvalidatable phase {phase:?}"
                );
            }
        }
    }

    /// Keep the Rust constructor's validation registry in exact identity
    /// parity with Python's public `MOE_A2A_BACKENDS` registry. A missing
    /// identity here turns a valid Python request into a Rust config error.
    #[test]
    fn registry_contains_every_python_backend_identity() {
        let names = MOE_A2A_BACKENDS
            .iter()
            .map(|(name, _)| *name)
            .collect::<Vec<_>>();
        assert_eq!(
            names,
            vec![
                "deepep_ht",
                "deepep_ll",
                "deepep_v2_context",
                "deepep_v2_generation",
                "trtllm_deepep_ht",
                "trtllm_deepep_ll",
                "nvlink_two_sided",
                "nvlink_one_sided",
            ]
        );
        for backend in names {
            validate_a2a_request(backend, "dispatch")
                .unwrap_or_else(|error| panic!("{backend} rejected: {error}"));
            validate_a2a_request(backend, "combine")
                .unwrap_or_else(|error| panic!("{backend} rejected: {error}"));
        }
    }

    // -----------------------------------------------------------------
    // Mode semantics (adjudication R5 / moe_comm.py:610-614, :639-643)
    // -----------------------------------------------------------------

    /// SOL / SOL_FULL / EMPIRICAL raise before any table access — Python
    /// `EmpiricalNotImplementedError` (moe_comm.py:611-614). The shape below
    /// IS collected, so a mode leak would return a number instead.
    #[test]
    fn estimation_tiers_raise_empirical_not_implemented() {
        for mode in [
            DatabaseMode::Sol,
            DatabaseMode::SolFull,
            DatabaseMode::Empirical,
        ] {
            let (_tmp, db) = synthetic_db(mode);
            let result = op("dispatch", "deepep_ht", 1).query(&db, 64);
            assert!(
                matches!(&result, Err(AicError::EmpiricalNotImplemented(msg))
                    if msg.contains("silicon data required")),
                "{mode:?}: got {result:?}"
            );
        }
    }

    /// HYBRID with data present stays on silicon; a HYBRID data MISS becomes
    /// the same typed not-implemented error, because Python's HYBRID
    /// `get_empirical()` leg is itself the raise (moe_comm.py:639-643).
    #[test]
    fn hybrid_uses_silicon_when_covered_and_raises_not_implemented_on_a_miss() {
        let (_tmp, db) = synthetic_db(DatabaseMode::Hybrid);
        let hit = op("dispatch", "deepep_ht", 1)
            .query(&db, 64)
            .expect("covered shape");
        assert!((hit.latency_ms - 0.640).abs() < 1e-12, "got {hit:?}");
        assert_eq!(hit.source, Source::Silicon);

        let mut missing = op("dispatch", "deepep_ht", 1);
        missing.hidden_size = 9999;
        let result = missing.query(&db, 64);
        assert!(
            matches!(&result, Err(AicError::EmpiricalNotImplemented(msg))
                if msg.contains("silicon data required")),
            "got {result:?}"
        );
    }

    /// The registry check precedes the mode gate (moe_comm.py:600 vs :610):
    /// an invalid backend under EMPIRICAL is still the `ValueError`.
    #[test]
    fn registry_validation_precedes_the_mode_gate() {
        let (_tmp, db) = synthetic_db(DatabaseMode::Empirical);
        let result = op("dispatch", "deepep_xl", 1).query(&db, 64);
        assert!(
            matches!(result, Err(AicError::InvalidEngineConfig(_))),
            "got {result:?}"
        );
    }

    // -----------------------------------------------------------------
    // Serde defaults (Python ctor defaults, moe_comm.py:497-499)
    // -----------------------------------------------------------------

    #[test]
    fn omitted_optional_fields_take_the_python_ctor_defaults() {
        let json = r#"{
            "name": "moe_dispatch",
            "scale_factor": 1.0,
            "phase": "dispatch",
            "comm_backend": "deepep_ht",
            "hidden_size": 7168,
            "topk": 8,
            "num_experts": 256,
            "moe_ep_size": 16,
            "node_num": 2
        }"#;
        let op: MoeAllToAllOp = serde_json::from_str(json).expect("defaults must fill in");
        assert_eq!(op.comm_dtype, "default");
        assert_eq!(op.sms, 0);
        assert_eq!(op.attention_tp_size, 1);
    }

    // -----------------------------------------------------------------
    // Python op-level oracle over the shipped data
    // -----------------------------------------------------------------

    const LFS_POINTER_PREFIX: &[u8] = b"version https://git-lfs";

    /// True when at least one of the family's parquet files is present and
    /// materialized (not a git-lfs pointer) under `data_root`.
    fn shipped_data_ready(data_root: &Path, basenames: &[&str]) -> bool {
        use crate::config::PerfDbSources;
        use crate::perf_database::resolve_op_sources;
        use std::io::Read;
        let mut any_file = false;
        for basename in basenames {
            for source in resolve_op_sources(&PerfDbSources::default(), basename, data_root) {
                let path = source.path();
                if !path.exists() {
                    continue;
                }
                let mut head = [0u8; LFS_POINTER_PREFIX.len()];
                let Ok(mut file) = File::open(path) else {
                    return false;
                };
                let Ok(read) = file.read(&mut head) else {
                    return false;
                };
                if read >= LFS_POINTER_PREFIX.len() && head == LFS_POINTER_PREFIX {
                    return false;
                }
                any_file = true;
            }
        }
        any_file
    }

    /// The `"moe_a2a"` slice of the shared op-level oracle fixture (the same
    /// file `moe_expert_compute.rs` reads for its `"moe_expert_compute"` slice — one generator run
    /// produces both).
    fn oracle_samples(op_kind: &str) -> Vec<serde_json::Value> {
        let oracle: serde_json::Value =
            serde_json::from_str(include_str!("testdata/op_oracle.json"))
                .expect("oracle fixture must parse");
        oracle["samples"]
            .as_array()
            .expect("samples array")
            .iter()
            .filter(|s| s["op"].as_str() == Some(op_kind))
            .cloned()
            .collect()
    }

    /// OP-level parity against Python `MoEAllToAll(...).query(db, x=...)` —
    /// not the raw table query: what is under test is the op layer's
    /// `x // attention_tp_size` key and the `* scale_factor` tail. The
    /// fixture is generated by `parity_tests/gen_op_oracle.py` (regeneration
    /// command in the JSON's `_regenerate` field) from the shipped
    /// h200_sxm/sglang (legacy DeepEP HT + LL) and gb200/trtllm (legacy
    /// NVLink alltoall, incl. the `fp4` low-precision combine keys) data over
    /// attention_tp_size in {1, 2}, both phases, scale factors, and token
    /// overflow.
    #[test]
    fn moe_a2a_op_matches_python_oracle() {
        let systems = systems_root();
        let samples = oracle_samples("moe_a2a");
        let mut dbs: BTreeMap<String, PerfDatabase> = BTreeMap::new();
        let mut max_rel = 0.0_f64;
        let mut checked = 0_usize;
        for sample in &samples {
            let system = sample["system"].as_str().expect("system");
            let backend = sample["backend"].as_str().expect("backend");
            let version = sample["version"].as_str().expect("version");
            let tuple = format!("{system}/{backend}/{version}");
            let data_root = systems.join(sample["data_root"].as_str().expect("data_root"));
            if !shipped_data_ready(
                &data_root,
                &[
                    "moe_a2a_perf.parquet",
                    "wideep_deepep_normal_perf.parquet",
                    "wideep_deepep_ll_perf.parquet",
                    "trtllm_alltoall_perf.parquet",
                ],
            ) {
                eprintln!(
                    "SKIP moe_a2a_op_matches_python_oracle: shipped perf data unavailable at {} \
                     (run `git lfs pull`)",
                    data_root.display()
                );
                return;
            }
            let db = dbs.entry(tuple.clone()).or_insert_with(|| {
                PerfDatabase::load(&systems, system, backend, version)
                    .unwrap_or_else(|err| panic!("{tuple} must load: {err}"))
            });
            let u32_of = |field: &str| {
                u32::try_from(sample[field].as_u64().expect(field)).expect("fits in u32")
            };
            let op = MoeAllToAllOp {
                name: "oracle".into(),
                scale_factor: sample["scale_factor"].as_f64().expect("scale_factor"),
                phase: sample["phase"].as_str().expect("phase").to_string(),
                comm_backend: sample["comm_backend"]
                    .as_str()
                    .expect("comm_backend")
                    .to_string(),
                comm_dtype: sample["comm_dtype"]
                    .as_str()
                    .expect("comm_dtype")
                    .to_string(),
                hidden_size: u32_of("hidden_size"),
                topk: u32_of("topk"),
                num_experts: u32_of("num_experts"),
                moe_ep_size: u32_of("moe_ep_size"),
                node_num: u32_of("node_num"),
                sms: u32_of("sms"),
                attention_tp_size: u32_of("attention_tp_size"),
            };
            let got = op
                .query(db, u32_of("x"))
                .unwrap_or_else(|err| panic!("oracle sample {sample} must resolve: {err}"));
            let want = sample["latency_ms"].as_f64().expect("latency_ms");
            assert!(
                want > 0.0,
                "oracle sample has a non-positive latency: {sample}"
            );
            let rel = ((got.latency_ms - want) / want).abs();
            max_rel = max_rel.max(rel);
            assert!(
                rel <= 1e-9,
                "sample {sample}: rust {} vs python {want} (rel {rel:e})",
                got.latency_ms
            );
            checked += 1;
        }
        assert!(
            checked >= 55,
            "oracle unexpectedly small: {checked} samples"
        );
        eprintln!("moe_a2a op oracle: {checked} samples, max relative error {max_rel:e}");
    }
}
