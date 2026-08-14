// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unified large-EP MoE expert-compute operator.
//!
//! Rust port of Python `sdk/operations/moe_comm.py::MoEExpertCompute` — the OP layer
//! over the `perf_database/moe_expert_compute.rs` table (`_query_ep_table`'s silicon body
//! lives there; everything the op adds is here):
//!
//! - `attention_dp_size` GLOBALIZES the token count before the lookup
//!   (moe_comm.py:1291): the A2A dispatch delivers every DP rank's tokens to
//!   the experts. This is the opposite convention from the comm-side
//!   [`super::moe_a2a::MoeAllToAllOp`], which stays per-rank.
//! - `kernel_source = None` auto-resolves at QUERY time (:1292-1294 ->
//!   `_resolve_kernel_source`, :1121-1153). Every production caller leaves it
//!   unset (`models/blocks/moe.py:592-607` passes no `kernel_source`), so
//!   this path is the norm, not an edge case.
//! - the EPLB context correction `int(tokens * 0.8)` (:1295-1301) — sglang
//!   -adapted kernel legs only.
//! - `moe_tp_size` is pinned to 1 in the table key (:1312): the large-EP
//!   family is EP-only.
//! - phase validation (`_validate_ep_phase`, :948-951) — Python raises
//!   `ValueError`, so this is a hard config error, NOT a data miss.
//! - the mode gate: SOL/SOL_FULL/EMPIRICAL have no estimation tier for this
//!   family (:1206-1210) and HYBRID's empirical leg is the same raise
//!   (:1270-1274). See [`MoeExpertComputeOp::query`].

use serde::{Deserialize, Serialize};

use crate::common::enums::{DatabaseMode, MoeQuantMode};
use crate::common::error::AicError;
use crate::operators::base::{PerformanceResult, Source};
use crate::perf_database::PerfDatabase;

/// Python `_EP_PHASES` (moe_comm.py:939).
const EP_PHASES: [&str; 2] = ["context", "generation"];

/// Python `_SGLANG_ADAPTED_KERNEL_SOURCES` (moe_comm.py:945) — the kernel
/// legs whose rows came through the sglang wideep adapters, and therefore the
/// only legs the legacy `int(tokens * 0.8)` EPLB correction applies to. The
/// trtllm legs express EPLB through the `_eplb` distribution suffix instead.
const SGLANG_ADAPTED_KERNEL_SOURCES: [&str; 1] = ["deepep_moe"];

/// The kernel every sglang/vllm large-EP MoE row is collected under
/// (moe_comm.py:1134-1135; spec §4.2).
const SGLANG_VLLM_KERNEL_SOURCE: &str = "deepep_moe";

/// Python `_validate_ep_phase` (moe_comm.py:948-951): an unknown
/// `inference_phase` is a `ValueError` — a programming error, deliberately
/// NOT a missing-data signal (it must not trigger HYBRID/`FallbackOp`
/// handling). Same [`AicError::InvalidEngineConfig`] mapping as the moe_a2a
/// registry guard and `moe_dispatch.rs`'s `op_name` guard.
fn validate_ep_phase(inference_phase: &str) -> Result<(), AicError> {
    if !EP_PHASES.contains(&inference_phase) {
        return Err(AicError::InvalidEngineConfig(format!(
            "Invalid inference_phase '{inference_phase}'. Must be one of {EP_PHASES:?}"
        )));
    }
    Ok(())
}

fn default_is_gated() -> bool {
    true
}

/// One inference phase of the unified large-EP expert compute (Python
/// `MoEExpertCompute`; one op instance per phase).
///
/// Field order is the wire format (bincode serializes struct fields
/// positionally) and matches the Python `_to_opspec` contract.
/// `#[serde(default)]` is applied exactly where the Python ctor has a default
/// (moe_comm.py:1036-1039).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MoeExpertComputeOp {
    pub name: String,
    pub scale_factor: f64,
    pub hidden_size: u32,
    /// FULL (un-partitioned) MoE intermediate size — the family is EP-only,
    /// so nothing is moe_tp-sharded.
    pub inter_size: u32,
    pub topk: u32,
    pub num_experts: u32,
    pub moe_ep_size: u32,
    pub quant_mode: MoeQuantMode,
    pub workload_distribution: String,
    /// Globalizes the token key (moe_comm.py:1291).
    pub attention_dp_size: u32,
    /// `"context"` or `"generation"`.
    pub inference_phase: String,
    /// EPLB slot count. `None` (the Python ctor's default) means
    /// `num_slots = num_experts` — no EPLB redundancy — resolved at query
    /// time exactly like moe_comm.py:1047.
    #[serde(default)]
    pub num_slots: Option<u32>,
    /// Pinned collected-kernel key, or `None` to auto-resolve per backend at
    /// query time (moe_comm.py:1037, :1292-1294). Production model builders
    /// always leave this `None`.
    #[serde(default)]
    pub kernel_source: Option<String>,
    /// Gated FFN (3 GEMMs) vs non-gated (2): sizes resident weights
    /// (`get_weights`, moe_comm.py:1058-1059) AND the beyond-range roofline
    /// SOL's gemm count (`3 if is_gated else 2`, mirroring the legacy oracle
    /// moe.py:309). Python default `True` (:1038).
    #[serde(default = "default_is_gated")]
    pub is_gated: bool,
    /// Python default `False` (:1039).
    #[serde(default)]
    pub enable_eplb: bool,
}

impl MoeExpertComputeOp {
    /// Query the expert compute at the PER-ATTENTION-DP-RANK token count
    /// `num_tokens` (Python `query`'s `x`).
    ///
    /// Mirrors moe_comm.py:1287-1320 in source order:
    /// 1. `num_tokens = kwargs.get("x") * self._attention_dp_size` (:1291).
    /// 2. `kernel_source` auto-resolution when unpinned (:1292-1294).
    /// 3. the EPLB context correction (:1295-1301) — expression-level
    ///    fidelity matters here: `int(num_tokens * 0.8)` is a FLOAT multiply
    ///    truncated toward zero, evaluated AFTER the attention-DP
    ///    globalization. `x=101, adp=2` gives `int(202 * 0.8) = 161`; the
    ///    reversed order would give `int(101 * 0.8) * 2 = 160` — an adjacent
    ///    collected point in the shipped tables.
    /// 4. `_validate_ep_phase` (:1195, reached through `query_moe_expert_compute`).
    /// 5. the mode gate (:1206-1210) — see below.
    /// 6. the table lookup with `moe_tp_size = 1` (:1302-1315) and
    ///    `PerformanceResult(float(result) * scale_factor, source=...)`
    ///    (:1316-1320); the silicon path tags `source="silicon"`
    ///    (`_interp_pr`, perf_database.py:2510-2516).
    ///
    /// MODE SEMANTICS (silicon-only family, identical to the comm side):
    /// - SILICON — table query; a miss propagates as the typed data miss.
    /// - HYBRID — table query, and a MISSING-DATA miss converts to
    ///   [`AicError::EmpiricalNotImplemented`], mirroring Python's HYBRID
    ///   `get_empirical()` leg (:1270-1274). Non-data errors propagate.
    /// - SOL / SOL_FULL / EMPIRICAL — `EmpiricalNotImplementedError` BEFORE
    ///   any table access (:1206-1210).
    pub fn query(&self, db: &PerfDatabase, num_tokens: u32) -> Result<PerformanceResult, AicError> {
        // moe_comm.py:1291 — attention DP globalizes tokens through the A2A
        // dispatch.
        let mut tokens = num_tokens.saturating_mul(self.attention_dp_size.max(1));
        // moe_comm.py:1292-1294.
        let kernel_source = match &self.kernel_source {
            Some(kernel) => kernel.clone(),
            None => self.resolve_kernel_source(db)?,
        };
        // moe_comm.py:1295-1301 — sglang-adapted legs, context phase only.
        if self.enable_eplb
            && self.inference_phase == "context"
            && SGLANG_ADAPTED_KERNEL_SOURCES.contains(&kernel_source.as_str())
        {
            tokens = (f64::from(tokens) * 0.8) as u32;
        }
        // moe_comm.py:1195 — validation precedes the mode gate (:1206), so an
        // invalid phase is a ValueError even under SOL/EMPIRICAL.
        validate_ep_phase(&self.inference_phase)?;
        match db.database_mode {
            DatabaseMode::Silicon | DatabaseMode::Hybrid => {}
            mode => {
                return Err(AicError::EmpiricalNotImplemented(format!(
                    "{mode:?} mode is not available for moe_expert_compute {kernel_source}/{}: silicon data \
                     required (estimation tier is a planned follow-up).",
                    self.inference_phase
                )))
            }
        }
        let latency = self
            .silicon_latency(db, &kernel_source, tokens)
            .map_err(|err| {
                if db.database_mode == DatabaseMode::Hybrid && err.is_missing_perf_data() {
                    // moe_comm.py:1270-1274 — HYBRID's empirical fallback for
                    // this family is itself the typed not-implemented raise.
                    AicError::EmpiricalNotImplemented(format!(
                        "HYBRID empirical fallback is not available for moe_expert_compute \
                         {kernel_source}/{}: silicon data required (estimation tier is a planned \
                         follow-up). Silicon miss: {err}",
                        self.inference_phase
                    ))
                } else {
                    err
                }
            })?;
        // Python: `PerformanceResult(float(result) * scale, source=...)` — no
        // clamp (nothing is subtracted on this path).
        Ok(PerformanceResult::new(latency, Source::Silicon).scaled(self.scale_factor))
    }

    fn silicon_latency(
        &self,
        db: &PerfDatabase,
        kernel_source: &str,
        tokens: u32,
    ) -> Result<f64, AicError> {
        db.moe_expert_compute.query(
            kernel_source,
            self.quant_mode,
            &self.workload_distribution,
            &self.inference_phase,
            self.topk,
            self.num_experts,
            // moe_comm.py:1047 — `num_slots if num_slots is not None else
            // num_experts`.
            self.num_slots.unwrap_or(self.num_experts),
            self.hidden_size,
            self.inter_size,
            1, // moe_tp_size — the large-EP family is EP-only (:1312)
            self.moe_ep_size,
            tokens,
            self.is_gated,
        )
    }

    /// Mirror of Python `MoEExpertCompute._resolve_kernel_source` (moe_comm.py:1120-1153)
    /// at QUERY time:
    /// 1. sglang / vllm -> `"deepep_moe"`, WITHOUT consulting the table;
    /// 2. otherwise (trtllm) SM >= 100 (Blackwell) with an `fp8_block`
    ///    quant-name SUBSTRING -> `"deepgemm"`, else `"moe_torch_flow"`;
    /// 3. keep that preference if the table collected it, else fall back to
    ///    the first collected kernel; an empty/unloadable table keeps the
    ///    preference (Python's `if ep_data:` falsy guard).
    ///
    /// `sm_version` reads through `unwrap_or(0)` where Python indexes
    /// `system_spec["gpu"]["sm_version"]` directly (KeyError if absent) —
    /// the same convention as `wideep_moe.rs::select_kernel`; every shipped
    /// system populates it.
    fn resolve_kernel_source(&self, db: &PerfDatabase) -> Result<String, AicError> {
        if db.backend == "sglang" || db.backend == "vllm" {
            return Ok(SGLANG_VLLM_KERNEL_SOURCE.to_string());
        }
        let is_blackwell = db.system_spec.gpu.sm_version.unwrap_or(0) >= 100;
        // Python: `"fp8_block" in quant_mode_str` (substring, not equality).
        let is_fp8_block = self.quant_mode.name().contains("fp8_block");
        let preferred = if is_blackwell && is_fp8_block {
            "deepgemm"
        } else {
            "moe_torch_flow"
        };
        let available = db.moe_expert_compute.available_kernels()?;
        if available.iter().any(|kernel| kernel == preferred) {
            return Ok(preferred.to_string());
        }
        if let Some(first) = available.into_iter().next() {
            return Ok(first);
        }
        Ok(preferred.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::enums::TransferPolicy;
    use crate::common::system_spec::SystemSpec;
    use crate::perf_database::MoeExpertComputeTable;
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

    /// One synthetic new-schema `moe_expert_compute_perf.parquet` row. Shape is pinned at
    /// (topk=8, experts=128, hidden=7168, inter=2048, moe_tp=1, ep=16); the
    /// axes under test are kernel/phase/slots/tokens.
    struct EpRow {
        kernel_source: &'static str,
        inference_phase: &'static str,
        num_slots: i64,
        num_tokens: i64,
        /// Already MILLIseconds — the moe_expert_compute schema's unit.
        latency_ms: f64,
    }

    fn ep_row(
        kernel_source: &'static str,
        inference_phase: &'static str,
        num_slots: i64,
        num_tokens: i64,
        latency_ms: f64,
    ) -> EpRow {
        EpRow {
            kernel_source,
            inference_phase,
            num_slots,
            num_tokens,
            latency_ms,
        }
    }

    fn write_moe_ep_parquet(path: &Path, rows: &[EpRow]) {
        let schema = Arc::new(
            parse_message_type(
                "message ep {
                    REQUIRED BYTE_ARRAY kernel_source (UTF8);
                    REQUIRED BYTE_ARRAY moe_dtype (UTF8);
                    REQUIRED BYTE_ARRAY distribution (UTF8);
                    REQUIRED BYTE_ARRAY inference_phase (UTF8);
                    REQUIRED INT64 topk;
                    REQUIRED INT64 num_experts;
                    REQUIRED INT64 num_slots;
                    REQUIRED INT64 hidden_size;
                    REQUIRED INT64 inter_size;
                    REQUIRED INT64 moe_tp_size;
                    REQUIRED INT64 moe_ep_size;
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
                .map(|r| ByteArray::from(r.kernel_source))
                .collect::<Vec<_>>(),
        );
        write_column::<ByteArrayType>(&mut rg, &vec![ByteArray::from("fp8_block"); n]);
        write_column::<ByteArrayType>(&mut rg, &vec![ByteArray::from("uniform"); n]);
        write_column::<ByteArrayType>(
            &mut rg,
            &rows
                .iter()
                .map(|r| ByteArray::from(r.inference_phase))
                .collect::<Vec<_>>(),
        );
        write_column::<Int64Type>(&mut rg, &vec![8_i64; n]);
        write_column::<Int64Type>(&mut rg, &vec![128_i64; n]);
        write_column::<Int64Type>(
            &mut rg,
            &rows.iter().map(|r| r.num_slots).collect::<Vec<_>>(),
        );
        write_column::<Int64Type>(&mut rg, &vec![7168_i64; n]);
        write_column::<Int64Type>(&mut rg, &vec![2048_i64; n]);
        write_column::<Int64Type>(&mut rg, &vec![1_i64; n]);
        write_column::<Int64Type>(&mut rg, &vec![16_i64; n]);
        write_column::<Int64Type>(
            &mut rg,
            &rows.iter().map(|r| r.num_tokens).collect::<Vec<_>>(),
        );
        write_column::<DoubleType>(
            &mut rg,
            &rows.iter().map(|r| r.latency_ms).collect::<Vec<_>>(),
        );
        rg.close().unwrap();
        writer.close().unwrap();
    }

    /// Token/latency pairs mirroring the Python op suite's injected store
    /// (`tests/unit/sdk/database/test_ep_moe_op.py:106-117`): the context
    /// curve carries 80 (= `int(100 * 0.8)`) and the adjacent 160/161 pair
    /// that pins the correction ORDER.
    fn synthetic_rows() -> Vec<EpRow> {
        vec![
            ep_row("deepep_moe", "context", 128, 64, 0.15),
            ep_row("deepep_moe", "context", 128, 80, 0.25),
            ep_row("deepep_moe", "context", 128, 100, 0.40),
            ep_row("deepep_moe", "context", 128, 160, 0.62),
            ep_row("deepep_moe", "context", 128, 161, 0.66),
            ep_row("deepep_moe", "generation", 128, 80, 0.33),
            ep_row("deepep_moe", "generation", 128, 100, 0.55),
            // EPLB-redundant slot slice on the same shape.
            ep_row("deepep_moe", "context", 256, 80, 0.90),
            ep_row("deepep_moe", "context", 256, 100, 0.95),
            // A trtllm-style leg: same points, different values, so an
            // EPLB correction leaking onto it is observable.
            ep_row("deepgemm", "context", 128, 80, 0.61),
            ep_row("deepgemm", "context", 128, 100, 0.77),
        ]
    }

    /// A real database (for the system spec / data-root plumbing) with the
    /// moe_expert_compute table swapped for a synthetic one — the `moe_dispatch.rs`
    /// pattern.
    fn synthetic_db(mode: DatabaseMode) -> (tempfile::TempDir, PerfDatabase) {
        let tmp = tempfile::tempdir().expect("tmpdir");
        write_moe_ep_parquet(
            &tmp.path().join("moe_expert_compute_perf.parquet"),
            &synthetic_rows(),
        );
        let systems = systems_root();
        let spec = SystemSpec::load(&systems.join("h200_sxm.yaml")).expect("system yaml must load");
        let mut db = PerfDatabase::load(&systems, "h200_sxm", "sglang", "0.5.6.post2")
            .expect("h200_sxm/sglang/0.5.6.post2 must load")
            .with_mode(mode, TransferPolicy::ALL);
        db.tables_mut().moe_expert_compute =
            MoeExpertComputeTable::new(tmp.path().to_path_buf(), spec);
        (tmp, db)
    }

    fn op() -> MoeExpertComputeOp {
        MoeExpertComputeOp {
            name: "moe".into(),
            scale_factor: 1.0,
            hidden_size: 7168,
            inter_size: 2048,
            topk: 8,
            num_experts: 128,
            moe_ep_size: 16,
            quant_mode: MoeQuantMode::Fp8Block,
            workload_distribution: "uniform".into(),
            attention_dp_size: 1,
            inference_phase: "context".into(),
            num_slots: None,
            kernel_source: Some("deepep_moe".into()),
            is_gated: true,
            enable_eplb: false,
        }
    }

    // -----------------------------------------------------------------
    // Token-key arithmetic (moe_comm.py:1291, :1295-1301)
    // -----------------------------------------------------------------

    /// `attention_dp_size` globalizes tokens BEFORE the lookup: x=20, adp=5
    /// keys 100.
    #[test]
    fn attention_dp_size_globalizes_the_token_key() {
        let (_tmp, db) = synthetic_db(DatabaseMode::Silicon);
        let mut scaled = op();
        scaled.attention_dp_size = 5;
        let got = scaled.query(&db, 20).expect("adp query");
        assert!((got.latency_ms - 0.40).abs() < 1e-12, "got {got:?}");
        // adp=1 at the same x keys 20 — off the collected low end, so the
        // value differs (the scaling is observable either way).
        let unscaled = op().query(&db, 20).expect("adp=1 query");
        assert!(
            (unscaled.latency_ms - 0.40).abs() > 1e-9,
            "adp must move the token key, got {unscaled:?}"
        );
    }

    /// EPLB fires on the sglang-adapted leg during context:
    /// `int(100 * 0.8) = 80`.
    #[test]
    fn eplb_context_correction_on_the_sglang_leg() {
        let (_tmp, db) = synthetic_db(DatabaseMode::Silicon);
        let mut eplb = op();
        eplb.enable_eplb = true;
        let corrected = eplb.query(&db, 100).expect("eplb query");
        assert!(
            (corrected.latency_ms - 0.25).abs() < 1e-12,
            "got {corrected:?}"
        );
        // Same value as an uncorrected query at 80.
        let plain = op().query(&db, 80).expect("plain query");
        assert_eq!(corrected.latency_ms, plain.latency_ms);
    }

    /// ORDER pin (the expression-level fidelity case): globalize by
    /// attention_dp FIRST, truncate SECOND. x=101, adp=2 -> `int(202 * 0.8)`
    /// = 161 (0.66), NOT `int(101 * 0.8) * 2` = 160 (0.62) — both are
    /// collected points, so a drift lands on the wrong leaf.
    #[test]
    fn eplb_correction_globalizes_before_truncating() {
        let (_tmp, db) = synthetic_db(DatabaseMode::Silicon);
        let mut eplb = op();
        eplb.enable_eplb = true;
        eplb.attention_dp_size = 2;
        let got = eplb.query(&db, 101).expect("order query");
        assert!(
            (got.latency_ms - 0.66).abs() < 1e-12,
            "int(101*2*0.8) = 161 -> 0.66, got {got:?}"
        );
    }

    /// Without `enable_eplb` the context query stays uncorrected: x=100 hits
    /// the 100-token point (0.40), not the corrected 80 (0.25).
    #[test]
    fn eplb_default_off_is_a_noop() {
        let (_tmp, db) = synthetic_db(DatabaseMode::Silicon);
        let plain = op().query(&db, 100).expect("eplb-off query");
        assert!((plain.latency_ms - 0.40).abs() < 1e-12, "got {plain:?}");
    }

    /// Generation NEVER takes the correction (the legacy sglang query applied
    /// it during prefill only).
    #[test]
    fn eplb_never_fires_in_the_generation_phase() {
        let (_tmp, db) = synthetic_db(DatabaseMode::Silicon);
        let mut eplb = op();
        eplb.enable_eplb = true;
        eplb.inference_phase = "generation".into();
        let got = eplb.query(&db, 100).expect("generation query");
        assert!(
            (got.latency_ms - 0.55).abs() < 1e-12,
            "generation must hit the 100-token point (0.55), got {got:?}"
        );
    }

    /// Non-sglang-adapted kernel legs never take the correction (TrtLLM's
    /// EPLB effect rides the `_eplb` distribution suffix instead).
    #[test]
    fn eplb_never_fires_on_a_non_deepep_kernel_source() {
        let (_tmp, db) = synthetic_db(DatabaseMode::Silicon);
        let mut eplb = op();
        eplb.enable_eplb = true;
        eplb.kernel_source = Some("deepgemm".into());
        let got = eplb.query(&db, 100).expect("deepgemm query");
        assert!(
            (got.latency_ms - 0.77).abs() < 1e-12,
            "the deepgemm leg must stay uncorrected (0.77), got {got:?}"
        );
    }

    /// `scale_factor` multiplies the resolved latency (moe_comm.py:1317).
    #[test]
    fn scale_factor_multiplies_the_resolved_latency() {
        let (_tmp, db) = synthetic_db(DatabaseMode::Silicon);
        let mut scaled = op();
        scaled.scale_factor = 61.0;
        let got = scaled.query(&db, 100).expect("scaled query");
        assert!((got.latency_ms - 0.40 * 61.0).abs() < 1e-12, "got {got:?}");
        assert_eq!(got.source, Source::Silicon);
    }

    // -----------------------------------------------------------------
    // num_slots defaulting (moe_comm.py:1047)
    // -----------------------------------------------------------------

    /// `None` -> `num_experts` (128), not the 256-slot EPLB-redundant slice
    /// collected at the same shape.
    #[test]
    fn num_slots_defaults_to_num_experts() {
        let (_tmp, db) = synthetic_db(DatabaseMode::Silicon);
        let defaulted = op().query(&db, 80).expect("default slots");
        assert!(
            (defaulted.latency_ms - 0.25).abs() < 1e-12,
            "got {defaulted:?}"
        );
        let mut pinned = op();
        pinned.num_slots = Some(256);
        let redundant = pinned.query(&db, 80).expect("redundant slots");
        assert!(
            (redundant.latency_ms - 0.90).abs() < 1e-12,
            "got {redundant:?}"
        );
    }

    // -----------------------------------------------------------------
    // kernel_source auto-resolution (moe_comm.py:1120-1153)
    // -----------------------------------------------------------------

    /// sglang / vllm resolve to `"deepep_moe"` WITHOUT consulting the table.
    #[test]
    fn kernel_source_auto_resolves_to_deepep_moe_on_sglang_and_vllm() {
        for backend in ["sglang", "vllm"] {
            let (_tmp, mut db) = synthetic_db(DatabaseMode::Silicon);
            db.tables_mut().backend = backend.to_string();
            let mut auto = op();
            auto.kernel_source = None;
            let got = auto.query(&db, 100).expect("auto-resolved query");
            assert!(
                (got.latency_ms - 0.40).abs() < 1e-12,
                "{backend}: got {got:?}"
            );
        }
    }

    /// trtllm on Blackwell with an fp8_block quant prefers `"deepgemm"`,
    /// which the synthetic table collects.
    #[test]
    fn kernel_source_auto_resolution_prefers_deepgemm_on_blackwell_trtllm() {
        let (_tmp, mut db) = synthetic_db(DatabaseMode::Silicon);
        {
            let tables = db.tables_mut();
            tables.backend = "trtllm".to_string();
            tables.system_spec.gpu.sm_version = Some(100);
        }
        let mut auto = op();
        auto.kernel_source = None;
        let got = auto.query(&db, 100).expect("auto-resolved query");
        assert!(
            (got.latency_ms - 0.77).abs() < 1e-12,
            "must resolve deepgemm (0.77), got {got:?}"
        );
    }

    /// The availability fallback: a non-fp8_block quant prefers
    /// `"moe_torch_flow"`, which is uncollected — Python takes the first
    /// collected kernel instead. Also the shape of every shipped trtllm
    /// table (one collected kernel, never the preferred name).
    #[test]
    fn kernel_source_auto_resolution_falls_back_to_a_collected_kernel() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        write_moe_ep_parquet(
            &tmp.path().join("moe_expert_compute_perf.parquet"),
            &[ep_row("wideep_compute_cutlass", "context", 128, 100, 0.42)],
        );
        let systems = systems_root();
        let spec = SystemSpec::load(&systems.join("h200_sxm.yaml")).expect("system yaml");
        let mut db = PerfDatabase::load(&systems, "h200_sxm", "sglang", "0.5.6.post2")
            .expect("db must load");
        {
            let tables = db.tables_mut();
            tables.backend = "trtllm".to_string();
            tables.system_spec.gpu.sm_version = Some(100);
            tables.moe_expert_compute = MoeExpertComputeTable::new(tmp.path().to_path_buf(), spec);
        }
        let mut auto = op();
        auto.kernel_source = None;
        let got = auto.query(&db, 100).expect("fallback query");
        assert!((got.latency_ms - 0.42).abs() < 1e-12, "got {got:?}");
        // ... and the EPLB correction stays off on that leg even though it is
        // a context query with eplb enabled.
        auto.enable_eplb = true;
        let uncorrected = auto.query(&db, 100).expect("fallback eplb query");
        assert!(
            (uncorrected.latency_ms - 0.42).abs() < 1e-12,
            "got {uncorrected:?}"
        );
    }

    // -----------------------------------------------------------------
    // Phase validation vs data miss (_validate_ep_phase)
    // -----------------------------------------------------------------

    /// Python's `ValueError` -> [`AicError::InvalidEngineConfig`], which is
    /// NOT `is_missing_perf_data` and never converts into a HYBRID estimate
    /// or a `FallbackOp` chain.
    #[test]
    fn unknown_inference_phase_is_a_config_error_not_a_data_miss() {
        let (_tmp, db) = synthetic_db(DatabaseMode::Silicon);
        let mut bad = op();
        bad.inference_phase = "prefill".into();
        let result = bad.query(&db, 100);
        assert!(
            matches!(&result, Err(AicError::InvalidEngineConfig(msg)) if msg.contains("inference_phase")),
            "got {result:?}"
        );
        assert!(!result.unwrap_err().is_missing_perf_data());
    }

    /// A KNOWN phase on a shape the table does not carry stays an ordinary
    /// typed data miss.
    #[test]
    fn uncollected_shape_is_a_data_miss() {
        let (_tmp, db) = synthetic_db(DatabaseMode::Silicon);
        let mut missing = op();
        missing.hidden_size = 9999;
        let result = missing.query(&db, 100);
        assert!(
            matches!(result, Err(AicError::PerfDatabase(_))),
            "got {result:?}"
        );
    }

    // -----------------------------------------------------------------
    // Mode semantics (adjudication R5 / moe_comm.py:1206-1210, :1270-1274)
    // -----------------------------------------------------------------

    #[test]
    fn estimation_tiers_raise_empirical_not_implemented() {
        for mode in [
            DatabaseMode::Sol,
            DatabaseMode::SolFull,
            DatabaseMode::Empirical,
        ] {
            let (_tmp, db) = synthetic_db(mode);
            let result = op().query(&db, 100);
            assert!(
                matches!(&result, Err(AicError::EmpiricalNotImplemented(msg))
                    if msg.contains("silicon data required")),
                "{mode:?}: got {result:?}"
            );
        }
    }

    #[test]
    fn hybrid_uses_silicon_when_covered_and_raises_not_implemented_on_a_miss() {
        let (_tmp, db) = synthetic_db(DatabaseMode::Hybrid);
        let hit = op().query(&db, 100).expect("covered shape");
        assert!((hit.latency_ms - 0.40).abs() < 1e-12, "got {hit:?}");
        assert_eq!(hit.source, Source::Silicon);

        let mut missing = op();
        missing.hidden_size = 9999;
        let result = missing.query(&db, 100);
        assert!(
            matches!(&result, Err(AicError::EmpiricalNotImplemented(msg))
                if msg.contains("silicon data required")),
            "got {result:?}"
        );
    }

    /// Phase validation precedes the mode gate (moe_comm.py:1195 vs :1206).
    #[test]
    fn phase_validation_precedes_the_mode_gate() {
        let (_tmp, db) = synthetic_db(DatabaseMode::Empirical);
        let mut bad = op();
        bad.inference_phase = "prefill".into();
        let result = bad.query(&db, 100);
        assert!(
            matches!(result, Err(AicError::InvalidEngineConfig(_))),
            "got {result:?}"
        );
    }

    // -----------------------------------------------------------------
    // Serde defaults (Python ctor defaults, moe_comm.py:1036-1039)
    // -----------------------------------------------------------------

    #[test]
    fn omitted_optional_fields_take_the_python_ctor_defaults() {
        let json = r#"{
            "name": "moe",
            "scale_factor": 1.0,
            "hidden_size": 7168,
            "inter_size": 2048,
            "topk": 8,
            "num_experts": 128,
            "moe_ep_size": 16,
            "quant_mode": "fp8_block",
            "workload_distribution": "uniform",
            "attention_dp_size": 1,
            "inference_phase": "context"
        }"#;
        let op: MoeExpertComputeOp = serde_json::from_str(json).expect("defaults must fill in");
        assert_eq!(op.num_slots, None);
        assert_eq!(op.kernel_source, None);
        assert!(op.is_gated);
        assert!(!op.enable_eplb);
    }

    // -----------------------------------------------------------------
    // Python op-level oracle over the shipped data
    // -----------------------------------------------------------------

    const LFS_POINTER_PREFIX: &[u8] = b"version https://git-lfs";

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

    /// The `"moe_expert_compute"` slice of the shared op-level oracle fixture (the same
    /// file `moe_a2a.rs` reads for its `"moe_a2a"` slice — one generator run
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

    /// OP-level parity against Python `MoEExpertCompute(...).query(db, x=...)` — not the
    /// raw table query: what is under test is the op layer's
    /// `x * attention_dp_size` globalization, the `int(tokens * 0.8)` EPLB
    /// correction, the `num_slots` default, the `kernel_source=None`
    /// auto-resolution and the `* scale_factor` tail. The fixture is
    /// generated by `parity_tests/gen_op_oracle.py` (regeneration command in
    /// the JSON's `_regenerate` field) from the shipped h200_sxm/sglang
    /// (legacy sglang wideep pair — the `deepep_moe` leg where EPLB fires)
    /// and gb200/trtllm (legacy trtllm wideep — the leg where it must not)
    /// data over attention_dp_size in {1, 2, 8}, both phases, eplb on/off and
    /// token overflow.
    #[test]
    fn moe_ep_op_matches_python_oracle() {
        let systems = systems_root();
        let samples = oracle_samples("moe_expert_compute");
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
                    "moe_expert_compute_perf.parquet",
                    "wideep_context_moe_perf.parquet",
                    "wideep_generation_moe_perf.parquet",
                    "wideep_moe_perf.parquet",
                ],
            ) {
                eprintln!(
                    "SKIP moe_ep_op_matches_python_oracle: shipped perf data unavailable at {} \
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
            let quant: MoeQuantMode = serde_json::from_value(sample["quant_mode"].clone())
                .expect("quant must map to a MoeQuantMode");
            let op = MoeExpertComputeOp {
                name: "oracle".into(),
                scale_factor: sample["scale_factor"].as_f64().expect("scale_factor"),
                hidden_size: u32_of("hidden_size"),
                inter_size: u32_of("inter_size"),
                topk: u32_of("topk"),
                num_experts: u32_of("num_experts"),
                moe_ep_size: u32_of("moe_ep_size"),
                quant_mode: quant,
                workload_distribution: sample["workload_distribution"]
                    .as_str()
                    .expect("workload_distribution")
                    .to_string(),
                attention_dp_size: u32_of("attention_dp_size"),
                inference_phase: sample["inference_phase"]
                    .as_str()
                    .expect("inference_phase")
                    .to_string(),
                num_slots: sample["num_slots"]
                    .as_u64()
                    .map(|v| u32::try_from(v).expect("fits in u32")),
                kernel_source: sample["kernel_source"].as_str().map(str::to_string),
                is_gated: sample["is_gated"].as_bool().expect("is_gated"),
                enable_eplb: sample["enable_eplb"].as_bool().expect("enable_eplb"),
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
        eprintln!(
            "moe_expert_compute op oracle: {checked} samples, max relative error {max_rel:e}"
        );
    }
}
