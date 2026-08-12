// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Whole-model forward-pass op (Python `forward_model="fpm"`, class
//! `FPMForwardOp` in `sdk/operations/fpm_forward.py`).
//!
//! With `forward_model="fpm"` the Python model builder replaces each phase op
//! list with exactly one of these; the compiled spec carries it as
//! `Op::FpmForward`. The op answers the standard `query(db, ctx)` contract
//! from the collected [`FpmForwardTable`](crate::perf_database::FpmForwardTable)
//! cells:
//!
//! - prefill coords `(B, B*s, B*prefix)` — `s` is NEW prefill tokens per
//!   request, `prefix` the past-KV per request;
//! - decode coords `(B, B*s)` — one new token per request, `s` the per-request
//!   KV length at this decode step;
//! - hard per-axis domain gate BEFORE interpolation (FPM never extrapolates);
//! - ScatteredSites resolution with the exact Python config
//!   (`own_curve_coverage_fallback=true`, `max_site_distance=2.0`).
//!
//! The SOL roofline anchoring the interpolation is the model's ORIGINAL
//! op-level list (carried as `sol_ops` on the wire) queried in SOL mode with
//! Python's coordinate back-mapping — see [`sol_total`].
//!
//! NOT the crate's `src/fpm/` (`ForwardPassPerfModel`) module: that is the
//! online-tuning model over Dynamo ForwardPassMetrics telemetry, an unrelated
//! concept that shares the "FPM" abbreviation.

use serde::{Deserialize, Serialize};

use crate::common::error::AicError;
use crate::operators::op::{Op, RuntimeContext};
use crate::operators::{PerformanceResult, Source};
use crate::perf_database::fpm_forward::{
    FpmForwardCell, FPM_DECODE_AXES, FPM_PREFILL_AXES,
};
use crate::perf_database::perf_interp::{OpInterpConfig, Resolver, ValueTransform};
use crate::perf_database::PerfDatabase;

/// Phase of the whole-model op. Serialized lowercase, matching the Python
/// `phase` string and the parquet `workload_kind` values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FpmPhase {
    Prefill,
    Decode,
}

impl FpmPhase {
    pub fn as_str(&self) -> &'static str {
        match self {
            FpmPhase::Prefill => "prefill",
            FpmPhase::Decode => "decode",
        }
    }
}

/// One whole-model forward pass for a single phase.
///
/// `match_identity` is the 11-string cell identity in
/// [`FPM_CELL_MATCH_COLUMNS`](crate::perf_database::fpm_forward::FPM_CELL_MATCH_COLUMNS)
/// order, computed by the PYTHON producer via `_norm_identity` (None -> "",
/// Enum -> `.name`) so Rust compares strings verbatim with no re-normalization
/// drift. `sol_ops` is the model's original op-level list for this phase —
/// the roofline source, serialized recursively like `Overlap`/`Fallback`
/// children.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FpmForwardOp {
    pub name: String,
    pub phase: FpmPhase,
    pub model_path: String,
    pub match_identity: Vec<String>,
    /// Per-rank whole-model weight bytes (captured from the original op-level
    /// lists at rewrite time). Carried for memory estimation; the latency
    /// path does not read it.
    #[serde(default)]
    pub weight_bytes: f64,
    pub sol_ops: Vec<Op>,
}

fn data_err(msg: String) -> AicError {
    AicError::PerfDatabase(msg)
}

impl FpmForwardOp {
    /// Mirror of Python `FPMForwardOp.query`: validate kwargs, map to
    /// iteration-total coordinates, resolve against the selected cell.
    pub fn query(
        &self,
        db: &PerfDatabase,
        ctx: &RuntimeContext,
    ) -> Result<PerformanceResult, AicError> {
        let batch_size = ctx.batch_size;
        let s = ctx.s;
        if batch_size < 1 || s < 1 {
            return Err(data_err(format!(
                "invalid FPM query: batch_size={batch_size}, s={s}"
            )));
        }
        if ctx.beam_width != 1 {
            return Err(data_err(format!(
                "forward_model='fpm' has no beam-search data (beam_width={}); use \
                 forward_model='op_level'.",
                ctx.beam_width
            )));
        }
        let cell = db
            .fpm_forward
            .select_cell(&self.match_identity, &self.model_path)?;
        let b = batch_size as f64;
        let coords: Vec<f64> = match self.phase {
            FpmPhase::Prefill => {
                let prefix = ctx.prefix as f64;
                vec![b, b * s as f64, b * prefix]
            }
            // One new token per request; `s` is the per-request KV length at
            // this decode step, so the iteration reads batch*s KV tokens.
            FpmPhase::Decode => vec![b, b * s as f64],
        };
        self.resolve(db, cell, &coords)
    }

    /// Resolve at RAW per-rank iteration totals — the table's native
    /// coordinate system (prefill `(n_req, total_prefill, total_kv)`, decode
    /// `(n_req, total_kv)`). Used by the ForwardPassMetrics rank dispatch,
    /// where the telemetry already carries totals: converting them to
    /// per-request averages and back (the op-level path's convention) would
    /// only add integer-division rounding.
    pub fn query_totals(
        &self,
        db: &PerfDatabase,
        coords: &[f64],
    ) -> Result<PerformanceResult, AicError> {
        let expected = match self.phase {
            FpmPhase::Prefill => 3,
            FpmPhase::Decode => 2,
        };
        if coords.len() != expected {
            return Err(data_err(format!(
                "FPM {} query_totals expects {expected} coords, got {:?}",
                self.phase.as_str(),
                coords
            )));
        }
        let cell = db
            .fpm_forward
            .select_cell(&self.match_identity, &self.model_path)?;
        self.resolve(db, cell, coords)
    }

    /// Decode-pass baseline at the smallest collectable KV for this batch:
    /// `kv_floor = max(batch, decode-domain KV min)`. See the Python
    /// docstring — `query(B, KV) - query_pass_baseline(B)` is the decode
    /// work's marginal cost when it rides an existing (mixed) pass.
    pub fn query_pass_baseline(
        &self,
        db: &PerfDatabase,
        batch_size: u32,
    ) -> Result<PerformanceResult, AicError> {
        if self.phase != FpmPhase::Decode {
            return Err(data_err(format!(
                "query_pass_baseline is decode-only, called on phase {:?}",
                self.phase.as_str()
            )));
        }
        if batch_size < 1 {
            return Err(data_err(format!(
                "invalid FPM baseline query: batch_size={batch_size}"
            )));
        }
        let cell = db
            .fpm_forward
            .select_cell(&self.match_identity, &self.model_path)?;
        let Some(domain) = cell.decode_domain else {
            return Err(data_err(format!(
                "FPM cell {:?} has no decode rows (model_path={:?}).",
                cell.cell_ids, cell.model_path
            )));
        };
        let kv_floor = batch_size.max(domain[1].0);
        self.resolve(db, cell, &[batch_size as f64, kv_floor as f64])
    }

    /// Domain gate + ScatteredSites resolution, mirroring `FPMForwardOp._resolve`.
    fn resolve(
        &self,
        db: &PerfDatabase,
        cell: &FpmForwardCell,
        coords: &[f64],
    ) -> Result<PerformanceResult, AicError> {
        // Data-certified prefill batch clamp (mirrors Python _resolve): the
        // regime coordinate is the token TOTAL, which stays untouched — the
        // clamped query prices the same side of the capture cliff and is a
        // bounded upper bound on attention. Decode is NEVER clamped (its
        // batch axis carries a real regime cliff — the Task B partition).
        // KV-pressure ceiling (mirrors Python _PREFILL_CLAMP_MAX_KV_PRESSURE):
        // randtok LOO measured the clamp at median <= 2% below kv/T = 2 and
        // up to median 14% / p90 96% above it — high-pressure queries stay
        // hard-gated, with no SOL-corrected substitute.
        const MAX_KV_PRESSURE: f64 = 2.0;
        let clamped: Vec<f64>;
        let coords = match (self.phase, cell.prefill_batch_clamp_max) {
            (FpmPhase::Prefill, Some(max))
                if coords[0] > max as f64 && coords[2] < MAX_KV_PRESSURE * coords[1] =>
            {
                clamped = std::iter::once(max as f64)
                    .chain(coords[1..].iter().copied())
                    .collect();
                clamped.as_slice()
            }
            _ => coords,
        };
        // Per-axis inclusive bounding-box gate BEFORE perf_interp: whole-model
        // latency has no principled boundary-hold semantics.
        let (axes, domain, index): (&[&str], &[(u32, u32)], _) = match self.phase {
            FpmPhase::Prefill => {
                let Some(domain) = cell.prefill_domain.as_ref() else {
                    return Err(self.no_rows_err(cell));
                };
                (&FPM_PREFILL_AXES, domain.as_slice(), cell.prefill_index.as_ref())
            }
            FpmPhase::Decode => {
                let Some(domain) = cell.decode_domain.as_ref() else {
                    return Err(self.no_rows_err(cell));
                };
                // Regime routing AFTER the full-domain gate (the gate uses
                // the whole decode box): the boundary batch itself is
                // graph-side; each side interpolates among its own regime.
                let index = match cell.decode_regime_boundary {
                    Some(boundary) if coords[0] <= boundary as f64 => {
                        cell.decode_graph_index.as_ref()
                    }
                    Some(_) => cell.decode_eager_index.as_ref(),
                    None => cell.decode_index.as_ref(),
                };
                (&FPM_DECODE_AXES, domain.as_slice(), index)
            }
        };
        for (axis_index, (axis_name, &value)) in axes.iter().zip(coords).enumerate() {
            let (low, high) = domain[axis_index];
            if !((low as f64) <= value && value <= (high as f64)) {
                return Err(data_err(format!(
                    "FPM {} query {axis_name}={value} is outside the collected domain \
                     [{low}, {high}] for model_path={:?}. FPM never extrapolates; collect a \
                     wider sweep or use forward_model='op_level'.",
                    self.phase.as_str(),
                    cell.model_path
                )));
            }
        }
        let index = index.ok_or_else(|| self.no_rows_err(cell))?;

        // SOL support is checked LAZILY, mirroring Python: exact hits and
        // in-curve lerps never invoke the roofline (Python's SOL view answers
        // every op family; `_oplevel_sol_fn` is only called on transfer/hold
        // paths). A family the Rust SOL port does not cover yet (DSA / MSA /
        // MLA modules — e.g. GLM-5.2's DSA attention) therefore only fails
        // the sol-dependent resolution paths, and the error names the op.
        let sol_failure: std::cell::RefCell<Option<AicError>> = std::cell::RefCell::new(None);
        let sol = |sol_coords: &[f64]| -> f64 {
            match sol_total(&self.sol_ops, self.phase, db, sol_coords) {
                Ok(v) => v,
                Err(err) => {
                    let mut slot = sol_failure.borrow_mut();
                    if slot.is_none() {
                        *slot = Some(err);
                    }
                    f64::NAN
                }
            }
        };
        let cfg = interp_config(self.phase, &sol);
        let latency = index
            .resolve_value(&cfg, coords)
            .map(|value| value.latency)
            .map_err(|err| {
                match sol_failure.borrow_mut().take() {
                    Some(sol_err) => data_err(format!("{err}; SOL roofline unavailable: {sol_err}")),
                    None => err,
                }
            })?;
        if !latency.is_finite() || latency <= 0.0 {
            return Err(data_err(format!(
                "FPM {} interpolation produced an invalid latency ({latency}) at {coords:?}.",
                self.phase.as_str()
            )));
        }
        // Latency-only dataset; scale_factor is fixed 1.0 in Python.
        Ok(PerformanceResult::new(latency, Source::Silicon))
    }

    fn no_rows_err(&self, cell: &FpmForwardCell) -> AicError {
        data_err(format!(
            "FPM cell {:?} has no {} rows (model_path={:?}).",
            cell.cell_ids,
            self.phase.as_str(),
            cell.model_path
        ))
    }
}

/// The exact Python interp configs (`fpm_prefill_config` / `fpm_decode_config`
/// + `ScatteredSites` defaults): prefill sites `(batch, kv)` owning the
/// new-token curve; decode sites `(batch,)` owning the KV curve.
fn interp_config<'a>(phase: FpmPhase, sol: &'a dyn Fn(&[f64]) -> f64) -> OpInterpConfig<'a> {
    let (axes, site_axes, curve_axis): (&'static [&'static str], Vec<usize>, usize) = match phase {
        FpmPhase::Prefill => (&FPM_PREFILL_AXES, vec![0, 2], 1),
        FpmPhase::Decode => (&FPM_DECODE_AXES, vec![0], 1),
    };
    OpInterpConfig {
        axes,
        resolver: Resolver::ScatteredSites {
            site_axes,
            curve_axis,
            nn_sites: 4,
            max_site_distance: Some(2.0),
            require_curve_coverage: true,
            k_tail: 3,
            own_curve_coverage_fallback: true,
        },
        sol_fn: sol,
        value_transform: ValueTransform::Raw,
        transform_axis: None,
    }
}

// ---------------------------------------------------------------------------
// SOL roofline: the op-level model queried in SOL mode
// ---------------------------------------------------------------------------

/// Whole-model SOL at the given FPM coordinates, mirroring Python
/// `_oplevel_sol_fn` exactly:
///
/// ```text
/// prefill (batch, total_prefill, total_kv):
///     s = max(total_prefill / batch, 1.0)      # float, NOT truncated
///     prefix = total_kv / batch                # float, unclamped
///     x = batch if "logits_gemm" in op.name else total_prefill
/// decode (batch, total_kv):
///     s = max(total_kv / batch, 1.0)
///     x = batch
/// total = Σ op.query(SOL view, x=x, batch_size=batch, beam_width=1, s=s, prefix=prefix)
/// ```
pub(crate) fn sol_total(
    sol_ops: &[Op],
    phase: FpmPhase,
    db: &PerfDatabase,
    coords: &[f64],
) -> Result<f64, AicError> {
    let (batch, s, prefix, default_x) = match phase {
        FpmPhase::Prefill => {
            let (batch, total_prefill, total_kv) = (coords[0], coords[1], coords[2]);
            ((batch), (total_prefill / batch).max(1.0), total_kv / batch, total_prefill)
        }
        FpmPhase::Decode => {
            let (batch, total_kv) = (coords[0], coords[1]);
            (batch, (total_kv / batch).max(1.0), 0.0, batch)
        }
    };
    let mut total = 0.0_f64;
    for op in sol_ops {
        let x = if matches!(op, Op::Gemm(_)) && op.name().contains("logits_gemm") {
            batch
        } else {
            default_x
        };
        total += crate::operators::fpm_sol::op_sol_latency_ms(op, db, x, batch, s, prefix)?;
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perf_database::fpm_forward::tests::{default_identity, default_rows, write_pair};

    const SYSTEMS_ROOT: &str =
        concat!(env!("CARGO_MANIFEST_DIR"), "/../../src/aiconfigurator_core/systems");

    /// A PerfDatabase whose fpm_forward table points at a temp pair; the rest
    /// of the tables point at the real checked-in b200/vllm/0.19.0 fixture
    /// (never touched by these tests).
    fn db_with_pair(dir: &std::path::Path) -> PerfDatabase {
        let mut db = PerfDatabase::load(
            std::path::Path::new(SYSTEMS_ROOT),
            "b200_sxm",
            "vllm",
            "0.19.0",
        )
        .expect("fixture db");
        db.set_fpm_forward_for_test(crate::perf_database::FpmForwardTable::new(
            dir.to_path_buf(),
            "b200_sxm",
            "vllm",
            "0.25.1",
        ));
        db
    }

    fn op(phase: FpmPhase) -> FpmForwardOp {
        FpmForwardOp {
            name: format!("fpm_forward_{}", phase.as_str()),
            phase,
            model_path: "org/model-a".to_string(),
            match_identity: default_identity(4),
            weight_bytes: 0.0,
            // Empty sol_ops: exact hits and in-curve lerps never call SOL.
            sol_ops: vec![],
        }
    }

    fn ctx(batch_size: u32, s: u32, prefix: u32) -> RuntimeContext {
        RuntimeContext {
            batch_size,
            s,
            prefix,
            ..Default::default()
        }
    }

    #[test]
    fn certified_prefill_batch_clamp_routes_to_the_ceiling() {
        use crate::perf_database::fpm_forward::tests::RowSpec;
        let mk = |batch: u32, total: u32, lat: f64| RowSpec {
            workload_kind: "prefill",
            batch_size: batch,
            total_prefill_tokens: total,
            total_kv_read_tokens: 0,
            latency_ms: lat,
            ..RowSpec::default()
        };
        let mut rows = Vec::new();
        for (b, bump) in [(1u32, 1.0), (2, 1.02), (4, 1.04)] {
            for (total, lat) in [(1024u32, 10.0), (2048, 20.0), (4096, 40.0)] {
                rows.push(mk(b, total, lat * bump));
            }
        }
        let tmp = tempfile::tempdir().unwrap();
        write_pair(tmp.path(), &rows);
        let db = db_with_pair(tmp.path());
        // batch 16 > ceiling 4: clamp keeps the TRUE total (regime
        // coordinate) and answers the (4, 4096, 0) leaf.
        let clamped = op(FpmPhase::Prefill)
            .query_totals(&db, &[16.0, 4096.0, 0.0])
            .unwrap();
        assert!((clamped.latency_ms - 41.6).abs() < 1e-9, "{}", clamped.latency_ms);
        // The token axis stays honestly gated.
        let err = op(FpmPhase::Prefill)
            .query_totals(&db, &[16.0, 16384.0, 0.0])
            .unwrap_err();
        assert!(err.to_string().contains("outside the collected domain"), "{err}");
    }

    #[test]
    fn clamp_respects_the_kv_pressure_ceiling() {
        use crate::perf_database::fpm_forward::tests::RowSpec;
        let mk = |batch: u32, total: u32, kv: u32, lat: f64| RowSpec {
            workload_kind: "prefill",
            batch_size: batch,
            total_prefill_tokens: total,
            total_kv_read_tokens: kv,
            latency_ms: lat,
            ..RowSpec::default()
        };
        let mut rows = Vec::new();
        for (b, bump) in [(1u32, 1.0), (2, 1.02), (4, 1.04)] {
            for (total, lat) in [(1024u32, 10.0), (2048, 20.0), (4096, 40.0)] {
                rows.push(mk(b, total, 0, lat * bump));
                rows.push(mk(b, total, total, lat * bump * 1.2));
                rows.push(mk(b, total, 4 * total, lat * bump * 1.8));
            }
        }
        let tmp = tempfile::tempdir().unwrap();
        write_pair(tmp.path(), &rows);
        let db = db_with_pair(tmp.path());
        // kv/T = 1 (< 2): clamps to the (4, 4096, 4096) leaf.
        let low = op(FpmPhase::Prefill)
            .query_totals(&db, &[16.0, 4096.0, 4096.0])
            .unwrap();
        assert!((low.latency_ms - 40.0 * 1.04 * 1.2).abs() < 1e-9, "{}", low.latency_ms);
        // kv/T = 4 (>= 2): the pressure ceiling keeps the hard gate.
        let err = op(FpmPhase::Prefill)
            .query_totals(&db, &[16.0, 4096.0, 16384.0])
            .unwrap_err();
        assert!(err.to_string().contains("outside the collected domain"), "{err}");
    }

    #[test]
    fn decode_exact_hit_returns_leaf_verbatim() {
        let tmp = tempfile::tempdir().unwrap();
        write_pair(tmp.path(), &default_rows());
        let db = db_with_pair(tmp.path());
        // decode row (8, 4096) -> 7.0; s = 4096/8 = 512 per request
        let r = op(FpmPhase::Decode).query(&db, &ctx(8, 512, 0)).unwrap();
        assert_eq!(r.latency_ms, 7.0);
        assert_eq!(r.source, Source::Silicon);
    }

    #[test]
    fn prefill_exact_hit_with_prefix() {
        let tmp = tempfile::tempdir().unwrap();
        write_pair(tmp.path(), &default_rows());
        let db = db_with_pair(tmp.path());
        // prefill row (1, 2048, 2048) -> 24.0: B=1, s=2048 new tokens, prefix=2048
        let r = op(FpmPhase::Prefill).query(&db, &ctx(1, 2048, 2048)).unwrap();
        assert_eq!(r.latency_ms, 24.0);
    }

    #[test]
    fn decode_in_curve_lerp_is_linear_raw() {
        let tmp = tempfile::tempdir().unwrap();
        write_pair(tmp.path(), &default_rows());
        let db = db_with_pair(tmp.path());
        // Between (8, 8)->6.0 and (8, 4096)->7.0 at kv=2052 (B=8, s=256.5 ->
        // not integral; use s=256 -> kv=2048): w=(2048-8)/(4096-8)
        let r = op(FpmPhase::Decode).query(&db, &ctx(8, 256, 0)).unwrap();
        let w = (2048.0 - 8.0) / (4096.0 - 8.0);
        let expected = 6.0 + (7.0 - 6.0) * w;
        assert!((r.latency_ms - expected).abs() < 1e-12, "{}", r.latency_ms);
    }

    #[test]
    fn out_of_domain_is_a_hard_error() {
        let tmp = tempfile::tempdir().unwrap();
        write_pair(tmp.path(), &default_rows());
        let db = db_with_pair(tmp.path());
        // decode kv domain is [8, 65536]; B=8, s=16384 -> kv=131072 > max
        let err = op(FpmPhase::Decode).query(&db, &ctx(8, 16384, 0)).unwrap_err();
        assert!(err.to_string().contains("outside the collected domain"), "{err}");
        // batch below domain min
        let err = op(FpmPhase::Decode).query(&db, &ctx(1, 512, 0)).unwrap_err();
        assert!(err.to_string().contains("outside the collected domain"), "{err}");
    }

    #[test]
    fn invalid_query_args_error() {
        let tmp = tempfile::tempdir().unwrap();
        write_pair(tmp.path(), &default_rows());
        let db = db_with_pair(tmp.path());
        let err = op(FpmPhase::Decode).query(&db, &ctx(0, 512, 0)).unwrap_err();
        assert!(err.to_string().contains("invalid FPM query"), "{err}");
        let mut c = ctx(8, 512, 0);
        c.beam_width = 4;
        let err = op(FpmPhase::Decode).query(&db, &c).unwrap_err();
        assert!(err.to_string().contains("no beam-search data"), "{err}");
    }

    #[test]
    fn pass_baseline_uses_kv_floor() {
        let tmp = tempfile::tempdir().unwrap();
        write_pair(tmp.path(), &default_rows());
        let db = db_with_pair(tmp.path());
        // decode domain kv min is 8; batch=8 -> kv_floor = max(8, 8) = 8,
        // which is the exact collected point (8, 8) -> 6.0.
        let r = op(FpmPhase::Decode)
            .query_pass_baseline(&db, 8)
            .unwrap();
        assert_eq!(r.latency_ms, 6.0);
        // prefill op: decode-only API
        let err = op(FpmPhase::Prefill).query_pass_baseline(&db, 8).unwrap_err();
        assert!(err.to_string().contains("decode-only"), "{err}");
    }

    /// SOL support is lazy (mirrors Python, whose SOL view answers every op
    /// family): an unported family (e.g. GLM-5.2's DSA modules) must NOT
    /// block exact hits or in-curve lerps — only sol-dependent resolution
    /// paths fail, naming the unported op.
    #[test]
    fn unsupported_sol_family_is_lazy() {
        let tmp = tempfile::tempdir().unwrap();
        write_pair(tmp.path(), &default_rows());
        let db = db_with_pair(tmp.path());
        let mut o = op(FpmPhase::Decode);
        o.sol_ops = vec![Op::MlaBmm(crate::operators::MlaBmmOp {
            name: "mla_bmm_pre".into(),
            scale_factor: 1.0,
            num_heads: 128,
            is_pre: true,
            quant_mode: crate::common::enums::GemmQuantMode::Bfloat16,
        })];
        // Exact hit (8, 4096) -> 7.0: never invokes the roofline.
        let r = o.query(&db, &ctx(8, 512, 0)).unwrap();
        assert_eq!(r.latency_ms, 7.0);
        // In-curve lerp on the own site: RAW linear, no roofline either.
        assert!(o.query(&db, &ctx(8, 256, 0)).is_ok());
        // Uncollected batch -> site transfer NEEDS the roofline -> structured
        // miss naming the unported op (not a panic, not a silent number).
        let err = o.query(&db, &ctx(12, 512, 0)).unwrap_err();
        assert!(err.to_string().contains("SOL roofline unavailable"), "{err}");
        assert!(err.to_string().contains("no Rust implementation"), "{err}");
    }
}
