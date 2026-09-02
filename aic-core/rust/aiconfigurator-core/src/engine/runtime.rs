// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `Engine`: the compiled-spec execution core.
//!
//! Mirrors `aiconfigurator.sdk.backends.base_backend`'s static orchestration
//! (`run_static` / `run_static_latency_only` / `_run_static_breakdown` /
//! `_run_context_phase` / `_run_generation_phase`) but executes a precompiled
//! [`EngineSpec`] — Python no longer walks the op list per call. The per-phase
//! op iteration is the shared logic in [`crate::session`]
//! ([`run_context_ops`] / [`run_generation_ops_step`]); the `Engine` wraps the
//! stride quadrature and the `(nextn + 1)` decode-batch multiplier around it.
//!
//! The `Engine` is pure-Rust internals; its PyO3 bindings (`run_static`,
//! `predict_*_latency`, `mixed_step_latency`, `decode_step_latency`) and the
//! embedded [`crate::AicEngineBuilder`] live in [`crate::py`]. The agg sweep is
//! orchestrated in Python — there is no Rust `run_agg`.

use std::sync::Arc;

use crate::common::enums::{DatabaseMode, TransferPolicy};
use crate::common::error::AicError;
use crate::engine::spec::EngineSpec;
use crate::operators::base::PerformanceResult;
use crate::operators::{FpmForwardOp, FpmPhase, Op};
use crate::perf_database::PerfDatabase;
use crate::session::{
    get_mix_step_ops, query_context_op, query_generation_op, run_context_ops, run_context_ops_with,
    run_generation_ops_step, run_generation_ops_step_beamed_with, ContextOpFilter,
};
use crate::{validate_forward_pass_metrics, ForwardPassMetrics};

/// Per-call runtime inputs. Field-for-field mirror of the Python
/// `sdk/config.RuntimeConfig`.
///
/// The imbalance-correction scales thread into the per-op queries exactly
/// where Python applies them (`base_backend.py:331,372`): context-attention
/// ops multiply by `seq_imbalance_correction_scale`, generation-attention ops
/// by `gen_seq_imbalance_correction_scale`. (The FPM telemetry path has no
/// scale concept and keeps 1.0.)
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RuntimeConfig {
    pub batch_size: u32,
    /// Beam width. The generation phase queries token-major ops at
    /// `x = batch_size * beam_width` (Python `_run_generation_phase`);
    /// attention ops key on the raw decode batch.
    pub beam_width: u32,
    pub isl: u32,
    pub osl: u32,
    /// Cached tokens already in the KV cache (context phase only).
    pub prefix: u32,
    /// Context-attention sequence-imbalance correction (default 1.0).
    pub seq_imbalance_correction_scale: f64,
    /// Generation-attention sequence-imbalance correction (default 1.0).
    pub gen_seq_imbalance_correction_scale: f64,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            batch_size: 1,
            beam_width: 1,
            isl: 1,
            osl: 1,
            prefix: 0,
            seq_imbalance_correction_scale: 1.0,
            gen_seq_imbalance_correction_scale: 1.0,
        }
    }
}

/// Static-inference mode. Mirrors Python's `mode` string in
/// `_run_static_breakdown`: `"static_ctx"` / `"static_gen"` / `"static"`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StaticMode {
    /// Python `mode="static_ctx"`: context (prefill) phase only.
    Context,
    /// Python `mode="static_gen"`: generation (decode) phase only.
    Generation,
    /// Python `mode="static"`: both phases.
    Both,
}

/// Result of [`Engine::run_static`]. Mirrors the latency portion of Python's
/// `run_static_latency_only` (`base_backend.py:322`): per-phase latency plus
/// the total. The latencies are **pre-`latency_correction_scale`** — that param
/// is intentionally dropped from the `run_static(runtime, mode, stride)`
/// signature; it is a flat post-multiply the Python bridge applies downstream.
#[derive(Clone, Debug, PartialEq)]
pub struct StaticResult {
    /// Context-phase latency in ms (0.0 for `StaticMode::Generation`).
    pub context_ms: f64,
    /// Generation-phase latency in ms (0.0 for `StaticMode::Context`).
    pub generation_ms: f64,
    /// `context_ms + generation_ms`. Equals Python `run_static_latency_only`.
    pub total_ms: f64,
}

/// Default decode-quadrature stride. Mirrors Python's `stride=32` default in
/// `run_static` / `_run_generation_phase` (the `DEFAULT_STATIC_STRIDE`).
pub const DEFAULT_STATIC_STRIDE: u32 = 32;

/// Executed MoE communication fallback as it crosses the private FFI:
/// `(inference_phase, comm_backend, requested_ep, requested_nodes,
/// measurement_ep, measurement_nodes)`.
pub(crate) type MoeCommFallbackValue = (&'static str, &'static str, u32, u32, u32, u32);

/// Inline-first fallback metadata for one name-folded op. The first record
/// lives inline; `additional` allocates only when a second distinct record is
/// inserted.
pub(crate) type MoeCommFallbackValues = (MoeCommFallbackValue, Vec<MoeCommFallbackValue>);

/// One evaluated op as it crosses the FFI: `(name, latency_ms, energy_wms,
/// source)`. Entries are NAME-FOLDED before crossing — repeated names
/// accumulate with `+=` and sources merge to `"mixed"` on mismatch, the
/// exact accumulation semantics of Python's phase dicts (addition is
/// commutative, so folding here instead of in Python changes nothing) —
/// because streaming the raw ops × stride-steps tuples through pyo3
/// measurably slowed the engine step on per-block puzzle nets (hundreds of
/// String allocations + Python tuple constructions per call). `source` is
/// the provenance tag (`silicon|empirical|sol|estimated|mixed`).
pub type PerOpValue = (String, f64, f64, &'static str);

/// Internal per-op value used by the provenance-aware engine walk. The fifth
/// field is `None` or `(first_record, additional_records)` in deterministic
/// encounter order; public Rust and Python methods strip it and retain their
/// documented four-tuple contract.
pub(crate) type PerOpValueWithMetadata = (
    String,
    f64,
    f64,
    &'static str,
    Option<MoeCommFallbackValues>,
);

/// Per-op values for the shared, context-attention, and decode-attention
/// buckets returned by the metadata-bearing mixed-step evaluation.
pub(crate) type MixedStepPerOpValuesWithMetadata = (
    Vec<PerOpValueWithMetadata>,
    Vec<PerOpValueWithMetadata>,
    Vec<PerOpValueWithMetadata>,
);

/// One SOL-decomposed per-op value: `(name, sol_time_ms, sol_math_ms,
/// sol_mem_ms)`, mirroring Python's SOL_FULL triple `(sol_time, sol_math,
/// sol_mem)` per query. `sol_time` is the op's SOL-mode latency (scale
/// factors and correction scales applied — for a single leaf query it is
/// exactly the Python triple's `sol_time = max(sol_math, sol_mem)`);
/// `sol_math`/`sol_mem` are the compute-/memory-bound components composed
/// the same way. NAME-FOLDED like [`PerOpValue`] (`+=` on all three).
pub type PerOpSolValue = (String, f64, f64, f64);

/// Name-folding accumulator for [`PerOpValue`] streams. First-encounter
/// order is preserved (mirrors Python dict insertion order). Linear scan on
/// purpose: unique-name counts are a few dozen (per-block families repeat
/// names), far below where a map would win.
struct PerOpFold {
    inference_phase: &'static str,
    entries: Vec<PerOpValueWithMetadata>,
}

fn insert_per_op_fallback(
    fallbacks: &mut Option<MoeCommFallbackValues>,
    fallback: MoeCommFallbackValue,
) {
    match fallbacks {
        None => *fallbacks = Some((fallback, Vec::new())),
        Some((first, additional)) if *first == fallback || additional.contains(&fallback) => {}
        Some((_first, additional)) => additional.push(fallback),
    }
}

fn extend_per_op_fallbacks(
    fallbacks: &mut Option<MoeCommFallbackValues>,
    other: Option<MoeCommFallbackValues>,
) {
    let Some((first, additional)) = other else {
        return;
    };
    insert_per_op_fallback(fallbacks, first);
    for fallback in additional {
        insert_per_op_fallback(fallbacks, fallback);
    }
}

impl PerOpFold {
    fn new(inference_phase: &'static str) -> Self {
        Self {
            inference_phase,
            entries: Vec::new(),
        }
    }

    fn add(&mut self, op: &Op, r: PerformanceResult) {
        let name = op.name();
        let source = r.source.as_str();
        let mut fallbacks = None;
        for fallback in r.moe_comm_fallbacks.iter() {
            insert_per_op_fallback(
                &mut fallbacks,
                (
                    self.inference_phase,
                    fallback.comm_backend,
                    fallback.requested_ep_size,
                    fallback.requested_node_num,
                    fallback.measurement_ep_size,
                    fallback.measurement_node_num,
                ),
            );
        }
        if let Some(entry) = self.entries.iter_mut().find(|e| e.0 == name) {
            entry.1 += r.latency_ms;
            entry.2 += r.energy_wms;
            if entry.3 != source {
                entry.3 = "mixed";
            }
            extend_per_op_fallbacks(&mut entry.4, fallbacks);
            return;
        }
        self.entries.push((
            name.to_string(),
            r.latency_ms,
            r.energy_wms,
            source,
            fallbacks,
        ));
    }

    fn into_values(self) -> Vec<PerOpValueWithMetadata> {
        self.entries
    }
}

fn strip_per_op_metadata(entries: Vec<PerOpValueWithMetadata>) -> Vec<PerOpValue> {
    entries
        .into_iter()
        .map(|(name, latency_ms, energy_wms, source, _fallbacks)| {
            (name, latency_ms, energy_wms, source)
        })
        .collect()
}

/// Name-folding accumulator for [`PerOpSolValue`] streams (fold semantics of
/// [`PerOpFold`]: first-encounter order, `+=` accumulation, linear scan).
#[derive(Default)]
struct PerOpSolFold {
    entries: Vec<PerOpSolValue>,
}

impl PerOpSolFold {
    fn add(&mut self, op: &Op, r: PerformanceResult) -> Result<(), AicError> {
        let (sol_math, sol_mem) = match r.sol {
            Some(c) => (c.math_ms, c.mem_ms),
            // No-op short-circuits (tp_size=1 allreduce, pp_size=1 P2P)
            // return plain zero results without a decomposition: a zero
            // contribution is exact, not a coverage gap.
            None if r.latency_ms == 0.0 && r.energy_wms == 0.0 => (0.0, 0.0),
            None => {
                return Err(AicError::SolNotImplemented(format!(
                    "evaluate_ops_sol_json: op '{}' has no SOL decomposition \
                     (family not exported yet — see PerformanceResult::sol)",
                    op.name()
                )))
            }
        };
        if let Some(entry) = self.entries.iter_mut().find(|e| e.0 == op.name()) {
            entry.1 += r.latency_ms;
            entry.2 += sol_math;
            entry.3 += sol_mem;
            return Ok(());
        }
        self.entries
            .push((op.name().to_string(), r.latency_ms, sol_math, sol_mem));
        Ok(())
    }

    fn into_values(self) -> Vec<PerOpSolValue> {
        self.entries
    }
}

/// Which of the three mixed-step passes produced a sinked per-op value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MixedPass {
    SharedNonAttention,
    ContextAttention,
    DecodeAttention,
}

/// Compiled engine: precompiled op lists + the matching perf database.
///
/// Built from an [`EngineSpec`] (Python's `compile_engine` output) plus a
/// loaded [`PerfDatabase`]. Holds only the scalars the static composition
/// reads: the two op lists and `nextn` (the MTP decode-batch multiplier).
/// Parallelism / quant scalars do not enter the latency sum — they drive
/// throughput and memory, which `StaticResult` omits — so they are not stored.
pub struct Engine {
    /// Context-phase ops in execution order (from `spec.context_ops`).
    context_ops: Vec<Op>,
    /// Generation-phase ops in execution order (from `spec.generation_ops`).
    generation_ops: Vec<Op>,
    /// Loaded perf database. `Arc` so the `AicEngine` can share it with the
    /// capacity API; free fns take `&PerfDatabase`, so deref works either way.
    db: Arc<PerfDatabase>,
    /// MTP speculative-decoding depth. The decode batch is scaled by
    /// `(nextn + 1)` exactly as Python `_run_generation_phase:200`
    /// (`batch_size = batch_size * (model._nextn + 1)`). 0 disables scaling.
    nextn: u32,
}

impl std::fmt::Debug for Engine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Engine")
            .field("context_ops", &self.context_ops.len())
            .field("generation_ops", &self.generation_ops.len())
            .field("nextn", &self.nextn)
            .finish_non_exhaustive()
    }
}

impl Engine {
    /// Build an `Engine` from a spec and a pre-loaded database.
    ///
    /// Extracts the op lists and the `nextn` scalar from `spec.engine`. The
    /// caller (`AicEngineBuilder` / `from_spec_bytes`) is responsible for
    /// having loaded the matching `PerfDatabase` from `spec.engine`'s identity.
    pub fn build(spec: EngineSpec, db: Arc<PerfDatabase>) -> Result<Engine, AicError> {
        let nextn = spec
            .engine
            .speculative
            .as_ref()
            .and_then(|s| s.nextn)
            .unwrap_or(0);
        // FPM whole-model specs must be exactly one op per phase (the Python
        // rewrite guarantees this shape) and never carry MTP: the Python model
        // builder rejects `nextn > 0` for forward_model="fpm" (commit
        // ad93e75f) and the collected data has no speculative points. Guarding
        // here keeps a hand-built or skewed spec from silently mis-composing.
        // The scan is RECURSIVE: an FpmForward nested inside Overlap/Fallback
        // (never produced by the Python rewrite, but expressible in a
        // hand-built spec) would evade a top-level check and ride the
        // name-filtered mix-step passes with the wrong workload shape — and
        // FallbackOp swallows the op's PerfDatabase-class misses silently.
        fn contains_fpm(ops: &[Op]) -> bool {
            ops.iter().any(|op| match op {
                Op::FpmForward(_) => true,
                Op::Overlap(o) => contains_fpm(&o.group_a) || contains_fpm(&o.group_b),
                Op::Fallback(o) => {
                    contains_fpm(std::slice::from_ref(&o.primary)) || contains_fpm(&o.fallback)
                }
                _ => false,
            })
        }
        let any_fpm = contains_fpm(&spec.context_ops) || contains_fpm(&spec.generation_ops);
        if any_fpm {
            let shape_ok = matches!(
                spec.context_ops.as_slice(),
                [Op::FpmForward(p)] if p.phase == FpmPhase::Prefill
            ) && matches!(
                spec.generation_ops.as_slice(),
                [Op::FpmForward(d)] if d.phase == FpmPhase::Decode
            );
            if !shape_ok {
                return Err(AicError::InvalidEngineConfig(
                    "forward_model='fpm' spec must contain exactly one FpmForward op per phase \
                     (prefill in context_ops, decode in generation_ops)"
                        .to_string(),
                ));
            }
            if nextn > 0 {
                return Err(AicError::InvalidEngineConfig(format!(
                    "forward_model='fpm' does not support MTP speculative decoding (nextn={nextn})"
                )));
            }
        }
        Ok(Engine {
            context_ops: spec.context_ops,
            generation_ops: spec.generation_ops,
            db,
            nextn,
        })
    }

    /// FPM whole-model engine: both phase lists are exactly one `FpmForward`
    /// (validated in [`Engine::build`]). Returns `(prefill_op, decode_op)`.
    fn fpm_ops(&self) -> Option<(&FpmForwardOp, &FpmForwardOp)> {
        match (self.context_ops.as_slice(), self.generation_ops.as_slice()) {
            ([Op::FpmForward(p)], [Op::FpmForward(d)]) => Some((p, d)),
            _ => None,
        }
    }

    /// Convenience constructor: deserialize a bincode `EngineSpec` and load the
    /// matching `PerfDatabase` from its identity, then [`Engine::build`].
    ///
    /// Runs the `Engine::from_spec_bytes(bytes) + PerfDatabase::load`
    /// flow. `systems_root` points at `src/aiconfigurator_core/systems` and is used
    /// only as a fallback: when the decoded `spec.engine.systems_path` is
    /// `Some`, that path is authoritative and overrides the `systems_root`
    /// argument.
    pub fn from_spec_bytes(
        bytes: &[u8],
        systems_root: &std::path::Path,
    ) -> Result<Engine, AicError> {
        let spec = EngineSpec::from_bincode(bytes)?;
        let version = spec.engine.backend_version.as_deref().ok_or_else(|| {
            AicError::InvalidEngineConfig(
                "backend_version is required to load the perf database".to_string(),
            )
        })?;
        // The spec's own `systems_path` wins when present; otherwise fall back
        // to the `systems_root` argument.
        let systems_root = spec.engine.systems_path.as_deref().unwrap_or(systems_root);
        let transfer_policy = TransferPolicy::from_wire(spec.engine.transfer_policy.as_deref())
            .map_err(AicError::InvalidEngineConfig)?;
        // The shared variant reuses already-parsed perf tables across engines
        // with the same DB identity: a sweep compiles one engine per
        // model/parallelism/quant point, and without sharing each of those
        // engines would lazily re-parse the same parquet files on its first
        // query (~0.5s per engine on data-rich systems). Mode/policy, memo
        // caches, and the provenance accumulator stay per-engine.
        let db = PerfDatabase::load_resolved_shared(
            systems_root,
            &spec.engine.system_name,
            spec.engine.backend.as_str(),
            version,
            // Shared-layer inheritance: explicit override when the spec
            // carries one (Python's `shared_layer=` kwarg), else derived
            // from the query mode exactly like Python `_shared_layer_enabled`
            // (SILICON/HYBRID = on).
            spec.engine.enable_shared_layer.unwrap_or(matches!(
                spec.engine.database_mode,
                DatabaseMode::Silicon | DatabaseMode::Hybrid
            )),
            spec.engine.strict_provenance,
            // Estimate-only systems (a spec yaml with no collected data) may
            // back a SOL view: every SOL answer is analytic from the system
            // spec, so tolerate a missing perf-data directory under SOL and
            // let table-backed lookups miss lazily. A directory-less
            // fleet-`next` spec (validated by the Python slot resolver, which
            // loaded the same identity through backward fill) also skips the
            // gate — the source resolver serves every table from sibling
            // versions. All other loads keep the loud gate.
            spec.engine.database_mode == DatabaseMode::Sol || spec.engine.tolerate_dirless_version,
        )?
        .with_mode(spec.engine.database_mode, transfer_policy);
        Engine::build(spec, Arc::new(db))
    }

    /// Shared perf database handle.
    pub fn database(&self) -> &Arc<PerfDatabase> {
        &self.db
    }

    /// Clear the empirical-provenance accumulator (start of a run). The PyO3
    /// boundary calls this at the top of every compute method so
    /// [`Self::last_provenance`] carries per-call semantics, mirroring
    /// Python's `capture_provenance()` scope. Deliberately NOT called inside
    /// `run_static` itself: `mixed_step_latency` composes multiple internal
    /// passes whose tiers must accumulate into one answer.
    pub fn reset_provenance(&self) {
        self.db.reset_provenance();
    }

    /// The least-confident empirical tier fired since the last
    /// [`Self::reset_provenance`], as the Python tag string; `None` when the
    /// run was answered purely from silicon tables (nothing to note — Python's
    /// `note_provenance` is skipped for silicon too).
    pub fn last_provenance(&self) -> Option<&'static str> {
        match self.db.worst_provenance() {
            crate::operators::util_empirical::ProvenanceTier::Silicon => None,
            tier => Some(tier.as_str()),
        }
    }

    /// Test-only accessor for the context op list (the field is private, but
    /// `fpm`'s `#[cfg(test)]` parity tests compare `forward_pass_time_ms`
    /// against the shared session free fns over these exact ops).
    #[cfg(test)]
    pub(crate) fn context_ops_for_test(&self) -> &[Op] {
        &self.context_ops
    }

    /// Test-only accessor for the generation op list. See
    /// [`Self::context_ops_for_test`].
    #[cfg(test)]
    pub(crate) fn generation_ops_for_test(&self) -> &[Op] {
        &self.generation_ops
    }

    /// Python `run_static` / `run_static_latency_only` (`base_backend.py:347`,
    /// `:322`) restricted to the latency breakdown. Dispatches on `mode` the
    /// way `_run_static_breakdown` does and sums context + generation.
    pub fn run_static(
        &self,
        runtime: &RuntimeConfig,
        mode: StaticMode,
        stride: u32,
    ) -> Result<StaticResult, AicError> {
        let context_ms = match mode {
            StaticMode::Context | StaticMode::Both => self.run_context_phase(runtime)?,
            StaticMode::Generation => 0.0,
        };
        let generation_ms = match mode {
            StaticMode::Generation | StaticMode::Both => {
                self.run_generation_phase(runtime, stride)?
            }
            StaticMode::Context => 0.0,
        };
        Ok(StaticResult {
            context_ms,
            generation_ms,
            total_ms: context_ms + generation_ms,
        })
    }

    /// Python `_run_context_phase` (`base_backend.py:144`): `effective_isl =
    /// isl - prefix`, validate `> 0`, then one full pass over `context_ops`.
    fn run_context_phase(&self, runtime: &RuntimeConfig) -> Result<f64, AicError> {
        // Python raises `ValueError` when `effective_isl <= 0`; mirror that.
        if runtime.prefix >= runtime.isl {
            return Err(AicError::InvalidEngineConfig(format!(
                "isl must be greater than 0 after removing prefix, but got {}",
                runtime.isl as i64 - runtime.prefix as i64
            )));
        }
        let effective_isl = runtime.isl - runtime.prefix;
        run_context_ops(
            &self.context_ops,
            &self.db,
            runtime.batch_size,
            effective_isl,
            runtime.prefix,
            runtime.seq_imbalance_correction_scale,
            ContextOpFilter::All,
        )
    }

    /// Python `_run_generation_phase` (`base_backend.py:185`): scale the decode
    /// batch by `(nextn + 1)`, then integrate over the decode trajectory with
    /// the stride quadrature.
    ///
    /// ```text
    /// bs = batch_size * (nextn + 1)
    /// for i in range(0, osl - 1, stride):
    ///     step = Σ generation_ops  with  batch_size=bs, s = isl + i + 1
    ///     repeat_count = min(stride, osl - 1 - i)
    ///     generation += step * repeat_count
    /// ```
    ///
    /// `osl <= 1` yields an empty loop and 0.0 (matches Python).
    fn run_generation_phase(&self, runtime: &RuntimeConfig, stride: u32) -> Result<f64, AicError> {
        self.run_generation_phase_with(runtime, stride, |_, _| {})
    }

    /// [`Self::run_generation_phase`] with a per-op sink. Python builds a
    /// per-iteration dict (folding same-name results), THEN multiplies the
    /// folded values by the stride `repeat_count` and merges them into the
    /// trajectory dicts (`base_backend.py:378-405`) — so the sink here
    /// observes ONE per-step-folded result per op name, already weighted by
    /// `repeat_count`, in that exact order: `(r1 + r2) * k`, not
    /// `r1*k + r2*k` (bit-identical for repeated-name model families).
    fn run_generation_phase_with(
        &self,
        runtime: &RuntimeConfig,
        stride: u32,
        mut on_op: impl FnMut(&Op, PerformanceResult),
    ) -> Result<f64, AicError> {
        let bs = runtime
            .batch_size
            .saturating_mul(self.nextn.saturating_add(1));
        let stride = stride.max(1);
        let mut total = 0.0_f64;
        if runtime.osl <= 1 {
            return Ok(0.0);
        }
        let upper = runtime.osl - 1; // exclusive, matches Python `range(0, osl-1, stride)`
        let mut i = 0u32;
        while i < upper {
            // Python `s = isl + i + 1`. NOTE the `+1` — distinct from the FPM
            // bridge's `context_length = isl + i` packing convention.
            let s = runtime.isl + i + 1;
            let repeat_count = stride.min(upper - i);
            // Per-step name fold FIRST (Python's per-iteration dict), with
            // the phase-dict source merge (mismatch -> Mixed, no
            // zero-identity — mirrors `base_backend.py:391-393`).
            let mut step_fold: Vec<(&Op, PerformanceResult)> = Vec::new();
            let step = run_generation_ops_step_beamed_with(
                &self.generation_ops,
                &self.db,
                bs,
                runtime.beam_width,
                s,
                runtime.gen_seq_imbalance_correction_scale,
                false,
                |op, r| {
                    if let Some(entry) = step_fold.iter_mut().find(|(e, _)| e.name() == op.name()) {
                        entry.1.latency_ms += r.latency_ms;
                        entry.1.energy_wms += r.energy_wms;
                        if entry.1.source != r.source {
                            entry.1.source = crate::operators::base::Source::Mixed;
                        }
                        entry.1.moe_comm_fallbacks.extend(r.moe_comm_fallbacks);
                    } else {
                        step_fold.push((op, r));
                    }
                },
            )?;
            for (op, folded) in step_fold {
                on_op(op, folded.scaled(repeat_count as f64));
            }
            total += step * repeat_count as f64;
            i += stride;
        }
        Ok(total)
    }

    /// Mocker H1: prefill-step latency in ms. Pure-Rust inherent method (no
    /// PyO3 `py` token), so the Mocker hot path runs without acquiring the GIL.
    /// Thin shim over [`Self::run_static`] with `mode=Context` (osl is
    /// irrelevant for the context phase, so it is fixed at 1).
    pub fn predict_prefill_latency(&self, bs: u32, isl: u32, prefix: u32) -> Result<f64, AicError> {
        let rt = RuntimeConfig {
            batch_size: bs,
            isl,
            osl: 1,
            prefix,
            ..Default::default()
        };
        Ok(self
            .run_static(&rt, StaticMode::Context, DEFAULT_STATIC_STRIDE)?
            .total_ms)
    }

    /// Mocker H2: decode-step latency in ms. Pure-Rust inherent method (no
    /// PyO3 `py` token). Thin shim over [`Self::run_static`] with
    /// `mode=Generation`. Mocker passes `osl=2` (one decode step at
    /// `s = isl + 1`).
    pub fn predict_decode_latency(&self, bs: u32, isl: u32, osl: u32) -> Result<f64, AicError> {
        let rt = RuntimeConfig {
            batch_size: bs,
            isl,
            osl,
            ..Default::default()
        };
        Ok(self
            .run_static(&rt, StaticMode::Generation, DEFAULT_STATIC_STRIDE)?
            .total_ms)
    }

    /// One mixed (chunked-prefill + decode) step latency. LITERAL mirror of
    /// Python `_get_mix_step_latency` / `run_mixed`, which composes three
    /// filtered phase passes (`_run_context_phase` / `_run_generation_phase`
    /// with `op_filter`) that query ONLY the ops each pass consumes — the
    /// same name-keyed sets the `ContextOpFilter` /
    /// `only_generation_attention` walks below visit (issue #1498
    /// follow-through: Python used to run the full lists and discard, so a
    /// raise in a discarded query was a one-sided error surface):
    ///
    /// ```text
    /// // Pass 1 — combined non-attention work:
    /// //   run_static(batch=1, isl=ctx+gen, osl=1,
    /// //              prefix=prefix*floor(ctx/isl), mode=static_ctx)
    /// //   sum every op EXCEPT "context_attention"
    /// // Pass 2 — context attention at the prefill shape:
    /// //   run_static(batch=ceil(ctx/isl), isl=isl, osl=1, prefix=prefix)
    /// //   take ONLY "context_attention", divide by ceil(isl/ctx)
    /// // Pass 3 — decode attention (only when gen_tokens > 0):
    /// //   run_static(batch=gen, isl=isl+osl//2, osl=2, mode=static_gen)
    /// //   -> one step at s = isl + osl//2 + 1 with the (nextn+1) batch
    /// //   take ONLY "generation_attention"
    /// ```
    ///
    /// Note the Python conventions this deliberately preserves (they differed
    /// from the pre-rewrite FPM packing): pass 1 uses
    /// `ctx + gen * (nextn + 1)` tokens (the speculative-progress model —
    /// every decode request verifies one target plus all drafts in the
    /// combined pass, mirroring Python `run_mixed`'s `decode_query_tokens`),
    /// the cached prefix multiplier is `floor(ctx/isl)` (not ceil), and the
    /// pass-3 kv position carries `_run_generation_phase`'s `+1`.
    ///
    /// The imbalance-correction scales mirror the `RuntimeConfig` fields
    /// Python threads into each pass (`base_backend.py:950-1043`).
    pub fn mixed_step_latency(
        &self,
        ctx_tokens: u32,
        gen_tokens: u32,
        isl: u32,
        osl: u32,
        prefix: u32,
        seq_imbalance_correction_scale: f64,
        gen_seq_imbalance_correction_scale: f64,
    ) -> Result<f64, AicError> {
        Ok(self.mixed_step_breakdown(
            ctx_tokens,
            gen_tokens,
            isl,
            osl,
            prefix,
            seq_imbalance_correction_scale,
            gen_seq_imbalance_correction_scale,
        )?[0])
    }

    /// Return ``[total, shared_non_attention, context_attention,
    /// decode_attention]`` for one mixed engine iteration — the three passes
    /// of the `_get_mix_step_latency` composition reported separately: pass 1
    /// is the shared non-attention work, pass 2 the context-attention slice
    /// (already divided by `ceil(isl/ctx)`), pass 3 the decode-attention
    /// slice. [`Engine::mixed_step_latency`] is their sum; the agg
    /// speculative scheduler consumes the components.
    pub fn mixed_step_breakdown(
        &self,
        ctx_tokens: u32,
        gen_tokens: u32,
        isl: u32,
        osl: u32,
        prefix: u32,
        seq_imbalance_correction_scale: f64,
        gen_seq_imbalance_correction_scale: f64,
    ) -> Result<[f64; 4], AicError> {
        self.mixed_step_breakdown_with(
            ctx_tokens,
            gen_tokens,
            isl,
            osl,
            prefix,
            seq_imbalance_correction_scale,
            gen_seq_imbalance_correction_scale,
            |_, _, _| {},
        )
    }

    /// [`Self::mixed_step_breakdown`] with a per-op sink. The sink observes
    /// `(pass, op, result)` for every queried op with RAW (undivided) pass-2
    /// values; the per-op wrapper applies the `ceil(isl/ctx)` division to the
    /// FOLDED entries (fold-then-divide, matching Python and the scalar
    /// bucket bit-for-bit).
    #[allow(clippy::too_many_arguments)]
    fn mixed_step_breakdown_with(
        &self,
        ctx_tokens: u32,
        gen_tokens: u32,
        isl: u32,
        osl: u32,
        prefix: u32,
        seq_imbalance_correction_scale: f64,
        gen_seq_imbalance_correction_scale: f64,
        mut on_op: impl FnMut(MixedPass, &Op, PerformanceResult),
    ) -> Result<[f64; 4], AicError> {
        if ctx_tokens == 0 && gen_tokens == 0 {
            return Ok([0.0; 4]);
        }
        // Whole-model FPM ops must never reach the name-filtered three-pass
        // composition below (they match neither attention filter and would
        // ride pass 1 with the wrong workload shape). Python branches the
        // same way at `_get_mix_step_latency` -> `_get_fpm_mix_step_latency`.
        // Component mapping: FPM has no non-attention/attention split, so the
        // breakdown reports [total, prefill_component, 0, marginal_decode].
        // The component consumers (speculative agg scheduling) only read the
        // split under MTP, which FPM rejects at build time.
        if let Some((prefill_op, decode_op)) = self.fpm_ops() {
            let (prefill_ms, marginal_decode_ms) = self.fpm_mixed_step_components(
                prefill_op,
                decode_op,
                ctx_tokens,
                gen_tokens,
                isl.max(1),
                osl.max(1),
                prefix,
            )?;
            return Ok([
                prefill_ms + marginal_decode_ms,
                prefill_ms,
                0.0,
                marginal_decode_ms,
            ]);
        }
        // Python divides by `isl` (`floor(ctx/isl)`, `ceil(ctx/isl)`) without
        // a guard — callers always pass isl >= 1. Clamp to avoid a Rust
        // div-by-zero panic on degenerate input Python would crash on.
        let isl = isl.max(1);

        // ---- Pass 1: combined non-attention work ----
        // Speculative progress model: every decode request verifies one
        // target token plus all scheduled drafts, so the combined pass sees
        // `gen * (nextn + 1)` decode tokens (mirrors Python `run_mixed`'s
        // `decode_query_tokens`). Acceptance does not reduce this
        // current-iteration work.
        let decode_query_tokens = gen_tokens.saturating_mul(self.nextn.saturating_add(1));
        let combined = ctx_tokens + decode_query_tokens;
        let prefix1 = prefix * (ctx_tokens / isl); // prefix * floor(ctx/isl)
        if prefix1 >= combined {
            return Err(AicError::InvalidEngineConfig(format!(
                "isl must be greater than 0 after removing prefix, but got {}",
                combined as i64 - prefix1 as i64
            )));
        }
        let shared_non_attention = run_context_ops_with(
            &self.context_ops,
            &self.db,
            1,
            combined - prefix1,
            prefix1,
            seq_imbalance_correction_scale,
            ContextOpFilter::SkipContextAttention,
            |op, r| on_op(MixedPass::SharedNonAttention, op, r),
        )?;

        // ---- Pass 2: context attention at the prefill shape ----
        // Python: batch = ceil(ctx/isl), effective_isl = isl - prefix, then
        // latency["context_attention"] / ceil(isl/ctx). With ctx_tokens == 0
        // Python's `np.ceil(isl/0)` is +inf and the division yields 0 — skip.
        let mut context_attention = 0.0_f64;
        if ctx_tokens > 0 {
            if prefix >= isl {
                return Err(AicError::InvalidEngineConfig(format!(
                    "isl must be greater than 0 after removing prefix, but got {}",
                    isl as i64 - prefix as i64
                )));
            }
            let batch2 = ctx_tokens.div_ceil(isl);
            let scale2 = isl.div_ceil(ctx_tokens) as f64;
            let attn = run_context_ops_with(
                &self.context_ops,
                &self.db,
                batch2,
                isl - prefix,
                prefix,
                seq_imbalance_correction_scale,
                ContextOpFilter::OnlyContextAttention,
                // RAW results to the sink; the per-op wrapper divides the
                // FOLDED values by scale2 with one true division per name
                // (Python folds `context_attention` into one key, then
                // `latency_dict["context_attention"] / scale_factor` —
                // fold-then-divide, `base_backend.py:1244-1246`).
                |op, r| on_op(MixedPass::ContextAttention, op, r),
            )?;
            context_attention = attn / scale2;
        }

        // ---- Pass 3: decode attention ----
        let mut decode_attention = 0.0_f64;
        if gen_tokens > 0 {
            let bs = gen_tokens.saturating_mul(self.nextn.saturating_add(1));
            // `_run_generation_phase` queries at s = isl_pass3 + i + 1 with
            // isl_pass3 = isl + osl//2 and a single step (osl=2, i=0).
            let s = isl + osl / 2 + 1;
            decode_attention = run_generation_ops_step_beamed_with(
                &self.generation_ops,
                &self.db,
                bs,
                1,
                s,
                gen_seq_imbalance_correction_scale,
                true,
                |op, r| on_op(MixedPass::DecodeAttention, op, r),
            )?;
        }

        Ok([
            shared_non_attention + context_attention + decode_attention,
            shared_non_attention,
            context_attention,
            decode_attention,
        ])
    }

    /// One generation-only step latency. LITERAL mirror of Python
    /// `_get_genonly_step_latency` (`base_backend.py:1040-1100`):
    /// `run_static(batch=gen_tokens, isl=isl+osl//2, osl=2, mode=static_gen)`
    /// summed over the FULL generation op list — one step at
    /// `s = isl + osl//2 + 1` (note `_run_generation_phase`'s `+1`) with the
    /// decode batch scaled by `(nextn + 1)`.
    pub fn decode_step_latency(
        &self,
        gen_tokens: u32,
        isl: u32,
        osl: u32,
        gen_seq_imbalance_correction_scale: f64,
    ) -> Result<f64, AicError> {
        if gen_tokens == 0 {
            return Ok(0.0);
        }
        // FPM keeps the PYTHON static-path convention `s = isl + osl/2 + 1`
        // (via `run_generation_phase`), not this method's op-level
        // `isl + osl/2` packing — a documented divergence the FPM port must
        // not inherit (its parity target is the Python FPM branch, which
        // routes through `run_static(mode="static_gen")`).
        if self.fpm_ops().is_some() {
            let rt = RuntimeConfig {
                batch_size: gen_tokens,
                isl: isl.saturating_add(osl / 2),
                osl: 2,
                ..Default::default()
            };
            return self.run_generation_phase(&rt, DEFAULT_STATIC_STRIDE);
        }
        let effective_batch = gen_tokens.saturating_mul(self.nextn.saturating_add(1));
        let s = isl.max(1).saturating_add(osl.max(1) / 2).saturating_add(1);
        run_generation_ops_step(
            &self.generation_ops,
            &self.db,
            effective_batch,
            s,
            gen_seq_imbalance_correction_scale,
            false,
        )
    }

    /// Mixed-step composition, mirroring Python
    /// `_get_fpm_mix_step_latency` exactly: the prefill component prices the
    /// iteration's REAL scheduled totals (chunk + decode tokens — the count
    /// the engine picks its CUDA-graph/eager regime and GEMM width from) via
    /// `query_totals`; chunked requests are priced per chunk at their own
    /// `(chunk + gen, past_kv)` coordinates and averaged. The decode
    /// component stays the pass-baseline marginal. Correct only when the
    /// deployed engine configuration (especially the CUDA-graph capture
    /// surface) matches the collection — the cliffs live in the data.
    fn fpm_mixed_step_components(
        &self,
        prefill_op: &FpmForwardOp,
        decode_op: &FpmForwardOp,
        ctx_tokens: u32,
        gen_tokens: u32,
        isl: u32,
        osl: u32,
        prefix: u32,
    ) -> Result<(f64, f64), AicError> {
        let mut prefill_component = 0.0_f64;
        if ctx_tokens > 0 {
            let new_tokens = isl.saturating_sub(prefix);
            if new_tokens == 0 {
                return Err(AicError::PerfDatabase(format!(
                    "isl must be greater than prefix, got isl={isl} prefix={prefix}"
                )));
            }
            if ctx_tokens >= new_tokens {
                // Whole prefills this iteration: the scheduled total picks
                // the regime row.
                let batch = ctx_tokens.div_ceil(new_tokens);
                prefill_component = prefill_op
                    .query_totals(
                        &self.db,
                        &[
                            batch as f64,
                            (ctx_tokens + gen_tokens) as f64,
                            (batch * prefix) as f64,
                        ],
                    )?
                    .latency_ms;
            } else {
                // Chunked prefill: per-chunk totals, per-iteration average.
                let mut total = 0.0_f64;
                let mut chunks = 0u32;
                let mut done = 0u32;
                while done < new_tokens {
                    let chunk = ctx_tokens.min(new_tokens - done);
                    total += prefill_op
                        .query_totals(
                            &self.db,
                            &[1.0, (chunk + gen_tokens) as f64, (prefix + done) as f64],
                        )?
                        .latency_ms;
                    done += chunk;
                    chunks += 1;
                }
                prefill_component = total / chunks as f64;
            }
        }
        let mut marginal_decode = 0.0_f64;
        if gen_tokens > 0 {
            let rt = RuntimeConfig {
                batch_size: gen_tokens,
                isl: isl.saturating_add(osl / 2),
                osl: 2,
                ..Default::default()
            };
            let gen_ms = self.run_generation_phase(&rt, DEFAULT_STATIC_STRIDE)?;
            let baseline_ms = if ctx_tokens > 0 {
                // run_generation_phase scaled the batch by (nextn + 1); the
                // baseline must be sampled at the same effective batch.
                decode_op
                    .query_pass_baseline(
                        &self.db,
                        gen_tokens.saturating_mul(self.nextn.saturating_add(1)),
                    )?
                    .latency_ms
            } else {
                0.0
            };
            marginal_decode = (gen_ms - baseline_ms).max(0.0);
        }
        Ok((prefill_component, marginal_decode))
    }

    /// [`Self::run_static`] with the per-op values kept instead of summed:
    /// `(context, generation)` lists of `(name, latency_ms, energy_wms,
    /// source)`, NAME-FOLDED (see [`PerOpValue`]): each name crosses once,
    /// pre-accumulated with Python's phase-dict semantics. Generation values
    /// are per-step-folded, then weighted by the stride `repeat_count`.
    pub fn run_static_per_op(
        &self,
        runtime: &RuntimeConfig,
        mode: StaticMode,
        stride: u32,
    ) -> Result<(Vec<PerOpValue>, Vec<PerOpValue>), AicError> {
        let (context, generation) = self.run_static_per_op_impl(runtime, mode, stride)?;
        Ok((
            strip_per_op_metadata(context),
            strip_per_op_metadata(generation),
        ))
    }

    /// Metadata-bearing counterpart used only by the private PyO3 provenance
    /// endpoint. Evaluation stays in this single implementation so the value
    /// and its fallback records always come from the same query.
    pub(crate) fn run_static_per_op_with_metadata(
        &self,
        runtime: &RuntimeConfig,
        mode: StaticMode,
        stride: u32,
    ) -> Result<(Vec<PerOpValueWithMetadata>, Vec<PerOpValueWithMetadata>), AicError> {
        self.run_static_per_op_impl(runtime, mode, stride)
    }

    fn run_static_per_op_impl(
        &self,
        runtime: &RuntimeConfig,
        mode: StaticMode,
        stride: u32,
    ) -> Result<(Vec<PerOpValueWithMetadata>, Vec<PerOpValueWithMetadata>), AicError> {
        let mut context = PerOpFold::new("context");
        if matches!(mode, StaticMode::Context | StaticMode::Both) {
            if runtime.prefix >= runtime.isl {
                return Err(AicError::InvalidEngineConfig(format!(
                    "isl must be greater than 0 after removing prefix, but got {}",
                    runtime.isl as i64 - runtime.prefix as i64
                )));
            }
            run_context_ops_with(
                &self.context_ops,
                &self.db,
                runtime.batch_size,
                runtime.isl - runtime.prefix,
                runtime.prefix,
                runtime.seq_imbalance_correction_scale,
                ContextOpFilter::All,
                |op, r| context.add(op, r),
            )?;
        }
        let mut generation = PerOpFold::new("generation");
        if matches!(mode, StaticMode::Generation | StaticMode::Both) {
            self.run_generation_phase_with(runtime, stride, |op, r| generation.add(op, r))?;
        }
        Ok((context.into_values(), generation.into_values()))
    }

    /// [`Self::mixed_step_breakdown`] with the per-op values kept:
    /// `(shared_non_attention, context_attention, decode_attention)` lists of
    /// `(name, latency_ms, energy_wms, source)`. Context-attention entries
    /// arrive already divided by the `ceil(isl/ctx)` scale.
    #[allow(clippy::too_many_arguments)]
    pub fn mixed_step_breakdown_per_op(
        &self,
        ctx_tokens: u32,
        gen_tokens: u32,
        isl: u32,
        osl: u32,
        prefix: u32,
        seq_imbalance_correction_scale: f64,
        gen_seq_imbalance_correction_scale: f64,
    ) -> Result<(Vec<PerOpValue>, Vec<PerOpValue>, Vec<PerOpValue>), AicError> {
        let (shared, context_attention, decode_attention) = self.mixed_step_breakdown_per_op_impl(
            ctx_tokens,
            gen_tokens,
            isl,
            osl,
            prefix,
            seq_imbalance_correction_scale,
            gen_seq_imbalance_correction_scale,
        )?;
        Ok((
            strip_per_op_metadata(shared),
            strip_per_op_metadata(context_attention),
            strip_per_op_metadata(decode_attention),
        ))
    }

    /// Metadata-bearing counterpart used only by the private PyO3 provenance
    /// endpoint. See [`Self::mixed_step_breakdown_per_op`].
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn mixed_step_breakdown_per_op_with_metadata(
        &self,
        ctx_tokens: u32,
        gen_tokens: u32,
        isl: u32,
        osl: u32,
        prefix: u32,
        seq_imbalance_correction_scale: f64,
        gen_seq_imbalance_correction_scale: f64,
    ) -> Result<MixedStepPerOpValuesWithMetadata, AicError> {
        self.mixed_step_breakdown_per_op_impl(
            ctx_tokens,
            gen_tokens,
            isl,
            osl,
            prefix,
            seq_imbalance_correction_scale,
            gen_seq_imbalance_correction_scale,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn mixed_step_breakdown_per_op_impl(
        &self,
        ctx_tokens: u32,
        gen_tokens: u32,
        isl: u32,
        osl: u32,
        prefix: u32,
        seq_imbalance_correction_scale: f64,
        gen_seq_imbalance_correction_scale: f64,
    ) -> Result<MixedStepPerOpValuesWithMetadata, AicError> {
        // Whole-model FPM: never the name-filtered three-pass split (see
        // mixed_step_breakdown_with). Report the scalar path's component
        // mapping as per-op entries — the prefill component under the
        // prefill op's name in the shared bucket, the decode marginal under
        // the decode op's name — so the Python fold sees the same keys as
        // its own FPM branch.
        if let Some((prefill_op, decode_op)) = self.fpm_ops() {
            let (prefill_ms, marginal_decode_ms) = self.fpm_mixed_step_components(
                prefill_op,
                decode_op,
                ctx_tokens,
                gen_tokens,
                isl.max(1),
                osl.max(1),
                prefix,
            )?;
            let mut shared: Vec<PerOpValueWithMetadata> = Vec::new();
            if ctx_tokens > 0 {
                shared.push((prefill_op.name.clone(), prefill_ms, 0.0, "silicon", None));
            }
            let mut dec_attn: Vec<PerOpValueWithMetadata> = Vec::new();
            if gen_tokens > 0 {
                dec_attn.push((
                    decode_op.name.clone(),
                    marginal_decode_ms,
                    0.0,
                    "silicon",
                    None,
                ));
            }
            return Ok((shared, Vec::new(), dec_attn));
        }
        let mut shared = PerOpFold::new("context");
        let mut ctx_attn = PerOpFold::new("context");
        let mut dec_attn = PerOpFold::new("generation");
        self.mixed_step_breakdown_with(
            ctx_tokens,
            gen_tokens,
            isl,
            osl,
            prefix,
            seq_imbalance_correction_scale,
            gen_seq_imbalance_correction_scale,
            |pass, op, r| {
                let out = match pass {
                    MixedPass::SharedNonAttention => &mut shared,
                    MixedPass::ContextAttention => &mut ctx_attn,
                    MixedPass::DecodeAttention => &mut dec_attn,
                };
                out.add(op, r);
            },
        )?;
        let mut ctx_attn = ctx_attn.into_values();
        if ctx_tokens > 0 {
            // Mirror the scalar bucket and Python's fold-then-single-true-
            // division (`base_backend.py:1244-1246`): one `/ scale2` per
            // folded name, never a per-entry reciprocal multiply.
            let scale2 = isl.max(1).div_ceil(ctx_tokens) as f64;
            for entry in &mut ctx_attn {
                entry.1 /= scale2;
                entry.2 /= scale2;
            }
        }
        Ok((shared.into_values(), ctx_attn, dec_attn.into_values()))
    }

    /// [`Self::decode_step_latency`] with the per-op values kept.
    pub fn decode_step_per_op(
        &self,
        gen_tokens: u32,
        isl: u32,
        osl: u32,
        gen_seq_imbalance_correction_scale: f64,
    ) -> Result<Vec<PerOpValue>, AicError> {
        self.decode_step_per_op_impl(gen_tokens, isl, osl, gen_seq_imbalance_correction_scale)
            .map(strip_per_op_metadata)
    }

    /// Metadata-bearing counterpart used only by the private PyO3 provenance
    /// endpoint. See [`Self::decode_step_per_op`].
    pub(crate) fn decode_step_per_op_with_metadata(
        &self,
        gen_tokens: u32,
        isl: u32,
        osl: u32,
        gen_seq_imbalance_correction_scale: f64,
    ) -> Result<Vec<PerOpValueWithMetadata>, AicError> {
        self.decode_step_per_op_impl(gen_tokens, isl, osl, gen_seq_imbalance_correction_scale)
    }

    fn decode_step_per_op_impl(
        &self,
        gen_tokens: u32,
        isl: u32,
        osl: u32,
        gen_seq_imbalance_correction_scale: f64,
    ) -> Result<Vec<PerOpValueWithMetadata>, AicError> {
        let mut out = PerOpFold::new("generation");
        if gen_tokens == 0 {
            return Ok(out.into_values());
        }
        let effective_batch = gen_tokens.saturating_mul(self.nextn.saturating_add(1));
        let s = isl.max(1).saturating_add(osl.max(1) / 2).saturating_add(1);
        run_generation_ops_step_beamed_with(
            &self.generation_ops,
            &self.db,
            effective_batch,
            1,
            s,
            gen_seq_imbalance_correction_scale,
            false,
            |op, r| out.add(op, r),
        )?;
        Ok(out.into_values())
    }

    /// Evaluate an index-addressed sublist of the compiled CONTEXT op list at
    /// the context-phase shape (the thin op-list evaluation FFI — Python-side
    /// orchestration like AFD partitions the compiled list and sources per-op
    /// values here instead of walking `Operation.query()`).
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_context_ops(
        &self,
        indices: &[usize],
        batch_size: u32,
        s: u32,
        prefix: u32,
        seq_imbalance_correction_scale: f64,
        x_override: Option<u32>,
    ) -> Result<Vec<PerOpValue>, AicError> {
        let mut out = PerOpFold::new("context");
        for &i in indices {
            let op = self.context_ops.get(i).ok_or_else(|| {
                AicError::InvalidEngineConfig(format!(
                    "evaluate_context_ops: index {i} out of range ({} context ops)",
                    self.context_ops.len()
                ))
            })?;
            let r = query_context_op(
                op,
                &self.db,
                batch_size,
                s,
                prefix,
                seq_imbalance_correction_scale,
                x_override,
            )?;
            out.add(op, r);
        }
        Ok(strip_per_op_metadata(out.into_values()))
    }

    /// Evaluate an index-addressed sublist of the compiled GENERATION op list
    /// at the decode-step shape (see [`Self::evaluate_context_ops`]).
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_generation_ops(
        &self,
        indices: &[usize],
        batch_size: u32,
        s: u32,
        gen_seq_imbalance_correction_scale: f64,
        prefix: u32,
        x_override: Option<u32>,
    ) -> Result<Vec<PerOpValue>, AicError> {
        let mut out = PerOpFold::new("generation");
        for &i in indices {
            let op = self.generation_ops.get(i).ok_or_else(|| {
                AicError::InvalidEngineConfig(format!(
                    "evaluate_generation_ops: index {i} out of range ({} generation ops)",
                    self.generation_ops.len()
                ))
            })?;
            let r = query_generation_op(
                op,
                &self.db,
                batch_size,
                1,
                s,
                gen_seq_imbalance_correction_scale,
                prefix,
                x_override,
            )?;
            out.add(op, r);
        }
        Ok(strip_per_op_metadata(out.into_values()))
    }

    /// Evaluate an ad-hoc op list (a JSON array of `OpSpec` objects, the same
    /// externally-tagged encoding `EngineSpec` uses) against this engine's
    /// database. Serves op lists that are deliberately NOT in the compiled
    /// spec — the VL encoder phase — while the shape math stays Python-side.
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_ops_json(
        &self,
        ops_json: &str,
        is_context: bool,
        batch_size: u32,
        s: u32,
        prefix: u32,
        imbalance_correction_scale: f64,
        x_override: Option<u32>,
    ) -> Result<Vec<PerOpValue>, AicError> {
        let ops: Vec<Op> = serde_json::from_str(ops_json).map_err(|e| {
            AicError::InvalidEngineConfig(format!("evaluate_ops_json: invalid op list JSON: {e}"))
        })?;
        let mut out = PerOpFold::new(if is_context { "context" } else { "generation" });
        for op in &ops {
            let r = if is_context {
                query_context_op(
                    op,
                    &self.db,
                    batch_size,
                    s,
                    prefix,
                    imbalance_correction_scale,
                    x_override,
                )?
            } else {
                query_generation_op(
                    op,
                    &self.db,
                    batch_size,
                    1,
                    s,
                    imbalance_correction_scale,
                    prefix,
                    x_override,
                )?
            };
            out.add(op, r);
        }
        Ok(strip_per_op_metadata(out.into_values()))
    }

    /// [`Self::evaluate_ops_json`] under the SOL_FULL view: evaluate an
    /// ad-hoc op list (JSON array of `OpSpec` objects) with every operator
    /// forced onto its analytic SOL branch, and keep the roofline
    /// decomposition. Returns `(name, sol_time_ms, sol_math_ms, sol_mem_ms)`
    /// per op (see [`PerOpSolValue`]) — the compiled-engine replacement for
    /// Python's per-call `query_*(..., database_mode=SOL_FULL)` triples.
    /// Errors when an op's family does not export its decomposition yet.
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_ops_sol_json(
        &self,
        ops_json: &str,
        is_context: bool,
        batch_size: u32,
        s: u32,
        prefix: u32,
        imbalance_correction_scale: f64,
        x_override: Option<u32>,
    ) -> Result<Vec<PerOpSolValue>, AicError> {
        let ops: Vec<Op> = serde_json::from_str(ops_json).map_err(|e| {
            AicError::InvalidEngineConfig(format!(
                "evaluate_ops_sol_json: invalid op list JSON: {e}"
            ))
        })?;
        let sol_db = self.db.sol_full_view();
        let mut out = PerOpSolFold::default();
        for op in &ops {
            let r = if is_context {
                query_context_op(
                    op,
                    &sol_db,
                    batch_size,
                    s,
                    prefix,
                    imbalance_correction_scale,
                    x_override,
                )?
            } else {
                query_generation_op(
                    op,
                    &sol_db,
                    batch_size,
                    1,
                    s,
                    imbalance_correction_scale,
                    prefix,
                    x_override,
                )?
            };
            out.add(op, r)?;
        }
        Ok(out.into_values())
    }

    /// Compute one forward-pass latency from a list of per-rank FPM entries.
    ///
    /// Re-platformed from the (deleted) `SessionEstimator::forward_pass_time_ms`
    /// (commit 520dcfff `session.rs:289`): validate every rank, dispatch each
    /// rank on its scheduled workload via [`Self::rank_latency_ms`], and take the
    /// max across ranks (attention-DP ranks run in lockstep, so the slowest rank
    /// gates the iteration).
    ///
    /// Unlike [`Self::mixed_step_latency`] / [`Self::decode_step_latency`], this
    /// consumes ALREADY-PACKED telemetry: the FPM fields are the observed
    /// per-iteration counts, so the `(nextn + 1)` MTP multiplier is NOT applied
    /// here (it is already baked into the scheduled-decode counts the engine
    /// emitted). The dispatch reuses the shared [`run_context_ops`] /
    /// [`run_generation_ops_step`] / [`get_mix_step_ops`] free fns so this path
    /// and the live engine-step path stay numerically identical.
    pub fn forward_pass_time_ms(
        &self,
        metrics_by_rank: &[ForwardPassMetrics],
    ) -> Result<f64, AicError> {
        if metrics_by_rank.is_empty() {
            return Err(AicError::InvalidForwardPassMetrics(
                "at least one attention-DP rank metric required".to_string(),
            ));
        }
        for metrics in metrics_by_rank {
            validate_forward_pass_metrics(metrics)?;
        }
        let mut max_latency = 0.0_f64;
        for metrics in metrics_by_rank {
            let rank_latency = self.rank_latency_ms(metrics)?;
            if rank_latency > max_latency {
                max_latency = rank_latency;
            }
        }
        Ok(max_latency)
    }

    /// Dispatch one rank's FPM on its scheduled workload. Literal port of
    /// `SessionEstimator::rank_latency_ms` (520dcfff `session.rs:308`):
    /// prefill+decode -> mix step ([`get_mix_step_ops`]); prefill-only ->
    /// [`run_context_ops`]; decode-only -> [`run_generation_ops_step`]. The FPM
    /// counts pass through unscaled (no `nextn` multiplier — see
    /// [`Self::forward_pass_time_ms`]).
    fn rank_latency_ms(&self, metrics: &ForwardPassMetrics) -> Result<f64, AicError> {
        let sched = &metrics.scheduled_requests;
        // Token-based dispatch, aligned with `IterationFeatures` (fpm/model.rs):
        // a fully prefix-cached payload can retain prefill request/KV metadata
        // (`num_prefill_requests = 1, sum_prefill_tokens = 0`) while scheduling
        // no fresh prefill compute — that iteration is decode-only. A count
        // check would query prefill at zero tokens (outside the FPM domain)
        // and price decode as marginal work riding a pass that does not exist.
        let has_prefill = sched.sum_prefill_tokens > 0;
        let has_decode = sched.num_decode_requests > 0 || sched.sum_decode_kv_tokens > 0;

        // FPM engines never enter the three-pass mix composition (its op-name
        // filters cannot see a whole-model op). Prefill-only and decode-only
        // dispatch through the same shared free fns as op-level (the FpmForward
        // op consumes batch/s/prefix from the RuntimeContext naturally); a
        // mixed rank composes prefill + marginal decode, mirroring
        // `_get_fpm_mix_step_latency` at the telemetry counts (already packed,
        // so no `(nextn + 1)` anywhere — and FPM engines enforce nextn == 0).
        if let Some((prefill_op, decode_op)) = self.fpm_ops() {
            // The telemetry sums ARE the fpm_forward tables' native coordinate
            // system (per-rank iteration totals) — query them via
            // `query_totals` instead of the op-level per-request-average
            // convention, which loses up to (n - 1) tokens to integer
            // division on each axis.
            let mut total = 0.0_f64;
            if has_prefill {
                total += prefill_op
                    .query_totals(
                        &self.db,
                        &[
                            sched.num_prefill_requests as f64,
                            sched.sum_prefill_tokens as f64,
                            sched.sum_prefill_kv_tokens as f64,
                        ],
                    )?
                    .latency_ms;
            }
            if has_decode {
                let decode_ms = decode_op
                    .query_totals(
                        &self.db,
                        &[
                            sched.num_decode_requests as f64,
                            sched.sum_decode_kv_tokens as f64,
                        ],
                    )?
                    .latency_ms;
                if has_prefill {
                    // Mixed rank: marginal-decode composition, mirroring
                    // `_get_fpm_mix_step_latency` (counts already packed, no
                    // `(nextn + 1)` — FPM engines enforce nextn == 0).
                    let baseline_ms = decode_op
                        .query_pass_baseline(&self.db, sched.num_decode_requests)?
                        .latency_ms;
                    total += (decode_ms - baseline_ms).max(0.0);
                } else {
                    total += decode_ms;
                }
            }
            return Ok(total);
        }

        if has_prefill && has_decode {
            // Mix step (continuous batching): compose like Python's
            // `_get_mix_step_latency`. `sum_prefill_kv_tokens` is exactly the
            // combined-prefix value the pass-1 non-attention call needs; pass
            // it through unchanged.
            let n_prefill = sched.num_prefill_requests.max(1);
            let new_tokens_per_req = sched.sum_prefill_tokens / n_prefill;
            let prefix_per_req = sched.sum_prefill_kv_tokens / n_prefill;
            let n_decode = sched.num_decode_requests.max(1);
            let kv_per_req = sched.sum_decode_kv_tokens / n_decode;
            let ctx_tokens = sched.sum_prefill_tokens;
            let gen_tokens = sched.num_decode_requests;
            return get_mix_step_ops(
                &self.context_ops,
                &self.generation_ops,
                &self.db,
                ctx_tokens,
                gen_tokens,
                new_tokens_per_req.max(1),
                prefix_per_req,
                sched.sum_prefill_kv_tokens,
                kv_per_req,
                n_decode,
            );
        }

        let mut total = 0.0_f64;

        if has_prefill {
            let n_prefill = sched.num_prefill_requests.max(1);
            let new_tokens_per_req = sched.sum_prefill_tokens / n_prefill;
            let prefix_per_req = sched.sum_prefill_kv_tokens / n_prefill;
            total += run_context_ops(
                &self.context_ops,
                &self.db,
                n_prefill,
                new_tokens_per_req,
                prefix_per_req,
                1.0,
                ContextOpFilter::All,
            )?;
        }

        if has_decode {
            let n_decode = sched.num_decode_requests.max(1);
            let kv_per_req = sched.sum_decode_kv_tokens / n_decode;
            total += run_generation_ops_step(
                &self.generation_ops,
                &self.db,
                n_decode,
                kv_per_req,
                1.0,
                false,
            )?;
        }

        Ok(total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    use crate::common::enums::{FmhaQuantMode, GemmQuantMode, KvCacheQuantMode};
    use crate::engine::spec::EngineSpec;
    use crate::operators::op::Op;
    use crate::operators::{
        ContextAttentionOp, ElementwiseOp, GemmOp, GenerationAttentionOp, MoeAllToAllOp,
    };
    use crate::{BackendKind, EngineConfig, ParallelMapping, QuantizationConfig};

    fn systems_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../src/aiconfigurator_core/systems")
    }

    const TEST_MODEL: &str = "MiniMaxAI/MiniMax-M2.5";

    /// Hand-built context op list against the b200_sxm/vllm/0.24.0 perf tables.
    /// `Elementwise` is DB-free (pure mem-bandwidth SOL); `Gemm` and
    /// `ContextAttention` hit existing perf tables. The (deleted) model layer
    /// previously sourced these lists from the HF config.
    fn context_ops() -> Vec<Op> {
        vec![
            Op::Elementwise(ElementwiseOp {
                name: "rmsnorm".into(),
                scale_factor: 1.0,
                bytes_per_token: 8192.0,
                scale_num_tokens: 1,
                seq_split: 1,
            }),
            Op::Gemm(GemmOp {
                name: "qkv_gemm".into(),
                scale_factor: 1.0,
                n: 4096,
                k: 4096,
                quant_mode: GemmQuantMode::Fp8Block,
                scale_num_tokens: 0,
                low_precision_input: false,
                seq_split: 1,
                below_grid_sol: false,
            }),
            Op::ContextAttention(ContextAttentionOp {
                name: "context_attention".into(),
                scale_factor: 1.0,
                n: 32,
                n_kv: 8,
                head_size: 128,
                window_size: 0,
                kv_cache_dtype: KvCacheQuantMode::Fp8,
                fmha_quant_mode: FmhaQuantMode::Bfloat16,
                use_qk_norm: false,
                cp_size: 1,
                lane_order: crate::operators::attention::b200_vllm_context_lane_order(),
            }),
        ]
    }

    fn generation_ops() -> Vec<Op> {
        vec![
            Op::Elementwise(ElementwiseOp {
                name: "rmsnorm".into(),
                scale_factor: 1.0,
                bytes_per_token: 8192.0,
                scale_num_tokens: 1,
                seq_split: 1,
            }),
            Op::GenerationAttention(GenerationAttentionOp {
                name: "generation_attention".into(),
                scale_factor: 1.0,
                n: 32,
                n_kv: 8,
                head_size: 128,
                window_size: 0,
                kv_cache_dtype: KvCacheQuantMode::Fp8,
                lane_order: crate::operators::attention::b200_vllm_generation_lane_order(),
            }),
        ]
    }

    fn fixture_engine_config(nextn: Option<u32>) -> EngineConfig {
        EngineConfig {
            schema_version: crate::ENGINE_CONFIG_SCHEMA_VERSION,
            model_name: TEST_MODEL.to_string(),
            system_name: "b200_sxm".to_string(),
            systems_path: None,
            backend: BackendKind::Vllm,
            backend_version: Some("0.24.0".to_string()),
            forward_model: None,
            kv_block_size: None,
            parallel: ParallelMapping {
                tp_size: 8,
                pp_size: 1,
                attention_dp_size: Some(1),
                moe_tp_size: Some(1),
                moe_ep_size: Some(8),
                cp_size: None,
            },
            quantization: QuantizationConfig {
                weight_dtype: None,
                moe_dtype: None,
                activation_dtype: None,
                kv_cache_dtype: None,
            },
            speculative: nextn.map(|n| crate::SpeculativeConfig { nextn: Some(n) }),
            enable_shared_layer: None,
            strict_provenance: false,
            tolerate_dirless_version: false,
            database_mode: Default::default(),
            transfer_policy: None,
            extra: BTreeMap::new(),
        }
    }

    /// Build an `Engine` from the hand-built op lists over the real fixture DB.
    fn build_engine(nextn: Option<u32>) -> Engine {
        let db = PerfDatabase::load(&systems_root(), "b200_sxm", "vllm", "0.24.0").unwrap();
        let spec = EngineSpec::new(
            fixture_engine_config(nextn),
            context_ops(),
            generation_ops(),
        );
        Engine::build(spec, Arc::new(db)).unwrap()
    }

    fn runtime(batch_size: u32, isl: u32, osl: u32) -> RuntimeConfig {
        RuntimeConfig {
            batch_size,
            isl,
            osl,
            ..Default::default()
        }
    }

    #[test]
    fn per_op_fold_attaches_the_inference_phase_only_to_executed_fallbacks() {
        use crate::operators::base::{MoeCommFallback, Source};

        let op = context_ops().remove(0);
        let fallback = MoeCommFallback {
            comm_backend: "deepep_ht",
            requested_ep_size: 32,
            requested_node_num: 8,
            measurement_ep_size: 8,
            measurement_node_num: 1,
        };
        for inference_phase in ["context", "generation"] {
            let mut fold = PerOpFold::new(inference_phase);
            fold.add(
                &op,
                PerformanceResult::new(1.0, Source::Estimated).with_moe_comm_fallback(fallback),
            );
            assert_eq!(
                fold.into_values()[0].4,
                Some(((inference_phase, "deepep_ht", 32, 8, 8, 1), vec![]))
            );
        }

        let mut repeated_name = PerOpFold::new("context");
        repeated_name.add(
            &op,
            PerformanceResult::new(1.0, Source::Estimated).with_moe_comm_fallback(fallback),
        );
        repeated_name.add(
            &op,
            PerformanceResult::new(1.0, Source::Estimated).with_moe_comm_fallback(
                MoeCommFallback {
                    comm_backend: "deepep_ll",
                    ..fallback
                },
            ),
        );
        assert_eq!(
            repeated_name.into_values()[0].4,
            Some((
                ("context", "deepep_ht", 32, 8, 8, 1),
                vec![("context", "deepep_ll", 32, 8, 8, 1)],
            ))
        );

        let mut exact = PerOpFold::new("context");
        exact.add(&op, PerformanceResult::new(1.0, Source::Silicon));
        assert_eq!(exact.into_values()[0].4, None);
    }

    #[test]
    fn per_op_fold_allocates_additional_storage_only_for_distinct_fallbacks_after_the_first() {
        use crate::operators::base::{MoeCommFallback, Source};

        let op = context_ops().remove(0);
        let ht = MoeCommFallback {
            comm_backend: "deepep_ht",
            requested_ep_size: 32,
            requested_node_num: 8,
            measurement_ep_size: 8,
            measurement_node_num: 1,
        };
        let ll = MoeCommFallback {
            comm_backend: "deepep_ll",
            ..ht
        };

        let mut empty = PerOpFold::new("context");
        empty.add(&op, PerformanceResult::new(1.0, Source::Silicon));
        assert!(empty.into_values().pop().unwrap().4.is_none());

        let mut single = PerOpFold::new("context");
        single.add(
            &op,
            PerformanceResult::new(1.0, Source::Estimated).with_moe_comm_fallback(ht),
        );
        let (first, additional) = single.into_values().pop().unwrap().4.unwrap();
        assert_eq!(first, ("context", "deepep_ht", 32, 8, 8, 1));
        assert_eq!(additional.capacity(), 0);

        let mut multiple = PerOpFold::new("generation");
        for fallback in [ht, ht, ll, ll] {
            multiple.add(
                &op,
                PerformanceResult::new(1.0, Source::Estimated).with_moe_comm_fallback(fallback),
            );
        }
        let (first, additional) = multiple.into_values().pop().unwrap().4.unwrap();
        assert_eq!(first, ("generation", "deepep_ht", 32, 8, 8, 1));
        assert_eq!(additional, vec![("generation", "deepep_ll", 32, 8, 8, 1)]);
    }

    #[test]
    fn generation_step_preserves_distinct_same_name_deepep_fallbacks() {
        let mut config = fixture_engine_config(None);
        config.system_name = "gb200".to_string();
        config.backend = BackendKind::Sglang;
        config.backend_version = Some("0.5.16".to_string());

        let a2a = |moe_ep_size, node_num| {
            Op::MoeAllToAll(MoeAllToAllOp {
                name: "generation_moe_dispatch".to_string(),
                scale_factor: 1.0,
                phase: "dispatch".to_string(),
                comm_backend: "deepep_ll".to_string(),
                comm_dtype: "default".to_string(),
                hidden_size: 7168,
                topk: 8,
                num_experts: 256,
                moe_ep_size,
                node_num,
                sms: 0,
                attention_tp_size: 1,
                workload_distribution: "power_law_1.2".into(),
                enable_eplb: false,
            })
        };
        let spec = EngineSpec::new(config, Vec::new(), vec![a2a(32, 8), a2a(64, 16)]);
        let engine = Engine::from_spec_bytes(&spec.to_bincode().unwrap(), &systems_root())
            .expect("shipped GB200 SGLang DeepEP data must load");
        let runtime = RuntimeConfig {
            batch_size: 1,
            isl: 1024,
            osl: 2,
            ..Default::default()
        };

        let (_, generation) = engine
            .run_static_per_op_with_metadata(&runtime, StaticMode::Generation, 32)
            .unwrap();
        assert_eq!(generation.len(), 1, "same-name ops must remain name-folded");
        assert_eq!(
            generation[0].4,
            Some((
                ("generation", "deepep_ll", 32, 8, 4, 1),
                vec![("generation", "deepep_ll", 64, 16, 4, 1)],
            ))
        );
    }

    #[test]
    fn from_spec_bytes_shares_parsed_tables_across_engines() {
        use crate::operators::util_empirical::ProvenanceTier;

        // Two DIFFERENT engine identities (nextn differs) over the SAME db
        // identity: the sweep pattern that motivates the shared-tables memo.
        let spec1 = EngineSpec::new(fixture_engine_config(None), context_ops(), generation_ops());
        let spec2 = EngineSpec::new(
            fixture_engine_config(Some(1)),
            context_ops(),
            generation_ops(),
        );
        let e1 = Engine::from_spec_bytes(&spec1.to_bincode().unwrap(), &systems_root()).unwrap();
        let e2 = Engine::from_spec_bytes(&spec2.to_bincode().unwrap(), &systems_root()).unwrap();
        assert!(
            std::sync::Arc::ptr_eq(e1.database().tables_arc(), e2.database().tables_arc()),
            "engines over the same db identity must share parsed tables"
        );
        // ... while their run state stays per-engine: provenance noted through
        // one engine's database must not appear on the other's accumulator.
        e1.database().note_provenance(ProvenanceTier::Empirical);
        assert_eq!(e2.database().worst_provenance(), ProvenanceTier::Silicon);
    }

    #[test]
    fn both_equals_context_plus_generation() {
        let engine = build_engine(None);
        let rt = runtime(1, 1024, 8);
        let both = engine.run_static(&rt, StaticMode::Both, 32).unwrap();
        let ctx = engine.run_static(&rt, StaticMode::Context, 32).unwrap();
        let gen = engine.run_static(&rt, StaticMode::Generation, 32).unwrap();

        assert!((both.context_ms - ctx.context_ms).abs() < 1e-9);
        assert!((both.generation_ms - gen.generation_ms).abs() < 1e-9);
        assert!((both.total_ms - (ctx.context_ms + gen.generation_ms)).abs() < 1e-9);
        // total of `Both` is the sum of the two single-phase totals.
        assert!((both.total_ms - (ctx.total_ms + gen.total_ms)).abs() < 1e-9);
    }

    #[test]
    fn context_mode_has_zero_generation() {
        let engine = build_engine(None);
        let rt = runtime(1, 1024, 8);
        let ctx = engine.run_static(&rt, StaticMode::Context, 32).unwrap();
        assert!(ctx.context_ms > 0.0, "context latency must be non-trivial");
        assert_eq!(ctx.generation_ms, 0.0);
        assert_eq!(ctx.total_ms, ctx.context_ms);
    }

    #[test]
    fn generation_mode_has_zero_context() {
        let engine = build_engine(None);
        let rt = runtime(1, 1024, 8);
        let gen = engine.run_static(&rt, StaticMode::Generation, 32).unwrap();
        assert!(
            gen.generation_ms > 0.0,
            "generation latency must be non-trivial"
        );
        assert_eq!(gen.context_ms, 0.0);
        assert_eq!(gen.total_ms, gen.generation_ms);
    }

    #[test]
    fn stride_honored() {
        let engine = build_engine(None);
        // osl=9 → range(0,8,stride). stride=1 visits i=0..7 (8 steps each
        // repeat_count=1); stride=32 visits only i=0 (repeat_count=8). The
        // per-step latency grows with the decode position (s = isl+i+1), so
        // the fine-grained integration differs from the single-sample one.
        let rt = runtime(1, 1024, 9);
        let fine = engine.run_static(&rt, StaticMode::Generation, 1).unwrap();
        let coarse = engine.run_static(&rt, StaticMode::Generation, 32).unwrap();
        assert!(fine.generation_ms > 0.0 && coarse.generation_ms > 0.0);
        assert!(
            (fine.generation_ms - coarse.generation_ms).abs() > 1e-9,
            "stride=1 ({}) and stride=32 ({}) must differ for osl=9",
            fine.generation_ms,
            coarse.generation_ms
        );

        // Hand-rolled expected sum for stride=32, osl=9: one step at i=0
        // (s = isl + 1), repeat_count = min(32, 8) = 8.
        let one_step = run_generation_ops_step(
            &engine.generation_ops,
            engine.database(),
            1, // batch_size * (nextn+1), nextn=0
            1024 + 0 + 1,
            1.0,
            false,
        )
        .unwrap();
        assert!((coarse.generation_ms - one_step * 8.0).abs() < 1e-6);
    }

    #[test]
    fn osl_one_yields_zero_generation() {
        let engine = build_engine(None);
        let rt = runtime(1, 1024, 1);
        let gen = engine.run_static(&rt, StaticMode::Generation, 32).unwrap();
        assert_eq!(gen.generation_ms, 0.0);
    }

    #[test]
    fn prefix_ge_isl_errors() {
        let engine = build_engine(None);
        let rt = RuntimeConfig {
            batch_size: 1,
            isl: 512,
            osl: 2,
            prefix: 512,
            ..Default::default()
        };
        assert!(engine.run_static(&rt, StaticMode::Context, 32).is_err());
    }

    #[test]
    fn mixed_step_empty_is_zero() {
        let engine = build_engine(None);
        assert_eq!(
            engine
                .mixed_step_latency(0, 0, 1024, 8, 0, 1.0, 1.0)
                .unwrap(),
            0.0
        );
    }

    #[test]
    fn mixed_step_nonempty_is_positive() {
        // The full three-pass composition (non-attention + context-attn +
        // gen-attn) over the hand-built fixture must produce a real latency.
        // End-to-end parity is covered by the mixed-step parity cases; this is
        // the fast pure-Rust smoke that the composition actually computes.
        let engine = build_engine(None);
        let ms = engine
            .mixed_step_latency(1024, 2, 1024, 8, 0, 1.0, 1.0)
            .unwrap();
        assert!(
            ms > 0.0 && ms.is_finite(),
            "mixed-step latency must be > 0, got {ms}"
        );
        let breakdown = engine
            .mixed_step_breakdown(1024, 2, 1024, 8, 0, 1.0, 1.0)
            .unwrap();
        assert_eq!(breakdown[0], breakdown[1] + breakdown[2] + breakdown[3]);
        assert_eq!(ms, breakdown[0]);
    }

    // ---- FPM whole-model engine branches ----

    /// FPM engine over the synthetic pair fixture: context = [FpmForward
    /// prefill], generation = [FpmForward decode], empty sol_ops (grid-exact
    /// queries never call SOL).
    fn build_fpm_engine(tmp: &std::path::Path, nextn: Option<u32>) -> Result<Engine, AicError> {
        use crate::perf_database::fpm_forward::tests::{
            default_identity, default_rows, write_pair,
        };
        write_pair(tmp, &default_rows());
        let mut db = PerfDatabase::load(&systems_root(), "b200_sxm", "vllm", "0.24.0").unwrap();
        db.set_fpm_forward_for_test(crate::perf_database::FpmForwardTable::new(
            tmp.to_path_buf(),
            "b200_sxm",
            "vllm",
            "0.25.1",
        ));
        let fpm_op = |phase: FpmPhase| {
            Op::FpmForward(FpmForwardOp {
                name: format!("fpm_forward_{}", phase.as_str()),
                phase,
                model_path: "org/model-a".to_string(),
                match_identity: default_identity(4),
                weight_bytes: 0.0,
                sol_ops: vec![],
            })
        };
        let spec = EngineSpec::new(
            fixture_engine_config(nextn),
            vec![fpm_op(FpmPhase::Prefill)],
            vec![fpm_op(FpmPhase::Decode)],
        );
        Engine::build(spec, Arc::new(db))
    }

    #[test]
    fn fpm_build_rejects_mtp_and_bad_shape() {
        let tmp = tempfile::tempdir().unwrap();
        let err = build_fpm_engine(tmp.path(), Some(1)).unwrap_err();
        assert!(err.to_string().contains("MTP"), "{err}");

        // Mixed granular + FPM list is invalid.
        use crate::perf_database::fpm_forward::tests::default_identity;
        let db = PerfDatabase::load(&systems_root(), "b200_sxm", "vllm", "0.24.0").unwrap();
        let fpm_op = Op::FpmForward(FpmForwardOp {
            name: "fpm_forward_prefill".into(),
            phase: FpmPhase::Prefill,
            model_path: "org/model-a".into(),
            match_identity: default_identity(4),
            weight_bytes: 0.0,
            sol_ops: vec![],
        });
        let spec = EngineSpec::new(
            fixture_engine_config(None),
            vec![fpm_op, context_ops().remove(0)],
            generation_ops(),
        );
        let err = Engine::build(spec, Arc::new(db)).unwrap_err();
        assert!(err.to_string().contains("exactly one FpmForward"), "{err}");
    }

    /// The marginal-decode mixed composition, exact arithmetic over the
    /// fixture rows: the prefill component prices the step's SCHEDULED TOTAL
    /// (ctx + gen tokens) on the prefill curve; decode is the in-curve lerp
    /// minus the (8, 8) -> 6.0 baseline floor.
    #[test]
    fn fpm_mixed_step_is_prefill_plus_marginal_decode() {
        let tmp = tempfile::tempdir().unwrap();
        let engine = build_fpm_engine(tmp.path(), None).unwrap();
        // ctx: 2048 tokens / isl 2048 -> batch 1, totals (1, 2048+8, 0):
        // in-curve lerp between (1,2048)->20.0 and (1,4096)->40.0.
        // gen: batch 8; osl=0 clamps to 1 -> isl' = 2048, one step at
        // s = 2049 -> kv = 8*2049 = 16392: lerp between (8,4096)->7.0 and
        // (8,65536)->9.0, minus baseline (8, kv_floor=8) -> 6.0.
        let ms = engine
            .mixed_step_latency(2048, 8, 2048, 0, 0, 1.0, 1.0)
            .unwrap();
        let pre = 20.0 + (40.0 - 20.0) * (2056.0 - 2048.0) / (4096.0 - 2048.0);
        let w = (16392.0 - 4096.0) / (65536.0 - 4096.0);
        let decode = 7.0 + (9.0 - 7.0) * w;
        let expected = pre + (decode - 6.0);
        assert!((ms - expected).abs() < 1e-9, "got {ms}, want {expected}");
    }

    /// FPM engine over CUSTOM rows (cliff pair + chunk coordinates); same
    /// wiring as [`build_fpm_engine`].
    fn build_fpm_engine_with_rows(
        tmp: &std::path::Path,
        rows: &[crate::perf_database::fpm_forward::tests::RowSpec],
    ) -> Result<Engine, AicError> {
        use crate::perf_database::fpm_forward::tests::{default_identity, write_pair};
        write_pair(tmp, rows);
        let mut db = PerfDatabase::load(&systems_root(), "b200_sxm", "vllm", "0.24.0").unwrap();
        db.set_fpm_forward_for_test(crate::perf_database::FpmForwardTable::new(
            tmp.to_path_buf(),
            "b200_sxm",
            "vllm",
            "0.25.1",
        ));
        let fpm_op = |phase: FpmPhase| {
            Op::FpmForward(FpmForwardOp {
                name: format!("fpm_forward_{}", phase.as_str()),
                phase,
                model_path: "org/model-a".to_string(),
                match_identity: default_identity(4),
                weight_bytes: 0.0,
                sol_ops: vec![],
            })
        };
        let spec = EngineSpec::new(
            fixture_engine_config(None),
            vec![fpm_op(FpmPhase::Prefill)],
            vec![fpm_op(FpmPhase::Decode)],
        );
        Engine::build(spec, Arc::new(db))
    }

    fn cliff_rows() -> Vec<crate::perf_database::fpm_forward::tests::RowSpec> {
        use crate::perf_database::fpm_forward::tests::RowSpec;
        let mk = |kind: &'static str, batch: u32, prefill: u32, kv: u32, lat: f64| RowSpec {
            workload_kind: kind,
            batch_size: batch,
            total_prefill_tokens: prefill,
            total_kv_read_tokens: kv,
            latency_ms: lat,
            ..RowSpec::default()
        };
        vec![
            // CUDA-graph cliff pair at capture=2048, plus the eager plateau.
            mk("prefill", 1, 2048, 0, 47.0),
            mk("prefill", 1, 2049, 0, 99.0),
            mk("prefill", 1, 4096, 0, 99.0),
            // Chunk coordinates for the multi-chunk average test.
            mk("prefill", 1, 1032, 0, 10.0),
            mk("prefill", 1, 1032, 1024, 14.0),
            mk("decode", 8, 0, 8, 6.0),
            mk("decode", 8, 0, 4096, 7.0),
            mk("decode", 8, 0, 65536, 9.0),
        ]
    }

    /// Spec test 1+2: the step's total (ctx + gen) picks the regime side.
    /// ctx=2048 alone sits ON the capture boundary (graph side, 47 ms); the
    /// same chunk with ANY decode riders crosses it and must price eager.
    #[test]
    fn fpm_mixed_step_total_crosses_the_graph_cliff() {
        let tmp = tempfile::tempdir().unwrap();
        let engine = build_fpm_engine_with_rows(tmp.path(), &cliff_rows()).unwrap();
        // In-graph: pure prefill step, totals (1, 2048, 0) -> exact 47.0.
        let graph = engine
            .mixed_step_breakdown(2048, 0, 2048, 0, 0, 1.0, 1.0)
            .unwrap();
        assert!(
            (graph[1] - 47.0).abs() < 1e-9,
            "graph-side prefill {}",
            graph[1]
        );
        // Crossing: 8 decode riders push the total to 2056 -> eager plateau.
        let eager = engine
            .mixed_step_breakdown(2048, 8, 2048, 0, 0, 1.0, 1.0)
            .unwrap();
        assert!(
            (eager[1] - 99.0).abs() < 1e-9,
            "eager-side prefill {}",
            eager[1]
        );
        assert!(eager[1] > graph[1] * 2.0 - 1e-9);
    }

    /// Spec test 4: chunked requests price each chunk at its own
    /// (chunk + gen, past_kv) coordinates; the component is their average.
    #[test]
    fn fpm_mixed_step_chunks_average_exact_coordinates() {
        let tmp = tempfile::tempdir().unwrap();
        let engine = build_fpm_engine_with_rows(tmp.path(), &cliff_rows()).unwrap();
        // ctx=1024 of isl=2048: chunk 1 -> (1, 1032, 0) = 10.0,
        // chunk 2 -> (1, 1032, 1024) = 14.0; average 12.0.
        let parts = engine
            .mixed_step_breakdown(1024, 8, 2048, 0, 0, 1.0, 1.0)
            .unwrap();
        assert!((parts[1] - 12.0).abs() < 1e-9, "chunk average {}", parts[1]);
    }

    /// A generation-only step keeps the FULL decode latency (no pass to ride
    /// on) and uses the Python static-path convention s = isl + osl/2 + 1.
    #[test]
    fn fpm_genonly_step_keeps_full_decode() {
        let tmp = tempfile::tempdir().unwrap();
        let engine = build_fpm_engine(tmp.path(), None).unwrap();
        // gen_tokens=8, isl=511, osl=0 -> isl'=511, one step at s=512 ->
        // kv = 8*512 = 4096: exact decode row -> 7.0, NOT 7.0 - 6.0.
        let ms = engine.decode_step_latency(8, 511, 0, 1.0).unwrap();
        assert!((ms - 7.0).abs() < 1e-12, "got {ms}");
        // mixed with ctx_tokens=0 must agree with the genonly convention
        let mixed = engine
            .mixed_step_latency(0, 8, 511, 0, 0, 1.0, 1.0)
            .unwrap();
        assert!((mixed - 7.0).abs() < 1e-12, "got {mixed}");
        assert_eq!(engine.decode_step_latency(0, 511, 0, 1.0).unwrap(), 0.0);
    }

    /// A fully prefix-cached payload retains prefill request/KV metadata
    /// while scheduling no fresh prefill compute: dispatch must be
    /// token-based (aligned with `IterationFeatures`) — a count-based check
    /// would query prefill at zero tokens (outside the FPM domain) and
    /// price decode as marginal work riding a pass that does not exist.
    #[test]
    fn fpm_rank_prefix_cached_payload_is_decode_only() {
        use crate::fpm::{ForwardPassMetrics, ScheduledRequestMetrics};
        let tmp = tempfile::tempdir().unwrap();
        let engine = build_fpm_engine(tmp.path(), None).unwrap();
        let metrics = ForwardPassMetrics {
            scheduled_requests: ScheduledRequestMetrics {
                num_prefill_requests: 1,
                sum_prefill_tokens: 0,
                sum_prefill_kv_tokens: 4096,
                num_decode_requests: 8,
                sum_decode_kv_tokens: 4096, // exact decode row -> 7.0
                ..Default::default()
            },
            ..Default::default()
        };
        // FULL decode latency (decode-only), not the marginal composition.
        let ms = engine.forward_pass_time_ms(&[metrics]).unwrap();
        assert!((ms - 7.0).abs() < 1e-12, "{ms}");
    }

    /// Telemetry dispatch: single-workload FPM ranks flow through the shared
    /// free fns; a mixed rank composes prefill + marginal decode.
    #[test]
    fn fpm_rank_latency_marginal_composition() {
        use crate::fpm::{ForwardPassMetrics, ScheduledRequestMetrics};
        let tmp = tempfile::tempdir().unwrap();
        let engine = build_fpm_engine(tmp.path(), None).unwrap();

        let mixed = ForwardPassMetrics {
            scheduled_requests: ScheduledRequestMetrics {
                num_prefill_requests: 2,
                sum_prefill_tokens: 2 * 1024,
                sum_prefill_kv_tokens: 0,
                num_decode_requests: 8,
                sum_decode_kv_tokens: 8 * 4096,
                ..Default::default()
            },
            ..Default::default()
        };
        // prefill: totals coords (2, 2048, 0) -> exact 21.0. decode: totals
        // coords (8, 32768): lerp between (8,4096)->7.0 and (8,65536)->9.0,
        // minus baseline (8, 8) -> 6.0.
        let w = (32768.0 - 4096.0) / (65536.0 - 4096.0);
        let decode = 7.0 + (9.0 - 7.0) * w;
        let expected = 21.0 + (decode - 6.0);
        let got = engine.forward_pass_time_ms(&[mixed]).unwrap();
        assert!((got - expected).abs() < 1e-9, "got {got}, want {expected}");
    }

    /// The FPM rank dispatch queries RAW iteration totals — the tables'
    /// native coordinate system — not the op-level per-request averages,
    /// which floor-divide away up to (n - 1) tokens per axis.
    #[test]
    fn fpm_rank_uses_iteration_totals_not_averages() {
        use crate::fpm::{ForwardPassMetrics, ScheduledRequestMetrics};
        let tmp = tempfile::tempdir().unwrap();
        let engine = build_fpm_engine(tmp.path(), None).unwrap();

        // 8 decode requests, 32,773 total KV: NOT divisible by 8. Totals
        // convention queries (8, 32773); the old average convention floored
        // to kv_per_req = 4096 -> (8, 32768).
        let decode_only = ForwardPassMetrics {
            scheduled_requests: ScheduledRequestMetrics {
                num_decode_requests: 8,
                sum_decode_kv_tokens: 32_773,
                ..Default::default()
            },
            ..Default::default()
        };
        let w = (32_773.0 - 4096.0) / (65_536.0 - 4096.0);
        let expected = 7.0 + (9.0 - 7.0) * w;
        let got = engine.forward_pass_time_ms(&[decode_only]).unwrap();
        assert!((got - expected).abs() < 1e-9, "got {got}, want {expected}");
    }

    /// The FPM shape guard must see through Overlap/Fallback nesting: a
    /// hand-built spec hiding an FpmForward inside a composite would
    /// otherwise ride the name-filtered mix-step passes with the wrong
    /// workload shape (and FallbackOp swallows its PerfDatabase misses).
    #[test]
    fn nested_fpm_op_is_rejected_at_build() {
        use crate::perf_database::fpm_forward::tests::default_identity;
        let db = PerfDatabase::load(&systems_root(), "b200_sxm", "vllm", "0.24.0").unwrap();
        let hidden = Op::Overlap(crate::operators::OverlapOp::new(
            "hidden",
            vec![Op::FpmForward(FpmForwardOp {
                name: "fpm_forward_prefill".into(),
                phase: FpmPhase::Prefill,
                model_path: "org/model-a".into(),
                match_identity: default_identity(4),
                weight_bytes: 0.0,
                sol_ops: vec![],
            })],
            vec![],
        ));
        let spec = EngineSpec::new(fixture_engine_config(None), vec![hidden], generation_ops());
        let err = Engine::build(spec, Arc::new(db)).unwrap_err();
        assert!(
            err.to_string()
                .contains("exactly one FpmForward op per phase"),
            "{err}"
        );
    }

    /// Lock the one piece of orchestration that lives ONLY in the Engine: the
    /// `(nextn + 1)` decode-batch multiplier (Python `_run_generation_phase:200`).
    /// Builds an Engine with `nextn=1` over the hand-built ops and asserts the
    /// generation phase queries the perf-DB at the doubled decode batch — i.e.
    /// it equals the shared `run_generation_ops_step` free fn at `2 *
    /// batch_size`. Proves `nextn` threads from `spec.engine.speculative` into
    /// the gen batch (the one behavior genuinely unique to the Engine layer).
    #[test]
    fn nextn_scales_decode_batch() {
        let engine_nextn1 = build_engine(Some(1));
        assert_eq!(engine_nextn1.nextn, 1);

        // osl=2 → one decode step at s = isl + 1. With nextn=1 the engine must
        // query at batch_size * 2; mirror that with the free fn at 2*batch.
        let rt = runtime(1, 1024, 2);
        let gen = engine_nextn1
            .run_static(&rt, StaticMode::Generation, 32)
            .unwrap();
        let doubled = run_generation_ops_step(
            &engine_nextn1.generation_ops,
            engine_nextn1.database(),
            2,
            1024 + 1,
            1.0,
            false,
        )
        .unwrap();
        assert!(
            (gen.generation_ms - doubled).abs() < 1e-9,
            "nextn=1 gen ({}) must equal the gen-step at 2*batch ({})",
            gen.generation_ms,
            doubled
        );
    }

    /// The SOL-decomposition FFI must agree with a Sol-view evaluation of
    /// the same ops: each entry's `sol_time` IS the op's Sol-mode latency
    /// (shared query path, shared shape math), and for single-leaf ops the
    /// leaf identity `sol_time = max(sol_math, sol_mem)` holds. The GEMM
    /// triple is additionally pinned to its closed-form roofline — the
    /// Python SOL_FULL `get_sol` verbatim.
    #[test]
    fn evaluate_ops_sol_json_matches_sol_view() {
        use crate::perf_database::gemm::quant_tc_flops;
        use crate::session::query_context_op;

        let engine = build_engine(None);
        let ops = context_ops();
        let ops_json = serde_json::to_string(&ops).unwrap();
        let (batch, s) = (4u32, 512u32);
        let sol = engine
            .evaluate_ops_sol_json(&ops_json, true, batch, s, 0, 1.0, None)
            .unwrap();
        assert_eq!(sol.len(), ops.len());

        let sol_db = engine.database().sol_full_view();
        for (op, entry) in ops.iter().zip(&sol) {
            let r = query_context_op(op, &sol_db, batch, s, 0, 1.0, None).unwrap();
            assert_eq!(entry.0, op.name());
            assert!(
                (entry.1 - r.latency_ms).abs() < 1e-12,
                "{}: sol_time {} != Sol-view latency {}",
                entry.0,
                entry.1,
                r.latency_ms
            );
        }

        // Single-leaf ops: sol_time = max(sol_math, sol_mem). (Composed ops
        // like context attention add fused-extras leaves AFTER the max, so
        // the identity intentionally does not hold there.)
        for entry in sol.iter().take(2) {
            assert!(
                (entry.1 - entry.2.max(entry.3)).abs() < 1e-12,
                "{}: leaf max identity broken: {:?}",
                entry.0,
                entry
            );
        }

        // GEMM triple == the closed-form roofline at m = batch * s
        // (Python `GEMM._query_gemm_table::get_sol`).
        let spec = &engine.database().system_spec;
        let quant = GemmQuantMode::Fp8Block;
        let tc_flops = quant_tc_flops(spec, quant.mapping()).unwrap();
        let (m, n, k) = ((batch * s) as f64, 4096.0, 4096.0);
        let math = 2.0 * m * n * k / tc_flops * 1000.0;
        let mem = quant.mapping().memory * (m * n + m * k + n * k) / spec.gpu.mem_bw * 1000.0;
        let gemm = &sol[1];
        assert!(
            (gemm.2 - math).abs() < 1e-12,
            "sol_math {} != {math}",
            gemm.2
        );
        assert!((gemm.3 - mem).abs() < 1e-12, "sol_mem {} != {mem}", gemm.3);
    }

    /// GLM-5.2 DSA full/skip amortization (`full_frac < 1`) must blend the
    /// SOL decomposition componentwise alongside the latency — the blended
    /// result reaches `PerOpSolFold::add` with components, and each component
    /// equals `w*full + (1-w)*skip` of the closed-form rooflines.
    #[test]
    fn evaluate_ops_sol_json_blends_dsa_full_skip() {
        use crate::common::enums::{FmhaQuantMode, KvCacheQuantMode};
        use crate::operators::DsaModuleOp;
        use crate::perf_database::dsa::{dsa_context_sol, dsa_context_sol_flops, dsa_dims};

        let engine = build_engine(None);
        let spec = &engine.database().system_spec;
        let mut op = DsaModuleOp::new(
            "dsa_context",
            128,
            KvCacheQuantMode::Bfloat16,
            FmhaQuantMode::Bfloat16,
            GemmQuantMode::Bfloat16,
            "DeepseekV32ForCausalLM",
            2048,
        );
        let w = 0.5;
        op.full_frac = w;
        let (b, s) = (1u32, 4096u32);
        let ops_json = serde_json::to_string(&vec![Op::DsaContext(op.clone())]).unwrap();
        let sol = engine
            .evaluate_ops_sol_json(&ops_json, true, b, s, 0, 1.0, None)
            .unwrap();
        assert_eq!(sol.len(), 1);

        let dims = dsa_dims(&op.architecture);
        let flops = dsa_context_sol_flops(spec, op.gemm_quant_mode, op.fmha_quant_mode).unwrap();
        let leaf = |skip: bool| {
            dsa_context_sol(
                spec,
                dims,
                op.index_topk as i64,
                op.kv_cache_dtype,
                op.fmha_quant_mode,
                op.gemm_quant_mode,
                b as i64,
                s as i64,
                0,
                op.num_heads as i64,
                skip,
                flops,
            )
        };
        let (full, skip) = (leaf(false), leaf(true));
        let expected_math = w * full.math_ms + (1.0 - w) * skip.math_ms;
        let expected_mem = w * full.mem_ms + (1.0 - w) * skip.mem_ms;
        let expected_time = w * full.time_ms() + (1.0 - w) * skip.time_ms();
        let (_, sol_time, sol_math, sol_mem) = &sol[0];
        assert!(
            (sol_time - expected_time).abs() < 1e-12,
            "{sol_time} vs {expected_time}"
        );
        assert!(
            (sol_math - expected_math).abs() < 1e-12,
            "{sol_math} vs {expected_math}"
        );
        assert!(
            (sol_mem - expected_mem).abs() < 1e-12,
            "{sol_mem} vs {expected_mem}"
        );
        // The skip leaf must actually differ from the full leaf, or this
        // test would pass vacuously on a broken blend.
        assert!(skip.time_ms() < full.time_ms());
    }

    /// CP DSA currently composes latency-only sparse MQA/top-k deltas, so the
    /// SOL_FULL API must reject that configuration at the DSA boundary. It
    /// must not run the composition and fail later with PerOpSolFold's generic
    /// `no SOL decomposition` error. The adjacent non-CP blend test pins the
    /// supported `cp_size=1` contract.
    #[test]
    fn evaluate_ops_sol_json_rejects_cp_dsa_explicitly() {
        use crate::operators::DsaModuleOp;

        let engine = build_engine(None);
        let mut op = DsaModuleOp::new(
            "dsa_context",
            64,
            KvCacheQuantMode::Bfloat16,
            FmhaQuantMode::Bfloat16,
            GemmQuantMode::Bfloat16,
            "GlmMoeDsaForCausalLM",
            2048,
        );
        op.cp_size = 2;
        op.full_frac = 0.5;
        let ops_json = serde_json::to_string(&vec![Op::DsaContext(op)]).unwrap();
        let err = engine
            .evaluate_ops_sol_json(&ops_json, true, 1, 4096, 0, 1.0, None)
            .unwrap_err();

        match err {
            AicError::InvalidEngineConfig(message) => {
                assert!(
                    message.contains("DSA context SOL_FULL decomposition is not supported")
                        && message.contains("cp_size=2")
                        && message.contains("sparse MQA/top-k deltas are latency-only"),
                    "unexpected message: {message}"
                );
            }
            other => panic!("expected explicit CP DSA configuration error, got {other}"),
        }
    }

    /// Op families whose SOL branch does not export its decomposition yet
    /// must error loudly (never silently chart a wrong breakdown).
    #[test]
    fn evaluate_ops_sol_json_rejects_unexported_families() {
        let engine = build_engine(None);
        let ops = vec![Op::Mamba2(crate::operators::Mamba2Op {
            name: "mamba2".into(),
            scale_factor: 1.0,
            kernel_source: "causal_conv1d_fn".into(),
            phase: "context".into(),
            d_model: 4096,
            d_state: 128,
            d_conv: 4,
            nheads: 128,
            head_dim: 64,
            n_groups: 8,
            chunk_size: 256,
        })];
        let ops_json = serde_json::to_string(&ops).unwrap();
        let err = engine
            .evaluate_ops_sol_json(&ops_json, true, 1, 128, 0, 1.0, None)
            .unwrap_err();
        assert!(matches!(&err, AicError::SolNotImplemented(_)));
        assert!(
            err.to_string().contains("no SOL decomposition"),
            "unexpected error: {err}"
        );
    }
}
