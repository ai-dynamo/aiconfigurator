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
//! `predict_*_latency`, `mixed_step_latency`, `decode_step_latency`,
//! `build_aic_engine`) live in [`crate::py`]. The agg sweep is orchestrated in
//! Python — there is no Rust `run_agg`.

use std::sync::Arc;

use crate::common::error::AicError;
use crate::engine::spec::EngineSpec;
use crate::operators::{FpmForwardOp, FpmPhase, Op};
use crate::perf_database::PerfDatabase;
use crate::session::{get_mix_step_ops, run_context_ops, run_generation_ops_step};
use crate::{validate_forward_pass_metrics, ForwardPassMetrics};

/// Per-call runtime inputs. Field-for-field mirror of the Python
/// `sdk/config.RuntimeConfig`.
///
/// Only the fields the static composition reads are consumed (`batch_size`,
/// `beam_width`, `isl`, `osl`, `prefix`). The two imbalance-correction scales
/// are carried for wire parity but are not yet threaded into the op
/// queries (the live FPM path hard-codes them to 1.0; see
/// `session::run_context_ops`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RuntimeConfig {
    pub batch_size: u32,
    /// Beam width. Defaults to 1; the engine-step path is not exercised for
    /// beam > 1 (matches `session.rs`), so it does not scale the gen batch.
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
    /// caller (`build_aic_engine` / `from_spec_bytes`) is responsible for
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
    pub fn from_spec_bytes(bytes: &[u8], systems_root: &std::path::Path) -> Result<Engine, AicError> {
        let spec = EngineSpec::from_bincode(bytes)?;
        let version = spec.engine.backend_version.as_deref().ok_or_else(|| {
            AicError::InvalidEngineConfig(
                "backend_version is required to load the perf database".to_string(),
            )
        })?;
        // The spec's own `systems_path` wins when present; otherwise fall back
        // to the `systems_root` argument.
        let systems_root = spec.engine.systems_path.as_deref().unwrap_or(systems_root);
        let db = PerfDatabase::load_with_sources(
            systems_root,
            &spec.engine.system_name,
            spec.engine.backend.as_str(),
            version,
            &spec.engine.perf_db_sources,
        )?;
        Engine::build(spec, Arc::new(db))
    }

    /// Shared perf database handle.
    pub fn database(&self) -> &Arc<PerfDatabase> {
        &self.db
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
            StaticMode::Generation | StaticMode::Both => self.run_generation_phase(runtime, stride)?,
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
        let bs = runtime.batch_size.saturating_mul(self.nextn.saturating_add(1));
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
            let step = run_generation_ops_step(&self.generation_ops, &self.db, bs, s)?;
            let repeat_count = stride.min(upper - i);
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

    /// One mixed (chunked-prefill + decode) step latency. Mirrors Python
    /// `_get_mix_step_latency` (`base_backend.py:706`) via the same FPM
    /// parameter packing the live ctypes bridge
    /// (`rust_engine_step.estimate_mixed_step_latency_with_rust`) uses, so the
    /// numbers match Python to within the parity tolerance.
    ///
    /// The packing reproduces the bridge's FPM build + the session unpack in
    /// `session::rank_latency_ms`, then calls the shared
    /// [`get_mix_step_ops`] composition over this engine's op lists:
    ///
    /// ```text
    /// // prefill chunk (FPM build)
    /// n_prefill   = max(ceil(ctx_tokens / isl), 1)
    /// cached_total= prefix * n_prefill
    /// new_prefill = ctx_tokens - cached_total (>=1) if cached_total else ctx_tokens
    /// // decode (FPM build, with the (nextn+1) MTP multiplier)
    /// eff_gen     = gen_tokens * (nextn + 1)
    /// kv_total    = eff_gen * (isl + osl/2)
    /// // session unpack -> get_mix_step_ops args
    /// new_tokens_per_req = new_prefill / n_prefill
    /// prefix_per_req     = cached_total / n_prefill
    /// combined_prefix    = cached_total
    /// kv_per_decode      = kv_total / eff_gen
    /// decode_batch       = eff_gen
    /// ctx_tokens(arg)    = new_prefill   // == sum_prefill_tokens
    /// gen_tokens(arg)    = eff_gen       // == num_decode_requests
    /// ```
    pub fn mixed_step_latency(
        &self,
        ctx_tokens: u32,
        gen_tokens: u32,
        isl: u32,
        osl: u32,
        prefix: u32,
    ) -> Result<f64, AicError> {
        let isl = isl.max(1);
        let osl = osl.max(1);

        // Whole-model FPM ops must never reach the name-filtered three-pass
        // composition below (they match neither attention filter and would
        // ride pass 1 with the wrong workload shape). Python branches the
        // same way at `_get_mix_step_latency` -> `_get_fpm_mix_step_latency`.
        if let Some((_, decode_op)) = self.fpm_ops() {
            return self.fpm_mixed_step_latency(decode_op, ctx_tokens, gen_tokens, isl, osl, prefix);
        }

        // ---- Prefill chunk FPM build (mirrors the bridge) ----
        let (sum_prefill_tokens, sum_prefill_kv_tokens, n_prefill) = if ctx_tokens > 0 {
            let n_prefill = ctx_tokens.div_ceil(isl).max(1);
            let cached_total = prefix * n_prefill;
            let new_prefill = if cached_total > 0 {
                ctx_tokens.saturating_sub(cached_total).max(1)
            } else {
                ctx_tokens
            };
            (new_prefill, cached_total, n_prefill)
        } else {
            (0, 0, 0)
        };

        // ---- Decode FPM build with the (nextn + 1) MTP multiplier ----
        let (num_decode_requests, sum_decode_kv_tokens) = if gen_tokens > 0 {
            let eff_gen = gen_tokens.saturating_mul(self.nextn.saturating_add(1));
            let kv_per_req = isl.saturating_add(osl / 2);
            (eff_gen, eff_gen.saturating_mul(kv_per_req))
        } else {
            (0, 0)
        };

        if sum_prefill_tokens == 0 && num_decode_requests == 0 {
            return Ok(0.0);
        }

        // ---- Session unpack (mirrors session::rank_latency_ms) ----
        let n_prefill_safe = n_prefill.max(1);
        let new_tokens_per_req = (sum_prefill_tokens / n_prefill_safe).max(1);
        let prefix_per_req = sum_prefill_kv_tokens / n_prefill_safe;
        let n_decode_safe = num_decode_requests.max(1);
        let kv_per_decode = sum_decode_kv_tokens / n_decode_safe;

        crate::session::get_mix_step_ops(
            &self.context_ops,
            &self.generation_ops,
            &self.db,
            sum_prefill_tokens,
            num_decode_requests,
            new_tokens_per_req,
            prefix_per_req,
            sum_prefill_kv_tokens,
            kv_per_decode,
            num_decode_requests,
        )
    }

    /// One generation-only step latency. Mirrors Python
    /// `_get_genonly_step_latency` (`base_backend.py:834`) /
    /// `rust_engine_step.estimate_decode_step_latency_with_rust`: one decode
    /// step at `s = isl + osl/2` with the decode batch scaled by `(nextn + 1)`.
    pub fn decode_step_latency(&self, gen_tokens: u32, isl: u32, osl: u32) -> Result<f64, AicError> {
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
        let context_length = isl.max(1).saturating_add(osl.max(1) / 2);
        run_generation_ops_step(&self.generation_ops, &self.db, effective_batch, context_length)
    }

    /// Mixed step for FPM engines: pure-prefill chunk cost plus the decode
    /// work's MARGINAL cost. Literal port of Python
    /// `_get_fpm_mix_step_latency` (`base_backend.py:1053`):
    ///
    /// ```text
    /// ctx_tokens > 0:
    ///   ctx_ms  = run_static(static_ctx, batch=ceil(ctx_tokens/isl), isl, osl=1, prefix)
    ///   total  += ctx_ms / ceil(isl / ctx_tokens)          # average per-chunk cost
    /// gen_tokens > 0:
    ///   gen_ms  = run_static(static_gen, batch=gen_tokens, isl'=isl+osl/2, osl=2)
    ///             # one step at s = isl + osl/2 + 1, batch scaled by (nextn+1)
    ///   base    = decode.query_pass_baseline(gen_tokens*(nextn+1))  if ctx_tokens > 0 else 0
    ///   total  += max(gen_ms - base, 0)   # gen-only steps keep the full latency
    /// ```
    ///
    /// A mixed step is one shared forward pass: weight reads and per-step
    /// fixed overheads are paid once, by the prefill component; sampling the
    /// decode curve at its KV-axis floor isolates that shared part, so the
    /// subtraction keeps only the KV-read/attention marginal cost.
    fn fpm_mixed_step_latency(
        &self,
        decode_op: &FpmForwardOp,
        ctx_tokens: u32,
        gen_tokens: u32,
        isl: u32,
        osl: u32,
        prefix: u32,
    ) -> Result<f64, AicError> {
        let mut total = 0.0_f64;
        if ctx_tokens > 0 {
            let rt = RuntimeConfig {
                batch_size: ctx_tokens.div_ceil(isl),
                isl,
                osl: 1,
                prefix,
                ..Default::default()
            };
            let ctx_ms = self.run_context_phase(&rt)?;
            let chunk_scale = (isl as f64 / ctx_tokens as f64).ceil();
            total += ctx_ms / chunk_scale;
        }
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
            total += (gen_ms - baseline_ms).max(0.0);
        }
        Ok(total)
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
        let has_prefill = sched.num_prefill_requests > 0;
        let has_decode = sched.num_decode_requests > 0;

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
            )?;
        }

        if has_decode {
            let n_decode = sched.num_decode_requests.max(1);
            let kv_per_req = sched.sum_decode_kv_tokens / n_decode;
            total += run_generation_ops_step(&self.generation_ops, &self.db, n_decode, kv_per_req)?;
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
    use crate::operators::{ContextAttentionOp, ElementwiseOp, GemmOp, GenerationAttentionOp};
    use crate::{BackendKind, EngineConfig, ParallelMapping, QuantizationConfig};

    fn systems_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../src/aiconfigurator_core/systems")
    }

    const TEST_MODEL: &str = "MiniMaxAI/MiniMax-M2.5";

    /// Hand-built context op list against the b200_sxm/vllm/0.19.0 perf tables.
    /// `Elementwise` is DB-free (pure mem-bandwidth SOL); `Gemm` and
    /// `ContextAttention` hit existing perf tables. The (deleted) model layer
    /// previously sourced these lists from the HF config.
    fn context_ops() -> Vec<Op> {
        vec![
            Op::Elementwise(ElementwiseOp {
                name: "rmsnorm".into(),
                scale_factor: 1.0,
                bytes_per_token: 8192.0,
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
            }),
        ]
    }

    fn generation_ops() -> Vec<Op> {
        vec![
            Op::Elementwise(ElementwiseOp {
                name: "rmsnorm".into(),
                scale_factor: 1.0,
                bytes_per_token: 8192.0,
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
            backend_version: Some("0.19.0".to_string()),
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
            speculative: nextn.map(|n| crate::SpeculativeConfig {
                nextn: Some(n),
                // Batch-scaling-only fixture; 0.0 keeps the enabled-MTP
                // contract (nextn > 0 requires an acceptance value) satisfied.
                nextn_accepted: Some(0.0),
            }),
            perf_db_sources: Default::default(),
            extra: BTreeMap::new(),
        }
    }

    /// Build an `Engine` from the hand-built op lists over the real fixture DB.
    fn build_engine(nextn: Option<u32>) -> Engine {
        let db = PerfDatabase::load(&systems_root(), "b200_sxm", "vllm", "0.19.0").unwrap();
        let spec = EngineSpec::new(fixture_engine_config(nextn), context_ops(), generation_ops());
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
        assert!(gen.generation_ms > 0.0, "generation latency must be non-trivial");
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
        assert_eq!(engine.mixed_step_latency(0, 0, 1024, 8, 0).unwrap(), 0.0);
    }

    #[test]
    fn mixed_step_nonempty_is_positive() {
        // The full three-pass composition (non-attention + context-attn +
        // gen-attn) over the hand-built fixture must produce a real latency.
        // End-to-end parity is covered by the mixed-step parity cases; this is
        // the fast pure-Rust smoke that the composition actually computes.
        let engine = build_engine(None);
        let ms = engine.mixed_step_latency(1024, 2, 1024, 8, 0).unwrap();
        assert!(ms > 0.0 && ms.is_finite(), "mixed-step latency must be > 0, got {ms}");
    }

    // ---- FPM whole-model engine branches ----

    /// FPM engine over the synthetic pair fixture: context = [FpmForward
    /// prefill], generation = [FpmForward decode], empty sol_ops (grid-exact
    /// queries never call SOL).
    fn build_fpm_engine(tmp: &std::path::Path, nextn: Option<u32>) -> Result<Engine, AicError> {
        use crate::perf_database::fpm_forward::tests::{default_identity, default_rows, write_pair};
        write_pair(tmp, &default_rows());
        let mut db = PerfDatabase::load(&systems_root(), "b200_sxm", "vllm", "0.19.0").unwrap();
        db.fpm_forward = crate::perf_database::FpmForwardTable::new(tmp.to_path_buf(), "0.25.1");
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
        let db = PerfDatabase::load(&systems_root(), "b200_sxm", "vllm", "0.19.0").unwrap();
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
    /// fixture rows: prefill (1, 2048, 0) -> 20.0 (chunk_scale 1), decode
    /// in-curve lerp minus the (8, 8) -> 6.0 baseline floor.
    #[test]
    fn fpm_mixed_step_is_prefill_plus_marginal_decode() {
        let tmp = tempfile::tempdir().unwrap();
        let engine = build_fpm_engine(tmp.path(), None).unwrap();
        // ctx: 2048 tokens / isl 2048 -> batch 1, coords (1, 2048, 0) -> exact
        // 20.0; chunk_scale = ceil(2048/2048) = 1.
        // gen: batch 8; osl=0 clamps to 1 -> isl' = 2048, one step at
        // s = 2049 -> kv = 8*2049 = 16392: lerp between (8,4096)->7.0 and
        // (8,65536)->9.0, minus baseline (8, kv_floor=8) -> 6.0.
        let ms = engine.mixed_step_latency(2048, 8, 2048, 0, 0).unwrap();
        let w = (16392.0 - 4096.0) / (65536.0 - 4096.0);
        let decode = 7.0 + (9.0 - 7.0) * w;
        let expected = 20.0 + (decode - 6.0);
        assert!((ms - expected).abs() < 1e-9, "got {ms}, want {expected}");
    }

    /// A generation-only step keeps the FULL decode latency (no pass to ride
    /// on) and uses the Python static-path convention s = isl + osl/2 + 1.
    #[test]
    fn fpm_genonly_step_keeps_full_decode() {
        let tmp = tempfile::tempdir().unwrap();
        let engine = build_fpm_engine(tmp.path(), None).unwrap();
        // gen_tokens=8, isl=511, osl=0 -> isl'=511, one step at s=512 ->
        // kv = 8*512 = 4096: exact decode row -> 7.0, NOT 7.0 - 6.0.
        let ms = engine.decode_step_latency(8, 511, 0).unwrap();
        assert!((ms - 7.0).abs() < 1e-12, "got {ms}");
        // mixed with ctx_tokens=0 must agree with the genonly convention
        let mixed = engine.mixed_step_latency(0, 8, 511, 0, 0).unwrap();
        assert!((mixed - 7.0).abs() < 1e-12, "got {mixed}");
        assert_eq!(engine.decode_step_latency(0, 511, 0).unwrap(), 0.0);
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
        let db = PerfDatabase::load(&systems_root(), "b200_sxm", "vllm", "0.19.0").unwrap();
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
            err.to_string().contains("exactly one FpmForward op per phase"),
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
        )
        .unwrap();
        assert!(
            (gen.generation_ms - doubled).abs() < 1e-9,
            "nextn=1 gen ({}) must equal the gen-step at 2*batch ({})",
            gen.generation_ms,
            doubled
        );
    }
}
