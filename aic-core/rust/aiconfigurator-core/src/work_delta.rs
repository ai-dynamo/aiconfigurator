// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Intra-batch prefill work delta.
//!
//! [`crate::engine::Engine::forward_pass_time_ms`] collapses a scheduled
//! prefill batch to its per-request means `(n, s̄, p̄)` and prices it as if every
//! request had the same shape. For a batch whose requests differ in length that
//! is systematically optimistic: causal attention work is quadratic in the new
//! token count, so spreading a fixed token budget unevenly costs strictly more
//! than spreading it evenly.
//!
//! This module prices that gap. Writing `s` for the newly computed tokens of a
//! request and `p` for its KV read tokens, full-attention prefill work is the
//! trapezoid `s * p + s² / 2` per request. Summing over the batch and
//! subtracting the value the mean point would have given:
//!
//! ```text
//! x_trap = Σ (s_i p_i + s_i² / 2) − n · (s̄ p̄ + s̄² / 2)
//!        = n · [Cov(s, p) + Var(s) / 2]
//! ```
//!
//! Both conserved sums (`Σs`, `Σp`) cancel, so `x_trap` is exactly zero for a
//! uniform batch and depends only on how the batch is partitioned.
//!
//! The per-request lengths are not on the wire. Writing `L = s + p` for the full
//! prompt length, `Var(L) = Var(s) + 2 Cov(s, p) + Var(p)`, hence
//!
//! ```text
//! Cov(s, p) + Var(s) / 2 = [Var(L) − Var(p)] / 2
//! ```
//!
//! and the delta is recoverable from two second moments alone:
//! [`ScheduledRequestMetrics::var_prefill_length`] (which Dynamo defines over
//! the full prompt length) and
//! [`ScheduledRequestMetrics::var_prefill_kv_tokens`].
//!
//! The correction is applied as a multiplier on the context-attention op —
//! [`crate::session::run_context_ops`]'s `seq_imbalance_correction_scale` —
//! because every term above is attention work. Normalising by the uniform
//! attention work `W_u = Σs · (p̄ + s̄ / 2)` makes the calibrated coefficient
//! dimensionless: `beta == 1.0` is the pure work-proportional prior (attention
//! time tracks attention work exactly), and the measured deviation from 1.0
//! absorbs kernel-efficiency effects that vary with attention depth.
//!
//! A [`PrefillDeltaModel`] with no calibration points returns `1.0` for every
//! batch, reproducing the pre-existing estimate bit for bit.

use crate::fpm::ScheduledRequestMetrics;

/// Attention family, which decides whether the trapezoid delta prices the
/// batch and under what conditions.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AttentionFamily {
    /// Every query attends to its whole prefix. The trapezoid is the work, in
    /// every regime.
    #[default]
    Full,
    /// A top-k indexer selects at most `topk` keys per query. Work is the
    /// GATED trapezoid `trap(s, p) * [s + p > topk]` plus a capped-pairs term
    /// `Σ min(p + k + 1, topk)`, neither of which is a function of moments in
    /// general. In the two pure regimes both collapse (see
    /// [`PrefillRegime`]); in a mixed batch they do not.
    SparseIndexer { topk: u32 },
}

/// Where a prefill batch sits relative to a sparse indexer's `topk`.
///
/// The criterion is over the EXTREMES, not the mean: a batch is saturated only
/// when every request's KV read already exceeds `topk`, and unsaturated only
/// when every request's full prompt still fits under it. A batch whose mean
/// sits on one side can easily straddle the boundary, which is why
/// `min_prefill_kv_tokens` / `max_prefill_length` are needed and the variances
/// are not enough.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefillRegime {
    /// Full attention: no `topk` to be on either side of.
    Dense,
    /// Every request has `p >= topk`. The indexer gate is on for every request
    /// AND for the mean point, so the gated trapezoid equals the plain
    /// trapezoid; the capped-pairs term becomes `topk * s`, linear in `s`, and
    /// the conserved `Σs` annihilates it.
    Saturated,
    /// Every request has `s + p <= topk`. The gate is off everywhere, so the
    /// gated trapezoid vanishes; the capped-pairs term is uncapped and equals
    /// `trap(s, p) + s/2`, whose linear part the conserved `Σs` annihilates,
    /// leaving the plain trapezoid.
    Unsaturated,
    /// The batch straddles `topk`. Both sparse terms depend on WHICH requests
    /// fall on which side, which no set of moments can recover. Priced as
    /// uniform rather than guessed.
    Mixed,
    /// The order statistics needed to classify are absent (a v1 emitter).
    Unknown,
}

impl PrefillRegime {
    /// Classify from the scheduled metrics. `Dense` for full attention.
    pub fn classify(family: AttentionFamily, scheduled: &ScheduledRequestMetrics) -> Self {
        let AttentionFamily::SparseIndexer { topk } = family else {
            return Self::Dense;
        };
        let min_kv = scheduled.min_prefill_kv_tokens;
        let max_len = scheduled.max_prefill_length;
        // A v1 emitter leaves both at 0. `max_len == 0` is impossible for a
        // batch that computed any token, so it marks the fields as absent.
        if max_len == 0 {
            return Self::Unknown;
        }
        if min_kv >= topk {
            Self::Saturated
        } else if max_len <= topk {
            Self::Unsaturated
        } else {
            Self::Mixed
        }
    }

    /// Whether the trapezoid work delta prices this batch.
    ///
    /// True for dense attention and for both pure sparse regimes, where the
    /// sparse terms collapse onto the same trapezoid (with a different
    /// coefficient, which the calibration supplies). False for `Mixed` and
    /// `Unknown`, where the correction is declined rather than guessed.
    pub fn trapezoid_applies(self) -> bool {
        matches!(self, Self::Dense | Self::Saturated | Self::Unsaturated)
    }
}

/// Shape summary of one scheduled prefill batch, in the coordinates the delta
/// model is expressed in.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PrefillShape {
    /// Number of prefill requests in the batch (`n`).
    pub num_requests: u32,
    /// Mean newly computed tokens per request (`s̄`).
    pub mean_new_tokens: f64,
    /// Mean KV read tokens per request (`p̄`).
    pub mean_kv_tokens: f64,
    /// Attention depth `p̄ + s̄ / 2`. The trapezoid work of the mean request
    /// divided by `s̄`; the axis the calibrated coefficient varies along.
    pub depth: f64,
    /// Uniform-batch attention work `Σs · depth`.
    pub uniform_work: f64,
    /// Work delta `n · [Cov(s, p) + Var(s) / 2]`, zero for a uniform batch.
    /// Negative values are possible when the batch is anti-correlated in
    /// `(s, p)`.
    pub work_delta: f64,
}

impl PrefillShape {
    /// Summarise the prefill portion of one rank's scheduled metrics.
    ///
    /// Returns `None` when the batch carries no prefill work, or when the
    /// second moments are absent (the Dynamo v1 wire schema omits
    /// `var_prefill_kv_tokens`, and an emitter that reports neither variance
    /// leaves both at `0.0`) — in both cases there is no delta to price.
    pub fn from_scheduled(scheduled: &ScheduledRequestMetrics) -> Option<Self> {
        let num_requests = scheduled.num_prefill_requests;
        if num_requests == 0 || scheduled.sum_prefill_tokens == 0 {
            return None;
        }
        let n = f64::from(num_requests);
        let sum_new = f64::from(scheduled.sum_prefill_tokens);
        let mean_new_tokens = sum_new / n;
        let mean_kv_tokens = f64::from(scheduled.sum_prefill_kv_tokens) / n;
        let depth = mean_kv_tokens + mean_new_tokens / 2.0;
        let uniform_work = sum_new * depth;
        if !(uniform_work > 0.0) {
            return None;
        }
        // Var(L) − Var(p) = Var(s) + 2 Cov(s, p); halved and scaled by the
        // batch size this is Σ trap(s, p) − n · trap(s̄, p̄).
        let work_delta =
            n * (scheduled.var_prefill_length - scheduled.var_prefill_kv_tokens) / 2.0;
        if !work_delta.is_finite() {
            return None;
        }
        Some(Self {
            num_requests,
            mean_new_tokens,
            mean_kv_tokens,
            depth,
            uniform_work,
            work_delta,
        })
    }

    /// Work delta as a fraction of the uniform-batch attention work. This is
    /// the quantity a `beta` of 1.0 would charge in full.
    pub fn relative_work_delta(&self) -> f64 {
        self.work_delta / self.uniform_work
    }
}

/// One calibration point: the dimensionless coefficient `beta` measured at a
/// given attention depth.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PrefillDeltaPoint {
    /// Attention depth `p̄ + s̄ / 2` the point was measured at.
    pub depth: f64,
    /// Measured `beta = (ΔT / T_attn) / (x_trap / W_u)`. `1.0` means attention
    /// time tracks attention work exactly.
    pub beta: f64,
}

/// Calibrated intra-batch work-delta model.
///
/// The coefficient is a function of attention depth alone: two batches at the
/// same `p̄ + s̄ / 2` share a coefficient even when their `s̄` differ severalfold.
/// Between calibration points `beta` is linearly interpolated; outside the
/// calibrated range it is clamped to the nearest endpoint rather than
/// extrapolated.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PrefillDeltaModel {
    /// Calibration points, sorted by depth. Empty means "uncalibrated": every
    /// scale is `1.0`.
    points: Vec<PrefillDeltaPoint>,
    /// Attention family this calibration was measured on. Guards the model
    /// against being applied to a batch whose work is not the trapezoid.
    family: AttentionFamily,
}

impl PrefillDeltaModel {
    /// An uncalibrated model. Every batch prices at scale `1.0`, i.e. the
    /// estimate is unchanged from the uniform collapse.
    pub fn uncalibrated() -> Self {
        Self {
            points: Vec::new(),
            family: AttentionFamily::Full,
        }
    }

    /// Declare the attention family this model was calibrated on. A
    /// `SparseIndexer` model declines to correct batches that straddle `topk`,
    /// and batches whose order statistics are missing.
    pub fn with_family(mut self, family: AttentionFamily) -> Self {
        self.family = family;
        self
    }

    /// The attention family this model is calibrated for.
    pub fn family(&self) -> AttentionFamily {
        self.family
    }

    /// Classify a batch under this model's attention family.
    pub fn regime(&self, scheduled: &ScheduledRequestMetrics) -> PrefillRegime {
        PrefillRegime::classify(self.family, scheduled)
    }

    /// Build from calibration points. Points are sorted by depth; points with a
    /// non-finite depth or beta are dropped. Duplicate depths keep the last
    /// value supplied.
    pub fn from_points(points: impl IntoIterator<Item = PrefillDeltaPoint>) -> Self {
        let mut points: Vec<_> = points
            .into_iter()
            .filter(|point| point.depth.is_finite() && point.beta.is_finite())
            .collect();
        let family = AttentionFamily::Full;
        points.sort_by(|left, right| {
            left.depth
                .partial_cmp(&right.depth)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        points.dedup_by(|later, earlier| {
            if later.depth == earlier.depth {
                earlier.beta = later.beta;
                true
            } else {
                false
            }
        });
        Self { points, family }
    }

    /// The pure work-proportional prior: `beta == 1.0` at every depth. Charges
    /// the full trapezoid delta with no measured correction.
    pub fn work_proportional() -> Self {
        Self {
            points: vec![PrefillDeltaPoint {
                depth: 0.0,
                beta: 1.0,
            }],
            family: AttentionFamily::Full,
        }
    }

    /// Whether this model has any calibration and can return a scale other
    /// than `1.0`.
    pub fn is_calibrated(&self) -> bool {
        !self.points.is_empty()
    }

    /// Interpolate `beta` at the given attention depth. Clamped to the endpoint
    /// values outside the calibrated range; `0.0` when uncalibrated.
    pub fn beta_at(&self, depth: f64) -> f64 {
        let points = self.points.as_slice();
        let (Some(first), Some(last)) = (points.first(), points.last()) else {
            return 0.0;
        };
        if depth <= first.depth {
            return first.beta;
        }
        if depth >= last.depth {
            return last.beta;
        }
        let index = points.partition_point(|point| point.depth < depth);
        let upper = points[index];
        let lower = points[index - 1];
        let span = upper.depth - lower.depth;
        if !(span > 0.0) {
            return upper.beta;
        }
        let weight = (depth - lower.depth) / span;
        lower.beta + weight * (upper.beta - lower.beta)
    }

    /// Context-attention correction scale for one scheduled prefill batch.
    ///
    /// Returns exactly `1.0` for a uniform batch, for an uncalibrated model,
    /// and for metrics that carry no usable second moments. The result is
    /// floored at a small positive value so a strongly anti-correlated batch
    /// can never drive the attention estimate to zero or negative.
    pub fn correction_scale(&self, scheduled: &ScheduledRequestMetrics) -> f64 {
        if !self.is_calibrated() {
            return 1.0;
        }
        if !self.regime(scheduled).trapezoid_applies() {
            return 1.0;
        }
        let Some(shape) = PrefillShape::from_scheduled(scheduled) else {
            return 1.0;
        };
        self.correction_scale_for_shape(&shape)
    }

    /// [`Self::correction_scale`] against an already-summarised batch shape.
    pub fn correction_scale_for_shape(&self, shape: &PrefillShape) -> f64 {
        if !self.is_calibrated() {
            return 1.0;
        }
        let relative = shape.relative_work_delta();
        if relative.abs() < MIN_PRICED_RELATIVE_DELTA {
            return 1.0;
        }
        let scale = 1.0 + self.beta_at(shape.depth) * relative;
        if !scale.is_finite() {
            return 1.0;
        }
        scale.max(MIN_CORRECTION_SCALE)
    }
}

/// Smallest relative work delta worth pricing.
///
/// Latency is not a function of work: at a fixed work scalar the measured
/// latency spans a range, because the same total work can be packed into
/// batches the kernel schedules differently. Charging `beta * x / W_u` assumes
/// the local derivative dominates, which only holds once the perturbation is
/// large compared with that intrinsic spread. Below this threshold the sign of
/// the measured delta is not even reliable -- calibration rungs at
/// `x / W_u < 0.03` come back NEGATIVE, i.e. the uneven batch appears faster
/// than the even one, which is physically impossible and is simply the scatter
/// showing through.
///
/// Batches under the threshold are priced as uniform rather than nudged by a
/// quantity smaller than the noise.
const MIN_PRICED_RELATIVE_DELTA: f64 = 0.05;

/// Floor on the context-attention correction scale. A batch anti-correlated in
/// `(s, p)` produces a negative delta; the floor keeps the priced attention
/// latency positive without silently discarding the correction.
const MIN_CORRECTION_SCALE: f64 = 0.05;

/// Why an (anchor, rung) pair cannot be used to solve a coefficient.
///
/// Every variant is a silent-corruption mode that has bitten this measurement
/// campaign before, so the pair is rejected outright rather than approximated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CalibrationRejection {
    /// The two batches hold different numbers of requests.
    BatchSizeMismatch,
    /// `Σs` or `Σp` drifted between anchor and rung. The delta is defined as
    /// the excess over the equal-length batch with the SAME totals; if the
    /// totals move, the difference in latency is not a work delta.
    TotalsNotConserved,
    /// Anchor and rung sit in different attention regimes. A coefficient
    /// solved here would be attributed to the anchor's regime and applied to
    /// batches it was never measured on. Raising a rung across a sparse
    /// indexer's `topk` is the usual cause.
    RegimeMismatch,
    /// The regime could not be determined (order statistics absent).
    RegimeUnknown,
    /// The anchor is not uniform: its own work delta is non-zero, so it is not
    /// a valid reference point.
    AnchorNotUniform,
    /// The rung carries no work delta -- quantisation flattened the intended
    /// perturbation, leaving a second anchor rather than a rung.
    RungHasNoDelta,
    /// The supplied native attention estimate is not usable as a denominator.
    InvalidAttentionEstimate,
}

/// One measured (anchor, rung) pair, as collected.
///
/// `anchor_attention_ms` is AIC's own native context-attention estimate at the
/// anchor -- the quantity the correction scale multiplies. It is what makes
/// the solved coefficient dimensionless.
#[derive(Clone, Copy, Debug)]
pub struct CalibrationPair<'a> {
    pub anchor: &'a ScheduledRequestMetrics,
    pub anchor_wall_ms: f64,
    pub rung: &'a ScheduledRequestMetrics,
    pub rung_wall_ms: f64,
    pub anchor_attention_ms: f64,
}

impl CalibrationPair<'_> {
    /// Solve `beta` at the anchor's depth, or explain why the pair is unusable.
    ///
    /// Validates conservation and regime agreement against the MEASURED order
    /// statistics rather than whatever the point generator intended, so a
    /// mis-generated rung cannot reach the coefficient table.
    pub fn solve(&self, family: AttentionFamily) -> Result<PrefillDeltaPoint, CalibrationRejection> {
        let (a, r) = (self.anchor, self.rung);
        if a.num_prefill_requests != r.num_prefill_requests || a.num_prefill_requests == 0 {
            return Err(CalibrationRejection::BatchSizeMismatch);
        }
        if a.sum_prefill_tokens != r.sum_prefill_tokens
            || a.sum_prefill_kv_tokens != r.sum_prefill_kv_tokens
        {
            return Err(CalibrationRejection::TotalsNotConserved);
        }
        let anchor_regime = PrefillRegime::classify(family, a);
        let rung_regime = PrefillRegime::classify(family, r);
        if anchor_regime == PrefillRegime::Unknown || rung_regime == PrefillRegime::Unknown {
            return Err(CalibrationRejection::RegimeUnknown);
        }
        if anchor_regime != rung_regime {
            return Err(CalibrationRejection::RegimeMismatch);
        }
        if !anchor_regime.trapezoid_applies() {
            return Err(CalibrationRejection::RegimeMismatch);
        }
        if !(self.anchor_attention_ms > 0.0) || !self.anchor_attention_ms.is_finite() {
            return Err(CalibrationRejection::InvalidAttentionEstimate);
        }
        let anchor_shape =
            PrefillShape::from_scheduled(a).ok_or(CalibrationRejection::AnchorNotUniform)?;
        let rung_shape =
            PrefillShape::from_scheduled(r).ok_or(CalibrationRejection::RungHasNoDelta)?;
        // The anchor's own delta must vanish to machine tolerance; a non-zero
        // one means the "uniform" point was not uniform.
        if anchor_shape.relative_work_delta().abs() > ANCHOR_UNIFORMITY_TOLERANCE {
            return Err(CalibrationRejection::AnchorNotUniform);
        }
        let relative = rung_shape.relative_work_delta();
        if relative.abs() < MIN_RELATIVE_RUNG_DELTA {
            return Err(CalibrationRejection::RungHasNoDelta);
        }
        let beta = ((self.rung_wall_ms - self.anchor_wall_ms) / self.anchor_attention_ms) / relative;
        if !beta.is_finite() {
            return Err(CalibrationRejection::InvalidAttentionEstimate);
        }
        Ok(PrefillDeltaPoint {
            depth: anchor_shape.depth,
            beta,
        })
    }
}

/// How far an anchor's own relative work delta may stray from zero before it
/// stops counting as uniform.
const ANCHOR_UNIFORMITY_TOLERANCE: f64 = 1e-9;

/// Smallest relative work delta a rung must carry to be worth solving from.
/// Below this the perturbation has been flattened (usually by block
/// quantisation) and the solved coefficient is dominated by timing noise.
const MIN_RELATIVE_RUNG_DELTA: f64 = 1e-4;

#[cfg(test)]
mod tests {
    use super::*;

    fn scheduled(
        num_prefill_requests: u32,
        sum_prefill_tokens: u32,
        sum_prefill_kv_tokens: u32,
        var_prefill_length: f64,
        var_prefill_kv_tokens: f64,
    ) -> ScheduledRequestMetrics {
        ScheduledRequestMetrics {
            num_prefill_requests,
            sum_prefill_tokens,
            var_prefill_length,
            sum_prefill_kv_tokens,
            var_prefill_kv_tokens,
            ..Default::default()
        }
    }

    /// Build the metrics a concrete `(s, p)` batch would emit, so tests state
    /// request shapes rather than pre-chewed moments.
    fn from_rows(rows: &[(u32, u32)]) -> ScheduledRequestMetrics {
        let n = rows.len() as f64;
        let mean = |values: &[f64]| values.iter().sum::<f64>() / n;
        let lengths: Vec<f64> = rows.iter().map(|(s, p)| f64::from(*s + *p)).collect();
        let kv: Vec<f64> = rows.iter().map(|(_, p)| f64::from(*p)).collect();
        let variance = |values: &[f64]| {
            let m = mean(values);
            values.iter().map(|value| (value - m).powi(2)).sum::<f64>() / n
        };
        let mut metrics = scheduled(
            rows.len() as u32,
            rows.iter().map(|(s, _)| *s).sum(),
            rows.iter().map(|(_, p)| *p).sum(),
            variance(&lengths),
            variance(&kv),
        );
        metrics.min_prefill_kv_tokens = rows.iter().map(|(_, p)| *p).min().unwrap_or(0);
        metrics.max_prefill_length = rows.iter().map(|(s, p)| *s + *p).max().unwrap_or(0);
        metrics
    }

    /// Direct definition of the delta, for cross-checking the moment identity.
    fn work_delta_from_rows(rows: &[(u32, u32)]) -> f64 {
        let n = rows.len() as f64;
        let trap = |s: f64, p: f64| s * p + s * s / 2.0;
        let total: f64 = rows
            .iter()
            .map(|(s, p)| trap(f64::from(*s), f64::from(*p)))
            .sum();
        let mean_s = rows.iter().map(|(s, _)| f64::from(*s)).sum::<f64>() / n;
        let mean_p = rows.iter().map(|(_, p)| f64::from(*p)).sum::<f64>() / n;
        total - n * trap(mean_s, mean_p)
    }

    #[test]
    fn uniform_batch_has_zero_delta_and_unit_scale() {
        let metrics = from_rows(&[(4096, 1024); 8]);
        let shape = PrefillShape::from_scheduled(&metrics).expect("prefill shape");
        assert_eq!(shape.work_delta, 0.0);
        assert_eq!(shape.depth, 1024.0 + 2048.0);
        assert_eq!(
            PrefillDeltaModel::work_proportional().correction_scale(&metrics),
            1.0
        );
    }

    #[test]
    fn moment_identity_matches_the_direct_trapezoid_sum() {
        // Varies both s and p, and correlates them, so every term of
        // Cov(s, p) + Var(s) / 2 is exercised.
        let rows = [
            (5312u32, 1024u32),
            (2880, 1536),
            (8192, 512),
            (1024, 3072),
            (4096, 1024),
        ];
        let shape = PrefillShape::from_scheduled(&from_rows(&rows)).expect("prefill shape");
        let expected = work_delta_from_rows(&rows);
        assert!(
            (shape.work_delta - expected).abs() <= expected.abs() * 1e-9,
            "moment identity {} vs direct {}",
            shape.work_delta,
            expected
        );
    }

    #[test]
    fn uniform_work_is_the_mean_point_trapezoid() {
        let rows = [(6784u32, 1088u32), (4608, 1344), (4352, 704), (4288, 1984)];
        let shape = PrefillShape::from_scheduled(&from_rows(&rows)).expect("prefill shape");
        let n = rows.len() as f64;
        let mean_s = rows.iter().map(|(s, _)| f64::from(*s)).sum::<f64>() / n;
        let mean_p = rows.iter().map(|(_, p)| f64::from(*p)).sum::<f64>() / n;
        let expected = n * (mean_s * mean_p + mean_s * mean_s / 2.0);
        assert!((shape.uniform_work - expected).abs() <= expected * 1e-9);
    }

    #[test]
    fn uncalibrated_model_never_changes_the_estimate() {
        let metrics = from_rows(&[(8192, 1024), (2048, 1024), (4096, 1024)]);
        let model = PrefillDeltaModel::uncalibrated();
        assert!(!model.is_calibrated());
        assert_eq!(model.correction_scale(&metrics), 1.0);
    }

    #[test]
    fn missing_second_moments_fall_back_to_unit_scale() {
        // A v1 emitter reports neither variance; the batch is priced as uniform.
        let metrics = scheduled(8, 32768, 8192, 0.0, 0.0);
        assert_eq!(
            PrefillDeltaModel::work_proportional().correction_scale(&metrics),
            1.0
        );
    }

    #[test]
    fn work_proportional_charges_the_full_relative_delta() {
        let metrics = from_rows(&[(5312, 1024), (5312, 1024), (2880, 1024), (2880, 1024)]);
        let shape = PrefillShape::from_scheduled(&metrics).expect("prefill shape");
        let scale = PrefillDeltaModel::work_proportional().correction_scale(&metrics);
        assert!(scale > 1.0, "imbalance must cost more than uniform");
        assert!((scale - (1.0 + shape.relative_work_delta())).abs() < 1e-12);
    }

    #[test]
    fn beta_interpolates_between_points_and_clamps_outside() {
        let model = PrefillDeltaModel::from_points([
            PrefillDeltaPoint {
                depth: 67_584.0,
                beta: 1.10,
            },
            PrefillDeltaPoint {
                depth: 99_328.0,
                beta: 0.50,
            },
        ]);
        assert!((model.beta_at(67_584.0) - 1.10).abs() < 1e-12);
        assert!((model.beta_at(99_328.0) - 0.50).abs() < 1e-12);
        assert!((model.beta_at(83_456.0) - 0.80).abs() < 1e-9);
        // Clamped, not extrapolated.
        assert!((model.beta_at(1_000.0) - 1.10).abs() < 1e-12);
        assert!((model.beta_at(500_000.0) - 0.50).abs() < 1e-12);
    }

    #[test]
    fn points_are_sorted_and_deduplicated_by_depth() {
        let model = PrefillDeltaModel::from_points([
            PrefillDeltaPoint {
                depth: 99_328.0,
                beta: 0.50,
            },
            PrefillDeltaPoint {
                depth: 67_584.0,
                beta: 0.90,
            },
            PrefillDeltaPoint {
                depth: 67_584.0,
                beta: 1.10,
            },
            PrefillDeltaPoint {
                depth: f64::NAN,
                beta: 1.0,
            },
        ]);
        assert!((model.beta_at(67_584.0) - 1.10).abs() < 1e-12);
        assert!((model.beta_at(0.0) - 1.10).abs() < 1e-12);
    }

    #[test]
    fn anti_correlated_batch_is_discounted_but_stays_positive() {
        // Long new-token requests paired with short prefixes and vice versa
        // drives Cov(s, p) negative.
        let rows = [(8192u32, 128u32), (1024, 8192)];
        let shape = PrefillShape::from_scheduled(&from_rows(&rows)).expect("prefill shape");
        let scale = PrefillDeltaModel::work_proportional().correction_scale(&from_rows(&rows));
        assert!(shape.work_delta.is_finite());
        assert!(scale > 0.0, "scale must stay positive, got {scale}");
        assert!(scale >= MIN_CORRECTION_SCALE);
    }

    #[test]
    fn decode_only_batch_has_no_prefill_shape() {
        let mut metrics = ScheduledRequestMetrics::default();
        metrics.num_decode_requests = 16;
        metrics.sum_decode_kv_tokens = 65_536;
        assert!(PrefillShape::from_scheduled(&metrics).is_none());
        assert_eq!(
            PrefillDeltaModel::work_proportional().correction_scale(&metrics),
            1.0
        );
    }
    #[test]
    fn dense_family_ignores_topk_and_always_applies() {
        let metrics = from_rows(&[(5312, 1024), (5312, 1024), (2880, 1024), (2880, 1024)]);
        let model = PrefillDeltaModel::work_proportional();
        assert_eq!(model.regime(&metrics), PrefillRegime::Dense);
        assert!(model.correction_scale(&metrics) > 1.0);
    }

    #[test]
    fn saturated_batch_is_classified_and_priced() {
        // Every request reads more KV than topk: the indexer gate is on for
        // all of them, so the gated trapezoid IS the plain trapezoid.
        let rows = [(512u32, 4096u32), (512, 4096), (256, 3072), (256, 3072)];
        let metrics = from_rows(&rows);
        let model = PrefillDeltaModel::work_proportional()
            .with_family(AttentionFamily::SparseIndexer { topk: 2048 });
        assert_eq!(model.regime(&metrics), PrefillRegime::Saturated);
        assert_ne!(model.correction_scale(&metrics), 1.0);
    }

    #[test]
    fn unsaturated_batch_is_classified_and_priced() {
        // Every full prompt still fits under topk: the gate is off everywhere
        // and the capped-pairs term reduces to the plain trapezoid.
        let rows = [(600u32, 400u32), (600, 400), (300, 200), (300, 200)];
        let metrics = from_rows(&rows);
        let model = PrefillDeltaModel::work_proportional()
            .with_family(AttentionFamily::SparseIndexer { topk: 2048 });
        assert_eq!(model.regime(&metrics), PrefillRegime::Unsaturated);
        assert_ne!(model.correction_scale(&metrics), 1.0);
    }

    #[test]
    fn mixed_batch_declines_the_correction() {
        // One request sits above topk, another entirely below. Which side each
        // request falls on is not recoverable from moments, so the batch is
        // priced as uniform instead of guessed.
        let rows = [(512u32, 4096u32), (512, 4096), (300, 200), (300, 200)];
        let metrics = from_rows(&rows);
        let model = PrefillDeltaModel::work_proportional()
            .with_family(AttentionFamily::SparseIndexer { topk: 2048 });
        assert_eq!(model.regime(&metrics), PrefillRegime::Mixed);
        assert_eq!(model.correction_scale(&metrics), 1.0);
        // The shape itself is still well defined; only the pricing declines.
        assert!(PrefillShape::from_scheduled(&metrics).unwrap().work_delta != 0.0);
    }

    #[test]
    fn sparse_without_order_statistics_declines() {
        // A v1 emitter reports the variances but not min/max, so the regime is
        // unknown and a sparse model must not guess.
        let mut metrics = from_rows(&[(5312, 1024), (5312, 1024), (2880, 1024), (2880, 1024)]);
        metrics.min_prefill_kv_tokens = 0;
        metrics.max_prefill_length = 0;
        let model = PrefillDeltaModel::work_proportional()
            .with_family(AttentionFamily::SparseIndexer { topk: 2048 });
        assert_eq!(model.regime(&metrics), PrefillRegime::Unknown);
        assert_eq!(model.correction_scale(&metrics), 1.0);
        // The same batch under full attention is still priced.
        assert!(PrefillDeltaModel::work_proportional().correction_scale(&metrics) > 1.0);
    }

    #[test]
    fn regime_uses_extremes_not_the_mean() {
        // Mean KV read is above topk, but one request is below it: saturated
        // by the mean, mixed by the criterion that actually matters.
        let rows = [(256u32, 6144u32), (256, 6144), (256, 1024), (256, 1024)];
        let metrics = from_rows(&rows);
        let mean_kv = f64::from(metrics.sum_prefill_kv_tokens) / 4.0;
        assert!(mean_kv > 2048.0, "mean {mean_kv} should look saturated");
        assert_eq!(
            PrefillRegime::classify(AttentionFamily::SparseIndexer { topk: 2048 }, &metrics),
            PrefillRegime::Mixed
        );
    }

    const SPARSE: AttentionFamily = AttentionFamily::SparseIndexer { topk: 2048 };

    /// The exact configuration that corrupted the first GLM-5 campaign:
    /// p_bar == topk makes the anchor saturated, and any rung on the KV axis
    /// drops its low group below topk.
    #[test]
    fn rung_that_crosses_topk_is_rejected() {
        let anchor = from_rows(&[(4096u32, 2048u32); 4]);
        let rung = from_rows(&[(4096, 3072), (4096, 3072), (4096, 1024), (4096, 1024)]);
        assert_eq!(PrefillRegime::classify(SPARSE, &anchor), PrefillRegime::Saturated);
        assert_eq!(PrefillRegime::classify(SPARSE, &rung), PrefillRegime::Mixed);
        let pair = CalibrationPair {
            anchor: &anchor,
            anchor_wall_ms: 1000.0,
            rung: &rung,
            rung_wall_ms: 1050.0,
            anchor_attention_ms: 500.0,
        };
        assert_eq!(pair.solve(SPARSE), Err(CalibrationRejection::RegimeMismatch));
        // Under full attention there is no topk to cross, but this rung is
        // still unusable for a different reason: it perturbs only the KV axis,
        // and the trapezoid is LINEAR in p at fixed s, so the conserved sum
        // annihilates the delta. A KV-axis rung carries no full-attention
        // signal by construction -- it is only informative for the sparse
        // terms it was designed to isolate.
        assert_eq!(
            pair.solve(AttentionFamily::Full),
            Err(CalibrationRejection::RungHasNoDelta)
        );
    }

    #[test]
    fn drifting_totals_are_rejected() {
        let anchor = from_rows(&[(4096u32, 1024u32); 4]);
        // One extra token in the total: no longer the same reference batch.
        let rung = from_rows(&[(5312, 1024), (5312, 1024), (2880, 1024), (2881, 1024)]);
        let pair = CalibrationPair {
            anchor: &anchor,
            anchor_wall_ms: 1000.0,
            rung: &rung,
            rung_wall_ms: 1050.0,
            anchor_attention_ms: 500.0,
        };
        assert_eq!(pair.solve(AttentionFamily::Full), Err(CalibrationRejection::TotalsNotConserved));
    }

    #[test]
    fn flattened_rung_is_rejected() {
        // Block quantisation collapsed the perturbation: the "rung" is a
        // second anchor, and dividing by its zero delta would explode.
        let anchor = from_rows(&[(4096u32, 1024u32); 4]);
        let rung = from_rows(&[(4096u32, 1024u32); 4]);
        let pair = CalibrationPair {
            anchor: &anchor,
            anchor_wall_ms: 1000.0,
            rung: &rung,
            rung_wall_ms: 1001.0,
            anchor_attention_ms: 500.0,
        };
        assert_eq!(pair.solve(AttentionFamily::Full), Err(CalibrationRejection::RungHasNoDelta));
    }

    #[test]
    fn non_uniform_anchor_is_rejected() {
        let anchor = from_rows(&[(5312u32, 1024u32), (5312, 1024), (2880, 1024), (2880, 1024)]);
        let rung = from_rows(&[(6144u32, 1024u32), (6144, 1024), (2048, 1024), (2048, 1024)]);
        let pair = CalibrationPair {
            anchor: &anchor,
            anchor_wall_ms: 1000.0,
            rung: &rung,
            rung_wall_ms: 1050.0,
            anchor_attention_ms: 500.0,
        };
        assert_eq!(pair.solve(AttentionFamily::Full), Err(CalibrationRejection::AnchorNotUniform));
    }

    #[test]
    fn v1_metrics_cannot_calibrate_a_sparse_model() {
        let mut anchor = from_rows(&[(4096u32, 1024u32); 4]);
        let mut rung = from_rows(&[(5312, 1024), (5312, 1024), (2880, 1024), (2880, 1024)]);
        for m in [&mut anchor, &mut rung] {
            m.min_prefill_kv_tokens = 0;
            m.max_prefill_length = 0;
        }
        let pair = CalibrationPair {
            anchor: &anchor,
            anchor_wall_ms: 1000.0,
            rung: &rung,
            rung_wall_ms: 1050.0,
            anchor_attention_ms: 500.0,
        };
        assert_eq!(pair.solve(SPARSE), Err(CalibrationRejection::RegimeUnknown));
    }

    #[test]
    fn a_clean_pair_solves_and_round_trips() {
        let anchor = from_rows(&[(4096u32, 1024u32); 4]);
        let rung = from_rows(&[(5312, 1024), (5312, 1024), (2880, 1024), (2880, 1024)]);
        let pair = CalibrationPair {
            anchor: &anchor,
            anchor_wall_ms: 1000.0,
            rung: &rung,
            rung_wall_ms: 1050.0,
            anchor_attention_ms: 500.0,
        };
        let point = pair.solve(AttentionFamily::Full).expect("clean pair");
        assert_eq!(point.depth, 1024.0 + 2048.0);
        // Feeding the solved beta back reproduces the measured latency.
        let model = PrefillDeltaModel::from_points([point]);
        let scale = model.correction_scale(&rung);
        let predicted = 1000.0 - 500.0 + 500.0 * scale;
        assert!(
            (predicted - 1050.0).abs() < 1e-6,
            "round trip gave {predicted}, expected 1050"
        );
    }

    #[test]
    fn a_barely_uneven_batch_is_left_alone() {
        // Work-delta pricing linearises latency around the mean point, but at
        // a fixed work scalar the measured latency spans a range: the same
        // work packed differently schedules differently. A perturbation
        // smaller than that spread carries no reliable sign, so it is not
        // priced at all.
        let rows = [(4096u32, 1024u32), (4096, 1024), (4160, 1024), (4032, 1024)];
        let metrics = from_rows(&rows);
        let shape = PrefillShape::from_scheduled(&metrics).expect("shape");
        assert!(shape.work_delta > 0.0, "the batch really is uneven");
        assert!(
            shape.relative_work_delta() < MIN_PRICED_RELATIVE_DELTA,
            "and only barely so: {}",
            shape.relative_work_delta()
        );
        assert_eq!(
            PrefillDeltaModel::work_proportional().correction_scale(&metrics),
            1.0
        );
    }

    #[test]
    fn a_clearly_uneven_batch_is_still_priced() {
        let rows = [(6144u32, 1024u32), (6144, 1024), (2048, 1024), (2048, 1024)];
        let metrics = from_rows(&rows);
        let shape = PrefillShape::from_scheduled(&metrics).expect("shape");
        assert!(shape.relative_work_delta() >= MIN_PRICED_RELATIVE_DELTA);
        assert!(PrefillDeltaModel::work_proportional().correction_scale(&metrics) > 1.0);
    }

}
