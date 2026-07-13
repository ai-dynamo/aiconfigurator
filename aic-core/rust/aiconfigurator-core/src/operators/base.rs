// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared infrastructure for the operator layer.
//!
//! Mirrors `aiconfigurator.sdk.performance_result.PerformanceResult` and the
//! `Operation` base class. Each per-family operator (`operators/gemm.rs`
//! etc.) owns its own struct with config-time parameters and a `query`
//! method that takes a `&PerfDatabase` plus its runtime args and returns
//! `PerformanceResult`.
//!
//! No unifying `Operator` trait yet — the per-op signatures diverge enough
//! that polymorphic dispatch would just add a wrapper layer with no
//! callers. Models compose typed ops directly; the session loop matches
//! on the operator kind when it needs to.

/// Source attribution for a latency result.
///
/// Mirrors Python's `result.source` string field. `Silicon` is used for
/// values derived from real collected data (incl. interpolation /
/// extrapolation); `Empirical` for SOL-anchored formula fallbacks;
/// `Sol` for pure speed-of-light estimates; `Mixed` when combining
/// values from different sources within one operator.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Source {
    #[default]
    Silicon,
    Empirical,
    Sol,
    /// Composed from measured pieces plus modeled deltas (Python's
    /// `source="estimated"`, e.g. the DSA CP prefill composition).
    Estimated,
    Mixed,
}

impl Source {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Silicon => "silicon",
            Self::Empirical => "empirical",
            Self::Sol => "sol",
            Self::Estimated => "estimated",
            Self::Mixed => "mixed",
        }
    }

    /// Combine two sources after an additive composition. Returns
    /// `Mixed` when the sources differ.
    pub fn combine(self, other: Source) -> Source {
        if self == other {
            self
        } else {
            Source::Mixed
        }
    }
}

/// Latency result returned by every operator query.
///
/// Energy / power fields are intentionally not modeled here; the Python
/// SDK does not expose them either, and adding them would require new
/// collected data we don't have today.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PerformanceResult {
    pub latency_ms: f64,
    pub source: Source,
}

impl PerformanceResult {
    pub fn new(latency_ms: f64, source: Source) -> Self {
        Self { latency_ms, source }
    }

    /// Convenience constructor — `Source::Silicon` is the most common case
    /// for SILICON-mode queries.
    pub fn silicon(latency_ms: f64) -> Self {
        Self::new(latency_ms, Source::Silicon)
    }

    pub fn zero() -> Self {
        Self::default()
    }

    /// Multiply the latency by `factor`, preserving the source tag.
    pub fn scaled(self, factor: f64) -> Self {
        Self {
            latency_ms: self.latency_ms * factor,
            source: self.source,
        }
    }

    /// Clamp latency to `>= 0` (sub-op subtraction can go negative when
    /// interpolation overshoots; the Python code clamps the same way).
    pub fn clamp_non_negative(self) -> Self {
        Self {
            latency_ms: self.latency_ms.max(0.0),
            source: self.source,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_default_is_silicon() {
        assert_eq!(Source::default(), Source::Silicon);
    }

    #[test]
    fn source_combine_same_keeps_tag() {
        assert_eq!(Source::Silicon.combine(Source::Silicon), Source::Silicon);
        assert_eq!(Source::Sol.combine(Source::Sol), Source::Sol);
    }

    #[test]
    fn source_combine_different_yields_mixed() {
        assert_eq!(Source::Silicon.combine(Source::Empirical), Source::Mixed);
        assert_eq!(Source::Sol.combine(Source::Silicon), Source::Mixed);
    }

    #[test]
    fn performance_result_scaled() {
        let r = PerformanceResult::silicon(10.0).scaled(0.5);
        assert_eq!(r.latency_ms, 5.0);
        assert_eq!(r.source, Source::Silicon);
    }

    #[test]
    fn performance_result_clamp_non_negative() {
        let r = PerformanceResult::silicon(-1.5).clamp_non_negative();
        assert_eq!(r.latency_ms, 0.0);
    }
}
