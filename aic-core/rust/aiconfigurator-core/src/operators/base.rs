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

/// SOL roofline decomposition of a `Source::Sol` latency.
///
/// Mirrors the `(sol_math, sol_mem)` tail of Python's SOL_FULL triple
/// (`get_sol` returns `(sol_time, sol_math, sol_mem)`; `sol_time` is the
/// result's latency). Compute-bound time and memory-bound time in ms; the
/// leaf latency is their max, but composed results (sums, scale factors)
/// keep the components additive, so `max(math_ms, mem_ms)` only equals the
/// latency at the leaf.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SolComponents {
    pub math_ms: f64,
    pub mem_ms: f64,
}

impl SolComponents {
    pub fn new(math_ms: f64, mem_ms: f64) -> Self {
        Self { math_ms, mem_ms }
    }

    /// Leaf SOL latency: `max(sol_math, sol_mem)` (Python `sol_time`).
    pub fn time_ms(self) -> f64 {
        self.math_ms.max(self.mem_ms)
    }
}

/// Actual MoE communication topology substitution used by one query.
///
/// Attached only after the measurement lookup succeeds. The inference phase
/// (context or generation) is supplied by the engine loop that evaluates the
/// operator; this payload owns the remaining lookup provenance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MoeCommFallback {
    pub comm_backend: &'static str,
    pub requested_ep_size: u32,
    pub requested_node_num: u32,
    pub measurement_ep_size: u32,
    pub measurement_node_num: u32,
}

/// Ordered, de-duplicated fallback records carried by one composed result.
///
/// The common zero- and one-record cases do not allocate. A heap allocation
/// is needed only when a composition executes two or more distinct topology
/// substitutions.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MoeCommFallbacks {
    first: Option<MoeCommFallback>,
    additional: Vec<MoeCommFallback>,
}

impl MoeCommFallbacks {
    pub fn is_empty(&self) -> bool {
        self.first.is_none()
    }

    pub fn iter(&self) -> impl Iterator<Item = &MoeCommFallback> {
        self.first.iter().chain(self.additional.iter())
    }

    fn insert(&mut self, fallback: MoeCommFallback) {
        if self.iter().any(|existing| *existing == fallback) {
            return;
        }
        if self.first.is_none() {
            self.first = Some(fallback);
        } else {
            self.additional.push(fallback);
        }
    }

    fn extend(&mut self, other: Self) {
        for fallback in other.iter().copied() {
            self.insert(fallback);
        }
    }
}

/// Componentwise subtraction for optional SOL decompositions (the GEMM
/// fp8_static overhead-table subtraction). Either side missing → `None`:
/// an incomplete breakdown must not masquerade as a full one.
pub(crate) fn subtract_sol(
    a: Option<SolComponents>,
    b: Option<SolComponents>,
) -> Option<SolComponents> {
    match (a, b) {
        (Some(a), Some(b)) => Some(SolComponents::new(
            a.math_ms - b.math_ms,
            a.mem_ms - b.mem_ms,
        )),
        _ => None,
    }
}

/// Componentwise weighted blend `w*a + (1-w)*b` for optional SOL
/// decompositions (the GLM-5.2 DSA full/skip shared-index amortization).
/// Either side missing → `None`.
pub(crate) fn blend_sol(
    w: f64,
    a: Option<SolComponents>,
    b: Option<SolComponents>,
) -> Option<SolComponents> {
    match (a, b) {
        (Some(a), Some(b)) => Some(SolComponents::new(
            w * a.math_ms + (1.0 - w) * b.math_ms,
            w * a.mem_ms + (1.0 - w) * b.mem_ms,
        )),
        _ => None,
    }
}

/// Latency + energy result returned by every operator query.
///
/// Mirrors Python's `PerformanceResult`: the float value is latency in ms
/// and `energy_wms` rides along in watt-milliseconds (0.0 for tables that
/// carry no power data and for empirical / SOL fallbacks, exactly like the
/// Python paths that construct results without an energy argument).
///
/// `sol` carries the SOL roofline decomposition when the value was computed
/// under `DatabaseMode::Sol`/`SolFull` by a family whose SOL path exports
/// its components (the notebook re-oracle FFI reads them); `None` everywhere
/// else. It rides along through `scaled`/`plus`/`clamp_non_negative` so op-
/// level composition (scale factors, additive modules) stays consistent
/// with the latency. `moe_comm_fallbacks` identifies successful substitute
/// topology lookups and follows the same combinators as an ordered,
/// de-duplicated collection.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PerformanceResult {
    pub latency_ms: f64,
    pub energy_wms: f64,
    pub source: Source,
    pub sol: Option<SolComponents>,
    pub moe_comm_fallbacks: MoeCommFallbacks,
}

impl PerformanceResult {
    pub fn new(latency_ms: f64, source: Source) -> Self {
        Self {
            latency_ms,
            energy_wms: 0.0,
            source,
            sol: None,
            moe_comm_fallbacks: MoeCommFallbacks::default(),
        }
    }

    pub fn with_energy(latency_ms: f64, energy_wms: f64, source: Source) -> Self {
        Self {
            latency_ms,
            energy_wms,
            source,
            sol: None,
            moe_comm_fallbacks: MoeCommFallbacks::default(),
        }
    }

    /// Leaf SOL result: latency = `max(math_ms, mem_ms)` (Python
    /// `sol_time`), `Source::Sol`, zero energy, components attached.
    pub fn sol(components: SolComponents) -> Self {
        Self {
            latency_ms: components.time_ms(),
            energy_wms: 0.0,
            source: Source::Sol,
            sol: Some(components),
            moe_comm_fallbacks: MoeCommFallbacks::default(),
        }
    }

    /// Attach (or replace) the SOL decomposition, keeping everything else.
    /// For SOL leaves whose latency is NOT the plain `max(math, mem)`
    /// (pure-bandwidth comm bounds, composed module SOLs).
    pub fn with_sol(mut self, components: SolComponents) -> Self {
        self.sol = Some(components);
        self
    }

    /// Attach an executed MoE communication topology substitution.
    pub fn with_moe_comm_fallback(mut self, fallback: MoeCommFallback) -> Self {
        self.moe_comm_fallbacks.insert(fallback);
        self
    }

    /// Attach every executed substitution from a composed result.
    pub fn with_moe_comm_fallbacks(mut self, fallbacks: MoeCommFallbacks) -> Self {
        self.moe_comm_fallbacks.extend(fallbacks);
        self
    }

    /// Convenience constructor — `Source::Silicon` is the most common case
    /// for SILICON-mode queries.
    pub fn silicon(latency_ms: f64) -> Self {
        Self::new(latency_ms, Source::Silicon)
    }

    pub fn zero() -> Self {
        Self::default()
    }

    /// Multiply latency AND energy by `factor`, preserving the source tag
    /// (Python `__mul__` / `__truediv__` scale energy the same way). SOL
    /// components scale with the latency they decompose.
    pub fn scaled(self, factor: f64) -> Self {
        Self {
            latency_ms: self.latency_ms * factor,
            energy_wms: self.energy_wms * factor,
            source: self.source,
            sol: self.sol.map(|c| SolComponents {
                math_ms: c.math_ms * factor,
                mem_ms: c.mem_ms * factor,
            }),
            moe_comm_fallbacks: self.moe_comm_fallbacks,
        }
    }

    /// Additive composition: latencies and energies sum, sources combine
    /// to `Mixed` on mismatch (Python `__add__`). A zero result (latency
    /// AND energy both 0.0) is a source-neutral identity — the other
    /// side's tag survives, mirroring Python's zero-identity rule.
    pub fn plus(self, other: PerformanceResult) -> Self {
        let mut moe_comm_fallbacks = self.moe_comm_fallbacks;
        moe_comm_fallbacks.extend(other.moe_comm_fallbacks);
        let (source, sol) = if self.latency_ms == 0.0 && self.energy_wms == 0.0 {
            (other.source, other.sol)
        } else if other.latency_ms == 0.0 && other.energy_wms == 0.0 {
            (self.source, self.sol)
        } else {
            // Components add only when BOTH sides carry them; a side
            // without a decomposition poisons the sum to `None` (an
            // incomplete breakdown must not masquerade as a full one).
            let sol = match (self.sol, other.sol) {
                (Some(a), Some(b)) => Some(SolComponents {
                    math_ms: a.math_ms + b.math_ms,
                    mem_ms: a.mem_ms + b.mem_ms,
                }),
                _ => None,
            };
            (self.source.combine(other.source), sol)
        };
        Self {
            latency_ms: self.latency_ms + other.latency_ms,
            energy_wms: self.energy_wms + other.energy_wms,
            source,
            sol,
            moe_comm_fallbacks,
        }
    }

    /// Clamp latency and energy to `>= 0` (sub-op subtraction can go
    /// negative when interpolation overshoots; the Python code clamps the
    /// same way).
    pub fn clamp_non_negative(self) -> Self {
        Self {
            latency_ms: self.latency_ms.max(0.0),
            energy_wms: self.energy_wms.max(0.0),
            source: self.source,
            sol: self.sol.map(|c| SolComponents {
                math_ms: c.math_ms.max(0.0),
                mem_ms: c.mem_ms.max(0.0),
            }),
            moe_comm_fallbacks: self.moe_comm_fallbacks,
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

    #[test]
    fn moe_comm_fallbacks_ride_through_result_combinators_without_loss() {
        assert_eq!(
            PerformanceResult::default()
                .moe_comm_fallbacks
                .additional
                .capacity(),
            0
        );
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
        let tagged = PerformanceResult::new(-2.0, Source::Estimated).with_moe_comm_fallback(ht);
        assert_eq!(tagged.moe_comm_fallbacks.additional.capacity(), 0);

        assert_eq!(
            tagged
                .clone()
                .scaled(2.0)
                .moe_comm_fallbacks
                .iter()
                .copied()
                .collect::<Vec<_>>(),
            vec![ht]
        );
        assert_eq!(
            tagged
                .clone()
                .clamp_non_negative()
                .moe_comm_fallbacks
                .iter()
                .copied()
                .collect::<Vec<_>>(),
            vec![ht]
        );
        assert_eq!(
            tagged
                .clone()
                .plus(PerformanceResult::new(1.0, Source::Silicon))
                .moe_comm_fallbacks
                .iter()
                .copied()
                .collect::<Vec<_>>(),
            vec![ht]
        );
        assert_eq!(
            tagged
                .clone()
                .plus(tagged.clone())
                .moe_comm_fallbacks
                .iter()
                .copied()
                .collect::<Vec<_>>(),
            vec![ht]
        );
        assert_eq!(
            tagged
                .plus(PerformanceResult::new(1.0, Source::Estimated).with_moe_comm_fallback(ll))
                .moe_comm_fallbacks
                .iter()
                .copied()
                .collect::<Vec<_>>(),
            vec![ht, ll]
        );
        assert_eq!(
            PerformanceResult::new(0.0, Source::Estimated)
                .with_moe_comm_fallback(ht)
                .plus(PerformanceResult::new(0.0, Source::Estimated).with_moe_comm_fallback(ll))
                .moe_comm_fallbacks
                .iter()
                .copied()
                .collect::<Vec<_>>(),
            vec![ht, ll]
        );
    }

    #[test]
    fn sol_components_ride_through_combinators() {
        // Leaf: latency = max(math, mem), Source::Sol.
        let leaf = PerformanceResult::sol(SolComponents::new(3.0, 5.0));
        assert_eq!(leaf.latency_ms, 5.0);
        assert_eq!(leaf.source, Source::Sol);

        // scaled: components scale with the latency.
        let scaled = leaf.clone().scaled(2.0);
        assert_eq!(scaled.sol, Some(SolComponents::new(6.0, 10.0)));

        // plus: componentwise sum when both sides carry components...
        let sum = leaf
            .clone()
            .plus(PerformanceResult::sol(SolComponents::new(1.0, 0.5)));
        assert_eq!(sum.latency_ms, 6.0);
        assert_eq!(sum.sol, Some(SolComponents::new(4.0, 5.5)));

        // ...poisoned to None when one side has none (incomplete breakdown)...
        let poisoned = leaf.clone().plus(PerformanceResult::new(1.0, Source::Sol));
        assert_eq!(poisoned.sol, None);

        // ...and passed through a zero identity (either side).
        let zero = PerformanceResult::zero();
        assert_eq!(leaf.clone().plus(zero.clone()).sol, leaf.sol);
        assert_eq!(zero.plus(leaf.clone()).sol, leaf.sol);

        // clamp: components clamp to >= 0 alongside the latency.
        let negative = subtract_sol(
            Some(SolComponents::new(1.0, 1.0)),
            Some(SolComponents::new(2.0, 0.5)),
        )
        .unwrap();
        assert_eq!(negative, SolComponents::new(-1.0, 0.5));
        let clamped = PerformanceResult::new(1.0, Source::Sol)
            .with_sol(negative)
            .clamp_non_negative();
        assert_eq!(clamped.sol, Some(SolComponents::new(0.0, 0.5)));

        // subtract_sol: either side missing -> None.
        assert_eq!(subtract_sol(Some(SolComponents::default()), None), None);
        assert_eq!(subtract_sol(None, Some(SolComponents::default())), None);
    }

    #[test]
    fn plus_zero_result_is_source_neutral() {
        // Mirrors Python test_zero_latency_energy_source_is_neutral: a
        // (0.0, 0.0) operand must not force `Mixed`.
        let zero = PerformanceResult::new(0.0, Source::Empirical);
        let real = PerformanceResult::with_energy(2.0, 10.0, Source::Silicon);
        assert_eq!(zero.clone().plus(real.clone()).source, Source::Silicon);
        assert_eq!(real.clone().plus(zero).source, Source::Silicon);
        // Non-zero operands with different tags still merge to Mixed.
        let sol = PerformanceResult::new(1.0, Source::Sol);
        assert_eq!(real.clone().plus(sol).source, Source::Mixed);
        // A zero-latency result that still carries energy is NOT neutral.
        let energetic_zero = PerformanceResult::with_energy(0.0, 5.0, Source::Empirical);
        assert_eq!(real.plus(energetic_zero).source, Source::Mixed);
    }
}
