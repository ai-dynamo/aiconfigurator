// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Immutable one-axis curves used by performance tables.

use std::collections::BTreeMap;

use super::perf_interp::LeafValue;
use crate::common::error::AicError;

pub(crate) trait AxisCoordinate: Copy + Ord {
    fn as_f64(self) -> f64;

    fn exact(value: f64) -> Option<Self>;
}

impl AxisCoordinate for u32 {
    fn as_f64(self) -> f64 {
        f64::from(self)
    }

    fn exact(value: f64) -> Option<Self> {
        if value >= 0.0 && value <= f64::from(u32::MAX) && value.fract() == 0.0 {
            Some(value as u32)
        } else {
            None
        }
    }
}

impl AxisCoordinate for u64 {
    fn as_f64(self) -> f64 {
        self as f64
    }

    fn exact(value: f64) -> Option<Self> {
        // `u64::MAX as f64` rounds to 2^64, so keep the upper bound exclusive.
        if (0.0..18_446_744_073_709_551_616.0).contains(&value) && value.fract() == 0.0 {
            Some(value as u64)
        } else {
            None
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct AxisCurve<K = u32> {
    axis_label: &'static str,
    points: Box<[(K, f64)]>,
}

impl Default for AxisCurve<u32> {
    fn default() -> Self {
        Self::from_map("num_tokens", BTreeMap::new())
    }
}

impl<K: AxisCoordinate> AxisCurve<K> {
    pub(crate) fn from_map(axis_label: &'static str, points: BTreeMap<K, f64>) -> Self {
        Self {
            axis_label,
            points: points.into_iter().collect(),
        }
    }

    pub(crate) fn from_sorted_iter(
        axis_label: &'static str,
        points: impl IntoIterator<Item = (K, f64)>,
    ) -> Self {
        let points: Vec<_> = points.into_iter().collect();
        assert!(
            points.windows(2).all(|pair| pair[0].0 < pair[1].0),
            "AxisCurve points must be strictly ascending and unique"
        );
        Self {
            axis_label,
            points: points.into_boxed_slice(),
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    pub(crate) fn get(&self, coordinate: K) -> Option<f64> {
        self.points
            .binary_search_by_key(&coordinate, |&(coordinate, _)| coordinate)
            .ok()
            .map(|index| self.points[index].1)
    }

    pub(crate) fn iter(&self) -> impl DoubleEndedIterator<Item = (K, f64)> + '_ {
        self.points.iter().copied()
    }

    pub(crate) fn singleton_underflow(&self, coordinate: K) -> Option<K> {
        if self.points.len() == 1 && coordinate < self.points[0].0 {
            Some(self.points[0].0)
        } else {
            None
        }
    }

    /// Resolve with the one-axis `perf_interp` Grid contract: exact hits,
    /// raw interpolation within the measured range, and a boundary-util hold
    /// outside it with `k_tail=1`. Keep this in sync with
    /// `perf_interp::grid_hold`; the differential tests below guard the shared
    /// behavior for `u32` curves.
    pub(crate) fn query(&self, coordinate: f64, sol: &dyn Fn(f64) -> f64) -> Result<f64, AicError> {
        if self.points.is_empty() {
            return Err(self.miss(coordinate, "empty table"));
        }

        if let Some(axis_value) = K::exact(coordinate) {
            if let Some(latency) = self.get(axis_value) {
                return Ok(latency);
            }
        }

        let upper = self
            .points
            .partition_point(|&(axis_value, _)| axis_value.as_f64() < coordinate);
        if upper == 0 || upper == self.points.len() {
            let anchor = if upper == 0 {
                self.points[0]
            } else {
                self.points[self.points.len() - 1]
            };
            let anchor_sol = sol(anchor.0.as_f64());
            if anchor.1.is_nan() || anchor.1 <= 0.0 || anchor_sol.is_nan() || anchor_sol <= 0.0 {
                return Err(self.miss(coordinate, "no positive-util boundary anchor"));
            }
            let query_sol = sol(coordinate);
            if query_sol.is_nan() || query_sol <= 0.0 {
                return Err(self.miss(coordinate, "non-positive SOL at query"));
            }
            return Ok(query_sol / (anchor_sol / anchor.1));
        }

        let lower = self.points[upper - 1];
        let upper = self.points[upper];
        let weight = (coordinate - lower.0.as_f64()) / (upper.0.as_f64() - lower.0.as_f64());
        Ok(lower.1 + (upper.1 - lower.1) * weight)
    }

    fn miss(&self, coordinate: f64, reason: &str) -> AicError {
        AicError::PerfDatabase(format!(
            "perf_interp: no data to anchor query {{{}={coordinate}}} ({reason})",
            self.axis_label
        ))
    }
}

/// Power-carrying twin of [`AxisCurve`]: an immutable one-axis curve over
/// measured `{latency, power, energy}` leaves, specialized away from the
/// generic nested-`Node` engine but preserving `perf_interp::query_value`
/// semantics bit-for-bit (exact hit returns the leaf verbatim; in-range
/// blends lerp latency and blend-power and re-derive
/// `energy = power * latency`; boundary util-holds scale latency by the SOL
/// ratio while power holds at the anchor — "energy scales with latency",
/// mirroring Python `_resolve_tokens`). Shared by the MoE, mHC, MegaMoE and
/// communication tables; the WideEP curve families stay latency-only on
/// [`AxisCurve`] by design (see the wideep module docs).
#[derive(Clone, Debug)]
pub(crate) struct LeafAxisCurve<K = u32> {
    axis_label: &'static str,
    points: Box<[(K, LeafValue)]>,
}

impl Default for LeafAxisCurve<u32> {
    fn default() -> Self {
        Self::from_map("num_tokens", BTreeMap::new())
    }
}

impl<K: AxisCoordinate> LeafAxisCurve<K> {
    pub(crate) fn from_map(axis_label: &'static str, points: BTreeMap<K, LeafValue>) -> Self {
        Self {
            axis_label,
            points: points.into_iter().collect(),
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    pub(crate) fn get(&self, coordinate: K) -> Option<LeafValue> {
        self.points
            .binary_search_by_key(&coordinate, |&(coordinate, _)| coordinate)
            .ok()
            .map(|index| self.points[index].1)
    }

    pub(crate) fn iter(&self) -> impl DoubleEndedIterator<Item = (K, LeafValue)> + '_ {
        self.points.iter().copied()
    }

    /// Python `_require_moe_token_points`: a singleton curve queried below
    /// its only measured point is a structured miss (it cannot define the
    /// low-token launch-overhead regime). Multi-point underflow and
    /// singleton overflow go to the engine's util-hold unchanged.
    pub(crate) fn singleton_underflow(&self, coordinate: K) -> Option<K> {
        if self.points.len() == 1 && coordinate < self.points[0].0 {
            Some(self.points[0].0)
        } else {
            None
        }
    }

    /// Resolve with the one-axis `perf_interp` Grid contract on full
    /// leaves: exact hits verbatim, raw interpolation within the measured
    /// range (latency and blend-power lerped with the same weight, energy
    /// re-derived as `power * latency`), and a boundary-util hold outside
    /// it with `k_tail=1` (latency scales by the SOL ratio; power holds at
    /// the anchor's blend power). Keep this in sync with
    /// `perf_interp::query_value`; the differential tests below guard the
    /// shared behavior.
    pub(crate) fn query(
        &self,
        coordinate: f64,
        sol: &dyn Fn(f64) -> f64,
    ) -> Result<LeafValue, AicError> {
        if self.points.is_empty() {
            return Err(self.miss(coordinate, "empty table"));
        }

        if let Some(axis_value) = K::exact(coordinate) {
            if let Some(leaf) = self.get(axis_value) {
                return Ok(leaf);
            }
        }

        let upper = self
            .points
            .partition_point(|&(axis_value, _)| axis_value.as_f64() < coordinate);
        if upper == 0 || upper == self.points.len() {
            let anchor = if upper == 0 {
                self.points[0]
            } else {
                self.points[self.points.len() - 1]
            };
            let anchor_sol = sol(anchor.0.as_f64());
            if anchor.1.latency.is_nan()
                || anchor.1.latency <= 0.0
                || anchor_sol.is_nan()
                || anchor_sol <= 0.0
            {
                return Err(self.miss(coordinate, "no positive-util boundary anchor"));
            }
            let query_sol = sol(coordinate);
            if query_sol.is_nan() || query_sol <= 0.0 {
                return Err(self.miss(coordinate, "non-positive SOL at query"));
            }
            let latency = query_sol / (anchor_sol / anchor.1.latency);
            let power = anchor.1.blend_power();
            return Ok(LeafValue {
                latency,
                power,
                energy: power * latency,
            });
        }

        let lower = self.points[upper - 1];
        let upper = self.points[upper];
        let weight = (coordinate - lower.0.as_f64()) / (upper.0.as_f64() - lower.0.as_f64());
        let latency = lower.1.latency + (upper.1.latency - lower.1.latency) * weight;
        let lower_power = lower.1.blend_power();
        let power = lower_power + (upper.1.blend_power() - lower_power) * weight;
        Ok(LeafValue {
            latency,
            power,
            energy: power * latency,
        })
    }

    fn miss(&self, coordinate: f64, reason: &str) -> AicError {
        AicError::PerfDatabase(format!(
            "perf_interp: no data to anchor query {{{}={coordinate}}} ({reason})",
            self.axis_label
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perf_database::perf_interp::{self, Node, OpInterpConfig};

    #[test]
    fn axis_curve_resolves_exact_interpolated_and_held_values() {
        let curve = AxisCurve::from_map("num_tokens", BTreeMap::from([(10_u32, 1.0), (20, 3.0)]));
        assert_eq!(curve.query(10.0, &|tokens| tokens).unwrap(), 1.0);
        assert_eq!(curve.query(15.0, &|tokens| tokens).unwrap(), 2.0);
        assert_eq!(curve.query(40.0, &|tokens| tokens).unwrap(), 6.0);
        assert_eq!(curve.query(5.0, &|tokens| tokens).unwrap(), 0.5);
    }

    #[test]
    fn axis_curve_preserves_singleton_and_invalid_hold_semantics() {
        let singleton = AxisCurve::from_map("num_tokens", BTreeMap::from([(10_u32, 0.0)]));
        assert_eq!(singleton.singleton_underflow(9), Some(10));
        assert!(singleton.query(20.0, &|tokens| tokens).is_err());
    }

    #[test]
    fn axis_curve_is_bit_exact_with_the_generic_grid() {
        let sol = |tokens: f64| tokens * tokens + 1.0;
        for stride in [1_u32, 3, 17, 257] {
            let points =
                BTreeMap::from([(10 * stride, 1.25), (20 * stride, 2.75), (40 * stride, 5.5)]);
            for tokens in [
                5.0 * f64::from(stride),
                10.0 * f64::from(stride),
                15.5 * f64::from(stride),
                20.0 * f64::from(stride),
                31.0 * f64::from(stride),
                40.0 * f64::from(stride),
                80.0 * f64::from(stride),
            ] {
                assert_query_parity(points.clone(), tokens, &sol);
            }
        }
    }

    fn assert_query_parity(points: BTreeMap<u32, f64>, num_tokens: f64, sol: &dyn Fn(f64) -> f64) {
        let curve = AxisCurve::from_map("num_tokens", points.clone());
        let mut node = Node::branch();
        for (&token, &latency) in &points {
            node.insert(&[token], latency);
        }
        let generic_sol = |coords: &[f64]| sol(coords[0]);
        let config = OpInterpConfig::grid(&["num_tokens"], &generic_sol);
        let expected = perf_interp::query(&config, &node, &[num_tokens]);
        let actual = curve.query(num_tokens, sol);
        match (actual, expected) {
            (Ok(actual), Ok(expected)) => assert_eq!(actual.to_bits(), expected.to_bits()),
            (Err(actual), Err(expected)) => assert_eq!(actual.to_string(), expected.to_string()),
            (actual, expected) => panic!("specialized={actual:?}, generic={expected:?}"),
        }
    }

    #[test]
    fn axis_curve_errors_match_the_generic_grid() {
        assert_query_parity(BTreeMap::new(), 10.0, &|tokens| tokens);
        assert_query_parity(BTreeMap::from([(10, 0.0)]), 20.0, &|tokens| tokens);
        assert_query_parity(BTreeMap::from([(10, 1.0)]), 20.0, &|tokens| {
            if tokens == 20.0 {
                0.0
            } else {
                tokens
            }
        });
        assert_query_parity(BTreeMap::from([(10, 1.0)]), 20.0, &|tokens| {
            if tokens == 20.0 {
                f64::NAN
            } else {
                tokens
            }
        });
    }

    #[test]
    fn axis_curve_uses_its_axis_label_for_reachable_misses() {
        let curve = AxisCurve::from_map("message_bytes", BTreeMap::from([(1024_u64, 0.0)]));
        let err = curve.query(2048.0, &|bytes| bytes).unwrap_err();
        assert_eq!(
            err.to_string(),
            "perf database error: perf_interp: no data to anchor query \
             {message_bytes=2048} (no positive-util boundary anchor)"
        );
    }

    #[test]
    #[should_panic(expected = "AxisCurve points must be strictly ascending and unique")]
    fn axis_curve_rejects_unsorted_points() {
        AxisCurve::from_sorted_iter("num_tokens", [(2_u32, 2.0), (1, 1.0)]);
    }

    #[test]
    #[should_panic(expected = "AxisCurve points must be strictly ascending and unique")]
    fn axis_curve_rejects_duplicate_points() {
        AxisCurve::from_sorted_iter("num_tokens", [(1_u32, 1.0), (1, 2.0)]);
    }

    /// The specialized leaf curve must be indistinguishable from the generic
    /// engine's `query_value` on power-carrying leaves — bit-exact
    /// latency/power/energy on exact hits, interior lerps, and both
    /// boundary util-holds, and identical error strings on the miss paths
    /// (the LeafValue twin of `axis_curve_is_bit_exact_with_the_generic_grid`).
    #[test]
    fn leaf_axis_curve_is_bit_exact_with_the_generic_engine() {
        let points = BTreeMap::from([
            (10, LeafValue::with_power(1.25, 100.0)),
            (20, LeafValue::with_power(2.75, 150.0)),
            (40, LeafValue::with_power(5.5, 275.0)),
        ]);
        let curve = LeafAxisCurve::from_map("num_tokens", points.clone());
        let mut node = Node::branch();
        for (&token, &leaf) in &points {
            node.insert_value(&[token], leaf);
        }
        let sol = |tokens: f64| tokens * tokens + 1.0;
        let generic_sol = |coords: &[f64]| sol(coords[0]);
        let config = OpInterpConfig::grid(&["num_tokens"], &generic_sol);

        for tokens in [5.0, 10.0, 15.5, 20.0, 31.0, 40.0, 80.0] {
            let expected = perf_interp::query_value(&config, &node, &[tokens]).unwrap();
            let actual = curve.query(tokens, &sol).unwrap();
            for (name, actual, expected) in [
                ("latency", actual.latency, expected.latency),
                ("power", actual.power, expected.power),
                ("energy", actual.energy, expected.energy),
            ] {
                assert_eq!(
                    actual.to_bits(),
                    expected.to_bits(),
                    "tokens={tokens}, field={name}"
                );
            }
        }

        // Miss paths: an empty curve and a non-positive SOL hold must
        // produce the exact generic-engine error strings.
        let empty = LeafAxisCurve::default();
        let empty_node = Node::branch();
        assert_eq!(
            empty.query(10.0, &sol).unwrap_err().to_string(),
            perf_interp::query_value(&config, &empty_node, &[10.0])
                .unwrap_err()
                .to_string()
        );
        let zero_sol = |_: f64| 0.0;
        let generic_zero_sol = |_: &[f64]| 0.0;
        let zero_config = OpInterpConfig::grid(&["num_tokens"], &generic_zero_sol);
        assert_eq!(
            curve.query(80.0, &zero_sol).unwrap_err().to_string(),
            perf_interp::query_value(&zero_config, &node, &[80.0])
                .unwrap_err()
                .to_string()
        );
    }

    #[test]
    fn leaf_axis_curve_uses_its_axis_label_and_u64_keys() {
        let curve = LeafAxisCurve::from_map(
            "message_bytes",
            BTreeMap::from([(1024_u64, LeafValue::latency_only(0.0))]),
        );
        assert_eq!(curve.singleton_underflow(512), Some(1024));
        let err = curve.query(2048.0, &|bytes| bytes).unwrap_err();
        assert_eq!(
            err.to_string(),
            "perf database error: perf_interp: no data to anchor query \
             {message_bytes=2048} (no positive-util boundary anchor)"
        );
    }
}
