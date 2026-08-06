// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Immutable one-axis curves used by performance tables.

use std::collections::BTreeMap;

use crate::common::error::AicError;

#[derive(Clone, Debug, Default)]
pub(crate) struct AxisCurve {
    points: Box<[(u32, f64)]>,
}

impl AxisCurve {
    pub(crate) fn from_map(points: BTreeMap<u32, f64>) -> Self {
        Self {
            points: points.into_iter().collect(),
        }
    }

    pub(crate) fn from_sorted_iter(points: impl IntoIterator<Item = (u32, f64)>) -> Self {
        let points: Vec<_> = points.into_iter().collect();
        assert!(
            points.windows(2).all(|pair| pair[0].0 < pair[1].0),
            "AxisCurve points must be strictly ascending and unique"
        );
        Self {
            points: points.into_boxed_slice(),
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    pub(crate) fn get(&self, coordinate: u32) -> Option<f64> {
        self.points
            .binary_search_by_key(&coordinate, |&(coordinate, _)| coordinate)
            .ok()
            .map(|index| self.points[index].1)
    }

    pub(crate) fn iter(&self) -> impl DoubleEndedIterator<Item = (u32, f64)> + '_ {
        self.points.iter().copied()
    }

    pub(crate) fn singleton_underflow(&self, coordinate: u32) -> Option<u32> {
        if self.points.len() == 1 && coordinate < self.points[0].0 {
            Some(self.points[0].0)
        } else {
            None
        }
    }

    /// Resolve with the one-axis `perf_interp` Grid contract: exact hits,
    /// raw interpolation within the measured range, and a boundary-util hold
    /// outside it with `k_tail=1`.
    pub(crate) fn query(
        &self,
        coordinate: f64,
        axis_label: &str,
        sol: &dyn Fn(f64) -> f64,
    ) -> Result<f64, AicError> {
        if self.points.is_empty() {
            return Err(miss(axis_label, coordinate, "empty table"));
        }

        if let Some(axis_value) = exact_coordinate(coordinate) {
            if let Some(latency) = self.get(axis_value) {
                return Ok(latency);
            }
        }

        let upper = self
            .points
            .partition_point(|&(axis_value, _)| f64::from(axis_value) < coordinate);
        if upper == 0 || upper == self.points.len() {
            let anchor = if upper == 0 {
                self.points[0]
            } else {
                self.points[self.points.len() - 1]
            };
            let anchor_sol = sol(f64::from(anchor.0));
            if anchor.1.is_nan() || anchor.1 <= 0.0 || anchor_sol.is_nan() || anchor_sol <= 0.0 {
                return Err(miss(
                    axis_label,
                    coordinate,
                    "no positive-util boundary anchor",
                ));
            }
            let query_sol = sol(coordinate);
            if query_sol.is_nan() || query_sol <= 0.0 {
                return Err(miss(axis_label, coordinate, "non-positive SOL at query"));
            }
            return Ok(query_sol / (anchor_sol / anchor.1));
        }

        let lower = self.points[upper - 1];
        let upper = self.points[upper];
        let weight = (coordinate - f64::from(lower.0)) / f64::from(upper.0 - lower.0);
        Ok(lower.1 + (upper.1 - lower.1) * weight)
    }
}

fn exact_coordinate(value: f64) -> Option<u32> {
    if value >= 0.0 && value <= f64::from(u32::MAX) && value.fract() == 0.0 {
        Some(value as u32)
    } else {
        None
    }
}

fn miss(axis_label: &str, coordinate: f64, reason: &str) -> AicError {
    AicError::PerfDatabase(format!(
        "perf_interp: no data to anchor query {{{axis_label}={coordinate}}} ({reason})"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perf_database::perf_interp::{self, Node, OpInterpConfig};

    #[test]
    fn axis_curve_resolves_exact_interpolated_and_held_values() {
        let curve = AxisCurve::from_map(BTreeMap::from([(10, 1.0), (20, 3.0)]));
        assert_eq!(
            curve.query(10.0, "num_tokens", &|tokens| tokens).unwrap(),
            1.0
        );
        assert_eq!(
            curve.query(15.0, "num_tokens", &|tokens| tokens).unwrap(),
            2.0
        );
        assert_eq!(
            curve.query(40.0, "num_tokens", &|tokens| tokens).unwrap(),
            6.0
        );
        assert_eq!(
            curve.query(5.0, "num_tokens", &|tokens| tokens).unwrap(),
            0.5
        );
    }

    #[test]
    fn axis_curve_preserves_singleton_and_invalid_hold_semantics() {
        let singleton = AxisCurve::from_map(BTreeMap::from([(10, 0.0)]));
        assert_eq!(singleton.singleton_underflow(9), Some(10));
        assert!(singleton
            .query(20.0, "num_tokens", &|tokens| tokens)
            .is_err());
    }

    #[test]
    fn axis_curve_is_bit_exact_with_the_generic_grid() {
        let points = BTreeMap::from([(10, 1.25), (20, 2.75), (40, 5.5)]);
        let curve = AxisCurve::from_map(points.clone());
        let mut node = Node::branch();
        for (&token, &latency) in &points {
            node.insert(&[token], latency);
        }
        let sol = |tokens: f64| tokens * tokens + 1.0;
        let generic_sol = |coords: &[f64]| sol(coords[0]);
        let config = OpInterpConfig::grid(&["num_tokens"], &generic_sol);

        for tokens in [5.0, 10.0, 15.5, 20.0, 31.0, 40.0, 80.0] {
            let expected = perf_interp::query(&config, &node, &[tokens]).unwrap();
            let actual = curve.query(tokens, "num_tokens", &sol).unwrap();
            assert_eq!(actual.to_bits(), expected.to_bits(), "tokens={tokens}");
        }
    }

    fn assert_query_parity(points: BTreeMap<u32, f64>, num_tokens: f64, sol: &dyn Fn(f64) -> f64) {
        let curve = AxisCurve::from_map(points.clone());
        let mut node = Node::branch();
        for (&token, &latency) in &points {
            node.insert(&[token], latency);
        }
        let generic_sol = |coords: &[f64]| sol(coords[0]);
        let config = OpInterpConfig::grid(&["num_tokens"], &generic_sol);
        let expected = perf_interp::query(&config, &node, &[num_tokens]);
        let actual = curve.query(num_tokens, "num_tokens", sol);
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
    fn axis_curve_uses_the_supplied_axis_label() {
        let curve = AxisCurve::default();
        let err = curve
            .query(1024.5, "message_bytes", &|bytes| bytes)
            .unwrap_err();
        assert_eq!(
            err.to_string(),
            "perf database error: perf_interp: no data to anchor query \
             {message_bytes=1024.5} (empty table)"
        );
    }

    #[test]
    #[should_panic(expected = "AxisCurve points must be strictly ascending and unique")]
    fn axis_curve_rejects_unsorted_points() {
        AxisCurve::from_sorted_iter([(2, 2.0), (1, 1.0)]);
    }

    #[test]
    #[should_panic(expected = "AxisCurve points must be strictly ascending and unique")]
    fn axis_curve_rejects_duplicate_points() {
        AxisCurve::from_sorted_iter([(1, 1.0), (1, 2.0)]);
    }
}
