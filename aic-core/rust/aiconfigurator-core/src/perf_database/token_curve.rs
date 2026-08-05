// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Immutable one-axis token curves used by MoE-family perf tables.

use std::collections::BTreeMap;

use crate::common::error::AicError;

#[derive(Clone, Debug, Default)]
pub(crate) struct TokenCurve {
    points: Box<[(u32, f64)]>,
}

impl TokenCurve {
    pub(crate) fn from_map(points: BTreeMap<u32, f64>) -> Self {
        Self {
            points: points.into_iter().collect(),
        }
    }

    pub(crate) fn from_iter(points: impl IntoIterator<Item = (u32, f64)>) -> Self {
        let points: Vec<_> = points.into_iter().collect();
        debug_assert!(points.windows(2).all(|pair| pair[0].0 < pair[1].0));
        Self {
            points: points.into_boxed_slice(),
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    pub(crate) fn get(&self, token: u32) -> Option<f64> {
        self.points
            .binary_search_by_key(&token, |&(token, _)| token)
            .ok()
            .map(|index| self.points[index].1)
    }

    pub(crate) fn iter(&self) -> impl DoubleEndedIterator<Item = (u32, f64)> + '_ {
        self.points.iter().copied()
    }

    pub(crate) fn singleton_underflow(&self, num_tokens: u32) -> Option<u32> {
        if self.points.len() == 1 && num_tokens < self.points[0].0 {
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
        num_tokens: f64,
        sol: &dyn Fn(f64) -> f64,
    ) -> Result<f64, AicError> {
        if self.points.is_empty() {
            return Err(miss(num_tokens, "empty table"));
        }

        if let Some(token) = exact_token(num_tokens) {
            if let Some(latency) = self.get(token) {
                return Ok(latency);
            }
        }

        let upper = self
            .points
            .partition_point(|&(token, _)| f64::from(token) < num_tokens);
        if upper == 0 || upper == self.points.len() {
            let anchor = if upper == 0 {
                self.points[0]
            } else {
                self.points[self.points.len() - 1]
            };
            let anchor_sol = sol(f64::from(anchor.0));
            if !(anchor.1 > 0.0 && anchor_sol > 0.0) {
                return Err(miss(num_tokens, "no positive-util boundary anchor"));
            }
            let query_sol = sol(num_tokens);
            if !(query_sol > 0.0) {
                return Err(miss(num_tokens, "non-positive SOL at query"));
            }
            return Ok(query_sol / (anchor_sol / anchor.1));
        }

        let lower = self.points[upper - 1];
        let upper = self.points[upper];
        let weight =
            (num_tokens - f64::from(lower.0)) / f64::from(upper.0 - lower.0);
        Ok(lower.1 + (upper.1 - lower.1) * weight)
    }
}

fn exact_token(value: f64) -> Option<u32> {
    if value >= 0.0 && value <= f64::from(u32::MAX) && value.fract() == 0.0 {
        Some(value as u32)
    } else {
        None
    }
}

fn miss(num_tokens: f64, reason: &str) -> AicError {
    AicError::PerfDatabase(format!(
        "perf_interp: no data to anchor query {{num_tokens={num_tokens}}} ({reason})"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perf_database::perf_interp::{self, Node, OpInterpConfig};

    #[test]
    fn token_curve_resolves_exact_interpolated_and_held_values() {
        let curve = TokenCurve::from_map(BTreeMap::from([(10, 1.0), (20, 3.0)]));
        assert_eq!(curve.query(10.0, &|tokens| tokens).unwrap(), 1.0);
        assert_eq!(curve.query(15.0, &|tokens| tokens).unwrap(), 2.0);
        assert_eq!(curve.query(40.0, &|tokens| tokens).unwrap(), 6.0);
        assert_eq!(curve.query(5.0, &|tokens| tokens).unwrap(), 0.5);
    }

    #[test]
    fn token_curve_preserves_singleton_and_invalid_hold_semantics() {
        let singleton = TokenCurve::from_map(BTreeMap::from([(10, 0.0)]));
        assert_eq!(singleton.singleton_underflow(9), Some(10));
        assert!(singleton.query(20.0, &|tokens| tokens).is_err());
    }

    #[test]
    fn token_curve_is_bit_exact_with_the_generic_grid() {
        let points = BTreeMap::from([(10, 1.25), (20, 2.75), (40, 5.5)]);
        let curve = TokenCurve::from_map(points.clone());
        let mut node = Node::branch();
        for (&token, &latency) in &points {
            node.insert(&[token], latency);
        }
        let sol = |tokens: f64| tokens * tokens + 1.0;
        let generic_sol = |coords: &[f64]| sol(coords[0]);
        let config = OpInterpConfig::grid(&["num_tokens"], &generic_sol);

        for tokens in [5.0, 10.0, 15.5, 20.0, 31.0, 40.0, 80.0] {
            let expected = perf_interp::query(&config, &node, &[tokens]).unwrap();
            let actual = curve.query(tokens, &sol).unwrap();
            assert_eq!(actual.to_bits(), expected.to_bits(), "tokens={tokens}");
        }
    }
}
