// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Latency-grid interpolation primitives.
//!
//! Mirrors the scalar paths of `src/aiconfigurator/sdk/interpolation.py`. The
//! Python module also carries dict-leaf (latency + power + energy) and
//! topk-piecewise variants; this Rust port covers the scalar latency cases
//! used by GEMM and the attention/MoE families. The topk-piecewise variants
//! land in C3c next to the DSA table loader.

use std::collections::BTreeMap;

use crate::common::error::AicError;

/// Bracketing neighbors of `query` in a sorted axis.
///
/// Mirrors `interpolation.nearest_1d_point_helper`. With
/// `inner_only = true`, returns an error if `query` is outside the
/// `[min, max]` range; with `inner_only = false`, returns the two values
/// nearest the appropriate end (used for extrapolation).
pub fn nearest_neighbors(
    query: u32,
    sorted_values: &[u32],
    inner_only: bool,
) -> Result<(u32, u32), AicError> {
    let n = sorted_values.len();
    if n == 0 {
        return Err(AicError::PerfDatabase(
            "interpolation axis is empty".to_string(),
        ));
    }
    if n == 1 {
        if inner_only && query != sorted_values[0] {
            return Err(AicError::PerfDatabase(format!(
                "interpolation query {query} does not match the single-point axis {:?}",
                sorted_values
            )));
        }
        return Ok((sorted_values[0], sorted_values[0]));
    }

    let lo = sorted_values[0];
    let hi = sorted_values[n - 1];
    if query < lo {
        if inner_only {
            return Err(AicError::PerfDatabase(format!(
                "interpolation query {query} is below the axis minimum {lo}"
            )));
        }
        return Ok((sorted_values[0], sorted_values[1]));
    }
    if query > hi {
        if inner_only {
            return Err(AicError::PerfDatabase(format!(
                "interpolation query {query} is above the axis maximum {hi}"
            )));
        }
        return Ok((sorted_values[n - 2], sorted_values[n - 1]));
    }

    // Linear scan matching Python — axes are small (typically <100 entries).
    let mut start = sorted_values[0];
    for (i, &value) in sorted_values.iter().enumerate() {
        if query >= value && i + 1 < n {
            start = value;
            continue;
        }
        return Ok((start, value));
    }
    // Should be unreachable given the bounds checks above.
    Ok((sorted_values[n - 2], sorted_values[n - 1]))
}

/// Linear interpolation between `(x0, y0)` and `(x1, y1)` evaluated at `x`.
///
/// Includes Python's anti-overshoot clamp: when the slope direction would
/// push `y` outside `[y0, y1]` during extrapolation, the result is clamped
/// to the nearer endpoint. This matches `interpolation.interp_1d` for
/// scalar leaves.
pub fn interp_1d(x0: f64, x1: f64, y0: f64, y1: f64, x: f64) -> f64 {
    // Python's overshoot guard:
    // if (x0 - x1) * (y0 - y1) < 0 and (value - x0) * (value - x1) > 0:
    //     y1 = y0
    let mut y1 = y1;
    if (x0 - x1) * (y0 - y1) < 0.0 && (x - x0) * (x - x1) > 0.0 {
        y1 = y0;
    }
    if y0 == y1 || x0 == x1 {
        return y0;
    }
    y0 + (y1 - y0) / (x1 - x0) * (x - x0)
}

/// Bilinear interpolation on a 2-D rectangular cell.
///
/// `q00`/`q01`/`q10`/`q11` are the corner values at
/// `(x0, y0)` / `(x0, y1)` / `(x1, y0)` / `(x1, y1)`. Mirrors
/// `interpolation.bilinear_interpolation`.
pub fn bilinear(
    x0: f64,
    x1: f64,
    y0: f64,
    y1: f64,
    q00: f64,
    q01: f64,
    q10: f64,
    q11: f64,
    x: f64,
    y: f64,
) -> f64 {
    let f00 = q00 * (x1 - x) * (y1 - y);
    let f01 = q01 * (x1 - x) * (y - y0);
    let f10 = q10 * (x - x0) * (y1 - y);
    let f11 = q11 * (x - x0) * (y - y0);
    let denom = (x1 - x0) * (y1 - y0);
    (f00 + f01 + f10 + f11) / denom
}

/// Type alias for a sparse 3-D scalar grid keyed by `(outer, middle, inner)`.
pub type Grid3<T> = BTreeMap<u32, BTreeMap<u32, BTreeMap<u32, T>>>;

/// 3-D interpolation done as bilinear over (`y`, `z`) followed by linear
/// over `x`. Mirrors `interpolation.interp_2d_1d` with `method="bilinear"`,
/// which is what the GEMM/attention/MLA/DSv4 queries use as their fallback
/// path. Now a thin wrapper over [`interp_2d_1d_grid_extrapolate_inner`]
/// because the two functions have converged: both extrapolate on all axes
/// (`inner_only=false`), and the buggy version of this function that took
/// z-axis bracket keys only from `left_row` raised
/// `"missing point at (x,y,z)"` whenever the perf grid was sparse across
/// y-rows (e.g. the attention/GEMM parquet shipped by b200_sxm/vllm has
/// `batch=1024` recorded at `seq=256` but not at `seq=512`, so a query at
/// `seq` between 256 and 512 with `batch=1024` looked the corner up in the
/// shorter `seq=512` row and crashed). The intersect-then-bracket pattern
/// in [`interp_2d_1d_grid_extrapolate_inner`] mirrors Python's load-time
/// `extrapolate_data_grid` densification, so call sites get the same
/// result Python would produce after pre-fill.
pub fn interp_2d_1d_grid(grid: &Grid3<f64>, x: u32, y: u32, z: u32) -> Result<f64, AicError> {
    interp_2d_1d_grid_extrapolate_inner(grid, x, y, z)
}

/// Same as [`interp_2d_1d_grid`] but allows linear extrapolation on the
/// `y` (middle) and `z` (inner) axes when the query lands outside the
/// data envelope. The `x` (outer) axis still requires the query to be
/// within range.
///
/// Mirrors Python's `extrapolate_data_grid` + `interp_2d_linear` pair
/// used by the DSA context/generation tables. Python pre-populates a
/// fixed target axis list by calling `interp_1d` on the nearest
/// neighbours of the existing axis (which allows extrapolation past the
/// boundary), then queries that pre-populated grid via strict
/// interpolation. For linear data, that pre-populate-then-interpolate
/// sequence is algebraically equivalent to a single direct linear
/// extrapolation from the original boundary pair (the same `interp_1d`
/// with its overshoot guard). We do the direct extrapolation here so
/// the table loader doesn't have to materialise the full extended grid
/// — the math the operator layer sees is the same.
pub fn interp_2d_1d_grid_extrapolate_inner(
    grid: &Grid3<f64>,
    x: u32,
    y: u32,
    z: u32,
) -> Result<f64, AicError> {
    let x_keys: Vec<u32> = grid.keys().copied().collect();
    // Allow x-axis extrapolation too — see `interp_2d_1d_grid` for the
    // rationale. The original name `extrapolate_inner` referred to y/z only;
    // both functions now extrapolate on all three axes.
    let (x_left, x_right) = nearest_neighbors(x, &x_keys, false)?;

    let interpolate_yz_at = |slice_key: u32| -> Result<f64, AicError> {
        let slice = grid
            .get(&slice_key)
            .ok_or_else(|| AicError::PerfDatabase(format!("missing slice at x={slice_key}")))?;
        let y_keys: Vec<u32> = slice.keys().copied().collect();
        // `inner_only=false` -> when `y` is out of range we get the boundary
        // pair, and `interp_1d` below performs the extrapolation.
        let (y_left, y_right) = nearest_neighbors(y, &y_keys, false)?;

        let left_row = slice.get(&y_left).ok_or_else(|| {
            AicError::PerfDatabase(format!("missing row at x={slice_key},y={y_left}"))
        })?;
        let right_row = slice.get(&y_right).ok_or_else(|| {
            AicError::PerfDatabase(format!("missing row at x={slice_key},y={y_right}"))
        })?;

        // Intersect z keys between left and right rows so the bilinear
        // cell is well-formed for either lattice. Python's
        // `extrapolate_data_grid` fills both rows before interpolation;
        // intersection here gets us to the same sub-grid the query would
        // see post-fill.
        let z_keys_left: std::collections::BTreeSet<u32> = left_row.keys().copied().collect();
        let z_keys: Vec<u32> = right_row
            .keys()
            .copied()
            .filter(|k| z_keys_left.contains(k))
            .collect();
        if z_keys.is_empty() {
            return Err(AicError::PerfDatabase(format!(
                "no shared z keys between y_left={y_left} and y_right={y_right} at x={slice_key}"
            )));
        }
        let (z_left, z_right) = nearest_neighbors(z, &z_keys, false)?;

        let q00 = *left_row.get(&z_left).ok_or_else(|| missing(slice_key, y_left, z_left))?;
        let q01 = *left_row.get(&z_right).ok_or_else(|| missing(slice_key, y_left, z_right))?;
        let q10 = *right_row.get(&z_left).ok_or_else(|| missing(slice_key, y_right, z_left))?;
        let q11 = *right_row.get(&z_right).ok_or_else(|| missing(slice_key, y_right, z_right))?;

        if y_left == y_right && z_left == z_right {
            return Ok(q00);
        }
        if y_left == y_right {
            return Ok(interp_1d(z_left as f64, z_right as f64, q00, q01, z as f64));
        }
        if z_left == z_right {
            return Ok(interp_1d(y_left as f64, y_right as f64, q00, q10, y as f64));
        }
        Ok(bilinear(
            y_left as f64,
            y_right as f64,
            z_left as f64,
            z_right as f64,
            q00,
            q01,
            q10,
            q11,
            y as f64,
            z as f64,
        ))
    };

    let v_left = interpolate_yz_at(x_left)?;
    let v_right = interpolate_yz_at(x_right)?;

    if x_left == x_right {
        return Ok(v_left);
    }
    Ok(interp_1d(
        x_left as f64,
        x_right as f64,
        v_left,
        v_right,
        x as f64,
    ))
}

fn missing(x: u32, y: u32, z: u32) -> AicError {
    AicError::PerfDatabase(format!("missing point at ({x},{y},{z})"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn nearest_neighbors_exact_hit_returns_value_and_next() {
        // Matches Python's loop: when query is exactly at axis[i] and not the
        // last value, returns (axis[i], axis[i+1]). At the last value, returns
        // (axis[-2], axis[-1]).
        let axis = [1, 2, 4, 8, 16];
        assert_eq!(nearest_neighbors(4, &axis, true).unwrap(), (4, 8));
        assert_eq!(nearest_neighbors(2, &axis, true).unwrap(), (2, 4));
        assert_eq!(nearest_neighbors(16, &axis, true).unwrap(), (8, 16));
    }

    #[test]
    fn nearest_neighbors_between() {
        let axis = [1, 2, 4, 8, 16];
        assert_eq!(nearest_neighbors(3, &axis, true).unwrap(), (2, 4));
        assert_eq!(nearest_neighbors(10, &axis, true).unwrap(), (8, 16));
    }

    #[test]
    fn nearest_neighbors_out_of_range_inner_only_errors() {
        let axis = [2, 4, 8];
        assert!(nearest_neighbors(1, &axis, true).is_err());
        assert!(nearest_neighbors(9, &axis, true).is_err());
    }

    #[test]
    fn nearest_neighbors_out_of_range_extrapolation() {
        let axis = [2, 4, 8];
        assert_eq!(nearest_neighbors(1, &axis, false).unwrap(), (2, 4));
        assert_eq!(nearest_neighbors(9, &axis, false).unwrap(), (4, 8));
    }

    #[test]
    fn nearest_neighbors_single_value() {
        let axis = [5];
        assert_eq!(nearest_neighbors(5, &axis, true).unwrap(), (5, 5));
        assert!(nearest_neighbors(6, &axis, true).is_err());
        assert_eq!(nearest_neighbors(6, &axis, false).unwrap(), (5, 5));
    }

    #[test]
    fn interp_1d_midpoint() {
        // Linear segment from (0, 10) to (10, 20): midpoint at 5 -> 15.
        assert!(approx(interp_1d(0.0, 10.0, 10.0, 20.0, 5.0), 15.0, 1e-9));
    }

    #[test]
    fn interp_1d_endpoint_equal_returns_y0() {
        assert!(approx(interp_1d(0.0, 10.0, 7.0, 7.0, 4.0), 7.0, 1e-9));
    }

    #[test]
    fn interp_1d_extrapolation_positive_slope() {
        // (0,10) -> (10,20); extrapolate to x=20 -> 30. The anti-overshoot
        // guard fires only when slope direction would over/undershoot the
        // bracket — not the case for clean monotone segments.
        assert!(approx(interp_1d(0.0, 10.0, 10.0, 20.0, 20.0), 30.0, 1e-9));
    }

    #[test]
    fn interp_1d_anti_monotone_extrapolation_is_clamped() {
        // (0,20) -> (10,10) is anti-monotone: as x increases, y decreases.
        // Python's guard fires for this case during extrapolation, clamping
        // y1 to y0 (= 20). Result is y0 because the clamp removes the slope.
        assert!(approx(interp_1d(0.0, 10.0, 20.0, 10.0, 20.0), 20.0, 1e-9));
        assert!(approx(interp_1d(0.0, 10.0, 20.0, 10.0, -5.0), 20.0, 1e-9));
    }

    #[test]
    fn bilinear_corners_are_recovered() {
        // Unit cell with corners 1, 2, 3, 4.
        assert!(approx(bilinear(0.0, 1.0, 0.0, 1.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0), 1.0, 1e-9));
        assert!(approx(bilinear(0.0, 1.0, 0.0, 1.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0), 2.0, 1e-9));
        assert!(approx(bilinear(0.0, 1.0, 0.0, 1.0, 1.0, 2.0, 3.0, 4.0, 1.0, 0.0), 3.0, 1e-9));
        assert!(approx(bilinear(0.0, 1.0, 0.0, 1.0, 1.0, 2.0, 3.0, 4.0, 1.0, 1.0), 4.0, 1e-9));
        // Center: 2.5.
        assert!(approx(bilinear(0.0, 1.0, 0.0, 1.0, 1.0, 2.0, 3.0, 4.0, 0.5, 0.5), 2.5, 1e-9));
    }

    fn make_grid(values: &[(u32, u32, u32, f64)]) -> Grid3<f64> {
        let mut grid: Grid3<f64> = BTreeMap::new();
        for &(x, y, z, v) in values {
            grid.entry(x)
                .or_default()
                .entry(y)
                .or_default()
                .insert(z, v);
        }
        grid
    }

    #[test]
    fn interp_2d_1d_grid_exact_corner() {
        let grid = make_grid(&[
            (1, 1, 1, 10.0),
            (1, 1, 2, 20.0),
            (1, 2, 1, 30.0),
            (1, 2, 2, 40.0),
            (2, 1, 1, 100.0),
            (2, 1, 2, 200.0),
            (2, 2, 1, 300.0),
            (2, 2, 2, 400.0),
        ]);
        // Exact corner (1,1,1) -> 10.
        assert!(approx(interp_2d_1d_grid(&grid, 1, 1, 1).unwrap(), 10.0, 1e-9));
        // Exact opposite corner (2,2,2) -> 400.
        assert!(approx(interp_2d_1d_grid(&grid, 2, 2, 2).unwrap(), 400.0, 1e-9));
    }

    #[test]
    fn interp_2d_1d_grid_midpoint() {
        let grid = make_grid(&[
            (1, 1, 1, 0.0),
            (1, 1, 3, 0.0),
            (1, 3, 1, 0.0),
            (1, 3, 3, 0.0),
            (3, 1, 1, 12.0),
            (3, 1, 3, 12.0),
            (3, 3, 1, 12.0),
            (3, 3, 3, 12.0),
        ]);
        // y/z constant along each x-slice, so result depends only on x.
        // At x=2 (midpoint of 1..3), result should be 6.0 by linear interp.
        assert!(approx(interp_2d_1d_grid(&grid, 2, 2, 2).unwrap(), 6.0, 1e-9));
    }

    #[test]
    fn interp_2d_1d_grid_out_of_range_extrapolates() {
        // `interp_2d_1d_grid` now mirrors Python's `extrapolate_data_grid`
        // semantics — every axis allows extrapolation when the query falls
        // outside the loaded grid. A query at x=3 (above the max of 2)
        // takes the boundary pair (1, 2) and linearly extrapolates the
        // 2-D slice latencies.
        let grid = make_grid(&[
            (1, 1, 1, 10.0),
            (1, 1, 2, 11.0),
            (1, 2, 1, 12.0),
            (1, 2, 2, 13.0),
            (2, 1, 1, 20.0),
            (2, 1, 2, 21.0),
            (2, 2, 1, 22.0),
            (2, 2, 2, 23.0),
        ]);
        // The extrapolated value at x=3,y=1,z=1 should be ~30 (linear
        // continuation of 10, 20).
        let v = interp_2d_1d_grid(&grid, 3, 1, 1).expect("extrapolation must not error");
        assert!((v - 30.0).abs() < 1e-9, "expected ~30.0, got {v}");
    }
}
