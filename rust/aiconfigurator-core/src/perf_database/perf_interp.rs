// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared resolver engine for perf-table interpolation (v2).
//!
//! Rust port of `src/aiconfigurator/sdk/perf_interp/engine.py` — the SAME
//! resolution chain, so the compiled engine-step backend and the Python SDK
//! answer queries identically:
//!
//! 1. exact hit             -> return the measured leaf verbatim
//! 2. resolve in the data   -> Grid: nested bracket+blend (per-axis transform;
//!                             a ragged branch is dropped, and a single
//!                             surviving branch is SOL-ratio-corrected along
//!                             the dropped axis). ScatteredSites: site curve
//!                             eval; unknown site -> nearest-site transfer in
//!                             util space (log2 IDW, curve-coverage filter,
//!                             distance gate).
//! 3. beyond the range      -> hold the boundary util (k_tail-median anchor),
//!                             latency = SOL(query) / util
//! 4. nothing to anchor on  -> Err (structured miss; never fabricate)
//!
//! Differences from the Python engine, all deliberate:
//! - Leaves are scalar latency (`f64`) — the Rust hot path carries no
//!   power/energy.
//! - Table keys are `u32` (matching every loader in this crate); query
//!   coordinates are `f64` so fractional queries interpolate.
//! - The in-slice UTIL transform is not ported (no op uses it; the Python
//!   config rejects it for Grid too).
//! - The GEMM site index is built once per table by the owner (tables are
//!   immutable after load) instead of Python's id-keyed LRU cache.

use std::collections::BTreeMap;

use crate::common::error::AicError;

/// Nested perf table: every level is a `u32`-keyed map, leaves are latency ms.
#[derive(Debug, Clone)]
pub enum Node {
    Branch(BTreeMap<u32, Node>),
    Leaf(f64),
}

impl Node {
    pub fn branch() -> Node {
        Node::Branch(BTreeMap::new())
    }

    /// Insert a leaf at the given path, creating intermediate branches.
    pub fn insert(&mut self, path: &[u32], value: f64) {
        match self {
            Node::Branch(map) => {
                if path.len() == 1 {
                    map.insert(path[0], Node::Leaf(value));
                } else {
                    map.entry(path[0])
                        .or_insert_with(Node::branch)
                        .insert(&path[1..], value);
                }
            }
            Node::Leaf(_) => panic!("insert into a leaf"),
        }
    }

    fn as_branch(&self) -> Option<&BTreeMap<u32, Node>> {
        match self {
            Node::Branch(map) => Some(map),
            Node::Leaf(_) => None,
        }
    }

    fn as_leaf(&self) -> Option<f64> {
        match self {
            Node::Leaf(v) => Some(*v),
            Node::Branch(_) => None,
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            Node::Branch(map) => map.is_empty(),
            Node::Leaf(_) => false,
        }
    }
}

/// In-slice interpolation space, applied per axis (see `transform_axis`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueTransform {
    Raw,
    /// Interpolate sqrt(latency): linearises ~seq^2 curvature (context ops).
    Sqrt,
}

fn to_space(vt: ValueTransform, lat: f64) -> f64 {
    match vt {
        ValueTransform::Raw => lat,
        ValueTransform::Sqrt => {
            if lat > 0.0 {
                lat.sqrt()
            } else {
                0.0
            }
        }
    }
}

fn from_space(vt: ValueTransform, v: f64) -> f64 {
    match vt {
        ValueTransform::Raw => v,
        ValueTransform::Sqrt => v * v,
    }
}

/// Which of the two table shapes an op is (see the Python config module).
pub enum Resolver {
    /// Grid-like, possibly corner-truncated tables (attention/MLA/DSA/...).
    Grid { k_tail: usize },
    /// Scattered-sites-plus-curve tables (GEMM): `site_axes` identify a
    /// collected shape, each owning a sweep along `curve_axis`.
    ScatteredSites {
        site_axes: Vec<usize>,
        curve_axis: usize,
        nn_sites: usize,
        max_site_distance: Option<f64>,
        require_curve_coverage: bool,
        k_tail: usize,
    },
}

/// Everything op-specific the shared engine needs. One record per query path.
pub struct OpInterpConfig<'a> {
    /// Axis names, outer -> inner (used in error messages only).
    pub axes: &'static [&'static str],
    pub resolver: Resolver,
    /// Analytic speed-of-light in axes order. Required: util-hold
    /// extrapolation and cross-site transfer are built on it.
    pub sol_fn: &'a dyn Fn(&[f64]) -> f64,
    pub value_transform: ValueTransform,
    /// Curvature is PER-AXIS: e.g. sqrt only when blending along seq.
    /// None = apply `value_transform` on every axis.
    pub transform_axis: Option<usize>,
}

impl<'a> OpInterpConfig<'a> {
    /// Grid config with RAW blending (generation-type ops).
    pub fn grid(axes: &'static [&'static str], sol_fn: &'a dyn Fn(&[f64]) -> f64) -> Self {
        OpInterpConfig {
            axes,
            resolver: Resolver::Grid { k_tail: 1 },
            sol_fn,
            value_transform: ValueTransform::Raw,
            transform_axis: None,
        }
    }

    /// Grid config with sqrt-on-one-axis blending (context-type ops, ~seq^2).
    pub fn grid_sqrt_axis(
        axes: &'static [&'static str],
        transform_axis: usize,
        sol_fn: &'a dyn Fn(&[f64]) -> f64,
    ) -> Self {
        assert!(transform_axis < axes.len());
        OpInterpConfig {
            axes,
            resolver: Resolver::Grid { k_tail: 1 },
            sol_fn,
            value_transform: ValueTransform::Sqrt,
            transform_axis: Some(transform_axis),
        }
    }
}

fn miss(cfg: &OpInterpConfig, coords: &[f64], reason: &str) -> AicError {
    let pairs: Vec<String> = cfg
        .axes
        .iter()
        .zip(coords)
        .map(|(a, c)| format!("{a}={c}"))
        .collect();
    AicError::PerfDatabase(format!(
        "perf_interp: no data to anchor query {{{}}} ({reason})",
        pairs.join(", ")
    ))
}

/// Internal signal: the query left the collected range at some level.
struct OutOfRange;

/// Resolve one query against a raw nested table.
pub fn query(cfg: &OpInterpConfig, data: &Node, coords: &[f64]) -> Result<f64, AicError> {
    assert_eq!(
        coords.len(),
        cfg.axes.len(),
        "query has {} coords; table axes are {:?}",
        coords.len(),
        cfg.axes
    );
    if data.is_empty() {
        return Err(miss(cfg, coords, "empty table"));
    }

    // Exact hit: walk the nesting verbatim.
    if let Some(v) = exact_hit(data, coords) {
        return Ok(v);
    }

    match &cfg.resolver {
        Resolver::ScatteredSites {
            site_axes,
            curve_axis,
            ..
        } => {
            // Convenience path (tests / cold callers): production owners
            // build the SiteIndex once at load and call `resolve` directly.
            let index = SiteIndex::build(site_axes, *curve_axis, data);
            index.resolve(cfg, coords)
        }
        Resolver::Grid { .. } => resolve_grid(cfg, data, coords),
    }
}

fn as_exact_key(c: f64) -> Option<u32> {
    if c >= 0.0 && c <= u32::MAX as f64 && c.fract() == 0.0 {
        Some(c as u32)
    } else {
        None
    }
}

fn exact_hit(data: &Node, coords: &[f64]) -> Option<f64> {
    let mut node = data;
    for &c in coords {
        let key = as_exact_key(c)?;
        node = node.as_branch()?.get(&key)?;
    }
    node.as_leaf()
}

// ---------------------------------------------------------------------------
// Grid: nested bracket+blend; out-of-range (incl. truncated corner) -> util-hold
// ---------------------------------------------------------------------------

fn resolve_grid(cfg: &OpInterpConfig, data: &Node, coords: &[f64]) -> Result<f64, AicError> {
    match grid_interior(cfg, data, coords, 0) {
        Ok(lat) => Ok(lat),
        Err(GridErr::Miss(e)) => Err(e),
        Err(GridErr::OutOfRange(_)) => grid_hold(cfg, data, coords),
    }
}

enum GridErr {
    OutOfRange(OutOfRange),
    Miss(AicError),
}

fn grid_interior(
    cfg: &OpInterpConfig,
    node: &Node,
    coords: &[f64],
    depth: usize,
) -> Result<f64, GridErr> {
    if depth == cfg.axes.len() {
        return node
            .as_leaf()
            .ok_or_else(|| GridErr::Miss(miss(cfg, coords, "malformed leaf")));
    }
    let map = node
        .as_branch()
        .ok_or_else(|| GridErr::Miss(miss(cfg, coords, "table shallower than axes")))?;
    if map.is_empty() {
        return Err(GridErr::Miss(miss(
            cfg,
            coords,
            &format!("empty branch at axis '{}'", cfg.axes[depth]),
        )));
    }

    let c = coords[depth];
    if let Some(key) = as_exact_key(c) {
        if let Some(child) = map.get(&key) {
            // exact key collapses this level
            return grid_interior(cfg, child, coords, depth + 1);
        }
    }

    let keys: Vec<u32> = map.keys().copied().collect();
    let (lo, hi) = (keys[0] as f64, keys[keys.len() - 1] as f64);
    if c < lo || c > hi {
        return Err(GridErr::OutOfRange(OutOfRange));
    }
    let idx = keys.partition_point(|&k| (k as f64) < c);
    let (k_lo, k_hi) = (keys[idx - 1], keys[idx]);

    let mut results: Vec<(u32, f64)> = Vec::with_capacity(2);
    let mut saw_out_of_range = false;
    for k in [k_lo, k_hi] {
        match grid_interior(cfg, &map[&k], coords, depth + 1) {
            Ok(lat) => results.push((k, lat)),
            Err(GridErr::OutOfRange(_)) => saw_out_of_range = true, // ragged branch: drop
            Err(GridErr::Miss(_)) => {} // ragged branch: drop
        }
    }
    match results.len() {
        0 => {
            // Both branches failed. Out-of-range anywhere below means the query
            // sits past the staircase frontier -> let util-hold anchor it.
            if saw_out_of_range {
                Err(GridErr::OutOfRange(OutOfRange))
            } else {
                Err(GridErr::Miss(miss(
                    cfg,
                    coords,
                    &format!("no usable branch at axis '{}'", cfg.axes[depth]),
                )))
            }
        }
        1 => {
            // One bracket branch dropped (ragged table). Returning the survivor
            // verbatim would CLAMP this axis with no correction (measured -41%
            // median on one-sided seq-row folds). Keep the survivor's resolved
            // value (it carries the measured inner-axis structure) and re-scale
            // along THIS axis by the SOL ratio, i.e. hold the survivor's util
            // across the dropped axis.
            let (k_surv, lat) = results[0];
            let mut snapped = coords.to_vec();
            snapped[depth] = k_surv as f64;
            let sol_q = (cfg.sol_fn)(coords);
            let sol_s = (cfg.sol_fn)(&snapped);
            if sol_q.is_finite() && sol_s.is_finite() && sol_q > 0.0 && sol_s > 0.0 {
                Ok(lat * (sol_q / sol_s))
            } else {
                Ok(lat)
            }
        }
        _ => {
            let (_, lat_lo) = results[0];
            let (_, lat_hi) = results[1];
            let w = (c - k_lo as f64) / (k_hi as f64 - k_lo as f64);
            // Curvature is per-axis: apply the transform only when blending
            // along the configured axis; other axes are ~linear -> raw.
            let vt = match cfg.transform_axis {
                Some(axis) if axis != depth => ValueTransform::Raw,
                _ => cfg.value_transform,
            };
            Ok(from_space(
                vt,
                to_space(vt, lat_lo) + (to_space(vt, lat_hi) - to_space(vt, lat_lo)) * w,
            ))
        }
    }
}

/// Anchor past-the-frontier queries: snap to the nearest collected path, hold
/// the boundary util (k_tail median along the innermost axis), and let
/// SOL(query) carry the growth.
fn grid_hold(cfg: &OpInterpConfig, data: &Node, coords: &[f64]) -> Result<f64, AicError> {
    let k_tail = match &cfg.resolver {
        Resolver::Grid { k_tail } => *k_tail,
        Resolver::ScatteredSites { .. } => unreachable!("grid_hold on scattered resolver"),
    };
    let n_axes = cfg.axes.len();
    let mut node = data;
    let mut snapped: Vec<f64> = Vec::with_capacity(n_axes - 1);
    for depth in 0..n_axes - 1 {
        let map = node
            .as_branch()
            .ok_or_else(|| miss(cfg, coords, "table shallower than axes"))?;
        if map.is_empty() {
            return Err(miss(
                cfg,
                coords,
                &format!("empty branch at axis '{}'", cfg.axes[depth]),
            ));
        }
        let c = coords[depth];
        let key = nearest_key(map, c);
        snapped.push(key as f64);
        node = &map[&key];
    }
    let map = node
        .as_branch()
        .ok_or_else(|| miss(cfg, coords, "table shallower than axes"))?;
    if map.is_empty() {
        return Err(miss(
            cfg,
            coords,
            &format!("empty branch at axis '{}'", cfg.axes[n_axes - 1]),
        ));
    }

    let keys: Vec<u32> = map.keys().copied().collect();
    let c = coords[n_axes - 1];
    let tail: Vec<u32> = if c > keys[keys.len() - 1] as f64 {
        keys[keys.len().saturating_sub(k_tail)..].to_vec()
    } else if c < keys[0] as f64 {
        keys[..k_tail.min(keys.len())].to_vec()
    } else {
        // innermost is in range; an OUTER axis was snapped
        vec![nearest_key(map, c)]
    };

    let mut utils: Vec<f64> = Vec::with_capacity(tail.len());
    for t in tail {
        let Some(lat) = map[&t].as_leaf() else {
            continue;
        };
        let mut anchor = snapped.clone();
        anchor.push(t as f64);
        let sol = (cfg.sol_fn)(&anchor);
        if lat > 0.0 && sol > 0.0 {
            utils.push(sol / lat);
        }
    }
    if utils.is_empty() {
        return Err(miss(cfg, coords, "no positive-util boundary anchor"));
    }
    let sol_q = (cfg.sol_fn)(coords);
    if !(sol_q > 0.0) {
        return Err(miss(cfg, coords, "non-positive SOL at query"));
    }
    Ok(sol_q / median(&mut utils))
}

fn nearest_key(map: &BTreeMap<u32, Node>, c: f64) -> u32 {
    *map.keys()
        .min_by(|a, b| {
            let da = (**a as f64 - c).abs();
            let db = (**b as f64 - c).abs();
            da.total_cmp(&db)
        })
        .expect("nearest_key on empty map")
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    }
}

// ---------------------------------------------------------------------------
// ScatteredSites: site curves + nearest-site util transfer (GEMM)
// ---------------------------------------------------------------------------

/// Site index for a scattered-sites table. Tables are immutable after load,
/// so owners build this once (e.g. in a `OnceLock`) and reuse it.
pub struct SiteIndex {
    /// site key -> sorted (curve coordinate, latency) sweep
    sites: BTreeMap<Vec<u32>, Vec<(u32, f64)>>,
    site_logs: Vec<(Vec<u32>, Vec<f64>)>,
    /// Per-site curve span `(first_curve_coord, last_curve_coord)`, aligned with
    /// `site_logs`. Cached so the curve-coverage filter is an array index rather
    /// than a per-query `sites` map lookup.
    curve_bounds: Vec<(u32, u32)>,
}

impl SiteIndex {
    pub fn build(site_axes: &[usize], curve_axis: usize, data: &Node) -> SiteIndex {
        let mut leaves: Vec<(Vec<u32>, f64)> = Vec::new();
        walk_leaves(data, &mut Vec::new(), &mut leaves);
        let mut sites: BTreeMap<Vec<u32>, Vec<(u32, f64)>> = BTreeMap::new();
        for (path, lat) in leaves {
            let key: Vec<u32> = site_axes.iter().map(|&p| path[p]).collect();
            sites.entry(key).or_default().push((path[curve_axis], lat));
        }
        for curve in sites.values_mut() {
            curve.sort_by_key(|&(c, _)| c);
        }
        let (site_logs, curve_bounds): (Vec<(Vec<u32>, Vec<f64>)>, Vec<(u32, u32)>) = sites
            .iter()
            .map(|(k, curve)| {
                let logs = k
                    .iter()
                    .map(|&v| (v.max(1) as f64).log2())
                    .collect::<Vec<f64>>();
                let bounds = (curve[0].0, curve[curve.len() - 1].0);
                ((k.clone(), logs), bounds)
            })
            .unzip();
        SiteIndex {
            sites,
            site_logs,
            curve_bounds,
        }
    }

    /// Resolve one query. Exact hits are answered by the site's own curve
    /// (exact curve point == the measured leaf), so owners with a prebuilt
    /// index call this directly instead of `query`.
    pub fn resolve(&self, cfg: &OpInterpConfig, coords: &[f64]) -> Result<f64, AicError> {
        let Resolver::ScatteredSites {
            site_axes,
            curve_axis,
            nn_sites,
            max_site_distance,
            require_curve_coverage,
            ..
        } = &cfg.resolver
        else {
            unreachable!()
        };

        let q = coords[*curve_axis];
        // Collected shape (integer site coords present in the index): its own
        // curve answers alone.
        let site_ints: Option<Vec<u32>> = site_axes
            .iter()
            .map(|&p| as_exact_key(coords[p]))
            .collect();
        if let Some(key) = &site_ints {
            if let Some(curve) = self.sites.get(key) {
                return self.eval_curve(cfg, curve, key, q, coords);
            }
        }

        // Unknown shape: transfer util from the nearest collected sites.
        if self.sites.is_empty() {
            return Err(miss(cfg, coords, "no sites collected"));
        }
        let q_log: Vec<f64> = site_axes
            .iter()
            .map(|&p| coords[p].max(1e-12).log2())
            .collect();
        let dist = |logs: &[f64]| -> f64 {
            logs.iter()
                .zip(&q_log)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt()
        };

        // Rank collected sites by distance to the query. Everything downstream
        // — coverage filter, distance gate, nearest-k selection, IDW weight —
        // reads off this ONE `(distance, site index)` buffer, so resolve does a
        // single allocation and computes each site's `dist` (a sqrt over the
        // site axes) exactly once. Recomputing `dist` inside the old sort
        // comparator, plus separate dists/candidates/covering vecs, made this
        // resolve the dominant engine-step cost for query-heavy models (e.g.
        // per-block puzzle nets whose GEMM shapes miss the collected sites).
        let mut ranked: Vec<(f64, usize)> = Vec::with_capacity(self.site_logs.len());
        if *require_curve_coverage {
            for (i, (_, logs)) in self.site_logs.iter().enumerate() {
                let (lo, hi) = self.curve_bounds[i];
                if (lo as f64) <= q && q <= (hi as f64) {
                    ranked.push((dist(logs), i));
                }
            }
        }
        // No coverage requirement, or nothing covered q -> fall back to all
        // sites (each later held at its own curve end).
        if ranked.is_empty() {
            for (i, (_, logs)) in self.site_logs.iter().enumerate() {
                ranked.push((dist(logs), i));
            }
        }

        // Gate first, then partial-select the nn_sites nearest — O(n) — instead
        // of a full O(n log n) sort we would immediately truncate to a handful.
        // (Equivalent to the former sort→gate→take: a full sort then gate then
        // take-k selects exactly the k nearest gated sites.) The `.then` index
        // tie-break reproduces the previous *stable* sort's order on equal
        // distances (sites are pushed in ascending-index order), so the selected
        // set and its ordering are identical.
        if let Some(gate) = max_site_distance {
            ranked.retain(|&(d, _)| d <= *gate);
            if ranked.is_empty() {
                return Err(miss(cfg, coords, "no site within max_site_distance"));
            }
        }
        let cmp =
            |a: &(f64, usize), b: &(f64, usize)| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1));
        let k = (*nn_sites).min(ranked.len());
        if k < ranked.len() {
            ranked.select_nth_unstable_by(k, cmp);
            ranked.truncate(k);
        }
        ranked.sort_by(cmp);

        let (mut wsum, mut u_acc) = (0.0_f64, 0.0_f64);
        for &(d, i) in ranked.iter().take(*nn_sites) {
            let neigh = &self.site_logs[i].0;
            let curve = &self.sites[neigh];
            // one bad neighbour must not poison the query
            let Ok(lat_i) = self.eval_curve(cfg, curve, neigh, q, coords) else {
                continue;
            };
            let mut n_coords = coords.to_vec();
            for (&p, &v) in site_axes.iter().zip(neigh) {
                n_coords[p] = v as f64;
            }
            n_coords[*curve_axis] = q;
            let sol_i = (cfg.sol_fn)(&n_coords);
            if !(lat_i.is_finite() && lat_i > 0.0 && sol_i.is_finite() && sol_i > 0.0) {
                continue;
            }
            let w = 1.0 / (d * d + 1e-12);
            u_acc += w * (sol_i / lat_i);
            wsum += w;
        }
        if wsum <= 0.0 {
            return Err(miss(cfg, coords, "no usable neighbour site"));
        }
        let sol_q = (cfg.sol_fn)(coords);
        if !(sol_q > 0.0) {
            return Err(miss(cfg, coords, "non-positive SOL at query"));
        }
        Ok(sol_q / (u_acc / wsum))
    }

    /// Evaluate one site's curve at coordinate `q`.
    fn eval_curve(
        &self,
        cfg: &OpInterpConfig,
        curve: &[(u32, f64)],
        site_vals: &[u32],
        q: f64,
        coords: &[f64],
    ) -> Result<f64, AicError> {
        let (curve_axis, site_axes, k_tail) = match &cfg.resolver {
            Resolver::ScatteredSites {
                curve_axis,
                site_axes,
                k_tail,
                ..
            } => (*curve_axis, site_axes, *k_tail),
            Resolver::Grid { .. } => unreachable!(),
        };
        let full_coords = |cv: f64| -> Vec<f64> {
            let mut out = coords.to_vec();
            for (&p, &v) in site_axes.iter().zip(site_vals) {
                out[p] = v as f64;
            }
            out[curve_axis] = cv;
            out
        };

        let idx = curve.partition_point(|&(c, _)| (c as f64) < q);
        if idx < curve.len() && (curve[idx].0 as f64) == q {
            return Ok(curve[idx].1); // exact point on the curve
        }

        if q < curve[0].0 as f64 || q > curve[curve.len() - 1].0 as f64 || curve.len() < 2 {
            // beyond the sweep -> util-hold on the k_tail boundary points
            let tail = if q < curve[0].0 as f64 {
                &curve[..k_tail.min(curve.len())]
            } else {
                &curve[curve.len().saturating_sub(k_tail)..]
            };
            let mut utils: Vec<f64> = Vec::with_capacity(tail.len());
            for &(cv, lat) in tail {
                let sol = (cfg.sol_fn)(&full_coords(cv as f64));
                if lat > 0.0 && sol > 0.0 {
                    utils.push(sol / lat);
                }
            }
            if utils.is_empty() {
                return Err(miss(cfg, coords, "no positive-util boundary anchor"));
            }
            let sol_q = (cfg.sol_fn)(&full_coords(q));
            if !(sol_q > 0.0) {
                return Err(miss(cfg, coords, "non-positive SOL at query"));
            }
            return Ok(sol_q / median(&mut utils));
        }

        let (c_lo, lat_lo) = curve[idx - 1];
        let (c_hi, lat_hi) = curve[idx];
        let w = (q - c_lo as f64) / (c_hi as f64 - c_lo as f64);
        let vt = match cfg.transform_axis {
            Some(axis) if axis != curve_axis => ValueTransform::Raw,
            _ => cfg.value_transform,
        };
        Ok(from_space(
            vt,
            to_space(vt, lat_lo) + (to_space(vt, lat_hi) - to_space(vt, lat_lo)) * w,
        ))
    }
}

/// Flatten a nested table into `(coords, latency_ms)` points with `f64`
/// coordinates — the input shape of the operator-layer util-calibration
/// grids (`operators::util_empirical::build_samples`, mirroring Python's
/// `iter_grid` over a nested dict slice).
pub(crate) fn node_points(node: &Node) -> Vec<(Vec<f64>, f64)> {
    let mut out = Vec::new();
    walk_leaves(node, &mut Vec::new(), &mut out);
    out.into_iter()
        .map(|(coords, latency)| (coords.into_iter().map(|c| c as f64).collect(), latency))
        .collect()
}

fn walk_leaves(node: &Node, prefix: &mut Vec<u32>, out: &mut Vec<(Vec<u32>, f64)>) {
    match node {
        Node::Leaf(v) => out.push((prefix.clone(), *v)),
        Node::Branch(map) => {
            for (&k, child) in map {
                prefix.push(k);
                walk_leaves(child, prefix, out);
                prefix.pop();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests: mirror tests/unit/sdk/database/test_perf_interp_engine.py
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) {
        assert!(
            (a - b).abs() <= 1e-9 * b.abs().max(1.0),
            "left: {a}, right: {b}"
        );
    }

    // Attention-like: (num_heads, seq, batch) grid, corner-truncated; lat ~ n*b*s^2
    fn attn_lat(c: &[f64]) -> f64 {
        1e-6 * c[0] * c[2] * c[1] * c[1]
    }

    fn attn_table() -> Node {
        // Staircase: the larger the seq, the fewer batches collected.
        let present: &[(u32, &[u32])] = &[
            (512, &[1, 2, 4, 8]),
            (1024, &[1, 2, 4, 8]),
            (2048, &[1, 2, 4]),
            (4096, &[1, 2]),
        ];
        let mut root = Node::branch();
        for n in [8u32, 16] {
            for &(s, bs) in present {
                for &b in bs {
                    root.insert(&[n, s, b], attn_lat(&[n as f64, s as f64, b as f64]));
                }
            }
        }
        root
    }

    fn attn_cfg(sol: &dyn Fn(&[f64]) -> f64) -> OpInterpConfig<'_> {
        // sqrt on the seq axis (index 1), like context_attention_config
        OpInterpConfig::grid_sqrt_axis(&["num_heads", "seq_len", "batch"], 1, sol)
    }

    #[test]
    fn exact_hit_returns_leaf_verbatim() {
        let t = attn_table();
        let cfg = attn_cfg(&attn_lat);
        let lat = query(&cfg, &t, &[8.0, 2048.0, 4.0]).unwrap();
        approx(lat, attn_lat(&[8.0, 2048.0, 4.0]));
    }

    #[test]
    fn grid_sqrt_blend_is_exact_for_quadratic_seq() {
        // seq=1536 between 1024 and 2048: sqrt(lat) is linear in s -> exact.
        let t = attn_table();
        let cfg = attn_cfg(&attn_lat);
        let lat = query(&cfg, &t, &[8.0, 1536.0, 2.0]).unwrap();
        approx(lat, attn_lat(&[8.0, 1536.0, 2.0]));
    }

    #[test]
    fn transform_applies_only_along_its_axis() {
        // batch=3 between 2 and 4 (seq exact): latency is LINEAR in batch, so
        // the blend must be raw-exact; sqrt must not distort the batch axis.
        let t = attn_table();
        let cfg = attn_cfg(&attn_lat);
        let lat = query(&cfg, &t, &[8.0, 1024.0, 3.0]).unwrap();
        approx(lat, attn_lat(&[8.0, 1024.0, 3.0]));
    }

    #[test]
    fn grid_hold_beyond_frontier_holds_util() {
        // seq=8192 beyond the sweep: SOL == latency (util 1) -> hold is exact.
        let t = attn_table();
        let cfg = attn_cfg(&attn_lat);
        let lat = query(&cfg, &t, &[8.0, 8192.0, 2.0]).unwrap();
        approx(lat, attn_lat(&[8.0, 8192.0, 2.0]));
    }

    #[test]
    fn grid_single_survivor_gets_sol_ratio_correction() {
        // seq=1536 at batch=8: 2048 branch lacks b=8 -> survivor is the 1024
        // branch (exact leaf), corrected by SOL(8,1536,8)/SOL(8,1024,8). The
        // fixture's SOL equals its latency (util == 1), so the correction
        // reproduces the formula exactly at the un-collected seq.
        let t = attn_table();
        let cfg = attn_cfg(&attn_lat);
        let lat = query(&cfg, &t, &[8.0, 1536.0, 8.0]).unwrap();
        approx(lat, attn_lat(&[8.0, 1536.0, 8.0]));
    }

    #[test]
    fn grid_empty_table_is_a_miss() {
        let t = Node::branch();
        let cfg = attn_cfg(&attn_lat);
        assert!(query(&cfg, &t, &[8.0, 512.0, 1.0]).is_err());
    }

    // 4-axis (DSA/CSA-like): [num_heads][prefix][seq][batch]
    fn dsa_lat(c: &[f64]) -> f64 {
        1e-6 * c[0] * c[3] * c[2] * (c[2] + c[1])
    }

    fn dsa_table() -> Node {
        let mut root = Node::branch();
        for n in [16u32, 32] {
            for p in [0u32, 4096, 8192] {
                for s in [1024u32, 2048, 4096] {
                    for b in [1u32, 2, 4] {
                        root.insert(
                            &[n, p, s, b],
                            dsa_lat(&[n as f64, p as f64, s as f64, b as f64]),
                        );
                    }
                }
            }
        }
        root
    }

    fn dsa_cfg(sol: &dyn Fn(&[f64]) -> f64) -> OpInterpConfig<'_> {
        OpInterpConfig::grid(&["num_heads", "prefix", "seq_len", "batch"], sol)
    }

    #[test]
    fn four_axis_exact_and_prefix_blend() {
        let t = dsa_table();
        let cfg = dsa_cfg(&dsa_lat);
        approx(
            query(&cfg, &t, &[16.0, 4096.0, 2048.0, 2.0]).unwrap(),
            dsa_lat(&[16.0, 4096.0, 2048.0, 2.0]),
        );
        // prefix=6144 between 4096 and 8192: lat is linear in p -> raw blend exact
        approx(
            query(&cfg, &t, &[16.0, 6144.0, 2048.0, 2.0]).unwrap(),
            dsa_lat(&[16.0, 6144.0, 2048.0, 2.0]),
        );
    }

    #[test]
    fn four_axis_prefix_hold() {
        // prefix=16384 beyond the collected range: util-hold (util==1 -> exact).
        let t = dsa_table();
        let cfg = dsa_cfg(&dsa_lat);
        approx(
            query(&cfg, &t, &[16.0, 16384.0, 2048.0, 2.0]).unwrap(),
            dsa_lat(&[16.0, 16384.0, 2048.0, 2.0]),
        );
    }

    // 1-axis (comm-size-like)
    #[test]
    fn one_axis_interp_and_hold() {
        let mut t = Node::branch();
        let lat1 = |c: &[f64]| 0.001 * c[0];
        for sz in [1024u32, 2048, 4096] {
            t.insert(&[sz], lat1(&[sz as f64]));
        }
        let sol: &dyn Fn(&[f64]) -> f64 = &lat1;
        let cfg = OpInterpConfig::grid(&["size"], sol);
        approx(query(&cfg, &t, &[3072.0]).unwrap(), lat1(&[3072.0]));
        approx(query(&cfg, &t, &[16384.0]).unwrap(), lat1(&[16384.0]));
        // below the smallest size: util-hold, never a negative linear trend
        assert!(query(&cfg, &t, &[128.0]).unwrap() > 0.0);
    }

    // GEMM-like scattered sites: data[m][n][k], sites (n,k), curve m
    fn gemm_lat(c: &[f64]) -> f64 {
        1e-9 * c[0] * c[1] * c[2]
    }

    fn gemm_cfg(sol: &dyn Fn(&[f64]) -> f64) -> OpInterpConfig<'_> {
        OpInterpConfig {
            axes: &["m", "n", "k"],
            resolver: Resolver::ScatteredSites {
                site_axes: vec![1, 2],
                curve_axis: 0,
                nn_sites: 4,
                max_site_distance: Some(2.0),
                require_curve_coverage: true,
                k_tail: 3,
            },
            sol_fn: sol,
            value_transform: ValueTransform::Raw,
            transform_axis: None,
        }
    }

    fn gemm_table() -> Node {
        // Two scattered sites with dense m sweeps; NO (k, n) mirror.
        let mut root = Node::branch();
        for &(n, k) in &[(4096u32, 1024u32), (5120, 2048)] {
            for m in [16u32, 32, 64, 128, 256, 512, 1024] {
                root.insert(&[m, n, k], gemm_lat(&[m as f64, n as f64, k as f64]));
            }
        }
        root
    }

    #[test]
    fn gemm_collected_site_answers_from_its_own_curve() {
        let t = gemm_table();
        let cfg = gemm_cfg(&gemm_lat);
        // m=48 between 32 and 64 at a collected (n,k): pure 1-D lerp on the
        // site's own curve (linear fixture -> exact).
        approx(
            query(&cfg, &t, &[48.0, 4096.0, 1024.0]).unwrap(),
            gemm_lat(&[48.0, 4096.0, 1024.0]),
        );
    }

    #[test]
    fn gemm_m_beyond_sweep_holds_util() {
        let t = gemm_table();
        let cfg = gemm_cfg(&gemm_lat);
        approx(
            query(&cfg, &t, &[8192.0, 4096.0, 1024.0]).unwrap(),
            gemm_lat(&[8192.0, 4096.0, 1024.0]),
        );
    }

    #[test]
    fn gemm_unknown_site_transfers_util_from_neighbours() {
        let t = gemm_table();
        let cfg = gemm_cfg(&gemm_lat);
        // (4608, 1536) sits between the two collected shapes (log-space) —
        // util transfer with util==1 fixture reproduces the formula.
        approx(
            query(&cfg, &t, &[64.0, 4608.0, 1536.0]).unwrap(),
            gemm_lat(&[64.0, 4608.0, 1536.0]),
        );
    }

    #[test]
    fn gemm_far_site_is_a_structured_miss() {
        let t = gemm_table();
        let cfg = gemm_cfg(&gemm_lat);
        // (64, 64): > 2 octaves from every collected site -> miss, not a guess.
        assert!(query(&cfg, &t, &[64.0, 64.0, 64.0]).is_err());
    }

    #[test]
    fn gemm_decode_only_site_does_not_answer_long_m() {
        // Site A covers m<=64 only; site B covers the full sweep. A query at
        // m=512 near site A must use B (coverage filter), not extrapolate A.
        let mut t = Node::branch();
        for m in [16u32, 32, 64] {
            t.insert(&[m, 4096, 1024], gemm_lat(&[m as f64, 4096.0, 1024.0]));
        }
        for m in [16u32, 32, 64, 128, 256, 512, 1024] {
            t.insert(&[m, 5120, 2048], gemm_lat(&[m as f64, 5120.0, 2048.0]));
        }
        let cfg = gemm_cfg(&gemm_lat);
        approx(
            query(&cfg, &t, &[512.0, 4608.0, 1536.0]).unwrap(),
            gemm_lat(&[512.0, 4608.0, 1536.0]),
        );
    }
}
