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
//!                             distance gate; the gate is waived for a site
//!                             beyond the scale-up frontier).
//! 3. beyond the range      -> hold the boundary util (k_tail-median anchor),
//!                             latency = SOL(query) / util
//! 4. nothing to anchor on  -> Err (structured miss; never fabricate)
//!
//! Differences from the Python engine, all deliberate:
//! - Table keys are `u32` (matching every loader in this crate); query
//!   coordinates are `f64` so fractional queries interpolate.
//! - The in-slice UTIL transform is not ported (no op uses it; the Python
//!   config rejects it for Grid too).
//! - The GEMM site index is built once per table by the owner (tables are
//!   immutable after load) instead of Python's id-keyed LRU cache.

use std::collections::BTreeMap;

use crate::common::error::AicError;

/// One measured table leaf — mirrors the Python loader dict
/// `{"latency", "power", "energy"}` (`power` straight from the parquet
/// column, `energy = power * latency` in W·ms, both 0.0 when the table
/// carries no power data).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct LeafValue {
    pub latency: f64,
    pub power: f64,
    pub energy: f64,
}

impl LeafValue {
    pub fn latency_only(latency: f64) -> LeafValue {
        LeafValue {
            latency,
            power: 0.0,
            energy: 0.0,
        }
    }

    /// Loader-side constructor: `energy = power * latency`, the exact
    /// expression every Python loader uses (W·ms).
    pub fn with_power(latency: f64, power: f64) -> LeafValue {
        LeafValue {
            latency,
            power,
            energy: power * latency,
        }
    }

    /// Average power used by every blend path — mirrors Python
    /// `_leaf_power`: energy/latency when both are positive, else the
    /// explicit power field (power-only rows), else 0.0.
    pub fn blend_power(self) -> f64 {
        if self.energy > 0.0 && self.latency > 0.0 {
            self.energy / self.latency
        } else {
            self.power
        }
    }
}

/// Nested perf table: every level is a `u32`-keyed map, leaves are the
/// measured `LeafValue` (latency ms + optional power/energy).
#[derive(Debug, Clone)]
pub enum Node {
    Branch(BTreeMap<u32, Node>),
    Leaf(LeafValue),
}

impl Node {
    pub fn branch() -> Node {
        Node::Branch(BTreeMap::new())
    }

    /// Insert a latency-only leaf at the given path, creating intermediate
    /// branches (tables without power columns).
    pub fn insert(&mut self, path: &[u32], value: f64) {
        self.insert_value(path, LeafValue::latency_only(value));
    }

    /// Insert a full measured leaf at the given path.
    pub fn insert_value(&mut self, path: &[u32], value: LeafValue) {
        match self {
            Node::Branch(map) => {
                if path.len() == 1 {
                    map.insert(path[0], Node::Leaf(value));
                } else {
                    map.entry(path[0])
                        .or_insert_with(Node::branch)
                        .insert_value(&path[1..], value);
                }
            }
            Node::Leaf(_) => panic!("insert into a leaf"),
        }
    }

    /// First-wins latency-only leaf insert (see `insert_value_first_wins`).
    pub fn insert_first_wins(&mut self, path: &[u32], value: f64) {
        self.insert_value_first_wins(path, LeafValue::latency_only(value));
    }

    /// First-wins leaf insert: keeps an existing leaf untouched. Loaders use
    /// this when merging priority-ordered shared-layer sources (the first
    /// source that has a coordinate wins; `insert` would let later, lower-
    /// priority sources overwrite it).
    pub fn insert_value_first_wins(&mut self, path: &[u32], value: LeafValue) {
        match self {
            Node::Branch(map) => {
                if path.len() == 1 {
                    map.entry(path[0]).or_insert(Node::Leaf(value));
                } else {
                    map.entry(path[0])
                        .or_insert_with(Node::branch)
                        .insert_value_first_wins(&path[1..], value);
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

    fn as_leaf(&self) -> Option<LeafValue> {
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

/// Resolve one query against a raw nested table (latency only).
pub fn query(cfg: &OpInterpConfig, data: &Node, coords: &[f64]) -> Result<f64, AicError> {
    query_value(cfg, data, coords).map(|v| v.latency)
}

/// Resolve one query against a raw nested table. Returns the measured leaf
/// verbatim on an exact hit, else `{latency, power, energy = power * latency}`
/// — the same contract as the Python engine's `query`.
pub fn query_value(
    cfg: &OpInterpConfig,
    data: &Node,
    coords: &[f64],
) -> Result<LeafValue, AicError> {
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

    let (latency, power) = match &cfg.resolver {
        Resolver::ScatteredSites {
            site_axes,
            curve_axis,
            ..
        } => {
            // Convenience path (tests / cold callers): production owners
            // build the SiteIndex once at load and call `resolve` directly.
            let index = SiteIndex::build(site_axes, *curve_axis, data);
            index.resolve_pair(cfg, coords)?
        }
        Resolver::Grid { .. } => resolve_grid(cfg, data, coords)?,
    };
    Ok(LeafValue {
        latency,
        power,
        energy: power * latency,
    })
}

fn as_exact_key(c: f64) -> Option<u32> {
    if c >= 0.0 && c <= u32::MAX as f64 && c.fract() == 0.0 {
        Some(c as u32)
    } else {
        None
    }
}

fn exact_hit(data: &Node, coords: &[f64]) -> Option<LeafValue> {
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

fn resolve_grid(cfg: &OpInterpConfig, data: &Node, coords: &[f64]) -> Result<(f64, f64), AicError> {
    match grid_interior(cfg, data, coords, 0) {
        Ok(pair) => Ok(pair),
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
) -> Result<(f64, f64), GridErr> {
    if depth == cfg.axes.len() {
        return node
            .as_leaf()
            .map(|leaf| (leaf.latency, leaf.blend_power()))
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

    let mut results: Vec<(u32, (f64, f64))> = Vec::with_capacity(2);
    let mut saw_out_of_range = false;
    for k in [k_lo, k_hi] {
        match grid_interior(cfg, &map[&k], coords, depth + 1) {
            Ok(pair) => results.push((k, pair)),
            Err(GridErr::OutOfRange(_)) => saw_out_of_range = true, // ragged branch: drop
            Err(GridErr::Miss(_)) => {}                             // ragged branch: drop
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
            // across the dropped axis. Power passes through unscaled (it is a
            // bounded intensive quantity; the Python engine does the same).
            let (k_surv, (lat, p)) = results[0];
            let mut snapped = coords.to_vec();
            snapped[depth] = k_surv as f64;
            let sol_q = (cfg.sol_fn)(coords);
            let sol_s = (cfg.sol_fn)(&snapped);
            if sol_q.is_finite() && sol_s.is_finite() && sol_q > 0.0 && sol_s > 0.0 {
                Ok((lat * (sol_q / sol_s), p))
            } else {
                Ok((lat, p))
            }
        }
        _ => {
            let (_, (lat_lo, p_lo)) = results[0];
            let (_, (lat_hi, p_hi)) = results[1];
            let w = (c - k_lo as f64) / (k_hi as f64 - k_lo as f64);
            // Curvature is per-axis: apply the transform only when blending
            // along the configured axis; other axes are ~linear -> raw.
            // Power is always blended linearly with the same weight.
            let vt = match cfg.transform_axis {
                Some(axis) if axis != depth => ValueTransform::Raw,
                _ => cfg.value_transform,
            };
            Ok((
                from_space(
                    vt,
                    to_space(vt, lat_lo) + (to_space(vt, lat_hi) - to_space(vt, lat_lo)) * w,
                ),
                p_lo + (p_hi - p_lo) * w,
            ))
        }
    }
}

/// Anchor past-the-frontier queries: snap to the nearest collected path, hold
/// the boundary util (k_tail median along the innermost axis), and let
/// SOL(query) carry the growth.
fn grid_hold(cfg: &OpInterpConfig, data: &Node, coords: &[f64]) -> Result<(f64, f64), AicError> {
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
    let mut powers: Vec<f64> = Vec::with_capacity(tail.len());
    for t in tail {
        let Some(leaf) = map[&t].as_leaf() else {
            continue;
        };
        let lat = leaf.latency;
        let mut anchor = snapped.clone();
        anchor.push(t as f64);
        let sol = (cfg.sol_fn)(&anchor);
        if lat > 0.0 && sol > 0.0 {
            utils.push(sol / lat);
            powers.push(leaf.blend_power());
        }
    }
    if utils.is_empty() {
        return Err(miss(cfg, coords, "no positive-util boundary anchor"));
    }
    let sol_q = (cfg.sol_fn)(coords);
    if !(sol_q > 0.0) {
        return Err(miss(cfg, coords, "non-positive SOL at query"));
    }
    Ok((sol_q / median(&mut utils), median(&mut powers)))
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
    /// site key -> sorted (curve coordinate, measured leaf) sweep
    sites: BTreeMap<Vec<u32>, Vec<(u32, LeafValue)>>,
    site_logs: Vec<(Vec<u32>, Vec<f64>)>,
    /// Per-site curve span `(first_curve_coord, last_curve_coord)`, aligned with
    /// `site_logs`. Cached so the curve-coverage filter is an array index rather
    /// than a per-query `sites` map lookup.
    curve_bounds: Vec<(u32, u32)>,
}

impl SiteIndex {
    pub fn build(site_axes: &[usize], curve_axis: usize, data: &Node) -> SiteIndex {
        let mut leaves: Vec<(Vec<u32>, LeafValue)> = Vec::new();
        walk_leaves(data, &mut Vec::new(), &mut leaves);
        let mut sites: BTreeMap<Vec<u32>, Vec<(u32, LeafValue)>> = BTreeMap::new();
        for (path, leaf) in leaves {
            let key: Vec<u32> = site_axes.iter().map(|&p| path[p]).collect();
            sites.entry(key).or_default().push((path[curve_axis], leaf));
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

    /// [`SiteIndex::build`] with an explicit site enumeration order.
    ///
    /// Selection rules are parity surface (issue #1456): Python enumerates
    /// sites in its nested dict's insertion order and breaks EXACT distance
    /// ties at the nearest-k cutoff by that order (stable sort). Ties are
    /// structural on the collected lattices (mirrored ratio pairs make the
    /// per-axis log2 deltas identical bit patterns swapped), so the
    /// enumeration order is load-bearing. `site_order` is the loader's
    /// replay of the Python dict walk (e.g. GEMM's first-encounter `(n, k)`
    /// order); sites not listed keep their sorted order after the listed
    /// ones. An empty order leaves the sorted enumeration (synthetic/test
    /// callers).
    pub fn build_with_site_order(
        site_axes: &[usize],
        curve_axis: usize,
        data: &Node,
        site_order: &[Vec<u32>],
    ) -> SiteIndex {
        let mut index = SiteIndex::build(site_axes, curve_axis, data);
        if site_order.is_empty() {
            return index;
        }
        let rank: std::collections::HashMap<&[u32], usize> = site_order
            .iter()
            .enumerate()
            .map(|(i, key)| (key.as_slice(), i))
            .collect();
        let mut zipped: Vec<((Vec<u32>, Vec<f64>), (u32, u32))> = index
            .site_logs
            .drain(..)
            .zip(index.curve_bounds.drain(..))
            .collect();
        // Stable sort: listed sites take their Python rank; unlisted sites
        // (none for a faithful loader replay) keep sorted order at the tail.
        zipped.sort_by_key(|((key, _), _)| rank.get(key.as_slice()).copied().unwrap_or(usize::MAX));
        let (site_logs, curve_bounds) = zipped.into_iter().unzip();
        index.site_logs = site_logs;
        index.curve_bounds = curve_bounds;
        index
    }

    /// Resolve one query to the full measured value — the `SiteIndex`
    /// counterpart of `query_value` for owners with a prebuilt index. Exact
    /// hits are answered by the site's own curve (exact curve point == the
    /// measured leaf).
    pub fn resolve_value(
        &self,
        cfg: &OpInterpConfig,
        coords: &[f64],
    ) -> Result<LeafValue, AicError> {
        self.resolve_pair(cfg, coords)
            .map(|(latency, power)| LeafValue {
                latency,
                power,
                energy: power * latency,
            })
    }

    fn resolve_pair(&self, cfg: &OpInterpConfig, coords: &[f64]) -> Result<(f64, f64), AicError> {
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
        let site_ints: Option<Vec<u32>> =
            site_axes.iter().map(|&p| as_exact_key(coords[p])).collect();
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
        // tie-break reproduces a *stable* sort over the index's enumeration
        // order (sites are pushed in ascending-index order) — which, for an
        // index built with `build_with_site_order`, is Python's dict-walk
        // order, so EXACT distance ties at the cutoff break identically to
        // the Python engine (issue #1456).
        let cmp =
            |a: &(f64, usize), b: &(f64, usize)| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1));
        if let Some(gate) = max_site_distance {
            if ranked.iter().any(|&(d, _)| d <= *gate) {
                ranked.retain(|&(d, _)| d <= *gate);
            } else {
                // The gate is waived for a query site beyond the collected
                // frontier in the scale-up direction (big-vocab LM heads at
                // low tp): anchors are the coverage-eligible sites that still
                // pass the gate on the non-overflow axes, and SOL(query)
                // carries the growth. Interior holes, scale-down / mixed
                // queries, and sparse-stub multi-axis overflow keep the miss.
                let q_site: Vec<f64> = site_axes.iter().map(|&p| coords[p]).collect();
                match self.frontier_waiver_anchors(&ranked, &q_site, &q_log, *gate) {
                    Some(admissible) => ranked = admissible,
                    None => return Err(miss(cfg, coords, "no site within max_site_distance")),
                }
            }
        }
        let k = (*nn_sites).min(ranked.len());
        if k < ranked.len() {
            ranked.select_nth_unstable_by(k, cmp);
            ranked.truncate(k);
        }
        ranked.sort_by(cmp);

        let (mut wsum, mut u_acc, mut p_acc) = (0.0_f64, 0.0_f64, 0.0_f64);
        for &(d, i) in ranked.iter().take(*nn_sites) {
            let neigh = &self.site_logs[i].0;
            let curve = &self.sites[neigh];
            // one bad neighbour must not poison the query
            let Ok((lat_i, p_i)) = self.eval_curve(cfg, curve, neigh, q, coords) else {
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
            p_acc += w * p_i;
            wsum += w;
        }
        if wsum <= 0.0 {
            return Err(miss(cfg, coords, "no usable neighbour site"));
        }
        let sol_q = (cfg.sol_fn)(coords);
        if !(sol_q > 0.0) {
            return Err(miss(cfg, coords, "non-positive SOL at query"));
        }
        // Latency transfers util (SOL-rescaled); power is the plain IDW mean
        // of the neighbours' measured power — no site transfer, no SOL
        // re-scaling (mirrors the Python engine).
        Ok((sol_q / (u_acc / wsum), p_acc / wsum))
    }

    /// Anchor set for a scale-up frontier query, or None (gate miss stands).
    ///
    /// The distance gate is waived ONLY for the intended frontier overflow,
    /// and everything is computed over `candidates` (the coverage-eligible
    /// `(full distance, site index)` buffer from `resolve`) so admissibility
    /// and the anchors actually used cannot diverge:
    ///
    /// - exactly ONE site axis overflows (strictly above the candidates'
    ///   maximum). Simultaneous multi-axis overflow means the table is a
    ///   sparse stub for this shape (e.g. an fp8_block table collected at
    ///   n,k<=128 queried with a real logits GEMM) — holding a launch-bound
    ///   stub's util across many octaves fabricates arbitrarily wrong
    ///   latencies, so it stays a miss;
    /// - no axis sits below the candidates' minimum (scale-down never
    ///   transfers);
    /// - the NON-overflow axes still pass the gate (residual log2 distance),
    ///   so anchors are genuine frontier neighbours of the query and the
    ///   transfer is scale-up along the overflow axis only.
    ///
    /// On the overflow axis every anchor is below the query by construction;
    /// the caller's inverse-distance weighting then favours the largest
    /// collected sites, and SOL(query) carries the growth (the same trust as
    /// the m curve axis and the grid out-of-range hold).
    fn frontier_waiver_anchors(
        &self,
        candidates: &[(f64, usize)],
        q_site: &[f64],
        q_log: &[f64],
        gate: f64,
    ) -> Option<Vec<(f64, usize)>> {
        let axes = q_site.len();
        let mut overflow: Vec<usize> = Vec::new();
        for a in 0..axes {
            let (mut lo, mut hi) = (u32::MAX, 0u32);
            for &(_, i) in candidates {
                let key = &self.site_logs[i].0;
                lo = lo.min(key[a]);
                hi = hi.max(key[a]);
            }
            if q_site[a] < lo as f64 {
                return None;
            }
            if q_site[a] > hi as f64 {
                overflow.push(a);
            }
        }
        if overflow.len() != 1 {
            return None;
        }
        let resid_dist = |i: usize| -> f64 {
            (0..axes)
                .filter(|a| !overflow.contains(a))
                .map(|a| {
                    let d = self.site_logs[i].1[a] - q_log[a];
                    d * d
                })
                .sum::<f64>()
                .sqrt()
        };
        let admissible: Vec<(f64, usize)> = candidates
            .iter()
            .copied()
            .filter(|&(_, i)| resid_dist(i) <= gate)
            .collect();
        if admissible.is_empty() {
            None
        } else {
            Some(admissible)
        }
    }

    /// Evaluate one site's curve at coordinate `q` -> `(latency, power)`.
    fn eval_curve(
        &self,
        cfg: &OpInterpConfig,
        curve: &[(u32, LeafValue)],
        site_vals: &[u32],
        q: f64,
        coords: &[f64],
    ) -> Result<(f64, f64), AicError> {
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
            let leaf = curve[idx].1; // exact point on the curve
            return Ok((leaf.latency, leaf.blend_power()));
        }

        if q < curve[0].0 as f64 || q > curve[curve.len() - 1].0 as f64 || curve.len() < 2 {
            // beyond the sweep -> util-hold on the k_tail boundary points
            let tail = if q < curve[0].0 as f64 {
                &curve[..k_tail.min(curve.len())]
            } else {
                &curve[curve.len().saturating_sub(k_tail)..]
            };
            let mut utils: Vec<f64> = Vec::with_capacity(tail.len());
            let mut powers: Vec<f64> = Vec::with_capacity(tail.len());
            for &(cv, leaf) in tail {
                let lat = leaf.latency;
                let sol = (cfg.sol_fn)(&full_coords(cv as f64));
                if lat > 0.0 && sol > 0.0 {
                    utils.push(sol / lat);
                    powers.push(leaf.blend_power());
                }
            }
            if utils.is_empty() {
                return Err(miss(cfg, coords, "no positive-util boundary anchor"));
            }
            let sol_q = (cfg.sol_fn)(&full_coords(q));
            if !(sol_q > 0.0) {
                return Err(miss(cfg, coords, "non-positive SOL at query"));
            }
            return Ok((sol_q / median(&mut utils), median(&mut powers)));
        }

        let (c_lo, leaf_lo) = curve[idx - 1];
        let (c_hi, leaf_hi) = curve[idx];
        let (lat_lo, lat_hi) = (leaf_lo.latency, leaf_hi.latency);
        let w = (q - c_lo as f64) / (c_hi as f64 - c_lo as f64);
        let vt = match cfg.transform_axis {
            Some(axis) if axis != curve_axis => ValueTransform::Raw,
            _ => cfg.value_transform,
        };
        let (p_lo, p_hi) = (leaf_lo.blend_power(), leaf_hi.blend_power());
        Ok((
            from_space(
                vt,
                to_space(vt, lat_lo) + (to_space(vt, lat_hi) - to_space(vt, lat_lo)) * w,
            ),
            p_lo + (p_hi - p_lo) * w,
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
        .map(|(coords, leaf)| (coords.into_iter().map(|c| c as f64).collect(), leaf.latency))
        .collect()
}

fn walk_leaves(node: &Node, prefix: &mut Vec<u32>, out: &mut Vec<(Vec<u32>, LeafValue)>) {
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
    fn gemm_scale_up_beyond_frontier_holds_util() {
        // Gemma-4-26B-A4B tp1 LM head (issue #1415): (n=262144, k=2816) is
        // ~2.005 octaves from the nearest collected site — past the distance
        // gate — but beyond the frontier in the scale-up direction, so the
        // engine holds the frontier util instead of missing. util==1 fixture
        // -> the hold is exact.
        let mut t = Node::branch();
        for m in [16u32, 32, 64] {
            for &(n, k) in &[(65536u32, 2560u32), (65536, 3072), (4096, 2816)] {
                t.insert(&[m, n, k], gemm_lat(&[m as f64, n as f64, k as f64]));
            }
        }
        let cfg = gemm_cfg(&gemm_lat);
        approx(
            query(&cfg, &t, &[32.0, 262144.0, 2816.0]).unwrap(),
            gemm_lat(&[32.0, 262144.0, 2816.0]),
        );
    }

    #[test]
    fn gemm_interior_hole_beyond_gate_still_misses() {
        // (8192, 8192) is dominated by the collected (65536, 65536) corner but
        // >2 octaves from every site — a sparse interior hole, not a frontier
        // query. The gate must still refuse to guess.
        let mut t = Node::branch();
        for m in [16u32, 32, 64] {
            for &(n, k) in &[(256u32, 256u32), (65536, 65536)] {
                t.insert(&[m, n, k], gemm_lat(&[m as f64, n as f64, k as f64]));
            }
        }
        let cfg = gemm_cfg(&gemm_lat);
        assert!(query(&cfg, &t, &[32.0, 8192.0, 8192.0]).is_err());
    }

    #[test]
    fn gemm_incomparable_notch_inside_bounding_box_still_misses() {
        // (4096, 4096) sits in the notch between (256, 65536) and (65536,
        // 256): no site dominates it, yet it exceeds the collected maximum on
        // NO axis — an interior hole in the Pareto staircase, not a scale-up
        // frontier query. The waiver must not fire; the gate stands.
        let mut t = Node::branch();
        for m in [16u32, 32, 64] {
            for &(n, k) in &[(256u32, 65536u32), (65536, 256)] {
                t.insert(&[m, n, k], gemm_lat(&[m as f64, n as f64, k as f64]));
            }
        }
        let cfg = gemm_cfg(&gemm_lat);
        assert!(query(&cfg, &t, &[32.0, 4096.0, 4096.0]).is_err());
    }

    #[test]
    fn gemm_sparse_stub_multi_axis_overflow_still_misses() {
        // Real-table regression (PR #1419 review): an fp8_block-like stub
        // collected only at n,k in 32..128 queried with a real logits GEMM
        // (m=1, n=151936, k=5120) — BOTH site axes past the collected
        // maximum. Simultaneous multi-axis overflow must stay a miss.
        let mut t = Node::branch();
        for m in [1u32, 2, 4] {
            for n in [32u32, 64, 128] {
                for k in [32u32, 64, 128] {
                    t.insert(&[m, n, k], gemm_lat(&[m as f64, n as f64, k as f64]));
                }
            }
        }
        let cfg = gemm_cfg(&gemm_lat);
        assert!(query(&cfg, &t, &[1.0, 151936.0, 5120.0]).is_err());
    }

    #[test]
    fn gemm_waiver_admissibility_computed_over_coverage_candidates() {
        // PR #1419 review: site (1, 1) covers only m<=2, site (64, 64) covers
        // m=16..64. Query (m=32, n=1024, k=1) used to pass a GLOBAL frontier
        // check while the only coverage-eligible anchor (64, 64) required a
        // k=64 -> k=1 scale-DOWN transfer. Admissibility over the coverage
        // candidates puts k=1 below the minimum -> miss.
        let mut t = Node::branch();
        for m in [1u32, 2] {
            t.insert(&[m, 1, 1], gemm_lat(&[m as f64, 1.0, 1.0]));
        }
        for m in [16u32, 32, 64] {
            t.insert(&[m, 64, 64], gemm_lat(&[m as f64, 64.0, 64.0]));
        }
        let cfg = gemm_cfg(&gemm_lat);
        assert!(query(&cfg, &t, &[32.0, 1024.0, 1.0]).is_err());
    }

    // -----------------------------------------------------------------------
    // Energy propagation: mirror tests/unit/sdk/database/test_data_loaders.py
    // ::test_query_dsv4_megamoe_module_interpolates_energy_from_rows and the
    // engine's _leaf_power blend semantics.
    // -----------------------------------------------------------------------

    #[test]
    fn exact_hit_returns_stored_energy_verbatim() {
        let mut t = Node::branch();
        t.insert_value(&[1024], LeafValue::with_power(1.0, 100.0));
        t.insert_value(&[2048], LeafValue::with_power(3.0, 200.0));
        let sol: &dyn Fn(&[f64]) -> f64 = &|c: &[f64]| c[0];
        let cfg = OpInterpConfig::grid(&["num_tokens"], sol);
        let v = query_value(&cfg, &t, &[1024.0]).unwrap();
        assert_eq!(
            v,
            LeafValue {
                latency: 1.0,
                power: 100.0,
                energy: 100.0
            }
        );
    }

    #[test]
    fn grid_blend_lerps_power_and_rederives_energy() {
        // The canonical distinguishing fixture (Python
        // test_query_dsv4_megamoe_module_interpolates_energy_from_rows):
        // leaves (latency 1.0, power 100) @1024 and (latency 3.0, power 200)
        // @2048, queried at 1536 (equal weight). POWER lerps to 150 and
        // energy re-derives as 150 * 2.0 = 300. A naive energy-lerp would
        // give (100*1 + 200*3)/2 = 350 (power 175) — conflating latency
        // growth into the blend.
        let mut t = Node::branch();
        t.insert_value(&[1024], LeafValue::with_power(1.0, 100.0));
        t.insert_value(&[2048], LeafValue::with_power(3.0, 200.0));
        let sol: &dyn Fn(&[f64]) -> f64 = &|c: &[f64]| c[0];
        let cfg = OpInterpConfig::grid(&["num_tokens"], sol);
        let v = query_value(&cfg, &t, &[1536.0]).unwrap();
        approx(v.latency, 2.0);
        approx(v.power, 150.0);
        approx(v.energy, 300.0);
    }

    #[test]
    fn grid_hold_uses_median_power_of_tail_anchors() {
        // Beyond the collected range: latency = SOL(q)/median(util), power =
        // median(anchor powers) — NOT SOL-rescaled (power is intensive).
        // k_tail = 3 over the last three anchors: powers [100, 250, 400],
        // median 250; energy re-derives from the held latency.
        let mut t = Node::branch();
        let lat1 = |c: &[f64]| 0.001 * c[0];
        for (sz, p) in [(1024u32, 50.0), (2048, 100.0), (4096, 250.0), (8192, 400.0)] {
            t.insert_value(&[sz], LeafValue::with_power(lat1(&[sz as f64]), p));
        }
        let sol: &dyn Fn(&[f64]) -> f64 = &lat1;
        let cfg = OpInterpConfig {
            axes: &["size"],
            resolver: Resolver::Grid { k_tail: 3 },
            sol_fn: sol,
            value_transform: ValueTransform::Raw,
            transform_axis: None,
        };
        let v = query_value(&cfg, &t, &[65536.0]).unwrap();
        approx(v.latency, lat1(&[65536.0])); // util == 1 fixture -> exact
        approx(v.power, 250.0);
        approx(v.energy, 250.0 * v.latency);
    }

    #[test]
    fn grid_single_survivor_passes_power_through_unscaled() {
        // seq=1536 at batch=8: the 2048 branch lacks b=8, so the 1024 branch
        // survives alone and its LATENCY is SOL-ratio-corrected along seq —
        // but its power passes through unscaled (bounded intensive quantity).
        let present: &[(u32, &[u32])] = &[(1024, &[1, 2, 4, 8]), (2048, &[1, 2, 4])];
        let mut t = Node::branch();
        for n in [8u32] {
            for &(s, bs) in present {
                for &b in bs {
                    t.insert_value(
                        &[n, s, b],
                        LeafValue::with_power(attn_lat(&[n as f64, s as f64, b as f64]), 123.0),
                    );
                }
            }
        }
        let cfg = attn_cfg(&attn_lat);
        let v = query_value(&cfg, &t, &[8.0, 1536.0, 8.0]).unwrap();
        approx(v.power, 123.0);
        approx(v.energy, 123.0 * v.latency);
    }

    #[test]
    fn scattered_site_transfer_takes_idw_mean_of_raw_power() {
        // Unknown (n, k) site between two collected shapes: LATENCY transfers
        // util (SOL-rescaled), POWER is the plain IDW mean of the neighbour
        // powers — no site transfer, no SOL re-scaling. Equidistant sites in
        // log2 space -> plain average of 100 and 300.
        let mut t = Node::branch();
        for m in [16u32, 32, 64, 128] {
            t.insert_value(
                &[m, 4096, 1024],
                LeafValue::with_power(gemm_lat(&[m as f64, 4096.0, 1024.0]), 100.0),
            );
            t.insert_value(
                &[m, 8192, 2048],
                LeafValue::with_power(gemm_lat(&[m as f64, 8192.0, 2048.0]), 300.0),
            );
        }
        let cfg = gemm_cfg(&gemm_lat);
        // (n, k) = (sqrt(4096*8192), sqrt(1024*2048)) is the exact log-space
        // midpoint of the two sites.
        let n = (4096.0_f64 * 8192.0).sqrt();
        let k = (1024.0_f64 * 2048.0).sqrt();
        let v = query_value(&cfg, &t, &[64.0, n, k]).unwrap();
        approx(v.power, 200.0);
        approx(v.energy, 200.0 * v.latency);
        // util == 1 fixture -> the transferred latency is the formula itself.
        approx(v.latency, gemm_lat(&[64.0, n, k]));
    }

    #[test]
    fn site_curve_hold_uses_median_power_and_exact_point_returns_leaf() {
        // m beyond the sweep on a collected site: k_tail=3 median power holds;
        // an exact curve point returns the measured leaf verbatim.
        let mut t = Node::branch();
        for (m, p) in [(16u32, 10.0), (32, 20.0), (64, 30.0), (128, 40.0)] {
            t.insert_value(
                &[m, 4096, 1024],
                LeafValue::with_power(gemm_lat(&[m as f64, 4096.0, 1024.0]), p),
            );
        }
        let cfg = gemm_cfg(&gemm_lat);
        let exact = query_value(&cfg, &t, &[32.0, 4096.0, 1024.0]).unwrap();
        assert_eq!(exact.power, 20.0);
        assert_eq!(exact.energy, 20.0 * gemm_lat(&[32.0, 4096.0, 1024.0]));
        let held = query_value(&cfg, &t, &[4096.0, 4096.0, 1024.0]).unwrap();
        approx(held.power, 30.0); // median of [20, 30, 40]
        approx(held.energy, 30.0 * held.latency);
    }

    #[test]
    fn latency_only_tables_resolve_energy_zero() {
        // Tables without a power column keep power = energy = 0.0 through
        // every resolution path (blend, hold).
        let t = attn_table();
        let cfg = attn_cfg(&attn_lat);
        for coords in [[8.0, 2048.0, 4.0], [8.0, 1536.0, 2.0], [8.0, 8192.0, 2.0]] {
            let v = query_value(&cfg, &t, &coords).unwrap();
            assert_eq!(v.power, 0.0);
            assert_eq!(v.energy, 0.0);
        }
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

    #[test]
    fn site_order_breaks_exact_distance_ties_like_python() {
        // Issue #1456's structural tie: (1536, 4096) and (1024, 6144) are
        // BITWISE-equidistant from query site (1280, 5120) in log2 space —
        // 1536/1280 = 6144/5120 = 1.2 and 1280/1024 = 5120/4096 = 1.25, so
        // the per-axis deltas are the same bit patterns swapped and the
        // summed square distance is identical. The winner at the nearest-k
        // cutoff is decided purely by enumeration order: Python uses dict
        // insertion order; the sorted default here would pick the other
        // site. `build_with_site_order` must reproduce the Python pick.
        let sol = |c: &[f64]| 1e-9 * c[0] * c[1] * c[2];
        let mut t = Node::branch();
        for m in [16u32, 32, 64] {
            // Site A (1536, 4096) runs at util 0.5, site B (1024, 6144) at
            // util 0.25 — the transferred answer reveals which site won.
            t.insert(&[m, 1536, 4096], sol(&[m as f64, 1536.0, 4096.0]) / 0.5);
            t.insert(&[m, 1024, 6144], sol(&[m as f64, 1024.0, 6144.0]) / 0.25);
        }
        let cfg = OpInterpConfig {
            axes: &["m", "n", "k"],
            resolver: Resolver::ScatteredSites {
                site_axes: vec![1, 2],
                curve_axis: 0,
                nn_sites: 1,
                max_site_distance: Some(2.0),
                require_curve_coverage: true,
                k_tail: 3,
            },
            sol_fn: &sol,
            value_transform: ValueTransform::Raw,
            transform_axis: None,
        };
        let coords = [32.0, 1280.0, 5120.0];
        let sol_q = sol(&coords);

        // Sorted enumeration ranks (1024, 6144) first -> util 0.25.
        let sorted_index = SiteIndex::build(&[1, 2], 0, &t);
        approx(
            sorted_index.resolve_value(&cfg, &coords).unwrap().latency,
            sol_q / 0.25,
        );

        // The Python dict walk saw (1536, 4096) first -> util 0.5.
        let walk_order = vec![vec![1536u32, 4096], vec![1024, 6144]];
        let ordered_index = SiteIndex::build_with_site_order(&[1, 2], 0, &t, &walk_order);
        approx(
            ordered_index.resolve_value(&cfg, &coords).unwrap().latency,
            sol_q / 0.5,
        );
    }
}
