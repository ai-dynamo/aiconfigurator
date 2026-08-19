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
//! 3. beyond the range      -> hold a boundary util, latency = SOL(query)/util.
//!                             Multi-axis grids transfer the util from the
//!                             `nn_leaves` nearest collected points in joint
//!                             log2 space with tapered (modified-Shepard)
//!                             weights — continuous across rank swaps, no
//!                             nearest-path snap; 1-D curves anchor on the
//!                             k_tail-median boundary points.
//! 4. nothing to anchor on  -> Err (structured miss; never fabricate)
//!
//! Differences from the Python engine, all deliberate:
//! - Table keys are `u32` (matching every loader in this crate); query
//!   coordinates are `f64` so fractional queries interpolate.
//! - The in-slice UTIL transform is not ported (no op uses it; the Python
//!   config rejects it for Grid too).
//! - The GEMM site index is built once per table by the owner (tables are
//!   immutable after load) instead of Python's id-keyed LRU cache.
//!
//! One-axis token and communication tables use the immutable `AxisCurve`
//! fast path instead of constructing a `Node` per query. Its query contract
//! must remain bit-identical to a RAW `Grid { k_tail: 1 }`; the differential
//! tests in `axis_curve.rs` enforce that relationship.

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
    /// `k_tail` anchors 1-D boundary holds; `nn_leaves` is the multi-axis
    /// hold's tapered joint-log blend width (4 won the frontier-holdout LOO).
    Grid { k_tail: usize, nn_leaves: usize },
    /// Scattered-sites-plus-curve tables (GEMM): `site_axes` identify a
    /// collected shape, each owning a sweep along `curve_axis`.
    ScatteredSites {
        site_axes: Vec<usize>,
        curve_axis: usize,
        nn_sites: usize,
        max_site_distance: Option<f64>,
        require_curve_coverage: bool,
        k_tail: usize,
        /// A collected site whose own curve does NOT cover the query defers to
        /// neighbour-site transfer with the own site excluded, instead of
        /// util-holding its own tail (degenerate sites must not anchor far
        /// extrapolation). Mirrors the Python config flag of the same name;
        /// `false` preserves the historical behaviour (GEMM).
        own_curve_coverage_fallback: bool,
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
            resolver: Resolver::Grid {
                k_tail: 1,
                nn_leaves: 4,
            },
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
            resolver: Resolver::Grid {
                k_tail: 1,
                nn_leaves: 4,
            },
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

/// Anchor past-the-frontier queries: transfer util from the `nn_leaves`
/// nearest collected points in joint log2 space, blended with tapered
/// modified-Shepard weights; latency = SOL(query)/util, power blended with
/// the same weights (intensive, not SOL-rescaled).
///
/// Weights are `w = ((R - d) / (R * d))^2` with the support radius R at the
/// (nn_leaves+1)-th valid leaf's distance (R = inf degrades smoothly to plain
/// 1/d^2). A neighbour enters/leaves the selection AT ZERO WEIGHT as the
/// query moves, so the estimate is continuous across rank swaps, and distance
/// ties need no ordering rule (weights are pure functions of distance —
/// independent of axis order and table insertion order).
///
/// This replaces the earlier nearest-path snap, which was discontinuous at
/// outer-axis midpoints (a +36.9% cliff between batch 192 and 193 on the B200
/// generation-attention staircase) and could anchor on a frontier point in a
/// different efficiency regime. Mirrors the Python engine's `_grid_hold`
/// exactly. Single-axis tables keep the k_tail-median boundary hold.
fn grid_hold(cfg: &OpInterpConfig, data: &Node, coords: &[f64]) -> Result<(f64, f64), AicError> {
    let (k_tail, nn_leaves) = match &cfg.resolver {
        Resolver::Grid { k_tail, nn_leaves } => (*k_tail, *nn_leaves),
        Resolver::ScatteredSites { .. } => unreachable!("grid_hold on scattered resolver"),
    };
    if cfg.axes.len() == 1 {
        return grid_hold_1d(cfg, data, coords, k_tail);
    }

    let anchors = hold_anchor_weights(cfg, data, coords, nn_leaves)?;
    let wsum: f64 = anchors.iter().map(|a| a.weight).sum();
    if wsum <= 0.0 {
        return Err(miss(cfg, coords, "no positive-util boundary anchor"));
    }
    let u_acc: f64 = anchors.iter().map(|a| a.weight * (a.sol / a.latency)).sum();
    let p_acc: f64 = anchors.iter().map(|a| a.weight * a.power).sum();
    let sol_q = (cfg.sol_fn)(coords);
    if !(sol_q.is_finite() && sol_q > 0.0) {
        return Err(miss(cfg, coords, "non-positive SOL at query"));
    }
    Ok((sol_q / (u_acc / wsum), p_acc / wsum))
}

/// 1-D curve past the sweep end: hold the k_tail-median boundary util; power
/// is the median of the tail anchors' powers (intensive, not SOL-rescaled).
fn grid_hold_1d(
    cfg: &OpInterpConfig,
    data: &Node,
    coords: &[f64],
    k_tail: usize,
) -> Result<(f64, f64), AicError> {
    let map = data
        .as_branch()
        .ok_or_else(|| miss(cfg, coords, "table shallower than axes"))?;
    if map.is_empty() {
        return Err(miss(
            cfg,
            coords,
            &format!("empty branch at axis '{}'", cfg.axes[0]),
        ));
    }
    let keys: Vec<u32> = map.keys().copied().collect();
    let c = coords[0];
    let tail: Vec<u32> = if c > keys[keys.len() - 1] as f64 {
        keys[keys.len().saturating_sub(k_tail)..].to_vec()
    } else {
        keys[..k_tail.min(keys.len())].to_vec()
    };

    let mut utils: Vec<f64> = Vec::with_capacity(tail.len());
    let mut powers: Vec<f64> = Vec::with_capacity(tail.len());
    for t in tail {
        let Some(leaf) = map[&t].as_leaf() else {
            continue;
        };
        let sol = (cfg.sol_fn)(&[t as f64]);
        if leaf.latency > 0.0 && sol > 0.0 {
            utils.push(sol / leaf.latency);
            powers.push(leaf.blend_power());
        }
    }
    if utils.is_empty() {
        return Err(miss(cfg, coords, "no positive-util boundary anchor"));
    }
    let sol_q = (cfg.sol_fn)(coords);
    if !(sol_q.is_finite() && sol_q > 0.0) {
        return Err(miss(cfg, coords, "non-positive SOL at query"));
    }
    Ok((sol_q / median(&mut utils), median(&mut powers)))
}

/// One selected hold anchor: a measured leaf with its blend weight.
pub(crate) struct HoldAnchor {
    pub(crate) coords: Vec<u32>,
    pub(crate) latency: f64,
    pub(crate) power: f64,
    pub(crate) sol: f64,
    pub(crate) weight: f64,
}

/// The multi-axis hold's anchor selection: the `nn_leaves` nearest valid
/// leaves in joint log2 space, with tapered modified-Shepard weights (support
/// radius R = the next valid leaf's distance; leaves at d == R weigh zero, so
/// no tie-ordering rule is needed).
/// `pub(crate)` so owners that must split a summed leaf into components
/// (wideep dispatch) reuse the engine's exact selection and weights.
///
/// Selection is a single pass over the tree with a small best-M buffer —
/// no full leaf vector, no full sort, and coordinate paths are cloned only
/// when a leaf enters the buffer. The buffer carries `SLACK` extra candidates
/// so validity filtering (lat/sol > 0) almost never needs the full-collect
/// fallback below.
pub(crate) fn hold_anchor_weights(
    cfg: &OpInterpConfig,
    data: &Node,
    coords: &[f64],
    nn_leaves: usize,
) -> Result<Vec<HoldAnchor>, AicError> {
    const SLACK: usize = 8;
    let m = nn_leaves + SLACK;
    let q_log: Vec<f64> = coords.iter().map(|&v| v.max(1e-12).log2()).collect();

    // (distance, coords, leaf), ascending by distance, at most m entries.
    let mut best: Vec<(f64, Vec<u32>, LeafValue)> = Vec::with_capacity(m + 1);
    let mut n_leaves = 0usize;
    visit_leaves(
        data,
        &mut Vec::new(),
        &mut |path: &[u32], leaf: LeafValue| {
            n_leaves += 1;
            let mut dd = 0.0;
            for (i, &v) in path.iter().enumerate() {
                let delta = ((v as f64).max(1e-12)).log2() - q_log[i];
                dd += delta * delta;
            }
            let d = dd.sqrt();
            if best.len() == m {
                if d >= best[m - 1].0 {
                    return;
                }
                best.pop();
            }
            let pos = best.partition_point(|e| e.0 <= d);
            best.insert(pos, (d, path.to_vec(), leaf));
        },
    );
    if n_leaves == 0 {
        return Err(miss(
            cfg,
            coords,
            &format!("empty branch at axis '{}'", cfg.axes[0]),
        ));
    }

    // Validity-check in distance order: the first nn_leaves valid candidates
    // are the anchors; the NEXT valid distance is the support radius R.
    let mut picked: Vec<HoldAnchor> = Vec::with_capacity(nn_leaves);
    let mut support_r = f64::INFINITY;
    let mut support_found = false;
    for (d, c, leaf) in &best {
        let anchor: Vec<f64> = c.iter().map(|&v| v as f64).collect();
        let sol = (cfg.sol_fn)(&anchor);
        if !(leaf.latency.is_finite() && leaf.latency > 0.0 && sol.is_finite() && sol > 0.0) {
            continue;
        }
        if picked.len() < nn_leaves {
            picked.push(HoldAnchor {
                coords: c.clone(),
                latency: leaf.latency,
                power: leaf.blend_power(),
                sol,
                weight: *d, // distance for now; weights assigned below
            });
        } else {
            support_r = *d;
            support_found = true;
            break;
        }
    }
    // The buffer proved too small to certify the selection (pathological
    // invalid density): fall back to the exhaustive path for correctness.
    if n_leaves > m && !support_found && picked.len() <= nn_leaves {
        return hold_anchor_weights_exhaustive(cfg, data, coords, nn_leaves, &q_log);
    }
    finish_hold_anchors(cfg, coords, picked, support_r)
}

/// Exhaustive fallback: collect and sort every leaf. Only reached when more
/// than `SLACK` of the nearest candidates were invalid (lat/sol <= 0).
fn hold_anchor_weights_exhaustive(
    cfg: &OpInterpConfig,
    data: &Node,
    coords: &[f64],
    nn_leaves: usize,
    q_log: &[f64],
) -> Result<Vec<HoldAnchor>, AicError> {
    let mut leaves: Vec<(Vec<u32>, LeafValue)> = Vec::new();
    walk_leaves(data, &mut Vec::new(), &mut leaves);
    let mut ranked: Vec<(f64, usize)> = leaves
        .iter()
        .enumerate()
        .map(|(i, (c, _))| {
            let dd: f64 = c
                .iter()
                .zip(q_log)
                .map(|(&v, ql)| {
                    let delta = ((v as f64).max(1e-12)).log2() - ql;
                    delta * delta
                })
                .sum();
            (dd.sqrt(), i)
        })
        .collect();
    ranked.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut picked: Vec<HoldAnchor> = Vec::with_capacity(nn_leaves);
    let mut support_r = f64::INFINITY;
    for &(d, i) in &ranked {
        let (c, leaf) = &leaves[i];
        let anchor: Vec<f64> = c.iter().map(|&v| v as f64).collect();
        let sol = (cfg.sol_fn)(&anchor);
        if !(leaf.latency.is_finite() && leaf.latency > 0.0 && sol.is_finite() && sol > 0.0) {
            continue;
        }
        if picked.len() < nn_leaves {
            picked.push(HoldAnchor {
                coords: c.clone(),
                latency: leaf.latency,
                power: leaf.blend_power(),
                sol,
                weight: d,
            });
        } else {
            support_r = d;
            break;
        }
    }
    finish_hold_anchors(cfg, coords, picked, support_r)
}

/// Turn `(distance in .weight)` candidates into tapered anchors:
/// w = ((R-d)/(R*d))^2, degrading to plain 1/d^2 when R = inf, and to equal
/// weights when every candidate sits exactly on the support boundary (equal
/// distances -> plain inverse-distance after normalization, order-invariant).
fn finish_hold_anchors(
    cfg: &OpInterpConfig,
    coords: &[f64],
    mut picked: Vec<HoldAnchor>,
    support_r: f64,
) -> Result<Vec<HoldAnchor>, AicError> {
    if picked.is_empty() {
        return Err(miss(cfg, coords, "no positive-util boundary anchor"));
    }
    let taper = |d: f64| -> f64 {
        if support_r.is_infinite() {
            1.0 / (d * d + 1e-12)
        } else {
            let t = (support_r - d).max(0.0) / (support_r * d + 1e-12);
            t * t
        }
    };
    let mut wsum = 0.0;
    for a in &mut picked {
        a.weight = taper(a.weight); // .weight carried the distance until here
        wsum += a.weight;
    }
    if wsum <= 0.0 {
        for a in &mut picked {
            a.weight = 1.0;
        }
    }
    Ok(picked)
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
#[derive(Debug)]
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
                // Python `_site_index`: `log2(max(v, 1e-12))` — the SAME floor
                // as the query logs in `resolve`. A zero site coordinate (FPM
                // prefill KV=0 sites) must land at ~-39.86 on BOTH sides so
                // P=0 queries stay near P=0 sites; flooring sites at 1 (the
                // old GEMM-era shortcut, identical for v >= 1) pushed those
                // sites ~40 log2-units away from their own queries.
                let logs = k
                    .iter()
                    .map(|&v| (v as f64).max(1e-12).log2())
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
            own_curve_coverage_fallback,
            ..
        } = &cfg.resolver
        else {
            unreachable!()
        };

        let q = coords[*curve_axis];
        // Collected shape (integer site coords present in the index): its own
        // curve answers alone... unless the curve does not cover the query and
        // the config opted into coverage fallback: treat the own site as
        // absent for the transfer below.
        let mut excluded_site: Option<&[u32]> = None;
        let site_ints: Option<Vec<u32>> =
            site_axes.iter().map(|&p| as_exact_key(coords[p])).collect();
        if let Some(key) = &site_ints {
            if let Some(curve) = self.sites.get(key) {
                let covers = (curve[0].0 as f64) <= q && q <= (curve[curve.len() - 1].0 as f64);
                if !(*own_curve_coverage_fallback && !covers) {
                    return self.eval_curve(cfg, curve, key, q, coords);
                }
                excluded_site = self.sites.get_key_value(key).map(|(k, _)| k.as_slice());
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
        let is_excluded =
            |i: usize| excluded_site.is_some_and(|e| self.site_logs[i].0.as_slice() == e);
        let mut ranked: Vec<(f64, usize)> = Vec::with_capacity(self.site_logs.len());
        if *require_curve_coverage {
            for (i, (_, logs)) in self.site_logs.iter().enumerate() {
                let (lo, hi) = self.curve_bounds[i];
                if (lo as f64) <= q && q <= (hi as f64) && !is_excluded(i) {
                    ranked.push((dist(logs), i));
                }
            }
        }
        // No coverage requirement, or nothing covered q -> fall back to all
        // sites (each later held at its own curve end), still minus the
        // coverage-fallback-excluded own site.
        if ranked.is_empty() {
            for (i, (_, logs)) in self.site_logs.iter().enumerate() {
                if !is_excluded(i) {
                    ranked.push((dist(logs), i));
                }
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
        if !(sol_q.is_finite() && sol_q > 0.0) {
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
            if !(sol_q.is_finite() && sol_q > 0.0) {
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

/// Visit every leaf without materializing a leaf vector (the hold path's
/// single-pass selection uses this; `walk_leaves` remains for callers that
/// genuinely need the full collection).
fn visit_leaves(node: &Node, prefix: &mut Vec<u32>, f: &mut impl FnMut(&[u32], LeafValue)) {
    match node {
        Node::Leaf(v) => f(prefix, *v),
        Node::Branch(map) => {
            for (&k, child) in map {
                prefix.push(k);
                visit_leaves(child, prefix, f);
                prefix.pop();
            }
        }
    }
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
        gemm_cfg_with_fallback(sol, false)
    }

    fn gemm_cfg_with_fallback(
        sol: &dyn Fn(&[f64]) -> f64,
        own_curve_coverage_fallback: bool,
    ) -> OpInterpConfig<'_> {
        OpInterpConfig {
            axes: &["m", "n", "k"],
            resolver: Resolver::ScatteredSites {
                site_axes: vec![1, 2],
                curve_axis: 0,
                nn_sites: 4,
                max_site_distance: Some(2.0),
                require_curve_coverage: true,
                k_tail: 3,
                own_curve_coverage_fallback,
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

    /// Own site (4096, 1024) sweeps only m<=64 and its measured latencies run
    /// at 2x the formula (util 0.5); neighbour (5120, 2048) covers the full
    /// sweep at util 1. A query at the own site beyond its sweep must:
    /// - fallback=false: util-hold the OWN tail -> 2x the formula;
    /// - fallback=true: exclude the own site and transfer from the clean
    ///   neighbour -> the formula exactly (Python own_curve_coverage_fallback).
    fn short_own_site_table() -> Node {
        let mut t = Node::branch();
        for m in [16u32, 32, 64] {
            t.insert(
                &[m, 4096, 1024],
                2.0 * gemm_lat(&[m as f64, 4096.0, 1024.0]),
            );
        }
        for m in [16u32, 32, 64, 128, 256, 512, 1024] {
            t.insert(&[m, 5120, 2048], gemm_lat(&[m as f64, 5120.0, 2048.0]));
        }
        t
    }

    #[test]
    fn own_curve_fallback_off_holds_own_tail() {
        let t = short_own_site_table();
        let cfg = gemm_cfg_with_fallback(&gemm_lat, false);
        approx(
            query(&cfg, &t, &[512.0, 4096.0, 1024.0]).unwrap(),
            2.0 * gemm_lat(&[512.0, 4096.0, 1024.0]),
        );
    }

    #[test]
    fn own_curve_fallback_on_transfers_from_neighbours() {
        let t = short_own_site_table();
        let cfg = gemm_cfg_with_fallback(&gemm_lat, true);
        approx(
            query(&cfg, &t, &[512.0, 4096.0, 1024.0]).unwrap(),
            gemm_lat(&[512.0, 4096.0, 1024.0]),
        );
    }

    #[test]
    fn own_curve_fallback_in_range_still_answers_from_own_curve() {
        // Inside the own sweep the flag must change nothing: m=48 lerps on the
        // own (distorted) curve.
        let t = short_own_site_table();
        let cfg = gemm_cfg_with_fallback(&gemm_lat, true);
        approx(
            query(&cfg, &t, &[48.0, 4096.0, 1024.0]).unwrap(),
            2.0 * gemm_lat(&[48.0, 4096.0, 1024.0]),
        );
    }

    /// FPM prefill tables have legitimate KV=0 sites. A query at an
    /// uncollected batch with KV=0 must transfer from the KV=0 neighbours
    /// (both floored at 1e-12 -> distance is the batch axis alone), exactly
    /// like Python `_site_index`.
    #[test]
    fn zero_valued_site_axis_stays_near_zero_valued_queries() {
        // FPM-prefill-shaped table: axes (batch, P, KV), sites (batch, KV).
        let mut t = Node::branch();
        for b in [2u32, 4] {
            for p in [1024u32, 2048, 4096] {
                t.insert(&[b, p, 0], 1e-3 * (b * p) as f64);
            }
        }
        let lat = |c: &[f64]| 1e-3 * c[0] * c[1];
        let cfg = OpInterpConfig {
            axes: &["batch_size", "total_prefill_tokens", "total_kv_read_tokens"],
            resolver: Resolver::ScatteredSites {
                site_axes: vec![0, 2],
                curve_axis: 1,
                nn_sites: 4,
                max_site_distance: Some(2.0),
                require_curve_coverage: true,
                k_tail: 3,
                own_curve_coverage_fallback: true,
            },
            sol_fn: &lat,
            value_transform: ValueTransform::Raw,
            transform_axis: None,
        };
        // B=3 is uncollected; KV=0 matches the collected sites' zero axis.
        approx(
            query(&cfg, &t, &[3.0, 2048.0, 0.0]).unwrap(),
            lat(&[3.0, 2048.0]),
        );
    }

    #[test]
    fn own_curve_fallback_sole_site_is_a_structured_miss() {
        // The excluded own site is the ONLY site: no candidates survive the
        // distance gate -> miss, never self-anchor.
        let mut t = Node::branch();
        for m in [16u32, 32, 64] {
            t.insert(&[m, 4096, 1024], gemm_lat(&[m as f64, 4096.0, 1024.0]));
        }
        let cfg = gemm_cfg_with_fallback(&gemm_lat, true);
        assert!(query(&cfg, &t, &[512.0, 4096.0, 1024.0]).is_err());
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
            resolver: Resolver::Grid {
                k_tail: 3,
                nn_leaves: 4,
            },
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
                own_curve_coverage_fallback: false,
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

    // Multi-axis hold: joint-log2 kNN util transfer (the B200 gen-attn report
    // case). Staircase with a REGIME SPLIT: the deep b=128 row is collected at
    // exact physics while the short b=256 row ends early in a 1.4x-latency
    // regime (like the real 128K-token-capped b>=256 rows). Every query below
    // is past the frontier on both rows.
    fn gen_lat(c: &[f64]) -> f64 {
        1e-6 * c[0] * c[1] * c[2] // decode physics: [n][b][s], linear in both
    }

    fn gen_split_table() -> Node {
        let mut root = Node::branch();
        for s in [512u32, 1024, 2048] {
            root.insert(&[64, 128, s], gen_lat(&[64.0, 128.0, s as f64]));
        }
        for s in [128u32, 256, 512] {
            root.insert(&[64, 256, s], 1.4 * gen_lat(&[64.0, 256.0, s as f64]));
        }
        root
    }

    #[test]
    fn grid_hold_is_continuous_across_outer_midpoint() {
        // The nearest-path snap flipped anchors at the bracket midpoint
        // (b=192 -> row 128, b=193 -> row 256), a +36.9% cliff on the real
        // B200 table where measured hardware moves +0.17%. The kNN hold must
        // stay continuous: the 192->193 step may not exceed the SOL growth by
        // more than a percent.
        let t = gen_split_table();
        let cfg = OpInterpConfig::grid(&["num_heads", "batch", "seq_len"], &gen_lat);
        let lat_192 = query(&cfg, &t, &[64.0, 192.0, 4096.0]).unwrap();
        let lat_193 = query(&cfg, &t, &[64.0, 193.0, 4096.0]).unwrap();
        let sol_step = gen_lat(&[64.0, 193.0, 4096.0]) / gen_lat(&[64.0, 192.0, 4096.0]);
        assert!((lat_193 / lat_192 - sol_step).abs() < 0.01);
    }

    #[test]
    fn grid_hold_prefers_nearby_saturated_evidence() {
        // A query deep past the short row's end (b=256, s=4096) must not
        // inherit that row's unsaturated 1.4x boundary util verbatim (the old
        // snap did, +41% on the reported B200 case): the joint-log nearest
        // leaves are the deep saturated ones, so the answer lands near physics.
        let t = gen_split_table();
        let cfg = OpInterpConfig::grid(&["num_heads", "batch", "seq_len"], &gen_lat);
        let lat = query(&cfg, &t, &[64.0, 256.0, 4096.0]).unwrap();
        let ratio = lat / gen_lat(&[64.0, 256.0, 4096.0]);
        assert!(ratio < 1.15, "snap gave 1.4x, got {ratio}");
        assert!(ratio > 0.95, "got {ratio}");
    }

    /// Perf comparison for the hold selection (review P2): single-pass
    /// best-M buffer vs the former collect-all + full-sort. Not asserted (CI
    /// timing is flaky); run manually:
    /// `cargo test -p aiconfigurator-core --release hold_selection_bench -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn hold_selection_bench() {
        use std::time::Instant;
        // ~1.9k leaves, the size of the real b200 gen-attention slice.
        let mut t = Node::branch();
        for n in [8u32, 16, 32, 64] {
            for b in [1u32, 2, 4, 8, 16, 32, 64, 128, 256, 512] {
                for s in [
                    2u32, 8, 32, 128, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
                ] {
                    t.insert(&[n, b, s], 1e-6 * (n * b) as f64 * s as f64);
                }
            }
        }
        let sol = |c: &[f64]| c[0] * c[1] * c[2];
        let cfg = OpInterpConfig::grid(&["n", "b", "s"], &sol);
        let coords: [f64; 3] = [64.0, 192.0, 131072.0]; // past-frontier hold query

        let old_selection = || {
            let mut leaves: Vec<(Vec<u32>, LeafValue)> = Vec::new();
            walk_leaves(&t, &mut Vec::new(), &mut leaves);
            let q_log: Vec<f64> = coords.iter().map(|&v: &f64| v.max(1e-12).log2()).collect();
            let mut ranked: Vec<(f64, usize)> = leaves
                .iter()
                .enumerate()
                .map(|(i, (c, _))| {
                    let dd: f64 = c
                        .iter()
                        .zip(&q_log)
                        .map(|(&v, ql)| {
                            let delta = ((v as f64).max(1e-12)).log2() - ql;
                            delta * delta
                        })
                        .sum();
                    (dd.sqrt(), i)
                })
                .collect();
            ranked.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
            ranked[..5].to_vec()
        };

        const ITERS: usize = 10_000;
        let start = Instant::now();
        for _ in 0..ITERS {
            std::hint::black_box(old_selection());
        }
        let old_ns = start.elapsed().as_nanos() / ITERS as u128;
        let start = Instant::now();
        for _ in 0..ITERS {
            std::hint::black_box(hold_anchor_weights(&cfg, &t, &coords, 4).unwrap());
        }
        let new_ns = start.elapsed().as_nanos() / ITERS as u128;
        println!("hold selection on 1920 leaves: collect+sort {old_ns} ns/query, single-pass {new_ns} ns/query");
    }

    #[test]
    fn grid_hold_boundary_ties_carry_zero_weight() {
        // Power-of-two grids tie EXACTLY. Under the tapered weights, leaves
        // sitting on the support radius R (the 5th valid distance) weigh
        // ZERO, so which side of the nn_leaves cutoff a tied leaf lands on
        // cannot matter — no tie-ordering rule needed, and the blend is
        // independent of axis order and table insertion order.
        // Query (64,32,1) is below the collected s range: one leaf at d=1,
        // then a FOUR-way tie at sqrt(2) == R -> every tie member weighs 0
        // and the answer equals the nearest anchor's held util exactly.
        let lat_of: &[(&[u32; 3], f64)] = &[
            (&[64, 32, 2], 1.00), // d=1: the only anchor with non-zero weight
            (&[64, 16, 2], 1.10), // d=sqrt(2) x4, distinct latencies so any
            (&[64, 64, 2], 1.30), // leaked tie weight would shift the blend
            (&[32, 32, 2], 1.50),
            (&[128, 32, 2], 1.70),
            (&[64, 32, 4], 2.00), // d=2, beyond the support: excluded
        ];
        let mut t = Node::branch();
        for (c, lat) in lat_of {
            t.insert(*c, *lat);
        }
        let sol = |_: &[f64]| 1.0; // constant SOL isolates the anchor selection
        let cfg = OpInterpConfig::grid(&["num_heads", "batch", "seq_len"], &sol);

        let expected = 1.00; // util held from the single live anchor
        let lat = query(&cfg, &t, &[64.0, 32.0, 1.0]).unwrap();
        assert!(
            (lat - expected).abs() <= 1e-9 * expected,
            "got {lat}, want {expected}"
        );

        let mut swapped = Node::branch();
        for (c, lat) in lat_of {
            swapped.insert(&[c[0], c[2], c[1]], *lat);
        }
        let cfg_swapped = OpInterpConfig::grid(&["num_heads", "seq_len", "batch"], &sol);
        let lat_swapped = query(&cfg_swapped, &swapped, &[64.0, 1.0, 32.0]).unwrap();
        assert!(
            (lat - lat_swapped).abs() <= 1e-12 * lat,
            "nesting changed the tie blend"
        );
    }

    #[test]
    fn grid_hold_is_continuous_across_rank_swaps() {
        // The support-radius taper sends a neighbour's weight to ZERO as it
        // leaves the nn_leaves selection, so the estimate is continuous when
        // leaves 4 and 5 exchange rank — the hard cutoff traded them at full
        // weight. L4=(16,.) and L5=(1024,.) swap ranks at x=128 with 100x
        // different latencies; epsilon steps across the swap must move the
        // estimate by less than 0.01%.
        let lat_of: &[(&[u32; 2], f64)] = &[
            (&[128, 8], 1.0),
            (&[96, 8], 1.1),
            (&[160, 8], 1.2),
            (&[16, 8], 0.5),
            (&[1024, 8], 50.0),
        ];
        let mut t = Node::branch();
        for (c, lat) in lat_of {
            t.insert(*c, *lat);
        }
        let sol = |_: &[f64]| 1.0;
        let cfg = OpInterpConfig::grid(&["a", "b"], &sol);
        let lo = query(&cfg, &t, &[128.0 * (1.0 - 1e-6), 32.0]).unwrap();
        let hi = query(&cfg, &t, &[128.0 * (1.0 + 1e-6), 32.0]).unwrap();
        assert!(
            (hi - lo).abs() / lo < 1e-4,
            "rank swap discontinuity: {lo} vs {hi}"
        );
    }

    #[test]
    fn grid_hold_is_axis_order_independent() {
        // The hold works on joint coordinates, not the nesting order: the
        // same data nested [n][b][s] and [n][s][b] must answer a past-frontier
        // query identically.
        let t = gen_split_table();
        let cfg = OpInterpConfig::grid(&["num_heads", "batch", "seq_len"], &gen_lat);
        let mut swapped = Node::branch();
        let mut leaves: Vec<(Vec<u32>, LeafValue)> = Vec::new();
        walk_leaves(&t, &mut Vec::new(), &mut leaves);
        for (c, leaf) in &leaves {
            swapped.insert_value(&[c[0], c[2], c[1]], *leaf);
        }
        let sol_swapped = |c: &[f64]| gen_lat(&[c[0], c[2], c[1]]);
        let cfg_swapped = OpInterpConfig::grid(&["num_heads", "seq_len", "batch"], &sol_swapped);
        let lat = query(&cfg, &t, &[64.0, 200.0, 4096.0]).unwrap();
        let lat_swapped = query(&cfg_swapped, &swapped, &[64.0, 4096.0, 200.0]).unwrap();
        assert!((lat - lat_swapped).abs() <= 1e-12 * lat.abs());
    }
}
