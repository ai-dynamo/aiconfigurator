// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Table-view folds: the engine-side mirror of the retired Python
//! `load_*_data` parsers (PR-6, #1357 phase 3).
//!
//! Each `view_*` function here reproduces, row for row, what one Python
//! loader in `aiconfigurator_core/sdk/operations/*.py` produced before the
//! Python data plane was deleted: the same source order (shared-layer
//! first-wins), the same column reads and casts, the same key layering, the
//! same derived fields (`energy = power * latency`), and the same INSERTION
//! ORDER (Python dicts preserve it and downstream consumers assign chart
//! colors positionally from key order).
//!
//! The folds deliberately do NOT reuse the query-path grids in the sibling
//! modules: those apply load-time normalizations the raw Python tables never
//! had (GEMM SOL clamping, attention key rewrites, phase folding, leaf
//! narrowing to bare f64). A view is the RAW collected data plane; queries
//! stay on their own clamped/indexed structures.
//!
//! Output is a JSON document built by the local writer below. Keys are plain
//! strings (quant names, decimal integers, `|`-joined tuples); the Python
//! side rehydrates them into enum/int/tuple keys per family. Leaves are JSON
//! objects carrying at least a `latency` field — the Python side (and the
//! baseline codec in `tests/cross_package/_data_plane_codec.py`) recognizes
//! leaves by that field, never by depth.

use std::fmt::Write as _;

use crate::common::error::AicError;
use crate::perf_database::parquet_loader::{PerfReader, PerfRow};
use crate::perf_database::source_resolution::PrioritizedSource;
use crate::perf_database::{kernel_source_ok, PerfSource, PerfTables};

/// Leaf field value. Integer-typed Python fields must serialize WITHOUT a
/// decimal point so `json.loads` rehydrates the exact Python type
/// (`int` vs `float`) the deleted loader stored.
#[derive(Clone, Debug)]
pub enum ViewValue {
    F64(f64),
    I64(i64),
    Str(String),
    Bool(bool),
}

/// Insertion-ordered nested mapping — the Rust twin of the Python loaders'
/// nested dicts. `Branch` entries keep first-seen key order; `Leaf` keeps the
/// loader's field order (not a contract, but free to preserve).
#[derive(Clone, Debug)]
pub enum ViewNode {
    Branch(Vec<(String, ViewNode)>),
    Leaf(Vec<(&'static str, ViewValue)>),
}

impl ViewNode {
    pub fn branch() -> Self {
        ViewNode::Branch(Vec::new())
    }

    fn entries_mut(&mut self) -> &mut Vec<(String, ViewNode)> {
        match self {
            ViewNode::Branch(entries) => entries,
            ViewNode::Leaf(_) => unreachable!("leaf reached where a branch was expected"),
        }
    }

    fn child_index(&self, key: &str) -> Option<usize> {
        match self {
            ViewNode::Branch(entries) => entries.iter().position(|(k, _)| k == key),
            ViewNode::Leaf(_) => None,
        }
    }

    /// Walk to the branch at `path`, vivifying missing levels in insertion
    /// order — the shared descent under every insert flavor (and Python's
    /// defaultdict vivification contract).
    fn descend(&mut self, path: &[String]) -> &mut ViewNode {
        let mut node = self;
        for key in path {
            let idx = match node.child_index(key) {
                Some(i) => i,
                None => {
                    let entries = node.entries_mut();
                    entries.push((key.clone(), ViewNode::branch()));
                    entries.len() - 1
                }
            };
            node = &mut node.entries_mut()[idx].1;
        }
        node
    }

    /// Descend to (creating) the branch at `path`, then insert `leaf` under
    /// `last` ONLY if absent — the Python loaders' first-wins conflict rule.
    /// Returns false on conflict (caller may log/count).
    pub fn insert_first_wins(&mut self, path: &[String], last: &str, leaf: ViewNode) -> bool {
        let node = self.descend(path);
        if node.child_index(last).is_some() {
            return false;
        }
        node.entries_mut().push((last.to_string(), leaf));
        true
    }

    /// Like `insert_first_wins` but REPLACES an existing leaf — the unified
    /// moe_a2a/moe_ep loaders' "first new-schema occurrence overwrites a
    /// legacy-adapted leaf" path. Replacement re-uses the existing slot, so
    /// the key keeps its original insertion position (Python dict semantics).
    pub fn insert_overwrite(&mut self, path: &[String], last: &str, leaf: ViewNode) {
        let node = self.descend(path);
        match node.child_index(last) {
            Some(i) => node.entries_mut()[i].1 = leaf,
            None => node.entries_mut().push((last.to_string(), leaf)),
        }
    }

    /// Recursively fill keys that are absent from this node while preserving
    /// every existing leaf and its insertion position. Used when a completed
    /// lower-priority source tier is admitted only for missing coordinates.
    fn merge_fill(&mut self, lower_priority: ViewNode) {
        let (ViewNode::Branch(existing), ViewNode::Branch(lower_entries)) = (self, lower_priority)
        else {
            return;
        };
        for (key, lower_child) in lower_entries {
            if let Some(index) = existing
                .iter()
                .position(|(existing_key, _)| existing_key == &key)
            {
                existing[index].1.merge_fill(lower_child);
            } else {
                existing.push((key, lower_child));
            }
        }
    }

    /// Serialize to JSON preserving entry order. Rust's `{:?}` float
    /// formatting is shortest-roundtrip, so `json.loads` on the Python side
    /// reconstructs the bit-identical f64.
    pub fn to_json(&self) -> String {
        let mut out = String::new();
        self.write_json(&mut out);
        out
    }

    fn write_json(&self, out: &mut String) {
        match self {
            ViewNode::Branch(entries) => {
                out.push('{');
                for (i, (key, value)) in entries.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    write_json_string(out, key);
                    out.push(':');
                    value.write_json(out);
                }
                out.push('}');
            }
            ViewNode::Leaf(fields) => {
                out.push('{');
                for (i, (key, value)) in fields.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    write_json_string(out, key);
                    out.push(':');
                    match value {
                        ViewValue::F64(v) => write_json_f64(out, *v),
                        ViewValue::I64(v) => {
                            let _ = write!(out, "{v}");
                        }
                        ViewValue::Str(v) => write_json_string(out, v),
                        ViewValue::Bool(v) => out.push_str(if *v { "true" } else { "false" }),
                    }
                }
                out.push('}');
            }
        }
    }
}

fn write_json_string(out: &mut String, value: &str) {
    out.push('"');
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

fn write_json_f64(out: &mut String, value: f64) {
    if value.is_finite() {
        // {:?} on f64 emits the shortest string that round-trips; integral
        // values print as "1.0" which json.loads parses back to Python float
        // — matching the Python loaders' float() casts.
        let _ = write!(out, "{value:?}");
    } else if value.is_nan() {
        // The retired Python loaders stored NaN cells verbatim (bare
        // float()), and the consumer decodes with Python's json.loads, whose
        // dialect accepts these tokens — so a NaN survives the round trip
        // instead of panicking across the FFI.
        out.push_str("NaN");
    } else if value > 0.0 {
        out.push_str("Infinity");
    } else {
        out.push_str("-Infinity");
    }
}

/// The standard `{latency, power, energy}` leaf every classic loader stores,
/// with the Python loaders' backward-compat default `power=0.0` and derived
/// `energy = power * latency` (W·ms).
fn classic_leaf(latency: f64, power: f64) -> ViewNode {
    ViewNode::Leaf(vec![
        ("latency", ViewValue::F64(latency)),
        ("power", ViewValue::F64(power)),
        ("energy", ViewValue::F64(power * latency)),
    ])
}

/// One decoded row surfaced to a fold closure: the reader, the row, and the
/// resolved optional `power` column (shared by nearly every classic loader).
struct RowCtx<'a> {
    reader: &'a PerfReader,
    row: &'a PerfRow,
    power_col: Option<usize>,
    ks_col: Option<usize>,
    path: &'a std::path::Path,
}

impl RowCtx<'_> {
    /// Python's `float(row.get("power", 0.0))` — an absent COLUMN defaults
    /// to 0.0, but a present-but-null cell fails loudly (parquet nulls read
    /// back as `""` through `_read_perf_rows`, and `float("")` raised): the
    /// classic loaders' documented fail-loud contract — only the moe_comm
    /// family was deliberately lenient (`row_power_lenient`).
    fn power(&self) -> Result<f64, AicError> {
        match self.power_col {
            Some(col) => self.row.f64(col),
            None => Ok(0.0),
        }
    }
}

/// Optional string column with a default when the COLUMN is absent; a null
/// CELL still reads as "" — Python's `row.get(name, default)` distinction
/// (the default applies only when the key is missing entirely).
fn str_col_or(ctx: &RowCtx<'_>, name: &str, default: &str) -> Result<String, AicError> {
    match ctx.reader.col_optional(name) {
        Some(col) => Ok(ctx.row.str_optional(Some(col))?.unwrap_or("").to_string()),
        None => Ok(default.to_string()),
    }
}

/// Iterate every row of every existing source in order, applying the
/// per-source `kernel_source` filter — the exact semantics of Python's
/// `_read_filtered_rows(sources)`: missing files are skipped; `None` is
/// returned (as `Ok(None)`) only when EVERY source path is missing. (The
/// legacy `.parquet`→`.txt` CSV fallback is NOT mirrored: the Rust query
/// loaders never supported it and no shipped data uses it.)
fn fold_sources<F>(sources: &[PerfSource], mut per_row: F) -> Result<Option<()>, AicError>
where
    F: FnMut(&RowCtx<'_>) -> Result<(), AicError>,
{
    fold_sources_grouped(sources, || (), |ctx, ()| per_row(ctx), |()| Ok(()))
}

/// `fold_sources` with a per-source accumulator — for folds whose merge rule
/// distinguishes WITHIN-source from ACROSS-source conflicts (DSA:
/// last-row-wins within one file, first-source-wins across files). Each
/// existing source gets a fresh `new_state()`; its rows run through
/// `per_row(ctx, &mut state)`; `flush(state)` fires at the source boundary.
fn fold_sources_grouped<S, N, F, G>(
    sources: &[PerfSource],
    mut new_state: N,
    mut per_row: F,
    mut flush: G,
) -> Result<Option<()>, AicError>
where
    N: FnMut() -> S,
    F: FnMut(&RowCtx<'_>, &mut S) -> Result<(), AicError>,
    G: FnMut(S) -> Result<(), AicError>,
{
    let mut any_exists = false;
    for source in sources {
        let path = &source.0;
        if !path.exists() {
            continue;
        }
        any_exists = true;
        let reader = PerfReader::open(path)?;
        let power_col = reader.col_optional("power");
        let ks_col = reader.col_optional("kernel_source");
        let mut state = new_state();
        for row in reader.rows()? {
            let row = row?;
            if !kernel_source_ok(source.1.as_deref(), ks_col, &row)? {
                continue;
            }
            per_row(
                &RowCtx {
                    reader: &reader,
                    row: &row,
                    power_col,
                    ks_col,
                    path,
                },
                &mut state,
            )?;
        }
        flush(state)?;
    }
    Ok(if any_exists { Some(()) } else { None })
}

/// `operations/gemm.py::load_gemm_data` —
/// `[gemm_dtype][m][n][k] -> {latency, power, energy}`, first-wins.
pub fn view_gemm(sources: &[PerfSource]) -> Result<Option<ViewNode>, AicError> {
    let mut root = ViewNode::branch();
    let found = fold_sources(sources, |ctx| {
        let r = ctx.reader;
        let quant = ctx.row.str_owned(r.col("gemm_dtype")?)?;
        let m = ctx.row.u32(r.col("m")?)?;
        let n = ctx.row.u32(r.col("n")?)?;
        let k = ctx.row.u32(r.col("k")?)?;
        let latency = ctx.row.f64(r.col("latency")?)?;
        let path = [quant, m.to_string(), n.to_string()];
        root.insert_first_wins(&path, &k.to_string(), classic_leaf(latency, ctx.power()?));
        Ok(())
    })?;
    Ok(found.map(|_| root))
}

/// `operations/gemm.py::load_compute_scale_data` / `load_scale_matrix_data` —
/// `[quant_dtype][m][k] -> {latency, power, energy}`, first-wins.
pub fn view_gemm_scale(sources: &[PerfSource]) -> Result<Option<ViewNode>, AicError> {
    let mut root = ViewNode::branch();
    let found = fold_sources(sources, |ctx| {
        let r = ctx.reader;
        let quant = ctx.row.str_owned(r.col("quant_dtype")?)?;
        let m = ctx.row.u32(r.col("m")?)?;
        let k = ctx.row.u32(r.col("k")?)?;
        let latency = ctx.row.f64(r.col("latency")?)?;
        let path = [quant, m.to_string()];
        root.insert_first_wins(&path, &k.to_string(), classic_leaf(latency, ctx.power()?));
        Ok(())
    })?;
    Ok(found.map(|_| root))
}

/// `operations/attention.py::load_context_attention_data` —
/// `[attn_dtype][kv_cache_dtype][kv_n][head_size][window_size][n][s][b]`.
/// The `kv_n == n -> 0` rewrite, the missing-`window_size` -> 0 default, and
/// first-wins are all the Python loader's own semantics.
pub fn view_context_attention(sources: &[PerfSource]) -> Result<Option<ViewNode>, AicError> {
    let mut root = ViewNode::branch();
    let found = fold_sources(sources, |ctx| {
        let r = ctx.reader;
        let quant = ctx.row.str_owned(r.col("attn_dtype")?)?;
        let kv_dtype = ctx.row.str_owned(r.col("kv_cache_dtype")?)?;
        let b = ctx.row.u32(r.col("batch_size")?)?;
        let s = ctx.row.u32(r.col("isl")?)?;
        let n = ctx.row.u32(r.col("num_heads")?)?;
        let mut kv_n = ctx.row.u32(r.col("num_key_value_heads")?)?;
        let head_size = ctx.row.u32(r.col("head_dim")?)?;
        // Python: `try: row["window_size"] except KeyError: 0`, then int(x) —
        // the 0 default applies ONLY when the COLUMN is absent. A null cell
        // or NaN failed the load loudly, and a DOUBLE cell loaded its true
        // (truncated) value — never a silent 0 that would conflate SWA rows
        // with global-attention rows.
        let window_size = match r.col_optional("window_size") {
            Some(col) => int_cell_loud(ctx, col, "window_size")?,
            None => 0,
        };
        let latency = ctx.row.f64(r.col("latency")?)?;
        if kv_n == n {
            kv_n = 0;
        }
        let path = [
            quant,
            kv_dtype,
            kv_n.to_string(),
            head_size.to_string(),
            window_size.to_string(),
            n.to_string(),
            s.to_string(),
        ];
        root.insert_first_wins(&path, &b.to_string(), classic_leaf(latency, ctx.power()?));
        Ok(())
    })?;
    Ok(found.map(|_| root))
}

/// `operations/attention.py::load_generation_attention_data` —
/// `[kv_cache_dtype][kv_n][head_size][window_size][n][b][s+step]`.
pub fn view_generation_attention(sources: &[PerfSource]) -> Result<Option<ViewNode>, AicError> {
    let mut root = ViewNode::branch();
    let found = fold_sources(sources, |ctx| {
        let r = ctx.reader;
        let kv_dtype = ctx.row.str_owned(r.col("kv_cache_dtype")?)?;
        let b = ctx.row.u32(r.col("batch_size")?)?;
        let s = ctx.row.u32(r.col("isl")?)?;
        let n = ctx.row.u32(r.col("num_heads")?)?;
        let mut kv_n = ctx.row.u32(r.col("num_key_value_heads")?)?;
        let head_size = ctx.row.u32(r.col("head_dim")?)?;
        let step = ctx.row.u32(r.col("step")?)?;
        // Python: `try: row["window_size"] except KeyError: 0`, then int(x) —
        // the 0 default applies ONLY when the COLUMN is absent. A null cell
        // or NaN failed the load loudly, and a DOUBLE cell loaded its true
        // (truncated) value — never a silent 0 that would conflate SWA rows
        // with global-attention rows.
        let window_size = match r.col_optional("window_size") {
            Some(col) => int_cell_loud(ctx, col, "window_size")?,
            None => 0,
        };
        let latency = ctx.row.f64(r.col("latency")?)?;
        if kv_n == n {
            kv_n = 0;
        }
        let full_seq = s + step;
        let path = [
            kv_dtype,
            kv_n.to_string(),
            head_size.to_string(),
            window_size.to_string(),
            n.to_string(),
            b.to_string(),
        ];
        root.insert_first_wins(
            &path,
            &full_seq.to_string(),
            classic_leaf(latency, ctx.power()?),
        );
        Ok(())
    })?;
    Ok(found.map(|_| root))
}

/// `operations/attention.py::load_encoder_attention_data` —
/// `[attn_dtype][head_size][n][s][b]` (MHA-only, no KV/window axes).
pub fn view_encoder_attention(sources: &[PerfSource]) -> Result<Option<ViewNode>, AicError> {
    let mut root = ViewNode::branch();
    let found = fold_sources(sources, |ctx| {
        let r = ctx.reader;
        let quant = ctx.row.str_owned(r.col("attn_dtype")?)?;
        let b = ctx.row.u32(r.col("batch_size")?)?;
        let s = ctx.row.u32(r.col("isl")?)?;
        let n = ctx.row.u32(r.col("num_heads")?)?;
        let head_size = ctx.row.u32(r.col("head_dim")?)?;
        let latency = ctx.row.f64(r.col("latency")?)?;
        let path = [quant, head_size.to_string(), n.to_string(), s.to_string()];
        root.insert_first_wins(&path, &b.to_string(), classic_leaf(latency, ctx.power()?));
        Ok(())
    })?;
    Ok(found.map(|_| root))
}

/// `operations/communication.py::load_custom_allreduce_data` —
/// `[CommQuantMode.half][num_gpus]["AUTO"][message_size]`. The dtype level is
/// the loader's hardcoded `CommQuantMode.half` (its `# TODO`), the strategy
/// level a constant "AUTO", and `*_eager` rows (kernel_source OR backend
/// suffix) are dropped — EXCEPT on b60, detected by "b60" in any source path.
pub fn view_custom_allreduce(sources: &[PerfSource]) -> Result<Option<ViewNode>, AicError> {
    let is_b60 = sources
        .iter()
        .any(|s| s.0.to_string_lossy().contains("b60"));
    let mut root = ViewNode::branch();
    let found = fold_sources(sources, |ctx| {
        let r = ctx.reader;
        let kernel_source = ctx
            .row
            .str_optional(r.col_optional("kernel_source"))?
            .unwrap_or("")
            .to_string();
        let backend = ctx
            .row
            .str_optional(r.col_optional("backend"))?
            .unwrap_or("")
            .to_string();
        if (kernel_source.ends_with("_eager") || backend.ends_with("_eager")) && !is_b60 {
            return Ok(());
        }
        let tp_size = ctx.row.u32(r.col("num_gpus")?)?;
        let message_size = ctx.row.u64(r.col("message_size")?)?;
        let latency = ctx.row.f64(r.col("latency")?)?;
        let path = ["half".to_string(), tp_size.to_string(), "AUTO".to_string()];
        root.insert_first_wins(
            &path,
            &message_size.to_string(),
            classic_leaf(latency, ctx.power()?),
        );
        Ok(())
    })?;
    Ok(found.map(|_| root))
}

/// `operations/communication.py::load_nccl_data` —
/// `[nccl_dtype][op_name][num_gpus][message_size]` (also serves oneccl).
pub fn view_nccl(sources: &[PerfSource]) -> Result<Option<ViewNode>, AicError> {
    let mut root = ViewNode::branch();
    let found = fold_sources(sources, |ctx| {
        let r = ctx.reader;
        let dtype = ctx.row.str_owned(r.col("nccl_dtype")?)?;
        let num_gpus = ctx.row.u32(r.col("num_gpus")?)?;
        let message_size = ctx.row.u64(r.col("message_size")?)?;
        let op_name = str_cell_or_empty(ctx, "op_name")?;
        let latency = ctx.row.f64(r.col("latency")?)?;
        let path = [dtype, op_name, num_gpus.to_string()];
        root.insert_first_wins(
            &path,
            &message_size.to_string(),
            classic_leaf(latency, ctx.power()?),
        );
        Ok(())
    })?;
    Ok(found.map(|_| root))
}

/// `operations/mla.py::load_context_mla_data` —
/// `[mla_dtype][kv_cache_dtype][num_heads][s][b]` (num_heads mandatory, #1458).
pub fn view_context_mla(sources: &[PerfSource]) -> Result<Option<ViewNode>, AicError> {
    let mut root = ViewNode::branch();
    let found = fold_sources(sources, |ctx| {
        let r = ctx.reader;
        let quant = ctx.row.str_owned(r.col("mla_dtype")?)?;
        let kv_dtype = ctx.row.str_owned(r.col("kv_cache_dtype")?)?;
        let b = ctx.row.u32(r.col("batch_size")?)?;
        let s = ctx.row.u32(r.col("isl")?)?;
        let num_heads = ctx.row.u32(r.col("num_heads")?)?;
        let latency = ctx.row.f64(r.col("latency")?)?;
        let path = [quant, kv_dtype, num_heads.to_string(), s.to_string()];
        root.insert_first_wins(&path, &b.to_string(), classic_leaf(latency, ctx.power()?));
        Ok(())
    })?;
    Ok(found.map(|_| root))
}

/// `operations/mla.py::load_generation_mla_data` —
/// `[kv_cache_dtype][num_heads][b][s+step]` (mla_dtype ignored on decode).
pub fn view_generation_mla(sources: &[PerfSource]) -> Result<Option<ViewNode>, AicError> {
    let mut root = ViewNode::branch();
    let found = fold_sources(sources, |ctx| {
        let r = ctx.reader;
        let kv_dtype = ctx.row.str_owned(r.col("kv_cache_dtype")?)?;
        let b = ctx.row.u32(r.col("batch_size")?)?;
        let s = ctx.row.u32(r.col("isl")?)?;
        let step = ctx.row.u32(r.col("step")?)?;
        let num_heads = ctx.row.u32(r.col("num_heads")?)?;
        let latency = ctx.row.f64(r.col("latency")?)?;
        let full_seq = s + step;
        let path = [kv_dtype, num_heads.to_string(), b.to_string()];
        root.insert_first_wins(
            &path,
            &full_seq.to_string(),
            classic_leaf(latency, ctx.power()?),
        );
        Ok(())
    })?;
    Ok(found.map(|_| root))
}

/// `operations/mla.py::load_mla_bmm_data` —
/// `[bmm_dtype][op_name][num_heads][num_tokens]`.
pub fn view_mla_bmm(sources: &[PerfSource]) -> Result<Option<ViewNode>, AicError> {
    let mut root = ViewNode::branch();
    let found = fold_sources(sources, |ctx| {
        let r = ctx.reader;
        let quant = ctx.row.str_owned(r.col("bmm_dtype")?)?;
        let num_tokens = ctx.row.u32(r.col("num_tokens")?)?;
        let num_heads = ctx.row.u32(r.col("num_heads")?)?;
        let op_name = str_cell_or_empty(ctx, "op_name")?;
        let latency = ctx.row.f64(r.col("latency")?)?;
        let path = [quant, op_name, num_heads.to_string()];
        root.insert_first_wins(
            &path,
            &num_tokens.to_string(),
            classic_leaf(latency, ctx.power()?),
        );
        Ok(())
    })?;
    Ok(found.map(|_| root))
}

/// `operations/mla.py::load_wideep_context_mla_data` —
/// `[kernel_source][mla_dtype][kv_cache_dtype][num_heads][s][b]`; a missing
/// kernel_source COLUMN defaults to "flashinfer" (a null cell stays "").
pub fn view_wideep_context_mla(sources: &[PerfSource]) -> Result<Option<ViewNode>, AicError> {
    let mut root = ViewNode::branch();
    let found = fold_sources(sources, |ctx| {
        let r = ctx.reader;
        let kernel_source = str_col_or(ctx, "kernel_source", "flashinfer")?;
        let quant = ctx.row.str_owned(r.col("mla_dtype")?)?;
        let kv_dtype = ctx.row.str_owned(r.col("kv_cache_dtype")?)?;
        let b = ctx.row.u32(r.col("batch_size")?)?;
        let s = ctx.row.u32(r.col("isl")?)?;
        let num_heads = ctx.row.u32(r.col("num_heads")?)?;
        let latency = ctx.row.f64(r.col("latency")?)?;
        let path = [
            kernel_source,
            quant,
            kv_dtype,
            num_heads.to_string(),
            s.to_string(),
        ];
        root.insert_first_wins(&path, &b.to_string(), classic_leaf(latency, ctx.power()?));
        Ok(())
    })?;
    Ok(found.map(|_| root))
}

/// `operations/mla.py::load_wideep_generation_mla_data` —
/// `[kernel_source][kv_cache_dtype][num_heads][b][s+step]`.
pub fn view_wideep_generation_mla(sources: &[PerfSource]) -> Result<Option<ViewNode>, AicError> {
    let mut root = ViewNode::branch();
    let found = fold_sources(sources, |ctx| {
        let r = ctx.reader;
        let kernel_source = str_col_or(ctx, "kernel_source", "flashinfer")?;
        let kv_dtype = ctx.row.str_owned(r.col("kv_cache_dtype")?)?;
        let b = ctx.row.u32(r.col("batch_size")?)?;
        let s = ctx.row.u32(r.col("isl")?)?;
        let step = ctx.row.u32(r.col("step")?)?;
        let num_heads = ctx.row.u32(r.col("num_heads")?)?;
        let latency = ctx.row.f64(r.col("latency")?)?;
        let full_seq = s + step;
        let path = [
            kernel_source,
            kv_dtype,
            num_heads.to_string(),
            b.to_string(),
        ];
        root.insert_first_wins(
            &path,
            &full_seq.to_string(),
            classic_leaf(latency, ctx.power()?),
        );
        Ok(())
    })?;
    Ok(found.map(|_| root))
}

/// Per-row native-head resolution for the MLA module views: the model pin
/// itself is `perf_database/mla.rs::mla_module_native_heads` (ONE table for
/// the query loader and the view — landing new module data extends it once);
/// this wrapper adds the view-side fail-loud messages and the per-row
/// rank-local consistency check (#1429/#1458).
fn view_mla_module_native_heads(ctx: &RowCtx<'_>, num_heads: u32) -> Result<u32, AicError> {
    let r = ctx.reader;
    let model = ctx
        .row
        .str_optional(r.col_optional("model"))?
        .unwrap_or("")
        .to_string();
    if model.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "MLA module row in {} carries no model column; the module table keys its \
             native-head identity off the model pin (#1458).",
            ctx.path.display()
        )));
    }
    let native = crate::perf_database::mla::mla_module_native_heads(&model).ok_or_else(|| {
        AicError::PerfDatabase(format!(
            "MLA module row in {} names unpinned model {model:?}; add its native head \
             count to the module native-head pin (perf_database/mla.rs) when landing \
             the data (#1458).",
            ctx.path.display()
        ))
    })?;
    // Python: max(1, int(row.get("tp_size", 1) or 1)) — an absent column, a
    // null cell and a falsy 0 all fall to 1, and a DOUBLE cell counts like
    // int(float(x)). A NaN/inf cell made that int() raise: the load failed
    // LOUDLY instead of quietly disarming the #1429 rank-local guard.
    let tp_size = match num_optional(ctx.row, r.col_optional("tp_size"))? {
        None => 1,
        Some(v) if !v.is_finite() => {
            return Err(AicError::PerfDatabase(format!(
                "non-finite tp_size cell in an MLA module row at {}: the #1429 rank-local \
                 guard needs a parseable tp_size (the retired loader's int() raised here)",
                ctx.path.display()
            )))
        }
        Some(v) => (v as i64).max(1) as u32,
    };
    if tp_size > 1 && num_heads.saturating_mul(tp_size) != native {
        return Err(AicError::PerfDatabase(format!(
            "MLA module row in {} for model {model:?} has num_heads={num_heads} at \
             tp_size={tp_size}, inconsistent with native {native} (num_heads must be \
             rank-local, #1429/#1458).",
            ctx.path.display()
        )));
    }
    Ok(native)
}

/// The module loaders' `has_power` is decided by the FIRST concatenated row
/// (i.e. the first existing source's schema): if that file has no power
/// column, every later row's power is forced to 0.0 even when its own file
/// carries one. Tracks Python's `has_power = "power" in rows[0]` exactly.
struct FirstFilePower(Option<bool>);

impl FirstFilePower {
    fn new() -> Self {
        FirstFilePower(None)
    }
    fn power(&mut self, ctx: &RowCtx<'_>) -> Result<f64, AicError> {
        let has_power = *self.0.get_or_insert(ctx.power_col.is_some());
        if has_power {
            ctx.power()
        } else {
            Ok(0.0)
        }
    }
}

/// `operations/mla.py::load_context_mla_module_data` —
/// `[mla_dtype][kv_cache_dtype][gemm_type][native][num_heads][s][b]`.
pub fn view_context_mla_module(sources: &[PerfSource]) -> Result<Option<ViewNode>, AicError> {
    let mut root = ViewNode::branch();
    let mut first_power = FirstFilePower::new();
    let found = fold_sources(sources, |ctx| {
        let r = ctx.reader;
        let num_heads = ctx.row.u32(r.col("num_heads")?)?;
        let native = view_mla_module_native_heads(ctx, num_heads)?;
        let b = ctx.row.u32(r.col("batch_size")?)?;
        let s = ctx.row.u32(r.col("isl")?)?;
        let latency = ctx.row.f64(r.col("latency")?)?;
        let power = first_power.power(ctx)?;
        let fmha = ctx.row.str_owned(r.col("mla_dtype")?)?;
        let kv_dtype = ctx.row.str_owned(r.col("kv_cache_dtype")?)?;
        let gemm = ctx.row.str_owned(r.col("gemm_type")?)?;
        let path = [
            fmha,
            kv_dtype,
            gemm,
            native.to_string(),
            num_heads.to_string(),
            s.to_string(),
        ];
        root.insert_first_wins(&path, &b.to_string(), classic_leaf(latency, power));
        Ok(())
    })?;
    Ok(found.map(|_| root))
}

/// `operations/mla.py::load_generation_mla_module_data` —
/// `[kv_cache_dtype][gemm_type][native][num_heads][b][isl+step]`
/// (mla_dtype deliberately dropped, mirroring the per-op decode loader).
pub fn view_generation_mla_module(sources: &[PerfSource]) -> Result<Option<ViewNode>, AicError> {
    let mut root = ViewNode::branch();
    let mut first_power = FirstFilePower::new();
    let found = fold_sources(sources, |ctx| {
        let r = ctx.reader;
        let num_heads = ctx.row.u32(r.col("num_heads")?)?;
        let native = view_mla_module_native_heads(ctx, num_heads)?;
        let b = ctx.row.u32(r.col("batch_size")?)?;
        let s = ctx.row.u32(r.col("isl")?)? + ctx.row.u32(r.col("step")?)?;
        let latency = ctx.row.f64(r.col("latency")?)?;
        let power = first_power.power(ctx)?;
        let kv_dtype = ctx.row.str_owned(r.col("kv_cache_dtype")?)?;
        let gemm = ctx.row.str_owned(r.col("gemm_type")?)?;
        let path = [
            kv_dtype,
            gemm,
            native.to_string(),
            num_heads.to_string(),
            b.to_string(),
        ];
        root.insert_first_wins(&path, &s.to_string(), classic_leaf(latency, power));
        Ok(())
    })?;
    Ok(found.map(|_| root))
}

/// Ensure a (possibly empty) branch exists at `path` — mirrors Python
/// defaultdict "vivification" that some loaders rely on (deliberately or,
/// in mamba2's generation case, as a preserved bug).
fn vivify_branch(root: &mut ViewNode, path: &[String]) {
    root.descend(path);
}

/// Tuple keys (mamba-family model keys) are encoded `a|b|c`; the Python
/// rehydration layer splits them back into int tuples.
fn tuple_key(parts: &[u32]) -> String {
    parts
        .iter()
        .map(|p| p.to_string())
        .collect::<Vec<_>>()
        .join("|")
}

/// `operations/mamba.py::load_mamba2_data` —
/// `[kernel_source][phase][model_key]` then `[b][s]` for context rows.
/// Generation rows reproduce the Python loader's preserved defect: the
/// conflict probe vivifies `[b]` on a defaultdict and therefore NEVER raises,
/// so the entry is never stored — each generation row leaves an EMPTY dict at
/// `[model_key][b]`. Mamba2 is deprecated (removed in PR-7); the view mirrors
/// the actual Python table bit for bit rather than silently fixing it.
pub fn view_mamba2(sources: &[PerfSource]) -> Result<Option<ViewNode>, AicError> {
    let mut root = ViewNode::branch();
    let found = fold_sources(sources, |ctx| {
        let r = ctx.reader;
        let kernel_source = str_cell_or_empty(ctx, "kernel_source")?;
        let phase = str_cell_or_empty(ctx, "phase")?;
        let b = ctx.row.u32(r.col("batch_size")?)?;
        let s = ctx.row.u32(r.col("seq_len")?)?;
        let model_key = tuple_key(&[
            ctx.row.u32(r.col("d_model")?)?,
            ctx.row.u32(r.col("d_state")?)?,
            ctx.row.u32(r.col("d_conv")?)?,
            ctx.row.u32(r.col("nheads")?)?,
            ctx.row.u32(r.col("head_dim")?)?,
            ctx.row.u32(r.col("n_groups")?)?,
            ctx.row.u32(r.col("chunk_size")?)?,
        ]);
        let latency = ctx.row.f64(r.col("latency")?)?;
        let leaf = classic_leaf(latency, ctx.power()?);
        if phase == "context" {
            let path = [kernel_source, phase, model_key, b.to_string()];
            root.insert_first_wins(&path, &s.to_string(), leaf);
        } else {
            vivify_branch(&mut root, &[kernel_source, phase, model_key, b.to_string()]);
        }
        Ok(())
    })?;
    Ok(found.map(|_| root))
}

/// Shared shape of `load_gdn_data` / `load_kda_data`:
/// `[kernel_source][phase][model_key]` with `[b][s]` for 2-D phases and
/// `[b] -> leaf` otherwise, explicit first-wins on both shapes.
fn view_gdn_like(
    sources: &[PerfSource],
    model_cols: &[&str],
    two_d_phases: &[&str],
    alias: bool,
) -> Result<Option<ViewNode>, AicError> {
    let mut root = ViewNode::branch();
    let found = fold_sources(sources, |ctx| {
        let r = ctx.reader;
        let raw_ks = str_cell_or_empty(ctx, "kernel_source")?;
        // The alias map is the query loader's (state_space.rs) — one home
        // for the sglang decode-kernel rename drift.
        let kernel_source = if alias {
            crate::perf_database::state_space::normalize_gdn_kernel_source(raw_ks)
        } else {
            raw_ks
        };
        let phase = str_cell_or_empty(ctx, "phase")?;
        let b = ctx.row.u32(r.col("batch_size")?)?;
        let s = ctx.row.u32(r.col("seq_len")?)?;
        let mut key_parts = Vec::with_capacity(model_cols.len());
        for col in model_cols {
            key_parts.push(ctx.row.u32(r.col(col)?)?);
        }
        let model_key = tuple_key(&key_parts);
        let latency = ctx.row.f64(r.col("latency")?)?;
        let leaf = classic_leaf(latency, ctx.power()?);
        if two_d_phases.contains(&phase.as_str()) {
            let path = [kernel_source, phase, model_key, b.to_string()];
            root.insert_first_wins(&path, &s.to_string(), leaf);
        } else {
            let path = [kernel_source, phase, model_key];
            root.insert_first_wins(&path, &b.to_string(), leaf);
        }
        Ok(())
    })?;
    Ok(found.map(|_| root))
}

/// `operations/mamba.py::load_gdn_data` — model key
/// `(d_model, num_k_heads, head_k_dim, num_v_heads, head_v_dim, d_conv)`;
/// only "context" is 2-D; decode kernel names alias-normalized.
pub fn view_gdn(sources: &[PerfSource]) -> Result<Option<ViewNode>, AicError> {
    view_gdn_like(
        sources,
        &[
            "d_model",
            "num_k_heads",
            "head_k_dim",
            "num_v_heads",
            "head_v_dim",
            "d_conv",
        ],
        &["context"],
        true,
    )
}

/// `operations/mamba.py::load_kda_data` — same columns as GDN, but "verify"
/// rows are 2-D too (seq_len carries the draft-token count), no aliasing.
pub fn view_kda(sources: &[PerfSource]) -> Result<Option<ViewNode>, AicError> {
    view_gdn_like(
        sources,
        &[
            "d_model",
            "num_k_heads",
            "head_k_dim",
            "num_v_heads",
            "head_v_dim",
            "d_conv",
        ],
        &["context", "verify"],
        false,
    )
}

/// `operations/moe.py::load_moe_data` — 9-level
/// `[moe_dtype][distribution][topk][num_experts][hidden][inter][moe_tp][moe_ep][num_tokens]`,
/// with `kernel_source == "moe_torch_flow_min_latency"` rows routed to the
/// low-latency twin table. Returns (default, low_latency).
pub fn view_moe(sources: &[PerfSource]) -> Result<Option<(ViewNode, ViewNode)>, AicError> {
    let mut default = ViewNode::branch();
    let mut low_latency = ViewNode::branch();
    let found = fold_sources(sources, |ctx| {
        let r = ctx.reader;
        let num_tokens = ctx.row.u32(r.col("num_tokens")?)?;
        let hidden = ctx.row.u32(r.col("hidden_size")?)?;
        let inter = ctx.row.u32(r.col("inter_size")?)?;
        let topk = ctx.row.u32(r.col("topk")?)?;
        let num_experts = ctx.row.u32(r.col("num_experts")?)?;
        let moe_tp = ctx.row.u32(r.col("moe_tp_size")?)?;
        let moe_ep = ctx.row.u32(r.col("moe_ep_size")?)?;
        let distribution = str_cell_or_empty(ctx, "distribution")?;
        let latency = ctx.row.f64(r.col("latency")?)?;
        let kernel_source = str_cell_or_empty(ctx, "kernel_source")?;
        // Kernel-routed quant remap: the query loader's rule (moe.rs).
        let quant = crate::perf_database::moe::moe_kernel_quant_rewrite(
            ctx.row.str_owned(r.col("moe_dtype")?)?,
            &kernel_source,
        );
        let target = if kernel_source == "moe_torch_flow_min_latency" {
            &mut low_latency
        } else {
            &mut default
        };
        let path = [
            quant,
            distribution,
            topk.to_string(),
            num_experts.to_string(),
            hidden.to_string(),
            inter.to_string(),
            moe_tp.to_string(),
            moe_ep.to_string(),
        ];
        target.insert_first_wins(
            &path,
            &num_tokens.to_string(),
            classic_leaf(latency, ctx.power()?),
        );
        Ok(())
    })?;
    Ok(found.map(|_| (default, low_latency)))
}

/// `operations/moe.py::load_wideep_context_moe_data` /
/// `load_wideep_generation_moe_data` — same 9 levels as `load_moe_data`,
/// no kernel routing, no quant rewrites.
pub fn view_wideep_moe(sources: &[PerfSource]) -> Result<Option<ViewNode>, AicError> {
    let mut root = ViewNode::branch();
    let found = fold_sources(sources, |ctx| {
        let r = ctx.reader;
        let quant = ctx.row.str_owned(r.col("moe_dtype")?)?;
        let num_tokens = ctx.row.u32(r.col("num_tokens")?)?;
        let hidden = ctx.row.u32(r.col("hidden_size")?)?;
        let inter = ctx.row.u32(r.col("inter_size")?)?;
        let topk = ctx.row.u32(r.col("topk")?)?;
        let num_experts = ctx.row.u32(r.col("num_experts")?)?;
        let moe_tp = ctx.row.u32(r.col("moe_tp_size")?)?;
        let moe_ep = ctx.row.u32(r.col("moe_ep_size")?)?;
        let distribution = str_cell_or_empty(ctx, "distribution")?;
        let latency = ctx.row.f64(r.col("latency")?)?;
        let path = [
            quant,
            distribution,
            topk.to_string(),
            num_experts.to_string(),
            hidden.to_string(),
            inter.to_string(),
            moe_tp.to_string(),
            moe_ep.to_string(),
        ];
        root.insert_first_wins(
            &path,
            &num_tokens.to_string(),
            classic_leaf(latency, ctx.power()?),
        );
        Ok(())
    })?;
    Ok(found.map(|_| root))
}

/// `operations/moe.py::load_wideep_deepep_ll_data` —
/// `[node_num][hidden_size][num_topk][num_experts][num_token]`; latency is
/// the µs sum `combine_avg_t_us + dispatch_avg_t_us` stored UNconverted
/// (the Python loader never rescaled it).
pub fn view_wideep_deepep_ll(sources: &[PerfSource]) -> Result<Option<ViewNode>, AicError> {
    let mut root = ViewNode::branch();
    let found = fold_sources(sources, |ctx| {
        let r = ctx.reader;
        let hidden = ctx.row.u32(r.col("hidden_size")?)?;
        let node_num = ctx.row.u32(r.col("node_num")?)?;
        let num_token = ctx.row.u32(r.col("num_token")?)?;
        let num_topk = ctx.row.u32(r.col("num_topk")?)?;
        let num_experts = ctx.row.u32(r.col("num_experts")?)?;
        let lat =
            ctx.row.f64(r.col("combine_avg_t_us")?)? + ctx.row.f64(r.col("dispatch_avg_t_us")?)?;
        let path = [
            node_num.to_string(),
            hidden.to_string(),
            num_topk.to_string(),
            num_experts.to_string(),
        ];
        root.insert_first_wins(
            &path,
            &num_token.to_string(),
            classic_leaf(lat, ctx.power()?),
        );
        Ok(())
    })?;
    Ok(found.map(|_| root))
}

/// `operations/moe.py::load_wideep_deepep_normal_data` —
/// `[node_num][hidden_size][topk][num_experts][dispatch_sms][num_token]`;
/// latency = sum of the four transmit/notify µs components, unconverted.
pub fn view_wideep_deepep_normal(sources: &[PerfSource]) -> Result<Option<ViewNode>, AicError> {
    let mut root = ViewNode::branch();
    let found = fold_sources(sources, |ctx| {
        let r = ctx.reader;
        let num_token = ctx.row.u32(r.col("num_token")?)?;
        let topk = ctx.row.u32(r.col("num_topk")?)?;
        let node_num = ctx.row.u32(r.col("node_num")?)?;
        let num_experts = ctx.row.u32(r.col("num_experts")?)?;
        let hidden = ctx.row.u32(r.col("hidden_size")?)?;
        let dispatch_sms = ctx.row.u32(r.col("dispatch_sms")?)?;
        let lat = ctx.row.f64(r.col("dispatch_transmit_us")?)?
            + ctx.row.f64(r.col("dispatch_notify_us")?)?
            + ctx.row.f64(r.col("combine_transmit_us")?)?
            + ctx.row.f64(r.col("combine_notify_us")?)?;
        let path = [
            node_num.to_string(),
            hidden.to_string(),
            topk.to_string(),
            num_experts.to_string(),
            dispatch_sms.to_string(),
        ];
        root.insert_first_wins(
            &path,
            &num_token.to_string(),
            classic_leaf(lat, ctx.power()?),
        );
        Ok(())
    })?;
    Ok(found.map(|_| root))
}

/// `operations/moe.py::load_wideep_moe_compute_data` — 11-level
/// `[kernel_source][moe_dtype][distribution][topk][num_experts][hidden][inter][num_slots][moe_tp][moe_ep][num_tokens]`;
/// a row without a kernel_source COLUMN defaults to "moe_torch_flow".
pub fn view_wideep_moe_compute(sources: &[PerfSource]) -> Result<Option<ViewNode>, AicError> {
    let mut root = ViewNode::branch();
    let found = fold_sources(sources, |ctx| {
        let r = ctx.reader;
        let quant = ctx.row.str_owned(r.col("moe_dtype")?)?;
        let num_tokens = ctx.row.u32(r.col("num_tokens")?)?;
        let hidden = ctx.row.u32(r.col("hidden_size")?)?;
        let inter = ctx.row.u32(r.col("inter_size")?)?;
        let topk = ctx.row.u32(r.col("topk")?)?;
        let num_experts = ctx.row.u32(r.col("num_experts")?)?;
        let num_slots = ctx.row.u32(r.col("num_slots")?)?;
        let moe_tp = ctx.row.u32(r.col("moe_tp_size")?)?;
        let moe_ep = ctx.row.u32(r.col("moe_ep_size")?)?;
        let distribution = str_cell_or_empty(ctx, "distribution")?;
        let latency = ctx.row.f64(r.col("latency")?)?;
        let kernel_source = str_col_or(
            ctx,
            "kernel_source",
            crate::perf_database::moe_expert_compute::LEGACY_TRTLLM_DEFAULT_KERNEL_SOURCE,
        )?;
        let path = [
            kernel_source,
            quant,
            distribution,
            topk.to_string(),
            num_experts.to_string(),
            hidden.to_string(),
            inter.to_string(),
            num_slots.to_string(),
            moe_tp.to_string(),
            moe_ep.to_string(),
        ];
        root.insert_first_wins(
            &path,
            &num_tokens.to_string(),
            classic_leaf(latency, ctx.power()?),
        );
        Ok(())
    })?;
    Ok(found.map(|_| root))
}

/// `operations/moe.py::load_trtllm_alltoall_data` — 9-level
/// `[kernel_source][op_name][moe_dtype][num_nodes][hidden][topk][num_experts][moe_ep][num_tokens]`.
/// `has_num_nodes` follows the FIRST concatenated row: when absent there,
/// every row (even from later files that carry the column) computes
/// `max(1, moe_ep // 4)`. kernel_source defaults per-file to "NVLinkTwoSided".
pub fn view_trtllm_alltoall(sources: &[PerfSource]) -> Result<Option<ViewNode>, AicError> {
    let mut root = ViewNode::branch();
    let mut first_has_num_nodes: Option<bool> = None;
    let found = fold_sources(sources, |ctx| {
        let r = ctx.reader;
        let num_nodes_col = r.col_optional("num_nodes");
        let has_num_nodes = *first_has_num_nodes.get_or_insert(num_nodes_col.is_some());
        let op_name = str_cell_or_empty(ctx, "op_name")?;
        let quant = ctx.row.str_owned(r.col("moe_dtype")?)?;
        let num_tokens = ctx.row.u32(r.col("num_tokens")?)?;
        let hidden = ctx.row.u32(r.col("hidden_size")?)?;
        let topk = ctx.row.u32(r.col("topk")?)?;
        let num_experts = ctx.row.u32(r.col("num_experts")?)?;
        let moe_ep = ctx.row.u32(r.col("moe_ep_size")?)?;
        let latency = ctx.row.f64(r.col("latency")?)?;
        let kernel_source = str_col_or(
            ctx,
            "kernel_source",
            crate::perf_database::moe_a2a::LEGACY_TRTLLM_DEFAULT_KERNEL_SOURCE,
        )?;
        let num_nodes = if has_num_nodes {
            ctx.row.u32(r.col("num_nodes")?)?
        } else {
            crate::perf_database::trtllm_alltoall::legacy_num_nodes_fallback(moe_ep)
        };
        let path = [
            kernel_source,
            op_name,
            quant,
            num_nodes.to_string(),
            hidden.to_string(),
            topk.to_string(),
            num_experts.to_string(),
            moe_ep.to_string(),
        ];
        root.insert_first_wins(
            &path,
            &num_tokens.to_string(),
            classic_leaf(latency, ctx.power()?),
        );
        Ok(())
    })?;
    Ok(found.map(|_| root))
}

/// Numeric cell as f64 regardless of the parquet storage type (DOUBLE or
/// INT64) — Python read both through `int(float(x))` / `float(x)` without
/// caring, and without narrowing: negative and NaN cells pass through
/// exactly like `float()` returned them. None when the column is absent or
/// the cell is null.
fn num_optional(row: &PerfRow, col: Option<usize>) -> Result<Option<f64>, AicError> {
    if let Some(v) = row.f64_optional(col)? {
        return Ok(Some(v));
    }
    Ok(row.i64_optional(col)?.map(|v| v as f64))
}

/// Python's `int(row[col])` INSIDE a try/except-continue: `Some` when the
/// cell parses (a DOUBLE cell truncates toward zero like `int(float)`;
/// negative values stay negative — Python kept them as negative keys),
/// `None` when the retired loader skipped the row (null cell, NaN/inf,
/// non-numeric storage — `int()` raised into the except).
fn py_int_optional(row: &PerfRow, col: usize) -> Result<Option<i64>, AicError> {
    if let Some(v) = row.i64_optional(Some(col))? {
        return Ok(Some(v));
    }
    Ok(row
        .f64_optional(Some(col))?
        .and_then(|v| v.is_finite().then_some(v as i64)))
}

/// Python's fail-loud `int(row[col])` OUTSIDE any try/except: a null cell or
/// a non-finite value raised ValueError and failed the whole load; a finite
/// DOUBLE cell loaded its truncated value.
fn int_cell_loud(ctx: &RowCtx<'_>, col: usize, what: &str) -> Result<i64, AicError> {
    match py_int_optional(ctx.row, col)? {
        Some(v) => Ok(v),
        None => Err(AicError::PerfDatabase(format!(
            "unparseable {what} cell (null or non-finite) in {}: the retired loader's int() \
             failed the whole load here rather than skipping or defaulting",
            ctx.path.display()
        ))),
    }
}

/// Python's `int(row.get(name) or default)`: an absent column, a null cell
/// and a falsy ZERO cell all take the default; a finite DOUBLE truncates;
/// NaN/inf made that int() raise and failed the load loudly.
fn int_cell_or_falsy_default(ctx: &RowCtx<'_>, name: &str, default: i64) -> Result<i64, AicError> {
    match num_optional(ctx.row, ctx.reader.col_optional(name))? {
        None => Ok(default),
        Some(v) if !v.is_finite() => Err(AicError::PerfDatabase(format!(
            "unparseable {name} cell (non-finite) in {}: the retired loader's int() failed \
             the whole load here rather than defaulting",
            ctx.path.display()
        ))),
        Some(v) if v == 0.0 => Ok(default),
        Some(v) => Ok(v as i64),
    }
}

/// Mandatory string column with Python's null-as-"" read: `_read_perf_rows`
/// mapped a present-but-null cell to "" and the retired loaders KEPT the row
/// keyed under "" for plain-string key layers (enum-decoded layers crashed on
/// "" either way and stay fail-loud via `str_owned`). A missing COLUMN still
/// errors, like Python's KeyError.
fn str_cell_or_empty(ctx: &RowCtx<'_>, name: &str) -> Result<String, AicError> {
    let col = ctx.reader.col(name)?;
    Ok(ctx.row.str_optional(Some(col))?.unwrap_or("").to_string())
}

/// `moe_comm.py::_row_power` — null/NaN/absent power -> 0.0; a measured but
/// non-finite value is corrupt data. Reads through `num_optional` because
/// Python's `float(raw)` was storage-agnostic: an integer-typed watts column
/// (merged/legacy files) loaded its value rather than zeroing the family.
fn row_power_lenient(ctx: &RowCtx<'_>) -> Result<f64, AicError> {
    match num_optional(ctx.row, ctx.power_col)? {
        None => Ok(0.0),
        Some(v) if v.is_nan() => Ok(0.0),
        Some(v) if !v.is_finite() => Err(AicError::PerfDatabase(
            "non-finite power cell in perf data: power must be finite when measured".to_string(),
        )),
        Some(v) => Ok(v),
    }
}

/// `moe_comm.py::_require_latency` — latency is schema-required and finite.
/// Storage-agnostic like Python's `float(raw)`: an INT64 latency cell is a
/// value, not a "null latency" corruption report.
fn require_latency(ctx: &RowCtx<'_>, table: &str) -> Result<f64, AicError> {
    let lat = num_optional(ctx.row, ctx.reader.col_optional("latency"))?.ok_or_else(|| {
        AicError::PerfDatabase(format!(
            "null latency cell in a {table} row: latency is schema-required and must be \
                 finite; refusing to load corrupt perf data"
        ))
    })?;
    if !lat.is_finite() {
        return Err(AicError::PerfDatabase(format!(
            "non-finite latency cell in a {table} row: latency is schema-required and must be \
             finite; refusing to load corrupt perf data"
        )));
    }
    Ok(lat)
}

/// `operations/moe_comm.py::load_moe_a2a_data` with global source priority
/// across the four physical table formats. Each priority tier loads legacy
/// adapters first and lets the first new-schema occurrence override a legacy
/// leaf in that same tier. Completed lower-priority tiers fill missing keys
/// only. New-schema latency is µs -> ms.
pub fn view_moe_a2a(
    sources: &[PrioritizedSource],
    legacy_normal: &[PrioritizedSource],
    legacy_ll: &[PrioritizedSource],
    legacy_trtllm_alltoall: &[PrioritizedSource],
) -> Result<Option<ViewNode>, AicError> {
    let mut root = ViewNode::branch();
    let mut any_loaded = false;
    for tier in crate::perf_database::moe_a2a::source_tiers(
        sources,
        legacy_normal,
        legacy_ll,
        legacy_trtllm_alltoall,
    ) {
        if let Some(tier_root) = view_moe_a2a_tier(
            &tier.moe_a2a,
            &tier.legacy_normal,
            &tier.legacy_ll,
            &tier.legacy_trtllm_alltoall,
        )? {
            root.merge_fill(tier_root);
            any_loaded = true;
        }
    }
    Ok(any_loaded.then_some(root))
}

/// Fold one completed resolver priority tier into the unified 10-level a2a
/// store `[comm_backend][phase][comm_dtype][ep_size][node_num][hidden][topk]
/// [num_experts][sms][num_tokens]`.
fn view_moe_a2a_tier(
    sources: &[PerfSource],
    legacy_normal: &[PerfSource],
    legacy_ll: &[PerfSource],
    legacy_trtllm_alltoall: &[PerfSource],
) -> Result<Option<ViewNode>, AicError> {
    let mut root = ViewNode::branch();
    let mut legacy_loaded = false;

    // _adapt_legacy_deepep (normal -> deepep_ht with the 4-way µs split; ll ->
    // deepep_ll with per-phase averages). ep_size = node_num * 8, dtype "default".
    for (legacy_sources, comm_backend, phase_columns) in [
        (
            legacy_normal,
            "deepep_ht",
            &[
                (
                    "dispatch",
                    &["dispatch_transmit_us", "dispatch_notify_us"][..],
                ),
                ("combine", &["combine_transmit_us", "combine_notify_us"][..]),
            ][..],
        ),
        (
            legacy_ll,
            "deepep_ll",
            &[
                ("dispatch", &["dispatch_avg_t_us"][..]),
                ("combine", &["combine_avg_t_us"][..]),
            ][..],
        ),
    ] {
        let found = fold_sources(legacy_sources, |ctx| {
            let r = ctx.reader;
            let node_num = ctx.row.u32(r.col("node_num")?)?;
            let sms = if comm_backend == "deepep_ht" {
                ctx.row.u32(r.col("dispatch_sms")?)?
            } else {
                0
            };
            let power = row_power_lenient(ctx)?;
            for (phase, columns) in phase_columns {
                let mut latency_us = 0.0;
                for column in *columns {
                    latency_us += ctx.row.f64(r.col(column)?)?;
                }
                let latency = latency_us / 1000.0;
                let path = [
                    comm_backend.to_string(),
                    phase.to_string(),
                    crate::perf_database::moe_a2a::LEGACY_DEEPEP_DTYPE.to_string(),
                    crate::perf_database::moe_a2a::legacy_deepep_ep_size(node_num).to_string(),
                    node_num.to_string(),
                    ctx.row.u32(r.col("hidden_size")?)?.to_string(),
                    ctx.row.u32(r.col("num_topk")?)?.to_string(),
                    ctx.row.u32(r.col("num_experts")?)?.to_string(),
                    sms.to_string(),
                ];
                let leaf = classic_leaf(latency, power);
                root.insert_first_wins(&path, &ctx.row.u32(r.col("num_token")?)?.to_string(), leaf);
            }
            Ok(())
        })?;
        legacy_loaded |= found.is_some();
    }

    // _adapt_legacy_trtllm_alltoall: latency already ms; per-row num_nodes
    // fallback max(1, ep//4); unmapped kernel/op rows are skipped. The
    // kernel->backend and op->phase/dtype maps are the query adapter's
    // (moe_a2a.rs) — one home for the legacy-adaptation vocabulary.
    let found = fold_sources(legacy_trtllm_alltoall, |ctx| {
        let r = ctx.reader;
        let kernel_source = str_col_or(
            ctx,
            "kernel_source",
            crate::perf_database::moe_a2a::LEGACY_TRTLLM_DEFAULT_KERNEL_SOURCE,
        )?;
        let Some(comm_backend) =
            crate::perf_database::moe_a2a::legacy_trtllm_backend(&kernel_source)
        else {
            return Ok(());
        };
        // Null op_name reads as "" (unmapped) and skips the row, like the
        // Python adapter's map .get('') miss.
        let op_name = str_cell_or_empty(ctx, "op_name")?;
        let Some((phase, fixed_dtype)) =
            crate::perf_database::moe_a2a::legacy_trtllm_phase_dtype(&op_name)
        else {
            return Ok(());
        };
        let comm_dtype = match fixed_dtype {
            Some(d) => d.to_string(),
            // Pass-through comm_dtype is a plain-string key: null reads ""
            // and the row stays, like Python's row["moe_dtype"] after
            // _read_perf_rows' null-as-"" mapping.
            None => str_cell_or_empty(ctx, "moe_dtype")?,
        };
        let ep_size = ctx.row.u32(r.col("moe_ep_size")?)?;
        let node_num = match r.col_optional("num_nodes") {
            Some(col) => ctx.row.u32(col)?,
            None => crate::perf_database::trtllm_alltoall::legacy_num_nodes_fallback(ep_size),
        };
        let latency = ctx.row.f64(r.col("latency")?)?;
        let power = row_power_lenient(ctx)?;
        let path = [
            comm_backend.to_string(),
            phase.to_string(),
            comm_dtype,
            ep_size.to_string(),
            node_num.to_string(),
            ctx.row.u32(r.col("hidden_size")?)?.to_string(),
            ctx.row.u32(r.col("topk")?)?.to_string(),
            ctx.row.u32(r.col("num_experts")?)?.to_string(),
            "0".to_string(),
        ];
        let leaf = classic_leaf(latency, power);
        root.insert_first_wins(&path, &ctx.row.u32(r.col("num_tokens")?)?.to_string(), leaf);
        Ok(())
    })?;
    legacy_loaded |= found.is_some();

    // New schema: µs -> ms, sms normalized, first occurrence overwrites legacy.
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let new_found = fold_sources(sources, |ctx| {
        let r = ctx.reader;
        // The sms normalization is the query loader's (moe_a2a.rs) —
        // int-first with the is_finite guard.
        let sms = crate::perf_database::moe_a2a::normalize_sms(ctx.row, r.col_optional("sms"))?;
        let path = [
            str_cell_or_empty(ctx, "comm_backend")?,
            str_cell_or_empty(ctx, "phase")?,
            str_cell_or_empty(ctx, "comm_dtype")?,
            ctx.row.u32(r.col("ep_size")?)?.to_string(),
            ctx.row.u32(r.col("node_num")?)?.to_string(),
            ctx.row.u32(r.col("hidden_size")?)?.to_string(),
            ctx.row.u32(r.col("topk")?)?.to_string(),
            ctx.row.u32(r.col("num_experts")?)?.to_string(),
            sms.to_string(),
        ];
        let num_tokens = ctx.row.u32(r.col("num_tokens")?)?.to_string();
        let latency = require_latency(ctx, "moe_a2a_perf")? / 1000.0;
        let power = row_power_lenient(ctx)?;
        let leaf = classic_leaf(latency, power);
        let full_key = format!("{}\x1f{}", path.join("\x1f"), num_tokens);
        if seen.insert(full_key) {
            root.insert_overwrite(&path, &num_tokens, leaf);
        } else {
            root.insert_first_wins(&path, &num_tokens, leaf);
        }
        Ok(())
    })?;

    Ok(if new_found.is_some() || legacy_loaded {
        Some(root)
    } else {
        None
    })
}

/// `operations/moe_comm.py::load_moe_expert_compute_data` — the unified
/// 12-level EP compute store `[kernel_source][quant][distribution]
/// [inference_phase][topk][num_experts][num_slots][hidden][inter][moe_tp]
/// [moe_ep][num_tokens]`. Latency is already ms everywhere. Legacy sglang
/// wideep tables adapt to kernel_source "deepep_moe" with num_slots =
/// num_experts; legacy trtllm wideep rows register under BOTH phases.
pub fn view_moe_expert_compute(
    sources: &[PerfSource],
    legacy_context: &[PerfSource],
    legacy_generation: &[PerfSource],
    legacy_trtllm_wideep: &[PerfSource],
) -> Result<Option<ViewNode>, AicError> {
    let mut root = ViewNode::branch();
    let mut legacy_loaded = false;

    for (legacy_sources, inference_phase) in [
        (legacy_context, "context"),
        (legacy_generation, "generation"),
    ] {
        let found = fold_sources(legacy_sources, |ctx| {
            let r = ctx.reader;
            let latency = ctx.row.f64(r.col("latency")?)?;
            let power = row_power_lenient(ctx)?;
            let num_experts = ctx.row.u32(r.col("num_experts")?)?;
            let path = [
                crate::perf_database::moe_expert_compute::SGLANG_ADAPTED_KERNEL_SOURCE.to_string(),
                ctx.row.str_owned(r.col("moe_dtype")?)?,
                str_cell_or_empty(ctx, "distribution")?,
                inference_phase.to_string(),
                ctx.row.u32(r.col("topk")?)?.to_string(),
                num_experts.to_string(),
                num_experts.to_string(),
                ctx.row.u32(r.col("hidden_size")?)?.to_string(),
                ctx.row.u32(r.col("inter_size")?)?.to_string(),
                ctx.row.u32(r.col("moe_tp_size")?)?.to_string(),
                ctx.row.u32(r.col("moe_ep_size")?)?.to_string(),
            ];
            let leaf = classic_leaf(latency, power);
            root.insert_first_wins(&path, &ctx.row.u32(r.col("num_tokens")?)?.to_string(), leaf);
            Ok(())
        })?;
        legacy_loaded |= found.is_some();
    }

    let found = fold_sources(legacy_trtllm_wideep, |ctx| {
        let r = ctx.reader;
        let latency = ctx.row.f64(r.col("latency")?)?;
        let power = row_power_lenient(ctx)?;
        let kernel_source = str_col_or(
            ctx,
            "kernel_source",
            crate::perf_database::moe_expert_compute::LEGACY_TRTLLM_DEFAULT_KERNEL_SOURCE,
        )?;
        for inference_phase in ["context", "generation"] {
            let path = [
                kernel_source.clone(),
                ctx.row.str_owned(r.col("moe_dtype")?)?,
                str_cell_or_empty(ctx, "distribution")?,
                inference_phase.to_string(),
                ctx.row.u32(r.col("topk")?)?.to_string(),
                ctx.row.u32(r.col("num_experts")?)?.to_string(),
                ctx.row.u32(r.col("num_slots")?)?.to_string(),
                ctx.row.u32(r.col("hidden_size")?)?.to_string(),
                ctx.row.u32(r.col("inter_size")?)?.to_string(),
                ctx.row.u32(r.col("moe_tp_size")?)?.to_string(),
                ctx.row.u32(r.col("moe_ep_size")?)?.to_string(),
            ];
            let leaf = classic_leaf(latency, power);
            root.insert_first_wins(&path, &ctx.row.u32(r.col("num_tokens")?)?.to_string(), leaf);
        }
        Ok(())
    })?;
    legacy_loaded |= found.is_some();

    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let new_found = fold_sources(sources, |ctx| {
        let r = ctx.reader;
        let path = [
            str_cell_or_empty(ctx, "kernel_source")?,
            ctx.row.str_owned(r.col("moe_dtype")?)?,
            str_cell_or_empty(ctx, "distribution")?,
            str_cell_or_empty(ctx, "inference_phase")?,
            ctx.row.u32(r.col("topk")?)?.to_string(),
            ctx.row.u32(r.col("num_experts")?)?.to_string(),
            ctx.row.u32(r.col("num_slots")?)?.to_string(),
            ctx.row.u32(r.col("hidden_size")?)?.to_string(),
            ctx.row.u32(r.col("inter_size")?)?.to_string(),
            ctx.row.u32(r.col("moe_tp_size")?)?.to_string(),
            ctx.row.u32(r.col("moe_ep_size")?)?.to_string(),
        ];
        let num_tokens = ctx.row.u32(r.col("num_tokens")?)?.to_string();
        let latency = require_latency(ctx, "moe_expert_compute_perf")?;
        let power = row_power_lenient(ctx)?;
        let leaf = classic_leaf(latency, power);
        let full_key = format!("{}\x1f{}", path.join("\x1f"), num_tokens);
        if seen.insert(full_key) {
            root.insert_overwrite(&path, &num_tokens, leaf);
        } else {
            root.insert_first_wins(&path, &num_tokens, leaf);
        }
        Ok(())
    })?;

    Ok(if new_found.is_some() || legacy_loaded {
        Some(root)
    } else {
        None
    })
}

/// Per-source accumulator reproducing the DSA loaders' two-rule merge:
/// last-row-wins WITHIN one source (dict assignment keeps first insertion
/// position), first-source-wins ACROSS sources.
struct DsaSourceValues {
    entries: Vec<(Vec<String>, ViewNode)>,
    index: std::collections::HashMap<String, usize>,
}

impl DsaSourceValues {
    fn new() -> Self {
        DsaSourceValues {
            entries: Vec::new(),
            index: std::collections::HashMap::new(),
        }
    }
    fn put(&mut self, coordinate: Vec<String>, leaf: ViewNode) {
        let key = coordinate.join("\x1f");
        match self.index.get(&key) {
            Some(&i) => self.entries[i].1 = leaf,
            None => {
                self.index.insert(key, self.entries.len());
                self.entries.push((coordinate, leaf));
            }
        }
    }
}

/// Shared body of the two DSA module loaders. `context` selects the 9-level
/// context layering vs the 7-level generation one; `skip` selects the
/// skip_indexer rows (shared file, split by op_name). `has_power` follows the
/// first row of the first existing source, like Python's `rows[0]` probe.
fn view_dsa_module(
    sources: &[PerfSource],
    context: bool,
    skip: bool,
) -> Result<Option<ViewNode>, AicError> {
    let mut root = ViewNode::branch();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut first_has_power: Option<bool> = None;

    let found = fold_sources_grouped(
        sources,
        DsaSourceValues::new,
        |ctx, source_values| {
            let r = ctx.reader;
            let row = ctx.row;
            let has_power = *first_has_power.get_or_insert(ctx.power_col.is_some());
            let op_name = row.str_optional(r.col_optional("op_name"))?.unwrap_or("");
            if op_name.contains("skip_indexer") != skip {
                return Ok(());
            }
            let num_heads = row.u32(r.col("num_heads")?)?;
            let b = row.u32(r.col("batch_size")?)?;
            let latency = row.f64(r.col("latency")?)?;
            let power = if has_power {
                // Fail-loud on a present-but-null cell, like the classic
                // loaders; a LATER source file without the column still
                // defaults per-row (Python's row.get).
                match ctx.power_col {
                    Some(col) => row.f64(col)?,
                    None => 0.0,
                }
            } else {
                0.0
            };
            let arch = match r.col_optional("architecture") {
                Some(col) => row.str_optional(Some(col))?.unwrap_or("").to_string(),
                None => "DeepseekV32ForCausalLM".to_string(),
            };
            let gemm = row.str_owned(r.col("gemm_type")?)?;
            let kv_dtype = row.str_owned(r.col("kv_cache_dtype")?)?;
            let ks = row.str_optional(ctx.ks_col)?.unwrap_or("").to_string();
            let leaf = classic_leaf(latency, power);
            if context {
                let s = row.u32(r.col("isl")?)?;
                let fmha = row.str_owned(r.col("mla_dtype")?)?;
                let step = num_optional(row, r.col_optional("step"))?;
                let step_missing = step.is_none();
                if arch == "GlmMoeDsaForCausalLM" && step_missing {
                    return Err(AicError::PerfDatabase(
                        "GLM-5 context DSA module data requires a non-empty step column for \
                         prefix/past_kv length"
                            .to_string(),
                    ));
                }
                // A stored NaN step must not silently file the row at
                // prefix=0 (Python's int(nan) raised and failed the load).
                if step.is_some_and(|v| !v.is_finite()) {
                    return Err(AicError::PerfDatabase(format!(
                        "non-finite step cell in a DSA context row at {}",
                        ctx.path.display()
                    )));
                }
                // int(step): negative values stayed negative prefix keys.
                let prefix = step.map(|v| v as i64).unwrap_or(0);
                for backend in crate::perf_database::dsa::dsa_kernel_source_buckets(&ks, &kv_dtype)
                {
                    source_values.put(
                        vec![
                            fmha.clone(),
                            kv_dtype.clone(),
                            gemm.clone(),
                            arch.clone(),
                            backend.to_string(),
                            num_heads.to_string(),
                            prefix.to_string(),
                            s.to_string(),
                            b.to_string(),
                        ],
                        leaf.clone(),
                    );
                }
            } else {
                let s = row.u32(r.col("isl")?)? + row.u32(r.col("step")?)?;
                for backend in crate::perf_database::dsa::dsa_kernel_source_buckets(&ks, &kv_dtype)
                {
                    source_values.put(
                        vec![
                            kv_dtype.clone(),
                            gemm.clone(),
                            arch.clone(),
                            backend.to_string(),
                            num_heads.to_string(),
                            b.to_string(),
                            s.to_string(),
                        ],
                        leaf.clone(),
                    );
                }
            }
            Ok(())
        },
        |source_values| {
            // Sources are priority-ordered: first source wins across files.
            for (coordinate, leaf) in source_values.entries {
                let key = coordinate.join("\x1f");
                if !seen.insert(key) {
                    continue;
                }
                let (path_part, last) = coordinate.split_at(coordinate.len() - 1);
                root.insert_first_wins(path_part, &last[0], leaf);
            }
            Ok(())
        },
    )?;

    Ok(found.map(|_| root))
}

/// `operations/dsa.py::load_context_dsa_module_data` —
/// `[mla_dtype][kv_cache_dtype][gemm_type][architecture][dsa_backend][num_heads][prefix][s][b]`.
pub fn view_context_dsa_module(
    sources: &[PerfSource],
    skip: bool,
) -> Result<Option<ViewNode>, AicError> {
    view_dsa_module(sources, true, skip)
}

/// `operations/dsa.py::load_generation_dsa_module_data` —
/// `[kv_cache_dtype][gemm_type][architecture][dsa_backend][num_heads][b][isl+step]`.
pub fn view_generation_dsa_module(
    sources: &[PerfSource],
    skip: bool,
) -> Result<Option<ViewNode>, AicError> {
    view_dsa_module(sources, false, skip)
}

/// `operations/dsv4.py::load_mhc_module_data` —
/// `[op_name][hc_mult][hidden_size][num_tokens]`; `has_power` follows the
/// first concatenated row.
pub fn view_mhc_module(sources: &[PerfSource]) -> Result<Option<ViewNode>, AicError> {
    let mut root = ViewNode::branch();
    let mut first_power = FirstFilePower::new();
    let found = fold_sources(sources, |ctx| {
        let r = ctx.reader;
        let op = str_cell_or_empty(ctx, "op_name")?;
        let hc_mult = ctx.row.u32(r.col("hc_mult")?)?;
        let hidden = ctx.row.u32(r.col("hidden_size")?)?;
        let num_tokens = ctx.row.u32(r.col("num_tokens")?)?;
        let latency = ctx.row.f64(r.col("latency")?)?;
        let power = first_power.power(ctx)?;
        let path = [op, hc_mult.to_string(), hidden.to_string()];
        root.insert_first_wins(&path, &num_tokens.to_string(), classic_leaf(latency, power));
        Ok(())
    })?;
    Ok(found.map(|_| root))
}

/// `operations/dsv4.py::_validate_dsv4_local_head_semantics` — reject stale
/// pre-#1131 NATIVE-heads files by the per-(model, version) fingerprint, and
/// require a parseable tp_size everywhere once the file carries the column.
fn validate_view_dsv4_head_semantics(
    rows: &[(u32, Option<u32>, String, String)],
    saw_tp_size: bool,
    missing_tp_rows: usize,
    path_label: &str,
) -> Result<(), AicError> {
    if saw_tp_size && missing_tp_rows > 0 {
        return Err(AicError::PerfDatabase(format!(
            "DSV4 module file {path_label} has {missing_tp_rows} row(s) without a parseable \
             tp_size; the #1429 convention requires tp_size in every row \
             (native = num_heads * tp_size)."
        )));
    }
    if !rows.is_empty() && !saw_tp_size {
        return Err(AicError::PerfDatabase(format!(
            "DSV4 module file {path_label} carries no parseable tp_size column; the #1429 \
             convention requires tp_size in every row (native = num_heads * tp_size)."
        )));
    }
    // The stale-fingerprint rule itself is the query loader's
    // (dsv4.rs::validate_dsv4_local_head_semantics) — one home for the #1429
    // pattern; only the tp-presence preconditions above are view-specific
    // (they carry the path label Python's loader printed).
    let mut observed: std::collections::BTreeMap<
        (String, String),
        std::collections::BTreeSet<(u32, u32)>,
    > = std::collections::BTreeMap::new();
    for (heads, tp, model, version) in rows {
        let tp = tp.map(|v| v.max(1)).unwrap_or(1);
        observed
            .entry((model.clone(), version.clone()))
            .or_default()
            .insert((*heads, tp));
    }
    crate::perf_database::dsv4::validate_dsv4_local_head_semantics(&observed)
}

/// Shared body of the two DSV4 attention-kind loaders. Malformed rows are
/// SKIPPED (the Python loaders' try/except-continue), matching the appended
/// duplicate-header tolerance.
fn view_dsv4_kind_module(
    sources: &[PerfSource],
    context: bool,
) -> Result<Option<ViewNode>, AicError> {
    // Two passes like Python: the semantics validator scans the full row set
    // first, then the fold runs. Collect the decoded fields once.
    struct Decoded {
        // Signed like Python's int(): negative key cells stayed negative keys.
        b: i64,
        s: i64,
        prefix: i64,
        cr: i64,
        latency: f64,
        heads: i64,
        tp: i64,
        power: f64,
        gemm: String,
        fmha: String,
        kv: String,
    }
    let mut decoded: Vec<Decoded> = Vec::new();
    let mut fingerprint: Vec<(u32, Option<u32>, String, String)> = Vec::new();
    let mut saw_tp_size = false;
    let mut missing_tp_rows = 0usize;
    let mut first_has_power: Option<bool> = None;
    let mut path_label = String::new();

    let found = fold_sources(sources, |ctx| {
        let r = ctx.reader;
        if path_label.is_empty() {
            path_label = ctx.path.display().to_string();
        }
        let has_power = *first_has_power.get_or_insert(ctx.power_col.is_some());
        // Fingerprint scan mirrors Python: heads parseable -> observe;
        // tp parseable -> saw_tp_size, else missing_tp_rows += 1. "Parseable"
        // is int(row[...]) succeeding: DOUBLE cells count like int(float(x)),
        // but null and NaN/inf cells raised into the except — a NaN tp_size
        // is MISSING (and via the preconditions fails the file loudly), never
        // a silent tp=1 that would disarm the #1429 validator.
        let py_cell = |name: &str| -> Result<Option<i64>, AicError> {
            match r.col_optional(name) {
                Some(col) => py_int_optional(ctx.row, col),
                None => Ok(None), // Python's KeyError branch
            }
        };
        let heads_opt = py_cell("num_heads")?;
        if let Some(heads) = heads_opt {
            let tp_opt = py_cell("tp_size")?;
            if tp_opt.is_some() {
                saw_tp_size = true;
            } else {
                missing_tp_rows += 1;
            }
            let model = ctx
                .row
                .str_optional(r.col_optional("model"))?
                .unwrap_or("")
                .to_string();
            let version = ctx
                .row
                .str_optional(r.col_optional("version"))?
                .unwrap_or("")
                .to_string();
            fingerprint.push((
                u32::try_from(heads).unwrap_or(0),
                tp_opt.map(|v| u32::try_from(v.max(1)).unwrap_or(u32::MAX)),
                model,
                version,
            ));
        }
        // Row decode with the retired loader's try/except-continue semantics:
        // DOUBLE cells load their truncated value, null/NaN cells skip the
        // ROW (never abort the view, never silently empty the table), and
        // negative values stay negative keys.
        let (Some(b), Some(s_raw), Some(cr), Some(heads)) = (
            py_cell("batch_size")?,
            py_cell("isl")?,
            py_cell("compress_ratio")?,
            heads_opt,
        ) else {
            return Ok(());
        };
        // float(row["latency"]) inside the try: a null cell skips the row, an
        // integer-typed column loads its value, a stored NaN stays NaN.
        let Some(latency) = num_optional(ctx.row, r.col_optional("latency"))? else {
            return Ok(());
        };
        let (s, prefix) = if context {
            // int(float(row.get("step", 0) or 0)): absent column and null
            // cell -> 0; a NaN/inf cell raised inside the try -> row skipped.
            let prefix = match num_optional(ctx.row, r.col_optional("step"))? {
                None => 0,
                Some(v) if !v.is_finite() => return Ok(()),
                Some(v) => v as i64,
            };
            (s_raw, prefix)
        } else {
            // int(row["isl"]) + int(row["step"]): missing/null/NaN step -> skip.
            let Some(step) = py_cell("step")? else {
                return Ok(());
            };
            (s_raw.saturating_add(step), 0)
        };
        // max(1, int(row.get("tp_size", 1) or 1)) inside the try: an absent
        // column and a null cell fall to 1 (falsy 0 via max), NaN/inf -> skip.
        let tp = match num_optional(ctx.row, r.col_optional("tp_size"))? {
            None => 1,
            Some(v) if !v.is_finite() => return Ok(()),
            Some(v) => (v as i64).max(1),
        };
        let power = if has_power {
            // Fail-loud on a present-but-null cell (see RowCtx::power); a
            // later source file without the column still defaults per-row.
            match ctx.power_col {
                Some(col) => ctx.row.f64(col)?,
                None => 0.0,
            }
        } else {
            0.0
        };
        // The dtype columns sat OUTSIDE the Python loader's
        // try/except-continue: a missing column or null cell crashed the
        // load loudly (KeyError / enum KeyError) rather than skipping rows.
        let gemm = ctx.row.str_owned(r.col("gemm_type")?)?;
        let kv = ctx.row.str_owned(r.col("kv_cache_dtype")?)?;
        let fmha = if context {
            crate::perf_database::dsv4::normalize_dsv4_dtype(ctx.row.str(r.col("mla_dtype")?)?)
        } else {
            String::new()
        };
        decoded.push(Decoded {
            b,
            s,
            prefix,
            cr,
            latency,
            heads,
            tp,
            power,
            gemm,
            fmha,
            kv: crate::perf_database::dsv4::normalize_dsv4_dtype(&kv),
        });
        Ok(())
    })?;
    if found.is_none() {
        return Ok(None);
    }
    validate_view_dsv4_head_semantics(&fingerprint, saw_tp_size, missing_tp_rows, &path_label)?;

    let mut root = ViewNode::branch();
    for row in decoded {
        let native = row.heads.saturating_mul(row.tp);
        let leaf = classic_leaf(row.latency, row.power);
        if context {
            let path = [
                row.fmha,
                row.kv,
                row.gemm,
                native.to_string(),
                row.heads.to_string(),
                row.cr.to_string(),
                row.prefix.to_string(),
                row.s.to_string(),
            ];
            root.insert_first_wins(&path, &row.b.to_string(), leaf);
        } else {
            let path = [
                row.kv,
                row.gemm,
                native.to_string(),
                row.heads.to_string(),
                row.cr.to_string(),
                row.b.to_string(),
            ];
            root.insert_first_wins(&path, &row.s.to_string(), leaf);
        }
    }
    Ok(Some(root))
}

/// `operations/dsv4.py::load_context_dsv4_kind_module_data` —
/// `[fmha][kv][gemm][native][local][compress_ratio][prefix][s][b]`.
pub fn view_context_dsv4_kind_module(sources: &[PerfSource]) -> Result<Option<ViewNode>, AicError> {
    view_dsv4_kind_module(sources, true)
}

/// `operations/dsv4.py::load_generation_dsv4_kind_module_data` —
/// `[kv][gemm][native][local][compress_ratio][b][isl+step]`.
pub fn view_generation_dsv4_kind_module(
    sources: &[PerfSource],
) -> Result<Option<ViewNode>, AicError> {
    view_dsv4_kind_module(sources, false)
}

/// `operations/dsv4.py::load_dsv4_megamoe_module_data` — the 15-level
/// wide-leaf MegaMoE table; single-source only, boolean invariants enforced,
/// DUPLICATE rows are a load error (not first-wins).
pub fn view_dsv4_megamoe_module(data_root: &std::path::Path) -> Result<Option<ViewNode>, AicError> {
    // Python passed the PRIMARY path only — resolve_op_data_path's
    // family-first-then-legacy walk, never the shared-layer source list. A
    // legacy INCOMPLETE veto (or an admitted donor) must not substitute a
    // sibling file here, and an empty source list must not re-resolve a
    // vetoed primary; resolving directly reproduces the retired loader
    // (find_in_family_dirs carries resolve_op_data_path's INCOMPLETE veto
    // and dot-dir skip; the legacy fallback below stays unvetoed, exactly
    // like resolve_op_data_path's final branch).
    let basename = "dsv4_megamoe_module_perf.parquet";
    let primary = crate::perf_database::find_in_family_dirs(data_root, basename)
        .unwrap_or_else(|| data_root.join(basename));
    let single = [PerfSource(primary, None)];
    let mut root = ViewNode::branch();
    let found = fold_sources(&single, |ctx| {
        let r = ctx.reader;
        let to_bool = |v: Option<&str>| -> bool {
            matches!(
                v.unwrap_or("").trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "y"
            )
        };
        let bool_col = |name: &str, default: Option<&str>| -> Result<bool, AicError> {
            match r.col_optional(name) {
                // `PerfRow::bool` mirrors Python's `_to_bool(str(value))`:
                // BOOLEAN, INT64 and string storages all coerce.
                // A present-but-null cell coerces to false, mirroring the
                // retired loader's _to_bool("") — PerfRow::bool would error.
                Some(col) => Ok(ctx.row.bool(col).unwrap_or(false)),
                None => Ok(to_bool(default)),
            }
        };
        if !bool_col("used_cuda_graph", None)? {
            return Err(AicError::PerfDatabase(format!(
                "DSv4 MegaMoE perf row was not collected with CUDA Graph: {}",
                ctx.path.display()
            )));
        }
        if bool_col("includes_gate_topk", Some("true"))? {
            return Err(AicError::PerfDatabase(format!(
                "DSv4 MegaMoE perf row includes gate/top-k outside the supported boundary: {}",
                ctx.path.display()
            )));
        }
        if !bool_col("includes_routed_scale", None)? {
            return Err(AicError::PerfDatabase(format!(
                "DSv4 MegaMoE perf row does not include SGLang routed output scaling: {}",
                ctx.path.display()
            )));
        }
        let kernel_source = str_col_or(ctx, "kernel_source", "deepgemm_megamoe")?;
        let kernel_dtype = str_cell_or_empty(ctx, "kernel_dtype")?;
        let quant = ctx.row.str_owned(r.col("moe_dtype")?)?;
        let pre_dispatch = str_cell_or_empty(ctx, "pre_dispatch")?;
        let source_policy = str_cell_or_empty(ctx, "source_policy")?;
        let distribution = str_cell_or_empty(ctx, "distribution")?;
        let topk = ctx.row.u32(r.col("topk")?)?;
        let num_experts = ctx.row.u32(r.col("num_experts")?)?;
        // int(row.get(name, DEFAULT)): the default applies only when the
        // COLUMN is absent — a null cell raised loudly, a DOUBLE cell loaded
        // its truncated value (never a silent 0/1).
        let num_fused = match r.col_optional("num_fused_shared_experts") {
            Some(col) => int_cell_loud(ctx, col, "num_fused_shared_experts")?,
            None => 0,
        };
        let hidden = ctx.row.u32(r.col("hidden_size")?)?;
        let inter = ctx.row.u32(r.col("inter_size")?)?;
        let moe_tp = match r.col_optional("moe_tp_size") {
            Some(col) => int_cell_loud(ctx, col, "moe_tp_size")?,
            None => 1,
        };
        let moe_ep = ctx.row.u32(r.col("moe_ep_size")?)?;
        let num_tokens = ctx.row.u32(r.col("num_tokens")?)?;
        let latency = ctx.row.f64(r.col("latency")?)?;
        // float(row.get("power") or 0.0): absent/null/0 -> 0.0, an integer
        // watts column loads its value, a NaN cell stays NaN.
        let power = num_optional(ctx.row, ctx.power_col)?.unwrap_or(0.0);
        let num_max = int_cell_or_falsy_default(ctx, "num_max_tokens_per_rank", 0)?;
        let effective_num_max =
            int_cell_or_falsy_default(ctx, "effective_num_max_tokens_per_rank", num_max)?;
        let global_tokens =
            int_cell_or_falsy_default(ctx, "global_num_tokens", num_tokens as i64 * moe_ep as i64)?;
        let phase = ctx
            .row
            .str_optional(r.col_optional("phase"))?
            .unwrap_or("")
            .trim()
            .to_string();
        if phase.is_empty() {
            return Err(AicError::PerfDatabase(format!(
                "DSv4 MegaMoE unified perf file requires a phase column: {}",
                ctx.path.display()
            )));
        }
        if phase != "context" && phase != "generation" {
            return Err(AicError::PerfDatabase(format!(
                "DSv4 MegaMoE perf row has unsupported phase={phase:?}"
            )));
        }
        let buffer_policy = ctx
            .row
            .str_optional(r.col_optional("buffer_policy"))?
            .unwrap_or("")
            .to_string();
        let includes_buffer_init = bool_col("includes_buffer_init", Some("false"))?;
        let routed_scaling_factor = ctx.row.f64(r.col("routed_scaling_factor")?)?;
        let leaf = ViewNode::Leaf(vec![
            ("latency", ViewValue::F64(latency)),
            ("power", ViewValue::F64(power)),
            ("energy", ViewValue::F64(power * latency)),
            ("global_num_tokens", ViewValue::I64(global_tokens)),
            ("num_max_tokens_per_rank", ViewValue::I64(num_max)),
            (
                "effective_num_max_tokens_per_rank",
                ViewValue::I64(effective_num_max),
            ),
            ("used_cuda_graph", ViewValue::Bool(true)),
            ("kernel_dtype", ViewValue::Str(kernel_dtype.clone())),
            (
                "routed_scaling_factor",
                ViewValue::F64(routed_scaling_factor),
            ),
            ("includes_routed_scale", ViewValue::Bool(true)),
            ("includes_gate_topk", ViewValue::Bool(false)),
            ("buffer_policy", ViewValue::Str(buffer_policy)),
            (
                "includes_buffer_init",
                ViewValue::Bool(includes_buffer_init),
            ),
            ("phase", ViewValue::Str(phase.clone())),
        ]);
        let path = [
            phase,
            kernel_source,
            kernel_dtype,
            quant,
            pre_dispatch,
            source_policy,
            distribution,
            topk.to_string(),
            num_experts.to_string(),
            num_fused.to_string(),
            hidden.to_string(),
            inter.to_string(),
            moe_tp.to_string(),
            moe_ep.to_string(),
        ];
        if !root.insert_first_wins(&path, &num_tokens.to_string(), leaf) {
            return Err(AicError::PerfDatabase(format!(
                "duplicate DSv4 MegaMoE data row for {}",
                ctx.path.display()
            )));
        }
        Ok(())
    })?;
    Ok(found.map(|_| root))
}

/// `operations/dsv4.py::load_dsv4_sparse_kernel_data` — the shared sparse-op
/// schema nested under `(num_heads, tp_size, step, isl, batch_size)`, leaf
/// `{"latency": ms}` only. Malformed / blank-key rows are skipped.
pub fn view_dsv4_sparse_kernel(sources: &[PerfSource]) -> Result<Option<ViewNode>, AicError> {
    const KEY_COLS: [&str; 5] = ["num_heads", "tp_size", "step", "isl", "batch_size"];
    let mut root = ViewNode::branch();
    let mut any_leaf = false;
    let found = fold_sources(sources, |ctx| {
        let r = ctx.reader;
        let mut keys: Vec<String> = Vec::with_capacity(KEY_COLS.len());
        for col in KEY_COLS {
            // Python: _coerce = int(float(row[col])) with KeyError -> skip
            // row; null and NaN/inf cells are "bad keys" that also skip the
            // ROW (never abort the whole view), and a negative cell stayed a
            // negative key (a collector-bug sentinel like step=-1 dropped
            // only itself, filed under -1, never the whole family). This
            // subsumes the loader's duplicate-header/blank batch_size guard.
            let Some(idx) = r.col_optional(col) else {
                return Ok(());
            };
            let Some(value) = py_int_optional(ctx.row, idx)? else {
                return Ok(());
            };
            keys.push(value.to_string());
        }
        // float(row["latency"]): null -> skip row; integer-typed -> value.
        let Some(latency) = num_optional(ctx.row, r.col_optional("latency"))? else {
            return Ok(());
        };
        let leaf = ViewNode::Leaf(vec![("latency", ViewValue::F64(latency))]);
        let (path, last) = keys.split_at(keys.len() - 1);
        root.insert_first_wins(path, &last[0], leaf);
        any_leaf = true;
        Ok(())
    })?;
    // Python returns `root or None`: an existing file with zero usable rows
    // yields None, unlike the classic loaders.
    Ok(match found {
        Some(()) if any_leaf => Some(root),
        _ => None,
    })
}

/// The retired `operations/dsv4.py::load_dsv4_sparse_op_data` under
/// `_TOPK_CALIB_KEYS` — the CSA topk DELTA calibration table nested under
/// `(num_heads, step, isl, batch_size, score_mode)`, leaf `{"latency": ms}`.
/// The first four keys are Python-`int`-coerced; `score_mode` keeps its
/// string form (`v1_top_last` / `v1_flat` / ...). Blank / NaN-sentinel key
/// cells skip the ROW (`_is_bad_key`), duplicate keys keep the FIRST source
/// (shared-layer contract), and an existing file with zero usable rows
/// yields `None` (`root or None`).
pub fn view_dsv4_csa_topk_calib(sources: &[PerfSource]) -> Result<Option<ViewNode>, AicError> {
    const INT_KEY_COLS: [&str; 4] = ["num_heads", "step", "isl", "batch_size"];
    let mut root = ViewNode::branch();
    let mut any_leaf = false;
    let found = fold_sources(sources, |ctx| {
        let r = ctx.reader;
        let mut keys: Vec<String> = Vec::with_capacity(5);
        for col in INT_KEY_COLS {
            // `_coerce` = int(float(cell)); a null / NaN / non-numeric cell
            // is a bad key that skips the row (this subsumes the loader's
            // duplicate-header guard on the batch_size cell).
            let Some(idx) = r.col_optional(col) else {
                return Ok(());
            };
            let Some(value) = py_int_optional(ctx.row, idx)? else {
                return Ok(());
            };
            keys.push(value.to_string());
        }
        // score_mode stays a string key; `_is_bad_key` skipped blank and
        // NaN/inf SENTINEL strings.
        let Some(mode_col) = r.col_optional("score_mode") else {
            return Ok(());
        };
        let Some(mode) = ctx.row.str_optional(Some(mode_col))? else {
            return Ok(());
        };
        let trimmed = mode.trim();
        if trimmed.is_empty()
            || matches!(
                trimmed.to_ascii_lowercase().as_str(),
                "nan" | "inf" | "-inf" | "+inf" | "infinity" | "-infinity"
            )
        {
            return Ok(());
        }
        let mode = mode.to_string();
        // float(row["latency"]): absent column / null cell -> skip row.
        let Some(latency) = num_optional(ctx.row, r.col_optional("latency"))? else {
            return Ok(());
        };
        let leaf = ViewNode::Leaf(vec![("latency", ViewValue::F64(latency))]);
        root.insert_first_wins(&keys, &mode, leaf);
        any_leaf = true;
        Ok(())
    })?;
    Ok(match found {
        Some(()) if any_leaf => Some(root),
        _ => None,
    })
}

/// `operations/dsv4.py::_deep_merge_dsv4_dicts` — in-place merge preserving
/// dest insertion order; at any level where both sides are branches, recurse,
/// otherwise src overwrites (leaf fields merge by name like Python dicts).
fn deep_merge_view(dest: &mut ViewNode, src: ViewNode) {
    match (dest, src) {
        (ViewNode::Branch(dest_entries), ViewNode::Branch(src_entries)) => {
            for (key, src_value) in src_entries {
                match dest_entries.iter().position(|(k, _)| *k == key) {
                    Some(i) => deep_merge_view(&mut dest_entries[i].1, src_value),
                    None => dest_entries.push((key, src_value)),
                }
            }
        }
        (ViewNode::Leaf(dest_fields), ViewNode::Leaf(src_fields)) => {
            for (key, src_value) in src_fields {
                match dest_fields.iter().position(|(k, _)| *k == key) {
                    Some(i) => dest_fields[i].1 = src_value,
                    None => dest_fields.push((key, src_value)),
                }
            }
        }
        (dest, src) => *dest = src,
    }
}

/// Merge the csa+hca kind loads like `_load_dsv4_split`: None when every
/// side failed to load OR the merge came out empty.
fn merge_dsv4_split(parts: Vec<Option<ViewNode>>) -> Option<ViewNode> {
    let mut merged = ViewNode::branch();
    let mut any = false;
    for part in parts.into_iter().flatten() {
        any = true;
        deep_merge_view(&mut merged, part);
    }
    match &merged {
        _ if !any => None,
        ViewNode::Branch(entries) if entries.is_empty() => None,
        _ => Some(merged),
    }
}

/// Attribute registry: every table-view attribute [`table_view_json`]'s
/// dispatch accepts, with the parquet basenames its fold consumes — exported
/// over the FFI (`aiconfigurator_core.table_view_attributes()`) so the Python
/// side derives its mirrors from THIS table instead of hand-syncing four
/// stringly-typed sites (Rust match arms / `VIEW_KEY_LAYERS` / per-class
/// `load_data` literals / the baseline codec) — the same
/// single-source pattern as `gemm_quant_util_levels`. Completeness both ways
/// is pinned by `tests/cross_package/test_table_view_registry.py`, which
/// set-compares the export against `VIEW_KEY_LAYERS` and the codec inventory
/// AND dispatches every exported attribute against a pinned database (so a
/// registry/dispatch mismatch fails in CI rather than answering None on
/// machines missing that family's data).
pub const TABLE_VIEW_ATTRIBUTES: &[(&str, &[&str])] = &[
    ("_gemm_data", &["gemm_perf.parquet"]),
    ("_compute_scale_data", &["computescale_perf.parquet"]),
    ("_scale_matrix_data", &["scale_matrix_perf.parquet"]),
    (
        "_context_attention_data",
        &["context_attention_perf.parquet"],
    ),
    (
        "_generation_attention_data",
        &["generation_attention_perf.parquet"],
    ),
    (
        "_encoder_attention_data",
        &["encoder_attention_perf.parquet"],
    ),
    ("_context_mla_data", &["context_mla_perf.parquet"]),
    ("_generation_mla_data", &["generation_mla_perf.parquet"]),
    ("_mla_bmm_data", &["mla_bmm_perf.parquet"]),
    (
        "_context_mla_module_data",
        &["mla_context_module_perf.parquet"],
    ),
    (
        "_generation_mla_module_data",
        &["mla_generation_module_perf.parquet"],
    ),
    (
        "_wideep_context_mla_data",
        &["wideep_context_mla_perf.parquet"],
    ),
    (
        "_wideep_generation_mla_data",
        &["wideep_generation_mla_perf.parquet"],
    ),
    ("_moe_data", &["moe_perf.parquet"]),
    ("_moe_low_latency_data", &["moe_perf.parquet"]),
    (
        "_wideep_context_moe_data",
        &["wideep_context_moe_perf.parquet"],
    ),
    (
        "_wideep_generation_moe_data",
        &["wideep_generation_moe_perf.parquet"],
    ),
    (
        "_wideep_deepep_normal_data",
        &["wideep_deepep_normal_perf.parquet"],
    ),
    ("_wideep_deepep_ll_data", &["wideep_deepep_ll_perf.parquet"]),
    ("_wideep_moe_compute_data", &["wideep_moe_perf.parquet"]),
    ("_trtllm_alltoall_data", &["trtllm_alltoall_perf.parquet"]),
    (
        "_moe_a2a_data",
        &[
            "moe_a2a_perf.parquet",
            "wideep_deepep_normal_perf.parquet",
            "wideep_deepep_ll_perf.parquet",
            "trtllm_alltoall_perf.parquet",
        ],
    ),
    (
        "_moe_ep_data",
        &[
            "moe_expert_compute_perf.parquet",
            "wideep_context_moe_perf.parquet",
            "wideep_generation_moe_perf.parquet",
            "wideep_moe_perf.parquet",
        ],
    ),
    ("_custom_allreduce_data", &["custom_allreduce_perf.parquet"]),
    ("_nccl_data", &["nccl_perf.parquet"]),
    ("_oneccl_data", &["oneccl_perf.parquet"]),
    (
        "_context_dsa_module_data",
        &["dsa_context_module_perf.parquet"],
    ),
    (
        "_context_dsa_module_skip_data",
        &["dsa_context_module_perf.parquet"],
    ),
    (
        "_generation_dsa_module_data",
        &["dsa_generation_module_perf.parquet"],
    ),
    (
        "_generation_dsa_module_skip_data",
        &["dsa_generation_module_perf.parquet"],
    ),
    ("_mhc_module_data", &["mhc_module_perf.parquet"]),
    (
        "_context_deepseek_v4_attention_module_data",
        &[
            "dsv4_csa_context_module_perf.parquet",
            "dsv4_hca_context_module_perf.parquet",
        ],
    ),
    (
        "_generation_deepseek_v4_attention_module_data",
        &[
            "dsv4_csa_generation_module_perf.parquet",
            "dsv4_hca_generation_module_perf.parquet",
        ],
    ),
    (
        "_dsv4_sparse_kernel_data.paged_mqa_logits",
        &["dsv4_paged_mqa_logits_module_perf.parquet"],
    ),
    (
        "_dsv4_sparse_kernel_data.hca_attn",
        &["dsv4_hca_attn_module_perf.parquet"],
    ),
    (
        "_dsv4_sparse_kernel_data.csa_attn",
        &["dsv4_csa_attn_module_perf.parquet"],
    ),
    (
        "_dsv4_csa_topk_calib_data",
        &["dsv4_csa_topk_calib_perf.parquet"],
    ),
    (
        "_dsv4_megamoe_module_data",
        &["dsv4_megamoe_module_perf.parquet"],
    ),
    ("_mamba2_data", &["mamba2_perf.parquet"]),
    ("_gdn_data", &["gdn_perf.parquet"]),
    ("_kda_data", &["kda_perf.parquet"]),
];

/// The engine-side table view: `attribute` is the PerfDatabase attribute name
/// the retired Python loader used to fill (`"_gemm_data"`, ...). Returns
/// `Ok(None)` exactly when that loader returned `None` (every source path
/// missing), and the loader-shaped nested JSON otherwise. The three DSV4
/// sparse sub-tables are addressed as
/// `"_dsv4_sparse_kernel_data.<paged_mqa_logits|hca_attn|csa_attn>"`.
pub fn table_view_json(tables: &PerfTables, attribute: &str) -> Result<Option<String>, AicError> {
    let src = |basename: &str| {
        tables
            .source_resolver
            .sources_for(basename, &tables.data_root)
    };
    let prioritized_src = |basename: &str| {
        tables
            .source_resolver
            .prioritized_sources_for(basename, &tables.data_root)
    };
    let comm_src = |root: Option<&std::path::Path>, basename: &str| -> Vec<PerfSource> {
        match root {
            // _build_op_sources refused an EXISTING primary whose version dir
            // carries the legacy INCOMPLETE veto ("Not admitting primary
            // source ..."), leaving the comm op with no sources at all — the
            // view answers None there, never the vetoed rows.
            Some(dir) if !crate::perf_database::version_dir_is_unusable(dir) => {
                vec![PerfSource(dir.join(basename), None)]
            }
            _ => Vec::new(),
        }
    };
    let node = match attribute {
        "_gemm_data" => view_gemm(&src("gemm_perf.parquet")?)?,
        "_compute_scale_data" => view_gemm_scale(&src("computescale_perf.parquet")?)?,
        "_scale_matrix_data" => view_gemm_scale(&src("scale_matrix_perf.parquet")?)?,
        "_context_attention_data" => {
            view_context_attention(&src("context_attention_perf.parquet")?)?
        }
        "_generation_attention_data" => {
            view_generation_attention(&src("generation_attention_perf.parquet")?)?
        }
        "_encoder_attention_data" => {
            view_encoder_attention(&src("encoder_attention_perf.parquet")?)?
        }
        "_context_mla_data" => view_context_mla(&src("context_mla_perf.parquet")?)?,
        "_generation_mla_data" => view_generation_mla(&src("generation_mla_perf.parquet")?)?,
        "_mla_bmm_data" => view_mla_bmm(&src("mla_bmm_perf.parquet")?)?,
        "_context_mla_module_data" => {
            view_context_mla_module(&src("mla_context_module_perf.parquet")?)?
        }
        "_generation_mla_module_data" => {
            view_generation_mla_module(&src("mla_generation_module_perf.parquet")?)?
        }
        "_wideep_context_mla_data" => {
            view_wideep_context_mla(&src("wideep_context_mla_perf.parquet")?)?
        }
        "_wideep_generation_mla_data" => {
            view_wideep_generation_mla(&src("wideep_generation_mla_perf.parquet")?)?
        }
        // Each arm folds moe_perf.parquet (the largest shipped table) and
        // discards the twin — accepted: the fold runs once per (database,
        // attribute) behind MoE.load_data's class cache, not per query. If
        // this ever matters, cache the (default, low_latency) pair here.
        "_moe_data" => view_moe(&src("moe_perf.parquet")?)?.map(|(default, _)| default),
        "_moe_low_latency_data" => {
            view_moe(&src("moe_perf.parquet")?)?.map(|(_, low_latency)| low_latency)
        }
        "_wideep_context_moe_data" => view_wideep_moe(&src("wideep_context_moe_perf.parquet")?)?,
        "_wideep_generation_moe_data" => {
            view_wideep_moe(&src("wideep_generation_moe_perf.parquet")?)?
        }
        "_wideep_deepep_normal_data" => {
            view_wideep_deepep_normal(&src("wideep_deepep_normal_perf.parquet")?)?
        }
        "_wideep_deepep_ll_data" => view_wideep_deepep_ll(&src("wideep_deepep_ll_perf.parquet")?)?,
        "_wideep_moe_compute_data" => view_wideep_moe_compute(&src("wideep_moe_perf.parquet")?)?,
        "_trtllm_alltoall_data" => view_trtllm_alltoall(&src("trtllm_alltoall_perf.parquet")?)?,
        "_moe_a2a_data" => view_moe_a2a(
            &prioritized_src("moe_a2a_perf.parquet")?,
            &prioritized_src("wideep_deepep_normal_perf.parquet")?,
            &prioritized_src("wideep_deepep_ll_perf.parquet")?,
            &prioritized_src("trtllm_alltoall_perf.parquet")?,
        )?,
        "_moe_ep_data" => view_moe_expert_compute(
            &src("moe_expert_compute_perf.parquet")?,
            &src("wideep_context_moe_perf.parquet")?,
            &src("wideep_generation_moe_perf.parquet")?,
            &src("wideep_moe_perf.parquet")?,
        )?,
        "_custom_allreduce_data" => view_custom_allreduce(&src("custom_allreduce_perf.parquet")?)?,
        "_nccl_data" => view_nccl(&comm_src(
            tables.communication.nccl_root(),
            "nccl_perf.parquet",
        ))?,
        "_oneccl_data" => view_nccl(&comm_src(
            tables.communication.oneccl_root(),
            "oneccl_perf.parquet",
        ))?,
        "_context_dsa_module_data" => {
            view_context_dsa_module(&src("dsa_context_module_perf.parquet")?, false)?
        }
        "_context_dsa_module_skip_data" => {
            view_context_dsa_module(&src("dsa_context_module_perf.parquet")?, true)?
        }
        "_generation_dsa_module_data" => {
            view_generation_dsa_module(&src("dsa_generation_module_perf.parquet")?, false)?
        }
        "_generation_dsa_module_skip_data" => {
            view_generation_dsa_module(&src("dsa_generation_module_perf.parquet")?, true)?
        }
        "_mhc_module_data" => view_mhc_module(&src("mhc_module_perf.parquet")?)?,
        "_context_deepseek_v4_attention_module_data" => merge_dsv4_split(vec![
            view_context_dsv4_kind_module(&src("dsv4_csa_context_module_perf.parquet")?)?,
            view_context_dsv4_kind_module(&src("dsv4_hca_context_module_perf.parquet")?)?,
        ]),
        "_generation_deepseek_v4_attention_module_data" => merge_dsv4_split(vec![
            view_generation_dsv4_kind_module(&src("dsv4_csa_generation_module_perf.parquet")?)?,
            view_generation_dsv4_kind_module(&src("dsv4_hca_generation_module_perf.parquet")?)?,
        ]),
        "_dsv4_sparse_kernel_data.paged_mqa_logits" => {
            view_dsv4_sparse_kernel(&src("dsv4_paged_mqa_logits_module_perf.parquet")?)?
        }
        "_dsv4_sparse_kernel_data.hca_attn" => {
            view_dsv4_sparse_kernel(&src("dsv4_hca_attn_module_perf.parquet")?)?
        }
        "_dsv4_sparse_kernel_data.csa_attn" => {
            view_dsv4_sparse_kernel(&src("dsv4_csa_attn_module_perf.parquet")?)?
        }
        "_dsv4_csa_topk_calib_data" => {
            view_dsv4_csa_topk_calib(&src("dsv4_csa_topk_calib_perf.parquet")?)?
        }
        "_dsv4_megamoe_module_data" => view_dsv4_megamoe_module(&tables.data_root)?,
        "_mamba2_data" => view_mamba2(&src("mamba2_perf.parquet")?)?,
        "_gdn_data" => view_gdn(&src("gdn_perf.parquet")?)?,
        "_kda_data" => view_kda(&src("kda_perf.parquet")?)?,
        other => {
            return Err(AicError::PerfDatabase(format!(
                "unknown table-view attribute {other:?}"
            )))
        }
    };
    Ok(node.map(|n| n.to_json()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attribute_registry_is_well_formed() {
        let mut seen = std::collections::HashSet::new();
        for (attribute, basenames) in TABLE_VIEW_ATTRIBUTES {
            assert!(
                seen.insert(*attribute),
                "duplicate registry attribute {attribute:?}"
            );
            assert!(
                attribute.starts_with('_'),
                "registry attribute {attribute:?} is not an underscored PerfDatabase attribute"
            );
            assert!(
                !basenames.is_empty(),
                "registry attribute {attribute:?} lists no source basenames"
            );
            for basename in *basenames {
                assert!(
                    basename.ends_with(".parquet"),
                    "registry basename {basename:?} for {attribute:?} is not a parquet file"
                );
            }
        }
        // Dispatch completeness both ways is pinned by
        // tests/cross_package/test_table_view_registry.py, which fetches
        // every exported attribute against a real pinned database.
    }

    #[test]
    fn view_node_preserves_insertion_order_and_first_wins() {
        let mut root = ViewNode::branch();
        assert!(root.insert_first_wins(&["b".into()], "2", classic_leaf(1.0, 2.0)));
        assert!(root.insert_first_wins(&["a".into()], "1", classic_leaf(3.0, 0.0)));
        assert!(!root.insert_first_wins(&["b".into()], "2", classic_leaf(9.0, 9.0)));
        let json = root.to_json();
        assert_eq!(
            json,
            r#"{"b":{"2":{"latency":1.0,"power":2.0,"energy":2.0}},"a":{"1":{"latency":3.0,"power":0.0,"energy":0.0}}}"#
        );
    }

    #[test]
    fn float_formatting_round_trips_shortest() {
        let mut out = String::new();
        write_json_f64(&mut out, 0.1234567890123);
        assert_eq!(out, "0.1234567890123");
        out.clear();
        write_json_f64(&mut out, 2.0);
        assert_eq!(out, "2.0");
    }
}
