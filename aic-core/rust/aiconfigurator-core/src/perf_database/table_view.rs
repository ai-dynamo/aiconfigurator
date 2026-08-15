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
use crate::perf_database::{kernel_source_ok, resolve_op_sources, PerfSource, PerfTables};

/// Leaf field value. Integer-typed Python fields must serialize WITHOUT a
/// decimal point so `json.loads` rehydrates the exact Python type
/// (`int` vs `float`) the deleted loader stored.
#[derive(Clone, Debug)]
pub enum ViewValue {
    F64(f64),
    U64(u64),
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

    /// Descend to (creating) the branch at `path`, then insert `leaf` under
    /// `last` ONLY if absent — the Python loaders' first-wins conflict rule.
    /// Returns false on conflict (caller may log/count).
    pub fn insert_first_wins(&mut self, path: &[String], last: &str, leaf: ViewNode) -> bool {
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
        match node.child_index(last) {
            Some(i) => node.entries_mut()[i].1 = leaf,
            None => node.entries_mut().push((last.to_string(), leaf)),
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
                        ViewValue::U64(v) => {
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
    } else {
        // Perf tables never carry non-finite values; fail loudly if one does.
        unreachable!("non-finite value in a table view: {value}");
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
    path: &'a std::path::Path,
}

impl RowCtx<'_> {
    /// Python's `float(row.get("power", 0.0))` — absent column defaults 0.0.
    fn power(&self) -> Result<f64, AicError> {
        Ok(self.row.f64_optional(self.power_col)?.unwrap_or(0.0))
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
        for row in reader.rows()? {
            let row = row?;
            if !kernel_source_ok(source.1.as_deref(), ks_col, &row)? {
                continue;
            }
            per_row(&RowCtx {
                reader: &reader,
                row: &row,
                power_col,
                path,
            })?;
        }
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
        let window_size = ctx.row.u32_optional(r.col_optional("window_size"))?.unwrap_or(0);
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
        let window_size = ctx.row.u32_optional(r.col_optional("window_size"))?.unwrap_or(0);
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
        root.insert_first_wins(&path, &full_seq.to_string(), classic_leaf(latency, ctx.power()?));
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
    let is_b60 = sources.iter().any(|s| s.0.to_string_lossy().contains("b60"));
    let mut root = ViewNode::branch();
    let found = fold_sources(sources, |ctx| {
        let r = ctx.reader;
        let kernel_source = ctx
            .row
            .str_optional(r.col_optional("kernel_source"))?
            .unwrap_or("")
            .to_string();
        let backend = ctx.row.str_optional(r.col_optional("backend"))?.unwrap_or("").to_string();
        if (kernel_source.ends_with("_eager") || backend.ends_with("_eager")) && !is_b60 {
            return Ok(());
        }
        let tp_size = ctx.row.u32(r.col("num_gpus")?)?;
        let message_size = ctx.row.u64(r.col("message_size")?)?;
        let latency = ctx.row.f64(r.col("latency")?)?;
        let path = ["half".to_string(), tp_size.to_string(), "AUTO".to_string()];
        root.insert_first_wins(&path, &message_size.to_string(), classic_leaf(latency, ctx.power()?));
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
        let op_name = ctx.row.str_owned(r.col("op_name")?)?;
        let latency = ctx.row.f64(r.col("latency")?)?;
        let path = [dtype, op_name, num_gpus.to_string()];
        root.insert_first_wins(&path, &message_size.to_string(), classic_leaf(latency, ctx.power()?));
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
        root.insert_first_wins(&path, &full_seq.to_string(), classic_leaf(latency, ctx.power()?));
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
        let op_name = ctx.row.str_owned(r.col("op_name")?)?;
        let latency = ctx.row.f64(r.col("latency")?)?;
        let path = [quant, op_name, num_heads.to_string()];
        root.insert_first_wins(&path, &num_tokens.to_string(), classic_leaf(latency, ctx.power()?));
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
        let kernel_source = match r.col_optional("kernel_source") {
            Some(col) => ctx.row.str_optional(Some(col))?.unwrap_or("").to_string(),
            None => "flashinfer".to_string(),
        };
        let quant = ctx.row.str_owned(r.col("mla_dtype")?)?;
        let kv_dtype = ctx.row.str_owned(r.col("kv_cache_dtype")?)?;
        let b = ctx.row.u32(r.col("batch_size")?)?;
        let s = ctx.row.u32(r.col("isl")?)?;
        let num_heads = ctx.row.u32(r.col("num_heads")?)?;
        let latency = ctx.row.f64(r.col("latency")?)?;
        let path = [kernel_source, quant, kv_dtype, num_heads.to_string(), s.to_string()];
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
        let kernel_source = match r.col_optional("kernel_source") {
            Some(col) => ctx.row.str_optional(Some(col))?.unwrap_or("").to_string(),
            None => "flashinfer".to_string(),
        };
        let kv_dtype = ctx.row.str_owned(r.col("kv_cache_dtype")?)?;
        let b = ctx.row.u32(r.col("batch_size")?)?;
        let s = ctx.row.u32(r.col("isl")?)?;
        let step = ctx.row.u32(r.col("step")?)?;
        let num_heads = ctx.row.u32(r.col("num_heads")?)?;
        let latency = ctx.row.f64(r.col("latency")?)?;
        let full_seq = s + step;
        let path = [kernel_source, kv_dtype, num_heads.to_string(), b.to_string()];
        root.insert_first_wins(&path, &full_seq.to_string(), classic_leaf(latency, ctx.power()?));
        Ok(())
    })?;
    Ok(found.map(|_| root))
}

/// Python `operations/mla.py::_MLA_MODULE_NATIVE_HEADS` — the model pin that
/// keys module rows' native-head identity (#1458).
const MLA_MODULE_NATIVE_HEADS: &[(&str, u32)] = &[
    ("deepseek-ai/DeepSeek-V3", 128),
    ("deepseek-ai/DeepSeek-R1", 128),
    ("nvidia/DeepSeek-V3.1-NVFP4", 128),
];

/// Python `operations/mla.py::_mla_module_native_heads`: model pin lookup +
/// the per-row rank-local consistency check (#1429/#1458), fail-loud.
fn mla_module_native_heads(ctx: &RowCtx<'_>, num_heads: u32) -> Result<u32, AicError> {
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
    let native = MLA_MODULE_NATIVE_HEADS
        .iter()
        .find(|(m, _)| *m == model)
        .map(|(_, n)| *n)
        .ok_or_else(|| {
            AicError::PerfDatabase(format!(
                "MLA module row in {} names unpinned model {model:?}; add its native head \
                 count to MLA_MODULE_NATIVE_HEADS when landing the data (#1458).",
                ctx.path.display()
            ))
        })?;
    let tp_size = ctx
        .row
        .u32_optional(r.col_optional("tp_size"))?
        .filter(|v| *v != 0)
        .unwrap_or(1);
    if tp_size > 1 && num_heads * tp_size != native {
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
        let native = mla_module_native_heads(ctx, num_heads)?;
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
        let native = mla_module_native_heads(ctx, num_heads)?;
        let b = ctx.row.u32(r.col("batch_size")?)?;
        let s = ctx.row.u32(r.col("isl")?)? + ctx.row.u32(r.col("step")?)?;
        let latency = ctx.row.f64(r.col("latency")?)?;
        let power = first_power.power(ctx)?;
        let kv_dtype = ctx.row.str_owned(r.col("kv_cache_dtype")?)?;
        let gemm = ctx.row.str_owned(r.col("gemm_type")?)?;
        let path = [kv_dtype, gemm, native.to_string(), num_heads.to_string(), b.to_string()];
        root.insert_first_wins(&path, &s.to_string(), classic_leaf(latency, power));
        Ok(())
    })?;
    Ok(found.map(|_| root))
}

/// Ensure a (possibly empty) branch exists at `path` — mirrors Python
/// defaultdict "vivification" that some loaders rely on (deliberately or,
/// in mamba2's generation case, as a preserved bug).
fn vivify_branch(root: &mut ViewNode, path: &[String]) {
    let mut node = root;
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
        let kernel_source = ctx.row.str_owned(r.col("kernel_source")?)?;
        let phase = ctx.row.str_owned(r.col("phase")?)?;
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

/// Python `operations/mamba.py::_GDN_DECODE_RECURRENCE_ALIASES` — the decode
/// recurrence kernel-name drift across sglang releases, normalized on load.
fn gdn_kernel_alias(kernel_source: &str) -> &str {
    match kernel_source {
        "fused_recurrent_gated_delta_rule" | "fused_recurrent_gated_delta_rule_packed_decode" => {
            "fused_sigmoid_gating_delta_rule_update"
        }
        other => other,
    }
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
        let raw_ks = ctx.row.str_owned(r.col("kernel_source")?)?;
        let kernel_source = if alias {
            gdn_kernel_alias(&raw_ks).to_string()
        } else {
            raw_ks
        };
        let phase = ctx.row.str_owned(r.col("phase")?)?;
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
        &["d_model", "num_k_heads", "head_k_dim", "num_v_heads", "head_v_dim", "d_conv"],
        &["context"],
        true,
    )
}

/// `operations/mamba.py::load_kda_data` — same columns as GDN, but "verify"
/// rows are 2-D too (seq_len carries the draft-token count), no aliasing.
pub fn view_kda(sources: &[PerfSource]) -> Result<Option<ViewNode>, AicError> {
    view_gdn_like(
        sources,
        &["d_model", "num_k_heads", "head_k_dim", "num_v_heads", "head_v_dim", "d_conv"],
        &["context", "verify"],
        false,
    )
}

/// The `load_moe_data` kernel-routed quant rewrites: DeepSeek-V4-Pro's
/// Blackwell trtllm-gen MXFP4xMXFP8 kernel and Hopper flashinfer-cutlass
/// mixed-GEMM rows get dedicated quant modes distinct from the modes the
/// collector logged them under.
fn moe_quant_rewrite(quant: &str, kernel_source: &str) -> String {
    if quant == "w4a8_mxfp4_mxfp8" && kernel_source == "sglang_mxfp4_flashinfer_trtllm_moe" {
        return "w4a8_mxfp4_mxfp8_trtllm".to_string();
    }
    if quant == "w4a16_mxfp4" && kernel_source == "sglang_flashinfer_cutlass_moe" {
        return "w4a16_mxfp4_cutlass".to_string();
    }
    quant.to_string()
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
        let raw_quant = ctx.row.str_owned(r.col("moe_dtype")?)?;
        let num_tokens = ctx.row.u32(r.col("num_tokens")?)?;
        let hidden = ctx.row.u32(r.col("hidden_size")?)?;
        let inter = ctx.row.u32(r.col("inter_size")?)?;
        let topk = ctx.row.u32(r.col("topk")?)?;
        let num_experts = ctx.row.u32(r.col("num_experts")?)?;
        let moe_tp = ctx.row.u32(r.col("moe_tp_size")?)?;
        let moe_ep = ctx.row.u32(r.col("moe_ep_size")?)?;
        let distribution = ctx.row.str_owned(r.col("distribution")?)?;
        let latency = ctx.row.f64(r.col("latency")?)?;
        let kernel_source = ctx.row.str_owned(r.col("kernel_source")?)?;
        let quant = moe_quant_rewrite(&raw_quant, &kernel_source);
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
        target.insert_first_wins(&path, &num_tokens.to_string(), classic_leaf(latency, ctx.power()?));
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
        let distribution = ctx.row.str_owned(r.col("distribution")?)?;
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
        root.insert_first_wins(&path, &num_tokens.to_string(), classic_leaf(latency, ctx.power()?));
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
        let lat = ctx.row.f64(r.col("combine_avg_t_us")?)? + ctx.row.f64(r.col("dispatch_avg_t_us")?)?;
        let path = [
            node_num.to_string(),
            hidden.to_string(),
            num_topk.to_string(),
            num_experts.to_string(),
        ];
        root.insert_first_wins(&path, &num_token.to_string(), classic_leaf(lat, ctx.power()?));
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
        root.insert_first_wins(&path, &num_token.to_string(), classic_leaf(lat, ctx.power()?));
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
        let distribution = ctx.row.str_owned(r.col("distribution")?)?;
        let latency = ctx.row.f64(r.col("latency")?)?;
        let kernel_source = match r.col_optional("kernel_source") {
            Some(col) => ctx.row.str_optional(Some(col))?.unwrap_or("").to_string(),
            None => "moe_torch_flow".to_string(),
        };
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
        root.insert_first_wins(&path, &num_tokens.to_string(), classic_leaf(latency, ctx.power()?));
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
        let op_name = ctx.row.str_owned(r.col("op_name")?)?;
        let quant = ctx.row.str_owned(r.col("moe_dtype")?)?;
        let num_tokens = ctx.row.u32(r.col("num_tokens")?)?;
        let hidden = ctx.row.u32(r.col("hidden_size")?)?;
        let topk = ctx.row.u32(r.col("topk")?)?;
        let num_experts = ctx.row.u32(r.col("num_experts")?)?;
        let moe_ep = ctx.row.u32(r.col("moe_ep_size")?)?;
        let latency = ctx.row.f64(r.col("latency")?)?;
        let kernel_source = match r.col_optional("kernel_source") {
            Some(col) => ctx.row.str_optional(Some(col))?.unwrap_or("").to_string(),
            None => "NVLinkTwoSided".to_string(),
        };
        let num_nodes = if has_num_nodes {
            ctx.row.u32(r.col("num_nodes")?)?
        } else {
            (moe_ep / 4).max(1)
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
        root.insert_first_wins(&path, &num_tokens.to_string(), classic_leaf(latency, ctx.power()?));
        Ok(())
    })?;
    Ok(found.map(|_| root))
}

/// Numeric cell as f64 regardless of the parquet storage type (DOUBLE or
/// INT64) — Python read both through `int(float(x))` / `float(x)` without
/// caring. None when the column is absent or the cell is null.
fn num_optional(row: &PerfRow, col: Option<usize>) -> Result<Option<f64>, AicError> {
    if let Some(v) = row.f64_optional(col)? {
        return Ok(Some(v));
    }
    Ok(row.u32_optional(col)?.map(|v| v as f64))
}

/// `moe_comm.py::_normalize_sms` — null/NaN/absent sms -> 0.
fn normalize_sms(ctx: &RowCtx<'_>, col: Option<usize>) -> Result<u32, AicError> {
    match num_optional(ctx.row, col)? {
        None => Ok(0),
        Some(v) if v.is_nan() => Ok(0),
        Some(v) => Ok(v as u32),
    }
}

/// `moe_comm.py::_row_power` — null/NaN/absent power -> 0.0; a measured but
/// non-finite value is corrupt data.
fn row_power_lenient(ctx: &RowCtx<'_>) -> Result<f64, AicError> {
    match ctx.row.f64_optional(ctx.power_col)? {
        None => Ok(0.0),
        Some(v) if v.is_nan() => Ok(0.0),
        Some(v) if !v.is_finite() => Err(AicError::PerfDatabase(
            "non-finite power cell in perf data: power must be finite when measured".to_string(),
        )),
        Some(v) => Ok(v),
    }
}

/// `moe_comm.py::_require_latency` — latency is schema-required and finite.
fn require_latency(ctx: &RowCtx<'_>, table: &str) -> Result<f64, AicError> {
    let lat = ctx
        .row
        .f64_optional(ctx.reader.col_optional("latency"))?
        .ok_or_else(|| {
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

/// `operations/moe_comm.py::load_moe_a2a_data` — the unified 10-level a2a
/// store `[comm_backend][phase][comm_dtype][ep_size][node_num][hidden][topk]
/// [num_experts][sms][num_tokens]`. Legacy adapters load first (keep-first);
/// the first new-schema occurrence of a key overwrites a legacy leaf, and
/// new-schema repeats keep the first row. New-schema latency is µs -> ms.
pub fn view_moe_a2a(
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
                ("dispatch", &["dispatch_transmit_us", "dispatch_notify_us"][..]),
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
                    "default".to_string(),
                    (node_num * 8).to_string(),
                    node_num.to_string(),
                    ctx.row.u32(r.col("hidden_size")?)?.to_string(),
                    ctx.row.u32(r.col("num_topk")?)?.to_string(),
                    ctx.row.u32(r.col("num_experts")?)?.to_string(),
                    sms.to_string(),
                ];
                let leaf = ViewNode::Leaf(vec![
                    ("latency", ViewValue::F64(latency)),
                    ("power", ViewValue::F64(power)),
                    ("energy", ViewValue::F64(power * latency)),
                ]);
                root.insert_first_wins(&path, &ctx.row.u32(r.col("num_token")?)?.to_string(), leaf);
            }
            Ok(())
        })?;
        legacy_loaded |= found.is_some();
    }

    // _adapt_legacy_trtllm_alltoall: latency already ms; per-row num_nodes
    // fallback max(1, ep//4); unmapped kernel/op rows are skipped.
    let found = fold_sources(legacy_trtllm_alltoall, |ctx| {
        let r = ctx.reader;
        let kernel_source = match r.col_optional("kernel_source") {
            Some(col) => ctx.row.str_optional(Some(col))?.unwrap_or("").to_string(),
            None => "NVLinkTwoSided".to_string(),
        };
        let comm_backend = match kernel_source.as_str() {
            "NVLinkTwoSided" => "nvlink_two_sided",
            "NVLinkOneSided" => "nvlink_one_sided",
            _ => return Ok(()),
        };
        let op_name = ctx.row.str_owned(r.col("op_name")?)?;
        let (phase, fixed_dtype) = match op_name.as_str() {
            "alltoall_prepare" => ("prepare", None),
            "alltoall_dispatch" => ("dispatch", None),
            "alltoall_combine" => ("combine", None),
            "alltoall_combine_low_precision" => ("combine", Some("fp4")),
            _ => return Ok(()),
        };
        let comm_dtype = match fixed_dtype {
            Some(d) => d.to_string(),
            None => ctx.row.str_owned(r.col("moe_dtype")?)?,
        };
        let ep_size = ctx.row.u32(r.col("moe_ep_size")?)?;
        let node_num = match r.col_optional("num_nodes") {
            Some(col) => ctx.row.u32(col)?,
            None => (ep_size / 4).max(1),
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
        let leaf = ViewNode::Leaf(vec![
            ("latency", ViewValue::F64(latency)),
            ("power", ViewValue::F64(power)),
            ("energy", ViewValue::F64(power * latency)),
        ]);
        root.insert_first_wins(&path, &ctx.row.u32(r.col("num_tokens")?)?.to_string(), leaf);
        Ok(())
    })?;
    legacy_loaded |= found.is_some();

    // New schema: µs -> ms, sms normalized, first occurrence overwrites legacy.
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let new_found = fold_sources(sources, |ctx| {
        let r = ctx.reader;
        let sms = normalize_sms(ctx, r.col_optional("sms"))?;
        let path = [
            ctx.row.str_owned(r.col("comm_backend")?)?,
            ctx.row.str_owned(r.col("phase")?)?,
            ctx.row.str_owned(r.col("comm_dtype")?)?,
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
        let leaf = ViewNode::Leaf(vec![
            ("latency", ViewValue::F64(latency)),
            ("power", ViewValue::F64(power)),
            ("energy", ViewValue::F64(power * latency)),
        ]);
        let full_key = format!("{}\x1f{}", path.join("\x1f"), num_tokens);
        if seen.insert(full_key) {
            root.insert_overwrite(&path, &num_tokens, leaf);
        } else {
            root.insert_first_wins(&path, &num_tokens, leaf);
        }
        Ok(())
    })?;

    Ok(if new_found.is_some() || legacy_loaded { Some(root) } else { None })
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

    for (legacy_sources, inference_phase) in [(legacy_context, "context"), (legacy_generation, "generation")] {
        let found = fold_sources(legacy_sources, |ctx| {
            let r = ctx.reader;
            let latency = ctx.row.f64(r.col("latency")?)?;
            let power = row_power_lenient(ctx)?;
            let num_experts = ctx.row.u32(r.col("num_experts")?)?;
            let path = [
                "deepep_moe".to_string(),
                ctx.row.str_owned(r.col("moe_dtype")?)?,
                ctx.row.str_owned(r.col("distribution")?)?,
                inference_phase.to_string(),
                ctx.row.u32(r.col("topk")?)?.to_string(),
                num_experts.to_string(),
                num_experts.to_string(),
                ctx.row.u32(r.col("hidden_size")?)?.to_string(),
                ctx.row.u32(r.col("inter_size")?)?.to_string(),
                ctx.row.u32(r.col("moe_tp_size")?)?.to_string(),
                ctx.row.u32(r.col("moe_ep_size")?)?.to_string(),
            ];
            let leaf = ViewNode::Leaf(vec![
                ("latency", ViewValue::F64(latency)),
                ("power", ViewValue::F64(power)),
                ("energy", ViewValue::F64(power * latency)),
            ]);
            root.insert_first_wins(&path, &ctx.row.u32(r.col("num_tokens")?)?.to_string(), leaf);
            Ok(())
        })?;
        legacy_loaded |= found.is_some();
    }

    let found = fold_sources(legacy_trtllm_wideep, |ctx| {
        let r = ctx.reader;
        let latency = ctx.row.f64(r.col("latency")?)?;
        let power = row_power_lenient(ctx)?;
        let kernel_source = match r.col_optional("kernel_source") {
            Some(col) => ctx.row.str_optional(Some(col))?.unwrap_or("").to_string(),
            None => "moe_torch_flow".to_string(),
        };
        for inference_phase in ["context", "generation"] {
            let path = [
                kernel_source.clone(),
                ctx.row.str_owned(r.col("moe_dtype")?)?,
                ctx.row.str_owned(r.col("distribution")?)?,
                inference_phase.to_string(),
                ctx.row.u32(r.col("topk")?)?.to_string(),
                ctx.row.u32(r.col("num_experts")?)?.to_string(),
                ctx.row.u32(r.col("num_slots")?)?.to_string(),
                ctx.row.u32(r.col("hidden_size")?)?.to_string(),
                ctx.row.u32(r.col("inter_size")?)?.to_string(),
                ctx.row.u32(r.col("moe_tp_size")?)?.to_string(),
                ctx.row.u32(r.col("moe_ep_size")?)?.to_string(),
            ];
            let leaf = ViewNode::Leaf(vec![
                ("latency", ViewValue::F64(latency)),
                ("power", ViewValue::F64(power)),
                ("energy", ViewValue::F64(power * latency)),
            ]);
            root.insert_first_wins(&path, &ctx.row.u32(r.col("num_tokens")?)?.to_string(), leaf);
        }
        Ok(())
    })?;
    legacy_loaded |= found.is_some();

    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let new_found = fold_sources(sources, |ctx| {
        let r = ctx.reader;
        let path = [
            ctx.row.str_owned(r.col("kernel_source")?)?,
            ctx.row.str_owned(r.col("moe_dtype")?)?,
            ctx.row.str_owned(r.col("distribution")?)?,
            ctx.row.str_owned(r.col("inference_phase")?)?,
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
        let leaf = ViewNode::Leaf(vec![
            ("latency", ViewValue::F64(latency)),
            ("power", ViewValue::F64(power)),
            ("energy", ViewValue::F64(power * latency)),
        ]);
        let full_key = format!("{}\x1f{}", path.join("\x1f"), num_tokens);
        if seen.insert(full_key) {
            root.insert_overwrite(&path, &num_tokens, leaf);
        } else {
            root.insert_first_wins(&path, &num_tokens, leaf);
        }
        Ok(())
    })?;

    Ok(if new_found.is_some() || legacy_loaded { Some(root) } else { None })
}

/// `operations/dsa.py::_dsa_kernel_source_buckets` — configured-backend
/// bucket fan-out. BF16 KV rows back BOTH buckets; FP8 rows bucket by the
/// executed kernel name with the legacy substring fallback.
fn dsa_kernel_source_buckets(kernel_source: &str, kv_dtype: &str) -> &'static [&'static str] {
    if kv_dtype == "bfloat16" {
        return &["trtllm", "flashmla_kv"];
    }
    match kernel_source {
        "sglang_dsa_indexer_trtllm" | "sglang_dsa_skip_indexer_trtllm" => &["trtllm"],
        "sglang_dsa_indexer_flashmla_sparse" | "sglang_dsa_skip_indexer_flashmla_sparse" => &["flashmla_kv"],
        "sglang_dsa_dense_mha_trtllm_ragged" => &["trtllm", "flashmla_kv"],
        ks if ks.contains("trtllm") => &["trtllm"],
        _ => &["flashmla_kv"],
    }
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
    let mut any_exists = false;
    let mut first_has_power: Option<bool> = None;

    for source in sources {
        let path = &source.0;
        if !path.exists() {
            continue;
        }
        any_exists = true;
        let reader = PerfReader::open(path)?;
        let power_col = reader.col_optional("power");
        let ks_col = reader.col_optional("kernel_source");
        let mut source_values = DsaSourceValues::new();
        for row in reader.rows()? {
            let row = row?;
            if !kernel_source_ok(source.1.as_deref(), ks_col, &row)? {
                continue;
            }
            let has_power = *first_has_power.get_or_insert(power_col.is_some());
            let op_name = row.str_optional(reader.col_optional("op_name"))?.unwrap_or("");
            if op_name.contains("skip_indexer") != skip {
                continue;
            }
            let num_heads = row.u32(reader.col("num_heads")?)?;
            let b = row.u32(reader.col("batch_size")?)?;
            let latency = row.f64(reader.col("latency")?)?;
            let power = if has_power {
                row.f64_optional(power_col)?.unwrap_or(0.0)
            } else {
                0.0
            };
            let arch = match reader.col_optional("architecture") {
                Some(col) => row.str_optional(Some(col))?.unwrap_or("").to_string(),
                None => "DeepseekV32ForCausalLM".to_string(),
            };
            let gemm = row.str_owned(reader.col("gemm_type")?)?;
            let kv_dtype = row.str_owned(reader.col("kv_cache_dtype")?)?;
            let ks = row.str_optional(ks_col)?.unwrap_or("").to_string();
            let leaf = ViewNode::Leaf(vec![
                ("latency", ViewValue::F64(latency)),
                ("power", ViewValue::F64(power)),
                ("energy", ViewValue::F64(power * latency)),
            ]);
            if context {
                let s = row.u32(reader.col("isl")?)?;
                let fmha = row.str_owned(reader.col("mla_dtype")?)?;
                let step = num_optional(&row, reader.col_optional("step"))?;
                let step_missing = step.is_none();
                if arch == "GlmMoeDsaForCausalLM" && step_missing {
                    return Err(AicError::PerfDatabase(
                        "GLM-5 context DSA module data requires a non-empty step column for \
                         prefix/past_kv length"
                            .to_string(),
                    ));
                }
                let prefix = step.map(|v| v as u32).unwrap_or(0);
                for backend in dsa_kernel_source_buckets(&ks, &kv_dtype) {
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
                let s = row.u32(reader.col("isl")?)? + row.u32(reader.col("step")?)?;
                for backend in dsa_kernel_source_buckets(&ks, &kv_dtype) {
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
        }
        for (coordinate, leaf) in source_values.entries {
            let key = coordinate.join("\x1f");
            if !seen.insert(key) {
                continue;
            }
            let (path_part, last) = coordinate.split_at(coordinate.len() - 1);
            root.insert_first_wins(path_part, &last[0], leaf);
        }
    }

    Ok(if any_exists { Some(root) } else { None })
}

/// `operations/dsa.py::load_context_dsa_module_data` —
/// `[mla_dtype][kv_cache_dtype][gemm_type][architecture][dsa_backend][num_heads][prefix][s][b]`.
pub fn view_context_dsa_module(sources: &[PerfSource], skip: bool) -> Result<Option<ViewNode>, AicError> {
    view_dsa_module(sources, true, skip)
}

/// `operations/dsa.py::load_generation_dsa_module_data` —
/// `[kv_cache_dtype][gemm_type][architecture][dsa_backend][num_heads][b][isl+step]`.
pub fn view_generation_dsa_module(sources: &[PerfSource], skip: bool) -> Result<Option<ViewNode>, AicError> {
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
        let op = ctx.row.str_owned(r.col("op_name")?)?;
        let hc_mult = ctx.row.u32(r.col("hc_mult")?)?;
        let hidden = ctx.row.u32(r.col("hidden_size")?)?;
        let num_tokens = ctx.row.u32(r.col("num_tokens")?)?;
        let latency = ctx.row.f64(r.col("latency")?)?;
        let power = first_power.power(ctx)?;
        let path = [op, hc_mult.to_string(), hidden.to_string()];
        root.insert_first_wins(&path, &num_tokens.to_string(), classic_leaf_with(latency, power));
        Ok(())
    })?;
    Ok(found.map(|_| root))
}

/// `operations/dsv4.py::_dsv4_normalize_dtype` — sglang column naming to
/// canonical enum names.
fn dsv4_normalize_dtype(name: &str) -> &str {
    match name {
        "fp8_e4m3" => "fp8",
        other => other,
    }
}

/// `operations/dsv4.py::_validate_dsv4_local_head_semantics` — reject stale
/// pre-#1131 NATIVE-heads files by the per-(model, version) fingerprint, and
/// require a parseable tp_size everywhere once the file carries the column.
fn validate_dsv4_local_head_semantics(
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
    let mut observed: std::collections::BTreeMap<(String, String), std::collections::BTreeSet<(u32, u32)>> =
        std::collections::BTreeMap::new();
    for (heads, tp, model, version) in rows {
        let tp = tp.map(|v| v.max(1)).unwrap_or(1);
        observed
            .entry((model.clone(), version.clone()))
            .or_default()
            .insert((*heads, tp));
    }
    for ((model, version), pairs) in observed {
        let tps: std::collections::BTreeSet<u32> = pairs.iter().map(|(_, tp)| *tp).collect();
        let heads: std::collections::BTreeSet<u32> = pairs.iter().map(|(h, _)| *h).collect();
        let products: std::collections::BTreeSet<u64> =
            pairs.iter().map(|(h, tp)| *h as u64 * *tp as u64).collect();
        if tps.len() > 1 && heads.len() == 1 && products.len() != 1 {
            return Err(AicError::PerfDatabase(format!(
                "DSV4 module rows for model={model:?} version={version:?} in {path_label} keep \
                 num_heads constant across tp_size values {tps:?}: that is the retired \
                 pre-#1131 NATIVE semantics (#1429). Migrate the file to rank-local heads \
                 (num_heads //= tp_size) before loading."
            )));
        }
    }
    Ok(())
}

/// Shared body of the two DSV4 attention-kind loaders. Malformed rows are
/// SKIPPED (the Python loaders' try/except-continue), matching the appended
/// duplicate-header tolerance.
fn view_dsv4_kind_module(sources: &[PerfSource], context: bool) -> Result<Option<ViewNode>, AicError> {
    // Two passes like Python: the semantics validator scans the full row set
    // first, then the fold runs. Collect the decoded fields once.
    struct Decoded {
        b: u32,
        s: u32,
        prefix: u32,
        cr: u32,
        latency: f64,
        heads: u32,
        tp: u32,
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
        // tp parseable -> saw_tp_size, else missing_tp_rows += 1.
        let heads_opt = ctx.row.u32_optional(r.col_optional("num_heads"))?;
        if let Some(heads) = heads_opt {
            let tp_opt = ctx.row.u32_optional(r.col_optional("tp_size"))?;
            if tp_opt.is_some() {
                saw_tp_size = true;
            } else {
                missing_tp_rows += 1;
            }
            let model = ctx.row.str_optional(r.col_optional("model"))?.unwrap_or("").to_string();
            let version = ctx
                .row
                .str_optional(r.col_optional("version"))?
                .unwrap_or("")
                .to_string();
            fingerprint.push((heads, tp_opt, model, version));
        }
        // Row decode with skip-on-malformed semantics.
        let (Some(b), Some(s_raw), Some(cr), Some(latency), Some(heads)) = (
            ctx.row.u32_optional(r.col_optional("batch_size"))?,
            ctx.row.u32_optional(r.col_optional("isl"))?,
            ctx.row.u32_optional(r.col_optional("compress_ratio"))?,
            ctx.row.f64_optional(r.col_optional("latency"))?,
            heads_opt,
        ) else {
            return Ok(());
        };
        let step = num_optional(ctx.row, r.col_optional("step"))?;
        let (s, prefix) = if context {
            (s_raw, step.map(|v| v as u32).unwrap_or(0))
        } else {
            let Some(step) = step else { return Ok(()) };
            (s_raw + step as u32, 0)
        };
        let tp = ctx
            .row
            .u32_optional(r.col_optional("tp_size"))?
            .map(|v| v.max(1))
            .unwrap_or(1);
        let power = if has_power {
            ctx.row.f64_optional(ctx.power_col)?.unwrap_or(0.0)
        } else {
            0.0
        };
        let (Some(gemm), Some(kv)) = (
            ctx.row.str_optional(r.col_optional("gemm_type"))?,
            ctx.row.str_optional(r.col_optional("kv_cache_dtype"))?,
        ) else {
            return Ok(());
        };
        let fmha = if context {
            let Some(fmha) = ctx.row.str_optional(r.col_optional("mla_dtype"))? else {
                return Ok(());
            };
            dsv4_normalize_dtype(fmha).to_string()
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
            gemm: gemm.to_string(),
            fmha,
            kv: dsv4_normalize_dtype(kv).to_string(),
        });
        Ok(())
    })?;
    if found.is_none() {
        return Ok(None);
    }
    validate_dsv4_local_head_semantics(&fingerprint, saw_tp_size, missing_tp_rows, &path_label)?;

    let mut root = ViewNode::branch();
    for row in decoded {
        let native = row.heads * row.tp;
        let leaf = classic_leaf_with(row.latency, row.power);
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
pub fn view_generation_dsv4_kind_module(sources: &[PerfSource]) -> Result<Option<ViewNode>, AicError> {
    view_dsv4_kind_module(sources, false)
}

/// `operations/dsv4.py::load_dsv4_megamoe_module_data` — the 15-level
/// wide-leaf MegaMoE table; single-source only, boolean invariants enforced,
/// DUPLICATE rows are a load error (not first-wins).
pub fn view_dsv4_megamoe_module(sources: &[PerfSource]) -> Result<Option<ViewNode>, AicError> {
    // Python receives the primary path only (never the shared-layer list).
    let Some(primary) = sources.first() else {
        return Ok(None);
    };
    let single = [PerfSource(primary.0.clone(), None)];
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
                Some(col) => ctx.row.bool(col),
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
        let kernel_source = match r.col_optional("kernel_source") {
            Some(col) => ctx.row.str_optional(Some(col))?.unwrap_or("").to_string(),
            None => "deepgemm_megamoe".to_string(),
        };
        let kernel_dtype = ctx.row.str_owned(r.col("kernel_dtype")?)?;
        let quant = ctx.row.str_owned(r.col("moe_dtype")?)?;
        let pre_dispatch = ctx.row.str_owned(r.col("pre_dispatch")?)?;
        let source_policy = ctx.row.str_owned(r.col("source_policy")?)?;
        let distribution = ctx.row.str_owned(r.col("distribution")?)?;
        let topk = ctx.row.u32(r.col("topk")?)?;
        let num_experts = ctx.row.u32(r.col("num_experts")?)?;
        let num_fused = ctx
            .row
            .u32_optional(r.col_optional("num_fused_shared_experts"))?
            .unwrap_or(0);
        let hidden = ctx.row.u32(r.col("hidden_size")?)?;
        let inter = ctx.row.u32(r.col("inter_size")?)?;
        let moe_tp = ctx.row.u32_optional(r.col_optional("moe_tp_size"))?.unwrap_or(1);
        let moe_ep = ctx.row.u32(r.col("moe_ep_size")?)?;
        let num_tokens = ctx.row.u32(r.col("num_tokens")?)?;
        let latency = ctx.row.f64(r.col("latency")?)?;
        let power = ctx.row.f64_optional(ctx.power_col)?.unwrap_or(0.0);
        let num_max = ctx
            .row
            .u32_optional(r.col_optional("num_max_tokens_per_rank"))?
            .unwrap_or(0);
        let effective_num_max = ctx
            .row
            .u32_optional(r.col_optional("effective_num_max_tokens_per_rank"))?
            .filter(|v| *v != 0)
            .unwrap_or(num_max);
        let global_tokens = ctx
            .row
            .u32_optional(r.col_optional("global_num_tokens"))?
            .filter(|v| *v != 0)
            .unwrap_or(num_tokens * moe_ep);
        let phase = ctx.row.str_optional(r.col_optional("phase"))?.unwrap_or("").trim().to_string();
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
            ("global_num_tokens", ViewValue::U64(global_tokens as u64)),
            ("num_max_tokens_per_rank", ViewValue::U64(num_max as u64)),
            (
                "effective_num_max_tokens_per_rank",
                ViewValue::U64(effective_num_max as u64),
            ),
            ("used_cuda_graph", ViewValue::Bool(true)),
            ("kernel_dtype", ViewValue::Str(kernel_dtype.clone())),
            ("routed_scaling_factor", ViewValue::F64(routed_scaling_factor)),
            ("includes_routed_scale", ViewValue::Bool(true)),
            ("includes_gate_topk", ViewValue::Bool(false)),
            ("buffer_policy", ViewValue::Str(buffer_policy)),
            ("includes_buffer_init", ViewValue::Bool(includes_buffer_init)),
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
        if ctx.row.u32_optional(r.col_optional("batch_size"))?.is_none() {
            return Ok(()); // header-dup / null guard
        }
        let mut keys: Vec<String> = Vec::with_capacity(KEY_COLS.len());
        for col in KEY_COLS {
            let Some(idx) = r.col_optional(col) else { return Ok(()) };
            // Numeric coercion int(float(v)); non-finite/null keys skip the row.
            let value = match ctx.row.u32_optional(Some(idx))? {
                Some(v) => v,
                None => match ctx.row.f64_optional(Some(idx))? {
                    Some(v) if v.is_finite() => v as u32,
                    _ => return Ok(()),
                },
            };
            keys.push(value.to_string());
        }
        let Some(latency) = ctx.row.f64_optional(r.col_optional("latency"))? else {
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

/// Standard leaf from a pre-resolved power value.
fn classic_leaf_with(latency: f64, power: f64) -> ViewNode {
    ViewNode::Leaf(vec![
        ("latency", ViewValue::F64(latency)),
        ("power", ViewValue::F64(power)),
        ("energy", ViewValue::F64(power * latency)),
    ])
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

/// The engine-side table view: `attribute` is the PerfDatabase attribute name
/// the retired Python loader used to fill (`"_gemm_data"`, ...). Returns
/// `Ok(None)` exactly when that loader returned `None` (every source path
/// missing), and the loader-shaped nested JSON otherwise. The three DSV4
/// sparse sub-tables are addressed as
/// `"_dsv4_sparse_kernel_data.<paged_mqa_logits|hca_attn|csa_attn>"`.
pub fn table_view_json(tables: &PerfTables, attribute: &str) -> Result<Option<String>, AicError> {
    let src = |basename: &str| resolve_op_sources(&tables.perf_db_sources, basename, &tables.data_root);
    let comm_src = |root: Option<&std::path::Path>, basename: &str| -> Vec<PerfSource> {
        match root {
            Some(dir) => vec![PerfSource(dir.join(basename), None)],
            None => Vec::new(),
        }
    };
    let node = match attribute {
        "_gemm_data" => view_gemm(&src("gemm_perf.parquet"))?,
        "_compute_scale_data" => view_gemm_scale(&src("computescale_perf.parquet"))?,
        "_scale_matrix_data" => view_gemm_scale(&src("scale_matrix_perf.parquet"))?,
        "_context_attention_data" => view_context_attention(&src("context_attention_perf.parquet"))?,
        "_generation_attention_data" => view_generation_attention(&src("generation_attention_perf.parquet"))?,
        "_encoder_attention_data" => view_encoder_attention(&src("encoder_attention_perf.parquet"))?,
        "_context_mla_data" => view_context_mla(&src("context_mla_perf.parquet"))?,
        "_generation_mla_data" => view_generation_mla(&src("generation_mla_perf.parquet"))?,
        "_mla_bmm_data" => view_mla_bmm(&src("mla_bmm_perf.parquet"))?,
        "_context_mla_module_data" => view_context_mla_module(&src("mla_context_module_perf.parquet"))?,
        "_generation_mla_module_data" => view_generation_mla_module(&src("mla_generation_module_perf.parquet"))?,
        "_wideep_context_mla_data" => view_wideep_context_mla(&src("wideep_context_mla_perf.parquet"))?,
        "_wideep_generation_mla_data" => view_wideep_generation_mla(&src("wideep_generation_mla_perf.parquet"))?,
        "_moe_data" => view_moe(&src("moe_perf.parquet"))?.map(|(default, _)| default),
        "_moe_low_latency_data" => view_moe(&src("moe_perf.parquet"))?.map(|(_, low_latency)| low_latency),
        "_wideep_context_moe_data" => view_wideep_moe(&src("wideep_context_moe_perf.parquet"))?,
        "_wideep_generation_moe_data" => view_wideep_moe(&src("wideep_generation_moe_perf.parquet"))?,
        "_wideep_deepep_normal_data" => view_wideep_deepep_normal(&src("wideep_deepep_normal_perf.parquet"))?,
        "_wideep_deepep_ll_data" => view_wideep_deepep_ll(&src("wideep_deepep_ll_perf.parquet"))?,
        "_wideep_moe_compute_data" => view_wideep_moe_compute(&src("wideep_moe_perf.parquet"))?,
        "_trtllm_alltoall_data" => view_trtllm_alltoall(&src("trtllm_alltoall_perf.parquet"))?,
        "_moe_a2a_data" => view_moe_a2a(
            &src("moe_a2a_perf.parquet"),
            &src("wideep_deepep_normal_perf.parquet"),
            &src("wideep_deepep_ll_perf.parquet"),
            &src("trtllm_alltoall_perf.parquet"),
        )?,
        "_moe_ep_data" => view_moe_expert_compute(
            &src("moe_expert_compute_perf.parquet"),
            &src("wideep_context_moe_perf.parquet"),
            &src("wideep_generation_moe_perf.parquet"),
            &src("wideep_moe_perf.parquet"),
        )?,
        "_custom_allreduce_data" => view_custom_allreduce(&src("custom_allreduce_perf.parquet"))?,
        "_nccl_data" => view_nccl(&comm_src(tables.communication.nccl_root(), "nccl_perf.parquet"))?,
        "_oneccl_data" => view_nccl(&comm_src(tables.communication.oneccl_root(), "oneccl_perf.parquet"))?,
        "_context_dsa_module_data" => view_context_dsa_module(&src("dsa_context_module_perf.parquet"), false)?,
        "_context_dsa_module_skip_data" => view_context_dsa_module(&src("dsa_context_module_perf.parquet"), true)?,
        "_generation_dsa_module_data" => {
            view_generation_dsa_module(&src("dsa_generation_module_perf.parquet"), false)?
        }
        "_generation_dsa_module_skip_data" => {
            view_generation_dsa_module(&src("dsa_generation_module_perf.parquet"), true)?
        }
        "_mhc_module_data" => view_mhc_module(&src("mhc_module_perf.parquet"))?,
        "_context_deepseek_v4_attention_module_data" => merge_dsv4_split(vec![
            view_context_dsv4_kind_module(&src("dsv4_csa_context_module_perf.parquet"))?,
            view_context_dsv4_kind_module(&src("dsv4_hca_context_module_perf.parquet"))?,
        ]),
        "_generation_deepseek_v4_attention_module_data" => merge_dsv4_split(vec![
            view_generation_dsv4_kind_module(&src("dsv4_csa_generation_module_perf.parquet"))?,
            view_generation_dsv4_kind_module(&src("dsv4_hca_generation_module_perf.parquet"))?,
        ]),
        "_dsv4_sparse_kernel_data.paged_mqa_logits" => {
            view_dsv4_sparse_kernel(&src("dsv4_paged_mqa_logits_module_perf.parquet"))?
        }
        "_dsv4_sparse_kernel_data.hca_attn" => view_dsv4_sparse_kernel(&src("dsv4_hca_attn_module_perf.parquet"))?,
        "_dsv4_sparse_kernel_data.csa_attn" => view_dsv4_sparse_kernel(&src("dsv4_csa_attn_module_perf.parquet"))?,
        "_dsv4_megamoe_module_data" => view_dsv4_megamoe_module(&src("dsv4_megamoe_module_perf.parquet"))?,
        "_mamba2_data" => view_mamba2(&src("mamba2_perf.parquet"))?,
        "_gdn_data" => view_gdn(&src("gdn_perf.parquet"))?,
        "_kda_data" => view_kda(&src("kda_perf.parquet"))?,
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
