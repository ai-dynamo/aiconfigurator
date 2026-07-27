// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! State-space layer perf tables: Mamba2 and Gated Delta Network (GDN).
//!
//! Used by hybrid models such as Nemotron-H. Both CSVs share a similar
//! shape: a `phase` discriminator (`context` / `generation`), a model-name
//! key, and several layer-specific dimension columns.
//!
//! Resolution mirrors Python v2 (`operations/mamba.py` + the perf_interp
//! engine): after shape-key selection, context queries are a 2-axis Grid
//! RAW engine query over `[batch][seq_len]`, generation queries a 1-axis
//! Grid RAW query over `[batch]` (generation rows are collected at a single
//! seq, and the query's seq only feeds SOL — where generation formulas
//! ignore it). The operator layer owns the SOL closure and the SOL
//! degradation contract: any `PerfDatabase` error here makes the op fall
//! back to its analytic `sol_latency_ms` (`source="sol"`).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use crate::common::error::AicError;
use crate::config::{PerfDbSources, PerfSource};
use super::perf_interp::{self, Node, OpInterpConfig};
use super::{kernel_source_ok, resolve_op_sources};
use crate::perf_database::parquet_loader::PerfReader;

pub struct StateSpaceTable {
    data_root: PathBuf,
    /// Ordered, priority-sorted sources for each state-space perf file
    /// (shared-layer aware; see [`PerfSource`]). Single-primary, no-filter by
    /// default (`StateSpaceTable::new`).
    mamba2_sources: Vec<PerfSource>,
    gdn_sources: Vec<PerfSource>,
    vllm_024_gdn_aliases: bool,
    mamba2: OnceLock<Result<Mamba2Grids, AicError>>,
    gdn: OnceLock<Result<GdnGrids, AicError>>,
}

/// Per shape-key engine table. Context keys hold a 2-level `[batch][seq]`
/// node; generation keys hold a 1-level `[batch]` node (Python v2 keys
/// generation leaves by batch only).
struct Mamba2Grids {
    by_keys: BTreeMap<Mamba2Key, Node>,
}

struct GdnGrids {
    by_keys: BTreeMap<GdnKey, Node>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Mamba2Key {
    /// Kernel routine name (e.g. `causal_conv1d_fn` /
    /// `causal_conv1d_update`); discriminates between context and
    /// generation kernels that share the rest of the shape.
    kernel_source: String,
    phase: String,
    d_model: u32,
    d_state: u32,
    d_conv: u32,
    nheads: u32,
    head_dim: u32,
    n_groups: u32,
    chunk_size: u32,
    // Note: Python keys by SHAPE tuple, not by `model_name`. The CSV's
    // `model_name` column is metadata identifying which model the row
    // was collected against; the lookup itself is shape-based, so a
    // matching shape is reused across model names. We mirror that.
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct GdnKey {
    kernel_source: String,
    phase: String,
    d_model: u32,
    d_conv: u32,
    num_k_heads: u32,
    head_k_dim: u32,
    num_v_heads: u32,
    head_v_dim: u32,
    // See `Mamba2Key`: shape is the key, `model_name` is metadata.
}

impl StateSpaceTable {
    /// Construct an empty table for the given data directory. No I/O. Each
    /// perf file is sourced solely from `data_root/<basename>` with no
    /// `kernel_source` filter (pre-shared-layer behaviour).
    pub fn new(data_root: PathBuf, backend: &str, version: &str) -> Self {
        Self::with_sources(data_root, backend, version, &PerfDbSources::default())
    }

    /// Construct with shared-layer (sibling/cross-version) sources resolved from
    /// `perf_db_sources` (Python-supplied). Each state-space file falls back to
    /// its primary `data_root/<basename>` when absent from the map. No I/O.
    pub fn with_sources(
        data_root: PathBuf,
        backend: &str,
        version: &str,
        perf_db_sources: &PerfDbSources,
    ) -> Self {
        let mamba2_sources =
            resolve_op_sources(perf_db_sources, "mamba2_perf.parquet", &data_root);
        let gdn_sources = resolve_op_sources(perf_db_sources, "gdn_perf.parquet", &data_root);
        Self {
            data_root,
            mamba2_sources,
            gdn_sources,
            vllm_024_gdn_aliases: backend == "vllm" && version == "0.24.0",
            mamba2: OnceLock::new(),
            gdn: OnceLock::new(),
        }
    }

    /// Mamba2 latency for a layer instance, resolved on the perf_interp
    /// engine (context: 2-axis `(batch, seq_len)` Grid RAW; see module
    /// doc). `sol` is the operator's analytic SOL in `(batch, seq)` order
    /// — it anchors util-hold beyond the collected range.
    #[allow(clippy::too_many_arguments)]
    pub fn query_mamba2(
        &self,
        kernel_source: &str,
        phase: &str,
        batch_size: u32,
        seq_len: u32,
        d_model: u32,
        d_state: u32,
        d_conv: u32,
        nheads: u32,
        head_dim: u32,
        n_groups: u32,
        chunk_size: u32,
        sol: &dyn Fn(f64, f64) -> f64,
    ) -> Result<f64, AicError> {
        // Mirror Python v2's `load_mamba2_data` defaultdict bug (still
        // present today, verified 2026-07-09): the row-population pattern
        // `try { data[ks][ph][mk][bs] } except KeyError: ... = entry` never
        // reaches the `except` branch for generation rows because the
        // fourth level is a `defaultdict(dict)` that lazily materialises an
        // empty dict on `[bs]` access (no KeyError). Generation leaves end
        // up as empty `{}`; `_query_mamba2_table`'s generation branch then
        // normalizes them to an empty curve, the perf_interp engine raises,
        // and the op falls back to SOL (`db.query_mamba2(generation)`
        // returns `source="sol"` for every query). The Rust parquet loader
        // populates the rows correctly, so returning silicon here would
        // give a numerically different (and arguably "more correct")
        // answer — but for apple-to-apple parity we mirror Python by
        // returning a PerfDatabase error so the operator-layer SOL branch
        // fires. GDN's loader is fine (uses explicit `in` checks), so this
        // workaround is Mamba2-generation-only.
        if phase == "generation" {
            return Err(AicError::PerfDatabase(format!(
                "Mamba2 generation data intentionally not used (matches Python v2 \
                 `load_mamba2_data` defaultdict bug in operations/mamba.py — generation \
                 leaves load empty, so every generation query degrades to SOL); \
                 operator must fall to SOL. ks={kernel_source}, d_model={d_model}"
            )));
        }
        let grids = self.load_mamba2()?;
        let key = Mamba2Key {
            kernel_source: kernel_source.to_string(),
            phase: phase.to_string(),
            d_model,
            d_state,
            d_conv,
            nheads,
            head_dim,
            n_groups,
            chunk_size,
        };
        // Mirror Python `_query_mamba2_table`: on exact-shape miss, fall back
        // to the first table entry sharing the same `d_model` (insertion order
        // in Python; sorted order here — which agrees whenever the per-d_model
        // bucket has a single entry, as in all current matrices). If no entry
        // shares d_model, surface as `PerfDatabase` so the operator layer's
        // SOL fallback applies.
        //
        // Only the context phase reaches this point — generation queries are
        // short-circuited above to match Python's degenerate behaviour.
        let node = match grids.by_keys.get(&key) {
            Some(node) => node,
            None => grids
                .by_keys
                .iter()
                .find(|(k, _)| {
                    k.kernel_source == key.kernel_source
                        && k.phase == key.phase
                        && k.d_model == key.d_model
                })
                .map(|(_, node)| node)
                .ok_or_else(|| missing("Mamba2", &self.data_root, format!("{key:?}")))?,
        };
        engine_query(node, phase, batch_size, seq_len, sol)
    }

    /// GDN latency for a layer instance. Same engine resolution as Mamba2
    /// (and, unlike Mamba2, generation queries really resolve: 1-axis
    /// `(batch,)` Grid RAW over the generation curve, per Python v2).
    #[allow(clippy::too_many_arguments)]
    pub fn query_gdn(
        &self,
        kernel_source: &str,
        phase: &str,
        batch_size: u32,
        seq_len: u32,
        d_model: u32,
        d_conv: u32,
        num_k_heads: u32,
        head_k_dim: u32,
        num_v_heads: u32,
        head_v_dim: u32,
        sol: &dyn Fn(f64, f64) -> f64,
    ) -> Result<f64, AicError> {
        let grids = self.load_gdn()?;
        let key = GdnKey {
            kernel_source: kernel_source.to_string(),
            phase: phase.to_string(),
            d_model,
            d_conv,
            num_k_heads,
            head_k_dim,
            num_v_heads,
            head_v_dim,
        };
        // Mirror Python `_query_gdn_table`: on exact-shape miss, fall back to
        // any same-d_model entry, breaking ties by minimum `|num_v_heads -
        // query.num_v_heads|`. (Mamba2 uses "first by d_model"; GDN uses
        // "nearest by num_v_heads" — keep them distinct.) Surface as
        // `PerfDatabase` if no d_model match exists.
        let node = match grids.by_keys.get(&key) {
            Some(node) => node,
            None => {
                // vLLM 0.24 persists the selected physical recurrence
                // implementation, while model operators retain stable logical
                // kernel names. Resolve a physical source only for an exact
                // model shape. Exact logical data always wins, and ambiguous
                // physical data fails closed.
                let aliases: &[&str] = if self.vllm_024_gdn_aliases {
                    match (key.kernel_source.as_str(), key.phase.as_str()) {
                        ("chunk_gated_delta_rule", "context") => &[
                            "chunk_gated_delta_rule_flashinfer",
                            "chunk_gated_delta_rule_triton",
                            "chunk_gated_delta_rule_cutedsl",
                        ],
                        ("fused_sigmoid_gating_delta_rule_update", "generation") => {
                            &["fused_recurrent_gated_delta_rule_packed_decode"]
                        }
                        _ => &[],
                    }
                } else {
                    &[]
                };
                let alias_matches: Vec<_> = aliases
                    .iter()
                    .filter_map(|alias| {
                        let mut alias_key = key.clone();
                        alias_key.kernel_source = (*alias).to_string();
                        grids.by_keys.get_key_value(&alias_key)
                    })
                    .collect();
                if alias_matches.len() > 1 {
                    let sources: Vec<_> = alias_matches
                        .iter()
                        .map(|(alias_key, _)| alias_key.kernel_source.as_str())
                        .collect();
                    return Err(AicError::PerfDatabase(format!(
                        "ambiguous vLLM 0.24.0 GDN physical kernels for {key:?}: {}",
                        sources.join(", ")
                    )));
                }
                if let Some((_, node)) = alias_matches.first() {
                    return engine_query(node, phase, batch_size, seq_len, sol);
                }

                let nearest = grids
                    .by_keys
                    .iter()
                    .filter(|(k, _)| {
                        k.kernel_source == key.kernel_source
                            && k.phase == key.phase
                            && k.d_model == key.d_model
                    })
                    .min_by_key(|(k, _)| (k.num_v_heads as i64 - key.num_v_heads as i64).abs());
                match nearest {
                    Some((_, node)) => node,
                    None => return Err(missing("GDN", &self.data_root, format!("{key:?}"))),
                }
            }
        };
        engine_query(node, phase, batch_size, seq_len, sol)
    }

    fn load_mamba2(&self) -> Result<&Mamba2Grids, AicError> {
        let cell = self
            .mamba2
            .get_or_init(|| load_mamba2_parquet(&self.mamba2_sources));
        cell.as_ref().map_err(clone_err)
    }

    fn load_gdn(&self) -> Result<&GdnGrids, AicError> {
        let cell = self
            .gdn
            .get_or_init(|| load_gdn_parquet(&self.gdn_sources));
        cell.as_ref().map_err(clone_err)
    }
}

/// One perf_interp engine query per phase, mirroring Python v2's
/// `_query_mamba2_table` / `_query_gdn_table`:
///
/// - context: `axes=("batch", "seq_len")`, Grid RAW, coords `(b, s)` — the
///   same axis order as the Python record.
/// - generation: `axes=("batch",)`, Grid RAW over the per-batch curve. The
///   query's `seq_len` is forwarded to `sol` only (Python passes the op's
///   `seq_len=None` there; generation SOL formulas ignore it either way).
fn engine_query(
    node: &Node,
    phase: &str,
    batch_size: u32,
    seq_len: u32,
    sol: &dyn Fn(f64, f64) -> f64,
) -> Result<f64, AicError> {
    if phase == "generation" {
        let s = seq_len as f64;
        let sol1 = move |c: &[f64]| sol(c[0], s);
        let cfg = OpInterpConfig::grid(&["batch"], &sol1);
        perf_interp::query(&cfg, node, &[batch_size as f64])
    } else {
        // Python: `if seq_len is None or seq_len <= 0: return SOL` — surface
        // as a PerfDatabase error so the operator's SOL branch fires.
        if seq_len == 0 {
            return Err(AicError::PerfDatabase(
                "state-space context query needs seq_len > 0".to_string(),
            ));
        }
        let sol2 = move |c: &[f64]| sol(c[0], c[1]);
        let cfg = OpInterpConfig::grid(&["batch", "seq_len"], &sol2);
        perf_interp::query(&cfg, node, &[batch_size as f64, seq_len as f64])
    }
}

/// First-wins leaf insert (Python loaders skip rows whose coordinate is
/// already populated; `Node::insert` would overwrite).
fn insert_first_wins(root: &mut Node, path: &[u32], value: f64) {
    let Node::Branch(map) = root else {
        return; // malformed nesting; keep the earlier row
    };
    if path.len() == 1 {
        map.entry(path[0]).or_insert(Node::Leaf(value));
    } else {
        let child = map.entry(path[0]).or_insert_with(Node::branch);
        insert_first_wins(child, &path[1..], value);
    }
}

/// Load the Mamba2 table from an ordered, priority-sorted source list. Sources
/// are read in order; the first source containing a coordinate wins
/// (`insert_first_wins`), mirroring Python's `_read_filtered_rows` concatenation
/// + `load_mamba2_data` skip-on-key-conflict. Missing files are skipped (a
/// sibling declared in the manifest need not exist for every system); an error
/// is returned only when no source yields rows.
fn load_mamba2_parquet(sources: &[PerfSource]) -> Result<Mamba2Grids, AicError> {
    let mut by_keys: BTreeMap<Mamba2Key, Node> = BTreeMap::new();
    let mut any_source = false;
    for source in sources {
        let path = source.path();
        if !path.exists() {
            continue;
        }
        any_source = true;
        let reader = PerfReader::open(path)?;
        let kernel_source_col = reader.col("kernel_source")?;
        let phase_col = reader.col("phase")?;
        let batch_size_col = reader.col("batch_size")?;
        let seq_len_col = reader.col("seq_len")?;
        let d_model_col = reader.col("d_model")?;
        let d_state_col = reader.col("d_state")?;
        let d_conv_col = reader.col("d_conv")?;
        let nheads_col = reader.col("nheads")?;
        let head_dim_col = reader.col("head_dim")?;
        let n_groups_col = reader.col("n_groups")?;
        let chunk_size_col = reader.col("chunk_size")?;
        let latency_col = reader.col("latency")?;
        let ks_col = reader.col_optional("kernel_source");
        for row in reader.rows()? {
            let row = row?;
            if !kernel_source_ok(source.kernel_sources(), ks_col, &row)? {
                continue;
            }
            let phase = row.str_owned(phase_col)?;
            let key = Mamba2Key {
                kernel_source: row.str_owned(kernel_source_col)?,
                phase: phase.clone(),
                d_model: row.u32(d_model_col)?,
                d_state: row.u32(d_state_col)?,
                d_conv: row.u32(d_conv_col)?,
                nheads: row.u32(nheads_col)?,
                head_dim: row.u32(head_dim_col)?,
                n_groups: row.u32(n_groups_col)?,
                chunk_size: row.u32(chunk_size_col)?,
            };
            // First-wins parity with Python `load_mamba2_data`, extended across
            // shared-layer sources (earlier source wins). Generation rows are
            // keyed by batch only (Python drops seq for generation). Note the
            // stored generation leaves are never read: `query_mamba2` mirrors
            // Python's empty-generation-leaves bug by erroring first.
            let node = by_keys.entry(key).or_insert_with(Node::branch);
            let batch = row.u32(batch_size_col)?;
            let latency = row.f64(latency_col)?;
            if phase == "generation" {
                insert_first_wins(node, &[batch], latency);
            } else {
                insert_first_wins(node, &[batch, row.u32(seq_len_col)?], latency);
            }
        }
    }
    if !any_source || by_keys.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "no Mamba2 rows loaded from {} source(s) (first: {})",
            sources.len(),
            sources.first().map(|s| s.path().display().to_string()).unwrap_or_default()
        )));
    }
    Ok(Mamba2Grids { by_keys })
}

/// Load the GDN table from an ordered, priority-sorted source list. Same
/// first-wins-across-sources + missing-file-skip semantics as
/// [`load_mamba2_parquet`].
/// The GDN decode-recurrence kernel name drifted across sglang releases
/// (0.5.10: `fused_recurrent_gated_delta_rule`; 0.5.14 records the executed
/// `fused_recurrent_gated_delta_rule_packed_decode`). Consumers query one
/// canonical modeling identity; normalize the LOOKUP key here (mirrors
/// Python `_GDN_DECODE_RECURRENCE_ALIASES`) — the parquet keeps the
/// executed-kernel truth.
fn normalize_gdn_kernel_source(kernel_source: String) -> String {
    match kernel_source.as_str() {
        "fused_recurrent_gated_delta_rule" | "fused_recurrent_gated_delta_rule_packed_decode" => {
            "fused_sigmoid_gating_delta_rule_update".to_string()
        }
        _ => kernel_source,
    }
}

fn load_gdn_parquet(sources: &[PerfSource]) -> Result<GdnGrids, AicError> {
    let mut by_keys: BTreeMap<GdnKey, Node> = BTreeMap::new();
    let mut any_source = false;
    for source in sources {
        let path = source.path();
        if !path.exists() {
            continue;
        }
        any_source = true;
        let reader = PerfReader::open(path)?;
        let kernel_source_col = reader.col("kernel_source")?;
        let phase_col = reader.col("phase")?;
        let batch_size_col = reader.col("batch_size")?;
        let seq_len_col = reader.col("seq_len")?;
        let d_model_col = reader.col("d_model")?;
        let d_conv_col = reader.col("d_conv")?;
        let num_k_heads_col = reader.col("num_k_heads")?;
        let head_k_dim_col = reader.col("head_k_dim")?;
        let num_v_heads_col = reader.col("num_v_heads")?;
        let head_v_dim_col = reader.col("head_v_dim")?;
        let latency_col = reader.col("latency")?;
        let ks_col = reader.col_optional("kernel_source");
        for row in reader.rows()? {
            let row = row?;
            if !kernel_source_ok(source.kernel_sources(), ks_col, &row)? {
                continue;
            }
            let phase = row.str_owned(phase_col)?;
            let key = GdnKey {
                kernel_source: normalize_gdn_kernel_source(row.str_owned(kernel_source_col)?),
                phase: phase.clone(),
                d_model: row.u32(d_model_col)?,
                d_conv: row.u32(d_conv_col)?,
                num_k_heads: row.u32(num_k_heads_col)?,
                head_k_dim: row.u32(head_k_dim_col)?,
                num_v_heads: row.u32(num_v_heads_col)?,
                head_v_dim: row.u32(head_v_dim_col)?,
            };
            // First-wins parity with Python `load_gdn_data`, extended across
            // shared-layer sources (earlier source wins): context leaves at
            // `[batch][seq]`, generation leaves at `[batch]`.
            let node = by_keys.entry(key).or_insert_with(Node::branch);
            let batch = row.u32(batch_size_col)?;
            let latency = row.f64(latency_col)?;
            if phase == "generation" {
                insert_first_wins(node, &[batch], latency);
            } else {
                insert_first_wins(node, &[batch, row.u32(seq_len_col)?], latency);
            }
        }
    }
    if !any_source || by_keys.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "no GDN rows loaded from {} source(s) (first: {})",
            sources.len(),
            sources.first().map(|s| s.path().display().to_string()).unwrap_or_default()
        )));
    }
    Ok(GdnGrids { by_keys })
}

fn missing(table: &str, data_root: &Path, descriptor: String) -> AicError {
    AicError::PerfDatabase(format!("{table} data missing for {descriptor} at {}", data_root.display()))
}

fn clone_err(err: &AicError) -> AicError {
    AicError::PerfDatabase(err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::system_spec::SystemSpec;

    fn data_root(rel: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../src/aiconfigurator_core/systems/data")
            .join(rel)
    }

    fn h100_sxm_mem_bw() -> f64 {
        let yaml = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../src/aiconfigurator_core/systems/h100_sxm.yaml");
        SystemSpec::load(&yaml)
            .expect("h100_sxm.yaml must parse")
            .gpu
            .mem_bw
    }

    fn dummy_sol(_b: f64, _s: f64) -> f64 {
        1.0
    }

    /// In-memory GDN table over one fixed model shape (d_model=5120,
    /// heads 16/128, v-dim 128), varying only `(kernel_source, phase,
    /// num_v_heads)`. Context rows land at `[batch=1][seq=1024]`, generation
    /// rows at `[batch=1]`, matching the loader's leaf layout — queries below
    /// hit those coordinates exactly, so engine RAW returns the stored value.
    fn in_memory_gdn_table(
        backend: &str,
        version: &str,
        rows: &[(&str, &str, u32, f64)],
    ) -> StateSpaceTable {
        let mut by_keys: BTreeMap<GdnKey, Node> = BTreeMap::new();
        for &(kernel_source, phase, num_v_heads, latency) in rows {
            let key = GdnKey {
                kernel_source: kernel_source.to_string(),
                phase: phase.to_string(),
                d_model: 5120,
                d_conv: 4,
                num_k_heads: 16,
                head_k_dim: 128,
                num_v_heads,
                head_v_dim: 128,
            };
            let node = by_keys.entry(key).or_insert_with(Node::branch);
            if phase == "generation" {
                insert_first_wins(node, &[1], latency);
            } else {
                insert_first_wins(node, &[1, 1024], latency);
            }
        }
        let table = StateSpaceTable::new(PathBuf::from("test-data"), backend, version);
        assert!(table.gdn.set(Ok(GdnGrids { by_keys })).is_ok());
        table
    }

    fn query_gdn_test_shape(
        table: &StateSpaceTable,
        kernel_source: &str,
        phase: &str,
        num_v_heads: u32,
    ) -> Result<f64, AicError> {
        table.query_gdn(
            kernel_source,
            phase,
            1,
            1024,
            5120,
            4,
            16,
            128,
            num_v_heads,
            128,
            &dummy_sol,
        )
    }

    #[test]
    fn vllm_024_gdn_resolves_context_and_generation_physical_aliases() {
        for source in [
            "chunk_gated_delta_rule_flashinfer",
            "chunk_gated_delta_rule_triton",
            "chunk_gated_delta_rule_cutedsl",
        ] {
            let table = in_memory_gdn_table("vllm", "0.24.0", &[(source, "context", 48, 2.0)]);
            assert_eq!(
                query_gdn_test_shape(&table, "chunk_gated_delta_rule", "context", 48).unwrap(),
                2.0
            );
        }

        let table = in_memory_gdn_table(
            "vllm",
            "0.24.0",
            &[(
                "fused_recurrent_gated_delta_rule_packed_decode",
                "generation",
                48,
                3.0,
            )],
        );
        assert_eq!(
            query_gdn_test_shape(
                &table,
                "fused_sigmoid_gating_delta_rule_update",
                "generation",
                48,
            )
            .unwrap(),
            3.0
        );
    }

    #[test]
    fn vllm_024_gdn_exact_logical_key_wins_over_alias() {
        let table = in_memory_gdn_table(
            "vllm",
            "0.24.0",
            &[
                ("chunk_gated_delta_rule", "context", 48, 1.0),
                ("chunk_gated_delta_rule_flashinfer", "context", 48, 2.0),
            ],
        );
        assert_eq!(
            query_gdn_test_shape(&table, "chunk_gated_delta_rule", "context", 48).unwrap(),
            1.0
        );
    }

    #[test]
    fn gdn_physical_aliases_are_vllm_024_only() {
        for (backend, version) in [("vllm", "0.23.0"), ("sglang", "0.24.0")] {
            let table = in_memory_gdn_table(
                backend,
                version,
                &[("chunk_gated_delta_rule_flashinfer", "context", 48, 2.0)],
            );
            assert!(query_gdn_test_shape(&table, "chunk_gated_delta_rule", "context", 48).is_err());
        }
    }

    #[test]
    fn vllm_024_gdn_ambiguous_exact_aliases_error() {
        let table = in_memory_gdn_table(
            "vllm",
            "0.24.0",
            &[
                ("chunk_gated_delta_rule_flashinfer", "context", 48, 2.0),
                ("chunk_gated_delta_rule_triton", "context", 48, 3.0),
            ],
        );
        match query_gdn_test_shape(&table, "chunk_gated_delta_rule", "context", 48) {
            Err(AicError::PerfDatabase(message)) => {
                assert!(message.contains("ambiguous vLLM 0.24.0 GDN physical kernels"));
                assert!(message.contains("chunk_gated_delta_rule_flashinfer"));
                assert!(message.contains("chunk_gated_delta_rule_triton"));
            }
            other => panic!("expected an explicit ambiguity error, got {other:?}"),
        }
    }

    #[test]
    fn vllm_024_gdn_alias_has_no_nearest_shape_fallback() {
        let table = in_memory_gdn_table(
            "vllm",
            "0.24.0",
            &[("chunk_gated_delta_rule_flashinfer", "context", 32, 2.0)],
        );
        assert!(query_gdn_test_shape(&table, "chunk_gated_delta_rule", "context", 48).is_err());
    }

    #[test]
    fn vllm_024_gdn_preserves_logical_source_nearest_fallback() {
        let table = in_memory_gdn_table(
            "vllm",
            "0.24.0",
            &[
                ("chunk_gated_delta_rule_flashinfer", "context", 32, 2.0),
                ("chunk_gated_delta_rule", "context", 16, 4.0),
                ("chunk_gated_delta_rule", "context", 64, 5.0),
            ],
        );
        assert_eq!(
            query_gdn_test_shape(&table, "chunk_gated_delta_rule", "context", 48).unwrap(),
            5.0
        );
    }

    #[test]
    fn state_space_loaders_smoke() {
        // GDN data exists on vLLM b200 (Nemotron-H slice); Mamba2 may not.
        let root = data_root("b200_sxm/vllm/0.19.0");
        let table = StateSpaceTable::new(root, "vllm", "0.19.0");
        // Just verify loader doesn't panic on missing-key path; we don't
        // assert a specific value here.
        let _ = table
            .query_gdn(
                "causal_conv1d_fn",
                "prefill",
                1,
                1024,
                4096,
                4,
                16,
                128,
                32,
                128,
                &dummy_sol,
            )
            .err();
    }

    #[test]
    fn gdn_table_finds_qwen35_27b_conv1d_update() {
        let root = data_root("b200_sxm/vllm/0.19.0");
        let table = StateSpaceTable::new(root, "vllm", "0.19.0");
        let r = table.query_gdn(
            "causal_conv1d_update",
            "generation",
            1,
            1,
            5120,
            4,
            16,
            128,
            48,
            128,
            &dummy_sol,
        );
        eprintln!("query: {r:?}");
        assert!(r.is_ok(), "expected silicon lookup to succeed: {r:?}");
        let latency = r.unwrap();
        assert!(latency > 0.0, "non-zero latency: {latency}");
        eprintln!("latency: {latency}");
    }

    /// Values generated from Python v2 on the same tables
    /// (`db.query_gdn` / `db.query_mamba2` via `get_database(...)`, default
    /// SILICON mode; the shared layer was verified to contribute no rows to
    /// these slices, so both engines see identical data). Covers, per table:
    /// exact hit, in-range interpolation, and beyond-range util-hold (which
    /// exercises the SOL closure). SOL closures replicate the operators'
    /// `sol_latency_ms` for the queried kernels. Latencies compared at the
    /// table/db layer — Python applies no extra factors there (the op-layer
    /// `scale_factor` sits above `db.query_*`).
    // NOTE(shared-layer merge): oracle generated pre-shared-layer; regenerate
    // if this fails (the multi-source loaders may now merge sibling/shared
    // rows into these curves).
    #[test]
    fn state_space_queries_match_python_v2_engine() {
        let bw = h100_sxm_mem_bw();

        // GDN causal_conv1d kernels: read = x*conv_channels*(d_conv+1)*2,
        // write = x*conv_channels*2; x = b*s (context) or b (generation).
        let conv_channels = (16 * 128 + 32 * 128) as f64;
        let gdn_conv_sol = move |x: f64| {
            (x * conv_channels * (4.0 + 1.0) * 2.0 + x * conv_channels * 2.0) / bw * 1000.0
        };
        let gdn_ctx_sol = move |b: f64, s: f64| gdn_conv_sol(b * s);
        let gdn_gen_sol = move |b: f64, _s: f64| gdn_conv_sol(b);

        let gdn = StateSpaceTable::new(data_root("h100_sxm/sglang/0.5.10"), "sglang", "0.5.10");
        // (batch, seq, expected): exact / seq-interp / batch-interp / beyond-seq.
        let ctx_cases: &[(u32, u32, f64)] = &[
            (8, 1024, 0.03154560029506683),
            (8, 1536, 0.04624959975481033),
            (3, 1024, 0.011241600289940833),
            (8, 65536, 1.7870464324951172),
        ];
        for &(b, s, expected) in ctx_cases {
            let got = gdn
                .query_gdn(
                    "causal_conv1d_fn",
                    "context",
                    b,
                    s,
                    2048,
                    4,
                    16,
                    128,
                    32,
                    128,
                    &gdn_ctx_sol,
                )
                .unwrap();
            assert!(
                ((got - expected) / expected).abs() < 1e-9,
                "gdn ctx (b={b}, s={s}): rust {got} vs python {expected}"
            );
        }
        // Generation: exact / interp / beyond-range on the 1-axis batch curve.
        let gen_cases: &[(u32, f64)] = &[
            (8, 0.0052767999470233916),
            (48, 0.005246399901807308),
            (2048, 0.04643200039863586),
        ];
        for &(b, expected) in gen_cases {
            let got = gdn
                .query_gdn(
                    "causal_conv1d_update",
                    "generation",
                    b,
                    1,
                    2048,
                    4,
                    16,
                    128,
                    32,
                    128,
                    &gdn_gen_sol,
                )
                .unwrap();
            assert!(
                ((got - expected) / expected).abs() < 1e-9,
                "gdn gen (b={b}): rust {got} vs python {expected}"
            );
        }

        // Mamba2 causal_conv1d: conv_dim = d_inner + 2*n_groups*d_state.
        let conv_dim = (64 * 64 + 2 * 8 * 128) as f64;
        let m2_ctx_sol = move |b: f64, s: f64| {
            let x = b * s;
            (x * conv_dim * (4.0 + 1.0) * 2.0 + x * conv_dim * 2.0) / bw * 1000.0
        };
        let mamba2 = StateSpaceTable::new(data_root("h100_sxm/trtllm/1.3.0rc10"), "trtllm", "1.3.0rc10");
        let m2_cases: &[(u32, u32, f64)] = &[
            (4, 1024, 0.058057600259780885),
            (4, 1536, 0.07725920081138611),
            (4, 65536, 2.520614433288574),
        ];
        for &(b, s, expected) in m2_cases {
            let got = mamba2
                .query_mamba2(
                    "causal_conv1d_fn",
                    "context",
                    b,
                    s,
                    2688,
                    128,
                    4,
                    64,
                    64,
                    8,
                    128,
                    &m2_ctx_sol,
                )
                .unwrap();
            assert!(
                ((got - expected) / expected).abs() < 1e-9,
                "mamba2 ctx (b={b}, s={s}): rust {got} vs python {expected}"
            );
        }
        // Mamba2 generation stays SOL-degraded (Python v2 loader bug parity):
        // the table layer must error so the operator falls to plain SOL.
        assert!(mamba2
            .query_mamba2(
                "causal_conv1d_update",
                "generation",
                4,
                1,
                2688,
                128,
                4,
                64,
                64,
                8,
                128,
                &dummy_sol,
            )
            .is_err());
    }
}
