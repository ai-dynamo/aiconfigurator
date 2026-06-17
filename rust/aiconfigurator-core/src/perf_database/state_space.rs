// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! State-space layer perf tables: Mamba2 and Gated Delta Network (GDN).
//!
//! Used by hybrid models such as Nemotron-H. Both CSVs share a similar
//! shape: a `phase` discriminator (`prefill` / `generation`), a model-name
//! key, and several layer-specific dimension columns. The numeric query
//! axis is `(seq_len, batch_size)` per (key, phase) slice.
//!
//! The operator layer owns the empirical/SOL wrappers; this layer just
//! stores the raw nested grid and provides 1-D-by-2-D lookup.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use crate::common::error::AicError;
use super::interpolation::{interp_1d, nearest_neighbors};
use crate::perf_database::parquet_loader::PerfReader;

pub struct StateSpaceTable {
    data_root: PathBuf,
    mamba2: OnceLock<Result<Mamba2Grids, AicError>>,
    gdn: OnceLock<Result<GdnGrids, AicError>>,
}

struct Mamba2Grids {
    by_keys: BTreeMap<Mamba2Key, BTreeMap<(u32, u32), f64>>,
}

struct GdnGrids {
    by_keys: BTreeMap<GdnKey, BTreeMap<(u32, u32), f64>>,
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
    pub fn new(data_root: PathBuf) -> Self {
        Self {
            data_root,
            mamba2: OnceLock::new(),
            gdn: OnceLock::new(),
        }
    }

    /// Mamba2 latency for a layer instance. 1-D along seq_len at the
    /// matching batch_size slice; if batch_size isn't exact, 1-D along
    /// batch within the matching seq_len slice. Falls back to nearest
    /// neighbour pair if neither is exact.
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
    ) -> Result<f64, AicError> {
        // Mirror Python `load_mamba2_data`'s defaultdict-without-KeyError
        // bug: for `phase == "generation"`, the row-population pattern
        // `try { data[ks][ph][mk][bs] } except KeyError: data[ks][ph][mk][bs]
        // = entry` never reaches the `except` branch because every level
        // is a `defaultdict` that lazily materialises an empty dict on
        // access (no KeyError raised). Python's generation Mamba2 leaves
        // therefore end up empty and every generation query silently falls
        // through to SOL. The Rust parquet loader populates the rows
        // correctly, so returning silicon here would give a numerically
        // different (and arguably "more correct") answer — but for
        // apple-to-apple parity we mirror Python by returning a
        // PerfDatabase error so the operator-layer SOL branch fires.
        // GDN's loader is fine (uses explicit `in` checks), so this
        // workaround is Mamba2-generation-only.
        if phase == "generation" {
            return Err(AicError::PerfDatabase(format!(
                "Mamba2 generation data intentionally not used (matches Python `load_mamba2_data` \
                 defaultdict bug at operations/mamba.py:719); operator must fall to SOL. \
                 ks={kernel_source}, d_model={d_model}"
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
        let by_bs_seq = match grids.by_keys.get(&key) {
            Some(table) => table,
            None => grids
                .by_keys
                .iter()
                .find(|(k, _)| {
                    k.kernel_source == key.kernel_source
                        && k.phase == key.phase
                        && k.d_model == key.d_model
                })
                .map(|(_, table)| table)
                .ok_or_else(|| missing("Mamba2", &self.data_root, format!("{key:?}")))?,
        };
        bs_seq_interp(by_bs_seq, batch_size, seq_len)
    }

    /// GDN latency for a layer instance. Same lookup semantics as Mamba2.
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
        let by_bs_seq = match grids.by_keys.get(&key) {
            Some(table) => table,
            None => {
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
                    Some((_, table)) => table,
                    None => return Err(missing("GDN", &self.data_root, format!("{key:?}"))),
                }
            }
        };
        bs_seq_interp(by_bs_seq, batch_size, seq_len)
    }

    fn load_mamba2(&self) -> Result<&Mamba2Grids, AicError> {
        let cell = self
            .mamba2
            .get_or_init(|| load_mamba2_parquet(&self.data_root.join("mamba2_perf.parquet")));
        cell.as_ref().map_err(clone_err)
    }

    fn load_gdn(&self) -> Result<&GdnGrids, AicError> {
        let cell = self
            .gdn
            .get_or_init(|| load_gdn_parquet(&self.data_root.join("gdn_perf.parquet")));
        cell.as_ref().map_err(clone_err)
    }
}

fn bs_seq_interp(
    by_bs_seq: &BTreeMap<(u32, u32), f64>,
    batch_size: u32,
    seq_len: u32,
) -> Result<f64, AicError> {
    if let Some(&latency) = by_bs_seq.get(&(batch_size, seq_len)) {
        return Ok(latency);
    }
    // Try 1-D along seq_len at the matching batch_size.
    let seqs_at_bs: Vec<u32> = by_bs_seq
        .keys()
        .filter_map(|&(bs, s)| if bs == batch_size { Some(s) } else { None })
        .collect();
    if seqs_at_bs.len() >= 2 {
        let (lo, hi) = nearest_neighbors(seq_len, &seqs_at_bs, false)?;
        return Ok(interp_1d(
            lo as f64,
            hi as f64,
            by_bs_seq[&(batch_size, lo)],
            by_bs_seq[&(batch_size, hi)],
            seq_len as f64,
        ));
    }
    // Otherwise 1-D along batch_size at matching seq_len.
    let bs_at_seq: Vec<u32> = by_bs_seq
        .keys()
        .filter_map(|&(bs, s)| if s == seq_len { Some(bs) } else { None })
        .collect();
    if bs_at_seq.len() >= 2 {
        let (lo, hi) = nearest_neighbors(batch_size, &bs_at_seq, false)?;
        return Ok(interp_1d(
            lo as f64,
            hi as f64,
            by_bs_seq[&(lo, seq_len)],
            by_bs_seq[&(hi, seq_len)],
            batch_size as f64,
        ));
    }
    // Fall back: tables for generation kernels store every row at a
    // single seq_len (typically 1) regardless of the query's seq_len.
    // Python ignores `seq_len` for gen-phase lookups and just walks
    // `list(table.keys())`. Mirror that: when no row matches `seq_len`,
    // pick the unique seq_len present in the table and interpolate by
    // batch_size at that slice.
    let unique_seqs: std::collections::BTreeSet<u32> =
        by_bs_seq.keys().map(|&(_, s)| s).collect();
    if let Some(&only_seq) = unique_seqs.iter().next().filter(|_| unique_seqs.len() == 1) {
        let bs_keys: Vec<u32> = by_bs_seq
            .keys()
            .map(|&(bs, _)| bs)
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        if let Some(&latency) = by_bs_seq.get(&(batch_size, only_seq)) {
            return Ok(latency);
        }
        if bs_keys.len() >= 2 {
            let (lo, hi) = nearest_neighbors(batch_size, &bs_keys, false)?;
            return Ok(interp_1d(
                lo as f64,
                hi as f64,
                by_bs_seq[&(lo, only_seq)],
                by_bs_seq[&(hi, only_seq)],
                batch_size as f64,
            ));
        }
    }
    Err(AicError::PerfDatabase(format!(
        "state-space lookup needs at least one bs/seq neighbour for ({batch_size}, {seq_len})"
    )))
}

fn load_mamba2_parquet(path: &Path) -> Result<Mamba2Grids, AicError> {
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

    let mut by_keys: BTreeMap<Mamba2Key, BTreeMap<(u32, u32), f64>> = BTreeMap::new();
    for row in reader.rows()? {
        let row = row?;
        let key = Mamba2Key {
            kernel_source: row.str_owned(kernel_source_col)?,
            phase: row.str_owned(phase_col)?,
            d_model: row.u32(d_model_col)?,
            d_state: row.u32(d_state_col)?,
            d_conv: row.u32(d_conv_col)?,
            nheads: row.u32(nheads_col)?,
            head_dim: row.u32(head_dim_col)?,
            n_groups: row.u32(n_groups_col)?,
            chunk_size: row.u32(chunk_size_col)?,
        };
        // First-wins parity with Python `load_mamba2_data`.
        by_keys
            .entry(key)
            .or_default()
            .entry((row.u32(batch_size_col)?, row.u32(seq_len_col)?))
            .or_insert(row.f64(latency_col)?);
    }
    if by_keys.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "no Mamba2 rows loaded from {}",
            path.display()
        )));
    }
    Ok(Mamba2Grids { by_keys })
}

fn load_gdn_parquet(path: &Path) -> Result<GdnGrids, AicError> {
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

    let mut by_keys: BTreeMap<GdnKey, BTreeMap<(u32, u32), f64>> = BTreeMap::new();
    for row in reader.rows()? {
        let row = row?;
        let key = GdnKey {
            kernel_source: row.str_owned(kernel_source_col)?,
            phase: row.str_owned(phase_col)?,
            d_model: row.u32(d_model_col)?,
            d_conv: row.u32(d_conv_col)?,
            num_k_heads: row.u32(num_k_heads_col)?,
            head_k_dim: row.u32(head_k_dim_col)?,
            num_v_heads: row.u32(num_v_heads_col)?,
            head_v_dim: row.u32(head_v_dim_col)?,
        };
        // First-wins parity with Python `load_gdn_data`.
        by_keys
            .entry(key)
            .or_default()
            .entry((row.u32(batch_size_col)?, row.u32(seq_len_col)?))
            .or_insert(row.f64(latency_col)?);
    }
    if by_keys.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "no GDN rows loaded from {}",
            path.display()
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

    #[test]
    fn state_space_loaders_smoke() {
        // GDN data exists on vLLM b200 (Nemotron-H slice); Mamba2 may not.
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../src/aiconfigurator/systems/data/b200_sxm/vllm/0.19.0");
        let table = StateSpaceTable::new(root);
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
            )
            .err();
    }

    #[test]
    fn gdn_table_finds_qwen35_27b_conv1d_update() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../src/aiconfigurator/systems/data/b200_sxm/vllm/0.19.0");
        let table = StateSpaceTable::new(root);
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
        );
        eprintln!("query: {r:?}");
        assert!(r.is_ok(), "expected silicon lookup to succeed: {r:?}");
        let latency = r.unwrap();
        assert!(latency > 0.0, "non-zero latency: {latency}");
        eprintln!("latency: {latency}");
    }
}
