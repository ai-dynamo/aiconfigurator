// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! MHC (Qwen3.5 / DeepSeek-V4 multi-head channel) module perf table.
//!
//! CSV columns: model, architecture, op_name, num_tokens, hc_mult,
//! hidden_size, latency. Indexed by (op_name, hc_mult, hidden_size)
//! → num_tokens → latency — exactly Python `load_mhc_module_data`'s
//! `data[op][hc_mult][hidden_size][num_tokens]` nesting. The `architecture`
//! column is IGNORED (Python's loader never reads it: mHC is selected by
//! compute shape); rows differing only in architecture merge into one curve
//! with per-row last-wins.
//!
//! The token curve rides the shared perf_interp v2 engine (1-axis Grid, RAW
//! lerp in range, boundary util-hold beyond it) — same wiring as Python
//! `_query_mhc_table`'s silicon path. The util-hold SOL is supplied by the
//! CALLER per resolved op half (`sol(op_name, tokens)`), threading the mHC
//! roofline from the operator exactly like `MoeTable::query` threads the MoE
//! roofline — Python anchors on `dsv4.py::_query_mhc_table.get_sol`.
//!
//! `op_name` is `pre` or `post` (the two halves of the mHC decoder layer) and
//! is part of the key: a given (hc_mult, hidden_size, num_tokens) has a
//! distinct latency for each.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::OnceLock;

use crate::common::error::AicError;
use crate::config::{PerfDbSources, PerfSource};
use super::{kernel_source_ok, resolve_op_sources};
use super::moe::query_token_curve;
use crate::perf_database::parquet_loader::PerfReader;

pub struct MhcTable {
    data_root: PathBuf,
    /// Ordered, priority-sorted sources for the mHC perf file (shared-layer
    /// aware; see [`PerfSource`]). Single-primary, no-filter by default
    /// (`MhcTable::new`).
    mhc_sources: Vec<PerfSource>,
    module: OnceLock<Result<MhcGrids, AicError>>,
}

struct MhcGrids {
    by_keys: BTreeMap<MhcKey, BTreeMap<u32, f64>>,
}

/// Python `load_mhc_module_data` keys `data[op][hc_mult][hidden_size]` — NO
/// architecture level. Keep this key architecture-free so both engines see
/// the same merged view for any data shape.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct MhcKey {
    op_name: String,
    hc_mult: u32,
    hidden_size: u32,
}

impl MhcTable {
    /// Construct an empty table for the given data directory. No I/O. The
    /// perf file is sourced solely from `data_root/mhc_module_perf.parquet`
    /// with no `kernel_source` filter (pre-shared-layer behaviour).
    pub fn new(data_root: PathBuf) -> Self {
        Self::with_sources(data_root, &PerfDbSources::default())
    }

    /// Construct with shared-layer (sibling/cross-version) sources resolved from
    /// `perf_db_sources` (Python-supplied). The mHC file falls back to its
    /// primary `data_root/mhc_module_perf.parquet` when absent from the map.
    /// No I/O.
    pub fn with_sources(data_root: PathBuf, perf_db_sources: &PerfDbSources) -> Self {
        let mhc_sources =
            resolve_op_sources(perf_db_sources, "mhc_module_perf.parquet", &data_root);
        Self {
            data_root,
            mhc_sources,
            module: OnceLock::new(),
        }
    }

    /// Query one mHC op. `op` is `pre`, `post`, or `both` (sum of pre+post),
    /// mirroring Python `_query_mhc_table`'s `op` argument.
    ///
    /// `sol(op_name, tokens)` is the analytic mHC roofline for one RESOLVED
    /// half (`"pre"` / `"post"`); it anchors beyond-range util-holds exactly
    /// like Python's `sol_fn=lambda t: get_sol(t, op_name)[0]`. For
    /// `op == "both"` the two halves are looked up (and SOL-anchored)
    /// separately and summed, matching Python's `_lookup_single("pre") +
    /// _lookup_single("post")`.
    pub fn query_module(
        &self,
        op: &str,
        num_tokens: u32,
        hc_mult: u32,
        hidden_size: u32,
        sol: &dyn Fn(&str, f64) -> f64,
    ) -> Result<f64, AicError> {
        let grids = self.load()?;
        // "both" aggregates the two silicon look-ups (Python sums pre+post).
        if op == "both" {
            return Ok(self.query_single("pre", num_tokens, hc_mult, hidden_size, sol, grids)?
                + self.query_single("post", num_tokens, hc_mult, hidden_size, sol, grids)?);
        }
        self.query_single(op, num_tokens, hc_mult, hidden_size, sol, grids)
    }

    fn query_single(
        &self,
        op: &str,
        num_tokens: u32,
        hc_mult: u32,
        hidden_size: u32,
        sol: &dyn Fn(&str, f64) -> f64,
        grids: &MhcGrids,
    ) -> Result<f64, AicError> {
        let key = MhcKey {
            op_name: op.to_string(),
            hc_mult,
            hidden_size,
        };
        let by_tokens = grids.by_keys.get(&key).ok_or_else(|| {
            AicError::PerfDatabase(format!(
                "MHC module data missing for {key:?} at {}",
                self.data_root.display()
            ))
        })?;
        // Engine 1-axis token curve; the caller-threaded per-op roofline
        // anchors beyond-range holds (Python `sol_fn=lambda t: get_sol(t,
        // op_name)[0]`).
        query_token_curve(by_tokens, num_tokens as f64, &|t| sol(op, t))
    }

    /// Collected `(num_tokens,) -> latency` points for one RESOLVED op half
    /// (`pre` / `post`), for the operator-layer util-calibration grid (Python
    /// `_query_mhc_table::get_empirical`'s
    /// `require_data_slice(mhc_data, op_name, hc_mult, hidden_size)` +
    /// `iter_grid(..., depth=1)`). Missing key / empty curve is a typed
    /// `PerfDatabase` miss; the `both` composition lives in the operator
    /// (Python sums `_emp_for_op("pre") + _emp_for_op("post")`).
    pub fn module_points(
        &self,
        op: &str,
        hc_mult: u32,
        hidden_size: u32,
    ) -> Result<Vec<(Vec<f64>, f64)>, AicError> {
        let grids = self.load()?;
        let key = MhcKey {
            op_name: op.to_string(),
            hc_mult,
            hidden_size,
        };
        let by_tokens = grids.by_keys.get(&key).ok_or_else(|| {
            AicError::PerfDatabase(format!(
                "MHC module data missing for {key:?} at {}",
                self.data_root.display()
            ))
        })?;
        if by_tokens.is_empty() {
            return Err(AicError::PerfDatabase(format!(
                "MHC module data empty for {key:?} at {}",
                self.data_root.display()
            )));
        }
        Ok(by_tokens
            .iter()
            .map(|(&tokens, &latency)| (vec![f64::from(tokens)], latency))
            .collect())
    }

    fn load(&self) -> Result<&MhcGrids, AicError> {
        let cell = self
            .module
            .get_or_init(|| load_mhc_parquet(&self.mhc_sources));
        cell.as_ref().map_err(clone_err)
    }
}

/// Load the mHC module table from an ordered, priority-sorted source list.
/// Sources are read in order (shared-layer aware). Missing files are skipped (a
/// sibling declared in the manifest need not exist for every system); an error
/// is returned only when no source yields rows.
fn load_mhc_parquet(sources: &[PerfSource]) -> Result<MhcGrids, AicError> {
    let mut by_keys: BTreeMap<MhcKey, BTreeMap<u32, f64>> = BTreeMap::new();
    let mut any_source = false;
    for source in sources {
        let path = source.path();
        if !path.exists() {
            continue;
        }
        any_source = true;
        let reader = PerfReader::open(path)?;
        let op_name_col = reader.col("op_name")?;
        let num_tokens_col = reader.col("num_tokens")?;
        let hc_mult_col = reader.col("hc_mult")?;
        let hidden_size_col = reader.col("hidden_size")?;
        let latency_col = reader.col("latency")?;
        let ks_col = reader.col_optional("kernel_source");

        for row in reader.rows()? {
            let row = row?;
            if !kernel_source_ok(source.kernel_sources(), ks_col, &row)? {
                continue;
            }
            let key = MhcKey {
                // `op_name` (pre/post) is part of the key — without it the pre and
                // post rows for the same (hc_mult, hidden_size, num_tokens)
                // collide and `post` silently reads `pre`'s latency. The
                // `architecture` column is intentionally NOT read: Python's
                // loader ignores it, so rows differing only in architecture
                // merge into one curve (per-row last-wins below).
                op_name: row.str_owned(op_name_col)?,
                hc_mult: row.u32(hc_mult_col)?,
                hidden_size: row.u32(hidden_size_col)?,
            };
            // Last-wins parity with Python `load_mhc_module_data`, which assigns
            // `mhc_data[op][hc_mult][hidden_size][num_tokens] = {...}` per row.
            by_keys
                .entry(key)
                .or_default()
                .insert(row.u32(num_tokens_col)?, row.f64(latency_col)?);
        }
    }
    if !any_source || by_keys.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "no MHC module rows loaded from {} source(s) (first: {})",
            sources.len(),
            sources
                .first()
                .map(|s| s.path().display().to_string())
                .unwrap_or_default()
        )));
    }
    Ok(MhcGrids { by_keys })
}

fn clone_err(err: &AicError) -> AicError {
    AicError::PerfDatabase(err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use std::sync::Arc;

    /// Linear token proxy: only valid for in-range assertions (in-range lerp
    /// is SOL-free) and for demonstrating the OLD beyond-range behaviour.
    fn linear_sol(_op: &str, t: f64) -> f64 {
        t
    }

    /// Cross-language parity with the Python v2 engine. Expected values from:
    ///
    /// ```text
    /// PYTHONPATH=src python3 -c "
    /// from aiconfigurator.sdk.perf_database import PerfDatabase
    /// from aiconfigurator.sdk import common
    /// db = PerfDatabase('b200_sxm','sglang','0.5.10',
    ///                   systems_root='src/aiconfigurator_core/systems', database_mode='SOL')
    /// for nt, op in [(3,'pre'), (3,'post'), (3,'both'), (8,'pre')]:
    ///     r = db.query_mhc_module(num_tokens=nt, hidden_size=7168, hc_mult=4,
    ///                             sinkhorn_iters=3, op=op,
    ///                             database_mode=common.DatabaseMode.SILICON)
    ///     print(nt, op, repr(float(r)))"
    /// ```
    ///
    /// In-range cases only: nt=3 is an interior RAW lerp (SOL-free, so any
    /// sol closure gives the same answer), nt=8 an exact hit, and op="both"
    /// exercises the pre+post summing. Beyond-range holds are covered by the
    /// operator-level test (`operators/mhc.rs`), which threads the real mHC
    /// roofline.
    // NOTE(shared-layer merge): oracle generated pre-shared-layer; regenerate if
    // this fails. `MhcTable::new` resolves to the single primary source with no
    // kernel_source filter, so no shared rows should join this curve.
    #[test]
    fn mhc_query_matches_python_v2_engine() {
        let table = MhcTable::new(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../src/aiconfigurator_core/systems/data/b200_sxm/sglang/0.5.10"),
        );
        let cases: &[(&str, u32, f64)] = &[
            ("pre", 3, 0.025050000000000003),
            ("post", 3, 0.01015),
            ("both", 3, 0.0352),
            ("pre", 8, 0.0251),
        ];
        for &(op, nt, expected) in cases {
            let got = table
                .query_module(op, nt, 4, 7168, &linear_sol)
                .expect("query must succeed");
            assert!(
                ((got - expected) / expected).abs() < 1e-9,
                "op={op}, nt={nt}: rust {got} vs python {expected}"
            );
        }
    }

    #[test]
    fn mhc_absent_on_vllm_b200_errors_clearly() {
        let table = MhcTable::new(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../src/aiconfigurator_core/systems/data/b200_sxm/vllm/0.19.0"));
        let err = table
            .query_module("pre", 1024, 2, 4096, &linear_sol)
            .unwrap_err();
        match err {
            AicError::Io { .. } | AicError::PerfDatabase(_) => {}
            other => panic!("unexpected error: {other:?}"),
        }
    }

    /// Write one synthetic mHC parquet with the collector's column set
    /// (`architecture, op_name, num_tokens, hc_mult, hidden_size, latency`).
    fn write_mhc_parquet(path: &Path, rows: &[(&str, &str, i64, i64, i64, f64)]) {
        use parquet::data_type::{ByteArray, ByteArrayType, DoubleType, Int64Type};
        use parquet::file::properties::WriterProperties;
        use parquet::file::writer::SerializedFileWriter;
        use parquet::schema::parser::parse_message_type;

        let schema = "message schema {
            REQUIRED BINARY architecture (UTF8);
            REQUIRED BINARY op_name (UTF8);
            REQUIRED INT64 num_tokens;
            REQUIRED INT64 hc_mult;
            REQUIRED INT64 hidden_size;
            REQUIRED DOUBLE latency;
        }";
        let schema = Arc::new(parse_message_type(schema).expect("schema must parse"));
        let file = std::fs::File::create(path).expect("create parquet");
        let mut writer =
            SerializedFileWriter::new(file, schema, Arc::new(WriterProperties::builder().build()))
                .expect("writer");
        let mut rg = writer.next_row_group().expect("row group");
        for str_field in [0usize, 1] {
            let values: Vec<ByteArray> = rows
                .iter()
                .map(|r| ByteArray::from(if str_field == 0 { r.0 } else { r.1 }))
                .collect();
            let mut col = rg.next_column().expect("next col").expect("str col");
            col.typed::<ByteArrayType>().write_batch(&values, None, None).expect("write str");
            col.close().expect("close col");
        }
        let int_cols: [Vec<i64>; 3] = [
            rows.iter().map(|r| r.2).collect(),
            rows.iter().map(|r| r.3).collect(),
            rows.iter().map(|r| r.4).collect(),
        ];
        for values in &int_cols {
            let mut col = rg.next_column().expect("next col").expect("int col");
            col.typed::<Int64Type>().write_batch(values, None, None).expect("write ints");
            col.close().expect("close col");
        }
        let latencies: Vec<f64> = rows.iter().map(|r| r.5).collect();
        let mut col = rg.next_column().expect("next col").expect("latency col");
        col.typed::<DoubleType>().write_batch(&latencies, None, None).expect("write latency");
        col.close().expect("close col");
        rg.close().expect("close row group");
        writer.close().expect("close writer");
    }

    /// Item 4: Python `load_mhc_module_data` keys `data[op][hc_mult][hidden]`
    /// — NO architecture level — so two rows differing ONLY in architecture
    /// merge into one curve with per-row last-wins (later row overwrites).
    /// The old Rust `MhcKey` carried `architecture`, splitting these rows
    /// into two per-arch views and answering 1.0 for arch "A" (Python: 2.0).
    #[test]
    fn mhc_rows_differing_only_in_architecture_merge_last_wins() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        write_mhc_parquet(
            &tmp.path().join("mhc_module_perf.parquet"),
            &[
                ("ArchA", "pre", 8, 4, 7168, 1.0),
                ("ArchB", "pre", 8, 4, 7168, 2.0), // same coordinate: LAST row wins
                ("ArchA", "pre", 16, 4, 7168, 4.0), // different token: merges into the curve
            ],
        );
        let table = MhcTable::new(tmp.path().to_path_buf());
        // Exact hit at the duplicated coordinate: the merged view answers 2.0.
        let got = table
            .query_module("pre", 8, 4, 7168, &linear_sol)
            .expect("query must succeed");
        assert_eq!(got, 2.0);
        // The ArchA-only token joins the same curve (single merged view):
        // interior lerp between 2.0@8 and 4.0@16.
        let mid = table
            .query_module("pre", 12, 4, 7168, &linear_sol)
            .expect("query must succeed");
        assert_eq!(mid, 3.0);
    }

    /// Item 1 (mechanism): beyond-range holds must anchor on the CALLER'S sol
    /// ratio, not a hardwired linear token proxy. With sol = t², the hold at
    /// q = 2·t_max is lat(t_max) · sol(q)/sol(t_max) = 3.0 · 4 = 12.0; the old
    /// built-in linear proxy returned 3.0 · 2 = 6.0.
    #[test]
    fn mhc_beyond_range_hold_uses_threaded_sol() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        write_mhc_parquet(
            &tmp.path().join("mhc_module_perf.parquet"),
            &[
                ("DeepseekV4ForCausalLM", "pre", 65536, 4, 7168, 1.0),
                ("DeepseekV4ForCausalLM", "pre", 131072, 4, 7168, 3.0),
            ],
        );
        let table = MhcTable::new(tmp.path().to_path_buf());
        let quadratic = |_op: &str, t: f64| t * t;
        let got = table
            .query_module("pre", 262144, 4, 7168, &quadratic)
            .expect("query must succeed");
        assert!(
            (got - 12.0).abs() < 1e-12,
            "hold must scale by the threaded sol ratio (expected 12.0, got {got})"
        );
    }
}
