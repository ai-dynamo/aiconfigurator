// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! MHC (Qwen3.5 / DeepSeek-V4 multi-head channel) module perf table.
//!
//! CSV columns: model, architecture, op_name, num_tokens, hc_mult,
//! hidden_size, latency. Indexed by `(op_name, hc_mult, hidden_size) →
//! num_tokens → latency`. `model` and `architecture` are provenance-only,
//! matching Python and the collector physical-row key. Query is 1-D
//! interpolation along `num_tokens`.
//!
//! `op_name` is `pre` or `post` (the two halves of the mHC decoder layer) and
//! is part of the key: a given `(hc_mult, hidden_size, num_tokens)` shape has a
//! distinct latency for each. Mirrors Python `load_mhc_module_data` /
//! `_query_mhc_table`, which key `data[op_name][hc_mult][hidden_size]`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use super::interpolation::{interp_1d, nearest_neighbors};
use crate::common::error::AicError;
use crate::perf_database::parquet_loader::PerfReader;

pub struct MhcTable {
    data_root: PathBuf,
    module: OnceLock<Result<MhcGrids, AicError>>,
}

struct MhcGrids {
    by_keys: BTreeMap<MhcKey, BTreeMap<u32, f64>>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct MhcKey {
    op_name: String,
    hc_mult: u32,
    hidden_size: u32,
}

impl MhcTable {
    pub fn new(data_root: PathBuf) -> Self {
        Self {
            data_root,
            module: OnceLock::new(),
        }
    }

    /// Query one mHC op. `op` is `pre`, `post`, or `both` (sum of pre+post),
    /// mirroring Python `_query_mhc_table`'s `op` argument.
    pub fn query_module(
        &self,
        op: &str,
        num_tokens: u32,
        hc_mult: u32,
        hidden_size: u32,
    ) -> Result<f64, AicError> {
        let grids = self.load()?;
        // "both" aggregates the two silicon look-ups (Python sums pre+post).
        if op == "both" {
            return Ok(
                self.query_single("pre", num_tokens, hc_mult, hidden_size, grids)?
                    + self.query_single("post", num_tokens, hc_mult, hidden_size, grids)?,
            );
        }
        self.query_single(op, num_tokens, hc_mult, hidden_size, grids)
    }

    fn query_single(
        &self,
        op: &str,
        num_tokens: u32,
        hc_mult: u32,
        hidden_size: u32,
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
        if let Some(&latency) = by_tokens.get(&num_tokens) {
            return Ok(latency);
        }
        let token_keys: Vec<u32> = by_tokens.keys().copied().collect();
        let (lo, hi) = nearest_neighbors(num_tokens, &token_keys, false)?;
        Ok(interp_1d(
            lo as f64,
            hi as f64,
            by_tokens[&lo],
            by_tokens[&hi],
            num_tokens as f64,
        ))
    }

    fn load(&self) -> Result<&MhcGrids, AicError> {
        let cell = self
            .module
            .get_or_init(|| load_mhc_parquet(&self.data_root.join("mhc_module_perf.parquet")));
        cell.as_ref().map_err(clone_err)
    }
}

fn load_mhc_parquet(path: &Path) -> Result<MhcGrids, AicError> {
    let reader = PerfReader::open(path)?;
    let op_name_col = reader.col("op_name")?;
    let num_tokens_col = reader.col("num_tokens")?;
    let hc_mult_col = reader.col("hc_mult")?;
    let hidden_size_col = reader.col("hidden_size")?;
    let latency_col = reader.col("latency")?;

    let mut by_keys: BTreeMap<MhcKey, BTreeMap<u32, f64>> = BTreeMap::new();
    for row in reader.rows()? {
        let row = row?;
        insert_mhc_measurement(
            &mut by_keys,
            row.str_owned(op_name_col)?,
            row.u32(hc_mult_col)?,
            row.u32(hidden_size_col)?,
            row.u32(num_tokens_col)?,
            row.f64(latency_col)?,
        );
    }
    if by_keys.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "no MHC module rows loaded from {}",
            path.display()
        )));
    }
    Ok(MhcGrids { by_keys })
}

fn insert_mhc_measurement(
    by_keys: &mut BTreeMap<MhcKey, BTreeMap<u32, f64>>,
    op_name: impl Into<String>,
    hc_mult: u32,
    hidden_size: u32,
    num_tokens: u32,
    latency: f64,
) {
    let key = MhcKey {
        // `op_name` (pre/post) is part of the physical key; architecture is
        // intentionally absent because Python selects MHC by compute shape.
        op_name: op_name.into(),
        hc_mult,
        hidden_size,
    };
    // Last-wins parity with Python `load_mhc_module_data`, which assigns
    // `mhc_data[op][hc_mult][hidden_size][num_tokens] = {...}` per row.
    by_keys.entry(key).or_default().insert(num_tokens, latency);
}

fn clone_err(err: &AicError) -> AicError {
    AicError::PerfDatabase(err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mhc_absent_on_vllm_b200_errors_clearly() {
        let table = MhcTable::new(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../src/aiconfigurator/systems/data/b200_sxm/vllm/0.19.0"),
        );
        let err = table.query_module("pre", 1024, 2, 4096).unwrap_err();
        match err {
            AicError::Io { .. } | AicError::PerfDatabase(_) => {}
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn mhc_architecture_provenance_collides_and_last_row_wins() {
        let rows = [
            ("Qwen3_5MoeForConditionalGeneration", 11.0),
            ("DeepseekV4ForCausalLM", 19.0),
        ];
        let mut grids = BTreeMap::new();
        for (_architecture, latency) in rows {
            // Architecture is deliberately not passed to the physical-key
            // insertion helper, exactly like Python's loader.
            insert_mhc_measurement(&mut grids, "pre", 4, 7168, 1024, latency);
        }

        let key = MhcKey {
            op_name: "pre".into(),
            hc_mult: 4,
            hidden_size: 7168,
        };
        assert_eq!(grids.len(), 1);
        assert_eq!(grids[&key][&1024], 19.0);
    }

    #[test]
    fn mhc_pre_and_post_remain_distinct_physical_buckets() {
        let mut grids = BTreeMap::new();
        insert_mhc_measurement(&mut grids, "pre", 4, 7168, 1024, 11.0);
        insert_mhc_measurement(&mut grids, "post", 4, 7168, 1024, 19.0);

        assert_eq!(grids.len(), 2);
        assert_eq!(
            grids[&MhcKey {
                op_name: "pre".into(),
                hc_mult: 4,
                hidden_size: 7168,
            }][&1024],
            11.0
        );
        assert_eq!(
            grids[&MhcKey {
                op_name: "post".into(),
                hc_mult: 4,
                hidden_size: 7168,
            }][&1024],
            19.0
        );
    }
}
