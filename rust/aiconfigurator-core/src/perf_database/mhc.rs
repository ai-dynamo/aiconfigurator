// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! MHC (Qwen3.5 / DeepSeek-V4 multi-head channel) module perf table.
//!
//! CSV columns: model, architecture, op_name, num_tokens, hc_mult,
//! hidden_size, latency. Indexed by (op_name, architecture, hc_mult,
//! hidden_size) → num_tokens → latency. The token curve rides the shared
//! perf_interp v2 engine (1-axis Grid, RAW lerp in range, boundary util-hold
//! beyond it) — same wiring as Python `_query_mhc_table`'s silicon path.
//!
//! The util-hold SOL here is a LINEAR num_tokens proxy. Python anchors on
//! the mHC roofline (`dsv4.py::_query_mhc_table.get_sol`), which this table
//! layer cannot compute (no SystemSpec / quant / sinkhorn context); the
//! proxy only affects beyond-range holds — in-range lerp is SOL-free and
//! matches Python exactly. mHC latency is near-linear in tokens once past
//! the launch floor, so the proxy ratio tracks the roofline ratio closely.
//!
//! `op_name` is `pre` or `post` (the two halves of the mHC decoder layer) and
//! is part of the key: a given (arch, hc_mult, hidden_size, num_tokens) has a
//! distinct latency for each. Mirrors Python `load_mhc_module_data` /
//! `_query_mhc_table`, which key `data[op_name][hc_mult][hidden_size]`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use crate::common::error::AicError;
use super::moe::query_token_curve;
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
    architecture: String,
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
        architecture: &str,
    ) -> Result<f64, AicError> {
        let grids = self.load()?;
        // "both" aggregates the two silicon look-ups (Python sums pre+post).
        if op == "both" {
            return Ok(self.query_single("pre", num_tokens, hc_mult, hidden_size, architecture, grids)?
                + self.query_single("post", num_tokens, hc_mult, hidden_size, architecture, grids)?);
        }
        self.query_single(op, num_tokens, hc_mult, hidden_size, architecture, grids)
    }

    fn query_single(
        &self,
        op: &str,
        num_tokens: u32,
        hc_mult: u32,
        hidden_size: u32,
        architecture: &str,
        grids: &MhcGrids,
    ) -> Result<f64, AicError> {
        let key = MhcKey {
            op_name: op.to_string(),
            architecture: architecture.to_string(),
            hc_mult,
            hidden_size,
        };
        let by_tokens = grids.by_keys.get(&key).ok_or_else(|| {
            AicError::PerfDatabase(format!(
                "MHC module data missing for {key:?} at {}",
                self.data_root.display()
            ))
        })?;
        // Engine 1-axis token curve; linear num_tokens proxy SOL (see the
        // module docs for the deliberate divergence from Python's mHC
        // roofline on beyond-range holds).
        query_token_curve(by_tokens, num_tokens as f64, &|t| t)
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
    let arch_col = reader.col("architecture")?;
    let op_name_col = reader.col("op_name")?;
    let num_tokens_col = reader.col("num_tokens")?;
    let hc_mult_col = reader.col("hc_mult")?;
    let hidden_size_col = reader.col("hidden_size")?;
    let latency_col = reader.col("latency")?;

    let mut by_keys: BTreeMap<MhcKey, BTreeMap<u32, f64>> = BTreeMap::new();
    for row in reader.rows()? {
        let row = row?;
        let key = MhcKey {
            // `op_name` (pre/post) is part of the key — without it the pre and
            // post rows for the same (arch, hc_mult, hidden_size, num_tokens)
            // collide and `post` silently reads `pre`'s latency.
            op_name: row.str_owned(op_name_col)?,
            architecture: row.str_owned(arch_col)?,
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
    if by_keys.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "no MHC module rows loaded from {}",
            path.display()
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

    /// Cross-language parity with the Python v2 engine. Expected values from:
    ///
    /// ```text
    /// PYTHONPATH=src python3 -c "
    /// from aiconfigurator.sdk.perf_database import PerfDatabase
    /// from aiconfigurator.sdk import common
    /// db = PerfDatabase('b200_sxm','sglang','0.5.10',
    ///                   systems_root='src/aiconfigurator/systems', database_mode='SOL')
    /// for nt, op in [(3,'pre'), (3,'post'), (3,'both'), (8,'pre')]:
    ///     r = db.query_mhc_module(num_tokens=nt, hidden_size=7168, hc_mult=4,
    ///                             sinkhorn_iters=3, op=op,
    ///                             database_mode=common.DatabaseMode.SILICON)
    ///     print(nt, op, repr(float(r)))"
    /// ```
    ///
    /// In-range cases only: nt=3 is an interior RAW lerp (SOL-free, so the
    /// linear-proxy vs mHC-roofline SOL difference cannot surface), nt=8 an
    /// exact hit, and op="both" exercises the pre+post summing. Beyond-range
    /// holds deliberately diverge (see module docs) and are not compared.
    #[test]
    fn mhc_query_matches_python_v2_engine() {
        let table = MhcTable::new(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../src/aiconfigurator/systems/data/b200_sxm/sglang/0.5.10"),
        );
        let arch = "DeepseekV4ForCausalLM";
        let cases: &[(&str, u32, f64)] = &[
            ("pre", 3, 0.025050000000000003),
            ("post", 3, 0.01015),
            ("both", 3, 0.0352),
            ("pre", 8, 0.0251),
        ];
        for &(op, nt, expected) in cases {
            let got = table
                .query_module(op, nt, 4, 7168, arch)
                .expect("query must succeed");
            assert!(
                ((got - expected) / expected).abs() < 1e-9,
                "op={op}, nt={nt}: rust {got} vs python {expected}"
            );
        }
    }

    #[test]
    fn mhc_absent_on_vllm_b200_errors_clearly() {
        let table = MhcTable::new(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../src/aiconfigurator/systems/data/b200_sxm/vllm/0.19.0"));
        let err = table
            .query_module("pre", 1024, 2, 4096, "Qwen3_5MoeForConditionalGeneration")
            .unwrap_err();
        match err {
            AicError::Io { .. } | AicError::PerfDatabase(_) => {}
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
