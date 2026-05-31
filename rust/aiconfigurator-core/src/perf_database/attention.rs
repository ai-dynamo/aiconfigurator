// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Attention family perf tables: context, generation, encoder.
//!
//! Mirrors the raw table layout used by Python's
//! `aiconfigurator.sdk.operations.attention.{ContextAttention,
//! GenerationAttention, EncoderAttention}._query_*_table` SILICON paths.
//!
//! Each variant nests its data as `(discrete keys) -> 3-D grid` where the
//! 3-D grid is keyed by the three continuous interpolation axes:
//! - context attention: `(num_heads, full_seq_tokens, batch_size)`
//! - generation attention: `(num_heads, kv_seq_tokens, batch_size)` where
//!   `kv_seq_tokens = isl + step`
//! - encoder attention: `(num_heads, seq_tokens, batch_size)`
//!
//! `n_kv` is normalized to `0` when `num_heads == num_key_value_heads`
//! (MHA sentinel), matching Python's `n_kv_lookup` rule. `window_size`
//! defaults to `0` for backends whose collectors don't record it.
//!
//! The query methods on this table return raw interpolated latency in ms.
//! The operator layer wraps these with prefix correction, SOL/EMPIRICAL
//! fallbacks, and extra fused-op accounting (qk_norm, rope, kv writes).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use crate::common::enums::{FmhaQuantMode, KvCacheQuantMode};
use crate::common::error::AicError;
use crate::interpolation::{interp_2d_1d_grid, Grid3};
use crate::perf_database::parquet_loader::PerfReader;

pub struct AttentionTable {
    data_root: PathBuf,
    context: OnceLock<Result<ContextGrids, AicError>>,
    generation: OnceLock<Result<GenerationGrids, AicError>>,
    encoder: OnceLock<Result<EncoderGrids, AicError>>,
}

struct ContextGrids {
    by_keys: BTreeMap<ContextKey, Grid3<f64>>,
}

struct GenerationGrids {
    by_keys: BTreeMap<GenerationKey, Grid3<f64>>,
}

struct EncoderGrids {
    by_keys: BTreeMap<EncoderKey, Grid3<f64>>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct ContextKey {
    fmha_quant: String,
    kv_quant: String,
    n_kv_lookup: u32,
    head_size: u32,
    window_size: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct GenerationKey {
    kv_quant: String,
    n_kv_lookup: u32,
    head_size: u32,
    window_size: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct EncoderKey {
    fmha_quant: String,
    head_size: u32,
}

impl AttentionTable {
    pub fn new(data_root: PathBuf) -> Self {
        Self {
            data_root,
            context: OnceLock::new(),
            generation: OnceLock::new(),
            encoder: OnceLock::new(),
        }
    }

    /// Raw interpolated context attention latency in ms.
    ///
    /// `full_seq_tokens = isl + prefix` from the caller's perspective. The
    /// operator layer applies the prefix correction multiplier
    /// `(full_s² - prefix²) / full_s²`.
    pub fn query_context(
        &self,
        b: u32,
        full_seq_tokens: u32,
        n: u32,
        n_kv: u32,
        head_size: u32,
        window_size: u32,
        kv_quant: KvCacheQuantMode,
        fmha_quant: FmhaQuantMode,
    ) -> Result<f64, AicError> {
        let grids = self.load_context()?;
        let key = ContextKey {
            fmha_quant: fmha_quant.name().to_string(),
            kv_quant: kv_quant.name().to_string(),
            n_kv_lookup: normalize_kv(n, n_kv),
            head_size,
            window_size,
        };
        let grid = grids.by_keys.get(&key).ok_or_else(|| missing_key(&self.data_root, &key))?;
        // Python uses (n, full_s, b) as the (x, y, z) axes for interp_3d.
        interp_2d_1d_grid(grid, n, full_seq_tokens, b)
    }

    /// Raw interpolated generation attention latency in ms.
    ///
    /// `kv_seq_tokens` is the total decode context length (Python passes
    /// `s` from the caller; the CSV stores `isl + step`).
    pub fn query_generation(
        &self,
        b: u32,
        kv_seq_tokens: u32,
        n: u32,
        n_kv: u32,
        head_size: u32,
        window_size: u32,
        kv_quant: KvCacheQuantMode,
    ) -> Result<f64, AicError> {
        let grids = self.load_generation()?;
        let key = GenerationKey {
            kv_quant: kv_quant.name().to_string(),
            n_kv_lookup: normalize_kv(n, n_kv),
            head_size,
            window_size,
        };
        let grid = grids
            .by_keys
            .get(&key)
            .ok_or_else(|| missing_gen_key(&self.data_root, &key))?;
        interp_2d_1d_grid(grid, n, kv_seq_tokens, b)
    }

    /// Raw interpolated encoder (non-causal) attention latency in ms.
    pub fn query_encoder(
        &self,
        b: u32,
        s: u32,
        n: u32,
        head_size: u32,
        fmha_quant: FmhaQuantMode,
    ) -> Result<f64, AicError> {
        let grids = self.load_encoder()?;
        let key = EncoderKey {
            fmha_quant: fmha_quant.name().to_string(),
            head_size,
        };
        let grid = grids
            .by_keys
            .get(&key)
            .ok_or_else(|| missing_encoder_key(&self.data_root, &key))?;
        interp_2d_1d_grid(grid, n, s, b)
    }

    fn load_context(&self) -> Result<&ContextGrids, AicError> {
        let cell = self.context.get_or_init(|| {
            load_context_parquet(&self.data_root.join("context_attention_perf.parquet"))
        });
        cell.as_ref().map_err(clone_err)
    }

    fn load_generation(&self) -> Result<&GenerationGrids, AicError> {
        let cell = self.generation.get_or_init(|| {
            load_generation_parquet(&self.data_root.join("generation_attention_perf.parquet"))
        });
        cell.as_ref().map_err(clone_err)
    }

    fn load_encoder(&self) -> Result<&EncoderGrids, AicError> {
        let cell = self.encoder.get_or_init(|| {
            load_encoder_parquet(&self.data_root.join("encoder_attention_perf.parquet"))
        });
        cell.as_ref().map_err(clone_err)
    }
}

/// Mirror Python's `n_kv_lookup = 0 if n == n_kv else n_kv` (MHA sentinel).
fn normalize_kv(n: u32, n_kv: u32) -> u32 {
    if n_kv == n {
        0
    } else {
        n_kv
    }
}

fn load_context_parquet(path: &Path) -> Result<ContextGrids, AicError> {
    let reader = PerfReader::open(path)?;
    let batch_size_col = reader.col("batch_size")?;
    let isl_col = reader.col("isl")?;
    let num_heads_col = reader.col("num_heads")?;
    let num_kv_col = reader.col("num_key_value_heads")?;
    let head_dim_col = reader.col("head_dim")?;
    let attn_dtype_col = reader.col("attn_dtype")?;
    let kv_cache_dtype_col = reader.col("kv_cache_dtype")?;
    let latency_col = reader.col("latency")?;
    let window_size_col = reader.col_optional("window_size");

    let mut by_keys: BTreeMap<ContextKey, Grid3<f64>> = BTreeMap::new();
    for row in reader.rows()? {
        let row = row?;
        let num_heads = row.u32(num_heads_col)?;
        let num_kv = row.u32(num_kv_col)?;
        let key = ContextKey {
            fmha_quant: row.str_owned(attn_dtype_col)?,
            kv_quant: row.str_owned(kv_cache_dtype_col)?,
            n_kv_lookup: normalize_kv(num_heads, num_kv),
            head_size: row.u32(head_dim_col)?,
            window_size: row.u32_optional(window_size_col)?.unwrap_or(0),
        };
        // First-wins parity with Python `load_context_attention_data`.
        by_keys
            .entry(key)
            .or_default()
            .entry(num_heads)
            .or_default()
            .entry(row.u32(isl_col)?)
            .or_default()
            .entry(row.u32(batch_size_col)?)
            .or_insert(row.f64(latency_col)?);
    }
    if by_keys.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "no context-attention rows loaded from {}",
            path.display()
        )));
    }
    Ok(ContextGrids { by_keys })
}

fn load_generation_parquet(path: &Path) -> Result<GenerationGrids, AicError> {
    let reader = PerfReader::open(path)?;
    let batch_size_col = reader.col("batch_size")?;
    let isl_col = reader.col("isl")?;
    let num_heads_col = reader.col("num_heads")?;
    let num_kv_col = reader.col("num_key_value_heads")?;
    let head_dim_col = reader.col("head_dim")?;
    let kv_cache_dtype_col = reader.col("kv_cache_dtype")?;
    let step_col = reader.col("step")?;
    let latency_col = reader.col("latency")?;
    let window_size_col = reader.col_optional("window_size");

    let mut by_keys: BTreeMap<GenerationKey, Grid3<f64>> = BTreeMap::new();
    for row in reader.rows()? {
        let row = row?;
        let num_heads = row.u32(num_heads_col)?;
        let num_kv = row.u32(num_kv_col)?;
        let key = GenerationKey {
            kv_quant: row.str_owned(kv_cache_dtype_col)?,
            n_kv_lookup: normalize_kv(num_heads, num_kv),
            head_size: row.u32(head_dim_col)?,
            window_size: row.u32_optional(window_size_col)?.unwrap_or(0),
        };
        let sequence_tokens = row.u32(isl_col)? + row.u32(step_col)?;
        // First-wins parity with Python `load_generation_attention_data`.
        by_keys
            .entry(key)
            .or_default()
            .entry(num_heads)
            .or_default()
            .entry(sequence_tokens)
            .or_default()
            .entry(row.u32(batch_size_col)?)
            .or_insert(row.f64(latency_col)?);
    }
    if by_keys.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "no generation-attention rows loaded from {}",
            path.display()
        )));
    }
    Ok(GenerationGrids { by_keys })
}

fn load_encoder_parquet(path: &Path) -> Result<EncoderGrids, AicError> {
    let reader = PerfReader::open(path)?;
    let batch_size_col = reader.col("batch_size")?;
    let isl_col = reader.col("isl")?;
    let num_heads_col = reader.col("num_heads")?;
    let head_dim_col = reader.col("head_dim")?;
    let attn_dtype_col = reader.col("attn_dtype")?;
    let latency_col = reader.col("latency")?;

    let mut by_keys: BTreeMap<EncoderKey, Grid3<f64>> = BTreeMap::new();
    for row in reader.rows()? {
        let row = row?;
        let key = EncoderKey {
            fmha_quant: row.str_owned(attn_dtype_col)?,
            head_size: row.u32(head_dim_col)?,
        };
        // First-wins parity with Python `load_encoder_attention_data`.
        by_keys
            .entry(key)
            .or_default()
            .entry(row.u32(num_heads_col)?)
            .or_default()
            .entry(row.u32(isl_col)?)
            .or_default()
            .entry(row.u32(batch_size_col)?)
            .or_insert(row.f64(latency_col)?);
    }
    if by_keys.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "no encoder-attention rows loaded from {}",
            path.display()
        )));
    }
    Ok(EncoderGrids { by_keys })
}

fn missing_key(data_root: &Path, key: &ContextKey) -> AicError {
    AicError::PerfDatabase(format!(
        "context attention data missing for {key:?} at {}",
        data_root.display()
    ))
}

fn missing_gen_key(data_root: &Path, key: &GenerationKey) -> AicError {
    AicError::PerfDatabase(format!(
        "generation attention data missing for {key:?} at {}",
        data_root.display()
    ))
}

fn missing_encoder_key(data_root: &Path, key: &EncoderKey) -> AicError {
    AicError::PerfDatabase(format!(
        "encoder attention data missing for {key:?} at {}",
        data_root.display()
    ))
}

fn clone_err(err: &AicError) -> AicError {
    AicError::PerfDatabase(err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    const REPO_ROOT_HINT: &str = env!("CARGO_MANIFEST_DIR");

    fn b200_vllm_data_root() -> PathBuf {
        PathBuf::from(REPO_ROOT_HINT)
            .join("../..")
            .join("src/aiconfigurator/systems/data/b200_sxm/vllm/0.19.0")
    }

    #[test]
    fn context_attention_exact_hit() {
        // First row of b200_sxm/vllm/0.19.0/context_attention_perf.txt:
        // batch=8 isl=16384 n=64 n_kv=1 head_dim=128 attn=bfloat16 kv=fp8 step=0 latency=19.82
        let table = AttentionTable::new(b200_vllm_data_root());
        let latency = table
            .query_context(
                8,
                16384,
                64,
                1,
                128,
                0,
                KvCacheQuantMode::Fp8,
                FmhaQuantMode::Bfloat16,
            )
            .expect("query must succeed");
        assert!(
            (latency - 19.820667266845703).abs() < 1e-9,
            "expected recorded latency, got {latency}"
        );
    }

    #[test]
    fn generation_attention_exact_hit() {
        // Second row of generation_attention_perf.txt:
        // batch=32 isl=1 n=64 n_kv=4 head_dim=128 kv=fp8 step=1 latency=0.00866
        let table = AttentionTable::new(b200_vllm_data_root());
        // The loader stores sequence_tokens = isl + step = 2.
        let latency = table
            .query_generation(32, 2, 64, 4, 128, 0, KvCacheQuantMode::Fp8)
            .expect("query must succeed");
        assert!(
            (latency - 0.008661333471536636).abs() < 1e-9,
            "expected recorded latency, got {latency}"
        );
    }

    #[test]
    fn context_attention_mha_normalizes_n_kv_to_zero() {
        // Real MHA row from vLLM b200 context attention:
        // b=4 isl=16384 n=64 n_kv=64 head=128 fmha=bfloat16 kv=fp8 latency=9.98
        // Caller passes n_kv=64; loader normalizes to n_kv_lookup=0 since
        // n==n_kv (MHA). Query should hit the same recorded row.
        let table = AttentionTable::new(b200_vllm_data_root());
        let latency = table
            .query_context(
                4,
                16384,
                64,
                64,
                128,
                0,
                KvCacheQuantMode::Fp8,
                FmhaQuantMode::Bfloat16,
            )
            .expect("MHA lookup must normalize and find the row");
        assert!(
            (latency - 9.983466466267904).abs() < 1e-9,
            "expected recorded MHA latency, got {latency}"
        );
    }

    #[test]
    fn context_attention_missing_quant_combo_errors() {
        let table = AttentionTable::new(b200_vllm_data_root());
        // vLLM b200 context attention has fmha=bfloat16 only; Fp8 fmha
        // is genuinely absent.
        match table.query_context(
            1,
            1024,
            64,
            1,
            128,
            0,
            KvCacheQuantMode::Fp8,
            FmhaQuantMode::Fp8,
        ) {
            Err(AicError::PerfDatabase(_)) => {}
            other => panic!("expected PerfDatabase error, got {other:?}"),
        }
    }

    #[test]
    fn encoder_attention_absent_on_vllm_b200_errors_clearly() {
        // vLLM b200 doesn't ship encoder_attention_perf.txt; expect a clear
        // IO error from the lazy loader.
        let table = AttentionTable::new(b200_vllm_data_root());
        let err = table
            .query_encoder(1, 1024, 16, 64, FmhaQuantMode::Bfloat16)
            .unwrap_err();
        match err {
            AicError::Io { .. } | AicError::PerfDatabase(_) => {}
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn context_attention_lazy_loads_once() {
        let table = AttentionTable::new(b200_vllm_data_root());
        let first = table
            .query_context(
                8,
                16384,
                64,
                1,
                128,
                0,
                KvCacheQuantMode::Fp8,
                FmhaQuantMode::Bfloat16,
            )
            .unwrap();
        let second = table
            .query_context(
                8,
                16384,
                64,
                1,
                128,
                0,
                KvCacheQuantMode::Fp8,
                FmhaQuantMode::Bfloat16,
            )
            .unwrap();
        assert_eq!(first, second);
    }
}
