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
use crate::common::system_spec::SystemSpec;
use super::interpolation::{interp_2d_1d_grid, interp_2d_1d_grid_strict, Grid3};
use crate::perf_database::parquet_loader::PerfReader;

pub struct AttentionTable {
    data_root: PathBuf,
    system_spec: SystemSpec,
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
    pub fn new(data_root: PathBuf, system_spec: SystemSpec) -> Self {
        Self {
            data_root,
            system_spec,
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
        // Python's `_query_generation_attention_table` calls
        // `interp_3d(..., "bilinear")` with the default
        // `allow_singleton_axes=False`, which runs `_require_3d_axis_coverage`
        // and raises `ValueError` when the grid does not vary across all three
        // axes (e.g. Gemma-4 generation attention with a single `n` value).
        // Use the strict interpolation variant so Rust surfaces the same error.
        interp_2d_1d_grid_strict(grid, n, kv_seq_tokens, b, "3-D bilinear interpolation")
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
            let mut grids = load_generation_parquet(
                &self.data_root.join("generation_attention_perf.parquet"),
            )?;
            // Mirror Python `GenerationAttention._correct_sol`: clamp every
            // stored grid entry to `>= SOL`. Context-phase attention is
            // intentionally NOT clamped here — Python's `_correct_data`
            // historically skipped it (see comment in
            // `aiconfigurator.sdk.operations.attention`).
            clamp_generation_attention_grids_to_sol(&self.system_spec, &mut grids);
            Ok(grids)
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

/// Speed-of-light generation-attention latency in ms.
///
/// Mirrors Python's `GenerationAttention._query_generation_attention_table::get_sol`.
/// `n_kv_lookup == 0` means MHA (n_kv == n); use `n` for the actual K/V head
/// count. `window_size > 0` clamps `kv_len` to `min(s-1, window_size)`.
fn generation_attention_sol_ms(
    spec: &SystemSpec,
    n_kv_lookup: u32,
    n: u32,
    head_size: u32,
    window_size: u32,
    kv_quant: KvCacheQuantMode,
    b: u32,
    s: u32,
) -> f64 {
    let n_kv = if n_kv_lookup == 0 { n } else { n_kv_lookup };
    let kv_len = if window_size > 0 {
        s.saturating_sub(1).min(window_size)
    } else {
        s.saturating_sub(1)
    };
    let quant_mode_gen_compute = if kv_quant == KvCacheQuantMode::Fp8 {
        FmhaQuantMode::Fp8.mapping().compute
    } else {
        FmhaQuantMode::Bfloat16.mapping().compute
    };
    let b_f = b as f64;
    let n_f = n as f64;
    let h_f = head_size as f64;
    let n_kv_f = n_kv as f64;
    let kv_len_f = kv_len as f64;
    let kv_mem = kv_quant.mapping().memory;
    let ops = 2.0 * b_f * n_f * h_f * 2.0 * kv_len_f;
    let mem_bytes =
        b_f * (n_f * h_f * 2.0 + 2.0 * n_kv_f * kv_len_f * h_f * kv_mem + n_f * h_f * 2.0);
    let bf16_flops = spec.gpu.bfloat16_tc_flops.unwrap_or(0.0);
    if bf16_flops <= 0.0 {
        return 0.0;
    }
    let sol_math = ops / bf16_flops * 1000.0 / quant_mode_gen_compute.max(1e-9);
    let sol_mem = mem_bytes / spec.gpu.mem_bw * 1000.0;
    sol_math.max(sol_mem)
}

/// In-place SOL clamp for every entry in the generation-attention grid set.
fn clamp_generation_attention_grids_to_sol(spec: &SystemSpec, grids: &mut GenerationGrids) {
    if spec.gpu.bfloat16_tc_flops.unwrap_or(0.0) <= 0.0 {
        return;
    }
    for (key, grid) in grids.by_keys.iter_mut() {
        let Some(kv_quant) = kv_cache_quant_by_name(&key.kv_quant) else {
            continue;
        };
        for (&n, by_s) in grid.iter_mut() {
            for (&s, by_b) in by_s.iter_mut() {
                for (&b, latency) in by_b.iter_mut() {
                    let sol = generation_attention_sol_ms(
                        spec,
                        key.n_kv_lookup,
                        n,
                        key.head_size,
                        key.window_size,
                        kv_quant,
                        b,
                        s,
                    );
                    if sol > *latency {
                        *latency = sol;
                    }
                }
            }
        }
    }
}

fn kv_cache_quant_by_name(name: &str) -> Option<KvCacheQuantMode> {
    use KvCacheQuantMode::*;
    Some(match name {
        "bfloat16" => Bfloat16,
        "int8" => Int8,
        "fp8" => Fp8,
        _ => return None,
    })
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

    fn b200_sxm_spec() -> SystemSpec {
        let systems_yaml = PathBuf::from(REPO_ROOT_HINT)
            .join("../..")
            .join("src/aiconfigurator/systems/b200_sxm.yaml");
        SystemSpec::load(&systems_yaml).expect("b200_sxm.yaml must parse")
    }

    #[test]
    fn context_attention_exact_hit() {
        // First row of b200_sxm/vllm/0.19.0/context_attention_perf.txt:
        // batch=8 isl=16384 n=64 n_kv=1 head_dim=128 attn=bfloat16 kv=fp8 step=0 latency=19.82
        let table = AttentionTable::new(b200_vllm_data_root(), b200_sxm_spec());
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
        let table = AttentionTable::new(b200_vllm_data_root(), b200_sxm_spec());
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
        let table = AttentionTable::new(b200_vllm_data_root(), b200_sxm_spec());
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
        let table = AttentionTable::new(b200_vllm_data_root(), b200_sxm_spec());
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
        let table = AttentionTable::new(b200_vllm_data_root(), b200_sxm_spec());
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
        let table = AttentionTable::new(b200_vllm_data_root(), b200_sxm_spec());
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
