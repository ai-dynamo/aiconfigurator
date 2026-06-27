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
use super::interpolation::{
    interp_1d, interp_2d_1d_grid, interp_2d_1d_grid_strict, nearest_neighbors, Grid3,
};
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
        // Mirror Python `GenerationAttention._query_generation_attention_table`
        // `get_silicon`: average 5 interp samples over s ∈ [0.9s, 1.1s].
        //   s_min = max(1, int(s*0.9)); s_max = max(s_min, int(s*1.1))
        //   s_samples[i] = s_min + (s_max - s_min) * i // (sample_cnt - 1)
        // Each sample calls `interp_3d(n, b, s_i, ...)` which is 1-D over n
        // and bilinear over (b, s) — hence `interp_2d_1d_grid_strict(grid, n,
        // b, s_i)` with the `[n][b][s]` grid. `interp_3d` uses
        // `allow_singleton_axes=False`, so the strict variant surfaces the same
        // `_require_3d_axis_coverage` error on degenerate grids.
        let s = kv_seq_tokens;
        let s_min = ((s as f64 * 0.9) as u32).max(1);
        let s_max = ((s as f64 * 1.1) as u32).max(s_min);
        const SAMPLE_CNT: u32 = 5;
        let mut latency_sum = 0.0_f64;
        for i in 0..SAMPLE_CNT {
            // Match Python integer arithmetic: multiply before integer divide.
            let s_i = s_min + ((u64::from(s_max - s_min) * u64::from(i)) / u64::from(SAMPLE_CNT - 1)) as u32;
            latency_sum += interp_2d_1d_grid_strict(grid, n, b, s_i, "3-D bilinear interpolation")?;
        }
        Ok(latency_sum / SAMPLE_CNT as f64)
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
            // Mirror Python `GenerationAttention.load_data` order:
            //   1. clamp to SOL (`_correct_sol`)
            //   2. densify the grid (`_extrapolate` -> `extrapolate_data_grid`)
            //   3. re-clamp to SOL (interpolated points can land below SOL)
            // Context-phase attention is intentionally NOT clamped here —
            // Python's `_correct_data` historically skipped it.
            clamp_generation_attention_grids_to_sol(&self.system_spec, &mut grids);
            for grid in grids.by_keys.values_mut() {
                densify_generation_grid(grid);
            }
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
        // Grid axis order is `[n][b][s]` to match Python's `interp_3d(n, b, s)`
        // (1-D over n, bilinear over (b, s)). Nesting: num_heads -> batch_size
        // -> sequence_tokens.
        by_keys
            .entry(key)
            .or_default()
            .entry(num_heads)
            .or_default()
            .entry(row.u32(batch_size_col)?)
            .or_default()
            .entry(sequence_tokens)
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

// Extrapolation target lattices — verbatim from Python
// `aiconfigurator.sdk.operations.attention`
// (`_GENERATION_ATTENTION_TARGET_{X,Y,Z}`). For generation attention the
// grid is `data[x][y][z]` with x=n (num_heads), y=b (batch_size), z=s
// (sequence_tokens).
const GENERATION_TARGET_X: [u32; 24] = [
    1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40, 48, 56, 72, 96, 128,
];
const GENERATION_TARGET_Y: [u32; 14] =
    [1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 1024, 2048, 8192];
const GENERATION_TARGET_Z: [u32; 20] = [
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072,
    262144, 16_777_216,
];

/// In-place port of Python `interpolation.extrapolate_data_grid` for the
/// generation-attention grid (`sqrt_y_value=False`). Densifies the grid to
/// the target lattice in z -> y -> x order so later queries interpolate only.
///
/// Grid layout is `[x][y][z]` = `[n][b][s]`: x=n (outer), y=b (middle),
/// z=s (inner). The target-list filtering mirrors Python's
/// `_extrapolate`: x uses only targets `>= min(measured n)` (`filtered_x`).
fn densify_generation_grid(grid: &mut Grid3<f64>) {
    if grid.is_empty() {
        return;
    }
    let min_x = *grid.keys().next().expect("grid is non-empty");
    let filtered_x: Vec<u32> = GENERATION_TARGET_X.iter().copied().filter(|&x| x >= min_x).collect();

    // --- z-direction: for each existing x, for each existing y, fill z ---
    let x_keys: Vec<u32> = grid.keys().copied().collect();
    for x in x_keys {
        let y_keys: Vec<u32> = grid[&x].keys().copied().collect();
        for y in y_keys {
            let z_keys: Vec<u32> = grid[&x][&y].keys().copied().collect();
            if z_keys.len() <= 1 {
                continue;
            }
            for &z in GENERATION_TARGET_Z.iter() {
                if grid[&x][&y].contains_key(&z) {
                    continue;
                }
                let Ok((z_left, z_right)) = nearest_neighbors(z, &z_keys, false) else {
                    continue;
                };
                let (Some(&v_left), Some(&v_right)) =
                    (grid[&x][&y].get(&z_left), grid[&x][&y].get(&z_right))
                else {
                    continue;
                };
                let value = interp_1d(z_left as f64, z_right as f64, v_left, v_right, z as f64);
                grid.get_mut(&x).unwrap().get_mut(&y).unwrap().insert(z, value);
            }
        }
    }

    // --- y-direction: for each existing x, fill missing target y's ---
    let x_keys: Vec<u32> = grid.keys().copied().collect();
    for x in x_keys {
        for &y in GENERATION_TARGET_Y.iter() {
            if grid[&x].contains_key(&y) {
                continue;
            }
            // Re-read current y keys each iteration so freshly added y's count.
            let y_keys: Vec<u32> = grid[&x].keys().copied().collect();
            if y_keys.len() < 2 {
                break;
            }
            let Ok((y_left, y_right)) = nearest_neighbors(y, &y_keys, false) else {
                continue;
            };
            if !grid[&x].contains_key(&y_left) || !grid[&x].contains_key(&y_right) {
                continue;
            }
            // Iterate z over sorted keys of y_left, only where present in both.
            let z_list: Vec<u32> = grid[&x][&y_left].keys().copied().collect();
            let mut new_row: BTreeMap<u32, f64> = BTreeMap::new();
            for z in z_list {
                let (Some(&yl), Some(&yr)) =
                    (grid[&x][&y_left].get(&z), grid[&x][&y_right].get(&z))
                else {
                    continue;
                };
                let value = interp_1d(y_left as f64, y_right as f64, yl, yr, y as f64);
                new_row.insert(z, value);
            }
            grid.get_mut(&x).unwrap().insert(y, new_row);
        }
    }

    // --- x-direction: fill missing filtered_x slices ---
    for &x in filtered_x.iter() {
        if grid.contains_key(&x) {
            continue;
        }
        // Re-read current x keys each iteration so freshly added x's count.
        let x_keys: Vec<u32> = grid.keys().copied().collect();
        if x_keys.len() < 2 {
            break;
        }
        let Ok((x_left, x_right)) = nearest_neighbors(x, &x_keys, false) else {
            continue;
        };
        if !grid.contains_key(&x_left) || !grid.contains_key(&x_right) {
            continue;
        }
        // Iterate y over sorted keys of x_left, only where present in both.
        let y_list: Vec<u32> = grid[&x_left].keys().copied().collect();
        let mut new_slice: BTreeMap<u32, BTreeMap<u32, f64>> = BTreeMap::new();
        for y in y_list {
            if !grid[&x_left].contains_key(&y) || !grid[&x_right].contains_key(&y) {
                continue;
            }
            let z_list: Vec<u32> = grid[&x_left][&y].keys().copied().collect();
            let mut new_row: BTreeMap<u32, f64> = BTreeMap::new();
            for z in z_list {
                let (Some(&xl), Some(&xr)) =
                    (grid[&x_left][&y].get(&z), grid[&x_right][&y].get(&z))
                else {
                    continue;
                };
                let value = interp_1d(x_left as f64, x_right as f64, xl, xr, x as f64);
                new_row.insert(z, value);
            }
            new_slice.insert(y, new_row);
        }
        grid.insert(x, new_slice);
    }
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
        // Grid order is `[n][b][s]`: outer=n, middle=b, inner=s.
        for (&n, by_b) in grid.iter_mut() {
            for (&b, by_s) in by_b.iter_mut() {
                for (&s, latency) in by_s.iter_mut() {
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

    fn gb200_vllm_data_root() -> PathBuf {
        PathBuf::from(REPO_ROOT_HINT)
            .join("../..")
            .join("src/aiconfigurator/systems/data/gb200/vllm/0.19.0")
    }

    fn gb200_spec() -> SystemSpec {
        let systems_yaml = PathBuf::from(REPO_ROOT_HINT)
            .join("../..")
            .join("src/aiconfigurator/systems/gb200.yaml");
        SystemSpec::load(&systems_yaml).expect("gb200.yaml must parse")
    }

    #[test]
    fn generation_query_ragged_corner_matches_python() {
        // Pins the exact regime this parity fix targets: large batch x long kv,
        // off-measured-grid, requiring densification + 5-sample s-averaging.
        // Before the fix Rust returned ~0.3673 here (single-interp, [n][s][b]
        // grid); Python returns 0.45830706 (5-sample avg over the densified
        // [n][b][s] grid). Verified against
        // `PerfDatabase.query_generation_attention` on gb200/vllm/0.19.0.
        let table = AttentionTable::new(gb200_vllm_data_root(), gb200_spec());
        let latency = table
            .query_generation(256, 2561, 32, 8, 128, 0, KvCacheQuantMode::Bfloat16)
            .expect("ragged-corner query must succeed");
        assert!(
            (latency - 0.45830706201571336).abs() < 1e-6,
            "expected Python-parity ragged-corner latency 0.4583, got {latency}"
        );
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
    fn generation_grid_densification_matches_python() {
        // Regression guard for the ragged/extrapolation parity fix. Python's
        // `GenerationAttention.load_data` runs clamp -> extrapolate_data_grid ->
        // re-clamp; the densified grid value at (n=32, b=256, s=2048|4096) for
        // the gb200/vllm/0.19.0 bfloat16 / n_kv_lookup=8 / head=128 key must
        // match Python's pre-filled lattice (verified against a live
        // `extrapolate_data_grid` run). These s values are off-measured-grid in
        // the b=256 column, so they exercise the z-direction fill specifically.
        let table = AttentionTable::new(gb200_vllm_data_root(), gb200_spec());
        let grids = table.load_generation().expect("generation grids must load");
        let key = GenerationKey {
            kv_quant: "bfloat16".to_string(),
            n_kv_lookup: 8,
            head_size: 128,
            window_size: 0,
        };
        let grid = grids.by_keys.get(&key).expect("gb200 vllm bfloat16 key present");
        let v2048 = grid[&32][&256][&2048];
        let v4096 = grid[&32][&256][&4096];
        assert!((v2048 - 0.368_437_310_06).abs() < 1e-6, "densified s=2048 got {v2048}");
        assert!((v4096 - 0.727_775_951_23).abs() < 1e-6, "densified s=4096 got {v4096}");
    }

    #[test]
    fn generation_attention_query_matches_python() {
        // batch=32 isl=1 n=64 n_kv=4 head_dim=128 kv=fp8 step=1 (stored
        // sequence_tokens = isl + step = 2). Python's
        // `_query_generation_attention_table` averages 5 interp samples over
        // s ∈ [max(1,int(2*0.9)), max(..,int(2*1.1))] = [1, 2], i.e.
        // s_samples = [1, 1, 1, 1, 2] over the densified grid. The result is
        // therefore the 5-sample mean, not the raw step=1 leaf. Verified
        // against `PerfDatabase.query_generation_attention` (= 0.0086442669).
        let table = AttentionTable::new(b200_vllm_data_root(), b200_sxm_spec());
        let latency = table
            .query_generation(32, 2, 64, 4, 128, 0, KvCacheQuantMode::Fp8)
            .expect("query must succeed");
        assert!(
            (latency - 0.008644266923268636).abs() < 1e-9,
            "expected 5-sample-averaged latency, got {latency}"
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
