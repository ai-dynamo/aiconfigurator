// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use csv::StringRecord;

use crate::AicError;

#[derive(Clone, Debug)]
pub(crate) struct PerfDatabase {
    gemm: Vec<GemmPoint>,
    context_attention: Vec<ContextAttentionPoint>,
    generation_attention: Vec<GenerationAttentionPoint>,
}

impl PerfDatabase {
    pub(crate) fn load(
        systems_root: &Path,
        system_name: &str,
        backend: &str,
        backend_version: Option<&str>,
    ) -> Result<Self, AicError> {
        let system_path = systems_root.join(format!("{system_name}.yaml"));
        let data_dir = read_data_dir(&system_path)?;
        let backend_root = systems_root.join(data_dir).join(backend);
        let version_path = resolve_backend_version(&backend_root, backend_version)?;

        let gemm_path = version_path.join("gemm_perf.txt");
        let context_attention_path = version_path.join("context_attention_perf.txt");
        let generation_attention_path = version_path.join("generation_attention_perf.txt");

        Ok(Self {
            gemm: load_gemm_points(&gemm_path)?,
            context_attention: load_context_attention_points(&context_attention_path)?,
            generation_attention: load_generation_attention_points(&generation_attention_path)?,
        })
    }

    pub(crate) fn query_gemm(&self, quant: &str, m: u32, n: u32, k: u32) -> Result<f64, AicError> {
        let mut best: Option<(f64, f64)> = None;
        for point in self.gemm.iter().filter(|point| point.quant == quant) {
            let score = relative_distance(point.m, m)
                + relative_distance(point.n, n)
                + relative_distance(point.k, k);
            if best.map_or(true, |(best_score, _)| score < best_score) {
                best = Some((score, point.latency_ms));
            }
        }
        best.map(|(_, latency)| latency).ok_or_else(|| {
            AicError::PerfDatabase(format!(
                "no GEMM perf point for quant={quant}, m={m}, n={n}, k={k}"
            ))
        })
    }

    pub(crate) fn query_context_attention(
        &self,
        fmha_quant: &str,
        kv_cache_quant: &str,
        batch_size: u32,
        sequence_tokens: u32,
        prefix_tokens: u32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
    ) -> Result<f64, AicError> {
        let full_sequence_tokens = sequence_tokens + prefix_tokens;
        let query_kv_heads = normalized_kv_heads(num_heads, num_kv_heads);
        let mut best: Option<(f64, f64)> = None;

        for point in self.context_attention.iter().filter(|point| {
            point.fmha_quant == fmha_quant
                && point.kv_cache_quant == kv_cache_quant
                && point.num_kv_heads == query_kv_heads
                && point.head_dim == head_dim
                && point.window_size == 0
        }) {
            let score = relative_distance(point.batch_size, batch_size)
                + relative_distance(point.sequence_tokens, full_sequence_tokens)
                + relative_distance(point.num_heads, num_heads);
            if best.map_or(true, |(best_score, _)| score < best_score) {
                best = Some((score, point.latency_ms));
            }
        }

        let latency = best.map(|(_, latency)| latency).ok_or_else(|| {
            AicError::PerfDatabase(format!(
                "no context-attention perf point for fmha_quant={fmha_quant}, kv_cache_quant={kv_cache_quant}, b={batch_size}, s={full_sequence_tokens}, prefix={prefix_tokens}, n={num_heads}, n_kv={num_kv_heads}, head_dim={head_dim}"
            ))
        })?;
        let denominator = full_sequence_tokens.saturating_mul(full_sequence_tokens);
        if denominator == 0 {
            return Ok(0.0);
        }
        let numerator = denominator.saturating_sub(prefix_tokens.saturating_mul(prefix_tokens));
        Ok(latency * f64::from(numerator) / f64::from(denominator))
    }

    pub(crate) fn query_generation_attention(
        &self,
        kv_cache_quant: &str,
        batch_size: u32,
        sequence_tokens: u32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
    ) -> Result<f64, AicError> {
        let query_kv_heads = normalized_kv_heads(num_heads, num_kv_heads);
        let mut best: Option<(f64, f64)> = None;

        for point in self.generation_attention.iter().filter(|point| {
            point.kv_cache_quant == kv_cache_quant
                && point.num_kv_heads == query_kv_heads
                && point.head_dim == head_dim
                && point.window_size == 0
        }) {
            let score = relative_distance(point.batch_size, batch_size)
                + relative_distance(point.sequence_tokens, sequence_tokens)
                + relative_distance(point.num_heads, num_heads);
            if best.map_or(true, |(best_score, _)| score < best_score) {
                best = Some((score, point.latency_ms));
            }
        }

        best.map(|(_, latency)| latency).ok_or_else(|| {
            AicError::PerfDatabase(format!(
                "no generation-attention perf point for kv_cache_quant={kv_cache_quant}, b={batch_size}, s={sequence_tokens}, n={num_heads}, n_kv={num_kv_heads}, head_dim={head_dim}"
            ))
        })
    }
}

#[derive(Clone, Debug)]
struct GemmPoint {
    quant: String,
    m: u32,
    n: u32,
    k: u32,
    latency_ms: f64,
}

#[derive(Clone, Debug)]
struct ContextAttentionPoint {
    fmha_quant: String,
    kv_cache_quant: String,
    batch_size: u32,
    sequence_tokens: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    window_size: u32,
    latency_ms: f64,
}

#[derive(Clone, Debug)]
struct GenerationAttentionPoint {
    kv_cache_quant: String,
    batch_size: u32,
    sequence_tokens: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    window_size: u32,
    latency_ms: f64,
}

fn read_data_dir(system_path: &Path) -> Result<PathBuf, AicError> {
    let text = fs::read_to_string(system_path).map_err(|source| AicError::Io {
        path: system_path.to_path_buf(),
        source,
    })?;
    for line in text.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        let Some(value) = line.strip_prefix("data_dir:") else {
            continue;
        };
        let value = value.trim().trim_matches('"').trim_matches('\'');
        if !value.is_empty() {
            return Ok(PathBuf::from(value));
        }
    }
    Err(AicError::PerfDatabase(format!(
        "missing data_dir in system file {}",
        system_path.display()
    )))
}

fn resolve_backend_version(
    backend_root: &Path,
    backend_version: Option<&str>,
) -> Result<PathBuf, AicError> {
    if let Some(version) = backend_version {
        let path = backend_root.join(version);
        if path.is_dir() {
            return Ok(path);
        }
        return Err(AicError::PerfDatabase(format!(
            "backend version '{version}' not found under {}",
            backend_root.display()
        )));
    }

    let mut versions = Vec::new();
    let entries = fs::read_dir(backend_root).map_err(|source| AicError::Io {
        path: backend_root.to_path_buf(),
        source,
    })?;
    for entry in entries {
        let entry = entry.map_err(|source| AicError::Io {
            path: backend_root.to_path_buf(),
            source,
        })?;
        let path = entry.path();
        if path.is_dir() && !path.join("INCOMPLETE.txt").exists() {
            versions.push(path);
        }
    }
    versions.sort();
    versions.pop().ok_or_else(|| {
        AicError::PerfDatabase(format!(
            "no complete backend versions under {}",
            backend_root.display()
        ))
    })
}

fn load_gemm_points(path: &Path) -> Result<Vec<GemmPoint>, AicError> {
    ensure_real_perf_file(path)?;
    let mut reader = csv::ReaderBuilder::new()
        .trim(csv::Trim::All)
        .from_path(path)
        .map_err(|source| AicError::Csv {
            path: path.to_path_buf(),
            source,
        })?;
    let headers = header_map(reader.headers().map_err(|source| AicError::Csv {
        path: path.to_path_buf(),
        source,
    })?);
    let mut points = Vec::new();
    for record in reader.records() {
        let record = record.map_err(|source| AicError::Csv {
            path: path.to_path_buf(),
            source,
        })?;
        let quant = required_field(&record, &headers, "gemm_dtype", path)?.to_string();
        if quant == "awq" || quant == "gptq" {
            continue;
        }
        points.push(GemmPoint {
            quant,
            m: parse_u32(&record, &headers, "m", path)?,
            n: parse_u32(&record, &headers, "n", path)?,
            k: parse_u32(&record, &headers, "k", path)?,
            latency_ms: parse_f64(&record, &headers, "latency", path)?,
        });
    }
    if points.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "no GEMM rows loaded from {}",
            path.display()
        )));
    }
    Ok(points)
}

fn load_context_attention_points(path: &Path) -> Result<Vec<ContextAttentionPoint>, AicError> {
    ensure_real_perf_file(path)?;
    let mut reader = csv::ReaderBuilder::new()
        .trim(csv::Trim::All)
        .from_path(path)
        .map_err(|source| AicError::Csv {
            path: path.to_path_buf(),
            source,
        })?;
    let headers = header_map(reader.headers().map_err(|source| AicError::Csv {
        path: path.to_path_buf(),
        source,
    })?);
    let mut points = Vec::new();
    for record in reader.records() {
        let record = record.map_err(|source| AicError::Csv {
            path: path.to_path_buf(),
            source,
        })?;
        let num_heads = parse_u32(&record, &headers, "num_heads", path)?;
        let num_kv_heads = parse_u32(&record, &headers, "num_key_value_heads", path)?;
        points.push(ContextAttentionPoint {
            fmha_quant: required_field(&record, &headers, "attn_dtype", path)?.to_string(),
            kv_cache_quant: required_field(&record, &headers, "kv_cache_dtype", path)?.to_string(),
            batch_size: parse_u32(&record, &headers, "batch_size", path)?,
            sequence_tokens: parse_u32(&record, &headers, "isl", path)?,
            num_heads,
            num_kv_heads: normalized_kv_heads(num_heads, num_kv_heads),
            head_dim: parse_u32(&record, &headers, "head_dim", path)?,
            window_size: parse_optional_u32(&record, &headers, "window_size", path)?.unwrap_or(0),
            latency_ms: parse_f64(&record, &headers, "latency", path)?,
        });
    }
    if points.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "no context-attention rows loaded from {}",
            path.display()
        )));
    }
    Ok(points)
}

fn load_generation_attention_points(
    path: &Path,
) -> Result<Vec<GenerationAttentionPoint>, AicError> {
    ensure_real_perf_file(path)?;
    let mut reader = csv::ReaderBuilder::new()
        .trim(csv::Trim::All)
        .from_path(path)
        .map_err(|source| AicError::Csv {
            path: path.to_path_buf(),
            source,
        })?;
    let headers = header_map(reader.headers().map_err(|source| AicError::Csv {
        path: path.to_path_buf(),
        source,
    })?);
    let mut points = Vec::new();
    for record in reader.records() {
        let record = record.map_err(|source| AicError::Csv {
            path: path.to_path_buf(),
            source,
        })?;
        let num_heads = parse_u32(&record, &headers, "num_heads", path)?;
        let num_kv_heads = parse_u32(&record, &headers, "num_key_value_heads", path)?;
        let sequence_tokens = parse_u32(&record, &headers, "isl", path)?
            + parse_u32(&record, &headers, "step", path)?;
        points.push(GenerationAttentionPoint {
            kv_cache_quant: required_field(&record, &headers, "kv_cache_dtype", path)?.to_string(),
            batch_size: parse_u32(&record, &headers, "batch_size", path)?,
            sequence_tokens,
            num_heads,
            num_kv_heads: normalized_kv_heads(num_heads, num_kv_heads),
            head_dim: parse_u32(&record, &headers, "head_dim", path)?,
            window_size: parse_optional_u32(&record, &headers, "window_size", path)?.unwrap_or(0),
            latency_ms: parse_f64(&record, &headers, "latency", path)?,
        });
    }
    if points.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "no generation-attention rows loaded from {}",
            path.display()
        )));
    }
    Ok(points)
}

fn ensure_real_perf_file(path: &Path) -> Result<(), AicError> {
    let text = fs::read_to_string(path).map_err(|source| AicError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    if text.starts_with("version https://git-lfs.github.com/spec/v1") {
        return Err(AicError::PerfDatabase(format!(
            "{} is a Git LFS pointer; run `git lfs pull` before using the Rust engine-step estimator with repository perf data",
            path.display()
        )));
    }
    Ok(())
}

fn header_map(headers: &StringRecord) -> HashMap<String, usize> {
    headers
        .iter()
        .enumerate()
        .map(|(idx, name)| (name.to_string(), idx))
        .collect()
}

fn required_field<'a>(
    record: &'a StringRecord,
    headers: &HashMap<String, usize>,
    name: &str,
    path: &Path,
) -> Result<&'a str, AicError> {
    let Some(idx) = headers.get(name) else {
        return Err(AicError::PerfDatabase(format!(
            "missing column '{name}' in {}",
            path.display()
        )));
    };
    record.get(*idx).ok_or_else(|| {
        AicError::PerfDatabase(format!(
            "missing value for column '{name}' in {}",
            path.display()
        ))
    })
}

fn parse_u32(
    record: &StringRecord,
    headers: &HashMap<String, usize>,
    name: &str,
    path: &Path,
) -> Result<u32, AicError> {
    let raw = required_field(record, headers, name, path)?;
    raw.parse::<u32>().map_err(|source| {
        AicError::PerfDatabase(format!(
            "invalid u32 value '{raw}' for column '{name}' in {}: {source}",
            path.display()
        ))
    })
}

fn parse_optional_u32(
    record: &StringRecord,
    headers: &HashMap<String, usize>,
    name: &str,
    path: &Path,
) -> Result<Option<u32>, AicError> {
    let Some(idx) = headers.get(name) else {
        return Ok(None);
    };
    let raw = record.get(*idx).unwrap_or("").trim();
    if raw.is_empty() {
        return Ok(None);
    }
    raw.parse::<u32>().map(Some).map_err(|source| {
        AicError::PerfDatabase(format!(
            "invalid u32 value '{raw}' for column '{name}' in {}: {source}",
            path.display()
        ))
    })
}

fn parse_f64(
    record: &StringRecord,
    headers: &HashMap<String, usize>,
    name: &str,
    path: &Path,
) -> Result<f64, AicError> {
    let raw = required_field(record, headers, name, path)?;
    raw.parse::<f64>().map_err(|source| {
        AicError::PerfDatabase(format!(
            "invalid f64 value '{raw}' for column '{name}' in {}: {source}",
            path.display()
        ))
    })
}

fn normalized_kv_heads(num_heads: u32, num_kv_heads: u32) -> u32 {
    if num_heads == num_kv_heads {
        0
    } else {
        num_kv_heads
    }
}

fn relative_distance(actual: u32, desired: u32) -> f64 {
    if actual == desired {
        return 0.0;
    }
    let scale = desired.max(actual).max(1) as f64;
    f64::from(actual.abs_diff(desired)) / scale
}
