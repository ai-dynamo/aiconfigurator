// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cmp::Ordering;
use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use csv::StringRecord;

use crate::AicError;

// Prefer exact MoE distribution matches; uniform is a cheap fallback, while
// mismatched shaped distributions should be selected only when closer overall.
const DIST_PENALTY_SAME: f64 = 0.0;
const DIST_PENALTY_UNIFORM: f64 = 0.05;
const DIST_PENALTY_OTHER: f64 = 0.25;

#[derive(Clone, Debug)]
pub(crate) struct PerfDatabase {
    system: SystemSpec,
    gemm: Vec<GemmPoint>,
    context_attention: Vec<ContextAttentionPoint>,
    generation_attention: Vec<GenerationAttentionPoint>,
    custom_allreduce: Vec<CustomAllReducePoint>,
    nccl: Vec<NcclPoint>,
    moe: Vec<MoePoint>,
    context_mla: Vec<ContextMlaPoint>,
    generation_mla: Vec<GenerationMlaPoint>,
    query_cache: Arc<Mutex<QueryCache>>,
}

#[derive(Clone, Debug, Default)]
struct QueryCache {
    gemm: HashMap<(String, u32, u32, u32), f64>,
    context_attention: HashMap<(String, String, u32, u32, u32, u32, u32, u32), f64>,
    generation_attention: HashMap<(String, u32, u32, u32, u32, u32), f64>,
    custom_allreduce: HashMap<(u32, u64), f64>,
    nccl: HashMap<(String, String, u32, u64), f64>,
    moe: HashMap<(String, u32, u32, u32, u32, u32, u32, u32, String), Option<f64>>,
    context_mla: HashMap<(String, String, u32, u32, u32), Option<f64>>,
    generation_mla: HashMap<(String, u32, u32, u32), Option<f64>>,
}

impl PerfDatabase {
    pub(crate) fn load(
        systems_root: &Path,
        system_name: &str,
        backend: &str,
        backend_version: Option<&str>,
    ) -> Result<Self, AicError> {
        let system_path = systems_root.join(format!("{system_name}.yaml"));
        let system = read_system_spec(&system_path)?;
        let backend_root = systems_root.join(&system.data_dir).join(backend);
        let version_path = resolve_backend_version(&backend_root, backend_version)?;

        let gemm_path = version_path.join("gemm_perf.txt");
        let context_attention_path = version_path.join("context_attention_perf.txt");
        let generation_attention_path = version_path.join("generation_attention_perf.txt");
        let custom_allreduce_path = version_path.join("custom_allreduce_perf.txt");
        let nccl_path = systems_root
            .join(&system.data_dir)
            .join("nccl")
            .join(&system.nccl_version)
            .join("nccl_perf.txt");
        let moe_path = version_path.join("moe_perf.txt");
        let context_mla_path = version_path.join("context_mla_perf.txt");
        let generation_mla_path = version_path.join("generation_mla_perf.txt");

        Ok(Self {
            system,
            gemm: load_gemm_points(&gemm_path)?,
            context_attention: load_context_attention_points(&context_attention_path)?,
            generation_attention: load_generation_attention_points(&generation_attention_path)?,
            custom_allreduce: load_optional_custom_allreduce_points(&custom_allreduce_path)?,
            nccl: load_optional_nccl_points(&nccl_path)?,
            moe: load_optional_moe_points(&moe_path)?,
            context_mla: load_optional_context_mla_points(&context_mla_path)?,
            generation_mla: load_optional_generation_mla_points(&generation_mla_path)?,
            query_cache: Arc::new(Mutex::new(QueryCache::default())),
        })
    }

    // TODO(remove-after-rust-migration): parity check/benchmark-only cache reset.
    pub(crate) fn clear_query_cache(&self) {
        if let Ok(mut cache) = self.query_cache.lock() {
            *cache = QueryCache::default();
        }
    }

    pub(crate) fn query_gemm(&self, quant: &str, m: u32, n: u32, k: u32) -> Result<f64, AicError> {
        let key = (quant.to_string(), m, n, k);
        if let Ok(cache) = self.query_cache.lock() {
            if let Some(latency) = cache.gemm.get(&key) {
                return Ok(*latency);
            }
        }

        let mut same_shape_points = Vec::new();
        let mut same_quant_points = Vec::new();
        for point in self
            .gemm
            .iter()
            .filter(|point| point.quant == quant && point.n == n && point.k == k)
        {
            same_shape_points.push((point.m, point.latency_ms));
        }
        for point in self.gemm.iter().filter(|point| point.quant == quant) {
            same_quant_points.push((point.m, point.n, point.k, point.latency_ms));
        }
        if let Some(latency) = interpolate_1d_latency(&same_shape_points, m) {
            if let Ok(mut cache) = self.query_cache.lock() {
                cache.gemm.insert(key, latency);
            }
            return Ok(latency);
        }
        if let Some(latency) = interpolate_3d_latency(&same_quant_points, m, n, k) {
            if let Ok(mut cache) = self.query_cache.lock() {
                cache.gemm.insert(key, latency);
            }
            return Ok(latency);
        }

        let mut best: Option<(f64, f64)> = None;
        for point in self.gemm.iter().filter(|point| point.quant == quant) {
            let score = relative_distance(point.m, m)
                + relative_distance(point.n, n)
                + relative_distance(point.k, k);
            if best.map_or(true, |(best_score, _)| score < best_score) {
                best = Some((score, point.latency_ms));
            }
        }
        let latency = best.map(|(_, latency)| latency).ok_or_else(|| {
            AicError::PerfDatabase(format!(
                "no GEMM perf point for quant={quant}, m={m}, n={n}, k={k}"
            ))
        })?;
        if let Ok(mut cache) = self.query_cache.lock() {
            cache.gemm.insert(key, latency);
        }
        Ok(latency)
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
        let key = (
            fmha_quant.to_string(),
            kv_cache_quant.to_string(),
            batch_size,
            sequence_tokens,
            prefix_tokens,
            num_heads,
            num_kv_heads,
            head_dim,
        );
        if let Ok(cache) = self.query_cache.lock() {
            if let Some(latency) = cache.context_attention.get(&key) {
                return Ok(*latency);
            }
        }

        let full_sequence_tokens = sequence_tokens.saturating_add(prefix_tokens);
        let query_kv_heads = normalized_kv_heads(num_heads, num_kv_heads);
        let mut same_heads_points = Vec::new();
        let mut same_shape_batch_points = Vec::new();
        let mut same_shape_sequence_points = Vec::new();
        let mut best: Option<(f64, f64)> = None;

        for point in self.context_attention.iter().filter(|point| {
            point.fmha_quant == fmha_quant
                && point.kv_cache_quant == kv_cache_quant
                && point.num_kv_heads == query_kv_heads
                && point.head_dim == head_dim
                && point.window_size == 0
        }) {
            if point.num_heads == num_heads {
                same_heads_points.push((point.sequence_tokens, point.batch_size, point.latency_ms));
            }
            if point.num_heads == num_heads && point.sequence_tokens == full_sequence_tokens {
                same_shape_batch_points.push((point.batch_size, point.latency_ms));
            }
            if point.num_heads == num_heads && point.batch_size == batch_size {
                same_shape_sequence_points.push((point.sequence_tokens, point.latency_ms));
            }
            let score = relative_distance(point.batch_size, batch_size)
                + relative_distance(point.sequence_tokens, full_sequence_tokens)
                + relative_distance(point.num_heads, num_heads);
            if best.map_or(true, |(best_score, _)| score < best_score) {
                best = Some((score, point.latency_ms));
            }
        }

        let latency = interpolate_2d_latency(&same_heads_points, full_sequence_tokens, batch_size)
            .or_else(|| interpolate_1d_latency(&same_shape_batch_points, batch_size))
            .or_else(|| interpolate_1d_latency(&same_shape_sequence_points, full_sequence_tokens))
            .or_else(|| best.map(|(_, latency)| latency))
            .ok_or_else(|| {
            AicError::PerfDatabase(format!(
                "no context-attention perf point for fmha_quant={fmha_quant}, kv_cache_quant={kv_cache_quant}, b={batch_size}, s={full_sequence_tokens}, prefix={prefix_tokens}, n={num_heads}, n_kv={num_kv_heads}, head_dim={head_dim}"
            ))
        })?;
        let full_sequence_tokens_f = f64::from(sequence_tokens) + f64::from(prefix_tokens);
        if full_sequence_tokens_f == 0.0 {
            return Ok(0.0);
        }
        let prefix_tokens_f = f64::from(prefix_tokens);
        let denominator = full_sequence_tokens_f * full_sequence_tokens_f;
        let numerator = (denominator - prefix_tokens_f * prefix_tokens_f).max(0.0);
        let latency = latency * numerator / denominator;
        if let Ok(mut cache) = self.query_cache.lock() {
            cache.context_attention.insert(key, latency);
        }
        Ok(latency)
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
        let key = (
            kv_cache_quant.to_string(),
            batch_size,
            sequence_tokens,
            num_heads,
            num_kv_heads,
            head_dim,
        );
        if let Ok(cache) = self.query_cache.lock() {
            if let Some(latency) = cache.generation_attention.get(&key) {
                return Ok(*latency);
            }
        }

        let query_kv_heads = normalized_kv_heads(num_heads, num_kv_heads);
        let mut same_heads_points = Vec::new();
        let mut same_shape_batch_points = Vec::new();
        let mut same_shape_sequence_points = Vec::new();
        let mut best: Option<(f64, f64)> = None;

        for point in self.generation_attention.iter().filter(|point| {
            point.kv_cache_quant == kv_cache_quant
                && point.num_kv_heads == query_kv_heads
                && point.head_dim == head_dim
                && point.window_size == 0
        }) {
            if point.num_heads == num_heads {
                same_heads_points.push((point.batch_size, point.sequence_tokens, point.latency_ms));
            }
            if point.num_heads == num_heads && point.sequence_tokens == sequence_tokens {
                same_shape_batch_points.push((point.batch_size, point.latency_ms));
            }
            if point.num_heads == num_heads && point.batch_size == batch_size {
                same_shape_sequence_points.push((point.sequence_tokens, point.latency_ms));
            }
            let score = relative_distance(point.batch_size, batch_size)
                + relative_distance(point.sequence_tokens, sequence_tokens)
                + relative_distance(point.num_heads, num_heads);
            if best.map_or(true, |(best_score, _)| score < best_score) {
                best = Some((score, point.latency_ms));
            }
        }

        let sampled_latency =
            average_generation_attention_latency(&same_heads_points, batch_size, sequence_tokens);
        let latency = sampled_latency
            .or_else(|| interpolate_1d_latency(&same_shape_batch_points, batch_size))
            .or_else(|| interpolate_1d_latency(&same_shape_sequence_points, sequence_tokens))
            .or_else(|| best.map(|(_, latency)| latency))
            .ok_or_else(|| {
            AicError::PerfDatabase(format!(
                "no generation-attention perf point for kv_cache_quant={kv_cache_quant}, b={batch_size}, s={sequence_tokens}, n={num_heads}, n_kv={num_kv_heads}, head_dim={head_dim}"
            ))
        })?;
        if let Ok(mut cache) = self.query_cache.lock() {
            cache.generation_attention.insert(key, latency);
        }
        Ok(latency)
    }

    pub(crate) fn query_mem_op(&self, mem_bytes: u64) -> f64 {
        (mem_bytes as f64 / (self.system.mem_bw * self.system.mem_bw_empirical_scaling_factor)
            + self.system.mem_empirical_constant_latency)
            * 1000.0
    }

    pub(crate) fn query_p2p(&self, message_bytes: u64) -> f64 {
        (message_bytes as f64 / self.system.inter_node_bw + self.system.p2p_latency) * 1000.0
    }

    pub(crate) fn query_custom_allreduce(&self, tp_size: u32, size: u64) -> f64 {
        if tp_size <= 1 {
            return 0.0;
        }
        let key = (tp_size, size);
        if let Ok(cache) = self.query_cache.lock() {
            if let Some(latency) = cache.custom_allreduce.get(&key) {
                return *latency;
            }
        }

        let effective_tp = tp_size.min(self.system.num_gpus_per_node.max(1));
        let mut points = Vec::new();
        for point in self
            .custom_allreduce
            .iter()
            .filter(|point| point.tp_size == effective_tp)
        {
            points.push((point.message_size, point.latency_ms));
        }
        let latency = interpolate_1d_latency_u64(&points, size)
            .map(|latency| {
                self.scale_collective_latency_for_gpu_count(latency, effective_tp, tp_size)
            })
            .unwrap_or_else(|| self.custom_allreduce_empirical(tp_size, size));
        if let Ok(mut cache) = self.query_cache.lock() {
            cache.custom_allreduce.insert(key, latency);
        }
        latency
    }

    pub(crate) fn query_nccl(&self, quant: &str, num_gpus: u32, operation: &str, size: u64) -> f64 {
        if num_gpus <= 1 {
            return 0.0;
        }
        let key = (quant.to_string(), operation.to_string(), num_gpus, size);
        if let Ok(cache) = self.query_cache.lock() {
            if let Some(latency) = cache.nccl.get(&key) {
                return *latency;
            }
        }

        let mut available_gpu_counts = self
            .nccl
            .iter()
            .filter(|point| point.quant == quant && point.operation == operation)
            .map(|point| point.num_gpus)
            .collect::<Vec<_>>();
        available_gpu_counts.sort_unstable();
        available_gpu_counts.dedup();
        let effective_gpus = available_gpu_counts
            .iter()
            .copied()
            .filter(|count| *count <= num_gpus)
            .max()
            .or_else(|| available_gpu_counts.first().copied());

        let latency = effective_gpus
            .and_then(|effective_gpus| {
                let mut points = Vec::new();
                for point in self.nccl.iter().filter(|point| {
                    point.quant == quant
                        && point.operation == operation
                        && point.num_gpus == effective_gpus
                }) {
                    points.push((point.message_size, point.latency_ms));
                }
                interpolate_1d_latency_u64(&points, size).map(|latency| {
                    if num_gpus > effective_gpus && effective_gpus > 1 {
                        let max_bw = self.p2p_bandwidth(effective_gpus);
                        let target_bw = self.p2p_bandwidth(num_gpus);
                        latency
                            * collective_gpu_count_scale(effective_gpus, num_gpus)
                            * (max_bw / target_bw)
                    } else {
                        latency
                    }
                })
            })
            .unwrap_or_else(|| self.nccl_empirical(quant, num_gpus, operation, size));
        if let Ok(mut cache) = self.query_cache.lock() {
            cache.nccl.insert(key, latency);
        }
        latency
    }

    fn custom_allreduce_empirical(&self, tp_size: u32, size: u64) -> f64 {
        if tp_size <= 1 {
            return 0.0;
        }
        let tp = f64::from(tp_size);
        let p2p_bw = if tp_size <= self.system.num_gpus_per_node {
            self.system.intra_node_bw
        } else {
            self.system.inter_node_bw
        };
        let sol_ms = 2.0 * size as f64 * 2.0 / tp * (tp - 1.0) / p2p_bw * 1000.0;
        sol_ms / 0.8
    }

    fn nccl_empirical(&self, quant: &str, num_gpus: u32, operation: &str, size: u64) -> f64 {
        let dtype_bytes = match quant {
            "fp8" | "fp8_block" | "int8" => 1.0,
            "nvfp4" | "fp4" => 0.5,
            _ => 2.0,
        };
        let multiplier = if operation == "all_reduce" { 2.0 } else { 1.0 };
        let sol_ms = multiplier * dtype_bytes * size as f64 * f64::from(num_gpus.saturating_sub(1))
            / f64::from(num_gpus.max(1))
            / self.p2p_bandwidth(num_gpus)
            * 1000.0;
        sol_ms / 0.8
    }

    fn p2p_bandwidth(&self, num_gpus: u32) -> f64 {
        if num_gpus <= self.system.num_gpus_per_node {
            self.system.intra_node_bw
        } else {
            self.system.inter_node_bw
        }
    }

    fn scale_collective_latency_for_gpu_count(
        &self,
        latency: f64,
        measured_gpus: u32,
        requested_gpus: u32,
    ) -> f64 {
        if requested_gpus <= measured_gpus || measured_gpus <= 1 {
            return latency;
        }
        let measured_bw = self.p2p_bandwidth(measured_gpus);
        let requested_bw = self.p2p_bandwidth(requested_gpus);
        latency
            * collective_gpu_count_scale(measured_gpus, requested_gpus)
            * (measured_bw / requested_bw)
    }

    pub(crate) fn query_moe(
        &self,
        quant: &str,
        num_tokens: u32,
        hidden_size: u32,
        inter_size: u32,
        top_k: u32,
        num_experts: u32,
        moe_tp_size: u32,
        moe_ep_size: u32,
        distribution: &str,
    ) -> Option<f64> {
        let key = (
            quant.to_string(),
            num_tokens,
            hidden_size,
            inter_size,
            top_k,
            num_experts,
            moe_tp_size,
            moe_ep_size,
            distribution.to_string(),
        );
        if let Ok(cache) = self.query_cache.lock() {
            if let Some(latency) = cache.moe.get(&key) {
                return *latency;
            }
        }

        let requested_distribution_available = self
            .moe
            .iter()
            .any(|point| point.quant == quant && point.distribution == distribution);
        let used_distribution = if requested_distribution_available {
            distribution
        } else {
            "uniform"
        };
        let mut same_shape_points = Vec::new();
        let mut best: Option<(f64, f64)> = None;
        for point in self.moe.iter().filter(|point| point.quant == quant) {
            let distribution_penalty = if point.distribution == used_distribution {
                DIST_PENALTY_SAME
            } else if point.distribution == "uniform" {
                DIST_PENALTY_UNIFORM
            } else {
                DIST_PENALTY_OTHER
            };
            let score = distribution_penalty
                + relative_distance(point.num_tokens, num_tokens)
                + relative_distance(point.hidden_size, hidden_size)
                + relative_distance(point.inter_size, inter_size)
                + relative_distance(point.top_k, top_k)
                + relative_distance(point.num_experts, num_experts)
                + relative_distance(point.moe_tp_size, moe_tp_size)
                + relative_distance(point.moe_ep_size, moe_ep_size);
            if best.map_or(true, |(best_score, _)| score < best_score) {
                best = Some((score, point.latency_ms));
            }
            if point.distribution == used_distribution
                && point.hidden_size == hidden_size
                && point.inter_size == inter_size
                && point.top_k == top_k
                && point.num_experts == num_experts
                && point.moe_tp_size == moe_tp_size
                && point.moe_ep_size == moe_ep_size
            {
                same_shape_points.push((point.num_tokens, point.latency_ms));
            }
        }
        let latency = interpolate_1d_latency(&same_shape_points, num_tokens)
            .or_else(|| best.map(|(_, latency)| latency));
        if let Ok(mut cache) = self.query_cache.lock() {
            cache.moe.insert(key, latency);
        }
        latency
    }

    pub(crate) fn query_context_mla(
        &self,
        fmha_quant: &str,
        kv_cache_quant: &str,
        batch_size: u32,
        sequence_tokens: u32,
        num_heads: u32,
    ) -> Option<f64> {
        let key = (
            fmha_quant.to_string(),
            kv_cache_quant.to_string(),
            batch_size,
            sequence_tokens,
            num_heads,
        );
        if let Ok(cache) = self.query_cache.lock() {
            if let Some(latency) = cache.context_mla.get(&key) {
                return *latency;
            }
        }

        let mut best: Option<(f64, f64)> = None;
        for point in self.context_mla.iter().filter(|point| {
            point.fmha_quant == fmha_quant && point.kv_cache_quant == kv_cache_quant
        }) {
            let score = relative_distance(point.batch_size, batch_size)
                + relative_distance(point.sequence_tokens, sequence_tokens)
                + relative_distance(point.num_heads, num_heads);
            if best.map_or(true, |(best_score, _)| score < best_score) {
                best = Some((score, point.latency_ms));
            }
        }
        let latency = best.map(|(_, latency)| latency);
        if let Ok(mut cache) = self.query_cache.lock() {
            cache.context_mla.insert(key, latency);
        }
        latency
    }

    pub(crate) fn query_generation_mla(
        &self,
        kv_cache_quant: &str,
        batch_size: u32,
        sequence_tokens: u32,
        num_heads: u32,
    ) -> Option<f64> {
        let key = (
            kv_cache_quant.to_string(),
            batch_size,
            sequence_tokens,
            num_heads,
        );
        if let Ok(cache) = self.query_cache.lock() {
            if let Some(latency) = cache.generation_mla.get(&key) {
                return *latency;
            }
        }

        let mut best: Option<(f64, f64)> = None;
        for point in self
            .generation_mla
            .iter()
            .filter(|point| point.kv_cache_quant == kv_cache_quant)
        {
            let score = relative_distance(point.batch_size, batch_size)
                + relative_distance(point.sequence_tokens, sequence_tokens)
                + relative_distance(point.num_heads, num_heads);
            if best.map_or(true, |(best_score, _)| score < best_score) {
                best = Some((score, point.latency_ms));
            }
        }
        let latency = best.map(|(_, latency)| latency);
        if let Ok(mut cache) = self.query_cache.lock() {
            cache.generation_mla.insert(key, latency);
        }
        latency
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

#[derive(Clone, Debug)]
struct MoePoint {
    quant: String,
    num_tokens: u32,
    hidden_size: u32,
    inter_size: u32,
    top_k: u32,
    num_experts: u32,
    moe_tp_size: u32,
    moe_ep_size: u32,
    distribution: String,
    latency_ms: f64,
}

#[derive(Clone, Debug)]
struct ContextMlaPoint {
    fmha_quant: String,
    kv_cache_quant: String,
    batch_size: u32,
    sequence_tokens: u32,
    num_heads: u32,
    latency_ms: f64,
}

#[derive(Clone, Debug)]
struct GenerationMlaPoint {
    kv_cache_quant: String,
    batch_size: u32,
    sequence_tokens: u32,
    num_heads: u32,
    latency_ms: f64,
}

#[derive(Clone, Debug)]
struct CustomAllReducePoint {
    tp_size: u32,
    message_size: u64,
    latency_ms: f64,
}

#[derive(Clone, Debug)]
struct NcclPoint {
    quant: String,
    operation: String,
    num_gpus: u32,
    message_size: u64,
    latency_ms: f64,
}

#[derive(Clone, Debug)]
struct SystemSpec {
    data_dir: PathBuf,
    nccl_version: String,
    mem_bw: f64,
    mem_bw_empirical_scaling_factor: f64,
    mem_empirical_constant_latency: f64,
    num_gpus_per_node: u32,
    inter_node_bw: f64,
    intra_node_bw: f64,
    p2p_latency: f64,
}

fn read_system_spec(system_path: &Path) -> Result<SystemSpec, AicError> {
    let text = fs::read_to_string(system_path).map_err(|source| AicError::Io {
        path: system_path.to_path_buf(),
        source,
    })?;
    let mut data_dir = None;
    let mut nccl_version = None;
    let mut section = "";
    let mut mem_bw = None;
    let mut mem_bw_empirical_scaling_factor = None;
    let mut mem_empirical_constant_latency = None;
    let mut num_gpus_per_node = None;
    let mut inter_node_bw = None;
    let mut intra_node_bw = None;
    let mut p2p_latency = None;

    for line in text.lines() {
        let raw = line.split('#').next().unwrap_or("");
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        if !raw.starts_with(' ') && line.ends_with(':') {
            section = line.trim_end_matches(':');
            continue;
        }
        if let Some(value) = line.strip_prefix("data_dir:") {
            data_dir = Some(PathBuf::from(clean_yaml_scalar(value)));
            continue;
        }
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        let value = clean_yaml_scalar(value);
        match (section, name.trim()) {
            ("gpu", "mem_bw") => {
                mem_bw = Some(parse_required_yaml_f64(system_path, "gpu.mem_bw", &value)?)
            }
            ("gpu", "mem_bw_empirical_scaling_factor") => {
                mem_bw_empirical_scaling_factor = parse_yaml_f64(&value)
            }
            ("gpu", "mem_empirical_constant_latency") => {
                mem_empirical_constant_latency = parse_yaml_f64(&value)
            }
            ("node", "num_gpus_per_node") => {
                num_gpus_per_node = Some(parse_required_yaml_u32(
                    system_path,
                    "node.num_gpus_per_node",
                    &value,
                )?)
            }
            ("node", "inter_node_bw") => {
                inter_node_bw = Some(parse_required_yaml_f64(
                    system_path,
                    "node.inter_node_bw",
                    &value,
                )?)
            }
            ("node", "intra_node_bw") => {
                intra_node_bw = Some(parse_required_yaml_f64(
                    system_path,
                    "node.intra_node_bw",
                    &value,
                )?)
            }
            ("node", "p2p_latency") => p2p_latency = parse_yaml_f64(&value),
            ("misc", "nccl_version") => nccl_version = Some(value),
            _ => {}
        }
    }
    Ok(SystemSpec {
        data_dir: data_dir.ok_or_else(|| {
            AicError::PerfDatabase(format!(
                "missing data_dir in system file {}",
                system_path.display()
            ))
        })?,
        nccl_version: nccl_version.unwrap_or_default(),
        mem_bw: require_system_field(system_path, "gpu.mem_bw", mem_bw)?,
        mem_bw_empirical_scaling_factor: mem_bw_empirical_scaling_factor.unwrap_or(1.0),
        mem_empirical_constant_latency: mem_empirical_constant_latency.unwrap_or(0.0),
        num_gpus_per_node: require_system_field(
            system_path,
            "node.num_gpus_per_node",
            num_gpus_per_node,
        )?,
        inter_node_bw: require_system_field(system_path, "node.inter_node_bw", inter_node_bw)?,
        intra_node_bw: require_system_field(system_path, "node.intra_node_bw", intra_node_bw)?,
        p2p_latency: p2p_latency.unwrap_or(0.0),
    })
}

fn clean_yaml_scalar(value: &str) -> String {
    value
        .trim()
        .trim_matches('"')
        .trim_matches('\'')
        .to_string()
}

fn parse_yaml_f64(value: &str) -> Option<f64> {
    value.split_whitespace().next()?.parse::<f64>().ok()
}

fn parse_required_yaml_f64(system_path: &Path, field: &str, value: &str) -> Result<f64, AicError> {
    value
        .split_whitespace()
        .next()
        .ok_or_else(|| missing_system_field_error(system_path, field))?
        .parse::<f64>()
        .map_err(|_| invalid_system_field_error(system_path, field, value))
}

fn parse_required_yaml_u32(system_path: &Path, field: &str, value: &str) -> Result<u32, AicError> {
    value
        .split_whitespace()
        .next()
        .ok_or_else(|| missing_system_field_error(system_path, field))?
        .parse::<u32>()
        .map_err(|_| invalid_system_field_error(system_path, field, value))
}

fn require_system_field<T>(
    system_path: &Path,
    field: &str,
    value: Option<T>,
) -> Result<T, AicError> {
    value.ok_or_else(|| missing_system_field_error(system_path, field))
}

fn missing_system_field_error(system_path: &Path, field: &str) -> AicError {
    AicError::PerfDatabase(format!(
        "missing {field} in system file {}",
        system_path.display()
    ))
}

fn invalid_system_field_error(system_path: &Path, field: &str, value: &str) -> AicError {
    AicError::PerfDatabase(format!(
        "invalid {field} value {value:?} in system file {}",
        system_path.display()
    ))
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
    versions.sort_by(|left, right| compare_backend_version_paths(left, right));
    versions.pop().ok_or_else(|| {
        AicError::PerfDatabase(format!(
            "no complete backend versions under {}",
            backend_root.display()
        ))
    })
}

fn compare_backend_version_paths(left: &Path, right: &Path) -> Ordering {
    let left_name = version_dir_name(left);
    let right_name = version_dir_name(right);
    compare_version_names(left_name, right_name)
        .then_with(|| left_name.cmp(right_name))
        .then_with(|| left.cmp(right))
}

fn version_dir_name(path: &Path) -> &str {
    path.file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default()
}

fn compare_version_names(left: &str, right: &str) -> Ordering {
    let left_numbers = version_number_runs(left);
    let right_numbers = version_number_runs(right);
    left_numbers.cmp(&right_numbers)
}

fn version_number_runs(version: &str) -> Vec<u64> {
    let mut numbers = Vec::new();
    let mut current = None;
    for byte in version.bytes() {
        if byte.is_ascii_digit() {
            let digit = u64::from(byte - b'0');
            current = Some(
                current
                    .unwrap_or(0_u64)
                    .saturating_mul(10)
                    .saturating_add(digit),
            );
        } else if let Some(number) = current.take() {
            numbers.push(number);
        }
    }
    if let Some(number) = current {
        numbers.push(number);
    }
    numbers
}

fn interpolate_1d_latency(points: &[(u32, f64)], query: u32) -> Option<f64> {
    if points.is_empty() {
        return None;
    }

    let mut sorted = points.to_vec();
    sorted.sort_by_key(|(x, _)| *x);
    sorted.dedup_by_key(|(x, _)| *x);

    if let Some((_, latency)) = sorted.iter().find(|(x, _)| *x == query) {
        return Some(*latency);
    }
    if sorted.len() < 2 {
        return None;
    }

    let (left, right) = if query < sorted[0].0 {
        (sorted[0], sorted[1])
    } else if query > sorted[sorted.len() - 1].0 {
        (sorted[sorted.len() - 2], sorted[sorted.len() - 1])
    } else {
        let mut bracket = None;
        for window in sorted.windows(2) {
            if query >= window[0].0 && query <= window[1].0 {
                bracket = Some((window[0], window[1]));
                break;
            }
        }
        bracket?
    };

    let (x0, y0) = (f64::from(left.0), left.1);
    let (x1, mut y1) = (f64::from(right.0), right.1);
    let x = f64::from(query);

    if x0 == x1 {
        return Some(y0);
    }
    // When extrapolating with a negative measured slope, flatten the line to
    // avoid nonsensical latency estimates; keep this in sync with the u64 path.
    if (x0 - x1) * (y0 - y1) < 0.0 && (x - x0) * (x - x1) > 0.0 {
        y1 = y0;
    }
    Some(y0 + (y1 - y0) / (x1 - x0) * (x - x0))
}

fn interpolate_1d_latency_u64(points: &[(u64, f64)], query: u64) -> Option<f64> {
    if points.is_empty() {
        return None;
    }

    let mut sorted = points.to_vec();
    sorted.sort_by_key(|(x, _)| *x);
    sorted.dedup_by_key(|(x, _)| *x);

    if let Some((_, latency)) = sorted.iter().find(|(x, _)| *x == query) {
        return Some(*latency);
    }
    if sorted.len() < 2 {
        return None;
    }

    let (left, right) = if query < sorted[0].0 {
        (sorted[0], sorted[1])
    } else if query > sorted[sorted.len() - 1].0 {
        (sorted[sorted.len() - 2], sorted[sorted.len() - 1])
    } else {
        let mut bracket = None;
        for window in sorted.windows(2) {
            if query >= window[0].0 && query <= window[1].0 {
                bracket = Some((window[0], window[1]));
                break;
            }
        }
        bracket?
    };

    let (x0, y0) = (left.0 as f64, left.1);
    let (x1, mut y1) = (right.0 as f64, right.1);
    let x = query as f64;

    if x0 == x1 {
        return Some(y0);
    }
    // Same negative-slope extrapolation guard used by interpolate_1d_latency.
    if (x0 - x1) * (y0 - y1) < 0.0 && (x - x0) * (x - x1) > 0.0 {
        y1 = y0;
    }
    Some(y0 + (y1 - y0) / (x1 - x0) * (x - x0))
}

fn interpolate_2d_latency(points: &[(u32, u32, f64)], query_x: u32, query_y: u32) -> Option<f64> {
    if points.is_empty() {
        return None;
    }
    let mut xs = points.iter().map(|(x, _, _)| *x).collect::<Vec<_>>();
    xs.sort_unstable();
    xs.dedup();
    let (x_left, x_right) = axis_pair(&xs, query_x)?;

    let left_points = points
        .iter()
        .filter(|(x, _, _)| *x == x_left)
        .map(|(_, y, latency)| (*y, *latency))
        .collect::<Vec<_>>();
    let left_latency = interpolate_1d_latency(&left_points, query_y)?;

    if x_left == x_right {
        return Some(left_latency);
    }

    let right_points = points
        .iter()
        .filter(|(x, _, _)| *x == x_right)
        .map(|(_, y, latency)| (*y, *latency))
        .collect::<Vec<_>>();
    let right_latency = interpolate_1d_latency(&right_points, query_y)?;
    interpolate_1d_latency(&[(x_left, left_latency), (x_right, right_latency)], query_x)
}

fn interpolate_3d_latency(
    points: &[(u32, u32, u32, f64)],
    query_x: u32,
    query_y: u32,
    query_z: u32,
) -> Option<f64> {
    if points.is_empty() {
        return None;
    }
    let mut xs = points.iter().map(|(x, _, _, _)| *x).collect::<Vec<_>>();
    xs.sort_unstable();
    xs.dedup();
    let (x_left, x_right) = axis_pair(&xs, query_x)?;

    let left_points = points
        .iter()
        .filter(|(x, _, _, _)| *x == x_left)
        .map(|(_, y, z, latency)| (*y, *z, *latency))
        .collect::<Vec<_>>();
    let left_latency = interpolate_2d_latency(&left_points, query_y, query_z)?;

    if x_left == x_right {
        return Some(left_latency);
    }

    let right_points = points
        .iter()
        .filter(|(x, _, _, _)| *x == x_right)
        .map(|(_, y, z, latency)| (*y, *z, *latency))
        .collect::<Vec<_>>();
    let right_latency = interpolate_2d_latency(&right_points, query_y, query_z)?;
    interpolate_1d_latency(&[(x_left, left_latency), (x_right, right_latency)], query_x)
}

fn axis_pair(sorted: &[u32], query: u32) -> Option<(u32, u32)> {
    if sorted.is_empty() {
        return None;
    }
    if sorted.contains(&query) {
        return Some((query, query));
    }
    if sorted.len() < 2 {
        return None;
    }
    if query < sorted[0] {
        return Some((sorted[0], sorted[1]));
    }
    if query > sorted[sorted.len() - 1] {
        return Some((sorted[sorted.len() - 2], sorted[sorted.len() - 1]));
    }
    for window in sorted.windows(2) {
        if query >= window[0] && query <= window[1] {
            return Some((window[0], window[1]));
        }
    }
    None
}

fn average_generation_attention_latency(
    points: &[(u32, u32, f64)],
    batch_size: u32,
    sequence_tokens: u32,
) -> Option<f64> {
    let s_min = (sequence_tokens.saturating_mul(9) / 10).max(1);
    let s_max = sequence_tokens.saturating_mul(11) / 10;
    let mut sum = 0.0;
    for i in 0..5_u32 {
        let sample = s_min + (s_max.saturating_sub(s_min)) * i / 4;
        sum += interpolate_2d_latency(points, batch_size, sample)?;
    }
    Some(sum / 5.0)
}

fn collective_gpu_count_scale(measured_gpus: u32, requested_gpus: u32) -> f64 {
    if measured_gpus <= 1 || requested_gpus <= 1 {
        return 1.0;
    }
    (f64::from(requested_gpus - 1) / f64::from(requested_gpus))
        * (f64::from(measured_gpus) / f64::from(measured_gpus - 1))
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

fn load_optional_custom_allreduce_points(
    path: &Path,
) -> Result<Vec<CustomAllReducePoint>, AicError> {
    if !path.is_file() || is_lfs_pointer(path)? {
        return Ok(Vec::new());
    }
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
    let is_b60 = path.to_string_lossy().contains("b60");
    let mut points = Vec::new();
    for record in reader.records() {
        let record = record.map_err(|source| AicError::Csv {
            path: path.to_path_buf(),
            source,
        })?;
        let kernel_source = optional_field(&record, &headers, "kernel_source").unwrap_or("");
        let backend = optional_field(&record, &headers, "backend").unwrap_or("");
        if !is_b60 && (kernel_source.ends_with("_eager") || backend.ends_with("_eager")) {
            continue;
        }
        points.push(CustomAllReducePoint {
            tp_size: parse_u32(&record, &headers, "num_gpus", path)?,
            message_size: parse_u64(&record, &headers, "message_size", path)?,
            latency_ms: parse_f64(&record, &headers, "latency", path)?,
        });
    }
    Ok(points)
}

fn load_optional_nccl_points(path: &Path) -> Result<Vec<NcclPoint>, AicError> {
    if !path.is_file() || is_lfs_pointer(path)? {
        return Ok(Vec::new());
    }
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
        points.push(NcclPoint {
            quant: required_field(&record, &headers, "nccl_dtype", path)?.to_string(),
            operation: required_field(&record, &headers, "op_name", path)?.to_string(),
            num_gpus: parse_u32(&record, &headers, "num_gpus", path)?,
            message_size: parse_u64(&record, &headers, "message_size", path)?,
            latency_ms: parse_f64(&record, &headers, "latency", path)?,
        });
    }
    Ok(points)
}

fn load_optional_moe_points(path: &Path) -> Result<Vec<MoePoint>, AicError> {
    if !path.is_file() || is_lfs_pointer(path)? {
        return Ok(Vec::new());
    }
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
        points.push(MoePoint {
            quant: required_field(&record, &headers, "moe_dtype", path)?.to_string(),
            num_tokens: parse_u32(&record, &headers, "num_tokens", path)?,
            hidden_size: parse_u32(&record, &headers, "hidden_size", path)?,
            inter_size: parse_u32(&record, &headers, "inter_size", path)?,
            top_k: parse_u32(&record, &headers, "topk", path)?,
            num_experts: parse_u32(&record, &headers, "num_experts", path)?,
            moe_tp_size: parse_u32(&record, &headers, "moe_tp_size", path)?,
            moe_ep_size: parse_u32(&record, &headers, "moe_ep_size", path)?,
            distribution: required_field(&record, &headers, "distribution", path)?.to_string(),
            latency_ms: parse_f64(&record, &headers, "latency", path)?,
        });
    }
    Ok(points)
}

fn load_optional_context_mla_points(path: &Path) -> Result<Vec<ContextMlaPoint>, AicError> {
    if !path.is_file() || is_lfs_pointer(path)? {
        return Ok(Vec::new());
    }
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
        points.push(ContextMlaPoint {
            fmha_quant: required_field(&record, &headers, "mla_dtype", path)?.to_string(),
            kv_cache_quant: required_field(&record, &headers, "kv_cache_dtype", path)?.to_string(),
            batch_size: parse_u32(&record, &headers, "batch_size", path)?,
            sequence_tokens: parse_u32(&record, &headers, "isl", path)?,
            num_heads,
            latency_ms: parse_f64(&record, &headers, "latency", path)?,
        });
    }
    Ok(points)
}

fn load_optional_generation_mla_points(path: &Path) -> Result<Vec<GenerationMlaPoint>, AicError> {
    if !path.is_file() || is_lfs_pointer(path)? {
        return Ok(Vec::new());
    }
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
        points.push(GenerationMlaPoint {
            kv_cache_quant: required_field(&record, &headers, "kv_cache_dtype", path)?.to_string(),
            batch_size: parse_u32(&record, &headers, "batch_size", path)?,
            sequence_tokens: parse_u32(&record, &headers, "isl", path)?
                + parse_u32(&record, &headers, "step", path)?,
            num_heads,
            latency_ms: parse_f64(&record, &headers, "latency", path)?,
        });
    }
    Ok(points)
}

const LFS_POINTER_PREFIX: &[u8] = b"version https://git-lfs.github.com/spec/v1";

fn read_lfs_header(path: &Path) -> Result<Vec<u8>, AicError> {
    let mut file = fs::File::open(path).map_err(|source| AicError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut buffer = vec![0_u8; LFS_POINTER_PREFIX.len()];
    let bytes_read = file.read(&mut buffer).map_err(|source| AicError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    buffer.truncate(bytes_read);
    Ok(buffer)
}

fn ensure_real_perf_file(path: &Path) -> Result<(), AicError> {
    if read_lfs_header(path)?.starts_with(LFS_POINTER_PREFIX) {
        return Err(AicError::PerfDatabase(format!(
            "{} is a Git LFS pointer; run `git lfs pull` before using the Rust core estimator with repository perf data",
            path.display()
        )));
    }
    Ok(())
}

fn is_lfs_pointer(path: &Path) -> Result<bool, AicError> {
    Ok(read_lfs_header(path)?.starts_with(LFS_POINTER_PREFIX))
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

fn optional_field<'a>(
    record: &'a StringRecord,
    headers: &HashMap<String, usize>,
    name: &str,
) -> Option<&'a str> {
    let idx = headers.get(name)?;
    record.get(*idx)
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

fn parse_u64(
    record: &StringRecord,
    headers: &HashMap<String, usize>,
    name: &str,
    path: &Path,
) -> Result<u64, AicError> {
    let raw = required_field(record, headers, name, path)?;
    raw.parse::<u64>().map_err(|source| {
        AicError::PerfDatabase(format!(
            "invalid u64 value '{raw}' for column '{name}' in {}: {source}",
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
