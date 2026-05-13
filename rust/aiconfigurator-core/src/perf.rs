// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cmp::Ordering;
use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use csv::StringRecord;

use crate::{AicError, DatabaseMode};

#[derive(Clone, Debug)]
pub(crate) struct PerfDatabase {
    system: SystemSpec,
    database_mode: DatabaseMode,
    gemm: Vec<GemmPoint>,
    context_attention: Vec<ContextAttentionPoint>,
    generation_attention: Vec<GenerationAttentionPoint>,
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
    moe: HashMap<(String, u32, u32, u32, u32, u32, u32, u32, String), Option<f64>>,
    context_mla: HashMap<(String, String, u32, u32, u32, u32), Option<f64>>,
    generation_mla: HashMap<(String, u32, u32, u32), Option<f64>>,
}

impl PerfDatabase {
    pub(crate) fn load(
        systems_root: &Path,
        system_name: &str,
        backend: &str,
        backend_version: Option<&str>,
        database_mode: DatabaseMode,
    ) -> Result<Self, AicError> {
        let system_path = systems_root.join(format!("{system_name}.yaml"));
        let system = read_system_spec(&system_path)?;
        let backend_root = systems_root.join(&system.data_dir).join(backend);
        let version_path =
            resolve_backend_version_for_mode(&backend_root, backend_version, database_mode)?;

        let empty_points = || {
            (
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
            )
        };
        let (gemm, context_attention, generation_attention, moe, context_mla, generation_mla) = if matches!(
            database_mode,
            DatabaseMode::Sol | DatabaseMode::SolFull | DatabaseMode::Empirical
        ) {
            empty_points()
        } else if let Some(version_path) = version_path {
            let gemm_path = version_path.join("gemm_perf.txt");
            let context_attention_path = version_path.join("context_attention_perf.txt");
            let generation_attention_path = version_path.join("generation_attention_perf.txt");
            let moe_path = version_path.join("moe_perf.txt");
            let context_mla_path = version_path.join("context_mla_perf.txt");
            let generation_mla_path = version_path.join("generation_mla_perf.txt");

            match database_mode {
                DatabaseMode::Silicon => (
                    load_gemm_points(&gemm_path)?,
                    load_context_attention_points(&context_attention_path)?,
                    load_generation_attention_points(&generation_attention_path)?,
                    load_optional_moe_points(&moe_path)?,
                    load_optional_context_mla_points(&context_mla_path)?,
                    load_optional_generation_mla_points(&generation_mla_path)?,
                ),
                DatabaseMode::Hybrid => (
                    load_optional_points(&gemm_path, load_gemm_points)?,
                    load_optional_points(&context_attention_path, load_context_attention_points)?,
                    load_optional_points(
                        &generation_attention_path,
                        load_generation_attention_points,
                    )?,
                    load_optional_moe_points(&moe_path)?,
                    load_optional_context_mla_points(&context_mla_path)?,
                    load_optional_generation_mla_points(&generation_mla_path)?,
                ),
                DatabaseMode::Sol | DatabaseMode::SolFull | DatabaseMode::Empirical => {
                    empty_points()
                }
            }
        } else {
            empty_points()
        };

        Ok(Self {
            system,
            database_mode,
            gemm,
            context_attention,
            generation_attention,
            moe,
            context_mla,
            generation_mla,
            query_cache: Arc::new(Mutex::new(QueryCache::default())),
        })
    }

    pub(crate) fn query_gemm(&self, quant: &str, m: u32, n: u32, k: u32) -> Result<f64, AicError> {
        match self.database_mode {
            DatabaseMode::Sol | DatabaseMode::SolFull => Ok(self.sol_gemm_ms(quant, m, n, k)),
            DatabaseMode::Empirical => Ok(self.sol_gemm_ms(quant, m, n, k) / 0.8),
            DatabaseMode::Hybrid => self
                .query_gemm_silicon(quant, m, n, k)
                .or_else(|_| Ok(self.sol_gemm_ms(quant, m, n, k) / 0.8)),
            DatabaseMode::Silicon => self.query_gemm_silicon(quant, m, n, k),
        }
    }

    fn query_gemm_silicon(&self, quant: &str, m: u32, n: u32, k: u32) -> Result<f64, AicError> {
        let key = (quant.to_string(), m, n, k);
        if let Ok(cache) = self.query_cache.lock() {
            if let Some(latency) = cache.gemm.get(&key) {
                return Ok(*latency);
            }
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
        match self.database_mode {
            DatabaseMode::Sol | DatabaseMode::SolFull => Ok(self.sol_context_attention_ms(
                fmha_quant,
                kv_cache_quant,
                batch_size,
                sequence_tokens,
                prefix_tokens,
                num_heads,
                num_kv_heads,
                head_dim,
            )),
            DatabaseMode::Empirical => Ok(self.sol_context_attention_ms(
                fmha_quant,
                kv_cache_quant,
                batch_size,
                sequence_tokens,
                prefix_tokens,
                num_heads,
                num_kv_heads,
                head_dim,
            ) / 0.6),
            DatabaseMode::Hybrid => self
                .query_context_attention_silicon(
                    fmha_quant,
                    kv_cache_quant,
                    batch_size,
                    sequence_tokens,
                    prefix_tokens,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                )
                .or_else(|_| {
                    Ok(self.sol_context_attention_ms(
                        fmha_quant,
                        kv_cache_quant,
                        batch_size,
                        sequence_tokens,
                        prefix_tokens,
                        num_heads,
                        num_kv_heads,
                        head_dim,
                    ) / 0.6)
                }),
            DatabaseMode::Silicon => self.query_context_attention_silicon(
                fmha_quant,
                kv_cache_quant,
                batch_size,
                sequence_tokens,
                prefix_tokens,
                num_heads,
                num_kv_heads,
                head_dim,
            ),
        }
    }

    fn query_context_attention_silicon(
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
        match self.database_mode {
            DatabaseMode::Sol | DatabaseMode::SolFull => Ok(self.sol_generation_attention_ms(
                kv_cache_quant,
                batch_size,
                sequence_tokens,
                num_heads,
                num_kv_heads,
                head_dim,
            )),
            DatabaseMode::Empirical => Ok(self.sol_generation_attention_ms(
                kv_cache_quant,
                batch_size,
                sequence_tokens,
                num_heads,
                num_kv_heads,
                head_dim,
            ) / 0.8),
            DatabaseMode::Hybrid => self
                .query_generation_attention_silicon(
                    kv_cache_quant,
                    batch_size,
                    sequence_tokens,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                )
                .or_else(|_| {
                    Ok(self.sol_generation_attention_ms(
                        kv_cache_quant,
                        batch_size,
                        sequence_tokens,
                        num_heads,
                        num_kv_heads,
                        head_dim,
                    ) / 0.8)
                }),
            DatabaseMode::Silicon => self.query_generation_attention_silicon(
                kv_cache_quant,
                batch_size,
                sequence_tokens,
                num_heads,
                num_kv_heads,
                head_dim,
            ),
        }
    }

    fn query_generation_attention_silicon(
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

        let latency = best.map(|(_, latency)| latency).ok_or_else(|| {
            AicError::PerfDatabase(format!(
                "no generation-attention perf point for kv_cache_quant={kv_cache_quant}, b={batch_size}, s={sequence_tokens}, n={num_heads}, n_kv={num_kv_heads}, head_dim={head_dim}"
            ))
        })?;
        if let Ok(mut cache) = self.query_cache.lock() {
            cache.generation_attention.insert(key, latency);
        }
        Ok(latency)
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
        match self.database_mode {
            DatabaseMode::Sol | DatabaseMode::SolFull => Some(self.sol_moe_ms(
                quant,
                num_tokens,
                hidden_size,
                inter_size,
                top_k,
                num_experts,
                moe_tp_size,
                moe_ep_size,
            )),
            DatabaseMode::Empirical => Some(
                self.sol_moe_ms(
                    quant,
                    num_tokens,
                    hidden_size,
                    inter_size,
                    top_k,
                    num_experts,
                    moe_tp_size,
                    moe_ep_size,
                ) / 0.4,
            ),
            DatabaseMode::Hybrid => self
                .query_moe_silicon(
                    quant,
                    num_tokens,
                    hidden_size,
                    inter_size,
                    top_k,
                    num_experts,
                    moe_tp_size,
                    moe_ep_size,
                    distribution,
                )
                .or_else(|| {
                    Some(
                        self.sol_moe_ms(
                            quant,
                            num_tokens,
                            hidden_size,
                            inter_size,
                            top_k,
                            num_experts,
                            moe_tp_size,
                            moe_ep_size,
                        ) / 0.4,
                    )
                }),
            DatabaseMode::Silicon => self.query_moe_silicon(
                quant,
                num_tokens,
                hidden_size,
                inter_size,
                top_k,
                num_experts,
                moe_tp_size,
                moe_ep_size,
                distribution,
            ),
        }
    }

    fn query_moe_silicon(
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

        let mut best: Option<(f64, f64)> = None;
        for point in self.moe.iter().filter(|point| point.quant == quant) {
            let distribution_penalty = if point.distribution == distribution {
                0.0
            } else if point.distribution == "uniform" {
                0.05
            } else {
                0.25
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
        }
        let latency = best.map(|(_, latency)| latency);
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
        prefix_tokens: u32,
        num_heads: u32,
    ) -> Option<f64> {
        match self.database_mode {
            DatabaseMode::Sol | DatabaseMode::SolFull => Some(self.sol_context_mla_ms(
                fmha_quant,
                kv_cache_quant,
                batch_size,
                sequence_tokens,
                prefix_tokens,
                num_heads,
            )),
            DatabaseMode::Empirical => Some(
                self.sol_context_mla_ms(
                    fmha_quant,
                    kv_cache_quant,
                    batch_size,
                    sequence_tokens,
                    prefix_tokens,
                    num_heads,
                ) / 0.6,
            ),
            DatabaseMode::Hybrid => self
                .query_context_mla_silicon(
                    fmha_quant,
                    kv_cache_quant,
                    batch_size,
                    sequence_tokens,
                    prefix_tokens,
                    num_heads,
                )
                .or_else(|| {
                    Some(
                        self.sol_context_mla_ms(
                            fmha_quant,
                            kv_cache_quant,
                            batch_size,
                            sequence_tokens,
                            prefix_tokens,
                            num_heads,
                        ) / 0.6,
                    )
                }),
            DatabaseMode::Silicon => self.query_context_mla_silicon(
                fmha_quant,
                kv_cache_quant,
                batch_size,
                sequence_tokens,
                prefix_tokens,
                num_heads,
            ),
        }
    }

    fn query_context_mla_silicon(
        &self,
        fmha_quant: &str,
        kv_cache_quant: &str,
        batch_size: u32,
        sequence_tokens: u32,
        prefix_tokens: u32,
        num_heads: u32,
    ) -> Option<f64> {
        let key = (
            fmha_quant.to_string(),
            kv_cache_quant.to_string(),
            batch_size,
            sequence_tokens,
            prefix_tokens,
            num_heads,
        );
        if let Ok(cache) = self.query_cache.lock() {
            if let Some(latency) = cache.context_mla.get(&key) {
                return *latency;
            }
        }

        let mut best: Option<(f64, f64)> = None;
        let full_sequence_tokens = sequence_tokens.saturating_add(prefix_tokens);
        for point in self.context_mla.iter().filter(|point| {
            point.fmha_quant == fmha_quant && point.kv_cache_quant == kv_cache_quant
        }) {
            let score = relative_distance(point.batch_size, batch_size)
                + relative_distance(point.sequence_tokens, full_sequence_tokens)
                + relative_distance(point.num_heads, num_heads);
            if best.map_or(true, |(best_score, _)| score < best_score) {
                best = Some((score, point.latency_ms));
            }
        }
        let latency = best.map(|(_, latency)| {
            let full_sequence_tokens_f = f64::from(full_sequence_tokens);
            if full_sequence_tokens_f == 0.0 {
                return 0.0;
            }
            let prefix_tokens_f = f64::from(prefix_tokens);
            let denominator = full_sequence_tokens_f * full_sequence_tokens_f;
            let numerator = (denominator - prefix_tokens_f * prefix_tokens_f).max(0.0);
            latency * numerator / denominator
        });
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
        match self.database_mode {
            DatabaseMode::Sol | DatabaseMode::SolFull => Some(self.sol_generation_mla_ms(
                kv_cache_quant,
                batch_size,
                sequence_tokens,
                num_heads,
            )),
            DatabaseMode::Empirical => Some(
                self.sol_generation_mla_ms(kv_cache_quant, batch_size, sequence_tokens, num_heads)
                    / 0.8,
            ),
            DatabaseMode::Hybrid => self
                .query_generation_mla_silicon(
                    kv_cache_quant,
                    batch_size,
                    sequence_tokens,
                    num_heads,
                )
                .or_else(|| {
                    Some(
                        self.sol_generation_mla_ms(
                            kv_cache_quant,
                            batch_size,
                            sequence_tokens,
                            num_heads,
                        ) / 0.8,
                    )
                }),
            DatabaseMode::Silicon => self.query_generation_mla_silicon(
                kv_cache_quant,
                batch_size,
                sequence_tokens,
                num_heads,
            ),
        }
    }

    fn query_generation_mla_silicon(
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

    fn sol_gemm_ms(&self, quant: &str, m: u32, n: u32, k: u32) -> f64 {
        let quant = quant_spec(quant);
        let tc_flops = self.system.gpu.quant_tc_flops(quant.compute_factor);
        let m = f64::from(m);
        let n = f64::from(n);
        let k = f64::from(k);
        let sol_math = 2.0 * m * n * k / tc_flops * 1000.0;
        let sol_mem =
            quant.memory_bytes * (m * n + m * k + n * k) / self.system.gpu.mem_bw * 1000.0;
        sol_math.max(sol_mem)
    }

    fn sol_context_attention_ms(
        &self,
        fmha_quant: &str,
        kv_cache_quant: &str,
        batch_size: u32,
        sequence_tokens: u32,
        prefix_tokens: u32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
    ) -> f64 {
        let fmha = quant_spec(fmha_quant);
        let kv = quant_spec(kv_cache_quant);
        let b = f64::from(batch_size);
        let s = f64::from(sequence_tokens);
        let prefix = f64::from(prefix_tokens);
        let full_s = s + prefix;
        let n = f64::from(num_heads);
        let n_kv = f64::from(num_kv_heads);
        let h = f64::from(head_dim);
        let ops = 2.0 * b * (full_s * full_s - prefix * prefix).max(0.0) * n * h;
        let mem_bytes =
            2.0 * b * (n * s * h + n * s * h) + kv.memory_bytes * b * (2.0 * n_kv * full_s * h);
        let sol_math = ops / self.system.gpu.bfloat16_tc_flops * 1000.0 / fmha.compute_factor;
        let sol_mem = mem_bytes / self.system.gpu.mem_bw * 1000.0;
        sol_math.max(sol_mem)
    }

    fn sol_generation_attention_ms(
        &self,
        kv_cache_quant: &str,
        batch_size: u32,
        sequence_tokens: u32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
    ) -> f64 {
        let kv = quant_spec(kv_cache_quant);
        let quant_gen = if kv_cache_quant == "fp8" { 2.0 } else { 1.0 };
        let b = f64::from(batch_size);
        let kv_len = f64::from(sequence_tokens.saturating_sub(1));
        let n = f64::from(num_heads);
        let n_kv = f64::from(num_kv_heads);
        let h = f64::from(head_dim);
        let ops = 2.0 * b * n * h * 2.0 * kv_len;
        let mem_bytes = b * (n * h * 2.0 + 2.0 * n_kv * kv_len * h * kv.memory_bytes + n * h * 2.0);
        let sol_math = ops / self.system.gpu.bfloat16_tc_flops * 1000.0 / quant_gen;
        let sol_mem = mem_bytes / self.system.gpu.mem_bw * 1000.0;
        sol_math.max(sol_mem)
    }

    fn sol_moe_ms(
        &self,
        quant: &str,
        num_tokens: u32,
        hidden_size: u32,
        inter_size: u32,
        top_k: u32,
        num_experts: u32,
        moe_tp_size: u32,
        moe_ep_size: u32,
    ) -> f64 {
        let quant = quant_spec(quant);
        let num_gemms = 3.0;
        let total_tokens = f64::from(num_tokens.saturating_mul(top_k));
        let hidden = f64::from(hidden_size);
        let inter = f64::from(inter_size);
        let tp = f64::from(moe_tp_size.max(1));
        let ep = f64::from(moe_ep_size.max(1));
        let slots = f64::from(num_experts.max(1));
        let ops = total_tokens * hidden * inter * num_gemms * 2.0 / ep / tp;
        let active_tokens = total_tokens / ep;
        let mem_bytes = quant.memory_bytes
            * (active_tokens * hidden * 2.0
                + active_tokens * inter * num_gemms / tp
                + hidden * inter * num_gemms / tp * (slots / ep).min(active_tokens));
        let sol_math = ops / (self.system.gpu.bfloat16_tc_flops * quant.compute_factor) * 1000.0;
        let sol_mem = mem_bytes / self.system.gpu.mem_bw * 1000.0;
        sol_math.max(sol_mem)
    }

    fn sol_context_mla_ms(
        &self,
        fmha_quant: &str,
        kv_cache_quant: &str,
        batch_size: u32,
        sequence_tokens: u32,
        prefix_tokens: u32,
        num_heads: u32,
    ) -> f64 {
        let fmha = quant_spec(fmha_quant);
        let kv = quant_spec(kv_cache_quant);
        let b = f64::from(batch_size);
        let s = f64::from(sequence_tokens);
        let prefix = f64::from(prefix_tokens);
        let full_s = s + prefix;
        let heads = f64::from(num_heads);
        let ops = b * heads * (192.0 + 128.0) * (full_s * full_s - prefix * prefix).max(0.0);
        let mem_bytes =
            b * heads * (kv.memory_bytes * full_s * (192.0 + 128.0) + 2.0 * s * (192.0 + 128.0));
        let sol_math = ops / self.system.gpu.bfloat16_tc_flops * 1000.0 / fmha.compute_factor;
        let sol_mem = mem_bytes / self.system.gpu.mem_bw * 1000.0;
        sol_math.max(sol_mem)
    }

    fn sol_generation_mla_ms(
        &self,
        kv_cache_quant: &str,
        batch_size: u32,
        sequence_tokens: u32,
        num_heads: u32,
    ) -> f64 {
        let kv = quant_spec(kv_cache_quant);
        let quant_gen = if kv_cache_quant == "fp8" { 2.0 } else { 1.0 };
        let b = f64::from(batch_size);
        let s = f64::from(sequence_tokens);
        let heads = f64::from(num_heads);
        let ops = 2.0 * b * heads * 1088.0 * s;
        let mem_bytes = b * (heads * 1088.0 * 2.0 + (s.max(1.0) - 1.0) * 576.0 * kv.memory_bytes);
        let sol_math = ops / self.system.gpu.bfloat16_tc_flops * 1000.0 / quant_gen;
        let sol_mem = mem_bytes / self.system.gpu.mem_bw * 1000.0;
        sol_math.max(sol_mem)
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
struct SystemSpec {
    data_dir: PathBuf,
    gpu: GpuSpec,
}

#[derive(Clone, Debug)]
struct GpuSpec {
    mem_bw: f64,
    bfloat16_tc_flops: f64,
    fp8_tc_flops: Option<f64>,
    fp4_tc_flops: Option<f64>,
}

impl GpuSpec {
    fn quant_tc_flops(&self, compute_factor: f64) -> f64 {
        if (compute_factor - 2.0).abs() < f64::EPSILON {
            return self
                .fp8_tc_flops
                .unwrap_or(self.bfloat16_tc_flops * compute_factor);
        }
        if (compute_factor - 4.0).abs() < f64::EPSILON {
            return self
                .fp4_tc_flops
                .unwrap_or(self.bfloat16_tc_flops * compute_factor);
        }
        self.bfloat16_tc_flops * compute_factor
    }
}

#[derive(Clone, Copy, Debug)]
struct QuantSpec {
    memory_bytes: f64,
    compute_factor: f64,
}

fn quant_spec(name: &str) -> QuantSpec {
    match name {
        "fp8" | "fp8_static" | "fp8_block" | "fp8_ootb" | "sq" => QuantSpec {
            memory_bytes: 1.0,
            compute_factor: 2.0,
        },
        "nvfp4" => QuantSpec {
            memory_bytes: 9.0 / 16.0,
            compute_factor: 4.0,
        },
        "int8" | "int8_wo" => QuantSpec {
            memory_bytes: 1.0,
            compute_factor: 1.0,
        },
        "int4" | "int4_wo" | "w4afp8" | "w4a16_mxfp4" | "w4a8_mxfp4_mxfp8" => QuantSpec {
            memory_bytes: 0.5,
            compute_factor: if name == "w4afp8" || name == "w4a8_mxfp4_mxfp8" {
                2.0
            } else {
                1.0
            },
        },
        _ => QuantSpec {
            memory_bytes: 2.0,
            compute_factor: 1.0,
        },
    }
}

fn read_system_spec(system_path: &Path) -> Result<SystemSpec, AicError> {
    let text = fs::read_to_string(system_path).map_err(|source| AicError::Io {
        path: system_path.to_path_buf(),
        source,
    })?;
    let mut data_dir = None;
    let mut mem_bw = None;
    let mut bfloat16_tc_flops = None;
    let mut fp8_tc_flops = None;
    let mut fp4_tc_flops = None;

    for line in text.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let key = key.trim();
        let value = yaml_scalar(value);
        if value.is_empty() {
            continue;
        }
        match key {
            "data_dir" => data_dir = Some(PathBuf::from(value)),
            "mem_bw" => mem_bw = Some(parse_system_f64(system_path, key, value)?),
            "bfloat16_tc_flops" => {
                bfloat16_tc_flops = Some(parse_system_f64(system_path, key, value)?)
            }
            "fp8_tc_flops" => fp8_tc_flops = Some(parse_system_f64(system_path, key, value)?),
            "fp4_tc_flops" => fp4_tc_flops = Some(parse_system_f64(system_path, key, value)?),
            _ => {}
        }
    }

    let data_dir = data_dir.ok_or_else(|| {
        AicError::PerfDatabase(format!(
            "missing data_dir in system file {}",
            system_path.display()
        ))
    })?;
    let mem_bw = required_system_f64(system_path, "gpu.mem_bw", mem_bw)?;
    let bfloat16_tc_flops =
        required_system_f64(system_path, "gpu.bfloat16_tc_flops", bfloat16_tc_flops)?;

    Ok(SystemSpec {
        data_dir,
        gpu: GpuSpec {
            mem_bw,
            bfloat16_tc_flops,
            fp8_tc_flops,
            fp4_tc_flops,
        },
    })
}

fn yaml_scalar(raw: &str) -> &str {
    raw.trim().trim_matches('"').trim_matches('\'')
}

fn parse_system_f64(system_path: &Path, key: &str, raw: &str) -> Result<f64, AicError> {
    raw.parse::<f64>().map_err(|source| {
        AicError::PerfDatabase(format!(
            "invalid numeric value '{raw}' for '{key}' in {}: {source}",
            system_path.display()
        ))
    })
}

fn required_system_f64(system_path: &Path, key: &str, value: Option<f64>) -> Result<f64, AicError> {
    value.ok_or_else(|| {
        AicError::PerfDatabase(format!(
            "missing {key} in system file {}",
            system_path.display()
        ))
    })
}

fn resolve_backend_version_for_mode(
    backend_root: &Path,
    backend_version: Option<&str>,
    database_mode: DatabaseMode,
) -> Result<Option<PathBuf>, AicError> {
    match database_mode {
        DatabaseMode::Silicon => {
            return resolve_backend_version(backend_root, backend_version).map(Some)
        }
        DatabaseMode::Sol | DatabaseMode::SolFull | DatabaseMode::Empirical => return Ok(None),
        DatabaseMode::Hybrid => {}
    }

    if let Some(version) = backend_version {
        let path = backend_root.join(version);
        return Ok(path.is_dir().then_some(path));
    }

    if !backend_root.is_dir() {
        return Ok(None);
    }

    Ok(resolve_backend_version(backend_root, None).ok())
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

fn load_optional_points<T>(
    path: &Path,
    load: fn(&Path) -> Result<Vec<T>, AicError>,
) -> Result<Vec<T>, AicError> {
    if !path.is_file() || is_lfs_pointer(path)? {
        return Ok(Vec::new());
    }
    load(path)
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
