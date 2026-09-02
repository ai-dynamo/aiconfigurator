// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cached Monte Carlo load model for DeepEP low-latency decode.
//!
//! This is the executable counterpart of `docs/DEEPEP_LL_MODELING.md`,
//! especially sections 2-3 and 8-10 (token conventions, endpoint aggregation,
//! topology, Monte Carlo, and final latency). It deliberately models logical source/destination
//! traffic and then collapses it onto physical TX/RX endpoints; the `P^2`
//! logical routes never receive `P^2` independent bandwidth budgets.

use std::sync::OnceLock;

use quick_cache::sync::{Cache, DefaultLifecycle};
use quick_cache::{DefaultHashBuilder, OptionsBuilder, UnitWeighter};
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::common::error::AicError;

pub(crate) const MONTE_CARLO_TRIALS: u32 = 4_096;
const MONTE_CARLO_CACHE_CAPACITY: usize = 4_096;
const MONTE_CARLO_CACHE_SHARDS: usize = 16;
const MONTE_CARLO_BASE_SEED: u64 = 0xA1C0_DEE5_EED0_0001;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(crate) enum RoutingDistribution {
    Balanced,
    PowerLaw { alpha_bits: u64 },
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(crate) enum LlCommPhase {
    Dispatch,
    Combine,
}

/// Whether the measured variable-time term already represents the requested
/// topology or is a single-domain donor that still needs topology guardrails.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(crate) enum LlCalibrationMode {
    ExactTopology,
    SingleDomainDonor,
}

impl RoutingDistribution {
    pub(crate) fn from_workload_name(name: &str) -> Self {
        let normalized = name.strip_suffix("_eplb").unwrap_or(name);
        if matches!(normalized, "uniform" | "balanced") {
            return Self::Balanced;
        }
        let alpha = normalized
            .strip_prefix("power_law_")
            .and_then(|value| value.parse::<f64>().ok())
            .filter(|value| value.is_finite() && *value > 0.0)
            .unwrap_or(1.2);
        Self::PowerLaw {
            alpha_bits: alpha.to_bits(),
        }
    }

    fn alpha(self) -> Option<f64> {
        match self {
            Self::Balanced => None,
            Self::PowerLaw { alpha_bits } => Some(f64::from_bits(alpha_bits)),
        }
    }
}

/// All floating-point request fields are represented by their exact bits so
/// repeated engine queries share one deterministic result without fuzzy cache
/// aliases.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(crate) struct MonteCarloRequest {
    pub phase: LlCommPhase,
    pub calibration_mode: LlCalibrationMode,
    pub per_rank_tokens: u32,
    pub num_ranks: u32,
    pub topk: u32,
    pub num_experts: u32,
    /// Contiguous rank count inside one NVLink/MNVL domain.
    pub nvl_domain_size: u32,
    pub distribution: RoutingDistribution,
    pub payload_bytes: u64,
    /// Measured/interpolated variable latency `max(T_base - t0, 0)`.
    pub fitted_variable_ms_bits: u64,
    pub nvl_bandwidth_bits: u64,
    /// Zero means that no rank pair in the modeled EP group crosses an IB
    /// domain. Otherwise bytes/s, single direction.
    pub ib_bandwidth_bits: u64,
}

impl MonteCarloRequest {
    fn fitted_variable_ms(self) -> f64 {
        f64::from_bits(self.fitted_variable_ms_bits)
    }

    fn nvl_bandwidth(self) -> f64 {
        f64::from_bits(self.nvl_bandwidth_bits)
    }

    fn ib_bandwidth(self) -> f64 {
        f64::from_bits(self.ib_bandwidth_bits)
    }

    fn cache_key(mut self) -> Self {
        if self.calibration_mode == LlCalibrationMode::ExactTopology {
            self.nvl_bandwidth_bits = 0.0_f64.to_bits();
            self.ib_bandwidth_bits = 0.0_f64.to_bits();
        }
        self
    }
}

fn monte_carlo_cache() -> Cache<MonteCarloRequest, f64> {
    let options = OptionsBuilder::new()
        .estimated_items_capacity(MONTE_CARLO_CACHE_CAPACITY)
        .weight_capacity(MONTE_CARLO_CACHE_CAPACITY as u64)
        .shards(MONTE_CARLO_CACHE_SHARDS)
        .build()
        .expect("valid static DeepEP-LL Monte Carlo cache options");
    Cache::with_options(
        options,
        UnitWeighter,
        DefaultHashBuilder::default(),
        DefaultLifecycle::default(),
    )
}

static MONTE_CARLO_CACHE: OnceLock<Cache<MonteCarloRequest, f64>> = OnceLock::new();

/// Return the P50 variable latency. The caller adds fitted startup latency
/// exactly once after the cached Monte Carlo estimate.
pub(crate) fn estimate(request: MonteCarloRequest) -> Result<f64, AicError> {
    let cache = MONTE_CARLO_CACHE.get_or_init(monte_carlo_cache);
    estimate_with_cache(cache, request)
}

fn estimate_with_cache(
    cache: &Cache<MonteCarloRequest, f64>,
    request: MonteCarloRequest,
) -> Result<f64, AicError> {
    let request = request.cache_key();
    if let Some(p50_ms) = cache.get(&request) {
        return Ok(p50_ms);
    }
    let p50_ms = estimate_uncached(request)?;
    if p50_ms.is_finite() {
        cache.insert(request, p50_ms);
    }
    Ok(p50_ms)
}

fn estimate_uncached(request: MonteCarloRequest) -> Result<f64, AicError> {
    validate_request(request)?;
    if request.per_rank_tokens == 0 {
        return Ok(0.0);
    }

    let trials = trial_count(request);
    let mut variable_latencies = run_trials(request, trials)?;
    median(&mut variable_latencies).ok_or_else(|| {
        AicError::PerfDatabase("DeepEP-LL Monte Carlo produced no trials".to_string())
    })
}

fn run_trials(request: MonteCarloRequest, trials: u32) -> Result<Vec<f64>, AicError> {
    let workers = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
        .min(8)
        .min(trials as usize)
        .max(1);
    let chunk = (trials as usize + workers - 1) / workers;
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(workers);
        for start in (0..trials as usize).step_by(chunk) {
            let end = (start + chunk).min(trials as usize);
            handles.push(scope.spawn(move || {
                let mut samples = Vec::with_capacity(end - start);
                for trial in start..end {
                    let seed =
                        MONTE_CARLO_BASE_SEED ^ (trial as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                    let mut rng = ChaCha8Rng::seed_from_u64(seed);
                    samples.push(one_trial(request, &mut rng)?);
                }
                Ok::<_, AicError>(samples)
            }));
        }
        let mut samples = Vec::with_capacity(trials as usize);
        for handle in handles {
            samples.extend(handle.join().map_err(|_| {
                AicError::PerfDatabase("DeepEP-LL Monte Carlo worker panicked".to_string())
            })??);
        }
        Ok(samples)
    })
}

fn trial_count(request: MonteCarloRequest) -> u32 {
    // Strictly balanced exact-topology routing always has alpha_comm=1, so
    // random source/destination pairing cannot change the result. Donor
    // topology floors still depend on that pairing and use all trials.
    match (request.distribution, request.calibration_mode) {
        (RoutingDistribution::Balanced, LlCalibrationMode::ExactTopology) => 1,
        _ => MONTE_CARLO_TRIALS,
    }
}

fn median(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    Some(if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    })
}

fn validate_request(request: MonteCarloRequest) -> Result<(), AicError> {
    if request.num_ranks == 0
        || request.topk == 0
        || request.num_experts == 0
        || request.nvl_domain_size == 0
        || request.num_experts % request.num_ranks != 0
        || request.topk > request.num_experts
    {
        return Err(AicError::InvalidEngineConfig(format!(
            "invalid DeepEP-LL Monte Carlo geometry: P={}, K={}, N={}, NVL-domain={}",
            request.num_ranks, request.topk, request.num_experts, request.nvl_domain_size
        )));
    }
    let fitted = request.fitted_variable_ms();
    let nvl_bw = request.nvl_bandwidth();
    let ib_bw = request.ib_bandwidth();
    if !fitted.is_finite()
        || fitted < 0.0
        || (request.calibration_mode == LlCalibrationMode::SingleDomainDonor
            && (!nvl_bw.is_finite()
                || nvl_bw <= 0.0
                || !ib_bw.is_finite()
                || ib_bw < 0.0
                || (request.num_ranks > request.nvl_domain_size && ib_bw == 0.0)))
    {
        return Err(AicError::InvalidEngineConfig(format!(
            "invalid DeepEP-LL latency/bandwidth request: fitted={fitted}, NVL={nvl_bw}, IB={ib_bw}"
        )));
    }
    if let Some(alpha) = request.distribution.alpha() {
        if !alpha.is_finite() || alpha <= 0.0 {
            return Err(AicError::InvalidEngineConfig(format!(
                "invalid DeepEP-LL power-law alpha {alpha}"
            )));
        }
    }
    Ok(())
}

#[derive(Debug, PartialEq, Eq)]
struct EndpointLoads {
    total_tx: Vec<u64>,
    total_rx: Vec<u64>,
    nvl_tx: Vec<u64>,
    nvl_rx: Vec<u64>,
    ib_tx: Vec<u64>,
    ib_rx: Vec<u64>,
}

fn one_trial(request: MonteCarloRequest, rng: &mut ChaCha8Rng) -> Result<f64, AicError> {
    let p = request.num_ranks as usize;
    let n = request.num_experts as usize;
    let k = request.topk as usize;
    let per_rank_tokens = request.per_rank_tokens as usize;
    let global_tokens = per_rank_tokens.checked_mul(p).ok_or_else(|| {
        AicError::InvalidEngineConfig("DeepEP-LL global token count overflow".to_string())
    })?;
    let total_assignments = global_tokens.checked_mul(k).ok_or_else(|| {
        AicError::InvalidEngineConfig("DeepEP-LL token-expert count overflow".to_string())
    })?;
    let counts = expert_counts(global_tokens, n, k, p, request.distribution, rng)?;
    let average_endpoint = total_assignments as f64 / p as f64;
    let experts_per_rank = n / p;
    let busiest_endpoint = counts
        .chunks_exact(experts_per_rank)
        .map(|rank| rank.iter().sum::<usize>())
        .max()
        .unwrap_or(0)
        .max(total_assignments / p) as f64;
    let alpha_comm = busiest_endpoint / average_endpoint;
    let fitted_ms = alpha_comm * request.fitted_variable_ms();
    if request.calibration_mode == LlCalibrationMode::ExactTopology {
        return Ok(fitted_ms);
    }

    let nvl_domain = request.nvl_domain_size as usize;
    let loads = endpoint_loads_from_counts(&counts, per_rank_tokens, p, k, nvl_domain, rng)?;
    let nvl_assignments = max_directional_endpoint(&loads.nvl_tx, &loads.nvl_rx);
    let ib_assignments = max_directional_endpoint(&loads.ib_tx, &loads.ib_rx);
    let payload = request.payload_bytes as f64;
    let nvl_ms = nvl_assignments as f64 * payload / request.nvl_bandwidth() * 1_000.0;
    let ib_ms = if ib_assignments == 0 {
        0.0
    } else {
        ib_assignments as f64 * payload / request.ib_bandwidth() * 1_000.0
    };
    Ok(fitted_ms.max(nvl_ms).max(ib_ms))
}

/// Randomly assign the exact expert quotas to tokens and aggregate logical
/// routes onto physical endpoints. Token order is shuffled so the mandatory
/// feasibility choices below do not correlate with source-rank order.
fn endpoint_loads_from_counts(
    counts: &[usize],
    per_rank_tokens: usize,
    num_ranks: usize,
    topk: usize,
    nvl_domain: usize,
    rng: &mut ChaCha8Rng,
) -> Result<EndpointLoads, AicError> {
    let global_tokens = per_rank_tokens.checked_mul(num_ranks).ok_or_else(|| {
        AicError::InvalidEngineConfig("DeepEP-LL global token count overflow".to_string())
    })?;
    let total_assignments = global_tokens.checked_mul(topk).ok_or_else(|| {
        AicError::InvalidEngineConfig("DeepEP-LL token-expert count overflow".to_string())
    })?;
    let experts_per_rank = counts.len() / num_ranks;
    let mut loads = EndpointLoads {
        total_tx: vec![0; num_ranks],
        total_rx: vec![0; num_ranks],
        nvl_tx: vec![0; num_ranks],
        nvl_rx: vec![0; num_ranks],
        ib_tx: vec![0; num_ranks],
        ib_rx: vec![0; num_ranks],
    };

    let assignments = random_assignments_from_counts(counts, topk, rng)?;
    for (token, experts) in assignments.chunks_exact(topk).enumerate() {
        let source = token / per_rank_tokens;
        for &expert in experts {
            let destination = expert / experts_per_rank;
            loads.total_tx[source] += 1;
            loads.total_rx[destination] += 1;
            if source == destination {
                continue;
            }
            if source / nvl_domain == destination / nvl_domain {
                loads.nvl_tx[source] += 1;
                loads.nvl_rx[destination] += 1;
            } else {
                loads.ib_tx[source] += 1;
                loads.ib_rx[destination] += 1;
            }
        }
    }
    let assigned = assignments.len();
    if assigned != total_assignments {
        return Err(AicError::PerfDatabase(format!(
            "DeepEP-LL Monte Carlo quota sum mismatch: got {assigned}, expected {total_assignments}"
        )));
    }
    Ok(loads)
}

/// Construct a simple bipartite token/expert graph with exact column degrees.
/// An expert whose remaining quota equals the number of remaining tokens is
/// mandatory for the current token. Selecting every mandatory expert first
/// preserves `remaining_quota <= remaining_tokens`; weighted sampling without
/// replacement fills the other slots and therefore cannot dead-end.
fn random_assignments_from_counts(
    counts: &[usize],
    topk: usize,
    rng: &mut ChaCha8Rng,
) -> Result<Vec<usize>, AicError> {
    let total_assignments = counts.iter().sum::<usize>();
    if topk == 0 || total_assignments % topk != 0 {
        return Err(AicError::InvalidEngineConfig(format!(
            "invalid DeepEP-LL quota/Top-K: assignments={total_assignments}, topk={topk}"
        )));
    }
    let global_tokens = total_assignments / topk;
    if counts.iter().any(|&count| count > global_tokens) {
        return Err(AicError::InvalidEngineConfig(format!(
            "DeepEP-LL expert quota exceeds token count {global_tokens}"
        )));
    }
    let mut remaining = counts.to_vec();
    let mut quota_pool = QuotaPool::new(&remaining);
    let mut quota_frequencies = vec![0_usize; global_tokens + 1];
    for &quota in &remaining {
        quota_frequencies[quota] += 1;
    }
    let mut token_order = (0..global_tokens).collect::<Vec<_>>();
    token_order.shuffle(rng);
    let mut assignments = vec![usize::MAX; total_assignments];
    let mut selected = Vec::with_capacity(topk);

    for (step, token) in token_order.into_iter().enumerate() {
        let remaining_tokens = global_tokens - step;
        selected.clear();
        if quota_frequencies[remaining_tokens] != 0 {
            for (expert, &quota) in remaining.iter().enumerate() {
                if quota == remaining_tokens {
                    selected.push(expert);
                }
            }
        }
        if selected.len() > topk {
            return Err(AicError::PerfDatabase(format!(
                "DeepEP-LL quota assignment has {} mandatory experts for Top-K {topk}",
                selected.len()
            )));
        }
        for &expert in selected.iter() {
            quota_pool.remove_expert(expert);
            quota_frequencies[remaining[expert]] -= 1;
            remaining[expert] -= 1;
            quota_frequencies[remaining[expert]] += 1;
        }
        while selected.len() < topk {
            let total_weight = quota_pool.len();
            if total_weight == 0 {
                return Err(AicError::PerfDatabase(
                    "DeepEP-LL quota assignment exhausted eligible experts".to_string(),
                ));
            }
            let (position, expert) = loop {
                let position = rng.gen_range(0..total_weight);
                let candidate = quota_pool.select(position);
                if !selected.contains(&candidate) {
                    break (position, candidate);
                }
            };
            selected.push(expert);
            quota_pool.remove_at(position);
            quota_frequencies[remaining[expert]] -= 1;
            remaining[expert] -= 1;
            quota_frequencies[remaining[expert]] += 1;
        }
        let start = token * topk;
        assignments[start..start + topk].copy_from_slice(&selected);
    }
    if remaining.iter().any(|&quota| quota != 0) {
        return Err(AicError::PerfDatabase(format!(
            "DeepEP-LL quota assignment left residual counts {remaining:?}"
        )));
    }
    Ok(assignments)
}

/// O(1) weighted draws and accepted quota decrements. Each live occurrence in
/// `experts` is one unit of remaining quota; drawing a uniform position is
/// therefore proportional to quota. A mandatory expert is rare and may use a
/// linear lookup to remove one of its occurrences.
struct QuotaPool {
    experts: Vec<usize>,
}

impl QuotaPool {
    fn new(quotas: &[usize]) -> Self {
        let capacity = quotas.iter().sum();
        let mut experts = Vec::with_capacity(capacity);
        for (expert, &quota) in quotas.iter().enumerate() {
            for _ in 0..quota {
                experts.push(expert);
            }
        }
        Self { experts }
    }

    fn len(&self) -> usize {
        self.experts.len()
    }

    fn select(&self, position: usize) -> usize {
        self.experts[position]
    }

    fn remove_at(&mut self, position: usize) {
        self.experts.swap_remove(position);
    }

    fn remove_expert(&mut self, expert: usize) {
        let position = self
            .experts
            .iter()
            .rposition(|&candidate| candidate == expert)
            .expect("mandatory expert has positive remaining quota");
        self.remove_at(position);
    }
}

fn max_directional_endpoint(tx: &[u64], rx: &[u64]) -> u64 {
    tx.iter()
        .zip(rx)
        .map(|(&sent, &received)| sent.max(received))
        .max()
        .unwrap_or(0)
}

fn expert_counts(
    global_tokens: usize,
    num_experts: usize,
    topk: usize,
    num_ranks: usize,
    distribution: RoutingDistribution,
    rng: &mut ChaCha8Rng,
) -> Result<Vec<usize>, AicError> {
    // FIXME: Add cross-language quota fixtures against
    // collector/helper.py::_generate_power_law_distribution. Rust uses
    // ChaCha8Rng while the collector uses torch.rand, so parity fixtures must
    // share deterministic sampled weights rather than assume equal seeds
    // produce equal random streams. Only quota generation should match: the
    // communication model intentionally randomizes token assignment instead
    // of using helper.py's compute-oriented descending fill.
    let target = global_tokens.checked_mul(topk).ok_or_else(|| {
        AicError::InvalidEngineConfig("DeepEP-LL quota target overflow".to_string())
    })?;
    let mut counts = match distribution.alpha() {
        None => {
            // `balanced` gives every destination rank exactly the same
            // target / num_ranks assignments. Split that rank-local total as
            // evenly as possible across its contiguous local experts.
            let experts_per_rank = num_experts / num_ranks;
            let assignments_per_rank = target / num_ranks;
            let base = assignments_per_rank / experts_per_rank;
            let remainder = assignments_per_rank % experts_per_rank;
            let mut counts = vec![base; num_experts];
            for rank in 0..num_ranks {
                let start = rank * experts_per_rank;
                let mut local = (0..experts_per_rank).collect::<Vec<_>>();
                local.shuffle(rng);
                for &local_expert in local.iter().take(remainder) {
                    counts[start + local_expert] += 1;
                }
            }
            counts
        }
        Some(alpha) => {
            let (xmin, xmax) = if target > num_experts {
                (1.0, global_tokens as f64 * 0.8)
            } else {
                (0.01, 2.0)
            };
            let weights: Vec<f64> = (0..num_experts)
                .map(|_| sample_power_law(alpha, xmin, xmax.max(xmin), rng))
                .collect();
            let sum = weights.iter().sum::<f64>();
            weights
                .iter()
                .map(|weight| (weight / sum * target as f64).round() as usize)
                .collect()
        }
    };

    for count in &mut counts {
        *count = (*count).min(global_tokens);
    }
    adjust_quota_round_robin(&mut counts, target, global_tokens, num_ranks);
    if counts.iter().sum::<usize>() != target {
        return Err(AicError::PerfDatabase(format!(
            "DeepEP-LL could not construct a valid Top-K quota: target={target}, actual={}",
            counts.iter().sum::<usize>()
        )));
    }

    // Match the collector's worst-rank convention: move the busiest
    // contiguous expert group to rank 0 without sorting experts globally.
    let experts_per_rank = num_experts / num_ranks;
    let busiest = (0..num_ranks)
        .max_by_key(|rank| {
            counts[rank * experts_per_rank..(rank + 1) * experts_per_rank]
                .iter()
                .sum::<usize>()
        })
        .unwrap_or(0);
    if busiest != 0 {
        for local in 0..experts_per_rank {
            counts.swap(local, busiest * experts_per_rank + local);
        }
    }
    Ok(counts)
}

fn adjust_quota_round_robin(
    counts: &mut [usize],
    target: usize,
    upper_bound: usize,
    num_ranks: usize,
) {
    let experts_per_rank = counts.len() / num_ranks;
    let mut actual = counts.iter().sum::<usize>();
    while actual < target {
        let mut progressed = false;
        for rank in 0..num_ranks {
            let start = rank * experts_per_rank;
            let end = start + experts_per_rank;
            if let Some(index) = (start..end)
                .filter(|&index| counts[index] < upper_bound)
                .min_by_key(|&index| counts[index])
            {
                counts[index] += 1;
                actual += 1;
                progressed = true;
                if actual == target {
                    return;
                }
            }
        }
        if !progressed {
            return;
        }
    }
    while actual > target {
        let mut progressed = false;
        for rank in 0..num_ranks {
            let start = rank * experts_per_rank;
            let end = start + experts_per_rank;
            if let Some(index) = (start..end)
                .filter(|&index| counts[index] > 0)
                .max_by_key(|&index| counts[index])
            {
                counts[index] -= 1;
                actual -= 1;
                progressed = true;
                if actual == target {
                    return;
                }
            }
        }
        if !progressed {
            return;
        }
    }
}

fn sample_power_law(alpha: f64, xmin: f64, xmax: f64, rng: &mut ChaCha8Rng) -> f64 {
    let u = rng.gen::<f64>();
    if (alpha - 1.0).abs() < 1e-12 {
        return xmin * (xmax / xmin).powf(u);
    }
    let exponent = 1.0 - alpha;
    ((xmax.powf(exponent) - xmin.powf(exponent)) * u + xmin.powf(exponent)).powf(1.0 / exponent)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(distribution: RoutingDistribution) -> MonteCarloRequest {
        MonteCarloRequest {
            phase: LlCommPhase::Dispatch,
            calibration_mode: LlCalibrationMode::ExactTopology,
            per_rank_tokens: 2,
            num_ranks: 4,
            topk: 2,
            num_experts: 8,
            nvl_domain_size: 4,
            distribution,
            payload_bytes: 1_024,
            fitted_variable_ms_bits: 0.010_f64.to_bits(),
            nvl_bandwidth_bits: 100e9_f64.to_bits(),
            ib_bandwidth_bits: 0.0_f64.to_bits(),
        }
    }

    #[test]
    fn workload_names_preserve_model_alpha_and_default_to_1_2() {
        assert_eq!(
            RoutingDistribution::from_workload_name("power_law_1.01").alpha(),
            Some(1.01)
        );
        assert_eq!(
            RoutingDistribution::from_workload_name("power_law").alpha(),
            Some(1.2)
        );
        assert_eq!(
            RoutingDistribution::from_workload_name("unknown").alpha(),
            Some(1.2)
        );
        assert_eq!(
            RoutingDistribution::from_workload_name("balanced").alpha(),
            None
        );
    }

    #[test]
    fn deterministic_p50_excludes_t0() {
        let req = request(RoutingDistribution::from_workload_name("power_law_1.2"));
        let first = estimate_uncached(req).unwrap();
        let second = estimate_uncached(req).unwrap();
        assert_eq!(first, second);
        assert!(first > 0.0);
    }

    #[test]
    fn repeated_request_hits_the_bounded_cache() {
        let mut req = request(RoutingDistribution::from_workload_name("power_law_1.01"));
        req.per_rank_tokens = 17;
        let cache = monte_carlo_cache();
        let first = estimate_with_cache(&cache, req).unwrap();
        let after_first = cache.len();
        let second = estimate_with_cache(&cache, req).unwrap();
        let after_second = cache.len();
        assert_eq!(first, second);
        assert_eq!(after_first, 1);
        assert_eq!(after_second, after_first);
        assert!(after_second <= MONTE_CARLO_CACHE_CAPACITY);
    }

    #[test]
    fn cache_ignores_exact_bandwidth_and_isolates_donor_mode() {
        let cache = monte_carlo_cache();
        let mut exact = request(RoutingDistribution::Balanced);
        exact.nvl_bandwidth_bits = 1.0_f64.to_bits();
        exact.ib_bandwidth_bits = 2.0_f64.to_bits();
        let exact_ms = estimate_with_cache(&cache, exact).unwrap();

        let exact_other_bandwidth = MonteCarloRequest {
            nvl_bandwidth_bits: 200e9_f64.to_bits(),
            ib_bandwidth_bits: 50e9_f64.to_bits(),
            ..exact
        };
        assert_eq!(
            estimate_with_cache(&cache, exact_other_bandwidth).unwrap(),
            exact_ms
        );
        assert_eq!(cache.len(), 1);

        let donor = MonteCarloRequest {
            calibration_mode: LlCalibrationMode::SingleDomainDonor,
            ..exact_other_bandwidth
        };
        assert!(estimate_with_cache(&cache, donor).unwrap() > 0.0);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn full_duplex_uses_max_direction_not_send_plus_receive() {
        assert_eq!(max_directional_endpoint(&[3, 8], &[7, 4]), 8);
    }

    #[test]
    fn balanced_quota_is_valid_and_topk_distinct() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let counts = expert_counts(8, 8, 2, 4, RoutingDistribution::Balanced, &mut rng).unwrap();
        assert_eq!(counts.iter().sum::<usize>(), 16);
        assert!(counts.iter().all(|&count| count <= 8));
        for rank in 0..4 {
            assert_eq!(counts[rank * 2..(rank + 1) * 2].iter().sum::<usize>(), 4);
        }
        let assignments = random_assignments_from_counts(&counts, 2, &mut rng).unwrap();
        let mut observed = vec![0; counts.len()];
        for experts in assignments.chunks_exact(2) {
            assert_eq!(experts.len(), 2);
            assert_ne!(experts[0], experts[1]);
            for &expert in experts {
                observed[expert] += 1;
            }
        }
        assert_eq!(observed, counts);
    }

    #[test]
    fn random_assignment_preserves_endpoint_totals_and_changes_with_seed() {
        let counts = [0, 1, 2, 3, 4, 5, 6, 11];
        let mut rng_a = ChaCha8Rng::seed_from_u64(11);
        let mut rng_b = ChaCha8Rng::seed_from_u64(12);
        let loads_a = endpoint_loads_from_counts(&counts, 4, 4, 2, 2, &mut rng_a).unwrap();
        let loads_b = endpoint_loads_from_counts(&counts, 4, 4, 2, 2, &mut rng_b).unwrap();
        assert_eq!(loads_a.total_tx, vec![8, 8, 8, 8]);
        assert_eq!(loads_a.total_rx, vec![1, 5, 9, 17]);
        assert_eq!(loads_a.total_tx.iter().sum::<u64>(), 32);
        assert_eq!(loads_a.total_rx.iter().sum::<u64>(), 32);
        assert_ne!(loads_a, loads_b);
    }

    #[test]
    fn power_law_one_is_still_a_full_monte_carlo_distribution() {
        let req = request(RoutingDistribution::from_workload_name("power_law_1.0"));
        assert_eq!(trial_count(req), MONTE_CARLO_TRIALS);
        let balanced_exact = request(RoutingDistribution::Balanced);
        assert_eq!(trial_count(balanced_exact), 1);
        let balanced_donor = MonteCarloRequest {
            calibration_mode: LlCalibrationMode::SingleDomainDonor,
            ..balanced_exact
        };
        assert_eq!(trial_count(balanced_donor), MONTE_CARLO_TRIALS);
    }

    #[test]
    fn runtime_statistic_is_standard_median_not_mean() {
        let mut odd = [100.0, 1.0, 3.0];
        assert_eq!(median(&mut odd), Some(3.0));
        let mut even = [100.0, 1.0, 3.0, 2.0];
        assert_eq!(median(&mut even), Some(2.5));
        let mut empty = [];
        assert_eq!(median(&mut empty), None);
    }

    #[test]
    fn estimator_returns_p50_when_trial_mean_differs() {
        let req = request(RoutingDistribution::from_workload_name("power_law_1.2"));
        let mut samples = run_trials(req, MONTE_CARLO_TRIALS).unwrap();
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let p50 = median(&mut samples).unwrap();
        assert_ne!(p50, mean);
        assert_eq!(estimate_uncached(req).unwrap(), p50);
    }

    #[test]
    fn fixed_seed_keeps_random_assignments_deterministic() {
        let counts = [4, 4, 3, 3, 2, 2, 1, 1];
        let mut first_rng = ChaCha8Rng::seed_from_u64(99);
        let mut second_rng = ChaCha8Rng::seed_from_u64(99);
        assert_eq!(
            random_assignments_from_counts(&counts, 2, &mut first_rng).unwrap(),
            random_assignments_from_counts(&counts, 2, &mut second_rng).unwrap()
        );
    }

    #[test]
    fn hot_experts_can_share_one_contiguous_rank() {
        let mut observed = false;
        for seed in 0..64 {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let counts = expert_counts(
                32,
                8,
                2,
                4,
                RoutingDistribution::from_workload_name("power_law_1.0"),
                &mut rng,
            )
            .unwrap();
            let mut order = (0..counts.len()).collect::<Vec<_>>();
            order.sort_by_key(|&expert| std::cmp::Reverse(counts[expert]));
            if order[0] / 2 == order[1] / 2 {
                observed = true;
                break;
            }
        }
        assert!(
            observed,
            "random quota placement never clustered the two hottest experts"
        );
    }

    #[test]
    fn balanced_exact_topology_returns_the_fitted_variable_time() {
        let req = request(RoutingDistribution::Balanced);
        assert_eq!(estimate_uncached(req).unwrap(), req.fitted_variable_ms());
    }

    #[test]
    fn exact_topology_ignores_spec_bandwidth_but_donor_enforces_it() {
        let mut exact = request(RoutingDistribution::Balanced);
        exact.nvl_bandwidth_bits = 1.0_f64.to_bits();
        let exact_ms = estimate_uncached(exact).unwrap();
        assert_eq!(exact_ms, exact.fitted_variable_ms());

        let donor = MonteCarloRequest {
            calibration_mode: LlCalibrationMode::SingleDomainDonor,
            ..exact
        };
        let donor_ms = estimate_uncached(donor).unwrap();
        assert!(donor_ms > exact_ms);
    }

    #[test]
    fn power_law_exact_topology_only_scales_the_fitted_variable_term() {
        let exact = request(RoutingDistribution::from_workload_name("power_law_1.2"));
        let once = estimate_uncached(exact).unwrap();
        let doubled = estimate_uncached(MonteCarloRequest {
            fitted_variable_ms_bits: (2.0 * exact.fitted_variable_ms()).to_bits(),
            nvl_bandwidth_bits: 1.0_f64.to_bits(),
            ib_bandwidth_bits: 1.0_f64.to_bits(),
            ..exact
        })
        .unwrap();
        assert!((doubled - 2.0 * once).abs() < 1e-12);
    }
}
