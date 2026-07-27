// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tuning controls for the forward-pass perf model.

use serde::{Deserialize, Serialize};

use crate::AicError;

use super::samples::integer_sqrt;

pub(crate) const DEFAULT_MAX_OBSERVATIONS: usize = 64;
pub(crate) const DEFAULT_MIN_OBSERVATIONS: usize = 5;
pub(crate) const DEFAULT_BUCKET_COUNT: usize = 16;
pub(crate) const DEFAULT_MAX_NUM_TOKENS: u32 = 8192;
pub(crate) const DEFAULT_MAX_BATCH_SIZE: u32 = 512;
pub(crate) const DEFAULT_MAX_KV_TOKENS: u32 = 2_000_000;

/// In-memory tuning controls for `ForwardPassPerfModel`.
///
/// These defaults match the current planner regression behavior: retain a
/// bounded sliding sample set, wait for enough observations before predicting
/// from learned data, and bucket observations by workload kind.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ForwardPassPerfOptions {
    /// Maximum retained observations across all buckets for each inferred workload kind.
    #[serde(default = "default_max_observations")]
    pub max_observations: usize,
    /// Minimum retained observations required before a regression fit or native
    /// correction is used for an inferred workload kind.
    #[serde(default = "default_min_observations")]
    pub min_observations: usize,
    /// Target bucket count for workload-specific sample retirement and correction lookup.
    #[serde(default = "default_bucket_count")]
    pub bucket_count: usize,
    /// Upper bound for the `sum_prefill_tokens` correction axis.
    ///
    /// Used by prefill and mixed/agg workload kinds. The lower bound is always `0`.
    #[serde(default = "default_max_num_tokens")]
    pub max_num_tokens: u32,
    /// Upper bound for the `num_decode_requests` correction axis.
    ///
    /// Used by the decode workload kind. The lower bound is always `0`.
    #[serde(default = "default_max_batch_size")]
    pub max_batch_size: u32,
    /// Upper bound for the `sum_decode_kv_tokens` correction axis.
    ///
    /// Used by decode and mixed/agg workload kinds. The lower bound is always `0`.
    #[serde(default = "default_max_kv_tokens")]
    pub max_kv_tokens: u32,
}

impl Default for ForwardPassPerfOptions {
    fn default() -> Self {
        Self {
            max_observations: DEFAULT_MAX_OBSERVATIONS,
            min_observations: DEFAULT_MIN_OBSERVATIONS,
            bucket_count: DEFAULT_BUCKET_COUNT,
            max_num_tokens: DEFAULT_MAX_NUM_TOKENS,
            max_batch_size: DEFAULT_MAX_BATCH_SIZE,
            max_kv_tokens: DEFAULT_MAX_KV_TOKENS,
        }
    }
}

pub(crate) fn validate_options(options: &ForwardPassPerfOptions) -> Result<(), AicError> {
    if options.max_observations == 0 {
        return Err(invalid_perf_options("max_observations must be >= 1"));
    }
    if options.min_observations == 0 {
        return Err(invalid_perf_options("min_observations must be >= 1"));
    }
    if options.bucket_count == 0 {
        return Err(invalid_perf_options("bucket_count must be >= 1"));
    }
    if options.max_num_tokens == 0 {
        return Err(invalid_perf_options("max_num_tokens must be >= 1"));
    }
    if options.max_batch_size == 0 {
        return Err(invalid_perf_options("max_batch_size must be >= 1"));
    }
    if options.max_kv_tokens == 0 {
        return Err(invalid_perf_options("max_kv_tokens must be >= 1"));
    }
    if options.min_observations > options.max_observations {
        return Err(invalid_perf_options(
            "min_observations must be <= max_observations",
        ));
    }
    let sqrt = integer_sqrt(options.bucket_count);
    if sqrt * sqrt != options.bucket_count {
        return Err(invalid_perf_options(
            "bucket_count must be a perfect square",
        ));
    }
    Ok(())
}

fn invalid_perf_options(message: &str) -> AicError {
    AicError::InvalidEngineConfig(format!("invalid forward pass perf options: {message}"))
}

fn default_max_observations() -> usize {
    DEFAULT_MAX_OBSERVATIONS
}

fn default_min_observations() -> usize {
    DEFAULT_MIN_OBSERVATIONS
}

fn default_bucket_count() -> usize {
    DEFAULT_BUCKET_COUNT
}

fn default_max_num_tokens() -> u32 {
    DEFAULT_MAX_NUM_TOKENS
}

fn default_max_batch_size() -> u32 {
    DEFAULT_MAX_BATCH_SIZE
}

fn default_max_kv_tokens() -> u32 {
    DEFAULT_MAX_KV_TOKENS
}
