// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Forward-pass metrics telemetry types.
//!
//! These mirror Dynamo's scheduler `ForwardPassMetrics` v1 wire schema. AIC owns
//! this Rust copy so it does not depend on Dynamo crates. The forward-pass perf
//! model consumes the scheduled portion as estimator input and uses `wall_time`
//! as the tuning target.

use serde::{Deserialize, Serialize};

use crate::AicError;

/// Wire-schema version for [`ForwardPassMetrics`] (Dynamo's scheduler
/// telemetry). The FPM input is rejected with [`AicError::UnsupportedSchemaVersion`]
/// when `version` does not match.
pub const FPM_VERSION: u32 = 1;

/// Metrics for requests scheduled in one forward-pass iteration.
///
/// This mirrors Dynamo ForwardPassMetrics v1 scheduled request telemetry.
/// AIC owns this Rust copy so AIC does not depend on Dynamo crates.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct ScheduledRequestMetrics {
    /// Number of prefill requests, including new requests and chunked-prefill
    /// continuations.
    #[serde(default)]
    pub num_prefill_requests: u32,
    /// Total tokens freshly computed for prefill requests in this iteration.
    #[serde(default)]
    pub sum_prefill_tokens: u32,
    /// Population variance of total prompt lengths across prefill requests.
    #[serde(default)]
    pub var_prefill_length: f64,
    /// Total KV tokens read for prefill requests, including prefix cache hits
    /// and previously computed chunks.
    #[serde(default)]
    pub sum_prefill_kv_tokens: u32,
    /// Number of decode requests.
    #[serde(default)]
    pub num_decode_requests: u32,
    /// Total KV context length across decode requests.
    #[serde(default)]
    pub sum_decode_kv_tokens: u32,
    /// Population variance of KV context lengths across decode requests.
    #[serde(default)]
    pub var_decode_kv_tokens: f64,
}

/// Metrics for requests queued but not scheduled in one iteration.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct QueuedRequestMetrics {
    #[serde(default)]
    pub num_prefill_requests: u32,
    #[serde(default)]
    pub sum_prefill_tokens: u32,
    #[serde(default)]
    pub var_prefill_length: f64,
    #[serde(default)]
    pub num_decode_requests: u32,
    #[serde(default)]
    pub sum_decode_kv_tokens: u32,
    #[serde(default)]
    pub var_decode_kv_tokens: f64,
}

/// Per-iteration forward-pass metrics.
///
/// In Dynamo this struct is telemetry emitted after an engine iteration. In
/// AIC the scheduled portion is also the estimator input. `wall_time` and
/// `queued_requests` are accepted for schema parity but ignored by the
/// latency estimator (the forward-pass perf model uses `wall_time` as the
/// tuning target, never as an estimation input).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ForwardPassMetrics {
    #[serde(default = "default_fpm_version")]
    pub version: u32,
    #[serde(default)]
    pub worker_id: String,
    #[serde(default)]
    pub dp_rank: u32,
    #[serde(default)]
    pub counter_id: u64,
    #[serde(default)]
    pub wall_time: f64,
    #[serde(default)]
    pub scheduled_requests: ScheduledRequestMetrics,
    #[serde(default)]
    pub queued_requests: QueuedRequestMetrics,
}

impl Default for ForwardPassMetrics {
    fn default() -> Self {
        Self {
            version: FPM_VERSION,
            worker_id: String::new(),
            dp_rank: 0,
            counter_id: 0,
            wall_time: 0.0,
            scheduled_requests: ScheduledRequestMetrics::default(),
            queued_requests: QueuedRequestMetrics::default(),
        }
    }
}

/// Validate one [`ForwardPassMetrics`] entry. Rejects mismatched schema
/// versions and inconsistent scheduled-request sums (token sums without a
/// matching request count). Shared by the forward-pass perf model and
/// [`crate::engine::Engine::forward_pass_time_ms`].
pub(crate) fn validate_forward_pass_metrics(metrics: &ForwardPassMetrics) -> Result<(), AicError> {
    if metrics.version != FPM_VERSION {
        return Err(AicError::UnsupportedSchemaVersion {
            kind: "ForwardPassMetrics",
            got: metrics.version,
            expected: FPM_VERSION,
        });
    }
    let scheduled = &metrics.scheduled_requests;
    if scheduled.num_prefill_requests == 0
        && (scheduled.sum_prefill_tokens > 0 || scheduled.sum_prefill_kv_tokens > 0)
    {
        return Err(AicError::InvalidForwardPassMetrics(
            "prefill token sums require num_prefill_requests > 0".to_string(),
        ));
    }
    if scheduled.num_decode_requests == 0 && scheduled.sum_decode_kv_tokens > 0 {
        return Err(AicError::InvalidForwardPassMetrics(
            "decode KV token sum requires num_decode_requests > 0".to_string(),
        ));
    }
    Ok(())
}

fn default_fpm_version() -> u32 {
    FPM_VERSION
}
