// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tuning controls for the forward-pass perf model.

use serde::{Deserialize, Serialize};

use crate::AicError;

use super::samples::integer_sqrt;

pub(crate) const DEFAULT_MAX_OBSERVATIONS: usize = 64;
pub(crate) const DEFAULT_MIN_OBSERVATIONS: usize = 5;
pub(crate) const DEFAULT_MIN_FASTER_CORRECTION_FACTOR: f64 = 0.5;
pub(crate) const DEFAULT_MAX_SLOWER_CORRECTION_FACTOR: f64 = 2.0;
pub(crate) const DEFAULT_BUCKET_COUNT: usize = 16;
pub(crate) const DEFAULT_MAX_NUM_TOKENS: u32 = 8192;
pub(crate) const DEFAULT_MAX_BATCH_SIZE: u32 = 512;
pub(crate) const DEFAULT_MAX_KV_TOKENS: u32 = 2_000_000;
pub(crate) const DEFAULT_REGRESSION_ATTENTION_KV_WEIGHT: f64 = 1.0;
pub(crate) const DEFAULT_REGRESSION_PREFILL_ATTENTION_PAIR_WEIGHT: f64 = 1.0;
pub(crate) const DEFAULT_REGRESSION_FFN_TOKEN_WEIGHT: f64 = 1.0;

/// In-memory tuning controls for `ForwardPassPerfModel`.
///
/// The defaults retain a bounded sliding sample set, wait for enough
/// observations before predicting from learned data, and bound native
/// correction factors to `[0.5, 2.0]`. Native correction retains observations
/// per inferred workload kind; regression retains one set for its fixed worker
/// type.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ForwardPassPerfOptions {
    /// Maximum retained observations across all buckets. The cap applies per
    /// inferred workload kind for Native and once per role-bound Regression
    /// model.
    #[serde(default = "default_max_observations")]
    pub max_observations: usize,
    /// Minimum retained observations required before a regression fit or
    /// native correction is used. Native applies it per inferred workload
    /// kind; Regression applies it to its single store.
    #[serde(default = "default_min_observations")]
    pub min_observations: usize,
    /// Optional absolute lower bound on native correction factors for
    /// observations faster than the native estimate.
    ///
    /// Values must be finite, greater than `0.0`, and at most `1.0`. Defaults
    /// to `0.5`, limiting learned speedups to `2x`. Setting this to `1.0`
    /// disables faster corrections; setting it to `None` removes the lower
    /// bound. Regression fallback does not use this option.
    #[serde(default = "default_min_faster_correction_factor")]
    pub min_faster_correction_factor: Option<f64>,
    /// Optional absolute upper bound on native correction factors for
    /// observations slower than the native estimate.
    ///
    /// Values must be finite and at least `1.0`. Defaults to `2.0`, limiting
    /// learned slowdowns to `2x`. Setting this to `1.0` disables slower
    /// corrections; setting it to `None` removes the upper bound. Regression
    /// fallback does not use this option.
    #[serde(default = "default_max_slower_correction_factor")]
    pub max_slower_correction_factor: Option<f64>,
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
    /// Weight applied to rank-local KV-transfer work in regression features.
    ///
    /// Regression-only. Must be finite and strictly positive. Native AIC and
    /// native correction ignore this option.
    #[serde(
        default = "default_regression_attention_kv_weight",
        with = "regression_weight_serde"
    )]
    pub regression_attention_kv_weight: f64,
    /// Weight applied to the balanced-request Prefill attention-pair proxy in
    /// regression features.
    ///
    /// Regression-only. Must be finite and strictly positive. Native AIC and
    /// native correction ignore this option.
    #[serde(
        default = "default_regression_prefill_attention_pair_weight",
        with = "regression_weight_serde"
    )]
    pub regression_prefill_attention_pair_weight: f64,
    /// Weight applied to global tokenwise FFN/MoE work in regression features.
    ///
    /// Regression-only. Must be finite and strictly positive. Native AIC and
    /// native correction ignore this option.
    #[serde(
        default = "default_regression_ffn_token_weight",
        with = "regression_weight_serde"
    )]
    pub regression_ffn_token_weight: f64,
}

/// JSON cannot represent nonfinite numbers. The Python facade transports them
/// through the options JSON as these exact strings so Native construction can
/// continue to ignore regression-only weights while Regression construction
/// still reports its existing field-specific validation error.
mod regression_weight_serde {
    use std::fmt;

    use serde::{
        de::{self, Visitor},
        Deserialize, Deserializer, Serializer,
    };

    const NAN: &str = "NaN";
    const POSITIVE_INFINITY: &str = "Infinity";
    const NEGATIVE_INFINITY: &str = "-Infinity";

    pub(super) fn serialize<S>(value: &f64, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        if !serializer.is_human_readable() || value.is_finite() {
            serializer.serialize_f64(*value)
        } else if value.is_nan() {
            serializer.serialize_str(NAN)
        } else if value.is_sign_positive() {
            serializer.serialize_str(POSITIVE_INFINITY)
        } else {
            serializer.serialize_str(NEGATIVE_INFINITY)
        }
    }

    pub(super) fn deserialize<'de, D>(deserializer: D) -> Result<f64, D::Error>
    where
        D: Deserializer<'de>,
    {
        if deserializer.is_human_readable() {
            deserializer.deserialize_any(RegressionWeightVisitor)
        } else {
            f64::deserialize(deserializer)
        }
    }

    struct RegressionWeightVisitor;

    impl Visitor<'_> for RegressionWeightVisitor {
        type Value = f64;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str(
                "a finite number or one of the exact strings \"NaN\", \"Infinity\", or \"-Infinity\"",
            )
        }

        fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            if value.is_finite() {
                Ok(value)
            } else {
                Err(E::custom(
                    "regression weights encoded as numbers must be finite",
                ))
            }
        }

        fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(value as f64)
        }

        fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(value as f64)
        }

        fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            match value {
                NAN => Ok(f64::NAN),
                POSITIVE_INFINITY => Ok(f64::INFINITY),
                NEGATIVE_INFINITY => Ok(f64::NEG_INFINITY),
                _ => Err(E::invalid_value(de::Unexpected::Str(value), &self)),
            }
        }
    }
}

impl Default for ForwardPassPerfOptions {
    fn default() -> Self {
        Self {
            max_observations: DEFAULT_MAX_OBSERVATIONS,
            min_observations: DEFAULT_MIN_OBSERVATIONS,
            min_faster_correction_factor: default_min_faster_correction_factor(),
            max_slower_correction_factor: default_max_slower_correction_factor(),
            bucket_count: DEFAULT_BUCKET_COUNT,
            max_num_tokens: DEFAULT_MAX_NUM_TOKENS,
            max_batch_size: DEFAULT_MAX_BATCH_SIZE,
            max_kv_tokens: DEFAULT_MAX_KV_TOKENS,
            regression_attention_kv_weight: DEFAULT_REGRESSION_ATTENTION_KV_WEIGHT,
            regression_prefill_attention_pair_weight:
                DEFAULT_REGRESSION_PREFILL_ATTENTION_PAIR_WEIGHT,
            regression_ffn_token_weight: DEFAULT_REGRESSION_FFN_TOKEN_WEIGHT,
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
    if let Some(min_faster_correction_factor) = options.min_faster_correction_factor {
        if !min_faster_correction_factor.is_finite()
            || min_faster_correction_factor <= 0.0
            || min_faster_correction_factor > 1.0
        {
            return Err(invalid_perf_options(
                "min_faster_correction_factor must be finite and in (0.0, 1.0]",
            ));
        }
    }
    if let Some(max_slower_correction_factor) = options.max_slower_correction_factor {
        if !max_slower_correction_factor.is_finite() || max_slower_correction_factor < 1.0 {
            return Err(invalid_perf_options(
                "max_slower_correction_factor must be finite and >= 1.0",
            ));
        }
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

pub(crate) fn validate_regression_options(
    options: &ForwardPassPerfOptions,
) -> Result<(), AicError> {
    validate_options(options)?;
    for (name, value) in [
        (
            "regression_attention_kv_weight",
            options.regression_attention_kv_weight,
        ),
        (
            "regression_prefill_attention_pair_weight",
            options.regression_prefill_attention_pair_weight,
        ),
        (
            "regression_ffn_token_weight",
            options.regression_ffn_token_weight,
        ),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(invalid_perf_options(&format!(
                "{name} must be finite and > 0.0"
            )));
        }
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

fn default_min_faster_correction_factor() -> Option<f64> {
    Some(DEFAULT_MIN_FASTER_CORRECTION_FACTOR)
}

fn default_max_slower_correction_factor() -> Option<f64> {
    Some(DEFAULT_MAX_SLOWER_CORRECTION_FACTOR)
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

fn default_regression_attention_kv_weight() -> f64 {
    DEFAULT_REGRESSION_ATTENTION_KV_WEIGHT
}

fn default_regression_prefill_attention_pair_weight() -> f64 {
    DEFAULT_REGRESSION_PREFILL_ATTENTION_PAIR_WEIGHT
}

fn default_regression_ffn_token_weight() -> f64 {
    DEFAULT_REGRESSION_FFN_TOKEN_WEIGHT
}

#[cfg(test)]
mod tests {
    use super::ForwardPassPerfOptions;

    const WEIGHT_FIELDS: [&str; 3] = [
        "regression_attention_kv_weight",
        "regression_prefill_attention_pair_weight",
        "regression_ffn_token_weight",
    ];

    fn weight(options: &ForwardPassPerfOptions, field: &str) -> f64 {
        match field {
            "regression_attention_kv_weight" => options.regression_attention_kv_weight,
            "regression_prefill_attention_pair_weight" => {
                options.regression_prefill_attention_pair_weight
            }
            "regression_ffn_token_weight" => options.regression_ffn_token_weight,
            _ => unreachable!("test only calls known weight fields"),
        }
    }

    #[test]
    fn finite_regression_weights_remain_json_numbers() {
        let options = ForwardPassPerfOptions {
            regression_attention_kv_weight: 2.25,
            regression_prefill_attention_pair_weight: 3.5,
            regression_ffn_token_weight: 4.75,
            ..Default::default()
        };

        let json = serde_json::to_value(&options).unwrap();
        assert_eq!(json[WEIGHT_FIELDS[0]], serde_json::json!(2.25));
        assert_eq!(json[WEIGHT_FIELDS[1]], serde_json::json!(3.5));
        assert_eq!(json[WEIGHT_FIELDS[2]], serde_json::json!(4.75));
        assert_eq!(
            serde_json::from_value::<ForwardPassPerfOptions>(json).unwrap(),
            options
        );
    }

    #[test]
    fn every_regression_weight_accepts_each_exact_nonfinite_sentinel() {
        for field in WEIGHT_FIELDS {
            for (sentinel, expected) in [
                ("NaN", f64::NAN),
                ("Infinity", f64::INFINITY),
                ("-Infinity", f64::NEG_INFINITY),
            ] {
                let json = format!(r#"{{"{field}":"{sentinel}"}}"#);
                let options: ForwardPassPerfOptions = serde_json::from_str(&json).unwrap();
                let actual = weight(&options, field);
                if expected.is_nan() {
                    assert!(actual.is_nan(), "{field} did not decode {sentinel}");
                } else {
                    assert_eq!(actual, expected, "{field} did not decode {sentinel}");
                }
            }
        }
    }

    #[test]
    fn nonfinite_regression_weights_serialize_as_sentinels_and_round_trip() {
        let options = ForwardPassPerfOptions {
            regression_attention_kv_weight: f64::NAN,
            regression_prefill_attention_pair_weight: f64::INFINITY,
            regression_ffn_token_weight: f64::NEG_INFINITY,
            ..Default::default()
        };

        let json = serde_json::to_value(&options).unwrap();
        assert_eq!(json[WEIGHT_FIELDS[0]], serde_json::json!("NaN"));
        assert_eq!(json[WEIGHT_FIELDS[1]], serde_json::json!("Infinity"));
        assert_eq!(json[WEIGHT_FIELDS[2]], serde_json::json!("-Infinity"));

        let round_trip: ForwardPassPerfOptions = serde_json::from_value(json).unwrap();
        assert!(round_trip.regression_attention_kv_weight.is_nan());
        assert_eq!(
            round_trip.regression_prefill_attention_pair_weight,
            f64::INFINITY
        );
        assert_eq!(round_trip.regression_ffn_token_weight, f64::NEG_INFINITY);
    }

    #[test]
    fn omitted_regression_weights_keep_their_defaults() {
        let omitted: ForwardPassPerfOptions = serde_json::from_str("{}").unwrap();
        let defaults = ForwardPassPerfOptions::default();
        assert_eq!(
            omitted.regression_attention_kv_weight,
            defaults.regression_attention_kv_weight
        );
        assert_eq!(
            omitted.regression_prefill_attention_pair_weight,
            defaults.regression_prefill_attention_pair_weight
        );
        assert_eq!(
            omitted.regression_ffn_token_weight,
            defaults.regression_ffn_token_weight
        );
    }

    #[test]
    fn regression_weights_reject_unknown_sentinels_and_non_numeric_types() {
        for field in WEIGHT_FIELDS {
            for invalid_json_value in [
                r#""nan""#,
                r#""Inf""#,
                r#""infinity""#,
                "null",
                "true",
                "[]",
                "{}",
            ] {
                let json = format!(r#"{{"{field}":{invalid_json_value}}}"#);
                assert!(
                    serde_json::from_str::<ForwardPassPerfOptions>(&json).is_err(),
                    "{field} unexpectedly accepted {invalid_json_value}"
                );
            }
        }
    }
}
