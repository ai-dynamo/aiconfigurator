// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::AicError;

#[derive(Clone, Debug)]
pub(crate) struct DenseModelConfig {
    pub(crate) num_hidden_layers: u32,
    pub(crate) num_attention_heads: u32,
    pub(crate) num_key_value_heads: u32,
    pub(crate) head_dim: u32,
    pub(crate) hidden_size: u32,
    pub(crate) intermediate_size: u32,
    pub(crate) vocab_size: u32,
    is_moe: bool,
}

impl DenseModelConfig {
    pub(crate) fn load(model_name: &str, root: &Path) -> Result<Self, AicError> {
        let path = resolve_model_config_path(model_name, root)?;
        let text = fs::read_to_string(&path).map_err(|source| AicError::Io {
            path: path.clone(),
            source,
        })?;
        let value: Value = serde_json::from_str(&text).map_err(|source| AicError::Json {
            path: path.clone(),
            source,
        })?;
        Self::from_value(&value, &path)
    }

    pub(crate) fn validate_supported(&self) -> Result<(), AicError> {
        if self.is_moe {
            return Err(AicError::UnsupportedModel(
                "Phase 1 Rust estimator supports dense transformer models; MoE support will be added later"
                    .to_string(),
            ));
        }
        Ok(())
    }

    pub(crate) fn kv_heads_per_gpu(&self, tp_size: u32) -> u32 {
        let tp_size = tp_size.max(1);
        (self.num_key_value_heads + tp_size - 1) / tp_size
    }

    fn from_value(value: &Value, path: &Path) -> Result<Self, AicError> {
        let num_hidden_layers = required_u32(value, "num_hidden_layers", path)?;
        let num_attention_heads = required_u32(value, "num_attention_heads", path)?;
        let hidden_size = required_u32(value, "hidden_size", path)?;
        let intermediate_size = required_u32(value, "intermediate_size", path)?;
        let vocab_size = required_u32(value, "vocab_size", path)?;
        let num_key_value_heads =
            optional_u32(value, "num_key_value_heads", path)?.unwrap_or(num_attention_heads);
        let head_dim =
            optional_u32(value, "head_dim", path)?.unwrap_or(hidden_size / num_attention_heads);
        let is_moe = [
            "num_experts",
            "n_routed_experts",
            "num_local_experts",
            "num_experts_per_tok",
            "moe_intermediate_size",
        ]
        .iter()
        .any(|key| value.get(*key).is_some());

        Ok(Self {
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            hidden_size,
            intermediate_size,
            vocab_size,
            is_moe,
        })
    }
}

fn resolve_model_config_path(model_name: &str, root: &Path) -> Result<PathBuf, AicError> {
    let requested = Path::new(model_name);
    if requested.is_file() {
        return Ok(requested.to_path_buf());
    }
    if requested.is_dir() {
        let candidate = requested.join("config.json");
        if candidate.is_file() {
            return Ok(candidate);
        }
    }

    let sanitized = model_name.replace('/', "--");
    let direct_candidates = [
        root.join(format!("{sanitized}_config.json")),
        root.join(format!("{sanitized}_hf_quant_config.json")),
        root.join(&sanitized).join("config.json"),
    ];
    for candidate in direct_candidates {
        if candidate.is_file() {
            return Ok(candidate);
        }
    }

    let entries = fs::read_dir(root).map_err(|source| AicError::Io {
        path: root.to_path_buf(),
        source,
    })?;
    for entry in entries {
        let entry = entry.map_err(|source| AicError::Io {
            path: root.to_path_buf(),
            source,
        })?;
        let path = entry.path();
        let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if file_name.starts_with(&sanitized) && file_name.ends_with("_config.json") {
            return Ok(path);
        }
    }

    Err(AicError::ModelConfig(format!(
        "could not find config for model '{model_name}' under {}",
        root.display()
    )))
}

fn required_u32(value: &Value, key: &str, path: &Path) -> Result<u32, AicError> {
    optional_u32(value, key, path)?.ok_or_else(|| {
        AicError::ModelConfig(format!(
            "missing required field '{key}' in model config {}",
            path.display()
        ))
    })
}

fn optional_u32(value: &Value, key: &str, path: &Path) -> Result<Option<u32>, AicError> {
    let Some(raw) = value.get(key) else {
        return Ok(None);
    };
    let Some(number) = raw.as_u64() else {
        return Err(AicError::ModelConfig(format!(
            "field '{key}' in model config {} must be an unsigned integer",
            path.display()
        )));
    };
    u32::try_from(number).map(Some).map_err(|_| {
        AicError::ModelConfig(format!(
            "field '{key}' in model config {} is too large for u32",
            path.display()
        ))
    })
}
