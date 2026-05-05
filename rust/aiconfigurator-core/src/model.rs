// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::AicError;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModelFamily {
    Gpt,
    Llama,
    Moe,
    DeepSeek,
    DeepSeekV32,
    DeepSeekV4,
    KimiK25,
    NemotronNas,
    NemotronH,
    HybridMoe,
    Qwen35,
}

#[derive(Clone, Debug)]
pub struct ModelSpec {
    pub architecture: String,
    pub family: ModelFamily,
    pub num_hidden_layers: u32,
    pub num_attention_heads: u32,
    pub num_key_value_heads: u32,
    pub head_dim: u32,
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub vocab_size: u32,
    pub context_length: u32,
    pub top_k: u32,
    pub num_experts: u32,
    pub moe_intermediate_size: u32,
}

impl ModelSpec {
    pub fn load(model_name: &str, root: &Path) -> Result<Self, AicError> {
        let path = resolve_model_config_path(model_name, root)?;
        Self::load_path(&path)
    }

    pub fn load_path(path: &Path) -> Result<Self, AicError> {
        let value = read_json(path)?;
        if value.get("architectures").is_none() {
            if let Some(base_path) = resolve_hf_quant_base_config(path) {
                return Self::load_path(&base_path);
            }
        }
        Self::from_value(&value, path)
    }

    pub fn kv_heads_per_gpu(&self, tp_size: u32) -> u32 {
        let tp_size = tp_size.max(1);
        (self.num_key_value_heads + tp_size - 1) / tp_size
    }

    pub fn uses_moe(&self) -> bool {
        matches!(
            self.family,
            ModelFamily::Moe
                | ModelFamily::DeepSeek
                | ModelFamily::DeepSeekV32
                | ModelFamily::DeepSeekV4
                | ModelFamily::KimiK25
                | ModelFamily::HybridMoe
        ) || (matches!(self.family, ModelFamily::NemotronH | ModelFamily::Qwen35)
            && self.num_experts > 0
            && self.top_k > 0)
    }

    pub fn uses_mla_attention(&self) -> bool {
        matches!(
            self.family,
            ModelFamily::DeepSeek | ModelFamily::DeepSeekV32 | ModelFamily::KimiK25
        )
    }

    pub fn uses_module_attention(&self) -> bool {
        matches!(self.family, ModelFamily::DeepSeekV4 | ModelFamily::Qwen35)
    }

    fn from_value(value: &Value, path: &Path) -> Result<Self, AicError> {
        let architecture = architecture(value, path)?;
        let model_value = llm_config_value(value, &architecture);
        let family = architecture_to_family(&architecture)?;

        let num_hidden_layers = required_u32(model_value, "num_hidden_layers", path)?;
        let num_attention_heads = required_u32(model_value, "num_attention_heads", path)?;
        let hidden_size = required_u32(model_value, "hidden_size", path)?;
        let intermediate_size = optional_u32(model_value, "intermediate_size", path)?.unwrap_or(0);
        let vocab_size = required_u32(model_value, "vocab_size", path)?;
        let context_length = optional_u32(model_value, "max_position_embeddings", path)?
            .or(optional_u32(model_value, "seq_length", path)?)
            .or(optional_u32(model_value, "max_seq_len", path)?)
            .unwrap_or(0);
        let num_key_value_heads =
            optional_u32(model_value, "num_key_value_heads", path)?.unwrap_or(num_attention_heads);
        let head_dim = optional_u32(model_value, "head_dim", path)?
            .or(optional_u32(model_value, "attention_head_dim", path)?)
            .unwrap_or(hidden_size / num_attention_heads.max(1));
        let top_k = optional_u32(model_value, "num_experts_per_tok", path)?.unwrap_or(0);
        let num_experts = optional_u32(model_value, "num_local_experts", path)?
            .or(optional_u32(model_value, "n_routed_experts", path)?)
            .or(optional_u32(model_value, "num_experts", path)?)
            .unwrap_or(0);
        let moe_intermediate_size =
            optional_u32(model_value, "moe_intermediate_size", path)?.unwrap_or(intermediate_size);

        Ok(Self {
            architecture,
            family,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            hidden_size,
            intermediate_size,
            vocab_size,
            context_length,
            top_k,
            num_experts,
            moe_intermediate_size,
        })
    }
}

fn read_json(path: &Path) -> Result<Value, AicError> {
    let text = fs::read_to_string(path).map_err(|source| AicError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    match serde_json::from_str(&text) {
        Ok(value) => Ok(value),
        Err(first_error) => {
            let sanitized = text
                .replace("-Infinity", "null")
                .replace("Infinity", "null")
                .replace("NaN", "null");
            serde_json::from_str(&sanitized).map_err(|_| AicError::Json {
                path: path.to_path_buf(),
                source: first_error,
            })
        }
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

fn resolve_hf_quant_base_config(path: &Path) -> Option<PathBuf> {
    let file_name = path.file_name()?.to_str()?;
    let base_name = file_name.replace("_hf_quant_config.json", "_config.json");
    if base_name == file_name {
        return None;
    }
    let base_path = path.with_file_name(base_name);
    base_path.is_file().then_some(base_path)
}

fn architecture(value: &Value, path: &Path) -> Result<String, AicError> {
    value
        .get("architectures")
        .and_then(Value::as_array)
        .and_then(|items| items.first())
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| {
            AicError::ModelConfig(format!(
                "missing required architectures[0] in model config {}",
                path.display()
            ))
        })
}

fn llm_config_value<'a>(value: &'a Value, architecture: &str) -> &'a Value {
    let text_key = match architecture {
        "KimiK25ForConditionalGeneration"
        | "Llama4ForConditionalGeneration"
        | "Qwen3_5ForConditionalGeneration"
        | "Qwen3_5MoeForConditionalGeneration" => Some("text_config"),
        _ => None,
    };
    text_key
        .and_then(|key| value.get(key))
        .filter(|nested| nested.is_object())
        .unwrap_or(value)
}

fn architecture_to_family(architecture: &str) -> Result<ModelFamily, AicError> {
    let family = match architecture {
        "GPT" => ModelFamily::Gpt,
        "LLAMA" | "LlamaForCausalLM" | "Qwen2ForCausalLM" | "Qwen3ForCausalLM"
        | "MiMoForCausalLM" => ModelFamily::Llama,
        "MOE"
        | "MixtralForCausalLM"
        | "GptOssForCausalLM"
        | "Qwen2MoeForCausalLM"
        | "Qwen3MoeForCausalLM"
        | "MiniMaxM2ForCausalLM" => ModelFamily::Moe,
        "DEEPSEEK" | "DeepSeekForCausalLM" | "DeepseekV3ForCausalLM" => ModelFamily::DeepSeek,
        "DEEPSEEKV32" | "DeepseekV32ForCausalLM" | "GlmMoeDsaForCausalLM" => {
            ModelFamily::DeepSeekV32
        }
        "DEEPSEEKV4" | "DeepseekV4ForCausalLM" => ModelFamily::DeepSeekV4,
        "KIMIK25" | "KimiK25ForConditionalGeneration" => ModelFamily::KimiK25,
        "NEMOTRONNAS" | "NemotronForCausalLM" | "DeciLMForCausalLM" => ModelFamily::NemotronNas,
        "NEMOTRONH" | "NemotronHForCausalLM" => ModelFamily::NemotronH,
        "HYBRIDMOE" | "MiMoV2FlashForCausalLM" | "Llama4ForConditionalGeneration" => {
            ModelFamily::HybridMoe
        }
        "QWEN35" | "Qwen3_5ForConditionalGeneration" | "Qwen3_5MoeForConditionalGeneration" => {
            ModelFamily::Qwen35
        }
        _ => {
            return Err(AicError::UnsupportedModel(format!(
                "architecture '{architecture}' is not mapped to an AIC model family"
            )))
        }
    };
    Ok(family)
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
    if raw.is_null() {
        return Ok(None);
    }
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
