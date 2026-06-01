// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HuggingFace architecture → AIC model family routing.
//!
//! Mirrors `common.ARCHITECTURE_TO_MODEL_FAMILY` and
//! `common.MULTIMODAL_TEXT_CONFIG_KEY` from
//! `src/aiconfigurator/sdk/common.py`. Lives in `models/` (not
//! `common/`) because the routing tables encode AIC-domain knowledge
//! about which family each HuggingFace architecture should be modeled
//! as.

use crate::common::enums::ModelFamily;

/// Map a HuggingFace `architectures[0]` string to its AIC model family.
///
/// Returns `None` for architectures AIC does not yet recognize.
pub fn architecture_to_family(architecture: &str) -> Option<ModelFamily> {
    Some(match architecture {
        "LlamaForCausalLM"
        | "Qwen2ForCausalLM"
        | "Qwen3ForCausalLM"
        | "MiMoForCausalLM" => ModelFamily::Llama,
        "Qwen3VLForConditionalGeneration" => ModelFamily::Qwen3Vl,
        "Qwen3VLMoeForConditionalGeneration" => ModelFamily::Qwen3VlMoe,
        "DeepSeekForCausalLM" | "DeepseekV3ForCausalLM" => ModelFamily::DeepSeek,
        "DeepseekV32ForCausalLM" | "GlmMoeDsaForCausalLM" => ModelFamily::DeepSeekV32,
        "DeepseekV4ForCausalLM" => ModelFamily::DeepSeekV4,
        "KimiK25ForConditionalGeneration" => ModelFamily::KimiK25,
        "NemotronForCausalLM" | "DeciLMForCausalLM" => ModelFamily::NemotronNas,
        "NemotronHForCausalLM" => ModelFamily::NemotronH,
        "MixtralForCausalLM"
        | "GptOssForCausalLM"
        | "Qwen2MoeForCausalLM"
        | "Qwen3MoeForCausalLM"
        | "MiniMaxM2ForCausalLM" => ModelFamily::Moe,
        "MiMoV2FlashForCausalLM" | "Llama4ForConditionalGeneration" => ModelFamily::HybridMoe,
        "Qwen3_5ForConditionalGeneration" | "Qwen3_5MoeForConditionalGeneration" => {
            ModelFamily::Qwen35
        }
        "Gemma4ForConditionalGeneration" => ModelFamily::Gemma4Moe,
        _ => return None,
    })
}

/// Multimodal architectures whose LLM config lives under a nested key in the
/// HF `config.json`.
pub fn multimodal_text_config_key(architecture: &str) -> Option<&'static str> {
    Some(match architecture {
        "KimiK25ForConditionalGeneration"
        | "Llama4ForConditionalGeneration"
        | "Qwen3_5ForConditionalGeneration"
        | "Qwen3_5MoeForConditionalGeneration"
        | "Gemma4ForConditionalGeneration"
        | "Qwen3VLForConditionalGeneration"
        | "Qwen3VLMoeForConditionalGeneration" => "text_config",
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn architecture_to_family_smoke_models() {
        assert_eq!(
            architecture_to_family("MiniMaxM2ForCausalLM"),
            Some(ModelFamily::Moe)
        );
        assert_eq!(
            architecture_to_family("KimiK25ForConditionalGeneration"),
            Some(ModelFamily::KimiK25)
        );
    }

    #[test]
    fn architecture_to_family_qwen3vl_split() {
        // QWEN3VL split: dense vs MoE map to distinct model families.
        assert_eq!(
            architecture_to_family("Qwen3VLForConditionalGeneration"),
            Some(ModelFamily::Qwen3Vl)
        );
        assert_eq!(
            architecture_to_family("Qwen3VLMoeForConditionalGeneration"),
            Some(ModelFamily::Qwen3VlMoe)
        );
    }

    #[test]
    fn architecture_to_family_unknown_is_none() {
        assert!(architecture_to_family("NotARealArchitecture").is_none());
    }

    #[test]
    fn multimodal_text_config_key_matches_python() {
        for arch in [
            "KimiK25ForConditionalGeneration",
            "Llama4ForConditionalGeneration",
            "Qwen3_5ForConditionalGeneration",
            "Qwen3_5MoeForConditionalGeneration",
            "Gemma4ForConditionalGeneration",
            "Qwen3VLForConditionalGeneration",
            "Qwen3VLMoeForConditionalGeneration",
        ] {
            assert_eq!(multimodal_text_config_key(arch), Some("text_config"), "{arch}");
        }
        assert!(multimodal_text_config_key("LlamaForCausalLM").is_none());
        assert!(multimodal_text_config_key("Qwen3MoeForCausalLM").is_none());
    }
}
