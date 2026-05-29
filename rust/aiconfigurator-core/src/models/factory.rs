// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Model factory: build a `Model` (op lists populated) from a
//! `ModelConfig` by routing on `ModelFamily`.

use crate::common::enums::ModelFamily;
use crate::common::error::AicError;
use crate::models::base::{Model, ModelConfig};
use crate::models::base::WideEpMode;
use crate::models::deepseek::build_deepseek_model;
use crate::models::deepseek_v32::build_deepseek_v32_model;
use crate::models::deepseek_v4::build_deepseek_v4_model;
use crate::models::deepseek_wideep::build_sglang_wideep_deepseek_model;
use crate::models::deepseek_wideep_trtllm::build_trtllm_wideep_deepseek_model;
use crate::models::gemma4_moe::build_gemma4_moe_model;
use crate::models::gpt::build_gpt_model;
use crate::models::hybrid_moe::build_hybrid_moe_model;
use crate::models::llama::build_llama_model;
use crate::models::moe::build_moe_model;
use crate::models::nemotron_h::build_nemotron_h_model;
use crate::models::nemotron_nas::build_nemotron_nas_model;
use crate::models::qwen35::build_qwen35_model;
use crate::models::qwen3vl::{build_qwen3vl_model, build_qwen3vl_moe_model};

pub fn build_model(config: ModelConfig) -> Result<Model, AicError> {
    let family = config.spec.family;
    match family {
        ModelFamily::Llama => Ok(build_llama_model(config)),
        ModelFamily::Qwen3Vl => build_qwen3vl_model(config),
        ModelFamily::Moe => Ok(build_moe_model(config)),
        ModelFamily::Qwen3VlMoe => build_qwen3vl_moe_model(config),
        ModelFamily::HybridMoe => build_hybrid_moe_model(config),
        ModelFamily::DeepSeek => match config.wideep_mode {
            // Python `WideEPDeepSeekModel.create` only fires on
            // `DEEPSEEK` (Kimi-K25 reuses the architecture but skips
            // the WideEP branch — matches Python's
            // `if family == "KIMIK25": return cls(...)` early-return).
            WideEpMode::SglangDeepEp => Ok(build_sglang_wideep_deepseek_model(config)),
            WideEpMode::Trtllm => build_trtllm_wideep_deepseek_model(config),
            WideEpMode::Off => Ok(build_deepseek_model(config)),
        },
        ModelFamily::KimiK25 => Ok(build_deepseek_model(config)),
        ModelFamily::DeepSeekV4 => build_deepseek_v4_model(config),
        ModelFamily::DeepSeekV32 => Ok(build_deepseek_v32_model(config)),
        ModelFamily::NemotronNas => Ok(build_nemotron_nas_model(config)),
        ModelFamily::NemotronH => Ok(build_nemotron_h_model(config)),
        ModelFamily::Qwen35 => Ok(build_qwen35_model(config)),
        ModelFamily::Gpt => Ok(build_gpt_model(config)),
        ModelFamily::Gemma4Moe => build_gemma4_moe_model(config),
    }
}
