// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HuggingFace `config.json` and `hf_quant_config.json` parsing for AIC.
//!
//! Mirrors the parsing pieces of `aiconfigurator.sdk.models.helpers` and
//! `aiconfigurator.sdk.utils`: resolve a model path to a `config.json`,
//! flatten the multimodal `text_config` nesting, and return a typed view
//! of the numeric fields used by op-graph builders. Quantization defaults
//! inference lives alongside the per-family model code that consumes it.

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::common::enums::ModelFamily;
use crate::common::error::AicError;
use crate::models::registry::{architecture_to_family, multimodal_text_config_key};

/// Typed view of the LLM-relevant fields of a HuggingFace `config.json`.
///
/// Vision-tower and other multimodal fields are excluded; only the text
/// backbone is needed by the op-graph builders.
#[derive(Clone, Debug, PartialEq)]
pub struct HfModelConfig {
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
    /// Number of experts activated per token (`num_experts_per_tok` /
    /// `top_k_experts`). Zero for dense models.
    pub top_k: u32,
    /// Total expert count (`num_local_experts` / `n_routed_experts` /
    /// `num_experts`). Zero for dense models.
    pub num_experts: u32,
    /// Per-expert FFN intermediate size; falls back to `intermediate_size`
    /// when the model config does not provide one.
    pub moe_intermediate_size: u32,
    /// Shared (always-on) expert FFN intermediate size; zero when absent.
    pub shared_expert_intermediate_size: u32,
    /// QK-norm flag; enabled by some models (MiniMax-M2, GptOss, etc.).
    /// When true, attention ops include extra elementwise work for the
    /// QK normalization layers.
    pub use_qk_norm: bool,
    /// NemotronNas/Puzzle-style per-block config (`DeciLMForCausalLM`).
    /// Empty for non-NAS models. Each entry corresponds to ONE layer of
    /// the model; the model builder may iterate them directly (Python's
    /// `_parse_nemotron_block_configs` groups consecutive identical
    /// blocks but the latency sum is invariant to grouping).
    pub block_configs: Vec<BlockConfig>,
    /// Qwen3.5 hybrid (GDN + full-attention) config. `None` for non-Qwen3.5
    /// architectures.
    pub qwen35: Option<Qwen35Config>,
    /// NemotronH hybrid (Mamba + attention + MLP/MoE) config. `None` for
    /// non-NemotronH architectures.
    pub nemotron_h: Option<NemotronHConfig>,
    /// DeepSeek-V4 compressed-attention + mHC config. `None` for non-DSv4
    /// architectures.
    pub deepseek_v4: Option<DeepSeekV4Config>,
    /// HybridMoE (Llama-4 et al.) per-layer pattern + SWA dims. `None`
    /// for non-HybridMoE architectures.
    pub hybrid_moe: Option<HybridMoEConfig>,
    /// Gemma-4 hybrid SWA/global + shared-MLP-plus-MoE config. `None`
    /// for non-Gemma4 architectures.
    pub gemma4_moe: Option<Gemma4MoEConfig>,
    /// Vision encoder config for multimodal VL models (Qwen3VL,
    /// Qwen3VLMoE). `None` for text-only architectures.
    pub vision_encoder: Option<VisionEncoderConfig>,
}

/// Per-family extras for Qwen3.5 (`Qwen3_5ForConditionalGeneration` and
/// `Qwen3_5MoeForConditionalGeneration`). Mirrors Python's
/// `common.Qwen35Config`.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen35Config {
    /// Per-layer attention type: `"linear_attention"` (GDN) or
    /// `"full_attention"` (standard GQA). Length == `num_hidden_layers`.
    pub layer_types: Vec<String>,
    pub linear_num_key_heads: u32,
    pub linear_key_head_dim: u32,
    pub linear_num_value_heads: u32,
    pub linear_value_head_dim: u32,
    pub linear_conv_kernel_dim: u32,
}

/// Per-family extras for NemotronH (`NemotronHForCausalLM`). Mirrors
/// Python's `common.NemotronHConfig`.
#[derive(Clone, Debug, PartialEq)]
pub struct NemotronHConfig {
    /// Per-layer marker string: `M` (Mamba2), `E` (MoE), `*`
    /// (transformer), `-` (dense MLP). Length == `num_hidden_layers`.
    pub hybrid_override_pattern: String,
    pub mamba_num_heads: u32,
    pub mamba_head_dim: u32,
    pub ssm_state_size: u32,
    pub conv_kernel: u32,
    pub n_groups: u32,
    pub chunk_size: u32,
    pub moe_shared_expert_intermediate_size: u32,
    /// Latent compression dim for routed experts (Nemotron-3-Super);
    /// 0 means no latent compression.
    pub moe_latent_size: u32,
}

/// Vision encoder (ViT) configuration for multimodal VL models
/// (Qwen3VL, Qwen3VLMoE). Mirrors Python's `common.VisionEncoderConfig`.
/// `projector_dims` is the per-layer `(in_dim, out_dim)` pair list for
/// the vision-to-LLM projector MLP (empty when no projector).
#[derive(Clone, Debug, PartialEq)]
pub struct VisionEncoderConfig {
    pub depth: u32,
    pub hidden_size: u32,
    pub num_heads: u32,
    pub intermediate_size: u32,
    pub patch_size: u32,
    pub temporal_patch_size: u32,
    pub spatial_merge_size: u32,
    pub out_hidden_size: u32,
    pub deepstack_visual_indexes: Vec<u32>,
    pub projector_dims: Vec<(u32, u32)>,
    pub projector_n_instances: u32,
}

/// Per-family extras for GEMMA4MOE (`Gemma4ForConditionalGeneration`).
/// Mirrors Python's `common.Gemma4MoEConfig`. Every layer runs both a
/// shared dense MLP and a routed MoE in parallel; attention alternates
/// `sliding_attention` (SWA) and `full_attention` (global) per the
/// `layer_types` array.
#[derive(Clone, Debug, PartialEq)]
pub struct Gemma4MoEConfig {
    /// Per-layer attention type. Each entry must be either
    /// `"sliding_attention"` or `"full_attention"`. Length ==
    /// `num_hidden_layers`.
    pub layer_types: Vec<String>,
    pub swa_num_kv_heads: u32,
    pub swa_head_dim: u32,
    pub global_num_kv_heads: u32,
    pub global_head_dim: u32,
    pub sliding_window_size: u32,
    /// When true, global layers reuse the K projection output as V
    /// (no v_proj GEMM); the QKV-out width collapses from Q+2KV to Q+K.
    pub attention_k_eq_v: bool,
}

/// Per-family extras for HYBRIDMOE (Llama-4, MiMo-V2-Flash, etc.).
/// Mirrors Python's `common.HybridMoEConfig`. Two per-layer 0/1 patterns
/// (attention type and FFN type) drive four layer-kind buckets:
/// global_moe, swa_moe, swa_dense, global_dense.
#[derive(Clone, Debug, PartialEq)]
pub struct HybridMoEConfig {
    /// Per-layer attention pattern: 1 = global (full) attention,
    /// 0 = SWA/local (sliding-window) attention. Length ==
    /// `num_hidden_layers`.
    pub attn_layer_pattern: Vec<u32>,
    /// Per-layer FFN pattern: 1 = MoE FFN, 0 = dense SwiGLU FFN.
    /// Length == `num_hidden_layers`.
    pub moe_layer_freq: Vec<u32>,
    /// Window size for SWA layers (passed to ContextAttention/GenerationAttention).
    pub sliding_window_size: u32,
    /// Per-layer SWA attention dims. Zero falls back to the model-level
    /// `num_kv_heads` / `head_dim` defaults (Llama-4 shares dims with
    /// global attention; MiMo-V2-Flash differs).
    pub swa_num_kv_heads: u32,
    pub swa_head_dim: u32,
    pub swa_v_head_dim: u32,
    /// V-head dim for global attention layers (zero -> model `head_dim`).
    pub global_v_head_dim: u32,
    /// Dense FFN intermediate size (zero -> model `intermediate_size`).
    pub dense_inter_size: u32,
}

/// Per-family extras for DeepSeek-V4 (`DeepseekV4ForCausalLM`). Mirrors
/// Python's `common.DeepSeekV4Config`. Compressed-attention models pair
/// per-layer `compress_ratio in {0, 4, 128}` with a manifold-constrained
/// hyper-connection (mHC) block.
#[derive(Clone, Debug, PartialEq)]
pub struct DeepSeekV4Config {
    pub q_lora_rank: u32,
    pub o_lora_rank: u32,
    pub o_groups: u32,
    pub head_dim: u32,
    pub qk_rope_head_dim: u32,
    pub index_head_dim: u32,
    pub index_n_heads: u32,
    pub index_topk: u32,
    pub sliding_window: u32,
    /// Per-layer compress ratio. Length equals `num_hidden_layers`.
    /// 0 = SWA (approximated as HCA), 4 = CSA, 128 = HCA.
    pub compress_ratios: Vec<u32>,
    pub hc_mult: u32,
    pub hc_sinkhorn_iters: u32,
    pub n_shared_experts: u32,
}

/// NemotronNas per-layer block descriptor mirroring Python's
/// `common.BlockConfig`. `num_inst` is the run-length count when blocks
/// are grouped; the Rust loader emits one entry per layer (count=1).
#[derive(Clone, Debug, PartialEq)]
pub struct BlockConfig {
    pub num_inst: u32,
    pub attn_no_op: bool,
    /// Number of attention heads per KV-group. KV head count =
    /// `num_attention_heads / attn_n_heads_in_group`. Only meaningful
    /// when `!attn_no_op`.
    pub attn_n_heads_in_group: u32,
    pub ffn_no_op: bool,
    /// FFN size multiplier relative to hidden size. The actual
    /// intermediate dim is `int(2 * mult * hidden_size / 3)` rounded up
    /// to a multiple of 256. Only meaningful when `!ffn_no_op`.
    pub ffn_ffn_mult: f64,
}

/// Resolve a model identifier (HF id, local directory, or
/// `<owner>--<name>_config.json` basename) to a concrete `config.json` path
/// under the AIC `model_configs_root`.
pub fn resolve_config_path(model_name: &str, root: &Path) -> Result<PathBuf, AicError> {
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

/// Load and parse the LLM-relevant fields of a HuggingFace `config.json`.
pub fn load(model_name: &str, root: &Path) -> Result<HfModelConfig, AicError> {
    let path = resolve_config_path(model_name, root)?;
    load_path(&path)
}

/// Load and parse a specific `config.json` file path.
///
/// If the file is a `*_hf_quant_config.json` shard that does not carry an
/// `architectures` array, the loader falls back to the sibling
/// `*_config.json` automatically.
pub fn load_path(path: &Path) -> Result<HfModelConfig, AicError> {
    let value = read_json(path)?;
    if value.get("architectures").is_none() {
        if let Some(base_path) = resolve_hf_quant_base_config(path) {
            return load_path(&base_path);
        }
    }
    parse_value(&value, path)
}

/// Parse a `serde_json::Value` already in memory.
///
/// Public so unit tests and future callers can drive parsing without
/// touching the filesystem.
pub fn parse_value(value: &Value, path: &Path) -> Result<HfModelConfig, AicError> {
    let architecture = read_architecture(value, path)?;
    let llm = llm_config_view(value, &architecture);
    let family = architecture_to_family(&architecture).ok_or_else(|| {
        AicError::UnsupportedModel(format!(
            "architecture '{architecture}' is not mapped to an AIC model family"
        ))
    })?;

    let num_hidden_layers = read_num_hidden_layers(llm, path)?;
    let num_attention_heads = required_u32(llm, "num_attention_heads", path)?;
    let hidden_size = required_u32(llm, "hidden_size", path)?;
    let intermediate_size = optional_u32(llm, "intermediate_size", path)?.unwrap_or(0);
    let vocab_size = required_u32(llm, "vocab_size", path)?;
    let context_length = optional_u32(llm, "max_position_embeddings", path)?
        .or(optional_u32(llm, "seq_length", path)?)
        .or(optional_u32(llm, "max_seq_len", path)?)
        .unwrap_or(0);
    let num_key_value_heads =
        optional_u32(llm, "num_key_value_heads", path)?.unwrap_or(num_attention_heads);
    let head_dim = optional_u32(llm, "head_dim", path)?
        .or(optional_u32(llm, "attention_head_dim", path)?)
        .unwrap_or(hidden_size / num_attention_heads.max(1));
    let top_k = optional_u32(llm, "num_experts_per_tok", path)?
        .or(optional_u32(llm, "top_k_experts", path)?)
        .unwrap_or(0);
    let num_experts = optional_u32(llm, "num_local_experts", path)?
        .or(optional_u32(llm, "n_routed_experts", path)?)
        .or(optional_u32(llm, "num_experts", path)?)
        .unwrap_or(0);
    let moe_intermediate_size =
        optional_u32(llm, "moe_intermediate_size", path)?.unwrap_or(intermediate_size);
    let shared_expert_intermediate_size =
        optional_u32(llm, "shared_expert_intermediate_size", path)?
            .or(optional_u32(llm, "moe_shared_expert_intermediate_size", path)?)
            .unwrap_or(0);
    // Mirrors Python `utils.py`: certain Qwen3 / MiniMax architectures have
    // per-layer Q/K normalization baked into the attention path, even when
    // the HF config doesn't carry an explicit `use_qk_norm` field.
    let qk_norm_by_architecture = matches!(
        architecture.as_str(),
        "Qwen3ForCausalLM" | "Qwen3MoeForCausalLM" | "MiniMaxM2ForCausalLM"
    );
    let use_qk_norm = qk_norm_by_architecture
        || llm
            .get("use_qk_norm")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

    // NemotronNas / `DeciLMForCausalLM`: HF config has a `block_configs`
    // array with per-layer `{attention: {n_heads_in_group, no_op},
    // ffn: {ffn_mult, no_op}}`. Mirrors Python's
    // `_parse_nemotron_block_configs` but emits one entry per layer
    // (the run-length grouping is a pure rendering choice; the latency
    // sum is invariant).
    let block_configs = parse_block_configs(llm).unwrap_or_default();
    let qwen35 = parse_qwen35_config(llm, &architecture);
    let nemotron_h = parse_nemotron_h_config(llm, &architecture);
    let deepseek_v4 = parse_deepseek_v4_config(llm, &architecture, num_hidden_layers);
    let hybrid_moe = parse_hybrid_moe_config(llm, &architecture, num_hidden_layers);
    let gemma4_moe = parse_gemma4_moe_config(llm, &architecture, num_hidden_layers);
    // Vision encoder reads from the top-level `vision_config` field — the
    // `llm` view already strips the text-only nesting, but ViT extras live
    // alongside `text_config` so we pull from the unwrapped JSON root.
    let vision_encoder = parse_vision_encoder_config(value, &architecture);

    Ok(HfModelConfig {
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
        shared_expert_intermediate_size,
        use_qk_norm,
        block_configs,
        qwen35,
        nemotron_h,
        deepseek_v4,
        hybrid_moe,
        gemma4_moe,
        vision_encoder,
    })
}

fn parse_vision_encoder_config(value: &Value, architecture: &str) -> Option<VisionEncoderConfig> {
    if !matches!(
        architecture,
        "Qwen3VLForConditionalGeneration" | "Qwen3VLMoeForConditionalGeneration"
    ) {
        return None;
    }
    let vc = value.get("vision_config")?;
    let read_u32 = |key: &str| -> u32 {
        vc.get(key)
            .and_then(|v| v.as_u64())
            .map(|n| n as u32)
            .unwrap_or(0)
    };
    let hidden_size = read_u32("hidden_size");
    let spatial_merge_size = read_u32("spatial_merge_size").max(1);
    let out_hidden_size = read_u32("out_hidden_size");
    let merger_dim = hidden_size * spatial_merge_size * spatial_merge_size;
    let deepstack_visual_indexes: Vec<u32> = vc
        .get("deepstack_visual_indexes")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|x| x.as_u64().map(|n| n as u32))
                .collect()
        })
        .unwrap_or_default();
    // PatchMerger projector: 2 layers (fc1, fc2). Matches Python
    // `utils.py:_parse_extra_params` for Qwen3VL.
    let projector_dims = vec![
        (merger_dim, merger_dim),
        (merger_dim, out_hidden_size),
    ];
    let projector_n_instances = 1 + deepstack_visual_indexes.len() as u32;
    Some(VisionEncoderConfig {
        depth: read_u32("depth"),
        hidden_size,
        num_heads: read_u32("num_heads"),
        intermediate_size: read_u32("intermediate_size"),
        patch_size: read_u32("patch_size"),
        temporal_patch_size: read_u32("temporal_patch_size"),
        spatial_merge_size,
        out_hidden_size,
        deepstack_visual_indexes,
        projector_dims,
        projector_n_instances,
    })
}

fn parse_gemma4_moe_config(
    llm: &Value,
    architecture: &str,
    num_hidden_layers: u32,
) -> Option<Gemma4MoEConfig> {
    if architecture != "Gemma4ForConditionalGeneration" {
        return None;
    }
    let layer_types: Vec<String> = llm
        .get("layer_types")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|x| x.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();
    if layer_types.len() != num_hidden_layers as usize {
        return None;
    }
    let read_u32 = |key: &str| -> u32 {
        llm.get(key)
            .and_then(|v| v.as_u64())
            .map(|n| n as u32)
            .unwrap_or(0)
    };
    Some(Gemma4MoEConfig {
        layer_types,
        swa_num_kv_heads: read_u32("num_key_value_heads"),
        swa_head_dim: read_u32("head_dim"),
        global_num_kv_heads: read_u32("num_global_key_value_heads"),
        global_head_dim: read_u32("global_head_dim"),
        sliding_window_size: read_u32("sliding_window"),
        attention_k_eq_v: llm
            .get("attention_k_eq_v")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
    })
}

fn parse_hybrid_moe_config(
    llm: &Value,
    architecture: &str,
    num_hidden_layers: u32,
) -> Option<HybridMoEConfig> {
    let layers = num_hidden_layers as usize;
    match architecture {
        "Llama4ForConditionalGeneration" => {
            // Llama-4: step-driven patterns. Attention alternates
            // local/global (even -> 0=local, odd -> 1=global); FFN is
            // MoE on layers where (i+1) % step == 0.
            let step = llm
                .get("interleave_moe_layer_step")
                .and_then(|v| v.as_u64())
                .map(|n| n as u32)
                .unwrap_or(1)
                .max(1);
            let attn_layer_pattern: Vec<u32> =
                (0..num_hidden_layers).map(|i| i % 2).collect();
            let moe_layer_freq: Vec<u32> = (0..num_hidden_layers)
                .map(|i| if (i + 1) % step == 0 { 1 } else { 0 })
                .collect();
            Some(HybridMoEConfig {
                attn_layer_pattern,
                moe_layer_freq,
                sliding_window_size: llm
                    .get("attention_chunk_size")
                    .and_then(|v| v.as_u64())
                    .map(|n| n as u32)
                    .unwrap_or(0),
                swa_num_kv_heads: 0,
                swa_head_dim: 0,
                swa_v_head_dim: 0,
                global_v_head_dim: 0,
                dense_inter_size: llm
                    .get("intermediate_size_mlp")
                    .and_then(|v| v.as_u64())
                    .map(|n| n as u32)
                    .unwrap_or(0),
            })
        }
        "MiMoV2FlashForCausalLM" => {
            // MiMo-V2-Flash: explicit per-layer arrays in HF config.
            let attn_layer_pattern: Vec<u32> = llm
                .get("hybrid_layer_pattern")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|x| x.as_u64().map(|n| n as u32))
                        .collect()
                })
                .unwrap_or_default();
            let moe_layer_freq_raw = llm.get("moe_layer_freq");
            let moe_layer_freq: Vec<u32> = match moe_layer_freq_raw {
                Some(v) if v.is_array() => v
                    .as_array()
                    .unwrap()
                    .iter()
                    .filter_map(|x| x.as_u64().map(|n| n as u32))
                    .collect(),
                Some(v) if v.is_u64() => {
                    let val = v.as_u64().unwrap() as u32;
                    vec![val; layers]
                }
                _ => Vec::new(),
            };
            if attn_layer_pattern.len() != layers || moe_layer_freq.len() != layers {
                return None;
            }
            Some(HybridMoEConfig {
                attn_layer_pattern,
                moe_layer_freq,
                sliding_window_size: llm
                    .get("sliding_window_size")
                    .and_then(|v| v.as_u64())
                    .map(|n| n as u32)
                    .unwrap_or(0),
                swa_num_kv_heads: llm
                    .get("swa_num_key_value_heads")
                    .and_then(|v| v.as_u64())
                    .map(|n| n as u32)
                    .unwrap_or(0),
                swa_head_dim: llm
                    .get("swa_head_dim")
                    .and_then(|v| v.as_u64())
                    .map(|n| n as u32)
                    .unwrap_or(0),
                swa_v_head_dim: llm
                    .get("swa_v_head_dim")
                    .and_then(|v| v.as_u64())
                    .map(|n| n as u32)
                    .unwrap_or(0),
                global_v_head_dim: llm
                    .get("v_head_dim")
                    .and_then(|v| v.as_u64())
                    .map(|n| n as u32)
                    .unwrap_or(0),
                dense_inter_size: 0,
            })
        }
        _ => None,
    }
}

fn parse_deepseek_v4_config(
    llm: &Value,
    architecture: &str,
    num_hidden_layers: u32,
) -> Option<DeepSeekV4Config> {
    if architecture != "DeepseekV4ForCausalLM" {
        return None;
    }
    let read_u32 = |key: &str| -> u32 {
        llm.get(key)
            .and_then(|v| v.as_u64())
            .map(|n| n as u32)
            .unwrap_or(0)
    };
    let compress_ratios: Vec<u32> = llm
        .get("compress_ratios")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|x| x.as_u64().map(|n| n as u32))
                .take(num_hidden_layers as usize)
                .collect()
        })
        .unwrap_or_default();
    Some(DeepSeekV4Config {
        q_lora_rank: read_u32("q_lora_rank"),
        o_lora_rank: read_u32("o_lora_rank"),
        o_groups: read_u32("o_groups"),
        head_dim: read_u32("head_dim"),
        qk_rope_head_dim: read_u32("qk_rope_head_dim"),
        index_head_dim: read_u32("index_head_dim"),
        index_n_heads: read_u32("index_n_heads"),
        index_topk: read_u32("index_topk"),
        sliding_window: read_u32("sliding_window"),
        compress_ratios,
        hc_mult: read_u32("hc_mult"),
        hc_sinkhorn_iters: read_u32("hc_sinkhorn_iters"),
        n_shared_experts: llm
            .get("n_shared_experts")
            .and_then(|v| v.as_u64())
            .map(|n| n as u32)
            .unwrap_or(1),
    })
}

fn parse_nemotron_h_config(llm: &Value, architecture: &str) -> Option<NemotronHConfig> {
    if architecture != "NemotronHForCausalLM" {
        return None;
    }
    let read_u32 = |key: &str| -> u32 {
        llm.get(key)
            .and_then(|v| v.as_u64())
            .map(|n| n as u32)
            .unwrap_or(0)
    };
    let read_u32_opt = |key: &str| -> u32 {
        llm.get(key)
            .and_then(|v| v.as_u64())
            .map(|n| n as u32)
            .unwrap_or(0)
    };
    let hybrid_override_pattern = llm
        .get("hybrid_override_pattern")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    Some(NemotronHConfig {
        hybrid_override_pattern,
        mamba_num_heads: read_u32("mamba_num_heads"),
        mamba_head_dim: read_u32("mamba_head_dim"),
        ssm_state_size: read_u32("ssm_state_size"),
        conv_kernel: read_u32("conv_kernel"),
        n_groups: read_u32("n_groups"),
        chunk_size: read_u32("chunk_size"),
        moe_shared_expert_intermediate_size: read_u32_opt("moe_shared_expert_intermediate_size"),
        moe_latent_size: read_u32_opt("moe_latent_size"),
    })
}

fn parse_qwen35_config(llm: &Value, architecture: &str) -> Option<Qwen35Config> {
    if !matches!(
        architecture,
        "Qwen3_5ForConditionalGeneration" | "Qwen3_5MoeForConditionalGeneration"
    ) {
        return None;
    }
    let layer_types: Vec<String> = llm
        .get("layer_types")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    let read_u32 = |key: &str| -> u32 {
        llm.get(key)
            .and_then(|v| v.as_u64())
            .map(|n| n as u32)
            .unwrap_or(0)
    };
    Some(Qwen35Config {
        layer_types,
        linear_num_key_heads: read_u32("linear_num_key_heads"),
        linear_key_head_dim: read_u32("linear_key_head_dim"),
        linear_num_value_heads: read_u32("linear_num_value_heads"),
        linear_value_head_dim: read_u32("linear_value_head_dim"),
        linear_conv_kernel_dim: read_u32("linear_conv_kernel_dim"),
    })
}

fn parse_block_configs(llm: &Value) -> Option<Vec<BlockConfig>> {
    let raw = llm.get("block_configs")?.as_array()?;
    let mut out = Vec::with_capacity(raw.len());
    for block in raw {
        let obj = block.as_object()?;
        let attn = obj
            .get("attention")
            .and_then(|v| v.as_object())
            .cloned()
            .unwrap_or_default();
        let ffn = obj
            .get("ffn")
            .and_then(|v| v.as_object())
            .cloned()
            .unwrap_or_default();
        out.push(BlockConfig {
            num_inst: 1,
            attn_no_op: attn.get("no_op").and_then(|v| v.as_bool()).unwrap_or(false),
            attn_n_heads_in_group: attn
                .get("n_heads_in_group")
                .and_then(|v| v.as_u64())
                .map(|n| n as u32)
                .unwrap_or(1),
            ffn_no_op: ffn.get("no_op").and_then(|v| v.as_bool()).unwrap_or(false),
            ffn_ffn_mult: ffn
                .get("ffn_mult")
                .and_then(|v| v.as_f64())
                .unwrap_or(3.5),
        });
    }
    Some(out)
}

fn read_json(path: &Path) -> Result<Value, AicError> {
    let text = fs::read_to_string(path).map_err(|source| AicError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    // Some HF configs include non-standard `Infinity`/`-Infinity`/`NaN`
    // literals that serde_json rejects. Match Python's tolerant behavior.
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

fn resolve_hf_quant_base_config(path: &Path) -> Option<PathBuf> {
    let file_name = path.file_name()?.to_str()?;
    let base_name = file_name.replace("_hf_quant_config.json", "_config.json");
    if base_name == file_name {
        return None;
    }
    let base_path = path.with_file_name(base_name);
    base_path.is_file().then_some(base_path)
}

fn read_architecture(value: &Value, path: &Path) -> Result<String, AicError> {
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

/// Return the view of the config that carries the LLM fields, flattening
/// `text_config` for multimodal architectures.
fn llm_config_view<'a>(value: &'a Value, architecture: &str) -> &'a Value {
    multimodal_text_config_key(architecture)
        .and_then(|key| value.get(key))
        .filter(|nested| nested.is_object())
        .unwrap_or(value)
}

/// Read `num_hidden_layers`, allowing the Nemotron-H-style
/// `hybrid_override_pattern` fallback used in Python.
fn read_num_hidden_layers(value: &Value, path: &Path) -> Result<u32, AicError> {
    if let Some(n) = optional_u32(value, "num_hidden_layers", path)? {
        return Ok(n);
    }
    // Fallback 1: NemotronNas / Puzzle (`DeciLMForCausalLM`) — the layer
    // count is implicit in the `layers_block_type` array length.
    if let Some(blocks) = value.get("layers_block_type").and_then(Value::as_array) {
        return u32::try_from(blocks.len()).map_err(|_| {
            AicError::ModelConfig(format!(
                "layers_block_type in model config {} is too large for u32",
                path.display()
            ))
        });
    }
    // Fallback 2: NemotronH — implicit in the `hybrid_override_pattern`
    // character count.
    if let Some(pattern) = value.get("hybrid_override_pattern").and_then(Value::as_str) {
        if !pattern.is_empty() {
            return Ok(pattern.chars().count() as u32);
        }
    }
    Err(AicError::ModelConfig(format!(
        "missing required field 'num_hidden_layers' in model config {}",
        path.display()
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
    if raw.is_null() {
        return Ok(None);
    }
    if let Some(n) = raw.as_u64() {
        return Ok(Some(n as u32));
    }
    if let Some(n) = raw.as_i64() {
        if n >= 0 {
            return Ok(Some(n as u32));
        }
    }
    if let Some(n) = raw.as_f64() {
        if n.is_finite() && n >= 0.0 {
            return Ok(Some(n as u32));
        }
    }
    Err(AicError::ModelConfig(format!(
        "field '{key}' in {} is not a non-negative integer",
        path.display()
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    const REPO_ROOT_HINT: &str = env!("CARGO_MANIFEST_DIR");

    fn model_configs_root() -> PathBuf {
        PathBuf::from(REPO_ROOT_HINT)
            .join("../..")
            .join("src/aiconfigurator/model_configs")
    }

    #[test]
    fn load_minimax_m25_smoke_model() {
        let cfg = load("MiniMaxAI/MiniMax-M2.5", &model_configs_root())
            .expect("MiniMax-M2.5 must resolve and parse");
        assert_eq!(cfg.architecture, "MiniMaxM2ForCausalLM");
        assert_eq!(cfg.family, ModelFamily::Moe);
        assert!(cfg.num_experts > 0);
        assert!(cfg.top_k > 0);
    }

    #[test]
    fn load_kimi_k25_smoke_model_flattens_text_config() {
        let cfg = load("moonshotai/Kimi-K2.5", &model_configs_root())
            .expect("Kimi-K2.5 must resolve and parse");
        assert_eq!(cfg.architecture, "KimiK25ForConditionalGeneration");
        assert_eq!(cfg.family, ModelFamily::KimiK25);
        // The MLA-family models still need num_hidden_layers / hidden_size
        // visible after text_config flattening.
        assert!(cfg.num_hidden_layers > 0);
        assert!(cfg.hidden_size > 0);
    }

    #[test]
    fn load_qwen3vl_dense_through_text_config() {
        // Exercises the multimodal flattening path; the parser must
        // succeed even when downstream runtime support for this model is
        // not exercised in this test.
        let cfg = load("Qwen/Qwen3-VL-8B-Instruct", &model_configs_root())
            .expect("Qwen3-VL-8B-Instruct must resolve and parse");
        assert_eq!(cfg.architecture, "Qwen3VLForConditionalGeneration");
        assert_eq!(cfg.family, ModelFamily::Qwen3Vl);
        assert_eq!(cfg.num_hidden_layers, 36);
        assert_eq!(cfg.num_attention_heads, 32);
        assert_eq!(cfg.num_key_value_heads, 8);
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.head_dim, 128);
    }

    #[test]
    fn unknown_architecture_is_unsupported_model() {
        let json = serde_json::json!({
            "architectures": ["NotARealArchitecture"],
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "hidden_size": 1,
            "vocab_size": 1,
        });
        let err = parse_value(&json, Path::new("synthetic.json")).unwrap_err();
        assert!(matches!(err, AicError::UnsupportedModel(_)), "got {err}");
    }

    #[test]
    fn json_loader_tolerates_infinity_literals() {
        // `serde_json` rejects unquoted Infinity by default; the loader
        // should accept it the same way Python's json does.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad_json.json");
        std::fs::write(
            &path,
            r#"{"architectures":["LlamaForCausalLM"],"num_hidden_layers":1,"num_attention_heads":1,"hidden_size":1,"vocab_size":1,"rope_theta":Infinity}"#,
        )
        .unwrap();
        let value = read_json(&path).unwrap();
        assert!(value.get("rope_theta").unwrap().is_null());
    }

    #[test]
    fn hybrid_override_pattern_fallback_for_layer_count() {
        let json = serde_json::json!({
            "architectures": ["NemotronHForCausalLM"],
            "hybrid_override_pattern": "MM*-MM*",
            "num_attention_heads": 32,
            "hidden_size": 4096,
            "vocab_size": 32000,
        });
        let cfg = parse_value(&json, Path::new("synthetic.json"))
            .expect("hybrid_override_pattern must yield a layer count");
        assert_eq!(cfg.num_hidden_layers, 7); // "MM*-MM*" is 7 chars
    }
}
