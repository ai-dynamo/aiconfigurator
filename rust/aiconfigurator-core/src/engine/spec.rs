// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `EngineSpec`: the serializable engine wire format.
//!
//! `EngineSpec` is what Python's `compile_engine` emits and what the
//! Rust `Engine` consumes. It bundles the engine identity
//! ([`EngineConfig`], needed later to load the matching [`PerfDatabase`]) with
//! the precompiled context / generation op lists. Each op is an
//! [`OpSpec`] — a public alias for the crate's [`Op`] enum — and the lists
//! round-trip through bincode, including the recursive `Overlap` / `Fallback`
//! children.
//!
//! ## Vision is never on the wire
//!
//! [`Op::Vision`] derives serde with every other variant (it remains part of
//! the shared session path), but a compiled `EngineSpec` never
//! contains a `Vision` op: `compile_engine` decomposes the vision encoder
//! into its child `Gemm` / `EncoderAttention` / `Elementwise` ops, each an
//! existing variant. The type round-trips soundly (see the test below); the
//! constraint is purely a producer-side rule, not enforced by the enum.
//!
//! [`PerfDatabase`]: crate::perf_database::PerfDatabase

use serde::{Deserialize, Serialize};

use crate::common::error::AicError;
use crate::{EngineConfig, ENGINE_SPEC_SCHEMA_VERSION};

/// Public name for the serializable op. Aliases the crate's [`Op`] enum so
/// the "OpSpec" surface exists without duplicating the definition.
pub use crate::operators::op::Op as OpSpec;

/// Serializable compiled engine.
///
/// `schema_version` guards forward/backward compatibility; `engine` carries
/// the identity used to load the matching perf database; `context_ops` and
/// `generation_ops` are the precompiled op lists the runner iterates.
///
/// `EngineSpec` derives serde so JSON / other self-describing formats work
/// directly. The **bincode** wire format, however, must go through
/// [`EngineSpec::to_bincode`] / [`EngineSpec::from_bincode`], NOT
/// `bincode::serialize(&spec)` directly: [`EngineConfig`] uses
/// `#[serde(flatten)]` (load-bearing for the flat ctypes FFI contract), and
/// bincode 1.x cannot serialize a flattened struct (it emits a map of unknown
/// length → `SequenceMustHaveLength`). The helpers sidestep this by
/// JSON-encoding the `engine` field inside the bincode payload, keeping
/// `EngineConfig` the single source of truth (no mirror struct, no drift).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EngineSpec {
    pub schema_version: u32,
    /// Engine identity (model / system / backend / parallelism / quant).
    /// Needed to locate and load the `PerfDatabase`.
    pub engine: EngineConfig,
    /// Context-phase ops, in execution order. Never contains `OpSpec::Vision`
    /// (decomposed into child ops at compile time).
    pub context_ops: Vec<OpSpec>,
    /// Generation-phase ops, in execution order.
    pub generation_ops: Vec<OpSpec>,
}

/// Private bincode payload. `engine` is carried as a JSON string so the
/// `#[serde(flatten)]` on [`EngineConfig`] never reaches the bincode
/// serializer (which rejects unknown-length maps). The op lists are plain
/// `Vec<OpSpec>` (no flatten) and bincode-serialize directly.
#[derive(Serialize, Deserialize)]
struct BincodeWire {
    schema_version: u32,
    engine_json: String,
    context_ops: Vec<OpSpec>,
    generation_ops: Vec<OpSpec>,
}

impl EngineSpec {
    /// Build a spec, stamping the current [`ENGINE_SPEC_SCHEMA_VERSION`].
    pub fn new(engine: EngineConfig, context_ops: Vec<OpSpec>, generation_ops: Vec<OpSpec>) -> Self {
        Self {
            schema_version: ENGINE_SPEC_SCHEMA_VERSION,
            engine,
            context_ops,
            generation_ops,
        }
    }

    /// Serialize to the bincode wire format. The `engine` field is
    /// JSON-encoded inside the payload (see the struct docs) so bincode never
    /// sees `EngineConfig`'s flattened layout.
    pub fn to_bincode(&self) -> Result<Vec<u8>, AicError> {
        let engine_json = serde_json::to_string(&self.engine)
            .map_err(|e| AicError::EngineSpec(format!("engine JSON encode: {e}")))?;
        let wire = BincodeWire {
            schema_version: self.schema_version,
            engine_json,
            context_ops: self.context_ops.clone(),
            generation_ops: self.generation_ops.clone(),
        };
        bincode::serialize(&wire).map_err(|e| AicError::EngineSpec(format!("bincode encode: {e}")))
    }

    /// Deserialize from the bincode wire format produced by [`Self::to_bincode`].
    pub fn from_bincode(bytes: &[u8]) -> Result<Self, AicError> {
        let wire: BincodeWire = bincode::deserialize(bytes)
            .map_err(|e| AicError::EngineSpec(format!("bincode decode: {e}")))?;
        if wire.schema_version != ENGINE_SPEC_SCHEMA_VERSION {
            return Err(AicError::UnsupportedSchemaVersion {
                kind: "EngineSpec",
                got: wire.schema_version,
                expected: ENGINE_SPEC_SCHEMA_VERSION,
            });
        }
        let engine: EngineConfig = serde_json::from_str(&wire.engine_json)
            .map_err(|e| AicError::EngineSpec(format!("engine JSON decode: {e}")))?;
        Ok(Self {
            schema_version: wire.schema_version,
            engine,
            context_ops: wire.context_ops,
            generation_ops: wire.generation_ops,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    use crate::common::enums::{
        BackendKind, CommQuantMode, FmhaQuantMode, GemmQuantMode, KvCacheQuantMode, MoeQuantMode,
    };
    use crate::operators::op::{FallbackOp, OverlapOp};
    use crate::operators::{
        ContextAttentionOp, ContextMlaOp, CustomAllReduceOp, DsaModuleOp, Dsv4ModuleOp,
        ElementwiseOp, EmbeddingOp, EncoderAttentionOp, GdnOp, GemmOp, GenerationAttentionOp,
        GenerationMlaOp, Mamba2Op, MhcModuleOp, MlaBmmOp, MlaModuleOp, MoEDispatchOp, MoeOp,
        NcclOp, P2POp, VisionEncoderOp, WideEpContextMlaOp, WideEpGenerationMlaOp, WideEpMoeOp,
    };
    use crate::operators::moe_dispatch::DispatchFlavor;
    use crate::perf_database::dsv4::AttnKind;
    use crate::{
        DataType, ParallelMapping, QuantizationConfig, SpeculativeConfig,
        ENGINE_CONFIG_SCHEMA_VERSION,
    };

    // ---- Representative-value builders for each Op variant ----

    fn gemm() -> GemmOp {
        GemmOp {
            name: "qkv_gemm".into(),
            scale_factor: 2.0,
            n: 4096,
            k: 4096,
            quant_mode: GemmQuantMode::Fp8,
            scale_num_tokens: 0,
            low_precision_input: true,
        }
    }

    fn embedding() -> EmbeddingOp {
        EmbeddingOp {
            name: "embedding".into(),
            scale_factor: 1.0,
            vocab_size: 128_256,
            hidden_size: 4096,
            quant_mode: GemmQuantMode::Bfloat16,
        }
    }

    fn elementwise() -> ElementwiseOp {
        ElementwiseOp {
            name: "rmsnorm".into(),
            scale_factor: 1.5,
            bytes_per_token: 8192.0,
        }
    }

    fn context_attention() -> ContextAttentionOp {
        ContextAttentionOp {
            name: "context_attention".into(),
            scale_factor: 1.0,
            n: 32,
            n_kv: 8,
            head_size: 128,
            window_size: 0,
            kv_cache_dtype: KvCacheQuantMode::Fp8,
            fmha_quant_mode: FmhaQuantMode::Bfloat16,
            use_qk_norm: true,
        }
    }

    fn generation_attention() -> GenerationAttentionOp {
        GenerationAttentionOp {
            name: "generation_attention".into(),
            scale_factor: 1.0,
            n: 32,
            n_kv: 8,
            head_size: 128,
            window_size: 4096,
            kv_cache_dtype: KvCacheQuantMode::Int8,
        }
    }

    fn encoder_attention() -> EncoderAttentionOp {
        EncoderAttentionOp {
            name: "encoder_attention".into(),
            scale_factor: 1.0,
            n: 16,
            head_size: 80,
            fmha_quant_mode: FmhaQuantMode::Fp8,
        }
    }

    fn context_mla() -> ContextMlaOp {
        ContextMlaOp {
            name: "context_mla".into(),
            scale_factor: 1.0,
            num_heads: 128,
            kv_cache_dtype: KvCacheQuantMode::Bfloat16,
            fmha_quant_mode: FmhaQuantMode::Bfloat16,
        }
    }

    fn generation_mla() -> GenerationMlaOp {
        GenerationMlaOp {
            name: "generation_mla".into(),
            scale_factor: 1.0,
            num_heads: 128,
            kv_cache_dtype: KvCacheQuantMode::Fp8,
        }
    }

    fn mla_module() -> MlaModuleOp {
        MlaModuleOp {
            name: "context_mla_module".into(),
            scale_factor: 1.0,
            num_heads: 128,
            kv_cache_dtype: KvCacheQuantMode::Fp8,
            fmha_quant_mode: FmhaQuantMode::Fp8,
            gemm_quant_mode: GemmQuantMode::Fp8Block,
        }
    }

    fn mla_bmm() -> MlaBmmOp {
        MlaBmmOp {
            name: "mla_bmm_pre".into(),
            scale_factor: 1.0,
            num_heads: 128,
            quant_mode: GemmQuantMode::Bfloat16,
            is_pre: true,
        }
    }

    fn moe() -> MoeOp {
        MoeOp {
            name: "moe".into(),
            scale_factor: 1.0,
            hidden_size: 7168,
            inter_size: 2048,
            topk: 8,
            num_experts: 256,
            moe_tp_size: 1,
            moe_ep_size: 8,
            quant_mode: MoeQuantMode::Fp8Block,
            workload_distribution: "power_law_1.2".into(),
            is_gated: true,
        }
    }

    fn moe_dispatch() -> MoEDispatchOp {
        MoEDispatchOp {
            name: "moe_dispatch".into(),
            scale_factor: 1.0,
            hidden_size: 7168,
            topk: 8,
            num_experts: 256,
            moe_tp_size: 1,
            moe_ep_size: 8,
            attention_dp_size: 8,
            pre_dispatch: true,
            backend: BackendKind::Trtllm,
            flavor: DispatchFlavor::TrtllmAlltoall,
            comm_quant: CommQuantMode::Half,
            moe_quant: MoeQuantMode::Fp8Block,
        }
    }

    fn custom_all_reduce() -> CustomAllReduceOp {
        CustomAllReduceOp {
            name: "custom_all_reduce".into(),
            scale_factor: 1.0,
            hidden_size: 4096,
            tp_size: 8,
            quant: CommQuantMode::Half,
        }
    }

    fn nccl() -> NcclOp {
        NcclOp {
            name: "nccl_all_reduce".into(),
            scale_factor: 1.0,
            hidden_size: 4096,
            num_gpus: 8,
            dtype: CommQuantMode::Half,
            operation: "all_reduce".into(),
        }
    }

    fn p2p() -> P2POp {
        P2POp {
            name: "p2p".into(),
            scale_factor: 1.0,
            pp_size: 4,
            hidden_size: 4096,
        }
    }

    fn vision() -> VisionEncoderOp {
        VisionEncoderOp {
            name: "vision_encoder".into(),
            scale_factor: 1.0,
            num_layers: 24,
            num_heads: 16,
            head_size: 80,
            hidden_size: 1280,
            intermediate_size: 5120,
            fmha_quant: FmhaQuantMode::Bfloat16,
            gemm_quant: GemmQuantMode::Bfloat16,
        }
    }

    fn dsa_module() -> DsaModuleOp {
        DsaModuleOp {
            name: "dsa_module".into(),
            scale_factor: 1.0,
            num_heads: 128,
            kv_cache_dtype: KvCacheQuantMode::Fp8,
            fmha_quant_mode: FmhaQuantMode::Fp8,
            gemm_quant_mode: GemmQuantMode::Fp8Block,
            architecture: "DeepseekV32ForCausalLM".into(),
        }
    }

    fn dsv4_module() -> Dsv4ModuleOp {
        Dsv4ModuleOp {
            name: "dsv4_module".into(),
            scale_factor: 1.0,
            attn_kind: AttnKind::Hca,
            num_heads: 128,
            kv_cache_dtype: KvCacheQuantMode::Fp8,
            fmha_quant_mode: FmhaQuantMode::Fp8,
            gemm_quant_mode: GemmQuantMode::Fp8Block,
            architecture: "DeepseekV4ForCausalLM".into(),
        }
    }

    fn mhc() -> MhcModuleOp {
        MhcModuleOp {
            name: "mhc_module".into(),
            scale_factor: 1.0,
            hc_mult: 4,
            hidden_size: 7168,
            architecture: "DeepseekV4ForCausalLM".into(),
        }
    }

    fn mamba2() -> Mamba2Op {
        Mamba2Op {
            name: "mamba2".into(),
            scale_factor: 1.0,
            kernel_source: "mamba_chunk_scan".into(),
            phase: "context".into(),
            d_model: 4096,
            d_state: 128,
            d_conv: 4,
            nheads: 128,
            head_dim: 64,
            n_groups: 8,
            chunk_size: 256,
        }
    }

    fn gdn() -> GdnOp {
        GdnOp {
            name: "gdn".into(),
            scale_factor: 1.0,
            kernel_source: "gdn_kernel".into(),
            phase: "generation".into(),
            d_model: 4096,
            d_conv: 4,
            num_k_heads: 16,
            head_k_dim: 128,
            num_v_heads: 32,
            head_v_dim: 128,
        }
    }

    fn wideep_context_mla() -> WideEpContextMlaOp {
        WideEpContextMlaOp {
            name: "wideep_context_mla".into(),
            scale_factor: 1.0,
            num_heads: 128,
            kv_cache_dtype: KvCacheQuantMode::Fp8,
            fmha_quant_mode: FmhaQuantMode::Fp8,
            attn_backend: "flashinfer".into(),
        }
    }

    fn wideep_generation_mla() -> WideEpGenerationMlaOp {
        WideEpGenerationMlaOp {
            name: "wideep_generation_mla".into(),
            scale_factor: 1.0,
            num_heads: 128,
            kv_cache_dtype: KvCacheQuantMode::Fp8,
            fmha_quant_mode: FmhaQuantMode::Fp8,
            attn_backend: "flashinfer".into(),
        }
    }

    fn wideep_moe() -> WideEpMoeOp {
        WideEpMoeOp {
            name: "wideep_moe".into(),
            scale_factor: 1.0,
            hidden_size: 7168,
            inter_size: 2048,
            topk: 8,
            num_experts: 256,
            moe_tp_size: 1,
            moe_ep_size: 8,
            attention_dp_size: 8,
            quant_mode: MoeQuantMode::Fp8Block,
            workload_distribution: "power_law_1.2_eplb".into(),
            num_slots: 288,
            kernel_source: "moe_torch_flow".into(),
        }
    }

    fn overlap() -> OverlapOp {
        // Recursive: nested children on both groups.
        OverlapOp {
            name: "overlap_attn_moe".into(),
            group_a: vec![OpSpec::ContextMla(context_mla()), OpSpec::Gemm(gemm())],
            group_b: vec![OpSpec::Moe(moe()), OpSpec::MoeDispatch(moe_dispatch())],
        }
    }

    fn fallback() -> FallbackOp {
        // Recursive: a primary module op with a granular per-kernel fallback
        // chain that itself contains a nested Overlap.
        FallbackOp {
            name: "mla_fallback".into(),
            primary: Box::new(OpSpec::MlaModuleContext(mla_module())),
            fallback: vec![
                OpSpec::Gemm(gemm()),
                OpSpec::ContextMla(context_mla()),
                OpSpec::Overlap(overlap()),
            ],
        }
    }

    /// Every `Op` variant, constructed once. The exhaustive `match` below the
    /// `Vec` build forces the compiler to flag any newly added variant that
    /// this round-trip suite forgot to cover.
    fn all_op_variants() -> Vec<OpSpec> {
        let ops = vec![
            OpSpec::Gemm(gemm()),
            OpSpec::Embedding(embedding()),
            OpSpec::Elementwise(elementwise()),
            OpSpec::ContextAttention(context_attention()),
            OpSpec::GenerationAttention(generation_attention()),
            OpSpec::EncoderAttention(encoder_attention()),
            OpSpec::ContextMla(context_mla()),
            OpSpec::GenerationMla(generation_mla()),
            OpSpec::MlaModuleContext(mla_module()),
            OpSpec::MlaModuleGeneration(mla_module()),
            OpSpec::MlaBmm(mla_bmm()),
            OpSpec::Moe(moe()),
            OpSpec::MoeDispatch(moe_dispatch()),
            OpSpec::CustomAllReduce(custom_all_reduce()),
            OpSpec::Nccl(nccl()),
            OpSpec::P2P(p2p()),
            OpSpec::Vision(vision()),
            OpSpec::DsaContext(dsa_module()),
            OpSpec::DsaGeneration(dsa_module()),
            OpSpec::Dsv4Context(dsv4_module()),
            OpSpec::Dsv4Generation(dsv4_module()),
            OpSpec::Mhc(mhc()),
            OpSpec::Mamba2(mamba2()),
            OpSpec::Gdn(gdn()),
            OpSpec::WideEpContextMla(wideep_context_mla()),
            OpSpec::WideEpGenerationMla(wideep_generation_mla()),
            OpSpec::WideEpMoe(wideep_moe()),
            OpSpec::Overlap(overlap()),
            OpSpec::Fallback(fallback()),
        ];

        // Exhaustiveness guard: if a variant is added to `Op`, this match
        // fails to compile until it is also added to `all_op_variants`.
        for op in &ops {
            match op {
                OpSpec::Gemm(_)
                | OpSpec::Embedding(_)
                | OpSpec::Elementwise(_)
                | OpSpec::ContextAttention(_)
                | OpSpec::GenerationAttention(_)
                | OpSpec::EncoderAttention(_)
                | OpSpec::ContextMla(_)
                | OpSpec::GenerationMla(_)
                | OpSpec::MlaModuleContext(_)
                | OpSpec::MlaModuleGeneration(_)
                | OpSpec::MlaBmm(_)
                | OpSpec::Moe(_)
                | OpSpec::MoeDispatch(_)
                | OpSpec::CustomAllReduce(_)
                | OpSpec::Nccl(_)
                | OpSpec::P2P(_)
                | OpSpec::Vision(_)
                | OpSpec::DsaContext(_)
                | OpSpec::DsaGeneration(_)
                | OpSpec::Dsv4Context(_)
                | OpSpec::Dsv4Generation(_)
                | OpSpec::Mhc(_)
                | OpSpec::Mamba2(_)
                | OpSpec::Gdn(_)
                | OpSpec::WideEpContextMla(_)
                | OpSpec::WideEpGenerationMla(_)
                | OpSpec::WideEpMoe(_)
                | OpSpec::Overlap(_)
                | OpSpec::Fallback(_) => {}
            }
        }
        ops
    }

    fn sample_engine_config() -> EngineConfig {
        EngineConfig {
            schema_version: ENGINE_CONFIG_SCHEMA_VERSION,
            model_name: "deepseek-ai/DeepSeek-V3".into(),
            system_name: "h200_sxm".into(),
            systems_path: None,
            backend: crate::BackendKind::Trtllm,
            backend_version: Some("1.0.0rc3".into()),
            kv_block_size: Some(64),
            parallel: ParallelMapping {
                tp_size: 8,
                pp_size: 1,
                attention_dp_size: Some(8),
                moe_tp_size: Some(1),
                moe_ep_size: Some(8),
            },
            quantization: QuantizationConfig {
                weight_dtype: Some(DataType::Fp8),
                moe_dtype: Some(DataType::Fp8),
                activation_dtype: Some(DataType::Fp8),
                kv_cache_dtype: Some(DataType::Fp8),
            },
            speculative: Some(SpeculativeConfig {
                nextn: Some(1),
                nextn_accept_rates: Some(vec![0.85, 0.3, 0.0, 0.0, 0.0]),
            }),
            extra: BTreeMap::new(),
        }
    }

    #[test]
    fn every_op_variant_round_trips_through_bincode() {
        for op in all_op_variants() {
            let bytes = bincode::serialize(&op).expect("serialize op");
            let decoded: OpSpec = bincode::deserialize(&bytes).expect("deserialize op");
            assert_eq!(op, decoded, "round-trip mismatch for {:?}", op);
        }
    }

    #[test]
    fn recursive_overlap_round_trips_with_nested_children() {
        let op = OpSpec::Overlap(overlap());
        let bytes = bincode::serialize(&op).unwrap();
        let decoded: OpSpec = bincode::deserialize(&bytes).unwrap();
        assert_eq!(op, decoded);
    }

    #[test]
    fn recursive_fallback_round_trips_with_nested_children() {
        let op = OpSpec::Fallback(fallback());
        let bytes = bincode::serialize(&op).unwrap();
        let decoded: OpSpec = bincode::deserialize(&bytes).unwrap();
        assert_eq!(op, decoded);
    }

    #[test]
    fn engine_spec_round_trips_through_bincode() {
        let spec = EngineSpec::new(
            sample_engine_config(),
            vec![
                OpSpec::Embedding(embedding()),
                OpSpec::ContextAttention(context_attention()),
                OpSpec::Overlap(overlap()),
                OpSpec::Gemm(gemm()),
            ],
            vec![
                OpSpec::GenerationAttention(generation_attention()),
                OpSpec::Fallback(fallback()),
                OpSpec::Moe(moe()),
            ],
        );

        assert_eq!(spec.schema_version, ENGINE_SPEC_SCHEMA_VERSION);

        let bytes = spec.to_bincode().expect("to_bincode");
        let decoded = EngineSpec::from_bincode(&bytes).expect("from_bincode");
        assert_eq!(spec, decoded);
    }
}
