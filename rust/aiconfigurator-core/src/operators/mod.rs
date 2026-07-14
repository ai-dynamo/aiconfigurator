// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Operator primitives: GEMM, attention, MLA, MoE, communication, elementwise,
//! embedding, overlap. Submodules each define a focused op type that holds
//! its config-time parameters and exposes a `query` method against
//! `PerfDatabase`.
//!
//! Each operator wraps the raw `perf_database/<family>.rs` table with
//! op-specific extras: SOL/EMPIRICAL/HYBRID database-mode dispatch, prefix
//! correction, fused-op accounting (rope/kv-write/qk-norm for attention),
//! and bandwidth scaling for collectives. The perf-DB layer stays
//! algorithm-free; this layer is where Python's `_query_*_table` static
//! methods live.

pub mod attention;
pub mod base;
pub mod communication;
pub mod dsa;
pub mod dsv4;
pub mod elementwise;
pub mod embedding;
pub mod gemm;
pub mod mamba;
pub mod mhc;
pub mod mla;
pub mod moe;
pub mod moe_dispatch;
pub mod op;
pub mod overlap;
pub mod vision;
pub mod wideep_mla;
pub mod wideep_moe;

pub use attention::{ContextAttentionOp, EncoderAttentionOp, GenerationAttentionOp};
pub use base::{PerformanceResult, Source};
pub use communication::{CustomAllReduceOp, NcclOp, P2POp};
pub use dsa::DsaModuleOp;
pub use dsv4::Dsv4ModuleOp;
pub use elementwise::ElementwiseOp;
pub use embedding::EmbeddingOp;
pub use gemm::GemmOp;
pub use mamba::{GdnOp, Mamba2Op};
pub use mhc::MhcModuleOp;
pub use mla::{ContextMlaOp, GenerationMlaOp, MlaBmmOp, MlaModuleOp};
pub use moe::MoeOp;
pub use moe_dispatch::{DispatchFlavor, MoEDispatchOp, TrtllmWideEpMoEDispatchOp};
pub use op::{FallbackOp, Op, OverlapOp, RuntimeContext};
pub use vision::VisionEncoderOp;
pub use wideep_mla::{WideEpContextMlaOp, WideEpGenerationMlaOp};
pub use wideep_moe::WideEpMoeOp;
