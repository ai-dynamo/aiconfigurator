// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SOL-mode op queries for the FPM whole-model roofline.
//!
//! Python's `forward_model="fpm"` derives its interpolation roofline from the
//! model's ORIGINAL op-level list queried on a `DatabaseMode.SOL` database
//! view: the op code runs unchanged, and every `database.query_*` call
//! answers its analytic `get_sol()` branch instead of a table lookup. Rust
//! has no database-mode switch, so this module mirrors, per op family, the
//! composition "op-level shape math × SOL leaf formula" as ONE function of
//! the (possibly NON-INTEGER) inputs.
//!
//! Inputs are `f64` on purpose: the FPM sol_fn back-maps iteration totals as
//! `s = max(total/batch, 1.0)` and `prefix = total_kv/batch`, which are
//! fractional whenever `batch` does not divide the totals — and Python keeps
//! them fractional through the whole SOL chain. The floor/ceil sites below
//! exist exactly where the Python chain floor-divides (`//` on floats is a
//! float floor; `-(-x // d)` is a float ceil) — there is NO `int()` cast
//! anywhere on the Python SOL path.
//!
//! Formula provenance (file:line refers to the Python SDK):
//! - GEMM: `operations/gemm.py:436-443` (`get_sol`), `:748-749` (m mapping),
//!   `:282-287` (tc_flops selection). fp8_static's subtraction chain is
//!   floored back to the plain GEMM SOL under SOL mode (`:787-800`), so both
//!   quant classes share one formula.
//! - Attention: `operations/attention.py:319-341` (context `get_sol`,
//!   prefix-aware), `:710-733` (generation `get_sol`), `:531` (CP chunk
//!   ceil), `:535-548` (fused rope/kv-write/qk-norm extras × 1.1 on the SOL
//!   mem-op `bytes / mem_bw * 1000`, `perf_database.py:2294`).
//! - MoE: `operations/moe.py:297-325` (`get_sol` with the activated-expert
//!   clamp; `workload_distribution` deliberately unused).
//! - MoE dispatch: `operations/moe.py:1083-1342` branch structure with the
//!   collective SOLs below; SGLang DeepEP SOL raises in Python
//!   (`NotImplementedError`) and errors here.
//! - Collectives: NCCL `operations/communication.py:384-394`, custom
//!   allreduce `:129-136` (ring, hard-coded 2 B/elem, quant ignored), P2P
//!   `:556-559` (always `inter_node_bw`, NO `p2p_latency` under SOL).
//! - Embedding/ElementWise: `operations/embedding.py:49-62`,
//!   `elementwise.py:49-66` over the SOL mem-op.
//!
//! Known, documented approximation: Python `ElementWise` floors
//! `x // scale_num_tokens` before converting tokens to bytes; the Rust
//! `ElementwiseOp` wire form folds `scale_num_tokens` into a continuous
//! `bytes_per_token` (see `engine.py::_elementwise`), losing that floor for
//! `x % scale_num_tokens != 0`. This is the SAME approximation the silicon
//! engine-step path already ships; the error is bounded by one token's bytes
//! and is negligible against the whole-model roofline.

use crate::common::enums::{BackendKind, GemmQuantMode};
use crate::common::error::AicError;
use crate::common::system_spec::SystemSpec;
use crate::operators::op::Op;
use crate::operators::{
    ContextAttentionOp, CustomAllReduceOp, ElementwiseOp, EmbeddingOp, GemmOp,
    GenerationAttentionOp, MoEDispatchOp, MoeOp, NcclOp, P2POp,
};
use crate::perf_database::PerfDatabase;

/// Python float floor-division `a // b`.
fn floor_div(a: f64, b: f64) -> f64 {
    (a / b).floor()
}

/// Python `-(-a // b)`: exact ceiling that stays float.
fn ceil_div(a: f64, b: f64) -> f64 {
    (a / b).ceil()
}

/// SOL-mode latency (ms) of one granular op at the FPM sol coordinates.
///
/// `x` is Python's `x` kwarg (token count for compute ops; `batch` for
/// `logits_gemm`), `batch`/`s`/`prefix` mirror `op.query(view, x=x,
/// batch_size=batch, beam_width=1, s=s, prefix=prefix)`. Ops ignore the
/// kwargs they ignore in Python.
pub(crate) fn op_sol_latency_ms(
    op: &Op,
    db: &PerfDatabase,
    x: f64,
    batch: f64,
    s: f64,
    prefix: f64,
) -> Result<f64, AicError> {
    let spec = &db.system_spec;
    match op {
        Op::Gemm(o) => Ok(gemm_sol(o, spec, x)),
        Op::Embedding(o) => Ok(embedding_sol(o, spec, x)),
        Op::Elementwise(o) => Ok(elementwise_sol(o, spec, x)),
        Op::ContextAttention(o) => Ok(context_attention_sol(o, spec, batch, s, prefix)),
        Op::GenerationAttention(o) => Ok(generation_attention_sol(o, spec, batch, s)),
        Op::Moe(o) => Ok(moe_sol(o, spec, x)),
        Op::MoeDispatch(o) => moe_dispatch_sol(o, spec, x),
        Op::CustomAllReduce(o) => Ok(custom_allreduce_op_sol(o, spec, x)),
        Op::Nccl(o) => Ok(nccl_op_sol(o, spec, x)),
        Op::P2P(o) => Ok(p2p_sol(o, spec, x)),
        // Python `OverlapOp.query` under SOL: each group summed, max of the
        // two totals.
        Op::Overlap(o) => {
            let mut total_a = 0.0;
            for inner in &o.group_a {
                total_a += op_sol_latency_ms(inner, db, x, batch, s, prefix)?;
            }
            let mut total_b = 0.0;
            for inner in &o.group_b {
                total_b += op_sol_latency_ms(inner, db, x, batch, s, prefix)?;
            }
            Ok(total_a.max(total_b))
        }
        // Python `FallbackOp.query` under SOL: the primary's analytic SOL
        // answers (no data to miss); mirror the perf-DB-miss fallback anyway.
        Op::Fallback(o) => match op_sol_latency_ms(&o.primary, db, x, batch, s, prefix) {
            Ok(v) => Ok(v),
            Err(AicError::PerfDatabase(_)) | Err(AicError::Io { .. }) => {
                let mut total = 0.0;
                for inner in &o.fallback {
                    total += op_sol_latency_ms(inner, db, x, batch, s, prefix)?;
                }
                Ok(total)
            }
            Err(other) => Err(other),
        },
        other => Err(AicError::UnsupportedModel(format!(
            "forward_model='fpm' SOL roofline has no Rust implementation for op {}",
            other.name()
        ))),
    }
}

/// Python `GEMM._get_quant_tc_flops` (gemm.py:282-287): compute factor
/// 1/2/4 -> the matching spec field when present, else
/// `bfloat16_tc_flops * compute`.
fn quant_tc_flops(spec: &SystemSpec, quant: GemmQuantMode) -> f64 {
    let compute = quant.mapping().compute;
    let direct = if compute == 1.0 {
        spec.gpu.bfloat16_tc_flops
    } else if compute == 2.0 {
        spec.gpu.fp8_tc_flops
    } else if compute == 4.0 {
        spec.gpu.fp4_tc_flops
    } else {
        None
    };
    direct.unwrap_or_else(|| spec.gpu.bfloat16_tc_flops.unwrap_or(0.0) * compute)
}

/// gemm.py:436-443 + the m mapping at :748-749. Under SOL, fp8_static's
/// subtraction chain is floored back to this same value (:787-800).
fn gemm_sol(op: &GemmOp, spec: &SystemSpec, x: f64) -> f64 {
    // Python: `x //= scale_num_tokens` (floor, fires even at 1 for fractional
    // x), then `x = -(-x // seq_split)` (ceil).
    let m = ceil_div(
        floor_div(x, op.scale_num_tokens.max(1) as f64),
        op.seq_split.max(1) as f64,
    );
    let (n, k) = (op.n as f64, op.k as f64);
    let mapping = op.quant_mode.mapping();
    let tc_flops = quant_tc_flops(spec, op.quant_mode);
    let sol_math = 2.0 * m * n * k / tc_flops * 1000.0;
    let sol_mem = mapping.memory * (m * n + m * k + n * k) / spec.gpu.mem_bw * 1000.0;
    sol_math.max(sol_mem) * op.scale_factor
}

/// The SOL mem-op leaf (perf_database.py:2294): `bytes / mem_bw * 1000` —
/// no empirical scaling factor, no constant latency.
fn mem_op_sol_ms(spec: &SystemSpec, mem_bytes: f64) -> f64 {
    mem_bytes / spec.gpu.mem_bw * 1000.0
}

/// embedding.py:49-62: `x = -(-x // seq_split)`, `d2d_bytes = x * hidden * 2`
/// (hard-coded bf16 bytes), one SOL mem-op.
fn embedding_sol(op: &EmbeddingOp, spec: &SystemSpec, x: f64) -> f64 {
    let tokens = ceil_div(x, op.seq_split.max(1) as f64);
    mem_op_sol_ms(spec, tokens * op.hidden_size as f64 * 2.0) * op.scale_factor
}

/// elementwise.py:49-66 over the folded `bytes_per_token` wire form (see the
/// module doc for the scale_num_tokens floor approximation).
fn elementwise_sol(op: &ElementwiseOp, spec: &SystemSpec, x: f64) -> f64 {
    // Python: `x //= scale_num_tokens` (floor) THEN `-(-x // seq_split)`
    // (ceil). The wire op carries scale_num_tokens since schema v4, so the
    // floor is exact (older folded-bytes specs deserialize with divisor 1).
    let tokens = ceil_div(
        floor_div(x, op.scale_num_tokens.max(1) as f64),
        op.seq_split.max(1) as f64,
    );
    mem_op_sol_ms(spec, op.bytes_per_token * tokens) * op.scale_factor
}

/// attention.py:319-341 — the prefix-aware context SOL (the crate's
/// `context_attention_sol_ms` is the prefix=0 specialization used by the
/// silicon interp anchors, so it cannot be reused here).
fn context_sol_one(op: &ContextAttentionOp, spec: &SystemSpec, b: f64, s: f64, p: f64) -> f64 {
    let (n, n_kv, h, w) = (
        op.n as f64,
        op.n_kv as f64,
        op.head_size as f64,
        op.window_size as f64,
    );
    let full_s = s + p;
    let ops = if op.window_size > 0 && full_s > w {
        // windowed: (full_s - p) = s new tokens each attend a w-window; no
        // causal halving in this branch (Python :332-333)
        2.0 * b * s * w * n * h * 2.0
    } else {
        2.0 * b * (full_s * full_s - p * p) * n * h * 2.0 / 2.0
    };
    // Q read + O write in bf16 on the new tokens; K and V over the FULL
    // sequence at kv-cache width. The window does NOT shrink mem (Python).
    let mem = 2.0 * b * (n * s * h + n * s * h)
        + op.kv_cache_dtype.mapping().memory * b * (2.0 * n_kv * full_s * h);
    let flops = spec.gpu.bfloat16_tc_flops.unwrap_or(0.0);
    let sol_math = ops / flops * 1000.0 / op.fmha_quant_mode.mapping().compute;
    let sol_mem = mem / spec.gpu.mem_bw * 1000.0;
    sol_math.max(sol_mem)
}

/// ContextAttention.query under SOL (attention.py:507-558): CP zigzag chunks
/// + the fused rope/kv-write/(qk-norm) extras × 1.1, each a SOL mem-op.
fn context_attention_sol(
    op: &ContextAttentionOp,
    spec: &SystemSpec,
    b: f64,
    s: f64,
    p: f64,
) -> f64 {
    let fmha = if op.cp_size > 1 {
        // Python :531: `c = max(1, -(-isl // (2 * cp)))` — float ceil.
        let c = ceil_div(s, 2.0 * op.cp_size as f64).max(1.0);
        context_sol_one(op, spec, b, c, p) + context_sol_one(op, spec, b, c, p + s - c)
    } else {
        context_sol_one(op, spec, b, s, p)
    };
    let q_num = (op.n * op.head_size) as f64;
    let k_num = (op.n_kv * op.head_size) as f64;
    let mut extra = 0.0;
    if op.use_qk_norm {
        let qk_norm =
            2.0 * mem_op_sol_ms(spec, q_num * 2.0) + 2.0 * mem_op_sol_ms(spec, k_num * 2.0);
        extra += qk_norm * 2.0;
    }
    extra += 2.0 * mem_op_sol_ms(spec, q_num * 2.0 + k_num * 2.0); // rope
    let fq_mem = op.fmha_quant_mode.mapping().memory;
    extra += mem_op_sol_ms(spec, k_num * fq_mem) + mem_op_sol_ms(spec, k_num * fq_mem); // kv write (k_num == v_num)
    (fmha + extra * 1.1) * op.scale_factor
}

/// attention.py:710-733: generation SOL. No extras, no 5-sample smoothing
/// (both are silicon-only), no prefix.
fn generation_attention_sol(op: &GenerationAttentionOp, spec: &SystemSpec, b: f64, s: f64) -> f64 {
    let (n, n_kv, h, w) = (
        op.n as f64,
        op.n_kv as f64,
        op.head_size as f64,
        op.window_size as f64,
    );
    let kv_len = if op.window_size > 0 {
        (s - 1.0).min(w)
    } else {
        s - 1.0
    };
    // fp8 KV -> fp8 compute; everything else (incl. int8 KV) -> bf16 compute.
    let compute = if op.kv_cache_dtype == crate::common::enums::KvCacheQuantMode::Fp8 {
        2.0
    } else {
        1.0
    };
    let kv_mem = op.kv_cache_dtype.mapping().memory;
    let ops = 2.0 * b * n * h * 2.0 * kv_len;
    let mem = b * (n * h * 2.0 + 2.0 * n_kv * kv_len * h * kv_mem + n * h * 2.0);
    let flops = spec.gpu.bfloat16_tc_flops.unwrap_or(0.0);
    let sol_math = ops / flops * 1000.0 / compute;
    let sol_mem = mem / spec.gpu.mem_bw * 1000.0;
    sol_math.max(sol_mem) * op.scale_factor
}

/// moe.py:297-325: MoE SOL with the activated-expert clamp. The `//` sites
/// are float floors in exactly Python's association order.
fn moe_sol(op: &MoeOp, spec: &SystemSpec, x: f64) -> f64 {
    let dp = op.attention_dp_size.max(1) as f64;
    let (h, inter) = (op.hidden_size as f64, op.inter_size as f64);
    let num_gemms = if op.is_gated { 3.0 } else { 2.0 };
    let (ep, tp) = (op.moe_ep_size.max(1) as f64, op.moe_tp_size.max(1) as f64);
    let total_tokens = x * dp * op.topk as f64;
    // ops = TT*H*I*G*2 // ep // tp
    let ops = floor_div(
        floor_div(total_tokens * h * inter * num_gemms * 2.0, ep),
        tp,
    );
    // mem = m * ( TT//ep*H*2 + TT//ep*I*G//tp + H*I*G//tp * min(E//ep, TT//ep) )
    let tt_ep = floor_div(total_tokens, ep);
    let mem = op.quant_mode.mapping().memory
        * (tt_ep * h * 2.0
            + floor_div(tt_ep * inter * num_gemms, tp)
            + floor_div(h * inter * num_gemms, tp)
                * floor_div(op.num_experts as f64, ep).min(tt_ep));
    let flops = spec.gpu.bfloat16_tc_flops.unwrap_or(0.0);
    let sol_math = ops / (flops * op.quant_mode.mapping().compute) * 1000.0;
    let sol_mem = mem / spec.gpu.mem_bw * 1000.0;
    sol_math.max(sol_mem) * op.scale_factor
}

/// communication.py:129-136: ring allreduce, hard-coded 2 B/elem, quant
/// ignored, real tp_size (no node capping), no latency constant.
fn custom_allreduce_sol(spec: &SystemSpec, tp_size: u32, size_elems: f64) -> f64 {
    if tp_size <= 1 {
        return 0.0;
    }
    let tp = tp_size as f64;
    let bw = spec.get_p2p_bandwidth(tp_size);
    2.0 * size_elems * 2.0 / tp * (tp - 1.0) / bw * 1000.0
}

/// communication.py:384-394: NCCL collective SOL. `message_size` is an
/// element count scaled by the dtype's byte width; unknown op names are 0.
fn nccl_sol(
    spec: &SystemSpec,
    num_gpus: u32,
    operation: &str,
    message_size: f64,
    bytes_per_elem: f64,
) -> f64 {
    let n = num_gpus as f64;
    let bw = spec.get_p2p_bandwidth(num_gpus);
    match operation {
        "all_gather" | "alltoall" | "reduce_scatter" => {
            bytes_per_elem * message_size * (n - 1.0) / n / bw * 1000.0
        }
        "all_reduce" => 2.0 * bytes_per_elem * message_size * (n - 1.0) / n / bw * 1000.0,
        _ => 0.0,
    }
}

/// CustomAllReduce.query under SOL (communication.py:252-268).
fn custom_allreduce_op_sol(op: &CustomAllReduceOp, spec: &SystemSpec, x: f64) -> f64 {
    if op.tp_size == 1 {
        return 0.0;
    }
    let size = ceil_div(x, op.seq_split.max(1) as f64) * op.hidden_size as f64;
    custom_allreduce_sol(spec, op.tp_size, size) * op.scale_factor
}

/// NCCL.query under SOL (communication.py:509-519).
fn nccl_op_sol(op: &NcclOp, spec: &SystemSpec, x: f64) -> f64 {
    let msg = ceil_div(x, op.seq_split.max(1) as f64) * op.hidden_size;
    nccl_sol(
        spec,
        op.num_gpus,
        &op.operation,
        msg,
        op.dtype.mapping().memory,
    ) * op.scale_factor
}

/// P2P.query under SOL (communication.py:583-598 + :556-559): always
/// `inter_node_bw`, literal 2 B/elem, NO `p2p_latency` term.
fn p2p_sol(op: &P2POp, spec: &SystemSpec, x: f64) -> f64 {
    if op.pp_size == 1 {
        return 0.0;
    }
    let p2p_bytes = ceil_div(x, op.seq_split.max(1) as f64) * op.hidden_size as f64 * 2.0;
    p2p_bytes / spec.node.inter_node_bw * 1000.0 * op.scale_factor
}

/// MoEDispatch.query under SOL (moe.py:1083-1342). Branch structure mirrors
/// the op's silicon query (`moe_dispatch.rs`), with the collective SOLs
/// substituted for the table lookups. Message sizes here are passed straight
/// to the DB-level collectives in Python (no per-op ceil), so no ceil either.
fn moe_dispatch_sol(op: &MoEDispatchOp, spec: &SystemSpec, x: f64) -> Result<f64, AicError> {
    use crate::operators::moe_dispatch::DispatchFlavor;

    let volume = x * op.hidden_size as f64; // element count, half precision
    let num_gpus = (op.moe_tp_size * op.moe_ep_size).max(1);
    let attn_dp = op.attention_dp_size.max(1);
    let attn_tp = (num_gpus / attn_dp).max(1);
    let dp = attn_dp as f64;
    let pre = op.pre_dispatch;
    let half_bytes = 2.0; // CommQuantMode::Half.memory — MoEDispatch always passes half

    let comm = match op.flavor {
        DispatchFlavor::RetiredDeepEp => {
            return Err(AicError::InvalidEngineConfig(format!(
                "MoEDispatch '{}' (moe_backend='deepep_moe') has no native SOL \
                 (retired with AIC-1601; large-EP comm is modeled by MoeAllToAll)",
                op.name
            )))
        }
        DispatchFlavor::CustomAllReduce => match op.backend {
            // vllm (moe.py:1222-1239): additive.
            BackendKind::Vllm => {
                let mut total = 0.0;
                if attn_tp > 1 {
                    total += custom_allreduce_sol(spec, num_gpus, volume);
                }
                if attn_dp > 1 {
                    let op_name = if pre { "all_gather" } else { "reduce_scatter" };
                    total += nccl_sol(spec, num_gpus, op_name, volume * dp, half_bytes);
                }
                total
            }
            // sglang non-deepep (moe.py:1261-1342).
            BackendKind::Sglang => {
                if attn_tp > 1 && attn_dp > 1 {
                    if pre {
                        nccl_sol(spec, attn_tp, "reduce_scatter", volume, half_bytes)
                            + nccl_sol(spec, num_gpus, "all_gather", volume * dp, half_bytes)
                    } else {
                        nccl_sol(spec, num_gpus, "reduce_scatter", volume * dp, half_bytes)
                            + nccl_sol(spec, attn_tp, "all_gather", volume, half_bytes)
                    }
                } else if op.attn_cp_size > 1 {
                    if op.is_context {
                        let op_name = if pre { "all_gather" } else { "reduce_scatter" };
                        nccl_sol(spec, num_gpus, op_name, volume, half_bytes)
                    } else if pre {
                        0.0
                    } else {
                        custom_allreduce_sol(spec, num_gpus, volume)
                    }
                } else if attn_tp > 1 {
                    custom_allreduce_sol(spec, num_gpus, volume)
                } else if attn_dp > 1 {
                    let op_name = if pre { "all_gather" } else { "reduce_scatter" };
                    nccl_sol(spec, num_gpus, op_name, volume * dp, half_bytes)
                } else {
                    0.0
                }
            }
            // trtllm, sm != 100 (moe.py:1194-1221): pre/post symmetric.
            BackendKind::Trtllm => {
                if attn_tp > 1 {
                    custom_allreduce_sol(spec, num_gpus, volume)
                } else if attn_dp > 1 {
                    let op_name = if pre { "all_gather" } else { "reduce_scatter" };
                    nccl_sol(spec, num_gpus, op_name, volume * dp, half_bytes)
                } else {
                    0.0
                }
            }
        },
        // trtllm SM100 (moe.py:1095-1193).
        DispatchFlavor::TrtllmAlltoall => {
            let is_nvl72 = spec.node.num_gpus_per_node >= 72;
            let enable_alltoall = op.attention_dp_size > 1 && op.moe_tp_size == 1 && is_nvl72;
            if enable_alltoall {
                // trtllm_alltoall SOL (moe.py:2068-2107): dispatch moves the
                // moe-quant-compressed activations, combine moves bf16.
                let node_num = if op.moe_ep_size < 4 {
                    1
                } else {
                    op.moe_ep_size / 4
                };
                let bw = if node_num > 1 {
                    spec.node.inter_node_bw
                } else {
                    spec.node.intra_node_bw
                };
                let remote_ranks =
                    op.topk
                        .min(op.num_experts)
                        .min(op.moe_ep_size.saturating_sub(1)) as f64;
                let bytes_per_elem = if pre {
                    op.moe_quant.mapping().memory
                } else {
                    2.0
                };
                let data_bytes = x * remote_ranks * op.hidden_size as f64 * bytes_per_elem;
                data_bytes / bw * 1000.0
            } else if op.attention_dp_size > 1 {
                // moe.py:1142-1145 / :1173-1179: the pre all_gather moves the
                // moe-quant-COMPRESSED volume (nvfp4: V/4 + V/32; fp8: V/2);
                // the post reduce_scatter is uncompressed (asymmetric).
                if pre {
                    let compressed = match op.moe_quant.mapping().name {
                        "nvfp4" => volume / 4.0 + volume / 4.0 / 8.0,
                        "fp8" | "fp8_block" => volume / 2.0,
                        _ => volume,
                    };
                    nccl_sol(spec, num_gpus, "all_gather", compressed * dp, half_bytes)
                } else {
                    nccl_sol(spec, num_gpus, "reduce_scatter", volume * dp, half_bytes)
                }
            } else if attn_tp > 1 {
                // reduce_results defaults true (mirrors the silicon port).
                if spec.node.num_gpus_per_node == 72 && num_gpus > 4 {
                    nccl_sol(spec, num_gpus, "all_reduce", volume, half_bytes)
                } else {
                    custom_allreduce_sol(spec, num_gpus, volume)
                }
            } else {
                0.0
            }
        }
    };
    Ok(comm * op.scale_factor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    use crate::common::enums::{
        CommQuantMode, FmhaQuantMode, GemmQuantMode, KvCacheQuantMode, MoeQuantMode,
    };

    /// The b200_sxm spec: mem_bw 7.7e12, bf16 2.25e15, fp8 4.5e15, fp4 9e15,
    /// intra_node_bw 8.1e11, inter_node_bw 4e10 (systems/b200_sxm.yaml).
    fn spec() -> SystemSpec {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../src/aiconfigurator_core/systems/b200_sxm.yaml");
        SystemSpec::load(&root).expect("b200 spec")
    }

    fn db() -> PerfDatabase {
        let root =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../src/aiconfigurator_core/systems");
        PerfDatabase::load(&root, "b200_sxm", "vllm", "0.19.0").expect("db")
    }

    fn approx(got: f64, expected: f64) {
        assert!(
            (got - expected).abs() <= 1e-9 * expected.abs().max(1e-12),
            "got {got}, expected {expected}"
        );
    }

    /// Python oracle:
    /// PYTHONPATH=aic-core/src python3 -c "
    /// from aiconfigurator_core.sdk import perf_database, common
    /// from aiconfigurator_core.sdk.operations.gemm import GEMM
    /// view = perf_database.get_database_view('b200_sxm','vllm','0.19.0',database_mode='SOL',allow_missing_data=True)
    /// op = GEMM('qkv_gemm', 1.0, 4096, 4096, common.GEMMQuantMode.nvfp4)
    /// print(repr(float(op.query(view, x=8192.0, batch_size=4, beam_width=1, s=2048.0, prefix=0.0))))"
    #[test]
    fn gemm_sol_matches_formula() {
        let op = GemmOp {
            name: "qkv_gemm".into(),
            scale_factor: 1.0,
            n: 4096,
            k: 4096,
            quant_mode: GemmQuantMode::Nvfp4,
            scale_num_tokens: 1,
            low_precision_input: false,
            seq_split: 1,
            below_grid_sol: false,
        };
        let s = spec();
        let m = 8192.0_f64;
        let (n, k) = (4096.0_f64, 4096.0_f64);
        let tc = s.gpu.fp4_tc_flops.unwrap();
        let expected = (2.0 * m * n * k / tc * 1000.0)
            .max(9.0 / 16.0 * (m * n + m * k + n * k) / s.gpu.mem_bw * 1000.0);
        approx(gemm_sol(&op, &s, 8192.0), expected);
        // fractional x floors first (even at scale_num_tokens == 1)
        approx(gemm_sol(&op, &s, 8192.7), expected);
    }

    #[test]
    fn embedding_vs_elementwise_rounding_directions() {
        let s = spec();
        let emb = EmbeddingOp {
            name: "context_embedding".into(),
            scale_factor: 1.0,
            vocab_size: 128000,
            hidden_size: 6144,
            quant_mode: GemmQuantMode::Bfloat16,
            seq_split: 1,
        };
        // Embedding CEILS fractional x: 10.5 -> 11 tokens.
        approx(
            embedding_sol(&emb, &s, 10.5),
            11.0 * 6144.0 * 2.0 / s.gpu.mem_bw * 1000.0,
        );
        let ew = ElementwiseOp {
            name: "add_norm".into(),
            scale_factor: 2.0,
            bytes_per_token: 8192.0,
            scale_num_tokens: 1,
            seq_split: 1,
        };
        // Elementwise FLOORS first (Python `x //= scale_num_tokens` fires
        // even at divisor 1): 10.5 -> 10 tokens — the OPPOSITE rounding
        // direction from Embedding. The wire op carries scale_num_tokens
        // since schema v4, so the floor is exact.
        approx(
            elementwise_sol(&ew, &s, 10.5),
            8192.0 * 10.0 / s.gpu.mem_bw * 1000.0 * 2.0,
        );
    }

    /// Python oracle:
    /// PYTHONPATH=aic-core/src python3 -c "
    /// from aiconfigurator_core.sdk import perf_database, common
    /// from aiconfigurator_core.sdk.operations.attention import ContextAttention, GenerationAttention
    /// view = perf_database.get_database_view('b200_sxm','vllm','0.19.0',database_mode='SOL',allow_missing_data=True)
    /// op = ContextAttention('context_attention', 1.0, 48, 8, 128, kvcache_quant_mode=common.KVCacheQuantMode.fp8, fmha_quant_mode=common.FMHAQuantMode.bfloat16)
    /// print(repr(float(op.query(view, x=1, batch_size=4.0, beam_width=1, s=682.6666666666666, prefix=128.5))))"
    #[test]
    fn context_attention_sol_prefix_aware() {
        let s = spec();
        let op = ContextAttentionOp {
            name: "context_attention".into(),
            scale_factor: 1.0,
            n: 48,
            n_kv: 8,
            head_size: 128,
            window_size: 0,
            kv_cache_dtype: KvCacheQuantMode::Fp8,
            fmha_quant_mode: FmhaQuantMode::Bfloat16,
            use_qk_norm: false,
            cp_size: 1,
            lane_order: crate::operators::attention::b200_vllm_lane_order(),
        };
        let (b, sq, p) = (4.0, 682.6666666666666_f64, 128.5_f64);
        let (n, n_kv, h) = (48.0, 8.0, 128.0);
        let full = sq + p;
        let ops = 2.0 * b * (full * full - p * p) * n * h * 2.0 / 2.0;
        let mem = 2.0 * b * (n * sq * h + n * sq * h) + 1.0 * b * (2.0 * n_kv * full * h);
        let fmha = (ops / s.gpu.bfloat16_tc_flops.unwrap() * 1000.0 / 1.0)
            .max(mem / s.gpu.mem_bw * 1000.0);
        let q_num = n * h;
        let k_num = n_kv * h;
        let extras = 2.0 * (q_num * 2.0 + k_num * 2.0) / s.gpu.mem_bw * 1000.0
            + (k_num * 2.0) / s.gpu.mem_bw * 1000.0
            + (k_num * 2.0) / s.gpu.mem_bw * 1000.0;
        approx(
            context_attention_sol(&op, &s, b, sq, p),
            fmha + extras * 1.1,
        );
    }

    #[test]
    fn generation_attention_sol_fp8_kv_uses_fp8_compute() {
        let s = spec();
        let op = GenerationAttentionOp {
            name: "generation_attention".into(),
            scale_factor: 1.0,
            n: 48,
            n_kv: 8,
            head_size: 128,
            window_size: 0,
            kv_cache_dtype: KvCacheQuantMode::Fp8,
            lane_order: crate::operators::attention::b200_vllm_lane_order(),
        };
        let (b, sq) = (256.0, 8441.75_f64);
        let kv_len = sq - 1.0;
        let ops = 2.0 * b * 48.0 * 128.0 * 2.0 * kv_len;
        let mem = b * (48.0 * 128.0 * 2.0 + 2.0 * 8.0 * kv_len * 128.0 * 1.0 + 48.0 * 128.0 * 2.0);
        let expected = (ops / s.gpu.bfloat16_tc_flops.unwrap() * 1000.0 / 2.0)
            .max(mem / s.gpu.mem_bw * 1000.0);
        approx(generation_attention_sol(&op, &s, b, sq), expected);
    }

    /// Mirrors moe.py:297-325 with the float-floor association order.
    #[test]
    fn moe_sol_floor_association() {
        let s = spec();
        let op = MoeOp {
            name: "context_moe".into(),
            scale_factor: 1.0,
            hidden_size: 6144,
            inter_size: 1536,
            topk: 8,
            num_experts: 256,
            moe_tp_size: 1,
            moe_ep_size: 4,
            attention_dp_size: 1,
            quant_mode: MoeQuantMode::Nvfp4,
            workload_distribution: "uniform".into(),
            is_gated: true,
            moe_backend: None,
            enable_eplb: false,
            is_context: true,
        };
        let x = 8192.0_f64;
        let tt = x * 8.0;
        let ops = ((tt * 6144.0 * 1536.0 * 3.0 * 2.0 / 4.0).floor() / 1.0).floor();
        let tt_ep = (tt / 4.0).floor();
        let mem = 9.0 / 16.0
            * (tt_ep * 6144.0 * 2.0
                + (tt_ep * 1536.0 * 3.0 / 1.0).floor()
                + (6144.0_f64 * 1536.0 * 3.0 / 1.0).floor() * (256.0_f64 / 4.0).floor().min(tt_ep));
        let expected = (ops / (s.gpu.bfloat16_tc_flops.unwrap() * 4.0) * 1000.0)
            .max(mem / s.gpu.mem_bw * 1000.0);
        approx(moe_sol(&op, &s, x), expected);
    }

    #[test]
    fn comm_sols_match_formulas() {
        let s = spec();
        // custom allreduce: ring, hard-coded 2 B/elem
        let car = CustomAllReduceOp {
            name: "ar".into(),
            scale_factor: 1.0,
            hidden_size: 6144,
            tp_size: 4,
            quant: CommQuantMode::Half,
            seq_split: 1,
        };
        let size = 8192.0 * 6144.0;
        let bw = s.get_p2p_bandwidth(4);
        approx(
            custom_allreduce_op_sol(&car, &s, 8192.0),
            2.0 * size * 2.0 / 4.0 * 3.0 / bw * 1000.0,
        );
        // tp==1 -> 0
        let car1 = CustomAllReduceOp {
            tp_size: 1,
            ..car.clone()
        };
        assert_eq!(custom_allreduce_op_sol(&car1, &s, 8192.0), 0.0);

        // P2P: always inter_node_bw, no latency constant
        let p2p = P2POp {
            name: "p2p".into(),
            scale_factor: 1.0,
            pp_size: 2,
            hidden_size: 6144,
            seq_split: 1,
        };
        approx(
            p2p_sol(&p2p, &s, 8192.0),
            8192.0 * 6144.0 * 2.0 / s.node.inter_node_bw * 1000.0,
        );

        // NCCL all_reduce doubles the gather/scatter traffic
        let nccl = NcclOp {
            name: "nccl".into(),
            scale_factor: 1.0,
            hidden_size: 6144.0,
            num_gpus: 8,
            dtype: CommQuantMode::Half,
            operation: "all_reduce".into(),
            seq_split: 1,
        };
        let bw8 = s.get_p2p_bandwidth(8);
        approx(
            nccl_op_sol(&nccl, &s, 1024.0),
            2.0 * 2.0 * (1024.0 * 6144.0) * 7.0 / 8.0 / bw8 * 1000.0,
        );
    }

    #[test]
    fn moe_dispatch_vllm_is_additive() {
        let s = spec();
        let op = MoEDispatchOp {
            name: "context_moe_pre_dispatch".into(),
            scale_factor: 1.0,
            hidden_size: 6144,
            topk: 8,
            num_experts: 256,
            moe_tp_size: 1,
            moe_ep_size: 4,
            attention_dp_size: 1,
            pre_dispatch: true,
            backend: BackendKind::Vllm,
            flavor: crate::operators::moe_dispatch::DispatchFlavor::CustomAllReduce,
            comm_quant: CommQuantMode::Half,
            moe_quant: MoeQuantMode::Nvfp4,
            attn_cp_size: 1,
            is_context: true,
            sms: 12,
            scale_num_tokens: 1,
            attn_ar_modeled: false,
        };
        // dp=1, attn_tp = 4/1 = 4 > 1 -> allreduce only
        let volume = 8192.0 * 6144.0;
        let bw = s.get_p2p_bandwidth(4);
        approx(
            moe_dispatch_sol(&op, &s, 8192.0).unwrap(),
            2.0 * volume * 2.0 / 4.0 * 3.0 / bw * 1000.0,
        );
    }

    #[test]
    fn overlap_and_fallback_compose() {
        let d = db();
        let ew = |bpt: f64| {
            Op::Elementwise(ElementwiseOp {
                name: "e".into(),
                scale_factor: 1.0,
                bytes_per_token: bpt,
                scale_num_tokens: 1,
                seq_split: 1,
            })
        };
        let overlap = Op::Overlap(crate::operators::OverlapOp::new(
            "ov",
            vec![ew(1000.0), ew(2000.0)],
            vec![ew(5000.0)],
        ));
        let expected = super::mem_op_sol_ms(&d.system_spec, 5000.0 * 64.0);
        approx(
            op_sol_latency_ms(&overlap, &d, 64.0, 1.0, 1.0, 0.0).unwrap(),
            expected,
        );
        let fb = Op::Fallback(crate::operators::FallbackOp::new("fb", ew(3000.0), vec![]));
        approx(
            op_sol_latency_ms(&fb, &d, 64.0, 1.0, 1.0, 0.0).unwrap(),
            super::mem_op_sol_ms(&d.system_spec, 3000.0 * 64.0),
        );
    }
}
