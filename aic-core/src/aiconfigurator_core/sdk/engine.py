# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compiled-engine builder.

This module is the Python half of the "Python builds, Rust executes"
architecture. ``compile_engine`` reuses the existing, unmodified Python model
layer (``sdk/models/*.py``) to build a model, walks its
``context_ops`` / ``generation_ops`` lists, converts each ``Operation`` to the
plain-data ``OpSpec`` wire form, and ships the whole thing across the boundary
as a bincode-serialised ``EngineSpec``.

The wire is shaped in two hops:

1. Python builds an ``EngineSpec`` as a JSON-serialisable dict — the ``Op``
   variants are externally tagged (``{"Gemm": {<fields>}}``), matching serde's
   default enum encoding, and the field names match the Rust struct field
   names. JSON is the debuggable wire; a misnamed field or a wrong phase-pair
   tag shows up as a ``serde_json`` decode error.
2. The Rust ``engine_spec_bincode_from_json`` ``#[pyfunction]`` decodes that
   JSON into an ``EngineSpec`` and re-encodes it as bincode bytes (Python can't
   bincode; serde_json round-trips ``EngineConfig``'s flattened layout where
   bincode can't). Those bytes are what ``AicEngine.from_spec`` /
   ``build_aic_engine`` consume.

``EngineHandle`` wraps the compiled bytes plus an ``AicEngine`` and exposes the
per-call surface (``run_static`` / ``predict_*_latency`` / ``mixed_step_latency``
/ ``decode_step_latency``). The agg sweep is orchestrated in Python, so there is
no ``run_agg`` here.

The live ``rust_engine_step.py`` helpers build on this path.
"""

from __future__ import annotations

import json
import os
from typing import Any

import aiconfigurator_core
from aiconfigurator_core.sdk.config_builders import apply_nextn, build_model_config
from aiconfigurator_core.sdk.models import get_model
from aiconfigurator_core.sdk.operations import (
    GEMM,
    NCCL,
    P2P,
    ContextAttention,
    ContextDeepSeekV4AttentionModule,
    ContextDSAModule,
    ContextMLA,
    CustomAllReduce,
    DeepSeekV4MHCModule,
    ElementWise,
    Embedding,
    EncoderAttention,
    FallbackOp,
    GDNKernel,
    GenerationAttention,
    GenerationDeepSeekV4AttentionModule,
    GenerationDSAModule,
    GenerationMLA,
    Mamba2Kernel,
    MLABmm,
    MLAModule,
    MoE,
    MoEDispatch,
    OverlapOp,
    TrtLLMWideEPMoE,
    WideEPContextMLA,
    WideEPGenerationMLA,
)
from aiconfigurator_core.sdk.operations.dsa import DEFAULT_DSA_ARCHITECTURE, DSA_MODEL_DIMS

# Reuse the exact quant-mode -> Rust ``DataType`` serde-string mappers the live
# ctypes bridge uses, so the compiled ``EngineConfig`` decodes the same way.
# The Python quant enum names (``int8_wo`` / ``int4_wo`` / ``sq`` / ``fp8_ootb``)
# are NOT valid ``DataType`` variants; these mappers collapse them to the
# accepted strings (``int8`` / ``int4`` / ...).
from aiconfigurator_core.sdk.rust_engine_step import (
    _moe_quant_to_dtype as _rust_moe_quant_to_dtype,
)
from aiconfigurator_core.sdk.rust_engine_step import (
    _quant_to_dtype as _rust_quant_to_dtype,
)

# Schema versions must match the Rust crate constants
# (`ENGINE_SPEC_SCHEMA_VERSION` / `ENGINE_CONFIG_SCHEMA_VERSION` in `lib.rs`).
# ENGINE_SPEC bumped to 2 for the 0.10.0 op-payload layout change (CP + perf-DB
# refactor added serialized `OpSpec` fields); the Rust consumer gates on this
# version before decoding the positional op lists. Keep in lockstep with the
# Rust `ENGINE_SPEC_SCHEMA_VERSION`.
ENGINE_SPEC_SCHEMA_VERSION = 3
ENGINE_CONFIG_SCHEMA_VERSION = 1


class OpConversionError(RuntimeError):
    """Raised when an ``Operation`` cannot be converted to an ``OpSpec``."""


# --------------------------------------------------------------------------- #
# Per-op conversion: Python Operation -> externally-tagged OpSpec dict.
#
# Each helper returns the inner field dict (Rust struct field names). The
# dispatch table below wraps it in `{"<VariantTag>": <fields>}`. All values
# read directly off the Python instance attributes (every field is a
# build-time instance attr); the few cat-2 derived fields
# (Elementwise.bytes_per_token, DSv4.attn_kind, MoeDispatch.flavor,
# WideEpMoe.kernel_source) replicate the Rust model-builder derivation.
# --------------------------------------------------------------------------- #


def _quant_name(quant_mode: Any) -> str:
    """Python quant-mode enum member -> the Rust serde `snake_case` string.

    The Python enum member ``.name`` (e.g. ``fp8_block``, ``int8_wo``) is
    exactly the string the Rust `#[serde(rename_all = "snake_case")]` quant
    enums expect, so this is a direct pass-through.
    """
    return quant_mode.name


def _gemm(op: GEMM) -> dict:
    return {
        "name": op._name,
        "scale_factor": op._scale_factor,
        "n": op._n,
        "k": op._k,
        "quant_mode": _quant_name(op._quant_mode),
        "scale_num_tokens": op._scale_num_tokens,
        "low_precision_input": op._low_precision_input,
        "seq_split": op._seq_split,
    }


def _embedding(op: Embedding) -> dict:
    return {
        "name": op._name,
        "scale_factor": op._scale_factor,
        "vocab_size": op._row_size,
        "hidden_size": op._column_size,
        # Embedding has no quant mode; Rust types it as GemmQuantMode and the
        # query ignores it (memory-only op). Bfloat16 is the neutral value the
        # Rust builders use for embeddings.
        "quant_mode": "bfloat16",
        "seq_split": op._seq_split,
    }


def _elementwise(op: ElementWise) -> dict:
    # Rust `ElementwiseOp.bytes_per_token * num_tokens` must equal Python's
    # `(x * dim_in * 2 + x * dim_out * 2)` after `x //= scale_num_tokens`.
    # Per-token bytes = (dim_in * 2 + dim_out * 2) / scale_num_tokens.
    scale = op._scale_num_tokens if op._scale_num_tokens else 1
    bytes_per_token = (op._dim_in * 2 + op._dim_out * 2) / scale
    return {
        "name": op._name,
        "scale_factor": op._scale_factor,
        "bytes_per_token": float(bytes_per_token),
        "seq_split": op._seq_split,
    }


def _context_attention(op: ContextAttention) -> dict:
    return {
        "name": op._name,
        "scale_factor": op._scale_factor,
        "n": op._n,
        "n_kv": op._n_kv,
        "head_size": op._head_size,
        "window_size": op._window_size,
        "kv_cache_dtype": _quant_name(op._kvcache_quant_mode),
        "fmha_quant_mode": _quant_name(op._fmha_quant_mode),
        "use_qk_norm": op._use_qk_norm,
        "cp_size": op._cp_size,
    }


def _generation_attention(op: GenerationAttention) -> dict:
    return {
        "name": op._name,
        "scale_factor": op._scale_factor,
        "n": op._n,
        "n_kv": op._n_kv,
        "head_size": op._head_size,
        "window_size": op._window_size,
        "kv_cache_dtype": _quant_name(op._kv_cache_dtype),
    }


def _encoder_attention(op: EncoderAttention) -> dict:
    return {
        "name": op._name,
        "scale_factor": op._scale_factor,
        "n": op._n,
        "head_size": op._head_size,
        "fmha_quant_mode": _quant_name(op._fmha_quant_mode),
    }


def _context_mla(op: ContextMLA) -> dict:
    return {
        "name": op._name,
        "scale_factor": op._scale_factor,
        "num_heads": op._num_heads,
        "kv_cache_dtype": _quant_name(op._kvcache_quant_mode),
        "fmha_quant_mode": _quant_name(op._fmha_quant_mode),
        "cp_size": op._cp_size,
    }


def _generation_mla(op: GenerationMLA) -> dict:
    return {
        "name": op._name,
        "scale_factor": op._scale_factor,
        "num_heads": op._num_heads,
        "kv_cache_dtype": _quant_name(op._kv_cache_dtype),
    }


def _mla_module(op: MLAModule) -> dict:
    return {
        "name": op._name,
        "scale_factor": op._scale_factor,
        "num_heads": op._num_heads,
        "kv_cache_dtype": _quant_name(op._kvcache_quant_mode),
        "fmha_quant_mode": _quant_name(op._fmha_quant_mode),
        "gemm_quant_mode": _quant_name(op._gemm_quant_mode),
    }


def _mla_bmm(op: MLABmm) -> dict:
    return {
        "name": op._name,
        "scale_factor": op._scale_factor,
        "num_heads": op._num_heads,
        "quant_mode": _quant_name(op._quant_mode),
        "is_pre": op._if_pre,
    }


def _moe(op: MoE) -> dict:
    return {
        "name": op._name,
        "scale_factor": op._scale_factor,
        "hidden_size": op._hidden_size,
        "inter_size": op._inter_size,
        "topk": op._topk,
        "num_experts": op._num_experts,
        "moe_tp_size": op._moe_tp_size,
        "moe_ep_size": op._moe_ep_size,
        "attention_dp_size": op._attention_dp_size,
        "quant_mode": _quant_name(op._quant_mode),
        "workload_distribution": op._workload_distribution,
        "is_gated": op._is_gated,
    }


def _dispatch_flavor(backend: str) -> str:
    """Mirror Rust `models/moe.rs::dispatch_flavor`: trtllm -> TrtllmAlltoall,
    everything else -> CustomAllReduce. The `DispatchFlavor` enum has no serde
    rename, so emit the exact Rust variant name. The fine SGLang-DeepEP /
    NVL72 gating stays inside the Rust `MoEDispatchOp::query`.

    KNOWN LIMITATION (WideEP / DeepEP, out of scope): this binary
    `trtllm -> TrtllmAlltoall, else -> CustomAllReduce` mapping is correct for
    the standard (non-WideEP) MoE path that the gated parity harness
    (`test_engine_step_parity.py`) and `cli_estimate` exercise. It does NOT emit
    the SGLang-WideEP `DeepEpNormal` / `DeepEpLowLatency` flavors — that DeepEP
    path is out-of-scope scan territory, not gated by the smoke harness. A future
    WideEP stage must thread the WideEP/DeepEP flavor selection through here
    (and likely needs the resolved dispatch kind off the Python op rather than a
    pure backend-string mapping).
    """
    return "TrtllmAlltoall" if backend == "trtllm" else "CustomAllReduce"


def _moe_dispatch(op: MoEDispatch, *, backend: str) -> dict:
    # Python `MoEDispatch._quant_mode` may be None (kwarg); Rust types it as a
    # required MoeQuantMode. Default to bfloat16 (the neutral value) when unset,
    # matching the Rust builders which pass the model's moe quant mode.
    quant = op._quant_mode
    return {
        "name": op._name,
        "scale_factor": op._scale_factor,
        "hidden_size": op._hidden_size,
        "topk": op._topk,
        "num_experts": op._num_experts,
        "moe_tp_size": op._moe_tp_size,
        "moe_ep_size": op._moe_ep_size,
        "attention_dp_size": op._attention_dp_size,
        "pre_dispatch": op._pre_dispatch,
        "backend": backend,
        "flavor": _dispatch_flavor(backend),
        "comm_quant": "half",
        "moe_quant": _quant_name(quant) if quant is not None else "bfloat16",
        "attn_cp_size": op._attn_cp_size,
        "is_context": op._is_context,
        # DeepEP-normal dispatch SM count; the Rust table keys the sms axis
        # and snaps off-grid values (mirrors `_query_wideep_deepep_normal_table`).
        "sms": op._sms,
    }


def _custom_all_reduce(op: CustomAllReduce) -> dict:
    return {
        "name": op._name,
        "scale_factor": op._scale_factor,
        "hidden_size": op._h,
        "tp_size": op._tp_size,
        "quant": "half",
        "seq_split": op._seq_split,
    }


def _nccl(op: NCCL) -> dict:
    return {
        "name": op._name,
        "scale_factor": op._scale_factor,
        "hidden_size": op._num_elements_per_token,
        "num_gpus": op._num_gpus,
        "dtype": "half",
        "operation": op._nccl_op,
        "seq_split": op._seq_split,
    }


def _p2p(op: P2P) -> dict:
    return {
        "name": op._name,
        "scale_factor": op._scale_factor,
        "pp_size": op._pp_size,
        "hidden_size": op._h,
        "seq_split": op._seq_split,
    }


def _dsa_module(op: ContextDSAModule | GenerationDSAModule, *, architecture: str) -> dict:
    # GenerationDSAModule stores `_kv_cache_dtype`; ContextDSAModule stores
    # `_kvcache_quant_mode` + `_fmha_quant_mode`. The Rust `DsaModuleOp` carries
    # both quant fields for either phase; fill the missing one with bfloat16
    # (generation has no separate fmha mode in Python).
    kv = getattr(op, "_kvcache_quant_mode", None) or getattr(op, "_kv_cache_dtype", None)
    fmha = getattr(op, "_fmha_quant_mode", None)
    arch = op._architecture or architecture
    # `index_topk` is the top-k boundary used by the context piecewise / robust
    # 3-D dispatch. Python sources it from `DSA_MODEL_DIMS[arch]` (defaulting to
    # the default DSA architecture); mirror that so the Rust op-spec carries the
    # same boundary. Field name MUST match the Rust `DsaModuleOp.index_topk`.
    dims = DSA_MODEL_DIMS.get(arch, DSA_MODEL_DIMS[DEFAULT_DSA_ARCHITECTURE])
    return {
        "name": op._name,
        "scale_factor": op._scale_factor,
        "num_heads": op._num_heads,
        "cp_size": getattr(op, "_cp_size", 1) or 1,
        "kv_cache_dtype": _quant_name(kv),
        "fmha_quant_mode": _quant_name(fmha) if fmha is not None else "bfloat16",
        "gemm_quant_mode": _quant_name(op._gemm_quant_mode),
        "architecture": arch,
        "index_topk": int(dims["index_topk"]),
    }


def _attn_kind_for_ratio(ratio: int) -> str:
    """Mirror Rust `models/deepseek_v4.rs::attn_kind_for_ratio`:
    compress_ratio 4 -> Csa, {0, 128} -> Hca. `AttnKind` has no serde rename.
    """
    return "Csa" if ratio == 4 else "Hca"


def _dsv4_module(
    op: ContextDeepSeekV4AttentionModule | GenerationDeepSeekV4AttentionModule,
    *,
    architecture: str,
) -> dict:
    kv = getattr(op, "_kvcache_quant_mode", None) or getattr(op, "_kv_cache_dtype", None)
    fmha = getattr(op, "_fmha_quant_mode", None)
    return {
        "name": op._name,
        "scale_factor": op._scale_factor,
        "attn_kind": _attn_kind_for_ratio(op._compress_ratio),
        "num_heads": op._num_heads,
        "cp_size": getattr(op, "_cp_size", 1) or 1,
        "window_size": getattr(op, "_window_size", None),
        # native_heads (model total head count) selects the table slice;
        # tp_size is the table's primary interpolation axis. See
        # `perf_database::dsv4` / `load_context_dsv4_kind_module_data`.
        "native_heads": op._native_heads,
        "tp_size": op._tp_size,
        "kv_cache_dtype": _quant_name(kv),
        "fmha_quant_mode": _quant_name(fmha) if fmha is not None else "bfloat16",
        "gemm_quant_mode": _quant_name(op._gemm_quant_mode),
        "architecture": architecture,
        # Structural dims for the Rust-side analytic SOL (beyond-grid
        # util-hold ratios). Python's op carries them from the model config
        # (`_deepseek_v4_attention_sol` inputs); without them the Rust side
        # would pin DeepSeek-V4-Pro dims and Flash's ratios would drift.
        "hidden_size": op._hidden_size,
        "q_lora_rank": op._q_lora_rank,
        "o_lora_rank": op._o_lora_rank,
        "head_dim": op._head_dim,
        "rope_head_dim": op._rope_head_dim,
        "index_n_heads": op._index_n_heads,
        "index_head_dim": op._index_head_dim,
        "index_topk": op._index_topk,
        # Rank-LOCAL o_groups (the model pre-divides by tp).
        "o_groups": op._o_groups,
    }


def _mhc_module(op: DeepSeekV4MHCModule, *, architecture: str) -> dict:
    return {
        "name": op._name,
        "scale_factor": op._scale_factor,
        # op (pre/post/both) is part of the mHC table key — pre and post have
        # distinct latencies. See `perf_database::mhc` / `_query_mhc_table`.
        "op": op._op,
        "hc_mult": op._hc_mult,
        "hidden_size": op._hidden_size,
        "architecture": architecture,
        # SOL inputs for the Rust-side mHC roofline (beyond-range util-hold
        # anchor; mirrors `_query_mhc_table.get_sol`).
        "sinkhorn_iters": op._sinkhorn_iters,
        "quant_mode": _quant_name(op._quant_mode),
    }


def _mamba2(op: Mamba2Kernel) -> dict:
    # Rust `Mamba2Op.d_model` == hidden_size (the Rust builders set `d_model: h`).
    return {
        "name": op._name,
        "scale_factor": op._scale_factor,
        "kernel_source": op._kernel_source,
        "phase": op._phase,
        "d_model": op._hidden_size,
        "d_state": op._d_state,
        "d_conv": op._d_conv,
        "nheads": op._nheads,
        "head_dim": op._head_dim,
        "n_groups": op._n_groups,
        "chunk_size": op._chunk_size,
    }


def _gdn(op: GDNKernel) -> dict:
    return {
        "name": op._name,
        "scale_factor": op._scale_factor,
        "kernel_source": op._kernel_source,
        "phase": op._phase,
        "d_model": op._d_model,
        "d_conv": op._d_conv,
        "num_k_heads": op._num_k_heads,
        "head_k_dim": op._head_k_dim,
        "num_v_heads": op._num_v_heads,
        "head_v_dim": op._head_v_dim,
    }


def _wideep_context_mla(op: WideEPContextMLA) -> dict:
    return {
        "name": op._name,
        "scale_factor": op._scale_factor,
        "num_heads": op._tp_size,  # Rust num_heads slot carries the per-rank head split
        "kv_cache_dtype": _quant_name(op._kvcache_quant_mode),
        "fmha_quant_mode": _quant_name(op._fmha_quant_mode),
        "attn_backend": op._attn_backend,
        "cp_size": op._cp_size,
    }


def _wideep_generation_mla(op: WideEPGenerationMLA) -> dict:
    return {
        "name": op._name,
        "scale_factor": op._scale_factor,
        "num_heads": op._tp_size,
        "kv_cache_dtype": _quant_name(op._kvcache_quant_mode),
        "fmha_quant_mode": _quant_name(op._fmha_quant_mode),
        "attn_backend": op._attn_backend,
    }


def _wideep_moe(op: TrtLLMWideEPMoE, *, database: Any) -> dict:
    # `kernel_source` is resolved in Python `_select_kernel(database, quant)` at
    # query time. Pre-bake the resolved value so the Rust side doesn't rely on
    # its hardcoded default + table-presence fallback.
    kernel_source = "moe_torch_flow"
    if database is not None:
        try:
            kernel_source = TrtLLMWideEPMoE._select_kernel(database, op._quant_mode)
        except Exception:
            kernel_source = "moe_torch_flow"
    return {
        "name": op._name,
        "scale_factor": op._scale_factor,
        "hidden_size": op._hidden_size,
        "inter_size": op._inter_size,
        "topk": op._topk,
        "num_experts": op._num_experts,
        "moe_tp_size": op._moe_tp_size,
        "moe_ep_size": op._moe_ep_size,
        "attention_dp_size": op._attention_dp_size,
        "quant_mode": _quant_name(op._quant_mode),
        "workload_distribution": op._workload_distribution,
        "num_slots": op._num_slots,
        "kernel_source": kernel_source,
    }


def _to_opspec(op: Any, *, backend: str, architecture: str, database: Any) -> dict:
    """Convert one Python ``Operation`` to its externally-tagged ``OpSpec`` dict.

    Dispatches on the concrete Python class. Phase-pair variants
    (``MLAModule`` -> ``MlaModuleContext`` / ``MlaModuleGeneration``) and the
    recursive composites (``OverlapOp`` / ``FallbackOp``) are handled here.
    """

    def recurse(child: Any) -> dict:
        return _to_opspec(child, backend=backend, architecture=architecture, database=database)

    # Recursive composites first.
    if isinstance(op, OverlapOp):
        return {
            "Overlap": {
                "name": op._name,
                "group_a": [recurse(c) for c in op._group_a],
                "group_b": [recurse(c) for c in op._group_b],
            }
        }
    if isinstance(op, FallbackOp):
        return {
            "Fallback": {
                "name": op._name,
                "primary": recurse(op._primary),
                "fallback": [recurse(c) for c in op._fallback],
            }
        }

    # MLAModule: one class, two Rust variants by phase.
    if isinstance(op, MLAModule):
        tag = "MlaModuleContext" if op._is_context else "MlaModuleGeneration"
        return {tag: _mla_module(op)}

    # DSA / DSv4: separate Python classes per phase.
    if isinstance(op, ContextDSAModule):
        return {"DsaContext": _dsa_module(op, architecture=architecture)}
    if isinstance(op, GenerationDSAModule):
        return {"DsaGeneration": _dsa_module(op, architecture=architecture)}
    if isinstance(op, ContextDeepSeekV4AttentionModule):
        return {"Dsv4Context": _dsv4_module(op, architecture=architecture)}
    if isinstance(op, GenerationDeepSeekV4AttentionModule):
        return {"Dsv4Generation": _dsv4_module(op, architecture=architecture)}

    # WideEP variants (must precede their non-WideEP base classes if any).
    if isinstance(op, WideEPContextMLA):
        return {"WideEpContextMla": _wideep_context_mla(op)}
    if isinstance(op, WideEPGenerationMLA):
        return {"WideEpGenerationMla": _wideep_generation_mla(op)}
    if isinstance(op, TrtLLMWideEPMoE):
        return {"WideEpMoe": _wideep_moe(op, database=database)}

    # Plain (single-variant) ops.
    if isinstance(op, GEMM):
        return {"Gemm": _gemm(op)}
    if isinstance(op, Embedding):
        return {"Embedding": _embedding(op)}
    if isinstance(op, ElementWise):
        return {"Elementwise": _elementwise(op)}
    if isinstance(op, ContextAttention):
        return {"ContextAttention": _context_attention(op)}
    if isinstance(op, GenerationAttention):
        return {"GenerationAttention": _generation_attention(op)}
    if isinstance(op, EncoderAttention):
        return {"EncoderAttention": _encoder_attention(op)}
    if isinstance(op, ContextMLA):
        return {"ContextMla": _context_mla(op)}
    if isinstance(op, GenerationMLA):
        return {"GenerationMla": _generation_mla(op)}
    if isinstance(op, MLABmm):
        return {"MlaBmm": _mla_bmm(op)}
    if isinstance(op, MoEDispatch):
        return {"MoeDispatch": _moe_dispatch(op, backend=backend)}
    if isinstance(op, MoE):
        return {"Moe": _moe(op)}
    if isinstance(op, CustomAllReduce):
        return {"CustomAllReduce": _custom_all_reduce(op)}
    if isinstance(op, NCCL):
        return {"Nccl": _nccl(op)}
    if isinstance(op, P2P):
        return {"P2P": _p2p(op)}
    if isinstance(op, DeepSeekV4MHCModule):
        return {"Mhc": _mhc_module(op, architecture=architecture)}
    if isinstance(op, Mamba2Kernel):
        return {"Mamba2": _mamba2(op)}
    if isinstance(op, GDNKernel):
        return {"Gdn": _gdn(op)}

    raise OpConversionError(
        f"no OpSpec conversion for {type(op).__module__}.{type(op).__name__} (op name={getattr(op, '_name', '?')!r})"
    )


# --------------------------------------------------------------------------- #
# EngineConfig assembly.
# --------------------------------------------------------------------------- #


def _compute_perf_db_sources(database: Any) -> dict:
    """Resolve the shared-layer (sibling/cross-version) source list per op file
    from the Python ``database``, so the Rust core can load the SAME rows Python
    does under SILICON/HYBRID.

    Returns ``{op_file_basename: [[abs_path, [kernel_sources] | None], ...]}``.
    ``_build_op_sources`` returns just ``[(primary, None)]`` when the shared
    layer is off or an op has no inheritable siblings, so the Rust side falls
    back to its primary ``data_root`` behaviour for those. Returns ``{}`` when a
    database is unavailable or introspection fails (Rust then uses its
    single-``data_root`` default). Discovery stays here (single source of truth)
    rather than being reimplemented in Rust.
    """
    if database is None:
        return {}
    try:
        from aiconfigurator_core.sdk.common import PerfDataFilename
        from aiconfigurator_core.sdk.operations.base import resolve_op_data_path

        system_data_root = os.path.join(database.systems_root, database.system_spec["data_dir"])
        out: dict[str, list] = {}
        for filename_enum in PerfDataFilename:
            primary_path = resolve_op_data_path(
                system_data_root, database.backend, database.version, filename_enum.value
            )
            sources = database._build_op_sources(filename_enum, primary_path, system_data_root)
            out[filename_enum.value] = [
                [os.path.abspath(path), (sorted(ks) if ks is not None else None)] for path, ks in sources
            ]
        return out
    except Exception:
        return {}


def _engine_config_dict(
    *,
    model: Any,
    model_path: str,
    system: str,
    backend: str,
    backend_version: str | None,
    kv_block_size: int | None,
    systems_path: str | None,
    nextn: int,
    database: Any = None,
) -> dict:
    """Build the ``EngineConfig`` JSON (matches the Rust modularised struct).

    The quant/parallel fields are read off the resolved ``model.config`` so the
    compiled engine identity reflects the quant inference that happened inside
    ``get_model``. The Rust ``EngineConfig`` only uses ``speculative.nextn``
    for latency (decode-batch scaling) and the model/system/backend/parallel
    fields to locate the perf database; the rest are carried for completeness.
    """
    cfg = model.config
    model_nextn = getattr(model, "_nextn", None)
    effective_nextn = int(model_nextn) if model_nextn is not None else int(nextn)
    speculative = None
    if effective_nextn:
        speculative = {"nextn": effective_nextn}
    # The Rust ``EngineConfig`` flattens ``parallel`` / ``quantization`` /
    # ``speculative`` via ``#[serde(flatten)]``, so their fields live at the
    # TOP LEVEL of this dict (NOT nested under sub-keys).
    engine: dict[str, Any] = {
        "schema_version": ENGINE_CONFIG_SCHEMA_VERSION,
        "model_name": model_path,
        "system_name": system,
        "systems_path": systems_path,
        "backend": backend,
        "backend_version": backend_version,
        "kv_block_size": kv_block_size,
        # ParallelMapping (flattened)
        "tp_size": int(cfg.tp_size or 1),
        "pp_size": int(cfg.pp_size or 1),
        "attention_dp_size": _opt_int(getattr(cfg, "attention_dp_size", None)),
        "moe_tp_size": _opt_int(getattr(cfg, "moe_tp_size", None)),
        "moe_ep_size": _opt_int(getattr(cfg, "moe_ep_size", None)),
        "cp_size": _opt_int(getattr(cfg, "cp_size", None)),
        # QuantizationConfig (flattened)
        "weight_dtype": _rust_quant_to_dtype(getattr(cfg, "gemm_quant_mode", None)),
        "moe_dtype": _rust_moe_quant_to_dtype(getattr(cfg, "moe_quant_mode", None)),
        "activation_dtype": _rust_quant_to_dtype(getattr(cfg, "fmha_quant_mode", None)),
        "kv_cache_dtype": _rust_quant_to_dtype(getattr(cfg, "kvcache_quant_mode", None)),
        # Shared-layer (sibling/cross-version) per-op perf-data sources, resolved
        # in Python so the Rust core inherits the same rows. Empty/absent = Rust
        # uses its single-``data_root`` default (back-compat with old specs).
        "perf_db_sources": _compute_perf_db_sources(database),
        "extra": {},
    }
    # SpeculativeConfig (flattened, Option<>): emit nextn at the top level
    # when MTP is active. When inactive, omit it so the
    # flattened Option deserializes to None.
    if speculative is not None:
        engine["nextn"] = speculative["nextn"]
    return engine


def _opt_int(value: Any) -> int | None:
    return None if value is None else int(value)


# --------------------------------------------------------------------------- #
# Public entry points.
# --------------------------------------------------------------------------- #


def compile_engine(
    model_path: str,
    system: str,
    backend: str,
    backend_version: str | None = None,
    *,
    tp_size: int = 1,
    pp_size: int = 1,
    attention_dp_size: int = 1,
    moe_tp_size: int | None = None,
    moe_ep_size: int | None = None,
    gemm_quant_mode: str | None = None,
    moe_quant_mode: str | None = None,
    kvcache_quant_mode: str | None = None,
    fmha_quant_mode: str | None = None,
    comm_quant_mode: str | None = None,
    nextn: int = 0,
    kv_block_size: int | None = None,
    systems_path: str | None = None,
) -> bytes:
    """Compile a model into bincoded ``EngineSpec`` bytes.

    Signature matches the kwargs ``build_aic_engine`` (Rust) passes. Reuses
    ``cli/api._build_model_config`` + ``sdk/models.get_model`` (quant inferred
    inside ``get_model``) to build the model, then walks ``encoder_ops`` (vision
    decomposed), ``context_ops`` and ``generation_ops`` into OpSpecs and returns
    the bytes produced by the Rust ``engine_spec_bincode_from_json`` pyfunction.
    """
    # `_build_model_config` resolves MoE parallelism defaults internally and
    # does not take a model_path (quant inference is done inside `get_model`).
    resolved_moe_tp = moe_tp_size if moe_tp_size is not None else 1
    resolved_moe_ep = moe_ep_size if moe_ep_size is not None else 1
    model_config = build_model_config(
        tp_size=tp_size,
        pp_size=pp_size,
        attention_dp_size=attention_dp_size,
        moe_tp_size=resolved_moe_tp,
        moe_ep_size=resolved_moe_ep,
        gemm_quant_mode=gemm_quant_mode,
        kvcache_quant_mode=kvcache_quant_mode,
        fmha_quant_mode=fmha_quant_mode,
        moe_quant_mode=moe_quant_mode,
        comm_quant_mode=comm_quant_mode,
    )
    # Apply MTP BEFORE get_model so the walked op lists carry the
    # (L+nextn)/L compute scale; accepted-token progress is applied above core.
    apply_nextn(model_config, nextn)
    model = get_model(model_path, model_config, backend)

    # The database is only needed to pre-bake the WideEP MoE kernel selection
    # (TRT-LLM only). Load lazily and tolerate failure; the converter falls back
    # to "moe_torch_flow" when it's unavailable.
    database = _maybe_load_database(system, backend, backend_version, systems_path)

    spec_json = build_engine_spec_json(
        model,
        model_path=model_path,
        system=system,
        backend=backend,
        backend_version=backend_version,
        kv_block_size=kv_block_size,
        systems_path=systems_path,
        nextn=model_config.nextn,
        database=database,
    )

    return bytes(aiconfigurator_core.engine_spec_bincode_from_json(spec_json))


def build_engine_spec_json(
    model: Any,
    *,
    model_path: str,
    system: str,
    backend: str,
    backend_version: str | None,
    kv_block_size: int | None,
    systems_path: str | None,
    nextn: int,
    database: Any = None,
) -> str:
    """Walk a built model's op lists into an ``EngineSpec`` JSON string.

    Separated from ``compile_engine`` so the op-transfer round-trip test can
    inspect the JSON (and the decoded ops) without going through bincode.
    """
    architecture = getattr(model, "architecture", "") or ""

    def conv(op: Any) -> dict:
        return _to_opspec(op, backend=backend, architecture=architecture, database=database)

    # Vision encoder ops are intentionally NOT emitted into the spec.
    #
    # The compile path threads no image configuration (num_images_per_request,
    # image_height/width, num_image_tokens), so the compiled engine cannot
    # reproduce `BaseBackend._run_encoder`'s token-count math (eff_batch, eff_s,
    # pre/post-merge counts) needed to query the vision ops with correct shapes.
    # Python's `run_static` already treats any request without image dimensions
    # as text-only and skips the encoder entirely (base_backend `_run_encoder`
    # early-return). Emitting the encoder ops here would make the compiled engine
    # query them unconditionally (with wrong shapes), diverging from the Python
    # reference for VL models. Vision modeling in the compiled path is deferred
    # until runtime image config is threaded through compile_engine.
    context_ops = [conv(op) for op in model.context_ops]
    generation_ops = [conv(op) for op in model.generation_ops]

    spec = {
        "schema_version": ENGINE_SPEC_SCHEMA_VERSION,
        "engine": _engine_config_dict(
            model=model,
            model_path=model_path,
            system=system,
            backend=backend,
            backend_version=backend_version,
            kv_block_size=kv_block_size,
            systems_path=systems_path,
            nextn=nextn,
            database=database,
        ),
        "context_ops": context_ops,
        "generation_ops": generation_ops,
    }
    return json.dumps(spec)


def _maybe_load_database(system: str, backend: str, backend_version: str | None, systems_path: str | None) -> Any:
    try:
        from aiconfigurator_core.sdk import perf_database

        return perf_database.get_database(system, backend, backend_version, systems_paths=systems_path)
    except Exception:
        return None


class EngineHandle:
    """Python wrapper over compiled ``EngineSpec`` bytes + a Rust ``AicEngine``.

    Exposes the per-call surface that shells through the PyO3 ``AicEngine``
    methods. The agg sweep is orchestrated in Python (mix/genonly step counting
    lives in ``base_backend``), so there is no ``run_agg`` here.
    """

    def __init__(self, spec_bytes: bytes, *, systems_path: str | None = None) -> None:
        self._bytes = bytes(spec_bytes)
        self._systems_path = systems_path
        self._engine = aiconfigurator_core.AicEngine.from_spec(self._bytes, systems_path)

    @classmethod
    def compile(cls, model_path: str, system: str, backend: str, **kwargs: Any) -> EngineHandle:
        """Compile + wrap in one call. ``systems_path`` is forwarded to both."""
        systems_path = kwargs.get("systems_path")
        spec_bytes = compile_engine(model_path, system, backend, **kwargs)
        return cls(spec_bytes, systems_path=systems_path)

    @property
    def spec_bytes(self) -> bytes:
        return self._bytes

    def run_static(
        self,
        *,
        batch_size: int,
        isl: int,
        osl: int,
        prefix: int = 0,
        beam_width: int = 1,
        seq_imbalance_correction_scale: float = 1.0,
        gen_seq_imbalance_correction_scale: float = 1.0,
        mode: str = "static",
        stride: int = 32,
    ) -> tuple[float, float, float]:
        """Return ``(context_ms, generation_ms, total_ms)``."""
        return self._engine.run_static(
            int(batch_size),
            int(beam_width),
            int(isl),
            int(osl),
            int(prefix),
            float(seq_imbalance_correction_scale),
            float(gen_seq_imbalance_correction_scale),
            mode,
            int(stride),
        )

    def predict_prefill_latency(self, bs: int, isl: int, prefix: int = 0) -> float:
        return self._engine.predict_prefill_latency(int(bs), int(isl), int(prefix))

    def predict_decode_latency(self, bs: int, isl: int, osl: int = 2) -> float:
        return self._engine.predict_decode_latency(int(bs), int(isl), int(osl))

    def mixed_step_latency(self, ctx_tokens: int, gen_tokens: int, isl: int, osl: int, prefix: int = 0) -> float:
        return self._engine.mixed_step_latency(int(ctx_tokens), int(gen_tokens), int(isl), int(osl), int(prefix))

    def mixed_step_breakdown(
        self,
        ctx_tokens: int,
        gen_tokens: int,
        isl: int,
        osl: int,
        prefix: int = 0,
    ) -> tuple[float, float, float, float]:
        """Return total/shared-non-attn/context-attn/decode-attn latency."""
        return self._engine.mixed_step_breakdown(
            int(ctx_tokens),
            int(gen_tokens),
            int(isl),
            int(osl),
            int(prefix),
        )

    def decode_step_latency(self, gen_tokens: int, isl: int, osl: int) -> float:
        return self._engine.decode_step_latency(int(gen_tokens), int(isl), int(osl))
