# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rehydration layer over the engine table-view FFI (PR-6, #1357 phase 3).

The compiled engine owns the data plane: ``AicEngine.table_view_json``
re-folds the raw parquet sources into the exact nested-dict shape the retired
Python ``load_*_data`` parsers produced (see the Rust twin,
``perf_database/table_view.rs``). What comes over the FFI is JSON, so every
key is a string; this module converts them back into the key TYPES the
loaders used — quant-mode enums, ints, and the mamba-family model-key tuples
— level by level, preserving the JSON document order (== the loaders'
insertion order, which chart legends consume positionally).

This is a types-only layer by design: any value math or row filtering belongs
in the Rust fold, never here (single-oracle rule).
"""

from __future__ import annotations

import json
from typing import Any, Callable

from aiconfigurator_core.sdk import common

KeyConverter = Callable[[str], Any]


def _enum(cls) -> KeyConverter:
    return lambda name: cls[name]


def _int(text: str) -> int:
    return int(text)


def _str(text: str) -> str:
    return text


def _int_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split("|"))


_GEMM_Q = _enum(common.GEMMQuantMode)
_MOE_Q = _enum(common.MoEQuantMode)
_FMHA_Q = _enum(common.FMHAQuantMode)
_KV_Q = _enum(common.KVCacheQuantMode)
_COMM_Q = _enum(common.CommQuantMode)

# Per-attribute key-converter sequences, one entry per nesting level, in the
# retired loader's layering order. Variable-depth families (the mamba family:
# 2-D context/verify vs 1-D generation) simply stop early — converters are
# looked up lazily per level, so a shallow branch never indexes past its
# depth. Leaves are recognized by their "latency" field, exactly like the
# baseline codec (never by depth).
VIEW_KEY_LAYERS: dict[str, tuple[KeyConverter, ...]] = {
    "_gemm_data": (_GEMM_Q, _int, _int, _int),
    "_compute_scale_data": (_GEMM_Q, _int, _int),
    "_scale_matrix_data": (_GEMM_Q, _int, _int),
    "_context_attention_data": (_FMHA_Q, _KV_Q, _int, _int, _int, _int, _int, _int),
    "_generation_attention_data": (_KV_Q, _int, _int, _int, _int, _int, _int),
    "_encoder_attention_data": (_FMHA_Q, _int, _int, _int, _int),
    "_context_mla_data": (_FMHA_Q, _KV_Q, _int, _int, _int),
    "_generation_mla_data": (_KV_Q, _int, _int, _int),
    "_mla_bmm_data": (_GEMM_Q, _str, _int, _int),
    "_context_mla_module_data": (_FMHA_Q, _KV_Q, _GEMM_Q, _int, _int, _int, _int),
    "_generation_mla_module_data": (_KV_Q, _GEMM_Q, _int, _int, _int, _int),
    "_wideep_context_mla_data": (_str, _FMHA_Q, _KV_Q, _int, _int, _int),
    "_wideep_generation_mla_data": (_str, _KV_Q, _int, _int, _int),
    "_moe_data": (_MOE_Q, _str, _int, _int, _int, _int, _int, _int, _int),
    "_moe_low_latency_data": (_MOE_Q, _str, _int, _int, _int, _int, _int, _int, _int),
    "_wideep_context_moe_data": (_MOE_Q, _str, _int, _int, _int, _int, _int, _int, _int),
    "_wideep_generation_moe_data": (_MOE_Q, _str, _int, _int, _int, _int, _int, _int, _int),
    "_wideep_deepep_normal_data": (_int, _int, _int, _int, _int, _int),
    "_wideep_deepep_ll_data": (_int, _int, _int, _int, _int),
    "_wideep_moe_compute_data": (_str, _MOE_Q, _str, _int, _int, _int, _int, _int, _int, _int, _int),
    "_trtllm_alltoall_data": (_str, _str, _MOE_Q, _int, _int, _int, _int, _int, _int),
    "_moe_a2a_data": (_str, _str, _str, _int, _int, _int, _int, _int, _int, _int),
    "_moe_ep_data": (_str, _MOE_Q, _str, _str, _int, _int, _int, _int, _int, _int, _int, _int),
    "_custom_allreduce_data": (_COMM_Q, _int, _str, _int),
    "_nccl_data": (_COMM_Q, _str, _int, _int),
    "_oneccl_data": (_COMM_Q, _str, _int, _int),
    "_context_dsa_module_data": (_FMHA_Q, _KV_Q, _GEMM_Q, _str, _str, _int, _int, _int, _int),
    "_context_dsa_module_skip_data": (_FMHA_Q, _KV_Q, _GEMM_Q, _str, _str, _int, _int, _int, _int),
    "_generation_dsa_module_data": (_KV_Q, _GEMM_Q, _str, _str, _int, _int, _int),
    "_generation_dsa_module_skip_data": (_KV_Q, _GEMM_Q, _str, _str, _int, _int, _int),
    "_mhc_module_data": (_str, _int, _int, _int),
    "_context_deepseek_v4_attention_module_data": (_FMHA_Q, _KV_Q, _GEMM_Q, _int, _int, _int, _int, _int, _int),
    "_generation_deepseek_v4_attention_module_data": (_KV_Q, _GEMM_Q, _int, _int, _int, _int, _int),
    "_dsv4_sparse_kernel_data.paged_mqa_logits": (_int, _int, _int, _int, _int),
    "_dsv4_sparse_kernel_data.hca_attn": (_int, _int, _int, _int, _int),
    "_dsv4_sparse_kernel_data.csa_attn": (_int, _int, _int, _int, _int),
    "_dsv4_megamoe_module_data": (
        _str,
        _str,
        _str,
        _MOE_Q,
        _str,
        _str,
        _str,
        _int,
        _int,
        _int,
        _int,
        _int,
        _int,
        _int,
        _int,
    ),
    "_mamba2_data": (_str, _str, _int_tuple, _int, _int),
    "_gdn_data": (_str, _str, _int_tuple, _int, _int),
    "_kda_data": (_str, _str, _int_tuple, _int, _int),
}


def _rehydrate(node: dict, layers: tuple[KeyConverter, ...], depth: int) -> dict:
    if "latency" in node:
        return node
    out: dict = {}
    for key, value in node.items():
        out[layers[depth](key)] = _rehydrate(value, layers, depth + 1)
    return out


def fetch_table_view(database, attribute: str):
    """Fetch one loader-shaped table from the engine, keys rehydrated.

    Returns ``None`` exactly when the retired Python loader returned ``None``
    (every source file missing). The engine handle is the cached probe handle
    for ``database`` — the same spec (and thus the same shared-layer source
    map) the query path uses.
    """
    from aiconfigurator_core.sdk import engine as _engine

    handle = _engine._probe_handle_for(database, None)
    raw = handle._engine.table_view_json(attribute)
    if raw is None:
        return None
    return _rehydrate(json.loads(raw), VIEW_KEY_LAYERS[attribute], 0)
