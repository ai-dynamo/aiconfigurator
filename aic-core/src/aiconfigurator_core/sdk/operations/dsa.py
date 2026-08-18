# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DSA (DeepSeek Sparse Attention) module-level ops (ISSUE-10 / AIC-538).

Both ContextDSAModule and GenerationDSAModule own their CSV-backed perf
tables and grid extrapolation. ``PerfDatabase.query_context_dsa_module``
and ``query_generation_dsa_module`` delegate here.

Both classes still bind a ``_raw_data_cache`` for backward compatibility,
but with load-time pre-expansion removed the table IS the raw measurements,
so it is a plain alias. (The PR #903 topk-piecewise lookup and the hand-rolled
boundary-util anchoring it served are superseded by the engine's interpolation: linear
bracket blends cannot overshoot the topk knee the way cubic did, and
util-hold is native.)

No SOL clamping in the legacy ``_correct_data`` for either DSA op —
extrapolation only. The legacy ``__init__`` loaded DSA twice (once near
the MLA/Mamba block, once after); both loads are consolidated into a
single ``load_data`` call per class.

The DSA-specific helper ``_format_dsa_unavailable_message`` also moves
here as a module-level function. ``DSA_MODEL_DIMS`` and ``DEFAULT_DSA_ARCHITECTURE`` stay on
``perf_database.py`` as module-level constants for now — the cleanup PR
revisits their home.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

import aiconfigurator_core._aiconfigurator_core as _core
from aiconfigurator_core.sdk.operations.base import OpShellKit

if TYPE_CHECKING:
    from aiconfigurator_core.sdk.perf_database import PerfDatabase

logger = logging.getLogger(__name__)


# kernel_source -> configured-backend bucket(s) for FP8-KV rows. 0.5.14
# SGLang DSA collectors record the EXECUTED kernel; dense ragged prefill is
# selected by SHAPE (isl <= 2048) under either configured backend, so its
# rows back both buckets.

DSA_MODEL_DIMS: dict[str, dict] = {
    "DeepseekV32ForCausalLM": {
        "hidden_size": 7168,
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "index_topk": 2048,
        "index_head_dim": 128,
        "index_n_heads": 64,
    },
    "GlmMoeDsaForCausalLM": {
        "hidden_size": 6144,
        "q_lora_rank": 2048,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 192,
        "qk_rope_head_dim": 64,
        "v_head_dim": 256,
        "index_topk": 2048,
        "index_head_dim": 128,
        "index_n_heads": 32,
    },
}

DEFAULT_DSA_ARCHITECTURE = "DeepseekV32ForCausalLM"

_DSA_PROJECTION_GROUPS = ("q", "kv", "o", "indexer")


def _normalize_projection_quant_modes(overrides, gemm_quant_mode) -> dict:
    """All four projection groups, missing ones filled from gemm_quant_mode.

    The opspec emission (and the Rust ``DsaProjectionQuants``
    deserialization, which requires every field) never sees an incomplete
    map, and an unknown group name fails loudly instead of being silently
    dropped by the engine."""
    overrides = overrides or {}
    unknown = sorted(set(overrides) - set(_DSA_PROJECTION_GROUPS))
    if unknown:
        raise ValueError(f"unknown DSA projection group(s) {unknown}; expected a subset of {_DSA_PROJECTION_GROUPS}")
    return {**dict.fromkeys(_DSA_PROJECTION_GROUPS, gemm_quant_mode), **overrides}


# Extrapolation grids — lifted verbatim from the legacy blocks in
# ``PerfDatabase.__init__``.

# fmt: on


def _cache_key(database: PerfDatabase) -> tuple:
    """Shared cache key — same shape as GEMM, Attention, and Communication.

    Still local to ``operations/dsa.py`` (Phase 3 has 5 duplicate copies
    so far); the cleanup PR hoists this to ``operations/base.py`` once
    Phase 3 settles.
    """
    return (
        database.systems_root,
        database.system,
        database.backend,
        database.version,
        database.enable_shared_layer,
    )


class ContextDSAModule(_core.ContextDSAModule, OpShellKit):
    """
    Context phase DSA (DeepSeek Sparse Attention) module-level operation.

    Owns ``_data_cache`` (extrapolated context_dsa_module CSV) AND
    ``_raw_data_cache`` (the same CSV pre-extrapolation, used by the
    topk-boundary piecewise interpolation path).

    Models the full DSA attention block including:
    - kv_a_proj_with_mqa GEMM (includes indexer K projection)
    - LayerNorm + q_b_proj GEMM
    - Indexer: wq_b GEMM, weights_proj GEMM, FP8 MQA logits, TopK selection
    - Sparse MLA attention (attends to top-k tokens instead of full sequence)
    - BMM pre/post (weight absorption + V projection)
    - o_proj GEMM
    """

    _data_cache: ClassVar[dict] = {}
    _raw_data_cache: ClassVar[dict] = {}
    _skip_data_cache: ClassVar[dict] = {}
    _raw_skip_data_cache: ClassVar[dict] = {}

    # ------------------------------------------------------------------
    # Data ownership
    # ------------------------------------------------------------------

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Fetches the engine's context_dsa_module table view
        (full + skip_indexer splits), binds
        ``database._context_dsa_module_data`` and
        ``database._raw_context_dsa_module_data``."""
        from aiconfigurator_core.sdk.engine_table_view import fetch_table_view, load_view
        from aiconfigurator_core.sdk.perf_database import LoadedOpData, PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            cls._data_cache[key] = load_view(database, "_context_dsa_module_data", PerfDataFilename.dsa_context_module)
            # The raw wrapper stays a plain alias of the table (no load-time
            # grid pre-expansion since PR-5).
            cls._raw_data_cache[key] = cls._data_cache[key]
            cls._record_load()

        # skip_indexer (GLM-5.2) rows live in the SAME file, tagged by op_name
        # (dsa_context_module_skip_indexer); the engine view splits them out.
        # Empty (no skip rows -> DeepSeek-V3.2 / GLM-5 freq==1) =>
        # slot None and the skip query path is never taken.
        if key not in cls._skip_data_cache:
            skip_view = fetch_table_view(database, "_context_dsa_module_skip_data")
            if skip_view:
                cls._skip_data_cache[key] = LoadedOpData(
                    skip_view, PerfDataFilename.dsa_context_module, cls._data_cache[key].filepath
                )
                cls._raw_skip_data_cache[key] = cls._skip_data_cache[key]
            else:
                cls._skip_data_cache[key] = None
                cls._raw_skip_data_cache[key] = None

        if "_context_dsa_module_data" not in database.__dict__:
            database._context_dsa_module_data = cls._data_cache[key]
        if "_raw_context_dsa_module_data" not in database.__dict__:
            database._raw_context_dsa_module_data = cls._raw_data_cache[key]
        if "_context_dsa_module_skip_data" not in database.__dict__:
            database._context_dsa_module_skip_data = cls._skip_data_cache[key]
        if "_raw_context_dsa_module_skip_data" not in database.__dict__:
            database._raw_context_dsa_module_skip_data = cls._raw_skip_data_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._data_cache.clear()
        cls._raw_data_cache.clear()
        cls._skip_data_cache.clear()
        cls._raw_skip_data_cache.clear()

    # ------------------------------------------------------------------
    # Query table (formerly PerfDatabase.query_context_dsa_module)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Op contract
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Context-Parallel (CP) prefill model — GLM-5 DSA only.
    # See docs/CONTEXT_PARALLEL_DSA_MODELING.md. Per-card =
    #   base dsa_module(isl/cp, bf16-KV row)
    #   + mqa(isl/cp)*(cp-1)                          (mqa ∝ isl², xcp identity)
    #   - [topk_full(flat) - topk_full(top_last)]/cp  (topk ∝ full/cp; module is dummy/flat)
    #   + AG_KV + AG_LSE                              (the two small attention all-gathers)
    # AG_hidden + RS belong to the MoE comm (modeled by MoEDispatch), not here.
    # ------------------------------------------------------------------


class GenerationDSAModule(_core.GenerationDSAModule, OpShellKit):
    """
    Generation phase DSA (DeepSeek Sparse Attention) module-level operation.

    Owns both an extrapolated working cache and the original measured rows.
    The raw view supplies trustworthy boundary utilization when a sequence
    query falls outside a collected curve.

    Models the full DSA attention block during decode:
    - Same components as ContextDSAModule
    - Uses paged MQA logits for indexer
    - Sparse MLA with KV cache lookup
    """

    _data_cache: ClassVar[dict] = {}
    _raw_data_cache: ClassVar[dict] = {}
    _skip_data_cache: ClassVar[dict] = {}
    _raw_skip_data_cache: ClassVar[dict] = {}

    # ------------------------------------------------------------------
    # Data ownership
    # ------------------------------------------------------------------

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Fetches the engine's generation_dsa_module table view
        (full + skip_indexer splits) and binds both views on ``database``."""
        from aiconfigurator_core.sdk.engine_table_view import fetch_table_view, load_view
        from aiconfigurator_core.sdk.perf_database import LoadedOpData, PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            cls._data_cache[key] = load_view(
                database, "_generation_dsa_module_data", PerfDataFilename.dsa_generation_module
            )
            # The raw wrapper stays a plain alias of the table (no load-time
            # grid pre-expansion since PR-5).
            cls._raw_data_cache[key] = cls._data_cache[key]
            cls._record_load()

        # skip_indexer rows share the same file (op_name tag); the engine view
        # splits them out.
        if key not in cls._skip_data_cache:
            skip_view = fetch_table_view(database, "_generation_dsa_module_skip_data")
            if skip_view:
                cls._skip_data_cache[key] = LoadedOpData(
                    skip_view, PerfDataFilename.dsa_generation_module, cls._data_cache[key].filepath
                )
                cls._raw_skip_data_cache[key] = cls._skip_data_cache[key]
            else:
                cls._skip_data_cache[key] = None
                cls._raw_skip_data_cache[key] = None

        if "_generation_dsa_module_data" not in database.__dict__:
            database._generation_dsa_module_data = cls._data_cache[key]
        if "_raw_generation_dsa_module_data" not in database.__dict__:
            database._raw_generation_dsa_module_data = cls._raw_data_cache[key]
        if "_generation_dsa_module_skip_data" not in database.__dict__:
            database._generation_dsa_module_skip_data = cls._skip_data_cache[key]
        if "_raw_generation_dsa_module_skip_data" not in database.__dict__:
            database._raw_generation_dsa_module_skip_data = cls._raw_skip_data_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._data_cache.clear()
        cls._raw_data_cache.clear()
        cls._skip_data_cache.clear()
        cls._raw_skip_data_cache.clear()

    # ------------------------------------------------------------------
    # Query table (formerly PerfDatabase.query_generation_dsa_module)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Op contract
    # ------------------------------------------------------------------


# ─────────────────────────────────────────────────────────
# CSV loaders (moved here from perf_database.py so each op family owns its data + parser)
# ─────────────────────────────────────────────────────────
