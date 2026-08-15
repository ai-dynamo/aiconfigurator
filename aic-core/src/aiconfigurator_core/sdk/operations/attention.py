# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Context + Generation attention ops (ISSUE-06 / AIC-543).

Both classes bind the engine's table views, SOL correction (generation
only — context attention has no SOL clamp in the legacy
``_correct_data``), and grid extrapolation.
``PerfDatabase.query_context_attention`` / ``query_generation_attention``
delegate here.

``ContextAttention.query`` keeps its three ``query_mem_op`` callers
(QK-norm, apply-RoPE, KV-write) pointed at ``database.query_mem_op`` —
deciding a long-term home for the analytical mem-op formula is deferred
to the post-refactor cleanup.

Cache key is ``(systems_root, system, backend, version,
enable_shared_layer)``, same as GEMM (and every other migrated op).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk.operations.base import Operation

if TYPE_CHECKING:
    from aiconfigurator_core.sdk.perf_database import PerfDatabase

logger = logging.getLogger(__name__)

# Extrapolation target grids — lifted verbatim from the legacy blocks in
# ``PerfDatabase.__init__`` so behavior stays bit-identical.

# fmt: on


def _cache_key(database: PerfDatabase) -> tuple:
    """Shared cache key — same shape as GEMM's, used by both Attention ops.

    TODO: hoist to ``operations/base.py`` once a third op family (Phase 3
    NCCL / MLA / Mamba) lands and needs the same key shape — preferring
    duplication over premature abstraction with only two callers.
    """
    return (
        database.systems_root,
        database.system,
        database.backend,
        database.version,
        database.enable_shared_layer,
    )


class ContextAttention(Operation):
    """
    Context (prefill) attention operation.

    Owns ``_data_cache: {key: LoadedOpData}`` for the context attention CSV —
    raw as-collected rows (no load-time clamp or grid pre-expansion; the
    engine owns interpolation and the SOL floor).
    """

    _data_cache: ClassVar[dict] = {}

    def __init__(
        self,
        name: str,
        scale_factor: float,
        n: int,
        n_kv: int,
        kvcache_quant_mode: common.KVCacheQuantMode,
        fmha_quant_mode: common.FMHAQuantMode,
        window_size: int = 0,
        head_size: int = 128,
        use_qk_norm: bool = False,
        cp_size: int = 1,
    ) -> None:
        """Initialize context attention query parameters."""
        super().__init__(name, scale_factor)
        self._n = n
        self._n_kv = n_kv
        self._kvcache_quant_mode = kvcache_quant_mode
        self._fmha_quant_mode = fmha_quant_mode
        self._window_size = window_size
        self._head_size = head_size
        self._use_qk_norm = use_qk_norm
        # Context parallelism (sglang AllGather, zigzag in-seq split). When
        # cp_size > 1, query() models ONE representative CP rank (rank 0): the
        # sequence is split into 2*cp contiguous chunks, rank 0 owns chunk 0
        # (prefix=0) and chunk 2*cp-1 (prefix=isl-c). Its two halves' work sums
        # to the balanced per-rank total isl^2/(2*cp). See
        # docs/CONTEXT_PARALLEL_DSA_MODELING.md (dense analog).
        self._cp_size = cp_size

    # ------------------------------------------------------------------
    # Data ownership
    # ------------------------------------------------------------------

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Fetches the engine's context_attention table view
        (raw rows) and binds ``database._context_attention_data``,
        respecting any pre-set test override."""
        from aiconfigurator_core.sdk.engine_table_view import load_view
        from aiconfigurator_core.sdk.perf_database import PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            cls._data_cache[key] = load_view(database, "_context_attention_data", PerfDataFilename.context_attention)
            cls._record_load()

        # Bind instance attr (respect intentional test pre-overrides).
        if "_context_attention_data" not in database.__dict__:
            database._context_attention_data = cls._data_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._data_cache.clear()

    # ------------------------------------------------------------------
    # Query table (formerly PerfDatabase.query_context_attention)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Op contract: query() + get_weights()
    # ------------------------------------------------------------------

    _ENGINE_QUERY_SHAPE = "context"


class GenerationAttention(Operation):
    """
    Generation (decode) attention operation.

    Owns the SILICON row cache (raw as-collected; the load-time SOL clamp
    and grid expansion retired with #1357 PR-5 — the engine owns both) plus
    the raw-cache alias kept for its historical consumers.
    """

    _data_cache: ClassVar[dict] = {}
    _raw_data_cache: ClassVar[dict] = {}

    def __init__(
        self,
        name: str,
        scale_factor: float,
        n: int,
        n_kv: int,
        kv_cache_dtype: common.KVCacheQuantMode,
        window_size: int = 0,
        head_size: int = 128,
        use_qk_norm: bool = False,
    ) -> None:
        """Initialize generation attention query parameters."""
        super().__init__(name, scale_factor)
        self._n = n
        self._n_kv = n_kv
        self._kv_cache_dtype = kv_cache_dtype
        self._window_size = window_size
        self._head_size = head_size
        self._use_qk_norm = use_qk_norm

    # ------------------------------------------------------------------
    # Data ownership
    # ------------------------------------------------------------------

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Fetches the engine's generation_attention table view
        (raw rows) and binds both database views.

        Mirrors ``GEMM.load_data``: loading operates on the
        canonical class-cache value (passed explicitly), then the instance
        attr is bound, respecting any pre-set test override."""
        from aiconfigurator_core.sdk.engine_table_view import load_view
        from aiconfigurator_core.sdk.perf_database import PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            cls._data_cache[key] = load_view(
                database, "_generation_attention_data", PerfDataFilename.generation_attention
            )
            # The raw wrapper stays a plain alias of the table (no load-time
            # grid expansion since PR-5).
            cls._raw_data_cache[key] = cls._data_cache[key]
            cls._record_load()

        # Bind instance attr (respect intentional test pre-overrides).
        if "_generation_attention_data" not in database.__dict__:
            database._generation_attention_data = cls._data_cache[key]
            database._raw_generation_attention_data = cls._raw_data_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._data_cache.clear()
        cls._raw_data_cache.clear()

    # NOTE(#1357 PR-5): the load-time SOL clamp (`_correct_sol`) retired with
    # the Python query math. The loaded table is now the RAW collected data
    # plane (enumeration/charts); the compiled engine applies the same clamp
    # to its own load (see perf_database/gemm.rs), so QUERY values stay
    # SOL-floored via the single oracle.

    # ------------------------------------------------------------------
    # Query table (formerly PerfDatabase.query_generation_attention)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Op contract: query() + get_weights()
    # ------------------------------------------------------------------

    _ENGINE_QUERY_SHAPE = "generation"


class EncoderAttention(Operation):
    """
    Non-causal encoder attention: full N^2, MHA, no KV cache, optional partial RoPE.

    Used to model bidirectional encoders — ViT (vision), audio encoders, and any
    other omni-modal encoder where the kernel runs full N^2 attention without a
    causal mask and without writing a KV cache. The optional
    ``partial_rotary_factor`` accounts for partial-rotation RoPE variants such as
    Qwen3-VL (factor=0.5, rotating half of head_dim). Defaults to 0.0 (no RoPE),
    matching CLIP / SigLIP / Whisper; set to 0.5 / 1.0 only for RoPE encoders.

    Owns ``_data_cache: {key: LoadedOpData}`` for the encoder attention CSV.
    Schema is simpler than context attention: MHA only (no n_kv), no KV cache
    (no kvcache_quant_mode), no sliding window. No SOL clamp. Grid extrapolation
    resolves on the raw grid via the engine's interpolation.
    """

    _data_cache: ClassVar[dict] = {}

    def __init__(
        self,
        name: str,
        scale_factor: float,
        num_heads: int,
        head_size: int,
        fmha_quant_mode: common.FMHAQuantMode = common.FMHAQuantMode.bfloat16,
        partial_rotary_factor: float = 0.0,
    ) -> None:
        super().__init__(name, scale_factor)
        # Encoder kernels currently only have bfloat16 perf data;
        if fmha_quant_mode != common.FMHAQuantMode.bfloat16:
            raise ValueError(f"EncoderAttention only supports FMHAQuantMode.bfloat16, got {fmha_quant_mode}")
        if not 0.0 <= partial_rotary_factor <= 1.0:
            raise ValueError(f"partial_rotary_factor must be in [0.0, 1.0], got {partial_rotary_factor}")
        self._n = num_heads
        self._head_size = head_size
        self._fmha_quant_mode = fmha_quant_mode
        self._partial_rotary_factor = partial_rotary_factor

    # ------------------------------------------------------------------
    # Data ownership
    # ------------------------------------------------------------------

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Fetches the engine's encoder_attention table view
        (raw rows), binds ``database._encoder_attention_data``.
        """
        from aiconfigurator_core.sdk.engine_table_view import load_view
        from aiconfigurator_core.sdk.perf_database import PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            cls._data_cache[key] = load_view(database, "_encoder_attention_data", PerfDataFilename.encoder_attention)
            cls._record_load()

        # Bind instance attr (respect intentional test pre-overrides).
        if "_encoder_attention_data" not in database.__dict__:
            database._encoder_attention_data = cls._data_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._data_cache.clear()

    # ------------------------------------------------------------------
    # Query table (formerly PerfDatabase.query_encoder_attention)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Op contract: query() + get_weights()
    # ------------------------------------------------------------------

    _ENGINE_QUERY_SHAPE = "context"
