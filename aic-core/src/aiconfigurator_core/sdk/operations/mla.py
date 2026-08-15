# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MLA (Multi-head Latent Attention) family (ISSUE-08 / AIC-540).

Six op classes migrate from ``_legacy.py`` into ``operations/mla.py``:

- ``ContextMLA`` / ``GenerationMLA`` — regular MLA ops; own
  ``_context_mla_data`` / ``_generation_mla_data`` respectively. Both
  delegate to ``PerfDatabase.query_context_mla`` / ``query_generation_mla``
  which become one-line forwards.
- ``MLABmm`` — pre/post BMM op for MLA decoding. Owns ``_mla_bmm_data``.
- ``MLAModule`` — module-level MLA (both context and generation in one
  class, dispatched by ``is_context`` flag). Owns BOTH
  ``_context_mla_module_data`` AND ``_generation_mla_module_data`` since
  ``MLAModule.query`` chooses between them at runtime.
- ``WideEPContextMLA`` / ``WideEPGenerationMLA`` — SGLang-only variants.
  Their CSV tables are loaded only when ``backend == "sglang"`` (matching
  the legacy conditional ``if backend == "sglang"`` block in
  ``PerfDatabase.__init__``).

No SOL clamping for any MLA variant in the legacy ``_correct_data``.
Extrapolation present for all 4 regular + 2 module variants + 2 WideEP
variants (the WideEP variants extrapolate only when their data was
loaded — SGLang-only).

Cache key matches every other migrated op:
``(systems_root, system, backend, version, enable_shared_layer)``. For
WideEP variants, ``backend`` in the key naturally encodes the SGLang
constraint (cache misses on non-SGLang backends).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk.operations.base import Operation

if TYPE_CHECKING:
    from aiconfigurator_core.sdk.perf_database import PerfDatabase

logger = logging.getLogger(__name__)


def _cache_key(database: PerfDatabase) -> tuple:
    """Shared cache key — same shape as every other migrated op family.

    TODO: hoist to ``operations/base.py`` once Phase 3 settles (7 op
    families duplicating this helper now).
    """
    return (
        database.systems_root,
        database.system,
        database.backend,
        database.version,
        database.enable_shared_layer,
    )


# fmt: on


class ContextMLA(Operation):
    """
    Context MLA operation. Owns ``_context_mla_data``.
    """

    _data_cache: ClassVar[dict] = {}

    def __init__(
        self,
        name: str,
        scale_factor: float,
        num_heads: int,
        kvcache_quant_mode: common.KVCacheQuantMode,
        fmha_quant_mode: common.FMHAQuantMode,
        cp_size: int = 1,
    ) -> None:
        super().__init__(name, scale_factor)
        self._num_heads = num_heads
        self._kvcache_quant_mode = kvcache_quant_mode
        self._fmha_quant_mode = fmha_quant_mode
        # Context parallelism (sglang AllGather zigzag in-seq split). When cp>1,
        # query() models CP rank 0's two zigzag chunks (prev: prefix..+c; next:
        # prefix+isl-c..isl), same as ContextAttention. c = ceil(isl/(2*cp)).
        self._cp_size = cp_size

    # ------------------------------------------------------------------
    # Data ownership
    # ------------------------------------------------------------------

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Fetches the engine's context_mla table view, binds
        ``database._context_mla_data``."""
        from aiconfigurator_core.sdk.engine_table_view import load_view
        from aiconfigurator_core.sdk.perf_database import PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            cls._data_cache[key] = load_view(database, "_context_mla_data", PerfDataFilename.context_mla)
            cls._record_load()

        if "_context_mla_data" not in database.__dict__:
            database._context_mla_data = cls._data_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._data_cache.clear()

    # ------------------------------------------------------------------
    # Query table (formerly PerfDatabase.query_context_mla)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Op contract
    # ------------------------------------------------------------------

    _ENGINE_QUERY_SHAPE = "context"


class GenerationMLA(Operation):
    """
    Generation MLA operation (MQA part). Owns ``_generation_mla_data``.
    """

    _data_cache: ClassVar[dict] = {}

    def __init__(
        self,
        name: str,
        scale_factor: float,
        num_heads: int,
        kv_cache_dtype: common.KVCacheQuantMode,
    ) -> None:
        super().__init__(name, scale_factor)
        self._num_heads = num_heads
        self._kv_cache_dtype = kv_cache_dtype

    # ------------------------------------------------------------------
    # Data ownership
    # ------------------------------------------------------------------

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Fetches the engine's generation_mla table view, binds
        ``database._generation_mla_data``."""
        from aiconfigurator_core.sdk.engine_table_view import load_view
        from aiconfigurator_core.sdk.perf_database import PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            cls._data_cache[key] = load_view(database, "_generation_mla_data", PerfDataFilename.generation_mla)
            cls._record_load()

        if "_generation_mla_data" not in database.__dict__:
            database._generation_mla_data = cls._data_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._data_cache.clear()

    # ------------------------------------------------------------------
    # Query table (formerly PerfDatabase.query_generation_mla)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Op contract
    # ------------------------------------------------------------------

    _ENGINE_QUERY_SHAPE = "generation"


class MLABmm(Operation):
    """
    MLABmm operation — pre/post BMM for MLA decoding. Owns ``_mla_bmm_data``.
    No extrapolation in the legacy ``__init__`` path; data is 1D-keyed by
    num_tokens within each (quant_mode, op_name, num_heads) bucket.
    """

    _data_cache: ClassVar[dict] = {}

    def __init__(
        self,
        name: str,
        scale_factor: float,
        num_heads: int,
        quant_mode: common.GEMMQuantMode,
        if_pre: bool = True,
    ) -> None:
        super().__init__(name, scale_factor)
        self._num_heads = num_heads
        self._quant_mode = quant_mode
        self._if_pre = if_pre

    # ------------------------------------------------------------------
    # Data ownership
    # ------------------------------------------------------------------

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Fetches the engine's mla_bmm table view, binds
        ``database._mla_bmm_data``."""
        from aiconfigurator_core.sdk.engine_table_view import load_view
        from aiconfigurator_core.sdk.perf_database import PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            cls._data_cache[key] = load_view(database, "_mla_bmm_data", PerfDataFilename.mla_bmm)
            cls._record_load()

        if "_mla_bmm_data" not in database.__dict__:
            database._mla_bmm_data = cls._data_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._data_cache.clear()

    # ------------------------------------------------------------------
    # Op contract
    # ------------------------------------------------------------------

    _ENGINE_QUERY_SHAPE = "generation"

    def _engine_query_plan(self, kwargs: dict):
        """Legacy signature has no ``s``: the BMM shape is batch-only."""
        beam_width = kwargs.get("beam_width", 1)
        if beam_width != 1:
            raise ValueError(f"{type(self).__name__} only supports beam_width=1, got {beam_width}")
        batch_size = kwargs.get("batch_size")
        if batch_size is None:
            raise ValueError(f"{type(self).__name__}.query requires 'batch_size'.")
        return self, {
            "is_context": False,
            "batch_size": int(batch_size),
            "s": int(kwargs.get("s", 1) or 1),
        }


class MLAModule(Operation):
    """
    Module-level MLA op for both context and generation phases.

    Owns BOTH ``_context_mla_module_data`` (via ``_context_data_cache``)
    AND ``_generation_mla_module_data`` (via ``_generation_data_cache``)
    because ``query()`` chooses between them at runtime based on the
    ``is_context`` flag.

    Models the complete MLA attention block as a single profiled operation.
    For context: replaces q_b_proj + kv_b_proj + ContextMLA + proj.
    For generation: replaces MLABmm(pre) + GenerationMLA + MLABmm(post).
    """

    _context_data_cache: ClassVar[dict] = {}
    _generation_data_cache: ClassVar[dict] = {}

    def __init__(
        self,
        name: str,
        scale_factor: float,
        is_context: bool,
        num_heads: int,
        kvcache_quant_mode: common.KVCacheQuantMode,
        fmha_quant_mode: common.FMHAQuantMode,
        gemm_quant_mode: common.GEMMQuantMode,
        native_num_heads: int | None = None,
    ) -> None:
        super().__init__(name, scale_factor)
        self._is_context = is_context
        self._num_heads = num_heads
        # Model-native identity for the [native][local] module table (#1458).
        self._native_num_heads = native_num_heads
        self._kvcache_quant_mode = kvcache_quant_mode
        self._fmha_quant_mode = fmha_quant_mode
        self._gemm_quant_mode = gemm_quant_mode

    # ------------------------------------------------------------------
    # Data ownership — two tables, one per phase
    # ------------------------------------------------------------------

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Fetches BOTH the engine's context and generation
        module table views, binds ``database._context_mla_module_data`` and
        ``database._generation_mla_module_data``."""
        from aiconfigurator_core.sdk.engine_table_view import load_view
        from aiconfigurator_core.sdk.perf_database import PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._context_data_cache:
            cls._context_data_cache[key] = load_view(
                database, "_context_mla_module_data", PerfDataFilename.mla_context_module
            )
            cls._generation_data_cache[key] = load_view(
                database, "_generation_mla_module_data", PerfDataFilename.mla_generation_module
            )
            cls._record_load()

        if "_context_mla_module_data" not in database.__dict__:
            database._context_mla_module_data = cls._context_data_cache[key]
        if "_generation_mla_module_data" not in database.__dict__:
            database._generation_mla_module_data = cls._generation_data_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._context_data_cache.clear()
        cls._generation_data_cache.clear()

    # ------------------------------------------------------------------
    # Query tables (formerly PerfDatabase.query_context_mla_module /
    # query_generation_mla_module)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Op contract
    # ------------------------------------------------------------------

    _ENGINE_QUERY_SHAPE = "module"


class WideEPGenerationMLA(Operation):
    """
    WideEP Generation MLA operation (SGLang-only). Owns
    ``_wideep_generation_mla_data``. Loaded only when ``backend == "sglang"``.
    """

    _data_cache: ClassVar[dict] = {}

    def __init__(
        self,
        name: str,
        scale_factor: float,
        tp_size: int,
        kvcache_quant_mode: common.KVCacheQuantMode,
        fmha_quant_mode: common.FMHAQuantMode,
        attn_backend: str = "flashinfer",
    ) -> None:
        super().__init__(name, scale_factor)
        self._tp_size = tp_size
        self._kvcache_quant_mode = kvcache_quant_mode
        self._fmha_quant_mode = fmha_quant_mode
        self._attn_backend = attn_backend

    # ------------------------------------------------------------------
    # Data ownership
    # ------------------------------------------------------------------

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Fetches the engine's wideep_generation_mla table view
        (SGLang only), binds ``database._wideep_generation_mla_data``.

        Non-SGLang backends get ``None`` (matching the legacy
        ``if backend == "sglang"`` guard in ``__init__``)."""
        from aiconfigurator_core.sdk.engine_table_view import load_view
        from aiconfigurator_core.sdk.perf_database import PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            if database.backend != "sglang":
                cls._data_cache[key] = None
            else:
                cls._data_cache[key] = load_view(
                    database, "_wideep_generation_mla_data", PerfDataFilename.wideep_generation_mla
                )
            cls._record_load()

        if "_wideep_generation_mla_data" not in database.__dict__:
            database._wideep_generation_mla_data = cls._data_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._data_cache.clear()

    # ------------------------------------------------------------------
    # Query table (formerly PerfDatabase.query_wideep_generation_mla)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Op contract
    # ------------------------------------------------------------------

    _ENGINE_QUERY_SHAPE = "generation"


class WideEPContextMLA(Operation):
    """
    WideEP Context MLA operation (SGLang-only). Owns
    ``_wideep_context_mla_data``. Loaded only when ``backend == "sglang"``.
    """

    _data_cache: ClassVar[dict] = {}

    def __init__(
        self,
        name: str,
        scale_factor: float,
        tp_size: int,
        kvcache_quant_mode: common.KVCacheQuantMode,
        fmha_quant_mode: common.FMHAQuantMode,
        attn_backend: str = "flashinfer",
        cp_size: int = 1,
    ) -> None:
        super().__init__(name, scale_factor)
        self._tp_size = tp_size
        self._kvcache_quant_mode = kvcache_quant_mode
        self._fmha_quant_mode = fmha_quant_mode
        self._attn_backend = attn_backend
        # CP (sglang AllGather zigzag); see ContextMLA. cp>1 -> rank-0 two-chunk.
        self._cp_size = cp_size

    # ------------------------------------------------------------------
    # Data ownership
    # ------------------------------------------------------------------

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Fetches the engine's wideep_context_mla table view
        (SGLang only), binds ``database._wideep_context_mla_data``."""
        from aiconfigurator_core.sdk.engine_table_view import load_view
        from aiconfigurator_core.sdk.perf_database import PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            if database.backend != "sglang":
                cls._data_cache[key] = None
            else:
                cls._data_cache[key] = load_view(
                    database, "_wideep_context_mla_data", PerfDataFilename.wideep_context_mla
                )
            cls._record_load()

        if "_wideep_context_mla_data" not in database.__dict__:
            database._wideep_context_mla_data = cls._data_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._data_cache.clear()

    # ------------------------------------------------------------------
    # Query table (formerly PerfDatabase.query_wideep_context_mla)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Op contract
    # ------------------------------------------------------------------

    _ENGINE_QUERY_SHAPE = "context"
