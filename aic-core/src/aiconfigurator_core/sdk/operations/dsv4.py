# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4 family (ISSUE-11 / AIC-1095).

Four op classes migrate from ``_legacy.py`` into ``operations/dsv4.py``:

- ``DeepSeekV4MHCModule`` — manifold-constrained hyper-connection pre/post.
  Owns ``_mhc_module_data``. Delegates to
  ``PerfDatabase.query_mhc_module`` which becomes a one-line forward.
- ``_BaseDeepSeekV4AttentionModule`` — shared weight metadata; not
  instantiated directly. Holds the shared SOL helper used by both
  context and generation phases.
- ``ContextDeepSeekV4AttentionModule`` — context-phase SWA/CSA/HCA. Owns
  ``_context_deepseek_v4_attention_module_data`` (merged from csa+hca
  split files), ``_raw_context_deepseek_v4_attention_module_data``
  (deepcopy used for topk piecewise lookup), and the
  ``_dsv4_sparse_kernel_data`` sidecar dict (paged_mqa_logits + hca_attn)
  used for prefix kernel-Δ correction.
- ``GenerationDeepSeekV4AttentionModule`` — decode-phase. Owns
  ``_generation_deepseek_v4_attention_module_data`` (merged from
  csa+hca split files).

No SOL clamping in the legacy ``_correct_data`` for DSV4 (the per-attn
SOL formula runs inside the query path). No grid extrapolation either —
Interpolation/fallback is handled by the engine's interpolation at query time.

Cache key matches every other migrated op:
``(systems_root, system, backend, version, enable_shared_layer)``.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, ClassVar

from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk.operations.base import Operation, _read_filtered_rows, resolve_op_data_path

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from aiconfigurator_core.sdk.perf_database import PerfDatabase


def _cache_key(database: PerfDatabase) -> tuple:
    """Shared cache key — same shape as every other migrated op family."""
    return (
        database.systems_root,
        database.system,
        database.backend,
        database.version,
        database.enable_shared_layer,
    )


# ───────────────────────────────────────────────────────────────────────
# DeepSeekV4MHCModule
# ───────────────────────────────────────────────────────────────────────


class DeepSeekV4MHCModule(Operation):
    """DeepSeek-V4 manifold-constrained hyper-connection pre/post module."""

    _data_cache: ClassVar[dict] = {}
    _CP_AWARE: ClassVar[bool] = True  # token-major: query divides num_tokens by self._seq_split

    def __init__(
        self,
        name: str,
        scale_factor: float,
        op: str,
        hidden_size: int,
        hc_mult: int,
        sinkhorn_iters: int,
        quant_mode: common.GEMMQuantMode,
        *,
        seq_split: int = 1,
    ) -> None:
        super().__init__(name, scale_factor, seq_split=seq_split)
        if op not in {"pre", "post", "both"}:
            raise ValueError(f"Unsupported DeepSeek-V4 mHC op: {op}")
        self._op = op
        self._hidden_size = hidden_size
        self._hc_mult = hc_mult
        self._sinkhorn_iters = sinkhorn_iters
        self._quant_mode = quant_mode

    # ------------------------------------------------------------------
    # Data ownership
    # ------------------------------------------------------------------

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Fetches the engine's mhc_module table view, binds
        ``database._mhc_module_data``."""
        from aiconfigurator_core.sdk.engine_table_view import load_view
        from aiconfigurator_core.sdk.perf_database import PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            cls._data_cache[key] = load_view(database, "_mhc_module_data", PerfDataFilename.mhc_module)
            cls._record_load()

        if "_mhc_module_data" not in database.__dict__:
            database._mhc_module_data = cls._data_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._data_cache.clear()

    # ------------------------------------------------------------------
    # Query table (formerly PerfDatabase.query_mhc_module)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Op contract
    # ------------------------------------------------------------------

    _ENGINE_QUERY_SHAPE = "tokens"


# ───────────────────────────────────────────────────────────────────────
# _BaseDeepSeekV4AttentionModule (shared metadata)
# ───────────────────────────────────────────────────────────────────────


class _BaseDeepSeekV4AttentionModule(Operation):
    """Common DeepSeek-V4 compressed attention module metadata.

    Not instantiated directly. Subclassed by ``ContextDeepSeekV4AttentionModule``
    and ``GenerationDeepSeekV4AttentionModule``, each of which owns its own
    silicon data cache.
    """

    def __init__(
        self,
        name: str,
        scale_factor: float,
        num_heads: int,
        native_heads: int,
        tp_size: int,
        hidden_size: int,
        q_lora_rank: int,
        o_lora_rank: int,
        head_dim: int,
        rope_head_dim: int,
        index_n_heads: int,
        index_head_dim: int,
        index_topk: int,
        window_size: int,
        compress_ratio: int,
        o_groups: int,
        kvcache_quant_mode: common.KVCacheQuantMode,
        fmha_quant_mode: common.FMHAQuantMode,
        gemm_quant_mode: common.GEMMQuantMode,
        *,
        cp_size: int = 1,
    ) -> None:
        super().__init__(name, scale_factor)
        self._cp_size = cp_size  # context parallelism (sglang AllGather); >1 only on context modules
        self._num_heads = num_heads
        self._native_heads = native_heads
        self._tp_size = tp_size
        self._hidden_size = hidden_size
        self._q_lora_rank = q_lora_rank
        self._o_lora_rank = o_lora_rank
        self._head_dim = head_dim
        self._rope_head_dim = rope_head_dim
        self._index_n_heads = index_n_heads
        self._index_head_dim = index_head_dim
        self._index_topk = index_topk
        self._window_size = window_size
        self._compress_ratio = compress_ratio
        self._o_groups = o_groups
        self._kvcache_quant_mode = kvcache_quant_mode
        self._fmha_quant_mode = fmha_quant_mode
        self._gemm_quant_mode = gemm_quant_mode


# ───────────────────────────────────────────────────────────────────────
# ContextDeepSeekV4AttentionModule
# ───────────────────────────────────────────────────────────────────────


class ContextDeepSeekV4AttentionModule(_BaseDeepSeekV4AttentionModule):
    """Context-phase DeepSeek-V4 SWA/CSA/HCA compressed attention module.

    Owns three class-level caches:
    - ``_data_cache`` — merged ctx table (csa + hca split files combined)
    - ``_raw_data_cache`` — deepcopy of the merged table, kept untouched
      so the topk-piecewise lookup can consult the original
      compress_ratio==4 rows for boundary correctness.
    - ``_sparse_kernel_cache`` — dict ``{"paged_mqa_logits", "hca_attn"}``
      of ``LoadedOpData`` used for prefix kernel-Δ correction.
    """

    _data_cache: ClassVar[dict] = {}
    _raw_data_cache: ClassVar[dict] = {}
    _sparse_kernel_cache: ClassVar[dict] = {}

    # ------------------------------------------------------------------
    # Data ownership
    # ------------------------------------------------------------------

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Fetches the engine's merged csa+hca context table view
        and the three DSV4 sparse-kernel views.

        Binds:
        - ``database._context_deepseek_v4_attention_module_data``
        - ``database._raw_context_deepseek_v4_attention_module_data``
        - ``database._dsv4_sparse_kernel_data``
        """

        from aiconfigurator_core.sdk.engine_table_view import fetch_table_view
        from aiconfigurator_core.sdk.perf_database import LoadedOpData, PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache or key not in cls._raw_data_cache or key not in cls._sparse_kernel_cache:
            system_data_root = os.path.join(database.systems_root, database.system_spec["data_dir"])

            def _primary(filename_enum):
                return resolve_op_data_path(system_data_root, database.backend, database.version, filename_enum.value)

            # Locals first, commit last — a failed sparse fetch must not
            # leave only the merged view cached (see GEMM.load_data).
            # The csa+hca merge happens engine-side; an absent-or-empty merge
            # binds None, matching the retired split-merge semantics
            # (whose filepath came from the csa side, loaded first).
            merged_view = fetch_table_view(database, "_context_deepseek_v4_attention_module_data")
            if merged_view:
                merged_loaded = LoadedOpData(
                    merged_view,
                    PerfDataFilename.dsv4_csa_context_module,
                    _primary(PerfDataFilename.dsv4_csa_context_module),
                )
            else:
                merged_loaded = None

            def _load_sparse(sub_key, filename_enum):
                view = fetch_table_view(database, f"_dsv4_sparse_kernel_data.{sub_key}")
                return LoadedOpData(view, filename_enum, _primary(filename_enum))

            sparse_loaded = {
                "paged_mqa_logits": _load_sparse("paged_mqa_logits", PerfDataFilename.dsv4_paged_mqa_logits_module),
                "hca_attn": _load_sparse("hca_attn", PerfDataFilename.dsv4_hca_attn_module),
                "csa_attn": _load_sparse("csa_attn", PerfDataFilename.dsv4_csa_attn_module),
            }

            cls._data_cache[key] = merged_loaded
            # The raw wrapper stays a plain alias for backward compatibility.
            cls._raw_data_cache[key] = merged_loaded
            cls._sparse_kernel_cache[key] = sparse_loaded
            cls._record_load()

        if "_context_deepseek_v4_attention_module_data" not in database.__dict__:
            database._context_deepseek_v4_attention_module_data = cls._data_cache[key]
        if "_raw_context_deepseek_v4_attention_module_data" not in database.__dict__:
            database._raw_context_deepseek_v4_attention_module_data = cls._raw_data_cache[key]
        if "_dsv4_sparse_kernel_data" not in database.__dict__:
            database._dsv4_sparse_kernel_data = cls._sparse_kernel_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._data_cache.clear()
        cls._raw_data_cache.clear()
        cls._sparse_kernel_cache.clear()

    # ------------------------------------------------------------------
    # Sparse-kernel lookup helper (formerly PerfDatabase._lookup_dsv4_sparse_kernel)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Query table (formerly PerfDatabase.query_context_deepseek_v4_attention_module)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Op contract
    # ------------------------------------------------------------------

    _ENGINE_QUERY_SHAPE = "context"

    # ------------------------------------------------------------------
    # NOTE(#1357 PR-5): the Python CP prefill model and chunked-mqa
    # decomposition that lived here retired with the per-call query stack;
    # their oracle is the compiled engine (operators/dsv4.rs).


# ───────────────────────────────────────────────────────────────────────
# GenerationDeepSeekV4AttentionModule
# ───────────────────────────────────────────────────────────────────────


class GenerationDeepSeekV4AttentionModule(_BaseDeepSeekV4AttentionModule):
    """Decode-phase DeepSeek-V4 SWA/CSA/HCA compressed attention module.

    Owns ``_generation_deepseek_v4_attention_module_data`` (merged from
    csa+hca split files).
    """

    _data_cache: ClassVar[dict] = {}

    # ------------------------------------------------------------------
    # Data ownership
    # ------------------------------------------------------------------

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Fetches the engine's merged csa+hca generation table
        view, binds ``database._generation_deepseek_v4_attention_module_data``.
        """

        from aiconfigurator_core.sdk.engine_table_view import fetch_table_view
        from aiconfigurator_core.sdk.perf_database import LoadedOpData, PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            # The csa+hca merge happens engine-side; an absent-or-empty merge
            # binds None, matching the retired _load_dsv4_split semantics
            # (whose filepath came from the csa side, loaded first).
            merged_view = fetch_table_view(database, "_generation_deepseek_v4_attention_module_data")
            if merged_view:
                system_data_root = os.path.join(database.systems_root, database.system_spec["data_dir"])
                primary = resolve_op_data_path(
                    system_data_root,
                    database.backend,
                    database.version,
                    PerfDataFilename.dsv4_csa_generation_module.value,
                )
                cls._data_cache[key] = LoadedOpData(merged_view, PerfDataFilename.dsv4_csa_generation_module, primary)
            else:
                cls._data_cache[key] = None

            cls._record_load()

        if "_generation_deepseek_v4_attention_module_data" not in database.__dict__:
            database._generation_deepseek_v4_attention_module_data = cls._data_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._data_cache.clear()

    # ------------------------------------------------------------------
    # Query table (formerly PerfDatabase.query_generation_deepseek_v4_attention_module)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Op contract
    # ------------------------------------------------------------------

    _ENGINE_QUERY_SHAPE = "generation"


class DeepSeekV4MegaMoEModule(Operation):
    """
    SGLang DeepSeek-V4 MegaMoE routed module.

    This models the measured routed MegaMoE module boundary used by
    ``collector/sglang/collect_dsv4_megamoe.py``: prepared hidden states and
    top-k tensors -> SGLang pre-dispatch -> ``deep_gemm.fp8_fp4_mega_moe`` ->
    routed output scaling. Gate/top-k and shared experts are modeled outside
    this operation.
    """

    _data_cache: ClassVar[dict] = {}

    def __init__(
        self,
        name: str,
        scale_factor: float,
        hidden_size: int,
        inter_size: int,
        topk: int,
        num_experts: int,
        moe_tp_size: int,
        moe_ep_size: int,
        quant_mode: common.MoEQuantMode,
        workload_distribution: str,
        is_context: bool = True,
        source_policy: str = "random",
        pre_dispatch: str = "sglang_jit",
        num_fused_shared_experts: int = 0,
        kernel_source: str = "deepgemm_megamoe",
        kernel_dtype: str = "fp8_fp4",
    ) -> None:
        super().__init__(name, scale_factor)
        self._hidden_size = hidden_size
        self._inter_size = inter_size
        self._topk = topk
        self._num_experts = num_experts
        self._moe_tp_size = moe_tp_size
        self._moe_ep_size = moe_ep_size
        self._quant_mode = quant_mode
        self._workload_distribution = self._normalize_distribution(workload_distribution)
        self._is_context = is_context
        self._source_policy = source_policy
        self._pre_dispatch = pre_dispatch
        self._num_fused_shared_experts = num_fused_shared_experts
        self._kernel_source = kernel_source
        self._kernel_dtype = kernel_dtype

    @staticmethod
    def _normalize_distribution(workload_distribution: str) -> str:
        if workload_distribution == "uniform":
            return "balanced"
        return workload_distribution

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        from aiconfigurator_core.sdk.engine_table_view import load_view
        from aiconfigurator_core.sdk.perf_database import PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            # Single-primary semantics live in the engine view (it reads only
            # the head of the resolved source list, like the retired loader).
            cls._data_cache[key] = load_view(
                database, "_dsv4_megamoe_module_data", PerfDataFilename.dsv4_megamoe_module
            )
            cls._record_load()

        if "_dsv4_megamoe_module_data" not in database.__dict__:
            database._dsv4_megamoe_module_data = cls._data_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._data_cache.clear()

    _ENGINE_QUERY_SHAPE = "tokens"

    def _engine_query_plan(self, kwargs: dict):
        """Legacy per-call ``quant_mode`` override: rebuild the twin with the
        requested quant before engine evaluation (an uncovered override quant
        must MISS loudly, exactly like the retired lookup did)."""
        op, eval_kwargs = super()._engine_query_plan(kwargs)
        quant_mode = kwargs.get("quant_mode")
        if quant_mode is not None and quant_mode != self._quant_mode:
            import copy

            op = copy.copy(self)
            op._quant_mode = quant_mode
        return op, eval_kwargs


# ─────────────────────────────────────────────────────────
# CSV loaders (moved here from perf_database.py so each op family owns its data + parser)
# ─────────────────────────────────────────────────────────


_TOPK_CALIB_KEYS = ("num_heads", "step", "isl", "batch_size", "score_mode")


def load_dsv4_sparse_op_data(file_or_sources, key_columns):
    """Generic loader for the DeepSeek-V4 sparse-op family.

    Reads the shared perf schema (parquet or txt, single path or override
    ``(path, kernel_source_filter)`` sources — see ``_read_filtered_rows``) and
    nests every row under ``key_columns`` in order, leaf == ``{"latency": ms}``.

    Numeric key cells coerce to ``int``; non-numeric stay ``str`` (e.g.
    ``score_mode``). Rows with a blank or NaN/inf key cell are skipped.
    Returns ``None`` when no source file exists.

    Consumers:
      - sparse kernels: ``_SPARSE_KERNEL_KEYS`` -> data[heads][tp][past_kv][isl][bs]
      - topk calib:     ``_TOPK_CALIB_KEYS``    -> data[native][step][isl][bs][score_mode]
    """
    rows = _read_filtered_rows(file_or_sources)
    if rows is None:
        return None

    def _coerce(value):
        try:
            return int(float(value))
        except (TypeError, ValueError, OverflowError):
            return value

    def _is_bad_key(k):
        # A key cell that is blank or a NaN/inf sentinel must not become a dict
        # key: such rows are malformed and would misbucket (or KeyError) the
        # downstream calibration lookup. Legitimate non-numeric keys (e.g.
        # ``score_mode`` values like ``"default"``) are kept.
        if k is None:
            return True
        if isinstance(k, float):  # uncoerced float NaN/inf
            return k != k or k in (float("inf"), float("-inf"))
        if isinstance(k, str):
            return k.strip() == "" or k.strip().lower() in (
                "nan",
                "inf",
                "-inf",
                "+inf",
                "infinity",
                "-infinity",
            )
        return False

    root: dict = {}
    for row in rows:
        # Skip duplicate header rows (files may be appended to across runs).
        if row.get("batch_size") in (None, "", "batch_size"):
            continue
        try:
            keys = [_coerce(row[col]) for col in key_columns]
            latency = float(row["latency"])
        except (KeyError, TypeError, ValueError):
            continue
        if any(_is_bad_key(k) for k in keys):  # blank / NaN / inf key cell
            continue
        node = root
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        if keys[-1] in node:
            # Check for conflict: first source wins (shared-layer contract).
            logger.debug(f"value conflict in dsv4 sparse-op data: {keys}")
            continue
        node[keys[-1]] = {"latency": latency}
    return root or None
