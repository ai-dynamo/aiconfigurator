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
from collections import defaultdict
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
# Module-level helpers (moved from perf_database.py).
# Re-exported from perf_database for back-compat with tests that imported
# them via ``from aiconfigurator_core.sdk.perf_database import ...``.
# ───────────────────────────────────────────────────────────────────────


def _deep_merge_dsv4_dicts(dest, src):
    """In-place merge ``src`` nested dict into ``dest``.

    Used to combine the per-(attn_kind) CSVs into one nested dict. At any
    level where both sides have a dict, recurse; otherwise overwrite.
    """
    if src is None:
        return dest
    for k, v in src.items():
        if k in dest and isinstance(dest[k], dict) and isinstance(v, dict):
            _deep_merge_dsv4_dicts(dest[k], v)
        else:
            dest[k] = v
    return dest


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
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * hidden_size
        # Two parameter sets per decoder block: attention mHC and FFN mHC.
        self._weights = 2 * (mix_hc * hc_dim + mix_hc + 3) * quant_mode.value.memory

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

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


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
        self._weights = self._estimate_weights()

    def _estimate_weights(self) -> float:
        gemm_weight_elems = (
            self._hidden_size * self._q_lora_rank
            + self._q_lora_rank * self._num_heads * self._head_dim
            + self._hidden_size * self._head_dim
            + self._o_groups * self._o_lora_rank * self._hidden_size
        )
        bfloat16_weight_elems = self._num_heads * self._head_dim * self._o_lora_rank
        float32_weight_elems = self._num_heads
        if self._compress_ratio:
            compressor_mult = 2 if self._compress_ratio == 4 else 1
            gemm_weight_elems += 2 * self._hidden_size * compressor_mult * self._head_dim
            float32_weight_elems += self._compress_ratio * compressor_mult * self._head_dim
        if self._compress_ratio == 4:
            gemm_weight_elems += self._q_lora_rank * self._index_n_heads * self._index_head_dim
            gemm_weight_elems += 2 * self._hidden_size * 2 * self._index_head_dim
            bfloat16_weight_elems += self._hidden_size * self._index_n_heads
            float32_weight_elems += self._compress_ratio * 2 * self._index_head_dim
        return (
            gemm_weight_elems * self._gemm_quant_mode.value.memory
            + bfloat16_weight_elems * common.GEMMQuantMode.bfloat16.value.memory
            + float32_weight_elems * 4
        )

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


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
        import os

        from aiconfigurator_core.sdk.engine_table_view import fetch_table_view
        from aiconfigurator_core.sdk.perf_database import LoadedOpData, PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            system_data_root = os.path.join(database.systems_root, database.system_spec["data_dir"])

            def _primary(filename_enum):
                return resolve_op_data_path(
                    system_data_root, database.backend, database.version, filename_enum.value
                )

            # The csa+hca merge happens engine-side; an absent-or-empty merge
            # binds None, matching the retired _load_dsv4_split semantics
            # (whose filepath came from the csa side, loaded first).
            merged_view = fetch_table_view(database, "_context_deepseek_v4_attention_module_data")
            if merged_view:
                cls._data_cache[key] = LoadedOpData(
                    merged_view,
                    PerfDataFilename.dsv4_csa_context_module,
                    _primary(PerfDataFilename.dsv4_csa_context_module),
                )
            else:
                cls._data_cache[key] = None
            # The raw wrapper stays a plain alias for backward compatibility.
            cls._raw_data_cache[key] = cls._data_cache[key]

            def _load_sparse(sub_key, filename_enum):
                view = fetch_table_view(database, f"_dsv4_sparse_kernel_data.{sub_key}")
                return LoadedOpData(view, filename_enum, _primary(filename_enum))

            cls._sparse_kernel_cache[key] = {
                "paged_mqa_logits": _load_sparse("paged_mqa_logits", PerfDataFilename.dsv4_paged_mqa_logits_module),
                "hca_attn": _load_sparse("hca_attn", PerfDataFilename.dsv4_hca_attn_module),
                "csa_attn": _load_sparse("csa_attn", PerfDataFilename.dsv4_csa_attn_module),
            }

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
        import os

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
                cls._data_cache[key] = LoadedOpData(
                    merged_view, PerfDataFilename.dsv4_csa_generation_module, primary
                )
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
        self._weights = (
            self._hidden_size
            * self._inter_size
            * self._num_experts
            * quant_mode.value.memory
            # DSv4 MegaMoE is always gated SwiGLU: 3 GEMMs (gate, up, down).
            * 3
            // self._moe_ep_size
            // self._moe_tp_size
        )

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
            cls._data_cache[key] = load_view(database, "_dsv4_megamoe_module_data", PerfDataFilename.dsv4_megamoe_module)
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

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


# ───────────────────────────────────────────────────────────────────────
# Init-time split-file merge helper (formerly in PerfDatabase.__init__)
# ───────────────────────────────────────────────────────────────────────


def _load_dsv4_split(loaded_list):
    """Merge per-(attn_kind) loaded data into one combined ``LoadedOpData``.

    Each DSV4 context/generation module CSV is collected per attention kind
    (csa/hca). Each loader returns a nested dict scoped to one
    compress_ratio. We merge into one aggregate dict so downstream queries
    do not need to know which attention kind produced each row.
    """
    from aiconfigurator_core.sdk.perf_database import LoadedOpData

    merged: dict = {}
    first_loaded = next((x for x in loaded_list if x is not None), None)
    if first_loaded is None:
        return None
    for loaded in loaded_list:
        if loaded is None or not loaded.loaded:
            continue
        _deep_merge_dsv4_dicts(merged, loaded.data)
    if not merged:
        return None
    return LoadedOpData(merged, first_loaded.op_name_enum, first_loaded.filepath)


# ─────────────────────────────────────────────────────────
# CSV loaders (moved here from perf_database.py so each op family owns its data + parser)
# ─────────────────────────────────────────────────────────


def load_mhc_module_data(mhc_file: str):
    """Load DeepSeek-V4 mHC pre/post module-level performance data.

    CSV columns: framework, version, device, op_name, kernel_source,
    architecture, num_tokens, hc_mult, hidden_size, latency [, power]
    Optional metadata columns: num_sites, sinkhorn_iters
    Legacy rows may include a ``model`` column; it is ignored because mHC is
    selected by compute shape.

    ``op_name`` is ``pre`` or ``post``, matching the ``op`` arg of
    ``query_mhc_module``.

    Dict structure (matches query_mhc_module silicon path):
        data[op][hc_mult][hidden_size][num_tokens]
    """
    rows = _read_filtered_rows(mhc_file)
    if rows is None:
        logger.debug(f"mHC module data file {mhc_file} not found.")
        return None

    mhc_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))

    has_power = len(rows) > 0 and "power" in rows[0]

    for row in rows:
        op = row["op_name"]
        hc_mult = int(row["hc_mult"])
        hidden_size = int(row["hidden_size"])
        num_tokens = int(row["num_tokens"])
        latency = float(row["latency"])
        power = float(row.get("power", 0.0)) if has_power else 0.0
        energy = power * latency

        try:
            # Check for conflict: first source wins (shared-layer contract).
            mhc_data[op][hc_mult][hidden_size][num_tokens]
            logger.debug(f"value conflict in mhc module data: {op} {hc_mult} {hidden_size} {num_tokens}")
        except KeyError:
            mhc_data[op][hc_mult][hidden_size][num_tokens] = {
                "latency": latency,
                "power": power,
                "energy": energy,
            }

    return mhc_data


_DSV4_DTYPE_ALIASES = {
    # CSV columns use sglang naming; aic_dev enums use canonical short names.
    "fp8_e4m3": "fp8",
}


def _dsv4_normalize_dtype(name: str) -> str:
    return _DSV4_DTYPE_ALIASES.get(name, name)


# DSV4 CSA topk DELTA calibration table — DATA PROVENANCE. The collector
# measures the topK kernel standalone under a degenerate "flat" and a
# representative "top_last" score construction per shape (four score_mode
# rows in dsv4_csa_topk_calib_perf); the loader below keeps those raw rows
# on the data plane. The query-time DELTA-subtraction correction that
# consumed them retired with #1357 PR-5 — its oracle is the compiled
# engine (operators/dsv4.rs).


def _validate_dsv4_local_head_semantics(rows, file_path):
    """Reject rows still carrying the retired pre-#1131 NATIVE ``num_heads``
    semantics.

    The unified DSV4 module-row convention (issue #1429) is rank-LOCAL:
    ``num_heads`` is the head count the benchmarked module actually ran with
    on one rank, ``tp_size`` is persisted in every row, and the model-native
    count is derived as ``num_heads * tp_size``.  Within one artifact a
    genuine local sweep varies ``num_heads`` as ``native // tp``, so a group
    whose ``num_heads`` stays constant across several ``tp_size`` values can
    only be a stale pre-migration file (Flash/Flash-FP8 64, Pro 128 constant
    across tp 1/2/4/8).  Reading such a file as local would collapse distinct
    tp shards onto wrong (native, local) coordinates again — rows whose
    latencies differ 30-50% — so raise instead.  The shipped sglang 0.5.10
    tables were migrated in-place; external files must be migrated the same
    way (``num_heads //= tp_size``).

    The fingerprint is checked per ``(model, version)`` because the shared
    layer concatenates sibling-version files into one row stream — a
    migrated (local) primary pooled with a stale (native) sibling of the
    same model would otherwise blur both patterns and mask the stale rows.
    """
    observed: dict[tuple[str, str], set[tuple[int, int]]] = {}
    saw_tp_size = False
    missing_tp_rows = 0
    for row in rows:
        try:
            heads = int(row["num_heads"])
        except (TypeError, ValueError, KeyError):
            continue
        try:
            tp = max(1, int(row["tp_size"]))
            saw_tp_size = True
        except (TypeError, ValueError, KeyError):
            tp = 1
            missing_tp_rows += 1
        group = (str(row.get("model", "")), str(row.get("version", "")))
        observed.setdefault(group, set()).add((heads, tp))

    if saw_tp_size and missing_tp_rows:
        # A per-row tp_size fallback to 1 would derive native = num_heads and
        # file the row under a wrong native bucket (#1460 review): fail on any
        # unparseable tp_size once the file demonstrably carries the column.
        raise ValueError(
            f"DSV4 module file {file_path} has {missing_tp_rows} row(s) without a parseable "
            f"tp_size; the #1429 convention requires tp_size in every row "
            f"(native = num_heads * tp_size)."
        )

    if observed and not saw_tp_size:
        # Without tp_size every row collapses to tp=1 and the stale fingerprint
        # below can never trigger — a stale file would load silently with wrong
        # (native, local) coordinates. The #1429 convention makes tp_size a
        # mandatory column, so fail like the Rust loader does.
        raise ValueError(
            f"DSV4 module file {file_path} carries no parseable tp_size column; the #1429 "
            f"convention requires tp_size in every row (native = num_heads * tp_size)."
        )

    for (model, version), pairs in observed.items():
        tps = {tp for _, tp in pairs}
        heads_constant = len({h for h, _ in pairs}) == 1
        product_constant = len({h * tp for h, tp in pairs}) == 1
        if len(tps) > 1 and heads_constant and not product_constant:
            raise ValueError(
                f"DSV4 module rows for model={model!r} version={version!r} in {file_path} keep "
                f"num_heads constant across tp_size values {sorted(tps)}: that is the retired "
                f"pre-#1131 NATIVE semantics (#1429). Migrate the file to rank-local heads "
                f"(num_heads //= tp_size) before loading."
            )


def load_context_dsv4_kind_module_data(file_path: str):
    """Load ONE DeepSeek-V4 context CSV (single attn_kind / compress_ratio).

    Returns an 8-level prefix-resolved nested dict:
        data[fmha_quant][kv_quant][gemm_quant][num_heads_native][num_heads_local]
            [compress_ratio][prefix][s][b] = {"latency": ms, "power": W, "energy": J}

    The head identity is (native, rank-local) under the unified #1429
    convention: the ``num_heads`` column is the rank-LOCAL head count the
    benchmarked module ran with, and the model-native count is derived as
    ``num_heads * tp_size`` (``_validate_dsv4_local_head_semantics`` rejects
    stale pre-#1131 files that stored native heads instead).  The native value
    is the row's model identity (separates Pro rows from Flash rows) and the
    local count is the physical per-rank shape; both are key dimensions.
    Collapsing either axis merged rows whose latencies differ 30-50%
    (different model shapes / tp shards) into one coordinate, leaving an
    arbitrary row-order winner.

    ``prefix`` is the past-KV length, ``int(float(row["step"]))``; ``s`` is the
    context chunk length (``isl``).  Multiple files (csa/hca) merge cleanly
    because compress_ratio is a key dimension.
    """
    rows = _read_filtered_rows(file_path)
    if rows is None:
        logger.debug(f"DSV4 module data file {file_path} not found.")
        return None
    _validate_dsv4_local_head_semantics(rows, file_path)

    # 8-level nesting: fmha → kv → gemm → native → local → cr → prefix → s → b
    def _make_nested(depth: int):
        if depth == 0:
            return defaultdict()
        return defaultdict(lambda d=depth: _make_nested(d - 1))

    data = _make_nested(8)
    has_power = bool(rows) and "power" in rows[0]

    for row in rows:
        if row.get("batch_size") in (None, "", "batch_size"):
            continue  # skip duplicate header rows from appended runs
        try:
            b = int(row["batch_size"])
            s = int(row["isl"])
            prefix = int(float(row.get("step", 0) or 0))
            cr = int(row["compress_ratio"])
            latency = float(row["latency"])
            heads_col = int(row["num_heads"])
            tp_size = max(1, int(row.get("tp_size", 1) or 1))
        except (TypeError, ValueError, KeyError):
            continue
        power = float(row.get("power", 0.0)) if has_power else 0.0

        num_heads_local = heads_col
        num_heads_native = heads_col * tp_size
        gemm_mode = common.GEMMQuantMode[row["gemm_type"]]
        fmha_mode = common.FMHAQuantMode[_dsv4_normalize_dtype(row["mla_dtype"])]
        kv_dtype = common.KVCacheQuantMode[_dsv4_normalize_dtype(row["kv_cache_dtype"])]

        # NOTE: the topK DELTA correction (degenerate -> representative) is
        # applied ONCE at query time for compress_ratio==4 (CSA). Do NOT
        # subtract it here, or the CSA module latency would be double-corrected.
        try:
            # Check for conflict: first source wins (shared-layer contract).
            data[fmha_mode][kv_dtype][gemm_mode][num_heads_native][num_heads_local][cr][prefix][s][b]
            logger.debug(
                f"value conflict in context dsv4 module data: {fmha_mode} {kv_dtype} {gemm_mode} "
                f"{num_heads_native} {num_heads_local} {cr} {prefix} {s} {b}"
            )
        except KeyError:
            data[fmha_mode][kv_dtype][gemm_mode][num_heads_native][num_heads_local][cr][prefix][s][b] = {
                "latency": latency,
                "power": power,
                "energy": power * latency,
            }
    return data


def load_generation_dsv4_kind_module_data(file_path: str):
    """Load ONE DeepSeek-V4 generation CSV.

    Generation lookup uses absolute KV length ``s_total = isl + step`` (decode
    is q_len=1 with past_kv = step).  Dict shape (same (native, local) head
    identity as ``load_context_dsv4_kind_module_data``: rank-local ``num_heads``
    column, native derived as ``num_heads * tp_size``, stale NATIVE-semantics
    files rejected by ``_validate_dsv4_local_head_semantics``):
        data[kv_quant][gemm_quant][num_heads_native][num_heads_local]
            [compress_ratio][b][s_total]
    """
    rows = _read_filtered_rows(file_path)
    if rows is None:
        logger.debug(f"DSV4 module data file {file_path} not found.")
        return None
    _validate_dsv4_local_head_semantics(rows, file_path)

    # 6-level nesting: kv → gemm → native → local → cr → b → s_total
    def _make_nested(depth: int):
        if depth == 0:
            return defaultdict()
        return defaultdict(lambda d=depth: _make_nested(d - 1))

    data = _make_nested(6)
    has_power = bool(rows) and "power" in rows[0]

    for row in rows:
        if row.get("batch_size") in (None, "", "batch_size"):
            continue
        try:
            b = int(row["batch_size"])
            s_total = int(row["isl"]) + int(row["step"])
            cr = int(row["compress_ratio"])
            latency = float(row["latency"])
            heads_col = int(row["num_heads"])
            tp_size = max(1, int(row.get("tp_size", 1) or 1))
        except (TypeError, ValueError, KeyError):
            continue
        power = float(row.get("power", 0.0)) if has_power else 0.0

        num_heads_local = heads_col
        num_heads_native = heads_col * tp_size
        gemm_mode = common.GEMMQuantMode[row["gemm_type"]]
        kv_dtype = common.KVCacheQuantMode[_dsv4_normalize_dtype(row["kv_cache_dtype"])]

        try:
            # Check for conflict: first source wins (shared-layer contract).
            data[kv_dtype][gemm_mode][num_heads_native][num_heads_local][cr][b][s_total]
            logger.debug(
                f"value conflict in generation dsv4 module data: {kv_dtype} {gemm_mode} "
                f"{num_heads_native} {num_heads_local} {cr} {b} {s_total}"
            )
        except KeyError:
            data[kv_dtype][gemm_mode][num_heads_native][num_heads_local][cr][b][s_total] = {
                "latency": latency,
                "power": power,
                "energy": power * latency,
            }
    return data


def load_dsv4_megamoe_module_data(dsv4_megamoe_module_file):
    """
    Load DeepSeek-V4 MegaMoE full-module data.

    The collected latency is the SGLang/DeepGEMM MegaMoE routed path:
    prepared hidden states and top-k tensors -> pre-dispatch -> fused MegaMoE.
    Gate/top-k generation is intentionally outside the measured region.

    Returns:
        dict: Nested dict whose leaves contain latency, power, energy and
        routing metadata.
    """
    if dsv4_megamoe_module_file is None:
        return None

    if isinstance(dsv4_megamoe_module_file, list | tuple):
        raise TypeError("DSv4 MegaMoE data loader expects a single unified perf file path")

    source_label = os.fspath(dsv4_megamoe_module_file)
    rows = _read_filtered_rows(source_label)
    if rows is None:
        logger.debug(f"DeepSeek-V4 MegaMoE data file {source_label} not found.")
        return None

    def _to_bool(value: object) -> bool:
        return str(value).strip().lower() in {"1", "true", "yes", "y"}

    row_bool_invariants = [
        ("used_cuda_graph", True, None, "DSv4 MegaMoE perf row was not collected with CUDA Graph"),
        (
            "includes_gate_topk",
            False,
            "true",
            "DSv4 MegaMoE perf row includes gate/top-k outside the supported boundary",
        ),
        ("includes_routed_scale", True, None, "DSv4 MegaMoE perf row does not include SGLang routed output scaling"),
    ]

    def _row_phase(row: dict[str, str]) -> str:
        phase = row.get("phase", "").strip()
        if not phase:
            raise ValueError(f"DSv4 MegaMoE unified perf file requires a phase column: {source_label} {row}")
        if phase not in {"context", "generation"}:
            raise ValueError(f"DSv4 MegaMoE perf row has unsupported phase={phase!r}: {row}")
        return phase

    def _put_nested(root: dict, keys: list[object], value: dict) -> None:
        current = root
        for key in keys[:-1]:
            current = current.setdefault(key, {})
        leaf_key = keys[-1]
        if leaf_key in current:
            raise ValueError(f"duplicate DSv4 MegaMoE data row for {source_label} {keys}")
        current[leaf_key] = value

    dsv4_megamoe_data: dict = {}
    logger.debug(f"Loading DeepSeek-V4 MegaMoE module data from: {source_label}")
    for row in rows:
        for field, expected_value, default, error in row_bool_invariants:
            if _to_bool(row.get(field, default)) != expected_value:
                raise ValueError(f"{error}: {source_label} {row}")

        kernel_source = row.get("kernel_source", "deepgemm_megamoe")
        kernel_dtype = row["kernel_dtype"]
        quant_mode = common.MoEQuantMode[row["moe_dtype"]]
        pre_dispatch = row["pre_dispatch"]
        source_policy = row["source_policy"]
        distribution = row["distribution"]
        topk = int(row["topk"])
        num_experts = int(row["num_experts"])
        num_fused_shared_experts = int(row.get("num_fused_shared_experts", 0))
        hidden_size = int(row["hidden_size"])
        inter_size = int(row["inter_size"])
        moe_tp_size = int(row.get("moe_tp_size", 1))
        moe_ep_size = int(row["moe_ep_size"])
        num_tokens = int(row["num_tokens"])
        latency = float(row["latency"])
        power = float(row.get("power") or 0.0)
        energy = power * latency
        num_max_tokens_per_rank = int(row.get("num_max_tokens_per_rank") or 0)
        effective_num_max_tokens_per_rank = int(row.get("effective_num_max_tokens_per_rank") or num_max_tokens_per_rank)

        entry = {
            "latency": latency,
            "power": power,
            "energy": energy,
            "global_num_tokens": int(row.get("global_num_tokens") or num_tokens * moe_ep_size),
            "num_max_tokens_per_rank": num_max_tokens_per_rank,
            "effective_num_max_tokens_per_rank": effective_num_max_tokens_per_rank,
            "used_cuda_graph": True,
            "kernel_dtype": kernel_dtype,
            "routed_scaling_factor": float(row["routed_scaling_factor"]),
            "includes_routed_scale": True,
            "includes_gate_topk": False,
            "buffer_policy": row.get("buffer_policy", ""),
            "includes_buffer_init": _to_bool(row.get("includes_buffer_init", "false")),
        }
        phase = _row_phase(row)
        entry["phase"] = phase
        _put_nested(
            dsv4_megamoe_data,
            [
                phase,
                kernel_source,
                kernel_dtype,
                quant_mode,
                pre_dispatch,
                source_policy,
                distribution,
                topk,
                num_experts,
                num_fused_shared_experts,
                hidden_size,
                inter_size,
                moe_tp_size,
                moe_ep_size,
                num_tokens,
            ],
            entry,
        )

    return dsv4_megamoe_data


# ───────────────────────────────────────────────────────────────────────
# DSV4 sparse-op family loader (ONE engine for all four)
# ───────────────────────────────────────────────────────────────────────
# csa_attn / hca_attn / paged_mqa_logits (FMLA & indexer kernels) and the
# csa_topk_calib DELTA rows share ONE column schema, so they all parse through
# ``load_dsv4_sparse_op_data``; each consumer just supplies the key columns it
# indexes on (declared here so callers stay in sync).
_SPARSE_KERNEL_KEYS = ("num_heads", "tp_size", "step", "isl", "batch_size")
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


def load_dsv4_sparse_kernel_data(file_or_sources):
    """DSV4 sparse-kernel CSV (csa_attn / hca_attn / paged_mqa_logits).

    Thin wrapper over ``load_dsv4_sparse_op_data`` with the kernel key columns,
    yielding ``data[native_heads][tp_size][past_kv][isl][bs] = {"latency": ms}``.
    """
    return load_dsv4_sparse_op_data(file_or_sources, _SPARSE_KERNEL_KEYS)
