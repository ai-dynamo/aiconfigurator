# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DSA (DeepSeek Sparse Attention) module-level ops (ISSUE-10 / AIC-538).

Both ContextDSAModule and GenerationDSAModule own their CSV-backed perf
tables and grid extrapolation. ``PerfDatabase.query_context_dsa_module``
and ``query_generation_dsa_module`` delegate here.

Both classes still bind a ``_raw_data_cache`` for backward compatibility,
but with load-time pre-expansion removed the table IS the raw measurements,
so it is a plain alias. (The PR #903 topk-piecewise lookup and the hand-rolled
boundary-util anchoring it served are superseded by perf_interp: linear
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
from collections import defaultdict
from typing import TYPE_CHECKING, ClassVar

from aiconfigurator_core.sdk import common, perf_interp
from aiconfigurator_core.sdk.errors import InterpolationDataNotAvailableError, PerfDataNotAvailableError
from aiconfigurator_core.sdk.operations import util_empirical
from aiconfigurator_core.sdk.operations.base import Operation, _read_filtered_rows, resolve_op_data_path
from aiconfigurator_core.sdk.performance_result import PerformanceResult

if TYPE_CHECKING:
    from aiconfigurator_core.sdk.perf_database import PerfDatabase

logger = logging.getLogger(__name__)


# kernel_source -> configured-backend bucket(s) for FP8-KV rows. 0.5.14
# SGLang DSA collectors record the EXECUTED kernel; dense ragged prefill is
# selected by SHAPE (isl <= 2048) under either configured backend, so its
# rows back both buckets.
_DSA_KERNEL_SOURCE_BUCKETS = {
    "sglang_dsa_indexer_trtllm": ("trtllm",),
    "sglang_dsa_skip_indexer_trtllm": ("trtllm",),
    "sglang_dsa_indexer_flashmla_sparse": ("flashmla_kv",),
    "sglang_dsa_skip_indexer_flashmla_sparse": ("flashmla_kv",),
    "sglang_dsa_dense_mha_trtllm_ragged": ("trtllm", "flashmla_kv"),
}


def _dsa_kernel_source_buckets(kernel_source: str, kv_dtype) -> tuple[str, ...]:
    """Configured-backend bucket(s) a DSA perf row supports.

    The trtllm/flashmla_kv split mirrors serving's FP8-KV sub-backend selector
    (an FP8-KV rule: SM90 -> flashmla_kv, SM100+ -> trtllm; BF16 KV stays on
    framework defaults). With a BF16 KV cache there is exactly ONE real
    execution path per shape, so every bf16 row backs BOTH buckets — a bare
    substring test split one measured b200 sweep across the two buckets and
    left the default query bucket with nothing beyond 2048 tokens. FP8 rows
    bucket by executed-kernel name; legacy (pre-0.5.14) names keep the old
    substring rule.
    """
    if kv_dtype is common.KVCacheQuantMode.bfloat16:
        return ("trtllm", "flashmla_kv")
    buckets = _DSA_KERNEL_SOURCE_BUCKETS.get(kernel_source)
    if buckets is not None:
        return buckets
    return ("trtllm",) if "trtllm" in kernel_source else ("flashmla_kv",)


def _dsa_module_has_prefix_axis(module_dict) -> bool:
    """Detect whether a context-DSA module slice carries an explicit prefix axis.

    Mirrors the silicon-path detection: a slice is ``[num_heads][prefix][s][b]``
    (prefix present) vs the legacy ``[num_heads][s][b]`` (prefix folded into the
    sequence length). We look one level under num_heads: if its values are still
    nested dicts (not latency leaves), there is an extra axis -> prefix present.
    Collected for BOTH DeepseekV32 and GlmMoeDsa across trtllm/sglang/vllm, so the
    empirical path must detect it from data rather than hardcode it per-arch.
    """
    for head_data in module_dict.values():
        if not isinstance(head_data, dict):
            continue
        for first_slice in head_data.values():
            if not isinstance(first_slice, dict):
                continue
            # Legacy [num_heads][s][b]: first_slice's values are latency-leaf dicts.
            return not any(isinstance(v, dict) and "latency" in v for v in first_slice.values())
    return False


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

# DSA sparse sub-kernel (mqa / topk / dsa_attn) data-file prefix per architecture.
# GLM-5 and DeepSeek-V3.2 share the same DSA kernels (only shapes/heads differ),
# so the CP delta strategy in ContextDSAModule._query_cp is identical -- only the
# collected data files differ: glm5_* vs dsv32_*. Defaults to glm5 for back-compat.
_DSA_SPARSE_FILE_PREFIX = {
    "GlmMoeDsaForCausalLM": "glm5",
    "DeepseekV32ForCausalLM": "dsv32",
}


def _dsa_sparse_file_prefix(architecture: str) -> str:
    return _DSA_SPARSE_FILE_PREFIX.get(architecture, "glm5")


# Extrapolation grids — lifted verbatim from the legacy blocks in
# ``PerfDatabase.__init__``.

# fmt: on


def _select_dsa_backend(arch_dict, dsa_backend):
    """Pick the per-backend sub-dict from a context-DSA architecture node.

    Context data is keyed ...[architecture][backend][num_heads]...; backend is
    "trtllm" (faster kernel, non-CP default) or "flashmla_kv" (used under CP).
    Falls back to whichever backend is present so single-backend parquets still
    resolve. Legacy nodes without a backend axis (int head keys) pass through."""
    if not isinstance(arch_dict, dict) or not arch_dict:
        return arch_dict
    if not any(isinstance(k, str) for k in arch_dict):
        return arch_dict
    return (
        arch_dict.get(dsa_backend)
        or arch_dict.get("flashmla_kv")
        or arch_dict.get("trtllm")
        or next(iter(arch_dict.values()))
    )


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


def _format_dsa_unavailable_message(
    phase: str,
    error: Exception,
    *,
    b: int,
    s: int,
    num_heads: int,
    architecture: str,
    index_n_heads: int,
    index_head_dim: int,
    index_topk: int,
    prefix: int | None = None,
) -> str:
    """Format the ``PerfDataNotAvailableError`` message body. Lifted verbatim
    from ``PerfDatabase._format_dsa_unavailable_message``."""
    prefix_part = "" if prefix is None else f", prefix={prefix}"
    return (
        f"{phase} DSA module perf data unavailable for candidate "
        f"b={b}, s={s}{prefix_part}, num_heads={num_heads}, architecture={architecture}, "
        f"index_n_heads={index_n_heads}, index_head_dim={index_head_dim}, index_topk={index_topk}: {error}"
    )


class ContextDSAModule(Operation):
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

    def __init__(
        self,
        name: str,
        scale_factor: float,
        num_heads: int,
        kvcache_quant_mode: common.KVCacheQuantMode,
        fmha_quant_mode: common.FMHAQuantMode,
        gemm_quant_mode: common.GEMMQuantMode,
        architecture: str = "DeepseekV32ForCausalLM",
        cp_size: int = 1,
        index_topk_freq: int = 1,
        dsa_full_layer_fraction: float | None = None,
    ) -> None:
        super().__init__(name, scale_factor)
        self._num_heads = num_heads
        self._kvcache_quant_mode = kvcache_quant_mode
        self._fmha_quant_mode = fmha_quant_mode
        self._gemm_quant_mode = gemm_quant_mode
        self._architecture = architecture
        self._cp_size = cp_size
        # GLM-5.2 shares one DSA topk index across a group of layers: some compute
        # the indexer (full), the rest reuse it (skip). query() amortizes
        # per_layer = full_frac*full + (1-full_frac)*skip, using the directly-
        # collected skip data (no delta). full_frac is the EXACT fraction of
        # indexer-computing layers (honors index_skip_topk_offset/pattern), passed
        # by the model; fall back to 1/freq only if not provided. full_frac==1.0
        # (DeepSeek-V3.2 / GLM-5) => pure full, skip path never taken.
        self._index_topk_freq = max(1, int(index_topk_freq or 1))
        self._full_frac = (
            float(dsa_full_layer_fraction) if dsa_full_layer_fraction is not None else 1.0 / self._index_topk_freq
        )
        self._weights = 0.0

    # ------------------------------------------------------------------
    # Data ownership
    # ------------------------------------------------------------------

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Loads context_dsa_module CSV, deepcopies the raw
        version, applies grid extrapolation to the main cache, binds
        ``database._context_dsa_module_data`` and
        ``database._raw_context_dsa_module_data``."""
        import os

        from aiconfigurator_core.sdk.perf_database import LoadedOpData, PerfDataFilename

        key = cls._cache_key(database)
        system_data_root = os.path.join(database.systems_root, database.system_spec["data_dir"])
        primary_path = resolve_op_data_path(
            system_data_root, database.backend, database.version, PerfDataFilename.dsa_context_module.value
        )
        sources = database._build_op_sources(PerfDataFilename.dsa_context_module, primary_path, system_data_root)
        if key not in cls._data_cache:
            cls._data_cache[key] = LoadedOpData(
                load_context_dsa_module_data(sources, op_kind="full"), PerfDataFilename.dsa_context_module, primary_path
            )
            # No load-time grid pre-expansion: queries resolve on the RAW grid
            # via perf_interp, so the raw wrapper is now an alias of the table.
            cls._raw_data_cache[key] = cls._data_cache[key]
            cls._record_load()

        # skip_indexer (GLM-5.2) rows live in the SAME file, tagged by op_name
        # (dsa_context_module_skip_indexer). Load them from the same sources with
        # op_kind="skip". Empty (no skip rows -> DeepSeek-V3.2 / GLM-5 freq==1) =>
        # slot None and the skip query path is never taken.
        if key not in cls._skip_data_cache:
            skip_dict = load_context_dsa_module_data(sources, op_kind="skip")
            if skip_dict:
                cls._skip_data_cache[key] = LoadedOpData(skip_dict, PerfDataFilename.dsa_context_module, primary_path)
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
        cls._glm5_sparse_cache.clear()
        cls._skip_data_cache.clear()
        cls._raw_skip_data_cache.clear()

    # ------------------------------------------------------------------
    # Query table (formerly PerfDatabase.query_context_dsa_module)
    # ------------------------------------------------------------------

    @classmethod
    def _query_context_dsa_module_table(
        cls,
        database: PerfDatabase,
        b: int,
        s: int,
        num_heads: int,
        kvcache_quant_mode: common.KVCacheQuantMode,
        fmha_quant_mode: common.FMHAQuantMode,
        gemm_quant_mode: common.GEMMQuantMode = common.GEMMQuantMode.bfloat16,
        database_mode: common.DatabaseMode | None = None,
        *,
        prefix: int = 0,
        architecture: str | None = None,
        index_n_heads: int | None = None,
        index_head_dim: int | None = None,
        index_topk: int | None = None,
        dsa_backend: str = "trtllm",
        skip_indexer: bool = False,
    ):
        """Query context DSA module table. Verbatim port of the legacy body.

        ``skip_indexer=True`` reads the GLM-5.2 reuse-layer table
        (``_context_dsa_module_skip_data``) instead of the full table; all other
        lookup logic is identical."""
        from aiconfigurator_core.sdk.perf_database import PerfDataNotAvailableError

        # ``DEFAULT_DSA_ARCHITECTURE`` and ``DSA_MODEL_DIMS`` live at module
        # level in this file — use them directly rather than round-tripping
        # through ``perf_database``'s backward-compat re-export.

        if architecture is None:
            architecture = DEFAULT_DSA_ARCHITECTURE

        dims = DSA_MODEL_DIMS.get(architecture, DSA_MODEL_DIMS[DEFAULT_DSA_ARCHITECTURE])
        hidden_size = dims["hidden_size"]
        q_lora = dims["q_lora_rank"]
        kv_lora = dims["kv_lora_rank"]
        qk_nope = dims["qk_nope_head_dim"]
        qk_rope = dims["qk_rope_head_dim"]
        v_dim = dims["v_head_dim"]
        if index_n_heads is None:
            index_n_heads = dims["index_n_heads"]
        if index_head_dim is None:
            index_head_dim = dims["index_head_dim"]
        if index_topk is None:
            index_topk = dims["index_topk"]
        qk_head_dim = qk_nope + qk_rope
        attn_head_dim = kv_lora + qk_rope

        def get_sol(
            b: int,
            s: int,
            prefix: int,
            num_heads: int,
            kvcache_quant_mode: common.KVCacheQuantMode,
            fmha_quant_mode: common.FMHAQuantMode,
        ) -> tuple[float, float, float]:
            """SOL estimate for the full DSA context attention block.

            Ops are split into two groups with different throughput/memory:
              - GEMM group (linear projections + absorption BMMs): gemm_quant_mode
              - Attention group (indexer logits + sparse MLA): fmha_quant_mode
            """
            full_s = s + prefix
            tokens = b * s

            # ── Compute (FLOPs) ─────────────────────────────────────────
            proj_out = q_lora + kv_lora + qk_rope + index_head_dim

            gemm_group_ops = (
                2 * tokens * hidden_size * proj_out
                + 2 * tokens * q_lora * (num_heads * qk_head_dim)
                + 2 * tokens * q_lora * (index_n_heads * index_head_dim)
                + 2 * tokens * hidden_size * index_n_heads
                + 2 * tokens * (num_heads * v_dim) * hidden_size
                + 2 * num_heads * tokens * qk_nope * kv_lora
                + 2 * num_heads * tokens * kv_lora * v_dim
            )

            # Indexer logits group: always FP8 (hardcoded in both vLLM and TRT-LLM).
            # A skip-indexer (reuse) layer does NOT run the per-layer indexer (no
            # mqa logits): it reuses a sibling layer's topk indices, so this group
            # is zero regardless of full_s.
            if skip_indexer or full_s <= index_topk:
                indexer_logits_ops = 0
            else:
                indexer_logits_ops = 2 * tokens * index_n_heads * index_head_dim * full_s

            # Sparse MLA attention group — throughput governed by fmha_quant_mode
            effective_kv = min(full_s, index_topk)
            # Exact KV pair count: sum_{i=0..s-1} min(prefix+i+1, topk)
            if full_s <= index_topk:
                total_kv_pairs = b * (full_s * (full_s + 1) - prefix * (prefix + 1)) // 2
            elif prefix >= index_topk:
                total_kv_pairs = tokens * index_topk
            else:
                ramp_pairs = b * (index_topk * (index_topk + 1) - prefix * (prefix + 1)) // 2
                sat_pairs = b * (full_s - index_topk) * index_topk
                total_kv_pairs = ramp_pairs + sat_pairs
            sparse_attn_ops = 2 * num_heads * (attn_head_dim + kv_lora) * total_kv_pairs

            # ── Memory (bytes) ──────────────────────────────────────────
            gemm_weight_bytes = (
                hidden_size * proj_out
                + q_lora * num_heads * qk_head_dim
                + q_lora * index_n_heads * index_head_dim
                + hidden_size * index_n_heads
                + num_heads * v_dim * hidden_size
            ) * gemm_quant_mode.value.memory

            kv_cache_bytes = b * num_heads * effective_kv * attn_head_dim * kvcache_quant_mode.value.memory
            indexer_entry_bytes = common.indexer_cache_entry_bytes(index_head_dim)
            # Skip layers never store the index-K cache (the indexer never runs).
            indexer_cache_bytes = 0 if (skip_indexer or full_s <= index_topk) else b * full_s * indexer_entry_bytes
            q_io_bytes = tokens * num_heads * qk_head_dim * fmha_quant_mode.value.memory * 2

            total_mem = gemm_weight_bytes + kv_cache_bytes + indexer_cache_bytes + q_io_bytes

            # ── SOL ─────────────────────────────────────────────────────
            from aiconfigurator_core.sdk.operations.gemm import GEMM

            gemm_flops = GEMM._get_quant_tc_flops(database.system_spec, gemm_quant_mode)
            indexer_fp8_flops = GEMM._get_quant_tc_flops(database.system_spec, common.FMHAQuantMode.fp8)
            attn_flops = GEMM._get_quant_tc_flops(database.system_spec, fmha_quant_mode)

            sol_math = (
                gemm_group_ops / gemm_flops + indexer_logits_ops / indexer_fp8_flops + sparse_attn_ops / attn_flops
            ) * 1000
            sol_mem = total_mem / database.system_spec["gpu"]["mem_bw"] * 1000
            sol_time = max(sol_math, sol_mem)
            return sol_time, sol_math, sol_mem

        def get_empirical(
            b: int,
            s: int,
            prefix: int,
            num_heads: int,
            kvcache_quant_mode: common.KVCacheQuantMode,
            fmha_quant_mode: common.FMHAQuantMode,
        ) -> float:
            # SOL / util, util read best-effort from own collected data. The prefix
            # axis ([num_heads][prefix][s][b] vs legacy [num_heads][s][b]) is
            # DETECTED from the data -- matching the silicon path -- not hardcoded
            # per-arch: both DeepseekV32 and GlmMoeDsa carry it on every framework.
            sol_time = get_sol(b, s, prefix, num_heads, kvcache_quant_mode, fmha_quant_mode)[0]

            def _select_slice(wrapper):
                if wrapper is None:
                    raise PerfDataNotAvailableError("Context DSA module data is not loaded.")
                arch_node = util_empirical.require_data_slice(
                    wrapper,
                    fmha_quant_mode,
                    kvcache_quant_mode,
                    gemm_quant_mode,
                    architecture,
                )
                # Loader stores ...[architecture][dsa_backend][num_heads]...; descend past
                # the backend axis exactly like the silicon path. Without this the grid sees
                # dsa_backend strings where it expects the num_heads axis and never resolves.
                selected = _select_dsa_backend(arch_node, dsa_backend)
                if selected is None:
                    raise PerfDataNotAvailableError(f"No context DSA data for backend {dsa_backend!r}.")
                return selected

            def _raw_slice():
                cls.load_data(database)
                raw_wrapper = getattr(
                    database,
                    "_raw_context_dsa_module_skip_data" if skip_indexer else "_raw_context_dsa_module_data",
                    None,
                )
                if raw_wrapper is None or not getattr(raw_wrapper, "loaded", True):
                    raise PerfDataNotAvailableError("Raw context DSA module data is not loaded.")
                return _select_slice(raw_wrapper)

            try:
                slc = _raw_slice()
                has_prefix = _dsa_module_has_prefix_axis(slc)
            except PerfDataNotAvailableError:
                slc = None
                has_prefix = architecture == "GlmMoeDsaForCausalLM"  # data unavailable: prior heuristic

            # num_heads identifies a TP/model shape. Keep a query on its exact
            # measured head slice whenever that slice exists; only retain the
            # historical cross-head fallback when the exact slice is absent.
            head_data = slc.get(num_heads) if isinstance(slc, dict) else None
            exact_head = isinstance(head_data, dict) and bool(head_data)
            calibration_data = head_data if exact_head else slc

            # Collected prefix values, when the slice carries an explicit prefix axis.
            prefix_keys: tuple = ()
            if has_prefix and isinstance(calibration_data, dict):
                if exact_head:
                    prefix_keys = tuple(sorted(calibration_data))
                else:
                    seen: set = set()
                    for candidate_head_data in calibration_data.values():
                        if isinstance(candidate_head_data, dict):
                            seen.update(candidate_head_data.keys())
                    prefix_keys = tuple(sorted(seen))
            # Genuine prefix interpolation needs >=2 collected prefix points bracketing the
            # query (mirrors the silicon path). A degenerate axis (e.g. prefix=0 only) or an
            # out-of-range query would otherwise borrow util at the query's own (prefix, s) --
            # which crosses the indexer on/off boundary (full_s vs index_topk) and the small-s
            # overhead floor, inflating the estimate. Fall back to anchoring util at the
            # prefix=0 slice at full_s = s + prefix (regime-matched) while the prefix effect is
            # carried entirely by the (true) SOL -- the windowed-attention correction pattern.
            interp_prefix = len(prefix_keys) >= 2 and prefix_keys[0] <= prefix <= prefix_keys[-1]

            if has_prefix and interp_prefix:
                # Genuine measured prefix axis. Samples are (prefix, s, b) on
                # an exact head, otherwise (num_heads, prefix, s, b).
                if exact_head:
                    depth, query, key_tag = 3, (prefix, s, b), "ctx_dsa_exact_head"

                    def _sol(c):
                        return get_sol(c[2], c[1], c[0], num_heads, kvcache_quant_mode, fmha_quant_mode)[0]

                else:
                    depth, query, key_tag = 4, (num_heads, prefix, s, b), "ctx_dsa"

                    def _sol(c):
                        return get_sol(c[3], c[2], c[1], c[0], kvcache_quant_mode, fmha_quant_mode)[0]
            elif has_prefix and 0 in prefix_keys:
                # Degenerate/out-of-range prefix axis: anchor utilization at
                # prefix=0 and full_s=s+prefix so indexer/top-k regime changes
                # remain in the true query SOL.
                key_tag = "ctx_dsa_p0anchor_exact_head" if exact_head else "ctx_dsa_p0anchor"
                if exact_head:
                    calibration_data = calibration_data[0]
                    depth, query = 2, (s + prefix, b)

                    def _sol(c):
                        return get_sol(c[1], c[0], 0, num_heads, kvcache_quant_mode, fmha_quant_mode)[0]

                else:
                    calibration_data = {
                        head: candidate_head_data[0]
                        for head, candidate_head_data in calibration_data.items()
                        if isinstance(candidate_head_data, dict) and 0 in candidate_head_data
                    }
                    depth, query = 3, (num_heads, s + prefix, b)

                    def _sol(c):
                        return get_sol(c[2], c[1], 0, c[0], kvcache_quant_mode, fmha_quant_mode)[0]
            elif has_prefix:
                # No prefix=0 anchor exists. Preserve coverage by freezing the
                # generic raw-grid utilization at the nearest measured prefix.
                if exact_head:
                    depth, query, key_tag = 3, (prefix, s, b), "ctx_dsa_prefix_boundary_exact_head"

                    def _sol(c):
                        return get_sol(c[2], c[1], c[0], num_heads, kvcache_quant_mode, fmha_quant_mode)[0]

                else:
                    depth, query, key_tag = 4, (num_heads, prefix, s, b), "ctx_dsa_prefix_boundary"

                    def _sol(c):
                        return get_sol(c[3], c[2], c[1], c[0], kvcache_quant_mode, fmha_quant_mode)[0]
            elif exact_head:
                # Legacy raw [num_heads][s][b] table.
                depth, query, key_tag = 2, (s + prefix, b), "ctx_dsa_legacy_exact_head"

                def _sol(c):
                    return get_sol(c[1], c[0], 0, num_heads, kvcache_quant_mode, fmha_quant_mode)[0]
            else:
                depth, query, key_tag = 3, (num_heads, s + prefix, b), "ctx_dsa_legacy"

                def _sol(c):
                    return get_sol(c[2], c[1], 0, c[0], kvcache_quant_mode, fmha_quant_mode)[0]

            def slice_fn():
                if calibration_data is None:
                    raise PerfDataNotAvailableError("Raw context DSA calibration data is not available.")
                return calibration_data

            grid = util_empirical.grid_for(
                (
                    key_tag,
                    database.system,
                    database.backend,
                    database.version,
                    fmha_quant_mode.name,
                    kvcache_quant_mode.name,
                    gemm_quant_mode.name,
                    architecture,
                    dsa_backend,
                    depth,
                ),
                slice_fn,
                _sol,
                depth=depth,
            )
            latency, _ = util_empirical.estimate(sol_time, query, grid)
            return latency

        if database_mode is None:
            database_mode = database._default_database_mode
        if database_mode == common.DatabaseMode.SOL:
            sol_latency = get_sol(b, s, prefix, num_heads, kvcache_quant_mode, fmha_quant_mode)[0]
            return PerformanceResult(sol_latency, energy=0.0, source="sol")
        elif database_mode == common.DatabaseMode.SOL_FULL:
            return get_sol(b, s, prefix, num_heads, kvcache_quant_mode, fmha_quant_mode)
        elif database_mode == common.DatabaseMode.EMPIRICAL:
            emp_latency = get_empirical(b, s, prefix, num_heads, kvcache_quant_mode, fmha_quant_mode)
            return PerformanceResult(emp_latency, energy=0.0, source="empirical")

        cls.load_data(database)

        def missing_context_dsa_error() -> PerfDataNotAvailableError:
            return PerfDataNotAvailableError(
                f"Context DSA module data not available for system='{database.system}', "
                f"backend='{database.backend}', version='{database.version}', architecture='{architecture}', "
                f"fmha_quant_mode={fmha_quant_mode}, kvcache_quant_mode={kvcache_quant_mode}, "
                f"gemm_quant_mode={gemm_quant_mode}, num_heads={num_heads}, s={s}, prefix={prefix}, b={b}. "
                "Missing silicon data for the requested lookup."
            )

        try:
            dsa_module_data = (
                database._context_dsa_module_skip_data if skip_indexer else database._context_dsa_module_data
            )
            if dsa_module_data is None:
                raise PerfDataNotAvailableError(
                    f"Context DSA module {'skip_indexer ' if skip_indexer else ''}perf data not loaded for "
                    f"system='{database.system}', backend='{database.backend}', version='{database.version}'."
                )
            try:
                dsa_dict = util_empirical.require_data_slice(
                    dsa_module_data,
                    fmha_quant_mode,
                    kvcache_quant_mode,
                    gemm_quant_mode,
                    architecture,
                )
            except PerfDataNotAvailableError as exc:
                raise missing_context_dsa_error() from exc
            dsa_dict = _select_dsa_backend(dsa_dict, dsa_backend)

            # Resolve on the RAW table via perf_interp. New collections carry the
            # past-KV axis ([heads][prefix][seq][batch] -> 4-axis grid); legacy
            # tables are [heads][seq][batch] and answer prefix=0 only. DSA is
            # densely sampled and the topk-saturation knee is itself collected,
            # so RAW blends everywhere (LOO: plain crossing ties/beats the old
            # topk-piecewise on every real config); out-of-range (incl. prefix)
            # is util-hold with the regime-aware SOL. The raw-vs-expanded table
            # juggling existed to work around load-time pre-expansion (gone).
            has_prefix_axis = _dsa_module_has_prefix_axis(dsa_dict)
            if architecture == "GlmMoeDsaForCausalLM" and not has_prefix_axis:
                raise missing_context_dsa_error()
            try:
                if has_prefix_axis:
                    config = perf_interp.OpInterpConfig(
                        axes=("num_heads", "prefix", "seq_len", "batch"),
                        resolver=perf_interp.Grid(),
                        sol_fn=lambda n_v, p_v, s_v, b_v: get_sol(
                            b_v, s_v, p_v, n_v, kvcache_quant_mode, fmha_quant_mode
                        )[0],
                    )
                    result = perf_interp.query(config, dsa_dict, num_heads, prefix, s, b)
                else:
                    if prefix:
                        raise missing_context_dsa_error()
                    config = perf_interp.OpInterpConfig(
                        axes=("num_heads", "seq_len", "batch"),
                        resolver=perf_interp.Grid(),
                        sol_fn=lambda n_v, s_v, b_v: get_sol(b_v, s_v, 0, n_v, kvcache_quant_mode, fmha_quant_mode)[0],
                    )
                    result = perf_interp.query(config, dsa_dict, num_heads, s, b)
                latency = perf_interp.get_value(result, "latency")
                energy = perf_interp.get_value(result, "energy")
            except InterpolationDataNotAvailableError as exc:
                raise missing_context_dsa_error() from exc
            return database._interp_pr(latency, energy=energy)
        except (PerfDataNotAvailableError, InterpolationDataNotAvailableError) as e:
            if database_mode == common.DatabaseMode.HYBRID:
                logger.debug(
                    f"Failed to query context DSA module for {b=}, {s=}, {prefix=}, {num_heads=}, "
                    f"{index_n_heads=}, {index_head_dim=}, {index_topk=}; using empirical"
                )
                latency = get_empirical(b, s, prefix, num_heads, kvcache_quant_mode, fmha_quant_mode)
                return PerformanceResult(latency, energy=0.0, source="empirical")
            if isinstance(e, PerfDataNotAvailableError):
                logger.warning(str(e))
                raise
            message = _format_dsa_unavailable_message(
                "Context",
                e,
                b=b,
                s=s,
                prefix=prefix,
                num_heads=num_heads,
                architecture=architecture,
                index_n_heads=index_n_heads,
                index_head_dim=index_head_dim,
                index_topk=index_topk,
            )
            logger.warning(message)
            raise PerfDataNotAvailableError(message) from None
        except Exception:
            logger.exception(
                f"Failed to query context DSA module for {b=}, {s=}, {prefix=}, {num_heads=}, "
                f"{index_n_heads=}, {index_head_dim=}, {index_topk=}, "
                f"{kvcache_quant_mode=}, {fmha_quant_mode=}, {database_mode=}."
            )
            raise

    # ------------------------------------------------------------------
    # Op contract
    # ------------------------------------------------------------------

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """Query context DSA latency with energy data.

        GLM-5.2: the shared-index group has some full layers (compute indexer)
        and the rest skip (reuse it). Per-layer cost is amortized
        full_frac*full + (1-full_frac)*skip, where full_frac is the EXACT
        indexer-computing-layer fraction (honors index_skip_topk_offset/pattern;
        GLM-5.2 = 21/78 = 0.2692, not 1/freq=0.25), using the directly-collected
        skip_indexer table (no delta). full_frac==1.0 (DeepSeek-V3.2 / GLM-5) ->
        pure full, unchanged. CP and non-CP share the same amortization (both full
        and skip carry the same scale_factor, so the weighted sum is exact)."""
        batch_size = kwargs.get("batch_size")
        isl = kwargs.get("s")
        prefix = kwargs.get("prefix", 0)
        w = self._full_frac

        if self._cp_size and self._cp_size > 1:
            full = self._query_cp(database, batch_size, isl, prefix)
            if w >= 1.0:
                return full
            skip = self._query_cp(database, batch_size, isl, prefix, skip_indexer=True)
            return self._amortize(full, skip, w)

        def _q(skip_indexer):
            return database.query_context_dsa_module(
                b=batch_size,
                s=isl,
                prefix=prefix,
                num_heads=self._num_heads,
                kvcache_quant_mode=self._kvcache_quant_mode,
                fmha_quant_mode=self._fmha_quant_mode,
                gemm_quant_mode=self._gemm_quant_mode,
                architecture=self._architecture,
                skip_indexer=skip_indexer,
            )

        full = _q(False)
        if w >= 1.0:
            lat = float(full)
            energy = full.energy
        else:
            skip = _q(True)
            lat = w * float(full) + (1.0 - w) * float(skip)
            energy = w * full.energy + (1.0 - w) * skip.energy
        return PerformanceResult(
            lat * self._scale_factor,
            energy=energy * self._scale_factor,
            source=getattr(full, "source", "silicon"),
        )

    @staticmethod
    def _amortize(full: PerformanceResult, skip: PerformanceResult, w: float) -> PerformanceResult:
        """w*full + (1-w)*skip on already-scaled PerformanceResults."""
        return PerformanceResult(
            w * float(full) + (1.0 - w) * float(skip),
            energy=w * getattr(full, "energy", 0.0) + (1.0 - w) * getattr(skip, "energy", 0.0),
            source=getattr(full, "source", "silicon"),
        )

    # ------------------------------------------------------------------
    # Context-Parallel (CP) prefill model — GLM-5 DSA only.
    # See docs/CONTEXT_PARALLEL_DSA_MODELING.md. Per-card =
    #   base dsa_module(isl/cp, bf16-KV row)
    #   + mqa(isl/cp)*(cp-1)                          (mqa ∝ isl², xcp identity)
    #   - [topk_full(flat) - topk_full(top_last)]/cp  (topk ∝ full/cp; module is dummy/flat)
    #   + AG_KV + AG_LSE                              (the two small attention all-gathers)
    # AG_hidden + RS belong to the MoE comm (modeled by MoEDispatch), not here.
    # ------------------------------------------------------------------
    _glm5_sparse_cache: ClassVar[dict] = {}

    def _query_cp(
        self, database: PerfDatabase, b: int, isl: int, prefix: int, skip_indexer: bool = False
    ) -> PerformanceResult:
        """CP (round-robin split) per-layer DSA, new strategy (2026-06-11):

            result = dsa(isl/cp, prefix)
                   + [mqa(isl, prefix)/cp      - mqa(isl/cp, prefix)]
                   + [topk_last(isl, prefix)/cp - topk_flat(isl/cp, prefix)]
                   + AG_KV + AG_LSE

        The per-card monolithic dsa_module(isl/cp, prefix) is the base; its
        internal mqa(isl/cp,prefix) and topk_flat(isl/cp,prefix) are swapped out
        by the two deltas, leaving proj + dsa_attn (both prefix-independent: proj
        by construction, dsa_attn topk-capped to index_topk) plus the CP-correct
        full-chunk mqa/topk_last divided across cp ranks. All sub-kernels are
        looked up at the REAL (q_len, prefix) shape — the parquet ``step`` column
        IS the prefix (past_kv) length.
        """
        cp = self._cp_size
        per_card = max(1, -(-isl // cp))  # ceil: critical path = busiest CP rank
        sp = self._load_glm5_sparse(database, self._architecture, self._num_heads)
        g = sp.get("_2d", {})
        file_prefix = _dsa_sparse_file_prefix(self._architecture)
        # Fail fast: CP DSA modeling REQUIRES the sparse mqa/topk tables for
        # the mqa/topk_last deltas. _lookup_2d clamps isl + interp/extrapolates
        # step, so a None below means the table is absent entirely (parquet not
        # collected) -- degrading silently to dsa_base would hide that.
        # skip_indexer layers carry NO indexer -> no mqa/topk deltas needed, so
        # don't require the sparse tables for them.
        missing = [] if skip_indexer else [k for k in ("mqa", "topk_last", "topk_flat") if not g.get(k)]
        if missing:
            raise PerfDataNotAvailableError(
                f"DSA CP modeling needs sparse tables {missing} for "
                f"{self._architecture} (num_heads={self._num_heads}); "
                f"collect {file_prefix}_mqa_logits/{file_prefix}_topk first."
            )
        # Base: per-card monolithic dsa_module at (per_card, prefix), follows the
        # run's kv_cache_dtype like the non-CP path.
        dsa_base = float(
            database.query_context_dsa_module(
                b=b,
                s=per_card,
                prefix=prefix,
                num_heads=self._num_heads,
                kvcache_quant_mode=self._kvcache_quant_mode,
                fmha_quant_mode=self._fmha_quant_mode,
                gemm_quant_mode=self._gemm_quant_mode,
                architecture=self._architecture,
                dsa_backend="flashmla_kv",
                skip_indexer=skip_indexer,
            )
        )
        # Look the sparse sub-kernels up at the REAL batch b (the bs slice carries
        # the measured bs=b latency), so the delta matches dsa_base (queried at b)
        # WITHOUT an external x b linearity assumption.
        mqa_tab = self._bs_slice(g.get("mqa", {}), b)
        tl_tab = self._bs_slice(g.get("topk_last", {}), b)
        tf_tab = self._bs_slice(g.get("topk_flat", {}), b)
        mqa_full = self._lookup_2d(mqa_tab, isl, prefix)
        mqa_perc = self._lookup_2d(mqa_tab, per_card, prefix)
        tl_full = self._lookup_2d(tl_tab, isl, prefix)
        tf_perc = self._lookup_2d(tf_tab, per_card, prefix)
        latency = dsa_base
        # skip layers reuse a sibling's topk index: no per-layer mqa/topk, so no
        # full/cp deltas — just the per-card skip base + the attention all-gathers.
        if not skip_indexer and None not in (mqa_full, mqa_perc, tl_full, tf_perc):
            delta_mqa = mqa_full / cp - mqa_perc
            delta_topk = tl_full / cp - tf_perc
            latency += delta_mqa + delta_topk
        # CP communication: AG of compressed KV (kv_lora+rope) + AG of LSE (kv_lora).
        dims = DSA_MODEL_DIMS.get(self._architecture, {})
        kv_lora = dims.get("kv_lora_rank", 512)
        rope = dims.get("qk_rope_head_dim", 64)
        index_head_dim = dims.get("index_head_dim", 128)
        # CP attention all-gather, verified by instrumenting sglang cp_utils
        # (cp_all_gather_rerange_output): per current-chunk tokens (isl, not
        # isl+prefix; prefix KV is already replicated), bf16. Two gathers:
        #   - compressed KV latent: kv_lora_rank + qk_rope_head_dim (= 576)
        #   - DSA indexer key: index_head_dim (= 128)
        # (The hidden_states 6144 AG/RS is the MoE token dispatch, modeled in
        # context_moe_pre/post_dispatch, not here.)
        # ag_kv = MQA-stage gather: DSA indexer key (index_head_dim), bf16.
        # ag_lse = FMHA-stage gather: compressed KV latent (kv_lora_rank +
        # qk_rope_head_dim), bf16. Both over the current chunk (isl), verified by
        # instrumenting sglang (dsa_indexer index_key 128; deepseek_v2
        # rebuild_cp_kv_cache latent 576).
        # x b: the all-gather moves b sequences' worth of current-chunk KV.
        # A skip-indexer (reuse) layer never runs the per-layer indexer, so it
        # does not all-gather the DSA indexer key -- only the MLA compressed-KV/LSE
        # gather remains. Don't charge the indexer-key AG to skip layers.
        ag_kv = (
            0.0
            if skip_indexer
            else float(database.query_nccl(common.CommQuantMode.half, cp, "all_gather", b * isl * index_head_dim))
        )
        ag_lse = float(database.query_nccl(common.CommQuantMode.half, cp, "all_gather", b * isl * (kv_lora + rope)))
        latency += ag_kv + ag_lse
        return PerformanceResult(latency * self._scale_factor, energy=0.0, source="estimated")

    @classmethod
    def _load_glm5_sparse(cls, database: PerfDatabase, architecture: str, num_heads: int) -> dict:
        """Load DSA sparse sub-kernel tables (mqa / topk / dsa_attn) for the CP
        composition path. Architecture-keyed: GLM-5 reads ``glm5_*`` filtered to
        its native num_heads (64); DeepSeek-V3.2 reads ``dsv32_*`` filtered to
        128. Same kernels, different shapes -- the delta strategy in _query_cp is
        identical (full/cp mqa + flat->top_last topk). dsa_attn is optional (not
        used by the delta; DSV3.2 only collects mqa + topk)."""
        key = (cls._cache_key(database), architecture, num_heads)
        if key in cls._glm5_sparse_cache:
            return cls._glm5_sparse_cache[key]
        import os

        import pandas as pd

        fp = _dsa_sparse_file_prefix(architecture)
        system_data_root = os.path.join(database.systems_root, database.system_spec["data_dir"])
        # Resolve the version DIR once (family dir first, else legacy) via a
        # representative filename, then apply the existing prefix-read logic
        # within that dir -- the other glm5_*/dsv32_* siblings live alongside it.
        # The three sparse tables are collected as independent ops, so anchor on
        # whichever sibling exists first: a dir may hold topk/dsa_attn without
        # mqa, and anchoring on mqa alone would fall back to the legacy dir and
        # silently drop the present siblings.
        candidates = [
            resolve_op_data_path(
                system_data_root,
                database.backend,
                database.version,
                f"{fp}_{table}_module_perf.parquet",
            )
            for table in ("mqa_logits", "topk", "dsa_attn")
        ]
        anchor = next((path for path in candidates if os.path.exists(path)), candidates[0])
        data_dir = os.path.dirname(anchor)
        # Grids keyed by batch_size -> {(isl, step): latency}. Keeping every
        # collected bs lets _query_cp look up the sparse deltas at the REAL
        # batch (real measured bs=b latency), instead of scaling a bs=1 value
        # by b (which would over-count: launch overhead amortises with batch).
        out = {}
        out2d = {"mqa": {}, "topk_last": {}, "topk_flat": {}, "dsa_attn": {}}

        def _read(fn):
            p = os.path.join(data_dir, fn)
            return pd.read_parquet(p) if os.path.exists(p) else None

        def _heads(df):
            return df[df["num_heads"] == num_heads] if "num_heads" in df else df

        def _put(tab, r):
            tab.setdefault(int(r["batch_size"]), {})[(int(r["isl"]), int(r["step"]))] = float(r["latency"])

        mdf = _read(f"{fp}_mqa_logits_module_perf.parquet")
        if mdf is not None:
            for _, r in _heads(mdf).iterrows():
                _put(out2d["mqa"], r)
        tdf = _read(f"{fp}_topk_module_perf.parquet")
        if tdf is not None:
            for _, r in _heads(tdf).iterrows():
                mode = "topk_flat" if str(r.get("score_mode", "")) == "flat" else "topk_last"
                _put(out2d[mode], r)
        adf = _read(f"{fp}_dsa_attn_module_perf.parquet")
        if adf is not None:
            for _, r in _heads(adf).iterrows():
                _put(out2d["dsa_attn"], r)
        out["_2d"] = out2d
        cls._glm5_sparse_cache[key] = out
        return out

    @staticmethod
    def _bs_slice(by_bs: dict, b: int) -> dict:
        """Pick the collected-batch slice nearest to ``b`` from a {bs: {(isl,step):lat}}
        table. Exact match when ``b`` was collected (the common case); otherwise the
        nearest collected batch."""
        if not by_bs:
            return {}
        if b in by_bs:
            return by_bs[b]
        return by_bs[min(by_bs, key=lambda x: abs(x - b))]

    @staticmethod
    def _lookup_2d(table, isl, step):
        """Lookup {(isl,step): latency} at a fixed isl (exact grid value), linear
        interp/extrap on step. Used by the CP sub-kernel composition."""
        if not table:
            return None
        isls = sorted({i for (i, _s) in table})
        if isl > isls[-1]:
            raise PerfDataNotAvailableError(
                f"DSA CP: isl={isl} exceeds the collected sparse-kernel grid "
                f"(max isl={isls[-1]}); mqa/topk scale super-linearly with isl, so "
                f"clamping the isl axis would silently under-estimate. Re-collect with "
                f"AIC_CHUNKED_PREFILL_SIZE >= {isl} "
                f"(docs/CONTEXT_PARALLEL_DSA_MODELING.md §9.1)."
            )
        use_isl = isl if isl in isls else min(isls, key=lambda x: abs(x - isl))
        steps = sorted(st for (i, st) in table if i == use_isl)
        if not steps:
            return None
        if (use_isl, step) in table:
            return table[(use_isl, step)]
        lo = max([st for st in steps if st <= step], default=steps[0])
        hi = min([st for st in steps if st >= step], default=steps[-1])
        if lo == hi:
            return table[(use_isl, lo)]
        a = table[(use_isl, lo)]
        bb = table[(use_isl, hi)]
        return a + (bb - a) * (step - lo) / (hi - lo)

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


class GenerationDSAModule(Operation):
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

    def __init__(
        self,
        name: str,
        scale_factor: float,
        num_heads: int,
        kv_cache_dtype: common.KVCacheQuantMode,
        gemm_quant_mode: common.GEMMQuantMode,
        architecture: str = "DeepseekV32ForCausalLM",
        index_topk_freq: int = 1,
        dsa_full_layer_fraction: float | None = None,
    ) -> None:
        super().__init__(name, scale_factor)
        self._num_heads = num_heads
        self._kv_cache_dtype = kv_cache_dtype
        self._gemm_quant_mode = gemm_quant_mode
        self._architecture = architecture
        # GLM-5.2 shared-index amortization (see ContextDSAModule): exact
        # full-layer fraction; fall back to 1/freq. full_frac==1.0 => pure full.
        self._index_topk_freq = max(1, int(index_topk_freq or 1))
        self._full_frac = (
            float(dsa_full_layer_fraction) if dsa_full_layer_fraction is not None else 1.0 / self._index_topk_freq
        )
        self._weights = 0.0

    # ------------------------------------------------------------------
    # Data ownership
    # ------------------------------------------------------------------

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Loads generation_dsa_module data, preserves the raw
        measured rows, applies the legacy grid extrapolation to a working copy,
        and binds both views on ``database``."""
        import os

        from aiconfigurator_core.sdk.perf_database import LoadedOpData, PerfDataFilename

        key = cls._cache_key(database)
        system_data_root = os.path.join(database.systems_root, database.system_spec["data_dir"])
        primary_path = resolve_op_data_path(
            system_data_root, database.backend, database.version, PerfDataFilename.dsa_generation_module.value
        )
        sources = database._build_op_sources(PerfDataFilename.dsa_generation_module, primary_path, system_data_root)
        if key not in cls._data_cache:
            cls._data_cache[key] = LoadedOpData(
                load_generation_dsa_module_data(sources, op_kind="full"),
                PerfDataFilename.dsa_generation_module,
                primary_path,
            )
            # No load-time grid pre-expansion: queries resolve on the RAW grid
            # via perf_interp (its util-hold IS the boundary-util anchoring).
            cls._raw_data_cache[key] = cls._data_cache[key]
            cls._record_load()

        # skip_indexer rows share the same file (op_name tag); load with op_kind="skip".
        if key not in cls._skip_data_cache:
            skip_dict = load_generation_dsa_module_data(sources, op_kind="skip")
            if skip_dict:
                cls._skip_data_cache[key] = LoadedOpData(
                    skip_dict, PerfDataFilename.dsa_generation_module, primary_path
                )
                cls._raw_skip_data_cache[key] = cls._skip_data_cache[key]
            else:
                cls._skip_data_cache[key] = None
                cls._raw_skip_data_cache[key] = None

        if "_generation_dsa_module_data" not in database.__dict__:
            database._generation_dsa_module_data = cls._data_cache[key]
            database._raw_generation_dsa_module_data = cls._raw_data_cache[key]
        if "_generation_dsa_module_skip_data" not in database.__dict__:
            database._generation_dsa_module_skip_data = cls._skip_data_cache[key]
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

    @classmethod
    def _query_generation_dsa_module_table(
        cls,
        database: PerfDatabase,
        b: int,
        s: int,
        num_heads: int,
        kv_cache_dtype: common.KVCacheQuantMode,
        gemm_quant_mode: common.GEMMQuantMode = common.GEMMQuantMode.bfloat16,
        database_mode: common.DatabaseMode | None = None,
        *,
        architecture: str | None = None,
        index_n_heads: int | None = None,
        index_head_dim: int | None = None,
        index_topk: int | None = None,
        dsa_backend: str = "trtllm",
        skip_indexer: bool = False,
    ):
        """Query generation DSA module table.
        ``skip_indexer=True`` reads the GLM-5.2 reuse-layer table."""
        from aiconfigurator_core.sdk.perf_database import PerfDataNotAvailableError

        # ``DEFAULT_DSA_ARCHITECTURE`` and ``DSA_MODEL_DIMS`` live at module
        # level in this file — use them directly rather than round-tripping
        # through ``perf_database``'s backward-compat re-export.

        if architecture is None:
            architecture = DEFAULT_DSA_ARCHITECTURE

        dims = DSA_MODEL_DIMS.get(architecture, DSA_MODEL_DIMS[DEFAULT_DSA_ARCHITECTURE])
        hidden_size = dims["hidden_size"]
        q_lora = dims["q_lora_rank"]
        kv_lora = dims["kv_lora_rank"]
        qk_nope = dims["qk_nope_head_dim"]
        qk_rope = dims["qk_rope_head_dim"]
        v_dim = dims["v_head_dim"]
        if index_n_heads is None:
            index_n_heads = dims["index_n_heads"]
        if index_head_dim is None:
            index_head_dim = dims["index_head_dim"]
        if index_topk is None:
            index_topk = dims["index_topk"]
        qk_head_dim = qk_nope + qk_rope
        attn_head_dim = kv_lora + qk_rope

        def get_sol(
            b: int, s: int, num_heads: int, kv_cache_dtype: common.KVCacheQuantMode
        ) -> tuple[float, float, float]:
            """SOL estimate for generation DSA module (1 token per request).

            Ops split into GEMM group (gemm_quant_mode) and attention group
            (fmha derived from kv_cache_dtype).
            """
            fmha_mode = common.FMHAQuantMode.bfloat16

            tokens = b
            proj_out = q_lora + kv_lora + qk_rope + index_head_dim
            effective_kv = min(s, index_topk)

            gemm_group_ops = (
                2 * tokens * hidden_size * proj_out
                + 2 * tokens * q_lora * num_heads * qk_head_dim
                + 2 * tokens * q_lora * index_n_heads * index_head_dim
                + 2 * tokens * hidden_size * index_n_heads
                + 2 * tokens * num_heads * v_dim * hidden_size
                + 2 * num_heads * tokens * qk_nope * kv_lora
                + 2 * num_heads * tokens * kv_lora * v_dim
            )

            indexer_logits_ops = 2 * tokens * index_n_heads * index_head_dim * s
            sparse_attn_ops = 2 * tokens * num_heads * (attn_head_dim + kv_lora) * effective_kv

            gemm_weight_bytes = (
                hidden_size * proj_out
                + q_lora * num_heads * qk_head_dim
                + q_lora * index_n_heads * index_head_dim
                + hidden_size * index_n_heads
                + num_heads * v_dim * hidden_size
            ) * gemm_quant_mode.value.memory
            indexer_entry_bytes = common.indexer_cache_entry_bytes(index_head_dim)
            indexer_cache_bytes = b * s * indexer_entry_bytes
            kv_cache_bytes = b * effective_kv * attn_head_dim * kv_cache_dtype.value.memory
            total_mem = gemm_weight_bytes + indexer_cache_bytes + kv_cache_bytes

            from aiconfigurator_core.sdk.operations.gemm import GEMM

            gemm_flops = GEMM._get_quant_tc_flops(database.system_spec, gemm_quant_mode)
            indexer_fp8_flops = GEMM._get_quant_tc_flops(database.system_spec, common.FMHAQuantMode.fp8)
            attn_flops = GEMM._get_quant_tc_flops(database.system_spec, fmha_mode)

            sol_math = (
                gemm_group_ops / gemm_flops + indexer_logits_ops / indexer_fp8_flops + sparse_attn_ops / attn_flops
            ) * 1000
            sol_mem = total_mem / database.system_spec["gpu"]["mem_bw"] * 1000
            sol_time = max(sol_math, sol_mem)
            return sol_time, sol_math, sol_mem

        def get_empirical(b: int, s: int, num_heads: int, kv_cache_dtype: common.KVCacheQuantMode) -> float:
            # SOL / util, util read best-effort from own collected data (the
            # (num_heads, b, s) grid for this slice); raises EmpiricalNotImplementedError if no data.
            sol_time = get_sol(b, s, num_heads, kv_cache_dtype)[0]

            def _slice():
                cls.load_data(database)
                # EMPIRICAL utilization is calibrated from measured rows only.
                # Using the extrapolated working table here would make a
                # synthesized SILICON latency masquerade as calibration data.
                raw_wrapper = getattr(
                    database,
                    "_raw_generation_dsa_module_skip_data" if skip_indexer else "_raw_generation_dsa_module_data",
                    None,
                )
                wrapper = (
                    raw_wrapper
                    if raw_wrapper is not None and getattr(raw_wrapper, "loaded", True)
                    else (
                        database._generation_dsa_module_skip_data
                        if skip_indexer
                        else database._generation_dsa_module_data
                    )
                )
                if wrapper is None:
                    raise PerfDataNotAvailableError("Generation DSA module data is not loaded.")
                arch_node = util_empirical.require_data_slice(
                    wrapper,
                    kv_cache_dtype,
                    gemm_quant_mode,
                    architecture,
                )
                # ...[architecture][dsa_backend][num_heads]...; descend past the backend
                # axis like the silicon path so the grid resolves the num_heads axis.
                selected = _select_dsa_backend(arch_node, dsa_backend)
                if selected is None:
                    raise PerfDataNotAvailableError(f"No generation DSA data for backend {dsa_backend!r}.")
                return selected

            try:
                data_slice = _slice()
            except PerfDataNotAvailableError:
                # Match ``grid_for``'s best-effort contract: unavailable table
                # data is reported by ``estimate`` as an empirical coverage
                # miss, not leaked as a SILICON file-loading exception.
                data_slice = None

            if data_slice is not None and num_heads in data_slice:
                # ``num_heads`` is a TP/model-shape identity, not an axis that
                # should drift merely because another TP has longer sequence
                # coverage.  Stay on the exact head slice whenever it exists;
                # only use cross-head nearest-neighbour when the slice is absent.
                grid = util_empirical.grid_for(
                    (
                        "gen_dsa_exact_heads",
                        database.system,
                        database.backend,
                        database.version,
                        kv_cache_dtype.name,
                        gemm_quant_mode.name,
                        architecture,
                        dsa_backend,
                        num_heads,
                    ),
                    lambda: data_slice[num_heads],
                    lambda c: get_sol(c[0], c[1], num_heads, kv_cache_dtype)[0],  # c = (b, s)
                    depth=2,
                )
                query = (b, s)
            elif data_slice is not None:
                grid = util_empirical.grid_for(
                    (
                        "gen_dsa",
                        database.system,
                        database.backend,
                        database.version,
                        kv_cache_dtype.name,
                        gemm_quant_mode.name,
                        architecture,
                        dsa_backend,
                    ),
                    lambda: data_slice,
                    lambda c: get_sol(c[1], c[2], c[0], kv_cache_dtype)[0],  # c = (num_heads, b, s)
                    depth=3,
                )
                query = (num_heads, b, s)
            else:
                grid = None
                query = (num_heads, b, s)
            latency, _ = util_empirical.estimate(sol_time, query, grid)
            return latency

        if database_mode is None:
            database_mode = database._default_database_mode
        if database_mode == common.DatabaseMode.SOL:
            sol_latency = get_sol(b, s, num_heads, kv_cache_dtype)[0]
            return PerformanceResult(sol_latency, energy=0.0, source="sol")
        elif database_mode == common.DatabaseMode.SOL_FULL:
            return get_sol(b, s, num_heads, kv_cache_dtype)
        elif database_mode == common.DatabaseMode.EMPIRICAL:
            emp_latency = get_empirical(b, s, num_heads, kv_cache_dtype)
            return PerformanceResult(emp_latency, energy=0.0, source="empirical")

        cls.load_data(database)

        def missing_generation_dsa_error() -> PerfDataNotAvailableError:
            return PerfDataNotAvailableError(
                f"Generation DSA module data not available for system='{database.system}', "
                f"backend='{database.backend}', version='{database.version}', architecture='{architecture}', "
                f"kv_cache_dtype={kv_cache_dtype}, gemm_quant_mode={gemm_quant_mode}, "
                f"num_heads={num_heads}, s={s}, b={b}. "
                "Missing silicon data for the requested lookup."
            )

        try:
            dsa_module_data = (
                database._generation_dsa_module_skip_data if skip_indexer else database._generation_dsa_module_data
            )
            if dsa_module_data is None:
                raise PerfDataNotAvailableError(
                    f"Generation DSA module {'skip_indexer ' if skip_indexer else ''}perf data not loaded for "
                    f"system='{database.system}', backend='{database.backend}', version='{database.version}'."
                )
            try:
                try:
                    dsa_dict = util_empirical.require_data_slice(
                        dsa_module_data,
                        kv_cache_dtype,
                        gemm_quant_mode,
                        architecture,
                    )
                except PerfDataNotAvailableError as exc:
                    raise missing_generation_dsa_error() from exc
                dsa_dict = _select_dsa_backend(dsa_dict, dsa_backend)

                # Resolve on the RAW [heads][batch][seq] table via perf_interp
                # (raw generation grid: ~linear in seq; the topk-saturation knee
                # is collected, so blends never smooth it away). Out-of-range
                # seq/batch is util-hold -- the former hand-rolled
                # boundary_util_value did exactly this, point for point.
                config = perf_interp.generation_grid_config(
                    sol_fn=lambda n_v, b_v, s_v: get_sol(b_v, s_v, n_v, kv_cache_dtype)[0]
                )
                result = perf_interp.query(config, dsa_dict, num_heads, b, s)
                latency = perf_interp.get_value(result, "latency")
                energy = perf_interp.get_value(result, "energy")
            except InterpolationDataNotAvailableError as exc:
                raise missing_generation_dsa_error() from exc
            return database._interp_pr(latency, energy=energy)
        except (PerfDataNotAvailableError, InterpolationDataNotAvailableError) as e:
            if database_mode == common.DatabaseMode.HYBRID:
                logger.debug(
                    f"Failed to query generation DSA module for {b=}, {s=}, {num_heads=}, "
                    f"{index_n_heads=}, {index_head_dim=}, {index_topk=}; using empirical"
                )
                latency = get_empirical(b, s, num_heads, kv_cache_dtype)
                return PerformanceResult(latency, energy=0.0, source="empirical")
            if isinstance(e, PerfDataNotAvailableError):
                logger.warning(str(e))
                raise
            message = _format_dsa_unavailable_message(
                "Generation",
                e,
                b=b,
                s=s,
                num_heads=num_heads,
                architecture=architecture,
                index_n_heads=index_n_heads,
                index_head_dim=index_head_dim,
                index_topk=index_topk,
            )
            logger.warning(message)
            raise PerfDataNotAvailableError(message) from None
        except Exception:
            logger.exception(
                f"Failed to query generation DSA module for {b=}, {s=}, {num_heads=}, "
                f"{index_n_heads=}, {index_head_dim=}, {index_topk=}, "
                f"{kv_cache_dtype=}, {database_mode=}."
            )
            raise

    # ------------------------------------------------------------------
    # Op contract
    # ------------------------------------------------------------------

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """Query generation DSA latency with energy data."""
        beam_width = kwargs.get("beam_width")
        if beam_width != 1:
            raise ValueError(f"{self.__class__.__name__} only supports beam_width=1, got {beam_width}")
        batch_size = kwargs.get("batch_size")
        s = kwargs.get("s")
        w = self._full_frac

        def _q(skip_indexer):
            return database.query_generation_dsa_module(
                b=batch_size,
                s=s,
                num_heads=self._num_heads,
                kv_cache_dtype=self._kv_cache_dtype,
                gemm_quant_mode=self._gemm_quant_mode,
                architecture=self._architecture,
                skip_indexer=skip_indexer,
            )

        full = _q(False)
        if w >= 1.0:
            lat = float(full)
            energy = full.energy
        else:
            # GLM-5.2 shared-index amortization (see ContextDSAModule.query).
            skip = _q(True)
            lat = w * float(full) + (1.0 - w) * float(skip)
            energy = w * full.energy + (1.0 - w) * skip.energy
        return PerformanceResult(
            lat * self._scale_factor,
            energy=energy * self._scale_factor,
            source=getattr(full, "source", "silicon"),
        )

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


# ─────────────────────────────────────────────────────────
# CSV loaders (moved here from perf_database.py so each op family owns its data + parser)
# ─────────────────────────────────────────────────────────


def _read_dsa_row_sources(file_or_sources):
    """Read rows while retaining priority-source boundaries.

    DSA files historically used last-row-wins for duplicates within one file.
    Shared-layer inputs add a second requirement: an earlier source (the active
    stack) must outrank every later sibling source. ``_read_filtered_rows``
    intentionally flattens sources, so DSA keeps the groups here and applies
    those two rules independently.
    """
    if isinstance(file_or_sources, str):
        rows = _read_filtered_rows(file_or_sources)
        return None if rows is None else [rows]

    row_sources = []
    any_source_exists = False
    for source in file_or_sources:
        rows = _read_filtered_rows([source])
        if rows is None:
            continue
        any_source_exists = True
        row_sources.append(rows)
    return row_sources if any_source_exists else None


def load_context_dsa_module_data(dsa_file: str, op_kind: str = "full"):
    """
    Load context DSA data.

    Dict structure:
        data[fmha_quant_mode][kv_cache_quant_mode][gemm_quant_mode][architecture][dsa_backend][num_heads][prefix][s][b]

    Quant modes are the outermost keys so that ``_enum_key_names`` can
    directly extract supported FMHAQuantMode names (aligned with
    ``_context_attention_data``).  ``architecture`` (e.g.
    "DeepseekV32ForCausalLM", "GlmMoeDsaForCausalLM") selects the
    model-specific structural dimensions from ``DSA_MODEL_DIMS``.
    Legacy CSV rows without an ``architecture`` column default to
    "DeepseekV32ForCausalLM".

    Full and skip-indexer (GLM-5.2 reuse-layer) rows live in the SAME file,
    distinguished by the ``op_name`` column (``dsa_context_module`` vs
    ``dsa_context_module_skip_indexer``) — no extra column. ``op_kind`` selects
    which to keep: ``"full"`` (op_name without ``skip_indexer``) or ``"skip"``.
    """
    row_sources = _read_dsa_row_sources(dsa_file)
    if row_sources is None:
        logger.debug(f"DSA context data file {dsa_file} not found.")
        return None

    def _nest():
        return defaultdict(_nest)

    dsa_data = _nest()

    first_row = next((row for source_rows in row_sources for row in source_rows), None)
    has_power = first_row is not None and "power" in first_row
    seen_coordinates = set()

    for source_rows in row_sources:
        # Preserve legacy last-row-wins behavior within each source.
        source_values = {}
        for row in source_rows:
            # full vs skip-indexer share one file, split by op_name.
            if ("skip_indexer" in (row.get("op_name") or "")) != (op_kind == "skip"):
                continue
            num_heads = int(row["num_heads"])
            b = int(row["batch_size"])
            s = int(row["isl"])
            latency = float(row["latency"])
            power = float(row.get("power", 0.0)) if has_power else 0.0
            energy = power * latency

            arch = row.get("architecture", DEFAULT_DSA_ARCHITECTURE)
            step = row.get("step")
            step_missing = step is None or (isinstance(step, str) and step.strip() == "")
            if arch == "GlmMoeDsaForCausalLM" and step_missing:
                raise ValueError(
                    "GLM-5 context DSA module data requires a non-empty step column for prefix/past_kv length"
                )
            prefix = 0 if step_missing else int(step)
            gemm_mode = common.GEMMQuantMode[row["gemm_type"]]
            fmha_mode = common.FMHAQuantMode[row["mla_dtype"]]
            kv_dtype = common.KVCacheQuantMode[row["kv_cache_dtype"]]

            ks = row.get("kernel_source") or ""
            for dsa_backend in _dsa_kernel_source_buckets(ks, kv_dtype):
                coordinate = (fmha_mode, kv_dtype, gemm_mode, arch, dsa_backend, num_heads, prefix, s, b)
                source_values[coordinate] = {
                    "latency": latency,
                    "power": power,
                    "energy": energy,
                }

        # Sources are priority-ordered: active first, shared fallbacks later.
        for coordinate, value in source_values.items():
            if coordinate in seen_coordinates:
                continue
            seen_coordinates.add(coordinate)
            fmha_mode, kv_dtype, gemm_mode, arch, dsa_backend, num_heads, prefix, s, b = coordinate
            dsa_data[fmha_mode][kv_dtype][gemm_mode][arch][dsa_backend][num_heads][prefix][s][b] = value

    return dsa_data


def load_generation_dsa_module_data(dsa_file: str, op_kind: str = "full"):
    """
    Load generation DSA data.

    Dict structure:
        data[kv_cache_quant_mode][gemm_quant_mode][architecture][dsa_backend][num_heads][b][s]

    Quant modes are the outermost keys so that ``_enum_key_names`` can
    directly extract supported KVCacheQuantMode names (aligned with
    ``_generation_attention_data``).  ``architecture`` selects the
    model-specific structural dimensions from ``DSA_MODEL_DIMS``.
    Legacy CSV rows without an ``architecture`` column default to
    "DeepseekV32ForCausalLM".

    Full and skip-indexer rows share one file, split by the ``op_name`` column;
    ``op_kind`` ("full"/"skip") selects which to keep.
    """
    row_sources = _read_dsa_row_sources(dsa_file)
    if row_sources is None:
        logger.debug(f"DSA generation data file {dsa_file} not found.")
        return None

    def _nest():
        return defaultdict(_nest)

    dsa_data = _nest()

    first_row = next((row for source_rows in row_sources for row in source_rows), None)
    has_power = first_row is not None and "power" in first_row
    seen_coordinates = set()

    for source_rows in row_sources:
        # Preserve legacy last-row-wins behavior within each source.
        source_values = {}
        for row in source_rows:
            if ("skip_indexer" in (row.get("op_name") or "")) != (op_kind == "skip"):
                continue
            num_heads = int(row["num_heads"])
            b = int(row["batch_size"])
            s = int(row["isl"]) + int(row["step"])
            latency = float(row["latency"])
            power = float(row.get("power", 0.0)) if has_power else 0.0
            energy = power * latency

            arch = row.get("architecture", DEFAULT_DSA_ARCHITECTURE)
            gemm_mode = common.GEMMQuantMode[row["gemm_type"]]
            kv_dtype = common.KVCacheQuantMode[row["kv_cache_dtype"]]

            ks = row.get("kernel_source") or ""
            # Total decode length is the canonical coordinate even if two rows
            # decompose it into different isl/step pairs.
            for dsa_backend in _dsa_kernel_source_buckets(ks, kv_dtype):
                coordinate = (kv_dtype, gemm_mode, arch, dsa_backend, num_heads, b, s)
                source_values[coordinate] = {
                    "latency": latency,
                    "power": power,
                    "energy": energy,
                }

        # Sources are priority-ordered: active first, shared fallbacks later.
        for coordinate, value in source_values.items():
            if coordinate in seen_coordinates:
                continue
            seen_coordinates.add(coordinate)
            kv_dtype, gemm_mode, arch, dsa_backend, num_heads, b, s = coordinate
            dsa_data[kv_dtype][gemm_mode][arch][dsa_backend][num_heads][b][s] = value

    return dsa_data
