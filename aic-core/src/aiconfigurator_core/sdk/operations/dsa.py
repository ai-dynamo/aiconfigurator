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
from collections import defaultdict
from typing import TYPE_CHECKING, ClassVar

from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk.errors import PerfDataNotAvailableError
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


def dsa_block_weights_bytes(
    architecture: str,
    local_heads: int,
    projection_quant_modes: dict,
) -> float:
    """Per-layer DSA attention block weight bytes for one rank.

    ``projection_quant_modes`` maps the projection groups ``q``/``kv``/``o``/
    ``indexer`` to their GEMMQuantMode — per-checkpoint fact (e.g.
    DeepSeek-V3.2-NVFP4 keeps q/kv/indexer in BF16 but quantizes o_proj).
    q_a / kv_a(+mqa, incl. the indexer K projection) and the indexer
    projections are replicated across TP (single latent / single index);
    q_b, the absorbed kv_b (W_UK/W_UV) and o_proj shard by heads
    (``local_heads`` is already per-rank).
    """
    dims = DSA_MODEL_DIMS.get(architecture) or DSA_MODEL_DIMS[DEFAULT_DSA_ARCHITECTURE]
    h = dims["hidden_size"]
    q_lora = dims["q_lora_rank"]
    kv_lora = dims["kv_lora_rank"]
    qk = dims["qk_nope_head_dim"] + dims["qk_rope_head_dim"]
    v = dims["v_head_dim"]
    idx = dims["index_head_dim"] * dims["index_n_heads"]

    def _b(group: str) -> float:
        return projection_quant_modes[group].value.memory

    q_params = h * q_lora + q_lora * local_heads * qk
    kv_params = h * (kv_lora + dims["qk_rope_head_dim"]) + kv_lora * local_heads * (dims["qk_nope_head_dim"] + v)
    o_params = local_heads * v * h
    indexer_params = q_lora * idx + h * dims["index_n_heads"]
    return q_params * _b("q") + kv_params * _b("kv") + o_params * _b("o") + indexer_params * _b("indexer")


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
        attn_projection_quant_modes: dict | None = None,
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
        modes = attn_projection_quant_modes or dict.fromkeys(("q", "kv", "o", "indexer"), gemm_quant_mode)
        self._weights = dsa_block_weights_bytes(architecture, num_heads, modes)

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
            # via the engine's interpolation, so the raw wrapper is now an alias of the table.
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

    # ------------------------------------------------------------------
    # Op contract
    # ------------------------------------------------------------------

    _ENGINE_QUERY_SHAPE = "context"

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
        attn_projection_quant_modes: dict | None = None,
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
        modes = attn_projection_quant_modes or dict.fromkeys(("q", "kv", "o", "indexer"), gemm_quant_mode)
        self._weights = dsa_block_weights_bytes(architecture, num_heads, modes)

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
            # via the engine's interpolation (its util-hold IS the boundary-util anchoring).
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

    # ------------------------------------------------------------------
    # Op contract
    # ------------------------------------------------------------------

    _ENGINE_QUERY_SHAPE = "generation"

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
