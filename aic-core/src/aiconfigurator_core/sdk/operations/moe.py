# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MoE family (ISSUE-12 / ISSUE-13).

Op classes migrated from ``_legacy.py``:

- ``MoE`` (ISSUE-12) — Mixture-of-Experts compute op. Owns:
    * ``_moe_data`` — regular MoE table
    * ``_moe_low_latency_data`` — TRT-LLM low-latency NVFP4 kernel table
      (folded from the same perf table as the regular MoE data by the engine
      view — `table_view.rs::view_moe` returns the twin pair)
    * ``_wideep_context_moe_data`` — SGLang WideEP context MoE table
    * ``_wideep_generation_moe_data`` — SGLang WideEP generation MoE table
  Table selection (backend + ``moe_backend`` + ``num_tokens`` +
  ``quant_mode`` + ``is_gated``) happens in the engine; Python keeps all
  four tables loaded on the data plane.

- ``MoEDispatch`` (ISSUE-12) — MoE comm-cost op. Owns:
    * ``_wideep_deepep_normal_data`` — SGLang DeepEP normal-mode dispatch
    * ``_wideep_deepep_ll_data`` — SGLang DeepEP low-latency dispatch
  Dispatches at query time across NCCL, CustomAllReduce, TRT-LLM AllToAll,
  and SGLang DeepEP based on backend + ``_sm_version`` + ``_moe_backend``.

The retired ``TrtLLMWideEPMoE`` / ``TrtLLMWideEPMoEDispatch`` classes
(ISSUE-13) were deleted in the deprecation cleanup (AIC-1357): no model
constructed them and their query surfaces were engine-routed shims. Their
data-plane views (``_wideep_moe_compute_data`` / ``_trtllm_alltoall_data``)
are rehomed onto ``moe_comm.MoEExpertCompute`` / ``moe_comm.MoEAllToAll``.

Cache key matches every other migrated op:
``(systems_root, system, backend, version, enable_shared_layer)``. The
WideEP tables are loaded only when ``database.backend == "sglang"`` (MoE /
MoEDispatch SGLang-only WideEP slots); on other backends the corresponding
cache slot is ``None`` and consumers must guard.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

import aiconfigurator_core
import aiconfigurator_core._aiconfigurator_core as _core
from aiconfigurator_core.sdk.operations import util_empirical
from aiconfigurator_core.sdk.operations.base import OpShellKit

if TYPE_CHECKING:
    from aiconfigurator_core.sdk.perf_database import PerfDatabase


logger = logging.getLogger(__name__)


def _cache_key(database: PerfDatabase) -> tuple:
    """Shared cache key — same shape as every other migrated op family."""
    return (
        database.systems_root,
        database.system,
        database.backend,
        database.version,
        database.enable_shared_layer,
    )


# Per-quant achieved-util LEVEL e(q) for MoE, keyed by the (memory, compute) profile
# (which encodes (weight, activation) precision: memory∈{2,1,0.5}↔w{16,8,4},
# compute∈{1,2,4}↔a{16,8,4}). Used ONLY by the cross-PROFILE transfer tier: when the
# query quant has no data of any profile, borrow the nearest collected quant's util
# curve and rescale it by e(query)/e(ref). util is the achieved kernel efficiency
# (SOL already absorbs the coefficients); its LEVEL differs systematically by quant —
# 4-bit-weight kernels run far below their (higher) roofline, and efficiency rises mildly
# as activation precision drops. The RATIO e(query)/e(ref) is what matters and is
# ~stack-stable (≈10%; e.g. w4a16/fp8 = 0.17 on b200 vs 0.18 on h100). Levels are
# data-derived on b200/trtllm where collected, inferred from the structure otherwise.
# A SINGLE scalar per quant by design: the analytic SOL's compute/mem split is not
# trustworthy enough to calibrate per-component (validated — splitting blows up because
# the SOL attribution doesn't match the kernel's real bottleneck). Levels are relative
# and tunable; only ratios are consumed.
# PROJECTION of the engine's table (PR-6): the Rust
# `operators/moe.rs::MOE_QUANT_UTIL_LEVEL` is the single source (with the
# same per-row [data]/[inferred] provenance notes); rebuilding the dict from
# the FFI ends the two-sided sync discipline every new quant used to need.
_MOE_QUANT_UTIL_LEVEL: dict[tuple[float, float], float] = {
    (memory, compute): level for memory, compute, level in aiconfigurator_core.moe_quant_util_levels()
}
_MOE_QUANT_UTIL_DEFAULT = 0.30  # unlisted profile: mid-range relative level


def xprofile_util_level_known(quant_mode) -> bool:
    """Whether the MoE util-LEVEL table lists this quant's profile.

    The runtime ladder falls back to ``_MOE_QUANT_UTIL_DEFAULT`` for unlisted
    profiles; the validate gate deliberately does NOT (admitting a quant
    nobody calibrated would hide the missing level line the add-a-quant
    recipe requires), so it asks this instead of reaching into the table."""
    return util_empirical.quant_profile(quant_mode) in _MOE_QUANT_UTIL_LEVEL


# ───────────────────────────────────────────────────────────────────────
# MoE
# ───────────────────────────────────────────────────────────────────────


class MoE(_core.MoE, OpShellKit):
    """MoE operation with power tracking."""

    # CP-invariant: the A2A dispatch globalizes tokens across all (cp*ep) ranks,
    # so expert compute sees the full token set regardless of CP and deliberately
    # ignores ``seq_split`` (per-rank cp sharding does not reduce expert work).
    # Marked audited so the post-construction CP wiring (gemma4/hybrid) does not
    # trip the _CP_AWARE gate.
    _data_cache: ClassVar[dict] = {}
    _low_latency_data_cache: ClassVar[dict] = {}
    _wideep_context_data_cache: ClassVar[dict] = {}
    _wideep_generation_data_cache: ClassVar[dict] = {}

    @property
    def _seq_split(self) -> int:
        """CP shard factor. ``MoE`` is token-major CP-aware (the
        shim divides x by it). The Rust op carries no
        ``seq_split`` field (nothing crosses the wire), so the value lives in
        the shell instance ``__dict__`` — written by the models' CP wiring
        (``apply_cp_to_context_ops``) and read by the shim x-division.
        Survives pickle via the default object state (``__dict__``)."""
        return self.__dict__.get("_py_seq_split", 1)

    @_seq_split.setter
    def _seq_split(self, value: int) -> None:
        self.__dict__["_py_seq_split"] = int(value)

    # ------------------------------------------------------------------
    # Data ownership
    # ------------------------------------------------------------------

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Loads the regular MoE table (tuple of regular +
        low-latency) on all backends, and the SGLang WideEP context /
        generation MoE tables only when ``database.backend == "sglang"``.

        Binds these instance attributes for downstream consumers:
        - ``_moe_data``
        - ``_moe_low_latency_data``
        - ``_wideep_context_moe_data`` (None on non-SGLang)
        - ``_wideep_generation_moe_data`` (None on non-SGLang)
        """
        from aiconfigurator_core.sdk.engine_table_view import load_view
        from aiconfigurator_core.sdk.perf_database import PerfDataFilename

        key = cls._cache_key(database)
        if (
            key not in cls._data_cache
            or key not in cls._low_latency_data_cache
            or key not in cls._wideep_context_data_cache
            or key not in cls._wideep_generation_data_cache
        ):
            # Fetch every view into locals first so a failure on a later
            # fetch can't leave the caches half-populated — a retry would
            # then KeyError at the bind lines below, masking the real error
            # until clear_cache (same hardening as GEMM.load_data).
            # Regular MoE table — the engine folds one moe_perf read into the
            # default and low-latency views (rows tagged
            # ``kernel_source="moe_torch_flow_min_latency"`` route to the twin).
            moe_loaded = load_view(database, "_moe_data", PerfDataFilename.moe)
            low_latency_loaded = load_view(database, "_moe_low_latency_data", PerfDataFilename.moe)

            # WideEP MoE tables — SGLang-only.
            if database.backend == "sglang":
                wideep_context_loaded = load_view(
                    database, "_wideep_context_moe_data", PerfDataFilename.wideep_context_moe
                )
                wideep_generation_loaded = load_view(
                    database, "_wideep_generation_moe_data", PerfDataFilename.wideep_generation_moe
                )
            else:
                wideep_context_loaded = None
                wideep_generation_loaded = None

            cls._data_cache[key] = moe_loaded
            cls._low_latency_data_cache[key] = low_latency_loaded
            cls._wideep_context_data_cache[key] = wideep_context_loaded
            cls._wideep_generation_data_cache[key] = wideep_generation_loaded
            cls._record_load()

        if "_moe_data" not in database.__dict__:
            database._moe_data = cls._data_cache[key]
        if "_moe_low_latency_data" not in database.__dict__:
            database._moe_low_latency_data = cls._low_latency_data_cache[key]
        if "_wideep_context_moe_data" not in database.__dict__:
            database._wideep_context_moe_data = cls._wideep_context_data_cache[key]
        if "_wideep_generation_moe_data" not in database.__dict__:
            database._wideep_generation_moe_data = cls._wideep_generation_data_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._data_cache.clear()
        cls._low_latency_data_cache.clear()
        cls._wideep_context_data_cache.clear()
        cls._wideep_generation_data_cache.clear()

    # ------------------------------------------------------------------
    # Op contract
    # ------------------------------------------------------------------


# ───────────────────────────────────────────────────────────────────────
# MoEDispatch
# ───────────────────────────────────────────────────────────────────────


# a comm op to deduce the communication cost of MoE
class MoEDispatch(_core.MoEDispatch, OpShellKit):
    """MoE dispatch operation. For fine-grained MoE dispatch.

    Owns the SGLang DeepEP tables. On non-SGLang backends, both caches are
    bound to ``None`` and consumers must guard before dereference. Most of
    ``MoEDispatch.query()``'s body delegates to other ops' query methods
    (NCCL, CustomAllReduce, TRT-LLM AllToAll) — only the SGLang DeepEP
    branch consults this class's own tables.
    """

    _normal_data_cache: ClassVar[dict] = {}
    _ll_data_cache: ClassVar[dict] = {}

    def __init__(self, *args, **kwargs) -> None:
        """Capture the RAW ``quant_mode`` kwarg (possibly ``None``).

        The wire types ``moe_quant`` as a required ``MoeQuantMode`` — the Rust
        constructor maps ``None`` to bfloat16, the retired serializer's rule —
        but quant-AGNOSTIC dispatches (``quant_mode=None``, the hybrid family)
        are a Python-visible contract, so the shell keeps the original in the
        instance ``__dict__`` (pickle restores it via the default object
        state)."""
        del args
        self.__dict__["_py_quant_mode"] = kwargs.get("quant_mode")

    @property
    def _quant_mode(self):
        if "_py_quant_mode" in self.__dict__:
            return self.__dict__["_py_quant_mode"]
        # Rust-wrapped instances (e.g. composite children) fall back to the
        # wire value.
        return _core.MoEDispatch._quant_mode.__get__(self)

    # ------------------------------------------------------------------
    # Data ownership
    # ------------------------------------------------------------------

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Fetches the engine's SGLang DeepEP normal +
        low-latency table views on ``backend == "sglang"`` only; binds
        ``None`` on other backends.
        """
        from aiconfigurator_core.sdk.engine_table_view import load_view
        from aiconfigurator_core.sdk.perf_database import PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._normal_data_cache or key not in cls._ll_data_cache:
            # Locals first, commit last — a failed second fetch must not
            # leave the caches half-populated (see GEMM.load_data).
            if database.backend == "sglang":
                normal_loaded = load_view(database, "_wideep_deepep_normal_data", PerfDataFilename.wideep_deepep_normal)
                ll_loaded = load_view(database, "_wideep_deepep_ll_data", PerfDataFilename.wideep_deepep_ll)
            else:
                normal_loaded = None
                ll_loaded = None

            cls._normal_data_cache[key] = normal_loaded
            cls._ll_data_cache[key] = ll_loaded
            cls._record_load()

        if "_wideep_deepep_normal_data" not in database.__dict__:
            database._wideep_deepep_normal_data = cls._normal_data_cache[key]
        if "_wideep_deepep_ll_data" not in database.__dict__:
            database._wideep_deepep_ll_data = cls._ll_data_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._normal_data_cache.clear()
        cls._ll_data_cache.clear()

    # ------------------------------------------------------------------
    # The legacy deepep raw-table queries retired with the per-call stack
    # (#1357 PR-5); the engine reads the same rows via its moe_a2a adapters.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Op contract — legacy body lifted verbatim. Heavy branching across
    # backends; calls ``database.query_*`` helpers that are already
    # migrated (NCCL, CustomAllReduce, TRT-LLM AllToAll) or live in this
    # same class (DeepEP normal / ll via the database delegations).
    # ------------------------------------------------------------------
