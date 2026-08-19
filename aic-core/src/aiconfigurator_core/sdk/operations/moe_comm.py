# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Large-EP MoE communication family, unified across SGLang, vLLM, and TRT-LLM.

Models the all-to-all communication of large-scale expert-parallel MoE
(dispatch/combine, plus TRT-LLM's prepare phase) with one comm-backend
registry shared by all three inference backends. On TRT-LLM this covers the
*wideEP* path only — non-wideEP TRT-LLM paths are untouched.

``MOE_A2A_BACKENDS`` maps backend name to its :class:`MoECommBackendSpec`
(framework/phase applicability plus feasibility rules).
``MoEAllToAll`` is the op class over the unified ``moe_a2a_perf.parquet``
comm table, served by the engine table view (``_moe_a2a_data``, with legacy
per-backend adapters folded engine-side) as one nested dict keyed by
``[comm_backend][phase][comm_dtype][ep_size][node_num][hidden_size][topk]
[num_experts][sms][num_tokens]``. The class owns the view binding
(class-level cache + ``load_data``).

Large-EP expert compute is modeled from stock ``moe_perf`` by
``operations.moe.ModeledEPMoE``; this module owns communication only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import aiconfigurator_core._aiconfigurator_core as _core
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


@dataclass(frozen=True)
class MoECommBackendSpec:
    """Static description of one MoE all-to-all comm backend."""

    name: str
    frameworks: tuple[str, ...]  # ("sglang", "vllm") or ("trtllm",)
    inference_phases: tuple[str, ...]  # ("context",) | ("generation",) | ("context", "generation")
    comm_phases: tuple[str, ...]  # ("dispatch", "combine") | ("prepare", "dispatch", "combine")
    min_sm: int = 0
    max_topk: int = 8

    def feasible(
        self,
        *,
        topk: int,
        num_experts: int,
        moe_tp_size: int,
        moe_ep_size: int,
        sm_version: int | None = None,
    ) -> bool:
        """Whether this backend can serve the given MoE parallelism config."""
        return (
            topk <= self.max_topk
            and moe_tp_size == 1
            and 1 < moe_ep_size <= num_experts
            and num_experts % moe_ep_size == 0
            and (sm_version is None or sm_version >= self.min_sm)
        )


MOE_A2A_BACKENDS: dict[str, MoECommBackendSpec] = {
    "deepep_ht": MoECommBackendSpec(
        name="deepep_ht",
        frameworks=("sglang", "vllm"),
        inference_phases=("context",),
        comm_phases=("dispatch", "combine"),
    ),
    "deepep_ll": MoECommBackendSpec(
        name="deepep_ll",
        frameworks=("sglang", "vllm"),
        inference_phases=("generation",),
        comm_phases=("dispatch", "combine"),
    ),
    "deepep_v2": MoECommBackendSpec(
        name="deepep_v2",
        frameworks=("vllm",),
        inference_phases=("context", "generation"),
        comm_phases=("dispatch", "combine"),
    ),
    "trtllm_deepep_ht": MoECommBackendSpec(
        name="trtllm_deepep_ht",
        frameworks=("trtllm",),
        inference_phases=("context",),
        comm_phases=("dispatch", "combine"),
    ),
    "trtllm_deepep_ll": MoECommBackendSpec(
        name="trtllm_deepep_ll",
        frameworks=("trtllm",),
        inference_phases=("generation",),
        comm_phases=("dispatch", "combine"),
    ),
    "nvlink_two_sided": MoECommBackendSpec(
        name="nvlink_two_sided",
        frameworks=("trtllm",),
        inference_phases=("context", "generation"),
        comm_phases=("prepare", "dispatch", "combine"),
        min_sm=100,
    ),
    "nvlink_one_sided": MoECommBackendSpec(
        name="nvlink_one_sided",
        frameworks=("trtllm",),
        inference_phases=("context", "generation"),
        comm_phases=("dispatch", "combine"),
        min_sm=100,
    ),
}


def nodes_for(ep_size: int, num_gpus_per_node: int) -> int:
    """Node count needed to host ``ep_size`` EP ranks (ceil division)."""
    return -(-ep_size // num_gpus_per_node)


_A2A_PHASES = ("prepare", "dispatch", "combine")


def _validate_a2a_request(comm_backend: str, phase: str) -> None:
    """Shared ctor/query validation: unknown backend or phase is a ValueError.

    The per-backend check is a guard, not a live code path: the block builder
    iterates ``spec.comm_phases`` itself and the coverage probe walks table
    keys without validating — so a combination like ``("deepep_ht",
    "prepare")`` can only come from future misuse, and should fail here where
    the intent is expressed rather than later as a data miss.
    """
    if comm_backend not in MOE_A2A_BACKENDS:
        raise ValueError(f"Invalid comm_backend '{comm_backend}'. Must be one of {sorted(MOE_A2A_BACKENDS)}")
    if phase not in _A2A_PHASES:
        raise ValueError(f"Invalid phase '{phase}'. Must be one of {list(_A2A_PHASES)}")
    supported = MOE_A2A_BACKENDS[comm_backend].comm_phases
    if phase not in supported:
        raise ValueError(
            f"comm_backend '{comm_backend}' does not implement phase '{phase}'; supported: {list(supported)}"
        )


class MoEAllToAll(_core.MoEAllToAll, OpShellKit):
    """Unified large-EP MoE all-to-all comm op (one phase per instance).

    Owns ``_moe_a2a_data`` — the unified comm table loaded by
    the engine view (new-schema ``moe_a2a_perf.parquet`` plus the
    three legacy per-backend adapters). Loaded on every inference backend
    ({"sglang", "vllm", "trtllm"} all have legacy comm sources); ``None``
    otherwise. Comm ops see per-rank token counts: ``query(x=...)`` scales by
    ``scale_factor`` only — never by ``attention_dp_size``.
    ``attention_tp_size`` divides the token key before the lookup — legacy
    fidelity with ``MoEDispatch``'s ``num_tokens // self._scale_num_tokens``
    (plain floor division, no ``max(1, ...)`` guard). ``comm_dtype``
    resolves exact-first, then the legacy ``fp8_block`` -> ``fp8`` behavioral
    aliasing, then the sole-collected-dtype fallback (typed miss otherwise).
    """

    _data_cache: ClassVar[dict] = {}
    _trtllm_alltoall_data_cache: ClassVar[dict] = {}

    _SUPPORTED_BACKENDS: ClassVar[tuple[str, ...]] = ("sglang", "vllm", "trtllm")

    def __init__(self, *args, **kwargs) -> None:
        """Ctor-time spec guard on top of the Rust ``__new__`` (which has
        already consumed the args and built the op): the backend/phase matrix
        (``MOE_A2A_BACKENDS``) is Python-owned policy, so the ValueError fires
        here where the intent is expressed. The values are read back from the
        CONSTRUCTED op (not the call kwargs), so the guard holds however the
        constructor was invoked. Pickle rebuilds bypass this
        (``__getnewargs_ex__`` -> ``__new__``), which is safe: those values
        came from an already-validated instance."""
        del args, kwargs
        _validate_a2a_request(self._comm_backend, self._phase)

    # ------------------------------------------------------------------
    # Data ownership
    # ------------------------------------------------------------------

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Fetches the engine's unified moe_a2a table view (new
        schema + legacy adapters, merged engine-side) on the three inference
        backends; binds ``None`` otherwise.
        """
        from aiconfigurator_core.sdk.engine_table_view import load_view
        from aiconfigurator_core.sdk.perf_database import PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            if database.backend in cls._SUPPORTED_BACKENDS:
                cls._data_cache[key] = load_view(database, "_moe_a2a_data", PerfDataFilename.moe_a2a)
            else:
                cls._data_cache[key] = None

            cls._record_load()

        # Rehomed from the deleted ``TrtLLMWideEPMoEDispatch`` (AIC-1357): the
        # legacy trtllm alltoall view stays loadable for charts / the support
        # matrix even though no op family constructs against it anymore.
        if key not in cls._trtllm_alltoall_data_cache:
            if database.backend == "trtllm":
                cls._trtllm_alltoall_data_cache[key] = load_view(
                    database, "_trtllm_alltoall_data", PerfDataFilename.trtllm_alltoall
                )
            else:
                cls._trtllm_alltoall_data_cache[key] = None

        if "_moe_a2a_data" not in database.__dict__:
            database._moe_a2a_data = cls._data_cache[key]
        if "_trtllm_alltoall_data" not in database.__dict__:
            database._trtllm_alltoall_data = cls._trtllm_alltoall_data_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._data_cache.clear()
        cls._trtllm_alltoall_data_cache.clear()

    # ------------------------------------------------------------------
    # Op contract
    # ------------------------------------------------------------------
