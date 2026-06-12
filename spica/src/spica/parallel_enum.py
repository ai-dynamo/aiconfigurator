# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Enumerate legal per-worker parallel shapes and the replica counts that fit a GPU budget.

The per-worker shape enumeration mirrors
``aiconfigurator.sdk.utils.enumerate_parallel_config`` + ``filter_real_silicon_configs``
(real-silicon profile): ``pp`` is pinned to 1; the MoE width constraint
``dp*tp == moe_tp*moe_ep`` holds; for MoE only the pure TEP / DEP / TP patterns are
kept; dense models use plain TP. The backend-specific MoE filters are mirrored too.

``enumerate_parallel_config`` stops at *one worker's* shape (GPUs/worker = tp*pp*dp).
On top of that, this module iterates the replica counts ``r`` such that
``gpus_per_worker * r`` fits the GPU budget — the replica/worker count AIC derives
separately in its sweep layer.

Kept standalone (no aiconfigurator import) so it is light and unit-testable;
parity with AIC's rules is covered by tests. ``is_moe`` / ``mla`` are inputs here
(resolved from the model via AIC's ``check_is_moe`` / ``get_model_family`` by the
caller).
"""

from __future__ import annotations

from dataclasses import dataclass

# GPUs-per-worker ladder (matches AIC's default num_gpu_per_worker).
_DEFAULT_GPUS_PER_WORKER: tuple[int, ...] = (1, 2, 4, 8, 16)
# Ladder used to enumerate tp / dp / moe_tp / moe_ep candidates within a worker.
_DIM_LADDER: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)


@dataclass(frozen=True)
class ParallelShape:
    """One worker's parallel shape (``pp`` pinned to 1, real-silicon profile).

    ``dp`` is the attention data-parallel size (attention_dp_size).
    """

    tp: int
    dp: int
    moe_tp: int
    moe_ep: int
    pp: int = 1

    @property
    def gpus_per_worker(self) -> int:
        return self.tp * self.pp * self.dp

    @property
    def strategy(self) -> str:
        """Label per AIC's real-silicon patterns: ``tp`` (dense, or MoE pure
        expert-TP), ``tep`` (attention-TP + expert-EP), ``dep`` (attention-DP +
        expert-EP)."""
        if self.moe_tp == 1 and self.moe_ep == 1:
            return "tp"  # dense
        if self.tp > 1 and self.dp == 1 and self.moe_tp == 1 and self.moe_ep > 1:
            return "tep"
        if self.tp == 1 and self.dp > 1 and self.moe_tp == 1 and self.moe_ep > 1:
            return "dep"
        if self.tp > 1 and self.dp == 1 and self.moe_tp > 1 and self.moe_ep == 1:
            return "tp"  # MoE pure expert-TP
        return "mixed"


@dataclass(frozen=True)
class ReplicaParallelConfig:
    """A worker shape plus how many replicas of it run, under a GPU budget."""

    shape: ParallelShape
    replicas: int

    @property
    def total_gpus(self) -> int:
        return self.shape.gpus_per_worker * self.replicas


def _ladder_upto(max_value: int, ladder: tuple[int, ...] = _DIM_LADDER) -> list[int]:
    return [v for v in ladder if v <= max_value]


def _backend_allows_moe_tp(backend: str, *, enable_wideep: bool, moe_backend: str | None) -> bool:
    """sglang EP-only MoE backends (wideep / deepep_moe / megamoe) require moe_tp=1."""
    if backend == "sglang" and (enable_wideep or moe_backend in {"deepep_moe", "megamoe"}):
        return False
    return True


def enumerate_worker_shapes(
    *,
    is_moe: bool,
    mla: bool,
    backend: str,
    gpus_per_worker: int,
    enable_wideep: bool = False,
    moe_backend: str | None = None,
) -> list[ParallelShape]:
    """Legal per-worker shapes at exactly ``gpus_per_worker`` GPUs (``pp`` = 1).

    Mirrors ``enumerate_parallel_config`` (width + backend filters) followed by
    ``filter_real_silicon_configs`` (pure TEP / DEP / TP only for MoE).
    """
    g = gpus_per_worker
    if not is_moe:
        # dense: plain TP, no attention-dp, no experts.
        return [ParallelShape(tp=g, dp=1, moe_tp=1, moe_ep=1)]

    shapes: list[ParallelShape] = []
    cand = _ladder_upto(g)
    for tp in cand:
        for dp in cand:
            if tp * dp != g:  # one worker spans tp*pp*dp = g GPUs (pp=1)
                continue
            width = tp * dp
            for moe_tp in cand:
                for moe_ep in cand:
                    if moe_tp * moe_ep != width:  # MoE width constraint
                        continue
                    # backend filters (from enumerate_parallel_config)
                    if backend == "trtllm" and dp > 1 and tp > 1:
                        continue
                    if (
                        backend == "sglang"
                        and moe_tp > 1
                        and not _backend_allows_moe_tp(backend, enable_wideep=enable_wideep, moe_backend=moe_backend)
                    ):
                        continue
                    if backend == "vllm" and moe_tp > 1 and moe_ep > 1:
                        continue
                    # real-silicon pure-pattern filter
                    is_tep = tp > 1 and dp == 1 and moe_tp == 1 and moe_ep > 1
                    is_dep = tp == 1 and dp > 1 and moe_tp == 1 and moe_ep > 1
                    is_pure_tp = tp > 1 and dp == 1 and moe_tp > 1 and moe_ep == 1 and not mla
                    if not (is_tep or is_dep or is_pure_tp):
                        continue
                    shapes.append(ParallelShape(tp=tp, dp=dp, moe_tp=moe_tp, moe_ep=moe_ep))
    return shapes


def enumerate_parallel_configs(
    *,
    is_moe: bool,
    mla: bool,
    backend: str,
    gpu_budget: int,
    min_gpu_budget: int | None = None,
    gpus_per_worker_candidates: tuple[int, ...] = _DEFAULT_GPUS_PER_WORKER,
    enable_wideep: bool = False,
    moe_backend: str | None = None,
) -> list[ReplicaParallelConfig]:
    """Enumerate ``(worker shape, replica count)`` configs that fit ``gpu_budget``.

    For each candidate GPUs-per-worker ``g`` (<= budget), enumerate the legal
    worker shapes, then iterate replica counts ``r`` in ``1..(gpu_budget // g)``
    so the total ``g * r`` stays within ``[min_gpu_budget, gpu_budget]``.

    This is branch-agnostic: call once for an ``agg`` worker, or once per role
    (prefill / decode) for ``disagg`` — the prefill/decode pairing under the
    shared budget is the downstream rate-matching step.
    """
    configs: list[ReplicaParallelConfig] = []
    for g in gpus_per_worker_candidates:
        if g > gpu_budget:
            continue
        shapes = enumerate_worker_shapes(
            is_moe=is_moe,
            mla=mla,
            backend=backend,
            gpus_per_worker=g,
            enable_wideep=enable_wideep,
            moe_backend=moe_backend,
        )
        if not shapes:
            continue
        max_replicas = gpu_budget // g
        for shape in shapes:
            for r in range(1, max_replicas + 1):
                total = g * r
                if min_gpu_budget is not None and total < min_gpu_budget:
                    continue
                configs.append(ReplicaParallelConfig(shape=shape, replicas=r))
    return configs
