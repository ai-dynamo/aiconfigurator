# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared data types for collector registries."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class PerfFile(str, Enum):
    """Canonical output filenames for collector operations.

    Inherits from ``str`` so values pass directly to ``open()`` / ``log_perf()``
    without ``.value``.
    """

    def __str__(self) -> str:
        """
        Override behavior of str(x) and f"{x}" to return
        the perf filename instead of the enum name like "PerfFile.GEMM".
        """
        return self.value

    GEMM = "gemm_perf.txt"
    CONTEXT_ATTENTION = "context_attention_perf.txt"
    GENERATION_ATTENTION = "generation_attention_perf.txt"
    MOE = "moe_perf.txt"
    CONTEXT_MLA = "context_mla_perf.txt"
    GENERATION_MLA = "generation_mla_perf.txt"
    MLA_BMM = "mla_bmm_perf.txt"
    GDN = "gdn_perf.txt"
    MAMBA2 = "mamba2_perf.txt"
    COMPUTESCALE = "computescale_perf.txt"
    WIDEEP_MOE = "wideep_moe_perf.txt"
    WIDEEP_CONTEXT_MLA = "wideep_context_mla_perf.txt"
    WIDEEP_GENERATION_MLA = "wideep_generation_mla_perf.txt"
    WIDEEP_CONTEXT_MOE = "wideep_context_moe_perf.txt"
    WIDEEP_GENERATION_MOE = "wideep_generation_moe_perf.txt"
    MLA_CONTEXT_MODULE = "mla_context_module_perf.txt"
    MLA_GENERATION_MODULE = "mla_generation_module_perf.txt"
    DSA_CONTEXT_MODULE = "dsa_context_module_perf.txt"
    DSA_GENERATION_MODULE = "dsa_generation_module_perf.txt"
    NCCL = "nccl_perf.txt"
    CUSTOM_ALLREDUCE = "custom_allreduce_perf.txt"
    TRTLLM_ALLTOALL = "trtllm_alltoall_perf.txt"


@dataclass(frozen=True, slots=True)
class VersionRoute:
    """A (min_version, module_path) pair for version-based module routing.

    ``min_version`` is a PEP 440 version string. The resolver picks the first
    ``VersionRoute`` whose ``min_version`` is <= the runtime version (entries
    must be listed in descending order).
    """

    min_version: str
    module: str


@dataclass(frozen=True, slots=True)
class OpEntry:
    """One operation in a collector registry.

    Exactly one of ``module`` (unversioned) or ``versions`` (versioned) must be
    provided.  This invariant is validated at construction time.
    """

    op: str
    get_func: str
    run_func: str
    perf_filename: str
    module: str | None = None
    versions: tuple[VersionRoute, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.module and not self.versions:
            raise ValueError(f"OpEntry '{self.op}': must specify 'module' or 'versions'")
        if self.module and self.versions:
            raise ValueError(f"OpEntry '{self.op}': cannot specify both 'module' and 'versions'")
