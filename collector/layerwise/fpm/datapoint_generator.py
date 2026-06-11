# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate FPM deployment and workload cases from public CLI filters."""

from __future__ import annotations

from dataclasses import dataclass


FULL_CONTEXTS = "128,1024,4096"
FULL_CONTEXT_REPEATS = "6"
FULL_DECODE_BATCHES = "1,4,16"
FULL_DECODE_PAST_KV = "4096"
FULL_DECODE_OSL = "8"
FULL_DECODE_REPEATS = "6"
REAL_WORKLOAD_REQUESTS = 128
REAL_WORKLOAD_CONCURRENCY = 32
REAL_WORKLOAD_DATASET = "OpenAssistant/oasst1"
REAL_WORKLOAD_SHAPE_SOURCE = "scaled_dataset"
REAL_WORKLOAD_ISL_MIN = 100
REAL_WORKLOAD_ISL_MAX = 16384
REAL_WORKLOAD_ISL_MEAN = 4096
REAL_WORKLOAD_OSL_MIN = 100
REAL_WORKLOAD_OSL_MAX = 4096
REAL_WORKLOAD_OSL_MEAN = 1024


@dataclass(frozen=True)
class FpmCase:
    """One real vLLM/Dynamo deployment to run for FPM collection."""

    tp_size: int
    ep_size: int
    decode_past_kv: int

    @property
    def label(self) -> str:
        """Return the run-subdirectory label for this case."""
        return f"tp{self.tp_size}_ep{self.ep_size}_past{self.decode_past_kv}"


def parse_int_csv(raw: str) -> list[int]:
    """Parse a comma-separated integer list."""
    return [int(part) for part in raw.split(",") if part.strip()]


def default_shapes() -> dict[str, str]:
    """Return the default FPM shape set."""
    return {
        "contexts": FULL_CONTEXTS,
        "context_repeats": FULL_CONTEXT_REPEATS,
        "decode_batches": FULL_DECODE_BATCHES,
        "decode_past_kv": FULL_DECODE_PAST_KV,
        "decode_osl": FULL_DECODE_OSL,
        "decode_repeats": FULL_DECODE_REPEATS,
        "real_workload_requests": str(REAL_WORKLOAD_REQUESTS),
        "real_workload_concurrency": str(REAL_WORKLOAD_CONCURRENCY),
        "real_workload_dataset": REAL_WORKLOAD_DATASET,
        "real_workload_shape_source": REAL_WORKLOAD_SHAPE_SOURCE,
        "real_workload_isl_min": str(REAL_WORKLOAD_ISL_MIN),
        "real_workload_isl_max": str(REAL_WORKLOAD_ISL_MAX),
        "real_workload_isl_mean": str(REAL_WORKLOAD_ISL_MEAN),
        "real_workload_osl_min": str(REAL_WORKLOAD_OSL_MIN),
        "real_workload_osl_max": str(REAL_WORKLOAD_OSL_MAX),
        "real_workload_osl_mean": str(REAL_WORKLOAD_OSL_MEAN),
    }


def generate_fpm_cases(tp_sizes: str, ep_sizes: str, decode_past_kv: str) -> list[FpmCase]:
    """Expand TP/EP/KV filters into concrete vLLM-parity deployment cases.

    vLLM supports TP-only MoE execution (ep=1) and expert-parallel execution
    over the full TP/DP group (represented here as ep=tp). It does not expose
    an independent MoE tensor-parallel axis, so intermediate ep values are not
    generated.
    """
    cases: list[FpmCase] = []
    for tp in parse_int_csv(tp_sizes):
        for ep in parse_int_csv(ep_sizes):
            if ep != 1 and ep != tp:
                continue
            for past_kv in parse_int_csv(decode_past_kv):
                cases.append(FpmCase(tp_size=tp, ep_size=ep, decode_past_kv=past_kv))
    return cases
