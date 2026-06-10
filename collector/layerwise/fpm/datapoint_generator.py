# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate FPM deployment and workload cases from public CLI filters."""

from __future__ import annotations

from dataclasses import dataclass


PRODUCTION_CONTEXTS = "128,1024,4096"
PRODUCTION_DECODE_BATCHES = "1,4,16"
PRODUCTION_DECODE_PAST_KV = "4096"
PRODUCTION_DECODE_OSL = "8"
SMOKE_CONTEXTS = "128"
SMOKE_DECODE_BATCHES = "1"
SMOKE_DECODE_PAST_KV = "1024"
SMOKE_DECODE_OSL = "2"


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


def preset_defaults(preset: str) -> dict[str, str]:
    """Return public FPM shape defaults for a run preset."""
    if preset == "production":
        return {
            "contexts": PRODUCTION_CONTEXTS,
            "decode_batches": PRODUCTION_DECODE_BATCHES,
            "decode_past_kv": PRODUCTION_DECODE_PAST_KV,
            "decode_osl": PRODUCTION_DECODE_OSL,
        }
    if preset == "smoke":
        return {
            "contexts": SMOKE_CONTEXTS,
            "decode_batches": SMOKE_DECODE_BATCHES,
            "decode_past_kv": SMOKE_DECODE_PAST_KV,
            "decode_osl": SMOKE_DECODE_OSL,
        }
    raise ValueError(f"unknown FPM preset: {preset}")


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
