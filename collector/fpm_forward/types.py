# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared immutable records for FPM planning and collection."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, order=True, slots=True)
class FPMPoint:
    """One local-DP-rank physical workload point."""

    workload_kind: str
    batch_size: int
    suffix_length: int
    prefix_length: int

    def __post_init__(self) -> None:
        if self.workload_kind not in {"prefill", "decode"}:
            raise ValueError(f"unknown workload_kind: {self.workload_kind}")
        if self.batch_size < 1 or self.suffix_length < 1 or self.prefix_length < 0:
            raise ValueError(f"invalid FPM point: {self}")
        if self.workload_kind == "decode" and self.suffix_length != 1:
            raise ValueError("decode points must execute exactly one new token per request")

    @property
    def key(self) -> tuple[str, int, int, int]:
        return (
            self.workload_kind,
            self.batch_size,
            self.suffix_length,
            self.prefix_length,
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ParallelTopology:
    tp: int
    pp: int
    dp: int
    moe_tp: int
    moe_ep: int
    cp: int

    @property
    def total_gpus(self) -> int:
        return self.tp * self.pp * self.dp * self.cp

    def to_dict(self) -> dict[str, int]:
        return asdict(self)
