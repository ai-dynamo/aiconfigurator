# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared immutable records for FPM planning and collection."""

from __future__ import annotations

from dataclasses import asdict, dataclass

# Decode strategies whose measurements require the engine's real-KV warm-up
# protocol. This is shared by rendering and artifact validation so an image
# cannot be rendered under one protocol and published under another.
KVWARM_STRATEGIES = frozenset({"tep", "dep", "pure_tp"})


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
