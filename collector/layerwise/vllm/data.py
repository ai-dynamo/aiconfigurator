# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data structures used by the vLLM layerwise collection pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class DataPoint:
    """One context or decode shape collected for a representative layer."""

    phase: str
    batch_size: int
    new_tokens: int
    past_kv: int

    @property
    def shape_key(self) -> str:
        """Return the stable shape label used in status and manifest IDs."""

        return (
            f"{self.phase}:bs{self.batch_size}:"
            f"new{self.new_tokens}:past{self.past_kv}"
        )

    def datapoint_id(self, work_unit_id: str) -> str:
        """Return the globally unique datapoint ID within a run."""

        return f"{work_unit_id}:{self.shape_key}"

    def parse_key(self) -> tuple[int, int, int]:
        """Return the parser lookup key emitted by the vLLM step marker."""

        if self.phase == "ctx":
            return self.new_tokens, self.batch_size, self.past_kv
        return self.past_kv + 1, self.batch_size, self.past_kv

@dataclass(frozen=True)
class RepresentativeLayer:
    """Transformer layer slice measured to represent one layer type."""

    layer_index: int
    layer_type: str
    measured_layer_count: int
    layer_multiplier: int
    target_layers: tuple[int, ...] = ()

    def kept_layers(self) -> list[int]:
        """Return concrete model layer indices retained in the patched config."""

        if self.target_layers:
            return list(self.target_layers)
        return list(range(self.layer_index, self.layer_index + self.measured_layer_count))

@dataclass(frozen=True)
class WorkUnit:
    """One patched engine configuration plus the datapoints measured on it."""

    work_unit_id: str
    model_dir: str
    row_base: dict[str, Any]
    representative: RepresentativeLayer
    target_layers: list[int]
    datapoints: list[DataPoint]
    moe_noop: bool = False
    includes_moe: bool = False

    def manifest_rows(self) -> list[dict[str, Any]]:
        """Expand the work unit into one manifest row per datapoint."""

        rows = []
        for dp in self.datapoints:
            rows.append({
                "work_unit_id": self.work_unit_id,
                "datapoint_id": dp.datapoint_id(self.work_unit_id),
                **self.row_base,
                **asdict(self.representative),
                "moe_noop": self.moe_noop,
                "includes_moe": self.includes_moe,
                "phase": dp.phase,
                "batch_size": dp.batch_size,
                "new_tokens": dp.new_tokens,
                "past_kv": dp.past_kv,
            })
        return rows
