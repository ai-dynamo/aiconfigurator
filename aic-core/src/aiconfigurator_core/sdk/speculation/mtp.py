# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MTP speculative decoding as a scheme.

MTP's draft is ``depth`` extra target-shaped layers. Its compute cost is
embedded in the target op graph: model families multiply generation op
counts by ``mtp_scale_factor(depth, L) = (depth + L) / L`` — algebraically
"depth extra layers per iteration". This scheme therefore reports empty
draft op lists and zero extra bytes, and ``verify_width = depth + 1``:
every consumer substitution is an identity with the legacy ``nextn`` path,
which is what guarantees bit-identical MTP predictions.
"""

from __future__ import annotations

from typing import ClassVar

from aiconfigurator_core.sdk.speculation.base import (
    DraftOpSpec,
    SpecSchemeBase,
    SpeculationConfig,
    register_spec_scheme,
)


@register_spec_scheme("mtp")
class MTPScheme(SpecSchemeBase):
    kind: ClassVar[str] = "mtp"

    def __init__(self, depth: int) -> None:
        depth = int(depth)
        if depth < 1:
            raise ValueError(f"MTP depth must be >= 1, got {depth}. Use kind='none' to disable.")
        self._depth = depth

    @property
    def depth(self) -> int:
        return self._depth

    @classmethod
    def from_configs(cls, model_config, spec_config: SpeculationConfig) -> MTPScheme:
        return cls(depth=spec_config.params.get("depth", 0))

    def validate(self, model, backend_name: str) -> None:
        # Family-level MTP gates remain the model classes' own asserts
        # (e.g. nemotron_nas/GPT require nextn == 0 at construction time).
        return None

    def verify_width(self) -> int:
        return self._depth + 1

    def build_draft_generation_ops(self, model) -> list[DraftOpSpec]:
        return []

    def build_draft_context_ops(self, model) -> list[DraftOpSpec]:
        return []

    def draft_weights_bytes(self, model) -> float:
        # Preserved legacy behavior: the MTP layer's weights are not
        # separately accounted (see memory.py rationale). Changing this is a
        # deliberate future accuracy fix, not part of the compatibility phase.
        return 0.0

    def draft_kv_bytes_per_sequence(self, model, seq_len: int) -> float:
        return 0.0
