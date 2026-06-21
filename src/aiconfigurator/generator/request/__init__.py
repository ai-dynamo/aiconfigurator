# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public input contract for the generator. See generator_request_design.md."""
from __future__ import annotations

from .cli import from_cli
from .legacy import from_legacy_params, to_legacy_params
from .schema import (
    SCHEMA_VERSION,
    BackendSpec,
    CacheSpec,
    EmitTargets,
    GeneratorRequest,
    ModelFacts,
    ModelSpec,
    Overrides,
    Platform,
    RoleSizing,
    SlaSpec,
    Topology,
)

__all__ = [
    "SCHEMA_VERSION",
    "BackendSpec",
    "CacheSpec",
    "EmitTargets",
    "GeneratorRequest",
    "ModelFacts",
    "ModelSpec",
    "Overrides",
    "Platform",
    "RoleSizing",
    "SlaSpec",
    "Topology",
    "from_cli",
    "from_legacy_params",
    "to_legacy_params",
]
