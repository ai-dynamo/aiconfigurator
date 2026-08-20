# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Speculative-decoding schemes package.

Scheme modules register themselves via ``@register_spec_scheme("<kind>")``;
``pkgutil.iter_modules`` imports every sibling module at package import time
(the model-registry idiom), so **adding a scheme file is enough** — no edits
here.
"""

from __future__ import annotations

import importlib
import pkgutil

from aiconfigurator_core.sdk.speculation.base import (
    DraftOpSpec,
    NullScheme,
    SpecSchemeBase,
    SpeculationConfig,
    build_spec_scheme,
    get_spec_scheme_cls,
    register_spec_scheme,
)

_SKIP = {"base"}
for _, _name, _ in pkgutil.iter_modules(__path__):
    if _name not in _SKIP:
        importlib.import_module(f".{_name}", __name__)
del _SKIP

__all__ = [
    "DraftOpSpec",
    "NullScheme",
    "SpecSchemeBase",
    "SpeculationConfig",
    "build_spec_scheme",
    "get_spec_scheme_cls",
    "register_spec_scheme",
]
