# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Speculative-decoding schemes package.

Scheme modules register themselves via ``@register_spec_scheme("<kind>")``;
``pkgutil.iter_modules`` imports every sibling module at package import time
(the model-registry idiom), so **adding a scheme file is enough** — no edits
here.

Known accuracy envelope (zero-calibration contract, measured on
Qwen3-8B x H100 x vLLM against static-batch pure-decode rounds):

- The sequence-basis width channel has no fitted constants; deep-concurrency
  pure-decode rounds under-predict by ~10-18% (the shared-KV floor misses
  real wide-kernel inefficiencies). A bracket-blend calibration was
  evaluated and REJECTED: its two fitted parameters did not transfer out of
  the fitted domain (isl-1k deep-c +41-45% on long-context pure decode).
  Structural fixes (a physically-bounded token-basis component for the
  KV-read share) are follow-up work; do not reintroduce fitted patches.
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
