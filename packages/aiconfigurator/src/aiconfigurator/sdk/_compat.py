# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for the legacy aiconfigurator.sdk compatibility surface."""

from __future__ import annotations

import importlib
import sys
from collections.abc import MutableMapping
from types import ModuleType


def alias_module(alias_name: str, canonical_name: str) -> ModuleType:
    """Make an import path resolve to the canonical module object.

    Re-exporting names would create two distinct module objects. That can split
    module-level caches and private state. Replacing the wrapper entry in
    sys.modules makes both import paths share all implementation state.
    """
    module = importlib.import_module(canonical_name)
    sys.modules[alias_name] = module
    return module


def export_public_package(
    canonical_name: str,
    namespace: MutableMapping[str, object],
) -> tuple[ModuleType, list[str]]:
    """Re-export an API while keeping the legacy package search path.

    Models and operations retain real compatibility packages so their legacy
    child-module wrappers remain importable. Public objects come from the
    canonical package; missing attributes are delegated by the wrappers.
    """
    module = importlib.import_module(canonical_name)
    public_names = list(getattr(module, "__all__", ()))
    namespace.update({name: getattr(module, name) for name in public_names})
    return module, public_names
