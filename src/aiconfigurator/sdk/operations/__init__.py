# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility facade for aiconfigurator_core.sdk.operations."""

from __future__ import annotations

from aiconfigurator.sdk._compat import export_public_package as _export_public_package

_canonical_module, __all__ = _export_public_package("aiconfigurator_core.sdk.operations", globals())


def __getattr__(name: str) -> object:
    """Delegate private and future attributes to the canonical package."""
    return getattr(_canonical_module, name)


def __dir__() -> list[str]:
    """Expose both facade and canonical attributes to introspection."""
    return sorted(set(globals()) | set(dir(_canonical_module)))
