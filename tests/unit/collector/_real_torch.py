# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Borrow real torch without disturbing collector tests that require a mock."""

import importlib
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from types import ModuleType
from unittest.mock import MagicMock

import pytest

_real_torch: ModuleType | None = None
_MISSING = object()


@contextmanager
def real_torch() -> Iterator[ModuleType]:
    """Borrow torch, restoring the original module state on exit."""
    global _real_torch
    previous = sys.modules.get("torch", _MISSING)
    try:
        if _real_torch is None:
            if isinstance(previous, MagicMock):
                sys.modules.pop("torch")
            try:
                _real_torch = importlib.import_module("torch")
            except ImportError:
                pytest.skip("real torch required for tensor operations", allow_module_level=True)
        sys.modules["torch"] = _real_torch
        yield _real_torch
    finally:
        if previous is _MISSING:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = previous
