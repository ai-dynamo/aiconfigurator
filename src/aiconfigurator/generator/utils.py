# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for generator modules."""

from __future__ import annotations

from typing import Any, Optional

DEFAULT_BACKEND = "trtllm"


def normalize_backend(backend: Optional[str], default: str = DEFAULT_BACKEND) -> str:
    """Normalize backend names to lowercase strings with a fallback."""
    if backend:
        return str(backend).strip().lower()
    return default


def coerce_bool(value: Optional[Any]) -> Optional[bool]:
    """Best-effort conversion of user input into booleans."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return bool(value)


def coerce_int(value: Optional[Any]) -> Optional[int]:
    """Convert values to ints while swallowing Type/Value errors."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
