# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request-local bridge between case catalogs and legacy case generators.

This module intentionally does not import :mod:`collector.model_cases`.
Callers may provide its ``CollectionCasePlan`` or any structurally compatible
catalog, without introducing a planner/model-loader import cycle.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class CaseCatalog(Protocol):
    """Structural subset used by planner-aware generators."""

    model_path: str | None
    op_cases: Mapping[str, Any]


active_case_catalog: ContextVar[CaseCatalog | Any | None] = ContextVar("active_case_catalog", default=None)
active_model_path: ContextVar[str | None] = ContextVar("active_model_path", default=None)


def get_active_case_catalog() -> CaseCatalog | Any | None:
    return active_case_catalog.get()


def get_active_model_path() -> str | None:
    return active_model_path.get()


@contextmanager
def use_case_catalog(catalog: CaseCatalog | Any | None, *, model_path: str | None = None) -> Iterator[None]:
    """Set the active catalog/model for this context and restore it on exit."""

    resolved_model_path = model_path if model_path is not None else getattr(catalog, "model_path", None)
    catalog_token = active_case_catalog.set(catalog)
    model_token = active_model_path.set(resolved_model_path)
    try:
        yield
    finally:
        active_model_path.reset(model_token)
        active_case_catalog.reset(catalog_token)


# A descriptive alias for call sites that prefer an action noun.
activate_case_catalog = use_case_catalog
