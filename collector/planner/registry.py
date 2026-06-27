# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Registry for op-specific logical case schemas."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from collector.planner.models import InvocationKey, LogicalCandidate, PlanContext


@runtime_checkable
class OpCaseSchema(Protocol):
    """Minimal contract implemented by a migrated operation family."""

    def normalize(self, candidate: LogicalCandidate, context: PlanContext) -> LogicalCandidate | Any: ...

    def is_reachable(self, candidate: LogicalCandidate, context: PlanContext) -> bool | tuple[bool, str | None]: ...

    def is_supported(self, candidate: LogicalCandidate, context: PlanContext) -> bool | tuple[bool, str | None]: ...

    def invocation_key(self, candidate: LogicalCandidate, context: PlanContext) -> InvocationKey | Any: ...


class LegacyPassthroughSchema:
    """Default schema that preserves a legacy raw case without policy changes."""

    name = "legacy_passthrough"

    def normalize(self, candidate: LogicalCandidate, context: PlanContext) -> LogicalCandidate:
        return candidate

    def is_reachable(self, candidate: LogicalCandidate, context: PlanContext) -> bool:
        return True

    def is_supported(self, candidate: LogicalCandidate, context: PlanContext) -> bool:
        return True

    def invocation_key(self, candidate: LogicalCandidate, context: PlanContext) -> InvocationKey:
        return InvocationKey.from_value(context, candidate.payload)


legacy_passthrough = LegacyPassthroughSchema()

# Backend-specific entries override backend-agnostic op entries.
_SCHEMAS: dict[tuple[str | None, str], OpCaseSchema] = {}


def register_schema(op: str, schema: OpCaseSchema, *, backend: str | None = None, replace: bool = False) -> None:
    """Register ``schema`` for an op, optionally scoped to one backend."""

    key = (backend, op)
    if key in _SCHEMAS and not replace:
        raise ValueError(f"case schema already registered for backend={backend!r}, op={op!r}")
    _SCHEMAS[key] = schema


def unregister_schema(op: str, *, backend: str | None = None) -> None:
    _SCHEMAS.pop((backend, op), None)


def clear_schemas() -> None:
    """Clear registrations. Intended for isolated tests and plugin reloads."""

    _SCHEMAS.clear()


def get_schema(op: str, *, backend: str | None = None) -> OpCaseSchema:
    """Resolve a backend override, generic op schema, or legacy fallback."""

    if backend is not None and (backend, op) in _SCHEMAS:
        return _SCHEMAS[(backend, op)]
    return _SCHEMAS.get((None, op), legacy_passthrough)
