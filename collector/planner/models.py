# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed values shared by collector case population.

The planner deliberately keeps a logical candidate's payload opaque.  An op
schema owns that payload; the common compiler only needs to normalize it,
filter it, and derive a stable invocation identity.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

CanonicalJSON = None | bool | int | float | str | list["CanonicalJSON"] | dict[str, "CanonicalJSON"]


def canonicalize(value: Any) -> CanonicalJSON:
    """Convert supported Python values to deterministic JSON-compatible data.

    Lists and tuples intentionally share an identity.  Sets are ordered by the
    canonical JSON representation of their members.  Mapping order and
    dataclass implementation details therefore cannot change an invocation id.
    """

    if isinstance(value, Enum):
        return canonicalize(value.value)
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        # Reject NaN and infinity rather than producing non-standard JSON with
        # platform-dependent comparison behavior.
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("non-finite floats cannot be used in canonical planner values")
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: canonicalize(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        if all(isinstance(key, str) for key in value):
            return {key: canonicalize(item) for key, item in sorted(value.items())}

        # JSON object keys can only be strings.  Keep non-string keys typed by
        # representing the mapping as a sorted sequence of key/value pairs.
        pairs = [(canonicalize(key), canonicalize(item)) for key, item in value.items()]
        pairs.sort(key=lambda pair: _dump_canonical(pair[0]))
        return {"__mapping__": [[key, item] for key, item in pairs]}
    if isinstance(value, list | tuple):
        return [canonicalize(item) for item in value]
    if isinstance(value, set | frozenset):
        items = [canonicalize(item) for item in value]
        return sorted(items, key=_dump_canonical)
    raise TypeError(f"unsupported canonical planner value: {type(value).__name__}")


def _dump_canonical(value: CanonicalJSON) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True)


def canonical_json(value: Any) -> str:
    """Return a stable, compact JSON representation of ``value``."""

    return _dump_canonical(canonicalize(value))


@dataclass(frozen=True, slots=True)
class PlanContext:
    """Immutable scope in which rules are populated."""

    backend: str
    op: str
    perf_file: str
    schema_version: str = "1"
    model_path: str | None = None
    model_architecture: str | None = None
    gpu_type: str | None = None
    sm_version: int | None = None
    framework_version: str | None = None
    plan_fingerprint: str | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict, compare=False, hash=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "backend", str(self.backend))
        object.__setattr__(self, "op", str(self.op))
        object.__setattr__(self, "perf_file", str(self.perf_file))
        object.__setattr__(self, "schema_version", str(self.schema_version))
        if self.gpu_type is not None:
            object.__setattr__(self, "gpu_type", str(self.gpu_type))
        if self.framework_version is not None:
            object.__setattr__(self, "framework_version", str(self.framework_version))
        if self.model_path is not None:
            object.__setattr__(self, "model_path", str(self.model_path))
        if self.model_architecture is not None:
            object.__setattr__(self, "model_architecture", str(self.model_architecture))
        if self.plan_fingerprint is not None:
            object.__setattr__(self, "plan_fingerprint", str(self.plan_fingerprint))
        if isinstance(self.sm_version, bool):
            raise TypeError("sm_version must be an integer, not bool")
        if self.sm_version is not None:
            object.__setattr__(self, "sm_version", int(self.sm_version))

    @property
    def scope(self) -> dict[str, str | int | None]:
        """Fields that namespace an invocation identity."""

        return {
            "backend": self.backend,
            "op": self.op,
            "perf_file": self.perf_file,
            "schema_version": self.schema_version,
            "framework_version": self.framework_version,
            "gpu_type": self.gpu_type,
            "sm_version": self.sm_version,
            "model_path": self.model_path,
            "model_architecture": self.model_architecture,
            "plan_fingerprint": self.plan_fingerprint,
        }


@dataclass(frozen=True, slots=True)
class RuleSource:
    """Provenance for one independently expanded population rule."""

    rule_id: str
    path: Path | None = None
    model_path: str | None = None
    model_architecture: str | None = None
    backend: str | None = None
    sm_version: int | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict, compare=False, hash=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "rule_id", str(self.rule_id))
        if self.path is not None and not isinstance(self.path, Path):
            object.__setattr__(self, "path", Path(self.path))

    @property
    def id(self) -> str:
        return canonical_json(
            {
                "backend": self.backend,
                "model_architecture": self.model_architecture,
                "model_path": self.model_path,
                "path": self.path,
                "rule_id": self.rule_id,
                "sm_version": self.sm_version,
                "attributes": self.attributes,
            }
        )


@dataclass(frozen=True, slots=True)
class LogicalCandidate:
    """One rule-produced case before common population decisions."""

    payload: Any
    source: RuleSource
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False, hash=False)

    @property
    def raw_case(self) -> Any:
        """Compatibility name used by legacy case-list adapters."""

        return self.payload


@dataclass(frozen=True, slots=True)
class InvocationKey:
    """Canonical identity for one benchmark process invocation."""

    backend: str
    op: str
    perf_file: str
    schema_version: str
    framework_version: str | None
    gpu_type: str | None
    sm_version: int | None
    model_path: str | None
    model_architecture: str | None
    plan_fingerprint: str | None
    payload_json: str

    @classmethod
    def from_value(cls, context: PlanContext, value: Any) -> InvocationKey:
        return cls(
            backend=context.backend,
            op=context.op,
            perf_file=context.perf_file,
            schema_version=context.schema_version,
            framework_version=context.framework_version,
            gpu_type=context.gpu_type,
            sm_version=context.sm_version,
            model_path=context.model_path,
            model_architecture=context.model_architecture,
            plan_fingerprint=context.plan_fingerprint,
            payload_json=canonical_json(value),
        )

    @property
    def id(self) -> str:
        identity = canonical_json(
            {
                "backend": self.backend,
                "op": self.op,
                "perf_file": self.perf_file,
                "schema_version": self.schema_version,
                "framework_version": self.framework_version,
                "gpu_type": self.gpu_type,
                "sm_version": self.sm_version,
                "model_path": self.model_path,
                "model_architecture": self.model_architecture,
                "plan_fingerprint": self.plan_fingerprint,
                "payload": json.loads(self.payload_json),
            }
        )
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    def __str__(self) -> str:
        return self.id


@dataclass(frozen=True, slots=True)
class PopulationRule:
    """An additive rule whose candidates are expanded independently."""

    source: RuleSource
    candidates: Iterable[Any] | Callable[[PlanContext], Iterable[Any]]

    def expand(self, context: PlanContext) -> Iterable[LogicalCandidate]:
        values = self.candidates(context) if callable(self.candidates) else self.candidates
        for value in values:
            if isinstance(value, LogicalCandidate):
                yield value
            else:
                yield LogicalCandidate(payload=value, source=self.source)

    @classmethod
    def from_legacy_cases(
        cls,
        cases: Iterable[Any] | Callable[[PlanContext], Iterable[Any]],
        *,
        rule_id: str = "legacy_cases",
        source: RuleSource | None = None,
    ) -> PopulationRule:
        """Wrap an existing raw tuple/dict case iterable as one additive rule."""

        return cls(source=source or RuleSource(rule_id=rule_id), candidates=cases)


@dataclass(frozen=True, slots=True)
class PlannedCase:
    """A normalized invocation with ordered, merged provenance."""

    candidate: LogicalCandidate
    invocation_key: InvocationKey
    provenance: tuple[RuleSource, ...]

    @property
    def id(self) -> str:
        return self.invocation_key.id

    @property
    def payload(self) -> Any:
        return self.candidate.payload


class DecisionKind(str, Enum):
    SCHEDULED = "scheduled"
    DUPLICATE_INVOCATION = "duplicate_invocation"
    UNREACHABLE = "unreachable"
    UNSUPPORTED = "unsupported"


class PopulationConflictError(ValueError):
    """Two candidates share an invocation key but disagree on metadata."""


@dataclass(frozen=True, slots=True)
class CaseDecision:
    """Auditable outcome for one expanded logical candidate."""

    kind: DecisionKind
    candidate: LogicalCandidate
    reason: str | None = None
    invocation_key: InvocationKey | None = None

    @property
    def outcome(self) -> str:
        return self.kind.value


@dataclass(frozen=True, slots=True)
class PopulationReport:
    """Aggregate and optional per-case output from one compiler run."""

    expanded: int
    scheduled: int
    duplicate_invocations: int
    unreachable: int
    unsupported: int
    decisions: tuple[CaseDecision, ...] = ()

    @property
    def filtered(self) -> int:
        return self.unreachable + self.unsupported

    @property
    def counts(self) -> dict[str, int]:
        return {
            "expanded": self.expanded,
            "scheduled": self.scheduled,
            "duplicate_invocations": self.duplicate_invocations,
            "unreachable": self.unreachable,
            "unsupported": self.unsupported,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.counts,
            "decisions": [
                {
                    "outcome": decision.outcome,
                    "reason": decision.reason,
                    "invocation_id": decision.invocation_key.id if decision.invocation_key else None,
                    "source": decision.candidate.source.rule_id,
                }
                for decision in self.decisions
            ],
        }


@dataclass(frozen=True, slots=True)
class PopulationResult:
    """Deterministic plan plus its explanation report."""

    cases: tuple[PlannedCase, ...]
    report: PopulationReport


def legacy_rule(
    cases: Iterable[Any] | Callable[[PlanContext], Iterable[Any]],
    *,
    rule_id: str = "legacy_cases",
    source: RuleSource | None = None,
) -> PopulationRule:
    """Convenience adapter for raw case lists returned by existing collectors."""

    return PopulationRule.from_legacy_cases(cases, rule_id=rule_id, source=source)
