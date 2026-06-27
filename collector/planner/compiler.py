# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common lazy compiler for additive collector population rules."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Any

from collector.planner.models import (
    CaseDecision,
    DecisionKind,
    InvocationKey,
    LogicalCandidate,
    PlanContext,
    PlannedCase,
    PopulationConflictError,
    PopulationReport,
    PopulationResult,
    PopulationRule,
    RuleSource,
    canonical_json,
)
from collector.planner.registry import OpCaseSchema, get_schema


def _normalized_candidate(value: LogicalCandidate | Any, original: LogicalCandidate) -> LogicalCandidate:
    if isinstance(value, LogicalCandidate):
        return value
    return LogicalCandidate(payload=value, source=original.source, metadata=original.metadata)


def _filter_result(result: bool | tuple[bool, str | None], default_reason: str) -> tuple[bool, str | None]:
    if isinstance(result, tuple):
        allowed, reason = result
        return bool(allowed), reason if reason is not None else (None if allowed else default_reason)
    allowed = bool(result)
    return allowed, None if allowed else default_reason


def _merge_provenance(existing: tuple[RuleSource, ...], source: RuleSource) -> tuple[RuleSource, ...]:
    source_id = source.id
    if any(item.id == source_id for item in existing):
        return existing
    return (*existing, source)


def _merge_metadata(existing: Mapping[str, Any], incoming: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(existing)
    for key, value in incoming.items():
        if key in merged and canonical_json(merged[key]) != canonical_json(value):
            raise PopulationConflictError(
                f"duplicate invocation has conflicting metadata for {key!r}: {merged[key]!r} != {value!r}"
            )
        merged[key] = value
    return merged


def compile_population(
    rules: Iterable[PopulationRule],
    context: PlanContext,
    *,
    schema: OpCaseSchema | None = None,
    record_decisions: bool = True,
) -> PopulationResult:
    """Compile additive rules into an ordered, invocation-deduplicated plan.

    Each rule is fully independent: the compiler iterates its candidates and
    never merges dimensions or candidate payloads between rules.  Filtering is
    performed as each candidate is yielded, so an op schema may expand lazily.
    """

    resolved_schema = schema or get_schema(context.op, backend=context.backend)
    planned: OrderedDict[InvocationKey, PlannedCase] = OrderedDict()
    decisions: list[CaseDecision] = []
    expanded = duplicate_invocations = unreachable = unsupported = 0

    def record(decision: CaseDecision) -> None:
        if record_decisions:
            decisions.append(decision)

    for rule in rules:
        for candidate in rule.expand(context):
            expanded += 1
            normalized = _normalized_candidate(resolved_schema.normalize(candidate, context), candidate)

            reachable, reason = _filter_result(
                resolved_schema.is_reachable(normalized, context),
                "unreachable model shape",
            )
            if not reachable:
                unreachable += 1
                record(CaseDecision(DecisionKind.UNREACHABLE, normalized, reason=reason))
                continue

            supported, reason = _filter_result(
                resolved_schema.is_supported(normalized, context),
                "unsupported backend or hardware",
            )
            if not supported:
                unsupported += 1
                record(CaseDecision(DecisionKind.UNSUPPORTED, normalized, reason=reason))
                continue

            raw_key = resolved_schema.invocation_key(normalized, context)
            invocation_key = (
                raw_key if isinstance(raw_key, InvocationKey) else InvocationKey.from_value(context, raw_key)
            )
            existing = planned.get(invocation_key)
            if existing is not None:
                duplicate_invocations += 1
                planned[invocation_key] = replace(
                    existing,
                    candidate=replace(
                        existing.candidate,
                        metadata=_merge_metadata(existing.candidate.metadata, normalized.metadata),
                    ),
                    provenance=_merge_provenance(existing.provenance, normalized.source),
                )
                record(
                    CaseDecision(
                        DecisionKind.DUPLICATE_INVOCATION,
                        normalized,
                        reason="same normalized invocation key",
                        invocation_key=invocation_key,
                    )
                )
                continue

            planned_case = PlannedCase(
                candidate=normalized,
                invocation_key=invocation_key,
                provenance=(normalized.source,),
            )
            planned[invocation_key] = planned_case
            record(CaseDecision(DecisionKind.SCHEDULED, normalized, invocation_key=invocation_key))

    report = PopulationReport(
        expanded=expanded,
        scheduled=len(planned),
        duplicate_invocations=duplicate_invocations,
        unreachable=unreachable,
        unsupported=unsupported,
        decisions=tuple(decisions),
    )
    return PopulationResult(cases=tuple(planned.values()), report=report)


# Short action-oriented alias used by integrations.
populate = compile_population
