# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evidence resolver (AIC-1219): changed-operation manifest -> required evidence.

Collector V3 design §9 (docs/perf_database/collector-v3-op-centric-design.md):
a pure-function resolver over `collector/evidence_policy.yaml` and the
`tools/perf_database/changed_ops.py` manifest (design §8, a LOCKED schema).
Consumed by the AIC-1214 CI gate and the support-matrix healer — both must
get identical requirements from identical (manifest, policy) inputs, which is
why the core function (`resolve_requirements`) does no I/O and is
deterministic: sorted entries, sorted tables/systems/evidence_systems, and a
canonical reason order (`pin_version`, `collector_code`, `case_plan`) so byte-
identical inputs always produce byte-identical output.

Per changed (framework, family) entry, every reason present emits its own
requirement item — reasons are never merged or deduplicated into "the
strictest one", because bundling a change under multiple reasons must not let
it dodge any single reason's evidence (design §9: "combined reasons emit the
union of requirements").

Fail-closed: an unknown `reasons` value in the manifest, a malformed manifest
or policy file, a changed entry's `systems` naming a system the policy's
`system_generations` doesn't map to any SM generation, or a policy that
cannot resolve an evidence system for a rule that needs one, all abort with a
loud error and exit 1 — never a silent partial answer.

Usage:
    evidence_check.py --manifest changed_ops.yaml [--policy PATH] [--out FILE]
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY_PATH = REPO_ROOT / "collector" / "evidence_policy.yaml"

# The changed_ops.py manifest schema (design §8) admits exactly these three
# reasons. Also the canonical, deterministic emission order for a combined
# entry's requirements.
KNOWN_REASONS = ("pin_version", "collector_code", "case_plan")

EXIT_OK = 0
EXIT_ERROR = 1


class EvidencePolicyError(ValueError):
    """`collector/evidence_policy.yaml` is missing, malformed, or cannot
    resolve an evidence system a rule needs (fail-closed).
    """


class EvidenceManifestError(ValueError):
    """The changed_ops manifest is missing, malformed, or names a reason the
    policy does not recognize (fail-closed).
    """


# --------------------------------------------------------------------------
# policy loading
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Policy:
    threshold_pct: float
    system_generations: dict[str, frozenset[str]]  # SM generation -> member systems (authored fleet map)
    evidence_systems: dict[str, str]  # SM generation -> representative system, validated against system_generations
    rule_types: dict[str, str]  # reason -> requirement type name (authored in the policy file)
    exceptions_file: str


def load_policy(path: Path) -> Policy:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise EvidencePolicyError(f"evidence policy not found: {path}") from exc
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise EvidencePolicyError(f"evidence policy is not valid YAML: {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise EvidencePolicyError(f"evidence policy must be a YAML mapping: {path}")

    if raw.get("schema_version") != 1:
        raise EvidencePolicyError(
            f"evidence policy schema_version must be 1, got {raw.get('schema_version')!r}: {path}"
        )

    thresholds = raw.get("thresholds")
    threshold_value = thresholds.get("parquet_diff_median_pct") if isinstance(thresholds, dict) else None
    if (
        not isinstance(threshold_value, (int, float))
        or isinstance(threshold_value, bool)
        or not math.isfinite(threshold_value)
        or threshold_value < 0
    ):
        raise EvidencePolicyError(
            f"evidence policy 'thresholds.parquet_diff_median_pct' must be a finite non-negative number: {path}"
        )

    system_generations = _resolve_system_generations(raw.get("system_generations"), path)
    evidence_systems = _resolve_evidence_systems(raw.get("evidence_systems"), system_generations, path)

    rules = raw.get("rules")
    if not isinstance(rules, dict):
        raise EvidencePolicyError(f"evidence policy 'rules' must be a mapping: {path}")
    rule_types: dict[str, str] = {}
    for reason in KNOWN_REASONS:
        rule = rules.get(reason)
        requirement = rule.get("requirement") if isinstance(rule, dict) else None
        if not isinstance(requirement, str) or not requirement.strip():
            raise EvidencePolicyError(f"evidence policy missing/invalid 'rules.{reason}.requirement': {path}")
        rule_types[reason] = requirement

    exceptions_file = raw.get("exceptions_file")
    if not isinstance(exceptions_file, str) or not exceptions_file.strip():
        raise EvidencePolicyError(f"evidence policy 'exceptions_file' must be a non-empty string: {path}")

    return Policy(
        threshold_pct=float(threshold_value),
        system_generations=system_generations,
        evidence_systems=evidence_systems,
        rule_types=rule_types,
        exceptions_file=exceptions_file,
    )


def _resolve_system_generations(raw_system_generations: Any, path: Path) -> dict[str, frozenset[str]]:
    """{SM generation: [system, ...]} -> {SM generation: frozenset(system, ...)}.

    Fails closed when the mapping is empty/malformed, a generation's
    system list is missing, empty, or contains a non-string/blank entry, or
    a system is listed under more than one generation (ambiguous ownership
    would silently match every owning generation in ``_touched_generations``).
    """
    if not isinstance(raw_system_generations, dict) or not raw_system_generations:
        raise EvidencePolicyError(
            f"evidence policy 'system_generations' must be a non-empty mapping of "
            f"SM generation -> [system, ...]: {path}"
        )
    resolved: dict[str, frozenset[str]] = {}
    owners: dict[str, str] = {}
    for generation, systems in raw_system_generations.items():
        if (
            not isinstance(systems, list)
            or not systems
            or not all(isinstance(system, str) and system.strip() for system in systems)
        ):
            raise EvidencePolicyError(
                f"evidence policy 'system_generations.{generation}' must be a non-empty list of system names: {path}"
            )
        for system in systems:
            if system in owners:
                raise EvidencePolicyError(
                    f"evidence policy 'system_generations' lists system {system!r} under both "
                    f"{owners[system]!r} and {generation!r}; each system must belong to exactly "
                    f"one SM generation: {path}"
                )
            owners[system] = generation
        resolved[generation] = frozenset(systems)
    return resolved


def _resolve_evidence_systems(
    raw_evidence_systems: Any, system_generations: dict[str, frozenset[str]], path: Path
) -> dict[str, str]:
    """{SM generation: representative system} -> validated mapping.

    Fails closed (distinctly from generic malformed-policy errors) when: the
    mapping is empty/malformed; a value is blank/non-string; a generation key
    has no entry in `system_generations` (unresolved evidence_system); or the
    representative is not itself a member of its own generation.
    """
    if not isinstance(raw_evidence_systems, dict) or not raw_evidence_systems:
        raise EvidencePolicyError(
            f"unresolved evidence_system: evidence policy 'evidence_systems' must be a non-empty "
            f"mapping of SM generation -> system: {path}"
        )
    resolved: dict[str, str] = {}
    for generation, system in raw_evidence_systems.items():
        if not isinstance(system, str) or not system.strip():
            raise EvidencePolicyError(
                f"unresolved evidence_system for generation {generation!r}: value must be a non-empty string: {path}"
            )
        if generation not in system_generations:
            raise EvidencePolicyError(
                f"unresolved evidence_system: evidence policy 'evidence_systems' generation {generation!r} "
                f"has no entry in 'system_generations': {path}"
            )
        if system not in system_generations[generation]:
            raise EvidencePolicyError(
                f"unresolved evidence_system: evidence policy 'evidence_systems.{generation}' representative "
                f"{system!r} is not a member of system_generations.{generation!r}: {path}"
            )
        resolved[generation] = system
    return resolved


# --------------------------------------------------------------------------
# manifest loading (design §8's locked `changed_ops.py` schema)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ChangedEntry:
    framework: str
    family: str
    reasons: tuple[str, ...]
    tables: tuple[str, ...]
    systems: tuple[str, ...]


def load_manifest(path: Path) -> list[ChangedEntry]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise EvidenceManifestError(f"changed_ops manifest not found: {path}") from exc
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise EvidenceManifestError(f"changed_ops manifest is not valid YAML: {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise EvidenceManifestError(f"changed_ops manifest must be a YAML mapping: {path}")

    changed_raw = raw.get("changed")
    if not isinstance(changed_raw, list):
        raise EvidenceManifestError(f"changed_ops manifest 'changed' must be a list: {path}")

    entries: list[ChangedEntry] = []
    for index, item in enumerate(changed_raw):
        entries.append(_parse_changed_entry(item, index, path))
    return entries


def _parse_changed_entry(item: Any, index: int, path: Path) -> ChangedEntry:
    if not isinstance(item, dict):
        raise EvidenceManifestError(f"changed_ops manifest 'changed[{index}]' must be a mapping: {path}")

    framework = item.get("framework")
    family = item.get("family")
    reasons = item.get("reasons")
    tables = item.get("tables")
    systems = item.get("systems")

    if not isinstance(framework, str) or not isinstance(family, str):
        raise EvidenceManifestError(
            f"changed_ops manifest 'changed[{index}]': 'framework'/'family' must be strings: {path}"
        )
    if not isinstance(reasons, list) or not reasons:
        raise EvidenceManifestError(
            f"changed_ops manifest 'changed[{index}]': 'reasons' must be a non-empty list: {path}"
        )
    if not isinstance(tables, list) or not isinstance(systems, list):
        raise EvidenceManifestError(
            f"changed_ops manifest 'changed[{index}]': 'tables'/'systems' must be lists: {path}"
        )
    if not all(isinstance(table, str) for table in tables) or not all(isinstance(system, str) for system in systems):
        raise EvidenceManifestError(
            f"changed_ops manifest 'changed[{index}]': 'tables'/'systems' items must be strings: {path}"
        )

    for reason in reasons:
        if reason not in KNOWN_REASONS:
            raise EvidenceManifestError(
                f"changed_ops manifest 'changed[{index}]' ({framework}/{family}): unknown reason {reason!r}; "
                f"known reasons: {KNOWN_REASONS}: {path}"
            )

    return ChangedEntry(
        framework=framework,
        family=family,
        reasons=tuple(reasons),
        tables=tuple(tables),
        systems=tuple(systems),
    )


# --------------------------------------------------------------------------
# resolution: pure function, no I/O
# --------------------------------------------------------------------------


def _touched_generations(entry: ChangedEntry, policy: Policy) -> set[str]:
    """The SM generations `entry.systems` belong to, per policy `system_generations`.

    Fail-closed: a system not mapped to any generation aborts loudly (design
    review AIC-1219) instead of silently omitting it from the evidence scope.
    """
    generations: set[str] = set()
    for system in entry.systems:
        matches = [generation for generation, members in policy.system_generations.items() if system in members]
        if not matches:
            raise EvidencePolicyError(
                f"changed_ops entry ({entry.framework}/{entry.family}): system {system!r} is not mapped to "
                f"any SM generation in evidence policy 'system_generations'"
            )
        generations.update(matches)
    return generations


def _requirement_for(reason: str, entry: ChangedEntry, policy: Policy) -> dict[str, Any]:
    # case_plan is additive/GPU-cheap by design (§9 row 3): it never needs a
    # designated evidence system, unlike pin_version's declared-reuse spot
    # bench or collector_code's before/after diff. For those two, scope the
    # evidence systems to the SM generations the entry's `systems` actually
    # touch — demanding e.g. Hopper evidence for a Blackwell-only (nvfp4)
    # family is both wasted and, per capabilities.yaml, sometimes impossible
    # to satisfy. The unmapped-system fail-closed check runs for EVERY
    # reason, case_plan included — only representative selection is skipped.
    touched = _touched_generations(entry, policy)
    if reason == "case_plan":
        evidence_systems: tuple[str, ...] = ()
    else:
        missing = touched - policy.evidence_systems.keys()
        if missing:
            raise EvidencePolicyError(
                f"changed_ops entry ({entry.framework}/{entry.family}) touches SM generation(s) "
                f"{sorted(missing)} with no representative in evidence policy 'evidence_systems'"
            )
        evidence_systems = tuple(sorted(policy.evidence_systems[generation] for generation in touched))

    requirement: dict[str, Any] = {
        "type": policy.rule_types[reason],
        "tables": sorted(set(entry.tables)),
        "systems": sorted(set(entry.systems)),
        "evidence_systems": list(evidence_systems),
    }
    if reason == "collector_code":
        requirement["threshold"] = policy.threshold_pct
    return requirement


def resolve_requirements(policy: Policy, entries: list[ChangedEntry]) -> list[dict[str, Any]]:
    """Pure function: (policy, changed entries) -> deterministic requirements.

    Every reason on an entry contributes its own requirement item — the
    union, never a single "strictest" pick — so a change cannot dodge one
    reason's evidence by being bundled with another (design §9).
    """
    items: list[dict[str, Any]] = []
    for entry in entries:
        ordered_reasons = [reason for reason in KNOWN_REASONS if reason in entry.reasons]
        items.append(
            {
                "framework": entry.framework,
                "family": entry.family,
                "reasons": ordered_reasons,
                "requirements": [_requirement_for(reason, entry, policy) for reason in ordered_reasons],
            }
        )
    items.sort(key=lambda item: (item["framework"], item["family"]))
    return items


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def render_report(items: list[dict[str, Any]]) -> str:
    return yaml.safe_dump({"requirements": items}, sort_keys=False, default_flow_style=False)


def render_summary(items: list[dict[str, Any]]) -> str:
    if not items:
        return "no evidence required\n"
    lines = [f"evidence required for {len(items)} changed (framework, family) pair(s):"]
    for item in items:
        reasons = ", ".join(item["reasons"])
        types = ", ".join(requirement["type"] for requirement in item["requirements"])
        lines.append(f"  - {item['framework']}/{item['family']}: reasons=[{reasons}] requirements=[{types}]")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, type=Path, help="changed_ops.py output (design §8 schema)")
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY_PATH, help="evidence_policy.yaml path")
    parser.add_argument("--out", type=Path, default=None, help="write machine-readable yaml here instead of stdout")
    args = parser.parse_args(argv)

    try:
        policy = load_policy(args.policy)
        entries = load_manifest(args.manifest)
        items = resolve_requirements(policy, entries)
    except (EvidencePolicyError, EvidenceManifestError) as exc:
        print(f"evidence_check: {exc}", file=sys.stderr)
        return EXIT_ERROR

    sys.stderr.write(render_summary(items))

    report = render_report(items)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report)
    else:
        sys.stdout.write(report)

    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
