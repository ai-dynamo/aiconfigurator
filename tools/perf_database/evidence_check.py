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
or policy file, or a policy that cannot resolve an evidence system for a rule
that needs one, all abort with a loud error and exit 1 — never a silent
partial answer.

Usage:
    evidence_check.py --manifest changed_ops.yaml [--policy PATH] [--out FILE]
"""

from __future__ import annotations

import argparse
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
    evidence_systems: tuple[str, ...]  # resolved from evidence_systems{generation: system}, sorted+deduped
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
    if not isinstance(threshold_value, (int, float)) or isinstance(threshold_value, bool):
        raise EvidencePolicyError(f"evidence policy 'thresholds.parquet_diff_median_pct' must be a number: {path}")

    evidence_systems = _resolve_evidence_systems(raw.get("evidence_systems"), path)

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
        evidence_systems=evidence_systems,
        rule_types=rule_types,
        exceptions_file=exceptions_file,
    )


def _resolve_evidence_systems(raw_evidence_systems: Any, path: Path) -> tuple[str, ...]:
    """{SM generation: system name} -> sorted, deduped system names.

    Fails closed (distinctly from generic malformed-policy errors) when a
    generation cannot be resolved to a concrete system: an empty mapping, or
    a blank/non-string value for some generation.
    """
    if not isinstance(raw_evidence_systems, dict) or not raw_evidence_systems:
        raise EvidencePolicyError(
            f"unresolved evidence_system: evidence policy 'evidence_systems' must be a non-empty "
            f"mapping of SM generation -> system: {path}"
        )
    resolved: set[str] = set()
    for generation, system in raw_evidence_systems.items():
        if not isinstance(system, str) or not system.strip():
            raise EvidencePolicyError(
                f"unresolved evidence_system for generation {generation!r}: value must be a non-empty string: {path}"
            )
        resolved.add(system)
    return tuple(sorted(resolved))


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


def _requirement_for(reason: str, entry: ChangedEntry, policy: Policy) -> dict[str, Any]:
    # case_plan is additive/GPU-cheap by design (§9 row 3): it never needs a
    # designated evidence system, unlike pin_version's declared-reuse spot
    # bench or collector_code's before/after diff.
    evidence_systems = () if reason == "case_plan" else policy.evidence_systems

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
