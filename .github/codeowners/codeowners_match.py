# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Canonical CODEOWNERS matcher + resolution pipeline (single source of truth).

The CODEOWNERS pipeline used to have three subtly-different path matchers --
one in `build_codeowners.py` for coverage gating, byte-identical copies in
`emit_codeowners.py` and `who_owns.py` for routing. Coverage and routing could
disagree, and nothing cross-checked them. Everything in the pipeline now goes
through `match()` here so the gate and the artifact are forced to agree.

This module exposes:

  - ``match(pattern, path) -> bool`` -- canonical GitHub CODEOWNERS matcher.
  - ``resolve_owners(rules, path) -> list[str]`` -- last-match-wins resolution.
  - ``parse_codeowners(text) -> list[(pattern, owners)]`` -- shared parser.
  - ``anchor(glob) -> str`` -- repo-root anchoring helper.
  - ``compute_resolution(spec) -> ResolvedModel`` -- pure, tree-independent
    function used by both ``build_codeowners.py`` and ``emit_codeowners.py``.
  - ``minimal_cover(file_team, catch_all)`` -- min-cost last-match cover.
  - Typed shapes (``Area``, ``SharedSpec``, ``FiletypeRule``,
    ``ResolvedArea``, ``FiletypeShared``, ``ResolvedModel``).

GitHub CODEOWNERS semantics implemented here:

  * ``*``                  -- catch-all.
  * ``/foo/``              -- anchored directory subtree.
  * ``/foo`` (no wildcards)-- exact anchored path.
  * ``/foo/*.rs``          -- anchored glob; ``*``/``?`` stop at ``/``,
    ``**`` crosses directories (as GitHub resolves them).
  * ``foo/``               -- unanchored directory; matches any subtree named
    ``foo/`` at any depth.
  * ``*.md`` / ``Dockerfile`` (no slash) -- basename glob at any depth.
  * any with ``/`` and wildcards -- full-path glob, same ``*`` semantics.
"""

from __future__ import annotations

import functools
import re
import subprocess
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict

# ----------------------------------------------------------------------
# Typed shapes (S6)
# ----------------------------------------------------------------------


class Area(TypedDict, total=False):
    """Area entry as declared in ``areas.yaml``."""

    label: str
    github_team: str
    path_globs: list[str]


class SharedSpec(TypedDict):
    """Multi-owner override (``shared:`` entry)."""

    glob: str
    owners: list[str]


class FiletypeRule(TypedDict, total=False):
    """Filetype-level rule (``classify.filetype_rules`` entry).

    ``pattern`` is the single source of truth for the glob; ``coowner`` names
    the area that joins the enclosing owner.
    """

    pattern: str
    coowner: str


@dataclass
class ResolvedArea:
    """An area after auto-classify has expanded its ``path_globs``."""

    label: str
    github_team: str
    path_globs: list[str]


@dataclass
class FiletypeShared:
    """File-type co-ownership row (file glob + ordered owner labels)."""

    glob: str
    owners: list[str]


@dataclass
class ResolvedModel:
    """Resolved taxonomy, ready for emission OR coverage gating.

    Both ``build_codeowners.py`` and ``emit_codeowners.py`` run resolution
    through this dataclass instead of round-tripping a YAML file between the
    two processes.
    """

    catch_all: str
    areas: list[ResolvedArea]
    shared: list[SharedSpec]
    filetype_shared: list[FiletypeShared]
    auto_classified: list[tuple[str, str]] = field(default_factory=list)
    keyword_coowned: list[SharedSpec] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def label_to_team(self) -> dict[str, str]:
        return {a.label: a.github_team for a in self.areas}

    def owned_patterns(self) -> list[str]:
        """Every glob that contributes to explicit (non-catch-all) ownership.

        The set used by the coverage gate -- if some pattern here matches a
        path, the path is "owned" and won't be reported as catch-all-only.
        Area and shared globs are anchored exactly as ``emit_codeowners.py``
        anchors them at emission, so gate and artifact can't disagree: an
        unanchored ``README.md`` must not silently cover ``foo/README.md``.
        Filetype-rule patterns are NOT anchored -- a bare ``*Dockerfile*``
        matches by basename at any depth under GitHub CODEOWNERS rules, and
        the emitter writes it that way.
        """
        pats: list[str] = []
        for area in self.areas:
            pats.extend(anchor(g) for g in area.path_globs)
        for s in self.shared:
            pats.append(anchor(s["glob"]))
        # Filetype patterns are emitted unanchored (basename-any-depth), so
        # keep them unanchored in the coverage set too.
        for fs in self.filetype_shared:
            pats.append(fs.glob)
        return pats

    def unmatched_paths(self, tree: Iterable[str]) -> list[str]:
        """Paths in ``tree`` that fall through to the catch-all only."""
        patterns = self.owned_patterns()
        return [p for p in tree if not any(match(g, p) for g in patterns)]


# ----------------------------------------------------------------------
# Matching primitives (S1)
# ----------------------------------------------------------------------


def _glob_to_re(pattern: str) -> str:
    """Translate a CODEOWNERS glob to a regex: ``*``/``?`` stop at ``/``,
    ``**`` crosses directories. fnmatch is wrong here -- its ``*`` greedily
    crosses path separators, which GitHub's resolver does not."""
    out: list[str] = []
    i = 0
    while i < len(pattern):
        c = pattern[i]
        if c == "*":
            if pattern[i : i + 2] == "**":
                if pattern[i : i + 3] == "**/":
                    # Git's ``**/`` is zero or more complete directories, so
                    # ``a/**/b`` must match both ``a/b`` and ``a/x/b``.
                    out.append("(?:.*/)?")
                    i += 3
                    continue
                out.append(".*")
                i += 2
                continue
            out.append("[^/]*")
        elif c == "?":
            out.append("[^/]")
        elif c == "[":
            j = i + 1
            if j < len(pattern) and pattern[j] in "!^":
                j += 1
            if j < len(pattern) and pattern[j] == "]":
                j += 1
            while j < len(pattern) and pattern[j] != "]":
                j += 1
            if j >= len(pattern):
                out.append(re.escape(c))
            else:
                cls = pattern[i + 1 : j]
                if cls.startswith("!"):
                    cls = "^" + cls[1:]
                out.append("[" + cls + "]")
                i = j
        else:
            out.append(re.escape(c))
        i += 1
    return "".join(out)


@functools.cache
def _compiled(pattern: str) -> re.Pattern[str]:
    return re.compile(_glob_to_re(pattern))


def _glob_match(pattern: str, path: str) -> bool:
    return _compiled(pattern).fullmatch(path) is not None


def match(pattern: str, filepath: str) -> bool:
    """True if ``filepath`` matches ``pattern`` per GitHub CODEOWNERS rules.

    This is the ONLY matcher in the pipeline. Build coverage, emit routing,
    and who_owns lookups all call this. If you change a case, update the
    tests in ``test_codeowners.py``.
    """
    if pattern == "*":
        return True
    if pattern.startswith("/"):
        body = pattern[1:]
        if body.endswith("/"):
            return filepath.startswith(body)
        if any(c in body for c in "*?["):
            return _glob_match(body, filepath)
        return filepath == body
    if pattern.endswith("/"):
        return ("/" + pattern) in ("/" + filepath) or filepath.startswith(pattern)
    if "/" not in pattern:
        base = filepath.rsplit("/", 1)[-1]
        return _glob_match(pattern, base) or _glob_match(pattern, filepath)
    return _glob_match(pattern, filepath)


def resolve_owners(rules: list[tuple[str, list[str]]], filepath: str) -> list[str]:
    """Last-match-wins owners of ``filepath``. ``[]`` if unrouted."""
    owners: list[str] = []
    for pattern, rule_owners in rules:
        if match(pattern, filepath):
            owners = rule_owners
    return owners


def parse_codeowners(text: str) -> list[tuple[str, list[str]]]:
    """Parse a CODEOWNERS file body into ordered ``(pattern, [owner, ...])``."""
    rules: list[tuple[str, list[str]]] = []
    for line in text.splitlines():
        stripped = line.split("#", 1)[0].strip()
        if not stripped:
            continue
        pattern, *owners = stripped.split()
        if owners:
            rules.append((pattern, owners))
    return rules


def anchor(glob: str) -> str:
    """Anchor a glob to repo root for CODEOWNERS output (leading slash)."""
    return glob if glob.startswith("/") else "/" + glob


def load_tree(repo: Path) -> list[str]:
    """Return tracked files under ``repo`` via ``git ls-files``."""
    out = subprocess.check_output(["git", "-C", str(repo), "ls-files"], text=True)
    return [p for p in out.splitlines() if p.strip()]


def changed_paths(repo: Path, base: str) -> list[str]:
    """Paths this branch adds/changes vs ``base`` (``git diff base...HEAD``).

    ``--diff-filter=ACMR`` keeps Added/Copied/Modified/Renamed and drops
    Deletions -- a removed file is not a coverage concern. Three-dot
    ``base...HEAD`` diffs against the merge-base, so a long-running branch is
    judged only on the surface it actually touched, not on unrelated paths
    that landed on ``base`` after it forked. This is the diff-aware input to
    the ``--strict`` gate; it reads the tree, like ``load_tree``, and lives
    only in the coverage tool, never in emission.
    """
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo), "diff", "--name-only", "--diff-filter=ACMR", f"{base}...HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as err:
        raise SystemExit(f"git diff failed in {repo!r} (not a checkout, or base {base!r} unavailable): {err}") from err
    return [p for p in out.splitlines() if p.strip()]


# ----------------------------------------------------------------------
# Taxonomy validation
# ----------------------------------------------------------------------


_GITHUB_LOGIN = r"(?:[A-Za-z0-9]|[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9]))"
_GITHUB_TEAM_SLUG = r"(?:[A-Za-z0-9]|[A-Za-z0-9](?:[A-Za-z0-9_-]*[A-Za-z0-9]))"
_GITHUB_OWNER_RE = re.compile(rf"^@{_GITHUB_LOGIN}(?:/{_GITHUB_TEAM_SLUG})?$")


def validate_github_owner(owner: object, *, field: str) -> str:
    """Return one valid ``@user`` or ``@org/team`` token, or fail closed."""
    if not isinstance(owner, str) or _GITHUB_OWNER_RE.fullmatch(owner) is None:
        raise SystemExit(f"{field} must be one GitHub owner token ('@user' or '@org/team'), got {owner!r}")
    return owner


def _validate_owner_rules(section: str, rules: object, labels: set[str]) -> None:
    if not isinstance(rules, list):
        raise SystemExit(f"areas.yaml: {section} must be a list")
    for index, rule in enumerate(rules):
        field = f"{section}[{index}]"
        if not isinstance(rule, dict):
            raise SystemExit(f"areas.yaml: {field} must be a mapping")
        owners = rule.get("owners")
        if not isinstance(owners, list) or not owners:
            raise SystemExit(f"areas.yaml: {field}.owners must be a non-empty list")
        for owner in owners:
            if isinstance(owner, str) and owner in labels:
                continue
            validate_github_owner(owner, field=f"areas.yaml: {field}.owners")


def _validate_classification_rules(classify: object, labels: set[str]) -> None:
    if not isinstance(classify, dict):
        raise SystemExit("areas.yaml: classify must be a mapping")

    keyword_rules = classify.get("keyword_rules", []) or []
    if not isinstance(keyword_rules, list):
        raise SystemExit("areas.yaml: classify.keyword_rules must be a list")
    for index, rule in enumerate(keyword_rules):
        field = f"classify.keyword_rules[{index}]"
        if not isinstance(rule, dict):
            raise SystemExit(f"areas.yaml: {field} must be a mapping")
        for target in ("area", "coowner"):
            if target in rule and rule[target] not in labels:
                raise SystemExit(f"areas.yaml: {field}.{target} must name a declared area; got {rule[target]!r}")

    filetype_rules = classify.get("filetype_rules", []) or []
    if not isinstance(filetype_rules, list):
        raise SystemExit("areas.yaml: classify.filetype_rules must be a list")
    for index, rule in enumerate(filetype_rules):
        field = f"classify.filetype_rules[{index}]"
        if not isinstance(rule, dict):
            raise SystemExit(f"areas.yaml: {field} must be a mapping")
        if "advisory" in rule:
            raise SystemExit(f"areas.yaml: {field}.advisory was removed; use blocking CODEOWNERS ownership")
        if rule.get("coowner") not in labels:
            raise SystemExit(f"areas.yaml: {field}.coowner must name a declared area; got {rule.get('coowner')!r}")


def validate_spec(spec: object) -> None:
    """Validate ownership-bearing fields before coverage or emission.

    Coverage is a security boundary: a malformed owner must never count as an
    explicit claim merely because its glob matches a tracked path.
    """
    if not isinstance(spec, dict):
        raise SystemExit("areas.yaml: document must be a mapping")
    if "advisory" in spec:
        raise SystemExit("areas.yaml: advisory was removed; use shared blocking CODEOWNERS ownership")

    meta = spec.get("meta", {}) or {}
    if not isinstance(meta, dict):
        raise SystemExit("areas.yaml: meta must be a mapping")
    validate_github_owner(meta.get("catch_all"), field="areas.yaml: meta.catch_all")

    raw_areas = spec.get("areas", [])
    if not isinstance(raw_areas, list):
        raise SystemExit("areas.yaml: areas must be a list")
    labels: set[str] = set()
    for index, area in enumerate(raw_areas):
        field = f"areas[{index}]"
        if not isinstance(area, dict):
            raise SystemExit(f"areas.yaml: {field} must be a mapping")
        label = area.get("label")
        if not isinstance(label, str) or not label.strip():
            raise SystemExit(f"areas.yaml: {field}.label must be non-empty")
        if label in labels:
            raise SystemExit(f"areas.yaml: duplicate area label {label!r}")
        labels.add(label)
        validate_github_owner(area.get("github_team"), field=f"areas.yaml: {field}.github_team")

    _validate_owner_rules("shared", spec.get("shared", []) or [], labels)
    _validate_classification_rules(spec.get("classify", {}) or {}, labels)


# ----------------------------------------------------------------------
# Resolution pipeline (S2, S5)
# ----------------------------------------------------------------------
#
# ``compute_resolution`` is a PURE FUNCTION of the parsed policy YAML: it
# never touches ``git ls-files``. The base-branch race the old generator
# suffered from -- unrelated tree churn on ``main`` mutating the minimal
# cover, the auto-classified globs, or the filetype co-ownership rows for a
# PR that changed none of them -- came entirely from resolving against a
# live tree at emit time. Coverage is still checked against the tree, but the
# check lives in ``build_codeowners.py --strict`` and does not feed back into
# the emitted rules.


def compute_resolution(spec: dict, tree: list[str] | None = None) -> ResolvedModel:
    """Resolve an ``areas.yaml`` spec into the model the emitter renders.

    Pure function of ``spec``: no tree, no filesystem, no ``git``. Same YAML
    in -> byte-identical model out, so the emitted CODEOWNERS is a pure
    function of the policy inputs. ``tree`` is accepted for backward
    compatibility (older callers passed it) and ignored.

    Semantics per section:

    * ``areas``       -- ``path_globs`` are emitted verbatim (sorted).
    * ``shared``      -- passed through as declared.
    * ``classify.filetype_rules`` -- each rule becomes one stable row with the
      coowner as the sole owner (a single ``*Dockerfile*`` line owns every
      Dockerfile via last-match at any depth).
    * ``classify.keyword_rules`` -- validated, then ignored at emit time.
      Auto-promotion of unmatched dirs into an area, and keyword-level
      co-ownership, both required walking the live tree -- pure poison for a
      stable output. Authors materialize the equivalent as explicit
      ``path_globs`` / ``shared`` entries in ``areas.yaml``; the strict
      coverage gate catches any new directory that slipped through.
    """
    validate_spec(spec)
    if tree is not None:
        # Deprecated argument: accepted for legacy callers, ignored so
        # emission stays a pure function of the policy inputs.
        _ = tree

    catch_all = spec.get("meta", {}).get("catch_all", "")
    raw_areas = spec.get("areas", [])
    classify = spec.get("classify", {}) or {}
    filetype_rules: list[FiletypeRule] = classify.get("filetype_rules", []) or []

    spec_shared: list[SharedSpec] = spec.get("shared", []) or []

    areas = [
        ResolvedArea(
            label=a["label"],
            github_team=a["github_team"],
            path_globs=sorted(set(a.get("path_globs", []) or [])),
        )
        for a in raw_areas
    ]

    # Blocking filetype rule -> one stable coowner-only row (bare pattern
    # matches by basename at any depth per GitHub CODEOWNERS semantics). The
    # old "enclosing area + coowner" behavior required walking the tree; if a
    # specific subtree wants that co-ownership, declare it explicitly in
    # ``shared`` with a path glob.
    filetype_shared: list[FiletypeShared] = []
    for rule in filetype_rules:
        pattern = rule.get("pattern")
        coowner = rule.get("coowner")
        if not pattern or not coowner:
            continue
        filetype_shared.append(FiletypeShared(glob=pattern, owners=[coowner]))

    return ResolvedModel(
        catch_all=catch_all,
        areas=areas,
        shared=list(spec_shared),
        filetype_shared=filetype_shared,
        auto_classified=[],
        keyword_coowned=[],
        meta=dict(spec.get("meta", {})),
    )


# ----------------------------------------------------------------------
# minimal_cover (S5, S7) -- min-cost last-match cover
# ----------------------------------------------------------------------


def minimal_cover(file_team: dict[str, str], catch_all: str) -> list[tuple[str, str]]:
    """Smallest set of base rules reproducing ``file_team`` under last-match.

    Returns ``(anchored_pattern, team)`` pairs: directory globs covering whole
    subtrees plus file globs for in-directory exceptions. The catch-all is
    the root default and is NOT returned. Emit shortest-path-first (or
    grouped, with deeper rules after) so a more-specific rule still wins.
    """
    children: dict[str, set[str]] = defaultdict(set)
    dir_files: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for path, team in file_team.items():
        parts = path.split("/")
        for i in range(1, len(parts)):
            children["/".join(parts[: i - 1])].add("/".join(parts[:i]))
        dir_files["/".join(parts[:-1])].append((path, team))

    subtree: dict[str, set[str]] = {}

    def teams_under(d: str) -> set[str]:
        if d not in subtree:
            ts = {t for _, t in dir_files.get(d, ())}
            for c in children.get(d, ()):
                ts |= teams_under(c)
            subtree[d] = ts
        return subtree[d]

    memo: dict[tuple[str, str], int] = {}

    def cost(d: str, inh: str) -> int:
        key = (d, inh)
        if key not in memo:
            best = None
            for c in {inh} | teams_under(d):
                x = 0 if c == inh else 1
                x += sum(1 for _, t in dir_files.get(d, ()) if t != c)
                x += sum(cost(ch, c) for ch in children.get(d, ()))
                best = x if best is None else min(best, x)
            memo[key] = best or 0
        return memo[key]

    def choose(d: str, inh: str) -> str:
        best_c, best_x = inh, None
        for c in [inh, *sorted(teams_under(d) - {inh})]:
            x = 0 if c == inh else 1
            x += sum(1 for _, t in dir_files.get(d, ()) if t != c)
            x += sum(cost(ch, c) for ch in children.get(d, ()))
            if best_x is None or x < best_x:
                best_c, best_x = c, x
        return best_c

    rules: list[tuple[str, str]] = []

    def emit(d: str, inh: str) -> None:
        c = catch_all if d == "" else choose(d, inh)
        if d != "" and c != inh:
            rules.append(("/" + d + "/", c))
        for path, team in dir_files.get(d, ()):
            if team != c:
                rules.append(("/" + path, team))
        for ch in sorted(children.get(d, ())):
            emit(ch, c)

    emit("", catch_all)
    return rules
