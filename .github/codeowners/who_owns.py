#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""who_owns.py -- "who reviews this?" from a generated CODEOWNERS.

The CODEOWNERS file is a machine input: GitHub auto-requests the owning team
when a PR opens. This tool answers the human question on demand, so nobody
has to read 300 rules to find a reviewer.

  # owners of specific paths (last-match-wins, exactly as GitHub resolves)
  python who_owns.py --codeowners CODEOWNERS lib/llm/foo.rs components/.../snapshot.py

  # the teams that will be auto-requested on your PR (union over changed files)
  python who_owns.py --codeowners CODEOWNERS --changed --base main

  # same, expanding each team to its member logins (org members only --
  # GitHub does not show team membership to non-members)
  python who_owns.py --codeowners CODEOWNERS --changed --people

Owners listed on a single line are co-owners (any one's approval satisfies
the gate).

The CODEOWNERS parser and matcher live in ``codeowners_match`` so this tool
resolves a path exactly the same way ``emit_codeowners.py`` routes it -- there
is no second implementation that could drift.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from codeowners_match import parse_codeowners, resolve_owners

_TEAM_CACHE: dict[str, list[str] | None] = {}
_PEOPLE_WARNED = False


def _gh_fetch(org: str, slug: str) -> list[str]:
    """Fetch a team's member logins via the ``gh`` CLI."""
    out = subprocess.check_output(
        [
            "gh",
            "api",
            f"orgs/{org}/teams/{slug}/members",
            "--paginate",
            "--jq",
            ".[].login",
        ],
        text=True,
        stderr=subprocess.DEVNULL,
    )
    return [line for line in out.splitlines() if line.strip()]


def _parse_team(owner: str):
    """``@org/slug`` -> ``(org, slug)``; ``None`` for individual ``@handle``s."""
    if owner.startswith("@") and "/" in owner:
        return tuple(owner[1:].split("/", 1))
    return None


def team_members(team, fetch=_gh_fetch, cache=None):
    """Member logins for an ``@org/slug`` team, or ``None`` when unavailable.

    GitHub only shows team membership to members of the org, so this works
    for org members with an authenticated ``gh`` and cannot work for external
    contributors -- callers degrade to the team handle. Individual ``@handle``
    owners (no slash) return ``None`` and pass through unchanged. Results,
    including failures, are cached per run.
    """
    if cache is None:
        cache = _TEAM_CACHE
    if team in cache:
        return cache[team]
    members = None
    if parsed := _parse_team(team):
        try:
            members = sorted(fetch(*parsed))
        except (OSError, subprocess.CalledProcessError):
            members = None
    cache[team] = members
    return members


def _warn_people_unavailable() -> None:
    global _PEOPLE_WARNED
    if not _PEOPLE_WARNED:
        print(
            "note: team membership is only visible to org members (authenticated gh required); showing teams only",
            file=sys.stderr,
        )
        _PEOPLE_WARNED = True


def _with_people(owner: str) -> str:
    """Render an owner as ``@org/team (member, member, ...)`` when possible."""
    members = team_members(owner)
    if members is None:
        if _parse_team(owner):
            _warn_people_unavailable()
        return owner
    return f"{owner} ({', '.join(members) if members else 'no members'})"


def _team_url(team: str) -> str | None:
    if parsed := _parse_team(team):
        return "https://github.com/orgs/{}/teams/{}".format(*parsed)
    return None


def changed_files(repo: str, base: str) -> list[str]:
    """Files changed vs ``base`` (merge-base diff), falling back to plain diff.

    Untracked (not yet staged) files are included too -- brand-new paths are
    exactly the ones the coverage gate cares about, and ``git diff`` alone
    never shows them.

    Unstaged working-tree edits are included so local modifications that are
    not yet in the index still appear in the result.

    Returns ``[]`` only when the lookups actually succeeded and were empty.
    If everything fails (not a git checkout, unknown base), the last git
    error is surfaced instead of masquerading as "no changed files".
    """
    last_err: OSError | subprocess.CalledProcessError | None = None
    diff_ok = False
    changed: list[str] = []
    for args in ([f"{base}...HEAD"], [base]):
        try:
            out = subprocess.check_output(
                ["git", "-C", repo, "diff", "--name-only", *args],
                text=True,
                stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.CalledProcessError) as err:
            last_err = err
            continue
        diff_ok = True
        changed = [p for p in out.splitlines() if p.strip()]
        break
    untracked: list[str] = []
    try:
        out = subprocess.check_output(
            ["git", "-C", repo, "ls-files", "--others", "--exclude-standard"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        untracked = [p for p in out.splitlines() if p.strip()]
    except (OSError, subprocess.CalledProcessError):
        pass
    if not diff_ok:
        raise SystemExit(f"git diff failed in {repo!r} (not a checkout, or base {base!r} unavailable): {last_err}")
    working_tree: list[str] = []
    try:
        out = subprocess.check_output(
            ["git", "-C", repo, "diff", "--name-only"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        working_tree = [p for p in out.splitlines() if p.strip()]
    except (OSError, subprocess.CalledProcessError):
        pass
    return sorted(set(changed) | set(working_tree) | set(untracked))


def main() -> int:
    ap = argparse.ArgumentParser(description="Who reviews a path, per a generated CODEOWNERS.")
    ap.add_argument(
        "--codeowners",
        required=True,
        type=Path,
        help="path to the CODEOWNERS file",
    )
    ap.add_argument(
        "--changed",
        action="store_true",
        help="resolve the repo's changed files instead of explicit paths",
    )
    ap.add_argument("--base", default="main", help="base ref for --changed (default: main)")
    ap.add_argument(
        "--people",
        action="store_true",
        help="expand teams to member logins (org members with an "
        "authenticated gh only; membership is not publicly visible)",
    )
    ap.add_argument("--repo", default=".", help="repo root for --changed (default: .)")
    ap.add_argument("paths", nargs="*", help="paths to resolve (when not using --changed)")
    args = ap.parse_args()

    rules = parse_codeowners(args.codeowners.read_text())
    if args.changed:
        files = changed_files(args.repo, args.base)
        if not files:
            print(f"No changed files vs {args.base}.")
            return 0
    else:
        files = args.paths
        if not files:
            ap.error("pass one or more paths, or use --changed")

    # Per-path expansion only for explicit paths (small N); --changed PRs can
    # touch many files, so people are shown once in the union summary instead.
    expand_paths = args.people and not args.changed
    union_owners: set[str] = set()
    for f in files:
        owners = resolve_owners(rules, f)
        union_owners.update(owners)
        owners_str = (
            " ".join(_with_people(o) if expand_paths else o for o in owners)
            if owners
            else "(no owner -- falls through; CI coverage gate should block this)"
        )
        print(f"{f}\n    review: {owners_str}")

    if args.changed:
        print("\n" + "=" * 60)
        print(f"Teams auto-requested on this PR ({len(union_owners)}):")
        for t in sorted(union_owners):
            print(f"  {_with_people(t) if args.people else t}")
            if args.people and (url := _team_url(t)):
                print(f"      {url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
