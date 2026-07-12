#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""min_approvers.py -- distribution of distinct approvers needed per PR.

Replays a corpus of merged PRs against a generated CODEOWNERS and computes,
for each PR, the MINIMUM number of distinct people whose approvals satisfy
every matched rule (GitHub "require Code Owners review" + any-one-approves).
This is the tool that catches stale claims like "never more than two
approvers": rosters drift, and the true tail only shows up when rule owner
sets stop overlapping.

Inputs:
  --codeowners  generated CODEOWNERS file
  --corpus      jsonl, one PR per line: {"n": <pr number>, "files": [paths]}
  --org         expand @org/team owners via `gh api` (needs org membership), OR
  --rosters     tsv of "<team-slug>\\t<login>" rows (offline / pinned rosters)

Regenerating the corpus (the file is runtime data, not committed):
  gh pr list --repo <owner>/<repo> --state merged --limit 1000 \\
      --json number --jq '.[].number' | while read n; do
    gh api "repos/<owner>/<repo>/pulls/$n/files" --paginate \\
        --jq '{n: '"$n"', files: [.[].filename]}'
  done > last1000_prs.jsonl

Exact min hitting set over each PR's owner-member sets; constraint sets are
small (rarely more than ~10 distinct rules per PR), so brute force is fine.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from codeowners_match import parse_codeowners, resolve_owners


def load_rosters_tsv(path: Path) -> dict[str, set[str]]:
    members: dict[str, set[str]] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != 2 or any(not field.strip() for field in fields):
            raise SystemExit(f"{path}:{line_number}: expected exactly '<team-slug>\\t<login>'")
        slug, login = (field.strip() for field in fields)
        members.setdefault(slug, set()).add(login)
    return members


def fetch_rosters(org: str, teams: set[str]) -> dict[str, set[str]]:
    members: dict[str, set[str]] = {}
    for team in sorted(teams):
        if not team.startswith(f"@{org}/"):
            continue
        slug = team.split("/", 1)[1]
        try:
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
        except (OSError, subprocess.CalledProcessError):
            continue
        members[slug] = {ln for ln in out.splitlines() if ln.strip()}
    return members


def required_team_slugs(owners: set[str], org: str | None = None) -> set[str]:
    """Return every referenced team slug, rejecting cross-org online lookups."""
    slugs: set[str] = set()
    for owner in owners:
        if "/" not in owner:
            continue
        if not owner.startswith("@"):
            raise SystemExit(f"invalid CODEOWNERS team token: {owner!r}")
        owner_org, slug = owner[1:].split("/", 1)
        if not owner_org or not slug:
            raise SystemExit(f"invalid CODEOWNERS team token: {owner!r}")
        if org is not None and owner_org != org:
            raise SystemExit(f"cannot expand {owner!r} with --org {org!r}; use a complete offline roster")
        slugs.add(slug)
    return slugs


def validate_rosters(required: set[str], rosters: dict[str, set[str]]) -> None:
    """Fail unless every referenced team has at least one known member."""
    incomplete = sorted(slug for slug in required if not rosters.get(slug))
    if incomplete:
        raise SystemExit(
            "incomplete team rosters for: " + ", ".join(incomplete) + "; refusing to compute misleading approver counts"
        )


def load_corpus(path: Path) -> list[dict]:
    """Load a JSONL PR corpus, ignoring formatting-only blank lines."""
    return [json.loads(raw) for raw in path.read_text().splitlines() if raw.strip()]


def min_hitting(sets: list[frozenset]) -> int:
    """Exact minimum hitting set size (small inputs only)."""
    sets = [s for s in sets if s]
    uniq = sorted(set(sets), key=len)
    kept = [s for i, s in enumerate(uniq) if not any(t < s for t in uniq[:i])]
    if not kept:
        return 0
    universe = sorted(set().union(*kept))
    for k in range(1, len(kept) + 1):
        for combo in itertools.combinations(universe, k):
            cs = set(combo)
            if all(cs & s for s in kept):
                return k
    return len(kept)


def main() -> int:
    ap = argparse.ArgumentParser(description="Min distinct approvers per PR, replayed against CODEOWNERS.")
    ap.add_argument("--codeowners", required=True, type=Path)
    ap.add_argument(
        "--corpus",
        required=True,
        type=Path,
        help='jsonl of {"n": <pr>, "files": [...]} (see module docstring)',
    )
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--org", help="expand @org/team via gh api (org members only)")
    group.add_argument("--rosters", type=Path, help="tsv of '<team-slug>\\t<login>' rows")
    ap.add_argument("--worst", type=int, default=10, help="how many worst PRs to list")
    args = ap.parse_args()

    rules = parse_codeowners(args.codeowners.read_text())
    all_teams = {o for _, owners in rules for o in owners}
    required_slugs = required_team_slugs(all_teams, args.org)
    if args.rosters:
        rosters = load_rosters_tsv(args.rosters)
    else:
        rosters = fetch_rosters(args.org, all_teams)
    validate_rosters(required_slugs, rosters)

    def people(owner: str) -> frozenset:
        if "/" not in owner:
            return frozenset({owner})
        slug = owner.split("/", 1)[1]
        return frozenset(rosters[slug])

    dist: Counter = Counter()
    worst: list[tuple[int, int, int]] = []
    for pr in load_corpus(args.corpus):
        owner_sets = {
            frozenset(itertools.chain.from_iterable(people(o) for o in owners))
            for f in pr["files"]
            if (owners := resolve_owners(rules, f))
        }
        k = min_hitting(list(owner_sets))
        dist[k] += 1
        worst.append((k, pr["n"], len(pr["files"])))

    total = sum(dist.values())
    print(f"PRs replayed: {total}")
    for k in sorted(dist):
        print(f"  min approvers = {k}: {dist[k]} ({100 * dist[k] / total:.1f}%)")
    worst.sort(reverse=True)
    print(f"worst {args.worst}:")
    for k, n, nfiles in worst[: args.worst]:
        print(f"  #{n}: {k} approvers ({nfiles} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
