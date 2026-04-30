"""Verify that the h100_sxm/sglang/0.5.9 perf data on the current checkout
matches byte-for-byte what lives on `origin/main`.

The perf `.txt` files under `src/aiconfigurator/systems/` are tracked with
git LFS (see `.gitattributes`). That means the blob stored in git is a small
LFS pointer, while the working-copy file is the smudged multi-MB payload.
Comparing them directly would always disagree. Instead this script compares
the git object SHAs — the working-copy file is hashed through the LFS clean
filter (`git hash-object`) so the resulting SHA can be matched against the
blob recorded in `origin/main`'s tree. If both sides agree, every byte —
including the LFS-stored payload — is identical.

Intended to be run from the repo root on the release/0.8.0 branch (or a PR
branched from it) after copying the sglang 0.5.9 perf data from main to
resolve issue #53.

Usage:
    python tools/verify_sglang_0_5_9_matches_main.py
    python tools/verify_sglang_0_5_9_matches_main.py --ref origin/main
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REL_DIR = "src/aiconfigurator/systems/data/h100_sxm/sglang/0.5.9"

EXPECTED_FILES = [
    "context_attention_perf.txt",
    "context_mla_perf.txt",
    "custom_allreduce_perf.txt",
    "dsa_context_module_perf.txt",
    "dsa_generation_module_perf.txt",
    "gdn_perf.txt",
    "gemm_perf.txt",
    "generation_attention_perf.txt",
    "generation_mla_perf.txt",
    "mla_bmm_perf.txt",
    "moe_perf.txt",
    "wideep_context_mla_perf.txt",
    "wideep_generation_mla_perf.txt",
]


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def local_blob_sha(path: str) -> str | None:
    """git hash-object respects LFS clean filter, so this returns the SHA of
    the pointer blob that would be committed — comparable to tree blob SHAs."""
    res = run(["git", "hash-object", "--", path])
    if res.returncode != 0:
        return None
    return res.stdout.strip() or None


def ref_blob_sha(ref: str, path: str) -> str | None:
    """Blob SHA recorded in the tree at `ref` for `path`."""
    res = run(["git", "ls-tree", ref, "--", path])
    if res.returncode != 0 or not res.stdout.strip():
        return None
    # format: <mode> <type> <sha>\t<path>
    return res.stdout.split()[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", default="origin/main", help="Git ref to compare against.")
    args = parser.parse_args()

    run(["git", "fetch", "origin", "main"])

    mismatches: list[tuple[str, str | None, str | None]] = []
    missing_local: list[str] = []
    missing_ref: list[str] = []

    for name in EXPECTED_FILES:
        posix = f"{REL_DIR}/{name}"

        local_exists = Path(posix).exists()
        if not local_exists:
            missing_local.append(posix)

        lsha = local_blob_sha(posix) if local_exists else None
        rsha = ref_blob_sha(args.ref, posix)

        if rsha is None:
            missing_ref.append(posix)

        if lsha is None or rsha is None:
            status = "MISS "
        elif lsha == rsha:
            status = "OK   "
        else:
            status = "DIFF "
            mismatches.append((posix, lsha, rsha))

        print(f"{status} {posix}  local={(lsha or '-')[:12]}  ref={(rsha or '-')[:12]}")

    print()
    print(f"checked against ref : {args.ref}")
    print(f"total files         : {len(EXPECTED_FILES)}")
    print(f"missing locally     : {len(missing_local)}")
    print(f"missing on ref      : {len(missing_ref)}")
    print(f"blob-sha mismatches : {len(mismatches)}")

    for label, items in (("MISSING LOCALLY", missing_local), (f"MISSING ON {args.ref}", missing_ref)):
        if items:
            print(f"{label}:")
            for p in items:
                print(f"  - {p}")
    if mismatches:
        print("MISMATCHES (local blob sha vs ref blob sha):")
        for p, l, r in mismatches:
            print(f"  - {p}  local={l}  ref={r}")

    if mismatches or missing_local or missing_ref:
        print(f"\nFAIL: sglang/0.5.9 data does not match {args.ref}.")
        return 1

    print(f"\nPASS: sglang/0.5.9 data matches {args.ref} exactly (all blob SHAs equal).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
