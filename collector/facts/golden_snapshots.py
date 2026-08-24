#!/usr/bin/env python3
"""Facts freshness tripwire: pin the rendered generator surface each archived
fact was measured under.

Scope is deliberately the PROBED SLICE ONLY (the roster plans: model x
backend x tp1, agg, one system, pinned versions). This is NOT a generator
regression suite — the generator's input space (tp/pp/ep, disagg, systems,
template versions) is far larger than this slice, and generator-wide
correctness belongs to upstream's own template tests. What this file pins is
provenance: "every pass/fail in the facts archive was measured under exactly
this rendered command". If a re-render diverges, those cells are STALE and
need re-probing — that is the only claim checked here.

Snapshots are one merged YAML per backend (small, review-diffable text — no
LFS: pointer files would defeat the reviewable diff, and canonicalization is
the answer to size, not storage).

Usage:
  AIC_PROBE_WORKSPACE=<ws> python3 golden_snapshots.py --update   # rebuild from goldens
  AIC_PROBE_WORKSPACE=<ws> python3 golden_snapshots.py --check    # re-render + diff
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("AIC_PROBE_WORKSPACE", HERE.parent))
SNAP_DIR_DEFAULT = HERE / "golden_snapshots"

sys.path.insert(0, str(HERE))
import gen_facts  # noqa: E402  (render_golden, extract_sglang_cli_from_run_sh)

PLANS = {
    "sglang": "plan_roster_sgl.json",
    "vllm": "plan_roster_vllm.json",
    "trtllm": "plan_roster_trt.json",
}


def canonicalize(text: str) -> str:
    """Strip volatile workspace-specific noise from a rendered surface."""
    text = text.replace(str(ROOT), "{WS}")
    # generator save-dir components embed the run id via golden/<id>/
    out = []
    for line in text.splitlines():
        if "{WS}/archive/golden/" in line:
            import re
            line = re.sub(r"\{WS\}/archive/golden/[0-9a-f]+", "{WS}/archive/golden/{ID}", line)
        out.append(line.rstrip())
    return "\n".join(out).strip() + "\n"


def surface(run: dict, art: Path) -> str | None:
    """The exact rendered text each probe consumes for this backend."""
    be = run["backend"]
    if be == "sglang":
        p = art / "run_0.sh"
        return gen_facts.extract_sglang_cli_from_run_sh(p) + "\n" if p.exists() else None
    if be == "vllm":
        p = next((q for q in (art / "run.sh", art / "run_0.sh") if q.exists()), None)
        if p is None:
            return None
        # run.sh = fpm template boilerplate + ONE model-specific engine_command
        # line (the argv the probe consumes). Pin the line verbatim and the
        # rest as a hash: template drift flips every cell's hash at once.
        import hashlib
        lines = canonicalize(p.read_text()).splitlines()
        cmd = [ln for ln in lines if ln.lstrip().startswith("engine_command=(")]
        rest = "\n".join(ln for ln in lines if ln not in cmd)
        h = hashlib.sha256(rest.encode()).hexdigest()[:16]
        return "\n".join(cmd) + f"\n# template_sha={h}\n"
    p = art / "agg_config.yaml"
    return canonicalize(p.read_text()) if p.exists() else None


def iter_runs(backend: str):
    for run in json.loads((ROOT / "archive" / PLANS[backend]).read_text()):
        if isinstance(run, dict) and "skip" not in run and run.get("id"):
            yield run


def build(backend: str, render: bool) -> tuple[dict, list[str]]:
    entries, errors = {}, []
    for run in iter_runs(backend):
        key = f"{run['repo']}|tp{run['tp']}"
        art = None
        if render:
            art = gen_facts.render_golden(dict(run))  # copy: render mutates on failure
        else:
            # the artifact subdir name carries a generator-chosen suffix, so a
            # re-render invalidates the path stored in the plan — resolve from
            # the stable golden/<id>/ parent instead of trusting the string
            g = run.get("golden_dir")
            art = Path(g) if g and Path(g).exists() else None
            if art is None:
                gdir = ROOT / "archive" / "golden" / run["id"]
                if gdir.exists():
                    art = next((d for d in gdir.iterdir() if d.is_dir()), None)
        if art is None:
            errors.append(f"{backend} {key}: golden render failed / missing")
            continue
        s = surface(run, art)
        if s is None:
            errors.append(f"{backend} {key}: expected artifact missing in {art}")
            continue
        entries[key] = s
    return entries, errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true", help="rebuild snapshots from current goldens")
    ap.add_argument("--check", action="store_true", help="re-render and diff against snapshots")
    ap.add_argument("--snapshot-dir", type=Path, default=SNAP_DIR_DEFAULT)
    args = ap.parse_args()
    if args.update == args.check:
        ap.error("pass exactly one of --update / --check")

    args.snapshot_dir.mkdir(parents=True, exist_ok=True)
    import subprocess
    gen_commit = subprocess.run(["git", "-C", str(ROOT / "aic"), "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True).stdout.strip()
    rc = 0
    for backend in PLANS:
        snap_path = args.snapshot_dir / f"{backend}.yaml"
        entries, errors = build(backend, render=args.check)
        for e in errors:
            print(f"[render-error] {e}")
        if args.update:
            doc = {"_meta": {"generator_commit": gen_commit,
                             "scope": "probed slice only (facts provenance, not a generator test)"},
                   "surfaces": dict(sorted(entries.items()))}
            snap_path.write_text(yaml.safe_dump(doc, width=4096, sort_keys=False,
                                                default_style="|", allow_unicode=True))
            print(f"wrote {snap_path} ({len(entries)} surfaces, {snap_path.stat().st_size//1024} KB)")
            continue
        if not snap_path.exists():
            print(f"[missing-snapshot] {snap_path}")
            rc = 1
            continue
        snap = yaml.safe_load(snap_path.read_text())["surfaces"]
        stale = sorted(set(snap) ^ set(entries)) + \
            [k for k in sorted(set(snap) & set(entries)) if snap[k] != entries[k]]
        for k in stale:
            print(f"[STALE] {backend} {k} — facts for this cell were measured under a different render")
            for line in difflib.unified_diff((snap.get(k) or "").splitlines(),
                                             (entries.get(k) or "").splitlines(),
                                             "snapshot", "re-render", lineterm="", n=1):
                print(f"    {line}")
        print(f"{backend}: {len(entries)} rendered, {len(stale)} stale")
        rc = rc or (1 if stale else 0)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
