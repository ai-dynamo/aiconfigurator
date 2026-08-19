#!/usr/bin/env python3
"""Conformance checker: reconcile probe records against the hand-written
collector/op_backend_facts.yaml through ONE shared backend vocabulary.

Translation chain (no per-label heuristics):

  facts row kernel_sources --kernel_source_backends.yaml--> canonical backends
  probe record kernels     --kernel_taxonomy.yaml--------->  canonical backends

Per facts row verdict:
  confirmed       — a probe record matches the identity AND every canonical
                    backend the row claims was observed executing
  contradicted    — identity matches, the probe classified its kernels, and
                    the claimed backend did NOT run
  needs-taxonomy  — identity matches but the evidence is not classifiable:
                    the record still has unclassified kernels, or the claimed
                    backend has no taxonomy rule yet (backlog signal for
                    kernel_taxonomy.yaml)
  unprobed        — no probe record covers this (framework, version,
                    architecture, kv dtype) identity yet

Usage:
  AIC_PROBE_WORKSPACE=<ws> python3 collector/facts/check_facts.py \
      [--facts collector/op_backend_facts.yaml] [--ignore-version]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ROOT = Path(os.environ.get("AIC_PROBE_WORKSPACE", Path.cwd()))

sys.path.insert(0, str(HERE))
from make_records import label_kernels, load_taxonomy  # noqa: E402

# probe platform -> facts `system` values with the same SM (identity-level
# equivalence: backend selection keys on SM, not on the exact SKU)
SM_SYSTEMS = {"h20_sm90": {"h100_sxm", "h200_sxm"}}

# comm/collective backends: single-GPU probes cannot adjudicate them
COMM_BACKENDS = {"nccl", "custom_allreduce", "nvlink_one_sided", "nvlink_two_sided", "deepep"}

# canonical-name equivalences: the collector vocabulary distinguishes the
# DISPATCHER (vllm's flash_attn wrapper) from the KERNEL GENERATION the
# taxonomy observes (fa2/fa3/fa4 cubins are the same wheel's kernels)
BACKEND_EQUIV = {"flash_attn": {"fa2", "fa3", "fa4"}}


def load_records() -> list[dict]:
    path = ROOT / "archive" / "records.jsonl"
    return [json.loads(line) for line in path.open()]


def load_kernel_source_map() -> dict[tuple[str | None, str], str]:
    """(framework, kernel_source label) -> canonical backend."""
    data = yaml.safe_load((REPO / "collector" / "kernel_source_backends.yaml").read_text())
    return {(m.get("framework"), m["kernel_source"]): m["backend"] for m in data["mappings"]}


def taxonomy_backends() -> set[str]:
    """Canonical backends the taxonomy can currently produce as evidence."""
    return {b for _, b, _ in load_taxonomy()}


def record_backends(rec: dict) -> tuple[set[str], bool]:
    """(observed canonical backends, fully-classified?) for one record."""
    observed: set[str] = set()
    unclassified = False
    for op in rec.get("ops") or []:
        observed |= set(op.get("backends") or [])
        if op.get("unclassified_kernels"):
            unclassified = True
    orphans = rec.get("orphan_kernels") or []
    if orphans:
        labels, unmatched = label_kernels(orphans)
        observed |= labels
        unclassified = unclassified or bool(unmatched)
    return observed, unclassified


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", type=Path, default=REPO / "collector" / "op_backend_facts.yaml")
    ap.add_argument("--ignore-version", action="store_true",
                    help="match rows across framework versions (probe vs facts version skew)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    facts = yaml.safe_load(args.facts.read_text())
    records = load_records()
    ks_map = load_kernel_source_map()
    producible = taxonomy_backends()

    # index records by (framework, version, architecture-ish, kv dtype)
    idx: dict[tuple, list[dict]] = {}
    for rec in records:
        if rec.get("outcome", {}).get("status") not in (None, "ok"):
            continue
        arch = ((rec.get("identity") or {}).get("model_class") or "").lower()
        kv = (rec.get("resolved") or {}).get("kv_cache_dtype") or rec["target"].get("kvcache_quant_mode")
        key = (rec["runtime"]["backend"], rec["runtime"]["version"], arch, str(kv).lower())
        idx.setdefault(key, []).append(rec)

    def find_records(row: dict) -> list[dict]:
        out = []
        arch = (row.get("architecture") or "").lower()
        kv = str(row.get("kv_cache_dtype") or "").lower()
        norm = {"auto": "bfloat16", "fp8": "fp8_e4m3"}
        for (backend, version, rec_arch, rec_kv), recs in idx.items():
            if backend != row.get("framework"):
                continue
            if not args.ignore_version and version != str(row.get("version")):
                continue
            if arch and rec_arch and arch != rec_arch:
                continue
            if norm.get(kv, kv) != norm.get(rec_kv, rec_kv):
                continue
            out.extend(recs)
        return out

    verdicts: Counter = Counter()
    findings: list[str] = []
    for op in facts.get("ops", []):
        for row in op.get("facts", []):
            systems = SM_SYSTEMS.get("h20_sm90", set())
            if row.get("system") not in systems:
                continue  # probe platform can only adjudicate same-SM rows
            fw = row.get("framework")
            expected = {ks_map.get((fw, ks), ks_map.get((None, ks)))
                        for ks in row.get("kernel_sources", [])}
            expected -= COMM_BACKENDS | {None, "unverified", "torch", "sgl_kernel", "vllm_kernel"}
            if not expected:
                continue  # nothing kernel-identifying left to check
            recs = find_records(row)
            if not recs:
                verdicts["unprobed"] += 1
                continue
            observed: set[str] = set()
            any_unclassified = False
            for r in recs:
                obs, uncls = record_backends(r)
                observed |= obs
                any_unclassified = any_unclassified or uncls
            missing = {e for e in expected - observed
                       if not (BACKEND_EQUIV.get(e, set()) & observed)}
            if not missing:
                verdict = "confirmed"
            elif missing - producible or any_unclassified:
                # claimed backend has no taxonomy rule yet, or evidence
                # incomplete — cannot adjudicate
                verdict = "needs-taxonomy"
            else:
                verdict = "contradicted"
            verdicts[verdict] += 1
            if verdict != "confirmed" or args.verbose:
                findings.append(
                    f"[{verdict}] {op['op_file']} :: {fw}=={row.get('version')} "
                    f"{row.get('system')} {row.get('architecture')} kv={row.get('kv_cache_dtype')} "
                    f"expected={sorted(expected)} missing={sorted(missing)} "
                    f"observed={sorted(observed)}"
                )

    print(f"facts rows on probe-adjudicable systems: {sum(verdicts.values())}")
    for k, v in verdicts.most_common():
        print(f"  {k:14} {v}")
    for line in findings[:60]:
        print(line)


if __name__ == "__main__":
    main()
