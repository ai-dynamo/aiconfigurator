#!/usr/bin/env python3
"""Conformance checker v1: reconcile probe records against the hand-written
collector/op_backend_facts.yaml.

Per facts row verdict:
  confirmed       — a probe record matches the identity AND its kernel/api
                    evidence clearly supports the row's kernel_sources
  contradicted    — identity matches but the probed evidence names a
                    DIFFERENT kernel family
  needs-taxonomy  — identity matches; kernel evidence exists but cannot be
                    mapped confidently (blocked on the kernel-name taxonomy)
  unprobed        — no probe record covers this (framework, version,
                    architecture, kv dtype) identity yet

This is deliberately conservative: 'confirmed' requires a substring-level
match between the row's kernel_sources and the record's normalized kernels or
Python api chains. The taxonomy table replaces that heuristic later.

Usage:
  AIC_PROBE_WORKSPACE=<ws> python3 collector/facts/check_facts.py \
      [--facts collector/op_backend_facts.yaml] [--ignore-version]
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("AIC_PROBE_WORKSPACE", Path.cwd()))

# probe platform -> facts `system` values with the same SM (identity-level
# equivalence: backend selection keys on SM, not on the exact SKU)
SM_SYSTEMS = {"h20_sm90": {"h100_sxm", "h200_sxm"}}


def load_records() -> list[dict]:
    path = ROOT / "archive" / "records.jsonl"
    return [json.loads(line) for line in path.open()]


def kernel_evidence(rec: dict) -> str:
    """One lowercase haystack of everything the probe saw execute."""
    parts = []
    for op in rec.get("ops") or []:
        parts += op.get("kernels") or []
        if op.get("api"):
            parts.append(op["api"])
        if op.get("op"):
            parts.append(op["op"])
    parts += rec.get("orphan_kernels") or []
    ab = (rec.get("identity") or {}).get("attn_backend")
    if ab:
        parts.append(ab)
    return " ".join(parts).lower()


# kernel_source label -> substrings that count as clear probe evidence.
# Seed set only; superseded by the taxonomy table (see facts task list).
LABEL_EVIDENCE = {
    "fa3": ("flashattnfwdsm90", "flash_attn_varlen", "fa3"),
    "compressed_flashmla": ("flash_fwd_splitkv_mla", "flash_mla_with_kvcache", "flashmla"),
    "flash_mla_sparse_fwd": ("flash_mla_sparse", "flashmla_sparse"),
    "fused_moe_triton": ("fused_moe_kernel", "fused_experts_none_to_triton"),
    "marlin": ("marlin",),
    "deep_gemm": ("deep_gemm",),
    "trtllm_internal": ("fmha_v2", "_matmul_ogs_", "trtllm"),
}


def label_verdict(label: str, evidence: str) -> str:
    probes = LABEL_EVIDENCE.get(label)
    if probes is None:
        # fall back to matching the label text itself
        probes = (label.lower().replace("_", ""),)
        evidence_flat = evidence.replace("_", "")
        return "confirmed" if any(p in evidence_flat for p in probes) else "needs-taxonomy"
    return "confirmed" if any(p in evidence for p in probes) else "needs-taxonomy"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", type=Path, default=REPO / "collector" / "op_backend_facts.yaml")
    ap.add_argument("--ignore-version", action="store_true",
                    help="match rows across framework versions (probe vs facts version skew)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    facts = yaml.safe_load(args.facts.read_text())
    records = load_records()

    # index records by (framework, version, architecture-ish, kv dtype)
    idx: dict[tuple, list[dict]] = {}
    for rec in records:
        if rec.get("outcome", {}).get("status") not in (None, "ok"):
            continue
        arch = ((rec.get("identity") or {}).get("model_class") or "").lower()
        fam = rec["target"]["family"]
        kv = (rec.get("resolved") or {}).get("kv_cache_dtype") or rec["target"].get("kvcache_quant_mode")
        key = (rec["runtime"]["backend"], rec["runtime"]["version"], fam, arch, str(kv).lower())
        idx.setdefault(key, []).append(rec)

    def find_records(row: dict) -> list[dict]:
        out = []
        arch = (row.get("architecture") or "").lower()
        kv = str(row.get("kv_cache_dtype") or "").lower()
        for (backend, version, _fam, rec_arch, rec_kv), recs in idx.items():
            if backend != row.get("framework"):
                continue
            if not args.ignore_version and version != str(row.get("version")):
                continue
            if arch and rec_arch and arch != rec_arch:
                continue
            # kv dtype: treat auto/bfloat16 as one identity
            norm = {"auto": "bfloat16", "fp8": "fp8_e4m3"}
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
            recs = find_records(row)
            if not recs:
                verdicts["unprobed"] += 1
                continue
            evidence = " ".join(kernel_evidence(r) for r in recs)
            row_verdicts = {label_verdict(ks, evidence) for ks in row.get("kernel_sources", [])}
            verdict = ("confirmed" if row_verdicts == {"confirmed"}
                       else "needs-taxonomy" if "needs-taxonomy" in row_verdicts
                       else "contradicted")
            verdicts[verdict] += 1
            if verdict != "confirmed" or args.verbose:
                findings.append(
                    f"[{verdict}] {op['op_file']} :: {row.get('framework')}=={row.get('version')} "
                    f"{row.get('system')} {row.get('architecture')} kv={row.get('kv_cache_dtype')} "
                    f"kernel_sources={row.get('kernel_sources')}"
                )

    print(f"facts rows on probe-adjudicable systems: {sum(verdicts.values())}")
    for k, v in verdicts.most_common():
        print(f"  {k:14} {v}")
    for line in findings[:40]:
        print(line)


if __name__ == "__main__":
    main()
