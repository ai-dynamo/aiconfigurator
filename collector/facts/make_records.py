#!/usr/bin/env python3
"""Curated record generation: raw probe facts -> slim consumption records.

Raw JSONs under archive/raw/ stay as full evidence; this layer keeps ONLY the
fields that analyses actually consumed (see README "record schema rationale"),
normalizes kernel names, drops infrastructure noise, and merges nested spans.

Usage: python3 probe/make_records.py   -> archive/records.jsonl
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

ROOT = Path(os.environ.get("AIC_PROBE_WORKSPACE", Path.cwd()))

# kernels that are infrastructure, never op identity
KERNEL_DENY = re.compile(
    r"Memcpy|Memset|Lazy Function Loading|Runtime Triggered Module Loading|"
    r"at::native::(vectorized_elementwise|elementwise|index_elementwise|"
    r"unrolled_elementwise|reduce_kernel|distribution_|fill)|aten::(fill_|copy_|zero_)|"
    r"^void at::native::.*FillFunctor"
)
# wrapper identifiers to skip when extracting a meaningful kernel name
NAME_WRAPPERS = {"void", "cutlass::device_kernel", "flash::enable_sm90_or_later",
                 "cute", "std", "c10", "at", "at::native", "int", "bool", "float",
                 "unsigned", "long", "char"}
FRAME_DENY = re.compile(r"_inductor/runtime|pybind11_detail|<built-in method")
FW_FRAME = re.compile(r"(sglang|vllm|tensorrt_llm|cutlass|flashinfer|deep_gemm|sgl_kernel|flash)")


def normalize_kernel(name: str) -> str | None:
    if KERNEL_DENY.search(name):
        return None
    if "<" not in name and " " not in name.strip():
        return name[:80]
    idents = re.findall(r"[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)+|[A-Za-z_][A-Za-z0-9_]{5,}", name)
    for ident in idents:
        if (ident in NAME_WRAPPERS or ident.split("::")[0] in NAME_WRAPPERS
                or "anonymous" in ident):
            continue
        return ident[:80]
    return name.split("(")[0][:80]


def clean_path(path: str) -> str:
    parts = [p.strip() for p in path.split("<-")]
    kept = [p for p in parts if not FRAME_DENY.search(p)]
    return " <- ".join(kept[:5])


def build_ops(facts: dict) -> tuple[list[dict], list[str]]:
    """Merge api_trace spans into ops; return (ops, orphan_kernels)."""
    ops: list[dict] = []
    attributed: set[str] = set()
    trace = facts.get("trace") or facts  # sglang nests under trace; vllm flat
    for phase_key, phase in (("prefill_api", "prefill"), ("decode_api", "decode"), ("api_trace", "decode")):
        spans = trace.get(phase_key) or {}
        merged: dict[frozenset, dict] = {}
        for span, s in spans.items():
            _, kind, detail = span.split("::", 2)
            kerns = set(filter(None, (normalize_kernel(k) for k in s.get("kernels", {}))))
            attributed |= kerns
            key = frozenset(kerns) or frozenset({span})
            slot = merged.setdefault(key, {"phase": phase, "op": None, "quant": None,
                                           "api": None, "kernels": sorted(kerns),
                                           "calls": s.get("calls", 0)})
            if kind == "quant_apply":
                slot["quant"] = detail
            else:
                slot["op"] = f"{kind}:{detail}"
            paths = [clean_path(p) for p in (s.get("py_paths") or {})
                     if not FRAME_DENY.search(p) and not KERNEL_DENY.search(p.split("<-")[0])]
            if paths and (slot["api"] is None or kind != "quant_apply"):
                slot["api"] = paths[0]
        ops.extend(v for v in merged.values() if v["kernels"] or v["op"])
    # trtllm probe: flat kernels list, no spans
    if not ops and facts.get("kernels"):
        kerns = sorted(set(filter(None, (normalize_kernel(k["kernel"]) for k in facts["kernels"]))))
        attributed |= set(kerns)
        ops.append({"phase": "generate", "op": "all", "quant": None, "api": None,
                    "kernels": kerns, "calls": 1})
    orphans: list[str] = []
    for tbl in ("prefill_kernels", "decode_kernels"):
        for k in (trace.get(tbl) or []):
            name = k.get("kernel", "")
            if (name.startswith(("AIC::", "step", "aten::")) or name.isupper()
                    or re.match(r"^(sglang|sgl_kernel|_\w*C\w*|triton_)\w*::", name)):
                continue  # spans, phase markers, custom-op launchers — not kernels
            n = normalize_kernel(name)
            if n and n not in attributed:
                orphans.append(n)
    return ops, sorted(set(orphans))[:15]


def compress_error(stage: str, tb: str) -> dict:
    lines = tb.strip().splitlines()
    frames = [ln.strip()[:110] for ln in lines
              if ln.strip().startswith("File") and FW_FRAME.search(ln)][-5:]
    return {"stage": stage, "exc": lines[-1][:160], "frames": frames}


def main() -> None:
    plan: dict = {}
    for pf in sorted((ROOT / "archive").glob("plan*.json")):
        plan.update({r["id"]: r for r in json.loads(pf.read_text()) if "skip" not in r})
    out = ROOT / "archive" / "records.jsonl"
    n = 0
    with out.open("w") as fh:
        for rid, run in plan.items():
            raw = ROOT / "archive" / "raw" / f"{rid}.json"
            if not raw.exists():
                continue
            f = json.loads(raw.read_text())
            ops, orphans = build_ops(f)
            sa = f.get("server_args_resolved") or f.get("engine_args_resolved") or {}
            keep = re.compile(
                r"^(kv_cache_dtype|page_size|block_size|quantization|attention_backend|"
                r"(prefill|decode)_attention_backend|dsa_(prefill|decode|topk|paged_mqa_logits)_backend|"
                r"moe_(runner|a2a)_backend|fp8_gemm_runner_backend|fp4_gemm_runner_backend|"
                r"bf16_gemm_backend|linear_attn_backend|mamba_backend|dtype|load_format|"
                r"tensor_parallel_size|max_model_len|context_length)$")
            sa = {k: v for k, v in sa.items() if keep.match(k)}
            rec = {
                "id": rid,
                "target": {k: run.get(k) for k in ("repo", "family", "variant", "profile",
                                                   "kvcache_quant_mode", "aic_registered")},
                "runtime": {**{k: run.get(k) for k in ("backend", "version", "image", "tp")},
                            "engine_cli": run.get("engine_cli"),
                            "unknown_args": f.get("engine_cli_unknown_args") or None,
                            "platform": "h20_sm90", "evidence": "real"},
                "resolved": {k: v for k, v in sa.items() if v is not None},
                "identity": {
                    "model_class": f.get("model_class"),
                    "attn_backend": (f.get("attn_backend") or "").rsplit(".", 1)[-1] or None,
                    "modules": {k.rsplit(".", 1)[-1]: v.get("modules", v.get("examples", []))
                                for k, v in (f.get("quant_methods") or {}).items()},
                    "param_dtypes": f.get("param_dtypes"),
                    "weight_samples": f.get("weight_samples") or None,
                },
                "ops": ops or None,
                "orphan_kernels": orphans or None,
                "outcome": ({"status": "ok"} if not f.get("errors") else
                            compress_error(*next(iter(f["errors"].items())))),
            }
            fh.write(json.dumps({k: v for k, v in rec.items() if v is not None}) + "\n")
            n += 1
    raw_bytes = sum(p.stat().st_size for p in (ROOT / "archive" / "raw").glob("*.json"))
    print(f"wrote {out}: {n} records, {out.stat().st_size // 1024}KB (raw evidence: {raw_bytes // 1024}KB)")


if __name__ == "__main__":
    main()
