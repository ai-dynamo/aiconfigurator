#!/usr/bin/env python3
"""Generate depth-shrunk dummy-model configs for backend-identity probing.

Rules (agreed 2026-08-16):
  * Cut depth only — every width dimension (hidden, heads, kv-heads, expert
    count, moe_intermediate) keeps its real value so TP/EP divisibility and
    quant shape checks behave exactly like the full checkpoint.
  * Interleaved models get one variant per layer kind (DSV4 csa/hca/full,
    GLM full/shared indexer, M3 dense/moe) so each kind is probed alone.
  * MoE models drop their leading dense layers except in an explicit
    head variant.
  * quantization_config is preserved verbatim except for per-layer entries
    (quantized_layers.layers.N, model.layers.N.* ignore/not-convert lists),
    which are filtered to the selected layers and renumbered.
  * MTP / next-N heads are zeroed for lean dummy loading.

Every variant records its provenance (source repo, original layer indices,
edits applied, caveats) in variants_manifest.yaml. A post-check scans the
final config for any surviving reference to a layer index outside the new
range and fails loudly — silent misalignment is the one failure mode this
tool must not have.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from pathlib import Path

# Collector ground truth (collect_dsv4_attn.py:313-316): csa=4, hca=128.
DSV4_RATIO_KIND = {4: "csa", 128: "hca", 0: "full"}

# Special-family adapters keyed by exact repo; EVERYTHING else uses the
# generic adapter. The roster itself lives in repos.txt (one repo per line,
# optional "<repo> <family>") so growing coverage never edits code.
_SPECIAL = {
    "deepseek-ai/DeepSeek-V4-Pro-0813": "dsv4",
    "deepseek-ai/DeepSeek-V4-Flash-0731": "dsv4",
    "nvidia/DeepSeek-V4-Pro-NVFP4": "dsv4",
    "nvidia/DeepSeek-V4-Flash-NVFP4": "dsv4",
    "zai-org/GLM-5.2": "glm",
    "zai-org/GLM-5.2-FP8": "glm",
    "nvidia/GLM-5.2-NVFP4": "glm",
    "MiniMaxAI/MiniMax-M3": "m3",
    "MiniMaxAI/MiniMax-M3-MXFP8": "m3",
    "nvidia/MiniMax-M3-NVFP4": "m3",
    "openai/gpt-oss-120b": "gptoss",
    "openai/gpt-oss-20b": "gptoss",
}


def load_repos(configs_dir: Path) -> dict[str, str]:
    roster = configs_dir / "repos.txt"
    repos: dict[str, str] = {}
    if roster.exists():
        for line in roster.read_text().splitlines():
            line = line.split("#")[0].strip()
            if not line:
                continue
            parts = line.split()
            repos[parts[0]] = parts[1] if len(parts) > 1 else _SPECIAL.get(parts[0], "generic")
    else:  # fall back to every fetched config
        for f in configs_dir.glob("*.json"):
            if f.name.endswith("_hfquant.json"):
                continue
            repo = f.stem.replace("_", "/", 1)
            repos[repo] = _SPECIAL.get(repo, "generic")
    return repos

_LAYER_REF_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|\*|$)")


def _slice_layer_lists(cfg: dict, n_layers: int, sel: list[int], edits: list[str], prefix: str = "") -> None:
    """Slice every list field whose length equals n_layers down to sel."""
    for key, val in cfg.items():
        if isinstance(val, list) and len(val) == n_layers:
            cfg[key] = [val[i] for i in sel]
            edits.append(f"sliced {prefix}{key}[{n_layers}] -> {len(sel)}")
        elif isinstance(val, dict):
            _slice_layer_lists(val, n_layers, sel, edits, prefix=f"{prefix}{key}.")


def _remap_quant_layer_entries(cfg: dict, sel: list[int], edits: list[str]) -> None:
    """Filter/renumber per-layer quantization entries to the selected layers.

    Layer references appear in several shapes across quantizers:
      * quantized_layers: {"layers.N.ffn.experts": {...}}          (modelopt dsv4)
      * ignore / modules_to_not_convert / ignored_layers / exclude_modules
      * config_groups.<g>.targets: ["backbone.layers.N.mixer...."] (nemotron)
    """
    qc = cfg.get("quantization_config")
    if not isinstance(qc, dict):
        return
    new_index = {orig: new for new, orig in enumerate(sel)}

    ql = qc.get("quantized_layers")
    if isinstance(ql, dict):  # flat keys like "layers.5.ffn.experts"
        kept = {}
        for k, v in ql.items():
            m = re.search(r"^(.*?\blayers\.)(\d+)(.*)$", k)
            if m is None:
                kept[k] = v
            elif int(m.group(2)) in new_index:
                kept[f"{m.group(1)}{new_index[int(m.group(2))]}{m.group(3)}"] = v
        edits.append(f"quantized_layers: {len(ql)} -> {len(kept)} entries, renumbered")
        qc["quantized_layers"] = kept

    containers = [(qc, f) for f in
                  ("ignore", "modules_to_not_convert", "ignored_layers", "exclude_modules")]
    for grp in (qc.get("config_groups") or {}).values():
        if isinstance(grp, dict):
            containers.append((grp, "targets"))

    for container, field in containers:
        entries = container.get(field)
        if not isinstance(entries, list):
            continue
        kept, dropped = [], 0
        for e in entries:
            m = re.search(r"^(.*?\blayers\.)(\d+)(.*)$", e) if isinstance(e, str) else None
            if m is None:
                kept.append(e)  # no layer index (lm_head, wildcards, module classes)
            elif int(m.group(2)) in new_index:
                kept.append(f"{m.group(1)}{new_index[int(m.group(2))]}{m.group(3)}")
            else:
                dropped += 1
        if dropped or kept != entries:
            edits.append(f"{field}: renumbered, dropped {dropped} out-of-range entries")
        container[field] = kept


def _check_no_stale_layer_refs(cfg: dict, max_layer: int) -> list[str]:
    """Scan the final config for layer-index references outside [0, max_layer)."""
    stale = []

    def walk(obj, path):
        if isinstance(obj, dict):
            for k, v in obj.items():
                for m in _LAYER_REF_RE.finditer(str(k)):
                    if int(m.group(1)) >= max_layer:
                        stale.append(f"{path}.{k}")
                walk(v, f"{path}.{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                walk(v, f"{path}[{i}]")
        elif isinstance(obj, str):
            for m in _LAYER_REF_RE.finditer(obj):
                if int(m.group(1)) >= max_layer:
                    stale.append(f"{path} = {obj}")

    walk(cfg, "$")
    return stale


# ---------------------------------------------------------------- adapters

def variants_dsv4(cfg: dict) -> list[dict]:
    n = cfg["num_hidden_layers"]
    n_hash = cfg.get("num_hash_layers", 0)
    ratios = cfg["compress_ratios"]
    # base checkpoints: len == n + n_hash; nvidia NVFP4 requants ship n + 1
    # with num_hash_layers still 3 — upstream inconsistency, tolerate it.
    assert len(ratios) >= n, f"compress_ratios len {len(ratios)} < num_hidden_layers {n}"
    tail = ratios[n:]
    main = ratios[:n]
    dspark = set(cfg.get("dspark_target_layer_ids", []))

    def pick(kind: str, count: int) -> list[int]:
        return [i for i, r in enumerate(main) if DSV4_RATIO_KIND[r] == kind and i not in dspark][:count]

    out = []
    for kind in ("csa", "hca", "full"):
        sel = pick(kind, 2)
        if sel:
            out.append({"name": kind, "sel": sel, "hash": False, "dspark": False})
    # one csa + one hca adjacent pair: the pool configurator sees both kinds
    csa1, hca1 = pick("csa", 1), pick("hca", 1)
    if csa1 and hca1:
        out.append({"name": "interleave_pair", "sel": sorted(csa1 + hca1), "hash": False, "dspark": False})
    # one layer of EVERY kv-spec kind: vllm's DSV4 kv grouping asserts the
    # full-MLA group exists and bounds SWA page sizes — variants missing a
    # kind violate that structural invariant (same lesson as gpt-oss SWA)
    full1 = pick("full", 1)
    if csa1 and hca1 and full1:
        out.append({"name": "rep_mix", "sel": sorted(full1 + csa1 + hca1), "hash": False, "dspark": False})
    if dspark:
        out.append({"name": "dspark", "sel": sorted(dspark), "hash": False, "dspark": True})
    if n_hash and len(tail) == n_hash:
        out.append({"name": "hash", "sel": pick("csa", 1) or [0], "hash": True, "dspark": False})
    return out


def apply_dsv4(cfg: dict, var: dict, edits: list[str]) -> None:
    n = cfg["num_hidden_layers"]
    n_hash = cfg.get("num_hash_layers", 0)
    sel = var["sel"]
    ratios = cfg["compress_ratios"]
    # compress_ratios is main+hash; handle it explicitly, then generic-slice the rest
    tail = ratios[n:] if (var["hash"] and n_hash) else []
    del cfg["compress_ratios"]  # main+hash length; sliced explicitly below
    _slice_layer_lists(cfg, n, sel, edits)
    cfg["compress_ratios"] = [ratios[i] for i in sel] + tail
    edits.append(f"compress_ratios -> {cfg['compress_ratios']}")
    cfg["num_hidden_layers"] = len(sel)
    if not var["hash"]:
        if n_hash:
            cfg["num_hash_layers"] = 0
            edits.append("num_hash_layers -> 0")
    if var["dspark"]:
        remap = {o: i for i, o in enumerate(sel)}
        cfg["dspark_target_layer_ids"] = [remap[i] for i in cfg["dspark_target_layer_ids"]]
        edits.append(f"dspark_target_layer_ids -> {cfg['dspark_target_layer_ids']}")
    else:
        dropped = [k for k in list(cfg) if k.startswith("dspark_")]
        for k in dropped:
            del cfg[k]
        if dropped:
            edits.append(f"dropped {dropped} (precedent: nvidia NVFP4 configs ship without dspark_*)")


def variants_glm(cfg: dict) -> list[dict]:
    idx_types = cfg["indexer_types"]
    mlp_types = cfg["mlp_layer_types"]
    out = []
    for indexer in ("full", "shared"):
        sel = [i for i, (a, b) in enumerate(zip(idx_types, mlp_types))
               if a == indexer and b == "sparse"][:2]
        if sel:
            out.append({"name": f"{indexer}_indexer_moe", "sel": sel})
    return out


def apply_glm(cfg: dict, var: dict, edits: list[str]) -> None:
    n = cfg["num_hidden_layers"]
    _slice_layer_lists(cfg, n, var["sel"], edits)
    cfg["num_hidden_layers"] = len(var["sel"])
    if cfg.get("first_k_dense_replace"):
        cfg["first_k_dense_replace"] = 0
        edits.append("first_k_dense_replace -> 0 (dense head dropped)")


def variants_m3(cfg: dict) -> list[dict]:
    tc = cfg["text_config"]
    moe = tc["moe_layer_freq"]
    out = []
    sel = [i for i, f in enumerate(moe) if f == 1][:2]
    if sel:
        out.append({"name": "moe_sparse_attn", "sel": sel})
    head = [i for i, f in enumerate(moe) if f == 0][:2]
    if head:
        out.append({"name": "dense_full_attn_head", "sel": head})
    return out


def apply_m3(cfg: dict, var: dict, edits: list[str]) -> None:
    tc = cfg["text_config"]
    n = tc["num_hidden_layers"]
    _slice_layer_lists(tc, n, var["sel"], edits, prefix="text_config.")
    tc["num_hidden_layers"] = len(var["sel"])
    for k in ("num_mtp_modules", "num_nextn_predict_layers"):
        if tc.get(k):
            tc[k] = 0
            edits.append(f"text_config.{k} -> 0")


def variants_gptoss(cfg: dict) -> list[dict]:
    lt = cfg["layer_types"]  # alternating sliding_attention / full_attention
    out = []
    for kind in ("sliding_attention", "full_attention"):
        sel = [i for i, t in enumerate(lt) if t == kind][:2]
        if sel:
            out.append({"name": kind, "sel": sel})
    if len(lt) > 1 and lt[0] != lt[1]:
        out.append({"name": "interleave_pair", "sel": [0, 1]})
    return out


def apply_gptoss(cfg: dict, var: dict, edits: list[str]) -> None:
    n = cfg["num_hidden_layers"]
    _slice_layer_lists(cfg, n, var["sel"], edits)
    cfg["num_hidden_layers"] = len(var["sel"])



_PERIOD_FIELDS = ("full_attention_interval", "attention_interval",
                  "linear_attention_interval", "moe_layer_interval")


def _min_depth_for_periods(tc: dict) -> int:
    """Minimum layer count that keeps a period-derived architecture faithful.

    Some models derive layer kinds from a PERIOD rather than a per-layer list
    (Qwen3.5: full attention where (i+1) %% full_attention_interval == 0).
    Rescaling the period to fit a 2-layer cut produces a configuration that
    does not exist upstream (interval=1 means every layer is full attention)
    and still trips capacity asserts (mamba_cache_per_req > 0). Keep the real
    period and cut to a whole number of periods instead: a dummy must be a
    SHORTENED model, never a MODIFIED one.
    """
    periods = [int(tc[f]) for f in _PERIOD_FIELDS
               if isinstance(tc.get(f), int) and tc[f] > 1]
    if not periods:
        return 0
    import math
    step = math.lcm(*periods) if len(periods) > 1 else periods[0]
    return step * 2  # two full periods: exercises both kinds with real spacing


def _layer_axis(cfg: dict) -> tuple[str | None, list]:
    """Find the per-layer type list (name, values) if the model interleaves."""
    tc = cfg.get("text_config", cfg)
    n = tc.get("num_hidden_layers")
    for key in ("layer_types", "attn_type_list", "hybrid_layer_pattern",
                "layers_block_type", "indexer_types", "mlp_layer_types", "moe_layer_freq"):
        v = tc.get(key)
        if isinstance(v, list) and n and len(v) == n and len(set(map(str, v))) > 1:
            return key, v
    pat = tc.get("hybrid_override_pattern")
    if isinstance(pat, str) and n and len(pat) == n:
        return "hybrid_override_pattern", list(pat)
    return None, []


def variants_generic(cfg: dict) -> list[dict]:
    """Depth-cut variants for any architecture: one per layer kind on the
    detected interleave axis (plus a mixed pair), or a single 2-layer variant
    for homogeneous models. MoE models skip the leading dense block."""
    tc = cfg.get("text_config", cfg)
    if "num_hidden_layers" not in tc and isinstance(tc.get("layers_block_type"), list):
        tc["num_hidden_layers"] = len(tc["layers_block_type"])  # Nemotron-Ultra schema
    n = tc["num_hidden_layers"]
    skip = int(tc.get("first_k_dense_replace") or 0)
    min_depth = _min_depth_for_periods(tc)
    if min_depth and min_depth <= n:
        # period-derived architecture: keep the real period, take whole periods
        return [{"name": f"depth{min_depth}", "sel": list(range(min_depth))}]
    axis, values = _layer_axis(cfg)
    if not axis:
        sel = list(range(skip, min(skip + 2, n))) or list(range(min(2, n)))
        return [{"name": "rep", "sel": sel}]
    kinds: list[str] = []
    for i, v in enumerate(values):
        if i >= skip and str(v) not in kinds:
            kinds.append(str(v))
    out = []
    for k in kinds[:4]:
        sel = [i for i, v in enumerate(values) if str(v) == k and i >= skip][:2]
        if sel:
            safe = "".join(ch if ch.isalnum() else "_" for ch in k)[:20]
            out.append({"name": f"{axis}_{safe}"[:40], "sel": sel})
    # A variant holding only ONE layer kind is often structurally illegal:
    # hybrid-SWA models assert "at least one SWA layer", mamba/GDN hybrids
    # divide by the attention-layer count, vllm's kv grouping needs every
    # kv-spec kind. So the FULL-COVERAGE variant (one layer of every kind)
    # comes first and single-kind variants are kept only as extras.
    cover: list[int] = []
    for k in kinds:
        idx = [i for i, v in enumerate(values) if str(v) == k and i >= skip]
        if idx:
            cover.append(idx[0])
    if len(cover) > 1:
        out.insert(0, {"name": "all_kinds", "sel": sorted(cover)})
    return out


def apply_generic(cfg: dict, var: dict, edits: list[str]) -> None:
    tc = cfg.get("text_config", cfg)
    n = tc["num_hidden_layers"]
    sel = var["sel"]
    pat = tc.get("hybrid_override_pattern")
    if isinstance(pat, str) and len(pat) == n:
        tc["hybrid_override_pattern"] = "".join(pat[i] for i in sel)
        edits.append(f"hybrid_override_pattern -> {tc['hybrid_override_pattern']}")
    _slice_layer_lists(tc, n, sel, edits)
    tc["num_hidden_layers"] = len(sel)
    if tc.get("first_k_dense_replace"):
        tc["first_k_dense_replace"] = 0
        edits.append("first_k_dense_replace -> 0")
    for k in ("num_nextn_predict_layers", "num_mtp_modules"):
        if tc.get(k):
            tc[k] = 0
            edits.append(f"{k} -> 0")


ADAPTERS = {
    "dsv4": (variants_dsv4, apply_dsv4),
    "glm": (variants_glm, apply_glm),
    "m3": (variants_m3, apply_m3),
    "gptoss": (variants_gptoss, apply_gptoss),
    "generic": (variants_generic, apply_generic),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", type=Path, required=True, help="dir of <org>_<repo>.json originals")
    ap.add_argument("--out", type=Path, required=True, help="output root for dummy model dirs")
    args = ap.parse_args()

    manifest = {"generator": Path(__file__).name, "rule": "depth-only cut, width preserved", "variants": []}
    failures = 0
    for repo, family in load_repos(args.configs).items():
        src = args.configs / (repo.replace("/", "_") + ".json")
        if not src.exists():
            print(f"MISSING {src}", file=sys.stderr)
            failures += 1
            continue
        base = json.loads(src.read_text())
        src_sha = hashlib.sha256(src.read_bytes()).hexdigest()[:16]
        make_variants, apply = ADAPTERS[family]
        for var in make_variants(base):
            cfg = copy.deepcopy(base)
            edits: list[str] = []
            apply(cfg, var, edits)
            for k in ("num_nextn_predict_layers",):
                if cfg.get(k):
                    cfg[k] = 0
                    edits.append(f"{k} -> 0")
            _remap_quant_layer_entries(cfg, var["sel"], edits)
            new_n = cfg.get("num_hidden_layers") or cfg["text_config"]["num_hidden_layers"]
            stale = _check_no_stale_layer_refs(cfg, new_n)
            tag = f"{repo.split('/')[-1]}__{var['name']}"
            out_dir = args.out / family / tag
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
            # modelopt/NVFP4 repos carry the authoritative quant description in
            # a SEPARATE hf_quant_config.json; without it the framework loads
            # the checkpoint as unquantized (looked like a silent-downgrade bug
            # until the missing file was found). Remap its layer refs too.
            hq_src = args.configs / (repo.replace("/", "_") + "_hfquant.json")
            if hq_src.exists():
                hq = json.loads(hq_src.read_text())
                _remap_quant_layer_entries({"quantization_config": hq.get("quantization", hq)},
                                           var["sel"], edits)
                (out_dir / "hf_quant_config.json").write_text(json.dumps(hq, indent=2))
                edits.append("hf_quant_config.json copied + layer refs remapped")
            entry = {
                "variant": tag,
                "repo": repo,
                "family": family,
                "layer_kind": var["name"],
                "original_layer_indices": var["sel"],
                "num_layers": new_n,
                "source_config_sha256_16": src_sha,
                "edits": edits,
                "stale_layer_refs": stale,
                "caveats": [],
            }
            if family == "glm":
                freq, off = base.get("index_topk_freq"), base.get("index_skip_topk_offset")
                entry["caveats"].append(
                    f"index_topk_freq={freq}/offset={off} are phase-based on absolute layer index; "
                    f"original indices {var['sel']} remap to 0..{new_n - 1}, so topk phase may differ "
                    "from the full model — cross-check the indexer topk path on the real depth once."
                )
            if stale:
                print(f"STALE LAYER REFS in {tag}: {stale}", file=sys.stderr)
                failures += 1
            manifest["variants"].append(entry)
            print(f"wrote {out_dir}  layers={new_n}  orig={var['sel']}")

    mpath = args.out / "variants_manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2))
    print(f"\nmanifest: {mpath}  ({len(manifest['variants'])} variants, {failures} failures)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
