# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OFFLINE single-GPU real-routing capture for the AIC MoE perf collector.

WHY THIS EXISTS
---------------
``collector/vllm/collect_moe.py`` decides how many tokens land on each expert
with a synthetic ``power_law_1.2`` distribution. That distribution is far too
concentrated: replayed onto ep=4 it puts the bottleneck rank at MAX/MEAN ~= 1.82x,
whereas the golden tp4/ep4 deployment (real routing, no EPLB) sits at ~1.06x and
the prior *serving* routing capture at ~1.25-1.30x. See
``runs/BUSY_METRIC_VERDICT_PARTD_RESOLUTION.md``.

The fix is to drive the perf collector from a *real* routing histogram. Real
routing can only be measured by running a real model on real text -- routing is a
function of the gate logits, which are a function of real activations, so random
activations measure nothing.

WHAT THIS DOES
--------------
A pure-offline harness (NO serving / Dynamo stack): loads the real model on a
single GPU with ``vllm.LLM``, feeds real tokenized text, and loops ``num_tokens``
directly (prefill big blocks + decode small batches) to mirror the perf
collector's ``num_tokens`` sweep (``collector/cases/base_ops/moe.yaml``). Real
per-(layer, expert) routing counts are captured by the EXISTING, UNMODIFIED hook
``inject/fpm_routing_capture.py`` (which already solves the B300 monolithic-kernel
bypass) and flushed to ``.npz`` sidecars. We then aggregate to an EP-agnostic
per-expert histogram keyed by ``(phase, num_tokens_bin)``.

HOOK INSTALL ORDER (critical)
-----------------------------
The hook patches ``MoERunner._apply_quant_method`` + ``GPUModelRunner`` at import
time, and must be installed BEFORE the model is loaded / CUDA graphs captured.
  * tp1: we force ``VLLM_ENABLE_V1_MULTIPROCESSING=0`` (engine runs in THIS
    process) and ``import fpm_routing_capture`` here, before ``LLM()``. The hook
    patches the very classes the in-process engine uses. We can then flush
    deterministically by calling the module's own ``_flush()``.
  * tp>1: TP workers are separate processes, so we put the inject dir on
    ``PYTHONPATH``; ``inject/sitecustomize.py`` auto-imports the hook at worker
    interpreter startup (same mechanism the serving runbook uses). We rely on a
    small ``FPM_ROUTING_FLUSH_EVERY`` for mid-run flushes.

USAGE
-----
  uv run python collector/layerwise/fpm_ground_truth/routing_capture/offline_capture.py \
      capture --tp 1 --out /workspace/.../offline_tp1
  uv run python .../offline_capture.py aggregate \
      --out /workspace/.../offline_tp1 --json routing_dist_qwen36.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

# --- perf-collector num_tokens sweep (collector/cases/base_ops/moe.yaml) -------
SCAN = [1, 2, 4, 8, 16, 32, 48, 64, 80, 96, 128, 160, 192, 256, 320, 384, 512,
        768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384]

HERE = os.path.dirname(os.path.abspath(__file__))
INJECT_DIR = os.path.join(HERE, "inject")
DEFAULT_MODEL = "/workspace/models/Qwen3.6-35B-A3B"


# ============================================================================ #
# corpus: REAL text tokens (never random activations)
# ============================================================================ #
def _build_corpus_text(min_chars: int = 400_000) -> str:
    """Concatenate real repo text (markdown docs + python source) as the prompt
    corpus. Real natural-language + code tokens exercise real gate logits."""
    roots = [
        os.path.join(HERE, "..", "..", "..", ".."),  # repo root
    ]
    repo_root = os.path.abspath(roots[0])
    patterns = [
        "docs/**/*.md", "*.md", "README*",
        "src/aiconfigurator/**/*.py", "collector/**/*.py",
        "benchmarks/**/*.py",
    ]
    chunks: list[str] = []
    total = 0
    seen: set[str] = set()
    for pat in patterns:
        for p in sorted(glob.glob(os.path.join(repo_root, pat), recursive=True)):
            if p in seen or not os.path.isfile(p):
                continue
            seen.add(p)
            try:
                with open(p, "r", errors="ignore") as f:
                    txt = f.read()
            except Exception:
                continue
            if len(txt) < 200:
                continue
            chunks.append(txt)
            total += len(txt)
            if total >= min_chars:
                break
        if total >= min_chars:
            break
    text = "\n\n".join(chunks)
    if len(text) < min_chars:
        # tile until we have enough real tokens to slice from
        reps = (min_chars // max(1, len(text))) + 2
        text = "\n\n".join([text] * reps)
    return text


# ============================================================================ #
# capture
# ============================================================================ #
def cmd_capture(args: argparse.Namespace) -> int:
    out_dir = os.path.abspath(args.out)
    routing_dir = os.path.join(out_dir, "routing")
    os.makedirs(routing_dir, exist_ok=True)

    # --- env: hook config (must be set before importing the inject module) ----
    os.environ["FPM_ROUTING_STAGE"] = "A"
    os.environ["FPM_ROUTING_OUT"] = routing_dir
    os.environ["FPM_ROUTING_CAPTURE_RANKS"] = "0"
    os.environ.setdefault("FPM_ROUTING_FLUSH_EVERY", "8")
    # make sitecustomize reachable by spawned TP workers (tp>1 path)
    os.environ["PYTHONPATH"] = INJECT_DIR + os.pathsep + os.environ.get("PYTHONPATH", "")

    inproc = args.tp == 1
    if inproc:
        # run the engine in THIS process so the in-process import patches the
        # exact classes the engine uses, and so we can flush deterministically.
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    if INJECT_DIR not in sys.path:
        sys.path.insert(0, INJECT_DIR)

    # IMPORT THE HOOK BEFORE LLM() -- order matters.
    fpm_mod = None
    try:
        import fpm_routing_capture as fpm_mod  # noqa: F401
        print(f"[offline] imported hook in-process (installed={fpm_mod._S['installed']})",
              file=sys.stderr)
    except Exception as e:  # pragma: no cover
        print(f"[offline] WARN: could not import hook in-process: {e!r}", file=sys.stderr)

    from vllm import LLM, SamplingParams

    print(f"[offline] loading model={args.model} tp={args.tp} (this can take minutes)",
          file=sys.stderr)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_len,
        max_num_batched_tokens=args.max_len,
        enable_prefix_caching=False,           # full recompute every prompt
        trust_remote_code=True,
        enforce_eager=False,                   # keep CUDA-graph capture (real path)
        limit_mm_per_prompt={"image": 0, "video": 0},
        seed=0,
    )

    tok = llm.get_tokenizer()
    corpus_ids = tok(_build_corpus_text(), add_special_tokens=False)["input_ids"]
    n_corpus = len(corpus_ids)
    print(f"[offline] corpus tokens={n_corpus}", file=sys.stderr)

    base_off = (args.content_offset * 1009) % max(1, n_corpus)

    def slice_ids(start: int, length: int) -> list[int]:
        start %= n_corpus
        if start + length <= n_corpus:
            return corpus_ids[start:start + length]
        out = []
        while len(out) < length:
            take = min(length - len(out), n_corpus - start)
            out.extend(corpus_ids[start:start + take])
            start = 0
        return out

    # ---- PREFILL sweep: one long real prompt per scan point, max_tokens=1 ----
    prefill_pts = [n for n in SCAN if args.min_prefill <= n <= args.max_prefill]
    print(f"[offline] PREFILL points: {prefill_pts}", file=sys.stderr)
    off = base_off
    for n in prefill_pts:
        ids = slice_ids(off, n)
        off += n + 7  # rotate so successive prompts differ
        sp = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
        llm.generate([{"prompt_token_ids": ids}], sp, use_tqdm=False)
        print(f"[offline]   prefill N={n} done", file=sys.stderr)

    # ---- DECODE sweep: batch B real prompts, force D decode steps -------------
    decode_batches = [b for b in [1, 2, 4, 8, 16, 32, 64, 128, 256]
                      if b <= args.max_decode_batch]
    decode_len = args.decode_len
    plen = 48
    print(f"[offline] DECODE batches: {decode_batches} (decode_len={decode_len})",
          file=sys.stderr)
    off = base_off + 50_000
    for b in decode_batches:
        prompts = []
        for _ in range(b):
            prompts.append({"prompt_token_ids": slice_ids(off, plen)})
            off += plen + 5
        sp = SamplingParams(temperature=0.0, max_tokens=decode_len,
                            min_tokens=decode_len, ignore_eos=True)
        llm.generate(prompts, sp, use_tqdm=False)
        print(f"[offline]   decode B={b} done", file=sys.stderr)

    # ---- flush ---------------------------------------------------------------
    if inproc and fpm_mod is not None:
        try:
            fpm_mod._flush()
            fpm_mod._write_manifest()
            print(f"[offline] in-process flush: hook_calls={fpm_mod._S['hook_calls']} "
                  f"steps={fpm_mod._S['step']}", file=sys.stderr)
        except Exception as e:
            print(f"[offline] WARN flush failed: {e!r}", file=sys.stderr)

    try:
        del llm
    except Exception:
        pass

    print(f"[offline] capture done -> {routing_dir}", file=sys.stderr)
    return 0


# ============================================================================ #
# aggregate
# ============================================================================ #
def _snap(v: int) -> int:
    return min(SCAN, key=lambda s: abs(s - v))


def _load_dir(run_dir: str):
    import numpy as np
    parts = sorted(glob.glob(os.path.join(run_dir, "routing", "routing_rank0_part*.npz")))
    if not parts:
        parts = sorted(glob.glob(os.path.join(run_dir, "routing_rank0_part*.npz")))
    counts, ct, cr, dr, tot = [], [], [], [], []
    for p in parts:
        z = np.load(p)
        counts.append(z["counts"])
        ct.append(z["ctx_tokens"]); cr.append(z["ctx_requests"])
        dr.append(z["decode_requests"]); tot.append(z["total_tokens"])
    if not counts:
        raise SystemExit(f"no npz under {run_dir}")
    return (np.concatenate(counts), np.concatenate(ct), np.concatenate(cr),
            np.concatenate(dr), np.concatenate(tot), parts)


def _read_manifest(run_dir: str) -> dict:
    for cand in (os.path.join(run_dir, "routing", "manifest_rank0.txt"),
                 os.path.join(run_dir, "manifest_rank0.txt")):
        if os.path.isfile(cand):
            d = {}
            for line in open(cand):
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    d[k] = v
            return d
    return {}


def _perrank_maxmean(dist, ep: int):
    """ep-agnostic per-rank token load MAX/MEAN from a [E] probability vector
    (contiguous EP map: expert e -> rank e // (E//ep))."""
    import numpy as np
    E = len(dist)
    per = E // ep
    loads = np.asarray(dist[:per * ep]).reshape(ep, per).sum(1)
    m = loads.mean()
    return float(loads.max() / m) if m > 0 else float("nan")


def cmd_aggregate(args: argparse.Namespace) -> int:
    import numpy as np

    all_counts, all_ct, all_cr, all_dr, all_tot = [], [], [], [], []
    manifest = {}
    for run_dir in args.out:
        c, ct, cr, dr, tot, parts = _load_dir(os.path.abspath(run_dir))
        manifest = manifest or _read_manifest(os.path.abspath(run_dir))
        # drop first few warmup/compile steps per run
        keep = slice(args.warmup, None)
        all_counts.append(c[keep]); all_ct.append(ct[keep]); all_cr.append(cr[keep])
        all_dr.append(dr[keep]); all_tot.append(tot[keep])
        print(f"[agg] {run_dir}: {len(parts)} npz, {c.shape[0]} steps "
              f"(layers={c.shape[1]} experts={c.shape[2]})", file=sys.stderr)
    counts = np.concatenate(all_counts)
    ct = np.concatenate(all_ct); cr = np.concatenate(all_cr)
    dr = np.concatenate(all_dr); tot = np.concatenate(all_tot)
    n_steps, n_layers, n_experts = counts.shape
    top_k = int(manifest.get("top_k", 0)) or 8

    # ---- A. routing conservation: per layer, Sum_e(counts) == num_tokens*topk.
    # The npz sums every layer, so the whole-step total is num_layers x that.
    persum = counts.reshape(n_steps, -1).sum(1)
    expect = tot.astype(np.int64) * top_k * n_layers
    nz = expect > 0
    exact = int((persum[nz] == expect[nz]).sum())
    print(f"\n[A] conservation: {exact}/{int(nz.sum())} steps have "
          f"Sum(counts)==num_tokens*topk*num_layers (top_k={top_k}, "
          f"num_layers={n_layers})", file=sys.stderr)
    if nz.sum():
        ratio = persum[nz] / expect[nz]
        print(f"    ratio Sum/(tok*topk*layers): mean={ratio.mean():.6f} "
              f"min={ratio.min():.6f} max={ratio.max():.6f} "
              f"(1.0 == perfect routing conservation)", file=sys.stderr)
    print(f"    manifest hook_calls={manifest.get('hook_calls')}", file=sys.stderr)

    # ---- phase + num_tokens per step ----------------------------------------
    phase = np.where((ct > 0) & (dr == 0), "context",
             np.where((ct == 0) & (dr > 0), "decode",
              np.where((ct > 0) & (dr > 0), "mixed", "empty")))
    ntok = np.where(phase == "context", ct,
            np.where(phase == "decode", dr, tot))

    # ---- group by (phase, snapped bin) -> cross-step+cross-layer histogram ---
    groups: dict[str, dict] = {}
    for i in range(n_steps):
        ph = str(phase[i])
        if ph in ("empty", "mixed"):
            continue
        b = _snap(int(ntok[i]))
        key = f"{ph}:{b}"
        g = groups.setdefault(key, {"sum": np.zeros(n_experts, np.float64),
                                    "n_steps": 0, "tokens": 0,
                                    "perlayer_mm": []})
        layer_sum = counts[i].sum(0).astype(np.float64)  # [E] over layers
        g["sum"] += layer_sum
        g["n_steps"] += 1
        g["tokens"] += int(tot[i])
        # per-layer per-rank max/mean (matches analyze_final.add_skew), ep=4
        ep = args.ep
        per = n_experts // ep
        loads = counts[i][:, :per * ep].reshape(n_layers, ep, per).sum(2).astype(float)
        mean = loads.mean(1); mx = loads.max(1); act = mean > 0
        if act.any():
            g["perlayer_mm"].append(float(np.nanmean(np.where(act, mx / mean, np.nan))))

    out = {
        "model": "qwen36",
        "model_path": DEFAULT_MODEL,
        "num_layers": n_layers,
        "num_experts": n_experts,
        "top_k": top_k,
        "ep_for_perrank_diag": args.ep,
        "experts_per_rank_diag": n_experts // args.ep,
        "bins": {},
    }
    for key in sorted(groups, key=lambda k: (k.split(":")[0], int(k.split(":")[1]))):
        g = groups[key]
        s = g["sum"]
        dist = (s / s.sum()).tolist() if s.sum() > 0 else s.tolist()
        out["bins"][key] = {
            "dist": dist,
            "n_steps": g["n_steps"],
            "tokens": g["tokens"],
            # aggregate (cross-layer-summed) per-rank imbalance
            "perrank_maxmean_agg": _perrank_maxmean(dist, args.ep),
            # per-layer-per-step averaged (comparable to serving capture number)
            "perrank_maxmean_perlayer": (
                float(np.mean(g["perlayer_mm"])) if g["perlayer_mm"] else float("nan")),
        }

    with open(args.json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[agg] wrote {args.json}: {len(out['bins'])} bins", file=sys.stderr)

    # ---- B. report per-rank imbalance ---------------------------------------
    print(f"\n[B] per-rank token load MAX/MEAN  (ep={args.ep}, "
          f"experts_per_rank={n_experts // args.ep})", file=sys.stderr)
    print(f"    {'bin':22s} {'n_steps':>7s} {'agg':>7s} {'perlayer':>9s}", file=sys.stderr)
    for key in sorted(out["bins"], key=lambda k: (k.split(":")[0], int(k.split(":")[1]))):
        bb = out["bins"][key]
        print(f"    {key:22s} {bb['n_steps']:7d} {bb['perrank_maxmean_agg']:7.3f} "
              f"{bb['perrank_maxmean_perlayer']:9.3f}", file=sys.stderr)
    # overall (all steps pooled)
    pooled = np.zeros(n_experts, np.float64)
    for g in groups.values():
        pooled += g["sum"]
    if pooled.sum() > 0:
        pooled /= pooled.sum()
        print(f"    {'POOLED(all)':22s} {n_steps:7d} "
              f"{_perrank_maxmean(pooled, args.ep):7.3f}", file=sys.stderr)
    return 0


# ============================================================================ #
# compare (C / D): per-expert distribution similarity between two runs
# ============================================================================ #
def cmd_compare(args: argparse.Namespace) -> int:
    import numpy as np

    def pooled_dist(run_dir):
        c, ct, cr, dr, tot, _ = _load_dir(os.path.abspath(run_dir))
        c = c[args.warmup:]
        s = c.reshape(c.shape[0], c.shape[1], c.shape[2]).sum(0).sum(0).astype(np.float64)
        return s / s.sum()

    a = pooled_dist(args.a)
    b = pooled_dist(args.b)
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    maxabs = float(np.max(np.abs(a - b)))
    l1 = float(np.sum(np.abs(a - b)))
    print(f"[compare] {args.a}  vs  {args.b}", file=sys.stderr)
    print(f"    cosine_sim = {cos:.6f}", file=sys.stderr)
    print(f"    max_abs_diff = {maxabs:.6e}  (uniform={1/len(a):.3e})", file=sys.stderr)
    print(f"    L1_diff = {l1:.6f}", file=sys.stderr)
    print(f"    per-rank MAX/MEAN: a={_perrank_maxmean(a, args.ep):.3f} "
          f"b={_perrank_maxmean(b, args.ep):.3f}", file=sys.stderr)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("capture", help="load model, drive real workload, capture routing")
    c.add_argument("--out", required=True)
    c.add_argument("--model", default=DEFAULT_MODEL)
    c.add_argument("--tp", type=int, default=1)
    c.add_argument("--gpu-mem", type=float, default=0.9)
    c.add_argument("--max-len", type=int, default=16384)
    c.add_argument("--min-prefill", type=int, default=64)
    c.add_argument("--max-prefill", type=int, default=8192)
    c.add_argument("--max-decode-batch", type=int, default=256)
    c.add_argument("--decode-len", type=int, default=32)
    c.add_argument("--content-offset", type=int, default=0,
                   help="vary corpus content (for workload-sufficiency test D)")
    c.set_defaults(func=cmd_capture)

    a = sub.add_parser("aggregate", help="npz -> routing_dist json + A/B report")
    a.add_argument("--out", nargs="+", required=True, help="capture out dir(s)")
    a.add_argument("--json", default="routing_dist_qwen36.json")
    a.add_argument("--ep", type=int, default=4)
    a.add_argument("--warmup", type=int, default=10)
    a.set_defaults(func=cmd_aggregate)

    cm = sub.add_parser("compare", help="per-expert dist similarity (C/D)")
    cm.add_argument("--a", required=True)
    cm.add_argument("--b", required=True)
    cm.add_argument("--ep", type=int, default=4)
    cm.add_argument("--warmup", type=int, default=10)
    cm.set_defaults(func=cmd_compare)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
