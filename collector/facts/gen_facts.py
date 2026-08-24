#!/usr/bin/env python3
"""Facts generator (sweep driver): targets.yaml -> generator-rendered engine args
-> probe runs -> facts archive with provenance.

Runs on the host (needs the aiconfigurator repo importable for rendering).

  PYTHONPATH=<aic>/src python3 probe/gen_facts.py --plan            # show runs
  PYTHONPATH=<aic>/src python3 probe/gen_facts.py --emit-queues     # write per-GPU queue scripts
  python3 probe/gen_facts.py --collect                              # raw JSONs -> archive.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import sys
from pathlib import Path

import yaml

# workspace: where dummy_models/, archive/ and probe outputs live
ROOT = Path(os.environ.get("AIC_PROBE_WORKSPACE", Path.cwd()))
# generator source: this repo by default; override to pin a specific checkout
AIC_SRC = os.environ.get("AIC_GENERATOR_SRC",
                         str(Path(__file__).resolve().parents[1] / "aic" / "src"))
if AIC_SRC not in sys.path:
    sys.path.insert(0, AIC_SRC)
WORK = "/work"  # container mount of ROOT
SCRATCH_QUEUES = ROOT / "archive" / "queues"





GOLDEN_TARGET = {"sglang": "dynamo-python", "vllm": "fpm", "trtllm": "dynamo-python"}
VENV_PY = ROOT / "venv_aic" / "bin" / "python"


def render_golden(run: dict) -> Path | None:
    """Invoke the REAL user-facing generator command and archive it verbatim.

    golden/<id>/command.txt is the exact `aiconfigurator cli generate` argv —
    the thing we converge on and guarantee. Artifacts are stored untouched;
    every probe-side adaptation happens later as a RECORDED post-process.
    Owner decisions: --system comes from targets.platform (h200_sxm proxies
    the H20 probe box — same VRAM, h20 deliberately not added to code); per-checkpoint extra args live in targets.yaml
    checkpoint_overrides.cli_extra_args and are spliced into the command.
    """
    import shutil
    import subprocess
    gdir = ROOT / "archive" / "golden" / run["id"]
    cmd = [str(VENV_PY), "-m", "aiconfigurator.main", "cli", "generate",
           "--model-path", run["repo"],
           "--total-gpus", str(run["tp"]),
           "--system", run["system"],
           "--backend", run["backend"],
           "--deployment-target", GOLDEN_TARGET[run["backend"]],
           "--config-template-version", run["version"],
           "--save-dir", str(gdir)]
    cmd += list(run.get("cli_extra_args") or [])
    cmd_txt = shlex.join(cmd)
    import subprocess as _sp
    gen_commit = _sp.run(["git", "-C", str(ROOT / "aic"), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    stamp = gdir / "command.txt"
    # cache valid only for the SAME command rendered by the SAME generator code
    if stamp.exists() and stamp.read_text().splitlines()[:2] == [cmd_txt, f"# generator={gen_commit}"]:
        sub = next((d for d in gdir.iterdir() if d.is_dir()), None)
        if sub is not None:
            return sub  # cached golden for the identical command
    if gdir.exists():
        shutil.rmtree(gdir)
    gdir.mkdir(parents=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "aic" / "aic-core" / "src")
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=900)
    stamp.write_text(cmd_txt + f"\n# generator={gen_commit}\n# exit={r.returncode}\n")
    (gdir / "render.log").write_text((r.stdout or "")[-8000:] + (r.stderr or "")[-8000:])
    if r.returncode != 0:
        run["golden_error"] = (r.stderr or r.stdout or "").strip().splitlines()[-1][:200] if (r.stderr or r.stdout) else "no output"
        return None
    return next((d for d in gdir.iterdir() if d.is_dir()), None)


def extract_sglang_cli_from_run_sh(run_sh: Path) -> str:
    """Post-process: lift the dynamo.sglang engine args out of golden run_0.sh.
    Drops only wrapper plumbing ($MODEL_PATH placeholder, host/metrics/shell);
    engine-selection flags pass through verbatim."""
    import re
    text = run_sh.read_text()
    m = re.search(r"python3 -m dynamo\.sglang((?:[^\n]*\\\n)*[^\n]*)", text)
    if not m:
        raise SystemExit(f"golden run_0.sh has no dynamo.sglang block: {run_sh}")
    block = m.group(1).replace("\\\n", " ")
    block = re.split(r"\s(?:2>&1|\||&|;|\))", block)[0]
    toks = shlex.split(block)
    out, skip = [], False
    DROP = {"--model-path", "--served-model-name", "--host", "--port"}
    FLAG_DROP = {"--enable-metrics"}
    for i, tk in enumerate(toks):
        if skip:
            skip = False
            continue
        if tk in DROP:
            skip = True
            continue
        if tk in FLAG_DROP:
            continue
        out.append(tk)
    return " ".join(out)


def _cea(v):
    """cli_extra_args entry: plain list, or {args: [...], fact: "<evidence>"} —
    the fact field cites the probe evidence this generator input derives from."""
    if not v:
        return []
    return list(v["args"]) if isinstance(v, dict) else list(v)


def derive_roster_checkpoints(fam: dict, targets: dict) -> list[dict]:
    """Roster checkpoints DERIVED from the collector's own case declarations:
    every org/repo its cases yamls mention, minus gated repos and repos owned
    by other (special-adapter) families, plus probe-only extra_repos. Profile
    comes from the checkpoint's quant metadata; variants from the dummy
    manifest. targets.yaml holds only the overlay (checkpoint_overrides)."""
    import re
    sys.path.insert(0, str(Path(__file__).parent))

    cases = Path(AIC_SRC).parent / "collector" / "cases" / "models"
    org = (r"(?:deepseek-ai|zai-org|moonshotai|nvidia|openai|meta-llama|"
           r"mistralai|google|Qwen|XiaomiMiMo|MiniMaxAI|sgl-project)")
    mentioned: set[str] = set()
    for f in cases.glob("*_cases.yaml"):
        for m in re.findall(rf"\b({org}/[\w.\-]+)", f.read_text()):
            # brace-expansion prose like org/Name-{A,B}-X truncates at '{'
            if not m.endswith("-") and not m.endswith(".py"):
                mentioned.add(m)
    # owner-decided exclusions live in targets.yaml roster.excluded (each entry
    # carries decided_by/reason); any OTHER missing config is a hard stop that
    # goes back to the owner — there is no self-service escape hatch.
    excluded = {e["repo"] for e in (fam.get("excluded") or [])}
    owned_elsewhere = {ck["repo"] for fname, f in targets["families"].items()
                      if not f.get("derive") for ck in f.get("checkpoints", [])}
    repos = sorted((mentioned - excluded - owned_elsewhere)
                   | set(fam.get("extra_repos") or []))
    manifest = json.loads((ROOT / "dummy_models" / "variants_manifest.json").read_text())
    variants_of: dict[str, list[str]] = {}
    for v in manifest["variants"]:
        variants_of.setdefault(v["repo"], []).append(v["variant"].split("__", 1)[1])
    # representative-first ordering: index 0 is the default probe variant
    _head = {"rep": 0, "all_kinds": 1, "rep_mix": 2, "interleave_pair": 3}

    def _rank(n: str):
        if n.startswith("depth"):  # deeper = more faithful; depth8 before depth4
            return (4, -int(n[5:]))
        return (_head.get(n, 9), n)
    for vs in variants_of.values():
        vs.sort(key=_rank)
    overrides = fam.get("checkpoint_overrides") or {}
    out = []
    for repo in repos:
        profile = derive_profile(repo, ROOT / "configs")
        if profile == "MISSING":
            raise SystemExit(f"derive_roster: no fetched config for {repo} — OWNER DECISION NEEDED "
                             f"(fetch the config, or the owner records an exclusion in targets.yaml roster.excluded)")
        ck = {"repo": repo, "profile": profile, "variants": variants_of.get(repo, [])}
        ov = dict(overrides.get(repo) or {})
        if "profile" in ov:  # explicit pin wins, but derivation drift is loud
            if ov["profile"] != profile:
                print(f"derive_roster: {repo} profile pinned {ov['profile']} != derived {profile}")
            ck["profile"] = ov.pop("profile")
        ck.update(ov)
        out.append(ck)
    return out


def enumerate_runs(targets: dict, full: bool, backends: list[str]) -> list[dict]:
    runs = []
    topos = [t for t in targets["topologies"] if t["evidence"] == "real" and (full or t["tp"] == 1)]
    for backend in backends:
        be = targets["backends"][backend]
        versions = be["versions"] if full else [be["versions"][-1]]
        for fam_name, fam in targets["families"].items():
            if fam.get("derive") and "checkpoints" not in fam:
                fam["checkpoints"] = derive_roster_checkpoints(fam, targets)
            variants = fam.get("dummy_variants") or []
            if not variants and not any(c.get("variants") for c in fam["checkpoints"]):
                continue  # adapter pending (kimi_k3)
            override = (fam.get("variant_overrides") or {}).get(backend)
            for ck in fam["checkpoints"]:
                repo_tag = ck["repo"].split("/")[-1]
                # per-checkpoint variants win (architectures in a mixed family
                # each have their own layer kinds); else the family list
                ck_variants = ck.get("variants") or variants
                ck_override = (ck.get("variant_overrides") or {}).get(backend) or override
                use_variants = ck_variants if full else [ck_override or ck_variants[0]]
                for variant in use_variants:
                    # dummy dirs are keyed by ADAPTER family (a roster repo may
                    # still use a special adapter) — search every adapter dir,
                    # preferring the targets family, then generic, then the rest
                    _adapter_dirs = [fam.get("dummy_dir") or fam_name, "generic"] + \
                        sorted(d.name for d in (ROOT / "dummy_models").iterdir() if d.is_dir())
                    for _famdir in dict.fromkeys(_adapter_dirs):
                        vdir = ROOT / "dummy_models" / _famdir / f"{repo_tag}__{variant}"
                        if vdir.exists():
                            break
                    if not vdir.exists():
                        runs.append({"skip": f"no dummy dir {vdir.name}", "repo": ck["repo"], "variant": variant})
                        continue
                    for version in versions:
                        for topo in topos:
                            rid = hashlib.sha1(
                                f"{ck['repo']}|{variant}|{backend}|{version}|{ck['profile']}|tp{topo['tp']}".encode()
                            ).hexdigest()[:12]
                            plat = targets.get("platform") or {"name": "h20_sm90", "sm": 90, "system": "h200_sxm"}
                            runs.append({
                                "id": rid, "family": fam_name, "repo": ck["repo"], "profile": ck["profile"],
                                "platform": plat["name"], "sm": plat["sm"], "system": plat["system"],
                                "variant": variant, "backend": backend, "version": version,
                                "image": be["images"][version], "tp": topo["tp"],
                                "model_dir": f"{WORK}/{vdir.relative_to(ROOT)}",
                                "aic_registered": ck.get("aic_registered", False),
                                "cli_extra_args": (list(be.get("cli_extra_args") or [])
                                                   + _cea((ck.get("cli_extra_args") or {}).get(backend)
                                                          or (fam.get("cli_extra_args") or {}).get(backend))),
                            })
    return runs



def _generator_src_commit() -> dict:
    """Record WHICH aiconfigurator code rendered the engine args, so archive
    provenance survives checkout/branch changes."""
    import subprocess

    repo = str(Path(AIC_SRC).parent)
    try:
        rev = subprocess.run(["git", "-C", repo, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        branch = subprocess.run(["git", "-C", repo, "branch", "--show-current"],
                                capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        rev = branch = "unknown"
    return {"generator_src": repo, "generator_commit": rev, "generator_branch": branch or "detached"}


def emit_queues(runs: list[dict], n_gpus: int, gpu_offset: int, plan_name: str) -> None:
    SCRATCH_QUEUES.mkdir(parents=True, exist_ok=True)
    src_info = _generator_src_commit()
    for run in runs:
        if "skip" not in run:
            run.update(src_info)
    (ROOT / "archive" / "raw").mkdir(parents=True, exist_ok=True)
    (ROOT / "archive" / "run_sh").mkdir(parents=True, exist_ok=True)
    queues: dict[int, list[str]] = {g: [] for g in range(gpu_offset, gpu_offset + n_gpus)}
    for i, run in enumerate(r for r in runs if "skip" not in r):
        g = gpu_offset + i % n_gpus
        head = (f"[ -f {ROOT}/archive/raw/{run['id']}.json ] && "
                f"echo 'skip [{run['id']}] (done)' || {{ "
                f"echo '### [{run['id']}] {run['backend']} {run['repo']} {run['variant']} "
                f"{run['version']} tp{run['tp']}' && timeout 1500 docker run --rm "
                f"--gpus '\"device={g}\"' --shm-size 16g -e HF_HUB_OFFLINE=1 "
                f"-v {ROOT}:{WORK} -v {ROOT}/jitcache:/root/.cache "
                f"-e TRITON_CACHE_DIR=/root/.cache/triton -e DG_JIT_CACHE_DIR=/root/.cache/deep_gemm ")
        if run["backend"] == "sglang":
            art = render_golden(run)
            if art is None:
                run["skip"] = f"golden render failed: {run.get('golden_error')}"
                continue
            cli = extract_sglang_cli_from_run_sh(art / "run_0.sh")
            run["engine_cli"] = cli
            run["engine_args_fidelity"] = "cli-golden"
            run["golden_dir"] = str(art)
            cmd = (head + f"{run['image']} python3 {WORK}/probe/probe_sglang.py "
                   f"--model {run['model_dir']} --engine-cli {shlex.quote(cli)} --trace "
                   f"--out {WORK}/archive/raw/{run['id']}.json 2>&1 | tail -1 ; }}")
        elif run["backend"] == "vllm":  # golden fpm run.sh, consumed verbatim
            art = render_golden(run)
            if art is None:
                run["skip"] = f"golden render failed: {run.get('golden_error')}"
                continue
            src = next((p for p in (art / "run.sh", art / "run_0.sh") if p.exists()), None)
            if src is None:
                run["skip"] = f"golden has no run.sh: {art}"
                continue
            rsh = ROOT / "archive" / "run_sh" / f"{run['id']}.sh"
            rsh.write_text(src.read_text())
            run["run_sh"] = str(rsh)
            run["engine_args_fidelity"] = "cli-golden"
            run["golden_dir"] = str(art)
            cmd = (head + f"--entrypoint python3 {run['image']} {WORK}/probe/probe_vllm.py "
                   f"--run-sh {WORK}/archive/run_sh/{run['id']}.sh --model-override {run['model_dir']} "
                   f"--trace --out {WORK}/archive/raw/{run['id']}.json 2>&1 | tail -1 ; }}")
        else:  # trtllm: golden extra_engine_args (agg_config.yaml), consumed verbatim
            art = render_golden(run)
            if art is None:
                run["skip"] = f"golden render failed: {run.get('golden_error')}"
                continue
            src = art / "agg_config.yaml"
            if not src.exists():
                run["skip"] = f"golden has no agg_config.yaml: {art}"
                continue
            eyml = ROOT / "archive" / "run_sh" / f"{run['id']}.engine.yaml"
            eyml.write_text(src.read_text())
            run["engine_args_fidelity"] = "cli-golden"
            run["render_artifact"] = str(eyml)
            run["golden_dir"] = str(art)
            # any checkpoint with custom code (auto_map) needs it; cheapest
            # correct rule is to always pass it for dummy probing
            trc = "--trust-remote-code "
            cmd = (head.replace("docker run --rm ",
                                "docker run --rm -e TLLM_WORKER_USE_SINGLE_PROCESS=1 ")
                   + f"{run['image']} bash -lc 'python3 {WORK}/probe/probe_trtllm.py "
                   f"--model {run['model_dir']} {trc}"
                   f"--engine-yaml {WORK}/archive/run_sh/{run['id']}.engine.yaml "
                   f"--out {WORK}/archive/raw/{run['id']}.json' "
                   f"2>&1 | tail -1 ; }}")
        queues[g].append(cmd)
    for g, cmds in queues.items():
        p = SCRATCH_QUEUES / f"gpu{g}.sh"
        p.write_text("#!/bin/bash\n" + "\n".join(cmds) + f"\necho ARCHIVE_QUEUE_GPU{g}_DONE\n")
        print(f"{p}: {len(cmds)} jobs")
    (ROOT / "archive" / plan_name).write_text(json.dumps(runs, indent=1))
    print(f"plan: {ROOT / 'archive' / plan_name} ({sum(1 for r in runs if 'skip' not in r)} runs, "
          f"{sum(1 for r in runs if 'skip' in r)} skipped)")


def collect() -> None:
    plan: dict = {}
    for pf in sorted((ROOT / "archive").glob("plan*.json")):
        plan.update({r["id"]: r for r in json.loads(pf.read_text()) if "skip" not in r})
    out = ROOT / "archive" / "archive.jsonl"
    n_ok = n_err = n_missing = 0
    with out.open("w") as f:
        for rid, run in plan.items():
            raw = ROOT / "archive" / "raw" / f"{rid}.json"
            if not raw.exists():
                run["status"] = "missing"
                n_missing += 1
                f.write(json.dumps({"provenance": run}) + "\n")
                continue
            facts = json.loads(raw.read_text())
            soft = {"attn_hook", "moe_hook", "quant_hook"}  # probe degradations, not run failures
            errs = set(facts.get("errors", {}))
            status = "ok" if not errs else ("ok_degraded" if errs <= soft else "error")
            n_ok += status == "ok"
            n_err += status == "error"
            f.write(json.dumps({
                "provenance": {**run, "status": status, "evidence": "real", "platform": "h20_sm90"},
                "facts": facts,
            }) + "\n")
    print(f"{out}: {n_ok} ok, {n_err} with recorded errors, {n_missing} missing")



def check_coverage(targets: dict) -> None:
    """Coverage floor: the collector's declared model roster is a LOWER bound
    for probe targets (targets may exceed it, never trail it)."""
    import re
    cases = Path(AIC_SRC).parent / "collector" / "cases" / "models"
    mentioned: set[str] = set()
    org = r"(?:deepseek-ai|zai-org|moonshotai|nvidia|openai|meta-llama|mistralai|google|Qwen|XiaomiMiMo|MiniMaxAI|sgl-project)"
    for f in cases.glob("*_cases.yaml"):
        for m in re.findall(rf"\b({org}/[\w.\-]+)", f.read_text()):
            if not m.endswith("-") and not m.endswith(".py"):
                mentioned.add(m)
    for fam in targets["families"].values():
        if fam.get("derive") and "checkpoints" not in fam:
            fam["checkpoints"] = derive_roster_checkpoints(fam, targets)
    covered = {ck["repo"] for fam in targets["families"].values() for ck in fam["checkpoints"]}
    excluded = {e["repo"] for f in targets["families"].values()
                for e in (f.get("excluded") or [])}
    missing = sorted(mentioned - covered - excluded)
    print(f"collector mentions {len(mentioned)} repos; targets cover {len(covered)}; "
          f"owner-excluded {len(mentioned & excluded)}")
    if missing:
        print("MISSING FROM TARGETS (coverage floor violated):")
        for r in missing:
            print("  ", r)
        raise SystemExit(1)
    print("coverage floor satisfied")



# ---------------------------------------------------------------------------
# checkpoint quant profile (merged from derive_profile.py): derived from the
# checkpoint's own quant metadata, NEVER the repo name (dummy fidelity rule 4)
def derive_profile(repo: str, configs_dir: Path) -> str:
    stem = repo.replace('/', '_')
    p = configs_dir / f'{stem}.json'
    if not p.exists():
        return 'MISSING'
    c = json.loads(p.read_text())
    qc = c.get('quantization_config') or (c.get('text_config') or {}).get('quantization_config')
    hq_p = configs_dir / f'{stem}_hfquant.json'
    if hq_p.exists():
        algo = ((json.loads(hq_p.read_text()).get('quantization') or {}).get('quant_algo') or '').upper()
        if 'NVFP4' in algo or 'FP4' in algo:
            return 'nvfp4'
        if 'MXFP8' in algo:
            return 'mxfp8'
        if 'FP8' in algo:
            return 'fp8'
    if not qc:
        return 'bfloat16'

    def groups_have_4bit_float() -> bool:
        for g in (qc.get('config_groups') or {}).values():
            for part in ('weights', 'input_activations'):
                w = g.get(part) or {}
                if isinstance(w, dict) and w.get('num_bits') == 4 and w.get('type') == 'float':
                    return True
        # modelopt's other MIXED_PRECISION shape: flat per-layer dict
        for v in (qc.get('quantized_layers') or {}).values():
            if isinstance(v, dict) and 'FP4' in str(v.get('quant_algo', '')).upper():
                return True
        return False

    m = (qc.get('quant_method') or '').lower()
    algo = (qc.get('quant_algo') or '').upper()
    if m == 'mxfp8':
        return 'mxfp8'
    if m == 'mxfp4':
        return 'mxfp4'
    if m in ('modelopt', 'modelopt_mixed') or algo == 'MIXED_PRECISION':
        return 'nvfp4' if (groups_have_4bit_float() or 'FP4' in algo) else 'fp8'
    if m == 'compressed-tensors':
        # 4-bit float groups = fp4-family weights (e.g. Kimi-K3 native: w4 float
        # group-32); 4-bit int = packed w4 (marlin path), served fp8-activation
        if groups_have_4bit_float():
            return 'nvfp4'
        return 'fp8'
    if m == 'fp8':
        return 'fp8'
    return f'?{m}'

# ---------------------------------------------------------------------------
# records stage (merged from make_records.py): raw probe JSONs -> curated
# records.jsonl — kernel normalization, taxonomy labeling, error compression

ROOT = Path(__file__).resolve().parent.parent

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


def load_taxonomy():
    import yaml
    path = Path(__file__).parent / "kernel_taxonomy.yaml"
    rules = yaml.safe_load(path.read_text())["rules"]
    return [(re.compile(r["match"]), r["backend"], r["role"]) for r in rules]


_TAXONOMY = load_taxonomy()


def label_kernels(kernels):
    """kernel names -> ({backend labels}, {unmatched kernels})."""
    labels, unmatched = set(), set()
    for k in kernels:
        for rx, backend, role in _TAXONOMY:
            if rx.search(k):
                if backend not in ("infra", "framework_native"):
                    labels.add(backend)
                break
        else:
            unmatched.add(k)
    return labels, unmatched


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
        for v in merged.values():
            if v["kernels"] or v["op"]:
                labels, unmatched = label_kernels(v["kernels"])
                v["backends"] = sorted(labels) or None
                if unmatched:
                    v["unclassified_kernels"] = sorted(unmatched)
                ops.append(v)
    # trtllm probe: flat kernels list, no spans
    if not ops and facts.get("kernels"):
        kerns = sorted(set(filter(None, (normalize_kernel(k["kernel"]) for k in facts["kernels"]))))
        attributed |= set(kerns)
        labels, unmatched = label_kernels(kerns)
        ops.append({"phase": "generate", "op": "all", "quant": None, "api": None,
                    "kernels": kerns, "calls": 1, "backends": sorted(labels) or None,
                    "unclassified_kernels": sorted(unmatched) or None})
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


def build_records() -> None:
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
                            # parallel dims are identity keys: ep sharding is
                            # kernel-invariant on the a2a=none path (measured,
                            # facts/tepdep/), but a2a=deepep changes the route
                            "ep": run.get("ep", 1), "dp": run.get("dp", 1),
                            "a2a": run.get("a2a", "none"),
                            "engine_cli": run.get("engine_cli"),
                            "unknown_args": f.get("engine_cli_unknown_args") or None,
                            "platform": run.get("platform", "h20_sm90"),
                            "sm_measured": f.get("device_capability"),
                            "evidence": "real"},
                "resolved": {k: v for k, v in sa.items() if v is not None},
                "identity": {
                    "model_class": f.get("model_class"),
                    # sglang exposes the backend object; vllm/trtllm only reveal
                    # it through the wrapped attention spans — take either.
                    # sglang: backend object; vllm: wrapped attention spans;
                    # trtllm: no spans — fall back to the attention kernel family
                    "attn_backend": ((f.get("attn_backend") or "").rsplit(".", 1)[-1]
                                     or next((s.split("::")[2] for s in (f.get("api_trace") or {})
                                              if "::attn::" in s), None)
                                     or next((normalize_kernel(k["kernel"])
                                              for k in (f.get("kernels") or [])
                                              if re.search(r"fmha|flash_?attn|flash_fwd|"
                                                           r"mla_|attention_kernel|paged_kv",
                                                           k["kernel"], re.I)
                                              and "norm" not in k["kernel"].lower()), None)),
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

# ---------------------------------------------------------------------------
# golden render snapshots (merged from golden_snapshots.py): facts freshness

SNAP_PLANS = {
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
        return extract_sglang_cli_from_run_sh(p) + "\n" if p.exists() else None
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


def snap_iter_runs(backend: str):
    for run in json.loads((ROOT / "archive" / SNAP_PLANS[backend]).read_text()):
        if isinstance(run, dict) and "skip" not in run and run.get("id"):
            yield run


def snap_build(backend: str, render: bool) -> tuple[dict, list[str]]:
    entries, errors = {}, []
    for run in snap_iter_runs(backend):
        key = f"{run['repo']}|tp{run['tp']}"
        art = None
        if render:
            art = render_golden(dict(run))  # copy: render mutates on failure
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

def snapshots_cmd(update: bool, snapshot_dir: Path | None = None) -> int:
    """--snapshot-update / --snapshot-check: facts freshness tripwire.

    Pins the rendered surface each archived fact was measured under (probed
    slice only — provenance, NOT a generator regression suite). check
    re-renders (CPU, cached on generator commit) and lists STALE cells.
    """
    import difflib
    import subprocess
    snapshot_dir = snapshot_dir or Path(__file__).parent / "golden_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    gen_commit = subprocess.run(["git", "-C", str(ROOT / "aic"), "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True).stdout.strip()
    rc = 0
    for backend in SNAP_PLANS:
        snap_path = snapshot_dir / f"{backend}.yaml"
        entries, errors = snap_build(backend, render=not update)
        for e in errors:
            print(f"[render-error] {e}")
        if update:
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=Path, default=Path(__file__).parent / "targets.yaml")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--emit-queues", action="store_true")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--check-coverage", action="store_true",
                    help="every model repo the collector's case yamls mention must be a target")
    ap.add_argument("--full", action="store_true", help="all variants x both sglang versions (default: representative)")
    ap.add_argument("--gpus", type=int, default=4)
    ap.add_argument("--gpu-offset", type=int, default=0)
    ap.add_argument("--backends", default="sglang", help="comma list: sglang,vllm")
    ap.add_argument("--plan-name", default="plan.json")
    ap.add_argument("--records", action="store_true", help="raw probe JSONs -> archive/records.jsonl")
    ap.add_argument("--snapshot-update", action="store_true", help="rebuild golden render snapshots")
    ap.add_argument("--snapshot-check", action="store_true", help="re-render goldens and diff vs snapshots")
    args = ap.parse_args()

    if args.records:
        build_records()
        return
    if args.snapshot_update or args.snapshot_check:
        raise SystemExit(snapshots_cmd(update=args.snapshot_update))
    if args.collect:
        collect()
        return
    if args.check_coverage:
        check_coverage(yaml.safe_load(args.targets.read_text()))
        return
    targets = yaml.safe_load(args.targets.read_text())
    runs = enumerate_runs(targets, args.full, args.backends.split(","))
    if args.plan:
        for r in runs:
            print(json.dumps(r))
        return
    if args.emit_queues:
        emit_queues(runs, args.gpus, args.gpu_offset, args.plan_name)


if __name__ == "__main__":
    main()
