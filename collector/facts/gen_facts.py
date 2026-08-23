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
import shlex
import sys
from pathlib import Path

import yaml

# workspace: where dummy_models/, archive/ and probe outputs live
ROOT = Path(os.environ.get("AIC_PROBE_WORKSPACE", Path.cwd()))
# generator source: this repo by default; override to pin a specific checkout
AIC_SRC = os.environ.get("AIC_GENERATOR_SRC",
                         str(Path(__file__).resolve().parents[2] / "src"))
if AIC_SRC not in sys.path:
    sys.path.insert(0, AIC_SRC)
WORK = "/work"  # container mount of ROOT
SCRATCH_QUEUES = ROOT / "archive" / "queues"


def model_config_from_dummy(model_dir_in_container: str) -> dict:
    """Derive ModelConfig facts from the variant's config.json instead of
    hardcoding (is_moe=True bit a review; dense targets would have gotten
    MoE-branch parallelism args). nextn is reported as 0 because the dummy
    generator zeroes MTP modules — a recorded decision, not an assumption."""
    cfg_path = ROOT / model_dir_in_container.replace(WORK + "/", "") / "config.json"
    cfg = json.loads(cfg_path.read_text())
    tc = cfg.get("text_config", cfg)
    experts = tc.get("n_routed_experts") or tc.get("num_local_experts") or 0
    return {"is_moe": bool(experts and experts > 1), "prefix": 0, "nextn": 0}


def render_sglang_cli(model_dir_in_container: str, tp: int, version: str) -> str:
    """Render the generator's cli_args_agg for one config — the deployment-true
    engine args are the probe input (zero translation drift)."""
    from aiconfigurator.generator.rendering.engine import render_backend_templates

    params = {
        "ServiceConfig": {"model_path": model_dir_in_container, "served_model_path": model_dir_in_container,
                          "served_model_name": "probe", "include_frontend": False},
        "K8sConfig": {"name_prefix": "probe", "k8s_namespace": "default", "k8s_image": "unused",
                      "k8s_pvc_name": "x", "k8s_pvc_mount_path": WORK, "k8s_model_path_in_pvc": "m",
                      "k8s_model_cache": "x", "k8s_hf_home": model_dir_in_container, "extra_env": []},
        "DynConfig": {"mode": "agg"},
        "WorkerConfig": {"agg_workers": 1, "agg_gpus_per_worker": tp, "prefill_workers": 0, "decode_workers": 0},
        "NodeConfig": {"system_name": "h200_sxm", "num_gpus_per_node": 8},
        "SlaConfig": {"isl": 1024, "osl": 256},
        "ModelConfig": model_config_from_dummy(model_dir_in_container),
        "BenchConfig": {},
        "params": {"agg": {"tensor_parallel_size": tp, "pipeline_parallel_size": 1, "data_parallel_size": 1,
                           "gpus_per_worker": tp, "max_batch_size": 64, "max_num_tokens": 4096,
                           "max_seq_len": 8192, "tokens_per_block": 64, "trust_remote_code": True,
                           "extra_cli_args": []}},
    }
    arts = render_backend_templates(params, "sglang", version=version, deployment_target="dynamo-python")
    return " ".join(arts["cli_args_agg"].split())


def render_trtllm_engine_yaml(run: dict) -> str:
    """dynamo.trtllm path: the generator renders extra_engine_args yaml; the
    probe feeds it to llmapi (same contract the deployment worker uses)."""
    from aiconfigurator.generator.rendering.engine import render_backend_templates

    model = run["model_dir"]
    params = {
        "ServiceConfig": {"model_path": model, "served_model_path": model,
                          "served_model_name": "probe", "include_frontend": False},
        "K8sConfig": {"name_prefix": "probe", "k8s_namespace": "default", "k8s_image": run["image"],
                      "k8s_pvc_name": "x", "k8s_pvc_mount_path": WORK, "k8s_model_path_in_pvc": "m",
                      "k8s_model_cache": "x", "k8s_hf_home": model, "extra_env": []},
        "DynConfig": {"mode": "agg"},
        "WorkerConfig": {"agg_workers": 1, "agg_gpus_per_worker": run["tp"],
                         "prefill_workers": 0, "decode_workers": 0},
        "NodeConfig": {"system_name": "h200_sxm", "num_gpus_per_node": 8},
        "SlaConfig": {"isl": 1024, "osl": 256},
        "ModelConfig": model_config_from_dummy(model),
        "BenchConfig": {},
        "params": {"agg": {"tensor_parallel_size": run["tp"], "pipeline_parallel_size": 1,
                           "data_parallel_size": 1, "gpus_per_worker": run["tp"],
                           "max_batch_size": 64, "max_num_tokens": 4096, "max_seq_len": 8192,
                           "tokens_per_block": 64, "trust_remote_code": True,
                           "extra_cli_args": []}},
    }
    if run.get("kvcache_quant_mode") == "fp8":
        # module_bridge.py:140 — deployment passes kvcache_quant_mode through
        params["params"]["agg"]["kv_cache_dtype"] = "fp8"
    for k, v in (run.get("render_overrides") or {}).items():
        params["params"]["agg"][k] = v
    arts = render_backend_templates(params, "trtllm", version=run["version"])
    key = next(k for k in arts if k.startswith("extra_engine_args"))
    return arts[key]


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
    stamp = gdir / "command.txt"
    if stamp.exists() and stamp.read_text().splitlines()[0] == cmd_txt:
        sub = next((d for d in gdir.iterdir() if d.is_dir()), None)
        if sub is not None:
            return sub  # cached golden for the identical command
    if gdir.exists():
        shutil.rmtree(gdir)
    gdir.mkdir(parents=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "aic" / "aic-core" / "src")
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=900)
    stamp.write_text(cmd_txt + f"\n# exit={r.returncode}\n")
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
    from derive_profile import derive_profile

    cases = Path(AIC_SRC).parent / "collector" / "cases" / "models"
    org = (r"(?:deepseek-ai|zai-org|moonshotai|nvidia|openai|meta-llama|"
           r"mistralai|google|Qwen|XiaomiMiMo|MiniMaxAI|sgl-project)")
    mentioned: set[str] = set()
    for f in cases.glob("*_cases.yaml"):
        mentioned.update(re.findall(rf"\b({org}/[\w.\-]+)", f.read_text()))
    inaccessible: set[str] = set()
    for inacc in (ROOT / "configs" / "inaccessible.json",
                  Path(__file__).parent / "configs" / "inaccessible.json"):
        if inacc.exists():
            inaccessible = set(json.loads(inacc.read_text()))
            break
    owned_elsewhere = {ck["repo"] for fname, f in targets["families"].items()
                      if not f.get("derive") for ck in f.get("checkpoints", [])}
    repos = sorted((mentioned - inaccessible - owned_elsewhere)
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
            raise SystemExit(f"derive_roster: no fetched config for {repo} — fetch it or add to inaccessible.json")
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
                pairing = targets["kv_pairing"][ck["profile"]]
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
                                "kv_cli": pairing["cli"], "kvcache_quant_mode": pairing["kvcache"],
                                "variant": variant, "backend": backend, "version": version,
                                "image": be["images"][version], "tp": topo["tp"],
                                "model_dir": f"{WORK}/{vdir.relative_to(ROOT)}",
                                "aic_registered": ck.get("aic_registered", False),
                                "render_overrides": ((ck.get("render_overrides") or {}).get(backend)
                                                     or (fam.get("render_overrides") or {}).get(backend) or {}),
                                "cli_extra_args": (list(be.get("cli_extra_args") or [])
                                                   + _cea((ck.get("cli_extra_args") or {}).get(backend)
                                                          or (fam.get("cli_extra_args") or {}).get(backend))),
                            })
    return runs


def render_vllm_run_sh(run: dict) -> str:
    """FPM path: generator renders the full run.sh; kv pairing enters via CLI."""
    from aiconfigurator.generator.rendering.engine import render_backend_templates

    model = run["model_dir"]
    extra = ["--benchmark-mode", "agg"]
    if run["kv_cli"]:
        extra += ["--kv-cache-dtype", run["kv_cli"]]
    params = {
        "ServiceConfig": {"model_path": model, "served_model_path": model,
                          "served_model_name": "probe", "include_frontend": False},
        "K8sConfig": {"name_prefix": "probe", "k8s_namespace": "default",
                      "k8s_image": run["image"], "k8s_pvc_name": "x", "k8s_pvc_mount_path": WORK,
                      "k8s_model_path_in_pvc": "m", "k8s_model_cache": "x", "k8s_hf_home": model,
                      "extra_env": []},
        "DynConfig": {"mode": "agg"},
        "WorkerConfig": {"agg_workers": 1, "agg_gpus_per_worker": run["tp"],
                         "prefill_workers": 0, "decode_workers": 0},
        "NodeConfig": {"system_name": "h200_sxm", "num_gpus_per_node": 8},
        "SlaConfig": {"isl": 1024, "osl": 256},
        "ModelConfig": model_config_from_dummy(model),
        "BenchConfig": {},
        "params": {"agg": {"tensor_parallel_size": run["tp"], "pipeline_parallel_size": 1,
                           "data_parallel_size": 1, "gpus_per_worker": run["tp"],
                           "max_batch_size": 64, "max_num_tokens": 4096, "max_seq_len": 8192,
                           "tokens_per_block": 64, "trust_remote_code": True,
                           "extra_cli_args": extra}},
    }
    for k, v in (run.get("render_overrides") or {}).items():
        if v is None:
            params["params"]["agg"].pop(k, None)  # omit -> framework default
        else:
            params["params"]["agg"][k] = v
    arts = render_backend_templates(params, "vllm", version=run["version"], deployment_target="fpm")
    # FPM V1 has preconditions (agg mode, no router/planner, single worker);
    # when they do not hold the generator SILENTLY falls back to the default
    # dynamo target, which emits run_0.sh instead of run.sh. Accept either —
    # both carry the engine command — and record which one we got.
    for key in ("run.sh", "run_0.sh"):
        if key in arts:
            run["render_artifact"] = key
            return arts[key]
    raise KeyError(f"no run script in rendered artifacts: {list(arts)}")


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
            kv = f"--kv-dtype {run['kv_cli']} " if run["kv_cli"] else ""
            cmd = (head + f"{run['image']} python3 {WORK}/probe/probe_sglang.py "
                   f"--model {run['model_dir']} --engine-cli {shlex.quote(cli)} {kv}--trace "
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
        mentioned.update(re.findall(rf"\b({org}/[\w.\-]+)", f.read_text()))
    for fam in targets["families"].values():
        if fam.get("derive") and "checkpoints" not in fam:
            fam["checkpoints"] = derive_roster_checkpoints(fam, targets)
    covered = {ck["repo"] for fam in targets["families"].values() for ck in fam["checkpoints"]}
    inaccessible = set()
    inacc = ROOT / "configs" / "inaccessible.json"
    if inacc.exists():
        inaccessible = set(json.loads(inacc.read_text()))
    missing = sorted(mentioned - covered - inaccessible)
    print(f"collector mentions {len(mentioned)} repos; targets cover {len(covered)}; "
          f"gated/inaccessible {len(mentioned & inaccessible)}")
    if missing:
        print("MISSING FROM TARGETS (coverage floor violated):")
        for r in missing:
            print("  ", r)
        raise SystemExit(1)
    print("coverage floor satisfied")


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
    args = ap.parse_args()

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
