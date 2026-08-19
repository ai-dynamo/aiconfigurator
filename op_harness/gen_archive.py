#!/usr/bin/env python3
"""Step-1 archive generator: targets.yaml -> generator-rendered engine args
-> probe runs -> facts archive with provenance.

Runs on the host (needs the aiconfigurator repo importable for rendering).

  PYTHONPATH=<aic>/src python3 probe/gen_archive.py --plan            # show runs
  PYTHONPATH=<aic>/src python3 probe/gen_archive.py --emit-queues     # write per-GPU queue scripts
  python3 probe/gen_archive.py --collect                              # raw JSONs -> archive.jsonl
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
                         str(Path(__file__).resolve().parents[1] / "src"))
if AIC_SRC not in sys.path:
    sys.path.insert(0, AIC_SRC)
WORK = "/work"  # container mount of ROOT
SCRATCH_QUEUES = ROOT / "archive" / "queues"


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
        "ModelConfig": {"is_moe": True, "prefix": 0, "nextn": 0},
        "BenchConfig": {},
        "params": {"agg": {"tensor_parallel_size": tp, "pipeline_parallel_size": 1, "data_parallel_size": 1,
                           "gpus_per_worker": tp, "max_batch_size": 64, "max_num_tokens": 4096,
                           "max_seq_len": 8192, "tokens_per_block": 64, "trust_remote_code": True,
                           "extra_cli_args": []}},
    }
    arts = render_backend_templates(params, "sglang", version=version, deployment_target="dynamo-python")
    return " ".join(arts["cli_args_agg"].split())


def enumerate_runs(targets: dict, full: bool, backends: list[str]) -> list[dict]:
    runs = []
    topos = [t for t in targets["topologies"] if t["evidence"] == "real" and (full or t["tp"] == 1)]
    for backend in backends:
        be = targets["backends"][backend]
        versions = be["versions"] if full else [be["versions"][-1]]
        for fam_name, fam in targets["families"].items():
            variants = fam.get("dummy_variants") or []
            if not variants:
                continue  # adapter pending (kimi_k3)
            override = (fam.get("variant_overrides") or {}).get(backend)
            use_variants = variants if full else [override or variants[0]]
            for ck in fam["checkpoints"]:
                pairing = targets["kv_pairing"][ck["profile"]]
                repo_tag = ck["repo"].split("/")[-1]
                for variant in use_variants:
                    vdir = ROOT / "dummy_models" / fam_name / f"{repo_tag}__{variant}"
                    if not vdir.exists():
                        runs.append({"skip": f"no dummy dir {vdir.name}", "repo": ck["repo"], "variant": variant})
                        continue
                    for version in versions:
                        for topo in topos:
                            rid = hashlib.sha1(
                                f"{ck['repo']}|{variant}|{backend}|{version}|{ck['profile']}|tp{topo['tp']}".encode()
                            ).hexdigest()[:12]
                            runs.append({
                                "id": rid, "family": fam_name, "repo": ck["repo"], "profile": ck["profile"],
                                "kv_cli": pairing["cli"], "kvcache_quant_mode": pairing["kvcache"],
                                "variant": variant, "backend": backend, "version": version,
                                "image": be["images"][version], "tp": topo["tp"],
                                "model_dir": f"{WORK}/dummy_models/{fam_name}/{repo_tag}__{variant}",
                                "aic_registered": ck.get("aic_registered", False),
                                "render_overrides": (fam.get("render_overrides") or {}).get(backend) or {},
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
        "ModelConfig": {"is_moe": True, "prefix": 0, "nextn": 0},
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
    return arts["run.sh"]


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
                f"--gpus '\"device={g}\"' --shm-size 16g -e HF_HUB_OFFLINE=1 -v {ROOT}:{WORK} ")
        if run["backend"] == "sglang":
            cli = render_sglang_cli(run["model_dir"], run["tp"], run["version"])
            run["engine_cli"] = cli
            kv = f"--kv-dtype {run['kv_cli']} " if run["kv_cli"] else ""
            cmd = (head + f"{run['image']} python3 {WORK}/probe/probe_runner.py "
                   f"--model {run['model_dir']} --engine-cli {shlex.quote(cli)} {kv}--trace "
                   f"--out {WORK}/archive/raw/{run['id']}.json 2>&1 | tail -1 ; }}")
        elif run["backend"] == "vllm":  # via the FPM adapter
            rsh = ROOT / "archive" / "run_sh" / f"{run['id']}.sh"
            rsh.write_text(render_vllm_run_sh(run))
            run["run_sh"] = str(rsh)
            cmd = (head + f"--entrypoint python3 {run['image']} {WORK}/probe/fpm_adapter.py "
                   f"--run-sh {WORK}/archive/run_sh/{run['id']}.sh --model-override {run['model_dir']} "
                   f"--trace --out {WORK}/archive/raw/{run['id']}.json 2>&1 | tail -1 ; }}")
        else:  # trtllm: probe-default engine args (generator fidelity pending)
            run["engine_args_fidelity"] = "probe-defaults"
            trc = "--trust-remote-code " if run["family"] in ("m3", "kimi_k3") else ""
            cmd = (head.replace("docker run --rm ",
                                "docker run --rm -e TLLM_WORKER_USE_SINGLE_PROCESS=1 ")
                   + f"{run['image']} bash -lc 'python3 {WORK}/probe/trtllm_probe.py "
                   f"--model {run['model_dir']} {trc}--out {WORK}/archive/raw/{run['id']}.json' "
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=Path, default=Path(__file__).parent / "targets.yaml")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--emit-queues", action="store_true")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--full", action="store_true", help="all variants x both sglang versions (default: representative)")
    ap.add_argument("--gpus", type=int, default=4)
    ap.add_argument("--gpu-offset", type=int, default=0)
    ap.add_argument("--backends", default="sglang", help="comma list: sglang,vllm")
    ap.add_argument("--plan-name", default="plan.json")
    args = ap.parse_args()

    if args.collect:
        collect()
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
