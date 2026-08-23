#!/usr/bin/env python3
"""FPM run.sh -> framework parser -> identity probe (vLLM path).

Bridges aiconfigurator's FPM artifacts to the probe: the engine command the
generator rendered IS the probe input — parsed by vLLM's own CLI parser so
there is zero translation drift between "what a deployment runs" and "what
the probe runs". The only mutations: model_path may be swapped to a dummy
variant, load_format forced to dummy, eager mode forced (identity probe).

Runs INSIDE the vllm image.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import traceback
from collections import Counter, defaultdict

# flags owned by dynamo/FPM orchestration, not vLLM engine args
_NON_ENGINE_FLAGS_WITH_VALUE = {
    "--benchmark-mode", "--dump-config-to", "--benchmark-output-path",
    "--nnodes", "--node-rank", "--master-addr", "--master-port",
    "--data-parallel-size-local", "--data-parallel-start-rank",
    "--data-parallel-address", "--data-parallel-rpc-port",
}
_NON_ENGINE_FLAGS_BARE = {"--headless", "--data-parallel-hybrid-lb"}


def parse_run_sh(path: str) -> tuple[list[str], dict[str, str]]:
    """Extract the engine argv (launcher + FPM flags stripped) and exported env."""
    text = open(path).read()
    m = re.search(r"engine_command=\((.*?)\)\s*$", text, re.M | re.S)
    if m:
        argv = shlex.split(m.group(1))
    else:
        # dynamo target (generator falls back to it when FPM preconditions
        # do not hold): a multi-line `python3 -m dynamo.vllm \` invocation
        # with shell variables. Take that block and resolve $MODEL_PATH.
        # take ONLY the backslash-continued invocation, not the shell plumbing
        # that follows it (pipes, subshell/loop tails)
        blk = re.search(r"python3 -m dynamo\.vllm((?:[^\n]*\\\n)*[^\n]*)", text)
        if not blk:
            raise ValueError(f"no engine command found in {path}")
        line = blk.group(1).replace("\\\n", " ")
        line = re.split(r"\s(?:2>&1|\||&|;|\))", line)[0]
        model = re.search(r'^export MODEL_PATH=\$\{MODEL_PATH:-"([^"]+)"\}', text, re.M)
        line = line.replace('"$MODEL_PATH"', model.group(1) if model else "")
        line = re.sub(r'"?\$\{?[A-Z_]+\}?"?', "", line)  # drop unresolved vars
        argv = shlex.split(line)
    # strip launcher prefix: python3 -m dynamo.vllm / vllm serve ...
    while argv and not argv[0].startswith("--"):
        argv.pop(0)
    out = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in _NON_ENGINE_FLAGS_BARE:
            i += 1
        elif a in _NON_ENGINE_FLAGS_WITH_VALUE:
            i += 2
        else:
            out.append(a)
            i += 1
    env = dict(re.findall(r"^export ([A-Za-z_][A-Za-z0-9_]*)=(\S+)$", text, re.M))
    return out, env


def find_torch_model(root, max_depth: int = 8):
    """BFS the object graph for the biggest nn.Module — version-agnostic."""
    import torch.nn as nn

    seen, queue, best = set(), [(root, 0)], None
    while queue:
        obj, d = queue.pop(0)
        if id(obj) in seen or d > max_depth:
            continue
        seen.add(id(obj))
        if isinstance(obj, nn.Module):
            n = sum(1 for _ in obj.parameters(recurse=True))
            if best is None or n > best[1]:
                best = (obj, n)
            continue  # don't descend into modules
        for name in dir(obj):
            if name.startswith("__"):
                continue
            try:
                child = getattr(obj, name)
            except Exception:
                continue
            if isinstance(child, nn.Module):
                queue.append((child, d + 1))  # Modules are callable — check first
            elif callable(child) or isinstance(child, (str, int, float, bool, bytes)):
                continue
            elif isinstance(child, (list, tuple)) and len(child) < 32:
                queue.extend((c, d + 1) for c in child)
            elif isinstance(child, dict) and len(child) < 32:
                queue.extend((c, d + 1) for c in child.values())
            elif not isinstance(child, set):
                queue.append((child, d + 1))
    return best[0] if best else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-sh", required=True)
    ap.add_argument("--model-override", default=None, help="swap --model to this dummy variant dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--py-paths", action="store_true")
    args = ap.parse_args()

    rec: dict = {"run_sh": args.run_sh, "errors": {}}
    try:
        import torch
        rec["device_capability"] = "sm%d%d" % torch.cuda.get_device_capability()
    except Exception:
        rec["device_capability"] = None
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")  # keep EngineCore in-process

    argv, sh_env = parse_run_sh(args.run_sh)
    for k, v in sh_env.items():  # generator-owned env is part of the contract
        if k not in os.environ and not k.startswith(("HF_", "DYN_FPM", "FPM_")):
            os.environ[k] = v
    if args.model_override:
        i = argv.index("--model")
        argv[i + 1] = args.model_override
    rec["engine_argv"] = argv

    import vllm

    rec["vllm_version"] = vllm.__version__

    try:
        from vllm.engine.arg_utils import EngineArgs
        try:
            from vllm.utils import FlexibleArgumentParser
        except ImportError:  # 0.24 moved utils into submodules
            from vllm.utils.argparse_utils import FlexibleArgumentParser
        parser = FlexibleArgumentParser()
        EngineArgs.add_cli_args(parser)
        ns = parser.parse_args(argv)
        ea = EngineArgs.from_cli_args(ns)
        ea.load_format = "dummy"
        ea.enforce_eager = True  # identity probe: no graph capture
        rec["engine_args_resolved"] = {
            k: v for k, v in vars(ea).items()
            if isinstance(v, (str, int, float, bool, type(None)))
            and any(s in k for s in ("quant", "dtype", "parallel", "block", "model",
                                     "attention", "kv", "moe", "backend", "eager", "load"))
        }
    except Exception:
        rec["errors"]["parse"] = traceback.format_exc()
        json.dump(rec, open(args.out, "w"), indent=1, default=str)
        return

    try:
        try:
            from vllm.v1.engine.llm_engine import LLMEngine
        except ImportError:
            from vllm import LLMEngine
        engine = LLMEngine.from_engine_args(ea)
        model = find_torch_model(engine)
        if model is None:
            raise RuntimeError("no nn.Module found via object-graph search")
        rec["model_class"] = type(model).__qualname__

        qm = defaultdict(list)
        for name, mod in model.named_modules():
            q = getattr(mod, "quant_method", None)
            if q is not None:
                qm[f"{type(q).__module__}.{type(q).__name__}"].append(name)
        rec["quant_methods"] = {k: {"count": len(v), "modules": v[:6]} for k, v in qm.items()}
        rec["param_dtypes"] = dict(Counter(str(p.dtype) for p in model.parameters()))
        samples = {}
        for name, p in model.named_parameters():
            for key in ("experts", "qkv", "kv_b", "o_proj", "gate_up", "down_proj", "indexer"):
                if key in name and key not in {s.split("::")[0] for s in samples}:
                    samples[f"{key}::{name}"] = f"{p.dtype} {tuple(p.shape)}"
        rec["weight_samples"] = samples
    except Exception:
        rec["errors"]["load"] = traceback.format_exc()

    if args.trace and not rec["errors"]:
        try:
            import torch
            from torch.profiler import ProfilerActivity, profile, record_function

            def wrap_span(cls, meth, label_fn):
                orig = getattr(cls, meth)
                if getattr(orig, "_aic_wrapped", False):
                    return

                def wrapped(self, *a, _o=orig, **k):
                    with record_function(label_fn(self)):
                        return _o(self, *a, **k)

                wrapped._aic_wrapped = True
                setattr(cls, meth, wrapped)

            for _n, m in model.named_modules():
                q = getattr(m, "quant_method", None)
                if q is not None and hasattr(type(q), "apply"):
                    wrap_span(type(q), "apply", lambda s: f"AIC::quant_apply::{type(s).__name__}")
            try:  # attention boundaries: scan the LOADED model for attention-ish
                # module classes (Attention/MLA/Mixer/linear-attn) — generic across
                # model families, no hardcoded module-path list to maintain
                import torch.nn as _nn
                wrapped = set()
                for _n, m in model.named_modules():
                    t = type(m)
                    if t in wrapped:
                        continue
                    if any(k in t.__name__ for k in ("Attention", "Attn", "MLA", "Mixer", "SSM", "Compressor", "Indexer")):
                        # forward may be inherited (DSV4 classes) — wrap the
                        # class in the MRO that actually defines it; the span
                        # label reads the runtime type, so sharing a base is fine
                        holder = next((c for c in t.__mro__
                                       if "forward" in vars(c) and c is not _nn.Module), None)
                        if holder is None or holder in wrapped:
                            continue
                        wrap_span(holder, "forward",
                                  lambda s: f"AIC::attn::{type(getattr(s, 'impl', s)).__name__}")
                        wrapped.add(t)
                        wrapped.add(holder)
                rec["attn_classes_wrapped"] = sorted(t.__name__ for t in wrapped)
                if not wrapped:
                    rec["errors"]["attn_hook"] = "no attention-ish module classes found in model"
            except Exception as e:
                rec["errors"]["attn_hook"] = f"{type(e).__name__}: {e}"[:200]

            from vllm import SamplingParams
            prompt = {"prompt_token_ids": list(range(32))}
            engine.add_request("probe0", prompt, SamplingParams(max_tokens=2, temperature=0))
            engine.step()  # prefill warmup for lazy JIT / autotune
            try:
                _exp = torch._C._profiler._ExperimentalConfig(verbose=True)
            except Exception:
                _exp = None
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                         with_stack=True,
                         **({"experimental_config": _exp} if _exp else {})) as p:
                while engine.has_unfinished_requests():
                    engine.step()
                torch.cuda.synchronize()

            def fw_frames(ev):
                frames = []
                for fr in getattr(ev, "stack", None) or []:
                    if "/vllm/" in fr or "flash" in fr or "triton" in fr or "marlin" in fr:
                        frames.append(fr.split("site-packages/")[-1])
                return tuple(frames[:6])

            spans: dict = {}

            def collect(ev, acc, paths, seen):
                if id(ev) in seen:
                    return
                seen.add(id(ev))
                kerns = getattr(ev, "kernels", None) or []
                for kern in kerns:
                    a = acc.setdefault(kern.name, {"us": 0.0, "launches": 0})
                    a["us"] += kern.duration
                    a["launches"] += 1
                if kerns:
                    key = (ev.name, fw_frames(ev))
                    pth = paths.setdefault(key, {"kernels": set(), "launches": 0})
                    pth["kernels"].update(k.name.split("(")[0][:60] for k in kerns)
                    pth["launches"] += len(kerns)
                for c in getattr(ev, "cpu_children", None) or []:
                    collect(c, acc, paths, seen)

            for ev in p.profiler.function_events:
                if ev.name.startswith("AIC::"):
                    slot = spans.setdefault(ev.name, {"calls": 0, "kernels": {}, "py_paths": {}})
                    slot["calls"] += 1
                    acc: dict = {}
                    paths: dict = {}
                    collect(ev, acc, paths, set())
                    for n, agg in acc.items():
                        k = slot["kernels"].setdefault(n, {"us": 0.0, "launches": 0})
                        k["us"] += agg["us"]
                        k["launches"] += agg["launches"]
                    for (opname, frames), pth in paths.items():
                        key = opname + (" <- " + " <- ".join(frames) if frames else "")
                        s = slot["py_paths"].setdefault(key, {"kernels": set(), "launches": 0})
                        s["kernels"].update(pth["kernels"])
                        s["launches"] += pth["launches"]
            for slot in spans.values():
                slot["kernels"] = dict(sorted(slot["kernels"].items(), key=lambda kv: -kv[1]["us"])[:10])
                slot["py_paths"] = {k: {"kernels": sorted(v["kernels"])[:6], "launches": v["launches"]}
                                    for k, v in list(slot["py_paths"].items())[:10]}
            rec["api_trace"] = spans
        except Exception:
            rec["errors"]["trace"] = traceback.format_exc()

    json.dump(rec, open(args.out, "w"), indent=1, default=str)
    print("WROTE", args.out, "errors:", list(rec["errors"]))


if __name__ == "__main__":
    main()
