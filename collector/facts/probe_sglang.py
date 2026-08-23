#!/usr/bin/env python3
"""Backend-identity probe. Runs INSIDE the sglang image against a dummy model dir.

Stage 1 (no GPU work): ServerArgs.__post_init__ resolution — every
backend/quant-related field sglang derives for this model on this hardware.
Stage 2 (GPU): real dummy-weight load via sglang's own loader, then introspect
per-module quant_method classes, weight dtypes, and the attention backend
actually instantiated.

Output is one JSON record; errors are recorded as facts, never swallowed.
"""

from __future__ import annotations

import argparse
import json
import traceback
from collections import Counter, defaultdict

INTERESTING = ("backend", "quant", "page", "kv_cache", "attention", "moe", "mamba", "dsa_", "gemm")


def dump_server_args(sa) -> dict:
    out = {}
    for f in type(sa).__dataclass_fields__:
        if any(k in f for k in INTERESTING):
            v = getattr(sa, f)
            if isinstance(v, (str, int, float, bool, type(None))) or (
                isinstance(v, list) and all(isinstance(x, (str, int, float)) for x in v)
            ):
                out[f] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", type=int, default=2)
    ap.add_argument("--override", default=None, help="json_model_override_args, e.g. '{\"expert_dtype\": \"fp8\"}'")
    ap.add_argument("--quantization", default=None, help="explicit ServerArgs.quantization (collector sets fp8 for dsv4)")
    ap.add_argument("--kv-dtype", default=None, help="explicit kv_cache_dtype (generator passes fp8_e4m3 for fp8 profiles)")
    ap.add_argument("--engine-cli", default=None,
                    help="generator-rendered sglang CLI args (cli_args_agg); parsed by sglang's own parser")
    ap.add_argument("--run-forward", action="store_true",
                    help="keep cuda-graph capture ON so a real decode forward executes (execution check, not just load)")
    ap.add_argument("--trace", action="store_true",
                    help="run one eager prefill + decode under torch.profiler; record kernel names and MoE dispatch")
    ap.add_argument("--py-paths", action="store_true",
                    help="capture Python call stacks per kernel (verbose kineto; "
                    "slower and it has broken tvm_ffi kernels); kernels+spans "
                    "are captured either way")
    args = ap.parse_args()

    rec: dict = {"model_path": args.model, "errors": {}}
    try:
        import torch
        rec["device_capability"] = "sm%d%d" % torch.cuda.get_device_capability()
    except Exception:
        rec["device_capability"] = None
    import sglang

    rec["sglang_version"] = sglang.__version__

    from sglang.srt.server_args import PortArgs, ServerArgs

    # cuda-graph disable flags MUST be passed at construction: __post_init__
    # derives capture state from them, so post-hoc assignment is ignored
    # (verified: DSV4 decode-graph capture ran with all flags set post-hoc).
    graph_off = {} if args.run_forward else {
        f: True for f in ServerArgs.__dataclass_fields__
        if "cuda_graph" in f and f.startswith("disable")
    }
    if args.engine_cli:
        # generator-faithful path: the rendered CLI is the probe input, parsed
        # by sglang's own parser; probe overrides appended as CLI flags too.
        import shlex

        argv = shlex.split(args.engine_cli)
        argv += ["--model-path", args.model, "--load-format", "dummy",
                 "--trust-remote-code", "--disable-radix-cache",
                 "--max-total-tokens", "16384", "--max-running-requests", "32"]
        argv += [f"--{f.replace('_', '-')}" for f in graph_off]
        if args.kv_dtype:
            argv += ["--kv-cache-dtype", args.kv_dtype]
        if args.quantization:
            argv += ["--quantization", args.quantization]
        if args.override:
            argv += ["--json-model-override-args", args.override]
        cli_parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(cli_parser)
        ns, unknown = cli_parser.parse_known_args(argv)
        rec["engine_cli"] = args.engine_cli
        rec["engine_cli_unknown_args"] = unknown  # generator flags this version doesn't know = drift facts
        sa = ServerArgs.from_cli_args(ns)
    else:
        sa = ServerArgs(
            model_path=args.model,
            load_format="dummy",
            trust_remote_code=True,
            tp_size=1,  # stage2 is single-process; tp>1 goes launch_server+inject
            disable_radix_cache=True,
            max_running_requests=32,
            # identity probe: cap the KV pool so tiny dummy models don't let
            # sglang size max_total_tokens to the whole GPU (DSV4 OOMed at 138GB)
            max_total_tokens=16384,
            **graph_off,
            **({"json_model_override_args": args.override} if args.override else {}),
            **({"quantization": args.quantization} if args.quantization else {}),
            **({"kv_cache_dtype": args.kv_dtype} if args.kv_dtype else {}),
        )
    rec["cuda_graph_fields_disabled"] = sorted(graph_off)
    if args.override:
        rec["json_model_override_args"] = args.override
    rec["server_args_resolved"] = dump_server_args(sa)

    from sglang.srt.configs.model_config import ModelConfig

    try:
        mc = ModelConfig.from_server_args(sa)
        hf_qc = getattr(mc.hf_config, "quantization_config", None)
        rec["model_config"] = {
            "architectures": getattr(mc.hf_config, "architectures", None),
            "hf_quant_method": hf_qc.get("quant_method") if isinstance(hf_qc, dict) else str(type(hf_qc).__name__),
            "resolved_quantization": getattr(mc, "quantization", None),
            "attention_arch": str(getattr(mc, "attention_arch", "<absent>")),
            "num_hidden_layers": getattr(mc.hf_config, "num_hidden_layers", None),
        }
    except Exception:
        rec["errors"]["model_config"] = traceback.format_exc()[-1500:]

    if args.stage >= 2 and not rec["errors"]:
        try:
            from sglang.bench_one_batch import load_model
            # Replicate serving's global initialization sequence (scheduler.py
            # does exactly this before building the model). Skipping it made the
            # probe take a DIFFERENT dispatch path than a real deployment:
            # MoE runner overrides live in process-global flags, not ServerArgs,
            # so DSV4-NVFP4 silently fell back to marlin here while serving
            # rejects the config outright.
            from sglang.srt.layers.moe import initialize_moe_config
            initialize_moe_config(sa)
            # scheduler.py:726 — mamba/linear-attention models need this before
            # any forward; bench_one_batch does NOT call it, so hybrid models
            # fail there in ways real serving does not.
            try:
                from sglang.srt.managers.scheduler import (
                    initialize_mamba_selective_state_update_backend,
                )
                initialize_mamba_selective_state_update_backend(sa)
            except Exception as _e:
                rec.setdefault("init_warnings", []).append(
                    f"mamba_ssu_backend: {type(_e).__name__}")
            for _fn in ("initialize_fp8_gemm_config", "initialize_fp4_gemm_config"):
                try:
                    getattr(__import__("sglang.benchmark.one_batch", fromlist=[_fn]), _fn)(sa)
                except Exception as _e:
                    rec.setdefault("init_warnings", []).append(f"{_fn}: {type(_e).__name__}")
            rec["serving_init_applied"] = True
            try:
                from sglang.srt.layers.moe.utils import get_moe_runner_backend
                rec["effective_moe_runner_backend"] = str(get_moe_runner_backend())
            except Exception:
                pass


            ret, _tok = load_model(sa, PortArgs.init_new(sa), 0, 0)
            # 0.5.16 wraps ModelRunner in _TorchBenchRunner(.torch_runner)
            model_runner = getattr(ret, "torch_runner", ret)
            model = model_runner.model

            qm_classes = defaultdict(list)
            for name, mod in model.named_modules():
                qm = getattr(mod, "quant_method", None)
                if qm is not None:
                    qm_classes[f"{type(qm).__module__}.{type(qm).__name__}"].append(name)
            rec["quant_methods"] = {k: {"count": len(v), "modules": v} for k, v in qm_classes.items()}
            rec["param_dtypes"] = dict(Counter(str(p.dtype) for p in model.parameters()))

            # ground truth for kv dtype: what the KV pool actually allocates
            # with ('auto' in server args resolves here, not in ServerArgs)
            kvres = {"server_arg": str(getattr(sa, "kv_cache_dtype", None))}
            rd = getattr(model_runner, "kv_cache_dtype", None)
            if rd is not None:
                kvres["runner_kv_cache_dtype"] = str(rd)
            pool = getattr(model_runner, "token_to_kv_pool", None)
            if pool is not None:
                kvres["pool_class"] = type(pool).__name__
                for a in ("dtype", "kv_cache_dtype", "store_dtype"):
                    v = getattr(pool, a, None)
                    if v is not None:
                        kvres[f"pool.{a}"] = str(v)
            rec["kv_cache_resolved"] = kvres

            samples = {}
            for name, p in model.named_parameters():
                for key in ("experts", "q_proj", "kv_b_proj", "o_proj", "qkv_proj",
                            "gate_up", "down_proj", "indexer", "shared_expert"):
                    if key in name and key not in {s.split("::")[0] for s in samples}:
                        samples[f"{key}::{name}"] = f"{p.dtype} {tuple(p.shape)}"
            rec["weight_samples"] = samples

            attn = getattr(model_runner, "attn_backend", None)
            if attn is None:
                for m in ("alloc_memory_pool", "init_attention_backends"):
                    try:
                        getattr(model_runner, m)()
                        rec.setdefault("init_calls_succeeded", []).append(m)
                    except Exception as e:  # record, keep going — absence is a fact too
                        rec["errors"][m] = f"{type(e).__name__}: {e}"[:400]
                attn = getattr(model_runner, "attn_backend", None)
            rec["attn_backend"] = f"{type(attn).__module__}.{type(attn).__name__}" if attn is not None else None
            for attr in ("attn_backends", "decode_attn_backend", "prefill_attn_backend",
                         "prefill_attention_backend_str", "decode_attention_backend_str"):
                v = getattr(model_runner, attr, None)
                if v is not None:
                    rec[f"model_runner.{attr}"] = str(v)[:300]
        except Exception:
            rec["errors"]["stage2"] = traceback.format_exc()

    if args.trace and "stage2" not in rec["errors"]:
        trace: dict = {}
        from torch.profiler import record_function

        def wrap_span(cls, meth: str, label_fn) -> None:
            """Wrap cls.meth in a profiler span named by label_fn(self) — the
            API-boundary layer: one span per collector-op-granularity call."""
            orig = getattr(cls, meth)
            if getattr(orig, "_aic_wrapped", False):
                return

            def wrapped(self, *a, **k):
                with record_function(label_fn(self)):
                    return orig(self, *a, **k)

            wrapped._aic_wrapped = True
            setattr(cls, meth, wrapped)

        try:  # B_api boundaries, collector-op granularity
            from sglang.srt.layers.moe.moe_runner.runner import MoeRunner

            wrap_span(MoeRunner, "run", lambda s: "AIC::moe::"
                      + getattr(getattr(s, "fused_func", None), "__qualname__", "?"))
        except Exception as e:
            rec["errors"]["moe_hook"] = f"{type(e).__name__}: {e}"
        try:
            ab_cls = type(model_runner.attn_backend)
            # cover dispatch wrappers too: models may enter via forward()/forward_mixed()
            # rather than calling forward_extend/forward_decode directly
            for meth in ("forward", "forward_extend", "forward_decode",
                         "forward_mixed", "forward_unified"):
                if meth in vars(ab_cls) or any(meth in vars(b) for b in ab_cls.__mro__[1:]):
                    wrap_span(ab_cls, meth,
                              lambda s, m=meth: f"AIC::attn::{type(s).__name__}.{m}")
        except Exception as e:
            rec["errors"]["attn_hook"] = f"{type(e).__name__}: {e}"
        try:  # every quant-method class seen in the model gets an apply span
            seen_qm = {type(getattr(mod, "quant_method"))
                       for _n, mod in model.named_modules() if getattr(mod, "quant_method", None) is not None}
            for qm_cls in seen_qm:
                if hasattr(qm_cls, "apply"):
                    wrap_span(qm_cls, "apply", lambda s: f"AIC::quant_apply::{type(s).__name__}")
        except Exception as e:
            rec["errors"]["quant_hook"] = f"{type(e).__name__}: {e}"

        def fw_frames(ev) -> tuple:
            """Python frames of a launching op, filtered to framework code."""
            frames = []
            for fr in getattr(ev, "stack", None) or []:
                if ("/sglang/" in fr or "/vllm/" in fr or "flash" in fr or
                        "deep_gemm" in fr or "flashinfer" in fr or "triton" in fr):
                    fr = fr.split("site-packages/")[-1].split("sgl-workspace/sglang/python/")[-1]
                    frames.append(fr)
            return tuple(frames[:6])

        def api_kernel_map(prof) -> dict:
            """Group kernels under their enclosing AIC:: span via the event tree,
            with the Python call path (framework frames) of each launching op."""
            spans: dict = {}

            def collect(ev, acc, paths, seen):
                if id(ev) in seen:  # cpu_children can repeat shared events
                    return
                seen.add(id(ev))
                kerns = getattr(ev, "kernels", None) or []
                for kern in kerns:
                    a = acc.setdefault(kern.name, {"us": 0.0, "launches": 0})
                    a["us"] += kern.duration
                    a["launches"] += 1
                if kerns:
                    key = (ev.name, fw_frames(ev))
                    p = paths.setdefault(key, {"kernels": set(), "launches": 0})
                    p["kernels"].update(k.name.split("(")[0][:60] for k in kerns)
                    p["launches"] += len(kerns)
                for child in getattr(ev, "cpu_children", None) or []:
                    collect(child, acc, paths, seen)

            for ev in prof.profiler.function_events:
                if not ev.name.startswith("AIC::"):
                    continue
                slot = spans.setdefault(ev.name, {"calls": 0, "kernels": {}, "py_paths": {}})
                slot["calls"] += 1
                acc: dict = {}
                paths: dict = {}
                collect(ev, acc, paths, set())
                for name, agg in acc.items():
                    k = slot["kernels"].setdefault(name, {"us": 0.0, "launches": 0})
                    k["us"] += agg["us"]
                    k["launches"] += agg["launches"]
                for (opname, frames), p in paths.items():
                    key = opname + (" <- " + " <- ".join(frames) if frames else "")
                    slot["py_paths"].setdefault(key, {"kernels": set(), "launches": 0})
                    slot["py_paths"][key]["kernels"].update(p["kernels"])
                    slot["py_paths"][key]["launches"] += p["launches"]
            for slot in spans.values():  # keep the top kernels per span
                slot["kernels"] = dict(sorted(slot["kernels"].items(),
                                              key=lambda kv: -kv[1]["us"])[:12])
                slot["py_paths"] = {k: {"kernels": sorted(v["kernels"])[:6], "launches": v["launches"]}
                                    for k, v in list(slot["py_paths"].items())[:12]}
            return spans

        def kernel_table(prof) -> list[dict]:
            rows = []
            for e in prof.key_averages():
                dt = getattr(e, "self_device_time_total", 0) or getattr(e, "self_cuda_time_total", 0)
                if dt > 0:
                    rows.append({"kernel": e.key, "calls": e.count, "us": round(dt, 1)})
            return sorted(rows, key=lambda r: -r["us"])[:50]

        try:
            import torch
            from torch.profiler import ProfilerActivity, profile

            from sglang.benchmark import one_batch as ob

            # warmup pass outside the profiler: lazy JIT loading / autotune noise
            try:
                out = ret.extend(ob.prepare_synthetic_inputs_for_latency_test(2, 32))
                torch.cuda.synchronize()
            except Exception:
                # transient tvm_ffi 'Mismatched Tensor' seen when parallel runs
                # race the shared flashinfer cutlass-DSL JIT cache — one retry
                # after the cache settles distinguishes flake from real fact
                rec["warmup_retry"] = traceback.format_exc().strip().splitlines()[-1][:200]
                import time
                time.sleep(5)
                out = ret.extend(ob.prepare_synthetic_inputs_for_latency_test(2, 32))
                torch.cuda.synchronize()
            ret.clear()
            try:  # source-line attribution for stacks needs the verbose kineto config
                _exp = torch._C._profiler._ExperimentalConfig(verbose=True)
            except Exception:
                _exp = None
            _prof_kw = dict(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                            **({"with_stack": True, "experimental_config": _exp}
                               if args.py_paths and _exp else {}))
            reqs = ob.prepare_synthetic_inputs_for_latency_test(2, 32)
            try:
                with profile(**_prof_kw) as p1:
                    out = ret.extend(reqs)
                    torch.cuda.synchronize()
            except Exception:
                # the verbose stack profiler breaks some tvm_ffi-dispatched
                # kernels (flashinfer cutlass-DSL fused_add_rmsnorm rejects its
                # tensors under with_stack=True — seen on MiniMax-M2/M2.5).
                # Retry stackless: kernels still captured, py_paths lost.
                rec["trace_stack_fallback"] = traceback.format_exc().strip().splitlines()[-1][:200]
                _prof_kw = dict(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
                reqs = ob.prepare_synthetic_inputs_for_latency_test(2, 32)
                with profile(**_prof_kw) as p1:
                    out = ret.extend(reqs)
                    torch.cuda.synchronize()
            trace["prefill_kernels"] = kernel_table(p1)
            trace["prefill_api"] = api_kernel_map(p1)
            try:
                next_ids, batch = out[0], out[-1]
                with profile(**_prof_kw) as p2:
                    ret.decode(next_ids, batch)
                    torch.cuda.synchronize()
                trace["decode_kernels"] = kernel_table(p2)
                trace["decode_api"] = api_kernel_map(p2)
            except Exception:
                rec["errors"]["trace_decode"] = traceback.format_exc()[-1500:]
        except Exception:
            rec["errors"]["trace"] = traceback.format_exc()
        rec["trace"] = trace

    with open(args.out, "w") as f:
        json.dump(rec, f, indent=1, default=str)
    print("WROTE", args.out, "errors:", list(rec["errors"]))


if __name__ == "__main__":
    main()
