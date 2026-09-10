"""Rank-side probe injection (minimal, for multi-rank runs).

Activated by AIC_PROBE_OUT=<path-prefix>. Loaded automatically in EVERY python
process (including mp-spawned ranks) via PYTHONPATH. Hooks sglang after model
load, dumps per-rank identity facts + per-call MoE/attention kernel captures
to <prefix>.rank<N>.json.
"""

from __future__ import annotations

import json
import os

_OUT = os.environ.get("AIC_PROBE_OUT")

if _OUT:
    import sys as _sys
    print(f"[AIC-inject] armed pid={os.getpid()} out={_OUT}", file=_sys.stderr)

    def _install() -> None:
        import sglang.srt.model_executor.model_runner as mr_mod

        state: dict = {"moe_calls": [], "attn_calls": [], "identity": {}}

        def _dump(rank: int) -> None:
            # atomic replace: concurrent dumpers (e.g. dp/ep attn workers that
            # share a tp_rank) must never interleave writes in one file
            import os
            tmp = f"{_OUT}.rank{rank}.json.{os.getpid()}.tmp"
            with open(tmp, "w") as f:
                json.dump(state, f, indent=1, default=str)
            os.replace(tmp, f"{_OUT}.rank{rank}.json")

        def _norm(name: str) -> str:
            return name.split("(")[0][:90]

        def _capture(orig, bucket: str, label_fn, max_calls: int = 3):
            counter = {"n": 0}

            def wrapped(self, *a, **k):
                if counter["n"] >= max_calls:
                    return orig(self, *a, **k)
                counter["n"] += 1
                import torch
                from torch.profiler import ProfilerActivity, profile

                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
                    out = orig(self, *a, **k)
                    torch.cuda.synchronize()
                kernels = sorted({
                    _norm(e.key) for e in p.key_averages()
                    if (getattr(e, "self_device_time_total", 0)
                        or getattr(e, "self_cuda_time_total", 0)) > 0
                    and not e.key.startswith(("aten::", "Memcpy", "Memset"))
                })
                state[bucket].append({"label": label_fn(self), "kernels": kernels})
                _dump(state["identity"].get("tp_rank", 0))
                return out

            return wrapped

        orig_init_attn = mr_mod.ModelRunner.init_attention_backends

        def init_attn(self, *a, **k):
            out = orig_init_attn(self, *a, **k)
            try:
                ident = state["identity"]
                ident["tp_rank"] = getattr(self, "tp_rank", 0)
                sa = self.server_args
                ident["server_args"] = {
                    key: getattr(sa, key, None)
                    for key in ("tp_size", "ep_size", "dp_size", "enable_dp_attention",
                                "attention_backend", "kv_cache_dtype", "moe_a2a_backend",
                                "moe_runner_backend", "quantization")}
                ident["attn_backend"] = type(self.attn_backend).__name__ if getattr(
                    self, "attn_backend", None) else None
                mods: dict = {}
                weights: dict = {}
                for name, mod in self.model.named_modules():
                    q = getattr(mod, "quant_method", None)
                    if q is not None:
                        mods.setdefault(type(q).__name__, []).append(name)
                for name, prm in self.model.named_parameters():
                    if "w13" in name or "experts" in name:
                        weights[name] = f"{prm.dtype} {tuple(prm.shape)}"
                        if len(weights) >= 4:
                            break
                ident["quant_methods"] = mods
                ident["expert_weights"] = weights

                from sglang.srt.layers.moe.moe_runner.runner import MoeRunner
                if not getattr(MoeRunner.run, "_aic", False):
                    MoeRunner.run = _capture(
                        MoeRunner.run, "moe_calls",
                        lambda s: getattr(getattr(s, "fused_func", None), "__qualname__", "?"))
                    MoeRunner.run._aic = True
                # quant methods that bypass MoeRunner (compressed-tensors ->
                # marlin, flashinfer trtllm, ...) are all reached through the
                # FusedMoE module — wrap it too, labeled by quant method class
                try:
                    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
                    if not getattr(FusedMoE.forward, "_aic", False):
                        FusedMoE.forward = _capture(
                            FusedMoE.forward, "moe_calls",
                            lambda s: f"FusedMoE[{type(getattr(s, 'quant_method', None)).__name__}]")
                        FusedMoE.forward._aic = True
                except Exception:
                    pass
                # instance-driven wrap: entry points move between releases
                # (deepep/TBO dispatchers bypassed the class imports above),
                # so hook the ACTUAL expert-module classes found in the model
                seen_cls: set = set()
                for _n, _m in self.model.named_modules():
                    tn = type(_m).__name__
                    if ("MoE" in tn or "Experts" in tn) and type(_m) not in seen_cls:
                        seen_cls.add(type(_m))
                        for klass in type(_m).__mro__:
                            if "forward" in vars(klass) and klass.__module__.startswith("sglang"):
                                if not getattr(klass.forward, "_aic", False):
                                    w = _capture(
                                        klass.forward, "moe_calls",
                                        lambda s, t=tn: f"{t}[{type(getattr(s, 'quant_method', None)).__name__}]")
                                    w._aic = True
                                    setattr(klass, "forward", w)
                                break
                ab_cls = type(self.attn_backend)
                for meth in ("forward_extend", "forward_decode"):
                    fn = getattr(ab_cls, meth, None)
                    if fn is not None and not getattr(fn, "_aic", False):
                        w = _capture(fn, "attn_calls",
                                     lambda s, m=meth: f"{type(s).__name__}.{m}", max_calls=2)
                        w._aic = True
                        setattr(ab_cls, meth, w)
                _dump(ident["tp_rank"])
            except Exception as e:  # probe must never kill the run
                state["identity"]["inject_error"] = f"{type(e).__name__}: {e}"
                _dump(state["identity"].get("tp_rank", 0))
            return out

        mr_mod.ModelRunner.init_attention_backends = init_attn
        print("[AIC-inject] model_runner hooked", file=__import__("sys").stderr)

    try:
        # defer until sglang is importable; cheap retry via import hook
        import importlib.abc
        import importlib.util
        import sys

        class _Late(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name == "sglang.srt.model_executor.model_runner":
                    sys.meta_path.remove(self)
                    spec = importlib.util.find_spec(name)

                    orig_exec = spec.loader.exec_module

                    def exec_module(module):
                        orig_exec(module)
                        _install()

                    spec.loader.exec_module = exec_module
                    return spec
                return None

        sys.meta_path.insert(0, _Late())
    except Exception:
        pass
