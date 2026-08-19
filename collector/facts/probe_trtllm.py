#!/usr/bin/env python3
"""TRT-LLM identity probe — SIMPLE version (agreed: try simple first).

llmapi LLM(load_format='dummy') -> object-graph search for the torch model
(A-level introspection) -> one tiny generate under torch.profiler (C-level
kernels). Expected failure mode: the executor lives in a subprocess and the
model/kernels are invisible in-process — if so, this records exactly that,
which is the datapoint that justifies the complex design.

Runs INSIDE trtllm-probe:1.3.0rc20-onnxfix with LD_LIBRARY_PATH set by the
image entrypoint (invoke via `bash -lc`).
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import traceback
from collections import Counter, defaultdict


import importlib.abc
import importlib.util


class _CutlassWalkGuard(importlib.abc.MetaPathFinder):
    """trtllm's warmup pkgutil.walk_packages force-imports every cutlass
    submodule, including `cutlass._mlir_helpers` — a module the normal flow
    never imports because `cutlass.base_dsl._mlir_helpers` already registered
    the same MLIR value casters -> fatal double registration. Raising
    ImportError here is safe: walk_packages ignores ImportError by design,
    and any legitimate later import of this module would have crashed anyway."""

    # only the never-legitimately-imported duplicate-caster module is blocked;
    # blocking wider cutlass._mlir broke legit cute-dsl runner imports
    BLOCK = ("cutlass._mlir_helpers",)

    def find_spec(self, name, path=None, target=None):
        for b in self.BLOCK:
            if name == b or name.startswith(b + "."):
                raise ImportError(f"blocked by AIC probe walk-guard: {name}")
        return None


sys.meta_path.insert(0, _CutlassWalkGuard())

# cutlass DSL hashes its own module tree via pkgutil.walk_packages for a JIT
# cache key (cutlass.py:512). walk_packages only forgives ImportError; broken
# generated dialect modules raise AttributeError and kill the load. The MLIR
# dialect imports bypass meta_path (file-based loaders), so guard at the
# walk itself: truncate on ANY exception — a shorter hash input is harmless.
import pkgutil

_orig_walk = pkgutil.walk_packages


def _safe_walk(*a, **k):
    it = _orig_walk(*a, **k)
    while True:
        try:
            yield next(it)
        except StopIteration:
            return
        except Exception:
            return


pkgutil.walk_packages = _safe_walk


def find_torch_model(root, max_depth: int = 8):
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
            continue
        for name in dir(obj):
            if name.startswith("__"):
                continue
            try:
                child = getattr(obj, name)
            except Exception:
                continue
            if isinstance(child, nn.Module):
                queue.append((child, d + 1))
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
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--trust-remote-code", action="store_true")
    args = ap.parse_args()
    rec: dict = {"model_path": args.model, "errors": {}}

    import tensorrt_llm

    rec["trtllm_version"] = tensorrt_llm.__version__
    # This image's _cutlass_ir C lib predates register_traceback_file_exclusion,
    # so the generated `_iket_ops_gen` module crashes on import — but the iket
    # dialect IS legitimately imported (SM100 cute-dsl runners) even on SM90,
    # where its ops never execute. Give ONLY the generated-ops module a
    # permissive stub (PEP 562 __getattr__): imports succeed, attribute
    # accesses yield dummies, nothing SM90 actually runs is affected.
    # Only stub when the real module is actually broken (rc20 image had a
    # stale _cutlass_ir C lib); newer images import it fine and must not be
    # shadowed.
    import types
    try:
        import cutlass._mlir.dialects._iket_ops_gen  # noqa: F401
        rec["cutlass_stub_modules"] = []
    except Exception:
        _gen = types.ModuleType("cutlass._mlir.dialects._iket_ops_gen")
        _gen.__all__ = []
        _gen.__getattr__ = lambda name: type(name, (object,), {"__init__": lambda self, *a, **k: None})
        sys.modules["cutlass._mlir.dialects._iket_ops_gen"] = _gen
        rec["cutlass_stub_modules"] = ["cutlass._mlir.dialects._iket_ops_gen"]
    try:
        from tensorrt_llm import LLM
        from tensorrt_llm.llmapi import KvCacheConfig

        llm = LLM(
            model=args.model,
            load_format="dummy",
            trust_remote_code=args.trust_remote_code,
            kv_cache_config=KvCacheConfig(max_tokens=16384),
            max_batch_size=8,
            max_seq_len=4096,
        )
        rec["llm_class"] = type(llm).__qualname__
        rec["executor_class"] = type(getattr(llm, "_executor", None)).__qualname__

        model = find_torch_model(llm)
        if model is None:
            rec["in_process_model"] = False  # THE datapoint: subprocess executor
        else:
            rec["in_process_model"] = True
            rec["model_class"] = type(model).__qualname__
            qm = defaultdict(list)
            for name, mod in model.named_modules():
                q = getattr(mod, "quant_method", None) or getattr(mod, "quant_config", None)
                if q is not None:
                    qm[type(q).__name__].append(name)
            rec["quant_methods"] = {k: {"count": len(v), "modules": v[:4]} for k, v in qm.items()}
            rec["param_dtypes"] = dict(Counter(str(p.dtype) for p in model.parameters()))

        try:
            import torch
            from torch.profiler import ProfilerActivity, profile

            from tensorrt_llm import SamplingParams

            _ = llm.generate([[1] * 16], SamplingParams(max_tokens=2))  # warmup
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
                llm.generate([[1] * 16], SamplingParams(max_tokens=2))
                torch.cuda.synchronize()
            rows = []
            for e in p.key_averages():
                dt = getattr(e, "self_device_time_total", 0) or getattr(e, "self_cuda_time_total", 0)
                if dt > 0:
                    rows.append({"kernel": e.key, "calls": e.count, "us": round(dt, 1)})
            rec["kernels_visible_in_process"] = bool(rows)  # False again == subprocess
            rec["kernels"] = sorted(rows, key=lambda r: -r["us"])[:40]
        except Exception:
            rec["errors"]["generate"] = traceback.format_exc()[-2000:]
    except Exception:
        rec["errors"]["load"] = traceback.format_exc()

    json.dump(rec, open(args.out, "w"), indent=1, default=str)
    print("WROTE", args.out, "errors:", list(rec["errors"]),
          "in_process:", rec.get("in_process_model"), "kernels:", rec.get("kernels_visible_in_process"))


if __name__ == "__main__":
    main()
