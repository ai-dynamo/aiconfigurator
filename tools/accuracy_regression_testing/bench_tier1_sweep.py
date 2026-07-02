#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tier-1 shaped sweep: every bundled model x one (system, backend, latest) combo.

Per (model, combo): build model+session once, run a fixed prefill/decode point
grid via run_static, record latency vector or structured miss. Reports wall
time and per-model cost -- the empirical basis for sizing the tier-1 CI job.

Usage:
  PYTHONPATH=src python tools/accuracy_regression_testing/bench_tier1_sweep.py [system] [backend] [--no-shared-layer]
"""

import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

from aiconfigurator.cli.api import _build_model_config, _resolve_moe_parallelism
from aiconfigurator.sdk import config as sdk_config
from aiconfigurator.sdk import perf_database
from aiconfigurator.sdk.backends.factory import get_backend
from aiconfigurator.sdk.errors import PerfDataNotAvailableError
from aiconfigurator.sdk.inference_session import InferenceSession
from aiconfigurator.sdk.models import get_model
from aiconfigurator.sdk.perf_database import get_database_view, get_latest_database_version

# prefill: bs=1 across isl; decode: bs sweep at fixed isl -- small, fixed grid.
PREFILL_POINTS = [(1, 1024), (1, 8192)]
DECODE_POINTS = [(1, 1024), (32, 1024), (128, 1024), (32, 8192)]


def model_names() -> list[str]:
    names = []
    for p in sorted((REPO / "src/aiconfigurator/model_configs").glob("*_config.json")):
        name = p.name.removesuffix("_config.json").replace("--", "/")
        if name.endswith("_hf_quant"):  # HF-download quant-config variants; not offline-buildable
            continue
        names.append(name)
    return names


def sweep(system: str, backend: str, shared_layer: bool) -> None:
    if not shared_layer:
        perf_database._shared_layer_enabled = lambda mode: False  # simulate the future knob
        perf_database.databases_cache.clear()

    version = get_latest_database_version(system=system, backend=backend)
    print(f"combo: {system}/{backend}/{version}  shared_layer={shared_layer}")
    t0 = time.perf_counter()
    db = get_database_view(system, backend, version, database_mode="SILICON")
    t_db = time.perf_counter() - t0
    print(f"db load: {t_db:.2f}s")

    n_pts = ok_models = 0
    t0 = time.perf_counter()
    for name in model_names():
        t_m = time.perf_counter()
        try:
            moe_tp, moe_ep = _resolve_moe_parallelism(4, 1, 1, 4, model_path=name)
            mc = _build_model_config(4, 1, 1, moe_tp, moe_ep, None, None, None, None, None)
            model = get_model(name, mc, backend)
            sess = InferenceSession(model, db, get_backend(backend))
        except Exception as e:
            print(f"  {name:55s} BUILD_FAIL {type(e).__name__}: {e}")
            continue
        misses = errors = 0
        for mode, pts in (("static_ctx", PREFILL_POINTS), ("static_gen", DECODE_POINTS)):
            for bs, isl in pts:
                osl = 8 if mode == "static_ctx" else 256
                rc = sdk_config.RuntimeConfig(batch_size=bs, isl=isl, osl=osl)
                try:
                    with redirect_stdout(sys.stderr):
                        sess.run_static(runtime_config=rc, mode=mode, stride=32)
                except PerfDataNotAvailableError:
                    misses += 1
                except Exception:
                    errors += 1
                n_pts += 1
        ok_models += 1
        dt = time.perf_counter() - t_m
        tag = f"misses={misses} errors={errors}" if (misses or errors) else "all ok"
        print(f"  {name:55s} {dt * 1000:7.0f} ms  {tag}")
    total = time.perf_counter() - t0
    n_models = ok_models
    print(
        f"\nTOTAL {total:.1f}s for {n_models} models x {len(PREFILL_POINTS) + len(DECODE_POINTS)} points"
        f" ({n_pts} points, {total / max(n_models, 1):.2f}s/model incl. warmup)"
    )


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    system = args[0] if args else "h200_sxm"
    backend = args[1] if len(args) > 1 else "trtllm"
    sweep(system, backend, shared_layer="--no-shared-layer" not in sys.argv)
