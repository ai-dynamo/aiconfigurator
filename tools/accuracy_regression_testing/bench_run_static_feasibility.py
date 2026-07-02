#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Feasibility micro-benchmark for a run_static-based regression tier.

Times, per representative silicon_sample row:
  - cli_estimate(mode='agg' / 'disagg')      -> what the current regression test pays per row
  - cli_estimate(mode='static_ctx' / '_gen') -> what a naive per-row static harness would pay
  - session-reuse loop                       -> marginal run_static cost once DB+model are built

Run:  PYTHONPATH=src .venv/bin/python tools/accuracy_regression_testing/bench_run_static_feasibility.py
"""

import csv
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SAMPLE = REPO / "src/aiconfigurator/systems/silicon_sample.csv"

t0 = time.perf_counter()
from aiconfigurator.cli.api import cli_estimate
from aiconfigurator.sdk import config as sdk_config
from aiconfigurator.sdk.backends.factory import get_backend
from aiconfigurator.sdk.inference_session import InferenceSession
from aiconfigurator.sdk.models import get_model
from aiconfigurator.sdk.perf_database import get_database_view

IMPORT_S = time.perf_counter() - t0

# (model, system, backend): version resolves to the latest local database,
# mirroring the CI missing-version fallback in predict_silicon_sample.py.
TARGETS = [
    ("Qwen/Qwen3-32B", "h100_sxm", "trtllm"),
    ("deepseek-ai/DeepSeek-R1", "h200_sxm", "trtllm"),
]


def pick_rows():
    with open(SAMPLE) as f:
        rows = list(csv.DictReader(f))
    picked = {}
    for model, system, backend in TARGETS:
        match = lambda r, mode: (
            r["model_path"] == model and r["system"] == system and r["backend"] == backend and r["mode"] == mode
        )
        agg = next((r for r in rows if match(r, "agg")), None)
        dis = next((r for r in rows if match(r, "disagg")), None)
        if agg is not None:
            picked[model] = (agg, dis)
    return picked


def timed(label, fn, n=1):
    t = time.perf_counter()
    try:
        for _ in range(n):
            with redirect_stdout(sys.stderr):
                fn()
    except Exception as e:
        dt = (time.perf_counter() - t) / n
        print(f"{label:58s} {dt * 1000:10.1f} ms  [FAILED: {type(e).__name__}]")
        return dt
    dt = (time.perf_counter() - t) / n
    print(f"{label:58s} {dt * 1000:10.1f} ms")
    return dt


def base_kwargs(row):
    kw = dict(
        model_path=row["model_path"],
        system_name=row["system"],
        backend_name=row["backend"],
        backend_version=None,  # latest local DB, like the CI missing-version fallback
        isl=int(row["isl"]),
        osl=int(row["osl"]),
        gemm_quant_mode=row["gemm_quant_mode"] or None,
        moe_quant_mode=row["moe_quant_mode"] or None,
    )
    return kw


def agg_kwargs(row):
    kw = base_kwargs(row)
    for f in ("batch_size", "tp_size", "pp_size", "attention_dp_size", "moe_tp_size", "moe_ep_size"):
        if row.get(f):
            kw[f] = int(row[f])
    return kw


def disagg_kwargs(row):
    kw = base_kwargs(row)
    kw["mode"] = "disagg"
    for stage in ("prefill", "decode"):
        for f in ("batch_size", "num_workers", "tp_size", "pp_size", "attention_dp_size", "moe_tp_size", "moe_ep_size"):
            v = row.get(f"{stage}_{f}")
            if v:
                kw[f"{stage}_{f}"] = int(v)
    return kw


def main():
    print(f"{'import aiconfigurator':58s} {IMPORT_S * 1000:10.1f} ms")
    picked = pick_rows()

    for model, (agg_row, dis_row) in picked.items():
        if agg_row is None:
            continue
        print(f"\n=== {model} @ {agg_row['system']}/{agg_row['backend']}/{agg_row['backend_version']} ===")
        akw = agg_kwargs(agg_row)

        # 1) static point, cold DB (first touch of this system/backend/version)
        skw = dict(akw)
        skw.pop("batch_size", None)
        timed(
            "cli_estimate static_ctx (cold DB load + model build)",
            lambda: cli_estimate(mode="static_ctx", batch_size=1, **skw),
        )
        # 2) static points, warm DB
        timed("cli_estimate static_ctx (warm)", lambda: cli_estimate(mode="static_ctx", batch_size=1, **skw))
        timed("cli_estimate static_gen  (warm)", lambda: cli_estimate(mode="static_gen", batch_size=32, **skw))
        # 3) today's cost: full agg estimate on the same combo
        timed("cli_estimate agg (warm DB)  [current per-row cost]", lambda: cli_estimate(**akw))
        if dis_row is not None:
            dkw = disagg_kwargs(dis_row)
            timed("cli_estimate disagg (warm DB) [current per-row cost]", lambda: cli_estimate(**dkw))

        # 4) marginal run_static cost with session reuse
        model_cfg_kwargs = {}
        for f, name in (
            ("tp_size", "tp_size"),
            ("pp_size", "pp_size"),
            ("attention_dp_size", "attention_dp_size"),
            ("moe_tp_size", "moe_tp_size"),
            ("moe_ep_size", "moe_ep_size"),
        ):
            if agg_row.get(f):
                model_cfg_kwargs[name] = int(agg_row[f])
        from aiconfigurator.cli.api import _build_model_config, _resolve_moe_parallelism

        moe_tp, moe_ep = _resolve_moe_parallelism(
            model_cfg_kwargs.get("tp_size", 1),
            model_cfg_kwargs.get("attention_dp_size", 1),
            model_cfg_kwargs.get("moe_tp_size"),
            model_cfg_kwargs.get("moe_ep_size"),
            model_path=model,
        )
        mc = _build_model_config(
            model_cfg_kwargs.get("tp_size", 1),
            model_cfg_kwargs.get("pp_size", 1),
            model_cfg_kwargs.get("attention_dp_size", 1),
            moe_tp,
            moe_ep,
            agg_row["gemm_quant_mode"] or None,
            None,
            None,
            agg_row["moe_quant_mode"] or None,
            None,
        )
        from aiconfigurator.sdk.perf_database import get_latest_database_version

        latest = get_latest_database_version(system=agg_row["system"], backend=agg_row["backend"])
        t = time.perf_counter()
        db = get_database_view(agg_row["system"], agg_row["backend"], latest, database_mode="SILICON")
        print(f"{'get_database_view (warm template)':58s} {(time.perf_counter() - t) * 1000:10.1f} ms")
        t = time.perf_counter()
        m = get_model(model, mc, agg_row["backend"])
        be = get_backend(agg_row["backend"])
        sess = InferenceSession(m, db, be)
        print(f"{'get_model + backend + session':58s} {(time.perf_counter() - t) * 1000:10.1f} ms")

        isl = int(agg_row["isl"])
        points = []
        for bs in (1, 4, 16, 64):
            points.append(("static_ctx", sdk_config.RuntimeConfig(batch_size=bs, isl=isl, osl=8)))
            points.append(("static_gen", sdk_config.RuntimeConfig(batch_size=bs, isl=isl, osl=int(agg_row["osl"]))))
        ok = 0
        t = time.perf_counter()
        for mode, rc in points:
            try:
                with redirect_stdout(sys.stderr):
                    sess.run_static(runtime_config=rc, mode=mode, stride=32)
                ok += 1
            except Exception as e:
                print(f"  point {mode} bs={rc.batch_size}: {type(e).__name__}", file=sys.stderr)
        dt = time.perf_counter() - t
        print(
            f"{'run_static x' + str(len(points)) + ' pts (session reuse, ' + str(ok) + ' ok), avg':58s} "
            f"{dt / len(points) * 1000:10.1f} ms"
        )


if __name__ == "__main__":
    main()
