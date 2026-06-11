#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public vLLM layerwise collector CLI with registry-backed defaults."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

try:
    from collector.layerwise.common.paths import default_run_dir
    from collector.layerwise.vllm.datapoint_generator import build_public_work_units
    from collector.layerwise.vllm.registry import LayerwiseModel
    from collector.layerwise.vllm.registry import select_models
    from collector.layerwise.vllm.scheduler import Scheduler
    from collector.layerwise.vllm.worker import run_worker
except ModuleNotFoundError:  # pragma: no cover - direct script compatibility
    _REPO_ROOT = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(_REPO_ROOT))
    from collector.layerwise.common.paths import default_run_dir
    from collector.layerwise.vllm.datapoint_generator import build_public_work_units
    from collector.layerwise.vllm.registry import LayerwiseModel
    from collector.layerwise.vllm.registry import select_models
    from collector.layerwise.vllm.scheduler import Scheduler
    from collector.layerwise.vllm.worker import run_worker


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the public layerwise collector argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None, help="Run directory. Defaults under .tmp/layerwise-artifacts/runs.")
    parser.add_argument("--model", default=None, help="Single ad-hoc HF/local model to collect. Use --models for registry filters.")
    parser.add_argument("--models", default=None, help="Comma-separated registry model filter. Defaults to all models.")
    parser.add_argument("--tp-sizes", default="1,2,4,8", help="Comma-separated TP sizes to collect.")
    parser.add_argument("--ep-sizes", default="auto", help="Comma-separated EP sizes, or auto for model defaults.")
    parser.add_argument("--phases", choices=("ctx", "gen", "both"), default="both")
    parser.add_argument("--run-preset", choices=("production", "smoke"), default="production")
    parser.add_argument("--ctx-new-tokens", default=None, help="Override context new-token grid.")
    parser.add_argument("--ctx-past-kv", default=None, help="Override context past-KV grid.")
    parser.add_argument("--gen-batch-sizes", default=None, help="Override decode batch-size grid.")
    parser.add_argument("--gen-past-kv", default=None, help="Override decode past-KV grid.")
    parser.add_argument(
        "--max-decode-batch-size",
        default="auto",
        help="Maximum default decode batch size for production presets. "
        "'auto' uses vLLM's hardware-dependent default max_num_seqs.",
    )
    parser.add_argument("--max-workers", type=int, default=None, help="Limit concurrent one-GPU workers.")

    advanced = parser.add_argument_group("advanced/debug options")
    advanced.add_argument("--gpus", default=None, help="Comma-separated physical GPU IDs. Defaults to visible GPUs.")
    advanced.add_argument("--system", default=None, help="Override output system label.")
    advanced.add_argument("--framework-version", default=None, help="Override output vLLM version label.")
    advanced.add_argument("--no-config-cache", action="store_true")
    advanced.add_argument("--model-kind", choices=("dense", "moe"), default="dense", help="Layer type for --model ad-hoc entries.")
    advanced.add_argument("--num-slots", type=int, default=None, help="MoE/EPLB expert slot count for ad-hoc or overridden entries.")
    advanced.add_argument("--gemm-quant", default=None)
    advanced.add_argument("--moe-quant", default=None)
    advanced.add_argument("--attn-quant", default=None)
    advanced.add_argument("--kv-quant", default=None)
    advanced.add_argument("--moe-noop", action="store_true")
    advanced.add_argument("--target-layer-count", type=int, default=1)
    advanced.add_argument("--target-layers", default=None)
    advanced.add_argument("--target-layer-config-depth", type=int, default=None)
    advanced.add_argument("--no-filter-model-max-len", action="store_true")
    advanced.add_argument("--rank-reduce", choices=("sum", "max"), default="max")
    advanced.add_argument("--latency-source", choices=("span", "gpu", "gpu_capped"), default="span")
    advanced.add_argument("--ctx-warmup-runs", type=int, default=0)
    advanced.add_argument("--ctx-measured-runs", type=int, default=6)
    advanced.add_argument("--ctx-repeat-aggregation", choices=("median", "mean", "trimmed_mean", "min"), default="trimmed_mean")
    advanced.add_argument("--gen-warmup-runs", type=int, default=0)
    advanced.add_argument("--gen-measured-runs", type=int, default=6)
    advanced.add_argument("--gen-repeat-aggregation", choices=("median", "mean", "trimmed_mean", "min"), default="trimmed_mean")
    advanced.add_argument("--timeout", type=int, default=1800)
    advanced.add_argument("--nsys-capture", choices=("full", "cuda_profiler_api"), default="full")
    advanced.add_argument("--rollup", default=None)
    advanced.add_argument("--extra-vllm-arg", action="append", default=[])
    return parser


def _resolve_run_dir(raw_run_dir: Path | None) -> Path:
    """Return the user-supplied or default layerwise run directory."""
    return (raw_run_dir or default_run_dir("layerwise_vllm")).expanduser().resolve()


def _selected_models(args: argparse.Namespace) -> list[LayerwiseModel]:
    """Return registry or ad-hoc model entries selected by the public CLI."""
    if args.model and args.models:
        raise SystemExit("Use either --model for one ad-hoc model or --models for registry filters, not both.")

    if args.model:
        ep_sizes = (1, 2, 4, 8) if args.model_kind == "moe" else (1,)
        return [
            LayerwiseModel(
                model=args.model,
                kind=args.model_kind,
                ep_sizes=ep_sizes,
                gemm_quant=args.gemm_quant or "bf16",
                moe_quant=args.moe_quant or "bf16",
                attn_quant=args.attn_quant or "bf16",
                kv_quant=args.kv_quant or "bf16",
                num_slots=args.num_slots,
            )
        ]

    models = select_models(args.models)
    overrides = {
        key: value for key, value in {
            "gemm_quant": args.gemm_quant,
            "moe_quant": args.moe_quant,
            "attn_quant": args.attn_quant,
            "kv_quant": args.kv_quant,
            "num_slots": args.num_slots,
        }.items() if value is not None
    }
    if overrides:
        return [replace(model, **overrides) for model in models]
    return models


def main() -> None:
    """Run the public layerwise CLI or worker subcommand."""
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        worker_parser = argparse.ArgumentParser(description="Internal vLLM layerwise worker")
        worker_parser.add_argument("worker")
        worker_parser.add_argument("--spec", required=True)
        worker_args = worker_parser.parse_args()
        run_worker(Path(worker_args.spec))
        return

    parser = _build_arg_parser()
    args = parser.parse_args()
    args.run_dir = _resolve_run_dir(args.run_dir)
    args.work_dir = str(args.run_dir / "profiles")
    args.output = str(args.run_dir / "layerwise.csv")
    args.config_cache_dir = str(args.run_dir / "config_cache")

    models = _selected_models(args)
    work_units = build_public_work_units(args, models)
    if not work_units:
        raise SystemExit("No layerwise work units selected; check --models/--tp-sizes/--ep-sizes.")
    Scheduler(args, work_units, worker_entrypoint=Path(__file__).resolve()).run()


if __name__ == "__main__":
    main()
