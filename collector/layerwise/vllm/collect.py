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
    from collector.layerwise.vllm.registry import LayerwiseModel, select_models
    from collector.layerwise.vllm.scheduler import Scheduler
    from collector.layerwise.vllm.worker import run_worker
except ModuleNotFoundError:  # pragma: no cover - direct script compatibility
    _REPO_ROOT = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(_REPO_ROOT))
    from collector.layerwise.common.paths import default_run_dir
    from collector.layerwise.vllm.datapoint_generator import build_public_work_units
    from collector.layerwise.vllm.registry import LayerwiseModel, select_models
    from collector.layerwise.vllm.scheduler import Scheduler
    from collector.layerwise.vllm.worker import run_worker


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the public layerwise collector argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Run directory. Defaults under .tmp/layerwise-artifacts/runs.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Single ad-hoc HF/local model to collect. Use --models for registry filters.",
    )
    parser.add_argument("--models", default=None, help="Comma-separated registry model filter. Defaults to all models.")
    parser.add_argument("--tp-sizes", default="1,2,4,8", help="Comma-separated TP sizes to collect.")
    parser.add_argument("--ep-sizes", default="auto", help="Comma-separated EP sizes, or auto for model defaults.")
    parser.add_argument("--phases", choices=("ctx", "gen", "both"), default="both")
    parser.add_argument("--run-preset", choices=("full", "smoke"), default="full")
    parser.add_argument("--ctx-new-tokens", default=None, help="Override context new-token grid.")
    parser.add_argument("--ctx-past-kv", default=None, help="Override context past-KV grid.")
    parser.add_argument("--gen-batch-sizes", default=None, help="Override decode batch-size grid.")
    parser.add_argument("--gen-past-kv", default=None, help="Override decode past-KV grid.")
    parser.add_argument(
        "--max-decode-batch-size",
        default="512",
        help="Maximum default decode batch size for full presets. "
        "Use 'auto' to follow vLLM's hardware-dependent default max_num_seqs.",
    )
    parser.add_argument("--max-workers", type=int, default=None, help="Limit concurrent one-GPU workers.")

    advanced = parser.add_argument_group("advanced/debug options")
    advanced.add_argument("--gpus", default=None, help="Comma-separated physical GPU IDs. Defaults to visible GPUs.")
    advanced.add_argument("--system", default=None, help="Override output system label.")
    advanced.add_argument("--framework-version", default=None, help="Override output vLLM version label.")
    advanced.add_argument("--no-config-cache", action="store_true")
    advanced.add_argument(
        "--model-kind",
        choices=("dense", "moe"),
        default="dense",
        help="Layer type for --model ad-hoc entries.",
    )
    advanced.add_argument(
        "--num-slots",
        type=int,
        default=None,
        help="MoE/EPLB expert slot count for ad-hoc or overridden entries.",
    )
    advanced.add_argument("--gemm-quant", default=None)
    advanced.add_argument("--moe-quant", default=None)
    advanced.add_argument("--attn-quant", default=None)
    advanced.add_argument("--kv-quant", default=None)
    advanced.add_argument("--moe-noop", action="store_true")
    advanced.add_argument(
        "--moe-dummy-router",
        action="store_true",
        help=(
            "Diagnostic mode: keep dummy MoE weights/router active instead of "
            "replacing MoE with a no-op or loading real router weights."
        ),
    )
    advanced.add_argument(
        "--physical-tp",
        action="store_true",
        help=(
            "Diagnostic mode: run vLLM with real tensor parallelism across "
            "physical GPUs instead of single-GPU dimension patching. "
            "Requires --allow-multi-gpu-diagnostic and must not be promoted "
            "into canonical layerwise data."
        ),
    )
    advanced.add_argument(
        "--allow-multi-gpu-diagnostic",
        action="store_true",
        help=(
            "Allow diagnostic work units that reserve more than one physical "
            "GPU. Canonical layerwise collection should not use this flag."
        ),
    )
    advanced.add_argument(
        "--physical-tp-real-weights",
        action="store_true",
        help=(
            "Diagnostic mode: with --physical-tp full-depth runs, load the "
            "original model weights instead of dummy weights."
        ),
    )
    advanced.add_argument(
        "--moe-real-router",
        action="store_true",
        help=(
            "Diagnostic mode: load real MoE router weights for layerwise rows. "
            "The default avoids model-weight downloads and composes no-op "
            "layerwise rows with op-level MoE data."
        ),
    )
    advanced.add_argument("--target-layer-count", type=int, default=None)
    advanced.add_argument(
        "--ctx-target-layer-count",
        type=int,
        default=0,
        help="Context active layer count. 0 means all layers.",
    )
    advanced.add_argument("--gen-target-layer-count", type=int, default=1)
    advanced.add_argument("--target-layers", default=None)
    advanced.add_argument("--target-layer-config-depth", type=int, default=None)
    advanced.add_argument(
        "--max-num-seqs",
        type=int,
        default=None,
        help="Override vLLM scheduler max_num_seqs for FPM parity diagnostics.",
    )
    advanced.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help="Override vLLM scheduler max_num_batched_tokens for FPM parity diagnostics.",
    )
    advanced.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Override vLLM max_model_len for FPM parity diagnostics.",
    )
    advanced.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help=(
            "vLLM GPU memory utilization for layerwise worker engines. "
            "Increase only when the requested high-KV decode envelope is "
            "intended to fit on the target system."
        ),
    )
    advanced.add_argument("--no-filter-model-max-len", action="store_true")
    advanced.add_argument("--rank-reduce", choices=("sum", "max"), default="max")
    advanced.add_argument(
        "--latency-source",
        choices=("auto", "span", "gpu", "gpu_capped", "schedule_to_update", "worker_wall"),
        default="auto",
        help=(
            "Latency source for output rows. auto uses scheduler timing for "
            "context, kernel span timing for decode, and gpu-sum for high-batch "
            "MoE decode."
        ),
    )
    advanced.add_argument("--moe-decode-gpu-batch-threshold", type=int, default=8)
    advanced.add_argument("--ctx-warmup-runs", type=int, default=1)
    advanced.add_argument("--ctx-measured-runs", type=int, default=6)
    advanced.add_argument(
        "--ctx-repeat-aggregation",
        choices=("median", "mean", "trimmed_mean", "min"),
        default="trimmed_mean",
    )
    advanced.add_argument("--gen-warmup-runs", type=int, default=1)
    advanced.add_argument("--gen-measured-runs", type=int, default=6)
    advanced.add_argument(
        "--gen-repeat-aggregation",
        choices=("median", "mean", "trimmed_mean", "min"),
        default="trimmed_mean",
    )
    advanced.add_argument(
        "--gen-driver",
        choices=("prefix_cache", "live_decode"),
        default=None,
        help=(
            "Decode measurement driver. Defaults to the registry setting for "
            "known models and prefix_cache for ad-hoc models."
        ),
    )
    advanced.add_argument(
        "--live-step-driver",
        action="store_true",
        help=(
            "Use the lower-level live LLMEngine stepping driver for eligible "
            "ctx/gen datapoints. This keeps the requested shape grid but "
            "avoids repeated LLM.generate submission overhead."
        ),
    )
    advanced.add_argument(
        "--live-step-gen-min-past-kv",
        type=int,
        default=8192,
        help=(
            "Minimum decode past-KV for --live-step-driver generation rows. "
            "Lower-past decode rows stay on the prefix-cache generate path."
        ),
    )
    advanced.add_argument("--timeout", type=int, default=1800)
    advanced.add_argument("--nsys-capture", choices=("full", "cuda_profiler_api", "none"), default="full")
    advanced.add_argument(
        "--prompt-seed",
        type=int,
        default=None,
        help="Seed synthetic prompt token sampling. Defaults to random prompts each worker run.",
    )
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
                gen_driver=args.gen_driver or "prefix_cache",
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
            "gen_driver": args.gen_driver,
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
