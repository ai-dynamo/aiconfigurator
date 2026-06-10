#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public FPM ground-truth collector CLI for Dynamo/vLLM."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

try:
    from collector.layerwise.fpm.artifacts import case_run_dir, resolve_run_dir
    from collector.layerwise.fpm.datapoint_generator import generate_fpm_cases, preset_defaults
    from collector.layerwise.fpm.docker import DEFAULT_IMAGE, build_collect_command
except ModuleNotFoundError:  # pragma: no cover - direct script compatibility
    _REPO_ROOT = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(_REPO_ROOT))
    from collector.layerwise.fpm.artifacts import case_run_dir, resolve_run_dir
    from collector.layerwise.fpm.datapoint_generator import generate_fpm_cases, preset_defaults
    from collector.layerwise.fpm.docker import DEFAULT_IMAGE, build_collect_command


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the public FPM collector argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF model name or local path to serve.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Run directory. Defaults under .tmp/layerwise-artifacts/runs.")
    parser.add_argument("--tp-sizes", default="1", help="Comma-separated TP sizes to collect.")
    parser.add_argument("--ep-sizes", default="1", help="Comma-separated EP sizes to collect.")
    parser.add_argument("--phases", default="context,decode", help="Comma-separated phases: context,decode,mixed.")
    parser.add_argument("--run-preset", choices=("production", "smoke"), default="production")
    parser.add_argument("--contexts", default=None, help="Override context ISL values.")
    parser.add_argument("--decode-batches", default=None, help="Override decode batch sizes.")
    parser.add_argument("--decode-past-kv", default=None, help="Override decode past-KV values.")
    parser.add_argument("--decode-osl", default=None, help="Override decode output token count.")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Dynamo vLLM runtime image.")

    advanced = parser.add_argument_group("advanced/debug options")
    advanced.add_argument("--gpus", default=None)
    advanced.add_argument("--warmup-requests", type=int, default=None)
    advanced.add_argument("--keep-running", action="store_true")
    advanced.add_argument("--dry-run", action="store_true")
    advanced.add_argument("--extra-vllm-arg", action="append", default=[])
    return parser


def _apply_preset_defaults(args: argparse.Namespace) -> None:
    """Fill omitted shape args from the selected preset."""
    defaults = preset_defaults(args.run_preset)
    args.contexts = args.contexts or defaults["contexts"]
    args.decode_batches = args.decode_batches or defaults["decode_batches"]
    args.decode_past_kv = args.decode_past_kv or defaults["decode_past_kv"]
    args.decode_osl = args.decode_osl or defaults["decode_osl"]


def main() -> None:
    """Run FPM collection for each requested TP/EP/KV case."""
    parser = _build_arg_parser()
    args = parser.parse_args()
    _apply_preset_defaults(args)
    root = resolve_run_dir(args.run_dir, args.model)
    cases = generate_fpm_cases(args.tp_sizes, args.ep_sizes, args.decode_past_kv)
    if not cases:
        raise SystemExit("No FPM cases selected; check --tp-sizes/--ep-sizes/--decode-past-kv.")

    env = os.environ.copy()
    for case in cases:
        run_dir = case_run_dir(root, case, multiple_cases=len(cases) > 1)
        cmd = build_collect_command(args, case, run_dir)
        print("+ " + " ".join(cmd.argv), flush=True)
        result = subprocess.run(cmd.argv, env={**env, **cmd.env}, check=False)
        if result.returncode != 0:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
