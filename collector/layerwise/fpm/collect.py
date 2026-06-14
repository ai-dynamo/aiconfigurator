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
    from collector.layerwise.fpm.datapoint_generator import FpmCase, default_shapes, generate_fpm_cases
    from collector.layerwise.fpm.docker import DEFAULT_IMAGE, build_collect_command
except ModuleNotFoundError:  # pragma: no cover - direct script compatibility
    _REPO_ROOT = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(_REPO_ROOT))
    from collector.layerwise.fpm.artifacts import case_run_dir, resolve_run_dir
    from collector.layerwise.fpm.datapoint_generator import FpmCase, default_shapes, generate_fpm_cases
    from collector.layerwise.fpm.docker import DEFAULT_IMAGE, build_collect_command


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the public FPM collector argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF model name or local path to serve.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Run directory. Defaults under .tmp/layerwise-artifacts/runs.",
    )
    parser.add_argument("--tp-sizes", default="1", help="Comma-separated TP sizes to collect.")
    parser.add_argument("--ep-sizes", default="1", help="Comma-separated EP sizes to collect.")
    parser.add_argument("--phases", default="context,decode", help="Comma-separated phases: context,decode,mixed.")
    parser.add_argument("--contexts", default=None, help="Override context ISL values.")
    parser.add_argument("--context-repeats", type=int, default=None, help="Context repeats per ISL.")
    parser.add_argument("--decode-batches", default=None, help="Override decode batch sizes.")
    parser.add_argument("--decode-past-kv", default=None, help="Override decode past-KV values.")
    parser.add_argument("--decode-osl", default=None, help="Override decode output token count.")
    parser.add_argument("--decode-repeats", type=int, default=None, help="Decode repeats per batch size.")
    parser.add_argument("--real-workload", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--include-sweep",
        action="store_true",
        help="When --real-workload is enabled, also run the static phase sweep in the same deployment.",
    )
    parser.add_argument("--real-workload-requests", type=int, default=None)
    parser.add_argument("--real-workload-concurrency", type=int, default=None)
    parser.add_argument("--real-workload-dataset", default=None)
    parser.add_argument("--real-workload-shape-source", choices=["scaled_dataset", "synthetic"], default=None)
    parser.add_argument("--real-workload-isl-min", type=int, default=None)
    parser.add_argument("--real-workload-isl-max", type=int, default=None)
    parser.add_argument("--real-workload-isl-mean", type=float, default=None)
    parser.add_argument("--real-workload-osl-min", type=int, default=None)
    parser.add_argument("--real-workload-osl-max", type=int, default=None)
    parser.add_argument("--real-workload-osl-mean", type=float, default=None)
    parser.add_argument(
        "--request-allow-failures",
        type=int,
        default=0,
        help="Continue if at most this many driven requests fail after retries.",
    )
    parser.add_argument(
        "--prompt-token-mode",
        choices=["random_vocab_excluding_special", "safe_ascii"],
        default="safe_ascii",
        help="Prompt token sampling mode for the request driver.",
    )
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Dynamo vLLM runtime image.")

    advanced = parser.add_argument_group("advanced/debug options")
    advanced.add_argument("--gpus", default=None)
    advanced.add_argument("--warmup-requests", type=int, default=None)
    advanced.add_argument("--keep-running", action="store_true")
    advanced.add_argument("--dry-run", action="store_true")
    advanced.add_argument(
        "--continue-on-case-failure",
        action="store_true",
        help="Run remaining TP/EP cases after a case fails, then exit nonzero.",
    )
    advanced.add_argument("--extra-vllm-arg", action="append", default=[])
    return parser


def _apply_shape_defaults(args: argparse.Namespace) -> None:
    """Fill omitted shape args from the default FPM shape set."""
    defaults = default_shapes()
    args.contexts = args.contexts or defaults["contexts"]
    args.context_repeats = args.context_repeats or int(defaults["context_repeats"])
    args.decode_batches = args.decode_batches or defaults["decode_batches"]
    args.decode_past_kv = args.decode_past_kv or defaults["decode_past_kv"]
    args.decode_osl = args.decode_osl or defaults["decode_osl"]
    args.decode_repeats = args.decode_repeats or int(defaults["decode_repeats"])
    args.real_workload_requests = args.real_workload_requests or int(defaults["real_workload_requests"])
    args.real_workload_concurrency = args.real_workload_concurrency or int(defaults["real_workload_concurrency"])
    args.real_workload_dataset = args.real_workload_dataset or defaults["real_workload_dataset"]
    args.real_workload_shape_source = args.real_workload_shape_source or defaults["real_workload_shape_source"]
    args.real_workload_isl_min = args.real_workload_isl_min or int(defaults["real_workload_isl_min"])
    args.real_workload_isl_max = args.real_workload_isl_max or int(defaults["real_workload_isl_max"])
    args.real_workload_isl_mean = args.real_workload_isl_mean or float(defaults["real_workload_isl_mean"])
    args.real_workload_osl_min = args.real_workload_osl_min or int(defaults["real_workload_osl_min"])
    args.real_workload_osl_max = args.real_workload_osl_max or int(defaults["real_workload_osl_max"])
    args.real_workload_osl_mean = args.real_workload_osl_mean or float(defaults["real_workload_osl_mean"])


def main() -> None:
    """Run FPM collection for each requested TP/EP/KV case."""
    parser = _build_arg_parser()
    args = parser.parse_args()
    _apply_shape_defaults(args)
    root = resolve_run_dir(args.run_dir, args.model)
    cases = generate_fpm_cases(args.tp_sizes, args.ep_sizes, args.decode_past_kv)
    if not cases:
        raise SystemExit("No FPM cases selected; check --tp-sizes/--ep-sizes/--decode-past-kv.")

    env = os.environ.copy()
    failed_cases: list[tuple[FpmCase, int]] = []
    for case in cases:
        run_dir = case_run_dir(root, case, multiple_cases=len(cases) > 1)
        cmd = build_collect_command(args, case, run_dir)
        print("+ " + " ".join(cmd.argv), flush=True)
        result = subprocess.run(cmd.argv, env={**env, **cmd.env}, check=False)
        if result.returncode != 0:
            if not args.continue_on_case_failure:
                raise SystemExit(result.returncode)
            failed_cases.append((case, result.returncode))
            print(
                f"FPM case {case.label} failed with exit code {result.returncode}; continuing.",
                file=sys.stderr,
                flush=True,
            )
    if failed_cases:
        failed = ", ".join(f"{case.label}:{code}" for case, code in failed_cases)
        print(f"FPM failed cases: {failed}", file=sys.stderr, flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
