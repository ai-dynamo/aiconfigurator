# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dedicated command line for whole-model FPM campaigns."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence

from collector.model_cases import build_collection_case_plan

from .config import add_fpm_arguments, add_fpm_generator_arguments
from .entry import resolve_inputs, resolve_run_inputs, run_resolved

_INPUT_ERRORS = (FileNotFoundError, RuntimeError, TypeError, ValueError)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m collector.fpm_forward",
        description="Plan or run a Generator-resolved Dynamo-native FPM campaign.",
    )
    parser.add_argument("--backend", choices=("vllm",), default="vllm")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--model-architecture", default=None)
    parser.add_argument("--model-cases", default=None, help="Optional model cases YAML path.")
    parser.add_argument("--gpu", required=True, help="Target AIC system, for example b200_sxm.")
    parser.add_argument("--sm", type=int, default=None, help="Optional explicit SM version for case planning.")
    parser.add_argument("--plan-only", action="store_true", help="Print the frozen FPM plan and exit.")
    parser.add_argument("--smoke", action="store_true", help="Run the minimal smoke sampling profile.")
    parser.add_argument("--limit", type=int, default=None, help="Limit cells; allowed only with --smoke.")
    parser.add_argument("--resume", action="store_true", help="Resume the matching frozen-plan checkpoint.")
    parser.add_argument(
        "--resume-retry-failed",
        action="store_true",
        help="Retry failed cells while resuming; requires --resume.",
    )
    parser.add_argument("--checkpoint-dir", default=".collector_checkpoint")
    add_fpm_arguments(parser)
    add_fpm_generator_arguments(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not (args.model_path or args.model_architecture or args.model_cases):
        parser.error("FPM requires --model-path, --model-architecture, or --model-cases")
    if args.resume_retry_failed and not args.resume:
        parser.error("--resume-retry-failed requires --resume")

    try:
        case_plan = build_collection_case_plan(
            backend=args.backend,
            model_path=args.model_path,
            model_architecture=args.model_architecture,
            gpu_type=args.gpu,
            sm_version=args.sm,
            model_cases_path=args.model_cases,
        )
        if case_plan.model_path:
            os.environ["COLLECTOR_MODEL_PATH"] = case_plan.model_path
        if args.plan_only:
            plan, _generator_overrides = resolve_inputs(args, case_plan)
            print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
            return 0
        resolved_inputs = resolve_run_inputs(args, case_plan)
    except _INPUT_ERRORS as error:
        parser.error(str(error))

    errors = run_resolved(args, resolved_inputs)
    if errors:
        print(json.dumps(errors, indent=2, sort_keys=True), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
