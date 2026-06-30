# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command construction for the Dynamo/vLLM FPM shell collector."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from collector.layerwise.fpm.datapoint_generator import FpmCase

DEFAULT_IMAGE = "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.0"


@dataclass(frozen=True)
class FpmShellCommand:
    """A shell command and environment for one FPM case."""

    argv: list[str]
    env: dict[str, str] = field(default_factory=dict)


def build_collect_command(args, case: FpmCase, run_dir: Path) -> FpmShellCommand:
    """Build the existing shell-wrapper command for one FPM deployment."""
    script = Path("collector/layerwise/fpm_ground_truth/collect_fpm_metrics.sh")
    argv = [
        "bash",
        str(script),
        "--model",
        args.model,
        "--tp-size",
        str(case.tp_size),
        "--ep-size",
        str(case.ep_size),
        "--run-dir",
        str(run_dir),
        "--phases",
        args.phases,
        "--contexts",
        args.contexts,
        "--context-repeats",
        str(args.context_repeats),
        "--decode-batches",
        args.decode_batches,
        "--decode-past-kv",
        str(case.decode_past_kv),
        "--decode-osl",
        str(args.decode_osl),
        "--decode-repeats",
        str(args.decode_repeats),
        "--image",
        args.image,
        "--request-allow-failures",
        str(args.request_allow_failures),
        "--prompt-token-mode",
        args.prompt_token_mode,
    ]
    if args.include_sweep:
        argv.append("--include-sweep")
    if args.real_workload:
        argv.extend(
            [
                "--real-workload",
                "--real-workload-requests",
                str(args.real_workload_requests),
                "--real-workload-concurrency",
                str(args.real_workload_concurrency),
                "--real-workload-dataset",
                args.real_workload_dataset,
                "--real-workload-shape-source",
                args.real_workload_shape_source,
                "--real-workload-isl-min",
                str(args.real_workload_isl_min),
                "--real-workload-isl-max",
                str(args.real_workload_isl_max),
                "--real-workload-isl-mean",
                str(args.real_workload_isl_mean),
                "--real-workload-osl-min",
                str(args.real_workload_osl_min),
                "--real-workload-osl-max",
                str(args.real_workload_osl_max),
                "--real-workload-osl-mean",
                str(args.real_workload_osl_mean),
            ]
        )
    else:
        argv.append("--no-real-workload")
    if args.warmup_requests is not None:
        argv.extend(["--warmup-requests", str(args.warmup_requests)])
    if args.gpus:
        argv.extend(["--gpus", args.gpus])
    if args.keep_running:
        argv.append("--keep-running")
    if args.dry_run:
        argv.append("--dry-run")
    if args.extra_vllm_arg:
        argv.append("--")
        argv.extend(args.extra_vllm_arg)
    return FpmShellCommand(argv=argv)
