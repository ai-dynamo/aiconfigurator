#!/usr/bin/env python3
"""Collect DSv4 MegaMoE communication-path perf data.

This is the user-facing collector for MegaMoE communication modeling.  It
keeps the DeepGEMM fused-kernel runs isolated in subprocesses because the
underlying symmetric-memory cleanup can hang in some environments.  The public
contract is still a single command: sweep ``intermediate_hidden``, analyze the
fixed-routing curves, and write ``dsv4_megamoe_comm_path_perf.txt``.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    from collector.sglang.analyze_dsv4_megamoe_comm_sweep import (
        COMM_PATH_PERF_FILENAME,
        analyze_comm_sweep,
    )
except ModuleNotFoundError:
    _THIS_DIR = Path(__file__).resolve().parent
    sys.path.append(str(_THIS_DIR))
    from analyze_dsv4_megamoe_comm_sweep import (  # type: ignore[no-redef]
        COMM_PATH_PERF_FILENAME,
        analyze_comm_sweep,
    )


DEFAULT_TOKENS = "1,8,16,32,64,128,256,512,1024,2048,4096,8192"
DEFAULT_INTERMEDIATE_HIDDENS = "512,1024,1536,2048,2560,3072"
EFFECTIVE_COLLECTOR_FILENAME = "collect_dsv4_megamoe_effective_nvl_bw.py"


def _parse_int_csv(value: str, *, name: str) -> list[int]:
    values: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    if not values:
        raise ValueError(f"{name} must contain at least one integer")
    return values


def _csv(values: list[int]) -> str:
    return ",".join(str(value) for value in values)


def _resolve_device_name(explicit: str) -> str:
    if explicit:
        return explicit
    try:
        import torch

        return str(torch.cuda.get_device_name(0))
    except Exception:
        return "unknown"


def _resolve_effective_collector(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    return Path(__file__).resolve().parent / EFFECTIVE_COLLECTOR_FILENAME


def _intermediate_source(source: str, intermediate_hidden: int) -> str:
    if source:
        return f"{source}-intermediate-{intermediate_hidden}"
    return f"intermediate-{intermediate_hidden}"


def _build_effective_collect_command(
    *,
    python_executable: str,
    effective_collector: Path,
    output_dir: Path,
    args: argparse.Namespace,
    tokens_csv: str,
    intermediate_hidden: int,
) -> list[str]:
    command = [
        python_executable,
        str(effective_collector),
        "--num-processes",
        str(args.num_processes),
        "--num-max-tokens-per-rank",
        str(args.num_max_tokens_per_rank),
        "--num-tokens-list",
        tokens_csv,
        "--repeat-samples",
        str(args.repeat_samples),
        "--routing-mode",
        args.routing_mode,
        "--intermediate-hidden",
        str(intermediate_hidden),
        "--hidden",
        str(args.hidden_size),
        "--num-experts",
        str(args.num_experts),
        "--num-topk",
        str(args.topk),
        "--masked-ratio",
        str(args.masked_ratio),
        "--activation-clamp",
        str(args.activation_clamp),
        "--fast-math",
        str(args.fast_math),
        "--seed",
        str(args.seed),
        "--flush-l2",
        str(args.flush_l2),
        "--output",
        str(output_dir / "samples.json"),
        "--perf-output",
        str(output_dir / "dsv4_megamoe_effective_nvl_bw_perf.txt"),
        "--framework",
        args.framework,
        "--backend-version",
        args.backend_version,
        "--kernel-source",
        args.kernel_source,
        "--source",
        _intermediate_source(args.source, intermediate_hidden),
        "--hard-exit-after-write",
    ]
    if args.device_name:
        command.extend(["--device-name", args.device_name])
    if args.routing_mode == "power-law":
        command.extend(["--power-law-alpha", str(args.power_law_alpha)])
        if args.power_law_remap_hot_rank_to_zero:
            command.append("--power-law-remap-hot-rank-to-zero")
    return command


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_comm_path_collection(args: argparse.Namespace) -> dict[str, Any]:
    tokens = _parse_int_csv(args.num_tokens_list, name="--num-tokens-list")
    intermediate_hiddens = _parse_int_csv(args.intermediate_hiddens, name="--intermediate-hiddens")
    invalid_hiddens = [hidden for hidden in intermediate_hiddens if hidden % 512 != 0]
    if invalid_hiddens:
        raise ValueError(
            "DeepGEMM MegaMoE requires intermediate_hidden multiples of 512; "
            f"invalid values: {invalid_hiddens}"
        )
    if args.target_intermediate_hidden not in intermediate_hiddens:
        raise ValueError(
            "--target-intermediate-hidden must be included in --intermediate-hiddens "
            f"for target_fused_ms reporting, got {args.target_intermediate_hidden}"
        )

    output_dir = args.output_dir
    analysis_dir = args.analysis_output_dir or output_dir / "analysis"
    perf_output = args.perf_output or analysis_dir / COMM_PATH_PERF_FILENAME
    effective_collector = _resolve_effective_collector(args.effective_collector)
    if not effective_collector.exists():
        raise FileNotFoundError(f"effective collector not found: {effective_collector}")

    tokens_csv = _csv(tokens)
    commands: list[list[str]] = []
    for intermediate_hidden in intermediate_hiddens:
        intermediate_dir = output_dir / f"intermediate_{intermediate_hidden}"
        command = _build_effective_collect_command(
            python_executable=args.python_executable,
            effective_collector=effective_collector,
            output_dir=intermediate_dir,
            args=args,
            tokens_csv=tokens_csv,
            intermediate_hidden=intermediate_hidden,
        )
        commands.append(command)
        if args.dry_run:
            print(shlex.join(command), flush=True)
            continue
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        print(f"Collecting intermediate_hidden={intermediate_hidden}", flush=True)
        print(shlex.join(command), flush=True)
        subprocess.run(command, check=True)

    if args.dry_run:
        summary = {
            "dry_run": True,
            "output_dir": str(output_dir),
            "analysis_dir": str(analysis_dir),
            "perf_output": str(perf_output),
            "commands": [shlex.join(command) for command in commands],
        }
        _write_summary(output_dir / "collection_summary.json", summary)
        return summary

    device_name = _resolve_device_name(args.device_name)
    analysis_result = analyze_comm_sweep(
        input_dir=output_dir,
        output_dir=analysis_dir,
        plateau_tolerance_pct=args.plateau_tolerance_pct,
        target_intermediate_hidden=args.target_intermediate_hidden,
        perf_output=perf_output,
        framework=args.framework,
        backend_version=args.backend_version,
        device_name=device_name,
        kernel_source=args.kernel_source,
        hidden_size=args.hidden_size,
        topk=args.topk,
        num_experts=args.num_experts,
        moe_ep_size=args.num_processes,
        routing_mode=args.routing_mode,
        power_law_alpha=args.power_law_alpha,
        source=args.source or str(output_dir),
    )
    summary = {
        "dry_run": False,
        "output_dir": str(output_dir),
        "analysis_dir": str(analysis_dir),
        "perf_output": str(perf_output),
        "intermediate_hiddens": intermediate_hiddens,
        "tokens": tokens,
        "commands": [shlex.join(command) for command in commands],
        "analysis": analysis_result,
    }
    _write_summary(output_dir / "collection_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--num-max-tokens-per-rank", type=int, default=8192)
    parser.add_argument("--num-tokens-list", default=DEFAULT_TOKENS)
    parser.add_argument("--intermediate-hiddens", default=DEFAULT_INTERMEDIATE_HIDDENS)
    parser.add_argument("--target-intermediate-hidden", type=int, default=3072)
    parser.add_argument("--repeat-samples", type=int, default=5)
    parser.add_argument("--routing-mode", choices=("random", "power-law"), default="random")
    parser.add_argument("--power-law-alpha", type=float, default=1.01)
    parser.add_argument("--power-law-remap-hot-rank-to-zero", action="store_true")
    parser.add_argument("--hidden-size", type=int, default=7168)
    parser.add_argument("--num-experts", type=int, default=384)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--masked-ratio", type=float, default=0.0)
    parser.add_argument("--activation-clamp", type=float, default=10.0)
    parser.add_argument("--fast-math", type=int, default=1)
    parser.add_argument("--flush-l2", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plateau-tolerance-pct", type=float, default=5.0)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/dsv4_megamoe_comm_path_sweep/run"))
    parser.add_argument("--analysis-output-dir", type=Path, default=None)
    parser.add_argument("--perf-output", type=Path, default=None)
    parser.add_argument("--framework", default="SGLang")
    parser.add_argument("--backend-version", default="unknown")
    parser.add_argument("--device-name", default="")
    parser.add_argument("--kernel-source", default="DeepGEMM_fp8_fp4_mega_moe")
    parser.add_argument("--source", default="")
    parser.add_argument("--effective-collector", type=Path, default=None)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        summary = run_comm_path_collection(args)
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Wrote {summary['output_dir']}/collection_summary.json", flush=True)
    if not args.dry_run:
        print(f"Wrote {summary['perf_output']}", flush=True)


if __name__ == "__main__":
    main()
