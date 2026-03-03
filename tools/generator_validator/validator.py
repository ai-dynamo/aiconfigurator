# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from importlib import import_module
from pathlib import Path
from typing import Optional


def _load_backend_handlers():
    try:
        from .backend import sglang as sglang_backend
        from .backend import trtllm as trtllm_backend
        from .backend import vllm as vllm_backend

        return {
            "trtllm": {
                "title": "TRT-LLM Config Validation (PyTorch backend)",
                "collect": trtllm_backend.collect_config_paths,
                "validate": trtllm_backend.validate_torchllm_engine_config_file,
            },
            "vllm": {
                "title": "vLLM Config Validation",
                "collect": vllm_backend.collect_config_paths,
                "validate": vllm_backend.validate_vllm_engine_config_file,
            },
            "sglang": {
                "title": "SGLang Config Validation",
                "collect": sglang_backend.collect_config_paths,
                "validate": sglang_backend.validate_sglang_engine_config_file,
            },
        }
    except ImportError:  # Allow running as a script without package context.
        tools_root = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(tools_root))
        trtllm_backend = import_module("generator_validator.backend.trtllm")
        vllm_backend = import_module("generator_validator.backend.vllm")
        sglang_backend = import_module("generator_validator.backend.sglang")
        return {
            "trtllm": {
                "title": "TRT-LLM Config Validation (PyTorch backend)",
                "collect": trtllm_backend.collect_config_paths,
                "validate": trtllm_backend.validate_torchllm_engine_config_file,
            },
            "vllm": {
                "title": "vLLM Config Validation",
                "collect": vllm_backend.collect_config_paths,
                "validate": vllm_backend.validate_vllm_engine_config_file,
            },
            "sglang": {
                "title": "SGLang Config Validation",
                "collect": sglang_backend.collect_config_paths,
                "validate": sglang_backend.validate_sglang_engine_config_file,
            },
        }


BACKEND_HANDLERS = _load_backend_handlers()

logger = logging.getLogger(__name__)

ANSI_GREEN = "\033[32m"
ANSI_RED = "\033[31m"
ANSI_RESET = "\033[0m"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate engine configs for TRT-LLM, vLLM, or SGLang.")
    parser.add_argument(
        "--backend",
        required=True,
        choices=["trtllm", "vllm", "sglang"],
        help="Which engine backend to validate.",
    )
    parser.add_argument(
        "--path",
        required=True,
        help=(
            "Path to either a single engine YAML or a root results directory. "
            "For TRT-LLM, a directory uses agg/top1/agg_config.yaml and "
            "disagg/top1/{decode,prefill}_config.yaml. For vLLM, a directory "
            "uses agg/top1/k8s_deploy.yaml."
        ),
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional model path for validation (not used to load files).",
    )
    parser.add_argument(
        "--show-traceback",
        action="store_true",
        help="Include full tracebacks for failed validations.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO)
    parser = _build_parser()
    args = parser.parse_args(argv)
    user_path = Path(args.path).resolve()
    handler = BACKEND_HANDLERS[args.backend]

    try:
        if user_path.is_dir():
            engine_configs = handler["collect"](user_path)
        else:
            engine_configs = [("engine", user_path)]
    except Exception as exc:
        print("\n" + "=" * 72)
        print(handler["title"])
        print("=" * 72)
        print(f"Error: {exc}")
        print("=" * 72 + "\n")
        return 1

    failures: list[tuple[str, str]] = []
    resolved_model = None
    for entry in engine_configs:
        label = entry[0]
        path = entry[1]
        service_key = entry[2] if len(entry) > 2 else None
        try:
            _, resolved_model = handler["validate"](
                str(path),
                model_path=args.model_path,
                service_key=service_key,
            )
        except Exception as exc:
            if args.show_traceback:
                msg = "".join(traceback.format_exception(exc)).rstrip()
            else:
                msg = f"{type(exc).__name__}: {exc}"
            failures.append((label, msg))

    print("\n" + "=" * 72)
    print(handler["title"])
    print("=" * 72)
    if resolved_model:
        print(f"Model path       : {resolved_model}")
    print("Checked files    :")
    for entry in engine_configs:
        label = entry[0]
        path = entry[1]
        print(f"  - {label:<7} -> {path}")
    if failures:
        print(f"Result           : {ANSI_RED}FAIL{ANSI_RESET}")
        print("Failures         :")
        for label, msg in failures:
            print(f"  - {label}: {msg}")
        print("=" * 72 + "\n")
        return 1
    print(f"Result           : {ANSI_GREEN}PASS{ANSI_RESET}")
    print("=" * 72 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
