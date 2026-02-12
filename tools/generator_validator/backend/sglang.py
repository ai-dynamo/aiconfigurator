# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import io
import re
import shlex
from contextlib import redirect_stderr
from pathlib import Path
from typing import Any, Optional

import yaml


def _import_sglang_srt_args():
    from sglang.srt.server_args import ServerArgs as SrtServerArgs

    return SrtServerArgs


def _load_yaml_payload(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        payload = yaml.safe_load(fh) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping in YAML file {path}, got {type(payload).__name__}")
    return payload


def _parse_args_with_errors(parser, args: list[str], context: str):
    buf = io.StringIO()
    try:
        with redirect_stderr(buf):
            parsed, unknown = parser.parse_known_args(args)
        if unknown:
            raise ValueError(f"{context} Unrecognized arguments: {' '.join(unknown)}")
        return parsed
    except SystemExit:
        detail = buf.getvalue().strip() or "Invalid SGLang args."
        raise ValueError(f"{context} {detail}") from None


def _extract_cli_args_from_script(script: str) -> list[str]:
    tokens: list[str] = []
    lines = script.splitlines()
    capture = False
    block_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not capture and stripped.startswith("args=("):
            capture = True
            remainder = stripped[len("args=(") :].strip()
            if remainder and remainder != ")":
                block_lines.append(remainder)
            if remainder.endswith(")"):
                capture = False
                block_lines[-1] = remainder[:-1].strip()
        elif capture:
            if stripped == ")":
                capture = False
            else:
                block_lines.append(stripped)

    for line in block_lines:
        if not line:
            continue
        tokens.extend(shlex.split(line))

    for match in re.finditer(r"args\+\=\((.*?)\)", script, flags=re.DOTALL):
        content = match.group(1).strip()
        if not content:
            continue
        tokens.extend(shlex.split(content))

    return tokens


def _extract_sglang_cli_args(
    payload: dict[str, Any],
    service_key: Optional[str],
) -> Optional[list[str]]:
    try:
        services = payload.get("spec", {}).get("services", {})
        worker = services.get(service_key or "SGLangWorker", {})
        container = worker.get("extraPodSpec", {}).get("mainContainer", {})
        args = container.get("args")
    except AttributeError:
        return None
    if not args:
        return None
    if not isinstance(args, list):
        raise TypeError("SGLangWorker.extraPodSpec.mainContainer.args must be a list.")
    if len(args) == 1 and isinstance(args[0], str) and "args=(" in args[0]:
        return _extract_cli_args_from_script(args[0])
    return [str(item) for item in args]


def _find_model_path(cli_args: list[str]) -> Optional[str]:
    for idx, item in enumerate(cli_args):
        if item in ("--model-path", "--model") and idx + 1 < len(cli_args):
            next_val = cli_args[idx + 1]
            if not next_val.startswith("--"):
                return next_val
    return None


def _normalize_model_path(
    cli_args: list[str],
    model_path: Optional[str],
    *,
    default_model: str = "dummy",
) -> list[str]:
    args = list(cli_args)
    replacement = model_path or default_model
    for idx, item in enumerate(args):
        if item in ("--model-path", "--model"):
            if idx + 1 < len(args) and not str(args[idx + 1]).startswith("--"):
                args[idx + 1] = replacement
            else:
                args.insert(idx + 1, replacement)
            return args
    args.extend(["--model-path", replacement])
    return args


def collect_config_paths(root_dir: Path) -> list[tuple[str, Path, str]]:
    agg_path = root_dir / "agg" / "top1" / "k8s_deploy.yaml"
    disagg_path = root_dir / "disagg" / "top1" / "k8s_deploy.yaml"
    missing = []
    if not agg_path.exists():
        missing.append("agg/top1/k8s_deploy.yaml")
    if not disagg_path.exists():
        missing.append("disagg/top1/k8s_deploy.yaml")
    if missing:
        raise ValueError(f"Missing expected config under {root_dir}: {', '.join(missing)}.")
    return [
        ("agg", agg_path, "SGLangWorker"),
        ("prefill", disagg_path, "SGLangPrefillWorker"),
        ("decode", disagg_path, "SGLangDecodeWorker"),
    ]


def validate_sglang_engine_args_from_cli(
    cli_args: list[str],
    model_path: Optional[str] = None,
):
    original_model = _find_model_path(cli_args)
    normalized_args = _normalize_model_path(cli_args, model_path)

    srt_server_args_cls = _import_sglang_srt_args()
    parser = argparse.ArgumentParser(add_help=False)
    srt_server_args_cls.add_cli_args(parser)
    parsed = _parse_args_with_errors(
        parser,
        normalized_args,
        "Invalid SGLang CLI args (SRT server).",
    )
    return parsed, (model_path or original_model)


def validate_sglang_engine_config_file(
    engine_config_path: str,
    model_path: Optional[str],
    service_key: Optional[str] = None,
) -> tuple[Any, Optional[str]]:
    engine_args = _load_yaml_payload(engine_config_path)
    cli_args = _extract_sglang_cli_args(engine_args, service_key)
    if cli_args is None:
        raise ValueError(
            "Unable to locate SGLang CLI args in k8s_deploy.yaml "
            "(expected spec.services.<SGLANG*>.extraPodSpec.mainContainer.args)."
        )
    parsed, resolved_model = validate_sglang_engine_args_from_cli(cli_args, model_path)
    return parsed, resolved_model
