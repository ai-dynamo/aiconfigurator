# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import json
from contextlib import redirect_stderr
from pathlib import Path
from typing import Any, Optional

import yaml


def _import_vllm_engine_args():
    from vllm.engine.arg_utils import EngineArgs
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    return EngineArgs, FlexibleArgumentParser


def _dict_to_cli_args(payload: dict[str, Any]) -> list[str]:
    args: list[str] = []
    for key, value in payload.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                args.append(flag)
        elif isinstance(value, list):
            if value:
                args.append(flag)
                args.extend([str(item) for item in value])
        elif isinstance(value, dict):
            args.append(flag)
            args.append(json.dumps(value))
        elif value is not None:
            args.append(flag)
            args.append(str(value))
    return args


def _normalize_cli_args(args: list[str]) -> list[str]:
    ignored_flags = {"--is-prefill-worker", "--is-decode-worker"}
    return [str(item) for item in args if str(item) not in ignored_flags]


def _parse_args_with_errors(parser, args: list[str], context: str):
    buf = io.StringIO()
    try:
        with redirect_stderr(buf):
            parsed, unknown = parser.parse_known_args(args)
        if unknown:
            raise ValueError(f"{context} Unrecognized arguments: {' '.join(unknown)}")
        return parsed
    except SystemExit:
        detail = buf.getvalue().strip() or "Invalid vLLM args."
        raise ValueError(f"{context} {detail}") from None


def _load_yaml_payload(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        payload = yaml.safe_load(fh) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping in YAML file {path}, got {type(payload).__name__}")
    return payload


def _extract_vllm_cli_args(
    payload: dict[str, Any],
    service_key: Optional[str],
) -> Optional[list[str]]:
    try:
        services = payload.get("spec", {}).get("services", {})
        worker = services.get(service_key or "VLLMWorker", {})
        container = worker.get("extraPodSpec", {}).get("mainContainer", {})
        args = container.get("args")
    except AttributeError:
        return None
    if not args:
        return None
    if not isinstance(args, list):
        raise TypeError("VLLMWorker.extraPodSpec.mainContainer.args must be a list.")
    return [str(item) for item in args]


def collect_config_paths(root_dir: Path) -> list[tuple[str, Path, str]]:
    agg_path = root_dir / "agg" / "top1" / "k8s_deploy.yaml"
    disagg_path = root_dir / "disagg" / "top1" / "k8s_deploy.yaml"
    missing = []
    if not agg_path.exists():
        missing.append("agg/top1/k8s_deploy.yaml")
    if not disagg_path.exists():
        missing.append("disagg/top1/k8s_deploy.yaml")
    if missing:
        raise ValueError(
            f"Missing expected config under {root_dir}: {', '.join(missing)}."
        )
    return [
        ("agg", agg_path, "VLLMWorker"),
        ("prefill", disagg_path, "VLLMPrefillWorker"),
        ("decode", disagg_path, "VLLMDecodeWorker"),
    ]


def validate_vllm_engine_args(
    engine_args: dict[str, Any],
    model_path: Optional[str] = None,
):
    engine_args_cls, flexible_parser_cls = _import_vllm_engine_args()
    payload = dict(engine_args)
    if not payload.get("model"):
        payload["model"] = model_path or "dummy-model"
    parser = flexible_parser_cls(add_help=False)
    engine_args_cls.add_cli_args(parser)
    parsed = _parse_args_with_errors(
        parser,
        _dict_to_cli_args(payload),
        "Invalid vLLM args in config payload.",
    )
    engine_args_obj = engine_args_cls.from_cli_args(parsed)
    return engine_args_obj.create_engine_config()


def validate_vllm_engine_args_from_cli(cli_args: list[str]):
    engine_args_cls, flexible_parser_cls = _import_vllm_engine_args()
    parser = flexible_parser_cls(add_help=False)
    engine_args_cls.add_cli_args(parser)
    parsed = _parse_args_with_errors(
        parser,
        _normalize_cli_args(cli_args),
        "Invalid vLLM CLI args in k8s_deploy.yaml.",
    )
    engine_args_obj = engine_args_cls.from_cli_args(parsed)
    return engine_args_obj.create_engine_config()


def validate_vllm_engine_config_file(
    engine_config_path: str,
    model_path: Optional[str],
    service_key: Optional[str] = None,
) -> tuple[Any, Optional[str]]:
    engine_args = _load_yaml_payload(engine_config_path)
    cli_args = _extract_vllm_cli_args(engine_args, service_key)
    if cli_args is not None:
        vllm_config = validate_vllm_engine_args_from_cli(cli_args)
    else:
        vllm_config = validate_vllm_engine_args(engine_args, model_path)
    return vllm_config, model_path
