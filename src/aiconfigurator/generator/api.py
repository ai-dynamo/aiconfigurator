# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
API layer for parameter alignment and validation.

This module provides a unified interface for collecting and validating parameters
from different input sources (function calls, CLI, etc.) and converting them
into the internal configuration format.
"""

import argparse
import logging
import os
import sys
from typing import Any, Optional

import yaml
from prettytable import PrettyTable

from .artifacts import ArtifactWriter
from .rendering import _cast_literal, render_backend_parameters, render_backend_templates
from .utils import DEFAULT_BACKEND, normalize_backend

GENERATOR_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "config")
DEFAULT_DEPLOYMENT_SCHEMA_PATH = os.path.join(GENERATOR_CONFIG_DIR, "deployment_config.yaml")
DEFAULT_BACKEND_MAPPING_PATH = os.path.join(GENERATOR_CONFIG_DIR, "backend_config_mapping.yaml")
_VALID_GENERATOR_HELP_SECTIONS = {"all", "deploy", "backend"}


def _load_yaml_payload(path: str) -> Any:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _format_default_value(value: Any) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, (int, float, bool, str)):
        return str(value)
    return yaml.safe_dump(value, default_flow_style=True).strip()


def _format_backend_list(backends: Any) -> str:
    if not backends:
        return "-"
    if isinstance(backends, str):
        return backends
    if isinstance(backends, (list, tuple, set)):
        parts = [str(item).strip() for item in backends if item is not None]
        return ", ".join(parts) if parts else "-"
    return str(backends)


def _format_backend_defaults(values: Any) -> str:
    if not isinstance(values, dict):
        return "-"
    parts = [f"{k!s}={v!s}" for k, v in values.items()]
    return " | ".join(parts) if parts else "-"


def _format_mapping_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, float, bool, str)):
        return str(value)
    if isinstance(value, dict):
        key = value.get("key")
        val = value.get("value")
        default = value.get("default")
        segments: list[str] = []
        if key and val is not None:
            segments.append(f"{key} <- {val}")
        elif key:
            segments.append(str(key))
        elif val is not None:
            segments.append(str(val))
        if default is not None:
            segments.append(f"default={default}")
        return " | ".join(segments) if segments else "-"
    return str(value)


def _entry_matches_backend(entry: dict[str, Any], backend: Optional[str]) -> bool:
    if not backend:
        return True
    allowed = entry.get("backends")
    if not allowed:
        return True
    if isinstance(allowed, str):
        allowed_set = {allowed.strip().lower()}
    elif isinstance(allowed, (list, tuple, set)):
        allowed_set = {str(item).strip().lower() for item in allowed if item}
    else:
        allowed_set = set()
    return not allowed_set or backend in allowed_set


def _build_inputs_table(schema_path: str, backend: Optional[str]) -> PrettyTable:
    payload = _load_yaml_payload(schema_path)
    if isinstance(payload, list):
        inputs = payload
    else:
        inputs = payload.get("inputs", [])
    table = PrettyTable()
    table.field_names = ["Key", "Required", "Default", "Backends", "Backend defaults"]
    table.align["Key"] = "l"
    backend_key = normalize_backend(backend, DEFAULT_BACKEND) if backend else None
    rows_added = False
    for entry in inputs:
        key = entry.get("key", "-")
        if not key:
            continue
        if not _entry_matches_backend(entry, backend_key):
            continue
        table.add_row(
            [
                key,
                "yes" if entry.get("required") else "",
                _format_default_value(entry.get("default")),
                _format_backend_list(entry.get("backends")),
                _format_backend_defaults(entry.get("backend_defaults")),
            ]
        )
        rows_added = True
    if not rows_added:
        table.add_row(["-", "-", "-", "-", "-"])
    return table


def _collect_mapping_backends(parameters: list[dict[str, Any]]) -> list[str]:
    discovered: set[str] = set()
    for entry in parameters:
        for key in entry:
            if key not in {"param_key", "description"}:
                discovered.add(str(key))
    preferred_order = ["trtllm", "vllm", "sglang"]
    ordered = [name for name in preferred_order if name in discovered]
    ordered.extend(sorted(name for name in discovered if name not in preferred_order))
    return ordered


def _build_mapping_table(mapping_path: str, backend: Optional[str]) -> PrettyTable:
    payload = _load_yaml_payload(mapping_path)
    if isinstance(payload, list):
        parameters = payload
    else:
        parameters = payload.get("parameters", [])
    backend_key = normalize_backend(backend, DEFAULT_BACKEND) if backend else None
    if backend_key:
        backend_names = [backend_key]
    else:
        backend_names = _collect_mapping_backends(parameters)
    table = PrettyTable()
    headers = ["Param key"] + list(backend_names)
    table.field_names = headers
    table.align["Param key"] = "l"
    rows_added = False
    for entry in parameters:
        param_key = entry.get("param_key", "-")
        row = [param_key]
        for backend_name in backend_names:
            row.append(_format_mapping_value(entry.get(backend_name)))
        table.add_row(row)
        rows_added = True
    if not rows_added:
        table.add_row(["-", *["-"] * (len(headers) - 1)])
    return table


def print_generator_help(
    section: str = "all",
    backend: Optional[str] = None,
    *,
    schema_path: str = DEFAULT_DEPLOYMENT_SCHEMA_PATH,
    mapping_path: str = DEFAULT_BACKEND_MAPPING_PATH,
    stream=None,
) -> None:
    """
    Print generator configuration reference tables or raw deployment schema.

    Args:
        section: one of {"deploy", "backend", "all"}.
        backend: optional backend filter (e.g., "trtllm").
        schema_path: deployment schema path (tests can override).
        mapping_path: backend mapping path (tests can override).
        stream: destination stream (defaults to sys.stdout).
    """
    section_lower = (section or "all").lower()
    if section_lower not in _VALID_GENERATOR_HELP_SECTIONS:
        raise ValueError(f"Unsupported generator help section: {section}")
    stream = stream or sys.stdout
    blocks: list[str] = []
    if section_lower in {"deploy", "all"}:
        # Show the complete deployment_config.yaml so users can see the full schema.
        payload = _load_yaml_payload(schema_path)
        deploy_yaml = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
        blocks.append(f"Deployment schema ({schema_path}):\n{deploy_yaml}")
    if section_lower in {"backend", "all"}:
        mapping_table = _build_mapping_table(mapping_path, backend)
        blocks.append(f"Backend parameter mappings ({mapping_path}):\n{mapping_table}")
    stream.write("\n\n".join(blocks) + "\n")


def generator_cli_helper(argv: list[str]) -> bool:
    """
    Inspect argv for generator help flags, print the tables, and exit early.

    Returns True if help was printed (caller should skip normal parsing).
    """
    helper = argparse.ArgumentParser(add_help=False)
    helper.add_argument(
        "--generator-help",
        nargs="?",
        const="all",
        default=None,
        dest="_generator_help_section",
    )
    helper.add_argument(
        "--generator-help-backend",
        default=None,
        dest="_generator_help_backend",
    )
    helper.add_argument(
        "--backend",
        default=None,
        dest="_generator_backend_choice",
    )
    try:
        parsed, _ = helper.parse_known_args(argv)
    except SystemExit:
        return False
    section = parsed._generator_help_section
    if section is None:
        return False
    backend = parsed._generator_help_backend or parsed._generator_backend_choice
    print_generator_help(section=section, backend=backend)
    return True


def generate_backend_config(
    params: dict[str, Any], backend: str, mapping_path: Optional[str] = None
) -> dict[str, dict[str, Any]]:
    """
    Generate backend-specific configuration from parameters.

    Args:
        params: Complete parameter configuration
        backend: Target backend name (e.g., 'sglang', 'vllm')
        mapping_path: Optional path to mapping YAML file

    Returns:
        Backend-specific configuration dict
    """
    return render_backend_parameters(params, backend, yaml_path=mapping_path)


def generate_backend_artifacts(
    params: dict[str, Any],
    role_backends: dict[str, str],
    role_versions: Optional[dict[str, str]] = None,
    templates_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> dict[str, str]:
    """
    Generate complete backend artifacts including run scripts, configs, and k8s YAML.

    Args:
        params: Complete parameter configuration
        role_backends: Mapping of role (prefill/decode/agg) to backend name
        role_versions: Optional mapping of role to version string
        templates_dir: Optional directory containing templates
        output_dir: Optional directory to save generated files

    Returns:
        Dictionary mapping artifact names to their content
    """
    logger = logging.getLogger(__name__)
    artifacts = render_backend_templates(
        params,
        role_backends=role_backends,
        role_versions=role_versions,
        templates_dir=templates_dir,
    )

    if output_dir:
        params_obj = params.get("params", {})
        has_prefill = bool(params_obj.get("prefill"))
        has_decode = bool(params_obj.get("decode"))
        has_agg = bool(params_obj.get("agg"))
        prefer_disagg = has_prefill and has_decode
        writer = ArtifactWriter(
            output_dir=os.path.abspath(output_dir),
            prefer_disagg=prefer_disagg,
            has_agg_role=has_agg,
        )
        try:
            writer.write(artifacts)
        except OSError:
            logger.exception("Failed to write artifacts")

    return artifacts


# CLI Interface Functions
def parse_cli_params(argv: list[str]) -> dict[str, Any]:
    """
    Parse command-line parameters in key=value format.

    Args:
        argv: List of command-line arguments

    Returns:
        Dictionary of parsed parameters
    """
    cli_params: dict[str, Any] = {}
    for item in argv:
        if "=" not in item:
            continue
        key, val = item.split("=", 1)
        _assign_path(cli_params, key.strip(), _cast_literal(val))
    return cli_params


def add_generator_override_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Attach generator override arguments to an argparse parser.

    Args:
        parser: Target ArgumentParser that should receive the shared options.
    """
    grp = parser.add_argument_group(
        "Generator overrides",
        "Options forwarded to the generator. "
        "Use dotted keys (e.g. ServiceConfig.model_path=Qwen/Qwen3-32B-FP8). "
        "See generator config docs for the available keys.",
    )
    grp.add_argument(
        "--generator-config",
        type=str,
        default=None,
        help="Path to a unified generator YAML file.",
    )
    grp.add_argument(
        "--generator-set",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Inline overrides for generator config (repeatable).",
    )
    grp.add_argument(
        "--generator-help-backend",
        dest="generator_help_backend",
        type=str,
        default=None,
        help="Filter --generator-help output to a backend (e.g. trtllm, vllm, sglang).",
    )
    grp.add_argument(
        "--generator-help",
        nargs="?",
        const="all",
        default=None,
        metavar="SECTION",
        help="Print generator schema help (deploy, backend, or all) and exit.",
    )
    grp.add_argument(
        "--generated_config_version",
        type=str,
        default=None,
        help="Backend version for workers in agg mode and prefill workers in disagg mode "
        "for deployment artifact generation (e.g. 1.1.0rc5). "
        "Defaults to --backend_version if omitted.",
    )
    grp.add_argument(
        "--generated_decode_config_version",
        type=str,
        default=None,
        help=(
            "Backend version for disagg decode workers in deployment artifact generation. "
            "Defaults to --generated_config_version if omitted."
        ),
    )


def load_generator_overrides(
    config_path: Optional[str],
    inline_overrides: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    Load generator overrides from a YAML file and optional inline CLI overrides.

    Args:
        config_path: Optional path to a YAML file containing overrides.
        inline_overrides: Optional list of dotted KEY=VALUE strings.
    """
    config_payload: dict[str, Any] = {}
    if config_path:
        expanded = os.path.abspath(config_path)
        with open(expanded, encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
            if not isinstance(loaded, dict):
                raise TypeError("--generator-config must point to a YAML mapping.")
            config_payload = loaded

    inline_payload = parse_cli_params(inline_overrides or [])
    if not inline_payload:
        return config_payload
    return _deep_merge_dicts(config_payload, inline_payload)


def load_generator_overrides_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """
    Convenience wrapper that pulls generator override fields from an argparse namespace.
    """
    return load_generator_overrides(
        getattr(args, "generator_config", None),
        getattr(args, "generator_set", None),
    )


def parse_backend_arg(argv: list[str]) -> Optional[str]:
    """
    Extract backend argument from command-line arguments.

    Args:
        argv: List of command-line arguments

    Returns:
        Backend name if found, None otherwise
    """
    for item in argv:
        if item.startswith("backend="):
            _, val = item.split("=", 1)
            return val.strip()
    return None


def parse_mapping_arg(argv: list[str]) -> Optional[str]:
    """
    Extract mapping argument from command-line arguments.

    Args:
        argv: List of command-line arguments

    Returns:
        Mapping path if found, None otherwise
    """
    for item in argv:
        if item.startswith("mapping="):
            _, val = item.split("=", 1)
            return val.strip()
    return None


def resolve_mapping_yaml(mapping_arg: Optional[str], default_mapping_path: str) -> str:
    """
    Resolve mapping YAML file path from argument or default location.

    Args:
        mapping_arg: Optional mapping path from command line
        default_mapping_path: Default mapping file path

    Returns:
        Absolute path to mapping YAML file

    Raises:
        FileNotFoundError: If mapping file cannot be found
    """
    if mapping_arg:
        candidate = mapping_arg
        if not os.path.isabs(candidate):
            candidate = os.path.join(os.getcwd(), candidate)
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
        raise FileNotFoundError(f"Mapping file not found: {candidate}")

    if os.path.isfile(default_mapping_path):
        return os.path.abspath(default_mapping_path)

    raise FileNotFoundError(
        f"Cannot resolve mapping YAML. Expected at {default_mapping_path} or provided via mapping=<path>."
    )


def prepare_generator_params(
    config_path: Optional[str],
    overrides: Optional[dict[str, Any]] = None,
    schema_path: Optional[str] = None,
    backend: Optional[str] = None,
) -> dict[str, Any]:
    """
    Load generator inputs from YAML (if provided), apply CLI overrides, and emit normalized params.

    Args:
        config_path: Optional path to a unified config YAML.
        overrides: Inline overrides parsed from CLI.
        schema_path: Optional alternative schema file.
        backend: Backend name used for backend-scoped defaults (e.g., trtllm, vllm, sglang).
    """
    raw_config: dict[str, Any] = {}

    if config_path:
        if not os.path.isabs(config_path):
            config_path = os.path.abspath(config_path)
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, encoding="utf-8") as f:
            raw_config = yaml.safe_load(f) or {}

    if overrides:
        raw_config = _deep_merge_dicts(raw_config, overrides)

    if not raw_config:
        raise ValueError("No generator inputs provided via --config or --set.")

    return generate_config_from_input_dict(raw_config, schema_path=schema_path, backend=backend)


def _assign_path(target: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = [p for p in dotted_key.split(".") if p]
    if not parts:
        return
    node = target
    for segment in parts[:-1]:
        next_node = node.setdefault(segment, {})
        if not isinstance(next_node, dict):
            next_node = {}
            node[segment] = next_node
        node = next_node
    node[parts[-1]] = value


def _deep_merge_dicts(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


from .aggregators import (
    collect_generator_params as collect_generator_params,
)
from .aggregators import (
    generate_config_from_input_dict as generate_config_from_input_dict,
)
from .aggregators import (
    generate_config_from_yaml as generate_config_from_yaml,
)
from .naive import build_naive_generator_params


def generate_naive_config(
    model_path: str,
    total_gpus: int,
    system: str,
    backend: str = "trtllm",
    output_dir: str | None = None,
    backend_version: str | None = None,
) -> dict[str, Any]:
    """
    Generate a naive agg configuration without SLA optimization.

    Calculates the smallest TP that fits the model in memory:
    TP * VRAM_per_GPU > 1.5 * model_weight_size. No parameter sweeping
    or memory validation is performed.

    Args:
        model_path: HuggingFace model path (e.g., 'Qwen/Qwen3-32B') or
            local path to directory containing config.json.
        total_gpus: Total number of GPUs for deployment.
        system: System name (GPU type), e.g., 'h200_sxm', 'gb200_sxm'.
        backend: Backend name ('trtllm', 'vllm', 'sglang'). Defaults to 'trtllm'.
        output_dir: Directory to save generated artifacts. If None, artifacts
            are generated but not saved to disk.
        backend_version: Optional backend version for version-specific templates.
            If None, uses the latest available version.

    Returns:
        Dictionary containing:
            - 'generator_params': The generator parameter configuration
            - 'artifacts': Dict mapping artifact names to their content
            - 'output_dir': Path where artifacts were saved (if output_dir provided)
            - 'backend_version': The backend version used
            - 'parallelism': Dict with 'tp', 'pp', 'replicas' info

    Example:
        >>> from aiconfigurator.generator.api import generate_naive_config
        >>> result = generate_naive_config(
        ...     model_path="Qwen/Qwen3-32B",
        ...     total_gpus=8,
        ...     system="h200_sxm",
        ...     backend="trtllm",
        ...     output_dir="./output",
        ... )
        >>> print(result['parallelism'])
        {'tp': 1, 'pp': 1, 'replicas': 8, 'gpus_used': 8}
    """
    from aiconfigurator.sdk.perf_database import get_latest_database_version
    from aiconfigurator.sdk.utils import safe_mkdir

    logger = logging.getLogger(__name__)

    # Build generator parameters
    generator_params = build_naive_generator_params(
        model_name=model_path,
        total_gpus=total_gpus,
        system_name=system,
        backend_name=backend,
    )

    # Determine backend version
    if backend_version is None:
        backend_version = get_latest_database_version(system=system, backend=backend)
    logger.info("Using backend version: %s", backend_version)

    # Prepare output directory if provided
    actual_output_dir = None
    if output_dir:
        import random

        model_name_safe = model_path.replace("/", "_")
        tp = generator_params["params"]["agg"]["tensor_parallel_size"]
        pp = generator_params["params"]["agg"]["pipeline_parallel_size"]
        result_dir = os.path.join(
            output_dir,
            f"{model_name_safe}_naive_tp{tp}_pp{pp}_{random.randint(0, 999999):06d}",
        )
        actual_output_dir = safe_mkdir(result_dir, exist_ok=True)
        logger.info("Saving results to %s", actual_output_dir)

        # Save generator config
        generator_config_path = os.path.join(actual_output_dir, "generator_config.yaml")
        with open(generator_config_path, "w") as f:
            yaml.safe_dump(generator_params, f, sort_keys=False)
        logger.info("Saved generator config to %s", generator_config_path)

    # Generate backend artifacts
    role_backends = {}
    params_obj = generator_params.get("params", {})
    for role in ("prefill", "decode", "agg"):
        if params_obj.get(role):
            role_backends[role] = backend

    role_versions = dict.fromkeys(role_backends, backend_version) if backend_version else None

    artifacts = render_backend_templates(
        generator_params,
        role_backends=role_backends,
        role_versions=role_versions,
    )

    if actual_output_dir:
        params_obj = generator_params.get("params", {})
        has_prefill = bool(params_obj.get("prefill"))
        has_decode = bool(params_obj.get("decode"))
        has_agg = bool(params_obj.get("agg"))
        prefer_disagg = has_prefill and has_decode
        writer = ArtifactWriter(
            output_dir=os.path.abspath(actual_output_dir),
            prefer_disagg=prefer_disagg,
            has_agg_role=has_agg,
        )
        try:
            writer.write(artifacts)
        except OSError:
            logger.exception("Failed to write artifacts")

    logger.info("Generated %d artifacts", len(artifacts))

    # Build result metadata
    tp = generator_params["params"]["agg"]["tensor_parallel_size"]
    pp = generator_params["params"]["agg"]["pipeline_parallel_size"]
    replicas = total_gpus // (tp * pp)
    gpus_used = tp * pp * replicas

    return {
        "generator_params": generator_params,
        "artifacts": artifacts,
        "output_dir": actual_output_dir,
        "backend_version": backend_version,
        "parallelism": {
            "tp": tp,
            "pp": pp,
            "replicas": replicas,
            "gpus_used": gpus_used,
        },
    }


__all__ = [
    "add_generator_override_arguments",
    "collect_generator_params",
    "generate_backend_artifacts",
    "generate_backend_config",
    "generate_config_from_input_dict",
    "generate_config_from_yaml",
    "generate_naive_config",
    "generator_cli_helper",
    "load_generator_overrides",
    "load_generator_overrides_from_args",
    "parse_backend_arg",
    "parse_cli_params",
    "parse_mapping_arg",
    "prepare_generator_params",
    "print_generator_help",
    "resolve_mapping_yaml",
]
