"""
API layer for parameter alignment and validation.

This module provides a unified interface for collecting and validating parameters
from different input sources (function calls, CLI, etc.) and converting them
into the internal configuration format.
"""

import argparse
import logging
import os
from typing import Any, Dict, List, Optional

import yaml

from .artifacts import ArtifactWriter
from .rendering import render_backend_parameters, render_backend_templates, _cast_literal

def generate_backend_config(
    params: Dict[str, Any],
    backend: str,
    mapping_path: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
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
    params: Dict[str, Any],
    backend: str,
    templates_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    backend_version: Optional[str] = None
) -> Dict[str, str]:
    """
    Generate complete backend artifacts including run scripts, configs, and k8s YAML.
    
    Args:
        params: Complete parameter configuration
        backend: Target backend name (e.g., 'trtllm', 'vllm', 'sglang')
        templates_dir: Optional directory containing templates
        output_dir: Optional directory to save generated files
        backend_version: Optional version string for version-specific template selection
        
    Returns:
        Dictionary mapping artifact names to their content
    """
    logger = logging.getLogger(__name__)
    artifacts = render_backend_templates(params, backend, templates_dir, backend_version)
    
    if output_dir:
        params_obj = params.get('params', {})
        has_prefill = bool(params_obj.get('prefill'))
        has_decode = bool(params_obj.get('decode'))
        has_agg = bool(params_obj.get('agg'))
        prefer_disagg = has_prefill and has_decode
        writer = ArtifactWriter(
            output_dir=os.path.abspath(output_dir),
            prefer_disagg=prefer_disagg,
            has_agg_role=has_agg,
        )
        try:
            writer.write(artifacts)
        except OSError as exc:
            logger.error("Failed to write artifacts: %s", exc)
    
    return artifacts



# CLI Interface Functions
def parse_cli_params(argv: List[str]) -> Dict[str, Any]:
    """
    Parse command-line parameters in key=value format.
    
    Args:
        argv: List of command-line arguments
        
    Returns:
        Dictionary of parsed parameters
    """
    cli_params: Dict[str, Any] = {}
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
        "Use dotted keys (e.g. ServiceConfig.model_path=Qwen/Qwen3-32B-FP8), please refer to aiconfigurator/src/aiconfigurator/generator/config for the available keys.",
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
        "--generated_config_version",
        type=str,
        default=None,
        help="Backend template version for generated artifacts (e.g. 1.1.0rc5).",
    )


def load_generator_overrides(
    config_path: Optional[str],
    inline_overrides: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Load generator overrides from a YAML file and optional inline CLI overrides.

    Args:
        config_path: Optional path to a YAML file containing overrides.
        inline_overrides: Optional list of dotted KEY=VALUE strings.
    """
    config_payload: Dict[str, Any] = {}
    if config_path:
        expanded = os.path.abspath(config_path)
        with open(expanded, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
            if not isinstance(loaded, dict):
                raise ValueError("--generator-config must point to a YAML mapping.")
            config_payload = loaded

    inline_payload = parse_cli_params(inline_overrides or [])
    if not inline_payload:
        return config_payload
    return _deep_merge_dicts(config_payload, inline_payload)


def load_generator_overrides_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Convenience wrapper that pulls generator override fields from an argparse namespace.
    """
    return load_generator_overrides(
        getattr(args, "generator_config", None),
        getattr(args, "generator_set", None),
    )


def parse_backend_arg(argv: List[str]) -> Optional[str]:
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


def parse_mapping_arg(argv: List[str]) -> Optional[str]:
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
    overrides: Optional[Dict[str, Any]] = None,
    schema_path: Optional[str] = None,
    backend: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load generator inputs from YAML (if provided), apply CLI overrides, and emit normalized params.

    Args:
        config_path: Optional path to a unified config YAML.
        overrides: Inline overrides parsed from CLI.
        schema_path: Optional alternative schema file.
        backend: Backend name used for backend-scoped defaults (e.g., trtllm, vllm, sglang).
    """
    raw_config: Dict[str, Any] = {}

    if config_path:
        if not os.path.isabs(config_path):
            config_path = os.path.abspath(config_path)
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f) or {}

    if overrides:
        raw_config = _deep_merge_dicts(raw_config, overrides)

    if not raw_config:
        raise ValueError("No generator inputs provided via --config or --set.")

    return generate_config_from_input_dict(raw_config, schema_path=schema_path, backend=backend)


def _assign_path(target: Dict[str, Any], dotted_key: str, value: Any) -> None:
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


def _deep_merge_dicts(base: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in incoming.items():
        if (
            isinstance(value, dict)
            and isinstance(merged.get(key), dict)
        ):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


from .aggregators import (
    collect_generator_params as collect_generator_params,
    generate_config_from_yaml as generate_config_from_yaml,
    generate_config_from_input_dict as generate_config_from_input_dict,
)
__all__ = [
    "collect_generator_params",
    "generate_config_from_yaml",
    "generate_config_from_input_dict",
    "generate_backend_config",
    "generate_backend_artifacts",
    "parse_cli_params",
    "add_generator_override_arguments",
    "load_generator_overrides",
    "load_generator_overrides_from_args",
    "parse_backend_arg",
    "parse_mapping_arg",
    "resolve_mapping_yaml",
    "prepare_generator_params",
]
