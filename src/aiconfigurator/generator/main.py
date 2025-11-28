"""
AI Generator - Main entry point.

This module provides the main CLI interface for the AI generator.
It uses the new architecture with separate API layer for parameter collection
and configuration modules for better organization.
"""

import json
import os
import sys
import argparse
import logging
from typing import Any, Dict, List, Optional

from api import (
    resolve_mapping_yaml,
    generate_backend_config,
    parse_cli_params,
    generate_backend_artifacts,
    prepare_generator_params,
)


def main(argv: Optional[List[str]] = None):
    """
    Main entry point for the AI generator CLI.
    
    This function handles command-line argument parsing, configuration generation,
    and output formatting.
    """
    # Get current directory for default mapping path
    current_dir = os.path.dirname(__file__)
    default_mapping_path = os.path.join(current_dir, "config", "backend_config_mapping.yaml")
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(
        prog="aic-generator",
        description="Generate backend-specific configuration or artifacts from unified parameters",
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True)
    p_cfg = subparsers.add_parser("render-config")
    p_art = subparsers.add_parser("render-artifacts")
    for p in (p_cfg, p_art):
        p.add_argument("--backend", required=True, help="Target backend name (e.g., vllm, trtllm, sglang)")
        p.add_argument("--mapping", help="Path to backend_config_mapping.yaml")
        p.add_argument("--config", help="Path to generator input YAML file")
        p.add_argument(
            "--set",
            action="append",
            default=[],
            metavar="KEY=VALUE",
            help="Inline override using dotted keys (e.g., ServiceConfig.model_path=...)",
        )
    p_cfg.add_argument(
        "--role",
        choices=["auto", "prefill", "decode", "agg", "all"],
        default="auto",
        help="Worker role to render (auto detects based on available params).",
    )
    p_art.add_argument("--templates-dir", help="Templates directory override")
    p_art.add_argument("--version", help="Backend version for template selection")
    p_art.add_argument("--output", help="Directory to save generated artifacts")
    args = parser.parse_args(argv)

    backend = args.backend
    explicit_mapping = args.mapping
    try:
        yaml_path = resolve_mapping_yaml(explicit_mapping, default_mapping_path)
    except FileNotFoundError as e:
        logger.error("%s", e)
        sys.exit(2)

    try:
        cli_params = parse_cli_params(args.set or [])
        generator_params = prepare_generator_params(args.config, cli_params, backend=backend)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        sys.exit(2)

    cmd = args.cmd
    if cmd == "render-artifacts":
        artifacts = generate_backend_artifacts(
            generator_params,
            backend,
            templates_dir=args.templates_dir,
            output_dir=args.output,
            backend_version=args.version,
        )
        print(json.dumps(artifacts, ensure_ascii=False, indent=2))
        return

    roles = _resolve_roles(args.role, generator_params, logger)
    rendered_backend: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for role in roles:
        ctx = _build_worker_context(generator_params, role)
        rendered_backend[role] = generate_backend_config(ctx, backend, yaml_path)
    print(json.dumps(rendered_backend, ensure_ascii=False, indent=2))


def _resolve_roles(requested: str, params: Dict[str, Any], logger: logging.Logger) -> List[str]:
    available = [
        role for role, data in (params.get("params") or {}).items()
        if data
    ]
    if requested in {"prefill", "decode", "agg"}:
        if requested not in available:
            logger.warning("Requested role '%s' not present in inputs, falling back to auto detection.", requested)
        else:
            return [requested]
    if requested == "all":
        return available or ["prefill"]
    return available or ["prefill"]


def _build_worker_context(params: Dict[str, Any], role: str) -> Dict[str, Any]:
    ctx: Dict[str, Any] = {}
    ctx.update(params.get("service") or {})
    ctx.update(params.get("k8s") or {})
    ctx.update(params.get("workers") or {})
    ctx.update(params.get("sla") or {})
    ctx.update(params.get("params", {}).get(role, {}) or {})
    ctx["role"] = role
    return ctx


if __name__ == "__main__":
    main()
