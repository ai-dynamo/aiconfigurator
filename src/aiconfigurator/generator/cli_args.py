# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Argparse for generator.

This module scans template files to auto-inject CLI options for
dynamo_config.* overrides, and exposes a 'build_dynamo_config' helper
to materialize user overrides into a DynamoConfig object.
"""
import sys, re, argparse
from pathlib import Path
from typing import Dict, Any
from .inputs.schema import DynamoConfig

TEMPLATE_ROOT = Path(__file__).resolve().parent / "templates"
_PREFIX = r"dynamo_config"

def _detect_backend_from_argv(default_backend: str) -> str:
    it = iter(sys.argv[1:])
    for tok in it:
        if tok.startswith("--backend="):
            return tok.split("=", 1)[1]
        if tok == "--backend":
            try:
                return next(it)
            except StopIteration:
                break
    return default_backend

def _backend_template_dir(backend: str) -> Path:
    return TEMPLATE_ROOT / backend

def _scan_templates_for_help(backend: str) -> Dict[str, Dict[str, str]]:
    """Parse Jinja templates to auto-extract CLI help metadata."""
    tmpl_dir = _backend_template_dir(backend)
    comment = re.compile(rf"#\s*{_PREFIX}\.([A-Za-z_][\w]*)\s+(.*)$")
    default = re.compile(rf"{_PREFIX}\.([A-Za-z_][\w]*)\s*\|\s*default\(([^)]+)\)")
    var_pat = re.compile(rf"{_PREFIX}\.([A-Za-z_][\w]*)")

    meta: Dict[str, Dict[str, str]] = {}
    for name in ("run.sh.j2", "extra_engine_args.yaml.j2", "k8s_deploy.yaml.j2"):
        p = tmpl_dir / name
        if not p.exists():
            continue
        for ln in p.read_text().splitlines():
            if (m := comment.search(ln)):
                meta.setdefault(m[1], {})["desc"] = m[2].strip()
            for v, d in default.findall(ln):
                meta.setdefault(v, {})["default"] = d.strip().strip("'\"")
            for v in var_pat.findall(ln):
                meta.setdefault(v, {})
    for k in meta:
        meta[k].setdefault("desc", "")
        meta[k].setdefault("default", "")
    return meta

def smart_cast(val: str):
    """Cast CLI string into bool / int / float / list when possible."""
    if val.lower() in {"true", "false"}:
        return val.lower() == "true"
    if "," in val:
        return [smart_cast(x) for x in val.split(",")]
    try:
        return int(val) if val.isdigit() else float(val)
    except ValueError:
        return val

def build_dynamo_config(args) -> DynamoConfig:
    """Collect overrides from argparse Namespace into DynamoConfig."""
    meta = args._dyn_meta
    extras = {
        k: smart_cast(v)
        for k, v in vars(args).items()
        if (
            k in meta
            or any(k.startswith(p) for p in ("decode_", "prefill_", "agg_"))
        )
        and v is not None
    }
    return DynamoConfig(extras=extras)

def add_config_generation_cli(parser: argparse.ArgumentParser, default_backend: str) -> str:
    """
    Inject dynamo_config.* override options into an existing ArgumentParser.
    """
    chosen_backend = _detect_backend_from_argv(default_backend=default_backend)
    meta = _scan_templates_for_help(chosen_backend)

    grp = parser.add_argument_group(
        "Template overrides for generated engine configs. "
        "Prefix with decode_/prefill_/agg_ to scope to a specific worker."
    )

    grp.add_argument(
        "--generated_config_version",
        type=str,
        default=None,
        help="Target engine version for generated configs. If none is given, will skip the generation",
    )

    for v, m in sorted(meta.items()):
        doc = m["desc"] + (f" (default: {m['default']})" if m["default"] else "")
        grp.add_argument(f"--{v}", type=str, help=doc)
    for v in meta:
        for role in ("decode", "prefill", "agg"):
            grp.add_argument(f"--{role}_{v}", type=str, help=argparse.SUPPRESS)

    parser.set_defaults(_dyn_meta=meta)
    return chosen_backend
