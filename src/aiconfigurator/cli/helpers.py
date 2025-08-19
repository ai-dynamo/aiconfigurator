# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import yaml
import re, argparse, logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

PKG_ROOT = Path(__file__).resolve().parent
TEMPLATE_ROOT = PKG_ROOT / "templates"
_PREFIX = r"dynamo_config"
GPU_PER_NODE = 8  # TODO: Support for various GPU_PER_NODE


@dataclass
class DynamoConfig:
    """
    Keep user-supplied overrides (from CLI) and expose helpers
    for Jinja rendering.

    extras: a flat dict  {key: value} collected from CLI
    """
    extras: Dict[str, Any] = field(default_factory=dict)

    # attach attrs for Jinja usage
    def __post_init__(self):
        """
        After dataclass init, copy each extra entry to self.*  so that
        templates can refer to ``dynamo_config.<key>`` directly.
        """
        for k, v in self.extras.items():
            setattr(self, k, v)

    def split_by_worker_type(self) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """
        Split overrides into global‑level and worker‑specific maps.

        Returns:
            (common, role_specific) where role_specific has keys
            ``decode``, ``prefill``, ``agg``.
        """
        common, role = {}, {"decode": {}, "prefill": {}, "agg": {}}
        for k, v in self.extras.items():
            for r in role:
                p = f"{r}_"
                if k.startswith(p):
                    role[r][k[len(p):]] = v
                    break
            else:
                common[k] = v
        return common, role


def allocate_disagg_nodes(p_worker, p_gpu, d_worker, d_gpu, gpu_per_node=GPU_PER_NODE):
    """
    Decide the number of prefill and decode workers on each node.

    Args:
        p_worker: number of prefill workers
        p_gpu: GPUs per prefill worker
        d_worker: number of decode workers
        d_gpu: GPUs per decode worker
        gpu_per_node: GPUs available on each physical node

    Returns:
        List of dicts, each describing workers on that node, e.g.
        [{'p_worker': 2, 'd_worker': 1}, ...]
    """
    nodes = []
    # first prefill
    for _ in range(p_worker):
        placed = False
        for n in nodes:
            if n['used'] + p_gpu <= gpu_per_node:
                n['p_worker'] += 1
                n['used'] += p_gpu
                placed = True
                break
        if not placed:
            nodes.append({'p_worker': 1, 'd_worker': 0, 'used': p_gpu})
    # then decode
    for _ in range(d_worker):
        placed = False
        for n in nodes:
            if n['used'] + d_gpu <= gpu_per_node:
                n['d_worker'] += 1
                n['used'] += d_gpu
                placed = True
                break
        if not placed:
            nodes.append({'p_worker': 0, 'd_worker': 1, 'used': d_gpu})
    return [{'p_worker': n['p_worker'], 'd_worker': n['d_worker']} for n in nodes]


def backend_template_dir(backend: str) -> Path:
    """
    Get the directory that contains templates for a given backend.
    """
    return TEMPLATE_ROOT / backend / "dynamo"


def build_dynamo_config(args) -> DynamoConfig:
    """
    Extract user overrides (dynamo_config.*) from argparse Namespace 
    and build a DynamoConfig object.
    """
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


def _detect_backend_from_argv(default_backend: str) -> str:
    """
    Scan sys.argv to discover an explicit --backend selection.
    Fallback to default_backend.
    """
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


def _scan_templates_for_help(backend: str) -> Dict[str, Dict[str, str]]:
    """
    Parse Jinja templates to auto-extract CLI help metadata.

    Returns:
        {var: {'desc': str, 'default': str}}
    """
    tmpl_dir = backend_template_dir(backend)
    comment = re.compile(rf"#\s*{_PREFIX}\.([A-Za-z_][\w]*)\s+(.*)$")
    default = re.compile(rf"{_PREFIX}\.([A-Za-z_][\w]*)\s*\|\s*default\(([^)]+)\)")
    var_pat = re.compile(rf"{_PREFIX}\.([A-Za-z_][\w]*)")

    meta: Dict[str, Dict[str, str]] = {}
    for name in ("run.sh.j2", "extra_engine_args.yaml.j2"):
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


def _dump_backend_file(path: str, content: Any) -> None:
    """
    Dump generated backend artefact.

    YAML  → safe_dump  
    *.sh  → write text (keep LF endings)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def _render_shell_text(val: Any) -> str:
        """Return shell script text without comment"""
        if isinstance(val, (list, tuple)):
            text = "\n".join(str(x) for x in val)
        else:
            text = str(val)

        lines: list[str] = []
        for ln in text.splitlines():
            stripped = ln.lstrip()
            if stripped.startswith("#!") or not stripped.startswith("#"):
                lines.append(ln)
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        return "\n".join(lines)
    
    with open(path, "w", newline="\n") as f:
        if path.endswith(".sh"):
            f.write(_render_shell_text(content))
        else:
            yaml.safe_dump(content, f, sort_keys=False, allow_unicode=True)


def add_dynamo_cli(parser: argparse.ArgumentParser, default_backend: str) -> str:
    """
    Inject *dynamo_config.* override options into an existing ArgumentParser.

    Returns:
        The backend chosen via CLI (or default).
    """
    chosen_backend = _detect_backend_from_argv(default_backend=default_backend)
    meta = _scan_templates_for_help(chosen_backend)

    grp = parser.add_argument_group(
        "Template overrides - values here are copied into the generated shell "
        "scripts / engine YAML.\n"
        "Add decode_, prefill_, or agg_ in front of a key to limit it to that "
        "worker only."
    )

    grp.add_argument(
        "--generated_config_version",
        type=str,
        default=None,
        help="Controls which engine version the generated configuration is targeted at. For example, if you run with --version 1.0.0rc3 but set --generated_config_version 1.0.0rc4, the tool produces a config file compatible with 1.0.0rc4. (default to --version)",
    )

    for v, m in sorted(meta.items()):
        doc = m["desc"] + (f" (default: {m['default']})" if m["default"] else "")
        grp.add_argument(f"--{v}", type=str, help=doc)
    for v in meta:
        for role in ("decode", "prefill", "agg"):
            grp.add_argument(f"--{role}_{v}", type=str, help=argparse.SUPPRESS)
    parser.set_defaults(_dyn_meta=meta)
    return chosen_backend


def smart_cast(val: str) -> Any:
    """
    Try to convert CLI string into bool / int / float / list as appropriate.
    """
    if val.lower() in {"true", "false"}:
        return val.lower() == "true"
    if "," in val:
        return [smart_cast(x) for x in val.split(",")]
    try:
        return int(val) if val.isdigit() else float(val)
    except ValueError:
        return val
