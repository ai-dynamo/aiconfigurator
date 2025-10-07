# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Artifact saving utilities.
"""
import os, yaml
from typing import Dict, Any


class _LiteralDumper(yaml.SafeDumper):
    pass

def _represent_multiline_str(dumper: yaml.SafeDumper, data: str):
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)

_LiteralDumper.add_representer(str, _represent_multiline_str)

def _render_shell_text(val: Any) -> str:
    """Return shell script that can be used for launch the service."""
    if isinstance(val, (list, tuple)):
        text = "\n".join(str(x) for x in val)
    else:
        text = str(val)
    lines = []
    for ln in text.splitlines():
        stripped = ln.lstrip()
        if stripped.startswith("#!") or not stripped.startswith("#"):
            lines.append(ln)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    return "\n".join(lines)

def save_artifacts(by_mode: Dict[str, Dict[str, Any]], root_dir: str) -> None:
    """Persist artifacts to filesystem."""
    os.makedirs(root_dir, exist_ok=True)
    for mode, files in by_mode.items():
        mode_dir = os.path.join(root_dir, mode)
        os.makedirs(mode_dir, exist_ok=True)
        for fname, content in files.items():
            out_path = os.path.join(mode_dir, fname)
            with open(out_path, "w", newline="\n") as f:
                if fname.endswith(".sh"):
                    f.write(_render_shell_text(content))
                else:
                    yaml.dump(
                        content, f,
                        Dumper=_LiteralDumper,
                        sort_keys=False,
                        allow_unicode=True,
                        width=4096,
                        default_flow_style=False,
                    )