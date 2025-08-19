# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Optional

LOG = logging.getLogger(__name__)

def run_stream(cmd, cwd: Path | None = None, env=None) -> int:
    LOG.debug("Exec: %s", " ".join(map(str, cmd)))
    with subprocess.Popen(
        cmd, cwd=str(cwd) if cwd else None, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    ) as p:
        assert p.stdout
        for line in p.stdout:
            print(line, end="")
        p.wait()
        return p.returncode

def mkdir_p(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)

def find_newest_subdir(base: Path) -> Optional[Path]:
    cands = [p for p in base.glob("*") if p.is_dir()]
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)

def parse_disagg_start_script(path: Path) -> dict:
    """
    Extract prefill/decode workers and GPUs per worker from disagg/node_0_run.sh.

    Looking for lines like:
      PREFILL_GPU=1
      PREFILL_WORKERS=5
      DECODE_GPU=1
      DECODE_WORKERS=3
    """
    out = {
        "PREFILL_GPU": 0,
        "PREFILL_WORKERS": 0,
        "DECODE_GPU": 0,
        "DECODE_WORKERS": 0,
    }
    try:
        s = path.read_text()
    except Exception as e:
        LOG.warning("Cannot read %s: %s", path, e)
        return out

    def grab(name: str) -> int:
        m = re.search(rf"^\s*{name}\s*=\s*(\d+)\s*$", s, flags=re.M)
        return int(m.group(1)) if m else 0

    out["PREFILL_GPU"] = grab("PREFILL_GPU")
    out["PREFILL_WORKERS"] = grab("PREFILL_WORKERS")
    out["DECODE_GPU"] = grab("DECODE_GPU")
    out["DECODE_WORKERS"] = grab("DECODE_WORKERS")
    return out
