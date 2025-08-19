# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Sequence
import logging
import subprocess
import sys
import json
import re
import pandas as pd

from . import register

LOG = logging.getLogger(__name__)
Cfg = Dict[str, object]

_METRICS = {
    "request_throughput",
    "request_latency",
    "time_to_first_token",
    "inter_token_latency",
    "output_token_throughput",
    "output_token_throughput_per_user",
}
_STATS = {"avg", "p50", "p90", "p95", "p99", "min", "max", "std"}

def _to_list(v) -> Sequence[int]:
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return list(map(int, v))
    return [int(v)]

def _stream(cmd: List[str], cwd: Path | None = None, env=None) -> int:
    LOG.debug("Exec: %s", " ".join(cmd))
    with subprocess.Popen(
        cmd, cwd=cwd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    ) as p:
        assert p.stdout
        for line in p.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
        p.wait()
        return p.returncode

def _json_to_df(p: Path) -> pd.DataFrame:
    with p.open() as f:
        js = json.load(f)
    row = {"experiment": p.parent.name}
    for m in _METRICS:
        blob = js.get(m, {})
        for stat in _STATS:
            if stat in blob:
                row[f"{m}_{stat}"] = blob[stat]
    # extract concurrency if present in filename or JSON
    m = re.search(r"_concurrency_(\d+)", p.stem)
    if m:
        row["load_type"] = "concurrency"
        row["load_value"] = int(m.group(1))
        row["load_label"] = f"cc{row['load_value']}"
    else:
        stim = js.get("input_config", {}).get("perf_analyzer", {}).get("stimulus", {})
        cc = stim.get("concurrency")
        if cc:
            row["load_type"] = "concurrency"
            row["load_value"] = int(cc)
            row["load_label"] = f"cc{cc}"
    return pd.DataFrame([row])

def parse(path: Path) -> pd.DataFrame:
    if path.is_dir():
        dfs = [_json_to_df(p) for p in path.rglob("profile_export*.json")]
        if not dfs:
            raise FileNotFoundError(f"No genai-perf JSON in {path}")
        return pd.concat(dfs, ignore_index=True)
    return _json_to_df(path)

@register("genai_perf", parse=parse)
def run(cfg: Cfg, *, bin_path: str = "genai-perf") -> None:
    """
    Execute genai-perf – text-only, concurrency only.
    """
    art_dir = Path(cfg["base_folder"]) / cfg.get("result_folder", cfg["name"])
    art_dir.mkdir(parents=True, exist_ok=True)

    model     = str(cfg.get("model", "unused"))
    tokenizer = str(cfg.get("tokenizer", model))
    url       = str(cfg["url"])
    isl       = int(cfg.get("input_sequence_length", 1024))
    osl       = int(cfg.get("output_sequence_length", 128))
    concs     = _to_list(cfg.get("concurrency"))
    if not concs:
        raise ValueError("concurrency list is required")

    LOG.info("genai-perf (chat) url=%s conc=%s isl=%d osl=%d", url, concs, isl, osl)

    for v in concs:
        prof_name = f"profile_export_isl_{isl}_osl_{osl}_concurrency_{v}.json"
        cmd = [
            bin_path, "profile",
            "-m", model,
            "--tokenizer", tokenizer,
            "--endpoint-type", "chat",
            "--url", url,
            "--streaming",
            "--profile-export-file", prof_name,
            "--artifact-dir", str(art_dir),
            "--endpoint", "/v1/chat/completions",
            "--synthetic-input-tokens-mean", str(isl),
            "--synthetic-input-tokens-stddev", "0",
            "--output-tokens-mean", str(osl),
            "--output-tokens-stddev", "0",
            "--extra-inputs", f"max_tokens:{osl}",
            "--extra-inputs", f"min_tokens:{osl}",
            "--extra-inputs", "ignore_eos:true",
            "--concurrency", str(v),
            "--request-count", str(v * 10),
            "--warmup-request-count", str(v * 2),
            "--num-dataset-entries", str(v * 12),
            "--", "-v", "--max-threads", "256"
        ]
        rc = _stream(cmd)
        if rc:
            LOG.error("genai-perf failed at concurrency=%s (rc=%s)", v, rc)
        else:
            LOG.info("genai-perf finished at concurrency=%s", v)
