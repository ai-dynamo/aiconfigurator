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
import os
import shlex
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

_AIPERF_RECORD_KEYS = {
    "request_throughput": "request_throughput",
    "request_latency": "request_latency",
    "time_to_first_token": "ttft",  # aiperf uses 'ttft'
    "inter_token_latency": "inter_token_latency",
    "output_token_throughput": "output_token_throughput",
    "output_token_throughput_per_user": "output_token_throughput_per_user",
}


def _to_list(v) -> Sequence[int]:
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return list(map(int, v))
    return [int(v)]

def _stream(cmd: List[str], cwd: Path | None = None, env=None) -> int:
    LOG.debug("Exec: %s", " ".join(map(str, cmd)))
    with subprocess.Popen(
        cmd, cwd=str(cwd) if cwd else None, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    ) as p:
        assert p.stdout
        for line in p.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
        p.wait()
        return p.returncode


def _ensure_uv_venv(venv_dir: Path) -> Path:
    """
    venv for aiperf
    """
    act = venv_dir / "bin" / "activate"
    if act.exists():
        LOG.info("Using existing venv: %s", venv_dir)
        return act
    LOG.info("Creating virtual environment with uv: uv venv %s", venv_dir)
    rc = _stream(["uv", "venv", str(venv_dir)])
    if rc:
        raise RuntimeError(f"`uv venv {venv_dir}` failed with rc={rc}")
    if not act.exists():
        raise FileNotFoundError(f"Virtualenv created but activate not found: {act}")
    return act


def _warn_if_tool_missing(venv_dir: Path, tool: str = "genai-perf") -> None:
    exe = venv_dir / "bin" / tool
    if not exe.exists():
        LOG.warning("%s not found at %s – make sure it is installed in the venv.",
                    tool, exe)


def _json_to_df(p: Path, folder_cc: int | None = None) -> pd.DataFrame:
    """
        aiperf result parser helper
    """
    with p.open() as f:
        js = json.load(f)
    row = {"experiment": p.parent.name}
    # Prefer aiperf 'records' layout; fallback to legacy top-level fields
    records = js.get("records") or {}
    for m in _METRICS:
        key = _AIPERF_RECORD_KEYS.get(m, m)
        blob = records.get(key, js.get(m, {})) or {}
        for stat in _STATS:
            if stat in blob:
                row[f"{m}_{stat}"] = blob[stat]

    # extract concurrency from folder name, or JSON
    if folder_cc is not None:
        row["load_type"] = "concurrency"
        row["load_value"] = int(folder_cc)
        row["load_label"] = f"cc{folder_cc}"
    else:
        m = re.search(r"_concurrency_(\d+)", p.stem)
        if m:
            row["load_type"] = "concurrency"
            row["load_value"] = int(m.group(1))
            row["load_label"] = f"cc{row['load_value']}"
        else:
            # aiperf JSON
            cc = (js.get("input_config", {}) or {}).get("loadgen", {}).get("concurrency")
            if not cc:
                # legacy genai-perf JSON
                stim = (js.get("input_config", {}) or {}).get("perf_analyzer", {}).get("stimulus", {}) or {}
                cc = stim.get("concurrency")
            if cc:
                row["load_type"] = "concurrency"
                row["load_value"] = int(cc)
                row["load_label"] = f"cc{cc}"

    return pd.DataFrame([row])


def parse(path: Path) -> pd.DataFrame:
    """
    Prefer new layout: <bench>/concurrency_<N>/profile_export_aiperf.json,
    but keep backward compatibility with older 'profile_export*.json' files.
    """
    p = Path(path)
    dfs: list[pd.DataFrame] = []

    if p.is_dir():
        # aiperf output structure: one folder per concurrency
        for cc_dir in sorted(p.rglob("concurrency_*")):
            if cc_dir.is_dir():
                m = re.match(r"concurrency_(\d+)$", cc_dir.name)
                f = cc_dir / "profile_export_aiperf.json"
                if m and f.exists():
                    dfs.append(_json_to_df(f, folder_cc=int(m.group(1))))
        if not dfs:
            # Fallback: legacy layout anywhere under the dir
            dfs = [_json_to_df(fp) for fp in p.rglob("profile_export*.json")]
    else:
        dfs = [_json_to_df(p)]

    if not dfs:
        raise FileNotFoundError(f"No genai-perf/aiperf JSON in {path}")
    return pd.concat(dfs, ignore_index=True)


@register("genai_perf", parse=parse)
def run(cfg: Cfg, *, bin_path: str = "aiperf") -> None:
    """
    Execute aiperf.
    """
    art_dir = Path(cfg["base_folder"]) / str(cfg.get("result_folder", cfg["name"]))
    art_dir.mkdir(parents=True, exist_ok=True)

    model     = str(cfg.get("model", "unused"))
    tokenizer = str(cfg.get("tokenizer", model))
    url       = str(cfg["url"])
    isl       = int(cfg.get("input_sequence_length", 1024))
    osl       = int(cfg.get("output_sequence_length", 128))
    concs     = _to_list(cfg.get("concurrency"))
    if not concs:
        raise ValueError("concurrency list is required")

    # make sure venv already created
    venv_dir = Path(str(cfg.get("venv_dir") or os.environ.get("AIC_VENV", "aic")))
    activate = _ensure_uv_venv(venv_dir)
    _warn_if_tool_missing(venv_dir, tool=Path(bin_path).name)

    LOG.info("genai-perf (chat) url=%s conc=%s isl=%d osl=%d [venv=%s]",
             url, concs, isl, osl, venv_dir)

    for v in concs:
        # Write each run into its own folder: <bench>/concurrency_<N>/
        cc_dir = art_dir / f"concurrency_{v}"
        cc_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            bin_path, "profile",
            "-m", model,
            "--tokenizer", tokenizer,
            "--endpoint-type", "chat",
            "--url", url,
            "--streaming",
            "--artifact-dir", str(cc_dir),
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
            "-v",
        ]

        cmdline = shlex.join(cmd)
        shell_line = f"set -e; source {shlex.quote(str(activate))} && {cmdline}"
        rc = _stream(["bash", "-lc", shell_line])
        if rc:
            LOG.error("genai-perf failed at concurrency=%s (rc=%s)", v, rc)
        else:
            LOG.info("genai-perf finished at concurrency=%s", v)
