# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Optional
import logging
import subprocess
import sys
import json
import re

import pandas as pd

from . import register

LOG = logging.getLogger(__name__)
Cfg = Dict[str, object]

# Keep metric naming aligned with genai_perf_runner
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
    """Run a command and stream stdout to the console. Returns the process return code."""
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


def _loads_with_inf(s: str) -> Dict[str, Any]:
    """Accept JSON constants Infinity / -Infinity / NaN (mirrors Python json's allow_nan behavior)."""
    return json.loads(
        s,
        parse_constant=lambda const: (
            float("inf")
            if const == "Infinity"
            else (-float("inf") if const == "-Infinity" else float("nan"))
        ),
    )


def _infer_cc_from_filename(p: Path) -> Optional[int]:
    """Infer max concurrency from filename suffix pattern *_cc_<num>.jsonl."""
    m = re.search(r"_cc_(\d+)\.jsonl$", p.name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _collect_jsonl_files(bench_dir: Path) -> List[Path]:
    """Collect benchmark JSONL files. Prefer bench_serving_*; fall back to *.jsonl."""
    files = sorted(bench_dir.glob("bench_serving_*.jsonl"))
    if not files:
        files = sorted(bench_dir.glob("*.jsonl"))
    return files


def _bench_json_to_row(p: Path, last: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize one benchmark record into a single row compatible with genai-perf output."""
    # Concurrency: prefer filename hint, otherwise take it from JSON
    cc_from_name = _infer_cc_from_filename(p)
    try:
        cc_from_json = int(last.get("max_concurrency") or 0)
    except Exception:
        cc_from_json = 0
    cc = cc_from_name if cc_from_name is not None else cc_from_json

    # Normalize input/output sequence lengths (isl/osl)
    def _as_int(x, default=0) -> int:
        try:
            return int(x)
        except Exception:
            return default

    isl = _as_int(last.get("random_input_len") or last.get("sharegpt_input_len"))
    osl = _as_int(last.get("random_output_len") or last.get("sharegpt_output_len"))

    # Raw aggregates emitted by bench_serving
    req_tput = float(last.get("request_throughput", 0.0))
    in_tput  = float(last.get("input_throughput", 0.0))
    out_tput = float(last.get("output_throughput", 0.0))
    tot_tput = float(last.get("total_throughput", 0.0))

    # Observed effective concurrency during the run
    concurrency_eff = float(last.get("concurrency", 0.0)) or float(cc or 0)

    # Map into the genai-perf metric space:
    # 1) request_throughput -> avg
    # 2) request_latency_*  -> e2e latency (ms)
    # 3) time_to_first_token_* -> ttft (ms)
    # 4) inter_token_latency_* -> itl (ms)
    # 5) output_token_throughput -> avg (tok/s)
    # 6) output_token_throughput_per_user -> avg = out_tput / concurrency_eff
    row = {
        "experiment": p.parent.name,
        "load_type": "concurrency",
        "load_value": int(cc or 0),
        "load_label": f"cc{int(cc or 0)}",
        # Handy passthroughs used downstream
        "isl": isl,
        "osl": osl,
        "backend": last.get("backend"),
        "dataset_name": last.get("dataset_name"),
        "request_rate": last.get("request_rate"),
        "duration": float(last.get("duration", 0.0)),
        "completed": _as_int(last.get("completed", 0)),
        "total_input_tokens": _as_int(last.get("total_input_tokens", 0)),
        "total_output_tokens": _as_int(last.get("total_output_tokens", 0)),
        "total_output_tokens_retokenized": _as_int(last.get("total_output_tokens_retokenized", 0)),
        "input_throughput": in_tput,
        "output_throughput": out_tput,
        "total_throughput": tot_tput,
        "concurrency_effective": concurrency_eff,
        "accept_length": last.get("accept_length"),
        "source_file": str(p),
    }

    # request_throughput
    row["request_throughput_avg"] = req_tput

    # request_latency_* (use end-to-end latency)
    row["request_latency_avg"] = float(last.get("mean_e2e_latency_ms", 0.0))
    row["request_latency_p50"] = float(last.get("median_e2e_latency_ms", 0.0))
    row["request_latency_p99"] = float(last.get("p99_e2e_latency_ms", 0.0))
    row["request_latency_std"] = float(last.get("std_e2e_latency_ms", 0.0))
    # p90/p95/min/max not emitted by bench_serving; leave as NaN if needed

    # time_to_first_token_*
    row["time_to_first_token_avg"] = float(last.get("mean_ttft_ms", 0.0))
    row["time_to_first_token_p50"] = float(last.get("median_ttft_ms", 0.0))
    row["time_to_first_token_p99"] = float(last.get("p99_ttft_ms", 0.0))
    row["time_to_first_token_std"] = float(last.get("std_ttft_ms", 0.0))

    # inter_token_latency_*
    row["inter_token_latency_avg"] = float(last.get("mean_itl_ms", 0.0))
    row["inter_token_latency_p50"] = float(last.get("median_itl_ms", 0.0))
    row["inter_token_latency_p95"] = float(last.get("p95_itl_ms", 0.0))
    row["inter_token_latency_p99"] = float(last.get("p99_itl_ms", 0.0))
    row["inter_token_latency_std"] = float(last.get("std_itl_ms", 0.0))
    row["inter_token_latency_max"] = float(last.get("max_itl_ms", 0.0))

    # output_token_throughput_* (tok/s)
    row["output_token_throughput_avg"] = out_tput

    # output_token_throughput_per_user_avg (tok/s/user)
    row["output_token_throughput_per_user_avg"] = (
        (out_tput / concurrency_eff) if concurrency_eff and concurrency_eff > 0 else float("nan")
    )

    return row


def _jsonl_to_df(p: Path) -> pd.DataFrame:
    """Read a .jsonl file and treat the last record as the aggregate result for that run."""
    last: Optional[Dict[str, Any]] = None
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = _loads_with_inf(line)
            except json.JSONDecodeError:
                rec = json.loads(line)
            last = rec

    if not last:
        # Empty file: return empty frame to be filtered by caller
        return pd.DataFrame()

    row = _bench_json_to_row(p, last)
    return pd.DataFrame([row])


def parse(path: Path) -> pd.DataFrame:
    """
    Parse bench_serving output (directory or single .jsonl) and return a DataFrame
    compatible with the genai-perf parser. The DataFrame includes:
      - experiment / load_type / load_value / load_label
      - normalized metrics columns: *_avg, *_p50, *_p95, *_p99, *_std, etc. (where available)
    """
    path = Path(path)
    if path.is_file():
        if path.suffix.lower() != ".jsonl":
            raise FileNotFoundError(f"Expect a .jsonl file, got: {path}")
        df = _jsonl_to_df(path)
        if df.empty:
            raise FileNotFoundError(f"No valid bench_serving JSON in file: {path}")
        return df

    # Directory: aggregate across all detected .jsonl files
    files = _collect_jsonl_files(path)
    if not files:
        raise FileNotFoundError(f"No bench_serving JSONL in {path}")

    dfs = []
    for f in files:
        df = _jsonl_to_df(f)
        if not df.empty:
            dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"Found JSONL but none had valid rows under {path}")
    df_all = pd.concat(dfs, ignore_index=True)

    # Sort by concurrency, then isl/osl if present
    sort_cols = [c for c in ["load_value", "isl", "osl"] if c in df_all.columns]
    if sort_cols:
        df_all = df_all.sort_values(by=sort_cols, ignore_index=True)

    return df_all


@register("bench_serving", parse=parse)
def run(cfg: Cfg, *, py_bin: str = "python3", script: str = "bench_serving.py") -> None:
    """
    Optional convenience runner to execute bench_serving.py locally.
    Most workloads run inside K8s pods; this keeps interfaces consistent.

    Expected cfg keys:
      - base_folder: artifacts root
      - name / result_folder: subdirectory name
      - url: base URL (e.g., http://127.0.0.1:8000)
      - model, tokenizer
      - input_sequence_length (isl), output_sequence_length (osl)
      - concurrency: int or list[int]
      - backend: defaults to 'sglang-oai' or 'sglang-oai-chat'
      - dataset_name: defaults to 'random'
      - num_prompts_per_cc: optional override; default is cc * 10
      - seed: default 42
      - warmup_requests: default 0
    """
    art_dir = Path(cfg["base_folder"]) / cfg.get("result_folder", cfg["name"])
    art_dir.mkdir(parents=True, exist_ok=True)

    url       = str(cfg["url"])
    model     = str(cfg.get("model", ""))
    tokenizer = str(cfg.get("tokenizer", model))

    isl  = int(cfg.get("input_sequence_length", 1024))
    osl  = int(cfg.get("output_sequence_length", 128))
    conc = _to_list(cfg.get("concurrency"))
    if not conc:
        raise ValueError("concurrency list is required")

    backend       = str(cfg.get("backend", "sglang-oai"))
    dataset_name  = str(cfg.get("dataset_name", "random"))
    seed          = int(cfg.get("seed", 42))
    warmups       = int(cfg.get("warmup_requests", 0))
    num_prompts_override = cfg.get("num_prompts_per_cc")

    LOG.info("bench_serving url=%s backend=%s conc=%s isl=%d osl=%d", url, backend, conc, isl, osl)

    for v in conc:
        # Default volume matches genai-perf scale: ~10 requests per active user
        num_prompts = int(num_prompts_override) if num_prompts_override is not None else (int(v) * 10)

        out_file = art_dir / f"bench_serving_{backend}_isl_{isl}_osl_{osl}_cc_{v}.jsonl"

        cmd = [
            py_bin, script,
            "--backend", backend,
            "--base-url", url,
            "--model", model,
            "--tokenizer", tokenizer,
            "--dataset-name", dataset_name,
            "--num-prompts", str(num_prompts),
            "--random-input-len", str(isl),
            "--random-output-len", str(osl),
            "--random-range-ratio", "1",
            "--max-concurrency", str(v),
            "--seed", str(seed),
            "--warmup-requests", str(warmups),
            "--output-file", str(out_file),
        ]
        rc = _stream(cmd)
        if rc:
            LOG.error("bench_serving failed at concurrency=%s (rc=%s)", v, rc)
        else:
            LOG.info("bench_serving finished at concurrency=%s -> %s", v, out_file)
