# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Input parsing for sflow config generation from JSON and CSV files.

Supports both aggregated and disaggregated configs in either format.
See the CLI user guide for the expected schema of each format.
"""

import json
import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Column aliases: maps alternative column names (e.g. from eval CSVs) to canonical names.
_CSV_COLUMN_ALIASES: dict[str, str] = {
    "backend_name": "backend",
    "silicon_ttft_avg": "ttft",
    "silicon_tpot_avg": "tpot",
    "attention_dp": "dp",
}


@dataclass
class SflowInputConfig:
    """Normalized descriptor for a single sflow config to generate."""

    name: str
    model: str
    system: str
    backend: str
    backend_version: Optional[str]
    serving_mode: str  # "agg" or "disagg"
    isl: int
    osl: int
    ttft: Optional[float] = None
    tpot: Optional[float] = None
    concurrency: Optional[int] = None
    num_total_gpus: Optional[int] = None
    # agg mode
    agg_params: Optional[dict[str, Any]] = None
    agg_workers: int = 0
    # disagg mode
    prefill_params: Optional[dict[str, Any]] = None
    prefill_workers: int = 0
    decode_params: Optional[dict[str, Any]] = None
    decode_workers: int = 0


def load_sflow_input(file_path: str, backend_default: str = "trtllm") -> list[SflowInputConfig]:
    """Load sflow input configs from a JSON or CSV file.

    Auto-detects format by file extension.

    Args:
        file_path: Path to the input file (.json or .csv).
        backend_default: Default backend name when not specified in the input.

    Returns:
        List of normalized config descriptors.

    Raises:
        FileNotFoundError: If file_path does not exist.
        ValueError: If file extension is unsupported or content is invalid.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".json":
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("JSON input must be a top-level object mapping config names to config objects.")
        return _parse_json_input(data, backend_default)
    elif ext == ".csv":
        return _parse_csv_input(file_path, backend_default)
    else:
        raise ValueError(f"Unsupported input file extension '{ext}'. Expected .json or .csv.")


def _parse_json_input(data: dict[str, Any], backend_default: str) -> list[SflowInputConfig]:
    """Parse JSON input where each key is a config name.

    Mode detection:
    - Has ``prefill`` and ``decode`` keys → disagg
    - Has ``agg`` key → agg
    - Otherwise → agg with top-level parallelism fields
    """
    configs: list[SflowInputConfig] = []
    for name, entry in data.items():
        if not isinstance(entry, dict):
            logger.warning("Skipping non-dict entry '%s'", name)
            continue

        model = entry.get("model", "")
        backend = entry.get("backend", backend_default)
        isl = _safe_int(entry.get("isl"), 0)
        osl = _safe_int(entry.get("osl"), 0)
        ttft = _safe_float(entry.get("ttft"))
        tpot = _safe_float(entry.get("tpot"))
        concurrency = _safe_int(entry.get("concurrency"))
        num_total_gpus = _safe_int(entry.get("num_total_gpus"))

        is_disagg = "prefill" in entry and "decode" in entry
        has_agg_key = "agg" in entry

        if is_disagg:
            prefill_d = entry["prefill"]
            decode_d = entry["decode"]
            system = prefill_d.get("system", entry.get("system", ""))
            backend_version = prefill_d.get("version", entry.get("backend_version"))

            configs.append(
                SflowInputConfig(
                    name=name,
                    model=model,
                    system=system,
                    backend=backend,
                    backend_version=backend_version,
                    serving_mode="disagg",
                    isl=isl,
                    osl=osl,
                    ttft=ttft,
                    tpot=tpot,
                    concurrency=concurrency,
                    num_total_gpus=num_total_gpus,
                    prefill_params=_build_worker_params(prefill_d),
                    prefill_workers=_safe_int(prefill_d.get("workers"), 1),
                    decode_params=_build_worker_params(decode_d),
                    decode_workers=_safe_int(decode_d.get("workers"), 1),
                )
            )
        elif has_agg_key:
            agg_d = entry["agg"]
            system = agg_d.get("system", entry.get("system", ""))
            backend_version = agg_d.get("version", entry.get("backend_version"))

            configs.append(
                SflowInputConfig(
                    name=name,
                    model=model,
                    system=system,
                    backend=backend,
                    backend_version=backend_version,
                    serving_mode="agg",
                    isl=isl,
                    osl=osl,
                    ttft=ttft,
                    tpot=tpot,
                    concurrency=concurrency,
                    num_total_gpus=num_total_gpus,
                    agg_params=_build_worker_params(agg_d),
                    agg_workers=_safe_int(agg_d.get("workers"), 1),
                )
            )
        else:
            # Fallback: treat top-level parallelism fields as agg
            system = entry.get("system", "")
            backend_version = entry.get("version", entry.get("backend_version"))

            configs.append(
                SflowInputConfig(
                    name=name,
                    model=model,
                    system=system,
                    backend=backend,
                    backend_version=backend_version,
                    serving_mode="agg",
                    isl=isl,
                    osl=osl,
                    ttft=ttft,
                    tpot=tpot,
                    concurrency=concurrency,
                    num_total_gpus=num_total_gpus,
                    agg_params=_build_worker_params(entry),
                    agg_workers=_safe_int(entry.get("workers"), 1),
                )
            )

    return configs


def _parse_csv_input(file_path: str, backend_default: str) -> list[SflowInputConfig]:
    """Parse CSV input where each row is a config.

    Mode detected from ``mode`` column (``agg`` or ``disagg``).
    For disagg rows, parallelism columns are prefixed with ``prefill_`` and ``decode_``.
    """
    df = pd.read_csv(file_path)

    # Apply column aliases
    rename_map = {old: new for old, new in _CSV_COLUMN_ALIASES.items() if old in df.columns and new not in df.columns}
    if rename_map:
        df = df.rename(columns=rename_map)

    configs: list[SflowInputConfig] = []
    for i, row in df.iterrows():
        name = _row_str(row, "name", f"row_{i}")
        mode = _row_str(row, "mode", "agg").lower()
        model = _row_str(row, "model", "")
        backend = _row_str(row, "backend", backend_default)
        system = _row_str(row, "system", "")
        backend_version = _row_str(row, "backend_version")
        isl = _row_int(row, "isl", 0)
        osl = _row_int(row, "osl", 0)
        ttft = _row_float(row, "ttft")
        tpot = _row_float(row, "tpot")
        concurrency = _row_int(row, "concurrency")
        num_total_gpus = _row_int(row, "num_total_gpus")

        if mode == "disagg":
            prefill_d = _extract_prefixed_worker_fields(row, "prefill_")
            decode_d = _extract_prefixed_worker_fields(row, "decode_")

            configs.append(
                SflowInputConfig(
                    name=name,
                    model=model,
                    system=system,
                    backend=backend,
                    backend_version=backend_version,
                    serving_mode="disagg",
                    isl=isl,
                    osl=osl,
                    ttft=ttft,
                    tpot=tpot,
                    concurrency=concurrency,
                    num_total_gpus=num_total_gpus,
                    prefill_params=_build_worker_params(prefill_d),
                    prefill_workers=_safe_int(prefill_d.get("workers"), 1),
                    decode_params=_build_worker_params(decode_d),
                    decode_workers=_safe_int(decode_d.get("workers"), 1),
                )
            )
        else:
            agg_d = _extract_flat_worker_fields(row)

            configs.append(
                SflowInputConfig(
                    name=name,
                    model=model,
                    system=system,
                    backend=backend,
                    backend_version=backend_version,
                    serving_mode="agg",
                    isl=isl,
                    osl=osl,
                    ttft=ttft,
                    tpot=tpot,
                    concurrency=concurrency,
                    num_total_gpus=num_total_gpus,
                    agg_params=_build_worker_params(agg_d),
                    agg_workers=_safe_int(agg_d.get("workers"), 1),
                )
            )

    return configs


# ---------------------------------------------------------------------------
# Worker param builder
# ---------------------------------------------------------------------------


def _build_worker_params(d: dict[str, Any]) -> dict[str, Any]:
    """Map input fields to the canonical worker params format expected by ``collect_generator_params``.

    Input keys: tp, pp, dp, moe_tp, moe_ep, bs, gemm, kvcache, fmha, moe.
    Output keys: tensor_parallel_size, pipeline_parallel_size, etc.
    """
    tp = _safe_int(d.get("tp"), 1)
    pp = _safe_int(d.get("pp"), 1)
    dp = _safe_int(d.get("dp"), 1)
    moe_tp = _safe_int(d.get("moe_tp"), 0)
    moe_ep = _safe_int(d.get("moe_ep"), 0)
    bs = _safe_int(d.get("bs"), 1)

    params: dict[str, Any] = {
        "tensor_parallel_size": tp,
        "pipeline_parallel_size": pp,
        "data_parallel_size": dp,
        "gpus_per_worker": tp * pp * dp,
        "moe_tensor_parallel_size": moe_tp,
        "moe_expert_parallel_size": moe_ep,
        "max_batch_size": bs,
    }

    # Quantization modes — only set when present and non-null
    gemm = _non_null_str(d.get("gemm"))
    kvcache = _non_null_str(d.get("kvcache"))
    fmha = _non_null_str(d.get("fmha"))
    moe_quant = _non_null_str(d.get("moe"))

    if gemm:
        params["gemm_quant_mode"] = gemm
    if kvcache:
        params["kvcache_quant_mode"] = kvcache
        params["kv_cache_dtype"] = kvcache
    if fmha:
        params["fmha_quant_mode"] = fmha
    if moe_quant:
        params["moe_quant_mode"] = moe_quant

    return params


# ---------------------------------------------------------------------------
# CSV field extraction helpers
# ---------------------------------------------------------------------------


def _extract_flat_worker_fields(row: pd.Series) -> dict[str, Any]:
    """Extract worker fields from a flat (agg) CSV row."""
    return {
        "tp": _row_val(row, "tp"),
        "pp": _row_val(row, "pp"),
        "dp": _row_val(row, "dp"),
        "moe_tp": _row_val(row, "moe_tp"),
        "moe_ep": _row_val(row, "moe_ep"),
        "bs": _row_val(row, "bs"),
        "workers": _row_val(row, "workers"),
        "gemm": _row_val(row, "gemm"),
        "kvcache": _row_val(row, "kvcache"),
        "fmha": _row_val(row, "fmha"),
        "moe": _row_val(row, "moe"),
    }


def _extract_prefixed_worker_fields(row: pd.Series, prefix: str) -> dict[str, Any]:
    """Extract worker fields with a given prefix (e.g. ``prefill_``, ``decode_``)."""
    return {
        "tp": _row_val(row, f"{prefix}tp"),
        "pp": _row_val(row, f"{prefix}pp"),
        "dp": _row_val(row, f"{prefix}dp"),
        "moe_tp": _row_val(row, f"{prefix}moe_tp"),
        "moe_ep": _row_val(row, f"{prefix}moe_ep"),
        "bs": _row_val(row, f"{prefix}bs"),
        "workers": _row_val(row, f"{prefix}workers"),
        "gemm": _row_val(row, f"{prefix}gemm"),
        "kvcache": _row_val(row, f"{prefix}kvcache"),
        "fmha": _row_val(row, f"{prefix}fmha"),
        "moe": _row_val(row, f"{prefix}moe"),
    }


# ---------------------------------------------------------------------------
# Value helpers
# ---------------------------------------------------------------------------


def _safe_int(val: Any, default: Optional[int] = None) -> Optional[int]:
    """Convert to int, returning *default* on failure or NaN."""
    if val is None:
        return default
    try:
        if isinstance(val, float) and math.isnan(val):
            return default
        return int(val)
    except (TypeError, ValueError):
        return default


def _safe_float(val: Any, default: Optional[float] = None) -> Optional[float]:
    """Convert to float, returning *default* on failure or NaN."""
    if val is None:
        return default
    try:
        fval = float(val)
        if math.isnan(fval):
            return default
        return fval
    except (TypeError, ValueError):
        return default


def _non_null_str(val: Any) -> Optional[str]:
    """Return string value if present and not NaN/None/empty, else None."""
    if val is None:
        return None
    if isinstance(val, float) and math.isnan(val):
        return None
    s = str(val).strip()
    if not s or s.lower() in ("none", "null", "nan"):
        return None
    return s


def _row_val(row: pd.Series, key: str, default: Any = None) -> Any:
    """Get a value from a pandas Series, returning *default* if missing or NaN."""
    if key not in row.index:
        return default
    val = row[key]
    if val is None:
        return default
    if isinstance(val, float) and math.isnan(val):
        return default
    try:
        if pd.isna(val):
            return default
    except (TypeError, ValueError):
        pass
    return val


def _row_str(row: pd.Series, key: str, default: Optional[str] = None) -> Optional[str]:
    """Get a string value from a pandas Series."""
    val = _row_val(row, key)
    if val is None:
        return default
    return str(val).strip() or default


def _row_int(row: pd.Series, key: str, default: Optional[int] = None) -> Optional[int]:
    """Get an int value from a pandas Series."""
    return _safe_int(_row_val(row, key), default)


def _row_float(row: pd.Series, key: str, default: Optional[float] = None) -> Optional[float]:
    """Get a float value from a pandas Series."""
    return _safe_float(_row_val(row, key), default)
