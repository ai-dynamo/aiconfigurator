#!/usr/bin/env python3
"""Capture the pre-deletion data-plane baseline for PR-6.

Run on a tree where the Python ``load_*_data`` parsers are still alive.  The
output (``data_plane_baseline.json``) freezes, for a pinned set of
(system, backend, version) databases:

  1. table digests  — every ``PerfDatabase._<family>_data`` attribute walked
     insertion-ordered and reduced to structure/order/value hashes
     (see ``_data_plane_codec.py``);
  2. support matrix — the lazy AND eager ``supported_quant_mode`` dumps;
  3. weights        — per-op ``get_weights()`` for representative models plus
     the backend memory-report ``weights`` component.

After the parsers are deleted and the attributes are served by the engine
table-view FFI, ``test_data_plane_baseline.py`` replays every digest — like
PR-5's ``test_query_shim_baseline.py``, regeneration is only meaningful on
the OLD tree; the checked-in JSON is the historical evidence.

Usage (from the repo root):
    uv run --no-sync python tests/cross_package/capture_data_plane_baseline.py --probe
    uv run --no-sync python tests/cross_package/capture_data_plane_baseline.py --write
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _data_plane_codec import (
    TABLE_ATTRIBUTES,
    snapshot_database_tables,
    snapshot_support_matrix,
)

BASELINE_PATH = Path(__file__).resolve().parent / "data_plane_baseline.json"

# Pinned databases.  Chosen so that every table attribute is non-empty on at
# least one pin (--probe prints the coverage matrix):
#  - h200_sxm/trtllm/1.2.0rc5 : the notebook-e2e pin (standard families)
#  - gb200/trtllm/1.3.0rc10   : notebook cell-13 pin (trtllm alltoall/wideep)
#  - b200_sxm/sglang/0.5.14   : sglang wideep MLA/MoE + deepep + gdn/dsa
#  - b200_sxm/sglang/0.5.16   : kda (+ shared-layer fallback for the rest)
#  - b200_sxm/vllm/0.24.0     : vllm families (mhc, dsv4)
#  - b60/vllm/0.10.2          : oneccl comm tables
#  - l40s/trtllm/1.3.0rc20    : the known below-SOL GEMM system (raw rows)
DATABASE_PINS: tuple[tuple[str, str, str], ...] = (
    ("h200_sxm", "trtllm", "1.2.0rc5"),
    ("gb200", "trtllm", "1.3.0rc10"),
    ("b200_sxm", "sglang", "0.5.14"),
    ("b200_sxm", "sglang", "0.5.16"),
    ("b200_sxm", "vllm", "0.24.0"),
    ("b60", "vllm", "0.20.0"),
    ("l40s", "trtllm", "1.3.0rc20"),
)

# Representative models for the weights snapshot: cover the non-zero
# get_weights() op families end to end (GEMM/Embedding via dense blocks, MoE,
# DSA mixed-dtype, MLA-adjacent GEMMs).  Op-class-level numeric pins for the
# families not constructible here (TrtLLMWideEPMoE, MoEExpertCompute, MHC,
# DSv4, MegaMoE) already exist in unit tests and keep guarding the math.
WEIGHT_MODEL_PINS: tuple[dict, ...] = (
    {
        "hf_id": "meta-llama/Meta-Llama-3.1-70B",
        "backend": "trtllm",
        "config": {
            "tp_size": 4,
            "pp_size": 1,
            "gemm_quant_mode": "fp8",
            "moe_quant_mode": "fp8",
            "kvcache_quant_mode": "fp8",
            "fmha_quant_mode": "fp8",
        },
    },
    {
        "hf_id": "deepseek-ai/DeepSeek-V4-Pro",
        "backend": "sglang",
        "config": {
            "tp_size": 1,
            "pp_size": 1,
            "attention_dp_size": 4,
            "moe_tp_size": 1,
            "moe_ep_size": 4,
            "gemm_quant_mode": "fp8",
            "moe_quant_mode": "fp8_block",
            "kvcache_quant_mode": "fp8",
            "fmha_quant_mode": "fp8",
        },
    },
    {
        "hf_id": "deepseek-ai/DeepSeek-V4-Pro",
        "backend": "sglang",
        "config": {
            "tp_size": 1,
            "pp_size": 1,
            "attention_dp_size": 4,
            "moe_tp_size": 1,
            "moe_ep_size": 4,
            "moe_backend": "megamoe",
            "gemm_quant_mode": "fp8",
            "moe_quant_mode": "fp8_block",
            "kvcache_quant_mode": "fp8",
            "fmha_quant_mode": "fp8",
        },
    },
    {
        "hf_id": "deepseek-ai/DeepSeek-V3",
        "backend": "trtllm",
        "config": {
            "tp_size": 1,
            "pp_size": 1,
            "attention_dp_size": 4,
            "moe_tp_size": 1,
            "moe_ep_size": 4,
            "gemm_quant_mode": "fp8",
            "moe_quant_mode": "fp8_block",
            "kvcache_quant_mode": "fp8",
            "fmha_quant_mode": "fp8",
        },
    },
    {
        "hf_id": "nvidia/DeepSeek-V3.2-NVFP4",
        "backend": "vllm",
        "config": {
            "tp_size": 1,
            "pp_size": 1,
            "attention_dp_size": 4,
            "moe_tp_size": 1,
            "moe_ep_size": 4,
            "gemm_quant_mode": "nvfp4",
            "moe_quant_mode": "nvfp4",
            "kvcache_quant_mode": "fp8",
            "fmha_quant_mode": "bfloat16",
        },
    },
    {
        "hf_id": "nvidia/Kimi-K2.5-NVFP4",
        "backend": "vllm",
        "config": {
            "tp_size": 1,
            "pp_size": 1,
            "attention_dp_size": 4,
            "moe_tp_size": 1,
            "moe_ep_size": 4,
            "gemm_quant_mode": "nvfp4",
            "moe_quant_mode": "nvfp4",
            "kvcache_quant_mode": "fp8",
            "fmha_quant_mode": "bfloat16",
        },
    },
)


def _get_database(system: str, backend: str, version: str):
    from aiconfigurator_core.sdk import perf_database

    return perf_database.get_database(system, backend, version)


def _warm(database) -> None:
    from aiconfigurator_core.sdk.operations.base import warm_all_op_data

    warm_all_op_data(database)


def _build_model(pin: dict):
    from aiconfigurator_core.sdk import common, config, models

    cfg = dict(pin["config"])
    cfg["gemm_quant_mode"] = common.GEMMQuantMode[cfg["gemm_quant_mode"]]
    cfg["moe_quant_mode"] = common.MoEQuantMode[cfg["moe_quant_mode"]]
    cfg["kvcache_quant_mode"] = common.KVCacheQuantMode[cfg["kvcache_quant_mode"]]
    cfg["fmha_quant_mode"] = common.FMHAQuantMode[cfg["fmha_quant_mode"]]
    model_config = config.ModelConfig(**cfg)
    return models.get_model(pin["hf_id"], model_config, backend_name=pin["backend"])


def _snapshot_model_weights(pin: dict) -> dict:
    model = _build_model(pin)
    result = {"context_ops": [], "generation_ops": []}
    for phase in ("context_ops", "generation_ops"):
        for op in getattr(model, phase):
            result[phase].append([type(op).__name__, getattr(op, "_name", ""), repr(float(op.get_weights()))])
    return result


def capture(write: bool) -> dict:
    baseline: dict = {"databases": {}, "weights": {}}

    for system, backend, version in DATABASE_PINS:
        pin_name = f"{system}/{backend}/{version}"
        print(f"== capturing tables: {pin_name}", flush=True)
        db = _get_database(system, backend, version)
        _warm(db)
        entry = snapshot_database_tables(db)
        entry["support_matrix"] = snapshot_support_matrix(db)
        baseline["databases"][pin_name] = entry

    for pin in WEIGHT_MODEL_PINS:
        key = f"{pin['hf_id']}@{pin['backend']}"
        moe_backend = pin["config"].get("moe_backend")
        if moe_backend:
            key += f"+{moe_backend}"
        print(f"== capturing weights: {key}", flush=True)
        baseline["weights"][key] = {"pin": pin, **_snapshot_model_weights(pin)}

    if write:
        BASELINE_PATH.write_text(json.dumps(baseline, indent=1, sort_keys=True) + "\n")
        print(f"wrote {BASELINE_PATH}")
    return baseline


def probe() -> None:
    """Print the attribute-state coverage matrix across pins (no hashing)."""
    from _data_plane_codec import snapshot_attribute

    states: dict[str, dict[str, str]] = {attr: {} for attr in TABLE_ATTRIBUTES}
    for system, backend, version in DATABASE_PINS:
        pin_name = f"{system}/{backend}/{version}"
        print(f"== probing {pin_name}", flush=True)
        try:
            db = _get_database(system, backend, version)
            _warm(db)
        except Exception as exc:  # pragma: no cover - probe diagnostics
            print(f"   FAILED: {exc}")
            for attr in TABLE_ATTRIBUTES:
                states[attr][pin_name] = "ERROR"
            continue
        for attr in TABLE_ATTRIBUTES:
            snap = snapshot_attribute(db, attr)
            state = snap["state"]
            if state == "loaded":
                n = snap.get("n_leaves", sum(s.get("n_leaves", 0) for s in snap.get("subtables", {}).values()))
                state = f"loaded:{n}"
            states[attr][pin_name] = state

    pins = [f"{s}/{b}/{v}" for s, b, v in DATABASE_PINS]
    width = max(len(a) for a in TABLE_ATTRIBUTES) + 1
    print("\n" + " " * width + " | ".join(pins))
    uncovered = []
    for attr, row in states.items():
        cells = [row.get(p, "?") for p in pins]
        print(attr.ljust(width) + " | ".join(c.ljust(len(p)) for c, p in zip(cells, pins, strict=True)))
        if not any(c.startswith("loaded") for c in cells):
            uncovered.append(attr)
    if uncovered:
        print(f"\nNOT COVERED by any pin: {uncovered}")
    else:
        print("\nAll table attributes covered by at least one pin.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--probe", action="store_true", help="print the coverage matrix only")
    mode.add_argument("--write", action="store_true", help="capture and write the baseline JSON")
    mode.add_argument("--dry-run", action="store_true", help="capture but do not write")
    args = parser.parse_args()

    if args.probe:
        probe()
    else:
        capture(write=args.write)


if __name__ == "__main__":
    main()
