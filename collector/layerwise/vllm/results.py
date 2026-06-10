# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CSV schema and row writing helpers for vLLM layerwise collection."""

from __future__ import annotations

import csv
import fcntl
import os
from pathlib import Path
from typing import Any


CSV_COLUMNS = [
    "framework", "framework_version", "system", "model", "attn_tp", "moe_tp", "ep",
    "num_slots", "gemm_quant", "moe_quant", "attn_quant", "kv_quant", "phase",
    "batch_size", "new_tokens", "past_kv", "layer_type", "layer_index",
    "measured_layer_count", "layer_multiplier", "latency_ms", "rms_latency_ms",
    "rms_kernel_count", "includes_moe", "vllm_config_hash",
]


def _write_csv_header_if_needed(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    with path.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_COLUMNS).writeheader()

def _append_success_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writerow({k: row.get(k, "") for k in CSV_COLUMNS})
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f, fcntl.LOCK_UN)

def _work_unit_includes_moe(work_unit: WorkUnit) -> bool:
    return bool(work_unit.includes_moe)
