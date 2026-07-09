# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hardware capability floors and the hang denylist.

Both are flat, generation-time filters applied before cases are queued:

- ``capabilities.yaml``: positive dtype/op -> min SM floors. A case whose
  dtype requires a newer SM than the target is never queued.
- ``denylist.yaml``: substring matches for cases that hang or kill the node.
  Ordinary crashes do not belong there -- they fail fast, get classified in
  the failure log with a (model, dtype) group label, and systemic groups are
  surfaced in the collection summary as fix-me signals (no auto-skip).

Neither file supports shape conditions, framework-version conditions, or
rule operators. That is intentional; do not add them.
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_CASES_DIR = Path(__file__).resolve().parent / "cases"

# Case attribute names that carry a dtype/quantization mode.
_DTYPE_FIELDS = ("moe_type", "gemm_type", "dtype", "kv_cache_dtype", "compute_dtype")
# Boolean case attributes that imply an FP8 requirement.
_FP8_FLAG_FIELDS = ("use_fp8_kv_cache", "use_fp8_context_fmha")


@functools.lru_cache(maxsize=1)
def _load_capabilities() -> tuple[dict[str, int], dict[str, int]]:
    path = _CASES_DIR / "capabilities.yaml"
    data = yaml.safe_load(path.read_text()) or {}
    dtype_min_sm = {str(k): int(v) for k, v in (data.get("dtype_min_sm") or {}).items()}
    op_min_sm = {str(k): int(v) for k, v in (data.get("op_min_sm") or {}).items()}
    return dtype_min_sm, op_min_sm


@functools.lru_cache(maxsize=1)
def _load_denylist() -> tuple[tuple[str, str], ...]:
    path = _CASES_DIR / "denylist.yaml"
    if not path.exists():
        return ()
    data = yaml.safe_load(path.read_text()) or {}
    entries = []
    for entry in data.get("entries") or []:
        contains = str(entry["contains"])
        reason = str(entry.get("reason", ""))
        entries.append((contains, reason))
    return tuple(entries)


def case_dtypes(test_case) -> list[str]:
    """Return the dtype/quant names a case requires (typed attribute access)."""
    dtypes = []
    for field in _DTYPE_FIELDS:
        value = getattr(test_case, field, None)
        if isinstance(value, str) and value:
            dtypes.append(value)
    for field in _FP8_FLAG_FIELDS:
        if getattr(test_case, field, False):
            dtypes.append("fp8")
    return dtypes


def unsupported_reason(test_case, op: str, sm_version: int | None) -> str | None:
    """Return why a case is below the hardware floor, or None if supported.

    Unknown dtypes and unknown ops are permissive (no floor): the capability
    table only encodes known hardware facts, never guesses.
    """
    if sm_version is None:
        return None
    dtype_min_sm, op_min_sm = _load_capabilities()
    op_floor = op_min_sm.get(op)
    if op_floor is not None and sm_version < op_floor:
        return f"op {op} requires SM>={op_floor}"
    for dtype in case_dtypes(test_case):
        floor = dtype_min_sm.get(dtype)
        if floor is not None and sm_version < floor:
            return f"dtype {dtype} requires SM>={floor}"
    return None


def denylist_reason(test_case) -> str | None:
    """Return the denylist reason matching this case, or None."""
    case_str = str(test_case)
    for contains, reason in _load_denylist():
        if contains in case_str:
            return reason or f"denylisted (contains {contains!r})"
    return None


def filter_cases(test_cases, op: str, sm_version: int | None):
    """Apply capability floors and the denylist before queueing.

    Returns (kept_cases, dropped) where dropped is a list of
    (test_case, kind, reason) with kind in {"capability", "denylist"}.
    """
    kept = []
    dropped = []
    for test_case in test_cases:
        reason = unsupported_reason(test_case, op, sm_version)
        if reason is not None:
            dropped.append((test_case, "capability", reason))
            continue
        deny = denylist_reason(test_case)
        if deny is not None:
            dropped.append((test_case, "denylist", deny))
            continue
        kept.append(test_case)
    if dropped:
        by_reason: dict[str, int] = {}
        for _, _, reason in dropped:
            by_reason[reason] = by_reason.get(reason, 0) + 1
        summary = "; ".join(f"{count}x {reason}" for reason, count in sorted(by_reason.items()))
        logger.info(f"{op}: dropped {len(dropped)}/{len(test_cases)} cases before queueing ({summary})")
    return kept, dropped


def detect_sm_version() -> int | None:
    """Detect the SM version of the local CUDA device, or None (e.g. XPU)."""
    try:
        import torch

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            return major * 10 + minor
    except Exception:
        pass
    return None
