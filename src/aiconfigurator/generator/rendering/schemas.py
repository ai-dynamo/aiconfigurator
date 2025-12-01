# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Optional

import yaml

from .engine import evaluate_expression

_SCHEMA_CACHE: dict[str, list[dict[str, Any]]] = {}
_DEFAULT_BACKEND = "trtllm"
_BASE_DIR = Path(__file__).resolve().parent
_CONFIG_DIR = _BASE_DIR.parent / "config"
_SCHEMA_FILE = (_CONFIG_DIR / "deployment_config.yaml").resolve()


def _load_schema_inputs(schema_path: str) -> list[dict[str, Any]]:
    path = os.path.abspath(schema_path)
    cached = _SCHEMA_CACHE.get(path)
    if cached is not None:
        return cached
    try:
        with open(path, encoding="utf-8") as f:
            schema = yaml.safe_load(f) or []
        inputs: list[dict[str, Any]]
        if isinstance(schema, list):
            inputs = schema
        else:
            inputs = schema.get("inputs", []) or []
    except Exception:
        inputs = []
    _SCHEMA_CACHE[path] = inputs
    return inputs


def _normalize_backend(backend: Optional[str]) -> str:
    if backend:
        return str(backend).strip().lower()
    return _DEFAULT_BACKEND


def _entry_allows_backend(entry: dict[str, Any], backend: str) -> bool:
    allowed = entry.get("backends")
    if not allowed:
        return True
    if isinstance(allowed, str):
        allowed_set = {allowed.strip().lower()}
    elif isinstance(allowed, Iterable):
        allowed_set = {str(item).strip().lower() for item in allowed if item is not None}
    else:
        allowed_set = set()
    return not allowed_set or backend in allowed_set


def _select_backend_default(entry: dict[str, Any], backend: str) -> Any:
    backend_defaults = entry.get("backend_defaults")
    if isinstance(backend_defaults, dict):
        for key, value in backend_defaults.items():
            if isinstance(key, str) and key.strip().lower() == backend:
                return value
    return entry.get("default")


def apply_defaults(group: str, cfg: dict[str, Any], backend: Optional[str] = None) -> dict[str, Any]:
    inputs = _load_schema_inputs(str(_SCHEMA_FILE))
    eval_ctx = {group: dict(cfg)}
    out = dict(cfg)
    backend_key = _normalize_backend(backend)
    for entry in inputs:
        key = entry.get("key")
        if not key or not key.startswith(group + "."):
            continue
        if not _entry_allows_backend(entry, backend_key):
            continue
        parts = key.split(".")
        subkey = parts[1] if len(parts) > 1 else None
        if not subkey:
            continue
        if out.get(subkey) is None:
            default_expr = _select_backend_default(entry, backend_key)
            if default_expr is not None:
                try:
                    val = evaluate_expression(str(default_expr), eval_ctx)
                except Exception:
                    val = default_expr
                out[subkey] = val
    return out
