# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for generator modules."""

from __future__ import annotations

import os
from functools import cache
from typing import Any, Optional

import yaml

DEFAULT_BACKEND = "trtllm"
GENERATOR_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "config")
DEFAULT_BACKEND_VERSION_MATRIX_PATH = os.path.join(GENERATOR_CONFIG_DIR, "backend_version_matrix.yaml")


def normalize_backend(backend: Optional[str], default: str = DEFAULT_BACKEND) -> str:
    """Normalize backend names to lowercase strings with a fallback."""
    if backend:
        return str(backend).strip().lower()
    return default


def coerce_bool(value: Optional[Any]) -> Optional[bool]:
    """Best-effort conversion of user input into booleans."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return bool(value)


def coerce_int(value: Optional[Any]) -> Optional[int]:
    """Convert values to ints while swallowing Type/Value errors."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _load_yaml_payload(path: str) -> Any:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@cache
def load_backend_version_matrix(matrix_path: str) -> dict[str, dict[str, Any]]:
    payload = _load_yaml_payload(matrix_path)
    if not isinstance(payload, dict):
        raise TypeError(f"Backend version matrix must be a YAML mapping: {matrix_path}")
    matrix = payload.get("matrix", payload)
    if not isinstance(matrix, dict):
        raise TypeError(f"Backend version matrix missing 'matrix' mapping: {matrix_path}")
    return matrix


def get_default_dynamo_version_mapping(
    matrix_path: str = DEFAULT_BACKEND_VERSION_MATRIX_PATH,
) -> tuple[str, dict[str, Any]]:
    """
    Return the default Dynamo version and its backend-version mapping.

    The default entry is the first item in backend_version_matrix.yaml.
    """
    matrix = load_backend_version_matrix(matrix_path)
    if not matrix:
        raise ValueError(f"Backend version matrix is empty: {matrix_path}")
    dynamo_version, entry = next(iter(matrix.items()))
    if not isinstance(entry, dict):
        raise TypeError(f"Invalid backend version entry for {dynamo_version}: {entry!r}")
    return str(dynamo_version), entry


def resolve_backend_version_for_dynamo(
    dynamo_version: str,
    backend: str | None = None,
    matrix_path: str = DEFAULT_BACKEND_VERSION_MATRIX_PATH,
) -> str:
    """
    Given a Dynamo (generator) version, look up the corresponding backend version for a specified backend.

    Parameters:
        dynamo_version (str): The target Dynamo generator release (e.g., "v0.8.1").
        backend (str | None): Name of the backend to look up ("trtllm", "vllm", "sglang", or "any").
        matrix_path (str): Path to the backend version matrix YAML file.

    Returns:
        str | dict: The backend version(str) for the given backend, or a dict of versions if backend is "any" or None.

    Raises:
        ValueError: If the dynamo_version is missing or not present in the matrix.
        TypeError: If the loaded matrix or entry is invalid, or if no mapping exists for the given backend.
    """
    version_key = str(dynamo_version).strip()
    if not version_key:
        raise ValueError("dynamo_version must be a non-empty string.")
    matrix = load_backend_version_matrix(matrix_path)
    entry = matrix.get(version_key)
    if not isinstance(entry, dict):
        supported = ", ".join(sorted(matrix.keys()))
        raise TypeError(f"Unsupported dynamo_version '{version_key}'. Supported versions: {supported or 'none'}.")

    backend_key = normalize_backend(backend, DEFAULT_BACKEND)
    # return all backend versions for "any" backend
    if not backend or backend_key == "any":
        return entry

    backend_version = entry.get(backend_key)
    if backend_version is None:
        supported_backends = ", ".join(sorted(entry.keys()))
        raise ValueError(
            f"No backend version mapping for backend '{backend_key}' in dynamo '{version_key}'. "
            f"Supported backends: {supported_backends or 'none'}."
        )
    return str(backend_version)
