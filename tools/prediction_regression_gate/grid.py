# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tier-1 sampling grid for the prediction-regression-gate static snapshot.

Everything here is harness *policy*. Both revisions of an old-vs-new
comparison run their own copy of this grid, so changing any list below shows
up in the comparison report as added/removed rows — attributed to the PR that
changed it, exactly like a code change.

Axes:
  - shape: fixed prefill (static_ctx) and decode (static_gen) points
  - parallelism: per model family (dense vs MoE)
  - quant: model default plus explicit overrides (nvfp4 only on systems whose
    data can contain it)
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "src" / "aiconfigurator" / "systems" / "data"
MODEL_CONFIG_DIR = REPO_ROOT / "src" / "aiconfigurator" / "model_configs"

# Non-engine data dirs living next to backend dirs in the data tree.
NON_ENGINE_BACKENDS = {"nccl", "oneccl"}
METADATA_FILES = {"SHARED_LAYER_REUSE.txt", "INCOMPLETE.txt", "reuse.yaml", "collection_meta.yaml"}


def _dir_is_incomplete(path: str) -> bool:
    """Yaml-first partial-dir check (collection_meta.yaml status:partial), with
    INCOMPLETE.txt as the legacy fallback. Duplicated (not imported) from
    aiconfigurator_core.sdk.perf_database._version_dir_state, the source of
    truth for this semantic — kept local so this tool doesn't take an aic-core
    dependency for one predicate. Malformed collection_meta.yaml raises
    ValueError naming the file, matching that canonical loader's fail-loudly
    behavior (unlike operations/base.py's deliberately lenient hot-path
    duplicate of this same predicate). See the CONTRACT NOTE on
    _version_dir_is_partial in
    aic-core/src/aiconfigurator_core/sdk/operations/base.py
    for the intentional resolver-lenient/admission-strict split and
    the full list of copies."""
    meta_path = os.path.join(path, "collection_meta.yaml")
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"{meta_path}: failed to parse collection_meta.yaml: {e}") from e
        tables = meta.get("tables") if isinstance(meta, dict) else None
        return isinstance(tables, dict) and any(
            isinstance(t, dict) and t.get("status") == "partial" for t in tables.values()
        )
    return os.path.isfile(os.path.join(path, "INCOMPLETE.txt"))


# Legacy top-level backend dirs. Family-first layout (Collector V3) treats any
# other first-level directory under a system dir as a family dir containing
# <backend>/<version> subtrees. Keep this set textually identical to
# audit_kernel_source.py's _LEGACY_BACKEND_DIRS union _SKIP_BACKEND_DIRS
# (tools/perf_database/audit_kernel_source.py).
_LEGACY_BACKEND_DIRS = {"trtllm", "sglang", "vllm"}

# --------------------------------------------------------------------------
# Shape axis
# --------------------------------------------------------------------------
# (batch_size, isl); prefill runs mode=static_ctx with osl=CTX_OSL, decode
# runs mode=static_gen with osl=GEN_OSL and the default stride.
PREFILL_POINTS: list[tuple[int, int]] = [
    (1, 1024),
    (1, 8192),
    (1, 32768),
    (4, 4096),
]
DECODE_POINTS: list[tuple[int, int]] = [
    (1, 1024),
    (32, 1024),
    (128, 1024),
    (32, 8192),
    (8, 32768),
]
CTX_OSL = 8
GEN_OSL = 256
STRIDE = 32

# --------------------------------------------------------------------------
# Parallelism axis: (tp, pp, adp, moe_tp, moe_ep)
# --------------------------------------------------------------------------
DENSE_PARALLEL: list[tuple[int, int, int, int | None, int | None]] = [
    (1, 1, 1, None, None),
    (4, 1, 1, None, None),
    (8, 1, 1, None, None),
]
# Constraint: tp * adp * cp == moe_tp * moe_ep (parallelism width match).
MOE_PARALLEL: list[tuple[int, int, int, int | None, int | None]] = [
    (4, 1, 1, 1, 4),
    (8, 1, 1, 1, 8),
    (1, 1, 8, 1, 8),  # attention-DP layout (DeepSeek-style wide-EP decode)
]

# --------------------------------------------------------------------------
# Quant axis: label -> (gemm_quant_mode, moe_quant_mode) overrides.
# "default" keeps the model config's own quantization.
# --------------------------------------------------------------------------
QUANT_VARIANTS: list[tuple[str, str | None, str | None]] = [
    ("default", None, None),
    ("fp8", "fp8", "fp8"),
]
# Only meaningful where NVFP4 silicon can exist (sm100+ family systems).
NVFP4_SYSTEMS = {"b200_sxm", "b300_sxm", "gb200", "gb300"}
NVFP4_VARIANT: tuple[str, str | None, str | None] = ("nvfp4", "nvfp4", "nvfp4")


@dataclass(frozen=True)
class Combo:
    system: str
    backend: str
    version: str

    @property
    def relpath(self) -> str:
        return os.path.join(self.system, self.backend, f"{self.version}.csv")


def quant_variants_for(system: str) -> list[tuple[str, str | None, str | None]]:
    variants = list(QUANT_VARIANTS)
    if system in NVFP4_SYSTEMS:
        variants.append(NVFP4_VARIANT)
    return variants


def bundled_models() -> list[str]:
    """Offline-buildable bundled models (HF-download quant variants excluded)."""
    names = []
    for p in sorted(MODEL_CONFIG_DIR.glob("*_config.json")):
        name = p.name.removesuffix("_config.json").replace("--", "/")
        if name.endswith("_hf_quant"):
            continue
        names.append(name)
    return names


def _version_sort_key(version: str) -> tuple:
    """PEP 440 ordering where possible; unparseable names sort last, lexically."""
    from aiconfigurator.sdk.common import parse_support_matrix_version

    parsed = parse_support_matrix_version(version)
    return (1, version) if parsed is None else (0, parsed)


def _iter_backend_dirs(sys_dir: Path) -> Iterable[tuple[str, Path]]:
    """Yield (backend, backend_dir) for every backend dir under a system dir,
    across both the legacy (<backend>/<version>) and family-first
    (<family>/<backend>/<version>) layouts. NON_ENGINE_BACKENDS entries are
    excluded at whichever level they appear (top-level or inside a family dir).
    """
    for entry in sorted(sys_dir.iterdir()):
        if not entry.is_dir() or entry.name in NON_ENGINE_BACKENDS:
            continue
        if entry.name in _LEGACY_BACKEND_DIRS:
            yield entry.name, entry
        else:  # family dir
            for backend_dir in sorted(entry.iterdir()):
                if not backend_dir.is_dir() or backend_dir.name in NON_ENGINE_BACKENDS:
                    continue
                yield backend_dir.name, backend_dir


def _version_dirs(system: str, backend: str) -> list[str]:
    """Data-carrying version dirs for (system, backend), merged across every
    family dir that contributes to that backend (e.g. gemm/vllm and moe/vllm
    both feed the "vllm" backend)."""
    sys_dir = DATA_ROOT / system
    if not sys_dir.is_dir():
        return []
    versions: set[str] = set()
    for found_backend, backend_dir in _iter_backend_dirs(sys_dir):
        if found_backend != backend:
            continue
        for v in os.listdir(backend_dir):
            vdir = backend_dir / v
            if not vdir.is_dir():
                continue
            files = [f for f in os.listdir(vdir) if f not in METADATA_FILES]
            # Marker-only dirs have nothing to test with the shared layer off;
            # partial dirs (collection_meta.yaml status:partial, or legacy
            # INCOMPLETE.txt) are excluded from loading entirely.
            if files and not _dir_is_incomplete(str(vdir)):
                versions.add(v)
    return sorted(versions, key=_version_sort_key)


def enumerate_combos(
    systems: list[str] | None = None,
    backends: list[str] | None = None,
    versions: str = "latest",
) -> list[Combo]:
    """Enumerate (system, backend, version) combos backed by real local data.

    versions="latest" keeps only the newest data-carrying version per
    (system, backend) — the PR profile. "all" keeps every data-carrying
    version — the scheduled/full profile.
    """
    from aiconfigurator.sdk.perf_database import get_latest_database_version

    combos: list[Combo] = []
    for system in sorted(os.listdir(DATA_ROOT)):
        if systems and system not in systems:
            continue
        sys_dir = DATA_ROOT / system
        if not sys_dir.is_dir() or not (DATA_ROOT.parent / f"{system}.yaml").exists():
            continue
        for backend in sorted({b for b, _ in _iter_backend_dirs(sys_dir)}):
            if backends and backend not in backends:
                continue
            data_versions = _version_dirs(system, backend)
            if not data_versions:
                continue
            if versions == "all":
                combos.extend(Combo(system, backend, v) for v in data_versions)
            else:
                # Resolve "latest" with the SDK's version ordering, restricted
                # to data-carrying dirs so marker-only versions never win.
                latest = get_latest_database_version(system=system, backend=backend)
                if latest not in data_versions:
                    latest = data_versions[-1]
                combos.append(Combo(system, backend, latest))
    return combos
