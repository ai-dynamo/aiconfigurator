# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Resolve version-specific collector modules from registry entries.

Given a registry entry and a runtime framework version, returns the
appropriate module path. For unversioned entries, returns the module
directly. For versioned entries, picks the highest min_version that
does not exceed the runtime version.
"""

from __future__ import annotations

import re

_CONSTRAINT_RE = re.compile(r"(>=|<=|>|<|==|!=)\s*([^\s,]+)")


_PRE_RELEASE_TAG_RE = re.compile(r"(dev|a|alpha|b|beta|rc)(\d*)", re.IGNORECASE)
_POST_RELEASE_TAG_RE = re.compile(r"(post)(\d*)", re.IGNORECASE)

# Ordering weights for the suffix slot:
#   dev/alpha/beta/rc < 0 (below release)
#   (no suffix) = 0      (final release)
#   post > 0              (above release)
_SUFFIX_RANK = {"dev": -4, "a": -3, "alpha": -3, "b": -2, "beta": -2, "rc": -1}


def _normalize_version(v: str) -> tuple[int, ...]:
    """Parse a PEP 440-ish version into a comparable tuple.

    The returned tuple is ``(*numeric_parts, suffix_rank, suffix_num)``
    so that pre-release < release < post-release:

        "1.2.0dev1"   -> (1, 2, 0, -4, 1)   # dev < alpha < beta < rc < release
        "1.2.0a2"     -> (1, 2, 0, -3, 2)
        "1.2.0rc2"    -> (1, 2, 0, -1, 2)
        "1.2.0"       -> (1, 2, 0,  0, 0)    # final release
        "0.5.5.post2" -> (0, 5, 5,  1, 2)    # post > release
        "0.20.0+cu124"-> (0, 20, 0, 0, 0)    # local metadata ignored
    """
    # Strip local metadata (+cu124, +git.abc)
    v = v.split("+")[0]

    # Split on the first pre/post tag boundary
    m_pre = _PRE_RELEASE_TAG_RE.search(v)
    m_post = _POST_RELEASE_TAG_RE.search(v)

    if m_pre:
        base_str = v[: m_pre.start()].rstrip(".")
        tag = m_pre.group(1).lower()
        tag_num = int(m_pre.group(2)) if m_pre.group(2) else 0
        suffix_rank = _SUFFIX_RANK.get(tag, -1)
    elif m_post:
        base_str = v[: m_post.start()].rstrip(".")
        tag_num = int(m_post.group(2)) if m_post.group(2) else 0
        suffix_rank = 1
    else:
        base_str = v
        tag_num = 0
        suffix_rank = 0

    parts: list[int] = []
    for seg in base_str.split("."):
        try:
            parts.append(int(seg))
        except ValueError:
            break

    return (*parts, suffix_rank, tag_num) if parts else (0, 0, 0)


def _check_compat(compat_str: str, runtime_version: str) -> bool:
    """Check if runtime_version satisfies a __compat__ specifier.

    The specifier format is: ``<framework><constraints>``
    where constraints are comma-separated comparisons.
    Examples: "trtllm>=1.1.0", "trtllm>=0.21.0,<1.1.0"

    The framework prefix is stripped; only version constraints are evaluated.
    """
    # Strip framework name prefix (everything before the first operator)
    spec = re.sub(r"^[a-zA-Z_]+", "", compat_str)
    rv = _normalize_version(runtime_version)

    _ops = {
        ">=": lambda a, b: a >= b,
        "<=": lambda a, b: a <= b,
        ">": lambda a, b: a > b,
        "<": lambda a, b: a < b,
        "==": lambda a, b: a == b,
        "!=": lambda a, b: a != b,
    }

    for match in _CONSTRAINT_RE.finditer(spec):
        op, ver = match.group(1), match.group(2)
        if not _ops[op](rv, _normalize_version(ver)):
            return False
    return True


def resolve_module(entry: dict, runtime_version: str) -> str | None:
    """Return the collector module path for a registry entry.

    Args:
        entry: A registry dict. Must have either "module" (unversioned)
               or "versions" (list of (min_version, module) tuples,
               sorted descending by min_version).
        runtime_version: The framework version string detected at runtime.

    Returns:
        Module path string, or None if no version matches.
    """
    if "versions" not in entry:
        return entry["module"]

    rv = _normalize_version(runtime_version)
    for min_ver_str, module in entry["versions"]:
        if rv >= _normalize_version(min_ver_str):
            return module
    return None


def build_collections(
    registry: list[dict],
    backend_name: str,
    runtime_version: str,
    ops: list[str] | None = None,
    *,
    logger=None,
) -> list[dict]:
    """Build the collections list from a registry.

    Args:
        registry: The backend's REGISTRY list.
        backend_name: e.g. "trtllm", "vllm", "sglang".
        runtime_version: Framework version string.
        ops: Optional filter — only include these op types.
        logger: Optional logger for warnings.

    Returns:
        List of collection dicts ready for collect_ops().
    """
    collections = []
    for entry in registry:
        if ops and entry["op"] not in ops:
            continue

        module = resolve_module(entry, runtime_version)
        if module is None:
            if logger:
                logger.warning(f"Skipping {backend_name}.{entry['op']} — no collector for v{runtime_version}")
            continue

        # Sanity-check: if the resolved module declares __compat__,
        # verify it agrees with the runtime version.
        if logger:
            try:
                mod = __import__(module, fromlist=["__compat__"])
                declared = getattr(mod, "__compat__", None)
                if declared and not _check_compat(declared, runtime_version):
                    logger.warning(
                        f"{backend_name}.{entry['op']}: module {module} declares "
                        f"__compat__={declared!r} but runtime is v{runtime_version}"
                    )
            except ImportError:
                pass  # module can't be imported yet (missing framework); skip check

        collections.append(
            {
                "name": backend_name,
                "type": entry["op"],
                "module": module,
                "get_func": entry["get_func"],
                "run_func": entry["run_func"],
            }
        )

    return collections
