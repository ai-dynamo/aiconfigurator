# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure resolver that loads the facts files into a ResolvedFacts dataclass.

This module is intentionally isolated from the pipeline (not imported by pipeline.py).
The pipeline wires it in.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from aiconfigurator.generator.utils import load_backend_version_matrix

_FACTS_DIR = Path(__file__).parent


@dataclass
class ResolvedFacts:
    """Resolved facts for a single generation request.

    Fields
    ------
    backend:         The requested backend name (e.g. ``"trtllm"``).
    dynamo_version:  The requested Dynamo release string (e.g. ``"1.2.0"``).
    backend_version: The backend image version that corresponds to the
                     (dynamo_version, backend) pair looked up in
                     ``runtimes/dynamo.yaml``.
    hardware:        The raw profile dict from ``hardware.yaml`` for the
                     requested hardware profile.
    transport:       The raw profile dict from ``transport.yaml`` for the
                     requested transport profile.
    model:           The raw model dict from ``models.yaml`` for the requested
                     ``model_profile_id``, or ``None`` when the model has no
                     special profile entry (generic path).
    hardware_key:    The bare ``hardware.yaml`` profile key (e.g. ``"gb200"``)
                     that ``hardware`` was resolved from. ``hardware`` itself is
                     the profile *dict* and carries no key inside it, so this is
                     stashed for matching model ``defaults`` ``match.system``.
                     ``None`` when resolved via the pure :func:`resolve_facts`
                     entrypoint (it does not have the bare key in hand).
    """

    backend: str
    dynamo_version: str
    backend_version: str
    hardware: dict
    transport: dict
    model: dict | None
    hardware_key: str | None = None


def resolve_facts(
    *,
    model_profile_id: str | None,
    hardware: str,
    transport: str,
    dynamo_version: str,
    backend: str,
) -> ResolvedFacts:
    """Load facts files and return a :class:`ResolvedFacts` for one request.

    Parameters
    ----------
    model_profile_id:
        Logical model profile key (e.g. ``"deepseek-v4"``).  Pass ``None``
        (or any value absent from ``models.yaml``) to get ``model=None``
        (the generic path).  An absent key is **not** an error.
    hardware:
        Hardware profile key (e.g. ``"gb200"``).  An unknown key raises
        ``KeyError``.
    transport:
        Transport profile key (e.g. ``"nvlink"``).  An unknown key raises
        ``KeyError``.
    dynamo_version:
        Dynamo release string (e.g. ``"1.2.0"``).
    backend:
        Backend name (e.g. ``"trtllm"``).

    Returns
    -------
    ResolvedFacts
    """
    # --- backend_version --------------------------------------------------
    runtimes_path = str(_FACTS_DIR / "runtimes" / "dynamo.yaml")
    matrix = load_backend_version_matrix(runtimes_path)
    try:
        backend_version: str = matrix[dynamo_version][backend]
    except KeyError as exc:
        raise KeyError(
            f"No entry for dynamo_version={dynamo_version!r} backend={backend!r} in {runtimes_path}"
        ) from exc

    # --- hardware ---------------------------------------------------------
    hardware_path = _FACTS_DIR / "hardware.yaml"
    with hardware_path.open(encoding="utf-8") as fh:
        hw_data = yaml.safe_load(fh) or {}
    profiles: dict = hw_data.get("profiles", {})
    if hardware not in profiles:
        raise KeyError(f"Unknown hardware profile {hardware!r}. Available profiles: {sorted(profiles)}")
    hardware_profile: dict = profiles[hardware]

    # --- transport --------------------------------------------------------
    transport_path = _FACTS_DIR / "transport.yaml"
    with transport_path.open(encoding="utf-8") as fh:
        tr_data = yaml.safe_load(fh) or {}
    transport_profiles: dict = tr_data.get("profiles", {})
    if transport not in transport_profiles:
        raise KeyError(f"Unknown transport profile {transport!r}. Available profiles: {sorted(transport_profiles)}")
    transport_profile: dict = transport_profiles[transport]

    # --- model ------------------------------------------------------------
    models_path = _FACTS_DIR / "models.yaml"
    with models_path.open(encoding="utf-8") as fh:
        models_data = yaml.safe_load(fh) or {}
    models: dict = models_data.get("models", {})
    # .get() returns None for absent/None keys — generic path, NOT an error.
    model_profile: dict | None = models.get(model_profile_id) if model_profile_id else None

    return ResolvedFacts(
        backend=backend,
        dynamo_version=dynamo_version,
        backend_version=backend_version,
        hardware=hardware_profile,
        transport=transport_profile,
        model=model_profile,
    )
