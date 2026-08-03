# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve serving-visible attention backends before querying performance data.

The resolver deliberately does not inspect which performance rows happen to be
available.  An explicit request must stay explicit; only ``None`` / ``auto`` may
follow a framework default.  This keeps prediction and generated deployment
configuration on the same backend.
"""

from __future__ import annotations

from dataclasses import dataclass

_WIDEEP_MLA_BACKENDS = frozenset({"fa3", "flashinfer", "trtllm_mla"})

# SGLang defaults are versioned because they are framework policy, not a GPU
# capability rule. Keep this table limited to versions whose DeepSeek WideEP
# MLA behavior has been reviewed against the framework or collector contract.
_SGLANG_DEEPSEEK_WIDEEP_MLA_DEFAULTS: dict[str, dict[int, str]] = {
    "0.5.6.post2": {90: "fa3"},
    "0.5.9": {90: "fa3", 100: "trtllm_mla", 103: "trtllm_mla"},
    "0.5.10": {90: "fa3", 100: "trtllm_mla", 103: "trtllm_mla"},
    "0.5.12": {90: "fa3", 100: "trtllm_mla", 103: "trtllm_mla"},
    "0.5.14": {90: "fa3", 100: "trtllm_mla", 103: "trtllm_mla"},
}


@dataclass(frozen=True)
class ResolvedAttentionBackend:
    """Requested and effective backend plus how the decision was made."""

    requested: str | None
    effective: str
    source: str  # ``explicit`` or ``framework_default``


def resolve_wideep_mla_attention_backend(
    requested_backend: str | None,
    *,
    framework: str,
    framework_version: str | None,
    model_family: str,
    sm_version: int,
) -> ResolvedAttentionBackend:
    """Resolve the effective WideEP MLA backend without consulting perf rows.

    Explicit requests are returned unchanged so missing exact measurements fail
    closed in the normal database lookup.  Automatic resolution is deliberately
    version/model/hardware-specific and errors when no reviewed rule exists.
    """

    requested = requested_backend.lower() if isinstance(requested_backend, str) else None
    if requested not in _WIDEEP_MLA_BACKENDS | {None, "auto"}:
        supported = ", ".join(sorted(_WIDEEP_MLA_BACKENDS | {"auto"}))
        raise ValueError(
            f"Unsupported WideEP MLA attention backend {requested_backend!r}; expected one of {supported}."
        )

    if requested not in {None, "auto"}:
        return ResolvedAttentionBackend(requested=requested, effective=requested, source="explicit")

    normalized_framework = framework.lower()
    normalized_family = model_family.upper()
    normalized_version = framework_version.removeprefix("v") if framework_version else None
    if normalized_framework != "sglang" or normalized_family != "DEEPSEEK":
        raise ValueError(
            "No automatic WideEP MLA attention-backend policy for "
            f"framework={framework!r}, model_family={model_family!r}."
        )

    defaults_by_sm = _SGLANG_DEEPSEEK_WIDEEP_MLA_DEFAULTS.get(normalized_version or "")
    effective = defaults_by_sm.get(sm_version) if defaults_by_sm else None
    if effective is None:
        raise ValueError(
            "No reviewed SGLang DeepSeek WideEP MLA default for "
            f"version={framework_version!r}, sm_version={sm_version}; set attention_backend explicitly."
        )

    return ResolvedAttentionBackend(
        requested=requested,
        effective=effective,
        source="framework_default",
    )
