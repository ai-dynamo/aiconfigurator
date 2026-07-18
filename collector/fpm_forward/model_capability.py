# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-config and declared attention-family resolution for FPM cells."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aiconfigurator.sdk.utils import HuggingFaceDownloadError, _attach_hf_quant_config, _download_hf_json

_ATTENTION_SOURCE_OPS = {
    "dsa_module": frozenset({"dsa_context_module", "dsa_generation_module"}),
    "dsv4_module": frozenset(
        {
            "dsv4_csa_context_module",
            "dsv4_hca_context_module",
            "dsv4_csa_generation_module",
            "dsv4_hca_generation_module",
        }
    ),
    "mla_module": frozenset({"mla_context_module", "mla_generation_module"}),
    "dense_attention": frozenset({"attention_context", "attention_generation"}),
}


@dataclass(frozen=True, slots=True, init=False)
class ResolvedModelConfig:
    """Immutable model-config evidence used to derive and hash an FPM plan."""

    source_kind: str
    source_reference: str
    sha256: str
    _payload_json: str

    def __init__(self, payload: dict[str, Any], *, source_kind: str, source_reference: str) -> None:
        if not isinstance(payload, dict):
            raise TypeError("model config must be a mapping")
        if not payload:
            raise ValueError(f"model config must not be empty: {source_reference}")
        payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        object.__setattr__(self, "source_kind", source_kind)
        object.__setattr__(self, "source_reference", source_reference)
        object.__setattr__(self, "sha256", hashlib.sha256(payload_json.encode()).hexdigest())
        object.__setattr__(self, "_payload_json", payload_json)

    @property
    def payload(self) -> dict[str, Any]:
        """Return a detached source-config snapshot."""

        return json.loads(self._payload_json)

    @property
    def effective_payload(self) -> dict[str, Any]:
        """Return the text-model view while preserving top-level quant metadata."""

        payload = self.payload
        text_config = payload.get("text_config")
        if isinstance(text_config, dict):
            payload = {**payload, **text_config}
        return payload

    def to_dict(self) -> dict[str, object]:
        return {
            "source_kind": self.source_kind,
            "sha256": self.sha256,
            "payload": self.payload,
        }


def _aic_cached_model_config_path(model_path: str) -> Path | None:
    root = Path(__file__).resolve().parents[2]
    cached = root / "src" / "aiconfigurator" / "model_configs" / f"{model_path.replace('/', '--')}_config.json"
    return cached if cached.exists() else None


def _load_json_mapping(path: Path, *, description: str) -> tuple[dict[str, Any], Path]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"{description} must be a mapping: {path}")
    return payload, path


def _attach_sibling_quant_config(payload: dict[str, Any], config_path: Path, *, cached: bool) -> dict[str, Any]:
    suffix = "_hf_quant_config.json" if cached else "hf_quant_config.json"
    quant_path = (
        config_path.with_name(config_path.name.removesuffix("_config.json") + suffix)
        if cached
        else config_path.parent / suffix
    )
    if not quant_path.is_file():
        return payload
    quant_config, _ = _load_json_mapping(quant_path, description="model quantization config")
    return _attach_hf_quant_config(payload, quant_config)


def _local_config_path(raw_path: str, *, explicit: bool) -> Path | None:
    path = Path(raw_path).expanduser()
    if path.is_dir():
        config_path = path / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"model config does not exist: {config_path}")
        return config_path
    if path.is_file():
        return path
    if explicit or path.is_absolute() or raw_path.startswith(("./", "../", "~")):
        raise FileNotFoundError(f"model config does not exist: {path}")
    return None


def _load_file_config(path: Path, *, source_kind: str, cached: bool = False) -> ResolvedModelConfig:
    payload, _ = _load_json_mapping(path, description="model config")
    payload = _attach_sibling_quant_config(payload, path, cached=cached)
    return ResolvedModelConfig(
        payload,
        source_kind=source_kind,
        source_reference=str(path.resolve()),
    )


def _download_model_config(model_path: str) -> ResolvedModelConfig:
    try:
        payload = _download_hf_json(model_path, "config.json", raise_on_404=True)
        quant_config = _download_hf_json(model_path, "hf_quant_config.json", raise_on_404=False)
    except HuggingFaceDownloadError as error:
        raise ValueError(
            f"cannot resolve a real model config for {model_path!r} from local files, "
            f"the AIC cache, or Hugging Face: {error}"
        ) from error
    if payload is None:
        raise ValueError(f"Hugging Face returned no config.json for model {model_path!r}")
    if not isinstance(payload, dict):
        raise TypeError(f"Hugging Face config.json must be a mapping for model {model_path!r}")
    if quant_config is not None:
        if not isinstance(quant_config, dict):
            raise TypeError(f"Hugging Face hf_quant_config.json must be a mapping for model {model_path!r}")
        payload = _attach_hf_quant_config(payload, quant_config)
    return ResolvedModelConfig(
        payload,
        source_kind="huggingface",
        source_reference=model_path,
    )


def load_model_config(model_path: str, *, explicit_config_path: str | None = None) -> ResolvedModelConfig:
    """Resolve real model metadata or fail before any FPM plan is generated."""

    if explicit_config_path is not None:
        explicit = _local_config_path(explicit_config_path, explicit=True)
        assert explicit is not None
        return _load_file_config(explicit, source_kind="explicit")

    local = _local_config_path(model_path, explicit=False)
    if local is not None:
        return _load_file_config(local, source_kind="local")

    cached = _aic_cached_model_config_path(model_path)
    if cached is not None:
        return _load_file_config(cached, source_kind="aic_cache", cached=True)

    return _download_model_config(model_path)


def resolve_attention_source(selected_ops: set[str], *, required: bool = True) -> str | None:
    """Return the unique attention family declared by the model case plan."""

    matches = [name for name, ops in _ATTENTION_SOURCE_OPS.items() if ops.issubset(selected_ops)]
    if not matches:
        if not required:
            return None
        raise ValueError(
            "fpm_forward requires one supported AIC context/generation attention pair; "
            f"selected ops were {sorted(selected_ops)}"
        )
    if len(matches) > 1:
        raise ValueError(
            "fpm_forward attention source is ambiguous; model cases must select one pair: " + ", ".join(matches)
        )
    return matches[0]
