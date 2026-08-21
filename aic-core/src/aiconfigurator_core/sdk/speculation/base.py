# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Speculative-decoding scheme abstraction (cost side).

A speculative scheme's *entire* influence on the performance model is
captured by ``SpecSchemeBase``:

* ``verify_width``            — tokens per target verify forward per request
* ``build_draft_*_ops``       — draft-side compute as ops over existing perf tables
* ``draft_weights_bytes``     — draft weight memory (per GPU)
* ``draft_kv_bytes_per_sequence`` — draft KV memory
* ``expected_progress``       — benefit fold (upper layer supplies acceptance)
* ``validate``                — capability gate per (model, backend)

Schemes register with ``@register_spec_scheme("<kind>")`` and are discovered
automatically (same idiom as the model registry). Identity is content-based:
``SpeculationConfig.identity_hash()`` hashes the scheme kind, its explicit
parameters, and the draft checkpoint config — never a display name — so two
same-named community drafts with different configs are mechanically distinct.

Accepted-token progress deliberately stays OUT of this layer: aic-core owns
iteration cost only; the upper prediction layer applies acceptance
assumptions (see ``aiconfigurator.sdk.speculative``).
"""

from __future__ import annotations

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpeculationConfig:
    """Declarative description of a speculative-decoding configuration.

    ``kind`` is a registry key ("none", "mtp", "dspark", ...). ``params``
    are explicit scheme parameters; each scheme documents which keys it
    reads and where their defaults come from. ``draft_model_path`` anchors
    schemes with a separate draft artifact; ``draft_config`` is the parsed
    draft ``config.json`` (callers may inject it directly, e.g. in tests or
    when the checkpoint is already cached).
    """

    kind: str = "none"
    params: dict = field(default_factory=dict)
    draft_model_path: str | None = None
    draft_config: dict | None = None

    def identity_hash(self) -> str:
        """Content hash for cache keys — independent of display names.

        ``draft_model_path`` is part of the identity: when ``draft_config``
        is not injected, the path is what resolves the draft op graph, and
        two configs differing only in it must never share a cached engine.
        """
        payload = {
            "kind": self.kind,
            "params": self.params,
            "draft_model_path": self.draft_model_path,
            "draft_config": self.draft_config,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DraftOpSpec:
    """One draft-side operation plus its per-request token width.

    ``tokens_per_request`` decouples the draft forward width from the target
    verify width (block-drafting schemes run the two at different widths).
    The generation phase queries ``op`` with
    ``x = batch_size * tokens_per_request``.

    ``query_overrides`` pins query kwargs that must not follow the target's
    values — e.g. ``{"s": 135}`` for a sliding-window draft attention whose
    KV length is window-capped regardless of the target context length.
    """

    op: Any
    tokens_per_request: int
    query_overrides: dict | None = None


class SpecSchemeBase(ABC):
    """Interface every speculative scheme implements. See module docstring."""

    kind: ClassVar[str] = "abstract"

    @classmethod
    @abstractmethod
    def from_configs(cls, model_config, spec_config: SpeculationConfig) -> SpecSchemeBase:
        """Construct from a ModelConfig + SpeculationConfig pair."""

    @abstractmethod
    def validate(self, model, backend_name: str) -> None:
        """Raise ValueError when this (model, backend, params) combination is
        not supported. Called by ``get_model`` after construction."""

    @abstractmethod
    def verify_width(self) -> int:
        """Tokens per target verify forward per request (>= 1)."""

    @abstractmethod
    def build_draft_generation_ops(self, model) -> list[DraftOpSpec]:
        """Per-verify-round draft compute. Empty for schemes whose draft cost
        is embedded in the target op graph (MTP)."""

    @abstractmethod
    def build_draft_context_ops(self, model) -> list[DraftOpSpec]:
        """Prefill-side draft precompute (e.g. context-KV build). May be empty."""

    @abstractmethod
    def draft_weights_bytes(self, model) -> float:
        """Draft weight bytes resident per GPU (0.0 when none)."""

    @abstractmethod
    def draft_kv_bytes_per_sequence(self, model, seq_len: int) -> float:
        """Draft KV bytes for one sequence of ``seq_len`` tokens (0.0 when none)."""

    def expected_progress(self, accepted_tokens: float) -> float:
        """Expected output tokens per decode iteration given the acceptance
        assumption. Default: 1 (bonus/anchor) + accepted draft tokens."""
        return 1.0 + accepted_tokens

    def verify_attention_sequence_basis(self) -> bool:
        """Whether decode attention should be priced at the REQUEST batch
        (one KV pass shared by all verify-width query tokens of a request)
        instead of the token batch.

        Block-verify kernels (FlashAttention decode with query_len = w) read
        each request's KV once for all w queries; pricing them as b*w
        independent single-token rows overcounts the KV traffic ~w-fold —
        measured 4.5x on Qwen3-8B TP1 GQA at c16/K7. Sequence basis drops
        the (second-order) extra score compute of the w queries; a proper
        fix is a query_len axis in the attention collection (upstream item).

        Default False: legacy MTP consumers price attention at token batch
        (bit-compat with the historical nextn contract).
        """
        return False


_SPEC_SCHEME_REGISTRY: dict[str, type[SpecSchemeBase]] = {}


def register_spec_scheme(kind: str):
    """Decorator: register ``cls`` as the implementation of ``kind``.

    Mirrors ``models.base.register_model``: a warning (not an error) on
    re-registration keeps test doubles and reloads workable while still
    flagging accidental duplicates.
    """

    def decorator(cls: type[SpecSchemeBase]) -> type[SpecSchemeBase]:
        if kind in _SPEC_SCHEME_REGISTRY:
            logger.warning(
                "Overwriting spec scheme registration for kind %r: %s -> %s",
                kind,
                _SPEC_SCHEME_REGISTRY[kind].__name__,
                cls.__name__,
            )
        _SPEC_SCHEME_REGISTRY[kind] = cls
        return cls

    return decorator


def get_spec_scheme_cls(kind: str) -> type[SpecSchemeBase]:
    cls = _SPEC_SCHEME_REGISTRY.get(kind)
    if cls is None:
        raise ValueError(
            f"Unknown speculative scheme kind: {kind!r}. Registered kinds: {', '.join(sorted(_SPEC_SCHEME_REGISTRY))}"
        )
    return cls


def build_spec_scheme(model_config, spec_config: SpeculationConfig | None) -> SpecSchemeBase:
    """Instantiate the scheme for a resolved SpeculationConfig.

    ``None`` (or kind "none") yields ``NullScheme`` so consumers never
    branch on missing schemes.
    """
    if spec_config is None or spec_config.kind == "none":
        return NullScheme()
    return get_spec_scheme_cls(spec_config.kind).from_configs(model_config, spec_config)


@register_spec_scheme("none")
class NullScheme(SpecSchemeBase):
    """Speculation disabled: all substitutions are identities."""

    kind: ClassVar[str] = "none"

    @classmethod
    def from_configs(cls, model_config, spec_config: SpeculationConfig) -> NullScheme:
        return cls()

    def validate(self, model, backend_name: str) -> None:
        return None

    def verify_width(self) -> int:
        return 1

    def build_draft_generation_ops(self, model) -> list[DraftOpSpec]:
        return []

    def build_draft_context_ops(self, model) -> list[DraftOpSpec]:
        return []

    def draft_weights_bytes(self, model) -> float:
        return 0.0

    def draft_kv_bytes_per_sequence(self, model, seq_len: int) -> float:
        return 0.0
