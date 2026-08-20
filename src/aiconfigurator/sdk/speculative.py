# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Workload-level speculative-decoding progress assumptions.

``aic-core`` predicts the cost of one decode/verification iteration from ``nextn``.
This module converts that iteration cost into expected service metrics using
the average number of accepted draft tokens. Keeping the projection here means
the same ``aic-core`` engine can be reused across acceptance-rate sweeps.
"""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass
from typing import Literal

from aiconfigurator.sdk.config_builders import normalize_nextn
from aiconfigurator.sdk.inference_summary import InferenceSummary

logger = logging.getLogger(__name__)

ProjectionRole = Literal["agg", "prefill", "decode", "static"]


def normalize_speculative_decoding(
    nextn: int | None,
    nextn_accepted: float | None,
) -> tuple[int, float | None]:
    """Normalize public MTP inputs.

    Active MTP requires an explicit acceptance assumption. When MTP is
    disabled, the value is retained for compatibility but ignored by
    prediction.
    """
    normalized_nextn = normalize_nextn(nextn)
    if normalized_nextn <= 0:
        return normalized_nextn, nextn_accepted

    if nextn_accepted is None:
        raise ValueError(
            f"nextn={normalized_nextn} requires 'nextn_accepted' (average accepted draft tokens "
            f"per step, 0 <= nextn_accepted <= nextn); there is no built-in acceptance assumption."
        )
    accepted = float(nextn_accepted)
    if not math.isfinite(accepted) or not 0 <= accepted <= normalized_nextn:
        raise ValueError(f"nextn_accepted ({nextn_accepted}) must be within [0, nextn={normalized_nextn}].")
    return normalized_nextn, accepted


@dataclass(frozen=True)
class SpeculativeDecodingProfile:
    """Expected accepted-token progress applied above ``aic-core``."""

    expected_accepted_tokens: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.expected_accepted_tokens) or self.expected_accepted_tokens < 0:
            raise ValueError("expected_accepted_tokens must be finite and non-negative.")

    @classmethod
    def from_inputs(
        cls,
        nextn: int | None,
        nextn_accepted: float | None,
    ) -> SpeculativeDecodingProfile:
        """Construct the effective upper-layer profile from public inputs."""
        normalized_nextn, normalized_accepted = normalize_speculative_decoding(nextn, nextn_accepted)
        effective_accepted = float(normalized_accepted or 0.0) if normalized_nextn > 0 else 0.0
        return cls(effective_accepted)

    @classmethod
    def from_scheme(cls, scheme, accepted_tokens: float) -> SpeculativeDecodingProfile:
        """Construct from a speculation-module scheme (cost side) plus an
        acceptance assumption (workload side).

        The bound is scheme-derived: a scheme drafting ``verify_width - 1``
        tokens per round can accept at most that many. The progress fold is
        delegated to the scheme (default ``1 + accepted``), so schemes with
        non-linear folds stay correct without changes here.
        """
        accepted = float(accepted_tokens)
        max_accepted = scheme.verify_width() - 1
        if not math.isfinite(accepted) or not 0 <= accepted <= max_accepted:
            raise ValueError(
                f"accepted_tokens ({accepted_tokens}) must be within [0, verify_width-1="
                f"{max_accepted}] for scheme {scheme.kind!r}."
            )
        return cls(scheme.expected_progress(accepted) - 1.0)

    @property
    def tokens_per_iteration(self) -> float:
        """Expected output-token progress made by one decode iteration."""
        return 1.0 + self.expected_accepted_tokens

    def project_summary(
        self,
        summary: InferenceSummary,
        *,
        role: ProjectionRole,
    ) -> InferenceSummary:
        """Project raw ``aic-core`` iteration metrics into expected service metrics.

        The returned summary is a deep copy. Raw operation/iteration breakdowns
        remain untouched, so callers can still inspect the simulated cost from
        ``aic-core``.
        """
        if role == "prefill" or self.expected_accepted_tokens <= 0:
            return summary

        projected = copy.deepcopy(summary)
        frame = projected.get_summary_df()
        if frame is None or frame.empty:
            return projected

        progress = self.tokens_per_iteration
        if role == "agg":
            step_estimates = projected.get_step_estimates()
            scheduling = step_estimates.get("scheduling", {}) if step_estimates else {}
            applied_progress = scheduling.get("decode_tokens_per_iteration")
            if applied_progress is not None:
                # The agg scheduler already modeled speculative progress; its
                # metrics are authoritative and must never be re-scaled here.
                if not math.isclose(float(applied_progress), progress):
                    logger.warning(
                        "run_agg applied decode_tokens_per_iteration=%s but the projection "
                        "profile expects %s; keeping the scheduler-applied value.",
                        applied_progress,
                        progress,
                    )
                return projected

        frame = frame.copy(deep=True)
        original_request_latency = frame.get("request_latency")

        if "tpot" in frame:
            frame["tpot"] = frame["tpot"] / progress
        if "generation_latency" in frame:
            frame["generation_latency"] = frame["generation_latency"] / progress

        if {"ttft", "tpot", "osl"}.issubset(frame.columns):
            frame["request_latency"] = frame["ttft"] + frame["tpot"] * (frame["osl"] - 1).clip(lower=0)

        if role == "static" and original_request_latency is not None and "request_latency" in frame:
            # Combined static inference includes an unscaled prefill segment, so
            # request throughput follows the old/new end-to-end latency ratio.
            ratio = original_request_latency / frame["request_latency"].replace(0, float("nan"))
            ratio = ratio.fillna(1.0)
        else:
            ratio = progress

        if role == "agg" and {"backend", "concurrency", "request_latency", "seq/s"}.issubset(frame.columns):
            # vLLM caps aggregate output throughput with Little's Law in
            # aic-core. Reapply the equivalent request-rate cap after TPOT is
            # projected because TTFT remains fixed and therefore prevents the
            # end-to-end rate from scaling by ``progress`` in every case.
            vllm_rows = frame["backend"].astype(str).str.lower() == "vllm"
            projected_seq_cap = frame["concurrency"] * 1000.0 / frame["request_latency"].replace(0, float("nan"))
            capped_ratio = projected_seq_cap / frame["seq/s"].replace(0, float("nan"))
            capped_ratio = capped_ratio.clip(lower=0.0, upper=progress).fillna(progress)
            ratio = frame["seq/s"] * 0.0 + progress
            ratio.loc[vllm_rows] = capped_ratio.loc[vllm_rows]

        for column in ("request_rate", "seq/s", "seq/s/gpu", "tokens/s", "tokens/s/gpu"):
            if column in frame:
                frame[column] = frame[column] * ratio
        if "tokens/s/user" in frame:
            frame["tokens/s/user"] = frame["tokens/s/user"] * progress

        frame = frame.round(3)
        projected.set_summary_df(frame)
        projected.set_result_dict(frame.iloc[0].to_dict())
        return projected


@dataclass(frozen=True)
class SpeculativeBlockResolution:
    """Outcome of normalizing a ``speculative:`` block.

    ``method="mtp"`` desugars onto the (nextn, nextn_accepted) pair — the
    legacy code paths stay authoritative and ``speculation_config`` is None.
    Scheme-based methods carry a SpeculationConfig plus the validated
    acceptance value for ``SpeculativeDecodingProfile.from_scheme``.
    """

    nextn: int | str
    nextn_accepted: float | None
    speculation_config: object | None = None
    accepted_tokens: float | None = None


def resolve_speculative_block(
    speculative: dict | None,
    *,
    nextn: int | str = 0,
    nextn_accepted: float | None = None,
) -> SpeculativeBlockResolution:
    """Normalize a ``speculative:`` mapping (single source of truth for the
    task_v2 field and the programmatic API).

    Raises on unknown keys/methods, on conflicts with the legacy nextn pair,
    and on missing or out-of-range ``accepted_tokens`` — bad inputs fail at
    construction time, never mid-sweep.
    """
    if not speculative:
        return SpeculativeBlockResolution(nextn=nextn, nextn_accepted=nextn_accepted)
    if not isinstance(speculative, dict):
        raise TypeError("speculative must be a mapping (method/params/draft_model_path/accepted_tokens).")
    from aiconfigurator.sdk.speculation import SpeculationConfig, build_spec_scheme

    block = dict(speculative)
    method = block.pop("method", None)
    params = dict(block.pop("params", None) or {})
    draft_model_path = block.pop("draft_model_path", None)
    draft_config = block.pop("draft_config", None)
    accepted = block.pop("accepted_tokens", None)
    if block:
        raise ValueError(f"Unknown speculative keys: {sorted(block)}.")
    if not method:
        raise ValueError("speculative.method is required.")
    if method == "none":
        return SpeculativeBlockResolution(nextn=nextn, nextn_accepted=nextn_accepted)
    if method == "mtp":
        depth = int(params.get("depth", 0))
        if nextn not in (0, "auto") and nextn != depth:
            raise ValueError(f"Conflicting speculative inputs: nextn={nextn} vs speculative mtp depth={depth}.")
        return SpeculativeBlockResolution(
            nextn=depth,
            nextn_accepted=float(accepted) if accepted is not None else nextn_accepted,
        )
    if nextn not in (0,):
        raise ValueError(
            f"nextn ({nextn}) is MTP-only sugar and cannot be combined with speculative method {method!r}; set nextn=0."
        )
    if draft_config is None and draft_model_path:
        from aiconfigurator.sdk.utils import get_model_config_from_model_path

        draft_config = dict(get_model_config_from_model_path(draft_model_path).get("raw_config", {}))
    spec_config = SpeculationConfig(
        kind=method, params=params, draft_model_path=draft_model_path, draft_config=draft_config
    )
    scheme = build_spec_scheme(None, spec_config)  # raises on unknown kind / bad params
    if accepted is None:
        raise ValueError(
            f"speculative.accepted_tokens is required for method {method!r} "
            "(measured value; there is no built-in acceptance assumption)."
        )
    accepted = float(accepted)
    max_accepted = scheme.verify_width() - 1
    if not 0 <= accepted <= max_accepted:
        raise ValueError(f"speculative.accepted_tokens ({accepted}) must be within [0, verify_width-1={max_accepted}].")
    return SpeculativeBlockResolution(
        nextn=nextn,
        nextn_accepted=nextn_accepted,
        speculation_config=spec_config,
        accepted_tokens=accepted,
    )


__all__ = [
    "ProjectionRole",
    "SpeculativeBlockResolution",
    "SpeculativeDecodingProfile",
    "normalize_speculative_decoding",
    "resolve_speculative_block",
]
