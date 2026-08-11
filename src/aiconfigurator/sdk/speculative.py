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


__all__ = [
    "ProjectionRole",
    "SpeculativeDecodingProfile",
    "normalize_speculative_decoding",
]
