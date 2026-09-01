# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""N-gram / prompt-lookup speculative decoding (model-free drafting).

The draft comes from matching the recent output suffix against the
prompt/output history (vLLM ``method="ngram"``): no draft network, no
draft weights, no draft KV. On the GPU cost side the scheme is purely a
verify-width change — the target verifies ``num_speculative_tokens + 1``
tokens per round. The lookup itself runs on the host and is not modeled
(host-side glue is documented as out of scope in
``speculation/materialize.py``).

Acceptance is strongly workload-dependent (repetition-heavy text accepts
well); as everywhere else in this module it stays an upper-layer input.
"""

from __future__ import annotations

from typing import ClassVar

from aiconfigurator_core.sdk.speculation.base import (
    DraftOpSpec,
    SpecSchemeBase,
    SpeculationConfig,
    register_spec_scheme,
)


@register_spec_scheme("ngram")
class NgramScheme(SpecSchemeBase):
    kind: ClassVar[str] = "ngram"

    def __init__(
        self,
        num_speculative_tokens: int,
        trigger_rate: float = 1.0,
    ) -> None:
        if num_speculative_tokens < 1:
            raise ValueError(f"num_speculative_tokens must be >= 1, got {num_speculative_tokens}")
        self.num_speculative_tokens = int(num_speculative_tokens)
        # Fraction of decode rounds in which the proposer actually drafts
        # (prompt-lookup fires only on a suffix match). Measured, workload-
        # dependent — e.g. Qwen3-8B gsm8k greedy: p = 0.301, flat across
        # concurrency. The acceptance input `accepted_tokens` is the
        # PER-DRAFTED-ROUND value (vLLM accepted/drafts counters); the
        # expected-progress fold mixes in the (1 - p) draft-less rounds.
        # Default 1.0 = every round drafts (dense-drafting semantics).
        if not 0.0 < float(trigger_rate) <= 1.0:
            raise ValueError(f"trigger_rate must be in (0, 1], got {trigger_rate}")
        self.trigger_rate = float(trigger_rate)

    @classmethod
    def from_configs(cls, model_config, spec_config: SpeculationConfig) -> NgramScheme:
        params = spec_config.params
        n = params.get("num_speculative_tokens")
        if n is None:
            raise ValueError("ngram requires params['num_speculative_tokens'] (no draft config to default from).")
        return cls(
            num_speculative_tokens=int(n),
            trigger_rate=float(params.get("trigger_rate", 1.0)),
        )

    def expected_progress(self, accepted_tokens: float) -> float:
        # Mixed rounds: p drafted rounds commit (1 + accepted), (1 - p)
        # draft-less rounds commit 1 -> 1 + p * accepted per round.
        # Cost-side note: verify compute is still priced at full width every
        # round (the op graph cannot express mixed widths yet), so with
        # p < 1 the GEMM token volume is over-priced by (1-p)*(width-1)
        # tokens/round — a conservative bias, flagged for the mixed-round
        # costing follow-up.
        return 1.0 + self.trigger_rate * float(accepted_tokens)

    def verify_attention_sequence_basis(self) -> bool:
        # Block verify: one shared KV pass per request (see SpecSchemeBase).
        return True

    def validate(self, model, backend_name: str) -> None:
        # Model-free drafting has no family constraint; any verify-capable
        # backend qualifies.
        return None

    def verify_width(self) -> int:
        return self.num_speculative_tokens + 1

    def build_draft_generation_ops(self, model) -> list[DraftOpSpec]:
        return []

    def build_draft_context_ops(self, model) -> list[DraftOpSpec]:
        return []

    def draft_weights_bytes(self, model) -> float:
        return 0.0

    def draft_kv_bytes_per_sequence(self, model, seq_len: int) -> float:
        return 0.0
