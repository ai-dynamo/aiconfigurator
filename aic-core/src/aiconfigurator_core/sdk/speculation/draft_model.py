# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone draft-model speculative decoding (classic two-model form).

The draft is a complete, self-contained smaller model (e.g. Qwen3-0.6B
drafting for Qwen3-8B): it consumes NO target hidden states, carries its
own embedding/head/KV, and runs K sequential full autoregressive forwards
per verify round (vLLM ``method="draft_model"``; SGLang STANDALONE).

Unlike the injection-style schemes, the draft here is itself a first-class
AIC model: the scheme instantiates it through the model registry
(``get_model``), so ANY model family AIC supports — dense, MoE, whatever —
is automatically a valid draft, with weights/KV accounting inherited from
the model's own methods instead of scheme-local formulas.

Cost structure per verify round:

* K sequential draft forwards at 1 token/request (the draft model's full
  ``generation_ops`` with per-op counts scaled by K — count is how ops
  encode repetition).
* Target verify width K + 1. No injection/fc GEMM exists.
* Prefill side: one full draft-model context pass builds the draft KV.

Parallelism: the draft's TP defaults to the target's (SGLang deep-copies
the target server args; vLLM is configurable) and can be overridden with
``params["draft_tp_size"]``. Quant modes are inherited from the target
config; small drafts are typically served bf16 — pass explicit overrides
in ``params`` when the deployment differs.

Acceptance stays an upper-layer measured input, as everywhere in this
module. Measured surprise (Qwen3-0.6B -> Qwen3-8B, gsm8k greedy): the
full small model out-accepts a 1-layer eagle3 head at equal K
(E+1 3.40 vs 2.80 at K=3) despite seeing no target hidden states.

GRAPH-DOMAIN CAVEAT (measured 2026-08-09, vLLM nightly caafca24, H100):
this scheme prices the draft on the graph basis like every other scheme,
but that vLLM build runs the draft_model path's draft OUTSIDE CUDA
graphs — draft step wall time was a constant 15.3 ms/step (28-layer
eager launch floor; flat in concurrency, linear in K) vs ~1.4 ms
graph-basis, making the arm 0.4x vs AR end to end. Until the runtime
graphs the draft, treat predictions for this scheme as the
graphed-runtime potential.
"""

from __future__ import annotations

from typing import ClassVar

from aiconfigurator_core.sdk.speculation.base import (
    DraftOpSpec,
    SpecSchemeBase,
    SpeculationConfig,
    register_spec_scheme,
)

_SUPPORTED_BACKENDS = ("vllm",)


@register_spec_scheme("draft_model")
class DraftModelScheme(SpecSchemeBase):
    kind: ClassVar[str] = "draft_model"

    def __init__(
        self,
        draft_model_path: str,
        num_speculative_tokens: int,
        draft_tp_size: int | None = None,
    ) -> None:
        if not draft_model_path:
            raise ValueError("draft_model requires draft_model_path (the standalone draft checkpoint).")
        if num_speculative_tokens < 1:
            raise ValueError(f"num_speculative_tokens must be >= 1, got {num_speculative_tokens}")
        self.draft_model_path = draft_model_path
        self.num_speculative_tokens = int(num_speculative_tokens)
        self.draft_tp_size = int(draft_tp_size) if draft_tp_size else None
        self._draft_model = None
        self._draft_backend: str | None = None

    @classmethod
    def from_configs(cls, model_config, spec_config: SpeculationConfig) -> DraftModelScheme:
        params = spec_config.params
        n = params.get("num_speculative_tokens")
        if n is None:
            raise ValueError(
                "draft_model requires params['num_speculative_tokens'] "
                "(the deployment K; there is no checkpoint default)."
            )
        if isinstance(n, bool) or not isinstance(n, (int, float)) or float(n) != int(n) or int(n) < 1:
            raise ValueError(f"draft_model params['num_speculative_tokens'] must be a positive integer, got {n!r}.")
        return cls(
            draft_model_path=spec_config.draft_model_path or "",
            num_speculative_tokens=int(n),
            draft_tp_size=params.get("draft_tp_size"),
        )

    def verify_attention_sequence_basis(self) -> bool:
        # Target block verify: one shared KV pass per request. Draft steps
        # are width-1, where token and sequence basis coincide.
        return True

    # ------------------------------------------------------------------
    # Draft model construction (through the model registry)
    # ------------------------------------------------------------------
    def _build_draft(self, model, backend_name: str | None = None):
        if self._draft_model is not None:
            return self._draft_model
        import dataclasses

        from aiconfigurator_core.sdk.models import get_model

        backend = backend_name or self._draft_backend or "vllm"
        # Inherit the target's quant/parallel config; standalone drafts are
        # dense small models in every known deployment, so MoE widths are
        # reset alongside an optional TP override.
        draft_config = dataclasses.replace(
            model.config,
            tp_size=self.draft_tp_size or model.config.tp_size,
            pp_size=1,
            moe_tp_size=None,
            moe_ep_size=None,
            speculation=None,
        )
        self._draft_model = get_model(self.draft_model_path, draft_config, backend)
        # Weights before K-scaling: get_weights folds the op count (layer
        # repetition), which must not include the per-round step count.
        self._draft_weights = float(sum(op.get_weights() for op in self._draft_model.generation_ops))
        # K sequential steps per round: fold K into the op counts once
        # (ops encode repetition via their count/scale factor).
        for op in self._draft_model.generation_ops:
            op._scale_factor = op._scale_factor * self.num_speculative_tokens
        return self._draft_model

    def validate(self, model, backend_name: str) -> None:
        if backend_name not in _SUPPORTED_BACKENDS:
            raise ValueError(f"draft_model modeling supports backends {_SUPPORTED_BACKENDS}, got {backend_name!r}.")
        self._draft_backend = backend_name
        # Build eagerly so an unsupported draft checkpoint fails at
        # get_model(target) time, not mid-prediction.
        self._build_draft(model, backend_name)

    def verify_width(self) -> int:
        return self.num_speculative_tokens + 1

    # ------------------------------------------------------------------
    # Draft op graphs — the draft model's own graphs, verbatim
    # ------------------------------------------------------------------
    def build_draft_generation_ops(self, model) -> list[DraftOpSpec]:
        draft = self._build_draft(model)
        return [DraftOpSpec(op=op, tokens_per_request=1) for op in draft.generation_ops]

    def build_draft_context_ops(self, model) -> list[DraftOpSpec]:
        draft = self._build_draft(model)
        return [DraftOpSpec(op=op, tokens_per_request=1) for op in draft.context_ops]

    # ------------------------------------------------------------------
    # Memory accounting — inherited from the draft model itself
    # ------------------------------------------------------------------
    def draft_weights_bytes(self, model) -> float:
        # generation_ops carries the full unique weight set (embedding +
        # layers + head); the sum is cached pre-K-scaling in _build_draft.
        self._build_draft(model)
        return self._draft_weights

    def draft_kv_bytes_per_sequence(self, model, seq_len: int) -> float:
        draft = self._build_draft(model)
        return float(draft.get_kvcache_bytes_per_sequence(seq_len))
