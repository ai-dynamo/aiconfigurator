# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""EAGLE-3 speculative decoding (sequential single-layer drafting, chain or
tree).

Ground facts (draft checkpoint ``Tengyunw/qwen3_8b_eagle3`` config +
checkpoint size + the vLLM ``method="eagle3"`` implementation):

* Draft = ONE decoder layer whose attention input is the concatenation of
  the token embedding and the previous hidden state (QKV input width 2h —
  the EAGLE structural signature), plus an ``fc`` projection that fuses
  ``num_aux_hiddens`` (= 3) target-layer hidden states into the draft
  input once per round, plus an lm_head over a REDUCED draft vocabulary
  (``draft_vocab_size``, mapped back to target ids via d2t). The 799 MB
  checkpoint closes as layer(2h QKV) + fc(3h->h) + head(draft_vocab) to
  <0.2%. Embedding is aliased from the target.
* Chain mode: ``num_speculative_tokens`` sequential draft forwards per
  round, each 1 token per request, feeding back the draft's own hidden.
* Tree mode: ``tree_shape`` = tokens drafted per level (chain is
  ``[1] * K``). Level i is one forward of ``tree_shape[i]`` tokens per
  request; the target then verifies the whole tree in one forward of
  ``sum(tree_shape) + 1`` tokens. Attention inside a tree level is
  approximated as dense queries against the full sequence (path masking
  is second-order for cost).
* Acceptance stays an upper-layer input for BOTH modes — a tree topology
  changes the expected-progress value the caller supplies, not the
  scheme's cost interface.
"""

from __future__ import annotations

from typing import ClassVar

from aiconfigurator_core.sdk.speculation.base import (
    DraftOpSpec,
    SpecSchemeBase,
    SpeculationConfig,
    register_spec_scheme,
)
from aiconfigurator_core.sdk.speculation.dense_draft import DenseDraftGeometry, dense_kv_bytes_per_sequence

_SUPPORTED_FAMILIES = ("LLAMA",)
_SUPPORTED_BACKENDS = ("vllm", "sglang")
# EAGLE-3 fuses low/mid/high target hidden states into the draft input.
_NUM_AUX_HIDDENS = 3


@register_spec_scheme("eagle3")
class EagleScheme(SpecSchemeBase):
    kind: ClassVar[str] = "eagle3"

    def __init__(
        self,
        tree_shape: tuple[int, ...],
        draft_geometry: DenseDraftGeometry,
        draft_vocab_size: int,
        verify_token_budget: int | None = None,
    ) -> None:
        if not tree_shape or any(w < 1 for w in tree_shape):
            raise ValueError(f"tree_shape must be a non-empty tuple of widths >= 1, got {tree_shape!r}")
        self.tree_shape = tuple(int(w) for w in tree_shape)
        self.draft_geometry = draft_geometry
        self.draft_vocab_size = int(draft_vocab_size)
        # Draft forward width and verify budget are independent quantities
        # in tree runtimes: each forwarded node contributes up to top-k
        # candidate tokens via its logits (free on the draft forward), and
        # the runtime selects `num_draft_tokens` tree nodes (root included)
        # for the verify forward — the selection pool is LARGER than the
        # forwarded width (SGLang: topk logit branches per node). So the
        # budget may legitimately exceed sum(tree_shape) + 1.
        # None = verify everything forwarded + root (no pruning, no logit
        # branching).
        if verify_token_budget is not None:
            if int(verify_token_budget) <= 1:
                raise ValueError(f"verify_token_budget must be > 1, got {verify_token_budget}")
            self.verify_token_budget = int(verify_token_budget)
        else:
            self.verify_token_budget = sum(self.tree_shape) + 1

    @property
    def num_draft_tokens(self) -> int:
        return sum(self.tree_shape)

    @property
    def is_tree(self) -> bool:
        return any(w > 1 for w in self.tree_shape)

    @classmethod
    def from_configs(cls, model_config, spec_config: SpeculationConfig) -> EagleScheme:
        draft_config = spec_config.draft_config
        if not draft_config:
            raise ValueError(
                "EAGLE-3 requires the draft checkpoint's config.json "
                "(SpeculationConfig.draft_config): it carries the draft layer "
                "geometry and draft_vocab_size."
            )
        params = spec_config.params
        tree_shape = params.get("tree_shape")
        if tree_shape is None:
            depth = int(params.get("num_speculative_tokens", 3))
            tree_shape = (1,) * depth
        return cls(
            tree_shape=tuple(tree_shape),
            draft_geometry=DenseDraftGeometry.from_hf_config(draft_config),
            draft_vocab_size=int(draft_config.get("draft_vocab_size", draft_config.get("vocab_size", 0))),
            # SGLang's --speculative-num-draft-tokens (verify tokens incl. root).
            verify_token_budget=params.get("verify_token_budget"),
        )

    def verify_attention_sequence_basis(self) -> bool:
        # Block verify: one shared KV pass per request (see SpecSchemeBase).
        return True

    def validate(self, model, backend_name: str) -> None:
        family = getattr(model, "model_family", None)
        if family not in _SUPPORTED_FAMILIES:
            raise ValueError(
                f"EAGLE-3 modeling supports model families {_SUPPORTED_FAMILIES}, got {family!r}. "
                "Add the family's draft geometry before enabling it."
            )
        if backend_name not in _SUPPORTED_BACKENDS:
            raise ValueError(f"EAGLE-3 modeling supports backends {_SUPPORTED_BACKENDS}, got {backend_name!r}.")
        if self.draft_geometry.hidden_size != model._hidden_size:
            raise ValueError(
                f"draft hidden_size {self.draft_geometry.hidden_size} != target hidden_size "
                f"{model._hidden_size}; the fc fusion requires matching widths."
            )

    def verify_width(self) -> int:
        return self.verify_token_budget

    # ------------------------------------------------------------------
    # Draft op graphs
    # ------------------------------------------------------------------
    def _fc_op(self, model):
        import aiconfigurator_core.sdk.operations as ops

        h = self.draft_geometry.hidden_size
        return ops.GEMM("eagle3_fc", 1, h, h * _NUM_AUX_HIDDENS, model.config.gemm_quant_mode)

    def _layer_ops(self, model, *, is_context: bool) -> list:
        """The single EAGLE-3 decoder layer (QKV input width 2h) + reduced
        lm_head. Instantiated once per tree level (count = num_layers = 1)."""
        import aiconfigurator_core.sdk.operations as ops
        from aiconfigurator_core.sdk import common

        cfg = model.config
        geom = self.draft_geometry
        tp_size = cfg.tp_size
        h = geom.hidden_size
        n = float(geom.num_layers)
        kv_per_gpu = max(1, geom.num_kv_heads // tp_size)
        attn_args = dict(head_size=geom.head_dim, use_qk_norm=geom.use_qk_norm)
        attn = (
            ops.ContextAttention(
                "eagle3_attention",
                n,
                geom.num_heads // tp_size,
                kv_per_gpu,
                cfg.kvcache_quant_mode,
                cfg.fmha_quant_mode,
                cp_size=1,
                **attn_args,
            )
            if is_context
            else ops.GenerationAttention(
                "eagle3_attention",
                n,
                geom.num_heads // tp_size,
                kv_per_gpu,
                cfg.kvcache_quant_mode,
                **attn_args,
            )
        )
        return [
            ops.ElementWise("eagle3_add_norm_1", n, 2 * h, 2 * h, 0.8),
            ops.GEMM(
                "eagle3_qkv_gemm",
                n,
                geom.num_heads * geom.head_dim // tp_size + geom.head_dim * kv_per_gpu * 2,
                2 * h,  # cat(embedding, hidden) input — the EAGLE signature
                cfg.gemm_quant_mode,
            ),
            attn,
            ops.GEMM(
                "eagle3_proj_gemm",
                n,
                h,
                geom.num_heads * geom.head_dim // tp_size,
                cfg.gemm_quant_mode,
                low_precision_input=True,
            ),
            ops.ElementWise("eagle3_add_norm_2", n, 2 * h, 2 * h, 0.8),
            ops.GEMM("eagle3_gate_ffn1_gemm", n, 2 * geom.inter_size // tp_size, h, cfg.gemm_quant_mode),
            ops.ElementWise("eagle3_act_gate", n, 2 * geom.inter_size // tp_size, geom.inter_size // tp_size, 0.8),
            ops.GEMM(
                "eagle3_ffn2_gemm",
                n,
                h,
                geom.inter_size // tp_size,
                cfg.gemm_quant_mode,
                low_precision_input=True,
            ),
            ops.CustomAllReduce("eagle3_ar_1", n, h, tp_size),
            ops.CustomAllReduce("eagle3_ar_2", n, h, tp_size),
            ops.GEMM(
                "eagle3_head_gemm",
                1,
                max(1, self.draft_vocab_size // tp_size),
                h,
                common.GEMMQuantMode.bfloat16,
            ),
        ]

    def build_draft_generation_ops(self, model) -> list[DraftOpSpec]:
        # fc fusion of the target's aux hiddens: once per round.
        specs = [DraftOpSpec(op=self._fc_op(model), tokens_per_request=1)]
        # One sequential draft forward per tree level; level width sets the
        # per-request token count (chain: width 1 at every level). Same op
        # names across levels accumulate into one latency entry.
        for width in self.tree_shape:
            specs.extend(
                DraftOpSpec(op=op, tokens_per_request=width) for op in self._layer_ops(model, is_context=False)
            )
        return specs

    def build_draft_context_ops(self, model) -> list[DraftOpSpec]:
        # Prefill-side draft KV build over the prompt (one layer) + fc.
        specs = [DraftOpSpec(op=self._fc_op(model), tokens_per_request=1)]
        specs.extend(DraftOpSpec(op=op, tokens_per_request=1) for op in self._layer_ops(model, is_context=True))
        return specs

    # ------------------------------------------------------------------
    # Memory accounting
    # ------------------------------------------------------------------
    def draft_weights_bytes(self, model) -> float:
        # One level's op list carries each weight exactly once (layer + head);
        # fc is separate. Embedding is aliased from the target (not counted).
        weight_ops = [self._fc_op(model), *self._layer_ops(model, is_context=False)]
        return float(sum(op.get_weights() for op in weight_ops))

    def draft_kv_bytes_per_sequence(self, model, seq_len: int) -> float:
        return dense_kv_bytes_per_sequence(self.draft_geometry, model, seq_len)
