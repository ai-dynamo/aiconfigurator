# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DFlash speculative decoding (non-autoregressive block-diffusion drafting).

Ground facts (draft checkpoint ``z-lab/Qwen3-8B-DFlash-b16`` config +
safetensors size + the vLLM ``method="dflash"`` implementation):

* Draft = ``num_hidden_layers`` full-width dense decoder layers with the
  same block geometry as the DSpark dense draft (the two Qwen3-8B
  community drafts are block-identical: 5 full-attention Qwen3 layers).
  ``dflash_config.target_layer_ids`` feeds the same ``main_proj``
  injection (len * h -> h).
* Drafting is single-shot: ONE parallel forward over the anchor token
  plus ``num_draft_tokens`` mask tokens per round — no sequential Markov
  chain and no per-step feedback (this is the structural difference from
  DSpark; the block compute itself prices identically).
* The z-lab checkpoint carries NO embedding, lm_head, or sampling heads —
  the 2.097 GB safetensors closes as 5 blocks + main_proj to <0.1%. Embed
  and logits use the target's (draft-side logits cost is still incurred
  over the drafted positions).
* ``block_size`` counts the anchor: default drafts = block_size - 1.
  Target verify width = drafts + 1.

Second checkpoint style (speculators; ground facts from
``RedHatAI/DeepSeek-V4-Flash-speculator.dflash`` config + safetensors
header decomposition):

* Geometry nests under ``transformer_layer_config`` and may differ wildly
  from the target (V4 draft: 5 llama-type layers, MQA 64:1 heads,
  head_dim 256, inter 2048, SWA-2048) — still a dense stack, same op
  model.
* mHC targets inject ``hc_mult`` residual streams per aux layer: fc reads
  ``len(aux) * hc_mult * target_h`` (checkpoint fc [4096, 81920] =
  5 x 4 x 4096).
* Carries its OWN full-vocab embedding (129280 x 4096) and a REDUCED
  draft-vocab head (32000, d2t/t2d mapped, eagle3-like). The 3.608 GB
  safetensors closes exactly as blocks + fc + embed + head.
* Official val_metrics.json per-position acceptance (pos1 0.788 ... pos7
  0.193) implies E+1 ~ 2.6 under position-independence — a far weaker
  speculator than V4 DSpark (measured 5.15 committed); yield stays an
  upper-layer input as always.
"""

from __future__ import annotations

from typing import ClassVar

from aiconfigurator_core.sdk.speculation.base import (
    DraftOpSpec,
    SpecSchemeBase,
    SpeculationConfig,
    register_spec_scheme,
)
from aiconfigurator_core.sdk.speculation.dense_draft import (
    DenseDraftGeometry,
    dense_block_ops,
    dense_kv_bytes_per_sequence,
)

# The draft stack is dense in BOTH known checkpoint styles; the target may
# be a dense family (z-lab style) or DeepSeek-V4 (speculators style).
_SUPPORTED_FAMILIES = ("LLAMA", "DEEPSEEKV4")
_SUPPORTED_BACKENDS = ("vllm",)


@register_spec_scheme("dflash")
class DFlashScheme(SpecSchemeBase):
    kind: ClassVar[str] = "dflash"

    def __init__(
        self,
        num_draft_tokens: int,
        target_layer_ids: tuple[int, ...],
        draft_geometry: DenseDraftGeometry,
        injection_streams: int = 1,
        draft_vocab_size: int = 0,
        owns_embed_and_head: bool = False,
    ) -> None:
        if num_draft_tokens < 1:
            raise ValueError(f"num_draft_tokens must be >= 1, got {num_draft_tokens}")
        self.num_draft_tokens = int(num_draft_tokens)
        self.target_layer_ids = tuple(int(i) for i in target_layer_ids)
        self.draft_geometry = draft_geometry
        # mHC targets expose hc_mult residual streams per aux layer: the
        # injection fc reads len(target_layer_ids) * injection_streams * h
        # (RedHatAI V4 checkpoint: fc [4096, 81920] = 5 aux x 4 streams x 4096).
        self.injection_streams = int(injection_streams)
        # 0 = logits via the target's (aliased) head at full vocab.
        self.draft_vocab_size = int(draft_vocab_size)
        # speculators-style checkpoints carry a full-vocab embedding and a
        # reduced-vocab head (d2t/t2d mapped, eagle3-like); z-lab 8B aliases
        # both from the target.
        self.owns_embed_and_head = bool(owns_embed_and_head)

    @classmethod
    def from_configs(cls, model_config, spec_config: SpeculationConfig) -> DFlashScheme:
        draft_config = spec_config.draft_config
        if not draft_config:
            raise ValueError(
                "DFlash requires the draft checkpoint's config.json "
                "(SpeculationConfig.draft_config): it carries block_size, the "
                "stack geometry and the aux/target layer ids."
            )
        params = spec_config.params
        block_size = draft_config.get("block_size")
        layer_cfg = draft_config.get("transformer_layer_config")
        if layer_cfg is not None and draft_config.get("aux_hidden_state_layer_ids") is not None:
            # speculators-style checkpoint (RedHatAI V4 DFlash): nested layer
            # geometry, mHC-aware injection, own embed + reduced-vocab head.
            target_layer_ids = draft_config["aux_hidden_state_layer_ids"]
            geometry = DenseDraftGeometry.from_hf_config(layer_cfg)
            injection_streams = int(layer_cfg.get("hc_mult", 1) or 1)
            draft_vocab_size = int(draft_config.get("draft_vocab_size", 0) or 0)
            owns_embed_and_head = True
        else:
            target_layer_ids = (draft_config.get("dflash_config") or {}).get("target_layer_ids")
            geometry = DenseDraftGeometry.from_hf_config(draft_config) if target_layer_ids is not None else None
            injection_streams = 1
            draft_vocab_size = 0
            owns_embed_and_head = False
        if block_size is None or target_layer_ids is None:
            raise ValueError(
                "draft config lacks 'block_size' plus 'dflash_config.target_layer_ids' "
                "(z-lab style) or 'aux_hidden_state_layer_ids' (speculators style); "
                "not a DFlash draft checkpoint?"
            )
        return cls(
            # block_size counts the anchor position; drafts = block - 1.
            num_draft_tokens=int(params.get("num_draft_tokens", block_size - 1)),
            target_layer_ids=tuple(target_layer_ids),
            draft_geometry=geometry,
            injection_streams=injection_streams,
            draft_vocab_size=draft_vocab_size,
            owns_embed_and_head=owns_embed_and_head,
        )

    def verify_attention_sequence_basis(self) -> bool:
        # Block verify: one shared KV pass per request (see SpecSchemeBase).
        return True

    def validate(self, model, backend_name: str) -> None:
        family = getattr(model, "model_family", None)
        if family not in _SUPPORTED_FAMILIES:
            raise ValueError(
                f"DFlash modeling supports model families {_SUPPORTED_FAMILIES}, got {family!r}. "
                "Add the family's draft geometry before enabling it."
            )
        if backend_name not in _SUPPORTED_BACKENDS:
            raise ValueError(f"DFlash modeling supports backends {_SUPPORTED_BACKENDS}, got {backend_name!r}.")
        if self.draft_geometry.hidden_size != model._hidden_size:
            raise ValueError(
                f"draft hidden_size {self.draft_geometry.hidden_size} != target hidden_size "
                f"{model._hidden_size}; mismatched-width DFlash drafts are unvalidated modeling "
                "territory (the injection GEMM itself supports the mismatch) — validate a "
                "real mismatched checkpoint before relaxing this guard."
            )
        bad_ids = [i for i in self.target_layer_ids if i >= model._num_layers]
        if bad_ids:
            raise ValueError(f"dflash target_layer_ids {bad_ids} out of range for a {model._num_layers}-layer target.")

    def verify_width(self) -> int:
        return self.num_draft_tokens + 1

    # ------------------------------------------------------------------
    # Draft op graphs
    # ------------------------------------------------------------------
    def _main_proj_op(self, model):
        import aiconfigurator_core.sdk.operations as ops

        h = self.draft_geometry.hidden_size
        k = model._hidden_size * max(1, len(self.target_layer_ids)) * self.injection_streams
        return ops.GEMM("dflash_main_proj", 1, h, k, model.config.gemm_quant_mode)

    def _head_op(self, model):
        import aiconfigurator_core.sdk.operations as ops
        from aiconfigurator_core.sdk import common

        vocab = self.draft_vocab_size or model._vocab_size
        return ops.GEMM(
            "dflash_head_gemm",
            1,
            max(1, vocab // model.config.tp_size),
            self.draft_geometry.hidden_size,
            common.GEMMQuantMode.bfloat16,
        )

    def build_draft_generation_ops(self, model) -> list[DraftOpSpec]:
        # One parallel forward: anchor + N mask tokens per request.
        forward_tokens = self.num_draft_tokens + 1
        specs = [DraftOpSpec(op=self._main_proj_op(model), tokens_per_request=1)]
        specs.extend(
            DraftOpSpec(op=op, tokens_per_request=forward_tokens)
            for op in dense_block_ops(self.draft_geometry, model, "dflash", is_context=False)
        )
        # Logits over the N drafted positions (own reduced head or the
        # target's aliased full-vocab head).
        specs.append(DraftOpSpec(op=self._head_op(model), tokens_per_request=self.num_draft_tokens))
        return specs

    def build_draft_context_ops(self, model) -> list[DraftOpSpec]:
        # Context-KV precompute: the draft blocks ingest the prompt once,
        # plus the injection GEMM over the prompt's aux hiddens.
        specs = [DraftOpSpec(op=self._main_proj_op(model), tokens_per_request=1)]
        specs.extend(
            DraftOpSpec(op=op, tokens_per_request=1)
            for op in dense_block_ops(self.draft_geometry, model, "dflash", is_context=True)
        )
        return specs

    # ------------------------------------------------------------------
    # Memory accounting
    # ------------------------------------------------------------------
    def draft_weights_bytes(self, model) -> float:
        import aiconfigurator_core.sdk.operations as ops
        from aiconfigurator_core.sdk import common

        weight_ops = list(dense_block_ops(self.draft_geometry, model, "dflash", is_context=False))
        weight_ops.append(self._main_proj_op(model))
        if self.owns_embed_and_head:
            # speculators-style checkpoint: full-vocab embedding + reduced
            # draft-vocab head resident (RedHatAI V4: 1.06 GB + 0.26 GB).
            weight_ops.append(self._head_op(model))
            weight_ops.append(
                ops.GEMM(
                    "dflash_embed_weights",
                    1,
                    model._vocab_size // model.config.tp_size,
                    self.draft_geometry.hidden_size,
                    common.GEMMQuantMode.bfloat16,
                )
            )
        # z-lab style aliases embed/head from the target: nothing extra.
        return float(sum(op.get_weights() for op in weight_ops))

    def draft_kv_bytes_per_sequence(self, model, seq_len: int) -> float:
        return dense_kv_bytes_per_sequence(self.draft_geometry, model, seq_len)
