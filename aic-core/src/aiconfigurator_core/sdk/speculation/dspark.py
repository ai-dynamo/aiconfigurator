# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DSpark speculative decoding (semi-autoregressive block drafting).

First target: DeepSeek-V4. Ground facts (draft checkpoint
``deepseek-ai/DeepSeek-V4-Flash-DSpark`` safetensors index + the vLLM
implementation ``models/deepseek_v4/nvidia/dspark.py`` /
``v1/worker/gpu/spec_decode/dspark/speculator.py``):

* Draft = ``mtp.{0..L-1}`` blocks in the checkpoint — full-width V4 decoder
  layers (MLA attention + mHC + shared/routed MoE). ``num_dspark_layers``
  defaults to 3. Embed/head are aliased from the target (no extra weights).
* Draft attention is window-capped (the target config's ``compress_ratios``
  carries ratio-0 = pure-SWA entries for the draft blocks;
  ``sliding_window`` = 128). Latency is approximated with the HCA
  (ratio 128) module, mirroring DeepSeekV4Model's own treatment of ratio-0
  layers; KV capacity uses the true window cap.
* Per verify round: ONE parallel backbone forward over
  ``N = num_draft_tokens`` query tokens per request (anchor + N-1 noise),
  a ``main_proj`` injection GEMM (concat of the target-layer aux hiddens,
  ``3h -> h``) once per request, the draft-logits head GEMM over the N
  positions, then N sequential Markov-bias steps (rank ``markov_rank``).
* Target verify width = ``N + 1`` (N drafts + bonus).

Dense targets (second family, ground facts from
``deepseek-ai/dspark_qwen3_8b_block7`` config + safetensors size):

* Draft = ``num_hidden_layers`` full-width dense decoder layers with the
  TARGET's geometry class (Qwen3 GQA + gated MLP) but its OWN config —
  full attention (no sliding window), own KV cache over the full sequence.
* ``target_layer_ids`` lists the target layers whose hidden states feed
  ``main_proj`` (len * h -> h); the 8B draft taps 5 evenly spaced layers
  [1, 9, 17, 25, 33] where V4 taps the trailing 3.
* Unlike the V4 draft (embed/head aliased from the target), the dense
  checkpoint carries its OWN embedding and lm_head — the 4.742 GB
  safetensors closes as 5 blocks + main_proj + markov + embed + head
  to within 0.1%, so both are counted as resident draft weights.
* Same round structure: one parallel N-token forward, main_proj, head
  over N positions, N sequential Markov-bias steps, verify width N + 1.

Draft compute reuses existing op classes and perf tables only — no new
collector data is required for this scheme.
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

# V4-family drafts derive geometry from the target model; dense-family
# drafts carry their own geometry (draft_geometry is not None).
_V4_FAMILIES = ("DEEPSEEKV4",)
_DENSE_FAMILIES = ("LLAMA",)
_SUPPORTED_BACKENDS = ("vllm", "sglang")
# Ratio-0 (pure SWA) draft layers are approximated with the HCA module,
# exactly as DeepSeekV4Model does for the target's own ratio-0 layers.
_SWA_LATENCY_PROXY_RATIO = 128


@register_spec_scheme("dspark")
class DSparkScheme(SpecSchemeBase):
    kind: ClassVar[str] = "dspark"

    def __init__(
        self,
        num_draft_tokens: int,
        num_draft_layers: int,
        target_layer_ids: tuple[int, ...],
        markov_rank: int,
        draft_geometry: DenseDraftGeometry | None = None,
    ) -> None:
        if num_draft_tokens < 1:
            raise ValueError(f"num_draft_tokens must be >= 1, got {num_draft_tokens}")
        if num_draft_layers < 1:
            raise ValueError(f"num_draft_layers must be >= 1, got {num_draft_layers}")
        self.num_draft_tokens = int(num_draft_tokens)
        self.num_draft_layers = int(num_draft_layers)
        self.target_layer_ids = tuple(int(i) for i in target_layer_ids)
        self.markov_rank = int(markov_rank)
        # None = V4-family draft (geometry derived from the target model);
        # set = dense-family draft with its own stack geometry.
        self.draft_geometry = draft_geometry

    @classmethod
    def from_configs(cls, model_config, spec_config: SpeculationConfig) -> DSparkScheme:
        draft_config = spec_config.draft_config
        if not draft_config:
            raise ValueError(
                "DSpark requires the draft checkpoint's config.json "
                "(SpeculationConfig.draft_config): it carries the block size, "
                "target_layer_ids and markov_rank. Point draft_model_path at "
                "the draft repo or inject the parsed config."
            )
        params = spec_config.params
        if draft_config.get("dspark_block_size") is not None:
            # V4-style draft config (fields prefixed, geometry from target).
            block_size = draft_config["dspark_block_size"]
            geometry = None
            num_draft_layers = int(params.get("num_draft_layers", draft_config.get("n_mtp_layers") or 3))
            if not draft_config.get("dspark_target_layer_ids"):
                # Required: these are the target hidden-state taps feeding
                # main_proj — an empty default would silently price the
                # injection GEMM at width h instead of len(ids)*h.
                raise ValueError(
                    "V4-style DSpark draft config lacks 'dspark_target_layer_ids' "
                    "(the target hidden-state taps that size the main_proj injection GEMM)."
                )
            target_layer_ids = tuple(draft_config["dspark_target_layer_ids"])
            markov_rank = int(draft_config.get("dspark_markov_rank", 256))
        elif draft_config.get("block_size") is not None and draft_config.get("target_layer_ids") is not None:
            # Standalone dense draft checkpoint (e.g. Qwen3DSparkModel):
            # carries its own full stack geometry.
            block_size = draft_config["block_size"]
            geometry = DenseDraftGeometry.from_hf_config(draft_config)
            num_draft_layers = int(params.get("num_draft_layers", geometry.num_layers))
            target_layer_ids = tuple(draft_config["target_layer_ids"])
            markov_rank = int(draft_config.get("markov_rank", 256))
        else:
            raise ValueError(
                "draft config lacks 'dspark_block_size' (V4 style) or "
                "'block_size'+'target_layer_ids' (dense style); not a DSpark draft checkpoint?"
            )
        return cls(
            num_draft_tokens=int(params.get("num_draft_tokens", block_size)),
            num_draft_layers=num_draft_layers,
            target_layer_ids=target_layer_ids,
            markov_rank=markov_rank,
            draft_geometry=geometry,
        )

    def verify_attention_sequence_basis(self) -> bool:
        # Block verify: one shared KV pass per request (see SpecSchemeBase).
        return True

    # ------------------------------------------------------------------
    # Capability gate
    # ------------------------------------------------------------------
    def validate(self, model, backend_name: str) -> None:
        family = getattr(model, "model_family", None)
        supported = _DENSE_FAMILIES if self.draft_geometry is not None else _V4_FAMILIES
        if family not in supported:
            raise ValueError(
                f"DSpark modeling supports model families {supported} for this draft style, got {family!r}. "
                "Add the family's draft geometry to DSparkScheme before enabling it."
            )
        if backend_name not in _SUPPORTED_BACKENDS:
            raise ValueError(f"DSpark modeling supports backends {_SUPPORTED_BACKENDS}, got {backend_name!r}.")
        if self.draft_geometry is not None and self.draft_geometry.hidden_size != model._hidden_size:
            raise ValueError(
                f"draft hidden_size {self.draft_geometry.hidden_size} != target hidden_size "
                f"{model._hidden_size}; the main_proj injection requires matching widths."
            )
        bad_ids = [i for i in self.target_layer_ids if i >= model._num_layers]
        if bad_ids:
            raise ValueError(f"dspark target_layer_ids {bad_ids} out of range for a {model._num_layers}-layer target.")

    def verify_width(self) -> int:
        return self.num_draft_tokens + 1

    # ------------------------------------------------------------------
    # Draft op graphs (reuse the exact op classes/shapes of DeepSeekV4Model)
    # ------------------------------------------------------------------
    def _sliding_window(self, model) -> int:
        return int(getattr(model.extra_params, "sliding_window", 128) or 128)

    def _block_ops(self, model, *, is_context: bool) -> list:
        """One draft decoder-block stack (count = num_draft_layers)."""
        if self.draft_geometry is not None:
            import dataclasses

            geom = dataclasses.replace(self.draft_geometry, num_layers=self.num_draft_layers)
            return dense_block_ops(geom, model, "dspark", is_context=is_context)
        return self._v4_block_ops(model, is_context=is_context)

    def _v4_block_ops(self, model, *, is_context: bool) -> list:
        import aiconfigurator_core.sdk.operations as ops
        from aiconfigurator_core.sdk import common

        cfg = model.config
        v4 = model.extra_params
        h = model._hidden_size
        tp_size = cfg.tp_size
        n = float(self.num_draft_layers)
        local_heads = model._num_heads // tp_size
        local_o_groups = max(1, v4.o_groups // tp_size)
        local_moe_inter = model._moe_inter_size // tp_size
        workload_distribution = (
            cfg.workload_distribution + f"_{model._power_law_alpha}"
            if cfg.workload_distribution == "power_law"
            else cfg.workload_distribution
        )
        attn_cls = ops.ContextDeepSeekV4AttentionModule if is_context else ops.GenerationDeepSeekV4AttentionModule
        phase = "context" if is_context else "generation"

        return [
            attn_cls(
                "dspark_attention",
                n,
                local_heads,
                model._num_heads,
                tp_size,
                h,
                v4.q_lora_rank,
                v4.o_lora_rank,
                v4.head_dim,
                v4.qk_rope_head_dim,
                v4.index_n_heads,
                v4.index_head_dim,
                v4.index_topk,
                v4.sliding_window,
                _SWA_LATENCY_PROXY_RATIO,
                local_o_groups,
                cfg.kvcache_quant_mode,
                cfg.fmha_quant_mode,
                cfg.gemm_quant_mode,
                architecture=model.architecture,
                cp_size=1,
            ),
            ops.DeepSeekV4MHCModule(
                "dspark_mhc_pre",
                n,
                "pre",
                h,
                v4.hc_mult,
                v4.hc_sinkhorn_iters,
                common.GEMMQuantMode.bfloat16,
                architecture=model.architecture,
            ),
            ops.DeepSeekV4MHCModule(
                "dspark_mhc_post",
                n,
                "post",
                h,
                v4.hc_mult,
                v4.hc_sinkhorn_iters,
                common.GEMMQuantMode.bfloat16,
                architecture=model.architecture,
            ),
            ops.ElementWise("dspark_attn_norm", n, h, h, 0.8),
            ops.ElementWise("dspark_ffn_norm", n, h, h, 0.8),
            ops.GEMM("dspark_shared_gate_up_gemm", n, 2 * local_moe_inter, h, cfg.gemm_quant_mode),
            ops.ElementWise("dspark_shared_act_gate", n, 2 * local_moe_inter, local_moe_inter, 0.8),
            ops.GEMM("dspark_shared_ffn2_gemm", n, h, local_moe_inter, cfg.gemm_quant_mode),
            ops.GEMM("dspark_router_gemm", n, model._num_experts, h, common.GEMMQuantMode.bfloat16),
            ops.MoEDispatch(
                f"dspark_{phase}_moe_pre_dispatch",
                n,
                h,
                model._topk,
                model._num_experts,
                cfg.moe_tp_size,
                cfg.moe_ep_size,
                cfg.attention_dp_size,
                True,
                quant_mode=cfg.moe_quant_mode,
                backend=model._backend_name,
            ),
            ops.MoE(
                f"dspark_{phase}_moe",
                n,
                h,
                model._moe_inter_size,
                model._topk,
                model._num_experts,
                cfg.moe_tp_size,
                cfg.moe_ep_size,
                cfg.moe_quant_mode,
                workload_distribution,
                cfg.attention_dp_size,
            ),
            ops.MoEDispatch(
                f"dspark_{phase}_moe_post_dispatch",
                n,
                h,
                model._topk,
                model._num_experts,
                cfg.moe_tp_size,
                cfg.moe_ep_size,
                cfg.attention_dp_size,
                False,
                quant_mode=cfg.moe_quant_mode,
                backend=model._backend_name,
            ),
        ]

    def _main_proj_op(self, model):
        import aiconfigurator_core.sdk.operations as ops

        h = model._hidden_size
        # ReplicatedLinear(h * len(target_layer_ids) -> h) in vLLM.
        return ops.GEMM("dspark_main_proj", 1, h, h * max(1, len(self.target_layer_ids)), model.config.gemm_quant_mode)

    def build_draft_generation_ops(self, model) -> list[DraftOpSpec]:
        import aiconfigurator_core.sdk.operations as ops
        from aiconfigurator_core.sdk import common

        n_tokens = self.num_draft_tokens
        # Window-capped drafts (V4) pin the attention KV length; full-attention
        # dense drafts follow the target's sequence length (no override).
        if self.draft_geometry is not None:
            window = self.draft_geometry.sliding_window
        else:
            window = self._sliding_window(model)
        kv_cap = window + n_tokens if window else None
        specs = [DraftOpSpec(op=self._main_proj_op(model), tokens_per_request=1)]
        for op in self._block_ops(model, is_context=False):
            overrides = {"s": kv_cap} if kv_cap and op._name == "dspark_attention" else None
            specs.append(DraftOpSpec(op=op, tokens_per_request=n_tokens, query_overrides=overrides))
        # Draft-logits head over the N positions (aliases the target head).
        specs.append(
            DraftOpSpec(
                op=ops.GEMM(
                    "dspark_head_gemm",
                    1,
                    model._vocab_size // model.config.tp_size,
                    model._hidden_size,
                    common.GEMMQuantMode.bfloat16,
                ),
                tokens_per_request=n_tokens,
            )
        )
        # Sequential Markov sampling: N small bias GEMMs (rank -> vocab).
        specs.append(
            DraftOpSpec(
                op=ops.GEMM(
                    "dspark_markov_bias_gemm",
                    n_tokens,
                    model._vocab_size,
                    self.markov_rank,
                    common.GEMMQuantMode.bfloat16,
                ),
                tokens_per_request=1,
            )
        )
        return specs

    def build_draft_context_ops(self, model) -> list[DraftOpSpec]:
        # DFlash-style context-KV precompute: the draft blocks ingest the
        # prompt once (window-capped attention), plus the injection GEMM
        # over the prompt's aux hiddens.
        specs = [DraftOpSpec(op=self._main_proj_op(model), tokens_per_request=1)]
        specs.extend(DraftOpSpec(op=op, tokens_per_request=1) for op in self._block_ops(model, is_context=True))
        return specs

    # ------------------------------------------------------------------
    # Memory accounting
    # ------------------------------------------------------------------
    def draft_weights_bytes(self, model) -> float:
        import aiconfigurator_core.sdk.operations as ops
        from aiconfigurator_core.sdk import common

        weight_ops = list(self._block_ops(model, is_context=False))
        weight_ops.append(self._main_proj_op(model))
        # Markov head: embed (markov_w1) + bias (markov_w2), each vocab x rank,
        # replicated.
        weight_ops.append(
            ops.GEMM("dspark_markov_weights", 2, model._vocab_size, self.markov_rank, common.GEMMQuantMode.bfloat16)
        )
        if self.draft_geometry is not None:
            # Dense draft checkpoints carry their OWN embedding + lm_head
            # (the 8B safetensors closes only with both counted); V4 drafts
            # alias the target's and contribute nothing here.
            geom = self.draft_geometry
            weight_ops.append(
                ops.GEMM(
                    "dspark_embed_head_weights",
                    2,
                    # Speculators-style partial configs may omit vocab_size
                    # (geometry carries 0): fall back to the target's vocab
                    # rather than zeroing the embed+head bytes.
                    (geom.vocab_size or model._vocab_size) // model.config.tp_size,
                    geom.hidden_size,
                    common.GEMMQuantMode.bfloat16,
                )
            )
        return float(sum(op.get_weights() for op in weight_ops))

    def draft_kv_bytes_per_sequence(self, model, seq_len: int) -> float:
        if self.draft_geometry is not None:
            return dense_kv_bytes_per_sequence(self.draft_geometry, model, seq_len)
        # Mirrors DeepSeekV4Model.get_kvcache_bytes_per_sequence's ratio-0
        # (pure SWA) branch: window-capped entries, no compressed stream.
        window = self._sliding_window(model)
        entry_bytes = model.extra_params.head_dim * model.config.kvcache_quant_mode.value.memory
        return float(self.num_draft_layers * min(max(seq_len, 0), window) * entry_bytes)
