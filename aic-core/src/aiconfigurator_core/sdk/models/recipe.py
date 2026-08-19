# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RecipeModel — a model whose op graph is built from a machine-extracted recipe.

A recipe (``aic-model-recipe/v0`` YAML, shipped under
``aiconfigurator_core/recipes/``) is extracted from REAL framework execution
traces: per (layer_kind, phase) it records the ordered op sequence with module
identity, quant-method classes, tensor shapes, kernels, and framework call
paths, plus derived branch guards and tp-validated sharding rules. RecipeModel
maps those traced facts onto the existing perf-database queries and fills
``context_ops`` / ``generation_ops`` like any hand-written model — the
evaluation/scheduling layer is untouched.

Design rules (from the GLM-5.2 pilot, ``docs/recipe_model.md``):

* **Facts vs policy are separate.** The recipe always records what the
  framework executed. Where reality diverges from the collected data grid, an
  explicit *mapping policy* decides how to query — every application is
  appended to ``mapping_notes`` so a shadow diff can attribute it. Nothing is
  silently dropped or hand-tuned per model.
* **Tolerated divergences are rules, not edits.** Example: sglang fuses the
  shared expert into the routed MoE (traced 257 experts / topk 9 for GLM-5.2;
  router logits stay 256-wide). Re-collecting fused shapes would ripple through
  the whole collection pipeline for a bounded, small effect (same
  expert-invocations per token, same weight bytes), so the default
  ``decompose_fused_shared`` policy maps the traced shape back onto the
  collected decomposition. The rule triggers ONLY on the traced discrepancy
  (traced experts > router width) — no per-model knowledge.
* **Scaffold ops the trace cannot see** (residual+norm elementwise, embedding,
  logits GEMM, P2P) reuse the same generic formulas as hand models.
* **Fail loud on coverage gaps.** A layer kind present in the checkpoint but
  never traced, an unmapped quant class, or an unreadable traced shape raises
  ``RecipeGapError`` — that is a finding, not something to paper over.

Pilot scope: DSA-attention MoE models (GLM-5.2 family). CP is out of scope.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import yaml

import aiconfigurator_core.sdk.operations as ops
from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk.models.base import BaseModel
from aiconfigurator_core.sdk.models.helpers import (
    _apply_model_quant_defaults,
    _architecture_to_model_family,
    _get_model_info,
    mtp_scale_factor,
)

logger = logging.getLogger(__name__)

RECIPES_DIR = Path(__file__).resolve().parents[2] / "recipes"


class RecipeGapError(RuntimeError):
    """The recipe cannot express / does not cover something the model needs."""


# Traced quant-method class -> perf-database quant mode. Extend as recipes for
# new quant families land; an unmapped class raises RecipeGapError (fail loud).
GEMM_QUANT_BY_CLASS = {
    "Fp8LinearMethod": common.GEMMQuantMode.fp8_block,  # GLM ckpts are 128x128 block-quant
    "UnquantizedLinearMethod": common.GEMMQuantMode.bfloat16,
}
MOE_QUANT_BY_CLASS = {
    "Fp8MoEMethod": common.MoEQuantMode.fp8_block,
    "UnquantizedFusedMoEMethod": common.MoEQuantMode.bfloat16,
}
KV_BY_DTYPE = {
    "fp8_e4m3": common.KVCacheQuantMode.fp8,
    "fp8_e5m2": common.KVCacheQuantMode.fp8,
    "bfloat16": common.KVCacheQuantMode.bfloat16,
    "auto": common.KVCacheQuantMode.bfloat16,
    None: common.KVCacheQuantMode.bfloat16,
}

_SHAPE_RE = re.compile(r"^(\w+)\[(.*)\]$")


def _parse_shape(s: str | None) -> tuple[str | None, list[int]]:
    """``'bfloat16[257, 4096, 6144]'`` -> ``('bfloat16', [257, 4096, 6144])``."""
    m = _SHAPE_RE.match(s or "")
    return (m.group(1), [int(x) for x in m.group(2).split(",") if x.strip()]) if m else (None, [])


def _gemm_mode(qmods: dict, needles: list[str], what: str) -> common.GEMMQuantMode:
    """Quant mode of the first traced module matching any needle."""
    for mod, cls in qmods.items():
        if any(n in mod for n in needles):
            mode = GEMM_QUANT_BY_CLASS.get(cls)
            if mode is None:
                raise RecipeGapError(f"{what}: no GEMMQuantMode mapping for quant class {cls} ({mod})")
            return mode
    raise RecipeGapError(f"{what}: no module matching {needles} in recipe quant_methods")


def _pick_phase(kind_data: dict, prefix: str, prefer_long: bool) -> dict:
    """Representative traced phase: longest-isl prefill (sparse/long branch) or
    shortest-context decode."""
    cands = {k: v for k, v in kind_data.items() if k.startswith(prefix)}
    if not cands:
        raise RecipeGapError(f"no {prefix}* phase in recipe layer kind")
    key = (max if prefer_long else min)(cands, key=lambda k: cands[k].get("isl") or cands[k].get("kv_len") or 0)
    return cands[key]


def get_recipe_model(
    model_path: str,
    model_config,
    backend_name: str = "sglang",
    recipe_path: str | Path | None = None,
    moe_policy: str = "decompose_fused_shared",
    dsa_gemm_override: common.GEMMQuantMode | None = None,
) -> "RecipeModel":
    """Build a RecipeModel, mirroring ``get_model``'s model_info / quant-default
    flow but constructing from a recipe instead of a registered hand class.

    Args:
        recipe_path: explicit recipe YAML; default resolves
            ``recipes/<model_path with '/'->'--'>.recipe.yaml``.
        moe_policy: ``decompose_fused_shared`` (default; see module docstring)
            or ``faithful`` (query traced shapes as-is — typically lands in the
            empirical tier; used to quantify the tolerated divergence).
        dsa_gemm_override: force the DSA module-table gemm key instead of the
            traced o_proj mode. ONLY for documented data-gap workarounds (the
            sglang 0.5.14 skip-indexer tables carry gemm=bfloat16 rows only,
            while GLM-5.2-FP8 traces prove fp8_block projections).
    """
    if recipe_path is None:
        recipe_path = RECIPES_DIR / f"{model_path.replace('/', '--')}.recipe.yaml"
    recipe_path = Path(recipe_path)
    if not recipe_path.exists():
        raise RecipeGapError(
            f"no recipe for {model_path!r}: {recipe_path} does not exist. "
            f"Extract one from framework traces (see docs/recipe_model.md)."
        )
    recipe = yaml.safe_load(recipe_path.read_text())

    model_info = dict(_get_model_info(model_path))
    raw_config = model_info.get("raw_config", {})
    architecture = model_info["architecture"]
    model_info["model_path"] = model_path
    model_info["model_family"] = _architecture_to_model_family(architecture)
    _apply_model_quant_defaults(model_config, raw_config, architecture, backend_name)
    model_config.resolve_moe_parallelism()
    if model_config.cp_size > 1:
        raise RecipeGapError("RecipeModel does not model context parallelism yet (pilot scope)")
    model_config.cp_style = "none"

    rec_arch = (recipe.get("identity") or {}).get("architecture")
    if rec_arch and rec_arch != architecture:
        raise RecipeGapError(f"recipe architecture {rec_arch} != checkpoint {architecture}")
    return RecipeModel(recipe, model_info, model_config, backend_name=backend_name,
                       moe_policy=moe_policy, dsa_gemm_override=dsa_gemm_override)


class RecipeModel(BaseModel):
    """A BaseModel whose per-layer ops come from a traced recipe.

    Not ``@register_model``-registered: recipes are per-(model, framework,
    identity) artifacts, not an architecture family — construct via
    ``get_recipe_model``.
    """

    def __init__(self, recipe: dict, model_info: dict, model_config, *,
                 backend_name: str = "sglang",
                 moe_policy: str = "decompose_fused_shared",
                 dsa_gemm_override: common.GEMMQuantMode | None = None) -> None:
        super().__init__(
            model_info["model_path"], model_info["model_family"], model_info["architecture"],
            model_info["layers"], model_info["n"], model_info["n_kv"], model_info["d"],
            model_info["hidden_size"], model_info["inter_size"], model_info["vocab"],
            model_info["context"], model_config, dict(model_info.get("extra_params") or {}),
        )
        self.recipe = recipe
        self._backend_name = backend_name
        # The engine-handle cache keys on (model_path, ModelConfig, ...) and
        # assumes the op graph is a pure function of those. A recipe model
        # shares its checkpoint path with the registry class but builds a
        # different graph, so declare a discriminator; rust_engine_step folds
        # it into the cache identity (found the hard way: hand-model results
        # silently answered from a recipe-compiled handle).
        prov = (recipe.get("provenance") or {}).get("extractor", "")
        self.engine_identity_extra = (
            f"recipe:{prov}:{moe_policy}:{getattr(dsa_gemm_override, 'name', None)}")
        self.moe_policy = moe_policy
        self.dsa_gemm_override = dsa_gemm_override
        # Every non-obvious recipe->query decision lands here; shadow diffs and
        # reports read it for attribution.
        self.mapping_notes: list[str] = []
        # Memory-path attributes the evaluation layer reads (memory modeling is
        # recipe-independent; checkpoint values, same as hand models).
        self._num_experts = model_info.get("num_experts")
        self._topk = model_info.get("topk")
        self._moe_inter_size = model_info.get("moe_inter_size")

        h = self._hidden_size
        tp = self.config.tp_size
        mtp = mtp_scale_factor(self._nextn, self._num_layers)
        self._mtp_scale_factor = mtp

        counts = (recipe.get("layer_map") or {}).get("layer_kind_counts")
        if not counts:
            raise RecipeGapError("recipe has no layer_map.layer_kind_counts")
        total = sum(counts.values())
        if total != self._num_layers:
            raise RecipeGapError(f"layer_map covers {total} layers, checkpoint has {self._num_layers}")
        missing = [k for k in counts if k not in (recipe.get("layer_kinds") or {})]
        if missing:
            raise RecipeGapError(f"layer kinds present in the real model but never traced: {missing}")

        kv_mode = KV_BY_DTYPE.get((recipe.get("identity") or {}).get("kv_cache_dtype"))
        if kv_mode is None:
            raise RecipeGapError(f"unmapped kv_cache_dtype {recipe.get('identity', {}).get('kv_cache_dtype')}")
        if self.config.kvcache_quant_mode is not None and self.config.kvcache_quant_mode != kv_mode:
            self.mapping_notes.append(
                f"kv identity: recipe traced {kv_mode.name}, ModelConfig default was "
                f"{self.config.kvcache_quant_mode.name} -> recipe wins")

        # ---- scaffold (trace-invisible; same generic formulas as hand models) ----
        self.context_ops.append(ops.Embedding("context_embedding", 1, self._vocab_size, h, 0.3))
        self.context_ops.append(ops.ElementWise("context_add_norm_1", self._num_layers, 2 * h, 2 * h, 0.8))
        self.context_ops.append(ops.ElementWise("context_add_norm_2", self._num_layers, 2 * h, 2 * h, 0.8))
        self.generation_ops.append(ops.Embedding("generation_embedding", mtp, self._vocab_size, h, 0.3))
        self.generation_ops.append(ops.ElementWise("generation_add_norm_1", self._num_layers * mtp, 2 * h, 2 * h, 0.8))
        self.generation_ops.append(ops.ElementWise("generation_add_norm_2", self._num_layers * mtp, 2 * h, 2 * h, 0.8))

        # ---- recipe-driven per-layer ops, one group per traced layer kind ----
        for kind, count in sorted(counts.items()):
            self._build_kind(kind, count, recipe["layer_kinds"][kind], kv_mode)

        self.context_ops.append(ops.GEMM(
            "context_logits_gemm", 1, self._vocab_size // tp, h, common.GEMMQuantMode.bfloat16))
        self.generation_ops.append(ops.GEMM(
            "generation_logits_gemm", mtp, self._vocab_size // tp, h, common.GEMMQuantMode.bfloat16))
        pp = self.config.pp_size
        self.context_ops.append(ops.P2P("context_p2p", pp - 1, h, pp))
        self.generation_ops.append(ops.P2P("generation_p2p", (pp - 1) * mtp, h, pp))

    # ------------------------------------------------------------------
    def _build_kind(self, kind: str, count: int, kdata: dict, kv_mode) -> None:
        qmods = self.recipe["quant_methods_by_module"].get(kind) or {}
        weights = (self.recipe.get("weights_by_kind") or {}).get(kind) or {}
        ctx = _pick_phase(kdata, "prefill:", prefer_long=True)  # sparse/long branch representative
        gen = _pick_phase(kdata, "decode:", prefer_long=False)

        self._build_attention(kind, count, gen, qmods, kv_mode)
        if any(o["op"].startswith("moe::") for o in gen["layer_ops"]):
            self._build_moe(kind, count, gen, qmods, weights)
        else:
            self._build_dense_mlp(kind, count, qmods, weights)

    def _build_attention(self, kind: str, count: int, gen: dict, qmods: dict, kv_mode) -> None:
        cfg = self.config
        attn_ops_gen = [o for o in gen["layer_ops"] if o["op"].startswith("attn::")]
        if not attn_ops_gen:
            raise RecipeGapError(f"{kind}: no attention span in decode trace")
        backend_cls = attn_ops_gen[0]["op"].split("::")[1].split(".")[0]
        if backend_cls != "DeepseekSparseAttnBackend":
            raise RecipeGapError(f"{kind}: pilot only maps DSA attention, got {backend_cls}")

        # FMHA dtype from the traced attention input dtype (GLM: bf16 even on
        # fp8 checkpoints — matches the DSA tables' mla_dtype coverage).
        q_dtype = _parse_shape((attn_ops_gen[0].get("in_shapes") or [None])[0])[0]
        fmha_mode = common.FMHAQuantMode.bfloat16 if q_dtype in ("bfloat16", None) else common.FMHAQuantMode.fp8
        attn_modes = {
            "q": _gemm_mode(qmods, ["self_attn.q_b_proj", "self_attn.q_proj"], f"{kind}.q"),
            "kv": _gemm_mode(qmods, ["self_attn.kv_b_proj", "self_attn.fused_qkv_a_proj"], f"{kind}.kv"),
            "o": _gemm_mode(qmods, ["self_attn.o_proj"], f"{kind}.o"),
            "indexer": _gemm_mode(qmods, ["self_attn.indexer.wk_weights_proj", "self_attn.indexer.wk"],
                                  f"{kind}.indexer"),
        }
        indexer_classes = {c for m, c in qmods.items() if ".indexer." in m and "norm" not in m}
        if len(indexer_classes) > 1:
            self.mapping_notes.append(
                f"{kind}: indexer projections have mixed quant {sorted(indexer_classes)}; "
                f"module perf key follows wk_weights_proj -> {attn_modes['indexer'].name}")

        # Module tables carry ONE gemm key; follow o_proj (largest projection —
        # same convention the hand model documents).
        dsa_gemm = attn_modes["o"]
        if self.dsa_gemm_override is not None and self.dsa_gemm_override != dsa_gemm:
            self.mapping_notes.append(
                f"{kind}: DSA module gemm key OVERRIDDEN {dsa_gemm.name} -> "
                f"{self.dsa_gemm_override.name} (data-gap workaround; traced projections are {dsa_gemm.name})")
            dsa_gemm = self.dsa_gemm_override

        # full vs shared indexer comes from the layer-kind taxonomy itself; the
        # per-kind split (frac 1.0 / 0.0 x config-derived counts) is numerically
        # identical to the hand model's fraction amortization — validated exact
        # in the pilot shadow diff.
        frac = 1.0 if kind.startswith("full_indexer") else 0.0
        local_heads = self._num_heads // cfg.tp_size
        topk_freq = int(((self.recipe.get("layer_map") or {}).get("branch_params") or {})
                        .get("index_topk_freq") or 1)

        self.context_ops.append(ops.ContextDSAModule(
            f"context_attention[{kind}]", count, local_heads, kv_mode, fmha_mode, dsa_gemm,
            architecture=self.architecture, cp_size=cfg.cp_size, index_topk_freq=topk_freq,
            dsa_full_layer_fraction=frac, attn_projection_quant_modes=attn_modes))
        self.generation_ops.append(ops.GenerationDSAModule(
            f"generation_attention[{kind}]", count * self._mtp_scale_factor, local_heads, kv_mode,
            dsa_gemm, architecture=self.architecture, index_topk_freq=topk_freq,
            dsa_full_layer_fraction=frac, attn_projection_quant_modes=attn_modes))

    def _build_moe(self, kind: str, count: int, gen: dict, qmods: dict, weights: dict) -> None:
        cfg = self.config
        h = self._hidden_size
        mtp = self._mtp_scale_factor

        _, w13 = _parse_shape(weights.get("model.layers.*.mlp.experts.w13_weight"))
        if len(w13) != 3:
            raise RecipeGapError(f"{kind}: cannot read w13_weight [E, 2I, H] from recipe weights: {w13}")
        num_experts_traced, inter_traced = w13[0], w13[1] // 2
        _, gate_w = _parse_shape(weights.get("model.layers.*.mlp.gate.weight"))
        router_n = gate_w[0] if gate_w else num_experts_traced

        topk_traced = None
        for o in gen["layer_ops"]:
            for s in (o.get("in_shapes") or []) + list((o.get("kw") or {}).values()):
                if isinstance(s, str) and re.search(r"(^|\.)topk_ids=", s):
                    topk_traced = _parse_shape(s.split("=", 1)[1])[1][-1]
        if topk_traced is None:
            raise RecipeGapError(f"{kind}: topk_ids shape not captured in decode trace")

        moe_cls = next((c for c in set(qmods.values()) if c in MOE_QUANT_BY_CLASS), None)
        if moe_cls is None:
            raise RecipeGapError(f"{kind}: no known FusedMoE quant class in {sorted(set(qmods.values()))}")
        moe_mode = MOE_QUANT_BY_CLASS[moe_cls]

        # Tolerated-divergence policy (owner decision, GLM-5.2 pilot): frameworks
        # that fuse the shared expert into the routed MoE (traced experts >
        # router width) are mapped back onto the collected decomposition instead
        # of forcing a collection-pipeline change — physically equivalent work;
        # impact quantified <=3% TTFT, <0.5% TPOT at b=128. "faithful" keeps the
        # traced shape (empirical tier) for re-quantification.
        n_fused_shared = max(0, num_experts_traced - router_n)
        decompose = self.moe_policy == "decompose_fused_shared" and n_fused_shared > 0
        if decompose:
            num_experts_q, topk_q = router_n, topk_traced - n_fused_shared
            self.mapping_notes.append(
                f"{kind}: TOLERATED divergence — traced fused MoE ({num_experts_traced} experts, "
                f"topk {topk_traced}) mapped to collected decomposition MoE({num_experts_q}, "
                f"topk {topk_q}) + {n_fused_shared} shared-expert FFN (policy {self.moe_policy})")
        else:
            num_experts_q, topk_q = num_experts_traced, topk_traced
            if n_fused_shared > 0:
                self.mapping_notes.append(
                    f"{kind}: FAITHFUL MoE query ({num_experts_traced} experts, topk {topk_traced}); "
                    f"collected grid is ({router_n}, topk {topk_traced - n_fused_shared}) + shared "
                    f"FFN, so expect empirical (not silicon) provenance here.")

        wd = cfg.workload_distribution + "_1.01" if cfg.workload_distribution == "power_law" \
            else cfg.workload_distribution
        moe_args = (h, inter_traced, topk_q, num_experts_q, cfg.moe_tp_size, cfg.moe_ep_size)
        shared_gemm_mode = {common.MoEQuantMode.fp8_block: common.GEMMQuantMode.fp8_block}.get(
            moe_mode, common.GEMMQuantMode.bfloat16)

        def shared_ffn_ops(phase: str, scale: float) -> list:
            return [
                ops.GEMM(f"{phase}_shared_gate_up_gemm[{kind}]", scale,
                         2 * inter_traced * n_fused_shared // cfg.moe_tp_size, h, shared_gemm_mode),
                ops.ElementWise(f"{phase}_shared_act_gate[{kind}]", scale,
                                2 * inter_traced * n_fused_shared // cfg.moe_tp_size,
                                inter_traced * n_fused_shared // cfg.moe_tp_size, 0.8),
                ops.GEMM(f"{phase}_shared_ffn2_gemm[{kind}]", scale,
                         h, inter_traced * n_fused_shared // cfg.moe_tp_size, shared_gemm_mode),
            ]

        if decompose:
            self.context_ops.extend(shared_ffn_ops("context", count))
        self.context_ops.append(ops.GEMM(
            f"context_router_gemm[{kind}]", count, router_n, h, common.GEMMQuantMode.bfloat16))
        self.context_ops.append(ops.MoEDispatch(
            f"context_moe_pre_dispatch[{kind}]", count, h, topk_q, num_experts_q,
            cfg.moe_tp_size, cfg.moe_ep_size, cfg.attention_dp_size, True,
            quant_mode=moe_mode, attn_cp_size=cfg.cp_size, backend=self._backend_name))
        self.context_ops.append(ops.MoE(
            f"context_moe[{kind}]", count, *moe_args, moe_mode, wd, cfg.attention_dp_size))
        self.context_ops.append(ops.MoEDispatch(
            f"context_moe_post_dispatch[{kind}]", count, h, topk_q, num_experts_q,
            cfg.moe_tp_size, cfg.moe_ep_size, cfg.attention_dp_size, False,
            quant_mode=moe_mode, attn_cp_size=cfg.cp_size, backend=self._backend_name))

        gen_routed = [
            ops.GEMM(f"generation_router_gemm[{kind}]", count * mtp, router_n, h,
                     common.GEMMQuantMode.bfloat16),
            ops.MoEDispatch(f"generation_moe_pre_dispatch[{kind}]", count * mtp, h, topk_q,
                            num_experts_q, cfg.moe_tp_size, cfg.moe_ep_size, cfg.attention_dp_size,
                            True, quant_mode=moe_mode, attn_cp_size=cfg.cp_size, is_context=False,
                            backend=self._backend_name),
            ops.MoE(f"generation_moe[{kind}]", count * mtp, *moe_args, moe_mode, wd,
                    cfg.attention_dp_size),
            ops.MoEDispatch(f"generation_moe_post_dispatch[{kind}]", count * mtp, h, topk_q,
                            num_experts_q, cfg.moe_tp_size, cfg.moe_ep_size, cfg.attention_dp_size,
                            False, quant_mode=moe_mode, attn_cp_size=cfg.cp_size, is_context=False,
                            backend=self._backend_name),
        ]
        if decompose:
            # decode: mirror the collected decomposition's overlap structure
            self.generation_ops.append(ops.OverlapOp(
                f"generation_moe_overlap[{kind}]", group_a=gen_routed,
                group_b=shared_ffn_ops("generation", count * mtp)))
        else:
            # faithful: the trace shows ONE fused sequence; nothing to overlap
            self.generation_ops.extend(gen_routed)

    def _build_dense_mlp(self, kind: str, count: int, qmods: dict, weights: dict) -> None:
        cfg = self.config
        h = self._hidden_size
        mtp = self._mtp_scale_factor
        gu_key = next((k for k in weights if "mlp.gate_up_proj" in k), None)
        dn_key = next((k for k in weights if "mlp.down_proj" in k), None)
        if not (gu_key and dn_key):
            raise RecipeGapError(f"{kind}: dense MLP weights not found in recipe "
                                 f"(have {[k for k in weights if '.mlp.' in k]})")
        _, gu = _parse_shape(weights[gu_key])  # [2I/traced_tp, H]; weights_by_kind comes from tp1
        inter = gu[0] // 2
        gu_mode = _gemm_mode(qmods, ["mlp.gate_up_proj"], f"{kind}.gate_up")
        dn_mode = _gemm_mode(qmods, ["mlp.down_proj"], f"{kind}.down")
        tp = cfg.tp_size

        self.context_ops.append(ops.GEMM(
            f"context_dense_gate_up_gemm[{kind}]", count, 2 * inter // tp, h, gu_mode))
        self.context_ops.append(ops.ElementWise(
            f"context_dense_act[{kind}]", count, 2 * inter // tp, inter // tp, 0.8))
        self.context_ops.append(ops.GEMM(
            f"context_dense_down_gemm[{kind}]", count, h, inter // tp, dn_mode))
        self.generation_ops.append(ops.GEMM(
            f"generation_dense_gate_up_gemm[{kind}]", count * mtp, 2 * inter // tp, h, gu_mode))
        self.generation_ops.append(ops.ElementWise(
            f"generation_dense_act[{kind}]", count * mtp, 2 * inter // tp, inter // tp, 0.8))
        self.generation_ops.append(ops.GEMM(
            f"generation_dense_down_gemm[{kind}]", count * mtp, h, inter // tp, dn_mode))

    # ------------------------------------------------------------------
    def get_kvcache_bytes_per_sequence(self, seq_len: int) -> float:
        # Same DSA formula as the hand model — memory modeling is recipe-independent.
        seq_len = max(0, seq_len)
        extra = self.extra_params if isinstance(self.extra_params, dict) else {}
        kv_lora_rank = extra.get("kv_lora_rank", 512)
        qk_rope_head_dim = extra.get("qk_rope_head_dim", 64)
        index_head_dim = extra.get("index_head_dim", 128)
        return self._num_layers * seq_len * (
            kv_lora_rank * self.config.kvcache_quant_mode.value.memory
            + qk_rope_head_dim * common.GEMMQuantMode.bfloat16.value.memory
            + common.indexer_cache_entry_bytes(index_head_dim)
        )
