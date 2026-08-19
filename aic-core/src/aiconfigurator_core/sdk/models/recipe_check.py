# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recipe check — validate a model's op graph against traced framework facts.

A recipe (``aic-model-recipe/v0`` YAML, reference artifacts under the repo's
``recipes/``) is machine-extracted from REAL framework execution traces: per
(layer_kind, phase) it records the ordered op sequence with module identity,
quant-method classes, tensor shapes and kernels, plus derived branch guards
and tp-validated sharding rules.

The recipe is a REFERENCE, not a source of truth for construction. Models keep
being built the existing way — config interpretation (``_get_model_info`` +
quant defaults) feeding a hand- or AI-authored model class. What this module
adds is the other half of that loop: ``check_model_against_recipe(model,
recipe)`` introspects the ALREADY-BUILT op graph and reports, per op family,
whether it agrees with what the framework actually executed. It never queries
the perf database and never blocks — every comparison lands in a
``RecipeCheckReport`` as ``MATCH`` / ``TOLERATED`` / ``DIVERGENT`` /
``UNCHECKED`` so drift is visible in review and CI, and an AI authoring a new
model class can iterate against it.

Extension model (the anti-"one big stew" rule): comparisons are owned by
**op-family checkers** registered in ``_CHECKER_REGISTRY``. A checker's
``matches()`` keys on TRACED EVIDENCE inside one block of a layer's op
sequence (attention-backend class in the span, ``moe::`` span presence,
module naming) — never on a model name. All LLMs share a small set of op
families, so new models that recombine known families need zero code here; a
genuinely new framework behavior is ONE new registered checker. A traced
block no checker claims is reported ``UNCHECKED``, not silently skipped.

Tolerated divergences are rules, not edits: e.g. sglang fuses the shared
expert into the routed MoE (traced 257 experts / topk 9 for GLM-5.2), while
models + collected data use the 256/topk-8 + shared-FFN decomposition. The MoE
checker recognizes that exact decomposition and reports ``TOLERATED`` (owner
decision: bounded, small effect — re-collecting fused shapes is not worth a
collection-pipeline change); anything else diverging from the trace is
``DIVERGENT``.

Pilot scope: GLM-5.2 family checkers (DSA attention, fused MoE, dense MLP).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import aiconfigurator_core.sdk.operations as ops
from aiconfigurator_core.sdk import common

logger = logging.getLogger(__name__)


class RecipeGapError(RuntimeError):
    """The recipe itself is malformed / missing something the checker needs."""


# Traced quant-method class -> perf-database quant mode (extend as recipes for
# new quant families land).
GEMM_QUANT_BY_CLASS = {
    "Fp8LinearMethod": common.GEMMQuantMode.fp8_block,  # GLM ckpts are 128x128 block-quant
    "UnquantizedLinearMethod": common.GEMMQuantMode.bfloat16,
}
KV_BY_DTYPE = {
    "fp8_e4m3": common.KVCacheQuantMode.fp8,
    "fp8_e5m2": common.KVCacheQuantMode.fp8,
    "bfloat16": common.KVCacheQuantMode.bfloat16,
    "auto": common.KVCacheQuantMode.bfloat16,
    None: common.KVCacheQuantMode.bfloat16,
}

_SHAPE_RE = re.compile(r"^(\w+)\[(.*)\]$")
_SEGMENT_RE = re.compile(r"model\.layers\.\*\.(\w+)")

MATCH = "MATCH"
TOLERATED = "TOLERATED"
DIVERGENT = "DIVERGENT"
UNCHECKED = "UNCHECKED"


@dataclass
class RecipeFinding:
    area: str    # e.g. "attention/dsa", "mlp/fused_moe", "identity/kv"
    status: str  # MATCH | TOLERATED | DIVERGENT | UNCHECKED
    detail: str


@dataclass
class RecipeCheckReport:
    model_path: str
    recipe_path: str
    findings: list[RecipeFinding] = field(default_factory=list)

    @property
    def divergent(self) -> list[RecipeFinding]:
        return [f for f in self.findings if f.status == DIVERGENT]

    @property
    def ok(self) -> bool:
        return not self.divergent

    def render(self) -> str:
        lines = [f"recipe check: {self.model_path} vs {self.recipe_path}"]
        for f in self.findings:
            lines.append(f"  [{f.status:9s}] {f.area}: {f.detail}")
        return "\n".join(lines)


def _parse_shape(s: str | None) -> tuple[str | None, list[int]]:
    """``'bfloat16[257, 4096, 6144]'`` -> ``('bfloat16', [257, 4096, 6144])``."""
    m = _SHAPE_RE.match(s or "")
    return (m.group(1), [int(x) for x in m.group(2).split(",") if x.strip()]) if m else (None, [])


def _segment_blocks(layer_ops: list[dict]) -> list[tuple[str, list[dict]]]:
    """Split one layer's ordered traced ops into functional blocks by top-level
    submodule (``self_attn``, ``mlp``, ...). Module-less spans (e.g. the
    ``moe::`` runner) attach to the block in progress — same attribution rule
    the extractor uses for layers."""
    blocks: list[tuple[str, list[dict]]] = []
    for o in layer_ops:
        m = _SEGMENT_RE.search(o.get("module") or o["op"] or "")
        seg = m.group(1) if m else None
        if seg is None and blocks:
            blocks[-1][1].append(o)
            continue
        seg = seg or "?"
        if blocks and blocks[-1][0] == seg:
            blocks[-1][1].append(o)
        else:
            blocks.append((seg, [o]))
    return blocks


def _pick_phase(kind_data: dict, prefix: str, prefer_long: bool) -> dict:
    cands = {k: v for k, v in kind_data.items() if k.startswith(prefix)}
    if not cands:
        raise RecipeGapError(f"no {prefix}* phase in recipe layer kind")
    key = (max if prefer_long else min)(cands, key=lambda k: cands[k].get("isl") or cands[k].get("kv_len") or 0)
    return cands[key]


def _traced_gemm_mode(qmods: dict, needles: list[str]) -> common.GEMMQuantMode | None:
    for mod, cls in qmods.items():
        if any(n in mod for n in needles):
            return GEMM_QUANT_BY_CLASS.get(cls)
    return None


# ---------------------------------------------------------------------------
# Op-family checkers — the extension axis.
# ---------------------------------------------------------------------------

_CHECKER_REGISTRY: list[type["OpFamilyChecker"]] = []


def register_op_checker(cls: type["OpFamilyChecker"]) -> type["OpFamilyChecker"]:
    _CHECKER_REGISTRY.append(cls)
    return cls


class OpFamilyChecker:
    """Accumulates traced expectations across layer kinds, then compares them
    against the built model once (models may amortize kinds into one op)."""

    family = "?"

    @staticmethod
    def matches(block_ops: list[dict]) -> bool:
        raise NotImplementedError

    def accumulate(self, kind: str, count: int, block_ops: list[dict],
                   qmods: dict, weights: dict, recipe: dict) -> None:
        raise NotImplementedError

    def finalize(self, model, recipe: dict) -> list[RecipeFinding]:
        raise NotImplementedError


@register_op_checker
class DsaAttentionChecker(OpFamilyChecker):
    """DeepSeek sparse attention: the traced self_attn subtree vs the model's
    DSA module ops (layer coverage, full/skip fraction, kv identity, quant
    keys, FMHA dtype)."""

    family = "attention/dsa"

    def __init__(self) -> None:
        self.total_layers = 0
        self.full_layers = 0
        self.traced_o_mode: common.GEMMQuantMode | None = None
        self.traced_q_dtype: str | None = None

    @staticmethod
    def matches(block_ops: list[dict]) -> bool:
        return any(o["op"].startswith("attn::DeepseekSparseAttnBackend") for o in block_ops)

    def accumulate(self, kind, count, block_ops, qmods, weights, recipe) -> None:
        self.total_layers += count
        if kind.startswith("full_indexer"):
            self.full_layers += count
        mode = _traced_gemm_mode(qmods, ["self_attn.o_proj"])
        if mode is not None:
            self.traced_o_mode = mode
        attn = next((o for o in block_ops if o["op"].startswith("attn::")), None)
        if attn:
            self.traced_q_dtype = _parse_shape((attn.get("in_shapes") or [None])[0])[0]

    def finalize(self, model, recipe) -> list[RecipeFinding]:
        out = []
        dsa_ops = [op for op in model.context_ops if isinstance(op, ops.ContextDSAModule)]
        if not dsa_ops:
            return [RecipeFinding(self.family, DIVERGENT,
                                  f"trace shows DSA attention on {self.total_layers} layers, "
                                  f"model has no ContextDSAModule op")]
        scale = sum(op._scale_factor for op in dsa_ops)
        out.append(RecipeFinding(
            self.family, MATCH if scale == self.total_layers else DIVERGENT,
            f"layer coverage: model {scale:g} vs traced {self.total_layers}"))
        model_full = sum(op._scale_factor * op._full_frac for op in dsa_ops)
        out.append(RecipeFinding(
            self.family, MATCH if abs(model_full - self.full_layers) < 0.5 else DIVERGENT,
            f"full-indexer layers: model {model_full:g} vs traced {self.full_layers} "
            f"(shared-index skip amortization)"))
        kv_traced = KV_BY_DTYPE.get((recipe.get("identity") or {}).get("kv_cache_dtype"))
        kv_model = {op._kvcache_quant_mode for op in dsa_ops}
        out.append(RecipeFinding(
            self.family, MATCH if kv_model == {kv_traced} else DIVERGENT,
            f"kv cache identity: model {[m.name for m in kv_model]} vs traced {kv_traced.name}"))
        if self.traced_o_mode is not None:
            gemm_model = {op._gemm_quant_mode for op in dsa_ops}
            status = MATCH if gemm_model == {self.traced_o_mode} else DIVERGENT
            out.append(RecipeFinding(
                self.family, status,
                f"DSA module gemm key: model {[m.name for m in gemm_model]} vs traced projections "
                f"{self.traced_o_mode.name}" + (
                    "" if status == MATCH else
                    " — the checkpoint's attention projections executed with a different quant "
                    "than the model queries the module tables with")))
        if self.traced_q_dtype is not None:
            fmha_expected = common.FMHAQuantMode.bfloat16 if self.traced_q_dtype == "bfloat16" \
                else common.FMHAQuantMode.fp8
            fmha_model = {op._fmha_quant_mode for op in dsa_ops}
            out.append(RecipeFinding(
                self.family, MATCH if fmha_model == {fmha_expected} else DIVERGENT,
                f"FMHA dtype: model {[m.name for m in fmha_model]} vs traced attention input "
                f"{self.traced_q_dtype}"))
        return out


@register_op_checker
class FusedMoeChecker(OpFamilyChecker):
    """Fused-MoE mlp subtree: expert count / topk / intermediate size and MoE
    layer coverage vs the trace. Owns the fused-shared-expert tolerance rule."""

    family = "mlp/fused_moe"

    def __init__(self) -> None:
        self.moe_layers = 0
        self.traced: dict | None = None

    @staticmethod
    def matches(block_ops: list[dict]) -> bool:
        return any(o["op"].startswith("moe::") for o in block_ops)

    def accumulate(self, kind, count, block_ops, qmods, weights, recipe) -> None:
        self.moe_layers += count
        _, w13 = _parse_shape(weights.get("model.layers.*.mlp.experts.w13_weight"))
        if len(w13) != 3:
            raise RecipeGapError(f"{kind}: cannot read w13_weight [E, 2I, H] from recipe weights: {w13}")
        _, gate_w = _parse_shape(weights.get("model.layers.*.mlp.gate.weight"))
        topk = None
        for o in block_ops:
            for s in (o.get("in_shapes") or []) + list((o.get("kw") or {}).values()):
                if isinstance(s, str) and re.search(r"(^|\.)topk_ids=", s):
                    topk = _parse_shape(s.split("=", 1)[1])[1][-1]
        traced = {"experts": w13[0], "inter": w13[1] // 2,
                  "router_n": gate_w[0] if gate_w else w13[0], "topk": topk}
        if self.traced is not None and self.traced != traced:
            raise RecipeGapError(f"{kind}: MoE shape differs across layer kinds: {self.traced} vs {traced}")
        self.traced = traced

    def finalize(self, model, recipe) -> list[RecipeFinding]:
        out = []
        t = self.traced or {}
        moe_ops = [op for op in model.context_ops if isinstance(op, ops.MoE)]
        if not moe_ops:
            return [RecipeFinding(self.family, DIVERGENT,
                                  f"trace shows fused MoE on {self.moe_layers} layers, model has no MoE op")]
        scale = sum(op._scale_factor for op in moe_ops)
        out.append(RecipeFinding(
            self.family, MATCH if scale == self.moe_layers else DIVERGENT,
            f"MoE layer coverage: model {scale:g} vs traced {self.moe_layers}"
            + ("" if scale == self.moe_layers else
               " — layers whose traced mlp is NOT MoE (e.g. the dense head) are being counted as MoE")))
        model_shape = {(op._num_experts, op._topk, op._inter_size) for op in moe_ops}
        faithful = (t.get("experts"), t.get("topk"), t.get("inter"))
        n_fused_shared = max(0, (t.get("experts") or 0) - (t.get("router_n") or 0))
        decomposed = (t.get("router_n"), (t.get("topk") or 0) - n_fused_shared, t.get("inter"))
        if model_shape == {faithful}:
            out.append(RecipeFinding(self.family, MATCH,
                                     f"MoE shape matches traced fused op {faithful}"))
        elif n_fused_shared > 0 and model_shape == {decomposed}:
            # Tolerated divergence (owner decision, GLM-5.2 pilot): the framework
            # fuses the shared expert into the routed MoE; models + collected
            # data keep the decomposition. Physically equivalent work; impact
            # quantified <=3% TTFT / <0.5% TPOT @ b=128. Require the shared-FFN
            # GEMMs to actually be present, else the shared expert vanished.
            h = t["inter"] and next(iter(moe_ops))._hidden_size
            gemm_nk = {(op._n, op._k) for op in model.context_ops if isinstance(op, ops.GEMM)}
            per_rank = t["inter"] * n_fused_shared // (model.config.moe_tp_size or 1)
            has_shared = (2 * per_rank, h) in gemm_nk and (h, per_rank) in gemm_nk
            out.append(RecipeFinding(
                self.family, TOLERATED if has_shared else DIVERGENT,
                f"traced fused MoE {faithful} modeled as decomposition MoE{decomposed} + "
                f"{n_fused_shared} shared-expert FFN"
                + ("" if has_shared else " — but the shared-expert FFN GEMMs are missing")))
        else:
            out.append(RecipeFinding(
                self.family, DIVERGENT,
                f"MoE shape: model {sorted(model_shape)} matches neither traced fused {faithful} "
                f"nor its shared-expert decomposition {decomposed}"))
        return out


@register_op_checker
class DenseMlpChecker(OpFamilyChecker):
    """Plain gated dense MLP: layers whose traced mlp is dense must appear in
    the model as dense-FFN GEMMs with the traced widths."""

    family = "mlp/dense"

    def __init__(self) -> None:
        self.dense_layers = 0
        self.inter: int | None = None

    @staticmethod
    def matches(block_ops: list[dict]) -> bool:
        has_moe = any(o["op"].startswith("moe::") for o in block_ops)
        has_dense = any("mlp.gate_up_proj" in (o.get("module") or "") for o in block_ops)
        return has_dense and not has_moe

    def accumulate(self, kind, count, block_ops, qmods, weights, recipe) -> None:
        self.dense_layers += count
        gu_key = next((k for k in weights if "mlp.gate_up_proj" in k), None)
        if gu_key:
            self.inter = _parse_shape(weights[gu_key])[1][0] // 2  # weights traced at tp1

    def finalize(self, model, recipe) -> list[RecipeFinding]:
        if not self.dense_layers:
            return []
        tp = model.config.tp_size
        h = model._hidden_size
        gemms = [op for op in model.context_ops if isinstance(op, ops.GEMM)]
        gu = [op for op in gemms if (op._n, op._k) == (2 * self.inter // tp, h)]
        dn = [op for op in gemms if (op._n, op._k) == (h, self.inter // tp)]
        covered = min(sum(op._scale_factor for op in gu), sum(op._scale_factor for op in dn)) \
            if gu and dn else 0
        status = MATCH if covered >= self.dense_layers else DIVERGENT
        return [RecipeFinding(
            self.family, status,
            f"dense-MLP layers (inter={self.inter}): traced {self.dense_layers}, "
            f"model covers {covered:g}"
            + ("" if status == MATCH else " — dense head layers are missing from the model op graph"))]


# ---------------------------------------------------------------------------


def check_model_against_recipe(model, recipe: dict | str | Path) -> RecipeCheckReport:
    """Compare a built model's op graph against a traced recipe.

    ``model`` is any ``BaseModel`` (built the normal way, from config
    interpretation). ``recipe`` is a loaded dict or a path to the YAML. Returns
    a report; never raises on divergence (malformed recipes raise
    ``RecipeGapError``).
    """
    recipe_path = "<dict>"
    if not isinstance(recipe, dict):
        import yaml

        recipe_path = str(recipe)
        recipe = yaml.safe_load(Path(recipe).read_text())

    report = RecipeCheckReport(model_path=getattr(model, "model_path", "?"), recipe_path=recipe_path)
    counts = (recipe.get("layer_map") or {}).get("layer_kind_counts")
    if not counts:
        raise RecipeGapError("recipe has no layer_map.layer_kind_counts")
    missing = [k for k in counts if k not in (recipe.get("layer_kinds") or {})]
    if missing:
        raise RecipeGapError(f"layer kinds present in the real model but never traced: {missing}")

    num_layers = getattr(model, "_num_layers", None)
    total = sum(counts.values())
    report.findings.append(RecipeFinding(
        "coverage/layers", MATCH if total == num_layers else DIVERGENT,
        f"recipe layer_map covers {total} layers, model has {num_layers}"))

    kv_traced = KV_BY_DTYPE.get((recipe.get("identity") or {}).get("kv_cache_dtype"))
    kv_model = getattr(model.config, "kvcache_quant_mode", None)
    report.findings.append(RecipeFinding(
        "identity/kv", MATCH if kv_model == kv_traced else DIVERGENT,
        f"kv cache identity: ModelConfig {getattr(kv_model, 'name', None)} vs traced {kv_traced.name}"))

    checkers: dict[type, OpFamilyChecker] = {}
    for kind, count in sorted(counts.items()):
        kdata = recipe["layer_kinds"][kind]
        gen = _pick_phase(kdata, "decode:", prefer_long=False)
        qmods = (recipe.get("quant_methods_by_module") or {}).get(kind) or {}
        weights = (recipe.get("weights_by_kind") or {}).get(kind) or {}
        for block, block_ops in _segment_blocks(gen.get("layer_ops") or gen.get("ops") or []):
            claims = [cls for cls in _CHECKER_REGISTRY if cls.matches(block_ops)]
            if not claims:
                report.findings.append(RecipeFinding(
                    f"block/{kind}.{block}", UNCHECKED,
                    f"no op-family checker claims traced block "
                    f"({[o['op'][:50] for o in block_ops[:3]]}); register one — do not edit existing checkers"))
                continue
            for cls in claims:
                checkers.setdefault(cls, cls()).accumulate(kind, count, block_ops, qmods, weights, recipe)

    for checker in checkers.values():
        report.findings.extend(checker.finalize(model, recipe))
    return report
