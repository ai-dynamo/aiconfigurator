# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model facts — the explicit hf→config conversion step that feeds get_model.

The real failure surface for model building is not the op graph itself but the
facts assembled on the way to it: what the HF config implies structurally
(layer kinds, expert configuration, kv identity), which quant modes apply on
this system/backend, and which of those derivations match what the framework
actually executes. Every bug the GLM-5.2 pilot found lived in that conversion,
not in the evaluation layer.

This module makes the conversion explicit and checkable:

* ``assemble_model_facts(model_path, model_config, backend_name, ...)`` —
  ONE place that derives the structural facts from config interpretation and
  resolves the quant modes. It folds in the system-aware ``resolve_*`` helpers
  that today are the caller's duty (three copies in cli/api, one inline in
  task_v2, and MISSING on the ``compile_engine``/Mocker path), so every entry
  point gets the same answer.
* ``APPROXIMATIONS`` — deliberate modeling simplifications, declared as rules
  with rationale and a measured impact bound. Facts always record the TRUE
  structure; an approximation says how a model is ALLOWED to blur it. Example
  (owner decision): ``first_k_dense_replace`` dense head layers are modeled as
  MoE — simpler, measured -0.4%…-3.5% e2e (overestimate, conservative).
  Checks report such blurs as ``APPROX``, never ``DIVERGENT``.
* ``check_facts_against_dryrun(facts, dryrun_paths)`` — validates the derived
  facts against dry-run trace JSONs (real framework loads: per-module
  quant-method classes, weight shapes, runtime MoE shape, kv dtype). This is
  where "the framework fused the shared expert into a 257-expert/topk-9 op"
  meets the config-derived (256, 8, +1 shared) facts and is recognized as the
  declared approximation.
* ``check_model_against_facts(model, facts)`` — cheap structural assertions on
  a BUILT model (any BaseModel, built the normal way): layer coverage,
  skip-indexer fraction, MoE shape, quant keys, kv identity.

Both checks return reports (``MATCH`` / ``APPROX`` / ``DIVERGENT`` /
``UNCHECKED``) and never block: they run at model-authoring time and in CI,
not inside the build hot path. Dry-run JSONs are the single evidence artifact
(see ``references/dryrun/``); there is no intermediate "recipe" format.

Extension model (the anti-"one big stew" rule): evidence comparisons are owned
by small registered checkers keyed on TRACED/CONFIG EVIDENCE (which config
fields exist, which spans appear), never on a model name.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import aiconfigurator_core.sdk.operations as ops
from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk.models.helpers import (
    _apply_model_quant_defaults,
    _architecture_to_model_family,
    _get_model_info,
    resolve_context_fmha_by_data,
    resolve_dsv4_moe_arch,
    resolve_nvfp4_for_system,
)

logger = logging.getLogger(__name__)

MATCH = "MATCH"
APPROX = "APPROX"
DIVERGENT = "DIVERGENT"
UNCHECKED = "UNCHECKED"

_LAYER_RE = re.compile(r"model\.layers\.(\d+)\.")
_SHAPE_RE = re.compile(r"^(\w+)\[(.*)\]$")

# Traced quant-method class -> quant mode (extend as new quant families land).
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


class FactsGapError(RuntimeError):
    """The inputs (config / dry-run JSON) are missing something facts need."""


@dataclass
class FactFinding:
    area: str    # e.g. "moe/runtime_shape", "attention/quant_key", "coverage/kinds"
    status: str  # MATCH | APPROX | DIVERGENT | UNCHECKED
    detail: str


@dataclass
class FactCheckReport:
    subject: str
    findings: list[FactFinding] = field(default_factory=list)

    @property
    def divergent(self) -> list[FactFinding]:
        return [f for f in self.findings if f.status == DIVERGENT]

    @property
    def ok(self) -> bool:
        return not self.divergent

    def render(self) -> str:
        lines = [f"facts check: {self.subject}"]
        for f in self.findings:
            lines.append(f"  [{f.status:9s}] {f.area}: {f.detail}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Declared approximations — deliberate simplifications, not bugs.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Approximation:
    name: str
    rationale: str
    impact: str

    def applies(self, facts: "ModelFacts") -> bool:
        raise NotImplementedError


class _DenseHeadAsMoe(Approximation):
    def applies(self, facts: "ModelFacts") -> bool:
        return facts.moe is not None and facts.dense_head_layers > 0


class _FusedSharedExpertDecomposed(Approximation):
    def applies(self, facts: "ModelFacts") -> bool:
        return bool(facts.moe and facts.moe.get("n_shared"))


APPROXIMATIONS: list[Approximation] = [
    _DenseHeadAsMoe(
        "dense_head_as_moe",
        rationale="first_k_dense_replace dense head layers are deliberately modeled as MoE "
                  "layers — simpler model classes, bounded effect (owner decision, 2026-08-19)",
        impact="GLM-5.2 pilot: -0.4%..-3.5% e2e, overestimate direction (conservative)",
    ),
    _FusedSharedExpertDecomposed(
        "fused_shared_expert_decomposed",
        rationale="frameworks fuse the shared expert into the routed MoE at runtime "
                  "(traced experts = routed + shared, topk + shared); models and collected "
                  "data keep the decomposition — same expert-invocations/token and weight "
                  "bytes, not worth a collection-pipeline change (owner decision)",
        impact="GLM-5.2 pilot: <=3% TTFT / <0.5% TPOT @ b=128 vs faithful fused query",
    ),
]


# ---------------------------------------------------------------------------
# Facts assembly (the explicit hf→config conversion).
# ---------------------------------------------------------------------------


@dataclass
class ModelFacts:
    model_path: str
    architecture: str
    model_family: str
    num_layers: int
    # layer-kind taxonomy derived from config interpretation, e.g.
    # {"full_indexer_dense": 3, "full_indexer_moe": 18, "shared_indexer_moe": 57}
    layer_kinds: dict[str, int]
    # {"num_routed", "topk", "n_shared", "inter"} or None for dense-only models
    moe: dict | None
    # resolved quant identity (after checkpoint inference + system-aware remaps)
    quant: dict  # {"gemm", "moe", "kvcache", "fmha", "comm"} -> enum
    branch_params: dict
    approximations: list[str]

    @property
    def dense_head_layers(self) -> int:
        return sum(c for k, c in self.layer_kinds.items() if k.endswith("_dense") or k == "dense")

    @property
    def moe_layers(self) -> int:
        return sum(c for k, c in self.layer_kinds.items() if k.endswith("_moe") or k == "moe")


def resolve_model_quant_modes(
    model_config,
    model_path: str,
    backend_name: str,
    system_name: str | None = None,
    database=None,
    *,
    is_context_role: bool = True,
) -> None:
    """ONE choke point for quant-mode resolution (checkpoint inference + the
    system-aware remaps that are currently each caller's duty).

    Today cli/api.py carries three copies of these calls, task_v2 inlines a
    fourth, and the ``compile_engine`` (Rust/Mocker) path has none — so an
    embedded caller silently keeps e.g. native ``nvfp4`` compute on Hopper.
    Entry points should call this instead of hand-picking resolve_* helpers.
    """
    info = _get_model_info(model_path)
    raw_config = info.get("raw_config", {})
    _apply_model_quant_defaults(model_config, raw_config, info["architecture"], backend_name)
    resolve_dsv4_moe_arch(model_config, model_path, system_name=system_name, backend_name=backend_name)
    resolve_nvfp4_for_system(model_config, system_name, model_path)
    if database is not None:
        resolve_context_fmha_by_data(model_config, model_path, database, backend_name,
                                     is_context_role=is_context_role)


def _derive_layer_kinds(raw_config: dict, num_layers: int) -> dict[str, int]:
    """Layer-kind taxonomy from config interpretation, keyed on WHICH FIELDS
    EXIST (config evidence), not on a model name.

    * ``indexer_types`` x ``mlp_layer_types`` (GLM-5.2 style): full/shared
      indexer x moe/dense — matches the dummy-variant taxonomy dry runs use.
    * ``first_k_dense_replace`` + routed experts (DeepSeek style): dense head
      + uniform moe.
    * routed experts only: all moe. Otherwise: all dense.
    """
    it, ml = raw_config.get("indexer_types"), raw_config.get("mlp_layer_types")
    if it and ml:
        counts = Counter(f"{a}_indexer_{'moe' if b == 'sparse' else 'dense'}"
                         for a, b in zip(it, ml))
        if sum(counts.values()) != num_layers:
            raise FactsGapError(
                f"indexer_types x mlp_layer_types cover {sum(counts.values())} layers, "
                f"config says {num_layers}")
        return dict(counts)
    if raw_config.get("n_routed_experts") or raw_config.get("num_experts"):
        k = int(raw_config.get("first_k_dense_replace") or 0)
        out = {}
        if k:
            out["dense"] = k
        out["moe"] = num_layers - k
        return out
    return {"dense": num_layers}


def assemble_model_facts(
    model_path: str,
    model_config,
    backend_name: str = "sglang",
    system_name: str | None = None,
    database=None,
) -> ModelFacts:
    """Derive the facts that feed get_model, with quant resolution folded in.

    Mutates ``model_config`` exactly like the resolve chain the CLI/task paths
    run today (idempotent: ``get_model`` re-applying the defaults is a no-op).
    """
    info = dict(_get_model_info(model_path))
    raw_config = info.get("raw_config", {})
    architecture = info["architecture"]
    num_layers = int(info["layers"])

    resolve_model_quant_modes(model_config, model_path, backend_name,
                              system_name=system_name, database=database)

    moe = None
    routed = raw_config.get("n_routed_experts") or raw_config.get("num_experts") or info.get("num_experts")
    if routed:
        moe = {
            "num_routed": int(routed),
            "topk": int(raw_config.get("num_experts_per_tok") or info.get("topk")),
            "n_shared": int(raw_config.get("n_shared_experts") or 0),
            "inter": int(raw_config.get("moe_intermediate_size") or info.get("moe_inter_size")),
        }

    facts = ModelFacts(
        model_path=model_path,
        architecture=architecture,
        model_family=_architecture_to_model_family(architecture),
        num_layers=num_layers,
        layer_kinds=_derive_layer_kinds(raw_config, num_layers),
        moe=moe,
        quant={
            "gemm": model_config.gemm_quant_mode,
            "moe": model_config.moe_quant_mode,
            "kvcache": model_config.kvcache_quant_mode,
            "fmha": model_config.fmha_quant_mode,
            "comm": model_config.comm_quant_mode,
        },
        branch_params={k: raw_config.get(k) for k in
                       ("index_topk", "index_topk_freq", "index_skip_topk_offset",
                        "first_k_dense_replace") if raw_config.get(k) is not None},
        approximations=[],
    )
    facts.approximations = [a.name for a in APPROXIMATIONS if a.applies(facts)]
    return facts


# ---------------------------------------------------------------------------
# Dry-run evidence: distilled summaries, not raw dumps.
#
# The probe's raw trace JSON (~100-180KB per variant: ordered op sequences,
# kernel timings, call paths, full weight tables) belongs to the facts
# archive. What the repo carries — and what checks consume — is the DISTILLED
# summary: per layer kind, the per-module quant classes, the runtime MoE /
# dense-MLP shape, the kv identity, and compact branch evidence, plus
# provenance pointing back to the raw traces. ``summarize_dryruns`` is the
# single owner of that format (producer and consumer live together).
# ---------------------------------------------------------------------------

DRYRUN_SUMMARY_SCHEMA = "aic-dryrun-summary/v1"


def _parse_shape(s: str | None) -> tuple[str | None, list[int]]:
    m = _SHAPE_RE.match(s or "")
    return (m.group(1), [int(x) for x in m.group(2).split(",") if x.strip()]) if m else (None, [])


def _norm_module(name: str) -> str:
    return _LAYER_RE.sub("", name).removeprefix("model.layers.")


def _dryrun_kind(rec: dict) -> str:
    """Dummy-variant dry runs encode the layer kind in the model dir name."""
    name = Path(rec.get("model_path", "")).name
    return name.split("__", 1)[1] if "__" in name else name


def _dryrun_topk(rec: dict) -> int | None:
    for phase in (rec.get("phases") or {}).values():
        for o in phase.get("ops") or []:
            for s in (o.get("in") or []) + list((o.get("kw") or {}).values()):
                if isinstance(s, str) and re.search(r"(^|\.)topk_ids=", s):
                    return _parse_shape(s.split("=", 1)[1])[1][-1]
    return None


def _prefill_attn_kernels(rec: dict) -> dict[int, str]:
    """isl -> leading attention kernel per traced prefill phase (branch evidence)."""
    out: dict[int, str] = {}
    for key, phase in (rec.get("phases") or {}).items():
        m = re.match(r"prefill:.*isl(\d+)", key)
        if not m:
            continue
        for o in phase.get("ops") or []:
            if o.get("span", "").startswith("AIC::attn::") and o.get("depth") == 1 and o.get("kernels"):
                out[int(m.group(1))] = next(iter(o["kernels"])).removeprefix("void ")[:70]
                break
    return out


def summarize_dryruns(raw_records: list[dict]) -> dict:
    """Distill raw probe trace records (one per layer-kind variant) into the
    per-model evidence summary the checks consume. Deterministic; re-run it
    whenever traces are refreshed."""
    if not raw_records:
        raise FactsGapError("no dry-run records to summarize")
    first = raw_records[0]
    summary: dict = {
        "schema": DRYRUN_SUMMARY_SCHEMA,
        "model": Path(first.get("model_path", "")).name.split("__")[0],
        "framework": {"name": "sglang", "version": first.get("sglang_version")},
        "platform": "sm90",
        "tp": first.get("tp", 1),
        "kv_cache_dtype": (first.get("server_args_resolved") or {}).get("kv_cache_dtype"),
        "layer_kinds": {},
        "provenance": {"probe": "opharness probe/recipe_probe.py",
                       "traces": [rec.get("model_path") for rec in raw_records]},
    }
    for rec in raw_records:
        kind = _dryrun_kind(rec)
        qmods = {}
        for mod, cls in (rec.get("quant_methods_by_module") or {}).items():
            key = _norm_module(mod)
            if key.startswith(("self_attn.", "mlp.")):
                if qmods.get(key, cls) != cls:
                    raise FactsGapError(f"{kind}: quant class differs across layers for {key}")
                qmods[key] = cls
        entry: dict = {"quant_by_module": dict(sorted(qmods.items()))}
        w13 = next((v for k, v in (rec.get("weights") or {}).items()
                    if "mlp.experts.w13_weight" in k), None)
        _, w13_shape = _parse_shape(w13)
        if len(w13_shape) == 3:
            gate = next((v for k, v in (rec.get("weights") or {}).items()
                         if "mlp.gate.weight" in k), None)
            entry["moe_runtime"] = {
                "num_experts": w13_shape[0], "inter": w13_shape[1] // 2,
                "router_width": _parse_shape(gate)[1][0] if gate else w13_shape[0],
                "topk": _dryrun_topk(rec),
            }
        gu = next((v for k, v in (rec.get("weights") or {}).items()
                   if "mlp.gate_up_proj.weight" in k), None)
        if gu:
            entry["dense_mlp"] = {"inter": _parse_shape(gu)[1][0] // 2}
        kernels_by_isl = _prefill_attn_kernels(rec)
        if len(kernels_by_isl) >= 2:
            lo, hi = min(kernels_by_isl), max(kernels_by_isl)
            if kernels_by_isl[lo] != kernels_by_isl[hi]:
                branch_params = (rec.get("model_config") or {}).get("branch_params") or {}
                entry["prefill_branch"] = {
                    f"isl{lo}": kernels_by_isl[lo], f"isl{hi}": kernels_by_isl[hi],
                    # only config scalars that lie between the two probed lengths
                    # can be the switch threshold (dummy-variant fields filtered out)
                    "threshold_candidates": {k: v for k, v in branch_params.items()
                                             if isinstance(v, int) and lo <= v < hi},
                }
        summary["layer_kinds"][kind] = entry
    return summary


def check_facts_against_dryrun(facts: ModelFacts, summary: dict | str | Path) -> FactCheckReport:
    """Validate config-derived facts against a distilled dry-run summary
    (``summarize_dryruns`` output; see ``references/dryrun/``).

    Reports per fact area; layer kinds without evidence come back ``UNCHECKED``
    (coverage is the probe target matrix's job — surface, never silence).
    """
    if not isinstance(summary, dict):
        import yaml

        summary = yaml.safe_load(Path(summary).read_text())
    kinds_ev: dict = summary.get("layer_kinds") or {}
    report = FactCheckReport(subject=f"{facts.model_path} facts vs dry-run summary "
                                     f"({summary.get('framework', {}).get('version')}, "
                                     f"{summary.get('platform')})")

    unknown = sorted(set(kinds_ev) - set(facts.layer_kinds))
    if unknown:
        report.findings.append(FactFinding(
            "coverage/kinds", DIVERGENT,
            f"dry-run kinds {unknown} do not exist in config-derived taxonomy "
            f"{sorted(facts.layer_kinds)}"))
    for kind in sorted(set(facts.layer_kinds) - set(kinds_ev)):
        report.findings.append(FactFinding(
            "coverage/kinds", UNCHECKED,
            f"layer kind '{kind}' ({facts.layer_kinds[kind]} layers) has no dry-run evidence"))

    kv_traced = KV_BY_DTYPE.get(summary.get("kv_cache_dtype"))
    report.findings.append(FactFinding(
        "identity/kv", MATCH if kv_traced == facts.quant["kvcache"] else DIVERGENT,
        f"facts kv {getattr(facts.quant['kvcache'], 'name', None)} vs framework "
        f"{getattr(kv_traced, 'name', None)}"))

    for kind, ev in sorted(kinds_ev.items()):
        if kind not in facts.layer_kinds:
            continue
        o_cls = (ev.get("quant_by_module") or {}).get("self_attn.o_proj")
        if o_cls is not None:
            traced_gemm = GEMM_QUANT_BY_CLASS.get(o_cls)
            report.findings.append(FactFinding(
                f"quant/{kind}",
                MATCH if traced_gemm == facts.quant["gemm"] else DIVERGENT,
                f"facts gemm {getattr(facts.quant['gemm'], 'name', None)} vs framework "
                f"attention projections {o_cls}"))
        rt = ev.get("moe_runtime")
        if rt is not None:
            if not facts.moe:
                report.findings.append(FactFinding(
                    f"moe/{kind}", DIVERGENT, "framework runs MoE but facts derive none"))
                continue
            cfg = facts.moe
            fused = (rt["num_experts"] == cfg["num_routed"] + cfg["n_shared"]
                     and rt.get("topk") == cfg["topk"] + cfg["n_shared"])
            exact = rt["num_experts"] == cfg["num_routed"] and rt.get("topk") == cfg["topk"]
            inter_ok = rt["inter"] == cfg["inter"]
            if exact and inter_ok:
                status, note = MATCH, "runtime MoE matches config decomposition"
            elif fused and inter_ok and "fused_shared_expert_decomposed" in facts.approximations:
                status = APPROX
                note = (f"runtime fuses shared expert: ({rt['num_experts']} experts, topk "
                        f"{rt.get('topk')}) vs config ({cfg['num_routed']}, {cfg['topk']}, "
                        f"+{cfg['n_shared']} shared) — declared approximation "
                        f"'fused_shared_expert_decomposed'")
            else:
                status = DIVERGENT
                note = (f"runtime MoE ({rt['num_experts']} experts, topk {rt.get('topk')}, "
                        f"inter {rt['inter']}) matches neither config ({cfg['num_routed']}, "
                        f"{cfg['topk']}, inter {cfg['inter']}) nor a declared approximation")
            report.findings.append(FactFinding(f"moe/{kind}", status, note))
    return report


# ---------------------------------------------------------------------------
# Check 2: built model vs facts (any BaseModel, built the normal way).
# ---------------------------------------------------------------------------


def check_model_against_facts(model, facts: ModelFacts) -> FactCheckReport:
    """Structural assertions on a built model's op graph, honoring declared
    approximations. Introspection-only — no perf-database queries."""
    report = FactCheckReport(subject=f"{getattr(model, 'model_path', '?')} model vs facts")

    kv_model = getattr(model.config, "kvcache_quant_mode", None)
    report.findings.append(FactFinding(
        "identity/kv", MATCH if kv_model == facts.quant["kvcache"] else DIVERGENT,
        f"model kv {getattr(kv_model, 'name', None)} vs facts "
        f"{getattr(facts.quant['kvcache'], 'name', None)}"))

    dsa_ops = [op for op in model.context_ops if isinstance(op, ops.ContextDSAModule)]
    if dsa_ops:
        scale = sum(op._scale_factor for op in dsa_ops)
        report.findings.append(FactFinding(
            "attention/coverage", MATCH if scale == facts.num_layers else DIVERGENT,
            f"DSA attention covers {scale:g} layers vs facts {facts.num_layers}"))
        full_expected = sum(c for k, c in facts.layer_kinds.items() if k.startswith("full_indexer"))
        if full_expected:
            model_full = sum(op._scale_factor * op._full_frac for op in dsa_ops)
            report.findings.append(FactFinding(
                "attention/skip_fraction",
                MATCH if abs(model_full - full_expected) < 0.5 else DIVERGENT,
                f"full-indexer layers: model {model_full:g} vs facts {full_expected}"))
        gemm_model = {op._gemm_quant_mode for op in dsa_ops}
        report.findings.append(FactFinding(
            "attention/quant_key",
            MATCH if gemm_model == {facts.quant["gemm"]} else DIVERGENT,
            f"DSA module gemm key {[m.name for m in gemm_model]} vs facts gemm "
            f"{getattr(facts.quant['gemm'], 'name', None)}"))

    if facts.moe:
        moe_ops = [op for op in model.context_ops if isinstance(op, ops.MoE)]
        if not moe_ops:
            report.findings.append(FactFinding(
                "moe/coverage", DIVERGENT, f"facts say {facts.moe_layers} MoE layers, model has no MoE op"))
        else:
            scale = sum(op._scale_factor for op in moe_ops)
            if scale == facts.moe_layers:
                report.findings.append(FactFinding(
                    "moe/coverage", MATCH, f"MoE covers {scale:g} layers == facts {facts.moe_layers}"))
            elif scale == facts.num_layers and facts.dense_head_layers \
                    and "dense_head_as_moe" in facts.approximations:
                report.findings.append(FactFinding(
                    "moe/coverage", APPROX,
                    f"MoE covers all {scale:g} layers incl. {facts.dense_head_layers} dense head "
                    f"layers — declared approximation 'dense_head_as_moe'"))
            else:
                report.findings.append(FactFinding(
                    "moe/coverage", DIVERGENT,
                    f"MoE covers {scale:g} layers; facts say {facts.moe_layers} MoE "
                    f"(+{facts.dense_head_layers} dense head)"))
            shapes = {(op._num_experts, op._topk, op._inter_size) for op in moe_ops}
            expected = (facts.moe["num_routed"], facts.moe["topk"], facts.moe["inter"])
            report.findings.append(FactFinding(
                "moe/shape", MATCH if shapes == {expected} else DIVERGENT,
                f"MoE (experts, topk, inter): model {sorted(shapes)} vs facts {expected}"))
    return report
