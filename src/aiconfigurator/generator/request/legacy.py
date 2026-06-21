# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Boundary adapter: GeneratorRequest <-> legacy params dict.

`to_legacy_params` maps the request's typed fields to the per-section input
dicts and re-derives the full params dict via `collect_generator_params` (the
existing derivation), then overlays `overrides.raw`. `from_legacy_params` is the
inverse and captures every section into `overrides.raw` so a
params -> request -> params round-trip is lossless on legacy shapes.

The load-bearing invariant (tests/baseline): for the canary matrix,
`to_legacy_params(from_legacy_params(p))` renders byte-identically to `p`.
"""
from __future__ import annotations

from typing import Any, Optional

from ..aggregators import collect_generator_params
from .schema import (
    BackendSpec,
    CacheSpec,
    GeneratorRequest,
    ModelFacts,
    ModelSpec,
    Overrides,
    Platform,
    RoleSizing,
    SlaSpec,
    Topology,
)

# Sections captured verbatim into overrides.raw (everything except per-role
# `params`, which is represented by topology.roles).
_SECTIONS = (
    "ServiceConfig",
    "K8sConfig",
    "DynConfig",
    "WorkerConfig",
    "SlaConfig",
    "BenchConfig",
    "SflowConfig",
    "NodeConfig",
    "ModelConfig",
    "LlmdConfig",
)
_TOP_LEVEL = ("backend", "generator_dynamo_version", "rule")


def from_legacy_params(params: dict[str, Any], backend: Optional[str] = None) -> GeneratorRequest:
    svc = params.get("ServiceConfig", {}) or {}
    k8s = params.get("K8sConfig", {}) or {}
    dyn = params.get("DynConfig", {}) or {}
    sla = params.get("SlaConfig", {}) or {}
    worker = params.get("WorkerConfig", {}) or {}
    node = params.get("NodeConfig", {}) or {}
    pdict = params.get("params", {}) or {}
    mc = params.get("ModelConfig", {}) or {}

    mode = dyn.get("mode") or ("disagg" if (pdict.get("prefill") and pdict.get("decode")) else "agg")
    roles = {r: RoleSizing.from_params(v) for r, v in pdict.items() if v}
    workers: dict[str, int] = {}
    for r in ("prefill", "decode", "agg"):
        wc = worker.get(f"{r}_workers")
        if wc is not None:
            workers[r] = int(wc)

    # Capture every section + top-level key so to_legacy_params can reproduce
    # the exact legacy shape (the typed fields are the public surface; raw is
    # the lossless backing store).
    raw: dict[str, Any] = {}
    for sec in _SECTIONS:
        for key, val in (params.get(sec) or {}).items():
            raw[f"{sec}.{key}"] = val
    for top in _TOP_LEVEL:
        if top in params:
            raw[top] = params[top]

    model_facts = None
    if mc:
        known = ("is_moe", "nextn", "prefix", "architecture")
        model_facts = ModelFacts(
            is_moe=mc.get("is_moe"),
            nextn=mc.get("nextn"),
            prefix=mc.get("prefix"),
            architecture=mc.get("architecture"),
            extra={k: v for k, v in mc.items() if k not in known},
        )

    return GeneratorRequest(
        model=ModelSpec(
            model_path=svc.get("model_path") or svc.get("served_model_path") or "",
            served_model_name=svc.get("served_model_name"),
        ),
        backend=BackendSpec(
            name=backend or params.get("backend") or "",
            dynamo_version=params.get("generator_dynamo_version"),
        ),
        topology=Topology(mode=mode, total_gpus=None, roles=roles, workers=workers),
        sla=SlaSpec(isl=sla.get("isl"), osl=sla.get("osl")),
        platform=Platform(
            hardware_profile=k8s.get("system_name") or node.get("system_name"),
            transport=k8s.get("transport"),
        ),
        cache=CacheSpec(pvc_name=k8s.get("k8s_pvc_name"), model_cache=k8s.get("k8s_model_cache")),
        model_facts=model_facts,
        overrides=Overrides(raw=raw),
    )


def _section_from_raw(raw: dict[str, Any], section: str, seed: dict[str, Any]) -> dict[str, Any]:
    """Seed values (from typed fields) merged with the captured raw section."""
    out = dict(seed)
    prefix = section + "."
    for key, val in raw.items():
        if key.startswith(prefix):
            out[key[len(prefix):]] = val
    return out


def to_legacy_params(req: GeneratorRequest) -> dict[str, Any]:
    raw = dict(req.overrides.raw or {})

    service_seed: dict[str, Any] = {}
    if req.model.model_path:
        service_seed["model_path"] = req.model.model_path
    if req.model.served_model_name:
        service_seed["served_model_name"] = req.model.served_model_name

    k8s_seed: dict[str, Any] = {}
    if req.platform.hardware_profile:
        k8s_seed["system_name"] = req.platform.hardware_profile
    if req.platform.transport:
        k8s_seed["transport"] = req.platform.transport
    if req.cache.pvc_name:
        k8s_seed["k8s_pvc_name"] = req.cache.pvc_name
    if req.cache.model_cache:
        k8s_seed["k8s_model_cache"] = req.cache.model_cache

    sla_seed: dict[str, Any] = {}
    if req.sla.isl is not None:
        sla_seed["isl"] = req.sla.isl
    if req.sla.osl is not None:
        sla_seed["osl"] = req.sla.osl

    service = _section_from_raw(raw, "ServiceConfig", service_seed)
    k8s = _section_from_raw(raw, "K8sConfig", k8s_seed)
    dyn = _section_from_raw(raw, "DynConfig", {"mode": req.topology.mode})
    sla = _section_from_raw(raw, "SlaConfig", sla_seed)
    bench = _section_from_raw(raw, "BenchConfig", {})
    sflow = _section_from_raw(raw, "SflowConfig", {})
    node = _section_from_raw(raw, "NodeConfig", {})
    worker = _section_from_raw(raw, "WorkerConfig", {})
    model_cfg = _section_from_raw(raw, "ModelConfig", {})
    llmd = _section_from_raw(raw, "LlmdConfig", {})

    role_params = {r: rs.to_params() for r, rs in req.topology.roles.items()}
    # First-class EncodeSpec config lowers into the encode role params (the
    # per-role RoleSizing.extra still wins for any overlapping key).
    if req.encode is not None and "encode" in role_params:
        merged = dict(req.encode.to_params())
        merged.update(role_params["encode"])
        role_params["encode"] = merged

    def _workers(role: str) -> int:
        if role in req.topology.workers:
            return int(req.topology.workers[role])
        return int(worker.get(f"{role}_workers", 1))

    params = collect_generator_params(
        service=service,
        k8s=k8s,
        prefill_params=role_params.get("prefill"),
        decode_params=role_params.get("decode"),
        agg_params=role_params.get("agg"),
        prefill_workers=_workers("prefill"),
        decode_workers=_workers("decode"),
        agg_workers=_workers("agg"),
        num_gpus_per_node=int(node.get("num_gpus_per_node", 8)),
        sla=sla,
        bench=bench or None,
        sflow=sflow or None,
        dyn_config=dyn,
        backend=req.backend.name or None,
        generator_dynamo_version=req.backend.dynamo_version or raw.get("generator_dynamo_version"),
        encode_params=role_params.get("encode"),
        encode_workers=req.topology.workers.get("encode"),
    )

    if model_cfg:
        params["ModelConfig"] = model_cfg
    if llmd:
        params["LlmdConfig"] = llmd
    if raw.get("backend"):
        params["backend"] = raw["backend"]
    if raw.get("rule"):
        params["rule"] = raw["rule"]
    if node.get("system_name"):
        params.setdefault("NodeConfig", {})["system_name"] = node["system_name"]
    return params
