# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from pathlib import Path
from typing import Any, Optional

from jinja2 import Environment

_ENV = Environment()
logger = logging.getLogger(__name__)
_BASE_DIR = Path(__file__).resolve().parent
_RULES_DIR = (_BASE_DIR.parent / "rule_plugin").resolve()
_DEFAULT_RULE_PLUGIN = "default"


def _resolve_rule_plugin_dir(plugin: Optional[str], base_dir: Optional[str] = None) -> str:
    base = Path(base_dir).resolve() if base_dir else _RULES_DIR
    if not plugin or plugin == _DEFAULT_RULE_PLUGIN:
        return str(base)
    if os.path.sep in plugin or (os.path.altsep and os.path.altsep in plugin):
        raise ValueError(f"Rule plugin must be a simple name, got: {plugin!r}")
    candidate = (base / plugin).resolve()
    if not candidate.exists() or not candidate.is_dir():
        raise FileNotFoundError(f"Rule plugin directory not found: {candidate}")
    if base not in candidate.parents and candidate != base:
        raise ValueError(f"Rule plugin path escapes base directory: {candidate}")
    return str(candidate)


def _ensure_scope(pv: dict[str, Any], scope: str) -> dict[str, Any]:
    params = pv.setdefault("params", {})
    return params.setdefault(scope, {})


def _get_scope(pv: dict[str, Any], scope: str) -> Optional[dict[str, Any]]:
    params = pv.setdefault("params", {})
    sc = params.get(scope)
    if isinstance(sc, dict) and sc:
        return sc
    return None


def _eval(expr: str, scope: str, pv: dict[str, Any]) -> Any:
    ctx: dict[str, Any] = {}
    ctx.update(pv)
    service_cfg = pv.get("ServiceConfig", {})
    ctx.update(service_cfg)
    if isinstance(service_cfg, dict):
        ctx.setdefault("ServiceConfig", service_cfg)
    k8s_cfg = pv.get("K8sConfig", {})
    ctx.update(k8s_cfg)
    if isinstance(k8s_cfg, dict):
        ctx.setdefault("K8sConfig", k8s_cfg)
    node_cfg = pv.get("NodeConfig", {})
    if isinstance(node_cfg, dict):
        ctx.update(node_cfg)
        ctx.setdefault("NodeConfig", node_cfg)
    dyn_cfg = pv.get("DynConfig")
    if isinstance(dyn_cfg, dict):
        ctx.setdefault("DynConfig", dyn_cfg)
    # Provide structured aliases for DSL compatibility
    if "SlaConfig" not in ctx:
        sla_cfg = pv.get("SlaConfig")
        if isinstance(sla_cfg, dict):
            ctx["SlaConfig"] = sla_cfg
    sc = pv.get("params", {}).get(scope, {})
    ctx.update(sc)
    # Alias ModelConfig.is_moe -> is_moe for convenience
    mc = pv.get("ModelConfig") or pv.get("model") or pv.get("model_config") or {}
    if not mc and "ServiceConfig" in pv and isinstance(pv["ServiceConfig"], dict):
        svc = pv["ServiceConfig"]
        modeled: dict[str, Any] = {}
        if "is_moe" in svc:
            modeled["is_moe"] = svc.get("is_moe")
        if modeled:
            mc = modeled
    if isinstance(mc, dict):
        ctx.setdefault("ModelConfig", mc)
        if "is_moe" in mc and "is_moe" not in ctx:
            ctx["is_moe"] = mc.get("is_moe")
    isl = sc.get("max_seq_len", pv.get("max_seq_len"))
    bs = sc.get("max_batch_size", pv.get("max_batch_size"))
    ctx["isl"] = isl if isl is not None else 0
    ctx["bs"] = bs if bs is not None else 1
    fn = _ENV.compile_expression(expr.strip())
    return fn(**ctx)


def _parse_assign(line: str) -> Optional[tuple[Optional[str], str, str]]:
    s = line.strip().rstrip(";")
    if not s or s.startswith("#"):
        return None
    parts = s.split("=", 1)
    if len(parts) != 2:
        return None
    left, right = parts[0].strip(), parts[1].strip()
    toks = left.split()
    if len(toks) == 1:
        return (None, toks[0], right)
    if toks[0] in ("prefill", "decode", "agg"):
        return (toks[0], " ".join(toks[1:]).strip(), right)
    # Support multi-scope alias or 'all'
    alias = toks[0]
    allowed = {"prefill", "decode", "agg"}
    parts0 = alias.split("_")
    if "_" in alias and all(p in allowed for p in parts0):
        return (alias, " ".join(toks[1:]).strip(), right)
    return (None, left, right)


def _apply_line(
    assign: tuple[Optional[str], str, str],
    backend: str,
    pv: dict[str, Any],
    default_scope: Optional[str],
) -> None:
    scope, name, expr = assign
    if "." in name:
        prefix, remainder = name.split(".", 1)
        config_targets = {
            "DynConfig": "DynConfig",
            "K8sConfig": "K8sConfig",
            "ServiceConfig": "ServiceConfig",
            "ModelConfig": "ModelConfig",
            "NodeConfig": "NodeConfig",
            "SlaConfig": "SlaConfig",
            "BenchConfig": "BenchConfig",
        }
        target_key = config_targets.get(prefix)
        if target_key:
            target = pv.setdefault(target_key, {})
            if isinstance(target, dict):
                value = _eval(expr, default_scope or "prefill", pv)
                parts = remainder.split(".")
                cur = target
                for part in parts[:-1]:
                    if part not in cur or not isinstance(cur[part], dict):
                        cur[part] = {}
                    cur = cur[part]
                cur[parts[-1]] = value
            return
    sc = scope or default_scope
    if not sc:
        return
    # Expand multi-scope aliases like "prefill_decode", "agg_prefill_decode", or "all"
    allowed = {"prefill", "decode", "agg"}
    if "_" in sc:
        scopes = [s for s in sc.split("_") if s in allowed]
    else:
        scopes = [sc]

    promote_keys = {
        "max_num_tokens",
        "enable_chunked_prefill",
        "max_seq_len",
        "max_batch_size",
    }
    for scope_name in scopes:
        target = _get_scope(pv, scope_name)
        if target is None:
            continue
        value = _eval(expr, scope_name, pv)
        target[name] = value
        # Promote selected names to top-level only for default scope to avoid ambiguity
        if name in promote_keys and default_scope and scope_name == default_scope:
            pv[name] = value


def _load_rule_path(base_dir: str, backend: str) -> Optional[str]:
    p = os.path.join(base_dir, f"{backend}.rule")
    return p if os.path.exists(p) else None


def apply_rule_plugins(
    param_values: dict[str, Any],
    backend: str,
    dsl_dir: Optional[str] = None,
) -> dict[str, Any]:
    rule_name = param_values.get("rule")
    base = _resolve_rule_plugin_dir(rule_name, base_dir=dsl_dir)
    rule_path = _load_rule_path(base, backend)
    if not rule_path:
        return param_values
    with open(rule_path, encoding="utf-8") as f:
        content = f.read().splitlines()

    # Ensure agg_prefill_decode has data so default_scope eval can access tp/ep, etc.
    params_obj = param_values.setdefault("params", {})
    if "agg_prefill_decode" not in params_obj:
        merged: dict[str, Any] = {}
        for sc in ("agg", "prefill", "decode"):
            sc_val = params_obj.get(sc)
            if isinstance(sc_val, dict):
                merged.update(sc_val)
        if merged:
            params_obj["agg_prefill_decode"] = merged

    default_scope = "agg_prefill_decode" if backend in ("trtllm", "vllm", "sglang") else None
    cond_stack: list[tuple[int, bool]] = []
    for idx, line in enumerate(content, start=1):
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        while cond_stack and cond_stack[-1][0] >= indent:
            cond_stack.pop()
        s = line.strip()
        if s.startswith("when ") and s.endswith(":"):
            expr = s[5:-1].strip()
            try:
                active = bool(_eval(expr, default_scope or "prefill", param_values))
            except Exception:
                logger.exception("rule when compile failed at line %d", idx)
                active = False
            cond_stack.append((indent, active))
            continue
        if cond_stack and not all(flag for _, flag in cond_stack):
            continue
        try:
            assign = _parse_assign(s)
        except Exception:
            logger.exception("rule parse failed at line %d", idx)
            assign = None
        if assign:
            try:
                _apply_line(assign, backend, param_values, default_scope)
            except Exception:
                logger.exception("rule apply failed at line %d", idx)
    return param_values
