# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    ctx.update(pv.get("service", {}))
    ctx.update(pv.get("k8s", {}))
    # Provide structured aliases for DSL compatibility
    if "SlaConfig" not in ctx:
        sla_cfg = pv.get("sla")
        if isinstance(sla_cfg, dict):
            ctx["SlaConfig"] = sla_cfg
    sc = pv.get("params", {}).get(scope, {})
    ctx.update(sc)
    # Alias ModelConfig.is_moe -> is_moe for convenience
    mc = pv.get("ModelConfig") or pv.get("model") or pv.get("model_config") or {}
    if not mc and "service" in pv and isinstance(pv["service"], dict):
        svc = pv["service"]
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


def apply_rule_plugins(param_values: dict[str, Any], backend: str, dsl_dir: Optional[str] = None) -> dict[str, Any]:
    base = str(Path(dsl_dir).resolve()) if dsl_dir else str(_RULES_DIR)
    rule_path = _load_rule_path(base, backend)
    if not rule_path:
        return param_values
    with open(rule_path, encoding="utf-8") as f:
        content = f.read().splitlines()

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
