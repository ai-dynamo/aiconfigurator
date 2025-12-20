# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Configuration mappings for AI generator.

This module provides the mapping engine for converting parameters
to backend-specific configurations using YAML mapping files and Jinja2 templates.
"""

import logging
import os
import shlex
from pathlib import Path
from typing import Any, Optional

import yaml
from jinja2 import Environment, FileSystemLoader, Undefined

from .rule_engine import apply_rule_plugins

_JINJA_ENV = Environment(trim_blocks=True, lstrip_blocks=True)
_TEMPLATE_ENV_CACHE: dict[str, Environment] = {}
logger = logging.getLogger(__name__)
_YAML_CACHE: dict[str, Any] = {}
_PARAM_KEYS_CACHE: dict[str, list[str]] = {}
_BASE_DIR = Path(__file__).resolve().parent
_CONFIG_DIR = (_BASE_DIR.parent / "config").resolve()
_TEMPLATE_ROOT = _CONFIG_DIR / "backend_templates"
_BACKEND_MAPPING_FILE = str((_CONFIG_DIR / "backend_config_mapping.yaml").resolve())


def render_backend_templates(
    param_values: dict[str, Any], backend: str, templates_dir: Optional[str] = None, version: Optional[str] = None
) -> dict[str, str]:
    """
    Render templates for a specific backend with version-specific template selection.

    Args:
        param_values: Dictionary of parameter values to use in template rendering
        backend: Backend name (e.g., 'trtllm', 'vllm', 'sglang')
        templates_dir: Directory containing backend-specific template directories
        version: Version string (e.g., '1.1.0rc5'). If None, uses default templates

    Returns:
        Dictionary mapping template names to rendered content
    """
    if templates_dir is None:
        templates_dir = str(_TEMPLATE_ROOT / backend)

    if not os.path.exists(templates_dir):
        raise FileNotFoundError(f"Templates directory not found: {templates_dir}")

    # Set up Jinja2 environment with FileSystemLoader
    env = _TEMPLATE_ENV_CACHE.get(templates_dir)
    if env is None:
        env = Environment(loader=FileSystemLoader(templates_dir), trim_blocks=True, lstrip_blocks=True)
        _TEMPLATE_ENV_CACHE[templates_dir] = env

    param_values = apply_rule_plugins(dict(param_values), backend)
    context = prepare_template_context(param_values, backend)
    # Assign backend-specific working_dir (removes need for input-driven path)
    backend_dirs = {
        "trtllm": "/workspace/components/backends/trtllm",
        "sglang": "/workspace/components/backends/sglang",
        "vllm": "/workspace/components/backends/vllm",
    }
    wd = backend_dirs.get(backend)
    if wd:
        # k8s.working_dir used in k8s_deploy templates, also expose top-level
        context.setdefault("k8s", {})
        context["k8s"]["working_dir"] = wd
        context["working_dir"] = wd

    # Determine generation mode by params presence
    params_obj = param_values.get("params", {})
    has_prefill = bool(params_obj.get("prefill"))
    has_decode = bool(params_obj.get("decode"))
    has_agg = bool(params_obj.get("agg"))
    generate_disagg = has_prefill and has_decode
    generate_agg = has_agg and not generate_disagg
    # Prefer disagg when both are present
    if generate_disagg:
        worker_plan = ["prefill", "decode"]
    elif generate_agg:
        worker_plan = ["agg"]
    else:
        # Fallback: prefer disagg if any prefill/decode provided, else agg
        worker_plan = ["prefill", "decode"] if (has_prefill or has_decode) else ["agg"]

    rendered_templates = {}

    # Find template files
    template_path = Path(templates_dir)

    # Resolve engine template (version-specific preferred). Some backends (e.g., vllm, sglang)
    # do not ship engine configs at all, so only warn when such templates actually exist.
    engine_template_file = None
    engine_template_candidates = list(template_path.glob("extra_engine_args*.yaml.j2"))
    has_engine_templates = bool(engine_template_candidates)
    if version and has_engine_templates:
        candidates = [p for p in engine_template_candidates if p.name == f"extra_engine_args.{version}.yaml.j2"]
        if candidates:
            engine_template_file = candidates[0]
        else:
            logger.warning(f"No version-specific engine template for {version}, using default")
    if engine_template_file is None and has_engine_templates:
        default_candidates = [p for p in engine_template_candidates if p.name == "extra_engine_args.yaml.j2"]
        if default_candidates:
            engine_template_file = default_candidates[0]
        # If no engine args template exists (e.g., sglang/vllm), proceed without it

    # Render engine templates per worker plan with worker-specific context
    mapping_data = load_yaml_mapping(_BACKEND_MAPPING_FILE)
    param_keys = get_param_keys(_BACKEND_MAPPING_FILE)

    def make_worker_context(
        base_ctx: dict[str, Any],
        worker: str,
        worker_param_keys: list[str],
        mapping_def: dict[str, Any],
    ) -> dict[str, Any]:
        wc = dict(base_ctx)

        def _remove_key_and_nested(key: str) -> None:
            if key in wc:
                wc.pop(key, None)
            if "." in key:
                parts = key.split(".")
                cursor = wc
                for p in parts[:-1]:
                    node = cursor.get(p)
                    if not isinstance(node, dict):
                        return
                    cursor = node
                if isinstance(cursor, dict):
                    cursor.pop(parts[-1], None)

        for k in worker_param_keys:
            wk = f"{worker}_{k}"
            if wk in base_ctx:
                wc[k] = base_ctx[wk]
            else:
                _remove_key_and_nested(k)

        # Promote worker-scoped dotted backend keys into nested dicts
        prefix = f"{worker}_"
        for bk, val in list(base_ctx.items()):
            if bk.startswith(prefix):
                name = bk[len(prefix) :]
                if "." in name:
                    parts = name.split(".")
                    cursor = wc
                    for p in parts[:-1]:
                        if p not in cursor or not isinstance(cursor[p], dict):
                            cursor[p] = {}
                        cursor = cursor[p]
                    cursor[parts[-1]] = val

        # Build backend dict with worker-scoped keys (including hyphenated names)
        backend_keys = []
        for entry in mapping_def.get("parameters", []):
            m = entry.get(backend)
            if isinstance(m, str):
                backend_keys.append(m)
            elif isinstance(m, dict):
                dest = m.get("key")
                if dest:
                    backend_keys.append(dest)
        wc.setdefault(backend, {})
        for bk, val in list(base_ctx.items()):
            if bk.startswith(prefix):
                name = bk[len(prefix) :]
                if name in backend_keys and "." not in name:
                    wc[backend][name] = val
        for bk in backend_keys:
            wk = f"{worker}_{bk}"
            if wk in base_ctx:
                wc[bk] = base_ctx[wk]
            else:
                _remove_key_and_nested(bk)
        return wc

    if engine_template_file is not None:
        try:
            eng_tmpl = env.get_template(engine_template_file.name)
            for worker in worker_plan:
                wc = make_worker_context(context, worker, param_keys, mapping_data)
                rendered = eng_tmpl.render(**wc)
                if worker == "agg":
                    out_name = "extra_engine_args_agg.yaml"
                elif worker == "prefill":
                    out_name = "extra_engine_args_prefill.yaml"
                else:
                    out_name = "extra_engine_args_decode.yaml"
                rendered_templates[out_name] = rendered
        except Exception as e:
            logger.warning(f"Failed to render engine template {engine_template_file.name}: {e}")

    # Inject inline engine args content into context for k8s template
    # These are used when k8s.k8s_engine_mode == 'inline'
    context["prefill_engine_args_inline"] = rendered_templates.get("extra_engine_args_prefill.yaml", "")
    context["decode_engine_args_inline"] = rendered_templates.get("extra_engine_args_decode.yaml", "")
    context["agg_engine_args_inline"] = rendered_templates.get("extra_engine_args_agg.yaml", "")

    # Resolve CLI args template (version-specific preferred)
    cli_template_file = None
    cli_template_candidates = list(template_path.glob("cli_args*.j2"))
    if version and cli_template_candidates:
        candidates = [p for p in cli_template_candidates if p.name == f"cli_args.{version}.j2"]
        if candidates:
            cli_template_file = candidates[0]
    if cli_template_file is None:
        default_candidates = [p for p in cli_template_candidates if p.name == "cli_args.j2"]
        if default_candidates:
            cli_template_file = default_candidates[0]

    # Compute CLI args per worker using template if present, else mapping fallback
    for worker in worker_plan:
        wc = make_worker_context(context, worker, param_keys, mapping_data)
        if cli_template_file is not None:
            try:
                cli_tmpl = env.get_template(cli_template_file.name)
                cli = cli_tmpl.render(**wc).strip()
            except Exception:
                cli = _format_cli_args(backend, wc)
        else:
            cli = _format_cli_args(backend, wc)
        cli_list = shlex.split(cli) if cli else []
        if worker == "prefill":
            context["prefill_cli_args"] = cli
            context["prefill_cli_args_list"] = cli_list
            rendered_templates["cli_args_prefill"] = cli
        elif worker == "decode":
            context["decode_cli_args"] = cli
            context["decode_cli_args_list"] = cli_list
            rendered_templates["cli_args_decode"] = cli
        else:
            context["agg_cli_args"] = cli
            context["agg_cli_args_list"] = cli_list
            rendered_templates["cli_args_agg"] = cli

    # Compute GPU counts per worker using rule outputs (minimal fallback)
    pv_params = param_values.get("params", {})
    prefill_gpu = int(pv_params.get("prefill", {}).get("gpus_per_worker") or 1)
    decode_gpu = int(pv_params.get("decode", {}).get("gpus_per_worker") or 1)
    agg_gpu = int(pv_params.get("agg", {}).get("gpus_per_worker") or 1)

    context["prefill_gpu"] = prefill_gpu
    context["decode_gpu"] = decode_gpu
    context["agg_gpu"] = agg_gpu

    # Render auxiliary templates (k8s deploy and run script)
    # k8s deploy: single file
    k8s_aux = template_path / "k8s_deploy.yaml.j2"
    if k8s_aux.exists():
        try:
            tmpl = env.get_template("k8s_deploy.yaml.j2")
            rendered = tmpl.render(**context)
            rendered_templates["k8s_deploy.yaml"] = rendered
        except Exception as e:
            logger.warning(f"Failed to render template k8s_deploy.yaml.j2: {e}")

    # run scripts: generate per-node scripts when disagg; single when agg
    run_aux = template_path / "run.sh.j2"
    if run_aux.exists():
        try:
            tmpl = env.get_template("run.sh.j2")

            # Determine mode
            mode = context.get("k8s", {}).get("mode", "disagg")

            if mode == "agg":
                node_ctx = dict(context)
                # Ensure nested service dict exists and set include_frontend
                svc = dict(node_ctx.get("service", {}))
                svc["include_frontend"] = True
                node_ctx["service"] = svc
                rendered = tmpl.render(**node_ctx)
                rendered_templates["run_0.sh"] = rendered
            else:
                # Use GPU counts injected earlier from rule outputs
                prefill_gpu = int(context.get("prefill_gpu", 1))
                decode_gpu = int(context.get("decode_gpu", 1))

                prefill_workers = int(context.get("prefill_workers", 1))
                decode_workers = int(context.get("decode_workers", 1))

                # Simple greedy allocation (8 GPUs per node default)
                def _allocate_disagg_nodes(p_worker: int, p_gpu: int, d_worker: int, d_gpu: int, gpu_per_node: int = 8):
                    nodes = []
                    for _ in range(p_worker):
                        placed = False
                        for n in nodes:
                            if n["used"] + p_gpu <= gpu_per_node:
                                n["p_worker"] += 1
                                n["used"] += p_gpu
                                placed = True
                                break
                        if not placed:
                            nodes.append({"p_worker": 1, "d_worker": 0, "used": p_gpu})
                    for _ in range(d_worker):
                        placed = False
                        for n in nodes:
                            if n["used"] + d_gpu <= gpu_per_node:
                                n["d_worker"] += 1
                                n["used"] += d_gpu
                                placed = True
                                break
                        if not placed:
                            nodes.append({"p_worker": 0, "d_worker": 1, "used": d_gpu})
                    return [{"p_worker": n["p_worker"], "d_worker": n["d_worker"]} for n in nodes]

                plan = _allocate_disagg_nodes(prefill_workers, prefill_gpu, decode_workers, decode_gpu)

                for idx, cnt in enumerate(plan):
                    node_ctx = dict(context)
                    svc = dict(node_ctx.get("service", {}))
                    svc["include_frontend"] = idx == 0
                    node_ctx["service"] = svc
                    node_ctx["prefill_gpu"] = prefill_gpu
                    node_ctx["decode_gpu"] = decode_gpu
                    node_ctx["prefill_workers"] = int(cnt.get("p_worker", 0))
                    node_ctx["decode_workers"] = int(cnt.get("d_worker", 0))
                    node_ctx["decode_gpu_offset"] = int(cnt.get("p_worker", 0)) * prefill_gpu
                    rendered = tmpl.render(**node_ctx)
                    rendered_templates[f"run_{idx}.sh"] = rendered
        except Exception as e:
            logger.warning(f"Failed to render template run.sh.j2: {e}")

    return rendered_templates


def prepare_template_context(param_values: dict[str, Any], backend: str) -> dict[str, Any]:
    """
    Prepare the context dictionary for template rendering.

    This function transforms the parameter values into the format expected by the original templates,
    following the backend_config_mapping.yaml structure exactly.

    Args:
        param_values: Dictionary of parameter values
        backend: Backend name

    Returns:
        Context dictionary for template rendering
    """
    context = {}

    # Extract ModelConfig (is_moe, nextn, etc.)
    model_config = param_values.get("ModelConfig", {})
    if model_config.get("is_moe"):
        context["is_moe"] = model_config["is_moe"]

    # Extract unified service configuration
    service_config = param_values.get("service", {})
    context["model_name"] = service_config.get("model_name") or service_config.get("served_model_name", "")
    context["model_path"] = service_config.get("model_path")
    context["served_model_name"] = service_config.get("served_model_name")
    context["service"] = dict(service_config)

    # Extract K8s configuration
    k8s_config = param_values.get("k8s", {})
    context["name_prefix"] = k8s_config.get("name_prefix")
    context["mode"] = k8s_config.get("mode")
    context["k8s_namespace"] = k8s_config.get("k8s_namespace")
    context["k8s_image"] = k8s_config.get("k8s_image")
    context["k8s_image_pull_secret"] = k8s_config.get("k8s_image_pull_secret")
    context["working_dir"] = k8s_config.get("working_dir")
    context["k8s_engine_mode"] = k8s_config.get("k8s_engine_mode")
    context["k8s_model_cache"] = k8s_config.get("k8s_model_cache")
    enable_router = bool(k8s_config.get("enable_router"))
    context["router_mode"] = "kv" if enable_router else ""
    context["is_kv"] = enable_router
    context["enable_router"] = enable_router
    name_suffix = "agg" if context["mode"] == "agg" else "disagg"
    router_suffix = "-router" if enable_router else ""
    full_name = f"{context['name_prefix']}-{name_suffix}{router_suffix}"
    context["name"] = k8s_config.get("name") or full_name
    k8s_copy = dict(k8s_config)
    k8s_copy["router_mode"] = context["router_mode"]
    k8s_copy["is_kv"] = enable_router
    k8s_copy["enable_router"] = enable_router
    context["k8s"] = k8s_copy

    # Runtime is part of service
    context["head_node_ip"] = service_config.get("head_node_ip")
    context["port"] = service_config.get("port")
    context["include_frontend"] = service_config.get("include_frontend")

    # Extract worker parameters
    worker_params = param_values.get("params", {})
    context["prefill_params"] = worker_params.get("prefill", {})
    context["decode_params"] = worker_params.get("decode", {})
    context["agg_params"] = worker_params.get("agg", {})

    # Extract worker counts
    workers = param_values.get("workers", {})
    context["prefill_workers"] = workers.get("prefill_workers", 1)
    context["decode_workers"] = workers.get("decode_workers", 1)
    context["agg_workers"] = workers.get("agg_workers", 1)
    context["prefill_gpus_per_worker"] = workers.get("prefill_gpus_per_worker")
    context["decode_gpus_per_worker"] = workers.get("decode_gpus_per_worker")
    context["agg_gpus_per_worker"] = workers.get("agg_gpus_per_worker")

    fr = 1 if (context.get("include_frontend") is True) else 0
    context["frontend_replicas"] = fr

    # Load backend_config_mapping.yaml to understand parameter mappings
    mapping_data = load_yaml_mapping(_BACKEND_MAPPING_FILE)

    # Create a mapping from parameter keys to backend-specific keys and template variables
    param_to_backend = {}
    param_to_template_var = {}

    # Build mapping from backend_config_mapping.yaml
    for entry in mapping_data.get("parameters", []):
        param_key = entry.get("param_key")
        backend_mapping = entry.get(backend)
        if backend_mapping is not None and backend_mapping != "null":
            param_to_backend[param_key] = backend_mapping

            # Map to template variable names (based on template analysis)
            template_var_mapping = {
                "tensor_parallel_size": "tp",
                "pipeline_parallel_size": "pp",
                "data_parallel_size": "dp",
                "max_batch_size": "max_batch_size",
                "max_num_tokens": "max_num_tokens",
                "max_seq_len": "max_seq_len",
                "kv_cache_dtype": "kv_cache_dtype",
                "tokens_per_block": "tokens_per_block",
                "enable_chunked_prefill": "enable_chunked_prefill",
                "cuda_graph_enable_padding": "cuda_graph_enable_padding",
                "disable_prefix_cache": "disable_prefix_cache",
            }

            if param_key in template_var_mapping:
                param_to_template_var[param_key] = template_var_mapping[param_key]

    # Apply parameter mapping for each worker type
    for worker_type in ["prefill", "decode", "agg"]:
        worker_config = worker_params.get(worker_type, {})

        for param_key, value in worker_config.items():
            # Always expose param_key variables
            context[param_key] = value
            context[f"{worker_type}_{param_key}"] = value
            if param_key in param_to_backend:
                backend_mapping = param_to_backend[param_key]

                # Handle different types of backend mappings
                if isinstance(backend_mapping, str):
                    # Simple string mapping (e.g., "tensor_parallel_size" -> "tensor_parallel_size")
                    context[backend_mapping] = value
                    # Also add with worker prefix for disambiguation
                    context[f"{worker_type}_{backend_mapping}"] = value

                    # Map to template variable if available
                    if param_key in param_to_template_var:
                        template_var = param_to_template_var[param_key]
                        context[template_var] = value
                        context[f"{worker_type}_{template_var}"] = value

                elif isinstance(backend_mapping, dict):
                    # Complex mapping with key and value expressions
                    dest_key = backend_mapping.get("key")
                    value_expr = backend_mapping.get("value")

                    if dest_key:
                        # Evaluate the value expression if provided
                        if value_expr:
                            # Create a simple context for evaluation
                            eval_context = {param_key: value}
                            evaluated_value = evaluate_expression(value_expr, eval_context)
                            context[dest_key] = evaluated_value
                            context[f"{worker_type}_{dest_key}"] = evaluated_value
                        else:
                            context[dest_key] = value
                            context[f"{worker_type}_{dest_key}"] = value

    # Add individual parameter shortcuts for easy template access
    # Expose worker-scoped parameters with role prefixes only
    for worker_type in ["prefill", "decode", "agg"]:
        worker_config = worker_params.get(worker_type, {})
        for key, value in worker_config.items():
            context[f"{worker_type}_{key}"] = value

    # No dynamo_config in new templates

    # Add engine args paths for templates
    context["prefill_engine_args"] = "/workspace/engine_configs/prefill_config.yaml"
    context["decode_engine_args"] = "/workspace/engine_configs/decode_config.yaml"
    context["agg_engine_args"] = "/workspace/engine_configs/agg_config.yaml"

    # Initialize nested backend config dicts for template access
    for nested in [
        "kv_cache_config",
        "cache_transceiver_config",
        "cuda_graph_config",
        "build_config",
        "speculative_config",
        "moe_config",
    ]:
        if nested not in context or not isinstance(context.get(nested), dict):
            context[nested] = {}

    return context


def _cast_literal(s: str) -> Any:
    """
    Lightweight casting via YAML loader to get bool/int/float.

    Args:
        s: String value to cast

    Returns:
        Casted value (bool, int, float, or original string)
    """
    try:
        return yaml.safe_load(s)
    except Exception:
        return s


def evaluate_expression(expr: Any, context: dict[str, Any]) -> Any:
    """
    Evaluate Jinja2 expressions with the provided context.

    Supports conditionals, logical operators, arithmetic, and identity lookup.

    Args:
        expr: Expression to evaluate (string or other type)
        context: Context dictionary for variable resolution

    Returns:
        Evaluated expression result
    """
    if expr is None:
        return None
    if not isinstance(expr, str):
        return expr
    s = expr.strip()
    try:
        func = _JINJA_ENV.compile_expression(s)
        result = func(**context)
    except Exception:
        result = context.get(s, _cast_literal(s))
    if isinstance(result, Undefined):
        return None
    return result


def load_yaml_mapping(yaml_path: str) -> dict[str, Any]:
    """
    Load YAML mapping file.

    Args:
        yaml_path: Path to YAML file

    Returns:
        Parsed YAML content as dictionary
    """
    path = os.path.abspath(str(yaml_path))
    cached = _YAML_CACHE.get(path)
    if cached is not None:
        return cached
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    _YAML_CACHE[path] = data
    return data


def render_parameters(
    param_values: dict[str, Any],
    yaml_path: Optional[str] = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """
    Render parameter mappings to concrete key/value dicts per framework.

    Behavior:
    - String shorthand (e.g. vllm: some-flag) includes only when the input value is not None.
    - Dict form retention is controlled solely by presence of "default":
      { "key": <dest_key>, "value": <jinja_expr>, ["default": <jinja_expr>] }
      * If value evaluates to None and default is present, include dest_key
        with the evaluated default (which may be None).
      * If value evaluates to a non-None value, include it.
      * If value is None and no default is present, omit.
      * Inline form {dest_key: <jinja_expr>} behaves like shorthand (omit when None).

    Args:
        param_values: Dictionary of parameter values
        yaml_path: Optional path to YAML mapping file

    Returns:
        Nested dictionary structure: {param_key: {framework: {key: value}}}
    """
    yaml_file = _BACKEND_MAPPING_FILE if yaml_path is None else str(yaml_path)
    data = load_yaml_mapping(yaml_file)

    out: dict[str, dict[str, dict[str, Any]]] = {}
    parameters = data.get("parameters", [])

    for entry in parameters:
        param_key = entry.get("param_key")
        if not param_key:
            continue

        # Determine frameworks dynamically from entry keys
        framework_keys = [k for k in entry if k not in ("param_key",)]
        result: dict[str, dict[str, Any]] = {}
        for fw in framework_keys:
            mapping = entry.get(fw)

            # Missing or null -> skip
            if mapping is None:
                continue

            # Empty object -> skip
            if isinstance(mapping, dict) and len(mapping) == 0:
                continue

            # String shorthand: interpret as key name, value := param_values[param_key]
            if isinstance(mapping, str):
                v = param_values.get(param_key)
                if v is not None:
                    result.setdefault(fw, {})[mapping] = v
                continue

            # Dict with explicit key/value and optional default-only retention
            if isinstance(mapping, dict) and "key" in mapping and "value" in mapping:
                k = mapping.get("key")
                v_expr = mapping.get("value")
                default_expr = mapping.get("default")

                v = evaluate_expression(v_expr, param_values)
                has_default = "default" in mapping
                if v is None and has_default:
                    # Presence of default implies retention, even if evaluated default is None
                    v_default = evaluate_expression(default_expr, param_values)
                    result.setdefault(fw, {})[k] = v_default
                elif v is not None:
                    result.setdefault(fw, {})[k] = v
                # else: omit when None and no default present
                continue

            # Dict as {key: expr} mapping (fallback)
            if isinstance(mapping, dict):
                rendered: dict[str, Any] = {}
                for k, v_expr in mapping.items():
                    v = evaluate_expression(v_expr, param_values)
                    if v is not None:
                        rendered[k] = v
                if rendered:
                    result[fw] = rendered
                continue

            # Otherwise, unsupported type -> skip
            continue

        # Only record this param_key if any backend has values
        if result:
            out[param_key] = result

    return out


def render_backend_parameters(
    param_values: dict[str, Any],
    backend: str,
    yaml_path: Optional[str] = None,
) -> dict[str, dict[str, Any]]:
    """
    Render parameter mappings only for a specific backend.

    Args:
        param_values: Dictionary of parameter values
        backend: Target backend name
        yaml_path: Optional path to YAML mapping file

    Returns:
        Dictionary structure: {param_key: {backend_key: value}} (pruned for None)
    """
    all_rendered = render_parameters(param_values, yaml_path=yaml_path)
    out: dict[str, dict[str, Any]] = {}
    for param_key, fw_map in all_rendered.items():
        backend_dict = fw_map.get(backend)
        if not backend_dict:
            continue
        out[param_key] = backend_dict
    return out


def get_param_keys(yaml_path: str) -> list[str]:
    """
    Get parameter keys from YAML mapping file.

    Args:
        yaml_path: Path to YAML mapping file

    Returns:
        List of parameter keys
    """
    path = os.path.abspath(str(yaml_path))
    cached = _PARAM_KEYS_CACHE.get(path)
    if cached is not None:
        return cached
    data = load_yaml_mapping(path)
    keys = [e["param_key"] for e in data.get("parameters", []) if e.get("param_key")]
    _PARAM_KEYS_CACHE[path] = keys
    return keys


def _format_cli_args(backend: str, worker_ctx: dict[str, Any]) -> str:
    rendered = render_backend_parameters(worker_ctx, backend, yaml_path=_BACKEND_MAPPING_FILE)
    parts: list[str] = []
    for _pk, kv in rendered.items():
        for flag, val in kv.items():
            if isinstance(val, bool):
                if val:
                    parts.append(f"--{flag}")
                continue
            if isinstance(val, (list, tuple)):
                v = ",".join(str(x) for x in val)
                parts.append(f'--{flag} "{v}"')
                continue
            parts.append(f'--{flag} "{val}"')
    return " ".join(parts)
