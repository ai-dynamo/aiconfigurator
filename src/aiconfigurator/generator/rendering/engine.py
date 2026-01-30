# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import shlex
from pathlib import Path
from typing import Any, Optional

import yaml
from jinja2 import Environment, FileSystemLoader, Undefined

from aiconfigurator.generator.rendering.rule_engine import apply_rule_plugins

logger = logging.getLogger(__name__)

_BASE_DIR = Path(__file__).resolve().parent
_CONFIG_DIR = (_BASE_DIR.parent / "config").resolve()
_TEMPLATE_ROOT = _CONFIG_DIR / "backend_templates"
_BACKEND_MAPPING_FILE = _CONFIG_DIR / "backend_config_mapping.yaml"

_TEMPLATE_ENV_CACHE: dict[str, Environment] = {}
_JINJA_ENV = Environment()
_YAML_CACHE: dict[str, dict[str, Any]] = {}


def render_backend_templates(
    param_values: dict[str, Any],
    role_backends: dict[str, str],
    role_versions: Optional[dict[str, str]] = None,
    templates_dir: Optional[str] = None,
) -> dict[str, str]:
    """
    Render templates based on role-specific backends and versions.

    Args:
        param_values: Dictionary of parameter values
        role_backends: Mapping of role (prefill/decode/agg) to backend name
        role_versions: Optional mapping of role to version string
        templates_dir: Optional override for the base templates directory.
                      If None, derived from the primary backend (decode or agg).

    Returns:
        Dictionary mapping template names to rendered content
    """
    role_versions = role_versions or {}

    # Determine the primary backend to use for the base deployment templates (K8s/Slurm)
    # We prefer 'decode' for disagg as it's the more constrained side, or 'agg' for single-node.
    primary_backend = role_backends.get("decode") or role_backends.get("agg") or role_backends.get("prefill")
    if not primary_backend:
        raise ValueError("role_backends must contain at least one backend (agg, decode, or prefill)")

    primary_version = role_versions.get("decode") or role_versions.get("agg") or role_versions.get("prefill")

    # Check if we have heterogeneous backends (different backends for different roles)
    unique_backends = set(role_backends.values())
    is_heterogeneous = len(unique_backends) > 1

    if templates_dir is None:
        if is_heterogeneous:
            # Use the 'any' template directory for heterogeneous deployments
            templates_dir = str(_TEMPLATE_ROOT / "any")
        else:
            templates_dir = str(_TEMPLATE_ROOT / primary_backend)

    if not os.path.exists(templates_dir):
        raise FileNotFoundError(f"Templates directory not found: {templates_dir}")

    # Set up Jinja2 environment with FileSystemLoader
    # Use multiple search paths to enable cross-directory imports:
    # 1. Primary templates directory (backend-specific or 'any')
    # 2. _common/ for shared base templates
    # 3. _workers/ for backend-specific worker macros
    # 4. Template root for relative imports like '_workers/sglang.j2'
    env = _TEMPLATE_ENV_CACHE.get(templates_dir)
    if env is None:
        search_paths = [
            templates_dir,
            str(_TEMPLATE_ROOT / "_common"),
            str(_TEMPLATE_ROOT / "_workers"),
            str(_TEMPLATE_ROOT),
        ]
        env = Environment(
            loader=FileSystemLoader(search_paths),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        _TEMPLATE_ENV_CACHE[templates_dir] = env

    param_values = apply_rule_plugins(dict(param_values), primary_backend)
    context = prepare_template_context(param_values, primary_backend)

    # Inject role information into context
    for role, rb in role_backends.items():
        context[f"{role}_backend"] = rb
    for role, rv in role_versions.items():
        context[f"{role}_backend_version"] = rv

    # Inject role-specific K8s images for heterogeneous deployments
    if is_heterogeneous and "K8sConfig" in context:
        for role, rb in role_backends.items():
            rv = role_versions.get(role, "0.8.0")  # Default version if not specified
            # Determine the appropriate image for this backend
            backend_image_map = {
                "trtllm": f"nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:{rv}",
                "sglang": f"nvcr.io/nvidia/ai-dynamo/sglang-runtime:{rv}",
                "vllm": f"nvcr.io/nvidia/ai-dynamo/vllm-runtime:{rv}",
            }
            role_image = backend_image_map.get(rb)
            if role_image:
                context[f"{role}_k8s_image"] = role_image
                # Also add to K8sConfig dict for generator_config.yaml output
                if "K8sConfig" in param_values:
                    param_values["K8sConfig"][f"{role}_k8s_image"] = role_image

    # Also set top-level backend for backward compatibility in templates
    context["backend"] = primary_backend
    if primary_version:
        context["backend_version"] = primary_version

    # Assign backend-specific working_dir (removes need for input-driven path)
    backend_dirs = {
        "trtllm": "/workspace/components/backends/trtllm",
        "sglang": "/workspace/components/backends/sglang",
        "vllm": "/workspace/components/backends/vllm",
    }
    wd = backend_dirs.get(primary_backend)
    if wd:
        # K8sConfig.working_dir used in k8s_deploy templates, also expose top-level
        context.setdefault("K8sConfig", {})
        context["K8sConfig"]["working_dir"] = wd
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

    def resolve_template(
        target_backend: str, target_version: Optional[str], pattern: str
    ) -> tuple[Optional[Path], Any]:
        """Resolve a template file and its environment for a specific backend and version."""
        target_dir = str(_TEMPLATE_ROOT / target_backend)
        target_env = _TEMPLATE_ENV_CACHE.get(target_dir)
        if target_env is None:
            if not os.path.exists(target_dir):
                return None, None
            target_env = Environment(loader=FileSystemLoader(target_dir), trim_blocks=True, lstrip_blocks=True)
            _TEMPLATE_ENV_CACHE[target_dir] = target_env

        tp = Path(target_dir)
        if "extra_engine_args" in pattern:
            base_name = "extra_engine_args"
            ext = ".yaml.j2"
        elif "cli_args" in pattern:
            base_name = "cli_args"
            ext = ".j2"
        else:
            exact = tp / pattern
            return (exact, target_env) if exact.exists() else (None, None)

        if target_version:
            v_match = tp / f"{base_name}.{target_version}{ext}"
            if v_match.exists():
                return v_match, target_env

        default = tp / f"{base_name}{ext}"
        return (default, target_env) if default.exists() else (None, None)

    # Render engine and CLI templates per worker plan with worker-specific context
    mapping_data_cache = {}

    def get_mapping_data(target_backend: str):
        if target_backend not in mapping_data_cache:
            mapping_data_cache[target_backend] = load_yaml_mapping(_BACKEND_MAPPING_FILE)
        return mapping_data_cache[target_backend]

    param_keys = get_param_keys(_BACKEND_MAPPING_FILE)

    for worker in worker_plan:
        w_backend = role_backends.get(worker, primary_backend)
        w_version = role_versions.get(worker, primary_version)

        w_mapping_data = get_mapping_data(w_backend)

        # 1. Resolve and render engine template for this worker
        engine_pattern = "extra_engine_args*.yaml.j2"
        w_engine_tmpl_file, w_env = resolve_template(w_backend, w_version, engine_pattern)

        if w_engine_tmpl_file:
            try:
                eng_tmpl = w_env.get_template(w_engine_tmpl_file.name)
                wc = make_worker_context(context, worker, param_keys, w_mapping_data, w_backend)
                rendered = eng_tmpl.render(**wc)
                if worker == "agg":
                    out_name = "extra_engine_args_agg.yaml"
                elif worker == "prefill":
                    out_name = "extra_engine_args_prefill.yaml"
                else:
                    out_name = "extra_engine_args_decode.yaml"
                rendered_templates[out_name] = rendered
            except Exception as e:
                logger.warning(f"Failed to render engine template {w_engine_tmpl_file.name} for {worker}: {e}")

        # 2. Resolve and render CLI args template for this worker
        cli_pattern = "cli_args*.j2"
        w_cli_tmpl_file, w_env = resolve_template(w_backend, w_version, cli_pattern)

        wc = make_worker_context(context, worker, param_keys, w_mapping_data, w_backend)
        cli = ""
        if w_cli_tmpl_file:
            try:
                cli_tmpl = w_env.get_template(w_cli_tmpl_file.name)
                cli = cli_tmpl.render(**wc).strip()
            except Exception:
                cli = _format_cli_args(w_backend, wc)
        else:
            cli = _format_cli_args(w_backend, wc)

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

    # Inject inline engine args content into context for k8s template
    # These are used when K8sConfig.k8s_engine_mode == 'inline'
    context["prefill_engine_args_inline"] = rendered_templates.get("extra_engine_args_prefill.yaml", "")
    context["decode_engine_args_inline"] = rendered_templates.get("extra_engine_args_decode.yaml", "")
    context["agg_engine_args_inline"] = rendered_templates.get("extra_engine_args_agg.yaml", "")

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
            mode = context.get("DynConfig", {}).get("mode", "disagg")

            if mode == "agg":
                # Use GPU counts injected earlier from rule outputs
                agg_gpu = int(context.get("agg_gpu", 1))
                agg_workers = int(context.get("agg_workers", 1))
                node_cfg = context.get("NodeConfig", {})
                num_gpus_per_node = int(node_cfg.get("num_gpus_per_node", 8))

                # Simple greedy allocation
                def _allocate_agg_nodes(workers: int, gpu: int, gpu_per_node: int):
                    nodes = []
                    for _ in range(workers):
                        placed = False
                        for n in nodes:
                            if n["used"] + gpu <= gpu_per_node:
                                n["workers"] += 1
                                n["used"] += gpu
                                placed = True
                                break
                        if not placed:
                            nodes.append({"workers": 1, "used": gpu})
                    return nodes

                plan = _allocate_agg_nodes(agg_workers, agg_gpu, num_gpus_per_node)

                for idx, cnt in enumerate(plan):
                    node_ctx = dict(context)
                    # Ensure nested ServiceConfig dict exists and set include_frontend
                    svc = dict(node_ctx.get("ServiceConfig", {}))
                    svc["include_frontend"] = idx == 0
                    node_ctx["ServiceConfig"] = svc
                    node_ctx["agg_gpu"] = agg_gpu
                    node_ctx["agg_workers"] = int(cnt["workers"])
                    node_ctx["agg_gpu_offset"] = 0
                    rendered = tmpl.render(**node_ctx)
                    rendered_templates[f"run_{idx}.sh"] = rendered
            else:
                # Use GPU counts injected earlier from rule outputs
                prefill_gpu = int(context.get("prefill_gpu", 1))
                decode_gpu = int(context.get("decode_gpu", 1))

                prefill_workers = int(context.get("prefill_workers", 1))
                decode_workers = int(context.get("decode_workers", 1))
                node_cfg = context.get("NodeConfig", {})
                num_gpus_per_node = int(node_cfg.get("num_gpus_per_node", 8))

                # Simple greedy allocation
                def _allocate_disagg_nodes(p_worker: int, p_gpu: int, d_worker: int, d_gpu: int, gpu_per_node: int):
                    nodes = []
                    # Place prefill workers
                    for _ in range(p_worker):
                        placed = False
                        for n in nodes:
                            if n["used"] + p_gpu <= gpu_per_node:
                                n["p_workers"] += 1
                                n["used"] += p_gpu
                                placed = True
                                break
                        if not placed:
                            nodes.append({"p_workers": 1, "d_workers": 0, "used": p_gpu})
                    # Place decode workers
                    for _ in range(d_worker):
                        placed = False
                        for n in nodes:
                            if n["used"] + d_gpu <= gpu_per_node:
                                n["d_workers"] += 1
                                n["used"] += d_gpu
                                placed = True
                                break
                        if not placed:
                            nodes.append({"p_workers": 0, "d_workers": 1, "used": d_gpu})
                    return nodes

                plan = _allocate_disagg_nodes(
                    prefill_workers, prefill_gpu, decode_workers, decode_gpu, num_gpus_per_node
                )

                for idx, cnt in enumerate(plan):
                    node_ctx = dict(context)
                    # Ensure nested ServiceConfig dict exists and set include_frontend
                    svc = dict(node_ctx.get("ServiceConfig", {}))
                    svc["include_frontend"] = idx == 0
                    node_ctx["ServiceConfig"] = svc
                    node_ctx["prefill_gpu"] = prefill_gpu
                    node_ctx["decode_gpu"] = decode_gpu
                    node_ctx["prefill_workers"] = int(cnt["p_workers"])
                    node_ctx["decode_workers"] = int(cnt["d_workers"])
                    # Use a fixed offset for simplicity; actual multi-process placement might vary
                    node_ctx["decode_gpu_offset"] = int(cnt["p_workers"]) * prefill_gpu
                    rendered = tmpl.render(**node_ctx)
                    rendered_templates[f"run_{idx}.sh"] = rendered
        except Exception as e:
            logger.warning(f"Failed to render template run.sh.j2: {e}")

    return rendered_templates


def make_worker_context(
    base_ctx: dict[str, Any],
    worker: str,
    worker_param_keys: list[str],
    mapping_def: dict[str, Any],
    target_backend: Optional[str] = None,
) -> dict[str, Any]:
    wc = dict(base_ctx)
    # Use primary backend from context if target_backend not provided
    target_backend = target_backend or base_ctx.get("backend")

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
        m = entry.get(target_backend)
        if isinstance(m, str):
            backend_keys.append(m)
        elif isinstance(m, dict):
            dest = m.get("key")
            if dest:
                backend_keys.append(dest)
    wc.setdefault(target_backend, {})
    for bk, val in list(base_ctx.items()):
        if bk.startswith(prefix):
            name = bk[len(prefix) :]
            if name in backend_keys and "." not in name:
                wc[target_backend][name] = val
    for bk in backend_keys:
        wk = f"{worker}_{bk}"
        if wk in base_ctx:
            wc[bk] = base_ctx[wk]
        else:
            _remove_key_and_nested(bk)
    return wc


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
    service_config = param_values.get("ServiceConfig", {})
    context["model_path"] = service_config.get("model_path") or service_config.get("served_model_path", "")
    context["served_model_path"] = service_config.get("served_model_path")
    context["ServiceConfig"] = dict(service_config)

    # Extract K8s configuration
    k8s_config = param_values.get("K8sConfig", {})
    context["name_prefix"] = k8s_config.get("name_prefix")
    context["k8s_namespace"] = k8s_config.get("k8s_namespace")
    context["k8s_image"] = k8s_config.get("k8s_image")
    context["k8s_image_pull_secret"] = k8s_config.get("k8s_image_pull_secret")
    context["working_dir"] = k8s_config.get("working_dir")
    context["k8s_engine_mode"] = k8s_config.get("k8s_engine_mode")
    context["k8s_model_cache"] = k8s_config.get("k8s_model_cache")
    context["k8s_hf_home"] = k8s_config.get("k8s_hf_home")
    # Extract role-specific K8s images if present (for heterogeneous deployments)
    if "prefill_k8s_image" in k8s_config:
        context["prefill_k8s_image"] = k8s_config["prefill_k8s_image"]
    if "decode_k8s_image" in k8s_config:
        context["decode_k8s_image"] = k8s_config["decode_k8s_image"]
    if "agg_k8s_image" in k8s_config:
        context["agg_k8s_image"] = k8s_config["agg_k8s_image"]

    # Extract DynConfig for mode/router decisions
    dyn_config = param_values.get("DynConfig", {})
    if isinstance(dyn_config, dict):
        context["DynConfig"] = dyn_config
    mode_value = dyn_config.get("mode") if isinstance(dyn_config, dict) else None
    mode_value = mode_value or "disagg"
    enable_router = bool(dyn_config.get("enable_router")) if isinstance(dyn_config, dict) else False
    name_suffix = "agg" if mode_value == "agg" else "disagg"
    router_suffix = "-router" if enable_router else ""
    full_name = f"{context.get('name_prefix', 'dynamo')}-{name_suffix}{router_suffix}"
    context["name"] = k8s_config.get("name") or full_name
    context["K8sConfig"] = dict(k8s_config)

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
    workers = param_values.get("WorkerConfig", {})
    context["prefill_workers"] = workers.get("prefill_workers", 1)
    context["decode_workers"] = workers.get("decode_workers", 1)
    context["agg_workers"] = workers.get("agg_workers", 1)
    context["prefill_gpus_per_worker"] = workers.get("prefill_gpus_per_worker")
    context["decode_gpus_per_worker"] = workers.get("decode_gpus_per_worker")
    context["agg_gpus_per_worker"] = workers.get("agg_gpus_per_worker")
    context["prefill_gpu"] = context["prefill_gpus_per_worker"]
    context["decode_gpu"] = context["decode_gpus_per_worker"]
    context["agg_gpu"] = context["agg_gpus_per_worker"]

    fr = 1 if (context.get("include_frontend") is True) else 0
    context["frontend_replicas"] = fr

    node_config = param_values.get("NodeConfig", {})
    if isinstance(node_config, dict):
        context["NodeConfig"] = dict(node_config)

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
                "gemm_quant_mode": "gemm_quant",
                "moe_quant_mode": "moe_quant",
                "kvcache_quant_mode": "kv_cache_quant",
                "fmha_quant_mode": "fmha_quant",
                "comm_quant_mode": "comm_quant",
            }
            if param_key in template_var_mapping:
                param_to_template_var[param_key] = template_var_mapping[param_key]

    # Map worker parameters to their specific context keys
    for worker_type in ["prefill", "decode", "agg"]:
        worker_config = worker_params.get(worker_type, {})
        for param_key, value in worker_config.items():
            if param_key in param_to_backend:
                dest_key = param_to_backend[param_key]
                if isinstance(dest_key, dict):
                    # Complex mapping with nested dictionary
                    dest_path = dest_key.get("key")
                    value_expr = dest_key.get("value")
                    if dest_path:
                        if value_expr:
                            # Evaluate Jinja2-style expression
                            eval_context = {param_key: value}
                            evaluated_value = evaluate_expression(value_expr, eval_context)
                            _set_by_path(context, f"{worker_type}_{dest_path}", evaluated_value)
                            _set_by_path(context, dest_path, evaluated_value)
                        else:
                            _set_by_path(context, f"{worker_type}_{dest_path}", value)
                            _set_by_path(context, dest_path, value)
                elif isinstance(dest_key, str):
                    # Simple key mapping
                    if "." in dest_key:
                        _set_by_path(context, f"{worker_type}_{dest_key}", value)
                        _set_by_path(context, dest_key, value)
                    else:
                        # For top-level keys, also check expression mapping if exists in mapping_data
                        # This part handles expressions defined as strings in the mapping file
                        entry = next(
                            (e for e in mapping_data.get("parameters", []) if e.get("param_key") == param_key), None
                        )
                        mapping_def = entry.get(backend) if entry else None
                        if isinstance(mapping_def, dict) and mapping_def.get("value"):
                            value_expr = mapping_def["value"]
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


def get_param_keys(yaml_path: str) -> list[str]:
    """
    Get list of all supported parameter keys from mapping file.

    Args:
        yaml_path: Path to mapping YAML file

    Returns:
        List of parameter keys
    """
    data = load_yaml_mapping(yaml_path)
    return [entry.get("param_key") for entry in data.get("parameters", []) if entry.get("param_key")]


def _format_cli_args(backend: str, context: dict[str, Any]) -> str:
    """
    Fallback CLI arg formatter when template is missing.
    Uses backend_config_mapping.yaml to determine keys.
    """
    mapping_data = load_yaml_mapping(_BACKEND_MAPPING_FILE)
    args = []
    for entry in mapping_data.get("parameters", []):
        param_key = entry.get("param_key")
        backend_mapping = entry.get(backend)
        if not backend_mapping:
            continue

        # Get key name from mapping
        if isinstance(backend_mapping, str):
            arg_name = backend_mapping
        elif isinstance(backend_mapping, dict):
            arg_name = backend_mapping.get("key")
        else:
            continue

        if not arg_name or "." in arg_name:
            continue

        # Check context for value (prioritize role-scoped)
        val = context.get(param_key)
        if val is None or val is False:
            continue

        # Format based on value type
        if val is True:
            args.append(f"--{arg_name.replace('_', '-')}")
        else:
            args.append(f"--{arg_name.replace('_', '-')} {val}")

    return " ".join(args)


def _set_by_path(dst: dict[str, Any], path: str, value: Any) -> None:
    """Set value in nested dict using dot notation path."""
    parts = path.split(".")
    cur = dst
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def render_backend_parameters(
    params: dict[str, Any], backend: str, yaml_path: Optional[str] = None
) -> dict[str, dict[str, Any]]:
    """
    Transform unified parameters into backend-specific configuration for all roles.

    Args:
        params: Unified parameters dictionary
        backend: Target backend name
        yaml_path: Optional path to mapping YAML file

    Returns:
        Dictionary mapping roles to backend-specific configurations
    """
    if yaml_path is None:
        yaml_path = str(_BACKEND_MAPPING_FILE)
    mapping_data = load_yaml_mapping(yaml_path)
    param_keys = get_param_keys(yaml_path)

    context = prepare_template_context(params, backend)
    rendered = {}

    # Available roles
    for role in ["prefill", "decode", "agg"]:
        if params.get("params", {}).get(role):
            wc = make_worker_context(context, role, param_keys, mapping_data, backend)
            # Backend-specific keys are collected into a nested dict named after the backend
            rendered[role] = wc.get(backend, {})

    return rendered


def render_parameters(params: dict[str, Any], backend: str, yaml_path: Optional[str] = None) -> dict[str, Any]:
    """
    Transform unified parameters into a single backend-specific configuration dict.
    If multiple roles exist, returns the aggregate or prefill one.

    Args:
        params: Unified parameters dictionary
        backend: Target backend name
        yaml_path: Optional path to mapping YAML file

    Returns:
        Backend-specific configuration dictionary
    """
    rendered = render_backend_parameters(params, backend, yaml_path)
    if "agg" in rendered:
        return rendered["agg"]
    if "prefill" in rendered:
        return rendered["prefill"]
    if "decode" in rendered:
        return rendered["decode"]
    return {}
