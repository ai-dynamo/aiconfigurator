# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Configuration mappings for AI generator.

This module provides the mapping engine for converting parameters
to backend-specific configurations using YAML mapping files and Jinja2 templates.
"""

import logging
import os
import shlex
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

import yaml
from jinja2 import Environment, FileSystemLoader, Undefined
from packaging.version import InvalidVersion, Version

from aiconfigurator.generator.dynamo_features import (
    frontend_cli_args_string,
    kvbm_shell_exports_from_dyn_config,
)

from .rule_engine import apply_rule_plugins

_JINJA_ENV = Environment(trim_blocks=True, lstrip_blocks=True)
_JINJA_ENV.filters.setdefault("multiply", lambda value, factor: value * factor)
_TEMPLATE_ENV_CACHE: dict[str, Environment] = {}
logger = logging.getLogger(__name__)
_YAML_CACHE: dict[str, Any] = {}
_PARAM_KEYS_CACHE: dict[str, list[str]] = {}
_BASE_DIR = Path(__file__).resolve().parent
_CONFIG_DIR = (_BASE_DIR.parent / "config").resolve()
_TEMPLATE_ROOT = _CONFIG_DIR / "backend_templates"
_BACKEND_MAPPING_FILE = str((_CONFIG_DIR / "backend_config_mapping.yaml").resolve())


def _parse_template_version(version: str | None) -> Version | None:
    if version is None:
        return None
    normalized = str(version).strip()
    if normalized.lower().startswith("v") and len(normalized) > 1 and normalized[1].isdigit():
        normalized = normalized[1:]
    if not normalized:
        return None
    try:
        return Version(normalized)
    except InvalidVersion:
        return None


def _versioned_template_part(template_name: str, prefix: str, suffix: str) -> str | None:
    default_name = f"{prefix}{suffix}"
    if template_name == default_name:
        return None
    version_prefix = f"{prefix}."
    if not template_name.startswith(version_prefix) or not template_name.endswith(suffix):
        return None
    return template_name[len(version_prefix) : -len(suffix)]


def _select_versioned_template(
    template_candidates: list[Path],
    prefix: str,
    suffix: str,
    version: Optional[str],
) -> Path | None:
    """
    Select a versioned template using exact match, then closest prior version.

    Versioned backend templates are named like ``cli_args.0.5.11.j2`` or
    ``extra_engine_args.1.3.0rc14.yaml.j2``. If no compatible version exists,
    the unversioned base template is used.
    """
    default_template = next((p for p in template_candidates if p.name == f"{prefix}{suffix}"), None)
    if not version:
        return default_template

    requested_version = _parse_template_version(version)
    normalized_requested = str(version).strip()
    if normalized_requested.lower().startswith("v") and len(normalized_requested) > 1:
        normalized_requested = normalized_requested[1:]

    parsed_candidates: list[tuple[Version, Path]] = []
    for candidate in template_candidates:
        candidate_version_str = _versioned_template_part(candidate.name, prefix, suffix)
        if candidate_version_str is None:
            continue
        if candidate_version_str == normalized_requested:
            return candidate
        candidate_version = _parse_template_version(candidate_version_str)
        if candidate_version is not None:
            parsed_candidates.append((candidate_version, candidate))

    if requested_version is None:
        return default_template

    floor_candidates = [
        (candidate_version, candidate)
        for candidate_version, candidate in parsed_candidates
        if candidate_version <= requested_version
    ]
    if floor_candidates:
        return max(floor_candidates, key=lambda item: item[0])[1]
    return default_template


def _log_versioned_template_selection(
    template_kind: str,
    selected_template: Path | None,
    prefix: str,
    suffix: str,
    version: Optional[str],
) -> None:
    if not version:
        return
    if selected_template is None:
        logger.warning("No %s template available for %s; using mapping fallback.", template_kind, version)
        return

    normalized_requested = str(version).strip()
    if normalized_requested.lower().startswith("v") and len(normalized_requested) > 1:
        normalized_requested = normalized_requested[1:]
    selected_version = _versioned_template_part(selected_template.name, prefix, suffix)
    if selected_version == normalized_requested:
        return
    if selected_version is None:
        logger.warning(
            "No version-specific %s template for %s, using default %s.",
            template_kind,
            version,
            selected_template.name,
        )
    else:
        logger.warning(
            "No exact %s template for %s, using closest prior template %s.",
            template_kind,
            version,
            selected_template.name,
        )


def _generate_k8s_via_dynamo(
    param_values: dict[str, Any],
    backend: str,
    context: dict[str, Any],
) -> str:
    """
    Generate a DynamoGraphDeployment YAML using Dynamo's config modifier system
    instead of Jinja2 templates.

    This loads a base DGD YAML from Dynamo and injects the pre-computed CLI args,
    model, image, replicas, and GPU resources that AIC has already calculated.

    Args:
        param_values: The full AIC parameter dict (ServiceConfig, K8sConfig, params, etc.)
        backend: Backend name (e.g., 'vllm', 'sglang', 'trtllm')
        context: The template context dict with pre-computed CLI args and worker info

    Returns:
        Rendered k8s_deploy.yaml content as a string
    """
    # for now, we import the config modifiers from dynamo, will migrate to aiconfigurator later
    from dynamo.profiler.utils.config_modifiers import CONFIG_MODIFIERS

    modifier = CONFIG_MODIFIERS[backend]

    mode = context.get("DynConfig", {}).get("mode", "disagg")
    service_cfg = param_values.get("ServiceConfig", {})
    k8s_cfg = param_values.get("K8sConfig", {})

    kwargs: dict[str, Any] = {
        "mode": mode,
        "model_name": service_cfg.get("model_path") or service_cfg.get("served_model_path", ""),
        "image": k8s_cfg.get("k8s_image", ""),
        "namespace": k8s_cfg.get("k8s_namespace"),
    }
    node_cfg = param_values.get("NodeConfig", {})
    if node_cfg.get("num_gpus_per_node") is not None:
        kwargs["num_gpus_per_node"] = int(node_cfg["num_gpus_per_node"])

    if mode == "disagg":
        kwargs.update(
            prefill_cli_args=context.get("prefill_cli_args_list", []),
            prefill_replicas=int(context.get("prefill_workers", 1)),
            prefill_gpus=int(context.get("prefill_gpu", 1)),
            decode_cli_args=context.get("decode_cli_args_list", []),
            decode_replicas=int(context.get("decode_workers", 1)),
            decode_gpus=int(context.get("decode_gpu", 1)),
        )
    else:
        kwargs.update(
            agg_cli_args=context.get("agg_cli_args_list", []),
            agg_replicas=int(context.get("agg_workers", 1)),
            agg_gpus=int(context.get("agg_gpu", 1)),
        )

    # PVC support
    pvc_name = (k8s_cfg.get("k8s_pvc_name") or k8s_cfg.get("k8s_model_cache") or "").strip()
    if pvc_name:
        pvc_mount = (k8s_cfg.get("k8s_pvc_mount_path") or "/workspace/model_cache").strip()
        model_in_pvc = (
            k8s_cfg.get("k8s_model_path_in_pvc")
            or k8s_cfg.get("k8s_pvc_model_path")
            or k8s_cfg.get("k8s_hf_home")
            or ""
        ).strip(" /")
        kwargs["pvc_name"] = pvc_name
        kwargs["pvc_mount_path"] = pvc_mount
        kwargs["model_path"] = f"{pvc_mount}/{model_in_pvc}".rstrip("/") if model_in_pvc else pvc_mount

    config_dict = modifier.build_dgd_config(**kwargs)
    return yaml.dump(config_dict, sort_keys=False)


def _assemble_k8s_context(context: dict[str, Any], has_engine_templates: bool) -> dict[str, Any]:
    """Assemble the exact context the dynamo-j2 ``k8s_deploy.yaml`` render consumes.

    For backends with extra_engine_args templates (trtllm), suppress cli_args_list
    so the k8s template uses the ``--extra-engine-args`` file approach instead of
    inlining all parameters as redundant CLI flags. Pure helper: no behavior change
    versus the prior inline assembly.
    """
    if not has_engine_templates:
        return context
    k8s_context = dict(context)
    k8s_context["agg_cli_args_list"] = None
    k8s_context["prefill_cli_args_list"] = None
    k8s_context["decode_cli_args_list"] = None
    k8s_context["encode_cli_args_list"] = None
    return k8s_context


def render_backend_templates(
    param_values: dict[str, Any],
    backend: str,
    templates_dir: Optional[str] = None,
    version: Optional[str] = None,
    deployment_target: str = "dynamo-j2",
    _context_sink: Optional[Callable[[dict[str, Any]], None]] = None,
    _engine_context_sink: Optional[Callable[[dict[str, dict[str, Any]]], None]] = None,
    resolved_facts: Any = None,
) -> dict[str, str]:
    """
    Render templates for a specific backend with version-specific template selection.

    Args:
        param_values: Dictionary of parameter values to use in template rendering
        backend: Backend name (e.g., 'trtllm', 'vllm', 'sglang')
        templates_dir: Directory containing backend-specific template directories
        version: Version string (e.g., '1.1.0rc5'). If None, uses default templates
        deployment_target: Deployment platform ('dynamo-j2', 'dynamo-python', 'llm-d-helm',
            'llm-d-kustomize', or 'fpm')
        resolved_facts: Optional ``ResolvedFacts`` (typed ``Any`` to avoid an import
            cycle). When it carries a matched model profile, model ``defaults:``
            cli flags are appended (facts-default precedence: fill-if-absent) at the
            single cli seam so they reach BOTH the ``cli_args_*`` string artifact
            and the typed k8s builder. When ``None`` (the public API path), facts
            are SELF-RESOLVED from ``param_values`` so every caller applies the
            same model defaults; ``run_pipeline`` passes its already-resolved facts
            to avoid double resolution. A generic model (``model is None``) is a
            strict no-op, leaving the no-fact baseline byte-identical.

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
        benchmark_dir = str(_TEMPLATE_ROOT / "benchmark")
        sflow_dir = str(_TEMPLATE_ROOT / "sflow")
        search_paths = [templates_dir]
        if os.path.isdir(benchmark_dir):
            search_paths.append(benchmark_dir)
        if os.path.isdir(sflow_dir):
            search_paths.append(sflow_dir)
        env = Environment(loader=FileSystemLoader(search_paths), trim_blocks=True, lstrip_blocks=True)
        _TEMPLATE_ENV_CACHE[templates_dir] = env

    # Capture the raw request params BEFORE rule plugins run so facts can be
    # self-resolved from the same input ``run_pipeline`` resolves from (the
    # top-level ``ServiceConfig.model_path`` / ``*.system_name`` sections). This
    # keeps the two render paths' facts identical (see facts self-resolve below).
    _raw_param_values = param_values
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
        # K8sConfig.working_dir used in k8s_deploy templates, also expose top-level
        context.setdefault("K8sConfig", {})
        context["K8sConfig"]["working_dir"] = wd
        context["working_dir"] = wd

    # Determine generation mode by params presence
    params_obj = param_values.get("params", {})
    has_prefill = bool(params_obj.get("prefill"))
    has_decode = bool(params_obj.get("decode"))
    has_agg = bool(params_obj.get("agg"))
    has_encode = bool(params_obj.get("encode"))
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
    # Multimodal EPD: the encode worker is an optional extra role rendered
    # alongside prefill/decode/agg. Presence-guarded: no encode role -> unchanged.
    if has_encode:
        worker_plan = worker_plan + ["encode"]

    rendered_templates = {}

    # ── Resolve facts from our own params when the caller did not supply them ──
    # ``run_pipeline`` resolves facts up front and threads them in via
    # ``resolved_facts``; the PUBLIC api (CLI / dynamo profiler / freeze script)
    # calls this function WITHOUT it. To make every caller apply facts from a
    # single path, self-resolve here when ``resolved_facts is None`` — from the
    # RAW request params (pre-rule-plugin), matching exactly what ``run_pipeline``
    # resolves from so the two paths reconverge. Resolution is best-effort: any
    # failure degrades to ``None`` and never breaks rendering. Resolved BEFORE the
    # engine/cli render loops so the hardware ``moe_backend`` fact (applied just
    # below) is present when those loops consume the context.
    if resolved_facts is None:
        try:
            from aiconfigurator.generator.facts.request_resolution import (
                resolve_facts_for_request,
            )

            resolved_facts = resolve_facts_for_request(_raw_param_values, backend, version)
        except Exception:
            logger.warning("Fact self-resolution failed; continuing without facts.", exc_info=True)
            resolved_facts = None

    # ── Apply the hardware-derived moe_backend fact onto the shared context ──
    # HARDWARE selection (not a model default): trtllm needs the right MoE kernel
    # (WIDEEP on Blackwell, CUTLASS on Hopper — the wrong choice on Blackwell is a
    # startup crash). Applied here, BEFORE the engine render loop, which reads
    # ``context["moe_config"]["backend"]`` via each per-worker ``wc = dict(context)``
    # (the shallow copy aliases the ``moe_config`` dict, so the fill is visible to
    # every role). The sglang side (``--moe-runner-backend``) is applied to the
    # per-worker cli token lists further below, alongside the model defaults,
    # because the sglang cli template reads the *nested* ``sglang[...]`` dict that
    # ``make_worker_context`` would otherwise prune. Fill-if-absent and MoE-only
    # (``is_moe`` guard) — dense models untouched. Unlike the model-default block,
    # this runs even when ``model is None`` (the fact is keyed on hardware, not on
    # a model profile). No-op without resolved facts.
    if resolved_facts is not None and backend == "trtllm":
        from aiconfigurator.generator.facts.apply import apply_moe_backend

        apply_moe_backend(context, getattr(resolved_facts, "hardware", None), backend=backend)

    # Find template files
    template_path = Path(templates_dir)

    # Resolve engine template (version-specific preferred). Some backends (e.g., vllm, sglang)
    # do not ship engine configs at all, so only warn when such templates actually exist.
    engine_template_candidates = list(template_path.glob("extra_engine_args*.yaml.j2"))
    has_engine_templates = bool(engine_template_candidates)
    engine_template_file = _select_versioned_template(
        engine_template_candidates,
        "extra_engine_args",
        ".yaml.j2",
        version,
    )
    if has_engine_templates:
        _log_versioned_template_selection(
            "engine",
            engine_template_file,
            "extra_engine_args",
            ".yaml.j2",
            version,
        )

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

    def _populate_trtllm_nested_engine_config(wc: dict[str, Any]) -> None:
        def _set_nested(config_name: str, nested_key: str, source_key: str) -> None:
            source_value = wc.get(source_key)
            if source_value is None:
                return
            config = dict(wc.get(config_name) or {})
            if nested_key not in config or config.get(nested_key) is None:
                config[nested_key] = source_value
            wc[config_name] = config

        _set_nested("kv_cache_config", "free_gpu_memory_fraction", "kv_cache_free_gpu_memory_fraction")
        _set_nested("kv_cache_config", "dtype", "kv_cache_dtype")
        _set_nested("kv_cache_config", "tokens_per_block", "tokens_per_block")
        _set_nested("cuda_graph_config", "enable_padding", "cuda_graph_enable_padding")
        _set_nested("cuda_graph_config", "batch_sizes", "cuda_graph_batch_sizes")
        _set_nested("cache_transceiver_config", "max_tokens_in_buffer", "cache_transceiver_max_tokens_in_buffer")

        if wc.get("disable_prefix_cache") is False:
            kv_cache_config = dict(wc.get("kv_cache_config") or {})
            kv_cache_config.setdefault("enable_block_reuse", True)
            wc["kv_cache_config"] = kv_cache_config

        if wc.get("cache_transceiver_config"):
            cache_transceiver_config = dict(wc["cache_transceiver_config"])
            cache_transceiver_config.setdefault("backend", "DEFAULT")
            wc["cache_transceiver_config"] = cache_transceiver_config

        decoding_type = wc.get("speculative_decoding_type")
        num_nextn_predict_layers = wc.get("num_nextn_predict_layers")
        if decoding_type is not None or num_nextn_predict_layers is not None:
            speculative_config = dict(wc.get("speculative_config") or {})
            if decoding_type is not None:
                speculative_config.setdefault("decoding_type", decoding_type)
            if num_nextn_predict_layers is not None:
                speculative_config.setdefault("num_nextn_predict_layers", num_nextn_predict_layers)
            wc["speculative_config"] = speculative_config

    def build_engine_worker_context(worker: str) -> dict[str, Any]:
        """Assemble the exact per-worker engine render context for one worker role.

        Pure extraction of the inline engine-context assembly used by the engine
        render loop below: ``make_worker_context(...)`` plus, for trtllm, the
        nested-engine-config population. No behavior change.
        """
        wc = make_worker_context(context, worker, param_keys, mapping_data)
        if backend == "trtllm":
            _populate_trtllm_nested_engine_config(wc)
        return wc

    if _engine_context_sink is not None:
        _engine_context_sink({worker: build_engine_worker_context(worker) for worker in worker_plan})

    if engine_template_file is not None:
        try:
            eng_tmpl = env.get_template(engine_template_file.name)
            for worker in worker_plan:
                wc = build_engine_worker_context(worker)
                rendered = eng_tmpl.render(**wc)
                # Optional per-role passthrough of arbitrary engine-config keys
                # the template doesn't model. Presence-guarded (absent ->
                # output unchanged). Appended as YAML so a duplicate key lets the
                # user override a template-emitted value (engine parser: last wins).
                extra_engine = (param_values.get("params", {}).get(worker) or {}).get("extra_engine_args")
                if isinstance(extra_engine, dict) and extra_engine:
                    import yaml as _yaml

                    rendered = rendered.rstrip("\n") + "\n" + _yaml.safe_dump(extra_engine, sort_keys=False)
                if worker == "agg":
                    out_name = "extra_engine_args_agg.yaml"
                elif worker == "prefill":
                    out_name = "extra_engine_args_prefill.yaml"
                elif worker == "encode":
                    out_name = "extra_engine_args_encode.yaml"
                else:
                    out_name = "extra_engine_args_decode.yaml"
                rendered_templates[out_name] = rendered
        except Exception as e:
            logger.warning(f"Failed to render engine template {engine_template_file.name}: {e}")

    # Inject inline engine args content into context for k8s template
    # These are used when K8sConfig.k8s_engine_mode == 'inline'
    context["prefill_engine_args_inline"] = rendered_templates.get("extra_engine_args_prefill.yaml", "")
    context["decode_engine_args_inline"] = rendered_templates.get("extra_engine_args_decode.yaml", "")
    context["agg_engine_args_inline"] = rendered_templates.get("extra_engine_args_agg.yaml", "")
    context["encode_engine_args_inline"] = rendered_templates.get("extra_engine_args_encode.yaml", "")

    # Resolve CLI args template (version-specific preferred)
    cli_template_candidates = list(template_path.glob("cli_args*.j2"))
    cli_template_file = _select_versioned_template(cli_template_candidates, "cli_args", ".j2", version)
    _log_versioned_template_selection("CLI args", cli_template_file, "cli_args", ".j2", version)

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
        elif worker == "encode":
            context["encode_cli_args"] = cli
            context["encode_cli_args_list"] = cli_list
            rendered_templates["cli_args_encode"] = cli
        else:
            context["agg_cli_args"] = cli
            context["agg_cli_args_list"] = cli_list
            rendered_templates["cli_args_agg"] = cli

    # ── Apply facts-default cli flags from resolved facts (facts-default layer) ──
    # Single seam: after the per-role token lists are built and BEFORE the cli
    # string formatting / k8s build consume them, so appended flags appear in
    # BOTH the ``cli_args_*`` string artifact and the typed k8s builder.
    # Fill-if-absent only (user/recipe/rule values already in the list win).
    # Two facts feed this seam:
    #   1. The hardware ``moe_backend`` fact for sglang (``--moe-runner-backend``),
    #      applied on the token list here — the sglang cli template reads a nested
    #      ``sglang[...]`` dict that ``make_worker_context`` prunes for a fact set
    #      after ``prepare_template_context``, so the post-render token list is the
    #      reliable seam. Runs even when ``model is None`` (hardware-keyed fact).
    #   2. Model ``defaults:`` flags, only when a model profile matched.
    # ``_facts_touched_cli`` gates the single re-sync below so the no-fact canary
    # (no profile + non-MoE / no hardware fact) is byte-identical through here.
    _facts_touched_cli = False
    if resolved_facts is not None and backend == "sglang":
        from aiconfigurator.generator.facts.apply import apply_moe_backend

        apply_moe_backend(context, getattr(resolved_facts, "hardware", None), backend="sglang")
        moe_choice = context.get("moe_backend")
        if moe_choice:
            for worker in worker_plan:
                tokens = context.get(f"{worker}_cli_args_list")
                if isinstance(tokens, list) and "--moe-runner-backend" not in tokens:
                    tokens.append("--moe-runner-backend")
                    tokens.append(str(moe_choice))
                    _facts_touched_cli = True

    if resolved_facts is not None and getattr(resolved_facts, "model", None) is not None:
        from aiconfigurator.generator.facts.apply import apply_facts

        apply_facts(context, resolved_facts, backend)
        _facts_touched_cli = True

    # Re-sync the string artifacts from the (possibly extended) token lists so
    # appended flags render identically to user-supplied ones. Only runs when a
    # fact actually touched the cli, so the no-fact path's strings are untouched.
    if _facts_touched_cli:
        for worker in worker_plan:
            list_key = f"{worker}_cli_args_list"
            str_key = f"{worker}_cli_args"
            tmpl_key = f"cli_args_{worker}"
            token_list = context.get(list_key)
            if not isinstance(token_list, list):
                continue
            cli_str = " ".join(shlex.quote(tok) for tok in token_list)
            context[str_key] = cli_str
            rendered_templates[tmpl_key] = cli_str

    # ── Translate: append --trtllm.* dynamic flags from extra_engine_args ──
    # When the dynamo-python path is active and the backend is trtllm, convert
    # the rendered extra_engine_args YAML into --trtllm.<key>.<subkey> <value>
    # flags and merge them into cli_args.
    # Note: list-typed values (e.g. cuda_graph_config.batch_sizes) are skipped
    # because dynamo's infer_type cannot round-trip lists; the engine uses its
    # built-in defaults for those fields.
    if deployment_target == "dynamo-python" and backend == "trtllm":
        from .translate import yaml_to_dynamic_flags

        for worker in worker_plan:
            yaml_key = f"extra_engine_args_{worker}.yaml"
            yaml_content = rendered_templates.get(yaml_key, "")
            if not yaml_content:
                continue

            dynamic_flags = yaml_to_dynamic_flags(yaml_content)

            cli_key = f"{worker}_cli_args"
            list_key = f"{worker}_cli_args_list"
            tmpl_key = f"cli_args_{worker}"

            existing_list = list(context.get(list_key) or [])
            existing_list.extend(dynamic_flags)
            context[list_key] = existing_list
            cli_str = " ".join(shlex.quote(a) for a in existing_list)
            context[cli_key] = cli_str
            rendered_templates[tmpl_key] = cli_str

    # Compute GPU counts per worker using rule outputs, falling back to WorkerConfig.
    pv_params = param_values.get("params", {}) or {}
    worker_config = param_values.get("WorkerConfig", {}) or {}
    prefill_gpu = int(
        pv_params.get("prefill", {}).get("gpus_per_worker") or worker_config.get("prefill_gpus_per_worker") or 1
    )
    decode_gpu = int(
        pv_params.get("decode", {}).get("gpus_per_worker") or worker_config.get("decode_gpus_per_worker") or 1
    )
    agg_gpu = int(pv_params.get("agg", {}).get("gpus_per_worker") or worker_config.get("agg_gpus_per_worker") or 1)
    encode_gpu = int(
        pv_params.get("encode", {}).get("gpus_per_worker") or worker_config.get("encode_gpus_per_worker") or 1
    )

    context["prefill_gpu"] = prefill_gpu
    context["decode_gpu"] = decode_gpu
    context["agg_gpu"] = agg_gpu
    context["encode_gpu"] = encode_gpu

    # Render auxiliary templates based on deployment target
    if deployment_target == "llm-d-kustomize":
        # llm-d v0.7+ modelserver deployment: render Kustomize overlay patches.
        is_agg_mode = (context.get("DynConfig") or {}).get("mode", "disagg") == "agg"
        kustomize_templates = [
            ("llm-d-kustomization.yaml.j2", "kustomization.yaml"),
            ("llm-d-patch-decode.yaml.j2", "patch-vllm.yaml" if is_agg_mode else "patch-decode.yaml"),
        ]
        if not is_agg_mode:
            kustomize_templates.append(("llm-d-patch-prefill.yaml.j2", "patch-prefill.yaml"))
        for template_name, artifact_name in kustomize_templates:
            if not (template_path / template_name).exists():
                continue
            try:
                tmpl = env.get_template(template_name)
                rendered_templates[artifact_name] = tmpl.render(**context)
            except Exception as e:
                logger.warning(f"Failed to render template {template_name}: {e}")
    elif deployment_target == "llm-d-helm":
        # llm-d deployment: render Helm values for llm-d-modelservice chart
        llmd_values_aux = template_path / "llm-d-values.yaml.j2"
        if llmd_values_aux.exists():
            try:
                tmpl = env.get_template("llm-d-values.yaml.j2")
                rendered = tmpl.render(**context)
                rendered_templates["llm-d-values.yaml"] = rendered
            except Exception as e:
                logger.warning(f"Failed to render template llm-d-values.yaml.j2: {e}")
    elif deployment_target == "fpm":
        # FPM is deliberately isolated from the normal DGD/run-script path. It
        # emits exactly two artifacts: a reusable, keepalive resource Pod and a
        # complete engine launch script. The builder consumes the same fully
        # resolved vLLM context as the normal typed K8s builder, so rule/mapping/
        # versioned-template output remains the source of the base argv.
        from aiconfigurator.generator.builders.fpm_builder import build_fpm_artifacts

        fpm_context = _assemble_k8s_context(context, has_engine_templates)
        if _context_sink is not None:
            _context_sink(fpm_context)
        return build_fpm_artifacts(
            fpm_context,
            backend,
            resolved_facts=resolved_facts,
            param_values=param_values,
        )
    elif deployment_target == "dynamo-python":
        # Dynamo deployment using Dynamo's Python config modifiers
        try:
            rendered_templates["k8s_deploy.yaml"] = _generate_k8s_via_dynamo(param_values, backend, context)
        except Exception as e:
            logger.warning(f"Failed to generate k8s config via Dynamo: {e}")
    else:
        # Dynamo deployment (default: dynamo-j2). The k8s_deploy.yaml for all
        # backends (vllm/sglang/trtllm) is now produced by the typed k8s builder
        # (build_dgd); the legacy k8s_deploy.yaml.j2 templates have been retired.
        try:
            k8s_context = _assemble_k8s_context(context, has_engine_templates)
            if _context_sink is not None:
                _context_sink(k8s_context)
            from aiconfigurator.generator.builders.dgd_model import dgd_documents_to_yaml
            from aiconfigurator.generator.builders.k8s_builder import build_dgd

            rendered_templates["k8s_deploy.yaml"] = dgd_documents_to_yaml(
                build_dgd(k8s_context, backend, resolved_facts=resolved_facts)
            )
        except Exception as e:
            logger.warning(f"Failed to build k8s_deploy.yaml: {e}")

    # Benchmark templates (Dynamo-specific)
    if deployment_target in ("dynamo-j2", "dynamo-python"):
        # benchmark job: single file from shared benchmark template folder
        bench_dir = _TEMPLATE_ROOT / "benchmark"
        bench_aux = bench_dir / "k8s_bench.yaml.j2"
        if bench_aux.exists():
            try:
                tmpl = env.get_template("k8s_bench.yaml.j2")
                rendered = tmpl.render(**context)
                rendered_templates["k8s_bench.yaml"] = rendered
            except Exception as e:
                logger.warning(f"Failed to render template k8s_bench.yaml.j2: {e}")

        # benchmark run script: single file from shared benchmark template folder
        bench_run_aux = bench_dir / "bench_run.sh.j2"
        if bench_run_aux.exists():
            try:
                tmpl = env.get_template("bench_run.sh.j2")
                rendered = tmpl.render(**context)
                rendered_templates["bench_run.sh"] = rendered
            except Exception as e:
                logger.warning(f"Failed to render template bench_run.sh.j2: {e}")

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

                    # Interleave allocation to balance prefill and decode workers across nodes
                    # We try to keep p_worker/d_worker ratio consistent across nodes
                    if p_worker > 0 and d_worker > 0:
                        import math

                        gcd = math.gcd(p_worker, d_worker)
                        p_unit = p_worker // gcd
                        d_unit = d_worker // gcd

                        p_rem, d_rem = p_worker, d_worker
                        while p_rem > 0 or d_rem > 0:
                            # Allocate p_unit prefill
                            for _ in range(min(p_unit, p_rem)):
                                placed = False
                                for n in nodes:
                                    if n["used"] + p_gpu <= gpu_per_node:
                                        n["p_worker"] += 1
                                        n["used"] += p_gpu
                                        placed = True
                                        break
                                if not placed:
                                    nodes.append({"p_worker": 1, "d_worker": 0, "used": p_gpu})
                                p_rem -= 1

                            # Allocate d_unit decode
                            for _ in range(min(d_unit, d_rem)):
                                placed = False
                                for n in nodes:
                                    if n["used"] + d_gpu <= gpu_per_node:
                                        n["d_worker"] += 1
                                        n["used"] += d_gpu
                                        placed = True
                                        break
                                if not placed:
                                    nodes.append({"p_worker": 0, "d_worker": 1, "used": d_gpu})
                                d_rem -= 1
                    else:
                        # Fallback for simple cases where one type is missing
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

                    return [{"p_worker": n.get("p_worker", 0), "d_worker": n.get("d_worker", 0)} for n in nodes]

                plan = _allocate_disagg_nodes(
                    prefill_workers, prefill_gpu, decode_workers, decode_gpu, num_gpus_per_node
                )

                for idx, cnt in enumerate(plan):
                    node_ctx = dict(context)
                    svc = dict(node_ctx.get("ServiceConfig", {}))
                    svc["include_frontend"] = idx == 0
                    node_ctx["ServiceConfig"] = svc
                    node_ctx["prefill_gpu"] = prefill_gpu
                    node_ctx["decode_gpu"] = decode_gpu
                    node_ctx["prefill_workers"] = int(cnt.get("p_worker", 0))
                    node_ctx["decode_workers"] = int(cnt.get("d_worker", 0))
                    node_ctx["decode_gpu_offset"] = int(cnt.get("p_worker", 0)) * prefill_gpu
                    rendered = tmpl.render(**node_ctx)
                    rendered_templates[f"run_{idx}.sh"] = rendered
        except Exception as e:
            logger.warning(f"Failed to render template run.sh.j2: {e}")

    # Multimodal EPD single-pod (colocated) artifacts. trtllm's image-URL E-PD
    # flow transfers vision embeddings via CUDA IPC, which needs the encode and
    # prefill/PD workers to share GPU memory; k8s per-pod GPU isolation breaks
    # that. So when an encode role is present we additionally emit a launch
    # script (encode colocated on GPU 0 with prefill/PD) and a single Pod that
    # runs all workers together with pod-local etcd/nats. Presence-guarded
    # (trtllm + encode role only) -> no effect on non-EPD output.
    if has_encode and backend == "trtllm":
        epd_run_tmpl = template_path / "epd_run.sh.j2"
        epd_pod_tmpl = template_path / "epd_pod.yaml.j2"
        if epd_run_tmpl.exists() and epd_pod_tmpl.exists():
            try:
                _enc = param_values.get("params", {}).get("encode") or {}
                if context.get("DynConfig", {}).get("mode") == "agg":
                    _total = int(context.get("agg_workers", 1)) * int(context.get("agg_gpu", 1))
                else:
                    _total = int(context.get("prefill_workers", 1)) * int(context.get("prefill_gpu", 1)) + int(
                        context.get("decode_workers", 1)
                    ) * int(context.get("decode_gpu", 1))
                from aiconfigurator.generator.naive import _sanitize_rfc1123

                epd_ctx = dict(context)
                epd_ctx["epd_total_gpus"] = max(_total, 1)
                epd_ctx["epd_name"] = _sanitize_rfc1123(f"{context.get('name') or 'dynamo'}-epd")
                epd_ctx["encode_modality"] = _enc.get("modality") or "multimodal"
                epd_ctx["encode_allowed_local_media_path"] = _enc.get("allowed_local_media_path") or "/tmp"
                epd_ctx["encode_max_file_size_mb"] = _enc.get("max_file_size_mb") or 50
                run_sh = env.get_template("epd_run.sh.j2").render(**epd_ctx)
                rendered_templates["epd_run.sh"] = run_sh
                epd_ctx["epd_run_sh"] = run_sh
                rendered_templates["epd_pod.yaml"] = env.get_template("epd_pod.yaml.j2").render(**epd_ctx)
            except Exception as e:
                logger.warning(f"Failed to render EPD single-pod artifacts: {e}")

    # sflow deploy: shared template from sflow/ folder
    sflow_tmpl_name = "sflow_deploy.yaml.j2"
    try:
        env.get_template(sflow_tmpl_name)
    except Exception:
        pass  # template not available — skip
    else:
        try:
            from ..sflow import enrich_context_for_sflow, postprocess_sflow

            sflow_ctx = enrich_context_for_sflow(context, param_values, backend, rendered_templates)
            sflow_tmpl = env.get_template(sflow_tmpl_name)
            rendered = sflow_tmpl.render(**sflow_ctx)
            rendered_templates["sflow.yaml"] = postprocess_sflow(rendered)
        except Exception as e:
            logger.warning(f"Failed to render sflow template: {e}")

    return rendered_templates


def build_k8s_context_for_test(param_values, backend, templates_dir=None, backend_version=None):
    """Return the exact context dict the dynamo-j2 k8s_deploy render uses.

    Pure extraction of existing logic so the typed builder and tests can consume
    the identical context. Runs the same ``render_backend_templates`` pipeline the
    Jinja k8s render uses and captures the assembled k8s context via a sink, so the
    returned dict is byte-for-byte what ``k8s_deploy.yaml.j2`` is rendered with.
    Must not change ``render_backend_templates`` output.
    """
    captured: dict[str, Any] = {}

    def _sink(ctx: dict[str, Any]) -> None:
        captured["ctx"] = ctx

    render_backend_templates(
        param_values,
        backend,
        templates_dir,
        backend_version,
        deployment_target="dynamo-j2",
        _context_sink=_sink,
    )
    return captured["ctx"]


def build_engine_worker_contexts_for_test(param_values, backend, templates_dir=None, backend_version=None) -> dict:
    """Return {worker: wc} — the exact per-worker engine contexts the engine render uses.

    Pure extraction of existing logic; must not change render_backend_templates output.
    Runs the same ``render_backend_templates`` pipeline the Jinja engine render uses and
    captures the assembled per-worker engine contexts via a sink, so each returned context
    is byte-for-byte what ``extra_engine_args*.yaml.j2`` is rendered with.
    """
    captured: dict[str, dict[str, Any]] = {}

    def _sink(contexts: dict[str, dict[str, Any]]) -> None:
        captured.update(contexts)

    render_backend_templates(
        param_values,
        backend,
        templates_dir,
        backend_version,
        deployment_target="dynamo-j2",
        _engine_context_sink=_sink,
    )
    return captured


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
    model_config = param_values.get("ModelConfig") or {}
    if not isinstance(model_config, dict):
        model_config = {}
    context["ModelConfig"] = dict(model_config)
    if model_config.get("is_moe"):
        context["is_moe"] = model_config["is_moe"]

    # Extract unified service configuration
    service_config = param_values.get("ServiceConfig", {})
    context["model_path"] = service_config.get("model_path") or service_config.get("served_model_path", "")
    context["served_model_path"] = service_config.get("served_model_path")
    context["ServiceConfig"] = dict(service_config)

    # Extract K8s configuration
    k8s_config = param_values.get("K8sConfig", {})
    context["name_prefix"] = k8s_config.get("name_prefix") or "dynamo"
    context["k8s_namespace"] = k8s_config.get("k8s_namespace")
    context["k8s_image"] = k8s_config.get("k8s_image")
    context["k8s_image_pull_secret"] = k8s_config.get("k8s_image_pull_secret")
    context["working_dir"] = k8s_config.get("working_dir")
    context["k8s_engine_mode"] = k8s_config.get("k8s_engine_mode")
    # PVC config: new unified names with backward compat fallbacks
    context["k8s_pvc_name"] = k8s_config.get("k8s_pvc_name") or k8s_config.get("k8s_model_cache")
    context["k8s_pvc_mount_path"] = k8s_config.get("k8s_pvc_mount_path") or "/workspace/model_cache"
    context["k8s_model_path_in_pvc"] = (
        k8s_config.get("k8s_model_path_in_pvc") or k8s_config.get("k8s_pvc_model_path") or k8s_config.get("k8s_hf_home")
    )
    # Backward compat aliases for Jinja2 templates
    context["k8s_model_cache"] = context["k8s_pvc_name"]
    context["k8s_hf_home"] = (
        f"{context['k8s_pvc_mount_path']}/{context['k8s_model_path_in_pvc']}".rstrip("/")
        if context["k8s_model_path_in_pvc"]
        else ""
    )

    # Extract DynConfig for mode/router decisions
    dyn_config = param_values.get("DynConfig", {})
    if isinstance(dyn_config, dict):
        context["DynConfig"] = dyn_config
        frontend_dyn = dyn_config if dyn_config.get("router_mode") or dyn_config.get("router_config") else {}
        context["frontend_extra_args"] = frontend_cli_args_string(
            frontend_dyn,
            service_config,
            include_http_port=False,
        )
        context["kvbm_env_exports"] = kvbm_shell_exports_from_dyn_config(dyn_config, backend=backend)
    mode_value = dyn_config.get("mode") if isinstance(dyn_config, dict) else None
    mode_value = mode_value or "disagg"
    enable_router = bool(dyn_config.get("enable_router")) if isinstance(dyn_config, dict) else False
    name_suffix = "agg" if mode_value == "agg" else "disagg"
    router_suffix = "-router" if enable_router else ""
    full_name = f"{context['name_prefix']}-{name_suffix}{router_suffix}"
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
    # Multimodal EPD encode worker (set only when present so non-EPD context is
    # unchanged; the k8s builder reads these only when building the encode worker).
    if worker_params.get("encode"):
        context["encode_params"] = worker_params.get("encode", {})
        context["encode_workers"] = workers.get("encode_workers", 1)
        context["encode_gpus_per_worker"] = workers.get("encode_gpus_per_worker")
        context["encode_gpu"] = context["encode_gpus_per_worker"]

    fr = 1 if (context.get("include_frontend") is True) else 0
    context["frontend_replicas"] = fr

    node_config = param_values.get("NodeConfig", {})
    if isinstance(node_config, dict):
        context["NodeConfig"] = dict(node_config)

    # SLA + benchmark configuration (used by k8s_bench.yaml)
    sla_config = param_values.get("SlaConfig", {})
    if isinstance(sla_config, dict):
        context["SlaConfig"] = dict(sla_config)
    bench_config = param_values.get("BenchConfig", {}) or {}
    if isinstance(bench_config, dict):
        bench_context = dict(bench_config)
        if bench_context.get("prefix") is None:
            model_prefix = model_config.get("prefix")
            service_prefix = service_config.get("prefix")
            if model_prefix is not None:
                bench_context["prefix"] = model_prefix
            elif service_prefix is not None:
                bench_context["prefix"] = service_prefix
            else:
                bench_context["prefix"] = 0
        if bench_context.get("prefix_prompt_pool_size") is None:
            bench_context["prefix_prompt_pool_size"] = 1
        context["BenchConfig"] = bench_context

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
    for worker_type in ["prefill", "decode", "agg", "encode"]:
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
    for worker_type in ["prefill", "decode", "agg", "encode"]:
        worker_config = worker_params.get(worker_type, {})
        for key, value in worker_config.items():
            context[f"{worker_type}_{key}"] = value

    # No dynamo_config in new templates

    # Add engine args paths for templates
    context["prefill_engine_args"] = "/workspace/engine_configs/prefill_config.yaml"
    context["decode_engine_args"] = "/workspace/engine_configs/decode_config.yaml"
    context["agg_engine_args"] = "/workspace/engine_configs/agg_config.yaml"
    context["encode_engine_args"] = "/workspace/engine_configs/encode_config.yaml"

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

    # Extract LlmdConfig for llm-d deployment target
    llmd_config = param_values.get("LlmdConfig", {})
    if isinstance(llmd_config, dict):
        context["LlmdConfig"] = dict(llmd_config)

    # Support top-level worker counts and parallelism settings (for backward compatibility and testing)
    # These override WorkerConfig values if present
    for key in [
        "prefill_workers",
        "decode_workers",
        "agg_workers",
        "prefill_tensor_parallel_size",
        "decode_tensor_parallel_size",
        "agg_tensor_parallel_size",
        "prefill_data_parallel_size",
        "decode_data_parallel_size",
        "agg_data_parallel_size",
        "prefill_cli_args_list",
        "decode_cli_args_list",
        "agg_cli_args_list",
    ]:
        if key in param_values:
            context[key] = param_values[key]

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
