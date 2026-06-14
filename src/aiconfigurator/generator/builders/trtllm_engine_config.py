# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed TRT-LLM engine-config builder.

Builds the ``extra_engine_args`` YAML payload as a Python dict, mirroring
``backend_templates/trtllm/extra_engine_args*.yaml.j2`` top-to-bottom so a
template edit maps to exactly one builder section (design doc Section 3.1).

This is THE default rendering path for trtllm engine configs (Stage 3
cutover): ``rendering/engine.py`` builds the engine config dicts here for
every covered version and serializes them with ``yaml.safe_dump``. Versions
below the coverage floor fall back to the legacy Jinja templates (floor
match). Parity tests live in
tests/unit/generator/test_engine_config_builder_parity.py.

Version handling
----------------
Version behavior is **manifest-driven**: each anchor file under
``manifests/backends/trtllm/versions/<ver>.yaml`` carries a ``builder_facts``
block describing the versioned template's variable-source differences, and
``manifests/backends/trtllm/versions/base.yaml`` carries the facts for the
unversioned base template (``version=None``). Anchors are selected with the
contract loader's floor matching (exactly like ``_select_versioned_template``)
— never as ``if version >= literal`` branches in builder logic. The coverage
floor is the lowest anchor manifest (currently ``1.2.0rc6.post1``, the oldest
trtllm version reachable from the active Dynamo releases in
``config/backend_version_matrix.yaml``); requesting anything below it raises.
"""
from __future__ import annotations

from typing import Any, Mapping

# builder_facts keys every anchor manifest must define:
#   num_nextn_source:
#     "top_level"           -> {{ num_nextn_predict_layers }}
#     "speculative_config"  -> {{ speculative_config.num_nextn_predict_layers }},
#                              guarded by `is defined`
#   print_iter_log_source:
#     "top_level"           -> {{ print_iter_log | default(false) }}
#     "service_config"      -> {{ ServiceConfig.print_iter_log | default(false) }}
_REQUIRED_FACT_KEYS = ("num_nextn_source", "print_iter_log_source")

# Default mirrored from the template's `cuda_graph_config.batch_sizes | default(...)`.
_DEFAULT_CUDA_GRAPH_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def _validated_facts(manifest: Mapping[str, Any], origin: str) -> dict[str, str]:
    facts = manifest.get("builder_facts")
    if not isinstance(facts, Mapping) or any(k not in facts for k in _REQUIRED_FACT_KEYS):
        raise ValueError(
            f"trtllm version manifest {origin} has no usable builder_facts block "
            f"(required keys: {list(_REQUIRED_FACT_KEYS)})"
        )
    return {key: str(facts[key]) for key in _REQUIRED_FACT_KEYS}


def _facts_for_version(version: str | None) -> dict[str, str]:
    """Load builder facts from the version manifests (floor match, like templates).

    Mirrors ``_select_versioned_template``: ``version=None`` (or an unparseable
    version string) selects the base template, so it loads
    ``versions/base.yaml``; otherwise the contract loader floor-matches the
    anchor manifests and a request below the lowest anchor is an error.
    """
    from aiconfigurator.generator.contract.loader import (
        MANIFESTS_DIR,
        _select_version_manifest,
        load_yaml,
    )
    from aiconfigurator.generator.rendering.engine import _parse_template_version

    versions_dir = MANIFESTS_DIR / "backends" / "trtllm" / "versions"
    if version is not None:
        manifest = _select_version_manifest(versions_dir, str(version).strip())
        if manifest is not None:
            return _validated_facts(manifest, origin=str(manifest.get("version")))
        if _parse_template_version(version) is not None:
            raise ValueError(
                f"Typed TRT-LLM engine-config builder has no version-facts manifest "
                f"covering requested version {version!r} (below the coverage floor — "
                f"the lowest anchor in {versions_dir})"
            )
        # Unparseable version: the renderer falls back to the base template.
    return _validated_facts(load_yaml(str(versions_dir / "base.yaml")), origin="base")


def engine_builder_covers(version: str | None) -> bool:
    """True when the version-facts manifests cover *version* (floor included).

    ``None`` and unparseable versions resolve to the base facts (covered);
    parseable versions below the lowest anchor manifest are not covered and
    must fall back to the legacy Jinja engine templates.
    """
    try:
        _facts_for_version(version)
    except ValueError:
        return False
    return True


def _dict_at(context: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """Nested lookup helper: Jinja `x.y` on a missing/non-dict `x` is undefined."""
    value = context.get(key)
    return value if isinstance(value, Mapping) else {}


def _present_or(mapping: Mapping[str, Any], key: str, default: Any) -> Any:
    """Mirror Jinja's `| default(...)`: applies only when the key is undefined."""
    return mapping[key] if key in mapping else default


def build_engine_config(
    context: Mapping[str, Any],
    role: str | None = None,
    version: str | None = None,
) -> dict[str, Any]:
    """Build the TRT-LLM engine config dict for one worker role.

    ``context`` is the per-role engine render context produced by
    :func:`aiconfigurator.generator.rendering.engine.build_engine_render_contexts`
    (i.e. the exact context the Jinja template receives). The template has no
    role conditionals — role differences (``disable_overlap_scheduler``,
    ``cache_transceiver_config`` presence, ...) already live in the context —
    so ``role`` is accepted for API symmetry/diagnostics only.
    """
    del role  # see docstring: role differences are context-borne
    facts = _facts_for_version(version)
    cfg: dict[str, Any] = {"backend": "pytorch"}

    # {% if is_moe is defined %}
    if "is_moe" in context:
        cfg["moe_expert_parallel_size"] = context.get("moe_expert_parallel_size")
        cfg["moe_tensor_parallel_size"] = context.get("moe_tensor_parallel_size")
        moe_src = _dict_at(context, "moe_config")
        moe_config: dict[str, Any] = {"backend": _present_or(moe_src, "backend", "CUTLASS")}
        if "load_balancer" in moe_src:
            moe_config["load_balancer"] = moe_src["load_balancer"]
        cfg["moe_config"] = moe_config

    cfg["tensor_parallel_size"] = context.get("tensor_parallel_size")
    cfg["pipeline_parallel_size"] = context.get("pipeline_parallel_size")
    cfg["enable_attention_dp"] = _present_or(context, "enable_attention_dp", False)
    cfg["enable_chunked_prefill"] = _present_or(context, "enable_chunked_prefill", False)

    cfg["max_batch_size"] = context.get("max_batch_size")
    cfg["max_num_tokens"] = context.get("max_num_tokens")
    # {% if _max_seq_len is not none and _max_seq_len != '' %}
    max_seq_len = context.get("max_seq_len")
    if max_seq_len is not None and max_seq_len != "":
        cfg["max_seq_len"] = max_seq_len

    kv_src = _dict_at(context, "kv_cache_config")
    kv_cache_config: dict[str, Any] = {
        "free_gpu_memory_fraction": _present_or(kv_src, "free_gpu_memory_fraction", 0.80),
        "dtype": _present_or(kv_src, "dtype", "auto"),
    }
    if "tokens_per_block" in kv_src:
        kv_cache_config["tokens_per_block"] = kv_src["tokens_per_block"]
    kv_cache_config["enable_block_reuse"] = _present_or(kv_src, "enable_block_reuse", False)
    cfg["kv_cache_config"] = kv_cache_config

    # {% if cache_transceiver_config.max_tokens_in_buffer is defined %}
    # (present for disagg prefill/decode roles, absent for agg)
    ct_src = _dict_at(context, "cache_transceiver_config")
    if "max_tokens_in_buffer" in ct_src:
        cfg["cache_transceiver_config"] = {
            "backend": _present_or(ct_src, "backend", "DEFAULT"),
            "max_tokens_in_buffer": ct_src["max_tokens_in_buffer"],
        }

    cg_src = _dict_at(context, "cuda_graph_config")
    cfg["cuda_graph_config"] = {
        "enable_padding": _present_or(cg_src, "enable_padding", False),
        "batch_sizes": list(cg_src["batch_sizes"])
        if "batch_sizes" in cg_src
        else list(_DEFAULT_CUDA_GRAPH_BATCH_SIZES),
    }

    cfg["disable_overlap_scheduler"] = _present_or(context, "disable_overlap_scheduler", True)
    if facts["print_iter_log_source"] == "top_level":
        cfg["print_iter_log"] = _present_or(context, "print_iter_log", False)
    else:
        cfg["print_iter_log"] = _present_or(_dict_at(context, "ServiceConfig"), "print_iter_log", False)

    # {% if speculative_config.decoding_type is defined %}
    spec_src = _dict_at(context, "speculative_config")
    if "decoding_type" in spec_src:
        speculative_config: dict[str, Any] = {"decoding_type": spec_src["decoding_type"]}
        if spec_src["decoding_type"] == "MTP":
            if facts["num_nextn_source"] == "speculative_config":
                if "num_nextn_predict_layers" in spec_src:
                    speculative_config["num_nextn_predict_layers"] = spec_src["num_nextn_predict_layers"]
            else:
                speculative_config["num_nextn_predict_layers"] = context.get("num_nextn_predict_layers")
        cfg["speculative_config"] = speculative_config

    return cfg


def build_engine_configs_for_case(case: dict[str, Any], version: str | None = None) -> dict[str, dict[str, Any]]:
    """Build typed engine configs for a baseline case (parity-test entry point).

    Rebuilds params exactly like ``tests/baseline/conftest.generate_case_artifacts``
    and reuses the engine's own pre-render pipeline (per-role engine contexts),
    never re-deriving values.
    """
    from aiconfigurator.generator.naive import build_naive_generator_params
    from aiconfigurator.generator.rendering.engine import build_engine_render_contexts

    params = build_naive_generator_params(
        model_name=case["model_name"],
        total_gpus=case["total_gpus"],
        system_name=case["system"],
        backend_name=case["backend"],
        mode=case.get("mode", "agg"),
    )
    contexts = build_engine_render_contexts(params, case["backend"], version=version)
    return {role: build_engine_config(ctx, role=role, version=version) for role, ctx in contexts.items()}
