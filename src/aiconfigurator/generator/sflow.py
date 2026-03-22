# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for sflow YAML generation within the rendering pipeline.

The sflow template (``backend_templates/sflow/sflow_deploy.yaml.j2``) is a
pure orchestration skeleton. ALL backend-specific logic (launch commands,
benchmark scripts, operator config, probe patterns) is built here and
injected as pre-formatted YAML snippets via the template context.

The launch commands directly reuse rendered backend artifacts:
- ``agg/prefill/decode_cli_args`` from ``cli_args*.j2``
- ``extra_engine_args_*.yaml`` from TRTLLM engine templates
- ``bench_run.sh`` from shared benchmark template

This guarantees single-source-of-truth behavior: changing backend templates
automatically propagates to both k8s and sflow outputs.
"""

from __future__ import annotations

import math
import re
import shlex
from typing import Any

_SUPPORTED_BACKENDS = {"trtllm", "sglang", "vllm"}
_SFLOW_EXPR_RE = re.compile(r"\$\[\[(.+?)\]\]")
_SFLOW_VARIABLE_PROFILES = {"minimal", "expanded"}


def postprocess_sflow(rendered: str) -> str:
    r"""Replace ``$[[ expr ]]`` placeholders with sflow ``${{ expr }}`` syntax."""
    return _SFLOW_EXPR_RE.sub(r"${{\1}}", rendered)


def enrich_context_for_sflow(
    context: dict[str, Any],
    param_values: dict[str, Any],
    backend: str,
    rendered_templates: dict[str, str],
) -> dict[str, Any]:
    """Return a copy of *context* enriched with sflow-specific keys.

    All backend-specific logic (scripts, operator config, probes) is
    pre-computed here so the template contains zero backend conditionals
    in the server/benchmark task sections.
    """
    if backend not in _SUPPORTED_BACKENDS:
        return context

    ctx = dict(context)
    ctx["backend"] = backend
    wp = param_values.get("params", {})
    bench = param_values.get("BenchConfig", {})
    node_cfg = param_values.get("NodeConfig", {})
    worker_cfg = param_values.get("WorkerConfig", {})
    dyn = param_values.get("DynConfig", {})
    k8s = param_values.get("K8sConfig", {})
    sla = param_values.get("SlaConfig", {})
    sflow_cfg = param_values.get("SflowConfig", {})

    mode = dyn.get("mode") or ("disagg" if (wp.get("prefill") and wp.get("decode")) else "agg")
    gpn = _int(node_cfg.get("num_gpus_per_node"), 8)
    variable_profile = str(sflow_cfg.get("variable_profile") or "expanded").strip().lower()
    if variable_profile not in _SFLOW_VARIABLE_PROFILES:
        variable_profile = "expanded"

    ctx["sflow_mode"] = mode
    ctx["sflow_operator"] = f"dynamo_{backend}"
    ctx["sflow_gpus_per_node"] = gpn
    ctx["sflow_slurm_account"] = sflow_cfg.get("slurm_account") or "YOUR_ACCOUNT"
    ctx["sflow_slurm_partition"] = sflow_cfg.get("slurm_partition") or "YOUR_PARTITION"
    ctx["sflow_slurm_timelimit"] = sflow_cfg.get("slurm_timelimit")
    if ctx["sflow_slurm_timelimit"] in (None, ""):
        ctx["sflow_slurm_timelimit"] = 240
    ctx["sflow_aiperf_image"] = sflow_cfg.get("aiperf_image") or "python:3.12-slim"
    ctx["sflow_extra_frontend_args"] = sflow_cfg.get("extra_frontend_args") or ""
    ctx["sflow_variable_profile"] = variable_profile

    total_gpus = _total_gpus(wp, worker_cfg, mode)
    ctx["sflow_slurm_nodes"] = max(math.ceil(total_gpus / gpn), 1)
    ctx["sflow_concurrency"] = _resolve_sflow_concurrency(bench)
    ctx["sflow_dynamo_image"] = (
        sflow_cfg.get("dynamo_image") or k8s.get("k8s_image") or f"nvcr.io/nvidia/ai-dynamo/{backend}-runtime:latest"
    )

    # Inline engine-args content for trtllm artifacts
    ctx["sflow_agg_engine_content"] = rendered_templates.get("extra_engine_args_agg.yaml", "")
    ctx["sflow_prefill_engine_content"] = rendered_templates.get("extra_engine_args_prefill.yaml", "")
    ctx["sflow_decode_engine_content"] = rendered_templates.get("extra_engine_args_decode.yaml", "")
    ctx["sflow_bench_run_content"] = rendered_templates.get("bench_run.sh", "")

    # Probe pattern
    if backend == "trtllm":
        ctx["sflow_server_ready_pattern"] = "Setting PyTorch memory fraction"
    elif backend == "sglang":
        ctx["sflow_server_ready_pattern"] = "orker handler initialized"
    else:
        # vLLM: emit explicit marker before launch for deterministic readiness.
        ctx["sflow_server_ready_pattern"] = "Worker starting"

    # Per-role: operator config, script, resources
    roles = (
        [("agg", "agg", None)]
        if mode == "agg"
        else [
            ("prefill", "prefill", "prefill"),
            ("decode", "decode", "decode"),
        ]
    )
    for ctx_key, worker_key, disagg_mode in roles:
        rp = wp.get(worker_key, {})
        tp = _int(rp.get("tensor_parallel_size"), 1)
        prefix = _role_variable_prefix(worker_key, mode)
        role_workers = _int(worker_cfg.get(f"{worker_key}_workers"), _int(context.get(f"{worker_key}_workers"), 1))
        role_gpu = _int(context.get(f"{worker_key}_gpu"), 1)

        ctx[f"sflow_{ctx_key}_ntasks"] = tp if backend == "trtllm" else None
        ctx[f"sflow_{ctx_key}_ntasks_per_node"] = min(tp, gpn) if backend == "trtllm" else 1
        ctx[f"sflow_{ctx_key}_replicas_count"] = role_workers
        ctx[f"sflow_{ctx_key}_replicas_policy"] = "parallel"
        ctx[f"sflow_{ctx_key}_gpu_count"] = role_gpu
        if variable_profile == "expanded":
            ctx[f"sflow_{ctx_key}_replicas_count"] = f"$[[ variables.NUM_{prefix}_SERVERS ]]"
            ctx[f"sflow_{ctx_key}_replicas_policy"] = f"$[[ variables.{prefix}_REPLICAS_POLICY ]]"
            ctx[f"sflow_{ctx_key}_gpu_count"] = (
                f"$[[ (variables.{prefix}_TP_SIZE * variables.{prefix}_DP_SIZE * variables.{prefix}_PP_SIZE) "
                f"if variables.{prefix}_ENABLE_ATTENTION_DP == '' else "
                f"(variables.{prefix}_TP_SIZE * variables.{prefix}_PP_SIZE) ]]"
            )
            if backend == "trtllm":
                ctx[f"sflow_{ctx_key}_ntasks"] = f"$[[ variables.{prefix}_TP_SIZE ]]"
                ctx[f"sflow_{ctx_key}_ntasks_per_node"] = (
                    f"$[[ [ variables.{prefix}_TP_SIZE, variables.GPUS_PER_NODE ] | min ]]"
                )
        ctx[f"sflow_{ctx_key}_script"] = _build_server_script(
            backend,
            ctx_key,
            disagg_mode,
            ctx,
            rp,
            gpn,
        )

    # Benchmark script
    ctx["sflow_bench_script"] = _build_bench_script(
        context=context,
        sla=sla,
        bench_run_content=ctx["sflow_bench_run_content"],
        profile=variable_profile,
    )
    ctx["sflow_variables"] = _build_sflow_variables(
        backend=backend,
        mode=mode,
        profile=variable_profile,
        ctx=ctx,
        param_values=param_values,
        context=context,
    )

    return ctx


# ---------------------------------------------------------------------------
# Script builders — reuse rendered cli_args / engine_args from context
# ---------------------------------------------------------------------------


def _build_server_script(
    backend: str,
    role: str,
    disagg_mode: str | None,
    context: dict[str, Any],
    role_params: dict[str, Any],
    gpus_per_node: int,
) -> str:
    """Build the YAML script section (list of ``- item`` lines) for a server task."""
    if backend == "sglang":
        return _sglang_server_script(role, disagg_mode, context, role_params, gpus_per_node)
    if backend == "vllm":
        return _vllm_server_script(role, disagg_mode, context)
    return _trtllm_server_script(role, disagg_mode, context)


def _sglang_server_script(
    role: str,
    disagg_mode: str | None,
    context: dict[str, Any],
    role_params: dict[str, Any],
    gpus_per_node: int,
) -> str:
    cli_args = context.get(f"{role}_cli_args", "")
    expanded_profile = context.get("sflow_variable_profile") == "expanded"
    var_prefix = _role_variable_prefix(role, "agg" if disagg_mode is None else "disagg")
    if expanded_profile:
        cli_args = _bind_sglang_cli_args(cli_args, var_prefix)
    tp = _int(role_params.get("tensor_parallel_size"), 1)
    dp = _int(role_params.get("data_parallel_size"), 1)
    pp = _int(role_params.get("pipeline_parallel_size"), 1)
    attn_dp = bool(role_params.get("enable_attention_dp"))
    gpw = (tp * pp) if attn_dp else (tp * pp * dp)
    npw = max(gpw // gpus_per_node, 1)

    extra_var = {
        "agg": "EXTRA_AGG_ARGS",
        "prefill": "EXTRA_PREFILL_ARGS",
        "decode": "EXTRA_DECODE_ARGS",
    }[role]

    lines = [
        "- set -x",
        "- echo ${CUDA_VISIBLE_DEVICES}",
        "- export FIRST_CUDA_DEVICE=$(echo ${CUDA_VISIBLE_DEVICES} | cut -d',' -f1)",
    ]

    if npw > 1:
        lines.extend(
            [
                "- export NODE_RANK=${SLURM_NODEID}",
                "- export FIRST_NODE_IP=$(echo ${SFLOW_TASK_ASSIGNED_NODE_IPS} | cut -d',' -f1)",
                (
                    '- export MULTI_NODE_EXTRA_ARGS="'
                    "--dist-init-addr ${FIRST_NODE_IP}:29500 "
                    f"--nnodes {npw} "
                    '--node-rank ${NODE_RANK}"'
                ),
            ]
        )

    lines.extend(
        [
            "- export VLLM_NIXL_SIDE_CHANNEL_PORT=$((5557 + ${FIRST_CUDA_DEVICE}))",
            "- export DYN_SYSTEM_PORT=$((8082 + ${FIRST_CUDA_DEVICE}))",
        ]
    )

    # Launch command — reuses rendered cli_args (single source of truth)
    cmd_parts = [
        "python3 -m dynamo.sglang \\",
        "  --model-path $[[ artifacts.LOCAL_MODEL_PATH.path ]] \\",
        "  --served-model-name $[[ variables.SERVED_MODEL_NAME ]] \\",
        f"  {cli_args} \\",
    ]
    if disagg_mode:
        cmd_parts.append(f"  --disaggregation-mode {disagg_mode} \\")
        if "--disaggregation-transfer-backend" not in cli_args:
            cmd_parts.append("  --disaggregation-transfer-backend nixl \\")
        if disagg_mode == "prefill" and "--load-balance-method" not in cli_args:
            cmd_parts.append("  --load-balance-method round_robin \\")
        if disagg_mode == "decode" and "--prefill-round-robin-balance" not in cli_args:
            cmd_parts.append("  --prefill-round-robin-balance \\")
        cmd_parts.append(
            '  --disaggregation-bootstrap-port $(python3 -c "import socket; '
            "s=socket.socket(); s.bind(('', 0)); "
            'print(s.getsockname()[1]); s.close()") \\'
        )
    if npw > 1:
        cmd_parts.append("  ${MULTI_NODE_EXTRA_ARGS} \\")
    cmd_parts.append(f"  --host 0.0.0.0 ${{{extra_var}}}")

    lines.append("- |-")
    lines.extend(f"  {p}" for p in cmd_parts)

    return "\n".join(lines)


def _trtllm_server_script(
    role: str,
    disagg_mode: str | None,
    context: dict[str, Any],
) -> str:
    config_map = {"agg": "AGG_CONFIG", "prefill": "PREFILL_CONFIG", "decode": "DECODE_CONFIG"}
    config_name = config_map[role]
    extra_var = {
        "agg": "EXTRA_AGG_ARGS",
        "prefill": "EXTRA_PREFILL_ARGS",
        "decode": "EXTRA_DECODE_ARGS",
    }[role]

    lines = [
        "- set -x",
        "- echo ${CUDA_VISIBLE_DEVICES}",
        "- export TLLM_LOG_LEVEL=INFO",
        "- env | grep UCX",
        "- unset UCX_TLS",
    ]

    cmd_parts = [
        "trtllm-llmapi-launch python3 -m dynamo.trtllm \\",
        "  --model-path $[[ artifacts.LOCAL_MODEL_PATH.path ]] \\",
        "  --served-model-name $[[ variables.SERVED_MODEL_NAME ]] \\",
    ]
    if disagg_mode:
        cmd_parts.append(f"  --disaggregation-mode {disagg_mode} \\")
    cmd_parts.append(f"  --extra-engine-args $[[ artifacts.{config_name}.path ]] ${{{extra_var}}}")

    lines.append("- |-")
    lines.extend(f"  {p}" for p in cmd_parts)

    return "\n".join(lines)


def _vllm_server_script(
    role: str,
    disagg_mode: str | None,
    context: dict[str, Any],
) -> str:
    cli_args = context.get(f"{role}_cli_args", "")
    extra_var = {
        "agg": "EXTRA_AGG_ARGS",
        "prefill": "EXTRA_PREFILL_ARGS",
        "decode": "EXTRA_DECODE_ARGS",
    }[role]

    lines = [
        "- set -x",
        '- echo "Worker starting"',
        "- echo ${CUDA_VISIBLE_DEVICES}",
    ]

    cmd_parts = [
        "python3 -m dynamo.vllm \\",
        "  --model $[[ artifacts.LOCAL_MODEL_PATH.path ]] \\",
    ]
    if cli_args:
        cmd_parts.append(f"  {cli_args} \\")

    if disagg_mode == "prefill":
        cmd_parts.append("  --is-prefill-worker \\")
        cmd_parts.append('  --kv-transfer-config \'{"kv_connector":"NixlConnector","kv_role":"kv_both"}\' \\')
    elif disagg_mode == "decode":
        cmd_parts.append("  --is-decode-worker \\")
        cmd_parts.append('  --kv-transfer-config \'{"kv_connector":"NixlConnector","kv_role":"kv_both"}\' \\')

    cmd_parts.append(f"  ${{{extra_var}}}")

    lines.append("- |-")
    lines.extend(f"  {p}" for p in cmd_parts)
    return "\n".join(lines)


def _build_bench_script(
    context: dict[str, Any],
    sla: dict[str, Any],
    bench_run_content: str,
    profile: str,
) -> str:
    """Build the benchmark aiperf script section."""
    if bench_run_content:
        lines = [
            "- set -x",
            "- pip install aiperf==0.3.0",
            "- export COLUMNS=200",
            "- export BENCH_ARTIFACT_DIR=${SFLOW_WORKFLOW_OUTPUT_DIR}/aiperf_concurrency_${CONCURRENCY}",
            "- export AICONFIGURATOR_BENCH_CONCURRENCY=${CONCURRENCY}",
            "- export AICONFIGURATOR_BENCH_MODEL=$[[ variables.SERVED_MODEL_NAME ]]",
            "- export AICONFIGURATOR_BENCH_TOKENIZER=$[[ variables.SERVED_MODEL_NAME ]]",
            "- export AICONFIGURATOR_BENCH_ENDPOINT_URL=http://$[[ variables.HEAD_NODE_IP ]]:8000",
            "- export AICONFIGURATOR_BENCH_ISL=$[[ variables.ISL ]]",
            "- export AICONFIGURATOR_BENCH_OSL=$[[ variables.OSL ]]",
            "- export AICONFIGURATOR_BENCH_MULTI_ROUND=$[[ variables.MULTI_ROUND ]]",
        ]
        lines.extend(["- bash $[[ artifacts.BENCH_RUN.path ]]", '- echo "Benchmarking finished"'])
        return "\n".join(lines)

    lines = [
        "- set -x",
        "- pip install aiperf==0.3.0",
        "- export COLUMNS=200",
        "- |-",
    ]
    cmd = [
        "  aiperf profile --artifact-dir ${SFLOW_WORKFLOW_OUTPUT_DIR}/aiperf_concurrency_${CONCURRENCY} \\",
        "    --model $[[ variables.SERVED_MODEL_NAME ]] \\",
        "    --tokenizer $[[ artifacts.LOCAL_MODEL_PATH.path ]] \\",
        "    --endpoint-type chat \\",
        "    --endpoint /v1/chat/completions \\",
        "    --streaming \\",
        "    --url http://$[[ variables.HEAD_NODE_IP ]]:8000 \\",
        "    --synthetic-input-tokens-mean $[[ variables.ISL ]] \\",
        "    --synthetic-input-tokens-stddev 0 \\",
        "    --output-tokens-mean $[[ variables.OSL ]] \\",
        "    --output-tokens-stddev 0 \\",
        '    --extra-inputs "max_tokens:$[[ variables.OSL ]]" \\',
        '    --extra-inputs "min_tokens:$[[ variables.OSL ]]" \\',
        '    --extra-inputs "ignore_eos:true" \\',
        '    --extra-inputs "{\\"nvext\\":{\\"ignore_eos\\":true}}" \\',
        '    --extra-inputs "repetition_penalty:1.0" \\',
        '    --extra-inputs "temperature: 0.0" \\',
        "    --concurrency ${CONCURRENCY} \\",
        "    --request-count $(($[[ variables.MULTI_ROUND ]]*${CONCURRENCY})) \\",
        "    --warmup-request-count ${CONCURRENCY} \\",
        "    --num-dataset-entries $(($[[ variables.MULTI_ROUND ]]*${CONCURRENCY})) \\",
        "    --random-seed 100 -H 'Authorization: Bearer NOT USED' -H 'Accept: text/event-stream' \\",
        "    --record-processors 8 \\",
        "    --ui simple",
    ]
    lines.extend(cmd)
    lines.append('- echo "Benchmarking finished"')

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Variables translator
# ---------------------------------------------------------------------------


def _build_sflow_variables(
    backend: str,
    mode: str,
    profile: str,
    ctx: dict[str, Any],
    param_values: dict[str, Any],
    context: dict[str, Any],
) -> list[dict[str, Any]]:
    bench = param_values.get("BenchConfig", {}) or {}
    wp = param_values.get("params", {}) or {}
    wc = param_values.get("WorkerConfig", {}) or {}
    model_path = context.get("model_path", "")
    local_model_uri = _default_local_model_uri(model_path)

    def _var(
        name: str,
        description: str,
        value: Any,
        vtype: str | None = None,
        domain: list[Any] | None = None,
    ) -> dict[str, Any]:
        out: dict[str, Any] = {"name": name, "description": description, "value": value}
        if vtype:
            out["type"] = vtype
        if domain:
            out["domain"] = domain
        return out

    vars_out: list[dict[str, Any]] = [
        _var("SLURM_ACCOUNT", "SLURM account", ctx["sflow_slurm_account"]),
        _var("SLURM_PARTITION", "SLURM partition", ctx["sflow_slurm_partition"]),
        _var("SLURM_TIMELIMIT", "SLURM time limit", ctx["sflow_slurm_timelimit"], "integer"),
        _var("GPUS_PER_NODE", "GPUs per node", ctx["sflow_gpus_per_node"], "integer"),
        _var("SLURM_NODES", "Number of nodes", ctx["sflow_slurm_nodes"], "integer"),
        _var("SERVED_MODEL_NAME", "Served model name", model_path),
        _var("MODEL_NAME", "Model path", model_path),
        _var("LOCAL_MODEL_PATH", "Local model artifact URI", local_model_uri),
        _var("EXTRA_FRONTEND_ARGS", "Extra frontend arguments", ctx["sflow_extra_frontend_args"]),
    ]

    # role-specific extra args are runtime override knobs and should always exist.
    if mode == "agg":
        vars_out.append(_var("EXTRA_AGG_ARGS", "Extra aggregated server arguments", ""))
    else:
        vars_out.append(_var("EXTRA_PREFILL_ARGS", "Extra prefill arguments", ""))
        vars_out.append(_var("EXTRA_DECODE_ARGS", "Extra decode arguments", ""))

    vars_out.extend(
        [
            _var("ISL", "Input sequence length", _int(bench.get("isl"), 4000), "integer"),
            _var("OSL", "Output sequence length", _int(bench.get("osl"), 1000), "integer"),
            _var("MULTI_ROUND", "Number of benchmark rounds", 20, "integer"),
            _var("CONCURRENCY", "Concurrency", ctx["sflow_concurrency"], "integer", domain=[ctx["sflow_concurrency"]]),
            _var("AIPERF_IMAGE", "AIPerf container image", ctx["sflow_aiperf_image"]),
            _var("DYNAMO_IMAGE", "Dynamo container image", ctx["sflow_dynamo_image"]),
        ]
    )

    if profile == "minimal":
        return vars_out

    # Expanded profile: preserve minimal variables and expose backend knobs for readability/compat.
    role_specs = (
        [("agg", "AGG", "aggregated")]
        if mode == "agg"
        else [("prefill", "CTX", "context/prefill"), ("decode", "GEN", "generation/decode")]
    )
    for role_key, prefix, role_label in role_specs:
        rp = wp.get(role_key, {}) or {}
        cli_args = context.get(f"{role_key}_cli_args", "") or ""
        workers = _int(wc.get(f"{role_key}_workers"), 1)
        tp = _int(rp.get("tensor_parallel_size"), _int(_extract_option_value(cli_args, "--tensor-parallel-size"), 1))
        pp = _int(
            rp.get("pipeline_parallel_size"),
            _int(_extract_option_value(cli_args, "--pipeline-parallel-size"), 1),
        )
        dp = _int(rp.get("data_parallel_size"), _int(_extract_option_value(cli_args, "--data-parallel-size"), 1))
        ep = _int(
            rp.get("moe_expert_parallel_size"),
            _int(_extract_option_value(cli_args, "--expert-parallel-size"), 1),
        )
        max_batch = _int(rp.get("max_batch_size"), _int(_extract_option_value(cli_args, "--max-running-requests"), 1))
        max_num_tokens = _int(
            rp.get("max_num_tokens")
            or rp.get("max_prefill_tokens")
            or _extract_option_value(cli_args, "--max-prefill-tokens")
            or _extract_option_value(cli_args, "--max-total-tokens"),
            0,
        )
        max_seq_len = _int(
            rp.get("max_seq_len") or _extract_option_value(cli_args, "--context-length"),
            0,
        )
        enable_attention = bool(rp.get("enable_attention_dp")) or ("--enable-dp-attention" in cli_args)
        free_mem_fraction = (
            rp.get("free_gpu_memory_fraction") or _extract_option_value(cli_args, "--mem-fraction-static") or "0.9"
        )

        vars_out.append(_var(f"NUM_{prefix}_SERVERS", f"Number of {role_label} servers", workers, "integer"))
        vars_out.append(_var(f"{prefix}_REPLICAS_POLICY", f"{role_label} replicas policy", "parallel"))
        vars_out.append(_var(f"{prefix}_TP_SIZE", f"{role_label} tensor parallel size", tp, "integer"))
        vars_out.append(_var(f"{prefix}_PP_SIZE", f"{role_label} pipeline parallel size", pp, "integer"))
        vars_out.append(_var(f"{prefix}_DP_SIZE", f"{role_label} data parallel size", dp, "integer"))
        vars_out.append(_var(f"{prefix}_EP_SIZE", f"{role_label} expert parallel size", ep, "integer"))
        vars_out.append(_var(f"{prefix}_BATCH_SIZE", f"{role_label} batch size", max_batch, "integer"))
        vars_out.append(
            _var(f"{prefix}_MAX_NUM_TOKENS", f"{role_label} max number of tokens", max_num_tokens, "integer")
        )
        vars_out.append(_var(f"{prefix}_MAX_SEQ_LEN", f"{role_label} max sequence length", max_seq_len, "integer"))
        vars_out.append(
            _var(
                f"{prefix}_ENABLE_ATTENTION_DP",
                f"{role_label} enable attention DP",
                "--enable-dp-attention" if enable_attention else "",
            )
        )
        vars_out.append(
            _var(
                f"{prefix}_FREE_GPU_MEMORY_FRACTION",
                f"{role_label} free GPU memory fraction",
                free_mem_fraction,
            )
        )

        if backend == "sglang":
            graph_bs = " ".join(_extract_multi_option_values(cli_args, "--cuda-graph-bs"))
            vars_out.append(
                _var(
                    f"{prefix}_CUDA_GRAPH_BS",
                    f"{role_label} CUDA graph batch sizes (space-separated list)",
                    graph_bs,
                )
            )
            spec_algo = _extract_option_value(cli_args, "--speculative-algorithm") or "NEXTN"
            vars_out.append(_var(f"{prefix}_SPECULATIVE_ALGORITHM", f"{role_label} speculative algorithm", spec_algo))

    if backend == "sglang" and mode != "agg":
        compat_spec = _extract_option_value(context.get("decode_cli_args", "") or "", "--speculative-algorithm")
        if not compat_spec:
            compat_spec = _extract_option_value(context.get("prefill_cli_args", "") or "", "--speculative-algorithm")
        vars_out.append(
            _var(
                "AGG_SPECULATIVE_ALGORITHM",
                "Compatibility speculative algorithm alias",
                compat_spec or "NEXTN",
            )
        )

    # Expose kv-cache dtype in expanded profile for compatibility with common reference style.
    primary_role = "agg" if mode == "agg" else "decode"
    kv_dtype = (wp.get(primary_role) or {}).get("kv_cache_dtype") or _extract_option_value(
        context.get(f"{primary_role}_cli_args", "") or "",
        "--kv-cache-dtype",
    )
    vars_out.append(_var("KV_CACHE_DTYPE", "KV cache dtype", kv_dtype or "auto"))

    return vars_out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _role_variable_prefix(role_key: str, mode: str) -> str:
    if mode == "agg":
        return "AGG"
    if role_key == "prefill":
        return "CTX"
    if role_key == "decode":
        return "GEN"
    return role_key.upper()


def _bind_sglang_cli_args(cli_args: str, prefix: str) -> str:
    tokens = shlex.split(cli_args) if cli_args else []
    if not tokens:
        return cli_args

    value_bindings = {
        "--tensor-parallel-size": f"$[[ variables.{prefix}_TP_SIZE ]]",
        "--pipeline-parallel-size": f"$[[ variables.{prefix}_PP_SIZE ]]",
        "--data-parallel-size": f"$[[ variables.{prefix}_DP_SIZE ]]",
        "--expert-parallel-size": f"$[[ variables.{prefix}_EP_SIZE ]]",
        "--max-running-requests": f"$[[ variables.{prefix}_BATCH_SIZE ]]",
        "--max-prefill-tokens": f"$[[ variables.{prefix}_MAX_NUM_TOKENS ]]",
        "--max-total-tokens": f"$[[ variables.{prefix}_MAX_NUM_TOKENS ]]",
        "--context-length": f"$[[ variables.{prefix}_MAX_SEQ_LEN ]]",
        "--mem-fraction-static": f"$[[ variables.{prefix}_FREE_GPU_MEMORY_FRACTION ]]",
        "--kv-cache-dtype": "$[[ variables.KV_CACHE_DTYPE ]]",
        "--cuda-graph-bs": f"$[[ variables.{prefix}_CUDA_GRAPH_BS ]]",
        "--cuda-graph-max-bs": f"$[[ variables.{prefix}_BATCH_SIZE ]]",
        "--speculative-algorithm": f"$[[ variables.{prefix}_SPECULATIVE_ALGORITHM ]]",
    }
    attention_var = f"$[[ variables.{prefix}_ENABLE_ATTENTION_DP ]]"

    out: list[str] = []
    present_flags: set[str] = set()
    attention_bound = False
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        # SGLang accepts moe_dense_tp_size 1 or None only; expert TP is tensor_parallel_size.
        if token.startswith("--moe-dense-tp-size"):
            if "=" in token:
                idx += 1
                continue
            idx += 1
            if idx < len(tokens) and not tokens[idx].startswith("--"):
                idx += 1
            continue
        if token == "--enable-dp-attention":
            out.append(attention_var)
            attention_bound = True
            present_flags.add(token)
            idx += 1
            continue

        if "=" in token and token.startswith("--"):
            flag, _val = token.split("=", 1)
            if flag in value_bindings:
                out.append(flag)
                out.append(value_bindings[flag])
                present_flags.add(flag)
                idx += 1
                continue

        if token in value_bindings:
            out.append(token)
            present_flags.add(token)
            if idx + 1 < len(tokens) and not tokens[idx + 1].startswith("--"):
                if token == "--cuda-graph-bs":
                    while idx + 1 < len(tokens) and not tokens[idx + 1].startswith("--"):
                        idx += 1
                else:
                    idx += 1
            out.append(value_bindings[token])
            idx += 1
            continue

        out.append(token)
        if token.startswith("--") and idx + 1 < len(tokens) and not tokens[idx + 1].startswith("--"):
            out.append(tokens[idx + 1])
            idx += 2
            continue
        idx += 1

    required_bindings = [
        ("--tensor-parallel-size", f"$[[ variables.{prefix}_TP_SIZE ]]"),
        ("--pipeline-parallel-size", f"$[[ variables.{prefix}_PP_SIZE ]]"),
        ("--data-parallel-size", f"$[[ variables.{prefix}_DP_SIZE ]]"),
        ("--expert-parallel-size", f"$[[ variables.{prefix}_EP_SIZE ]]"),
        ("--max-running-requests", f"$[[ variables.{prefix}_BATCH_SIZE ]]"),
        ("--max-prefill-tokens", f"$[[ variables.{prefix}_MAX_NUM_TOKENS ]]"),
        ("--kv-cache-dtype", "$[[ variables.KV_CACHE_DTYPE ]]"),
    ]
    for flag, value in required_bindings:
        if flag not in present_flags:
            out.extend([flag, value])

    if not attention_bound:
        out.append(attention_var)

    return " ".join(token if token.startswith("$[[") and token.endswith("]]") else shlex.quote(token) for token in out)


def _default_local_model_uri(model_path: Any) -> str:
    raw = str(model_path or "").strip()
    if not raw:
        return "fs:///path/to/local/model"
    if "://" in raw:
        return raw
    if raw.startswith(("~", "/", "./", "../")):
        return f"fs://{raw}"
    # Non-local identifiers (for example, HuggingFace repo IDs) are not valid fs:// artifact URIs.
    return "fs:///path/to/local/model"


def _resolve_sflow_concurrency(bench: dict[str, Any]) -> int:
    """Resolve a single benchmark concurrency value for sflow.

    sflow benchmark runs should use the estimated concurrency as one scalar
    value, not the multi-point BenchConfig.concurrency sweep list.
    """
    estimated = bench.get("estimated_concurrency")
    if isinstance(estimated, (list, tuple)):
        estimated = estimated[0] if estimated else None

    resolved = _int(estimated, 0)
    if resolved > 0:
        return resolved
    return 64


def _int(val: Any, default: int = 0) -> int:
    try:
        if val is None:
            return default
        return int(val)
    except (TypeError, ValueError):
        return default


def _total_gpus(wp: dict[str, Any], wc: dict[str, Any], mode: str) -> int:
    """Compute total GPUs from WorkerConfig (pre-rule) with params fallback."""

    def gpw_from_params(rp: dict[str, Any]) -> int:
        return _int(rp.get("gpus_per_worker")) or (
            _int(rp.get("tensor_parallel_size"), 1)
            * _int(rp.get("pipeline_parallel_size"), 1)
            * _int(rp.get("data_parallel_size"), 1)
        )

    if mode == "agg":
        gpu = _int(wc.get("agg_gpus_per_worker")) or gpw_from_params(wp.get("agg", {}))
        return _int(wc.get("agg_workers"), 1) * gpu
    p_gpu = _int(wc.get("prefill_gpus_per_worker")) or gpw_from_params(wp.get("prefill", {}))
    d_gpu = _int(wc.get("decode_gpus_per_worker")) or gpw_from_params(wp.get("decode", {}))
    return _int(wc.get("prefill_workers"), 1) * p_gpu + _int(wc.get("decode_workers"), 1) * d_gpu


def _extract_option_value(cli_args: str, flag: str) -> str:
    tokens = shlex.split(cli_args) if cli_args else []
    for idx, token in enumerate(tokens):
        if token == flag and idx + 1 < len(tokens):
            return tokens[idx + 1]
        if token.startswith(f"{flag}="):
            return token.split("=", 1)[1]
    return ""


def _extract_multi_option_values(cli_args: str, flag: str) -> list[str]:
    tokens = shlex.split(cli_args) if cli_args else []
    for idx, token in enumerate(tokens):
        if token == flag:
            values: list[str] = []
            pos = idx + 1
            while pos < len(tokens) and not tokens[pos].startswith("--"):
                values.append(tokens[pos])
                pos += 1
            return values
        if token.startswith(f"{flag}="):
            return token.split("=", 1)[1].split()
    return []
