# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed per-backend K8s (DynamoGraphDeployment) builders.

THE rendering path for ``k8s_deploy.yaml`` (Stage-3 cutover). Each
``_populate_<backend>`` function consumes the render context built by
:func:`aiconfigurator.generator.rendering.engine.build_k8s_render_context`
(the same context the backend's ``k8s_deploy.yaml.j2`` template received
before the templates were deleted at the cutover; see git history) and
constructs typed DGD documents semantically equal to the old template output.

Design constraint (design doc Section 3.1, PR #314 -> #340 history): the DGD
object MODEL is shared, but the POPULATION LOGIC is strictly per backend.
The three populate functions intentionally duplicate structure-building code
instead of sharing helpers, mirroring each Jinja template top-to-bottom so a
template edit maps to exactly one builder.
"""
from __future__ import annotations

import copy
import json
from typing import Any

from .dgd_model import DGD, ConfigMapDoc, DGDService, ExtraPodSpec, MainContainer


def build_dgd(
    context: dict[str, Any], backend: str, resolved_facts: Any = None
) -> list[Any]:
    """Build the list of typed K8s documents for ``backend`` from a render context.

    Document order matches the template stream order (trtllm emits the engine
    ConfigMap before the DGD when ``k8s_engine_mode == 'configmap'``).

    ``resolved_facts`` is the optional ``ResolvedFacts`` for the request (typed
    ``Any`` to avoid an import cycle). It is threaded through to each per-backend
    ``_populate_<backend>`` but NOT yet read or emitted from — wiring only, so
    output stays byte-identical. Defaults to ``None`` to preserve every existing
    caller.
    """
    populate = {
        "vllm": _populate_vllm,
        "sglang": _populate_sglang,
        "trtllm": _populate_trtllm,
    }.get(backend)
    if populate is None:
        raise ValueError(f"No typed K8s builder for backend: {backend}")
    return populate(context, resolved_facts=resolved_facts)


# ---------------------------------------------------------------------------
# vLLM — mirrors backend_templates/vllm/k8s_deploy.yaml.j2 top to bottom.
# ---------------------------------------------------------------------------


def _populate_vllm(context: dict[str, Any], resolved_facts: Any = None) -> list[Any]:
    k8s = context.get("K8sConfig", {}) or {}
    dyn = context.get("DynConfig", {}) or {}
    svc_cfg = context.get("ServiceConfig", {}) or {}

    # {%- set runtime_working_dir = ... -%}
    working_dir = context.get("working_dir")
    if working_dir and working_dir != "/workspace/components/backends/vllm":
        runtime_working_dir = working_dir
    else:
        runtime_working_dir = "/workspace/examples/backends/vllm"
    # {%- set k8s_use_model_cache / k8s_model_cache_pvc / mount -%}
    model_cache_input = str(k8s.get("k8s_model_cache") or "").strip()
    use_model_cache = model_cache_input != ""
    model_cache_pvc = model_cache_input if use_model_cache else "model-cache"
    model_cache_mount = "/workspace/model_cache"
    enable_router = bool(dyn.get("enable_router") or False)
    etcd_endpoints = k8s.get("k8s_etcd_endpoints")
    hf_home = k8s.get("k8s_hf_home")
    image_pull_secret = k8s.get("k8s_image_pull_secret")

    # Phase 4b-1 pod facts (hardware/transport). Guard on resolved_facts and
    # key presence so a None facts / missing key emits nothing (keeps the
    # crosscheck / no-fact callers byte-identical).
    hw_facts = getattr(resolved_facts, "hardware", None) if resolved_facts is not None else None
    tr_facts = getattr(resolved_facts, "transport", None) if resolved_facts is not None else None
    node_selector_fact = hw_facts.get("node_selector") if isinstance(hw_facts, dict) else None
    tolerations_fact = (hw_facts.get("tolerations") or None) if isinstance(hw_facts, dict) else None

    def worker_main_env(role: str | None) -> list[dict[str, str]] | None:
        # NCCL (hardware) first, then transport env not already set (hardware wins).
        if not isinstance(hw_facts, dict):
            return None
        out: list[dict[str, str]] = []
        seen: set[str] = set()
        for k, v in (hw_facts.get("nccl_env") or {}).items():
            out.append({"name": k, "value": str(v)})
            seen.add(k)
        tr_env = tr_facts.get("env") if isinstance(tr_facts, dict) else None
        for k, v in (tr_env or {}).items():
            if k in seen:
                continue
            out.append({"name": k, "value": str(v)})
            seen.add(k)
        return out or None

    def worker_shared_memory(role: str | None) -> dict[str, str] | None:
        if not isinstance(hw_facts, dict):
            return None
        shm = hw_facts.get("shared_memory")
        if not isinstance(shm, dict):
            return None
        if role == "decode" and "disagg_decode" in shm:
            sz = shm["disagg_decode"]
        elif "default" in shm:
            sz = shm["default"]
        else:
            return None
        return {"size": sz}

    # macro render_worker(component_name, role, replicas, gpu, cli_args_list)
    def render_worker(role: str | None, replicas: Any, gpu: Any, cli_args_list: Any) -> DGDService:
        envs = None
        if hf_home or etcd_endpoints:
            envs = []
            if etcd_endpoints:
                envs.append({"name": "ETCD_ENDPOINTS", "value": etcd_endpoints})
            if hf_home:
                envs.append({"name": "HF_HOME", "value": hf_home})

        volumes = None
        if use_model_cache:
            volumes = [{"name": "model-cache", "persistentVolumeClaim": {"claimName": model_cache_pvc}}]
        image_pull_secrets = None
        if image_pull_secret:
            image_pull_secrets = [{"name": image_pull_secret}]

        volume_mounts = None
        if use_model_cache:
            volume_mounts = [{"name": "model-cache", "mountPath": model_cache_mount}]

        args: list[str] = ["--model", str(svc_cfg.get("model_path"))]
        # Gate B fix (served-model-name): the vllm cli_args templates now emit
        # `--served-model-name <ServiceConfig.served_model_name>` as the FIRST
        # cli_args token (guarded on a non-empty value), so extending with
        # cli_args_list places it directly after --model — same relative
        # placement as the sglang worker script. Do NOT also insert it here:
        # cli_args_list is the rendered template output, and an explicit
        # insertion would duplicate the flag.
        args.extend(cli_args_list or [])
        if enable_router:
            port = svc_cfg.get("dyn_vllm_kv_event_port") or 20081
            args.append("--kv-events-config")
            args.append(
                '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:'
                f"{port}"
                '","enable_kv_cache_events":true}'
            )
        if role == "prefill":
            args.extend(
                ["--is-prefill-worker", "--kv-transfer-config", '{"kv_connector":"NixlConnector","kv_role":"kv_both"}']
            )
        elif role == "decode":
            args.extend(
                ["--is-decode-worker", "--kv-transfer-config", '{"kv_connector":"NixlConnector","kv_role":"kv_both"}']
            )

        return DGDService(
            env_from_secret="hf-token-secret",
            envs=envs,
            component_type="worker",
            sub_component_type=role,
            replicas=replicas if replicas is not None else 1,
            resources={"limits": {"gpu": str(gpu)}},
            shared_memory=worker_shared_memory(role),
            extra_pod_spec=ExtraPodSpec(
                volumes=volumes,
                image_pull_secrets=image_pull_secrets,
                node_selector=copy.deepcopy(node_selector_fact),
                tolerations=copy.deepcopy(tolerations_fact),
                main_container=MainContainer(
                    image=k8s.get("k8s_image"),
                    working_dir=runtime_working_dir,
                    image_pull_policy="IfNotPresent",
                    volume_mounts=volume_mounts,
                    command=["python3", "-m", "dynamo.vllm"],
                    args=args,
                    env=worker_main_env(role),
                ),
            ),
        )

    # Frontend (template emits envs AFTER extraPodSpec -> keep in extra to
    # preserve emission order; semantic equality is unaffected either way).
    # Gate B fix: when a model-cache PVC is configured the frontend mounts it
    # too (same volume/mount as the workers, mirroring the trtllm frontend) —
    # kube-discovery model cards reference in-container file paths, so the
    # frontend must read tokenizer/config to materialize local-path models.
    fe_volumes = None
    fe_volume_mounts = None
    if use_model_cache:
        fe_volumes = [{"name": "model-cache", "persistentVolumeClaim": {"claimName": model_cache_pvc}}]
        fe_volume_mounts = [{"name": "model-cache", "mountPath": model_cache_mount}]
    frontend = DGDService(
        env_from_secret="hf-token-secret",
        component_type="frontend",
        replicas=context.get("frontend_replicas", 1),
        extra_pod_spec=ExtraPodSpec(
            volumes=fe_volumes,
            image_pull_secrets=[{"name": image_pull_secret}] if image_pull_secret else None,
            node_selector=copy.deepcopy(node_selector_fact),
            tolerations=copy.deepcopy(tolerations_fact),
            main_container=MainContainer(
                image=k8s.get("k8s_image"),
                image_pull_policy="IfNotPresent",
                volume_mounts=fe_volume_mounts,
            ),
        ),
    )
    if enable_router or hf_home or etcd_endpoints:
        fe_envs = []
        if etcd_endpoints:
            fe_envs.append({"name": "ETCD_ENDPOINTS", "value": etcd_endpoints})
        if enable_router:
            fe_envs.append({"name": "DYN_ROUTER_MODE", "value": "kv"})
        if hf_home:
            fe_envs.append({"name": "HF_HOME", "value": hf_home})
        frontend.extra["envs"] = fe_envs

    services: dict[str, DGDService] = {"Frontend": frontend}
    mode = dyn.get("mode", "disagg") or "disagg"
    if mode == "agg":
        services["VllmWorker"] = render_worker(
            None, context.get("agg_workers"), context.get("agg_gpu"), context.get("agg_cli_args_list") or []
        )
    else:
        services["VllmPrefillWorker"] = render_worker(
            "prefill",
            context.get("prefill_workers"),
            context.get("prefill_gpu"),
            context.get("prefill_cli_args_list") or [],
        )
        services["VllmDecodeWorker"] = render_worker(
            "decode",
            context.get("decode_workers"),
            context.get("decode_gpu"),
            context.get("decode_cli_args_list") or [],
        )

    dgd = DGD(name=context.get("name"), namespace=k8s.get("k8s_namespace"), services=services)
    return [dgd]


# ---------------------------------------------------------------------------
# SGLang — mirrors backend_templates/sglang/k8s_deploy.yaml.j2 top to bottom.
# ---------------------------------------------------------------------------


def _populate_sglang(context: dict[str, Any], resolved_facts: Any = None) -> list[Any]:
    k8s = context.get("K8sConfig", {}) or {}
    dyn = context.get("DynConfig", {}) or {}
    svc_cfg = context.get("ServiceConfig", {}) or {}

    enable_router = bool(dyn.get("enable_router") or False)
    runtime_working_dir = context.get("working_dir") or "/workspace/components/backends/sglang"
    model_cache_input = str(k8s.get("k8s_model_cache") or "").strip()
    use_model_cache = model_cache_input != ""
    model_cache_pvc = model_cache_input if use_model_cache else "model-cache"
    model_cache_mount = "/workspace/model_cache"
    etcd_endpoints = k8s.get("k8s_etcd_endpoints")
    hf_home = k8s.get("k8s_hf_home")
    image_pull_secret = k8s.get("k8s_image_pull_secret")
    mode = dyn.get("mode", "disagg") or "disagg"

    # Phase 4b-1 pod facts (hardware/transport). Guard on resolved_facts and
    # key presence so a None facts / missing key emits nothing (keeps the
    # crosscheck / no-fact callers byte-identical). Replicated per backend (no
    # shared helper, per PR#314->#340 rule).
    hw_facts = getattr(resolved_facts, "hardware", None) if resolved_facts is not None else None
    tr_facts = getattr(resolved_facts, "transport", None) if resolved_facts is not None else None
    node_selector_fact = hw_facts.get("node_selector") if isinstance(hw_facts, dict) else None
    tolerations_fact = (hw_facts.get("tolerations") or None) if isinstance(hw_facts, dict) else None

    def worker_main_env(role: str | None) -> list[dict[str, str]] | None:
        if not isinstance(hw_facts, dict):
            return None
        out: list[dict[str, str]] = []
        seen: set[str] = set()
        for k, v in (hw_facts.get("nccl_env") or {}).items():
            out.append({"name": k, "value": str(v)})
            seen.add(k)
        tr_env = tr_facts.get("env") if isinstance(tr_facts, dict) else None
        for k, v in (tr_env or {}).items():
            if k in seen:
                continue
            out.append({"name": k, "value": str(v)})
            seen.add(k)
        return out or None

    def worker_shared_memory(role: str | None) -> dict[str, str] | None:
        if not isinstance(hw_facts, dict):
            return None
        shm = hw_facts.get("shared_memory")
        if not isinstance(shm, dict):
            return None
        if role == "decode" and "disagg_decode" in shm:
            sz = shm["disagg_decode"]
        elif "default" in shm:
            sz = shm["default"]
        else:
            return None
        return {"size": sz}

    # macro render_worker(component_name, role, replicas, gpu, cli_args)
    def render_worker(role: str | None, replicas: Any, gpu: Any, cli_args: Any) -> DGDService:
        envs = None
        if hf_home or etcd_endpoints:
            envs = []
            if etcd_endpoints:
                envs.append({"name": "ETCD_ENDPOINTS", "value": etcd_endpoints})
            if hf_home:
                envs.append({"name": "HF_HOME", "value": hf_home})

        volumes = None
        if use_model_cache:
            volumes = [{"name": "model-cache", "persistentVolumeClaim": {"claimName": model_cache_pvc}}]
        image_pull_secrets = None
        if image_pull_secret:
            image_pull_secrets = [{"name": image_pull_secret}]
        volume_mounts = None
        if use_model_cache:
            volume_mounts = [{"name": "model-cache", "mountPath": model_cache_mount}]

        # The template emits a single block-scalar shell script; reproduce it
        # line-for-line (block scalar '|' clips to one trailing newline).
        lines = [
            "set -euo pipefail",
            "args=(",
            f'  --model-path "{svc_cfg.get("model_path")}"',
            f'  --served-model-name "{svc_cfg.get("served_model_name")}"',
            f"  {cli_args}",
            ")",
        ]
        if mode != "agg":
            lines.append('args+=(--host "0.0.0.0")')
        if role == "prefill":
            lines.append("args+=(--disaggregation-mode prefill)")
        elif role == "decode":
            lines.append("args+=(--disaggregation-mode decode)")
        if enable_router:
            port = svc_cfg.get("sglang_kv_event_port") or 5557
            lines.append(
                "args+=(--kv-events-config '"
                '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:'
                f"{port}"
                "\"}')"
            )
        lines.append('exec python3 -m dynamo.sglang "${args[@]}"')
        script = "\n".join(lines) + "\n"

        return DGDService(
            env_from_secret="hf-token-secret",
            envs=envs,
            component_type="worker",
            sub_component_type=role,
            replicas=replicas if replicas is not None else 1,
            resources={"limits": {"gpu": str(gpu)}},
            shared_memory=worker_shared_memory(role),
            extra_pod_spec=ExtraPodSpec(
                volumes=volumes,
                image_pull_secrets=image_pull_secrets,
                node_selector=copy.deepcopy(node_selector_fact),
                tolerations=copy.deepcopy(tolerations_fact),
                main_container=MainContainer(
                    image=k8s.get("k8s_image"),
                    working_dir=runtime_working_dir,
                    image_pull_policy="IfNotPresent",
                    volume_mounts=volume_mounts,
                    command=["/bin/bash", "-c"],
                    args=[script],
                    env=worker_main_env(role),
                ),
            ),
        )

    # Gate B fix: when a model-cache PVC is configured the frontend mounts it
    # too (same volume/mount as the workers, mirroring the trtllm frontend) —
    # kube-discovery model cards reference in-container file paths, so the
    # frontend must read tokenizer/config to materialize local-path models.
    fe_volumes = None
    fe_volume_mounts = None
    if use_model_cache:
        fe_volumes = [{"name": "model-cache", "persistentVolumeClaim": {"claimName": model_cache_pvc}}]
        fe_volume_mounts = [{"name": "model-cache", "mountPath": model_cache_mount}]
    frontend = DGDService(
        env_from_secret="hf-token-secret",
        component_type="frontend",
        replicas=context.get("frontend_replicas", 1),
        extra_pod_spec=ExtraPodSpec(
            volumes=fe_volumes,
            image_pull_secrets=[{"name": image_pull_secret}] if image_pull_secret else None,
            node_selector=copy.deepcopy(node_selector_fact),
            tolerations=copy.deepcopy(tolerations_fact),
            main_container=MainContainer(
                image=k8s.get("k8s_image"),
                image_pull_policy="IfNotPresent",
                volume_mounts=fe_volume_mounts,
            ),
        ),
    )
    if enable_router or hf_home or etcd_endpoints:
        fe_envs = []
        if etcd_endpoints:
            fe_envs.append({"name": "ETCD_ENDPOINTS", "value": etcd_endpoints})
        if enable_router:
            fe_envs.append({"name": "DYN_ROUTER_MODE", "value": "kv"})
        if hf_home:
            fe_envs.append({"name": "HF_HOME", "value": hf_home})
        frontend.extra["envs"] = fe_envs

    services: dict[str, DGDService] = {"Frontend": frontend}
    if mode == "agg":
        services["SGLangWorker"] = render_worker(
            None, context.get("agg_workers"), context.get("agg_gpu"), context.get("agg_cli_args")
        )
    else:
        services["SGLangPrefillWorker"] = render_worker(
            "prefill", context.get("prefill_workers"), context.get("prefill_gpu"), context.get("prefill_cli_args")
        )
        services["SGLangDecodeWorker"] = render_worker(
            "decode", context.get("decode_workers"), context.get("decode_gpu"), context.get("decode_cli_args")
        )

    dgd = DGD(name=context.get("name"), namespace=k8s.get("k8s_namespace"), services=services)
    return [dgd]


# ---------------------------------------------------------------------------
# TRT-LLM — mirrors backend_templates/trtllm/k8s_deploy.yaml.j2 top to bottom.
# ---------------------------------------------------------------------------

_TRTLLM_ENGINE_CM_NAME = "engine-configs"
_TRTLLM_ENGINE_MOUNT_PATH = "/workspace/engine_configs"


def _populate_trtllm(context: dict[str, Any], resolved_facts: Any = None) -> list[Any]:
    k8s = context.get("K8sConfig", {}) or {}
    dyn = context.get("DynConfig", {}) or {}
    svc_cfg = context.get("ServiceConfig", {}) or {}

    enable_router = bool(dyn.get("enable_router") or False)
    use_engine_cm = k8s.get("k8s_engine_mode") == "configmap"
    model_cache_input = str(k8s.get("k8s_model_cache") or "").strip()
    use_model_cache = model_cache_input != ""
    model_cache_pvc = model_cache_input if use_model_cache else "model-cache"
    model_cache_mount = "/workspace/model_cache"
    runtime_working_dir = context.get("working_dir") or "/workspace/"
    etcd_endpoints = k8s.get("k8s_etcd_endpoints")
    hf_home = k8s.get("k8s_hf_home")
    image_pull_secret = k8s.get("k8s_image_pull_secret")
    mode = dyn.get("mode", "disagg") or "disagg"

    # Phase 4b-1 pod facts (hardware/transport). Guard on resolved_facts and
    # key presence so a None facts / missing key emits nothing (keeps the
    # crosscheck / no-fact callers byte-identical). Replicated per backend (no
    # shared helper, per PR#314->#340 rule).
    hw_facts = getattr(resolved_facts, "hardware", None) if resolved_facts is not None else None
    tr_facts = getattr(resolved_facts, "transport", None) if resolved_facts is not None else None
    node_selector_fact = hw_facts.get("node_selector") if isinstance(hw_facts, dict) else None
    tolerations_fact = (hw_facts.get("tolerations") or None) if isinstance(hw_facts, dict) else None

    def worker_main_env(role: str | None) -> list[dict[str, str]] | None:
        if not isinstance(hw_facts, dict):
            return None
        out: list[dict[str, str]] = []
        seen: set[str] = set()
        for k, v in (hw_facts.get("nccl_env") or {}).items():
            out.append({"name": k, "value": str(v)})
            seen.add(k)
        tr_env = tr_facts.get("env") if isinstance(tr_facts, dict) else None
        for k, v in (tr_env or {}).items():
            if k in seen:
                continue
            out.append({"name": k, "value": str(v)})
            seen.add(k)
        return out or None

    def worker_shared_memory(role: str | None) -> dict[str, str] | None:
        if not isinstance(hw_facts, dict):
            return None
        shm = hw_facts.get("shared_memory")
        if not isinstance(shm, dict):
            return None
        if role == "decode" and "disagg_decode" in shm:
            sz = shm["disagg_decode"]
        elif "default" in shm:
            sz = shm["default"]
        else:
            return None
        return {"size": sz}

    # macro render_volumes()
    def render_volumes() -> list[dict[str, Any]]:
        volumes: list[dict[str, Any]] = []
        if use_model_cache:
            volumes.append({"name": "model-cache", "persistentVolumeClaim": {"claimName": model_cache_pvc}})
        if use_engine_cm:
            volumes.append({"name": "engine-configs", "configMap": {"name": _TRTLLM_ENGINE_CM_NAME}})
        else:
            volumes.append({"name": "engine-configs", "emptyDir": {}})
        volumes.append({"name": "tmp", "emptyDir": {"medium": "Memory", "sizeLimit": "10Gi"}})
        return volumes

    # macro render_volume_mounts()
    def render_volume_mounts() -> list[dict[str, Any]]:
        mounts: list[dict[str, Any]] = []
        if use_model_cache:
            mounts.append({"name": "model-cache", "mountPath": model_cache_mount})
        engine_mount: dict[str, Any] = {"name": "engine-configs", "mountPath": _TRTLLM_ENGINE_MOUNT_PATH}
        if use_engine_cm:
            engine_mount["readOnly"] = True
        mounts.append(engine_mount)
        mounts.append({"name": "tmp", "mountPath": "/tmp"})
        return mounts

    # macro render_probes()
    def render_probes() -> dict[str, dict[str, Any]]:
        return {
            "startup_probe": {
                "httpGet": {"path": "/health", "port": 9090},
                "initialDelaySeconds": 120,
                "periodSeconds": 30,
                "timeoutSeconds": 10,
                "failureThreshold": 40,
            },
            "liveness_probe": {
                "httpGet": {"path": "/live", "port": 9090},
                "initialDelaySeconds": 300,
                "periodSeconds": 30,
                "timeoutSeconds": 10,
                "failureThreshold": 10,
            },
            "readiness_probe": {
                "httpGet": {"path": "/live", "port": 9090},
                "initialDelaySeconds": 300,
                "periodSeconds": 30,
                "timeoutSeconds": 10,
                "failureThreshold": 10,
            },
        }

    # macro render_worker(component_name, sub_component_type, replicas, gpu,
    #                     engine_path, inline_payload, cli_args_list, disagg_mode, publish_metrics)
    def render_worker(
        sub_component_type: str | None,
        replicas: Any,
        gpu: Any,
        engine_path: str,
        inline_payload: Any,
        cli_args_list: Any,
        disagg_mode: str | None,
        publish_metrics: bool,
    ) -> DGDService:
        envs = None
        if hf_home or etcd_endpoints:
            envs = []
            if etcd_endpoints:
                envs.append({"name": "ETCD_ENDPOINTS", "value": etcd_endpoints})
            if hf_home:
                envs.append({"name": "HF_HOME", "value": hf_home})

        image_pull_secrets = None
        if image_pull_secret:
            image_pull_secrets = [{"name": image_pull_secret}]

        # Block-scalar shell script, reproduced line-for-line from the template.
        lines = ["set -euo pipefail"]
        if not use_engine_cm:
            lines.append(f"mkdir -p {_TRTLLM_ENGINE_MOUNT_PATH}")
            lines.append(f"cat > {engine_path} <<'YAML'")
            lines.append(str(inline_payload or "").strip())
            lines.append("YAML")
        if cli_args_list:
            # The template's `{%- for %}` whitespace control collapses the arg
            # list onto a single line joined by the 16-space template indent.
            joined = "".join(f"{' ' * 16}{json.dumps(arg)}" for arg in cli_args_list)
            lines.append(f"args=({joined}{' ' * 16}--extra-engine-args \"{engine_path}\"")
            lines.append(")")
        else:
            lines.append("args=(")
            lines.append(f'  --model-path "{svc_cfg.get("model_path")}"')
            lines.append(f'  --served-model-name "{svc_cfg.get("served_model_name")}"')
            lines.append(f'  --extra-engine-args "{engine_path}"')
            lines.append(")")
        if disagg_mode:
            lines.append(f"args+=(--disaggregation-mode {disagg_mode})")
        if publish_metrics:
            lines.append("args+=(--publish-events-and-metrics)")
        lines.append('exec python3 -m dynamo.trtllm "${args[@]}"')
        script = "\n".join(lines) + "\n"

        probes = render_probes()
        return DGDService(
            env_from_secret="hf-token-secret",
            envs=envs,
            component_type="worker",
            sub_component_type=sub_component_type,
            replicas=replicas,
            resources={"limits": {"gpu": str(gpu)}},
            shared_memory=worker_shared_memory(sub_component_type),
            extra_pod_spec=ExtraPodSpec(
                volumes=render_volumes(),
                image_pull_secrets=image_pull_secrets,
                node_selector=copy.deepcopy(node_selector_fact),
                tolerations=copy.deepcopy(tolerations_fact),
                main_container=MainContainer(
                    image=k8s.get("k8s_image"),
                    working_dir=runtime_working_dir,
                    image_pull_policy="IfNotPresent",
                    volume_mounts=render_volume_mounts(),
                    command=["/bin/bash", "-c"],
                    args=[script],
                    startup_probe=probes["startup_probe"],
                    liveness_probe=probes["liveness_probe"],
                    readiness_probe=probes["readiness_probe"],
                    env=worker_main_env(sub_component_type),
                ),
            ),
        )

    docs: list[Any] = []

    # Engine ConfigMap document (emitted FIRST when k8s_engine_mode == 'configmap').
    if use_engine_cm:
        def cm_value(inline: Any) -> str:
            trimmed = str(inline or "").strip()
            return f"{trimmed}\n" if trimmed else ""

        data: dict[str, str] = {}
        if mode == "agg":
            agg_key = str(context.get("agg_engine_args", "")).split("/")[-1]
            data[agg_key] = cm_value(context.get("agg_engine_args_inline"))
        else:
            prefill_key = str(context.get("prefill_engine_args", "")).split("/")[-1]
            decode_key = str(context.get("decode_engine_args", "")).split("/")[-1]
            data[prefill_key] = cm_value(context.get("prefill_engine_args_inline"))
            data[decode_key] = cm_value(context.get("decode_engine_args_inline"))
        # Gate B fix: stamp the same namespace the DGD carries so that a bare
        # `kubectl apply -f` (no -n) lands the ConfigMap next to the workers;
        # otherwise kai-scheduler leaves pods Pending on a missing ConfigMap.
        docs.append(
            ConfigMapDoc(name=_TRTLLM_ENGINE_CM_NAME, namespace=k8s.get("k8s_namespace"), data=data)
        )

    # Frontend (trtllm Frontend optionally mounts the model cache).
    fe_volumes = None
    fe_volume_mounts = None
    if use_model_cache:
        fe_volumes = [{"name": "model-cache", "persistentVolumeClaim": {"claimName": model_cache_pvc}}]
        fe_volume_mounts = [{"name": "model-cache", "mountPath": model_cache_mount}]
    frontend = DGDService(
        env_from_secret="hf-token-secret",
        component_type="frontend",
        replicas=context.get("frontend_replicas"),
        extra_pod_spec=ExtraPodSpec(
            volumes=fe_volumes,
            image_pull_secrets=[{"name": image_pull_secret}] if image_pull_secret else None,
            node_selector=copy.deepcopy(node_selector_fact),
            tolerations=copy.deepcopy(tolerations_fact),
            main_container=MainContainer(
                image=k8s.get("k8s_image"),
                image_pull_policy="IfNotPresent",
                volume_mounts=fe_volume_mounts,
            ),
        ),
    )
    if enable_router or hf_home or etcd_endpoints:
        fe_envs = []
        if etcd_endpoints:
            fe_envs.append({"name": "ETCD_ENDPOINTS", "value": etcd_endpoints})
        if enable_router:
            fe_envs.append({"name": "DYN_ROUTER_MODE", "value": "kv"})
        if hf_home:
            fe_envs.append({"name": "HF_HOME", "value": hf_home})
        frontend.extra["envs"] = fe_envs

    services: dict[str, DGDService] = {"Frontend": frontend}
    if mode == "agg":
        services["TRTLLMWorker"] = render_worker(
            None,
            context.get("agg_workers"),
            context.get("agg_gpu"),
            context.get("agg_engine_args"),
            context.get("agg_engine_args_inline"),
            context.get("agg_cli_args_list"),
            None,
            enable_router,
        )
    else:
        services["TRTLLMPrefillWorker"] = render_worker(
            "prefill",
            context.get("prefill_workers"),
            context.get("prefill_gpu"),
            context.get("prefill_engine_args"),
            context.get("prefill_engine_args_inline"),
            context.get("prefill_cli_args_list"),
            "prefill",
            enable_router,
        )
        services["TRTLLMDecodeWorker"] = render_worker(
            "decode",
            context.get("decode_workers"),
            context.get("decode_gpu"),
            context.get("decode_engine_args"),
            context.get("decode_engine_args_inline"),
            context.get("decode_cli_args_list"),
            "decode",
            enable_router,
        )

    docs.append(DGD(name=context.get("name"), namespace=k8s.get("k8s_namespace"), services=services))
    return docs
