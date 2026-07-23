# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed per-backend K8s (DynamoGraphDeployment) builders.

THE rendering path for ``k8s_deploy.yaml`` (replaced the Jinja templates). Each
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
import math
from typing import Any

from aiconfigurator.generator.dynamo_features import (
    frontend_cli_args_from_dyn_config,
    kvbm_env_from_dyn_config,
    normalize_router_mode,
    planner_config_from_dyn_config,
    planner_config_json,
    planner_image_from_k8s_config,
)

from .dgd_model import DGD, ComputeDomainDoc, ConfigMapDoc, DGDService, ExtraPodSpec, MainContainer


def build_dgd(context: dict[str, Any], backend: str, resolved_facts: Any = None) -> list[Any]:
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
    docs = populate(context, resolved_facts=resolved_facts)
    _apply_k8s_passthrough(docs, context)
    return docs


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base``; ``override`` wins.

    Nested dicts merge; every non-dict value (incl. lists) is replaced wholesale.
    """
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _apply_k8s_passthrough(docs: list[Any], context: dict[str, Any]) -> None:
    """Apply optional user-supplied K8s passthrough to the built DGD documents.

    A single central post-process (covers all three backends without touching
    the per-backend population logic, which deliberately mirrors the templates).
    Reads three optional ``K8sConfig`` keys; when none are set this is a no-op
    so output stays byte-identical:

    - ``extra_env``: list of env entries appended to each WORKER's main
      container env (after the hardware/transport fact env). Network-style
      customization is a worker concern; the frontend is left untouched.
    - ``worker_extra_pod_spec`` / ``frontend_extra_pod_spec``: dicts deep-merged
      into the worker / frontend ``extraPodSpec`` respectively (user values win;
      e.g. nodeSelector, tolerations, affinity, runtimeClassName).
    """
    k8s = context.get("K8sConfig", {}) or {}
    extra_env = k8s.get("extra_env") or None
    worker_eps = k8s.get("worker_extra_pod_spec") or None
    frontend_eps = k8s.get("frontend_extra_pod_spec") or None
    if not (extra_env or worker_eps or frontend_eps):
        return

    for doc in docs:
        services = getattr(doc, "services", None)
        if not isinstance(services, dict):  # only DGD docs carry services
            continue
        for svc in services.values():
            if svc.component_type == "worker":
                _apply_passthrough_to_service(svc, worker_eps, extra_env)
            elif svc.component_type == "frontend":
                _apply_passthrough_to_service(svc, frontend_eps, None)


def _apply_passthrough_to_service(svc: DGDService, eps_overlay: Any, extra_env: Any) -> None:
    eps = svc.extra_pod_spec or ExtraPodSpec()
    if isinstance(eps_overlay, dict) and eps_overlay:
        eps = ExtraPodSpec.from_dict(_deep_merge(eps.to_dict(), eps_overlay))
    if extra_env:
        mc = eps.main_container or MainContainer()
        mc.env = (mc.env or []) + copy.deepcopy(list(extra_env))
        eps.main_container = mc
    svc.extra_pod_spec = eps


def _append_envs(base: list[dict[str, str]] | None, extra: list[dict[str, str]]) -> list[dict[str, str]] | None:
    if not extra:
        return base
    merged = list(base or [])
    names = {entry.get("name") for entry in merged if isinstance(entry, dict)}
    for entry in extra:
        if entry.get("name") in names:
            continue
        merged.append(copy.deepcopy(entry))
    return merged or None


def _kvbm_env_for_role(
    dyn: dict[str, Any],
    role: str | None,
    *,
    backend: str,
) -> list[dict[str, str]]:
    if role not in (None, "prefill"):
        return []
    return kvbm_env_from_dyn_config(dyn, backend=backend)


def _frontend_main_container(
    *,
    image: str | None,
    image_pull_policy: str | None,
    volume_mounts: list[Any] | None,
    dyn: dict[str, Any],
    svc_cfg: dict[str, Any],
) -> MainContainer:
    # Keep the historical enable_router-only DGD shape unchanged; externally
    # evaluated candidates set router_mode/router_config when a frontend command
    # is required to pass richer CLI flags.
    frontend_dyn = dyn if dyn.get("router_mode") or dyn.get("router_config") else {}
    frontend_args = frontend_cli_args_from_dyn_config(frontend_dyn, svc_cfg)
    if frontend_args:
        return MainContainer(
            image=image,
            image_pull_policy=image_pull_policy,
            volume_mounts=volume_mounts,
            command=["python3", "-m", "dynamo.frontend"],
            args=frontend_args,
        )
    return MainContainer(
        image=image,
        image_pull_policy=image_pull_policy,
        volume_mounts=volume_mounts,
    )


def _planner_service(
    *,
    k8s: dict[str, Any],
    dyn: dict[str, Any],
    backend: str,
    mode: str,
    svc_cfg: dict[str, Any],
    node_selector: dict[str, Any] | None,
    tolerations: list[Any] | None,
    image_pull_secret: str | None,
) -> DGDService | None:
    planner_config = planner_config_from_dyn_config(dyn, backend=backend, mode=mode, service=svc_cfg)
    if not planner_config:
        return None
    return DGDService(
        env_from_secret="hf-token-secret",
        component_type="planner",
        replicas=1,
        extra_pod_spec=ExtraPodSpec(
            image_pull_secrets=[{"name": image_pull_secret}] if image_pull_secret else None,
            node_selector=copy.deepcopy(node_selector),
            tolerations=copy.deepcopy(tolerations),
            main_container=MainContainer(
                image=planner_image_from_k8s_config(k8s),
                image_pull_policy="IfNotPresent",
                command=["python3", "-m", "dynamo.planner"],
                args=["--config", planner_config_json(planner_config)],
            ),
        ),
    )


def _frontend_router_env(enable_router: bool, dyn: dict[str, Any]) -> dict[str, str] | None:
    if not enable_router and not dyn.get("router_mode"):
        return None
    return {"name": "DYN_ROUTER_MODE", "value": normalize_router_mode(dyn.get("router_mode") or "kv")}


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
    model_cache_mount = str(k8s.get("k8s_pvc_mount_path") or "/workspace/model_cache")
    enable_router = bool(dyn.get("enable_router") or False)
    etcd_endpoints = k8s.get("k8s_etcd_endpoints")
    hf_home = k8s.get("k8s_hf_home")
    image_pull_secret = k8s.get("k8s_image_pull_secret")

    # Hardware/transport pod facts. Guard on resolved_facts and
    # key presence so a None facts / missing key emits nothing (keeps the
    # crosscheck / no-fact callers byte-identical).
    hw_facts = getattr(resolved_facts, "hardware", None) if resolved_facts is not None else None
    tr_facts = getattr(resolved_facts, "transport", None) if resolved_facts is not None else None
    node_selector_fact = hw_facts.get("node_selector") if isinstance(hw_facts, dict) else None
    tolerations_fact = (hw_facts.get("tolerations") or None) if isinstance(hw_facts, dict) else None

    # Multinode (NVLink-fabric / GB200) detection: a worker is multinode when its
    # GPU count exceeds the node's GPU count. Opt-in — single-node deployments
    # emit nothing. Replicated per backend (no shared helper, per PR#314->#340).
    gpn = int(context.get("NodeConfig", {}).get("num_gpus_per_node", 8) or 8)
    dgd_name = context.get("name")
    any_multinode = [False]

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
    def render_worker(
        role: str | None,
        replicas: Any,
        gpu: Any,
        cli_args_list: Any,
        extra_args: list[str] | None = None,
    ) -> DGDService:
        envs = None
        if hf_home or etcd_endpoints:
            envs = []
            if etcd_endpoints:
                envs.append({"name": "ETCD_ENDPOINTS", "value": etcd_endpoints})
            if hf_home:
                envs.append({"name": "HF_HOME", "value": hf_home})
        envs = _append_envs(envs, _kvbm_env_for_role(dyn, role, backend="vllm"))

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
        # served-model-name: emit it here (directly after --model), matching the
        # sglang/trtllm workers. The vllm cli_args templates do NOT emit this
        # service-level flag, so without this the deployed model advertises only
        # the HF model id and requests using the configured alias 404.
        served_model_name = svc_cfg.get("served_model_name")
        if served_model_name:
            args.extend(["--served-model-name", str(served_model_name)])
        args.extend(cli_args_list or [])
        if enable_router:
            port = svc_cfg.get("dyn_vllm_kv_event_port") or 20081
            args.append("--kv-events-config")
            args.append(
                f'{{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:{port}","enable_kv_cache_events":true}}'
            )
        kvbm_enabled = bool(kvbm_env_from_dyn_config(dyn, backend="vllm"))
        if role == "prefill":
            if kvbm_enabled:
                transfer_config = (
                    '{"kv_connector":"PdConnector","kv_role":"kv_both","kv_connector_extra_config":'
                    '{"connectors":[{"kv_connector":"DynamoConnector","kv_connector_module_path":'
                    '"kvbm.vllm_integration.connector","kv_role":"kv_both"},{"kv_connector":"NixlConnector",'
                    '"kv_role":"kv_both"}]},"kv_connector_module_path":"kvbm.vllm_integration.connector"}'
                )
            else:
                transfer_config = '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'
            args.extend(["--is-prefill-worker", "--kv-transfer-config", transfer_config])
        elif role == "decode":
            args.extend(
                ["--is-decode-worker", "--kv-transfer-config", '{"kv_connector":"NixlConnector","kv_role":"kv_both"}']
            )
        elif role is None and kvbm_enabled:
            args.extend(
                [
                    "--kv-transfer-config",
                    '{"kv_connector":"DynamoConnector","kv_connector_module_path":"kvbm.vllm_integration.connector","kv_role":"kv_both"}',
                ]
            )
        elif role == "encode":
            # Multimodal EPD encode worker (vision encoder only).
            args.extend(["--multimodal-encode-worker", "--enable-multimodal"])
            gmu = (context.get("encode_params") or {}).get("gpu_memory_utilization")
            if gmu is not None:
                args.extend(["--gpu-memory-utilization", str(gmu)])
            args.extend(["--kv-transfer-config", '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'])
        args.extend(extra_args or [])

        # EFA pod requirements (worker-only, opt-in). Only the efa
        # transport profile carries a `pod` block; non-efa transports emit
        # nothing. Replicated per backend (no shared helper, per PR#314->#340).
        efa_pod = tr_facts.get("pod") if isinstance(tr_facts, dict) else None
        resources = {"limits": {"gpu": str(gpu)}}
        security_context = None
        host_ipc = None
        if isinstance(efa_pod, dict):
            if efa_pod.get("privileged"):
                security_context = {"privileged": True, "capabilities": {"add": ["IPC_LOCK"]}}
            if efa_pod.get("host_ipc"):
                host_ipc = True
            if efa_pod.get("efa_resource"):
                resources = {"limits": {"gpu": str(gpu), "custom": {efa_pod["efa_resource"]: str(gpu)}}}

        # Multinode worker: emit ComputeDomain wiring (resourceClaims + channel).
        multinode = None
        resource_claims = None
        try:
            gpu_count = int(gpu)
        except (TypeError, ValueError):
            gpu_count = 0
        if gpu_count > gpn:
            any_multinode[0] = True
            multinode = {"nodeCount": math.ceil(gpu_count / gpn)}
            resources = copy.deepcopy(resources)
            resources["claims"] = [{"name": "compute-domain-channel"}]
            resource_claims = [
                {"name": "compute-domain-channel", "resourceClaimTemplateName": f"{dgd_name}-compute-domain-channel"}
            ]

        return DGDService(
            env_from_secret="hf-token-secret",
            envs=envs,
            component_type="worker",
            sub_component_type=role,
            replicas=replicas if replicas is not None else 1,
            resources=resources,
            shared_memory=worker_shared_memory(role),
            multinode=multinode,
            extra_pod_spec=ExtraPodSpec(
                volumes=volumes,
                image_pull_secrets=image_pull_secrets,
                node_selector=copy.deepcopy(node_selector_fact),
                tolerations=copy.deepcopy(tolerations_fact),
                host_ipc=host_ipc,
                resource_claims=resource_claims,
                main_container=MainContainer(
                    image=k8s.get("k8s_image"),
                    working_dir=runtime_working_dir,
                    image_pull_policy="IfNotPresent",
                    volume_mounts=volume_mounts,
                    command=["python3", "-m", "dynamo.vllm"],
                    args=args,
                    env=worker_main_env(role),
                    security_context=security_context,
                ),
            ),
        )

    # Frontend (template emits envs AFTER extraPodSpec -> keep in extra to
    # preserve emission order; semantic equality is unaffected either way).
    # When a model-cache PVC is configured the frontend mounts it
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
            main_container=_frontend_main_container(
                image=k8s.get("k8s_image"),
                image_pull_policy="IfNotPresent",
                volume_mounts=fe_volume_mounts,
                dyn=dyn,
                svc_cfg=svc_cfg,
            ),
        ),
    )
    if enable_router or hf_home or etcd_endpoints:
        fe_envs = []
        if etcd_endpoints:
            fe_envs.append({"name": "ETCD_ENDPOINTS", "value": etcd_endpoints})
        if enable_router:
            fe_envs.append(_frontend_router_env(enable_router, dyn))
        if hf_home:
            fe_envs.append({"name": "HF_HOME", "value": hf_home})
        frontend.extra["envs"] = fe_envs

    services: dict[str, DGDService] = {"Frontend": frontend}
    mode = dyn.get("mode", "disagg") or "disagg"
    planner = _planner_service(
        k8s=k8s,
        dyn=dyn,
        backend="vllm",
        mode=mode,
        svc_cfg=svc_cfg,
        node_selector=node_selector_fact,
        tolerations=tolerations_fact,
        image_pull_secret=image_pull_secret,
    )
    if planner is not None:
        services["Planner"] = planner
    # Multimodal EPD: optional encode worker + mm flags on the PD workers
    # (the prefill/entry worker also routes to the encoder). Presence-guarded.
    is_epd = bool(context.get("encode_workers"))
    pd_mm_flags = ["--enable-multimodal", "--enable-mm-embeds"] if is_epd else None
    entry_mm_flags = ["--route-to-encoder", "--enable-multimodal", "--enable-mm-embeds"] if is_epd else None
    if mode == "agg":
        services["VllmWorker"] = render_worker(
            None,
            context.get("agg_workers"),
            context.get("agg_gpu"),
            context.get("agg_cli_args_list") or [],
            extra_args=entry_mm_flags,
        )
    else:
        services["VllmPrefillWorker"] = render_worker(
            "prefill",
            context.get("prefill_workers"),
            context.get("prefill_gpu"),
            context.get("prefill_cli_args_list") or [],
            extra_args=entry_mm_flags,
        )
        services["VllmDecodeWorker"] = render_worker(
            "decode",
            context.get("decode_workers"),
            context.get("decode_gpu"),
            context.get("decode_cli_args_list") or [],
            extra_args=pd_mm_flags,
        )
    if is_epd:
        services["VllmEncodeWorker"] = render_worker(
            "encode",
            context.get("encode_workers"),
            context.get("encode_gpu"),
            [],
        )

    dgd = DGD(name=context.get("name"), namespace=k8s.get("k8s_namespace"), services=services)
    docs: list[Any] = [dgd]
    if any_multinode[0]:
        docs.append(
            ComputeDomainDoc(
                name=f"{dgd_name}-compute-domain",
                namespace=k8s.get("k8s_namespace"),
                channel_name=f"{dgd_name}-compute-domain-channel",
                # numNodes=0 = DRA compute-domain on-demand mode (driver sizes the domain as pods
                # schedule); intentional, matches the standalone template.
                num_nodes=0,
            )
        )
    return docs


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
    model_cache_mount = str(k8s.get("k8s_pvc_mount_path") or "/workspace/model_cache")
    etcd_endpoints = k8s.get("k8s_etcd_endpoints")
    hf_home = k8s.get("k8s_hf_home")
    image_pull_secret = k8s.get("k8s_image_pull_secret")
    mode = dyn.get("mode", "disagg") or "disagg"

    # Hardware/transport pod facts. Guard on resolved_facts and
    # key presence so a None facts / missing key emits nothing (keeps the
    # crosscheck / no-fact callers byte-identical). Replicated per backend (no
    # shared helper, per PR#314->#340 rule).
    hw_facts = getattr(resolved_facts, "hardware", None) if resolved_facts is not None else None
    tr_facts = getattr(resolved_facts, "transport", None) if resolved_facts is not None else None
    node_selector_fact = hw_facts.get("node_selector") if isinstance(hw_facts, dict) else None
    tolerations_fact = (hw_facts.get("tolerations") or None) if isinstance(hw_facts, dict) else None

    # Multinode (NVLink-fabric / GB200) detection: a worker is multinode when its
    # GPU count exceeds the node's GPU count. Opt-in — single-node deployments
    # emit nothing. Replicated per backend (no shared helper, per PR#314->#340).
    gpn = int(context.get("NodeConfig", {}).get("num_gpus_per_node", 8) or 8)
    dgd_name = context.get("name")
    any_multinode = [False]

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
    def render_worker(
        role: str | None,
        replicas: Any,
        gpu: Any,
        cli_args: Any,
        extra_flags: list[str] | None = None,
    ) -> DGDService:
        envs = None
        if hf_home or etcd_endpoints:
            envs = []
            if etcd_endpoints:
                envs.append({"name": "ETCD_ENDPOINTS", "value": etcd_endpoints})
            if hf_home:
                envs.append({"name": "HF_HOME", "value": hf_home})
        envs = _append_envs(envs, _kvbm_env_for_role(dyn, role, backend="sglang"))

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
        elif role == "encode":
            # Multimodal EPD encode worker (vision encoder only).
            lines.append("args+=(--multimodal-encode-worker)")
            # --chat-template is REQUIRED: the sglang encode worker looks the
            # template up in its registry and crashes (KeyError: None) without
            # it. Default to dynamo's Qwen-VL E/PD default ("qwen2-vl", which the
            # Qwen3-VL family aiconfigurator models also uses); user-overridable.
            _ct = (context.get("encode_params") or {}).get("chat_template") or "qwen2-vl"
            lines.append(f'args+=(--chat-template "{_ct}")')
            lines.append("args+=(--skip-tokenizer-init)")
        for _flag in extra_flags or []:
            lines.append(f"args+=({_flag})")
        if enable_router:
            port = svc_cfg.get("sglang_kv_event_port") or 5557
            lines.append(
                f'args+=(--kv-events-config \'{{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:{port}"}}\')'
            )
        lines.append('exec python3 -m dynamo.sglang "${args[@]}"')
        script = "\n".join(lines) + "\n"

        # EFA pod requirements (worker-only, opt-in). Only the efa
        # transport profile carries a `pod` block; non-efa transports emit
        # nothing. Replicated per backend (no shared helper, per PR#314->#340).
        efa_pod = tr_facts.get("pod") if isinstance(tr_facts, dict) else None
        resources = {"limits": {"gpu": str(gpu)}}
        security_context = None
        host_ipc = None
        if isinstance(efa_pod, dict):
            if efa_pod.get("privileged"):
                security_context = {"privileged": True, "capabilities": {"add": ["IPC_LOCK"]}}
            if efa_pod.get("host_ipc"):
                host_ipc = True
            if efa_pod.get("efa_resource"):
                resources = {"limits": {"gpu": str(gpu), "custom": {efa_pod["efa_resource"]: str(gpu)}}}

        # Multinode worker: emit ComputeDomain wiring (resourceClaims + channel).
        multinode = None
        resource_claims = None
        try:
            gpu_count = int(gpu)
        except (TypeError, ValueError):
            gpu_count = 0
        if gpu_count > gpn:
            any_multinode[0] = True
            multinode = {"nodeCount": math.ceil(gpu_count / gpn)}
            resources = copy.deepcopy(resources)
            resources["claims"] = [{"name": "compute-domain-channel"}]
            resource_claims = [
                {"name": "compute-domain-channel", "resourceClaimTemplateName": f"{dgd_name}-compute-domain-channel"}
            ]

        return DGDService(
            env_from_secret="hf-token-secret",
            envs=envs,
            component_type="worker",
            sub_component_type=role,
            replicas=replicas if replicas is not None else 1,
            resources=resources,
            shared_memory=worker_shared_memory(role),
            multinode=multinode,
            extra_pod_spec=ExtraPodSpec(
                volumes=volumes,
                image_pull_secrets=image_pull_secrets,
                node_selector=copy.deepcopy(node_selector_fact),
                tolerations=copy.deepcopy(tolerations_fact),
                host_ipc=host_ipc,
                resource_claims=resource_claims,
                main_container=MainContainer(
                    image=k8s.get("k8s_image"),
                    working_dir=runtime_working_dir,
                    image_pull_policy="IfNotPresent",
                    volume_mounts=volume_mounts,
                    command=["/bin/bash", "-c"],
                    args=[script],
                    env=worker_main_env(role),
                    security_context=security_context,
                ),
            ),
        )

    # When a model-cache PVC is configured the frontend mounts it
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
            main_container=_frontend_main_container(
                image=k8s.get("k8s_image"),
                image_pull_policy="IfNotPresent",
                volume_mounts=fe_volume_mounts,
                dyn=dyn,
                svc_cfg=svc_cfg,
            ),
        ),
    )
    if enable_router or hf_home or etcd_endpoints:
        fe_envs = []
        if etcd_endpoints:
            fe_envs.append({"name": "ETCD_ENDPOINTS", "value": etcd_endpoints})
        if enable_router:
            fe_envs.append(_frontend_router_env(enable_router, dyn))
        if hf_home:
            fe_envs.append({"name": "HF_HOME", "value": hf_home})
        frontend.extra["envs"] = fe_envs

    services: dict[str, DGDService] = {"Frontend": frontend}
    planner = _planner_service(
        k8s=k8s,
        dyn=dyn,
        backend="sglang",
        mode=mode,
        svc_cfg=svc_cfg,
        node_selector=node_selector_fact,
        tolerations=tolerations_fact,
        image_pull_secret=image_pull_secret,
    )
    if planner is not None:
        services["Planner"] = planner
    # Multimodal EPD: optional encode worker + --multimodal-worker on the PD
    # worker(s) (sglang E/PD is typically 2-stage: encode + aggregated PD).
    is_epd = bool(context.get("encode_workers"))
    pd_mm_flags = ["--multimodal-worker", "--disaggregation-transfer-backend nixl"] if is_epd else None
    if mode == "agg":
        services["SGLangWorker"] = render_worker(
            None,
            context.get("agg_workers"),
            context.get("agg_gpu"),
            context.get("agg_cli_args"),
            extra_flags=pd_mm_flags,
        )
    else:
        services["SGLangPrefillWorker"] = render_worker(
            "prefill",
            context.get("prefill_workers"),
            context.get("prefill_gpu"),
            context.get("prefill_cli_args"),
            extra_flags=pd_mm_flags,
        )
        services["SGLangDecodeWorker"] = render_worker(
            "decode",
            context.get("decode_workers"),
            context.get("decode_gpu"),
            context.get("decode_cli_args"),
            extra_flags=pd_mm_flags,
        )
    if is_epd:
        services["SGLangEncodeWorker"] = render_worker(
            "encode",
            context.get("encode_workers"),
            context.get("encode_gpu"),
            context.get("encode_cli_args"),
        )

    dgd = DGD(name=context.get("name"), namespace=k8s.get("k8s_namespace"), services=services)
    docs: list[Any] = [dgd]
    if any_multinode[0]:
        docs.append(
            ComputeDomainDoc(
                name=f"{dgd_name}-compute-domain",
                namespace=k8s.get("k8s_namespace"),
                channel_name=f"{dgd_name}-compute-domain-channel",
                # numNodes=0 = DRA compute-domain on-demand mode (driver sizes the domain as pods
                # schedule); intentional, matches the standalone template.
                num_nodes=0,
            )
        )
    return docs


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
    model_cache_mount = str(k8s.get("k8s_pvc_mount_path") or "/workspace/model_cache")
    runtime_working_dir = context.get("working_dir") or "/workspace/"
    etcd_endpoints = k8s.get("k8s_etcd_endpoints")
    hf_home = k8s.get("k8s_hf_home")
    image_pull_secret = k8s.get("k8s_image_pull_secret")
    mode = dyn.get("mode", "disagg") or "disagg"

    # Multimodal EPD: an optional encode worker rendered alongside the PD
    # worker(s). Presence-guarded by encode_workers (set only when an encode
    # role is present), so non-EPD output is unchanged.
    is_epd = bool(context.get("encode_workers"))
    encode_params = context.get("encode_params") or {}
    # The prefill/PD worker reaches the encode worker via an explicit dyn://
    # endpoint, so both sides must agree on it. dynamo's per-mode default encode
    # component name is image-version-specific (e.g. `tensorrt_llm_encode`), so
    # we PIN the encode worker's --endpoint to a stable `encode` component inside
    # the operator-injected DYN_NAMESPACE (`{k8s_namespace}-{DGD name}`) and point
    # the prefill --encode-endpoint at the same value. Without the explicit
    # --endpoint, prefill cannot find the encode worker ("no instances found for
    # endpoint dynamo/encode/generate").
    _dyn_namespace = f"{k8s.get('k8s_namespace')}-{context.get('name')}"
    _encode_endpoint = f"dyn://{_dyn_namespace}.encode.generate"
    _modality = encode_params.get("modality") or "multimodal"

    def encode_flags() -> list[str]:
        flags = [f'--endpoint "{_encode_endpoint}"', f"--modality {_modality}"]
        almp = encode_params.get("allowed_local_media_path")
        if almp:
            flags.append(f'--allowed-local-media-path "{almp}"')
        mfs = encode_params.get("max_file_size_mb")
        if mfs is not None:
            flags.append(f"--max-file-size-mb {mfs}")
        return flags

    # PD workers that consume the encode worker carry --modality; the entry
    # worker (prefill in disagg, the agg/PD worker in 2-stage) also points at
    # the encode endpoint.
    pd_modality = [f"--modality {_modality}"] if is_epd else None
    pd_entry_flags = [f"--modality {_modality}", f'--encode-endpoint "{_encode_endpoint}"'] if is_epd else None

    # Hardware/transport pod facts. Guard on resolved_facts and
    # key presence so a None facts / missing key emits nothing (keeps the
    # crosscheck / no-fact callers byte-identical). Replicated per backend (no
    # shared helper, per PR#314->#340 rule).
    hw_facts = getattr(resolved_facts, "hardware", None) if resolved_facts is not None else None
    tr_facts = getattr(resolved_facts, "transport", None) if resolved_facts is not None else None
    node_selector_fact = hw_facts.get("node_selector") if isinstance(hw_facts, dict) else None
    tolerations_fact = (hw_facts.get("tolerations") or None) if isinstance(hw_facts, dict) else None

    # Multinode (NVLink-fabric / GB200) detection: a worker is multinode when its
    # GPU count exceeds the node's GPU count. Opt-in — single-node deployments
    # emit nothing. Replicated per backend (no shared helper, per PR#314->#340).
    gpn = int(context.get("NodeConfig", {}).get("num_gpus_per_node", 8) or 8)
    dgd_name = context.get("name")
    any_multinode = [False]

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
        extra_flags: list[str] | None = None,
    ) -> DGDService:
        envs = None
        if hf_home or etcd_endpoints:
            envs = []
            if etcd_endpoints:
                envs.append({"name": "ETCD_ENDPOINTS", "value": etcd_endpoints})
            if hf_home:
                envs.append({"name": "HF_HOME", "value": hf_home})
        envs = _append_envs(envs, _kvbm_env_for_role(dyn, sub_component_type, backend="trtllm"))

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
            # --model-path/--served-model-name must lead: every dynamo.trtllm
            # worker needs them, the trtllm cli_args template never emits them,
            # and without them dynamo.trtllm falls back to its default model
            # (the EPD encode worker otherwise loads TinyLlama, not the VL model).
            joined = "".join(f"{' ' * 16}{json.dumps(arg)}" for arg in cli_args_list)
            model_lead = (
                f'{" " * 16}--model-path "{svc_cfg.get("model_path")}"'
                f'{" " * 16}--served-model-name "{svc_cfg.get("served_model_name")}"'
            )
            lines.append(f'args=({model_lead}{joined}{" " * 16}--extra-engine-args "{engine_path}"')
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
        if sub_component_type in (None, "prefill") and kvbm_env_from_dyn_config(dyn, backend="trtllm"):
            lines.append("args+=(--connector kvbm)")
        for _flag in extra_flags or []:
            lines.append(f"args+=({_flag})")
        lines.append('exec python3 -m dynamo.trtllm "${args[@]}"')
        script = "\n".join(lines) + "\n"

        probes = render_probes()
        # EFA pod requirements (worker-only, opt-in). Only the efa
        # transport profile carries a `pod` block; non-efa transports emit
        # nothing. Replicated per backend (no shared helper, per PR#314->#340).
        efa_pod = tr_facts.get("pod") if isinstance(tr_facts, dict) else None
        resources = {"limits": {"gpu": str(gpu)}}
        security_context = None
        host_ipc = None
        if isinstance(efa_pod, dict):
            if efa_pod.get("privileged"):
                security_context = {"privileged": True, "capabilities": {"add": ["IPC_LOCK"]}}
            if efa_pod.get("host_ipc"):
                host_ipc = True
            if efa_pod.get("efa_resource"):
                resources = {"limits": {"gpu": str(gpu), "custom": {efa_pod["efa_resource"]: str(gpu)}}}

        # Multinode worker: emit ComputeDomain wiring (resourceClaims + channel).
        multinode = None
        resource_claims = None
        try:
            gpu_count = int(gpu)
        except (TypeError, ValueError):
            gpu_count = 0
        if gpu_count > gpn:
            any_multinode[0] = True
            multinode = {"nodeCount": math.ceil(gpu_count / gpn)}
            resources = copy.deepcopy(resources)
            resources["claims"] = [{"name": "compute-domain-channel"}]
            resource_claims = [
                {"name": "compute-domain-channel", "resourceClaimTemplateName": f"{dgd_name}-compute-domain-channel"}
            ]

        return DGDService(
            env_from_secret="hf-token-secret",
            envs=envs,
            component_type="worker",
            sub_component_type=sub_component_type,
            replicas=replicas,
            resources=resources,
            shared_memory=worker_shared_memory(sub_component_type),
            multinode=multinode,
            extra_pod_spec=ExtraPodSpec(
                volumes=render_volumes(),
                image_pull_secrets=image_pull_secrets,
                node_selector=copy.deepcopy(node_selector_fact),
                tolerations=copy.deepcopy(tolerations_fact),
                host_ipc=host_ipc,
                resource_claims=resource_claims,
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
                    security_context=security_context,
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
        if is_epd:
            encode_key = str(context.get("encode_engine_args", "")).split("/")[-1]
            data[encode_key] = cm_value(context.get("encode_engine_args_inline"))
        # Stamp the same namespace the DGD carries so that a bare
        # `kubectl apply -f` (no -n) lands the ConfigMap next to the workers;
        # otherwise kai-scheduler leaves pods Pending on a missing ConfigMap.
        docs.append(ConfigMapDoc(name=_TRTLLM_ENGINE_CM_NAME, namespace=k8s.get("k8s_namespace"), data=data))

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
            main_container=_frontend_main_container(
                image=k8s.get("k8s_image"),
                image_pull_policy="IfNotPresent",
                volume_mounts=fe_volume_mounts,
                dyn=dyn,
                svc_cfg=svc_cfg,
            ),
        ),
    )
    if enable_router or hf_home or etcd_endpoints:
        fe_envs = []
        if etcd_endpoints:
            fe_envs.append({"name": "ETCD_ENDPOINTS", "value": etcd_endpoints})
        if enable_router:
            fe_envs.append(_frontend_router_env(enable_router, dyn))
        if hf_home:
            fe_envs.append({"name": "HF_HOME", "value": hf_home})
        frontend.extra["envs"] = fe_envs

    services: dict[str, DGDService] = {"Frontend": frontend}
    planner = _planner_service(
        k8s=k8s,
        dyn=dyn,
        backend="trtllm",
        mode=mode,
        svc_cfg=svc_cfg,
        node_selector=node_selector_fact,
        tolerations=tolerations_fact,
        image_pull_secret=image_pull_secret,
    )
    if planner is not None:
        services["Planner"] = planner
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
            extra_flags=pd_entry_flags,  # 2-stage E/PD: agg worker is the PD entry point
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
            extra_flags=pd_entry_flags,  # 3-stage EPD: prefill is the encode entry point
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
            extra_flags=pd_modality,
        )
    if is_epd:
        services["TRTLLMEncodeWorker"] = render_worker(
            "encode",
            context.get("encode_workers"),
            context.get("encode_gpu"),
            context.get("encode_engine_args"),
            context.get("encode_engine_args_inline"),
            context.get("encode_cli_args_list"),
            "encode",
            False,
            extra_flags=encode_flags(),
        )

    docs.append(DGD(name=context.get("name"), namespace=k8s.get("k8s_namespace"), services=services))
    if any_multinode[0]:
        docs.append(
            ComputeDomainDoc(
                name=f"{dgd_name}-compute-domain",
                namespace=k8s.get("k8s_namespace"),
                channel_name=f"{dgd_name}-compute-domain-channel",
                # numNodes=0 = DRA compute-domain on-demand mode (driver sizes the domain as pods
                # schedule); intentional, matches the standalone template.
                num_nodes=0,
            )
        )
    return docs
