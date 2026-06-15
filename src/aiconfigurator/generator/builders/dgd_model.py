# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed object model of the DynamoGraphDeployment v1alpha1 resource.

Mirrors what the backend k8s_deploy templates emit (vllm/sglang/trtllm),
covering every field the generator can produce. Fidelity guarantees:

1. Lossless round-trip: ``from_dict(doc).to_dict() == doc`` for every real
   generated document. Each object keeps unknown keys in an ``extra`` dict
   (same philosophy as ``generator/request.py``) and merges them back in
   ``to_dict``.
2. Emission order of typed keys follows the template emission order so
   serialized YAML diffs stay readable; semantic (dict) equality is the
   actual round-trip gate.

Only objects that builders construct field-by-field are typed
(``MainContainer``, ``ExtraPodSpec``, ``DGDService``). Deep K8s pod
structures (probes, volumes, env entries, resources) stay as raw
dicts/lists passed through verbatim.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import yaml

DGD_API_VERSION = "nvidia.com/v1alpha1"
DGD_KIND = "DynamoGraphDeployment"


def _put(out: dict[str, Any], key: str, value: Any) -> None:
    """Insert ``key`` only when the typed field is actually set (non-None)."""
    if value is not None:
        out[key] = value


@dataclass
class MainContainer:
    """``spec.services.<name>.extraPodSpec.mainContainer``."""

    image: str | None = None
    working_dir: str | None = None
    image_pull_policy: str | None = None
    volume_mounts: list[Any] | None = None
    command: list[str] | None = None
    args: list[str] | None = None
    startup_probe: dict[str, Any] | None = None
    liveness_probe: dict[str, Any] | None = None
    readiness_probe: dict[str, Any] | None = None
    env: list[Any] | None = None
    security_context: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    # Emission order of the legacy k8s_deploy.yaml.j2 render_worker macros
    # (templates deleted at the Stage-3 cutover; see git history).
    _KEYS = (
        ("image", "image"),
        ("workingDir", "working_dir"),
        ("imagePullPolicy", "image_pull_policy"),
        ("volumeMounts", "volume_mounts"),
        ("command", "command"),
        ("args", "args"),
        ("startupProbe", "startup_probe"),
        ("livenessProbe", "liveness_probe"),
        ("readinessProbe", "readiness_probe"),
        ("env", "env"),
        ("securityContext", "security_context"),
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MainContainer":
        data = copy.deepcopy(data)
        kwargs = {attr: data.pop(key) for key, attr in cls._KEYS if key in data}
        return cls(extra=data, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, attr in self._KEYS:
            _put(out, key, getattr(self, attr))
        out.update(copy.deepcopy(self.extra))
        return out


@dataclass
class ExtraPodSpec:
    """``spec.services.<name>.extraPodSpec``."""

    volumes: list[Any] | None = None
    image_pull_secrets: list[Any] | None = None
    main_container: MainContainer | None = None
    node_selector: dict[str, Any] | None = None
    tolerations: list[Any] | None = None
    host_ipc: bool | None = None
    resource_claims: list[Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    _KEYS = (
        ("volumes", "volumes"),
        ("imagePullSecrets", "image_pull_secrets"),
        ("nodeSelector", "node_selector"),
        ("tolerations", "tolerations"),
        ("hostIPC", "host_ipc"),
        ("resourceClaims", "resource_claims"),
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExtraPodSpec":
        data = copy.deepcopy(data)
        kwargs = {attr: data.pop(key) for key, attr in cls._KEYS if key in data}
        main_container = None
        if "mainContainer" in data:
            main_container = MainContainer.from_dict(data.pop("mainContainer"))
        return cls(main_container=main_container, extra=data, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, attr in self._KEYS:
            _put(out, key, getattr(self, attr))
        if self.main_container is not None:
            out["mainContainer"] = self.main_container.to_dict()
        out.update(copy.deepcopy(self.extra))
        return out


@dataclass
class DGDService:
    """A single entry of ``spec.services`` (Frontend, *Worker, ...)."""

    env_from_secret: str | None = None
    envs: list[Any] | None = None
    component_type: str | None = None
    sub_component_type: str | None = None
    replicas: int | None = None
    resources: dict[str, Any] | None = None
    extra_pod_spec: ExtraPodSpec | None = None
    shared_memory: dict[str, Any] | None = None
    multinode: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    _KEYS = (
        ("envFromSecret", "env_from_secret"),
        ("envs", "envs"),
        ("componentType", "component_type"),
        ("subComponentType", "sub_component_type"),
        ("replicas", "replicas"),
        ("resources", "resources"),
        ("sharedMemory", "shared_memory"),
        ("multinode", "multinode"),
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DGDService":
        data = copy.deepcopy(data)
        kwargs = {attr: data.pop(key) for key, attr in cls._KEYS if key in data}
        extra_pod_spec = None
        if "extraPodSpec" in data:
            extra_pod_spec = ExtraPodSpec.from_dict(data.pop("extraPodSpec"))
        return cls(extra_pod_spec=extra_pod_spec, extra=data, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, attr in self._KEYS:
            _put(out, key, getattr(self, attr))
        if self.extra_pod_spec is not None:
            out["extraPodSpec"] = self.extra_pod_spec.to_dict()
        out.update(copy.deepcopy(self.extra))
        return out


@dataclass
class DGD:
    """A DynamoGraphDeployment v1alpha1 document."""

    name: str | None = None
    namespace: str | None = None
    api_version: str = DGD_API_VERSION
    kind: str = DGD_KIND
    services: dict[str, DGDService] = field(default_factory=dict)
    metadata_extra: dict[str, Any] = field(default_factory=dict)
    spec_extra: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DGD":
        data = copy.deepcopy(data)
        api_version = data.pop("apiVersion", DGD_API_VERSION)
        kind = data.pop("kind", DGD_KIND)
        metadata = data.pop("metadata", {}) or {}
        spec = data.pop("spec", {}) or {}
        services_raw = spec.pop("services", {}) or {}
        return cls(
            api_version=api_version,
            kind=kind,
            name=metadata.pop("name", None),
            namespace=metadata.pop("namespace", None),
            services={name: DGDService.from_dict(svc) for name, svc in services_raw.items()},
            metadata_extra=metadata,
            spec_extra=spec,
            extra=data,
        )

    def to_dict(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        _put(metadata, "name", self.name)
        _put(metadata, "namespace", self.namespace)
        metadata.update(copy.deepcopy(self.metadata_extra))
        spec: dict[str, Any] = {
            "services": {name: svc.to_dict() for name, svc in self.services.items()},
        }
        spec.update(copy.deepcopy(self.spec_extra))
        out: dict[str, Any] = {
            "apiVersion": self.api_version,
            "kind": self.kind,
            "metadata": metadata,
            "spec": spec,
        }
        out.update(copy.deepcopy(self.extra))
        return out

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False)


@dataclass
class ConfigMapDoc:
    """A plain v1 ConfigMap document (trtllm engine-configs)."""

    name: str | None = None
    namespace: str | None = None
    api_version: str = "v1"
    kind: str = "ConfigMap"
    data: dict[str, str] = field(default_factory=dict)
    metadata_extra: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConfigMapDoc":
        data = copy.deepcopy(data)
        api_version = data.pop("apiVersion", "v1")
        kind = data.pop("kind", "ConfigMap")
        metadata = data.pop("metadata", {}) or {}
        cm_data = data.pop("data", {}) or {}
        return cls(
            api_version=api_version,
            kind=kind,
            name=metadata.pop("name", None),
            namespace=metadata.pop("namespace", None),
            data=cm_data,
            metadata_extra=metadata,
            extra=data,
        )

    def to_dict(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        _put(metadata, "name", self.name)
        _put(metadata, "namespace", self.namespace)
        metadata.update(copy.deepcopy(self.metadata_extra))
        out: dict[str, Any] = {
            "apiVersion": self.api_version,
            "kind": self.kind,
            "metadata": metadata,
            "data": copy.deepcopy(self.data),
        }
        out.update(copy.deepcopy(self.extra))
        return out

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False)


@dataclass
class ComputeDomainDoc:
    """A ComputeDomain CRD document (resource.nvidia.com/v1beta1).

    Emitted once per deployment when any worker is multinode (a worker whose
    GPU count exceeds the node's GPU count). Recipes ship the ComputeDomain
    only — NOT a ResourceClaimTemplate.
    """

    name: str | None = None
    namespace: str | None = None
    channel_name: str | None = None
    num_nodes: int = 0
    api_version: str = "resource.nvidia.com/v1beta1"
    kind: str = "ComputeDomain"
    metadata_extra: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ComputeDomainDoc":
        data = copy.deepcopy(data)
        api_version = data.pop("apiVersion", "resource.nvidia.com/v1beta1")
        kind = data.pop("kind", "ComputeDomain")
        metadata = data.pop("metadata", {}) or {}
        spec = data.pop("spec", {}) or {}
        channel = spec.pop("channel", {}) or {}
        rct = channel.pop("resourceClaimTemplate", {}) or {}
        return cls(
            api_version=api_version,
            kind=kind,
            name=metadata.pop("name", None),
            namespace=metadata.pop("namespace", None),
            channel_name=rct.pop("name", None),
            num_nodes=spec.pop("numNodes", 0),
            metadata_extra=metadata,
            extra=data,
        )

    def to_dict(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        _put(metadata, "name", self.name)
        _put(metadata, "namespace", self.namespace)
        metadata.update(copy.deepcopy(self.metadata_extra))
        out: dict[str, Any] = {
            "apiVersion": self.api_version,
            "kind": self.kind,
            "metadata": metadata,
            "spec": {
                "channel": {"resourceClaimTemplate": {"name": self.channel_name}},
                "numNodes": self.num_nodes,
            },
        }
        out.update(copy.deepcopy(self.extra))
        return out

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False)


def dgd_documents_to_yaml(docs: list[Any]) -> str:
    """Serialize a list of model objects (DGD / ConfigMapDoc / ComputeDomainDoc) as a multi-doc YAML stream."""
    return "---\n".join(doc.to_yaml() for doc in docs)
