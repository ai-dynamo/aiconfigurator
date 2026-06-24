# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeploymentIR: a general service graph (design v2 §4, §8).

Components and edges are open lists so new topologies (e.g. multimodal EPD:
Processor + Encode workers) are added by appending nodes/edges, never by
changing the IR's field definitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Component:
    name: str
    role: str  # frontend | router | worker | prefill | decode | encode | processor | ...
    replicas: int = 1
    backend_args: dict[str, Any] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    resources: dict[str, Any] = field(default_factory=dict)


@dataclass
class DeploymentIR:
    backend: str
    backend_version: str
    mode: str  # agg | disagg
    model_profile_id: str | None = None
    sdk_model_family: str | None = None
    architecture: str | None = None
    components: list[Component] = field(default_factory=list)
    edges: list[tuple[str, str]] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    provenance: dict[str, dict[str, Any]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def add_component(self, component: Component) -> None:
        if any(c.name == component.name for c in self.components):
            raise ValueError(f"duplicate component: {component.name}")
        self.components.append(component)

    def add_edge(self, src: str, dst: str) -> None:
        names = {c.name for c in self.components}
        for n in (src, dst):
            if n not in names:
                raise ValueError(f"edge references unknown component: {n}")
        self.edges.append((src, dst))

    def add_provenance(self, key: str, value: Any, source: str, stage: str) -> None:
        self.provenance[key] = {"value": value, "source": source, "stage": stage}
