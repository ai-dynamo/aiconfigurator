# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The single v2 orchestrator (design v2 §3, §4).

Phase 1: build the IR from params, then render through the UNCHANGED legacy
path so output is byte-identical to api.generate_backend_artifacts. Later
phases replace the render step with typed builders; the seam stays here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .ir import Component, DeploymentIR
from .rendering import render_backend_templates

_FRONTEND_NAME = {"vllm": "Frontend", "sglang": "Frontend", "trtllm": "Frontend"}
_WORKER_NAME = {"vllm": "VllmWorker", "sglang": "SglangWorker", "trtllm": "TrtllmWorker"}


@dataclass
class PipelineResult:
    artifacts: dict[str, str]
    ir: DeploymentIR


def _build_ir(params: dict[str, Any], backend: str, backend_version: str | None) -> DeploymentIR:
    roles = params.get("params", {}) or {}
    mode = "disagg" if (roles.get("prefill") and roles.get("decode")) else "agg"
    ir = DeploymentIR(backend=backend, backend_version=backend_version or "", mode=mode)
    frontend = _FRONTEND_NAME.get(backend, "Frontend")
    ir.add_component(Component(name=frontend, role="frontend"))
    if mode == "disagg":
        ir.add_component(Component(name="PrefillWorker", role="prefill",
                                   backend_args=dict(roles.get("prefill") or {})))
        ir.add_component(Component(name="DecodeWorker", role="decode",
                                   backend_args=dict(roles.get("decode") or {})))
        ir.add_edge(frontend, "PrefillWorker")
        ir.add_edge("PrefillWorker", "DecodeWorker")
    else:
        worker = _WORKER_NAME.get(backend, "Worker")
        ir.add_component(Component(name=worker, role="worker",
                                   backend_args=dict(roles.get("agg") or {})))
        ir.add_edge(frontend, worker)
    return ir


def run_pipeline(
    params: dict[str, Any],
    backend: str,
    templates_dir: Optional[str] = None,
    backend_version: Optional[str] = None,
    deployment_target: Optional[str] = None,
) -> PipelineResult:
    ir = _build_ir(params, backend, backend_version)
    artifacts = render_backend_templates(
        params, backend, templates_dir, backend_version,
        deployment_target=deployment_target or "dynamo-j2",
    )
    return PipelineResult(artifacts=artifacts, ir=ir)
