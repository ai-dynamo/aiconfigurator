# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prove ``resolved_facts`` is threaded through the typed k8s builder path.

Phase 4b plumbing: ``build_dgd`` accepts ``resolved_facts`` (default ``None``)
and forwards it to each ``_populate_<backend>``. Nothing is read or emitted from
it yet, so output must stay byte-identical — this exercises the wiring only.
"""
import copy

import pytest

from aiconfigurator.generator.api import generate_backend_artifacts
from aiconfigurator.generator.builders import k8s_builder
from aiconfigurator.generator.rendering.engine import build_k8s_context_for_test
from tests.baseline.canary import CANARY_CASES


def test_resolved_facts_reach_populate(monkeypatch):
    """A sentinel passed to ``build_dgd`` arrives at the per-backend populate fn."""
    seen = {}
    orig = k8s_builder._populate_vllm

    def spy(*args, **kwargs):
        seen["facts"] = kwargs.get("resolved_facts")
        return orig(*args, **kwargs)

    monkeypatch.setattr(k8s_builder, "_populate_vllm", spy)

    case = next(c for c in CANARY_CASES if c.backend == "vllm" and "agg" in c.name)
    ctx = build_k8s_context_for_test(
        copy.deepcopy(case.params), case.backend, backend_version=case.backend_version
    )
    sentinel = {"hardware": {"_probe": True}}
    k8s_builder.build_dgd(ctx, "vllm", resolved_facts=sentinel)

    assert seen["facts"] is sentinel


def test_build_dgd_default_resolved_facts_is_none(monkeypatch):
    """Default call (no resolved_facts) forwards ``None`` — preserves old callers."""
    seen = {}
    orig = k8s_builder._populate_vllm

    def spy(*args, **kwargs):
        seen["facts"] = kwargs.get("resolved_facts")
        return orig(*args, **kwargs)

    monkeypatch.setattr(k8s_builder, "_populate_vllm", spy)

    case = next(c for c in CANARY_CASES if c.backend == "vllm" and "agg" in c.name)
    ctx = build_k8s_context_for_test(
        copy.deepcopy(case.params), case.backend, backend_version=case.backend_version
    )
    k8s_builder.build_dgd(ctx, "vllm")

    assert "facts" in seen
    assert seen["facts"] is None


def test_end_to_end_render_with_facts_threaded_is_unchanged():
    """End-to-end: a deepseek gb200 case still renders k8s_deploy.yaml.

    The render path self-resolves facts and threads them into ``build_dgd``;
    because nothing is emitted from facts yet, the artifact must be unchanged.
    Render twice and assert determinism (the threaded facts have no effect).
    """
    case = next(c for c in CANARY_CASES if c.name == "deepseek_sglang_gb200_agg")

    a = generate_backend_artifacts(
        copy.deepcopy(case.params), case.backend, backend_version=case.backend_version
    )
    b = generate_backend_artifacts(
        copy.deepcopy(case.params), case.backend, backend_version=case.backend_version
    )

    assert "k8s_deploy.yaml" in a
    assert a["k8s_deploy.yaml"]
    assert a["k8s_deploy.yaml"] == b["k8s_deploy.yaml"]
