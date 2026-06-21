# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Facts are LIVE on the public render path (Phase 4a).

``generate_backend_artifacts`` is the public API used by the CLI, the dynamo
profiler, and the freeze script. After the self-resolve fix in
``render_backend_templates``, it must apply model-default cli flags (resolved
from the request's own params) — not only the v2 ``run_pipeline`` orchestrator.

These tests assert the model-default flags actually land for fact-bearing
deepseek models, and stay ABSENT for a generic Qwen model (no profile match ->
``facts.model is None`` -> apply is a strict no-op).
"""
from __future__ import annotations

import copy

from aiconfigurator.generator.api import generate_backend_artifacts
from tests.baseline.canary import CANARY_CASES


def _blob(name: str) -> str:
    c = next(c for c in CANARY_CASES if c.name == name)
    arts = generate_backend_artifacts(copy.deepcopy(c.params), c.backend, backend_version=c.backend_version)
    return "\n".join(arts.values())


def test_deepseek_vllm_gets_model_default_flags():
    blob = _blob("deepseek_vllm_h200_agg")
    assert "--block-size 256" in blob
    assert "--trust-remote-code" in blob


def test_deepseek_sglang_gets_eagle_speculative_defaults():
    assert "EAGLE" in _blob("deepseek_sglang_h200_agg")


def test_generic_qwen_has_no_deepseek_defaults():
    blob = _blob("vllm_dense_agg")
    assert "--block-size 256" not in blob and "--trust-remote-code" not in blob


def test_trtllm_b200_engine_uses_wideep():
    c = next(c for c in CANARY_CASES if c.name == "deepseek_trtllm_b200_disagg")
    arts = generate_backend_artifacts(copy.deepcopy(c.params), c.backend, backend_version=c.backend_version)
    eng = "\n".join(v for k, v in arts.items() if k.startswith("extra_engine_args_"))
    # Check the emitted value line, not a bare substring: the template's
    # explanatory comment enumerates WIDEEP as a valid option.
    assert "backend: WIDEEP" in eng


def test_sglang_gb200_uses_deepep_moe():
    c = next(c for c in CANARY_CASES if c.name == "deepseek_sglang_gb200_agg")
    blob = "\n".join(generate_backend_artifacts(copy.deepcopy(c.params), c.backend, backend_version=c.backend_version).values())
    assert "deepep_moe" in blob


def test_trtllm_h200_engine_stays_cutlass():
    # contrast: Hopper must NOT get WIDEEP (would crash on the wrong silicon target)
    c = next(c for c in CANARY_CASES if c.name == "trtllm_moe_disagg")  # Qwen MoE, h200
    arts = generate_backend_artifacts(copy.deepcopy(c.params), c.backend, backend_version=c.backend_version)
    eng = "\n".join(v for k, v in arts.items() if k.startswith("extra_engine_args_"))
    # Check the emitted value line, not a bare substring: the template's
    # explanatory comment enumerates WIDEEP as a valid option.
    assert "backend: WIDEEP" not in eng
