# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional per-role trtllm engine-config passthrough.

Lets a user inject arbitrary ``extra_engine_args`` keys the template doesn't
model. Additive and presence-guarded: absent -> output byte-identical.
"""

from __future__ import annotations

import copy

import yaml

from aiconfigurator.generator.api import generate_backend_artifacts
from tests.baseline.canary import CANARY_CASES


def _trtllm_agg():
    return next(c for c in CANARY_CASES if c.backend == "trtllm" and "agg" in c.name)


def test_absent_passthrough_is_byte_identical():
    c = _trtllm_agg()
    base = generate_backend_artifacts(copy.deepcopy(c.params), c.backend, backend_version=c.backend_version)
    p = copy.deepcopy(c.params)
    # empty / missing -> no effect
    p["params"]["agg"]["extra_engine_args"] = {}
    out = generate_backend_artifacts(p, c.backend, backend_version=c.backend_version)
    assert out["extra_engine_args_agg.yaml"] == base["extra_engine_args_agg.yaml"]


def test_extra_engine_args_appended_and_parseable():
    c = _trtllm_agg()
    p = copy.deepcopy(c.params)
    p["params"]["agg"]["extra_engine_args"] = {"some_experimental_flag": True, "trust_remote_code": True}
    out = generate_backend_artifacts(p, c.backend, backend_version=c.backend_version)
    blob = out["extra_engine_args_agg.yaml"]
    # The appended keys are present AND the whole artifact is still valid YAML.
    parsed = yaml.safe_load(blob)
    assert parsed["some_experimental_flag"] is True
    assert parsed["trust_remote_code"] is True
    # A real template field is still there (passthrough is additive, not a replace).
    assert "backend" in parsed
