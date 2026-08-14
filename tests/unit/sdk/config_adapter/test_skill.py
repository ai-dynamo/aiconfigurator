# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
HELPER = REPO_ROOT / ".agents" / "skills" / "adapt-server-config" / "scripts" / "adapt_config.py"


def test_known_format_helper_is_deterministic_and_validates_schema(tmp_path):
    config = {
        "config_id": 1,
        "hardware": "h100",
        "framework": "trt",
        "silicon_model": "llama70b",
        "precision": "bf16",
        "spec_method": "none",
        "disagg": False,
        "decode_tp": 2,
        "decode_ep": 1,
        "decode_dp_attention": False,
        "decode_num_workers": 1,
        "num_decode_gpu": 2,
    }
    benchmark = {"id": "bench", "isl": 1024, "osl": 128, "conc": 8}
    config_path = tmp_path / "config.json"
    benchmark_path = tmp_path / "benchmark.json"
    config_path.write_text(json.dumps(config))
    benchmark_path.write_text(json.dumps(benchmark))
    command = [
        sys.executable,
        str(HELPER),
        "--format",
        "inferencex",
        "--config",
        str(config_path),
        "--benchmark",
        str(benchmark_path),
    ]

    first = subprocess.run(command, cwd=REPO_ROOT, check=True, capture_output=True, text=True)
    second = subprocess.run(command, cwd=REPO_ROOT, check=True, capture_output=True, text=True)
    payload = json.loads(first.stdout)

    assert first.stdout == second.stdout
    assert payload["outcomes"][0]["status"] == "adapted"
    assert payload["outcomes"][0]["request"]["schema_version"] == "aic-estimate-request/1.0.0"

    config["hardware"] = "unsupported"
    config_path.write_text(json.dumps(config))
    malformed = subprocess.run(command, cwd=REPO_ROOT, check=False, capture_output=True, text=True)
    malformed_payload = json.loads(malformed.stdout)

    assert malformed.returncode == 1
    assert malformed_payload["outcomes"][0]["status"] == "rejected"
    assert malformed_payload["outcomes"][0]["request"] is None
    diagnostic = malformed_payload["outcomes"][0]["diagnostics"][0]
    assert diagnostic["severity"] == "error"
    assert diagnostic["code"] == "inferencex_mapping_failed"
    assert "unsupported or missing InferenceX hardware" in diagnostic["message"]


def test_known_format_helper_adapts_concrete_dynamo_ci_recipe(tmp_path):
    deployment = tmp_path / "recipe.yaml"
    deployment.write_text(
        """
name: h100-bf16-test
model: {path: qwen, container: latest, precision: bf16}
resources:
  gpu_type: h100
  prefill_nodes: 1
  prefill_workers: 1
  decode_nodes: 1
  decode_workers: 1
  gpus_per_node: 1
backend:
  sglang_config:
    prefill:
      served-model-name: Qwen/Qwen3-32B
      tensor-parallel-size: 1
      mem-fraction-static: 0.9
    decode:
      served-model-name: Qwen/Qwen3-32B
      tensor-parallel-size: 1
      mem-fraction-static: 0.9
benchmark: {isl: 128, osl: 16, concurrencies: '1x2'}
"""
    )
    command = [
        sys.executable,
        str(HELPER),
        "--format",
        "dynamo",
        "--deploy",
        str(deployment),
    ]

    first = subprocess.run(command, cwd=REPO_ROOT, check=True, capture_output=True, text=True)
    second = subprocess.run(command, cwd=REPO_ROOT, check=True, capture_output=True, text=True)
    payload = json.loads(first.stdout)

    assert first.stdout == second.stdout
    assert [outcome["point_id"] for outcome in payload["outcomes"]] == ["concurrency-1", "concurrency-2"]
    assert [outcome["status"] for outcome in payload["outcomes"]] == ["adapted", "adapted"]
    assert [outcome["request"]["workload"]["concurrency"] for outcome in payload["outcomes"]] == [1, 2]
    for outcome in payload["outcomes"]:
        request = outcome["request"]
        assert request["model"]["path"] == "Qwen/Qwen3-32B"
        assert request["backend"]["name"] == "sglang"
        assert request["workload"]["isl"] == 128
        assert request["workload"]["osl"] == 16
        assert request["quantization"]["gemm"] is None
        assert request["quantization"]["moe"] is None
        assert request["topology"]["prefill"]["tp_size"] == 1
        assert request["topology"]["decode"]["tp_size"] == 1
        assert request["provenance"]["source_ids"]["format"] == "dynamo-ci-concrete"


def test_skill_contains_confirmation_and_explicit_estimate_guards():
    skill = (HELPER.parents[1] / "SKILL.md").read_text()

    assert "Obtain confirmation before creating canonical request JSON" in skill
    assert "Run estimates only on explicit request" in skill
    assert "--run-estimate" in skill
