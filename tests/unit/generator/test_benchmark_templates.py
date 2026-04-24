# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for shared benchmark templates."""

from pathlib import Path

import pytest
from jinja2 import Environment, FileSystemLoader

_TEMPLATE_DIR = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "aiconfigurator"
    / "generator"
    / "config"
    / "backend_templates"
    / "benchmark"
)


@pytest.fixture(scope="module")
def benchmark_env():
    return Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _render(template_name: str, env: Environment, **ctx) -> str:
    return env.get_template(template_name).render(**ctx).strip()


def _base_context() -> dict:
    return {
        "BenchConfig": {
            "estimated_concurrency": 1,
            "model": "test/model",
            "endpoint_type": "chat",
            "endpoint_url": "http://bench.example:8000",
            "tokenizer": "test/tokenizer",
            "isl": 1000,
            "isl_stddev": 0,
            "osl": 100,
            "osl_stddev": 0,
            "ui": "simple",
            "name": "test-benchmark",
            "image": "python:3.12-slim",
            "profile_start_timeout": 400,
        },
        "ServiceConfig": {"head_node_ip": "127.0.0.1", "port": 8000},
        "K8sConfig": {"k8s_namespace": "default"},
    }


@pytest.mark.unit
class TestBenchmarkTemplates:
    def test_bench_run_filters_zero_concurrency(self, benchmark_env):
        rendered = _render("bench_run.sh.j2", benchmark_env, **_base_context())
        assert "concurrency_array=(0" not in rendered
        assert "concurrency_array=(1 2 8 16 32 64 128)" in rendered

    def test_k8s_bench_filters_zero_concurrency(self, benchmark_env):
        rendered = _render("k8s_bench.yaml.j2", benchmark_env, **_base_context())
        assert "concurrency_array=(0" not in rendered
        assert "concurrency_array=(1 2 8 16 32 64 128)" in rendered
