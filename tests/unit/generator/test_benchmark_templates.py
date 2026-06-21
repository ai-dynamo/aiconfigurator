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

    def test_bench_run_emits_prefix_prompt_args_when_configured(self, benchmark_env):
        context = _base_context()
        context["BenchConfig"]["prefix"] = 512
        context["BenchConfig"]["prefix_prompt_pool_size"] = 3

        rendered = _render("bench_run.sh.j2", benchmark_env, **context)

        prefix_arg_line = (
            'prefix_args+=(--prefix-prompt-length "${BENCH_PREFIX}" --num-prefix-prompts "${BENCH_PREFIX_PROMPTS}")'
        )
        assert 'BENCH_PREFIX="${AICONFIGURATOR_BENCH_PREFIX:-512}"' in rendered
        assert 'BENCH_PREFIX_PROMPTS="${AICONFIGURATOR_BENCH_PREFIX_PROMPTS:-3}"' in rendered
        assert '"${BENCH_PREFIX_PROMPTS}" =~ ^[0-9]+$' in rendered
        assert '"${BENCH_PREFIX_PROMPTS}" -gt 0' in rendered
        assert prefix_arg_line in rendered
        assert '"${prefix_args[@]}" \\' in rendered

    def test_k8s_bench_emits_prefix_prompt_args_when_configured(self, benchmark_env):
        context = _base_context()
        context["BenchConfig"]["prefix"] = 512
        context["BenchConfig"]["prefix_prompt_pool_size"] = 3

        rendered = _render("k8s_bench.yaml.j2", benchmark_env, **context)

        prefix_arg_line = (
            'prefix_args+=(--prefix-prompt-length "${BENCH_PREFIX}" --num-prefix-prompts "${BENCH_PREFIX_PROMPTS}")'
        )
        assert 'BENCH_PREFIX="${AICONFIGURATOR_BENCH_PREFIX:-512}"' in rendered
        assert 'BENCH_PREFIX_PROMPTS="${AICONFIGURATOR_BENCH_PREFIX_PROMPTS:-3}"' in rendered
        assert '"${BENCH_PREFIX_PROMPTS}" =~ ^[0-9]+$' in rendered
        assert '"${BENCH_PREFIX_PROMPTS}" -gt 0' in rendered
        assert prefix_arg_line in rendered
        assert '"${prefix_args[@]}" \\' in rendered

    @pytest.mark.parametrize("template_name", ["bench_run.sh.j2", "k8s_bench.yaml.j2"])
    def test_image_args_absent_when_batch_size_zero(self, benchmark_env, template_name):
        # image_batch_size omitted (defaults to 0) -> no image scaffolding rendered.
        rendered = _render(template_name, benchmark_env, **_base_context())
        assert "image_args=()" not in rendered
        assert "--image-batch-size" not in rendered
        assert "BENCH_IMAGE_BATCH_SIZE" not in rendered
        assert '"${image_args[@]}"' not in rendered

        # Explicit 0 is also disabled.
        context = _base_context()
        context["BenchConfig"]["image_batch_size"] = 0
        rendered_zero = _render(template_name, benchmark_env, **context)
        assert "image_args=()" not in rendered_zero
        assert "--image-batch-size" not in rendered_zero
        assert '"${image_args[@]}"' not in rendered_zero

    @pytest.mark.parametrize("template_name", ["bench_run.sh.j2", "k8s_bench.yaml.j2"])
    def test_image_args_emitted_when_batch_size_positive(self, benchmark_env, template_name):
        context = _base_context()
        context["BenchConfig"].update(
            {
                "image_batch_size": 1,
                "image_width_mean": 512,
                "image_width_stddev": 10,
                "image_height_mean": 512,
                "image_height_stddev": 10,
                "image_format": "png",
                "image_source": "random",
            }
        )
        rendered = _render(template_name, benchmark_env, **context)

        assert 'BENCH_IMAGE_BATCH_SIZE="${AICONFIGURATOR_BENCH_IMAGE_BATCH_SIZE:-1}"' in rendered
        assert "image_args=()" in rendered
        assert 'image_args+=(--image-batch-size "${BENCH_IMAGE_BATCH_SIZE}")' in rendered
        assert (
            '[[ -n "${BENCH_IMAGE_WIDTH_MEAN}" ]] && image_args+=(--image-width-mean "${BENCH_IMAGE_WIDTH_MEAN}")'
            in rendered
        )
        assert (
            '[[ -n "${BENCH_IMAGE_HEIGHT_MEAN}" ]] && image_args+=(--image-height-mean "${BENCH_IMAGE_HEIGHT_MEAN}")'
            in rendered
        )
        assert (
            '[[ -n "${BENCH_IMAGE_FORMAT}" ]] && image_args+=(--image-format "${BENCH_IMAGE_FORMAT}")' in rendered
        )
        assert (
            '[[ -n "${BENCH_IMAGE_SOURCE}" ]] && image_args+=(--image-source "${BENCH_IMAGE_SOURCE}")' in rendered
        )
        assert '"${image_args[@]}"' in rendered
        assert 'BENCH_IMAGE_WIDTH_MEAN="${AICONFIGURATOR_BENCH_IMAGE_WIDTH_MEAN:-512}"' in rendered
        assert 'BENCH_IMAGE_FORMAT="${AICONFIGURATOR_BENCH_IMAGE_FORMAT:-png}"' in rendered

    @pytest.mark.parametrize("template_name", ["bench_run.sh.j2", "k8s_bench.yaml.j2"])
    def test_input_file_args_absent_by_default(self, benchmark_env, template_name):
        rendered = _render(template_name, benchmark_env, **_base_context())
        assert "input_file_args=()" not in rendered
        assert "--input-file" not in rendered
        assert "BENCH_INPUT_FILE" not in rendered
        assert '"${input_file_args[@]}"' not in rendered

    @pytest.mark.parametrize("template_name", ["bench_run.sh.j2", "k8s_bench.yaml.j2"])
    def test_input_file_args_emitted_when_configured(self, benchmark_env, template_name):
        context = _base_context()
        context["BenchConfig"]["input_file"] = "/data/prompts.jsonl"
        context["BenchConfig"]["custom_dataset_type"] = "single_turn"
        rendered = _render(template_name, benchmark_env, **context)

        assert 'BENCH_INPUT_FILE="${AICONFIGURATOR_BENCH_INPUT_FILE:-/data/prompts.jsonl}"' in rendered
        assert "input_file_args=()" in rendered
        assert 'input_file_args+=(--input-file "${BENCH_INPUT_FILE}")' in rendered
        assert (
            '[[ -n "${BENCH_CUSTOM_DATASET_TYPE}" ]] && input_file_args+=(--custom-dataset-type "${BENCH_CUSTOM_DATASET_TYPE}")'
            in rendered
        )
        assert '"${input_file_args[@]}"' in rendered
