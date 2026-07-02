# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the vllm cli_args templates.

The k8s builder deliberately does NOT insert --served-model-name for vllm
(see builders/k8s_builder.py render_worker comment): the cli_args templates
own that flag and must emit it as the FIRST token, guarded on a non-empty
ServiceConfig.served_model_name. These tests pin that contract for the base
template and every active versioned template so the two layers can't drift
apart again (the 0.10.0 regression: the builder comment claimed template
ownership while no template emitted the flag, so the alias was dropped from
every generated worker command and alias requests returned HTTP 404).
"""

import shlex
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
    / "vllm"
)

# Base template + every version active in backend_version_matrix.yaml for the
# latest five Dynamo releases (1.2.0 -> 0.8.x).
_ACTIVE_TEMPLATES = [
    "cli_args.j2",
    "cli_args.0.12.0.j2",
    "cli_args.0.14.1.j2",
    "cli_args.0.16.0.j2",
    "cli_args.0.19.0.j2",
    "cli_args.0.20.1.j2",
]

_BASE_CONTEXT = {
    "vllm": {
        "tensor-parallel-size": 4,
        "max-model-len": 4096,
    },
    "speculative_config": {},
}


@pytest.fixture(scope="module")
def jinja_env():
    return Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _render(jinja_env, template_name: str, **extra) -> list[str]:
    rendered = jinja_env.get_template(template_name).render(**_BASE_CONTEXT, **extra)
    return shlex.split(rendered)


@pytest.mark.unit
@pytest.mark.parametrize("template_name", _ACTIVE_TEMPLATES)
class TestServedModelName:
    def test_emitted_first_when_set(self, jinja_env, template_name):
        tokens = _render(
            jinja_env,
            template_name,
            ServiceConfig={"served_model_name": "Qwen3-32B-FP8"},
        )
        assert tokens[:2] == ["--served-model-name", "Qwen3-32B-FP8"], (
            f"{template_name} must emit --served-model-name as the first cli_args "
            "token; the k8s builder relies on the template owning this flag"
        )
        assert tokens.count("--served-model-name") == 1

    def test_omitted_when_unset(self, jinja_env, template_name):
        tokens = _render(jinja_env, template_name)
        assert "--served-model-name" not in tokens

    def test_omitted_when_empty(self, jinja_env, template_name):
        tokens = _render(
            jinja_env,
            template_name,
            ServiceConfig={"served_model_name": ""},
        )
        assert "--served-model-name" not in tokens

    def test_other_flags_unaffected(self, jinja_env, template_name):
        tokens = _render(
            jinja_env,
            template_name,
            ServiceConfig={"served_model_name": "alias"},
        )
        assert "--tensor-parallel-size" in tokens
        assert "--max-model-len" in tokens
