# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the trtllm cli_args.j2 template.

After the translate module was introduced, cli_args.j2 only emits direct CLI
flags accepted by dynamo.trtllm's argparser.  Model path, served model name,
and override-engine-args are no longer part of this template.
"""

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
    / "trtllm"
)


@pytest.fixture(scope="module")
def cli_args_template():
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return env.get_template("cli_args.j2")


def _render(template, **ctx) -> str:
    """Render the template and return stripped output."""
    return template.render(**ctx).strip()


@pytest.mark.unit
class TestDirectCliArgs:
    """Tests for flags that map directly to dynamo.trtllm argparser."""

    def test_no_model_path(self, cli_args_template):
        """model-path is no longer in cli_args.j2."""
        rendered = _render(cli_args_template, ServiceConfig={"model_path": "Qwen/Qwen3-32B"})
        assert "--model-path" not in rendered

    def test_no_served_model_name(self, cli_args_template):
        rendered = _render(
            cli_args_template,
            ServiceConfig={"model_path": "m", "served_model_name": "my-model"},
        )
        assert "--served-model-name" not in rendered

    def test_no_override_engine_args(self, cli_args_template):
        """override-engine-args block is removed."""
        rendered = _render(
            cli_args_template,
            ServiceConfig={},
            tokens_per_block=32,
            disable_overlap_scheduler=True,
        )
        assert "--override-engine-args" not in rendered

    def test_tensor_parallel_size(self, cli_args_template):
        rendered = _render(cli_args_template, ServiceConfig={}, tensor_parallel_size=4)
        assert "--tensor-parallel-size 4" in rendered

    def test_tensor_parallel_size_zero_omitted(self, cli_args_template):
        rendered = _render(cli_args_template, ServiceConfig={}, tensor_parallel_size=0)
        assert "--tensor-parallel-size" not in rendered

    def test_pipeline_parallel_size(self, cli_args_template):
        rendered = _render(cli_args_template, ServiceConfig={}, pipeline_parallel_size=2)
        assert "--pipeline-parallel-size 2" in rendered

    def test_expert_parallel_size(self, cli_args_template):
        rendered = _render(cli_args_template, ServiceConfig={}, moe_expert_parallel_size=8)
        assert "--expert-parallel-size 8" in rendered

    def test_expert_parallel_size_none_omitted(self, cli_args_template):
        rendered = _render(cli_args_template, ServiceConfig={}, moe_expert_parallel_size=None)
        assert "--expert-parallel-size" not in rendered

    def test_enable_attention_dp(self, cli_args_template):
        rendered = _render(cli_args_template, ServiceConfig={}, enable_attention_dp=True)
        assert "--enable-attention-dp" in rendered

    def test_enable_attention_dp_false_omitted(self, cli_args_template):
        rendered = _render(cli_args_template, ServiceConfig={}, enable_attention_dp=False)
        assert "--enable-attention-dp" not in rendered

    def test_max_batch_size(self, cli_args_template):
        rendered = _render(cli_args_template, ServiceConfig={}, max_batch_size=64)
        assert "--max-batch-size 64" in rendered

    def test_max_num_tokens(self, cli_args_template):
        rendered = _render(cli_args_template, ServiceConfig={}, max_num_tokens=8192)
        assert "--max-num-tokens 8192" in rendered

    def test_max_seq_len(self, cli_args_template):
        rendered = _render(cli_args_template, ServiceConfig={}, max_seq_len=4096)
        assert "--max-seq-len 4096" in rendered

    def test_free_gpu_memory_fraction(self, cli_args_template):
        rendered = _render(cli_args_template, ServiceConfig={}, kv_cache_free_gpu_memory_fraction=0.9)
        assert "--free-gpu-memory-fraction 0.9" in rendered
