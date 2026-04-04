# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the trtllm cli_args.j2 template.

cli_args.j2 only emits direct CLI flags accepted by dynamo.trtllm's argparser.
Model path, served model name, and override-engine-args are handled elsewhere.
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


@pytest.mark.unit
class TestCliArgsTemplate:
    def test_emits_all_direct_flags(self, cli_args_template):
        """All direct argparser flags are present when their values are set."""
        rendered = cli_args_template.render(
            ServiceConfig={"model_path": "Qwen/Qwen3-32B", "served_model_name": "my-model"},
            tensor_parallel_size=4,
            pipeline_parallel_size=2,
            moe_expert_parallel_size=8,
            enable_attention_dp=True,
            max_batch_size=64,
            max_num_tokens=8192,
            max_seq_len=4096,
            kv_cache_free_gpu_memory_fraction=0.9,
        ).strip()

        assert "--tensor-parallel-size 4" in rendered
        assert "--pipeline-parallel-size 2" in rendered
        assert "--expert-parallel-size 8" in rendered
        assert "--enable-attention-dp" in rendered
        assert "--max-batch-size 64" in rendered
        assert "--max-num-tokens 8192" in rendered
        assert "--max-seq-len 4096" in rendered
        assert "--free-gpu-memory-fraction 0.9" in rendered

        # These must NOT be in the template (handled elsewhere)
        assert "--model-path" not in rendered
        assert "--served-model-name" not in rendered
        assert "--override-engine-args" not in rendered

    def test_omits_flags_with_zero_or_false_values(self, cli_args_template):
        """Flags are omitted when values are zero, False, or None."""
        rendered = cli_args_template.render(
            ServiceConfig={},
            tensor_parallel_size=0,
            pipeline_parallel_size=0,
            moe_expert_parallel_size=None,
            enable_attention_dp=False,
        ).strip()

        assert "--tensor-parallel-size" not in rendered
        assert "--pipeline-parallel-size" not in rendered
        assert "--expert-parallel-size" not in rendered
        assert "--enable-attention-dp" not in rendered
