# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the trtllm cli_args.j2 template (NVBug 5974038)."""

import json
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


def _parse_override_engine_args(rendered: str) -> dict:
    """Extract and parse the JSON value of --override-engine-args from rendered CLI string."""
    args = shlex.split(rendered)
    idx = args.index("--override-engine-args")
    return json.loads(args[idx + 1])


@pytest.mark.unit
class TestCacheTransceiverConfig:
    """Regression tests for NVBug 5974038: cache_transceiver_config.backend missing in disagg config."""

    def test_backend_field_present_when_cache_transceiver_defined(self, cli_args_template):
        """cache_transceiver_config must include backend: DEFAULT — the missing field that caused AssertionError."""
        rendered = _render(
            cli_args_template,
            ServiceConfig={"model_path": "Qwen/Qwen3-0.6B"},
            cache_transceiver_max_tokens_in_buffer=6528,
        )
        override = _parse_override_engine_args(rendered)
        assert "cache_transceiver_config" in override
        assert override["cache_transceiver_config"]["backend"] == "DEFAULT"

    def test_max_tokens_in_buffer_field_present(self, cli_args_template):
        """cache_transceiver_config must also retain max_tokens_in_buffer."""
        rendered = _render(
            cli_args_template,
            ServiceConfig={"model_path": "Qwen/Qwen3-0.6B"},
            cache_transceiver_max_tokens_in_buffer=6528,
        )
        override = _parse_override_engine_args(rendered)
        assert override["cache_transceiver_config"]["max_tokens_in_buffer"] == 6528

    def test_cache_transceiver_absent_when_not_defined(self, cli_args_template):
        """cache_transceiver_config must not appear if cache_transceiver_max_tokens_in_buffer is not set."""
        rendered = _render(
            cli_args_template,
            ServiceConfig={"model_path": "Qwen/Qwen3-0.6B"},
        )
        assert "cache_transceiver_config" not in rendered

    def test_cache_transceiver_combined_with_other_engine_args(self, cli_args_template):
        """cache_transceiver_config coexists correctly with kv_cache_config and disable_overlap_scheduler."""
        rendered = _render(
            cli_args_template,
            ServiceConfig={"model_path": "Qwen/Qwen3-0.6B"},
            cache_transceiver_max_tokens_in_buffer=6528,
            tokens_per_block=32,
            kv_cache_dtype="auto",
            disable_overlap_scheduler=True,
        )
        override = _parse_override_engine_args(rendered)
        assert override["cache_transceiver_config"] == {"max_tokens_in_buffer": 6528, "backend": "DEFAULT"}
        assert override["kv_cache_config"]["tokens_per_block"] == 32
        assert override["kv_cache_config"]["dtype"] == "auto"
        assert override["disable_overlap_scheduler"] is True


@pytest.mark.unit
class TestDirectCliArgs:
    """Tests for flags that map directly to dynamo.trtllm argparser."""

    def test_model_path(self, cli_args_template):
        rendered = _render(cli_args_template, ServiceConfig={"model_path": "Qwen/Qwen3-32B"})
        assert '--model-path "Qwen/Qwen3-32B"' in rendered

    def test_served_model_name(self, cli_args_template):
        rendered = _render(
            cli_args_template,
            ServiceConfig={"model_path": "m", "served_model_name": "my-model"},
        )
        assert '--served-model-name "my-model"' in rendered

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


@pytest.mark.unit
class TestOverrideEngineArgs:
    """Tests for parameters that go into --override-engine-args JSON."""

    def test_no_override_engine_args_when_empty(self, cli_args_template):
        rendered = _render(cli_args_template, ServiceConfig={"model_path": "m"})
        assert "--override-engine-args" not in rendered

    def test_kv_cache_tokens_per_block(self, cli_args_template):
        rendered = _render(cli_args_template, ServiceConfig={}, tokens_per_block=32)
        override = _parse_override_engine_args(rendered)
        assert override["kv_cache_config"]["tokens_per_block"] == 32

    def test_kv_cache_dtype(self, cli_args_template):
        rendered = _render(cli_args_template, ServiceConfig={}, kv_cache_dtype="fp8")
        override = _parse_override_engine_args(rendered)
        assert override["kv_cache_config"]["dtype"] == "fp8"

    def test_disable_prefix_cache_sets_enable_block_reuse_false(self, cli_args_template):
        rendered = _render(cli_args_template, ServiceConfig={}, disable_prefix_cache=True)
        override = _parse_override_engine_args(rendered)
        assert override["kv_cache_config"]["enable_block_reuse"] is False

    def test_enable_prefix_cache_sets_enable_block_reuse_true(self, cli_args_template):
        rendered = _render(cli_args_template, ServiceConfig={}, disable_prefix_cache=False)
        override = _parse_override_engine_args(rendered)
        assert override["kv_cache_config"]["enable_block_reuse"] is True

    def test_disable_overlap_scheduler(self, cli_args_template):
        rendered = _render(cli_args_template, ServiceConfig={}, disable_overlap_scheduler=True)
        override = _parse_override_engine_args(rendered)
        assert override["disable_overlap_scheduler"] is True

    def test_skip_tokenizer_init(self, cli_args_template):
        rendered = _render(cli_args_template, ServiceConfig={}, skip_tokenizer_init=True)
        override = _parse_override_engine_args(rendered)
        assert override["skip_tokenizer_init"] is True

    def test_trust_remote_code(self, cli_args_template):
        rendered = _render(cli_args_template, ServiceConfig={}, trust_remote_code=True)
        override = _parse_override_engine_args(rendered)
        assert override["trust_remote_code"] is True

    def test_enable_chunked_prefill(self, cli_args_template):
        rendered = _render(cli_args_template, ServiceConfig={}, enable_chunked_prefill=True)
        override = _parse_override_engine_args(rendered)
        assert override["enable_chunked_prefill"] is True

    def test_override_engine_args_json_is_valid(self, cli_args_template):
        """The escaped JSON in --override-engine-args must round-trip cleanly through shlex."""
        rendered = _render(
            cli_args_template,
            ServiceConfig={"model_path": "Qwen/Qwen3-0.6B"},
            cache_transceiver_max_tokens_in_buffer=6528,
            tokens_per_block=32,
            kv_cache_dtype="auto",
            disable_overlap_scheduler=True,
        )
        override = _parse_override_engine_args(rendered)
        assert isinstance(override, dict)
