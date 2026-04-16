# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for rendering.translate — YAML to --trtllm.* dynamic CLI flags."""

import pytest

from aiconfigurator.generator.rendering.translate import yaml_to_dynamic_flags


@pytest.mark.unit
class TestYamlToDynamicFlags:
    """Core behavior: convert rendered extra_engine_args YAML to --trtllm.* flags."""

    def test_realistic_yaml(self):
        """Full YAML covering scalars, nesting, default skips, and list skip."""
        yaml_content = (
            "backend: pytorch\n"
            "\n"
            "tensor_parallel_size: 4\n"
            "pipeline_parallel_size: 1\n"
            "enable_attention_dp: false\n"
            "enable_chunked_prefill: true\n"
            "\n"
            "max_batch_size: 512\n"
            "max_num_tokens: 8192\n"
            "max_seq_len: 4096\n"
            "\n"
            "kv_cache_config:\n"
            "  free_gpu_memory_fraction: 0.80\n"
            "  dtype: auto\n"
            "  tokens_per_block: 32\n"
            "  enable_block_reuse: false\n"
            "\n"
            "cache_transceiver_config:\n"
            "  backend: DEFAULT\n"
            "  max_tokens_in_buffer: 6528\n"
            "\n"
            "cuda_graph_config:\n"
            "  enable_padding: true\n"
            "  batch_sizes: [1, 2, 4, 8, 16, 32, 64, 128, 256]\n"
            "\n"
            "disable_overlap_scheduler: false\n"
            "print_iter_log: false\n"
        )
        flags = yaml_to_dynamic_flags(yaml_content)
        pairs = dict(zip(flags[::2], flags[1::2], strict=True))

        # DEFAULT_SKIP_KEYS must be absent
        for key in (
            "backend",
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "enable_attention_dp",
            "max_batch_size",
            "max_num_tokens",
            "max_seq_len",
        ):
            assert f"--trtllm.{key}" not in pairs

        # DEFAULT_SKIP_NESTED_KEYS must be absent
        assert "--trtllm.kv_cache_config.free_gpu_memory_fraction" not in pairs

        # List values must be absent
        assert "--trtllm.cuda_graph_config.batch_sizes" not in pairs

        # Present scalar engine params (covers bool, int, str, nested)
        assert pairs["--trtllm.enable_chunked_prefill"] == "true"
        assert pairs["--trtllm.kv_cache_config.dtype"] == "auto"
        assert pairs["--trtllm.kv_cache_config.tokens_per_block"] == "32"
        assert pairs["--trtllm.kv_cache_config.enable_block_reuse"] == "false"
        assert pairs["--trtllm.cache_transceiver_config.backend"] == "DEFAULT"
        assert pairs["--trtllm.cache_transceiver_config.max_tokens_in_buffer"] == "6528"
        assert pairs["--trtllm.cuda_graph_config.enable_padding"] == "true"
        assert pairs["--trtllm.disable_overlap_scheduler"] == "false"
        assert pairs["--trtllm.print_iter_log"] == "false"

    def test_none_and_empty_values_skipped(self):
        """None and empty-string values produce no flags."""
        yaml_content = (
            "trust_remote_code:\n"  # YAML null
            "skip_tokenizer_init: ''\n"  # empty string
            "enable_chunked_prefill: true\n"
        )
        flags = yaml_to_dynamic_flags(yaml_content, skip_keys=set())
        assert flags == ["--trtllm.enable_chunked_prefill", "true"]

    def test_custom_skip_keys(self):
        """Callers can override the skip lists."""
        yaml_content = "group:\n  keep_me: 1\n  skip_me: 2\nskip_top: 42\n"
        flags = yaml_to_dynamic_flags(
            yaml_content,
            skip_keys={"skip_top"},
            skip_nested_keys={("group", "skip_me")},
        )
        flag_names = flags[::2]
        assert "--trtllm.group.keep_me" in flag_names
        assert "--trtllm.group.skip_me" not in flag_names
        assert "--trtllm.skip_top" not in flag_names

    def test_empty_yaml_returns_empty_list(self):
        """Empty or null YAML input returns an empty flag list."""
        assert yaml_to_dynamic_flags("") == []
        assert yaml_to_dynamic_flags("---\n") == []
