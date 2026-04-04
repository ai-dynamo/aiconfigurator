# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for rendering.translate — YAML to --trtllm.* dynamic CLI flags."""

import pytest

from aiconfigurator.generator.rendering.translate import yaml_to_dynamic_flags


@pytest.mark.unit
class TestScalarConversion:
    """Basic scalar types are converted to --trtllm.key value pairs."""

    def test_top_level_bool_true(self):
        yaml_content = "disable_overlap_scheduler: true\n"
        flags = yaml_to_dynamic_flags(yaml_content)
        assert flags == ["--trtllm.disable_overlap_scheduler", "true"]

    def test_top_level_bool_false(self):
        yaml_content = "enable_chunked_prefill: false\n"
        flags = yaml_to_dynamic_flags(yaml_content)
        assert flags == ["--trtllm.enable_chunked_prefill", "false"]

    def test_top_level_int(self):
        yaml_content = "tokens_per_block: 32\n"
        flags = yaml_to_dynamic_flags(yaml_content, skip_keys=set())
        assert flags == ["--trtllm.tokens_per_block", "32"]

    def test_top_level_float(self):
        yaml_content = "some_fraction: 0.85\n"
        flags = yaml_to_dynamic_flags(yaml_content, skip_keys=set())
        assert flags == ["--trtllm.some_fraction", "0.85"]

    def test_top_level_string(self):
        yaml_content = "kv_cache_dtype: auto\n"
        flags = yaml_to_dynamic_flags(yaml_content, skip_keys=set())
        assert flags == ["--trtllm.kv_cache_dtype", "auto"]


@pytest.mark.unit
class TestNestedConversion:
    """Nested dicts produce dotted key paths."""

    def test_one_level_nesting(self):
        yaml_content = (
            "kv_cache_config:\n"
            "  dtype: auto\n"
            "  enable_block_reuse: true\n"
        )
        flags = yaml_to_dynamic_flags(yaml_content, skip_keys=set())
        assert "--trtllm.kv_cache_config.dtype" in flags
        idx = flags.index("--trtllm.kv_cache_config.dtype")
        assert flags[idx + 1] == "auto"
        assert "--trtllm.kv_cache_config.enable_block_reuse" in flags
        idx2 = flags.index("--trtllm.kv_cache_config.enable_block_reuse")
        assert flags[idx2 + 1] == "true"

    def test_two_level_nesting(self):
        yaml_content = (
            "speculative_config:\n"
            "  decoding_type: MTP\n"
        )
        flags = yaml_to_dynamic_flags(yaml_content, skip_keys=set())
        assert flags == ["--trtllm.speculative_config.decoding_type", "MTP"]

    def test_cache_transceiver_config(self):
        yaml_content = (
            "cache_transceiver_config:\n"
            "  backend: DEFAULT\n"
            "  max_tokens_in_buffer: 6528\n"
        )
        flags = yaml_to_dynamic_flags(yaml_content, skip_keys=set())
        assert "--trtllm.cache_transceiver_config.backend" in flags
        assert "--trtllm.cache_transceiver_config.max_tokens_in_buffer" in flags
        idx = flags.index("--trtllm.cache_transceiver_config.max_tokens_in_buffer")
        assert flags[idx + 1] == "6528"


@pytest.mark.unit
class TestSkipKeys:
    """Keys in the skip lists are excluded from output."""

    def test_default_skip_keys(self):
        yaml_content = (
            "backend: pytorch\n"
            "tensor_parallel_size: 4\n"
            "pipeline_parallel_size: 1\n"
            "max_batch_size: 512\n"
            "max_num_tokens: 8192\n"
            "disable_overlap_scheduler: true\n"
        )
        flags = yaml_to_dynamic_flags(yaml_content)
        flag_names = flags[::2]
        assert "--trtllm.backend" not in flag_names
        assert "--trtllm.tensor_parallel_size" not in flag_names
        assert "--trtllm.pipeline_parallel_size" not in flag_names
        assert "--trtllm.max_batch_size" not in flag_names
        assert "--trtllm.max_num_tokens" not in flag_names
        assert "--trtllm.disable_overlap_scheduler" in flag_names

    def test_default_skip_nested_keys(self):
        yaml_content = (
            "kv_cache_config:\n"
            "  free_gpu_memory_fraction: 0.80\n"
            "  dtype: auto\n"
        )
        flags = yaml_to_dynamic_flags(yaml_content)
        flag_names = flags[::2]
        assert "--trtllm.kv_cache_config.free_gpu_memory_fraction" not in flag_names
        assert "--trtllm.kv_cache_config.dtype" in flag_names

    def test_custom_skip_keys(self):
        yaml_content = "my_key: 42\n"
        flags = yaml_to_dynamic_flags(yaml_content, skip_keys={"my_key"})
        assert flags == []

    def test_custom_skip_nested_keys(self):
        yaml_content = (
            "group:\n"
            "  keep_me: 1\n"
            "  skip_me: 2\n"
        )
        flags = yaml_to_dynamic_flags(
            yaml_content,
            skip_keys=set(),
            skip_nested_keys={("group", "skip_me")},
        )
        assert "--trtllm.group.keep_me" in flags
        assert "--trtllm.group.skip_me" not in flags


@pytest.mark.unit
class TestSkipValues:
    """None, empty strings, and list values are skipped."""

    def test_none_value_skipped(self):
        yaml_content = "trust_remote_code:\n"
        flags = yaml_to_dynamic_flags(yaml_content, skip_keys=set())
        assert flags == []

    def test_empty_string_skipped(self):
        yaml_content = "trust_remote_code: ''\n"
        flags = yaml_to_dynamic_flags(yaml_content, skip_keys=set())
        assert flags == []

    def test_list_value_skipped(self):
        yaml_content = (
            "cuda_graph_config:\n"
            "  batch_sizes: [1, 2, 4, 8, 16]\n"
            "  enable_padding: true\n"
        )
        flags = yaml_to_dynamic_flags(yaml_content, skip_keys=set())
        flag_names = flags[::2]
        assert "--trtllm.cuda_graph_config.batch_sizes" not in flag_names
        assert "--trtllm.cuda_graph_config.enable_padding" in flag_names


@pytest.mark.unit
class TestRealisticYaml:
    """End-to-end test with a realistic rendered extra_engine_args YAML."""

    def test_full_yaml(self):
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
        pairs = dict(zip(flags[::2], flags[1::2]))

        assert "--trtllm.backend" not in pairs
        assert "--trtllm.tensor_parallel_size" not in pairs
        assert "--trtllm.pipeline_parallel_size" not in pairs
        assert "--trtllm.enable_attention_dp" not in pairs
        assert "--trtllm.max_batch_size" not in pairs
        assert "--trtllm.max_num_tokens" not in pairs
        assert "--trtllm.max_seq_len" not in pairs
        assert "--trtllm.kv_cache_config.free_gpu_memory_fraction" not in pairs
        assert "--trtllm.cuda_graph_config.batch_sizes" not in pairs

        assert pairs["--trtllm.enable_chunked_prefill"] == "true"
        assert pairs["--trtllm.kv_cache_config.dtype"] == "auto"
        assert pairs["--trtllm.kv_cache_config.tokens_per_block"] == "32"
        assert pairs["--trtllm.kv_cache_config.enable_block_reuse"] == "false"
        assert pairs["--trtllm.cache_transceiver_config.backend"] == "DEFAULT"
        assert pairs["--trtllm.cache_transceiver_config.max_tokens_in_buffer"] == "6528"
        assert pairs["--trtllm.cuda_graph_config.enable_padding"] == "true"
        assert pairs["--trtllm.disable_overlap_scheduler"] == "false"
        assert pairs["--trtllm.print_iter_log"] == "false"
