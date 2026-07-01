# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiconfigurator.generator.rendering.engine import _filter_vllm_k8s_cli_args


@pytest.mark.unit
def test_filter_vllm_k8s_cli_args_keeps_safe_minimum():
    cli_args = [
        "--tensor-parallel-size",
        "2",
        "--pipeline-parallel-size",
        "1",
        "--data-parallel-size",
        "4",
        "--enable-expert-parallel",
        "--kv-cache-dtype",
        "auto",
        "--max-model-len",
        "32768",
        "--max-num-batched-tokens",
        "8192",
        "--skip-tokenizer-init",
        "--trust-remote-code",
        "--no-enable-prefix-caching",
        "--speculative-config",
        '{"method":"mtp","num_speculative_tokens":1}',
    ]

    assert _filter_vllm_k8s_cli_args(cli_args) == cli_args


@pytest.mark.unit
def test_filter_vllm_k8s_cli_args_removes_performance_recommendations():
    cli_args = [
        "--tensor-parallel-size",
        "2",
        "--block-size",
        "16",
        "--max-num-seqs",
        "64",
        "--enforce-eager",
        "--cudagraph-capture-sizes",
        "1",
        "2",
        "4",
        "--max-model-len",
        "32768",
    ]

    assert _filter_vllm_k8s_cli_args(cli_args) == [
        "--tensor-parallel-size",
        "2",
        "--max-model-len",
        "32768",
    ]
