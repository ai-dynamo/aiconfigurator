# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for nvfp4 dimension filtering in MOE test case generation."""

import pytest


def _nvfp4_alignment_ok(inter_size: int, tp: int) -> bool:
    """Reproduce the nvfp4 alignment check from collect_moe.get_moe_test_cases().

    Returns True if the config should be INCLUDED (passes filter).
    """
    shard_k = inter_size // tp
    # CuteDSL grouped GEMM requires 16-byte contiguous alignment:
    # for fp4 (4-bit), that's 32 elements (16 * 8 // 4 = 32).
    return shard_k % 32 == 0


@pytest.mark.unit
class TestNvfp4DimensionFilter:
    """Validate that nvfp4 configs with unsupported dimensions are filtered out."""

    @pytest.mark.parametrize(
        "inter_size,tp,expected",
        [
            # Configs that caused TypeErrors in the B200 run
            (1856, 4, False),  # K=464, 464 % 32 = 16 → reject
            (1536, 32, False),  # K=48,  48 % 32 = 16  → reject
            (2560, 32, False),  # K=80,  80 % 32 = 16  → reject
            (2688, 8, False),  # K=336, 336 % 32 = 16 → reject
            # Configs that should pass (well-aligned K)
            (2048, 1, True),  # K=2048, 2048 % 32 = 0 → accept
            (7168, 1, True),  # K=7168, 7168 % 32 = 0 → accept
            (1536, 1, True),  # K=1536, 1536 % 32 = 0 → accept
            (2048, 2, True),  # K=1024, 1024 % 32 = 0 → accept
            (2560, 16, True),  # K=160,  160 % 32 = 0  → accept
            # Edge cases
            (1536, 16, True),  # K=96,   96 % 32 = 0   → accept
            (768, 16, False),  # K=48,   48 % 32 = 16   → reject
            (1024, 32, True),  # K=32,   32 % 32 = 0   → accept (minimum aligned)
            (512, 32, False),  # K=16,   16 % 32 = 16   → reject
        ],
    )
    def test_nvfp4_alignment_check(self, inter_size, tp, expected):
        assert _nvfp4_alignment_ok(inter_size, tp) == expected

    def test_known_failing_models(self):
        """All models/TP combos that produced TypeErrors in B200 run must be rejected."""
        failing_configs = [
            # (model, inter_size, tp) from collector_log.txt TypeError tasks
            ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", 1856, 4),  # K=464
            ("MiniMaxAI/MiniMax-M2.5", 1536, 32),  # K=48
            ("Qwen/Qwen3-235B-A22B", 1536, 32),  # K=48
            ("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4", 2688, 8),  # K=336
            ("Qwen/Qwen3-Coder-480B-A35B-Instruct", 2560, 32),  # K=80
        ]
        for model, inter_size, tp in failing_configs:
            assert not _nvfp4_alignment_ok(inter_size, tp), (
                f"{model} with inter_size={inter_size}, tp={tp} (K={inter_size // tp}) should be filtered out"
            )

    def test_known_passing_models(self):
        """Standard configs with well-aligned dimensions must be accepted."""
        passing_configs = [
            ("deepseek-ai/DeepSeek-V3", 2048, 1),  # K=2048
            ("deepseek-ai/DeepSeek-V3", 2048, 2),  # K=1024
            ("moonshotai/Kimi-K2-Instruct", 2048, 1),  # K=2048
            ("Qwen/Qwen3-30B-A3B", 768, 1),  # K=768
            ("Qwen/Qwen3-235B-A22B", 1536, 1),  # K=1536
            ("MiniMaxAI/MiniMax-M2.5", 1536, 1),  # K=1536
        ]
        for model, inter_size, tp in passing_configs:
            assert _nvfp4_alignment_ok(inter_size, tp), (
                f"{model} with inter_size={inter_size}, tp={tp} (K={inter_size // tp}) should be accepted"
            )
