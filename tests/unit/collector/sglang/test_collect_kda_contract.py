# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
SOURCE_PATH = Path(__file__).resolve().parents[4] / "collector" / "sglang" / "collect_kda.py"


def test_kda_context_raises_on_conv_int32_offset_overflow():
    # Silicon-proven Triton kernel limit (see the guard's own comment in the
    # collector): the guard must span the full 3-block mixed_qkv buffer, not
    # the per-block proj_size.
    source = SOURCE_PATH.read_text(encoding="utf-8")
    assert "total_tokens * conv_channels >= 2**31" in source


def test_kda_case_phases_cover_context_generation_verify():
    # The registry getter must emit all three phases for every declared shape;
    # verify rows carry the draft-token width in the seq_len slot. Asserted on
    # the shared spec generator (importable without torch), which both the
    # sglang and vllm backend getters adapt.
    from collector.case_generator import get_common_kda_test_cases

    phases = {case.phase for case in get_common_kda_test_cases()}
    assert phases == {"context", "generation", "verify"}
