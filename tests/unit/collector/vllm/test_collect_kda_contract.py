# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Phase coverage of the shared KDA case generator is pinned once, in
# tests/unit/collector/sglang/test_collect_kda_contract.py (both backend
# getters adapt the same generator).

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
SOURCE_PATH = Path(__file__).resolve().parents[4] / "collector" / "vllm" / "collect_kda.py"


def test_kda_context_raises_on_conv_int32_offset_overflow():
    # Same int32 token-offset guard as the sglang kda collector: the guard
    # must RAISE inside the sweep loop so the cell lands in the classified
    # failure log instead of corrupting the CUDA context.
    source = SOURCE_PATH.read_text(encoding="utf-8")
    assert "2**31" in source
