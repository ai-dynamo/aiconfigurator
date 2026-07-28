# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
    assert "int32 token-offset overflow" in source


def test_kda_dispatch_mirrors_serving():
    # The collector must dispatch prefill like serving (FlashKDA when
    # supported, Triton fallback) and probe the fused decode kernel via the
    # same predicate serving uses — never pin a kernel unconditionally.
    source = SOURCE_PATH.read_text(encoding="utf-8")
    assert "is_flashkda_supported" in source
    assert "is_fused_kda_decode_supported" in source
    for phase in ("context", "generation", "verify"):
        assert f'"{phase}"' in source
