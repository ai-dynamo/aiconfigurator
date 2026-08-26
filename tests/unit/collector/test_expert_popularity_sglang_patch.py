# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from collector.expert_popularity.patch_sglang_flashinfer_recorder import (
    _MXFP4_ORIGINAL,
    _MXFP4_PATCHED,
    _ORIGINAL,
    _PATCHED,
)
from collector.expert_popularity.patch_sglang_flashinfer_replay_recorder import (
    _RETURN_ORIGINAL as _REPLAY_RETURN_ORIGINAL,
)
from collector.expert_popularity.patch_sglang_flashinfer_replay_recorder import (
    _RETURN_PATCHED as _REPLAY_RETURN_PATCHED,
)
from collector.expert_popularity.patch_sglang_flashinfer_replay_recorder import (
    _RUNNER_PATCHED as _REPLAY_RUNNER_PATCHED,
)
from collector.expert_popularity.patch_sglang_hash_topk_capturer import (
    _CAPTURE_PATCHED as _HASH_CAPTURE_PATCHED,
)

pytestmark = pytest.mark.unit


def test_flashinfer_bridge_preserves_bypass_outside_recording_window():
    assert "if get_global_expert_distribution_recorder().recording:" in _PATCHED
    assert "return bypassed_output.to_standard(layer_id=self.layer_id)" in _PATCHED
    assert "return bypassed_output" in _PATCHED
    assert "return BypassedTopKOutput(" in _ORIGINAL


def test_flashinfer_bridge_routes_mxfp4_through_consumed_explicit_ids():
    assert "trtllm_fp4_block_scale_routed_moe" in _MXFP4_PATCHED
    assert "topk_ids=packed_topk" in _MXFP4_PATCHED
    assert "TopKOutputChecker.format_is_bypassed" in _MXFP4_PATCHED
    assert "assert TopKOutputChecker.format_is_bypassed" in _MXFP4_ORIGINAL


def test_flashinfer_replay_bridge_records_kernel_selected_ids_without_rerouting():
    assert "routing_replay_out = torch.empty" in _REPLAY_RETURN_PATCHED
    assert 'kwargs["routing_replay_out"] = routing_replay_out' in _REPLAY_RETURN_PATCHED
    assert "recorder.on_select_experts(topk_ids=routing_replay_out)" in _REPLAY_RETURN_PATCHED
    assert "to_standard" not in _REPLAY_RETURN_PATCHED
    assert _REPLAY_RETURN_ORIGINAL.strip() == "return trtllm_fp8_block_scale_moe(**kwargs)"


def test_flashinfer_replay_bridge_keeps_hash_topk_on_routed_kernel():
    assert "TopKOutputChecker.format_is_standard" in _REPLAY_RUNNER_PATCHED
    assert "use_routed_topk = use_routed_topk or" in _REPLAY_RUNNER_PATCHED
    assert "hash-routed layers" in _REPLAY_RUNNER_PATCHED


def test_hash_topk_response_bridge_captures_produced_logical_ids_before_placement():
    capture = _HASH_CAPTURE_PATCHED.index("capturer.capture")
    placement = _HASH_CAPTURE_PATCHED.index("topk_ids_logical_to_physical")
    assert capture < placement
    assert "topk_indices=topk_ids" in _HASH_CAPTURE_PATCHED
    assert "self.layer_id is None" in _HASH_CAPTURE_PATCHED
