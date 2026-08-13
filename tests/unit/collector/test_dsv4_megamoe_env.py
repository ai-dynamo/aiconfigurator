# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit

_MODULE = "collector.sglang.collect_dsv4_megamoe"
# collect_dsv4_megamoe transitively imports dsv4_megamoe_workload at module level,
# so that module also gets cached with mock torch. Evict it too so subsequent tests
# that need real torch operations get a fresh import rather than the mock-polluted
# cached version.
_WORKLOAD_MODULE = "collector.sglang.dsv4_megamoe_workload"
_saved_module = sys.modules.pop(_MODULE, None)
_saved_workload_module = sys.modules.pop(_WORKLOAD_MODULE, None)
_saved_torch = sys.modules.get("torch")
_saved_torch_distributed = sys.modules.get("torch.distributed")
sys.modules["torch"] = MagicMock()
sys.modules["torch.distributed"] = MagicMock()
try:
    from collector.sglang.collect_dsv4_megamoe import (
        DEFAULT_MODEL_CONFIGS,
        CaseRunResult,
        MegaMoECase,
        activation_for_lane,
        aggregate_case_run_results,
        group_cases_for_logging,
        routed_scale_for_measurement,
    )
finally:
    sys.modules.pop(_MODULE, None)
    sys.modules.pop(_WORKLOAD_MODULE, None)
    if _saved_module is not None:
        sys.modules[_MODULE] = _saved_module
    if _saved_workload_module is not None:
        sys.modules[_WORKLOAD_MODULE] = _saved_workload_module
    if _saved_torch is None:
        sys.modules.pop("torch", None)
    else:
        sys.modules["torch"] = _saved_torch
    if _saved_torch_distributed is None:
        sys.modules.pop("torch.distributed", None)
    else:
        sys.modules["torch.distributed"] = _saved_torch_distributed


def test_group_cases_for_logging_groups_seed_variants():
    cases = [
        MegaMoECase("context", 1024, "power_law_sampled_1.9", 8, 0),
        MegaMoECase("context", 1024, "power_law_sampled_1.9", 8, 1),
        MegaMoECase("context", 2048, "power_law_sampled_1.9", 8, 0),
    ]

    groups = group_cases_for_logging(cases)

    assert groups == [cases[:2], cases[2:]]


def test_aggregate_case_run_results_averages_latency_and_power():
    results = [
        CaseRunResult({"latency": "1.000000", "distribution": "power_law_sampled_1.9"}, {"power": 100.0}),
        CaseRunResult({"latency": "3.000000", "distribution": "power_law_sampled_1.9"}, {"power": 300.0}),
    ]

    aggregated = aggregate_case_run_results(results)

    assert aggregated.row["latency"] == "2.000000"
    assert aggregated.row["distribution"] == "power_law_sampled_1.9"
    assert aggregated.power_stats["power"] == 200.0


def test_routed_scale_measurement_applies_identity_factor_for_k3():
    # kimi_k3 default factor is 1.0; skipping the mul used to persist
    # includes_routed_scale=true while the timed region omitted the scale op.
    assert DEFAULT_MODEL_CONFIGS["kimi_k3"]["routed_scaling_factor"] == 1.0
    column, scale = routed_scale_for_measurement(
        include_routed_scale=True,
        routed_scaling_factor=1.0,
    )
    assert column == "true"
    assert scale == 1.0


def test_routed_scale_measurement_matches_flag_and_rejects_nonpositive():
    assert routed_scale_for_measurement(include_routed_scale=True, routed_scaling_factor=2.5) == (
        "true",
        2.5,
    )
    assert routed_scale_for_measurement(include_routed_scale=False, routed_scaling_factor=2.5) == (
        "false",
        None,
    )
    with pytest.raises(ValueError, match="positive"):
        routed_scale_for_measurement(include_routed_scale=True, routed_scaling_factor=0.0)


def test_vllm_lane_selects_situ_by_name_not_clamp_sentinel():
    """sglang keys SiTU off activation_clamp==0.03125; vLLM names the activation.

    Verified in-container at vllm v0.27.0: fp8_fp4_mega_moe takes
    ``activation: str = 'swiglu'`` and K3 serving passes activation="situ" with
    activation_clamp=None. Collecting the vLLM lane with the sglang sentinel
    would benchmark a kernel path serving never runs.
    """
    k3_clamp = DEFAULT_MODEL_CONFIGS["kimi_k3"]["activation_clamp"]
    assert k3_clamp == 0.03125
    assert activation_for_lane(pre_dispatch="vllm", activation_clamp=k3_clamp) == ("situ", None)
    assert activation_for_lane(pre_dispatch="sglang_jit", activation_clamp=k3_clamp) == ("swiglu", k3_clamp)
    # dsv4 profiles (non-sentinel clamp) stay swiglu on both lanes.
    assert activation_for_lane(pre_dispatch="vllm", activation_clamp=10.0) == ("swiglu", 10.0)
    assert activation_for_lane(pre_dispatch="sglang_jit", activation_clamp=10.0) == ("swiglu", 10.0)
