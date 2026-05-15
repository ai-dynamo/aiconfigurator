# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from collector.common_test_cases import get_common_moe_test_cases


@pytest.mark.unit
def test_qwen35_35b_a3b_moe_shape_has_gb300_required_tp_case():
    cases = [case for case in get_common_moe_test_cases() if case.model_name == "Qwen/Qwen3.5-35B-A3B"]

    assert cases
    assert any(
        case.hidden_size == 2048
        and case.inter_size == 512
        and case.topk == 8
        and case.num_experts == 256
        and case.tp == 8
        and case.ep == 1
        and case.token_expert_distribution == "power_law"
        and case.power_law_alpha == 1.2
        for case in cases
    )
