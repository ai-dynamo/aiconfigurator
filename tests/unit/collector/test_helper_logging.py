# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from collector.helper import _collector_log_dir_name


@pytest.mark.unit
def test_collector_log_dir_name_caps_long_op_scope():
    scope = [
        "wideep_moe",
        "mhc_module",
        "wideep_mla_generation",
        "dsa_context_module",
        "dsv4_flash_hca_generation_module",
        "dsv4_flash_paged_mqa_logits_module",
        "dsa_generation_module",
        "dsv4_flash_hca_context_module",
        "dsv4_flash_csa_generation_module",
        "dsv4_flash_csa_context_module",
        "dsv4_flash_hca_attn_module",
        "wideep_mla_context",
    ]

    log_dir = _collector_log_dir_name(scope, "20260515_184221")

    assert len(log_dir) < 255
    assert log_dir.endswith("_20260515_184221")
    assert "wideep_moe+mhc_module" in log_dir
