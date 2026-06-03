# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collector import helper


def test_collector_log_scope_name_is_bounded_for_full_model_case_ops():
    scope = [f"long_collector_operation_name_{idx:02d}" for idx in range(50)]

    name = helper._collector_log_scope_name(scope)

    assert len(name.encode()) <= helper._MAX_LOG_SCOPE_BYTES
    assert "50ops" in name
    assert name != "+".join(scope)
