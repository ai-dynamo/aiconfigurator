# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiconfigurator.sdk.utils import representative_parallel_failure


def test_representative_parallel_failure_skips_trailing_invalid_tp_candidate():
    missing_perf_data = RuntimeError("Missing silicon data for the requested lookup.")
    invalid_tp = AssertionError("num_heads 24 should be divisible by tp_size 16")

    assert representative_parallel_failure([missing_perf_data, invalid_tp]) is missing_perf_data


def test_representative_parallel_failure_keeps_last_when_all_failures_are_invalid_configs():
    invalid_width = ValueError(
        "Parallelism width mismatch: tp_size(2) * attention_dp_size(1) = 2, but moe_tp_size(1) * moe_ep_size(1) = 1."
    )
    invalid_tp = AssertionError("num_heads 24 should be divisible by tp_size 16")

    assert representative_parallel_failure([invalid_width, invalid_tp]) is invalid_tp
