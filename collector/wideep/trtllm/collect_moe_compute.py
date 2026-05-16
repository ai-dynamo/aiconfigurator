# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT-LLM WideEP MoE compute collector entrypoint."""

__compat__ = "trtllm>=1.2.0rc6"

from collector.trtllm.collect_wideep_moe_compute import (
    get_wideep_moe_compute_all_test_cases as _get_wideep_moe_compute_all_test_cases,
)
from collector.trtllm.collect_wideep_moe_compute import (
    run_wideep_moe_compute as _run_wideep_moe_compute,
)


def get_wideep_moe_compute_all_test_cases(*args, **kwargs):
    return _get_wideep_moe_compute_all_test_cases(*args, **kwargs)


def run_wideep_moe_compute(*args, **kwargs):
    return _run_wideep_moe_compute(*args, **kwargs)
