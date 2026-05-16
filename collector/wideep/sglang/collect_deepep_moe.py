# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang DeepEP WideEP MoE collector entrypoint."""

__compat__ = "sglang>=0.5.10"

from collector.sglang.collect_wideep_deepep_moe import (
    get_wideep_moe_test_cases as _get_wideep_moe_test_cases,
)
from collector.sglang.collect_wideep_deepep_moe import run_wideep_moe as _run_wideep_moe


def get_wideep_moe_test_cases(*args, **kwargs):
    return _get_wideep_moe_test_cases(*args, **kwargs)


def run_wideep_moe(*args, **kwargs):
    return _run_wideep_moe(*args, **kwargs)
