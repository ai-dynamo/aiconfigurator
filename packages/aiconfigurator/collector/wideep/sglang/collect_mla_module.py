# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang WideEP MLA collector entrypoints."""

from collector.sglang.collect_mla_module import (
    get_wideep_mla_context_test_cases as _get_wideep_mla_context_test_cases,
)
from collector.sglang.collect_mla_module import (
    get_wideep_mla_generation_test_cases as _get_wideep_mla_generation_test_cases,
)
from collector.sglang.collect_mla_module import run_mla_module_worker as _run_mla_module_worker


def get_wideep_mla_context_test_cases(*args, **kwargs):
    return _get_wideep_mla_context_test_cases(*args, **kwargs)


def get_wideep_mla_generation_test_cases(*args, **kwargs):
    return _get_wideep_mla_generation_test_cases(*args, **kwargs)


def run_mla_module_worker(*args, **kwargs):
    return _run_mla_module_worker(*args, **kwargs)
