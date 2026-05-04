# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Version-routed wrapper for the vLLM GDN collector."""

from collector.vllm.collect_gdn import get_gdn_test_cases as _get_gdn_test_cases
from collector.vllm.collect_gdn import run_gdn_torch as _run_gdn_torch

__compat__ = "vllm>=0.17.0"


def get_gdn_test_cases():
    return _get_gdn_test_cases()


def run_gdn_torch(*args, **kwargs):
    return _run_gdn_torch(*args, **kwargs)
