# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

__compat__ = "vllm>=0.0.0"


def get_gdn_test_cases():
    raise NotImplementedError("GDN collector not yet implemented")


def run_gdn_torch(*args, **kwargs):
    raise NotImplementedError("GDN collector not yet implemented")
