# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse

import pytest

from collector.fpm_forward.config import reject_fpm_arguments_without_fpm

pytestmark = pytest.mark.unit


def test_normal_op_arguments_do_not_activate_fpm():
    args = argparse.Namespace(
        ops=["gemm"],
        fpm_max_gpus=None,
        fpm_gpu_counts=None,
        fpm_weight_quantizations=None,
        fpm_kv_cache_dtypes=None,
        fpm_tp_sizes=None,
        fpm_pp_sizes=None,
        fpm_dp_sizes=None,
        fpm_moe_tp_sizes=None,
        fpm_smoke_points=None,
        fpm_database_root=None,
        fpm_moe_ep_sizes=None,
        fpm_cp_sizes=None,
    )
    reject_fpm_arguments_without_fpm(args)


def test_fpm_only_arguments_are_rejected_for_normal_ops():
    args = argparse.Namespace(
        ops=["gemm"],
        fpm_max_gpus=4,
        fpm_gpu_counts=None,
        fpm_weight_quantizations=None,
        fpm_kv_cache_dtypes=None,
        fpm_tp_sizes=None,
        fpm_pp_sizes=None,
        fpm_dp_sizes=None,
        fpm_moe_tp_sizes=None,
        fpm_smoke_points=None,
        fpm_database_root=None,
        fpm_moe_ep_sizes=None,
        fpm_cp_sizes=None,
    )
    with pytest.raises(ValueError, match="require --ops fpm_forward"):
        reject_fpm_arguments_without_fpm(args)
