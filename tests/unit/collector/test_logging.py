# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

_COLLECTOR_DIR = str(Path(__file__).resolve().parents[3] / "collector")
if _COLLECTOR_DIR not in sys.path:
    sys.path.insert(0, _COLLECTOR_DIR)

from helper import _MAX_LOG_SCOPE_NAME_BYTES, _log_scope_dir_name


def test_log_scope_dir_name_shortens_full_collector_v2_plan():
    ops = [
        "attention_context",
        "attention_generation",
        "dsa_context_module",
        "dsa_generation_module",
        "dsv4_flash_csa_context_module",
        "dsv4_flash_csa_generation_module",
        "dsv4_flash_hca_attn_module",
        "dsv4_flash_hca_context_module",
        "dsv4_flash_hca_generation_module",
        "dsv4_flash_paged_mqa_logits_module",
        "gdn",
        "gemm",
        "mamba2",
        "mhc_module",
        "mla_bmm_gen_post",
        "mla_bmm_gen_pre",
        "mla_context",
        "mla_generation",
        "moe",
        "wideep_mla_context",
        "wideep_mla_generation",
        "wideep_moe",
    ]

    dir_name = _log_scope_dir_name(ops)

    assert dir_name.startswith("attention_context+22ops+")
    assert len(dir_name.encode("utf-8")) <= _MAX_LOG_SCOPE_NAME_BYTES
    assert "/" not in dir_name


def test_log_scope_dir_name_preserves_short_scope_names():
    assert _log_scope_dir_name(["gemm"]) == "gemm"
