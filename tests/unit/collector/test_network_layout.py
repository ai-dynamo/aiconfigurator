# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
COLLECTOR_ROOT = REPO_ROOT / "packages" / "aiconfigurator-core" / "collector"
NETWORK_ROOT = COLLECTOR_ROOT / "network"


def test_network_collectors_are_grouped_under_network_folder():
    expected_paths = [
        NETWORK_ROOT / "collect_comm.sh",
        NETWORK_ROOT / "collect_nccl.py",
        NETWORK_ROOT / "collect_oneccl_xpu.py",
        NETWORK_ROOT / "collect_all_reduce.py",
        NETWORK_ROOT / "slurm" / "collect_allreduce.py",
        NETWORK_ROOT / "slurm" / "collect_trtllm_alltoall.py",
    ]

    for path in expected_paths:
        assert path.exists(), f"missing network collector: {path}"

    old_top_level_paths = [
        COLLECTOR_ROOT / "collect_comm.sh",
        COLLECTOR_ROOT / "collect_nccl.py",
        COLLECTOR_ROOT / "collect_oneccl_xpu.py",
        COLLECTOR_ROOT / "collect_all_reduce.py",
        COLLECTOR_ROOT / "slurm_comm_collector",
    ]

    for path in old_top_level_paths:
        assert not path.exists(), f"network collector should live under collector/network: {path}"


def test_slurm_network_docs_use_new_folder_name():
    docs = [
        NETWORK_ROOT / "README.md",
        NETWORK_ROOT / "slurm" / "README.md",
        NETWORK_ROOT / "slurm" / "submit_trtllm_alltoall.sh",
    ]

    for path in docs:
        assert "slurm_comm_collector" not in path.read_text(encoding="utf-8")
