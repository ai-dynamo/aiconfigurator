# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from tools.support_matrix import support_matrix as support_matrix_module
from tools.support_matrix.support_matrix import SupportMatrix, TestConstraints

pytestmark = pytest.mark.unit


def test_run_mode_reports_missing_taskrunner_result(monkeypatch):
    class NoResultTaskRunner:
        def run(self, _task_config):
            return None

    monkeypatch.setattr(support_matrix_module, "TaskRunner", NoResultTaskRunner)

    constraints = TestConstraints(total_gpus=8, isl=256, osl=256, prefix=128, ttft=2000.0, tpot=50.0)

    with pytest.raises(RuntimeError, match="TaskRunner returned no result"):
        SupportMatrix._run_mode(
            mode="agg",
            model="Qwen/Qwen3-32B",
            system="gb300",
            backend="sglang",
            version="0.5.12",
            constraints=constraints,
            engine_step_backend=None,
        )
