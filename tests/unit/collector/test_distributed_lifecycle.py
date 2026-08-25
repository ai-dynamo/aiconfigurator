# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from collector.wideep.distributed_lifecycle import (
    DistributedLifecycleError,
    agree_stage,
    raise_for_stage,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
COLLECTORS = {
    "sglang": REPO_ROOT / "collector" / "wideep" / "sglang" / "collect_moe_a2a.py",
    "trtllm": REPO_ROOT / "collector" / "wideep" / "trtllm" / "collect_moe_a2a.py",
    "vllm": REPO_ROOT / "collector" / "wideep" / "vllm" / "collect_moe_a2a.py",
}


def test_stage_agreement_preserves_order_and_local_error():
    stages = []

    def agreement(stage, failed):
        stages.append((stage, failed))
        return failed

    error = RuntimeError("root write failed")
    outcome = agree_stage("row_write", error, agreement=agreement)

    assert outcome.error is error
    assert stages == [("row_write", True)]
    with pytest.raises(RuntimeError, match="root write failed"):
        raise_for_stage(outcome)


def test_successful_peer_receives_named_error_when_another_rank_fails():
    outcome = agree_stage(
        "sidecar_write",
        None,
        agreement=lambda stage, failed: stage == "sidecar_write" and not failed,
    )

    assert isinstance(outcome.error, DistributedLifecycleError)
    with pytest.raises(DistributedLifecycleError, match="sidecar_write"):
        raise_for_stage(outcome)


def test_successful_stage_does_not_raise():
    outcome = agree_stage("final_ready", None, agreement=lambda _stage, failed: failed)

    assert not outcome.failed
    raise_for_stage(outcome)


@pytest.mark.parametrize("framework,source_path", COLLECTORS.items())
def test_collectors_agree_before_every_publish_stage_and_barrier(framework, source_path):
    source = source_path.read_text(encoding="utf-8")
    stage_offsets = [
        source.index('"preflight"'),
        source.index('"parquet_finalize"'),
        source.index('"sidecar_write"'),
        source.index('"final_ready"'),
    ]

    assert stage_offsets == sorted(stage_offsets), framework
    assert ':benchmark"' in source, framework
    assert ':row_write"' in source or '"row_write"' in source, framework
    assert "agree_stage(" in source, framework

    final_ready = stage_offsets[-1]
    barrier = source.find("barrier", final_ready)
    assert barrier > final_ready, framework
