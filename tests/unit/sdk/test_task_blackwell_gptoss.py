# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the Blackwell + GPT-OSS MOE-quant promotion in TaskConfigFactory.

The promotion writes ``moe_quant_mode = "w4a8_mxfp4_mxfp8"`` into different
config keys for each serving mode:

* ``agg``    -> ``worker_config.moe_quant_mode``
* ``afd``    -> ``worker_config.moe_quant_mode``  (AFD reads the same key
                as agg, *not* the disagg-style prefill/decode keys, so the
                promotion must branch explicitly on ``afd`` rather than
                falling through to the disagg arm).
* ``disagg`` -> ``prefill_worker_config.moe_quant_mode`` and/or
                ``decode_worker_config.moe_quant_mode``.

Tests are hermetic: ``get_latest_database_version`` is monkeypatched to a
fixed string so we don't depend on LFS perf databases being downloaded.
"""

from __future__ import annotations

import pytest

from aiconfigurator.sdk import task as task_module
from aiconfigurator.sdk.task import TaskConfigFactory, TaskContext

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _stub_latest_db_version(monkeypatch):
    """Avoid touching the on-disk perf database when resolving versions."""
    monkeypatch.setattr(task_module, "get_latest_database_version", lambda **_: "test-version")


def _make_ctx(
    serving_mode: str,
    system_name: str,
    *,
    model_path: str = "openai/gpt-oss-120b",
    backend_name: str = "trtllm",
    decode_system_name: str | None = None,
) -> TaskContext:
    return TaskContext(
        serving_mode=serving_mode,
        model_path=model_path,
        model_family="GPTOSS",
        system_name=system_name,
        decode_system_name=decode_system_name,
        backend_name=backend_name,
        backend_version=None,
        isl=1024,
        osl=256,
        prefix=0,
        ttft=None,
        tpot=None,
        request_latency=None,
        enable_wideep=False,
        enable_chunked_prefill=False,
        moe_backend=None,
        total_gpus=16,
    )


def test_afd_gptoss_blackwell_promotes_worker_config_moe_quant():
    """AFD + GPT-OSS-120b + Blackwell must promote ``moe_quant_mode``.

    The AFD serving mode reads ``worker_config.moe_quant_mode`` (the same
    key as ``agg``), not the disagg-style ``prefill_worker_config`` /
    ``decode_worker_config`` keys. If the promotion only writes the
    disagg-shaped keys for AFD, the MXFP8 activation throughput win on
    B200/B300/GB200/GB300 is silently lost, so this test pins the
    correct key + audit label and asserts the disagg keys stay absent.
    """
    ctx = _make_ctx("afd", "b200_sxm")
    config, applied = TaskConfigFactory.create(ctx)

    assert config["worker_config"]["moe_quant_mode"] == "w4a8_mxfp4_mxfp8"
    assert "gptoss-blackwell-mxfp8-afd" in applied
    # AFD must not touch disagg-shaped keys (would be silently ignored).
    assert "prefill_worker_config" not in config
    assert "decode_worker_config" not in config


@pytest.mark.parametrize("system_name", ["gb200", "gb300", "b200_sxm", "b300_sxm"])
def test_afd_gptoss_promotion_covers_all_blackwell_systems(system_name):
    """Sanity: the AFD branch fires on every system in ``_blackwell_systems``."""
    ctx = _make_ctx("afd", system_name)
    config, applied = TaskConfigFactory.create(ctx)

    assert config["worker_config"]["moe_quant_mode"] == "w4a8_mxfp4_mxfp8"
    assert "gptoss-blackwell-mxfp8-afd" in applied


def test_afd_gptoss_non_blackwell_no_promotion():
    """Non-Blackwell + AFD must NOT promote; no Blackwell layer should be recorded."""
    ctx = _make_ctx("afd", "h200_sxm")
    config, applied = TaskConfigFactory.create(ctx)

    assert config["worker_config"].get("moe_quant_mode") != "w4a8_mxfp4_mxfp8"
    assert not any(label.startswith("gptoss-blackwell-mxfp8") for label in applied)


def test_agg_gptoss_blackwell_back_compat():
    """The agg branch keeps writing ``worker_config.moe_quant_mode``.

    Emits the bare ``gptoss-blackwell-mxfp8`` label (no ``-afd`` suffix).
    """
    ctx = _make_ctx("agg", "b200_sxm")
    config, applied = TaskConfigFactory.create(ctx)

    assert config["worker_config"]["moe_quant_mode"] == "w4a8_mxfp4_mxfp8"
    assert "gptoss-blackwell-mxfp8" in applied
    # The AFD-specific label is reserved for the afd branch and must not
    # leak into agg's audit log.
    assert "gptoss-blackwell-mxfp8-afd" not in applied


def test_disagg_gptoss_blackwell_back_compat_both_phases():
    """The disagg branch promotes both ``prefill_worker_config`` and ``decode_worker_config``."""
    ctx = _make_ctx("disagg", "b200_sxm")
    config, applied = TaskConfigFactory.create(ctx)

    assert config["prefill_worker_config"]["moe_quant_mode"] == "w4a8_mxfp4_mxfp8"
    assert config["decode_worker_config"]["moe_quant_mode"] == "w4a8_mxfp4_mxfp8"
    # agg-style key must not appear in disagg output (would shadow the
    # per-phase keys downstream).
    assert "moe_quant_mode" not in config.get("worker_config", {})
    assert "gptoss-blackwell-mxfp8" in applied


def test_disagg_gptoss_mixed_systems_only_blackwell_side_promoted():
    """When disagg straddles Blackwell + non-Blackwell, only the Blackwell side is promoted."""
    ctx = _make_ctx("disagg", "h200_sxm", decode_system_name="b200_sxm")
    config, applied = TaskConfigFactory.create(ctx)

    # Prefill on Hopper -> untouched.
    assert config["prefill_worker_config"].get("moe_quant_mode") != "w4a8_mxfp4_mxfp8"
    # Decode on Blackwell -> promoted.
    assert config["decode_worker_config"]["moe_quant_mode"] == "w4a8_mxfp4_mxfp8"
    assert "gptoss-blackwell-mxfp8" in applied


def test_non_gptoss_model_skips_promotion():
    """Promotion only fires for the two GPT-OSS checkpoints, not unrelated models."""
    ctx = _make_ctx("afd", "b200_sxm", model_path="deepseek-ai/DeepSeek-V3")
    # Family is wrong for the model_path, but check_is_moe re-reads the
    # config, so this is fine for the assertion below.
    config, applied = TaskConfigFactory.create(ctx)

    assert config["worker_config"].get("moe_quant_mode") != "w4a8_mxfp4_mxfp8"
    assert not any(label.startswith("gptoss-blackwell-mxfp8") for label in applied)


def test_non_trtllm_backend_skips_promotion():
    """Promotion is trtllm-only; vllm/sglang must not get the MXFP8 override."""
    ctx = _make_ctx("afd", "b200_sxm", backend_name="vllm")
    config, applied = TaskConfigFactory.create(ctx)

    assert config["worker_config"].get("moe_quant_mode") != "w4a8_mxfp4_mxfp8"
    assert not any(label.startswith("gptoss-blackwell-mxfp8") for label in applied)
