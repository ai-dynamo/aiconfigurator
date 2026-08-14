# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""EPD single-point evaluation against a real packaged perf database.

Synthetic databases route the engine step to the Python path, so only a
real database exercises the default compiled-engine path where the encode
worker's ``EncoderOnlyModel`` must satisfy the full-engine spec contract.
"""

import json

import pytest

from aiconfigurator.sdk import config
from aiconfigurator.sdk.task_v2 import Task

_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
_SYSTEM = "h200_sxm"
_BACKEND = "sglang"
_WORKLOAD = dict(
    isl=2048,
    osl=256,
    image_height=768,
    image_width=768,
    num_images_per_request=2,
)


def test_encoder_only_model_satisfies_engine_spec_contract():
    from aiconfigurator_core.sdk.engine import build_engine_spec_json
    from aiconfigurator_core.sdk.models.vit_ops import EncoderOnlyModel

    model = EncoderOnlyModel(
        encoder_ops=[],
        encoder_config=None,
        config=config.ModelConfig(tp_size=2, enable_encoder_dp=False),
    )
    spec = json.loads(
        build_engine_spec_json(
            model,
            model_path="epd-encoder-only",
            system=_SYSTEM,
            backend=_BACKEND,
            backend_version=None,
            kv_block_size=None,
            systems_path=None,
            nextn=0,
        )
    )
    assert spec["context_ops"] == []
    assert spec["generation_ops"] == []


@pytest.mark.parametrize("engine_step_backend", [None, "rust"])
def test_run_single_agg_epd_real_database(engine_step_backend):
    task = Task(
        serving_mode="agg",
        model_path=_MODEL,
        system_name=_SYSTEM,
        backend_name=_BACKEND,
        enable_epd=True,
        engine_step_backend=engine_step_backend,
        **_WORKLOAD,
    )
    row = task.run_single_agg(tp=1, batch_size=1, encoder_tp=1)
    assert row["(e)workers"] == 1
    assert row["encoder_latency"] > 0
    assert row["ttft"] > row["encoder_latency"]


@pytest.mark.parametrize("engine_step_backend", [None, "rust"])
def test_run_single_disagg_epd_real_database(engine_step_backend):
    task = Task(
        serving_mode="disagg",
        prefill_model_path=_MODEL,
        decode_model_path=_MODEL,
        prefill_system_name=_SYSTEM,
        decode_system_name=_SYSTEM,
        prefill_backend_name=_BACKEND,
        decode_backend_name=_BACKEND,
        enable_epd=True,
        engine_step_backend=engine_step_backend,
        **_WORKLOAD,
    )
    row = task.run_single_disagg(
        prefill_tp=1,
        decode_tp=1,
        decode_batch_size=8,
        encoder_tp=1,
    )
    assert row["(e)workers"] == 1
    assert row["encoder_latency"] > 0
    assert row["ttft"] > row["encoder_latency"]
