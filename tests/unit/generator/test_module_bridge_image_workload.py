# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The bridge must carry the Task's multimodal image workload into BenchConfig.

Image dimensions live on the Task (image_height/image_width/num_images_per_request)
but the bridge previously built BenchConfig only from explicit overrides, so
image_batch_size fell back to 0 and the benchmark artifacts ran text-only.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from aiconfigurator.generator.module_bridge import task_config_to_generator_config


def _task(*, image_height: int, image_width: int, num_images_per_request: int) -> SimpleNamespace:
    return SimpleNamespace(
        primary_backend_name="vllm",
        primary_system_name="h200_sxm",
        primary_backend_version="0.20.1",
        primary_model_path="Qwen/Qwen3-VL-4B-Instruct",
        prefix=0,
        is_moe=False,
        nextn=0,
        nextn_accepted=None,
        serving_mode="agg",
        total_gpus=0,
        system_name="h200_sxm",
        isl=256,
        osl=256,
        ttft=2000.0,
        tpot=50.0,
        image_height=image_height,
        image_width=image_width,
        num_images_per_request=num_images_per_request,
    )


def test_image_workload_populates_bench_config():
    task = _task(image_height=1024, image_width=1024, num_images_per_request=1)
    row = pd.Series({"workers": 1, "tp": 1})

    result = task_config_to_generator_config(task, row, num_gpus_per_node=4)

    bench = result["BenchConfig"]
    assert bench["image_batch_size"] == 1
    assert bench["image_width_mean"] == 1024
    assert bench["image_height_mean"] == 1024


def test_explicit_bench_override_wins():
    task = _task(image_height=1024, image_width=1024, num_images_per_request=1)
    row = pd.Series({"workers": 1, "tp": 1})

    result = task_config_to_generator_config(
        task,
        row,
        generator_overrides={"BenchConfig": {"image_batch_size": 4}},
        num_gpus_per_node=4,
    )

    assert result["BenchConfig"]["image_batch_size"] == 4
    # Task-derived dimensions still fill the unspecified fields.
    assert result["BenchConfig"]["image_width_mean"] == 1024


def test_text_only_workload_keeps_image_disabled():
    task = _task(image_height=0, image_width=0, num_images_per_request=1)
    row = pd.Series({"workers": 1, "tp": 1})

    result = task_config_to_generator_config(task, row, num_gpus_per_node=4)

    assert result["BenchConfig"]["image_batch_size"] == 0
    assert not result["BenchConfig"].get("image_width_mean")


def test_explicit_zero_image_count_disables_encoder_even_with_dimensions():
    # num_images_per_request=0 with dimensions set = "disable the image encoder"
    # (used with 448x448 by the web UI). The zero must survive (not become 1),
    # so image_batch_size stays 0 and the benchmark templates omit image args.
    task = _task(image_height=448, image_width=448, num_images_per_request=0)
    row = pd.Series({"workers": 1, "tp": 1})

    result = task_config_to_generator_config(task, row, num_gpus_per_node=4)

    assert result["BenchConfig"]["image_batch_size"] == 0
