# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multimodal image workload propagation from Task v2 into BenchConfig.

The Task carries the image workload as first-class fields (image_height,
image_width, num_images_per_request) and the search uses them, but the bridge
built BenchConfig purely from explicit --generator-set overrides. Without an
override, the schema default image_batch_size: 0 disabled every image block in
bench_run.sh / k8s_bench.yaml, so a requested VL benchmark silently ran
text-only.
"""

import pandas as pd
import pytest

from aiconfigurator.generator.api import generate_backend_artifacts
from aiconfigurator.generator.module_bridge import task_config_to_generator_config
from aiconfigurator.sdk.task_v2 import Task


def _task(**workload) -> Task:
    return Task(
        serving_mode="agg",
        model_path="Qwen/Qwen3-VL-4B-Instruct",
        system_name="h200_sxm",
        backend_name="vllm",
        isl=256,
        osl=256,
        ttft=2000.0,
        tpot=50.0,
        **workload,
    )


def _bridge(task: Task, overrides: dict | None = None) -> dict:
    return task_config_to_generator_config(
        task_config=task, result_df=pd.Series({"tp": 1}), generator_overrides=overrides
    )


@pytest.mark.unit
def test_image_workload_propagates_to_bench_config():
    task = _task(image_height=1024, image_width=1024, num_images_per_request=1)
    bench = _bridge(task)["BenchConfig"]
    assert bench["image_batch_size"] == 1
    assert bench["image_width_mean"] == 1024
    assert bench["image_height_mean"] == 1024


@pytest.mark.unit
def test_text_only_task_keeps_images_disabled():
    bench = _bridge(_task())["BenchConfig"]
    assert bench["image_batch_size"] == 0
    assert not bench.get("image_width_mean")
    assert not bench.get("image_height_mean")


@pytest.mark.unit
def test_explicit_bench_overrides_win_over_task_workload():
    task = _task(image_height=1024, image_width=1024, num_images_per_request=1)
    bench = _bridge(task, {"BenchConfig": {"image_batch_size": 4, "image_width_mean": 512}})[
        "BenchConfig"
    ]
    assert bench["image_batch_size"] == 4
    assert bench["image_width_mean"] == 512
    assert bench["image_height_mean"] == 1024


@pytest.mark.unit
def test_image_arguments_reach_benchmark_artifacts():
    task = _task(image_height=1024, image_width=1024, num_images_per_request=1)
    cfg = _bridge(task)
    artifacts = generate_backend_artifacts(
        cfg, "vllm", backend_version="0.20.1", deployment_target="dynamo-j2"
    )
    bench_artifacts = {k: v for k, v in artifacts.items() if "bench" in k}
    assert bench_artifacts, f"expected benchmark artifacts, got {sorted(artifacts)}"
    for name, content in bench_artifacts.items():
        assert "BENCH_IMAGE_BATCH_SIZE" in content, f"{name}: image block missing"
        assert "1024" in content, f"{name}: image dimensions missing"


@pytest.mark.unit
def test_text_only_benchmark_artifacts_omit_image_arguments():
    cfg = _bridge(_task())
    artifacts = generate_backend_artifacts(
        cfg, "vllm", backend_version="0.20.1", deployment_target="dynamo-j2"
    )
    for name, content in artifacts.items():
        if "bench" in name:
            assert "BENCH_IMAGE_BATCH_SIZE" not in content, f"{name}: unexpected image block"
