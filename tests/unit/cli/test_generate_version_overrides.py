# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for cli generate version override forwarding."""

import pytest

from aiconfigurator.cli import main as cli_main

pytestmark = pytest.mark.unit


def test_run_generate_mode_forwards_generator_version_overrides(cli_args_factory, monkeypatch, tmp_path, capsys):
    output_dir = tmp_path / "generated"
    output_dir.mkdir()
    (output_dir / "k8s_deploy.yaml").write_text("image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.0\n")
    calls = {}

    def fake_generate_naive_config(**kwargs):
        calls["kwargs"] = kwargs
        return {
            "generator_params": {
                "params": {"agg": {"max_batch_size": 128}},
            },
            "backend_version": "0.20.1",
            "output_dir": str(output_dir),
            "parallelism": {
                "tp": 1,
                "pp": 1,
                "gpus_per_worker": 1,
                "replicas": 8,
                "gpus_used": 8,
            },
        }

    monkeypatch.setattr(cli_main, "generate_naive_config", fake_generate_naive_config)

    args = cli_args_factory(
        mode="generate",
        backend="vllm",
        save_dir=str(tmp_path),
        extra_args=[
            "--generator-dynamo-version",
            "1.2.0",
            "--generated-config-version",
            "9.9.9-does-not-exist",
            "--generator-set",
            "K8sConfig.k8s_namespace=custom-ns",
        ],
    )

    cli_main._run_generate_mode(args)

    kwargs = calls["kwargs"]
    assert kwargs["generated_config_version"] == "9.9.9-does-not-exist"
    assert kwargs["generator_dynamo_version"] == "1.2.0"
    assert kwargs["generator_overrides"]["generator_dynamo_version"] == "1.2.0"
    assert kwargs["generator_overrides"]["K8sConfig"]["k8s_namespace"] == "custom-ns"
    capsys.readouterr()


def test_run_generate_mode_summary_handles_missing_agg_params(cli_args_factory, monkeypatch, tmp_path, capsys):
    output_dir = tmp_path / "generated"
    output_dir.mkdir()
    (output_dir / "k8s_deploy.yaml").write_text("image: test\n")

    def fake_generate_naive_config(**_kwargs):
        return {
            "generator_params": {
                "params": {
                    "prefill": {"max_batch_size": 64},
                    "decode": {"max_batch_size": 32},
                },
            },
            "backend_version": "0.20.1",
            "output_dir": str(output_dir),
            "parallelism": {
                "tp": 1,
                "pp": 1,
                "gpus_per_worker": 1,
                "replicas": 2,
                "gpus_used": 2,
            },
        }

    monkeypatch.setattr(cli_main, "generate_naive_config", fake_generate_naive_config)

    args = cli_args_factory(
        mode="generate",
        backend="vllm",
        save_dir=str(tmp_path),
    )

    cli_main._run_generate_mode(args)

    assert "Max Batch Size:  64" in capsys.readouterr().out
