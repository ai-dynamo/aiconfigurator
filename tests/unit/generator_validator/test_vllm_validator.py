# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for vLLM generator validator directory mode."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import yaml

from tools.generator_validator import validator
from tools.generator_validator.backend import vllm as vllm_backend

pytestmark = pytest.mark.unit


class _FakeEngineArgs:
    def __init__(self, parsed):
        self.parsed = parsed

    @staticmethod
    def add_cli_args(parser):
        parser.add_argument("--model")
        parser.add_argument("--served-model-name")
        parser.add_argument("--tensor-parallel-size")
        parser.add_argument("--max-model-len")

    @classmethod
    def from_cli_args(cls, parsed):
        return cls(parsed)

    def create_engine_config(self):
        return {"model": self.parsed.model}


def _fake_vllm_import():
    return _FakeEngineArgs, argparse.ArgumentParser


def _write_k8s_manifest(path: Path, services: dict[str, list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "apiVersion": "nvidia.com/v1",
        "kind": "DynamoGraphDeployment",
        "spec": {
            "services": {
                service_name: {
                    "extraPodSpec": {
                        "mainContainer": {
                            "args": args,
                        },
                    },
                }
                for service_name, args in services.items()
            },
        },
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def _base_vllm_args() -> list[str]:
    return [
        "--model",
        "Qwen/Qwen3-0.6B",
        "--served-model-name",
        "qwen3",
        "--tensor-parallel-size",
        "1",
        "--max-model-len",
        "4096",
    ]


def _write_vllm_result_tree(root: Path) -> None:
    _write_k8s_manifest(
        root / "agg" / "top1" / "k8s_deploy.yaml",
        {
            "VllmWorker": _base_vllm_args(),
        },
    )
    _write_k8s_manifest(
        root / "disagg" / "top1" / "k8s_deploy.yaml",
        {
            "VllmPrefillWorker": [*_base_vllm_args(), "--is-prefill-worker"],
            "VllmDecodeWorker": [*_base_vllm_args(), "--is-decode-worker"],
        },
    )


def test_vllm_directory_mode_uses_vllm_worker_for_agg(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(vllm_backend, "_import_vllm_engine_args", _fake_vllm_import)
    _write_vllm_result_tree(tmp_path)

    paths = vllm_backend.collect_config_paths(tmp_path)
    assert paths[0] == ("agg", tmp_path / "agg" / "top1" / "k8s_deploy.yaml", "VllmWorker")

    result = validator.main(["--backend", "vllm", "--path", str(tmp_path)])

    assert result == 0
    assert "Result" in capsys.readouterr().out


def test_missing_explicit_vllm_service_does_not_fallback_to_manifest(tmp_path):
    manifest_path = tmp_path / "k8s_deploy.yaml"
    _write_k8s_manifest(manifest_path, {"VllmWorker": _base_vllm_args()})

    with pytest.raises(ValueError, match="Unable to locate vLLM service 'VllmDecodeWorker'"):
        vllm_backend.validate_vllm_engine_config_file(
            str(manifest_path),
            model_path=None,
            service_key="VllmDecodeWorker",
        )


@pytest.mark.parametrize(
    "spec",
    [
        {"services": None},
        "not-a-mapping",
    ],
)
def test_missing_vllm_services_mapping_reports_clear_error(tmp_path, spec):
    manifest_path = tmp_path / "k8s_deploy.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "apiVersion": "nvidia.com/v1",
                "kind": "DynamoGraphDeployment",
                "spec": spec,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unable to locate vLLM services mapping"):
        vllm_backend.validate_vllm_engine_config_file(
            str(manifest_path),
            model_path=None,
            service_key="VllmWorker",
        )


def test_vllm_engine_args_yaml_still_uses_dict_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(vllm_backend, "_import_vllm_engine_args", _fake_vllm_import)
    config_path = tmp_path / "engine_args.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": "Qwen/Qwen3-0.6B",
                "served_model_name": "qwen3",
                "tensor_parallel_size": 1,
                "max_model_len": 4096,
            }
        ),
        encoding="utf-8",
    )

    config, resolved_model = vllm_backend.validate_vllm_engine_config_file(
        str(config_path),
        model_path=None,
    )

    assert config == {"model": "Qwen/Qwen3-0.6B"}
    assert resolved_model is None
