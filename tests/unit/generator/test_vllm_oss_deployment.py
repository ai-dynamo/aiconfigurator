# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import yaml
from jinja2 import Environment, FileSystemLoader

from aiconfigurator.generator.rendering import apply_defaults


@pytest.mark.unit
def test_s3_model_path_enables_oss_defaults():
    service = apply_defaults(
        "ServiceConfig",
        {"model_path": "s3://aiplat/ai-lab/qwen3-0.6b/2026-07-01"},
        backend="vllm",
    )
    k8s = apply_defaults(
        "K8sConfig",
        {},
        backend="vllm",
        extra_context={"ServiceConfig": service, "generator_dynamo_version": "1.1.1"},
    )

    assert service["served_model_name"] == "qwen3-0.6b"
    assert k8s["oss_enabled"] is True
    assert k8s["oss_endpoint_url"] == "https://oss-s3.haiercash.com"
    assert k8s["k8s_image"].endswith("1.1.1-mx-pip-0.4.1-s3fix6")


@pytest.mark.unit
def test_vllm_oss_deployment_uses_model_express_and_secret_refs():
    template_dir = Path("src/aiconfigurator/generator/config/backend_templates/vllm")
    template = Environment(loader=FileSystemLoader(template_dir)).get_template("k8s_deploy.yaml.j2")
    rendered = template.render(
        working_dir="/workspace/examples/backends/vllm",
        K8sConfig={
            "oss_enabled": True,
            "oss_endpoint_url": "https://oss-s3.haiercash.com",
            "oss_region": "cn-east-1",
            "oss_secret_name": "oss-s3-secret",
            "oss_model_express_url": "http://model-express:8001",
            "oss_streamer_concurrency": 4,
            "k8s_namespace": "aic-system",
            "k8s_image": "example/vllm:modelexpress",
            "frontend_node_selector": {"kubernetes.io/hostname": "master2"},
            "worker_node_selector": {"kubernetes.io/hostname": "master2"},
        },
        ServiceConfig={
            "model_path": "s3://aiplat/ai-lab/qwen3-0.6b/2026-07-01",
            "served_model_name": "qwen3-0.6b",
        },
        DynConfig={"mode": "agg", "enable_router": False},
        name="dynamo-qwen3-oss",
        frontend_replicas=1,
        agg_workers=1,
        agg_gpu=1,
        agg_cli_args_list=["--tensor-parallel-size", "1", "--max-model-len", "4096"],
    )

    deployment = yaml.safe_load(rendered)
    worker = deployment["spec"]["services"]["VllmWorker"]["extraPodSpec"]["mainContainer"]
    script = worker["args"][0]
    env = {item["name"]: item for item in worker["env"]}

    assert env["MX_MODEL_URI"]["value"] == "s3://aiplat/ai-lab/qwen3-0.6b/2026-07-01"
    assert env["AWS_ENDPOINT_URL"]["value"] == "https://oss-s3.haiercash.com"
    assert env["AWS_ACCESS_KEY_ID"]["valueFrom"]["secretKeyRef"]["name"] == "oss-s3-secret"
    assert "--load-format modelexpress" in script
    assert 'worker_cmd+=( "--tensor-parallel-size" )' in script
    assert "config.json" in script
