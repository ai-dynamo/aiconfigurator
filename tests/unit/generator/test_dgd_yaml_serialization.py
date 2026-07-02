# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""K8s-safe YAML serialization of the typed DGD document models.

Kubernetes converts manifests YAML->JSON with go-yaml (full YAML 1.1), whose
implicit boolean token set includes the single letters y/Y/n/N that PyYAML's
resolver omits. PyYAML therefore emits the Python string "y" as the plain
scalar ``y``, which the API server decodes as the JSON boolean ``true`` and
the DGD admission webhook rejects (EnvVar.value must be a string). These
tests pin that every boolean-like string survives serialization as a quoted
scalar in every emitted K8s document type.
"""

import re

import pytest
import yaml

from aiconfigurator.generator.builders.dgd_model import (
    DGD,
    ComputeDomainDoc,
    ConfigMapDoc,
    DGDService,
    ExtraPodSpec,
    MainContainer,
    dgd_documents_to_yaml,
)

# The go-yaml (YAML 1.1) implicit boolean tokens. PyYAML quotes the multi-letter
# ones on its own; the single letters y/Y/n/N are the regression trigger.
GO_YAML_BOOL_TOKENS = [
    "y", "Y", "n", "N",
    "yes", "Yes", "YES", "no", "No", "NO",
    "true", "True", "TRUE", "false", "False", "FALSE",
    "on", "On", "ON", "off", "Off", "OFF",
]

_UNQUOTED_BOOL_LIKE = re.compile(
    r":\s+(?:y|Y|n|N|yes|Yes|YES|no|No|NO|true|True|TRUE"
    r"|false|False|FALSE|on|On|ON|off|Off|OFF)\s*$",
    re.MULTILINE,
)


def _dgd_with_env(value: str) -> DGD:
    container = MainContainer(
        image="nvcr.io/nvstaging/ai-dynamo/sglang-runtime:1.3.0-rc1",
        env=[{"name": "UCX_CUDA_IPC_ENABLE_MNNVL", "value": value}],
    )
    service = DGDService(
        component_type="worker",
        replicas=1,
        extra_pod_spec=ExtraPodSpec(main_container=container),
    )
    return DGD(name="dynamo-agg", namespace="qchi-aic", services={"SGLangWorker": service})


@pytest.mark.unit
@pytest.mark.parametrize("token", GO_YAML_BOOL_TOKENS)
def test_dgd_env_bool_like_value_is_quoted(token):
    out = _dgd_with_env(token).to_yaml()
    assert not _UNQUOTED_BOOL_LIKE.search(out), (
        f"boolean-like env value {token!r} emitted as a plain scalar; "
        "Kubernetes YAML-to-JSON would decode it as a boolean"
    )
    doc = yaml.safe_load(out)
    env = doc["spec"]["services"]["SGLangWorker"]["extraPodSpec"]["mainContainer"]["env"]
    assert env[0]["value"] == token
    assert isinstance(env[0]["value"], str)


@pytest.mark.unit
def test_dgd_regular_strings_stay_plain():
    out = _dgd_with_env("1").to_yaml()
    # Non-ambiguous strings must not pick up gratuitous quoting.
    assert "image: nvcr.io/nvstaging/ai-dynamo/sglang-runtime:1.3.0-rc1" in out
    assert "name: dynamo-agg" in out
    assert "componentType: worker" in out


@pytest.mark.unit
def test_dgd_round_trip_preserves_env_value():
    dgd = _dgd_with_env("y")
    reparsed = DGD.from_dict(yaml.safe_load(dgd.to_yaml()))
    assert reparsed.to_dict() == dgd.to_dict()


@pytest.mark.unit
@pytest.mark.parametrize("token", ["y", "n", "on"])
def test_configmap_data_bool_like_value_is_quoted(token):
    doc = ConfigMapDoc(name="engine-configs", namespace="qchi-aic", data={"FLAG": token})
    out = doc.to_yaml()
    assert not _UNQUOTED_BOOL_LIKE.search(out)
    assert yaml.safe_load(out)["data"]["FLAG"] == token


@pytest.mark.unit
def test_compute_domain_serialization_unaffected():
    doc = ComputeDomainDoc(
        name="dynamo-agg-compute-domain",
        namespace="qchi-aic",
        channel_name="dynamo-agg-channel",
        num_nodes=2,
    )
    parsed = yaml.safe_load(doc.to_yaml())
    assert parsed["spec"]["numNodes"] == 2
    assert parsed["spec"]["channel"]["resourceClaimTemplate"]["name"] == "dynamo-agg-channel"


@pytest.mark.unit
def test_multi_doc_stream_quotes_bool_like_values():
    docs = [
        _dgd_with_env("y"),
        ConfigMapDoc(name="engine-configs", data={"FLAG": "n"}),
    ]
    out = dgd_documents_to_yaml(docs)
    assert not _UNQUOTED_BOOL_LIKE.search(out)
    loaded = list(yaml.safe_load_all(out))
    assert len(loaded) == 2
