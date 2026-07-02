# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DGD YAML serialization must keep boolean-like env values as strings.

Kubernetes converts the generated YAML to JSON with a YAML 1.1 parser before the
DGD admission webhook validates it. A bare scalar such as ``value: y`` is read as
a boolean and rejected because ``EnvVar.value`` must be a string, so every
boolean-like string the generator emits must stay quoted.
"""

from __future__ import annotations

import pytest
import yaml

from aiconfigurator.generator.builders.dgd_model import DGD

# Tokens a YAML 1.1 parser resolves to a boolean (Kubernetes' behavior).
BOOLISH_TOKENS = ["y", "Y", "n", "N", "yes", "no", "on", "off", "true", "false", "YES", "Off"]


def _dgd_with_env_value(value: str) -> DGD:
    doc = {
        "apiVersion": "nvidia.com/v1alpha1",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": "dynamo-agg"},
        "spec": {
            "services": {
                "Worker": {
                    "extraPodSpec": {
                        "mainContainer": {
                            "env": [{"name": "UCX_CUDA_IPC_ENABLE_MNNVL", "value": value}],
                        }
                    }
                }
            }
        },
    }
    return DGD.from_dict(doc)


@pytest.mark.parametrize("token", BOOLISH_TOKENS)
def test_boolean_like_env_value_is_quoted(token: str):
    text = _dgd_with_env_value(token).to_yaml()

    # The scalar must be quoted, never emitted bare (a bare token is read as bool).
    assert f"value: {token}\n" not in text
    assert f'value: "{token}"' in text

    # And it must round-trip back as a string.
    reloaded = yaml.safe_load(text)
    env = reloaded["spec"]["services"]["Worker"]["extraPodSpec"]["mainContainer"]["env"][0]
    assert env["value"] == token
    assert isinstance(env["value"], str)


def test_plain_strings_are_not_quoted():
    # Non-boolean strings keep the compact plain style (no gratuitous quoting).
    text = _dgd_with_env_value("/workspace/model_cache").to_yaml()
    assert "value: /workspace/model_cache\n" in text
