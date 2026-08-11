# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import jsonschema
import pytest
from pydantic import ValidationError

from aiconfigurator.sdk.config_adapter import EstimateRequestV1, to_cli_estimate_kwargs
from aiconfigurator.sdk.config_adapter.schema import (
    AggregatedTopologyV1,
    BackendSettingsV1,
    ModelSettingsV1,
    SourceProvenanceV1,
    SystemSettingsV1,
    WorkerSettingsV1,
    WorkloadSettingsV1,
)

pytestmark = pytest.mark.unit


def _request() -> EstimateRequestV1:
    return EstimateRequestV1(
        model=ModelSettingsV1(path="QWEN/QWEN3-32B"),
        backend=BackendSettingsV1(name="trtllm", version="0.20.0", database_mode="SOL"),
        systems=SystemSettingsV1(prefill="h100_sxm"),
        workload=WorkloadSettingsV1(isl=1024, osl=128, concurrency=16),
        topology=AggregatedTopologyV1(
            worker=WorkerSettingsV1(
                replicas=1,
                gpus_per_replica=2,
                batch_size=16,
                tp_size=2,
                moe_tp_size=2,
                moe_ep_size=1,
            )
        ),
        provenance=SourceProvenanceV1(source_type="custom", assumptions=("confirmed",)),
    )


def test_schema_json_round_trip_and_json_schema_validation():
    request = _request()
    restored = EstimateRequestV1.model_validate_json(request.model_dump_json())
    schema = json.loads(EstimateRequestV1.schema_path().read_text())

    jsonschema.validate(restored.model_dump(mode="json"), schema)
    assert restored == request


def test_unknown_schema_version_and_fields_are_rejected():
    payload = _request().model_dump(mode="json")
    payload["schema_version"] = "aic-estimate-request/2.0.0"
    with pytest.raises(ValidationError, match="schema_version"):
        EstimateRequestV1.model_validate(payload)

    payload = _request().model_dump(mode="json")
    payload["unknown"] = True
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        EstimateRequestV1.model_validate(payload)


def test_packaged_json_schema_does_not_drift_from_python_model():
    expected = EstimateRequestV1.model_json_schema(by_alias=True)
    expected["$id"] = "https://github.com/ai-dynamo/aiconfigurator/schemas/aic-estimate-request-1.0.0.json"
    expected["title"] = "AIC Estimate Request v1"

    assert json.loads(EstimateRequestV1.schema_path().read_text()) == expected


def test_exact_cli_estimate_kwargs():
    assert to_cli_estimate_kwargs(_request()) == {
        "model_path": "QWEN/QWEN3-32B",
        "system_name": "h100_sxm",
        "mode": "agg",
        "backend_name": "trtllm",
        "backend_version": "0.20.0",
        "database_mode": "SOL",
        "isl": 1024,
        "osl": 128,
        "image_height": 0,
        "image_width": 0,
        "num_images": 1,
        "enable_encoder_dp": True,
        "prefix": 0,
        "nextn": 0,
        "batch_size": 16,
        "tp_size": 2,
        "pp_size": 1,
        "attention_dp_size": 1,
        "moe_tp_size": 2,
        "moe_ep_size": 1,
    }
