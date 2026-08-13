# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import jsonschema
import pytest
from pydantic import ValidationError

from aiconfigurator.sdk.config_adapter import (
    AdaptationDiagnostic,
    AdaptationOutcome,
    EstimateRequestV1,
    to_cli_estimate_kwargs,
)
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


@pytest.mark.parametrize(
    "values",
    [
        {"nextn": 1},
        {"nextn": 0, "nextn_accepted": 0.5},
        {"nextn": 1, "nextn_accepted": 1.5},
        {"nextn": "auto"},
    ],
)
def test_model_speculation_invariants_are_enforced(values):
    with pytest.raises(ValidationError):
        ModelSettingsV1(path="QWEN/QWEN3-32B", **values)


@pytest.mark.parametrize(
    "values",
    [
        {"isl": 128, "osl": 16, "concurrency": 1, "prefix": 129},
        {"isl": 128, "osl": 16, "concurrency": 1, "image_height": 32, "image_width": 0},
    ],
)
def test_workload_cross_field_invariants_are_enforced(values):
    with pytest.raises(ValidationError):
        WorkloadSettingsV1(**values)


@pytest.mark.parametrize(
    "values",
    [
        {"gpus_per_replica": 2, "tp_size": 1},
        {"gpus_per_replica": 2, "tp_size": 2, "moe_tp_size": 2},
        {"gpus_per_replica": 2, "tp_size": 2, "moe_tp_size": 1, "moe_ep_size": 1},
    ],
)
def test_worker_width_invariants_are_enforced(values):
    with pytest.raises(ValidationError):
        WorkerSettingsV1(replicas=1, batch_size=1, **values)


def test_adaptation_outcome_status_invariants_are_enforced():
    error = AdaptationDiagnostic(severity="error", code="invalid", message="invalid")
    warning = AdaptationDiagnostic(severity="warning", code="warning", message="warning")

    with pytest.raises(ValidationError, match="adapted outcome must contain a request"):
        AdaptationOutcome(point_id="point", status="adapted")
    with pytest.raises(ValidationError, match="rejected outcome cannot contain a request"):
        AdaptationOutcome(point_id="point", status="rejected", request=_request(), diagnostics=(error,))
    with pytest.raises(ValidationError, match="rejected outcome must contain an error diagnostic"):
        AdaptationOutcome(point_id="point", status="rejected", diagnostics=(warning,))


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
