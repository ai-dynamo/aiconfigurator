#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapt known server configs, validate canonical requests, and optionally estimate."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

import jsonschema

from aiconfigurator.cli.api import cli_estimate
from aiconfigurator.sdk.config_adapter import (
    AdaptationOutcome,
    AdaptationReport,
    AdapterOverrides,
    DynamoRecipeSource,
    EstimateRequestV1,
    InferenceXSource,
    adapt_config,
    to_cli_estimate_kwargs,
)


def _json_object(value: str) -> dict[str, Any]:
    try:
        path = Path(value)
        text = path.read_text() if path.is_file() else value
    except OSError:
        text = value
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise TypeError("expected a JSON object")
    return parsed


def _schema() -> dict[str, Any]:
    resource = files("aiconfigurator.sdk.config_adapter") / "schemas" / "estimate-request-v1.schema.json"
    parsed = json.loads(resource.read_text())
    if not isinstance(parsed, dict):
        raise TypeError("packaged estimate-request schema must be an object")
    return parsed


def _validate_request(request: EstimateRequestV1, schema: dict[str, Any]) -> None:
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(request.model_dump(mode="json"), schema)


def _serialize_estimate(result: Any) -> Any:
    if hasattr(result, "model_dump"):
        return result.model_dump(mode="json")
    if is_dataclass(result):
        return asdict(result)
    if hasattr(result, "__dict__"):
        return vars(result)
    return str(result)


def _report(args: argparse.Namespace) -> AdaptationReport:
    overrides = (
        AdapterOverrides.model_validate_json(json.dumps(_json_object(args.overrides)))
        if args.overrides
        else AdapterOverrides()
    )
    if args.format == "inferencex":
        if not args.config or not args.benchmark:
            raise ValueError("--format inferencex requires --config and --benchmark")
        source = InferenceXSource(
            config=_json_object(args.config),
            benchmark=_json_object(args.benchmark),
            source_reference=args.source_reference,
        )
        return adapt_config(source, overrides)
    if args.format == "dynamo":
        if not args.deploy:
            raise ValueError("--format dynamo requires --deploy")
        return adapt_config(
            DynamoRecipeSource(
                deployment=Path(args.deploy),
                performance=Path(args.perf) if args.perf else None,
                source_reference=args.source_reference,
            ),
            overrides,
        )
    if not args.request:
        raise ValueError("--format request requires --request")
    request = EstimateRequestV1.model_validate_json(json.dumps(_json_object(args.request)))
    return AdaptationReport(outcomes=(AdaptationOutcome(point_id="request-0", status="adapted", request=request),))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("inferencex", "dynamo", "request"), required=True)
    parser.add_argument("--config", help="InferenceX configs.json record or inline JSON")
    parser.add_argument("--benchmark", help="InferenceX benchmark row or inline JSON")
    parser.add_argument("--deploy", help="Dynamo deployment YAML path")
    parser.add_argument("--perf", help="Optional Dynamo performance YAML path")
    parser.add_argument("--request", help="Canonical request JSON path or inline JSON")
    parser.add_argument("--overrides", help="Adapter overrides JSON path or inline JSON")
    parser.add_argument("--source-reference")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run-estimate", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = _report(args)
    schema = _schema()
    for request in report.requests:
        _validate_request(request, schema)

    payload = report.model_dump(mode="json")
    if args.run_estimate:
        payload["estimates"] = [
            _serialize_estimate(cli_estimate(**to_cli_estimate_kwargs(request))) for request in report.requests
        ]
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(f"{rendered}\n")
    print(rendered)
    return 1 if report.rejections else 0


if __name__ == "__main__":
    raise SystemExit(main())
