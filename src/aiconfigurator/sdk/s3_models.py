# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discover and read Hugging Face model layouts stored in S3-compatible OSS."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import cache
from typing import Any
from urllib.parse import urlparse


class S3ModelError(ValueError):
    """Raised when an OSS model cannot be discovered or read."""


@dataclass(frozen=True)
class S3ModelLocation:
    bucket: str
    namespace: str
    model_name: str
    model_version: str

    @property
    def prefix(self) -> str:
        return f"{self.namespace}/{self.model_name}/{self.model_version}"

    @property
    def uri(self) -> str:
        return f"s3://{self.bucket}/{self.prefix}"


def _boto3():
    try:
        import boto3
    except ModuleNotFoundError as exc:
        raise S3ModelError(
            "Reading models from S3-compatible OSS requires boto3. Install aiconfigurator with the service extra."
        ) from exc
    return boto3


@cache
def get_s3_client():
    """Build an S3 client from the standard AWS environment contract."""
    from botocore.config import Config

    endpoint = (
        os.environ.get("AWS_ENDPOINT_URL_S3") or os.environ.get("AWS_ENDPOINT_URL") or "https://oss-s3.haiercash.com"
    )
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "cn-east-1"
    return _boto3().client(
        "s3",
        endpoint_url=endpoint,
        region_name=region,
        config=Config(s3={"addressing_style": "path"}),
    )


def parse_s3_model_uri(uri: str) -> S3ModelLocation:
    """Parse ``s3://bucket/namespace/model/version`` into its model coordinates."""
    parsed = urlparse(uri)
    parts = [part for part in parsed.path.split("/") if part]
    if parsed.scheme != "s3" or not parsed.netloc or len(parts) != 3:
        raise S3ModelError(f"Invalid OSS model URI {uri!r}; expected s3://bucket/namespace/model_name/model_version")
    return S3ModelLocation(parsed.netloc, parts[0], parts[1], parts[2])


def list_s3_models(bucket: str | None = None) -> list[S3ModelLocation]:
    """List versioned models by finding ``config.json`` objects under a bucket."""
    bucket = (bucket or os.environ.get("AIC_MODEL_S3_BUCKET") or "aiplat").strip()
    if not bucket:
        return []
    request: dict[str, Any] = {"Bucket": bucket}

    models: set[S3ModelLocation] = set()
    client = get_s3_client()
    while True:
        try:
            response = client.list_objects_v2(**request)
        except Exception as exc:
            raise S3ModelError(f"Failed to list models in s3://{bucket}: {exc}") from exc
        for item in response.get("Contents", []):
            key = str(item.get("Key", "")).strip("/")
            if not key.endswith("/config.json"):
                continue
            model_prefix = key[: -len("/config.json")]
            parts = model_prefix.split("/")
            if len(parts) != 3:
                continue
            models.add(S3ModelLocation(bucket, parts[0], parts[1], parts[2]))
        token = response.get("NextContinuationToken")
        if not token:
            break
        request["ContinuationToken"] = token

    by_version = sorted(models, key=lambda item: item.model_version, reverse=True)
    return sorted(by_version, key=lambda item: (item.namespace.lower(), item.model_name.lower()))


@cache
def load_s3_json(model_uri: str, filename: str, *, required: bool = True) -> dict | None:
    """Load one JSON metadata file from a versioned OSS model directory."""
    location = parse_s3_model_uri(model_uri)
    key = f"{location.prefix}/{filename}"
    try:
        response = get_s3_client().get_object(Bucket=location.bucket, Key=key)
        return json.loads(response["Body"].read())
    except Exception as exc:
        if not required:
            return None
        raise S3ModelError(f"Failed to read s3://{location.bucket}/{key}: {exc}") from exc


def get_configured_s3_model_uris() -> list[str]:
    """Return OSS-backed model choices when an OSS bucket is configured."""
    return [model.uri for model in list_s3_models()]


def is_s3_model_source_configured() -> bool:
    return os.environ.get("AIC_MODEL_SOURCE", "s3").strip().lower() == "s3"
