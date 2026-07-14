# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io

import pytest

from aiconfigurator.sdk import s3_models


class _FakeS3:
    def __init__(self):
        self.pages = [
            {
                "Contents": [
                    {"Key": "ai-lab/qwen3-0.6b/2026-06-01/config.json"},
                    {"Key": "ai-lab/qwen3-0.6b/2026-07-01/config.json"},
                    {"Key": "ai-lab/qwen3-0.6b/README.md"},
                    {"Key": "invalid/config.json"},
                ],
                "NextContinuationToken": "next",
            },
            {"Contents": [{"Key": "other/model/2026-05-01/config.json"}]},
        ]

    def list_objects_v2(self, **kwargs):
        return self.pages[1] if kwargs.get("ContinuationToken") else self.pages[0]

    def get_object(self, **kwargs):
        assert kwargs["Bucket"] == "aiplat"
        assert kwargs["Key"] == "ai-lab/qwen3-0.6b/2026-07-01/config.json"
        return {"Body": io.BytesIO(b'{"architectures":["Qwen3ForCausalLM"]}')}


@pytest.fixture(autouse=True)
def _clear_s3_caches():
    s3_models.get_s3_client.cache_clear()
    s3_models.load_s3_json.cache_clear()


def test_parse_s3_model_uri_requires_versioned_layout():
    location = s3_models.parse_s3_model_uri("s3://aiplat/ai-lab/qwen3-0.6b/2026-07-01")
    assert location.namespace == "ai-lab"
    assert location.model_name == "qwen3-0.6b"
    assert location.model_version == "2026-07-01"

    with pytest.raises(s3_models.S3ModelError, match="expected s3://bucket"):
        s3_models.parse_s3_model_uri("s3://aiplat/ai-lab/qwen3-0.6b")


def test_list_s3_models_discovers_config_json_and_sorts_latest_first(monkeypatch):
    monkeypatch.setattr(s3_models, "get_s3_client", lambda: _FakeS3())

    models = s3_models.list_s3_models("aiplat")

    assert [item.uri for item in models] == [
        "s3://aiplat/ai-lab/qwen3-0.6b/2026-07-01",
        "s3://aiplat/ai-lab/qwen3-0.6b/2026-06-01",
        "s3://aiplat/other/model/2026-05-01",
    ]


def test_load_s3_json_reads_model_metadata(monkeypatch):
    monkeypatch.setattr(s3_models, "get_s3_client", lambda: _FakeS3())
    assert s3_models.load_s3_json("s3://aiplat/ai-lab/qwen3-0.6b/2026-07-01", "config.json") == {
        "architectures": ["Qwen3ForCausalLM"]
    }
