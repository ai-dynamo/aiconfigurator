# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Versioned ``collection_meta.yaml`` parser contract."""

from pathlib import Path

import pytest
import yaml

from aiconfigurator.sdk.perf_database import _load_collection_meta_yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]


def _write_meta(tmp_path, document: dict):
    path = tmp_path / "collection_meta.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def _event(**overrides):
    event = {
        "collector_ref": "a" * 40,
        "collector_hash": "sha256:" + "b" * 64,
        "case_plan_hash": "sha256:" + "c" * 64,
        "collected_at": "2026-08-26",
        "rows": 3,
        "status": "complete",
    }
    event.update(overrides)
    return event


def _v2_document(collections: list[dict]) -> dict:
    return {
        "schema_version": 2,
        "runtime": {"framework": "sglang", "version": "0.5.14"},
        "tables": {
            "gemm_perf": {
                "rows": 45,
                "status": "complete",
                "collections": collections,
            }
        },
    }


def test_v2_rejects_collection_event_without_status(tmp_path):
    event = _event()
    del event["status"]
    path = _write_meta(tmp_path, _v2_document([event]))

    with pytest.raises(ValueError, match=r"gemm_perf\.collections\[0\].*status"):
        _load_collection_meta_yaml(str(path))


def test_v2_accepts_complete_multi_event_history(tmp_path):
    document = _v2_document([_event(), _event(case_plan_hash="sha256:" + "d" * 64, rows=15)])
    path = _write_meta(tmp_path, document)

    assert _load_collection_meta_yaml(str(path)) == document


def test_v2_rejects_blank_collection_event_plan(tmp_path):
    path = _write_meta(tmp_path, _v2_document([_event(case_plan_hash="")]))

    with pytest.raises(ValueError, match=r"collections\[0\]\.case_plan_hash must be a non-empty string"):
        _load_collection_meta_yaml(str(path))


def test_v1_remains_backward_compatible(tmp_path):
    document = {
        "schema_version": 1,
        "runtime": {"framework": "sglang", "version": "0.5.14"},
        "tables": {"legacy_perf": {"status": "complete"}},
    }
    path = _write_meta(tmp_path, document)

    assert _load_collection_meta_yaml(str(path)) == document


def test_v1_rejects_unversioned_collection_histories(tmp_path):
    document = _v2_document([_event()])
    document["schema_version"] = 1
    path = _write_meta(tmp_path, document)

    with pytest.raises(ValueError, match="collections requires schema_version 2"):
        _load_collection_meta_yaml(str(path))


def test_rejects_unknown_schema_version(tmp_path):
    path = _write_meta(tmp_path, {"schema_version": 3})

    with pytest.raises(ValueError, match=r"unsupported collection_meta\.yaml schema_version 3"):
        _load_collection_meta_yaml(str(path))


def test_v2_event_runtime_rejects_unknown_fields(tmp_path):
    event = _event(runtime={"framework": "sglang", "version": "0.5.14", "upstream_image": "invented"})
    path = _write_meta(tmp_path, _v2_document([event]))

    with pytest.raises(ValueError, match=r"collections\[0\]\.runtime has unsupported key.*upstream_image"):
        _load_collection_meta_yaml(str(path))


def test_b300_gdn_records_mixed_runtime_history_honestly():
    path = (
        REPO_ROOT
        / "aic-core/src/aiconfigurator_core/systems/data/b300_sxm/linear_attention/sglang/0.5.14/collection_meta.yaml"
    )

    document = _load_collection_meta_yaml(str(path))
    runtime = document["runtime"]
    events = document["tables"]["gdn_perf"]["collections"]

    assert runtime["image"] == "gitlab-master.nvidia.com/yimingl/aic-eval-tools/sglang:v0.5.14-cu130-amd64"
    assert runtime["image_digest"] == "sha256:9611bd4c5624b0e9e17829506188a12f17205f2083de0dd44d6c521733553a50"
    assert events[0]["runtime"]["image"] == "lmsysorg/sglang:v0.5.14"
    assert events[0]["runtime"]["image_digest"] == (
        "sha256:5027e95bf6ec536856b1b52a91d1f35ff5c564ab83e8a94758a169ff09bb8df3"
    )
    assert "runtime" not in events[1]  # inherits the exact private JET runtime above
