# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from aiconfigurator_core.sdk.expert_popularity import (
    METADATA_FILENAME,
    PARQUET_FILENAME,
    PARQUET_SCHEMA,
    PUBLIC_PROVENANCE_FIELDS,
    ExpertPopularityDataError,
    list_expert_popularity_models,
    load_expert_popularity,
    load_expert_popularity_metadata,
    model_id_to_bundle_name,
    validate_expert_popularity_bundle,
)

pytestmark = pytest.mark.unit


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_bundle(root: Path) -> Path:
    bundle = root / "example--Tiny-MoE"
    bundle.mkdir()
    counts = [4, 3, 1]
    routed_tokens = 4
    top_k = 2
    table = pa.Table.from_pydict(
        {
            "layer_id": [1, 1, 1],
            "expert_id": [0, 1, 2],
            "activation_count": counts,
            "routed_token_count": [routed_tokens] * 3,
            "token_hit_rate": [value / routed_tokens for value in counts],
            "assignment_share": [value / (routed_tokens * top_k) for value in counts],
            "popularity_rank": [1, 2, 3],
        },
        schema=PARQUET_SCHEMA,
    )
    parquet_path = bundle / PARQUET_FILENAME
    pq.write_table(table, parquet_path)
    metadata = {
        "schema_version": 1,
        "model": {"id": "example/Tiny-MoE", "aliases": ["example/Tiny-MoE-BF16"]},
        "routing": {"moe_layer_ids": [1], "num_routed_experts": 3, "top_k": top_k},
        "provenance": {"framework": "sglang", "framework_version": "0.5.14"},
        "files": {
            "parquet": {
                "name": PARQUET_FILENAME,
                "sha256": _sha256(parquet_path),
                "rows": 3,
            }
        },
    }
    (bundle / METADATA_FILENAME).write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")
    return bundle


def test_valid_bundle_loads_by_canonical_id_and_alias(tmp_path: Path):
    bundle = _write_bundle(tmp_path)

    assert validate_expert_popularity_bundle(bundle)["model"]["id"] == "example/Tiny-MoE"
    assert list_expert_popularity_models(tmp_path) == ["example/Tiny-MoE"]
    assert load_expert_popularity_metadata("example/Tiny-MoE-BF16", tmp_path)["model"]["id"] == ("example/Tiny-MoE")
    table = load_expert_popularity("example/Tiny-MoE", tmp_path)
    assert table["activation_count"].tolist() == [4, 3, 1]
    assert table["assignment_share"].sum() == pytest.approx(1.0)


def test_bundle_rejects_digest_mismatch(tmp_path: Path):
    bundle = _write_bundle(tmp_path)
    metadata_path = bundle / METADATA_FILENAME
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    metadata["files"]["parquet"]["sha256"] = "0" * 64
    metadata_path.write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")

    with pytest.raises(ExpertPopularityDataError, match="SHA-256"):
        validate_expert_popularity_bundle(bundle)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("slurm_node", "private-node"),
        ("gpu_info", ["GPU-01234567-89ab-cdef-0123-456789abcdef"]),
        ("server_args", ["--cache", "/home/private/cache"]),
    ],
)
def test_bundle_rejects_private_provenance(tmp_path: Path, field: str, value: object):
    bundle = _write_bundle(tmp_path)
    metadata_path = bundle / METADATA_FILENAME
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    metadata["provenance"][field] = value
    metadata_path.write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")

    with pytest.raises(ExpertPopularityDataError, match="non-public fields"):
        validate_expert_popularity_bundle(bundle)


def test_bundle_rejects_private_value_in_public_provenance(tmp_path: Path):
    bundle = _write_bundle(tmp_path)
    metadata_path = bundle / METADATA_FILENAME
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    metadata["provenance"]["image_reference"] = "/home/private/image.sqsh"
    metadata_path.write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")

    with pytest.raises(ExpertPopularityDataError, match="private infrastructure identity"):
        validate_expert_popularity_bundle(bundle)


def test_model_id_requires_org_and_name():
    with pytest.raises(ValueError, match="<org>/<model>"):
        model_id_to_bundle_name("not-a-model-id")


@pytest.mark.parametrize(
    ("model_id", "revision", "num_layers", "num_experts", "top_k"),
    [
        (
            "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
            "e434a23f91ba5b4923cf6c9d9a238eb4a08e3a11",
            26,
            64,
            6,
        ),
        (
            "deepseek-ai/DeepSeek-R1",
            "56d4cbbb4d29f4355bab4b9a39ccb717a14ad5ad",
            58,
            256,
            8,
        ),
        (
            "deepseek-ai/DeepSeek-V3",
            "e815299b0bcbac849fa540c768ef21845365c9eb",
            58,
            256,
            8,
        ),
        (
            "deepseek-ai/DeepSeek-V3.1",
            "c0781d039fb7a1ba2abc4add0bdc293e92d2b8db",
            58,
            256,
            8,
        ),
        (
            "deepseek-ai/DeepSeek-V3.2",
            "a7e62ac04ecb2c0a54d736dc46601c5606cf10a6",
            58,
            256,
            8,
        ),
        (
            "deepseek-ai/DeepSeek-V4-Flash",
            "60d8d70770c6776ff598c94bb586a859a38244f1",
            43,
            256,
            6,
        ),
        (
            "deepseek-ai/DeepSeek-V4-Pro",
            "b5968e9190ef611bbf34a7229255be88a0e937c1",
            61,
            384,
            6,
        ),
        (
            "MiniMaxAI/MiniMax-M2.7",
            "d494266a4affc0d2995ba1fa35c8481cbd84294b",
            62,
            256,
            8,
        ),
        (
            "moonshotai/Kimi-K2.5",
            "4d01dfe0332d63057c186e0b262165819efb6611",
            60,
            384,
            8,
        ),
        (
            "moonshotai/Kimi-K2.7-Code",
            "74797c9c62378b951a1f6fcf5c4631024e9b8bef",
            60,
            384,
            8,
        ),
        (
            "zai-org/GLM-5.2",
            "b4734de4facf877f85769a911abafc5283eab3d9",
            75,
            256,
            8,
        ),
        (
            "Qwen/Qwen1.5-MoE-A2.7B",
            "1a758c50ecb6350748b9ce0a99d2352fd9fc11c9",
            24,
            60,
            4,
        ),
        (
            "Qwen/Qwen3-30B-A3B",
            "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
            48,
            128,
            8,
        ),
        (
            "Qwen/Qwen3-235B-A22B",
            "8efa61729e24bd65b1d152b5ab5409052aa80e65",
            94,
            128,
            8,
        ),
        (
            "openai/gpt-oss-20b",
            "6cee5e81ee83917806bbde320786a8fb61efebee",
            24,
            32,
            4,
        ),
    ],
)
def test_packaged_bundle_is_loadable(model_id: str, revision: str, num_layers: int, num_experts: int, top_k: int):
    assert model_id in list_expert_popularity_models()
    table = load_expert_popularity(model_id)
    metadata = load_expert_popularity_metadata(model_id)

    assert len(table) == num_layers * num_experts
    assert metadata["model"]["revision"] == revision
    expected_assignments = metadata["measurement"]["workload"]["total_prompt_tokens"] * top_k
    assert table.groupby("layer_id")["activation_count"].sum().eq(expected_assignments).all()


def test_packaged_bundles_do_not_publish_cluster_identity():
    for model_id in list_expert_popularity_models():
        metadata = load_expert_popularity_metadata(model_id)
        provenance = metadata["provenance"]
        serialized = yaml.safe_dump(metadata)
        assert set(provenance) <= PUBLIC_PROVENANCE_FIELDS
        assert "slurm_job_id" not in provenance
        assert "slurm_node" not in provenance
        assert "gpu_info" not in provenance
        assert "server_args" not in provenance
        assert "runtime_environment" not in provenance
        assert "GPU-" not in serialized
        assert "/home/" not in serialized
        assert "/scratch/" not in serialized
