# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load and validate model-specific routed-expert popularity bundles."""

from __future__ import annotations

import hashlib
import importlib.resources as pkg_resources
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

PARQUET_FILENAME = "model_expert_popularity.parquet"
METADATA_FILENAME = "metadata.yaml"
SCHEMA_VERSION = 1
PUBLIC_PROVENANCE_FIELDS = frozenset(
    {
        "collection_checkpoint",
        "collection_result_sha256",
        "collector_code_sha256",
        "count_replication_factor",
        "framework",
        "framework_version",
        "image_reference",
        "image_sha256",
        "normalized_counts_sha256",
        "observation_source",
        "published_at",
        "routing_equivalence_evidence",
        "routing_observation_method",
    }
)
_PRIVATE_METADATA_PATTERNS = (
    re.compile(r"/(?:home|Users|scratch|fsx|lustre)/"),
    re.compile(r"\bGPU-[0-9a-fA-F-]{16,}\b"),
    re.compile(r"\b10(?:[.][0-9]{1,3}){3}\b"),
    re.compile(r"\b192[.]168(?:[.][0-9]{1,3}){2}\b"),
    re.compile(r"\b172[.](?:1[6-9]|2[0-9]|3[01])(?:[.][0-9]{1,3}){2}\b"),
)
PARQUET_SCHEMA = pa.schema(
    [
        pa.field("layer_id", pa.int32(), nullable=False),
        pa.field("expert_id", pa.int32(), nullable=False),
        pa.field("activation_count", pa.uint64(), nullable=False),
        pa.field("routed_token_count", pa.uint64(), nullable=False),
        pa.field("token_hit_rate", pa.float64(), nullable=False),
        pa.field("assignment_share", pa.float64(), nullable=False),
        pa.field("popularity_rank", pa.int32(), nullable=False),
    ]
)


class ExpertPopularityDataError(ValueError):
    """Raised when an expert-popularity bundle violates its data contract."""


def model_id_to_bundle_name(model_id: str) -> str:
    """Return the stable on-disk name for a Hugging Face-style model ID."""
    parts = model_id.split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"model_id must have the form '<org>/<model>', got {model_id!r}")
    return "--".join(parts)


def _default_data_root():
    return pkg_resources.files("aiconfigurator_core") / "model_data" / "expert_popularity"


def _sha256(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ExpertPopularityDataError(f"{path}: metadata must be a YAML mapping")
    return value


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExpertPopularityDataError(f"metadata.{name} must be a mapping")
    return value


def _validate_public_metadata(metadata: Mapping[str, Any]) -> None:
    provenance = _require_mapping(metadata.get("provenance"), "provenance")
    unexpected = sorted(set(provenance) - PUBLIC_PROVENANCE_FIELDS)
    if unexpected:
        raise ExpertPopularityDataError(f"metadata.provenance contains non-public fields: {unexpected}")

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                visit(nested, f"{path}.{key}")
        elif isinstance(value, list):
            for index, nested in enumerate(value):
                visit(nested, f"{path}[{index}]")
        elif isinstance(value, str):
            for pattern in _PRIVATE_METADATA_PATTERNS:
                if pattern.search(value):
                    raise ExpertPopularityDataError(f"{path} contains private infrastructure identity")

    visit(metadata, "metadata")


def _validate_table(table: pa.Table, metadata: Mapping[str, Any], source: Path) -> None:
    if table.schema != PARQUET_SCHEMA:
        raise ExpertPopularityDataError(
            f"{source}: unexpected parquet schema\nexpected: {PARQUET_SCHEMA}\nactual: {table.schema}"
        )
    if table.num_rows == 0:
        raise ExpertPopularityDataError(f"{source}: table must not be empty")

    routing = _require_mapping(metadata.get("routing"), "routing")
    layer_ids = [int(value) for value in routing.get("moe_layer_ids", [])]
    num_experts = int(routing.get("num_routed_experts", 0))
    top_k = int(routing.get("top_k", 0))
    if not layer_ids or num_experts <= 0 or not 0 < top_k <= num_experts:
        raise ExpertPopularityDataError("metadata.routing has invalid MoE dimensions")
    if table.num_rows != len(layer_ids) * num_experts:
        raise ExpertPopularityDataError(f"{source}: expected {len(layer_ids) * num_experts} rows, got {table.num_rows}")

    columns = {name: table[name].to_numpy(zero_copy_only=False) for name in table.column_names}
    expected_pairs = {(layer_id, expert_id) for layer_id in layer_ids for expert_id in range(num_experts)}
    actual_pairs = set(zip(columns["layer_id"].tolist(), columns["expert_id"].tolist(), strict=True))
    if actual_pairs != expected_pairs:
        raise ExpertPopularityDataError(f"{source}: layer/expert keys do not form the declared dense grid")

    for layer_id in layer_ids:
        mask = columns["layer_id"] == layer_id
        counts = columns["activation_count"][mask]
        routed_tokens = columns["routed_token_count"][mask]
        expert_ids = columns["expert_id"][mask]
        if len(set(routed_tokens.tolist())) != 1 or int(routed_tokens[0]) <= 0:
            raise ExpertPopularityDataError(f"{source}: layer {layer_id} has inconsistent routed_token_count")
        expected_assignments = int(routed_tokens[0]) * top_k
        if int(counts.sum(dtype=np.uint64)) != expected_assignments:
            raise ExpertPopularityDataError(
                f"{source}: layer {layer_id} activation conservation failed: "
                f"{int(counts.sum(dtype=np.uint64))} != {expected_assignments}"
            )
        expected_hit_rate = counts.astype(np.float64) / float(routed_tokens[0])
        expected_share = counts.astype(np.float64) / float(expected_assignments)
        if not np.allclose(columns["token_hit_rate"][mask], expected_hit_rate, rtol=0.0, atol=1e-15):
            raise ExpertPopularityDataError(f"{source}: layer {layer_id} token_hit_rate is inconsistent")
        if not np.allclose(columns["assignment_share"][mask], expected_share, rtol=0.0, atol=1e-15):
            raise ExpertPopularityDataError(f"{source}: layer {layer_id} assignment_share is inconsistent")
        order = np.lexsort((expert_ids, -counts.astype(np.int64)))
        expected_ranks = np.empty(num_experts, dtype=np.int32)
        expected_ranks[order] = np.arange(1, num_experts + 1, dtype=np.int32)
        if not np.array_equal(columns["popularity_rank"][mask], expected_ranks):
            raise ExpertPopularityDataError(f"{source}: layer {layer_id} popularity_rank is inconsistent")


def validate_expert_popularity_bundle(bundle_path: os.PathLike[str] | str) -> dict[str, Any]:
    """Validate one two-file bundle and return its parsed metadata."""
    bundle = Path(bundle_path)
    metadata_path = bundle / METADATA_FILENAME
    parquet_path = bundle / PARQUET_FILENAME
    if not metadata_path.is_file() or not parquet_path.is_file():
        raise ExpertPopularityDataError(
            f"{bundle}: bundle must contain exactly {PARQUET_FILENAME!r} and {METADATA_FILENAME!r}"
        )
    unexpected = sorted(
        path.name for path in bundle.iterdir() if path.name not in {METADATA_FILENAME, PARQUET_FILENAME}
    )
    if unexpected:
        raise ExpertPopularityDataError(f"{bundle}: unexpected bundle files: {unexpected}")

    metadata = _load_yaml(metadata_path)
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ExpertPopularityDataError(
            f"{metadata_path}: expected schema_version {SCHEMA_VERSION}, got {metadata.get('schema_version')!r}"
        )
    model = _require_mapping(metadata.get("model"), "model")
    _validate_public_metadata(metadata)
    model_id = model.get("id")
    if not isinstance(model_id, str) or bundle.name != model_id_to_bundle_name(model_id):
        raise ExpertPopularityDataError(f"{bundle}: directory name does not match metadata.model.id")
    files = _require_mapping(metadata.get("files"), "files")
    parquet_meta = _require_mapping(files.get("parquet"), "files.parquet")
    if parquet_meta.get("name") != PARQUET_FILENAME:
        raise ExpertPopularityDataError("metadata.files.parquet.name is not canonical")
    actual_digest = _sha256(parquet_path)
    if parquet_meta.get("sha256") != actual_digest:
        raise ExpertPopularityDataError(f"{parquet_path}: SHA-256 does not match metadata")

    table = pq.read_table(parquet_path)
    if parquet_meta.get("rows") != table.num_rows:
        raise ExpertPopularityDataError(f"{parquet_path}: row count does not match metadata")
    _validate_table(table, metadata, parquet_path)
    return metadata


def _resolve_bundle(model_id: str, data_root: os.PathLike[str] | str | None):
    root = Path(data_root) if data_root is not None else _default_data_root()
    direct = root / model_id_to_bundle_name(model_id)
    if direct.is_dir():
        return direct
    if root.is_dir():
        for candidate in sorted(root.iterdir(), key=lambda path: path.name):
            metadata_path = candidate / METADATA_FILENAME
            if not metadata_path.is_file():
                continue
            metadata = _load_yaml(metadata_path)
            model = metadata.get("model") or {}
            if model_id in model.get("aliases", []):
                return candidate
    raise FileNotFoundError(f"No expert-popularity bundle found for {model_id!r} under {root}")


def list_expert_popularity_models(data_root: os.PathLike[str] | str | None = None) -> list[str]:
    """List canonical model IDs with packaged expert-popularity data."""
    root = Path(data_root) if data_root is not None else _default_data_root()
    if not root.is_dir():
        return []
    result = []
    for candidate in sorted(root.iterdir(), key=lambda path: path.name):
        metadata_path = candidate / METADATA_FILENAME
        if metadata_path.is_file():
            model_id = (_load_yaml(metadata_path).get("model") or {}).get("id")
            if isinstance(model_id, str):
                result.append(model_id)
    return result


def load_expert_popularity_metadata(model_id: str, data_root: os.PathLike[str] | str | None = None) -> dict[str, Any]:
    """Load validated provenance and collection metadata for ``model_id``."""
    return validate_expert_popularity_bundle(_resolve_bundle(model_id, data_root))


def load_expert_popularity(model_id: str, data_root: os.PathLike[str] | str | None = None):
    """Load a validated model-specific expert-popularity table as a DataFrame."""
    bundle = _resolve_bundle(model_id, data_root)
    validate_expert_popularity_bundle(bundle)
    return pq.read_table(bundle / PARQUET_FILENAME).to_pandas()
