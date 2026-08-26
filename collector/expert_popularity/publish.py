# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert a passing collection result into the canonical two-file bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from aiconfigurator_core.sdk.expert_popularity import (
    METADATA_FILENAME,
    PARQUET_FILENAME,
    PARQUET_SCHEMA,
    PUBLIC_PROVENANCE_FIELDS,
    SCHEMA_VERSION,
    model_id_to_bundle_name,
    validate_expert_popularity_bundle,
)

PRODUCTION_SEEDS = (2026082501, 2026082502, 2026082503, 2026082504)
PRODUCTION_TOKENS_PER_SHARD = 65536
PRODUCTION_ISL_MIN = 128
PRODUCTION_ISL_MAX = 4096
MIN_PRODUCTION_SHARD_PEARSON = 0.95
MAX_PRODUCTION_SHARD_JSD = 0.01
MIN_PRODUCTION_REPEAT_PEARSON = 0.999
MAX_PRODUCTION_REPEAT_JSD = 0.001
_METADATA_HEADER = """# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return value


def _make_table(counts_data: dict) -> pa.Table:
    layer_ids = [int(value) for value in counts_data["moe_layer_ids"]]
    num_experts = int(counts_data["num_experts"])
    top_k = int(counts_data["top_k"])
    routed_tokens = int(counts_data["routed_token_count"])
    rows = {name: [] for name in PARQUET_SCHEMA.names}
    for layer_id in layer_ids:
        counts = np.asarray(counts_data["counts"][str(layer_id)], dtype=np.uint64)
        if counts.shape != (num_experts,):
            raise ValueError(f"layer {layer_id}: expected {num_experts} expert counts, got {counts.shape}")
        expected_assignments = routed_tokens * top_k
        if int(counts.sum(dtype=np.uint64)) != expected_assignments:
            raise ValueError(f"layer {layer_id}: assignment conservation failed")
        expert_ids = np.arange(num_experts, dtype=np.int32)
        order = np.lexsort((expert_ids, -counts.astype(np.int64)))
        ranks = np.empty(num_experts, dtype=np.int32)
        ranks[order] = np.arange(1, num_experts + 1, dtype=np.int32)
        for expert_id in range(num_experts):
            count = int(counts[expert_id])
            rows["layer_id"].append(layer_id)
            rows["expert_id"].append(expert_id)
            rows["activation_count"].append(count)
            rows["routed_token_count"].append(routed_tokens)
            rows["token_hit_rate"].append(count / routed_tokens)
            rows["assignment_share"].append(count / expected_assignments)
            rows["popularity_rank"].append(int(ranks[expert_id]))
    return pa.Table.from_pydict(rows, schema=PARQUET_SCHEMA)


def _public_collection_provenance(collection: dict) -> dict:
    """Retain reproducibility facts without publishing cluster identity."""
    return {name: collection[name] for name in PUBLIC_PROVENANCE_FIELDS if name in collection}


def _validate_production_collection(result: dict, counts: dict) -> None:
    workload = result.get("workload")
    gates = result.get("validation_gates")
    stability = result.get("stability")
    repeat_stability = result.get("repeat_stability")
    if not all(isinstance(value, dict) for value in (workload, gates, stability, repeat_stability)):
        raise ValueError("collection is missing production workload or stability evidence")

    expected_workload = {
        "phase": "prefill",
        "token_distribution": "uniform_valid_non_special_token_ids",
        "isl_distribution": "discrete_uniform",
        "isl_min": PRODUCTION_ISL_MIN,
        "isl_max": PRODUCTION_ISL_MAX,
        "repeat_count": 2,
        "max_new_tokens": 1,
        "temperature": 0,
        "sequential": True,
        "shard_count": len(PRODUCTION_SEEDS),
    }
    mismatches = {
        name: {"expected": expected, "actual": workload.get(name)}
        for name, expected in expected_workload.items()
        if workload.get(name) != expected
    }
    if mismatches:
        raise ValueError(f"collection does not use the canonical production workload: {mismatches}")
    if tuple(workload.get("seeds", ())) != PRODUCTION_SEEDS:
        raise ValueError("collection does not use the canonical production seeds")
    if int(workload.get("tokens_per_shard_minimum", 0)) < PRODUCTION_TOKENS_PER_SHARD:
        raise ValueError("collection has fewer than 65,536 prompt tokens per shard")
    total_prompt_tokens = int(workload.get("total_prompt_tokens", 0))
    if total_prompt_tokens < len(PRODUCTION_SEEDS) * PRODUCTION_TOKENS_PER_SHARD:
        raise ValueError("collection has fewer than the required total prompt tokens")
    if int(counts.get("routed_token_count", -1)) != total_prompt_tokens:
        raise ValueError("normalized counts do not cover the declared production workload")

    if float(gates.get("minimum_mean_layer_pearson", -1.0)) < MIN_PRODUCTION_SHARD_PEARSON:
        raise ValueError("collection used a cross-shard Pearson gate weaker than production")
    if float(gates.get("maximum_mean_layer_jensen_shannon_divergence_bits", 1.0)) > MAX_PRODUCTION_SHARD_JSD:
        raise ValueError("collection used a cross-shard JSD gate weaker than production")
    if float(stability.get("minimum_mean_layer_pearson", -1.0)) < MIN_PRODUCTION_SHARD_PEARSON:
        raise ValueError("collection failed the production cross-shard Pearson threshold")
    if float(stability.get("maximum_mean_layer_jensen_shannon_divergence_bits", 1.0)) > MAX_PRODUCTION_SHARD_JSD:
        raise ValueError("collection failed the production cross-shard JSD threshold")
    if gates.get("layer_assignment_conservation") is not True:
        raise ValueError("collection lacks the layer assignment conservation gate")

    repeat_mode = gates.get("repeat_validation_mode", "exact")
    if repeat_mode == "exact":
        if gates.get("repeat_counts_exact") is not True or repeat_stability.get("all_counts_exact") is not True:
            raise ValueError("collection failed exact repeat validation")
    elif repeat_mode == "aggregate":
        if float(gates.get("minimum_repeat_mean_layer_pearson", -1.0)) < MIN_PRODUCTION_REPEAT_PEARSON:
            raise ValueError("collection used a repeat Pearson gate weaker than production")
        if (
            float(gates.get("maximum_repeat_mean_layer_jensen_shannon_divergence_bits", 1.0))
            > MAX_PRODUCTION_REPEAT_JSD
        ):
            raise ValueError("collection used a repeat JSD gate weaker than production")
        if float(repeat_stability.get("minimum_mean_layer_pearson", -1.0)) < MIN_PRODUCTION_REPEAT_PEARSON:
            raise ValueError("collection failed the production repeat Pearson threshold")
        if (
            float(repeat_stability.get("maximum_mean_layer_jensen_shannon_divergence_bits", 1.0))
            > MAX_PRODUCTION_REPEAT_JSD
        ):
            raise ValueError("collection failed the production repeat JSD threshold")
    else:
        raise ValueError(f"unsupported repeat validation mode: {repeat_mode!r}")


def build_bundle(collection_dir: Path, output_root: Path) -> Path:
    result_path = collection_dir / "collection_result.json"
    counts_path = collection_dir / "normalized_counts.json"
    result = _load_json(result_path)
    counts = _load_json(counts_path)
    if result.get("schema_version") != SCHEMA_VERSION or result.get("status") != "PASS":
        raise ValueError(f"{result_path}: only passing schema-v{SCHEMA_VERSION} collections may be published")
    if counts.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{counts_path}: unsupported schema version")
    if result["model"]["id"] != counts["model_id"] or result["model"]["revision"] != counts["model_revision"]:
        raise ValueError("collection result and normalized counts have different model identities")
    _validate_production_collection(result, counts)

    bundle = output_root / model_id_to_bundle_name(result["model"]["id"])
    bundle.mkdir(parents=True, exist_ok=True)
    unexpected = [path for path in bundle.iterdir() if path.name not in {PARQUET_FILENAME, METADATA_FILENAME}]
    if unexpected:
        raise ValueError(f"refusing to publish into non-canonical bundle directory: {unexpected}")
    parquet_path = bundle / PARQUET_FILENAME
    table = _make_table(counts)
    pq.write_table(table, parquet_path, compression="zstd", version="2.6")

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "model": {**result["model"], "aliases": []},
        "routing": result["routing"],
        "measurement": {
            "phase": result["workload"]["phase"],
            "count_semantics": (
                "activation_count[layer, expert] is the number of canonical prompt tokens "
                "whose serving router selected that logical routed expert"
            ),
            "token_hit_rate_denominator": "routed_token_count",
            "assignment_share_denominator": "routed_token_count * top_k",
            "popularity_rank_order": "activation_count descending, then expert_id ascending",
            "workload": result["workload"],
        },
        "validation": {
            "status": "PASS",
            "gates": result["validation_gates"],
            "stability": result["stability"],
            "repeat_stability": result.get("repeat_stability"),
        },
        "provenance": {
            **_public_collection_provenance(result["collection"]),
            "collection_result_sha256": _sha256(result_path),
            "normalized_counts_sha256": _sha256(counts_path),
            "published_at": datetime.now(UTC).isoformat(),
        },
        "files": {
            "parquet": {
                "name": PARQUET_FILENAME,
                "sha256": _sha256(parquet_path),
                "rows": table.num_rows,
            }
        },
    }
    metadata_path = bundle / METADATA_FILENAME
    with metadata_path.open("w", encoding="utf-8") as handle:
        handle.write(_METADATA_HEADER)
        yaml.safe_dump(metadata, handle, sort_keys=False, allow_unicode=True)
    validate_expert_popularity_bundle(bundle)
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(build_bundle(args.collection_dir, args.output_root))


if __name__ == "__main__":
    main()
