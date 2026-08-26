# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build fail-closed artifact evidence for a repacked collection checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import traceback
from datetime import UTC, datetime
from pathlib import Path

import torch
from safetensors import safe_open

_ROUTING_CONFIG_FIELDS = (
    "architectures",
    "hidden_size",
    "model_type",
    "moe_intermediate_size",
    "n_group",
    "n_routed_experts",
    "n_shared_experts",
    "norm_topk_prob",
    "num_experts",
    "num_experts_per_tok",
    "num_hidden_layers",
    "routed_scaling_factor",
    "scoring_func",
    "topk_group",
    "topk_method",
)
_ROUTER_KEY = re.compile(r"(?:^|\.)(?:ffn|mlp)\.gate\.")
_REQUIRED_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")
_OPTIONAL_TOKENIZER_FILES = ("chat_template.jinja", "special_tokens_map.json", "tokenizer.model")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return value


def _routing_config(config: dict) -> dict:
    text_config = config.get("text_config")
    source = text_config if isinstance(text_config, dict) else config
    return {name: source.get(name, config.get(name)) for name in _ROUTING_CONFIG_FIELDS}


def _weight_index(model_dir: Path) -> tuple[Path, dict]:
    candidates = sorted(model_dir.glob("*.safetensors.index.json"))
    if len(candidates) != 1:
        raise RuntimeError(f"{model_dir}: expected exactly one safetensors index, found {candidates}")
    return candidates[0], _load_json(candidates[0])


def _tensor_digest(model_dir: Path, weight_map: dict[str, str], keys: list[str]) -> tuple[str, dict[str, dict]]:
    digest = hashlib.sha256()
    descriptions: dict[str, dict] = {}
    keys_by_file: dict[str, list[str]] = {}
    for key in keys:
        keys_by_file.setdefault(weight_map[key], []).append(key)
    for filename in sorted(keys_by_file):
        with safe_open(model_dir / filename, framework="pt", device="cpu") as handle:
            for key in sorted(keys_by_file[filename]):
                tensor = handle.get_tensor(key).contiguous()
                payload = tensor.view(torch.uint8).numpy().tobytes()
                description = {
                    "dtype": str(tensor.dtype),
                    "shape": list(tensor.shape),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
                descriptions[key] = description
                digest.update(key.encode())
                digest.update(json.dumps(description, sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest(), descriptions


def _physical_tensor_keys(model_dir: Path) -> set[str]:
    result: set[str] = set()
    for path in sorted(model_dir.glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
        duplicates = result & keys
        if duplicates:
            raise RuntimeError(f"duplicate physical tensor keys across shards: {sorted(duplicates)[:10]}")
        result.update(keys)
    if not result:
        raise RuntimeError(f"{model_dir}: no safetensors shards found")
    return result


def _load_tensor(model_dir: Path, weight_map: dict[str, str], key: str) -> torch.Tensor:
    with safe_open(model_dir / weight_map[key], framework="pt", device="cpu") as handle:
        return handle.get_tensor(key).contiguous()


def _dequantize_fp8_blocks(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    block_size = 128
    if weight.ndim != 2 or scale.ndim != 2:
        raise RuntimeError(f"expected 2-D FP8 weight and scale, got {weight.shape=} {scale.shape=}")
    rows, columns = weight.shape
    if rows % block_size or columns % block_size:
        raise RuntimeError(f"FP8 weight shape is not divisible by {block_size}: {weight.shape}")
    expected_scale_shape = (rows // block_size, columns // block_size)
    if tuple(scale.shape) != expected_scale_shape:
        raise RuntimeError(f"FP8 scale shape {tuple(scale.shape)} != expected {expected_scale_shape}")
    blocked = weight.float().reshape(rows // block_size, block_size, columns // block_size, block_size)
    return (blocked * scale.float()[:, None, :, None]).reshape(rows, columns).to(torch.bfloat16)


def _verify_dequantized_wo_a(
    canonical_model_dir: Path,
    checkpoint_model_dir: Path,
    canonical_map: dict[str, str],
    checkpoint_map: dict[str, str],
    weight_keys: list[str],
) -> str:
    digest = hashlib.sha256()
    mismatches = []
    for key in weight_keys:
        scale_key = key.removesuffix(".weight") + ".scale"
        canonical_weight = _load_tensor(canonical_model_dir, canonical_map, key)
        canonical_scale = _load_tensor(canonical_model_dir, canonical_map, scale_key)
        checkpoint_weight = _load_tensor(checkpoint_model_dir, checkpoint_map, key)
        dequantized = _dequantize_fp8_blocks(canonical_weight, canonical_scale)
        if dequantized.dtype != checkpoint_weight.dtype or not torch.equal(dequantized, checkpoint_weight):
            mismatches.append(key)
            continue
        payload = checkpoint_weight.view(torch.uint8).numpy().tobytes()
        digest.update(key.encode())
        digest.update(hashlib.sha256(payload).digest())
    if mismatches:
        raise RuntimeError(f"dequantized canonical wo_a tensors differ from checkpoint BF16: {mismatches[:10]}")
    return digest.hexdigest()


def _parse_runtime_environment(values: list[str]) -> dict[str, str]:
    result = {}
    for value in values:
        name, separator, setting = value.partition("=")
        if not separator or not name or name in result:
            raise ValueError(f"invalid or duplicate required runtime environment entry: {value!r}")
        result[name] = setting
    return result


def build_report(args: argparse.Namespace) -> dict:
    canonical_config = _routing_config(_load_json(args.canonical_model_dir / "config.json"))
    checkpoint_config = _routing_config(_load_json(args.checkpoint_model_dir / "config.json"))
    if canonical_config != checkpoint_config:
        raise RuntimeError("routing-relevant config fields differ")

    canonical_index_path, canonical_index = _weight_index(args.canonical_model_dir)
    checkpoint_index_path, checkpoint_index = _weight_index(args.checkpoint_model_dir)
    canonical_map = canonical_index.get("weight_map")
    checkpoint_map = checkpoint_index.get("weight_map")
    if not isinstance(canonical_map, dict) or not isinstance(checkpoint_map, dict):
        raise TypeError("safetensors index is missing weight_map")
    if set(canonical_map) != set(checkpoint_map):
        raise RuntimeError("canonical and checkpoint tensor key sets differ")
    if canonical_index.get("metadata", {}).get("total_size") != checkpoint_index.get("metadata", {}).get("total_size"):
        raise RuntimeError("canonical and checkpoint safetensors total_size differ")
    router_keys = sorted(key for key in canonical_map if _ROUTER_KEY.search(key))
    if not router_keys:
        raise RuntimeError("no router tensors found")
    canonical_router_digest, canonical_router_tensors = _tensor_digest(
        args.canonical_model_dir, canonical_map, router_keys
    )
    _, checkpoint_router_tensors = _tensor_digest(args.checkpoint_model_dir, checkpoint_map, router_keys)
    if canonical_router_tensors != checkpoint_router_tensors:
        mismatches = [key for key in router_keys if canonical_router_tensors[key] != checkpoint_router_tensors[key]]
        raise RuntimeError(f"router tensors differ: {mismatches[:10]}")

    canonical_physical_keys = _physical_tensor_keys(args.canonical_model_dir)
    checkpoint_physical_keys = _physical_tensor_keys(args.checkpoint_model_dir)
    extra_checkpoint_keys = sorted(checkpoint_physical_keys - canonical_physical_keys)
    missing_checkpoint_keys = sorted(canonical_physical_keys - checkpoint_physical_keys)
    allowed_missing_patterns = [re.compile(pattern) for pattern in args.allow_missing_checkpoint_key_regex]
    unexpected_missing_keys = [
        key
        for key in missing_checkpoint_keys
        if not any(pattern.fullmatch(key) for pattern in allowed_missing_patterns)
    ]
    if extra_checkpoint_keys or unexpected_missing_keys:
        raise RuntimeError(
            "physical tensor keys differ outside the declared repack contract: "
            f"extra={extra_checkpoint_keys[:10]}, missing={unexpected_missing_keys[:10]}"
        )
    if set(canonical_map) != canonical_physical_keys:
        raise RuntimeError("canonical safetensors index does not exactly describe its physical tensors")
    if set(checkpoint_map) - checkpoint_physical_keys != set(missing_checkpoint_keys):
        raise RuntimeError("checkpoint safetensors index/physical delta differs from the declared repack delta")

    wo_a_weight_keys = sorted(
        key for key in canonical_physical_keys & checkpoint_physical_keys if key.endswith(".attn.wo_a.weight")
    )
    if not wo_a_weight_keys:
        raise RuntimeError("no wo_a tensors found for repack verification")
    dequantized_wo_a_digest = _verify_dequantized_wo_a(
        args.canonical_model_dir,
        args.checkpoint_model_dir,
        canonical_map,
        checkpoint_map,
        wo_a_weight_keys,
    )

    tokenizer_files = {}
    for filename in (*_REQUIRED_TOKENIZER_FILES, *_OPTIONAL_TOKENIZER_FILES):
        canonical_path = args.canonical_model_dir / filename
        checkpoint_path = args.checkpoint_model_dir / filename
        if filename in _REQUIRED_TOKENIZER_FILES and (not canonical_path.is_file() or not checkpoint_path.is_file()):
            raise RuntimeError(f"required tokenizer artifact missing: {filename}")
        if canonical_path.is_file() != checkpoint_path.is_file():
            raise RuntimeError(f"tokenizer artifact presence differs: {filename}")
        if canonical_path.is_file():
            canonical_sha256 = _sha256(canonical_path)
            checkpoint_sha256 = _sha256(checkpoint_path)
            if canonical_sha256 != checkpoint_sha256:
                raise RuntimeError(f"tokenizer artifact differs: {filename}")
            tokenizer_files[filename] = canonical_sha256

    readme = (args.checkpoint_model_dir / "README.md").read_text(encoding="utf-8")
    missing_claims = [claim for claim in args.expected_readme_substring if claim not in readme]
    if missing_claims:
        raise RuntimeError(f"checkpoint README is missing provenance claims: {missing_claims}")

    required_runtime_environment = _parse_runtime_environment(args.required_runtime_environment)
    return {
        "schema_version": 1,
        "status": "PASS",
        "created_at": _utc_now(),
        "canonical_model": {"id": args.canonical_model_id, "revision": args.canonical_model_revision},
        "collection_checkpoint": {
            "id": args.checkpoint_model_id,
            "revision": args.checkpoint_model_revision,
        },
        "checks": {
            "routing_config_exact": True,
            "routing_config": canonical_config,
            "weight_index_key_set_exact": True,
            "weight_index_key_count": len(canonical_map),
            "canonical_physical_tensor_count": len(canonical_physical_keys),
            "checkpoint_physical_tensor_count": len(checkpoint_physical_keys),
            "checkpoint_missing_physical_tensor_keys": missing_checkpoint_keys,
            "checkpoint_missing_physical_tensor_key_patterns": args.allow_missing_checkpoint_key_regex,
            "checkpoint_physical_tensor_delta_accounted": True,
            "safetensors_total_size": canonical_index.get("metadata", {}).get("total_size"),
            "router_tensors_exact": True,
            "router_tensor_count": len(router_keys),
            "router_tensor_set_sha256": canonical_router_digest,
            "dequantized_wo_a_tensors_exact": True,
            "dequantized_wo_a_tensor_count": len(wo_a_weight_keys),
            "dequantized_wo_a_tensor_set_sha256": dequantized_wo_a_digest,
            "tokenizer_artifacts_exact": True,
            "tokenizer_artifact_sha256": tokenizer_files,
            "checkpoint_readme_sha256": _sha256(args.checkpoint_model_dir / "README.md"),
            "checkpoint_readme_claims": args.expected_readme_substring,
            "canonical_weight_index_sha256": _sha256(canonical_index_path),
            "checkpoint_weight_index_sha256": _sha256(checkpoint_index_path),
        },
        "required_runtime_environment": required_runtime_environment,
        "scope": (
            "Artifact-level routing-equivalence evidence for an official weight-only repack: "
            "routing configuration, weight-index identity, accounted physical tensor deltas, every router "
            "tensor, exact canonical-FP8-to-checkpoint-BF16 wo_a dequantization, tokenizer artifacts, "
            "required runtime semantics, and the checkpoint's "
            "no-retraining provenance claim are verified."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-model-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-model-dir", type=Path, required=True)
    parser.add_argument("--canonical-model-id", required=True)
    parser.add_argument("--canonical-model-revision", required=True)
    parser.add_argument("--checkpoint-model-id", required=True)
    parser.add_argument("--checkpoint-model-revision", required=True)
    parser.add_argument("--expected-readme-substring", action="append", default=[])
    parser.add_argument("--allow-missing-checkpoint-key-regex", action="append", default=[])
    parser.add_argument("--required-runtime-environment", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = build_report(args)
    except Exception as error:
        report = {
            "schema_version": 1,
            "status": "FAIL",
            "created_at": _utc_now(),
            "canonical_model": {"id": args.canonical_model_id, "revision": args.canonical_model_revision},
            "collection_checkpoint": {
                "id": args.checkpoint_model_id,
                "revision": args.checkpoint_model_revision,
            },
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
