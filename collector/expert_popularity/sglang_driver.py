# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Drive a running SGLang routing observer with canonical inputs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import subprocess
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import requests
import torch
import transformers
from transformers import AutoTokenizer, PreTrainedTokenizerBase

if __package__:
    from .model_config import validate_declared_routing_dimensions
    from .response_capture import (
        aggregate_routed_experts,
        decode_routed_experts,
        pairwise_stability,
        repeat_stability,
    )
else:
    from model_config import validate_declared_routing_dimensions
    from response_capture import (
        aggregate_routed_experts,
        decode_routed_experts,
        pairwise_stability,
        repeat_stability,
    )

SCHEMA_VERSION = 1
DEFAULT_SEEDS = (2026082501, 2026082502, 2026082503, 2026082504)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def _append_jsonl(path: Path, value: object) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")))
        handle.write("\n")


def _load_tokenizer(tokenizer_path: Path) -> PreTrainedTokenizerBase:
    """Load the tokenizer without requiring AutoConfig to know a new model type."""
    tokenizer_config_path = tokenizer_path / "tokenizer_config.json"
    if tokenizer_config_path.is_file():
        tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
        tokenizer_class_name = tokenizer_config.get("tokenizer_class")
        tokenizer_class = (
            getattr(transformers, tokenizer_class_name, None) if isinstance(tokenizer_class_name, str) else None
        )
        if isinstance(tokenizer_class, type) and issubclass(tokenizer_class, PreTrainedTokenizerBase):
            return tokenizer_class.from_pretrained(
                tokenizer_path,
                trust_remote_code=True,
                local_files_only=True,
            )
    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, local_files_only=True)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    return _sha256_bytes(value.contiguous().numpy().tobytes())


def _post(base_url: str, endpoint: str, payload: object | None = None, timeout: int = 1800) -> requests.Response:
    response = requests.post(base_url + endpoint, json={} if payload is None else payload, timeout=timeout)
    response.raise_for_status()
    return response


def _parse_layer_ids(raw: str) -> list[int]:
    result: list[int] = []
    for part in raw.split(","):
        if "-" in part:
            start, end = (int(value) for value in part.split("-", 1))
            result.extend(range(start, end + 1))
        else:
            result.append(int(part))
    if not result or len(result) != len(set(result)) or result != sorted(result):
        raise ValueError(f"invalid --moe-layer-ids: {raw!r}")
    return result


def _build_workload(tokenizer, seed: int, tokens_per_shard: int, isl_min: int, isl_max: int) -> list[dict]:
    special_ids = {int(value) for value in tokenizer.all_special_ids}
    vocabulary = sorted({int(value) for value in tokenizer.get_vocab().values()} - special_ids)
    if not vocabulary:
        raise ValueError("tokenizer has no valid non-special token IDs")
    rng = random.Random(seed)
    requests_for_shard: list[dict] = []
    total_tokens = 0
    while total_tokens < tokens_per_shard:
        isl = rng.randint(isl_min, isl_max)
        input_ids = [vocabulary[rng.randrange(len(vocabulary))] for _ in range(isl)]
        encoded = json.dumps(input_ids, separators=(",", ":")).encode()
        requests_for_shard.append(
            {
                "request_index": len(requests_for_shard),
                "isl": isl,
                "input_ids": input_ids,
                "input_ids_sha256": _sha256_bytes(encoded),
            }
        )
        total_tokens += isl
    return requests_for_shard


def _wait_for_dump(raw_dir: Path, before: set[Path]) -> Path:
    for _ in range(120):
        created = sorted(set(raw_dir.glob("*.pt")) - before)
        if created:
            if len(created) != 1:
                raise RuntimeError(f"expected one recorder dump, found {[str(path) for path in created]}")
            return created[0]
        time.sleep(1)
    raise TimeoutError("recorder dump did not appear within 120 seconds")


def _save_numpy(path: Path, value: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    temporary.replace(path)


def _run_recorder_window(
    *,
    args: argparse.Namespace,
    prepared: list[dict],
    raw_dir: Path,
    responses_path: Path,
    repeat_index: int,
    shard_index: int,
    moe_layer_ids: list[int],
) -> tuple[np.ndarray, dict]:
    before = set(raw_dir.glob("*.pt"))
    start_response = _post(args.base_url, "/start_expert_distribution_record")
    total_tokens = 0
    for request in prepared:
        response = _post(
            args.base_url,
            "/generate",
            {
                "input_ids": request["input_ids"],
                "sampling_params": {"max_new_tokens": 1, "temperature": 0, "ignore_eos": True},
            },
        )
        body = response.json()
        prompt_tokens = int(body["meta_info"]["prompt_tokens"])
        if prompt_tokens != request["isl"]:
            raise RuntimeError(f"request ISL changed in serving: {prompt_tokens} != {request['isl']}")
        total_tokens += prompt_tokens
        _append_jsonl(
            responses_path,
            {
                "repeat_index": repeat_index,
                "shard_index": shard_index,
                "request_index": request["request_index"],
                "http_status": response.status_code,
                "response_id": body["meta_info"]["id"],
                "output_ids": body["output_ids"],
            },
        )
    stop_response = _post(args.base_url, "/stop_expert_distribution_record")
    dump_response = _post(args.base_url, "/dump_expert_distribution_record")
    dump_path = _wait_for_dump(raw_dir, before)

    raw_count = torch.load(dump_path, map_location="cpu", weights_only=True)["logical_count"]
    if list(raw_count.shape[-2:]) != [args.num_layers, args.num_experts]:
        raise RuntimeError(
            f"unexpected logical_count tail shape {list(raw_count.shape)}; "
            f"expected [..., {args.num_layers}, {args.num_experts}]"
        )
    if raw_count.dtype not in {torch.int32, torch.int64} or not bool(torch.all(raw_count >= 0)):
        raise RuntimeError(f"logical_count must be a non-negative integer tensor, got {raw_count.dtype}")
    aggregate = raw_count.reshape(-1, args.num_layers, args.num_experts).sum(dim=0, dtype=torch.int64)
    layer_totals = aggregate.sum(dim=1)
    moe_layer_set = set(moe_layer_ids)
    expected = total_tokens * args.top_k * args.replication_factor
    if all(int(layer_totals[layer_id].item()) == 0 for layer_id in moe_layer_ids):
        raise RuntimeError(
            "expert recorder observed zero assignments for every declared MoE layer; "
            "the framework-selected routing backend may bypass the recorder hook"
        )
    for layer_id in range(args.num_layers):
        actual = int(layer_totals[layer_id].item())
        if layer_id in moe_layer_set:
            if actual != expected:
                raise RuntimeError(f"layer {layer_id} conservation failed before normalization: {actual} != {expected}")
        elif actual != 0:
            raise RuntimeError(f"dense layer {layer_id} unexpectedly recorded {actual} assignments")
    if bool(torch.any(aggregate % args.replication_factor != 0)):
        raise RuntimeError("logical_count is not exactly divisible by the declared replication factor")
    normalized = (aggregate // args.replication_factor).numpy().astype(np.int64)
    return normalized, {
        "repeat_index": repeat_index,
        "shard_index": shard_index,
        "prompt_tokens": total_tokens,
        "request_count": len(prepared),
        "recorder_file": str(dump_path),
        "raw_logical_count_shape": list(raw_count.shape),
        "raw_logical_count_sha256": _tensor_sha256(raw_count),
        "normalized_logical_count_sha256": _sha256_bytes(normalized.tobytes()),
        "endpoint_status": {
            "start": start_response.status_code,
            "stop": stop_response.status_code,
            "dump": dump_response.status_code,
        },
    }


def _run_response_window(
    *,
    args: argparse.Namespace,
    prepared: list[dict],
    raw_dir: Path,
    responses_path: Path,
    repeat_index: int,
    shard_index: int,
    moe_layer_ids: list[int],
) -> tuple[np.ndarray, dict]:
    route_dir = raw_dir / "routed_experts"
    route_dir.mkdir(exist_ok=True)
    aggregate = np.zeros((args.num_layers, args.num_experts), dtype=np.int64)
    total_tokens = 0
    raw_files: list[dict] = []
    for request in prepared:
        request_id = f"expert-pop-r{repeat_index:02d}-s{shard_index:02d}-q{request['request_index']:04d}"
        response = _post(
            args.base_url,
            "/generate",
            {
                "rid": request_id,
                "input_ids": request["input_ids"],
                "sampling_params": {"max_new_tokens": 1, "temperature": 0, "ignore_eos": True},
                "return_routed_experts": True,
                "routed_experts_start_len": 0,
            },
            timeout=args.request_timeout,
        )
        body = response.json()
        meta_info = body.get("meta_info")
        if not isinstance(meta_info, dict) or "prompt_tokens" not in meta_info:
            diagnostic_body = dict(body)
            if isinstance(diagnostic_body.get("meta_info"), dict):
                diagnostic_body["meta_info"] = {
                    key: value for key, value in diagnostic_body["meta_info"].items() if key != "routed_experts"
                }
            _append_jsonl(
                responses_path,
                {
                    "repeat_index": repeat_index,
                    "shard_index": shard_index,
                    "request_index": request["request_index"],
                    "request_id": request_id,
                    "http_status": response.status_code,
                    "invalid_response": diagnostic_body,
                },
            )
            raise RuntimeError(
                f"request {request_id} returned no prompt_tokens; "
                f"meta_info keys={sorted(meta_info) if isinstance(meta_info, dict) else None}"
            )
        prompt_tokens = int(meta_info["prompt_tokens"])
        if prompt_tokens != request["isl"]:
            raise RuntimeError(f"request ISL changed in serving: {prompt_tokens} != {request['isl']}")
        routed_experts = decode_routed_experts(
            meta_info.get("routed_experts"),
            prompt_tokens=prompt_tokens,
            num_layers=args.num_layers,
            top_k=args.top_k,
        )
        request_counts = aggregate_routed_experts(
            routed_experts,
            num_layers=args.num_layers,
            num_experts=args.num_experts,
            top_k=args.top_k,
            moe_layer_ids=moe_layer_ids,
        )
        aggregate += request_counts
        total_tokens += prompt_tokens
        raw_path = route_dir / (
            f"repeat-{repeat_index:02d}-shard-{shard_index:02d}-request-{request['request_index']:04d}.npy"
        )
        _save_numpy(raw_path, routed_experts)
        raw_sha256 = _sha256_bytes(raw_path.read_bytes())
        raw_files.append(
            {
                "path": str(raw_path),
                "shape": list(routed_experts.shape),
                "dtype": str(routed_experts.dtype),
                "sha256": raw_sha256,
            }
        )
        _append_jsonl(
            responses_path,
            {
                "repeat_index": repeat_index,
                "shard_index": shard_index,
                "request_index": request["request_index"],
                "request_id": request_id,
                "http_status": response.status_code,
                "response_id": meta_info["id"],
                "prompt_tokens": prompt_tokens,
                "output_ids": body["output_ids"],
                "routed_experts_file": str(raw_path),
                "routed_experts_shape": list(routed_experts.shape),
                "routed_experts_sha256": raw_sha256,
            },
        )

    layer_totals = aggregate.sum(axis=1)
    expected = total_tokens * args.top_k
    moe_layer_set = set(moe_layer_ids)
    for layer_id in range(args.num_layers):
        actual = int(layer_totals[layer_id])
        if layer_id in moe_layer_set:
            if actual != expected:
                raise RuntimeError(f"layer {layer_id} response conservation failed: {actual} != {expected}")
        elif actual != 0:
            raise RuntimeError(f"dense layer {layer_id} unexpectedly recorded {actual} assignments")
    return aggregate, {
        "repeat_index": repeat_index,
        "shard_index": shard_index,
        "prompt_tokens": total_tokens,
        "request_count": len(prepared),
        "observation_source": "response_routed_experts",
        "count_replication_factor": 1,
        "raw_file_count": len(raw_files),
        "raw_files": raw_files,
        "normalized_logical_count_sha256": _sha256_bytes(aggregate.tobytes()),
        "endpoint_status": {"generate": 200},
    }


def _run_window(**kwargs) -> tuple[np.ndarray, dict]:
    args = kwargs["args"]
    if args.observation_source == "recorder":
        return _run_recorder_window(**kwargs)
    if args.observation_source == "response_routed_experts":
        return _run_response_window(**kwargs)
    raise AssertionError(f"unhandled observation source {args.observation_source!r}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:31014")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--checkpoint-model-id")
    parser.add_argument("--checkpoint-model-revision")
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--num-experts", type=int, required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--moe-layer-ids", required=True)
    parser.add_argument("--seeds", default=",".join(str(value) for value in DEFAULT_SEEDS))
    parser.add_argument("--tokens-per-shard", type=int, default=65536)
    parser.add_argument("--isl-min", type=int, default=128)
    parser.add_argument("--isl-max", type=int, default=4096)
    parser.add_argument("--repeat-count", type=int, default=2)
    parser.add_argument("--request-timeout", type=int, default=600)
    parser.add_argument("--replication-factor", type=int, default=1)
    parser.add_argument(
        "--observation-source",
        choices=("recorder", "response_routed_experts"),
        default="recorder",
    )
    parser.add_argument("--expected-framework-version", default="0.5.14")
    parser.add_argument("--image-reference", required=True)
    parser.add_argument("--image-sha256", required=True)
    parser.add_argument(
        "--collector-code-sha256",
        default=os.environ.get("COLLECTOR_CODE_SHA256", "unrecorded_smoke"),
    )
    parser.add_argument("--server-args-json", required=True)
    parser.add_argument("--runtime-environment-json", default="{}")
    parser.add_argument("--routing-observation-method", required=True)
    parser.add_argument("--routing-equivalence-evidence-json", default="{}")
    parser.add_argument("--min-shard-pearson", type=float, default=0.95)
    parser.add_argument("--max-shard-jsd", type=float, default=0.01)
    parser.add_argument("--repeat-validation-mode", choices=("exact", "aggregate"), default="exact")
    parser.add_argument("--min-repeat-pearson", type=float, default=0.999)
    parser.add_argument("--max-repeat-jsd", type=float, default=0.001)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    checkpoint_model_id = args.checkpoint_model_id or args.model_id
    checkpoint_model_revision = args.checkpoint_model_revision or args.model_revision
    if bool(args.checkpoint_model_id) != bool(args.checkpoint_model_revision):
        raise ValueError("checkpoint model ID and revision must be supplied together")
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.artifact_dir / "raw"
    workload_dir = args.artifact_dir / "workload"
    raw_dir.mkdir(exist_ok=True)
    workload_dir.mkdir(exist_ok=True)
    result_path = args.artifact_dir / "collection_result.json"
    result: dict = {
        "schema_version": SCHEMA_VERSION,
        "status": "FAIL",
        "created_at": _utc_now(),
        "model": {"id": args.model_id, "revision": args.model_revision},
        "collection_checkpoint": {
            "id": checkpoint_model_id,
            "revision": checkpoint_model_revision,
        },
    }
    _write_json(result_path, result)

    try:
        framework_version = importlib.metadata.version("sglang")
        if framework_version != args.expected_framework_version:
            raise RuntimeError(
                f"installed SGLang version {framework_version!r} != expected {args.expected_framework_version!r}"
            )
        moe_layer_ids = _parse_layer_ids(args.moe_layer_ids)
        if moe_layer_ids[-1] >= args.num_layers or not 0 < args.top_k <= args.num_experts:
            raise ValueError("declared routing dimensions are inconsistent")
        seeds = [int(value) for value in args.seeds.split(",")]
        if len(seeds) < 2 or len(seeds) != len(set(seeds)):
            raise ValueError("at least two distinct seeds are required")
        if args.repeat_count < 2:
            raise ValueError("repeat-count must be at least two")
        if args.request_timeout <= 0:
            raise ValueError("request-timeout must be positive")
        if not -1.0 <= args.min_shard_pearson <= 1.0:
            raise ValueError("min-shard-pearson must be between -1 and 1")
        if not 0.0 <= args.max_shard_jsd <= 1.0:
            raise ValueError("max-shard-jsd must be between 0 and 1")
        if not -1.0 <= args.min_repeat_pearson <= 1.0:
            raise ValueError("min-repeat-pearson must be between -1 and 1")
        if not 0.0 <= args.max_repeat_jsd <= 1.0:
            raise ValueError("max-repeat-jsd must be between 0 and 1")
        if args.observation_source == "response_routed_experts" and args.replication_factor < 1:
            raise ValueError("serving replication factor must be positive")

        model_config = json.loads((Path(args.tokenizer_path) / "config.json").read_text(encoding="utf-8"))
        routing_equivalence_evidence = json.loads(args.routing_equivalence_evidence_json)
        runtime_environment = json.loads(args.runtime_environment_json)
        if not isinstance(routing_equivalence_evidence, dict):
            raise TypeError("routing equivalence evidence must be a JSON object")
        if not isinstance(runtime_environment, dict):
            raise TypeError("runtime environment must be a JSON object")
        checkpoint_differs = (checkpoint_model_id, checkpoint_model_revision) != (
            args.model_id,
            args.model_revision,
        )
        if checkpoint_differs:
            expected_canonical = {"id": args.model_id, "revision": args.model_revision}
            expected_checkpoint = {
                "id": checkpoint_model_id,
                "revision": checkpoint_model_revision,
            }
            if routing_equivalence_evidence.get("status") != "PASS":
                raise ValueError("a passing routing-equivalence report is required for a non-canonical checkpoint")
            if routing_equivalence_evidence.get("canonical_model") != expected_canonical:
                raise ValueError("routing-equivalence report has the wrong canonical model identity")
            if routing_equivalence_evidence.get("collection_checkpoint") != expected_checkpoint:
                raise ValueError("routing-equivalence report has the wrong collection checkpoint identity")
            required_environment = routing_equivalence_evidence.get("required_runtime_environment")
            if not isinstance(required_environment, dict):
                raise ValueError("routing-equivalence report is missing required runtime environment")
            mismatched_environment = {
                name: {"expected": value, "actual": runtime_environment.get(name)}
                for name, value in required_environment.items()
                if runtime_environment.get(name) != value
            }
            if mismatched_environment:
                raise ValueError(
                    f"runtime environment does not satisfy routing-equivalence requirements: {mismatched_environment}"
                )
        resolved_dimensions = validate_declared_routing_dimensions(
            model_config,
            num_layers=args.num_layers,
            num_experts=args.num_experts,
            top_k=args.top_k,
            moe_layer_ids=moe_layer_ids,
        )
        tokenizer = _load_tokenizer(Path(args.tokenizer_path))
        workloads = []
        for shard_index, seed in enumerate(seeds):
            prepared = _build_workload(tokenizer, seed, args.tokens_per_shard, args.isl_min, args.isl_max)
            workload_path = workload_dir / f"shard-{shard_index:02d}.json"
            workload = {
                "shard_index": shard_index,
                "seed": seed,
                "requests": prepared,
                "total_prompt_tokens": sum(item["isl"] for item in prepared),
            }
            _write_json(workload_path, workload)
            workloads.append(workload)

        responses_path = args.artifact_dir / "responses.jsonl"
        responses_path.unlink(missing_ok=True)
        repeat_counts: list[list[np.ndarray]] = []
        windows = []
        for repeat_index in range(args.repeat_count):
            one_repeat = []
            for shard_index, workload in enumerate(workloads):
                normalized, window = _run_window(
                    args=args,
                    prepared=workload["requests"],
                    raw_dir=raw_dir,
                    responses_path=responses_path,
                    repeat_index=repeat_index,
                    shard_index=shard_index,
                    moe_layer_ids=moe_layer_ids,
                )
                one_repeat.append(normalized)
                windows.append(window)
            repeat_counts.append(one_repeat)

        repeat_stability_result = repeat_stability(repeat_counts, moe_layer_ids)
        if args.repeat_validation_mode == "exact" and not repeat_stability_result["all_counts_exact"]:
            first_mismatch = next(item for item in repeat_stability_result["comparisons"] if not item["counts_exact"])
            raise RuntimeError(
                "routing was not exactly repeatable for "
                f"shard {first_mismatch['shard_index']}, repeat {first_mismatch['candidate_repeat']}"
            )
        if args.repeat_validation_mode == "aggregate":
            if repeat_stability_result["minimum_mean_layer_pearson"] < args.min_repeat_pearson:
                raise RuntimeError(
                    "repeat popularity was insufficiently correlated: "
                    f"{repeat_stability_result['minimum_mean_layer_pearson']} < {args.min_repeat_pearson}"
                )
            if repeat_stability_result["maximum_mean_layer_jensen_shannon_divergence_bits"] > args.max_repeat_jsd:
                raise RuntimeError(
                    "repeat popularity diverged beyond the gate: "
                    f"{repeat_stability_result['maximum_mean_layer_jensen_shannon_divergence_bits']} "
                    f"> {args.max_repeat_jsd}"
                )
        stability = pairwise_stability(repeat_counts[0], moe_layer_ids)
        if stability["minimum_mean_layer_pearson"] < args.min_shard_pearson:
            raise RuntimeError(
                "independent shard popularity was insufficiently correlated: "
                f"{stability['minimum_mean_layer_pearson']} < {args.min_shard_pearson}"
            )
        if stability["maximum_mean_layer_jensen_shannon_divergence_bits"] > args.max_shard_jsd:
            raise RuntimeError(
                "independent shard popularity diverged beyond the gate: "
                f"{stability['maximum_mean_layer_jensen_shannon_divergence_bits']} > {args.max_shard_jsd}"
            )

        canonical_count = np.sum(np.stack(repeat_counts[0], axis=0), axis=0, dtype=np.int64)
        total_prompt_tokens = sum(workload["total_prompt_tokens"] for workload in workloads)
        for layer_id in moe_layer_ids:
            actual = int(canonical_count[layer_id].sum())
            expected = total_prompt_tokens * args.top_k
            if actual != expected:
                raise RuntimeError(f"canonical layer {layer_id} conservation failed: {actual} != {expected}")
        counts_path = args.artifact_dir / "normalized_counts.json"
        _write_json(
            counts_path,
            {
                "schema_version": SCHEMA_VERSION,
                "model_id": args.model_id,
                "model_revision": args.model_revision,
                "num_layers": args.num_layers,
                "num_experts": args.num_experts,
                "top_k": args.top_k,
                "moe_layer_ids": moe_layer_ids,
                "routed_token_count": total_prompt_tokens,
                "counts": {str(layer_id): canonical_count[layer_id].tolist() for layer_id in moe_layer_ids},
            },
        )

        result.update(
            {
                "status": "PASS",
                "finished_at": _utc_now(),
                "model": {
                    "id": args.model_id,
                    "revision": args.model_revision,
                    "tokenizer_revision": checkpoint_model_revision,
                    "architecture": resolved_dimensions.architecture,
                },
                "routing": {
                    "num_layers": args.num_layers,
                    "moe_layer_ids": moe_layer_ids,
                    "num_routed_experts": args.num_experts,
                    "top_k": args.top_k,
                    "replication_factor": args.replication_factor,
                },
                "workload": {
                    "phase": "prefill",
                    "token_distribution": "uniform_valid_non_special_token_ids",
                    "seeds": seeds,
                    "shard_count": len(seeds),
                    "tokens_per_shard_minimum": args.tokens_per_shard,
                    "total_prompt_tokens": total_prompt_tokens,
                    "isl_distribution": "discrete_uniform",
                    "isl_min": args.isl_min,
                    "isl_max": args.isl_max,
                    "request_count": sum(len(workload["requests"]) for workload in workloads),
                    "repeat_count": args.repeat_count,
                    "max_new_tokens": 1,
                    "temperature": 0,
                    "sequential": True,
                    "tokenizer_vocab_size": int(tokenizer.vocab_size),
                    "excluded_special_token_ids": sorted(int(value) for value in tokenizer.all_special_ids),
                },
                "stability": stability,
                "repeat_stability": repeat_stability_result,
                "validation_gates": {
                    "repeat_validation_mode": args.repeat_validation_mode,
                    "repeat_counts_exact": repeat_stability_result["all_counts_exact"],
                    "minimum_repeat_mean_layer_pearson": args.min_repeat_pearson,
                    "maximum_repeat_mean_layer_jensen_shannon_divergence_bits": args.max_repeat_jsd,
                    "minimum_mean_layer_pearson": args.min_shard_pearson,
                    "maximum_mean_layer_jensen_shannon_divergence_bits": args.max_shard_jsd,
                    "layer_assignment_conservation": True,
                },
                "collection": {
                    "framework": "sglang",
                    "framework_version": framework_version,
                    "collection_checkpoint": {
                        "id": checkpoint_model_id,
                        "revision": checkpoint_model_revision,
                    },
                    "routing_equivalence_evidence": routing_equivalence_evidence or None,
                    "image_reference": args.image_reference,
                    "image_sha256": args.image_sha256,
                    "collector_code_sha256": args.collector_code_sha256,
                    "server_args": json.loads(args.server_args_json),
                    "runtime_environment": runtime_environment,
                    "routing_observation_method": args.routing_observation_method,
                    "observation_source": args.observation_source,
                    "count_replication_factor": (
                        args.replication_factor if args.observation_source == "recorder" else 1
                    ),
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                    "slurm_node": os.environ.get("SLURMD_NODENAME"),
                    "gpu_info": subprocess.check_output(
                        [
                            "nvidia-smi",
                            "--query-gpu=index,name,uuid,memory.total,driver_version,compute_cap",
                            "--format=csv,noheader",
                        ],
                        text=True,
                    ).splitlines(),
                },
                "windows": windows,
                "artifacts": {
                    "normalized_counts": str(counts_path),
                    "responses": str(responses_path),
                    "workload_directory": str(workload_dir),
                    "raw_observation_directory": str(raw_dir),
                    "raw_recorder_directory": str(raw_dir),
                },
            }
        )
    except Exception as error:
        result.update(
            {
                "status": "FAIL",
                "finished_at": _utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        _write_json(result_path, result)
        raise
    _write_json(result_path, result)


if __name__ == "__main__":
    main()
