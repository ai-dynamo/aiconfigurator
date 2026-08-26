# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decode and validate SGLang response-level routed-expert captures."""

from __future__ import annotations

import base64
import binascii
import math

import numpy as np


def _js_divergence(left: np.ndarray, right: np.ndarray) -> float:
    left = left.astype(np.float64) / float(left.sum())
    right = right.astype(np.float64) / float(right.sum())
    middle = (left + right) / 2.0

    def kl_divergence(value: np.ndarray, reference: np.ndarray) -> float:
        mask = value > 0
        return float(np.sum(value[mask] * np.log2(value[mask] / reference[mask])))

    return 0.5 * kl_divergence(left, middle) + 0.5 * kl_divergence(right, middle)


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = left.astype(np.float64) - float(np.mean(left))
    right = right.astype(np.float64) - float(np.mean(right))
    denominator = math.sqrt(float(np.dot(left, left) * np.dot(right, right)))
    if denominator == 0:
        return 1.0 if np.array_equal(left, right) else 0.0
    return float(np.dot(left, right) / denominator)


def pairwise_stability(shard_counts: list[np.ndarray], moe_layer_ids: list[int]) -> dict:
    comparisons = []
    for left_index in range(len(shard_counts)):
        for right_index in range(left_index + 1, len(shard_counts)):
            pearsons = []
            divergences = []
            top1_matches = []
            for layer_id in moe_layer_ids:
                left = shard_counts[left_index][layer_id]
                right = shard_counts[right_index][layer_id]
                pearsons.append(_pearson(left, right))
                divergences.append(_js_divergence(left, right))
                top1_matches.append(int(np.argmax(left)) == int(np.argmax(right)))
            comparisons.append(
                {
                    "left_shard": left_index,
                    "right_shard": right_index,
                    "mean_layer_pearson": float(np.mean(pearsons)),
                    "mean_layer_jensen_shannon_divergence_bits": float(np.mean(divergences)),
                    "top1_expert_match_rate": float(np.mean(top1_matches)),
                }
            )
    return {
        "comparisons": comparisons,
        "minimum_mean_layer_pearson": min(item["mean_layer_pearson"] for item in comparisons),
        "maximum_mean_layer_jensen_shannon_divergence_bits": max(
            item["mean_layer_jensen_shannon_divergence_bits"] for item in comparisons
        ),
    }


def repeat_stability(repeat_counts: list[list[np.ndarray]], moe_layer_ids: list[int]) -> dict:
    comparisons = []
    for repeat_index in range(1, len(repeat_counts)):
        for shard_index, baseline in enumerate(repeat_counts[0]):
            candidate = repeat_counts[repeat_index][shard_index]
            comparison = pairwise_stability([baseline, candidate], moe_layer_ids)["comparisons"][0]
            comparisons.append(
                {
                    "baseline_repeat": 0,
                    "candidate_repeat": repeat_index,
                    "shard_index": shard_index,
                    "counts_exact": bool(np.array_equal(baseline, candidate)),
                    "mean_layer_pearson": comparison["mean_layer_pearson"],
                    "mean_layer_jensen_shannon_divergence_bits": comparison[
                        "mean_layer_jensen_shannon_divergence_bits"
                    ],
                    "top1_expert_match_rate": comparison["top1_expert_match_rate"],
                }
            )
    return {
        "comparisons": comparisons,
        "all_counts_exact": all(item["counts_exact"] for item in comparisons),
        "minimum_mean_layer_pearson": min(item["mean_layer_pearson"] for item in comparisons),
        "maximum_mean_layer_jensen_shannon_divergence_bits": max(
            item["mean_layer_jensen_shannon_divergence_bits"] for item in comparisons
        ),
    }


def decode_routed_experts(
    encoded: object,
    *,
    prompt_tokens: int,
    num_layers: int,
    top_k: int,
) -> np.ndarray:
    if not isinstance(encoded, str) or not encoded:
        raise RuntimeError("response is missing non-empty meta_info.routed_experts")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise RuntimeError("meta_info.routed_experts is not valid base64") from error
    expected_values = prompt_tokens * num_layers * top_k
    expected_bytes = expected_values * np.dtype(np.int32).itemsize
    if len(raw) != expected_bytes:
        raise RuntimeError(
            "routed-expert response has the wrong byte length: "
            f"{len(raw)} != {expected_bytes} "
            f"({prompt_tokens=} {num_layers=} {top_k=})"
        )
    return np.frombuffer(raw, dtype=np.int32).reshape(prompt_tokens, num_layers, top_k).copy()


def aggregate_routed_experts(
    routed_experts: np.ndarray,
    *,
    num_layers: int,
    num_experts: int,
    top_k: int,
    moe_layer_ids: list[int],
) -> np.ndarray:
    if routed_experts.ndim != 3 or routed_experts.shape[1:] != (num_layers, top_k):
        raise RuntimeError(
            f"unexpected routed-expert shape {routed_experts.shape}; expected [tokens, {num_layers}, {top_k}]"
        )
    counts = np.zeros((num_layers, num_experts), dtype=np.int64)
    for layer_id in moe_layer_ids:
        selected = routed_experts[:, layer_id, :]
        invalid = selected[(selected < 0) | (selected >= num_experts)]
        if invalid.size:
            raise RuntimeError(
                f"layer {layer_id} returned out-of-range logical expert IDs: {np.unique(invalid)[:16].tolist()}"
            )
        counts[layer_id] = np.bincount(selected.reshape(-1), minlength=num_experts)
        expected = routed_experts.shape[0] * top_k
        if int(counts[layer_id].sum()) != expected:
            raise RuntimeError(
                f"layer {layer_id} response-capture conservation failed: {int(counts[layer_id].sum())} != {expected}"
            )
    return counts
