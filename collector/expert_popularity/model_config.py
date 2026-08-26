# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed structural resolution for expert-popularity campaigns."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RoutingDimensions:
    architecture: str
    num_layers: int
    num_experts: int
    top_k: int
    moe_layer_ids: tuple[int, ...]


def resolve_routing_dimensions(config: dict) -> RoutingDimensions:
    architectures = config.get("architectures")
    if not isinstance(architectures, list) or len(architectures) != 1 or not isinstance(architectures[0], str):
        raise ValueError("config.architectures must contain exactly one architecture")
    architecture = architectures[0]
    routing_config = config
    if architecture == "KimiK25ForConditionalGeneration":
        routing_config = config.get("text_config")
        if not isinstance(routing_config, dict):
            raise ValueError("KimiK25ForConditionalGeneration requires config.text_config")

    num_layers = int(routing_config.get("num_hidden_layers", 0))
    num_experts = int(
        routing_config.get("n_routed_experts")
        or routing_config.get("num_experts")
        or routing_config.get("num_local_experts")
        or 0
    )
    top_k = int(routing_config.get("num_experts_per_tok") or routing_config.get("experts_per_token") or 0)
    if num_layers <= 0 or num_experts <= 0 or not 0 < top_k <= num_experts:
        raise ValueError("config does not declare valid routed-expert dimensions")

    first_dense = routing_config.get("first_k_dense_replace")
    moe_frequency = routing_config.get("moe_layer_freq")
    sparse_step = routing_config.get("decoder_sparse_step")
    mlp_only_layers = tuple(int(value) for value in (routing_config.get("mlp_only_layers") or ()))
    if first_dense is not None and moe_frequency is not None:
        first_dense = int(first_dense)
        moe_frequency = int(moe_frequency)
        if first_dense < 0 or moe_frequency <= 0:
            raise ValueError("config has invalid DeepSeek MoE layer placement")
        layer_ids = tuple(
            layer_id for layer_id in range(num_layers) if layer_id >= first_dense and layer_id % moe_frequency == 0
        )
    elif sparse_step is not None:
        if int(sparse_step) != 1:
            raise ValueError("decoder_sparse_step values other than 1 require an explicit serving-source integration")
        excluded = set(mlp_only_layers)
        layer_ids = tuple(layer_id for layer_id in range(num_layers) if layer_id not in excluded)
    elif architecture in {
        "DeepseekV4ForCausalLM",
        "GptOssForCausalLM",
        "MiniMaxM2ForCausalLM",
    }:
        # SGLang 0.5.14 constructs routed MoE in every decoder layer for these
        # architectures: deepseek_v4.py:1091-1128 and
        # minimax_m2.py:931-960 in the pinned runtime image.
        layer_ids = tuple(range(num_layers))
    else:
        raise ValueError(f"unsupported MoE layer-placement contract for architecture {architecture!r}")
    if not layer_ids:
        raise ValueError("config resolves to zero MoE layers")
    return RoutingDimensions(architecture, num_layers, num_experts, top_k, layer_ids)


def validate_declared_routing_dimensions(
    config: dict,
    *,
    num_layers: int,
    num_experts: int,
    top_k: int,
    moe_layer_ids: list[int],
) -> RoutingDimensions:
    resolved = resolve_routing_dimensions(config)
    declared = (num_layers, num_experts, top_k, tuple(moe_layer_ids))
    actual = (resolved.num_layers, resolved.num_experts, resolved.top_k, resolved.moe_layer_ids)
    if declared != actual:
        raise ValueError(f"declared routing dimensions {declared!r} do not match checkpoint config {actual!r}")
    return resolved
