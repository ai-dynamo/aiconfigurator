# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collector-facing DeepEP backend identities and serving contracts.

These names are persisted in ``moe_a2a_perf.comm_backend``. Frameworks may
share DeepEP source code, but they do not share latency rows: initialization,
buffer sizing, quant ordering, and scheduler capacity are framework-owned.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DeepEPCollectorContract:
    framework: str
    selector: str
    inference_phases: tuple[str, ...]
    comm_phases: tuple[str, ...]
    comm_dtypes: tuple[str, ...]
    sms_policy: str
    capacity_policy: str
    kernel_source: str = "deepep"


DEEPEP_COLLECTOR_CONTRACTS: dict[str, DeepEPCollectorContract] = {
    "deepep_ht": DeepEPCollectorContract(
        framework="vllm",
        selector="--all2all-backend deepep_high_throughput",
        inference_phases=("context",),
        comm_phases=("dispatch", "combine"),
        comm_dtypes=("default",),
        # vLLM serving defaults to 20; arbitrary SGLang Config sweeps are not
        # serving-equivalent and therefore are not written under this identity.
        sms_policy="fixed:20",
        capacity_policy="VLLM_DEEPEP_BUFFER_SIZE_MB=1024",
    ),
    "deepep_ll": DeepEPCollectorContract(
        framework="vllm",
        selector="--all2all-backend deepep_low_latency",
        inference_phases=("generation",),
        comm_phases=("dispatch", "combine"),
        comm_dtypes=("default",),
        sms_policy="fixed:0",
        capacity_policy="max_num_tokens_per_dp_rank",
    ),
    "deepep_v2": DeepEPCollectorContract(
        framework="vllm",
        selector="--all2all-backend deepep_v2",
        inference_phases=("context", "generation"),
        comm_phases=("dispatch", "combine"),
        comm_dtypes=("default",),
        sms_policy="ElasticBuffer.get_theoretical_num_sms",
        capacity_policy="power_of_two_num_max_tokens_per_rank",
    ),
    "trtllm_deepep_ht": DeepEPCollectorContract(
        framework="trtllm",
        selector="TRTLLM_FORCE_COMM_METHOD=DEEPEP",
        inference_phases=("context",),
        comm_phases=("dispatch", "combine"),
        comm_dtypes=("bfloat16", "nvfp4"),
        sms_policy="fixed:0",
        capacity_policy="CommunicationFactory DeepEP auto config",
    ),
    "trtllm_deepep_ll": DeepEPCollectorContract(
        framework="trtllm",
        selector="TRTLLM_FORCE_COMM_METHOD=DEEPEPLOWLATENCY",
        inference_phases=("generation",),
        comm_phases=("dispatch", "combine"),
        comm_dtypes=("bfloat16", "fp8", "nvfp4", "w4afp8", "fp4"),
        sms_policy="fixed:0",
        capacity_policy="TRTLLM_DEEP_EP_TOKEN_LIMIT",
    ),
}


def contract_for(framework: str, backend: str) -> DeepEPCollectorContract:
    """Return a contract only when both persisted identity parts agree."""
    contract = DEEPEP_COLLECTOR_CONTRACTS.get(backend)
    if contract is None:
        raise KeyError(f"unknown DeepEP collector backend {backend!r}")
    if contract.framework != framework:
        raise ValueError(
            f"DeepEP backend {backend!r} belongs to {contract.framework}, "
            f"not {framework}; cross-framework latency reuse is forbidden"
        )
    return contract
