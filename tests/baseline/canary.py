"""
Canary matrix fixtures for generator v2 parity testing.

These cases capture real generator outputs using build_naive_generator_params
and are used to establish byte-level parity baselines for the generator rewrite.
"""

from dataclasses import dataclass
from typing import Any

from aiconfigurator.generator.naive import build_naive_generator_params


@dataclass(frozen=True)
class CanaryCase:
    name: str
    backend: str
    backend_version: str
    params: dict


def _make_params(
    model_name: str,
    backend_name: str,
    mode: str = "agg",
    total_gpus: int = 8,
    system_name: str = "h200_sxm",
    transport: str | None = None,
) -> dict[str, Any]:
    k8s_overrides: dict[str, Any] = {"k8s_namespace": "dynamo"}
    if transport is not None:
        k8s_overrides["transport"] = transport
    return build_naive_generator_params(
        model_name=model_name,
        total_gpus=total_gpus,
        system_name=system_name,
        backend_name=backend_name,
        mode=mode,
        generator_overrides={"K8sConfig": k8s_overrides},
    )


_DENSE_MODEL = "Qwen/Qwen3-0.6B"
_MOE_MODEL = "Qwen/Qwen3-30B-A3B"
_SYSTEM = "h200_sxm"
_TOTAL_GPUS = 8

CANARY_CASES: list[CanaryCase] = [
    CanaryCase(
        name="vllm_dense_agg",
        backend="vllm",
        backend_version="0.20.1",
        params=_make_params(_DENSE_MODEL, "vllm", mode="agg", total_gpus=_TOTAL_GPUS, system_name=_SYSTEM),
    ),
    CanaryCase(
        name="sglang_dense_agg",
        backend="sglang",
        backend_version="0.5.11",
        params=_make_params(_DENSE_MODEL, "sglang", mode="agg", total_gpus=_TOTAL_GPUS, system_name=_SYSTEM),
    ),
    CanaryCase(
        name="trtllm_dense_agg",
        backend="trtllm",
        backend_version="1.3.0rc14",
        params=_make_params(_DENSE_MODEL, "trtllm", mode="agg", total_gpus=_TOTAL_GPUS, system_name=_SYSTEM),
    ),
    CanaryCase(
        name="vllm_moe_disagg",
        backend="vllm",
        backend_version="0.20.1",
        params=_make_params(_MOE_MODEL, "vllm", mode="disagg", total_gpus=_TOTAL_GPUS, system_name=_SYSTEM),
    ),
    CanaryCase(
        name="trtllm_moe_disagg",
        backend="trtllm",
        backend_version="1.3.0rc14",
        params=_make_params(_MOE_MODEL, "trtllm", mode="disagg", total_gpus=_TOTAL_GPUS, system_name=_SYSTEM),
    ),
    CanaryCase(
        name="deepseek_vllm_h200_agg",
        backend="vllm",
        backend_version="0.20.1",
        params=_make_params("deepseek-ai/DeepSeek-V4-Pro", "vllm", mode="agg", total_gpus=_TOTAL_GPUS, system_name=_SYSTEM),
    ),
    CanaryCase(
        name="deepseek_sglang_h200_agg",
        backend="sglang",
        backend_version="0.5.11",
        params=_make_params("deepseek-ai/DeepSeek-V4-Pro", "sglang", mode="agg", total_gpus=_TOTAL_GPUS, system_name=_SYSTEM),
    ),
    # Blackwell cases: exercise the hardware moe_backend fact (Blackwell-gated).
    # trtllm -> WIDEEP on b200/gb200 (CUTLASS on Hopper); sglang -> deepep_moe on b200/gb200.
    CanaryCase(
        name="deepseek_trtllm_b200_disagg",
        backend="trtllm",
        backend_version="1.3.0rc14",
        params=_make_params("deepseek-ai/DeepSeek-V4-Pro", "trtllm", mode="disagg", total_gpus=_TOTAL_GPUS, system_name="b200_sxm"),
    ),
    CanaryCase(
        name="deepseek_sglang_gb200_agg",
        backend="sglang",
        backend_version="0.5.11",
        params=_make_params("deepseek-ai/DeepSeek-V4-Pro", "sglang", mode="agg", total_gpus=_TOTAL_GPUS, system_name="gb200"),
    ),
    # GB200 disagg: exercises the disagg_decode shared-memory fact (128Gi on the
    # decode worker vs 64Gi default on prefill) — the only path that reads
    # hardware.shared_memory.disagg_decode.
    CanaryCase(
        name="deepseek_trtllm_gb200_disagg",
        backend="trtllm",
        backend_version="1.3.0rc14",
        params=_make_params("deepseek-ai/DeepSeek-V4-Pro", "trtllm", mode="disagg", total_gpus=_TOTAL_GPUS, system_name="gb200"),
    ),
    # Transport selector: ib transport injects UCX_* env into worker pods.
    CanaryCase(
        name="qwen_moe_vllm_gb200_ib",
        backend="vllm",
        backend_version="0.20.1",
        params=_make_params(_MOE_MODEL, "vllm", mode="agg", total_gpus=_TOTAL_GPUS, system_name="gb200", transport="ib"),
    ),
    # Transport selector: efa transport injects FI_* env AND privileged pod
    # requirements (hostIPC / IPC_LOCK / vpc.amazonaws.com/efa NIC) into WORKER
    # pods only (frontend gets none).
    CanaryCase(
        name="deepseek_sglang_gb200_efa",
        backend="sglang",
        backend_version="0.5.11",
        params=_make_params("deepseek-ai/DeepSeek-V4-Pro", "sglang", mode="agg", total_gpus=_TOTAL_GPUS, system_name="gb200", transport="efa"),
    ),
    # Multinode: a worker whose GPU count exceeds the node's GPU count triggers
    # ComputeDomain + resourceClaims + multinode.nodeCount emission (NVLink-fabric
    # / GB200-class deployments). b200_sxm has 8 GPUs/node so naive picks a TP=8
    # worker; overriding NodeConfig.num_gpus_per_node to 4 makes 8 > 4 multinode
    # (nodeCount=2). Only path that exercises the compute-domain builder branch.
    CanaryCase(
        name="deepseek_trtllm_b200_multinode",
        backend="trtllm",
        backend_version="1.3.0rc14",
        params=build_naive_generator_params(
            model_name="deepseek-ai/DeepSeek-V4-Pro",
            total_gpus=16,
            system_name="b200_sxm",
            backend_name="trtllm",
            mode="disagg",
            generator_overrides={
                "K8sConfig": {"k8s_namespace": "dynamo"},
                "NodeConfig": {"num_gpus_per_node": 4},
            },
        ),
    ),
]
