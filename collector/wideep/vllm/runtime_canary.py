# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Non-publishing torchrun canary for every vLLM DeepEP serving path."""

from __future__ import annotations

import json
import os

from collector.wideep.sglang.collect_moe_a2a import MoeA2AShape, derive_dist_identity
from collector.wideep.vllm.collect_moe_a2a import (
    BACKENDS,
    VllmBenchmarkAdapter,
    _init_nccl_group,
    build_case_plan,
    collect_with_adapter,
    select_canary_cases,
)


def main() -> None:
    import torch
    import torch.distributed as dist

    identity = derive_dist_identity(
        dict(os.environ),
        gpus_per_node=int(os.environ["AIC_GPUS_PER_NODE"]),
        visible_device_count=torch.cuda.device_count(),
    )
    group = _init_nccl_group(identity)

    def agree(failed: bool) -> bool:
        value = torch.tensor([int(failed)], dtype=torch.int64, device="cuda")
        dist.all_reduce(value, op=dist.ReduceOp.MAX, group=group)
        return bool(value.item())

    # 512 is HT-only and 1 is LL, so v2 exercises both eager/context and
    # cudagraph-compatible/generation invocation contracts.
    cases = select_canary_cases(
        build_case_plan(
            shapes=[MoeA2AShape(hidden_size=7168, topk=8, num_experts=256)],
            grid={"ht_token_counts": [512], "ll_token_counts": [1], "sms": [20]},
            world_size=identity.world_size,
            backends=BACKENDS,
        )
    )
    try:
        result = collect_with_adapter(
            cases,
            adapter=VllmBenchmarkAdapter(group, identity, warmups=1, runs=2),
            world_size=identity.world_size,
            failure_agreement=agree,
        )
        if result.failures:
            raise RuntimeError(
                "vLLM DeepEP runtime canary failures: "
                + json.dumps([failure.__dict__ for failure in result.failures], default=str)
            )
        if identity.rank == 0:
            print(
                json.dumps(
                    {
                        "status": "passed",
                        "cases": len(cases),
                        "rows": len(result.rows),
                        "modes": [[case.comm_backend, case.inference_phase] for case in result.resolved_cases],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        if dist.is_initialized():
            dist.barrier(group=group)
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
