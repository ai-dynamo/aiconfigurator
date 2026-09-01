# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Non-publishing MPI canary for every TensorRT-LLM DeepEP serving path."""

from __future__ import annotations

import json
import os

from collector.wideep.sglang.collect_moe_a2a import MoeA2AShape
from collector.wideep.trtllm.collect_moe_a2a import (
    TensorRTLLMBenchmarkAdapter,
    build_case_plan,
    build_unified_rows,
    derive_dist_identity,
    select_canary_cases,
)


def main() -> None:
    from tensorrt_llm._utils import mpi_allgather, mpi_barrier

    identity = derive_dist_identity(
        dict(os.environ),
        gpus_per_node=int(os.environ["AIC_GPUS_PER_NODE"]),
    )
    cases = select_canary_cases(
        build_case_plan(
            shapes=[MoeA2AShape(hidden_size=7168, topk=8, num_experts=256)],
            token_grid={"ht_token_counts": [16], "ll_token_counts": [1]},
            ep_size=identity.ep_size,
            node_num=identity.node_num,
        )
    )
    adapter = TensorRTLLMBenchmarkAdapter(warmup=1, iterations=2)
    injected_rank = int(os.environ.get("AIC_INJECT_FAILURE_RANK", "-1"))
    failures = []
    rows = 0
    for case in cases:
        injected_error = "RuntimeError: injected pre-benchmark rank failure" if identity.rank == injected_rank else None
        injected_errors = mpi_allgather(injected_error)
        if any(error is not None for error in injected_errors):
            failures.append(
                {
                    "backend": case.comm_backend,
                    "dtype": case.quant.comm_dtype,
                    "rank_errors": injected_errors,
                }
            )
            continue
        local_error = None
        result = None
        try:
            all_rank_num_tokens = list(mpi_allgather(case.num_tokens))
            result = adapter.run(case, all_rank_num_tokens)
            build_unified_rows(case, result)
        except Exception as error:
            local_error = f"{type(error).__name__}: {error}"
        gathered = mpi_allgather(local_error)
        if any(error is not None for error in gathered):
            failures.append(
                {
                    "backend": case.comm_backend,
                    "dtype": case.quant.comm_dtype,
                    "rank_errors": gathered,
                }
            )
        else:
            assert result is not None
            rows += len(result.measurements)
    mpi_barrier()
    if failures:
        raise RuntimeError("TensorRT-LLM DeepEP runtime canary failures: " + json.dumps(failures))
    if identity.rank == 0:
        print(
            json.dumps(
                {
                    "status": "passed",
                    "cases": len(cases),
                    "rows": rows,
                    "modes": [[case.comm_backend, case.quant.comm_dtype] for case in cases],
                    "injected_failure_rank": injected_rank,
                },
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
