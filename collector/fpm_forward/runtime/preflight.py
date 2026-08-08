# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail before model loading unless the image provides PR11509 native FPM."""

from __future__ import annotations

import json
from pathlib import Path

from dynamo.vllm.instrumented_scheduler import BenchmarkPoint, InstrumentedScheduler

GRAPH_AWARE_FIELDS = {
    "point_type",
    "benchmark_id",
    "total_prefill_tokens",
    "total_kv_read_tokens",
    "batch_size",
}
GRAPH_AWARE_METHODS = {
    "_bench_prefill_scheduled_tokens_per_req",
    "_bench_prefill_blocks_per_req",
    "_bench_blocks_per_req",
    "_bench_available_blocks",
    "_bench_usable_blocks",
    "_bench_prefill_point_feasible",
    "_bench_decode_point_feasible",
    "_bench_cudagraph_metadata",
    "_bench_seed_prompt_len",
    "_bench_cache_fake_prefixes",
    "_bench_save_current_point",
    "_bench_write_results",
}


def main() -> None:
    fields = set(getattr(BenchmarkPoint, "__dataclass_fields__", {}))
    missing_fields = sorted(GRAPH_AWARE_FIELDS - fields)
    missing_methods = sorted(name for name in GRAPH_AWARE_METHODS if not hasattr(InstrumentedScheduler, name))
    audit = {
        "schema_version": 1,
        "runtime_contract": "dynamo_pr11509_native_schema_v2",
        "benchmark_point_fields": sorted(fields),
        "missing_fields": missing_fields,
        "missing_methods": missing_methods,
        "status": "passed" if not missing_fields and not missing_methods else "failed",
    }
    Path("/results/runtime-preflight.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if missing_fields or missing_methods:
        raise RuntimeError(
            "Dynamo runtime lacks the required PR11509 native FPM contract; "
            f"missing_fields={missing_fields}, missing_methods={missing_methods}. "
            "Provide a compatible Dynamo image."
        )


if __name__ == "__main__":
    main()
