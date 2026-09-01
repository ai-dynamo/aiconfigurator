# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail before model loading unless the image provides native FPM + KV warm-up."""

from __future__ import annotations

import json
from pathlib import Path

_AUDIT_PATH = Path("/results/runtime-preflight.json")

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
    "_kvwarm_warm_eligible",
}


def _write_audit(audit: dict) -> None:
    _AUDIT_PATH.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")


def main() -> None:
    # The audit artifact exists precisely to document a rejected image, so an
    # image whose runtime module is missing entirely (pre-PR11509) must still
    # produce it before this process fails the pod.
    try:
        from dynamo.vllm.instrumented_scheduler import BenchmarkPoint, InstrumentedScheduler
    except ImportError as error:
        _write_audit(
            {
                "schema_version": 1,
                "runtime_contract": "dynamo_pr11509_native_schema_v2_kvwarm_v1",
                "benchmark_point_fields": [],
                "missing_fields": [],
                "missing_methods": [],
                "import_error": str(error),
                "status": "failed",
            }
        )
        raise RuntimeError(
            "Dynamo runtime lacks the required native FPM/KV-warm contract; "
            f"importing dynamo.vllm.instrumented_scheduler failed: {error}. "
            "Provide a compatible Dynamo image."
        ) from error

    fields = set(getattr(BenchmarkPoint, "__dataclass_fields__", {}))
    missing_fields = sorted(GRAPH_AWARE_FIELDS - fields)
    missing_methods = sorted(name for name in GRAPH_AWARE_METHODS if not hasattr(InstrumentedScheduler, name))
    audit = {
        "schema_version": 1,
        "runtime_contract": "dynamo_pr11509_native_schema_v2_kvwarm_v1",
        "benchmark_point_fields": sorted(fields),
        "missing_fields": missing_fields,
        "missing_methods": missing_methods,
        "status": "passed" if not missing_fields and not missing_methods else "failed",
    }
    _write_audit(audit)
    if missing_fields or missing_methods:
        raise RuntimeError(
            "Dynamo runtime lacks the required native FPM/KV-warm contract; "
            f"missing_fields={missing_fields}, missing_methods={missing_methods}. "
            "Provide a compatible Dynamo image."
        )


if __name__ == "__main__":
    main()
