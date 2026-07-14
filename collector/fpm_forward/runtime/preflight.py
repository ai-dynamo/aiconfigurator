# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail before model loading when the runtime lacks the P>0 benchmark contract."""

from __future__ import annotations

import json
from pathlib import Path

from dynamo.vllm.instrumented_scheduler import BenchmarkPoint, InstrumentedScheduler


def main() -> None:
    fields = set(getattr(BenchmarkPoint, "__dataclass_fields__", {}))
    required_fields = {"point_type", "isl", "kv_read_tokens", "context_length", "batch_size"}
    required_methods = {
        "_bench_prefill_scheduled_tokens_per_req",
        "_bench_prefill_blocks_per_req",
        "_bench_blocks_per_req",
    }
    missing_fields = sorted(required_fields - fields)
    missing_methods = sorted(name for name in required_methods if not hasattr(InstrumentedScheduler, name))
    audit = {
        "schema_version": 1,
        "benchmark_point_fields": sorted(fields),
        "missing_fields": missing_fields,
        "missing_methods": missing_methods,
        "status": "passed" if not missing_fields and not missing_methods else "failed",
    }
    Path("/results/runtime-preflight.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if missing_fields or missing_methods:
        raise RuntimeError(
            "Dynamo runtime lacks the required FPM P>0 benchmark contract; "
            f"missing_fields={missing_fields}, missing_methods={missing_methods}. "
            "Provide a compatible image or FpmCollector.runtime_overlay_dir."
        )


if __name__ == "__main__":
    main()
