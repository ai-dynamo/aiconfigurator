# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Workload generation and mooncake-style trace loading for the DES oracle."""

from __future__ import annotations

import json
import random
from typing import Optional

from vllm_sim import Request


def synthetic(
    request_count: int,
    isl: int,
    osl: int,
    block_size: int,
    arrival_interval_ms: float = 0.0,
    poisson: bool = False,
    shared_prefix_ratio: float = 0.0,
    num_prefix_groups: int = 8,
    seed: int = 0,
) -> list[Request]:
    """Fixed (isl, osl) requests; optional shared-prefix groups.

    shared_prefix_ratio r: the leading r*isl tokens of each prompt come from
    one of `num_prefix_groups` shared pools (block hashes reused across
    requests in the same group), mirroring dynamo.replay synthetic workloads.
    """
    rng = random.Random(seed)
    reqs = []
    t = 0.0
    shared_blocks = int((isl * shared_prefix_ratio) // block_size)
    n_full = isl // block_size
    for rid in range(request_count):
        group = rng.randrange(num_prefix_groups) if shared_blocks > 0 else 0
        hashes = tuple(("g", group, i) if i < shared_blocks else ("r", rid, i) for i in range(n_full))
        reqs.append(Request(rid=rid, isl=isl, osl=osl, prompt_hashes=hashes, arrival_ms=t))
        if poisson and arrival_interval_ms > 0:
            t += rng.expovariate(1.0 / arrival_interval_ms)
        else:
            t += arrival_interval_ms
    return reqs


def load_mooncake_trace(
    path: str,
    engine_block_size: int,
    trace_block_size: int = 512,
    limit: Optional[int] = None,
) -> list[Request]:
    """Load a mooncake-style jsonl trace.

    Each record: {timestamp, input_length, output_length, hash_ids}.
    Each trace hash_id covers `trace_block_size` tokens; it is expanded into
    trace_block_size/engine_block_size engine-block hashes (must divide).
    Multi-turn `session_id`/`delay` records are flattened to absolute time.
    """
    if trace_block_size % engine_block_size != 0:
        raise ValueError("trace_block_size must be a multiple of engine block size")
    expand = trace_block_size // engine_block_size

    reqs = []
    session_last_end: dict[str, float] = {}
    with open(path) as f:
        for rid, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            isl = int(rec["input_length"])
            osl = int(rec["output_length"])
            sid = rec.get("session_id")
            if "timestamp" in rec or "created_time" in rec:
                t = float(rec.get("timestamp", rec.get("created_time", 0.0)))
            elif sid is not None and sid in session_last_end:
                t = session_last_end[sid] + float(rec.get("delay", rec.get("delay_ms", 0.0)))
            else:
                t = 0.0
            if sid is not None:
                session_last_end[sid] = t

            n_full = isl // engine_block_size
            hashes: list = []
            for h in rec.get("hash_ids", []):
                hashes.extend((h, j) for j in range(expand))
            hashes = hashes[:n_full]
            # pad with unique hashes if the trace under-covers the prompt
            while len(hashes) < n_full:
                hashes.append(("u", rid, len(hashes)))
            reqs.append(Request(rid=rid, isl=isl, osl=max(1, osl), prompt_hashes=tuple(hashes), arrival_ms=t))
            if limit is not None and len(reqs) >= limit:
                break
    return reqs
