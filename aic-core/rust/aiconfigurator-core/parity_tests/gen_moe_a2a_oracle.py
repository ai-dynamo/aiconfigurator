# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate the Python oracle for the Rust ``moe_a2a`` perf table.

Dumps stratified ``(key, num_tokens) -> PerfDatabase.query_moe_a2a(...)``
samples from the shipped h200_sxm/sglang and gb200/trtllm data to
``src/perf_database/testdata/moe_a2a_oracle.json``; the Rust
``#[cfg(test)] moe_a2a_matches_python_oracle`` test loads the same parquet
files through ``MoeA2aTable`` and asserts a relative error <= 1e-9.

Regenerate (from the repo root, after `git lfs pull`):

    .venv/bin/python aic-core/rust/aiconfigurator-core/parity_tests/gen_moe_a2a_oracle.py

Sampling is fully deterministic (sorted coordinates + fixed per-category
strides), so a regeneration with unchanged data produces an identical file.
Categories: exact collected points; interior token lerps; token overflow /
underflow (boundary util-hold on the linear token proxy); interior sms lerps
and off-grid sms snaps (the 2-D ``(sms, num_tokens)`` grid); and the
comm-dtype chain's ``fp8_block`` -> ``fp8`` alias and sole-collected-dtype
fallback.
"""

from __future__ import annotations

import itertools
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from aiconfigurator_core.sdk.operations.moe_comm import MoEAllToAll
from aiconfigurator_core.sdk.perf_database import get_database

OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "src", "perf_database", "testdata", "moe_a2a_oracle.json")

# (system, backend, version) tuples whose shipped data feeds the oracle: the
# sglang legacy DeepEP pair (deepep_ht + deepep_ll, sole "default" dtype) and
# the trtllm legacy alltoall table (both NVLink kernels, four dtypes incl. the
# low-precision combine's "fp4").
TUPLES = [
    ("h200_sxm", "sglang", "0.5.6.post2"),
    ("gb200", "trtllm", "1.3.0rc10"),
]

# Per-category sample budget. Categories are filled with an even stride over
# their deterministically ordered candidate list, so every backend/phase/dtype
# family stays represented.
QUOTAS = {
    "exact": 30,
    "token_interp": 20,
    "token_overflow": 14,
    "token_underflow": 6,
    "sms_interp": 14,
    "sms_snap": 10,
    "dtype_alias": 6,
    "dtype_sole": 8,
}


def walk_store(data):
    """Yield ``(coord_tuple, {num_tokens: leaf})`` for every collected slice.

    ``coord_tuple`` is ``(comm_backend, phase, comm_dtype, ep_size, node_num,
    hidden_size, topk, num_experts, sms)`` — the 9 levels above the token axis.
    """
    for comm_backend, by_phase in data.items():
        for phase, by_dtype in by_phase.items():
            for comm_dtype, by_ep in by_dtype.items():
                for ep_size, by_node in by_ep.items():
                    for node_num, by_hidden in by_node.items():
                        for hidden_size, by_topk in by_hidden.items():
                            for topk, by_experts in by_topk.items():
                                for num_experts, by_sms in by_experts.items():
                                    for sms, curve in by_sms.items():
                                        yield (
                                            (
                                                comm_backend,
                                                phase,
                                                comm_dtype,
                                                ep_size,
                                                node_num,
                                                hidden_size,
                                                topk,
                                                num_experts,
                                                sms,
                                            ),
                                            curve,
                                        )


def candidates_for(db):
    """Build the per-category candidate lists for one database tuple."""
    MoEAllToAll.load_data(db)
    store = db._moe_a2a_data
    slices = sorted(walk_store(store), key=lambda item: item[0])

    # sms values collected per shape prefix (everything above the sms level).
    sms_by_prefix: dict[tuple, list[int]] = {}
    for coord, _curve in slices:
        sms_by_prefix.setdefault(coord[:-1], []).append(coord[-1])
    for values in sms_by_prefix.values():
        values.sort()

    # comm_dtypes collected per (comm_backend, phase) — the dtype chain's scope.
    dtypes_by_phase: dict[tuple, set[str]] = {}
    for coord, _curve in slices:
        dtypes_by_phase.setdefault((coord[0], coord[1]), set()).add(coord[2])

    out: dict[str, list] = {name: [] for name in QUOTAS}
    for coord, curve in slices:
        tokens = sorted(curve)
        if not tokens:
            continue
        prefix, sms = coord[:-1], coord[-1]
        sms_values = sms_by_prefix[prefix]

        # Exact collected points: first, middle, last token of the curve.
        for t in dict.fromkeys([tokens[0], tokens[len(tokens) // 2], tokens[-1]]):
            out["exact"].append((coord, t, "exact"))

        # Interior lerp: midpoint of the widest adjacent token gap.
        gaps = [(hi - lo, lo, hi) for lo, hi in itertools.pairwise(tokens) if hi - lo > 1]
        if gaps:
            _, lo, hi = max(gaps)
            out["token_interp"].append((coord, (lo + hi) // 2, "token_interp"))

        # Beyond the collected range: boundary util-hold on the linear proxy.
        out["token_overflow"].append((coord, tokens[-1] * 2 + 1, "token_overflow"))
        if tokens[0] > 1:
            out["token_underflow"].append((coord, max(1, tokens[0] // 2), "token_underflow"))

        # sms axis (2-D grid): only meaningful off the collected sms keys.
        if sms == sms_values[0]:
            sms_gaps = [(hi - lo, lo, hi) for lo, hi in itertools.pairwise(sms_values) if hi - lo > 1]
            if sms_gaps:
                _, lo, hi = max(sms_gaps)
                out["sms_interp"].append((coord, tokens[len(tokens) // 2], "sms_interp", (lo + hi) // 2))
            # Off-grid below the smallest and above the largest collected sms.
            out["sms_snap"].append((coord, tokens[len(tokens) // 2], "sms_snap", max(0, sms_values[0] - 1) or 1))
            out["sms_snap"].append((coord, tokens[len(tokens) // 2], "sms_snap", sms_values[-1] + 7))

        # comm_dtype chain.
        collected = dtypes_by_phase[(coord[0], coord[1])]
        if "fp8" in collected and "fp8_block" not in collected:
            out["dtype_alias"].append((coord, tokens[len(tokens) // 2], "dtype_alias", "fp8_block"))
        # Sole-dtype fallback samples exist only for the UNTYPED legacy slice:
        # a sole TYPED dtype raises the named miss (the review-fix contract in
        # `_resolve_comm_dtype_slice`), so typed-sole coordinates produce no
        # resolvable oracle sample.
        if len(collected) == 1 and "default" in collected:
            for requested in ("fp8", "nvfp4", "fp8_block"):
                if requested not in collected:
                    out["dtype_sole"].append((coord, tokens[len(tokens) // 2], "dtype_sole", requested))

    return out


def build_samples(db, data_root):
    out = candidates_for(db)
    samples = []
    for category, quota in QUOTAS.items():
        pool = out[category]
        if not pool:
            continue
        stride = max(1, len(pool) // quota)
        for item in pool[::stride][:quota]:
            coord, num_tokens, kind = item[0], item[1], item[2]
            comm_backend, phase, comm_dtype, ep, node, hidden, topk, experts, sms = coord
            if kind in ("sms_interp", "sms_snap"):
                sms = item[3]
            elif kind in ("dtype_alias", "dtype_sole"):
                comm_dtype = item[3]
            result = db.query_moe_a2a(
                comm_backend,
                phase,
                comm_dtype,
                ep,
                node,
                hidden,
                topk,
                experts,
                num_tokens,
                sms=sms,
            )
            samples.append(
                {
                    "data_root": data_root,
                    "kind": kind,
                    "comm_backend": comm_backend,
                    "phase": phase,
                    "comm_dtype": comm_dtype,
                    "ep_size": int(ep),
                    "node_num": int(node),
                    "hidden_size": int(hidden),
                    "topk": int(topk),
                    "num_experts": int(experts),
                    "num_tokens": int(num_tokens),
                    "sms": int(sms),
                    "latency_ms": float(result),
                }
            )
    return samples


def main() -> None:
    samples = []
    for system, backend, version in TUPLES:
        db = get_database(system, backend, version, shared_layer=False)
        if db is None:
            raise SystemExit(f"no database for {system}/{backend}/{version}")
        data_root = os.path.join(db.system_spec["data_dir"], backend, version)
        samples.extend(build_samples(db, data_root))

    header = {
        "_regenerate": (".venv/bin/python aic-core/rust/aiconfigurator-core/parity_tests/gen_moe_a2a_oracle.py"),
        "_source": "PerfDatabase.query_moe_a2a (shared_layer=False), SILICON mode",
        "_tuples": [f"{s}/{b}/{v}" for s, b, v in TUPLES],
    }
    out_path = os.path.normpath(OUT_PATH)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # One sample per line: this fixture is machine-generated and read as a
    # diff, so a pretty-printed 14-lines-per-sample dump is pure noise.
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("{\n")
        for key, value in header.items():
            f.write(f" {json.dumps(key)}: {json.dumps(value)},\n")
        f.write(' "samples": [\n')
        rows = ",\n".join(f"  {json.dumps(sample, sort_keys=True)}" for sample in samples)
        f.write(rows + "\n")
        f.write(" ]\n}\n")
    print(f"wrote {len(samples)} samples to {out_path}")


if __name__ == "__main__":
    main()
