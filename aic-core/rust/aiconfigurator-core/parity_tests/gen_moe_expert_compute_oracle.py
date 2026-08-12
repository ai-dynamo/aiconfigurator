# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate the Python oracle for the Rust ``moe_expert_compute`` perf table.

Dumps stratified ``(key, num_tokens) -> PerfDatabase.query_moe_expert_compute(...)``
samples from the shipped h200_sxm/sglang and gb200/trtllm data to
``src/perf_database/testdata/moe_expert_compute_oracle.json``; the Rust
``#[cfg(test)] moe_ep_matches_python_oracle`` test loads the same parquet
files through ``MoeExpertComputeTable`` and asserts a relative error <= 1e-9.

Regenerate (from the repo root, after `git lfs pull`):

    .venv/bin/python aic-core/rust/aiconfigurator-core/parity_tests/gen_moe_expert_compute_oracle.py

Sampling is fully deterministic (sorted coordinates + fixed per-category
strides), so a regeneration with unchanged data produces an identical file.
Categories: exact collected points; interior token lerps; token overflow /
underflow (boundary util-hold anchored on the WideEP roofline SOL);
EPLB-redundant ``num_slots != num_experts`` slices (gb200); and the
distribution chain's ``"uniform"`` fallback (h200 sglang) and
first-available-in-insertion-order fallback (gb200, whose table has no
``uniform`` — the production-default ``"power_law"`` request resolves to
``power_law_1.01_eplb``, the first collected distribution).
"""

from __future__ import annotations

import itertools
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from aiconfigurator_core.sdk.common import MoEQuantMode
from aiconfigurator_core.sdk.operations.moe_comm import MoEExpertCompute
from aiconfigurator_core.sdk.perf_database import get_database

OUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "src", "perf_database", "testdata", "moe_expert_compute_oracle.json"
)

# (system, backend, version) tuples whose shipped data feeds the oracle: the
# legacy sglang wideep context/generation pair (kernel "deepep_moe", six
# distributions incl. "uniform", num_slots == num_experts) and the legacy
# trtllm wideep table (native kernel, both-phase duplication, ``_eplb``
# distributions, num_slots 256/288/384 incl. the EPLB-redundant 288-over-256
# slices, and no "uniform").
TUPLES = [
    ("h200_sxm", "sglang", "0.5.6.post2"),
    ("gb200", "trtllm", "1.3.0rc10"),
]

# Per-category sample budget (per database tuple). Categories are filled with
# an even stride over their deterministically ordered candidate list, so every
# kernel/distribution/phase family stays represented.
QUOTAS = {
    "exact": 35,
    "token_interp": 20,
    "token_overflow": 15,
    "token_underflow": 10,
    "slots_redundant_exact": 10,
    "dist_uniform_fallback": 10,
    "dist_first_available": 10,
}

# The production-default distribution request neither shipped table collects
# under this exact name — the chain's fallback trigger.
UNCOLLECTED_DISTRIBUTION = "power_law"


def walk_store(data):
    """Yield ``(coord_tuple, {num_tokens: leaf})`` for every collected slice.

    ``coord_tuple`` is ``(kernel_source, quant_name, distribution,
    inference_phase, topk, num_experts, num_slots, hidden_size, inter_size,
    moe_tp_size, moe_ep_size)`` — the 11 levels above the token axis, with
    the ``MoEQuantMode`` member flattened to its name so tuples sort.
    """
    for kernel_source, by_quant in data.items():
        for quant, by_dist in by_quant.items():
            for distribution, by_phase in by_dist.items():
                for inference_phase, by_topk in by_phase.items():
                    for topk, by_experts in by_topk.items():
                        for num_experts, by_slots in by_experts.items():
                            for num_slots, by_hidden in by_slots.items():
                                for hidden_size, by_inter in by_hidden.items():
                                    for inter_size, by_tp in by_inter.items():
                                        for moe_tp_size, by_ep in by_tp.items():
                                            for moe_ep_size, curve in by_ep.items():
                                                yield (
                                                    (
                                                        kernel_source,
                                                        quant.name,
                                                        distribution,
                                                        inference_phase,
                                                        topk,
                                                        num_experts,
                                                        num_slots,
                                                        hidden_size,
                                                        inter_size,
                                                        moe_tp_size,
                                                        moe_ep_size,
                                                    ),
                                                    curve,
                                                )


def first_available_distributions(store):
    """``(kernel_source, quant_name, phase) -> first collected distribution``
    carrying that phase, in dict-INSERTION (file row) order — the anchor of
    ``_resolve_ep_distribution``'s third leg."""
    first: dict[tuple, str] = {}
    for kernel_source, by_quant in store.items():
        for quant, by_dist in by_quant.items():
            # Distribution buckets iterate in insertion (file row) order, so
            # the first one seen per phase is the chain's fallback pick.
            for distribution, by_phase in by_dist.items():
                for phase in by_phase:
                    first.setdefault((kernel_source, quant.name, phase), distribution)
    return first


def candidates_for(db):
    """Build the per-category candidate lists for one database tuple."""
    MoEExpertCompute.load_data(db)
    store = db._moe_ep_data
    slices = sorted(walk_store(store), key=lambda item: item[0])
    first_by_phase = first_available_distributions(store)

    # Distributions collected per (kernel_source, quant_name, phase) — the
    # chain's candidate scope.
    dists_by_phase: dict[tuple, set[str]] = {}
    for coord, _curve in slices:
        dists_by_phase.setdefault((coord[0], coord[1], coord[3]), set()).add(coord[2])

    out: dict[str, list] = {name: [] for name in QUOTAS}
    for coord, curve in slices:
        tokens = sorted(curve)
        if not tokens:
            continue

        # Exact collected points: first, middle, last token of the curve.
        for t in dict.fromkeys([tokens[0], tokens[len(tokens) // 2], tokens[-1]]):
            out["exact"].append((coord, t, "exact"))

        # Interior lerp: midpoint of the widest adjacent token gap.
        gaps = [(hi - lo, lo, hi) for lo, hi in itertools.pairwise(tokens) if hi - lo > 1]
        if gaps:
            _, lo, hi = max(gaps)
            out["token_interp"].append((coord, (lo + hi) // 2, "token_interp"))

        # Beyond the collected range: boundary util-hold on the roofline SOL.
        out["token_overflow"].append((coord, tokens[-1] * 2 + 1, "token_overflow"))
        # Below it — only where the singleton-underflow guard cannot fire.
        if tokens[0] > 1 and len(tokens) > 1:
            out["token_underflow"].append((coord, max(1, tokens[0] // 2), "token_underflow"))

        # EPLB redundant slices: num_slots != num_experts (gb200 288-over-256).
        if coord[6] != coord[5]:
            out["slots_redundant_exact"].append((coord, tokens[len(tokens) // 2], "slots_redundant_exact"))

        # Distribution chain: request an uncollected distribution; step 2
        # fires where "uniform" is collected for the phase, step 3 otherwise.
        # Candidate shapes come from the slice of the distribution the chain
        # RESOLVES to, so the post-fallback shape walk is guaranteed to hit.
        scope = (coord[0], coord[1], coord[3])
        collected = dists_by_phase[scope]
        assert UNCOLLECTED_DISTRIBUTION not in collected, f"{UNCOLLECTED_DISTRIBUTION} unexpectedly collected"
        if "uniform" in collected:
            if coord[2] == "uniform":
                out["dist_uniform_fallback"].append(
                    (coord, tokens[len(tokens) // 2], "dist_uniform_fallback", UNCOLLECTED_DISTRIBUTION)
                )
        elif coord[2] == first_by_phase[scope]:
            out["dist_first_available"].append(
                (coord, tokens[len(tokens) // 2], "dist_first_available", UNCOLLECTED_DISTRIBUTION)
            )

    return out


def build_samples(db, system, data_root):
    out = candidates_for(db)
    samples = []
    for category, quota in QUOTAS.items():
        pool = out[category]
        if not pool:
            continue
        stride = max(1, len(pool) // quota)
        for item in pool[::stride][:quota]:
            coord, num_tokens, kind = item[0], item[1], item[2]
            (
                kernel_source,
                quant_name,
                distribution,
                inference_phase,
                topk,
                num_experts,
                num_slots,
                hidden_size,
                inter_size,
                moe_tp_size,
                moe_ep_size,
            ) = coord
            if kind in ("dist_uniform_fallback", "dist_first_available"):
                distribution = item[3]
            result = db.query_moe_expert_compute(
                kernel_source,
                MoEQuantMode[quant_name],
                distribution,
                inference_phase,
                topk,
                num_experts,
                num_slots,
                hidden_size,
                inter_size,
                moe_tp_size,
                moe_ep_size,
                num_tokens,
            )
            samples.append(
                {
                    "data_root": data_root,
                    "system": system,
                    "kind": kind,
                    "kernel_source": kernel_source,
                    "quant": quant_name,
                    "distribution": distribution,
                    "inference_phase": inference_phase,
                    "topk": int(topk),
                    "num_experts": int(num_experts),
                    "num_slots": int(num_slots),
                    "hidden_size": int(hidden_size),
                    "inter_size": int(inter_size),
                    "moe_tp_size": int(moe_tp_size),
                    "moe_ep_size": int(moe_ep_size),
                    "num_tokens": int(num_tokens),
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
        samples.extend(build_samples(db, system, data_root))

    header = {
        "_regenerate": (
            ".venv/bin/python aic-core/rust/aiconfigurator-core/parity_tests/gen_moe_expert_compute_oracle.py"
        ),
        "_source": "PerfDatabase.query_moe_expert_compute (shared_layer=False), SILICON mode",
        "_tuples": [f"{s}/{b}/{v}" for s, b, v in TUPLES],
    }
    out_path = os.path.normpath(OUT_PATH)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # One sample per line: this fixture is machine-generated and read as a
    # diff, so a pretty-printed 16-lines-per-sample dump is pure noise.
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
