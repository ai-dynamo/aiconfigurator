# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate the Python oracle for the Rust large-EP MoE OPERATORS.

Unlike ``gen_moe_a2a_oracle.py`` / ``gen_moe_expert_compute_oracle.py`` — which sample the
raw ``PerfDatabase.query_moe_a2a`` / ``query_moe_expert_compute`` table
lookups — this generator drives
the PYTHON OP OBJECTS (``MoEAllToAll(...).query(db, x=...)`` and
``MoEExpertCompute(...).query(db, x=...)``). What is under test is therefore the op
layer's own arithmetic:

* ``MoEAllToAll``: ``x // attention_tp_size`` (plain floor division, no
  ``max(1, ...)`` guard; never ADP-scaled) and the ``* scale_factor`` tail;
* ``MoEExpertCompute``: ``x * attention_dp_size``, the ``int(tokens * 0.8)`` EPLB context
  correction on the sglang-adapted kernel legs, the ``num_slots`` default, the
  ``kernel_source=None`` auto-resolution, the pinned ``moe_tp_size=1`` and the
  ``* scale_factor`` tail.

Output goes to ``src/operators/testdata/op_oracle.json``; the Rust
``moe_a2a_op_matches_python_oracle`` / ``moe_ep_op_matches_python_oracle``
tests read the two ``op`` slices of that one file, rebuild the same ops
against the same shipped parquet files and assert a relative error <= 1e-9.

Regenerate (from the repo root, after `git lfs pull`):

    .venv/bin/python aic-core/rust/aiconfigurator-core/parity_tests/gen_op_oracle.py

Sampling is fully deterministic (sorted coordinates + fixed per-category
strides), so a regeneration with unchanged data produces an identical file.
"""

from __future__ import annotations

import itertools
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from aiconfigurator_core.sdk.common import MoEQuantMode
from aiconfigurator_core.sdk.operations.moe_comm import MoEAllToAll, MoEExpertCompute
from aiconfigurator_core.sdk.perf_database import get_database

OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "src", "operators", "testdata", "op_oracle.json")

# (system, backend, version) tuples whose shipped data feeds the oracle.
#
# h200_sxm/sglang: the legacy DeepEP comm pair (deepep_ht + deepep_ll, sole
# "default" dtype) and the legacy sglang wideep compute pair — the
# ``deepep_moe`` kernel leg, i.e. the ONLY leg where the EPLB correction
# fires. gb200/trtllm: the legacy NVLink alltoall table (both kernels, four
# dtypes incl. the low-precision combine's "fp4") and the legacy trtllm
# wideep compute table (``wideep_compute_cutlass`` — the leg where EPLB must
# NOT fire, and where ``kernel_source=None`` exercises the availability
# fallback of ``_resolve_kernel_source``).
TUPLES = [
    ("h200_sxm", "sglang", "0.5.6.post2"),
    ("gb200", "trtllm", "1.3.0rc10"),
]

# Per-category sample budget, per database tuple. Categories are filled with
# an even stride over their deterministically ordered candidate list, so every
# backend/phase/kernel family stays represented.
A2A_QUOTAS = {
    "tp1_exact": 8,
    "tp2_exact": 6,
    "tp2_floor": 6,
    "tp4_floor": 4,
    "token_interp": 5,
    "token_overflow": 4,
    "scale_factor": 4,
}

EP_QUOTAS = {
    "adp1_exact": 7,
    "adp2_exact": 5,
    "adp8_exact": 4,
    "eplb_context": 5,
    "eplb_generation": 3,
    "kernel_auto": 5,
    "slots_default": 4,
    "token_overflow": 4,
    "scale_factor": 3,
}

# A non-unit scale_factor for the "scale_factor" categories: the model
# builders pass the layer count, so a realistic value is a plain integer.
SCALE_FACTOR = 61.0


# ---------------------------------------------------------------------------
# moe_a2a (comm)
# ---------------------------------------------------------------------------


def walk_a2a_store(data):
    """Yield ``(coord, {num_tokens: leaf})`` for every collected comm slice.

    ``coord`` is ``(comm_backend, phase, comm_dtype, ep_size, node_num,
    hidden_size, topk, num_experts, sms)``.
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


def a2a_candidates(db):
    """Per-category ``(coord, x, attention_tp_size, scale_factor)`` candidates."""
    MoEAllToAll.load_data(db)
    slices = sorted(walk_a2a_store(db._moe_a2a_data), key=lambda item: item[0])

    out: dict[str, list] = {name: [] for name in A2A_QUOTAS}
    for coord, curve in slices:
        tokens = sorted(curve)
        if not tokens:
            continue
        mid = tokens[len(tokens) // 2]

        # tp=1: x reaches the table untouched.
        out["tp1_exact"].append((coord, tokens[0], 1, 1.0))
        out["tp1_exact"].append((coord, mid, 1, 1.0))
        # tp=2 with an exactly divisible x: x // 2 == mid.
        out["tp2_exact"].append((coord, mid * 2, 2, 1.0))
        # tp=2 with the ODD neighbour: floor division must key the SAME point
        # (a rounding implementation would key mid + 1 instead).
        out["tp2_floor"].append((coord, mid * 2 + 1, 2, 1.0))
        # tp=4 with the largest non-divisible remainder.
        out["tp4_floor"].append((coord, mid * 4 + 3, 4, 1.0))
        # Interior lerp at the midpoint of the widest adjacent gap, reached
        # through a tp=2 key so the division and the interpolation compose.
        gaps = [(hi - lo, lo, hi) for lo, hi in itertools.pairwise(tokens) if hi - lo > 1]
        if gaps:
            _, lo, hi = max(gaps)
            out["token_interp"].append((coord, ((lo + hi) // 2) * 2, 2, 1.0))
        # Beyond the collected range: the boundary util-hold on the linear
        # token proxy SOL.
        out["token_overflow"].append((coord, tokens[-1] * 2 + 1, 1, 1.0))
        out["scale_factor"].append((coord, mid, 2, SCALE_FACTOR))
    return out


def build_a2a_samples(db, system, backend, version, data_root):
    out = a2a_candidates(db)
    samples = []
    for category, quota in A2A_QUOTAS.items():
        pool = out[category]
        if not pool:
            continue
        stride = max(1, len(pool) // quota)
        for coord, x, attention_tp_size, scale_factor in pool[::stride][:quota]:
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
            ) = coord
            op = MoEAllToAll(
                "oracle",
                scale_factor,
                phase=phase,
                comm_backend=comm_backend,
                comm_dtype=comm_dtype,
                hidden_size=hidden_size,
                topk=topk,
                num_experts=num_experts,
                moe_ep_size=ep_size,
                node_num=node_num,
                sms=sms,
                attention_tp_size=attention_tp_size,
            )
            result = op.query(db, x=x)
            samples.append(
                {
                    "op": "moe_a2a",
                    "kind": category,
                    "system": system,
                    "backend": backend,
                    "version": version,
                    "data_root": data_root,
                    "scale_factor": scale_factor,
                    "phase": phase,
                    "comm_backend": comm_backend,
                    "comm_dtype": comm_dtype,
                    "hidden_size": int(hidden_size),
                    "topk": int(topk),
                    "num_experts": int(num_experts),
                    "moe_ep_size": int(ep_size),
                    "node_num": int(node_num),
                    "sms": int(sms),
                    "attention_tp_size": int(attention_tp_size),
                    "x": int(x),
                    "latency_ms": float(result),
                }
            )
    return samples


# ---------------------------------------------------------------------------
# moe_expert_compute (compute)
# ---------------------------------------------------------------------------


def walk_ep_store(data):
    """Yield ``(coord, {num_tokens: leaf})`` for every collected compute slice.

    ``coord`` is ``(kernel_source, quant_name, distribution, inference_phase,
    topk, num_experts, num_slots, hidden_size, inter_size, moe_tp_size,
    moe_ep_size)``.
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


def ep_candidates(db):
    """Per-category ``(coord, x, overrides)`` candidates.

    ``overrides`` carries the op-ctor knobs under test: ``attention_dp_size``,
    ``enable_eplb``, ``num_slots`` (``None`` = exercise the default),
    ``kernel_source`` (``None`` = exercise the auto-resolution) and
    ``scale_factor``.
    """
    MoEExpertCompute.load_data(db)
    slices = sorted(walk_ep_store(db._moe_ep_data), key=lambda item: item[0])

    out: dict[str, list] = {name: [] for name in EP_QUOTAS}
    for coord, curve in slices:
        kernel_source, _quant, _dist, inference_phase = coord[0], coord[1], coord[2], coord[3]
        num_experts, num_slots = coord[5], coord[6]
        tokens = sorted(curve)
        if not tokens:
            continue
        mid = tokens[len(tokens) // 2]

        base = {"attention_dp_size": 1, "enable_eplb": False, "num_slots": num_slots, "kernel_source": kernel_source}

        out["adp1_exact"].append((coord, mid, dict(base)))
        # attention_dp globalizes: x * adp keys the collected point.
        if mid % 2 == 0:
            out["adp2_exact"].append((coord, mid // 2, {**base, "attention_dp_size": 2}))
        if mid % 8 == 0:
            out["adp8_exact"].append((coord, mid // 8, {**base, "attention_dp_size": 8}))

        # EPLB: fires only on the sglang-adapted leg during context. Both
        # legs are sampled so the fixture pins the fire AND the no-fire case;
        # x is chosen so int(x * adp * 0.8) is an interior point.
        eplb_x = max(1, (mid * 5) // 4)
        if inference_phase == "context":
            out["eplb_context"].append((coord, eplb_x, {**base, "enable_eplb": True}))
            # Order pin: globalize FIRST, truncate SECOND. Only an x where the
            # two orders actually DISAGREE discriminates — for most x,
            # ``int(2x * 0.8)`` and ``2 * int(x * 0.8)`` coincide.
            half = max(1, eplb_x // 2)
            order_x = next(
                (c for c in range(half, half + 5) if int(c * 2 * 0.8) != 2 * int(c * 0.8)),
                half,
            )
            out["eplb_context"].append((coord, order_x, {**base, "attention_dp_size": 2, "enable_eplb": True}))
        else:
            out["eplb_generation"].append((coord, eplb_x, {**base, "enable_eplb": True}))

        # kernel_source=None -> _resolve_kernel_source. On sglang/vllm it
        # short-circuits to "deepep_moe"; on trtllm it prefers
        # deepgemm/moe_torch_flow and falls back to the collected kernel.
        out["kernel_auto"].append((coord, mid, {**base, "kernel_source": None}))
        # num_slots=None -> num_experts. Only meaningful (and only correct)
        # where the collected slice HAS num_slots == num_experts.
        if num_slots == num_experts:
            out["slots_default"].append((coord, mid, {**base, "num_slots": None}))
        # Beyond the collected range: the boundary util-hold on the WideEP
        # roofline SOL.
        out["token_overflow"].append((coord, tokens[-1] * 2 + 1, dict(base)))
        out["scale_factor"].append((coord, mid, {**base, "scale_factor": SCALE_FACTOR}))
    return out


def build_ep_samples(db, system, backend, version, data_root):
    out = ep_candidates(db)
    samples = []
    for category, quota in EP_QUOTAS.items():
        pool = out[category]
        if not pool:
            continue
        stride = max(1, len(pool) // quota)
        for coord, x, overrides in pool[::stride][:quota]:
            (
                _kernel_source,
                quant_name,
                distribution,
                inference_phase,
                topk,
                num_experts,
                _num_slots,
                hidden_size,
                inter_size,
                _moe_tp_size,
                moe_ep_size,
            ) = coord
            scale_factor = overrides.get("scale_factor", 1.0)
            op = MoEExpertCompute(
                "oracle",
                scale_factor,
                hidden_size=hidden_size,
                inter_size=inter_size,
                topk=topk,
                num_experts=num_experts,
                moe_ep_size=moe_ep_size,
                quant_mode=MoEQuantMode[quant_name],
                workload_distribution=distribution,
                attention_dp_size=overrides["attention_dp_size"],
                inference_phase=inference_phase,
                num_slots=overrides["num_slots"],
                kernel_source=overrides["kernel_source"],
                is_gated=True,
                enable_eplb=overrides["enable_eplb"],
            )
            result = op.query(db, x=x)
            samples.append(
                {
                    "op": "moe_expert_compute",
                    "kind": category,
                    "system": system,
                    "backend": backend,
                    "version": version,
                    "data_root": data_root,
                    "scale_factor": scale_factor,
                    "hidden_size": int(hidden_size),
                    "inter_size": int(inter_size),
                    "topk": int(topk),
                    "num_experts": int(num_experts),
                    "moe_ep_size": int(moe_ep_size),
                    "quant_mode": quant_name,
                    "workload_distribution": distribution,
                    "attention_dp_size": int(overrides["attention_dp_size"]),
                    "inference_phase": inference_phase,
                    "num_slots": None if overrides["num_slots"] is None else int(overrides["num_slots"]),
                    "kernel_source": overrides["kernel_source"],
                    "is_gated": True,
                    "enable_eplb": bool(overrides["enable_eplb"]),
                    "x": int(x),
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
        samples.extend(build_a2a_samples(db, system, backend, version, data_root))
        samples.extend(build_ep_samples(db, system, backend, version, data_root))

    header = {
        "_regenerate": (".venv/bin/python aic-core/rust/aiconfigurator-core/parity_tests/gen_op_oracle.py"),
        "_source": "MoEAllToAll(...).query / MoEExpertCompute(...).query (shared_layer=False), SILICON mode",
        "_tuples": [f"{s}/{b}/{v}" for s, b, v in TUPLES],
    }
    out_path = os.path.normpath(OUT_PATH)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # One sample per line: this fixture is machine-generated and read as a
    # diff, so a pretty-printed dump is pure noise.
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
