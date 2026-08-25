# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""G4 goal verification: auto-participation (spec section 4.6).

Large EP rides the DEFAULT search -- no flag opts in, and no flag can opt a
covered model out into a different candidate set. The goal has two halves:

- TRANSPOSITION (already pinned, referenced here rather than duplicated):
  ``test_wideep_deprecation.py::test_flag_vs_flagless_agg_trtllm_identical``
  and ``::test_flag_vs_flagless_disagg_trtllm_identical`` assert a Task with
  ``enable_wideep=True`` and the same Task without it produce identical
  candidate ladders, identical ``iter_parallel`` tuples and identical
  per-tuple ``moe_comm_backend`` resolution -- the flag is gate-shaped
  transposition-inert. ``test_coverage_candidates.py`` pins the synthetic
  covered/uncovered split the resolution rule keys on.

- SHIPPED-DATA CONTROLS (this module): on the same shipped h200_sxm/sglang
  tree, flagless DeepSeek-R1 (covered shape) explores large-EP tuples with
  the deepep per-phase backends, while Qwen3-235B (no moe_a2a/moe_ep rows
  for its shape) gets ZERO large-EP tuples and purely fused ladders -- data
  is the only gate that separates the two.

Both tests read the shipped h200_sxm sglang wideEP parquets and skip when
the data is absent (same gate as ``test_moe_block_builder_large_ep.py``).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from aiconfigurator.sdk.operations.base import resolve_op_data_path
from aiconfigurator.sdk.task_v2 import Task

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
SYSTEMS_DATA_ROOT = REPO_ROOT / "aic-core" / "src" / "aiconfigurator_core" / "systems" / "data"

# The legacy-adapted large-EP sources for h200_sxm/sglang (comm + compute).
H200_SGLANG_LARGE_EP_PATHS = [
    resolve_op_data_path(str(SYSTEMS_DATA_ROOT / "h200_sxm"), "sglang", "0.5.6.post2", filename)
    for filename in (
        "wideep_deepep_normal_perf.parquet",
        "wideep_deepep_ll_perf.parquet",
        "wideep_context_moe_perf.parquet",
        "wideep_generation_moe_perf.parquet",
    )
]

h200_large_ep_data_present = pytest.mark.skipif(
    not all(os.path.exists(p) for p in H200_SGLANG_LARGE_EP_PATHS),
    reason="shipped h200_sxm sglang wideEP parquets not present",
)

DSR1 = "deepseek-ai/DeepSeek-R1"
QWEN3 = "Qwen/Qwen3-235B-A22B"


def _h200_sglang_task(model_path: str, **overrides) -> Task:
    kwargs = {
        "serving_mode": "agg",
        "model_path": model_path,
        "system_name": "h200_sxm",
        "backend_name": "sglang",
        "backend_version": "0.5.14",
    }
    kwargs.update(overrides)
    return Task(**kwargs)


@h200_large_ep_data_present
def test_deepseek_r1_flagless_default_search_explores_large_ep():
    """Positive control: the shipped data covers the DeepSeek-R1 shape, so the
    FLAGLESS default search offers large-EP tuples and resolves the deepep
    pair per phase (context -> deepep_ht, generation -> deepep_ll). No
    ``enable_wideep``, no ``moe_backend`` -- participation is automatic."""
    t = _h200_sglang_task(DSR1)

    # Coverage flipped the ladders to the fused-union-multi-node lists.
    assert 64 in t.agg_moe_ep_candidates
    assert 64 in t.agg_num_gpu_candidates

    resolved = [(tup, t._resolve_moe_comm_backend("agg", tup)) for tup in t.iter_parallel("agg")]
    large = [(tup, comm) for tup, comm in resolved if comm]
    fused = [tup for tup, comm in resolved if comm is None]
    assert large, "shipped coverage must yield large-EP candidates flaglessly"
    assert fused, "the same task must keep exploring the fused regime"
    for tup, comm in large:
        assert tup[3] == 1, tup  # moe_tp == 1 by the rule
        assert tup[4] > 1, tup
        assert comm["context"] == "deepep_ht", tup
        assert comm["generation"] == "deepep_ll", tup

    # The per-tuple ModelConfig carries the backend and the node width the
    # large-EP graph construction requires (they are set together).
    tup, comm = large[0]
    mc = t.build_model_config(role="agg", parallel=tup)
    assert mc.moe_comm_backend == comm
    assert mc.num_gpus_per_node == 8


@h200_large_ep_data_present
def test_qwen3_no_coverage_control_stays_fully_fused():
    """Negative control: same system, same backend, same (present) large-EP
    tables -- but no rows for the Qwen3-235B shape. The task must keep the
    fused defaults, resolve NO tuple to a comm backend, and hand every tuple
    a ModelConfig without one (no spurious ``moe_comm_backend``)."""
    t = _h200_sglang_task(QWEN3)

    assert t._large_ep_coverage("agg") == {}
    # Fused defaults, not the unioned multi-node ladders.
    assert t.agg_moe_ep_candidates == [1, 2, 4, 8, 16]
    assert t.agg_num_gpu_candidates == [1, 2, 4, 8, 16]

    tuples = list(t.iter_parallel("agg"))
    assert tuples
    assert all(t._resolve_moe_comm_backend("agg", tup) is None for tup in tuples)

    mc = t.build_model_config(role="agg", parallel=tuples[0])
    assert mc.moe_comm_backend is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
