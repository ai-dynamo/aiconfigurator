# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay the pre-deletion data-plane baseline over the engine views (PR-6).

``data_plane_baseline.json`` was captured while the Python ``load_*_data``
parsers were still alive (see ``capture_data_plane_baseline.py``): for 7
pinned databases it froze order-sensitive digests of every
``PerfDatabase._<family>_data`` table, the alias-identity map, both support
-matrix paths, and per-op ``get_weights`` for 6 representative model builds.

These tests recompute everything over the engine-backed bindings and demand
bit-for-bit equality — structure, key ORDER (chart legends consume it
positionally), leaf values, empty subtrees, and load-state trichotomy. Like
PR-5's ``test_query_shim_baseline.py``, the JSON is historical evidence:
``--regen`` only means anything on a tree that still has the Python parsers.

The two DEFAULT_PINS cover every table family except the trtllm-wideep and
b60/oneccl slices; set ``AIC_DATA_PLANE_BASELINE_FULL=1`` to replay all 7
pins (adds a few minutes).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from _data_plane_codec import (
    snapshot_database_tables,
    snapshot_support_matrix,
)

pytestmark = pytest.mark.unit

_BASELINE = json.loads((Path(__file__).parent / "data_plane_baseline.json").read_text())

DEFAULT_PINS = ("h200_sxm/trtllm/1.2.0rc5", "b200_sxm/sglang/0.5.16")
_FULL = os.environ.get("AIC_DATA_PLANE_BASELINE_FULL") == "1"
_PINS = tuple(_BASELINE["databases"]) if _FULL else DEFAULT_PINS


@pytest.fixture(scope="module")
def _warmed_db():
    from aiconfigurator_core.sdk import perf_database
    from aiconfigurator_core.sdk.operations.base import warm_all_op_data

    cache: dict[str, object] = {}

    def get(pin_name: str):
        if pin_name not in cache:
            system, backend, version = pin_name.split("/")
            db = perf_database.get_database(system, backend, version)
            warm_all_op_data(db)
            cache[pin_name] = db
        return cache[pin_name]

    return get


@pytest.mark.parametrize("pin_name", _PINS)
def test_table_views_match_pre_deletion_python_loaders(pin_name, _warmed_db):
    """Every table attribute digest + the alias map, bit for bit."""
    entry = _BASELINE["databases"][pin_name]
    got = snapshot_database_tables(_warmed_db(pin_name))
    mismatches = []
    for attr, want in entry["tables"].items():
        have = got["tables"][attr]
        if have != want:
            mismatches.append(
                f"{attr}: state {have.get('state')}/{want.get('state')}, "
                f"leaves {have.get('n_leaves')}/{want.get('n_leaves')}, "
                f"ordered {have.get('ordered_sha256', '')[:12]}/{want.get('ordered_sha256', '')[:12]}, "
                f"sorted-match={have.get('sorted_sha256') == want.get('sorted_sha256')}"
            )
    assert not mismatches, f"{pin_name}: engine views diverge from the Python loaders:\n" + "\n".join(mismatches)
    assert got["aliases"] == entry["aliases"], f"{pin_name}: alias-identity map diverged"


@pytest.mark.parametrize("pin_name", _PINS)
def test_support_matrix_matches_pre_deletion_python_loaders(pin_name, _warmed_db):
    """Both support-matrix paths (lazy resolver AND eager rebuild), including
    the order of every quant-name list."""
    entry = _BASELINE["databases"][pin_name]["support_matrix"]
    got = snapshot_support_matrix(_warmed_db(pin_name))
    assert got["lazy"] == entry["lazy"], f"{pin_name}: lazy support matrix diverged"
    assert got["eager"] == entry["eager"], f"{pin_name}: eager support matrix diverged"


@pytest.mark.parametrize("weights_key", sorted(_BASELINE["weights"]))
def test_op_weights_match_pre_deletion_python_math(weights_key):
    """Per-op get_weights() for the pinned model builds, repr-exact."""
    from aiconfigurator_core.sdk import common, config, models

    snap = _BASELINE["weights"][weights_key]
    pin = snap["pin"]
    cfg = dict(pin["config"])
    cfg["gemm_quant_mode"] = common.GEMMQuantMode[cfg["gemm_quant_mode"]]
    cfg["moe_quant_mode"] = common.MoEQuantMode[cfg["moe_quant_mode"]]
    cfg["kvcache_quant_mode"] = common.KVCacheQuantMode[cfg["kvcache_quant_mode"]]
    cfg["fmha_quant_mode"] = common.FMHAQuantMode[cfg["fmha_quant_mode"]]
    model = models.get_model(pin["hf_id"], config.ModelConfig(**cfg), backend_name=pin["backend"])
    for phase in ("context_ops", "generation_ops"):
        ops = list(getattr(model, phase))
        want = snap[phase]
        assert len(ops) == len(want), f"{weights_key} {phase}: op count {len(ops)} != {len(want)}"
        for op, (cls_name, op_name, weight_repr) in zip(ops, want, strict=True):
            assert type(op).__name__ == cls_name and getattr(op, "_name", "") == op_name
            assert repr(float(op.get_weights())) == weight_repr, (
                f"{weights_key} {phase} {cls_name}({op_name}): {float(op.get_weights())!r} != {weight_repr}"
            )
