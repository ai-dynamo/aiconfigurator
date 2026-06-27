# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests pinning the Context-Parallel DSA prefill model.

Guards the subtle CP formulas (``ContextDSAModule._query_cp`` full/cp mqa+topk
deltas, ``_lookup_2d`` grid lookup + fail-loud on out-of-grid isl) against
future drift -- the nsys validation in docs/CONTEXT_PARALLEL_DSA_MODELING.md is
not a CI regression gate.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from aiconfigurator.sdk.operations.dsa import ContextDSAModule

pytestmark = pytest.mark.unit


def test_lookup_2d_exact_and_step_interp():
    db = SimpleNamespace()
    t = {(4096, 0): 100.0, (4096, 1024): 200.0, (8192, 0): 400.0}
    assert ContextDSAModule._lookup_2d(db, "mqa", t, 4096, 0) == 100.0
    assert ContextDSAModule._lookup_2d(db, "mqa", t, 4096, 512) == pytest.approx(150.0)
    assert ContextDSAModule._lookup_2d(db, "mqa", t, 4096, 4096) == 200.0
    assert ContextDSAModule._lookup_2d(db, "mqa", {}, 4096, 0) is None


def test_lookup_2d_fails_loud_on_out_of_grid_isl():
    # isl beyond the collected grid must RAISE, not silently clamp -- mqa is
    # quadratic in isl, so clamping 16384->8192 would halve... quarter it.
    t = {(4096, 0): 100.0, (8192, 0): 400.0}
    with pytest.raises(ValueError, match="exceeds the collected"):
        ContextDSAModule._lookup_2d(SimpleNamespace(), "mqa", t, 16384, 0)


def test_query_cp_composition(monkeypatch):
    cp, isl, prefix = 8, 16384, 0
    per_card = -(-isl // cp)  # ceil = 2048

    # known sparse tables: mqa/topk_last at full isl + per_card, topk_flat at per_card
    tables = {
        "_2d": {
            "mqa": {(isl, 0): 1600.0, (per_card, 0): 25.0},
            "topk_last": {(isl, 0): 800.0, (per_card, 0): 190.0},
            "topk_flat": {(per_card, 0): 100.0},
        }
    }
    monkeypatch.setattr(ContextDSAModule, "_load_glm5_sparse", classmethod(lambda cls, db: tables))

    db = MagicMock()
    db._sparse_surrogate_cache = {}
    db.query_context_dsa_module.return_value = 4300.0  # per-card monolithic base
    db.query_nccl.return_value = 50.0  # each AG

    m = ContextDSAModule.__new__(ContextDSAModule)
    m._cp_size = cp
    m._num_heads = 64
    m._kvcache_quant_mode = None
    m._fmha_quant_mode = None
    m._gemm_quant_mode = None
    m._architecture = "GlmMoeDsaForCausalLM"
    m._scale_factor = 1.0

    res = m._query_cp(db, b=1, isl=isl, prefix=prefix)

    # delta_mqa  = mqa_full/cp - mqa_perc  = 1600/8 - 25  = 175  (full/cp form)
    # delta_topk = tl_full/cp  - tf_perc   = 800/8  - 100 = 0
    # latency    = base 4300 + 175 + 0 + ag_kv 50 + ag_lse 50 = 4575
    assert float(res) == pytest.approx(4575.0)
    assert res.source == "cp_model"

    # AG volumes (4th positional arg of query_nccl): indexer key isl*128,
    # compressed latent isl*(kv_lora 512 + rope 64 = 576). Both bf16.
    ag_sizes = sorted(call.args[3] for call in db.query_nccl.call_args_list)
    assert ag_sizes == sorted([isl * 128, isl * (512 + 64)])


def test_query_cp_raises_when_isl_beyond_grid(monkeypatch):
    # _query_cp must propagate the _lookup_2d fail-loud (no silent under-estimate).
    cp, isl, prefix = 8, 32768, 0
    tables = {
        "_2d": {
            "mqa": {(16384, 0): 1600.0, (4096, 0): 25.0},  # grid caps at 16384
            "topk_last": {(16384, 0): 800.0, (4096, 0): 190.0},
            "topk_flat": {(4096, 0): 100.0},
        }
    }
    monkeypatch.setattr(ContextDSAModule, "_load_glm5_sparse", classmethod(lambda cls, db: tables))
    db = MagicMock()
    db._sparse_surrogate_cache = {}
    db.query_context_dsa_module.return_value = 4300.0
    db.query_nccl.return_value = 50.0

    m = ContextDSAModule.__new__(ContextDSAModule)
    m._cp_size = cp
    m._num_heads = 64
    m._kvcache_quant_mode = None
    m._fmha_quant_mode = None
    m._gemm_quant_mode = None
    m._architecture = "GlmMoeDsaForCausalLM"
    m._scale_factor = 1.0

    with pytest.raises(ValueError, match="exceeds the collected"):
        m._query_cp(db, b=1, isl=isl, prefix=prefix)  # isl=32768 > grid 16384
