# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``_sum_latency`` Python fallback loop over the internal query shim.

``AFDInferenceSession._sum_latency`` first tries the index-addressed op-list FFI
(``_sum_latency_with_rust``); ops OUTSIDE the compiled phase lists fall back
to the per-op loop, which routes each op through the warning-free internal
shim entry (``Operation._engine_query``). Review #1552 (finding 6, round 2)
asked for a committed regression on that in-repo path with a verify-phase
``KDAKernel`` — the speculative Kimi-K3 shape whose ``verify`` token the
first shim version did not map.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from aiconfigurator.sdk.config import RuntimeConfig
from aiconfigurator.sdk.inference_session import AFDInferenceSession
from aiconfigurator.sdk.operations import KDAKernel
from aiconfigurator.sdk.perf_database import get_database

pytestmark = pytest.mark.unit

_SYSTEM, _BACKEND, _VERSION = "h200_sxm", "sglang", "0.5.16"
_LEGACY_VERIFY_MS = 0.00067456  # merge-base KDAKernel("chunk_kda", "verify") @ b=1, s=4096


@pytest.fixture(scope="module")
def database():
    db = get_database(_SYSTEM, _BACKEND, _VERSION)
    if db is None:
        pytest.skip(f"{_SYSTEM}/{_BACKEND}/{_VERSION} data missing")
    return db


def _verify_kda() -> KDAKernel:
    return KDAKernel("context_kda_verify", 1.0, "chunk_kda", "verify", 7168, 24, 128, 24, 128, 4, draft_tokens=3)


def test_sum_latency_fallback_loop_evaluates_verify_kda(database):
    """Force the fallback: the op is NOT in the fake model's compiled phase
    lists, so ``_sum_latency_with_rust`` returns ``None`` (index miss) and the
    Python loop evaluates the op through ``_engine_query``. The value must be
    the merge-base legacy answer."""
    op = _verify_kda()
    fake_session = SimpleNamespace(
        _database=database,
        _sum_latency_with_rust=lambda *a, **k: None,  # the index-miss outcome, forced deterministically
    )
    fake_model = SimpleNamespace(context_ops=[], generation_ops=[], model_name="kimi-k3-fixture")
    total, per_op = AFDInferenceSession._sum_latency(
        fake_session,
        [op],
        batch_size=1,
        seq_len=4096,
        model=fake_model,
        runtime_config=RuntimeConfig(batch_size=1, isl=4096, osl=8),
        is_context=False,
    )
    assert total == pytest.approx(_LEGACY_VERIFY_MS, rel=1e-9)
    assert per_op == {"context_kda_verify": pytest.approx(_LEGACY_VERIFY_MS, rel=1e-9)}


def test_sum_latency_index_miss_takes_python_loop(database):
    """Same path without stubbing the rust helper: the real
    ``_sum_latency_with_rust`` hits the id->index KeyError (op not in the
    phase lists) and returns None, exercising the genuine in-repo fallback."""
    op = _verify_kda()
    fake_model = SimpleNamespace(context_ops=[], generation_ops=[], model_name="kimi-k3-fixture")
    fake_session = SimpleNamespace(_database=database)
    fake_session._sum_latency_with_rust = AFDInferenceSession._sum_latency_with_rust.__get__(fake_session)
    total, _ = AFDInferenceSession._sum_latency(
        fake_session,
        [op],
        batch_size=1,
        seq_len=4096,
        model=fake_model,
        runtime_config=RuntimeConfig(batch_size=1, isl=4096, osl=8),
        is_context=False,
    )
    assert total == pytest.approx(_LEGACY_VERIFY_MS, rel=1e-9)


def test_verify_kda_spec_serialization_keeps_phase_and_draft_tokens(database):
    """The evaluation-context mapping (verify -> generation-like) must NOT
    leak into the wire: the serialized OpSpec keeps phase="verify" and
    draft_tokens."""
    import json

    from aiconfigurator_core.sdk.engine import _PROBE_MODEL_STUB, build_ops_json

    spec = json.loads(
        build_ops_json([_verify_kda()], model=_PROBE_MODEL_STUB, backend=database.backend, database=database)
    )[0]
    (_, payload) = next(iter(spec.items()))
    assert payload["phase"] == "verify"
    assert payload["draft_tokens"] == 3
