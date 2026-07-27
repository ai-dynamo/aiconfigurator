# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration: ``aiconfigurator.sdk.memory`` capacity estimate over a REAL perf DB.

Exercises the NATIVE path end-to-end (the full ``_get_memory_usage`` backend
memory model + the OfFree/OfTotal budget math), which the unit tests in
``tests/unit/sdk/test_memory_estimation.py`` only cover with a synthetic
breakdown. This is NOT a Python-vs-Rust parity test: the Rust
``estimate_kv_cache`` is a pure forwarder into this same Python code, so there is
no independent Rust implementation to compare against -- that round-trip is
covered separately by ``aic-core/rust/aiconfigurator-core/tests/memory_round_trip.rs``.

Native cases (Qwen3-32B on h200_sxm, TRT-LLM OfFree and vLLM OfTotal), asserting:

- the returned dict carries the expected fields and a ``source`` string;
- ``tolerance_adjusted`` appears iff ``tolerance_fraction`` is passed, with bytes
  == ``floor(raw * (1 - t))`` at ``t = 0.05``;
- ``estimate_num_gpu_blocks`` returns ``floor(total_kv_size_tokens /
  scheduler_block_size)``, using the tolerance-adjusted token count when set.

Requires the perf DB (LFS) for the SystemSpec capacity used by the native path,
plus ``aiconfigurator_core`` importable (transitively, via ``sdk.memory``).
Soft-skips when the import or the native breakdown is unavailable (e.g. no
``git lfs pull``) so bare runs stay green.
"""

from __future__ import annotations

import math

import pytest

pytestmark = pytest.mark.integration

# `sdk.memory` (transitively) needs the compiled `aiconfigurator_core` extension,
# so it must be importable; skip rather than error when it is not built.
memory = pytest.importorskip("aiconfigurator.sdk.memory")


# (model, system, backend, backend_version, memory_fraction_kind) — the
# native cases. h200_sxm perf DB supplies the SystemSpec capacity.
NATIVE_CASES = [
    ("Qwen/Qwen3-32B", "h200_sxm", "trtllm", "1.3.0rc10", "of_free"),
    ("Qwen/Qwen3-32B", "h200_sxm", "vllm", "0.19.0", "of_total"),
]

_FRACTION = 0.9
_BLOCK_SIZE = 64
_TOLERANCE = 0.05


def _skip_if_fixture_unavailable(exc: ValueError) -> None:
    """Skip ONLY when the native build could not run because the fixture is absent.

    With ``allow_naive_fallback=False`` a missing perf DB / model surfaces as the
    ``"unsupported model/backend/GPU"`` ValueError. Any OTHER ValueError (budget
    math, dict shape, a real native-path regression) is re-raised so it fails the
    test instead of hiding behind an unconditional skip.
    """
    if "unsupported model/backend/GPU" in str(exc):
        pytest.skip(f"native breakdown unavailable (perf DB / model?): {exc}")
    raise exc


def _estimate(case, *, tolerance_fraction=None):
    model, system, backend, version, kind = case
    return memory.estimate_kv_cache(
        model,
        system,
        backend,
        backend_version=version,
        max_num_tokens=8192,
        max_batch_size=256,
        memory_fraction_kind=kind,
        memory_fraction_value=_FRACTION,
        tp_size=1,
        pp_size=1,
        attention_dp_size=1,
        tolerance_fraction=tolerance_fraction,
        allow_naive_fallback=False,
        allow_hf_config_download=False,
    )


def test_estimate_kv_cache_validates_before_breakdown():
    """Fraction + tolerance validation raise ValueError without touching the perf DB.

    Both cases fail in the up-front validation (no model / perf DB needed), so
    this runs deterministically in a bare environment:
      - incompatible memory fraction (TRT-LLM rejects ``of_total``);
      - out-of-range tolerance (``t = 1.5``).
    """
    with pytest.raises(ValueError, match="incompatible memory fraction"):
        memory.estimate_kv_cache(
            "Qwen/Qwen3-32B",
            "h200_sxm",
            "trtllm",
            max_num_tokens=8192,
            max_batch_size=256,
            memory_fraction_kind="of_total",  # TRT-LLM requires of_free.
            memory_fraction_value=0.9,
        )

    with pytest.raises(ValueError, match="tolerance_fraction"):
        memory.estimate_kv_cache(
            "Qwen/Qwen3-32B",
            "h200_sxm",
            "trtllm",
            max_num_tokens=8192,
            max_batch_size=256,
            memory_fraction_kind="of_free",
            memory_fraction_value=0.9,
            tolerance_fraction=1.5,  # out of [0, 1).
        )


@pytest.mark.parametrize("case", NATIVE_CASES, ids=lambda c: f"{c[2]}-{c[4]}")
def test_estimate_kv_cache_native_dict_shape(case):
    """The dict has the expected fields; tolerance_adjusted is None without a tolerance."""
    try:
        est = _estimate(case)
    except ValueError as exc:
        _skip_if_fixture_unavailable(exc)

    # Required scalar fields.
    for key in (
        "total_gpu_capacity_bytes",
        "total_kv_size_bytes",
        "kv_size_per_token_bytes",
        "total_kv_size_tokens",
    ):
        assert key in est and isinstance(est[key], int) and est[key] > 0

    assert est["source"] == "native"
    # Native populates the breakdown; the four components are present.
    mb = est["memory_breakdown"]
    assert mb is not None
    for key in (
        "weights_bytes",
        "activations_bytes",
        "runtime_overhead_bytes",
        "comm_overhead_bytes",
    ):
        assert key in mb

    # No tolerance passed -> tolerance_adjusted absent.
    assert est["tolerance_adjusted"] is None


@pytest.mark.parametrize("case", NATIVE_CASES, ids=lambda c: f"{c[2]}-{c[4]}")
def test_estimate_kv_cache_tolerance_applied(case):
    """tolerance_fraction=0.05 -> tolerance_adjusted set, bytes == floor(raw * 0.95)."""
    try:
        raw = _estimate(case)
        adj_est = _estimate(case, tolerance_fraction=_TOLERANCE)
    except ValueError as exc:
        _skip_if_fixture_unavailable(exc)

    adjusted = adj_est["tolerance_adjusted"]
    assert adjusted is not None
    assert adjusted["tolerance_fraction"] == pytest.approx(_TOLERANCE)

    # adj bytes = floor(raw bytes * 0.95); tokens recomputed by floor-divide.
    expected_bytes = int(raw["total_kv_size_bytes"] * (1.0 - _TOLERANCE))
    assert adjusted["total_kv_size_bytes"] == expected_bytes
    assert adjusted["total_kv_size_tokens"] == expected_bytes // raw["kv_size_per_token_bytes"]
    # The raw fields are unchanged between the two calls.
    assert adj_est["total_kv_size_bytes"] == raw["total_kv_size_bytes"]


@pytest.mark.parametrize("case", NATIVE_CASES, ids=lambda c: f"{c[2]}-{c[4]}")
def test_estimate_num_gpu_blocks_floor_conversion(case):
    """The AIC helper returns floor(tokens / block_size); tolerance picks adjusted tokens."""
    model, system, backend, version, kind = case
    common = dict(
        backend_version=version,
        scheduler_block_size=_BLOCK_SIZE,
        max_num_tokens=8192,
        max_batch_size=256,
        memory_fraction_kind=kind,
        memory_fraction_value=_FRACTION,
        tp_size=1,
        pp_size=1,
        attention_dp_size=1,
    )

    try:
        raw_est = _estimate(case)
        blocks_raw = memory.estimate_num_gpu_blocks(model, system, backend, **common)
        blocks_tol = memory.estimate_num_gpu_blocks(model, system, backend, tolerance_fraction=_TOLERANCE, **common)
    except ValueError as exc:
        _skip_if_fixture_unavailable(exc)

    # Raw path: floor(raw tokens / block_size).
    assert blocks_raw == raw_est["total_kv_size_tokens"] // _BLOCK_SIZE

    # Tolerance path: floor(adjusted tokens / block_size), which is <= raw blocks.
    adj_est = _estimate(case, tolerance_fraction=_TOLERANCE)
    adj_tokens = adj_est["tolerance_adjusted"]["total_kv_size_tokens"]
    assert blocks_tol == adj_tokens // _BLOCK_SIZE
    assert blocks_tol <= blocks_raw
    assert math.floor(adj_tokens / _BLOCK_SIZE) == blocks_tol


# Regression: AIC-1110 — Kimi-K2.5 + GB200 DEP4 prefill worker with Eagle (nextn).
# A prefill/context worker never verifies nextn+1 tokens, so the MTP draft only adds
# its own ~nextn draft-layer activation (a few %), NOT a full (nextn+1)x of the whole
# context activation. The blanket `activations *= (nextn+1)` in _get_memory_usage
# inflated prefill non-KV memory past GPU capacity, so estimate_kv_cache raised
# "no KV budget" once the draft length grew — the worker could not start.
_EAGLE_PREFILL_CASE = dict(
    model_path="moonshotai/Kimi-K2.5",
    system="gb200",
    backend="trtllm",
    backend_version="1.3.0rc10",
    # A realistic long-context prefill chunk (the reporter's agentic workload runs up
    # to ~200k ISL); the over-count scales with the chunk, so it surfaces here at a
    # modest draft length rather than only at nextn>=4 with an 8k chunk.
    max_num_tokens=16384,
    max_batch_size=1,
    memory_fraction_kind="of_free",
    memory_fraction_value=0.9,
    tp_size=1,
    pp_size=1,
    attention_dp_size=4,
    moe_tp_size=1,
    moe_ep_size=4,
    gemm_quant_mode="nvfp4",
    moe_quant_mode="nvfp4",
    kvcache_quant_mode="fp8",
)


@pytest.mark.parametrize("nextn", [1, 2, 3])
def test_eagle_prefill_kv_budget_nonnegative_kimi_gb200(nextn):
    """AIC-1110: a feasible Eagle draft length must not drive the prefill KV budget negative.

    Also pins the post-fix invariant: because ``max_num_tokens`` already bounds the
    per-forward tokens (draft tokens included), the capacity breakdown is independent
    of ``nextn`` -- the activation must match the ``nextn=0`` baseline exactly, not be
    scaled by ``(nextn+1)`` as the latency sweep does.
    """
    try:
        baseline = memory.estimate_kv_cache(**_EAGLE_PREFILL_CASE, nextn=0)
        est = memory.estimate_kv_cache(**_EAGLE_PREFILL_CASE, nextn=nextn)
    except ValueError as exc:
        # A genuinely-missing perf DB / model soft-skips; the "no KV budget" regression
        # (any other ValueError) is re-raised by the helper and fails the test.
        _skip_if_fixture_unavailable(exc)

    # Non-negative budget for any feasible draft length (the reported failure).
    assert est["total_kv_size_bytes"] > 0
    assert est["total_kv_size_tokens"] > 0

    # nextn-independence: activation (and hence the whole non-KV breakdown and budget)
    # must not grow with the draft length on the capacity path.
    assert est["memory_breakdown"]["activations_bytes"] == baseline["memory_breakdown"]["activations_bytes"]
    assert est["total_kv_size_bytes"] == baseline["total_kv_size_bytes"]
