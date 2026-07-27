# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for the static-batching estimate mode and its breakdown report.

These tests cover the new behavior added alongside the
CLI estimate detail report rollout:

* ``cli_estimate(mode="static" | "static_ctx" | "static_gen")`` runs and
  produces sane values (positive latency, populated memory dict, summary handle).
* The new CLI short aliases (``--bs`` / ``--tp`` / ``--pp`` / etc.) parse to the
  same attribute names as the long forms (smoke check on argparse wiring; does
  not require running a database).
"""

import argparse

import pytest

from aiconfigurator.cli.api import EstimateResult, cli_estimate
from aiconfigurator.cli.main import configure_parser as configure_cli_parser

pytestmark = pytest.mark.e2e

# Use the same small dense model already covered by the existing
# test_cli_estimate_vs_default.py to keep the test runtime low.
_MODEL = "QWEN/QWEN3-32B"
_SYSTEM = "h100_sxm"


def _common_kwargs() -> dict:
    return dict(
        model_path=_MODEL,
        system_name=_SYSTEM,
        backend_name="trtllm",
        isl=2048,
        osl=512,
        batch_size=4,
        tp_size=2,
        pp_size=1,
    )


@pytest.mark.parametrize("static_mode", ["static", "static_ctx", "static_gen"])
def test_static_estimate_runs(static_mode):
    """All three static modes should produce a usable EstimateResult."""
    result = cli_estimate(mode=static_mode, **_common_kwargs())

    assert isinstance(result, EstimateResult)
    assert result.mode == static_mode
    assert result.summary is not None, "static modes must expose the InferenceSummary"

    # Memory dict should be populated (weights + activations + kvcache + nccl + others + total).
    memory = result.summary.get_memory()
    assert "total" in memory
    assert memory["total"] > 0

    if static_mode == "static_ctx":
        assert result.ttft >= 0
        # generation latency is zero by construction; tpot may be 0.
    elif static_mode == "static_gen":
        # ctx skipped; ttft is 0 by construction.
        assert result.tpot >= 0
    else:  # full static
        assert result.ttft > 0
        assert result.tpot >= 0


def test_static_estimate_memory_capacity_context():
    """The new capacity/KV-per-seq stash should be populated for static runs."""
    result = cli_estimate(mode="static", **_common_kwargs())
    summary = result.summary
    assert summary is not None
    assert summary.get_mem_capacity_bytes() is not None
    kv_per_seq, seq_len_used = summary.get_kv_per_seq()
    assert kv_per_seq is not None
    assert kv_per_seq > 0
    assert seq_len_used == 2048 + 1 * 512  # isl + beam_width * osl


def test_static_estimate_with_nextn_accepted():
    """nextn + nextn_accepted must actually change the estimate (not be silently
    ignored): MTP trades a slightly costlier verify step for ~(1+accepted)
    tokens per step, so tokens/s/user must improve vs the nextn=0 baseline."""
    kwargs = _common_kwargs()
    baseline = cli_estimate(mode="static", **kwargs)
    kwargs["nextn"] = 1
    kwargs["nextn_accepted"] = 0.85
    result = cli_estimate(mode="static", **kwargs)
    assert result.summary is not None
    assert result.tokens_per_second_per_user > baseline.tokens_per_second_per_user


def test_cli_short_aliases_parse_to_same_dest():
    """Short aliases (--bs/--tp/--pp/--dp/--etp/--ep) must hit the same attributes
    as their long counterparts. This is purely an argparse smoke check and does
    not exercise the database or backend, keeping it cheap to run."""
    parser = argparse.ArgumentParser()
    configure_cli_parser(parser)

    base = [
        "estimate",
        "--model-path",
        "Qwen/Qwen3-32B",
        "--system",
        "h200_sxm",
    ]
    long_form = parser.parse_args(
        base
        + [
            "--batch-size",
            "16",
            "--tp-size",
            "4",
            "--pp-size",
            "2",
            "--attention-dp-size",
            "1",
            "--moe-tp-size",
            "4",
            "--moe-ep-size",
            "1",
        ]
    )
    short_form = parser.parse_args(
        base
        + [
            "--bs",
            "16",
            "--tp",
            "4",
            "--pp",
            "2",
            "--dp",
            "1",
            "--etp",
            "4",
            "--ep",
            "1",
        ]
    )
    for attr in (
        "batch_size",
        "tp_size",
        "pp_size",
        "attention_dp_size",
        "moe_tp_size",
        "moe_ep_size",
    ):
        assert getattr(long_form, attr) == getattr(short_form, attr), (
            f"Short alias for {attr} did not match the long form."
        )


def test_cli_detail_flag_parses():
    """The new --detail flag should round-trip a comma list."""
    parser = argparse.ArgumentParser()
    configure_cli_parser(parser)
    ns = parser.parse_args(
        [
            "estimate",
            "--model-path",
            "Qwen/Qwen3-32B",
            "--system",
            "h200_sxm",
            "--detail",
            "memory,time",
        ]
    )
    assert ns.detail == "memory,time"


def test_cli_static_modes_in_choices():
    """The estimate-mode choices must include the new static variants."""
    parser = argparse.ArgumentParser()
    configure_cli_parser(parser)
    for mode in ("static", "static_ctx", "static_gen"):
        ns = parser.parse_args(
            [
                "estimate",
                "--model-path",
                "Qwen/Qwen3-32B",
                "--system",
                "h200_sxm",
                "--estimate-mode",
                mode,
            ]
        )
        assert ns.estimate_mode == mode


def _flatten_sources(summary):
    """Collect every source tag from both context + generation per-op dicts."""
    src_ctx = summary.get_context_source_dict() or {}
    src_gen = summary.get_generation_source_dict() or {}
    return list(src_ctx.values()) + list(src_gen.values())


def test_static_estimate_source_tag_silicon_default():
    """SILICON database mode with table-covered config should tag ops as 'silicon'.

    Regression guard for the source-tagging fix in perf_database.py: silicon
    table lookups, including interpolation, should keep the 'silicon' tag,
    while formula-derived ops should not.
    """
    result = cli_estimate(database_mode="SILICON", mode="static", **_common_kwargs())
    sources = _flatten_sources(result.summary)
    assert sources, "expected per-op source tags to be populated"
    # At least one op should be a real table hit. (We don't require *all* to be
    # silicon, since a few ops -- p2p, custom_allreduce when tp=1, etc. -- may
    # legitimately be empirical-derived even in SILICON mode.)
    assert any(s == "silicon" for s in sources), (
        f"expected at least one 'silicon' tag in SILICON mode, got: {set(sources)}"
    )


def test_static_estimate_source_tag_empirical_in_empirical_mode():
    """EMPIRICAL database mode should never tag any op as 'silicon'."""
    result = cli_estimate(database_mode="EMPIRICAL", mode="static", **_common_kwargs())
    sources = _flatten_sources(result.summary)
    assert sources, "expected per-op source tags to be populated"
    # Every measurable op should be from the empirical formula path.
    assert all(s != "silicon" for s in sources), (
        f"'silicon' tag leaked in EMPIRICAL mode (sources should all be 'empirical'): {sources}"
    )
    # The bulk of ops should be tagged 'empirical' (a few might be 'sol' if
    # certain operations only have an SOL fallback, but that's still not
    # silicon).
    assert any(s == "empirical" for s in sources), (
        f"expected at least one 'empirical' tag in EMPIRICAL mode, got: {set(sources)}"
    )


def test_agg_estimate_responds_to_common_prefix():
    """Regression: ``--prefix`` is now a common param. With prefix > 0 the
    effective TTFT should differ from prefix=0 because the context phase
    sees fewer real tokens.
    """
    base = cli_estimate(mode="agg", prefix=0, **_common_kwargs())
    with_prefix = cli_estimate(mode="agg", prefix=512, **_common_kwargs())
    # Prefix caching strictly reduces context work, so TTFT should drop.
    assert with_prefix.ttft < base.ttft, (
        f"prefix did not reach the agg path: ttft@prefix=0 was {base.ttft:.3f}, "
        f"ttft@prefix=512 was {with_prefix.ttft:.3f}"
    )


def test_agg_estimate_responds_to_common_nextn():
    """Regression: ``--nextn`` is now a common param. Toggling MTP on must
    visibly change the agg estimate; previously the kwarg was silently
    dropped on the way to ``_run_agg_estimate``.
    """
    base = cli_estimate(mode="agg", nextn=0, **_common_kwargs())
    with_mtp = cli_estimate(
        mode="agg",
        nextn=1,
        nextn_accepted=0.85,
        **_common_kwargs(),
    )
    # Some metric must change — MTP affects activation memory, generation
    # token count, and the inferred per-step latency. We don't assert a
    # direction (the sign depends on speculation acceptance vs. extra cost),
    # only that the parameter actually reached the underlying inference.
    assert (
        base.ttft != with_mtp.ttft or base.tpot != with_mtp.tpot or base.raw.get("memory") != with_mtp.raw.get("memory")
    ), "nextn=1 produced an estimate identical to nextn=0; the kwarg was dropped"


def _disagg_kwargs() -> dict:
    """Common disagg test fixture: small split so the run is cheap."""
    return dict(
        model_path=_MODEL,
        system_name=_SYSTEM,
        backend_name="trtllm",
        mode="disagg",
        isl=2048,
        osl=512,
        tp_size=2,
        pp_size=1,
        prefill_tp_size=2,
        prefill_batch_size=1,
        prefill_num_workers=1,
        decode_tp_size=2,
        decode_batch_size=4,
        decode_num_workers=1,
    )


def test_disagg_estimate_responds_to_common_prefix():
    """Regression: ``--prefix`` must reach the disagg path too.

    Disagg builds two ``RuntimeConfig`` copies (one for prefill, one for
    decode) by deep-copying the shared one; if ``prefix`` weren't plumbed in
    via ``_run_disagg_estimate``, both copies would run at prefix=0 and the
    TTFT wouldn't move when the user passes ``--prefix``.
    """
    base = cli_estimate(prefix=0, **_disagg_kwargs())
    with_prefix = cli_estimate(prefix=512, **_disagg_kwargs())
    # Prefix caching strictly reduces context work, so TTFT should drop on
    # the prefill side. tpot is decode-only and should stay unchanged.
    assert with_prefix.ttft < base.ttft, (
        f"prefix did not reach the disagg prefill path: ttft@prefix=0 was {base.ttft:.3f}, "
        f"ttft@prefix=512 was {with_prefix.ttft:.3f}"
    )


def test_disagg_estimate_responds_to_common_nextn():
    """Regression: ``--nextn`` must apply to BOTH the prefill and decode
    worker ModelConfigs (we call ``_apply_nextn`` on both inside
    ``_run_disagg_estimate``)."""
    base = cli_estimate(nextn=0, **_disagg_kwargs())
    with_mtp = cli_estimate(
        nextn=1,
        nextn_accepted=0.85,
        **_disagg_kwargs(),
    )
    assert (
        base.ttft != with_mtp.ttft
        or base.tpot != with_mtp.tpot
        or base.raw.get("(p)memory") != with_mtp.raw.get("(p)memory")
        or base.raw.get("(d)memory") != with_mtp.raw.get("(d)memory")
    ), "nextn=1 produced a disagg estimate identical to nextn=0; the kwarg was dropped"


def test_static_estimate_source_tag_sol_in_sol_mode():
    """SOL database mode should report SOL-derived ops as sol."""
    result = cli_estimate(database_mode="SOL", mode="static", **_common_kwargs())
    sources = _flatten_sources(result.summary)
    assert sources, "expected per-op source tags to be populated"
    # No op should be tagged 'silicon' in SOL mode.
    assert all(s != "silicon" for s in sources), f"'silicon' tag leaked in SOL mode: {sources}"
    assert any(s == "sol" for s in sources), f"expected at least one 'sol' tag in SOL mode, got: {set(sources)}"
