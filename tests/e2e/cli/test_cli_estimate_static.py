# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for the static-batching estimate mode and its breakdown report.

These tests cover the new behavior added alongside the
``InferenceSummary.format_detail_report`` rollout:

* ``cli_estimate(mode="static" | "static_ctx" | "static_gen")`` runs and
  produces sane values (positive latency, populated memory dict, summary handle).
* ``EstimateResult.summary.format_detail_report`` accepts both single-section
  and ``all`` detail strings without raising.
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


def test_static_estimate_detail_report_renders():
    """format_detail_report must accept summary / memory / time / all without raising."""
    result = cli_estimate(mode="static", **_common_kwargs())
    summary = result.summary
    assert summary is not None

    # Every individual section should produce non-empty output.
    for section in ("summary", "memory", "time"):
        text = summary.format_detail_report(detail=section)
        assert isinstance(text, str)
        assert text.strip(), f"detail={section!r} produced empty output"

    # "all" should superset the individual sections (at least char-wise).
    full = summary.format_detail_report(detail="all")
    assert "Memory Layout" in full
    assert "Performance Summary" in full

    # Unknown sections should raise a clear ValueError.
    with pytest.raises(ValueError):
        summary.format_detail_report(detail="not_a_section")


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


def test_static_estimate_with_nextn_accept_rates_list():
    """Passing nextn + accept rates as a Python list should not error."""
    kwargs = _common_kwargs()
    kwargs["nextn"] = 1
    kwargs["nextn_accept_rates"] = [0.85, 0.3, 0.0, 0.0, 0.0]
    result = cli_estimate(mode="static", **kwargs)
    assert result.summary is not None


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
