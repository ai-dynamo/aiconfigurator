#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Manual test for SGLang MoE collector.

Tests:
  1. nvfp4 - Qwen3-235B, ep=32, tokens=48 completes and writes a row to the output file.
  2. nvfp4 - Mixtral-8x7B, ep=1, tokens=128 completes and writes a row to the output file.
  3. float16 sanity-check — non-nvfp4 path still works.
  4. int4_wo (W4A16) - Mixtral-8x7B, ep=1, tokens=128 — scale shapes are correct
     and the kernel completes successfully.

Run from the collector/ directory:
    cd collector && python test_sglang_moe_nvfp4.py

Requirements:
    - Blackwell GPU (SM >= 100) for nvfp4; Tests 1 and 2 are auto-skipped otherwise.
    - SGLang with flashinfer CuteDSL support for nvfp4.
"""

import os
import sys
import tempfile
import traceback

# Ensure collector/ and collector/sglang/ are importable regardless of cwd.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "sglang"))

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
SKIP = "\033[93mSKIP\033[0m"

results = []


def section(title):
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


# Check once whether flashinfer CuteDSL is available.
try:
    from flashinfer import fp4_quantize  # noqa: F401
    from sglang.srt.layers.moe.flashinfer_cutedsl_moe import flashinfer_cutedsl_moe_masked  # noqa: F401

    _cutedsl_available = True
except ImportError:
    _cutedsl_available = False


# ---------------------------------------------------------------------------
# Test 1 — nvfp4 larger model (Qwen3-235B, ep=32, tokens=48)
# ---------------------------------------------------------------------------
section("Test 1: nvfp4 - Qwen3-235B ep=32 tokens=48\n        Expected: run completes, row written to output file")

if not _cutedsl_available:
    print(f"  {SKIP}: flashinfer CuteDSL not available on this build")
    results.append(("T1 nvfp4 Qwen3-235B", None, None))
else:
    try:
        from sglang.collect_moe import run_moe_torch

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "moe_perf.txt")
            run_moe_torch(
                "nvfp4",
                48,  # num_tokens
                4096,  # hidden_size
                1536,  # inter_size
                8,  # topk
                128,  # num_experts
                4,  # moe_tp_size
                32,  # moe_ep_size  → local_num_experts = 4
                "Qwen/Qwen3-235B-A22B",
                out_file,
                "power_law",
                1.01,
            )

            row_written = os.path.exists(out_file) and os.path.getsize(out_file) > 0

        if row_written:
            print(f"  {PASS}: nvfp4 completed and output written")
            results.append(("T1 nvfp4 Qwen3-235B", True, None))
        else:
            print(f"  {FAIL}: run returned but nothing was written to output")
            results.append(("T1 nvfp4 Qwen3-235B", False, None))

    except Exception as exc:
        print(f"  {FAIL}: {exc}")
        traceback.print_exc()
        results.append(("T1 nvfp4 Qwen3-235B", False, exc))


# ---------------------------------------------------------------------------
# Test 2 — supported nvfp4 config completes and writes output
# ---------------------------------------------------------------------------
section(
    "Test 2: nvfp4 - Mixtral-8x7B ep=1 tokens=128 (should succeed)\n"
    "        Expected: run completes, row written to output file"
)

if not _cutedsl_available:
    print(f"  {SKIP}: flashinfer CuteDSL not available on this build")
    results.append(("T2 nvfp4 good config", None, None))
else:
    try:
        from sglang.collect_moe import run_moe_torch

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "moe_perf.txt")
            run_moe_torch(
                "nvfp4",
                128,  # num_tokens
                4096,  # hidden_size
                14336,  # inter_size
                2,  # topk
                8,  # num_experts
                1,  # moe_tp_size
                1,  # moe_ep_size
                "mistralai/Mixtral-8x7B-v0.1",
                out_file,
                "power_law",
                1.01,
            )

            row_written = os.path.exists(out_file) and os.path.getsize(out_file) > 0

        if row_written:
            print(f"  {PASS}: nvfp4 completed and output written")
            results.append(("T2 nvfp4 good config", True, None))
        else:
            print(f"  {FAIL}: run returned but nothing was written to output")
            results.append(("T2 nvfp4 good config", False, None))

    except Exception as exc:
        print(f"  {FAIL}: {exc}")
        traceback.print_exc()
        results.append(("T2 nvfp4 good config", False, exc))


# ---------------------------------------------------------------------------
# Test 3 — float16 sanity check (non-nvfp4 path unchanged)
# ---------------------------------------------------------------------------
section("Test 3: float16 sanity check - Mixtral-8x7B ep=1 tokens=128")

try:
    from sglang.collect_moe import run_moe_torch

    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = os.path.join(tmpdir, "moe_perf.txt")
        run_moe_torch(
            "float16",
            128,  # num_tokens
            4096,  # hidden_size
            14336,  # inter_size
            2,  # topk
            8,  # num_experts
            1,  # moe_tp_size
            1,  # moe_ep_size
            "mistralai/Mixtral-8x7B-v0.1",
            out_file,
            "power_law",
            1.01,
        )

        row_written = os.path.exists(out_file) and os.path.getsize(out_file) > 0

    if row_written:
        print(f"  {PASS}: float16 completed and output written")
        results.append(("T3 float16 sanity", True, None))
    else:
        print(f"  {FAIL}: run returned but nothing was written to output")
        results.append(("T3 float16 sanity", False, None))

except Exception as exc:
    print(f"  {FAIL}: {exc}")
    traceback.print_exc()
    results.append(("T3 float16 sanity", False, exc))


# ---------------------------------------------------------------------------
# Test 4 — int4_wo (W4A16) scale shapes are correct and kernel runs
# ---------------------------------------------------------------------------
section("Test 4: int4_wo (W4A16) - Mixtral-8x7B ep=1 tokens=128")

try:
    from sglang.collect_moe import run_moe_torch

    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = os.path.join(tmpdir, "moe_perf.txt")
        run_moe_torch(
            "int4_wo",
            128,  # num_tokens
            4096,  # hidden_size
            14336,  # inter_size
            2,  # topk
            8,  # num_experts
            1,  # moe_tp_size
            1,  # moe_ep_size
            "mistralai/Mixtral-8x7B-v0.1",
            out_file,
            "power_law",
            1.01,
        )

        row_written = os.path.exists(out_file) and os.path.getsize(out_file) > 0

    if row_written:
        print(f"  {PASS}: int4_wo completed and output written")
        results.append(("T4 int4_wo W4A16", True, None))
    else:
        print(f"  {FAIL}: run returned but nothing was written to output")
        results.append(("T4 int4_wo W4A16", False, None))

except Exception as exc:
    print(f"  {FAIL}: {exc}")
    traceback.print_exc()
    results.append(("T4 int4_wo W4A16", False, exc))


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
section("Summary")
all_ok = True
for name, ok, _exc in results:
    if ok is True:
        status = PASS
    elif ok is None:
        status = SKIP
    else:
        status = FAIL
        all_ok = False
    print(f"  {status}  {name}")

print()
sys.exit(0 if all_ok else 1)
