#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Manual test for vLLM nvfp4 MoE collector fix.

Covers two things:
  1. Signature probes (_scaled_fp4_quant_accepts_swizzled,
     _trtllm_moe_accepts_tile_tokens_dim) are correctly set at import time.
  2. nvfp4 run on Mixtral-8x7B dims — the exact task that raised:
       TypeError: scaled_fp4_quant() got an unexpected keyword argument 'is_sf_swizzled_layout'
       TypeError: trtllm_fp4_block_scale_routed_moe() got an unexpected keyword argument 'tile_tokens_dim'
  3. float16 sanity-check so we know the non-nvfp4 path still works.

The collector module is selected by vLLM version to match the production registry:
  vLLM >= 0.17.0 → collect_moe_v2
  vLLM <  0.17.0 → collect_moe_v1

Run from the collector/ directory:
    cd collector && python test_vllm_moe_nvfp4.py

Requirements:
    - Blackwell GPU (SM >= 100) for nvfp4; Test 2 is auto-skipped otherwise.
    - vLLM >= 0.14.0 with flashinfer for nvfp4 support.
"""

import os
import sys
import tempfile
import traceback

# Ensure collector/ and collector/vllm/ are importable regardless of cwd.
# collect_moe_v* uses `from collector.common_test_cases import ...`, so the
# *parent* of collector/ must also be on sys.path.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))  # parent of collector/
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "vllm"))

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
SKIP = "\033[93mSKIP\033[0m"

results = []


def section(title):
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


# Select the right collector module based on vLLM version, matching the registry:
#   vLLM >= 0.17.0 → collect_moe_v2  (int4_wo, set_forward_context, ...)
#   vLLM <  0.17.0 → collect_moe_v1
try:
    from vllm.version import __version__ as _vllm_version

    _moe_module_name = "collect_moe_v2" if _vllm_version >= "0.17.0" else "collect_moe_v1"
except Exception:
    _moe_module_name = "collect_moe_v1"

print(f"vLLM version: {_vllm_version!r}  →  using {_moe_module_name}")

# ---------------------------------------------------------------------------
# Test 1 — signature probe flags
# ---------------------------------------------------------------------------
section("Test 1: signature probe flags at import time")

try:
    import importlib

    _moe_mod = importlib.import_module(_moe_module_name)
    _nvfp4_available = _moe_mod._nvfp4_available
    _scaled_fp4_quant_accepts_swizzled = _moe_mod._scaled_fp4_quant_accepts_swizzled

    print(f"  _nvfp4_available                   = {_nvfp4_available}")
    print(f"  _scaled_fp4_quant_accepts_swizzled = {_scaled_fp4_quant_accepts_swizzled}")
    assert isinstance(_scaled_fp4_quant_accepts_swizzled, bool), "expected bool"

    # v2 also probes tile_tokens_dim support
    if hasattr(_moe_mod, "_trtllm_moe_accepts_tile_tokens_dim"):
        _tile_dim_flag = _moe_mod._trtllm_moe_accepts_tile_tokens_dim
        print(f"  _trtllm_moe_accepts_tile_tokens_dim = {_tile_dim_flag}")
        assert isinstance(_tile_dim_flag, bool), "expected bool"

    print(f"  {PASS}")
    results.append(("T1 probe flags", True, None))
except Exception as exc:
    print(f"  {FAIL}: {exc}")
    traceback.print_exc()
    results.append(("T1 probe flags", False, exc))
    # Without the import we cannot proceed further.
    _nvfp4_available = False
    _moe_mod = None

# ---------------------------------------------------------------------------
# Test 2 — nvfp4 on Mixtral-8x7B (original failing task)
# ---------------------------------------------------------------------------
section("Test 2: nvfp4 - Mixtral-8x7B dims (original failing task)")
# Use a short token list so the run finishes in seconds.
TOKENS_SHORT = [1, 8, 128]

if not _nvfp4_available:
    print(f"  {SKIP}: nvfp4 not available on this GPU/build")
    results.append(("T2 nvfp4 Mixtral-8x7B", None, None))
else:
    try:
        run_moe_torch = _moe_mod.run_moe_torch

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "moe_perf.txt")
            run_moe_torch(
                "nvfp4",
                TOKENS_SHORT,
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

        print(f"  {PASS}: nvfp4 completed for tokens={TOKENS_SHORT}")
        results.append(("T2 nvfp4 Mixtral-8x7B", True, None))

    except TypeError as exc:
        if "is_sf_swizzled_layout" in str(exc):
            print(f"  {FAIL}: is_sf_swizzled_layout fix did NOT take effect: {exc}")
        elif "tile_tokens_dim" in str(exc):
            print(f"  {FAIL}: tile_tokens_dim fix did NOT take effect: {exc}")
        else:
            print(f"  {FAIL}: unexpected TypeError: {exc}")
            traceback.print_exc()
        results.append(("T2 nvfp4 Mixtral-8x7B", False, exc))

    except Exception as exc:
        print(f"  {FAIL}: {exc}")
        traceback.print_exc()
        results.append(("T2 nvfp4 Mixtral-8x7B", False, exc))

# ---------------------------------------------------------------------------
# Test 3 — float16 sanity check (non-nvfp4 path unchanged)
# ---------------------------------------------------------------------------
section("Test 3: float16 sanity check - Mixtral-8x7B")

try:
    run_moe_torch = _moe_mod.run_moe_torch

    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = os.path.join(tmpdir, "moe_perf.txt")
        run_moe_torch(
            "float16",
            [1, 8, 128],
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

    print(f"  {PASS}: float16 completed")
    results.append(("T3 float16 sanity", True, None))

except Exception as exc:
    print(f"  {FAIL}: {exc}")
    traceback.print_exc()
    results.append(("T3 float16 sanity", False, exc))

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
