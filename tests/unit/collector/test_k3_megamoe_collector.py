# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import sys
from pathlib import Path

import pytest

from collector.vllm.collect_k3_megamoe import apply_k3_vllm_defaults

pytestmark = pytest.mark.unit

_SGLANG_COLLECTOR = Path("collector/sglang/collect_dsv4_megamoe.py")


def test_vllm_k3_defaults_force_serving_labels():
    # "VLLM" matches the label every other vLLM perf table in the data tree uses.
    assert apply_k3_vllm_defaults([]) == [
        "--framework",
        "VLLM",
        "--pre-dispatch",
        "vllm",
        "--model-config",
        "kimi_k3",
    ]


def test_vllm_k3_defaults_reject_wrong_profile():
    with pytest.raises(ValueError, match="kimi_k3"):
        apply_k3_vllm_defaults(["--model-config", "dsv4_pro"])
    with pytest.raises(ValueError, match="pre-dispatch vllm"):
        apply_k3_vllm_defaults(["--pre-dispatch", "sglang_jit"])
    with pytest.raises(ValueError, match="--framework VLLM"):
        apply_k3_vllm_defaults(["--framework", "SGLang"])


def test_vllm_k3_defaults_handle_conflicting_forms():
    # =form is enforced (it used to slip past the separate-token scan)
    with pytest.raises(ValueError, match="kimi_k3"):
        apply_k3_vllm_defaults(["--model-config=dsv4_pro"])
    # missing value must not IndexError
    with pytest.raises(ValueError, match="requires a value"):
        apply_k3_vllm_defaults(["--model-config"])
    # duplicate options reject even when agreeing (no silent last-wins)
    with pytest.raises(ValueError, match="multiple"):
        apply_k3_vllm_defaults(["--framework", "VLLM", "--framework", "VLLM"])
    # =form with the forced value is honored
    out = apply_k3_vllm_defaults(["--framework=VLLM"])
    assert "--framework" not in out
    assert "--model-config" in out


def test_main_delegates_to_the_shared_harness(monkeypatch):
    """The lane is live (since the pinned vllm 0.27.0 image): the wrapper must
    rewrite argv with the forced serving labels and hand off to the shared DSv4
    MegaMoE harness -- no torch/serving imports in the unit-test path."""
    import collector.vllm.collect_k3_megamoe as wrapper

    calls = []
    monkeypatch.setattr(wrapper, "_shared_harness_main", lambda: lambda: calls.append(list(sys.argv)))
    monkeypatch.setattr(sys, "argv", ["collect_k3_megamoe.py", "--version", "0.27.0", "--output-path", "/tmp/x"])
    wrapper.main()
    assert len(calls) == 1
    forwarded = calls[0]
    for needle in ("--model-config", "kimi_k3", "--pre-dispatch", "vllm", "--framework", "VLLM", "--version", "0.27.0"):
        assert needle in forwarded


def test_vllm_pre_dispatch_uses_the_pinned_serving_import():
    """The vLLM lane must call vLLM's OWN staging helper, with a version citation.

    Superseded the earlier fail-loud contract: the import is now verified
    in-container at vllm v0.27.0 (GB300/SM103), so guessing is no longer the
    risk -- drifting off the serving helper is.
    """
    source = _SGLANG_COLLECTOR.read_text()
    tree = ast.parse(source)
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "make_pre_dispatch")
    src = ast.get_source_segment(source, fn)
    assert src is not None
    # Serving truth: vLLM stages with its own triton kernel, not sglang's JIT copy.
    assert "from vllm.models.deepseek_v4.nvidia.ops.prepare_megamoe import prepare_megamoe_inputs" in src
    assert "prepare_megamoe_inputs(" in src
    # Every manual pin needs a re-auditable file:line @ version citation.
    assert "v0.27.0" in src
    assert "vllm/models/kimi_k3/nvidia/model.py:395" in src
    # No unpinned fallback chain on this lane.
    assert "_import_vllm_mega_pre_dispatch" not in src


def test_deep_gemm_resolution_is_lane_aware():
    """The sglang lane uses the top-level deep_gemm package; the vLLM lane must
    resolve vLLM's vendored copy through serving's own entry point -- the 0.27.0
    image has NO site-packages deep_gemm (container-verified)."""
    source = _SGLANG_COLLECTOR.read_text()
    tree = ast.parse(source)
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_resolve_deep_gemm")
    imports = []
    for node in ast.walk(fn):
        if isinstance(node, ast.ImportFrom):
            imports.append(node.module)
        elif isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
    assert "vllm.utils.deep_gemm" in imports
    assert "deep_gemm" in imports
    # vLLM vendored alias must not be imported directly -- the serving resolver
    # chooses between vendored and site-packages builds.
    assert not any(mod and mod.startswith("vllm.third_party") for mod in imports)


def test_kernel_call_and_weight_transform_use_the_lane_module():
    """fp8_fp4_mega_moe and transform_weights_for_mega_moe must be called on the
    lane-resolved module (dg.*), never on a bare top-level deep_gemm import --
    otherwise the vLLM lane measures the wrong package."""
    source = _SGLANG_COLLECTOR.read_text()
    assert "        dg = _resolve_deep_gemm(args.pre_dispatch)" not in source  # module scope only
    assert "dg.fp8_fp4_mega_moe(" in source
    assert "dg.transform_weights_for_mega_moe(" in source
