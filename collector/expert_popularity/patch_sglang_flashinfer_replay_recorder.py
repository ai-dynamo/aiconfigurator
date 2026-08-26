# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record the routing IDs produced by FlashInfer's fused TRT-LLM MoE kernel.

SGLang's non-routed FlashInfer backend normally keeps top-k selection inside
the fused MoE kernel, so the framework recorder cannot see the selected expert
IDs.  FlashInfer exposes ``routing_replay_out`` specifically for retrieving the
IDs selected by that same kernel.  This fail-closed bridge supplies that output
only while the recorder is active and forwards it to SGLang's recorder.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path

SGLANG_VERSION = "0.5.14"
EXPECTED_SOURCE_SHA256 = "067753d34e2b258939508c98e65b5ac5883217245b78563b0a7e759310b6e3b5"
EXPECTED_RUNNER_SOURCE_SHA256 = "09a54bdf8636ed9f9af3dd946bf61d4b40f06c246ba8baaa077ff3c459ea92ca"

_IMPORT_ORIGINAL = "from sglang.srt.utils.custom_op import register_custom_op\n"
_IMPORT_PATCHED = """from sglang.srt.eplb.expert_distribution import (
    get_global_expert_distribution_recorder,
)
from sglang.srt.utils.custom_op import register_custom_op
"""

_RETURN_ORIGINAL = "    return trtllm_fp8_block_scale_moe(**kwargs)\n"
_RETURN_PATCHED = """    recorder = get_global_expert_distribution_recorder()
    routing_replay_out = None
    if recorder.recording:
        routing_replay_out = torch.empty(
            (hidden_states.shape[0], top_k),
            dtype=torch.int16,
            device=hidden_states.device,
        )
        kwargs["routing_replay_out"] = routing_replay_out

    output = trtllm_fp8_block_scale_moe(**kwargs)
    if routing_replay_out is not None:
        # The producer and recorder operations use the same CUDA stream, so
        # scatter-add observes the IDs emitted by the fused routing kernel.
        recorder.on_select_experts(topk_ids=routing_replay_out)
    return output
"""

_RUNNER_ORIGINAL = """    hidden_states = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output
    if TopKOutputChecker.format_is_bypassed(topk_output):
"""
_RUNNER_PATCHED = """    hidden_states = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output
    # DeepSeek-V4 hash-routed layers already produce and record exact Standard
    # top-k IDs. Keep those layers on the routed kernel while learned-routing
    # layers remain on the fused internal-routing path observed via replay.
    use_routed_topk = use_routed_topk or TopKOutputChecker.format_is_standard(
        topk_output
    )
    if TopKOutputChecker.format_is_bypassed(topk_output):
"""


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def apply_bridge(report_path: Path) -> dict:
    installed_version = importlib.metadata.version("sglang")
    if installed_version != SGLANG_VERSION:
        raise RuntimeError(f"SGLang {installed_version!r} != pinned {SGLANG_VERSION!r}")

    wrapper_spec = importlib.util.find_spec("sglang.srt.layers.moe.flashinfer_trtllm_moe")
    runner_spec = importlib.util.find_spec("sglang.srt.layers.moe.moe_runner.flashinfer_trtllm")
    if wrapper_spec is None or wrapper_spec.origin is None:
        raise RuntimeError("could not locate SGLang FlashInfer TRT-LLM MoE wrapper")
    if runner_spec is None or runner_spec.origin is None:
        raise RuntimeError("could not locate SGLang FlashInfer TRT-LLM MoE runner")
    wrapper_path = Path(wrapper_spec.origin)
    runner_path = Path(runner_spec.origin)
    wrapper_source = wrapper_path.read_bytes()
    runner_source = runner_path.read_bytes()
    wrapper_original_sha256 = _sha256(wrapper_source)
    runner_original_sha256 = _sha256(runner_source)
    if wrapper_original_sha256 != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            f"refusing to patch unexpected {wrapper_path}: {wrapper_original_sha256}; expected {EXPECTED_SOURCE_SHA256}"
        )
    if runner_original_sha256 != EXPECTED_RUNNER_SOURCE_SHA256:
        raise RuntimeError(
            f"refusing to patch unexpected {runner_path}: {runner_original_sha256}; "
            f"expected {EXPECTED_RUNNER_SOURCE_SHA256}"
        )

    wrapper_decoded = wrapper_source.decode("utf-8")
    runner_decoded = runner_source.decode("utf-8")
    if wrapper_decoded.count(_IMPORT_ORIGINAL) != 1:
        raise RuntimeError("expected custom-op import was not uniquely present")
    if wrapper_decoded.count(_RETURN_ORIGINAL) != 1:
        raise RuntimeError("expected fused FP8 MoE return was not uniquely present")
    if runner_decoded.count(_RUNNER_ORIGINAL) != 1:
        raise RuntimeError("expected fused FP8 MoE runner dispatch block was not uniquely present")
    wrapper_patched = wrapper_decoded.replace(_IMPORT_ORIGINAL, _IMPORT_PATCHED, 1)
    wrapper_patched = wrapper_patched.replace(_RETURN_ORIGINAL, _RETURN_PATCHED, 1).encode("utf-8")
    runner_patched = runner_decoded.replace(_RUNNER_ORIGINAL, _RUNNER_PATCHED, 1).encode("utf-8")

    for source_path, patched in ((wrapper_path, wrapper_patched), (runner_path, runner_patched)):
        temporary = source_path.with_suffix(".py.collector-replay-tmp")
        temporary.write_bytes(patched)
        temporary.replace(source_path)

    report = {
        "status": "APPLIED",
        "framework": "sglang",
        "framework_version": installed_version,
        "observation": "flashinfer_fused_moe_routing_replay_out",
        "source_files": {
            "flashinfer_trtllm_moe.py": {
                "path": str(wrapper_path),
                "original_sha256": wrapper_original_sha256,
                "patched_sha256": _sha256(wrapper_patched),
            },
            "moe_runner/flashinfer_trtllm.py": {
                "path": str(runner_path),
                "original_sha256": runner_original_sha256,
                "patched_sha256": _sha256(runner_patched),
            },
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(apply_bridge(args.report), sort_keys=True))


if __name__ == "__main__":
    main()
