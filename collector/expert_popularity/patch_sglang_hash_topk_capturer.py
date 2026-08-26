# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bridge DeepSeek-V4 HashTopK into SGLang's routed-experts capturer.

SGLang 0.5.14 already captures logical IDs from its standard TopK path, but
HashTopK only forwards them to the expert-distribution recorder.  This pinned,
fail-closed patch sends the IDs produced by HashTopK to the same response
capturer before logical-to-physical placement mapping.  It observes the tensor
that serving subsequently dispatches; it does not recompute routing.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path

SGLANG_VERSION = "0.5.14"
EXPECTED_SOURCE_SHA256 = "c179b50aca0309c4dfeedf089037dc4089b36fc4131fdda77f54b2a672cc55df"

_IMPORT_ORIGINAL = "from sglang.srt.utils import is_hip, is_npu\n"
_IMPORT_PATCHED = """from sglang.srt.state_capturer.routed_experts import get_global_experts_capturer
from sglang.srt.utils import is_hip, is_npu
"""

_CAPTURE_ORIGINAL = """        topk_ids = topk_ids_logical_to_physical(
            topk_ids, expert_location_dispatch_info, log2phy_prob
        )
"""
_CAPTURE_PATCHED = """        if (capturer := get_global_experts_capturer()) is not None:
            if self.layer_id is None:
                raise RuntimeError("HashTopK routed-experts capture requires layer_id")
            capturer.capture(layer_id=self.layer_id, topk_indices=topk_ids)
        topk_ids = topk_ids_logical_to_physical(
            topk_ids, expert_location_dispatch_info, log2phy_prob
        )
"""


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def apply_bridge(report_path: Path) -> dict:
    installed_version = importlib.metadata.version("sglang")
    if installed_version != SGLANG_VERSION:
        raise RuntimeError(f"SGLang {installed_version!r} != pinned {SGLANG_VERSION!r}")

    spec = importlib.util.find_spec("sglang.srt.layers.moe.hash_topk")
    if spec is None or spec.origin is None:
        raise RuntimeError("could not locate SGLang HashTopK source")
    source_path = Path(spec.origin)
    source = source_path.read_bytes()
    original_sha256 = _sha256(source)
    if original_sha256 != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            f"refusing to patch unexpected {source_path}: {original_sha256}; expected {EXPECTED_SOURCE_SHA256}"
        )

    decoded = source.decode("utf-8")
    if decoded.count(_IMPORT_ORIGINAL) != 1:
        raise RuntimeError("expected HashTopK utils import was not uniquely present")
    if decoded.count(_CAPTURE_ORIGINAL) != 1:
        raise RuntimeError("expected HashTopK placement block was not uniquely present")
    patched = decoded.replace(_IMPORT_ORIGINAL, _IMPORT_PATCHED, 1)
    patched = patched.replace(_CAPTURE_ORIGINAL, _CAPTURE_PATCHED, 1).encode("utf-8")

    temporary = source_path.with_suffix(".py.collector-routed-experts-tmp")
    temporary.write_bytes(patched)
    temporary.replace(source_path)

    report = {
        "status": "APPLIED",
        "framework": "sglang",
        "framework_version": installed_version,
        "observation": "hash_topk_logical_ids_to_response_routed_experts_capturer",
        "source_file": {
            "path": str(source_path),
            "original_sha256": original_sha256,
            "patched_sha256": _sha256(patched),
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
