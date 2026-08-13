# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM Kimi-K3 MegaMoE module collector.

Thin serving-truth wrapper around the DSv4 MegaMoE harness
(``collector/sglang/collect_dsv4_megamoe.py``). The measured boundary is the
same fused kernel (prepared hidden + top-k -> pre-dispatch ->
``deep_gemm.fp8_fp4_mega_moe``); only the serving lane differs. This entry
forces:

- ``--model-config kimi_k3`` (3584 latent, 896 experts; SiTU is selected by
  the lane-aware ``activation_for_lane`` mapping: vLLM passes
  ``activation="situ"`` with ``activation_clamp=None``)
- ``--pre-dispatch vllm`` (vLLM's OWN triton staging kernel,
  ``prepare_megamoe_inputs`` -- see the harness for the pinned serving
  citations; NOT sglang's deep_gemm JIT copy)
- ``--framework VLLM`` so rows land under the vLLM data tree with the same
  label as every other vLLM table

Image pin: the ``megamoe`` family override in
``collector/framework_manifest.yaml`` (vllm/vllm-openai:v0.27.0, amd64+arm64
digests), verified in-container on GB300/SM103 @ vllm 0.27.0.

Run with torchrun, EP = world size, same token/distribution flags as
``collector/sglang/collect_dsv4_megamoe.py``::

    torchrun --nproc-per-node 8 collector/vllm/collect_k3_megamoe.py \
        --system-name gb300 --version 0.27.0 --output-path <staging>
"""

from __future__ import annotations

import sys

FRAMEWORK_LABEL = "VLLM"

_FORCED = {
    "--model-config": "kimi_k3",
    "--pre-dispatch": "vllm",
    "--framework": FRAMEWORK_LABEL,
}


def apply_k3_vllm_defaults(argv: list[str]) -> list[str]:
    """Inject the K3 vLLM lane options; every occurrence of a forced option must agree.

    Handles both `--flag value` and `--flag=value` forms. Missing values,
    conflicting values and duplicate occurrences raise — a later duplicate
    would otherwise silently override the injected default.
    """
    out = list(argv)
    for flag, forced in _FORCED.items():
        seen: list[str] = []
        i = 0
        while i < len(out):
            tok = out[i]
            if tok == flag:
                if i + 1 >= len(out):
                    raise ValueError(f"{flag} requires a value")
                seen.append(out[i + 1])
                i += 2
                continue
            if tok.startswith(flag + "="):
                seen.append(tok.split("=", 1)[1])
            i += 1
        if not seen:
            out = [flag, forced, *out]
        elif all(value == forced for value in seen):
            if len(seen) > 1:
                raise ValueError(f"{flag} passed multiple times: {seen}")
        else:
            raise ValueError(f"vLLM K3 MegaMoE collector requires {flag} {forced}, got {seen}")
    return out


def _shared_harness_main():
    """Import the shared DSv4 MegaMoE harness and return its ``main``.

    Imported lazily: the harness needs torch and the serving image, which the
    dev/test environment does not have.
    """
    try:
        from collector.sglang import collect_dsv4_megamoe
    except ModuleNotFoundError as exc:
        # Only the package path itself may trigger the repo-root fallback —
        # a missing runtime dependency (torch, vllm) inside the harness must
        # stay a loud import error.
        if exc.name != "collector":
            raise
        # Flat script execution (torchrun collector/vllm/collect_k3_megamoe.py)
        # puts only collector/vllm on sys.path; add the repo root.
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from collector.sglang import collect_dsv4_megamoe
    return collect_dsv4_megamoe.main


def main() -> None:
    sys.argv = [sys.argv[0], *apply_k3_vllm_defaults(sys.argv[1:])]
    _shared_harness_main()()


if __name__ == "__main__":
    main()
