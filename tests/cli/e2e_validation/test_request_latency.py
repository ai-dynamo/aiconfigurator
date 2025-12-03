# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Smoke tests for the request-latency CLI flow documented in docs/cli_user_guide.md.

We cover multiple backends and both "with explicit TTFT" and "derive TTFT automatically"
scenarios to ensure the request-latency mode remains stable.
"""

import subprocess as sp

import pytest

REQUEST_LATENCY = 12_000
DOC_MODEL = "QWEN3_32B"
DOC_SYSTEM = "h200_sxm"
DOC_TOTAL_GPUS = 16
DOC_ISL = 4000
DOC_OSL = 500

BACKEND_MATRIX = [
    # (backend, include_ttft)
    ("trtllm", True),
    ("sglang", False),
    ("vllm", False),
]


def _build_cmd(backend: str, include_ttft: bool) -> list[str]:
    cmd = [
        "aiconfigurator",
        "cli",
        "default",
        "--model",
        DOC_MODEL,
        "--total_gpus",
        str(DOC_TOTAL_GPUS),
        "--system",
        DOC_SYSTEM,
        "--backend",
        backend,
        "--request_latency",
        str(REQUEST_LATENCY),
        "--isl",
        str(DOC_ISL),
        "--osl",
        str(DOC_OSL),
    ]
    if include_ttft:
        cmd.extend(["--ttft", "4000"])
    return cmd


class TestRequestLatency:
    """Test that request-latency CLI flows work for a matrix of backends/options."""

    @pytest.mark.e2e
    @pytest.mark.parametrize("backend,include_ttft", BACKEND_MATRIX)
    def test_request_latency(self, backend: str, include_ttft: bool):
        """Run the request-latency command across different backends and ttft settings."""
        completed = sp.run(_build_cmd(backend, include_ttft), capture_output=True, text=True, check=True)
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        combined_output = f"{stdout}\n{stderr}"

        assert "Request Latency" in combined_output, "request latency summary missing"
        if include_ttft:
            assert "TTFT:" in combined_output, "explicit TTFT should appear in summary"
