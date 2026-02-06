# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
E2E tests comparing TrtLLM performance across different WideEP and EPLB configurations.
Tests the actual CLI interface to ensure WideEP integration works correctly.

Supported modes:
- standard: Standard MoE (no WideEP)
- wideep: WideEP with EPLB off
- wideep_with_eplb: WideEP with EPLB on (num_slots = num_experts)
- wideep_with_eplb_redundant: WideEP with EPLB on + redundant slots (num_slots > num_experts)
"""

import re
import subprocess as sp
from typing import Optional

import pytest

pytestmark = pytest.mark.e2e

# Test configurations
TEST_MODEL = "deepseek-ai/DeepSeek-V3"
TEST_BACKEND = "trtllm"
TEST_VERSION = "1.2.0rc6"
TEST_SYSTEM = "gb200_sxm"

# Test matrix for different configurations
TEST_MATRIX = [
    # (batch_size, isl, osl, description)
    (1, 1024, 256, "small_batch_short_seq"),
    (256, 1024, 256, "large_batch_short_seq"),
    (1, 8192, 256, "small_batch_long_seq"),
    (256, 8192, 256, "large_batch_long_seq"),
]

# WideEP mode configurations
WIDEEP_MODES = [
    # (mode_name, enable_wideep, eplb_mode, num_slots, description)
    ("standard", False, None, None, "Standard MoE (no WideEP)"),
    ("wideep", True, "off", 256, "WideEP with EPLB off"),
    ("wideep_with_eplb", True, "on", 256, "WideEP with EPLB on"),
    ("wideep_with_eplb_num_slots288", True, "on", 288, "WideEP with EPLB redundant (num_slots=288)"),
]


def build_base_cmd(batch_size: int, isl: int, osl: int) -> list[str]:
    """Build the base command common to all tests."""
    import os

    # Get the path to the CLI script relative to the test file location
    test_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(test_dir, "..", "..", ".."))
    cli_path = os.path.join(project_root, "tools", "simple_sdk_demo", "cli", "main.py")

    return [
        "python3",
        cli_path,
        "--model",
        TEST_MODEL,
        "--backend",
        TEST_BACKEND,
        "--version",
        TEST_VERSION,
        "--system",
        TEST_SYSTEM,
        "--mode",
        "static",
        "--batch_size",
        str(batch_size),
        "--isl",
        str(isl),
        "--osl",
        str(osl),
        "--tp_size",
        "1",
        "--attention_dp_size",
        "8",
        "--moe_tp_size",
        "1",
        "--moe_ep_size",
        "8",
        "--moe_quant_mode",
        "nvfp4",
        "--workload_distribution",
        "power_law",
    ]


def build_cmd(
    batch_size: int,
    isl: int,
    osl: int,
    enable_wideep: bool = False,
    eplb_mode: Optional[str] = None,
    num_slots: Optional[int] = None,
) -> list[str]:
    """Build command for TrtLLM with specified configuration.

    Args:
        batch_size: Batch size
        isl: Input sequence length
        osl: Output sequence length
        enable_wideep: Whether to enable WideEP
        eplb_mode: EPLB mode - "off" or "on" (only used when enable_wideep=True)
        num_slots: Number of expert slots (only used when enable_wideep=True)

    Workload distribution mapping:
    - Standard: "power_law" -> "power_law_1.01" (handled by standard MoE)
    - WideEP EPLB off: "power_law" -> "power_law_1.01"
    - WideEP EPLB on: "power_law" -> "power_law_1.01_eplb"
    - WideEP EPLB redundant: "power_law" -> "power_law_1.01_eplb" (with num_slots > num_experts)
    """
    cmd = build_base_cmd(batch_size, isl, osl)

    if enable_wideep:
        cmd.append("--enable_wideep")
        if eplb_mode is not None:
            cmd.extend(["--wideep_eplb_mode", eplb_mode])
        if num_slots is not None:
            cmd.extend(["--wideep_num_slots", str(num_slots)])

    return cmd


# Backward compatible helper functions
def build_standard_cmd(batch_size: int, isl: int, osl: int) -> list[str]:
    """Build command for standard TrtLLM (without WideEP)."""
    return build_cmd(batch_size, isl, osl, enable_wideep=False)


def build_wideep_cmd(batch_size: int, isl: int, osl: int, eplb_mode: str = "off", num_slots: int = 256) -> list[str]:
    """Build command for TrtLLM with WideEP enabled."""
    return build_cmd(batch_size, isl, osl, enable_wideep=True, eplb_mode=eplb_mode, num_slots=num_slots)


def extract_metrics(output: str) -> dict[str, float]:
    """Extract performance metrics from CLI output."""
    metrics = {}

    # Pattern to match latency values (e.g., "latency: 12.34ms" or "12.34 ms")
    latency_patterns = [
        r"latency[:\s]+([0-9.]+)\s*ms",
        r"([0-9.]+)\s*ms.*latency",
        r"MoE.*compute.*:\s*([0-9.]+)\s*ms",
        r"dispatch.*:\s*([0-9.]+)\s*ms",
    ]

    for pattern in latency_patterns:
        matches = re.findall(pattern, output, re.IGNORECASE)
        for i, match in enumerate(matches):
            try:
                key = f"latency_{pattern.split('(')[0].strip()}_{i}"
                metrics[key] = float(match)
            except:
                pass

    # Extract throughput if present
    throughput_pattern = r"throughput[:\s]+([0-9.]+)"
    matches = re.findall(throughput_pattern, output, re.IGNORECASE)
    if matches:
        try:
            metrics["throughput"] = float(matches[0])
        except:
            pass

    return metrics


class TestTrtLLMWideEPComparison:
    """Test suite comparing different WideEP and EPLB configurations."""

    @pytest.mark.build
    @pytest.mark.parametrize("mode_name,enable_wideep,eplb_mode,num_slots,description", WIDEEP_MODES)
    def test_all_modes_basic(
        self, mode_name: str, enable_wideep: bool, eplb_mode: str, num_slots: int, description: str
    ):
        """Test all WideEP/EPLB modes work correctly."""
        cmd = build_cmd(1, 1024, 256, enable_wideep=enable_wideep, eplb_mode=eplb_mode, num_slots=num_slots)
        result = sp.run(cmd, capture_output=True, text=True)

        print(f"\n--- {description} ({mode_name}) ---")
        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        assert result.returncode == 0, f"{description} test failed: {result.stderr}"

        output = result.stdout + result.stderr

        # Verify we have MoE operations in the output
        assert "generation_moe" in output.lower(), f"MoE operations not detected in {mode_name} output"

        # Verify we get performance results
        assert "latency" in output.lower(), f"No latency results in {mode_name} output"

        # WideEP modes should have dispatch operations
        if enable_wideep:
            assert "moe_pre_dispatch" in output.lower() or "moe_post_dispatch" in output.lower(), (
                f"MoE dispatch operations not detected in {mode_name} (WideEP feature)"
            )

    @pytest.mark.parametrize("batch_size,isl,osl,test_desc", TEST_MATRIX)
    def test_compare_all_modes(self, batch_size: int, isl: int, osl: int, test_desc: str):
        """Compare all WideEP/EPLB modes for various configurations."""

        print("\n" + "=" * 80)
        print(f"Test: {test_desc} - Batch: {batch_size}, ISL: {isl}, OSL: {osl}")
        print("=" * 80)

        results = {}
        metrics = {}

        # Run all modes
        for mode_name, enable_wideep, eplb_mode, num_slots, description in WIDEEP_MODES:
            cmd = build_cmd(batch_size, isl, osl, enable_wideep=enable_wideep, eplb_mode=eplb_mode, num_slots=num_slots)
            result = sp.run(cmd, capture_output=True, text=True)
            results[mode_name] = result

            if result.returncode != 0:
                print(f"\n{mode_name} FAILED!")
                print("STDERR:", result.stderr)
                print("STDOUT:", result.stdout)
            else:
                metrics[mode_name] = extract_metrics(result.stdout)
                print(f"\n--- {mode_name}: {description} ---")
                print(f"Metrics: {metrics[mode_name]}")

        # All modes should succeed
        for mode_name, result in results.items():
            assert result.returncode == 0, f"{mode_name} test failed: {result.stderr}"

        # Print comparison table
        print("\n" + "-" * 80)
        print("Performance Comparison:")
        print("-" * 80)

        # Use standard as baseline
        if metrics.get("standard"):
            baseline = metrics["standard"]
            for mode_name, mode_metrics in metrics.items():
                if mode_name == "standard":
                    continue
                print(f"\n{mode_name} vs standard:")
                for key in mode_metrics:
                    if key in baseline and baseline[key] > 0:
                        improvement = ((baseline[key] - mode_metrics[key]) / baseline[key]) * 100
                        print(f"  {key}: {mode_metrics[key]:.2f} (improvement: {improvement:+.1f}%)")

    @pytest.mark.build
    def test_standard_basic(self):
        """Quick test to verify standard TrtLLM is working correctly."""
        cmd = build_standard_cmd(1, 1024, 256)
        result = sp.run(cmd, capture_output=True, text=True)

        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        assert result.returncode == 0, f"Standard test failed: {result.stderr}"

        output = result.stdout + result.stderr
        assert "generation_moe" in output.lower(), "MoE operations not detected in output"
        assert "latency" in output.lower(), "No latency results in output"

    @pytest.mark.build
    def test_wideep_basic(self):
        """Quick test to verify WideEP (EPLB off) is working correctly."""
        cmd = build_wideep_cmd(1, 1024, 256, eplb_mode="off", num_slots=256)
        result = sp.run(cmd, capture_output=True, text=True)

        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        assert result.returncode == 0, f"WideEP test failed: {result.stderr}"

        output = result.stdout + result.stderr
        assert "generation_moe" in output.lower(), "MoE operations not detected in output"
        assert "moe_pre_dispatch" in output.lower() or "moe_post_dispatch" in output.lower(), (
            "MoE dispatch operations not detected (WideEP feature)"
        )
        assert "latency" in output.lower(), "No latency results in output"

    @pytest.mark.build
    def test_wideep_with_eplb_basic(self):
        """Quick test to verify WideEP with EPLB on is working correctly."""
        cmd = build_wideep_cmd(1, 1024, 256, eplb_mode="on", num_slots=256)
        result = sp.run(cmd, capture_output=True, text=True)

        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        assert result.returncode == 0, f"WideEP with EPLB test failed: {result.stderr}"

        output = result.stdout + result.stderr
        assert "generation_moe" in output.lower(), "MoE operations not detected in output"
        assert "latency" in output.lower(), "No latency results in output"

    @pytest.mark.build
    def test_wideep_with_eplb_redundant_basic(self):
        """Quick test to verify WideEP with EPLB redundant (num_slots=288) is working correctly."""
        cmd = build_wideep_cmd(1, 1024, 256, eplb_mode="on", num_slots=288)
        result = sp.run(cmd, capture_output=True, text=True)

        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        assert result.returncode == 0, f"WideEP with EPLB redundant test failed: {result.stderr}"

        output = result.stdout + result.stderr
        assert "generation_moe" in output.lower(), "MoE operations not detected in output"
        assert "latency" in output.lower(), "No latency results in output"

    def test_wideep_configuration(self):
        """Test that WideEP configuration parameters are correctly applied."""
        cmd = build_wideep_cmd(1, 1024, 256, eplb_mode="off", num_slots=256)
        result = sp.run(cmd, capture_output=True, text=True)

        assert result.returncode == 0, f"WideEP configuration test failed: {result.stderr}"

        output = result.stdout + result.stderr

        # Check for WideEP specific operations in performance breakdown
        assert "moe_pre_dispatch" in output.lower(), "WideEP pre-dispatch operation not found in performance breakdown"
        assert "moe_post_dispatch" in output.lower(), (
            "WideEP post-dispatch operation not found in performance breakdown"
        )


if __name__ == "__main__":
    # Allow running as a standalone script for debugging
    import sys

    pytest.main([__file__] + sys.argv[1:])
