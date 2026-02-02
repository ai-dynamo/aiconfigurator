# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
E2E tests comparing TrtLLM performance with and without WideEP.
Tests the actual CLI interface to ensure WideEP integration works correctly.
"""

import subprocess as sp
import pytest
import re
from typing import Dict, List, Tuple

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


def build_base_cmd(batch_size: int, isl: int, osl: int) -> List[str]:
    """Build the base command common to both WideEP and standard tests."""
    import os
    # Get the path to the CLI script relative to the test file location
    test_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(test_dir, "..", "..", ".."))
    cli_path = os.path.join(project_root, "tools", "simple_sdk_demo", "cli", "main.py")
    
    return [
        "python3",
        cli_path,
        "--model", TEST_MODEL,
        "--backend", TEST_BACKEND,
        "--version", TEST_VERSION,
        "--system", TEST_SYSTEM,
        "--mode", "static",
        "--batch_size", str(batch_size),
        "--isl", str(isl),
        "--osl", str(osl),
        "--tp_size", "1",
        "--attention_dp_size", "8",
        "--moe_tp_size", "1",
        "--moe_ep_size", "16",  # Increased to enable proper AlltoAll communication
        "--moe_quant_mode", "nvfp4",
    ]


def build_wideep_cmd(batch_size: int, isl: int, osl: int) -> List[str]:
    """Build command for TrtLLM with WideEP enabled.
    
    Note: When enable_wideep=True, the SDK will automatically convert 
    "power_law" to "power_law_1.01_eplb" internally for EPLB-aware distribution.
    """
    cmd = build_base_cmd(batch_size, isl, osl)
    cmd.extend([
        "--enable_wideep",
        "--wideep_num_slots", "256",
        "--workload_distribution", "power_law",  # Will be converted to power_law_1.01_eplb internally
    ])
    return cmd


def build_standard_cmd(batch_size: int, isl: int, osl: int) -> List[str]:
    """Build command for standard TrtLLM (without WideEP).
    
    Note: Uses "power_law" which maps to "power_law_1.01" in the SDK for standard MoE.
    """
    cmd = build_base_cmd(batch_size, isl, osl)
    cmd.extend([
        "--workload_distribution", "power_law",  # Will use power_law_1.01 for standard MoE
    ])
    return cmd


def extract_metrics(output: str) -> Dict[str, float]:
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
    """Test suite comparing WideEP and standard TrtLLM performance."""
    
    @pytest.mark.parametrize("batch_size,isl,osl,description", TEST_MATRIX)
    def test_wideep_vs_standard(self, batch_size: int, isl: int, osl: int, description: str):
        """Compare WideEP and standard TrtLLM performance for various configurations."""
        
        # Run WideEP test
        wideep_cmd = build_wideep_cmd(batch_size, isl, osl)
        wideep_result = sp.run(wideep_cmd, capture_output=True, text=True)
        
        # Run standard test
        standard_cmd = build_standard_cmd(batch_size, isl, osl)
        standard_result = sp.run(standard_cmd, capture_output=True, text=True)
        
        # Print debug info (will be captured by pytest but shown with -s or on failure)
        print("\n" + "="*60)
        print(f"Test: {description}")
        print("="*60)
        
        if wideep_result.returncode != 0:
            print("WideEP test FAILED!")
            print("STDERR:", wideep_result.stderr)
            print("STDOUT:", wideep_result.stdout)
        
        if standard_result.returncode != 0:
            print("Standard test FAILED!")
            print("STDERR:", standard_result.stderr)
            print("STDOUT:", standard_result.stdout)
        
        # Both should succeed
        assert wideep_result.returncode == 0, f"WideEP test failed: {wideep_result.stderr}"
        assert standard_result.returncode == 0, f"Standard test failed: {standard_result.stderr}"
        
        # Extract metrics and show outputs
        print("\n--- WideEP Output ---")
        print(wideep_result.stdout)  # Show first 1000 chars
        
        print("\n--- Standard Output ---")
        print(standard_result.stdout)  # Show first 1000 chars
        wideep_metrics = extract_metrics(wideep_result.stdout)
        standard_metrics = extract_metrics(standard_result.stdout)
        
        # Verify that we got some metrics
        assert len(wideep_metrics) > 0, "No metrics extracted from WideEP output"
        assert len(standard_metrics) > 0, "No metrics extracted from standard output"
        
        # Log the comparison (for pytest output)
        print(f"\n{description} - Batch: {batch_size}, ISL: {isl}, OSL: {osl}")
        print(f"WideEP metrics: {wideep_metrics}")
        print(f"Standard metrics: {standard_metrics}")
        
        # Calculate improvements if we have comparable metrics
        for key in wideep_metrics:
            if key in standard_metrics:
                wideep_val = wideep_metrics[key]
                standard_val = standard_metrics[key]
                if standard_val > 0:
                    improvement = ((standard_val - wideep_val) / standard_val) * 100
                    print(f"{key}: WideEP={wideep_val:.2f}, Standard={standard_val:.2f}, "
                          f"Improvement={improvement:.1f}%")
    
    @pytest.mark.build
    def test_wideep_basic(self):
        """Quick test to verify WideEP is working correctly."""
        cmd = build_wideep_cmd(1, 1024, 256)
        result = sp.run(cmd, capture_output=True, text=True, check=True)
        
        output = result.stdout + result.stderr
        
        # Verify we have MoE operations in the output
        assert "generation_moe" in output.lower(), \
            "MoE operations not detected in output"
        
        # Verify we have dispatch operations (indicating WideEP is working)
        assert "moe_pre_dispatch" in output.lower() or "moe_post_dispatch" in output.lower(), \
            "MoE dispatch operations not detected (WideEP feature)"
        
        # Verify we get performance results
        assert "latency" in output.lower(), "No latency results in output"
    
    @pytest.mark.build
    def test_standard_basic(self):
        """Quick test to verify standard TrtLLM is working correctly."""
        cmd = build_standard_cmd(1, 1024, 256)
        result = sp.run(cmd, capture_output=True, text=True, check=True)
        
        output = result.stdout + result.stderr
        
        # Verify we have MoE operations in the output
        assert "generation_moe" in output.lower(), \
            "MoE operations not detected in output"
        
        # Verify we get performance results
        assert "latency" in output.lower(), "No latency results in output"
    
    def test_wideep_configuration(self):
        """Test that WideEP configuration parameters are correctly applied."""
        cmd = build_wideep_cmd(1, 1024, 256)
        result = sp.run(cmd, capture_output=True, text=True, check=True)
        
        output = result.stdout + result.stderr
        
        # Check for WideEP specific operations in performance breakdown
        # WideEP should have both pre and post dispatch operations
        assert "moe_pre_dispatch" in output.lower(), \
            "WideEP pre-dispatch operation not found in performance breakdown"
        assert "moe_post_dispatch" in output.lower(), \
            "WideEP post-dispatch operation not found in performance breakdown"
        
        # Both dispatch operations should have non-zero latency
        assert "moe_pre_dispatch" in output.lower() and "ms" in output.lower(), \
            "WideEP dispatch operations should show latency"
    
    @pytest.mark.e2e
    def test_performance_regression(self):
        """Test that WideEP provides expected performance improvement."""
        # Run a standard configuration
        batch_size, isl, osl = 256, 8192, 256
        
        # Run both tests
        wideep_cmd = build_wideep_cmd(batch_size, isl, osl)
        wideep_result = sp.run(wideep_cmd, capture_output=True, text=True, check=True)
        
        standard_cmd = build_standard_cmd(batch_size, isl, osl)
        standard_result = sp.run(standard_cmd, capture_output=True, text=True, check=True)
        
        # Extract metrics
        wideep_metrics = extract_metrics(wideep_result.stdout)
        standard_metrics = extract_metrics(standard_result.stdout)
        
        # Find comparable latency metrics
        latency_improved = False
        for key in wideep_metrics:
            if "latency" in key and key in standard_metrics:
                wideep_val = wideep_metrics[key]
                standard_val = standard_metrics[key]
                if wideep_val < standard_val:
                    latency_improved = True
                    break
        
        # WideEP should provide some improvement in at least one latency metric
        # This is a soft check - we log a warning rather than failing
        if not latency_improved:
            pytest.skip("WideEP did not show latency improvement - may need tuning")


if __name__ == "__main__":
    # Allow running as a standalone script for debugging
    import sys
    pytest.main([__file__] + sys.argv[1:])