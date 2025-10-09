#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unified E2E Test Runner - CI/CD Entry Point

This script provides a unified interface for all E2E testing modes:
- Smoke tests: Quick validation (30s-8m)  
- Selective tests: Targeted validation (variable)
- Full sweep: Complete validation (3-6h)

Usage:
  python3 e2e_runner.py --mode smoke --level basic
  python3 e2e_runner.py --mode selective --models QWEN3_32B LLAMA3.1_8B
  python3 e2e_runner.py --mode full --parallel 4
"""

import sys
import time
import argparse
from pathlib import Path

# Add the current directory to path to import common_utils
sys.path.insert(0, str(Path(__file__).parent))
from common_utils import (
    check_test_infrastructure,
    run_pytest_command,
    generate_test_filter,
    print_cicd_summary,
    print_cicd_recommendations,
    get_cicd_exit_code
)


class E2ETestRunner:
    """Unified E2E test runner for all testing modes."""
    
    def _filter_kwargs(self, **kwargs):
        """Filter out conflicting parameter keys from kwargs."""
        excluded_keys = {
            'models', 'systems', 'gpu_configs', 'isl_osl_combinations', 
            'tpot_values', 'maxfail', 'continue_on_error', 'level'
        }
        return {k: v for k, v in kwargs.items() if k not in excluded_keys}
    
    def run_test(self, mode, **kwargs):
        """
        Run E2E test based on mode and parameters.
        
        Args:
            mode: 'smoke', 'selective', or 'full'
            **kwargs: Additional parameters for the test
        """
        if mode == "smoke":
            return self._run_smoke_test(**kwargs)
        elif mode == "selective":
            return self._run_selective_test(**kwargs)
        elif mode == "full":
            return self._run_full_test(**kwargs)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def _run_smoke_test(self, level="basic", **kwargs):
        """Run smoke test with specified level."""
        configs = {
            "basic": {
                "models": ["QWEN3_32B"],
                "systems": ["h200_sxm"],
                "gpu_configs": [8],
                "isl_osl_combinations": [(4000, 1000)],
                "tpot_values": [10],
                "maxfail": 1
            },
            "model": {
                "models": ["QWEN3_32B", "LLAMA3.1_8B", "DEEPSEEK_V3"],
                "systems": ["h200_sxm"],
                "gpu_configs": [8],
                "isl_osl_combinations": [(4000, 1000)],
                "tpot_values": [10],
                "maxfail": 3
            },
            "system": {
                "models": ["QWEN3_32B"],
                "systems": ["h100_sxm", "h200_sxm", "b200_sxm", "gb200_sxm"],
                "gpu_configs": [8],
                "isl_osl_combinations": [(4000, 1000)],
                "tpot_values": [10],
                "maxfail": 4
            },
            "comprehensive": {
                "models": ["QWEN3_32B", "LLAMA3.1_8B"],
                "systems": ["h100_sxm", "h200_sxm"],
                "gpu_configs": [8],
                "isl_osl_combinations": [(4000, 1000), (1000, 2)],
                "tpot_values": [10],
                "maxfail": 5
            }
        }
        
        if level not in configs:
            raise ValueError(f"Unknown smoke test level: {level}")
        
        config = configs[level]
        return self._execute_test(
            description=f"Smoke test - {level} level",
            **config,
            **self._filter_kwargs(**kwargs)
        )
    
    def _run_selective_test(self, models=None, systems=None, gpu_configs=None,
                           isl_osl_combinations=None, tpot_values=None, 
                           maxfail=10, **kwargs):
        """Run selective test with custom parameters."""
        
        # Validate at least one filter is provided
        if not any([models, systems, gpu_configs, isl_osl_combinations, tpot_values]):
            print("⚠️ Warning: No filters specified for selective test!")
            print("Use --mode full for comprehensive testing.")
            return False, None
        
        return self._execute_test(
            description="Selective test",
            models=models,
            systems=systems, 
            gpu_configs=gpu_configs,
            isl_osl_combinations=isl_osl_combinations,
            tpot_values=tpot_values,
            maxfail=maxfail,
            **self._filter_kwargs(**kwargs)
        )
    
    def _run_full_test(self, maxfail=50, continue_on_error=True, **kwargs):
        """Run full sweep test with all combinations."""
        return self._execute_test(
            description="Full sweep test - ALL combinations",
            maxfail=0 if continue_on_error else maxfail,
            **self._filter_kwargs(**kwargs)
        )
    
    def _execute_test(self, description, models=None, systems=None, gpu_configs=None,
                     isl_osl_combinations=None, tpot_values=None, maxfail=10,
                     parallel_workers=None, show_warnings=False, verbose=True):
        """Execute the actual pytest command."""
        
        # Generate test filter
        test_filter = generate_test_filter(
            models=models,
            systems=systems,
            gpu_configs=gpu_configs,
            isl_osl_combinations=isl_osl_combinations,
            tpot_values=tpot_values
        )
        
        # Build pytest command
        cmd = [
            "python3", "-m", "pytest",
            "tests/cli/e2e_validation/test_e2e_sweep.py",
            "-v" if verbose else "-q",
            "--tb=auto",
            f"--maxfail={maxfail}"
        ]
        
        # Add test filter if specified
        if test_filter:
            cmd.extend(["-k", test_filter])
        
        # Add parallel execution if specified
        if parallel_workers:
            cmd.extend(["-n", str(parallel_workers)])
            description += f" (parallel: {parallel_workers})"
        
        # Add warning display options if requested
        if show_warnings:
            cmd.extend(["-W", "default", "--tb=long"])
        
        return run_pytest_command(cmd, description)
    
    def estimate_test_duration(self, mode, **kwargs):
        """Estimate test duration based on mode and parameters."""
        if mode == "smoke":
            level = kwargs.get("level", "basic")
            durations = {"basic": "30-60s", "model": "2-3m", "system": "2-3m", "comprehensive": "5-8m"}
            return durations.get(level, "Unknown")
        elif mode == "selective":
            return "Variable (depends on filters)"
        elif mode == "full":
            return "3-6 hours"
        return "Unknown"


def main():
    """Main entry point with unified argument parsing."""
    parser = argparse.ArgumentParser(
        description="Unified E2E Test Runner for CI/CD",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Smoke tests
  python3 e2e_runner.py --mode smoke --level basic
  python3 e2e_runner.py --mode smoke --level comprehensive --parallel 2
  
  # Selective tests  
  python3 e2e_runner.py --mode selective --models QWEN3_32B LLAMA3.1_8B
  python3 e2e_runner.py --mode selective --systems h200_sxm --gpu-configs 8 512
  
  # Full sweep
  python3 e2e_runner.py --mode full --parallel 4 --continue-on-error
        """
    )
    
    # Required mode argument
    parser.add_argument(
        "--mode",
        choices=["smoke", "selective", "full"],
        required=True,
        help="Test mode: smoke (quick), selective (targeted), full (comprehensive)"
    )
    
    # Smoke test specific
    parser.add_argument(
        "--level",
        choices=["basic", "model", "system", "comprehensive"],
        default="basic",
        help="Smoke test level (default: basic)"
    )
    
    # Selective test filters
    parser.add_argument("--models", nargs="+", help="Models to test")
    parser.add_argument("--systems", nargs="+", choices=["h100_sxm", "h200_sxm", "b200_sxm", "gb200_sxm"], help="Systems to test")
    parser.add_argument("--gpu-configs", nargs="+", type=int, choices=[8, 512], help="GPU configurations")
    parser.add_argument("--isl-osl", nargs="+", help="ISL,OSL combinations (format: 4000,1000)")
    parser.add_argument("--tpot", nargs="+", type=int, choices=[10, 100], help="TPOT values")
    
    # Execution options
    parser.add_argument("--parallel", type=int, help="Number of parallel workers")
    parser.add_argument("--maxfail", type=int, default=10, help="Maximum failures before stopping")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue testing after failures")
    
    # Output options
    parser.add_argument("--show-warnings", action="store_true", help="Show pytest warnings")
    parser.add_argument("--quiet", action="store_true", help="Minimize output for CI/CD")
    parser.add_argument("--estimate-only", action="store_true", help="Show time estimate only")
    parser.add_argument("--check-infrastructure", action="store_true", help="Check test infrastructure only")
    
    args = parser.parse_args()
    
    runner = E2ETestRunner()
    
    # Handle special actions
    if args.check_infrastructure:
        success, message, test_count = check_test_infrastructure()
        if not args.quiet:
            print(f"Infrastructure Status: {message}")
            if success:
                print(f"Available test combinations: {test_count}")
        sys.exit(get_cicd_exit_code(success))
    
    if args.estimate_only:
        duration = runner.estimate_test_duration(args.mode, level=args.level)
        print(f"Estimated duration for {args.mode} test: {duration}")
        sys.exit(0)
    
    # Parse ISL/OSL combinations
    isl_osl_combinations = None
    if args.isl_osl:
        try:
            isl_osl_combinations = []
            for combo in args.isl_osl:
                isl, osl = map(int, combo.split(','))
                isl_osl_combinations.append((isl, osl))
        except ValueError:
            print("❌ Error: ISL/OSL format should be 'isl,osl' (e.g., '4000,1000')")
            sys.exit(1)
    
    # Print header
    if not args.quiet:
        print(f"🚀 E2E TEST RUNNER - {args.mode.upper()} MODE")
        print("=" * 80)
        duration = runner.estimate_test_duration(args.mode, level=args.level)
        print(f"Estimated duration: {duration}")
    
    # Check infrastructure
    success, message, test_count = check_test_infrastructure()
    if not success:
        if not args.quiet:
            print(f"❌ Infrastructure check failed: {message}")
        sys.exit(get_cicd_exit_code(False))
    
    # Run test
    start_time = time.time()
    
    try:
        success, result = runner.run_test(
            mode=args.mode,
            level=args.level,
            models=args.models,
            systems=args.systems,
            gpu_configs=args.gpu_configs,
            isl_osl_combinations=isl_osl_combinations,
            tpot_values=args.tpot,
            parallel_workers=args.parallel,
            maxfail=args.maxfail,
            continue_on_error=args.continue_on_error,
            show_warnings=args.show_warnings,
            verbose=not args.quiet
        )
    except ValueError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    execution_time = time.time() - start_time
    
    # Print summary
    if not args.quiet:
        print_cicd_summary(args.mode, success, execution_time=execution_time)
        print_cicd_recommendations(args.mode, success)
    
    sys.exit(get_cicd_exit_code(success))


if __name__ == "__main__":
    main() 