#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Script to iterate over all model/system/backend/version combinations for complete support matrix generation

Usage:
    --output <output_file.csv> Save results to a CSV file
"""

import argparse
import logging
import os
import sys

# Ensure local repo paths are importable when running as a standalone script.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
sys.path.insert(0, _REPO_ROOT)

from tools.support_matrix.support_matrix import (
    DEFAULT_ENGINE_STEP_COMPARISON_ATOL,
    DEFAULT_ENGINE_STEP_COMPARISON_RTOL,
    DEFAULT_ENGINE_STEP_FRONTIER_ATOL,
    DEFAULT_ENGINE_STEP_FRONTIER_RTOL,
    SupportMatrix,
)


def main():
    # Default output location: <package>/systems/support_matrix.csv
    default_output = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "src",
        "aiconfigurator",
        "systems",
        "support_matrix.csv",
    )

    parser = argparse.ArgumentParser(
        description="Test AIConfigurator support matrix across all model/system/backend combinations"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=default_output,
        help=f"Output file to save results (CSV format) (default: {default_output})",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of processes for parallel execution (default: auto)",
    )
    parser.add_argument(
        "--compare-engine-step-backends",
        action="store_true",
        default=False,
        help="Run both Python and Rust engine-step backends and fail rows whose Pareto outputs drift.",
    )
    parser.add_argument(
        "--engine-step-comparison-rtol",
        type=float,
        default=DEFAULT_ENGINE_STEP_COMPARISON_RTOL,
        help="Relative tolerance for Python-vs-Rust Pareto metric comparison.",
    )
    parser.add_argument(
        "--engine-step-comparison-atol",
        type=float,
        default=DEFAULT_ENGINE_STEP_COMPARISON_ATOL,
        help="Absolute tolerance for Python-vs-Rust Pareto metric comparison.",
    )
    parser.add_argument(
        "--engine-step-frontier-rtol",
        type=float,
        default=DEFAULT_ENGINE_STEP_FRONTIER_RTOL,
        help="Loose relative tolerance when Python and Rust Pareto frontiers choose different rows.",
    )
    parser.add_argument(
        "--engine-step-frontier-atol",
        type=float,
        default=DEFAULT_ENGINE_STEP_FRONTIER_ATOL,
        help="Loose absolute tolerance when Python and Rust Pareto frontiers choose different rows.",
    )

    args = parser.parse_args()

    print(f"Saving results to {args.output}")

    # Setup logging
    logging.basicConfig(
        level=logging.CRITICAL,
        format="%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s",
    )

    support_matrix = SupportMatrix(
        compare_engine_step_backends=args.compare_engine_step_backends,
        engine_step_comparison_rtol=args.engine_step_comparison_rtol,
        engine_step_comparison_atol=args.engine_step_comparison_atol,
        engine_step_frontier_rtol=args.engine_step_frontier_rtol,
        engine_step_frontier_atol=args.engine_step_frontier_atol,
    )
    results = support_matrix.test_support_matrix(max_workers=args.max_workers)

    # Always save results (now has a default output location)
    support_matrix.save_results_to_csv(results, args.output)


if __name__ == "__main__":
    main()
