#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Add tests directory to path for support_matrix module
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tests"))

from sdk.support_matrix.suppport_matrix import SupportMatrix


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

    args = parser.parse_args()

    print(f"Saving results to {args.output}")

    # Setup logging
    logging.basicConfig(
        level=logging.CRITICAL,
        format="%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s",
    )

    support_matrix = SupportMatrix()
    results = support_matrix.test_support_matrix()

    # Always save results (now has a default output location)
    support_matrix.save_results_to_csv(results, args.output)


if __name__ == "__main__":
    main()
