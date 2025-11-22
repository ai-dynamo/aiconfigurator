#!/usr/bin/env python3
"""
Script to iterate over all model/system/backend/version combinations for complete support matrix generation

Usage:
    python examples/run_support_matrix_test.py --output results.csv
"""

import argparse
import logging

from aiconfigurator.sdk.suppport_matrix import SupportMatrix


def main():
    parser = argparse.ArgumentParser(
        description="Test AIConfigurator support matrix across all model/system/backend combinations"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file to save results (CSV format)",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.CRITICAL,
        format="%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s",
    )

    support_matrix = SupportMatrix()
    results = support_matrix.test_support_matrix()

    if args.output:
        support_matrix.save_results_to_csv(results, args.output)


if __name__ == "__main__":
    main()
