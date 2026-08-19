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
    VALID_STATUSES,
    SupportMatrix,
)


def main():
    # Default output location: split per-system CSVs under <package>/systems/support_matrix/
    default_output = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "src",
        "aiconfigurator",
        "systems",
        "support_matrix",
    )

    parser = argparse.ArgumentParser(
        description="Test AIConfigurator support matrix across all model/system/backend combinations"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=default_output,
        help=f"Output directory for split CSV results, or a legacy CSV file path (default: {default_output})",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of processes for parallel execution (default: auto)",
    )
    parser.add_argument("--model", "--model-path", dest="model", type=str, default=None, help="Only test this model.")
    parser.add_argument("--system", type=str, default=None, help="Only test this system.")
    parser.add_argument("--backend", type=str, default=None, help="Only test this backend.")
    parser.add_argument(
        "--backend-version",
        "--version",
        dest="backend_version",
        type=str,
        default=None,
        help="Only test this backend database version.",
    )
    parser.add_argument(
        "--mode",
        choices=["agg", "disagg", "all"],
        default="all",
        help="Only test one serving mode. Default: all.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        default=False,
        help="Run checks and print the summary without writing support-matrix CSV files.",
    )
    parser.add_argument(
        "--expect-status",
        choices=sorted(VALID_STATUSES),
        default=None,
        help="Replay assertion: require every selected row to have this status.",
    )
    parser.add_argument(
        "--expect-error-prefix",
        default=None,
        help="Replay assertion: require every selected row error to start with this prefix.",
    )
    args = parser.parse_args()

    has_filters = (
        any(arg is not None for arg in (args.model, args.system, args.backend, args.backend_version))
        or args.mode != "all"
    )
    if has_filters and not args.no_save and args.output == default_output:
        parser.error("filtered support-matrix runs require --no-save or an explicit --output path")
    if (args.expect_status is not None or args.expect_error_prefix is not None) and not args.no_save:
        parser.error("replay expectations require --no-save")

    if args.no_save:
        print("Running support-matrix checks without writing CSV output")
    else:
        print(f"Saving results to {args.output}")

    # Setup logging
    logging.basicConfig(
        level=logging.CRITICAL,
        format="%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s",
    )

    support_matrix = SupportMatrix()
    combinations = support_matrix.generate_combinations()
    if args.model is not None:
        combinations = [combo for combo in combinations if combo[0] == args.model]
    if args.system is not None:
        combinations = [combo for combo in combinations if combo[1] == args.system]
    if args.backend is not None:
        combinations = [combo for combo in combinations if combo[2] == args.backend]
    if args.backend_version is not None:
        combinations = [combo for combo in combinations if combo[3] == args.backend_version]
    if not combinations:
        parser.error("No support-matrix combinations matched the provided filters.")

    modes_to_test = None if args.mode == "all" else (args.mode,)
    results = support_matrix.test_support_matrix(
        max_workers=args.max_workers,
        combinations=combinations,
        modes_to_test=modes_to_test,
    )

    if args.expect_status is not None or args.expect_error_prefix is not None:
        mismatches = []
        for model, _architecture, system, backend, version, mode, status, error, *_rest in results:
            error = str(error or "")
            print(
                "Observed support-matrix result: "
                f"{model} {system}/{backend} v{version} {mode}: {status}: {error or '<no error>'}"
            )
            if args.expect_status is not None and status != args.expect_status:
                mismatches.append(f"expected status {args.expect_status}, observed {status}")
            if args.expect_error_prefix is not None and not error.startswith(args.expect_error_prefix):
                mismatches.append(
                    f"expected error prefix {args.expect_error_prefix!r}, observed {error or '<no error>'!r}"
                )
        if mismatches:
            parser.exit(1, "Replay expectation failed: " + "; ".join(mismatches) + "\n")

    if args.no_save:
        return
    support_matrix.save_results_to_csv(results, args.output)


if __name__ == "__main__":
    main()
