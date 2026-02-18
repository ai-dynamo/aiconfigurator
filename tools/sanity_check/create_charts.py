#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Script to create visualization charts from performance data files in the systems directory.
This script is designed to run in GitHub Actions to visualize new performance data added in PRs.
"""

import argparse
import functools
import os
import subprocess
import sys
from collections import defaultdict

from aiconfigurator.sdk.perf_database import get_database

# Disable interactive backend
os.environ["MPLBACKEND"] = "agg"
import matplotlib.pyplot as plt

# Import validate_database.ipynb jupyter notebook
old_cwd = os.getcwd()
os.chdir(os.path.abspath(os.path.dirname(__file__)))
import import_ipynb  # noqa: F401
import validate_database

os.chdir(old_cwd)


def get_changed_files(base_ref: str, head_ref: str) -> list[str]:
    """Get list of files changed between base and head refs."""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", base_ref, head_ref],
            capture_output=True,
            text=True,
            check=True,
        )
        changed_files = [f.strip() for f in result.stdout.split("\n") if f.strip()]
        return [f for f in changed_files if f.startswith("src/aiconfigurator/systems/")]
    except subprocess.CalledProcessError as e:
        print(f"Error getting changed files: {e}", file=sys.stderr)
        return []


def create_charts(
    backend: str,
    backend_version: str,
    system: str,
    perf_files: list[str],
    output_dir: str,
    pr_body_file: str,
):
    new_nccl_perf_collected = False  # FIXME
    database = get_database(system=system, backend=backend, version=backend_version)

    # TODO: for simplicity & maintainability, maybe better to ignore perf_files and
    # just call all chart functions in validate_database?

    op_to_chart_function = {
        "gemm": [validate_database.visualize_gemm],
        "context_attention": [
            validate_database.visualize_context_attention,
            validate_database.visualize_context_attention_with_prefix,
        ],
        "generation_attention": [
            validate_database.visualize_generation_attention,
            validate_database.visualize_generation_attention_b,
        ],
        "context_mla": [
            validate_database.visualize_context_mla_with_prefix,
        ],
        "generation_mla": [
            validate_database.visualize_generation_mla,
            validate_database.visualize_generation_mla_b,
        ],
        "moe": [validate_database.visualize_moe],
        "custom_allreduce": [validate_database.visualize_allreduce],
        "nccl": [
            functools.partial(validate_database.visualize_nccl, operation="all_gather"),
            functools.partial(validate_database.visualize_nccl, operation="all_reduce"),
            functools.partial(validate_database.visualize_nccl, operation="alltoall"),
            functools.partial(validate_database.visualize_nccl, operation="reduce_scatter"),
        ],
    }

    with open(pr_body_file, "a") as f:
        f.write("### Sanity check plots\n")

    # Create sanity check plots for each op and save them to the output directory.
    # Append the plot image URLs to the PR body file.
    for op, funcs_to_create_charts in op_to_chart_function.items():
        op_perf_file = f"{op}_perf.txt"
        if op_perf_file not in perf_files and not (op == "nccl" and new_nccl_perf_collected):
            continue

        for create_chart_func in funcs_to_create_charts:
            if isinstance(create_chart_func, functools.partial):
                chart_op_name = create_chart_func.func.__name__
                for val in create_chart_func.keywords.values():
                    chart_op_name += f"_{val}"
            else:
                chart_op_name = create_chart_func.__name__
            chart_op_name = chart_op_name.replace("visualize_", "")
            img_path = f"{chart_op_name}_{system}_{backend}_{backend_version}.png"

            try:
                create_chart_func(database)
            except Exception as e:
                err_msg = f"Error creating chart for {chart_op_name}: ```{e}```"
                print(err_msg)
                with open(pr_body_file, "a") as f:
                    f.write(err_msg + "\n")
                print(err_msg)
                continue

            plt.savefig(os.path.join(output_dir, img_path))

            img_description = f"Sanity check plot for {system}, {backend}:{backend_version}, {chart_op_name}"
            with open(pr_body_file, "a") as f:
                f.write(f"#### {img_description}\n")
                # FIXME: replace with actual URL
                f.write(f'<img alt="{img_description}" src="https://example.com/{img_path}" />\n')
                f.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="./charts_output")
    parser.add_argument("--output-pr-body", type=str, default="./pr_body.md")
    parser.add_argument("--base-ref", type=str, default="origin/main")
    parser.add_argument("--head-ref", type=str, default="HEAD")
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    with open(args.output_pr_body, "w") as f:
        f.write("# Sanity Check Charts and Error Summary\n")

    changed_files = get_changed_files(args.base_ref, args.head_ref)

    # Organize changed files by (system, backend, backend_version) so that
    # they are grouped togather in the output text.
    # map (system, backend, backend_version) -> [changed_file]
    system_backend_version_to_changed_files = defaultdict(list)

    for changed_file in changed_files:
        # remove prefix
        changed_file = changed_file.replace("src/aiconfigurator/systems/", "")
        # split by /
        parts = changed_file.split("/")

        # <system>.yaml
        if len(parts) == 1 and parts[0].endswith(".yaml"):
            # Ignore for now
            continue

        # data/<system>/<backend>/<backend_version>/*.txt
        elif len(parts) == 5 and parts[0] == "data":
            system = parts[1]
            backend = parts[2]
            backend_version = parts[3]

            # data/<system>/nccl/<nccl_version>/nccl_perf.txt
            if backend == "nccl":
                # Ignore for now
                continue

            perf_file = parts[4]
            system_backend_version_to_changed_files[(system, backend, backend_version)].append(perf_file)

        else:
            print(f"Unhandled changed file: {changed_file}")
            continue

    for (system, backend, backend_version), perf_files in system_backend_version_to_changed_files.items():
        with open(args.output_pr_body, "a") as f:
            f.write(f"## Summary and charts for {system} {backend}:{backend_version}\n")
        try:
            print(f"Creating charts for {system} {backend} {backend_version} with perf files: {perf_files}")
            create_charts(
                backend,
                backend_version,
                system,
                perf_files,
                output_dir,
                args.output_pr_body,
            )
        except Exception as e:
            err_msg = f"Error creating charts for {system} {backend} {backend_version}: ```{e}```"
            print(err_msg)
            with open(args.output_pr_body, "a") as f:
                f.write(err_msg + "\n")


if __name__ == "__main__":
    sys.exit(main())
