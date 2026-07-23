#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Script to create visualization charts from performance data files in the systems directory.
This script is designed to run in GitHub Actions to visualize new performance data added in PRs.
"""

import argparse
import functools
import logging
import math
import os
import subprocess
import sys
import textwrap
from collections import defaultdict
from pathlib import Path

import yaml

from aiconfigurator.sdk.perf_database import PerfDataNotAvailableError, get_database

# Disable interactive backend
os.environ["MPLBACKEND"] = "agg"
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SYSTEMS_PREFIX = "aic-core/src/aiconfigurator_core/systems/"

# Import validate_database.ipynb jupyter notebook
old_cwd = os.getcwd()
os.chdir(os.path.abspath(os.path.dirname(__file__)))
import import_ipynb  # noqa: F401
import validate_database

os.chdir(old_cwd)


CLI_SMOKE_REQUIRED_PERF_FILES = (
    "gemm_perf.parquet",
    "context_attention_perf.parquet",
    "generation_attention_perf.parquet",
)

OPTIONAL_CHART_ERROR_SNIPPETS = (
    "values is None or empty",
    "QH6013 qhull input error",
    "input is less than",
    "File does not exist at",
    "list index out of range",
)


# Legacy top-level backend dirs (family-first layout treats any other first-level
# directory under <system>/ as a family dir containing <backend>/<version>
# subtrees). Keep textually identical to the CANONICAL _KNOWN_BACKEND_DIRS in
# aic-core/src/aiconfigurator_core/sdk/operations/base.py, which lists every
# copy that must stay in sync (this standalone tool cannot import aic-core).
_KNOWN_BACKEND_DIRS = {"trtllm", "sglang", "vllm", "nccl", "oneccl"}


def _dir_is_incomplete(path: str) -> bool:
    """Yaml-first partial-dir check (collection_meta.yaml status:partial), with
    INCOMPLETE.txt as the legacy fallback. Duplicated (not imported) from
    aiconfigurator_core.sdk.perf_database._version_dir_state, the source of
    truth for this semantic — kept local so this tool doesn't take an aic-core
    dependency for one predicate. Malformed collection_meta.yaml raises
    ValueError naming the file, matching that canonical loader's fail-loudly
    behavior (unlike operations/base.py's deliberately lenient hot-path
    duplicate of this same predicate). See the CONTRACT NOTE on
    _version_dir_is_partial in
    aic-core/src/aiconfigurator_core/sdk/operations/base.py
    for the intentional resolver-lenient/admission-strict split and
    the full list of copies."""
    meta_path = os.path.join(path, "collection_meta.yaml")
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"{meta_path}: failed to parse collection_meta.yaml: {e}") from e
        tables = meta.get("tables") if isinstance(meta, dict) else None
        return isinstance(tables, dict) and any(
            isinstance(t, dict) and t.get("status") == "partial" for t in tables.values()
        )
    return os.path.isfile(os.path.join(path, "INCOMPLETE.txt"))


def _systems_data_root() -> str:
    return os.path.join(REPO_ROOT, "aic-core", "src", "aiconfigurator_core", "systems", "data")


def _legacy_data_dir(system: str, backend: str, backend_version: str) -> str:
    return os.path.join(_systems_data_root(), system, backend, backend_version)


def _perf_file_paths(perf_root: str, system: str, backend: str, backend_version: str) -> dict[str, Path]:
    """Map each *_perf.parquet (and *_perf.txt) basename for this
    (system, backend, backend_version) to its full path, aggregated across the
    legacy (<backend>/<version>) dir AND every family-first
    (<family>/<backend>/<version>) dir holding the same (backend, version).

    This deliberately does not pick a single winning directory: one
    (backend, version) can legitimately split its perf files across multiple
    family dirs (e.g. gemm files under <system>/gemm/<backend>/<version>/ and
    attention files under <system>/attention/<backend>/<version>/), and callers
    need the union of all of them.

    On a basename collision across directories, the legacy dir's copy wins (for
    determinism during the migration window) and a warning is logged.
    """
    system_dir = Path(perf_root) / system
    legacy_dir = system_dir / backend / backend_version

    try:
        entries = sorted(os.listdir(system_dir))
    except OSError:
        entries = []

    family_dirs = []
    for entry in entries:
        if entry in _KNOWN_BACKEND_DIRS:
            continue
        candidate = system_dir / entry / backend / backend_version
        if candidate.is_dir():
            family_dirs.append(candidate)

    def _is_perf_file(name: str) -> bool:
        return name.endswith("_perf.parquet") or name.endswith("_perf.txt")

    paths: dict[str, Path] = {}
    for data_dir in family_dirs:
        for name in sorted(os.listdir(data_dir)):
            if not _is_perf_file(name):
                continue
            full_path = data_dir / name
            if name in paths:
                logger.warning(
                    "Duplicate perf file %r found in multiple family dirs (keeping %s, ignoring %s)",
                    name,
                    paths[name],
                    full_path,
                )
                continue
            paths[name] = full_path

    if legacy_dir.is_dir():
        for name in sorted(os.listdir(legacy_dir)):
            if not _is_perf_file(name):
                continue
            full_path = legacy_dir / name
            if name in paths:
                logger.warning(
                    "Perf file %r present in both legacy dir and a family dir (keeping legacy copy %s, ignoring %s)",
                    name,
                    full_path,
                    paths[name],
                )
            paths[name] = full_path

    return paths


def _short_error(exc: Exception) -> str:
    short_error_str = str(exc).split("\n")[0].strip()
    if len(short_error_str) > 100:
        short_error_str = short_error_str[:97] + "..."
    return short_error_str


def _is_optional_chart_error(exc: Exception) -> bool:
    if isinstance(exc, PerfDataNotAvailableError):
        return True
    error = str(exc)
    return any(snippet in error for snippet in OPTIONAL_CHART_ERROR_SNIPPETS)


def _chart_op_name(create_chart_func) -> str:
    if isinstance(create_chart_func, functools.partial):
        chart_op_name = create_chart_func.func.__name__
        for val in create_chart_func.keywords.values():
            chart_op_name += f"_{val}"
    else:
        chart_op_name = create_chart_func.__name__
    return chart_op_name.replace("visualize_", "")


def _load_chart_op_data(database, op: str) -> None:
    """Load the lazy op data expected by the legacy notebook visualizers."""
    from aiconfigurator.sdk.operations import (
        GEMM,
        NCCL,
        ContextAttention,
        ContextDSAModule,
        ContextMLA,
        CustomAllReduce,
        GenerationAttention,
        GenerationDSAModule,
        GenerationMLA,
        MLAModule,
        MoE,
    )

    op_loaders = {
        "gemm": (GEMM,),
        "context_attention": (ContextAttention,),
        "generation_attention": (GenerationAttention,),
        "context_mla": (ContextMLA, MLAModule),
        "generation_mla": (GenerationMLA, MLAModule),
        "moe": (MoE,),
        "custom_allreduce": (CustomAllReduce,),
        "nccl": (NCCL,),
        "dsa_module": (ContextDSAModule, GenerationDSAModule),
    }

    for loader in op_loaders.get(op, ()):
        loader.load_data(database)


class SkippedSiliconPoints:
    """Temporarily skip unavailable SILICON query points while drawing charts."""

    QUERY_METHODS = (
        "query_gemm",
        "query_context_attention",
        "query_generation_attention",
        "query_context_mla",
        "query_generation_mla",
    )

    def __init__(self, database):
        self.database = database
        self.calls = 0
        self.successes = 0
        self.skipped = 0
        self._originals = {}

    def __enter__(self):
        for name in self.QUERY_METHODS:
            if not hasattr(self.database, name):
                continue
            original = getattr(self.database, name)
            self._originals[name] = original
            setattr(self.database, name, self._wrap(original))
        return self

    def __exit__(self, exc_type, exc, tb):
        for name, original in self._originals.items():
            setattr(self.database, name, original)

    def _wrap(self, original):
        def wrapped(*args, **kwargs):
            database_mode = str(kwargs.get("database_mode", ""))
            is_silicon = database_mode.endswith(".SILICON") or database_mode == "SILICON"
            if is_silicon:
                self.calls += 1
            try:
                result = original(*args, **kwargs)
            except Exception:
                if not is_silicon:
                    raise
                self.skipped += 1
                return math.nan
            if is_silicon:
                try:
                    if math.isfinite(float(result)):
                        self.successes += 1
                except (TypeError, ValueError):
                    pass
            return result

        return wrapped


def run_cli_smoke_test(system: str, backend: str, backend_version: str) -> tuple[list[str], bool, str, str]:
    """Run aiconfigurator cli default for the given system/backend/version. Returns (cmd, success, stdout, stderr)."""
    cmd = [
        "aiconfigurator",
        "cli",
        "default",
        "--backend",
        backend,
        "--backend-version",
        backend_version,
        "--system",
        system,
        "--model",
        "Qwen/Qwen3-32B",
        "--total-gpus",
        "16",
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=240,
        )
        return (cmd, result.returncode == 0, result.stdout or "", result.stderr or "")
    except subprocess.TimeoutExpired as e:
        out = getattr(e, "stdout", None) or ""
        err = getattr(e, "stderr", None) or ""
        if isinstance(out, bytes):
            out = out.decode("utf-8", errors="replace")
        if isinstance(err, bytes):
            err = err.decode("utf-8", errors="replace")
        return (cmd, False, out, err + "\n(Command timed out after 240s)")
    except Exception as e:
        return (cmd, False, "", str(e))


def should_run_cli_smoke_test(system: str, backend: str, backend_version: str) -> tuple[bool, str]:
    """Return whether the default Qwen CLI smoke test has enough data to be meaningful."""
    perf_file_paths = _perf_file_paths(_systems_data_root(), system, backend, backend_version)
    missing_files = [perf_file for perf_file in CLI_SMOKE_REQUIRED_PERF_FILES if perf_file not in perf_file_paths]
    if missing_files:
        return (
            False,
            f"required default-model perf files are not present for this backend version: {', '.join(missing_files)}",
        )
    return True, ""


def get_changed_files(base_ref: str, head_ref: str) -> list[str]:
    """Get list of files changed between base and head refs."""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", f"{base_ref}...{head_ref}"],
            capture_output=True,
            text=True,
            check=True,
        )
        changed_files = [f.strip() for f in result.stdout.split("\n") if f.strip()]
        return [f for f in changed_files if f.startswith(SYSTEMS_PREFIX)]
    except subprocess.CalledProcessError as e:
        print(f"Error getting changed files: {e}", file=sys.stderr)
        return []


def get_csv_to_parquet_conversion_files(base_ref: str, head_ref: str) -> set[str]:
    """Return added parquet files that replace same-stem legacy text files."""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-status", f"{base_ref}...{head_ref}"],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error getting changed file statuses: {e}", file=sys.stderr)
        return set()

    added_parquet: set[str] = set()
    deleted_legacy_as_parquet: set[str] = set()
    renamed_legacy_as_parquet: set[str] = set()
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        status = parts[0]
        paths = parts[1:]
        if status.startswith("A") and paths:
            path = paths[-1]
            if path.startswith(SYSTEMS_PREFIX) and path.endswith("_perf.parquet"):
                added_parquet.add(path)
        elif status.startswith("D") and paths:
            path = paths[0]
            if path.startswith(SYSTEMS_PREFIX) and path.endswith("_perf.txt"):
                deleted_legacy_as_parquet.add(f"{os.path.splitext(path)[0]}.parquet")
        elif status.startswith("R") and len(paths) == 2:
            old_path, new_path = paths
            if (
                old_path.startswith(SYSTEMS_PREFIX)
                and new_path.startswith(SYSTEMS_PREFIX)
                and old_path.endswith("_perf.txt")
                and new_path.endswith("_perf.parquet")
                and os.path.splitext(old_path)[0] == os.path.splitext(new_path)[0]
            ):
                renamed_legacy_as_parquet.add(new_path)

    return (added_parquet & deleted_legacy_as_parquet) | renamed_legacy_as_parquet


def create_charts(
    backend: str,
    backend_version: str,
    system: str,
    perf_files: list[str],
    output_dir: str,
    output_md_file: str,
):
    new_nccl_perf_collected = False  # FIXME
    all_perf_files = set(_perf_file_paths(_systems_data_root(), system, backend, backend_version))

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
        "dsa_module": [validate_database.visualize_dsa_module],
    }
    op_to_required_perf_files = {
        "dsa_module": ("dsa_context_module_perf.parquet", "dsa_generation_module_perf.parquet"),
    }

    xpu_systems = ["b60"]
    if system in xpu_systems:
        op_to_chart_function["generation_attention"] = [
            fn
            for fn in op_to_chart_function["generation_attention"]
            if fn != validate_database.visualize_generation_attention_b
        ]

    with open(output_md_file, "a") as f:
        f.write(
            "### Chart Generation Report for "
            f"system: {system}, backend: {backend}, backend_version: {backend_version}\n"
        )

    database = get_database(
        system=system,
        backend=backend,
        version=backend_version,
    )
    if database is None:
        with open(output_md_file, "a") as f:
            f.write("- Skipped ⚠️: no complete perf database is available for this system/backend/version\n")
        return

    # Create sanity check plots for each op and save them to the output directory.
    # Append the plot image URLs to the output md file.
    perf_files = set(perf_files)
    for op, funcs_to_create_charts in op_to_chart_function.items():
        required_perf_files = op_to_required_perf_files.get(op, (f"{op}_perf.parquet",))
        if not any(perf_file in perf_files for perf_file in required_perf_files) and not (
            op == "nccl" and new_nccl_perf_collected
        ):
            continue

        missing_required = [perf_file for perf_file in required_perf_files if perf_file not in all_perf_files]
        if missing_required:
            with open(output_md_file, "a") as f:
                f.write(
                    f"- `{op}` Skipped ⚠️: required paired perf files are not present: {', '.join(missing_required)}\n"
                )
            continue

        for create_chart_func in funcs_to_create_charts:
            chart_op_name = _chart_op_name(create_chart_func)
            img_path = f"{chart_op_name}_{system}_{backend}_{backend_version}.png"

            try:
                plt.close("all")
                _load_chart_op_data(database, op)
                with SkippedSiliconPoints(database) as skipped_points:
                    create_chart_func(database)
                if skipped_points.calls and not skipped_points.successes:
                    with open(output_md_file, "a") as f:
                        f.write(
                            f"- `{chart_op_name}` Skipped ⚠️: no available silicon data points "
                            f"for the sanity chart grid ({skipped_points.skipped} unavailable points)\n"
                        )
                    continue
                if not plt.get_fignums():
                    with open(output_md_file, "a") as f:
                        f.write(f"- `{chart_op_name}` Skipped ⚠️: no chart was generated for this data shape\n")
                    continue
            except Exception as e:
                short_error_str = _short_error(e)
                status = "Skipped ⚠️" if _is_optional_chart_error(e) else "Error ❌"
                with open(output_md_file, "a") as f:
                    f.write(f"- `{chart_op_name}` {status}: {short_error_str}\n")

                print(f"Error creating chart for {chart_op_name}: {e}")
                plt.close("all")
                continue

            plt.savefig(os.path.join(output_dir, img_path))
            plt.close("all")

            with open(output_md_file, "a") as f:
                if skipped_points.skipped:
                    f.write(f"- `{chart_op_name}` ✅ ({skipped_points.skipped} unavailable points skipped)\n")
                else:
                    f.write(f"- `{chart_op_name}` ✅\n")

    _max_output_len = 4000
    run_smoke, skip_reason = should_run_cli_smoke_test(system, backend, backend_version)
    with open(output_md_file, "a") as f:
        if not run_smoke:
            f.write(f"- CLI smoke test Skipped ⚠️: {skip_reason}\n")
            return

    # Smoke test: run aiconfigurator cli default for this system/backend/version
    smoke_cmd, smoke_ok, smoke_stdout, smoke_stderr = run_cli_smoke_test(system, backend, backend_version)
    with open(output_md_file, "a") as f:
        if smoke_ok:
            f.write("- CLI smoke test ✅\n")
        else:
            f.write("- CLI smoke test ❌\n")
            out_trunc = (
                (smoke_stdout[:_max_output_len] + "... (truncated)\n")
                if len(smoke_stdout) > _max_output_len
                else smoke_stdout
            )
            err_trunc = (
                (smoke_stderr[:_max_output_len] + "... (truncated)\n")
                if len(smoke_stderr) > _max_output_len
                else smoke_stderr
            )
            cmd_str = " ".join(smoke_cmd)
            f.write("\n\n<details><summary>command / stdout / stderr</summary>\n\n")
            f.write("```text\n")
            f.write("command:\n" + cmd_str + "\n\n")
            if out_trunc:
                f.write("stdout:\n" + out_trunc + "\n")
            if err_trunc:
                f.write("stderr:\n" + err_trunc + "\n")
            f.write("```\n\n</details>\n\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="./charts_output")
    parser.add_argument("--output-md-file", type=str, default="./comment.md")
    parser.add_argument("--base-ref", type=str, default="origin/main")
    parser.add_argument("--head-ref", type=str, default="HEAD")
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    changed_files = get_changed_files(args.base_ref, args.head_ref)
    csv_to_parquet_conversion_files = get_csv_to_parquet_conversion_files(args.base_ref, args.head_ref)
    if csv_to_parquet_conversion_files:
        print(
            "Skipping "
            f"{len(csv_to_parquet_conversion_files)} CSV-to-parquet conversion files already covered by parquet diff."
        )

    # Organize changed files by (system, backend, backend_version) so that
    # they are grouped together in the output text.
    # map (system, backend, backend_version) -> [changed_file]
    system_backend_version_to_changed_files = defaultdict(list)

    for changed_file in changed_files:
        if changed_file in csv_to_parquet_conversion_files:
            continue

        # remove prefix
        changed_file = changed_file.removeprefix(SYSTEMS_PREFIX)
        # split by /
        parts = changed_file.split("/")

        # <system>.yaml
        if len(parts) == 1 and parts[0].endswith(".yaml"):
            # Ignore for now
            continue

        # data/<system>/<backend>/<backend_version>/*.parquet
        elif len(parts) == 5 and parts[0] == "data":
            system = parts[1]
            backend = parts[2]
            backend_version = parts[3]

            # data/<system>/nccl/<nccl_version>/nccl_perf.parquet
            # data/<system>/oneccl/<oneccl_version>/oneccl_perf.parquet
            if backend in ("nccl", "oneccl"):
                # Ignore for now
                continue

            perf_file = parts[4]
            if perf_file == "INCOMPLETE.txt" or not perf_file.endswith("_perf.parquet"):
                continue

            # A 5-part changed path is legacy-shaped, so its partial marker
            # (collection_meta.yaml status, INCOMPLETE.txt fallback) lives in
            # the exact legacy dir — a best-guess resolution across layouts
            # could pick an unrelated family dir here.
            data_dir = _legacy_data_dir(system, backend, backend_version)
            if _dir_is_incomplete(data_dir):
                continue
            system_backend_version_to_changed_files[(system, backend, backend_version)].append(perf_file)

        # data/<system>/<family>/<backend>/<backend_version>/*.parquet (family-first layout)
        elif len(parts) == 6 and parts[0] == "data":
            system = parts[1]
            family = parts[2]
            backend = parts[3]
            backend_version = parts[4]

            # data/<system>/<family>/nccl/<nccl_version>/nccl_perf.parquet
            # data/<system>/<family>/oneccl/<oneccl_version>/oneccl_perf.parquet
            if backend in ("nccl", "oneccl"):
                # Ignore for now
                continue

            perf_file = parts[5]
            if perf_file == "INCOMPLETE.txt" or not perf_file.endswith("_perf.parquet"):
                continue

            data_dir = os.path.join(_systems_data_root(), system, family, backend, backend_version)
            if _dir_is_incomplete(data_dir):
                continue
            system_backend_version_to_changed_files[(system, backend, backend_version)].append(perf_file)

        else:
            print(f"Unhandled changed file: {changed_file}")
            continue

    # Only create comment file if there are files to process
    if not system_backend_version_to_changed_files:
        print("No matching perf data files found to process. Skipping chart generation.")
        return 0

    with open(args.output_md_file, "w") as f:
        f.write("## Sanity Check Chart Generation Report\n")
        # github action will insert a link here
        f.write("download_link_placeholder\n")
        f.write(
            textwrap.dedent("""
            New perf data files were detected in this PR. Please use the link above to
            download sanity check charts for the new perf data to compare the collected
            perf data vs SOL (theoretical max performance).

            Below is a report of whether the chart generation was successful for each op.
            If doesn't validate whether the perf data itself is sane.
        """)
        )

    for (system, backend, backend_version), perf_files in system_backend_version_to_changed_files.items():
        try:
            print(f"Creating charts for {system} {backend} {backend_version} with perf files: {perf_files}")
            create_charts(
                backend,
                backend_version,
                system,
                perf_files,
                output_dir,
                args.output_md_file,
            )
        except Exception as e:
            err_msg = f"Error creating charts for {system} {backend} {backend_version}: ```{e}```"
            print(err_msg)
            with open(args.output_md_file, "a") as f:
                f.write(err_msg + "\n")


if __name__ == "__main__":
    sys.exit(main())
