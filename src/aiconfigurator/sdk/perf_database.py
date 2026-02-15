# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import functools
import importlib.resources as pkg_resources
import logging
import math
import os
from collections import defaultdict
from collections.abc import Iterable
from typing import Optional

import numpy as np
import yaml
from scipy import interpolate

from aiconfigurator.sdk import common
from aiconfigurator.sdk.performance_result import PerformanceResult

databases_cache = defaultdict(lambda: defaultdict(lambda: defaultdict()))
logger = logging.getLogger(__name__)

_SYSTEMS_PATHS: list[str] = [os.fspath(pkg_resources.files("aiconfigurator") / "systems")]


def _normalize_systems_paths(raw_paths: str | Iterable[str] | None) -> list[str]:
    default_path = os.fspath(pkg_resources.files("aiconfigurator") / "systems")
    if raw_paths is None:
        return [default_path]
    if isinstance(raw_paths, str):
        entries = [part.strip() for part in raw_paths.split(",") if part.strip()]
    else:
        entries = [os.fspath(entry) for entry in raw_paths if entry is not None]
    if not entries:
        return [default_path]
    resolved: list[str] = []
    for entry in entries:
        if str(entry).lower() == "default":
            resolved.append(default_path)
        else:
            resolved.append(os.fspath(entry))
    return resolved


def set_systems_paths(raw_paths: str | Iterable[str] | None) -> None:
    """
    Override the system search paths for the current process.
    """
    global _SYSTEMS_PATHS
    resolved_paths = _normalize_systems_paths(raw_paths)
    invalid_paths = [path for path in resolved_paths if not os.path.isdir(path)]
    if invalid_paths:
        raise ValueError(
            "Invalid --systems-paths: each entry must be an existing directory. "
            f"Invalid entries: {', '.join(invalid_paths)}"
        )
    _SYSTEMS_PATHS = resolved_paths


def get_systems_paths() -> list[str]:
    return list(_SYSTEMS_PATHS)


def build_no_databases_message() -> str:
    """Build a concise error message for systems path/db validation failures."""
    resolved_paths = get_systems_paths()
    resolved_display = ", ".join(resolved_paths) if resolved_paths else "<none>"
    default_path = os.fspath(pkg_resources.files("aiconfigurator") / "systems")
    has_default = default_path in resolved_paths

    lines = [
        "No loadable performance databases found under --systems-paths.",
        f"Configured systems paths: {resolved_display}",
    ]
    if has_default:
        lines.append(
            "Built-in `default` systems path is already included, and no databases "
            "could be loaded from either default or extra paths."
        )
    else:
        lines.append("Tip: try adding `default` to --systems-paths and run again.")
    return "\n".join(lines)


class PerfDataNotAvailableError(RuntimeError):
    """Raised when required performance data is missing or unsupported for a requested mode."""


def get_supported_databases(
    systems_paths: str | list[str] | None = None,
) -> dict[str, dict[str, list[str]]]:
    """
    Get all supported databases for all systems, backends and versions without loading them.
    """
    supported_sets: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    if systems_paths is None:
        systems_paths = get_systems_paths()
    elif isinstance(systems_paths, str):
        systems_paths = [systems_paths]

    for systems_root in systems_paths:
        try:
            entries = os.listdir(systems_root)
        except Exception as e:
            logger.warning("Could not list systems dir %s: %s", systems_root, e)
            continue
        for entry in entries:
            if not entry.endswith(".yaml"):
                continue
            system = entry[:-5]
            system_yaml_path = os.path.join(systems_root, entry)
            try:
                with open(system_yaml_path) as f:
                    system_spec = yaml.safe_load(f)

                data_dir = os.path.join(systems_root, system_spec.get("data_dir", ""))
                if not os.path.isdir(data_dir):
                    continue

                for backend in common.BackendName:
                    backend_path = os.path.join(data_dir, backend.value)
                    if not os.path.isdir(backend_path):
                        continue

                    versions = [
                        v
                        for v in os.listdir(backend_path)
                        if not v.startswith(".") and os.path.isdir(os.path.join(backend_path, v))
                    ]
                    if versions:
                        supported_sets[system][backend.value].update(versions)
            except Exception as e:
                logger.warning(f"Could not process system config {os.path.basename(system_yaml_path)}: {e}")

    supported_dict = defaultdict(lambda: defaultdict(list))
    for system, backend_versions in supported_sets.items():
        for backend, versions in backend_versions.items():
            supported_dict[system][backend] = sorted(versions)

    return supported_dict


def get_latest_database_version(
    system: str,
    backend: str,
) -> str | None:
    """
    Get the latest database version for a given system and backend
    """
    import re

    supported_databases = get_supported_databases()
    try:
        database_versions = supported_databases[system][backend]
    except KeyError:
        logger.exception(f"database not found for {system=}, {backend=}")
        return None

    def parse_version(version_str):
        """Parse version string into comparable tuple"""
        # Handle different version formats
        version_str = version_str.lower()

        # Extract numeric version pattern (e.g., "1.2.3" from "v1.2.3rc4" or "1.2.3_suffix")
        version_match = re.search(r"(\d+)\.(\d+)\.(\d+)", version_str)
        if version_match:
            major, minor, patch = map(int, version_match.groups())
            version_parts = [major, minor, patch]

            # Handle release candidates (lower priority than stable releases)
            if "rc" in version_str:
                rc_match = re.search(r"rc(\d+)", version_str)
                if rc_match:
                    rc_num = int(rc_match.group(1))
                    version_parts.append(0)  # Stable release indicator
                    version_parts.append(rc_num)  # RC number
                else:
                    version_parts.append(0)  # Stable release indicator
                    version_parts.append(0)  # No RC number
            else:
                version_parts.append(1)  # Stable release (higher priority than RC)
                version_parts.append(0)  # No RC number

            return tuple(version_parts)

        # Try to extract version from other patterns (e.g., "v0.20_fix0719")
        version_match = re.search(r"v?(\d+)\.(\d+)", version_str)
        if version_match:
            major, minor = map(int, version_match.groups())
            version_parts = [major, minor, 0, 1, 0]  # Assume stable release
            return tuple(version_parts)

        # For completely non-standard versions, try to extract any numbers
        numbers = re.findall(r"\d+", version_str)
        if numbers:
            # Use first few numbers found, pad with zeros
            version_parts = [int(x) for x in numbers[:3]]
            while len(version_parts) < 3:
                version_parts.append(0)
            version_parts.extend([0, 0])  # Add RC indicators
            return tuple(version_parts)

        # If no numbers found, return a very low priority tuple
        return (0, 0, 0, -1, 0)

    # Convert version strings to comparable tuples
    versions_ids = []
    for version in database_versions:
        try:
            version_parts = parse_version(version)
            versions_ids.append((version_parts, version))
            logger.debug(f"Parsed version {version} as {version_parts}")
        except Exception as e:
            logger.warning(f"Failed to parse version {version}: {e}")
            continue

    if not versions_ids:
        logger.error(f"no valid versions parsed for {system=}, {backend=}")
        return None

    # Find the latest version by comparing version tuples.
    # The tuple format (major, minor, patch, is_stable, rc_num) ensures
    # correct sorting across stable and RC releases.
    latest_version = max(versions_ids, key=lambda x: x[0])

    logger.debug(f"Latest version for {system}/{backend}: {latest_version[1]} (parsed as {latest_version[0]})")
    return latest_version[1]


def get_database(
    system: str,
    backend: str,
    version: str,
    systems_paths: str | list[str] | None = None,
) -> PerfDatabase | None:
    """
    Get the database for a given system, backend and version

    Args:
        system (str): the system name
        backend (str): the backend name
        version (str): the version name
        systems_paths (str | list[str] | None): the systems search paths

    Returns:
        PerfDatabase: the database for the given system, backend and version
    """
    if systems_paths is None:
        systems_paths = get_systems_paths()
    elif isinstance(systems_paths, str):
        systems_paths = [systems_paths]

    for systems_root in systems_paths:
        system_yaml_path = os.path.join(systems_root, f"{system}.yaml")
        if not os.path.isfile(system_yaml_path):
            continue
        cache_key = (systems_root, system)
        try:
            database = databases_cache[cache_key][backend][version]
            return database
        except KeyError:
            logger.info(f"loading {system=}, {backend=}, {version=}")
            try:
                with open(system_yaml_path) as f:
                    system_spec = yaml.load(f, Loader=yaml.SafeLoader)
                data_dir = system_spec["data_dir"]
            except Exception:
                logger.warning(f"failed to read system spec at {system_yaml_path}, continuing searching")
                continue
            data_path = os.path.join(systems_root, data_dir, backend, version)
            if os.path.exists(data_path):
                try:
                    database = PerfDatabase(system, backend, version, systems_root)
                    databases_cache[cache_key][backend][version] = database
                    return database
                except Exception:
                    logger.warning(f"failed to load {system=}, {backend=}, {version=}, continuing searching")
            else:
                logger.warning(f"data path {data_path} not found, continuing searching")

    logger.error(f"failed to get {system=}, {backend=}, {version=}")
    return None


def get_all_databases(
    systems_paths: str | os.PathLike | Iterable[str] | None = None,
) -> dict[str, dict[str, dict[str, PerfDatabase]]]:
    """
    Get all the databases for all the systems, backends and versions
    """
    database_dict = defaultdict(lambda: defaultdict(lambda: defaultdict()))
    if systems_paths is None:
        systems_paths = get_systems_paths()
    elif isinstance(systems_paths, str):
        systems_paths = [systems_paths]

    seen_systems: dict[str, str] = {}
    for systems_root in systems_paths:
        try:
            entries = os.listdir(systems_root)
        except Exception as e:
            logger.warning("Could not list systems dir %s: %s", systems_root, e)
            continue
        for entry in entries:
            if not entry.endswith(".yaml"):
                continue
            system = entry[:-5]
            if system in seen_systems:
                logger.warning(
                    "System config '%s' already loaded from %s; also found in %s",
                    system,
                    seen_systems[system],
                    systems_root,
                )
            else:
                seen_systems[system] = systems_root
            system_yaml_path = os.path.join(systems_root, entry)
            try:
                with open(system_yaml_path) as f:
                    system_spec = yaml.load(f, Loader=yaml.SafeLoader)
                data_dir = os.path.join(systems_root, system_spec["data_dir"])
                if not os.path.exists(data_dir):
                    continue
                for backend in common.BackendName:
                    if not os.path.exists(os.path.join(data_dir, backend.value)):
                        continue
                    for version in os.listdir(os.path.join(data_dir, backend.value)):
                        if version.startswith("."):
                            continue
                        database = get_database(system, backend.value, version, systems_root)
                        if database is None:
                            continue
                        if version in database_dict[system][backend.value]:
                            existing = database_dict[system][backend.value][version]
                            existing_root = getattr(existing, "systems_root", None) or "unknown"
                            logger.warning(
                                "Database '%s/%s/%s' already loaded from %s; ignoring %s",
                                system,
                                backend.value,
                                version,
                                existing_root,
                                systems_root,
                            )
                            continue
                        database_dict[system][backend.value][version] = database
            except Exception as e:
                logger.warning(f"Could not process system config {os.path.basename(system_yaml_path)}: {e}")

    return database_dict


# by default float16
def load_custom_allreduce_data(custom_allreduce_file):
    """
    Load the custom allreduce data for trtllm with power support (backward compatible).

    Returns:
        dict: Nested dict structure where leaf values are dicts with 'latency' and 'power' keys.
    """
    if not os.path.exists(custom_allreduce_file):
        logger.warning(f"Custom allreduce data file {custom_allreduce_file} not found.")
        return None
    custom_allreduce_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))

    with open(custom_allreduce_file) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Check if power columns exist (backward compatibility)
    has_power = len(rows) > 0 and "power" in rows[0]
    if not has_power:
        logger.debug(f"Legacy database format detected in {custom_allreduce_file} - power will default to 0.0")

    for row in rows:
        dtype, tp_size, message_size, latency = (
            row["allreduce_dtype"],
            row["num_gpus"],
            row["message_size"],
            row["latency"],
        )
        allreduce_strategy = "AUTO"
        message_size = int(message_size)
        latency = float(latency)
        tp_size = int(tp_size)
        dtype = common.CommQuantMode.half  # TODO

        # NEW: Read power with backward compatibility
        power = float(row.get("power", 0.0))

        # NEW: Calculate energy from power and latency
        energy = power * latency  # watt-milliseconds

        try:
            # Check for conflict
            custom_allreduce_data[dtype][tp_size][allreduce_strategy][message_size]
            logger.debug(
                f"value conflict in custom allreduce data: {dtype} {tp_size} {allreduce_strategy} {message_size}"
            )
        except KeyError:
            # Store all three values
            custom_allreduce_data[dtype][tp_size][allreduce_strategy][message_size] = {
                "latency": latency,
                "power": power,
                "energy": energy,  # NEW: precomputed energy
            }

    return custom_allreduce_data


def load_nccl_data(nccl_file):
    """
    Load the nccl data with power support (backward compatible).

    Returns:
        dict: Nested dict structure where leaf values are dicts with 'latency' and 'power' keys.
    """
    if not os.path.exists(nccl_file):
        logger.warning(f"NCCL data file {nccl_file} not found.")
        return None
    nccl_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))

    with open(nccl_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Check if power columns exist (backward compatibility)
    has_power = len(rows) > 0 and "power" in rows[0]
    if not has_power:
        logger.debug(f"Legacy database format detected in {nccl_file} - power will default to 0.0")

    for row in rows:
        dtype, num_gpus, message_size, op_name, latency = (
            row["nccl_dtype"],
            row["num_gpus"],
            row["message_size"],
            row["op_name"],
            row["latency"],
        )
        message_size = int(message_size)
        latency = float(latency)
        num_gpus = int(num_gpus)

        # NEW: Read power with backward compatibility
        power = float(row.get("power", 0.0))

        # NEW: Calculate energy from power and latency
        energy = power * latency  # watt-milliseconds

        dtype = common.CommQuantMode[dtype]
        try:
            # Check for conflict
            nccl_data[dtype][op_name][num_gpus][message_size]
            logger.debug(f"value conflict in nccl data: {dtype} {op_name} {num_gpus} {message_size}")
        except KeyError:
            # Store all three values
            nccl_data[dtype][op_name][num_gpus][message_size] = {
                "latency": latency,
                "power": power,
                "energy": energy,  # NEW: precomputed energy
            }

    return nccl_data


def load_gemm_data(gemm_file):
    """
    Load the gemm data with power support (backward compatible).

    Returns:
        dict: Nested dict structure where leaf values are dicts with
              'latency', 'power', and 'energy' keys.
              For old database formats without power, defaults to power=0.0 and energy=0.0.
    """
    if not os.path.exists(gemm_file):
        logger.warning(f"GEMM data file {gemm_file} not found.")
        return None
    gemm_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))

    with open(gemm_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Check if power columns exist (backward compatibility)
    has_power = len(rows) > 0 and "power" in rows[0]
    if not has_power:
        logger.debug(f"Legacy database format detected in {gemm_file} - power will default to 0.0")

    for row in rows:
        quant_mode, m, n, k, latency = (
            row["gemm_dtype"],
            row["m"],
            row["n"],
            row["k"],
            row["latency"],
        )
        m = int(m)
        n = int(n)
        k = int(k)
        latency = float(latency)

        # NEW: Read power with backward compatibility
        power = float(row.get("power", 0.0))
        # Note: power_limit is available in row.get("power_limit") if needed for validation

        # NEW: Calculate energy from power and latency
        energy = power * latency  # watt-milliseconds (W·ms)

        # vllm gemm has some awq and gptq data, discard it.
        if quant_mode in ["awq", "gptq"]:
            continue

        quant_mode = common.GEMMQuantMode[quant_mode]

        try:
            # Check for conflict
            gemm_data[quant_mode][m][n][k]
            logger.debug(f"value conflict in gemm data: {quant_mode} {m} {n} {k}")
        except KeyError:
            # Store all three values
            gemm_data[quant_mode][m][n][k] = {
                "latency": latency,
                "power": power,  # Keep for reference
                "energy": energy,  # NEW: precomputed energy
            }

    return gemm_data


def load_compute_scale_data(compute_scale_file):
    """
    Load the compute scale data with power support (backward compatible).

    Returns:
        dict: Nested dict structure {quant_mode: {m: {k: {latency, power, energy}}}}
              For old database formats without power, defaults to power=0.0 and energy=0.0.
    """
    if not os.path.exists(compute_scale_file):
        logger.warning(f"Compute scale data file {compute_scale_file} not found.")
        return None
    compute_scale_data = defaultdict(lambda: defaultdict(lambda: defaultdict()))

    with open(compute_scale_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Check if power columns exist (backward compatibility)
    has_power = len(rows) > 0 and "power" in rows[0]
    if not has_power:
        logger.debug(f"Legacy database format detected in {compute_scale_file} - power will default to 0.0")

    for row in rows:
        quant_mode, m, k, latency = (
            row["quant_dtype"],
            row["m"],
            row["k"],
            row["latency"],
        )
        m = int(m)
        k = int(k)
        latency = float(latency)

        # Read power with backward compatibility
        power = float(row.get("power", 0.0))

        # Calculate energy from power and latency
        energy = power * latency  # watt-milliseconds (W·ms)

        quant_mode = common.GEMMQuantMode[quant_mode]

        try:
            # Check for conflict
            compute_scale_data[quant_mode][m][k]
            logger.debug(f"value conflict in compute_scale data: {quant_mode} {m} {k}")
        except KeyError:
            # Store all three values
            compute_scale_data[quant_mode][m][k] = {
                "latency": latency,
                "power": power,
                "energy": energy,
            }

    return compute_scale_data


def load_scale_matrix_data(scale_matrix_file):
    """
    Load the scale matrix data with power support (backward compatible).

    Returns:
        dict: Nested dict structure {quant_mode: {m: {k: {latency, power, energy}}}}
              For old database formats without power, defaults to power=0.0 and energy=0.0.
    """
    if not os.path.exists(scale_matrix_file):
        logger.warning(f"Scale matrix data file {scale_matrix_file} not found.")
        return None
    scale_matrix_data = defaultdict(lambda: defaultdict(lambda: defaultdict()))

    with open(scale_matrix_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Check if power columns exist (backward compatibility)
    has_power = len(rows) > 0 and "power" in rows[0]
    if not has_power:
        logger.debug(f"Legacy database format detected in {scale_matrix_file} - power will default to 0.0")

    for row in rows:
        quant_mode, m, k, latency = (
            row["quant_dtype"],
            row["m"],
            row["k"],
            row["latency"],
        )
        m = int(m)
        k = int(k)
        latency = float(latency)

        # Read power with backward compatibility
        power = float(row.get("power", 0.0))

        # Calculate energy from power and latency
        energy = power * latency  # watt-milliseconds (W·ms)

        quant_mode = common.GEMMQuantMode[quant_mode]

        try:
            # Check for conflict
            scale_matrix_data[quant_mode][m][k]
            logger.debug(f"value conflict in scale_matrix data: {quant_mode} {m} {k}")
        except KeyError:
            # Store all three values
            scale_matrix_data[quant_mode][m][k] = {
                "latency": latency,
                "power": power,
                "energy": energy,
            }

    return scale_matrix_data


def load_moe_data(moe_file):
    """
    Load the moe data with power support (backward compatible).

    Returns:
        tuple: (moe_default_data, moe_low_latency_data) where leaf values are dicts
               with 'latency', 'power', and 'energy' keys. For old formats, power/energy default to 0.0.
    """
    if not os.path.exists(moe_file):
        logger.warning(f"MOE data file {moe_file} not found.")
        return None, None

    moe_default_data = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(
                        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))
                    )
                )
            )
        )
    )
    moe_low_latency_data = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(
                        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))
                    )
                )
            )
        )
    )

    with open(moe_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Check if power columns exist (backward compatibility)
    has_power = len(rows) > 0 and "power" in rows[0]
    if not has_power:
        logger.debug(f"Legacy database format detected in {moe_file} - power will default to 0.0")

    for row in rows:
        (
            quant_mode,
            num_tokens,
            hidden_size,
            inter_size,
            topk,
            num_experts,
            moe_tp_size,
            moe_ep_size,
            workload_distribution,
            latency,
        ) = (
            row["moe_dtype"],
            row["num_tokens"],
            row["hidden_size"],
            row["inter_size"],
            row["topk"],
            row["num_experts"],
            row["moe_tp_size"],
            row["moe_ep_size"],
            row["distribution"],
            row["latency"],
        )
        kernel_source = row["kernel_source"]  # moe_torch_flow, moe_torch_flow_min_latency, moe_torch_flow
        num_tokens = int(num_tokens)
        hidden_size = int(hidden_size)
        inter_size = int(inter_size)
        topk = int(topk)
        num_experts = int(num_experts)
        moe_tp_size = int(moe_tp_size)
        moe_ep_size = int(moe_ep_size)
        latency = float(latency)

        # NEW: Read power with backward compatibility
        power = float(row.get("power", 0.0))

        # NEW: Calculate energy from power and latency
        energy = power * latency  # watt-milliseconds

        quant_mode = common.MoEQuantMode[quant_mode]

        moe_data = moe_low_latency_data if kernel_source == "moe_torch_flow_min_latency" else moe_default_data

        try:
            # Check for conflict
            moe_data[quant_mode][workload_distribution][topk][num_experts][hidden_size][inter_size][moe_tp_size][
                moe_ep_size
            ][num_tokens]
            logger.debug(
                f"value conflict in moe data: {workload_distribution} {quant_mode} {topk} "
                f"{num_experts} {hidden_size} {inter_size} {moe_tp_size} {moe_ep_size} "
                f"{num_tokens}"
            )
        except KeyError:
            # Store all three values
            moe_data[quant_mode][workload_distribution][topk][num_experts][hidden_size][inter_size][moe_tp_size][
                moe_ep_size
            ][num_tokens] = {
                "latency": latency,
                "power": power,
                "energy": energy,  # NEW: precomputed energy
            }

    return moe_default_data, moe_low_latency_data


def load_context_attention_data(context_attention_file):
    """
    Load the context attention data with power support (backward compatible).

    Returns:
        dict: Nested dict structure where leaf values are dicts with 'latency' and 'power' keys.
    """
    if not os.path.exists(context_attention_file):
        logger.warning(f"Context attention data file {context_attention_file} not found.")
        return None
    context_attention_data = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))
                )
            )
        )
    )
    with open(context_attention_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Check if power columns exist (backward compatibility)
    has_power = len(rows) > 0 and "power" in rows[0]
    if not has_power:
        logger.debug(f"Legacy database format detected in {context_attention_file} - power will default to 0.0")

    for row in rows:
        try:
            window_size = row["window_size"]
        except KeyError:  # catch potential error for backward comptability
            window_size = 0
        quant_mode, kv_cache_dtype, b, s, n, kv_n, head_size, latency = (
            row["attn_dtype"],
            row["kv_cache_dtype"],
            row["batch_size"],
            row["isl"],
            row["num_heads"],
            row["num_key_value_heads"],
            row["head_dim"],
            row["latency"],
        )
        b = int(b)
        s = int(s)
        n = int(n)
        kv_n = int(kv_n)
        head_size = int(head_size)
        window_size = int(window_size)
        latency = float(latency)

        # NEW: Read power with backward compatibility
        power = float(row.get("power", 0.0))

        # NEW: Calculate energy from power and latency
        energy = power * latency  # watt-milliseconds

        # we only have kv_n==n(MHA) and kv_n==1,2,4,8(XQA), interp/extrap all other num_kv_heads.
        # Use kv_n = 0 to mean n_kv == n.
        kv_n = 0 if n == kv_n else kv_n

        quant_mode = common.FMHAQuantMode[quant_mode]
        kv_cache_dtype = common.KVCacheQuantMode[kv_cache_dtype]

        try:
            # Check for conflict
            context_attention_data[quant_mode][kv_cache_dtype][kv_n][head_size][window_size][n][s][b]
            logger.debug(
                f"value conflict in context attention data: {quant_mode} {kv_cache_dtype} "
                f"{head_size} {window_size} {kv_n} {n} {s}"
            )
        except KeyError:
            # Store all three values
            context_attention_data[quant_mode][kv_cache_dtype][kv_n][head_size][window_size][n][s][b] = {
                "latency": latency,
                "power": power,
                "energy": energy,  # NEW: precomputed energy
            }

    return context_attention_data


def load_generation_attention_data(generation_attention_file):
    """
    Load the generation attention data with power support (backward compatible).

    Returns:
        dict: Nested dict structure where leaf values are dicts with 'latency' and 'power' keys.
    """
    if not os.path.exists(generation_attention_file):
        logger.warning(f"Generation attention data file {generation_attention_file} not found.")
        return None
    generation_attention_data = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict()))))
        )
    )
    with open(generation_attention_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Check if power columns exist (backward compatibility)
    has_power = len(rows) > 0 and "power" in rows[0]
    if not has_power:
        logger.debug(f"Legacy database format detected in {generation_attention_file} - power will default to 0.0")

    for row in rows:
        try:
            window_size = row["window_size"]
        except KeyError:
            window_size = 0
        quant_mode, kv_cache_dtype, b, s, n, kv_n, head_size, step, latency = (  # noqa: F841
            row["attn_dtype"],
            row["kv_cache_dtype"],
            row["batch_size"],
            row["isl"],
            row["num_heads"],
            row["num_key_value_heads"],
            row["head_dim"],
            row["step"],
            row["latency"],
        )
        b = int(b)
        s = int(s)
        n = int(n)
        kv_n = int(kv_n)
        head_size = int(head_size)
        window_size = int(window_size)
        step = int(step)
        latency = float(latency)

        # NEW: Read power with backward compatibility
        power = float(row.get("power", 0.0))

        # NEW: Calculate energy from power and latency
        energy = power * latency  # watt-milliseconds

        # we only have kv_n==n(MHA) and kv_n==1,2,4,8(XQA), interp/extrap all other num_kv_heads.
        # Use kv_n = 0 to mean n_kv == n.
        kv_n = 0 if n == kv_n else kv_n
        s = s + step

        kv_cache_dtype = common.KVCacheQuantMode[kv_cache_dtype]

        try:
            # Check for conflict
            generation_attention_data[kv_cache_dtype][kv_n][head_size][window_size][n][b][s]
            logger.debug(
                f"value conflict in generation attention data: {kv_cache_dtype} {kv_n} "
                f"{head_size} {window_size} {n} {b}"
            )
        except KeyError:
            # Store all three values
            generation_attention_data[kv_cache_dtype][kv_n][head_size][window_size][n][b][s] = {
                "latency": latency,
                "power": power,
                "energy": energy,  # NEW: precomputed energy
            }

    return generation_attention_data


def load_context_mla_data(context_mla_file):
    """
    Load the context mla data for trtllm with power support (backward compatible).

    Returns:
        dict: Nested dict structure where leaf values are dicts with 'latency' and 'power' keys.
    """
    if not os.path.exists(context_mla_file):
        logger.warning(f"Context mla data file {context_mla_file} not found.")
        return None
    context_mla_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict()))))

    with open(context_mla_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Check if power columns exist (backward compatibility)
    has_power = len(rows) > 0 and "power" in rows[0]
    if not has_power:
        logger.debug(f"Legacy database format detected in {context_mla_file} - power will default to 0.0")

    for row in rows:
        (
            quant_mode,
            kv_cache_dtype,
            b,
            s,
            latency,
        ) = row["mla_dtype"], row["kv_cache_dtype"], row["batch_size"], row["isl"], row["latency"]

        if "num_heads" not in row:
            tp_size = int(row["tp_size"])
            num_heads = 128 // tp_size
        else:
            num_heads = int(row["num_heads"])

        b = int(b)
        s = int(s)
        latency = float(latency)

        # NEW: Read power with backward compatibility
        power = float(row.get("power", 0.0))

        # NEW: Calculate energy from power and latency
        energy = power * latency  # watt-milliseconds

        quant_mode = common.FMHAQuantMode[quant_mode]
        kv_cache_dtype = common.KVCacheQuantMode[kv_cache_dtype]

        try:
            # Check for conflict
            context_mla_data[quant_mode][kv_cache_dtype][num_heads][s][b]
            logger.debug(f"value conflict in context mla data: {quant_mode} {kv_cache_dtype} {num_heads} {s} {b}")
        except KeyError:
            # Store all three values
            context_mla_data[quant_mode][kv_cache_dtype][num_heads][s][b] = {
                "latency": latency,
                "power": power,
                "energy": energy,  # NEW: precomputed energy
            }

    return context_mla_data


def load_generation_mla_data(generation_mla_file):
    """
    Load the generation mla data for trtllm with power support (backward compatible).

    Returns:
        dict: Nested dict structure where leaf values are dicts with 'latency' and 'power' keys.
    """
    if not os.path.exists(generation_mla_file):
        logger.warning(f"Generation mla data file {generation_mla_file} not found.")
        return None
    generation_mla_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))
    with open(generation_mla_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Check if power columns exist (backward compatibility)
    has_power = len(rows) > 0 and "power" in rows[0]
    if not has_power:
        logger.debug(f"Legacy database format detected in {generation_mla_file} - power will default to 0.0")

    for row in rows:
        quant_mode, kv_cache_dtype, b, s, step, latency = (  # noqa: F841
            row["mla_dtype"],
            row["kv_cache_dtype"],
            row["batch_size"],
            row["isl"],
            row["step"],
            row["latency"],
        )

        if "num_heads" not in row:
            tp_size = int(row["tp_size"])
            num_heads = 128 // tp_size
        else:
            num_heads = int(row["num_heads"])

        b = int(b)
        s = int(s)
        step = int(step)
        latency = float(latency)

        # NEW: Read power with backward compatibility
        power = float(row.get("power", 0.0))

        # NEW: Calculate energy from power and latency
        energy = power * latency  # watt-milliseconds

        s = s + step

        kv_cache_dtype = common.KVCacheQuantMode[kv_cache_dtype]

        try:
            # Check for conflict
            generation_mla_data[kv_cache_dtype][num_heads][b][s]
            logger.debug(f"value conflict in generation mla data: {kv_cache_dtype} {num_heads} {b} {s} ")
        except KeyError:
            # Store all three values
            generation_mla_data[kv_cache_dtype][num_heads][b][s] = {
                "latency": latency,
                "power": power,
                "energy": energy,  # NEW: precomputed energy
            }

    return generation_mla_data


def load_mla_bmm_data(mla_bmm_file):
    """
    Load the mla bmm data for trtllm with power support (backward compatible).

    Returns:
        dict: Nested dict structure where leaf values are dicts with 'latency' and 'power' keys.
    """
    if not os.path.exists(mla_bmm_file):
        logger.warning(f"MLA BMM data file {mla_bmm_file} not found.")
        return None
    mla_bmm_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))

    with open(mla_bmm_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Check if power columns exist (backward compatibility)
    has_power = len(rows) > 0 and "power" in rows[0]
    if not has_power:
        logger.debug(f"Legacy database format detected in {mla_bmm_file} - power will default to 0.0")

    for row in rows:
        quant_mode, num_tokens, num_heads, latency, op_name = (
            row["bmm_dtype"],
            row["num_tokens"],
            row["num_heads"],
            row["latency"],
            row["op_name"],
        )
        num_tokens = int(num_tokens)
        num_heads = int(num_heads)
        latency = float(latency)

        # NEW: Read power with backward compatibility
        power = float(row.get("power", 0.0))

        # NEW: Calculate energy from power and latency
        energy = power * latency  # watt-milliseconds

        quant_mode = common.GEMMQuantMode[quant_mode]

        try:
            # Check for conflict
            mla_bmm_data[quant_mode][op_name][num_heads][num_tokens]
            logger.debug(f"value conflict in mla bmm data: {op_name} {quant_mode} {num_heads} {num_tokens} ")
        except KeyError:
            # Store all three values
            mla_bmm_data[quant_mode][op_name][num_heads][num_tokens] = {
                "latency": latency,
                "power": power,
                "energy": energy,  # NEW: precomputed energy
            }

    return mla_bmm_data


def load_mamba2_data(mamba2_file: str):
    """
    Load Mamba2 Conv1D + SSM kernel performance data from mamba2_perf.txt.

    CSV columns: framework, version, device, op_name, kernel_source, phase,
    batch_size, seq_len, num_tokens, d_model, d_state, d_conv, nheads, head_dim,
    n_groups, chunk_size, model_name, latency (optional: power).
    All rows must have the same columns (context and generation both include
    seq_len and num_tokens so columns align).

    Returns:
        dict: data[kernel_source][phase][model_key] where model_key is
              (d_model, d_state, d_conv, nheads, head_dim, n_groups, chunk_size).
              For phase "context" the leaf is [batch_size][seq_len] -> {latency, power, energy}.
              For phase "generation" the leaf is [batch_size] -> {latency, power, energy}.
              Returns None if file does not exist.
    """
    if not os.path.exists(mamba2_file):
        logger.warning(f"Mamba2 data file {mamba2_file} not found.")
        return None

    # data[kernel_source][phase][model_key] -> nested batch_size [seq_len] -> {latency, power, energy}
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))

    with open(mamba2_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    has_power = len(rows) > 0 and "power" in rows[0]
    if not has_power:
        logger.debug(f"Legacy database format detected in {mamba2_file} - power will default to 0.0")

    for row in rows:
        kernel_source = row["kernel_source"]
        phase = row["phase"]
        batch_size = int(row["batch_size"])
        seq_len = int(row["seq_len"])
        d_model = int(row["d_model"])
        d_state = int(row["d_state"])
        d_conv = int(row["d_conv"])
        nheads = int(row["nheads"])
        head_dim = int(row["head_dim"])
        n_groups = int(row["n_groups"])
        chunk_size = int(row["chunk_size"])
        latency = float(row["latency"])
        power = float(row.get("power", 0.0))
        energy = power * latency

        model_key = (d_model, d_state, d_conv, nheads, head_dim, n_groups, chunk_size)
        entry = {"latency": latency, "power": power, "energy": energy}

        try:
            if phase == "context":
                data[kernel_source][phase][model_key][batch_size][seq_len]
                logger.debug(
                    f"value conflict in mamba2 data: {kernel_source} {phase} {model_key} {batch_size} {seq_len}"
                )
            else:
                data[kernel_source][phase][model_key][batch_size]
                logger.debug(f"value conflict in mamba2 data: {kernel_source} {phase} {model_key} {batch_size}")
        except KeyError:
            if phase == "context":
                data[kernel_source][phase][model_key][batch_size][seq_len] = entry
            else:
                data[kernel_source][phase][model_key][batch_size] = entry

    # Convert default dicts to regular dicts for predictable behavior; keep generation as 1D
    result = {}
    for ks, by_phase in data.items():
        result[ks] = {}
        for ph, by_key in by_phase.items():
            result[ks][ph] = dict(by_key)

    return result


def load_wideep_context_moe_data(wideep_context_moe_file):
    """
    Load the SGLang wideep context MoE data from wideep_context_moe_perf.txt
    with power support (backward compatible).

    Returns:
        dict: Nested dict structure where leaf values are dicts with 'latency' and 'power' keys.
    """
    if not os.path.exists(wideep_context_moe_file):
        logger.warning(f"Context MoE data file {wideep_context_moe_file} not found.")
        return None

    wideep_context_moe_data = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(
                        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))
                    )
                )
            )
        )
    )

    logger.debug(f"Loading SGLang wideep context MoE data from: {wideep_context_moe_file}")
    with open(wideep_context_moe_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

        # Check if power columns exist (backward compatibility)
        has_power = len(rows) > 0 and "power" in rows[0]
        if not has_power:
            logger.debug(f"Legacy database format detected in {wideep_context_moe_file} - power will default to 0.0")

        for row in rows:
            # Parse the CSV format with num_tokens instead of batch_size and input_len
            quant_mode = row["moe_dtype"]
            num_tokens = int(row["num_tokens"])
            hidden_size = int(row["hidden_size"])
            inter_size = int(row["inter_size"])
            topk = int(row["topk"])
            num_experts = int(row["num_experts"])
            moe_tp_size = int(row["moe_tp_size"])
            moe_ep_size = int(row["moe_ep_size"])
            distribution = row["distribution"]
            latency = float(row["latency"])
            quant_mode = common.MoEQuantMode[quant_mode]

            # NEW: Read power with backward compatibility
            power = float(row.get("power", 0.0))

            # NEW: Calculate energy from power and latency
            energy = power * latency  # watt-milliseconds

            # Store all three values
            wideep_context_moe_data[quant_mode][distribution][topk][num_experts][hidden_size][inter_size][moe_tp_size][
                moe_ep_size
            ][num_tokens] = {
                "latency": latency,
                "power": power,
                "energy": energy,  # NEW: precomputed energy
            }
            logger.debug(
                f"Loaded SGLang wideep context MoE data: {quant_mode}, {distribution}, {topk}, "
                f"{num_experts}, {hidden_size}, {inter_size}, {moe_tp_size}, "
                f"{moe_ep_size}, {num_tokens} -> {latency}"
            )

    return wideep_context_moe_data


def load_wideep_generation_moe_data(wideep_generation_moe_file):
    """
    Load the SGLang wideep generation MoE data from wideep_generation_moe_perf.txt
    with power support (backward compatible).

    Returns:
        dict: Nested dict structure where leaf values are dicts with 'latency' and 'power' keys.
    """
    if not os.path.exists(wideep_generation_moe_file):
        logger.warning(f"Generation MoE data file {wideep_generation_moe_file} not found.")
        return None

    wideep_generation_moe_data = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(
                        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))
                    )
                )
            )
        )
    )

    logger.debug(f"Loading SGLang wideep generation MoE data from: {wideep_generation_moe_file}")
    with open(wideep_generation_moe_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

        # Check if power columns exist (backward compatibility)
        has_power = len(rows) > 0 and "power" in rows[0]
        if not has_power:
            logger.debug(f"Legacy database format detected in {wideep_generation_moe_file} - power will default to 0.0")

        for row in rows:
            # Parse the CSV format with num_tokens instead of batch_size and input_len
            quant_mode = row["moe_dtype"]
            num_tokens = int(row["num_tokens"])
            hidden_size = int(row["hidden_size"])
            inter_size = int(row["inter_size"])
            topk = int(row["topk"])
            num_experts = int(row["num_experts"])
            moe_tp_size = int(row["moe_tp_size"])
            moe_ep_size = int(row["moe_ep_size"])
            distribution = row["distribution"]
            latency = float(row["latency"])
            quant_mode = common.MoEQuantMode[quant_mode]

            # NEW: Read power with backward compatibility
            power = float(row.get("power", 0.0))

            # NEW: Calculate energy from power and latency
            energy = power * latency  # watt-milliseconds

            # Store all three values
            wideep_generation_moe_data[quant_mode][distribution][topk][num_experts][hidden_size][inter_size][
                moe_tp_size
            ][moe_ep_size][num_tokens] = {
                "latency": latency,
                "power": power,
                "energy": energy,  # NEW: precomputed energy
            }
            logger.debug(
                f"Loaded SGLang wideep generation MoE data: {quant_mode}, {distribution}, {topk}, "
                f"{num_experts}, {hidden_size}, {inter_size}, {moe_tp_size}, "
                f"{moe_ep_size}, {num_tokens} -> {latency}"
            )

    return wideep_generation_moe_data


def load_wideep_context_mla_data(wideep_context_mla_file):
    """
    Load the SGLang wideep context mla data from wideep_context_mla_perf.txt
    with power support (backward compatible).

    Returns:
        dict: Nested dict structure where leaf values are dicts with 'latency' and 'power' keys.
    """
    if not os.path.exists(wideep_context_mla_file):
        logger.warning(f"SGLang wideep context mla data file {wideep_context_mla_file} not found.")
        return None
    wideep_context_mla_data = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict()))))
    )

    with open(wideep_context_mla_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Check if power columns exist (backward compatibility)
    has_power = len(rows) > 0 and "power" in rows[0]
    if not has_power:
        logger.debug(f"Legacy database format detected in {wideep_context_mla_file} - power will default to 0.0")

    for row in rows:
        (
            quant_mode,
            kv_cache_dtype,
            b,
            s,
            latency,
        ) = row["mla_dtype"], row["kv_cache_dtype"], row["batch_size"], row["isl"], row["latency"]

        kernel_source = row.get("kernel_source", "flashinfer")

        if "num_heads" not in row:
            tp_size = int(row["tp_size"])
            num_heads = 128 // tp_size
        else:
            num_heads = int(row["num_heads"])

        b = int(b)
        s = int(s)
        latency = float(latency)

        # NEW: Read power with backward compatibility
        power = float(row.get("power", 0.0))

        # NEW: Calculate energy from power and latency
        energy = power * latency  # watt-milliseconds

        quant_mode = common.FMHAQuantMode[quant_mode]
        kv_cache_dtype = common.KVCacheQuantMode[kv_cache_dtype]

        try:
            # Check for conflict
            wideep_context_mla_data[kernel_source][quant_mode][kv_cache_dtype][num_heads][s][b]
            logger.debug(
                f"value conflict in context mla data: {kernel_source} {quant_mode} {kv_cache_dtype} {num_heads} {s} {b}"
            )
        except KeyError:
            # Store all three values
            wideep_context_mla_data[kernel_source][quant_mode][kv_cache_dtype][num_heads][s][b] = {
                "latency": latency,
                "power": power,
                "energy": energy,  # NEW: precomputed energy
            }

    return wideep_context_mla_data


def load_wideep_generation_mla_data(wideep_generation_mla_file):
    """
    Load the SGLang wideep generation mla data from wideep_generation_mla_perf.txt
    with power support (backward compatible).

    Returns:
        dict: Nested dict structure where leaf values are dicts with 'latency' and 'power' keys.
    """
    if not os.path.exists(wideep_generation_mla_file):
        logger.warning(f"SGLang wideep generation mla data file {wideep_generation_mla_file} not found.")
        return None
    wideep_generation_mla_data = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))
    )
    with open(wideep_generation_mla_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Check if power columns exist (backward compatibility)
    has_power = len(rows) > 0 and "power" in rows[0]
    if not has_power:
        logger.debug(f"Legacy database format detected in {wideep_generation_mla_file} - power will default to 0.0")

    for row in rows:
        kv_cache_dtype, b, s, step, latency = (
            row["kv_cache_dtype"],
            row["batch_size"],
            row["isl"],
            row["step"],
            row["latency"],
        )

        kernel_source = row.get("kernel_source", "flashinfer")

        if "num_heads" not in row:
            tp_size = int(row["tp_size"])
            num_heads = 128 // tp_size
        else:
            num_heads = int(row["num_heads"])

        b = int(b)
        s = int(s)
        step = int(step)
        latency = float(latency)

        # NEW: Read power with backward compatibility
        power = float(row.get("power", 0.0))

        # NEW: Calculate energy from power and latency
        energy = power * latency  # watt-milliseconds

        s = s + step

        kv_cache_dtype = common.KVCacheQuantMode[kv_cache_dtype]

        try:
            # Check for conflict
            wideep_generation_mla_data[kernel_source][kv_cache_dtype][num_heads][b][s]
            logger.debug(
                f"value conflict in generation mla data: {kernel_source} {kv_cache_dtype} {num_heads} {b} {s} "
            )
        except KeyError:
            # Store all three values
            wideep_generation_mla_data[kernel_source][kv_cache_dtype][num_heads][b][s] = {
                "latency": latency,
                "power": power,
                "energy": energy,  # NEW: precomputed energy
            }

    return wideep_generation_mla_data


def load_wideep_deepep_ll_data(wideep_deepep_ll_file):
    """
    Load the SGLang wideep deepep LL operation data from wideep_deepep_ll_perf.txt
    with power support (backward compatible).

    Returns:
        dict: Nested dict structure where leaf values are dicts with 'latency' and 'power' keys.
    """
    if not os.path.exists(wideep_deepep_ll_file):
        logger.warning(f"SGLang wideep deepep LL operation data file {wideep_deepep_ll_file} not found.")
        return None

    wideep_deepep_ll_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))

    with open(wideep_deepep_ll_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Check if power columns exist (backward compatibility)
    has_power = len(rows) > 0 and "power" in rows[0]
    if not has_power:
        logger.debug(f"Legacy database format detected in {wideep_deepep_ll_file} - power will default to 0.0")

    for row in rows:
        hidden_size = int(row["hidden_size"])
        node_num = int(row["node_num"])
        num_token = int(row["num_token"])
        num_topk = int(row["num_topk"])
        num_experts = int(row["num_experts"])
        combine_avg_t_us = float(row["combine_avg_t_us"])
        dispatch_avg_t_us = float(row["dispatch_avg_t_us"])
        lat = combine_avg_t_us + dispatch_avg_t_us

        # NEW: Read power with backward compatibility
        power = float(row.get("power", 0.0))

        # NEW: Calculate energy from power and latency
        energy = power * lat  # watt-milliseconds

        # Store the data with key structure: [hidden_size][num_topk][num_experts][num_token]
        # -> timing data
        if num_token in wideep_deepep_ll_data[node_num][hidden_size][num_topk][num_experts]:
            logger.debug(
                f"value conflict in SGLang wideep deepep LL operation data: "
                f"{hidden_size} {num_topk} {num_experts} {num_token}"
            )
        else:
            # Store all three values
            wideep_deepep_ll_data[node_num][hidden_size][num_topk][num_experts][num_token] = {
                "latency": lat,
                "power": power,
                "energy": energy,  # NEW: precomputed energy
            }

    return wideep_deepep_ll_data


def load_wideep_deepep_normal_data(wideep_deepep_normal_file):
    """
    Load the SGLang wideep deepep normal operation data from wideep_deepep_normal_perf.txt
    with power support (backward compatible).

    Returns:
        dict: Nested dict structure where leaf values are dicts with 'latency' and 'power' keys.
    """
    if not os.path.exists(wideep_deepep_normal_file):
        logger.warning(f"SGLang wideep deepep normal operation data file {wideep_deepep_normal_file} not found.")
        return None

    wideep_deepep_normal_data = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    )

    with open(wideep_deepep_normal_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Check if power columns exist (backward compatibility)
    has_power = len(rows) > 0 and "power" in rows[0]
    if not has_power:
        logger.debug(f"Legacy database format detected in {wideep_deepep_normal_file} - power will default to 0.0")

    for row in rows:
        num_token = int(row["num_token"])
        topk = int(row["num_topk"])
        node_num = int(row["node_num"])
        num_experts = int(row["num_experts"])
        hidden_size = int(row["hidden_size"])
        dispatch_sms = int(row["dispatch_sms"])
        dispatch_transmit_us = float(row["dispatch_transmit_us"])
        dispatch_notify_us = float(row["dispatch_notify_us"])
        combine_transmit_us = float(row["combine_transmit_us"])
        combine_notify_us = float(row["combine_notify_us"])
        lat = dispatch_transmit_us + dispatch_notify_us + combine_transmit_us + combine_notify_us

        # NEW: Read power with backward compatibility
        power = float(row.get("power", 0.0))

        # NEW: Calculate energy from power and latency
        energy = power * lat  # watt-milliseconds

        # Store the data with key structure:
        # [hidden_size][topk][num_experts][dispatch_sms][num_token] -> timing data
        if num_token in wideep_deepep_normal_data[node_num][hidden_size][topk][num_experts][dispatch_sms]:
            logger.debug(
                f"value conflict in deepep normal data: {hidden_size} {topk} {num_experts} {dispatch_sms} {num_token}"
            )
        else:
            # Store all three values
            wideep_deepep_normal_data[node_num][hidden_size][topk][num_experts][dispatch_sms][num_token] = {
                "latency": lat,
                "power": power,
                "energy": energy,  # NEW: precomputed energy
            }

    return wideep_deepep_normal_data


def load_wideep_moe_compute_data(wideep_moe_compute_file):
    """
    Load the TensorRT-LLM wideep MoE compute data from wideep_moe_compute_perf.txt.
    This data represents pure computation time (excluding All2All communication).

    Returns:
        dict: Nested dict structure where leaf values are dicts with 'latency' and 'power' keys.
        Structure: [kernel_source][quant_mode][distribution][topk][num_experts][hidden_size][inter_size]
                   [num_slots][moe_tp_size][moe_ep_size][num_tokens] -> {latency, power, energy}

    Note:
        kernel_source identifies the MoE computation kernel:
        - "moe_torch_flow": Cutlass-based kernel (default for SM < 100)
        - "deepgemm": DeepGemm kernel (SM >= 100 with fp8_block)
        If data file does not have 'kernel_source' column, it defaults to "moe_torch_flow".
    """
    if not os.path.exists(wideep_moe_compute_file):
        logger.warning(f"TensorRT-LLM wideep MoE compute data file {wideep_moe_compute_file} not found.")
        return None

    wideep_moe_compute_data = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(
                        lambda: defaultdict(
                            lambda: defaultdict(
                                lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))
                            )
                        )
                    )
                )
            )
        )
    )

    logger.debug(f"Loading TensorRT-LLM wideep MoE compute data from: {wideep_moe_compute_file}")
    with open(wideep_moe_compute_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

        # Check if power columns exist (backward compatibility)
        has_power = len(rows) > 0 and "power" in rows[0]
        if not has_power:
            logger.debug(f"Legacy database format detected in {wideep_moe_compute_file} - power will default to 0.0")

        # Check if kernel_source column exists
        has_kernel_source = len(rows) > 0 and "kernel_source" in rows[0]
        if not has_kernel_source:
            logger.debug(
                f"kernel_source column not found in {wideep_moe_compute_file} - will default to 'moe_torch_flow'"
            )

        for row in rows:
            quant_mode = row["moe_dtype"]
            num_tokens = int(row["num_tokens"])
            hidden_size = int(row["hidden_size"])
            inter_size = int(row["inter_size"])
            topk = int(row["topk"])
            num_experts = int(row["num_experts"])
            num_slots = int(row["num_slots"])
            moe_tp_size = int(row["moe_tp_size"])
            moe_ep_size = int(row["moe_ep_size"])
            distribution = row["distribution"]
            latency = float(row["latency"])
            quant_mode = common.MoEQuantMode[quant_mode]

            # Get kernel_source from data or use default
            kernel_source = row.get("kernel_source", "moe_torch_flow")

            # Read power with backward compatibility
            power = float(row.get("power", 0.0))
            energy = power * latency  # watt-milliseconds

            # Store all three values with kernel_source dimension
            wideep_moe_compute_data[kernel_source][quant_mode][distribution][topk][num_experts][hidden_size][
                inter_size
            ][num_slots][moe_tp_size][moe_ep_size][num_tokens] = {
                "latency": latency,
                "power": power,
                "energy": energy,
            }
            # logger.debug(
            #     f"Loaded TensorRT-LLM wideep MoE compute data: kernel={kernel_source}, {quant_mode}, "
            #     f"{distribution}, {topk}, {num_experts}, {hidden_size}, {inter_size}, {num_slots}, "
            #     f"{moe_tp_size}, {moe_ep_size}, {num_tokens} -> {latency}"
            # )

    return wideep_moe_compute_data


def load_wideep_alltoall_data(wideep_alltoall_file):
    """
    Load the TensorRT-LLM wideep All2All data from wideep_alltoall_perf.txt.
    This data represents All2All communication time (prepare, dispatch, combine).

    Returns:
        dict: Nested dict structure where leaf values are dicts with 'latency' and 'power' keys.
        Structure: [kernel_source][op_name][quant_mode][num_nodes][hidden_size][topk][num_experts]
                   [moe_ep_size][num_tokens] -> {latency, power, energy}
        op_name can be: alltoall_prepare, alltoall_dispatch, alltoall_combine, alltoall_combine_low_precision

    Note:
        kernel_source identifies the All2All communication method:
        - "MnnvlMoe": NVLink Two-Sided via MNNVL (GB200, SM >= 100)
        - "DeepEP": DeepEP normal mode (H100/H200, cross-node)
        - "DeepEPLowLatency": DeepEP low-latency mode (H100/H200, intra-node)
        - "NCCL": Standard NCCL communication (fallback)
        If data file does not have 'kernel_source' column, it defaults to "MnnvlMoe".

        If data file does not have 'num_nodes' column, it will be computed as moe_ep_size // 4.
        This assumes 4 GPUs per node (e.g., GB200 NVL4).
    """
    if not os.path.exists(wideep_alltoall_file):
        logger.warning(f"TensorRT-LLM wideep All2All data file {wideep_alltoall_file} not found.")
        return None

    wideep_alltoall_data = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(
                        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))
                    )
                )
            )
        )
    )

    logger.debug(f"Loading TensorRT-LLM wideep All2All data from: {wideep_alltoall_file}")
    with open(wideep_alltoall_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

        # Check if power columns exist (backward compatibility)
        has_power = len(rows) > 0 and "power" in rows[0]
        if not has_power:
            logger.debug(f"Legacy database format detected in {wideep_alltoall_file} - power will default to 0.0")

        # Check if num_nodes column exists
        has_num_nodes = len(rows) > 0 and "num_nodes" in rows[0]
        if not has_num_nodes:
            logger.debug(f"num_nodes column not found in {wideep_alltoall_file} - will be computed as moe_ep_size // 4")

        # Check if kernel_source column exists
        has_kernel_source = len(rows) > 0 and "kernel_source" in rows[0]
        if not has_kernel_source:
            logger.debug(f"kernel_source column not found in {wideep_alltoall_file} - will default to 'MnnvlMoe'")

        for row in rows:
            op_name = row["op_name"]  # alltoall_prepare, alltoall_dispatch, alltoall_combine, etc.
            quant_mode = row["moe_dtype"]
            num_tokens = int(row["num_tokens"])
            hidden_size = int(row["hidden_size"])
            topk = int(row["topk"])
            num_experts = int(row["num_experts"])
            moe_ep_size = int(row["moe_ep_size"])
            latency = float(row["latency"])
            quant_mode = common.MoEQuantMode[quant_mode]

            # Get kernel_source from data or use default
            kernel_source = row.get("kernel_source", "MnnvlMoe")

            # Get num_nodes from data or compute from moe_ep_size
            if has_num_nodes:
                num_nodes = int(row["num_nodes"])
            else:
                # Default: assume 4 GPUs per node
                if moe_ep_size % 4 != 0:
                    logger.warning(
                        f"moe_ep_size={moe_ep_size} is not divisible by 4, using moe_ep_size // 4 = {moe_ep_size // 4}"
                    )
                num_nodes = max(1, moe_ep_size // 4)

            # Read power with backward compatibility
            power = float(row.get("power", 0.0))
            energy = power * latency  # watt-milliseconds

            # Store all three values with kernel_source and num_nodes dimensions
            wideep_alltoall_data[kernel_source][op_name][quant_mode][num_nodes][hidden_size][topk][num_experts][
                moe_ep_size
            ][num_tokens] = {
                "latency": latency,
                "power": power,
                "energy": energy,
            }
            # logger.debug(
            #     f"Loaded TensorRT-LLM wideep All2All data: kernel={kernel_source}, {op_name}, {quant_mode}, "
            #     f"num_nodes={num_nodes}, {hidden_size}, {topk}, {num_experts}, {moe_ep_size}, "
            #     f"{num_tokens} -> {latency}"
            # )

    return wideep_alltoall_data


class PerfDatabase:
    """
    The perf database for a given system, backend and version

    Attributes:
        system (str): the system name
        backend (str): the backend name
        version (str): the version name
        system_spec (dict): the system spec
        _default_database_mode (common.DatabaseMode): the default mode of the database
        _gemm_data (dict): the gemm data
        _context_attention_data (dict): the context attention data
        _generation_attention_data (dict): the generation attention data
        _custom_allreduce_data (dict): the custom allreduce data
        _moe_data (dict): the moe data
        _context_mla_data (dict): the context mla data
        _generation_mla_data (dict): the generation mla data
        _nccl_data (dict): the nccl data
        _mla_bmm_data (dict): the mla bmm data
        SGLang wideep:
        _wideep_context_moe_data (dict): the wideep context moe data
        _wideep_generation_moe_data (dict): the wideep generation moe data
        _wideep_context_mla_data (dict): the wideep context mla data
        _wideep_generation_mla_data (dict): the wideep generation mla data
        _wideep_deepep_normal_data (dict): the wideep deepep normal data
        _wideep_deepep_ll_data (dict): the wideep deepep ll data
        TensorRT-LLM wideep:
        _wideep_moe_compute_data (dict): the wideep moe compute data (pure computation, no all2all)
        _wideep_alltoall_data (dict): the wideep all2all data (prepare, dispatch, combine)

    Methods:
        query_gemm: query the gemm data
        query_context_attention: query the context attention data
        query_generation_attention: query the generation attention data
        query_context_mla: query the context mla data
        query_generation_mla: query the generation mla data
        query_nccl: query the nccl data
        query_mla_bmm: query the mla bmm data
        query_mem_op: query the mem op data
        query_p2p: query the p2p data
        query_custom_allreduce: query the custom allreduce data
        query_moe: query the moe data
    """

    def __init__(self, system: str, backend: str, version: str, systems_root: str = "./systems") -> None:
        """
        Initialize the perf database
        """
        self.system = system
        self.backend = backend
        self.version = version
        self.systems_root = systems_root
        with open(os.path.join(systems_root, system + ".yaml")) as f:
            self.system_spec = yaml.load(f, Loader=yaml.SafeLoader)
        self._default_database_mode = common.DatabaseMode.SILICON  # default mode is SILICON

        # Cache for extracted metric data to avoid repeated extraction in _interp_3d
        self._extracted_metrics_cache = {}

        data_dir = os.path.join(systems_root, self.system_spec["data_dir"], backend, version)
        nccl_data_dir = os.path.join(
            systems_root,
            self.system_spec["data_dir"],
            "nccl",
            self.system_spec["misc"]["nccl_version"],
            common.PerfDataFilename.nccl.value,
        )

        if backend == "sglang":
            # For SGLang, only load MoE data and provide empty structures for other data
            # regular path
            self._gemm_data = load_gemm_data(os.path.join(data_dir, common.PerfDataFilename.gemm.value))
            self._context_attention_data = load_context_attention_data(
                os.path.join(data_dir, common.PerfDataFilename.context_attention.value)
            )
            self._generation_attention_data = load_generation_attention_data(
                os.path.join(data_dir, common.PerfDataFilename.generation_attention.value)
            )
            self._moe_data, self._moe_low_latency_data = load_moe_data(
                os.path.join(data_dir, common.PerfDataFilename.moe.value)
            )
            self._context_mla_data = load_context_mla_data(
                os.path.join(data_dir, common.PerfDataFilename.context_mla.value)
            )
            self._generation_mla_data = load_generation_mla_data(
                os.path.join(data_dir, common.PerfDataFilename.generation_mla.value)
            )
            self._custom_allreduce_data = load_custom_allreduce_data(
                os.path.join(data_dir, common.PerfDataFilename.custom_allreduce.value)
            )
            self._nccl_data = load_nccl_data(nccl_data_dir)
            self._mla_bmm_data = load_mla_bmm_data(os.path.join(data_dir, common.PerfDataFilename.mla_bmm.value))

            # wideep path
            self._wideep_context_moe_data = load_wideep_context_moe_data(
                os.path.join(data_dir, common.PerfDataFilename.wideep_context_moe.value)
            )
            self._wideep_generation_moe_data = load_wideep_generation_moe_data(
                os.path.join(data_dir, common.PerfDataFilename.wideep_generation_moe.value)
            )
            self._wideep_context_mla_data = load_wideep_context_mla_data(
                os.path.join(data_dir, common.PerfDataFilename.wideep_context_mla.value)
            )
            self._wideep_generation_mla_data = load_wideep_generation_mla_data(
                os.path.join(data_dir, common.PerfDataFilename.wideep_generation_mla.value)
            )
            self._wideep_deepep_normal_data = load_wideep_deepep_normal_data(
                os.path.join(data_dir, common.PerfDataFilename.wideep_deepep_normal.value)
            )
            self._wideep_deepep_ll_data = load_wideep_deepep_ll_data(
                os.path.join(data_dir, common.PerfDataFilename.wideep_deepep_ll.value)
            )
            self._compute_scale_data = None
            self._scale_matrix_data = None
        elif backend == "vllm":
            self._gemm_data = load_gemm_data(os.path.join(data_dir, common.PerfDataFilename.gemm.value))
            self._context_attention_data = load_context_attention_data(
                os.path.join(data_dir, common.PerfDataFilename.context_attention.value)
            )
            self._generation_attention_data = load_generation_attention_data(
                os.path.join(data_dir, common.PerfDataFilename.generation_attention.value)
            )
            self._custom_allreduce_data = load_custom_allreduce_data(
                os.path.join(data_dir, common.PerfDataFilename.custom_allreduce.value)
            )
            self._nccl_data = load_nccl_data(nccl_data_dir)
            self._moe_data, _ = load_moe_data(os.path.join(data_dir, common.PerfDataFilename.moe.value))
            self._mla_bmm_data = None
            self._context_mla_data = load_context_mla_data(
                os.path.join(data_dir, common.PerfDataFilename.context_mla.value)
            )
            self._generation_mla_data = load_generation_mla_data(
                os.path.join(data_dir, common.PerfDataFilename.generation_mla.value)
            )
            self._compute_scale_data = None
            self._scale_matrix_data = None
        else:  # TRTLLM
            self._gemm_data = load_gemm_data(os.path.join(data_dir, common.PerfDataFilename.gemm.value))
            self._context_attention_data = load_context_attention_data(
                os.path.join(data_dir, common.PerfDataFilename.context_attention.value)
            )
            self._generation_attention_data = load_generation_attention_data(
                os.path.join(data_dir, common.PerfDataFilename.generation_attention.value)
            )
            self._custom_allreduce_data = load_custom_allreduce_data(
                os.path.join(data_dir, common.PerfDataFilename.custom_allreduce.value)
            )
            self._moe_data, self._moe_low_latency_data = load_moe_data(
                os.path.join(data_dir, common.PerfDataFilename.moe.value)
            )
            self._context_mla_data = load_context_mla_data(
                os.path.join(data_dir, common.PerfDataFilename.context_mla.value)
            )
            self._generation_mla_data = load_generation_mla_data(
                os.path.join(data_dir, common.PerfDataFilename.generation_mla.value)
            )
            self._nccl_data = load_nccl_data(nccl_data_dir)
            self._mla_bmm_data = load_mla_bmm_data(os.path.join(data_dir, common.PerfDataFilename.mla_bmm.value))
            self._mamba2_data = load_mamba2_data(os.path.join(data_dir, common.PerfDataFilename.mamba2.value))
            self._compute_scale_data = load_compute_scale_data(
                os.path.join(data_dir, common.PerfDataFilename.compute_scale.value)
            )
            self._scale_matrix_data = load_scale_matrix_data(
                os.path.join(data_dir, common.PerfDataFilename.scale_matrix.value)
            )

            # TensorRT-LLM wideep path
            self._wideep_moe_compute_data = load_wideep_moe_compute_data(
                os.path.join(data_dir, common.PerfDataFilename.wideep_moe_compute.value)
            )
            self._wideep_alltoall_data = load_wideep_alltoall_data(
                os.path.join(data_dir, common.PerfDataFilename.wideep_alltoall.value)
            )

        # pre-correction
        self._correct_data()

        # regular context attention
        if self._context_attention_data is not None:
            for quant_mode in self._context_attention_data:
                for kv_cache_dtype in self._context_attention_data[quant_mode]:
                    for num_kv_heads in self._context_attention_data[quant_mode][kv_cache_dtype]:
                        for head_size in self._context_attention_data[quant_mode][kv_cache_dtype][num_kv_heads]:
                            for window_size in self._context_attention_data[quant_mode][kv_cache_dtype][num_kv_heads][
                                head_size
                            ]:
                                data_dict = self._context_attention_data[quant_mode][kv_cache_dtype][num_kv_heads][
                                    head_size
                                ][window_size]
                                min_x = min(data_dict.keys())
                                target_x_list = [
                                    1,
                                    2,
                                    3,
                                    4,
                                    5,
                                    6,
                                    8,
                                    9,
                                    10,
                                    12,
                                    14,
                                    16,
                                    18,
                                    20,
                                    24,
                                    28,
                                    32,
                                    36,
                                    40,
                                    48,
                                    56,
                                    72,
                                    96,
                                    128,
                                ]  # n
                                # currently, support max seq to 1M. Because all the system is linear for
                                # now. it will be difficult to do square interpolation. Use more points
                                # to do the approximation.
                                # Note: start from 1 to make sure any small ISL can be interpolated,
                                # even if the ISL is smaller than what exists in the collected data.
                                target_y_list = (
                                    [1, 16, 32, 64, 128, 256, 512, 1024, 2048]
                                    + [4096 + i * 2048 for i in range(14)]
                                    + [32768 + 16384 * i for i in range(6)]
                                    + [131072 + 32768 * i for i in range(12)]
                                    + [524288 + 65536 * i for i in range(9)]
                                )  # s
                                target_z_list = [
                                    1,
                                    2,
                                    4,
                                    8,
                                    16,
                                    32,
                                    64,
                                    128,
                                    256,
                                    512,
                                    384,
                                    1024,
                                    2048,
                                ]  # b

                                filtered_x_list = []
                                for i in target_x_list:
                                    if i >= min_x:
                                        filtered_x_list.append(i)
                                self._extrapolate_data_grid(
                                    data_dict=data_dict,  # nsb
                                    target_x_list=filtered_x_list,
                                    target_y_list=target_y_list,
                                    target_z_list=target_z_list,
                                    sqrt_y_value=True,
                                )

        # regular generation attention
        if self._generation_attention_data is not None:
            for kv_cache_dtype in self._generation_attention_data:
                for num_kv_heads in self._generation_attention_data[kv_cache_dtype]:
                    for head_size in self._generation_attention_data[kv_cache_dtype][num_kv_heads]:
                        for window_size in self._generation_attention_data[kv_cache_dtype][num_kv_heads][head_size]:
                            target_x_list = [
                                1,
                                2,
                                3,
                                4,
                                5,
                                6,
                                8,
                                9,
                                10,
                                12,
                                14,
                                16,
                                18,
                                20,
                                24,
                                28,
                                32,
                                36,
                                40,
                                48,
                                56,
                                72,
                                96,
                                128,
                            ]  # n
                            target_y_list = [
                                1,
                                2,
                                4,
                                8,
                                16,
                                32,
                                64,
                                128,
                                256,
                                384,
                                512,
                                1024,
                                2048,
                                8192,
                            ]  # b
                            target_z_list = [
                                1,
                                2,
                                4,
                                8,
                                16,
                                32,
                                64,
                                128,
                                256,
                                512,
                                1024,
                                2048,
                                4096,
                                8192,
                                16384,
                                32768,
                                65536,
                                131072,
                                262144,
                                2097152 * 8,
                            ]  # s
                            data_dict = self._generation_attention_data[kv_cache_dtype][num_kv_heads][head_size][
                                window_size
                            ]
                            min_x = min(data_dict.keys())
                            filtered_x_list = []
                            for i in target_x_list:
                                if i >= min_x:
                                    filtered_x_list.append(i)

                            self._extrapolate_data_grid(
                                data_dict=data_dict,  # nbs
                                target_x_list=filtered_x_list,
                                target_y_list=target_y_list,
                                target_z_list=target_z_list,
                            )

        # regular gemm
        if self._gemm_data is not None:
            for quant_mode, data_dict in self._gemm_data.items():
                target_x_list = [
                    1,
                    2,
                    4,
                    8,
                    16,
                    32,
                    48,
                    64,
                    80,
                    96,
                    128,
                    160,
                    192,
                    224,
                    256,
                    320,
                    384,
                    448,
                    512,
                    640,
                    768,
                    896,
                    1024,
                    2048,
                    4096,
                    8192,
                    16384,
                    32768,
                    131072,
                    524288,
                    1048576,
                    2097152 * 8,
                ]  # num_tokens
                target_y_list = [
                    32,
                    64,
                    128,
                    256,
                    512,
                    768,
                    1024,
                    1536,
                    2048,
                    2560,
                    3072,
                    3584,
                    4096,
                    5120,
                    6144,
                    7168,
                    8192,
                    10240,
                    12288,
                    14336,
                    16384,
                    20480,
                    24576,
                    28672,
                    32768,
                    40960,
                    49152,
                    57344,
                    65536,
                    131072,
                    262144,
                ]  # to fit vocab gemm
                target_z_list = target_y_list
                self._extrapolate_data_grid(
                    data_dict=data_dict,
                    target_x_list=target_x_list,
                    target_y_list=target_y_list,
                    target_z_list=target_z_list,
                )

        # mla
        # wideep context mla
        if getattr(self, "_wideep_context_mla_data", None) is not None:
            for kernel_source in self._wideep_context_mla_data:
                for quant_mode in self._wideep_context_mla_data[kernel_source]:
                    for kv_cache_dtype in self._wideep_context_mla_data[kernel_source][quant_mode]:
                        num_heads_list = list(
                            self._wideep_context_mla_data[kernel_source][quant_mode][kv_cache_dtype].keys()
                        )
                        data_dict = self._wideep_context_mla_data[kernel_source][quant_mode][kv_cache_dtype]
                        target_x_list = num_heads_list  # to reuse x dim
                        # currently, support max seq to 1M.
                        # Because all the system is linear for now.
                        # it will be difficult to do square interpolation.
                        # Use more points to do the approximation
                        target_y_list = (
                            [16, 32, 64, 128, 256, 512, 1024, 2048]
                            + [4096 + i * 2048 for i in range(14)]
                            + [32768 + 16384 * i for i in range(6)]
                            + [131072 + 32768 * i for i in range(12)]
                            + [524288 + 65536 * i for i in range(9)]
                        )  # s
                        target_z_list = [
                            1,
                            2,
                            4,
                            8,
                            16,
                            32,
                            64,
                            128,
                            256,
                            384,
                            512,
                            1024,
                            2048,
                        ]  # b

                        self._extrapolate_data_grid(
                            data_dict=data_dict,  # tpsize,sb
                            target_x_list=target_x_list,
                            target_y_list=target_y_list,
                            target_z_list=target_z_list,
                            sqrt_y_value=True,
                        )

        # regular context mla
        if self._context_mla_data is not None:
            for quant_mode in self._context_mla_data:
                for kv_cache_dtype in self._context_mla_data[quant_mode]:
                    num_heads_list = list(self._context_mla_data[quant_mode][kv_cache_dtype].keys())
                    data_dict = self._context_mla_data[quant_mode][kv_cache_dtype]
                    target_x_list = num_heads_list  # to reuse x dim
                    # currently, support max seq to 1M. Because all the system is linear for now.
                    # it will be difficult to do square interpolation.
                    # Use more points to do the approximation
                    target_y_list = (
                        [1, 16, 32, 64, 128, 256, 512, 1024, 2048]
                        + [4096 + i * 2048 for i in range(14)]
                        + [32768 + 16384 * i for i in range(6)]
                        + [131072 + 32768 * i for i in range(12)]
                        + [524288 + 65536 * i for i in range(9)]
                    )  # s
                    target_z_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 1024, 2048]  # b

                    self._extrapolate_data_grid(
                        data_dict=data_dict,  # tpsize,sb
                        target_x_list=target_x_list,
                        target_y_list=target_y_list,
                        target_z_list=target_z_list,
                        sqrt_y_value=True,
                    )
        # wideep generation mla
        if getattr(self, "_wideep_generation_mla_data", None) is not None:
            for kernel_source in self._wideep_generation_mla_data:
                for kv_cache_dtype in self._wideep_generation_mla_data[kernel_source]:
                    tp_list = list(self._wideep_generation_mla_data[kernel_source][kv_cache_dtype].keys())
                    data_dict = self._wideep_generation_mla_data[kernel_source][kv_cache_dtype]
                    target_x_list = tp_list  # n
                    target_y_list = [
                        1,
                        2,
                        4,
                        8,
                        16,
                        32,
                        64,
                        128,
                        256,
                        384,
                        512,
                        1024,
                        2048,
                        8192,
                    ]  # b
                    target_z_list = [
                        1,
                        2,
                        4,
                        8,
                        16,
                        32,
                        64,
                        128,
                        256,
                        512,
                        1024,
                        2048,
                        4096,
                        8192,
                        16384,
                        32768,
                        65536,
                        131072,
                        262144,
                        2097152 * 8,
                    ]  # s

                    self._extrapolate_data_grid(
                        data_dict=data_dict,  # tpsize, bs
                        target_x_list=target_x_list,
                        target_y_list=target_y_list,
                        target_z_list=target_z_list,
                    )

        # regular generation mla
        if self._generation_mla_data is not None:
            for kv_cache_dtype in self._generation_mla_data:
                tp_list = list(self._generation_mla_data[kv_cache_dtype].keys())
                data_dict = self._generation_mla_data[kv_cache_dtype]
                target_x_list = tp_list  # n
                target_y_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 1024, 2048, 8192]  # b
                target_z_list = [
                    1,
                    2,
                    4,
                    8,
                    16,
                    32,
                    64,
                    128,
                    256,
                    512,
                    1024,
                    2048,
                    4096,
                    8192,
                    16384,
                    32768,
                    65536,
                    131072,
                    262144,
                    2097152 * 8,
                ]  # s

                self._extrapolate_data_grid(
                    data_dict=data_dict,  # tpsize, bs
                    target_x_list=target_x_list,
                    target_y_list=target_y_list,
                    target_z_list=target_z_list,
                )

        # post-correction
        self._correct_data()

        self._update_support_matrix()

    def _update_support_matrix(self):
        """
        Update the support matrix
        """

        def _enum_key_names(data: dict | None) -> list[str]:
            """
            Safely extract Enum key names from a mapping.

            Many perf tables are optional and loaders return None when data files
            are missing. Treat missing/empty tables as supporting no modes.
            """
            if not data:
                return []
            names: list[str] = []
            for key in data:
                names.append(key.name if hasattr(key, "name") else str(key))
            return names

        # For sglang backend, context_mla_data and generation_mla_data have kernel_source as first
        # level
        # We need to collect quant_modes from the nested structure
        if self.backend == "sglang":
            wideep_context_mla_modes = set()
            for kernel_source in self._wideep_context_mla_data or {}:
                for quant_mode in (self._wideep_context_mla_data or {})[kernel_source]:
                    wideep_context_mla_modes.add(quant_mode.name)

            wideep_generation_mla_modes = set()
            for kernel_source in self._wideep_generation_mla_data or {}:
                for kv_cache_dtype in (self._wideep_generation_mla_data or {})[kernel_source]:
                    wideep_generation_mla_modes.add(kv_cache_dtype.name)

            self.supported_quant_mode = {
                "gemm": _enum_key_names(getattr(self, "_gemm_data", None)),
                "context_attention": _enum_key_names(getattr(self, "_context_attention_data", None)),
                "generation_attention": _enum_key_names(getattr(self, "_generation_attention_data", None)),
                "context_mla": _enum_key_names(getattr(self, "_context_mla_data", None)),
                "generation_mla": _enum_key_names(getattr(self, "_generation_mla_data", None)),
                "mla_bmm": _enum_key_names(getattr(self, "_mla_bmm_data", None)),
                "nccl": _enum_key_names(getattr(self, "_nccl_data", None)),
                "moe": _enum_key_names(getattr(self, "_moe_data", None)),
                "wideep_context_moe": _enum_key_names(getattr(self, "_wideep_context_moe_data", None)),
                "wideep_generation_moe": _enum_key_names(getattr(self, "_wideep_generation_moe_data", None)),
                "wideep_context_mla": list(wideep_context_mla_modes),
                "wideep_generation_mla": list(wideep_generation_mla_modes),
            }
        elif self.backend == "trtllm":
            self.supported_quant_mode = {
                "gemm": _enum_key_names(getattr(self, "_gemm_data", None)),
                "context_attention": _enum_key_names(getattr(self, "_context_attention_data", None)),
                "generation_attention": _enum_key_names(getattr(self, "_generation_attention_data", None)),
                "context_mla": _enum_key_names(getattr(self, "_context_mla_data", None)),
                "generation_mla": _enum_key_names(getattr(self, "_generation_mla_data", None)),
                "mla_bmm": _enum_key_names(getattr(self, "_mla_bmm_data", None)),
                "nccl": _enum_key_names(getattr(self, "_nccl_data", None)),
                "moe": _enum_key_names(getattr(self, "_moe_data", None)),
            }
            # `fp8_static` is a behavioral mode that reuses `fp8` GEMM perf tables.
            gemm_modes = self.supported_quant_mode.get("gemm", []) or []
            if common.GEMMQuantMode.fp8.name in gemm_modes and common.GEMMQuantMode.fp8_static.name not in gemm_modes:
                gemm_modes.append(common.GEMMQuantMode.fp8_static.name)
        elif self.backend == "vllm":  # TODO: deepseek
            self.supported_quant_mode = {
                "gemm": _enum_key_names(getattr(self, "_gemm_data", None)),
                "context_attention": _enum_key_names(getattr(self, "_context_attention_data", None)),
                "generation_attention": _enum_key_names(getattr(self, "_generation_attention_data", None)),
                "nccl": _enum_key_names(getattr(self, "_nccl_data", None)),
                "context_mla": [],
                "generation_mla": [],
                "mla_bmm": [],
                "moe": _enum_key_names(getattr(self, "_moe_data", None)),
            }

    def is_inter_node(self, num_gpus: int) -> bool:
        """
        Check if the number of GPUs is an inter node
        """
        return num_gpus > self.system_spec["node"]["num_gpus_per_node"]

    def _select_alltoall_kernel(
        self,
        quant_mode: common.MoEQuantMode,
        moe_ep_size: int,
        topk: int,
    ) -> str:
        """
        Automatically select All2All communication method based on GPU architecture and configuration.

        Selection logic (based on TensorRT-LLM's select_alltoall_method_type):
        1. SM >= 100 (GB200 with MNNVL support) -> MnnvlMoe (NVLink Two-Sided)
        2. SM >= 90 (H100/H200) with cross-node -> DeepEP
        3. SM >= 90 (H100/H200) within single node -> DeepEPLowLatency
        4. SM < 90 -> NCCL (fallback)

        Args:
            quant_mode: MoE quantization mode
            moe_ep_size: MoE expert parallelism size
            topk: Number of experts activated per token

        Returns:
            str: The selected kernel_source name
        """
        sm_version = self.system_spec["gpu"]["sm_version"]
        num_gpus_per_node = self.system_spec["node"]["num_gpus_per_node"]
        is_inter_node = moe_ep_size > num_gpus_per_node

        # Preferred kernel based on hardware
        if sm_version >= 100:
            # GB200 supports MNNVL (Multi-Node NVLink)
            preferred = "MnnvlMoe"
        elif sm_version >= 90:
            # H100/H200: Check DeepEP feasibility
            # DeepEP requires: tp_size == 1, ep_size > 1, top_k in supported range
            deepep_feasible = moe_ep_size > 1 and topk <= 8

            if deepep_feasible and is_inter_node:
                # Cross-node: use DeepEP normal mode
                preferred = "DeepEP"
            elif deepep_feasible:
                # Intra-node: use DeepEP low-latency mode
                preferred = "DeepEPLowLatency"
            else:
                preferred = "MnnvlMoe"
        else:
            # SM89 and below: use NCCL standard communication
            preferred = "NCCL"

        # Check if preferred kernel is available in data, otherwise fallback
        if self._wideep_alltoall_data is not None:
            available_kernels = list(self._wideep_alltoall_data.keys())
            if preferred in available_kernels:
                return preferred
            elif available_kernels:
                # Fallback to any available kernel
                fallback = available_kernels[0]
                logger.debug(f"Preferred All2All kernel '{preferred}' not available, falling back to '{fallback}'")
                return fallback

        return preferred

    def _select_moe_kernel(
        self,
        quant_mode: common.MoEQuantMode,
    ) -> str:
        """
        Automatically select MoE computation kernel based on GPU architecture and quantization mode.

        Selection logic (based on TensorRT-LLM's MoEOpSelector.select_op):
        1. SM >= 100 (Blackwell) with fp8_block -> deepgemm (DeepGemm kernel)
        2. Otherwise -> moe_torch_flow (Cutlass kernel)

        Args:
            quant_mode: MoE quantization mode

        Returns:
            str: The selected kernel_source name
        """
        sm_version = self.system_spec["gpu"]["sm_version"]
        is_blackwell = sm_version >= 100

        # Convert quant_mode to string for comparison if needed
        quant_mode_str = quant_mode.name if hasattr(quant_mode, "name") else str(quant_mode)
        is_fp8_block = "fp8_block" in quant_mode_str

        # Preferred kernel based on hardware and quant mode
        if is_blackwell and is_fp8_block:
            # Blackwell + FP8 block scales -> DeepGemm kernel
            preferred = "deepgemm"
        else:
            # Default: Cutlass kernel
            preferred = "moe_torch_flow"

        # Check if preferred kernel is available in data, otherwise fallback
        if self._wideep_moe_compute_data is not None:
            available_kernels = list(self._wideep_moe_compute_data.keys())
            if preferred in available_kernels:
                return preferred
            elif available_kernels:
                # Fallback to any available kernel
                fallback = available_kernels[0]
                logger.debug(f"Preferred MoE kernel '{preferred}' not available, falling back to '{fallback}'")
                return fallback

        return preferred

    def _get_value(self, data_value, metric: str = "latency"):
        """
        Extract a metric from a data value (handles both dict and float formats).

        Args:
            data_value: Either a dict {"latency": float, "power": float} or a float (legacy)
            metric: Which metric to extract ("latency" or "power")

        Returns:
            float: The requested metric value
        """
        if isinstance(data_value, dict):
            return data_value.get(metric, 0.0)
        else:
            # Legacy format: raw float is latency, power is 0
            return data_value if metric == "latency" else 0.0

    def _extract_metric_data_3d(self, data: dict, metric: str) -> dict:
        """
        Extract a specific metric from 3D dict-based data structure.

        Converts {k1: {k2: {k3: {"latency": l, "power": p}}}}
        to      {k1: {k2: {k3: l}}} or {k1: {k2: {k3: p}}}

        Args:
            data: Nested 3-level dict where leaf values are dicts or floats
            metric: Which metric to extract ("latency" or "power")

        Returns:
            dict: Same structure but with scalar leaf values
        """
        result = {}
        for k1, v1 in data.items():
            result[k1] = {}
            for k2, v2 in v1.items():
                result[k1][k2] = {}
                for k3, v3 in v2.items():
                    result[k1][k2][k3] = self._get_value(v3, metric)
        return result

    def _extract_latency_and_energy_2d(self, data: dict) -> tuple[dict, dict]:
        """
        Extract both latency and energy from 2D dict-based data structure in a single pass.

        Args:
            data: Nested 2-level dict where leaf values are dicts {"latency": l, "power": p, "energy": e}

        Returns:
            tuple: (latency_data, energy_data) - two dicts with same structure but scalar values
        """
        latency_result = {}
        energy_result = {}

        for k1, v1 in data.items():
            latency_result[k1] = {}
            energy_result[k1] = {}

            for k2, v2 in v1.items():
                latency_result[k1][k2] = self._get_value(v2, "latency")
                energy_result[k1][k2] = self._get_value(v2, "energy")

        return latency_result, energy_result

    def _extract_latency_and_energy_3d(self, data: dict) -> tuple[dict, dict]:
        """
        Extract both latency and energy from 3D dict-based data structure in a single pass.

        This is more efficient than calling _extract_metric_data_3d twice.

        Args:
            data: Nested 3-level dict where leaf values are dicts {"latency": l, "power": p, "energy": e}

        Returns:
            tuple: (latency_data, energy_data) - two dicts with same structure but scalar values
        """
        latency_result = {}
        energy_result = {}

        for k1, v1 in data.items():
            latency_result[k1] = {}
            energy_result[k1] = {}

            for k2, v2 in v1.items():
                latency_result[k1][k2] = {}
                energy_result[k1][k2] = {}

                for k3, v3 in v2.items():
                    latency_result[k1][k2][k3] = self._get_value(v3, "latency")
                    energy_result[k1][k2][k3] = self._get_value(v3, "energy")

        return latency_result, energy_result

    def _extrapolate_data_grid(
        self,
        data_dict: dict[int, dict[int, dict[int, float]]],
        target_x_list: list[int],
        target_y_list: list[int],
        target_z_list: list[int],
        sqrt_y_value: bool = False,
    ) -> None:
        """
        Extrapolate the data grid, we extrapolate the data grid at the initialization stage.
        Future query will based on interpolation.
        """
        x_list = sorted(data_dict.keys())
        for x in x_list:
            # z_direction
            for y in sorted(data_dict[x].keys()):
                z_dict = data_dict[x][y]
                if len(z_dict) <= 1:
                    logger.warning(
                        f"only one data point for a given xy, might trigger error. "
                        f"Please revisit data collection. {x=}, {y=}, {z_dict=}"
                    )
                    continue
                for z in target_z_list:
                    if z not in z_dict:
                        z_left, z_right = self._nearest_1d_point_helper(z, list(z_dict.keys()), False)
                        # Check if both left and right boundaries exist
                        if z_left not in z_dict or z_right not in z_dict:
                            logger.warning(
                                f"Skipping interpolation for z={z} as boundaries z_left={z_left} "
                                f"or z_right={z_right} do not exist in z_dict for x={x}, y={y}"
                            )
                            continue
                        value = self._interp_1d(
                            [z_left, z_right],
                            [data_dict[x][y][z_left], data_dict[x][y][z_right]],
                            z,
                        )
                        z_dict[z] = value

            # y_direction
            for y in target_y_list:
                if y not in data_dict[x]:
                    y_left, y_right = self._nearest_1d_point_helper(y, list(data_dict[x].keys()), False)
                    # Check if both left and right boundaries exist
                    if y_left not in data_dict[x] or y_right not in data_dict[x]:
                        logger.warning(
                            f"Skipping interpolation for y={y} as boundaries y_left={y_left} "
                            f"or y_right={y_right} do not exist in data_dict[{x}]"
                        )
                        continue

                    z_list = sorted(data_dict[x][y_left].keys())
                    for z in z_list:
                        # Check if z exists in both y_left and y_right
                        if z not in data_dict[x][y_left] or z not in data_dict[x][y_right]:
                            logger.warning(
                                f"Skipping interpolation for z={z} as it does not exist in both "
                                f"y_left={y_left} and y_right={y_right}"
                            )
                            continue

                        y_left_value = data_dict[x][y_left][z]
                        y_right_value = data_dict[x][y_right][z]
                        assert y_right_value is not None, "y_right_value cannot be None"
                        if sqrt_y_value:
                            if isinstance(y_left_value, dict):
                                # Handle dict format: apply sqrt to both latency and power
                                y_left_value = {
                                    "latency": math.sqrt(y_left_value["latency"]),
                                    "power": math.sqrt(y_left_value["power"]) if y_left_value["power"] > 0 else 0.0,
                                }
                                y_right_value = {
                                    "latency": math.sqrt(y_right_value["latency"]),
                                    "power": math.sqrt(y_right_value["power"]) if y_right_value["power"] > 0 else 0.0,
                                }
                            else:
                                # Handle legacy float format
                                y_left_value = math.sqrt(y_left_value)
                                y_right_value = math.sqrt(y_right_value)
                        value = self._interp_1d([y_left, y_right], [y_left_value, y_right_value], y)
                        if sqrt_y_value:
                            if isinstance(value, dict):
                                # Square both latency and power
                                value = {
                                    "latency": value["latency"] * value["latency"],
                                    "power": value["power"] * value["power"],
                                }
                            else:
                                value = value * value

                        if y not in data_dict[x]:
                            data_dict[x][y] = {z: value}
                        else:
                            data_dict[x][y][z] = value

        for x in target_x_list:
            if x not in data_dict:
                x_left, x_right = self._nearest_1d_point_helper(x, list(data_dict.keys()), False)
                # Check if both left and right boundaries exist
                if x_left not in data_dict or x_right not in data_dict:
                    logger.warning(
                        f"Skipping interpolation for x={x} as boundaries x_left={x_left} "
                        f"or x_right={x_right} do not exist in data_dict"
                    )
                    continue

                for y in sorted(data_dict[x_left].keys()):
                    # Check if y exists in both x_left and x_right
                    if y not in data_dict[x_left] or y not in data_dict[x_right]:
                        logger.warning(
                            f"Skipping interpolation for y={y} as it does not exist in both "
                            f"x_left={x_left} and x_right={x_right}"
                        )
                        continue

                    for z in sorted(data_dict[x_left][y].keys()):
                        # Check if z exists in both x_left and x_right for the given y
                        if z not in data_dict[x_left][y] or z not in data_dict[x_right][y]:
                            logger.warning(
                                f"Skipping interpolation for z={z} as it does not exist in both "
                                f"x_left={x_left} and x_right={x_right} for y={y}"
                            )
                            continue

                        x_left_value = data_dict[x_left][y][z]
                        x_right_value = data_dict[x_right][y][z]
                        assert x_right_value is not None, "x_right_value cannot be None"
                        value = self._interp_1d([x_left, x_right], [x_left_value, x_right_value], x)
                        if x not in data_dict:
                            data_dict[x] = {y: {z: value}}
                        elif y not in data_dict[x]:
                            data_dict[x][y] = {z: value}
                        else:
                            data_dict[x][y][z] = value

    def _nearest_1d_point_helper(self, x: int, values: list[int], inner_only: bool = True) -> tuple[int, int]:
        """
        Find the nearest 1d point
        """
        assert values is not None and len(values) >= 2, "values is None or len(values) < 2"
        sorted_values = sorted(values)

        if x < sorted_values[0]:
            if inner_only:
                raise ValueError(f"x is less than the smallest value in the list. {x=}, {sorted_values=}")
            else:
                return sorted_values[0], sorted_values[1]
        elif x > sorted_values[-1]:
            if inner_only:
                raise ValueError(f"x is greater than the largest value in the list. {x=}, {sorted_values=}")
            else:
                return sorted_values[-2], sorted_values[-1]

        for i, value in enumerate(sorted_values):
            if x >= value and i != len(sorted_values) - 1:
                continue
            else:
                end = value
                start = sorted_values[i - 1]
                break
        if start is None or end is None:
            raise ValueError(f"start or end is None. {x=}, {sorted_values=}, start={start=}, end={end=}")
        return start, end

    def _validate(self, value: float) -> float:
        """
        Validate the value
        """
        if value < 0.0:
            logger.debug(f"Negative value detected {value}, pass")
        return value

    def _interp_3d_linear(self, x: int, y: int, z: int, data: dict) -> float:
        """
        Interpolate the 3d data using linear interpolation
        """
        points_list = []
        values_list = []
        x_left, x_right = self._nearest_1d_point_helper(x, list(data.keys()))
        for i in [x_left, x_right]:
            y_left, y_right = self._nearest_1d_point_helper(y, list(data[i].keys()))
            for j in [y_left, y_right]:
                z_left, z_right = self._nearest_1d_point_helper(z, list(data[i][j].keys()))
                points_list.append([i, j, z_left])
                points_list.append([i, j, z_right])
                values_list.append(data[i][j][z_left])
                values_list.append(data[i][j][z_right])

        return self._validate(
            interpolate.griddata(np.array(points_list), np.array(values_list), (x, y, z), method="linear")
        )

    def _interp_2d_linear(self, x: int, y: int, data: dict) -> dict:
        """
        Interpolate the 2D data using linear interpolation.

        Returns:
            dict: {"latency": float, "power": float, "energy": float} - interpolated values for all metrics
        """
        # Check if data uses new dict format by sampling a leaf value
        sample_value = self._get_sample_leaf_value(data)

        if isinstance(sample_value, dict):
            # New format: interpolate latency and energy separately
            data_id = id(data)
            if data_id not in self._extracted_metrics_cache:
                self._extracted_metrics_cache[data_id] = self._extract_latency_and_energy_2d(data)

            latency_data, energy_data = self._extracted_metrics_cache[data_id]

            # Interpolate latency
            points_list = []
            latency_values = []
            x_left, x_right = self._nearest_1d_point_helper(x, list(latency_data.keys()))
            for i in [x_left, x_right]:
                y_left, y_right = self._nearest_1d_point_helper(y, list(latency_data[i].keys()))
                for j in [y_left, y_right]:
                    points_list.append([i, j])
                    latency_values.append(latency_data[i][j])

            latency = self._validate(
                interpolate.griddata(np.array(points_list), np.array(latency_values), (x, y), method="linear")
            )

            # Interpolate energy using same points
            energy_values = []
            for i in [x_left, x_right]:
                y_left, y_right = self._nearest_1d_point_helper(y, list(energy_data[i].keys()))
                for j in [y_left, y_right]:
                    energy_values.append(energy_data[i][j])

            energy = self._validate(
                interpolate.griddata(np.array(points_list), np.array(energy_values), (x, y), method="linear")
            )

            return {"latency": latency, "power": 0.0, "energy": energy}
        else:
            # Legacy format: data values are floats
            points_list = []
            values_list = []
            x_left, x_right = self._nearest_1d_point_helper(x, list(data.keys()))
            for i in [x_left, x_right]:
                y_left, y_right = self._nearest_1d_point_helper(y, list(data[i].keys()))
                for j in [y_left, y_right]:
                    points_list.append([i, j])
                    values_list.append(data[i][j])

            latency = self._validate(
                interpolate.griddata(np.array(points_list), np.array(values_list), (x, y), method="linear")
            )

            return {"latency": latency, "power": 0.0, "energy": 0.0}

    def _interp_3d(self, x: int, y: int, z: int, data: dict, method: str) -> dict:
        """
        Interpolate the 3d data using the given method.

        Returns:
            dict: {"latency": float, "power": float, "energy": float} - interpolated values for all metrics
            Note: power is always 0.0 as it's not currently used by callers (only latency and energy are used)
        """
        # Check if data uses new dict format by sampling a leaf value
        sample_value = self._get_sample_leaf_value(data)

        if isinstance(sample_value, dict):
            # New format: interpolate latency and energy only (power is not used by callers)
            # Use cache to avoid repeated extraction of the same data dictionary
            data_id = id(data)
            if data_id not in self._extracted_metrics_cache:
                # Extract both metrics in a single pass for maximum efficiency
                self._extracted_metrics_cache[data_id] = self._extract_latency_and_energy_3d(data)

            latency_data, energy_data = self._extracted_metrics_cache[data_id]

            if method == "linear":
                latency = self._interp_3d_linear(x, y, z, latency_data)
                energy = self._interp_3d_linear(x, y, z, energy_data)
            else:
                latency = self._interp_2d_1d(x, y, z, latency_data, method)
                energy = self._interp_2d_1d(x, y, z, energy_data, method)

            return {"latency": latency, "power": 0.0, "energy": energy}
        else:
            # Legacy format: data values are floats
            if method == "linear":
                latency = self._interp_3d_linear(x, y, z, data)
            else:
                latency = self._interp_2d_1d(x, y, z, data, method)

            return {"latency": latency, "power": 0.0, "energy": 0.0}

    def _get_sample_leaf_value(self, data: dict):
        """Get a sample leaf value from nested dict to determine format."""
        current = data
        max_depth = 20  # Safety limit to prevent infinite loops
        depth = 0
        visited = set()  # Track visited dict ids to detect cycles

        while isinstance(current, dict) and current and depth < max_depth:
            dict_id = id(current)
            if dict_id in visited:
                # Circular reference detected
                logger.warning("Circular reference detected in _get_sample_leaf_value")
                break
            visited.add(dict_id)

            # Check if this is a leaf dict with latency/power keys
            if "latency" in current or "power" in current:
                return current

            try:
                key = next(iter(current))
                current = current[key]
                depth += 1
            except (StopIteration, KeyError, TypeError):
                # Handle edge cases: empty dict, missing key, or non-dict value
                break

        if depth >= max_depth:
            logger.warning(f"Maximum depth ({max_depth}) exceeded in _get_sample_leaf_value")

        return current

    def _get_p2p_bandwidth(self, num_gpus: int) -> float:
        """
        Get the appropriate point-to-point bandwidth based on the number of GPUs.

        Three-tier bandwidth selection:
        - num_gpus <= num_gpus_per_node: intra_node_bw (NVLink within node)
        - num_gpus <= num_gpus_per_rack: inter_node_bw (NVLink via NVSwitch within rack)
        - num_gpus > num_gpus_per_rack: inter_rack_bw (InfiniBand between racks)

        Args:
            num_gpus: Number of GPUs involved in the communication

        Returns:
            Bandwidth in Bytes/s
        """
        node_spec = self.system_spec["node"]
        num_gpus_per_node = node_spec["num_gpus_per_node"]
        num_gpus_per_rack = node_spec.get("num_gpus_per_rack", float("inf"))

        if num_gpus <= num_gpus_per_node:
            return node_spec["intra_node_bw"]
        elif num_gpus <= num_gpus_per_rack:
            return node_spec["inter_node_bw"]
        else:
            # Inter-rack communication, fallback to inter_node_bw if inter_rack_bw not defined
            return node_spec.get("inter_rack_bw", node_spec["inter_node_bw"])

    def _bilinear_interpolation(self, x_list: list[int], y_list: list[int], x: int, y: int, data: dict) -> float:
        """
        Interpolate the 2d data using bilinear interpolation
        """
        x1, x2 = x_list
        # assure xy has a rectengle grid
        y1, y2 = y_list
        # Calculate the weights for the corners
        Q11, Q12, Q21, Q22 = data[x1][y1], data[x1][y2], data[x2][y1], data[x2][y2]  # noqa: N806

        f_x1_y1 = Q11 * (x2 - x) * (y2 - y)
        f_x1_y2 = Q12 * (x2 - x) * (y - y1)
        f_x2_y1 = Q21 * (x - x1) * (y2 - y)
        f_x2_y2 = Q22 * (x - x1) * (y - y1)
        # Calculate the total weight
        total_weight = (x2 - x1) * (y2 - y1)
        # Calculate the interpolated value
        interpolated_value = (f_x1_y1 + f_x1_y2 + f_x2_y1 + f_x2_y2) / total_weight
        return interpolated_value

    def _interp_2d_1d(self, x: int, y: int, z: int, data: dict, method="bilinear") -> float:
        """
        Interpolate the 3d data using the given method, 2d after 1d.
        """
        x_values = []
        x_left, x_right = self._nearest_1d_point_helper(x, list(data.keys()))

        for i in [x_left, x_right]:
            points_list = []
            values_list = []
            y_left, y_right = self._nearest_1d_point_helper(y, list(data[i].keys()))
            for j in [y_left, y_right]:
                z_left, z_right = self._nearest_1d_point_helper(z, list(data[i][j].keys()))
                points_list.append([j, z_left])
                points_list.append([j, z_right])
                values_list.append(data[i][j][z_left])
                values_list.append(data[i][j][z_right])
            if method == "cubic":
                x_values.append(
                    self._validate(
                        interpolate.griddata(np.array(points_list), np.array(values_list), (y, z), method="cubic")
                    )
                )
            elif method == "bilinear":
                x_values.append(
                    self._validate(self._bilinear_interpolation([y_left, y_right], [z_left, z_right], y, z, data[i]))
                )
            else:
                raise NotImplementedError

        return self._validate(self._interp_1d([x_left, x_right], x_values, x))

    def _interp_1d(self, x: list[int], y: list, value: int):
        """
        Interpolate the 1d data using linear interpolation.
        Handles both float and dict values.

        Args:
            x: list of x coordinates
            y: list of y values (can be floats or dicts)
            value: target x value

        Returns:
            float or dict: Interpolated result (dict if input was dict, float otherwise)
        """
        x0, x1 = x
        y0, y1 = y

        # Check if values are dicts (new format) or floats (legacy)
        if isinstance(y0, dict) and isinstance(y1, dict):
            # New format: interpolate latency and power separately
            lat0, lat1 = y0["latency"], y1["latency"]
            pow0, pow1 = y0["power"], y1["power"]

            # Apply interpolation logic for latency
            if (x0 - x1) * (lat0 - lat1) < 0 and (value - x0) * (value - x1) > 0:
                lat1 = lat0
            if lat0 == lat1:
                lat_result = lat0
            else:
                lat_result = lat0 + (lat1 - lat0) / (x1 - x0) * (value - x0)

            # Apply interpolation logic for power
            if (x0 - x1) * (pow0 - pow1) < 0 and (value - x0) * (value - x1) > 0:
                pow1 = pow0
            if pow0 == pow1:
                pow_result = pow0
            else:
                pow_result = pow0 + (pow1 - pow0) / (x1 - x0) * (value - x0)

            return {"latency": lat_result, "power": pow_result}
        else:
            # Legacy format: y values are floats
            if (x0 - x1) * (y0 - y1) < 0 and (value - x0) * (value - x1) > 0:
                y1 = y0
            if y0 == y1:
                return y0
            return y0 + (y1 - y0) / (x1 - x0) * (value - x0)

    def set_default_database_mode(self, mode: common.DatabaseMode) -> None:
        """
        Set the default database mode
        """
        if mode != self._default_database_mode:
            # Clear cached query methods since default database mode affects the results
            for attr_name in dir(self):
                attr = getattr(self, attr_name)
                if hasattr(attr, "cache_clear") and callable(attr):
                    attr.cache_clear()
            self._default_database_mode = mode

    def get_default_database_mode(self) -> common.DatabaseMode:
        """
        Get the default database mode
        """
        return self._default_database_mode

    @staticmethod
    def _normalize_gemm_quant_mode_for_table(quant_mode: common.GEMMQuantMode) -> common.GEMMQuantMode:
        """
        Normalize GEMM quant modes for perf table lookup.

        `fp8_static` is a behavioral mode that reuses `fp8` perf tables.
        """
        if quant_mode == common.GEMMQuantMode.fp8_static:
            return common.GEMMQuantMode.fp8
        return quant_mode

    @functools.lru_cache(maxsize=32768)
    def query_gemm(
        self,
        m: int,
        n: int,
        k: int,
        quant_mode: common.GEMMQuantMode,
        database_mode: common.DatabaseMode | None = None,
    ) -> PerformanceResult | tuple[float, float, float]:
        """
        Query GEMM operation latency and energy.

        Args:
            m: Number of rows in output matrix
            n: Number of columns in output matrix
            k: Inner dimension
            quant_mode: Quantization mode
            database_mode: Database mode (SILICON, EMPIRICAL, SOL, HYBRID)

        Returns:
            PerformanceResult: Acts as float (latency in ms).
                              Energy accessible via .energy attribute (W·ms).
                              Power can be computed as energy/latency (W).

        Example:
            >>> result = db.query_gemm(4096, 4096, 4096, GEMMQuantMode.nvfp4)
            >>> latency_ms = float(result)  # Use as float
            >>> energy_wms = result.energy
            >>> power_w = result.power  # or result.energy / float(result)
        """

        def get_sol(m: int, n: int, k: int, quant_mode: common.GEMMQuantMode) -> tuple[float, float, float]:
            """
            Get the sol time, sol math and sol mem
            """
            sol_math = 2 * m * n * k / (self.system_spec["gpu"]["float16_tc_flops"] * quant_mode.value.compute) * 1000
            sol_mem = quant_mode.value.memory * (m * n + m * k + n * k) / self.system_spec["gpu"]["mem_bw"] * 1000
            sol_time = max(sol_math, sol_mem)
            return sol_time, sol_math, sol_mem

        def get_empirical(m: int, n: int, k: int, quant_mode: common.GEMMQuantMode) -> float:
            """
            Get the empirical time
            """
            sol_time = get_sol(m, n, k, quant_mode)[0]
            scale_factor = 0.8
            return sol_time / scale_factor

        if database_mode is None:
            database_mode = self._default_database_mode

        table_quant_mode = self._normalize_gemm_quant_mode_for_table(quant_mode)

        # SOL and EMPIRICAL modes don't have power/energy data
        if database_mode == common.DatabaseMode.SOL:
            return PerformanceResult(get_sol(m, n, k, quant_mode)[0], energy=0.0)
        elif database_mode == common.DatabaseMode.SOL_FULL:
            return get_sol(m, n, k, quant_mode)
        elif database_mode == common.DatabaseMode.EMPIRICAL:
            return PerformanceResult(get_empirical(m, n, k, quant_mode), energy=0.0)
        else:
            # SILICON or HYBRID mode - use database
            try:
                if self._gemm_data is None:
                    msg = (
                        "GEMM perf table is missing for "
                        f"system='{self.system}', backend='{self.backend}', version='{self.version}'."
                    )
                    raise PerfDataNotAvailableError(msg)
                if table_quant_mode not in self._gemm_data:
                    supported = sorted([k.name for k in self._gemm_data])
                    raise PerfDataNotAvailableError(
                        "GEMM perf data not available for requested quant mode. "
                        f"system='{self.system}', backend='{self.backend}', version='{self.version}', "
                        f"quant_mode='{quant_mode.name}'. "
                        f"Supported gemm modes: {supported}"
                    )
                result = self._interp_3d(m, n, k, self._gemm_data[table_quant_mode], "cubic")
                # Result is dict: {"latency": ..., "power": ..., "energy": ...}
                return PerformanceResult(result["latency"], energy=result.get("energy", 0.0))
            except Exception:
                if database_mode == common.DatabaseMode.HYBRID:
                    logger.debug(f"Failed to query gemm data for {m=}, {n=}, {k=}, {quant_mode=}, using empirical mode")
                    return PerformanceResult(get_empirical(m, n, k, quant_mode), energy=0.0)
                else:
                    logger.exception(
                        f"Failed to query gemm data for {m=}, {n=}, {k=}, {quant_mode=}. Please consider Hybrid mode."
                    )
                    raise

    @functools.lru_cache(maxsize=32768)
    def query_compute_scale(
        self,
        m: int,
        k: int,
        quant_mode: common.GEMMQuantMode,
        database_mode: common.DatabaseMode | None = None,
    ) -> PerformanceResult | tuple[float, float, float]:
        """
        Query compute scale latency (dynamic quantization - static quantization).

        Args:
            m: Number of rows in input matrix
            k: Number of columns in input matrix
            quant_mode: Quantization mode
            database_mode: Database mode (SILICON, EMPIRICAL, SOL, HYBRID)

        Returns:
            PerformanceResult: Acts as float (latency in ms).
                              Energy accessible via .energy attribute (W·ms).
        """

        def get_sol(m: int, k: int) -> tuple[float, float, float]:
            """
            Get the sol time, sol math and sol mem
            """
            sol_mem = 2 * m * k / self.system_spec["gpu"]["mem_bw"] * 1000.0
            sol_time = sol_mem
            return sol_time, 0, sol_mem

        def get_empirical(m: int, k: int) -> float:
            """
            Get the empirical time
            """
            sol_time = get_sol(m, k)[0]
            scale_factor = 0.8
            return sol_time / scale_factor

        if database_mode is None:
            database_mode = self._default_database_mode

        table_quant_mode = self._normalize_gemm_quant_mode_for_table(quant_mode)

        if database_mode == common.DatabaseMode.SOL:
            return PerformanceResult(get_sol(m, k)[0], energy=0.0)
        elif database_mode == common.DatabaseMode.SOL_FULL:
            return get_sol(m, k)
        elif database_mode == common.DatabaseMode.EMPIRICAL:
            return PerformanceResult(get_empirical(m, k), energy=0.0)
        else:
            # SILICON or HYBRID mode - use database
            try:
                if self._compute_scale_data is None:
                    msg = (
                        "Compute scale perf table is missing for "
                        f"system='{self.system}', backend='{self.backend}', version='{self.version}'."
                    )
                    raise PerfDataNotAvailableError(msg)
                if table_quant_mode not in self._compute_scale_data:
                    supported = sorted([k.name for k in self._compute_scale_data])
                    raise PerfDataNotAvailableError(
                        "Compute scale perf data not available for requested quant mode. "
                        f"system='{self.system}', backend='{self.backend}', version='{self.version}', "
                        f"quant_mode='{quant_mode.name}'. "
                        f"Supported modes: {supported}"
                    )
                table = self._compute_scale_data[table_quant_mode]
                m_i = int(m)
                k_i = int(k)

                m_keys = sorted(table.keys())
                m_i = max(m_keys[0], min(m_i, m_keys[-1]))

                k_min = None
                k_max = None
                for row in table.values():
                    if not row:
                        continue
                    row_min = min(row.keys())
                    row_max = max(row.keys())
                    k_min = row_min if k_min is None else min(k_min, row_min)
                    k_max = row_max if k_max is None else max(k_max, row_max)

                if k_min is not None and k_max is not None:
                    k_i = max(k_min, min(k_i, k_max))

                result = self._interp_2d_linear(m_i, k_i, table)
                return PerformanceResult(result["latency"], energy=result.get("energy", 0.0))
            except Exception:
                if database_mode == common.DatabaseMode.HYBRID:
                    logger.debug(
                        f"Failed to query compute_scale data for {m=}, {k=}, {quant_mode=}, using empirical mode"
                    )
                    return PerformanceResult(get_empirical(m, k), energy=0.0)
                else:
                    logger.exception(
                        "Failed to query compute_scale data for "
                        f"{m=}, {k=}, {quant_mode=}. "
                        "Please consider Hybrid mode."
                    )
                    raise

    @functools.lru_cache(maxsize=32768)
    def query_scale_matrix(
        self,
        m: int,
        k: int,
        quant_mode: common.GEMMQuantMode,
        database_mode: common.DatabaseMode | None = None,
    ) -> PerformanceResult | tuple[float, float, float]:
        """
        Query scale matrix (static quantization) latency.

        Args:
            m: Number of rows in input matrix
            k: Number of columns in input matrix
            quant_mode: Quantization mode
            database_mode: Database mode (SILICON, EMPIRICAL, SOL, HYBRID)

        Returns:
            PerformanceResult: Acts as float (latency in ms).
                              Energy accessible via .energy attribute (W·ms).
        """

        def get_sol(m: int, k: int) -> tuple[float, float, float]:
            """
            Get the sol time, sol math and sol mem
            """
            sol_mem = 3 * m * k / self.system_spec["gpu"]["mem_bw"] * 1000.0
            sol_time = sol_mem
            return sol_time, 0, sol_mem

        def get_empirical(m: int, k: int) -> float:
            """
            Get the empirical time
            """
            sol_time = get_sol(m, k)[0]
            scale_factor = 0.8
            return sol_time / scale_factor

        if database_mode is None:
            database_mode = self._default_database_mode

        table_quant_mode = self._normalize_gemm_quant_mode_for_table(quant_mode)

        if database_mode == common.DatabaseMode.SOL:
            return PerformanceResult(get_sol(m, k)[0], energy=0.0)
        elif database_mode == common.DatabaseMode.SOL_FULL:
            return get_sol(m, k)
        elif database_mode == common.DatabaseMode.EMPIRICAL:
            return PerformanceResult(get_empirical(m, k), energy=0.0)
        else:
            # SILICON or HYBRID mode - use database
            try:
                if self._scale_matrix_data is None:
                    msg = (
                        "Scale matrix perf table is missing for "
                        f"system='{self.system}', backend='{self.backend}', version='{self.version}'."
                    )
                    raise PerfDataNotAvailableError(msg)
                if table_quant_mode not in self._scale_matrix_data:
                    supported = sorted([k.name for k in self._scale_matrix_data])
                    raise PerfDataNotAvailableError(
                        "Scale matrix perf data not available for requested quant mode. "
                        f"system='{self.system}', backend='{self.backend}', version='{self.version}', "
                        f"quant_mode='{quant_mode.name}'. "
                        f"Supported modes: {supported}"
                    )
                table = self._scale_matrix_data[table_quant_mode]
                m_i = int(m)
                k_i = int(k)

                m_keys = sorted(table.keys())
                m_i = max(m_keys[0], min(m_i, m_keys[-1]))

                k_min = None
                k_max = None
                for row in table.values():
                    if not row:
                        continue
                    row_min = min(row.keys())
                    row_max = max(row.keys())
                    k_min = row_min if k_min is None else min(k_min, row_min)
                    k_max = row_max if k_max is None else max(k_max, row_max)

                if k_min is not None and k_max is not None:
                    k_i = max(k_min, min(k_i, k_max))

                result = self._interp_2d_linear(m_i, k_i, table)
                return PerformanceResult(result["latency"], energy=result.get("energy", 0.0))
            except Exception:
                if database_mode == common.DatabaseMode.HYBRID:
                    logger.debug(
                        f"Failed to query scale_matrix data for {m=}, {k=}, {quant_mode=}, using empirical mode"
                    )
                    return PerformanceResult(get_empirical(m, k), energy=0.0)
                else:
                    logger.exception(
                        f"Failed to query scale_matrix data for {m=}, {k=}, {quant_mode=}. Please consider Hybrid mode."
                    )
                    raise

    @functools.lru_cache(maxsize=32768)
    def query_context_attention(
        self,
        b: int,
        s: int,  # s is the seq len to be computed, full_s = s + prefix
        prefix: int,
        n: int,
        n_kv: int,
        kvcache_quant_mode: common.KVCacheQuantMode,
        fmha_quant_mode: common.FMHAQuantMode,
        database_mode: Optional[common.DatabaseMode] = None,
        window_size: int = 0,
        head_size: int = 128,
    ) -> PerformanceResult | tuple[float, float, float]:
        """
        Query context (prefill) attention latency and energy.

        Args:
            b: Batch size
            s: Sequence length to be computed
            prefix: Prefix cache length
            n: Number of attention heads
            n_kv: Number of KV heads (for GQA)
            kvcache_quant_mode: KV cache quantization mode
            fmha_quant_mode: Attention computation quantization mode
            database_mode: Database mode (SILICON, EMPIRICAL, SOL, HYBRID)
            window_size: Sliding window size (0 for no window)
            head_size: Dimension per head

        Returns:
            PerformanceResult: Acts as float (latency in ms).
                              Energy accessible via .energy attribute (W·ms).
        """

        def get_sol(
            b: int,
            s: int,
            prefix: int,
            n: int,
            n_kv: int,
            h: int,
            w: int,
            kvcache_quant_mode: common.KVCacheQuantMode,
            fmha_quant_mode: common.FMHAQuantMode,
        ) -> tuple[float, float, float]:
            """
            Get the sol time, sol math and sol mem
            """
            full_s = s + prefix
            if w > 0 and full_s > w:
                # Sliding window attention
                # Each position attends to at most w previous positions
                ops = 2 * b * (full_s - prefix) * w * n * h * 2
            else:
                # Normal no sliding window
                ops = (
                    2 * b * (full_s * full_s - prefix * prefix) * n * h * 2 / 2
                )  # 2 for fma, 2 for q*k^t+*v, /2 for causality.
            mem_bytes = (
                2
                * b
                * (
                    n * (full_s - prefix) * h  # Q read, assuming 16 bits
                    + n * (full_s - prefix) * h  # Output write, assuming 16 bits
                )
                + kvcache_quant_mode.value.memory * b * (2 * n_kv * full_s * h)  # K,V read
            )  # TODO fp8 io
            sol_math = ops / self.system_spec["gpu"]["float16_tc_flops"] * 1000 / fmha_quant_mode.value.compute
            sol_mem = mem_bytes / self.system_spec["gpu"]["mem_bw"] * 1000
            sol_time = max(sol_math, sol_mem)
            return sol_time, sol_math, sol_mem

        def get_empirical(
            b: int,
            s: int,
            prefix: int,
            n: int,
            n_kv: int,
            head_size: int,
            window_size: int,
            kvcache_quant_mode: common.KVCacheQuantMode,
            fmha_quant_mode: common.FMHAQuantMode,
        ) -> float:
            """
            Get the empirical time
            """
            latency = get_sol(b, s, prefix, n, n_kv, head_size, window_size, kvcache_quant_mode, fmha_quant_mode)[0]
            scale_factor = 0.6
            return latency / scale_factor

        # query logic starts
        assert n_kv <= n, "n_kv must be less than or equal to n"

        if database_mode is None:
            database_mode = self._default_database_mode
        if database_mode == common.DatabaseMode.SOL:
            sol_latency = get_sol(b, s, prefix, n, n_kv, head_size, window_size, kvcache_quant_mode, fmha_quant_mode)[0]
            return PerformanceResult(sol_latency, energy=0.0)
        elif database_mode == common.DatabaseMode.SOL_FULL:
            return get_sol(b, s, prefix, n, n_kv, head_size, window_size, kvcache_quant_mode, fmha_quant_mode)
        elif database_mode == common.DatabaseMode.EMPIRICAL:
            emp_latency = get_empirical(
                b,
                s,
                prefix,
                n,
                n_kv,
                head_size,
                window_size,
                kvcache_quant_mode,
                fmha_quant_mode,
            )
            return PerformanceResult(emp_latency, energy=0.0)
        else:
            try:
                if self._context_attention_data is None:
                    raise PerfDataNotAvailableError(
                        f"Context attention perf table is missing for system='{self.system}', "
                        f"backend='{self.backend}', version='{self.version}'. "
                        "Please use HYBRID or EMPIRICAL database mode, or provide the data file."
                    )
                full_s = s + prefix
                prefix_correction = (full_s * full_s - prefix * prefix) / (full_s * full_s)
                # In self._context_attention_data, we use n_kv = 0 to mean n_kv == n.
                n_kv = 0 if n == n_kv else n_kv
                attention_dict = self._context_attention_data[fmha_quant_mode][kvcache_quant_mode][n_kv][head_size][
                    window_size
                ]
                result = self._interp_3d(n, full_s, b, attention_dict, "cubic")
                latency = result["latency"] * prefix_correction
                energy = result.get("energy", 0.0) * prefix_correction
                return PerformanceResult(latency, energy=energy)
            except Exception as e:
                if database_mode == common.DatabaseMode.HYBRID:
                    logger.debug(
                        f"Failed to query context attention data for {b=}, {s=}, {prefix=}, {n=}, {n_kv=}, "
                        f"{head_size=}, {window_size=}, {kvcache_quant_mode=}, {fmha_quant_mode=}, using empirical mode"
                    )
                    latency = get_empirical(
                        b,
                        s,
                        prefix,
                        n,
                        n_kv,
                        head_size,
                        window_size,
                        kvcache_quant_mode,
                        fmha_quant_mode,
                    )
                    return PerformanceResult(latency, energy=0.0)
                else:
                    # Missing perf data is expected for some system/backend/mode combinations.
                    # Avoid spamming full tracebacks during Pareto enumeration.
                    if isinstance(e, PerfDataNotAvailableError):
                        logger.debug(
                            f"Missing context attention perf data for {b=}, {s=}, {prefix=}, {n=}, {n_kv=}, "
                            f"{head_size=}, {window_size=}, {kvcache_quant_mode=}, {fmha_quant_mode=}"
                        )
                    else:
                        logger.exception(
                            f"Failed to query context attention data for {b=}, {s=}, {prefix=}, {n=}, "
                            f"{n_kv=}, {head_size=}, {window_size=}, {kvcache_quant_mode=}, {fmha_quant_mode=}. "
                            "Please consider Hybrid mode."
                        )
                    raise

    @functools.lru_cache(maxsize=32768)
    def query_generation_attention(
        self,
        b: int,
        s: int,
        n: int,
        n_kv: int,
        kvcache_quant_mode: common.KVCacheQuantMode,
        database_mode: Optional[common.DatabaseMode] = None,
        window_size: int = 0,
        head_size: int = 128,
    ) -> PerformanceResult | tuple[float, float, float]:
        """
        Query generation (decode) attention latency and energy.

        Args:
            b: Batch size
            s: KV cache length
            n: Number of attention heads
            n_kv: Number of KV heads (for GQA)
            kvcache_quant_mode: KV cache quantization mode
            database_mode: Database mode (SILICON, EMPIRICAL, SOL, HYBRID)
            window_size: Sliding window size (0 for no window)
            head_size: Dimension per head

        Returns:
            PerformanceResult: Acts as float (latency in ms).
                              Energy accessible via .energy attribute (W·ms).
        """

        def get_sol(
            b: int,
            s: int,
            n: int,
            n_kv: int,
            h: int,
            w: int,
            kvcache_quant_mode: common.KVCacheQuantMode,
        ) -> tuple[float, float, float]:
            """
            Get the sol time, sol math and sol mem
            """
            if kvcache_quant_mode == common.KVCacheQuantMode.fp8:
                quant_mode_gen = common.FMHAQuantMode.fp8
            else:
                quant_mode_gen = common.FMHAQuantMode.float16
            if w > 0:
                kv_len = min(s - 1, w)
            else:
                kv_len = s - 1
            # only consider fp16 mmha
            ops = 2 * b * n * h * 2 * (kv_len)  # 2 for fma, 2 for q*k^t+*v
            # kvcache load bytes will depend on kvcache quant. while input q and output might be in
            # fp16.
            mem_bytes = b * (
                n * h * 2  # Query read, assuming 16bits
                + 2 * n_kv * (kv_len) * h * kvcache_quant_mode.value.memory  # K, V cache read
                + n * h * 2  # Output write, assuming 16bits
            )

            sol_math = ops / self.system_spec["gpu"]["float16_tc_flops"] * 1000 / quant_mode_gen.value.compute
            sol_mem = mem_bytes / self.system_spec["gpu"]["mem_bw"] * 1000
            sol_time = max(sol_math, sol_mem)
            return sol_time, sol_math, sol_mem

        def get_empirical(
            b: int,
            s: int,
            n: int,
            n_kv: int,
            h: int,
            w: int,
            kvcache_quant_mode: common.KVCacheQuantMode,
        ) -> float:
            """
            Get the hybrid time
            """
            latency = get_sol(b, s, n, n_kv, h, w, kvcache_quant_mode)[0]
            scale_factor = 0.8
            return latency / scale_factor

        # query logic starts
        assert n_kv <= n, "n_kv must be less than or equal to n"

        if database_mode is None:
            database_mode = self._default_database_mode
        if database_mode == common.DatabaseMode.SOL:
            sol_latency = get_sol(b, s, n, n_kv, head_size, window_size, kvcache_quant_mode)[0]
            return PerformanceResult(sol_latency, energy=0.0)
        elif database_mode == common.DatabaseMode.SOL_FULL:
            return get_sol(b, s, n, n_kv, head_size, window_size, kvcache_quant_mode)
        elif database_mode == common.DatabaseMode.EMPIRICAL:
            emp_latency = get_empirical(b, s, n, n_kv, head_size, window_size, kvcache_quant_mode)
            return PerformanceResult(emp_latency, energy=0.0)
        else:
            try:
                if self._generation_attention_data is None:
                    raise PerfDataNotAvailableError(
                        f"Generation attention perf table is missing for system='{self.system}', "
                        f"backend='{self.backend}', version='{self.version}'. "
                        "Please use HYBRID or EMPIRICAL database mode, or provide the data file."
                    )
                # In self._generation_attention_data, we use n_kv = 0 to mean n_kv == n.
                if n_kv == n:
                    n_kv = 0

                attention_dict = self._generation_attention_data[kvcache_quant_mode][n_kv][head_size][window_size]
                result = self._interp_3d(n, b, s, attention_dict, "bilinear")
                latency = result["latency"]
                energy = result.get("energy", 0.0)
                return PerformanceResult(latency, energy=energy)
            except Exception:
                if database_mode == common.DatabaseMode.HYBRID:
                    logger.debug(
                        f"Failed to query generation attention data for {b=}, {s=}, {n=}, {n_kv=}, "
                        f"{head_size=}, {window_size=}, {kvcache_quant_mode=}, using empirical mode"
                    )
                    latency = get_empirical(b, s, n, n_kv, head_size, window_size, kvcache_quant_mode)
                    return PerformanceResult(latency, energy=0.0)
                else:
                    logger.exception(
                        f"Failed to query generation attention data for {b=}, {s=}, {n=}, {n_kv=}, "
                        f"{head_size=}, {window_size=}, {kvcache_quant_mode=}, {database_mode=}. "
                        "Please consider Hybrid mode."
                    )
                    raise

    @functools.lru_cache(maxsize=32768)
    def query_context_mla(
        self,
        b: int,
        s: int,
        prefix: int,
        num_heads: int,
        kvcache_quant_mode: common.KVCacheQuantMode,
        fmha_quant_mode: common.FMHAQuantMode,
        database_mode: common.DatabaseMode | None = None,
    ) -> PerformanceResult | tuple[float, float, float]:
        """
        Query context MLA (Multi-head Latent Attention) latency and energy.

        Args:
            b: Batch size
            s: Sequence length to be computed
            prefix: Prefix cache length
            num_heads: Number of attention heads
            kvcache_quant_mode: KV cache quantization mode
            fmha_quant_mode: Attention computation quantization mode
            database_mode: Database mode (SILICON, EMPIRICAL, SOL, HYBRID)

        Returns:
            PerformanceResult: Acts as float (latency in ms).
                              Energy accessible via .energy attribute (W·ms).
        """

        def get_sol(
            b: int,
            s: int,
            prefix: int,
            num_heads: int,
            kvcache_quant_mode: common.KVCacheQuantMode,
            fmha_quant_mode: common.FMHAQuantMode,
        ) -> tuple[float, float, float]:
            """
            Get the sol time, sol math and sol mem
            """
            full_s = s + prefix
            ops = (
                b * num_heads * 2 / 2 * (192 + 128) * (full_s * full_s - prefix * prefix)
            )  # 2 for fma, 2 for causality. num_heads, for local heads
            # s * 192 for q read, full_s * 192 for k read, full_s * 128 for v read, s * 192 for write.
            mem_bytes = (
                b * num_heads * (kvcache_quant_mode.value.memory * full_s * (192 + 128) + 2 * s * (192 + 128))
            )  # 2 for qk, TODO
            sol_math = ops / self.system_spec["gpu"]["float16_tc_flops"] * 1000 / fmha_quant_mode.value.compute
            sol_mem = mem_bytes / self.system_spec["gpu"]["mem_bw"] * 1000
            sol_time = max(sol_math, sol_mem)
            return sol_time, sol_math, sol_mem

        def get_empirical(
            b: int,
            s: int,
            prefix: int,
            num_heads: int,
            kvcache_quant_mode: common.KVCacheQuantMode,
            fmha_quant_mode: common.FMHAQuantMode,
        ) -> float:
            """
            Get the hybrid time
            """
            latency = get_sol(b, s, prefix, num_heads, kvcache_quant_mode, fmha_quant_mode)[0]
            scale_factor = 0.6
            return latency / scale_factor

        if database_mode is None:
            database_mode = self._default_database_mode
        if database_mode == common.DatabaseMode.SOL:
            sol_latency = get_sol(b, s, prefix, num_heads, kvcache_quant_mode, fmha_quant_mode)[0]
            return PerformanceResult(sol_latency, energy=0.0)
        elif database_mode == common.DatabaseMode.SOL_FULL:
            return get_sol(b, s, prefix, num_heads, kvcache_quant_mode, fmha_quant_mode)
        elif database_mode == common.DatabaseMode.EMPIRICAL:
            emp_latency = get_empirical(b, s, prefix, num_heads, kvcache_quant_mode, fmha_quant_mode)
            return PerformanceResult(emp_latency, energy=0.0)
        else:
            try:
                if self._context_mla_data is None:
                    raise PerfDataNotAvailableError(
                        f"Context MLA perf table is missing for system='{self.system}', "
                        f"backend='{self.backend}', version='{self.version}'. "
                        "Please use HYBRID or EMPIRICAL database mode, or provide the data file."
                    )
                full_s = s + prefix
                prefix_correction = (full_s * full_s - prefix * prefix) / (full_s * full_s)
                mla_dict = self._context_mla_data[fmha_quant_mode][kvcache_quant_mode]
                result = self._interp_3d(num_heads, full_s, b, mla_dict, "cubic")
                latency = result["latency"] * prefix_correction
                energy = result.get("energy", 0.0) * prefix_correction
                return PerformanceResult(latency, energy=energy)
            except Exception:
                if database_mode == common.DatabaseMode.HYBRID:
                    logger.debug(
                        f"Failed to query context mla data for {b=}, {s=}, {prefix=}, {num_heads=}, "
                        f"{kvcache_quant_mode=}, {fmha_quant_mode=}, using empirical mode"
                    )
                    latency = get_empirical(b, s, prefix, num_heads, kvcache_quant_mode, fmha_quant_mode)
                    return PerformanceResult(latency, energy=0.0)
                else:
                    logger.exception(
                        f"Failed to query context mla data for {b=}, {s=}, {prefix=}, {num_heads=}, \
                        {kvcache_quant_mode=}, {fmha_quant_mode=}, {database_mode=}. Please consider Hybrid mode."
                    )
                    raise

    @functools.lru_cache(maxsize=32768)
    def query_generation_mla(
        self,
        b: int,
        s: int,
        num_heads: int,
        kvcache_quant_mode: common.KVCacheQuantMode,
        database_mode: common.DatabaseMode | None = None,
    ) -> PerformanceResult | tuple[float, float, float]:
        """
        Query generation MLA (Multi-head Latent Attention) latency and energy.

        Args:
            b: Batch size
            s: KV cache length
            num_heads: Number of attention heads
            kvcache_quant_mode: KV cache quantization mode
            database_mode: Database mode (SILICON, EMPIRICAL, SOL, HYBRID)

        Returns:
            PerformanceResult: Acts as float (latency in ms).
                              Energy accessible via .energy attribute (W·ms).
        """

        def get_sol(
            b: int, s: int, num_heads: int, kvcache_quant_mode: common.KVCacheQuantMode
        ) -> tuple[float, float, float]:
            """
            Get the sol time, sol math and sol mem
            """
            if kvcache_quant_mode == common.KVCacheQuantMode.fp8:
                quant_mode_gen = common.FMHAQuantMode.fp8
            else:
                quant_mode_gen = common.FMHAQuantMode.float16
            # only consider fp16 mmha
            ops = 2 * b * num_heads * 1088 * s  # 2 for fma
            # kvcache load bytes will depend on kvcache quant.
            # while input q and output might be in fp16.
            mem_bytes = b * (num_heads * 1088 * 2 + (s - 1) * 576 * kvcache_quant_mode.value.memory)
            # fp16 io + fp16/fp8 kv cache, TODO fp8 io
            sol_math = ops / self.system_spec["gpu"]["float16_tc_flops"] * 1000 / quant_mode_gen.value.compute
            sol_mem = mem_bytes / self.system_spec["gpu"]["mem_bw"] * 1000
            sol_time = max(sol_math, sol_mem)
            return sol_time, sol_math, sol_mem

        def get_empirical(
            b: int,
            s: int,
            num_heads: int,
            kvcache_quant_mode: common.KVCacheQuantMode,
        ) -> float:
            """
            Get the hybrid time
            """
            latency = get_sol(b, s, num_heads, kvcache_quant_mode)[0]
            scale_factor = 0.8
            return latency / scale_factor

        if database_mode is None:
            database_mode = self._default_database_mode
        if database_mode == common.DatabaseMode.SOL:
            sol_latency = get_sol(b, s, num_heads, kvcache_quant_mode)[0]
            return PerformanceResult(sol_latency, energy=0.0)
        elif database_mode == common.DatabaseMode.SOL_FULL:
            return get_sol(b, s, num_heads, kvcache_quant_mode)
        elif database_mode == common.DatabaseMode.EMPIRICAL:
            emp_latency = get_empirical(b, s, num_heads, kvcache_quant_mode)
            return PerformanceResult(emp_latency, energy=0.0)
        else:
            try:
                if self._generation_mla_data is None:
                    raise PerfDataNotAvailableError(
                        f"Generation MLA perf table is missing for system='{self.system}', "
                        f"backend='{self.backend}', version='{self.version}'. "
                        "Please use HYBRID or EMPIRICAL database mode, or provide the data file."
                    )
                mla_dict = self._generation_mla_data[kvcache_quant_mode]
                result = self._interp_3d(num_heads, b, s, mla_dict, "bilinear")
                latency = result["latency"]
                energy = result.get("energy", 0.0)
                return PerformanceResult(latency, energy=energy)
            except Exception:
                if database_mode == common.DatabaseMode.HYBRID:
                    logger.debug(
                        f"Failed to query generation mla data for {b=}, {s=}, {num_heads=}, "
                        f"{kvcache_quant_mode=}, using empirical mode"
                    )
                    latency = get_empirical(b, s, num_heads, kvcache_quant_mode)
                    return PerformanceResult(latency, energy=0.0)
                else:
                    logger.exception(
                        f"Failed to query generation mla data for {b=}, {s=}, {num_heads=}, \
                        {kvcache_quant_mode=}, {database_mode=}. Please consider Hybrid mode."
                    )
                    raise

    @functools.lru_cache(maxsize=32768)
    def query_context_dsa(
        self,
        b: int,
        s: int,
        prefix: int,
        num_heads: int,
        index_n_heads: int,
        index_topk: int,
        kvcache_quant_mode: common.KVCacheQuantMode,
        fmha_quant_mode: common.FMHAQuantMode,
        database_mode: common.DatabaseMode | None = None,
    ) -> PerformanceResult | tuple[float, float, float]:
        """
        Query context DSA (DeepSeek Sparse Attention) latency and energy.

        DSA uses an index network to select top-k relevant KV positions,
        reducing attention complexity from O(n²) to O(n×k).

        Args:
            b: Batch size
            s: Sequence length to be computed
            prefix: Prefix cache length
            num_heads: Number of attention heads
            index_n_heads: Number of heads in the DSA index network
            index_topk: Number of top-k KV positions to select
            kvcache_quant_mode: KV cache quantization mode
            fmha_quant_mode: Attention computation quantization mode
            database_mode: Database mode (SILICON, EMPIRICAL, SOL, HYBRID)

        Returns:
            PerformanceResult: Acts as float (latency in ms).
                              Energy accessible via .energy attribute (W·ms).
        """

        def get_sol(
            b: int,
            s: int,
            prefix: int,
            num_heads: int,
            index_n_heads: int,
            index_topk: int,
            kvcache_quant_mode: common.KVCacheQuantMode,
            fmha_quant_mode: common.FMHAQuantMode,
        ) -> tuple[float, float, float]:
            """
            Get the SOL time for DSA context phase.
            
            DSA reduces attention from O(n²) to O(n×k) where k=index_topk.
            Additional cost: index network computation.
            """
            full_s = s + prefix
            
            # Index network: projects query to select top-k positions
            # Index network uses index_n_heads with index_head_dim=128
            index_ops = 2 * b * index_n_heads * 128 * full_s  # Index scoring
            
            # Sparse attention: only attend to top-k positions per query
            # This is the key efficiency: O(n×k) instead of O(n²)
            sparse_attn_ops = (
                b * num_heads * 2 * (full_s * index_topk) * 128  # Q @ K^T
                + b * num_heads * 2 * (full_s * index_topk) * 128  # Softmax @ V
            )
            
            ops = index_ops + sparse_attn_ops
            
            # Memory access pattern for DSA:
            # - Q: s × num_heads × 128
            # - K/V: full_s × num_heads × 128 (but only top-k accessed)
            # - Index network weights: index_n_heads × 128 × hidden_size
            q_bytes = b * s * num_heads * 128 * 2
            kv_bytes = b * full_s * num_heads * 128 * kvcache_quant_mode.value.memory
            # Only load top-k positions for sparse attention
            sparse_kv_bytes = b * full_s * index_topk * 128 * 2
            # Index network weights (shared across batch)
            index_weight_bytes = index_n_heads * 128 * 7168 * 2  # hidden_size=7168
            
            mem_bytes = q_bytes + kv_bytes + sparse_kv_bytes + index_weight_bytes
            
            sol_math = ops / self.system_spec["gpu"]["float16_tc_flops"] * 1000 / fmha_quant_mode.value.compute
            sol_mem = mem_bytes / self.system_spec["gpu"]["mem_bw"] * 1000
            sol_time = max(sol_math, sol_mem)
            
            return sol_time, sol_math, sol_mem

        def get_empirical(
            b: int,
            s: int,
            prefix: int,
            num_heads: int,
            index_n_heads: int,
            index_topk: int,
            kvcache_quant_mode: common.KVCacheQuantMode,
            fmha_quant_mode: common.FMHAQuantMode,
        ) -> float:
            """
            Get the empirical time for DSA context phase.
            
            DSA has higher constant overhead due to index network,
            so use a lower scale factor than dense attention.
            """
            latency = get_sol(b, s, prefix, num_heads, index_n_heads, index_topk, kvcache_quant_mode, fmha_quant_mode)[0]
            # DSA has more overhead due to index network and sparse gather/scatter
            scale_factor = 0.4  # Lower than MLA's 0.6 due to sparse overhead
            return latency / scale_factor

        if database_mode is None:
            database_mode = self._default_database_mode
        if database_mode == common.DatabaseMode.SOL:
            sol_latency = get_sol(b, s, prefix, num_heads, index_n_heads, index_topk, kvcache_quant_mode, fmha_quant_mode)[0]
            return PerformanceResult(sol_latency, energy=0.0)
        elif database_mode == common.DatabaseMode.SOL_FULL:
            return get_sol(b, s, prefix, num_heads, index_n_heads, index_topk, kvcache_quant_mode, fmha_quant_mode)
        elif database_mode == common.DatabaseMode.EMPIRICAL:
            emp_latency = get_empirical(b, s, prefix, num_heads, index_n_heads, index_topk, kvcache_quant_mode, fmha_quant_mode)
            return PerformanceResult(emp_latency, energy=0.0)
        else:
            # HYBRID mode: try to load empirical data, fall back to SOL
            try:
                # TODO: Add DSA empirical data loading
                # For now, use empirical estimation
                logger.debug(
                    f"DSA empirical data not yet available, using SOL estimation for "
                    f"{b=}, {s=}, {prefix=}, {num_heads=}, {index_topk=}"
                )
                emp_latency = get_empirical(b, s, prefix, num_heads, index_n_heads, index_topk, kvcache_quant_mode, fmha_quant_mode)
                return PerformanceResult(emp_latency, energy=0.0)
            except Exception:
                if database_mode == common.DatabaseMode.HYBRID:
                    logger.debug(
                        f"Failed to query context DSA data, using empirical mode"
                    )
                    latency = get_empirical(b, s, prefix, num_heads, index_n_heads, index_topk, kvcache_quant_mode, fmha_quant_mode)
                    return PerformanceResult(latency, energy=0.0)
                else:
                    logger.exception(
                        f"Failed to query context DSA data for {b=}, {s=}, {prefix=}, {num_heads=}, \
                        {index_topk=}, {database_mode=}. Please consider Hybrid mode."
                    )
                    raise

    @functools.lru_cache(maxsize=32768)
    def query_generation_dsa(
        self,
        b: int,
        s: int,
        num_heads: int,
        index_n_heads: int,
        index_topk: int,
        kvcache_quant_mode: common.KVCacheQuantMode,
        database_mode: common.DatabaseMode | None = None,
    ) -> PerformanceResult | tuple[float, float, float]:
        """
        Query generation DSA (DeepSeek Sparse Attention) latency and energy.

        Args:
            b: Batch size
            s: KV cache length
            num_heads: Number of attention heads
            index_n_heads: Number of heads in the DSA index network
            index_topk: Number of top-k KV positions to select
            kvcache_quant_mode: KV cache quantization mode
            database_mode: Database mode (SILICON, EMPIRICAL, SOL, HYBRID)

        Returns:
            PerformanceResult: Acts as float (latency in ms).
                              Energy accessible via .energy attribute (W·ms).
        """

        def get_sol(
            b: int,
            s: int,
            num_heads: int,
            index_n_heads: int,
            index_topk: int,
            kvcache_quant_mode: common.KVCacheQuantMode,
        ) -> tuple[float, float, float]:
            """
            Get the SOL time for DSA generation phase.
            
            In generation, we have 1 new token attending to s KV positions.
            DSA selects top-k positions from s.
            """
            if kvcache_quant_mode == common.KVCacheQuantMode.fp8:
                quant_mode_gen = common.FMHAQuantMode.fp8
            else:
                quant_mode_gen = common.FMHAQuantMode.float16
            
            # Index network for the new query token
            index_ops = 2 * b * index_n_heads * 128 * s  # Score all s positions
            
            # Sparse attention: new token attends to top-k positions
            sparse_attn_ops = (
                2 * b * num_heads * 128 * index_topk  # Q @ K^T (1 × top-k)
                + 2 * b * num_heads * 128 * index_topk  # Softmax @ V
            )
            
            ops = index_ops + sparse_attn_ops
            
            # Memory: load KV cache (but only top-k for actual attention)
            q_bytes = b * num_heads * 128 * 2
            # Need to scan entire KV cache for index network
            kv_cache_bytes = b * s * num_heads * 128 * kvcache_quant_mode.value.memory
            # But only load top-k for sparse attention
            sparse_attn_bytes = b * index_topk * num_heads * 128 * 2
            
            mem_bytes = q_bytes + kv_cache_bytes + sparse_attn_bytes
            
            sol_math = ops / self.system_spec["gpu"]["float16_tc_flops"] * 1000 / quant_mode_gen.value.compute
            sol_mem = mem_bytes / self.system_spec["gpu"]["mem_bw"] * 1000
            sol_time = max(sol_math, sol_mem)
            
            return sol_time, sol_math, sol_mem

        def get_empirical(
            b: int,
            s: int,
            num_heads: int,
            index_n_heads: int,
            index_topk: int,
            kvcache_quant_mode: common.KVCacheQuantMode,
        ) -> float:
            """
            Get the empirical time for DSA generation phase.
            """
            latency = get_sol(b, s, num_heads, index_n_heads, index_topk, kvcache_quant_mode)[0]
            scale_factor = 0.5
            return latency / scale_factor

        if database_mode is None:
            database_mode = self._default_database_mode
        if database_mode == common.DatabaseMode.SOL:
            sol_latency = get_sol(b, s, num_heads, index_n_heads, index_topk, kvcache_quant_mode)[0]
            return PerformanceResult(sol_latency, energy=0.0)
        elif database_mode == common.DatabaseMode.SOL_FULL:
            return get_sol(b, s, num_heads, index_n_heads, index_topk, kvcache_quant_mode)
        elif database_mode == common.DatabaseMode.EMPIRICAL:
            emp_latency = get_empirical(b, s, num_heads, index_n_heads, index_topk, kvcache_quant_mode)
            return PerformanceResult(emp_latency, energy=0.0)
        else:
            try:
                # TODO: Add DSA empirical data loading
                logger.debug(
                    f"DSA generation empirical data not yet available, using SOL estimation"
                )
                emp_latency = get_empirical(b, s, num_heads, index_n_heads, index_topk, kvcache_quant_mode)
                return PerformanceResult(emp_latency, energy=0.0)
            except Exception:
                if database_mode == common.DatabaseMode.HYBRID:
                    logger.debug(
                        f"Failed to query generation DSA data, using empirical mode"
                    )
                    latency = get_empirical(b, s, num_heads, index_n_heads, index_topk, kvcache_quant_mode)
                    return PerformanceResult(latency, energy=0.0)
                else:
                    logger.exception(
                        f"Failed to query generation DSA data for {b=}, {s=}, {num_heads=}, \
                        {index_topk=}, {database_mode=}. Please consider Hybrid mode."
                    )
                    raise

    @functools.lru_cache(maxsize=32768)
    def query_wideep_generation_mla(
        self,
        b: int,
        s: int,
        tp_size: int,
        kvcache_quant_mode: common.KVCacheQuantMode,
        fmha_quant_mode: common.FMHAQuantMode,
        attention_backend: str | None = None,
        database_mode: common.DatabaseMode | None = None,
    ) -> PerformanceResult | tuple[float, float, float]:
        """
        Query the generation mla data for SGLang backend with SOL calculation
        """

        def get_sol(
            b: int,
            s: int,
            tp_size: int,
            kvcache_quant_mode: common.KVCacheQuantMode,
            fmha_quant_mode: common.FMHAQuantMode,
        ) -> tuple[float, float, float]:
            hidden_size = 7168
            q_lora_rank = 1536
            kv_lora_rank = 512
            qk_rope_head_dim = 64
            qk_nope_head_dim = 128
            v_head_dim = 128
            num_head = 128 // tp_size

            # qkv_a projection (decode mode)
            qkv_a_flop = 2 * hidden_size * (q_lora_rank + kv_lora_rank + qk_rope_head_dim) * b
            qkv_a_mem = (
                b * hidden_size
                + hidden_size * (q_lora_rank + kv_lora_rank + qk_rope_head_dim)
                + 2 * b * (q_lora_rank + kv_lora_rank + qk_rope_head_dim)
            )

            # q_b projection
            q_b_flop = 2 * q_lora_rank * num_head * (qk_rope_head_dim + qk_nope_head_dim) * b
            q_b_mem = (
                b * q_lora_rank
                + q_lora_rank * num_head * (qk_rope_head_dim + qk_nope_head_dim)
                + 2 * b * num_head * (qk_rope_head_dim + qk_nope_head_dim)
            )

            # q_w_kc (attention computation)
            q_w_kc_flop = 2 * num_head * qk_nope_head_dim * kv_lora_rank * b
            q_w_kc_mem = (
                b * num_head * qk_nope_head_dim
                + num_head * kv_lora_rank * qk_nope_head_dim
                + 2 * b * num_head * kv_lora_rank
            )

            attn_flop = 2 * b * s * num_head * (qk_rope_head_dim + kv_lora_rank * 2)
            attn_mem = (
                b * num_head * (kv_lora_rank + qk_rope_head_dim)
                + b * s * (qk_rope_head_dim + kv_lora_rank)
                + b * num_head * kv_lora_rank
            )

            # s_w_vc (attention output projection)
            s_w_vc_flop = 2 * b * num_head * kv_lora_rank * v_head_dim
            s_w_vc_mem = (
                b * num_head * kv_lora_rank + num_head * v_head_dim * kv_lora_rank + 2 * b * num_head * v_head_dim
            )

            # attention output projection
            attn_out_flop = 2 * num_head * v_head_dim * hidden_size * b
            attn_out_mem = b * num_head * v_head_dim + num_head * v_head_dim * hidden_size + 2 * b * hidden_size

            ops = qkv_a_flop + q_b_flop + q_w_kc_flop + s_w_vc_flop + attn_out_flop
            mem_bytes = (
                qkv_a_mem + q_b_mem + q_w_kc_mem + attn_mem * 2 + s_w_vc_mem + attn_out_mem
            ) * fmha_quant_mode.value.memory
            sol_math = ops / (self.system_spec["gpu"]["float16_tc_flops"] * fmha_quant_mode.value.compute) * 1000
            sol_math += attn_flop / (self.system_spec["gpu"]["float16_tc_flops"]) * 1000
            sol_mem = mem_bytes / self.system_spec["gpu"]["mem_bw"] * 1000
            sol_time = max(sol_math, sol_mem)

            return sol_time, sol_math, sol_mem

        def get_empirical(
            b: int,
            s: int,
            tp_size: int,
            kvcache_quant_mode: common.KVCacheQuantMode,
            fmha_quant_mode: common.FMHAQuantMode,
        ) -> float:
            """
            Get the hybrid time
            """
            latency = get_sol(b, s, tp_size, kvcache_quant_mode, fmha_quant_mode)[0]
            scale_factor = 0.7
            return latency / scale_factor

        if database_mode is None:
            database_mode = self._default_database_mode
        if database_mode == common.DatabaseMode.SOL:
            sol_time = get_sol(b, s, tp_size, kvcache_quant_mode, fmha_quant_mode)[0]
            return PerformanceResult(sol_time, energy=0.0)
        elif database_mode == common.DatabaseMode.SOL_FULL:
            return get_sol(b, s, tp_size, kvcache_quant_mode, fmha_quant_mode)
        elif database_mode == common.DatabaseMode.EMPIRICAL:
            return PerformanceResult(get_empirical(b, s, tp_size, kvcache_quant_mode, fmha_quant_mode), energy=0.0)
        else:
            try:
                if self._wideep_generation_mla_data is None:
                    raise PerfDataNotAvailableError(
                        f"WiDeep generation MLA perf table is missing for system='{self.system}', "
                        f"backend='{self.backend}', version='{self.version}'. "
                        "Please use HYBRID or EMPIRICAL database mode, or provide the data file."
                    )
                if attention_backend is None:
                    attention_backend = "flashinfer"
                if attention_backend == "flashinfer":
                    attn_data = self._wideep_generation_mla_data["flashinfer"]
                elif attention_backend == "fa3":
                    attn_data = self._wideep_generation_mla_data["fa3"]
                else:
                    raise ValueError(f"Unsupported attention backend: {attention_backend}")
                # Convert tp_size to num_heads (assuming 128 total heads for DeepSeek)
                num_heads = 128 // tp_size
                mla_dict = attn_data[kvcache_quant_mode]
                result = self._interp_3d(num_heads, b, s, mla_dict, "bilinear")
                latency = result["latency"]
                energy = result.get("energy", 0.0)
            except Exception:
                if database_mode == common.DatabaseMode.HYBRID:
                    logger.debug(
                        f"Failed to query wideep generation mla data for {b=}, {s=}, {tp_size=}, "
                        f"{kvcache_quant_mode=}, {fmha_quant_mode=}, using empirical mode"
                    )
                    latency = get_empirical(b, s, tp_size, kvcache_quant_mode, fmha_quant_mode)
                    energy = 0.0
                else:
                    logger.exception(
                        f"Failed to query wideep generation mla data for {b=}, {s=}, {tp_size=}, \
                        {kvcache_quant_mode=}, {fmha_quant_mode=}, {database_mode=}. Please consider Hybrid mode."
                    )
                    raise
            return PerformanceResult(latency, energy=energy)

    @functools.lru_cache(maxsize=32768)
    def query_wideep_context_mla(
        self,
        b: int,
        s: int,
        prefix: int,
        tp_size: int,
        kvcache_quant_mode: common.KVCacheQuantMode,
        fmha_quant_mode: common.FMHAQuantMode,
        attention_backend: str | None = None,
        database_mode: common.DatabaseMode | None = None,
    ) -> PerformanceResult | tuple[float, float, float]:
        def get_sol(
            b: int,
            s: int,
            prefix: int,
            tp_size: int,
            kvcache_quant_mode: common.KVCacheQuantMode,
            fmha_quant_mode: common.FMHAQuantMode,
        ) -> tuple[float, float, float]:
            hidden_size = 7168
            q_lora_rank = 1536
            kv_lora_rank = 512
            qk_rope_head_dim = 64
            qk_nope_head_dim = 128
            v_head_dim = 128
            num_head = 128 // tp_size

            # qkv_a projection (prefill mode)
            qkv_a_flop = 2 * hidden_size * (q_lora_rank + kv_lora_rank + qk_rope_head_dim) * b * s
            qkv_a_mem = (
                b * hidden_size * s
                + hidden_size * (q_lora_rank + kv_lora_rank + qk_rope_head_dim)
                + 2 * b * (q_lora_rank + kv_lora_rank + qk_rope_head_dim) * s
            )

            # q_b projection
            q_b_flop = 2 * q_lora_rank * num_head * (qk_rope_head_dim + qk_nope_head_dim) * b * s
            q_b_mem = (
                b * q_lora_rank * s
                + q_lora_rank * num_head * (qk_rope_head_dim + qk_nope_head_dim)
                + 2 * b * num_head * (qk_rope_head_dim + qk_nope_head_dim) * s
            )

            # kv_b projection
            kv_b_flop = 2 * kv_lora_rank * num_head * (qk_nope_head_dim + v_head_dim) * b * s
            kv_b_mem = (
                b * s * kv_lora_rank
                + num_head * (qk_nope_head_dim + v_head_dim) * kv_lora_rank
                + 2 * b * num_head * (qk_nope_head_dim + v_head_dim) * s
            )

            # attention computation (prefill mode)
            full_s = s + prefix
            attn_flop = (
                2 * num_head * (qk_nope_head_dim * 2 + qk_rope_head_dim) * b * (full_s * full_s - prefix * prefix) // 2
            )
            attn_mem = (
                b * s * num_head * (qk_nope_head_dim + qk_rope_head_dim)  # q read
                + b * full_s * num_head * (qk_nope_head_dim + qk_rope_head_dim)  # k read
                + b * full_s * num_head * qk_nope_head_dim  # v read
                + b * s * num_head * qk_nope_head_dim  # write
            )

            # attention output projection
            attn_out_flop = 2 * num_head * v_head_dim * hidden_size * b * s
            attn_out_mem = b * num_head * v_head_dim * s + num_head * v_head_dim * hidden_size + 2 * b * hidden_size * s

            ops = qkv_a_flop + q_b_flop + kv_b_flop + attn_out_flop
            mem_bytes = (qkv_a_mem + q_b_mem + kv_b_mem + attn_mem * 2 + attn_out_mem) * fmha_quant_mode.value.memory
            sol_math = ops / (self.system_spec["gpu"]["float16_tc_flops"] * fmha_quant_mode.value.compute) * 1000
            sol_math += attn_flop / (self.system_spec["gpu"]["float16_tc_flops"]) * 1000
            sol_mem = mem_bytes / self.system_spec["gpu"]["mem_bw"] * 1000
            sol_time = max(sol_math, sol_mem)
            return sol_time, sol_math, sol_mem

        def get_empirical(
            b: int,
            s: int,
            prefix: int,
            tp_size: int,
            kvcache_quant_mode: common.KVCacheQuantMode,
            fmha_quant_mode: common.FMHAQuantMode,
        ) -> float:
            """
            Get the hybrid time
            """
            latency = get_sol(b, s, prefix, tp_size, kvcache_quant_mode, fmha_quant_mode)[0]
            scale_factor = 0.6
            return latency / scale_factor

        if database_mode is None:
            database_mode = self._default_database_mode
        if database_mode == common.DatabaseMode.SOL:
            sol_time = get_sol(b, s, prefix, tp_size, kvcache_quant_mode, fmha_quant_mode)[0]
            return PerformanceResult(sol_time, energy=0.0)
        elif database_mode == common.DatabaseMode.SOL_FULL:
            return get_sol(b, s, prefix, tp_size, kvcache_quant_mode, fmha_quant_mode)
        elif database_mode == common.DatabaseMode.EMPIRICAL:
            return PerformanceResult(
                get_empirical(b, s, prefix, tp_size, kvcache_quant_mode, fmha_quant_mode),
                energy=0.0,
            )
        else:
            try:
                if self._wideep_context_mla_data is None:
                    raise PerfDataNotAvailableError(
                        f"WiDeep context MLA perf table is missing for system='{self.system}', "
                        f"backend='{self.backend}', version='{self.version}'. "
                        "Please use HYBRID or EMPIRICAL database mode, or provide the data file."
                    )
                if attention_backend is None:
                    attention_backend = "flashinfer"
                if attention_backend == "flashinfer":
                    attn_data = self._wideep_context_mla_data["flashinfer"]
                elif attention_backend == "fa3":
                    attn_data = self._wideep_context_mla_data["fa3"]
                else:
                    raise ValueError(f"Unsupported attention backend: {attention_backend}")

                # Convert tp_size to num_heads (assuming 128 total heads for DeepSeek)
                num_heads = 128 // tp_size
                mla_dict = attn_data[fmha_quant_mode][kvcache_quant_mode]
                full_s = s + prefix
                prefix_correction = (full_s * full_s - prefix * prefix) / (full_s * full_s)
                result = self._interp_3d(num_heads, full_s, b, mla_dict, "cubic")
                latency = result["latency"] * prefix_correction
                energy = result.get("energy", 0.0) * prefix_correction
            except Exception:
                if database_mode == common.DatabaseMode.HYBRID:
                    logger.debug(
                        f"Failed to query wideep context mla data for {b=}, {s=}, {prefix=}, {tp_size=}, "
                        f"{kvcache_quant_mode=}, {fmha_quant_mode=}, using empirical mode"
                    )
                    latency = get_empirical(b, s, prefix, tp_size, kvcache_quant_mode, fmha_quant_mode)
                    energy = 0.0
                else:
                    logger.exception(
                        f"Failed to query wideep context mla data for {b=}, {s=}, {prefix=}, {tp_size=}, \
                        {kvcache_quant_mode=}, {fmha_quant_mode=}, {database_mode=}. Please consider Hybrid mode."
                    )
                    raise
            return PerformanceResult(latency, energy=energy)

    # to simplify, we no longer support allreduce_strategy
    @functools.lru_cache(maxsize=32768)
    def query_custom_allreduce(
        self,
        quant_mode: common.CommQuantMode,
        tp_size: int,
        size: int,
        database_mode: common.DatabaseMode | None = None,
    ) -> PerformanceResult | tuple[float, float, float]:
        """
        Query custom AllReduce operation latency and energy.

        Args:
            quant_mode: Communication quantization mode
            tp_size: Tensor parallelism size
            size: Number of elements to reduce
            database_mode: Database mode (SILICON, EMPIRICAL, SOL, HYBRID)

        Returns:
            PerformanceResult: Acts as float (latency in ms).
                              Energy accessible via .energy attribute (W·ms).
        """

        def get_sol(quant_mode: common.CommQuantMode, tp_size: int, size: int) -> tuple[float, float, float]:
            """
            Get the sol time, sol math and sol mem
            """
            if tp_size == 1:
                return 0, 0, 0
            # count, not size in bytes
            p2p_bw = self._get_p2p_bandwidth(tp_size)

            # assume all are ring allreduce, ignore constant latency
            # (~1us for hopper, ~2us for two-die blackwell)
            # assume float16
            sol_time = 2 * size * 2 / tp_size * (tp_size - 1) / p2p_bw
            return sol_time * 1000, 0, 0

        def get_empirical(quant_mode: common.CommQuantMode, tp_size: int, size: int) -> float:
            """
            Get the empirical time
            """
            latency = get_sol(quant_mode, tp_size, size)[0]
            scale_factor = 0.8
            return latency / scale_factor

        if database_mode is None:
            database_mode = self._default_database_mode
        if database_mode == common.DatabaseMode.SOL:
            sol_latency = get_sol(quant_mode, tp_size, size)[0]
            return PerformanceResult(sol_latency, energy=0.0)
        elif database_mode == common.DatabaseMode.SOL_FULL:
            return get_sol(quant_mode, tp_size, size)
        elif database_mode == common.DatabaseMode.EMPIRICAL:
            emp_latency = get_empirical(quant_mode, tp_size, size)
            return PerformanceResult(emp_latency, energy=0.0)
        else:
            try:
                if tp_size == 1:
                    return PerformanceResult(0.0, energy=0.0)
                if self.system_spec["node"]["num_gpus_per_node"] == 72 and tp_size > 4:
                    # on GB200, we only have custom all reduce for up to tp4.
                    return self.query_nccl(quant_mode, tp_size, "all_reduce", size)

                if self._custom_allreduce_data is None:
                    raise PerfDataNotAvailableError(
                        f"Custom allreduce perf table is missing for system='{self.system}', "
                        f"backend='{self.backend}', version='{self.version}'. "
                        "Please use HYBRID or EMPIRICAL database mode, or provide the data file."
                    )

                comm_dict = self._custom_allreduce_data[quant_mode][min(tp_size, 8)][
                    "AUTO"
                ]  # use AUTO for allreduce strategy
                size_left, size_right = self._nearest_1d_point_helper(size, list(comm_dict.keys()), inner_only=False)
                result = self._interp_1d([size_left, size_right], [comm_dict[size_left], comm_dict[size_right]], size)

                # Extract latency and energy
                if isinstance(result, dict):
                    lat = result["latency"]
                    energy = result.get("energy", 0.0)
                else:
                    lat = result
                    energy = 0.0

                if tp_size > self.system_spec["node"]["num_gpus_per_node"]:
                    base_bw = self._get_p2p_bandwidth(self.system_spec["node"]["num_gpus_per_node"])
                    target_bw = self._get_p2p_bandwidth(tp_size)
                    scale_factor = (
                        (tp_size - 1)
                        / tp_size
                        * self.system_spec["node"]["num_gpus_per_node"]
                        / (self.system_spec["node"]["num_gpus_per_node"] - 1)
                        * base_bw
                        / target_bw
                    )
                    lat = lat * scale_factor
                    energy = energy * scale_factor

                return PerformanceResult(lat, energy=energy)
            except Exception:
                if database_mode == common.DatabaseMode.HYBRID:
                    logger.debug(
                        f"Failed to query custom allreduce data for {quant_mode=}, {tp_size=}, {size=}, \
                        {database_mode=}, using empirical mode"
                    )
                    lat = get_empirical(quant_mode, tp_size, size)
                    return PerformanceResult(lat, energy=0.0)
                else:
                    logger.exception(
                        f"Failed to query custom allreduce data for {quant_mode=}, {tp_size=}, {size=}, \
                        {database_mode=}. Please consider Hybrid mode."
                    )
                    raise

    @functools.lru_cache(maxsize=32768)
    def query_nccl(
        self,
        dtype: common.CommQuantMode,
        num_gpus: int,
        operation: str,
        message_size: int,  # element number
        database_mode: common.DatabaseMode | None = None,
    ) -> PerformanceResult | tuple[float, float, float]:
        """
        Query NCCL collective communication latency and energy.

        Args:
            dtype: Communication quantization mode
            num_gpus: Number of GPUs in collective
            operation: NCCL operation type ("all_reduce", "all_gather", etc.)
            message_size: Number of elements to communicate
            database_mode: Database mode (SILICON, EMPIRICAL, SOL, HYBRID)

        Returns:
            PerformanceResult: Acts as float (latency in ms).
                              Energy accessible via .energy attribute (W·ms).
                              Power can be computed as energy/latency (W).

        Example:
            >>> result = db.query_nccl(CommQuantMode.half, 8, "all_reduce", 16384)
            >>> latency_ms = float(result)
            >>> energy_wms = result.energy
            >>> power_w = result.power
        """

        def get_sol(
            dtype: common.CommQuantMode, num_gpus: int, operation: str, message_size: int
        ) -> tuple[float, float, float]:
            """
            Get the sol time, sol math and sol mem
            message_size: element number
            """
            sol_time = 0.0
            p2p_bw = self._get_p2p_bandwidth(num_gpus)

            if operation == "all_gather" or operation == "alltoall" or operation == "reduce_scatter":
                sol_time = dtype.value.memory * message_size * (num_gpus - 1) / num_gpus / p2p_bw * 1000
            elif operation == "all_reduce":
                sol_time = 2 * dtype.value.memory * message_size * (num_gpus - 1) / num_gpus / p2p_bw * 1000
            return sol_time, 0, sol_time

        def get_empirical(dtype: common.CommQuantMode, num_gpus: int, operation: str, message_size: int) -> float:
            """
            Get the empirical time
            """
            latency = get_sol(dtype, num_gpus, operation, message_size)[0]
            scale_factor = 0.8
            return latency / scale_factor

        if database_mode is None:
            database_mode = self._default_database_mode
        if database_mode == common.DatabaseMode.SOL:
            return PerformanceResult(get_sol(dtype, num_gpus, operation, message_size)[0], energy=0.0)
        elif database_mode == common.DatabaseMode.SOL_FULL:
            return get_sol(dtype, num_gpus, operation, message_size)
        elif database_mode == common.DatabaseMode.EMPIRICAL:
            return PerformanceResult(get_empirical(dtype, num_gpus, operation, message_size), energy=0.0)
        else:
            try:
                if num_gpus == 1:
                    return PerformanceResult(0.0, energy=0.0)

                if self._nccl_data is None:
                    raise PerfDataNotAvailableError(
                        f"NCCL perf table is missing for system='{self.system}', "
                        f"backend='{self.backend}', version='{self.version}'. "
                        "Please use HYBRID or EMPIRICAL database mode, or provide the data file."
                    )

                max_num_gpus = max(self._nccl_data[dtype][operation].keys())
                nccl_dict = self._nccl_data[dtype][operation][min(num_gpus, max_num_gpus)]
                size_left, size_right = self._nearest_1d_point_helper(
                    message_size,
                    list(nccl_dict.keys()),
                    inner_only=False,
                )
                result = self._interp_1d(
                    [size_left, size_right],
                    [nccl_dict[size_left], nccl_dict[size_right]],
                    message_size,
                )

                # Extract latency and energy from result
                if isinstance(result, dict):
                    lat = result["latency"]
                    energy = result.get("energy", 0.0)
                else:
                    lat = result
                    energy = 0.0

                if num_gpus > max_num_gpus:  # need to do some correction
                    logger.debug(f"nccl num_gpus {num_gpus} > max_num_gpus {max_num_gpus}, need to do some correction")
                    # Scale factor based on bandwidth ratio between measured and target GPU counts
                    max_num_gpus_bw = self._get_p2p_bandwidth(max_num_gpus)
                    num_gpus_bw = self._get_p2p_bandwidth(num_gpus)
                    scale_factor = max_num_gpus_bw / num_gpus_bw
                    # Apply the same scaling formula to both latency and energy
                    scaling_formula = (num_gpus - 1) / num_gpus * max_num_gpus / (max_num_gpus - 1) * scale_factor
                    lat = lat * scaling_formula
                    energy = energy * scaling_formula

                return PerformanceResult(lat, energy=energy)
            except Exception:
                if database_mode == common.DatabaseMode.HYBRID:
                    logger.debug(
                        f"Failed to query nccl data for {dtype=}, {num_gpus=}, "
                        f"{operation=}, {message_size=}, using empirical mode"
                    )
                    lat = get_empirical(dtype, num_gpus, operation, message_size)
                    return PerformanceResult(lat, energy=0.0)
                else:
                    logger.exception(
                        f"Failed to query nccl data for {dtype=}, {num_gpus=}, \
                        {operation=}, {message_size=}, {database_mode=}. Please consider Hybrid mode."
                    )
                    raise

    @functools.lru_cache(maxsize=32768)
    def query_moe(
        self,
        num_tokens: int,
        hidden_size: int,
        inter_size: int,
        topk: int,
        num_experts: int,
        moe_tp_size: int,
        moe_ep_size: int,
        quant_mode: common.MoEQuantMode,
        workload_distribution: str,
        is_context: bool = True,
        moe_backend: str | None = None,
        database_mode: common.DatabaseMode | None = None,
        is_gated: bool = True,
        enable_eplb: bool = False,
    ) -> PerformanceResult | tuple[float, float, float]:
        """
        Query MoE (Mixture of Experts) layer latency and energy.

        Args:
            num_tokens: Number of tokens
            hidden_size: Hidden dimension size
            inter_size: Intermediate size
            topk: Number of experts activated per token
            num_experts: Total number of experts
            moe_tp_size: MoE tensor parallelism size
            moe_ep_size: MoE expert parallelism size
            quant_mode: MoE quantization mode
            workload_distribution: Workload distribution pattern
            is_context: Whether this is context (prefill) phase
            moe_backend: MoE backend type (for SGLang)
            database_mode: Database mode (SILICON, EMPIRICAL, SOL, HYBRID)
            is_gated: Whether MoE uses gated activation (SwiGLU=True, Relu2=False).
                      Low-latency kernel only available for gated MoE.
            enable_eplb: Expert Parallel Load Balancing. When enabled, applies
                        num_tokens correction (0.8x) during prefill phase only.

        Returns:
            PerformanceResult: Acts as float (latency in ms).
                              Energy accessible via .energy attribute (W·ms).
        """

        def get_sol(
            num_tokens: int,
            hidden_size: int,
            inter_size: int,
            topk: int,
            num_experts: int,
            moe_tp_size: int,
            moe_ep_size: int,
            quant_mode: common.MoEQuantMode,
            workload_distribution: str,
        ) -> tuple[float, float, float]:
            """
            Get the sol time, sol math and sol mem
            """
            # we ignore router part. only consider mlp
            # tp already impacted inter_size.
            # only consider even workload.
            total_tokens = num_tokens * topk
            ops = total_tokens * hidden_size * inter_size * 3 * 2 // moe_ep_size // moe_tp_size  # ffn1, ffn2, gate
            mem_bytes = quant_mode.value.memory * (
                total_tokens // moe_ep_size * hidden_size * 2  # input+output
                + total_tokens
                // moe_ep_size
                * inter_size
                * 3
                // moe_tp_size  # intermediate, assume ffn1/gate all need to write results.
                + hidden_size
                * inter_size
                * 3
                // moe_tp_size
                * min(num_experts // moe_ep_size, total_tokens // moe_ep_size)
            )
            sol_math = ops / (self.system_spec["gpu"]["float16_tc_flops"] * quant_mode.value.compute) * 1000
            sol_mem = mem_bytes / self.system_spec["gpu"]["mem_bw"] * 1000
            sol_time = max(sol_math, sol_mem)
            return sol_time, sol_math, sol_mem

        def get_empirical(
            num_tokens: int,
            hidden_size: int,
            inter_size: int,
            topk: int,
            num_experts: int,
            moe_tp_size: int,
            moe_ep_size: int,
            quant_mode: common.MoEQuantMode,
            workload_distribution: str,
        ) -> float:
            """
            Get the hybrid time
            """
            latency = get_sol(
                num_tokens,
                hidden_size,
                inter_size,
                topk,
                num_experts,
                moe_tp_size,
                moe_ep_size,
                quant_mode,
                workload_distribution,
            )[0]
            scale_factor = 0.4
            return latency / scale_factor

        if database_mode is None:
            database_mode = self._default_database_mode
        if database_mode == common.DatabaseMode.SOL:
            sol_latency = get_sol(
                num_tokens,
                hidden_size,
                inter_size,
                topk,
                num_experts,
                moe_tp_size,
                moe_ep_size,
                quant_mode,
                workload_distribution,
            )[0]
            return PerformanceResult(sol_latency, energy=0.0)
        elif database_mode == common.DatabaseMode.SOL_FULL:
            return get_sol(
                num_tokens,
                hidden_size,
                inter_size,
                topk,
                num_experts,
                moe_tp_size,
                moe_ep_size,
                quant_mode,
                workload_distribution,
            )
        elif database_mode == common.DatabaseMode.EMPIRICAL:
            emp_latency = get_empirical(
                num_tokens,
                hidden_size,
                inter_size,
                topk,
                num_experts,
                moe_tp_size,
                moe_ep_size,
                quant_mode,
                workload_distribution,
            )
            return PerformanceResult(emp_latency, energy=0.0)
        else:
            try:
                if self.backend == common.BackendName.sglang.value:
                    # deepep_moe is for sglang wideep only
                    # Apply num_tokens correction when eplb is enabled (only during prefill)
                    if enable_eplb and is_context:
                        num_tokens = int(num_tokens * 0.8)
                    if moe_backend == "deepep_moe":
                        if is_context:
                            moe_data = self._wideep_context_moe_data
                        else:
                            moe_data = self._wideep_generation_moe_data
                    else:
                        moe_data = self._moe_data

                    if moe_data is None:
                        raise PerfDataNotAvailableError(
                            f"MoE perf table is missing for system='{self.system}', "
                            f"backend='{self.backend}', version='{self.version}', moe_backend='{moe_backend}'. "
                            "Please use HYBRID or EMPIRICAL database mode, or provide the data file."
                        )

                    used_workload_distribution = (
                        workload_distribution if workload_distribution in moe_data[quant_mode] else "uniform"
                    )
                    moe_dict = moe_data[quant_mode][used_workload_distribution][topk][num_experts][hidden_size][
                        inter_size
                    ][moe_tp_size][moe_ep_size]
                    num_left, num_right = self._nearest_1d_point_helper(
                        num_tokens,
                        list(moe_dict.keys()),
                        inner_only=False,
                    )
                    result = self._interp_1d(
                        [num_left, num_right],
                        [moe_dict[num_left], moe_dict[num_right]],
                        num_tokens,
                    )
                    if isinstance(result, dict):
                        lat = result["latency"]
                        energy = result.get("energy", 0.0)
                    else:
                        lat = result
                        energy = 0.0
                    return PerformanceResult(lat, energy=energy)
                elif self.backend == common.BackendName.trtllm.value:
                    if self._moe_data is None and self._moe_low_latency_data is None:
                        raise PerfDataNotAvailableError(
                            f"MoE perf table is missing for system='{self.system}', "
                            f"backend='{self.backend}', version='{self.version}'. "
                            "Please use HYBRID or EMPIRICAL database mode, or provide the data file."
                        )
                    # aligned with trtllm, kernel source selection.
                    # Low-latency kernel only available for gated MoE (SwiGLU), not for Relu2
                    if (
                        num_tokens <= 128
                        and self._moe_low_latency_data
                        and quant_mode == common.MoEQuantMode.nvfp4
                        and is_gated
                    ):
                        try:
                            used_workload_distribution = (
                                workload_distribution
                                if workload_distribution in self._moe_low_latency_data[quant_mode]
                                else "uniform"
                            )
                            moe_dict = self._moe_low_latency_data[quant_mode][used_workload_distribution][topk][
                                num_experts
                            ][hidden_size][inter_size][moe_tp_size][moe_ep_size]
                            logger.debug(
                                f"trying to find low latency data for moe {quant_mode} "
                                f"{workload_distribution} {topk} {num_experts} {hidden_size} "
                                f"{inter_size} {moe_tp_size} {moe_ep_size} but failed."
                            )
                        except:
                            used_workload_distribution = (
                                workload_distribution
                                if workload_distribution in self._moe_data[quant_mode]
                                else "uniform"
                            )
                            moe_dict = self._moe_data[quant_mode][used_workload_distribution][topk][num_experts][
                                hidden_size
                            ][inter_size][moe_tp_size][moe_ep_size]
                    else:
                        used_workload_distribution = (
                            workload_distribution if workload_distribution in self._moe_data[quant_mode] else "uniform"
                        )
                        moe_dict = self._moe_data[quant_mode][used_workload_distribution][topk][num_experts][
                            hidden_size
                        ][inter_size][moe_tp_size][moe_ep_size]

                    num_left, num_right = self._nearest_1d_point_helper(
                        num_tokens,
                        list(moe_dict.keys()),
                        inner_only=False,
                    )
                    result = self._interp_1d(
                        [num_left, num_right],
                        [moe_dict[num_left], moe_dict[num_right]],
                        num_tokens,
                    )
                    if isinstance(result, dict):
                        lat = result["latency"]
                        energy = result.get("energy", 0.0)
                    else:
                        lat = result
                        energy = 0.0
                    return PerformanceResult(lat, energy=energy)
                elif self.backend == common.BackendName.vllm.value:
                    if self._moe_data is None:
                        raise PerfDataNotAvailableError(
                            f"MoE perf table is missing for system='{self.system}', "
                            f"backend='{self.backend}', version='{self.version}'. "
                            "Please use HYBRID or EMPIRICAL database mode, or provide the data file."
                        )
                    used_workload_distribution = (
                        workload_distribution if workload_distribution in self._moe_data[quant_mode] else "uniform"
                    )
                    moe_dict = self._moe_data[quant_mode][used_workload_distribution][topk][num_experts][hidden_size][
                        inter_size
                    ][moe_tp_size][moe_ep_size]
                    num_left, num_right = self._nearest_1d_point_helper(
                        num_tokens, list(moe_dict.keys()), inner_only=False
                    )
                    result = self._interp_1d(
                        [num_left, num_right], [moe_dict[num_left], moe_dict[num_right]], num_tokens
                    )
                    if isinstance(result, dict):
                        latency = result["latency"]
                        energy = result.get("energy", 0.0)
                    else:
                        latency = result
                        energy = 0.0
                    return PerformanceResult(latency, energy=energy)
                else:
                    raise NotImplementedError(f"backend {self.backend} not supported for moe")
            except Exception:
                if database_mode == common.DatabaseMode.HYBRID:
                    logger.debug(
                        "Failed to query moe data for "
                        f"{num_tokens=}, {hidden_size=}, {inter_size=}, {topk=}, {num_experts=}, "
                        f"{moe_tp_size=}, {moe_ep_size=}, {quant_mode=}, {workload_distribution=}, using empirical mode"
                    )
                    latency = get_empirical(
                        num_tokens,
                        hidden_size,
                        inter_size,
                        topk,
                        num_experts,
                        moe_tp_size,
                        moe_ep_size,
                        quant_mode,
                        workload_distribution,
                    )
                    return PerformanceResult(latency, energy=0.0)
                else:
                    logger.exception(
                        "Failed to query moe data for "
                        f"{num_tokens=}, {hidden_size=}, {inter_size=}, {topk=}, {num_experts=}, "
                        f"{moe_tp_size=}, {moe_ep_size=}, {quant_mode=}, {workload_distribution=}, "
                        f"{database_mode=}. Please consider Hybrid mode."
                    )
                    raise

    @functools.lru_cache(maxsize=32768)
    def query_mla_bmm(
        self,
        num_tokens: int,
        num_heads: int,
        quant_mode: common.GEMMQuantMode,
        if_pre: bool = True,
        database_mode: common.DatabaseMode | None = None,
    ) -> PerformanceResult | tuple[float, float, float]:
        """
        Query MLA batch matrix multiply latency and energy.

        Args:
            num_tokens: Number of tokens
            num_heads: Number of attention heads
            quant_mode: Quantization mode
            if_pre: Whether this is pre or post operation
            database_mode: Database mode (SILICON, EMPIRICAL, SOL, HYBRID)

        Returns:
            PerformanceResult: Acts as float (latency in ms).
                              Energy accessible via .energy attribute (W·ms).
        """

        def get_sol(
            num_tokens: int, num_heads: int, quant_mode: common.GEMMQuantMode, if_pre: bool
        ) -> tuple[float, float, float]:
            """
            Get the sol time, sol math and sol mem
            """
            ops = 2 * num_tokens * num_heads * 128 * 512  # 2 for fma
            mem_bytes = num_heads * (num_tokens * 640 + 128 * 512) * quant_mode.value.memory
            sol_math = ops / (self.system_spec["gpu"]["float16_tc_flops"] * quant_mode.value.compute) * 1000
            sol_mem = mem_bytes / self.system_spec["gpu"]["mem_bw"] * 1000
            sol_time = max(sol_math, sol_mem)
            return sol_time, sol_math, sol_mem

        def get_empirical(
            num_tokens: int,
            num_heads: int,
            quant_mode: common.GEMMQuantMode,
            if_pre: bool,
        ) -> float:
            """
            Get the hybrid time
            """
            latency = get_sol(num_tokens, num_heads, quant_mode, if_pre)[0]
            scale_factor = 0.8
            return latency / scale_factor

        if database_mode is None:
            database_mode = self._default_database_mode
        if database_mode == common.DatabaseMode.SOL:
            sol_latency = get_sol(num_tokens, num_heads, quant_mode, if_pre)[0]
            return PerformanceResult(sol_latency, energy=0.0)
        elif database_mode == common.DatabaseMode.SOL_FULL:
            return get_sol(num_tokens, num_heads, quant_mode, if_pre)
        elif database_mode == common.DatabaseMode.EMPIRICAL:
            emp_latency = get_empirical(num_tokens, num_heads, quant_mode, if_pre)
            return PerformanceResult(emp_latency, energy=0.0)
        else:
            try:
                if self._mla_bmm_data is None:
                    raise PerfDataNotAvailableError(
                        f"MLA BMM perf table is missing for system='{self.system}', "
                        f"backend='{self.backend}', version='{self.version}'. "
                        "Please use HYBRID or EMPIRICAL database mode, or provide the data file."
                    )
                if quant_mode not in self._mla_bmm_data:
                    quant_mode = common.GEMMQuantMode.float16
                mla_bmm_dict = self._mla_bmm_data[quant_mode]["mla_gen_pre" if if_pre else "mla_gen_post"][num_heads]
                num_left, num_right = self._nearest_1d_point_helper(
                    num_tokens,
                    list(mla_bmm_dict.keys()),
                    inner_only=False,
                )
                result = self._interp_1d(
                    [num_left, num_right],
                    [mla_bmm_dict[num_left], mla_bmm_dict[num_right]],
                    num_tokens,
                )
                if isinstance(result, dict):
                    lat = result["latency"]
                    energy = result.get("energy", 0.0)
                else:
                    lat = result
                    energy = 0.0
                return PerformanceResult(lat, energy=energy)
            except Exception:
                if database_mode == common.DatabaseMode.HYBRID:
                    logger.debug(
                        f"Failed to query mla bmm data for {num_tokens=}, {num_heads=}, {quant_mode=}, "
                        f"{if_pre=}, using empirical mode"
                    )
                    lat = get_empirical(num_tokens, num_heads, quant_mode, if_pre)
                    return PerformanceResult(lat, energy=0.0)
                else:
                    logger.exception(
                        f"Failed to query mla bmm data for {num_tokens=}, {num_heads=}, {quant_mode=}, \
                        {if_pre=}, {database_mode=}. Please consider Hybrid mode."
                    )
                    raise

    @functools.lru_cache(maxsize=32768)
    def query_mem_op(
        self, mem_bytes: int, database_mode: common.DatabaseMode | None = None
    ) -> PerformanceResult | tuple[float, float, float]:
        """
        Query memory operation latency and energy.

        Args:
            mem_bytes: Number of bytes to transfer
            database_mode: Database mode (SILICON, EMPIRICAL, SOL, HYBRID)

        Returns:
            PerformanceResult: Acts as float (latency in ms).
                              Energy accessible via .energy attribute (W·ms).
        """

        def get_sol(mem_bytes: int) -> tuple[float, float, float]:
            """
            Get the sol time, sol math and sol mem
            """
            sol_time = mem_bytes / self.system_spec["gpu"]["mem_bw"] * 1000
            return sol_time, 0, sol_time

        def get_empirical(mem_bytes: int) -> float:
            """
            Get the empirical time
            """
            return (
                mem_bytes
                / (self.system_spec["gpu"]["mem_bw"] * self.system_spec["gpu"]["mem_bw_empirical_scaling_factor"])
                + self.system_spec["gpu"]["mem_empirical_constant_latency"]
            ) * 1000

        if database_mode is None:
            database_mode = self._default_database_mode
        if database_mode == common.DatabaseMode.SOL:
            return PerformanceResult(get_sol(mem_bytes)[0], energy=0.0)
        elif database_mode == common.DatabaseMode.SOL_FULL:
            return get_sol(mem_bytes)
        elif database_mode == common.DatabaseMode.EMPIRICAL:
            return PerformanceResult(get_empirical(mem_bytes), energy=0.0)
        else:
            # hybrid and silicon modes have same logic
            return PerformanceResult(get_empirical(mem_bytes), energy=0.0)

    def query_mamba2(
        self,
        phase: str,
        kernel_source: str,
        batch_size: int,
        seq_len: int | None,
        d_model: int,
        d_state: int,
        d_conv: int,
        nheads: int,
        head_dim: int,
        n_groups: int,
        chunk_size: int,
    ) -> PerformanceResult:
        """
        Query Mamba2 kernel (Conv1D or SSM) latency and energy.

        Args:
            phase: "context" or "generation"
            kernel_source: "causal_conv1d_fn", "mamba_chunk_scan_combined",
                           "causal_conv1d_update", or "selective_state_update"
            batch_size: batch size
            seq_len: sequence length (context only; use 0 or any for generation)
            d_model, d_state, d_conv, nheads, head_dim, n_groups, chunk_size: model config

        Returns:
            PerformanceResult with latency (ms) and energy (W·ms).
            Uses SOL-based fallback when mamba2_perf data is not loaded.
        """
        mamba2_data = getattr(self, "_mamba2_data", None)

        def _sol_fallback() -> PerformanceResult:
            # SOL estimate for this kernel only (conv1d or ssm)
            d_inner = nheads * head_dim
            conv_dim = d_inner + 2 * n_groups * d_state
            x = (batch_size * seq_len) if phase == "context" and seq_len else batch_size
            if kernel_source in ("causal_conv1d_fn", "causal_conv1d_update"):
                conv_read_bytes = x * conv_dim * (d_conv + 1) * 2
                conv_write_bytes = x * conv_dim * 2
                return self.query_mem_op(conv_read_bytes + conv_write_bytes)
            else:
                ssm_read_bytes = x * (d_inner + n_groups * d_state * 2 + nheads) * 2
                ssm_write_bytes = x * d_inner * 2
                return self.query_mem_op(ssm_read_bytes + ssm_write_bytes)

        if mamba2_data is None:
            return _sol_fallback()

        model_key = (d_model, d_state, d_conv, nheads, head_dim, n_groups, chunk_size)
        try:
            by_phase = mamba2_data[kernel_source]
        except KeyError:
            return _sol_fallback()
        try:
            by_key = by_phase[phase]
        except KeyError:
            return _sol_fallback()
        if model_key not in by_key:
            # Nearest config by d_model
            keys_with_d_model = [k for k in by_key if k[0] == d_model]
            if keys_with_d_model:
                model_key = keys_with_d_model[0]
            else:
                return _sol_fallback()

        table = by_key[model_key]

        if phase == "context":
            if seq_len is None or seq_len <= 0:
                return _sol_fallback()
            try:
                result = self._interp_2d_linear(batch_size, seq_len, table)
            except (KeyError, ValueError):
                return _sol_fallback()
            return PerformanceResult(
                latency=result["latency"],
                energy=result.get("energy", result.get("power", 0.0) * result["latency"]),
            )
        else:
            try:
                batch_left, batch_right = self._nearest_1d_point_helper(
                    batch_size, list(table.keys()), inner_only=False
                )
            except (KeyError, ValueError):
                return _sol_fallback()

            # Ensure we pass entry dicts {latency, power, energy}; handle legacy nested batch_size -> seq_len -> entry
            def _mamba2_gen_entry(val):
                if isinstance(val, dict) and "latency" in val:
                    return val
                if isinstance(val, dict) and val:
                    inner = next(iter(val.values()))
                    if isinstance(inner, dict) and "latency" in inner:
                        return inner
                return None

            y_left = _mamba2_gen_entry(table[batch_left])
            y_right = _mamba2_gen_entry(table[batch_right])
            if y_left is None or y_right is None:
                return _sol_fallback()
            result = self._interp_1d(
                [batch_left, batch_right],
                [y_left, y_right],
                batch_size,
            )
            if isinstance(result, dict):
                lat = result["latency"]
                energy = result.get("energy", result.get("power", 0.0) * lat)
            else:
                lat = result
                energy = 0.0
            return PerformanceResult(lat, energy=energy)

    @functools.lru_cache(maxsize=32768)
    def query_p2p(
        self, message_bytes: int, database_mode: common.DatabaseMode | None = None
    ) -> PerformanceResult | tuple[float, float, float]:
        """
        Query P2P (point-to-point) communication latency and energy.

        Args:
            message_bytes: Number of bytes to transfer
            database_mode: Database mode (SILICON, EMPIRICAL, SOL, HYBRID)

        Returns:
            PerformanceResult: Acts as float (latency in ms).
                              Energy accessible via .energy attribute (W·ms).
        """

        def get_sol(message_bytes: int) -> tuple[float, float, float]:
            """
            Get the sol time, sol math and sol mem
            """
            # TODO, use intra_node_bw if num_gpus < num_gpus_per_node
            sol_time = message_bytes / self.system_spec["node"]["inter_node_bw"] * 1000
            return sol_time, 0, sol_time

        def get_empirical(message_bytes: int) -> float:
            """
            Get the empirical time
            """
            return (
                message_bytes / self.system_spec["node"]["inter_node_bw"] + self.system_spec["node"]["p2p_latency"]
            ) * 1000

        if database_mode is None:
            database_mode = self._default_database_mode
        if database_mode == common.DatabaseMode.SOL:
            return PerformanceResult(get_sol(message_bytes)[0], energy=0.0)
        elif database_mode == common.DatabaseMode.SOL_FULL:
            return get_sol(message_bytes)
        elif database_mode == common.DatabaseMode.EMPIRICAL:
            return PerformanceResult(get_empirical(message_bytes), energy=0.0)
        else:
            # hybrid and silicon modes have same logic
            return PerformanceResult(get_empirical(message_bytes), energy=0.0)

    @functools.lru_cache(maxsize=32768)
    def query_wideep_deepep_ll(
        self,
        node_num: int,
        num_tokens: int,
        num_experts: int,
        topk: int,
        hidden_size: int,
        database_mode: common.DatabaseMode | None = None,
    ) -> PerformanceResult | tuple[float, float, float]:
        """
        Query the DeepEP LL operation data
        """

        def get_sol(num_tokens: int, topk: int, num_experts: int) -> tuple[float, float, float]:
            raise NotImplementedError("WideEP deepep ll operation's sol is not implemented yet")
            return

        def get_empirical(num_tokens: int, topk: int, num_experts: int) -> float:
            raise NotImplementedError("WideEP deepep ll operation's empirical is not implemented yet")
            return

        if database_mode is None:
            database_mode = self._default_database_mode
        if database_mode == common.DatabaseMode.SOL:
            return PerformanceResult(get_sol(num_tokens, topk, num_experts)[0], energy=0.0)
        elif database_mode == common.DatabaseMode.SOL_FULL:
            return get_sol(num_tokens, topk, num_experts)
        elif database_mode == common.DatabaseMode.EMPIRICAL:
            return PerformanceResult(get_empirical(num_tokens, topk, num_experts), energy=0.0)
        else:
            data = self._wideep_deepep_ll_data[node_num][hidden_size][topk][num_experts]
            num_left, num_right = self._nearest_1d_point_helper(num_tokens, list(data.keys()), inner_only=False)
            result = self._interp_1d([num_left, num_right], [data[num_left], data[num_right]], num_tokens)
            lat = result["latency"] if isinstance(result, dict) else result
            energy = result.get("energy", 0.0) if isinstance(result, dict) else 0.0
            return PerformanceResult(lat / 1000.0, energy=energy / 1000.0)

    @functools.lru_cache(maxsize=32768)
    def query_wideep_deepep_normal(
        self,
        node_num: int,
        num_tokens: int,
        num_experts: int,
        topk: int,
        hidden_size: int,
        sms: int,
        database_mode: common.DatabaseMode | None = None,
    ) -> PerformanceResult | tuple[float, float, float]:
        """
        Query the DeepEP normal operation data
        """

        def get_sol(num_tokens: int, num_experts: int, topk: int, hidden_size: int) -> tuple[float, float, float]:
            raise NotImplementedError("WideEP deepep normal operation's sol is not implemented yet")
            return

        def get_empirical(num_tokens: int, num_experts: int, topk: int, hidden_size: int) -> float:
            raise NotImplementedError("WideEP deepep normal operation's empirical is not implemented yet")
            return

        if database_mode is None:
            database_mode = self._default_database_mode
        if database_mode == common.DatabaseMode.SOL:
            return PerformanceResult(get_sol(num_tokens, num_experts, topk, hidden_size)[0], energy=0.0)
        elif database_mode == common.DatabaseMode.SOL_FULL:
            return get_sol(num_tokens, num_experts, topk, hidden_size)
        elif database_mode == common.DatabaseMode.EMPIRICAL:
            return PerformanceResult(get_empirical(num_tokens, num_experts, topk, hidden_size), energy=0.0)
        else:
            if node_num == 1 and sms == 20:  # only collect sm=20 for now
                data = self._wideep_deepep_normal_data[node_num][hidden_size][topk][num_experts][sms]
                num_left, num_right = self._nearest_1d_point_helper(num_tokens, list(data.keys()), inner_only=False)
                result = self._interp_1d([num_left, num_right], [data[num_left], data[num_right]], num_tokens)
                lat = result["latency"] if isinstance(result, dict) else result
                energy = result.get("energy", 0.0) if isinstance(result, dict) else 0.0
            else:
                data = self._wideep_deepep_normal_data[node_num][hidden_size][topk][num_experts]
                result = self._interp_2d_linear(sms, num_tokens, data)
                lat = result["latency"] if isinstance(result, dict) else result
                energy = result.get("energy", 0.0) if isinstance(result, dict) else 0.0
            return PerformanceResult(lat / 1000.0, energy=energy / 1000.0)

    def _correct_data(self) -> None:
        """
        Correct the data based on sol time reference.
        """
        # regular gemm
        if self._gemm_data is not None:
            for quant_mode in self._gemm_data:
                for m in self._gemm_data[quant_mode]:
                    for n in self._gemm_data[quant_mode][m]:
                        for k in self._gemm_data[quant_mode][m][n]:
                            sol = self.query_gemm(m, n, k, quant_mode, database_mode=common.DatabaseMode.SOL)
                            data = self._gemm_data[quant_mode][m][n][k]
                            current_latency = data["latency"] if isinstance(data, dict) else data
                            if sol > current_latency:
                                logger.debug(
                                    f"gemm quant {quant_mode} m{m} n{n} k{k}: sol {sol} > perf_db {current_latency}"
                                )
                                if isinstance(data, dict):
                                    # Update only latency, keep power unchanged
                                    # Convert PerformanceResult to float
                                    self._gemm_data[quant_mode][m][n][k]["latency"] = float(max(sol, current_latency))
                                else:
                                    # Legacy format (float)
                                    self._gemm_data[quant_mode][m][n][k] = float(max(sol, current_latency))

        # regular generation attention
        if self._generation_attention_data is not None:
            for quant_mode in self._generation_attention_data:
                for n_kv in self._generation_attention_data[quant_mode]:
                    for head_size in self._generation_attention_data[quant_mode][n_kv]:
                        for window_size in self._generation_attention_data[quant_mode][n_kv][head_size]:
                            for n in self._generation_attention_data[quant_mode][n_kv][head_size][window_size]:
                                for b in self._generation_attention_data[quant_mode][n_kv][head_size][window_size][n]:
                                    for s in self._generation_attention_data[quant_mode][n_kv][head_size][window_size][
                                        n
                                    ][b]:
                                        if n_kv == 0:
                                            n_kv_local = n
                                        else:
                                            n_kv_local = n_kv
                                        sol = self.query_generation_attention(
                                            b,
                                            s,
                                            n,
                                            n_kv_local,
                                            quant_mode,
                                            database_mode=common.DatabaseMode.SOL,
                                            window_size=window_size,
                                            head_size=head_size,
                                        )
                                        data = self._generation_attention_data[quant_mode][n_kv][head_size][
                                            window_size
                                        ][n][b][s]
                                        current_latency = data["latency"] if isinstance(data, dict) else data
                                        if sol > current_latency:
                                            logger.debug(
                                                f"generation attention quant {quant_mode} n{n} "
                                                f"n_kv{n_kv_local} b{b} s{s}: sol {sol} > "
                                                f"perf_db {current_latency}"
                                            )
                                            if isinstance(data, dict):
                                                # Update only latency, keep power unchanged
                                                # Convert PerformanceResult to float
                                                self._generation_attention_data[quant_mode][n_kv][head_size][
                                                    window_size
                                                ][n][b][s]["latency"] = float(sol)
                                            else:
                                                # Legacy format (float)
                                                self._generation_attention_data[quant_mode][n_kv][head_size][
                                                    window_size
                                                ][n][b][s] = float(sol)

    def query_wideep_moe_compute(
        self,
        num_tokens: int,
        hidden_size: int,
        inter_size: int,
        topk: int,
        num_experts: int,
        num_slots: int,
        moe_tp_size: int,
        moe_ep_size: int,
        quant_mode: common.MoEQuantMode,
        workload_distribution: str,
        database_mode: common.DatabaseMode | None = None,
    ) -> PerformanceResult:
        """
        Query WideEP MoE compute latency (pure computation, excluding All2All communication).

        This is for TensorRT-LLM WideEP scenarios with three EPLB modes:
        - EPLB off: workload_distribution without "_eplb" suffix, num_slots = num_experts
        - EPLB on: workload_distribution with "_eplb" suffix, num_slots = num_experts
        - EPLB redundant: workload_distribution with "_eplb" suffix, num_slots > num_experts

        The MoE computation kernel is automatically selected based on GPU architecture and quantization mode:
        - SM >= 100 (Blackwell) with fp8_block -> DeepGemm kernel
        - Otherwise -> Cutlass kernel (wideep_compute_cutlass)

        Args:
            num_tokens: Number of tokens
            hidden_size: Hidden dimension size
            inter_size: Intermediate size (FFN)
            topk: Number of experts activated per token
            num_experts: Total number of experts
            num_slots: Number of expert slots (= num_experts for EPLB off/on, > num_experts for EPLB redundant)
            moe_tp_size: MoE tensor parallelism size
            moe_ep_size: MoE expert parallelism size
            quant_mode: MoE quantization mode
            workload_distribution: Workload distribution pattern (e.g., "power_law_1.01" or "power_law_1.01_eplb")
            database_mode: Database mode (SILICON, EMPIRICAL, SOL, HYBRID)

        Returns:
            PerformanceResult: Latency in ms, energy accessible via .energy attribute.
        """
        if database_mode is None:
            database_mode = self._default_database_mode

        if self._wideep_moe_compute_data is None:
            raise PerfDataNotAvailableError(
                f"WideEP MoE compute perf table is missing for system='{self.system}', "
                f"backend='{self.backend}', version='{self.version}'. "
                "Please provide the wideep_moe_compute_perf.txt data file."
            )

        # Automatically select MoE kernel based on GPU architecture and quant mode
        kernel_source = self._select_moe_kernel(quant_mode)
        logger.debug(f"query_wideep_moe_compute: auto-selected kernel_source='{kernel_source}'")

        try:
            # Find the best matching distribution
            kernel_data = self._wideep_moe_compute_data[kernel_source]
            available_distributions = list(kernel_data[quant_mode].keys())
            if workload_distribution in available_distributions:
                used_distribution = workload_distribution
            else:
                # Fallback: try to find a similar distribution or use the first available
                used_distribution = available_distributions[0] if available_distributions else None
                if used_distribution is None:
                    raise KeyError(f"No distribution available for kernel={kernel_source}, quant_mode={quant_mode}")
                logger.debug(f"Distribution '{workload_distribution}' not found, using '{used_distribution}' instead")

            moe_dict = kernel_data[quant_mode][used_distribution][topk][num_experts][hidden_size][inter_size][
                num_slots
            ][moe_tp_size][moe_ep_size]

            num_left, num_right = self._nearest_1d_point_helper(
                num_tokens,
                list(moe_dict.keys()),
                inner_only=False,
            )
            result = self._interp_1d(
                [num_left, num_right],
                [moe_dict[num_left], moe_dict[num_right]],
                num_tokens,
            )

            if isinstance(result, dict):
                lat = result["latency"]
                energy = result.get("energy", 0.0)
            else:
                lat = result
                energy = 0.0

            return PerformanceResult(lat, energy=energy)

        except Exception as e:
            if database_mode == common.DatabaseMode.HYBRID:
                logger.debug(
                    f"Failed to query wideep moe compute data (kernel={kernel_source}): {e}, "
                    "falling back to empirical estimation"
                )
                # Simple empirical fallback based on SOL
                total_tokens = num_tokens * topk
                ops = total_tokens * hidden_size * inter_size * 3 * 2 // moe_ep_size // moe_tp_size
                sol_math = ops / (self.system_spec["gpu"]["float16_tc_flops"] * quant_mode.value.compute) * 1000
                lat = sol_math / 0.4  # Empirical scale factor
                return PerformanceResult(lat, energy=0.0)
            else:
                raise

    def query_wideep_alltoall(
        self,
        op_name: str,
        num_tokens: int,
        hidden_size: int,
        topk: int,
        num_experts: int,
        moe_ep_size: int,
        quant_mode: common.MoEQuantMode,
        node_num: int | None = None,
        database_mode: common.DatabaseMode | None = None,
    ) -> PerformanceResult:
        """
        Query WideEP All2All communication latency.

        The All2All communication method is automatically selected based on GPU architecture:
        - SM >= 100 (GB200 with MNNVL) -> MnnvlMoe (NVLink Two-Sided)
        - SM >= 90 (H100/H200) cross-node -> DeepEP
        - SM >= 90 (H100/H200) intra-node -> DeepEPLowLatency
        - SM < 90 -> NCCL (fallback)

        Args:
            op_name: Operation name, one of:
                - "alltoall_prepare": Prepare phase
                - "alltoall_dispatch": Token dispatch phase
                - "alltoall_combine": Result combine phase
                - "alltoall_combine_low_precision": Low precision combine
            num_tokens: Number of tokens
            hidden_size: Hidden dimension size
            topk: Number of experts activated per token
            num_experts: Total number of experts
            moe_ep_size: MoE expert parallelism size (must be divisible by 4 if node_num is None)
            quant_mode: MoE quantization mode
            node_num: Number of nodes. If None, computed as moe_ep_size // 4 (assuming 4 GPUs per node)
            database_mode: Database mode

        Returns:
            PerformanceResult: Latency in ms, energy accessible via .energy attribute.

        Raises:
            ValueError: If moe_ep_size is not divisible by 4 when node_num is None
        """
        if database_mode is None:
            database_mode = self._default_database_mode

        # Compute node_num if not provided
        if node_num is None:
            if moe_ep_size < 4:
                node_num = 1
            else:
                node_num = moe_ep_size // 4
            logger.debug(f"query_wideep_alltoall: node_num not specified, using {node_num} (moe_ep_size={moe_ep_size})")

        if self._wideep_alltoall_data is None:
            raise PerfDataNotAvailableError(
                f"WideEP All2All perf table is missing for system='{self.system}', "
                f"backend='{self.backend}', version='{self.version}'. "
                "Please provide the wideep_alltoall_perf.txt data file."
            )

        valid_op_names = ["alltoall_prepare", "alltoall_dispatch", "alltoall_combine", "alltoall_combine_low_precision"]
        if op_name not in valid_op_names:
            raise ValueError(f"Invalid op_name '{op_name}'. Must be one of {valid_op_names}")

        # Automatically select All2All kernel based on GPU architecture
        kernel_source = self._select_alltoall_kernel(quant_mode, moe_ep_size, topk)
        logger.debug(f"query_wideep_alltoall: auto-selected kernel_source='{kernel_source}'")

        try:
            kernel_data = self._wideep_alltoall_data[kernel_source]
            alltoall_dict = kernel_data[op_name][quant_mode][node_num][hidden_size][topk][num_experts][moe_ep_size]

            num_left, num_right = self._nearest_1d_point_helper(
                num_tokens,
                list(alltoall_dict.keys()),
                inner_only=False,
            )
            result = self._interp_1d(
                [num_left, num_right],
                [alltoall_dict[num_left], alltoall_dict[num_right]],
                num_tokens,
            )

            if isinstance(result, dict):
                lat = result["latency"]
                energy = result.get("energy", 0.0)
            else:
                lat = result
                energy = 0.0

            return PerformanceResult(lat, energy=energy)

        except Exception as e:
            if database_mode == common.DatabaseMode.HYBRID:
                logger.debug(
                    f"Failed to query wideep alltoall data for {op_name} (kernel={kernel_source}), "
                    f"node_num={node_num}: {e}, using empirical estimation"
                )
                # Simple empirical fallback: ~0.1ms per operation as baseline
                lat = 0.1
                return PerformanceResult(lat, energy=0.0)
            else:
                raise


if __name__ == "__main__":
    database_dict = get_all_databases()
