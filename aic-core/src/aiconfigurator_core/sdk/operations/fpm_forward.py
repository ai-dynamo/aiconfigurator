# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Whole-model forward-pass op backed by collected ``fpm_forward_perf`` data.

With ``ModelConfig.forward_model == "fpm"`` the model builder replaces each
phase op list with exactly one :class:`FPMForwardOp`. The op answers the same
``query(database, **kwargs)`` contract as granular ops, but from the formal
FPM database pair written by the collector campaign:

    systems/data/<system>/<backend>/<version>/fpm_forward_perf.parquet
    systems/data/<system>/<backend>/<version>/fpm_forward_perf.metadata.json

Row coordinates are per-DP-rank iteration totals under the ``balanced_v1``
partition policy (every DP rank executes the same point; the stored
``latency_ms`` is the max across DP ranks) — which matches the modeling
convention that ops are queried with the LOCAL per-rank batch:

    prefill: (batch_size, total_prefill_tokens, total_kv_read_tokens)
    decode:  (batch_size, total_kv_read_tokens), one new token per request

Resolution follows the SDK-wide perf_interp contract: exact hit first, then
ScatteredSites interpolation, and a hard ``PerfDataNotAvailableError`` outside
the per-cell collected domain (whole-model latency has no principled
boundary-hold semantics, so the domain gate runs BEFORE perf_interp).

Energy: this dataset is latency-only. Queries return ``energy=0.0``, the same
convention as the Rust engine-step path (``base_backend`` zeroes energy dicts
when routing through Rust).
"""

from __future__ import annotations

import bisect
import functools
import hashlib
import itertools
import json
import math
import os
import statistics
from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING, ClassVar

from aiconfigurator_core.sdk.errors import PerfDataNotAvailableError
from aiconfigurator_core.sdk.operations.base import Operation
from aiconfigurator_core.sdk.perf_interp import OpInterpConfig, ScatteredSites
from aiconfigurator_core.sdk.perf_interp import engine as perf_interp
from aiconfigurator_core.sdk.performance_result import PerformanceResult

if TYPE_CHECKING:
    from aiconfigurator_core.sdk.perf_database import PerfDatabase

FPM_FORWARD_SCHEMA_NAME = "aic_fpm_forward_perf"
FPM_FORWARD_SCHEMA_VERSION = 6
FPM_FORWARD_COORDINATE_SYSTEM = "iteration_totals_balanced_v1"
FPM_FORWARD_PARTITION_POLICY = "balanced_v1"
# The only measurement policy the collector publishes; pinned in the sidecar
# gate so a pair measured under a different regime is a loud structural error.
FPM_FORWARD_MEASUREMENT_POLICY = "dynamo_native_single_sample_v1"
_PHASES = ("prefill", "decode")


# Identity columns that select a cell, in row-column order. ``model_path`` is
# handled separately (see FPMForwardOp._select_cell); ``weight_quantization``
# is redundant with ``gemm_quant_mode`` (the collector falls one back to the
# other) so only ``gemm_quant_mode`` participates in matching.
_CELL_MATCH_COLUMNS = (
    "gemm_quant_mode",
    "moe_quant_mode",
    "fmha_quant_mode",
    "comm_quant_mode",
    "kv_cache_dtype",
    "tp",
    "pp",
    "dp",
    "moe_tp",
    "moe_ep",
    "cp",
    # Explicit backend identity (schema v6): "auto" = the engine decided;
    # pinned values were plumbed to the engine and verified by the collector.
    # The two enable_* columns are real parquet booleans; _norm_identity's
    # str() lowers them to "True"/"False", matching the request side's
    # Python bools.
    "moe_backend",
    "attention_backend",
    "enable_wideep",
    "enable_eplb",
)
# Full physical row key (collector contract) used for duplicate detection.
_ROW_KEY_COLUMNS = (
    "cell_id",
    "model_path",
    "system",
    "backend",
    "backend_version",
    "weight_quantization",
    "gemm_quant_mode",
    "moe_quant_mode",
    "fmha_quant_mode",
    "comm_quant_mode",
    "kv_cache_dtype",
    "tp",
    "pp",
    "dp",
    "moe_tp",
    "moe_ep",
    "cp",
    "moe_backend",
    "attention_backend",
    "enable_wideep",
    "enable_eplb",
    "workload_kind",
    "batch_size",
    "total_prefill_tokens",
    "total_kv_read_tokens",
    "partition_policy",
)


def _norm_backend_request(value, *, engine_default: str | None = None) -> str:
    """Request-side normalization for the string backend identity columns.

    The collector records "auto" when the knob was left to the engine.
    ``engine_default`` folds AIC's spelled-out default (ModelConfig ships
    attention_backend="flashinfer" rather than None) back to "auto" so the
    default config reaches the auto-collected cells.
    """
    if value is None or value == "" or value == engine_default:
        return "auto"
    return str(value)


def _norm_identity(value) -> str:
    """Normalize an identity field for matching: None -> "", Enum -> name."""
    if value is None:
        return ""
    if isinstance(value, Enum):
        return str(value.name)
    return str(value)


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sidecar(
    metadata_path: str,
    parquet_path: str,
    expected_system: str,
    expected_backend: str,
    expected_version: str,
) -> dict:
    """The sidecar is the writer's commit record: an unmatched pair (e.g. after
    an interrupted writer) must be rejected, not silently served. The commit
    record also names the database identity it was published for — metadata
    that contradicts the resolved database (a pair copied into the wrong tree
    with its sidecar) is rejected here, before any row is read."""
    if not os.path.exists(metadata_path):
        raise ValueError(
            f"FPM database is missing its metadata sidecar: {metadata_path}. "
            "The parquet/metadata pair is atomic; refusing to load an unmatched parquet."
        )
    with open(metadata_path, encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        # Retype-exempt: load_fpm_forward_data documents a uniform ValueError
        # for every structural violation of the parquet/metadata pair.
        raise ValueError(  # noqa: TRY004
            f"FPM metadata sidecar must be a JSON object: {metadata_path}"
        )
    if metadata.get("schema_name") != FPM_FORWARD_SCHEMA_NAME:
        raise ValueError(
            f"unsupported FPM schema_name={metadata.get('schema_name')!r} "
            f"(expected {FPM_FORWARD_SCHEMA_NAME!r}): {metadata_path}"
        )
    if metadata.get("schema_version") != FPM_FORWARD_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported FPM schema_version={metadata.get('schema_version')!r} "
            f"(expected {FPM_FORWARD_SCHEMA_VERSION}): {metadata_path}"
        )
    if metadata.get("coordinate_system") != FPM_FORWARD_COORDINATE_SYSTEM:
        raise ValueError(
            f"unsupported FPM coordinate_system={metadata.get('coordinate_system')!r} "
            f"(expected {FPM_FORWARD_COORDINATE_SYSTEM!r}): {metadata_path}"
        )
    if metadata.get("measurement_policy") != FPM_FORWARD_MEASUREMENT_POLICY:
        raise ValueError(
            f"unsupported FPM measurement_policy={metadata.get('measurement_policy')!r} "
            f"(expected {FPM_FORWARD_MEASUREMENT_POLICY!r}): {metadata_path}"
        )
    for key, expected in (
        ("system", expected_system),
        ("backend", expected_backend),
        ("backend_version", expected_version),
    ):
        if metadata.get(key) != expected:
            raise ValueError(
                f"FPM sidecar {key}={metadata.get(key)!r} does not match the database {key} "
                f"{expected!r}: {metadata_path}"
            )
    actual_sha = _sha256_file(parquet_path)
    if metadata.get("parquet_sha256") != actual_sha:
        raise ValueError(
            f"FPM parquet digest mismatch: sidecar={metadata.get('parquet_sha256')!r} actual={actual_sha!r}. "
            f"The pair at {os.path.dirname(parquet_path)} is inconsistent (interrupted writer?)."
        )
    return metadata


def _validate_row(row: dict, index: int, expected_version: str, expected_system: str, expected_backend: str) -> None:
    phase = row.get("workload_kind")
    if phase not in _PHASES:
        raise ValueError(f"FPM row {index} has unknown workload_kind={phase!r}")
    if row.get("partition_policy") != FPM_FORWARD_PARTITION_POLICY:
        raise ValueError(
            f"FPM row {index} has unsupported partition_policy={row.get('partition_policy')!r} "
            f"(expected {FPM_FORWARD_PARTITION_POLICY!r})"
        )
    if str(row.get("backend_version")) != expected_version:
        raise ValueError(
            f"FPM row {index} backend_version={row.get('backend_version')!r} does not match "
            f"the database version directory {expected_version!r}"
        )
    # `system`/`backend` are part of the physical row key but NOT the cell
    # key, so a misplaced parquet (e.g. an h200 pair copied into a b200 tree)
    # would otherwise merge into the same cells and silently serve wrong
    # latencies. Pin them to the resolved database identity like the version.
    if str(row.get("system")) != expected_system:
        raise ValueError(
            f"FPM row {index} system={row.get('system')!r} does not match the database system {expected_system!r}"
        )
    if str(row.get("backend")) != expected_backend:
        raise ValueError(
            f"FPM row {index} backend={row.get('backend')!r} does not match the database backend {expected_backend!r}"
        )
    # Schema v6 backend identity: the strings must be present ("auto" or a
    # pinned name) and the enables must be REAL booleans — a string "true"
    # would str()-normalize to "true" while the request side produces "True",
    # a silent never-matches identity. Fail loudly instead.
    for column in ("moe_backend", "attention_backend"):
        value = row.get(column)
        if not isinstance(value, str) or not value:
            raise ValueError(f'FPM row {index} {column}={value!r} must be a non-empty string ("auto" or a pinned name)')
    for column in ("enable_wideep", "enable_eplb"):
        value = row.get(column)
        if not isinstance(value, bool):
            # Retype-exempt: load_fpm_forward_data documents a uniform
            # ValueError for every structural violation of the pair.
            raise ValueError(f"FPM row {index} {column}={value!r} must be a boolean")  # noqa: TRY004
    latency = row.get("latency_ms")
    if not isinstance(latency, (int, float)) or not math.isfinite(float(latency)) or float(latency) <= 0:
        raise ValueError(f"FPM row {index} has non-finite/non-positive latency_ms={latency!r}")
    batch = int(row.get("batch_size", 0))
    total_prefill = int(row.get("total_prefill_tokens", -1))
    total_kv = int(row.get("total_kv_read_tokens", -1))
    if batch < 1 or total_prefill < 0 or total_kv < 0:
        raise ValueError(
            f"FPM row {index} has invalid workload coordinates: batch_size={batch}, "
            f"total_prefill_tokens={total_prefill}, total_kv_read_tokens={total_kv}"
        )
    if phase == "prefill" and total_prefill < 1:
        raise ValueError(f"FPM row {index} is a prefill point with no prefill tokens")
    if phase == "decode" and total_prefill != 0:
        raise ValueError(f"FPM row {index} is a decode point carrying prefill tokens")


def load_fpm_forward_data(primary_path: str, expected_version: str, expected_system: str, expected_backend: str):
    """Load and validate the fpm_forward parquet/metadata pair.

    Returns ``None`` when the parquet is absent (normal "no FPM data collected
    for this version" case — surfaces later as PerfDataNotAvailableError via
    the LoadedOpData wrapper). Any structural violation of the pair raises
    ``ValueError`` loudly: a corrupt supported-database entry is a data bug,
    not a fallback condition.
    """
    if not os.path.exists(primary_path):
        return None
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Loading fpm_forward perf data requires the 'pyarrow' package. "
            "Install aiconfigurator with its declared runtime dependencies."
        ) from exc

    metadata_path = os.path.splitext(primary_path)[0] + ".metadata.json"
    metadata = _validate_sidecar(metadata_path, primary_path, expected_system, expected_backend, expected_version)
    try:
        rows = pq.read_table(primary_path).to_pylist()
    except ValueError:
        raise
    except Exception as exc:
        # A truncated/corrupt parquet raises OSError/ArrowException; the
        # documented contract is ValueError for every structural violation
        # of the pair (the sha256 gate passed, so this is a data bug).
        raise ValueError(f"FPM parquet is unreadable: {primary_path}: {exc}") from exc
    if metadata.get("row_count") != len(rows):
        raise ValueError(
            f"FPM row_count mismatch: sidecar={metadata.get('row_count')!r} actual={len(rows)}: {primary_path}"
        )
    if not rows:
        raise ValueError(f"FPM database contains no rows: {primary_path}")

    seen_keys: set[tuple] = set()
    seen_coords: set[tuple] = set()
    cells: dict[tuple, dict] = {}
    for index, row in enumerate(rows):
        _validate_row(row, index, expected_version, expected_system, expected_backend)
        row_key = tuple(_norm_identity(row.get(column)) for column in _ROW_KEY_COLUMNS)
        if row_key in seen_keys:
            raise ValueError(f"FPM database contains a duplicate physical row key: {row_key}")
        seen_keys.add(row_key)

        cell_key = (
            _norm_identity(row.get("model_path")),
            *(_norm_identity(row.get(column)) for column in _CELL_MATCH_COLUMNS),
        )
        cell = cells.setdefault(
            cell_key,
            {
                "model_path": str(row.get("model_path")),
                "match_identity": tuple(_norm_identity(row.get(column)) for column in _CELL_MATCH_COLUMNS),
                "cell_ids": [],
                "tables": {"prefill": {}, "decode": {}},
            },
        )
        cell_id = str(row.get("cell_id"))
        if cell_id not in cell["cell_ids"]:
            cell["cell_ids"].append(cell_id)

        phase = row["workload_kind"]
        batch = int(row["batch_size"])
        total_prefill = int(row["total_prefill_tokens"])
        total_kv = int(row["total_kv_read_tokens"])
        latency = float(row["latency_ms"])
        # The physical row key includes cell_id/weight_quantization, which the
        # cell key does not: two rows differing only there pass duplicate
        # detection yet target the SAME table slot. Producing such rows is the
        # collector's bug to prevent (aggregate_cell dedups; the formal writer
        # keeps per-cell identities disjoint) — but a hand-merged pair must
        # fail loudly here instead of silently last-winning.
        coord_key = (cell_key, phase, batch, total_prefill, total_kv)
        if coord_key in seen_coords:
            raise ValueError(
                f"FPM row {index} collides with an earlier row at the same cell "
                f"coordinates (phase={phase!r}, batch_size={batch}, "
                f"total_prefill_tokens={total_prefill}, total_kv_read_tokens={total_kv}); "
                "refusing to silently overwrite latencies."
            )
        seen_coords.add(coord_key)
        table = cell["tables"][phase]
        if phase == "prefill":
            table.setdefault(batch, {}).setdefault(total_prefill, {})[total_kv] = latency
        else:
            table.setdefault(batch, {})[total_kv] = latency

    for cell in cells.values():
        domains = {}
        for phase, axes in (
            ("prefill", ("batch_size", "total_prefill_tokens", "total_kv_read_tokens")),
            ("decode", ("batch_size", "total_kv_read_tokens")),
        ):
            points = _walk_points(cell["tables"][phase], len(axes))
            if points:
                domains[phase] = tuple(
                    (min(point[axis] for point in points), max(point[axis] for point in points))
                    for axis in range(len(axes))
                )
        cell["domains"] = domains
        # Decode batch-axis regime partition: the ScatteredSites batch
        # distance is regime-blind (a b=600 query would mix one eager vote
        # with three graph votes), so split at the data-encoded capture
        # boundary and let each side interpolate among its own kind. The
        # domain gate stays on the FULL decode box (gate first, route after).
        # Prefill batch-axis clamp certificate: real steps can schedule more
        # whole prefills than the collected batch ceiling (short sequences);
        # at fixed totals the batch coordinate is provably near-flat for
        # attention-light models, and the DATA must show it before the op
        # may clamp. No certificate -> the hard domain gate stays.
        prefill_batches = sorted(cell["tables"]["prefill"])
        cell["prefill_batch_clamp_max"] = (
            prefill_batches[-1] if prefill_batches and _prefill_batch_axis_is_flat(cell["tables"]["prefill"]) else None
        )
        boundary = _detect_decode_regime_boundary(cell["tables"]["decode"])
        cell["decode_regime_boundary"] = boundary
        if boundary is not None:
            decode_table = cell["tables"]["decode"]
            # Two NEW dicts -> two new id(data) keys: the perf_interp site
            # index cache can never serve one side's index for the other.
            cell["tables"]["decode_graph"] = {b: c for b, c in decode_table.items() if b <= boundary}
            cell["tables"]["decode_eager"] = {b: c for b, c in decode_table.items() if b > boundary}

    return {"cells": cells}


def _decode_curve_value(curve: dict, kv: float) -> float:
    """Piecewise-linear evaluation of one decode KV curve at ``kv``.

    Detection-only helper (the query path uses perf_interp); callers
    guarantee ``kv`` lies within the curve's key range.
    """
    keys = sorted(curve)
    if kv <= keys[0]:
        return float(curve[keys[0]])
    if kv >= keys[-1]:
        return float(curve[keys[-1]])
    hi = bisect.bisect_left(keys, kv)
    lo = hi - 1
    k0, k1 = keys[lo], keys[hi]
    if k1 == k0:
        return float(curve[k0])
    w = (kv - k0) / (k1 - k0)
    return float(curve[k0]) + (float(curve[k1]) - float(curve[k0])) * w


def _prefill_batch_axis_is_flat(table: dict) -> bool:
    """Certify from the collected data that the prefill batch axis is flat.

    At fixed TOTAL tokens the batch count only changes how the tokens split
    into causal-attention segments (more, shorter segments = slightly less
    work); GEMM/MoE see the total alone. Whether that effect is negligible
    is a property of the MODEL, so it is certified from the data rather than
    assumed: for each consecutive pair of collected batches, compare their
    total-token curves (per shared KV slice) on the overlapping range —
    median |ratio - 1| <= 5% over >= 3 points across the ladder passes.
    A single collected batch offers no evidence and fails the certificate.

    The comparison happens at matched totals, i.e. within one CUDA-graph
    regime — the capture cliff lives on the token axis and cannot leak in.
    """
    batches = sorted(table)
    if len(batches) < 2:
        return False
    deviations: list[float] = []
    for lo_b, hi_b in itertools.pairwise(batches):
        # per-KV-slice total->latency curves for both batches
        for kv in {kv for totals in table[lo_b].values() for kv in totals} & {
            kv for totals in table[hi_b].values() for kv in totals
        }:
            lower = {total: kvs[kv] for total, kvs in table[lo_b].items() if kv in kvs}
            upper = {total: kvs[kv] for total, kvs in table[hi_b].items() if kv in kvs}
            if not lower or not upper:
                continue
            lo_keys = sorted(lower)
            for total in sorted(upper):
                if lo_keys[0] <= total <= lo_keys[-1]:
                    base = _decode_curve_value(lower, total)
                    if base > 0:
                        deviations.append(abs(float(upper[total]) / base - 1.0))
    return len(deviations) >= 3 and statistics.median(deviations) <= 0.05


def _detect_decode_regime_boundary(table: dict) -> int | None:
    """Find the CUDA-graph capture boundary encoded in the decode grid.

    The collector deliberately samples cliff PAIRS (adjacent batches b, b+1
    straddling the capture limit, e.g. 512/513). On the overlapping KV range
    the eager side runs a whole regime above the graph side (measured
    2.6-3.5x), while pad-up neighbours inside the graph region differ by only
    5-15% — a median ratio >= 2.0 over >= 3 overlapping KV points (the >= 3
    floor keeps one bad row from minting a cliff) cleanly separates the two.
    Exactly one cliff is expected per cell; several mean the data does not
    look like one capture boundary and a human must look.
    """
    batches = sorted(table)
    hits = []
    for b in batches:
        upper = table.get(b + 1)
        if upper is None:
            continue
        lower = table[b]
        lo_keys = sorted(lower)
        ratios = []
        for kv in sorted(upper):
            if lo_keys[0] <= kv <= lo_keys[-1]:
                base = _decode_curve_value(lower, kv)
                if base > 0:
                    ratios.append(float(upper[kv]) / base)
        if len(ratios) >= 3 and statistics.median(ratios) >= 2.0:
            hits.append(b)
    if len(hits) > 1:
        raise ValueError(
            f"ambiguous decode regime cliffs at batches {hits}: expected at most one "
            "CUDA-graph capture boundary per cell; the pair data does not look like "
            "one capture surface — inspect the collection before serving it."
        )
    return hits[0] if hits else None


def _walk_points(table: dict, depth: int) -> list[tuple]:
    points: list[tuple] = []

    def _walk(node, prefix):
        if len(prefix) == depth:
            points.append(tuple(prefix))
            return
        for key, sub in node.items():
            _walk(sub, [*prefix, key])

    _walk(table, [])
    return points


# ---------------------------------------------------------------------------
# perf_interp configs
#
# The collected grid is generated by the Dynamo runtime per batch level: the
# token-axis point sets under different batch sizes are NOT aligned, so the
# data is "sites, each owning its own token curve" — the ScatteredSites shape,
# not a Cartesian Grid. Revisit with the LOO harness once a real cell lands
# (documented as open decision D2 in docs/fpm/aic-fpm-modeling-plan.md).
# ---------------------------------------------------------------------------


def fpm_prefill_config(sol_fn: Callable[[float, float, float], float]) -> OpInterpConfig:
    """Prefill: data[batch][total_prefill][total_kv]. Sites are (batch, kv)
    pairs — P=0 rows sit at kv=0, far from every P>0 site in log space, so
    ordinary-prefill and past-KV-prefill never cross-contaminate. The curve is
    the densely swept new-token axis."""
    return OpInterpConfig(
        axes=("batch_size", "total_prefill_tokens", "total_kv_read_tokens"),
        resolver=ScatteredSites(
            site_axes=("batch_size", "total_kv_read_tokens"),
            curve_axis="total_prefill_tokens",
            # The runtime grid emits orphan coordinates (max-batch stragglers,
            # capacity-endpoint KV) whose "curves" are a few stray points; they
            # must answer only inside their own coverage, never anchor far
            # extrapolation. The distance gate also keeps P=0 queries on P=0
            # neighbour sites (KV=0 is ~40 log2-units from any KV>0 site).
            own_curve_coverage_fallback=True,
            max_site_distance=2.0,
        ),
        sol_fn=sol_fn,
    )


def fpm_decode_config(sol_fn: Callable[[float, float], float]) -> OpInterpConfig:
    """Decode: data[batch][total_kv]. Each batch level is a site owning its
    KV curve (0 + block-aligned powers of two + max)."""
    return OpInterpConfig(
        axes=("batch_size", "total_kv_read_tokens"),
        resolver=ScatteredSites(
            site_axes=("batch_size",),
            curve_axis="total_kv_read_tokens",
            own_curve_coverage_fallback=True,
            max_site_distance=2.0,
        ),
        sol_fn=sol_fn,
    )


def _oplevel_sol_fn(sol_ops: list, phase: str, database) -> Callable[..., float]:
    """Whole-model roofline = the op-level model queried in DatabaseMode.SOL.

    Every op\'s analytic max(compute, mem) already encodes the real physics
    (MoE activated experts, DSA index_topk saturation, per-op quantization),
    so the FPM sol inherits it instead of hand-rolling an approximation. SOL
    queries read only the system spec — no perf data files are touched."""
    from aiconfigurator_core.sdk.perf_database import get_database_view

    view = get_database_view(
        database.system,
        database.backend,
        database.version,
        systems_paths=[database.systems_root],
        database_mode="SOL",
        allow_missing_data=True,
    )
    if view is None:
        raise PerfDataNotAvailableError(
            f"cannot build a SOL database view for {database.system}/{database.backend}/{database.version}"
        )

    @functools.lru_cache(maxsize=8192)
    def _sol(*coords) -> float:
        total = 0.0
        if phase == "prefill":
            batch, total_prefill, total_kv = coords
            s = max(total_prefill / batch, 1.0)
            prefix = total_kv / batch
            for op in sol_ops:
                x = batch if "logits_gemm" in op._name else total_prefill
                total += float(op.query(view, x=x, batch_size=batch, beam_width=1, s=s, prefix=prefix))
        else:
            batch, total_kv = coords
            s = max(total_kv / batch, 1.0)
            for op in sol_ops:
                total += float(op.query(view, x=batch, batch_size=batch, beam_width=1, s=s))
        return total

    return _sol


class FPMForwardOp(Operation):
    """One whole-model forward pass for a single phase (prefill or decode)."""

    _data_cache: ClassVar[dict] = {}

    def __init__(
        self,
        phase: str,
        model_config,
        model_path: str,
        sol_fn: Callable[..., float] | None = None,
        weight_bytes: float = 0.0,
        sol_ops: list | None = None,
    ) -> None:
        """``sol_fn`` injects an explicit roofline (tests/experiments).
        ``sol_ops`` — the model's ORIGINAL op-level list for this phase —
        derives the roofline from the op-level model itself queried in
        DatabaseMode.SOL: per-op analytic max(compute, mem) with the real
        physics (MoE activation, DSA index_topk saturation, per-op quant).
        Exactly one of the two must be provided."""
        if phase not in _PHASES:
            raise ValueError(f"unknown FPM phase: {phase!r}")
        if (sol_fn is None) == (sol_ops is None):
            raise ValueError("provide exactly one of sol_fn or sol_ops")
        super().__init__(f"fpm_forward_{phase}", 1.0)
        self._phase = phase
        self._model_path = str(model_path)
        self._weight_bytes = float(weight_bytes)
        self._match_identity = (
            _norm_identity(model_config.gemm_quant_mode),
            _norm_identity(model_config.moe_quant_mode),
            _norm_identity(model_config.fmha_quant_mode),
            _norm_identity(model_config.comm_quant_mode),
            _norm_identity(model_config.kvcache_quant_mode),
            _norm_identity(model_config.tp_size),
            _norm_identity(model_config.pp_size),
            _norm_identity(model_config.attention_dp_size),
            _norm_identity(model_config.moe_tp_size if model_config.moe_tp_size is not None else 1),
            _norm_identity(model_config.moe_ep_size if model_config.moe_ep_size is not None else 1),
            _norm_identity(model_config.cp_size),
            _norm_backend_request(getattr(model_config, "moe_backend", None)),
            # ModelConfig spells the engine default out ("flashinfer"); the
            # collector records engine-decided knobs as "auto".
            _norm_backend_request(getattr(model_config, "attention_backend", None), engine_default="flashinfer"),
            _norm_identity(bool(getattr(model_config, "enable_wideep", False))),
            _norm_identity(bool(getattr(model_config, "enable_eplb", False))),
        )
        self._sol_ops = list(sol_ops) if sol_ops is not None else None
        self._interp_configs: dict[tuple, OpInterpConfig] = {}
        if sol_fn is not None:
            key = ("static",)
            self._interp_configs[key] = (
                fpm_prefill_config(sol_fn) if self._phase == "prefill" else fpm_decode_config(sol_fn)
            )

    def _interp_config(self, database: PerfDatabase) -> OpInterpConfig:
        if ("static",) in self._interp_configs:
            return self._interp_configs[("static",)]
        key = (database.systems_root, database.system, database.backend, database.version)
        config = self._interp_configs.get(key)
        if config is None:
            sol = _oplevel_sol_fn(self._sol_ops, self._phase, database)
            config = fpm_prefill_config(sol) if self._phase == "prefill" else fpm_decode_config(sol)
            self._interp_configs[key] = config
        return config

    # ------------------------------------------------------------------
    # Data ownership
    # ------------------------------------------------------------------

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return (database.systems_root, database.system, database.backend, database.version)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent: loads the fpm_forward pair into the class cache and
        binds ``database._fpm_forward_data``. No shared-layer inheritance —
        FPM whole-model data is valid only for its exact backend/version."""
        from aiconfigurator_core.sdk.perf_database import LoadedOpData, PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            system_data_root = os.path.join(database.systems_root, database.system_spec["data_dir"])
            primary_path = os.path.join(
                system_data_root, database.backend, database.version, PerfDataFilename.fpm_forward.value
            )
            cls._data_cache[key] = LoadedOpData(
                load_fpm_forward_data(primary_path, database.version, database.system, database.backend),
                PerfDataFilename.fpm_forward,
                primary_path,
            )
            cls._record_load()

        if "_fpm_forward_data" not in database.__dict__:
            database._fpm_forward_data = cls._data_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._data_cache.clear()
        perf_interp.clear_caches()

    # ------------------------------------------------------------------
    # Cell selection (decision D1 resolved: exact model_path only — the
    # match identity carries no architecture fingerprint, so borrowing the
    # sole collected path could silently answer for a different model)
    # ------------------------------------------------------------------

    def _select_cell(self, cells: dict) -> dict:
        matches = [
            cell
            for cell in cells.values()
            if cell["match_identity"] == self._match_identity and cell["model_path"] == self._model_path
        ]
        if not matches:
            available = sorted({(cell["model_path"], *cell["match_identity"]) for cell in cells.values()})
            raise PerfDataNotAvailableError(
                f"No FPM cell matches model_path={self._model_path!r} with identity "
                f"{dict(zip(_CELL_MATCH_COLUMNS, self._match_identity, strict=True))}. FPM never "
                "substitutes another model's curves; collect data under this exact model path or "
                f"query with the collected path. Collected cells (model_path, identity): "
                f"{available[:8]}"
            )
        if len(matches) > 1:
            # Unreachable through the loader (cells are keyed by exactly these
            # fields) but kept as a hard guard against future key widening.
            raise PerfDataNotAvailableError(
                f"Ambiguous FPM cell selection for model_path={self._model_path!r}: "
                f"{len(matches)} cells share this identity and backend policy."
            )
        return matches[0]

    def _validate_deployment_identity(self, database: PerfDatabase) -> None:
        """Reject FPM identities the standard deployment bridge cannot emit.

        The schema-v6 collector can label vLLM measurements taken with pinned
        backend knobs, but the common Task -> generator path does not yet carry
        those knobs into the generated vLLM command. Allowing a direct SDK/YAML
        request to select such a cell would therefore model a different runtime
        than AIC deploys. Keep the rows producer-valid, but fail the consumer
        closed until structured generator fields land.
        """
        if database.backend != "vllm":
            return

        identity = dict(zip(_CELL_MATCH_COLUMNS, self._match_identity, strict=True))
        unsupported = []
        if identity["moe_backend"] != "auto":
            unsupported.append(f"moe_backend={identity['moe_backend']!r}")
        if identity["attention_backend"] != "auto":
            unsupported.append(f"attention_backend={identity['attention_backend']!r}")
        if identity["enable_eplb"] != "False":
            unsupported.append(f"enable_eplb={identity['enable_eplb']}")

        if unsupported:
            raise PerfDataNotAvailableError(
                "FPM cannot select this vLLM deployment identity because AIC's standard "
                "Task-to-generator path cannot emit the corresponding pinned backend/EPLB "
                f"settings yet: {', '.join(unsupported)}. Use automatic backend selection "
                "with EPLB disabled, or use forward_model='op_level', until those settings "
                "have structured end-to-end generator support."
            )

    # ------------------------------------------------------------------
    # Op contract
    # ------------------------------------------------------------------

    def _load_cell(self, database: PerfDatabase) -> dict:
        self._validate_deployment_identity(database)
        self.load_data(database)
        wrapper = database._fpm_forward_data
        wrapper.raise_if_not_loaded()
        return self._select_cell(wrapper["cells"])

    def _resolve(self, cell: dict, coords: tuple, database: PerfDatabase) -> PerformanceResult:
        interp_config = self._interp_config(database)
        table = cell["tables"][self._phase]
        domain = cell["domains"].get(self._phase)
        if self._phase == "prefill":
            clamp_max = cell.get("prefill_batch_clamp_max")
            if clamp_max is not None and coords[0] > clamp_max:
                # Data-certified batch clamp: answer at the collected batch
                # ceiling with the TRUE totals. Same totals = same GEMM/MoE
                # work and the same side of the CUDA-graph capture cliff (the
                # regime coordinate is the token total, untouched here);
                # fewer, longer segments do slightly MORE attention work, so
                # the answer is a bounded upper bound. total_kv stays the
                # real per-step read and is gated honestly below.
                coords = (clamp_max, *coords[1:])
        if not table or domain is None:
            raise PerfDataNotAvailableError(
                f"FPM cell {cell['cell_ids']} has no {self._phase} rows (model_path={cell['model_path']!r})."
            )
        for axis_index, (axis_name, value) in enumerate(zip(interp_config.axes, coords, strict=True)):
            low, high = domain[axis_index]
            if not low <= value <= high:
                raise PerfDataNotAvailableError(
                    f"FPM {self._phase} query {axis_name}={value} is outside the collected domain "
                    f"[{low}, {high}] for model_path={cell['model_path']!r}. "
                    "FPM never extrapolates; collect a wider sweep or use forward_model='op_level'."
                )

        if self._phase == "decode" and cell.get("decode_regime_boundary") is not None:
            # Route AFTER the full-domain gate: each side interpolates among
            # its own regime; the boundary batch itself is graph-side.
            side = "decode_graph" if coords[0] <= cell["decode_regime_boundary"] else "decode_eager"
            table = cell["tables"][side]
        result = perf_interp.query(interp_config, table, *coords)
        latency = perf_interp.get_value(result, "latency")
        if not math.isfinite(latency) or latency <= 0:
            raise PerfDataNotAvailableError(
                f"FPM {self._phase} interpolation produced an invalid latency ({latency}) at {coords}."
            )
        # Latency-only dataset: energy follows the Rust engine-step zero-energy
        # convention rather than fabricating a power figure.
        return PerformanceResult(latency * self._scale_factor, energy=0.0, source="silicon")

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        batch_size = int(kwargs["batch_size"])
        s = int(kwargs["s"])
        if batch_size < 1 or s < 1:
            raise ValueError(f"invalid FPM query: batch_size={batch_size}, s={s}")
        beam_width = int(kwargs.get("beam_width") or 1)
        if beam_width != 1:
            raise PerfDataNotAvailableError(
                f"forward_model='fpm' has no beam-search data (beam_width={beam_width}); use forward_model='op_level'."
            )

        cell = self._load_cell(database)
        if self._phase == "prefill":
            prefix = int(kwargs.get("prefix") or 0)
            coords = (batch_size, batch_size * s, batch_size * prefix)
        else:
            # One new token per request; ``s`` is the per-request KV length at
            # this decode step, so the iteration reads batch*s KV tokens.
            coords = (batch_size, batch_size * s)
        return self._resolve(cell, coords, database)

    def query_totals(
        self,
        database: PerfDatabase,
        *,
        batch_size: int,
        total_prefill_tokens: int = 0,
        total_kv_read_tokens: int,
    ) -> PerformanceResult:
        """Query by raw iteration-total coordinates.

        The mixed-step composition prices a scheduled iteration whose totals
        (prefill chunk + decode tokens) are generally not expressible as the
        per-request ``(batch, s, prefix)`` shape :meth:`query` converts from;
        this entry addresses the collected ``(batch_size,
        total_prefill_tokens, total_kv_read_tokens)`` coordinates directly
        (decode phase: ``(batch_size, total_kv_read_tokens)``). Same domain
        gate, interpolation, and no-extrapolation contract as :meth:`query`;
        mirrors the Rust op's ``query_totals``.
        """
        batch_size = int(batch_size)
        total_prefill_tokens = int(total_prefill_tokens)
        total_kv_read_tokens = int(total_kv_read_tokens)
        if batch_size < 1 or total_kv_read_tokens < 0:
            raise ValueError(
                f"invalid FPM totals query: batch_size={batch_size}, total_kv_read_tokens={total_kv_read_tokens}"
            )
        cell = self._load_cell(database)
        if self._phase == "prefill":
            if total_prefill_tokens < 1:
                raise ValueError(f"prefill query_totals needs total_prefill_tokens >= 1, got {total_prefill_tokens}")
            coords = (batch_size, total_prefill_tokens, total_kv_read_tokens)
        else:
            if total_prefill_tokens:
                raise ValueError(f"decode query_totals takes no prefill tokens, got {total_prefill_tokens}")
            coords = (batch_size, total_kv_read_tokens)
        return self._resolve(cell, coords, database)

    def query_pass_baseline(self, database: PerfDatabase, *, batch_size: int) -> PerformanceResult:
        """Decode-pass baseline at the smallest collectable KV for this batch.

        A pure decode step is ``weights_read + fixed_overheads + gemm(B) +
        kv_attention(B, KV)``. Everything except the KV term is paid once per
        forward pass — shared with the prefill work in a mixed step. Sampling
        the decode curve at the KV-axis floor (``max(B, domain_min)``: one KV
        token per request is the physical minimum) isolates that shared part,
        so ``query(B, KV) - query_pass_baseline(B)`` is the decode work's true
        marginal cost when it rides an existing pass.
        """
        if self._phase != "decode":
            raise ValueError(f"query_pass_baseline is decode-only, called on phase {self._phase!r}")
        batch_size = int(batch_size)
        if batch_size < 1:
            raise ValueError(f"invalid FPM baseline query: batch_size={batch_size}")
        cell = self._load_cell(database)
        domain = cell["domains"].get("decode")
        if domain is None:
            raise PerfDataNotAvailableError(
                f"FPM cell {cell['cell_ids']} has no decode rows (model_path={cell['model_path']!r})."
            )
        kv_floor = max(batch_size, domain[1][0])
        return self._resolve(cell, (batch_size, kv_floor), database)

    def get_weights(self, **kwargs) -> float:
        """Per-rank weight bytes of the whole model (captured from the original
        op-level lists before the rewrite), so memory estimation that sums
        ``op.get_weights()`` over the phase list keeps working."""
        return self._weight_bytes
