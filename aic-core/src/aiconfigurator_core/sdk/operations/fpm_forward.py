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

import hashlib
import json
import logging
import math
import os
from enum import Enum
from typing import TYPE_CHECKING, Callable, ClassVar

from aiconfigurator_core.sdk.errors import PerfDataNotAvailableError
from aiconfigurator_core.sdk.operations.base import Operation
from aiconfigurator_core.sdk.perf_interp import OpInterpConfig, ScatteredSites
from aiconfigurator_core.sdk.perf_interp import engine as perf_interp
from aiconfigurator_core.sdk.performance_result import PerformanceResult

if TYPE_CHECKING:
    from aiconfigurator_core.sdk.perf_database import PerfDatabase

logger = logging.getLogger(__name__)

FPM_FORWARD_SCHEMA_NAME = "aic_fpm_forward_perf"
FPM_FORWARD_SCHEMA_VERSION = 5
FPM_FORWARD_COORDINATE_SYSTEM = "iteration_totals_balanced_v1"
FPM_FORWARD_PARTITION_POLICY = "balanced_v1"
_PHASES = ("prefill", "decode")
_SUPPORTED_BACKEND_AXIS = "baseline"

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
    "backend_axis",
    "backend_policy",
    "workload_kind",
    "batch_size",
    "total_prefill_tokens",
    "total_kv_read_tokens",
    "partition_policy",
)


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


def _validate_sidecar(metadata_path: str, parquet_path: str) -> dict:
    """The sidecar is the writer's commit record: an unmatched pair (e.g. after
    an interrupted writer) must be rejected, not silently served."""
    if not os.path.exists(metadata_path):
        raise ValueError(
            f"FPM database is missing its metadata sidecar: {metadata_path}. "
            "The parquet/metadata pair is atomic; refusing to load an unmatched parquet."
        )
    with open(metadata_path, encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise ValueError(f"FPM metadata sidecar must be a JSON object: {metadata_path}")
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
    actual_sha = _sha256_file(parquet_path)
    if metadata.get("parquet_sha256") != actual_sha:
        raise ValueError(
            f"FPM parquet digest mismatch: sidecar={metadata.get('parquet_sha256')!r} actual={actual_sha!r}. "
            f"The pair at {os.path.dirname(parquet_path)} is inconsistent (interrupted writer?)."
        )
    return metadata


def _validate_row(row: dict, index: int, expected_version: str) -> None:
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


def load_fpm_forward_data(primary_path: str, expected_version: str):
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
    metadata = _validate_sidecar(metadata_path, primary_path)
    rows = pq.read_table(primary_path).to_pylist()
    if metadata.get("row_count") != len(rows):
        raise ValueError(
            f"FPM row_count mismatch: sidecar={metadata.get('row_count')!r} actual={len(rows)}: {primary_path}"
        )
    if not rows:
        raise ValueError(f"FPM database contains no rows: {primary_path}")

    seen_keys: set[tuple] = set()
    cells: dict[tuple, dict] = {}
    for index, row in enumerate(rows):
        _validate_row(row, index, expected_version)
        row_key = tuple(_norm_identity(row.get(column)) for column in _ROW_KEY_COLUMNS)
        if row_key in seen_keys:
            raise ValueError(f"FPM database contains a duplicate physical row key: {row_key}")
        seen_keys.add(row_key)

        cell_key = (
            _norm_identity(row.get("model_path")),
            _norm_identity(row.get("backend_axis")),
            _norm_identity(row.get("backend_policy")),
            *(_norm_identity(row.get(column)) for column in _CELL_MATCH_COLUMNS),
        )
        cell = cells.setdefault(
            cell_key,
            {
                "model_path": str(row.get("model_path")),
                "backend_axis": _norm_identity(row.get("backend_axis")),
                "backend_policy": _norm_identity(row.get("backend_policy")),
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

    return {"cells": cells}


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


def build_fpm_sol_fns(model) -> tuple[Callable, Callable, float]:
    """Crude per-rank rooflines for perf_interp's ratio-only uses.

    perf_interp consumes SOL exclusively in ratios (one-sided bracket
    recovery, cross-site util transfer), so only the scaling trend along each
    axis matters — absolute units and constant factors cancel. V1 therefore
    models the leading physics only:

      prefill (compute-bound): dense FLOPs ~ total_prefill_tokens, plus
          attention pair work ~ total_prefill * (total_prefill + total_kv) / B
      decode (memory-bound):   weight bytes + KV bytes read ~ total_kv

    Returns ``(prefill_sol, decode_sol, weight_bytes)`` where ``weight_bytes``
    is the per-rank op-level weights sum (reused by FPMForwardOp.get_weights
    so memory estimation keeps working after the op-list rewrite).
    """
    cfg = model.config
    weight_bytes = float(sum(op.get_weights() for op in model.context_ops))
    mem_bytes_per_weight = float(getattr(cfg.gemm_quant_mode.value, "memory", 2.0)) if cfg.gemm_quant_mode else 2.0
    flops_per_token = max(2.0 * weight_bytes / max(mem_bytes_per_weight, 1e-9), 1.0)
    layers_per_rank = model._num_layers / max(cfg.pp_size, 1)
    heads_per_rank = max(model._num_heads // max(cfg.tp_size, 1), 1)
    attn_flops_per_pair = 4.0 * layers_per_rank * heads_per_rank * model._head_size
    kv_bytes_per_token = max(float(model.get_kvcache_bytes_per_sequence(1)), 1.0)

    def prefill_sol(batch: float, total_prefill: float, total_kv: float) -> float:
        return flops_per_token * total_prefill + attn_flops_per_pair * total_prefill * (
            total_prefill + total_kv
        ) / max(batch, 1.0)

    def decode_sol(batch: float, total_kv: float) -> float:
        return weight_bytes + kv_bytes_per_token * total_kv

    return prefill_sol, decode_sol, weight_bytes


class FPMForwardOp(Operation):
    """One whole-model forward pass for a single phase (prefill or decode)."""

    _data_cache: ClassVar[dict] = {}

    def __init__(
        self,
        phase: str,
        model_config,
        model_path: str,
        sol_fn: Callable[..., float],
        weight_bytes: float,
    ) -> None:
        if phase not in _PHASES:
            raise ValueError(f"unknown FPM phase: {phase!r}")
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
        )
        if self._phase == "prefill":
            self._interp_config = fpm_prefill_config(sol_fn)
        else:
            self._interp_config = fpm_decode_config(sol_fn)

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
                load_fpm_forward_data(primary_path, database.version),
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
    # Cell selection (open decision D1: exact model_path, else unique fallback)
    # ------------------------------------------------------------------

    def _select_cell(self, cells: dict) -> dict:
        matches = [
            cell
            for cell in cells.values()
            if cell["match_identity"] == self._match_identity and cell["backend_axis"] == _SUPPORTED_BACKEND_AXIS
        ]
        if not matches:
            available = sorted({(cell["model_path"], *cell["match_identity"]) for cell in cells.values()})
            raise PerfDataNotAvailableError(
                f"No FPM cell matches this model identity: "
                f"{dict(zip(_CELL_MATCH_COLUMNS, self._match_identity, strict=True))}. "
                f"Collected cell identities (model_path first): {available[:8]}"
            )
        exact = [cell for cell in matches if cell["model_path"] == self._model_path]
        if exact:
            matches = exact
        distinct_paths = sorted({cell["model_path"] for cell in matches})
        if len(distinct_paths) > 1:
            raise PerfDataNotAvailableError(
                f"Ambiguous FPM cell selection: model_path={self._model_path!r} matched none exactly and "
                f"multiple collected model paths share this identity: {distinct_paths}. "
                "Collect/promote data under the exact model path, or query with the matching path."
            )
        if len(matches) > 1:
            raise PerfDataNotAvailableError(
                f"Ambiguous FPM cell selection: multiple backend policies for model_path={distinct_paths[0]!r}: "
                f"{sorted(cell['backend_policy'] for cell in matches)}"
            )
        return matches[0]

    # ------------------------------------------------------------------
    # Op contract
    # ------------------------------------------------------------------

    def _load_cell(self, database: PerfDatabase) -> dict:
        self.load_data(database)
        wrapper = database._fpm_forward_data
        wrapper.raise_if_not_loaded()
        return self._select_cell(wrapper["cells"])

    def _resolve(self, cell: dict, coords: tuple) -> PerformanceResult:
        table = cell["tables"][self._phase]
        domain = cell["domains"].get(self._phase)
        if not table or domain is None:
            raise PerfDataNotAvailableError(
                f"FPM cell {cell['cell_ids']} has no {self._phase} rows (model_path={cell['model_path']!r})."
            )
        for axis_index, (axis_name, value) in enumerate(zip(self._interp_config.axes, coords, strict=True)):
            low, high = domain[axis_index]
            if not low <= value <= high:
                raise PerfDataNotAvailableError(
                    f"FPM {self._phase} query {axis_name}={value} is outside the collected domain "
                    f"[{low}, {high}] for model_path={cell['model_path']!r}. "
                    "FPM never extrapolates; collect a wider sweep or use forward_model='op_level'."
                )

        result = perf_interp.query(self._interp_config, table, *coords)
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
        return self._resolve(cell, coords)

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
        return self._resolve(cell, (batch_size, kv_floor))

    def get_weights(self, **kwargs) -> float:
        """Per-rank weight bytes of the whole model (captured from the original
        op-level lists before the rewrite), so memory estimation that sums
        ``op.get_weights()`` over the phase list keeps working."""
        return self._weight_bytes
