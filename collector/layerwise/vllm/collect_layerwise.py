#!/usr/bin/env python3
"""Resilient vLLM layerwise collector.

The public CLI is the scheduler.  It builds one work unit per mocked model
configuration, assigns work units to one-GPU slots, and launches this same file
in hidden ``worker`` mode under ``nsys profile``.  The scheduler never imports
vLLM; CUDA/vLLM failures stay inside worker subprocesses.

Workers append progress events before/after datapoints.  The scheduler parses
complete or partial nsys reports, writes successful CSV rows, and marks failed
datapoints terminal so they are not retried on resume.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import gc
import hashlib
import json
import os
import shlex
import statistics
import subprocess
import sys
import time
import traceback
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_THIS_DIR = Path(__file__).resolve().parent
_COMMON_DIR = _THIS_DIR.parent / "common"
sys.path.insert(0, str(_COMMON_DIR))

from parallel_config_patch import EXPERT_COUNT_KEYS, _load_original_config, patch_for_parallelism
from parse_nsys_step_sweep import parse_step_sweep
from random_prompt_tokens import (
    RandomPromptTokenConfig,
    load_random_prompt_token_config,
    make_prompt_token_ids,
    sample_prompt_token_ids,
)
from vllm_deployment import (
    VllmDeploymentConfig,
    build_engine_args,
    find_runtime_vllm_config,
    gpt_oss_runtime_defaults,
    has_cli_flag,
    make_metadata,
    summarize_vllm_config,
    write_metadata,
)

CTX_NEW_TOKENS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
CTX_PAST_KV = [
    0,
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
]
GEN_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
GEN_PAST_KV = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
VLLM_DEFAULT_BLOCK_SIZE = 16
PREFILL_DECODE_MAX_NUM_BATCHED_TOKENS = 16384
DEFAULT_EXTRA_VLLM_ARGS = (
    ("flag", "--skip-mm-profiling", ("--no-skip-mm-profiling",)),
    ("pair", "--limit-mm-per-prompt", '{"image":0,"video":0}'),
    ("pair", "--generation-config", "vllm"),
)

CSV_COLUMNS = [
    "framework",
    "framework_version",
    "system",
    "model",
    "attn_tp",
    "moe_tp",
    "ep",
    "num_slots",
    "gemm_quant",
    "moe_quant",
    "attn_quant",
    "kv_quant",
    "phase",
    "batch_size",
    "new_tokens",
    "past_kv",
    "latency_ms",
    "rms_latency_ms",
    "rms_kernel_count",
    "measurement_mode",
    "attribution_target",
    "includes_moe",
    "vllm_config_hash",
]

DEFAULT_ATTRIBUTION_ROLLUP = r"layers\.(\d+)\.(self_attn|mlp|input_layernorm|post_attention_layernorm)"
DEFAULT_PARITY_ROLLUP = r"^(CUDAGraphWrapper)$"

TERMINAL_EVENTS = {
    "success",
    "failed_oom",
    "failed_error",
    "failed_fatal_cuda",
    "failed_parse",
    "skipped_oom_dominated",
    "skipped_same_error",
}

FATAL_STREAK_LIMIT = 3


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_hash(payload: Any, *, n: int = 16) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha1(raw).hexdigest()[:n]


def _parse_ints(raw: str) -> list[int]:
    return [int(x) for x in raw.split(",") if x.strip()]


def _append_default_vllm_args(extra_vllm_args: list[str]) -> None:
    for kind, flag, value_or_aliases in DEFAULT_EXTRA_VLLM_ARGS:
        if kind == "flag":
            aliases = tuple(value_or_aliases)
            if not has_cli_flag(extra_vllm_args, flag, *aliases):
                extra_vllm_args.append(flag)
        elif not has_cli_flag(extra_vllm_args, flag):
            extra_vllm_args.extend([flag, value_or_aliases])


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


@dataclass(frozen=True)
class DataPoint:
    phase: str
    batch_size: int
    new_tokens: int
    past_kv: int

    @property
    def shape_key(self) -> str:
        return (
            f"{self.phase}:bs{self.batch_size}:"
            f"new{self.new_tokens}:past{self.past_kv}"
        )

    def datapoint_id(self, work_unit_id: str) -> str:
        return f"{work_unit_id}:{self.shape_key}"

    def parse_key(self) -> tuple[int, int, int]:
        if self.phase == "ctx":
            return self.new_tokens, self.batch_size, self.past_kv
        return self.past_kv + 1, self.batch_size, self.past_kv


@dataclass(frozen=True)
class WorkUnit:
    work_unit_id: str
    model_dir: str
    row_base: dict[str, Any]
    target_layers: list[int]
    datapoints: list[DataPoint]
    moe_noop: bool = False
    includes_moe: bool = False

    def manifest_rows(self) -> list[dict[str, Any]]:
        rows = []
        for dp in self.datapoints:
            rows.append({
                "work_unit_id": self.work_unit_id,
                "datapoint_id": dp.datapoint_id(self.work_unit_id),
                **self.row_base,
                "moe_noop": self.moe_noop,
                "includes_moe": self.includes_moe,
                "phase": dp.phase,
                "batch_size": dp.batch_size,
                "new_tokens": dp.new_tokens,
                "past_kv": dp.past_kv,
            })
        return rows


@dataclass
class Attempt:
    work_unit: WorkUnit
    gpu: str
    attempt_id: int
    spec_path: Path
    report_base: Path
    stdout_path: Path
    stderr_path: Path
    process: subprocess.Popen
    stdout_handle: Any
    stderr_handle: Any
    pending_ids: set[str]


class StatusIndex:
    """Reconstructed view of the append-only status log."""

    def __init__(self, events: list[dict[str, Any]]):
        self.events = events
        self.terminal: dict[str, dict[str, Any]] = {}
        self.started: dict[str, list[dict[str, Any]]] = {}
        self.completed: set[str] = set()
        for event in events:
            dpid = event.get("datapoint_id")
            if not dpid:
                continue
            name = event.get("event")
            if name == "started":
                self.started.setdefault(dpid, []).append(event)
            elif name == "completed_execution":
                self.completed.add(dpid)
            elif name in TERMINAL_EVENTS:
                self.terminal[dpid] = event

    def is_terminal(self, datapoint_id: str) -> bool:
        return datapoint_id in self.terminal

    def terminal_ids(self) -> set[str]:
        return set(self.terminal)

    def active_started(self, work_unit_id: str, pending_ids: set[str]) -> str | None:
        """Return newest started datapoint without a terminal event.

        This is the crash contract between worker and scheduler.  If the
        worker dies, the newest non-terminal started datapoint is considered
        the one that caused the crash and is never retried.
        """
        for event in reversed(self.events):
            if event.get("event") != "started":
                continue
            if event.get("work_unit_id") != work_unit_id:
                continue
            dpid = event.get("datapoint_id")
            if dpid in pending_ids and dpid not in self.terminal:
                return dpid
        return None


class StatusStore:
    """Append-only manifest/status files shared by scheduler and workers."""

    def __init__(self, work_dir: Path):
        self.work_dir = work_dir
        self.manifest_path = work_dir / "manifest.jsonl"
        self.status_path = work_dir / "status.jsonl"
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def append_jsonl(self, path: Path, row: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            f.flush()
            os.fsync(f.fileno())
            fcntl.flock(f, fcntl.LOCK_UN)

    def append_event(self, event: str, *, work_unit_id: str, datapoint_id: str | None = None, **extra: Any) -> None:
        row = {
            "event": event,
            "work_unit_id": work_unit_id,
            "datapoint_id": datapoint_id,
            "ts": _utc_now(),
            **extra,
        }
        self.append_jsonl(self.status_path, {k: v for k, v in row.items() if v is not None})

    def load_events(self) -> list[dict[str, Any]]:
        if not self.status_path.exists():
            return []
        events = []
        with self.status_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        return events

    def index(self) -> StatusIndex:
        return StatusIndex(self.load_events())

    def existing_manifest_ids(self) -> set[str]:
        if not self.manifest_path.exists():
            return set()
        ids = set()
        with self.manifest_path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                ids.add(json.loads(line)["datapoint_id"])
        return ids

    def write_missing_manifest(self, work_units: Iterable[WorkUnit]) -> None:
        seen = self.existing_manifest_ids()
        for unit in work_units:
            for row in unit.manifest_rows():
                if row["datapoint_id"] in seen:
                    continue
                self.append_jsonl(self.manifest_path, row)
                seen.add(row["datapoint_id"])


def _get_system_name() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i", "0"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _get_vllm_version() -> str:
    """Query vLLM in a child process so the scheduler never imports it."""
    code = "import vllm; print(getattr(vllm, '__version__', 'unknown'))"
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip() or "unknown"
    except Exception:
        pass
    return "unknown"


def _detect_gpus(gpus_arg: str | None) -> list[str]:
    if gpus_arg:
        return [x.strip() for x in gpus_arg.split(",") if x.strip()]

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and visible not in ("-1", "NoDevFiles"):
        return [x.strip() for x in visible.split(",") if x.strip()]

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        gpus = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if gpus:
            return gpus
    except Exception:
        pass
    return ["0"]


def _detect_layer_schedule(
    config: dict[str, Any],
    include_moe_layer: bool,
    target_layer_count: int = 1,
    target_layers: list[int] | None = None,
    target_layer_config_depth: int | None = None,
) -> tuple[list[dict[str, Any]], int, dict[str, Any] | None]:
    config = _decoder_config_view(config)
    max_config_layers = int(config.get("num_hidden_layers") or 0)
    if target_layers is not None:
        if not target_layers:
            raise ValueError("target_layers must not be empty")
        if any(i < 0 for i in target_layers):
            raise ValueError(f"target_layers must be non-negative, got {target_layers}")
        if max_config_layers and max(target_layers) >= max_config_layers:
            raise ValueError(
                f"target_layers {target_layers} exceed config num_hidden_layers="
                f"{max_config_layers}"
            )
        is_moe = _is_moe_config(config)
        if is_moe and not _is_all_moe_config(config):
            raise ValueError(
                "explicit target_layers for hybrid MoE configs is not supported; "
                "use --include-moe-layer for the dense+MoE schedule"
            )
        sorted_layers = sorted(set(target_layers))
        num_hidden_layers = max(sorted_layers) + 1
        if target_layer_config_depth is not None:
            if target_layer_config_depth < num_hidden_layers:
                raise ValueError(
                    f"target_layer_config_depth={target_layer_config_depth} is smaller "
                    f"than required depth {num_hidden_layers}"
                )
            if max_config_layers and target_layer_config_depth > max_config_layers:
                raise ValueError(
                    f"target_layer_config_depth={target_layer_config_depth} exceeds "
                    f"config num_hidden_layers={max_config_layers}"
                )
            num_hidden_layers = target_layer_config_depth
        return [
            {"layer_index": i, "layer_type": "moe" if is_moe else "dense"}
            for i in sorted_layers
        ], num_hidden_layers, _layer_types_override(config, num_hidden_layers)

    if target_layer_count < 1:
        raise ValueError(f"target_layer_count must be >= 1, got {target_layer_count}")
    if not _is_moe_config(config):
        num_hidden_layers = target_layer_count
        if target_layer_config_depth is not None:
            if target_layer_config_depth < target_layer_count:
                raise ValueError(
                    f"target_layer_config_depth={target_layer_config_depth} is smaller "
                    f"than target_layer_count={target_layer_count}"
                )
            if max_config_layers and target_layer_config_depth > max_config_layers:
                raise ValueError(
                    f"target_layer_config_depth={target_layer_config_depth} exceeds "
                    f"config num_hidden_layers={max_config_layers}"
                )
            num_hidden_layers = target_layer_config_depth
        return [
            {"layer_index": i, "layer_type": "dense"}
            for i in range(target_layer_count)
        ], num_hidden_layers, _layer_types_override(config, num_hidden_layers)

    if _is_all_moe_config(config):
        num_hidden_layers = target_layer_count
        if target_layer_config_depth is not None:
            if target_layer_config_depth < target_layer_count:
                raise ValueError(
                    f"target_layer_config_depth={target_layer_config_depth} is smaller "
                    f"than target_layer_count={target_layer_count}"
                )
            if max_config_layers and target_layer_config_depth > max_config_layers:
                raise ValueError(
                    f"target_layer_config_depth={target_layer_config_depth} exceeds "
                    f"config num_hidden_layers={max_config_layers}"
                )
            num_hidden_layers = target_layer_config_depth
        return [
            {"layer_index": i, "layer_type": "moe"}
            for i in range(target_layer_count)
        ], num_hidden_layers, _layer_types_override(config, num_hidden_layers)

    # Hybrid vLLM MoE is opt-in because dummy-weight routing still underestimates MoE.
    if target_layer_count != 1:
        raise ValueError(
            "target_layer_count > 1 is currently only supported for dense models"
        )
    layer_schedule = [{"layer_index": 0, "layer_type": "dense"}]
    num_hidden_layers = 1
    if include_moe_layer:
        layer_schedule.append({"layer_index": 1, "layer_type": "moe"})
        num_hidden_layers = 2

    overrides = {"first_k_dense_replace": 1}
    if "decoder_sparse_step" in config:
        overrides["decoder_sparse_step"] = 1
        overrides["mlp_only_layers"] = []
    layer_type_overrides = _layer_types_override(config, num_hidden_layers)
    if layer_type_overrides:
        overrides.update(layer_type_overrides)
    return layer_schedule, num_hidden_layers, overrides


def _layer_types_override(
    config: dict[str, Any],
    num_hidden_layers: int,
) -> dict[str, Any] | None:
    layer_types = config.get("layer_types")
    if not isinstance(layer_types, list):
        return None
    if len(layer_types) < num_hidden_layers:
        raise ValueError(
            f"layer_types has length {len(layer_types)} but num_hidden_layers="
            f"{num_hidden_layers}"
        )
    return {"layer_types": list(layer_types[:num_hidden_layers])}


def _decoder_config_view(config: dict[str, Any]) -> dict[str, Any]:
    text_config = config.get("text_config")
    if isinstance(text_config, dict) and "num_attention_heads" in text_config:
        return text_config
    return config


def _is_moe_config(config: dict[str, Any]) -> bool:
    config = _decoder_config_view(config)
    return any((config.get(k, 0) or 0) > 0 for k in EXPERT_COUNT_KEYS)


def _is_all_moe_config(config: dict[str, Any]) -> bool:
    config = _decoder_config_view(config)
    model_type = str(config.get("model_type") or "").lower()
    architectures = [str(x) for x in config.get("architectures") or []]
    if model_type == "gpt_oss" or "GptOssForCausalLM" in architectures:
        return True
    # Hybrid MoE configs usually carry replacement/sparse-step controls.  If a
    # config exposes experts but none of those controls, assume every decoder
    # block owns a routed MLP.
    return _is_moe_config(config) and not any(
        key in config
        for key in ("first_k_dense_replace", "decoder_sparse_step", "mlp_only_layers")
    )


def _work_unit_id(
    row_base: dict[str, Any],
    target_layers: list[int],
    num_hidden_layers: int,
    moe_noop: bool = False,
) -> str:
    payload = {
        **row_base,
        "target_layers": target_layers,
        "num_hidden_layers": num_hidden_layers,
        "moe_noop": moe_noop,
    }
    return "wu_" + _stable_hash(payload)


def _filter_rows_to_target_layers(
    rows: list[dict[str, Any]],
    target_layers: Iterable[int],
) -> list[dict[str, Any]]:
    targets = {int(x) for x in target_layers}
    if not targets:
        return rows
    out = []
    for row in rows:
        parts = row.get("rollup_parts") or ()
        if parts and int(parts[0]) in targets:
            out.append(row)
    return out


def _build_datapoints(
    *,
    phases: str,
    ctx_new_tokens: list[int],
    ctx_past_kv: list[int],
    gen_batch_sizes: list[int],
    gen_past_kv: list[int],
) -> list[DataPoint]:
    datapoints: list[DataPoint] = []
    if phases in ("ctx", "both"):
        for past_kv in ctx_past_kv:
            for new_tokens in ctx_new_tokens:
                datapoints.append(DataPoint("ctx", 1, new_tokens, past_kv))
    if phases in ("gen", "both"):
        for batch_size in gen_batch_sizes:
            for past_kv in gen_past_kv:
                datapoints.append(DataPoint("gen", batch_size, 1, past_kv))
    return datapoints


def _max_num_batched_tokens_for_datapoints(
    datapoints: list[DataPoint],
    min_max_num_batched_tokens: int = 1,
    *,
    gen_driver: str = "prefix_cache",
) -> int:
    ctx_points = [dp for dp in datapoints if dp.phase == "ctx"]
    gen_points = [dp for dp in datapoints if dp.phase == "gen"]
    ctx_max_new = max((dp.new_tokens for dp in ctx_points), default=0)
    gen_batch_sizes = sorted({dp.batch_size for dp in gen_points})
    min_budget = 1
    if gen_driver == "prefix_cache" and gen_points:
        # vLLM's prefix-cache path can use Mamba cache align mode, which
        # requires the token budget to be at least one KV block.
        min_budget = VLLM_DEFAULT_BLOCK_SIZE
    return max(
        min_budget,
        min_max_num_batched_tokens,
        ctx_max_new,
        max(gen_batch_sizes, default=0),
    )


def _validate_ctx_past_kv(
    datapoints: list[DataPoint],
    max_num_batched_tokens: int,
    *,
    ctx_driver: str,
) -> None:
    if ctx_driver not in {"chunked", "prefix_cache"}:
        raise ValueError(f"unsupported ctx driver: {ctx_driver}")
    if ctx_driver == "prefix_cache":
        return
    for dp in datapoints:
        if dp.phase != "ctx" or dp.past_kv == 0:
            continue
        if dp.past_kv % max_num_batched_tokens != 0:
            raise ValueError(
                "ctx past_kv measurements must start on a chunk boundary: "
                f"past_kv={dp.past_kv}, max_num_batched_tokens={max_num_batched_tokens}"
            )


def _model_max_position_embeddings(config: dict[str, Any]) -> int | None:
    config = _decoder_config_view(config)
    for key in ("max_position_embeddings", "model_max_length", "seq_length", "n_positions"):
        raw = config.get(key)
        if isinstance(raw, bool) or raw is None:
            continue
        try:
            value = int(raw)
        except (TypeError, ValueError):
            continue
        # Some tokenizer/model configs use a huge sentinel for "unbounded".
        if 0 < value < 1_000_000_000:
            return value
    return None


def _filter_datapoints_for_model_max_len(
    datapoints: list[DataPoint],
    max_model_len: int | None,
) -> tuple[list[DataPoint], int]:
    if max_model_len is None:
        return datapoints, 0

    filtered: list[DataPoint] = []
    skipped = 0
    for dp in datapoints:
        if dp.phase == "ctx":
            # The ctx driver uses max_tokens=1 to force vLLM to execute the
            # measured prefill, so max_model_len must fit the prompt plus that
            # generated token.
            required_len = dp.past_kv + dp.new_tokens + 1
        else:
            required_len = dp.past_kv + 2
        if required_len > max_model_len:
            skipped += 1
            continue
        filtered.append(dp)
    return filtered, skipped


def build_work_units(args: argparse.Namespace) -> list[WorkUnit]:
    ctx_new_tokens = _parse_ints(args.ctx_new_tokens)
    ctx_past_kv = _parse_ints(args.ctx_past_kv)
    gen_batch_sizes = _parse_ints(args.gen_batch_sizes)
    gen_past_kv = _parse_ints(args.gen_past_kv)
    tp_sizes = _parse_ints(args.tp_sizes)

    orig_config = _load_original_config(args.model)
    is_moe = _is_moe_config(orig_config)
    explicit_target_layers = (
        _parse_ints(args.target_layers) if getattr(args, "target_layers", None) else None
    )
    layer_schedule, num_hidden_layers, extra_overrides = _detect_layer_schedule(
        orig_config, args.include_moe_layer, args.target_layer_count,
        explicit_target_layers, args.target_layer_config_depth,
    )
    target_layers = [int(x["layer_index"]) for x in layer_schedule]
    moe_noop = bool(getattr(args, "moe_noop", False) and is_moe)
    includes_moe = (not moe_noop) and any(
        str(layer.get("layer_type", "")).lower() == "moe"
        for layer in layer_schedule
    )

    work_dir = Path(args.work_dir).resolve()
    config_cache_dir = None if args.no_config_cache else (args.config_cache_dir or str(work_dir / "config_cache"))

    system = args.system or _get_system_name()
    version = args.framework_version or _get_vllm_version()
    datapoints = _build_datapoints(
        phases=args.phases,
        ctx_new_tokens=ctx_new_tokens,
        ctx_past_kv=ctx_past_kv,
        gen_batch_sizes=gen_batch_sizes,
        gen_past_kv=gen_past_kv,
    )
    if not getattr(args, "no_filter_model_max_len", False):
        model_max_len = _model_max_position_embeddings(orig_config)
        datapoints, skipped = _filter_datapoints_for_model_max_len(datapoints, model_max_len)
        if skipped:
            print(
                f"[skip] {skipped} datapoints require more than "
                f"model_max_len={model_max_len} tokens"
            )
        if not datapoints:
            raise ValueError("all datapoints exceed the model's configured max length")
    _validate_ctx_past_kv(
        datapoints,
        _max_num_batched_tokens_for_datapoints(
            datapoints,
            args.min_max_num_batched_tokens,
            gen_driver=getattr(args, "gen_driver", "prefix_cache"),
        ),
        ctx_driver=getattr(args, "ctx_driver", "prefix_cache"),
    )

    work_units: list[WorkUnit] = []
    for tp in tp_sizes:
        if is_moe and tp % args.moe_tp != 0:
            print(f"[skip] tp={tp} not divisible by moe_tp={args.moe_tp}")
            continue
        attn_tp = tp
        moe_tp = args.moe_tp if is_moe else 1
        ep = (tp // moe_tp) if is_moe else 1
        num_slots = args.num_slots if is_moe else None
        model_dir = patch_for_parallelism(
            args.model,
            attn_tp=attn_tp,
            moe_tp=moe_tp,
            ep=ep,
            num_slots=num_slots,
            num_hidden_layers=num_hidden_layers,
            extra_overrides=extra_overrides,
            model_type_rewrites={"glm_moe_dsa": "deepseek_v3"},
            cache_dir=config_cache_dir,
            original_config=orig_config,
        )
        row_base = {
            "framework": "vLLM",
            "framework_version": version,
            "system": system,
            "model": args.model,
            "attn_tp": attn_tp,
            "moe_tp": moe_tp,
            "ep": ep,
            "num_slots": num_slots or "",
            "gemm_quant": args.gemm_quant,
            "moe_quant": args.moe_quant,
            "attn_quant": args.attn_quant,
            "kv_quant": args.kv_quant,
        }
        work_units.append(WorkUnit(
            work_unit_id=_work_unit_id(
                row_base,
                target_layers,
                num_hidden_layers,
                moe_noop,
            ),
            model_dir=model_dir,
            row_base=row_base,
            target_layers=target_layers,
            datapoints=datapoints,
            moe_noop=moe_noop,
            includes_moe=includes_moe,
        ))
    return work_units


def _is_oom_text(text: str) -> bool:
    lowered = text.lower()
    return "out of memory" in lowered or "cuda oom" in lowered or "cublas_status_alloc_failed" in lowered


def _is_fatal_cuda_text(text: str) -> bool:
    lowered = text.lower()
    return "illegal memory access" in lowered or "device-side assert" in lowered or "cuda error" in lowered


def _attempt_signature(returncode: int, stderr_tail: str) -> str:
    if _is_oom_text(stderr_tail):
        return "oom"
    if _is_fatal_cuda_text(stderr_tail):
        return "fatal_cuda"
    return f"exit_{returncode}"


def oom_dominates(failed: DataPoint, candidate: DataPoint) -> bool:
    """Return whether a failed OOM point should prune a candidate.

    OOM pruning is phase-local.  A ctx OOM says larger ctx tokens/past are
    unsafe; a gen OOM says larger batch/past points are unsafe.  It never
    prunes the other phase.
    """
    if failed.phase != candidate.phase:
        return False
    if failed.phase == "ctx":
        same_or_larger = (
            candidate.new_tokens >= failed.new_tokens
            and candidate.past_kv >= failed.past_kv
        )
        strictly_larger = (
            candidate.new_tokens > failed.new_tokens
            or candidate.past_kv > failed.past_kv
        )
        return same_or_larger and strictly_larger
    same_or_larger = (
        candidate.batch_size >= failed.batch_size
        and candidate.past_kv >= failed.past_kv
    )
    strictly_larger = (
        candidate.batch_size > failed.batch_size
        or candidate.past_kv > failed.past_kv
    )
    return same_or_larger and strictly_larger


def _aggregate_step_rows(rows: list[dict[str, Any]]) -> dict[tuple[int, int, int, int], dict[str, Any]]:
    out: dict[tuple[int, int, int, int], dict[str, Any]] = {}
    for row in rows:
        key = (
            row["step"],
            row["batch_size"],
            row["past_kv"],
            int(row.get("measure_run", 0)),
        )
        agg = out.setdefault(key, {
            "gpu_us": 0.0,
            "rms_us": 0.0,
            "span_us": 0.0,
            "start_ns": None,
            "end_ns": None,
            "kernel_count": 0,
            "rms_kernel_count": 0,
        })
        agg["gpu_us"] += row["gpu_us"]
        agg["rms_us"] += row.get("rms_us", 0.0)
        if "start_ns" in row and "end_ns" in row:
            start_ns = int(row["start_ns"])
            end_ns = int(row["end_ns"])
            if agg["start_ns"] is None or start_ns < agg["start_ns"]:
                agg["start_ns"] = start_ns
            if agg["end_ns"] is None or end_ns > agg["end_ns"]:
                agg["end_ns"] = end_ns
            agg["span_us"] = (agg["end_ns"] - agg["start_ns"]) / 1000.0
        else:
            # Backward-compatible path for older parser rows.
            agg["span_us"] += row.get("span_us", row["gpu_us"])
        agg["kernel_count"] += row["kernel_count"]
        agg["rms_kernel_count"] += row.get("rms_kernel_count", 0)
    return out


def _latency_us_from_agg(agg: dict[str, Any], latency_source: str) -> float:
    if latency_source == "span":
        return float(agg["span_us"])
    if latency_source == "gpu":
        return float(agg["gpu_us"])
    if latency_source == "gpu_capped":
        return min(float(agg["gpu_us"]), float(agg["span_us"]))
    raise ValueError(f"unsupported latency source: {latency_source}")


def _lookup_aggs(
    parsed: dict[tuple[int, int, int, int], dict[str, Any]],
    expected_key: tuple[int, int, int],
) -> list[dict[str, Any]]:
    exact = [
        value for key, value in sorted(parsed.items())
        if key[:3] == expected_key
    ]
    if exact:
        return exact

    step, _batch_size, past_kv = expected_key
    candidate_items = [
        (key, value) for key, value in sorted(parsed.items())
        if key[0] == step and key[2] == past_kv
    ]
    candidate_batches = {key[1] for key, _ in candidate_items}
    if len(candidate_batches) == 1:
        return [value for _, value in candidate_items]
    return []


def _reduce_agg_latency(
    aggs: list[dict[str, Any]],
    *,
    latency_source: str,
    aggregation: str,
) -> tuple[float, float, int, int, int]:
    if not aggs:
        raise ValueError("cannot reduce empty aggregate list")
    values = [_latency_us_from_agg(agg, latency_source) for agg in aggs]
    rms_values = [float(agg.get("rms_us", 0.0)) for agg in aggs]
    if aggregation == "median":
        latency_us = float(statistics.median(values))
        rms_us = float(statistics.median(rms_values))
    elif aggregation == "mean":
        latency_us = float(statistics.fmean(values))
        rms_us = float(statistics.fmean(rms_values))
    elif aggregation == "trimmed_mean":
        if len(values) < 3:
            latency_us = float(statistics.fmean(values))
            rms_us = float(statistics.fmean(rms_values))
        else:
            latency_us = float(statistics.fmean(sorted(values)[1:-1]))
            rms_us = float(statistics.fmean(sorted(rms_values)[1:-1]))
    elif aggregation == "min":
        latency_us = float(min(values))
        rms_us = float(min(rms_values))
    else:
        raise ValueError(f"unsupported repeat aggregation: {aggregation}")
    kernel_count = int(statistics.median([int(agg["kernel_count"]) for agg in aggs]))
    rms_kernel_count = int(statistics.median([int(agg.get("rms_kernel_count", 0)) for agg in aggs]))
    return latency_us, rms_us, kernel_count, rms_kernel_count, len(aggs)


def _write_csv_header_if_needed(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    with path.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_COLUMNS).writeheader()


def _append_success_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writerow({k: row.get(k, "") for k in CSV_COLUMNS})
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f, fcntl.LOCK_UN)


def _effective_rollup(args: argparse.Namespace) -> str:
    if args.rollup:
        return str(args.rollup)
    if args.measurement_mode == "deployment-parity":
        return DEFAULT_PARITY_ROLLUP
    return DEFAULT_ATTRIBUTION_ROLLUP


def _should_filter_target_layers(args: argparse.Namespace) -> bool:
    if args.no_filter_target_layers:
        return False
    return args.measurement_mode != "deployment-parity"


def _attribution_target(args: argparse.Namespace) -> str:
    if args.measurement_mode == "deployment-parity":
        return "cudagraph_wrapper"
    return "module_nvtx"


def _work_unit_includes_moe(work_unit: WorkUnit) -> bool:
    return bool(work_unit.includes_moe)


def _attempt_config_hash(attempt: Attempt, store: StatusStore) -> str:
    for event in reversed(store.index().events):
        if event.get("event") != "engine_metadata_written":
            continue
        if event.get("work_unit_id") != attempt.work_unit.work_unit_id:
            continue
        if event.get("attempt_id") != attempt.attempt_id:
            continue
        return str(event.get("vllm_config_hash") or "")
    return ""


class Scheduler:
    """One-GPU-slot scheduler for nsys-wrapped workers."""

    def __init__(self, args: argparse.Namespace, work_units: list[WorkUnit]):
        self.args = args
        self.work_units = work_units
        self.work_dir = Path(args.work_dir).resolve()
        self.store = StatusStore(self.work_dir)
        self.output_path = Path(args.output).resolve()
        self.gpus = _detect_gpus(args.gpus)
        if args.max_workers:
            self.gpus = self.gpus[: args.max_workers]
        if not self.gpus:
            raise RuntimeError("No GPU slots available")
        self.attempt_counter = 0
        self.fatal_streak: dict[tuple[str, str], int] = {}

    def run(self) -> None:
        _write_csv_header_if_needed(self.output_path)
        self.store.write_missing_manifest(self.work_units)

        queue = list(self.work_units)
        active: dict[str, Attempt] = {}
        print(f"[scheduler] GPU slots: {','.join(self.gpus)}")

        while queue or active:
            for gpu in self.gpus:
                if gpu in active or not queue:
                    continue
                unit = queue.pop(0)
                pending = self._pending_datapoints(unit)
                if not pending:
                    continue
                active[gpu] = self._launch_attempt(unit, gpu, pending)

            finished = []
            for gpu, attempt in active.items():
                rc = attempt.process.poll()
                if rc is not None:
                    finished.append((gpu, attempt, rc))

            for gpu, attempt, rc in finished:
                del active[gpu]
                still_pending = self._finish_attempt(attempt, rc)
                if still_pending:
                    queue.append(attempt.work_unit)

            if active:
                time.sleep(1.0)

        print(f"[scheduler] Done. Results written to {self.output_path}")
        print(f"[scheduler] Status written to {self.store.status_path}")

    def _pending_datapoints(self, unit: WorkUnit) -> list[DataPoint]:
        terminal = self.store.index().terminal_ids()
        return [dp for dp in unit.datapoints if dp.datapoint_id(unit.work_unit_id) not in terminal]

    def _launch_attempt(self, unit: WorkUnit, gpu: str, pending: list[DataPoint]) -> Attempt:
        self.attempt_counter += 1
        attempt_id = self.attempt_counter
        paths = {
            "spec": self.work_dir / "specs" / f"{unit.work_unit_id}_a{attempt_id}.json",
            "report": self.work_dir / "nsys" / f"{unit.work_unit_id}_a{attempt_id}",
            "stdout": self.work_dir / "logs" / f"{unit.work_unit_id}_a{attempt_id}.out",
            "stderr": self.work_dir / "logs" / f"{unit.work_unit_id}_a{attempt_id}.err",
            "metadata": self.work_dir / "metadata" / f"{unit.work_unit_id}_a{attempt_id}.json",
        }
        paths["report"].parent.mkdir(parents=True, exist_ok=True)
        spec = self._make_spec(unit, pending, attempt_id)
        spec["metadata_path"] = str(paths["metadata"])
        _json_dump(paths["spec"], spec)
        paths["stdout"].parent.mkdir(parents=True, exist_ok=True)
        stdout_handle = paths["stdout"].open("w")
        stderr_handle = paths["stderr"].open("w")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        cmd = self._worker_cmd(paths["spec"], paths["report"])
        print(
            f"[scheduler] launch gpu={gpu} attempt={attempt_id} "
            f"{unit.work_unit_id} pending={len(pending)}"
        )
        process = subprocess.Popen(
            cmd,
            cwd=str(_THIS_DIR),
            env=env,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
        )
        self.store.append_event(
            "attempt_started",
            work_unit_id=unit.work_unit_id,
            attempt_id=attempt_id,
            gpu=gpu,
            report_base=str(paths["report"]),
            spec=str(paths["spec"]),
        )
        return Attempt(
            work_unit=unit,
            gpu=gpu,
            attempt_id=attempt_id,
            spec_path=paths["spec"],
            report_base=paths["report"],
            stdout_path=paths["stdout"],
            stderr_path=paths["stderr"],
            process=process,
            stdout_handle=stdout_handle,
            stderr_handle=stderr_handle,
            pending_ids={dp.datapoint_id(unit.work_unit_id) for dp in pending},
        )

    def _make_spec(self, unit: WorkUnit, pending: list[DataPoint], attempt_id: int) -> dict[str, Any]:
        extra_vllm_args = []
        if unit.row_base["kv_quant"] == "fp8":
            extra_vllm_args.extend(["--kv-cache-dtype", "fp8"])
        extra_vllm_args.extend(shlex.split(self.args.extra_vllm_args))
        extra_vllm_args.extend(self.args.extra_vllm_arg)
        _append_default_vllm_args(extra_vllm_args)
        has_ctx = any(dp.phase == "ctx" for dp in pending)
        has_gen_prefix_cache = (
            any(dp.phase == "gen" for dp in pending)
            and self.args.gen_driver == "prefix_cache"
        )
        runtime_defaults = gpt_oss_runtime_defaults(
            model=unit.row_base["model"],
            system=unit.row_base["system"],
            disable_prefix_caching=not (
                (has_ctx and self.args.ctx_driver == "prefix_cache")
                or has_gen_prefix_cache
            ),
            extra_args=tuple(extra_vllm_args),
        )
        extra_vllm_args = list(runtime_defaults.extra_args)
        if runtime_defaults.kv_cache_dtype:
            extra_vllm_args = [
                "--kv-cache-dtype",
                runtime_defaults.kv_cache_dtype,
                *extra_vllm_args,
            ]
        if runtime_defaults.disable_prefix_caching and not has_cli_flag(
            extra_vllm_args, "--no-enable-prefix-caching"
        ):
            extra_vllm_args.append("--no-enable-prefix-caching")
        if (
            ((has_ctx and self.args.ctx_driver == "prefix_cache") or has_gen_prefix_cache)
            and not has_cli_flag(extra_vllm_args, "--enable-prefix-caching", "--no-enable-prefix-caching")
        ):
            extra_vllm_args.append("--enable-prefix-caching")

        return {
            "attempt_id": attempt_id,
            "work_unit_id": unit.work_unit_id,
            "model_dir": unit.model_dir,
            "target_layers": unit.target_layers,
            "moe_noop": unit.moe_noop,
            "datapoints": [asdict(dp) for dp in pending],
            "status_path": str(self.store.status_path),
            "restrict_cudagraph_sizes": not self.args.no_restrict_cudagraph_sizes,
            "extra_vllm_args": extra_vllm_args,
            "min_max_num_batched_tokens": self.args.min_max_num_batched_tokens,
            "measurement_mode": self.args.measurement_mode,
            "compilation_config_json": self.args.compilation_config_json,
            "ctx_driver": self.args.ctx_driver,
            "ctx_warmup_runs": self.args.ctx_warmup_runs,
            "ctx_measured_runs": self.args.ctx_measured_runs,
            "gen_driver": self.args.gen_driver,
            "gen_warmup_runs": self.args.gen_warmup_runs,
            "gen_measured_runs": self.args.gen_measured_runs,
            "nsys_capture": self.args.nsys_capture,
        }

    def _worker_cmd(self, spec_path: Path, report_base: Path) -> list[str]:
        worker_cmd = [sys.executable, str(Path(__file__).resolve()), "worker", "--spec", str(spec_path)]
        cmd = [
            "nsys",
            "profile",
            "--trace=cuda,nvtx",
            "--sample=none",
            "--cpuctxsw=none",
            "--cuda-graph-trace=node",
            "--force-overwrite=true",
        ]
        if self.args.nsys_capture == "cuda_profiler_api":
            cmd.extend([
                "--capture-range=cudaProfilerApi",
                "--capture-range-end=stop",
            ])
        elif self.args.nsys_capture != "full":
            raise ValueError(f"unsupported nsys capture mode: {self.args.nsys_capture}")
        cmd.extend(["-o", str(report_base), *worker_cmd])
        return cmd

    def _finish_attempt(self, attempt: Attempt, returncode: int) -> bool:
        attempt.stdout_handle.close()
        attempt.stderr_handle.close()
        stderr_tail = _tail(attempt.stderr_path, 120)
        print(
            f"[scheduler] finish gpu={attempt.gpu} attempt={attempt.attempt_id} "
            f"rc={returncode} {attempt.work_unit.work_unit_id}"
        )

        successes = self._parse_attempt_report(attempt)
        if successes:
            key = (attempt.work_unit.work_unit_id, "fatal_cuda")
            self.fatal_streak.pop(key, None)

        if returncode != 0:
            self._mark_crashed_attempt(attempt, returncode, stderr_tail, successes)
        else:
            self._mark_clean_parse_failures(attempt)

        self.store.append_event(
            "attempt_finished",
            work_unit_id=attempt.work_unit.work_unit_id,
            attempt_id=attempt.attempt_id,
            returncode=returncode,
            parsed_successes=successes,
        )
        return bool(self._pending_datapoints(attempt.work_unit))

    def _parse_attempt_report(self, attempt: Attempt) -> int:
        sqlite_path = attempt.report_base.with_suffix(".sqlite")
        rep_path = attempt.report_base.with_suffix(".nsys-rep")
        if not rep_path.exists() or rep_path.stat().st_size == 0:
            return 0
        self.store.append_event(
            "nsys_export_started",
            work_unit_id=attempt.work_unit.work_unit_id,
            attempt_id=attempt.attempt_id,
            rep=str(rep_path),
            rep_bytes=rep_path.stat().st_size,
        )
        export_cmd = [
            "nsys",
            "export",
            "--type",
            "sqlite",
            "--force-overwrite=true",
            "--output",
            str(sqlite_path),
            str(rep_path),
        ]
        try:
            result = subprocess.run(
                export_cmd,
                text=True,
                capture_output=True,
                timeout=self.args.timeout,
                check=False,
            )
            if result.returncode != 0:
                self.store.append_event(
                    "nsys_export_failed",
                    work_unit_id=attempt.work_unit.work_unit_id,
                    attempt_id=attempt.attempt_id,
                    returncode=result.returncode,
                    stderr=result.stderr[-4000:],
                )
                return 0
        except Exception as exc:
            self.store.append_event(
                "nsys_export_failed",
                work_unit_id=attempt.work_unit.work_unit_id,
                attempt_id=attempt.attempt_id,
                error=repr(exc),
            )
            return 0

        self.store.append_event(
            "nsys_export_succeeded",
            work_unit_id=attempt.work_unit.work_unit_id,
            attempt_id=attempt.attempt_id,
            sqlite=str(sqlite_path),
            sqlite_bytes=sqlite_path.stat().st_size if sqlite_path.exists() else 0,
        )
        self.store.append_event(
            "nsys_parse_started",
            work_unit_id=attempt.work_unit.work_unit_id,
            attempt_id=attempt.attempt_id,
            sqlite=str(sqlite_path),
        )
        try:
            rows, meta = parse_step_sweep(
                str(sqlite_path),
                rollup=_effective_rollup(self.args),
                layer=None,
                rank_reduce=self.args.rank_reduce,
                force_nvtx_span=(
                    _effective_rollup(self.args) == DEFAULT_PARITY_ROLLUP
                    and self.args.latency_source == "span"
                ),
            )
            if _should_filter_target_layers(self.args):
                rows = _filter_rows_to_target_layers(rows, attempt.work_unit.target_layers)
        except Exception as exc:
            self.store.append_event(
                "nsys_parse_failed",
                work_unit_id=attempt.work_unit.work_unit_id,
                attempt_id=attempt.attempt_id,
                sqlite=str(sqlite_path),
                error=repr(exc),
            )
            return 0

        parsed = _aggregate_step_rows(rows)
        self.store.append_event(
            "nsys_parse_succeeded",
            work_unit_id=attempt.work_unit.work_unit_id,
            attempt_id=attempt.attempt_id,
            sqlite=str(sqlite_path),
            rows=len(rows),
            meta=meta,
        )
        index = self.store.index()
        successes = 0
        for dp in attempt.work_unit.datapoints:
            dpid = dp.datapoint_id(attempt.work_unit.work_unit_id)
            if dpid not in attempt.pending_ids or index.is_terminal(dpid):
                continue
            aggs = _lookup_aggs(parsed, dp.parse_key())
            if not aggs:
                continue
            latency_us, rms_us, kernel_count, rms_kernel_count, measure_count = _reduce_agg_latency(
                aggs,
                latency_source=self.args.latency_source,
                aggregation=self.args.ctx_repeat_aggregation if dp.phase == "ctx" else self.args.gen_repeat_aggregation,
            )
            row = {
                **attempt.work_unit.row_base,
                "phase": dp.phase,
                "batch_size": dp.batch_size,
                "new_tokens": dp.new_tokens,
                "past_kv": dp.past_kv,
                "latency_ms": latency_us / 1000.0,
                "rms_latency_ms": rms_us / 1000.0,
                "rms_kernel_count": rms_kernel_count,
                "measurement_mode": self.args.measurement_mode,
                "attribution_target": _attribution_target(self.args),
                "includes_moe": _work_unit_includes_moe(attempt.work_unit),
                "vllm_config_hash": _attempt_config_hash(attempt, self.store),
            }
            _append_success_row(self.output_path, row)
            self.store.append_event(
                "success",
                work_unit_id=attempt.work_unit.work_unit_id,
                datapoint_id=dpid,
                attempt_id=attempt.attempt_id,
                latency_ms=row["latency_ms"],
                latency_source=self.args.latency_source,
                repeat_aggregation=(
                    self.args.ctx_repeat_aggregation if dp.phase == "ctx" else self.args.gen_repeat_aggregation
                ),
                measure_count=measure_count,
                kernel_count=kernel_count,
                rms_latency_ms=row["rms_latency_ms"],
                rms_kernel_count=rms_kernel_count,
                sqlite=str(sqlite_path),
            )
            successes += 1
        return successes

    def _mark_clean_parse_failures(self, attempt: Attempt) -> None:
        index = self.store.index()
        for dpid in sorted(attempt.pending_ids):
            if index.is_terminal(dpid):
                continue
            if dpid in index.started or dpid in index.completed:
                self.store.append_event(
                    "failed_parse",
                    work_unit_id=attempt.work_unit.work_unit_id,
                    datapoint_id=dpid,
                    attempt_id=attempt.attempt_id,
                    message="worker exited cleanly but no parsed latency row was found",
                )

    def _mark_crashed_attempt(
        self,
        attempt: Attempt,
        returncode: int,
        stderr_tail: str,
        successes: int,
    ) -> None:
        index = self.store.index()
        signature = _attempt_signature(returncode, stderr_tail)
        active = index.active_started(attempt.work_unit.work_unit_id, attempt.pending_ids)

        if active and not index.is_terminal(active):
            event = "failed_oom" if signature == "oom" else "failed_fatal_cuda"
            self.store.append_event(
                event,
                work_unit_id=attempt.work_unit.work_unit_id,
                datapoint_id=active,
                attempt_id=attempt.attempt_id,
                returncode=returncode,
                signature=signature,
                stderr_tail=stderr_tail[-4000:],
            )
            if event == "failed_oom":
                failed_dp = self._find_datapoint(attempt.work_unit, active)
                if failed_dp:
                    self._mark_oom_dominated(attempt.work_unit, failed_dp, active)

        # Completed-but-unparsed datapoints were attempted.  Mark them terminal
        # so crash recovery never reruns work merely because sqlite parsing lost it.
        index = self.store.index()
        for dpid in sorted(attempt.pending_ids):
            if index.is_terminal(dpid):
                continue
            if dpid in index.completed:
                self.store.append_event(
                    "failed_parse",
                    work_unit_id=attempt.work_unit.work_unit_id,
                    datapoint_id=dpid,
                    attempt_id=attempt.attempt_id,
                    message="worker crashed after execution but no parsed row was found",
                )

        if successes:
            return
        streak_key = (attempt.work_unit.work_unit_id, signature)
        self.fatal_streak[streak_key] = self.fatal_streak.get(streak_key, 0) + 1
        if self.fatal_streak[streak_key] >= FATAL_STREAK_LIMIT:
            index = self.store.index()
            for dp in attempt.work_unit.datapoints:
                dpid = dp.datapoint_id(attempt.work_unit.work_unit_id)
                if dpid in attempt.pending_ids and not index.is_terminal(dpid):
                    self.store.append_event(
                        "skipped_same_error",
                        work_unit_id=attempt.work_unit.work_unit_id,
                        datapoint_id=dpid,
                        attempt_id=attempt.attempt_id,
                        signature=signature,
                        message=f"{FATAL_STREAK_LIMIT} consecutive crashes with no parsed success",
                    )

    def _mark_oom_dominated(self, unit: WorkUnit, failed_dp: DataPoint, failed_id: str) -> None:
        index = self.store.index()
        for dp in unit.datapoints:
            dpid = dp.datapoint_id(unit.work_unit_id)
            if dpid == failed_id or index.is_terminal(dpid):
                continue
            if oom_dominates(failed_dp, dp):
                self.store.append_event(
                    "skipped_oom_dominated",
                    work_unit_id=unit.work_unit_id,
                    datapoint_id=dpid,
                    caused_by=failed_id,
                )

    @staticmethod
    def _find_datapoint(unit: WorkUnit, datapoint_id: str) -> DataPoint | None:
        for dp in unit.datapoints:
            if dp.datapoint_id(unit.work_unit_id) == datapoint_id:
                return dp
        return None


def _tail(path: Path, max_lines: int) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def _worker_append_event(
    status_path: Path,
    event: str,
    *,
    work_unit_id: str,
    datapoint_id: str | None = None,
    **extra: Any,
) -> None:
    row = {
        "event": event,
        "work_unit_id": work_unit_id,
        "datapoint_id": datapoint_id,
        "ts": _utc_now(),
        **extra,
    }
    row = {k: v for k, v in row.items() if v is not None}
    status_path.parent.mkdir(parents=True, exist_ok=True)
    with status_path.open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f, fcntl.LOCK_UN)


def _dummy_prompts(
    batch_size: int,
    input_len: int,
    token_config: RandomPromptTokenConfig,
):
    import random

    return [
        {"prompt_token_ids": sample_prompt_token_ids(random, input_len, token_config)}
        for _ in range(batch_size)
    ]


def _deterministic_token_prompts(
    batch_size: int,
    input_len: int,
    token_config: RandomPromptTokenConfig,
    *,
    request_index_offset: int,
    cache_salt_prefix: str | None = None,
) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    for request_idx in range(batch_size):
        prompt: dict[str, Any] = {
            "prompt_token_ids": make_prompt_token_ids(
                prompt_token_seed=0,
                token_count=input_len,
                request_index=request_index_offset + request_idx,
                token_config=token_config,
            )
        }
        if cache_salt_prefix is not None:
            prompt["cache_salt"] = f"{cache_salt_prefix}:req{request_idx}"
        prompts.append(prompt)
    return prompts


def _deterministic_prefix_suffix_prompts(
    batch_size: int,
    prefix_len: int,
    suffix_len: int,
    token_config: RandomPromptTokenConfig,
    *,
    prefix_request_index_offset: int,
    suffix_request_index_offset: int,
    cache_salt_prefix: str | None = None,
) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    for request_idx in range(batch_size):
        prefix = make_prompt_token_ids(
            prompt_token_seed=0,
            token_count=prefix_len,
            request_index=prefix_request_index_offset + request_idx,
            token_config=token_config,
        )
        suffix = make_prompt_token_ids(
            prompt_token_seed=0,
            token_count=suffix_len,
            request_index=suffix_request_index_offset + request_idx,
            token_config=token_config,
        )
        prompt: dict[str, Any] = {"prompt_token_ids": [*prefix, *suffix]}
        if cache_salt_prefix is not None:
            prompt["cache_salt"] = f"{cache_salt_prefix}:req{request_idx}"
        prompts.append(prompt)
    return prompts


def _run_generate(
    llm,
    sampling_params,
    *,
    batch_size: int,
    input_len: int,
    token_config: RandomPromptTokenConfig,
    request_index_offset: int | None = None,
    cache_salt_prefix: str | None = None,
) -> None:
    if request_index_offset is None and cache_salt_prefix is None:
        prompts = _dummy_prompts(batch_size, input_len, token_config)
    else:
        prompts = _deterministic_token_prompts(
            batch_size,
            input_len,
            token_config,
            request_index_offset=0 if request_index_offset is None else request_index_offset,
            cache_salt_prefix=cache_salt_prefix,
        )
    llm.generate(
        prompts,
        sampling_params=sampling_params,
        use_tqdm=False,
    )


def _run_generate_prefix_suffix(
    llm,
    sampling_params,
    *,
    batch_size: int,
    prefix_len: int,
    suffix_len: int,
    token_config: RandomPromptTokenConfig,
    prefix_request_index_offset: int,
    suffix_request_index_offset: int,
    cache_salt_prefix: str | None = None,
) -> None:
    prompts = _deterministic_prefix_suffix_prompts(
        batch_size,
        prefix_len,
        suffix_len,
        token_config,
        prefix_request_index_offset=prefix_request_index_offset,
        suffix_request_index_offset=suffix_request_index_offset,
        cache_salt_prefix=cache_salt_prefix,
    )
    llm.generate(
        prompts,
        sampling_params=sampling_params,
        use_tqdm=False,
    )


def _engine_tokens(
    *,
    model_dir: str,
    datapoints: list[DataPoint],
    restrict_cudagraph_sizes: bool,
    extra_vllm_args: list[str],
    min_max_num_batched_tokens: int = 1,
    measurement_mode: str = "deployment-parity",
    compilation_config_json: str | None = None,
    gen_driver: str = "prefix_cache",
) -> list[str]:
    ctx_points = [dp for dp in datapoints if dp.phase == "ctx"]
    gen_points = [dp for dp in datapoints if dp.phase == "gen"]
    ctx_max_total = max((dp.new_tokens + dp.past_kv for dp in ctx_points), default=0)
    gen_max_past = max((dp.past_kv for dp in gen_points), default=0)
    gen_batch_sizes = sorted({dp.batch_size for dp in gen_points})
    max_seq_len = max(
        2,
        ctx_max_total + 1 if ctx_points else 0,
        gen_max_past + 2 if gen_points else 0,
    )
    max_num_batched_tokens = _max_num_batched_tokens_for_datapoints(
        datapoints,
        min_max_num_batched_tokens,
        gen_driver=gen_driver,
    )
    engine_max_num_batched_tokens: int | None = max_num_batched_tokens
    if gen_driver == "prefill" and gen_points and not ctx_points:
        max_batch_size = max(gen_batch_sizes)
        normal_prefill_budget = max(
            PREFILL_DECODE_MAX_NUM_BATCHED_TOKENS,
            128 * max_batch_size,
        )
        uniform_decode_budget = max(
            (dp.batch_size * dp.past_kv for dp in gen_points),
            default=0,
        )
        if min_max_num_batched_tokens <= 1:
            engine_max_num_batched_tokens = max(
                normal_prefill_budget,
                uniform_decode_budget,
            )
        else:
            engine_max_num_batched_tokens = max(
                engine_max_num_batched_tokens,
                128 * max_batch_size,
                uniform_decode_budget,
            )

    tokens = build_engine_args(
        VllmDeploymentConfig(
            model=model_dir,
            max_model_len=max_seq_len,
            max_num_seqs=max(gen_batch_sizes) if gen_batch_sizes else None,
            max_num_batched_tokens=engine_max_num_batched_tokens,
        )
    )

    if compilation_config_json == "default":
        compilation_config = None
    elif compilation_config_json is not None:
        compilation_config = json.loads(compilation_config_json)
    elif measurement_mode == "deployment-parity":
        compilation_config = None
    elif measurement_mode == "attribution" and gen_batch_sizes:
        # Match real vLLM deployment shape: one engine.  FULL_DECODE_ONLY lets
        # vLLM use full CUDA graphs for uniform decode batches while running
        # prefill/mixed batches through its normal non-full-graph path.
        compilation_config: dict[str, Any] = {
            "mode": 0,
            "cudagraph_mode": "FULL_DECODE_ONLY",
        }
        if restrict_cudagraph_sizes:
            compilation_config.update({
                "cudagraph_capture_sizes": gen_batch_sizes,
                "max_cudagraph_capture_size": max(gen_batch_sizes),
            })
    elif measurement_mode == "attribution":
        compilation_config = {"mode": 0, "cudagraph_mode": "NONE"}
    else:
        raise ValueError(f"unsupported measurement mode: {measurement_mode}")

    if compilation_config is not None:
        tokens.extend(["--compilation-config", json.dumps(compilation_config)])
    tokens.extend(extra_vllm_args)
    return tokens


def _create_llm(engine_tokens: list[str]):
    from vllm.engine.arg_utils import EngineArgs

    parser = argparse.ArgumentParser(add_help=False)
    EngineArgs.add_cli_args(parser)
    parser.set_defaults(
        load_format="dummy",
        trust_remote_code=True,
        enable_layerwise_nvtx_tracing=True,
        skip_tokenizer_init=True,
    )
    args = parser.parse_args(engine_tokens)
    engine_args = EngineArgs.from_cli_args(args)
    from vllm import LLM

    return LLM.from_engine_args(engine_args)


def _classify_exception(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}"
    if _is_oom_text(text):
        return "oom"
    if _is_fatal_cuda_text(text):
        return "fatal_cuda"
    return "error"


def _worker_datapoint_id(work_unit_id: str, dp: DataPoint) -> str:
    return dp.datapoint_id(work_unit_id)


def _write_marker_control(
    *,
    active_iterations: str | Iterable[int] = "",
    trigger: str | None = None,
    phase: str | None = None,
    step: int | None = None,
    bs: int | None = None,
    past: int | None = None,
    run: int | None = None,
    **extras: Any,
) -> None:
    raw_path = os.environ.get("LAYERWISE_CONTROL_FILE")
    if not raw_path:
        return
    if isinstance(active_iterations, str):
        active = [int(x) for x in active_iterations.split(",") if x.strip()]
    else:
        active = [int(x) for x in active_iterations]
    payload = {
        "active_iterations": active,
        "trigger": trigger,
        "phase": phase,
        "step": step,
        "bs": bs,
        "past": past,
        "run": run,
    }
    payload.update(extras)
    path = Path(raw_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, sort_keys=True, separators=(",", ":"))
        f.write("\n")
    os.replace(tmp, path)


def _set_marker_state(
    marker_mod,
    *,
    active_iterations: str | Iterable[int] = "",
    trigger: str | None = None,
    phase: str | None = None,
    step: int | None = None,
    bs: int | None = None,
    past: int | None = None,
    run: int | None = None,
    **extras: Any,
) -> None:
    if isinstance(active_iterations, str):
        active_text = active_iterations
    else:
        active_text = ",".join(str(int(x)) for x in active_iterations)
    os.environ["LAYERWISE_ACTIVE_ITERATIONS"] = active_text
    if phase is not None:
        os.environ["LAYERWISE_PROGRESS_PHASE"] = phase
    else:
        phase = os.environ.get("LAYERWISE_PROGRESS_PHASE")
    if run is None:
        os.environ.pop("LAYERWISE_MEASURE_RUN", None)
    else:
        os.environ["LAYERWISE_MEASURE_RUN"] = str(run)
    if step is None and bs is None and past is None and run is None:
        marker_mod.clear_forced_step_meta()
    else:
        marker_mod.set_forced_step_meta(step=step, bs=bs, past=past, run=run)
    _write_marker_control(
        active_iterations=active_iterations,
        trigger=trigger,
        phase=phase,
        step=step,
        bs=bs,
        past=past,
        run=run,
        **extras,
    )


def _ctx_marker_iteration(
    dp: DataPoint,
    max_num_batched_tokens: int,
    *,
    ctx_driver: str = "chunked",
) -> int:
    if ctx_driver not in {"chunked", "prefix_cache"}:
        raise ValueError(f"unsupported ctx driver: {ctx_driver}")
    if max_num_batched_tokens < 1:
        raise ValueError(f"max_num_batched_tokens must be >= 1, got {max_num_batched_tokens}")
    if ctx_driver == "prefix_cache":
        return 1
    if dp.past_kv == 0:
        return 1
    return int(dp.past_kv // max_num_batched_tokens) + 1


def run_worker(spec_path: Path) -> None:
    spec = json.loads(spec_path.read_text())
    status_path = Path(spec["status_path"])
    work_unit_id = spec["work_unit_id"]
    datapoints = [DataPoint(**raw) for raw in spec["datapoints"]]
    _worker_append_event(
        status_path,
        "worker_entered",
        work_unit_id=work_unit_id,
        attempt_id=spec["attempt_id"],
    )

    # Layerwise simulates TP/EP by patching the model config; the runtime engine
    # itself stays single-process on one physical GPU.
    _worker_append_event(
        status_path,
        "worker_environment_started",
        work_unit_id=work_unit_id,
        attempt_id=spec["attempt_id"],
    )
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ["LAYERWISE_TARGET_LAYERS"] = ",".join(str(x) for x in spec["target_layers"])
    if spec.get("moe_noop"):
        os.environ["LAYERWISE_MOE_NOOP"] = "1"
    else:
        os.environ.pop("LAYERWISE_MOE_NOOP", None)
    max_num_batched_tokens = _max_num_batched_tokens_for_datapoints(
        datapoints,
        int(spec.get("min_max_num_batched_tokens", 1)),
        gen_driver=str(spec.get("gen_driver", "prefix_cache")),
    )
    ctx_driver = str(spec.get("ctx_driver", "chunked"))
    iterations = {1}
    iterations.update(
        _ctx_marker_iteration(dp, max_num_batched_tokens, ctx_driver=ctx_driver)
        for dp in datapoints
        if dp.phase == "ctx"
    )
    iterations.update(dp.past_kv + 1 for dp in datapoints if dp.phase == "gen")
    os.environ["LAYERWISE_STEP_ITERATIONS"] = ",".join(str(x) for x in sorted(iterations))
    os.environ["LAYERWISE_BENCH_MIN_NEW"] = "1"
    os.environ["LAYERWISE_PROGRESS_FILE"] = str(status_path)
    os.environ["LAYERWISE_WORK_UNIT_ID"] = work_unit_id
    os.environ["LAYERWISE_CONTROL_FILE"] = str(
        status_path.parent / "marker_control" / f"{work_unit_id}_a{spec['attempt_id']}.json"
    )
    # Keep the marker installed but inactive during vLLM profile/capture work.
    os.environ["LAYERWISE_ACTIVE_ITERATIONS"] = ""
    _write_marker_control(active_iterations="")
    _worker_append_event(
        status_path,
        "worker_environment_finished",
        work_unit_id=work_unit_id,
        attempt_id=spec["attempt_id"],
    )

    sys.path.insert(0, str(_THIS_DIR))
    import multiprocessing as mp

    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass

    # These patches install vLLM hooks at import time.  Keep them inside worker
    # mode so scheduler/test imports remain vLLM-free.
    _worker_append_event(
        status_path,
        "worker_imports_started",
        work_unit_id=work_unit_id,
        attempt_id=spec["attempt_id"],
    )
    import vllm_layer_skip_patch  # noqa: F401
    import vllm_step_marker
    from vllm import SamplingParams
    _worker_append_event(
        status_path,
        "worker_imports_finished",
        work_unit_id=work_unit_id,
        attempt_id=spec["attempt_id"],
    )

    _worker_append_event(status_path, "work_unit_started", work_unit_id=work_unit_id, attempt_id=spec["attempt_id"])
    _worker_append_event(status_path, "engine_started", work_unit_id=work_unit_id)
    _worker_append_event(status_path, "engine_args_started", work_unit_id=work_unit_id, attempt_id=spec["attempt_id"])
    prompt_token_config = load_random_prompt_token_config(spec["model_dir"])
    engine_tokens = _engine_tokens(
        model_dir=spec["model_dir"],
        datapoints=datapoints,
        restrict_cudagraph_sizes=spec["restrict_cudagraph_sizes"],
        extra_vllm_args=spec["extra_vllm_args"],
        min_max_num_batched_tokens=spec.get("min_max_num_batched_tokens", 1),
        measurement_mode=str(spec.get("measurement_mode", "deployment-parity")),
        compilation_config_json=spec.get("compilation_config_json"),
        gen_driver=str(spec.get("gen_driver", "prefix_cache")),
    )
    _worker_append_event(status_path, "engine_args_finished", work_unit_id=work_unit_id, attempt_id=spec["attempt_id"])
    _worker_append_event(status_path, "engine_create_started", work_unit_id=work_unit_id, attempt_id=spec["attempt_id"])
    llm = _create_llm(engine_tokens)
    _worker_append_event(status_path, "engine_create_finished", work_unit_id=work_unit_id, attempt_id=spec["attempt_id"])
    _worker_append_event(status_path, "engine_metadata_started", work_unit_id=work_unit_id, attempt_id=spec["attempt_id"])
    effective_config = None
    runtime_vllm_config = find_runtime_vllm_config(llm)
    has_ctx = any(dp.phase == "ctx" for dp in datapoints)
    has_gen_prefix_cache = any(dp.phase == "gen" for dp in datapoints) and str(
        spec.get("gen_driver", "prefix_cache")
    ) == "prefix_cache"
    if (
        (has_ctx and ctx_driver == "prefix_cache") or has_gen_prefix_cache
    ) and runtime_vllm_config is not None:
        cache_config = getattr(runtime_vllm_config, "cache_config", None)
        if getattr(cache_config, "enable_prefix_caching", None) is False:
            raise RuntimeError("prefix-cache ctx/gen driver requires vLLM prefix caching")
    if runtime_vllm_config is not None:
        effective_config = summarize_vllm_config(runtime_vllm_config)
    metadata = make_metadata(
        artifact_kind="layerwise",
        measurement_mode=str(spec.get("measurement_mode", "deployment-parity")),
        engine_args=engine_tokens,
        effective_config=effective_config,
        extra={
            "work_unit_id": work_unit_id,
            "attempt_id": spec["attempt_id"],
            "target_layers": spec["target_layers"],
            "moe_noop": bool(spec.get("moe_noop")),
            "attribution_target": (
                "cudagraph_wrapper"
                if str(spec.get("measurement_mode", "deployment-parity")) == "deployment-parity"
                else "module_nvtx"
            ),
        },
    )
    if spec.get("metadata_path"):
        write_metadata(spec["metadata_path"], metadata)
    _worker_append_event(status_path, "engine_metadata_finished", work_unit_id=work_unit_id, attempt_id=spec["attempt_id"])
    _worker_append_event(
        status_path,
        "engine_metadata_written",
        work_unit_id=work_unit_id,
        attempt_id=spec["attempt_id"],
        metadata_path=spec.get("metadata_path"),
        vllm_config_hash=metadata["vllm_config_hash"],
    )
    _worker_append_event(status_path, "engine_ready", work_unit_id=work_unit_id)
    cuda_profiler_capture = str(spec.get("nsys_capture", "cuda_profiler_api")) == "cuda_profiler_api"
    try:
        _worker_append_event(status_path, "measurement_started", work_unit_id=work_unit_id, attempt_id=spec["attempt_id"])
        _worker_set_cuda_profiler_capture(
            status_path=status_path,
            work_unit_id=work_unit_id,
            attempt_id=spec["attempt_id"],
            enabled=cuda_profiler_capture,
            action="start",
        )
        try:
            ctx_points = [dp for dp in datapoints if dp.phase == "ctx"]
            if ctx_points:
                os.environ["LAYERWISE_PROGRESS_PHASE"] = "ctx"
                os.environ["LAYERWISE_ACTIVE_ITERATIONS"] = "1"
                _worker_run_ctx(
                    status_path,
                    work_unit_id,
                    llm,
                    SamplingParams,
                    vllm_step_marker,
                    ctx_points,
                    prompt_token_config=prompt_token_config,
                    warmup_runs=int(spec.get("ctx_warmup_runs", 0)),
                    measured_runs=int(spec.get("ctx_measured_runs", 1)),
                    max_num_batched_tokens=max_num_batched_tokens,
                    driver=ctx_driver,
                )

            gen_points = [dp for dp in datapoints if dp.phase == "gen"]
            if gen_points:
                gen_iterations = sorted({dp.past_kv + 1 for dp in gen_points})
                os.environ["LAYERWISE_PROGRESS_PHASE"] = "gen"
                os.environ["LAYERWISE_ACTIVE_ITERATIONS"] = ",".join(str(x) for x in gen_iterations)
                _worker_run_gen(
                    status_path,
                    work_unit_id,
                    llm,
                    SamplingParams,
                    vllm_step_marker,
                    gen_points,
                    prompt_token_config=prompt_token_config,
                    driver=str(spec.get("gen_driver", "prefix_cache")),
                    warmup_runs=int(spec.get("gen_warmup_runs", 0)),
                    measured_runs=int(spec.get("gen_measured_runs", 1)),
                )
        finally:
            _worker_set_cuda_profiler_capture(
                status_path=status_path,
                work_unit_id=work_unit_id,
                attempt_id=spec["attempt_id"],
                enabled=cuda_profiler_capture,
                action="stop",
            )
        _worker_append_event(status_path, "measurement_finished", work_unit_id=work_unit_id, attempt_id=spec["attempt_id"])
    finally:
        _worker_append_event(status_path, "worker_cleanup_started", work_unit_id=work_unit_id, attempt_id=spec["attempt_id"])
        del llm
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except Exception:
            pass
        os.environ.pop("LAYERWISE_ACTIVE_ITERATIONS", None)
        _worker_append_event(status_path, "worker_cleanup_finished", work_unit_id=work_unit_id, attempt_id=spec["attempt_id"])
    _worker_append_event(status_path, "work_unit_finished", work_unit_id=work_unit_id, attempt_id=spec["attempt_id"])


def _worker_run_ctx(
    status_path: Path,
    work_unit_id: str,
    llm,
    sampling_cls,
    marker_mod,
    datapoints: list[DataPoint],
    *,
    prompt_token_config: RandomPromptTokenConfig,
    warmup_runs: int = 0,
    measured_runs: int = 1,
    max_num_batched_tokens: int = 1,
    driver: str = "chunked",
) -> None:
    if warmup_runs < 0:
        raise ValueError(f"warmup_runs must be >= 0, got {warmup_runs}")
    if measured_runs < 1:
        raise ValueError(f"measured_runs must be >= 1, got {measured_runs}")
    if driver not in {"chunked", "prefix_cache"}:
        raise ValueError(f"unsupported ctx driver: {driver}")
    sampling_params = sampling_cls(
        temperature=1.0,
        top_p=1.0,
        ignore_eos=True,
        max_tokens=1,
        detokenize=False,
    )
    pruned: set[str] = set()
    filled_prefixes: set[tuple[int, int]] = set()
    for dp in datapoints:
        dpid = _worker_datapoint_id(work_unit_id, dp)
        if dpid in pruned:
            continue
        input_len = dp.past_kv + dp.new_tokens
        marker_iteration = _ctx_marker_iteration(
            dp,
            max_num_batched_tokens,
            ctx_driver=driver,
        )
        _set_marker_state(
            marker_mod,
            active_iterations="",
            phase="ctx",
            step=dp.new_tokens,
            bs=dp.batch_size,
            past=dp.past_kv,
        )
        try:
            if driver == "prefix_cache":
                prefix_key = (int(dp.batch_size), int(dp.past_kv))
                for run_idx in range(warmup_runs):
                    fill_prefix = dp.past_kv > 0 and prefix_key not in filled_prefixes
                    _run_prefix_cached_ctx_iteration(
                        llm,
                        sampling_params,
                        dp,
                        work_unit_id=work_unit_id,
                        run_idx=run_idx,
                        warmup=True,
                        fill_prefix=fill_prefix,
                        token_config=prompt_token_config,
                        active_iteration="",
                        marker_mod=marker_mod,
                    )
                    if fill_prefix:
                        filled_prefixes.add(prefix_key)
                for run_idx in range(measured_runs):
                    fill_prefix = dp.past_kv > 0 and prefix_key not in filled_prefixes
                    _set_marker_state(
                        marker_mod,
                        active_iterations=str(marker_iteration),
                        phase="ctx",
                        step=dp.new_tokens,
                        bs=dp.batch_size,
                        past=dp.past_kv,
                        run=run_idx,
                    )
                    _run_prefix_cached_ctx_iteration(
                        llm,
                        sampling_params,
                        dp,
                        work_unit_id=work_unit_id,
                        run_idx=run_idx,
                        warmup=False,
                        fill_prefix=fill_prefix,
                        token_config=prompt_token_config,
                        active_iteration=str(marker_iteration),
                        marker_mod=marker_mod,
                    )
                    if fill_prefix:
                        filled_prefixes.add(prefix_key)
            else:
                _set_marker_state(marker_mod, active_iterations="", phase="ctx")
                for _ in range(warmup_runs):
                    _run_generate(
                        llm,
                        sampling_params,
                        batch_size=dp.batch_size,
                        input_len=input_len,
                        token_config=prompt_token_config,
                    )
                for run_idx in range(measured_runs):
                    _set_marker_state(
                        marker_mod,
                        active_iterations=str(marker_iteration),
                        phase="ctx",
                        step=dp.new_tokens,
                        bs=dp.batch_size,
                        past=dp.past_kv,
                        run=run_idx,
                    )
                    _run_generate(
                        llm,
                        sampling_params,
                        batch_size=dp.batch_size,
                        input_len=input_len,
                        token_config=prompt_token_config,
                    )
        except Exception as exc:
            kind = _classify_exception(exc)
            if kind == "oom":
                _worker_append_event(
                    status_path,
                    "failed_oom",
                    work_unit_id=work_unit_id,
                    datapoint_id=dpid,
                    message=str(exc),
                )
                _worker_empty_cache()
                for candidate in datapoints:
                    cid = _worker_datapoint_id(work_unit_id, candidate)
                    if cid != dpid and oom_dominates(dp, candidate):
                        pruned.add(cid)
                        _worker_append_event(
                            status_path,
                            "skipped_oom_dominated",
                            work_unit_id=work_unit_id,
                            datapoint_id=cid,
                            caused_by=dpid,
                        )
                continue
            if kind == "fatal_cuda":
                _worker_append_event(
                    status_path,
                    "failed_fatal_cuda",
                    work_unit_id=work_unit_id,
                    datapoint_id=dpid,
                    message=str(exc),
                )
                raise
            _worker_append_event(
                status_path,
                "failed_error",
                work_unit_id=work_unit_id,
                datapoint_id=dpid,
                message=str(exc),
                traceback=traceback.format_exc()[-4000:],
            )
            _worker_empty_cache()
        finally:
            _set_marker_state(marker_mod, active_iterations="", phase="ctx")


def _ctx_prefix_request_index_offset(dp: DataPoint) -> int:
    return int(dp.past_kv) * 1_000_003 + int(dp.batch_size) * 101


def _ctx_suffix_request_index_offset(dp: DataPoint, run_idx: int, *, warmup: bool) -> int:
    phase_offset = 0 if warmup else 100_000_000
    return (
        phase_offset
        + int(dp.past_kv) * 1_000_003
        + int(dp.new_tokens) * 10_007
        + int(dp.batch_size) * 101
        + int(run_idx) * 10_000
    )


def _ctx_cache_salt_prefix(
    work_unit_id: str,
    dp: DataPoint,
) -> str:
    return f"layerwise-ctx:{work_unit_id}:bs{dp.batch_size}:past{dp.past_kv}"


def _run_prefix_cached_ctx_iteration(
    llm,
    sampling_params,
    dp: DataPoint,
    *,
    work_unit_id: str,
    run_idx: int,
    warmup: bool,
    fill_prefix: bool,
    token_config: RandomPromptTokenConfig,
    active_iteration: str,
    marker_mod,
) -> None:
    prefix_request_index_offset = _ctx_prefix_request_index_offset(dp)
    suffix_request_index_offset = _ctx_suffix_request_index_offset(dp, run_idx, warmup=warmup)
    cache_salt_prefix = _ctx_cache_salt_prefix(work_unit_id, dp)
    if fill_prefix and dp.past_kv > 0:
        _set_marker_state(marker_mod, active_iterations="", phase="ctx")
        _run_generate(
            llm,
            sampling_params,
            batch_size=dp.batch_size,
            input_len=dp.past_kv,
            token_config=token_config,
            request_index_offset=prefix_request_index_offset,
            cache_salt_prefix=cache_salt_prefix,
        )
    _set_marker_state(
        marker_mod,
        active_iterations=active_iteration,
        phase="ctx",
        step=dp.new_tokens,
        bs=dp.batch_size,
        past=dp.past_kv,
        run=run_idx if not warmup else None,
    )
    _run_generate_prefix_suffix(
        llm,
        sampling_params,
        batch_size=dp.batch_size,
        prefix_len=dp.past_kv,
        suffix_len=dp.new_tokens,
        token_config=token_config,
        prefix_request_index_offset=prefix_request_index_offset,
        suffix_request_index_offset=suffix_request_index_offset,
        cache_salt_prefix=cache_salt_prefix,
    )


def _gen_prompt_request_index_offset(dp: DataPoint) -> int:
    return 0


def _gen_cache_salt_prefix(work_unit_id: str, dp: DataPoint) -> str:
    return f"layerwise-gen:{work_unit_id}"


def _run_prefix_cached_gen_iteration(
    llm,
    fill_sampling_params,
    measure_sampling_params,
    dp: DataPoint,
    *,
    token_config: RandomPromptTokenConfig,
    fill_cache: bool,
) -> None:
    if dp.past_kv <= 0:
        raise ValueError("gen_driver=prefix_cache requires past_kv > 0")
    request_index_offset = _gen_prompt_request_index_offset(dp)
    cache_salt_prefix = _gen_cache_salt_prefix(os.environ.get("LAYERWISE_WORK_UNIT_ID", ""), dp)
    if fill_cache:
        _run_generate(
            llm,
            fill_sampling_params,
            batch_size=dp.batch_size,
            input_len=dp.past_kv,
            token_config=token_config,
            request_index_offset=request_index_offset,
            cache_salt_prefix=cache_salt_prefix,
        )
        return
    _run_generate(
        llm,
        measure_sampling_params,
        batch_size=dp.batch_size,
        input_len=dp.past_kv,
        token_config=token_config,
        request_index_offset=request_index_offset,
        cache_salt_prefix=cache_salt_prefix,
    )


def _worker_run_gen(
    status_path: Path,
    work_unit_id: str,
    llm,
    sampling_cls,
    marker_mod,
    datapoints: list[DataPoint],
    *,
    prompt_token_config: RandomPromptTokenConfig,
    driver: str = "prefill",
    warmup_runs: int = 0,
    measured_runs: int = 1,
) -> None:
    if warmup_runs < 0:
        raise ValueError(f"warmup_runs must be >= 0, got {warmup_runs}")
    if measured_runs < 1:
        raise ValueError(f"measured_runs must be >= 1, got {measured_runs}")
    by_batch: dict[int, list[DataPoint]] = {}
    for dp in datapoints:
        by_batch.setdefault(dp.batch_size, []).append(dp)
    max_past = max(dp.past_kv for dp in datapoints)
    sampling_params = sampling_cls(
        temperature=1.0,
        top_p=1.0,
        ignore_eos=True,
        max_tokens=2 if driver in {"prefill", "prefix_cache"} else max_past + 1,
        detokenize=False,
    )
    fill_sampling_params = sampling_cls(
        temperature=1.0,
        top_p=1.0,
        ignore_eos=True,
        max_tokens=1,
        detokenize=False,
    )
    if driver not in {"decode_sweep", "prefill", "prefix_cache"}:
        raise ValueError(f"unsupported gen driver: {driver}")
    if driver == "prefix_cache":
        max_batch_size = max(by_batch)
        fill_dp = DataPoint("gen", max_batch_size, 1, max_past)
        _set_marker_state(marker_mod, active_iterations="", phase="gen")
        _run_prefix_cached_gen_iteration(
            llm,
            fill_sampling_params,
            sampling_params,
            fill_dp,
            token_config=prompt_token_config,
            fill_cache=True,
        )
    for batch_size in sorted(by_batch):
        _set_marker_state(marker_mod, active_iterations="", phase="gen")
        _worker_append_event(
            status_path,
            "batch_started",
            work_unit_id=work_unit_id,
            batch_size=batch_size,
            max_past_kv=max_past,
        )
        if driver == "prefix_cache":
            try:
                for dp in sorted(by_batch[batch_size], key=lambda item: item.past_kv):
                    _set_marker_state(marker_mod, active_iterations="", phase="gen")
                    for _ in range(warmup_runs):
                        _set_marker_state(marker_mod, active_iterations="", phase="gen")
                        _run_prefix_cached_gen_iteration(
                            llm,
                            fill_sampling_params,
                            sampling_params,
                            dp,
                            token_config=prompt_token_config,
                            fill_cache=False,
                        )
                    for run_idx in range(measured_runs):
                        _set_marker_state(
                            marker_mod,
                            active_iterations="",
                            trigger="decode_only",
                            allow_new_cached=True,
                            phase="gen",
                            step=dp.past_kv + 1,
                            bs=batch_size,
                            past=dp.past_kv,
                            run=run_idx,
                        )
                        _run_prefix_cached_gen_iteration(
                            llm,
                            fill_sampling_params,
                            sampling_params,
                            dp,
                            token_config=prompt_token_config,
                            fill_cache=False,
                        )
            finally:
                _set_marker_state(marker_mod, active_iterations="", phase="gen")
            _worker_append_event(status_path, "batch_finished", work_unit_id=work_unit_id, batch_size=batch_size)
            continue
        if driver == "prefill":
            try:
                for dp in sorted(by_batch[batch_size], key=lambda item: item.past_kv):
                    _set_marker_state(marker_mod, active_iterations="", phase="gen")
                    for _ in range(warmup_runs):
                        _run_generate(
                            llm,
                            sampling_params,
                            batch_size=batch_size,
                            input_len=dp.past_kv,
                            token_config=prompt_token_config,
                        )
                    for run_idx in range(measured_runs):
                        _set_marker_state(
                            marker_mod,
                            active_iterations="",
                            trigger="decode_only",
                            phase="gen",
                            step=dp.past_kv + 1,
                            bs=batch_size,
                            past=dp.past_kv,
                            run=run_idx,
                        )
                        _run_generate(
                            llm,
                            sampling_params,
                            batch_size=batch_size,
                            input_len=dp.past_kv,
                            token_config=prompt_token_config,
                        )
            finally:
                _set_marker_state(marker_mod, active_iterations="", phase="gen")
            _worker_append_event(status_path, "batch_finished", work_unit_id=work_unit_id, batch_size=batch_size)
            continue
        # Gen datapoint starts/completions are emitted by vllm_step_marker at
        # each target iteration.  One generate call intentionally covers many past_kv
        # datapoints for the same batch size.
        _set_marker_state(marker_mod, active_iterations="", phase="gen")
        try:
            for _ in range(warmup_runs):
                _run_generate(
                    llm,
                    sampling_params,
                    batch_size=batch_size,
                    input_len=1,
                    token_config=prompt_token_config,
                )
            for run_idx in range(measured_runs):
                _set_marker_state(
                    marker_mod,
                    active_iterations=sorted({dp.past_kv + 1 for dp in by_batch[batch_size]}),
                    phase="gen",
                    run=run_idx,
                )
                _run_generate(
                    llm,
                    sampling_params,
                    batch_size=batch_size,
                    input_len=1,
                    token_config=prompt_token_config,
                )
        finally:
            _set_marker_state(marker_mod, active_iterations="", phase="gen")
        _worker_append_event(status_path, "batch_finished", work_unit_id=work_unit_id, batch_size=batch_size)


def _worker_empty_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _cuda_profiler_call(action: str) -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            cudart = torch.cuda.cudart()
            if action == "start":
                cudart.cudaProfilerStart()
            elif action == "stop":
                cudart.cudaProfilerStop()
            else:
                raise ValueError(f"unsupported cuda profiler action: {action}")
            return
    except Exception:
        pass

    try:
        from cuda.bindings import runtime as cuda_runtime

        if action == "start":
            cuda_runtime.cudaProfilerStart()
        elif action == "stop":
            cuda_runtime.cudaProfilerStop()
        else:
            raise ValueError(f"unsupported cuda profiler action: {action}")
    except Exception as exc:
        raise RuntimeError(f"cudaProfiler{action.title()} failed") from exc


def _worker_set_cuda_profiler_capture(
    *,
    status_path: Path,
    work_unit_id: str,
    attempt_id: int,
    enabled: bool,
    action: str,
) -> None:
    if not enabled:
        return
    event_prefix = f"cuda_profiler_{action}"
    _worker_append_event(
        status_path,
        f"{event_prefix}_started",
        work_unit_id=work_unit_id,
        attempt_id=attempt_id,
    )
    try:
        _cuda_profiler_call(action)
    except Exception as exc:
        _worker_append_event(
            status_path,
            f"{event_prefix}_failed",
            work_unit_id=work_unit_id,
            attempt_id=attempt_id,
            error=repr(exc),
        )
        raise
    _worker_append_event(
        status_path,
        f"{event_prefix}_finished",
        work_unit_id=work_unit_id,
        attempt_id=attempt_id,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF model name or local path")
    parser.add_argument("--output", default="vllm_layerwise_perf.csv")
    parser.add_argument("--work-dir", default="profiles/vllm_layerwise")
    parser.add_argument("--config-cache-dir", default=None)
    parser.add_argument("--no-config-cache", action="store_true")
    parser.add_argument("--system", default=None)
    parser.add_argument("--framework-version", default=None)
    parser.add_argument("--tp-sizes", default="1,2,4,8")
    parser.add_argument("--moe-tp", type=int, default=1)
    parser.add_argument("--num-slots", type=int, default=None)
    parser.add_argument("--include-moe-layer", action="store_true")
    parser.add_argument(
        "--moe-noop",
        action="store_true",
        help=(
            "For MoE configs, replace target-layer routed MLP/MoE modules with "
            "identity so layerwise can collect attention/norm overhead while "
            "MoE compute comes from the op-level collector."
        ),
    )
    parser.add_argument(
        "--target-layer-count",
        type=int,
        default=1,
        help="Number of initial dense layers to keep in the patched model.",
    )
    parser.add_argument(
        "--target-layers",
        default=None,
        help=(
            "Comma-separated explicit dense layer indices to keep. Overrides "
            "--target-layer-count and patches num_hidden_layers to max(index)+1."
        ),
    )
    parser.add_argument(
        "--target-layer-config-depth",
        type=int,
        default=None,
        help=(
            "Dense config depth to instantiate when using layer skipping. "
            "Defaults to the minimum depth needed for the kept layers."
        ),
    )
    parser.add_argument("--phases", choices=("ctx", "gen", "both"), default="both")
    parser.add_argument("--ctx-new-tokens", default=",".join(map(str, CTX_NEW_TOKENS)))
    parser.add_argument("--ctx-past-kv", default=",".join(map(str, CTX_PAST_KV)))
    parser.add_argument(
        "--ctx-driver",
        choices=("chunked", "prefix_cache"),
        default="prefix_cache",
        help=(
            "Context measurement driver. chunked marks later chunks in one long prefill and "
            "requires past_kv to be a max_num_batched_tokens boundary; prefix_cache replays "
            "a cached prefix so arbitrary past_kv grid points can be requested."
        ),
    )
    parser.add_argument(
        "--no-filter-model-max-len",
        action="store_true",
        help="Do not skip datapoints that exceed the model config's max sequence length.",
    )
    parser.add_argument("--gen-batch-sizes", default=",".join(map(str, GEN_BATCH_SIZES)))
    parser.add_argument("--gen-past-kv", default=",".join(map(str, GEN_PAST_KV)))
    parser.add_argument("--gemm-quant", default="bf16")
    parser.add_argument("--moe-quant", default="bf16")
    parser.add_argument("--attn-quant", default="bf16")
    parser.add_argument("--kv-quant", default="bf16")
    parser.add_argument(
        "--measurement-mode",
        choices=("deployment-parity", "attribution"),
        default="deployment-parity",
        help=(
            "deployment-parity omits vLLM compile/CUDA graph overrides and is comparable to default FPM; "
            "attribution forces the old compile-off/module-NVTX path and is for decomposition only."
        ),
    )
    parser.add_argument(
        "--rollup",
        default=None,
        help=(
            "Module rollup regex. Defaults to CUDAGraphWrapper in deployment-parity mode "
            "and layer module groups in attribution mode."
        ),
    )
    parser.add_argument("--rank-reduce", choices=("sum", "max"), default="sum")
    parser.add_argument(
        "--latency-source",
        choices=("span", "gpu", "gpu_capped"),
        default="span",
        help=(
            "Write latency_ms from attributed kernel wall span, summed GPU time, or GPU time capped by span. "
            "Default matches decode span collection."
        ),
    )
    parser.add_argument(
        "--min-max-num-batched-tokens",
        type=int,
        default=1,
        help=(
            "Floor for vLLM --max-num-batched-tokens; useful for context-only grids that need "
            "FlashInfer warmup headroom."
        ),
    )
    parser.add_argument(
        "--ctx-warmup-runs",
        type=int,
        default=0,
        help="Unmarked context runs to execute per datapoint before measurement.",
    )
    parser.add_argument(
        "--ctx-measured-runs",
        type=int,
        default=6,
        help="Marked context runs to execute per datapoint and aggregate.",
    )
    parser.add_argument(
        "--ctx-repeat-aggregation",
        choices=("median", "mean", "trimmed_mean", "min"),
        default="trimmed_mean",
        help="Aggregation for repeated context measurements.",
    )
    parser.add_argument(
        "--gen-warmup-runs",
        type=int,
        default=0,
        help="Unmarked generation runs to execute per batch before measurement.",
    )
    parser.add_argument(
        "--gen-driver",
        choices=("decode_sweep", "prefill", "prefix_cache"),
        default="prefix_cache",
        help=(
            "Generation driver. decode_sweep reaches past_kv by generating from a 1-token prompt; "
            "prefill measures the first decode after a prompt of length past_kv; prefix_cache "
            "initializes KV through vLLM prefix caching and then measures the first cached decode."
        ),
    )
    parser.add_argument(
        "--gen-measured-runs",
        type=int,
        default=6,
        help="Marked generation runs to execute per batch and aggregate.",
    )
    parser.add_argument(
        "--gen-repeat-aggregation",
        choices=("median", "mean", "trimmed_mean", "min"),
        default="trimmed_mean",
        help="Aggregation for repeated generation measurements.",
    )
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--gpus", default=None, help="Comma-separated physical GPU IDs. Defaults to visible GPUs.")
    parser.add_argument("--max-workers", type=int, default=None, help="Limit concurrent one-GPU workers.")
    parser.add_argument(
        "--nsys-capture",
        choices=("cuda_profiler_api", "full"),
        default="cuda_profiler_api",
        help=(
            "Nsight Systems capture mode. cuda_profiler_api records only the worker measurement "
            "region using cudaProfilerStart/Stop; full traces the entire worker process."
        ),
    )
    parser.add_argument("--no-restrict-cudagraph-sizes", action="store_true")
    parser.add_argument(
        "--no-filter-target-layers",
        action="store_true",
        help=(
            "Do not filter parsed rows by target layer index. Useful for default-compile runs "
            "where layer modules collapse into CUDAGraphWrapper."
        ),
    )
    parser.add_argument(
        "--compilation-config-json",
        default=None,
        help=(
            "Override the collector's vLLM compilation config JSON. Use 'default' to omit "
            "the flag and request vLLM's default compilation behavior; '{}' is equivalent "
            "for vLLM 0.20.1."
        ),
    )
    parser.add_argument("--extra-vllm-arg", action="append", default=[])
    parser.add_argument("--extra-vllm-args", default="")
    return parser


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        worker_parser = argparse.ArgumentParser(description="Internal vLLM layerwise worker")
        worker_parser.add_argument("worker")
        worker_parser.add_argument("--spec", required=True)
        worker_args = worker_parser.parse_args()
        run_worker(Path(worker_args.spec))
        return

    parser = _build_arg_parser()
    args = parser.parse_args()
    work_units = build_work_units(args)
    Scheduler(args, work_units).run()


if __name__ == "__main__":
    main()
