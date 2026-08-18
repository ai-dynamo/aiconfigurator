# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AIConfigurator REST API.

Minimal service wrapping the aiconfigurator SDK for GPU recommendation,
single-point performance estimation, and memory estimation.
See docs/api/openapi.yaml for the full spec.
"""

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field, model_validator

from aiconfigurator.cli.api import cli_estimate, cli_recommend, _execute_and_wrap_result, _build_recommend_tasks
from aiconfigurator.sdk.errors import NoFeasibleConfigError
from aiconfigurator.cli.main import build_default_tasks
from aiconfigurator.sdk.common import get_default_models
from aiconfigurator.sdk.memory import estimate_kv_cache
from aiconfigurator_core.sdk.common import SupportedSystems
from aiconfigurator_core.sdk.perf_database import load_system_spec
from aiconfigurator_core.sdk.utils import get_model_config_from_model_path

logger = logging.getLogger(__name__)

# ─── Pydantic models ────────────────────────────────────────────────────────


class RecommendRequest(BaseModel):
    model_path: str = Field(examples=["Qwen/Qwen3-32B"], description="HuggingFace model path or SDK model key.")
    system: str = Field(examples=["h200_sxm"], description="GPU system identifier.")
    backend: str = Field(default="vllm", description="Inference backend.")
    backend_version: str | None = Field(default=None, examples=[None], description="Backend version.")
    target_request_rate: float | None = Field(default=None, examples=[None], description="Target req/s.")
    target_concurrency: float | None = Field(default=None, examples=[32], description="Target concurrent users.")
    isl: int = Field(default=4000, description="Input sequence length.")
    osl: int = Field(default=1000, description="Output sequence length.")
    ttft: float = Field(default=2000.0, description="TTFT target (ms).")
    tpot: float = Field(default=30.0, description="TPOT target (ms).")
    request_latency: float | None = Field(default=None, description="E2E latency target (ms).")
    prefix: int = Field(default=0, description="Prefix cache length.")
    database_mode: str = Field(default="HYBRID", description="Perf database mode.")
    top_n: int = Field(default=5, ge=1, le=20, examples=[2], description="Number of configs to return.")
    model_config_data: dict | None = Field(
        default=None,
        alias="model_config",
        description="Pre-fetched HuggingFace model config.json. Skips HF download.",
    )

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def exactly_one_load_target(self):
        has_rate = self.target_request_rate is not None
        has_conc = self.target_concurrency is not None
        if has_rate == has_conc:
            raise ValueError("Exactly one of target_request_rate or target_concurrency must be provided.")
        return self


class EstimateRequest(BaseModel):
    model_path: str = Field(examples=["Qwen/Qwen3-32B"], description="HuggingFace model path or SDK model key.")
    system: str = Field(examples=["h200_sxm"], description="GPU system identifier.")
    backend: str = Field(default="vllm", description="Inference backend.")
    backend_version: str | None = Field(default=None, examples=[None], description="Backend version.")
    isl: int = Field(default=4000, description="Input sequence length.")
    osl: int = Field(default=1000, description="Output sequence length.")
    tp_size: int = Field(default=1, description="Tensor parallel size.")
    pp_size: int = Field(default=1, description="Pipeline parallel size.")
    batch_size: int = Field(default=128, description="Batch size (max concurrent requests).")
    database_mode: str = Field(default="HYBRID", description="Perf database mode.")
    gemm_quant_mode: str | None = Field(default=None)
    kvcache_quant_mode: str | None = Field(default=None)
    fmha_quant_mode: str | None = Field(default=None)
    moe_tp_size: int | None = Field(default=None)
    moe_ep_size: int | None = Field(default=None)
    attention_dp_size: int = Field(default=1)
    inclusive_tpot: bool = Field(
        default=False,
        description="Report TPOT as (ttft + tpot * (osl - 1)) / osl, spreading TTFT across all output tokens. "
        "Useful for comparing with benchmarks that report inclusive TPOT (e.g. GuideLLM).",
    )
    model_config_data: dict | None = Field(
        default=None,
        alias="model_config",
        description="Pre-fetched HuggingFace model config.json. Skips HF download.",
    )

    model_config = {"populate_by_name": True}


class MemoryBreakdown(BaseModel):
    weights_bytes: int = 0
    activations_bytes: int = 0
    runtime_overhead_bytes: int = 0
    comm_overhead_bytes: int = 0
    kv_cache_bytes: int = 0


class ServingConfig(BaseModel):
    backend: str
    tensor_parallel_size: int
    max_model_len: int
    max_num_seqs: int
    gpu_memory_utilization: float
    enable_chunked_prefill: bool
    enable_prefix_caching: bool
    quantization: str


class WorkerConfig(BaseModel):
    """Parallelism and serving config for one worker role in a disagg deployment."""
    tp: int | None = None
    pp: int | None = None
    dp: int | None = None
    moe_tp: int | None = None
    moe_ep: int | None = None
    num_workers: int | None = None
    memory_gb: float | None = None
    gemm: str | None = None
    kvcache: str | None = None
    fmha: str | None = None
    moe: str | None = None
    comm: str | None = None
    backend_version: str | None = None
    memory_breakdown: MemoryBreakdown | None = None


class RecommendConfig(BaseModel):
    total_gpus_needed: int | None = None
    replicas_needed: int | None = None
    num_total_gpus: int | None = None
    # Parallelism (agg only; None for disagg — see prefill/decode_config)
    tp: int | None = None
    pp: int | None = None
    dp: int | None = None
    moe_tp: int | None = None
    moe_ep: int | None = None
    cp: int | None = None
    ttft: float | None = None
    tpot: float | None = None
    request_latency: float | None = None
    concurrency: int | None = None
    request_rate: float | None = None
    tokens_per_second: float | None = None
    tokens_per_second_per_gpu: float | None = None
    tokens_per_second_per_user: float | None = None
    memory: float | None = None
    model: str | None = None
    system: str | None = None
    backend: str | None = None
    backend_version: str | None = None
    gemm: str | None = None
    kvcache: str | None = None
    fmha: str | None = None
    moe: str | None = None
    comm: str | None = None
    power_w: float | None = None
    # Optional detail sections (include=config / include=memory)
    serving_config: ServingConfig | None = None
    memory_breakdown: MemoryBreakdown | None = None
    # Disagg detail (present for disagg results)
    prefill_config: WorkerConfig | None = None
    decode_config: WorkerConfig | None = None


class RecommendResponse(BaseModel):
    configs: list[RecommendConfig]
    chosen_mode: str


class EstimateResponse(BaseModel):
    """Single-point performance estimate for a given parallelism configuration."""
    ttft: float
    tpot: float
    request_latency: float | None = None
    tokens_per_second: float | None = None
    tokens_per_second_per_gpu: float | None = None
    tokens_per_second_per_user: float | None = None
    memory: float | None = None
    concurrency: int | None = None
    tp: int | None = None
    pp: int | None = None
    dp: int | None = None
    system: str | None = None
    backend: str | None = None
    backend_version: str | None = None
    gemm: str | None = None
    kvcache: str | None = None
    power_w: float | None = None
    serving_config: ServingConfig | None = None
    memory_breakdown: MemoryBreakdown | None = None


class MemoryRequest(BaseModel):
    model_path: str = Field(examples=["Qwen/Qwen3-32B"], description="HuggingFace model path or SDK model key.")
    system: str = Field(examples=["h200_sxm"], description="GPU system identifier.")
    backend: str = Field(default="vllm", description="Inference backend.")
    backend_version: str | None = Field(default=None, examples=[None])
    max_num_tokens: int = Field(default=8192)
    max_batch_size: int = Field(default=128)
    memory_fraction_kind: str = Field(default="of_total")
    memory_fraction_value: float = Field(default=1.0, ge=0.0, le=1.0)
    tp_size: int = Field(default=1)
    pp_size: int = Field(default=1)
    attention_dp_size: int = Field(default=1)
    moe_tp_size: int | None = Field(default=None)
    moe_ep_size: int | None = Field(default=None)
    gemm_quant_mode: str | None = Field(default=None)
    moe_quant_mode: str | None = Field(default=None)
    kvcache_quant_mode: str | None = Field(default=None)
    fmha_quant_mode: str | None = Field(default=None)
    comm_quant_mode: str | None = Field(default=None)
    tolerance_fraction: float | None = Field(default=None)
    model_config_data: dict | None = Field(
        default=None,
        alias="model_config",
        description="Pre-fetched HuggingFace model config.json. Skips HF download.",
    )

    model_config = {"populate_by_name": True}


class MemoryResponse(BaseModel):
    total_gpu_capacity_bytes: int
    total_kv_size_bytes: int
    kv_size_per_token_bytes: int
    total_kv_size_tokens: int
    source: str
    memory_breakdown: MemoryBreakdown
    tolerance_adjusted: dict[str, Any] | None = None


class SystemDetail(BaseModel):
    id: str
    name: str
    vendor: str
    architecture: str
    memory_bytes: int
    tdp_watts: float
    gpus_per_node: int


# ─── Helpers ─────────────────────────────────────────────────────────────────

_COLUMN_MAP = {
    "tokens/s": "tokens_per_second",
    "tokens/s/gpu": "tokens_per_second_per_gpu",
    "tokens/s/user": "tokens_per_second_per_user",
    "num_total_gpus": "num_total_gpus",
    "version": "backend_version",
}

_INT_FIELDS = frozenset({
    "total_gpus_needed", "replicas_needed", "num_total_gpus",
    "tp", "pp", "dp", "moe_tp", "moe_ep", "cp", "concurrency",
})

_SM_ARCHITECTURE = {
    70: "volta",
    75: "turing",
    80: "ampere",
    86: "ampere",
    89: "ada-lovelace",
    90: "hopper",
    100: "blackwell",
    120: "blackwell",
}

_DEFAULT_GPU_MEMORY_UTILIZATION = 0.9


class _noop_context:
    def __init__(self, model_path: str):
        self._path = model_path

    def __enter__(self):
        return self._path

    def __exit__(self, *args):
        pass


class _tempdir_context:
    def __init__(self, model_path: str, config_dict: dict):
        self._original = model_path
        self._config = config_dict
        self._tmpdir: Any = None

    def __enter__(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        config_path = Path(self._tmpdir.name) / "config.json"
        config_path.write_text(json.dumps(self._config))
        return self._tmpdir.name

    def __exit__(self, *args):
        if self._tmpdir:
            self._tmpdir.cleanup()


def _with_model_config(model_path: str, config_dict: dict | None):
    if config_dict is None:
        return _noop_context(model_path)
    return _tempdir_context(model_path, config_dict)


def _coerce_int(val: Any) -> int | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _coerce_float(val: Any) -> float | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _worker_config_from_row(row: pd.Series, prefix: str, req: RecommendRequest) -> WorkerConfig | None:
    def g(col: str) -> Any:
        v = row.get(f"({prefix}){col}")
        return None if v is None or (isinstance(v, float) and pd.isna(v)) else v

    tp = _coerce_int(g("tp"))
    if tp is None:
        return None
    return WorkerConfig(
        tp=tp,
        pp=_coerce_int(g("pp")),
        dp=_coerce_int(g("dp")),
        moe_tp=_coerce_int(g("moe_tp")),
        moe_ep=_coerce_int(g("moe_ep")),
        num_workers=_coerce_int(g("workers")),
        memory_gb=_coerce_float(g("memory")),
        gemm=g("gemm"),
        kvcache=g("kvcache"),
        fmha=g("fmha"),
        moe=g("moe"),
        comm=g("comm"),
        backend_version=row.get(f"({prefix})version") or req.backend_version,
    )


def _row_to_config(row: pd.Series, req: RecommendRequest) -> RecommendConfig:
    is_disagg = "(p)tp" in row.index

    d: dict[str, Any] = {}
    for col, val in row.items():
        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue
        col_str = str(col)
        if col_str.startswith("(p)") or col_str.startswith("(d)") or col_str.startswith("(e)"):
            continue
        key = _COLUMN_MAP.get(col_str, col_str)
        if isinstance(val, float) and val == int(val) and key in _INT_FIELDS:
            val = int(val)
        d[key] = val

    d.setdefault("system", req.system)
    d.setdefault("backend", req.backend)
    d.setdefault("backend_version", req.backend_version)

    cfg = RecommendConfig.model_validate(d)

    if is_disagg:
        cfg.prefill_config = _worker_config_from_row(row, "p", req)
        cfg.decode_config = _worker_config_from_row(row, "d", req)
        if cfg.memory is None:
            p_mem = _coerce_float(row.get("(p)memory"))
            d_mem = _coerce_float(row.get("(d)memory"))
            if p_mem is not None or d_mem is not None:
                cfg.memory = (p_mem or 0.0) + (d_mem or 0.0)

    return cfg


def _build_serving_config(
    backend: str,
    tp: int,
    isl: int,
    osl: int,
    concurrency: int,
    gemm: str | None,
    prefix: int = 0,
) -> ServingConfig:
    quant_map = {"fp8": "fp8", "fp8_block": "fp8", "int8": "int8"}
    quantization = quant_map.get(gemm or "", "auto")
    return ServingConfig(
        backend=backend,
        tensor_parallel_size=tp,
        max_model_len=isl + osl,
        max_num_seqs=min(concurrency, 256),
        gpu_memory_utilization=_DEFAULT_GPU_MEMORY_UTILIZATION,
        enable_chunked_prefill=isl >= 4096 or concurrency >= 64,
        enable_prefix_caching=prefix > 0,
        quantization=quantization,
    )


def _build_memory_breakdown(
    model_path: str,
    system: str,
    backend: str,
    backend_version: str | None,
    tp: int,
    pp: int,
    isl: int,
    osl: int,
    concurrency: int,
    gemm_quant: str | None = None,
    kvcache_quant: str | None = None,
    moe_tp: int | None = None,
    moe_ep: int | None = None,
) -> MemoryBreakdown | None:
    try:
        raw = estimate_kv_cache(
            model_path=model_path,
            system=system,
            backend=backend,
            backend_version=backend_version,
            max_num_tokens=isl + osl,
            max_batch_size=concurrency,
            memory_fraction_kind="of_total",
            memory_fraction_value=_DEFAULT_GPU_MEMORY_UTILIZATION,
            tp_size=tp,
            pp_size=pp,
            moe_tp_size=moe_tp,
            moe_ep_size=moe_ep,
            gemm_quant_mode=gemm_quant if gemm_quant and gemm_quant != "half" else None,
            kvcache_quant_mode=kvcache_quant if kvcache_quant and kvcache_quant != "half" else None,
        )
    except Exception:
        logger.debug("memory breakdown unavailable for %s on %s", model_path, system)
        return None

    breakdown = raw.get("memory_breakdown") or {}
    return MemoryBreakdown(
        weights_bytes=int(breakdown.get("weights_bytes", 0)),
        activations_bytes=int(breakdown.get("activations_bytes", 0)),
        runtime_overhead_bytes=int(breakdown.get("runtime_overhead_bytes", 0)),
        comm_overhead_bytes=int(breakdown.get("comm_overhead_bytes", 0)),
        kv_cache_bytes=int(raw.get("total_kv_size_bytes", 0)),
    )


def _common_error_handler(e: Exception, op: str, model_path: str, backend: str, system: str) -> None:
    msg = str(e)
    if isinstance(e, NoFeasibleConfigError):
        raise HTTPException(status_code=422, detail=msg)
    if isinstance(e, (ValueError, AttributeError)):
        if "system_spec" in msg or "NoneType" in msg or "unsupported model" in msg.lower():
            detail = f"No performance data available for model={model_path}, backend={backend}, system={system}."
            raise HTTPException(status_code=422, detail=detail)
        raise HTTPException(status_code=422, detail=msg)
    logger.exception("%s failed", op)
    raise HTTPException(status_code=500, detail=msg)


def _architecture_from_sm(sm_version: int) -> str:
    return _SM_ARCHITECTURE.get(sm_version, f"sm_{sm_version}")


def _system_display_name(system_id: str) -> str:
    parts = system_id.replace("_", " ").upper().split()
    spec = load_system_spec(system_id)
    vendor_name = spec.get("misc", {}).get("vendor", "unknown")
    return str.title(vendor_name) + " " + " ".join(parts)


def _parse_include(include: str | None) -> set[str]:
    if not include:
        return set()
    return {s.strip().lower() for s in include.split(",")}


# ─── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AIConfigurator API",
    description="GPU recommendation, performance estimation, and memory estimation for LLM inference.",
    version="1.0.0",
    default_response_class=ORJSONResponse,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _recommend_quick(req: RecommendRequest):
    """Fast agg-only, single-node, top-1 recommendation."""
    spec = load_system_spec(req.system)
    gpus_per_node = spec["node"]["num_gpus_per_node"]

    base_tasks = build_default_tasks(
        model_path=req.model_path,
        total_gpus=gpus_per_node,
        system=req.system,
        backend=req.backend,
        backend_version=req.backend_version,
        database_mode=req.database_mode,
        isl=req.isl,
        osl=req.osl,
        ttft=req.ttft,
        tpot=req.tpot,
        request_latency=req.request_latency,
        prefix=req.prefix,
    )
    base_tasks = {k: v for k, v in base_tasks.items() if v.serving_mode == "agg"}

    tasks = _build_recommend_tasks(base_tasks, gpus_per_node)
    return _execute_and_wrap_result(
        tasks,
        mode="default",
        top_n=1,
        target_request_rate=req.target_request_rate,
        target_concurrency=req.target_concurrency,
    )


def _recommend_full(req: RecommendRequest):
    """Full recommendation across agg and disagg modes."""
    return cli_recommend(
        model_path=req.model_path,
        system=req.system,
        backend=req.backend,
        backend_version=req.backend_version,
        target_request_rate=req.target_request_rate,
        target_concurrency=req.target_concurrency,
        database_mode=req.database_mode,
        isl=req.isl,
        osl=req.osl,
        ttft=req.ttft,
        tpot=req.tpot,
        request_latency=req.request_latency,
        prefix=req.prefix,
        top_n=req.top_n,
    )


@app.post("/recommend")
def post_recommend(
    req: RecommendRequest,
    include: str | None = Query(default=None, examples=["config,memory"], description="Comma-separated extras: config, memory."),
    mode: str | None = Query(default=None, examples=["quick"], description="Search mode: 'quick' for fast agg-only single-node top-1 result, or omit for full search across agg and disagg modes."),
):
    """Find optimal GPU configuration for a workload."""
    try:
        if mode == "quick":
            result = _recommend_quick(req)
        else:
            result = _recommend_full(req)
    except NoFeasibleConfigError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except AttributeError as e:
        msg = str(e)
        if "system_spec" in msg or "NoneType" in msg:
            raise HTTPException(
                status_code=422,
                detail=f"No performance data available for model={req.model_path}, backend={req.backend}, system={req.system}.",
            )
        logger.exception("recommend failed")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("recommend failed")
        raise HTTPException(status_code=500, detail=str(e))

    best = result.best_configs.get(result.chosen_exp)
    chosen = result.chosen_exp

    if best is None or best.empty:
        raise HTTPException(status_code=422, detail="No configuration meets the specified requirements.")

    includes = _parse_include(include)
    want_config = "config" in includes
    want_memory = "memory" in includes
    is_disagg = chosen.startswith("disagg") if chosen else False

    configs = []
    with _with_model_config(req.model_path, req.model_config_data) as effective_path:
        for _, row in best.head(req.top_n).iterrows():
            cfg = _row_to_config(row, req)
            backend = cfg.backend or req.backend
            bv = cfg.backend_version or req.backend_version

            if is_disagg:
                if want_memory:
                    for worker in [cfg.prefill_config, cfg.decode_config]:
                        if worker and worker.tp:
                            worker.memory_breakdown = _build_memory_breakdown(
                                effective_path, req.system, backend,
                                bv or worker.backend_version,
                                worker.tp or 1, worker.pp or 1,
                                req.isl, req.osl, worker.num_workers or 1,
                                worker.gemm if worker.gemm and worker.gemm != "half" else None,
                                worker.kvcache if worker.kvcache and worker.kvcache != "half" else None,
                                worker.moe_tp, worker.moe_ep,
                            )
            else:
                tp = cfg.tp or 1
                concurrency = cfg.concurrency or 128
                if want_config:
                    cfg.serving_config = _build_serving_config(
                        backend, tp, req.isl, req.osl, concurrency, cfg.gemm, req.prefix,
                    )
                if want_memory:
                    cfg.memory_breakdown = _build_memory_breakdown(
                        effective_path, req.system, backend, bv,
                        tp, cfg.pp or 1, req.isl, req.osl, concurrency,
                        cfg.gemm, cfg.kvcache, cfg.moe_tp, cfg.moe_ep,
                    )
            configs.append(cfg)

    return RecommendResponse(configs=configs, chosen_mode=chosen)


@app.post("/estimate")
def post_estimate(
    req: EstimateRequest,
    include: str | None = Query(default=None, examples=["config,memory"], description="Comma-separated extras: config, memory."),
):
    """Single-point performance estimate for a given parallelism configuration.

    Given a model, GPU system, backend, and explicit parallelism settings
    (TP/PP/batch_size), returns predicted TTFT, TPOT, throughput, and memory.
    Use this when you know your deployment configuration and want to predict
    its performance. Use /recommend when you want to find the optimal config.
    """
    try:
        with _with_model_config(req.model_path, req.model_config_data) as effective_path:
            result = cli_estimate(
                effective_path,
                system_name=req.system,
                backend_name=req.backend,
                backend_version=req.backend_version,
                database_mode=req.database_mode,
                isl=req.isl,
                osl=req.osl,
                batch_size=req.batch_size,
                tp_size=req.tp_size,
                pp_size=req.pp_size,
                attention_dp_size=req.attention_dp_size,
                moe_tp_size=req.moe_tp_size,
                moe_ep_size=req.moe_ep_size,
                gemm_quant_mode=req.gemm_quant_mode,
                kvcache_quant_mode=req.kvcache_quant_mode,
                fmha_quant_mode=req.fmha_quant_mode,
            )
    except (ValueError, AttributeError, Exception) as e:
        _common_error_handler(e, "estimate", req.model_path, req.backend, req.system)

    raw = result.raw
    includes = _parse_include(include)

    tpot = result.tpot
    if req.inclusive_tpot and req.osl > 0:
        tpot = (result.ttft + tpot * (req.osl - 1)) / req.osl

    resp = EstimateResponse(
        ttft=result.ttft,
        tpot=tpot,
        request_latency=_coerce_float(raw.get("request_latency")),
        tokens_per_second=_coerce_float(raw.get("tokens/s")),
        tokens_per_second_per_gpu=_coerce_float(raw.get("tokens/s/gpu")),
        tokens_per_second_per_user=_coerce_float(raw.get("tokens/s/user")),
        memory=_coerce_float(raw.get("memory")),
        concurrency=_coerce_int(raw.get("bs")),
        tp=_coerce_int(raw.get("tp")) or req.tp_size,
        pp=_coerce_int(raw.get("pp")) or req.pp_size,
        dp=_coerce_int(raw.get("dp")),
        system=raw.get("system", req.system),
        backend=raw.get("backend", req.backend),
        backend_version=raw.get("version", req.backend_version),
        gemm=raw.get("gemm"),
        kvcache=raw.get("kvcache"),
        power_w=result.power_w,
    )

    if "config" in includes:
        resp.serving_config = _build_serving_config(
            resp.backend or req.backend, req.tp_size, req.isl, req.osl,
            req.batch_size, resp.gemm, 0,
        )

    if "memory" in includes:
        with _with_model_config(req.model_path, req.model_config_data) as effective_path:
            resp.memory_breakdown = _build_memory_breakdown(
                effective_path, req.system, req.backend, req.backend_version,
                req.tp_size, req.pp_size, req.isl, req.osl, req.batch_size,
                req.gemm_quant_mode if req.gemm_quant_mode and req.gemm_quant_mode != "half" else None,
                req.kvcache_quant_mode if req.kvcache_quant_mode and req.kvcache_quant_mode != "half" else None,
                req.moe_tp_size, req.moe_ep_size,
            )

    return resp


@app.post("/memory", response_model=MemoryResponse)
def post_memory(req: MemoryRequest):
    """Estimate GPU memory breakdown for a model configuration."""
    try:
        raw = estimate_kv_cache(
            model_path=req.model_path,
            system=req.system,
            backend=req.backend,
            backend_version=req.backend_version,
            max_num_tokens=req.max_num_tokens,
            max_batch_size=req.max_batch_size,
            memory_fraction_kind=req.memory_fraction_kind,
            memory_fraction_value=req.memory_fraction_value,
            tp_size=req.tp_size,
            pp_size=req.pp_size,
            attention_dp_size=req.attention_dp_size,
            moe_tp_size=req.moe_tp_size,
            moe_ep_size=req.moe_ep_size,
            gemm_quant_mode=req.gemm_quant_mode,
            moe_quant_mode=req.moe_quant_mode,
            kvcache_quant_mode=req.kvcache_quant_mode,
            fmha_quant_mode=req.fmha_quant_mode,
            comm_quant_mode=req.comm_quant_mode,
            tolerance_fraction=req.tolerance_fraction,
        )
    except (ValueError, AttributeError) as e:
        msg = str(e)
        if "system_spec" in msg or "NoneType" in msg or "unsupported model" in msg.lower():
            detail = (
                f"No performance data available for model={req.model_path}, "
                f"backend={req.backend}, system={req.system}"
            )
            if not req.backend_version:
                detail += ". Try specifying a backend_version (e.g. '0.24.0' for vllm)."
            raise HTTPException(status_code=422, detail=detail)
        raise HTTPException(status_code=422, detail=msg)
    except Exception as e:
        logger.exception("memory estimation failed")
        raise HTTPException(status_code=500, detail=str(e))

    breakdown = raw.get("memory_breakdown") or {}
    return MemoryResponse(
        total_gpu_capacity_bytes=raw["total_gpu_capacity_bytes"],
        total_kv_size_bytes=raw["total_kv_size_bytes"],
        kv_size_per_token_bytes=raw["kv_size_per_token_bytes"],
        total_kv_size_tokens=raw["total_kv_size_tokens"],
        source=raw.get("source", "unknown"),
        memory_breakdown=MemoryBreakdown(
            weights_bytes=int(breakdown.get("weights_bytes", 0)),
            activations_bytes=int(breakdown.get("activations_bytes", 0)),
            runtime_overhead_bytes=int(breakdown.get("runtime_overhead_bytes", 0)),
            comm_overhead_bytes=int(breakdown.get("comm_overhead_bytes", 0)),
            kv_cache_bytes=raw["total_kv_size_bytes"],
        ),
        tolerance_adjusted=raw.get("tolerance_adjusted"),
    )


@app.get("/models")
def get_models(
    detailed: bool = Query(False, description="whether to return detailed model information"),
):
    default_models = sorted(get_default_models())
    if not detailed:
        return {"models": default_models}
    else:
        return {"models": [get_model_config_from_model_path(model) for model in default_models]}


@app.get("/systems")
def get_systems(
    include: str | None = Query(default=None, examples=["specs"], description="Comma-separated extras: specs."),
):
    """List supported GPU systems."""
    includes = _parse_include(include)
    want_specs = "specs" in includes

    systems = []
    for sys_id in sorted(SupportedSystems):
        entry: dict[str, Any] = {"id": sys_id}
        try:
            spec = load_system_spec(sys_id)
            misc = spec.get("misc", {})
            entry["name"] = misc.get("name", sys_id)
            if want_specs:
                gpu = spec.get("gpu", {})
                node = spec.get("node", {})
                sm = int(gpu.get("sm_version", 0))
                entry.update({
                    "vendor": misc.get("vendor", "unknown"),
                    "architecture": _architecture_from_sm(sm),
                    "memory_bytes": int(gpu.get("mem_capacity", 0)),
                    "tdp_watts": float(gpu.get("power", 0)),
                    "gpus_per_node": int(node.get("num_gpus_per_node", 0)),
                    "memory_bandwidth_bytes": int(gpu.get("mem_bw", 0)),
                    "bf16_tflops": gpu.get("bfloat16_tc_flops", 0) / 1e12,
                })
        except Exception:
            logger.warning("failed to load spec for %s", sys_id)
            entry.setdefault("name", sys_id)
        systems.append(entry)
    return {"systems": systems}


# ─── Entrypoint ──────────────────────────────────────────────────────────────

def parse(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_name", type=str, default="127.0.0.1")
    parser.add_argument("--server_port", type=int, default=7860)
    return parser.parse_args(args=args)


if __name__ == "__main__":
    args = parse(sys.argv[1:])
    uvicorn.run(app, host=args.server_name, port=args.server_port)
