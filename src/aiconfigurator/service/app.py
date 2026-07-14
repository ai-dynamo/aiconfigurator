# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import logging
import os
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import orjson
import yaml
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse

from aiconfigurator import __version__
from aiconfigurator.cli.api import CLIResult, cli_default
from aiconfigurator.generator.api import generate_backend_artifacts, prepare_generator_params
from aiconfigurator.generator.module_bridge import task_config_to_generator_config
from aiconfigurator.sdk.common import get_default_models
from aiconfigurator.sdk.perf_database import (
    get_database,
    get_supported_databases,
    get_systems_paths,
    set_systems_paths,
)
from aiconfigurator.sdk.task import TaskConfig
from aiconfigurator.sdk.utils import HuggingFaceDownloadError

from .models import (
    AggregateArtifactRequest,
    ApplyDeploymentRequest,
    ArtifactRequest,
    DeleteDeploymentRequest,
    DGDPreviewRequest,
    EstimateRequest,
)

logger = logging.getLogger(__name__)
_STATIC_DIR = Path(__file__).resolve().parent / "static"
_DEFAULT_RUN_STORE = Path("/tmp/aiconfigurator-service/runs.json")
_DEFAULT_MODEL_CONFIGS_ROOT = Path("/data/model-configs")
_SYSTEMS_PATHS_LOCK = threading.Lock()
_RUN_STORE_LOCK = threading.Lock()
_TRANSIENT_RUNS: dict[str, Any] = {}


class APIError(Exception):
    def __init__(self, status_code: int, code: str, message: str, details: dict[str, Any] | None = None):
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details or {}
        super().__init__(message)


def create_app() -> FastAPI:
    app = FastAPI(title="AIConfigurator Service", version=__version__)

    @app.middleware("http")
    async def add_trace_id(request: Request, call_next):
        trace_id = request.headers.get("x-trace-id") or str(uuid.uuid4())
        request.state.trace_id = trace_id
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("Unhandled error", extra={"trace_id": trace_id, "path": request.url.path})
            raise
        response.headers["x-trace-id"] = trace_id
        return response

    @app.exception_handler(APIError)
    async def api_error_handler(request: Request, exc: APIError):
        return _error_response(request, exc.status_code, exc.code, exc.message, exc.details)

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        return _error_response(
            request,
            400,
            "invalid_request",
            "Request validation failed.",
            {"errors": exc.errors()},
        )

    @app.exception_handler(HTTPException)
    async def http_error_handler(request: Request, exc: HTTPException):
        detail = exc.detail if isinstance(exc.detail, dict) else {"detail": exc.detail}
        return _error_response(request, exc.status_code, "http_error", "HTTP error.", detail)

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "version": __version__}

    @app.get("/demo")
    def demo_page():
        return FileResponse(_STATIC_DIR / "index.html")

    @app.get("/readyz")
    def readyz():
        supported = get_supported_databases()
        if not supported:
            raise APIError(500, "not_ready", "No supported databases are available.")
        return {"status": "ready", "systems": len(supported)}

    @app.get("/api/v1/options")
    def options(
        system: str | None = Query(default=None),
        backend: str | None = Query(default=None),
        version: str | None = Query(default=None),
    ):
        supported = get_supported_databases()
        payload: dict[str, Any] = {
            "models": sorted(get_default_models()),
            "local_model_configs": _list_local_model_configs(),
            "supported_databases": supported,
            "systems": sorted(supported.keys()),
            "backends": sorted({backend_name for item in supported.values() for backend_name in item}),
        }
        if system and backend and version:
            database = get_database(system, backend, version)
            if database is None:
                raise APIError(
                    422,
                    "unsupported_backend_version",
                    "No database found for the requested system/backend/version.",
                    {"system": system, "backend": backend, "version": version},
                )
            payload["quant_modes"] = copy.deepcopy(database.supported_quant_mode)
        return payload

    @app.post("/api/v1/estimate")
    def estimate(request: EstimateRequest):
        result = _run_estimate(request)
        return _serialize_cli_result(result, request)

    @app.post("/api/v1/generate-k8s-deployment")
    def generate_k8s_deployment(request: ArtifactRequest):
        if request.estimate_request is not None and request.direct_generator_params is None:
            payload = _generate_top1_deployments(request)
            _attach_and_store_jid(payload, request)
            return payload
        params, estimate_result = _resolve_generator_params(request)
        artifacts = _render_artifacts(request, params)
        payload = {
            "backend": request.backend,
            "backend_version": request.backend_version,
            "artifact_name": "k8s_deploy.yaml",
            "content": _require_artifact(artifacts, "k8s_deploy.yaml"),
            "extras": _pick_artifacts(artifacts, {"run.sh"}),
            "generator_params": params,
        }
        if estimate_result is not None:
            payload["estimate"] = _serialize_cli_result(estimate_result, request.estimate_request)
        _attach_and_store_jid(payload, request)
        return payload

    @app.post("/api/v1/generate-k8s-benchmark")
    def generate_k8s_benchmark(request: ArtifactRequest):
        params, estimate_result = _resolve_generator_params(request)
        artifacts = _render_artifacts(request, params)
        payload = {
            "backend": request.backend,
            "backend_version": request.backend_version,
            "artifact_name": "k8s_bench.yaml",
            "content": _require_artifact(artifacts, "k8s_bench.yaml"),
            "extras": _pick_artifacts(artifacts, {"bench_run.sh"}),
            "generator_params": params,
        }
        if estimate_result is not None:
            payload["estimate"] = _serialize_cli_result(estimate_result, request.estimate_request)
        return payload

    @app.post("/api/v1/generate-artifacts")
    def generate_artifacts(request: AggregateArtifactRequest):
        params, estimate_result = _resolve_generator_params(request)
        artifacts = _render_artifacts(request, params)
        output: dict[str, Any] = {
            "backend": request.backend,
            "backend_version": request.backend_version,
            "generator_params": params,
            "artifacts": {},
        }
        if "deployment" in request.artifact_types:
            output["artifacts"]["deployment"] = {
                "artifact_name": "k8s_deploy.yaml",
                "content": _require_artifact(artifacts, "k8s_deploy.yaml"),
                "extras": _pick_artifacts(artifacts, {"run.sh"}),
            }
        if "benchmark" in request.artifact_types:
            output["artifacts"]["benchmark"] = {
                "artifact_name": "k8s_bench.yaml",
                "content": _require_artifact(artifacts, "k8s_bench.yaml"),
                "extras": _pick_artifacts(artifacts, {"bench_run.sh"}),
            }
        if estimate_result is not None:
            output["estimate"] = _serialize_cli_result(estimate_result, request.estimate_request)
        return output

    @app.post("/api/v1/apply-k8s-deployment")
    def apply_k8s_deployment(request: ApplyDeploymentRequest):
        return _apply_k8s_deployment(request)

    @app.post("/api/v1/delete-k8s-deployment")
    def delete_k8s_deployment(request: DeleteDeploymentRequest):
        return _delete_k8s_deployment(request)

    @app.get("/api/v1/runs/{jid}")
    def get_run(jid: str):
        record = _load_run_record(jid)
        if record is None:
            raise APIError(404, "run_not_found", "No run record found for the requested jid.", {"jid": jid})
        return record

    @app.get("/api/v1/deployment-records")
    def deployment_records():
        records = []
        for jid, record in _load_run_records().items():
            payload = record.get("payload") or {}
            deployed = payload.get("deployed_dgd") or {}
            if not deployed:
                continue
            request = record.get("request") or {}
            estimate_request = request.get("estimate_request") or {}
            generator_overrides = request.get("generator_overrides") or {}
            service_config = generator_overrides.get("service_config") or {}
            k8s_config = generator_overrides.get("k8s_config") or {}
            records.append(
                {
                    "jid": jid,
                    "created_at": record.get("created_at"),
                    "model_path": estimate_request.get("model_path")
                    or service_config.get("model_path")
                    or (request.get("direct_generator_params") or {}).get("ServiceConfig", {}).get("model_path"),
                    "served_model_name": service_config.get("served_model_name"),
                    "backend": request.get("backend") or payload.get("backend"),
                    "backend_version": request.get("backend_version") or payload.get("backend_version"),
                    "system": estimate_request.get("system"),
                    "namespace": k8s_config.get("k8s_namespace"),
                    "deployment_name": k8s_config.get("name"),
                    "recommended_mode": payload.get("recommended_mode"),
                    "deployed_dgd_id": deployed.get("dgd_id"),
                    "deployed_mode": deployed.get("mode"),
                    "deployed_row_index": deployed.get("row_index"),
                    "deployed_at": deployed.get("deployed_at"),
                    "has_deployment": bool(deployed),
                }
            )
        records.sort(key=lambda item: item.get("deployed_at") or item.get("created_at") or "", reverse=True)
        return {"records": records}

    @app.post("/api/v1/runs/{jid}/dgd")
    def preview_dgd(jid: str, request: DGDPreviewRequest):
        return _preview_dgd_from_run(jid, request)

    return app


def _list_local_model_configs() -> list[dict[str, str]]:
    """List mounted model config directories that contain config.json."""
    root = Path(os.environ.get("AICONFIGURATOR_MODEL_CONFIGS_ROOT", str(_DEFAULT_MODEL_CONFIGS_ROOT)))
    if not root.is_dir():
        return []
    configs: list[dict[str, str]] = []
    for child in sorted(root.iterdir(), key=lambda item: item.name.lower()):
        if child.is_dir() and (child / "config.json").is_file():
            configs.append({"label": child.name, "value": str(child)})
    return configs


def _error_response(
    request: Request,
    status_code: int,
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> JSONResponse:
    trace_id = getattr(request.state, "trace_id", None)
    body = {"code": code, "message": message, "details": details or {}, "trace_id": trace_id}
    return JSONResponse(
        status_code=status_code,
        content=orjson.loads(orjson.dumps(body)),
    )


def _run_estimate(request: EstimateRequest) -> CLIResult:
    try:
        if request.systems_paths:
            with _SYSTEMS_PATHS_LOCK:
                previous_systems_paths = get_systems_paths()
                try:
                    set_systems_paths(request.systems_paths)
                    _validate_estimate_request(request)
                    return _run_cli_default(request)
                finally:
                    set_systems_paths(previous_systems_paths)
        _validate_estimate_request(request)
        return _run_cli_default(request)
    except APIError:
        raise
    except RuntimeError as exc:
        raise _map_runtime_error(exc) from exc
    except SystemExit as exc:
        raise _map_system_exit(exc, request) from exc
    except HuggingFaceDownloadError as exc:
        raise APIError(400, "invalid_model_path", str(exc)) from exc
    except ValueError as exc:
        raise APIError(400, "invalid_request", str(exc)) from exc


def _run_cli_default(request: EstimateRequest) -> CLIResult:
    return cli_default(
        model_path=request.model_path,
        total_gpus=request.total_gpus,
        system=request.system,
        decode_system=request.decode_system,
        backend=request.backend,
        backend_version=request.backend_version,
        database_mode=request.database_mode,
        isl=request.isl,
        osl=request.osl,
        ttft=request.ttft,
        tpot=request.tpot,
        request_latency=request.request_latency,
        prefix=request.prefix,
        strict_sla=request.strict_sla,
        free_gpu_memory_fraction=request.free_gpu_memory_fraction,
        max_seq_len=request.max_seq_len,
        top_n=request.top_n,
    )


def _validate_estimate_request(request: EstimateRequest) -> None:
    if request.backend == "auto":
        return

    _require_database(request.system, request.backend, request.backend_version, role="worker")
    if request.decode_system is not None and request.decode_system != request.system:
        _require_database(request.decode_system, request.backend, request.backend_version, role="decode_worker")


def _require_database(system: str, backend: str, version: str | None, *, role: str) -> None:
    database = get_database(system=system, backend=backend, version=version)
    if database is None:
        raise APIError(
            422,
            "unsupported_backend_version",
            "No usable database found for the requested system/backend/version.",
            {"system": system, "backend": backend, "version": version, "role": role},
        )


def _serialize_cli_result(result: CLIResult, request: EstimateRequest | None) -> dict[str, Any]:
    comparison = _build_mode_comparison(result)
    return {
        "normalized_input": request.model_dump(exclude_none=True) if request is not None else None,
        "chosen_exp": result.chosen_exp,
        "recommended_mode": comparison["recommended_mode"],
        "mode_comparison": comparison["modes"],
        "best_throughputs": result.best_throughputs,
        "best_latencies": result.best_latencies,
        "best_configs": {name: _df_to_records(df) for name, df in result.best_configs.items()},
        "pareto_fronts": {name: _df_to_records(df) for name, df in result.pareto_fronts.items()},
    }


def _generate_top1_deployments(request: ArtifactRequest) -> dict[str, Any]:
    assert request.estimate_request is not None
    estimate_result = _run_estimate(request.estimate_request)
    artifacts_by_mode: dict[str, Any] = {}
    for mode in _iter_available_modes(estimate_result):
        params = _generator_params_from_estimate_result(estimate_result, request, mode, 0)
        artifacts = _render_artifacts(request, params)
        artifacts_by_mode[mode] = {
            "artifact_name": "k8s_deploy.yaml",
            "content": _require_artifact(artifacts, "k8s_deploy.yaml"),
            "extras": _pick_artifacts(artifacts, {"run.sh"}),
            "generator_params": params,
        }

    comparison = _build_mode_comparison(estimate_result)
    recommended_mode = comparison["recommended_mode"]
    estimate_payload = _serialize_cli_result(estimate_result, request.estimate_request)
    payload: dict[str, Any] = {
        "backend": request.backend,
        "backend_version": request.backend_version,
        "recommended_mode": recommended_mode,
        "mode_comparison": comparison["modes"],
        "candidates_by_mode": _build_candidates_by_mode(estimate_payload),
        "artifacts_by_mode": artifacts_by_mode,
        "estimate": estimate_payload,
    }
    if recommended_mode and recommended_mode in artifacts_by_mode:
        recommended_artifact = artifacts_by_mode[recommended_mode]
        payload["artifact_name"] = recommended_artifact["artifact_name"]
        payload["content"] = recommended_artifact["content"]
        payload["extras"] = recommended_artifact["extras"]
        payload["generator_params"] = recommended_artifact["generator_params"]
    return payload


def _attach_and_store_jid(payload: dict[str, Any], request: ArtifactRequest) -> None:
    jid = _new_jid()
    payload["jid"] = jid
    payload["run_store"] = str(_run_store_path())
    _attach_dgd_ids(payload, jid)
    _cache_transient_run(jid, request, payload)


def _attach_dgd_ids(payload: dict[str, Any], jid: str) -> None:
    for mode, candidates in (payload.get("candidates_by_mode") or {}).items():
        for candidate in candidates:
            row_index = candidate.get("row_index", 0)
            candidate["dgd_id"] = _dgd_id(jid, mode, row_index)
    for mode, artifact in (payload.get("artifacts_by_mode") or {}).items():
        artifact["mode"] = mode
        artifact["row_index"] = 0
        artifact["dgd_id"] = _dgd_id(jid, mode, 0)
    if "content" in payload and "dgd_id" not in payload:
        mode = payload.get("recommended_mode") or payload.get("selected_mode") or "direct"
        payload["mode"] = mode
        payload["row_index"] = payload.get("row_index", 0)
        payload["dgd_id"] = _dgd_id(jid, mode, payload["row_index"])


def _dgd_id(jid: str, mode: str, row_index: int) -> str:
    return f"{jid}:{mode}:{row_index}"


def _build_candidates_by_mode(estimate_payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    candidates: dict[str, list[dict[str, Any]]] = {}
    for mode, rows in (estimate_payload.get("best_configs") or {}).items():
        candidates[mode] = [
            {
                "mode": mode,
                "row_index": index,
                "summary": row,
            }
            for index, row in enumerate(rows)
        ]
    return candidates


def _new_jid() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"aic-{timestamp}-{uuid.uuid4().hex[:8]}"


def _run_store_path() -> Path:
    configured = os.environ.get("AIC_SERVICE_RUN_STORE")
    return Path(configured) if configured else _DEFAULT_RUN_STORE


def _load_run_records() -> dict[str, Any]:
    path = _run_store_path()
    with _RUN_STORE_LOCK:
        if not path.exists():
            return {}
        try:
            data = orjson.loads(path.read_bytes())
        except orjson.JSONDecodeError as exc:
            raise APIError(
                500,
                "run_store_corrupt",
                "Run store JSON could not be parsed.",
                {"path": str(path), "error": str(exc)},
            ) from exc
        if not isinstance(data, dict):
            raise APIError(500, "run_store_corrupt", "Run store JSON must be an object.", {"path": str(path)})
        return data


def _load_run_record(jid: str) -> dict[str, Any] | None:
    return _load_run_records().get(jid) or _TRANSIENT_RUNS.get(jid)


def _cache_transient_run(jid: str, request: ArtifactRequest, payload: dict[str, Any]) -> None:
    _TRANSIENT_RUNS[jid] = {
        "jid": jid,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "request": _normalize_obj(request.model_dump(exclude_none=True)),
        "payload": _normalize_obj(payload),
    }


def _save_run_record(jid: str, request: ArtifactRequest, payload: dict[str, Any]) -> None:
    record = {
        "jid": jid,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "request": _normalize_obj(request.model_dump(exclude_none=True)),
        "payload": _normalize_obj(payload),
    }
    _save_record(jid, record)


def _save_record(jid: str, record: dict[str, Any]) -> None:
    path = _run_store_path()
    with _RUN_STORE_LOCK:
        records: dict[str, Any] = {}
        if path.exists():
            try:
                loaded = orjson.loads(path.read_bytes())
            except orjson.JSONDecodeError as exc:
                raise APIError(
                    500,
                    "run_store_corrupt",
                    "Run store JSON could not be parsed.",
                    {"path": str(path), "error": str(exc)},
                ) from exc
            if isinstance(loaded, dict):
                records = loaded
        records[jid] = record
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(orjson.dumps(records, option=orjson.OPT_INDENT_2, default=str))


def _preview_dgd_from_run(jid: str, preview: DGDPreviewRequest) -> dict[str, Any]:
    record = _load_run_record(jid)
    if record is None:
        raise APIError(404, "run_not_found", "No run record found for the requested jid.", {"jid": jid})

    dgd_id = _dgd_id(jid, preview.mode, preview.row_index)
    cached = ((record.get("payload") or {}).get("dgds") or {}).get(dgd_id)
    if cached is not None:
        return cached

    artifact_request = ArtifactRequest(**record["request"])
    if artifact_request.estimate_request is None:
        if preview.row_index != 0:
            raise APIError(
                422,
                "invalid_row_index",
                "Direct generator runs only have row_index=0.",
                {"jid": jid, "mode": preview.mode, "row_index": preview.row_index},
            )
        payload = record.get("payload") or {}
        if "content" not in payload:
            raise APIError(409, "missing_artifact", "The stored run has no deployment artifact.", {"jid": jid})
        return {
            "jid": jid,
            "dgd_id": dgd_id,
            "mode": preview.mode,
            "row_index": 0,
            "artifact_name": payload.get("artifact_name", "k8s_deploy.yaml"),
            "content": payload["content"],
            "extras": payload.get("extras", {}),
            "generator_params": payload.get("generator_params", {}),
        }

    selected_request = artifact_request.model_copy(
        update={"selected_mode": preview.mode, "row_index": preview.row_index}
    )
    params, _ = _resolve_generator_params(selected_request)
    artifacts = _render_artifacts(selected_request, params)
    output = {
        "jid": jid,
        "dgd_id": dgd_id,
        "mode": preview.mode,
        "row_index": preview.row_index,
        "artifact_name": "k8s_deploy.yaml",
        "content": _require_artifact(artifacts, "k8s_deploy.yaml"),
        "extras": _pick_artifacts(artifacts, {"run.sh"}),
        "generator_params": params,
    }
    _cache_dgd(jid, dgd_id, output)
    return output


def _cache_dgd(jid: str, dgd_id: str, artifact: dict[str, Any]) -> None:
    path = _run_store_path()
    with _RUN_STORE_LOCK:
        records: dict[str, Any] = {}
        if path.exists():
            loaded = orjson.loads(path.read_bytes())
            if isinstance(loaded, dict):
                records = loaded
        record = records.get(jid)
        if record is None:
            record = _TRANSIENT_RUNS.get(jid)
            if record is not None:
                payload = record.setdefault("payload", {})
                payload.setdefault("dgds", {})[dgd_id] = _normalize_obj(artifact)
            return
        payload = record.setdefault("payload", {})
        payload.setdefault("dgds", {})[dgd_id] = _normalize_obj(artifact)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(orjson.dumps(records, option=orjson.OPT_INDENT_2, default=str))


def _mark_deployed_dgd(request: ApplyDeploymentRequest, apply_payload: dict[str, Any]) -> None:
    if not request.jid or not request.dgd_id:
        return
    record = _load_run_record(request.jid)
    if record is None:
        return

    payload = record.setdefault("payload", {})
    deployed_at = datetime.now(timezone.utc).isoformat()
    deployed_dgd = {
        "dgd_id": request.dgd_id,
        "mode": request.mode,
        "row_index": request.row_index,
        "deployed_at": deployed_at,
        "apply_result": _normalize_obj(apply_payload),
    }
    payload["deployed_dgd"] = deployed_dgd
    deployed_artifact = _deployed_artifact_from_record(record, request)
    payload["dgds"] = {request.dgd_id: deployed_artifact}
    if request.mode:
        payload["artifacts_by_mode"] = {request.mode: deployed_artifact}
    payload["artifact_name"] = deployed_artifact.get("artifact_name", "k8s_deploy.yaml")
    payload["content"] = request.content
    payload["mode"] = request.mode
    payload["row_index"] = request.row_index
    payload["dgd_id"] = request.dgd_id

    record["deployed_at"] = deployed_at
    _save_record(request.jid, _normalize_obj(record))
    _TRANSIENT_RUNS.pop(request.jid, None)


def _mark_deleted_dgd(request: DeleteDeploymentRequest, delete_payload: dict[str, Any]) -> None:
    if not request.jid:
        return
    record = _load_run_record(request.jid)
    if record is None:
        return

    payload = record.setdefault("payload", {})
    deleted_at = datetime.now(timezone.utc).isoformat()
    payload["deleted_dgd"] = {
        "dgd_id": request.dgd_id,
        "mode": request.mode,
        "row_index": request.row_index,
        "deleted_at": deleted_at,
        "delete_result": _normalize_obj(delete_payload),
    }
    payload.pop("deployed_dgd", None)
    record["deleted_at"] = deleted_at
    record.pop("deployed_at", None)
    _save_record(request.jid, _normalize_obj(record))


def _deployed_artifact_from_record(record: dict[str, Any], request: ApplyDeploymentRequest) -> dict[str, Any]:
    payload = record.get("payload") or {}
    cached = ((payload.get("dgds") or {}).get(request.dgd_id)) if request.dgd_id else None
    if cached is not None:
        artifact = copy.deepcopy(cached)
    else:
        artifact = copy.deepcopy(((payload.get("artifacts_by_mode") or {}).get(request.mode or "")) or {})
    artifact.update(
        {
            "jid": request.jid,
            "dgd_id": request.dgd_id,
            "mode": request.mode,
            "row_index": request.row_index,
            "artifact_name": artifact.get("artifact_name", "k8s_deploy.yaml"),
            "content": request.content,
        }
    )
    return _normalize_obj(artifact)


def _resolve_generator_params(request: ArtifactRequest) -> tuple[dict[str, Any], CLIResult | None]:
    overrides = _build_generator_overrides(request)
    if request.estimate_request is not None:
        estimate_result = _run_estimate(request.estimate_request)
        params = _generator_params_from_estimate_result(
            estimate_result, request, request.selected_mode, request.row_index, overrides
        )
        return params, estimate_result

    if request.direct_generator_params is None:
        raise APIError(
            400,
            "missing_generator_input",
            "Provide either estimate_request or direct_generator_params.",
        )
    try:
        params = prepare_generator_params(
            None,
            overrides=_deep_merge_dicts(request.direct_generator_params, overrides),
            backend=request.backend,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise APIError(400, "invalid_generator_input", str(exc)) from exc
    return params, None


def _render_artifacts(request: ArtifactRequest, params: dict[str, Any]) -> dict[str, str]:
    try:
        return generate_backend_artifacts(
            params=params,
            backend=request.backend,
            backend_version=request.backend_version,
            deployment_target=request.deployment_target,
        )
    except ValueError as exc:
        raise APIError(400, "invalid_generator_input", str(exc)) from exc
    except RuntimeError as exc:
        raise APIError(500, "artifact_generation_failed", str(exc)) from exc


def _apply_k8s_deployment(request: ApplyDeploymentRequest) -> dict[str, Any]:
    if shutil.which("kubectl") is None:
        raise APIError(
            503,
            "kubectl_not_available",
            "kubectl is not available in the service runtime.",
            {"hint": "Install kubectl in the AIC service image and grant its ServiceAccount apply/get permissions."},
        )

    resources = _parse_k8s_resources(request.content, request.namespace)
    apply_result = _run_kubectl(
        ["kubectl", "apply", "-f", "-"],
        stdin=request.content,
        timeout=request.timeout_seconds,
    )
    health = _check_k8s_health(resources, request.timeout_seconds)
    payload = {
        "applied": apply_result.returncode == 0,
        "stdout": apply_result.stdout,
        "stderr": apply_result.stderr,
        "resources": resources,
        "health": health,
        "healthy": all(item.get("exists") for item in health),
    }
    _mark_deployed_dgd(request, payload)
    return payload


def _delete_k8s_deployment(request: DeleteDeploymentRequest) -> dict[str, Any]:
    if shutil.which("kubectl") is None:
        raise APIError(
            503,
            "kubectl_not_available",
            "kubectl is not available in the service runtime.",
            {"hint": "Install kubectl in the AIC service image and grant its ServiceAccount delete/get permissions."},
        )

    resources = _parse_k8s_resources(request.content, request.namespace)
    delete_result = _run_kubectl(
        ["kubectl", "delete", "-f", "-", "--ignore-not-found=true"],
        stdin=request.content,
        timeout=request.timeout_seconds,
    )
    payload = {
        "deleted": delete_result.returncode == 0,
        "stdout": delete_result.stdout,
        "stderr": delete_result.stderr,
        "resources": resources,
    }
    _mark_deleted_dgd(request, payload)
    return payload


def _parse_k8s_resources(content: str, namespace_override: str | None = None) -> list[dict[str, str]]:
    try:
        docs = [doc for doc in yaml.safe_load_all(content) if doc]
    except yaml.YAMLError as exc:
        raise APIError(400, "invalid_yaml", f"Deployment YAML could not be parsed: {exc}") from exc

    resources: list[dict[str, str]] = []
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        metadata = doc.get("metadata") or {}
        kind = str(doc.get("kind") or "").strip()
        name = str(metadata.get("name") or "").strip()
        namespace = namespace_override or metadata.get("namespace") or "default"
        if kind and name:
            resources.append({"kind": kind, "name": name, "namespace": str(namespace)})
    if not resources:
        raise APIError(400, "invalid_yaml", "Deployment YAML does not contain any named Kubernetes resources.")
    return resources


def _run_kubectl(args: list[str], stdin: str | None = None, timeout: int = 180) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            args,
            input=stdin,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise APIError(504, "kubectl_timeout", f"kubectl timed out after {timeout} seconds.") from exc
    if result.returncode != 0:
        raise APIError(
            502,
            "kubectl_failed",
            "kubectl command failed.",
            {"command": args, "stdout": result.stdout, "stderr": result.stderr},
        )
    return result


def _check_k8s_health(resources: list[dict[str, str]], timeout_seconds: int) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout_seconds
    health: list[dict[str, Any]] = []
    for resource in resources:
        remaining = max(1, int(deadline - time.monotonic()))
        health.append(_check_one_k8s_resource(resource, remaining))
    return health


def _check_one_k8s_resource(resource: dict[str, str], timeout_seconds: int) -> dict[str, Any]:
    kind = resource["kind"]
    name = resource["name"]
    namespace = resource["namespace"]
    base = {"kind": kind, "name": name, "namespace": namespace}

    if kind.lower() in {"deployment", "statefulset", "daemonset"}:
        rollout = _run_kubectl(
            [
                "kubectl",
                "rollout",
                "status",
                f"{kind}/{name}",
                "-n",
                namespace,
                f"--timeout={timeout_seconds}s",
            ],
            timeout=timeout_seconds + 5,
        )
        return {**base, "exists": True, "ready": True, "message": rollout.stdout.strip()}

    get_result = _run_kubectl(
        ["kubectl", "get", kind, name, "-n", namespace, "-o", "json"],
        timeout=min(timeout_seconds, 30),
    )
    payload = orjson.loads(get_result.stdout)
    conditions = payload.get("status", {}).get("conditions", [])
    ready_conditions = [
        condition
        for condition in conditions
        if condition.get("type") in {"Ready", "Available", "Reconciled", "Succeeded"}
    ]
    if ready_conditions:
        ready = any(str(condition.get("status")).lower() == "true" for condition in ready_conditions)
        message = "; ".join(
            str(condition.get("message") or condition.get("reason") or condition.get("type"))
            for condition in ready_conditions
        )
    else:
        ready = None
        message = "Resource exists; no standard Ready/Available condition found."
    return {**base, "exists": True, "ready": ready, "message": message, "conditions": ready_conditions}


def _select_result_row(result: CLIResult, selected_mode: str | None, row_index: int):
    task_name = selected_mode or result.chosen_exp or next(iter(result.best_configs.keys()), None)
    if task_name is None or task_name not in result.best_configs:
        raise APIError(
            422,
            "invalid_selection",
            "The requested estimate result selection is not available.",
            {"selected_mode": selected_mode, "available_modes": sorted(result.best_configs.keys())},
        )
    df = result.best_configs[task_name]
    if df.empty:
        raise APIError(
            409,
            "empty_result_set",
            "The selected estimate result has no candidate configurations.",
            {"selected_mode": task_name},
        )
    if row_index >= len(df):
        raise APIError(
            422,
            "invalid_row_index",
            "The selected row index is out of range.",
            {"selected_mode": task_name, "row_index": row_index, "num_rows": len(df)},
        )
    return task_name, df.iloc[row_index]


def _generator_params_from_estimate_result(
    estimate_result: CLIResult,
    request: ArtifactRequest,
    selected_mode: str | None,
    row_index: int,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    task_name, row = _select_result_row(estimate_result, selected_mode, row_index)
    task_config = estimate_result.task_configs.get(task_name)
    if task_config is None:
        raise APIError(
            500,
            "missing_task_config",
            "The selected estimate result has no matching task configuration.",
            {"selected_mode": task_name},
        )
    merged_overrides = overrides or _build_generator_overrides(request)
    return task_config_to_generator_config(task_config, row, generator_overrides=merged_overrides)


def _iter_available_modes(result: CLIResult) -> list[str]:
    ordered_modes = ["agg", "disagg"]
    available: list[str] = []
    for mode in ordered_modes:
        df = result.best_configs.get(mode)
        if df is not None and not df.empty:
            available.append(mode)
    for mode, df in result.best_configs.items():
        if mode not in available and df is not None and not df.empty:
            available.append(mode)
    if not available:
        raise APIError(409, "empty_result_set", "The estimate produced no candidate configurations.")
    return available


def _build_mode_comparison(result: CLIResult) -> dict[str, Any]:
    modes: dict[str, Any] = {}
    recommended_mode = None
    best_score = float("-inf")
    for mode in _iter_available_modes(result):
        df = result.best_configs.get(mode)
        assert df is not None and not df.empty
        top1 = _normalize_obj(df.iloc[0].to_dict())
        throughput = result.best_throughputs.get(mode)
        latencies = _normalize_obj(result.best_latencies.get(mode, {}))
        score = float(throughput) if throughput is not None else float("-inf")
        if score > best_score:
            best_score = score
            recommended_mode = mode
        modes[mode] = {
            "available": True,
            "candidate_count": len(df),
            "top1": top1,
            "best_throughput": throughput,
            "best_latency": latencies,
        }
    return {"recommended_mode": recommended_mode or result.chosen_exp, "modes": modes}


def _build_generator_overrides(request: ArtifactRequest) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    raw = request.generator_overrides.model_dump(exclude_none=True) if request.generator_overrides is not None else {}
    mapping = {
        "service_config": "ServiceConfig",
        "k8s_config": "K8sConfig",
        "worker_config": "WorkerConfig",
        "sla_config": "SlaConfig",
        "bench_config": "BenchConfig",
        "dyn_config": "DynConfig",
        "model_cfg": "ModelConfig",
        "params": "Params",
        "workers": "Workers",
        "generator_dynamo_version": "generator_dynamo_version",
        "rule": "rule",
    }
    for src_key, dest_key in mapping.items():
        if src_key in raw:
            overrides[dest_key] = raw[src_key]
    return overrides


def _deep_merge_dicts(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge_dicts(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _require_artifact(artifacts: dict[str, str], name: str) -> str:
    if name not in artifacts:
        raise APIError(
            500,
            "missing_artifact",
            "The requested artifact was not generated.",
            {"artifact_name": name, "available_artifacts": sorted(artifacts.keys())},
        )
    return artifacts[name]


def _pick_artifacts(artifacts: dict[str, str], names: set[str]) -> dict[str, str]:
    return {name: content for name, content in artifacts.items() if name in names}


def _df_to_records(df) -> list[dict[str, Any]]:
    if df is None:
        return []
    records = []
    for row in df.to_dict(orient="records"):
        records.append(_normalize_obj(row))
    return records


def _normalize_obj(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize_obj(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_normalize_obj(item) for item in value]
    if is_dataclass(value):
        return _normalize_obj(asdict(value))
    if isinstance(value, TaskConfig):
        return repr(value)
    return value


def _map_runtime_error(exc: RuntimeError) -> APIError:
    message = str(exc)
    lowered = message.lower()
    if "does not fit in gpu memory" in lowered or "oom" in lowered:
        return APIError(409, "oom", message)
    if "no configuration satisfied" in lowered or "no results found" in lowered:
        return APIError(409, "no_feasible_config", message)
    if "support" in lowered and "not" in lowered:
        return APIError(422, "unsupported_combo", message)
    return APIError(500, "estimate_failed", message)


def _map_system_exit(exc: SystemExit, request: EstimateRequest) -> APIError:
    exit_code = exc.code if isinstance(exc.code, int) else 1
    if exit_code in (0, None):
        return APIError(500, "estimate_failed", "Estimate terminated unexpectedly.")

    return APIError(
        422,
        "estimate_failed",
        "Estimate could not be completed for the requested configuration.",
        {
            "system": request.system,
            "decode_system": request.decode_system,
            "backend": request.backend,
            "backend_version": request.backend_version,
            "database_mode": request.database_mode,
            "exit_code": exit_code,
        },
    )
