"""Shared vLLM deployment argument and metadata helpers for FPM/layerwise runs."""

from __future__ import annotations

import argparse
import dataclasses
import enum
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

CRITICAL_CONFIG_KEYS = (
    "vllm_version",
    "model_config.dtype",
    "cache_config.cache_dtype",
    "cache_config.enable_prefix_caching",
    "parallel_config.tensor_parallel_size",
    "parallel_config.pipeline_parallel_size",
    "scheduler_config.max_num_seqs",
    "scheduler_config.max_num_batched_tokens",
    "compilation_config.mode",
    "compilation_config.cudagraph_mode",
    "compilation_config.backend",
    "compilation_config.custom_ops",
    "compilation_config.pass_config",
    "attention_config.backend",
    "optimization_level",
)


@dataclasses.dataclass(frozen=True)
class VllmDeploymentConfig:
    model: str
    max_model_len: int | None = None
    max_num_seqs: int | None = None
    max_num_batched_tokens: int | None = None
    gpu_memory_utilization: float | None = None
    tensor_parallel_size: int | None = None
    pipeline_parallel_size: int | None = None
    dtype: str | None = None
    kv_cache_dtype: str | None = None
    enforce_eager: bool = False
    disable_prefix_caching: bool = False
    no_async_scheduling: bool = False
    extra_args: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class VllmRuntimeDefaults:
    kv_cache_dtype: str | None = None
    disable_prefix_caching: bool = False
    extra_args: tuple[str, ...] = ()


def is_gpt_oss_model(model: str) -> bool:
    normalized = str(model).lower().replace("_", "-")
    return "gpt-oss" in normalized


def is_blackwell_system(system: str | None) -> bool:
    if not system:
        return False
    normalized = str(system).lower().replace("_", "-")
    return any(
        marker in normalized
        for marker in (
            "blackwell",
            "b200",
            "b300",
            "gb200",
            "gb300",
            "sm100",
            "sm120",
            "compute-cap-10",
            "compute-cap=10",
            "compute-cap:10",
            "compute-cap-12",
            "compute-cap=12",
            "compute-cap:12",
        )
    )


def has_cli_flag(args: tuple[str, ...] | list[str], *flags: str) -> bool:
    for arg in args:
        for flag in flags:
            if arg == flag or arg.startswith(f"{flag}="):
                return True
    return False


def _append_default_flag(args: list[str], flag: str, *aliases: str) -> None:
    if not has_cli_flag(args, flag, *aliases):
        args.append(flag)


def _append_default_pair(args: list[str], flag: str, value: str) -> None:
    if not has_cli_flag(args, flag):
        args.extend([flag, value])


def gpt_oss_runtime_defaults(
    *,
    model: str,
    system: str | None = None,
    kv_cache_dtype: str | None = None,
    disable_prefix_caching: bool = False,
    extra_args: tuple[str, ...] | list[str] = (),
) -> VllmRuntimeDefaults:
    """Return vLLM recipe defaults that must match across FPM/layerwise.

    vLLM's GPT-OSS Blackwell recipe explicitly opts into FP8 KV cache.  vLLM
    itself does not infer that from the GPT-OSS checkpoint, so collectors must
    add the same flag when modeling Blackwell runs.
    """

    normalized_extra = list(extra_args)
    resolved_kv_cache_dtype = kv_cache_dtype
    resolved_disable_prefix_caching = disable_prefix_caching
    if not is_gpt_oss_model(model):
        return VllmRuntimeDefaults(
            kv_cache_dtype=resolved_kv_cache_dtype,
            disable_prefix_caching=resolved_disable_prefix_caching,
            extra_args=tuple(normalized_extra),
        )

    if (
        not resolved_kv_cache_dtype
        and not has_cli_flag(normalized_extra, "--kv-cache-dtype")
        and is_blackwell_system(system)
    ):
        resolved_kv_cache_dtype = "fp8"

    _append_default_pair(normalized_extra, "--max-cudagraph-capture-size", "2048")
    _append_default_pair(normalized_extra, "--stream-interval", "20")

    if resolved_disable_prefix_caching and has_cli_flag(
        normalized_extra, "--enable-prefix-caching"
    ):
        resolved_disable_prefix_caching = False

    return VllmRuntimeDefaults(
        kv_cache_dtype=resolved_kv_cache_dtype,
        disable_prefix_caching=resolved_disable_prefix_caching,
        extra_args=tuple(normalized_extra),
    )


def build_engine_args(config: VllmDeploymentConfig) -> list[str]:
    """Build explicit deployment/workload vLLM args.

    This intentionally does not encode vLLM's compile/CUDA graph defaults. If
    callers want non-default compilation behavior, it must be supplied through
    ``extra_args`` by an explicit non-parity mode.
    """

    args = ["--model", config.model]
    if config.max_model_len is not None:
        args.extend(["--max-model-len", str(config.max_model_len)])
    if config.max_num_seqs is not None:
        args.extend(["--max-num-seqs", str(config.max_num_seqs)])
    if config.max_num_batched_tokens is not None:
        args.extend(["--max-num-batched-tokens", str(config.max_num_batched_tokens)])
    if config.gpu_memory_utilization is not None:
        args.extend(["--gpu-memory-utilization", str(config.gpu_memory_utilization)])
    if config.tensor_parallel_size is not None:
        args.extend(["--tensor-parallel-size", str(config.tensor_parallel_size)])
    if config.pipeline_parallel_size is not None:
        args.extend(["--pipeline-parallel-size", str(config.pipeline_parallel_size)])
    if config.dtype:
        args.extend(["--dtype", config.dtype])
    if config.kv_cache_dtype:
        args.extend(["--kv-cache-dtype", config.kv_cache_dtype])
    if config.enforce_eager:
        args.append("--enforce-eager")
    if config.disable_prefix_caching:
        args.append("--no-enable-prefix-caching")
    if config.no_async_scheduling:
        args.append("--no-async-scheduling")
    args.extend(config.extra_args)
    return args


def config_hash(payload: Any) -> str:
    raw = json.dumps(_safe_json(payload), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha1(raw).hexdigest()[:16]


def _safe_json(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _safe_json(dataclasses.asdict(value))
    if isinstance(value, enum.Enum):
        return value.name
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _safe_json(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple, set)):
        return [_safe_json(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _get_attr_path(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        if cur is None:
            return None
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            cur = getattr(cur, part, None)
    return cur


def _summarize_obj(obj: Any) -> Any:
    if obj is None:
        return None
    if dataclasses.is_dataclass(obj):
        return _safe_json(obj)
    if isinstance(obj, enum.Enum):
        return obj.name
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_summarize_obj(v) for v in obj]
    if isinstance(obj, dict):
        return _safe_json(obj)
    if hasattr(obj, "__dict__"):
        out = {}
        for key, value in vars(obj).items():
            if key.startswith("_"):
                continue
            if callable(value):
                continue
            out[key] = _safe_json(value)
        return out
    return repr(obj)


def summarize_vllm_config(vllm_config: Any) -> dict[str, Any]:
    """Return parity-relevant fields from a resolved vLLM config object."""

    summary: dict[str, Any] = {}
    try:
        import vllm  # type: ignore

        summary["vllm_version"] = getattr(vllm, "__version__", None)
    except Exception:
        summary["vllm_version"] = None

    for path in CRITICAL_CONFIG_KEYS:
        if path == "vllm_version":
            continue
        value = _get_attr_path(vllm_config, path)
        summary[path] = _summarize_obj(value)
    return summary


def find_runtime_vllm_config(runtime_obj: Any) -> Any | None:
    """Best-effort lookup for a resolved VllmConfig from an LLM/runtime object."""

    candidates = (
        "vllm_config",
        "llm_engine.vllm_config",
        "llm_engine.engine.vllm_config",
        "llm_engine.model_executor.vllm_config",
        "llm_engine.model_executor.driver_worker.vllm_config",
        "llm_engine.model_executor.driver_worker.model_runner.vllm_config",
    )
    for path in candidates:
        value = _get_attr_path(runtime_obj, path)
        if value is not None:
            return value
    return None


def snapshot_effective_config_from_args(engine_args: list[str]) -> dict[str, Any]:
    """Create EngineArgs inside vLLM and summarize the resolved VllmConfig."""

    from vllm.engine.arg_utils import EngineArgs  # type: ignore

    parser = argparse.ArgumentParser(add_help=False)
    EngineArgs.add_cli_args(parser)
    parsed = parser.parse_args(engine_args)
    args = EngineArgs.from_cli_args(parsed)
    return summarize_vllm_config(args.create_engine_config())


def make_metadata(
    *,
    artifact_kind: str,
    measurement_mode: str,
    engine_args: list[str],
    deployment_config: VllmDeploymentConfig | None = None,
    effective_config: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    requested = {
        "deployment_config": dataclasses.asdict(deployment_config) if deployment_config else None,
        "engine_args": list(engine_args),
    }
    metadata = {
        "artifact_kind": artifact_kind,
        "measurement_mode": measurement_mode,
        "requested": requested,
        "effective_config": effective_config,
        "vllm_config_hash": config_hash(effective_config or requested),
    }
    if extra:
        metadata.update(extra)
    return metadata


def write_metadata(path: os.PathLike[str] | str, metadata: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(_safe_json(metadata), f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, target)


def load_metadata(path: os.PathLike[str] | str) -> dict[str, Any]:
    with Path(path).open() as f:
        return json.load(f)


def _flatten_effective(metadata: dict[str, Any]) -> dict[str, Any]:
    effective = metadata.get("effective_config") or {}
    return effective if isinstance(effective, dict) else {}


def compare_metadata(left: dict[str, Any], right: dict[str, Any]) -> list[dict[str, Any]]:
    mismatches = []
    left_effective = _flatten_effective(left)
    right_effective = _flatten_effective(right)
    for key in CRITICAL_CONFIG_KEYS:
        left_value = left_effective.get(key)
        right_value = right_effective.get(key)
        if left_value != right_value:
            mismatches.append({"key": key, "left": left_value, "right": right_value})
    return mismatches


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument("--tensor-parallel-size", type=int)
    parser.add_argument("--pipeline-parallel-size", type=int)
    parser.add_argument("--dtype")
    parser.add_argument("--kv-cache-dtype")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--disable-prefix-caching", action="store_true")
    parser.add_argument("--no-async-scheduling", action="store_true")
    parser.add_argument("--extra-arg", action="append", default=[])


def _config_from_args(args: argparse.Namespace) -> VllmDeploymentConfig:
    return VllmDeploymentConfig(
        model=args.model,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        dtype=args.dtype,
        kv_cache_dtype=args.kv_cache_dtype,
        enforce_eager=args.enforce_eager,
        disable_prefix_caching=args.disable_prefix_caching,
        no_async_scheduling=args.no_async_scheduling,
        extra_args=tuple(args.extra_arg or ()),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    build = sub.add_parser("build-args")
    _add_config_args(build)
    build.add_argument("--format", choices=("json", "lines"), default="json")

    snap = sub.add_parser("snapshot-effective")
    snap.add_argument("--args-json", required=True)
    snap.add_argument("--output")

    meta = sub.add_parser("write-metadata")
    _add_config_args(meta)
    meta.add_argument("--artifact-kind", required=True)
    meta.add_argument("--measurement-mode", required=True)
    meta.add_argument("--effective-config")
    meta.add_argument("--output", required=True)

    cmp_parser = sub.add_parser("compare")
    cmp_parser.add_argument("--left", required=True)
    cmp_parser.add_argument("--right", required=True)

    runtime = sub.add_parser("runtime-defaults")
    runtime.add_argument("--model", required=True)
    runtime.add_argument("--system")
    runtime.add_argument("--kv-cache-dtype")
    runtime.add_argument("--disable-prefix-caching", action="store_true")
    runtime.add_argument("--extra-arg", action="append", default=[])
    runtime.add_argument("--format", choices=("json", "lines"), default="json")

    args = parser.parse_args(argv)
    if args.cmd == "build-args":
        tokens = build_engine_args(_config_from_args(args))
        if args.format == "lines":
            for token in tokens:
                print(token)
        else:
            print(json.dumps(tokens))
        return 0
    if args.cmd == "snapshot-effective":
        tokens = json.loads(args.args_json)
        summary = snapshot_effective_config_from_args(tokens)
        if args.output:
            write_metadata(args.output, summary)
        else:
            print(json.dumps(_safe_json(summary), indent=2, sort_keys=True))
        return 0
    if args.cmd == "write-metadata":
        cfg = _config_from_args(args)
        tokens = build_engine_args(cfg)
        effective = load_metadata(args.effective_config) if args.effective_config else None
        write_metadata(
            args.output,
            make_metadata(
                artifact_kind=args.artifact_kind,
                measurement_mode=args.measurement_mode,
                engine_args=tokens,
                deployment_config=cfg,
                effective_config=effective,
            ),
        )
        return 0
    if args.cmd == "compare":
        mismatches = compare_metadata(load_metadata(args.left), load_metadata(args.right))
        if mismatches:
            print(json.dumps({"mismatches": mismatches}, indent=2, sort_keys=True), file=sys.stderr)
            return 1
        print("vllm metadata parity: ok")
        return 0
    if args.cmd == "runtime-defaults":
        defaults = gpt_oss_runtime_defaults(
            model=args.model,
            system=args.system,
            kv_cache_dtype=args.kv_cache_dtype,
            disable_prefix_caching=args.disable_prefix_caching,
            extra_args=tuple(args.extra_arg or ()),
        )
        payload = dataclasses.asdict(defaults)
        if args.format == "lines":
            print(f"KV_CACHE_DTYPE\t{defaults.kv_cache_dtype or ''}")
            print(f"DISABLE_PREFIX_CACHING\t{1 if defaults.disable_prefix_caching else 0}")
            for extra in defaults.extra_args:
                print(f"EXTRA_ARG\t{extra}")
        else:
            print(json.dumps(_safe_json(payload), indent=2, sort_keys=True))
        return 0
    raise AssertionError(args.cmd)


if __name__ == "__main__":
    raise SystemExit(main())
