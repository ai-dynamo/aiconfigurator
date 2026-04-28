# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang DeepSeek-V4 DeepGEMM MegaMoE path probe.

This module verifies that a SGLang runtime enters the DSv4 MegaMoE path.  It
intentionally writes only sidecar JSON reports.  Formal perf rows must come
from ``collect_dsv4_megamoe_compute`` so full-forward overhead is not mixed
into AIC's MoE compute table.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as get_version
from pathlib import Path
from typing import Any

try:
    from helper import benchmark_with_power
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from helper import benchmark_with_power


DEFAULT_DSV4_HIDDEN_SIZE = 4096
DEFAULT_DSV4_MOE_INTER_SIZE = 2048
DEFAULT_DSV4_TOPK = 6
DEFAULT_DSV4_NUM_EXPERTS = 256
DEFAULT_DSV4_NUM_HASH_LAYERS = 3
DEFAULT_MEGAMOE_TOKEN_CAP = 1024
DEFAULT_TEST_LAYER = 3
PERF_FILENAME = "dsv4_megamoe_probe.json"
REPORT_SCHEMA_VERSION = "aic-dsv4-megamoe-probe-v1"


def dsv4_megamoe_env_defaults() -> dict[str, str]:
    """Environment values required before SGLang model construction."""
    return {
        "SGLANG_DSV4_MODE": "2604",
        "SGLANG_DSV4_2604_SUBMODE": "2604B",
        "SGLANG_DSV4_FP4_EXPERTS": "1",
        "SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE": "1",
        "SGLANG_OPT_FIX_HASH_MEGA_MOE": "1",
        "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK": str(DEFAULT_MEGAMOE_TOKEN_CAP),
        "SGLANG_OPT_MEGA_MOE_FUSED_PRE_DISPATCH": "1",
    }


@dataclass(frozen=True)
class Dsv4MegaMoETask:
    moe_type: str
    num_tokens: int
    hidden_size: int
    inter_size: int
    topk: int
    num_experts: int
    moe_tp_size: int
    moe_ep_size: int
    model_name: str
    perf_filename: str
    distribution: str
    power_law_alpha: float | None

    def as_params(self) -> list[Any]:
        return [
            self.moe_type,
            self.num_tokens,
            self.hidden_size,
            self.inter_size,
            self.topk,
            self.num_experts,
            self.moe_tp_size,
            self.moe_ep_size,
            self.model_name,
            self.perf_filename,
            self.distribution,
            self.power_law_alpha,
        ]


def _parse_bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _safe_version(package: str) -> str:
    try:
        return get_version(package)
    except PackageNotFoundError:
        return "unknown"


def _apply_dsv4_megamoe_env_defaults() -> dict[str, str]:
    applied: dict[str, str] = {}
    for key, value in dsv4_megamoe_env_defaults().items():
        os.environ.setdefault(key, value)
        applied[key] = os.environ[key]
    return applied


def _artifact_dir(perf_filename: str) -> Path:
    report_dir = os.environ.get("AIC_DSV4_MEGAMOE_REPORT_DIR")
    if report_dir:
        path = Path(report_dir)
    else:
        perf_path = Path(perf_filename)
        path = perf_path.parent if perf_path.parent != Path("") else Path(".")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _report_path(task: Dsv4MegaMoETask, device_id: int, suffix: str = "probe") -> Path:
    safe_distribution = task.distribution.replace("/", "_").replace(":", "_")
    name = f"dsv4_megamoe_{suffix}_{task.num_tokens}_{safe_distribution}_gpu{device_id}.json"
    return _artifact_dir(task.perf_filename) / name


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _base_report(task: Dsv4MegaMoETask, device_id: int, status: str) -> dict[str, Any]:
    env_keys = sorted(dsv4_megamoe_env_defaults())
    return {
        "schema": REPORT_SCHEMA_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": status,
        "task": {
            "moe_type": task.moe_type,
            "num_tokens": task.num_tokens,
            "hidden_size": task.hidden_size,
            "inter_size": task.inter_size,
            "topk": task.topk,
            "num_experts": task.num_experts,
            "moe_tp_size": task.moe_tp_size,
            "moe_ep_size": task.moe_ep_size,
            "model_name": task.model_name,
            "distribution": task.distribution,
            "power_law_alpha": task.power_law_alpha,
        },
        "device_id": device_id,
        "env": {key: os.environ.get(key) for key in env_keys},
    }


def validate_megamoe_probe(report: dict[str, Any]) -> list[str]:
    """Return validation errors for a MegaMoE probe report."""
    errors: list[str] = []
    if report.get("schema") != REPORT_SCHEMA_VERSION:
        errors.append("unexpected_schema")
    if report.get("status") != "ok":
        errors.append(f"status_not_ok:{report.get('status')}")

    observed = report.get("observed") or {}
    if observed.get("should_use_mega_moe") is not True:
        errors.append("should_use_mega_moe_false")
    if observed.get("forward_mega_moe_calls", 0) < 1:
        errors.append("forward_mega_moe_not_called")
    if observed.get("run_mega_routed_calls", 0) < 1:
        errors.append("run_mega_routed_not_called")
    if observed.get("deep_gemm_fp8_fp4_mega_moe_calls", 0) < 1:
        errors.append("deep_gemm_mega_moe_not_called")
    if observed.get("mega_l1_weights_present") is not True:
        errors.append("mega_l1_weights_missing")
    if observed.get("mega_l2_weights_present") is not True:
        errors.append("mega_l2_weights_missing")
    if observed.get("mega_moe_weights_built") is not True:
        errors.append("mega_moe_weights_not_built")

    return errors


def verify_megamoe_probe_file(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load and validate a MegaMoE probe JSON report."""
    report_path = Path(path)
    with open(report_path) as f:
        report = json.load(f)
    errors = validate_megamoe_probe(report)
    if errors:
        raise RuntimeError(f"{report_path}: MegaMoE probe validation failed: {errors}")
    return report


def _default_model_config(task: Dsv4MegaMoETask) -> dict[str, Any]:
    num_hidden_layers = max(DEFAULT_TEST_LAYER + 1, DEFAULT_DSV4_NUM_HASH_LAYERS + 1)
    return {
        "architectures": ["DeepseekV4ForCausalLM"],
        "model_type": "deepseek_ref",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 0,
        "eos_token_id": 1,
        "pad_token_id": 1,
        "first_k_dense_replace": 0,
        "hidden_act": "silu",
        "hidden_size": task.hidden_size,
        "index_head_dim": 128,
        "index_n_heads": 64,
        "index_topk": 512,
        "intermediate_size": task.inter_size,
        "kv_lora_rank": 512,
        "max_position_embeddings": 65536,
        "moe_intermediate_size": task.inter_size,
        "moe_layer_freq": 1,
        "n_group": 8,
        "n_hash_layers": DEFAULT_DSV4_NUM_HASH_LAYERS,
        "num_hash_layers": DEFAULT_DSV4_NUM_HASH_LAYERS,
        "n_routed_experts": task.num_experts,
        "n_shared_experts": 1,
        "norm_topk_prob": True,
        "num_attention_heads": 64,
        "num_experts_per_tok": task.topk,
        "num_hidden_layers": num_hidden_layers,
        "num_key_value_heads": 1,
        "q_lora_rank": 1024,
        "qk_nope_head_dim": 448,
        "qk_rope_head_dim": 64,
        "rms_norm_eps": 1e-6,
        "rope_scaling": {},
        "rope_theta": 10000,
        "routed_scaling_factor": 1.5,
        "scoring_func": "sqrtsoftplus",
        "tie_word_embeddings": False,
        "topk_group": 8,
        "topk_method": "noaux_tc",
        "use_cache": True,
        "v_head_dim": 512,
        "vocab_size": 129280,
        "o_lora_rank": 1024,
        "o_groups": 8,
        "window_size": 128,
        "compress_rope_theta": 40000,
        "compress_ratios": [0] * num_hidden_layers,
        "hc_mult": 4,
        "hc_sinkhorn_iters": 20,
        "hc_eps": 1e-6,
        "quantization_config": {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
        },
    }


def _resolve_model_path(task: Dsv4MegaMoETask) -> str:
    for env_name in ("AIC_DSV4_MODEL_PATH", "SGLANG_DSV4_MODEL_PATH", "DEEPSEEK_V4_MODEL_PATH", "MOE_MODEL_PATH"):
        model_path = os.environ.get(env_name)
        if model_path:
            return model_path

    tmp_dir = Path(tempfile.gettempdir()) / f"aic_dsv4_megamoe_config_{os.getpid()}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    _write_json(tmp_dir / "config.json", _default_model_config(task))
    _write_json(
        tmp_dir / "generation_config.json",
        {"bos_token_id": 0, "eos_token_id": 1, "pad_token_id": 1},
    )
    return str(tmp_dir)


def get_dsv4_megamoe_test_cases() -> list[list[Any]]:
    """Return DSv4 MegaMoE collection tasks.

    The default set stays within SGLang's MegaMoE eager token cap.  Set
    ``AIC_DSV4_MEGAMOE_FULL=1`` to sweep a denser matrix.
    """
    if _parse_bool_env("AIC_DSV4_MEGAMOE_FULL", default=False):
        num_tokens_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    else:
        num_tokens_list = [1, 8, 32, 128, 512, 1024]

    tasks = [
        Dsv4MegaMoETask(
            moe_type="mxfp4",
            num_tokens=num_tokens,
            hidden_size=DEFAULT_DSV4_HIDDEN_SIZE,
            inter_size=DEFAULT_DSV4_MOE_INTER_SIZE,
            topk=DEFAULT_DSV4_TOPK,
            num_experts=DEFAULT_DSV4_NUM_EXPERTS,
            moe_tp_size=1,
            moe_ep_size=1,
            model_name="deepseek-ai/DeepSeek-V4",
            perf_filename=PERF_FILENAME,
            distribution="model_router",
            power_law_alpha=None,
        )
        for num_tokens in num_tokens_list
    ]
    return [task.as_params() for task in tasks]


def _coerce_task(
    moe_type: str,
    num_tokens: int,
    hidden_size: int,
    inter_size: int,
    topk: int,
    num_experts: int,
    moe_tp_size: int,
    moe_ep_size: int,
    model_name: str,
    perf_filename: str,
    distribution: str,
    power_law_alpha: float | None,
) -> Dsv4MegaMoETask:
    task = Dsv4MegaMoETask(
        moe_type=str(moe_type),
        num_tokens=int(num_tokens),
        hidden_size=int(hidden_size),
        inter_size=int(inter_size),
        topk=int(topk),
        num_experts=int(num_experts),
        moe_tp_size=int(moe_tp_size),
        moe_ep_size=int(moe_ep_size),
        model_name=str(model_name),
        perf_filename=str(perf_filename),
        distribution=str(distribution),
        power_law_alpha=None if power_law_alpha is None else float(power_law_alpha),
    )
    if task.moe_type != "mxfp4":
        raise ValueError(f"Unsupported DSv4 MegaMoE dtype: {task.moe_type}")
    if task.num_tokens <= 0:
        raise ValueError("num_tokens must be positive")
    if task.moe_tp_size != 1 or task.moe_ep_size != 1:
        raise ValueError("DSv4 MegaMoE collector currently validates single-rank MoE only")
    if task.distribution != "model_router":
        raise ValueError("DSv4 MegaMoE collector currently uses SGLang's model router distribution")
    token_cap = int(
        os.environ.get(
            "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK",
            DEFAULT_MEGAMOE_TOKEN_CAP,
        )
    )
    if task.num_tokens > token_cap:
        raise ValueError("num_tokens exceeds MegaMoE token cap")
    return task


def _wrap_instance_method(obj: Any, name: str, counters: dict[str, int], counter_key: str):
    original = getattr(obj, name)

    def wrapped(*args, **kwargs):
        counters[counter_key] += 1
        return original(*args, **kwargs)

    setattr(obj, name, wrapped)
    return original


def _find_megamoe_layer(model_runner: Any) -> tuple[int, Any]:
    layers = model_runner.model.model.layers
    for layer_id, layer in enumerate(layers):
        moe_layer = getattr(layer, "mlp", None)
        if moe_layer is None or not hasattr(moe_layer, "forward_mega_moe"):
            continue
        if getattr(moe_layer, "is_nextn", False):
            continue
        if getattr(moe_layer, "is_hash", False):
            continue
        return layer_id, moe_layer
    raise RuntimeError("No non-hash DSv4 MegaMoE-capable layer found")


def _load_model_runner(task: Dsv4MegaMoETask, gpu_id: int):
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.entrypoints.engine import _set_envs_and_config
    from sglang.srt.layers.moe import initialize_moe_config
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.server_args import PortArgs, ServerArgs
    from sglang.srt.utils import configure_logger, suppress_other_loggers

    model_path = _resolve_model_path(task)
    override = {
        "num_hidden_layers": max(DEFAULT_TEST_LAYER + 1, DEFAULT_DSV4_NUM_HASH_LAYERS + 1),
        "hidden_size": task.hidden_size,
        "intermediate_size": task.inter_size,
        "moe_intermediate_size": task.inter_size,
        "num_experts_per_tok": task.topk,
        "n_routed_experts": task.num_experts,
        "n_hash_layers": DEFAULT_DSV4_NUM_HASH_LAYERS,
        "num_hash_layers": DEFAULT_DSV4_NUM_HASH_LAYERS,
    }
    server_args = ServerArgs(
        model_path=model_path,
        dtype="auto",
        device="cuda",
        load_format="dummy",
        quantization="fp8",
        tp_size=1,
        trust_remote_code=True,
        mem_fraction_static=float(os.environ.get("AIC_DSV4_MEGAMOE_MEM_FRACTION_STATIC", "0.3")),
        moe_runner_backend="deep_gemm",
        ep_size=1,
        node_rank=0,
        host="localhost",
        port=31000 + gpu_id * 100,
        cuda_graph_max_bs=4,
        disable_cuda_graph=True,
        json_model_override_args=json.dumps(override),
    )

    suppress_other_loggers()
    configure_logger(server_args, prefix=f" DSV4-MEGAMOE-GPU{gpu_id}")
    _set_envs_and_config(server_args)
    initialize_moe_config(server_args)

    port_args = PortArgs.init_new(server_args)
    model_config = ModelConfig.from_server_args(server_args)
    model_runner = ModelRunner(
        model_config=model_config,
        mem_fraction_static=server_args.mem_fraction_static,
        gpu_id=0,
        tp_rank=0,
        tp_size=server_args.tp_size,
        pp_rank=0,
        pp_size=1,
        moe_ep_rank=0,
        moe_ep_size=server_args.ep_size,
        nccl_port=port_args.nccl_port,
        server_args=server_args,
    )
    return model_runner, server_args, model_path


def _run_verified_megamoe_forward(
    moe_layer: Any,
    task: Dsv4MegaMoETask,
    device: str,
) -> tuple[float, dict[str, Any], dict[str, Any] | None]:
    import torch

    counters = {
        "forward_mega_moe_calls": 0,
        "run_mega_routed_calls": 0,
        "deep_gemm_fp8_fp4_mega_moe_calls": 0,
    }
    original_forward = _wrap_instance_method(moe_layer, "forward_mega_moe", counters, "forward_mega_moe_calls")
    original_routed = _wrap_instance_method(moe_layer, "_run_mega_routed", counters, "run_mega_routed_calls")

    deep_gemm = None
    original_deep_gemm = None
    try:
        import deep_gemm

        original_deep_gemm = deep_gemm.fp8_fp4_mega_moe

        def counted_deep_gemm(*args, **kwargs):
            counters["deep_gemm_fp8_fp4_mega_moe_calls"] += 1
            return original_deep_gemm(*args, **kwargs)

        deep_gemm.fp8_fp4_mega_moe = counted_deep_gemm

        hidden_states = torch.randn(
            task.num_tokens,
            task.hidden_size,
            dtype=torch.bfloat16,
            device=device,
        )
        should_use = bool(moe_layer._should_use_mega_moe(hidden_states))
        if not should_use:
            raise RuntimeError("SGLang _should_use_mega_moe returned False")

        # Single eager run before timing gives the probe an unambiguous branch
        # observation even if CUDA graph capture later replays a captured path.
        _ = moe_layer(hidden_states, forward_batch=None, input_ids_global=None)
        torch.cuda.synchronize()

        def kernel_func():
            moe_layer(hidden_states, forward_batch=None, input_ids_global=None)

        with benchmark_with_power(
            device=torch.device(device),
            kernel_func=kernel_func,
            num_warmups=int(os.environ.get("AIC_DSV4_MEGAMOE_NUM_WARMUPS", "3")),
            num_runs=int(os.environ.get("AIC_DSV4_MEGAMOE_NUM_RUNS", "10")),
            repeat_n=1,
            allow_graph_fail=True,
        ) as results:
            pass

        observed = {
            **counters,
            "should_use_mega_moe": should_use,
            "mega_moe_weights_built": bool(getattr(moe_layer.experts, "_mega_moe_weights_built", False)),
            "mega_l1_weights_present": hasattr(moe_layer.experts, "mega_l1_weights"),
            "mega_l2_weights_present": hasattr(moe_layer.experts, "mega_l2_weights"),
            "moe_layer_class": moe_layer.__class__.__name__,
            "experts_class": moe_layer.experts.__class__.__name__,
            "experts_quant_method": getattr(
                getattr(moe_layer.experts, "quant_method", None),
                "__class__",
                type(None),
            ).__name__,
            "is_hash_layer": bool(getattr(moe_layer, "is_hash", False)),
            "is_nextn": bool(getattr(moe_layer, "is_nextn", False)),
        }
        return float(results["latency_ms"]), observed, results.get("power_stats")
    finally:
        moe_layer.forward_mega_moe = original_forward
        moe_layer._run_mega_routed = original_routed
        if deep_gemm is not None and original_deep_gemm is not None:
            deep_gemm.fp8_fp4_mega_moe = original_deep_gemm


def _run_dsv4_megamoe_inprocess(task: Dsv4MegaMoETask, device_id: int) -> None:
    env_snapshot = _apply_dsv4_megamoe_env_defaults()

    if _parse_bool_env("AIC_DSV4_MEGAMOE_DRY_RUN", default=False):
        report = _base_report(task, device_id, status="dry_run")
        report["env"] = env_snapshot
        report["message"] = "Dry run only; SGLang runtime was not imported."
        _write_json(_report_path(task, device_id, suffix="dry_run"), report)
        return

    import torch

    if not torch.cuda.is_available():
        report = _base_report(task, device_id, status="skipped")
        report["env"] = env_snapshot
        report["error"] = "CUDA is not available"
        _write_json(_report_path(task, device_id, suffix="probe"), report)
        raise RuntimeError("CUDA is not available")

    torch.cuda.set_device("cuda:0")
    model_runner = None
    report = _base_report(task, device_id, status="running")
    try:
        model_runner, server_args, model_path = _load_model_runner(task, device_id)
        layer_id, moe_layer = _find_megamoe_layer(model_runner)
        latency_ms, observed, power_stats = _run_verified_megamoe_forward(moe_layer, task, "cuda:0")

        report.update(
            {
                "status": "ok",
                "model_path": model_path,
                "sglang_version": _safe_version("sglang"),
                "layer_id": layer_id,
                "observed": observed,
                "latency_ms": latency_ms,
                "power_stats": power_stats,
            }
        )
        validation_errors = validate_megamoe_probe(report)
        report["validation_errors"] = validation_errors
        _write_json(_report_path(task, device_id, suffix="probe"), report)
        if validation_errors:
            raise RuntimeError(f"MegaMoE probe failed validation: {validation_errors}")
    except Exception as e:
        report.update(
            {
                "status": "error",
                "error_type": type(e).__name__,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
        )
        _write_json(_report_path(task, device_id, suffix="probe"), report)
        raise
    finally:
        if model_runner is not None:
            del model_runner
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def _run_dsv4_megamoe_from_env() -> None:
    payload = json.loads(os.environ["AIC_DSV4_MEGAMOE_TASK_JSON"])
    task = Dsv4MegaMoETask(**payload["task"])
    _run_dsv4_megamoe_inprocess(task, int(payload["device_id"]))


def _run_subprocess(task: Dsv4MegaMoETask, device_id: int) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device_id)
    env["AIC_DSV4_MEGAMOE_TASK_JSON"] = json.dumps(
        {
            "task": task.__dict__,
            "device_id": device_id,
        }
    )
    module_dir = os.path.dirname(os.path.abspath(__file__))
    code = f"""
import sys
sys.path.insert(0, {module_dir!r})
from collect_dsv4_megamoe import _run_dsv4_megamoe_from_env
_run_dsv4_megamoe_from_env()
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        cwd=module_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=int(os.environ.get("AIC_DSV4_MEGAMOE_TIMEOUT_SEC", "900")),
    )
    if proc.stdout:
        print(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"DSv4 MegaMoE subprocess failed with exit code {proc.returncode}")


def run_dsv4_megamoe(
    moe_type,
    num_tokens,
    hidden_size,
    inter_size,
    topk,
    num_experts,
    moe_tp_size,
    moe_ep_size,
    model_name,
    perf_filename,
    distribution,
    power_law_alpha=None,
    device="cuda:0",
):
    """Collect one DSv4 MegaMoE data point with an execution-path probe."""
    _apply_dsv4_megamoe_env_defaults()
    task = _coerce_task(
        moe_type,
        num_tokens,
        hidden_size,
        inter_size,
        topk,
        num_experts,
        moe_tp_size,
        moe_ep_size,
        model_name,
        perf_filename,
        distribution,
        power_law_alpha,
    )
    device_str = str(device)
    device_id = int(device_str.split(":")[-1]) if ":" in device_str else 0

    if _parse_bool_env("AIC_DSV4_MEGAMOE_DRY_RUN", default=False):
        _run_dsv4_megamoe_inprocess(task, device_id)
        return

    _run_subprocess(task, device_id)


def get_dsv4_megamoe_probe_test_cases() -> list[list[Any]]:
    return get_dsv4_megamoe_test_cases()


def run_dsv4_megamoe_probe(*args, **kwargs):
    return run_dsv4_megamoe(*args, **kwargs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SGLang DeepSeek-V4 MegaMoE collector")
    parser.add_argument("--dry-run", action="store_true", help="Write probe plan JSON without importing SGLang")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N cases")
    parser.add_argument(
        "--verify-report",
        action="append",
        default=[],
        help="Validate an existing MegaMoE probe JSON report and exit",
    )
    args = parser.parse_args()

    if args.verify_report:
        for report_file in args.verify_report:
            verify_megamoe_probe_file(report_file)
            print(f"{report_file}: ok")
        sys.exit(0)

    if args.dry_run:
        os.environ["AIC_DSV4_MEGAMOE_DRY_RUN"] = "1"

    cases = get_dsv4_megamoe_test_cases()
    if args.limit:
        cases = cases[: args.limit]
    for case in cases:
        run_dsv4_megamoe(*case, device=args.device)
