# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4 mHC pre/post module collector for SGLang."""

from __future__ import annotations

import argparse
import copy
import gc
import importlib.util
import json
import os
import random
import sys
import tempfile
from collections.abc import Sequence
from importlib.metadata import version as get_version

import torch

os.environ.setdefault("SGLANG_APPLY_CONFIG_BACKUP", "none")

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.append(THIS_DIR)

try:
    from common_test_cases import get_common_mhc_test_cases
    from registry_types import PerfFile

    from helper import EXIT_CODE_RESTART, benchmark_with_power, log_perf
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from common_test_cases import get_common_mhc_test_cases
    from registry_types import PerfFile

    from helper import EXIT_CODE_RESTART, benchmark_with_power, log_perf


DEFAULT_MODEL = "deepseek-ai/DeepSeek-V4-Pro"
PERF_FILENAME = PerfFile.MHC_MODULE.value

# AIC's cached HuggingFace model configs — avoids HF downloads and local
# model directories. Under dummy load_format the collector never needs
# tokenizer files or weights, so the packaged config.json alone is enough.
_MODEL_CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "src",
    "aiconfigurator",
    "model_configs",
)


def _parse_int_list(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def _read_model_config(model_id: str) -> dict:
    """Load AIC's packaged ``model_configs/<id>_config.json`` for ``model_id``.

    Only AIC-cached configs are supported — local model directories and HF
    Hub downloads are intentionally not attempted. The dummy ``load_format``
    used by this collector does not need tokenizer or weight files.
    """
    cfg_fname = model_id.replace("/", "--") + "_config.json"
    config_file = os.path.join(_MODEL_CONFIG_DIR, cfg_fname)
    if not os.path.isfile(config_file):
        raise FileNotFoundError(f"AIC packaged config not found for model_id={model_id!r}: expected {config_file}")
    with open(config_file) as f:
        return json.load(f)


def _default_num_tokens(model_path: str) -> list[int]:
    """Return the default num_tokens sweep for ``model_path`` from common_test_cases.

    Falls back to the first registered mHC test case when ``model_path`` is not
    listed in ``common_test_cases.py`` (e.g. custom / local model ids). All
    registered models share the same sweep, so the fallback is equivalent.
    """
    cases = get_common_mhc_test_cases()
    for case in cases:
        if case.model_name == model_path:
            return case.num_tokens_list
    if cases:
        return cases[0].num_tokens_list
    raise RuntimeError("get_common_mhc_test_cases() returned no cases")


def get_mhc_module_test_cases() -> list[dict]:
    """Return one task per op; each worker sweeps all num_tokens internally.

    Loading the one-layer runner is expensive, so we pay it once per op
    (pre/post) instead of per (op, num_tokens) combo.
    """
    if (
        os.environ.get("COLLECTOR_FORCE_MHC_MODULES") != "1"
        and importlib.util.find_spec("sglang.srt.models.deepseek_v4") is None
    ):
        return []
    return [{"id": f"mhc_{op}_all", "params": [op]} for op in ("pre", "post")]


def _resolve_perf_path(output_path: str | None, filename: str | None) -> str:
    filename = filename or PERF_FILENAME
    if not output_path:
        return filename
    if output_path.endswith(".txt"):
        return output_path
    os.makedirs(output_path, exist_ok=True)
    return os.path.join(output_path, filename)


def _patched_model_dir(model_id: str) -> str:
    """Build a patched model dir with a minimal ``config.json`` from AIC cache.

    Steps:
    1. Read the original config from AIC's packaged ``model_configs/``.
    2. Write a patched ``config.json`` into a temp dir — weights and tokenizer
       are NOT needed because the collector always runs with
       ``load_format="dummy"``.
    3. Preset ``SGLANG_DSV4_FP4_EXPERTS`` from ``original_config.expert_dtype``,
       since SGLang would otherwise probe routed-expert dtype from safetensors.
       An explicit user-provided env var always wins.
    """
    original_config = _read_model_config(model_id)
    config = copy.deepcopy(original_config)

    num_layers = int(os.environ.get("SGLANG_TEST_NUM_LAYERS", "2"))
    config["num_hidden_layers"] = num_layers  # shrink depth to speed up collector init
    config["model_type"] = "deepseek_ref"

    tmp_dir = os.path.join(
        tempfile.gettempdir(),
        f"aic_mhc_{model_id.replace('/', '_')}_{os.getpid()}",
    )
    os.makedirs(tmp_dir, exist_ok=True)
    with open(os.path.join(tmp_dir, "config.json"), "w") as f:
        json.dump(config, f)

    # Preset FP4 experts env from the untouched original config.
    if "SGLANG_DSV4_FP4_EXPERTS" not in os.environ:
        expert_dtype = str(original_config.get("expert_dtype", "")).lower()
        fp4_value = "1" if expert_dtype == "fp4" else "0"
        os.environ["SGLANG_DSV4_FP4_EXPERTS"] = fp4_value
        print(f"[mhc-collector] auto-set SGLANG_DSV4_FP4_EXPERTS={fp4_value} (expert_dtype={expert_dtype or 'unset'})")

    print(f"[mhc-collector] patched_dir={tmp_dir} model_id={model_id}")
    return tmp_dir


def _load_one_layer_runner(
    model_path: str,
    device: str,
    mem_fraction_static: float,
):
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.entrypoints.engine import _set_envs_and_config
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.utils import suppress_other_loggers

    suppress_other_loggers()
    device_obj = torch.device(device)
    torch.cuda.set_device(device_obj)

    local_model_path = _patched_model_dir(model_path)
    gpu_id = device_obj.index if device_obj.index is not None else torch.cuda.current_device()
    server_args = ServerArgs(
        model_path=local_model_path,
        dtype="auto",
        device="cuda",
        load_format="dummy",
        tp_size=1,
        trust_remote_code=True,
        mem_fraction_static=mem_fraction_static,
        disable_radix_cache=True,
        disable_cuda_graph=True,
        kv_cache_dtype="fp8_e4m3",
        max_total_tokens=4096,
        max_running_requests=16,
        max_prefill_tokens=4096,
    )
    server_args.enable_piecewise_cuda_graph = False
    server_args.attention_backend = "compressed"

    print(f"[mhc-collector] model_path {model_path} -> {local_model_path}")

    _set_envs_and_config(server_args)
    model_config = ModelConfig.from_server_args(server_args)
    return ModelRunner(
        model_config=model_config,
        mem_fraction_static=mem_fraction_static,
        gpu_id=gpu_id,
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        moe_ep_rank=0,
        moe_ep_size=1,
        nccl_port=29500 + random.randint(0, 10000),
        server_args=server_args,
    )


def _hidden_size(layer) -> int:
    return int(layer.config.hidden_size)


def _make_residual(layer, num_tokens: int, device: str) -> torch.Tensor:
    return torch.randn(
        num_tokens,
        layer.hc_mult,
        _hidden_size(layer),
        dtype=torch.bfloat16,
        device=device,
    )


def _mhc_call_args(layer):
    # A real DSV4 layer executes mHC once before attention and once before FFN.
    # This collector folds both calls into the reported pre/post op.
    return (
        (layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base),
        (layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base),
    )


def _make_kernel(layer, op: str, residual: torch.Tensor):
    if op == "pre":
        call_args = _mhc_call_args(layer)

        def kernel():
            return [layer.hc_pre(residual, *args) for args in call_args]

        return kernel

    if op == "post":
        with torch.no_grad():
            post_inputs = [layer.hc_pre(residual, *args) for args in _mhc_call_args(layer)]
        torch.cuda.synchronize()

        def kernel():
            return [layer.hc_post(x, residual, post, comb) for x, post, comb in post_inputs]

        return kernel

    raise ValueError(f"unsupported mHC op: {op}")


def _benchmark_mhc_kernel(
    *,
    device: str,
    kernel_func,
    num_warmup: int,
    num_iterations: int,
) -> dict:
    def timed_kernel():
        with torch.no_grad():
            return kernel_func()

    with benchmark_with_power(
        device=torch.device(device),
        kernel_func=timed_kernel,
        num_warmups=num_warmup,
        num_runs=num_iterations,
        repeat_n=1,
        allow_graph_fail=False,
        use_cuda_graph=True,
    ) as bench_result:
        pass

    if not bench_result.get("used_cuda_graph", False):
        raise RuntimeError("benchmark_with_power did not use CUDA Graph")
    return bench_result


def _log_result(
    *,
    output_path: str | None,
    perf_filename: str | None,
    model_path: str,
    op: str,
    num_tokens: int,
    hc_mult: int,
    hidden_size: int,
    latency_ms: float,
    version: str,
    device_name: str,
    power_stats: dict | None,
) -> None:
    log_perf(
        item_list=[
            {
                "model": model_path,
                "architecture": "DeepseekV4ForCausalLM",
                "num_tokens": num_tokens,
                "hc_mult": hc_mult,
                "hidden_size": hidden_size,
                "latency": f"{latency_ms:.4f}",
            }
        ],
        framework="SGLang",
        version=version,
        device_name=device_name,
        op_name=op,
        kernel_source="sglang_mhc",
        perf_filename=_resolve_perf_path(output_path, perf_filename),
        power_stats=power_stats,
    )


def run_mhc_module(
    *,
    ops: Sequence[str],
    num_tokens_cases: Sequence[int] | None = None,
    model_path: str = DEFAULT_MODEL,
    num_warmup: int = 5,
    num_iterations: int = 20,
    device: str = "cuda:0",
    output_path: str | None = None,
    mem_fraction_static: float = 0.5,
    perf_filename: str | None = None,
) -> list[dict[str, float]]:
    if num_iterations < 3:
        raise ValueError("num_iterations must be at least 3")

    token_cases = [int(num_tokens) for num_tokens in (num_tokens_cases or _default_num_tokens(model_path))]
    model_runner = _load_one_layer_runner(
        model_path,
        device=device,
        mem_fraction_static=mem_fraction_static,
    )

    layer = model_runner.model.model.layers[0]
    hidden_size = _hidden_size(layer)
    version = get_version("sglang")
    device_name = torch.cuda.get_device_name(device)
    results: list[dict[str, float]] = []

    print(
        "[mhc-collector] "
        f"hc_mult={layer.hc_mult}, hidden_size={hidden_size}, "
        f"tilelang_pre={os.environ.get('SGLANG_OPT_USE_TILELANG_MHC_PRE', 'default')}, "
        f"tilelang_post={os.environ.get('SGLANG_OPT_USE_TILELANG_MHC_POST', 'default')}"
    )

    try:
        for op in ops:
            for num_tokens in token_cases:
                try:
                    residual = _make_residual(layer, num_tokens, device)
                    bench_result = _benchmark_mhc_kernel(
                        device=device,
                        kernel_func=_make_kernel(layer, op, residual),
                        num_warmup=num_warmup,
                        num_iterations=num_iterations,
                    )
                except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError):
                    print(f"  OOM: op={op}, num_tokens={num_tokens}; skipping")
                    torch.cuda.empty_cache()
                    continue
                except RuntimeError as err:
                    # Runtime-incompatible kernels (e.g. tilelang version mismatch,
                    # unsupported shapes) should skip the single case rather than
                    # abort the whole sweep.
                    print(f"  RuntimeError: op={op}, num_tokens={num_tokens}; skipping ({err})")
                    torch.cuda.empty_cache()
                    continue

                latency_ms = float(bench_result["latency_ms"])
                _log_result(
                    output_path=output_path,
                    perf_filename=perf_filename,
                    model_path=model_path,
                    op=op,
                    num_tokens=num_tokens,
                    hc_mult=layer.hc_mult,
                    hidden_size=hidden_size,
                    latency_ms=latency_ms,
                    version=version,
                    device_name=device_name,
                    power_stats=bench_result.get("power_stats"),
                )
                results.append(
                    {
                        "op": op,
                        "num_tokens": num_tokens,
                        "mean_ms": latency_ms,
                        "n": int(bench_result.get("num_runs_executed", num_iterations)),
                        "used_cuda_graph": True,
                        "throttled": bool(bench_result.get("throttled", False)),
                    }
                )
                torch.cuda.empty_cache()
                gc.collect()
    finally:
        del model_runner
        torch.cuda.empty_cache()
        gc.collect()
    return results


def run_mhc_module_worker(
    op: str,
    *,
    perf_filename: str,
    device: str = "cuda:0",
) -> None:
    """Worker-compatible wrapper used by collector/collect.py.

    Each call sweeps all num_tokens for a single op (pre or post) in
    one subprocess. ``model_path`` is read from ``COLLECTOR_MODEL_PATH``
    (set by ``collect.py --model-path``). ``perf_filename`` and ``device``
    are keyword-only args supplied by collect.py via functools.partial
    and the worker dispatch loop.
    """
    model_path = os.environ.get("COLLECTOR_MODEL_PATH") or DEFAULT_MODEL
    output_path = os.path.dirname(perf_filename) or os.getcwd()
    run_mhc_module(
        ops=[op],
        num_tokens_cases=None,
        model_path=model_path,
        device=device,
        output_path=output_path,
        perf_filename=os.path.basename(perf_filename),
    )
    sys.exit(EXIT_CODE_RESTART)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect DeepSeek-V4 mHC pre/post module latency on SGLang.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--op", choices=["pre", "post", "all"], default="all")
    parser.add_argument("--num-tokens", default=None)
    parser.add_argument("--num-warmup", type=int, default=5)
    parser.add_argument("--num-iterations", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--mem-fraction-static", type=float, default=0.5)
    args = parser.parse_args()

    run_mhc_module(
        ops=["pre", "post"] if args.op == "all" else [args.op],
        num_tokens_cases=_parse_int_list(args.num_tokens) if args.num_tokens else None,
        model_path=args.model_path,
        num_warmup=args.num_warmup,
        num_iterations=args.num_iterations,
        device=args.device,
        output_path=args.output_path,
        mem_fraction_static=args.mem_fraction_static,
    )


if __name__ == "__main__":
    main()
