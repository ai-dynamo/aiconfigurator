#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark Python SDK and Rust engine-step latency.

- `cargo build` (compilation overhead): not timed; runs before benchmarking
  when the Rust shared library needs to be built or refreshed.
- Rust estimator setup: timed separately from step latency. Includes
  Python/ctypes shared-library load, Rust model metadata load, Rust perf DB
  load, and estimator construction, but not `cargo build`.
- Rust/Python step latency: timed samples use already-created runners. `hot`
  warms runtime query caches before timing; `cold` clears runtime query caches
  before each timed call. Cache clearing itself is not timed.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace

from aiconfigurator.sdk import config, perf_database, rust_engine_step
from aiconfigurator.sdk.backends.factory import get_backend
from aiconfigurator.sdk.inference_session import InferenceSession
from aiconfigurator.sdk.models import get_model
from aiconfigurator.sdk.models.helpers import _get_model_info
from aiconfigurator.sdk.operations import clear_all_op_caches


@dataclass(frozen=True)
class BenchmarkCase:
    model_path: str
    system_name: str = "b200_sxm"
    backend_name: str = "vllm"
    backend_version: str = "0.19.0"
    batch_size: int = 1
    isl: int = 1024
    osl: int = 2
    prefix: int = 0
    tp_size: int = 8
    pp_size: int = 1
    attention_dp_size: int = 1
    moe_tp_size: int = 1
    moe_ep_size: int = 8


CASES = {
    # Existing Phase 3 baseline cases (MoE family + DeepSeek family via Kimi).
    "minimax-m25": BenchmarkCase(model_path="MiniMaxAI/MiniMax-M2.5"),
    "kimi-k25": BenchmarkCase(model_path="moonshotai/Kimi-K2.5"),
    # Phase 4 full family coverage. One representative model per Rust-supported
    # ModelFamily not already covered above. Parallelism mirrors the smoke
    # suite (`test_engine_step_parity.py::SMOKE_CASES`) where one exists so
    # the perf-DB tables are known to resolve.
    # Llama/Qwen3 dense family. User asked for tp=4 (smaller dense sweep).
    "qwen3-32b": BenchmarkCase(
        model_path="Qwen/Qwen3-32B",
        tp_size=4,
        moe_ep_size=1,
    ),
    # MoE family (non-DeepSeek). Smoke uses tp=4, moe_ep=4 for this model;
    # tp=8/moe_ep=8 misses perf data.
    "qwen3-30b-a3b": BenchmarkCase(
        model_path="Qwen/Qwen3-30B-A3B",
        tp_size=4,
        moe_ep_size=4,
    ),
    # DeepSeek family diversity vs Kimi (DSv3 != Kimi-K2.5 architecturally).
    "deepseek-v3": BenchmarkCase(model_path="deepseek-ai/DeepSeek-V3"),
    # DeepSeekV32 family (DSA attention + MoE).
    "deepseek-v32": BenchmarkCase(model_path="deepseek-ai/DeepSeek-V3.2"),
    # NemotronNas (Puzzle / DeciLM per-block architecture). Smoke runs at
    # tp=8 default.
    "nemotron-nas-49b": BenchmarkCase(model_path="nvidia/Llama-3_3-Nemotron-Super-49B-v1"),
    # NemotronH hybrid Mamba2 + attention + MLP. Smoke runs at tp=8 default.
    "nemotron-h-56b": BenchmarkCase(model_path="nvidia/Nemotron-H-56B-Base-8K"),
    # Qwen3.5 hybrid GDN + MoE. User asked for Qwen3-Next-80B-A3B-Instruct
    # (not in support matrix); using the smoke-verified Qwen3.5-397B-A17B
    # representative (Qwen3_5MoeForConditionalGeneration) instead.
    "qwen35-397b-a17b": BenchmarkCase(model_path="Qwen/Qwen3.5-397B-A17B"),
}


@contextlib.contextmanager
def _suppress_output(enabled: bool):
    if not enabled:
        yield
        return
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def _build_session(
    case: BenchmarkCase,
    *,
    suppress_loader_output: bool,
) -> tuple[InferenceSession, config.RuntimeConfig]:
    with _suppress_output(suppress_loader_output):
        database = perf_database.get_database(case.system_name, case.backend_name, case.backend_version)
        if database is None:
            raise RuntimeError(
                f"failed to load perf database for {case.system_name}/{case.backend_name}/{case.backend_version}"
            )
        backend = get_backend(case.backend_name)
        model_config = config.ModelConfig(
            tp_size=case.tp_size,
            pp_size=case.pp_size,
            attention_dp_size=case.attention_dp_size,
            moe_tp_size=case.moe_tp_size,
            moe_ep_size=case.moe_ep_size,
        )
        model = get_model(case.model_path, model_config, case.backend_name)
    runtime_config = config.RuntimeConfig(
        batch_size=case.batch_size,
        beam_width=1,
        isl=case.isl,
        osl=case.osl,
        prefix=case.prefix,
    )
    return InferenceSession(model, database, backend), runtime_config


def _clear_caches(case: BenchmarkCase) -> None:
    perf_database.unload_database(case.system_name, case.backend_name, case.backend_version)
    clear_all_op_caches()
    _get_model_info.cache_clear()
    rust_engine_step._engine_handle_cache_clear()


def _ensure_rust_library_present() -> None:
    # The compiled engine ships as the maturin-built ``aiconfigurator_core``
    # extension; importing it is the availability check.
    import aiconfigurator_core  # noqa: F401


def _measure(
    call: Callable[[], float],
    *,
    warmup: int,
    iterations: int,
    before_call: Callable[[], None] | None = None,
) -> dict[str, float]:
    for _ in range(warmup):
        if before_call is not None:
            before_call()
        call()

    samples = []
    for _ in range(iterations):
        if before_call is not None:
            before_call()
        start = time.perf_counter_ns()
        call()
        elapsed_ns = time.perf_counter_ns() - start
        samples.append(elapsed_ns / 1000.0)

    return {
        "call_mean_us": statistics.fmean(samples),
        "call_median_us": statistics.median(samples),
        "call_p50_us": _percentile(samples, 50),
        "call_p90_us": _percentile(samples, 90),
        "call_p99_us": _percentile(samples, 99),
        "call_min_us": min(samples),
        "call_max_us": max(samples),
        "iterations": float(iterations),
    }


def _percentile(samples: list[float], percentile: float) -> float:
    ordered = sorted(samples)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _phase_call(
    session: InferenceSession,
    runtime_config: config.RuntimeConfig,
    *,
    phase: str,
    engine_step_backend: str,
    suppress_output: bool,
) -> Callable[[], float]:
    mode = {"context": "static_ctx", "generation": "static_gen"}[phase]
    runtime = replace(runtime_config, engine_step_backend=engine_step_backend)

    def call() -> float:
        with _suppress_output(suppress_output):
            return session.run_static_latency_only(runtime, mode=mode, stride=1)

    return call


def _reset_python_runtime_caches(session: InferenceSession) -> None:
    session._database.clear_runtime_caches()


def _measure_python_session_setup_ms(
    case: BenchmarkCase,
    *,
    suppress_loader_output: bool,
) -> tuple[float, InferenceSession, config.RuntimeConfig]:
    start = time.perf_counter_ns()
    session, runtime_config = _build_session(case, suppress_loader_output=suppress_loader_output)
    return (time.perf_counter_ns() - start) / 1_000_000.0, session, runtime_config


def _measure_rust_estimator_setup_ms(session: InferenceSession) -> float:
    rust_engine_step._engine_handle_cache_clear()
    start = time.perf_counter_ns()
    rust_engine_step._cached_engine_handle(session._model, session._database)
    return (time.perf_counter_ns() - start) / 1_000_000.0


def _record_phase_results(
    results: dict,
    phase: str,
    python: dict[str, float],
    rust: dict[str, float],
) -> None:
    rust["speedup_vs_python_mean"] = python["call_mean_us"] / rust["call_mean_us"]
    rust["speedup_vs_python_median"] = python["call_median_us"] / rust["call_median_us"]
    rust["speedup_vs_python_p50"] = python["call_p50_us"] / rust["call_p50_us"]
    rust["speedup_vs_python_p90"] = python["call_p90_us"] / rust["call_p90_us"]
    rust["speedup_vs_python_p99"] = python["call_p99_us"] / rust["call_p99_us"]
    results["phases"][phase] = {"python": python, "rust": rust}


def _run_case(
    case: BenchmarkCase,
    *,
    warmup: int,
    iterations: int,
    suppress_output: bool,
    cache_mode: str,
) -> dict:
    _clear_caches(case)
    _ensure_rust_library_present()

    python_session_setup_ms, session, runtime_config = _measure_python_session_setup_ms(
        case,
        suppress_loader_output=suppress_output,
    )
    rust_estimator_setup_ms = _measure_rust_estimator_setup_ms(session)
    results = {
        "case": asdict(case),
        "cache_mode": cache_mode,
        "python_session_setup_ms": python_session_setup_ms,
        "rust_estimator_setup_ms": rust_estimator_setup_ms,
        "phases": {},
    }

    for phase in ("context", "generation"):
        python_reset = lambda: _reset_python_runtime_caches(session)
        rust_reset = lambda: rust_engine_step._engine_handle_cache_clear()

        python_reset()
        python = _measure(
            _phase_call(
                session,
                runtime_config,
                phase=phase,
                engine_step_backend="python",
                suppress_output=suppress_output,
            ),
            warmup=warmup,
            iterations=iterations,
            before_call=python_reset if cache_mode == "cold" else None,
        )
        rust_reset()
        rust = _measure(
            _phase_call(
                session,
                runtime_config,
                phase=phase,
                engine_step_backend="rust",
                suppress_output=suppress_output,
            ),
            warmup=warmup,
            iterations=iterations,
            before_call=rust_reset if cache_mode == "cold" else None,
        )
        _record_phase_results(results, phase, python, rust)

    return results


def _print_table(result: dict) -> None:
    case = result["case"]
    print(
        f"Case: {case['model_path']} on {case['system_name']}/{case['backend_name']} "
        f"{case['backend_version']} "
        f"(bs={case['batch_size']}, isl={case['isl']}, osl={case['osl']}, prefix={case['prefix']}, "
        f"tp={case['tp_size']}, pp={case['pp_size']}, dp={case['attention_dp_size']}, "
        f"etp={case['moe_tp_size']}, ep={case['moe_ep_size']})"
    )
    print(f"Cache mode: {result['cache_mode']}")
    print(
        "Python session setup (model + perf DB + backend): "
        f"{result['python_session_setup_ms']:.2f} ms (one-time, excluded from step latency)"
    )
    print(
        "Rust estimator setup (ctypes load + Rust model/perf DB load + constructor): "
        f"{result['rust_estimator_setup_ms']:.2f} ms (one-time, excluded from step latency)"
    )
    print()
    print("phase       engine    mean_us   p50_us   p90_us   p99_us   speedup")
    print("----------  -------  --------  -------  -------  -------  -------")
    for phase, engines in result["phases"].items():
        for engine_name in ("python", "rust"):
            row = engines[engine_name]
            speedup = row.get("speedup_vs_python_p50")
            speedup_text = "-" if speedup is None else f"{speedup:.2f}x"
            print(
                f"{phase:<10}  {engine_name:<7}  "
                f"{row['call_mean_us']:>8.2f}  "
                f"{row['call_p50_us']:>7.2f}  "
                f"{row['call_p90_us']:>7.2f}  "
                f"{row['call_p99_us']:>7.2f}  "
                f"{speedup_text:>7}"
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark cached Python SDK vs Rust engine-step calls.")
    parser.add_argument("--case", choices=sorted(CASES), help="Benchmark one predefined case. Defaults to all cases.")
    parser.add_argument("--model-path")
    parser.add_argument("--system-name")
    parser.add_argument("--backend-name")
    parser.add_argument("--backend-version")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--isl", type=int)
    parser.add_argument("--osl", type=int)
    parser.add_argument("--prefix", type=int)
    parser.add_argument("--tp-size", type=int)
    parser.add_argument("--pp-size", type=int)
    parser.add_argument("--attention-dp-size", type=int)
    parser.add_argument("--moe-tp-size", type=int)
    parser.add_argument("--moe-ep-size", type=int)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument(
        "--cache-mode",
        choices=("hot", "cold"),
        default="hot",
        help=(
            "hot clears runtime query caches once per table row; cold also clears them before every warmup "
            "and timed sample."
        ),
    )
    parser.add_argument("--show-loader-output", action="store_true", help="Do not suppress perf DB loader output.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of a table.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    overrides = {
        key: getattr(args, key)
        for key in (
            "model_path",
            "system_name",
            "backend_name",
            "backend_version",
            "batch_size",
            "isl",
            "osl",
            "prefix",
            "tp_size",
            "pp_size",
            "attention_dp_size",
            "moe_tp_size",
            "moe_ep_size",
        )
        if getattr(args, key) is not None
    }
    case_names = [args.case] if args.case is not None else list(CASES)
    results: list[dict] = []
    for case_name in case_names:
        case = replace(CASES[case_name], **overrides)
        try:
            result = _run_case(
                case,
                warmup=args.warmup,
                iterations=args.iterations,
                suppress_output=not args.show_loader_output,
                cache_mode=args.cache_mode,
            )
            result["case_name"] = case_name
            results.append(result)
        except Exception as exc:
            import traceback

            results.append(
                {
                    "case_name": case_name,
                    "case": asdict(case),
                    "cache_mode": args.cache_mode,
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc).splitlines()[0][:500] if str(exc) else "",
                        "traceback": traceback.format_exc(),
                    },
                }
            )
    if args.json:
        if args.case is not None and len(results) == 1:
            output = results[0]
        else:
            output = {"results": results}
        print(json.dumps(output, indent=2, sort_keys=True))
    else:
        for index, result in enumerate(results):
            if index:
                print()
            if "error" in result:
                case = result["case"]
                print(
                    f"Case: {case['model_path']} on {case['system_name']}/{case['backend_name']} "
                    f"{case['backend_version']} - FAILED ({result['error']['type']}): "
                    f"{result['error']['message']}"
                )
            else:
                _print_table(result)


if __name__ == "__main__":
    main()
