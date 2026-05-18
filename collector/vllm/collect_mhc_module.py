# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""mHC module collector for vLLM DeepSeek-V4."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from importlib.metadata import version as get_version
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from collector.case_generator import get_common_mhc_test_cases
from collector.helper import benchmark_with_power, log_perf
from collector.registry_types import PerfFile
from collector.vllm.utils import setup_distributed

__compat__ = "vllm>=0.20.1"


DEFAULT_HIDDEN_SIZE = 4096
DEFAULT_HC_MULT = 4
ARCHITECTURE = "DeepseekV4ForCausalLM"
MHC_NUM_SITES = 2
MHC_SINKHORN_ITERS = 20
MHC_EPS = 1.0e-6


def _parse_int_list(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def _resolve_perf_path(output_path: str | None, filename: str | None) -> str:
    if filename is None:
        raise ValueError("filename is required")
    if not output_path:
        return filename
    if output_path.endswith(".txt"):
        return output_path
    os.makedirs(output_path, exist_ok=True)
    return os.path.join(output_path, filename)


def _init_cuda(device: str) -> None:
    setup_distributed(device)
    torch.cuda.set_device(device)
    try:
        from vllm.v1.worker.workspace import init_workspace_manager

        init_workspace_manager(torch.device(device))
    except (ImportError, RuntimeWarning):
        pass


def _active_mhc_common_cases():
    seen: set[tuple[str, int, int]] = set()
    for case in get_common_mhc_test_cases():
        key = (case.phase, case.hidden_size, case.hc_mult)
        if key in seen:
            continue
        seen.add(key)
        yield case


def get_mhc_module_test_cases() -> list[dict]:
    cases: list[dict] = []
    for case in _active_mhc_common_cases():
        num_tokens_list = [16] if "--smoke" in sys.argv else case.num_tokens_list
        for num_tokens in num_tokens_list:
            cases.append(
                {
                    "id": f"mhc_{case.phase}_hs{case.hidden_size}_hcm{case.hc_mult}_{num_tokens}",
                    "params": [case.phase, num_tokens, case.hidden_size, case.hc_mult],
                }
            )
    return cases


def _default_num_tokens() -> list[int]:
    cases = get_common_mhc_test_cases()
    if not cases:
        raise RuntimeError("get_common_mhc_test_cases() returned no cases")
    return cases[0].num_tokens_list


def _make_mhc_tensors(num_tokens: int, hidden_size: int, hc_mult: int, *, device: str):
    mix_hc = (2 + hc_mult) * hc_mult
    hc_dim = hc_mult * hidden_size
    residual = torch.randn(num_tokens, hc_mult, hidden_size, dtype=torch.bfloat16, device=device)
    fn = torch.randn(mix_hc, hc_dim, dtype=torch.float32, device=device)
    base = torch.randn(mix_hc, dtype=torch.float32, device=device)
    scale = torch.ones(3, dtype=torch.float32, device=device)
    return residual, fn, base, scale


def _mhc_pre(residual, fn, base, scale):
    post, comb, layer_input = torch.ops.vllm.mhc_pre(
        residual=residual,
        fn=fn,
        hc_scale=scale,
        hc_base=base,
        rms_eps=MHC_EPS,
        hc_pre_eps=MHC_EPS,
        hc_sinkhorn_eps=MHC_EPS,
        hc_post_mult_value=2.0,
        sinkhorn_repeat=MHC_SINKHORN_ITERS,
    )
    return layer_input, post, comb


def run_mhc_module(
    *,
    ops: Sequence[str],
    num_tokens_cases: Sequence[int] | None = None,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    hc_mult: int = DEFAULT_HC_MULT,
    device: str = "cuda:0",
    output_path: str | None = None,
    perf_filename: str | None = None,
    num_warmup: int = 5,
    num_iterations: int = 10,
) -> list[dict]:
    import vllm.model_executor.layers.mhc  # noqa: F401

    _init_cuda(device)
    hidden_size = int(hidden_size)
    hc_mult = int(hc_mult)
    token_cases = list(num_tokens_cases or _default_num_tokens())
    if "--smoke" in sys.argv and num_tokens_cases is None:
        token_cases = [16]

    results = []
    for op in ops:
        if op not in {"pre", "post"}:
            raise ValueError(f"unsupported mHC op: {op}")
        for num_tokens in token_cases:
            site_inputs = [
                _make_mhc_tensors(num_tokens, hidden_size, hc_mult, device=device) for _ in range(MHC_NUM_SITES)
            ]
            if op == "pre":

                def kernel_func(site_inputs=site_inputs):
                    with torch.no_grad():
                        return [_mhc_pre(residual, fn, base, scale) for residual, fn, base, scale in site_inputs]

            else:
                with torch.no_grad():
                    post_inputs = [
                        (_mhc_pre(residual, fn, base, scale), residual) for residual, fn, base, scale in site_inputs
                    ]
                torch.cuda.synchronize()

                def kernel_func(post_inputs=post_inputs):
                    with torch.no_grad():
                        return [
                            torch.ops.vllm.mhc_post(x, residual, post, comb)
                            for (x, post, comb), residual in post_inputs
                        ]

            with benchmark_with_power(
                device=torch.device(device),
                kernel_func=kernel_func,
                num_warmups=num_warmup,
                num_runs=num_iterations,
                repeat_n=1,
                allow_graph_fail=False,
                use_cuda_graph=True,
            ) as result:
                pass
            latency = float(result["latency_ms"])
            log_perf(
                item_list=[
                    {
                        "architecture": ARCHITECTURE,
                        "num_tokens": num_tokens,
                        "num_sites": MHC_NUM_SITES,
                        "hc_mult": hc_mult,
                        "hidden_size": hidden_size,
                        "sinkhorn_iters": MHC_SINKHORN_ITERS,
                        "latency": f"{latency:.4f}",
                    }
                ],
                framework="VLLM",
                version=get_version("vllm"),
                device_name=torch.cuda.get_device_name(device),
                op_name=op,
                kernel_source="vllm_mhc",
                perf_filename=_resolve_perf_path(output_path, perf_filename or PerfFile.MHC_MODULE.value),
                power_stats=result.get("power_stats"),
            )
            print(f"[vllm-mhc] op={op} tokens={num_tokens} latency={latency:.4f} ms")
            results.append({"op": op, "num_tokens": num_tokens, "latency": latency})
            del site_inputs
            torch.cuda.empty_cache()
    return results


def run_mhc_module_worker(
    op: str,
    num_tokens: int,
    hidden_size: int,
    hc_mult: int,
    *,
    perf_filename: str,
    device: str = "cuda:0",
) -> None:
    output_path = os.path.dirname(perf_filename) or os.getcwd()
    run_mhc_module(
        ops=[op],
        num_tokens_cases=[num_tokens],
        hidden_size=hidden_size,
        hc_mult=hc_mult,
        device=device,
        output_path=output_path,
        perf_filename=os.path.basename(perf_filename),
        num_warmup=3 if "--smoke" in sys.argv else 5,
        num_iterations=3 if "--smoke" in sys.argv else 10,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect vLLM DeepSeek-V4 mHC module latency.")
    parser.add_argument("--op", choices=["pre", "post", "all"], default="all")
    parser.add_argument("--num-tokens", default="16")
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument("--hc-mult", type=int, default=DEFAULT_HC_MULT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-path", default=None)

    args = parser.parse_args()
    run_mhc_module(
        ops=["pre", "post"] if args.op == "all" else [args.op],
        num_tokens_cases=_parse_int_list(args.num_tokens),
        hidden_size=args.hidden_size,
        hc_mult=args.hc_mult,
        device=args.device,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
