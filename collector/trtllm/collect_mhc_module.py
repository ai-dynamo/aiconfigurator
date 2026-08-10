# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4 mHC pre/post module collector for TensorRT-LLM.

TRT-LLM 1.3.0rc20 ships the DeepSeek-V4 mHC (manifold-constrained
hyper-connection) module as ``tensorrt_llm._torch.modules.mhc`` —
``mHC.pre_mapping`` / ``mHC.post_mapping`` / ``mHC.fused_hc`` over the
``torch.ops.trtllm.mhc_*`` CUDA kernels — with backend selection handled by
TRT-LLM's AutoTuner (mhc_cuda.py: MhcPreMappingRunner profiles FMA vs
DeepGEMM tf32 tactics at warmup). The wheel does not yet include a
``modeling_deepseekv4.py`` model class, so unlike the SGLang collector this
one cannot build a framework-constructed decoder layer; it drives the module
API directly, exactly as the (unshipped) model forward does per the module's
own serving contract (hyper_connection.py HCState docstring).

KNOWN GAP vs the fused serving path — same decomposition as the vLLM
collector: the serving contract optionally runs ``mHC.fused_hc`` at layer
boundaries (previous site's post_mapping + next site's pre_mapping in one
autotuned pipeline, optionally folding the next RMSNorm), while this
collector measures ``pre_mapping`` (no norm fold) and ``post_mapping``
standalone, because the SDK's DeepSeekV4 model bills mhc_pre + attn_norm
(ElementWise) + mhc_post as separate per-layer ops
(aic-core/src/aiconfigurator_core/sdk/models/deepseek_v4.py) — fusing the
norm here would double-count it downstream. Measured on H20-3e (SM90,
hc_mult=4, hidden 4096/7168, 2026-08-05): fused_hc = 0.80x of pre+post at
M=16 (decode), but 3.8-6.5x SLOWER at M=1024/8192 — on pre-SM100 the
fused_hc tactic space is FMA-only (the tcgen05 MMA paths require SM100,
mhc_cuda.py:_fused_hc_mma_supported) while standalone pre_mapping can take
the DeepGEMM tensor-core tactics, so a Hopper deployment runs with fused_hc
disabled (HCState "resolved" mode) and this decomposition IS the serving
path there. On SM100/SM103 (hidden 4096/7168 are exactly the two
statically-instantiated MMA sizes) fused_hc is the serving fast path and
pre+post is the documented approximation. Aligning row semantics with the
fused path is a coordinated producer+consumer contract change; do not
switch variants unilaterally.
"""

from __future__ import annotations

__compat__ = "trtllm>=1.3.0rc20"

import argparse
import os
import sys
from collections.abc import Sequence

import torch

try:
    from case_generator import get_common_mhc_test_cases
    from registry_types import PerfFile

    from helper import benchmark_with_power, log_perf
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from case_generator import get_common_mhc_test_cases
    from registry_types import PerfFile

    from helper import benchmark_with_power, log_perf


ARCHITECTURE = "DeepseekV4ForCausalLM"
DEFAULT_HIDDEN_SIZE = 4096
DEFAULT_HC_MULT = 4
# hc_sinkhorn_iters/hc_eps from the DeepSeek-V4 checkpoint configs
# (src/aiconfigurator/model_configs/deepseek-ai--DeepSeek-V4-*_config.json:
# hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6). The remaining mHC ctor
# defaults (norm_eps/sinkhorn_eps=1e-6, post_mult_value=1.0) are the module's
# own serving defaults (hyper_connection.py mHC.__init__ @1.3.0rc20).
MHC_NUM_SITES = 2  # one decoder block runs mHC once before attention, once before FFN
MHC_SINKHORN_ITERS = 20


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


def _make_mhc_site(hidden_size: int, hc_mult: int, *, device: str):
    """One mHC parameter set (a real block holds two: attention and FFN).

    Serving-parity citations (TensorRT-LLM 1.3.0rc20,
    tensorrt_llm/_torch/modules/mhc/hyper_connection.py):
    parameter shapes/dtypes mirror ``mHC.__init__`` — ``fn`` [mix_hc, mult*hidden]
    fp32, ``base`` [mix_hc] fp32, ``scale`` [3] fp32 (hyper_connection.py:99-105),
    with mix_hc = (2+mult)*mult (hyper_connection.py:95). Random values stand in
    for checkpoint weights: both pre/post kernels dispatch on shapes/dtypes only
    (mhc_cuda.py tactic selection keys on num_tokens/hidden_size, never values).
    """
    from tensorrt_llm._torch.modules.mhc.hyper_connection import mHC

    site = mHC(mult=hc_mult, hidden_size=hidden_size, sinkhorn_iters=MHC_SINKHORN_ITERS)
    with torch.no_grad():
        site.fn.normal_()
        site.base.normal_()
        site.scale.fill_(1.0)
    return site.to(device)


def _make_residual(num_tokens: int, hidden_size: int, hc_mult: int, *, device: str) -> torch.Tensor:
    """Residual stream input in the exact ``pre_mapping`` contract (TensorRT-LLM
    1.3.0rc20 hyper_connection.py): x layout [..., mult, hidden]
    (hyper_connection.py:108), bfloat16 and trailing-dim shapes asserted at
    hyper_connection.py:114-116, outer dims flattened to
    [num_tokens, mult, hidden] at hyper_connection.py:117-119 — collapsing
    batch*seq into ``num_tokens`` up front is therefore identity-preserving.
    ``post_mapping`` consumes the same layout (hyper_connection.py:225-251).
    """
    return torch.randn(num_tokens, hc_mult, hidden_size, dtype=torch.bfloat16, device=device)


def _autotune_pre(site_inputs) -> None:
    """Serving-parity warmup: profile pre_mapping tactics like engine warmup does.

    Serving selects the pre_mapping backend via AutoTuner profiling at engine
    warmup (mhc_cuda.py mhc_pre_mapping_fused: AutoTuner.choose_one over
    MhcPreMappingRunner tactics — FMA tile variants, plus DeepGEMM
    split-K/no-split when tf32_hc_prenorm_gemm is importable). Without this
    pass every call would take the cache-miss fallback tactic instead of the
    tuned serving one.
    """
    from tensorrt_llm._torch.autotuner import autotune

    with torch.inference_mode(), autotune():
        for site, residual in site_inputs:
            site.pre_mapping(residual)
    torch.cuda.synchronize()


def _observe_pre_backend(site, residual) -> str:
    """Record which GEMM+sqrsum entry the tuned pre_mapping actually invokes.

    ``MhcPreMappingRunner.forward`` dispatches the tactic's backend through
    exactly one of two module-level entry points (mhc_cuda.py:333-347
    @1.3.0rc20): ``mhc_gemm_rms_dg_cuda`` (tactics ``dg_splitk`` /
    ``dg_nosplit``, DeepGEMM tf32) or ``mhc_gemm_rms_fma_cuda`` (``fma``,
    CUDA-core FMA). Observing the call is ground truth for ``kernel_source``
    — no assumption about autotuner cache internals.
    """
    from tensorrt_llm._torch.modules.mhc import mhc_cuda

    calls: list[str] = []
    orig_fma = mhc_cuda.mhc_gemm_rms_fma_cuda
    orig_dg = mhc_cuda.mhc_gemm_rms_dg_cuda

    def _fma(*args, **kwargs):
        calls.append("fma")
        return orig_fma(*args, **kwargs)

    def _dg(*args, **kwargs):
        num_splits = kwargs.get("num_splits", args[5] if len(args) > 5 else 1)
        calls.append("dg_splitk" if int(num_splits) > 1 else "dg_nosplit")
        return orig_dg(*args, **kwargs)

    mhc_cuda.mhc_gemm_rms_fma_cuda = _fma
    mhc_cuda.mhc_gemm_rms_dg_cuda = _dg
    try:
        with torch.no_grad():
            site.pre_mapping(residual)
    finally:
        mhc_cuda.mhc_gemm_rms_fma_cuda = orig_fma
        mhc_cuda.mhc_gemm_rms_dg_cuda = orig_dg

    backends = sorted(set(calls))
    if not backends:
        raise RuntimeError("pre_mapping dispatched no known GEMM+sqrsum entry point")
    return "+".join(backends)


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
    # TRT-LLM imports stay lazy in this module so that a mismatched install
    # fails inside collect.py's per-op error handling (after the __compat__
    # gate can label it) rather than at module import.
    import tensorrt_llm

    torch.cuda.set_device(device)
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
                (
                    _make_mhc_site(hidden_size, hc_mult, device=device),
                    _make_residual(num_tokens, hidden_size, hc_mult, device=device),
                )
                for _ in range(MHC_NUM_SITES)
            ]

            if op == "pre":
                _autotune_pre(site_inputs)
                kernel_source = f"trtllm_mhc_pre_{_observe_pre_backend(*site_inputs[0])}"

                def kernel_func(site_inputs=site_inputs):
                    with torch.no_grad():
                        return [site.pre_mapping(residual) for site, residual in site_inputs]

            else:
                # Single CUDA kernel, no tactic axis (mhc_cuda.py
                # mhc_post_mapping_cuda -> torch.ops.trtllm.mhc_post_mapping),
                # so no autotune pass; the pre_mapping below only builds
                # inputs and its tactic cannot change post_mapping latency.
                kernel_source = "trtllm_mhc_post_mapping"
                with torch.no_grad():
                    post_inputs = [(site, residual, site.pre_mapping(residual)) for site, residual in site_inputs]
                torch.cuda.synchronize()

                def kernel_func(post_inputs=post_inputs):
                    with torch.no_grad():
                        return [
                            site.post_mapping(layer_input, residual, post_mix, comb_mix)
                            for site, residual, (post_mix, comb_mix, layer_input) in post_inputs
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
                framework="TRTLLM",
                version=tensorrt_llm.__version__,
                device_name=torch.cuda.get_device_name(device),
                op_name=op,
                kernel_source=kernel_source,
                perf_filename=_resolve_perf_path(output_path, perf_filename or PerfFile.MHC_MODULE.value),
                power_stats=result.get("power_stats"),
            )
            print(f"[trtllm-mhc] op={op} tokens={num_tokens} kernel={kernel_source} latency={latency:.4f} ms")
            results.append({"op": op, "num_tokens": num_tokens, "kernel_source": kernel_source, "latency": latency})
            # kernel_func retains site_inputs/post_inputs through its default
            # argument; drop every reference before empty_cache() so the next
            # token case doesn't allocate on top of the previous one.
            del kernel_func
            if op == "post":
                del post_inputs
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
    parser = argparse.ArgumentParser(description="Collect TRT-LLM DeepSeek-V4 mHC module latency.")
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
