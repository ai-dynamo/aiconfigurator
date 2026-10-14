# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
KDA (Kimi Delta Attention) Collector for AIConfigurator — vLLM backend.

Kimi-K3 support lives on the vLLM `kimi-k3` branch (official preview image
vllm/vllm-openai:kimi-k3, version 0.1.dev19262+gb6bbf29dd, CUDA 13 build).
The K3 KDA layer (vllm/models/kimi_k3/nvidia/kda.py) dispatches differently
from SGLang's Triton-only path, so this collector mirrors vLLM's own
dispatch on the target SM and records the actually-invoked kernel:

Context (prefill) phase (KimiK3DeltaAttention._forward prefill branch):
    - causal_conv1d_fn x3: separate Q/K/V causal convolutions
      (kernel_source "causal_conv1d_fn_qkv3")
    - prefill core, dispatched like serving (resolve_kda_prefill_backend):
        * "flashkda_fwd" — FlashKDA CUDA extension (vllm._flashkda_C),
          the SM90/SM100/SM120 default for bf16 head_dim 128 with a gate
          lower bound; or
        * "chunk_kda_with_fused_gate" — Triton fallback.

Generation (decode) phase:
    - "fused_kda_decode" — the CUDA fused conv+recurrence+gated-RMSNorm
      decode kernel (CUDA>=13 builds; probed via is_fused_kda_decode_supported
      exactly like serving). This is the NOSPEC serving fast path and
      includes the conv update and output norm.
    - "causal_conv1d_update" + "fused_recurrent_kda_packed_decode" — the
      fallback pair, and the path serving uses whenever speculative decoding
      is enabled (num_spec != 0 permanently disables the fused kernel).

Verify (speculative target-verify, DSPARK/MTP) phase:
    - "causal_conv1d_update" (spec form: query_start_loc + num_accepted_tokens)
    - "fused_recurrent_kda" — chain verify with per-draft-token state
      checkpointing via 2-hd ssm_state_indices [num_seqs, num_spec_tokens].

The in_proj/out_proj/gate GEMMs are standard linear layers modeled by the
existing GEMM infrastructure. Tensor constructions mirror the branch's own
tests (tests/models/kimi_k3/test_kda.py).

Output:
    kda_perf.txt — same column layout as the sglang kda collector.
"""

# Serve-parity `metadata=` argument validated on the 0.27.0 release image
# (Kimi-K3 landed upstream there). The manifest kda family pin stays on the
# kimi-k3 preview build until the SDK consumer routes 0.27.0's split decode
# (see collector/framework_manifest.yaml), so this module exactly pins the
# preview version; it runs on either image — the DS-layout probe
# (is_fused_kda_decode_supported) yields fused_kda_decode rows on the
# preview and the packed conv-update + recurrence pair on 0.27.0.
__compat__ = "vllm==0.1.dev19262"

import gc
import os

import torch

try:
    from collector.case_generator import get_common_kda_test_cases
    from collector.helper import (
        WORKER_RESTART,
        benchmark_with_power,
        get_sm_version,
        log_perf,
    )
except ModuleNotFoundError:
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from case_generator import get_common_kda_test_cases

    from helper import (
        WORKER_RESTART,
        benchmark_with_power,
        get_sm_version,
        log_perf,
    )

aic_debug = int(os.getenv("aic_kda_debug", "0"))  # noqa: SIM112

# KDA safe-gate lower bound: fixed model constant for Kimi-K3
# (config.json linear_attn_config.gate_lower_bound).
KDA_LOWER_BOUND = -5.0

NUM_WARMUPS = 3
NUM_RUNS = 10


def get_kda_test_cases():
    """Test cases for KDA kernel benchmarking (context/generation/verify)."""
    test_cases = []
    for c in get_common_kda_test_cases():
        test_cases.append(
            [
                c.phase,
                c.d_model,
                c.d_conv,
                c.num_k_heads,
                c.head_k_dim,
                c.num_v_heads,
                c.head_v_dim,
                c.batch_size_list,
                c.seq_len_list,
                c.model_name,
            ]
        )
    return test_cases


def _cleanup(tag: str):
    cleanup_errors = []
    for name, fn in (("gc.collect", gc.collect), ("torch.cuda.empty_cache", torch.cuda.empty_cache)):
        try:
            fn()
        except Exception as e:
            cleanup_errors.append(f"{name}: {type(e).__name__}: {e}")
    if cleanup_errors:
        raise RuntimeError(f"vLLM KDA {tag} cleanup failed: {'; '.join(cleanup_errors)}")


def _log(common, latency_ms, kernel_source, perf_filename, vllm_version, device, power_stats):
    if not log_perf(
        item_list=[{**common, "latency": latency_ms}],
        framework="VLLM",
        version=vllm_version,
        device_name=torch.cuda.get_device_name(device),
        op_name="kda",
        kernel_source=kernel_source,
        perf_filename=perf_filename,
        power_stats=power_stats,
    ):
        raise RuntimeError(f"failed to persist vLLM KDA row to {perf_filename}")


def _resolve_prefill_kernel(dtype: torch.dtype):
    """Mirror serving's resolve_kda_prefill_backend on this device: FlashKDA
    when supported AND importable, else the Triton chunk kernel
    (vllm/models/kimi_k3/nvidia/kda.py resolve_kda_prefill_backend)."""
    from vllm.models.kimi_k3.nvidia.kda import is_flashkda_supported

    if is_flashkda_supported(128, dtype, KDA_LOWER_BOUND):
        try:
            import vllm._flashkda_C  # noqa: F401
            from vllm.models.kimi_k3.nvidia.kda import _flashkda_prefill

            return "flashkda_fwd", _flashkda_prefill
        except ImportError:
            pass
    return "chunk_kda_with_fused_gate", None


def _format_failures(failures: list[str], limit: int = 8) -> str:
    """Compact per-cell failure evidence for the strict-completeness raise.

    The full list is on stdout; the raised message carries the first ``limit``
    cells so the classified failure record is traceable to shapes without
    ballooning the errors json."""
    if not failures:
        return "<none>"
    shown = "; ".join(failures[:limit])
    extra = len(failures) - limit
    return shown + (f"; ... and {extra} more (see worker stdout)" if extra > 0 else "")


def run_kda_context_benchmark(
    d_model,
    d_conv,
    num_k_heads,
    head_k_dim,
    num_v_heads,
    head_v_dim,
    batch_size_list,
    seq_len_list,
    model_name,
    perf_filename,
    vllm_version,
    device="cuda:0",
):
    """Context (prefill): 3-way Q/K/V causal conv + prefill core kernel,
    dispatched like KimiK3DeltaAttention._forward on this SM."""
    if any(seq_len <= 1 for seq_len in seq_len_list):
        raise ValueError(
            "vLLM KDA context collection requires seq_len > 1; "
            "run_kda_torch routes seq_len=1 through the serving decode path"
        )

    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
    from vllm.models.kimi_k3.nvidia.ops.third_party.kda import chunk_kda_with_fused_gate
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
    from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata

    device = torch.device(device)
    torch.cuda.set_device(device)
    dtype = torch.bfloat16
    assert num_k_heads == num_v_heads and head_k_dim == head_v_dim
    nh, hd, cw = num_v_heads, head_v_dim, d_conv
    proj = nh * hd

    prefill_source, flashkda_prefill = _resolve_prefill_kernel(dtype)
    conv_weight = torch.randn(3 * proj, cw, dtype=torch.float32, device=device)
    q_w, k_w, v_w = conv_weight.split([proj] * 3, dim=0)
    ok = err = 0
    failures: list[str] = []

    for batch_size in batch_size_list:
        for seq_len in seq_len_list:
            nt = batch_size * seq_len
            try:
                # Resolved at the 0.27.0 era bump: the former FIXME
                # kernel-limit guard (nt * proj >= 2**31 rejected) is
                # deleted. 0.27.0's causal_conv1d keeps token addressing in
                # int64 end-to-end
                # (vllm/model_executor/layers/mamba/ops/causal_conv1d.py:40,48
                # `stride_x_token: tl.int64` plus explicit .to(tl.int64)
                # casts), establishing no int32 token-offset limit; GB300
                # silicon at 0.27.0 confirms the two tightest formerly-vetoed
                # cells pass with SOL-sane timings (nv12 bs64 s32768: conv
                # 3-way 15.8ms for ~116GB traffic ≈ 7.3TB/s; nv96 bs8 s32768:
                # 14.3ms for the same traffic — both within the 8TB/s byte
                # model margin).
                cu = torch.arange(0, nt + 1, seq_len, dtype=torch.int32, device=device)
                idx = torch.arange(batch_size, dtype=torch.int32, device=device)
                has_init = torch.zeros(batch_size, dtype=torch.bool, device=device)
                conv_state = torch.zeros(batch_size, 3 * proj, cw - 1, dtype=dtype, device=device)
                q_cs, k_cs, v_cs = conv_state.split([proj] * 3, dim=-2)
                mixed = torch.randn(nt, 3 * proj, dtype=dtype, device=device)
                q_in, k_in, v_in = mixed.transpose(0, 1).split([proj] * 3, dim=0)

                common = {
                    "phase": "context",
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "num_tokens": nt,
                    "d_model": d_model,
                    "d_conv": d_conv,
                    "num_k_heads": num_k_heads,
                    "head_k_dim": head_k_dim,
                    "num_v_heads": num_v_heads,
                    "head_v_dim": head_v_dim,
                    "model_name": model_name,
                }

                # Serving precomputes the conv metadata once per step in the
                # KDA attention-metadata builder (KimiK3KDAMetadata subclasses
                # GDNAttentionMetadata;
                # vllm/models/kimi_k3/nvidia/kda_metadata.py @0.27.0 image era)
                # and passes it into every layer's prefill conv call
                # (vllm/models/kimi_k3/nvidia/kda.py::_prefill_conv passes
                # metadata=m). Omitting metadata takes the non-serving branch:
                # causal_conv1d_fn recomputes nums/offset lists with
                # np.repeat + nums.sum().item() (D2H) inside EVERY timed
                # call, adding ~0.25ms host overhead per conv (a flat
                # ~0.8ms/iter floor on GB300 that dominated the 0.1.dev19262
                # re-collect, conv1d SOL ~15%). Build it per-shape and pass
                # it through like collect_gdn.py does. The entry point routes
                # seq_len=1 through the decode benchmark before reaching this
                # function, matching KimiK3KDAMetadataBuilder's
                # split_decodes_and_prefills(..., decode_threshold=1).
                # Pinned serving provenance at vLLM v0.27.0:
                # - kda_metadata.py:261-279 derives num_spec_decodes and all
                #   prefill/decode request and token counts.
                # - kda_metadata.py:403-418 computes nums_dict, batch_ptr, and
                #   token_chunk_offset_ptr from non_spec_query_start_loc_cpu
                #   only when num_prefills > 0.
                # - kda_metadata.py:466-485 constructs KimiK3KDAMetadata with
                #   those fields and num_actual_tokens=m.num_actual_tokens.
                nums_dict, batch_ptr, token_chunk_offset_ptr = compute_causal_conv1d_metadata(
                    torch.arange(0, nt + 1, seq_len, dtype=torch.int32),
                    device=device,
                )
                conv_metadata = GDNAttentionMetadata(
                    num_prefills=batch_size,
                    num_prefill_tokens=nt,
                    num_decodes=0,
                    num_decode_tokens=0,
                    num_spec_decodes=0,
                    num_spec_decode_tokens=0,
                    num_actual_tokens=nt,
                    nums_dict=nums_dict,
                    batch_ptr=batch_ptr,
                    token_chunk_offset_ptr=token_chunk_offset_ptr,
                )

                def run_conv_qkv3():
                    for x, w, cs in ((q_in, q_w, q_cs), (k_in, k_w, k_cs), (v_in, v_w, v_cs)):
                        causal_conv1d_fn(
                            x,
                            w,
                            None,
                            conv_states=cs,
                            query_start_loc=cu,
                            cache_indices=idx,
                            has_initial_state=has_init,
                            activation="silu",
                            metadata=conv_metadata,
                        )

                with benchmark_with_power(
                    device=device,
                    kernel_func=run_conv_qkv3,
                    num_warmups=NUM_WARMUPS,
                    num_runs=NUM_RUNS,
                    # With the metadata handed in, production launches this op
                    # eagerly but back-to-back in a deep queue, so the row must
                    # hold GPU service time (same as the GDN precedent in
                    # collect_gdn.py — repeat graph replay, not a sync-bounded
                    # eager loop).
                    repeat_n=10,
                ) as results:
                    _log(
                        common,
                        results["latency_ms"],
                        "causal_conv1d_fn_qkv3",
                        perf_filename,
                        vllm_version,
                        device,
                        results["power_stats"],
                    )

                q = torch.randn(1, nt, nh, hd, dtype=dtype, device=device)
                k = torch.randn(1, nt, nh, hd, dtype=dtype, device=device)
                v = torch.randn(1, nt, nh, hd, dtype=dtype, device=device)
                raw_g = torch.randn(1, nt, nh, hd, dtype=dtype, device=device)
                raw_beta = torch.randn(1, nt, nh, dtype=dtype, device=device)
                a_log = torch.zeros(nh, dtype=torch.float32, device=device)
                dt_bias = 0.1 * torch.randn(nh * hd, dtype=torch.float32, device=device)
                init_state = torch.zeros(batch_size, nh, hd, hd, dtype=torch.float32, device=device)

                if prefill_source == "flashkda_fwd":

                    def run_prefill():
                        flashkda_prefill(
                            q,
                            k,
                            v,
                            raw_g,
                            raw_beta,
                            a_log,
                            dt_bias,
                            KDA_LOWER_BOUND,
                            init_state,
                            cu,
                        )

                else:

                    def run_prefill():
                        chunk_kda_with_fused_gate(
                            q,
                            k,
                            v,
                            raw_g,
                            raw_beta,
                            a_log,
                            dt_bias,
                            initial_state=init_state,
                            output_final_state=True,
                            lower_bound=KDA_LOWER_BOUND,
                            use_qk_l2norm_in_kernel=True,
                            cu_seqlens=cu,
                        )

                with benchmark_with_power(
                    device=device, kernel_func=run_prefill, num_warmups=NUM_WARMUPS, num_runs=NUM_RUNS, repeat_n=1
                ) as results:
                    _log(
                        common,
                        results["latency_ms"],
                        prefill_source,
                        perf_filename,
                        vllm_version,
                        device,
                        results["power_stats"],
                    )
                ok += 1
            except Exception as e:
                err += 1
                failures.append(f"batch_size={batch_size} seq_len={seq_len}: {type(e).__name__}: {e}")
                print(f"  Error at batch_size={batch_size}, seq_len={seq_len}: {e}")
                continue
            finally:
                # Drop the per-iteration tensor references before empty_cache
                # (same pattern as the sglang kda collector's finally blocks).
                cu = idx = has_init = conv_state = q_cs = k_cs = v_cs = None
                mixed = q_in = k_in = v_in = q = k = v = raw_g = raw_beta = None
                a_log = dt_bias = init_state = None
                _cleanup("context")

    summary = f"ok={ok} error={err} skip=0"
    print(f"KDA context summary: {summary}")
    if err or ok == 0:
        raise RuntimeError(
            f"vLLM KDA context collection failed strict completeness: {summary}; "
            f"failed cells: {_format_failures(failures)}"
        )


def run_kda_generation_benchmark(
    d_model,
    d_conv,
    num_k_heads,
    head_k_dim,
    num_v_heads,
    head_v_dim,
    batch_size_list,
    model_name,
    perf_filename,
    vllm_version,
    row_phase="generation",
    device="cuda:0",
):
    """Generation (decode): the CUDA fused_kda_decode fast path (probed like
    serving) plus the packed conv-update + Triton recurrence fallback pair
    (which is also the serving path under speculative decoding).

    ``row_phase`` remains ``context`` for one-token cells originating from the
    shared context grid; the recorded kernels still follow vLLM's decode path.
    """
    if row_phase not in {"context", "generation"}:
        raise ValueError(f"Unsupported KDA decode row phase: {row_phase}")

    import vllm._custom_ops as ops
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
    from vllm.models.kimi_k3.nvidia.kda import is_fused_kda_decode_supported
    from vllm.models.kimi_k3.nvidia.ops.third_party.kda import fused_recurrent_kda_packed_decode

    device = torch.device(device)
    torch.cuda.set_device(device)
    dtype = torch.bfloat16
    assert num_k_heads == num_v_heads and head_k_dim == head_v_dim
    nh, hd, cw = num_v_heads, head_v_dim, d_conv
    dim = nh * hd

    fused_ok = is_fused_kda_decode_supported(nh, hd, cw, num_spec=0, input_dtype=dtype, conv_state_dtype=dtype)
    conv_weight = torch.randn(3 * dim, cw, dtype=torch.float32, device=device)
    fused_weight = conv_weight.reshape(3, dim, cw).transpose(1, 2).contiguous()
    norm_weight = torch.randn(hd, dtype=torch.float32, device=device)
    ok = err = 0
    failures: list[str] = []

    for batch_size in batch_size_list:
        try:
            nb = batch_size
            x = torch.randn(nb, 3 * dim, dtype=dtype, device=device)
            # KDA conv cache: [slots, 3*dim, cw-1] with stride(1)==1 ("SD" layout)
            conv_state = torch.zeros(nb, cw - 1, 3 * dim, dtype=dtype, device=device).transpose(1, 2)
            raw_g = torch.randn(1, nb, nh, hd, dtype=dtype, device=device)
            raw_beta = torch.randn(1, nb, nh, dtype=dtype, device=device)
            a_log = torch.zeros(nh, dtype=torch.float32, device=device)
            dt_bias_hd = 0.1 * torch.randn(nh, hd, dtype=torch.float32, device=device)
            state = torch.zeros(nb, nh, hd, hd, dtype=torch.float32, device=device)
            idx = torch.arange(nb, dtype=torch.int32, device=device)
            output_gate = torch.randn(nb, nh, hd, dtype=dtype, device=device)
            conv_out = torch.empty_like(x)

            common = {
                "phase": row_phase,
                "batch_size": nb,
                "seq_len": 1,
                "num_tokens": nb,
                "d_model": d_model,
                "d_conv": d_conv,
                "num_k_heads": num_k_heads,
                "head_k_dim": head_k_dim,
                "num_v_heads": num_v_heads,
                "head_v_dim": head_v_dim,
                "model_name": model_name,
            }

            if fused_ok:

                def run_fused_decode():
                    ops.fused_kda_decode(
                        x=x,
                        weight=fused_weight,
                        bias=None,
                        conv_state=conv_state,
                        raw_g=raw_g,
                        raw_beta=raw_beta,
                        A_log=a_log,
                        dt_bias=dt_bias_hd.reshape(-1),
                        state_indices=idx,
                        state=state,
                        lower_bound=KDA_LOWER_BOUND,
                        output_gate=output_gate,
                        norm_weight=norm_weight,
                        norm_eps=1e-5,
                    )

                with benchmark_with_power(
                    device=device,
                    kernel_func=run_fused_decode,
                    num_warmups=NUM_WARMUPS,
                    num_runs=NUM_RUNS,
                    repeat_n=1,
                ) as results:
                    _log(
                        common,
                        results["latency_ms"],
                        "fused_kda_decode",
                        perf_filename,
                        vllm_version,
                        device,
                        results["power_stats"],
                    )

            def run_conv_update():
                causal_conv1d_update(
                    x,
                    conv_state,
                    conv_weight,
                    None,
                    activation="silu",
                    conv_state_indices=idx,
                    validate_data=True,
                    out=conv_out,
                )

            with benchmark_with_power(
                device=device, kernel_func=run_conv_update, num_warmups=NUM_WARMUPS, num_runs=NUM_RUNS, repeat_n=1
            ) as results:
                _log(
                    common,
                    results["latency_ms"],
                    "causal_conv1d_update",
                    perf_filename,
                    vllm_version,
                    device,
                    results["power_stats"],
                )

            def run_packed_decode():
                fused_recurrent_kda_packed_decode(
                    mixed_qkv=x,
                    raw_g=raw_g,
                    raw_beta=raw_beta,
                    A_log=a_log,
                    dt_bias=dt_bias_hd,
                    lower_bound=KDA_LOWER_BOUND,
                    initial_state=state,
                    state_indices=idx,
                )

            with benchmark_with_power(
                device=device, kernel_func=run_packed_decode, num_warmups=NUM_WARMUPS, num_runs=NUM_RUNS, repeat_n=1
            ) as results:
                _log(
                    common,
                    results["latency_ms"],
                    "fused_recurrent_kda_packed_decode",
                    perf_filename,
                    vllm_version,
                    device,
                    results["power_stats"],
                )
            ok += 1
        except Exception as e:
            err += 1
            failures.append(f"batch_size={batch_size}: {type(e).__name__}: {e}")
            print(f"  Error at batch_size={batch_size}: {e}")
            continue
        finally:
            _cleanup("generation")

    summary = f"ok={ok} error={err} skip=0"
    print(f"KDA {row_phase} decode-path summary: {summary}")
    if err or ok == 0:
        raise RuntimeError(
            f"vLLM KDA {row_phase} decode-path collection failed strict completeness: {summary}; "
            f"failed cells: {_format_failures(failures)}"
        )


def run_kda_verify_benchmark(
    d_model,
    d_conv,
    num_k_heads,
    head_k_dim,
    num_v_heads,
    head_v_dim,
    batch_size_list,
    draft_token_list,
    model_name,
    perf_filename,
    vllm_version,
    device="cuda:0",
):
    """Speculative target-verify: spec-form packed conv update + the
    fused_recurrent_kda chain-verify kernel with per-draft-token state
    checkpointing (2-hd ssm_state_indices). num_accepted_tokens is set to the
    full draft width (cost is token-count dominated)."""
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
    from vllm.models.kimi_k3.nvidia.ops.third_party.kda import fused_recurrent_kda

    device = torch.device(device)
    torch.cuda.set_device(device)
    dtype = torch.bfloat16
    assert num_k_heads == num_v_heads and head_k_dim == head_v_dim
    nh, hd, cw = num_v_heads, head_v_dim, d_conv
    dim = nh * hd

    conv_weight = torch.randn(3 * dim, cw, dtype=torch.float32, device=device)
    ok = err = 0
    failures: list[str] = []

    for batch_size in batch_size_list:
        for ns in draft_token_list:
            nb = batch_size
            nt = nb * ns
            try:
                x = torch.randn(nt, 3 * dim, dtype=dtype, device=device)
                # spec conv cache carries num_spec extra columns
                conv_state = torch.zeros(nb, cw - 1 + ns, 3 * dim, dtype=dtype, device=device).transpose(1, 2)
                cu = torch.arange(0, nt + 1, ns, dtype=torch.int32, device=device)
                idx = torch.arange(nb, dtype=torch.int32, device=device)
                accepted = torch.full((nb,), ns, dtype=torch.int32, device=device)
                conv_out = torch.empty_like(x)

                q = torch.randn(1, nt, nh, hd, dtype=dtype, device=device)
                k = torch.randn(1, nt, nh, hd, dtype=dtype, device=device)
                v = torch.randn(1, nt, nh, hd, dtype=dtype, device=device)
                raw_g = torch.randn(1, nt, nh, hd, dtype=dtype, device=device)
                raw_beta = torch.randn(1, nt, nh, dtype=dtype, device=device)
                a_log = torch.zeros(nh, dtype=torch.float32, device=device)
                dt_bias_hd = 0.1 * torch.randn(nh, hd, dtype=torch.float32, device=device)
                state = torch.zeros(nb * (ns + 1), nh, hd, hd, dtype=torch.float32, device=device)
                ssm_idx = torch.arange(nb * ns, dtype=torch.int32, device=device).view(nb, ns)
                out = torch.empty(1, nt, nh, hd, dtype=dtype, device=device)

                common = {
                    "phase": "verify",
                    "batch_size": nb,
                    "seq_len": ns,
                    "num_tokens": nt,
                    "d_model": d_model,
                    "d_conv": d_conv,
                    "num_k_heads": num_k_heads,
                    "head_k_dim": head_k_dim,
                    "num_v_heads": num_v_heads,
                    "head_v_dim": head_v_dim,
                    "model_name": model_name,
                }

                def run_conv_update_spec():
                    causal_conv1d_update(
                        x,
                        conv_state,
                        conv_weight,
                        None,
                        activation="silu",
                        conv_state_indices=idx,
                        num_accepted_tokens=accepted,
                        query_start_loc=cu,
                        max_query_len=ns,
                        validate_data=False,
                        out=conv_out,
                    )

                with benchmark_with_power(
                    device=device,
                    kernel_func=run_conv_update_spec,
                    num_warmups=NUM_WARMUPS,
                    num_runs=NUM_RUNS,
                    repeat_n=1,
                ) as results:
                    _log(
                        common,
                        results["latency_ms"],
                        "causal_conv1d_update",
                        perf_filename,
                        vllm_version,
                        device,
                        results["power_stats"],
                    )

                def run_verify():
                    fused_recurrent_kda(
                        q=q,
                        k=k,
                        v=v,
                        raw_g=raw_g,
                        raw_beta=raw_beta,
                        A_log=a_log,
                        dt_bias=dt_bias_hd,
                        lower_bound=KDA_LOWER_BOUND,
                        initial_state=state,
                        cu_seqlens=cu,
                        ssm_state_indices=ssm_idx,
                        num_accepted_tokens=accepted,
                        out=out,
                    )

                with benchmark_with_power(
                    device=device, kernel_func=run_verify, num_warmups=NUM_WARMUPS, num_runs=NUM_RUNS, repeat_n=1
                ) as results:
                    _log(
                        common,
                        results["latency_ms"],
                        "fused_recurrent_kda",
                        perf_filename,
                        vllm_version,
                        device,
                        results["power_stats"],
                    )
                ok += 1
            except Exception as e:
                err += 1
                failures.append(f"batch_size={batch_size} draft_tokens={ns}: {type(e).__name__}: {e}")
                print(f"  Error at batch_size={batch_size}, draft_tokens={ns}: {e}")
                continue
            finally:
                _cleanup("verify")

    summary = f"ok={ok} error={err} skip=0"
    print(f"KDA verify summary: {summary}")
    if err or ok == 0:
        raise RuntimeError(
            f"vLLM KDA verify collection failed strict completeness: {summary}; "
            f"failed cells: {_format_failures(failures)}"
        )


def run_kda_torch(
    phase,
    d_model,
    d_conv,
    num_k_heads,
    head_k_dim,
    num_v_heads,
    head_v_dim,
    batch_size_list,
    seq_len_list,
    model_name,
    *,
    perf_filename,
    device="cuda:0",
):
    """Main entry point: routes phases and reports the installed vLLM version."""
    from vllm.version import __version__ as vllm_version

    kwargs = dict(
        d_model=d_model,
        d_conv=d_conv,
        num_k_heads=num_k_heads,
        head_k_dim=head_k_dim,
        num_v_heads=num_v_heads,
        head_v_dim=head_v_dim,
        model_name=model_name,
        perf_filename=perf_filename,
        vllm_version=vllm_version,
        device=device,
    )
    if phase == "context":
        if seq_len_list is None:
            raise ValueError("vLLM KDA context collection requires seq_len_list")
        if not seq_len_list:
            raise ValueError("vLLM KDA context collection requires at least one sequence length")
        if any(seq_len < 1 for seq_len in seq_len_list):
            raise ValueError(f"vLLM KDA context sequence lengths must be positive: {seq_len_list}")

        # vLLM 0.27.0 classifies one-token non-spec batches as pure decodes
        # (kda_metadata.py:274-278) and invokes causal_conv1d_update plus
        # fused_recurrent_kda_packed_decode (kda.py:733-760). Keep the shared
        # context grid identity, but dispatch that cell through the serving
        # decode kernels instead of manufacturing prefill metadata.
        if 1 in seq_len_list:
            run_kda_generation_benchmark(batch_size_list=batch_size_list, row_phase="context", **kwargs)

        prefill_seq_len_list = [seq_len for seq_len in seq_len_list if seq_len > 1]
        if prefill_seq_len_list:
            run_kda_context_benchmark(
                batch_size_list=batch_size_list,
                seq_len_list=prefill_seq_len_list,
                **kwargs,
            )
    elif phase == "generation":
        run_kda_generation_benchmark(batch_size_list=batch_size_list, **kwargs)
    elif phase == "verify":
        run_kda_verify_benchmark(batch_size_list=batch_size_list, draft_token_list=seq_len_list, **kwargs)
    else:
        raise ValueError(f"Unknown phase: {phase}")

    return WORKER_RESTART


if __name__ == "__main__":
    import sys

    from vllm.version import __version__ as _v

    from collector.registry_types import PerfFile

    print(f"KDA Collector - vLLM {_v}")
    print(f"SM Version: {get_sm_version()}")
    print(f"Device: {torch.cuda.get_device_name()}")

    last = 0
    cases = get_kda_test_cases()
    print(f"Total test cases: {len(cases)}")
    for i, tc in enumerate(cases):
        print(f"\n[{i + 1}/{len(cases)}] {tc[9]} - {tc[0]} heads={tc[5]}")
        last = run_kda_torch(
            phase=tc[0],
            d_model=tc[1],
            d_conv=tc[2],
            num_k_heads=tc[3],
            head_k_dim=tc[4],
            num_v_heads=tc[5],
            head_v_dim=tc[6],
            batch_size_list=tc[7],
            seq_len_list=tc[8],
            model_name=tc[9],
            perf_filename=PerfFile.KDA,
        )
    sys.exit(last)
