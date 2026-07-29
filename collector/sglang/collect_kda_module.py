# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Module-level KDA collector for AIConfigurator — SGLang backend.

Measures the WHOLE KimiK3DeltaAttention layer module (in_proj/qkvbfg
projections -> causal conv -> delta-rule recurrence -> gated RMSNorm ->
out_proj) instead of its constituent kernels, per the design in
docs/perf_database/linear-attention-module-design.md. Rationale: serving
fuses across the kernel-level op boundaries (kda_fused_decode folds
conv+recurrence+onorm; the qkvbfg path issues tiny side-stream GEMVs), so
the kernel-sum drifts from layer truth — on B300 the measured module decode
is ~5x the bare recurrence kernel; the delta (projection GEMMs + module
elementwise) is what these rows own.

Dispatch fidelity comes from CONSTRUCTION, not replication: a shrunk 4-layer
Kimi-K3 is built through sglang's own ModelRunner (dummy load), real
extend/decode batches are driven exactly like ``sglang.benchmark.one_batch``,
and a forward-pre-hook captures the target module's live (args, kwargs) and
active ForwardContext during one full ``model_runner.forward``. The module
is then replayed under ``forward_context(...)`` inside a CUDA graph captured
under the framework's own ``model_capture_mode()`` — whatever the backend
dispatches (fused CuTeDSL decode on covered shapes, Triton packed decode,
chunk_kda extend) is what the row records. Both phases are graphed: the
eager replay is python-launch-bound (742 vs 225 us/layer at seq 1024 on
B300) and serving hides that CPU time by running ahead of the GPU.

Shrunk-config choices (all structural, none affect the measured module):
  - 4 layers = 3 KDA + 1 MLA. ``is_kda_layer(layer_idx)`` checks
    ``(layer_idx + 1) in kda_layers`` — 1-indexed lists
    (srt/configs/kimi_linear.py:156-159 @ kimi-k3 branch).
  - The per-rank TP shard is realized at tp_size=1 by shrinking
    ``linear_attn_config.num_heads`` to heads/TP; in_proj/out_proj then have
    exactly the per-rank GEMM shapes and the 12-head shard is covered by the
    fused decode kernel just as in TP8 serving.
  - MLA ``num_attention_heads``/``num_key_value_heads`` are pinned to 12
    regardless of the KDA shard: the MLA layer is NOT measured, only needs
    to run, and the trtllm_mla decode kernel rejects the full 96 q-heads
    (computeCtaAndClusterConfig: numHeadsQ not supported).
  - ``num_experts`` 896 -> 64 (MoE not measured; num_expert_group=1 in the
    K3 config so any shrink is group-valid).

Under TP the o_proj all-reduce stays OUTSIDE the module (k3_ar_fusion comm
lane); at tp_size=1 there is no AR, so the rows are AR-free by construction.

Scheduler-side globals the direct-ModelRunner path must replicate
(benchmark/one_batch.py:878-880 does the same): ``initialize_moe_config`` +
``initialize_fp8_gemm_config`` + ``initialize_fp4_gemm_config``. Without the
first, the ServerArgs-resolved flashinfer_mxfp4 MoE backend is silently lost
and MoE falls back to the triton runner (no sm103 kernel image).

Phases:
  - context: bs x isl grid from cases/base_ops/kda.yaml, one graphed
    forward_extend replay per cell (kernel_source "kda_module[chunk_kda]").
  - generation: bs grid, decode state prepared by a real 128-token extend
    (decode cost is state-size-invariant; 129 avoids mamba_track flush
    boundaries), kernel_source records the OBSERVED path via the
    attempt-and-verify onorm stash: "kda_module[kda_fused_decode]" when
    ``_k3_onorm_consumed`` is set, else "kda_module[triton_packed_decode]".
  - verify (DSPARK target-verify): NOT collected yet — needs spec_info
    ForwardBatch plumbing; the fused verify kernel is covered at kernel
    level (kda_perf) and per-key consumer routing bridges the gap.

Output: linear_attn_module_perf.txt — same column layout as kda_perf
(phase, batch_size, seq_len, num_tokens, d_model, d_conv, num_k_heads,
head_k_dim, num_v_heads, head_v_dim, model_name, latency), op_name
"kda_module".

Runbook (kimi-k3 branch image, one geometry per process — the script
subprocesses itself when --heads lists several):

    python3 collector/sglang/collect_kda_module.py \
        --model-src /model --heads 12 24 48 96
"""

# The kimi-k3 branch build (https://github.com/sgl-project/sglang/tree/kimi-k3)
# reports 0.5.16; KDA and the fused decode kernel do not exist in stock
# sglang releases yet.
__compat__ = "sglang==0.5.16"

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
from array import array
from types import SimpleNamespace

import numpy as np
import torch
import yaml

try:
    from collector.helper import benchmark_with_power, log_perf
    from collector.registry_types import PerfFile
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from registry_types import PerfFile

    from helper import benchmark_with_power, log_perf

aic_debug = int(os.getenv("aic_kda_module_debug", "0"))  # noqa: SIM112

# 96 KDA heads in Kimi-K3; per-rank shard = 96 / tp for tp in (8, 4, 2, 1).
K3_SHARD_HEADS = (12, 24, 48, 96)
# Decode-state prep length: real tokens extended before the decode capture.
# Chosen so the captured step (position 129) is not a mamba_track interval
# boundary — a boundary step force-flushes SSM state and would be replayed
# into every graph iteration (a 1/interval-of-steps cost, not the steady state).
_DECODE_PREP_ISL = 128


def _base_grid() -> dict:
    """Batch/seq sweeps come from the declared base grid, not local copies."""
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "cases",
        "base_ops",
        "kda.yaml",
    )
    with open(path) as f:
        return yaml.safe_load(f)["common_case_values"]["kda"]


def _doctor_model_config(src_dir: str, dst_dir: str, kda_heads: int) -> str:
    """Copy the non-weight model files and shrink to the 4-layer per-rank-shard
    model described in the module docstring."""
    os.makedirs(dst_dir, exist_ok=True)
    for name in os.listdir(src_dir):
        if name.endswith((".safetensors", ".bin", ".pt", ".gguf")) or name == "model.safetensors.index.json":
            continue
        src = os.path.join(src_dir, name)
        if os.path.isfile(src):
            shutil.copy(src, os.path.join(dst_dir, name))
    cfg_path = os.path.join(dst_dir, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    tc = cfg["text_config"]
    tc["num_hidden_layers"] = 4
    tc["linear_attn_config"]["kda_layers"] = [1, 2, 3]  # 1-indexed: layers 0..2
    tc["linear_attn_config"]["full_attn_layers"] = [4]  # layer 3 is MLA
    tc["linear_attn_config"]["num_heads"] = kda_heads
    tc["num_attention_heads"] = 12
    tc["num_key_value_heads"] = 12
    tc["num_experts"] = 64
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)
    return dst_dir


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _load_model_runner(model_path: str):
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.distributed.parallel_state_wrapper import ParallelState
    from sglang.srt.entrypoints.engine import _set_envs_and_config
    from sglang.srt.layers.moe import initialize_moe_config
    from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
    from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.server_args import ServerArgs

    server_args = ServerArgs(
        model_path=model_path,
        dtype="auto",
        device="cuda",
        load_format="dummy",
        tp_size=1,
        trust_remote_code=True,
        disable_radix_cache=True,
        # The module benchmark captures its own CUDA graph; keep sglang's
        # serving-level graph runners off so hooks fire during capture.
        disable_cuda_graph=True,
        attention_backend="trtllm_mla",
        # bs sweep goes to 1024; alloc_req_slots exposes
        # max_running_requests - 1 usable slots.
        max_running_requests=1100,
        # The sglang-derived default fraction sizes the KV/state pool for a
        # full 93-layer serving model; on the shrunk 4-layer model it eats
        # ~93% of the device and large module captures OOM on activations
        # (B300 evidence: 131k-token context cells allocate ~14 GiB of
        # activations, 262k ~28 GiB, plus ~9 GiB of CUDA-graph private
        # pools). 0.60 still leaves the pool far above the largest context
        # cell (256k pool tokens after the int32 conv guard) while keeping
        # ~80 GB of activation headroom.
        mem_fraction_static=0.60,
    )
    _set_envs_and_config(server_args)
    initialize_moe_config(server_args)
    initialize_fp8_gemm_config(server_args)
    initialize_fp4_gemm_config(server_args)
    model_config = ModelConfig.from_server_args(server_args)
    runner = ModelRunner(
        model_config=model_config,
        mem_fraction_static=server_args.mem_fraction_static,
        gpu_id=0,
        ps=ParallelState.trivial(gpu_id=0),
        nccl_port=_pick_free_port(),
        server_args=server_args,
    )
    runner.alloc_memory_pool()
    runner.init_attention_backends()
    # Required even with disable_cuda_graph: sets the *_cuda_graph_runner
    # attributes _forward_raw reads (one_batch.load_model calls it too).
    runner.init_cuda_graphs()
    return runner


def _make_reqs(batch_size: int, input_len: int):
    # Mirrors sglang.benchmark.one_batch.prepare_synthetic_inputs_for_latency_test.
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    input_ids = np.random.randint(0, 10000, (batch_size, input_len), dtype=np.int32)
    sampling_params = SamplingParams(temperature=0, max_new_tokens=8)
    reqs = []
    for i in range(batch_size):
        req = Req(
            rid=i,
            origin_input_text="",
            origin_input_ids=array("q", input_ids[i]),
            sampling_params=sampling_params,
        )
        req.full_untruncated_fill_ids = req.origin_input_ids
        req.logprob_start_len = -1
        req.set_extend_range(len(req.prefix_indices), len(req.origin_input_ids))
        reqs.append(req)
    return reqs


class _TreeCacheNamespace(SimpleNamespace):
    # Mirrors sglang.benchmark.one_batch.TreeCacheNamespace (allocation-only
    # stand-in; radix caching is disabled).
    def supports_swa(self):
        return False

    def supports_mamba(self):
        return False

    def is_chunk_cache(self):
        return False

    def is_tree_cache(self):
        return not self.is_chunk_cache()

    def evict(self, params):
        pass


def _make_extend_batch(reqs, model_runner):
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    # One-shot benchmark batches are never released through the scheduler;
    # reclaim the pools each cell or the sweep exhausts req slots/tokens
    # (same as collect_dsv4_attn._build_forward_batch). Decode cells reuse
    # the extend state built in the SAME cell, so clearing here is safe.
    model_runner.req_to_token_pool.clear()
    model_runner.token_to_kv_pool_allocator.clear()

    tree_cache = _TreeCacheNamespace(
        page_size=model_runner.server_args.page_size,
        device=model_runner.device,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
    )
    batch = ScheduleBatch.init_new(
        reqs=reqs,
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
        tree_cache=tree_cache,
        model_config=model_runner.model_config,
        enable_overlap=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
    )
    batch.prepare_for_extend()
    if batch.input_ids is None and getattr(batch, "prefill_input_ids_cpu", None) is not None:
        batch.input_ids = batch.prefill_input_ids_cpu.to(batch.device, non_blocking=True)
        batch.prefill_input_ids_cpu = None
    fb = ForwardBatch.init_new(batch, model_runner, return_hidden_states_before_norm=False)
    return batch, fb


def _make_decode_batch(batch, model_runner):
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

    batch.input_ids = torch.randint(0, 10000, (batch.batch_size(),), dtype=torch.int64, device=batch.device)
    batch.prepare_for_decode()
    return ForwardBatch.init_new(batch, model_runner, return_hidden_states_before_norm=False)


def _find_kda_module(runner, layer_idx: int = 1):
    for name, mod in runner.model.named_modules():
        if type(mod).__name__ == "KimiK3DeltaAttention" and f".{layer_idx}." in f".{name}.":
            return name, mod
    raise RuntimeError("no KimiK3DeltaAttention module found in the doctored model")


def _capture_module_graph(runner, module, fb, tag: str):
    """Run one full model forward, hook the target module's live call, then
    capture the module call into a CUDA graph under the framework's own
    capture flag. Returns (replay_fn, fused_decode_engaged).

    Graph capture is mandatory: an uncapturable module is a collector bug or
    a framework change, and the eager fallback measures python launch
    overhead, not the layer — raise instead of degrading silently.
    """
    from sglang.srt.model_executor.forward_context import (
        forward_context,
        get_forward_context,
    )
    from sglang.srt.model_executor.runner_utils.capture_mode import model_capture_mode

    captured = {}

    def pre_hook(mod, args, kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        # RadixLinearAttention resolves its backend through the global
        # forward context the runner publishes per call; grab it while live.
        captured["ctx"] = get_forward_context()

    handle = module.register_forward_pre_hook(pre_hook, with_kwargs=True)
    with torch.no_grad():
        runner.forward(fb)
    handle.remove()
    if "args" not in captured:
        raise RuntimeError(f"{tag}: KDA module was not invoked during model_runner.forward")

    ctx = captured["ctx"]
    # model_capture_mode() is what serving's decode graph runner sets around
    # BOTH warmup and capture (decode_cuda_graph_runner.py:412 -> capture()):
    # the capture-gated branches (side-stream qkvbfg GEMVs) must run their
    # first-call kernel preparation EAGERLY here — a first call inside the
    # graph capture is a forbidden CUDA call and fails the whole capture.
    with torch.no_grad(), forward_context(ctx), model_capture_mode():
        for _ in range(3):
            module(*captured["args"], **captured["kwargs"])
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        # thread_local: sglang runs background threads (memory watchdog) whose
        # CUDA calls invalidate a default global-mode capture into an empty
        # graph + stuck-active capture state; the framework's own hand-built
        # capture does the same (multi_layer_draft_forward_cg.py:88).
        with torch.cuda.graph(graph, capture_error_mode="thread_local"):
            module(*captured["args"], **captured["kwargs"])

    fused = bool(getattr(module.attn, "_k3_onorm_consumed", False))

    def replay():
        graph.replay()

    return replay, fused


def _shrunk_model_dir(model_src: str, kda_heads: int) -> str:
    cache_root = os.environ.get("AIC_KDA_MODULE_TMP") or "/tmp/aic_kda_module"
    dst = os.path.join(cache_root, f"k3_shrunk_h{kda_heads}")
    return _doctor_model_config(model_src, dst, kda_heads)


def _geometry_from_config(model_dir: str) -> dict:
    with open(os.path.join(model_dir, "config.json")) as f:
        tc = json.load(f)["text_config"]
    lac = tc["linear_attn_config"]
    return {
        "d_model": tc["hidden_size"],
        "d_conv": lac["short_conv_kernel_size"],
        "num_k_heads": lac["num_heads"],
        "head_k_dim": lac["head_dim"],
        "num_v_heads": lac["num_heads"],
        "head_v_dim": lac["head_dim"],
    }


def _log_row(
    *,
    geometry: dict,
    phase: str,
    batch_size: int,
    seq_len: int,
    num_tokens: int,
    model_name: str,
    kernel_source: str,
    results,
    sglang_version: str,
    perf_filename: str,
):
    row = {
        "phase": phase,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "num_tokens": num_tokens,
        **geometry,
        "model_name": model_name,
        "latency": results["latency_ms"],
    }
    if not log_perf(
        item_list=[row],
        framework="SGLang",
        version=sglang_version,
        device_name=torch.cuda.get_device_name(0),
        op_name="kda_module",
        kernel_source=kernel_source,
        perf_filename=perf_filename,
        power_stats=results["power_stats"],
    ):
        raise RuntimeError(f"failed to persist SGLang kda_module {phase} row to {perf_filename}")


def run_kda_module_benchmark(
    model_src: str,
    kda_heads: int,
    model_name: str,
    perf_filename: str,
    device: str = "cuda:0",
):
    import sglang

    torch.cuda.set_device(device)
    sglang_version = sglang.__version__
    grid = _base_grid()

    model_dir = _shrunk_model_dir(model_src, kda_heads)
    geometry = _geometry_from_config(model_dir)
    runner = _load_model_runner(model_dir)
    module_name, module = _find_kda_module(runner)
    conv_channels = 3 * geometry["num_k_heads"] * geometry["head_k_dim"]
    print(f"[kda-module] measuring {module_name} (heads={kda_heads}, d_model={geometry['d_model']})", flush=True)

    successful_points = 0
    failed_points = 0

    # --- generation (decode) phase FIRST: the top of the context token range
    # can hit CUDA-context-fatal kernel faults (see FIXME(kernel-limit) below)
    # that kill everything after them; decode rows must not be collateral ----
    for batch_size in grid["generation_batch_sizes"]:
        replay = None
        try:
            if aic_debug:
                print(f"  generation bs={batch_size}", flush=True)
            # The prep extend is instrumental (decode cost is state-size
            # invariant); shrink it when bs x isl would cross the conv int32
            # bound that is fatal at large batch (silicon: h48 bs=1024 and
            # h96 bs=512 both died at prep_tokens*conv_channels = 2.4e9).
            prep_isl = _DECODE_PREP_ISL
            while prep_isl > 8 and batch_size * prep_isl * conv_channels >= 2**31:
                prep_isl //= 2
            batch, fb_ext = _make_extend_batch(_make_reqs(batch_size, prep_isl), runner)
            with torch.no_grad():
                runner.forward(fb_ext)  # real prefill so conv/SSM state exists
            fb_dec = _make_decode_batch(batch, runner)
            replay, fused = _capture_module_graph(runner, module, fb_dec, f"generation bs={batch_size}")
            with benchmark_with_power(
                device=torch.device(device),
                kernel_func=replay,
                num_warmups=5,
                num_runs=30,
                repeat_n=1,
                # kernel_func is already a captured-graph replay; the helper's
                # own capture would replay-inside-capture and abort.
                use_cuda_graph=False,
            ) as results:
                _log_row(
                    geometry=geometry,
                    phase="generation",
                    batch_size=batch_size,
                    seq_len=1,
                    num_tokens=batch_size,
                    model_name=model_name,
                    kernel_source=("kda_module[kda_fused_decode]" if fused else "kda_module[triton_packed_decode]"),
                    results=results,
                    sglang_version=sglang_version,
                    perf_filename=perf_filename,
                )
            successful_points += 1
        except Exception as e:
            failed_points += 1
            print(f"  Error at generation batch_size={batch_size}: {e}", flush=True)
            if aic_debug:
                import traceback

                traceback.print_exc()
        finally:
            replay = None  # drops the closure holding the graph
            torch.cuda.empty_cache()

    # --- context (extend) phase: guard-raising cells first (their classified
    # failure records cost nothing and must not be lost to a later fatal
    # cell), then real cells ascending by token count so a capacity failure
    # or the fatal top-of-range fault loses only the tail -------------------
    def _passes_guards(cell):
        tokens = cell[0] * cell[1]
        return tokens * conv_channels < 2**31 and tokens <= int(runner.max_total_num_tokens)

    ctx_cells = sorted(
        ((bs, sl) for bs in grid["context_batch_sizes"] for sl in grid["context_sequence_lengths"]),
        key=lambda c: (_passes_guards(c), c[0] * c[1]),
    )
    for batch_size, seq_len in ctx_cells:
        total_tokens = batch_size * seq_len
        replay = None
        try:
            # Same int32 token-offset overflow as the kernel-level collector
            # (collect_kda.py: causal_conv1d_triton.py:373-379 pointer math
            # spans total_tokens * conv_channels; silicon-confirmed on SM90/
            # SM100). The module drives the same kernel via the backend, and
            # the failure is a fatal illegal address — raise before capture.
            # FIXME(kernel-limit): the FULL-LAYER extend additionally crashed
            # with cudaErrorIllegalAddress at total_tokens=262144 on the
            # 12-head shard (B300/SM103, kimi-k3 image
            # lmsysorg/sglang@sha256:81a9c006..., 2026-07-29) — BELOW this
            # conv guard's bound, so another kernel in the layer path breaks
            # first (unidentified; MLA KV write / chunk internals suspected).
            # Unverified against framework source; the affected top-of-range
            # cells fail fatally at runtime and, with generation running
            # first and cells ascending, only that band is lost.
            if total_tokens * conv_channels >= 2**31:
                raise ValueError(
                    "SGLang causal_conv1d Triton kernel int32 token-offset overflow: "
                    f"total_tokens={total_tokens} * conv_channels={conv_channels} >= 2**31"
                )
            max_tokens = int(runner.max_total_num_tokens)
            if total_tokens > max_tokens:
                raise ValueError(
                    f"extend cell needs {total_tokens} pool tokens > sglang-derived "
                    f"max_total_num_tokens={max_tokens} (framework capacity, not a guess)"
                )
            if aic_debug:
                print(f"  context bs={batch_size} isl={seq_len}", flush=True)
            _, fb = _make_extend_batch(_make_reqs(batch_size, seq_len), runner)
            replay, _ = _capture_module_graph(runner, module, fb, f"context bs={batch_size} isl={seq_len}")
            with benchmark_with_power(
                device=torch.device(device),
                kernel_func=replay,
                num_warmups=3,
                num_runs=10,
                repeat_n=1,
                # kernel_func is already a captured-graph replay; the helper's
                # own capture would replay-inside-capture and abort.
                use_cuda_graph=False,
            ) as results:
                _log_row(
                    geometry=geometry,
                    phase="context",
                    batch_size=batch_size,
                    seq_len=seq_len,
                    num_tokens=total_tokens,
                    model_name=model_name,
                    kernel_source="kda_module[chunk_kda]",
                    results=results,
                    sglang_version=sglang_version,
                    perf_filename=perf_filename,
                )
            successful_points += 1
        except Exception as e:
            failed_points += 1
            print(f"  Error at context batch_size={batch_size}, seq_len={seq_len}: {e}", flush=True)
            if aic_debug:
                import traceback

                traceback.print_exc()
        finally:
            # release this cell's captured graph (its private pool holds
            # GiB-scale activations at large token counts) before the next
            replay = None  # drops the closure holding the graph
            torch.cuda.empty_cache()

    print(
        f"[kda-module] heads={kda_heads}: {successful_points} points collected, {failed_points} failed",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model-src",
        required=True,
        help="local Kimi-K3 model directory (config/tokenizer files; weights not needed, dummy load)",
    )
    parser.add_argument("--model-name", default="moonshotai/Kimi-K3")
    parser.add_argument(
        "--heads",
        type=int,
        nargs="+",
        default=list(K3_SHARD_HEADS),
        help="per-rank KDA head shards to collect (96/tp); one process per geometry",
    )
    parser.add_argument("--output", default=PerfFile.LINEAR_ATTN_MODULE.value)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if not os.path.isfile(os.path.join(args.model_src, "config.json")):
        raise FileNotFoundError(f"--model-src {args.model_src} has no config.json")

    if len(args.heads) > 1:
        # One geometry per process: ModelRunner initializes the distributed
        # group and pools; re-creating it in-process is not supported.
        for heads in args.heads:
            cmd = [
                sys.executable,
                os.path.abspath(__file__),
                "--model-src",
                args.model_src,
                "--model-name",
                args.model_name,
                "--heads",
                str(heads),
                "--output",
                args.output,
                "--device",
                args.device,
            ]
            print(f"[kda-module] spawning geometry heads={heads}", flush=True)
            result = subprocess.run(cmd, check=False)
            if result.returncode != 0:
                print(f"[kda-module] heads={heads} exited rc={result.returncode}", flush=True)
        return

    run_kda_module_benchmark(
        model_src=args.model_src,
        kda_heads=args.heads[0],
        model_name=args.model_name,
        perf_filename=args.output,
        device=args.device,
    )


if __name__ == "__main__":
    main()
