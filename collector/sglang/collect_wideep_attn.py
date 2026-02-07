# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import os
from importlib.metadata import version as get_version

import numpy as np
import torch
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.entrypoints.engine import _set_envs_and_config
from sglang.srt.layers.communicator import AttentionInputs, get_attn_tp_context
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.chunk_cache import ChunkCache
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.sampling.sampling_params import SamplingParams

# SGLang imports
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import BumpAllocator, suppress_other_loggers
from torch.profiler import ProfilerActivity, profile, record_function

try:
    from helper import _get_deepseek_model_path, benchmark_with_power, log_perf
except ModuleNotFoundError:
    import os
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from helper import _get_deepseek_model_path, benchmark_with_power, log_perf

DEEPSEEK_MODEL_PATH = _get_deepseek_model_path()


def cleanup_distributed():
    """Clean up distributed environment if it exists"""
    import sglang.srt.distributed.parallel_state as parallel_state

    # Reset all global group variables
    for var_name in ["_TP", "_PP", "_MOE_EP", "_MOE_TP", "_WORLD", "_PDMUX_PREFILL_TP_GROUP"]:
        if hasattr(parallel_state, var_name):
            setattr(parallel_state, var_name, None)

    import sglang.srt.eplb.expert_location as expert_location

    if hasattr(expert_location, "_global_expert_location_metadata"):
        expert_location._global_expert_location_metadata = None


def get_attention_prefill_test_cases():
    """Get prefill test cases for attention benchmarking
    Returns: list of [batch_size, seq_length, attention_backend, head_num, is_prefill]
    """
    test_cases = []

    context_batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    context_seq_lengths = [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

    attention_backends = ["flashinfer", "fa3"]
    head_nums = [128, 64, 32, 16]

    for attention_backend in attention_backends:
        for head_num in head_nums:
            for batch_size in sorted(context_batch_sizes):
                for seq_length in sorted(context_seq_lengths):
                    # Memory limit checks for context - reduced to avoid CUDA OOM/illegal access
                    # batch*seq limit: 128K tokens (was 2M, but large configs cause GPU errors)
                    if batch_size * seq_length > 128 * 1024:
                        continue
                    # Also skip very large individual seq_lengths with larger batches
                    if seq_length >= 8192 and batch_size > 8:
                        continue
                    test_cases.append([batch_size, seq_length, attention_backend, head_num, True])

    return test_cases


def get_attention_decode_test_cases():
    """
    Get decode test cases for attention benchmarking with batch_size, seq_length,
    attention_backend, and head_num
    """
    test_cases = []

    generation_batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    generation_seq_lengths = [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

    attention_backends = ["flashinfer", "fa3"]
    head_nums = [128, 64, 32, 16]

    for attention_backend in attention_backends:
        for head_num in head_nums:
            for batch_size in sorted(generation_batch_sizes):
                for seq_length in sorted(generation_seq_lengths):
                    # Memory limit checks for generation - reduced to avoid CUDA OOM/illegal access
                    # batch*seq limit: 256K tokens (generation is less memory intensive)
                    if batch_size * seq_length > 256 * 1024:
                        continue
                    # Also skip very large individual seq_lengths with larger batches
                    if seq_length >= 8192 and batch_size > 16:
                        continue
                    test_cases.append([batch_size, seq_length, attention_backend, head_num, False])

    return test_cases


def load_model_runner(model_path, attention_backend, head_num, test_layer, dtype="auto", device="cuda", tp_rank=0):
    """Load model runner
    Environment variables:
    - SGLANG_TEST_NUM_LAYERS=2  # Load only 2 layers
    - SGLANG_LOAD_FORMAT=dummy  # Use dummy weights
    """
    import random

    suppress_other_loggers()

    # Extract gpu_id from device string (e.g., "cuda:3" -> 3)
    device_str = str(device)
    if ":" in device_str:
        gpu_id = int(device_str.split(":")[-1])
        device_base = "cuda"  # ServerArgs expects just "cuda"
    else:
        gpu_id = tp_rank
        device_base = device_str

    num_layers = int(os.environ.get("SGLANG_TEST_NUM_LAYERS", "2"))
    load_format = os.environ.get("SGLANG_LOAD_FORMAT", "dummy")

    server_args = ServerArgs(
        model_path=model_path,
        dtype=dtype,
        device=device_base,
        load_format=load_format,
        tp_size=1,
        trust_remote_code=True,
        mem_fraction_static=0.5,
        disable_radix_cache=True,
    )

    server_args.attention_backend = attention_backend
    print(f"Using attention backend: {attention_backend}, gpu_id: {gpu_id}")

    if num_layers > 0 and load_format == "dummy":
        override_args = {"num_hidden_layers": num_layers}
        if head_num != 128:
            override_args["num_attention_heads"] = head_num
        server_args.json_model_override_args = json.dumps(override_args)

    _set_envs_and_config(server_args)

    # Use random nccl_port instead of PortArgs.init_new() to avoid distributed timeout
    # when running in parallel with collect.py framework
    nccl_port = 29500 + random.randint(0, 10000) + gpu_id * 100

    model_config = ModelConfig.from_server_args(server_args)
    model_runner = ModelRunner(
        model_config=model_config,
        mem_fraction_static=0.5,
        gpu_id=gpu_id,
        tp_rank=gpu_id,
        tp_size=server_args.tp_size,
        pp_rank=0,
        pp_size=1,
        moe_ep_rank=0,
        moe_ep_size=1,
        nccl_port=nccl_port,
        server_args=server_args,
    )

    return model_runner


def run_attention_torch(
    model_runner,
    cases,
    attention_backend,
    head_num,
    test_layer,
    num_warmup,
    num_iterations,
    enable_profiler,
    device,
    output_path,
):
    """Run attention benchmark for both prefill and decode phases"""

    attention_module = model_runner.model.model.layers[test_layer].self_attn
    model_runner.req_to_token_pool.clear()
    model_runner.token_to_kv_pool_allocator.clear()

    for test_case in cases:
        batch_size, seq_length, _, _, is_prefill = test_case

        if is_prefill:
            print(f"\nPrefill: batch_size={batch_size}, seq_length={seq_length}")

            try:
                model_runner.req_to_token_pool.clear()
                model_runner.token_to_kv_pool_allocator.clear()
                reqs = []
                for i in range(batch_size):
                    req = Req(
                        rid=str(i),
                        origin_input_text="",
                        origin_input_ids=list(torch.randint(0, 10000, (seq_length,)).tolist()),
                        sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
                    )
                    req.prefix_indices = torch.empty((0,), dtype=torch.int64)
                    req.fill_ids = req.origin_input_ids
                    req.extend_input_len = len(req.fill_ids)
                    req.logprob_start_len = 0
                    reqs.append(req)

                # Create tree_cache for new SGLang API
                cache_params = CacheInitParams(
                    disable=True,
                    req_to_token_pool=model_runner.req_to_token_pool,
                    token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
                    page_size=model_runner.token_to_kv_pool_allocator.page_size,
                )
                tree_cache = ChunkCache(cache_params)

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
                model_worker_batch = batch.get_model_worker_batch()
                forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)

                model_runner.attn_backend.init_forward_metadata(forward_batch)

                hidden_states = torch.randn(
                    batch_size * seq_length,
                    model_runner.model.config.hidden_size,
                    dtype=torch.bfloat16,
                    device="cuda",
                )
                positions = torch.arange(seq_length, device="cuda").unsqueeze(0).expand(batch_size, -1).flatten()
                zero_allocator = BumpAllocator(buffer_size=256, dtype=torch.float32, device="cuda")

                # Set up attn_inputs_ for new SGLang communicator pattern
                # qkv_latent needs shape: (tokens, q_lora_rank + kv_lora_rank + qk_rope_head_dim)
                q_lora_rank = getattr(attention_module, "q_lora_rank", 1536) or 1536
                kv_lora_rank = getattr(attention_module, "kv_lora_rank", 512)
                qk_rope_head_dim = getattr(attention_module, "qk_rope_head_dim", 64)
                qkv_latent_dim = q_lora_rank + kv_lora_rank + qk_rope_head_dim

                def dummy_qkv_latent_func(h, fb):
                    # Return tensor with correct dimension for qkv_latent
                    return torch.randn(h.shape[0], qkv_latent_dim, dtype=h.dtype, device=h.device)

                attn_inputs = AttentionInputs(hidden_states, forward_batch, dummy_qkv_latent_func)
                get_attn_tp_context().set_attn_inputs(attn_inputs)

                for _ in range(num_warmup):
                    with torch.no_grad():
                        _ = attention_module(
                            positions=positions,
                            hidden_states=hidden_states,
                            forward_batch=forward_batch,
                            zero_allocator=zero_allocator,
                        )

                cuda_times = []
                for i in range(num_iterations):
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    with torch.no_grad():
                        _ = attention_module(
                            positions=positions,
                            hidden_states=hidden_states,
                            forward_batch=forward_batch,
                            zero_allocator=zero_allocator,
                        )
                    end_event.record()
                    torch.cuda.synchronize()
                    if i > 1:
                        cuda_times.append(start_event.elapsed_time(end_event))

                # Profiler for detailed performance analysis (optional)
                if enable_profiler:
                    profiler_output_dir = "/aiconfigurator/profiler_output"
                    try:
                        os.makedirs(profiler_output_dir, exist_ok=True)
                        profiler_trace_path = os.path.join(
                            profiler_output_dir,
                            f"prefill_attention_b{batch_size}_s{seq_length}_layer{test_layer}",
                        )

                        with profile(
                            activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
                            record_shapes=True,
                            profile_memory=True,
                            with_stack=True,
                            schedule=torch.profiler.schedule(wait=1, warmup=1, active=10, repeat=1),
                        ) as prof:
                            for iter_idx in range(num_iterations):
                                with record_function("attention_prefill"), torch.no_grad():
                                    _ = attention_module(
                                        positions=positions,
                                        hidden_states=hidden_states,
                                        forward_batch=forward_batch,
                                        zero_allocator=zero_allocator,
                                    )
                                torch.cuda.synchronize()
                                prof.step()

                        prof.export_chrome_trace(f"{profiler_trace_path}.json")
                        print(f"  Profiler trace saved: {profiler_trace_path}.json")

                    except Exception as e:
                        print(f"  Warning: Profiler failed: {e!s}")

                avg_time_ms = np.mean(cuda_times)
                # Save via log_perf - save to collector/ directory to match non-wideep behavior
                try:
                    collector_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    perf_filename = (
                        os.path.join(collector_dir, "wideep_context_mla_perf.txt")
                        if output_path is None
                        else os.path.join(output_path, "wideep_context_mla_perf.txt")
                    )
                    device_name = torch.cuda.get_device_name(device)
                    version = get_version("sglang")
                    log_perf(
                        item_list=[
                            {
                                "mla_dtype": "fp8_block",
                                "kv_cache_dtype": "fp8",
                                "num_heads": head_num,
                                "batch_size": batch_size,
                                "isl": seq_length,
                                "tp_size": 1,
                                "step": 0,
                                "latency": avg_time_ms,
                            }
                        ],
                        framework="SGLang",
                        version=version,
                        device_name=device_name,
                        op_name="mla_context",
                        kernel_source=attention_backend,
                        perf_filename=perf_filename,
                    )
                except Exception as e:
                    print(f"  Warning: failed to log prefill metrics: {e}")

                print(
                    f"  Prefill attention time: {avg_time_ms:.3f} ms "
                    f"(min: {np.min(cuda_times):.3f}, max: {np.max(cuda_times):.3f}, "
                    f"std: {np.std(cuda_times):.3f})"
                )

                model_runner.req_to_token_pool.clear()
                model_runner.token_to_kv_pool_allocator.clear()
                del hidden_states, positions, forward_batch, batch
                torch.cuda.empty_cache()

            except Exception as e:
                import traceback

                print(f"  Prefill test failed: {e!s}")
                traceback.print_exc()

                # Only break on illegal memory access (context corruption), not OOM
                error_str = str(e).lower()
                if "cuda" in error_str and "illegal" in error_str:
                    print("  CUDA illegal access detected - stopping tests to prevent cascading failures")
                    break

                print("  Skipping this configuration...")
                continue

        else:  # decode phase
            print(f"\nDecode: batch_size={batch_size}, kv_cache_length={seq_length}")

            try:
                model_runner.req_to_token_pool.clear()
                model_runner.token_to_kv_pool_allocator.clear()
                reqs = []
                for i in range(batch_size):
                    req = Req(
                        rid=str(i),
                        origin_input_text="",
                        origin_input_ids=list(torch.randint(0, 10000, (seq_length,)).tolist()),
                        sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
                    )
                    req.prefix_indices = torch.empty((0,), dtype=torch.int64)
                    req.fill_ids = req.origin_input_ids
                    req.extend_input_len = len(req.fill_ids)
                    req.logprob_start_len = 0
                    req.cached_tokens = 0
                    req.already_computed = 0
                    reqs.append(req)

                # Create tree_cache for new SGLang API
                cache_params = CacheInitParams(
                    disable=True,
                    req_to_token_pool=model_runner.req_to_token_pool,
                    token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
                    page_size=model_runner.token_to_kv_pool_allocator.page_size,
                )
                tree_cache = ChunkCache(cache_params)

                batch = ScheduleBatch.init_new(
                    reqs=reqs,
                    req_to_token_pool=model_runner.req_to_token_pool,
                    token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
                    tree_cache=tree_cache,
                    model_config=model_runner.model_config,
                    enable_overlap=False,
                    spec_algorithm=SpeculativeAlgorithm.NONE,
                )
                # Allocate KV cache slots via prepare_for_extend, then switch to decode
                batch.prepare_for_extend()

                # Set up qkv_latent_func for AttentionInputs
                q_lora_rank = getattr(attention_module, "q_lora_rank", 1536) or 1536
                kv_lora_rank = getattr(attention_module, "kv_lora_rank", 512)
                qk_rope_head_dim = getattr(attention_module, "qk_rope_head_dim", 64)
                qkv_latent_dim = q_lora_rank + kv_lora_rank + qk_rope_head_dim

                def dummy_qkv_latent_func(h, fb):
                    return torch.randn(h.shape[0], qkv_latent_dim, dtype=h.dtype, device=h.device)

                # === DIRECTLY TO DECODE (skip prefill pass to reduce peak memory) ===
                # Set output_ids as tensor of last token IDs for decode
                batch.output_ids = torch.randint(0, 10000, (batch_size,), dtype=torch.int64, device="cuda")
                batch.prepare_for_decode()
                model_worker_batch_decode = batch.get_model_worker_batch()
                forward_batch_decode = ForwardBatch.init_new(model_worker_batch_decode, model_runner)
                model_runner.attn_backend.init_forward_metadata(forward_batch_decode)
                decode_hidden = torch.randn(
                    batch_size,
                    model_runner.model.config.hidden_size,
                    dtype=torch.bfloat16,
                    device="cuda",
                )
                decode_positions = torch.full((batch_size,), seq_length, device="cuda")
                zero_allocator = BumpAllocator(buffer_size=2048, dtype=torch.float32, device="cuda")

                # Set up attn_inputs_ for decode
                attn_inputs_decode = AttentionInputs(decode_hidden, forward_batch_decode, dummy_qkv_latent_func)
                get_attn_tp_context().set_attn_inputs(attn_inputs_decode)

                # Use benchmark_with_power for timing
                def kernel_func():
                    _ = attention_module(
                        positions=decode_positions,
                        hidden_states=decode_hidden,
                        forward_batch=forward_batch_decode,
                        zero_allocator=zero_allocator,
                    )

                with benchmark_with_power(
                    device=device,
                    kernel_func=kernel_func,
                    num_warmups=num_warmup,
                    num_runs=num_iterations,
                    repeat_n=1,
                ) as results:
                    pass

                avg_time_ms = results["latency_ms"]
                power_stats = results["power_stats"]

                if enable_profiler:
                    profiler_output_dir = "/aiconfigurator/profiler_output"
                    try:
                        os.makedirs(profiler_output_dir, exist_ok=True)
                        profiler_trace_path = os.path.join(
                            profiler_output_dir,
                            f"decode_attention_b{batch_size}_kv{seq_length}_layer{test_layer}",
                        )

                        with profile(
                            activities=[ProfilerActivity.CUDA],
                            record_shapes=False,
                            profile_memory=False,
                            with_stack=False,
                            schedule=torch.profiler.schedule(wait=1, warmup=1, active=10, repeat=1),
                        ) as prof:
                            for iter_idx in range(num_iterations):
                                with record_function("attention_decode"):
                                    kernel_func()
                                torch.cuda.synchronize()
                                prof.step()

                        prof.export_chrome_trace(f"{profiler_trace_path}.json")
                        print(f"  Profiler trace saved: {profiler_trace_path}.json")

                    except Exception as e:
                        print(f"  Warning: Profiler failed: {e!s}")

                torch.cuda.empty_cache()
                # Save via log_perf - save to collector/ directory to match non-wideep behavior
                try:
                    collector_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    perf_filename = (
                        os.path.join(collector_dir, "wideep_generation_mla_perf.txt")
                        if output_path is None
                        else os.path.join(output_path, "wideep_generation_mla_perf.txt")
                    )
                    device_name = torch.cuda.get_device_name(device)
                    version = get_version("sglang")
                    log_perf(
                        item_list=[
                            {
                                "mla_dtype": "fp8_block",
                                "kv_cache_dtype": "fp8",
                                "num_heads": head_num,
                                "batch_size": batch_size,
                                "isl": seq_length,
                                "tp_size": 1,
                                "step": 0,
                                "latency": avg_time_ms,
                            }
                        ],
                        framework="SGLang",
                        version=version,
                        device_name=device_name,
                        op_name="mla_generation",
                        kernel_source=attention_backend,
                        perf_filename=perf_filename,
                        power_stats=power_stats,
                    )
                except Exception as e:
                    print(f"  Warning: failed to log decode metrics: {e}")

                print(f"  Decode attention time: {avg_time_ms:.3f} ms")

                model_runner.req_to_token_pool.clear()
                model_runner.token_to_kv_pool_allocator.clear()
                del decode_hidden, decode_positions, forward_batch_decode, batch
                torch.cuda.empty_cache()

            except Exception as e:
                import traceback

                print(f"  Decode test failed: {e!s}")
                traceback.print_exc()

                # Only break on illegal memory access (context corruption), not OOM
                error_str = str(e).lower()
                if "cuda" in error_str and "illegal" in error_str:
                    print("  CUDA illegal access detected - stopping tests to prevent cascading failures")
                    break

                print("  Skipping this configuration...")
                continue


# ============================================================================
# Functions for collect.py framework (trtllm style: direct params, not index)
# ============================================================================


def get_wideep_mla_context_test_cases():
    """Returns list of (attention_backend, head_num, perf_filename) tuples."""
    backends = ["flashinfer", "fa3"]
    head_nums = [128, 64, 32, 16]
    return [[backend, head_num, "wideep_context_mla_perf.txt"] for backend in backends for head_num in head_nums]


def get_wideep_mla_generation_test_cases():
    """Returns list of (attention_backend, head_num, perf_filename) tuples."""
    backends = ["flashinfer", "fa3"]
    head_nums = [128, 64, 32, 16]
    return [[backend, head_num, "wideep_generation_mla_perf.txt"] for backend in backends for head_num in head_nums]


def run_mla(attention_backend, head_num, is_prefill, gpu_id, output_path=None):
    """Run MLA benchmark - shared logic for both context and generation.

    This function is called in a subprocess with CUDA_VISIBLE_DEVICES set.
    """
    # In subprocess, gpu_id=0 since CUDA_VISIBLE_DEVICES isolates the GPU
    torch.cuda.set_device("cuda:0")

    # Get test cases based on phase
    if is_prefill:
        all_cases = get_attention_prefill_test_cases()
        phase_name = "Context"
    else:
        all_cases = get_attention_decode_test_cases()
        phase_name = "Generation"

    cases = [tc for tc in all_cases if tc[2] == attention_backend and tc[3] == head_num]

    print(f"\n{'=' * 60}")
    print(f"MLA {phase_name}: backend={attention_backend}, head_num={head_num}, GPU={gpu_id}")
    print(f"Test cases: {len(cases)}")
    print(f"{'=' * 60}")

    cleanup_distributed()
    torch.cuda.empty_cache()

    model_runner = load_model_runner(
        DEEPSEEK_MODEL_PATH, attention_backend, head_num, test_layer=0, dtype="auto", device="cuda:0"
    )

    run_attention_torch(
        model_runner,
        cases,
        attention_backend,
        head_num,
        test_layer=0,
        num_warmup=3,
        num_iterations=10,
        enable_profiler=False,
        device="cuda:0",
        output_path=output_path,
    )

    del model_runner
    cleanup_distributed()
    torch.cuda.empty_cache()


def _run_mla_subprocess(attention_backend, head_num, is_prefill, gpu_id):
    """Helper to run MLA in subprocess with CUDA_VISIBLE_DEVICES isolation."""
    import subprocess
    import sys

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    phase = "context" if is_prefill else "generation"
    code = f'''
import sys
sys.path.insert(0, "{os.path.dirname(os.path.abspath(__file__))}")
from collect_wideep_attn import run_mla
run_mla("{attention_backend}", {head_num}, {is_prefill}, {gpu_id}, None)
'''

    proc = subprocess.Popen(
        [sys.executable, "-c", code],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )

    try:
        stdout, _ = proc.communicate(timeout=1800)  # 30 min for DeepGEMM JIT compile
        if stdout:
            print(stdout.decode("utf-8", errors="replace"))
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(f"MLA {phase} subprocess failed with exit code {proc.returncode}")


def run_wideep_mla_context(attention_backend, head_num, perf_filename, device="cuda:0"):
    """Run wideep MLA context benchmark for a specific (backend, head_num) config.

    Compatible with collect.py framework - uses subprocess for GPU isolation.
    """
    device_str = str(device) if not isinstance(device, str) else device
    gpu_id = int(device_str.split(":")[-1]) if ":" in device_str else 0

    print("\n" + "=" * 60)
    print(f"MLA Context: backend={attention_backend}, head_num={head_num}, GPU={gpu_id}")
    print("=" * 60)

    _run_mla_subprocess(attention_backend, head_num, True, gpu_id)


def run_wideep_mla_generation(attention_backend, head_num, perf_filename, device="cuda:0"):
    """Run wideep MLA generation benchmark for a specific (backend, head_num) config.

    Compatible with collect.py framework - uses subprocess for GPU isolation.
    """
    device_str = str(device) if not isinstance(device, str) else device
    gpu_id = int(device_str.split(":")[-1]) if ":" in device_str else 0

    print("\n" + "=" * 60)
    print(f"MLA Generation: backend={attention_backend}, head_num={head_num}, GPU={gpu_id}")
    print("=" * 60)

    _run_mla_subprocess(attention_backend, head_num, False, gpu_id)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SGLang Wideep MLA Benchmark")
    parser.add_argument("--output-path", default=None, help="Output directory for perf files")
    parser.add_argument("--device", default="cuda:0", help="CUDA device (e.g., cuda:0)")
    args = parser.parse_args()

    print(f"Loading model from {DEEPSEEK_MODEL_PATH}...")
    print("\nTip: SGLANG_LOAD_FORMAT=dummy SGLANG_TEST_NUM_LAYERS=2 python collect_wideep_attn.py")

    # Run all context and generation test cases
    for test_case in get_wideep_mla_context_test_cases():
        run_wideep_mla_context(*test_case, device=args.device)

    for test_case in get_wideep_mla_generation_test_cases():
        run_wideep_mla_generation(*test_case, device=args.device)

    print("\n" + "=" * 50)
    print("ALL TESTS COMPLETED")
    print("=" * 50)
