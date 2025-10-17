# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import logging
import os

import numpy as np
import torch
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.entrypoints.engine import _set_envs_and_config
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.sampling.sampling_params import SamplingParams

# SGLang imports
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import BumpAllocator, suppress_other_loggers
from torch.profiler import ProfilerActivity, profile, record_function

try:
    from helper import log_perf
except ModuleNotFoundError:
    import os
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from helper import log_perf
import pkg_resources

DEEPSEEK_MODEL_PATH = os.environ.get("DEEPSEEK_MODEL_PATH", "/deepseek-v3")
logger = logging.getLogger(__name__)


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
    context_seq_lengths = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

    attention_backends = ["flashinfer", "fa3"]
    head_nums = [128, 64, 32, 16]

    for attention_backend in attention_backends:
        for head_num in head_nums:
            for batch_size in sorted(context_batch_sizes):
                for seq_length in sorted(context_seq_lengths):
                    # Memory limit checks for context
                    if batch_size * seq_length > 1024 * 2048:
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
    generation_seq_lengths = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

    attention_backends = ["flashinfer", "fa3"]
    head_nums = [128, 64, 32, 16]

    for attention_backend in attention_backends:
        for head_num in head_nums:
            for batch_size in sorted(generation_batch_sizes):
                for seq_length in sorted(generation_seq_lengths):
                    # Memory limit checks for generation
                    if batch_size * seq_length > 1024 * 2048:
                        continue
                    test_cases.append([batch_size, seq_length, attention_backend, head_num, False])

    return test_cases


def load_model_runner(model_path, attention_backend, head_num, test_layer, dtype="auto", device="cuda", tp_rank=0):
    """Load model runner
    Environment variables:
    - SGLANG_TEST_NUM_LAYERS=2  # Load only 2 layers
    - SGLANG_LOAD_FORMAT=dummy  # Use dummy weights
    """
    suppress_other_loggers()

    num_layers = int(os.environ.get("SGLANG_TEST_NUM_LAYERS", "2"))
    load_format = os.environ.get("SGLANG_LOAD_FORMAT", "dummy")

    server_args = ServerArgs(
        model_path=model_path,
        dtype=dtype,
        device=device,
        load_format=load_format,
        tp_size=1,
        trust_remote_code=True,
        mem_fraction_static=0.5,
        disable_radix_cache=True,
    )

    server_args.attention_backend = attention_backend
    print(f"Using attention backend: {attention_backend}")

    if num_layers > 0 and load_format == "dummy":
        override_args = {"num_hidden_layers": num_layers}
        if head_num != 128:
            override_args["num_attention_heads"] = head_num
        server_args.json_model_override_args = json.dumps(override_args)

    _set_envs_and_config(server_args)

    port_args = PortArgs.init_new(server_args)
    model_config = ModelConfig.from_server_args(server_args)
    model_runner = ModelRunner(
        model_config=model_config,
        mem_fraction_static=0.5,
        gpu_id=tp_rank,
        tp_rank=tp_rank,
        tp_size=server_args.tp_size,
        pp_rank=0,
        pp_size=1,
        moe_ep_rank=0,
        moe_ep_size=1,
        nccl_port=port_args.nccl_port,
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
                    req.prefix_indices = []
                    req.fill_ids = req.origin_input_ids
                    req.extend_input_len = len(req.fill_ids)
                    req.logprob_start_len = 0
                    reqs.append(req)

                batch = ScheduleBatch.init_new(
                    reqs=reqs,
                    req_to_token_pool=model_runner.req_to_token_pool,
                    token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
                    tree_cache=None,
                    model_config=model_runner.model_config,
                    enable_overlap=False,
                    spec_algorithm=SpeculativeAlgorithm.NONE,
                    enable_custom_logit_processor=False,
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
                # Save via log_perf
                try:
                    perf_filename = os.path.join(output_path, "context_mla_perf.txt")
                    os.makedirs(os.path.dirname(perf_filename), exist_ok=True)
                    device_name = torch.cuda.get_device_name(device)
                    version = pkg_resources.get_distribution("sglang").version
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
                print(f"  Prefill test failed: {e!s}")
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
                    req.prefix_indices = []
                    req.fill_ids = req.origin_input_ids
                    req.extend_input_len = len(req.fill_ids)
                    req.logprob_start_len = 0
                    req.cached_tokens = 0
                    req.already_computed = 0
                    reqs.append(req)
                batch = ScheduleBatch.init_new(
                    reqs=reqs,
                    req_to_token_pool=model_runner.req_to_token_pool,
                    token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
                    tree_cache=None,
                    model_config=model_runner.model_config,
                    enable_overlap=False,
                    spec_algorithm=SpeculativeAlgorithm.NONE,
                    enable_custom_logit_processor=False,
                )
                batch.prepare_for_extend()
                batch.output_ids = seq_length
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

                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g), torch.no_grad():
                    _ = attention_module(
                        positions=decode_positions,
                        hidden_states=decode_hidden,
                        forward_batch=forward_batch_decode,
                        zero_allocator=zero_allocator,
                    )

                for _ in range(num_warmup):
                    g.replay()

                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                for _ in range(num_iterations):
                    g.replay()
                end_event.record()
                torch.cuda.synchronize()

                avg_time_ms = start_event.elapsed_time(end_event) / num_iterations

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
                                    g.replay()
                                torch.cuda.synchronize()
                                prof.step()

                        prof.export_chrome_trace(f"{profiler_trace_path}.json")
                        print(f"  Profiler trace saved: {profiler_trace_path}.json")

                    except Exception as e:
                        print(f"  Warning: Profiler failed: {e!s}")

                torch.cuda.empty_cache()
                # Save via log_perf
                try:
                    perf_filename = os.path.join(output_path, "generation_mla_perf.txt")
                    os.makedirs(os.path.dirname(perf_filename), exist_ok=True)
                    device_name = torch.cuda.get_device_name(device)
                    version = pkg_resources.get_distribution("sglang").version
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
                    )
                except Exception as e:
                    print(f"  Warning: failed to log decode metrics: {e}")

                print(f"  Decode attention time: {avg_time_ms:.3f} ms")

                model_runner.req_to_token_pool.clear()
                model_runner.token_to_kv_pool_allocator.clear()
                del decode_hidden, decode_positions, forward_batch_decode, batch
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"  Decode test failed: {e!s}")
                print("  Skipping this configuration...")
                continue


if __name__ == "__main__":
    output_path = "/aiconfigurator/src/aiconfigurator/systems/data/h100_sxm/sglang/0.5.0/"
    model_path = DEEPSEEK_MODEL_PATH
    test_layer = 0
    num_warmup = 3
    num_iterations = 10
    dtype = "auto"
    device = "cuda"
    enable_profiler = False

    cleanup_distributed()

    print(f"Loading model from {model_path}...")
    print("\nTip: To test with dummy weights and limited layers, use:")
    print("  SGLANG_LOAD_FORMAT=dummy SGLANG_TEST_NUM_LAYERS=2 python collect_attn.py")

    # Get prefill and decode test cases separately
    prefill_test_cases = get_attention_prefill_test_cases()
    decode_test_cases = get_attention_decode_test_cases()
    test_cases = prefill_test_cases + decode_test_cases

    grouped_cases = {}
    for test_case in test_cases:
        batch_size, seq_length, attention_backend, head_num, is_prefill = test_case
        key = (attention_backend, head_num)
        if key not in grouped_cases:
            grouped_cases[key] = []
        grouped_cases[key].append(test_case)

    for (attention_backend, head_num), cases in grouped_cases.items():
        print(f"\n{'=' * 60}")
        print(f"TESTING: Attention Backend={attention_backend}, Head Num={head_num}")
        print(f"Test cases: {len(cases)}")
        print(f"{'=' * 60}")
        cleanup_distributed()

        torch.cuda.empty_cache()
        model_runner = load_model_runner(model_path, attention_backend, head_num, test_layer, dtype, device)

        run_attention_torch(
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
        )

        del model_runner
        cleanup_distributed()
        torch.cuda.empty_cache()

    print("\n" + "=" * 50)
    print("ALL TESTS COMPLETED")
    print("=" * 50)
