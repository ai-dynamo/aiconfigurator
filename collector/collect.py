# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import time
import multiprocessing as mp
from tqdm import tqdm
import torch
import argparse
import traceback

import logging
logger = logging.getLogger(__name__)

def worker(queue, device_id : int, func, progress_value, lock):
    device = torch.device(f'cuda:{device_id}')
    status = True
    while True:
        task = queue.get()
        if task is None:
            break

        with lock:
            progress_value.value += 1
        try:
            result = func(*task, device)
        except Exception as e:
            print(f'Error: {e}', task, device)
            traceback.print_exc()
            status = False
        
        if not status:
            break


def parallel_run(tasks, func, num_processes):
    queue = mp.Queue()
    processes = []
    manager = mp.Manager()
    progress_value = manager.Value('i', 0)
    lock = manager.Lock()
    
    def start_process(device_id):
        p = mp.Process(target=worker, args=(queue, device_id, func, progress_value, lock))
        p.start()
        return p

    for device_id in range(num_processes):
        p = start_process(device_id)
        processes.append(p)

    for task in tasks:
        queue.put(task)

    for _ in range(len(processes)):
        queue.put(None)

    with tqdm(total=len(tasks)) as progress_bar:
        while progress_value.value < len(tasks):
            for i, p in enumerate(processes):
                if not p.is_alive():
                    print(f"Process {i} died with exit code {p.exitcode}, restarting...")
                    processes[i] = start_process(i)

            progress_bar.n = progress_value.value
            progress_bar.refresh()
            time.sleep(2)
    
    for p in processes:
        p.join()

def collect_trtllm(num_processes : int):
    """
    Collect performance data for TensorRT LLM.
    """


    os.environ['TLLM_LOG_LEVEL']= 'ERROR'
    os.environ['TRTLLM_DG_ENABLED'] = "1" # enable deepgemm by default
    try:
        import tensorrt_llm
        version = tensorrt_llm.__version__
    except:
        logger.error("TensorRT LLM is not installed. Please install it from https://github.com/NVIDIA/TensorRT-LLM")
        return

    # keep this to collect pre-hopper kernels for now.
    try:
        import trtllm.collect_gemm_trt
        test_cases = trtllm.collect_gemm_trt.get_gemm_test_cases()
        parallel_run(test_cases, trtllm.collect_gemm_trt.run_gemm, num_processes)
        logger.info(f"collected gemm_trt test cases for TensorRT LLM {version}")
    except:
        logger.warning("cannot collect gemm_trt test cases, skipping...")

    # only float16, fp8 and fp8_block
    try:
        import trtllm.collect_gemm
        test_cases = trtllm.collect_gemm.get_gemm_test_cases()
        parallel_run(test_cases, trtllm.collect_gemm.run_gemm, num_processes)
        logger.info(f"collected gemm test cases for TensorRT LLM {version}")
    except:
        logger.warning("cannot collect gemm test cases, skipping...")

    try:
        import trtllm.collect_mla
        test_cases = trtllm.collect_mla.get_context_mla_test_cases()
        if version.startswith('1.1'):
            import trtllm.collect_mla_1_1rc2
            parallel_run(test_cases, trtllm.collect_mla_1_1rc2.run_mla, num_processes)
        else:
            parallel_run(test_cases, trtllm.collect_mla.run_mla, num_processes)
        logger.info(f"collected mla test cases for TensorRT LLM {version}")
        
        test_cases = trtllm.collect_mla.get_generation_mla_test_cases()
        if version.startswith('1.1'):
            parallel_run(test_cases, trtllm.collect_mla_1_1rc2.run_mla, num_processes)
        else:
            parallel_run(test_cases, trtllm.collect_mla.run_mla, num_processes)
        logger.info(f"collected mla test cases for TensorRT LLM {version}")        
    except:
        logger.warning("cannot collect mla test cases, skipping...")

    try:
        if version.startswith('0.20.0'):
            import trtllm.collect_moe_pre_0_20 as collect_moe
        elif version.startswith('0.21.0') or version.startswith('1.0.0') or version.startswith('1.1.0'):
            import trtllm.collect_moe as collect_moe
        else:
            raise ValueError(f"cannot collect moe test cases for TensorRT LLM {version}, skipping...")
        test_cases = collect_moe.get_moe_test_cases()
        parallel_run(test_cases, collect_moe.run_moe_torch, num_processes)
        logger.info(f"collected moe test cases for TensorRT LLM {version}")
    except:
        logger.warning("cannot collect moe test cases, skipping...")

    try:
        import trtllm.collect_mla_bmm
        test_cases = trtllm.collect_mla_bmm.get_mla_gen_pre_test_cases()
        parallel_run(test_cases, trtllm.collect_mla_bmm.run_mla_gen_pre, num_processes)
        logger.info(f"collected mla_bmm test cases for TensorRT LLM {version}")
        test_cases = trtllm.collect_mla_bmm.get_mla_gen_post_test_cases()
        parallel_run(test_cases, trtllm.collect_mla_bmm.run_mla_gen_post, num_processes)
        logger.info(f"collected mla_bmm test cases for TensorRT LLM {version}")        
    except:
        logger.warning("cannot collect mla_bmm test cases, skipping...")

    try:
        import trtllm.collect_attn
        test_cases = trtllm.collect_attn.get_context_attention_test_cases()
        parallel_run(test_cases, trtllm.collect_attn.run_attention_torch, num_processes)
        logger.info(f"collected attention test cases for TensorRT LLM {version}")
        test_cases = trtllm.collect_attn.get_generation_attention_test_cases()
        parallel_run(test_cases, trtllm.collect_attn.run_attention_torch, num_processes)
        logger.info(f"collected attention test cases for TensorRT LLM {version}")        
    except:
        logger.warning("cannot collect attention test cases, skipping...")

def collect_sglang():
    pass

def collect_vllm(num_processes : int):
    """
    Collect performance data for VLLM v1.
    """

    try:
        from vllm.version import __version__ as vllm_version
        version = vllm_version

    except:
        logger.error("VLLM is not installed. Please install it from https://github.com/vllm-project/vllm")
        return

    # supported vllm v1 GEMM collection wich supports fp16,fp8,fp8_block_wise,awq and gptq
    try:
        import vllm_v1.collect_gemm
        test_cases = vllm_v1.collect_gemm.get_gemm_test_cases(is_unit_test=False)
        parallel_run(test_cases, vllm_v1.collect_gemm.run_gemm, num_processes)
        logger.info(f"collected gemm_vllm test cases for VLLM {version}")
    except:
        logger.warning("cannot collect gemm_vllm test cases, skipping...")

    # supported vllm v1 atten collection which supports fp16(auto) and fp8 kv cache. flashatten impl for prefill and flashinfer impl for decode.
    try:
        import vllm_v1.collect_attn
        test_cases = vllm_v1.collect_attn.get_context_attention_test_cases(if_unit_test=False)
        parallel_run(test_cases, vllm_v1.collect_attn.run_attention_torch, num_processes)
        logger.info(f"collected context attention test cases for VLLM {version}")
        test_cases = vllm_v1.collect_attn.get_generation_attention_test_cases()
        parallel_run(test_cases, vllm_v1.collect_attn.run_attention_torch, num_processes)
        logger.info(f"collected generation attention test cases for VLLM {version}")        
    except:
        logger.warning("cannot collect VLLM attention test cases, skipping...")

def main():
    parser = argparse.ArgumentParser(description='Collect performance data for backends')
    parser.add_argument('--backend', type=str, choices=['trtllm', 'sglang', 'vllm'], default='trtllm', help='Backend name')
    args = parser.parse_args()

    num_processes = torch.cuda.device_count()
    mp.set_start_method('spawn')    

    if args.backend == 'trtllm':
        collect_trtllm(num_processes)
    elif args.backend == 'sglang':
        collect_sglang(num_processes)
    elif args.backend == 'vllm':
        collect_vllm(num_processes)
if __name__ == '__main__':
    main()
    


