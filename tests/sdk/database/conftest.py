# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import yaml
from aiconfigurator.sdk.perf_database import PerfDatabase
from aiconfigurator.sdk import common
from collections import defaultdict


@pytest.fixture
def perf_db(tmp_path):
    """
    Instantiate a PerfDatabase with "dummy" system, backend, version and patched loaders.
    Because we patched yaml.load and every loader, no actual files are needed.
    """
    system = 'any_system'
    backend = 'any_backend'
    version = 'v1'
    systems_dir = str(tmp_path)  # path is never actually read because of our patches
    
    # Create the yaml file to avoid FileNotFoundError
    yaml_file = tmp_path / f'{system}.yaml'
    yaml_file.write_text("dummy: data")  # Content doesn't matter because yaml.load is patched

    return PerfDatabase(system, backend, version, systems_dir)


@pytest.fixture(autouse=True)
def patch_all_loaders_and_yaml(request, monkeypatch):
    """
    Automatically patch every data‐loading function (so no real files are required)
    and patch yaml.load to return a minimal system_spec. This lets us instantiate
    PerfDatabase without hitting the filesystem.
    """
    if "patch_loader_and_yaml" in request.keywords:
        # 1) Patch yaml.load so that PerfDatabase.__init__ sees a valid system_spec.
        dummy_system_spec = {
            'data_dir': 'data',  # PerfDatabase will look under systems_dir/data/<backend>/<version>
            'misc': {'nccl_version': 'v1'},
            'gpu': {
                # These two values are used in many "SOL"‐mode formulas:
                'float16_tc_flops': 1_000.0,
                'mem_bw': 100.0,
                # For query_nccl NON‐SOL branch:
                'mem_empirical_constant_latency': 1.0,
            },
            'node': {
                # Used by query_allreduce, query_nccl, query_p2p:
                'inter_node_bw': 100.0,
                'intra_node_bw': 100.0,
                'num_gpus_per_node': 8,
                'p2p_latency': 0.000001,
            },
        }
        monkeypatch.setattr(yaml, 'load', lambda stream, Loader=None: dummy_system_spec)

        # 2) Patch load_gemm_data to return a minimal nested dict keyed by common.GEMMQuantMode.float16
        #    with multiple entries to avoid extrapolation errors
        dummy_gemm_data = {
            common.GEMMQuantMode.float16: {
                64: {
                    128: {
                        256: 10.0,  # "latency" at (m=64, n=128, k=256)
                        512: 20.0   # Additional point to satisfy extrapolation
                    },
                    256: {
                        256: 15.0,
                        512: 25.0
                    }
                },
                128: {
                    128: {
                        256: 12.0,
                        512: 22.0
                    },
                    256: {
                        256: 17.0,
                        512: 27.0
                    }
                }
            }
        }
        monkeypatch.setattr(
            'aiconfigurator.sdk.perf_database.load_gemm_data',
            lambda path: dummy_gemm_data
        )

        # 3) Patch load_custom_allreduce_data to return proper structure
        #    Structure: { 'float16': { 2: { 'AUTO': { 1024:  5.0 } } } }
        dummy_custom_allreduce_data = {
            'float16': {
                2: {'AUTO': {1024: 5.0, 2048: 15.0}},
                4: {'AUTO': {1024: 10.0, 2048: 20.0}},
                8: {'AUTO': {1024: 15.0, 2048: 30.0}},
            }
        }
        monkeypatch.setattr(
            'aiconfigurator.sdk.perf_database.load_custom_allreduce_data',
            lambda path: dummy_custom_allreduce_data
        )

        # 4) load_moe_data needs to return 2 dicts
        monkeypatch.setattr(
            'aiconfigurator.sdk.perf_database.load_moe_data',
            lambda path: ({}, {})
        )

        # 5) For completeness, patch every other loader to return an empty dict
        for loader_name in [
            'load_context_attention_data',
            'load_generation_attention_data',
            'load_context_mla_data',
            'load_generation_mla_data',
            'load_mla_bmm_data',
            'load_nccl_data',
        ]:
            monkeypatch.setattr(
                f'aiconfigurator.sdk.perf_database.{loader_name}',
                lambda path: {}
            )


@pytest.fixture
def comprehensive_perf_db(tmp_path, monkeypatch):
    """
    Create a PerfDatabase with comprehensive test data for all query methods.
    """
    # System spec with all required fields
    dummy_system_spec = {
        'data_dir': 'data',
        'misc': {'nccl_version': 'v1'},
        'gpu': {
            'float16_tc_flops': 1_000_000_000_000.0,  # 1 TFLOPS
            'mem_bw': 1_000_000_000_000.0,  # 1 TB/s
            'mem_bw_empirical_scaling_factor': 0.8,
            'mem_empirical_constant_latency': 0.001,  # 1 us
        },
        'node': {
            'inter_node_bw': 100_000_000_000.0,  # 100 GB/s
            'intra_node_bw': 200_000_000_000.0,  # 200 GB/s
            'num_gpus_per_node': 8,
            'p2p_latency': 0.000001,  # 1 us
        },
    }
    
    # Create the yaml file
    yaml_file = tmp_path / 'test_system.yaml'
    with open(yaml_file, 'w') as f:
        yaml.dump(dummy_system_spec, f)
    
    monkeypatch.setattr('yaml.load', lambda stream, Loader=None: dummy_system_spec)
    
    # Comprehensive GEMM data
    dummy_gemm_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))
    for quant_mode in [common.GEMMQuantMode.float16, common.GEMMQuantMode.fp8]:
        for m in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
            for n in [128, 256, 512, 1024]:
                for k in [128, 256, 512, 1024]:
                    dummy_gemm_data[quant_mode][m][n][k] = 0.1 + m * 0.001 + n * 0.0001 + k * 0.00001
    
    # Context attention data
    dummy_context_attention_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))))
    for quant_mode in [common.FMHAQuantMode.float16, common.FMHAQuantMode.fp8]:
        for kv_cache_dtype in [common.KVCacheQuantMode.float16, common.KVCacheQuantMode.fp8]:
            for kv_n in [0, 1, 2, 4, 8]:  # 0 means MHA
                for n in [4, 8, 16, 32]:
                    for s in [16, 32, 64, 128, 256]:
                        for b in [1, 2, 4, 8]:
                            dummy_context_attention_data[quant_mode][kv_cache_dtype][kv_n][n][s][b] = 0.01 * (n * s * b) / 1000.0
    
    # Generation attention data
    dummy_generation_attention_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict()))))
    for kv_cache_dtype in [common.KVCacheQuantMode.float16, common.KVCacheQuantMode.fp8]:
        for kv_n in [0, 1, 2, 4, 8]:
            for n in [4, 8, 16, 32]:
                # Only create data where kv_n <= n to satisfy the assertion
                if kv_n <= n:
                    for b in [1, 2, 4, 8, 16]:
                        for s in [1, 16, 32, 64, 128, 256, 512, 1024]:
                            dummy_generation_attention_data[kv_cache_dtype][kv_n][n][b][s] = 0.001 * (n * b * s) / 1000.0
    
    # MoE data
    dummy_moe_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict()))))))))
    for quant_mode in [common.MoEQuantMode.float16, common.MoEQuantMode.fp8]:
        for workload in ['uniform', 'imbalanced']:
            for topk in [1, 2]:
                for num_experts in [8, 16]:
                    for hidden_size in [1024, 2048, 4096]:
                        for inter_size in [4096, 8192]:
                            for moe_tp in [1, 2]:
                                for moe_ep in [1, 2]:
                                    for num_tokens in [1, 2, 4, 8, 16, 32]:
                                        dummy_moe_data[quant_mode][workload][topk][num_experts][hidden_size][inter_size][moe_tp][moe_ep][num_tokens] = 0.1 * num_tokens
    
    # Context MLA data
    dummy_context_mla_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))))
    for quant_mode in [common.FMHAQuantMode.float16]:
        for kv_cache_dtype in [common.KVCacheQuantMode.float16]:
            for num_heads in [16, 32, 64, 128]:
                for s in [16, 32, 64, 128]:
                    for b in [1, 2, 4, 8]:
                        dummy_context_mla_data[quant_mode][kv_cache_dtype][num_heads][s][b] = 0.0001 * s * b * num_heads
    
    # Generation MLA data
    dummy_generation_mla_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict()))))
    for kv_cache_dtype in [common.KVCacheQuantMode.float16]:
        for num_heads in [16, 32, 64, 128]:
            for b in [1, 2, 4, 8]:
                for s in [1, 16, 32, 64, 128]:
                    dummy_generation_mla_data[kv_cache_dtype][num_heads][b][s] = 0.00001 * b * s * num_heads
    
    # MLA BMM data
    dummy_mla_bmm_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))
    for quant_mode in [common.GEMMQuantMode.float16, common.GEMMQuantMode.fp8]:
        for op_name in ['mla_gen_pre', 'mla_gen_post']:
            for num_heads in [1, 2, 4, 8]:
                for num_tokens in [1, 2, 4, 8, 16, 32]:
                    dummy_mla_bmm_data[quant_mode][op_name][num_heads][num_tokens] = 0.01 * num_heads * num_tokens
    
    # Custom allreduce data
    dummy_custom_allreduce_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))
    for dtype in [common.CommQuantMode.half]:
        for tp_size in [1, 2, 4, 8]:
            for strategy in ['AUTO']:
                for msg_size in [512, 1024, 2048, 4096, 8192]:
                    dummy_custom_allreduce_data[dtype][tp_size][strategy][msg_size] = 0.001 * msg_size * tp_size
    
    # NCCL data
    dummy_nccl_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))
    for dtype in [common.CommQuantMode.half, common.CommQuantMode.int8]:  # Use enum objects as keys
        for operation in ['all_gather', 'alltoall', 'reduce_scatter']:
            for num_gpus in [1, 2, 4, 8]:
                for msg_size in [512, 1024, 2048, 4096]:
                    dummy_nccl_data[dtype][operation][num_gpus][msg_size] = 0.001 * msg_size * num_gpus
    
    # Apply all patches
    monkeypatch.setattr('aiconfigurator.sdk.perf_database.load_gemm_data', lambda path: dummy_gemm_data)
    monkeypatch.setattr('aiconfigurator.sdk.perf_database.load_context_attention_data', lambda path: dummy_context_attention_data)
    monkeypatch.setattr('aiconfigurator.sdk.perf_database.load_generation_attention_data', lambda path: dummy_generation_attention_data)
    monkeypatch.setattr('aiconfigurator.sdk.perf_database.load_custom_allreduce_data', lambda path: dummy_custom_allreduce_data)
    monkeypatch.setattr('aiconfigurator.sdk.perf_database.load_moe_data', lambda path: (dummy_moe_data, dummy_moe_data))
    monkeypatch.setattr('aiconfigurator.sdk.perf_database.load_context_mla_data', lambda path: dummy_context_mla_data)
    monkeypatch.setattr('aiconfigurator.sdk.perf_database.load_generation_mla_data', lambda path: dummy_generation_mla_data)
    monkeypatch.setattr('aiconfigurator.sdk.perf_database.load_mla_bmm_data', lambda path: dummy_mla_bmm_data)
    monkeypatch.setattr('aiconfigurator.sdk.perf_database.load_nccl_data', lambda path: dummy_nccl_data)
    
    return PerfDatabase('test_system', 'test_backend', 'v1', str(tmp_path)) 