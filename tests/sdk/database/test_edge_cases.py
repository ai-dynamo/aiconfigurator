# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import math
from unittest.mock import patch, MagicMock
from aiconfigurator.sdk import common
from aiconfigurator.sdk.perf_database import PerfDatabase, get_database, databases_cache


@pytest.mark.patch_loader_and_yaml
class TestNcclEdgeCases:
    """Test edge cases for query_nccl method."""
    
    def test_query_nccl_non_sol_single_gpu(self, comprehensive_perf_db):
        """Test NCCL with single GPU returns 0."""
        result = comprehensive_perf_db.query_nccl(
            common.CommQuantMode.half, 1, 'all_gather', 1024,
            sol_mode=common.SOLMode.NON_SOL
        )
        assert result == 0.0
    
    def test_query_nccl_non_sol_interpolation(self, comprehensive_perf_db):
        """Test NCCL non-SOL mode with interpolation."""
        # Use values that exist in our test data
        result = comprehensive_perf_db.query_nccl(
            common.CommQuantMode.half, 4, 'all_gather', 1024,
            sol_mode=common.SOLMode.NON_SOL
        )
        
        # Should use interpolation from nccl_data
        expected = comprehensive_perf_db._nccl_data[common.CommQuantMode.half]['all_gather'][4][1024]
        assert math.isclose(result, expected, rel_tol=1e-6)
    
    def test_query_nccl_non_sol_large_gpu_count(self, comprehensive_perf_db):
        """Test NCCL with more than 8 GPUs applies scaling."""
        # First get baseline with 8 GPUs
        baseline = comprehensive_perf_db.query_nccl(
            common.CommQuantMode.half, 8, 'all_gather', 1024,
            sol_mode=common.SOLMode.NON_SOL
        )
        
        # Test with 16 GPUs
        result = comprehensive_perf_db.query_nccl(
            common.CommQuantMode.half, 16, 'all_gather', 1024,
            sol_mode=common.SOLMode.NON_SOL
        )

        node_info = comprehensive_perf_db.system_spec['node']
        intra_node_slowdown = node_info['intra_node_bw'] / node_info['inter_node_bw']
        baseline_transfers_per_gpu = (8 - 1) / 8
        result_transfers_per_gpu = (16 - 1) / 16
        correction_factor = intra_node_slowdown * result_transfers_per_gpu / baseline_transfers_per_gpu

        expected = baseline * correction_factor
        assert math.isclose(result, expected, rel_tol=1e-6)
    
    def test_query_nccl_edge_message_sizes(self, comprehensive_perf_db):
        """Test NCCL with very small and very large message sizes."""
        # Very small message
        result_small = comprehensive_perf_db.query_nccl(
            common.CommQuantMode.half, 4, 'alltoall', 1,
            sol_mode=common.SOLMode.NON_SOL
        )
        assert result_small > 0
        
        # Very large message (extrapolation)
        result_large = comprehensive_perf_db.query_nccl(
            common.CommQuantMode.half, 4, 'reduce_scatter', 1_000_000,
            sol_mode=common.SOLMode.NON_SOL
        )
        assert result_large > 0


@pytest.mark.patch_loader_and_yaml
class TestAllreduceEdgeCases:
    """Test edge cases for query_allreduce method."""
    
    def test_query_allreduce_single_gpu(self, comprehensive_perf_db):
        """Test allreduce with single GPU returns 0."""
        # SOL mode
        result_sol = comprehensive_perf_db.query_allreduce(
            common.CommQuantMode.half, 1, 1024,
            sol_mode=common.SOLMode.SOL
        )
        assert result_sol == 0.0
        
        # Non-SOL mode
        result_non_sol = comprehensive_perf_db.query_allreduce(
            common.CommQuantMode.half, 1, 1024,
            sol_mode=common.SOLMode.NON_SOL
        )
        assert result_non_sol == 0.0
    
    def test_query_allreduce_large_tp_scaling(self, comprehensive_perf_db):
        """Test allreduce with TP > 8 applies scaling factor."""
        # Get baseline with TP=8
        baseline = comprehensive_perf_db.query_allreduce(
            common.CommQuantMode.half, 8, 2048,
            sol_mode=common.SOLMode.NON_SOL
        )
        
        # Test with TP=16
        result = comprehensive_perf_db.query_allreduce(
            common.CommQuantMode.half, 16, 2048,
            sol_mode=common.SOLMode.NON_SOL
        )
        
        # Should apply scaling: lat * (tp_size-1)/tp_size * 8/7
        node_info = comprehensive_perf_db.system_spec['node']
        intra_node_slowdown = node_info['intra_node_bw'] / node_info['inter_node_bw']
        expected_scaling = (16-1)/16 * intra_node_slowdown
        baseline_unscaled = baseline / ((8-1)/8)  # Remove 8 GPU scaling
        expected = baseline_unscaled * expected_scaling
        assert math.isclose(result, expected, rel_tol=1e-6)
    
    def test_query_allreduce_extrapolation(self, comprehensive_perf_db):
        """Test allreduce with message size requiring extrapolation."""
        # Use a size not in our test data
        result = comprehensive_perf_db.query_allreduce(
            common.CommQuantMode.half, 4, 3000,  # 3000 is between 2048 and 4096
            sol_mode=common.SOLMode.NON_SOL
        )
        assert result > 0
        
        # Should be between the two surrounding values
        lower = comprehensive_perf_db.query_allreduce(
            common.CommQuantMode.half, 4, 2048,
            sol_mode=common.SOLMode.NON_SOL
        )
        upper = comprehensive_perf_db.query_allreduce(
            common.CommQuantMode.half, 4, 4096,
            sol_mode=common.SOLMode.NON_SOL
        )
        assert lower < result < upper


class TestInitializationEdgeCases:
    """Test edge cases during PerfDatabase initialization."""
    
    def test_extrapolation_during_init(self, tmp_path, monkeypatch, caplog):
        """Test that extrapolation runs during initialization."""
        # Set up minimal system spec
        import yaml
        dummy_system_spec = {
            'data_dir': 'data',
            'misc': {'nccl_version': 'v1'},
            'gpu': {
                'float16_tc_flops': 1000.0,
                'mem_bw': 100.0,
                'mem_bw_empirical_scaling_factor': 0.8,
                'mem_empirical_constant_latency': 0.001,
            },
            'node': {
                'inter_node_bw': 100.0,
                'intra_node_bw': 200.0,
                'num_gpus_per_node': 8,
                'p2p_latency': 0.000001,
            },
        }
        yaml_file = tmp_path / 'test.yaml'
        with open(yaml_file, 'w') as f:
            yaml.dump(dummy_system_spec, f)
        
        monkeypatch.setattr('yaml.load', lambda stream, Loader=None: dummy_system_spec)
        
        # Create minimal context attention data
        from collections import defaultdict
        dummy_context_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))))
        dummy_context_data[common.FMHAQuantMode.float16][common.KVCacheQuantMode.float16][0][4][16][1] = 0.1
        dummy_context_data[common.FMHAQuantMode.float16][common.KVCacheQuantMode.float16][0][4][32][1] = 0.2
        dummy_context_data[common.FMHAQuantMode.float16][common.KVCacheQuantMode.float16][0][8][16][1] = 0.15
        dummy_context_data[common.FMHAQuantMode.float16][common.KVCacheQuantMode.float16][0][8][32][1] = 0.25
        
        monkeypatch.setattr('aiconfigurator.sdk.perf_database.load_context_attention_data', lambda path: dummy_context_data)
        
        # Patch other loaders to return empty defaultdicts
        for loader, depth in [
            ('load_gemm_data', 4),  # quant_mode -> m -> n -> k -> value
            ('load_generation_attention_data', 5),  # kv_cache -> num_kv -> n -> b -> s -> value
            ('load_custom_allreduce_data', 4),  # quant -> tp -> strategy -> size -> value
            ('load_nccl_data', 4),  # quant -> op -> num_gpus -> size -> value
            ('load_moe_data', 9),  # quant -> workload -> topk -> experts -> hidden -> inter -> tp -> ep -> tokens -> value
            ('load_context_mla_data', 5),  # quant -> kv_cache -> tp -> s -> b -> value
            ('load_generation_mla_data', 4),  # kv_cache -> tp -> b -> s -> value
            ('load_mla_bmm_data', 4),  # quant -> pre/post -> heads -> tokens -> value
        ]:
            # Create nested defaultdict with appropriate depth
            def create_nested_defaultdict(depth):
                if depth == 1:
                    return defaultdict(float)
                return defaultdict(lambda: create_nested_defaultdict(depth - 1))
            
            if loader == "load_moe_data":
                loader_func = lambda path, d=depth: (create_nested_defaultdict(d), create_nested_defaultdict(d))
            else:
                loader_func = lambda path, d=depth: create_nested_defaultdict(d)
            monkeypatch.setattr(f'aiconfigurator.sdk.perf_database.{loader}', loader_func)
        
        # Initialize database - should trigger extrapolation
        db = PerfDatabase('test', 'backend', 'v1', str(tmp_path))
        
        # Check that extrapolation created new data points
        # Should have more than the 4 original points
        total_points = 0
        for quant_mode in db._context_attention_data:
            for kv_cache in db._context_attention_data[quant_mode]:
                for kv_n in db._context_attention_data[quant_mode][kv_cache]:
                    for n in db._context_attention_data[quant_mode][kv_cache][kv_n]:
                        for s in db._context_attention_data[quant_mode][kv_cache][kv_n][n]:
                            total_points += len(db._context_attention_data[quant_mode][kv_cache][kv_n][n][s])
        
        assert total_points > 4, "Extrapolation should have created additional data points"


@pytest.mark.patch_loader_and_yaml
class TestGemmInterpolation:
    """Test GEMM-specific interpolation behavior."""
    
    def test_query_gemm_interpolation(self, comprehensive_perf_db):
        """Test GEMM interpolation between known points."""
        quant_mode = common.GEMMQuantMode.float16
        
        # Query a point that requires interpolation
        # Our test data has points at m=[1,2,4,8,...], n=[128,256,...], k=[128,256,...]
        result = comprehensive_perf_db.query_gemm(
            3, 192, 192, quant_mode,  # All values between grid points
            sol_mode=common.SOLMode.NON_SOL
        )
        
        # Should return a reasonable interpolated value
        assert result > 0
        
        # Should be between surrounding values
        lower_bound = comprehensive_perf_db.query_gemm(2, 128, 128, quant_mode, sol_mode=common.SOLMode.NON_SOL)
        upper_bound = comprehensive_perf_db.query_gemm(4, 256, 256, quant_mode, sol_mode=common.SOLMode.NON_SOL)
        assert lower_bound < result < upper_bound
    
    def test_query_gemm_extrapolation(self, comprehensive_perf_db):
        """Test GEMM extrapolation beyond data range."""
        quant_mode = common.GEMMQuantMode.fp8
        
        # Query a very large size (beyond our test data)
        result = comprehensive_perf_db.query_gemm(
            512, 2048, 2048, quant_mode,
            sol_mode=common.SOLMode.NON_SOL
        )
        
        # Should return a reasonable value
        assert result > 0
        
        # For large sizes, should be larger than smaller sizes
        smaller = comprehensive_perf_db.query_gemm(
            256, 1024, 1024, quant_mode,
            sol_mode=common.SOLMode.NON_SOL
        )
        assert result > smaller


class TestDatabaseCache:
    """Test database caching functionality."""
    
    def test_get_database_caching(self, tmp_path, monkeypatch):
        """Test that get_database properly caches instances."""
        # Clear cache first
        databases_cache.clear()
        
        # Mock PerfDatabase to track instantiations
        instantiation_count = 0
        original_init = PerfDatabase.__init__
        
        def counting_init(self, *args, **kwargs):
            nonlocal instantiation_count
            instantiation_count += 1
            # Don't actually initialize to avoid file operations
            self.system = args[0]
            self.backend = args[1]
            self.version = args[2]
            self._default_sol_mode = common.SOLMode.NON_SOL
        
        monkeypatch.setattr(PerfDatabase, '__init__', counting_init)
        
        # Mock file operations
        system_spec = {
            'data_dir': 'data',
            'misc': {'nccl_version': 'v1'},
            'gpu': {
                'float16_tc_flops': 1000.0,
                'mem_bw': 100.0,
                'mem_bw_empirical_scaling_factor': 0.8,
                'mem_empirical_constant_latency': 0.001,
            },
            'node': {
                'inter_node_bw': 100.0,
                'intra_node_bw': 200.0,
                'num_gpus_per_node': 8,
                'p2p_latency': 0.000001,
            },
        }
        monkeypatch.setattr('yaml.load', lambda f, **kwargs: system_spec)
        monkeypatch.setattr('os.path.exists', lambda path: True)
        monkeypatch.setattr('builtins.open', lambda *args, **kwargs: MagicMock())
        
        # First call should create new instance
        db1 = get_database('sys1', 'backend1', 'v1', str(tmp_path))
        assert instantiation_count == 1
        
        # Second call with same parameters should return cached instance
        db2 = get_database('sys1', 'backend1', 'v1', str(tmp_path))
        assert instantiation_count == 1  # No new instantiation
        assert db1 is db2
        
        # Different parameters should create new instance
        db3 = get_database('sys2', 'backend1', 'v1', str(tmp_path))
        assert instantiation_count == 2
        assert db3 is not db1
    
    def test_get_database_no_data_path(self, tmp_path, monkeypatch):
        """Test get_database when data path doesn't exist."""
        databases_cache.clear()
        
        system_spec = {
            'data_dir': 'data',
            'misc': {'nccl_version': 'v1'},
            'gpu': {
                'float16_tc_flops': 1000.0,
                'mem_bw': 100.0,
                'mem_bw_empirical_scaling_factor': 0.8,
                'mem_empirical_constant_latency': 0.001,
            },
            'node': {
                'inter_node_bw': 100.0,
                'intra_node_bw': 200.0,
                'num_gpus_per_node': 8,
                'p2p_latency': 0.000001,
            },
        }
        monkeypatch.setattr('yaml.load', lambda f, **kwargs: system_spec)
        monkeypatch.setattr('builtins.open', lambda *args, **kwargs: MagicMock())
        
        # Mock os.path.exists to return True for yaml, False for data path
        def mock_exists(path):
            if path.endswith('.yaml'):
                return True
            return False
        
        monkeypatch.setattr('os.path.exists', mock_exists)
        
        # Should return None when data path doesn't exist
        db = get_database('sys1', 'backend1', 'v1', str(tmp_path))
        assert db is None


@pytest.mark.patch_loader_and_yaml
class TestSupportedQuantModes:
    """Test the supported quantization modes functionality."""
    
    def test_supported_quant_modes_structure(self, comprehensive_perf_db):
        """Test that supported_quant_mode has the correct structure."""
        supported = comprehensive_perf_db.supported_quant_mode
        
        # Check all expected operations are present
        expected_ops = ['gemm', 'context_attention', 'generation_attention', 
                       'context_mla', 'generation_mla', 'mla_bmm', 'nccl', 'moe']
        
        for op in expected_ops:
            assert op in supported
            assert isinstance(supported[op], list)
            assert len(supported[op]) > 0  # Should have at least one supported mode
    
    def test_supported_quant_modes_values(self, comprehensive_perf_db):
        """Test that supported modes match the data keys."""
        # GEMM should support float16 and fp8 based on our fixture
        assert 'float16' in comprehensive_perf_db.supported_quant_mode['gemm']
        assert 'fp8' in comprehensive_perf_db.supported_quant_mode['gemm']
        
        # Context attention should support float16 and fp8
        assert 'float16' in comprehensive_perf_db.supported_quant_mode['context_attention']
        assert 'fp8' in comprehensive_perf_db.supported_quant_mode['context_attention']
        
        # MoE should support float16 and fp8
        assert 'float16' in comprehensive_perf_db.supported_quant_mode['moe']
        assert 'fp8' in comprehensive_perf_db.supported_quant_mode['moe'] 