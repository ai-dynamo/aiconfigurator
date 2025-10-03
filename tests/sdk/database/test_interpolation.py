# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
from collections import defaultdict
from aiconfigurator.sdk.perf_database import PerfDatabase
from aiconfigurator.sdk import common

class TestInterpolationMethods:
    """Test cases for interpolation helper methods."""
    
    def test_nearest_1d_point_helper_inner(self, comprehensive_perf_db):
        """Test _nearest_1d_point_helper with inner_only=True."""
        values = [1, 5, 10, 20, 30]
        
        # Test value in the middle
        left, right = comprehensive_perf_db._nearest_1d_point_helper(7, values, inner_only=True)
        assert left == 5
        assert right == 10
        
        # Test exact match
        left, right = comprehensive_perf_db._nearest_1d_point_helper(10, values, inner_only=True)
        assert left == 10
        assert right == 20
        
        # Test at boundaries
        left, right = comprehensive_perf_db._nearest_1d_point_helper(1, values, inner_only=True)
        assert left == 1
        assert right == 5
    
    def test_nearest_1d_point_helper_outer(self, comprehensive_perf_db):
        """Test _nearest_1d_point_helper with inner_only=False."""
        values = [10, 20, 30]
        
        # Test value below range
        left, right = comprehensive_perf_db._nearest_1d_point_helper(5, values, inner_only=False)
        assert left == 10
        assert right == 20
        
        # Test value above range
        left, right = comprehensive_perf_db._nearest_1d_point_helper(40, values, inner_only=False)
        assert left == 20
        assert right == 30
    
    def test_nearest_1d_point_helper_errors(self, comprehensive_perf_db):
        """Test error cases for _nearest_1d_point_helper."""
        # Empty list
        with pytest.raises(AssertionError):
            comprehensive_perf_db._nearest_1d_point_helper(10, [], inner_only=True)
        
        # Single value list
        with pytest.raises(AssertionError):
            comprehensive_perf_db._nearest_1d_point_helper(10, [5], inner_only=True)
        
        # Value out of range with inner_only=True
        with pytest.raises(ValueError):
            comprehensive_perf_db._nearest_1d_point_helper(0, [10, 20], inner_only=True)
        
        with pytest.raises(ValueError):
            comprehensive_perf_db._nearest_1d_point_helper(30, [10, 20], inner_only=True)
    
    def test_validate(self, comprehensive_perf_db, caplog):
        """Test _validate method for negative value detection."""
        # Positive value should pass through
        assert comprehensive_perf_db._validate(10.5) == 10.5
        
        # Zero should pass through
        assert comprehensive_perf_db._validate(0.0) == 0.0
        
        # Negative value should log debug but still return
        with caplog.at_level('DEBUG'):
            result = comprehensive_perf_db._validate(-5.0)
            assert result == -5.0
            assert 'Negative value detected' in caplog.text
    
    def test_interp_1d(self, comprehensive_perf_db):
        """Test 1D interpolation."""
        # Linear interpolation
        x = [10, 20]
        y = [100, 200]
        
        # Middle value
        result = comprehensive_perf_db._interp_1d(x, y, 15)
        assert result == 150.0
        
        # At boundaries
        result = comprehensive_perf_db._interp_1d(x, y, 10)
        assert result == 100.0
        
        result = comprehensive_perf_db._interp_1d(x, y, 20)
        assert result == 200.0
        
        # Extrapolation
        result = comprehensive_perf_db._interp_1d(x, y, 25)
        assert result == 250.0
    
    def test_bilinear_interpolation(self, comprehensive_perf_db):
        """Test bilinear interpolation."""
        # Create a simple 2x2 grid
        data = {
            10: {20: 100, 40: 200},
            30: {20: 300, 40: 400}
        }
        
        # Test interpolation in the middle
        result = comprehensive_perf_db._bilinear_interpolation([10, 30], [20, 40], 20, 30, data)
        expected = 250.0  # Average of all four corners
        assert abs(result - expected) < 1e-6
        
        # Test at corner points
        result = comprehensive_perf_db._bilinear_interpolation([10, 30], [20, 40], 10, 20, data)
        assert result == 100.0
        
        # Test along edges
        result = comprehensive_perf_db._bilinear_interpolation([10, 30], [20, 40], 10, 30, data)
        assert result == 150.0  # Average of 100 and 200
    
    def test_interp_3d_linear(self, comprehensive_perf_db):
        """Test 3D linear interpolation."""
        # Create a simple 3D data structure
        data = defaultdict(lambda: defaultdict(lambda: defaultdict()))
        # Define a cube with values at corners
        for x in [10, 20]:
            for y in [30, 40]:
                for z in [50, 60]:
                    data[x][y][z] = x + y + z
        
        # Test interpolation at center
        result = comprehensive_perf_db._interp_3d_linear(15, 35, 55, data)
        expected = 15 + 35 + 55  # Linear function
        assert abs(result - expected) < 1e-6
        
        # Test at corner point
        result = comprehensive_perf_db._interp_3d_linear(10, 30, 50, data)
        assert result == 90.0
    
    def test_interp_2d_1d(self, comprehensive_perf_db):
        """Test 2D-1D interpolation methods."""
        # Create test data
        data = defaultdict(lambda: defaultdict(lambda: defaultdict()))
        for x in [10, 20]:
            for y in [30, 40]:
                for z in [50, 60]:
                    data[x][y][z] = x * 0.1 + y * 0.2 + z * 0.3
        
        # Test bilinear method
        result_bilinear = comprehensive_perf_db._interp_2d_1d(15, 35, 55, data, method='bilinear')
        assert result_bilinear > 0
        
        # Test cubic method (if scipy is available)
        result_cubic = comprehensive_perf_db._interp_2d_1d(15, 35, 55, data, method='cubic')
        assert result_cubic > 0
        
        # Invalid method should raise error
        with pytest.raises(NotImplementedError):
            comprehensive_perf_db._interp_2d_1d(15, 35, 55, data, method='invalid')
    
    def test_interp_3d(self, comprehensive_perf_db):
        """Test general 3D interpolation dispatcher."""
        data = defaultdict(lambda: defaultdict(lambda: defaultdict()))
        for x in [10, 20]:
            for y in [30, 40]:
                for z in [50, 60]:
                    data[x][y][z] = x + y + z
        
        # Test linear method
        result_linear = comprehensive_perf_db._interp_3d(15, 35, 55, data, 'linear')
        assert result_linear > 0
        
        # Test fallback to 2D-1D
        result_bilinear = comprehensive_perf_db._interp_3d(15, 35, 55, data, 'bilinear')
        assert result_bilinear > 0


class TestExtrapolateDataGrid:
    """Test cases for _extrapolate_data_grid method."""
    
    def test_extrapolate_data_grid_basic(self, comprehensive_perf_db):
        """Test basic extrapolation functionality."""
        # Create simple test data
        data_dict = defaultdict(lambda: defaultdict(lambda: defaultdict()))
        data_dict[10][20][30] = 100
        data_dict[10][20][40] = 110
        data_dict[10][30][30] = 120
        data_dict[10][30][40] = 130
        data_dict[20][20][30] = 200
        data_dict[20][20][40] = 210
        data_dict[20][30][30] = 220
        data_dict[20][30][40] = 230
        
        # Define target lists
        target_x_list = [10, 15, 20]
        target_y_list = [20, 25, 30]
        target_z_list = [30, 35, 40]
        
        # Apply extrapolation
        comprehensive_perf_db._extrapolate_data_grid(
            data_dict, target_x_list, target_y_list, target_z_list
        )
        
        # Check that new points were created
        assert 15 in data_dict  # New x value
        assert 25 in data_dict[10]  # New y value
        assert 35 in data_dict[10][20]  # New z value
        
        # Verify interpolated values make sense
        assert data_dict[15][20][30] > 100  # Should be between 100 and 200
        assert data_dict[15][20][30] < 200
    
    def test_extrapolate_data_grid_with_sqrt(self, comprehensive_perf_db):
        """Test extrapolation with sqrt_y_value option."""
        # Create test data where y dimension benefits from sqrt scaling
        data_dict = defaultdict(lambda: defaultdict(lambda: defaultdict()))
        # Use quadratic relationship in y
        for x in [10, 20]:
            for y in [4, 16]:  # Square numbers
                for z in [10, 20]:
                    data_dict[x][y][z] = x + y * y + z  # Quadratic in y
        
        target_x_list = [10, 15, 20]
        target_y_list = [4, 9, 16]  # Include 9 which is between 4 and 16
        target_z_list = [10, 15, 20]
        
        # Apply extrapolation with sqrt
        comprehensive_perf_db._extrapolate_data_grid(
            data_dict, target_x_list, target_y_list, target_z_list,
            sqrt_y_value=True
        )
        
        # Check that interpolation happened
        assert 9 in data_dict[10]
        # With sqrt scaling, the interpolation should be more accurate for quadratic data
        # Expected value at (10, 9, 10) should be 10 + 9*9 + 10 = 101
        assert abs(data_dict[10][9][10] - 101) < 20  # Some tolerance for sqrt approximation
    
    def test_extrapolate_data_grid_edge_cases(self, comprehensive_perf_db, caplog):
        """Test edge cases for extrapolation."""
        # Test with minimal data (single point in z dimension)
        data_dict = defaultdict(lambda: defaultdict(lambda: defaultdict()))
        data_dict[10][20][30] = 100
        
        target_x_list = [10]
        target_y_list = [20]
        target_z_list = [30, 40]  # Try to extrapolate in z
        
        with caplog.at_level('WARNING'):
            comprehensive_perf_db._extrapolate_data_grid(
                data_dict, target_x_list, target_y_list, target_z_list
            )
            # Should warn about insufficient data
            assert 'only one data point' in caplog.text
    
    def test_extrapolate_data_grid_boundary_extension(self, comprehensive_perf_db):
        """Test extrapolation beyond original data boundaries."""
        data_dict = defaultdict(lambda: defaultdict(lambda: defaultdict()))
        # Create 2x2x2 cube
        for x in [10, 20]:
            for y in [30, 40]:
                for z in [50, 60]:
                    data_dict[x][y][z] = x + y + z
        
        # Target lists that extend beyond original data
        target_x_list = [5, 10, 20, 30]  # 5 and 30 are outside
        target_y_list = [20, 30, 40, 50]  # 20 and 50 are outside
        target_z_list = [40, 50, 60, 70]  # 40 and 70 are outside
        
        comprehensive_perf_db._extrapolate_data_grid(
            data_dict, target_x_list, target_y_list, target_z_list
        )
        
        # Check extrapolated values exist
        assert 5 in data_dict
        assert 30 in data_dict
        assert 20 in data_dict[10]
        assert 50 in data_dict[10]
        assert 40 in data_dict[10][30]
        assert 70 in data_dict[10][30]
        
        # Verify extrapolated values follow the linear pattern
        assert data_dict[5][30][50] == 5 + 30 + 50  # Linear extrapolation
        assert data_dict[30][40][60] == 30 + 40 + 60


class TestCorrectData:
    """Test cases for _correct_data method."""
    
    def test_correct_gemm_data(self, comprehensive_perf_db, caplog):
        """Test that _correct_data adjusts GEMM data based on SOL."""
        # Manually set a GEMM value that's too optimistic (lower than SOL)
        quant_mode = common.GEMMQuantMode.float16
        m, n, k = 64, 128, 256
        
        # Calculate what SOL should be
        sol_value = comprehensive_perf_db.query_gemm(m, n, k, quant_mode, sol_mode=common.SOLMode.SOL)
        
        # Set an artificially low value
        comprehensive_perf_db._gemm_data[quant_mode][m][n][k] = sol_value * 0.5
        
        # Run correction
        with caplog.at_level('DEBUG'):
            comprehensive_perf_db._correct_data()
            
        # Check that the value was corrected
        assert comprehensive_perf_db._gemm_data[quant_mode][m][n][k] >= sol_value
        assert 'sol {} > perf_db'.format(sol_value) in caplog.text or 'gemm quant' in caplog.text
    
    def test_correct_generation_attention_data(self, comprehensive_perf_db, caplog):
        """Test that _correct_data adjusts generation attention data."""
        kv_cache_quant_mode = common.KVCacheQuantMode.float16
        n_kv = 0  # MHA case
        n, b, s = 16, 4, 64
        
        # Calculate SOL
        sol_value = comprehensive_perf_db.query_generation_attention(
            b, s, n, n, kv_cache_quant_mode, sol_mode=common.SOLMode.SOL
        )
        
        # Set an artificially low value
        comprehensive_perf_db._generation_attention_data[kv_cache_quant_mode][n_kv][n][b][s] = sol_value * 0.5
        
        # Run correction
        with caplog.at_level('DEBUG'):
            comprehensive_perf_db._correct_data()
        
        # Check that the value was corrected
        corrected_value = comprehensive_perf_db._generation_attention_data[kv_cache_quant_mode][n_kv][n][b][s]
        assert corrected_value >= sol_value


class TestUpdateSupportMatrix:
    """Test cases for _update_support_matrix method."""
    
    def test_support_matrix_creation(self, comprehensive_perf_db):
        """Test that supported_quant_mode is properly created."""
        # The fixture should have already called _update_support_matrix
        assert hasattr(comprehensive_perf_db, 'supported_quant_mode')
        assert isinstance(comprehensive_perf_db.supported_quant_mode, dict)
        
        # Check expected keys
        expected_keys = [
            'gemm', 'context_attention', 'generation_attention',
            'context_mla', 'generation_mla', 'mla_bmm', 'nccl', 'moe'
        ]
        for key in expected_keys:
            assert key in comprehensive_perf_db.supported_quant_mode
            assert isinstance(comprehensive_perf_db.supported_quant_mode[key], list)
        
        # Verify some expected quant modes
        assert 'float16' in comprehensive_perf_db.supported_quant_mode['gemm']
        assert 'float16' in comprehensive_perf_db.supported_quant_mode['context_attention'] 