<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# PerfDatabase Test Coverage Summary

This document summarizes the comprehensive pytest test coverage for `perf_database.py`.

## Test Files Created

### 1. **conftest.py**
- Added comprehensive fixtures for testing:
  - `perf_db`: Basic PerfDatabase instance with minimal data
  - `comprehensive_perf_db`: PerfDatabase with complete test data for all operations

### 2. **test_data_loaders.py** 
Tests for all data loading functions:
- `test_get_database()` - Database retrieval and caching
- `test_get_all_databases()` - Batch database loading
- Tests for each loader function:
  - `load_custom_allreduce_data()`
  - `load_nccl_data()`
  - `load_gemm_data()`
  - `load_moe_data()`
  - `load_context_attention_data()`
  - `load_generation_attention_data()`
  - `load_context_mla_data()`
  - `load_generation_mla_data()`
  - `load_mla_bmm_data()`

### 3. **test_perf_database.py**
Basic query method tests:
- `query_gemm()` - Exact match test
- `query_custom_allreduce()` - SOL, SOL_FULL, and SILICON modes
- `query_nccl()` - SOL mode for different operations
- `query_p2p()` - SOL mode
- System spec loading verification

### 4. **test_perf_database_attention.py**
Comprehensive tests for attention methods:
- **TestContextAttention**:
  - SOL mode calculation
  - SOL_FULL mode tuple return
  - SILICON mode for MHA and XQA
  - Assertion error for invalid n_kv
- **TestGenerationAttention**:
  - SOL mode calculation
  - SOL_FULL mode tuple return
  - SILICON mode with interpolation
  - Edge cases (s=1)
- **TestContextMLA**:
  - SOL mode calculation
  - SILICON mode with data lookup
  - Different tensor parallelism sizes
- **TestGenerationMLA**:
  - SOL mode calculation
  - SILICON mode with data lookup
  - SOL_FULL mode tuple return
- Default SOL mode setting/getting

### 5. **test_perf_database_moe_mla.py**
Tests for MoE and MLA BMM operations:
- **TestMoE**:
  - SOL mode calculation verification
  - SOL_FULL mode tuple return
  - SILICON mode with data lookup
  - Different workload distributions
  - Edge cases (single token, large EP size)
- **TestMLABMM**:
  - SOL mode for pre/post operations
  - SOL_FULL mode tuple return
  - SILICON mode for pre/post
  - Different configurations
- **TestMemoryOperations**:
  - SOL mode calculation
  - SOL_FULL mode tuple return
  - SILICON mode with empirical scaling
  - Edge cases (zero bytes, small/large transfers)
- **TestP2P**:
  - SOL mode calculation
  - SOL_FULL mode tuple return
  - SILICON mode with P2P latency
  - Edge cases

### 6. **test_perf_database_interpolation.py**
Tests for interpolation helper methods:
- **TestInterpolationMethods**:
  - `_nearest_1d_point_helper()` - inner/outer modes, error cases
  - `_validate()` - negative value detection
  - `_interp_1d()` - linear interpolation
  - `_bilinear_interpolation()` - 2D interpolation
  - `_interp_3d_linear()` - 3D linear interpolation
  - `_interp_2d_1d()` - bilinear/cubic methods
  - `_interp_3d()` - dispatcher method
- **TestExtrapolateDataGrid**:
  - Basic extrapolation functionality
  - Extrapolation with sqrt_y_value
  - Edge cases (insufficient data)
  - Boundary extension
- **TestCorrectData**:
  - GEMM data correction based on SOL
  - Generation attention data correction
- **TestUpdateSupportMatrix**:
  - Support matrix structure verification
  - Supported quant modes validation

### 7. **test_perf_database_edge_cases.py**
Edge cases and additional coverage:
- **TestNcclEdgeCases**:
  - Single GPU returns 0
  - SILICON interpolation
  - Large GPU count scaling
  - Edge message sizes
- **TestAllreduceEdgeCases**:
  - Single GPU returns 0
  - Large TP scaling (>8 GPUs)
  - Message size extrapolation
- **TestInitializationEdgeCases**:
  - Initialization with missing loaders
  - Extrapolation during initialization
- **TestGemmInterpolation**:
  - Interpolation between known points
  - Extrapolation beyond data range
- **TestDatabaseCache**:
  - Database caching functionality
  - Missing data path handling
- **TestSupportedQuantModes**:
  - Support matrix structure
  - Supported modes validation

## Coverage Statistics

### Methods Covered:
1. **Query Methods** (100%):
   - `query_gemm()` ✓
   - `query_context_attention()` ✓
   - `query_generation_attention()` ✓
   - `query_context_mla()` ✓
   - `query_generation_mla()` ✓
   - `query_custom_allreduce()` ✓
   - `query_nccl()` ✓
   - `query_moe()` ✓
   - `query_mla_bmm()` ✓
   - `query_mem_op()` ✓
   - `query_p2p()` ✓

2. **Interpolation Methods** (100%):
   - `_nearest_1d_point_helper()` ✓
   - `_validate()` ✓
   - `_interp_1d()` ✓
   - `_bilinear_interpolation()` ✓
   - `_interp_3d_linear()` ✓
   - `_interp_2d_1d()` ✓
   - `_interp_3d()` ✓
   - `_extrapolate_data_grid()` ✓

3. **Helper Methods** (100%):
   - `_correct_data()` ✓
   - `_update_support_matrix()` ✓
   - `set_default_database_mode()` ✓
   - `get_default_database_mode()` ✓

4. **Data Loading Functions** (100%):
   - All loader functions tested

5. **Module Functions** (100%):
   - `get_database()` ✓
   - `get_all_databases()` ✓

### Test Scenarios Covered:
- **SOL modes**: SOL, SOL_FULL, SILICON
- **Edge cases**: Single GPU, zero values, out-of-range interpolation
- **Scaling**: Large GPU counts, message sizes
- **Error handling**: Invalid inputs, missing data
- **Caching**: Database instance caching
- **Initialization**: With/without data files

## Running the Tests

To run all PerfDatabase tests:
```bash
pytest tests/test_perf_database*.py tests/test_data_loaders.py -v
```
To run all PerfDatabase tests and dump results to html:
```bash
pytest tests/test_perf_database*.py tests/test_data_loaders.py -v --html=report.html
```

To run with coverage report:
```bash
pytest tests/test_perf_database*.py tests/test_data_loaders.py --cov=aiconfigurator.sdk.perf_database --cov-report=html
```

## Notes
- All tests use fixtures to avoid file I/O operations
- Comprehensive test data is generated programmatically in fixtures
- Tests verify both correctness and edge case handling
- Interpolation tests ensure numerical accuracy within reasonable tolerances 