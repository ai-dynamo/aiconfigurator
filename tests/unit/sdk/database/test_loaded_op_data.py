# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.perf_database import LoadedOpData, PerfDataNotAvailableError

pytestmark = pytest.mark.unit


class TestLoadedOpDataInitialization:
    """Test LoadedOpData initialization and basic properties."""

    def test_init_with_data(self):
        """Test initialization with data dictionary."""
        data = {"key1": "value1", "key2": "value2"}
        op_data = LoadedOpData(data, common.PerfDataFilename.gemm, "/path/to/file.txt")

        assert op_data.loaded is True
        assert op_data.op_name_enum == common.PerfDataFilename.gemm
        assert op_data.filepath == "/path/to/file.txt"
        assert len(op_data) == 2
        assert op_data["key1"] == "value1"
        assert op_data["key2"] == "value2"

    def test_init_with_none(self):
        """Test initialization with None (not loaded)."""
        op_data = LoadedOpData(None, common.PerfDataFilename.compute_scale, "/path/to/file.txt")

        assert op_data.loaded is False
        assert op_data.op_name_enum == common.PerfDataFilename.compute_scale
        assert op_data.filepath == "/path/to/file.txt"
        assert len(op_data) == 0

    def test_init_with_empty_dict(self):
        """Test initialization with empty dictionary."""
        op_data = LoadedOpData({}, common.PerfDataFilename.scale_matrix, "/path/to/file.txt")

        assert op_data.loaded is True
        assert len(op_data) == 0


class TestLoadedOpDataDictionaryOperations:
    """Test LoadedOpData dictionary operations when data is loaded."""

    def test_getitem_when_loaded(self):
        """Test __getitem__ works when data is loaded."""
        data = {"a": 1, "b": 2, "c": {"nested": "value"}}
        op_data = LoadedOpData(data, common.PerfDataFilename.gemm, "dummy_path")

        assert op_data["a"] == 1
        assert op_data["b"] == 2
        assert op_data["c"]["nested"] == "value"

    def test_setitem_when_loaded(self):
        """Test __setitem__ works when data is loaded."""
        data = {"a": 1}
        op_data = LoadedOpData(data, common.PerfDataFilename.gemm, "dummy_path")

        op_data["b"] = 2
        assert op_data["b"] == 2
        assert op_data["a"] == 1

        op_data["a"] = 10
        assert op_data["a"] == 10

    def test_contains_when_loaded(self):
        """Test __contains__ works when data is loaded."""
        data = {"key1": "value1", "key2": "value2"}
        op_data = LoadedOpData(data, common.PerfDataFilename.gemm, "dummy_path")

        assert "key1" in op_data
        assert "key2" in op_data
        assert "key3" not in op_data

    def test_dict_methods_when_loaded(self):
        """Test standard dict methods work when data is loaded."""
        data = {"a": 1, "b": 2, "c": 3}
        op_data = LoadedOpData(data, common.PerfDataFilename.gemm, "dummy_path")

        assert set(op_data.keys()) == {"a", "b", "c"}
        assert set(op_data.values()) == {1, 2, 3}
        assert set(op_data.items()) == {("a", 1), ("b", 2), ("c", 3)}
        assert len(op_data) == 3

    def test_iteration_when_loaded(self):
        """Test iteration works when data is loaded."""
        data = {"a": 1, "b": 2}
        op_data = LoadedOpData(data, common.PerfDataFilename.gemm, "dummy_path")

        keys = list(op_data)
        assert set(keys) == {"a", "b"}


class TestLoadedOpDataNotLoaded:
    """Test LoadedOpData operations when data is not loaded."""

    def test_getitem_when_not_loaded(self, tmp_path):
        """Test __getitem__ raises error when data is not loaded."""
        filepath = str(tmp_path / "nonexistent.txt")
        op_data = LoadedOpData(None, common.PerfDataFilename.gemm, filepath)

        with pytest.raises(PerfDataNotAvailableError) as exc_info:
            _ = op_data["key"]

        assert "Error loading silicon data for op" in str(exc_info.value)
        assert "gemm" in str(exc_info.value)
        assert filepath in str(exc_info.value)

    def test_setitem_when_not_loaded(self, tmp_path):
        """Test __setitem__ raises error when data is not loaded."""
        filepath = str(tmp_path / "nonexistent.txt")
        op_data = LoadedOpData(None, common.PerfDataFilename.compute_scale, filepath)

        with pytest.raises(PerfDataNotAvailableError) as exc_info:
            op_data["key"] = "value"

        assert "Error loading silicon data for op" in str(exc_info.value)
        assert "compute_scale" in str(exc_info.value)

    def test_contains_when_not_loaded(self, tmp_path):
        """Test __contains__ raises error when data is not loaded."""
        filepath = str(tmp_path / "nonexistent.txt")
        op_data = LoadedOpData(None, common.PerfDataFilename.scale_matrix, filepath)

        with pytest.raises(PerfDataNotAvailableError) as exc_info:
            _ = "key" in op_data

        assert "Error loading silicon data for op" in str(exc_info.value)
        assert "scale_matrix" in str(exc_info.value)


class TestLoadedOpDataRaiseIfNotLoaded:
    """Test the raise_if_not_loaded method."""

    def test_raise_if_not_loaded_when_loaded(self):
        """Test raise_if_not_loaded does nothing when data is loaded."""
        op_data = LoadedOpData({"key": "value"}, common.PerfDataFilename.gemm, "dummy_path")
        # Should not raise
        op_data.raise_if_not_loaded()

    def test_raise_if_not_loaded_file_not_exists(self, tmp_path):
        """Test raise_if_not_loaded when file doesn't exist."""
        filepath = str(tmp_path / "nonexistent.txt")
        op_data = LoadedOpData(None, common.PerfDataFilename.gemm, filepath)

        with pytest.raises(PerfDataNotAvailableError) as exc_info:
            op_data.raise_if_not_loaded()

        assert "File does not exist at" in str(exc_info.value)
        assert filepath in str(exc_info.value)
        assert "gemm" in str(exc_info.value)
        assert "SILICON mode" in str(exc_info.value)

    def test_raise_if_not_loaded_file_exists(self, tmp_path):
        """Test raise_if_not_loaded when file exists but data is None."""
        filepath = tmp_path / "exists.txt"
        filepath.write_text("dummy content")
        op_data = LoadedOpData(None, common.PerfDataFilename.compute_scale, str(filepath))

        with pytest.raises(PerfDataNotAvailableError) as exc_info:
            op_data.raise_if_not_loaded()

        assert "Unknown error loading" in str(exc_info.value)
        assert "compute_scale" in str(exc_info.value)
        assert str(filepath) in str(exc_info.value)
        assert "SILICON mode" in str(exc_info.value)


class TestLoadedOpDataIntegration:
    """Test LoadedOpData integration scenarios."""

    def test_used_in_perf_database_query(self, comprehensive_perf_db):
        """Test that LoadedOpData works when used in PerfDatabase queries."""
        # Provide enough points for 2D interpolation (>=2 keys in each axis).
        # This is similar to what we do in test_fp8_static.py
        compute_scale_data_dict = {
            common.GEMMQuantMode.fp8: {
                64: {
                    256: {"latency": 1.0, "energy": 10.0},
                    512: {"latency": 2.0, "energy": 20.0},
                },
                128: {
                    256: {"latency": 1.5, "energy": 15.0},
                    512: {"latency": 2.5, "energy": 25.0},
                },
            }
        }
        comprehensive_perf_db._compute_scale_data = LoadedOpData(
            compute_scale_data_dict, common.PerfDataFilename.compute_scale, "dummy_path"
        )

        # Query should work - test exact match
        result = comprehensive_perf_db.query_compute_scale(64, 256, common.GEMMQuantMode.fp8)
        assert float(result) == pytest.approx(1.0)
        assert result.energy == pytest.approx(10.0)

    def test_error_when_querying_unloaded_data(self, comprehensive_perf_db, tmp_path):
        """Test that querying unloaded data raises appropriate error."""
        filepath = str(tmp_path / "nonexistent.txt")
        comprehensive_perf_db._compute_scale_data = LoadedOpData(None, common.PerfDataFilename.compute_scale, filepath)

        # Query should raise error in SILICON mode
        with pytest.raises(PerfDataNotAvailableError):
            comprehensive_perf_db.query_compute_scale(64, 256, common.GEMMQuantMode.fp8)
