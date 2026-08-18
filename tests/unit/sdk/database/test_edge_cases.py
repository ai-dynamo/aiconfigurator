# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.perf_database import PerfDatabase, databases_cache, get_database

pytestmark = pytest.mark.unit


# Retired with #1357 PR-5: the NCCL / custom-allreduce / GEMM query edge-case
# math this file pinned on the synthetic fixture (single-GPU zero, silicon
# interpolation, tp scaling, extrapolation) moved to the compiled engine and
# is anchored by the frozen parity goldens.


class TestInitializationEdgeCases:
    """Test edge cases during PerfDatabase initialization."""

    def test_load_binds_raw_grid_without_pre_expansion(self, tmp_path, monkeypatch, caplog):
        """ContextAttention.load_data binds the RAW grid, without densifying it.

        Earlier the op pre-expanded the grid at load time (eager extrapolation
        during ``PerfDatabase.__init__``). perf_interp removed that: interp and
        util-hold happen at query time on the raw grid, so loading must leave
        the four collected points untouched. Guards against re-introducing
        load-time pre-expansion."""
        # Set up minimal system spec
        import yaml

        dummy_system_spec = {
            "data_dir": "data",
            "misc": {"nccl_version": "v1"},
            "gpu": {
                "bfloat16_tc_flops": 1000.0,
                "mem_bw": 100.0,
                "mem_bw_empirical_scaling_factor": 0.8,
                "mem_empirical_constant_latency": 0.001,
            },
            "node": {
                "inter_node_bw": 100.0,
                "intra_node_bw": 200.0,
                "num_gpus_per_node": 8,
                "p2p_latency": 0.000001,
            },
        }
        yaml_file = tmp_path / "test.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(dummy_system_spec, f)

        monkeypatch.setattr("yaml.load", lambda stream, Loader=None: dummy_system_spec)  # noqa: N803

        # Create minimal context attention data
        from collections import defaultdict

        dummy_context_data = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(
                        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))
                    )
                )
            )
        )
        dummy_context_data[common.FMHAQuantMode.bfloat16][common.KVCacheQuantMode.bfloat16][0][128][0][4][16][1] = 0.1
        dummy_context_data[common.FMHAQuantMode.bfloat16][common.KVCacheQuantMode.bfloat16][0][128][0][4][32][1] = 0.2
        dummy_context_data[common.FMHAQuantMode.bfloat16][common.KVCacheQuantMode.bfloat16][0][128][0][8][16][1] = 0.15
        dummy_context_data[common.FMHAQuantMode.bfloat16][common.KVCacheQuantMode.bfloat16][0][128][0][8][32][1] = 0.25

        # Op load_data binds engine table views (PR-6): stub the single
        # fetch choke point by attribute name. Everything but the context
        # attention table fetches as None ("no source files").
        monkeypatch.setattr(
            "aiconfigurator_core.sdk.engine_table_view.fetch_table_view",
            lambda database, attribute: dummy_context_data if attribute == "_context_attention_data" else None,
        )

        # Initialize database, then trigger the lazy load explicitly so
        # extrapolation runs while loader patches are still active.
        from aiconfigurator.sdk.operations.attention import ContextAttention

        db = PerfDatabase("test", "backend", "v1", str(tmp_path))
        ContextAttention.load_data(db)

        # No load-time pre-expansion: the four collected points are preserved
        # verbatim (perf_interp interpolates/holds lazily at query time).
        total_points = 0
        for quant_mode in db._context_attention_data:
            for kv_cache in db._context_attention_data[quant_mode]:
                for kv_n in db._context_attention_data[quant_mode][kv_cache]:
                    for head_size in db._context_attention_data[quant_mode][kv_cache][kv_n]:
                        for window_size in db._context_attention_data[quant_mode][kv_cache][kv_n][head_size]:
                            for n in db._context_attention_data[quant_mode][kv_cache][kv_n][head_size][window_size]:
                                for s in db._context_attention_data[quant_mode][kv_cache][kv_n][head_size][window_size][
                                    n
                                ]:
                                    total_points += len(
                                        db._context_attention_data[quant_mode][kv_cache][kv_n][head_size][window_size][
                                            n
                                        ][s]
                                    )

        assert total_points == 4, "Load must preserve the raw grid; pre-expansion was removed"


class TestDatabaseCache:
    """Test database caching functionality."""

    def test_get_database_caching(self, tmp_path, monkeypatch):
        """Test that get_database properly caches instances."""
        # Clear cache first
        databases_cache.clear()

        # Mock PerfDatabase to track instantiations
        instantiation_count = 0

        def counting_init(self, *args, **kwargs):
            nonlocal instantiation_count
            instantiation_count += 1
            # Don't actually initialize to avoid file operations
            self.system = args[0]
            self.backend = args[1]
            self.version = args[2]
            self._default_database_mode = common.DatabaseMode.SILICON

        monkeypatch.setattr(PerfDatabase, "__init__", counting_init)

        # Real synthetic tree: <system>.yaml pointing at data/<system>, with a
        # legacy-layout backend/version dir holding a stub perf file so the
        # declaration check (os.listdir/os.path.isdir) finds real directories.
        for system in ("sys1", "sys2"):
            (tmp_path / f"{system}.yaml").write_text(f"data_dir: data/{system}\n", encoding="utf-8")
            version_dir = tmp_path / "data" / system / "backend1" / "v1"
            version_dir.mkdir(parents=True)
            (version_dir / "gemm_perf.parquet").write_bytes(b"PAR1stub")

        # First call should create new instance
        db1 = get_database("sys1", "backend1", "v1", systems_paths=str(tmp_path))
        assert instantiation_count == 1

        # Second call with same parameters should return cached instance
        db2 = get_database("sys1", "backend1", "v1", systems_paths=str(tmp_path))
        assert instantiation_count == 1  # No new instantiation
        assert db1 is db2

        # Different parameters should create new instance
        db3 = get_database("sys2", "backend1", "v1", systems_paths=str(tmp_path))
        assert instantiation_count == 2
        assert db3 is not db1

    def test_get_database_no_data_path(self, tmp_path, monkeypatch):
        """Test get_database when data path doesn't exist."""
        databases_cache.clear()

        system_spec = {
            "data_dir": "data",
            "misc": {"nccl_version": "v1"},
            "gpu": {
                "bfloat16_tc_flops": 1000.0,
                "mem_bw": 100.0,
                "mem_bw_empirical_scaling_factor": 0.8,
                "mem_empirical_constant_latency": 0.001,
            },
            "node": {
                "inter_node_bw": 100.0,
                "intra_node_bw": 200.0,
                "num_gpus_per_node": 8,
                "p2p_latency": 0.000001,
            },
        }
        monkeypatch.setattr("yaml.load", lambda f, **kwargs: system_spec)
        monkeypatch.setattr("builtins.open", lambda *args, **kwargs: MagicMock())
        (tmp_path / "sys1.yaml").write_text("dummy")

        # Mock os.path.exists to return True for yaml, False for data path
        def mock_exists(path):
            return path.endswith(".yaml")

        monkeypatch.setattr("os.path.exists", mock_exists)

        # Should return None when data path doesn't exist
        db = get_database("sys1", "backend1", "v1", systems_paths=str(tmp_path))
        assert db is None


class TestSupportedQuantModes:
    """Test the supported quantization modes functionality."""

    def test_supported_quant_modes_structure(self, comprehensive_perf_db):
        """Test that supported_quant_mode has the correct structure."""
        supported = comprehensive_perf_db.supported_quant_mode

        # Check all expected operations are present
        expected_ops = [
            "gemm",
            "context_attention",
            "generation_attention",
            "context_mla",
            "generation_mla",
            "mla_bmm",
            "nccl",
            "moe",
        ]

        for op in expected_ops:
            assert op in supported
            assert isinstance(supported[op], list)
            assert len(supported[op]) > 0  # Should have at least one supported mode

    def test_supported_quant_modes_values(self, comprehensive_perf_db):
        """Test that supported modes match the data keys."""
        # GEMM should support bfloat16 and fp8 based on our fixture
        assert "bfloat16" in comprehensive_perf_db.supported_quant_mode["gemm"]
        assert "fp8" in comprehensive_perf_db.supported_quant_mode["gemm"]

        # Context attention should support bfloat16 and fp8
        assert "bfloat16" in comprehensive_perf_db.supported_quant_mode["context_attention"]
        assert "fp8" in comprehensive_perf_db.supported_quant_mode["context_attention"]

        # MoE should support bfloat16 and fp8
        assert "bfloat16" in comprehensive_perf_db.supported_quant_mode["moe"]
        assert "fp8" in comprehensive_perf_db.supported_quant_mode["moe"]
