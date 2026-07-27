# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for common SDK configurations.

Tests supported systems, model families, and other common configurations.
"""

import csv
import json
from collections import Counter
from pathlib import Path

import pytest

from aiconfigurator.sdk import common

pytestmark = pytest.mark.unit


def _find_repo_root(start: Path) -> Path:
    """Find repository root.

    In the Docker test image we copy `src/` and `tests/` into `/workspace/` but do
    not copy `pyproject.toml`, so we detect the repo root via `src/aiconfigurator/`.
    """
    start = start.resolve()
    for parent in [start, *start.parents]:
        if (parent / "src" / "aiconfigurator").is_dir():
            return parent
    raise RuntimeError("Cannot find repository root (expected src/aiconfigurator/)")


class TestSupportedSystems:
    """Test supported systems configuration."""

    def test_supported_systems_exists(self):
        """Test that SupportedSystems set exists and has content."""
        assert hasattr(common, "SupportedSystems")
        assert isinstance(common.SupportedSystems, set)
        assert len(common.SupportedSystems) > 0

    def test_supported_systems_matches_yaml_files_and_folders(self):
        """Test that SupportedSystems set matches the YAML files and data folders in systems directory."""
        repo_root = _find_repo_root(Path(__file__))
        systems_dir = repo_root / "src" / "aiconfigurator" / "systems"
        data_dir = systems_dir / "data"

        # Get all YAML files in the systems directory (excluding subdirectories)
        yaml_files = list(systems_dir.glob("*.yaml"))

        # Extract system names from YAML filenames (without .yaml extension)
        yaml_system_names = {f.stem for f in yaml_files}

        # Get all folders in the data directory
        data_folders = [f for f in data_dir.iterdir() if f.is_dir()]
        data_folder_names = {f.name for f in data_folders}

        # Assert that the YAML files match SupportedSystems
        assert common.SupportedSystems.issubset(yaml_system_names), (
            "SupportedSystems set does not match YAML files in systems directory.\n"
        )

        # Assert that the data folders match SupportedSystems
        assert common.SupportedSystems.issubset(data_folder_names), (
            "SupportedSystems set does not match data folders in systems/data directory.\n"
        )

    def test_pcie_estimate_only_systems_are_registered(self):
        """Cloud/colo PCIe systems should be available for naive and SOL-style estimates."""
        assert {"h100_pcie", "a100_pcie", "l4", "a30"}.issubset(common.SupportedSystems)

    def test_support_matrix_systems_sort_by_display_priority(self):
        """Support matrix systems should sort by product priority before name."""
        systems = [
            "b60",
            "a100_sxm",
            "gb300",
            "h100_sxm",
            "l40s",
            "b200_sxm",
            "rtx_pro_6000_server",
            "gb200",
            "h200_sxm",
            "b300_sxm",
        ]

        assert common.sort_support_matrix_systems(systems) == [
            "b200_sxm",
            "gb200",
            "b300_sxm",
            "gb300",
            "rtx_pro_6000_server",
            "h200_sxm",
            "h100_sxm",
            "l40s",
            "a100_sxm",
            "b60",
        ]


class TestSupportMatrix:
    """Test support matrix functionality."""

    def test_get_support_matrix(self):
        """Test that get_support_matrix returns a list of dictionaries."""
        matrix = common.get_support_matrix()
        assert isinstance(matrix, list)
        assert len(matrix) > 0
        assert isinstance(matrix[0], dict)
        assert "HuggingFaceID" in matrix[0]
        assert "System" in matrix[0]
        assert "Mode" in matrix[0]
        assert "Status" in matrix[0]

    def test_support_matrix_files_are_split_by_system(self):
        """Each split support matrix CSV should contain rows for exactly one system."""
        repo_root = _find_repo_root(Path(__file__))
        systems_dir = repo_root / "src" / "aiconfigurator" / "systems"
        split_dir = systems_dir / "support_matrix"

        assert split_dir.is_dir()
        assert not (systems_dir / "support_matrix.csv").exists()

        csv_paths = sorted(split_dir.glob("*.csv"))
        assert csv_paths

        for csv_path in csv_paths:
            with csv_path.open(newline="") as f:
                systems = {row["System"] for row in csv.DictReader(f)}
            assert systems == {csv_path.stem}

    def test_support_matrix_index_uses_display_order(self):
        """The static support-matrix manifest should keep the preferred system order."""
        repo_root = _find_repo_root(Path(__file__))
        index_path = repo_root / "src" / "aiconfigurator" / "systems" / "support_matrix" / "index.json"

        with index_path.open() as f:
            files = json.load(f)["files"]

        systems = [Path(file_name).stem for file_name in files]
        assert systems == [
            "b200_sxm",
            "gb200",
            "b300_sxm",
            "gb300",
            "rtx_pro_6000_server",
            "h200_sxm",
            "h100_sxm",
            "l40s",
            "a100_sxm",
            "b60",
        ]
        assert systems == common.sort_support_matrix_systems(systems)

    @pytest.mark.parametrize(
        "model,system,backend,version,architecture,expected_agg,expected_disagg",
        [
            # Known supported combination (Qwen3-32B on H200)
            ("Qwen/Qwen3-32B", "h200_sxm", None, None, None, True, True),
            # Architecture-based support for a model not in the matrix
            ("Qwen/Qwen3-235B-A22B-Thinking-2507", "h200_sxm", None, None, "Qwen3ForCausalLM", True, True),
            # Specific backend and version that should pass
            ("Qwen/Qwen3-32B", "h200_sxm", "trtllm", "1.2.0rc5", None, True, True),
            # Unsupported model
            ("non-existent-model", "h100_sxm", None, None, None, False, False),
            # Unsupported system
            ("Qwen/Qwen3-32B", "non-existent-system", None, None, None, False, False),
        ],
    )
    def test_check_support(self, model, system, backend, version, architecture, expected_agg, expected_disagg):
        """Test check_support function with various model/system combinations."""
        agg, disagg = common.check_support(model, system, backend, version, architecture)
        assert agg is expected_agg
        assert disagg is expected_disagg

    @pytest.mark.parametrize(
        "status,source",
        [
            ("HYBRID_PASS", "xshape"),
            # Transitional rows generated before HYBRID_PASS existed must not
            # masquerade as measured-silicon support either.
            ("PASS", "empirical"),
            # Current 10-column rows must name silicon explicitly.
            ("PASS", ""),
        ],
    )
    def test_check_support_does_not_count_hybrid_estimability_as_silicon_support(self, monkeypatch, status, source):
        monkeypatch.setattr(
            common,
            "get_support_matrix",
            lambda: [
                {
                    "HuggingFaceID": "test/hybrid-only",
                    "Architecture": "TestForCausalLM",
                    "System": "b200_sxm",
                    "Backend": "trtllm",
                    "Version": "1.3.0rc10",
                    "Mode": mode,
                    "Status": status,
                    "Source": source,
                }
                for mode in ("agg", "disagg")
            ],
        )

        result = common.check_support("test/hybrid-only", "b200_sxm", "trtllm", "1.3.0rc10")

        assert result.agg_supported is False
        assert result.disagg_supported is False
        assert result.exact_match is True

    def test_check_support_accepts_legacy_pass_without_source_column(self, monkeypatch):
        monkeypatch.setattr(
            common,
            "get_support_matrix",
            lambda: [
                {
                    "HuggingFaceID": "test/legacy-silicon",
                    "Architecture": "TestForCausalLM",
                    "System": "b200_sxm",
                    "Backend": "trtllm",
                    "Version": "1.3.0rc10",
                    "Mode": mode,
                    "Status": "PASS",
                }
                for mode in ("agg", "disagg")
            ],
        )

        result = common.check_support("test/legacy-silicon", "b200_sxm", "trtllm", "1.3.0rc10")

        assert result.agg_supported is True
        assert result.disagg_supported is True
        assert result.exact_match is True

    def test_architecture_fallback_does_not_count_hybrid_pass(self, monkeypatch):
        monkeypatch.setattr(
            common,
            "get_support_matrix",
            lambda: [
                {
                    "HuggingFaceID": "test/hybrid-reference",
                    "Architecture": "HybridOnlyForCausalLM",
                    "System": "b200_sxm",
                    "Backend": "trtllm",
                    "Version": "1.3.0rc10",
                    "Mode": mode,
                    "Status": "HYBRID_PASS",
                    "Source": "xshape",
                }
                for mode in ("agg", "disagg")
            ],
        )

        result = common.check_support(
            "test/unlisted-model",
            "b200_sxm",
            architecture="HybridOnlyForCausalLM",
        )

        assert result.exact_match is False
        assert result.agg_supported is False
        assert result.disagg_supported is False

    def test_supported_architectures_excludes_hybrid_only_rows(self, monkeypatch):
        monkeypatch.setattr(
            common,
            "get_support_matrix",
            lambda: [
                {"Architecture": "SiliconArch", "Status": "PASS", "Source": "silicon"},
                {"Architecture": "HybridArch", "Status": "HYBRID_PASS", "Source": "xshape"},
            ],
        )
        common.get_supported_architectures.cache_clear()
        try:
            assert common.get_supported_architectures() == {"SiliconArch"}
        finally:
            common.get_supported_architectures.cache_clear()

    @pytest.mark.parametrize(
        "model,backend,version,expected_agg,expected_disagg",
        [
            ("zai-org/GLM-5-FP8", "sglang", "0.5.10", True, True),
            ("zai-org/GLM-5-FP8", "sglang", "0.5.9", True, True),
            ("zai-org/GLM-5-FP8", "trtllm", "1.3.0rc10", True, True),
            ("nvidia/GLM-5-NVFP4", "sglang", "0.5.10", True, True),
            ("nvidia/GLM-5-NVFP4", "vllm", "0.19.0", True, True),
        ],
    )
    def test_check_support_uses_exact_glm5_b200_variant_rows(
        self, model, backend, version, expected_agg, expected_disagg
    ):
        """GLM-5 quantized variants should use their exact support rows."""
        result = common.check_support(model, "b200_sxm", backend, version, "GlmMoeDsaForCausalLM")

        assert result.agg_supported is expected_agg
        assert result.disagg_supported is expected_disagg
        assert result.exact_match is True

    def test_glm5_quantized_variants_cover_all_database_combinations(self):
        """GLM-5 quantized variants should have exact rows for every support-matrix target."""
        matrix = common.get_support_matrix()
        target_models = {"zai-org/GLM-5-FP8", "nvidia/GLM-5-NVFP4"}
        expected_keys = {(row["System"], row["Backend"], row["Version"], row["Mode"]) for row in matrix}
        for model in target_models:
            model_rows = [row for row in matrix if row["HuggingFaceID"] == model]
            model_key_counts = Counter(
                (row["System"], row["Backend"], row["Version"], row["Mode"]) for row in model_rows
            )
            model_keys = set(model_key_counts)

            assert model_keys == expected_keys
            assert all(count == 1 for count in model_key_counts.values()), (
                f"{model} has duplicate support-matrix rows for one or more keys"
            )

    def test_check_support_matches_architecture_fallback_case_insensitively(self, monkeypatch):
        """Test system/backend case normalization for architecture-based fallback."""
        monkeypatch.setattr(
            common,
            "get_support_matrix",
            lambda: [
                {
                    "HuggingFaceID": "Qwen/Qwen3-32B",
                    "Architecture": "Qwen3ForCausalLM",
                    "System": "b200_sxm",
                    "Backend": "sglang",
                    "Version": "0.5.10",
                    "Mode": "agg",
                    "Status": "PASS",
                },
                {
                    "HuggingFaceID": "Qwen/Qwen3-32B",
                    "Architecture": "Qwen3ForCausalLM",
                    "System": "b200_sxm",
                    "Backend": "sglang",
                    "Version": "0.5.10",
                    "Mode": "disagg",
                    "Status": "PASS",
                },
            ],
        )

        result = common.check_support(
            "local-qwen-variant",
            "B200_SXM",
            backend="SGLang",
            version="0.5.10",
            architecture="Qwen3ForCausalLM",
        )

        assert result.agg_supported is True
        assert result.disagg_supported is True
        assert result.exact_match is False

    def test_check_support_ignores_hardware_incompatible_rows_for_architecture_fallback(self, monkeypatch):
        """Hardware-incompatible quantized variants should not skew architecture majority voting."""
        monkeypatch.setattr(
            common,
            "get_support_matrix",
            lambda: [
                {
                    "HuggingFaceID": "Qwen/Qwen3-32B",
                    "Architecture": "Qwen3ForCausalLM",
                    "System": "a100_sxm",
                    "Backend": "trtllm",
                    "Version": "1.0.0",
                    "Mode": "agg",
                    "Status": "PASS",
                },
                {
                    "HuggingFaceID": "Qwen/Qwen3-32B-FP8",
                    "Architecture": "Qwen3ForCausalLM",
                    "System": "a100_sxm",
                    "Backend": "trtllm",
                    "Version": "1.0.0",
                    "Mode": "agg",
                    "Status": "HW_INCOMPATIBLE",
                },
                {
                    "HuggingFaceID": "Qwen/Qwen3-32B",
                    "Architecture": "Qwen3ForCausalLM",
                    "System": "a100_sxm",
                    "Backend": "trtllm",
                    "Version": "1.0.0",
                    "Mode": "disagg",
                    "Status": "PASS",
                },
                {
                    "HuggingFaceID": "Qwen/Qwen3-32B-FP8",
                    "Architecture": "Qwen3ForCausalLM",
                    "System": "a100_sxm",
                    "Backend": "trtllm",
                    "Version": "1.0.0",
                    "Mode": "disagg",
                    "Status": "HW_INCOMPATIBLE",
                },
            ],
        )

        result = common.check_support(
            "local-qwen-variant",
            "a100_sxm",
            backend="trtllm",
            version="1.0.0",
            architecture="Qwen3ForCausalLM",
        )

        assert result.agg_supported is True
        assert result.disagg_supported is True
        assert result.agg_pass_count == 1
        assert result.agg_total_count == 1


class TestEncoderLatencyColumn:
    """Test encoder_latency column is in ColumnsStatic in the correct position."""

    def test_encoder_latency_in_columns_static(self):
        assert "encoder_latency" in common.ColumnsStatic

    def test_encoder_latency_before_context_latency(self):
        cols = common.ColumnsStatic
        assert cols.index("encoder_latency") < cols.index("context_latency")

    def test_encoder_latency_before_generation_latency(self):
        cols = common.ColumnsStatic
        assert cols.index("encoder_latency") < cols.index("generation_latency")
