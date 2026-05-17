# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``collector.helper._resolve_local_model_path``."""

import json
import os
import sys
from unittest.mock import patch

import pytest

# Ensure ``collector/`` is importable since collector ships as a top-level package
# of loose scripts (not installed via pyproject).
_COLLECTOR_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "collector")
sys.path.insert(0, os.path.abspath(_COLLECTOR_DIR))

from helper import _resolve_local_model_path


class TestResolveLocalModelPath:
    def test_local_directory_passthrough(self, tmp_path):
        # An existing directory must be returned unchanged.
        (tmp_path / "config.json").write_text("{}")
        result = _resolve_local_model_path(str(tmp_path))
        assert result == str(tmp_path)

    def test_aic_cache_hit(self, tmp_path, monkeypatch):
        # Point the helper at a synthetic AIC model_configs cache.
        cache_dir = tmp_path / "model_configs"
        cache_dir.mkdir()
        slug = "fake-org--fake-model"
        (cache_dir / f"{slug}_config.json").write_text(json.dumps({"model_type": "fake"}))

        monkeypatch.setattr("helper._AIC_MODEL_CONFIG_DIR", str(cache_dir))

        result = _resolve_local_model_path("fake-org/fake-model")
        assert os.path.isdir(result)
        with open(os.path.join(result, "config.json")) as f:
            assert json.load(f) == {"model_type": "fake"}
        # No hf_quant_config side-car was provided, so none should appear.
        assert not os.path.exists(os.path.join(result, "hf_quant_config.json"))

    def test_aic_cache_hit_with_quant_side_car(self, tmp_path, monkeypatch):
        cache_dir = tmp_path / "model_configs"
        cache_dir.mkdir()
        slug = "fake-org--fake-fp8"
        (cache_dir / f"{slug}_config.json").write_text(json.dumps({"model_type": "fake"}))
        (cache_dir / f"{slug}_hf_quant_config.json").write_text(json.dumps({"quant": "fp8"}))

        monkeypatch.setattr("helper._AIC_MODEL_CONFIG_DIR", str(cache_dir))

        result = _resolve_local_model_path("fake-org/fake-fp8")
        with open(os.path.join(result, "hf_quant_config.json")) as f:
            assert json.load(f) == {"quant": "fp8"}

    def test_hf_fallback_invoked_when_not_cached(self, tmp_path, monkeypatch):
        # Empty AIC cache forces the HF download branch.
        empty_cache = tmp_path / "empty"
        empty_cache.mkdir()
        monkeypatch.setattr("helper._AIC_MODEL_CONFIG_DIR", str(empty_cache))

        hf_dir = tmp_path / "hf"
        hf_dir.mkdir()
        (hf_dir / "config.json").write_text("{}")

        def fake_hf_hub_download(repo_id, filename):
            # config.json succeeds, tokenizer files are missing — typical for many MoE models.
            target = hf_dir / filename
            if filename == "config.json":
                return str(target)
            raise FileNotFoundError(filename)

        fake_hub = type(sys)("huggingface_hub")
        fake_hub.hf_hub_download = fake_hf_hub_download
        with patch.dict(sys.modules, {"huggingface_hub": fake_hub}):
            result = _resolve_local_model_path("any-org/any-model")
        assert result == str(hf_dir)

    def test_no_hardcoded_deepseek_fallback(self, tmp_path, monkeypatch):
        # When the model is not cached AND HF download fails entirely, the helper
        # must raise — there must be no implicit "/deepseek-v3" or similar fallback.
        monkeypatch.setattr("helper._AIC_MODEL_CONFIG_DIR", str(tmp_path / "nope"))

        fake_hub = type(sys)("huggingface_hub")
        fake_hub.hf_hub_download = lambda **kw: (_ for _ in ()).throw(RuntimeError("offline"))
        # Accept positional or keyword call signatures.
        def _raise(*a, **kw):
            raise RuntimeError("offline")
        fake_hub.hf_hub_download = _raise

        with patch.dict(sys.modules, {"huggingface_hub": fake_hub}), pytest.raises(FileNotFoundError):
            _resolve_local_model_path("unknown-org/unknown-model")

    def test_empty_model_id_rejected(self):
        with pytest.raises(ValueError):
            _resolve_local_model_path("")
