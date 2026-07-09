# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``collector.helper._resolve_local_model_path``."""

import json
import os
import sys
from unittest.mock import patch

import pytest

# Ensure the upper AIC project's ``collector/`` scripts are directly importable.
_COLLECTOR_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "packages", "aiconfigurator", "collector")
sys.path.insert(0, os.path.abspath(_COLLECTOR_DIR))

from helper import _resolve_local_model_path


@pytest.fixture
def isolated_tmp(tmp_path, monkeypatch):
    """Redirect tempfile.gettempdir() so the helper's deterministic per-slug
    cache lives under pytest's tmp_path and is auto-cleaned between tests."""
    tmp_root = tmp_path / "tmpdir"
    tmp_root.mkdir()
    monkeypatch.setattr("helper.tempfile.gettempdir", lambda: str(tmp_root))
    return tmp_root


class TestResolveLocalModelPath:
    def test_local_directory_passthrough(self, tmp_path):
        (tmp_path / "config.json").write_text("{}")
        result = _resolve_local_model_path(str(tmp_path))
        assert result == str(tmp_path)

    def test_local_path_is_file_rejected(self, tmp_path):
        # A MOE_MODEL_PATH pointing at a file (not a directory) must fail loudly
        # rather than silently falling through to an HF download attempt.
        f = tmp_path / "config.json"
        f.write_text("{}")
        with pytest.raises(NotADirectoryError):
            _resolve_local_model_path(str(f))

    def test_local_dir_missing_config_rejected(self, tmp_path):
        # An existing directory without config.json must fail at the helper
        # rather than deferring to SGLang startup.
        with pytest.raises(FileNotFoundError):
            _resolve_local_model_path(str(tmp_path))

    def test_aic_cache_hit(self, isolated_tmp, tmp_path, monkeypatch):
        cache_dir = tmp_path / "model_configs"
        cache_dir.mkdir()
        slug = "fake-org--fake-model"
        (cache_dir / f"{slug}_config.json").write_text(json.dumps({"model_type": "fake"}))

        monkeypatch.setattr("helper._AIC_MODEL_CONFIG_DIR", str(cache_dir))

        result = _resolve_local_model_path("fake-org/fake-model")
        assert os.path.isdir(result)
        with open(os.path.join(result, "config.json")) as f:
            assert json.load(f) == {"model_type": "fake"}
        assert not os.path.exists(os.path.join(result, "hf_quant_config.json"))

    def test_aic_cache_strips_auto_map(self, isolated_tmp, tmp_path, monkeypatch):
        # auto_map references .py files that AIC does not ship; with
        # trust_remote_code=True, transformers would try to import them and
        # crash. The helper must strip auto_map when materializing the cache.
        cache_dir = tmp_path / "model_configs"
        cache_dir.mkdir()
        slug = "fake-org--fake-with-automap"
        (cache_dir / f"{slug}_config.json").write_text(
            json.dumps(
                {
                    "model_type": "fake",
                    "auto_map": {
                        "AutoConfig": "configuration_fake.FakeConfig",
                        "AutoModel": "modeling_fake.FakeModel",
                    },
                }
            )
        )
        monkeypatch.setattr("helper._AIC_MODEL_CONFIG_DIR", str(cache_dir))

        result = _resolve_local_model_path("fake-org/fake-with-automap")
        with open(os.path.join(result, "config.json")) as f:
            materialized = json.load(f)
        assert "auto_map" not in materialized
        assert materialized["model_type"] == "fake"

    def test_aic_cache_hit_with_quant_side_car(self, isolated_tmp, tmp_path, monkeypatch):
        cache_dir = tmp_path / "model_configs"
        cache_dir.mkdir()
        slug = "fake-org--fake-fp8"
        (cache_dir / f"{slug}_config.json").write_text(json.dumps({"model_type": "fake"}))
        (cache_dir / f"{slug}_hf_quant_config.json").write_text(json.dumps({"quant": "fp8"}))

        monkeypatch.setattr("helper._AIC_MODEL_CONFIG_DIR", str(cache_dir))

        result = _resolve_local_model_path("fake-org/fake-fp8")
        with open(os.path.join(result, "hf_quant_config.json")) as f:
            assert json.load(f) == {"quant": "fp8"}

    def test_aic_cache_is_deterministic_across_calls(self, isolated_tmp, tmp_path, monkeypatch):
        # Two calls with the same model_id must converge on the same tempdir
        # so parallel subprocesses (wideep collector spawns one per GPU) don't
        # each create their own divergent copy.
        cache_dir = tmp_path / "model_configs"
        cache_dir.mkdir()
        slug = "fake-org--deterministic"
        (cache_dir / f"{slug}_config.json").write_text(json.dumps({"model_type": "fake"}))
        monkeypatch.setattr("helper._AIC_MODEL_CONFIG_DIR", str(cache_dir))

        first = _resolve_local_model_path("fake-org/deterministic")
        second = _resolve_local_model_path("fake-org/deterministic")
        assert first == second

    def test_hf_fallback_invoked_when_not_cached(self, tmp_path, monkeypatch):
        empty_cache = tmp_path / "empty"
        empty_cache.mkdir()
        monkeypatch.setattr("helper._AIC_MODEL_CONFIG_DIR", str(empty_cache))

        hf_dir = tmp_path / "hf"
        hf_dir.mkdir()
        (hf_dir / "config.json").write_text("{}")

        def fake_hf_hub_download(repo_id, filename):
            # config.json succeeds; tokenizer files are missing — typical for MoE models.
            target = hf_dir / filename
            if filename == "config.json":
                return str(target)
            raise FileNotFoundError(filename)

        fake_hub = type(sys)("huggingface_hub")
        fake_hub.hf_hub_download = fake_hf_hub_download
        with patch.dict(sys.modules, {"huggingface_hub": fake_hub}):
            result = _resolve_local_model_path("any-org/any-model")
        assert result == str(hf_dir)

    def test_hf_fallback_raises_if_config_json_missing(self, tmp_path, monkeypatch):
        # If config.json itself fails to download we must raise, even if a
        # tokenizer file happens to land first. The pre-fix code would have
        # returned a snapshot dir without config.json.
        empty_cache = tmp_path / "empty"
        empty_cache.mkdir()
        monkeypatch.setattr("helper._AIC_MODEL_CONFIG_DIR", str(empty_cache))

        def fake_hf_hub_download(repo_id, filename):
            raise RuntimeError(f"network down ({filename})")

        fake_hub = type(sys)("huggingface_hub")
        fake_hub.hf_hub_download = fake_hf_hub_download
        with patch.dict(sys.modules, {"huggingface_hub": fake_hub}), pytest.raises(FileNotFoundError):
            _resolve_local_model_path("any-org/any-model")

    def test_no_hardcoded_deepseek_fallback(self, tmp_path, monkeypatch):
        monkeypatch.setattr("helper._AIC_MODEL_CONFIG_DIR", str(tmp_path / "nope"))

        def _raise(*a, **kw):
            raise RuntimeError("offline")

        fake_hub = type(sys)("huggingface_hub")
        fake_hub.hf_hub_download = _raise

        with patch.dict(sys.modules, {"huggingface_hub": fake_hub}), pytest.raises(FileNotFoundError):
            _resolve_local_model_path("unknown-org/unknown-model")

    def test_empty_model_id_rejected(self):
        with pytest.raises(ValueError):
            _resolve_local_model_path("")
