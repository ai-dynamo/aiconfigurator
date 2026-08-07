# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``collector.helper._resolve_local_model_path``."""

import json
import os
import sys
import threading
from unittest.mock import patch

import pytest

# Ensure ``collector/`` is importable since collector ships as a top-level package
# of loose scripts (not installed via pyproject).
_COLLECTOR_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "collector")
sys.path.insert(0, os.path.abspath(_COLLECTOR_DIR))

from helper import _resolve_local_model_path, config_norm_cache_key

pytestmark = pytest.mark.unit


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


class TestBundledConfigRefresh:
    """Regressions for the #1487 review: a bundled-config update under an
    unchanged slug must reach consumers on a reused host (the materialized
    copy used to be write-once, so content-keyed caches downstream hashed
    stale bytes), and config.json + hf_quant_config.json must publish as ONE
    immutable snapshot so a concurrent reader can never copytree a torn pair
    (new config with an old or removed side-car)."""

    def _bundle(self, tmp_path, monkeypatch, slug, config, quant=None):
        cache_dir = tmp_path / "model_configs"
        cache_dir.mkdir(exist_ok=True)
        (cache_dir / f"{slug}_config.json").write_text(json.dumps(config))
        quant_path = cache_dir / f"{slug}_hf_quant_config.json"
        if quant is not None:
            quant_path.write_text(json.dumps(quant))
        elif quant_path.exists():
            quant_path.unlink()
        monkeypatch.setattr("helper._AIC_MODEL_CONFIG_DIR", str(cache_dir))

    def test_materialized_config_refreshes_when_bundled_source_changes(self, isolated_tmp, tmp_path, monkeypatch):
        self._bundle(
            tmp_path,
            monkeypatch,
            "fake-org--refresh",
            {"model_type": "fake", "rev": 1, "auto_map": {"AutoConfig": "configuration_fake.FakeConfig"}},
        )
        first = _resolve_local_model_path("fake-org/refresh")
        with open(os.path.join(first, "config.json")) as f:
            assert json.load(f)["rev"] == 1

        self._bundle(
            tmp_path,
            monkeypatch,
            "fake-org--refresh",
            {"model_type": "fake", "rev": 2, "auto_map": {"AutoConfig": "configuration_fake.FakeConfig"}},
        )
        second = _resolve_local_model_path("fake-org/refresh")
        assert second != first  # new content publishes a NEW immutable snapshot
        with open(os.path.join(second, "config.json")) as f:
            materialized = json.load(f)
        assert materialized["rev"] == 2  # the update reaches new resolvers
        assert "auto_map" not in materialized  # the strip survives the refresh
        with open(os.path.join(first, "config.json")) as f:
            assert json.load(f)["rev"] == 1  # published snapshots are immutable

    def test_quant_side_car_updates_and_removal_publish_fresh_snapshots(self, isolated_tmp, tmp_path, monkeypatch):
        slug, model_id = "fake-org--quant-refresh", "fake-org/quant-refresh"
        self._bundle(tmp_path, monkeypatch, slug, {"model_type": "fake"}, quant={"quant": "fp8", "rev": 1})
        p1 = _resolve_local_model_path(model_id)
        with open(os.path.join(p1, "hf_quant_config.json")) as f:
            assert json.load(f)["rev"] == 1

        # A side-car-only update publishes a new snapshot; the pair travels
        # together, so no reader can see new config with an old side-car.
        self._bundle(tmp_path, monkeypatch, slug, {"model_type": "fake"}, quant={"quant": "fp8", "rev": 2})
        p2 = _resolve_local_model_path(model_id)
        assert p2 != p1
        with open(os.path.join(p2, "hf_quant_config.json")) as f:
            assert json.load(f)["rev"] == 2

        # A side-car removed from the bundle must not linger for NEW
        # resolvers: a stale copy would keep feeding quant config to
        # ModelConfig.from_pretrained. In-flight readers of p1/p2 keep
        # their consistent (immutable) snapshots.
        self._bundle(tmp_path, monkeypatch, slug, {"model_type": "fake"}, quant=None)
        p3 = _resolve_local_model_path(model_id)
        assert p3 not in (p1, p2)
        assert not os.path.exists(os.path.join(p3, "hf_quant_config.json"))
        assert os.path.exists(os.path.join(p2, "hf_quant_config.json"))  # immutability

    def test_norm_cache_key_tracks_bundled_updates_end_to_end(self, isolated_tmp, tmp_path, monkeypatch):
        # The GLM DSA normalized-config cache keys on config_norm_cache_key;
        # the P1 failure mode was a bundled update that never changed the key
        # because the hash read write-once stale materialized bytes.
        slug, model_id = "fake-org--keyed", "fake-org/keyed"
        self._bundle(tmp_path, monkeypatch, slug, {"model_type": "fake", "rev": 1})
        path = _resolve_local_model_path(model_id)
        key_rev1 = config_norm_cache_key(path)
        assert key_rev1 == config_norm_cache_key(path)  # stable when nothing changes

        self._bundle(tmp_path, monkeypatch, slug, {"model_type": "fake", "rev": 2})
        key_rev2 = config_norm_cache_key(_resolve_local_model_path(model_id))
        assert key_rev2 != key_rev1  # the bundled update reaches the cache key

        # A side-car-only change must also move the key: the normalized copy
        # materializes side-cars and ModelConfig.from_pretrained reads them.
        self._bundle(tmp_path, monkeypatch, slug, {"model_type": "fake", "rev": 2}, quant={"quant": "fp8"})
        key_rev3 = config_norm_cache_key(_resolve_local_model_path(model_id))
        assert key_rev3 not in (key_rev1, key_rev2)

    def test_norm_cache_key_requires_config_json(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            config_norm_cache_key(str(tmp_path))

    def test_concurrent_materialization_from_threads_is_safe(self, isolated_tmp, tmp_path, monkeypatch):
        # Staging dirs must be unique per invocation, not per pid: threads
        # share a pid, and a shared staging path would let one thread rename
        # the directory out from under another mid-write.
        self._bundle(tmp_path, monkeypatch, "fake-org--threads", {"model_type": "fake", "rev": 1})
        results, errors = [], []

        def resolve():
            try:
                results.append(_resolve_local_model_path("fake-org/threads"))
            except Exception as e:  # pragma: no cover - the assertion below reports it
                errors.append(e)

        threads = [threading.Thread(target=resolve) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        assert len(set(results)) == 1  # all threads converge on one snapshot
        with open(os.path.join(results[0], "config.json")) as f:
            assert json.load(f)["rev"] == 1
