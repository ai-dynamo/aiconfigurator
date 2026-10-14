# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

from collector.runtime_stage_publication import RuntimeStagePublicationError, prepare_runtime_retry

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("published_count", range(3))
def test_vllm_interruption_is_cleaned_before_retry(tmp_path: Path, published_count: int):
    image = tmp_path / "runtime.sqsh"
    meta = tmp_path / "runtime.sqsh.meta.json"
    marker = tmp_path / "runtime.sqsh.SUCCESS"
    for path in (image, meta)[:published_count]:
        path.write_text("partial", encoding="utf-8")

    assert prepare_runtime_retry(marker, files=(image, meta), trees=()) is True
    assert not image.exists() and not meta.exists()


@pytest.mark.parametrize("published_count", range(4))
def test_trtllm_interruption_is_cleaned_before_retry(tmp_path: Path, published_count: int):
    image = tmp_path / "runtime.sqsh"
    meta = tmp_path / "runtime.sqsh.meta.json"
    wheel = tmp_path / "wheel"
    marker = wheel / "SUCCESS"
    steps = (image, meta, wheel)
    for path in steps[:published_count]:
        if path == wheel:
            path.mkdir()
        else:
            path.write_text("partial", encoding="utf-8")

    assert prepare_runtime_retry(marker, files=(image, meta), trees=(wheel,)) is True
    assert not image.exists() and not meta.exists() and not wheel.exists()


def test_complete_runtime_is_preserved_and_incomplete_commit_fails_closed(tmp_path: Path):
    image = tmp_path / "runtime.sqsh"
    meta = tmp_path / "runtime.sqsh.meta.json"
    wheel = tmp_path / "wheel"
    marker = wheel / "SUCCESS"
    image.write_text("image", encoding="utf-8")
    meta.write_text("meta", encoding="utf-8")
    wheel.mkdir()
    marker.touch()

    assert prepare_runtime_retry(marker, files=(image, meta), trees=(wheel,)) is False
    meta.unlink()
    with pytest.raises(RuntimeStagePublicationError, match="committed runtime is missing artifacts"):
        prepare_runtime_retry(marker, files=(image, meta), trees=(wheel,))


def test_cleanup_refuses_symlink_targets(tmp_path: Path):
    real = tmp_path / "real"
    real.write_text("keep", encoding="utf-8")
    link = tmp_path / "runtime.sqsh"
    link.symlink_to(real)
    with pytest.raises(RuntimeStagePublicationError, match="unsafe runtime staging target"):
        prepare_runtime_retry(tmp_path / "SUCCESS", files=(link,), trees=())
    assert real.read_text(encoding="utf-8") == "keep"
