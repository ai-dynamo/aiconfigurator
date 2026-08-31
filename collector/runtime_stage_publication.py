# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retry-safe cleanup for marker-last runtime staging."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


class RuntimeStagePublicationError(RuntimeError):
    """A committed runtime is incomplete or a cleanup target is unsafe."""


def prepare_runtime_retry(marker: Path, *, files: tuple[Path, ...], trees: tuple[Path, ...]) -> bool:
    """Validate a committed set, or remove an uncommitted partial generation.

    Returns ``False`` when the marker commits an intact set and ``True`` when
    an uncommitted set was cleaned for retry.
    """
    targets = (*files, *trees)
    if len(targets) != len(set(targets)) or marker in targets:
        raise RuntimeStagePublicationError("runtime staging targets must be unique and exclude the marker")
    for target in targets:
        if target == Path(target.anchor) or target.is_symlink():
            raise RuntimeStagePublicationError(f"unsafe runtime staging target: {target}")
    if marker.is_file():
        missing = [str(path) for path in targets if not path.exists()]
        if missing:
            raise RuntimeStagePublicationError(f"committed runtime is missing artifacts: {missing}")
        return False
    for path in files:
        if path.exists():
            if not path.is_file():
                raise RuntimeStagePublicationError(f"runtime file target is not a file: {path}")
            path.unlink()
    for path in trees:
        if path.exists():
            if not path.is_dir():
                raise RuntimeStagePublicationError(f"runtime tree target is not a directory: {path}")
            shutil.rmtree(path)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("marker", type=Path)
    parser.add_argument("--file", action="append", default=[], type=Path)
    parser.add_argument("--tree", action="append", default=[], type=Path)
    args = parser.parse_args()
    prepare_runtime_retry(args.marker, files=tuple(args.file), trees=tuple(args.tree))


if __name__ == "__main__":
    main()
