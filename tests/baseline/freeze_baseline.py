"""Freeze current main-generator output for the canary matrix.

Run from the repo root: python -m tests.baseline.freeze_baseline
Writes tests/baseline/frozen/<case>/<artifact_name> for each canary case.
Re-running overwrites; the committed result IS the baseline.
"""
from __future__ import annotations

import copy
import pathlib

from aiconfigurator.generator.api import generate_backend_artifacts
from tests.baseline.canary import CANARY_CASES

FROZEN_DIR = pathlib.Path(__file__).parent / "frozen"


def generate_case(case) -> dict[str, str]:
    # generate_backend_artifacts mutates its input params in place (the rule
    # engine writes computed fields back; the max_batch_size rule is
    # non-idempotent). CANARY_CASES are shared module-level singletons, so we
    # deep-copy before generating to guarantee the freeze and the parity gate
    # always run on pristine params, independent of any other test that may
    # have touched the same case object earlier in the session.
    return generate_backend_artifacts(
        copy.deepcopy(case.params), case.backend, backend_version=case.backend_version
    )


def freeze() -> None:
    for case in CANARY_CASES:
        artifacts = generate_case(case)
        case_dir = FROZEN_DIR / case.name
        case_dir.mkdir(parents=True, exist_ok=True)
        for name, content in sorted(artifacts.items()):
            (case_dir / name).write_text(content, encoding="utf-8")


if __name__ == "__main__":
    freeze()
    print(f"Froze {len(CANARY_CASES)} cases into {FROZEN_DIR}")
