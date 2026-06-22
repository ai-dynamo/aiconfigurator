# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Freeze current main-generator output for the canary matrix.

Run from the repo root: python -m tests.baseline.freeze_baseline
Writes <baseline-root>/frozen/<case>/<artifact_name> for each canary case, where
<baseline-root> is the out-of-repo location resolved in tests.baseline._paths
(``aiconfigurator-dev/refactor_baselines`` by default, or $AIC_BASELINE_DIR).
Re-running overwrites; the result IS the baseline (intentionally not committed).
"""

from __future__ import annotations

import copy

from aiconfigurator.generator.api import generate_backend_artifacts
from tests.baseline._paths import FROZEN_DIR
from tests.baseline.canary import CANARY_CASES


def generate_case(case) -> dict[str, str]:
    # generate_backend_artifacts mutates its input params in place (the rule
    # engine writes computed fields back; the max_batch_size rule is
    # non-idempotent). CANARY_CASES are shared module-level singletons, so we
    # deep-copy before generating to guarantee the freeze and the parity gate
    # always run on pristine params, independent of any other test that may
    # have touched the same case object earlier in the session.
    return generate_backend_artifacts(copy.deepcopy(case.params), case.backend, backend_version=case.backend_version)


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
