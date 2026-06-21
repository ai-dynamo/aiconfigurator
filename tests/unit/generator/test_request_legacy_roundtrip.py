# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load-bearing gate: params -> GeneratorRequest -> params renders byte-identically.

For every canary case, lowering a request reconstructed from the legacy params
must produce artifacts byte-identical to generating from the original params.
"""
from __future__ import annotations

import copy

import pytest

from aiconfigurator.generator.api import generate_backend_artifacts
from aiconfigurator.generator.request import from_legacy_params, to_legacy_params
from tests.baseline.canary import CANARY_CASES


@pytest.mark.parametrize("case", CANARY_CASES, ids=lambda c: c.name)
def test_request_roundtrip_renders_byte_identical(case):
    req = from_legacy_params(copy.deepcopy(case.params), backend=case.backend)
    assert req.validate() == [], f"{case.name}: {req.validate()}"

    lowered = to_legacy_params(req)
    arts_new = generate_backend_artifacts(lowered, case.backend, backend_version=case.backend_version)
    arts_old = generate_backend_artifacts(copy.deepcopy(case.params), case.backend, backend_version=case.backend_version)

    assert set(arts_new) == set(arts_old), f"{case.name}: artifact set differs"
    for name in sorted(arts_old):
        assert arts_new[name] == arts_old[name], f"{case.name}/{name} differs"
