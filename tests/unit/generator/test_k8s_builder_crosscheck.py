# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import pytest
import yaml

from aiconfigurator.generator.api import generate_backend_artifacts
from aiconfigurator.generator.builders.dgd_model import dgd_documents_to_yaml
from aiconfigurator.generator.builders.k8s_builder import build_dgd
from aiconfigurator.generator.facts.request_resolution import resolve_facts_for_request
from aiconfigurator.generator.rendering.engine import build_k8s_context_for_test
from tests.baseline.canary import CANARY_CASES


@pytest.mark.parametrize("case", CANARY_CASES, ids=lambda c: c.name)
def test_k8s_builder_context_seam_matches_render_path(case):
    """The test-seam context builder produces the same k8s output as production.

    Before the cutover this asserted builder == Jinja. Post-cutover the k8s
    templates are retired and ``generate_backend_artifacts`` IS the builder, so
    this now guards that ``build_k8s_context_for_test`` stays faithful to the
    real ``render_backend_templates`` context. The durable builder-vs-Jinja
    parity guarantee lives in ``tests/baseline/test_cutover_semantics.py``
    (compared against the immutable pre-cutover Jinja reference).
    """
    render_path = generate_backend_artifacts(
        copy.deepcopy(case.params), case.backend, backend_version=case.backend_version
    )["k8s_deploy.yaml"]
    ctx = build_k8s_context_for_test(copy.deepcopy(case.params), case.backend, backend_version=case.backend_version)
    # The render path emits the hardware/transport pod facts from the
    # resolved facts; thread the same facts here so the test-seam build mirrors
    # production (otherwise the seam would omit nodeSelector/tolerations/env/shm).
    facts = resolve_facts_for_request(copy.deepcopy(case.params), case.backend, case.backend_version)
    built = dgd_documents_to_yaml(build_dgd(ctx, case.backend, resolved_facts=facts))
    assert list(yaml.safe_load_all(built)) == list(yaml.safe_load_all(render_path)), (
        f"{case.name}: test-seam k8s context diverges from the render path"
    )
