import copy

import pytest

from aiconfigurator.generator.api import generate_backend_artifacts
from aiconfigurator.generator.ir import DeploymentIR
from aiconfigurator.generator.pipeline import run_pipeline
from tests.baseline.canary import CANARY_CASES


@pytest.mark.parametrize("case", CANARY_CASES, ids=lambda c: c.name)
def test_pipeline_artifacts_match_legacy(case):
    # Deep-copy params before each call because render_backend_templates mutates
    # the input dict in place (rule-engine writes computed fields back). Without
    # isolated copies, the second call would see already-mutated values and
    # produce different output, making the comparison meaningless.
    legacy = generate_backend_artifacts(
        copy.deepcopy(case.params), case.backend, backend_version=case.backend_version
    )
    result = run_pipeline(
        copy.deepcopy(case.params), case.backend, backend_version=case.backend_version
    )
    assert result.artifacts == legacy


@pytest.mark.parametrize("case", CANARY_CASES, ids=lambda c: c.name)
def test_pipeline_builds_ir_with_components(case):
    result = run_pipeline(case.params, case.backend, backend_version=case.backend_version)
    assert isinstance(result.ir, DeploymentIR)
    assert result.ir.backend == case.backend
    assert any(c.role == "frontend" for c in result.ir.components)
    assert any(c.role in {"worker", "prefill", "decode"} for c in result.ir.components)
