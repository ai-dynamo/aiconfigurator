import copy

import pytest
import yaml

from aiconfigurator.generator.api import generate_backend_artifacts
from aiconfigurator.generator.builders.dgd_model import dgd_documents_to_yaml
from aiconfigurator.generator.builders.k8s_builder import build_dgd
from aiconfigurator.generator.rendering.engine import build_k8s_context_for_test
from tests.baseline.canary import CANARY_CASES


@pytest.mark.parametrize("case", CANARY_CASES, ids=lambda c: c.name)
def test_builder_k8s_semantically_equals_jinja(case):
    jinja = generate_backend_artifacts(
        copy.deepcopy(case.params), case.backend, backend_version=case.backend_version
    )["k8s_deploy.yaml"]
    ctx = build_k8s_context_for_test(
        copy.deepcopy(case.params), case.backend, backend_version=case.backend_version
    )
    built = dgd_documents_to_yaml(build_dgd(ctx, case.backend))
    assert list(yaml.safe_load_all(built)) == list(yaml.safe_load_all(jinja)), (
        f"{case.name}: typed builder k8s differs semantically from Jinja"
    )
