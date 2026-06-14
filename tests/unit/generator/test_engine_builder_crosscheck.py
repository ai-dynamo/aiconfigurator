import copy

import pytest
import yaml

from aiconfigurator.generator.api import generate_backend_artifacts
from aiconfigurator.generator.builders.trtllm_engine_config import build_engine_config
from aiconfigurator.generator.rendering.engine import build_engine_worker_contexts_for_test
from tests.baseline.canary import CANARY_CASES

_TRTLLM = [c for c in CANARY_CASES if c.backend == "trtllm"]


@pytest.mark.parametrize("case", _TRTLLM, ids=lambda c: c.name)
def test_engine_builder_semantically_equals_jinja(case):
    arts = generate_backend_artifacts(
        copy.deepcopy(case.params), case.backend, backend_version=case.backend_version
    )
    contexts = build_engine_worker_contexts_for_test(
        copy.deepcopy(case.params), case.backend, backend_version=case.backend_version
    )
    checked = 0
    for worker, wc in contexts.items():
        name = f"extra_engine_args_{worker}.yaml"
        if name not in arts:
            continue
        built = yaml.safe_dump(
            build_engine_config(wc, role=worker, version=case.backend_version), sort_keys=False
        )
        assert yaml.safe_load(built) == yaml.safe_load(arts[name]), f"{case.name}/{name}"
        checked += 1
    assert checked > 0, f"{case.name}: no engine-args artifacts were crosschecked"
