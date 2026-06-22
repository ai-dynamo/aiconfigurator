"""Characterization test pinning kv_cache_dtype value translation.

These value translations are CRASH/SILENT-severity guard rails (see
.claude/rules/generator/guard_rails.md). They are being moved from the .rule
DSL (derive stage) into backend_config_mapping.yaml (mapping stage). The emitted
backend flag value MUST be byte-identical before and after the migration.

The EXPECTED dict below is filled from the REAL pre-migration emission (Step 2).
If post-migration output differs, the mapping expression is wrong -- fix the
EXPRESSION, never the EXPECTED.
"""

import copy

import pytest

from aiconfigurator.generator.api import generate_backend_artifacts
from tests.baseline.canary import CANARY_CASES


def _case(backend):
    return next(c for c in CANARY_CASES if c.backend == backend and "agg" in c.name)


def _set_kv_dtype(params, value):
    params = copy.deepcopy(params)
    roles = params.get("params", {})
    for role in ("agg", "prefill", "decode"):
        if isinstance(roles.get(role), dict):
            roles[role]["kv_cache_dtype"] = value
    return params


CASES = [
    ("vllm", "bfloat16"),
    ("sglang", "bfloat16"),
    ("sglang", "fp8"),
    ("trtllm", "bfloat16"),
    ("trtllm", "float16"),
]

# Filled from the pre-migration run (Step 2). Maps (backend, input) -> the exact
# substring that MUST appear in the generated artifacts proving the TRANSLATED
# kv-dtype value is emitted.
#   vLLM:   bfloat16 -> auto  (cli flag --kv-cache-dtype "auto")
#   SGLang: bfloat16 -> auto, fp8 -> fp8_e4m3  (cli flag --kv-cache-dtype "...")
#   TRT-LLM: float16/bfloat16 -> auto  (engine yaml kv_cache_config dtype: auto)
EXPECTED = {
    ("vllm", "bfloat16"): '--kv-cache-dtype "auto"',
    ("sglang", "bfloat16"): '--kv-cache-dtype "auto"',
    ("sglang", "fp8"): '--kv-cache-dtype "fp8_e4m3"',
    ("trtllm", "bfloat16"): "  dtype: auto",
    ("trtllm", "float16"): "  dtype: auto",
}


@pytest.mark.parametrize("backend,inp", CASES)
def test_kv_dtype_emission_is_stable(backend, inp):
    case = _case(backend)
    params = _set_kv_dtype(case.params, inp)
    arts = generate_backend_artifacts(params, backend, backend_version=case.backend_version)
    blob = "\n".join(arts.values())
    assert EXPECTED[(backend, inp)] in blob, f"{backend} kv={inp}: missing {EXPECTED[(backend, inp)]!r}"
