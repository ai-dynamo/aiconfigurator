import pytest

from aiconfigurator.generator.facts.resolve import ResolvedFacts, resolve_facts


def test_resolve_returns_hardware_model_and_runtime_facts():
    facts = resolve_facts(
        model_profile_id="deepseek-v4",
        hardware="gb200",
        transport="nvlink",
        dynamo_version="1.2.0",
        backend="trtllm",
    )
    assert isinstance(facts, ResolvedFacts)
    assert facts.backend_version == "1.3.0rc14"  # runtimes: dynamo 1.2.0 -> trtllm
    assert facts.hardware["moe_backend"]["trtllm"]  # hardware gb200 selection non-empty
    assert "moe" in facts.model["traits"]  # models deepseek-v4
    assert facts.transport is not None  # transport profile resolved


def test_unknown_model_resolves_to_none_not_error():
    facts = resolve_facts(
        model_profile_id=None,
        hardware="h200",
        transport="nvlink",
        dynamo_version="1.2.0",
        backend="vllm",
    )
    assert facts.model is None  # generic path: no profile
    assert facts.hardware["node_selector"]  # hardware still resolved
    assert facts.backend_version  # vllm version resolved


def test_unknown_hardware_raises_clear_error():
    with pytest.raises(KeyError):
        resolve_facts(
            model_profile_id=None, hardware="nonexistent", transport="nvlink", dynamo_version="1.2.0", backend="vllm"
        )
