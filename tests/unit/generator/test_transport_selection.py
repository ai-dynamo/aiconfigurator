import copy

import yaml

from aiconfigurator.generator.api import generate_backend_artifacts
from aiconfigurator.generator.facts.request_resolution import resolve_facts_for_request


def _params(transport=None):
    p = {"ServiceConfig": {"model_path": "Qwen/Qwen3-0.6B"},
         "K8sConfig": {"system_name": "gb200"}, "NodeConfig": {}}
    if transport is not None:
        p["K8sConfig"]["transport"] = transport
    return p


def test_default_transport_is_nvlink():
    f = resolve_facts_for_request(_params(), "vllm", "1.2.0")
    assert f.transport.get("env", {}) == {}


def test_ib_transport_selected():
    f = resolve_facts_for_request(_params("ib"), "vllm", "1.2.0")
    assert any(k.startswith("UCX_") for k in f.transport.get("env", {}))


def test_efa_transport_selected():
    f = resolve_facts_for_request(_params("efa"), "vllm", "1.2.0")
    assert any(k.startswith("FI_") for k in f.transport.get("env", {}))
    assert f.transport.get("pod", {}).get("privileged") is True


def test_unknown_transport_degrades_gracefully():
    # An unknown transport must not crash generation; resolution degrades to facts=None.
    import copy
    from aiconfigurator.generator.api import generate_backend_artifacts
    from aiconfigurator.generator.naive import build_naive_generator_params
    p = build_naive_generator_params(model_name="Qwen/Qwen3-0.6B", total_gpus=8,
        system_name="h200_sxm", backend_name="vllm", mode="agg",
        generator_overrides={"K8sConfig": {"k8s_namespace": "dynamo", "transport": "bogus"}})
    arts = generate_backend_artifacts(copy.deepcopy(p), "vllm", backend_version="0.20.1")
    assert "k8s_deploy.yaml" in arts  # generation succeeded despite bad transport


def test_ib_transport_env_reaches_worker_pod():
    """Live end-to-end: an ib request lands UCX_* env on the worker container."""
    from tests.baseline.canary import CANARY_CASES

    case = next(c for c in CANARY_CASES if c.name == "qwen_moe_vllm_gb200_ib")
    arts = generate_backend_artifacts(
        copy.deepcopy(case.params), case.backend, backend_version=case.backend_version
    )
    doc = [
        d
        for d in yaml.safe_load_all(arts["k8s_deploy.yaml"])
        if d and d.get("kind") == "DynamoGraphDeployment"
    ][0]
    worker = next(
        v for k, v in doc["spec"]["services"].items() if "frontend" not in k.lower()
    )
    env = {e["name"]: e["value"] for e in worker["extraPodSpec"]["mainContainer"].get("env", [])}
    assert any(k.startswith("UCX_") for k in env)
