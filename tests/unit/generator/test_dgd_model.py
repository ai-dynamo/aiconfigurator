import yaml

from aiconfigurator.generator.builders.dgd_model import (
    DGD,
    DGDService,
    ExtraPodSpec,
    MainContainer,
    dgd_documents_to_yaml,
)


def test_new_pod_fields_emit_when_set():
    svc = DGDService(
        replicas=1,
        resources={"limits": {"gpu": "8"}},
        shared_memory={"size": "64Gi"},
        extra_pod_spec=ExtraPodSpec(
            node_selector={"nvidia.com/gpu.product": "NVIDIA-H200"},
            tolerations=[{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}],
            main_container=MainContainer(image="img", env=[{"name": "NCCL_CUMEM_ENABLE", "value": "1"}]),
        ),
    )
    doc = yaml.safe_load(dgd_documents_to_yaml([DGD(name="d", services={"worker": svc})]))
    s = doc["spec"]["services"]["worker"]
    assert s["sharedMemory"] == {"size": "64Gi"}
    assert s["extraPodSpec"]["nodeSelector"] == {"nvidia.com/gpu.product": "NVIDIA-H200"}
    assert s["extraPodSpec"]["tolerations"][0]["key"] == "nvidia.com/gpu"
    assert s["extraPodSpec"]["mainContainer"]["env"][0]["name"] == "NCCL_CUMEM_ENABLE"


def test_new_pod_fields_absent_when_unset():
    svc = DGDService(
        replicas=1,
        resources={"limits": {"gpu": "8"}},
        extra_pod_spec=ExtraPodSpec(main_container=MainContainer(image="img")),
    )
    s = yaml.safe_load(dgd_documents_to_yaml([DGD(name="d", services={"w": svc})]))["spec"]["services"]["w"]
    assert "sharedMemory" not in s
    assert "nodeSelector" not in s["extraPodSpec"]
    assert "tolerations" not in s["extraPodSpec"]
    assert "env" not in s["extraPodSpec"]["mainContainer"]


def test_efa_pod_fields_emit_when_set():
    import yaml

    from aiconfigurator.generator.builders.dgd_model import (
        DGD,
        DGDService,
        ExtraPodSpec,
        MainContainer,
        dgd_documents_to_yaml,
    )

    svc = DGDService(
        replicas=1,
        resources={"limits": {"gpu": "4", "custom": {"vpc.amazonaws.com/efa": "4"}}},
        extra_pod_spec=ExtraPodSpec(
            host_ipc=True,
            main_container=MainContainer(
                image="i", security_context={"privileged": True, "capabilities": {"add": ["IPC_LOCK"]}}
            ),
        ),
    )
    s = yaml.safe_load(dgd_documents_to_yaml([DGD(name="d", services={"w": svc})]))["spec"]["services"]["w"]
    assert s["extraPodSpec"]["hostIPC"] is True
    assert s["extraPodSpec"]["mainContainer"]["securityContext"]["privileged"] is True
    assert s["resources"]["limits"]["custom"]["vpc.amazonaws.com/efa"] == "4"


def test_efa_pod_fields_absent_when_unset():
    import yaml

    from aiconfigurator.generator.builders.dgd_model import (
        DGD,
        DGDService,
        ExtraPodSpec,
        MainContainer,
        dgd_documents_to_yaml,
    )

    svc = DGDService(replicas=1, extra_pod_spec=ExtraPodSpec(main_container=MainContainer(image="i")))
    s = yaml.safe_load(dgd_documents_to_yaml([DGD(name="d", services={"w": svc})]))["spec"]["services"]["w"]
    assert "hostIPC" not in s["extraPodSpec"]
    assert "securityContext" not in s["extraPodSpec"]["mainContainer"]


def test_multinode_and_resource_claims_emit():
    import yaml

    from aiconfigurator.generator.builders.dgd_model import (
        DGD,
        DGDService,
        ExtraPodSpec,
        MainContainer,
        dgd_documents_to_yaml,
    )

    svc = DGDService(
        replicas=1,
        multinode={"nodeCount": 2},
        resources={"limits": {"gpu": "8"}, "claims": [{"name": "compute-domain-channel"}]},
        extra_pod_spec=ExtraPodSpec(
            resource_claims=[
                {"name": "compute-domain-channel", "resourceClaimTemplateName": "d-compute-domain-channel"}
            ],
            main_container=MainContainer(image="i"),
        ),
    )
    s = yaml.safe_load(dgd_documents_to_yaml([DGD(name="d", services={"w": svc})]))["spec"]["services"]["w"]
    assert s["multinode"] == {"nodeCount": 2}
    assert s["resources"]["claims"][0]["name"] == "compute-domain-channel"
    assert s["extraPodSpec"]["resourceClaims"][0]["resourceClaimTemplateName"] == "d-compute-domain-channel"


def test_compute_domain_doc():
    import yaml

    from aiconfigurator.generator.builders.dgd_model import ComputeDomainDoc, dgd_documents_to_yaml

    doc = yaml.safe_load(
        dgd_documents_to_yaml(
            [ComputeDomainDoc(name="d-compute-domain", channel_name="d-compute-domain-channel", num_nodes=0)]
        )
    )
    assert doc["apiVersion"] == "resource.nvidia.com/v1beta1"
    assert doc["kind"] == "ComputeDomain"
    assert doc["spec"]["channel"]["resourceClaimTemplate"]["name"] == "d-compute-domain-channel"
    # numNodes=0 is the intentional DRA on-demand value — the driver sizes the
    # domain as pods schedule. Do NOT "fix" this to a non-zero value.
    assert doc["spec"]["numNodes"] == 0


def test_compute_domain_doc_round_trips():
    import yaml

    from aiconfigurator.generator.builders.dgd_model import ComputeDomainDoc, dgd_documents_to_yaml

    doc = ComputeDomainDoc(name="d-compute-domain", channel_name="d-compute-domain-channel", num_nodes=0)
    y1 = dgd_documents_to_yaml([doc])
    # round-trip via from_dict if ComputeDomainDoc has one; else parse-stability
    parsed = yaml.safe_load(y1)
    assert parsed["kind"] == "ComputeDomain"
    # numNodes=0 is the intentional DRA on-demand value (not a missing default).
    assert parsed["spec"]["numNodes"] == 0
    assert parsed["spec"]["channel"]["resourceClaimTemplate"]["name"] == "d-compute-domain-channel"
    # if ComputeDomainDoc.from_dict exists, assert true round-trip:
    if hasattr(ComputeDomainDoc, "from_dict"):
        assert dgd_documents_to_yaml([ComputeDomainDoc.from_dict(parsed)]) == y1


def test_round_trip_via_from_dict():
    svc = DGDService(
        replicas=1,
        shared_memory={"size": "80Gi"},
        extra_pod_spec=ExtraPodSpec(node_selector={"a": "b"}, main_container=MainContainer(image="i")),
    )
    y1 = dgd_documents_to_yaml([DGD(name="d", services={"w": svc})])
    d2 = DGD.from_dict(yaml.safe_load(y1))
    assert dgd_documents_to_yaml([d2]) == y1
