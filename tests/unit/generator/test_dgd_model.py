import yaml
from aiconfigurator.generator.builders.dgd_model import (
    DGD, DGDService, ExtraPodSpec, MainContainer, dgd_documents_to_yaml,
)


def test_new_pod_fields_emit_when_set():
    svc = DGDService(
        replicas=1, resources={"limits": {"gpu": "8"}},
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
    svc = DGDService(replicas=1, resources={"limits": {"gpu": "8"}},
                     extra_pod_spec=ExtraPodSpec(main_container=MainContainer(image="img")))
    s = yaml.safe_load(dgd_documents_to_yaml([DGD(name="d", services={"w": svc})]))["spec"]["services"]["w"]
    assert "sharedMemory" not in s
    assert "nodeSelector" not in s["extraPodSpec"]
    assert "tolerations" not in s["extraPodSpec"]
    assert "env" not in s["extraPodSpec"]["mainContainer"]


def test_round_trip_via_from_dict():
    svc = DGDService(replicas=1, shared_memory={"size": "80Gi"},
                     extra_pod_spec=ExtraPodSpec(node_selector={"a": "b"}, main_container=MainContainer(image="i")))
    y1 = dgd_documents_to_yaml([DGD(name="d", services={"w": svc})])
    d2 = DGD.from_dict(yaml.safe_load(y1))
    assert dgd_documents_to_yaml([d2]) == y1
