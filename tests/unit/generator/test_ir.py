# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiconfigurator.generator.ir import Component, DeploymentIR


def _agg_ir() -> DeploymentIR:
    ir = DeploymentIR(backend="vllm", backend_version="0.20.1", mode="agg")
    ir.add_component(Component(name="Frontend", role="frontend"))
    ir.add_component(Component(name="VllmWorker", role="worker", replicas=1))
    ir.add_edge("Frontend", "VllmWorker")
    return ir


def test_ir_models_components_and_edges():
    ir = _agg_ir()
    assert [c.name for c in ir.components] == ["Frontend", "VllmWorker"]
    assert ("Frontend", "VllmWorker") in ir.edges


def test_provenance_and_warnings_are_recorded():
    ir = _agg_ir()
    ir.add_provenance("VllmWorker.tp", 8, source="sdk", stage="facts")
    ir.warnings.append("example")
    assert ir.provenance["VllmWorker.tp"]["value"] == 8
    assert ir.provenance["VllmWorker.tp"]["source"] == "sdk"
    assert "example" in ir.warnings


def test_adding_epd_nodes_is_append_only():
    # Acceptance checkpoint: extending to EPD must be "append a component
    # + an edge", never an IR field-definition change.
    ir = _agg_ir()
    ir.add_component(Component(name="Processor", role="processor"))
    ir.add_component(Component(name="EncodeWorker", role="encode"))
    ir.add_edge("Frontend", "Processor")
    ir.add_edge("Processor", "EncodeWorker")
    ir.add_edge("EncodeWorker", "VllmWorker")
    roles = {c.role for c in ir.components}
    assert {"processor", "encode"} <= roles
    assert ("EncodeWorker", "VllmWorker") in ir.edges


def test_duplicate_component_name_rejected():
    ir = _agg_ir()
    import pytest

    with pytest.raises(ValueError):
        ir.add_component(Component(name="Frontend", role="frontend"))


def test_edge_to_unknown_component_rejected():
    ir = _agg_ir()
    import pytest

    with pytest.raises(ValueError):
        ir.add_edge("Frontend", "Nonexistent")
