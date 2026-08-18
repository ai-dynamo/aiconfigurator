# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structural tripwire: every ``Operation`` must reach the compiled engine.

Since the pyo3 op unification the op classes ARE engine ops (Rust pyclasses
behind thin data-plane shells), so "reaches the engine" is structural: the
class subclasses the Rust ``Operation`` base (its instances self-serialize
via ``_spec_json``), or it converts through an explicit adapter branch in
``engine.py::_as_engine_op`` (``FPMForwardOp``). A Python-side op class with
neither silently crashes spec assembly at estimation time — exactly the
drift this test makes un-mergeable: a new op must either extend a Rust
family class (plus its Rust mirror and a parity case — see
``.claude/rules/rust-core/parity.md``) or carry an explicit, justified
entry in ``EXEMPT`` below.

The Rust side has the symmetric guard: ``engine/spec.rs::all_op_variants``
fails to compile when an ``Op`` variant is added without wiring.
"""

from __future__ import annotations

import inspect
import re

import pytest

pytestmark = pytest.mark.unit

# Op classes deliberately NOT convertible to the compiled engine. Every entry
# needs a reason; removing the op or porting it must also remove the entry
# (the staleness assertion below enforces that).
EXEMPT: dict[str, str] = {
    # AFD (attention-FFN disagg) is session-level Python orchestration by
    # design (`inference_session.py` builds and sums these directly; the
    # engine-step path is never involved, and each op's math composes
    # ENGINE-evaluated twin ops through the single-op plumbing). A future
    # Rust port is sketched in the afd_transfer.py module TODO.
    "AFDTransfer": "AFD orchestration is Python-side; composes engine-evaluated twins",
    "AFDCombine": "AFD orchestration is Python-side; composes engine-evaluated twins",
    "AFDFAllGather": "AFD orchestration is Python-side; composes engine-evaluated twins",
    "AFDFReduceScatter": "AFD orchestration is Python-side; composes engine-evaluated twins",
}

# Python-side classes with an explicit adapter branch in
# ``engine.py::_as_engine_op`` instead of a Rust base (the whole-model FPM
# wrapper converts via ``op_from_spec_json``). Pinned against the source so
# the set cannot silently drift.
ADAPTED: frozenset[str] = frozenset({"FPMForwardOp"})


def _operation_classes() -> dict[str, type]:
    """Every public op class reachable from the two roots.

    ``Operation`` (the Rust base) covers the engine-backed families and their
    shells; ``PythonOperation`` covers the Python-side orchestration ops.
    """
    # Importing the package registers every op subclass.
    import aiconfigurator.sdk.operations  # noqa: F401
    from aiconfigurator.sdk.operations.base import Operation, PythonOperation

    seen: set[type] = set()
    stack: list[type] = [Operation, PythonOperation]
    while stack:
        for sub in stack.pop().__subclasses__():
            if sub not in seen:
                seen.add(sub)
                stack.append(sub)
    # Private bases and test-local subclasses are implementation details;
    # the shipped classes live under the operations (or Rust core) packages.
    return {
        cls.__name__: cls
        for cls in seen
        if not cls.__name__.startswith("_")
        and (cls.__module__.startswith("aiconfigurator") or cls.__module__.endswith("_aiconfigurator_core"))
    }


def test_every_operation_reaches_the_engine_or_is_exempt():
    from aiconfigurator.sdk import engine
    from aiconfigurator.sdk.operations.base import Operation

    classes = _operation_classes()
    engine_backed = {name for name, cls in classes.items() if issubclass(cls, Operation)}

    missing = sorted(set(classes) - engine_backed - ADAPTED - set(EXEMPT))
    assert not missing, (
        f"Operation classes that cannot reach the compiled engine: {missing}. "
        "Extend a Rust family class (Rust mirror + parity case, see "
        ".claude/rules/rust-core/parity.md), add an _as_engine_op adapter "
        "branch, or add an explicit EXEMPT entry with a reason."
    )

    # The adapter set is pinned against the _as_engine_op source: a removed
    # branch (or a renamed class) must update ADAPTED in the same change.
    adapter_source = inspect.getsource(engine._as_engine_op)
    for name in sorted(ADAPTED):
        assert re.search(rf"isinstance\(op, {name}\)", adapter_source), (
            f"ADAPTED entry {name!r} has no isinstance branch in engine._as_engine_op"
        )
        assert name in classes, f"ADAPTED entry {name!r} is not a discovered op class"
        assert name not in engine_backed, f"ADAPTED entry {name!r} is already engine-backed; remove it"

    stale = sorted(set(EXEMPT) & engine_backed)
    assert not stale, f"EXEMPT entries are now engine-backed; remove them: {stale}"

    # EXEMPT is a closed set: every entry must name a live Operation class
    # (deleted/renamed ops must drop their entry) and carry a real reason.
    unknown = sorted(set(EXEMPT) - set(classes))
    assert not unknown, f"EXEMPT entries are not discovered Operation classes; remove them: {unknown}"

    blank = sorted(name for name, reason in EXEMPT.items() if not reason.strip())
    assert not blank, f"EXEMPT entries need a non-empty justification: {blank}"
