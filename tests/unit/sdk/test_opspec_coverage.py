# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structural tripwire: every ``Operation`` must reach the compiled engine.

The Rust engine executes whatever ``engine.py::_to_opspec`` emits. An op class
added on the Python side WITHOUT a conversion branch silently forces the whole
model back onto the (frozen) Python step — exactly the drift the Python-path
freeze forbids. This test makes that state un-mergeable: a new ``Operation``
subclass must either get a ``_to_opspec`` branch (plus its Rust mirror and a
parity case — see ``.claude/rules/rust-core/parity.md``) or an explicit,
justified entry in ``EXEMPT`` below.

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
    # engine-step path is never involved). Retirement prerequisite: the thin
    # op-list evaluation FFI (see the Python-path freeze tracking issue).
    "AFDTransfer": "AFD orchestration is Python-side; op-list FFI planned",
    "AFDCombine": "AFD orchestration is Python-side; op-list FFI planned",
    "AFDFAllGather": "AFD orchestration is Python-side; op-list FFI planned",
    "AFDFReduceScatter": "AFD orchestration is Python-side; op-list FFI planned",
    # Dead class: no model instantiates it (Mamba2Kernel is the live op and
    # converts). Remove the class or this entry together.
    "Mamba2": "dead code — never instantiated; Mamba2Kernel is the live op",
}


def _operation_subclasses() -> set[str]:
    # Importing the package registers every op subclass.
    import aiconfigurator.sdk.operations  # noqa: F401
    from aiconfigurator.sdk.operations.base import Operation

    seen: set[type] = set()
    stack: list[type] = [Operation]
    while stack:
        for sub in stack.pop().__subclasses__():
            if sub not in seen:
                seen.add(sub)
                stack.append(sub)
    # Private bases (e.g. _BaseMSAModule) are implementation details; their
    # public leaves are what models instantiate.
    return {cls.__name__ for cls in seen if not cls.__name__.startswith("_")}


def test_every_operation_converts_to_an_opspec_or_is_exempt():
    from aiconfigurator.sdk import engine

    handled = set(re.findall(r"isinstance\(op, (\w+)\)", inspect.getsource(engine._to_opspec)))
    all_ops = _operation_subclasses()

    missing = sorted(all_ops - handled - set(EXEMPT))
    assert not missing, (
        f"Operation classes without a _to_opspec branch: {missing}. "
        "Port them to the compiled engine (opspec branch + Rust mirror + "
        "parity case, see .claude/rules/rust-core/parity.md) or add an "
        "explicit EXEMPT entry with a reason."
    )

    stale = sorted(set(EXEMPT) & handled)
    assert not stale, f"EXEMPT entries now have conversion branches; remove them: {stale}"

    # EXEMPT is a closed set: every entry must name a live Operation class
    # (deleted/renamed ops must drop their entry) and carry a real reason.
    unknown = sorted(set(EXEMPT) - all_ops)
    assert not unknown, f"EXEMPT entries are not discovered Operation classes; remove them: {unknown}"

    blank = sorted(name for name, reason in EXEMPT.items() if not reason.strip())
    assert not blank, f"EXEMPT entries need a non-empty justification: {blank}"
