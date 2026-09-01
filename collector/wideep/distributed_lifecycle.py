# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rank-wide stage agreement for standalone distributed collectors.

Every rank calls ``agree`` in the same order.  The helper deliberately does
not own a process group: torch and MPI collectors inject their native
all-rank reduction while CPU tests inject a deterministic coordinator.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

StageAgreement = Callable[[str, bool], bool]


class DistributedLifecycleError(RuntimeError):
    """A peer failed a named lifecycle stage."""


@dataclass(frozen=True)
class StageOutcome:
    """The all-rank result of one lifecycle stage."""

    stage: str
    error: BaseException | None

    @property
    def failed(self) -> bool:
        return self.error is not None


def agree_stage(
    stage: str,
    local_error: BaseException | None,
    *,
    agreement: StageAgreement,
    peer_error_type: type[BaseException] = DistributedLifecycleError,
) -> StageOutcome:
    """Return one identical success/failure decision on every rank.

    The originating rank retains its concrete exception.  Successful peers
    receive a named peer error so they cannot advance into another collective
    or tear down while the failing rank is still handling the prior stage.
    """

    any_failed = agreement(stage, local_error is not None)
    if not any_failed:
        return StageOutcome(stage, None)
    if local_error is not None:
        return StageOutcome(stage, local_error)
    return StageOutcome(stage, peer_error_type(f"another rank failed lifecycle stage {stage!r}"))


def raise_for_stage(outcome: StageOutcome) -> None:
    """Raise the local or peer exception represented by ``outcome``."""

    if outcome.error is not None:
        raise outcome.error
