# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vizier-backed sampler over a :class:`spica.search_space.BranchSpace`.

One Vizier study per branch (per the design). The study's parameters are:

- ``parallel_config_index`` — a categorical index into the branch's KV-feasible
  parallel-config list (configs are unordered, so categorical not discrete).
- one parameter per multi-choice searchable knob (categorical for string choices
  like ``planner_scaling_policy``/``router_mode``; discrete for the numeric
  batching / router-weight choices). Single-choice knobs are injected as
  constants (not Vizier params).

``suggest`` decodes each trial into a ``selection`` dict (the shape
:func:`spica.sample.unroll_sample` consumes) plus the chosen parallel-config
object; ``observe`` reports the (higher-is-better) score back to Vizier.

The sampler is swappable behind the :class:`BranchSampler` Protocol so a lighter
backend can replace Vizier without touching the orchestration. kv-router weight
knobs are always offered as params; ``unroll_sample`` ignores them under
``round_robin`` (no Vizier conditional params in v1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .parallel_enum import DisaggParallelConfig, ReplicaParallelConfig
from .search_space import BranchSpace

_PARALLEL_PARAM = "parallel_config_index"
_METRIC = "objective"


@dataclass
class Suggestion:
    """One sampled candidate: the unroll selection + chosen parallel config,
    plus an opaque handle the sampler uses to report the score."""

    selection: dict[str, Any]
    parallel_config: ReplicaParallelConfig | DisaggParallelConfig
    handle: Any = field(repr=False)


class BranchSampler(Protocol):
    """Stateful optimizer over one branch (swappable: Vizier, random, ...)."""

    branch: BranchSpace

    def suggest(self, count: int) -> list[Suggestion]: ...

    def observe(self, suggestion: Suggestion, score: float) -> None: ...

    def observe_infeasible(self, suggestion: Suggestion, reason: str) -> None: ...


def _decoder_for(choices: list[Any]) -> Callable[[Any], Any]:
    """How to turn a Vizier trial value back into the knob's native type."""
    if all(isinstance(c, str) for c in choices):
        return str  # categorical -> already a str
    if all(isinstance(c, int) and not isinstance(c, bool) for c in choices):
        return lambda v: int(round(float(v)))  # discrete int (Vizier stores float)
    return float  # discrete float


def _index_decoder(choices: list[Any]) -> Callable[[Any], Any]:
    """Decode a categorical *index* back to the chosen entry. Used when a knob's
    choices include dicts (a composite knob with pinned-dict entries) — dicts can't
    be Vizier categorical values, so we categorize over the index instead."""
    return lambda v: choices[int(round(float(v)))]


class VizierBranchSampler:
    """A Vizier GP-bandit study over one :class:`BranchSpace`."""

    def __init__(self, branch: BranchSpace, *, study_id: str):
        from vizier.service import clients
        from vizier.service import pyvizier as vz

        self.branch = branch
        self._decoders: dict[str, Callable[[Any], Any]] = {}
        self._constants: dict[str, Any] = {}

        problem = vz.ProblemStatement()
        root = problem.search_space.root
        root.add_categorical_param(_PARALLEL_PARAM, [str(i) for i in range(len(branch.parallel_configs))])
        for knob, choices in branch.knob_choices.items():
            if not any(isinstance(c, dict) for c in choices):
                # defensively dedupe hashable choices (order-preserving); duplicates
                # would otherwise crash Vizier study construction with an opaque error.
                # Composite (dict-bearing) knobs are left alone — dicts are unhashable
                # and their categorical decode is index-based, not value-based.
                choices = list(dict.fromkeys(choices))
            if len(choices) <= 1:
                if choices:
                    self._constants[knob] = choices[0]  # fixed -> inject, not a param
                continue
            if any(isinstance(c, dict) for c in choices):
                # composite knob with pinned-dict entries -> categorical over index
                root.add_categorical_param(knob, [str(i) for i in range(len(choices))])
                self._decoders[knob] = _index_decoder(choices)
            elif all(isinstance(c, str) for c in choices):
                root.add_categorical_param(knob, list(choices))
                self._decoders[knob] = _decoder_for(choices)
            else:
                root.add_discrete_param(knob, sorted(float(c) for c in choices))
                self._decoders[knob] = _decoder_for(choices)

        problem.metric_information.append(vz.MetricInformation(name=_METRIC, goal=vz.ObjectiveMetricGoal.MAXIMIZE))
        study_config = vz.StudyConfig.from_problem(problem)
        study_config.algorithm = "DEFAULT"
        self._study = clients.Study.from_study_config(study_config, owner="spica", study_id=study_id)

    def suggest(self, count: int) -> list[Suggestion]:
        suggestions: list[Suggestion] = []
        for trial in self._study.suggest(count=count):
            params = dict(trial.parameters)
            # backend is a searched knob now (in knob_choices) -> comes via _constants
            # (single backend) or _decoders (multiple), not a per-branch constant.
            selection: dict[str, Any] = {
                "deployment_mode": self.branch.deployment_mode,
                **self._constants,
            }
            for knob, decode in self._decoders.items():
                selection[knob] = decode(params[knob])
            parallel_config = self.branch.parallel_configs[int(params[_PARALLEL_PARAM])]
            suggestions.append(Suggestion(selection=selection, parallel_config=parallel_config, handle=trial))
        return suggestions

    def observe(self, suggestion: Suggestion, score: float) -> None:
        from vizier.service import pyvizier as vz

        suggestion.handle.complete(vz.Measurement(metrics={_METRIC: float(score)}))

    def observe_infeasible(self, suggestion: Suggestion, reason: str) -> None:
        """Mark a candidate that could not be evaluated (e.g. replay error) so the
        study still closes the trial and the optimizer moves on."""
        from vizier.service import pyvizier as vz

        suggestion.handle.complete(vz.Measurement(), infeasible_reason=reason)


def make_branch_sampler(branch: BranchSpace, *, study_id: str) -> BranchSampler:
    """Construct the default (Vizier) sampler for a branch."""
    return VizierBranchSampler(branch, study_id=study_id)
