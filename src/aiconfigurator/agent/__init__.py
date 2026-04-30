# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from importlib.resources import files

_SKILL_FILES = {
    "usage": "skills/usage/SKILL.md",
    "development": "skills/development/SKILL.md",
}

_REFERENCE_FILES = {
    "usage": {
        "cli-modes": "skills/usage/references/cli-modes.md",
        "single-experiment-yaml": "skills/usage/references/single-experiment-yaml.md",
        "result-interpretation": "skills/usage/references/result-interpretation.md",
        "sdk-step-breakdown": "skills/usage/references/sdk-step-breakdown.md",
        "deployment-bench": "skills/usage/references/deployment-bench.md",
        "feature-pitfalls": "skills/usage/references/feature-pitfalls.md",
        "examples": "skills/usage/references/examples.md",
        "troubleshooting": "skills/usage/references/troubleshooting.md",
    },
    "development": {
        "repo-layout": "skills/development/references/repo-layout.md",
        "testing": "skills/development/references/testing.md",
        "model-support": "skills/development/references/model-support.md",
        "generator": "skills/development/references/generator.md",
        "pr-checklist": "skills/development/references/pr-checklist.md",
    },
}


def list_agent_skills() -> tuple[str, ...]:
    """Return the bundled agent skill names."""
    return tuple(_SKILL_FILES)


def list_agent_refs(skill: str) -> tuple[str, ...]:
    """Return reference names available for a bundled agent skill."""
    if skill not in _SKILL_FILES:
        raise ValueError(f"Unknown agent skill '{skill}'. Available skills: {', '.join(list_agent_skills())}")
    return tuple(_REFERENCE_FILES.get(skill, {}))


def get_agent_text(skill: str = "usage", reference: str | None = None) -> str:
    """Read a bundled agent skill or one of its progressive references."""
    if skill not in _SKILL_FILES:
        raise ValueError(f"Unknown agent skill '{skill}'. Available skills: {', '.join(list_agent_skills())}")

    if reference is None:
        resource_path = _SKILL_FILES[skill]
    else:
        references = _REFERENCE_FILES.get(skill, {})
        if reference not in references:
            available = ", ".join(references) or "<none>"
            raise ValueError(f"Unknown reference '{reference}' for skill '{skill}'. Available references: {available}")
        resource_path = references[reference]

    return files(__name__).joinpath(*resource_path.split("/")).read_text(encoding="utf-8")
