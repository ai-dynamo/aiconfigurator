# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Repository-specific routing contract for AIC's generated CODEOWNERS."""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from codeowners_match import parse_codeowners, resolve_owners

ROOT = Path(__file__).resolve().parents[2]
MAINTAINERS = "@ai-dynamo/maintainers-aiconfigurator"
AREA_TEAMS = {
    "@ai-dynamo/aiconfigurator-runtime",
    "@ai-dynamo/aiconfigurator-generators",
    "@ai-dynamo/aiconfigurator-infra",
}


def _owners(path: str) -> set[str]:
    rules = parse_codeowners((ROOT / "CODEOWNERS").read_text())
    return set(resolve_owners(rules, path))


def test_area_teams_retain_maintainer_coownership() -> None:
    """The migration must not drop the existing maintainer review request."""
    rules = parse_codeowners((ROOT / "CODEOWNERS").read_text())
    violations = []
    tracked = subprocess.check_output(["git", "-C", str(ROOT), "ls-files"], text=True).splitlines()
    for path in tracked:
        owners = set(resolve_owners(rules, path))
        if owners & AREA_TEAMS and MAINTAINERS not in owners:
            violations.append(f"{path}: {sorted(owners)}")
    assert not violations


def test_representative_routing_contract() -> None:
    assert _owners("src/aiconfigurator/main.py") == {
        "@ai-dynamo/aiconfigurator-runtime",
        MAINTAINERS,
    }
    assert _owners("src/aiconfigurator/generator/__init__.py") == {
        "@ai-dynamo/aiconfigurator-generators",
        MAINTAINERS,
    }
    assert _owners("aic-core/rust/aiconfigurator-core/Cargo.toml") == {
        "@ai-dynamo/aiconfigurator-runtime",
        MAINTAINERS,
    }
    assert _owners("tools/generator_validator/README.md") == {
        "@ai-dynamo/aiconfigurator-generators",
        MAINTAINERS,
    }
    assert _owners("tools/sanity_check/README.md") == {
        "@ai-dynamo/aiconfigurator-runtime",
        MAINTAINERS,
    }
    assert _owners(".github/workflows/build-test.yml") == {
        "@ai-dynamo/aiconfigurator-infra",
        MAINTAINERS,
    }
    assert _owners("tests/conftest.py") == AREA_TEAMS | {MAINTAINERS}
    assert _owners("docs/dynamo_deployment_guide.md") == {
        "@ai-dynamo/aiconfigurator-runtime",
        "@ai-dynamo/aiconfigurator-infra",
        MAINTAINERS,
    }
