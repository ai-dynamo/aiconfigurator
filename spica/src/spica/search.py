# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smart-search entry point.

The real implementation will build the branch-aware search space, run the
Vizier + replay sweep, and rank candidates. This milestone only lands the
input schema and a stub so the project installs and validates configs; the
search loop and the (injectable) deployment evaluator come in later steps.
"""

from __future__ import annotations

from typing import Any

from .config import Candidate, SmartSearchConfig


def run_smart_search(config: SmartSearchConfig, *, evaluator: Any = None) -> list[Candidate]:
    """Run the Vizier + replay sweep described by ``config`` and return the
    evaluated candidates sorted best-first.

    Not implemented yet — the search loop and the Replay-backed deployment
    evaluator are landed in later milestones.
    """
    raise NotImplementedError(
        "run_smart_search is not implemented yet; the input schema is in place. "
        "Search loop + evaluator land in later milestones."
    )
