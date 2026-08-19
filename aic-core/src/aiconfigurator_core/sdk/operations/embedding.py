# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Embedding operation (ISSUE-04 / AIC-477).

No CSV-backed data — latency derived analytically from ``mem_bw``. The
base ``Operation.load_data`` no-op default handles the missing table.
``query()`` calls ``database.query_mem_op`` (the legacy entry point on
``PerfDatabase``); deciding a long-term home for the analytical mem-op
formula is deferred to the post-refactor cleanup.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import aiconfigurator_core._aiconfigurator_core as _core
from aiconfigurator_core.sdk.operations.base import OpShellKit

if TYPE_CHECKING:
    pass


class Embedding(_core.Embedding, OpShellKit):
    """Embedding operation (Rust-backed; see py_ops.rs)."""
