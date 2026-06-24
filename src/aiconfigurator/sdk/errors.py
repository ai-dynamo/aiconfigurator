# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared SDK exception types."""


class NoFeasibleConfigError(RuntimeError):
    """Raised when no configuration satisfies user-provided SLA constraints."""


class UnsupportedWideepConfigError(ValueError):
    """Raised when a requested WideEP configuration is not in the perf database.

    Subclasses ``ValueError`` so callers that ``except ValueError`` still catch it.
    """
