# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Whole-model forward-pass collection support.

This package is an internal campaign runner used by ``collector/collect.py``.
It intentionally has no standalone CLI: users select ``--ops fpm_forward``
through the existing collector entry point.
"""

from .config import FPM_FORWARD_OP, FPMCollectionOptions, add_fpm_arguments

__all__ = ["FPM_FORWARD_OP", "FPMCollectionOptions", "add_fpm_arguments"]
