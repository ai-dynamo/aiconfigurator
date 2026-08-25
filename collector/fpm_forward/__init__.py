# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Whole-model forward-pass collection support.

Run the dedicated campaign entry point with ``python -m collector.fpm_forward``.
The legacy ``collector/collect.py --ops fpm_forward`` route remains compatible.
"""

from .config import FPM_FORWARD_OP, FPMCollectionOptions, add_fpm_arguments

__all__ = ["FPM_FORWARD_OP", "FPMCollectionOptions", "add_fpm_arguments"]
