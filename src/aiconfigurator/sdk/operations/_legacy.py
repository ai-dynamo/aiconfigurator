# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Legacy home for op classes that have not yet been migrated to their own
files. Each ISSUE-04..14 moves a family of classes out of this file into a
dedicated module. Final cleanup (ISSUE-15) deletes this file once empty."""

import logging

logger = logging.getLogger(__name__)
