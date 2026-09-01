# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""No single-host vLLM WideEP ops.

Distributed ``moe_a2a`` collection is intentionally standalone.
"""

from collector.registry_types import OpEntry

REGISTRY: list[OpEntry] = []
