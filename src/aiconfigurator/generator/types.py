# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ArtifactBundle:
    """In-memory representation of generated artifacts by mode."""
    by_mode: Dict[str, Dict[str, Any]]