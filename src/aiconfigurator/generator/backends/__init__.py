# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Backends initializer.

This module exposes get_generator from base and imports concrete backends to
ensure they are registered at import time.
"""

# Import concrete backends to trigger registration decorators.
from . import trtllm
from .base import BaseGenerator, get_generator, register

# from . import vllm
# from . import sglang

__all__ = ["BaseGenerator", "get_generator", "register", "trtllm"]
