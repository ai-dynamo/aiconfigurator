# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Backends initializer.

This module exposes get_generator from base and imports concrete backends to
ensure they are registered at import time.
"""
from .base import get_generator, register, BaseGenerator

# Import concrete backends to trigger registration decorators.
from . import trtllm
# from . import vllm
# from . import sglang