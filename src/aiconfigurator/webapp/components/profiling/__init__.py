# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Profiling module for the AI Configurator webapp.

PRIMARY ENTRY POINT:
    - create_profiling_tab: Create the profiling UI tab for main webapp

REUSABLE COMPONENTS:
    - create_profiling_ui_components: Create hidden UI components
    - create_performance_results_section: Create performance results visualization
    - inject_profiling_assets: Inject CSS and modal HTML
    - load_profiling_javascript: Load JavaScript modules for visualization
"""

from .create_profiling_tab import (
    _load_profiling_javascript as load_profiling_javascript,
)
from .create_profiling_tab import (
    create_performance_results_section,
    create_profiling_tab,
    create_profiling_ui_components,
    inject_profiling_assets,
)

__all__ = [
    "create_performance_results_section",
    "create_profiling_tab",
    "create_profiling_ui_components",
    "inject_profiling_assets",
    "load_profiling_javascript",
]
