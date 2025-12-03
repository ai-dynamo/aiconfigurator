# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Apply global Gradio monkey patches
# This must be imported before any Gradio components are created
from aiconfigurator.webapp import gradio_patches  # noqa: F401
