# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Global monkey patches for Gradio components.
This module patches Gradio components to add custom functionality across the entire application.
"""

import functools
import inspect

import gradio as gr


def _add_elem_class(kwargs, class_name):
    """Add a CSS class to elem_classes, handling existing values."""
    existing = kwargs.get("elem_classes", None)
    if existing is None:
        kwargs["elem_classes"] = [class_name]
    elif isinstance(existing, str):
        kwargs["elem_classes"] = [existing, class_name]
    elif isinstance(existing, list):
        kwargs["elem_classes"] = existing + [class_name]


def _create_patched_init(original_init):
    """
    Create a patched __init__ that adds 'required'/'optional' CSS classes.
    Preserves the original function signature for Gradio's introspection.
    """
    original_signature = inspect.signature(original_init)

    @functools.wraps(original_init)
    def patched_init(self, *args, **kwargs):
        # Extract our custom parameters before passing to original
        required = kwargs.pop("required", False)
        optional = kwargs.pop("optional", False)

        # Add CSS classes (styling handled via CSS ::after pseudo-element)
        if required:
            _add_elem_class(kwargs, "required")
        elif optional:
            _add_elem_class(kwargs, "optional")

        # Call original init
        return original_init(self, *args, **kwargs)

    # Preserve the original signature for Gradio's introspection
    patched_init.__signature__ = original_signature

    return patched_init


# Monkey patch Dropdown to support required/optional labels
gr.Dropdown.__init__ = _create_patched_init(gr.Dropdown.__init__)

# Monkey patch Number to support required/optional labels
gr.Number.__init__ = _create_patched_init(gr.Number.__init__)
