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
        required = kwargs.pop("required", True)
        optional = kwargs.pop("optional", False)

        # Add CSS classes (styling handled via CSS ::after pseudo-element)
        if optional:  # optional has higher precedence than required
            _add_elem_class(kwargs, "optional")
        elif required:
            _add_elem_class(kwargs, "required")

        # Call original init
        return original_init(self, *args, **kwargs)

    # Preserve the original signature for Gradio's introspection
    patched_init.__signature__ = original_signature

    return patched_init


# Patch Dropdown.preprocess to suppress "Value not in choices" errors
# This can happen when choices are dynamically updated but frontend has stale value
_original_dropdown_preprocess = gr.Dropdown.preprocess


def _patched_dropdown_preprocess(self, payload):
    """
    Patched preprocess that returns None instead of raising an error
    when the submitted value is not in the current list of choices.
    """
    try:
        return _original_dropdown_preprocess(self, payload)
    except gr.exceptions.Error as e:
        if "is not in the list of choices" in str(e):
            # Return None for invalid values - handlers should deal with None gracefully
            return None
        raise


gr.Dropdown.preprocess = _patched_dropdown_preprocess

# Monkey patch Gradio components for required/optional labels
gr.Dropdown.__init__ = _create_patched_init(gr.Dropdown.__init__)
gr.Number.__init__ = _create_patched_init(gr.Number.__init__)
gr.Textbox.__init__ = _create_patched_init(gr.Textbox.__init__)
gr.CheckboxGroup.__init__ = _create_patched_init(gr.CheckboxGroup.__init__)
gr.Checkbox.__init__ = _create_patched_init(gr.Checkbox.__init__)
