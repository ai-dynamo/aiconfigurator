# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Core-namespaced access to selected public AIConfigurator SDK APIs.

The implementations remain under :mod:`aiconfigurator.sdk` so existing users
keep a stable import path.  This package provides aliases for callers that want
their imports to reflect that the standalone ``aiconfigurator-core``
distribution owns the SDK payload.
"""
