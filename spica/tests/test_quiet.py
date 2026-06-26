# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``import spica`` must quiet the jax/vizier import noise (see spica._quiet)."""

import logging
import os


def test_import_spica_quiets_jax():
    import spica  # noqa: F401  -- importing triggers spica._quiet's side effects

    # jax pinned to CPU (the GP-bandit is CPU-only) -> no GPU-probe warning
    assert os.environ.get("JAX_PLATFORMS") == "cpu"
    # the noisy jax/absl loggers are raised to ERROR so a sweep doesn't spam warnings
    assert logging.getLogger("jax._src.xla_bridge").level == logging.ERROR
    assert logging.getLogger("absl").level == logging.ERROR
