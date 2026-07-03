# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Silence the jax / equinox / jaxopt import noise that Vizier's GP-bandit pulls in.

Vizier (the smart-sweep optimizer) depends on jax, and spica only ever runs jax on **CPU** —
the GP-bandit is a tiny CPU-side optimization; the actual replay compute lives in the Rust
mocker, not jax. So we:

- pin ``JAX_PLATFORMS=cpu`` to skip jax's GPU probe and its noisy
  *"An NVIDIA GPU may be present ... but a CUDA-enabled jaxlib is not installed. Falling back
  to cpu."* warning (spica's venv ships CPU jax on purpose);
- quiet the ``jax`` / ``absl`` loggers (e.g. absl's *"Python 3.8+ is required"*);
- drop the jax / jaxlib / equinox / jaxopt ``DeprecationWarning``s and the
  *"JAXopt is no longer maintained"* notice.

Imported **first** from :mod:`spica` so it runs before anything (lazily) imports vizier/jax.
Uses ``setdefault`` + module-scoped warning filters so a deliberate jax/GPU user is unaffected.
"""

from __future__ import annotations

import logging
import os
import warnings

# Must precede the first ``import jax`` (Vizier imports it lazily during a sweep).
os.environ.setdefault("JAX_PLATFORMS", "cpu")

for _logger in ("jax", "jax._src.xla_bridge", "absl"):
    logging.getLogger(_logger).setLevel(logging.ERROR)

for _module in (r"jax(\..*)?", r"jaxlib(\..*)?", r"equinox(\..*)?", r"jaxopt(\..*)?"):
    warnings.filterwarnings("ignore", category=DeprecationWarning, module=_module)
warnings.filterwarnings("ignore", message=".*JAXopt is no longer maintained.*")
