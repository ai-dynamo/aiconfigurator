# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang WideEP collectors."""

import os


def dataset_version_label(env_var: str) -> str:
    """Resolve the ``version`` column written on DeepEP perf rows.

    DeepEP kernels ship with ``deep_ep``, not with sglang, so this column is a
    dataset bucket key rather than the measured runtime. It must default to the
    WideEP bucket declared in ``collector/framework_manifest.yaml`` so rows agree
    with the directory they are finalized into -- defaulting to the installed
    sglang build mislabels every row whenever the DeepEP image ships a different
    sglang than the manifest pins (which is how the 0.5.10 tree ended up holding
    ``version: 0.5.12`` rows).
    """
    override = os.environ.get(env_var)
    if override:
        return override
    try:
        from collector.framework_manifest import get_collector_runtime

        return get_collector_runtime("sglang", workload="wideep").version
    except Exception:
        pass
    try:
        from importlib.metadata import version as get_version

        return get_version("sglang")
    except Exception:
        return "unknown"
