# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang WideEP collectors."""

import os


def dataset_version_label(env_var: str, op: str) -> str:
    """Resolve the ``version`` column written on the perf rows of ``op``.

    DeepEP kernels ship with ``deep_ep``, not with sglang, so this column is a
    dataset bucket key rather than the measured runtime. It must default to the
    bucket declared for that op in ``collector/framework_manifest.yaml`` so rows
    agree with the directory they are finalized into -- defaulting to the
    installed sglang build mislabels every row whenever the DeepEP image ships a
    different sglang than the manifest pins (which is how the 0.5.10 tree ended
    up holding ``version: 0.5.12`` rows). Resolution is per-op, not per-
    framework: a family override pins the DeepEP ops to a different runtime than
    the WideEP default that serves ``wideep_moe``.
    """
    override = os.environ.get(env_var)
    if override:
        return override
    try:
        from collector.framework_manifest import resolve_op_runtime

        return resolve_op_runtime("wideep_sglang", op).version
    except Exception:
        pass
    try:
        from importlib.metadata import version as get_version

        return get_version("sglang")
    except Exception:
        return "unknown"
