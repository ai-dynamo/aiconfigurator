# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed options for the explicit FPM campaign collector."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

FPM_FORWARD_OP = "fpm_forward"
FPM_WARMUP_REPEATS = 1
FPM_MEASUREMENT_REPEATS = 1

PARALLEL_AXES = ("tp", "pp", "dp", "moe_tp", "moe_ep", "cp")
BACKEND_AXES = (
    "baseline",
    "prefill_attention",
    "decode_attention",
    "moe",
    "tp_communication",
    "ep_communication",
)
SAMPLING_BUDGETS = ("one_eighth", "one_quarter", "one_half", "full")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _optional_size_list(values: list[int] | None) -> tuple[int, ...] | None:
    if values is None:
        return None
    return tuple(sorted(set(values)))


@dataclass(frozen=True, slots=True)
class FPMCollectionOptions:
    """Resolved FPM case-space controls.

    The options only narrow an AIC-declared/model-valid universe. They never
    grant support for a topology, quantization, or backend policy.
    """

    max_gpus: int
    gpu_counts: tuple[int, ...]
    parallel_axes: tuple[str, ...]
    backend_axes: tuple[str, ...]
    weight_quantizations: tuple[str, ...]
    kv_cache_dtypes: tuple[str, ...]
    sampling_budget: str
    tp_sizes: tuple[int, ...] | None = None
    pp_sizes: tuple[int, ...] | None = None
    dp_sizes: tuple[int, ...] | None = None
    moe_tp_sizes: tuple[int, ...] | None = None
    moe_ep_sizes: tuple[int, ...] | None = None
    cp_sizes: tuple[int, ...] | None = None
    kv_block_size: int = 64
    warmup_repeats: int = FPM_WARMUP_REPEATS
    measurement_repeats: int = FPM_MEASUREMENT_REPEATS
    smoke_points: int = 1

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> FPMCollectionOptions:
        max_gpus = getattr(args, "fpm_max_gpus", None)
        if max_gpus is None:
            raise ValueError("--fpm-max-gpus is required with --ops fpm_forward")

        requested_counts = getattr(args, "fpm_gpu_counts", None)
        if requested_counts is None:
            counts = []
            value = 1
            while value <= max_gpus:
                counts.append(value)
                value *= 2
            if counts[-1] != max_gpus:
                counts.append(max_gpus)
        else:
            counts = sorted(set(requested_counts))
        over_limit = [count for count in counts if count > max_gpus]
        if over_limit:
            raise ValueError(f"--fpm-gpu-counts values exceed --fpm-max-gpus={max_gpus}: {over_limit}")

        return cls(
            max_gpus=max_gpus,
            gpu_counts=tuple(counts),
            parallel_axes=tuple(dict.fromkeys(args.fpm_parallel_axes or ("tp", "pp", "dp", "moe_tp", "moe_ep"))),
            backend_axes=tuple(dict.fromkeys(args.fpm_backend_axes or ("baseline",))),
            weight_quantizations=tuple(dict.fromkeys(value.lower() for value in (args.fpm_weight_quantizations or ()))),
            kv_cache_dtypes=tuple(dict.fromkeys(args.fpm_kv_cache_dtypes or ("auto",))),
            sampling_budget=args.fpm_sampling_budget or "one_quarter",
            tp_sizes=_optional_size_list(args.fpm_tp_sizes),
            pp_sizes=_optional_size_list(args.fpm_pp_sizes),
            dp_sizes=_optional_size_list(args.fpm_dp_sizes),
            moe_tp_sizes=_optional_size_list(args.fpm_moe_tp_sizes),
            moe_ep_sizes=_optional_size_list(args.fpm_moe_ep_sizes),
            cp_sizes=_optional_size_list(args.fpm_cp_sizes),
            kv_block_size=args.fpm_kv_block_size or 64,
            smoke_points=args.fpm_smoke_points or 1,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "max_gpus": self.max_gpus,
            "gpu_counts": list(self.gpu_counts),
            "parallel_axes": list(self.parallel_axes),
            "backend_axes": list(self.backend_axes),
            "weight_quantizations": list(self.weight_quantizations),
            "kv_cache_dtypes": list(self.kv_cache_dtypes),
            "sampling_budget": self.sampling_budget,
            "tp_sizes": list(self.tp_sizes) if self.tp_sizes is not None else None,
            "pp_sizes": list(self.pp_sizes) if self.pp_sizes is not None else None,
            "dp_sizes": list(self.dp_sizes) if self.dp_sizes is not None else None,
            "moe_tp_sizes": list(self.moe_tp_sizes) if self.moe_tp_sizes is not None else None,
            "moe_ep_sizes": list(self.moe_ep_sizes) if self.moe_ep_sizes is not None else None,
            "cp_sizes": list(self.cp_sizes) if self.cp_sizes is not None else None,
            "kv_block_size": self.kv_block_size,
            "warmup_repeats": self.warmup_repeats,
            "measurement_repeats": self.measurement_repeats,
            "smoke_points": self.smoke_points,
        }


def add_fpm_arguments(parser: argparse.ArgumentParser) -> None:
    """Add explicit-only FPM campaign controls to the existing collector CLI."""

    group = parser.add_argument_group(
        "FPM forward collection",
        "Used only when --ops fpm_forward is selected.",
    )
    group.add_argument(
        "--fpm-max-gpus",
        type=_positive_int,
        default=None,
        help="Maximum GPUs used by one FPM deployment cell (required).",
    )
    group.add_argument(
        "--fpm-gpu-counts",
        nargs="+",
        type=_positive_int,
        default=None,
        help="Exact total-GPU counts to consider; defaults to powers of two up to the maximum.",
    )
    group.add_argument(
        "--fpm-parallel-axes",
        nargs="+",
        choices=PARALLEL_AXES,
        default=None,
        help="Parallel dimensions AIC may vary while enumerating valid topologies.",
    )
    group.add_argument(
        "--fpm-backend-axes",
        nargs="+",
        choices=BACKEND_AXES,
        default=None,
        help="Baseline and one-axis-at-a-time backend policies to collect.",
    )
    group.add_argument(
        "--fpm-weight-quantizations",
        nargs="+",
        default=None,
        help="Checkpoint-backed weight quantizations to retain; never changes checkpoint identity.",
    )
    group.add_argument(
        "--fpm-kv-cache-dtypes",
        nargs="+",
        default=None,
        help="Runtime KV-cache dtypes to collect; defaults to the generator/model setting.",
    )
    group.add_argument(
        "--fpm-sampling-budget",
        choices=SAMPLING_BUDGETS,
        default=None,
        help="Nested sparse-design budget to measure (default: one_quarter).",
    )
    group.add_argument(
        "--fpm-kv-block-size",
        type=_positive_int,
        default=None,
        help="KV block size used to derive block-aligned P>0 candidates (default: 64).",
    )
    group.add_argument(
        "--fpm-artifact-root",
        default=None,
        help="Out-of-place root for generated artifacts, raw results, logs, and checkpoints.",
    )
    group.add_argument(
        "--fpm-database-root",
        default=None,
        help="Optional out-of-place systems/data root for formal database publication.",
    )
    group.add_argument(
        "--fpm-smoke-points",
        type=_positive_int,
        default=None,
        help="Points per workload cell in --smoke mode (default: 1).",
    )
    for axis, option in (
        ("tp", "--fpm-tp-sizes"),
        ("pp", "--fpm-pp-sizes"),
        ("dp", "--fpm-dp-sizes"),
        ("moe_tp", "--fpm-moe-tp-sizes"),
        ("moe_ep", "--fpm-moe-ep-sizes"),
        ("cp", "--fpm-cp-sizes"),
    ):
        group.add_argument(
            option,
            nargs="+",
            type=_positive_int,
            default=None,
            help=f"Optional allowlist for the {axis} dimension.",
        )


def add_fpm_generator_arguments(parser: argparse.ArgumentParser) -> None:
    """Expose the existing Generator override surface without importing it.

    Importing ``aiconfigurator.generator.api`` loads the rendering stack. That
    is appropriate during execution, but unnecessary for ``--plan-only`` and
    must not become a dependency of ordinary op-level collection.
    """

    group = parser.add_argument_group("FPM Generator and Kubernetes overrides")
    group.add_argument("--generator-config", default=None)
    group.add_argument("--generator-set", action="append", default=None, metavar="KEY=VALUE")
    group.add_argument(
        "--config-template-version",
        "--generated-config-version",
        dest="generated_config_version",
        default=None,
    )
    group.add_argument(
        "--dynamo-version",
        "--generator-dynamo-version",
        dest="generator_dynamo_version",
        default=None,
    )
    group.add_argument("--namespace", default=None)
    group.add_argument("--model-cache", default=None, metavar="NAME[:MOUNT[:SUBPATH]]")
    group.add_argument("--transport", choices=["nvlink", "ib", "efa"], default=None)
    group.add_argument("--image-pull-secret", dest="image_pull_secret", default=None)


def reject_fpm_arguments_without_fpm(args: argparse.Namespace) -> None:
    """Fail when a user supplies FPM-only controls without selecting the op."""

    if FPM_FORWARD_OP in (args.ops or ()):
        return
    explicitly_set = []
    for name in (
        "fpm_max_gpus",
        "fpm_gpu_counts",
        "fpm_weight_quantizations",
        "fpm_kv_cache_dtypes",
        "fpm_tp_sizes",
        "fpm_pp_sizes",
        "fpm_dp_sizes",
        "fpm_moe_tp_sizes",
        "fpm_moe_ep_sizes",
        "fpm_cp_sizes",
        "fpm_parallel_axes",
        "fpm_backend_axes",
        "fpm_sampling_budget",
        "fpm_kv_block_size",
        "fpm_artifact_root",
        "fpm_database_root",
        "fpm_smoke_points",
    ):
        if getattr(args, name, None) is not None:
            explicitly_set.append("--" + name.replace("_", "-"))
    if explicitly_set:
        raise ValueError("FPM-only arguments require --ops fpm_forward: " + ", ".join(explicitly_set))
