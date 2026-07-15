# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed options for Dynamo-native FPM self-benchmark campaigns."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

FPM_FORWARD_OP = "fpm_forward"
FPM_WARMUP_ITERATIONS = 5
FPM_MEASUREMENT_REPEATS = 1
FPM_MAX_PREFILL_ISL = 8192
FPM_MAX_PREFILL_CUDAGRAPH_SIZE = 2048
VLLM_AUTO_FIT_MAX_MODEL_LEN = -1

PARALLEL_AXES = ("tp", "pp", "dp", "moe_tp", "moe_ep", "cp")
PARALLEL_PRESETS = ("auto", "tp", "tep", "dep", "pure_tp")
BACKEND_AXES = (
    "auto",
    "baseline",
    "prefill_attention",
    "decode_attention",
    "moe",
    "tp_communication",
    "ep_communication",
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _at_least_two_int(value: str) -> int:
    parsed = int(value)
    if parsed < 2:
        raise argparse.ArgumentTypeError("value must be at least 2")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return parsed


def _optional_size_list(values: list[int] | None) -> tuple[int, ...] | None:
    if values is None:
        return None
    return tuple(sorted(set(values)))


def _powers_of_two_up_to(limit: int) -> tuple[int, ...]:
    values = []
    value = 1
    while value <= limit:
        values.append(value)
        value *= 2
    return tuple(values)


def _prefill_cudagraph_capture_sizes(max_isl: int) -> tuple[int, ...]:
    """Build the formal prefill capture axis from vLLM's balanced defaults."""

    max_capture_size = min(max_isl, FPM_MAX_PREFILL_CUDAGRAPH_SIZE)
    sizes = [value for value in (1, 2, 4) if value <= max_capture_size]
    sizes.extend(range(8, min(max_capture_size + 1, 256), 8))
    if max_capture_size >= 256:
        sizes.extend(range(256, min(max_capture_size + 1, 513), 16))
    if max_capture_size > 512:
        sizes.extend(range(544, max_capture_size + 1, 32))
    # vLLM requires an explicitly supplied max capture size to equal the
    # largest explicit capture. Preserve an off-stride user endpoint exactly.
    sizes.append(max_capture_size)
    return tuple(sorted(set(sizes)))


def _dynamo_cudagraph_axis_points(capture_sizes: tuple[int, ...], limit: int) -> tuple[int, ...]:
    """Mirror PR11509's ``_cudagraph_axis_points`` candidate expansion."""

    configured = tuple(sorted({size for size in capture_sizes if size >= 1}))
    if not configured:
        return tuple(sorted({*_powers_of_two_up_to(limit), limit}))

    points: set[int] = set()
    for capture_size in configured:
        if capture_size > limit:
            continue
        points.add(capture_size)
        if capture_size < limit:
            points.add(capture_size + 1)

    if configured[-1] <= limit:
        value = configured[-1] * 2
        while value < limit:
            points.add(value)
            value *= 2
    points.add(limit)
    return tuple(sorted(points))


@dataclass(frozen=True, slots=True)
class PrefillSamplingProfile:
    """Collector-owned inputs to Dynamo's native prefill grid generator."""

    max_isl: int
    max_batch_size: int | None
    max_total_prefill_tokens: int
    cudagraph_capture_sizes: tuple[int, ...]
    max_cudagraph_capture_size: int
    new_token_axis_points: tuple[int, ...]
    max_new_token_samples: int
    max_kv_read_token_samples: int

    @classmethod
    def build(cls, *, max_isl: int, max_batch_size: int | None) -> PrefillSamplingProfile:
        if max_isl < 2:
            raise ValueError("FPM max prefill ISL must be at least 2")
        if max_batch_size is not None and max_batch_size < 1:
            raise ValueError("FPM max prefill batch size must be positive")

        capture_sizes = _prefill_cudagraph_capture_sizes(max_isl)
        new_token_points = _dynamo_cudagraph_axis_points(capture_sizes, max_isl)
        batch_upper_bound = min(max_isl, max_batch_size or max_isl)
        # PR11509's KV axis contains zero, a batch-aligned minimum, powers of
        # two in block units, and an exact maximum. This upper bound prevents
        # its uniform limiter from deleting any generated candidate.
        max_total_kv_tokens = max_isl * batch_upper_bound
        max_kv_read_samples = max_total_kv_tokens.bit_length() + 2
        return cls(
            max_isl=max_isl,
            max_batch_size=max_batch_size,
            max_total_prefill_tokens=max_isl,
            cudagraph_capture_sizes=capture_sizes,
            max_cudagraph_capture_size=capture_sizes[-1],
            new_token_axis_points=new_token_points,
            max_new_token_samples=max(2, len(new_token_points)),
            max_kv_read_token_samples=max_kv_read_samples,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "max_isl": self.max_isl,
            "max_batch_size": self.max_batch_size,
            "max_total_prefill_tokens": self.max_total_prefill_tokens,
            "cudagraph_capture_sizes": list(self.cudagraph_capture_sizes),
            "cudagraph_capture_size_count": len(self.cudagraph_capture_sizes),
            "max_cudagraph_capture_size": self.max_cudagraph_capture_size,
            "new_token_axis_points": list(self.new_token_axis_points),
            "new_token_axis_point_count": len(self.new_token_axis_points),
            "prefill_max_new_token_samples": self.max_new_token_samples,
            "prefill_max_kv_read_token_samples": self.max_kv_read_token_samples,
        }


@dataclass(frozen=True, slots=True)
class FPMCollectionOptions:
    """Resolved FPM case-space controls.

    The options only narrow an AIC-declared/model-valid universe. They never
    grant support for a topology, quantization, or backend policy.
    """

    max_gpus: int
    gpu_counts: tuple[int, ...]
    parallel_presets: tuple[str, ...]
    parallel_axes: tuple[str, ...]
    backend_axes: tuple[str, ...]
    weight_quantizations: tuple[str, ...]
    kv_cache_dtypes: tuple[str, ...]
    tp_sizes: tuple[int, ...] | None = None
    pp_sizes: tuple[int, ...] | None = None
    dp_sizes: tuple[int, ...] | None = None
    moe_tp_sizes: tuple[int, ...] | None = None
    moe_ep_sizes: tuple[int, ...] | None = None
    cp_sizes: tuple[int, ...] | None = None
    warmup_iterations: int = FPM_WARMUP_ITERATIONS
    # Keep the model context independent from the scheduled new-token budget.
    # vLLM 0.24 auto-fits this limit after model/CUDA-graph profiling, and
    # Dynamo records the resolved value in ``limits.max_model_len``.
    vllm_max_model_len: int = VLLM_AUTO_FIT_MAX_MODEL_LEN
    max_prefill_isl: int = FPM_MAX_PREFILL_ISL
    max_prefill_batch_size: int | None = None

    @property
    def prefill_sampling(self) -> PrefillSamplingProfile:
        return PrefillSamplingProfile.build(
            max_isl=self.max_prefill_isl,
            max_batch_size=self.max_prefill_batch_size,
        )

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

        explicit_presets = getattr(args, "fpm_parallel_presets", None)
        requested_axes = tuple(dict.fromkeys(getattr(args, "fpm_parallel_axes", None) or ()))
        requested_presets = tuple(dict.fromkeys(explicit_presets or (() if requested_axes else ("auto",))))
        if explicit_presets is not None and requested_axes:
            raise ValueError("--fpm-parallel-presets and legacy --fpm-parallel-axes cannot be combined")
        if "auto" in requested_presets and len(requested_presets) > 1:
            raise ValueError("parallel preset 'auto' cannot be combined with explicit presets")

        pp_sizes = _optional_size_list(getattr(args, "fpm_pp_sizes", None))
        cp_sizes = _optional_size_list(getattr(args, "fpm_cp_sizes", None))
        if pp_sizes not in {None, (1,)} or cp_sizes not in {None, (1,)}:
            raise ValueError("FPM typical-matrix V1 fixes PP=1 and CP=1")
        if {"pp", "cp"}.intersection(requested_axes):
            raise ValueError("FPM typical-matrix V1 does not vary PP or CP")

        return cls(
            max_gpus=max_gpus,
            gpu_counts=tuple(counts),
            parallel_presets=requested_presets,
            parallel_axes=requested_axes,
            backend_axes=tuple(dict.fromkeys(getattr(args, "fpm_backend_axes", None) or ("auto",))),
            weight_quantizations=tuple(
                dict.fromkeys(value.lower() for value in (getattr(args, "fpm_weight_quantizations", None) or ()))
            ),
            kv_cache_dtypes=tuple(dict.fromkeys(getattr(args, "fpm_kv_cache_dtypes", None) or ("auto",))),
            tp_sizes=_optional_size_list(getattr(args, "fpm_tp_sizes", None)),
            pp_sizes=pp_sizes,
            dp_sizes=_optional_size_list(getattr(args, "fpm_dp_sizes", None)),
            moe_tp_sizes=_optional_size_list(getattr(args, "fpm_moe_tp_sizes", None)),
            moe_ep_sizes=_optional_size_list(getattr(args, "fpm_moe_ep_sizes", None)),
            cp_sizes=cp_sizes,
            warmup_iterations=(
                FPM_WARMUP_ITERATIONS
                if getattr(args, "fpm_warmup_iterations", None) is None
                else args.fpm_warmup_iterations
            ),
            max_prefill_isl=getattr(args, "fpm_max_prefill_isl", None) or FPM_MAX_PREFILL_ISL,
            max_prefill_batch_size=getattr(args, "fpm_max_prefill_batch_size", None),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "max_gpus": self.max_gpus,
            "gpu_counts": list(self.gpu_counts),
            "parallel_presets": list(self.parallel_presets),
            "parallel_axes": list(self.parallel_axes),
            "backend_axes": list(self.backend_axes),
            "weight_quantizations": list(self.weight_quantizations),
            "kv_cache_dtypes": list(self.kv_cache_dtypes),
            "tp_sizes": list(self.tp_sizes) if self.tp_sizes is not None else None,
            "pp_sizes": list(self.pp_sizes) if self.pp_sizes is not None else None,
            "dp_sizes": list(self.dp_sizes) if self.dp_sizes is not None else None,
            "moe_tp_sizes": list(self.moe_tp_sizes) if self.moe_tp_sizes is not None else None,
            "moe_ep_sizes": list(self.moe_ep_sizes) if self.moe_ep_sizes is not None else None,
            "cp_sizes": list(self.cp_sizes) if self.cp_sizes is not None else None,
            "global_warmup_iterations": self.warmup_iterations,
            "vllm_max_model_len": self.vllm_max_model_len,
            "warmup_repeats": 0,
            "measurement_repeats": FPM_MEASUREMENT_REPEATS,
            "point_source": "dynamo_native_self_benchmark",
            "prefill_sampling": self.prefill_sampling.to_dict(),
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
        "--fpm-parallel-presets",
        nargs="+",
        choices=PARALLEL_PRESETS,
        default=None,
        help="Typical deployment families: dense TP, MoE pure TP, TEP, or DEP.",
    )
    group.add_argument(
        "--fpm-parallel-axes",
        nargs="+",
        choices=PARALLEL_AXES,
        default=None,
        help="Deprecated compatibility filter; use --fpm-parallel-presets for new campaigns.",
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
        "--fpm-warmup-iterations",
        type=_nonnegative_int,
        default=None,
        help=(
            "Dynamo scheduler global warmup decode iterations before the point sweep; "
            "does not repeat warmup for each point "
            f"(default: {FPM_WARMUP_ITERATIONS})."
        ),
    )
    group.add_argument(
        "--fpm-max-prefill-isl",
        dest="fpm_max_prefill_isl",
        type=_at_least_two_int,
        default=None,
        help=(
            "Maximum total scheduled prefill new-token axis; the batch=1 points also "
            "cover this per-request new-token length "
            f"(default: {FPM_MAX_PREFILL_ISL})."
        ),
    )
    group.add_argument(
        "--fpm-max-prefill-batch-size",
        dest="fpm_max_prefill_batch_size",
        type=_positive_int,
        default=None,
        help="Optional prefill max_num_seqs override; defaults to the Dynamo/vLLM runtime value.",
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
    """Expose deployment-only Generator inputs without importing its stack.

    Importing ``aiconfigurator.generator.api`` loads the rendering stack. That
    is appropriate during execution, but unnecessary for ``--plan-only`` and
    must not become a dependency of ordinary op-level collection.
    """

    group = parser.add_argument_group("FPM deployment inputs")
    group.add_argument(
        "--generator-config",
        default=None,
        help="Deployment-only YAML containing supported K8sConfig fields.",
    )
    group.add_argument(
        "--generator-set",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Inline deployment-only K8sConfig override.",
    )
    group.add_argument(
        "--config-template-version",
        "--generated-config-version",
        dest="generated_config_version",
        default=None,
        help="Rejected for FPM; the Collector resolves this from --dynamo-version.",
    )
    group.add_argument(
        "--dynamo-version",
        "--generator-dynamo-version",
        dest="generator_dynamo_version",
        default=None,
        help="Target Dynamo release used to resolve the Generator template.",
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
        "fpm_parallel_presets",
        "fpm_backend_axes",
        "fpm_warmup_iterations",
        "fpm_max_prefill_isl",
        "fpm_max_prefill_batch_size",
        "fpm_artifact_root",
        "fpm_database_root",
    ):
        if getattr(args, name, None) is not None:
            explicitly_set.append("--" + name.replace("_", "-"))
    if explicitly_set:
        raise ValueError("FPM-only arguments require --ops fpm_forward: " + ", ".join(explicitly_set))
