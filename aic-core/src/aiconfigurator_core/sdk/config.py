# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Union

from aiconfigurator_core.sdk import common


@dataclass
class ModelConfig:
    """
    Model configuration.
    """

    tp_size: int = 1
    pp_size: int = 1
    gemm_quant_mode: common.GEMMQuantMode | None = None
    moe_quant_mode: common.MoEQuantMode | None = None
    kvcache_quant_mode: common.KVCacheQuantMode | None = None
    fmha_quant_mode: common.FMHAQuantMode | None = None
    comm_quant_mode: common.CommQuantMode | None = common.CommQuantMode.half
    moe_tp_size: int = None
    moe_ep_size: int = None
    attention_dp_size: int = 1
    cp_size: int = 1  # context parallelism: splits sequence tokens (not heads); folds into attn_width = tp*cp*dp
    # CP variant ("none" / "allgather" / "ulysses" / "ring"). Set by get_model
    # from backend_name when cp_size > 1; default "none". Dense models branch on
    # this in their op pipeline; GLM-5 DSA ignores it (handled in ContextDSAModule).
    cp_style: str = "none"
    workload_distribution: str = "power_law"
    # quantization options
    # MTP speculative decoding: draft length (compute/verification cost only).
    # Accepted-token progress belongs to the upper prediction/simulation layer.
    nextn: int = 0
    overwrite_num_layers: int = 0
    # model builder falvors
    sms: int = 20
    moe_backend: str = None  # SGLang MoE backend: deepep_moe, megamoe, or None
    attention_backend: str = "flashinfer"  # 'flashinfer' or 'fa3', for sglang wideep only
    enable_wideep: bool = False
    enable_eplb: bool = False  # Expert Parallel Load Balancing
    wideep_num_slots: int = None  # EPLB num_slots, defaults to num_experts if None

    def resolve_moe_parallelism(self) -> tuple[int, int]:
        """Resolve and validate MoE parallelism dimensions in-place.

        For MoE models, the attention width must match the expert width:
        ``tp_size * attention_dp_size == moe_tp_size * moe_ep_size``. If one
        MoE dimension is missing, infer it from the other. If both are missing,
        raise an error so callers do not silently get an MoE layout they did
        not request.
        """

        def _validate_positive(name: str, value: int) -> None:
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")

        _validate_positive("tp_size", self.tp_size)
        _validate_positive("attention_dp_size", self.attention_dp_size)
        _validate_positive("cp_size", self.cp_size)
        # Context parallelism is an INDEPENDENT attention dimension: it splits
        # tokens (sequence), orthogonal to tp_size (which splits heads). It
        # contributes to the total width that the MoE must match:
        # tp_size * attention_dp_size * cp_size == moe_tp * moe_ep.

        attn_width = self.tp_size * self.attention_dp_size * self.cp_size
        moe_tp_size = self.moe_tp_size
        moe_ep_size = self.moe_ep_size
        if moe_tp_size is None and moe_ep_size is None:
            raise ValueError("At least one of moe_tp_size or moe_ep_size must be set for MoE models.")
        elif moe_tp_size is None:
            _validate_positive("moe_ep_size", moe_ep_size)
            if attn_width % moe_ep_size != 0:
                raise ValueError(
                    f"Cannot infer moe_tp_size: tp_size({self.tp_size}) * "
                    f"attention_dp_size({self.attention_dp_size}) * cp_size({self.cp_size}) = {attn_width} is not "
                    f"divisible by moe_ep_size({moe_ep_size})."
                )
            moe_tp_size = attn_width // moe_ep_size
        elif moe_ep_size is None:
            _validate_positive("moe_tp_size", moe_tp_size)
            if attn_width % moe_tp_size != 0:
                raise ValueError(
                    f"Cannot infer moe_ep_size: tp_size({self.tp_size}) * "
                    f"attention_dp_size({self.attention_dp_size}) * cp_size({self.cp_size}) = {attn_width} is not "
                    f"divisible by moe_tp_size({moe_tp_size})."
                )
            moe_ep_size = attn_width // moe_tp_size

        _validate_positive("moe_tp_size", moe_tp_size)
        _validate_positive("moe_ep_size", moe_ep_size)

        # TODO: enforce moe_tp_size == 1 when enable_wideep is set.
        moe_width = moe_tp_size * moe_ep_size
        if attn_width != moe_width:
            raise ValueError(
                f"Parallelism width mismatch: tp_size({self.tp_size}) * "
                f"attention_dp_size({self.attention_dp_size}) * cp_size({self.cp_size}) = {attn_width}, but "
                f"moe_tp_size({moe_tp_size}) * moe_ep_size({moe_ep_size}) = "
                f"{moe_width}. These must be equal."
            )

        self.moe_tp_size = moe_tp_size
        self.moe_ep_size = moe_ep_size
        return self.moe_tp_size, self.moe_ep_size

    @property
    def total_gpus_per_worker(self) -> int:
        """GPUs occupied by a single worker = tp * pp * dp * cp.

        CP is an independent sequence-sharding dimension, so it multiplies the
        worker's GPU count just like tp/pp/dp. Used by the enumeration and the
        throughput-per-GPU normalization so cp configs are sized correctly.
        """
        return self.tp_size * self.pp_size * self.attention_dp_size * self.cp_size

    @property
    def attn_width(self) -> int:
        """Attention/context-parallel width = tp * cp * dp.

        Must equal moe_tp * moe_ep for MoE models (the width the __post_init__
        validation enforces) and is the attention-side throughput normalizer.
        """
        return self.tp_size * self.cp_size * self.attention_dp_size


@dataclass
class RuntimeConfig:
    """
    Runtime configuration.
    """

    batch_size: int = None
    beam_width: int = 1
    isl: int = None
    osl: int = None
    prefix: int = 0  # prefix len of isl
    ttft: float = None
    tpot: Union[float, list] = None
    request_latency: float = None  # it works together with ttft. 1. <= req_lat 2. <= req_lat and <= ttft
    seq_imbalance_correction_scale: float = 1.0
    # Separate correction scale for generation/decoding stage (do NOT reuse ctx scale).
    gen_seq_imbalance_correction_scale: float = 1.0
    # Optional experimental static-latency backend. "python" preserves existing behavior;
    # "rust" routes static step estimates through the Rust FPM estimator.
    engine_step_backend: str | None = None
    image_height: int = 0
    image_width: int = 0
    num_images_per_request: int = 1
    num_image_tokens: int = 0  # override: ViT output tokens per image; ignored when image_height/width are set


@dataclass
class AFDConfig:
    """Configuration for Attention-FFN Disaggregated (AFD) serving mode.

    In AFD mode, Attention ops run on A-Workers while FFN/MoE ops run on
    F-Workers.  The two pools communicate activation tensors every layer.

    The deployment topology is described by the triple
    ``(n_a_nodes, n_f_nodes, gpus_per_node)``.  The A/F rank mapping is
    static and determined at deployment time — AIC models this as a
    configuration input, not a dynamic scheduling problem.

    Single-source-of-truth invariants
    ---------------------------------
    ``gpus_per_node`` is **not** a user-tunable knob: once the target
    system is picked, this is a hardware fact (h100_sxm/h200_sxm = 8,
    gb200 = 4, …). Callers must inject the value from
    ``system_spec["node"]["num_gpus_per_node"]`` — the default sentinel
    of ``0`` makes "I forgot to inject" loud rather than silently
    mis-shaping ``n_f_workers`` / ``AFDTransfer`` BW selection. A
    yaml/CLI "what-if" override is allowed but must be done deliberately
    (the constructing layer is responsible for cross-checking against
    the spec).

    ``tp_f`` is the **total GPU count of one F-replica** (i.e. the
    ``ModelConfig.tp_size`` used to shape the F-Worker model). Under the
    Phase 1 assumption F-side DP=1, that's exactly ``n_f_workers``, so
    ``tp_f`` is derived — exposing it as a user input opens a foot-gun
    (e.g. ``n_f_nodes=2, gpus_per_node=8, tp_f=8`` silently implies
    F-DP=2). The sentinel ``0`` requests automatic derivation; passing
    a non-zero value triggers an invariant check.

    Derived quantities (computed in ``__post_init__``):

    * ``n_a_workers`` — A-side DP count = ``n_a_nodes * gpus_per_node // tp_a``
    * ``n_f_workers`` — total F-side GPUs = ``n_f_nodes * gpus_per_node``
    * ``tp_f``        — set to ``n_f_workers`` (Phase 1: F-DP = 1)

    AFD is orthogonal to Prefill/Decode (P/D) disaggregation — it can be
    applied to the Decode phase (default, where batch concentration benefits
    are strongest), to the Prefill phase, or to both phases simultaneously.
    """

    # -- Topology inputs --
    n_a_nodes: int = 1
    n_f_nodes: int = 1
    # 0 = sentinel "inject from system_spec"; see class docstring.
    gpus_per_node: int = 0

    # -- Per-worker parallelism --
    tp_a: int = 1
    # 0 = sentinel "derive from n_f_workers" (Phase 1: F-DP = 1).
    # A non-zero value is treated as a what-if override and validated
    # against the F-DP=1 invariant in ``__post_init__``.
    tp_f: int = 0
    f_moe_ep_size: int = 1

    # -- Batch and pipeline --
    # Total in-flight batch size per A-Worker.  AFDInferenceSession
    # derives the per-microbatch batch as ceil(a_batch_size / num_microbatches)
    # for compute and transfer latency queries.
    a_batch_size: int = 128
    num_microbatches: int = 3
    pipeline_model: str = "optimistic"  # "optimistic" (K=3) or "conservative" (K=2)
    comm_overhead_factor: float = 1.0
    # Which phase(s) AFD should be applied to.
    # "decode" (default) mirrors existing behavior; "prefill" applies to
    # the context phase; "both" produces an aggregated estimate combining
    # prefill (TTFT) and decode (TPOT).
    phase: str = "decode"  # "prefill" | "decode" | "both"
    # Whether this AFD pool runs together with a separate static
    # (non-AFD) pool covering the *other* phase. Concretely, in CLI
    # ``estimate`` mode:
    #
    #   * ``phase`` ∈ {"prefill", "decode"} + ``combined_with_pd=True``
    #     → the AFD path estimates only its own phase and the CLI
    #     orchestration layer (``cli/api._combine_afd_static_estimate_results``)
    #     runs a standard static estimate for the other phase, then
    #     merges TTFT/TPOT, throughput (rate-matched on min seq/s), GPU
    #     budget (``afd_gpus + static_gpus``) and per-phase impl labels
    #     into a single ``EstimateResult``.
    #   * ``phase`` ∈ {"prefill", "decode"} + ``combined_with_pd=False``
    #     → the CLI returns the AFD-only estimate for the chosen phase
    #     (the other phase is left unmodeled; user is on their own to
    #     size it). Use this when you only care about that single phase.
    #   * ``phase == "both"`` → AFD covers both phases internally;
    #     ``combined_with_pd`` must be ``False`` (the two are
    #     mutually exclusive and ``__post_init__`` enforces this).
    #
    # Default is ``True`` because the typical AFD deployment pairs an
    # AFD decode pool with a regular prefill pool (or vice versa); the
    # combined estimate is the number a sizing exercise actually needs.
    #
    # TODO(afd, Phase-2): extend pareto_analysis to merge the AFD
    # frontier with the agg / disagg frontiers so end-to-end Pareto
    # evaluation across mixed AFD + P/D deployments is possible. Today
    # the CLI single-point combine path is in place, but the Pareto
    # sweep does not yet enumerate AFD-combined-with-PD points.
    combined_with_pd: bool = True
    # Boundary-op assignment: ``add_norm_2`` and ``logits_gemm`` sit at
    # the natural Attention/FFN boundary and can be assigned to either
    # pool.  Defaults to A-Worker, but exposed here as a configurable
    # knob.  When False the boundary ops are appended to the F-Worker
    # partition instead.
    boundary_on_attn: bool = True

    # -- Derived (set in __post_init__) --
    n_a_workers: int = field(init=False)
    n_f_workers: int = field(init=False)

    def __post_init__(self) -> None:
        valid_phases = ("prefill", "decode", "both")
        valid_pipeline_models = ("optimistic", "conservative", "serial")
        if self.phase not in valid_phases:
            raise ValueError(f"phase must be one of {valid_phases}, got {self.phase!r}.")
        if self.pipeline_model not in valid_pipeline_models:
            raise ValueError(f"pipeline_model must be one of {valid_pipeline_models}, got {self.pipeline_model!r}.")
        if self.n_a_nodes < 1 or self.n_f_nodes < 1:
            raise ValueError(f"n_a_nodes ({self.n_a_nodes}) and n_f_nodes ({self.n_f_nodes}) must both be >= 1.")
        if self.a_batch_size < 1:
            raise ValueError(f"a_batch_size ({self.a_batch_size}) must be >= 1.")
        if self.num_microbatches is not None and self.num_microbatches < 1:
            raise ValueError(f"num_microbatches ({self.num_microbatches}) must be >= 1.")
        if self.comm_overhead_factor <= 0:
            raise ValueError(f"comm_overhead_factor ({self.comm_overhead_factor}) must be > 0.")
        if self.f_moe_ep_size < 1:
            raise ValueError(f"f_moe_ep_size ({self.f_moe_ep_size}) must be >= 1.")
        if self.gpus_per_node < 1:
            raise ValueError(
                f"gpus_per_node ({self.gpus_per_node}) must be >= 1. "
                "AFDConfig must be constructed with gpus_per_node injected "
                "from system_spec['node']['num_gpus_per_node']; do not rely "
                "on the default sentinel."
            )
        if self.tp_a < 1 or self.gpus_per_node % self.tp_a != 0:
            raise ValueError(f"tp_a ({self.tp_a}) must be a positive divisor of gpus_per_node ({self.gpus_per_node}).")
        if self.phase == "both" and self.combined_with_pd:
            raise ValueError(
                "combined_with_pd=True is incompatible with phase='both': "
                "'both' means AFD covers prefill+decode internally, so there "
                "is no separate static pool to combine with. Set "
                "combined_with_pd=False, or pick phase in {'prefill','decode'}."
            )
        self.n_a_workers = self.n_a_nodes * self.gpus_per_node // self.tp_a
        self.n_f_workers = self.n_f_nodes * self.gpus_per_node

        # Phase 1: F-side DP = 1, so tp_f (= ModelConfig.tp_size of one
        # F-replica) is exactly n_f_workers. A caller-supplied non-zero
        # tp_f is treated as a what-if override and must match the
        # invariant; otherwise it silently implies F-DP > 1.
        derived_tp_f = self.n_f_workers
        if self.tp_f and self.tp_f != derived_tp_f:
            raise ValueError(
                f"tp_f ({self.tp_f}) is derived under the Phase 1 F-DP=1 "
                f"assumption as n_f_nodes * gpus_per_node = {derived_tp_f}; "
                "an explicit different value would silently imply F-DP > 1. "
                "Remove the override, or change n_f_nodes / gpus_per_node."
            )
        self.tp_f = derived_tp_f
