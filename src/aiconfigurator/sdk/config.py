# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Union

from aiconfigurator.sdk import common


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
    cp_size: int = 1
    # CP variant ("none" / "allgather" / "ulysses" / "ring"). Set by
    # ``get_model`` from backend_name when cp_size>1; default "none". Models
    # that build CP-aware op pipelines branch on this. Models can also
    # override via ``BaseModel._resolve_cp_style``.
    cp_style: str = "none"
    workload_distribution: str = "power_law"
    # quantization options
    nextn: int = 0  # at most mtp5
    nextn_accept_rates: list = None
    overwrite_num_layers: int = 0
    # model builder falvors
    sms: int = 20
    moe_backend: str = None  # SGLang MoE backend: deepep_moe, megamoe, or None
    attention_backend: str = "flashinfer"  # 'flashinfer' or 'fa3', for sglang wideep only
    enable_wideep: bool = False
    enable_eplb: bool = False  # Expert Parallel Load Balancing
    wideep_num_slots: int = None  # EPLB num_slots, defaults to num_experts if None

    @property
    def total_gpus_per_worker(self) -> int:
        """GPUs occupied by a single worker = tp * pp * dp * cp."""
        return self.tp_size * self.pp_size * self.attention_dp_size * self.cp_size

    @property
    def attn_width(self) -> int:
        """Attention/context parallelism width = tp * cp * dp.

        Used both for MoE width matching (must equal moe_tp * moe_ep) and for
        throughput-per-GPU normalization within a single attention-side worker.
        """
        return self.tp_size * self.cp_size * self.attention_dp_size

    def resolve_moe_parallelism(self) -> tuple[int, int]:
        """Resolve and validate MoE parallelism dimensions in-place.

        For MoE models, the attention/context/data-parallel width must match
        the expert width:
        ``tp_size * cp_size * attention_dp_size == moe_tp_size * moe_ep_size``. If one
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

        attn_width = self.attn_width
        moe_tp_size = self.moe_tp_size
        moe_ep_size = self.moe_ep_size
        if moe_tp_size is None and moe_ep_size is None:
            raise ValueError("At least one of moe_tp_size or moe_ep_size must be set for MoE models.")
        elif moe_tp_size is None:
            _validate_positive("moe_ep_size", moe_ep_size)
            if attn_width % moe_ep_size != 0:
                raise ValueError(
                    f"Cannot infer moe_tp_size: tp_size({self.tp_size}) * cp_size({self.cp_size}) * "
                    f"attention_dp_size({self.attention_dp_size}) = {attn_width} is not "
                    f"divisible by moe_ep_size({moe_ep_size})."
                )
            moe_tp_size = attn_width // moe_ep_size
        elif moe_ep_size is None:
            _validate_positive("moe_tp_size", moe_tp_size)
            if attn_width % moe_tp_size != 0:
                raise ValueError(
                    f"Cannot infer moe_ep_size: tp_size({self.tp_size}) * cp_size({self.cp_size}) * "
                    f"attention_dp_size({self.attention_dp_size}) = {attn_width} is not "
                    f"divisible by moe_tp_size({moe_tp_size})."
                )
            moe_ep_size = attn_width // moe_tp_size

        _validate_positive("moe_tp_size", moe_tp_size)
        _validate_positive("moe_ep_size", moe_ep_size)

        # TODO: enforce moe_tp_size == 1 when enable_wideep is set.
        moe_width = moe_tp_size * moe_ep_size
        if attn_width != moe_width:
            raise ValueError(
                f"Parallelism width mismatch: tp_size({self.tp_size}) * cp_size({self.cp_size}) * "
                f"attention_dp_size({self.attention_dp_size}) = {attn_width}, but "
                f"moe_tp_size({moe_tp_size}) * moe_ep_size({moe_ep_size}) = "
                f"{moe_width}. These must be equal."
            )

        self.moe_tp_size = moe_tp_size
        self.moe_ep_size = moe_ep_size
        return self.moe_tp_size, self.moe_ep_size


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
