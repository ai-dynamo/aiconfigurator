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
    workload_distribution: str = "power_law"
    # quantization options
    nextn: int = 0  # at most mtp5
    nextn_accept_rates: list = None
    overwrite_num_layers: int = 0
    # model builder falvors
    sms: int = 20
    moe_backend: str = None  # for sglang wideep only, deepep
    attention_backend: str = "flashinfer"  # 'flashinfer' or 'fa3', for sglang wideep only
    enable_wideep: bool = False
    enable_eplb: bool = False  # Expert Parallel Load Balancing, only works during prefill


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
