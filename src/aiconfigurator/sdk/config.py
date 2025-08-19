# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    gemm_quant_mode: common.GEMMQuantMode = common.GEMMQuantMode.float16
    moe_quant_mode: common.MoEQuantMode = common.MoEQuantMode.float16
    kvcache_quant_mode: common.KVCacheQuantMode = common.KVCacheQuantMode.float16
    fmha_quant_mode: common.FMHAQuantMode = common.FMHAQuantMode.float16
    comm_quant_mode: common.CommQuantMode = common.CommQuantMode.half
    moe_tp_size: int = None
    moe_ep_size: int = None
    attention_dp_size: int = 1
    workload_distribution: str = "power_law_0.5"
    nextn: int = 0 # at most mtp5
    nextn_accept_rates: list = None
    overwrite_num_layers: int = 0


@dataclass
class RuntimeConfig:
    """
    Runtime configuration.
    """
    batch_size: int = None
    beam_width: int = 1
    isl: int = None
    osl: int = None
    ttft: float = None
    tpot: Union[float, list] = None
