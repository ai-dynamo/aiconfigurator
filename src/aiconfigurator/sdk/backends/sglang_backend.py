# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiconfigurator.sdk.backends.base_backend import BaseBackend
from aiconfigurator.sdk import common
from aiconfigurator.sdk.config import RuntimeConfig
from aiconfigurator.sdk.inference_summary import InferenceSummary
import numpy as np
import pandas as pd
from collections import defaultdict
from aiconfigurator.sdk.models import BaseModel
from aiconfigurator.sdk.perf_database import PerfDatabase

class SGLANGBackend(BaseBackend):
    """
    SGLANG backend.
    """
    def __init__(self):
        super().__init__()

    def run_ifb(self, 
                model: BaseModel, 
                database: PerfDatabase, 
                runtime_config: RuntimeConfig, 
                **kwargs) -> InferenceSummary:
        pass

    def find_best_ifb_result_under_constraints(self, 
                                               model: BaseModel, 
                                               database: PerfDatabase, 
                                               runtime_config: RuntimeConfig, 
                                               **kwargs) -> InferenceSummary:
        pass
    
    def _get_memory_usage(self, 
                          model: BaseModel, 
                          database: PerfDatabase, 
                          batch_size: int, 
                          beam_width: int, 
                          isl: int, 
                          osl: int, num_tokens: int = 0) -> dict[str, float]:
        pass

