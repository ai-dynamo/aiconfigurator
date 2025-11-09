# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from collections import defaultdict

from aiconfigurator.sdk import common
from aiconfigurator.sdk.backends.base_backend import BaseBackend
from aiconfigurator.sdk.config import RuntimeConfig
from aiconfigurator.sdk.inference_summary import InferenceSummary
from aiconfigurator.sdk.models import BaseModel
from aiconfigurator.sdk.perf_database import PerfDatabase

logger = logging.getLogger(__name__)

MAX_NORMAL_CTX_TOKENS = 8192
MAX_CTX_TOKENS_MULTIPLE_OF_ISL = 2
MAX_CTX_TOKENS_SEARCH_STEPS = 8  # for ctx stride large for faster sweeping


class VLLMBackend(BaseBackend):
    """
    VLLM backend.
    """

    def __init__(
        self,
    ):
        super().__init__()
        self._agg_cache = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))
        self.name = common.BackendName.vllm

    def run_agg(
        self, model: BaseModel, database: PerfDatabase, runtime_config: RuntimeConfig, **kwargs
    ) -> InferenceSummary:
        """
        Run the agg inference for VLLM backend.
        """
        from aiconfigurator.sdk.backends.trtllm_backend import TRTLLMBackend

        return TRTLLMBackend().run_agg(model, database, runtime_config, **kwargs)

    def find_best_agg_result_under_constraints(
        self, model: BaseModel, database: PerfDatabase, runtime_config: RuntimeConfig, **kwargs
    ) -> InferenceSummary:
        """
        Find the best agg result under constraints for VLLM backend.

        Args:
            model: the model to be tested
            database: the database to be tested
            runtime_config: the runtime configuration
            top_k: the number of best results to return
            max_batch_size: the maximum batch size to test
            ctx_stride: the stride of ctx tokens to test, it will impact the time to run the test.
            enable_chunked_prefill: whether to enable chunked prefill, it will impact the time to
                run the test while have little impact on the result. Default off.

        Returns:
            A summary of the best agg result under constraints.
        """
        from aiconfigurator.sdk.backends.trtllm_backend import TRTLLMBackend

        return TRTLLMBackend().find_best_agg_result_under_constraints(model, database, runtime_config, **kwargs)

    def _get_memory_usage(
        self,
        model: BaseModel,
        database: PerfDatabase,
        batch_size: int,
        beam_width: int,
        isl: int,
        osl: int,
        num_tokens: int = 0,
    ) -> dict[str, float]:
        # TODO
        from aiconfigurator.sdk.backends.trtllm_backend import TRTLLMBackend

        return TRTLLMBackend()._get_memory_usage(model, database, batch_size, beam_width, isl, osl, num_tokens)
