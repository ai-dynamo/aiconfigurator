# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import functools
import logging
import warnings
from collections import defaultdict, namedtuple

import pandas as pd

from aiconfigurator.sdk import common, config, models, perf_database
from aiconfigurator.sdk.backends.base_backend import BaseBackend
from aiconfigurator.sdk.errors import NoFeasibleConfigError
from aiconfigurator.sdk.inference_summary import InferenceSummary
from aiconfigurator.sdk.picking import (
    _AUTOSCALE_TTFT_CORRECTION_FACTOR,
    _RATE_MATCHING_DECODE_DEGRADATION_FACTOR,
    _RATE_MATCHING_PREFILL_DEGRADATION_FACTOR,
    _build_disagg_summary_dict,
)
from aiconfigurator.sdk.speculative import SpeculativeDecodingProfile
from aiconfigurator.sdk.step_estimate import MixedStepInput, StepEstimate
from aiconfigurator.sdk.utils import enumerate_ttft_tpot_constraints, get_model_config_from_model_path

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)


class InferenceSession:
    """
    InferenceSession holds the model and database to run inference loop

    Attributes:
        model (models.BaseModel): the model to run inference
        database (perf_database.PerfDatabase): the database to run inference
        backend (backend.Backend): the backend to run inference

    Methods:
        run_static (static, static_ctx, static_gen): to support static batching and disagg,
            returns details of a static run
        run_mixed: estimate one mixed prefill/decode engine iteration
        run_agg (static, static_ctx, static_gen): run agg inference, returns summary of the
            perf result with given agg config and runtime config (concurrency)
        find_best_agg_result_under_constraints (static, static_ctx, static_gen):
            find the best agg result under constraints, returns summary
            which contains all the possible agg config and perf that matchs SLA.
    """

    def __init__(self, model: models.BaseModel, database: perf_database.PerfDatabase, backend: BaseBackend) -> None:
        """
        Initialize the InferenceSession
        """
        self._model = model
        self._database = database
        self._backend = backend

    def run_static(
        self,
        runtime_config: config.RuntimeConfig,
        mode: str,
        stride: int = 32,
        latency_correction_scale: float = 1.0,
    ) -> InferenceSummary:
        """
        Run static inference

        Args:
            runtime_config (RuntimeConfig): the runtime config
            mode (str): the mode to run inference, static, static_ctx, static_gen
            stride (int): the stride is used to accelerate the estimation, for a give osl,
                will only computes the i, i+stride, i+2*stride, ... step, default is 32.

        Returns:
            InferenceSummary: the summary of the inference result
        """
        return self._backend.run_static(
            self._model,
            self._database,
            runtime_config,
            mode,
            stride,
            latency_correction_scale,
        )

    def run_static_latency_only(
        self,
        runtime_config: config.RuntimeConfig,
        mode: str,
        stride: int = 32,
        latency_correction_scale: float = 1.0,
    ) -> float:
        """
        Run static inference and return only scalar latency in milliseconds.

        Args:
            runtime_config (RuntimeConfig): the runtime config
            mode (str): the mode to run inference, static, static_ctx, static_gen
            stride (int): the stride is used to accelerate the estimation, for a give osl,
                will only computes the i, i+stride, i+2*stride, ... step, default is 32.

        Returns:
            float: the total latency in milliseconds
        """
        return self._backend.run_static_latency_only(
            self._model, self._database, runtime_config, mode, stride, latency_correction_scale
        )

    def run_mixed(
        self,
        runtime_config: config.RuntimeConfig,
        step: MixedStepInput,
    ) -> StepEstimate:
        """Estimate one mixed prefill/decode engine iteration."""
        return self._backend.run_mixed(self._model, self._database, runtime_config, step)

    def run_agg(self, runtime_config: config.RuntimeConfig, **kwargs) -> InferenceSummary:
        """
        Run agg inference

        Args:
            runtime_config (RuntimeConfig): the runtime config
            **kwargs: other arguments to run agg, depends on the backend specific design

        Returns:
            InferenceSummary: the summary of the inference result
        """
        return self._backend.run_agg(self._model, self._database, runtime_config, **kwargs)

    # Optimization
    def find_best_agg_result_under_constraints(
        self, runtime_config: config.RuntimeConfig, **kwargs
    ) -> InferenceSummary:
        """
        Find the best agg result under constraints

        Args:
            runtime_config (RuntimeConfig): the runtime config
            **kwargs: other arguments to find the best agg result under constraints,
                depends on the backend specific design

        Returns:
            InferenceSummary: the summary of the inference result, contains all the possible
                agg config and perf that matchs SLA.
        """
        return self._backend.find_best_agg_result_under_constraints(
            self._model, self._database, runtime_config, **kwargs
        )


DECODE_FILTER_RATIO_MIN = 0.0
DECODE_FILTER_RATIO_MAX = 1.0
MAX_DECODE_WORKERS_PER_CATEGORY = 16
MAX_PREFILL_WORKERS = 32
MAX_NUM_DECODE_WORKER_CANDIDATES = 64
MAX_NUM_PREFILL_WORKER_CANDIDATES = 32


class DisaggInferenceSession:
    """
    Disaggregated inference session
    Run prefill and generation separately, with different models (parallel and precision config can
    be different) and databases
    0. init func only takes database and backend, model is passed in run_disagg
    1. run_disagg, given model, database and backend, given everything fixed ((max)batchsize and
       num_workers) , return the perf result of the system
    2. find_best_disagg_result_under_constraints, given database and backend, sweep batchsize and
       model parallel to match SLA, sweep workers to get best system perf/gpu if allowed.
       Return config (parallel, batchsize and num_workers) and perf.
    3. TODO, should consider kvcache model in future
    Disagg is more like a post processing step to do rate matching, that's why it's a
    DiaggInferenceSession instread of using InferenceSession.

    Attributes:
        prefill_database (perf_database.PerfDatabase): the database to run prefill
        prefill_backend (backend.Backend): the backend to run prefill
        decode_database (perf_database.PerfDatabase): the database to run decode
        decode_backend (backend.Backend): the backend to run decode

    Methods:
        run_disagg (model_path, runtime_config, prefill_model_config, prefill_batch_size,
                    prefill_num_worker, decode_model_config, decode_batch_size,
                    decode_num_worker)
            run disagg with given prefill/decode worker info
        find_best_disagg_result_under_constraints (model_path,runtime_config, prefill_model_config,
                    prefill_parallel_config_list, prefill_max_num_tokens, prefill_num_worker_list,
                    decode_model_config, decode_parallel_config_list, decode_max_num_tokens,
                    decode_num_worker_list, num_gpu_list)
            find the best disagg result under constraints
        set_latency_correction_scales (prefill_latency_correction_scale,
                                       decode_latency_correction_scale):
            set the correction scales for better alignment with real system
    """

    def __init__(
        self,
        prefill_database: perf_database.PerfDatabase,
        prefill_backend: BaseBackend,
        decode_database: perf_database.PerfDatabase,
        decode_backend: BaseBackend,
        encoder_database: perf_database.PerfDatabase | None = None,
        encoder_backend: BaseBackend | None = None,
    ) -> None:
        """
        Initialize the DisaggInferenceSession
        """
        self._prefill_database = prefill_database
        self._prefill_backend = prefill_backend
        self._decode_database = decode_database
        self._decode_backend = decode_backend
        self._encoder_database = encoder_database
        self._encoder_backend = encoder_backend

        # allow user to set correction scales for better alignment with real system
        # now the corection scales are used to correct the latency, not throughput,
        # corrected latency = latency * correction_scale
        self._prefill_latency_correction_scale = 1.0
        self._decode_latency_correction_scale = 1.0
        self._encoder_latency_correction_scale = 1.0

        self._rate_matching_prefill_degradation_factor = _RATE_MATCHING_PREFILL_DEGRADATION_FACTOR
        self._rate_matching_decode_degradation_factor = _RATE_MATCHING_DECODE_DEGRADATION_FACTOR

    def set_latency_correction_scales(
        self,
        prefill_latency_correction_scale: float,
        decode_latency_correction_scale: float,
        encoder_latency_correction_scale: float = 1.0,
    ):
        """
        Set the correction scales for better alignment with real system
        """
        self._prefill_latency_correction_scale = prefill_latency_correction_scale
        self._decode_latency_correction_scale = decode_latency_correction_scale
        self._encoder_latency_correction_scale = encoder_latency_correction_scale

    def set_rate_matching_degradation_factors(
        self,
        prefill_degradation_factor: float = _RATE_MATCHING_PREFILL_DEGRADATION_FACTOR,
        decode_degradation_factor: float = _RATE_MATCHING_DECODE_DEGRADATION_FACTOR,
    ):
        """
        Set the degradation factors used during rate matching between prefill and decode workers.

        Args:
            prefill_degradation_factor: Multiplicative factor applied to prefill throughput
                to account for pipeline bubbles (default 0.9).
            decode_degradation_factor: Multiplicative factor applied to decode throughput
                to account for batch-size under-saturation (default 0.92).
        """
        self._rate_matching_prefill_degradation_factor = prefill_degradation_factor
        self._rate_matching_decode_degradation_factor = decode_degradation_factor

    def _get_disagg_summary_df(
        self,
        prefill_summary_df: pd.DataFrame,
        prefill_num_worker: int,
        decode_summary_df: pd.DataFrame,
        decode_num_worker: int,
    ) -> pd.DataFrame:
        """
        Get the disagg summary df based on prefill and decode summary df
        """
        prefill_dict = prefill_summary_df.iloc[0].to_dict()
        prefill_dict["ttft"] = prefill_dict["ttft"] * _AUTOSCALE_TTFT_CORRECTION_FACTOR
        decode_dict = decode_summary_df.iloc[0].to_dict()

        summary_dict = _build_disagg_summary_dict(
            prefill_dict,
            prefill_num_worker,
            decode_dict,
            decode_num_worker,
            prefill_degradation_factor=self._rate_matching_prefill_degradation_factor,
            decode_degradation_factor=self._rate_matching_decode_degradation_factor,
        )
        return pd.DataFrame([summary_dict], columns=common.ColumnsDisagg).round(3)

    def run_disagg(
        self,
        model_path: str,
        runtime_config: config.RuntimeConfig,
        prefill_model_config: config.ModelConfig,
        prefill_batch_size: int,
        prefill_num_worker: int,
        decode_model_config: config.ModelConfig,
        decode_batch_size: int,
        decode_num_worker: int,
        speculative_profile: SpeculativeDecodingProfile | None = None,
    ) -> InferenceSummary:
        """
        Run disagg with given prefill/decode worker info

        Args:
            model_path (str): the model name
            runtime_config (RuntimeConfig): the runtime config
            prefill_model_config (ModelConfig): the prefill model config
            prefill_batch_size (int): the prefill batch size
            prefill_num_worker (int): the number of prefill workers
            decode_model_config (ModelConfig): the decode model config
            decode_batch_size (int): the decode batch size
            decode_num_worker (int): the number of decode workers
            speculative_profile: Optional accepted-token progress assumption.
                Projects decode metrics before prefill/decode rate matching.

        Returns:
            InferenceSummary: the summary of the inference result
        """
        prefill_model = models.get_model(model_path, prefill_model_config, self._prefill_backend.name.value)
        decode_model = models.get_model(model_path, decode_model_config, self._decode_backend.name.value)
        prefill_sess = InferenceSession(
            model=prefill_model, database=self._prefill_database, backend=self._prefill_backend
        )
        decode_sess = InferenceSession(model=decode_model, database=self._decode_database, backend=self._decode_backend)

        prefill_runtime_config = copy.deepcopy(runtime_config)
        prefill_runtime_config.batch_size = prefill_batch_size
        prefill_summary = prefill_sess.run_static(
            mode="static_ctx",
            runtime_config=prefill_runtime_config,
            latency_correction_scale=self._prefill_latency_correction_scale,
        )
        decode_runtime_config = copy.deepcopy(runtime_config)
        decode_runtime_config.batch_size = decode_batch_size

        decode_summary = decode_sess.run_static(
            mode="static_gen",
            runtime_config=decode_runtime_config,
            latency_correction_scale=self._decode_latency_correction_scale,
        )
        if speculative_profile is not None:
            decode_summary = speculative_profile.project_summary(decode_summary, role="decode")
        disagg_summary_df = self._get_disagg_summary_df(
            prefill_summary.get_summary_df(),
            prefill_num_worker,
            decode_summary.get_summary_df(),
            decode_num_worker,
        )

        disagg_summary = InferenceSummary(runtime_config=runtime_config)

        prefill_oom = prefill_summary.check_oom()
        decode_oom = decode_summary.check_oom()
        if prefill_oom or decode_oom:
            disagg_summary.set_oom(True)

        disagg_summary.set_summary_df(disagg_summary_df)

        # Carry per-op latency breakdowns from prefill/decode static runs
        per_ops_data = {}
        per_ops_source = {}
        prefill_encoder_latency = prefill_summary.get_encoder_latency_dict()
        if prefill_encoder_latency:
            per_ops_data["encoder"] = dict(prefill_encoder_latency)
            disagg_summary.set_encoder_latency_dict(dict(prefill_encoder_latency))
            disagg_summary.set_encoder_energy_wms_dict(dict(prefill_summary.get_encoder_energy_wms_dict()))
            disagg_summary.set_encoder_power_avg(prefill_summary.get_encoder_power_avg())
            encoder_memory = prefill_summary.get_encoder_memory()
            if encoder_memory:
                disagg_summary.set_encoder_memory(dict(encoder_memory))
        prefill_encoder_source = prefill_summary.get_encoder_source_dict()
        if prefill_encoder_source:
            encoder_source = dict(prefill_encoder_source)
            disagg_summary.set_encoder_source_dict(encoder_source)
            per_ops_source["encoder"] = encoder_source
        prefill_ctx_latency = prefill_summary.get_context_latency_dict()
        if prefill_ctx_latency:
            per_ops_data["prefill"] = dict(prefill_ctx_latency)
        prefill_ctx_source = prefill_summary.get_context_source_dict()
        if prefill_ctx_source:
            per_ops_source["prefill"] = dict(prefill_ctx_source)
        decode_gen_latency = decode_summary.get_generation_latency_dict()
        if decode_gen_latency:
            per_ops_data["decode"] = dict(decode_gen_latency)
        decode_gen_source = decode_summary.get_generation_source_dict()
        if decode_gen_source:
            per_ops_source["decode"] = dict(decode_gen_source)
        if per_ops_data:
            disagg_summary.set_per_ops_data(per_ops_data)
        if per_ops_source:
            disagg_summary.set_per_ops_source(per_ops_source)

        return disagg_summary

    def get_worker_candidates(
        self,
        model_path: str,
        model_config: config.ModelConfig,
        parallel_config_list: list[tuple[int, int, int, int, int, int]],
        b_list: list[int] | range,
        runtime_config: config.RuntimeConfig,
        mode: str,
        latency_correction_scale: float = 1.0,
    ) -> pd.DataFrame:
        """Get all worker candidates for a given search space.

        It enumerates all (parallel_config, batch_size) combinations,
        runs static inference, and returns a DataFrame with columns from
        :data:`common.ColumnsStatic`.

        Args:
            model_path: HuggingFace model ID or local path.
            model_config: Model configuration (quant modes etc.).
            parallel_config_list: List of (tp, pp, dp, moe_tp, moe_ep, cp) tuples.
            b_list: Batch sizes to sweep.
            runtime_config: Runtime config (isl, osl, etc.).
            mode: ``"static_ctx"`` for prefill or ``"static_gen"`` for decode.
            latency_correction_scale: Multiplicative correction applied to
                latencies (default 1.0).

        Returns:
            DataFrame with one row per (parallel_config, batch_size) that fits
            in memory.

        Raises:
            RuntimeError: If no valid results are found for any config.
        """
        summary_df = pd.DataFrame(columns=common.ColumnsStatic)
        exceptions: list[Exception] = []
        all_configs_oom = True

        for parallel_config in parallel_config_list:
            # 6-tuple (tp, pp, dp, moe_tp, moe_ep, cp); tolerate legacy 5-tuples.
            tp_size, pp_size, dp_size, moe_tp_size, moe_ep_size, cp_size = parallel_config
            logger.debug(
                "Getting candidate workers with parallel config: tp=%d, pp=%d, dp=%d, moe_tp=%d, moe_ep=%d, cp=%d",
                tp_size,
                pp_size,
                dp_size,
                moe_tp_size,
                moe_ep_size,
                cp_size,
            )

            try:
                overwritten_model_config = copy.deepcopy(model_config)
                overwritten_model_config.pp_size = pp_size
                overwritten_model_config.tp_size = tp_size
                overwritten_model_config.moe_tp_size = moe_tp_size
                overwritten_model_config.moe_ep_size = moe_ep_size
                overwritten_model_config.attention_dp_size = dp_size
                overwritten_model_config.cp_size = cp_size
                model = models.get_model(
                    model_path=model_path,
                    model_config=overwritten_model_config,
                    backend_name=self._prefill_backend.name.value,
                )
                if mode == "static_ctx":
                    sess = InferenceSession(
                        model=model,
                        database=self._prefill_database,
                        backend=self._prefill_backend,
                    )
                else:
                    sess = InferenceSession(
                        model=model,
                        database=self._decode_database,
                        backend=self._decode_backend,
                    )

                for b in b_list:
                    overwritten_runtime_config = copy.deepcopy(runtime_config)
                    overwritten_runtime_config.batch_size = b
                    summary = sess.run_static(
                        mode=mode,
                        runtime_config=overwritten_runtime_config,
                        latency_correction_scale=latency_correction_scale,
                    )
                    if not summary.check_oom():
                        all_configs_oom = False
                        summary_df = pd.concat(
                            [summary_df, summary.get_summary_df()],
                            axis=0,
                            ignore_index=True,
                        )
                    else:  # larger b will always OOM
                        break
            except Exception as e:
                logger.warning(
                    "Error getting candidate workers with parallel config: "
                    "tp=%d, pp=%d, dp=%d, moe_tp=%d, moe_ep=%d; "
                    "skipping this combination. Error: %s",
                    tp_size,
                    pp_size,
                    dp_size,
                    moe_tp_size,
                    moe_ep_size,
                    e,
                )
                exceptions.append(e)
                continue
        if summary_df.empty:
            if exceptions:
                raise RuntimeError(
                    f"No results found for any parallel configuration. Showing last exception: {exceptions[-1]}"
                ) from exceptions[-1]
            if all_configs_oom:
                raise RuntimeError(
                    "No results found: the model does not fit in GPU memory for any parallel "
                    "configuration. Try increasing --total-gpus, using a quantized model, or "
                    "using a system with more VRAM per GPU."
                )
            raise NoFeasibleConfigError(
                "No results found for any parallel configuration. No configuration satisfied the "
                "TTFT/TPOT or request-latency constraints. Try relaxing --ttft, --tpot, or "
                "--request_latency (e.g., higher ttft/tpot or higher request_latency)."
            )
        return summary_df

    def _pick_autoscale(
        self,
        prefill_summary_df: pd.DataFrame,
        decode_summary_df: pd.DataFrame,
        runtime_config: config.RuntimeConfig,
        disagg_summary: InferenceSummary,
        target_ttft: float | None = None,
        target_tpot: float | None = None,
        top_n: int = 5,
    ) -> InferenceSummary:
        """Pick best prefill and decode engines independently for autoscaling.

        Delegates to :func:`aiconfigurator.sdk.picking.pick_autoscale` and
        wraps the result in an ``InferenceSummary``.
        """
        from aiconfigurator.sdk.picking import pick_autoscale

        if target_ttft is None:
            target_ttft = runtime_config.ttft

        if target_tpot is None:
            tpot_values = runtime_config.tpot if isinstance(runtime_config.tpot, list) else [runtime_config.tpot]
            target_tpot = max(tpot_values)

        result = pick_autoscale(
            prefill_df=prefill_summary_df,
            decode_df=decode_summary_df,
            target_ttft=target_ttft,
            target_tpot=target_tpot,
            top_n=top_n,
        )

        disagg_summary_df = result["best_config_df"]
        if not disagg_summary_df.empty:
            disagg_summary.set_summary_df(disagg_summary_df)
        return disagg_summary

    # optimization
    def find_best_disagg_result_under_constraints(
        self,
        model_path: str,
        runtime_config: config.RuntimeConfig,
        prefill_model_config: config.ModelConfig,
        prefill_parallel_config_list: list[tuple[int, int, int, int, int, int]],
        prefill_max_num_tokens: int,
        prefill_num_worker_list: list[int],
        decode_model_config: config.ModelConfig,
        decode_parallel_config_list: list[tuple[int, int, int, int, int, int]],
        decode_max_num_tokens: int,
        decode_num_worker_list: list[int],
        num_gpu_list: list[int] | None,
        max_prefill_gpus: int | None = None,
        max_decode_gpus: int | None = None,
        require_same_tp: bool = False,
        autoscale: bool = False,
        target_tpot: float | None = None,
    ) -> InferenceSummary | None:
        """
        Run disagg with given constraints
        1. get all summary df, which matches the constraints
        2. find best config under constraints, call match scales to get the best scale
        3. call a func to get disagg_summary_df (this is shared by run_disgg func)
        4. return summary
        5. several empirical values:
            - 0.7 is the threshold to filter decode workers, because the performance of
              decode workers is much lower than prefill workers
            - 5 is the top k to return for drawing pareto frontier of each tpot

        Args:
            model_path (str): the model name
            runtime_config (RuntimeConfig): the runtime config
            prefill_model_config (ModelConfig): the prefill model config
            prefill_parallel_config_list (List[Tuple[int, int, int, int, int]]):
                the prefill parallel config list
            prefill_max_num_tokens (int): the prefill max num tokens
            prefill_num_worker_list (List[int]): the prefill num worker list
            decode_model_config (ModelConfig): the decode model config
            decode_parallel_config_list (List[Tuple[int, int, int, int, int]]):
                the decode parallel config list
            decode_max_num_tokens (int): the decode max num tokens
            decode_num_worker_list (List[int]): the decode num worker list
            num_gpu_list (Optional[List[int]]): the num gpu list

        Returns:
            Optional[InferenceSummary]: the summary of the inference result, contains all the
                possible disagg config and perf that matches SLA.
        """

        if max_prefill_gpus is not None and max_prefill_gpus <= 0:
            raise ValueError(f"max_prefill_gpus must be a positive integer, got {max_prefill_gpus}")
        if max_decode_gpus is not None and max_decode_gpus <= 0:
            raise ValueError(f"max_decode_gpus must be a positive integer, got {max_decode_gpus}")

        # minor perf optimization: convert num_gpu_list to a set to speed up lookup
        num_gpu_set = set[int](num_gpu_list) if num_gpu_list else set()

        @functools.lru_cache(maxsize=8192)
        def _match_workers(
            prefill_throughput: float,
            prefill_gpus: int,
            decode_throughput: float,
            decode_gpus: int,
            rate_matching_prefill_degradation_factor: float,
            rate_matching_decode_degradation_factor: float,
        ) -> tuple[int, int]:
            """
            Match the prefill and decode workers, return the best prefill and decode num worker
            """
            prefill_opt_num_worker, decode_opt_num_worker = -1, -1
            throughput_per_gpu_max = 0
            for decode_num_worker in decode_num_worker_list:
                for prefill_num_worker in prefill_num_worker_list:
                    num_gpu = prefill_gpus * prefill_num_worker + decode_gpus * decode_num_worker

                    # if num_gpu_set is empty, we don't have any constraint on the number of gpus
                    # if num_gpu_set is not empty, we only consider the gpus that are in the set
                    if len(num_gpu_set) > 0 and num_gpu not in num_gpu_set:
                        continue

                    # per-pool GPU budget for hetero disagg
                    if max_prefill_gpus is not None and max_decode_gpus is not None:
                        if prefill_gpus * prefill_num_worker > max_prefill_gpus:
                            continue
                        if decode_gpus * decode_num_worker > max_decode_gpus:
                            continue

                    prefill_throughput_corrected = (
                        prefill_throughput * prefill_num_worker * rate_matching_prefill_degradation_factor
                    )
                    decode_throughput_corrected = (
                        decode_throughput * decode_num_worker * rate_matching_decode_degradation_factor
                    )

                    # criteria 1, try to make prefill_throughput larger than decode_throughput
                    # otherwise, decode bs cannot be achieved and decode throughput cannot be
                    # achieved as well.
                    # if prefill_throughput < decode_throughput:
                    #    continue

                    # criteria 2, try to make the throughput per gpu larger
                    throughput_per_gpu = min(prefill_throughput_corrected, decode_throughput_corrected) / num_gpu

                    if throughput_per_gpu > throughput_per_gpu_max:
                        throughput_per_gpu_max = throughput_per_gpu
                        prefill_opt_num_worker, decode_opt_num_worker = (
                            prefill_num_worker,
                            decode_num_worker,
                        )

            return prefill_opt_num_worker, decode_opt_num_worker

        def _find_best_result_under_constraints(
            ttft: float,
            tpot: float,
            prefill_summary_df: pd.DataFrame,
            decode_summary_df: pd.DataFrame,
            return_top_k: int,
            num_gpu_list: list[int] | None,
            rate_matching_prefill_degradation_factor: float,
            rate_matching_decode_degradation_factor: float,
            require_same_tp: bool = False,
        ) -> InferenceSummary:
            """
            Find the best result under constraints
            """

            # 1. we categorize the decode summary
            #    df into different categories based on parallelism (we can use the parallel key in
            #    the df). do the rate matching and sort the result by category - throughput.
            # 2. for prefill, follow two rules: high throughput, if at same level, choose the one
            #    with small batchsize. add one func for correct ttft (we have some formula,
            #    just leave it blank for now)
            # 3. prefill/decode correction are already applied to workers.
            #    Additional correction can be a degradation factor for the final result during the
            #   rate matching process.
            # 4. rate matching. The prefill throughput should be 1.x larger than the decode
            #    throughput.
            #    "1.x" is an empirical value. Default is 1.1.

            # only ttft will be corrected here, other latency and throughput will not be
            # corrected. concurrency / num_prefill_workers = local_concurrency(lc);
            # N x concurrency requests. formula = (lc * (lc+1) / 2 + lc * (N-1) )/lc/N
            # if we use N=10, it's lc/20+0.95. assume lc can be 15-20, 1.8 is a reasonable
            # correction factor. as we need to get the lc after rate matching, we cannot get the
            # exact value now. Let's make it simple to do pre-correction instead of post-correction.
            correction_factor = _AUTOSCALE_TTFT_CORRECTION_FACTOR
            prefill_candidates = prefill_summary_df.assign(ttft=prefill_summary_df["ttft"] * correction_factor)

            prefill_candidates = prefill_candidates[prefill_candidates["ttft"] < ttft]
            if len(prefill_candidates) == 0:
                logger.debug(f"No prefill worker candidates found for ttft {ttft}ms.")
                return None
            prefill_candidates = (
                prefill_candidates.sort_values(by=["seq/s/gpu", "global_bs"], ascending=[False, True])
                .reset_index(drop=True)
                .head(MAX_PREFILL_WORKERS)
            )

            decode_candidates = decode_summary_df[
                (decode_summary_df["tpot"] < tpot * DECODE_FILTER_RATIO_MAX)
                & (decode_summary_df["tpot"] > tpot * DECODE_FILTER_RATIO_MIN)
            ].copy()
            if len(decode_candidates) == 0:
                logger.debug(f"No decode worker candidates found for tpot {tpot}ms.")
                return None

            all_category_results: list[dict] = []
            prefill_candidates_list = prefill_candidates.to_dict("records")

            for parallel_value, parallel_group in decode_candidates.groupby("parallel"):
                parallel_group_sorted = (
                    parallel_group.sort_values(by=["seq/s/gpu"], ascending=[False])
                    .reset_index(drop=True)
                    .head(MAX_DECODE_WORKERS_PER_CATEGORY)
                )

                decode_workers_list = parallel_group_sorted.to_dict("records")
                category_results: list[dict] = []
                for decode_worker in decode_workers_list:
                    decode_throughput = float(decode_worker["seq/s"])
                    decode_gpus = decode_worker["num_total_gpus"]
                    for prefill_worker in prefill_candidates_list:
                        # For SGLang non-wideep disaggregated serving
                        # See: https://github.com/ai-dynamo/dynamo/issues/5870
                        if require_same_tp and prefill_worker["tp"] != decode_worker["tp"]:
                            continue
                        prefill_throughput = float(prefill_worker["seq/s"])
                        prefill_gpus = prefill_worker["num_total_gpus"]
                        prefill_num_worker, decode_num_worker = _match_workers(
                            prefill_throughput=prefill_throughput,
                            prefill_gpus=prefill_gpus,
                            decode_throughput=decode_throughput,
                            decode_gpus=decode_gpus,
                            rate_matching_prefill_degradation_factor=rate_matching_prefill_degradation_factor,
                            rate_matching_decode_degradation_factor=rate_matching_decode_degradation_factor,
                        )
                        if prefill_num_worker == -1 or decode_num_worker == -1:
                            continue

                        disagg_dict = _build_disagg_summary_dict(
                            prefill_worker,
                            prefill_num_worker,
                            decode_worker,
                            decode_num_worker,
                            prefill_degradation_factor=rate_matching_prefill_degradation_factor,
                            decode_degradation_factor=rate_matching_decode_degradation_factor,
                        )
                        category_results.append(disagg_dict)

                if category_results:
                    # only return the best one for each category
                    best_result = max(category_results, key=lambda x: (x["tokens/s/gpu"], -x["num_total_gpus"]))
                    all_category_results.append(best_result)
                else:
                    logger.debug(f"No matched result for decode parallel {parallel_value}.")

            if not all_category_results:
                logger.debug("No disagg summary found after applying constraints.")
                return None

            disagg_summary_df = pd.DataFrame(all_category_results, columns=common.ColumnsDisagg).round(3)
            disagg_summary_df = (
                disagg_summary_df.sort_values(by=["tokens/s/gpu"], ascending=[False])
                .head(return_top_k)
                .reset_index(drop=True)
            )
            return disagg_summary_df
            # _find_best_result_under_constraints() ends here

        # start, get all possible p/d servers
        if decode_max_num_tokens < 1:
            logger.warning("decode_max_num_tokens is less than 1, set to 1")
            decode_max_num_tokens = 1
        decode_batch_size_list_default = (
            list(range(1, 16, 1)) + list(range(16, 32, 2)) + list(range(32, 128, 4)) + list(range(128, 512, 8)) + [512]
        )
        if decode_max_num_tokens > max(decode_batch_size_list_default):
            decode_batch_size_range = decode_batch_size_list_default + [decode_max_num_tokens]
        else:
            decode_batch_size_range = [i for i in decode_batch_size_list_default if i <= decode_max_num_tokens]

        try:
            enc_cfg = get_model_config_from_model_path(model_path).get("extra_params")
        except Exception:
            logger.debug("Could not resolve model config for VL effective ISL; using text ISL", exc_info=True)
            enc_cfg = None
        prefill_effective_isl = runtime_config.isl + BaseBackend._visual_context_tokens_from_encoder_config(
            enc_cfg, runtime_config
        )
        if prefill_max_num_tokens < prefill_effective_isl:
            logger.warning("prefill_max_num_tokens is less than effective prefill ISL, set to effective prefill ISL")
            prefill_max_num_tokens = prefill_effective_isl

        max_prefill_batch_size = prefill_max_num_tokens // prefill_effective_isl
        prefill_batch_size_range = range(1, max_prefill_batch_size + 1)

        # initialize disagg summary
        disagg_summary = InferenceSummary(runtime_config=runtime_config)
        disagg_summary_df = pd.DataFrame(columns=common.ColumnsDisagg)
        disagg_summary.set_summary_df(disagg_summary_df)

        # find prefill and decode workers
        prefill_summary_df = self.get_worker_candidates(
            model_path=model_path,
            model_config=prefill_model_config,
            parallel_config_list=prefill_parallel_config_list,
            b_list=prefill_batch_size_range,
            runtime_config=runtime_config,
            mode="static_ctx",
            latency_correction_scale=self._prefill_latency_correction_scale,
        )
        decode_summary_df = self.get_worker_candidates(
            model_path=model_path,
            model_config=decode_model_config,
            parallel_config_list=decode_parallel_config_list,
            b_list=decode_batch_size_range,
            runtime_config=runtime_config,
            mode="static_gen",
            latency_correction_scale=self._decode_latency_correction_scale,
        )
        if len(prefill_summary_df) == 0 or len(decode_summary_df) == 0:
            logger.debug(f"No prefill or decode workers found for {model_path} with given configs.")
            return disagg_summary

        # ----- autoscale mode: pick P and D independently, no rate matching -----
        if autoscale:
            return self._pick_autoscale(
                prefill_summary_df=prefill_summary_df,
                decode_summary_df=decode_summary_df,
                runtime_config=runtime_config,
                disagg_summary=disagg_summary,
                target_tpot=target_tpot,
            )

        # find best result under constraints
        constraint_pairs: list[tuple[float, float]] = []
        if runtime_config.request_latency is not None and runtime_config.request_latency > 0:
            constraint_pairs = enumerate_ttft_tpot_constraints(
                runtime_config.osl,
                runtime_config.request_latency,
                runtime_config.ttft,
            )
            if not constraint_pairs:
                logger.debug(
                    "No ttft/tpot constraints derived for request_latency=%s in disagg optimization.",
                    runtime_config.request_latency,
                )
        else:
            tpot_values = runtime_config.tpot if isinstance(runtime_config.tpot, list) else [runtime_config.tpot]
            constraint_pairs = [(runtime_config.ttft, tpot) for tpot in tpot_values]

        for ttft_constraint, tpot_constraint in constraint_pairs:
            logger.debug(
                "Finding best result under constraints for ttft=%sms, tpot=%sms...",
                ttft_constraint,
                tpot_constraint,
            )
            filtered_disagg_summary_df = _find_best_result_under_constraints(
                ttft=ttft_constraint,
                tpot=tpot_constraint,
                prefill_summary_df=prefill_summary_df,
                decode_summary_df=decode_summary_df,
                return_top_k=5,
                num_gpu_list=num_gpu_list,
                rate_matching_prefill_degradation_factor=self._rate_matching_prefill_degradation_factor,
                rate_matching_decode_degradation_factor=self._rate_matching_decode_degradation_factor,
                require_same_tp=require_same_tp,
            )
            if filtered_disagg_summary_df is not None:
                disagg_summary_df = pd.concat(
                    [disagg_summary_df, filtered_disagg_summary_df], axis=0, ignore_index=True
                )
        if len(disagg_summary_df) == 0:
            logger.debug(f"No disagg result found for {model_path} with given constraints.")
            return disagg_summary

        disagg_summary_df = disagg_summary_df.drop_duplicates(ignore_index=True)

        # set final disagg summary
        disagg_summary.set_summary_df(disagg_summary_df)
        return disagg_summary


# Private helper: bundles the five comm-side ops a single AFD layer needs.
# Kept module-private; if a second consumer appears in the future, promote
# to a public dataclass at that point.
_AFDCommOps = namedtuple("_AFDCommOps", ["a2f", "f2a", "f_ag", "f_rs", "a_combine"])


class AFDInferenceSession:
    """Attention-FFN Disaggregated inference session.

    Simulates the AFD pipeline where Attention ops run on A-Workers and
    FFN/MoE ops run on F-Workers, communicating hidden activations every
    layer via a ping-pong pipeline.

    AFD is **orthogonal** to Prefill/Decode (P/D) disaggregation:

    * ``phase="decode"`` (default) — matches historical behavior: per-layer
      ping-pong pipeline for generation steps.  ``TPOT`` is populated,
      ``TTFT`` is 0.
    * ``phase="prefill"`` — applies the same A/F split to context ops.
      ``TTFT`` (= one prefill ``T_step``) is populated, ``TPOT`` is 0.
    * ``phase="both"`` — combines both above and reports end-to-end
      ``request_latency = TTFT + (osl-1) * TPOT``.

    In combination with P/D disagg, an external caller can run two
    sessions (one for prefill workers, one for decode workers) and
    aggregate the two summaries.

    ``run_afd(runtime_config, phase=None)`` is the public entry point;
    ``run_afd_decode`` / ``run_afd_prefill`` are thin convenience wrappers.

    Memory (HBM) bound is checked for both A-Workers and F-Workers via
    :class:`aiconfigurator.sdk.inference_summary.InferenceSummary`.
    """

    def __init__(
        self,
        model_path: str,
        a_model_config: config.ModelConfig,
        f_model_config: config.ModelConfig,
        database: perf_database.PerfDatabase,
        backend: BaseBackend,
        afd_config: config.AFDConfig,
    ) -> None:
        self._model_path = model_path
        self._a_model_config = a_model_config
        self._f_model_config = f_model_config
        self._database = database
        self._backend = backend
        self._afd_config = afd_config

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #
    def _build_models(self):
        """Construct A-Worker and F-Worker model instances."""
        from aiconfigurator.sdk.models import get_model

        a_model = get_model(self._model_path, self._a_model_config, self._backend.name.value)
        f_model = get_model(self._model_path, self._f_model_config, self._backend.name.value)
        return a_model, f_model

    def _sum_latency(
        self,
        ops_iter,
        *,
        batch_size: int,
        seq_len: int,
        model,
        runtime_config: config.RuntimeConfig,
        is_context: bool,
    ):
        """Sum the query() latencies for a list of ops, returning (total, per-op dict).

        For prefill (``is_context=True``) we pass ``seq_imbalance_correction_scale``;
        for decode we pass ``gen_seq_imbalance_correction_scale``.  Tokens
        processed per call = ``batch_size`` for decode, ``batch_size*seq_len``
        for prefill (one token per sequence vs. full sequence).
        """
        x = batch_size * seq_len if is_context else batch_size

        kwargs_common = {
            "x": x,
            "batch_size": batch_size,
            "beam_width": 1,
            "s": seq_len,
            "prefix": runtime_config.prefix,
            "model_name": getattr(model, "model_name", ""),
        }
        if is_context:
            kwargs_common["seq_imbalance_correction_scale"] = runtime_config.seq_imbalance_correction_scale
        else:
            kwargs_common["gen_seq_imbalance_correction_scale"] = runtime_config.gen_seq_imbalance_correction_scale

        per_op = defaultdict(float)
        for op in ops_iter:
            result = op.query(self._database, **kwargs_common)
            per_op[op._name] += float(result)
        return sum(per_op.values()), per_op

    def _afd_batch_shape(self) -> tuple[int, int, int, int, int]:
        """Return AFD total and per-microbatch batch sizes.

        ``AFDConfig.a_batch_size`` is the total in-flight batch per
        A-Worker.  The pipeline executes one microbatch at a time, so
        latency and transfer queries use the derived per-microbatch
        sizes while summary/concurrency fields report the total batch.
        """
        cfg = self._afd_config
        num_microbatches = max(int(cfg.num_microbatches or 1), 1)
        a_total_batch_size = max(int(cfg.a_batch_size), 1)
        a_micro_batch_size = max(
            (a_total_batch_size + num_microbatches - 1) // num_microbatches,
            1,
        )
        b_total = cfg.n_a_workers * a_total_batch_size
        b_micro_total = cfg.n_a_workers * a_micro_batch_size
        return (
            a_total_batch_size,
            a_micro_batch_size,
            b_total,
            b_micro_total,
            num_microbatches,
        )

    def _build_afd_comm_ops(self, a_model, f_model, *, rank_mapping: str = "one_to_one") -> _AFDCommOps:
        """Construct the five comm-side ops modeling AFD per-layer traffic.

        AFD comm decomposes into five independent pieces. Each is now its
        own op returning :class:`PerformanceResult` (a float-like with
        ``.energy`` / ``.source``); the session synthesizes them into the
        per-layer ``t_a2f`` / ``t_f2a`` / ``t_a`` / ``t_f`` ingredients
        consumed by ``_pipeline_tcycle``.

        * ``a2f`` / ``f2a`` — cross-pool single-direction P2P transfers.
        * ``f_ag`` / ``f_rs`` — F-node intra-node AllGather (dispatch)
          and ReduceScatter (combine) along the token dimension. They
          return 0 when ``f_local <= 1`` or under broadcast mapping, so
          the session can sum them unconditionally.
        * ``a_combine`` — A-side local HBM reduce-add over EP partial
          results. Returns 0 when ``f_moe_ep_size <= 1``.

        ``rank_mapping`` selects the dispatch topology:
        ``"one_to_one"`` (default) keeps the F-side AG/RS;
        ``"broadcast"`` reports them as 0 (placeholder for future
        modeling of A-rank → all-F-ranks fan-out).

        MoE dispatch probability is driven by ``f_model._num_experts`` /
        ``f_model._topk`` (combinatorial formula when both present and
        ``num_f_nodes > 1``); for dense models the AFD ops fall back to a
        uniform ``1 / num_f_nodes`` split. EP=1 MoE models still use the
        combinatorial form since each token activates only ``topk``
        experts regardless of EP layout.
        """
        from aiconfigurator.sdk.operations import (
            AFDCombine,
            AFDFAllGather,
            AFDFReduceScatter,
            AFDTransfer,
        )

        cfg = self._afd_config
        comm_quant = self._a_model_config.comm_quant_mode
        num_experts = int(getattr(f_model, "_num_experts", 0) or 0)
        topk = int(getattr(f_model, "_topk", 0) or 0)

        shared = dict(
            hidden_size=a_model._hidden_size,
            n_a_workers=cfg.n_a_workers,
            n_f_workers=cfg.n_f_workers,
            gpus_per_node=cfg.gpus_per_node,
            num_experts=num_experts,
            topk=topk,
            comm_quant_mode=comm_quant,
        )
        return _AFDCommOps(
            a2f=AFDTransfer(
                name="afd_a2f_transfer",
                scale_factor=1.0,
                direction="a2f",
                comm_overhead_factor=cfg.comm_overhead_factor,
                **shared,
            ),
            f2a=AFDTransfer(
                name="afd_f2a_transfer",
                scale_factor=1.0,
                direction="f2a",
                comm_overhead_factor=cfg.comm_overhead_factor,
                **shared,
            ),
            f_ag=AFDFAllGather(
                name="afd_f_node_allgather",
                scale_factor=1.0,
                rank_mapping=rank_mapping,
                **shared,
            ),
            f_rs=AFDFReduceScatter(
                name="afd_f_node_reducescatter",
                scale_factor=1.0,
                rank_mapping=rank_mapping,
                **shared,
            ),
            a_combine=AFDCombine(
                name="afd_a_side_combine",
                scale_factor=1.0,
                hidden_size=a_model._hidden_size,
                tp_a=cfg.tp_a,
                f_moe_ep_size=cfg.f_moe_ep_size,
                comm_quant_mode=comm_quant,
            ),
        )

    def _pipeline_tcycle(self, t_a: float, t_f: float, t_a2f: float, t_f2a: float) -> tuple[float, bool]:
        """Compute per-layer cycle time and whether comm is hidden.

        Two pipeline regimes are supported:

        * **K=3 (optimistic, 3-batch overlap)** — the network round trip
          ``t_c = t_a2f + t_f2a`` is its own pipeline stage, so::

              TPOT_layer = max(t_a, t_f, t_c)            (N_min = 3)

          When ``t_c <= max(t_a, t_f)`` communication is fully hidden
          by computation; otherwise the network is the bottleneck.

        * **K=2 (conservative, blocking communication)** — each pool
          waits for its own outgoing/incoming transfer::

              TPOT_layer = max(t_a + t_a2f, t_f + t_f2a) (N_min = 2)

        The optimistic model falls back to conservative when there are
        not enough in-flight micro-batches to fill the K=3 pipeline.

        Returns:
            (t_cycle, comm_hidden).  ``comm_hidden`` is True only in the
            K=3 ideal case where the network stage does not dominate the
            cycle.
        """
        cfg = self._afd_config
        num_microbatches = max(int(cfg.num_microbatches or 1), 1)
        t_c = t_a2f + t_f2a
        if cfg.pipeline_model == "optimistic":
            # Need ≥ 2 + t_c / max(t_a, t_f) in-flight microbatches to
            # hide the network stage behind compute.  Equivalent to the
            # legacy ``2 * (1 + t_c_one_dir / max(...))`` under the
            # symmetric Phase-1 assumption (t_a2f == t_f2a).
            min_m = 2.0 + t_c / max(t_a, t_f, 1e-9)
            if num_microbatches < min_m:
                logger.warning(
                    "AFD optimistic pipeline: num_microbatches (%d) < min required (%.1f) "
                    "to hide communication. Falling back to conservative model.",
                    num_microbatches,
                    min_m,
                )
                return max(t_a + t_a2f, t_f + t_f2a), False
            t_cycle = max(t_a, t_f, t_c)
            comm_hidden = t_c <= max(t_a, t_f)
            return t_cycle, comm_hidden
        # conservative K=2
        return max(t_a + t_a2f, t_f + t_f2a), False

    def _estimate_a_memory_dict(
        self,
        *,
        a_model,
        a_partition,
        phase: str,
        batch_size: int,
        isl: int,
        osl: int,
        prefix: int,
        max_seq_len: int | None,
    ) -> dict[str, float]:
        """Estimate A-Worker per-GPU HBM usage as a standard memory dict."""
        cfg = self._afd_config
        if phase == "prefill":
            effective_max_seq_len = max_seq_len if max_seq_len is not None else max(isl, 1)
            num_tokens = 0
            kvcache_multiplier = 1
        else:
            effective_max_seq_len = max_seq_len if max_seq_len is not None else isl + osl
            num_tokens = batch_size
            kvcache_multiplier = max(int(cfg.num_microbatches or 1), 1)

        return self._backend.get_partition_memory_usage(
            a_model,
            self._database,
            partition_ops=a_partition.attn_ops,
            batch_size=batch_size,
            beam_width=1,
            isl=isl,
            osl=osl,
            num_tokens=num_tokens,
            prefix=prefix,
            max_seq_len=effective_max_seq_len,
            include_kvcache=True,
            kvcache_multiplier=kvcache_multiplier,
        )

    def _estimate_f_memory_dict(
        self,
        *,
        f_model,
        f_partition,
        phase: str,
        batch_size: int,
        isl: int,
        osl: int,
        prefix: int,
        max_seq_len: int | None,
    ) -> dict[str, float]:
        """Estimate F-Worker per-GPU HBM usage as a standard memory dict."""
        if phase == "prefill":
            effective_max_seq_len = max_seq_len if max_seq_len is not None else max(isl, 1)
            num_tokens = 0
        else:
            effective_max_seq_len = max_seq_len if max_seq_len is not None else isl + osl
            num_tokens = batch_size

        return self._backend.get_partition_memory_usage(
            f_model,
            self._database,
            partition_ops=f_partition.ffn_ops,
            batch_size=batch_size,
            beam_width=1,
            isl=isl,
            osl=osl,
            num_tokens=num_tokens,
            prefix=prefix,
            max_seq_len=effective_max_seq_len,
            include_kvcache=False,
        )

    def _check_memory_dict(
        self,
        memory: dict[str, float],
        runtime_config: config.RuntimeConfig,
        free_gpu_memory_fraction: float | None,
    ) -> InferenceSummary:
        summary = InferenceSummary(runtime_config)
        reserved_fraction, tolerance = self._backend.get_kv_cache_memory_check_params()
        summary.set_memory_and_check_oom(
            memory,
            self._database.system_spec["gpu"]["mem_capacity"],
            free_gpu_memory_fraction=free_gpu_memory_fraction,
            kv_cache_reserved_fraction=reserved_fraction,
            kv_cache_tolerance=tolerance,
        )
        return summary

    # Stride for sampling KV-cache length ``s`` along the decode trace.
    # Mirrors ``base_backend._run_generation_phase`` so the AFD path
    # uses the same numerical integration grid as agg/disagg.
    _AFD_DECODE_STRIDE = 32

    def _integrate_decode_phase(
        self,
        *,
        a_partition,
        f_partition,
        a_model,
        f_model,
        runtime_config: config.RuntimeConfig,
        isl: int,
        osl: int,
        a_batch_size: int,
        b_batch_size: int,
        num_layers: int,
        brk_t_a_per_layer: float,
        brk_t_f_per_layer: float,
        t_a2f_layer: float,
        t_f2a_layer: float,
    ) -> tuple[float, float, float, dict, dict]:
        """Integrate compute latency along the decode KV-cache length.

        Attention is the only op whose latency reads ``s``; sampling at
        ``stride = _AFD_DECODE_STRIDE`` mirrors the trapezoidal rule
        used by ``_run_generation_phase`` and recovers the average
        per-step latency over the full decode trace.

        Returns ``(t_a_layer_avg, t_f_layer_avg, t_step_avg, a_per_op,
        f_per_op)``, where the scalars are per-step averages and the
        per-op dicts are *per-step* totals (averaged across the trace)
        in the same units as ``_sum_latency`` output.

        ``brk_t_a_per_layer`` / ``brk_t_f_per_layer`` are the per-layer
        intra-pool comm contributions (A-side combine, F-side AG+RS) —
        s-independent — folded into ``t_a_layer_i`` / ``t_f_layer_i``
        *before* the per-step ``_pipeline_tcycle`` call so the pipeline
        max is
        evaluated on the full per-layer time, not on the compute-only
        time.
        """
        stride = self._AFD_DECODE_STRIDE

        t_a_layer_sum = 0.0
        t_f_layer_sum = 0.0
        t_step_sum = 0.0
        a_per_op_sum: dict[str, float] = defaultdict(float)
        f_per_op_sum: dict[str, float] = defaultdict(float)
        total_repeat = 0

        decode_steps = max(osl - 1, 1)
        for i in range(0, decode_steps, stride):
            s_i = isl + i + 1
            # ``osl <= 1`` is degenerate (no decode tokens); fall back
            # to a single representative sample so callers still get a
            # non-zero estimate rather than zero-filled metrics.
            repeat = min(stride, osl - 1 - i) if osl > 1 else 1
            if repeat <= 0:
                break

            t_a_step_i, a_per_op_i = self._sum_latency(
                a_partition.attn_ops,
                batch_size=a_batch_size,
                seq_len=s_i,
                model=a_model,
                runtime_config=runtime_config,
                is_context=False,
            )
            t_f_step_i, f_per_op_i = self._sum_latency(
                f_partition.ffn_ops,
                batch_size=b_batch_size,
                seq_len=s_i,
                model=f_model,
                runtime_config=runtime_config,
                is_context=False,
            )

            t_a_layer_i = t_a_step_i / num_layers + brk_t_a_per_layer
            t_f_layer_i = t_f_step_i / num_layers + brk_t_f_per_layer
            # IMPORTANT: pipeline max is applied per-step before
            # accumulation. ``sum_i max(...)`` ≠ ``max(sum_i ...)``;
            # the latter under-estimates the bottleneck whenever the
            # winning pool changes across the decode trace.
            t_cycle_i, _ = self._pipeline_tcycle(t_a_layer_i, t_f_layer_i, t_a2f_layer, t_f2a_layer)
            t_step_i = num_layers * t_cycle_i

            t_a_layer_sum += t_a_layer_i * repeat
            t_f_layer_sum += t_f_layer_i * repeat
            t_step_sum += t_step_i * repeat
            for k, v in a_per_op_i.items():
                a_per_op_sum[k] += v * repeat
            for k, v in f_per_op_i.items():
                f_per_op_sum[k] += v * repeat
            total_repeat += repeat

        denom = max(total_repeat, 1)
        t_a_layer_avg = t_a_layer_sum / denom
        t_f_layer_avg = t_f_layer_sum / denom
        t_step_avg = t_step_sum / denom
        a_per_op = {k: v / denom for k, v in a_per_op_sum.items()}
        f_per_op = {k: v / denom for k, v in f_per_op_sum.items()}
        return t_a_layer_avg, t_f_layer_avg, t_step_avg, a_per_op, f_per_op

    def _simulate_phase(
        self,
        *,
        phase: str,
        runtime_config: config.RuntimeConfig,
        a_model,
        f_model,
        free_gpu_memory_fraction: float | None,
        max_seq_len: int | None,
    ) -> dict:
        """Simulate one phase (prefill or decode) and return a metrics dict.

        Keys: ``t_a_layer``, ``t_f_layer``, ``t_a2f_layer``,
        ``t_f2a_layer``, ``t_c_layer`` (round-trip = t_a2f + t_f2a),
        ``t_cycle``, ``t_step``, ``comm_hidden``, ``balance_ratio``,
        ``a_per_op``, ``f_per_op``, ``a_memory``, ``f_memory``,
        ``a_is_oom``, ``f_is_oom``, ``num_layers``.

        Decode integrates per-step compute along the KV-cache length
        ``s`` (sampled every ``_AFD_DECODE_STRIDE`` tokens, mirroring
        ``base_backend._run_generation_phase``). Attention is the only
        op that reads ``s`` — sampling at a single ``s = isl + 1`` would
        under-count A-side latency by ~33% in the typical ``osl ~ isl``
        regime and several-fold for ``osl ≫ isl``, which silently flips
        the AFD bottleneck judgement and biases sizing.

        The pipeline cycle is evaluated **per step before summing**:
        ``sum_i max(t_a_i, t_f_i, t_c)`` is not equal to
        ``max(sum_i t_a_i, sum_i t_f_i, N · t_c)``, and the latter
        consistently under-estimates the bottleneck. Headline scalars
        in the returned dict are *per-step averages* so they remain
        compatible with the downstream ``request_latency = tpot ·
        (osl - 1)`` convention used by ``_build_summary``.
        """
        from aiconfigurator.sdk.afd_partition import build_afd_ops_partition

        cfg = self._afd_config
        ops_phase = "context" if phase == "prefill" else "generation"
        # Boundary ops (``add_norm_2`` / ``logits_gemm``) default to the
        # A-Worker, but ``cfg.boundary_on_attn`` lets the user reassign
        # them to the F-Worker for sensitivity studies.
        a_partition = build_afd_ops_partition(a_model, phase=ops_phase, boundary_on_attn=cfg.boundary_on_attn)
        f_partition = build_afd_ops_partition(f_model, phase=ops_phase, boundary_on_attn=cfg.boundary_on_attn)

        isl = runtime_config.isl
        osl = runtime_config.osl or 1
        effective_prefill_len = max(isl - int(runtime_config.prefix or 0), 1)
        num_layers = max(int(getattr(a_model, "_num_layers", 1)), 1)

        (
            _a_total_batch_size,
            a_micro_batch_size,
            _b_total,
            b_micro_total,
            _num_microbatches,
        ) = self._afd_batch_shape()

        # AFD comm (A↔F cross-pool transfers, F-node AllGather /
        # ReduceScatter, A-side combine) bills by *token* volume per
        # step, not request count.  In prefill each request contributes
        # the uncached suffix
        # (``isl - prefix``) per layer; in decode it contributes 1 token per
        # step.  Each comm op's ``query(x=...)`` takes the number of tokens
        # held by a single A-rank; the op internally fans this out to the
        # global token count via ``n_a_workers``.
        tokens_per_req = effective_prefill_len if phase == "prefill" else 1
        afd_a_batch_tokens = a_micro_batch_size * tokens_per_req

        # Five comm-side ops model the per-layer AFD traffic:
        #   * a2f / f2a — cross-pool single-direction P2P transfers.
        #   * f_ag / f_rs — F-node intra-node AG (dispatch) and RS (combine)
        #     along the token dimension; return 0 outside one-to-one mapping.
        #   * a_combine — A-side local HBM reduce-add over EP partials;
        #     returns 0 for dense FFN (``f_moe_ep_size <= 1``).
        # All five bill by token volume only — independent of ``s`` — so
        # one query each, outside the decode stride loop, is sufficient.
        # Each op's name flows into ``a_per_op`` / ``f_per_op`` as a
        # distinct label so the --detail report can attribute comm cost
        # back to the specific collective rather than a single bucket.
        comm_ops = self._build_afd_comm_ops(a_model, f_model)
        r_a2f = comm_ops.a2f.query(self._database, x=afd_a_batch_tokens)
        r_f2a = comm_ops.f2a.query(self._database, x=afd_a_batch_tokens)
        r_ag = comm_ops.f_ag.query(self._database, x=afd_a_batch_tokens)
        r_rs = comm_ops.f_rs.query(self._database, x=afd_a_batch_tokens)
        r_cmb = comm_ops.a_combine.query(self._database, x=afd_a_batch_tokens)

        # Re-pack into the legacy per-bucket breakdown so the downstream
        # per-op fold-in and per-step pipeline stay unchanged. Keys are
        # the op ``_name`` values from ``_build_afd_comm_ops``.
        brk = {
            "t_a2f": {comm_ops.a2f._name: float(r_a2f)},
            "t_f2a": {comm_ops.f2a._name: float(r_f2a)},
            "t_f": {
                comm_ops.f_ag._name: float(r_ag),
                comm_ops.f_rs._name: float(r_rs),
            },
            "t_a": {comm_ops.a_combine._name: float(r_cmb)},
        }
        t_a2f_layer = float(r_a2f)
        t_f2a_layer = float(r_f2a)
        t_c_layer = t_a2f_layer + t_f2a_layer
        brk_t_a_per_layer = float(r_cmb)
        brk_t_f_per_layer = float(r_ag) + float(r_rs)

        # Ops in :mod:`aiconfigurator.sdk.models` are constructed with
        # ``scale_factor=num_layers`` (per-layer ops such as qkv_gemm) or
        # ``scale_factor=1`` (once-per-step ops such as embedding /
        # logits_gemm).  ``_sum_latency`` therefore returns the *full
        # per-step* contribution of each pool, not a single layer.  The
        # AFD pipeline model is layer-granular, so amortize across layers
        # to recover the per-layer cycle ingredients before pipelining.
        # Once-per-step ops (embedding/logits_gemm) get folded into the
        # per-layer average; their absolute cost is small relative to
        # ``num_layers`` per-layer compute and AFD does not currently
        # model them as separate stages.
        if phase == "decode":
            t_a_layer, t_f_layer, t_step, a_per_op, f_per_op = self._integrate_decode_phase(
                a_partition=a_partition,
                f_partition=f_partition,
                a_model=a_model,
                f_model=f_model,
                runtime_config=runtime_config,
                isl=isl,
                osl=osl,
                a_batch_size=a_micro_batch_size,
                b_batch_size=b_micro_total,
                num_layers=num_layers,
                brk_t_a_per_layer=brk_t_a_per_layer,
                brk_t_f_per_layer=brk_t_f_per_layer,
                t_a2f_layer=t_a2f_layer,
                t_f2a_layer=t_f2a_layer,
            )
            t_cycle = t_step / num_layers
            # The ``comm_hidden`` flag is informational only; report it
            # at the *average* operating point so it stays a single
            # stable scalar even though the per-step pipeline above
            # already accounts for s-dependent bottleneck shifts.
            _, comm_hidden = self._pipeline_tcycle(t_a_layer, t_f_layer, t_a2f_layer, t_f2a_layer)
        else:
            # Prefill: single shot over the uncached input suffix; no
            # need to integrate, ``s == isl - prefix`` everywhere.
            seq_len_query = effective_prefill_len
            t_a_total, a_per_op = self._sum_latency(
                a_partition.attn_ops,
                batch_size=a_micro_batch_size,
                seq_len=seq_len_query,
                model=a_model,
                runtime_config=runtime_config,
                is_context=True,
            )
            t_f_total, f_per_op = self._sum_latency(
                f_partition.ffn_ops,
                batch_size=b_micro_total,
                seq_len=seq_len_query,
                model=f_model,
                runtime_config=runtime_config,
                is_context=True,
            )
            t_a_layer = t_a_total / num_layers + brk_t_a_per_layer
            t_f_layer = t_f_total / num_layers + brk_t_f_per_layer
            t_cycle, comm_hidden = self._pipeline_tcycle(t_a_layer, t_f_layer, t_a2f_layer, t_f2a_layer)
            t_step = num_layers * t_cycle

        # Per-op dicts are tracked in *per-step* units (matching
        # ``_sum_latency`` output), so amortize per-layer values up by
        # ``num_layers`` before recording.
        for label, ms in brk["t_a"].items():
            a_per_op[label] = ms * num_layers
        for label, ms in brk["t_f"].items():
            f_per_op[label] = ms * num_layers

        balance_ratio = min(t_a_layer, t_f_layer) / max(t_a_layer, t_f_layer, 1e-9)

        # HBM (memory) bound check — per-GPU on each pool.
        a_memory = self._estimate_a_memory_dict(
            a_model=a_model,
            a_partition=a_partition,
            phase=phase,
            batch_size=a_micro_batch_size,
            isl=isl,
            osl=osl,
            prefix=runtime_config.prefix,
            max_seq_len=max_seq_len,
        )
        f_memory = self._estimate_f_memory_dict(
            f_model=f_model,
            f_partition=f_partition,
            phase=phase,
            batch_size=b_micro_total,
            isl=isl,
            osl=osl,
            prefix=runtime_config.prefix,
            max_seq_len=max_seq_len,
        )
        a_memory_summary = self._check_memory_dict(a_memory, runtime_config, free_gpu_memory_fraction)
        f_memory_summary = self._check_memory_dict(f_memory, runtime_config, None)

        return {
            "t_a_layer": t_a_layer,
            "t_f_layer": t_f_layer,
            "t_a2f_layer": t_a2f_layer,
            "t_f2a_layer": t_f2a_layer,
            "t_c_layer": t_c_layer,
            "t_cycle": t_cycle,
            "t_step": t_step,
            "comm_hidden": comm_hidden,
            "balance_ratio": balance_ratio,
            "a_per_op": dict(a_per_op),
            "f_per_op": dict(f_per_op),
            "a_memory": a_memory,
            "f_memory": f_memory,
            "a_memory_gb": a_memory["total"],
            "f_memory_gb": f_memory["total"],
            "a_is_oom": a_memory_summary.check_oom() or a_memory_summary.check_kv_cache_oom(),
            "f_is_oom": f_memory_summary.check_oom() or f_memory_summary.check_kv_cache_oom(),
            "a_is_kv_cache_oom": a_memory_summary.check_kv_cache_oom(),
            "f_is_kv_cache_oom": f_memory_summary.check_kv_cache_oom(),
            "num_layers": num_layers,
            "a_partition": a_partition,
            "f_partition": f_partition,
        }

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run_afd(
        self,
        runtime_config: config.RuntimeConfig,
        *,
        phase: str | None = None,
        free_gpu_memory_fraction: float | None = None,
        max_seq_len: int | None = None,
        speculative_profile: SpeculativeDecodingProfile | None = None,
    ) -> InferenceSummary:
        """Run AFD performance simulation, possibly for prefill, decode, or both.

        AFD is orthogonal to P/D disaggregation: ``phase`` controls which
        phase is being modeled by *this* session.  When combined with P/D
        disagg, the caller typically constructs two sessions (one per pool)
        and reports both summaries.

        Args:
            runtime_config: ISL / OSL / prefix / correction scales.
            phase:          ``"prefill"``, ``"decode"``, or ``"both"``.
                             Defaults to ``self._afd_config.phase``.
            free_gpu_memory_fraction: Fraction of free GPU memory allocated
                             for KV cache. Defaults to backend behavior.
            max_seq_len:    Optional runtime max sequence length for KV cache.
            speculative_profile: Optional accepted-token progress assumption.
                             The projection role is derived from ``phase``.

        Returns:
            InferenceSummary.  ``check_oom()`` reflects the per-pool HBM check.
        """
        cfg = self._afd_config
        phase = phase if phase is not None else cfg.phase
        if phase not in ("prefill", "decode", "both"):
            raise ValueError(f"AFDInferenceSession.run_afd: invalid phase {phase!r}")
        if free_gpu_memory_fraction is None:
            free_gpu_memory_fraction = self._backend.get_default_free_gpu_memory_fraction()

        a_model, f_model = self._build_models()

        prefill_metrics = None
        decode_metrics = None
        if phase in ("prefill", "both"):
            prefill_metrics = self._simulate_phase(
                phase="prefill",
                runtime_config=runtime_config,
                a_model=a_model,
                f_model=f_model,
                free_gpu_memory_fraction=free_gpu_memory_fraction,
                max_seq_len=max_seq_len,
            )
        if phase in ("decode", "both"):
            decode_metrics = self._simulate_phase(
                phase="decode",
                runtime_config=runtime_config,
                a_model=a_model,
                f_model=f_model,
                free_gpu_memory_fraction=free_gpu_memory_fraction,
                max_seq_len=max_seq_len,
            )

        summary = self._build_summary(
            runtime_config=runtime_config,
            phase=phase,
            prefill_metrics=prefill_metrics,
            decode_metrics=decode_metrics,
        )
        if speculative_profile is None:
            return summary
        projection_role = "prefill" if phase == "prefill" else ("decode" if phase == "decode" else "static")
        return speculative_profile.project_summary(summary, role=projection_role)

    def run_afd_decode(
        self,
        runtime_config: config.RuntimeConfig,
        *,
        free_gpu_memory_fraction: float | None = None,
        max_seq_len: int | None = None,
    ) -> InferenceSummary:
        """Backwards-compatible Decode-only entry point."""
        return self.run_afd(
            runtime_config,
            phase="decode",
            free_gpu_memory_fraction=free_gpu_memory_fraction,
            max_seq_len=max_seq_len,
        )

    def run_afd_prefill(
        self,
        runtime_config: config.RuntimeConfig,
        *,
        free_gpu_memory_fraction: float | None = None,
        max_seq_len: int | None = None,
    ) -> InferenceSummary:
        """Prefill-only AFD entry point.

        Uses the same A/F split as decode but applies it to ``context_ops``,
        producing a TTFT estimate.  No persistent KV cache is required —
        the A-Worker HBM estimate tracks the transient prefill KV only.
        """
        return self.run_afd(
            runtime_config,
            phase="prefill",
            free_gpu_memory_fraction=free_gpu_memory_fraction,
            max_seq_len=max_seq_len,
        )

    # Per-phase layer scalars surfaced both un-prefixed (headline) and with
    # ``prefill_`` / ``decode_`` prefixes (paired). Keeping this in one place
    # avoids drift between the un-prefixed picker below, the prefixed
    # writer, and the CLI/CSV consumers.
    _PHASE_SCALAR_KEYS = (
        "t_a_layer",
        "t_f_layer",
        "t_a2f_layer",
        "t_f2a_layer",
        "t_c_layer",
        "t_step",
        "balance_ratio",
        "comm_hidden",
    )

    @classmethod
    def _phase_scalars(cls, metrics: dict | None) -> dict:
        """Return the 8 per-phase layer scalars (rounded), or NaN/None when
        the phase was not run.

        ``comm_hidden`` is the only boolean among them; surface it as
        ``None`` (cleanly serialized to ``NaN`` in DataFrames) when the
        phase did not run, so consumers can distinguish "the phase ran and
        comm was not hidden" from "the phase did not run at all".
        """
        if metrics is None:
            return {key: float("nan") if key != "comm_hidden" else None for key in cls._PHASE_SCALAR_KEYS}
        out: dict = {}
        for key in cls._PHASE_SCALAR_KEYS:
            value = metrics[key]
            if key == "comm_hidden":
                out[key] = bool(value)
            else:
                out[key] = round(float(value), 3)
        return out

    def _build_summary(
        self,
        *,
        runtime_config: config.RuntimeConfig,
        phase: str,
        prefill_metrics: dict | None,
        decode_metrics: dict | None,
    ) -> InferenceSummary:
        """Construct the InferenceSummary (result dict + per-ops + OOM flag).

        Output schema follows :data:`common.ColumnsAFD`. Per-phase layer
        scalars appear as paired ``prefill_<x>`` / ``decode_<x>`` columns
        plus an un-prefixed "headline" form. The un-prefixed form refuses
        to pick a single value in ``phase="both"`` (it is NaN/None) -- the
        paired columns are the source of truth in that mode.

        In AFD-with-PD combined runs the static side does not produce these
        scalars at all; ``_combine_afd_static_estimate_results`` inherits
        the AFD-side raw dict unchanged, so the NaN/None ``prefill_<x>`` or
        ``decode_<x>`` block produced here correctly flags the static side
        as non-AFD.
        """
        cfg = self._afd_config
        isl = runtime_config.isl
        osl = runtime_config.osl or 1
        (
            _a_total_batch_size,
            a_micro_batch_size,
            b_total,
            b_micro_total,
            num_microbatches,
        ) = self._afd_batch_shape()

        prefill_scalars = self._phase_scalars(prefill_metrics)
        decode_scalars = self._phase_scalars(decode_metrics)

        # Un-prefixed "headline" scalars: tied to the single ran phase in
        # single-phase mode, deliberately NaN/None in ``both`` mode so
        # downstream readers cannot accidentally treat the decode (or
        # prefill) value as covering both phases when the two estimates
        # generally disagree.
        if phase == "prefill":
            headline = prefill_scalars
        elif phase == "decode":
            headline = decode_scalars
        else:
            headline = self._phase_scalars(None)
        t_a_layer = headline["t_a_layer"]
        t_f_layer = headline["t_f_layer"]
        t_a2f_layer = headline["t_a2f_layer"]
        t_f2a_layer = headline["t_f2a_layer"]
        t_c_layer = headline["t_c_layer"]
        t_step = headline["t_step"]
        balance_ratio = headline["balance_ratio"]
        comm_hidden = headline["comm_hidden"]

        if decode_metrics is not None:
            tpot = decode_metrics["t_step"]
            tokens_per_s = b_total / (num_microbatches * (tpot / 1000.0)) if tpot > 0 else 0.0
        else:
            tpot = 0.0
            tokens_per_s = 0.0

        if prefill_metrics is not None:
            ttft = prefill_metrics["t_step"]
        else:
            ttft = 0.0

        if phase == "prefill":
            request_latency = ttft
        elif phase == "decode":
            request_latency = tpot * max(osl - 1, 0)
        else:  # both
            request_latency = ttft + tpot * max(osl - 1, 0)

        total_gpus = cfg.n_a_workers * cfg.tp_a + cfg.n_f_workers
        tokens_per_s_per_gpu = tokens_per_s / total_gpus if total_gpus > 0 else 0.0

        # HBM / OOM — take the worst of any simulated phase.
        def _max_memory_dict(key: str) -> dict[str, float]:
            vals = [m[key] for m in (prefill_metrics, decode_metrics) if m is not None]
            if not vals:
                return {
                    "total": 0.0,
                    "weights": 0.0,
                    "activations": 0.0,
                    "kvcache": 0.0,
                    "nccl": 0.0,
                    "others": 0.0,
                }
            return dict(max(vals, key=lambda item: item["total"]))

        def _any_oom(key: str) -> bool:
            return any(m[key] for m in (prefill_metrics, decode_metrics) if m is not None)

        a_memory = _max_memory_dict("a_memory")
        f_memory = _max_memory_dict("f_memory")
        a_memory_gb = a_memory["total"]
        f_memory_gb = f_memory["total"]
        a_is_oom = _any_oom("a_is_oom")
        f_is_oom = _any_oom("f_is_oom")
        a_is_kv_cache_oom = _any_oom("a_is_kv_cache_oom")
        f_is_kv_cache_oom = _any_oom("f_is_kv_cache_oom")
        is_oom = a_is_oom or f_is_oom

        tokens_per_s_per_user = (1000.0 / tpot) if tpot > 0 else 0.0
        seq_per_s = tokens_per_s / max(osl - 1, 1) if tokens_per_s > 0 else 0.0

        result_dict = {
            "model": self._model_path,
            "phase": phase,
            "isl": isl,
            "osl": osl,
            "gpus_per_node": cfg.gpus_per_node,
            "(a)nodes": cfg.n_a_nodes,
            "(a)tp": cfg.tp_a,
            "(a)bs": cfg.a_batch_size,
            "(a)micro_bs": a_micro_batch_size,
            "(a)workers": cfg.n_a_workers,
            "(a)memory": round(a_memory_gb, 2),
            "(a)is_oom": bool(a_is_oom),
            "(f)nodes": cfg.n_f_nodes,
            "(f)tp": cfg.tp_f,
            "(f)ep": cfg.f_moe_ep_size,
            "(f)workers": cfg.n_f_workers,
            "(f)memory": round(f_memory_gb, 2),
            "(f)is_oom": bool(f_is_oom),
            # Un-prefixed "headline" layer scalars: populated for the single
            # phase that ran; deliberately NaN/None when phase="both" so
            # downstream readers do not silently treat decode-only values
            # as covering both phases.  ``prefill_*`` / ``decode_*`` below
            # are the source of truth in that mode.
            "t_a_layer": t_a_layer,
            "t_f_layer": t_f_layer,
            "t_a2f_layer": t_a2f_layer,
            "t_f2a_layer": t_f2a_layer,
            "t_c_layer": t_c_layer,
            "t_step": t_step,
            "balance_ratio": balance_ratio,
            "comm_hidden": comm_hidden,
            "prefill_t_a_layer": prefill_scalars["t_a_layer"],
            "prefill_t_f_layer": prefill_scalars["t_f_layer"],
            "prefill_t_a2f_layer": prefill_scalars["t_a2f_layer"],
            "prefill_t_f2a_layer": prefill_scalars["t_f2a_layer"],
            "prefill_t_c_layer": prefill_scalars["t_c_layer"],
            "prefill_t_step": prefill_scalars["t_step"],
            "prefill_balance_ratio": prefill_scalars["balance_ratio"],
            "prefill_comm_hidden": prefill_scalars["comm_hidden"],
            "decode_t_a_layer": decode_scalars["t_a_layer"],
            "decode_t_f_layer": decode_scalars["t_f_layer"],
            "decode_t_a2f_layer": decode_scalars["t_a2f_layer"],
            "decode_t_f2a_layer": decode_scalars["t_f2a_layer"],
            "decode_t_c_layer": decode_scalars["t_c_layer"],
            "decode_t_step": decode_scalars["t_step"],
            "decode_balance_ratio": decode_scalars["balance_ratio"],
            "decode_comm_hidden": decode_scalars["comm_hidden"],
            "ttft": round(ttft, 3),
            "tpot": round(tpot, 3),
            "request_latency": round(request_latency, 3),
            "b_total": b_total,
            "b_micro_total": b_micro_total,
            "tokens/s": round(tokens_per_s, 2),
            "tokens/s/gpu": round(tokens_per_s_per_gpu, 2),
            "tokens/s/user": round(tokens_per_s_per_user, 2),
            "seq/s": round(seq_per_s, 3),
            # ``a_batch_size`` is the total in-flight batch per A-Worker;
            # latency is evaluated on the derived microbatch, while the
            # user-visible concurrency remains the total in-flight batch.
            "concurrency": b_total,
            "pipeline_model": cfg.pipeline_model,
            "num_microbatches": num_microbatches,
            "combined_with_pd": bool(cfg.combined_with_pd),
            "boundary_on_attn": bool(cfg.boundary_on_attn),
            "num_total_gpus": total_gpus,
            "memory": round(max(a_memory_gb, f_memory_gb), 2),
            "backend": self._backend.name.value,
            "version": str(self._database.version),
            "system": str(self._database.system),
            "power_w": 0.0,
        }

        summary_df = pd.DataFrame([result_dict], columns=common.ColumnsAFD)
        summary = InferenceSummary(runtime_config)
        summary_memory = dict(a_memory if a_memory_gb >= f_memory_gb else f_memory)
        summary.set_memory_and_check_oom(
            summary_memory,
            self._database.system_spec["gpu"]["mem_capacity"],
        )
        summary.set_oom(bool(is_oom))
        summary.set_kv_cache_oom(bool(a_is_kv_cache_oom or f_is_kv_cache_oom))
        summary.set_summary_df(summary_df)
        summary.set_result_dict(result_dict)

        # Per-ops breakdown by phase / pool.  AFD inserts two transfer
        # ops per layer (A→F and F→A); both per-direction values are
        # surfaced here alongside the round-trip total ``t_c_layer``.
        per_ops_data: dict = {}
        if prefill_metrics is not None:
            per_ops_data["prefill_a_worker"] = prefill_metrics["a_per_op"]
            per_ops_data["prefill_f_worker"] = prefill_metrics["f_per_op"]
            comm = per_ops_data.setdefault("comm", {})
            comm["prefill_afd_transfer_a2f"] = prefill_metrics["t_a2f_layer"]
            comm["prefill_afd_transfer_f2a"] = prefill_metrics["t_f2a_layer"]
            comm["prefill_afd_transfer"] = prefill_metrics["t_c_layer"]
        if decode_metrics is not None:
            per_ops_data["decode_a_worker"] = decode_metrics["a_per_op"]
            per_ops_data["decode_f_worker"] = decode_metrics["f_per_op"]
            comm = per_ops_data.setdefault("comm", {})
            comm["decode_afd_transfer_a2f"] = decode_metrics["t_a2f_layer"]
            comm["decode_afd_transfer_f2a"] = decode_metrics["t_f2a_layer"]
            comm["decode_afd_transfer"] = decode_metrics["t_c_layer"]
        memory_breakdown = {
            "a_worker": a_memory,
            "f_worker": f_memory,
            "a_is_kv_cache_oom": bool(a_is_kv_cache_oom),
            "f_is_kv_cache_oom": bool(f_is_kv_cache_oom),
        }
        if prefill_metrics is not None:
            memory_breakdown["prefill"] = {
                "a_worker": prefill_metrics["a_memory"],
                "f_worker": prefill_metrics["f_memory"],
            }
        if decode_metrics is not None:
            memory_breakdown["decode"] = {
                "a_worker": decode_metrics["a_memory"],
                "f_worker": decode_metrics["f_memory"],
            }
        per_ops_data["memory"] = memory_breakdown
        summary.set_per_ops_data(per_ops_data)

        return summary
