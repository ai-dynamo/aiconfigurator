# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import functools
import logging
import warnings
from collections import defaultdict

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
from aiconfigurator.sdk.utils import enumerate_ttft_tpot_constraints

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
            self._model, self._database, runtime_config, mode, stride, latency_correction_scale
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
        parallel_config_list: list[tuple[int, int, int, int, int]],
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
            parallel_config_list: List of (tp, pp, dp, moe_tp, moe_ep) tuples.
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
            tp_size, pp_size, dp_size, moe_tp_size, moe_ep_size = parallel_config
            logger.debug(
                "Getting candidate workers with parallel config: tp=%d, pp=%d, dp=%d, moe_tp=%d, moe_ep=%d",
                tp_size,
                pp_size,
                dp_size,
                moe_tp_size,
                moe_ep_size,
            )

            try:
                overwritten_model_config = copy.deepcopy(model_config)
                overwritten_model_config.pp_size = pp_size
                overwritten_model_config.tp_size = tp_size
                overwritten_model_config.moe_tp_size = moe_tp_size
                overwritten_model_config.moe_ep_size = moe_ep_size
                overwritten_model_config.attention_dp_size = dp_size
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
        prefill_parallel_config_list: list[tuple[int, int, int, int, int]],
        prefill_max_num_tokens: int,
        prefill_num_worker_list: list[int],
        decode_model_config: config.ModelConfig,
        decode_parallel_config_list: list[tuple[int, int, int, int, int]],
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

        if prefill_max_num_tokens < runtime_config.isl:
            logger.warning("prefill_max_num_tokens is less than runtime_config.isl, set to runtime_config.isl")
            prefill_max_num_tokens = runtime_config.isl

        max_prefill_batch_size = prefill_max_num_tokens // runtime_config.isl
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

    def _build_afd_transfer_op(self, a_model, f_model, *, rank_mapping: str = "one_to_one"):
        """Construct the single AFDTransfer op covering all AFD comm.

        ``AFDTransfer.query()`` returns a breakdown dict with the
        cross-pool A→F / F→A latencies plus the F-side intra-node AG/RS
        and A-side combine, so one op per layer is sufficient.

        ``rank_mapping`` selects the dispatch topology:
        ``"one_to_one"`` (current default) keeps the F-side AG/RS;
        ``"broadcast"`` reports them as 0 (placeholder for future
        modeling of A-rank → all-F-ranks fan-out).
        """
        from aiconfigurator.sdk.operations import AFDTransfer

        cfg = self._afd_config
        is_moe = hasattr(f_model, "_topk") and hasattr(f_model, "_num_experts")
        transfer_mode = "moe_selective" if (is_moe and cfg.f_moe_ep_size > 1) else "p2p"
        return AFDTransfer(
            name="afd_transfer",
            hidden_size=a_model._hidden_size,
            n_a_workers=cfg.n_a_workers,
            n_f_workers=cfg.n_f_workers,
            gpus_per_node=cfg.gpus_per_node,
            tp_a=cfg.tp_a,
            tp_f=cfg.tp_f,
            f_moe_ep_size=cfg.f_moe_ep_size,
            topk=getattr(f_model, "_topk", 1),
            num_experts=getattr(f_model, "_num_experts", 1),
            comm_quant_mode=self._a_model_config.comm_quant_mode,
            comm_overhead_factor=cfg.comm_overhead_factor,
            transfer_mode=transfer_mode,
            rank_mapping=rank_mapping,
        )

    def _pipeline_tcycle(
        self, t_a: float, t_f: float, t_a2f: float, t_f2a: float
    ) -> tuple[float, bool]:
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
        t_c = t_a2f + t_f2a
        if cfg.pipeline_model == "optimistic":
            # Need ≥ 2 + t_c / max(t_a, t_f) in-flight microbatches to
            # hide the network stage behind compute.  Equivalent to the
            # legacy ``2 * (1 + t_c_one_dir / max(...))`` under the
            # symmetric Phase-1 assumption (t_a2f == t_f2a).
            min_m = 2.0 + t_c / max(t_a, t_f, 1e-9)
            if cfg.num_microbatches < min_m:
                logger.warning(
                    "AFD optimistic pipeline: num_microbatches (%d) < min required (%.1f) "
                    "to hide communication. Falling back to conservative model.",
                    cfg.num_microbatches,
                    min_m,
                )
                return max(t_a + t_a2f, t_f + t_f2a), False
            t_cycle = max(t_a, t_f, t_c)
            comm_hidden = t_c <= max(t_a, t_f)
            return t_cycle, comm_hidden
        # conservative K=2
        return max(t_a + t_a2f, t_f + t_f2a), False

    def _estimate_a_memory_gb(
        self,
        *,
        a_model,
        a_partition,
        phase: str,
        batch_size: int,
        isl: int,
        osl: int,
    ) -> float:
        """Estimate A-Worker per-GPU HBM usage in GiB.

        * Weights: sum of attention-side op weights (per-GPU after TP shard).
        * KV cache:
            - ``phase == "prefill"``: transient KV for a single prefill sequence
              (grows to ISL).
            - ``phase == "decode"``:  persistent KV for (isl+osl) tokens per
              sequence, concurrently held for ``num_microbatches`` in-flight batches.
            - ``phase == "both"``: take the decode estimate (worst case).
        """
        cfg = self._afd_config
        num_layers = a_model._num_layers
        a_weights_gb = sum(op.get_weights() for op in a_partition.attn_ops) / (1 << 30)

        kv_bytes_per_element = (
            self._a_model_config.kvcache_quant_mode.value.memory
            if self._a_model_config.kvcache_quant_mode
            else 2
        )
        kv_per_token = (
            2 * a_model._num_kv_heads * a_model._head_size * kv_bytes_per_element / max(cfg.tp_a, 1)
        )

        if phase == "prefill":
            kv_tokens_per_seq = isl
            kv_multiplier = 1  # no persistent in-flight replicas for pure prefill
        else:  # decode or both → worst-case decode KV
            kv_tokens_per_seq = isl + osl
            kv_multiplier = max(cfg.num_microbatches, 1)

        kv_cache_gb = (
            num_layers * batch_size * kv_tokens_per_seq * kv_per_token * kv_multiplier
        ) / (1 << 30)
        return a_weights_gb + kv_cache_gb

    def _estimate_f_memory_gb(self, *, f_partition) -> float:
        """Estimate F-Worker per-GPU HBM usage in GiB (weights only; FFN is stateless)."""
        return sum(op.get_weights() for op in f_partition.ffn_ops) / (1 << 30)

    def _gpu_mem_capacity_gb(self) -> float:
        mem_capacity = self._database.system_spec.get("gpu", {}).get("mem_capacity", 80 * (1 << 30))
        return mem_capacity / (1 << 30)

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
        b_total: int,
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

        ``brk_t_a_per_layer`` / ``brk_t_f_per_layer`` are the
        AFDTransfer per-layer intra-pool contributions (s-independent);
        they are folded into ``t_a_layer_i`` / ``t_f_layer_i`` *before*
        the per-step ``_pipeline_tcycle`` call so the pipeline max is
        evaluated on the full per-layer time, not on the compute-only
        time.
        """
        cfg = self._afd_config
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
                batch_size=cfg.a_batch_size,
                seq_len=s_i,
                model=a_model,
                runtime_config=runtime_config,
                is_context=False,
            )
            t_f_step_i, f_per_op_i = self._sum_latency(
                f_partition.ffn_ops,
                batch_size=b_total,
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
            t_cycle_i, _ = self._pipeline_tcycle(
                t_a_layer_i, t_f_layer_i, t_a2f_layer, t_f2a_layer
            )
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
    ) -> dict:
        """Simulate one phase (prefill or decode) and return a metrics dict.

        Keys: ``t_a_layer``, ``t_f_layer``, ``t_a2f_layer``,
        ``t_f2a_layer``, ``t_c_layer`` (round-trip = t_a2f + t_f2a),
        ``t_cycle``, ``t_step``, ``comm_hidden``, ``balance_ratio``,
        ``a_per_op``, ``f_per_op``, ``a_memory_gb``, ``f_memory_gb``,
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
        a_partition = build_afd_ops_partition(
            a_model, phase=ops_phase, boundary_on_attn=cfg.boundary_on_attn
        )
        f_partition = build_afd_ops_partition(
            f_model, phase=ops_phase, boundary_on_attn=cfg.boundary_on_attn
        )

        isl = runtime_config.isl
        osl = runtime_config.osl or 1
        num_layers = max(int(getattr(a_model, "_num_layers", 1)), 1)

        # A-Worker sees its local batch (a_batch_size); F-Worker sees the full
        # combined batch post-AllGather.
        b_total = cfg.n_a_workers * cfg.a_batch_size

        # AFD cross-pool comm (AFDTransfer, F-node AllGather/ReduceScatter,
        # A-side combine) bills by *token* volume per step, not request
        # count.  In prefill each request contributes ``isl`` tokens per
        # layer; in decode it contributes 1 token per step.  Pre-compute
        # the per-step token totals to feed into the AFD ops so the
        # per-link byte volume is correctly scaled by sequence length
        # (without this, prefill comm latency is under-estimated by ``isl``).
        tokens_per_req = isl if phase == "prefill" else 1
        afd_b_tokens = b_total * tokens_per_req
        afd_a_batch_tokens = cfg.a_batch_size * tokens_per_req

        # Single AFDTransfer op covers cross-pool A→F / F→A and the
        # F-side intra-node AG/RS + A-side combine.  ``query()`` returns
        # a per-layer latency breakdown keyed by ``t_a2f`` / ``t_f2a``
        # (cross-pool, *one-direction*) and ``t_a`` / ``t_f`` (intra-pool
        # additions to each side's per-layer compute).  AFDTransfer
        # bills by token volume only — it is independent of ``s`` — so
        # we query it once outside the decode stride loop.
        transfer_op = self._build_afd_transfer_op(a_model, f_model)
        brk = transfer_op.query(
            self._database,
            b_total=afd_b_tokens,
            a_batch_size=afd_a_batch_tokens,
        )
        t_a2f_layer = sum(brk["t_a2f"].values())
        t_f2a_layer = sum(brk["t_f2a"].values())
        t_c_layer = t_a2f_layer + t_f2a_layer
        brk_t_a_per_layer = sum(brk["t_a"].values())
        brk_t_f_per_layer = sum(brk["t_f"].values())

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
                b_total=b_total,
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
            _, comm_hidden = self._pipeline_tcycle(
                t_a_layer, t_f_layer, t_a2f_layer, t_f2a_layer
            )
        else:
            # Prefill: single shot over the full input sequence; no
            # need to integrate, ``s == isl`` everywhere.
            seq_len_query = max(isl, 1)
            t_a_total, a_per_op = self._sum_latency(
                a_partition.attn_ops,
                batch_size=cfg.a_batch_size,
                seq_len=seq_len_query,
                model=a_model,
                runtime_config=runtime_config,
                is_context=True,
            )
            t_f_total, f_per_op = self._sum_latency(
                f_partition.ffn_ops,
                batch_size=b_total,
                seq_len=seq_len_query,
                model=f_model,
                runtime_config=runtime_config,
                is_context=True,
            )
            t_a_layer = t_a_total / num_layers + brk_t_a_per_layer
            t_f_layer = t_f_total / num_layers + brk_t_f_per_layer
            t_cycle, comm_hidden = self._pipeline_tcycle(
                t_a_layer, t_f_layer, t_a2f_layer, t_f2a_layer
            )
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
        a_memory_gb = self._estimate_a_memory_gb(
            a_model=a_model,
            a_partition=a_partition,
            phase=phase,
            batch_size=cfg.a_batch_size,
            isl=isl,
            osl=osl,
        )
        f_memory_gb = self._estimate_f_memory_gb(f_partition=f_partition)
        cap_gb = self._gpu_mem_capacity_gb()

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
            "a_memory_gb": a_memory_gb,
            "f_memory_gb": f_memory_gb,
            "a_is_oom": a_memory_gb >= cap_gb,
            "f_is_oom": f_memory_gb >= cap_gb,
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

        Returns:
            InferenceSummary.  ``check_oom()`` reflects the per-pool HBM check.
        """
        cfg = self._afd_config
        phase = phase if phase is not None else cfg.phase
        if phase not in ("prefill", "decode", "both"):
            raise ValueError(f"AFDInferenceSession.run_afd: invalid phase {phase!r}")

        a_model, f_model = self._build_models()

        prefill_metrics = None
        decode_metrics = None
        if phase in ("prefill", "both"):
            prefill_metrics = self._simulate_phase(
                phase="prefill",
                runtime_config=runtime_config,
                a_model=a_model,
                f_model=f_model,
            )
        if phase in ("decode", "both"):
            decode_metrics = self._simulate_phase(
                phase="decode",
                runtime_config=runtime_config,
                a_model=a_model,
                f_model=f_model,
            )

        return self._build_summary(
            runtime_config=runtime_config,
            phase=phase,
            prefill_metrics=prefill_metrics,
            decode_metrics=decode_metrics,
        )

    def run_afd_decode(self, runtime_config: config.RuntimeConfig) -> InferenceSummary:
        """Backwards-compatible Decode-only entry point."""
        return self.run_afd(runtime_config, phase="decode")

    def run_afd_prefill(self, runtime_config: config.RuntimeConfig) -> InferenceSummary:
        """Prefill-only AFD entry point.

        Uses the same A/F split as decode but applies it to ``context_ops``,
        producing a TTFT estimate.  No persistent KV cache is required —
        the A-Worker HBM estimate tracks the transient prefill KV only.
        """
        return self.run_afd(runtime_config, phase="prefill")

    def _build_summary(
        self,
        *,
        runtime_config: config.RuntimeConfig,
        phase: str,
        prefill_metrics: dict | None,
        decode_metrics: dict | None,
    ) -> InferenceSummary:
        """Construct the InferenceSummary (result dict + per-ops + OOM flag).

        Output schema follows :data:`common.ColumnsAFD`. Metrics that do not
        apply to the current phase are populated with zero.
        """
        cfg = self._afd_config
        isl = runtime_config.isl
        osl = runtime_config.osl or 1
        b_total = cfg.n_a_workers * cfg.a_batch_size

        # Pick the metrics that drive latency/throughput headline numbers.
        drive = decode_metrics if decode_metrics is not None else prefill_metrics
        assert drive is not None, "At least one phase must be simulated"

        # Per-phase scalar metrics (zero-fill the ones we didn't run).
        t_a_layer = drive["t_a_layer"]
        t_f_layer = drive["t_f_layer"]
        t_a2f_layer = drive["t_a2f_layer"]
        t_f2a_layer = drive["t_f2a_layer"]
        t_c_layer = drive["t_c_layer"]
        t_step = drive["t_step"]
        comm_hidden = drive["comm_hidden"]
        balance_ratio = drive["balance_ratio"]

        if decode_metrics is not None:
            tpot = decode_metrics["t_step"]
            tokens_per_s = b_total / (tpot / 1000.0) if tpot > 0 else 0.0
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
            request_latency = tpot * max(osl - 1, 1) if tpot > 0 else tpot
        else:  # both
            request_latency = ttft + tpot * max(osl - 1, 0)

        total_gpus = cfg.n_a_workers * cfg.tp_a + cfg.n_f_workers
        tokens_per_s_per_gpu = tokens_per_s / total_gpus if total_gpus > 0 else 0.0

        # HBM / OOM — take the worst of any simulated phase.
        def _max_mem(key: str) -> float:
            vals = [m[key] for m in (prefill_metrics, decode_metrics) if m is not None]
            return max(vals) if vals else 0.0

        def _any_oom(key: str) -> bool:
            return any(m[key] for m in (prefill_metrics, decode_metrics) if m is not None)

        a_memory_gb = _max_mem("a_memory_gb")
        f_memory_gb = _max_mem("f_memory_gb")
        a_is_oom = _any_oom("a_is_oom")
        f_is_oom = _any_oom("f_is_oom")
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
            "(a)workers": cfg.n_a_workers,
            "(a)memory": round(a_memory_gb, 2),
            "(a)is_oom": bool(a_is_oom),
            "(f)nodes": cfg.n_f_nodes,
            "(f)tp": cfg.tp_f,
            "(f)ep": cfg.f_moe_ep_size,
            "(f)workers": cfg.n_f_workers,
            "(f)memory": round(f_memory_gb, 2),
            "(f)is_oom": bool(f_is_oom),
            "t_a_layer": round(t_a_layer, 3),
            "t_f_layer": round(t_f_layer, 3),
            "t_a2f_layer": round(t_a2f_layer, 3),
            "t_f2a_layer": round(t_f2a_layer, 3),
            "t_c_layer": round(t_c_layer, 3),
            "t_step": round(t_step, 3),
            "ttft": round(ttft, 3),
            "tpot": round(tpot, 3),
            "request_latency": round(request_latency, 3),
            "b_total": b_total,
            "tokens/s": round(tokens_per_s, 2),
            "tokens/s/gpu": round(tokens_per_s_per_gpu, 2),
            "tokens/s/user": round(tokens_per_s_per_user, 2),
            "seq/s": round(seq_per_s, 3),
            "concurrency": b_total,
            "tpuc": round(tokens_per_s_per_gpu, 2),
            "balance_ratio": round(balance_ratio, 3),
            "comm_hidden": comm_hidden,
            "pipeline_model": cfg.pipeline_model,
            "num_microbatches": cfg.num_microbatches,
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
        summary.set_summary_df(summary_df)
        summary.set_result_dict(result_dict)

        summary.set_oom(bool(is_oom))

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
        summary.set_per_ops_data(per_ops_data)

        return summary
