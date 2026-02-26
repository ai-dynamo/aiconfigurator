# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import functools
import logging
import warnings

import pandas as pd

from aiconfigurator.sdk import common, config, models, perf_database
from aiconfigurator.sdk.backends.base_backend import BaseBackend
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
    ) -> None:
        """
        Initialize the DisaggInferenceSession
        """
        self._prefill_database = prefill_database
        self._prefill_backend = prefill_backend
        self._decode_database = decode_database
        self._decode_backend = decode_backend

        # allow user to set correction scales for better alignment with real system
        # now the corection scales are used to correct the latency, not throughput,
        # corrected latency = latency * correction_scale
        self._prefill_latency_correction_scale = 1.0
        self._decode_latency_correction_scale = 1.0

    def set_latency_correction_scales(
        self, prefill_latency_correction_scale: float, decode_latency_correction_scale: float
    ):
        """
        Set the correction scales for better alignment with real system
        """
        self._prefill_latency_correction_scale = prefill_latency_correction_scale
        self._decode_latency_correction_scale = decode_latency_correction_scale

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
        decode_dict = decode_summary_df.iloc[0].to_dict()

        summary_dict = _build_disagg_summary_dict(
            prefill_dict,
            prefill_num_worker,
            decode_dict,
            decode_num_worker,
            prefill_degradation_factor=_RATE_MATCHING_PREFILL_DEGRADATION_FACTOR,
            decode_degradation_factor=_RATE_MATCHING_DECODE_DEGRADATION_FACTOR,
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
        prefill_summary = prefill_sess.run_static(mode="static_ctx", runtime_config=prefill_runtime_config)
        decode_runtime_config = copy.deepcopy(runtime_config)
        decode_runtime_config.batch_size = decode_batch_size
        decode_summary = decode_sess.run_static(mode="static_gen", runtime_config=decode_runtime_config)
        disagg_summary_df = self._get_disagg_summary_df(
            prefill_summary.get_summary_df(),
            prefill_num_worker,
            decode_summary.get_summary_df(),
            decode_num_worker,
        )

        disagg_summary = InferenceSummary(runtime_config=runtime_config)
        disagg_summary.set_summary_df(disagg_summary_df)

        # Carry per-op latency breakdowns from prefill/decode static runs
        per_ops_data = {}
        prefill_ctx_latency = prefill_summary.get_context_latency_dict()
        if prefill_ctx_latency:
            per_ops_data["prefill"] = dict(prefill_ctx_latency)
        decode_gen_latency = decode_summary.get_generation_latency_dict()
        if decode_gen_latency:
            per_ops_data["decode"] = dict(decode_gen_latency)
        if per_ops_data:
            disagg_summary.set_per_ops_data(per_ops_data)

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
            raise RuntimeError(
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
                rate_matching_prefill_degradation_factor=_RATE_MATCHING_PREFILL_DEGRADATION_FACTOR,
                rate_matching_decode_degradation_factor=_RATE_MATCHING_DECODE_DEGRADATION_FACTOR,
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
