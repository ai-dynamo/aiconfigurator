# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from collections import defaultdict

import numpy as np
import pandas as pd

from aiconfigurator.sdk import common
from aiconfigurator.sdk.backends.base_backend import BaseBackend
from aiconfigurator.sdk.config import RuntimeConfig
from aiconfigurator.sdk.inference_summary import InferenceSummary
from aiconfigurator.sdk.models import BaseModel
from aiconfigurator.sdk.perf_database import PerfDatabase

logger = logging.getLogger(__name__)


class TRTLLMBackend(BaseBackend):
    """
    TRTLLM backend.
    """

    def __init__(
        self,
    ):
        super().__init__()
        self._agg_cache = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))
        self.name = common.BackendName.trtllm

    def run_agg(
        self, model: BaseModel, database: PerfDatabase, runtime_config: RuntimeConfig, **kwargs
    ) -> InferenceSummary:
        """
        Run the agg inference.
        """
        isl = runtime_config.isl
        osl = runtime_config.osl
        prefix = runtime_config.prefix
        b = runtime_config.batch_size
        ctx_tokens = kwargs.get("ctx_tokens")
        assert ctx_tokens is not None, "ctx_tokens is required"
        balance_score = isl * b / ctx_tokens / osl

        try:
            summary = self._agg_cache[isl][osl][b][ctx_tokens]
        except KeyError:
            # we would like to calculate num_mix_steps and num_genonly_steps based on
            # isl, osl, b, ctx_tokens within osl steps, need to finish all the ctx tokens
            steps_to_finish_ctx = np.ceil(isl * b / ctx_tokens)
            num_mix_steps = num_genonly_steps = 0
            num_mix_steps_for_tpot_calc = 0  # this is a correction for tpot calc only.
            if b > 1:
                if steps_to_finish_ctx >= osl:
                    num_mix_steps = steps_to_finish_ctx
                    num_mix_ctx_tokens = ctx_tokens
                    num_mix_gen_tokens = max(1, b // (steps_to_finish_ctx / osl))
                    num_genonly_steps = 0
                    num_genonly_tokens = 0
                    num_mix_steps_for_tpot_calc = num_mix_steps
                else:
                    # 3-step is an empirical correction for pipelining requests where new requests
                    # cannot be enqueued immediately after last request's exit
                    num_mix_steps = steps_to_finish_ctx
                    num_mix_ctx_tokens = ctx_tokens
                    num_mix_gen_tokens = b - np.ceil(ctx_tokens / isl)  # the error check is outside
                    assert num_mix_gen_tokens >= 1, (
                        f"num_mix_gen_tokens: {num_mix_gen_tokens}, b: {b}, ctx_tokens: {ctx_tokens}, isl: {isl}"
                    )
                    num_genonly_steps = osl - num_mix_steps
                    num_genonly_tokens = b
                    num_mix_steps_for_tpot_calc = max(1, num_mix_steps - 3)
            elif b == 1:
                # special case for b=1
                num_mix_steps = 1
                num_mix_ctx_tokens = ctx_tokens
                num_mix_gen_tokens = 0
                num_genonly_steps = osl - 1
                num_genonly_tokens = 1
                num_mix_steps_for_tpot_calc = 0

            # Per-ops latency collection
            per_ops_data = {}

            # FIXME, fix for DS. DS has different ops for attn in ctx and gen.
            def _get_mix_step_latency(
                model: BaseModel,
                database: PerfDatabase,
                ctx_tokens: int,
                gen_tokens: int,
                isl: int,
                osl: int,
                prefix: int,
            ) -> tuple[float, float]:
                """
                Get mixed step latency and energy.

                Returns:
                    tuple: (latency in ms, energy in watt-milliseconds)
                """
                num_tokens = ctx_tokens + gen_tokens
                # treat this as a combined single batch inference, extract non-attention latency
                summary = self.run_static(
                    model,
                    database,
                    # num tokens for gemm needs to be adjusted for prefix, depends on the avg prefix len per request
                    RuntimeConfig(
                        batch_size=1, beam_width=1, isl=num_tokens, osl=1, prefix=prefix * np.floor(ctx_tokens / isl)
                    ),
                    mode="static_ctx",
                )
                latency_dict = summary.get_context_latency_dict()
                energy_wms_dict = summary.get_context_energy_wms_dict()  # CHANGED from get_context_power_dict()
                non_attention_latency_ms = 0.0
                non_attention_energy_wms = 0.0  # RENAMED from non_attention_power_weighted
                mix_non_attn_ops = {}
                for layer_name, latency in latency_dict.items():
                    if layer_name != "context_attention":
                        non_attention_latency_ms += latency
                        non_attention_energy_wms += energy_wms_dict.get(layer_name, 0.0)  # SIMPLIFIED
                        mix_non_attn_ops[layer_name] = latency

                # second pass to get ctx attn, split full isl over
                # num_steps(=np.ceil(isl/ctx_tokens))
                # average the ctx attn latency with num_steps to get the ctx_attention_latency
                num_tokens = isl
                batch_size = np.ceil(ctx_tokens / isl)
                summary = self.run_static(
                    model,
                    database,
                    RuntimeConfig(batch_size=batch_size, beam_width=1, isl=num_tokens, osl=1, prefix=prefix),
                    mode="static_ctx",
                )
                latency_dict = summary.get_context_latency_dict()
                energy_wms_dict = summary.get_context_energy_wms_dict()
                scale_factor = np.ceil(isl / ctx_tokens)
                ctx_attention_latency_ms = latency_dict["context_attention"] / scale_factor
                ctx_attention_energy_wms = energy_wms_dict.get("context_attention", 0.0) / scale_factor  # CHANGED

                # third pass to get generation attn. use isl+osl//2 for avg generation attn latency.
                gen_attention_latency_ms = 0.0
                gen_attention_energy_wms = 0.0  # RENAMED from gen_attention_power_weighted
                if gen_tokens > 0:
                    num_tokens = gen_tokens
                    summary = self.run_static(
                        model,
                        database,
                        RuntimeConfig(batch_size=num_tokens, beam_width=1, isl=isl + osl // 2, osl=2),
                        mode="static_gen",
                    )
                    latency_dict = summary.get_generation_latency_dict()
                    energy_wms_dict = summary.get_generation_energy_wms_dict()
                    gen_attention_latency_ms = latency_dict["generation_attention"]
                    gen_attention_energy_wms = energy_wms_dict.get("generation_attention", 0.0)  # CHANGED

                # Collect per-op breakdown for mix step
                per_ops_data["mix_step"] = {
                    **mix_non_attn_ops,
                    "context_attention (scaled)": ctx_attention_latency_ms,
                    "generation_attention": gen_attention_latency_ms,
                }

                # Combine all components (simple addition)
                total_latency_ms = non_attention_latency_ms + ctx_attention_latency_ms + gen_attention_latency_ms
                total_energy_wms = non_attention_energy_wms + ctx_attention_energy_wms + gen_attention_energy_wms

                return total_latency_ms, total_energy_wms

            def _get_genonly_step_latency(
                model: BaseModel, database: PerfDatabase, gen_tokens: int, isl: int, osl: int
            ) -> tuple[float, float]:
                """
                Get generation-only step latency and energy.

                Returns:
                    tuple: (latency in ms, energy in watt-milliseconds)
                """
                if gen_tokens <= 0:
                    return 0.0, 0.0
                num_tokens = gen_tokens
                summary = self.run_static(
                    model,
                    database,
                    RuntimeConfig(batch_size=num_tokens, beam_width=1, isl=isl + osl // 2, osl=2),
                    mode="static_gen",
                )
                latency_dict = summary.get_generation_latency_dict()
                energy_wms_dict = summary.get_generation_energy_wms_dict()  # CHANGED
                genonly_step_latency_ms = 0.0
                genonly_step_energy_wms = 0.0  # RENAMED from genonly_power_weighted
                genonly_ops = {}
                for layer_name, latency in latency_dict.items():
                    genonly_step_latency_ms += latency
                    genonly_step_energy_wms += energy_wms_dict.get(layer_name, 0.0)  # SIMPLIFIED
                    genonly_ops[layer_name] = latency

                per_ops_data["genonly_step"] = genonly_ops

                return genonly_step_latency_ms, genonly_step_energy_wms

            # Call helpers (now return energy in W·ms instead of power)
            mix_step_latency_ms, mix_step_energy_wms = _get_mix_step_latency(
                model, database, num_mix_ctx_tokens, num_mix_gen_tokens, isl, osl, prefix
            )
            genonly_step_latency_ms, genonly_step_energy_wms = _get_genonly_step_latency(
                model, database, num_genonly_tokens, isl, osl
            )

            # Calculate timing (unchanged)
            ttft = mix_step_latency_ms * np.ceil(isl / ctx_tokens)
            # correction for ttft in trtllm agg mode, assume we have requests 10x of concurrency
            # (batch size here) to mitigate the impact of first round latency
            # assume we need to increase x of requests when concurrency gets larger.
            # thus capped to 4 to make it reasonable.
            correction_factor = min(2 + (steps_to_finish_ctx - 3) / 2 / 10, 4)
            ttft *= correction_factor
            logger.debug(
                f"ttft correction factor: {2 + (steps_to_finish_ctx - 3) / 2 / 10} capped to "
                f"{correction_factor} when b: {b}, ctx_tokens: {ctx_tokens} isl {isl}"
            )

            tpot = (mix_step_latency_ms * num_mix_steps_for_tpot_calc + genonly_step_latency_ms * num_genonly_steps) / (
                num_mix_steps_for_tpot_calc + num_genonly_steps
            )
            output_throughput = (
                1000
                / (num_mix_steps * mix_step_latency_ms + num_genonly_steps * genonly_step_latency_ms)
                * b
                * (osl - 1)
            )
            logger.debug(
                f"ctx_tokens: {ctx_tokens}, b: {b}, osl: {osl}, isl: {isl}, "
                f"num_mix_steps: {num_mix_steps}, num_genonly_steps: {num_genonly_steps}, "
                f"num_mix_ctx_tokens: {num_mix_ctx_tokens}, "
                f"num_mix_gen_tokens: {num_mix_gen_tokens}, "
                f"num_genonly_tokens: {num_genonly_tokens}"
            )
            logger.debug(
                f"mix_step_latency: {mix_step_latency_ms} ms, genonly_step_latency: {genonly_step_latency_ms} ms"
            )
            logger.debug(
                f"mix_step_energy: {mix_step_energy_wms} W·ms, genonly_step_energy: {genonly_step_energy_wms} W·ms"
            )
            logger.debug(f"ttft: {ttft}, tpot: {tpot}, output_throughput: {output_throughput}")

            # Calculate weighted average power (SIMPLIFIED!)
            # Step 1: Calculate total energy (simple multiplication and addition)
            total_mix_energy_wms = num_mix_steps * mix_step_energy_wms
            total_genonly_energy_wms = num_genonly_steps * genonly_step_energy_wms
            total_energy_wms = total_mix_energy_wms + total_genonly_energy_wms

            # Step 2: Calculate total latency (simple multiplication and addition)
            total_latency_ms = num_mix_steps * mix_step_latency_ms + num_genonly_steps * genonly_step_latency_ms

            # Step 3: Derive average power (single division)
            if total_latency_ms > 0:
                agg_power_avg_w = total_energy_wms / total_latency_ms
            else:
                agg_power_avg_w = 0.0

            logger.debug(f"Aggregated power: {agg_power_avg_w}W (from {total_energy_wms} W·ms / {total_latency_ms} ms)")

            num_ctx_requests = np.ceil(ctx_tokens / isl)
            num_gen_requests = b - num_ctx_requests
            if b == 1:
                num_ctx_requests = 1
                num_gen_requests = 1

            # correct output_throughput and concurrency for attention dp (global batch)
            scale_factor = model.config.pp_size * model.config.attention_dp_size
            output_throughput = output_throughput * scale_factor
            concurrency = b * scale_factor

            request_rate = output_throughput / (osl - 1)
            if b > 1:
                # will not be corrected by balance score when it's larger than 1.0
                # in order to indicate what's happening
                num_tokens = num_gen_requests + ctx_tokens
            else:
                num_tokens = ctx_tokens
            memory = self._get_memory_usage(model, database, b, 1, isl, osl, num_tokens)
            tp = model.config.tp_size
            pp = model.config.pp_size
            dp = model.config.attention_dp_size
            moe_tp = model.config.moe_tp_size
            moe_ep = model.config.moe_ep_size
            tokens_s_gpu = output_throughput / pp / tp / dp
            tokens_s_user = 1000 / tpot
            seq_s = request_rate
            seq_s_gpu = seq_s / pp / tp / dp
            tokens_s = output_throughput
            request_latency = ttft + tpot * max(osl - 1, 0)
            num_total_gpus = tp * pp * dp
            parallel = f"tp{tp}pp{pp}dp{dp}etp{moe_tp}ep{moe_ep}"
            gemm = model.config.gemm_quant_mode.name
            kvcache = model.config.kvcache_quant_mode.name
            fmha = model.config.fmha_quant_mode.name
            moe = model.config.moe_quant_mode.name
            comm = model.config.comm_quant_mode.name
            mem = memory["total"]

            result_dict = {
                "model": model.model_path,
                "isl": isl,
                "osl": osl,
                "prefix": prefix,
                "concurrency": concurrency,
                "request_rate": request_rate,
                "bs": b,
                "global_bs": b * model.config.attention_dp_size,
                "ttft": ttft,
                "tpot": tpot,
                "seq/s": seq_s,
                "seq/s/gpu": seq_s_gpu,
                "tokens/s": tokens_s,
                "tokens/s/gpu": tokens_s_gpu,
                "tokens/s/user": tokens_s_user,
                "request_latency": request_latency,
                "num_total_gpus": num_total_gpus,
                "tp": tp,
                "pp": pp,
                "dp": dp,
                "moe_tp": moe_tp,
                "moe_ep": moe_ep,
                "parallel": parallel,
                "gemm": gemm,
                "kvcache": kvcache,
                "fmha": fmha,
                "moe": moe,
                "comm": comm,
                "memory": mem,
                "balance_score": balance_score,
                "num_ctx_reqs": num_ctx_requests,
                "num_gen_reqs": num_gen_requests,
                "num_tokens": num_tokens,
                "ctx_tokens": ctx_tokens,
                "gen_tokens": num_gen_requests,
                "backend": database.backend,
                "version": database.version,
                "system": database.system,
                "power_w": agg_power_avg_w,  # Weighted average power for AGG mode
            }
            result = pd.DataFrame([result_dict], columns=common.ColumnsAgg).round(3)
            summary = InferenceSummary(RuntimeConfig(isl=isl, osl=osl))
            summary.set_memory_and_check_oom(memory, database.system_spec["gpu"]["mem_capacity"])
            summary.set_summary_df(result)
            summary.set_result_dict(result_dict)

            # Store per-ops latency breakdown
            per_ops_data["scheduling"] = {
                "num_mix_steps": float(num_mix_steps),
                "num_genonly_steps": float(num_genonly_steps),
                "mix_step_latency_ms": float(mix_step_latency_ms),
                "genonly_step_latency_ms": float(genonly_step_latency_ms),
            }
            summary.set_per_ops_data(per_ops_data)

            # caching
            self._agg_cache[isl][osl][b][ctx_tokens] = summary

        return summary

    def find_best_agg_result_under_constraints(
        self, model: BaseModel, database: PerfDatabase, runtime_config: RuntimeConfig, **kwargs
    ) -> InferenceSummary:
        """
        Find the best agg result under constraints.

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
        isl = runtime_config.isl
        osl = runtime_config.osl
        ttft = runtime_config.ttft
        tpot = runtime_config.tpot
        prefix = runtime_config.prefix
        top_k = kwargs.get("top_k", 1)
        max_batch_size = kwargs.get("max_batch_size", 512)
        ctx_stride = kwargs.get("ctx_stride", 512)
        enable_chunked_prefill = kwargs.get("enable_chunked_prefill", False)

        # when b is larger than 1024, the result is not good as the data collection is not enough
        # to cover this.
        b_list_default = (
            list(range(1, 16, 1))
            + list(range(16, 32, 4))
            + list(range(32, 64, 8))
            + list(range(64, 256, 16))
            + list(range(256, 512, 32))
            + list(range(512, 1024, 256))
            + [1024]
        )

        # sweep for batch_size and ctx_tokens
        # ctx_tokens will have a step of ctx_stride. When it's larger than 8192, we will increase
        # the step to ctx_stride_large.
        # outer_loop is over batch_size dimention, from 1 to max_batch_size
        # inner_loop is over ctx_tokens dimention, from 0 to max_ctx_tokens where it's
        # max(8192, 4*isl).
        # during the loop, as b, ctx_tokens and system memory are monotonic, we can break the
        # inner loop when the system is oom.
        b_list = [b for b in b_list_default if b <= max_batch_size]
        ctx_tokens_list = self._get_ctx_tokens_list_for_agg_sweep(isl, ctx_stride, enable_chunked_prefill)

        results_df = pd.DataFrame(columns=common.ColumnsAgg)
        results_dict_list = []
        capped_b = []
        all_oom = True
        for b in b_list:
            for ctx_tokens in ctx_tokens_list:
                if b - np.ceil(ctx_tokens / isl) < 0:  # allow b==1
                    break

                if b > 1 and (
                    b - np.ceil(ctx_tokens / isl) < 1
                ):  # general case, to ensure there's at least one gen req
                    break

                # filter out repeated records for balance score correction
                balance_score = isl * b / ctx_tokens / osl
                if balance_score > 1:
                    gen_tokens = b // balance_score
                    if gen_tokens > 1 and gen_tokens in capped_b:
                        continue
                    else:
                        capped_b.append(gen_tokens)

                summary = self.run_agg(
                    model=model,
                    database=database,
                    runtime_config=RuntimeConfig(batch_size=b, isl=isl, osl=osl, prefix=prefix),
                    ctx_tokens=ctx_tokens,
                )

                if summary.check_oom():
                    break  # larger ctx tokens will cause oom
                all_oom = False
                result_dict = summary.get_result_dict()
                if result_dict and result_dict["tpot"] <= tpot and result_dict["ttft"] <= ttft:
                    results_dict_list.append(result_dict)

        if results_dict_list:
            results_df = pd.DataFrame(results_dict_list, columns=common.ColumnsAgg).round(3)

        sorted_results_df = results_df.sort_values(by="seq/s", ascending=False).round(3)
        if top_k > 0:
            sorted_results_df = sorted_results_df.head(top_k)

        summary = InferenceSummary(runtime_config)
        summary.set_summary_df(sorted_results_df)
        summary.set_oom(all_oom)
        return summary

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
        """
        Get the memory usage of the backend.
        """
        weights, activations, kvcache = 0.0, 0.0, 0.0
        for op in model.context_ops:
            weights += op.get_weights()

        # count weights on a single GPU
        weights /= model.config.pp_size

        h = model._num_heads * model._head_size
        if num_tokens == 0:
            num_tokens = isl * batch_size

        # ==== this below section is backend specific ====
        # FIXME: the measurement is done based on trt workflow and traditional moe.
        #        needs to study the new model again. Expecially fine-grained moe will introduce
        #        more act/workspace memory.
        if model.model_family == "GPT":
            c_dict = {1: 10, 2: 6, 4: 5, 8: 5}
            activations = 2 * num_tokens * h * c_dict[min(model.config.tp_size, 8)]
            activations = max(activations, 70 * 1024 * 1024)  # minimum act
        elif model.model_family == "LLAMA":
            c_dict = {1: 11, 2: 6.5, 4: 5, 8: 5}
            activations = 2 * num_tokens * h * c_dict[min(model.config.tp_size, 8)]
            activations = max(activations, 70 * 1024 * 1024)  # minimum act
        elif model.model_family == "MOE":
            c_dict = {1: 22, 2: 13, 4: 10, 8: 10}
            activations = 2 * num_tokens * h * c_dict[min(model.config.tp_size, 8)]
            activations = max(activations, 70 * 1024 * 1024)  # minimum act
        elif model.model_family == "DEEPSEEK":
            c_dict = {1: 22, 2: 13, 4: 10, 8: 10}
            activations = 2 * num_tokens * h * c_dict[min(model.config.tp_size, 8)]
            # moe workspace, 128 for block scale, float for 4bytes
            activations += (
                num_tokens
                * h
                * model.config.attention_dp_size
                * model._num_experts
                * model._topk
                / model.config.moe_ep_size
                / 128
                * 4
            )  # still an improvement opportunity in trtllm to achieve this.
            # MTP correction: additional activation memory for draft tokens
            if model.config.nextn > 0:
                activations = activations * (model.config.nextn + 1)
            activations = max(activations, 70 * 1024 * 1024)  # minimum act
        else:
            c_dict = {
                1: 10,
                2: 6,
                4: 5,
                8: 5,
            }  # 4+6/TP, fp8 will have relatively low act, but ignore here. need more experiments
            activations = 2 * num_tokens * h * c_dict[min(model.config.tp_size, 8)]
            activations = max(activations, 70 * 1024 * 1024)  # minimum act
        # ==== this above section is backend specific ====

        if model.model_family == "DEEPSEEK":
            kvcache_per_token = model._num_layers * 576
        else:
            num_kv_heads_per_gpu = (model._num_kv_heads + model.config.tp_size - 1) // model.config.tp_size
            kvcache_per_token = num_kv_heads_per_gpu * model._head_size * model._num_layers * 2
        # should not be divided by pp_size as you need to hold all kvcache for stages.
        kvcache = (
            (batch_size * isl + batch_size * beam_width * osl)
            * model.config.kvcache_quant_mode.value.memory
            * kvcache_per_token
        )
        # if 'DEEPSEEK' in model.model_path or 'MOE' in model.model_path:
        #    kvcache = kvcache * model.config.attention_dp_size # this is incorrect. tp will
        #    duplicate the kvcache while attn_dp will not.

        # starting from 2.22
        nccl_mem = database.system_spec["misc"]["nccl_mem"][min(model.config.tp_size, 8)]

        # cuda, cublas, etc.
        others_mem = database.system_spec["misc"]["other_mem"]

        one_gib = 1 << 30
        return {
            "total": (weights + activations + kvcache + nccl_mem + others_mem) / one_gib,
            "weights": weights / one_gib,
            "activations": activations / one_gib,
            "kvcache": kvcache / one_gib,
            "nccl": nccl_mem / one_gib,
            "others": others_mem / one_gib,
        }
