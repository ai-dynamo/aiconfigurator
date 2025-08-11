# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiconfigurator.sdk.models import get_model, check_is_moe, get_model_family
from aiconfigurator.sdk.inference_session import InferenceSession
import pandas as pd
from aiconfigurator.sdk import config
import gradio as gr
from aiconfigurator.sdk.perf_database import get_all_databases
from aiconfigurator.sdk import common, models
import numpy as np
import plotly.graph_objects as go
from aiconfigurator.sdk.backends.factory import get_backend
from aiconfigurator.sdk import pareto_analysis
import copy
import traceback
from io import StringIO
import contextlib
import logging

class LogCapture:
    def __init__(self):
        self.log_buffer = StringIO()
        self.handler = logging.StreamHandler(self.log_buffer)
        self.handler.setLevel(logging.INFO)
        self.formatter = logging.Formatter("%(levelname)s %(asctime)s] %(message)s")
        self.handler.setFormatter(self.formatter)
        
    def __enter__(self):
        self.file_logger = logging.getLogger()
        self.file_logger.addHandler(self.handler)
        return self.file_logger, self.log_buffer
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file_logger.removeHandler(self.handler)

def create_scatter_plot(df, x_col, y_col, title, is_disagg=False):
    
    fig = go.Figure()
    if not is_disagg:
        fig.add_trace(go.Scatter(
            x=df[x_col], 
            y=df[y_col],
            mode="lines+markers",
            line=dict(color='red', width=2),
            marker=dict(color='blue', size=8),
            hovertemplate=
            "<b>tokens/s/user:</b> %{x:.2f}<br>" +
            "<b>tokens/s/gpu:</b> %{y:.2f}<br>" +
            "<b>TTFT(ms):</b> %{customdata[0]:.2f}<br>" +
            "<b>TPOT(ms):</b> %{customdata[1]:.2f}<br>" +
            "<b>seq/s (system):</b> %{customdata[2]:.2f}<br>" +
            "<b>concurrency:</b> %{customdata[3]}<br>" +
            "<b>parallel:</b> %{customdata[4]}<br>" +
            "<b>memory(GiB):</b> %{customdata[5]:.2f}<br>" +
            "<b>total gpus:</b> %{customdata[6]}<br>" +
            "<b>index:</b> %{customdata[7]}<extra></extra>",
            customdata=np.stack((df["ttft"],df["tpot"],df["seq/s"],df["concurrency"],df["parallel"], df["memory"], df["num_total_gpus"], df["index"]), axis=1)
        ))
    else:
        fig.add_trace(go.Scatter(
            x=df[x_col], 
            y=df[y_col],
            mode="lines+markers",
            line=dict(color='red', width=2),
            marker=dict(color='blue', size=8),
            hovertemplate=
            "<b>tokens/s/user:</b> %{x:.2f}<br>" +
            "<b>tokens/s/gpu:</b> %{y:.2f}<br>" +
            "<b>TTFT(ms):</b> %{customdata[0]:.2f}<br>" +
            "<b>TPOT(ms):</b> %{customdata[1]:.2f}<br>" +
            "<b>seq/s (system):</b> %{customdata[2]:.2f}<br>" +
            "<b>prefill hardware:</b> %{customdata[3]}<br>" +
            "<b>prefill workers:</b> %{customdata[4]}<br>" +
            "<b>prefill seq/s/worker:</b> %{customdata[5]:.2f}<br>" +
            "<b>prefill parallel:</b> %{customdata[6]}<br>" +
            "<b>prefill bs:</b> %{customdata[7]}<br>" +
            "<b>prefill memory:</b> %{customdata[8]}<br>" +
            "<b>decode hardware:</b> %{customdata[9]}<br>" +
            "<b>decode workers:</b> %{customdata[10]}<br>" +
            "<b>decode seq/s/worker:</b> %{customdata[11]:.2f}<br>" +
            "<b>decode parallel:</b> %{customdata[12]}<br>" +
            "<b>decode bs:</b> %{customdata[13]}<br>" +
            "<b>decode memory:</b> %{customdata[14]}<br>" +
            "<b>concurrency:</b> %{customdata[15]}<br>" +
            "<b>total gpus:</b> %{customdata[16]}<br>" +
            "<b>index:</b> %{customdata[17]}<extra></extra>",
            customdata=np.stack((df["ttft"],df["tpot"],df["seq/s"],df["(p)system"], df["(p)workers"], df["(p)seq/s/worker"], df['(p)parallel'], df["(p)bs"], df["(p)memory"], df["(d)system"], df["(d)workers"], df["(d)seq/s/worker"], df['(d)parallel'], df["(d)bs"], df["(d)memory"], df["concurrency"], df["num_total_gpus"], df["index"]), axis=1)
        ))

    fig.update_layout(
        title={
            'text': title,
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=14, family='Arial, sans-serif')
        },
        xaxis_title={
            'text': x_col,
            'font': dict(size=14, family='Arial, sans-serif')
        },
        yaxis_title={
            'text': y_col,
            'font': dict(size=14, family='Arial, sans-serif')
        },
        height=600,
    )
    
    html_str = fig.to_html(include_plotlyjs='cdn', full_html=False)
    iframe_html = f'<iframe srcdoc="{html_str.replace(chr(34), chr(39))}" width="100%" height="650px"></iframe>'
    return iframe_html

class EventFn:
    @staticmethod
    def run_estimation_static(model_name, system_name, backend_name, version, sol_mode,
                              batch_size, isl, osl,
                              tp_size, pp_size, dp_size, moe_tp_size, moe_ep_size, 
                              gemm_quant_mode, kvcache_quant_mode, fmha_quant_mode, 
                              moe_quant_mode, comm_quant_mode,
                              nextn, nextn_accept_rates,mode,
                              record_df):
        is_error = False
        stdout_buffer = StringIO()
        stderr_buffer = StringIO()
        
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer), LogCapture() as (logger, log_buffer):
            try:        
                database_dict = get_all_databases()
                database = database_dict[system_name][backend_name][version]
                database.set_default_sol_mode(common.SOLMode(int(sol_mode)))
                nextn_accept_rates = [float(x) for x in nextn_accept_rates.split(',')]
                model_config = config.ModelConfig(tp_size=tp_size,
                                        pp_size=pp_size,
                                        attention_dp_size=dp_size,
                                        moe_tp_size=moe_tp_size,
                                        moe_ep_size=moe_ep_size,
                                        gemm_quant_mode=common.GEMMQuantMode[gemm_quant_mode],
                                        kvcache_quant_mode=common.KVCacheQuantMode[kvcache_quant_mode],
                                        fmha_quant_mode=common.FMHAQuantMode[fmha_quant_mode],
                                        moe_quant_mode=common.MoEQuantMode[moe_quant_mode],
                                        comm_quant_mode=common.CommQuantMode[comm_quant_mode],
                                        nextn=nextn,
                                        nextn_accept_rates=nextn_accept_rates)
                runtime_config = config.RuntimeConfig(batch_size=batch_size, isl=isl, osl=osl)

                model = get_model(model_name, model_config)
                stride = (osl + 8 - 1) // 8 # run at most 8 steps
                backend = get_backend(backend_name)
                session = InferenceSession(model, database, backend)
                summary = session.run_static(runtime_config=runtime_config, mode=mode, stride=stride)
            except Exception as e:
                traceback_log = traceback.format_exc()
                is_error = True
        stdout_text = stdout_buffer.getvalue() + log_buffer.getvalue()
        stderr_text = stderr_buffer.getvalue()
        if is_error:
            return (gr.update(value=''), gr.update(value='ERROR!!!'),
                gr.update(value=''), gr.update(value=record_df),
                gr.update(value=stdout_text+stderr_text+traceback_log))
        else:
            perf_info, mem_info, context_info, generation_info = summary.get_static_info()
            summary_string = f"```\n{perf_info+mem_info}\n```"
            context_breakdown_string = f"```\n{context_info}\n```"
            generation_breakdown_string = f"```\n{generation_info}\n```"
            new_record_df = summary.get_summary_df()

            if len(record_df.values) == 1 and record_df.values[0][0] == '':
                updated_record_df = new_record_df
            else:
                updated_record_df = pd.concat([record_df, new_record_df], ignore_index=True)
            # need to update the textbox (breakdown) as well the table to be downloaded
            return (gr.update(value=summary_string), gr.update(value=context_breakdown_string), 
                    gr.update(value=generation_breakdown_string), gr.update(value=updated_record_df),
                    gr.update(value=stdout_text+stderr_text))

    @staticmethod
    def run_estimation_ifb(model_name, system_name, backend_name, version, sol_mode, 
                           isl, osl, ttft, tpot, 
                           tp_size, pp_size, dp_size, moe_tp_size, moe_ep_size, 
                           gemm_quant_mode, kvcache_quant_mode, fmha_quant_mode, 
                           moe_quant_mode, comm_quant_mode,
                           nextn, nextn_accept_rates):
        is_error = False
        traceback_log = ""
        stdout_buffer = StringIO()
        stderr_buffer = StringIO()
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer), LogCapture() as (logger, log_buffer):
            try:        
                database_dict = get_all_databases()
                database = database_dict[system_name][backend_name][version]
                database.set_default_sol_mode(common.SOLMode(int(sol_mode)))
                nextn_accept_rates = [float(x) for x in nextn_accept_rates.split(',')]
                model_config = config.ModelConfig(tp_size=tp_size,
                                                    pp_size=pp_size,
                                                    attention_dp_size=dp_size,
                                                    moe_tp_size=moe_tp_size,
                                                    moe_ep_size=moe_ep_size,
                                                    gemm_quant_mode=common.GEMMQuantMode[gemm_quant_mode],
                                                    kvcache_quant_mode=common.KVCacheQuantMode[kvcache_quant_mode],
                                                    fmha_quant_mode=common.FMHAQuantMode[fmha_quant_mode],
                                                    moe_quant_mode=common.MoEQuantMode[moe_quant_mode],
                                                    comm_quant_mode=common.CommQuantMode[comm_quant_mode],
                                                    nextn=nextn,
                                                    nextn_accept_rates=nextn_accept_rates)
                runtime_config = config.RuntimeConfig(isl=isl, osl=osl, ttft=ttft, tpot=tpot)

                is_moe = check_is_moe(model_name)
                parallel_config_list = pareto_analysis.enumerate_parallel_config(num_gpu_list=[tp_size*pp_size*dp_size],
                                                                            tp_list=[tp_size],
                                                                            pp_list=[pp_size],
                                                                            moe_tp_list=[moe_tp_size],
                                                                            moe_ep_list=[moe_ep_size],
                                                                            dp_list=[dp_size],
                                                                            is_moe=is_moe,
                                                                            backend=common.BackendName(backend_name))
                for parallel_config in parallel_config_list:
                    tp, pp, dp, moe_tp, moe_ep = parallel_config
                    if is_moe:
                        logger.info(f"enumerated config: tp {tp} pp {pp} dp {dp} moe_tp {moe_tp} moe_ep {moe_ep}")
                    else:
                        logger.info(f"enumerated config: tp {tp} pp {pp}")
                if len(parallel_config_list) == 0:
                    logger.error(f"No valid parallel config found for {model_name} with {tp_size} GPUs. Please double check your parallel configs.")

                results_df = None
        
                backend = get_backend(backend_name)
                model = get_model(model_name, model_config)
                session = InferenceSession(model, database, backend)
                summary = session.find_best_ifb_result_under_constraints(runtime_config=runtime_config, 
                                                                            top_k = 10,
                                                                            max_batch_size = 512,
                                                                            ctx_stride = 512)
                results_df = summary.get_summary_df()

                if results_df is None or results_df.size == 0:
                    logger.error(f"No result for {model_name} with {tp_size} GPUs under this restriction ttft {ttft}ms, tpot {tpot} ms and memory size. Try to set a larger ttft/tpot limit and use more GPUs.")

            except Exception as e:
                results_df = pd.DataFrame(columns=common.ColumnsIFB)
                traceback_log = traceback.format_exc()
                is_error = True
        stdout_text = stdout_buffer.getvalue() + log_buffer.getvalue()
        stderr_text = stderr_buffer.getvalue()

        return (gr.update(value=results_df), gr.update(value=stdout_text+stderr_text+traceback_log))


    @staticmethod
    def run_estimation_ifb_pareto(model_name, system_name, backend_name, version, sol_mode, 
                                  isl, osl, ttft,
                                  num_gpus, tp_size, pp_size, dp_size, moe_tp_size, moe_ep_size, 
                                  gemm_quant_mode, kvcache_quant_mode, fmha_quant_mode, 
                                  moe_quant_mode, comm_quant_mode,
                                  nextn, nextn_accept_rates):
        is_error = False
        traceback_log = ""
        stdout_buffer = StringIO()
        stderr_buffer = StringIO()
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer), LogCapture() as (logger, log_buffer):
            try:        
                database_dict = get_all_databases()
                database = database_dict[system_name][backend_name][version]
                database.set_default_sol_mode(common.SOLMode(int(sol_mode)))
                nextn_accept_rates = [float(x) for x in nextn_accept_rates.split(',')]
                model_config = config.ModelConfig(gemm_quant_mode=common.GEMMQuantMode[gemm_quant_mode],
                                                    kvcache_quant_mode=common.KVCacheQuantMode[kvcache_quant_mode],
                                                    fmha_quant_mode=common.FMHAQuantMode[fmha_quant_mode],
                                                    moe_quant_mode=common.MoEQuantMode[moe_quant_mode],
                                                    comm_quant_mode=common.CommQuantMode[comm_quant_mode],
                                                    nextn=nextn,
                                                    nextn_accept_rates=nextn_accept_rates)
                runtime_config = config.RuntimeConfig(isl=isl, osl=osl, ttft=ttft, tpot=list(range(2,20,1))+list(range(20,300,5)))

                is_moe = check_is_moe(model_name)
                parallel_config_list = pareto_analysis.enumerate_parallel_config(num_gpu_list=num_gpus,
                                                                            tp_list=tp_size,
                                                                            pp_list=pp_size,
                                                                            moe_tp_list=moe_tp_size,
                                                                            moe_ep_list=moe_ep_size,
                                                                            dp_list=dp_size,
                                                                            is_moe=is_moe,
                                                                            backend=common.BackendName(backend_name))
                for parallel_config in parallel_config_list:
                    tp, pp, dp, moe_tp, moe_ep = parallel_config
                    if is_moe:
                        logger.info(f"enumerated config: tp {tp} pp {pp} dp {dp} moe_tp {moe_tp} moe_ep {moe_ep}")
                    else:
                        logger.info(f"enumerated config: tp {tp} pp {pp}")

                if len(parallel_config_list) == 0:
                    logger.error(f"No valid parallel config found for {model_name} with {tp_size} GPUs. Please double check your parallel configs.")

                results_df = pareto_analysis.ifb_pareto(model_name=model_name, 
                                        runtime_config=runtime_config,
                                        database=database, 
                                        backend_name=backend_name,
                                        model_config=model_config,
                                        parallel_config_list=parallel_config_list)
                
                results_df = pareto_analysis.get_pareto_front(results_df, 'tokens/s/user', 'tokens/s/gpu')
                results_df = results_df.reset_index(drop=True).reset_index()
                if results_df.size == 0:
                    logger.error(f"No result for {model_name} with {tp_size} GPUs under this restriction ttft {ttft}ms and memory size. Try to set a larger ttft limit and use more GPUs.")
                title = f'{model_name}_isl{runtime_config.isl}_osl{runtime_config.osl}_ttft{runtime_config.ttft}_{system_name}_{backend_name}_{version}_{model_config.gemm_quant_mode}_{model_config.kvcache_quant_mode}_{model_config.fmha_quant_mode}_{model_config.moe_quant_mode}_{model_config.comm_quant_mode}_IFB_Pareto'
                pareto_html = create_scatter_plot(results_df, 'tokens/s/user', 'tokens/s/gpu', title)
            except Exception as e:
                results_df = pd.DataFrame(columns=common.ColumnsIFB)
                traceback_log = traceback.format_exc()
                is_error = True
        stdout_text = stdout_buffer.getvalue() + log_buffer.getvalue()
        stderr_text = stderr_buffer.getvalue()

        if is_error:
            return gr.update(value=results_df), gr.update(value=""), gr.update(value=""), gr.update(interactive=False), gr.update(value=stdout_text+stderr_text+traceback_log)
        return gr.update(value=results_df), gr.update(value=pareto_html), gr.update(value=title), gr.update(interactive=True, value="Save for comparison"), gr.update(value=stdout_text+stderr_text+traceback_log)
    

    @staticmethod
    def run_estimation_disagg_pareto(model_name, 
                                     isl, osl, ttft,
                                     nextn, nextn_accept_rates,
                                     prefill_system_name, prefill_backend_name, prefill_version, prefill_sol_mode,
                                     prefill_num_worker, prefill_num_gpus, prefill_tp_size, prefill_pp_size, prefill_dp_size, prefill_moe_tp_size, prefill_moe_ep_size,
                                     prefill_gemm_quant_mode, prefill_kvcache_quant_mode, prefill_fmha_quant_mode,
                                     prefill_moe_quant_mode, prefill_comm_quant_mode,
                                     prefill_correction_scale,
                                     decode_system_name, decode_backend_name, decode_version, decode_sol_mode,
                                     decode_num_worker, decode_num_gpus, decode_tp_size, decode_pp_size, decode_dp_size, decode_moe_tp_size, decode_moe_ep_size,
                                     decode_gemm_quant_mode, decode_kvcache_quant_mode, decode_fmha_quant_mode,
                                     decode_moe_quant_mode, decode_comm_quant_mode,
                                     decode_correction_scale,
                                     num_gpu_list, max_num_gpu,
                                     prefill_max_num_worker, decode_max_num_worker,
                                     prefill_max_batch_size, decode_max_batch_size):
        is_error = False
        traceback_log = ""
        stdout_buffer = StringIO()
        stderr_buffer = StringIO()
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer), LogCapture() as (logger, log_buffer):
            try:        
                database_dict = get_all_databases()
                prefill_database = database_dict[prefill_system_name][prefill_backend_name][prefill_version]
                decode_database = database_dict[decode_system_name][decode_backend_name][decode_version]
                # to avoid conflict.
                if prefill_sol_mode != decode_sol_mode:
                    decode_database = copy.deepcopy(decode_database)
                prefill_database.set_default_sol_mode(common.SOLMode(int(prefill_sol_mode)))
                decode_database.set_default_sol_mode(common.SOLMode(int(decode_sol_mode)))
                nextn_accept_rates = [float(x) for x in nextn_accept_rates.split(',')]
                prefill_model_config = config.ModelConfig(tp_size=prefill_tp_size,
                                                        pp_size=prefill_pp_size,
                                                        attention_dp_size=prefill_dp_size,
                                                        moe_tp_size=prefill_moe_tp_size,
                                                        moe_ep_size=prefill_moe_ep_size,
                                                        gemm_quant_mode=common.GEMMQuantMode[prefill_gemm_quant_mode],
                                                        kvcache_quant_mode=common.KVCacheQuantMode[prefill_kvcache_quant_mode],
                                                        fmha_quant_mode=common.FMHAQuantMode[prefill_fmha_quant_mode],
                                                        moe_quant_mode=common.MoEQuantMode[prefill_moe_quant_mode],
                                                        comm_quant_mode=common.CommQuantMode[prefill_comm_quant_mode],
                                                        nextn=nextn,
                                                        nextn_accept_rates=nextn_accept_rates)
                decode_model_config = config.ModelConfig(tp_size=decode_tp_size,
                                                        pp_size=decode_pp_size,
                                                        attention_dp_size=decode_dp_size,
                                                        moe_tp_size=decode_moe_tp_size,
                                                        moe_ep_size=decode_moe_ep_size,
                                                        gemm_quant_mode=common.GEMMQuantMode[decode_gemm_quant_mode],
                                                        kvcache_quant_mode=common.KVCacheQuantMode[decode_kvcache_quant_mode],
                                                        fmha_quant_mode=common.FMHAQuantMode[decode_fmha_quant_mode],
                                                        moe_quant_mode=common.MoEQuantMode[decode_moe_quant_mode],
                                                        comm_quant_mode=common.CommQuantMode[decode_comm_quant_mode],
                                                        nextn=nextn,
                                                        nextn_accept_rates=nextn_accept_rates)
                runtime_config = config.RuntimeConfig(isl=isl, osl=osl, ttft=ttft, tpot=list(range(1,20,1))+list(range(20,300,5)))

                is_moe = check_is_moe(model_name)

                prefill_parallel_config_list = pareto_analysis.enumerate_parallel_config(num_gpu_list=prefill_num_gpus,
                                                                                        tp_list=prefill_tp_size,
                                                                                        pp_list=prefill_pp_size,
                                                                                        moe_tp_list=prefill_moe_tp_size,
                                                                                        moe_ep_list=prefill_moe_ep_size,
                                                                                        dp_list=prefill_dp_size,
                                                                                        is_moe=is_moe,
                                                                                        backend=common.BackendName(prefill_backend_name))
                decode_parallel_config_list = pareto_analysis.enumerate_parallel_config(num_gpu_list=decode_num_gpus,
                                                                                        tp_list=decode_tp_size,
                                                                                        pp_list=decode_pp_size,
                                                                                        moe_tp_list=decode_moe_tp_size,
                                                                                        moe_ep_list=decode_moe_ep_size,
                                                                                        dp_list=decode_dp_size,
                                                                                        is_moe=is_moe,
                                                                                        backend=common.BackendName(decode_backend_name))
                
                for prefill_parallel_config in prefill_parallel_config_list:
                    tp, pp, dp, moe_tp, moe_ep = prefill_parallel_config
                    if is_moe:
                        logger.info(f"enumerated prefill config: tp {tp} pp {pp} dp {dp} moe_tp {moe_tp} moe_ep {moe_ep}")
                    else:
                        logger.info(f"enumerated prefill config: tp {tp} pp {pp}")
                
                for decode_parallel_config in decode_parallel_config_list:
                    tp, pp, dp, moe_tp, moe_ep = decode_parallel_config
                    if is_moe:
                        logger.info(f"enumerated decode config: tp {tp} pp {pp} dp {dp} moe_tp {moe_tp} moe_ep {moe_ep}")
                    else:
                        logger.info(f"enumerated decode config: tp {tp} pp {pp}")
                
                if len(prefill_parallel_config_list) == 0 or len(decode_parallel_config_list) == 0:
                    logger.error(f"No valid parallel config found for {model_name} in the prefill or decode system. Please double check your parallel configs.")

                if prefill_num_worker == -1:
                    prefill_num_worker_list = list(range(1, prefill_max_num_worker+1, 1))
                else:
                    prefill_num_worker_list = [prefill_num_worker]
                if decode_num_worker == -1:
                    decode_num_worker_list = list(range(1, decode_max_num_worker+1, 1))
                else:
                    decode_num_worker_list = [decode_num_worker]

                num_gpu_list = [int(x) for x in num_gpu_list.split(',')] if len(num_gpu_list) > 0 else None
                #logger.info(f"target num_gpu_list in the disagg system: {num_gpu_list}")
                results_df = pareto_analysis.disagg_pareto(model_name=model_name, 
                                                        runtime_config=runtime_config,
                                                        prefill_database=prefill_database, 
                                                        prefill_backend_name=prefill_backend_name,
                                                        prefill_model_config=prefill_model_config,
                                                        prefill_parallel_config_list=prefill_parallel_config_list,
                                                        prefill_num_worker_list=prefill_num_worker_list,
                                                        prefill_correction_scale=prefill_correction_scale,
                                                        decode_database=decode_database,
                                                        decode_backend_name=decode_backend_name,
                                                        decode_model_config=decode_model_config,
                                                        decode_parallel_config_list=decode_parallel_config_list,
                                                        decode_num_worker_list=decode_num_worker_list,
                                                        decode_correction_scale=decode_correction_scale,
                                                        num_gpu_list=num_gpu_list,
                                                        max_num_gpu=max_num_gpu if max_num_gpu > 0 else None,
                                                        prefill_max_num_tokens=prefill_max_batch_size*isl,
                                                        decode_max_num_tokens=decode_max_batch_size)
                
                results_df = pareto_analysis.get_pareto_front(results_df, 'tokens/s/user', 'tokens/s/gpu')
                results_df = results_df.reset_index(drop=True).reset_index()
                if results_df.size == 0:
                    logger.error(f"No result for {model_name} under this restriction ttft {ttft}ms and memory size. Try to set a larger ttft limit and use more GPUs.")
                title = f'{model_name}_isl{runtime_config.isl}_osl{runtime_config.osl}_ttft{runtime_config.ttft}_prefill_{prefill_system_name}_{prefill_backend_name}_{prefill_version}_{prefill_sol_mode}_{prefill_gemm_quant_mode}_{prefill_kvcache_quant_mode}_{prefill_fmha_quant_mode}_{prefill_moe_quant_mode}_{prefill_comm_quant_mode}_decode_{decode_system_name}_{decode_backend_name}_{decode_version}_{decode_sol_mode}_{decode_gemm_quant_mode}_{decode_kvcache_quant_mode}_{decode_fmha_quant_mode}_{decode_moe_quant_mode}_{decode_comm_quant_mode}_Disagg_Pareto'
                pareto_html = create_scatter_plot(results_df, 'tokens/s/user', 'tokens/s/gpu', title, is_disagg=True)
            except Exception as e:
                results_df = pd.DataFrame(columns=common.ColumnsDisagg)
                traceback_log = traceback.format_exc()
                is_error = True
        stdout_text = stdout_buffer.getvalue() + log_buffer.getvalue()
        stderr_text = stderr_buffer.getvalue()
        if is_error:
            return gr.update(value=results_df), gr.update(value="Error!!"), gr.update(value=""), gr.update(interactive=False, value="No valid parallel config found"), gr.update(value=stdout_text+stderr_text+traceback_log)
        return gr.update(value=results_df), gr.update(value=pareto_html), gr.update(value=title), gr.update(interactive=True, value="Save for comparison"), gr.update(value=stdout_text+stderr_text+traceback_log)
    
    @staticmethod
    def run_estimation_disagg_pd_ratio(model_name, 
                                     isl, osl, ttft, tpot,
                                     nextn, nextn_accept_rates,
                                     prefill_system_name, prefill_backend_name, prefill_version, prefill_sol_mode,
                                     prefill_tp_size, prefill_pp_size, prefill_dp_size, prefill_moe_tp_size, prefill_moe_ep_size,
                                     prefill_gemm_quant_mode, prefill_kvcache_quant_mode, prefill_fmha_quant_mode,
                                     prefill_moe_quant_mode, prefill_comm_quant_mode,
                                     decode_system_name, decode_backend_name, decode_version, decode_sol_mode,
                                     decode_tp_size, decode_pp_size, decode_dp_size, decode_moe_tp_size, decode_moe_ep_size,
                                     decode_gemm_quant_mode, decode_kvcache_quant_mode, decode_fmha_quant_mode,
                                     decode_moe_quant_mode, decode_comm_quant_mode
                              ):

        def create_scatter_plot(df, x_col, y_col, target_x, x_label, title):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df[x_col], 
                y=df[y_col],
                mode="lines+markers",
                line=dict(color='red', width=2),
                marker=dict(color='blue', size=8),
                hovertemplate=
                f"<b>{x_col}:</b> %{{x:.2f}}<br>" +
                f"<b>{y_col}:</b> %{{y:.2f}}<br>" +
                f"<b>{x_label}:</b> %{{customdata[0]:.2f}}<br>" +
                "<b>memory(GiB):</b> %{customdata[1]:.2f}<br>" +
                "<b>index:</b> %{customdata[2]}<extra></extra>",
                customdata=np.stack((df[x_label],df["memory"], df["index"]), axis=1)
            ))

            if target_x != 0:
                fig.add_vline(x=target_x, line_dash="dash", line_color="gray", line_width=2)
                # Add text annotation
                fig.add_annotation(
                    x=target_x,
                    y=fig.data[0].y.max(),
                    text=f"SLA_limit",
                    showarrow=True,
                    arrowhead=1,
                    arrowsize=1,
                    arrowwidth=1,
                    arrowcolor="gray",
                    font=dict(size=12, color="gray"),
                    ax=20,
                    ay=-40
                )

            fig.update_layout(
                title={
                    'text': title,
                    'y':0.95,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': dict(size=12, family='Arial, sans-serif')
                },
                xaxis_title={
                    'text': x_col,
                    'font': dict(size=14, family='Arial, sans-serif')
                },
                yaxis_title={
                    'text': y_col,
                    'font': dict(size=14, family='Arial, sans-serif')
                },
                height=800
            )
        
            html_str = fig.to_html(full_html=False, include_plotlyjs='cdn')
            
            iframe_html = f'<iframe srcdoc="{html_str.replace(chr(34), chr(39))}" width="100%" height="850px"></iframe>'
            return iframe_html

        is_error = False
        traceback_log = ""
        stdout_buffer = StringIO()
        stderr_buffer = StringIO()
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer), LogCapture() as (logger, log_buffer):
            try:
                database_dict = get_all_databases()
                
                nextn_accept_rates = [float(x) for x in nextn_accept_rates.split(',')]
                prefill_model_config = config.ModelConfig(tp_size=prefill_tp_size,
                                                            pp_size=prefill_pp_size,
                                                            attention_dp_size=prefill_dp_size,
                                                            moe_tp_size=prefill_moe_tp_size,
                                                            moe_ep_size=prefill_moe_ep_size,
                                                            gemm_quant_mode=common.GEMMQuantMode[prefill_gemm_quant_mode],
                                                            kvcache_quant_mode=common.KVCacheQuantMode[prefill_kvcache_quant_mode],
                                                            fmha_quant_mode=common.FMHAQuantMode[prefill_fmha_quant_mode],
                                                            moe_quant_mode=common.MoEQuantMode[prefill_moe_quant_mode],
                                                            comm_quant_mode=common.CommQuantMode[prefill_comm_quant_mode],
                                                            nextn=nextn,
                                                            nextn_accept_rates=nextn_accept_rates)
                decode_model_config = config.ModelConfig(tp_size=decode_tp_size,
                                                            pp_size=decode_pp_size,
                                                            attention_dp_size=decode_dp_size,
                                                            moe_tp_size=decode_moe_tp_size,
                                                            moe_ep_size=decode_moe_ep_size,
                                                            gemm_quant_mode=common.GEMMQuantMode[decode_gemm_quant_mode],
                                                            kvcache_quant_mode=common.KVCacheQuantMode[decode_kvcache_quant_mode],
                                                            fmha_quant_mode=common.FMHAQuantMode[decode_fmha_quant_mode],
                                                            moe_quant_mode=common.MoEQuantMode[decode_moe_quant_mode],
                                                            comm_quant_mode=common.CommQuantMode[decode_comm_quant_mode],
                                                            nextn=nextn,
                                                            nextn_accept_rates=nextn_accept_rates)

                prefill_max_num_tokens = 16384
                decode_max_num_tokens = 512
                decode_stride = (osl + 8 - 1) // 8
        
                # prefill
                prefill_model = get_model(model_name, prefill_model_config)
                prefill_database = database_dict[prefill_system_name][prefill_backend_name][prefill_version]
                prefill_database.set_default_sol_mode(common.SOLMode(int(prefill_sol_mode)))
                prefill_backend = get_backend(prefill_backend_name)
                prefill_session = InferenceSession(prefill_model, prefill_database, prefill_backend)
                prefill_results_df = pd.DataFrame(columns=common.ColumnsStatic)
                prefill_target_bs = 0
                for b in range(1, (prefill_max_num_tokens+isl-1)//isl+1+1):
                    prefill_summary = prefill_session.run_static(mode='static_ctx', runtime_config=config.RuntimeConfig(batch_size=b, isl=isl, osl=osl))
                    prefill_results_df = pd.concat([prefill_results_df, prefill_summary.get_summary_df()], ignore_index=True)
                    if prefill_summary.get_summary_df().loc[0,'context_latency'] > ttft and prefill_target_bs == 0:
                        prefill_target_bs = b * prefill_dp_size # global_bs
                prefill_results_df = prefill_results_df.reset_index(drop=True).reset_index()
                title = f'{model_name}_isl{isl}_osl{osl}_prefill_{prefill_system_name}_{prefill_backend_name}_{prefill_version}_{prefill_sol_mode}_{prefill_gemm_quant_mode}_{prefill_kvcache_quant_mode}_{prefill_fmha_quant_mode}_{prefill_moe_quant_mode}_{prefill_comm_quant_mode}_Throughput'
                prefill_throughput_html = create_scatter_plot(prefill_results_df, 'global_bs', 'seq/s', prefill_target_bs, 'context_latency', title)

                # decode
                decode_model = get_model(model_name, decode_model_config)
                decode_database = database_dict[decode_system_name][decode_backend_name][decode_version]
                decode_database.set_default_sol_mode(common.SOLMode(int(decode_sol_mode)))
                decode_backend = get_backend(decode_backend_name)
                decode_session = InferenceSession(decode_model, decode_database, decode_backend)
                decode_results_df = pd.DataFrame(columns=common.ColumnsStatic)
                decode_target_bs = 0
                for b in list(range(1, 16))+list(range(16, 64, 4))+list(range(64, 128, 8))+list(range(128, 256, 16))+list(range(256, 512, 32))+[512]:
                    decode_summary = decode_session.run_static(mode='static_gen', runtime_config=config.RuntimeConfig(batch_size=b, isl=isl, osl=osl), stride=decode_stride)
                    decode_results_df = pd.concat([decode_results_df, decode_summary.get_summary_df()], ignore_index=True)
                    if decode_summary.get_summary_df().loc[0,'tpot'] > tpot and decode_target_bs == 0:
                        decode_target_bs = b * decode_dp_size # global_bs
                    if decode_summary.get_summary_df().loc[0,'tpot'] > 1.5*tpot and b > 10: # early stop to make the figure clear
                        break
                decode_results_df = decode_results_df.reset_index(drop=True).reset_index()
                title = f'{model_name}_isl{isl}_osl{osl}_decode_{decode_system_name}_{decode_backend_name}_{decode_version}_{decode_sol_mode}_{decode_gemm_quant_mode}_{decode_kvcache_quant_mode}_{decode_fmha_quant_mode}_{decode_moe_quant_mode}_{decode_comm_quant_mode}_Throughput'
                decode_throughput_html = create_scatter_plot(decode_results_df, 'global_bs', 'seq/s', decode_target_bs, 'tpot', title)
            
            except Exception as e:
                prefill_results_df = pd.DataFrame(columns=common.ColumnsDisagg)
                decode_results_df = pd.DataFrame(columns=common.ColumnsDisagg)
                traceback_log = traceback.format_exc()
                is_error = True
        stdout_text = stdout_buffer.getvalue() + log_buffer.getvalue()
        stderr_text = stderr_buffer.getvalue()
        if is_error:
            return gr.update(value=prefill_results_df), gr.update(value=""), gr.update(value=decode_results_df), gr.update(value=""), gr.update(value=stdout_text+stderr_text+traceback_log)
        return gr.update(value=prefill_results_df), gr.update(value=prefill_throughput_html), gr.update(value=decode_results_df), gr.update(value=decode_throughput_html), gr.update(value=stdout_text+stderr_text+traceback_log)

    @staticmethod
    def save_result_for_comparison(result_name, result_df, pareto_results_state):
        is_error = False
        traceback_log = ""
        stdout_buffer = StringIO()
        stderr_buffer = StringIO()
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer), LogCapture() as (logger, log_buffer):
            try:
                is_interactive = False
                button_text = "Save for comparison"
                if result_name == '' or result_name in pareto_results_state:
                    logger.error(f"Result name {result_name} is invalid or already exists")
                    button_text = "Save for comparison (Result name is invalid or already exists, please rename)"
                    is_interactive = True
                else:
                    pareto_results_state[result_name] = result_df
            except Exception as e:
                traceback_log = traceback.format_exc()
                is_error = True

        return gr.update(choices=list(pareto_results_state.keys())), gr.update(interactive=is_interactive, value=button_text), gr.update(value=pareto_results_state)
    

    @staticmethod
    def clear_results(pareto_results_state):
        pareto_results_state.clear()
        return gr.update(choices=[], value=[]), gr.update(value=pareto_results_state)
    
    @staticmethod
    def compare_results(candidates_dropdown, pareto_results_state):
        color_list = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan', 'magenta']
        fig = go.Figure()
        for i, result_name in enumerate(candidates_dropdown):
            result_df = pareto_results_state[result_name]
            if 'parallel' in result_df.columns:
                system_type = 'agg'
                parallel = result_df['parallel']
                memory = result_df['memory'].astype(str)
            else:
                system_type = 'disagg'
                parallel = "(p)" + result_df["(p)parallel"].astype(str) + "_(d)" + result_df["(d)parallel"].astype(str)
                memory = "(p)" + result_df["(p)memory"].astype(str) + "_(d)" + result_df["(d)memory"].astype(str)
            fig.add_trace(go.Scatter(
                x=result_df['tokens/s/user'], 
                y=result_df['tokens/s/gpu'],
                mode="lines+markers",
                name=result_name,
                line=dict(color=color_list[i%len(color_list)], width=2),
                marker=dict(color=color_list[i%len(color_list)], size=8),
                hovertemplate=
                f"<b>system type:</b> {system_type}<br>" +
                "<b>tokens/s/user:</b> %{x:.2f}<br>" +
                "<b>tokens/s/gpu:</b> %{y:.2f}<br>" +
                "<b>TTFT(ms):</b> %{customdata[0]:.2f}<br>" +
                "<b>TPOT(ms):</b> %{customdata[1]:.2f}<br>" +
                "<b>seq/s (system):</b> %{customdata[2]:.2f}<br>" +
                "<b>concurrency:</b> %{customdata[3]}<br>" +
                "<b>parallel:</b> %{customdata[4]}<br>" +
                "<b>memory(GiB):</b> %{customdata[5]}<br>" +
                "<b>total gpus:</b> %{customdata[6]}<br>" +
                "<b>index:</b> %{customdata[7]}<extra></extra>",
                customdata=np.stack((result_df["ttft"],result_df["tpot"],result_df["seq/s"],result_df["concurrency"],parallel, memory, result_df["num_total_gpus"], result_df["index"]), axis=1)
            ))

        fig.update_layout(
            title={
                'text': 'Pareto Fronts',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title={
                'text': 'tokens/s/user',
                'font': dict(size=14, family='Arial, sans-serif')
            },
            yaxis_title={
                'text': 'tokens/s/gpu',
                'font': dict(size=14, family='Arial, sans-serif')
            },
            legend={
                'x':0.5,
                'y':-0.2,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            showlegend=True,
            height=800
        )
    
        html_str = fig.to_html(full_html=False, include_plotlyjs='cdn')
        
        iframe_html = f'<iframe srcdoc="{html_str.replace(chr(34), chr(39))}" width="100%" height="850px"></iframe>'
    
        return gr.update(value=iframe_html)
    
    @staticmethod
    def donwload_pareto_html(iframe_html):
        with open('pareto_comparison.html', 'w') as f:
            f.write(iframe_html)
        return 'pareto_comparison.html'

    # common functions
    # system change func and event
    @staticmethod
    def update_quant_mode_choices(model_name, system_name, backend_name, version):
        if version is None:
            return gr.update(choices=[], value=None, interactive=True), gr.update(choices=[], value=None, interactive=True), gr.update(choices=[], value=None, interactive=True), gr.update(choices=[], value=None, interactive=True)
        database_dict = get_all_databases()
        gemm_quant_mode_choices = sorted(database_dict[system_name][backend_name][version].supported_quant_mode['gemm'])
        kvcache_quant_mode_choices = sorted(database_dict[system_name][backend_name][version].supported_quant_mode['generation_attention']) if get_model_family(model_name) != 'DEEPSEEK' \
            else sorted(database_dict[system_name][backend_name][version].supported_quant_mode['generation_mla'])        
        fmha_quant_mode_choices = sorted(database_dict[system_name][backend_name][version].supported_quant_mode['context_attention']) if get_model_family(model_name) != 'DEEPSEEK' \
            else sorted(database_dict[system_name][backend_name][version].supported_quant_mode['context_mla'])
        moe_quant_mode_choices = sorted(database_dict[system_name][backend_name][version].supported_quant_mode['moe'])
        

        return (gr.update(choices=gemm_quant_mode_choices, value=gemm_quant_mode_choices[0], interactive=True),
                gr.update(choices=kvcache_quant_mode_choices, value=kvcache_quant_mode_choices[0], interactive=True),
                gr.update(choices=fmha_quant_mode_choices, value=fmha_quant_mode_choices[0], interactive=True),
                gr.update(choices=moe_quant_mode_choices, value=moe_quant_mode_choices[0], interactive=True))
    
    @staticmethod
    def update_system_value(model_name):
        return gr.update(value=None, interactive=True)
    
    @staticmethod
    def update_backend_choices(system_name):
        database_dict = get_all_databases()
        backend_choices = sorted(list(database_dict[system_name].keys()), reverse=True)
        return gr.update(choices=backend_choices, value=None, interactive=True), gr.update(choices=None, value=None, interactive=True)
    
    @staticmethod
    def update_version_choices(system_name, backend_name):
        database_dict = get_all_databases()
        version_choices = sorted(list(database_dict[system_name][backend_name].keys()), reverse=True)
        return gr.update(choices=version_choices, value=None, interactive=True)
    
    @staticmethod
    def update_model_related_components(model_name):
        # nextn, accept_rate, moe_quant_mode, moe_tp_size, moe_ep_size, dp_size
        if models.get_model_family(model_name) == 'DEEPSEEK':
            return gr.update(value=0, visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)
        elif models.get_model_family(model_name) == 'MOE':
            return gr.update(value=0, visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)
        else:
            return gr.update(value=0, visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
        
    # static clear button func and event
    @staticmethod
    def clear_records(record_df):
        return gr.update(value=pd.DataFrame(columns=common.ColumnsStatic))
    
    # static download button func and event
    @staticmethod
    def generate_csv(record_df):
        csv_filename = "data.csv"
        record_df.to_csv(csv_filename, index=False)
        return csv_filename

    @staticmethod
    def generate_combined_csv(record_df_prefill, record_df_decode):
        combined_df = pd.concat([record_df_prefill, record_df_decode], ignore_index=True)
        csv_filename = "data.csv"
        combined_df.to_csv(csv_filename, index=False)
        return csv_filename
