# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gradio as gr
from aiconfigurator.sdk.common import SupportedModels
from aiconfigurator.sdk.perf_database import get_all_databases
from aiconfigurator.sdk import common

def create_model_name_config(app_config):
    """create model name config components"""
    with gr.Accordion("Model"):
        model_name = gr.Dropdown(choices=list(SupportedModels.keys()),
                                label="model name",
                                value='DEEPSEEK_V3',
                                interactive=True)
    
    return {
        'model_name': model_name
    }

def create_system_config(app_config):
    """create system config components"""
    database_dict = get_all_databases()
    system_choices = sorted(list(database_dict.keys()))
    default_system = 'h200_sxm' if 'h200_sxm' in system_choices else system_choices[0]
    backend_choices = sorted(list(database_dict[default_system].keys()))
    default_backend = 'trtllm' if 'trtllm' in backend_choices else backend_choices[0]
    version_choices = sorted(list(database_dict[default_system][default_backend].keys()))
    default_version = version_choices[-1]
      
    with gr.Accordion("System config"):
        model_choices = list(SupportedModels.keys())
        with gr.Row():
            system = gr.Dropdown(choices=system_choices,
                                    label="System",
                                    value=default_system,
                                    interactive=True)
            backend = gr.Dropdown(choices=backend_choices,
                                    label="Backend",
                                    value=default_backend,
                                    interactive=True)            
            version = gr.Dropdown(choices=version_choices,
                                    label="Version",
                                    value=default_version,
                                    interactive=True)

            sol_mode = gr.Checkbox(label="SOL Mode", value=False, interactive=True, visible=app_config['experimental'])
    
    return {
        'system': system,
        'backend': backend,
        'version': version,
        'sol_mode': sol_mode
    }

def create_model_quant_config(app_config):
    """create model quantization config components"""
    database_dict = get_all_databases()
    system_choices = sorted(list(database_dict.keys()))
    default_system = 'h200_sxm' if 'h200_sxm' in system_choices else system_choices[0]
    backend_choices = sorted(list(database_dict[default_system].keys()))
    default_backend = 'trtllm' if 'trtllm' in backend_choices else backend_choices[0]
    version_choices = sorted(list(database_dict[default_system][default_backend].keys()))
    default_version = version_choices[-1]
    database = database_dict[default_system][default_backend][default_version]
    gemm_quant_mode_choices = database.supported_quant_mode['gemm']
    kvcache_quant_mode_choices = database.supported_quant_mode['generation_mla'] # DS V3 by default
    fmha_quant_mode_choices = database.supported_quant_mode['context_mla'] # DS V3 by default
    moe_quant_mode_choices = database.supported_quant_mode['moe']
    #comm_quant_mode_choices = database.supported_quant_mode['comm']

    with gr.Accordion("Quantization config"):
        with gr.Row():
            gemm_quant_mode = gr.Dropdown(choices=gemm_quant_mode_choices, 
                                    label="gemm quant mode", 
                                    allow_custom_value=False, 
                                    value='fp8' if 'fp8' in gemm_quant_mode_choices else gemm_quant_mode_choices[0],
                                    interactive=True)
            kvcache_quant_mode = gr.Dropdown(choices=kvcache_quant_mode_choices,
                                            label="kvcache quant mode", 
                                            allow_custom_value=False, 
                                            value='fp8' if 'fp8' in kvcache_quant_mode_choices else kvcache_quant_mode_choices[0],
                                            interactive=True)
            fmha_quant_mode = gr.Dropdown(choices=fmha_quant_mode_choices, 
                                        label="fmha quant mode", 
                                        allow_custom_value=False, 
                                        value='float16' if 'float16' in fmha_quant_mode_choices else fmha_quant_mode_choices[0],
                                        interactive=True)
            moe_quant_mode = gr.Dropdown(choices=moe_quant_mode_choices,
                                        label="moe quant mode", 
                                        allow_custom_value=False, 
                                        value='fp8_block' if 'fp8_block' in moe_quant_mode_choices else moe_quant_mode_choices[0],
                                        interactive=True)
            comm_quant_mode = gr.Dropdown(choices=common.CommQuantMode.__members__.keys(),
                                        label="comm quant mode", 
                                        allow_custom_value=False, 
                                        value='half',
                                        visible=False,
                                        interactive=True)
    
    return {
        'gemm_quant_mode': gemm_quant_mode,
        'kvcache_quant_mode': kvcache_quant_mode,
        'fmha_quant_mode': fmha_quant_mode,
        'moe_quant_mode': moe_quant_mode,
        'comm_quant_mode': comm_quant_mode
    }

def create_model_parallel_config(app_config, single_select=True, disagg=False):
    """create model parallel config components"""
    if single_select:
        with gr.Accordion("Parallel config"):
            with gr.Row():
                tp_size = gr.Dropdown(choices=[1,2,4,8,16,32,64], label="tp size", value=8, interactive=True)
                pp_size = gr.Dropdown(choices=[1,2,4,8], label="pp size", value=1, interactive=True)
                dp_size = gr.Dropdown(choices=[1,2,4,8,16,32,64,128,256], label="dp size", value=1, interactive=True)
                moe_tp_size = gr.Dropdown(choices=[1,2,4,8,16], label="moe tp size", value=1, interactive=True)
                moe_ep_size = gr.Dropdown(choices=[1,2,4,8,16,32,64,128,256], label="moe ep size", value=8, interactive=True)
    else:
        with gr.Accordion("Parallel config"):
            with gr.Row():
                if disagg:
                    num_worker = gr.Number(label="num worker, -1 for auto", value=-1, interactive=True)
                num_gpus = gr.CheckboxGroup(choices=[1,2,4,8,16,32,64,128,256], label="num gpus in a worker", value=8, interactive=True)
                tp_size = gr.CheckboxGroup(choices=[1,2,4,8,16,32,64], label="tp size", value=8, interactive=True)
                pp_size = gr.CheckboxGroup(choices=[1,2,4,8], label="pp size", value=1, interactive=True)
                dp_size = gr.CheckboxGroup(choices=[1,2,4,8,16,32,64,128,256], label="dp size", value=1, interactive=True)
                moe_tp_size = gr.CheckboxGroup(choices=[1,2,4,8,16], label="moe tp size", value=1, interactive=True)
                moe_ep_size = gr.CheckboxGroup(choices=[1,2,4,8,16,32,64,128,256], label="moe ep size", value=8, interactive=True)
    components_dict = {
        'tp_size': tp_size,
        'pp_size': pp_size,
        'dp_size': dp_size,
        'moe_tp_size': moe_tp_size,
        'moe_ep_size': moe_ep_size,
    }
    if not single_select:
        components_dict['num_gpus'] = num_gpus
        if disagg:
            components_dict['num_worker'] = num_worker
    return components_dict

def create_model_misc_config(app_config):
    """create model misc config components"""
    with gr.Accordion("Misc config"):
        with gr.Row():
            nextn = gr.Dropdown(choices=[0,1,2,3,4,5], value=0, label="nextn", interactive=True)
            nextn_accept_rates = gr.Textbox(value='0.85,0.2,0.0,0.0,0.0', label="nextn accept rates", interactive=True)

    return {
        'nextn': nextn,
        'nextn_accept_rates': nextn_accept_rates
    }

def create_runtime_config(app_config, with_sla=False):
    """create runtime config components"""
    
    with gr.Accordion("Runtime config"):
        with gr.Row():
            isl = gr.Number(value=2048, label='input sequence length', interactive=True)
            osl = gr.Number(value=128, label='output sequence length', interactive=True)
                
            if with_sla:
                ttft = gr.Number(value=2000, label='first token latency(ms)', interactive=True)
                tpot = gr.Number(value=50, label='inter token latency(ms)', interactive=True)
                return {
                    'isl': isl,
                    'osl': osl,
                    'ttft': ttft,
                    'tpot': tpot
                }
            else:
                batch_size = gr.Number(value=1, label='batch size', interactive=True)
                return {
                    'isl': isl,
                    'osl': osl,
                    'batch_size': batch_size
                }
