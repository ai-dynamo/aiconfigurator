# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiconfigurator.webapp.events.event_fn import EventFn

class EventHandler:
    @staticmethod
    def setup_static_events(components):
        components['estimate_btn'].click(
            fn=EventFn.run_estimation_static,
            inputs=[
                components['model_name_components']['model_name'],
                components['model_system_components']['system'],
                components['model_system_components']['backend'],
                components['model_system_components']['version'],
                components['model_system_components']['sol_mode'],
                components['runtime_config_components']['batch_size'],
                components['runtime_config_components']['isl'],
                components['runtime_config_components']['osl'],
                components['model_parallel_components']['tp_size'],
                components['model_parallel_components']['pp_size'],
                components['model_parallel_components']['dp_size'],
                components['model_parallel_components']['moe_tp_size'],
                components['model_parallel_components']['moe_ep_size'],
                components['model_quant_components']['gemm_quant_mode'],
                components['model_quant_components']['kvcache_quant_mode'],
                components['model_quant_components']['fmha_quant_mode'],
                components['model_quant_components']['moe_quant_mode'],
                components['model_quant_components']['comm_quant_mode'],
                components['model_misc_config_components']['nextn'],
                components['model_misc_config_components']['nextn_accept_rates'],
                components['mode'],
                components['record_df'] 
            ],
            outputs=[
                components['summary_box'],
                components['context_breakdown_box'],
                components['generation_breakdown_box'],
                components['record_df'],
                components['debugging_box'],
            ]
        )
        components['clear_btn'].click(
            fn=EventFn.clear_records,
            inputs=components['record_df'],
            outputs=components['record_df']
        )
        components['download_btn'].click(
            fn=EventFn.generate_csv,
            inputs=components['record_df'],
            outputs=components['output_file']
        )
        EventHandler.setup_common_events(components['model_name_components'], components['model_system_components'], components['model_quant_components'])
        EventHandler.setup_model_name_events(components['model_name_components'], components['model_quant_components'], components['model_parallel_components'], components['model_misc_config_components'])

    @staticmethod
    def setup_ifb_events(components):
        components['estimate_btn'].click(
            fn=EventFn.run_estimation_ifb,
            inputs=[
                components['model_name_components']['model_name'],
                components['model_system_components']['system'],
                components['model_system_components']['backend'],
                components['model_system_components']['version'],
                components['model_system_components']['sol_mode'],
                components['runtime_config_components']['isl'],
                components['runtime_config_components']['osl'],
                components['runtime_config_components']['ttft'],
                components['runtime_config_components']['tpot'],                
                components['model_parallel_components']['tp_size'],
                components['model_parallel_components']['pp_size'],
                components['model_parallel_components']['dp_size'],
                components['model_parallel_components']['moe_tp_size'],
                components['model_parallel_components']['moe_ep_size'],
                components['model_quant_components']['gemm_quant_mode'],
                components['model_quant_components']['kvcache_quant_mode'],
                components['model_quant_components']['fmha_quant_mode'],
                components['model_quant_components']['moe_quant_mode'],
                components['model_quant_components']['comm_quant_mode'],
                components['model_misc_config_components']['nextn'],
                components['model_misc_config_components']['nextn_accept_rates'],
            ],
            outputs=[components['result_df'], components['debugging_box']]
        )

        components['download_btn'].click(
            fn=EventFn.generate_csv,
            inputs=components['result_df'],
            outputs=components['output_file']
        )
        EventHandler.setup_common_events(components['model_name_components'], components['model_system_components'], components['model_quant_components'])
        EventHandler.setup_model_name_events(components['model_name_components'], components['model_quant_components'], components['model_parallel_components'], components['model_misc_config_components'])


    @staticmethod
    def setup_ifb_pareto_events(components):
        components['estimate_btn'].click(
            fn=EventFn.run_estimation_ifb_pareto,
            inputs=[
                components['model_name_components']['model_name'],
                components['model_system_components']['system'],
                components['model_system_components']['backend'],
                components['model_system_components']['version'],
                components['model_system_components']['sol_mode'],
                components['runtime_config_components']['isl'],
                components['runtime_config_components']['osl'],
                components['runtime_config_components']['ttft'],                
                components['model_parallel_components']['num_gpus'],
                components['model_parallel_components']['tp_size'],
                components['model_parallel_components']['pp_size'],
                components['model_parallel_components']['dp_size'],
                components['model_parallel_components']['moe_tp_size'],
                components['model_parallel_components']['moe_ep_size'],
                components['model_quant_components']['gemm_quant_mode'],
                components['model_quant_components']['kvcache_quant_mode'],
                components['model_quant_components']['fmha_quant_mode'],
                components['model_quant_components']['moe_quant_mode'],
                components['model_quant_components']['comm_quant_mode'],
                components['model_misc_config_components']['nextn'],
                components['model_misc_config_components']['nextn_accept_rates'],
            ],
            outputs=[
                components['result_df'],
                components['pareto_html'],
                components['result_name'],
                components['save_btn'],
                components['debugging_box']
            ]
        )
        components['download_btn'].click(
            fn=EventFn.generate_csv,
            inputs=components['result_df'],
            outputs=components['output_file']
        )
        EventHandler.setup_common_events(components['model_name_components'], components['model_system_components'], components['model_quant_components'])
        EventHandler.setup_model_name_events(components['model_name_components'], components['model_quant_components'], components['model_parallel_components'], components['model_misc_config_components'])


    @staticmethod
    def setup_disagg_pareto_events(components):
        components['estimate_btn'].click(
            fn=EventFn.run_estimation_disagg_pareto,
            inputs=[
                components['model_name_components']['model_name'],# model
                components['runtime_config_components']['isl'], # runtime
                components['runtime_config_components']['osl'],
                components['runtime_config_components']['ttft'],
                components['model_misc_config_components']['nextn'],
                components['model_misc_config_components']['nextn_accept_rates'],
                components['prefill_model_system_components']['system'],  # prefill              
                components['prefill_model_system_components']['backend'],
                components['prefill_model_system_components']['version'],
                components['prefill_model_system_components']['sol_mode'],
                components['prefill_model_parallel_components']['num_worker'],
                components['prefill_model_parallel_components']['num_gpus'],
                components['prefill_model_parallel_components']['tp_size'],
                components['prefill_model_parallel_components']['pp_size'],
                components['prefill_model_parallel_components']['dp_size'],
                components['prefill_model_parallel_components']['moe_tp_size'],
                components['prefill_model_parallel_components']['moe_ep_size'], 
                components['prefill_model_quant_components']['gemm_quant_mode'], 
                components['prefill_model_quant_components']['kvcache_quant_mode'],
                components['prefill_model_quant_components']['fmha_quant_mode'],                
                components['prefill_model_quant_components']['moe_quant_mode'],
                components['prefill_model_quant_components']['comm_quant_mode'],
                components['prefill_correction_scale'],
                components['decode_model_system_components']['system'], # decode        
                components['decode_model_system_components']['backend'],
                components['decode_model_system_components']['version'],
                components['decode_model_system_components']['sol_mode'],
                components['decode_model_parallel_components']['num_worker'],
                components['decode_model_parallel_components']['num_gpus'],                
                components['decode_model_parallel_components']['tp_size'],
                components['decode_model_parallel_components']['pp_size'],
                components['decode_model_parallel_components']['dp_size'],
                components['decode_model_parallel_components']['moe_tp_size'],
                components['decode_model_parallel_components']['moe_ep_size'], 
                components['decode_model_quant_components']['gemm_quant_mode'], 
                components['decode_model_quant_components']['kvcache_quant_mode'],
                components['decode_model_quant_components']['fmha_quant_mode'],                
                components['decode_model_quant_components']['moe_quant_mode'],
                components['decode_model_quant_components']['comm_quant_mode'],
                components['decode_correction_scale'],
                components['num_gpu_list'],
                components['max_num_gpu'],
                components['prefill_max_num_worker'],
                components['decode_max_num_worker'],
                components['prefill_max_batch_size'],
                components['decode_max_batch_size'],
            ],
            outputs=[
                components['result_df'],
                components['pareto_html'],
                components['result_name'],
                components['save_btn'],
                components['debugging_box']
            ]
        )
        components['download_btn'].click(
            fn=EventFn.generate_csv,
            inputs=components['result_df'],
            outputs=components['output_file']
        )
        EventHandler.setup_common_events(components['model_name_components'], components['prefill_model_system_components'], components['prefill_model_quant_components'])
        EventHandler.setup_common_events(components['model_name_components'], components['decode_model_system_components'], components['decode_model_quant_components'])
        EventHandler.setup_model_name_events(components['model_name_components'], components['prefill_model_quant_components'], components['prefill_model_parallel_components'], components['model_misc_config_components'])
        EventHandler.setup_model_name_events(components['model_name_components'], components['decode_model_quant_components'], components['decode_model_parallel_components'], components['model_misc_config_components'])


    @staticmethod
    def setup_disagg_pd_ratio_events(components):
        components['estimate_btn'].click(
            fn=EventFn.run_estimation_disagg_pd_ratio,
            inputs=[
                components['model_name_components']['model_name'],# model
                components['runtime_config_components']['isl'], # runtime
                components['runtime_config_components']['osl'],
                components['runtime_config_components']['ttft'],
                components['runtime_config_components']['tpot'],                
                components['model_misc_config_components']['nextn'],
                components['model_misc_config_components']['nextn_accept_rates'],
                components['prefill_model_system_components']['system'],  # prefill              
                components['prefill_model_system_components']['backend'],
                components['prefill_model_system_components']['version'],
                components['prefill_model_system_components']['sol_mode'], 
                components['prefill_model_parallel_components']['tp_size'],
                components['prefill_model_parallel_components']['pp_size'],
                components['prefill_model_parallel_components']['dp_size'],
                components['prefill_model_parallel_components']['moe_tp_size'],
                components['prefill_model_parallel_components']['moe_ep_size'],                
                components['prefill_model_quant_components']['gemm_quant_mode'], 
                components['prefill_model_quant_components']['kvcache_quant_mode'],
                components['prefill_model_quant_components']['fmha_quant_mode'],                
                components['prefill_model_quant_components']['moe_quant_mode'],
                components['prefill_model_quant_components']['comm_quant_mode'],
                components['decode_model_system_components']['system'], # decode        
                components['decode_model_system_components']['backend'],
                components['decode_model_system_components']['version'],
                components['decode_model_system_components']['sol_mode'],
                components['decode_model_parallel_components']['tp_size'],
                components['decode_model_parallel_components']['pp_size'],
                components['decode_model_parallel_components']['dp_size'],
                components['decode_model_parallel_components']['moe_tp_size'],
                components['decode_model_parallel_components']['moe_ep_size'], 
                components['decode_model_quant_components']['gemm_quant_mode'], 
                components['decode_model_quant_components']['kvcache_quant_mode'],
                components['decode_model_quant_components']['fmha_quant_mode'],                
                components['decode_model_quant_components']['moe_quant_mode'],
                components['decode_model_quant_components']['comm_quant_mode'],
            ],
            outputs=[
                components['prefill_result_df'],
                components['prefill_throughput_html'],
                components['decode_result_df'],
                components['decode_throughput_html'],
                components['debugging_box']
            ]
        )
        components['download_btn'].click(
            fn=EventFn.generate_combined_csv,
            inputs=[components['prefill_result_df'], components['decode_result_df']],
            outputs=components['output_file']
        )
        EventHandler.setup_common_events(components['model_name_components'], components['prefill_model_system_components'], components['prefill_model_quant_components'])
        EventHandler.setup_common_events(components['model_name_components'], components['decode_model_system_components'], components['decode_model_quant_components'])
        EventHandler.setup_model_name_events(components['model_name_components'], components['prefill_model_quant_components'], components['prefill_model_parallel_components'], components['model_misc_config_components'])
        EventHandler.setup_model_name_events(components['model_name_components'], components['decode_model_quant_components'], components['decode_model_parallel_components'], components['model_misc_config_components'])

    @staticmethod
    def setup_save_events(result_name, save_btn, result_df, candidates_dropdown, pareto_results_state):
        save_btn.click(
            fn=EventFn.save_result_for_comparison,
            inputs=[result_name, result_df, pareto_results_state],
            outputs=[candidates_dropdown, save_btn, pareto_results_state]
        )

    @staticmethod
    def setup_pareto_comparison_events(components, pareto_results_state):
        components['compare_btn'].click(
            fn=EventFn.compare_results,
            inputs=[components['candidates_dropdown'], pareto_results_state],
            outputs=[components['pareto_html']]
        )
        components['clear_btn'].click(
            fn=EventFn.clear_results,
            inputs=[pareto_results_state],
            outputs=[components['candidates_dropdown'], pareto_results_state]
        )
        components['download_btn'].click(
            fn=EventFn.donwload_pareto_html,
            inputs=[components['pareto_html']],
            outputs=[components['output_file']]
        )

    # common events
    @staticmethod
    def setup_common_events(model_name_components, model_system_components, model_quant_components):
        model_name_components['model_name'].change(
            fn=EventFn.update_system_value,
            inputs=[model_name_components['model_name']],
            outputs=[
                model_system_components['system']
            ]
        )
        
        model_system_components['system'].change(
            fn=EventFn.update_backend_choices,
            inputs=[model_system_components['system']],
            outputs=[model_system_components['backend'], model_system_components['version']]
        )

        model_system_components['backend'].change(
            fn=EventFn.update_version_choices,
            inputs=[model_system_components['system'], model_system_components['backend']],
            outputs=[model_system_components['version']]
        )
        
        model_system_components['version'].change(
            fn=EventFn.update_quant_mode_choices,
            inputs=[model_name_components['model_name'], model_system_components['system'], model_system_components['backend'], model_system_components['version']],
            outputs=[
                model_quant_components['gemm_quant_mode'],
                model_quant_components['kvcache_quant_mode'],
                model_quant_components['fmha_quant_mode'],
                model_quant_components['moe_quant_mode']
            ]
        )
    
    @staticmethod
    def setup_model_name_events(model_name_components, model_quant_components, model_parallel_components, model_misc_config_components):
        model_name_components['model_name'].change(
            fn=EventFn.update_model_related_components,
            inputs=[model_name_components['model_name']],
            outputs=[
                model_misc_config_components['nextn'], 
                model_misc_config_components['nextn_accept_rates'],
                model_quant_components['moe_quant_mode'],
                model_parallel_components['moe_tp_size'],                
                model_parallel_components['moe_ep_size'],
                model_parallel_components['dp_size'],
            ]
        )
