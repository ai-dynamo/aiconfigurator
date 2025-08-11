# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gradio as gr

def create_readme_tab(app_config):
    with gr.Tab("Readme"):
        with gr.Accordion("Tabs introduction and Use cases"):
            use_cases_guide = gr.Markdown(
                label="tabs introduction",
                value=r'''
                ### Static tab
                    This tab is used to do static batching estimation. 
                    1. Configure model, runtime, system, quantization, parallel, misc, inference mode one by one.
                    2. Click on Estimate Static Inference button to do estimation.
                    3. It prints the detailed breakdown and summary of the performance.
                    4. Click on Download button to download the result dataframe.
                    In Debugging box, it prints the error message if any.
                #### Use cases
                    1. Working with ifb/disagg tab, facilitating the performance analysis. You can,
                        1.1 estimate memory consumption to check OOM and memory breakdown
                        1.2 check TTFT and TPOT via static batching by setting inference mode to static_ctx or static_gen.
                        1.3 get step breakdown, similar to nsys profiling, you will understand the dominant operation.


                ### Agg(IFB) Pareto Estimation tab
                    This tab is used to do Agg(IFB) Pareto estimation. Points on pareto frontier are the best performance you can get with the given system at a certain tokens/s/user.
                #### Use cases
                    1. Estimate the tokens/s/gpu at a certain tokens/s/user (1/tpot). Understand the trade-off between latency and throughput.
                    2. Reproduce a certain point at the pareto frontier.
                        2.1 sepcifically, you can bench the service using concurrency reported by the modeling here. Hover on the point to see the concurrency and record index.
                        2.2 you can get more details using the point record index and find that in Records section.
                        2.3 Note, this aiconfigurator will report more details about how to do the deployment. But you can simply set the max num tokens to isl + max_batch_size in trtllm. It will not affect the performance a lot.
                    3. In near future, we will generate a deployment script automatically.


                ### Disagg Pareto Estimation tab
                    This tab is used to do Disagg Pareto estimation. Points on pareto frontier are the best performance you can get with the given system at a certain tokens/s/user.
                #### Use cases
                    1. Estimate the tokens/s/gpu at a certain tokens/s/user (1/tpot). Understand the trade-off between latency and throughput.                
                    2. Reproduce a certain point at the pareto forntier. 
                        2.1 sepcifically, you can bench the service using concurrency reported by the modeling here. Using concurrency==num_decode_workers*decode_batch_size. 
                        2.2 you can get more details using the point record index and find that in Records section. 
                        2.3 Note, this aiconfigurator will report more details about how to do the deployment. Get info about num prefill/decoder workers, batchsize and parallel config. 
                            2.3.1 Today, it's better to set batch size of prefill worker to 1. As in concurrency-based bench method, it's difficult to saturate prefill batch size slot.
                            2.3.2 Batch size for decoder worker will not always be accurate. You can try with -10%~+10% of the reported batch size to evaluate the performance: tpot, throughput/gpu, etc.
                    3. In near future, we will generate a deployment script automatically.


                ### Pareto Comparison tab
                    This tab is used to do comparison between different pareto estimation results.
                #### Use cases
                    1. comparing ifb vs. disagg, to understand whether disagg might help in your case and find the best deployment strategy of your target SLA. Disagg is not always better. You need to specify your SLA.
                '''
            )

        with gr.Accordion("Parameters settings"):
            parameters_settings_guide = gr.Markdown(
                label="Parameters Settings",
                value=r'''
                #### ---> model name
                    Please select the model, if you need to specify a customed model, you need to modify the code for now.
                #### ---> Runtime config
                    Input sequence length, output sequence length, batch size or first token latency limit, etc.
                    1. For static mode, you specify the batch size. Context and generation phase will use the same batch size.
                    2. For pareto estimation, you need to specify the first token latency to filter out those results that have TTFT larger than the limit.
                #### ---> Misc config
                    For DeepSeek models, you need to define misc config about MTP and accpetance rate.
                    1. nextn
                        mtp0, mtp1, mtp2, mtp3, mtp4, mtp5.
                    2. nextn accept rates
                        The probability of the model accepting the input for a given nextn. It's a list of 5 numbers.
                        0.85,0,0,0,0 means 1st draft token is accepted with 85% probability. Starting from 2nd draft token, the probability is zero.                  
                #### ---> System config
                    GPU type, which framework to use, specific version of the framework.
                #### ---> Quantization config
                    Define quant config for each part of the model.
                    1. For dnese models, you will only see gemm quant config, kvcache quant config and fmha(context flash attention) quant config.
                    2. While for MoE models, you will see moe quant config. Different from frameworks, this allows you to specify the quant config for each part of the model separately to study the performance impact.
                #### ---> Parallel config(in static tab)
                    tp size and pp size. For MoE models, please define attention dp size, moe tp size and moe ep size additionally.
                    1. Please better not use pp today. It's not well aligned with the framework implementation.
                    2. For MoE models, dp means attention dp size. We allow using attention dp in attention module. In this case, please make sure
                           tp*dp == moe_tp*moe_ep
                    3. Today, please expect parallel size being larger than 8 is not well aligned with the framework. It's based on the theoretical extrapolation. Will improve in near future.
                #### ---> Parallel config(in pareto estimation tabs)
                    In pareto estimation tab, specifically, parallel config might be much more complex than static estimation. It's defining a search space instead of a single config.
                    1. num gpus
                        Defines the the number of GPUs in the system that will be searched.
                    2. tp/pp/dp/moe_tp/moe_ep
                        Defines allowed parallel size to be searched for. The search space is defined as
                        ```
                        for config in space[tp x pp x dp x moe_tp x moe_ep]:
                            if config.tp * config.dp == config.etp * config.ep: # valid config
                                if config.tp * config.dp in num_gpus: # valid num_gpus
                                    yield config
                        ```
                        All enumerated functional configs will be printed in Debugging box.
                    3. num workers
                        Specifically, in disagg tab, you will need to configure num workers before specific parallel config. Num workers is used to do rate matching of prefill and decode workers.
                        num_prefill_workers * seq_throughput_per_prefill_worker <=matches=> num_decode_workers * decode_throughput_per_prefill_worker
                        This parameter controls num worker here. -1 for auto, which we will try to find a good rate matching. 
                        If a specific number is given, it will do fixed rate matching without searching. This is useful when you want to verify a certain p:d ratio real bench.
                #### ---> inference mode
                    Define the inference mode in static tab: static, static_ctx(only computes context phase), static_gen(only computes generation phase).
                    This mode can be leveraged to estimate a single prefill forward execution time (TTFT). configure isl=isl, osl=1, bs=1. Select static_ctx mode.
                    This mode can also be leveraged to do estimation of a single generation step. configure isl=isl+target_step, osl=2, bs=bs. Select static_gen mode. This is equivalent to target_step in e2e inference in an exact same way.
                #### ---> Advanced settings (in disagg tab)
                    1. prefill/decode correction scales
                        this is used to correct the throughput of prefill/decode worker to do manual adjustment of prefill/decode rate matching. The tool will not be 100% correct. You can do some correction by yourself.
                    2. num total gpu list of the disagg system, say:8,16 or max gpus used in the disagg system
                        When we are doing rate matching to find a good prefill/decode worker ratio, we're not limiting the total number of GPUs in the disagg system.
                        This will sometimes make the result difficult to reproduce. 
                            Say, if you find 17 prefill workers, each prefill worker has 8 gpus; The matched decode worker number is 23, each worker has 4 gpus.
                            Then the system has 17*8+23*4=228 gpus. Much more difficult than a 16-gpu disagg system to reproduce.
                        Thus we can add some constraints to the search space. (The perf gap will not be very very small actually. You can use comparison tab to verify this gap.)
                        a. one option is to set a target number of total gpus used in the disagg system. Then, the num total gpu list helps here. 8,16 means we only allow the disagg system which has exactly 8 or 16 gpus.
                        b. another option is to set a max number of gpus used in the disagg system. Then, the max gpus used in the disagg system helps here. The returned result will have gpus fewer than max gpus defined here.
                    3. prefill/decode max num instance
                        This is used to limit the number of prefill/decode workers when we do rate matching search. By default, it's 32. Typically no need to tune.
                        In some extreme cases, you need to specify a larger number, say when isl is 100000000 and osl is 2. You need many many more prefill workers vs. decode workers.
                    4. prefill/decode max batch size
                        This is used to limit the batch size of the prefill/decode workers when we do rate matching search. By default, it's 1 for prefill and 512 for decode.
                        Why 1 for prefill: in concurrency-based bench method, it's difficult to saturate prefill batch size slot.
                        If you isl is really small, say isl*prefill_batch_size is smaller than 1000, you can slightly increase the prefill batch size to 2 or 4 to make isl*prefill_batch_size larger than 1000.
                '''
            )

        with gr.Accordion("Troubleshooting"):
            troubleshooting_guide = gr.Markdown(
                label="troubleshooting",
                value=r'''
                ### Debugging box
                    Debugging box is contained in each tab. It prints the error message if any.
                #### interpolation error
                    It will report that there's no enough data points to interpolate. report as an error with traceback. 
                    Please check your runtime input (isl, osl) and parallel config. Choose a different parallel size. A super large isl will be invalid. Now we support ~1M isl.
                #### no result or none reported
                    Typically caused by strict limitation of TTFT or a small parallel size where no non-oom data points are found. You can try to set a larger TTFT limit.
                #### parallel config error
                    Fix as the error message says.
                ### hang
                    A typicall run of ifb pareto estimation takes 1~2 minutes with ~5 enumerated configs. Linear to how many parallel options you have.
                    A typical run of disagg pareto estimation takes ~30 seconds with ~5 enumerated configs. Linear to how many parallel options you have.
                    A run longer than 10 minutes is abnormal. Try to refresh the page. Set smaller search space by selecting fewer parallel options.
                ### Others
                    If you have any other questions, please report to us.
                '''
            )
        
        with gr.Accordion("Known Limitations"):
            known_limitations_guide = gr.Markdown(
                label="known limitations",
                value=r'''
                ### Inaccurate modeling
                    1. pipeline parallel is not well supported.
                    2. inter-node communication modeling is not well supported.
                    3. fine-grained moe large(wide)-ep is not well supported.
                    4. results might be too optimistic when batchsize is large.
                    5. kvcache transfer is not modeled in disagg.
                ### Usability
                    1. The Record dataframe's height can be 0 after using horizontal scroll bar. Try to use the vertical scroll bar to scroll back.
                '''
            )


    return {
        "parameters_settings_guide": parameters_settings_guide,
        "tabs_introduction_and_use_cases_guide": use_cases_guide,
        "troubleshooting_guide": troubleshooting_guide,
        "known_limitations_guide": known_limitations_guide
    }