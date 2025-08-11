# Advanced Tuning
In aiconfigurator, the inference framework and serving modeling is relatively complicated compared with the most simplified CLI entrypoint.  
For example, behind the command,
```bash
  aiconfigurator cli --model QWEN3_32B --total_gpus 512 --system h200_sxm
```
We hide a lot of default settings of the execution. Such as the quantization of each component, the matrix multiply, attention, moe, etc. We  
also hide the parallel config for how we search possible combinations.  

The optional params of cli contains the definition of ISL, OSL, TTFT and TPOT while we don't cover these params mentioned above. As it's a long 
list, we define it in a template under src/aiconfigurator/cli/templates/**_default.yaml

Let's take a look at deepseek_default.yaml
```yaml
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ============================= User config section begin =============================
# select a model from the following list:
# GPT_7B, GPT_13B, GPT_30B, GPT_66B, GPT_175B
# LLAMA2_7B, LLAMA2_13B, LLAMA2_70B, LLAMA3.1_8B, LLAMA3.1_70B, LLAMA3.1_405B
# MOE_Mixtral8x7B, MOE_Mixtral8x22B
# DEEPSEEK_V3
# QWEN2.5_1.5B, QWEN2.5_7B, QWEN2.5_32B, QWEN2.5_72B, QWEN3_32B, QWEN3_235B
nextn: 1 # mtp 1
nextn_accept_rates: [0.85,0,0,0,0] # each position maps to the accept rate of the ith draft token, nextn 1 will only use the first draft token accept rate.

# Your scenario and required SLA
isl: 4000 # input sequence length
osl: 1000 # output sequence length
ttft: 1000.0  # Target TTFT in ms
tpot: 40.0   # Target TPOT in ms

common_framework_config: &common_framework_config
  backend_name: "trtllm" # trtllm, sglang, vllm
  version: "0.20.0" # based on your local database

agg_system_name: &agg_system_name "h200_sxm" # h200_sxm, h100_sxm, depends on your local database
disagg_prefill_system_name: &disagg_prefill_system_name "h200_sxm" # h200_sxm, h100_sxm, depends on your local database
disagg_decode_system_name: &disagg_decode_system_name "h200_sxm" # h200_sxm, h100_sxm, depends on your local database
# ============================= User config section end =============================

# ----------------------------- no need to modify below --------------------------------


# you can modify system, framework(backend) and version to match your environment
# if you want to modify the quantization config, please refer to common.py in sdk
# ============================= search system config begin =============================
# agg(agg) config
agg_config:
  agg_worker_config:
    system_config:
      system_name: *agg_system_name
      <<: *common_framework_config
    quant_config:
      gemm_quant_mode: "fp8_block" # fp8, fp8_block, float16
      moe_quant_mode: "fp8_block" # fp8, fp8_block, w4afp8, float16
      kvcache_quant_mode: "float16" # fp8, int8, float16
      fmha_quant_mode: "float16" # fp8, float16
      comm_quant_mode: "half" # half
    parallel_config:
      num_gpu_per_worker: [4, 8]
      tp_list: [1, 2, 4, 8]
      pp_list: [1]
      dp_list: [1, 2, 4, 8]
      moe_tp_list: [1]
      moe_ep_list: [1, 2, 4, 8]

# disagg config
disagg_config:
  # the whole replica config, a replica is the minimum unit of disagg deployment. It contains xPyD workers.
  # x is the number of prefill workers, y is the number of decode workers
  # then we scale replicas to meet your total gpus requirement.
  replica_config:
    num_gpu_per_replica: [8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128] # It means the searched replica will have total gpus in this list, this list will be capped by max_gpu_per_replica
    max_gpu_per_replica: 128 # max gpus per replica, if specified as 0, it means no limit. Too many gpus per replica will make the prefill/decoder worker pair complicated. no need to be too large.
    max_prefill_worker: 32 # It means in every replica, you will have up to 32 prefill workers, x_max = 32
    max_decode_worker: 32 # It means in every replica, you will have up to 32 decode workers, y_max = 32
  # each prefill worker config
  prefill_worker_config:
    system_config:
      system_name: *disagg_prefill_system_name
      <<: *common_framework_config
    quant_config:
      gemm_quant_mode: "fp8_block" # fp8, fp8_block, float16
      moe_quant_mode: "fp8_block" # fp8, fp8_block, w4afp8, float16
      kvcache_quant_mode: "float16" # fp8, int8, float16
      fmha_quant_mode: "float16" # fp8, float16
      comm_quant_mode: "half" # half
    parallel_config:
      num_gpu_per_worker: [4, 8]
      tp_list: [1, 2, 4, 8]
      pp_list: [1]
      dp_list: [1] # we didn't enable attn dp here. You can enable it if you want.
      moe_tp_list: [1]
      moe_ep_list: [1, 2, 4, 8]
  # each decode worker config
  decode_worker_config:
    system_config:
      system_name: *disagg_decode_system_name
      <<: *common_framework_config
    quant_config:
      gemm_quant_mode: "fp8_block" # fp8, fp8_block, float16
      moe_quant_mode: "fp8_block" # fp8, fp8_block, w4afp8, float16
      kvcache_quant_mode: "float16" # fp8, int8, float16
      fmha_quant_mode: "float16" # fp8, float16
      comm_quant_mode: "half" # half
    parallel_config:
      num_gpu_per_worker: [4, 8]
      tp_list: [1, 2, 4, 8]
      pp_list: [1]
      dp_list: [1, 2, 4, 8]
      moe_tp_list: [1]
      moe_ep_list: [1, 2, 4, 8]

# advanced tuning config
prefill_correction_scale: 1.0 # If you find the predicted prefill perf is too optimistic, you can set a scale factor to make it more realistic, throughput_corrected = throughput_predicted * prefill_correction_scale
decode_correction_scale: 1.0 # If you find the predicted decode perf is too optimistic, you can set a scale factor to make it more realistic, throughput_corrected = throughput_predicted * decode_correction_scale
prefill_max_batch_size: 1
decode_max_batch_size: 512
```

In the `user config` section, we define user scenario and system. It's straighforward.  
Let's focus on search system config section. Let's take `disagg config` as an example,  
## replica config
A replica is defined the minimal scalable unit composed of xPyD, i.e., x prefill workers and y decode workers.  
In the replica config, we use a list `num_gpu_per_replica` to define how many gpus we can have in a replica. This parameter helps 
limit the max num gpu in a replica which avoids unreasonable results such as a single replica contains 2048 gpus. Even the theoretical perf 
is good, it's not practical. It also helps align the replica to a multiplier of 8, which aligns with num gpu in a typical server.  
`max_gpu_per_replica` is then capping the `num_gpu_per_replica` list if it's specified.  
`max_prefill_worker` and `max_decode_worker` limits the x and y of xPyD. This helps reduce the search space. In some extreme experiments, 
such as ISL:OSL is 8000:2, this will limit the disagg perf but in most cases, leave it to 32 makes sense.
## prefill/decode worker config
Once we have the xPyD config, let's look into the config of p or d worker.  
We have 3 sections, system_config, quant_config and parallel_config.
### system config
We also supporting using different system for prefill and decode to study the heterogenous perf. The default template is trying to use the same one for homogenous deployment.
### quant config
We allow users to specify different quant methods for different components even the framework doesn't support it for users to study perf impact. Choose the one you want.
Options are listed as comment. fp8 stands for fp8 per-tensor quant. fp8 block is for blockwise quant. float16 is bf16.
### parallel config
This is the most complicated part of the search space definition.  
First, `num_gpu_per_worker` is trying to define how many gpus in a worker, the searched result will do exact match.
Then, we define options for different components, tp for attention module, pp for transformer layer. Specifically for MoE, dp for attention data parallel, 
moe_tp for moe tensor parallel and moe_ep for moe expert paralell.  
Here's the pseudo code about how we enumerate valid configs based on the various list definitions,
```python
    for config in space[tp x pp x dp x moe_tp x moe_ep]:
        if config.tp * config.dp == config.moe_tp * config.moe_ep: # valid config, ensure the attention module has same gpus as ffn moe module
            if config.tp * config.dp * config.pp in num_gpus: # valid num_gpus
                yield config
```
All the valid combinations will print a line of log for each like this: `Enumerated Disagg decode parallel config: tp=1, pp=1, dp=1, moe_tp=1, moe_ep=1`  
We will then find a best one among these enumrations.
## advanced tuning config
The final tuning config is for some correction and deployment purpose.  
`prefill/decode_correction_scale` is to scale down the prefill/decode worker perf. If you find the predicted prefill perf is too optimistic, you can set a scale factor to make it more realistic, throughput_corrected = throughput_predicted * prefill_correction_scale. This can adjust the generated configs for better alignment with real deployment.  
`prefill/decode_max_batch_size`, in practical, you don't have to make decode batch size too large, 512 is a very high value. It's for local rank rather than the global batch size.  
And for prefill, for typical ISL larger than 1000, it's almost saturating the compute flops, doing batching will not give you too much perf gain but makes the TTFT x times.

## agg config
It's same for agg. You can treat agg as a prefill or decode worker.

## webapp
Webapp is actually a UI for these stuff. The logics are the same. They're building upon the same sdk.

## Practical suggestion
In order to save search time, you need to reduce the search space by choosing fewer parallel options. Say for `num_gpu_per_worker` here, it's DeepSeek V3 with 671B model 
parameters. With fp8_block, the rough estimation of the model weights is 671GB. You can not hold it on 4/2/1 gpus, you can modify it to `[8]` only. 
Of source, in most cases, we would like to have the default set work. Ideally, users don't have to modify them. But for specific perf studies, you can try it.
