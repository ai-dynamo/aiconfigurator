# Advanced Tuning
In aiconfigurator, the inference framework and serving modeling is relatively complicated compared with the most simplified CLI entrypoint.  
For example, behind the command,
```bash
  aiconfigurator cli default --model QWEN3_32B --total_gpus 512 --system h200_sxm
```
We hide a lot of default settings of the execution. Such as the quantization of each component, the matrix multiply, attention, moe, etc. We  
also hide the parallel config for how we search possible combinations.  

The optional params of cli contains the definition of ISL, OSL, TTFT and TPOT while we don't cover these params mentioned above. In CLI, We auto populate all these stuff for `default` mode and allow users to modify in `exp` mode. Since webapp follows the same logic, we take CLI as an example,
```bash
  aiconfigurator cli exp --yaml_path example.yaml
```
The example.yaml is defined [here](../src/aiconfigurator/cli/example.yaml).  
Let's take a look at example.yaml
```yaml
# exp_agg_full, agg, full config, please refer to this as a template.
exp_agg_full:
  mode: "patch" # patch or replace the config section, required
  serving_mode: "agg" # required
  model_name: "DEEPSEEK_V3" # required
  total_gpus: 8 # required
  system_name: "h200_sxm" # required
  backend_name: "trtllm" # optional, default to trtllm
  backend_version: "0.20.0" # optional, default to the latest version in the database
  isl: 4000 # input sequence length, optional, default to 4000
  osl: 1000 # output sequence length, optional, default to 1000
  ttft: 1000.0  # Target TTFT in ms, optional, default to 1000.0
  tpot: 40.0   # Target TPOT in ms, optional, default to 40.0
  enable_wideep: false # enable wide ep for prefill/decode, optional, default to false
  profiles: [] # some inherit presets for easier patch, optional
  config: # all optional, used to patch default values
    nextn: 1 # mtp 1
    nextn_accept_rates: [0.85,0,0,0,0] # each position maps to the accept rate of the ith draft token, nextn 1 will only use the first draft token accept rate.
    worker_config: # defines quantization of each component
      gemm_quant_mode: "fp8_block" # fp8, fp8_block, float16
      moe_quant_mode: "fp8_block" # fp8, fp8_block, w4afp8, float16
      kvcache_quant_mode: "float16" # fp8, int8, float16
      fmha_quant_mode: "float16" # fp8, float16
      comm_quant_mode: "half" # half
      num_gpu_per_worker: [4, 8] # num gpus per worker, please refer to enumerate_parallel_config in utils.py
      tp_list: [1, 2, 4, 8]
      pp_list: [1]
      dp_list: [1, 2, 4, 8]
      moe_tp_list: [1]
      moe_ep_list: [1, 2, 4, 8]

# exp_disagg_full, disagg, full config, please refer to this as a template.
exp_disagg_full:
  mode: "patch" # patch or replace the config section, required
  serving_mode: "disagg" # required
  model_name: "DEEPSEEK_V3" # required
  total_gpus: 32 # required
  system_name: "h200_sxm" # required
  decode_system_name: "h200_sxm" # optional, if not provided, it will use the same system name as the prefill system.
  backend_name: "trtllm" # optional, default to trtllm
  backend_version: "0.20.0" # optional, default to the latest version in the database
  isl: 4000 # input sequence length, optional, default to 4000
  osl: 1000 # output sequence length, optional, default to 1000
  ttft: 1000.0  # Target TTFT in ms, optional, default to 1000.0
  tpot: 40.0   # Target TPOT in ms, optional, default to 40.0
  enable_wideep: false # enable wide ep for prefill/decode, optional, default to false
  profiles: [] # some inherit presets for easier patch, optional
  config: # all optional, used to patch default values
    nextn: 1 # mtp 1
    nextn_accept_rates: [0.85,0,0,0,0] # each position maps to the accept rate of the ith draft token, nextn 1 will only use the first draft token accept rate.
    # each prefill worker config
    prefill_worker_config:
      gemm_quant_mode: "fp8_block" # fp8, fp8_block, float16
      moe_quant_mode: "fp8_block" # fp8, fp8_block, w4afp8, float16
      kvcache_quant_mode: "float16" # fp8, int8, float16
      fmha_quant_mode: "float16" # fp8, float16
      comm_quant_mode: "half" # half
      num_gpu_per_worker: [4, 8] # num gpus per worker, please refer to enumerate_parallel_config in utils.py
      tp_list: [1, 2, 4, 8]
      pp_list: [1]
      dp_list: [1] # we didn't enable attn dp here. You can enable it if you want.
      moe_tp_list: [1]
      moe_ep_list: [1, 2, 4, 8]
    # each decode worker config
    decode_worker_config:
      gemm_quant_mode: "fp8_block" # fp8, fp8_block, float16
      moe_quant_mode: "fp8_block" # fp8, fp8_block, w4afp8, float16
      kvcache_quant_mode: "float16" # fp8, int8, float16
      fmha_quant_mode: "float16" # fp8, float16
      comm_quant_mode: "half" # half
      num_gpu_per_worker: [4, 8] # num gpus per worker, please refer to enumerate_parallel_config in utils.py
      tp_list: [1, 2, 4, 8]
      pp_list: [1]
      dp_list: [1, 2, 4, 8]
      moe_tp_list: [1]
      moe_ep_list: [1, 2, 4, 8]
    # the whole replica config, a replica is the minimum unit of disagg deployment. It contains xPyD workers.
    # x is the number of prefill workers, y is the number of decode workers
    # then we scale replicas to meet your total gpus requirement.
    replica_config:
      num_gpu_per_replica: [8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128] # It means the searched replica will have total gpus in this list, this list will be capped by max_gpu_per_replica
      max_gpu_per_replica: 128 # max gpus per replica, if specified as 0, it means no limit. Too many gpus per replica will make the prefill/decoder worker pair complicated. no need to be too large.
      max_prefill_worker: 32 # It means in every replica, you will have up to 32 prefill workers, x_max = 32
      max_decode_worker: 32 # It means in every replica, you will have up to 32 decode workers, y_max = 32
    advanced_tuning_config:
      # advanced tuning config
      prefill_latency_correction_scale: 1.1 # If you find the predicted prefill latency is too optimistic, you can set a scale factor to make it more realistic, prefill_latency_corrected = prefill_latency * prefill_latency_correction_scale
      decode_latency_correction_scale: 1.08 # If you find the predicted decode perf is too optimistic, you can set a scale factor to make it more realistic, decode_latency_corrected = decode_latency * decode_latency_correction_scale
      prefill_max_batch_size: 1
      decode_max_batch_size: 512
```
We deleted the simplified exp definition and only keep the full version of agg and disagg. You can easily find  
1. there's a `agg_worker_config` in agg mode and `prefill_worker_config` and `decode_worker_config` in disagg mode. They look very similar  
2. additionally for disagg, there are two sections, `replica_config` and `advanced_tuning_config`  
Let's discuss them. Please refer to [CLI user guide](cli_user_guide.md) for basic info and as a pre-reading.

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
We have two types of setting, quantization and parallelism.
### quantization (gemm_quant_mode, etc.)
We allow users to specify different quant methods for different components even the framework doesn't support it for users to study perf impact. Choose the one you want.
Options are listed as comment. fp8 stands for fp8 per-tensor quant. fp8 block is for blockwise quant. float16 is bf16.
### parallelism (num_gpus_per_worker, tp_list, etc.)
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
`prefill/decode_latency_correction_scale` is to scale down the prefill/decode worker perf. If you find the predicted prefill latency is too optimistic, you can set a scale factor to make it more realistic, latency_corrected = latency_predicted * latency_correction_scale. This can adjust the generated configs for better alignment with real deployment.  
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
