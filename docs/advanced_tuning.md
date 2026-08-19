# Advanced Tuning

> **YAML format:** Experiment YAML uses the flat `Task` schema — every key maps
> 1:1 to a `Task` field, with no `mode:` selector and no `config:` /
> `worker_config:` nesting. The examples below are in this format; see
> [`example.yaml`](../src/aiconfigurator/cli/example.yaml) for the full template.
>
> The legacy V1 nested format is **deprecated** and kept only behind a limited
> compatibility shim: it auto-converts to V2 with a `DeprecationWarning`, and
> any field with no V2 equivalent is rejected rather than silently dropped. See
> [`example_v1_deprecated.yaml`](../src/aiconfigurator/cli/example_v1_deprecated.yaml)
> for the old shape.

In aiconfigurator, the inference framework and serving modeling is relatively complicated compared with the most simplified CLI entrypoint.  
For example, behind the command,
```bash
aiconfigurator cli default --model-path Qwen/Qwen3-32B-FP8 --total-gpus 512 --system h200_sxm
```
We hide a lot of default settings of the execution. Such as the quantization of each component, the matrix multiply, attention, moe, etc. We  
also hide the parallel config for how we search possible combinations.  

The optional params of cli contains the definition of ISL, OSL, TTFT and TPOT while we don't cover these params mentioned above. In CLI, We auto populate all these stuff for `default` mode and allow users to modify in `exp` mode.
```bash
aiconfigurator cli exp --yaml-path example.yaml
```
The example.yaml is defined [here](../src/aiconfigurator/cli/example.yaml).  
Let's take a look at example.yaml
```yaml
# agg_full: aggregated, full control. Use as a template.
agg_full:
  serving_mode: agg                    # required
  model_path: deepseek-ai/DeepSeek-V3  # required
  system_name: h200_sxm                # required
  total_gpus: 8                        # required
  backend_name: trtllm                 # trtllm (default) | vllm | sglang
  isl: 4000
  osl: 1000
  ttft: 1000.0
  tpot: 40.0
  # Large-EP (wideEP) is explored automatically when perf data covers the model
  # shape; restrict with *_moe_ep_candidates (see "large-EP exploration" below).
  moe_backend: null
  # Speculative decoding (MTP): opt-in only; nextn_accepted is required
  # when nextn > 0 and must lie in [0, nextn].
  nextn: 1
  nextn_accepted: 0.85
  # Quantization of each component (default: inferred from HF config)
  gemm_quant_mode: fp8_block           # fp8 | fp8_block | bfloat16
  moe_quant_mode: fp8_block            # fp8 | fp8_block | w4afp8 | bfloat16
  kvcache_quant_mode: bfloat16         # fp8 | int8 | bfloat16
  fmha_quant_mode: bfloat16            # fp8 | bfloat16
  comm_quant_mode: half
  # Parallelism search space
  agg_num_gpu_candidates: [4, 8]
  agg_tp_candidates: [1, 2, 4, 8]
  agg_pp_candidates: [1]
  agg_dp_candidates: [1, 2, 4, 8]
  agg_moe_tp_candidates: [1]
  agg_moe_ep_candidates: [1, 2, 4, 8]

# disagg_full: disaggregated, full control. Use as a template.
disagg_full:
  serving_mode: disagg                 # required
  total_gpus: 32                       # required
  isl: 4000
  osl: 1000
  ttft: 1000.0
  tpot: 40.0
  # MTP is opt-in; nextn_accepted required when nextn > 0, in [0, nextn].
  nextn: 1
  nextn_accepted: 0.85
  # MoE kernel backend (shared). Large-EP (wideEP) is explored automatically
  # when perf data covers the model shape; restrict with *_moe_ep_candidates.
  moe_backend: null

  # --- Prefill worker ---
  prefill_model_path: deepseek-ai/DeepSeek-V3
  prefill_system_name: h200_sxm
  prefill_backend_name: trtllm
  prefill_gemm_quant_mode: fp8_block
  prefill_moe_quant_mode: fp8_block
  prefill_kvcache_quant_mode: bfloat16
  prefill_fmha_quant_mode: bfloat16
  prefill_comm_quant_mode: half
  prefill_num_gpu_candidates: [4, 8]
  prefill_tp_candidates: [1, 2, 4, 8]
  prefill_pp_candidates: [1]
  prefill_dp_candidates: [1]            # attention DP off here; raise to enable
  prefill_moe_tp_candidates: [1]
  prefill_moe_ep_candidates: [1, 2, 4, 8]

  # --- Decode worker (model_path must equal the prefill model) ---
  decode_model_path: deepseek-ai/DeepSeek-V3
  decode_system_name: h200_sxm
  decode_backend_name: trtllm
  decode_gemm_quant_mode: fp8_block
  decode_moe_quant_mode: fp8_block
  decode_kvcache_quant_mode: bfloat16
  decode_fmha_quant_mode: bfloat16
  decode_comm_quant_mode: half
  decode_num_gpu_candidates: [4, 8]
  decode_tp_candidates: [1, 2, 4, 8]
  decode_pp_candidates: [1]
  decode_dp_candidates: [1, 2, 4, 8]
  decode_moe_tp_candidates: [1]
  decode_moe_ep_candidates: [1, 2, 4, 8]

  # --- Replica shaping + perf correction (disagg only) ---
  num_gpu_per_replica: [8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128]
  max_gpu_per_replica: 128
  max_prefill_workers: 32
  max_decode_workers: 32
  prefill_latency_correction: 1.1
  decode_latency_correction: 1.08
  prefill_max_batch_size: 1
  decode_max_batch_size: 512
```
We keep only the full agg and disagg versions here. Note:  
1. The worker spec is **top-level for agg** (`gemm_quant_mode`, `agg_tp_candidates`, ...) and **per-role for disagg** (`prefill_*` / `decode_*`); the two roles look very similar.  
2. Disagg additionally has the replica-shaping fields (`num_gpu_per_replica`, `max_*_workers`) and the correction fields (`*_latency_correction`, `*_max_batch_size`).  
Let's discuss them. Please refer to [CLI user guide](cli_user_guide.md) for basic info and as a pre-reading.

Let's focus on search system config section. Let's take `disagg config` as an example,  
## replica config
A replica is defined the minimal scalable unit composed of xPyD, i.e., x prefill workers and y decode workers.  
In the replica config, we use a list `num_gpu_per_replica` to define how many gpus we can have in a replica. This parameter helps 
limit the max num gpu in a replica which avoids unreasonable results such as a single replica contains 2048 gpus. Even the theoretical perf 
is good, it's not practical. It also helps align the replica to a multiplier of 8, which aligns with num gpu in a typical server.  
`max_gpu_per_replica` is then capping the `num_gpu_per_replica` list if it's specified.  
`max_prefill_workers` and `max_decode_workers` limit the x and y of xPyD. This helps reduce the search space. In some extreme experiments, 
such as ISL:OSL is 8000:2, this will limit the disagg perf but in most cases, leave it to 32 makes sense.
## prefill/decode worker config
Once we have the xPyD config, let's look into the config of p or d worker.  
We have two types of setting, quantization and parallelism.
### quantization (gemm_quant_mode, etc.)
We allow users to specify different quant methods for different components even the framework doesn't support it for users to study perf impact. Choose the one you want.
Options are listed as comment. fp8 stands for fp8 per-tensor quant. fp8 block is for blockwise quant. bfloat16 is bf16.

Quantization defaults are inferred from the Hugging Face model config (`config.json` plus optional `hf_quant_config.json`).  
Setting any `*_quant_mode` field explicitly overrides those defaults.
### parallelism (`*_num_gpu_candidates`, `*_tp_candidates`, etc.)
This is the most complicated part of the search space definition. Each dimension is a per-role list: `agg_*` for agg, `prefill_*` / `decode_*` for disagg.  
First, `*_num_gpu_candidates` defines how many GPUs in a worker; the searched result will do exact match.
Then, we define options for different components, tp for attention module, pp for transformer layer. Specifically for MoE, dp for attention data parallel, 
moe_tp for moe tensor parallel and moe_ep for moe expert parallel.
Here's the pseudo code about how we enumerate valid configs based on the various list definitions,
```python
    for config in space[tp x pp x dp x moe_tp x moe_ep]:
        if config.tp * config.dp == config.moe_tp * config.moe_ep: # valid config, ensure the attention module has same gpus as ffn moe module
            if config.tp * config.dp * config.pp in num_gpus: # valid num_gpus
                yield config
```
All the valid combinations will print a line of log for each like this: `Enumerated Disagg decode parallel config: tp=1, pp=1, dp=1, moe_tp=1, moe_ep=1`  
We will then find a best one among these enumrations.
### large-EP (wideEP) exploration
Large-EP — multi-node, EP-only MoE parallelism, formerly gated behind the `enable_wideep` flag — no longer has a switch. For every MoE model, aiconfigurator probes whether the role's performance database covers the model's MoE shape (the MoE all-to-all dispatch/combine data plus the EP compute data, under the resolved `moe_quant_mode`). When it does, the default search lists are widened with a multi-node EP ladder (`moe_ep` up to 64) and the large-EP tuples compete with the fused configs inside the same search; the MoE communication backend is resolved per tuple from data coverage.

When the shape is not covered, large-EP exploration stays off for that model/system and a one-time INFO log tells you exactly what to collect:
```text
large-EP exploration is OFF for deepseek-ai/DeepSeek-V3 on h200_sxm/trtllm: no MoE all-to-all + EP-compute coverage for this model shape (hidden=7168, topk=8, experts=256) under moe_quant_mode=fp8_block. Run the moe_a2a and moe_ep collectors for this model/system to enable it; the fused (small-EP) path is unaffected.
```
Run the named collectors (see `collector/`) for the model/system to enable it — no config change is needed afterwards.

To restrict or force EP sizes, set `*_moe_ep_candidates` explicitly: `decode_moe_ep_candidates: [16, 32, 64]` pins decode to large-EP tuples only, while `[1, 2, 4, 8]` keeps the search single-node. The deprecated keys (`enable_wideep`, `prefill_enable_wideep`, `decode_enable_wideep`, `moe_backend: deepep_moe`) are still accepted with a one-time warning and have no modeling effect; the one search-default residue is on SGLang, where spelling them still narrows the *default* `moe_tp` candidates to `[1]` — an explicit `*_moe_tp_candidates` list always wins.
The recommend-mode CLI spellings `--enable-wideep` and `--moe-backend deepep_moe` follow the same deprecated-and-ignored behavior.
## advanced tuning config
The final tuning config is for some correction and deployment purpose.  
`prefill_latency_correction` / `decode_latency_correction` scale the predicted prefill/decode worker perf. If you find the predicted latency too optimistic, set a factor to make it more realistic: `latency_corrected = latency_predicted * latency_correction`. This adjusts the generated configs for better alignment with real deployment.  
`prefill/decode_max_batch_size`, in practical, you don't have to make decode batch size too large, 512 is a very high value. It's for local rank rather than the global batch size.  
And for prefill, for typical ISL larger than 1000, it's almost saturating the compute flops, doing batching will not give you too much perf gain but makes the TTFT x times.

## agg config
It's same for agg. You can treat agg as a prefill or decode worker.

## EPD (encoder disaggregation) config
For vision-language workloads (image inputs set), `enable_epd: true` serves the vision encoder from a dedicated
encode-worker pool instead of colocated with the LM workers — agg becomes E+agg, disagg becomes E+P+D. The encode
pool is rate-matched against the LM pools; result rows carry `(e)workers`/`(e)tp`/`(e)bs` columns. Optional knobs
(all require `enable_epd`; keys are Task field names, not CLI flags):

- `encoder_tp_candidates`: encode-worker TP sizes to sweep (default `[1, 2, 4, 8]`)
- `encoder_batch_candidates`: encode batch sizes to sweep (default `[1, 2, 4, 8]`; capped at 8, SGLang's
  `SGLANG_ENCODER_MAX_BATCH_SIZE` default)
- `max_encoder_workers`: encode-worker cap per rate-matched cell (default 32)
- `encoder_latency_correction`: encode-latency correction scale (default 1.0)
- `encoder_system_name`: system (GPU type) for a heterogeneous encode pool; backend and version follow the P/agg side
- `rate_match_encoder_degradation`: encode-pool rate-matching degradation (default 0.9, alongside the
  prefill/decode factors)

See the `vl_epd_agg` experiment in `src/aiconfigurator/cli/example.yaml` for a complete template.

## Practical suggestion
In order to save search time, you need to reduce the search space by choosing fewer parallel options. Say for `*_num_gpu_candidates` here, it's DeepSeek V3 with 671B model 
parameters. With fp8_block, the rough estimation of the model weights is 671GB. You can not hold it on 4/2/1 gpus, you can modify it to `[8]` only. 
Of source, in most cases, we would like to have the default set work. Ideally, users don't have to modify them. But for specific perf studies, you can try it.
