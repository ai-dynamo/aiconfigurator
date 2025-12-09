# CLI User Guide
## Basic Command
As mentioned in root Readme, CLI supports two modes, `default` and `exp`. We'll go through these two modes one by one.

### Default mode
This mode is triggered by
```bash
aiconfigurator cli default --model QWEN3_32B --total_gpus 32 --system h200_sxm
or
aiconfigurator cli default --model QWEN3_32B --total_gpus 32 --system h200_sxm --ttft 1000 --tpot 10 --isl 3000 --osl 512 --prefix 0
```
`model`, `total_gpus`, `system` are three required arguments to define the problem.  
If you want to specify your problem with more details, we allow to define `ttft`, `tpot`, `isl`, `osl` and `prefix`.

#### Backend Selection

You can specify which inference backend to use with the `--backend` flag:

```bash
# Use TensorRT-LLM (default)
aiconfigurator cli default --model QWEN3_32B --total_gpus 32 --system h200_sxm --backend trtllm

# Use vLLM (dense models only, currently being evaluated)
aiconfigurator cli default --model QWEN3_32B --total_gpus 32 --system h200_sxm --backend vllm

# Use SGLang (dense and MoE models, currently being evaluated)
aiconfigurator cli default --model QWEN3_32B --total_gpus 32 --system h200_sxm --backend sglang
```

The command will create two experiments for the given problem, one is `agg` and another one is `disagg`. Compare them to find the better one and estimates the perf gain.

The command will print out the result to your terminal with the basic info of the comparison, the pareto curve (the best point is tagged as `x`), 
the worker setup for your reference. Let's split them into sections.

Let's run `aiconfigurator cli default --model QWEN3_32B --total_gpus 32 --system h200_sxm --ttft 1000 --tpot 10 --isl 3000 --osl 512 --prefix 0`
> Note that the result might differ based on different versions of your aiconfigurator.
1. Basic info of the comparison
```
  Input Configuration & SLA Target:
    Model: QWEN3_32B (is_moe: False)
    Total GPUs: 32
    Best Experiment Chosen: disagg at 913.82 tokens/s/gpu (1.43x better)
  ----------------------------------------------------------------------------
  Overall Best Configuration:
    - Best Throughput: 29,242.24 tokens/s
    - Per-GPU Throughput: 913.82 tokens/s/gpu
    - Per-User Throughput: 123.92 tokens/s/user
    - TTFT: 202.65ms
    - TPOT: 8.07ms
```
This shows that for model `QWEN3_32B` to deploy on 32 H200, if you require your TTFT to be less than 1000ms and TPOT to be less than 10ms, and your problem is isl=3000 osl=512, then disagg will be 1.43x of agg. The target result is shown as Overall Best Configuration.

2. Pareto frontier
```
  Pareto Frontier:
              QWEN3_32B Pareto Frontier: tokens/s/gpu vs tokens/s/user          
    ┌──────────────────────────────────────────────────────────────────────────┐
2250┤ •• disagg                                                                │
    │ ff agg                                                                   │
    │ xx disagg best                                                           │
    │                                                                          │
1875┤  ff                                                                      │
    │   fff                                                                    │
    │     ff                                                                   │
    │      fff••                                                               │
1500┤         f •••                                                            │
    │         ff   ••••••••                                                    │
    │          ffff       •                                                    │
    │              f       •••••••                                             │
1125┤               ff            •                                            │
    │                ff            ••••                                        │
    │                  ffff            ••••x                                   │
    │                     fff              ••••                                │
 750┤                        fff               •                               │
    │                          ffffff           •                              │
    │                                ffffff      ••                            │
    │                                      fffffff ••••••                      │
 375┤                                             ff    •                      │
    │                                               fffffff•••••••••           │
    │                                                      ffffffffff          │
    │                                                                          │
   0┤                                                                          │
    └┬─────────────────┬──────────────────┬─────────────────┬─────────────────┬┘
     0                60                 120               180              240 
tokens/s/gpu                        tokens/s/user                               
```
Pareto frontier shows the trade-off betwen generation speed `tokens/s/user` and throughput `tokens/s/gpu`. The best points is tagged as `x`. As you want the TPOT to be less than 10ms, which means the generation speed is faster than 1000/10ms = 100 tokens/s/user, then by reading the pareto froniter, you will get the point tagged as x. You can see that, if you want different TPOT, you will have different result. Sometimes, agg will be better than disagg (higher throughput at same tokens/s/user)

3. Worker setup
```
  Deployment Details:
    (p) stands for prefill, (d) stands for decode, bs stands for batch size, a replica stands for the smallest scalable unit xPyD of the disagg system
    Some math: total gpus used = replicas * gpus/replica
               gpus/replica = (p)gpus/worker * (p)workers + (d)gpus/worker * (d)workers; for Agg, gpus/replica = gpus/worker
               gpus/worker = tp * pp * dp = etp * ep * pp for MoE models; tp * pp for dense models (underlined numbers are the actual values in math)

disagg Top Configurations: (Sorted by tokens/s/gpu)
+------+--------------+---------------+--------+-------------+------------------+----------+----------------+------------+----------------+-------------+-------+------------+----------------+-------------+-------+
| Rank | tokens/s/gpu | tokens/s/user |  TTFT  | concurrency | total_gpus(used) | replicas |  gpus/replica  | (p)workers | (p)gpus/worker | (p)parallel | (p)bs | (d)workers | (d)gpus/worker | (d)parallel | (d)bs |
+------+--------------+---------------+--------+-------------+------------------+----------+----------------+------------+----------------+-------------+-------+------------+----------------+-------------+-------+
|  1   |    913.82    |     123.92    | 202.65 |  256(=64x4) |   32 (32=4x8)    |    4     |  8 (=4x1+1x4)  |     4      |    1 (=1x1)    |    tp1pp1   |   1   |     1      |    4 (=4x1)    |    tp4pp1   |   64  |
|  2   |    873.07    |     126.28    | 202.65 |  240(=60x4) |   32 (32=4x8)    |    4     |  8 (=4x1+1x4)  |     4      |    1 (=1x1)    |    tp1pp1   |   1   |     1      |    4 (=4x1)    |    tp4pp1   |   60  |
|  3   |    852.77    |     133.94    | 202.65 | 240(=240x1) |   32 (32=1x32)   |    1     | 32 (=12x1+5x4) |     12     |    1 (=1x1)    |    tp1pp1   |   1   |     5      |    4 (=4x1)    |    tp4pp1   |   48  |
|  4   |    568.51    |     148.82    | 202.65 |  144(=72x2) |   32 (32=2x16)   |    2     | 16 (=4x1+3x4)  |     4      |    1 (=1x1)    |    tp1pp1   |   1   |     3      |    4 (=4x1)    |    tp4pp1   |   24  |
|  5   |    434.77    |     145.12    | 123.20 | 104(=104x1) |   32 (24=1x24)   |    1     | 24 (=4x2+4x4)  |     4      |    2 (=2x1)    |    tp2pp1   |   1   |     4      |    4 (=4x1)    |    tp4pp1   |   26  |
+------+--------------+---------------+--------+-------------+------------------+----------+----------------+------------+----------------+-------------+-------+------------+----------------+-------------+-------+

agg Top Configurations: (Sorted by tokens/s/gpu)
+------+--------------+---------------+--------+-------------+------------------+----------+--------------+-------------+----------+----+
| Rank | tokens/s/gpu | tokens/s/user |  TTFT  | concurrency | total_gpus(used) | replicas | gpus/replica | gpus/worker | parallel | bs |
+------+--------------+---------------+--------+-------------+------------------+----------+--------------+-------------+----------+----+
|  1   |    638.28    |     100.97    | 187.72 |  224(=28x8) |   32 (32=8x4)    |    8     |      4       |   4 (=4x1)  |  tp4pp1  | 28 |
|  2   |    612.50    |     101.49    | 274.98 | 224(=14x16) |   32 (32=16x2)   |    16    |      2       |   2 (=2x1)  |  tp2pp1  | 14 |
|  3   |    594.71    |     108.14    | 149.46 |  192(=24x8) |   32 (32=8x4)    |    8     |      4       |   4 (=4x1)  |  tp4pp1  | 24 |
|  4   |    592.60    |     111.22    | 199.08 |  192(=24x8) |   32 (32=8x4)    |    8     |      4       |   4 (=4x1)  |  tp4pp1  | 24 |
|  5   |    544.17    |     119.83    | 149.25 |  160(=20x8) |   32 (32=8x4)    |    8     |      4       |   4 (=4x1)  |  tp4pp1  | 20 |
+------+--------------+---------------+--------+-------------+------------------+----------+--------------+-------------+----------+----+
```

If you want to reproduce the result we esimated, you need to follow the suggestions here. Take the disagg top1 result as an example.  
We're expecting to achieve 913.82 tokens/s/gpu and 123.92 tokens/s/user with this config.  
We have 1 definition `replica`, it means the number of copies of your xPyD disagg system. Say, here, we have 4 replicas, each replica contains 8 GPUs.  
Each replica has a system of 4 prefill workers and 1 decode workers. Each prefill worker is using tp1pp1 which is 1 GPU per worker; while each decoder worker is using tp4pp1 which is 4 GPU per workers. These workers compose a 4P1D replica with 8 GPUs. As you want to deploy on 32 GPUs, then you will have 4 replicas.  
`bs` is required to be set in framework as it limits the largest batch_size of the worker which is crucial to control the TPOT of the deployment.  
`concurrency` = `concurrency * replicas` Use it to benchmark your deployment on total GPUs. If you only want to benchmark 1 replica, divide it by `replicas`

As this is still a little bit challenging to get the right configs for your deployment, we can further specify `--save_dir DIR` to output all the results here as well as **generate the configs for frameworks automatically**. Here's a stucture of the output folder,
```text
results/QWEN3_32B_isl4000_osl1000_ttft1000_tpot20_904495
├── agg
│   ├── best_config_topn.csv
│   ├── config.yaml
│   ├── pareto.csv
│   ├── top1
│   │   ├── agg
│   │   │   ├── agg_config.yaml
│   │   │   ├── k8s_deploy.yaml
│   │   │   └── node_0_run.sh 
│   │   └── generator_config.yaml
│   ...
├── disagg
│   ├── best_config_topn.csv
│   ├── config.yaml
│   ├── pareto.csv
│   ├── top1
│   │   ├── disagg
│   │   │   ├── decode_config.yaml
│   │   │   ├── k8s_deploy.yaml
│   │   │   ├── node_0_run.sh
│   │   │   └── prefill_config.yaml
│   │   └── generator_config.yaml
│   ...
└── pareto_frontier.png
```
By default, we output top3 configs we have found. You can get the configs and scripts to deploy under each experiment's folder. `agg_config.yaml` and `node_0_run.sh` are the files you need to deploy with Dynamo. If you want to deploy using k8s, you can leverage `k8s_deploy.yaml`. Refer to [deployment guide](dynamo_deployment_guide.md) for info about deployment.

`--save_dir DIR` allows you to specify more information such as generating the config for a different version of the backend, say estimating the performance using trtllm 1.0.0rc3 but generate config for 1.0.0rc6. This is allowed and feasible. By passing `--generated_config_version 1.0.0rc6` can give you the right result.

Use `--generator-config path/to/file.yaml` to provide ServiceConfig/K8sConfig/Workers sections, or add inline overrides via `--generator-set KEY=VALUE`. Examples:

- `--generator-set ServiceConfig.model_path=Qwen/Qwen3-32B-FP8`
- `--generator-set K8sConfig.k8s_namespace=dynamo \`

Run `aiconfigurator cli default --generator-help` to print information that is sourced directly from `src/aiconfigurator/generator/config/deployment_config.yaml` and `backend_config_mapping.yaml`. 

The `--generator-help` command supports three section options:
- `--generator-help` or `--generator-help all` (default): Shows both the full deployment schema and the backend parameter mappings
- `--generator-help deploy`: Shows the complete content of `generator/config/deployment_config.yaml` in YAML format, including all sections such as `ServiceConfig.*`, `K8sConfig.*`, `WorkerConfig.*`, etc.
- `--generator-help backend`: Shows only the backend parameter mappings table from `generator/config/backend_config_mapping.yaml`, which maps unified parameter keys (e.g., `kv_cache_free_gpu_memory_fraction`, `kv_cache_dtype`) to backend-specific parameter names for trtllm, vllm, and sglang

You can filter the backend-mapping output to a specific backend using `--generator-help --generator-help-backend BACKEND`, where BACKEND can be `trtllm`, `vllm`, or `sglang`. For example:
- `aiconfigurator cli default --generator-help backend --generator-help-backend sglang`: Shows only sglang-specific parameter mappings
- `aiconfigurator cli default --generator-help backend --generator-help-backend trtllm`: Shows only trtllm-specific parameter mappings

The command exits after printing the help information, so you do not need to provide the required `default` mode arguments (like `--model`, `--backend`, etc.) when using this flag.

#### Request latency constraint
`--request_latency <ms>` gives you a single end-to-end SLA on TTFT + TPOT × (OSL − 1). When the flag is set, `default` mode automatically enumerates TTFT/TPOT pairs that satisfy that budget (respecting any explicit `--ttft`, if provided) and only keeps configurations whose estimated request latency stays within the bound. Because the CLI derives TPOT from the request latency target, any `--tpot` argument is ignored in this mode.

- The detailed tables printed for both agg and disagg add a `request_latency` column, and the global Pareto plot flips to “request latency vs tokens/s/gpu” whenever every experiment is operating under this constraint.
- You can still set `--ttft` to reserve more headroom for prefill. Leaving it unset lets the enumerator try multiple TTFT splits automatically.

Example: search for 16x H200 configs that meet a 12s end-to-end budget while capping TTFT at 4s.
```bash
aiconfigurator cli default \
  --model QWEN3_32B \
  --total_gpus 16 \
  --system h200_sxm \
  --backend trtllm \
  --request_latency 12000 \
  --isl 4000 \
  --osl 500 \
  --ttft 4000
```
The summary will highlight the fastest configuration whose estimated request latency is ≤ 12,000 ms and will show the derived TTFT/TPOT pair that satisfied the constraint. The example output,
```
********************************************************************************
*                     Dynamo aiconfigurator Final Results                      *
********************************************************************************
  ----------------------------------------------------------------------------
  Input Configuration & SLA Target:
    Model: QWEN3_32B (is_moe: False)
    Total GPUs: 16
    Best Experiment Chosen: disagg at 932.91 tokens/s/gpu (disagg 1.09x better)
  ----------------------------------------------------------------------------
  Overall Best Configuration:
    - Best Throughput: 14,926.50 tokens/s
    - Per-GPU Throughput: 932.91 tokens/s/gpu
    - Per-User Throughput: 57.49 tokens/s/user
    - TTFT: 542.58ms
    - TPOT: 17.39ms
    - Request Latency: 9222.18ms
  ----------------------------------------------------------------------------
  Pareto Frontier:
          QWEN3_32B Pareto Frontier: tokens/s/gpu_cluster vs request_latency    
      ┌────────────────────────────────────────────────────────────────────────┐
1150.0┤ •• agg                                                                 │
      │ ff disagg                                                              │
      │ xx disagg best                                                         │
      │                                                                        │
 958.3┤                                                                        │
      │                                     ffffffffffffffx                    │
      │                                    f                            •      │
      │                                    f                         •••       │
 766.7┤                                    f                    •••••          │
      │                                   f                •••••               │
      │                                 ff            •••••                    │
      │                               ff         •••••                         │
 575.0┤                             ff     ••••••                              │
      │                           ff     •••                                   │
      │                         ff     ••                                      │
      │                              ••                                        │
 383.3┤                          ••••                                          │
      │                        •••                                             │
      │                   ••••••                                               │
      │                 •••                                                    │
 191.7┤                                                                        │
      │                                                                        │
      │                                                                        │
      │                                                                        │
   0.0┤                                                                        │
      └┬─────────────────┬─────────────────┬────────────────┬─────────────────┬┘
       0               3220              6440             9660            12880 
tokens/s/gpu_cluster                request_latency                             

  ----------------------------------------------------------------------------
  Deployment Details:
    (p) stands for prefill, (d) stands for decode, bs stands for batch size, a replica stands for the smallest scalable unit xPyD of the disagg system
    Some math: total gpus used = replicas * gpus/replica
               gpus/replica = (p)gpus/worker * (p)workers + (d)gpus/worker * (d)workers; for Agg, gpus/replica = gpus/worker
               gpus/worker = tp * pp * dp = etp * ep * pp for MoE models; tp * pp for dense models (underlined numbers are the actual values in math)

agg Top Configurations: (Sorted by tokens/s/gpu)
+------+--------------+---------------+--------+-----------------+--------------+-------------------+----------+--------------+-------------+----------+----+
| Rank | tokens/s/gpu | tokens/s/user |  TTFT  | request_latency | concurrency  | total_gpus (used) | replicas | gpus/replica | gpus/worker | parallel | bs |
+------+--------------+---------------+--------+-----------------+--------------+-------------------+----------+--------------+-------------+----------+----+
|  1   |    852.23    |     46.35     | 937.94 |     11704.26    | 320 (=40x8)  |    16 (16=8x2)    |    8     |      2       |  2 (=2x1x1) |  tp2pp1  | 40 |
|  2   |    748.51    |     49.46     | 711.67 |     10799.77    | 256 (=64x4)  |    16 (16=4x4)    |    4     |      4       |  4 (=4x1x1) |  tp4pp1  | 64 |
|  3   |    742.79    |     50.12     | 735.24 |     10691.50    | 256 (=16x16) |    16 (16=16x1)   |    16    |      1       |  1 (=1x1x1) |  tp1pp1  | 16 |
|  4   |    550.53    |     47.56     | 568.11 |     11060.92    | 192 (=96x2)  |    16 (16=2x8)    |    2     |      8       |  8 (=8x1x1) |  tp8pp1  | 96 |
+------+--------------+---------------+--------+-----------------+--------------+-------------------+----------+--------------+-------------+----------+----+

disagg Top Configurations: (Sorted by tokens/s/gpu)
+------+--------------+---------------+--------+-----------------+--------------+-------------------+----------+----------------+------------+----------------+-------------+-------+------------+----------------+-------------+-------+
| Rank | tokens/s/gpu | tokens/s/user |  TTFT  | request_latency | concurrency  | total_gpus (used) | replicas |  gpus/replica  | (p)workers | (p)gpus/worker | (p)parallel | (p)bs | (d)workers | (d)gpus/worker | (d)parallel | (d)bs |
+------+--------------+---------------+--------+-----------------+--------------+-------------------+----------+----------------+------------+----------------+-------------+-------+------------+----------------+-------------+-------+
|  1   |    932.91    |     57.49     | 542.58 |     9222.18     | 384 (=384x1) |    16 (16=1x16)   |    1     | 16 (=10x1+3x2) |     10     |    1 (=1x1)    |    tp1pp1   |   1   |     3      |    2 (=2x1)    |    tp2pp1   |  128  |
|  2   |    932.91    |     49.29     | 542.58 |     10666.29    | 384 (=192x2) |    16 (16=2x8)    |    2     |  8 (=5x1+3x1)  |     5      |    1 (=1x1)    |    tp1pp1   |   1   |     3      |    1 (=1x1)    |    tp1pp1   |   64  |
|  3   |    818.83    |     43.33     | 326.26 |     11842.68    | 328 (=328x1) |    16 (16=1x16)   |    1     | 16 (=6x2+1x4)  |     6      |    2 (=2x1)    |    tp2pp1   |   1   |     1      |    4 (=4x1)    |    tp4pp1   |  328  |
|  4   |    746.33    |     43.72     | 542.58 |     11955.71    | 496 (=496x1) |    16 (16=1x16)   |    1     | 16 (=8x1+1x8)  |     8      |    1 (=1x1)    |    tp1pp1   |   1   |     1      |    8 (=8x1)    |    tp8pp1   |  496  |
+------+--------------+---------------+--------+-----------------+--------------+-------------------+----------+----------------+------------+----------------+-------------+-------+------------+----------------+-------------+-------+
********************************************************************************
2025-12-01 23:36:41,892 - aiconfigurator.cli.main - INFO - All experiments completed in 1.92 seconds
```

#### Database Mode

The `--database_mode` argument controls how performance is estimated:

| Mode | Description |
|------|-------------|
| `SILICON` | **(Default)** Uses actual collected silicon data. Most accurate when data is available for your configuration. |
| `HYBRID` | Uses silicon data when available, falls back to SOL+empirical factor when data is missing. Best for exploring configurations that may not have complete silicon data. |
| `EMPIRICAL` | Uses Speed-of-Light (SOL) + empirical correction factors for all estimations. Useful for rough estimates without relying on collected data. |
| `SOL` | Provides theoretical Speed-of-Light time only. Useful for understanding theoretical limits. |

Example using hybrid mode:
```bash
aiconfigurator cli default --model QWEN3_32B --total_gpus 32 --system h200_sxm --database_mode HYBRID
```

For exp mode, you can specify `database_mode` in your YAML file:
```yaml
exp_hybrid:
  serving_mode: "agg"
  model_name: "QWEN3_32B"
  system_name: "h200_sxm"
  total_gpus: 8
  database_mode: "HYBRID"
```

Hybrid mode is a quick solution to support new models without modeling the operation and collecting the data. 
Will be leveraged more and more in future.

See `src/aiconfigurator/cli/exps/database_mode_comparison.yaml` for an example comparing different database modes.

### Exp mode
If you want to customize your experiment apart from simple command which only compares disagg and agg of a same model, you can use `exp` mode. The command is,
```bash
aiconfigurator cli exp --yaml_path example.yaml
```
An example yaml file looks like this, the template is [here](../src/aiconfigurator/cli/example.yaml)  
Let's split the yaml file into several sections.  
1. exps
```yaml
exps:
  - exp_agg_full
  - exp_agg_simplified
  - exp_disagg_full
  - exp_disagg_simplified
```
`exps` section defines the experiments you want to run. If not specified, all exps will be run

2. A certain exp definition
```yaml
exp_disagg_full:
  mode: "patch" # patch or replace the config section, required
  serving_mode: "disagg" # required
  model_name: "DEEPSEEK_V3" # required
  total_gpus: 32 # required
  system_name: "h200_sxm" # required
  decode_system_name: "h200_sxm" # optional, if not provided, it will use the same system name as the prefill system.
  backend_name: "trtllm" # optional, can be "trtllm" (default), "vllm", or "sglang"
  backend_version: "0.20.0" # optional, default to the latest version in the database
  isl: 4000 # input sequence length, optional, default to 4000
  osl: 1000 # output sequence length, optional, default to 1000
  prefix: 0 # prefix cache len, default to 0
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
      prefill_latency_correction_scale: 1.1
      decode_latency_correction_scale: 1.08
      prefill_max_batch_size: 1
      decode_max_batch_size: 512
```
This section is very long, let's go through the basic setting quickly  
    - `mode`: patch means the `config` session below will do patch to default config while replace will overwrite everything. Typically, no need to modify  
    - `serving_mode`: defines agg or disagg of this exp  
    - `model_name`, `total_gpus`: defines the model and GPU resources
    - `backend_name`: specifies the inference backend - `trtllm` (default), `vllm`, or `sglang`
    - `backend_version`, `isl`, `osl`, `ttft`, `tpot`: defines the same things as in `default` mode  
    - `enable_wideep`: will trigger wide-ep for fined-grained moe model  
    - `profiles`: some inherit patch, we current have 'fp8_default', 'float16_default', 'nvfp4_default' to force the precision of a worker.  
    - `config`: the most important part. It defines `nextn` for MTP; It also defines the agg_/prefill_/decode_worker's quantization, and parallelism search space; It also defines more about how we search for the disagg replica and do correction for better performance alignment. We'll go through it in [Advanced Tuning](advanced_tuning.md). Typically, the only thing here for you to modify, perhaps, is the quantization of the worker.

If you don't want to patch the `config` details, you can just delete them. Here's a simplified one,
```yaml
exp_disagg_simplified:
  mode: "patch"
  serving_mode: "disagg"
  model_name: "DEEPSEEK_V3"
  total_gpus: 512
  system_name: "gb200_sxm"
  enable_wideep: true # enable wide ep for prefill/decode, default to false, optional
  config: # patch below default values
    nextn: 2 # mtp 1
    nextn_accept_rates: [0.85,0.3,0,0,0] # each position maps to the accept rate of the ith draft token, nextn 1 will only use the first draft token accept rate.
    replica_config:
      max_gpu_per_replica: 512 # max gpus per replica, wide ep needs larger max_gpu_per_replica value
```
This only defines the system you want to use. Overwrite what you want.

Let's go through some pre-defined experiments for reference.
1. homegeneous vs. heterogenous  
The example [yaml](../src/aiconfigurator/cli/exps/hetero_disagg.yaml)
```yaml
exps:
  - exp_h200_h200
  - exp_b200_h200

exp_h200_h200:
  mode: "patch"
  serving_mode: "disagg" # required
  model_name: "QWEN3_32B" # required
  total_gpus: 16 # required
  system_name: "h200_sxm" # required, for prefill
  decode_system_name: "h200_sxm" # optional, if not provided, it will use the same system name as the prefill system.
  backend_name: "trtllm" # can also be "vllm" or "sglang"
  profiles: []
  isl: 4000 # input sequence length
  osl: 500 # output sequence length
  ttft: 300.0  # Target TTFT in ms
  tpot: 50.0   # Target TPOT in ms

exp_b200_h200:
  mode: "patch"
  serving_mode: "disagg" # required
  model_name: "QWEN3_32B" # required
  total_gpus: 16 # required
  system_name: "b200_sxm" # required, for prefill
  decode_system_name: "h200_sxm" # optional, if not provided, it will use the same system name as the prefill system.
  backend_name: "trtllm" # can also be "vllm" or "sglang"
  profiles: []
  isl: 4000 # input sequence length
  osl: 500 # output sequence length
  ttft: 300.0  # Target TTFT in ms
  tpot: 50.0   # Target TPOT in ms
```
We defined two experiments. `exp_h200_h200` uses h200 for both prefill and decode. `exp_b200_h200` uses b200 for prefill and h200 for decode.

**Note**: You can also compare different backends by setting different `backend_name` values (trtllm, vllm, sglang) in your experiments.

2. use a specific quantization  
The example [yaml](../src/aiconfigurator/cli/exps/qwen3_32b_disagg_pertensor.yaml)
```yaml
exps:
  - exp_agg
  - exp_disagg

exp_agg:
  mode: "patch"
  serving_mode: "agg" # required
  model_name: "QWEN3_32B" # required
  total_gpus: 16 # required
  system_name: "h200_sxm" # required, for prefill
  backend_name: "trtllm" # can also be "vllm" or "sglang"
  profiles: ["fp8_default"]
  isl: 4000 # input sequence length
  osl: 500 # output sequence length
  ttft: 600.0  # Target TTFT in ms
  tpot: 16   # Target TPOT in ms

exp_disagg:
  mode: "patch"
  serving_mode: "disagg" # required
  model_name: "QWEN3_32B" # required
  total_gpus: 16 # required
  system_name: "h200_sxm" # required, for prefill
  decode_system_name: "h200_sxm" # optional, if not provided, it will use the same system name as the prefill system.
  backend_name: "trtllm" # can also be "vllm" or "sglang"
  profiles: ["fp8_default"]
  isl: 4000 # input sequence length
  osl: 500 # output sequence length
  ttft: 600.0  # Target TTFT in ms
  tpot: 16   # Target TPOT in ms
```
In this example, we use a pre-defined profile to overwrite quantization of QWEN3_32B. Default is blockwise FP8 for GEMM and here we use per-tensor FP8.

You can refer to [src/aiconfigurator/cli/exps](../src/aiconfigurator/cli/exps) to find more reference yaml files.

Use this `exp` mode will allow more flexible experiments and use `default` mode will give you the most convenience. Both modes support generating configs for frameworks automatically by `--save_dir DIR`