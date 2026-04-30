# Single Experiment YAML

Use `aiconfigurator cli exp --yaml-path ...` even for one experiment when the
user needs exact control over worker search space, quantization, WideEP, MTP,
replica shape, or latency correction. A YAML file is easier to review and reuse
than a long CLI command.

## Authoring Workflow

1. Start from `src/aiconfigurator/cli/example.yaml` or the closest file under
   `src/aiconfigurator/cli/exps/`.
2. Keep `exps` to one entry unless the user explicitly wants comparison.
3. Prefer `mode: patch` so defaults still supply backend versions, worker
   fields, and runtime metadata.
4. Put workload and SLA at the experiment top level: `isl`, `osl`, `prefix`,
   `ttft`, `tpot`, and optionally `request_latency`.
5. Put advanced model/search changes under `config`.
6. Run with `--save-dir` and inspect the saved `exp_config.yaml`,
   `best_config_topn.csv`, `pareto.csv`, and `top*/generator_config.yaml`.

## Minimal Aggregate Template

```yaml
exps:
  - agg_one

agg_one:
  mode: patch
  serving_mode: agg
  model_path: Qwen/Qwen3-32B
  total_gpus: 8
  system_name: h200_sxm
  backend_name: trtllm
  database_mode: HYBRID
  isl: 4000
  osl: 1000
  ttft: 2000.0
  tpot: 30.0
  config:
    worker_config:
      num_gpu_per_worker: [1, 2, 4, 8]
      tp_list: [1, 2, 4, 8]
      pp_list: [1]
```

## Minimal Disaggregated Template

```yaml
exps:
  - disagg_one

disagg_one:
  mode: patch
  serving_mode: disagg
  model_path: Qwen/Qwen3-32B
  total_gpus: 16
  system_name: h200_sxm
  decode_system_name: h200_sxm
  backend_name: trtllm
  database_mode: HYBRID
  isl: 4000
  osl: 1000
  ttft: 2000.0
  tpot: 30.0
  config:
    prefill_worker_config:
      num_gpu_per_worker: [1, 2, 4, 8]
      tp_list: [1, 2, 4, 8]
      pp_list: [1]
      dp_list: [1]
    decode_worker_config:
      num_gpu_per_worker: [1, 2, 4, 8]
      tp_list: [1, 2, 4, 8]
      pp_list: [1]
      dp_list: [1]
    replica_config:
      max_gpu_per_replica: 128
      max_prefill_worker: 32
      max_decode_worker: 32
```

## MTP and WideEP Fields

Add MTP only when the model is known to support it:

```yaml
  config:
    nextn: 2
    nextn_accept_rates: [0.85, 0.3, 0.0, 0.0, 0.0]
```

Enable WideEP only for MoE experiments where the backend and database have
WideEP coverage:

```yaml
  enable_wideep: true
  config:
    worker_config:
      moe_tp_list: [1]
      moe_ep_list: [8, 16, 32]
```

For disaggregated experiments, the same idea applies independently to
`prefill_worker_config` and `decode_worker_config`.

## Checklist

- Confirm `model_path`, `system_name`, `decode_system_name`, `backend_name`, and
  `backend_version` are real supported values.
- Use `database_mode: HYBRID` for frontier or new model work unless the user
  explicitly wants strict silicon-only data.
- Keep `replace` mode rare; it must provide every required config field.
- Reduce search space first when a YAML run is slow.
- Check the saved `exp_config.yaml`; it is the normalized config AIC actually
  used.
