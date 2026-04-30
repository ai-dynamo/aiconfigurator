# Deployment and Benchmark Artifacts

AIC can carry a planning run into deployment and benchmark artifacts. Use this
path when the user wants to run a chosen config, not just estimate it.

## Generate Artifacts From a Search

```bash
aiconfigurator cli default \
  --model Qwen/Qwen3-32B \
  --system h200_sxm \
  --backend trtllm \
  --total-gpus 8 \
  --isl 4000 \
  --osl 1000 \
  --ttft 2000 \
  --tpot 30 \
  --deployment-target dynamo-j2 \
  --save-dir results
```

Use `--deployment-target llm-d` for llm-d Helm values. Use
`--generator-dynamo-version` or `--generated-config-version` when the generated
config must match a specific runtime.

## Artifact Map

- `top*/generator_config.yaml`: normalized input to the generator.
- `k8s_deploy.yaml`: Dynamo Kubernetes deployment manifest.
- `llm-d-values.yaml`: llm-d Helm values for the model service.
- `run_*.sh`: manual run scripts for Dynamo-style launches.
- `agg_config.yaml`, `prefill_config.yaml`, `decode_config.yaml`: TRT-LLM engine
  config files when generated for that backend.
- `bench_run.sh`: local benchmark helper.
- `k8s_bench.yaml`: Kubernetes benchmark job.

## Runtime Version Checks

- Planning `--backend-version` selects the performance database version.
- `--generated-config-version` selects the generated backend config format.
- `--generator-dynamo-version` maps a Dynamo release to backend config versions
  and image defaults.
- These versions can differ. Call out that difference explicitly.

## Validation

When runtime images are available, validate generated artifacts before telling a
user to deploy:

```bash
python tools/generator_validator/validator.py \
  --backend trtllm \
  --path results/<run>/disagg/top1
```

For vLLM and SGLang Dynamo deployments, point the validator at `k8s_deploy.yaml`
or the result directory. llm-d validation is not currently covered by the
validator.

## Deployment Guidance

- For TRT-LLM run scripts, copy generated engine configs to
  `/workspace/engine_configs/` before launching.
- In multi-node manual runs, node 0 runs frontend plus workers; other nodes run
  workers. Set `ServiceConfig.head_node_ip` through `--generator-set`.
- Benchmark concurrency arrays include base values plus estimate-derived
  concurrency and nearby values when available.
- Treat generated configs as starting points; verify image tags, PVCs,
  namespace, model path, served model name, and head node IP.
