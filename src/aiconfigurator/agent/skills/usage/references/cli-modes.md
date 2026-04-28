# CLI Modes

Use the top-level executable as `aiconfigurator`.

## Support

Use support mode before a costly sweep or when the user asks whether a model,
system, or backend combination is covered.

```bash
aiconfigurator cli support \
  --model Qwen/Qwen3-32B \
  --system h200_sxm \
  --backend trtllm
```

Use `--system all --backend all` for a matrix view.

## Default

Use default mode for normal planning. It builds agg and disagg task configs,
runs the search, prints the best configs, and can save artifacts.

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
  --save-dir results
```

Use `--backend auto` to compare supported backends for the same system. Use
`--decode-system` when disaggregated decode hardware differs from prefill.

## Estimate

Use estimate mode for a known single configuration. Provide TP/PP/batch values
instead of asking AIC to search.

## Generate

Use generate mode only for a naive working aggregate config. It does not run SLA
optimization or a full parameter sweep.

## Experiment YAML

Use `exp` mode when comparing several explicit experiment definitions.

```bash
aiconfigurator cli exp --yaml-path experiments.yaml --save-dir results
```
