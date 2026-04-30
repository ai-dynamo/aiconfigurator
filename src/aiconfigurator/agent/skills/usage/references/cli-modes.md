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

Run default mode first for a rough range. It builds agg and disagg task configs,
runs the broad search, prints the best configs, and can save artifacts.

```bash
aiconfigurator cli default \
  --model Qwen/Qwen3-32B \
  --system h200_sxm \
  --backend trtllm \
  --total-gpus 8 \
  --database-mode SILICON \
  --isl 4000 \
  --osl 1000 \
  --ttft 2000 \
  --tpot 30 \
  --save-dir results
```

Use `--backend auto` to compare supported backends for the same system. Use
`--decode-system` when disaggregated decode hardware differs from prefill.

## Experiment YAML

After the rough default run, use `exp` mode for the precise agent-controlled
experiment definition. This is especially useful for narrowing disaggregated
search space.

```bash
aiconfigurator agent usage --ref experiment-template > template.yaml
# Edit template.yaml based on the default run's promising disagg shape.
aiconfigurator cli exp --yaml-path template.yaml --save-dir results
```

Keep `database_mode: SILICON` for final experiments. Modify search lists in the
YAML instead of adding quantization overrides by default. The template is the
packaged CLI example YAML, so remove demonstration-only fields that are not part
of the intended experiment.

## Estimate

Use estimate mode for a known single configuration. Provide TP/PP/batch values
instead of asking AIC to search.

## Generate

Use generate mode only for a naive working aggregate config. It does not run SLA
optimization or a full parameter sweep.
