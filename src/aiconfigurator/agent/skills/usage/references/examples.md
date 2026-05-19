# Usage Examples

## Support Matrix

```bash
aiconfigurator cli support \
  --model Qwen/Qwen3-32B \
  --system all \
  --backend all
```

## Rough Default Run

```bash
aiconfigurator cli default \
  --model Qwen/Qwen3-32B \
  --system h200_sxm \
  --backend trtllm \
  --total-gpus 16 \
  --database-mode SILICON \
  --isl 4000 \
  --osl 1000 \
  --ttft 2000 \
  --tpot 30 \
  --save-dir results
```

## Precise Disagg Experiment From Template

```bash
aiconfigurator agent usage --ref experiment-template > template.yaml
# Keep the disagg entry and narrow prefill/decode search lists based on the
# rough default run.
aiconfigurator cli exp \
  --yaml-path template.yaml \
  --save-dir results
```

## Single-Point Estimate

```bash
aiconfigurator cli estimate \
  --model Qwen/Qwen3-32B \
  --estimate-mode agg \
  --system h200_sxm \
  --backend trtllm \
  --database-mode SILICON \
  --isl 4000 \
  --osl 1000 \
  --batch-size 128 \
  --ctx-tokens 4000 \
  --tp-size 4
```
