# Usage Examples

## Support Matrix

```bash
aiconfigurator cli support \
  --model Qwen/Qwen3-32B \
  --system all \
  --backend all
```

## Default Search With Hybrid Coverage

```bash
aiconfigurator cli default \
  --model Qwen/Qwen3-32B \
  --system h200_sxm \
  --backend auto \
  --total-gpus 8 \
  --database-mode HYBRID \
  --isl 4000 \
  --osl 1000 \
  --ttft 2000 \
  --tpot 30 \
  --save-dir results
```

## Disaggregated Hardware Split

```bash
aiconfigurator cli default \
  --model Qwen/Qwen3-32B \
  --system b200_sxm \
  --decode-system h200_sxm \
  --backend trtllm \
  --total-gpus 16 \
  --save-dir results
```

## Single-Point Estimate

```bash
aiconfigurator cli estimate \
  --model Qwen/Qwen3-32B \
  --estimate-mode agg \
  --system h200_sxm \
  --backend trtllm \
  --isl 4000 \
  --osl 1000 \
  --batch-size 128 \
  --ctx-tokens 4000 \
  --tp-size 4
```
