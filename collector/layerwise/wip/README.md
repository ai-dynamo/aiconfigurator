# WIP layerwise MoE artifacts

Work-in-progress data for the MoE layerwise modeling effort. **Not** part of the
canonical perf database under `src/aiconfigurator/systems/` yet — kept here so the
next person can reproduce the FPM-vs-AIC charts.

## moe_perf.txt
Real measured MoE-op (fused-experts) kernel timings on **b300_sxm / vLLM 0.20.1**,
used as the MoE overlay when running `tools/plot_fpm_vs_aic.py` for MoE models.
Covers the Qwen3.6-35B-A3B and DeepSeek-V4-Flash decode/mixed MoE shapes:
`hidden in {2048,4096}`, `topk in {6,8}`, `num_experts=256`,
`moe_dtype in {bfloat16, w4a8_mxfp4_mxfp8}`.

Usage:
```
.venv/bin/python tools/plot_fpm_vs_aic.py \
  --layerwise <layerwise.csv> \
  --model "Qwen/Qwen3.6-35B-A3B" \
  --moe-perf-file collector/layerwise/wip/moe_perf.txt \
  --out-dir fpm_vs_aic_charts_qwen36
```

See `collector/layerwise/README.md` ("FPM-vs-AIC MoE modeling: status & handoff")
for the full status and findings.
