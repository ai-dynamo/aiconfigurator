# AIC Prefill FPM Validation Artifacts

This directory contains raw artifacts for validating AIC native forward-pass
predictions against GPT-OSS-120B / B200 SXM / TRT-LLM prefill FPM telemetry.

## Files

- `trtllm_b200_gpt_oss_120b_mixed_fp4_fpms.csv`: input FPM CSV, 2,000 prefill-only rows.
- `reproduce_aic_prefill_fpm_validation.py`: reproduction and plotting script.
- `trtllm_b200_gpt_oss_120b_aic_native_predictions.csv`: row-level AIC native predictions and signed percent error.
- `trtllm_b200_gpt_oss_120b_aic_native_2d_cell_diff_summary.csv`: 8x8 compute/read bucket summary.
- `trtllm_b200_gpt_oss_120b_heatmap_count.png`: FPM count heatmap.
- `trtllm_b200_gpt_oss_120b_heatmap_diff_pct_mean.png`: mean diff% heatmap.
- `trtllm_b200_gpt_oss_120b_heatmap_diff_pct_var.png`: variance of diff% heatmap.
- `trtllm_b200_gpt_oss_120b_heatmaps_combined.png`: combined heatmap view.

## Reproduce

From the repository root:

```bash
python issue-artifacts/aic-prefill-fpm-validation/reproduce_aic_prefill_fpm_validation.py \
  --input issue-artifacts/aic-prefill-fpm-validation/trtllm_b200_gpt_oss_120b_mixed_fp4_fpms.csv \
  --outdir issue-artifacts/aic-prefill-fpm-validation \
  --aiconfigurator-src src
```

Expected summary:

```text
overall: n=2000 observed_ms_mean=48.888 aic_ms_mean=28.831 diff_pct_mean=-46.898 diff_pct_var=590.182 diff_pct_p50=-51.198 diff_pct_p90=-11.862
```

The validation uses AIC native mode, not fallback regression:

```python
model = RustForwardPassPerfModel.from_native(config, {})
aic_ms = model.estimate_forward_pass_time_ms([fpm])
```

The local `b200_sxm/trtllm/1.3.0rc15` database directory is marker-only in the
checkout used for this analysis, so the script uses the latest complete local
database version, `1.3.0rc10`.
