# Accuracy Regression Testing

A workflow for comparing AIC TTFT/TPOT predictions between two revisions.

## 1. Generate predictions

Run the prediction wrapper once per AIC revision. Use a separate environment for
each revision so both the Python packages and the compiled core extension come
from that revision. The prefix controls the output column names.

```bash
# From an environment installed from the incoming/current branch
python tools/accuracy_regression_testing/predict_silicon_sample.py \
  --aic-output-prefix new \
  > tools/accuracy_regression_testing/results/silicon_result_new.csv

# From a separate worktree and environment installed from the old revision.
# Use absolute paths so the old script writes into the incoming worktree.
/path/to/old/aiconfigurator/.venv/bin/python \
  /path/to/old/aiconfigurator/tools/accuracy_regression_testing/predict_silicon_sample.py \
  /path/to/incoming/aiconfigurator/tools/accuracy_regression_testing/results/silicon_result_new.csv \
  --aic-output-prefix old \
  > /path/to/incoming/aiconfigurator/tools/accuracy_regression_testing/results/silicon_result.csv
```

The CI workflow installs `packages/aiconfigurator-core` and
`packages/aiconfigurator` into each revision's environment before running these
commands. Older monolithic revisions are installed from their repository root.

The final `silicon_result.csv` should contain both `old_predicted_*` and
`new_predicted_*` columns.

## 2. Compare predictions

```bash
python tools/accuracy_regression_testing/compare_silicon_predictions.py \
  tools/accuracy_regression_testing/results/silicon_result.csv \
  --output tools/accuracy_regression_testing/results/comparison_summary.csv \
  --plot-output tools/accuracy_regression_testing/results/comparison_plot.png
```

This writes a CSV summary and a plot. Positive MAPE improvement means the new
revision is closer to silicon. `num_samples_added` is the net prediction
coverage change: new successful predictions minus old successful predictions.

## 3. Gate a PR

```bash
python -m pytest tools/accuracy_regression_testing/test_regression_thresholds.py
```

Optional path overrides:

```bash
AIC_COMPARISON_SUMMARY=/path/to/comparison_summary.csv \
AIC_SILICON_RESULT=/path/to/silicon_result.csv \
python -m pytest tools/accuracy_regression_testing/test_regression_thresholds.py
```

The pytest checks:
- `all` partition MAPE regression is below 5%.
- each other partition MAPE regression is below 10%.
- no row regresses from old prediction success to new prediction failure.
