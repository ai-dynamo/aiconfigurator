# Accuracy Regression Testing

A workflow for comparing AIC TTFT/TPOT predictions between two revisions.

## 1. Generate predictions

Run the prediction wrapper once per AIC revision. The prefix controls the output
column names.

```bash
# From the incoming/current branch
PYTHONPATH=src python tools/accuracy_regression_testing/predict_silicon_sample.py \
  --aic-output-prefix new \
  > tools/accuracy_regression_testing/results/silicon_result_new.csv

# Still run from this incoming/current checkout. Only PYTHONPATH points to the old src.
# The script and CSV paths below are relative to the current worktree, not the old checkout.
PYTHONPATH=/path/to/old/aiconfigurator/src python tools/accuracy_regression_testing/predict_silicon_sample.py \
  tools/accuracy_regression_testing/results/silicon_result_new.csv \
  --aic-output-prefix old \
  > tools/accuracy_regression_testing/results/silicon_result.csv
```

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

## Empirical fidelity against silicon

`validate_empirical_fidelity.py` compares explicit `SILICON` and `EMPIRICAL`
`run_static` results. It does not run through `HYBRID`. Each observation gets a
fresh model/session so a failed `FallbackOp` lookup at one random point cannot
change later points. Transfer policy is a global run option, never a per-case
matrix field. The default `--transfer-policy off` is the primary methodology:
it isolates own-data empirical fidelity from cross-shape, cross-quant,
cross-profile, and cross-op transfer.

The default matrix covers dense, MoE, DSA, DSV4, and VL models across BF16,
FP8-block, FP8-static, NVFP4, INT4-WO, and MXFP4/MXFP8 profiles:

```bash
python tools/accuracy_regression_testing/validate_empirical_fidelity.py \
  --output-dir /tmp/empirical-fidelity \
  --worst-n 3
```

Use `--transfer-policy aggressive` for a separate production-coverage run that
allows every transfer tier. Do not pool that result into the strict primary
fidelity score. Every observation and pair records provenance tags and the
worst tier used. If a transfer tag nevertheless appears under strict `off`, the
pair is retained as `transfer_excluded`, counted in summary diagnostics, and
excluded from APE/WAPE aggregates. Own-data `empirical` provenance and
analytic operation fallbacks remain valid under strict mode and stay visible in
the per-op source/attribution output.

SILICON follows the production shared-layer policy: it loads active-version
measurements plus manifest-declared sibling-version/framework rows. Formula-only
EMPIRICAL loads only the active version. Consequently this is a direct comparison
of the two real modes, not an active-row-only matched-source experiment; shared-row
effects should be interpreted from the per-model and per-op breakdowns.

Keep cross-stack diagnostics separate from that primary, fixed-stack fidelity
population. The supplemental matrix compares representative SGLang, TRT-LLM,
and vLLM profiles on B200/H100/B300 without pooling a stack-specific SILICON
defect into the main empirical-model score:

```bash
python tools/accuracy_regression_testing/validate_empirical_fidelity.py \
  tools/accuracy_regression_testing/empirical_fidelity_stack_matrix.json \
  --output-dir /tmp/empirical-fidelity-stacks \
  --worst-n 3
```

Pass a custom JSON matrix as the positional argument. Cases may provide
explicit `points` or deterministic `point_generation`. Tag explicit points
with `sample_kind` such as `grid_anchor_control`, `offgrid`, `boundary`,
`extrapolation`, or `image`; summaries keep those populations separate. A
runtime-level anchor label is only a control population: each op transforms
`batch/isl/osl` into its own coordinates, so only a raw-grid audit can certify
`interior_offgrid`. Image points accept
`image_height`, `image_width`, `num_images_per_request`, and may include the
separate `encoder` phase. Generated workloads may set `max_batch_tokens` and
`max_sequence` so independent log-space sampling stays inside a useful runtime
envelope.

Outputs are written as both CSV and JSON:

- `observations`: every mode execution, including exceptions; failures are not
  silently skipped. OOM rows retain their latency and memory diagnostics but
  are excluded from fidelity aggregates.
- `pairs`: paired full-model context-latency/TPOT/encoder comparisons.
- `summary`: coverage, mean/median/p90/max APE, WAPE, and signed bias by case,
  phase, and sample kind.
- `op_summary`: complete per-op aggregates over every comparable pair, grouped
  by phase/op and case/phase/op. It reports op APE, WAPE, signed bias, mean
  silicon latency share, and explicit one-sided missing-op counts. A missing
  empirical op contributes zero latency to its additive delta (and therefore
  100% op APE when the silicon op is positive) instead of disappearing from
  the report; empirical-only ops remain visible but have no silicon-based APE.
- `attribution`: per-op deltas for the worst N points, including source,
  latency share, error-mass share, signed contribution, and cancellation.

Unexpected exceptions return a non-zero exit code. Expected silicon coverage
misses, `EmpiricalNotImplementedError`, and OOM remain report data. Optional
`--min-eligible-coverage` and `--max-mean-ape` thresholds make a selected matrix
usable as a CI gate; `eligible coverage` means comparable pairs divided by
successful, non-OOM silicon references. Pair this conditional percentage with
`--min-eligible-count` and/or `--min-silicon-coverage` so a loss of reference
coverage cannot make a tiny surviving population look healthy.

The `encoder` label selects the encoder scalar and encoder op breakdown, but it
still comes from a complete `static_ctx` execution. A failure in the coupled
context path can therefore prevent that encoder observation; do not interpret
it as an independently executable SDK phase.

Silicon success is necessary but not sufficient evidence that silicon is a
trustworthy reference. Keep known bad or out-of-domain silicon regions in a
separate sample kind (or matrix) and do not pool them into primary off-grid
fidelity conclusions.
