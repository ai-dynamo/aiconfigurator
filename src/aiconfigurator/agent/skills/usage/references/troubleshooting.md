# Troubleshooting

## No Results

Report the model, system, backend, version, database mode, ISL, OSL, TTFT, TPOT,
and total GPU count. Then check:

- SLA too tight: relax `--ttft`, `--tpot`, or `--request-latency`.
- Not enough GPUs: increase `--total-gpus`.
- Search space too narrow: inspect enumerated parallel configs in the log.
- Missing silicon rows: report a silicon coverage gap with the exact
  model/system/backend/version/workload. Do not silently switch the final
  experiment to another database mode.
- Unsupported backend/system: run `aiconfigurator cli support`.

## Very Few Results

If only a few rows survive:

- Check whether the log enumerated fewer parallel configs than expected.
- Compare top-row `ttft` and `tpot` against the targets. The tighter margin
  usually indicates which SLA is limiting the search.
- If TTFT is tight, adjust prefill-side candidates, prefill batch caps, or the
  TTFT target.
- If TPOT is tight, adjust decode-side candidates, decode batch caps, or the
  TPOT target.
- Keep workload and database mode fixed while tuning the search so the
  comparison stays interpretable.

## Small Batch Sizes and Low Throughput

If top rows have very small batch sizes and poor throughput:

- Check whether the SLA forced AIC into tiny batches.
- For disagg, inspect prefill and decode batch sizes separately. A decode batch
  that is too small can under-fill the system and limit throughput.
- Check whether `prefill_max_batch_size` or `decode_max_batch_size` capped the
  search too aggressively.
- Tune one side at a time based on the limiting metric: TTFT for prefill, TPOT
  for decode.
- If the batch size is intentionally small for latency reasons, report the
  throughput tradeoff explicitly.

## Missing Database

If AIC reports no perf database for a system/backend/version, remove
`--backend-version` to use latest, choose an available version, or pass
`--systems-paths default,/path/to/extra/systems`.

## New Model

For frontier or newly added models, do not assume `SILICON` coverage. Use
`support` first. If silicon data is incomplete, report that the final experiment
is blocked on coverage instead of presenting `HYBRID` or `EMPIRICAL` as a
deployment-quality result.

## Local Model Config

`--model` may be a Hugging Face ID or a local directory containing `config.json`.
If using a local path, verify the config architecture is supported by AIC.
