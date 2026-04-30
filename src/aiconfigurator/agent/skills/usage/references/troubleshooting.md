# Troubleshooting

## No Results

Report the model, system, backend, version, database mode, ISL, OSL, TTFT, TPOT,
and total GPU count. Then check:

- SLA too tight: relax `--ttft`, `--tpot`, or `--request-latency`.
- Not enough GPUs: increase `--total-gpus`.
- Missing silicon rows: report a silicon coverage gap with the exact
  model/system/backend/version/workload. Do not silently switch the final
  experiment to another database mode.
- Unsupported backend/system: run `aiconfigurator cli support`.

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
