# Troubleshooting

## No Results

Report the model, system, backend, version, database mode, ISL, OSL, TTFT, TPOT,
and total GPU count. Then check:

- SLA too tight: relax `--ttft`, `--tpot`, or `--request-latency`.
- Not enough GPUs: increase `--total-gpus`.
- Missing silicon rows: retry with `--database-mode HYBRID`.
- Unsupported backend/system: run `aiconfigurator cli support`.

## Missing Database

If AIC reports no perf database for a system/backend/version, remove
`--backend-version` to use latest, choose an available version, or pass
`--systems-paths default,/path/to/extra/systems`.

## New Model

For frontier or newly added models, do not assume `SILICON` coverage. Use
`support` first, then run `default` with `HYBRID` if silicon data is incomplete.

## Local Model Config

`--model` may be a Hugging Face ID or a local directory containing `config.json`.
If using a local path, verify the config architecture is supported by AIC.
