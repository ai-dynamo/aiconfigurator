---
name: aiconfigurator-usage
description: Use when an agent needs to run the installed AIConfigurator CLI to check support, estimate LLM serving performance, compare aggregate and disaggregated deployments, or generate deployment configs.
---

# AIConfigurator Usage

AIC is an offline planning tool for LLM serving deployments. It estimates
performance, memory, and practical serving configs from a model, hardware
system, backend, workload shape, and SLA.

Do not treat AIC output as a live benchmark. When silicon data is missing or a
model/backend is new, call out the database mode used and prefer `HYBRID` over
pretending the result is fully measured.

## Default Workflow

1. Identify the model, system, backend, total GPUs, ISL, OSL, and SLA.
2. Optionally run `aiconfigurator cli support` to inspect support.
3. Run `aiconfigurator cli default` for the normal agg vs disagg comparison.
4. Use `--save-dir` by default when results need analysis, reproducibility, or
   generated deployment artifacts.
5. Use YAML (`aiconfigurator cli exp`) for a single complex experiment when the
   configuration includes worker search-space, quantization, WideEP, MTP, or
   replica tuning.
6. Use `aiconfigurator cli estimate` only for a known single configuration.
7. Use `aiconfigurator cli generate` only when the user wants a naive config
   without performance sweeping.

## Mode Selection

- `support`: pre-flight support matrix check.
- `default`: recommended path for sizing and comparing agg/disagg configs.
- `estimate`: single-point latency/power estimate for a specific config.
- `generate`: naive config generation; no SLA optimization.
- `exp`: one or more explicit experiments from YAML.

## Guardrails

- Do not invent system names, backends, versions, or quant modes.
- If support or silicon data is missing for a new model, try `--database-mode HYBRID`.
- Keep `SILICON` results separate from `HYBRID`, `EMPIRICAL`, and `SOL`.
- Do not enable MTP, WideEP, EPLB, or DeepEP just because the flag exists. Check
  model family, backend, and database support, then verify the generated/search
  config reflects the intended path.
- If a command fails due to support or database coverage, report the exact model,
  system, backend, version, and database mode.

## Load References Only When Needed

- CLI modes and required arguments: `aiconfigurator agent usage --ref cli-modes`
- Single experiment YAML authoring: `aiconfigurator agent usage --ref single-experiment-yaml`
- Result analysis: `aiconfigurator agent usage --ref result-interpretation`
- SDK/per-step latency breakdown: `aiconfigurator agent usage --ref sdk-step-breakdown`
- Deployment and benchmark artifacts: `aiconfigurator agent usage --ref deployment-bench`
- Feature pitfalls such as MTP, WideEP, DeepEP, EPLB, and quantization:
  `aiconfigurator agent usage --ref feature-pitfalls`
- Command examples: `aiconfigurator agent usage --ref examples`
- Failure handling: `aiconfigurator agent usage --ref troubleshooting`
