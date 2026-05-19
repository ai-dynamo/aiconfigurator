---
name: aiconfigurator-usage
description: Use when an agent needs to run the installed AIConfigurator CLI to check support, estimate LLM serving performance, compare aggregate and disaggregated deployments, or generate deployment configs.
---

# AIConfigurator Usage

AIC is an offline planning tool for LLM serving deployments. It estimates
performance, memory, and practical serving configs from a model, hardware
system, backend, workload shape, and SLA.

Do not treat AIC output as a live benchmark. When silicon data is missing or a
model/backend is new, report the missing coverage instead of silently falling
back to another database mode.

For any result that will drive final analysis, deployment sizing, or generated
configs, use `SILICON`. Use `SOL` only as an upper-bound or sanity-check signal.
Avoid using `HYBRID` and `EMPIRICAL` for agent-led experiments unless a developer
explicitly asks for a non-final what-if estimate, and never present those results
as measured silicon data.

## Default Workflow

1. Identify the model, system, backend, total GPUs, ISL, OSL, and SLA.
2. Optionally run `aiconfigurator cli support` to inspect support.
3. Run `aiconfigurator cli default` first to get a rough operating range:
   agg/disagg direction, backend candidates, GPU-per-replica shape, TP/PP/DP,
   batch sizes, throughput, and SLA feasibility.
4. Use the rough `default` result to build a precise `exp` YAML, usually focused
   on disaggregated serving and a narrower search space.
5. Start from the bundled CLI example template, then keep only the experiments
   and fields needed for the target workload and search definition.
6. Use `--save-dir` by default when results need analysis, reproducibility, or
   generated deployment artifacts.
7. Use `aiconfigurator cli estimate` only for a known single configuration or
   step-level diagnosis.
8. Use `aiconfigurator cli generate` only when the user wants a naive config
   without performance sweeping.

## Mode Selection

- `support`: pre-flight support matrix check.
- `default`: first-pass sizing and rough agg/disagg/backend exploration.
- `exp`: precise agent-controlled experiments from YAML, especially disagg.
- `estimate`: single-point latency/power estimate for a specific config.
- `generate`: naive config generation; no SLA optimization.

## Guardrails

- Do not invent system names, backends, versions, or quant modes.
- Keep `SILICON` as the source for final experiments, analysis, and deployment
  config generation.
- If support or silicon data is missing for a new model, report the gap. Do not
  substitute `HYBRID` or `EMPIRICAL` into final results.
- Use `SOL` only to estimate an upper bound or diagnose whether a target is
  physically plausible.
- Do not enable MTP, WideEP, EPLB, or DeepEP just because the flag exists. Check
  model family, backend, and database support, then verify the generated/search
  config reflects the intended path.
- Do not override quantization in YAML unless the user asks for a specific
  quantization study or the runtime/model config requires it.
- If a command fails due to support or database coverage, report the exact model,
  system, backend, version, and database mode.

## Load References Only When Needed

- CLI modes and required arguments: `aiconfigurator agent usage --ref cli-modes`
- Single experiment YAML authoring: `aiconfigurator agent usage --ref single-experiment-yaml`
- CI-covered CLI YAML template: `aiconfigurator agent usage --ref experiment-template`
- Result analysis: `aiconfigurator agent usage --ref result-interpretation`
- SDK/per-step latency breakdown: `aiconfigurator agent usage --ref sdk-step-breakdown`
- Deployment and benchmark artifacts: `aiconfigurator agent usage --ref deployment-bench`
- Feature pitfalls such as MTP, WideEP, DeepEP, EPLB, and quantization:
  `aiconfigurator agent usage --ref feature-pitfalls`
- Command examples: `aiconfigurator agent usage --ref examples`
- Failure handling: `aiconfigurator agent usage --ref troubleshooting`
