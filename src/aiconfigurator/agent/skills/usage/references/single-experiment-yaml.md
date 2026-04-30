# Experiment YAML Guide

Use `aiconfigurator cli exp --yaml-path ...` after a rough `default` run when
the agent needs precise control over a planned experiment. This is usually the
right path for narrowing a disaggregated search after `default` has shown the
rough backend, GPU shape, batch-size, and SLA range.

Start from the bundled template:

```bash
aiconfigurator agent usage --ref experiment-template > template.yaml
```

Then edit the copied YAML for the user's model, system, backend, GPU budget,
workload, SLA, and disaggregated search space.

## Authoring Workflow

1. Copy the complete template and keep only the experiment entries needed for
   the task.
2. Run or inspect a `default` result first and extract the promising disagg
   region: backend, GPUs per replica, TP/PP/DP, MoE TP/EP, worker counts, batch
   sizes, and whether SLA failures are TTFT- or TPOT-driven.
3. Keep `mode: patch` unless a developer explicitly asks for full replacement.
4. Keep `database_mode: SILICON` for experiments that will drive analysis,
   deployment sizing, or generated configs.
5. Put workload and SLA at the experiment top level: `isl`, `osl`, `prefix`,
   `ttft`, `tpot`, and optionally `request_latency`.
6. Change search-space lists under `prefill_worker_config`,
   `decode_worker_config`, and `replica_config` for precise disagg experiments.
   Use `worker_config` for aggregate-only follow-ups.
7. Do not override quantization by default. Let model config and AIC defaults
   infer quantization unless the user asks for a quantization study or a known
   runtime requires an explicit mode.
8. Run with `--save-dir`, then inspect `exp_config.yaml`,
   `best_config_topn.csv`, `pareto.csv`, and `top*/generator_config.yaml`.

## Default-to-YAML Handoff

Use the `default` run to decide what to keep in the YAML:

- Keep the backend and serving mode that look promising under `SILICON`.
- Center `num_gpu_per_replica` and `max_gpu_per_replica` around realistic
  disagg shapes from the default result.
- Narrow `prefill_worker_config` around TTFT-sensitive candidates.
- Narrow `decode_worker_config` around TPOT-sensitive candidates.
- Keep batch-size caps practical; do not blindly expand decode batch size just
  because the theoretical result improves.
- Preserve the workload and SLA from the default command unless the user is
  intentionally changing the target.

## Post-Run Review Loop

After every YAML run, review the run before changing the next YAML:

- Logs: confirm actual quantization modes, MTP settings, WideEP/DeepEP/EPLB
  settings, and enumerated parallel configs.
- `exp_config.yaml`: confirm defaults plus patches match the intended search
  definition.
- `best_config_topn.csv`: check whether there are enough rows to compare and
  whether top rows use suspiciously small batch sizes.
- `pareto.csv`: check whether the Pareto frontier is sparse or missing an
  expected region.
- SLA headroom: compare TTFT and TPOT against targets. If TTFT is tight, tune
  prefill-side search or TTFT. If TPOT is tight, tune decode-side search or
  TPOT. Do not relax both without stating why.

## Multi-Experiment Comparisons

Use multiple YAML entries when comparing options. Keep only one intentional
difference per experiment group when possible, so the result explains the
option rather than a bundle of unrelated changes.

Good comparison axes:

- Backend: `trtllm` vs `sglang` vs `vllm`, when silicon coverage exists.
- Disagg replica shape: smaller versus larger `num_gpu_per_replica`.
- Prefill search: different TP/DP/MoE EP ranges for TTFT-sensitive workloads.
- Decode search: different TP/DP/MoE EP ranges or batch caps for TPOT-sensitive
  workloads.
- WideEP or DeepEP path: enabled versus disabled, only for supported MoE paths.
- MTP: `nextn` variants only when the model/runtime should use MTP.

Avoid comparison axes that hide the meaning of the result:

- Do not compare `SILICON` against `HYBRID` or `EMPIRICAL` as if they were the
  same evidence type.
- Do not change quantization, backend, and SLA all in one experiment unless the
  user explicitly asked for a broad scenario comparison.
- Do not reuse a what-if `SOL` result for deployment config selection.

## Database Mode Policy

- `SILICON`: required for final analysis, deployment sizing, and generated
  configs.
- `SOL`: allowed as an upper-bound or feasibility sanity check. It should not be
  mixed into final experiment tables.
- `HYBRID` and `EMPIRICAL`: avoid for agent-led experiments. If a developer
  explicitly requests them for a non-final what-if, label the output clearly and
  do not reuse those rows for deployment config decisions.
- Missing silicon data is a coverage gap. Report it with model, system, backend,
  version, workload, and database mode.

## Top-Level Experiment Fields

- `serving_mode`: `agg` or `disagg`.
- `model_path`: Hugging Face model ID or local model config directory.
- `total_gpus`: GPU budget for the cluster-level search.
- `system_name`: prefill or aggregate system.
- `decode_system_name`: decode system for disaggregated serving. Omit only when
  it is the same as `system_name`.
- `backend_name`: `trtllm`, `vllm`, or `sglang`.
- `backend_version`: optional. Prefer omitting it unless the user needs a
  specific silicon database version.
- `database_mode`: keep `SILICON`.
- `enable_wideep`: only for MoE experiments with supported WideEP data.

## Search-Space Fields

Aggregate experiments use `config.worker_config`. Disaggregated experiments,
which are the usual precise follow-up, use `config.prefill_worker_config`,
`config.decode_worker_config`, and `config.replica_config`.

Useful search fields:

- `num_gpu_per_worker`: exact worker GPU counts to evaluate.
- `tp_list`, `pp_list`, `dp_list`: attention/model parallel candidates.
- `moe_tp_list`, `moe_ep_list`: MoE parallel candidates.
- `max_gpu_per_replica`: cap for disaggregated replica size.
- `max_prefill_worker`, `max_decode_worker`: worker-count caps per replica.
- `prefill_max_batch_size`, `decode_max_batch_size`: practical per-worker batch
  caps.

## MTP and WideEP

Do not add MTP just because a model is MoE. Only set `nextn` when the model and
backend runtime are intended to use MTP/speculative decoding.

```yaml
  config:
    nextn: 2
    nextn_accept_rates: [0.85, 0.3, 0.0, 0.0, 0.0]
```

Only enable WideEP for supported MoE paths. WideEP changes the search space, so
verify the normalized `exp_config.yaml` after the run.

```yaml
  enable_wideep: true
  config:
    worker_config:
      moe_tp_list: [1]
      moe_ep_list: [8, 16, 32]
```

For disaggregated experiments, apply the corresponding search-space edits to
both `prefill_worker_config` and `decode_worker_config` when appropriate.

## Review Checklist

- The experiment keeps `database_mode: SILICON`.
- Quantization is not overridden unless there is a specific reason.
- Search lists are narrow enough to run in reasonable time.
- WideEP, DeepEP, EPLB, and MTP are only enabled with an explicit rationale.
- Logs show the expected quantization, MTP, and enumerated parallel configs.
- Result rows are not unexpectedly absent, sparse, or dominated by tiny batch
  sizes without explanation.
- Saved `exp_config.yaml` reflects the intended normalized search definition.
