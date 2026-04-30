# Result Interpretation

Do not rely only on the console table when explaining an AIC run. Use
`--save-dir` and inspect the saved CSV and YAML artifacts.

## Files to Inspect

- `best_config_topn.csv`: top configurations after picking. Start here.
- `pareto.csv`: candidate Pareto frontier for the experiment or merged mode.
- `exp_config.yaml`: normalized experiment config, including defaults and YAML
  patches.
- `top*/generator_config.yaml`: deployment-facing config derived from the picked
  row.
- `pareto_frontier.png`: useful for visual comparison, not for exact numbers.

## Logs and Normalized Config

Read the logs and normalized config before trusting the top result:

- Confirm the actual quantization modes AIC used. Do not assume the model name
  alone implies the precision path.
- Confirm `nextn` and `nextn_accept_rates`; MTP may be enabled by defaults for
  some model families or by YAML patch.
- Confirm `enable_wideep`, `moe_backend`, `attention_backend`, and `enable_eplb`
  when MoE paths are being compared.
- Check the enumerated parallel configs in the log. The listed TP/PP/DP and
  MoE TP/EP candidates should match the intended YAML search space.
- If the enumeration is empty or much narrower than expected, inspect
  `num_gpu_per_worker`, `tp_list`, `dp_list`, `moe_tp_list`, and `moe_ep_list`
  consistency before interpreting performance.
- Use saved `exp_config.yaml` as the final record of what AIC actually ran.

## Metrics That Matter

- `tokens/s/gpu_cluster`: throughput normalized to the requested GPU budget.
  Use this for agg vs disagg and backend comparisons.
- `tokens/s/gpu`: per-GPU throughput for the modeled worker or replica. Do not
  substitute it for cluster-normalized throughput.
- `tokens/s/user`: user-visible generation rate implied by TPOT.
- `request_rate`: per-replica request rate. Multiply by replicas for cluster
  request rate.
- `concurrency`: per-replica concurrency. Console output may show
  `per_replica x replicas`.
- `num_total_gpus`: GPUs per aggregate worker or per disaggregated replica.
- `total_gpus_used`: requested GPUs are not always fully used; check the saved
  CSV or console `total_gpus (used)` column.
- `ttft`, `tpot`, `request_latency`: compare against the user SLA before
  recommending a config.
- `power_w`: only treat it as meaningful when most rows have non-zero power data.

## Common Interpretation Pitfalls

- A console label may be shorter than the underlying metric. Prefer CSV column
  names when writing analysis.
- `default --backend auto` merges backend results by serving mode. Keep the
  `backend` column in the explanation.
- Disaggregated configs are scalable replicas. Explain both per-replica shape
  and cluster-scale replicas.
- `HYBRID`, `EMPIRICAL`, and `SOL` are not measured silicon experiment results.
  Do not mix them into final analysis or deployment config selection.
- `SOL` may be useful as an upper-bound sanity check, but final rows should be
  based on `SILICON`.
- A configuration can appear in a Pareto frontier yet fail a stricter SLA if
  `--strict-sla` was not used.
- If no rows meet SLA, report which constraint filtered the run instead of only
  saying "no result".
- If result count is very small, report that the search space or SLA is too
  narrow and explain which one is more likely.
- If winning rows use very small batch sizes, throughput may be limited by
  under-filled decode or overly tight latency constraints. Check whether TTFT or
  TPOT has less headroom before tuning.

## Useful Summary Shape

When reporting a chosen config, include:

1. Input: model, backend/version, system, total GPUs, ISL/OSL, SLA, database mode.
2. Effective modeling choices: quantization, MTP, WideEP/DeepEP/EPLB, backend
   version, and database mode.
3. Winner: agg/disagg, backend, normalized throughput, TTFT, TPOT, request
   latency, request rate, concurrency.
4. Deployment shape: replicas, GPUs per replica, workers, TP/PP/DP, MoE TP/EP,
   batch sizes.
5. Caveats: result count, small-batch concerns, missing power data, generated
   config version, and any manually patched YAML fields.
