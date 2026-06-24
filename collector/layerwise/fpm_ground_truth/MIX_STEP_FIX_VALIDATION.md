# Mixed-step MoE fix validation (qwen36 tp4_ep4) — WIP findings

Validating `vllm_backend._get_mix_step_latency`'s decode-MoE overlay (commit
`b6ad5e8f`) against the golden tp4_ep4 FPM mixed rows. In-image on the SLURM
B300 node (already inside the Dynamo vLLM 0.20.1 image — direct processes, no
docker). `nsys` installed from `/workspace/software/nsight-systems-2026.2.1_cli.deb`
(`dpkg -i`; symlinked to `/usr/local/bin/nsys`; glib dep warning is harmless,
profiling + sqlite export both work).

## Environment / data findings
- Model `Qwen/Qwen3.6-35B-A3B` local at `/workspace/models/Qwen3.6-35B-A3B`;
  exposed to HF offline via a cache symlink at `/workspace/models/hf_home`
  (`HF_HOME=/workspace/models/hf_home HF_HUB_OFFLINE=1`).
- Collector tp→parallelism map: `tp=4, ep=4` → `attn_tp=4, moe_tp=1, moe_ep=4`
  (matches golden). Command: `--tp-sizes 4 --ep-sizes 4 --phases both`.
- Fresh layerwise collected at `runs/layerwise_qwen36_tp4ep4_mixfix/layerwise.csv`
  (18 ctx new_tokens{256..8192}×past{0,4096,8192} + 15 gen batch{1,2,4,8,16}×past{2048,4096,8192}),
  `includes_moe=False`, gen `latency_source=execute_model_gpu`.
- The MoE overlay `collector/layerwise/wip/moe_perf.txt` only has **`power_law_1.2`**
  for ep=4 (rich num_tokens grid); `sampled_zipf_1.2` exists only for ep=1.
  The model default `workload_distribution=power_law` → `power_law_1.2`, OK for ep4.

## Eval blocker discovered (THE key WIP finding)
`plot_fpm_vs_aic.py --vllm-max-num-seqs 128` makes the **decode** detail lookup go
through the `max_num_seqs` index requiring layerwise GEN rows **tagged**
`max_num_seqs=128`. Neither the system perf CSV nor a fresh collection tags it
(the collector leaves `max_num_seqs` empty), so every mixed row raises
`PerfDataNotAvailableError: Layerwise data for max_num_seqs=128 not found for
.../GEN/tp4` → 0 mixed prediction points → no MAPE. (The plot tool defaults
`--vllm-max-num-seqs 256`, which fails the same way on untagged rows; the summary
tool defaults it to *none* to bypass the index — that's why ctx-only summaries work.)

Resolution: tag the collected GEN/CTX rows with `max_num_seqs=128` (and
`max_num_batched_tokens=2048`) — matches golden effective config; the measured
batch≤16 latencies are independent of the scheduler cap, so the tag is only a
matching key. Then re-run the eval fix-on / (git stash vllm_backend.py) fix-off.

## Status
- [x] nsys installed, model+HF cache ready, tp4/ep4 ctx+gen collected
- [x] diagnosed the max_num_seqs-tagging eval blocker
- [ ] tag rows, run fix-on/off, report MAPE + median AIC/FPM ratio + verdict

## Push note
This container has no ssh/gh/token; commits are local only but `/workspace` is the
persistent lustre FS so they survive the compute session. Push requires a
credential from the user.
