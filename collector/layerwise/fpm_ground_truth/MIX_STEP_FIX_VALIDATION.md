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

## Results (qwen36 tp4_ep4 mixed, fresh tp4 layerwise tagged max_num_seqs=128, n=92 of 300)

| version  | MAPE   | median(AIC/FPM) |
|----------|--------|-----------------|
| fix OFF  | 167.3% | 2.193           |
| fix ON   | 171.1% | 2.223           |

(`fix OFF` = `git checkout b6ad5e8f^ -- src/.../vllm_backend.py`; the fix is
committed, so `git stash` finds nothing — must revert the commit's hunk.)

Decode rows MUST be envelope-source: the SDK's `_validate_decode_layerwise_detail`
**rejects** the system CSV's `span` (per-module) GEN rows
("representative module timing, not a full scheduler step") and requires
`execute_model_gpu/schedule_to_update/worker_wall`. So the fresh collection (not
the system perf CSV) is the correct input for the mixed eval.

## Verdict
- **The fix's effect is tiny: +0.030 median ratio (2.193→2.223), +3.8pts MAPE —
  and it slightly WORSENS the prediction here.** It does NOT dramatically
  over-correct (the feared weight-load double-count would be a large jump; this is ~1.4%).
- **Baseline does NOT under-predict — it OVER-predicts ~2.2×**, opposite of the
  fix's premise. But this absolute over-prediction is dominated by an **inflated
  ctx envelope**, not the decode-MoE term: every prefill bucket over-predicts
  (118–403% MAPE) and the fix moves each by only +3–7pts. My fresh ctx
  `latency_source` (schedule_to_update/worker_wall) carries large scheduler/wall
  overhead — ctx 256 tok = 35.6ms vs golden context FPM ~17ms, with a ~35ms fixed
  floor; 8192 tok = 207ms (implausibly high for a 3B-active MoE on 4×B300).
- **Conclusion: the decode-MoE overlay is NOT the dominant mixed-step error here —
  the ctx envelope is.** The fix can be neither validated nor refuted against this
  ~2.2× ctx noise floor; as-is it is a small increment that slightly increases
  over-prediction on these ctx-prefill-dominated golden mixed steps.

## Recommended next step
Recollect **ctx with a compute-only latency source** (execute_model_gpu, as the
gen rows already use) so the ctx envelope matches FPM wall time; only then can the
decode-MoE overlay's effect be isolated. The golden mixed rows are also
ctx-dominated (ctx_tokens up to ~8184, decode 1–15), under-sampling the
decode-heavy regime where the overlay would matter most.

## Status
- [x] nsys installed, model+HF cache ready, tp4/ep4 ctx+gen collected
- [x] diagnosed max_num_seqs-tagging eval blocker + decode envelope-source requirement
- [x] fix on/off measured; verdict above
- [ ] (recommended) clean ctx-envelope recollection to isolate the decode-MoE term

## Push note
This container has no ssh/gh/token; commits are local only but `/workspace` is the
persistent lustre FS so they survive the compute session. Push requires a
credential from the user.
