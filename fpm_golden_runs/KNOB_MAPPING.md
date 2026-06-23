# FPM ↔ AIC Knob Mapping

Reference for keeping the FPM golden runs in this directory aligned with the AIC
layerwise model they are compared against. Use this as a per-run correctness
checklist before trusting any FPM-vs-AIC number (charts or summary tables).

**TL;DR:** the FPM deployment knobs themselves are fine — they are vLLM's
self-chosen defaults for the model + GPU memory, internally consistent. The real
risks are (1) the plot tool's default `max_num_seqs` does **not** match the golden
runs, (2) `fuse_allreduce_rms` differs between golden runs while AIC assumes it is
always on, and (3) the `effective_vllm_config.json` does **not** record expert/
data parallelism or weight quantization — those must be read from the launch
`engine_args` / directory name, not from the effective config.

## Three sources of truth

| Source | What it holds |
|---|---|
| `<run>/<case>/effective_vllm_config.json` | vLLM's *resolved* config (what actually ran). Also mirrored in `vllm_metadata.json → effective_config`. |
| `<run>/<case>/vllm_metadata.json → requested.engine_args` | The launch command — the **only** place EP/DP and quantization intent are recorded. |
| AIC side: `RuntimeConfig` (`src/aiconfigurator/sdk/config.py`) + `_model_defaults()` (`collector/layerwise/diagnostics/compare_aic_layerwise_fpm.py`) | The parameters fed to the layerwise backend. |

The two comparison tools consume the AIC side differently:

- `collector/layerwise/diagnostics/compare_aic_layerwise_fpm*.py` (**summary
  tables**) — auto-derives the knobs from FPM via `_load_fpm_max_num_seqs` and
  `_resolve_auto_max_num_batched_tokens` (falls back to the Dynamo runtime config).
  **This path stays aligned automatically.**
- `tools/plot_fpm_vs_aic.py` (**charts**) — uses **hardcoded** `RuntimeConfig`
  defaults (`max_num_batched_tokens=2048`, `max_num_seqs=256`), overridable only
  via `--vllm-max-num-batched-tokens` / `--vllm-max-num-seqs`. **This path can
  silently drift — see mismatch #1.**

## Mapping table

Values shown are for the canonical qwen36 run
`fpm_upfront_qwen36_moe_full_once_20260613_201336` (Qwen3.6-35B-A3B, bf16).

| vLLM knob (effective config) | Golden value | AIC counterpart | Used by AIC for | Aligned? |
|---|---|---|---|---|
| `scheduler_config.max_num_batched_tokens` | 2048 | `RuntimeConfig.vllm_max_num_batched_tokens` | chunked-prefill chunk split (context/mixed) | ✅ plot default 2048; summary auto-reads |
| `scheduler_config.max_num_seqs` | 128 | `RuntimeConfig.vllm_max_num_seqs` | decode/mixed perf-table lookup dimension | ⚠️ **plot default 256 ≠ 128**; summary auto-reads 128 |
| `model_config.dtype` (activation) | bfloat16 | `_Model` gemm/moe/fmha quant (default bf16 for qwen36) | GEMM/MoE/attention cost tables | ✅ qwen36 — but justify from **checkpoint** (no `--quantization` in engine_args ⇒ bf16 weights), not from this field, which is activation-only (see #3) |
| `cache_config.cache_dtype` | auto (=bf16) | `_Model.kvcache_quant_mode` (default bf16) | KV-cache / attention cost | ✅ qwen36 (DSV4: fp8 ↔ fp8 ✅) |
| `cache_config.enable_prefix_caching` | false | none (`RuntimeConfig.prefix` = prompt prefix *length*, different concept) | — | ✅ both model no prefix reuse |
| `cache_config.block_size` | 16 | not a RuntimeConfig knob — baked into collected attention data | paged-attention kernel timing | ✅ if collector used same block_size |
| `compilation_config.cudagraph_mode` | FULL_AND_PIECEWISE | collected with `use_cuda_graph=True` | decode kernel timing | ✅ |
| `optimization_level` / `enforce_eager` | O2 / false | (implies cuda graph) | — | ✅ |
| `parallel_config.tensor_parallel_size` | per case | `_Model.tp_size` (attention TP) | comm + per-rank shapes | ✅ |
| **expert / data parallelism** (NOT in effective config) | from `--enable-expert-parallel` + `--tensor-parallel-size` + `--data-parallel-size` | `_Model.moe_ep_size`, `_Model.moe_tp_size`; AIC auto-derives `attention_dp_size = (moe_tp·moe_ep)//tp` | MoE per-rank experts + EP all-to-all + attention-DP replicas | ✅ topology aligned (verified, see #3) — but config records only TP; read EP/DP from engine_args |
| `compilation_config.pass_config.fuse_allreduce_rms` | run-dependent (see #2) | `_LAYERWISE_USE_FUSED_ALLREDUCE_RMS=True` (always) | decode TP comm kernel | ⚠️ mismatch on unfused runs (#2) |
| `model_config.max_model_len` | 262144 | none | KV capacity only, not per-step latency | N/A |

## Known mismatches / gotchas

### #1 — Chart tool `max_num_seqs` default (256) ≠ golden (128)  · severity: medium
`tools/plot_fpm_vs_aic.py` defaults `RuntimeConfig.vllm_max_num_seqs=256`, but
every qwen36 golden case ran with `max_num_seqs=128`. `max_num_seqs` is a decode/
mixed perf-table lookup dimension on the AIC side, so a mismatch can hit a
different curve or trigger a fallback. **Always pass `--vllm-max-num-seqs <value
matching that run>`** when charting, or prefer the summary tool (auto-aligned).
(The README's gen example passes `--vllm-max-num-seqs 64` because it targets a
*separate* `max_num_seqs=64` decode run — not these 128 golden runs.)

> **Carry this into the mixed-step investigation.** The committed qwen36
> mixed/ctx charts were generated *without* `--vllm-max-num-seqs`, i.e. at the
> 256 default while the golden ran at 128. So part of the "MoE mixed is really
> bad" error could be this lookup-dimension mismatch, not `_get_mix_step_latency`
> composition. **Regenerate the mixed comparison at `--vllm-max-num-seqs 128`
> first**, then attribute the residual to composition.

### #2 — `fuse_allreduce_rms` is inconsistent across golden runs  · severity: medium (tp>1 only)
- `fpm_upfront_qwen36_moe_full_once_20260613_201336`: tp1 = false (no allreduce),
  **tp2/tp4/tp8 = true** (fused allreduce+residual+RMSNorm).
- `fpm_gap_qwen36_dp_ep_official_20260614_025420`: **all false**, including tp>1.

AIC fixes `_LAYERWISE_USE_FUSED_ALLREDUCE_RMS=True` and always models the cheaper
fused kernel. On the gap-fill (unfused) tp>1 runs this **under-counts decode TP
comm**. When comparing tp>1, check this field per run; treat fused-vs-unfused as
a known source of tp>1 decode/mixed error.

### #3 — `effective_vllm_config.json` does not record EP/DP or weight quant  · severity: high (silent)
The effective config dumps only `tensor_parallel_size` — there is **no**
data-parallel / expert-parallel / `enable_expert_parallel` field, and
`model_config.dtype` is the *activation* dtype, not the checkpoint's weight
quantization.

- **EP/DP:** read from `vllm_metadata.json → requested.engine_args`. Rule
  (verified): `--enable-expert-parallel` with `tensor_parallel_size=T` and
  `data-parallel-size=D` ⇒ MoE expert-parallel size `= T·D`, MoE tensor-parallel
  `= 1`. Confirmed on two topologies:
  - `tp2_ep2`: `--tensor-parallel-size 2 --enable-expert-parallel` (D=1) ⇒ EP=2.
  - `tp1_ep4` (gap_fill): `--tensor-parallel-size 1 --data-parallel-size 4
    --enable-expert-parallel` ⇒ EP=4.

  AIC matches both: `_Config` auto-derives `attention_dp_size =
  (moe_tp·moe_ep)//tp`, so `tp1_ep4` ⇒ `attention_dp_size=4` (4 attention
  replicas at batch/4) and `tp2_ep2` ⇒ `attention_dp_size=1`. The topology — not
  just expert sharding — is therefore modeled correctly; **the gap is only that
  `effective_vllm_config.json` doesn't record EP/DP**, so always cross-check
  engine_args rather than trusting the config alone.
- **Weight quant:** determined by the checkpoint (e.g. DSV4-Flash is w4a8/fp8),
  not visible in the effective config. AIC sets it explicitly in
  `_model_defaults()` (DSV4: `moe_quant_mode=w4a8_mxfp4_mxfp8`,
  `gemm_quant_mode=fp8_block`). Verify against the model card, not the config.

## Per-run verification checklist

Before trusting an FPM-vs-AIC comparison for a `<run>/<case>`:

```
[ ] effective max_num_batched_tokens == AIC vllm_max_num_batched_tokens (2048)
[ ] effective max_num_seqs           == AIC vllm_max_num_seqs   (chart tool: pass it explicitly!)
[ ] model_config.dtype + cache_dtype  match _model_defaults() quant for this model
[ ] enable_prefix_caching == false (AIC models no reuse)
[ ] engine_args EP/DP topology       == AIC (tp_size, moe_tp_size, moe_ep_size)
[ ] checkpoint weight quant           == _model_defaults() moe/gemm quant_mode
[ ] fuse_allreduce_rms (tp>1)         noted; AIC assumes fused
[ ] cudagraph on; block_size matches collector
```
