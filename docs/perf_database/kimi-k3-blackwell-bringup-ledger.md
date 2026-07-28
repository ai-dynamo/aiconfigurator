# Kimi-K3 Blackwell bring-up ledger

Campaign ledger for bringing Kimi-K3 (`KimiK3ForConditionalGeneration`) perf
data from the SM90 baseline (8x H20-3e, PR #1435) onto Blackwell. Read this
before collecting on any SM100/SM103 node; it records what is verified, what
each platform's serving dispatch actually runs, and what remains.

Status legend: `done` = collected, quality-gated, packaged under
`aic-core/src/aiconfigurator_core/systems/data/`; `open` = not started.

## Campaign status

| lane | system | backend/version | status | rows | notes |
|---|---|---|---|---|---|
| kda | h20_3e (SM90) | sglang 0.5.16 (kimi-k3 branch) | done | 744 | Triton kernels all phases; see coverage-hole note below |
| kda | h20_3e (SM90) | vllm 0.1.dev19262 (kimi-k3 preview) | done | 1203 | FlashKDA prefill + fused_kda_decode |
| moe (K3 shape, w4a16_mxfp4) | h20_3e (SM90) | sglang 0.5.14 | done | 488 | marlin lane; every moe_ep_size > 1 cell crashes (upstream candidate) |
| kda | b200_sxm (SM100) | sglang 0.5.16 | **done (2026-07-28)** | 987 | Triton context/generation + fused CuTeDSL DSPARK verify |
| kda | b200_sxm (SM100) | vllm 0.1.dev19262 | **done (2026-07-28)** | 1203 | collector unmodified; dispatch probes verified on SM100 |
| moe (K3 shape) | b200_sxm (SM100) | sglang | open | — | Blackwell lane is `flashinfer_mxfp4` (trtllm-gen SiTU), NOT marlin — see pre-work |
| kda + moe | b300_sxm / gb200 / gb300 (SM100/103) | both | open | — | expect kda collectors to work as on B200; SM103 still in `unverified_sms` |
| kda | rtx_pro_6000 (SM120) | both | open | — | FlashKDA claims SM120 support (vllm); sglang CuTe paths are SM100-only → Triton verify |

## B200 (SM100) campaign record — 2026-07-28

### Environment

- Node: 1x NVIDIA B200 183359 MiB, driver 595.58.03 (CUDA 13.2), 1000 W cap.
- sglang: `lmsysorg/sglang@sha256:4b8a7542...` (kimi-k3 branch build, reports
  **0.5.16**; torch 2.11.0+cu129).
- vllm: `vllm/vllm-openai@sha256:e90e2603...` (kimi-k3 preview, reports
  **0.1.dev19262+gb6bbf29dd.d20260727** — same digest as the manifest pin and
  the H20 campaign; torch 2.13.0+cu130).
- Both `kda` registry entries had SM100 removed from `unverified_sms` after
  the runs below (SM 80/89/103/120 remain unverified).

### SM100 serving dispatch (verified against framework source in-container)

sglang (kimi-k3 branch @ 0.5.16), for K3/KimiLinear architectures:

- **prefill**: Triton `chunk_kda` — `linear_attn_backend` defaults to
  `triton`; the CuTeDSL prefill pipeline exists but is opt-in.
- **decode**: Triton packed pair (`causal_conv1d_update` +
  `fused_recurrent_kda_packed_decode`) — `apply_kimi_k3_linear_attn_defaults`
  (arg_groups/kimi_k3_hook.py:44-60) pins decode back to Triton on SM100,
  preempting the generic SM100+bf16 flashinfer GDN switch ("~35% faster than
  recurrent_kda on KDA shapes"). With fp32 SSM state (K3 default) neither
  switch fires and decode is Triton anyway. → collector unchanged.
- **verify (DSPARK)**: `fused_kda_decode_mtp_dspark`, the CuTeDSL fused
  conv+chain-verify kernel (`_can_run_dspark_cutedsl_mtp`,
  kda_backend.py:858-938: capability==10, cutlass importable, draft width
  2..8, 128-dim symmetric bf16 heads, fp32 contiguous ssm state). Under spec
  the kimi_k3 hook pins `--linear-attn-verify-backend nv_cutedsl`. Serving
  folds gated RMSNorm into the kernel only on the TP8 12-head shard
  (`_prepare_fused_decode` is compiled for seg = 12*128,
  models/kimi_k3.py:1522-1560). → collector grew a dispatch branch mirroring
  exactly this probe, one row per point, `kernel_source
  "fused_kda_decode_mtp_dspark"`, per-shard onorm.

vllm (kimi-k3 preview): the collector already resolves dispatch through the
framework's own probes and needed **no changes**: `is_flashkda_supported` →
True (FlashKDA prefill), `vllm._flashkda_C` present, `fused_kda_decode`
engages (44 generation rows), `fused_recurrent_kda` chain verify.

### What was collected

| table | backend | rows | breakdown |
|---|---|---|---|
| kda_perf | sglang 0.5.16 | 987 | context 396+396 (conv_qkv3 + chunk_kda), generation 44+44, verify 107 (fused CuTeDSL) |
| kda_perf | vllm 0.1.dev19262 | 1203 | context 428+428 (conv_qkv3 + flashkda_fwd), generation 44+44+43, verify 108+108 |

Quality gates on both: 0 non-finite/zero/negative latencies, 0 duplicate
keys, device name `NVIDIA B200` on every row. Cross-platform sanity: over
shared (kernel, phase, shape, batch, seq) cells, H20/B200 latency ratio is
median 1.47x (sglang) / 1.49x (vllm) — consistent with the ~2x HBM bandwidth
step for memory-bound kernels; FlashKDA prefill gains up to ~2.7x.

### Failure groups (all classified, none hidden)

- **int32 conv guard** (sglang 52 cells, vllm 20 cells): by-design
  generation-time raise; see the corrected bound below.
- **sglang fused verify (batch=256, draft=8, 96 heads)**: single cell,
  `cudaErrorIllegalAddress`; every smaller cell passes. Suspected per-SM
  resource growth in the persistent CuTe kernel (its draft-width cap is
  documented as shared-memory-bound). `FIXME(kernel-limit)` at the call site
  in `collector/sglang/collect_kda.py`; unverified against kernel source; not
  yet filed upstream.
- **vllm packed decode (batch=1024, 96 heads)**: same single-cell failure as
  the H20 campaign (identical signature both platforms).

### Collector fixes that came out of the campaign

1. **sglang context int32 guard was 3x too loose** (`collect_kda.py`). The
   real overflow bound is `total_tokens * conv_channels` — the per-block conv
   views stride across the whole 3-block mixed_qkv buffer — not
   `total_tokens * proj_size`. Cells in `[2**31, 3*2**31)` reached the Triton
   kernel and died with `cudaErrorIllegalAddress`, and because the IMA
   poisons the CUDA context, each context case aborted at its first such
   cell. **The shipped h20_3e sglang dataset has this coverage hole**: the
   96-head shard stops at (batch=2, seq=16384) and never measured batch>=4 at
   any seq — 176 valid cells per kernel lost. B200 collected them (396 vs 220
   cells/kernel). An H20 recollection with the fixed guard would close the
   hole; until then interpolation extrapolates there.
2. **SM100 verify dispatch** in the sglang collector (above).

### Consumer-side change (Python + Rust, kept in lockstep)

`KDAKernel._query_kda_table` (Python) and `KdaOp::query` (Rust) detect
fused-verify datasets: when a dataset has no Triton verify rows but has
`fused_kda_decode_mtp_dspark` verify rows, the recurrence op is routed onto
the fused table and the conv verify op folds to 0 (its cost is inside the
fused row). The fused kernel's SOL byte model equals conv + recurrence (unit
tests assert this on both sides: `test_kda_fused_verify_routing.py`,
`kda_fused_verify_sol_is_conv_plus_recurrence`). h20_3e-style Triton datasets
and vllm's physical verify kernels are untouched. End-to-end on packaged
b200_sxm tables: sglang verify recurrence 0.0640 ms silicon + conv 0.0;
vllm 0.0504 + 0.0065 ms silicon.

## Open pre-work for the remaining lanes

1. **K3 MoE Blackwell lane (largest accuracy lever).** The kimi-k3 branch
   dispatches K3 routed experts on SM100/103 to `flashinfer_mxfp4`
   (trtllm-gen SiTU) — declared in
   `collector/cases/models/KimiK3ForConditionalGeneration_cases.yaml`
   (`w4a16_mxfp4: {90: marlin, 100: flashinfer_mxfp4, ...}`) but never
   collected: the sglang moe collector's flashinfer_mxfp4 path is unexercised
   on K3 shapes (3584/3072, 896 experts, top-16, SiTU). The B300 launch-gap
   analysis in PR #1435 attributes most of the 178-213 vs 423 tok/s/user
   HYBRID error to this missing lane. Also re-test the `moe_ep_size > 1`
   marlin crash on Blackwell images before assuming it transfers.
2. **W4A8 quant identity rows**: if a W4A8 K3 checkpoint artifact ships for
   Blackwell, it needs its own `model_case_values` moe row + allowed_modes —
   do NOT merge it into the w4a16_mxfp4 row (quant-distinct artifacts are
   separate rows).
3. **SM103 (B300/GB300)**: kda registry entries still carry 103 in
   `unverified_sms`. Expect the B200 collectors to work unchanged (CuTe MTP
   probe checks capability major == 10, which matches SM103); verify
   `torch.cuda.get_device_capability()[0] == 10` semantics on the actual
   node, run smoke, then lift the marker.
4. **sglang `kda_fused_decode` decode kernel** (TODO in the collector):
   attempt-and-verify fused decode, compiled for the TP8 12-head shard; not
   collected on any SM yet, so decode rows stay slightly pessimistic where it
   engages. Needs a `covered()`-mirroring dispatch like the verify one.
5. **H20 sglang context recollection** with the fixed int32 guard (closes the
   coverage hole documented above).
6. **SM120 (RTX Pro 6000)**: vllm FlashKDA claims SM120; sglang CuTe paths
   are SM100-only so verify falls back to the Triton pair there — the
   existing collector should be serving-truth without changes, but the probe
   ordering must be re-checked before lifting 120 from `unverified_sms`.

## Runbook for the next node (B300/GB300)

```bash
# inside each pinned runtime container, repo mounted at /workspace
export PYTHONPATH=/workspace
export COLLECTOR_LOG_DIR=/workspace/collector_logs/<backend>/kda
python3 collector/collect.py --backend <backend> --ops kda \
  --model-path moonshotai/Kimi-K3 --smoke        # gate only
python3 collector/collect.py --backend <backend> --ops kda \
  --model-path moonshotai/Kimi-K3 \
  --checkpoint-dir /workspace/collector_checkpoints/<backend>/kda --resume
```

Gotchas hit on B200 that will recur:

- NFS root-squash: run the collector as the host uid
  (`docker exec --user $(id -u):$(id -g)`), and export `HOME`, `USER`,
  `LOGNAME`, `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR`, `XDG_CACHE_HOME`,
  `CUTE_DSL_CACHE_DIR` to a writable tmpdir — an unmapped uid breaks
  `getpass.getuser()` inside torch inductor and fails every case instantly.
- The CuTe DSL verify kernel compiles per (heads, batch, draft) tuple; the
  full verify sweep costs ~25 s/shard-case in compile time on top of the
  measurements (cache the CUTE_DSL_CACHE_DIR between smoke and full runs).
- Delete stale `kda_perf.txt` in the working dir between backend runs —
  `log_perf` appends, and a mixed-backend staging file poisons the parquet.
