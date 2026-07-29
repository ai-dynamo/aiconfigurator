# Kimi-K3 Blackwell bring-up ledger

Campaign ledger for bringing Kimi-K3 (`KimiK3ForConditionalGeneration`) perf
data onto Blackwell. The collectors were first debugged on Hopper (SM90)
silicon; per the PR owner's scope decision (2026-07-28) the PR packages
**Blackwell data only** — no Hopper system or Hopper perf data ships with it.
Read this before collecting on any SM100/SM103 node; it records what is
verified, what each platform's serving dispatch actually runs, and what
remains.

Status legend: `done` = collected, quality-gated, packaged under
`aic-core/src/aiconfigurator_core/systems/data/`; `open` = not started.

## Campaign status

| lane | system | backend/version | status | rows | notes |
|---|---|---|---|---|---|
| kda | b200_sxm (SM100) | sglang 0.5.16 (kimi-k3 branch) | **done (2026-07-28)** | 987 | Triton context/generation + fused CuTeDSL DSPARK verify |
| kda | b200_sxm (SM100) | vllm 0.1.dev19262 (kimi-k3 preview) | **done (2026-07-28)** | 1203 | collector unmodified; dispatch probes verified on SM100 |
| moe (K3 shape) | b200_sxm (SM100) | sglang 0.5.14 | **done (2026-07-28)** | 3078 | `sglang_flashinfer_trtllm_moe` (flashinfer_mxfp4 lane), w4a16_mxfp4, TP 1-32 x EP 1-128 x 3 distributions, **zero failures** — the Hopper marlin EP>1 crash does not transfer; rows merged into the existing 139k-row 0.5.14 table (142,243 total) |
| kda + moe | b300_sxm / gb200 / gb300 (SM100/103) | both | open | — | expect kda collectors to work as on B200; SM103 still in `unverified_sms` |
| kda | rtx_pro_6000 (SM120) | both | open | — | FlashKDA claims SM120 support (vllm); sglang CuTe paths are SM100-only → Triton verify |

Hopper (SM90) collection history: the collectors were brought up on Hopper
silicon first (sglang kda 744 rows / vllm kda 1203 rows / K3-shape moe 488
rows on the marlin lane). That packaged data was removed from this PR on
2026-07-28 (Blackwell-first scope); it remains retrievable from the branch
history if a Hopper system is added later, but note the sglang context
coverage hole below — a Hopper RECOLLECTION with the fixed int32 guard is the
right way to bring Hopper back, not a restore.

## B200 (SM100) campaign record — 2026-07-28

### Environment

- Node: 1x NVIDIA B200 183359 MiB, driver 595.58.03 (CUDA 13.2), 1000 W cap.
- sglang: `lmsysorg/sglang@sha256:4b8a7542...` (kimi-k3 branch build, reports
  **0.5.16**; torch 2.11.0+cu129).
- vllm: `vllm/vllm-openai@sha256:e90e2603...` (kimi-k3 preview, reports
  **0.1.dev19262+gb6bbf29dd.d20260727** — same digest as the manifest pin;
  torch 2.13.0+cu130).
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
keys, device name `NVIDIA B200` on every row. Cross-platform sanity against
the Hopper bring-up measurements: over shared (kernel, phase, shape, batch,
seq) cells the Hopper/B200 latency ratio is median 1.47x (sglang) / 1.49x
(vllm) — consistent with the ~2x HBM bandwidth step for memory-bound
kernels; FlashKDA prefill gains up to ~2.7x.

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
  the Hopper campaign (identical signature both platforms).

### Collector fixes that came out of the campaign

1. **sglang context int32 guard was 3x too loose** (`collect_kda.py`). The
   real overflow bound is `total_tokens * conv_channels` — the per-block conv
   views stride across the whole 3-block mixed_qkv buffer — not
   `total_tokens * proj_size`. Cells in `[2**31, 3*2**31)` reached the Triton
   kernel and died with `cudaErrorIllegalAddress`, and because the IMA
   poisons the CUDA context, each context case aborted at its first such
   cell. The Hopper bring-up dataset was collected under the loose guard and
   silently lost the 96-head shard above (batch=2, seq=16384) — any future
   Hopper collection must use the fixed guard (B200 collected 396 vs 220
   cells/kernel).
2. **SM100 verify dispatch** in the sglang collector (above).

### Consumer-side change (Python + Rust, kept in lockstep)

`KDAKernel._query_kda_table` (Python) and `KdaOp::query` (Rust) detect
fused-verify datasets: when a dataset has no Triton verify rows but has
`fused_kda_decode_mtp_dspark` verify rows, the recurrence op is routed onto
the fused table and the conv verify op folds to 0 (its cost is inside the
fused row). The fused kernel's SOL byte model equals conv + recurrence (unit
tests assert this on both sides: `test_kda_fused_verify_routing.py`,
`kda_fused_verify_sol_is_conv_plus_recurrence`). Triton-verify (Hopper-style)
datasets and vllm's physical verify kernels are untouched. End-to-end on
packaged b200_sxm tables: sglang verify recurrence 0.0640 ms silicon + conv
0.0; vllm 0.0504 + 0.0065 ms silicon.

## Open pre-work for the remaining lanes

(The sglang K3 MoE Blackwell lane — previously the largest accuracy lever —
completed 2026-07-28 with zero failures; see the campaign table. Any future
K3 rows for another artifact/quant MERGE into the shared 0.5.14 table; never
overwrite.)

1. **K3 MegaMoE module lane (sglang, the launch serving path).** Probe
   evidence (2026-07-28, kimi-k3 branch image): K3 MoE serves through the
   patched DeepGEMM mega kernel when `get_moe_a2a_backend().is_megamoe()` —
   "MegaMoE SiTU sentinel" at models/kimi_k3.py:132, fused a2a+GEMM over the
   EP symmetric buffer (deep_gemm exposes `bf16/fp8/fp8_fp4_mega_moe`,
   `mega_moe_pre_dispatch`, symm-buffer APIs, plus `_sm90` variants). This is
   the same DeepGEMM implementation the vllm branch calls
   (`moe_backend == "deep_gemm_mega_moe"`); TRT-LLM's `wideep_moe` (cutlass +
   real A2A) is a DIFFERENT implementation and not a proxy. Collection plan:
   extend the `collect_dsv4_megamoe.py` pattern (module boundary = symm
   buffer lookup -> pre-dispatch -> mega kernel; gate/topk outside) to the K3
   shape (latent 3584, inter 3072, 896x16, SiTU sentinel), mirroring exactly
   which deep_gemm entry point and dtype the branch's `_use_mega_moe` path
   invokes. HARDWARE: the collector is cross-rank
   (`ep_size == WORLD_SIZE`), so EP8 needs an 8-GPU Blackwell node — cannot
   run on 1x B200. Perspective from the GB300 DSV4 megamoe table: at 8
   global tokens the module (incl. cross-rank transfer) costs ~0.10-0.11
   ms/layer — the same magnitude as the collected flashinfer_mxfp4 lane — so
   this lane matters for serving-truth fidelity and higher concurrency, not
   as a big bs1 latency win.
2. **K3 MoE on the vllm backend**: unblocked at the case-population level
   (kimi_linear grouped-topk mapping + the num_expert_group field spelling in
   `collector/vllm/collect_moe.py`), but not collected. Two open questions
   before running it: whether the moe family should route to the kimi-k3
   preview image instead of stock 0.24 (K3 serves only on the branch), and
   whether the runtime's FusedMoE supports the SiTU activation K3 declares —
   probe before burning a full grid.
3. **W4A8 quant identity rows**: if a W4A8 K3 checkpoint artifact ships for
   Blackwell, it needs its own `model_case_values` moe row + allowed_modes —
   do NOT merge it into the w4a16_mxfp4 row (quant-distinct artifacts are
   separate rows).
4. **SM103 (B300/GB300)**: kda registry entries still carry 103 in
   `unverified_sms`. Expect the B200 collectors to work unchanged (the CuTe
   MTP probe checks capability major == 10, which matches SM103); verify on
   the actual node, run smoke, then lift the marker.
5. **sglang `kda_fused_decode` decode kernel** (TODO in the collector):
   attempt-and-verify fused decode, compiled for the TP8 12-head shard; not
   collected on any SM yet, so decode rows stay slightly pessimistic where it
   engages. Needs a `covered()`-mirroring dispatch like the verify one.
6. **Hopper system (if ever re-added)**: recollect sglang kda context with
   the fixed int32 guard rather than restoring the old dataset (coverage hole
   above), and re-run the moe marlin lane to reconfirm the EP>1 crash before
   filing upstream.
7. **SM120 (RTX Pro 6000)**: vllm FlashKDA claims SM120; sglang CuTe paths
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
