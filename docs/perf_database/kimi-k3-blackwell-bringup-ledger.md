# Kimi-K3 Blackwell bring-up ledger

Campaign ledger for bringing Kimi-K3 (`KimiK3ForConditionalGeneration`) perf
data onto Blackwell. The collectors were first debugged on Hopper (SM90)
silicon; per the PR owner's scope decision (2026-07-28) the PR initially
packaged **Blackwell data only**. On 2026-08-01 the owner widened scope to
ada + hopper + blackwell **sglang-only** and the remaining sglang lanes were
collected via the aic-auto-collector harness (see the 2026-08-01 section);
vllm remains b200-only by that scope decision. This document records what is
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
| moe (K3 shape, **w4a8_mxfp4_mxfp8**) | b300_sxm (SM103) | sglang 0.5.14 | **done (2026-07-29)** | 3078 | the Blackwell serving-truth precision lane: mxfp8 activations (`Mxfp4MoEMethod` default, mxfp4.py:1311-1330 @ kimi-k3 branch); merged into the shared b300 0.5.14 table (251,441 total). Same-shape delta vs the bf16-activation w4a16 lane at 8 tokens: 89.8 vs 102.5 µs/layer (~12%) — small-batch MoE is weight-bytes-bound, so the E2E dummy-vs-model MoE gap is dominated by the dummy routing collapse artifact, not precision. SDK: `resolve_kimi_k3_moe_arch_mode` routes K3+sglang+Blackwell onto this label; Hopper keeps W4A16 marlin |
| kda | b300_sxm (SM103) | sglang 0.5.16 | **done (2026-07-29)** | 976 | SM103 verified and lifted from the sglang kda `unverified_sms`; includes the NEW `kda_fused_decode` generation lane for the TP8 12-head shard (attempt-and-verify fused conv+recurrence+onorm, 6.6 µs/layer @ 8 tokens vs 15 µs for the Triton pair — matches the E2E nsys 0.40 ms/step); same known failures as B200 (52 int32 cells + the (256,8,96h) verify kernel-limit) |
| kda | gb200 / gb300 | sglang 0.5.16 | **done (2026-08-01)** | 976 + 976 | identical dispatch and failure spectrum to b300 (fused CuTeDSL verify, `kda_fused_decode` on the 12-head shard, 52 int32 cells + the (256,8,96h) verify kernel-limit cell) |
| kda | h100_sxm / h200_sxm (SM90) | sglang 0.5.16 | **done (2026-08-01)** | 1085 + 1085 | Hopper RECOLLECTION with the fixed int32 guard (396 context cells/kernel vs the old H20 220); Triton verify pair + `kda_fused_decode` engages on the 12-head shard on SM90 too |
| kda | rtx_pro_6000_server (SM120) | sglang 0.5.16 | **done (2026-08-01)** | 1085 | full grid clean; SM120 lifted from `unverified_sms` (probe job 381312864); Triton verify, `kda_fused_decode` engages |
| kda | l40s (SM89) | sglang 0.5.16 | **done (2026-08-01)** | see note | SM89 lifted from `unverified_sms` (probe job 381312863, 1074 rows); `kda_fused_decode` JIT fails ptxas below sm_90 (`mbarrier.try_wait.parity`) — fused attempt now gated to SM90+ in `collect_kda.py`; recollected at the fixed revision for the 12-head-shard generation rows |
| moe (K3 shape) | gb200 / gb300 | sglang 0.5.14 | **done (2026-08-01)** | 6156 + 6156 | both precision lanes (w4a16_mxfp4 + w4a8_mxfp4_mxfp8) in one run, zero failures, merged into the shared 0.5.14 tables |
| moe (K3 shape, **w4a8_mxfp4_mxfp8**) | b200_sxm (SM100) | sglang 0.5.14 | **done (2026-08-01)** | 3078 | closes the gap where the Blackwell resolver queries w4a8 but b200 carried only w4a16 — b200 sglang SILICON E2E now resolves; 8xB200, stock 0.5.14 image, zero failures (resumed across a node recycle), merged into the shared table (145,321 total) |
| moe (K3 shape, w4a16_mxfp4) | b200_sxm | vllm 0.1.dev19262 | **done (2026-08-01)** | 972 | first vllm K3 MoE data: checkpoint-truth CompressedTensors path, kernel_source `vllm_compressedtensorsw4a4mxfp4moe_marlin_marlinexperts` (Marlin weight-only + SiTU); merged into the vllm 0.24.0 table (61,485 total) per owner decision; 3 classified moe_tp=32 marlin tile-limit failures. Unblocks the first full-silicon b200 vllm-vs-sglang comparison (see pr_summary addendum) |
| moe (K3 shape) | h100_sxm / h200_sxm | sglang 0.5.14 | **done (2026-08-01)** | 541 / see note | marlin lane; the EP>1 IMA crash family reproduces on H100/H200 (2537/2587 cells — third system family, upstream issue still unfiled). H200's first run lost 47 measured rows to perf-log lock races during the crash storms (log_perf window widened to 30s + stale-lock break); recollected |
| moe (K3 shape) | l40s (SM89) | sglang 0.5.14 | **done (2026-08-01)** | 2997 | SM89 dispatches `sglang_fused_moe_triton` (no marlin/flashinfer lane); 81 OOM cells classified (48 GB card) |
| linear_attn_module | b200_sxm (SM100) | sglang 0.5.16 | **done (2026-08-01)** | 435 | second system for the experimental module lane (image sha256:6d9594a4 — tag drift recorded in meta); values track b300 closely (h12 bs1 34.4 vs 33.3 us); fused decode observed on the 12-head shard only; the b300 262144-token illegal-address band did NOT reproduce on SM100/this build; 52 int32 + 4 OOM (183 GB) + 1 h96 bs1024 classified. Also records that the b200 kda_perf generation rows predate the fused-decode collector branch (Triton pair only) — module rows carry the b200 fused truth |
| megamoe module (K3 shape) | b200_sxm (SM100) | sglang 0.5.16 | **done (2026-08-01)** | 64 | FIRST K3 MegaMoE data (ledger item 1 executed): DeepGEMM fp8_fp4_mega_moe + fused symm-buffer a2a, SiTU, EP8 torchrun on 8xB200 (image sha256:6d9594a4); 24 context + 40 generation x 4 routing distributions; small-token generation ~0.10-0.12 ms/layer (matches the GB300 DSV4 reference magnitude). SiTU sentinel verified live (clamp 10.0 vs 0.03125: -3.5% latency at 64 tok/rank, identical would mean silent swiglu). Import shim landed for the branch's renamed pre-dispatch module (e3f5440b) |
| kda | b300_sxm vllm / gb200 / gb300 / rtx / l40s / hopper | vllm | out of scope | — | owner scope decision 2026-08-01: sglang only. Full vllm kda artifacts for h100/h200/gb200 + limit-probe artifacts for l40s/b300/rtx were already collected before the scope cut (pipelines 60591225/60591349/60591439) and can be ingested later; 103/89/120 stay in the vllm `unverified_sms` |

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
  "fused_kda_decode_mtp_dspark"`, with the output gated RMSNorm ("onorm"
  in the kernel's argument naming) folded per-shard exactly as serving does.

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

### E2E validation on 8x B300 (dummy weights, TP8, 2026-07-29)

- bs1 decode step 11.56 ms; bs8 step 11.67 ms (latency-bound — 8 tokens/step
  nearly free); 5 accepted tokens/step => ~428 tok/s/user, bracketing the 423
  launch number.
- **Dummy-weight caveat**: random gate weights collapse routing to ~16
  distinct experts. With forced-uniform routing (AIC_RANDOM_ROUTING patch,
  ~119 distinct experts at 8 tokens x topk16) the bs8 step is **16.8 ms** —
  implied MoE ~8.0 ms/step, which matches the collected w4a8 lane (89.8
  us/layer x 92 = 8.26 ms) within 3%. The MoE model is validated; the
  earlier "3x MoE gap" was entirely the collapse artifact. Real power-law
  routing lands between the two bounds (297-428 tok/s/user at 5 tokens/step),
  consistent with the 423 blog number.
- Per-op nsys alignment (--cuda-graph-trace=node): projection GEMMs model
  +26% pessimistic; AR model -34% optimistic (serving uses the branch's
  k3_ar_fusion symm kernels with norm/residual folded in — an uncollected
  branch-only lane); fused KDA decode now collected (above); the remaining
  known double-count is kda_onorm (~0.2 ms/step empirical) for the 12-head
  shard where the fused kernel folds it — pending the op-graph fusion-boundary
  discussion.
  OWNER DECISION 2026-08-02: the k3_ar_fusion comm lane is a WON'T-DO —
  the AR term's ~0.9 ms/step optimism at bs8 is judged acceptable for the
  PR's purposes; the -34% figure above stays documented as a known model
  characteristic. rtx_pro_6000 K3 MoE and the Hopper EP>1 marlin issue are
  DEFERRED (not dropped): rtx waits for the next SM120 node; Hopper waits
  on the upstream marlin fix or issue filing.

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
   run on 1x B200. The `kimi_k3` model-config entry is already in
   `collect_dsv4_megamoe.py`; on an 8-GPU node inside the kimi-k3 branch
   image (`lmsysorg/sglang@sha256:4b8a7542...`, repo at /workspace):

   ```bash
   cd /workspace && export PYTHONPATH=/workspace
   torchrun --nproc_per_node=8 collector/sglang/collect_dsv4_megamoe.py \
     --model-config kimi_k3 --activation-clamp 0.03125 \
     --system-name b200_sxm --phases context,generation
   ```

   Verify first that the image's deep_gemm carries the SiTU patch (the
   sentinel silently selects plain swiglu on an unpatched kernel — compare
   one cell against `activation_clamp=10.0` and expect a latency/output
   difference). Perspective from the GB300 DSV4 megamoe table: at 8
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
5. **sglang `kda_fused_decode` decode kernel** — DONE (2026-07-29, campaign
   row above): covered()-driven generation dispatch in the collector, per-key
   consumer routing in Python+Rust, b300_sxm rows collected. Kept here for
   history; the remaining fused-lane gap is the kda_onorm double-count for
   the fused shard (fusion-boundary discussion pending).
6. **Module-level KDA collection (direction).** The kda collector is
   kernel-level by construction (it followed the GDN hybrid-linear precedent:
   novel state kernels in kda_perf, projections via the shared gemm table,
   norms via mem_op — collect_kda.py docstring). The B300 audit showed the
   exact failure mode the DSV4 attention-module design doc predicted for
   kernel-level collection of complex modules ("the isolated kernel workload
   diverges from the real module workload"): serving fuses across the op
   boundaries (kda_fused_decode folds conv+recurrence+onorm; k3_ar_fusion
   folds norm/residual into the AR), so the sum-of-kernels drifts from the
   layer truth at small batch. The end state is a
   `dsv4_*_module`-style collector: build the layer through the framework's
   own constructor (RadixLinearAttention + kda_backend), measure
   forward_extend/decode/target_verify per phase — dispatch fidelity by
   construction, nothing replicated. Interim mitigations already landed:
   covered()-driven fused-decode dispatch, fused-verify probe with source
   citations, per-key consumer routing. Full design (KDA+GDN unified module
   family, prefix-cache and CP semantics):
   docs/perf_database/linear-attention-module-design.md.

   **Prototype VALIDATED (2026-07-29, 1×B300 SM103, kimi-k3 branch image;
   script + raw results in collector_artifacts/proto_kda_module.py|
   proto_kda_module_results_b300.txt).** Shrunk 4-layer K3 (3 KDA + 1 MLA,
   per-rank TP8 shard realized at tp1: linear_attn num_heads 96→12 AND MLA
   num_attention_heads 96→12 — full 96 q-heads are rejected by the
   trtllm_mla decode kernel; num_experts 896→64; `is_kda_layer` uses
   1-indexed lists), built through sglang's own ModelRunner, batches via the
   `sglang.benchmark.one_batch` pattern. One KimiK3DeltaAttention module is
   timed by capture-and-replay: a forward-pre-hook grabs the layer's real
   (args, kwargs) AND the active ForwardContext during a full
   `model_runner.forward`, then replays the module under
   `forward_context(...)` — decode/graphed under the framework's own
   `model_capture_mode()` + torch.cuda.graph, mirroring serving. The fused
   CuTeDSL decode ENGAGED inside the replay (`_k3_onorm_consumed=True`) with
   zero dispatch logic replicated. Numbers vs the kernel-level table
   (12-head shard): decode bs1 34.5 µs, bs8 36.6 µs, bs64 66.0 µs per layer
   vs kda_fused_decode 6.7/6.7/17.0 µs — the module rows absorb the
   projection GEMMs, tiny bfa GEMVs and ghost elementwise the kernel-sum
   misses (the audit's "norm/elementwise 4x pessimistic" gap). Prefill bs1
   seq1024: 225 µs graphed (GPU truth; kernels 126 µs + projections ~99 µs)
   vs 742 µs eager — the eager number is python-launch-bound, so the
   collector must graph both phases (dsv4 pattern). Framework plumbing
   required outside the scheduler (0.5.16): `ParallelState.trivial()` for
   the ModelRunner ctor, `init_cuda_graphs()` even when graphs are disabled,
   and the scheduler-side global trio `initialize_moe_config` +
   `initialize_fp8_gemm_config` + `initialize_fp4_gemm_config` — without
   the first, the resolved flashinfer_mxfp4 MoE backend is silently lost and
   MoE falls back to the sm103-less triton runner
   (benchmark/one_batch.py:878 does the same). E2E cross-check: 36.6 µs ×
   69 KDA layers = 2.5 ms/step at bs8 vs ~2 ms KDA-attributed in the nsys
   breakdown — same zone, delta = intra-graph gaps + module-boundary
   elementwise now correctly owned by the module row.

   **Collector + dataset LANDED (2026-07-29):**
   `collector/sglang/collect_kda_module.py` (standalone, megamoe pattern,
   `PerfFile.LINEAR_ATTN_MODULE`, experimental — no SDK consumer yet) and
   `b300_sxm/kda/sglang/0.5.16/linear_attn_module_perf.parquet` — 435 rows
   over the four K3 shard geometries (96/tp heads for tp 1/2/4/8), grid =
   the declared cases/base_ops/kda.yaml sweeps. kernel_source is the
   OBSERVED path: fused decode engaged exactly on the 12-head shard (11
   rows), triton packed on 24/48/96 (32), chunk_kda context (392). Sanity:
   module−kernel delta = projection cost, scales with shard weights (h12
   bs1 33.3 vs 6.7 µs; h96 bs1 165.0 vs 12.2 µs) and stays flat bs1→bs8
   (weight-bound). Failures all classified: conv int32 guard raises
   (mirror the kernel lane), ONE fatal FIXME(kernel-limit) — the 262144-
   token context band on the 12-head shard dies with
   cudaErrorIllegalAddress below the conv int32 bound (culprit kernel in
   the full-layer path unidentified; generation runs first + guard cells
   first so only that band is lost) — and h96 generation bs=1024 (Triton
   invalid argument in the packed decode path, non-fatal). Decode-prep
   extends auto-shrink below the conv int32 bound (h48 bs1024 / h96 bs512
   died there before the fix). Verify (DSPARK) module phase deferred —
   kernel-level fused-verify rows + per-key routing cover it meanwhile.
   ENVIRONMENT DRIFT NOTE: collected from lmsysorg/sglang@sha256:81a9c006
   (re-pulled tag; reports 0.5.16 like the original 4b8a7542-digest image
   the kda_perf/E2E work used, but is a newer kimi-k3 branch build — its
   sgl JIT cache was ABI-incompatible with the old snapshot). Next: GDN
   module collector on the same harness, then SDK/Rust module ops
   (migration order in the design doc).
7. **Hopper system (if ever re-added)**: recollect sglang kda context with
   the fixed int32 guard rather than restoring the old dataset (coverage hole
   above), and re-run the moe marlin lane to reconfirm the EP>1 crash before
   filing upstream.
8. **SM120 (RTX Pro 6000)**: vllm FlashKDA claims SM120; sglang CuTe paths
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

## 2026-08-01 multi-arch sglang campaign (ada + hopper + blackwell)

Run via the aic-auto-collector GitLab harness (ref `campaign/rc20-restore`,
AIC revision = this branch), one pipeline per (framework image, op family):
kda 60591215 + probes 60591341, moe 60591217, reruns 60603368 (l40s kda at
the fixed revision) / 60603372 (h200 moe). Verified by artifact content, not
job status: rows per op, `errors_*.json` decomposition, duplicate-key scan,
device-name check, and `KDAKernel._query_kda_table` spot queries per system.

Collector fixes that came out of the campaign (commit 39a950b24):

1. **SM89 fused-decode gate** (`collect_kda.py`): the `kda_fused_decode` JIT
   kernel emits `mbarrier.try_wait.parity`, which ptxas rejects below sm_90.
   `covered()` accepts the shapes first, so on L40S the compile failure
   killed all 11 generation batch cells of the 12-head shard. The fused
   attempt is now `get_sm_version() >= 90`, mirroring serving's fallback to
   the Triton pair.
2. **log_perf lock hardening** (`helper.py`): the 1s lock window dropped 47
   measured H200 K3 moe rows while sibling workers were wedged in marlin IMA
   crash storms (two bursts, 07:42 and 08:07). Window widened to 30s and
   locks older than 60s (SIGKILLed owner skipped `finally`) are broken.

Registry: sglang kda `unverified_sms` is now `(80,)` — SM89/SM120 verified by
the probe runs above.

Open observation (pre-existing, NOT changed by this campaign): with the
experimental `linear_attn_module_perf` table marked `status: partial` in
b300_sxm's 0.5.16 collection_meta, `get_database("b300_sxm", "sglang",
"0.5.16")` fails with "marked incomplete in either layout" while every other
kda system (kda_perf-only meta) loads. If that is not intended, either the
loader's completeness rule or the b300 meta needs an owner decision.
RESOLVED 2026-08-01 (owner, commit d4ac8532): the meta was wrong, not the
loader — `partial` gates the whole version dir by design; the module table's
in-scope phases are complete (verify is deliberately out of scope), so the
status is now `complete` with the scope documented in the meta comment.

## vLLM K3 serving-truth audit + owner decisions — 2026-08-01

Audited in the manifest-pinned preview image
(`vllm/vllm-openai@sha256:e90e2603`, reports 0.1.dev19262+gb6bbf29dd),
cross-checked against the vLLM Day-0 posts
(vllm.ai/blog/2026-07-22-kimi-k3-preview, /2026-07-27-k3) and
recipes.vllm.ai/moonshotai/Kimi-K3. K3's checkpoint descriptor
(compressed-tensors mxfp4-pack: float/4bit/group32/symmetric, no
input_activations) resolves to `CompressedTensorsW4A4Mxfp4MoEMethod`; the
serving MoE path then splits by deployment tier:

1. **Single-node / TP (recipes carry no quantization override)**: SiTU is
   not in `CutlassExpertsMxfp4._supports_activation` (SILU/GELU/GELU_TANH/
   SWIGLUOAI only, cutlass_moe.py:321-327), so the method falls back to
   **MarlinExperts weight-only = W4A16 + SiTU**
   (compressed_tensors_moe_w4a4_mxfp4.py:58-71). This is TRUE ON EVERY SM —
   the CUTLASS W4A4 device gate covers capability families 100 (incl.
   SM103)/110/120; the block is the activation, not the GPU. → the
   kernel-level vllm MoE lane label `w4a16_mxfp4` is correct.
2. **Recommended DEP scale-out** (`--moe-backend deep_gemm_mega_moe` in the
   recipes): DeepGEMM MegaMoE (`KimiK3MegaMoEExperts`, fp8 activations x
   fp4 weights = W4A8 semantics, SiTU/EP/latent required) — but the pinned
   image DOES NOT SHIP `deep_gemm` (ModuleNotFoundError), so this path is
   unrunnable at the pin.
3. **TRTLLM-Gen mxfp4**: the kernel + explicit SiTU mapping exist in-image
   (`TrtLlmMxfp4ExpertsBase` maps situ beta/linear_beta to gatedActAlpha/
   Beta; flashinfer 0.6.15, same prerelease as sglang), but the
   compressed-tensors method never selects it (oracle
   `select_mxfp4_moe_backend` is consumed by the OCP/DSV4 methods only).
   The Day-0 deep-dive's "SiTU wired into TRTLLM-Gen" is not reachable for
   K3 at this pin.

Image drift: RETRACTED 2026-08-02 — the "drift to sha256:d61e062d" was
the manifest list's arm64 entry; the tag's amd64 image is byte-identical
to the e90e2603 pin (0.1.dev19262+gb6bbf29dd, still no deep_gemm). The
pin is current; the vllm MegaMoE W4A8 lane stays blocked on an upstream
image that ships deep_gemm.

**Owner decisions (2026-08-01):**
- Keep the vllm pin at e90e2603 for now — the W4A16 Marlin lane is
  build-invariant; a pin bump triggers the full kda dispatch re-audit and
  is deferred to the MegaMoE lane work.
- W4A8 scope is covered by the SGLANG trtllm-gen lane
  (`sglang_flashinfer_trtllm_moe`, collected on b200/b300/gb200/gb300);
  vllm W4A8 = future **vllm MegaMoE module lane** (needs the newer image
  with deep_gemm + 8-GPU EP; open item, peer of the sglang MegaMoE item).
- vllm kernel-level W4A16 lane proceeds: fix `collect_moe.py` to construct
  the checkpoint-truth `CompressedTensorsConfig` (mxfp4-pack descriptor)
  with SiTU instead of `Mxfp4Config()` (OCP path), run in the preview
  image, and merge rows incrementally into the vllm 0.24.0 moe table
  (rows keep their honest runtime version column; meta annotates).
- Lane order after that: B200 `linear_attn_module` (single GPU) → vllm K3
  MLA precise-case work → sglang MegaMoE module lane (8-GPU torchrun,
  first host that qualifies).

**vllm K3 MLA precise-case audit outcome (2026-08-01):** do NOT activate
`mla_*_module` for vllm — the SDK's K3 vllm path consumes the ATTENTION
(GQA) tables (`ops.ContextAttention`/`GenerationAttention` at the
per-rank 12q/12kv/128 shard) plus `MLABmm` from the mla_bmm tables, not
the MLA module tables (models/kimi_k3.py vllm branches); module rows
would have no consumer. The preview's custom MLA core does use vllm's
standard backend selector (models/kimi_k3/nvidia/mla.py:259-312 —
get_attn_backend/get_mla_prefill_backend), so attention-table pricing is
structurally right. The REAL precision gaps for the vllm column are:
(1) attention-table exact hits at the K3 shard geometries, and (2) the
vllm mla_bmm lane — every K3 prediction currently logs "Loading
low-fidelity fallback rows for mla_bmm_perf... from trtllm", i.e. the
absorb BMMs are priced from another backend's table. mla_bmm's generator
sweeps a global head grid with no model axis, so K3-exact coverage needs
a generator extension — a mechanism change parked for owner approval.

**2026-08-02 vllm precision items executed + one new finding:**
- mla_bmm: base grids gained the 96-family heads; NEW vllm mla_bmm
  collector + b200 table (636 rows, bf16 torch.bmm per NVIDIA dispatch) —
  the trtllm-fallback warning is gone; sglang b200 table gained the
  96-family rows (+424). Other systems queued.
- NEW FINDING (open, needs owner decision): the vllm DSPARK draft
  checkpoint (Inferact/Kimi-K3-DSpark) has DIFFERENT geometry from
  sglang's RadixArk draft — 64q/64kv MLA-style heads (qk_nope 128 +
  rope 64, v 128), inter 14336, 5 layers with target_layer_ids
  [2,23,47,71,89] — while the SDK's DRAFT_* constants encode the RadixArk
  GQA shape (64q/16kv/64hd) for BOTH backends. vllm DSPARK draft ops and
  draft-KV bytes are therefore mispriced (5 draft layers, small but
  real). Fix = backend-conditional draft geometry in models/kimi_k3.py
  (SDK scope). OWNER DECISION 2026-08-02: WON'T-FIX — the draft is 5
  layers vs the 93-layer target, percent-level error accepted; this note
  stays as the known-inaccuracy record for the vllm DSPARK column.

## vLLM wave-2 — 2026-08-02

Wave-1 vllm kda artifacts (pipelines 60591225/60591349/60591439) pulled
and verified by content: h100/h200/gb200 full + l40s/b300/rtx probe runs,
each 1203 rows (l40s 1145 — SM89 lane substitution: chunk_kda_with_fused_gate
prefill, no fused_kda_decode), zero bad latencies, zero duplicate keys,
device/version labels correct. The identical deterministic failure
spectrum on all six systems: 20 context cells at the causal_conv1d int32
token-offset overflow guard + 1 generation Triton bs=1024 grid limit.
`collector/vllm/collect_kda.py` is byte-identical between the collection
revision (72b6c0eeb) and current head, so these artifacts ARE current-code
data — no recollection needed.

Ingested: gb200 kda (b8f9ef27f, loader + KDAKernel spot queries verified).
Held local per the Blackwell-first deferral: h100/h200 (Hopper out of PR
scope), l40s/rtx (SM89/120 stay in vllm `unverified_sms`; probe evidence
archived).

Submitted (REF=campaign/rc20-restore, AIC_REVISION=b8f9ef27f, one
pipeline per image x op family):
- 60709585 vllm kda @ gb300 (preview image e90e2603) — last missing
  Blackwell kda system.
- 60709588 vllm moe, AIC_MODEL_FILTER=moonshotai/Kimi-K3 @ gb200,gb300
  (STOCK v0.24.0 digest-pinned image — first campaign run of the
  situ-as-silu Marlin lane + EP-local shim, 8555ac6a4/3004f56ca).
- 60709589 vllm mla_bmm_gen_pre,mla_bmm_gen_post @ b300_sxm,gb200,gb300
  (stock image; clears the "other systems queued" mla_bmm item — every
  K3 vllm prediction on those systems currently prices absorb BMMs from
  the trtllm fallback).
Stock v0.24.0 digest verified multi-arch (arm64+amd64) before submission.

Wave-2 COMPLETE same day (all six jobs finished in ~15 min wall,
verified by content, ingested at 21a03ecd8): gb300 kda 1203 rows;
gb200/gb300 K3 moe +972 rows each — first campaign run of the stock
situ-as-silu Marlin lane + EP-local shim, same 3 classified moe_tp=32
tile-limit failures as b200/b300; b300/gb200/gb300 mla_bmm 636 rows
each, zero failures. GPU allocation note: kda/mla_bmm jobs ran gres=gpu:1
and moe jobs gres=gpu:4 (gb tray) — all three op families are
single-device benchmarks (moe_tp/moe_ep are shard-geometry axes, EP
sharding is constructed locally by the shim), so GPU count affects
wall-clock only, not the numbers. Every Blackwell system now carries
vllm kda + K3 moe + mla_bmm; the vllm trtllm-fallback for MLA BMMs is
cleared on b200/b300/gb200/gb300.

2026-08-02 addendum (owner reopened vllm kda scope beyond Blackwell):
h100/h200/l40s/rtx wave-1 kda artifacts ingested at 88fbf9a9a — vllm kda
now covers all EIGHT systems; SM89/SM120 lifted from `unverified_sms`
(only SM80 remains, no probe hardware). The 2026-08-01 "sglang only"
kda scope row above is superseded. NOTE for CI: 5 pre-existing collector
unit-test failures on the branch (mla_bmm 400->600 case-count pin from
the 96-family grid extension, GLM model-set pins in sglang mla_module
tests, sm90 attention population pin 147->151) predate this ingest and
need their pins refreshed before the PR goes ready.

## PP scale-out data audit + Hopper vllm wave-3 — 2026-08-02

Owner question: with 16/32-GPU workers (tp/ep<=8 per node, PP across
nodes), is the DATA sufficient? Audit answer: PP itself needs no
collection (ops.P2P is analytic: inter_node_bw + p2p_latency); per-rank
op shapes do not change with pp. Task-API probe with
agg_num_gpu_candidates=[16,32] + pp[2,4,8] priced successfully from
existing silicon: b200 sglang 452 tok/s/gpu (32 GPUs, tp2 pp4 dp4
moe_ep4), h200 sglang 132 (moe_tp8 route — Hopper K3 moe ep 2..8 rows
are missing because 0.5.14 marlin EP>1 crashes on Hopper, the honest
serving gap), h100 sglang 73.5, b200 vllm 157. Also: gb200 fits at 8
GPUs once pp is enabled (tp2 pp4 = per-stage MLA KV; pp=1 OOMs by ~2
GiB) — the support-matrix FAILs on gb200/b200 are the pp=1 search-space
pin (agg_pp_candidates [1], should_enable_pp never set) plus the
worker<=8 domain cap, not data. SDK follow-ups (not this PR): open pp
search for capacity-bound models, worker>8 domain, 93%pp rounding.

Hard gap: Hopper vllm K3 moe rows = 0 (Blackwell-first scope cut).
Wave-3 submitted (REF=campaign/rc20-restore, AIC_REVISION=04036a56c,
stock v0.24.0 digest image): 60730461 moe K3 filter @ h100,h200
(situ-as-silu Marlin lane — watch whether vllm's marlin_moe_wna16 hits
the Hopper EP>1 crash the sglang kernel has; Blackwell collected EP up
to 128 cleanly) + 60730462 mla_bmm @ h100,h200 (clears trtllm
fallback).

Wave-3 COMPLETE same day (verified + ingested at b01050dc5): h100/h200
vllm K3 moe +972 rows each with FULL EP 1..128 — vllm's marlin_moe_wna16
does NOT have the Hopper EP>1 crash that blocks the sglang 0.5.14
marlin lane (kernel-stack difference, useful upstream data point);
h100/h200 mla_bmm 636 rows each, zero failures. Hopper vllm PP probe
now prices from silicon: h100 69.5 / h200 117.8 tok/s/gpu @32 GPUs.
K3 vllm data coverage is now identical in scope to sglang across all
eight systems (kda + K3 moe + mla_bmm); sglang's only remaining hole is
the Hopper ep2..8 moe cells (real serving crash, honest gap).

## vLLM upstream W4A8 status check — 2026-08-02 (GitHub research)

K3 merged to vllm main Jul 29-30 (#50089 + #50000; original #49999
closed for conflicts); NO tagged release contains it (v0.26.1rc0 cut
Jul 27, pre-merge). The MegaMoE W4A8 blocker is GONE on main: DeepGEMM
is vendored into the official image at build time
(vllm.third_party.deep_gemm, #41516) and the K3 merge pinned a
SITU-capable vllm-project/DeepGEMM fork (fp8_fp4_mega_moe kernels;
ep_gather fix #50458 Jul 31) — `--moe-backend deep_gemm_mega_moe`
should run on nightly images (nightly-* tags Jul 31 onward). The
kimi-k3 tag itself is unchanged (e90e2603, Jul 27). CUTLASS W4A4 SITU
whitelist still NOT extended → single-node path remains Marlin W4A16;
our w4a16_mxfp4 lane stays serving-true. New on main: FlashInfer
trtllm-gen mxfp4xmxfp8 SITU experts exist behind a flashinfer
capability check, but whether the compressed-tensors scheme can route
into them is undetermined (compressed_tensors_moe_w4a4_mxfp4.py still
hardcodes Cutlass/Marlin).

Implication: the deferred vllm MegaMoE W4A8 module lane is now
UNBLOCKED via nightly images — but running it means a pin bump, which
per the standing owner decision triggers the full kda dispatch
re-audit (nightly = merged K3, routing may differ from the preview
branch); also needs 8-GPU Blackwell EP. Scheduling is an owner call.

OWNER DECISION 2026-08-02 (scope cut): the experimental module-level KDA
lane is REMOVED from this PR entirely — collector
(collect_kda_module.py), PerfFile.LINEAR_ATTN_MODULE, unit test, design
doc, and both linear_attn_module_perf tables (b200/b300, 435 rows each,
meta entries stripped; loaders re-verified). Rationale: zero consumers
(the SDK PerfDataFilename enum never included the file — get_database
cannot even load it), the lane's diagnostic value is already banked in
the kernel tables (fused-dispatch generation replacement, projection-
delta validation) and in this ledger, and the PR is too large. The one
open item it leaves — the kda_onorm ~0.2 ms/step double-count on the
fused decode shard — is now tracked as issue #1463 (with recovery
pointers to the branch history for every deleted artifact). W4A8
MegaMoE follow-up is issue #1462.

rtx K3 moe DONE 2026-08-03 (the last deferred sglang data item): 551
w4a16_mxfp4 marlin rows merged (114,211 -> 114,762), collected on 8x
RTX PRO 6000 via the comm-op computelab routing (pipeline 60790079, 70
min wall vs the abandoned 1-GPU dlcluster run that burned 5 windows on
failure replay — lesson: high-failure grids go straight to 8-GPU).
SM120 confirmed as the FOURTH marlin EP>1 IMA family (2,527 cells vs
h100 2,537 / h200 2,587). The 1-GPU job was canceled after the 8-GPU
artifact verified. Upstream sglang issue for the marlin EP>1 crash is
still unfiled — now reproducible on four system families.

RE-BASELINE ITEM — RESOLVED same day (2026-08-03): there was no step
regression. The 2026-08-02 body line "implied spec step (17.7 ms)" was
an arithmetic slip — 2.94 ms x 6 with accepted=6, but the progress
convention is tokens/step = 1 + accepted, so the true old step was
2.94 x 7 = 20.6 ms. With correct arithmetic every reference number
moved exactly by the shared-expert-width fix (-3..4%): no-spec 14.37
-> 13.94, acc6 tpot 2.94 -> 2.83, acc7 404.3 tok/s/user, spec step
20.6 -> 19.8. Upstream #1410 was exonerated by direct experiment
(revert + python engine pinned: numbers identical). Consequently the
old "~5% above the 16.8 ms E2E" claim dies with the slip: the honest
deltas are +18% vs the dummy-weight E2E step (expected-fast per the
documented MoE routing-collapse artifact) and +4.7% vs the sglang
Day-0 blog's real-weight bs1 operating point (step 19.8 vs implied
18.9 ms) — the blog is the meaningful external anchor.

Fresh-context PR review (2026-08-03, background agent, full diff vs
rules): verdict structurally sound. Fixed same day: attention_op_keys
was backend-blind for KIMIK3 (vllm now falls through to the GQA keys —
it validates/resolves against the tables it actually queries); vllm
DSPARK draft geometry upgraded from the RadixArk approximation to the
real Inferact MLA-style geometry (latent projections + latent KV,
priced via the same MLA-as-attention convention as the main layers;
the former WON'T-FIX is closed). Remaining findings parked as
fast-follows: 96-family mla_bmm rows unreachable by the pow2 query
(RESOLVED 2026-08-03, see below), SM89 vllm
kda rows fall to SOL (no reverse alias), tp16/32 nearest-shard
unscaled fallback, vllm mla_bmm dtype filter belongs in YAML override,
resolve_kimi_k3_moe_arch_mode should match config not path string,
underived 232/128 literals in the gate GEMV chain, dead
sol_latency_ms in Rust, no-op tensors.clear() in vllm collect_kda,
zero-expanding trtllm quant lane needs a comment.

96-family mla_bmm rows RESOLVED 2026-08-03 (owner picked exact-first
routing): KimiK3Model now passes the exact local head count (96/48/24/12)
to MLABmm at count scale 1.0, and the mla_bmm query layer routes
exact-head-first with a data-presence fallback — when the exact slice has
no rows it reroutes to the next-pow2 DeepSeek slice scaled linearly by the
head ratio, arithmetically reproducing the old pow2+count-ratio modeling
(Python MLABmm._query_mla_bmm_table + Rust
operators/mla.rs::resolve_bmm_slice_heads, twin-commented). Effect: b200
sglang (the only system with exact rows today, +424) prices the absorb
BMMs from measured 96-family rows — the tp8 12-head shard is ~34% slower
than the old 16-head*0.75 linear scaling at small tokens (launch-overhead
regime the ratio model idealized away): agg tp8pp2 spec bs1 tpot 3.381 ->
3.391 ms, ttft unchanged. Every other system is numerically unchanged
(b300 agg spec bs1 tpot 3.296 ms bit-identical). Python-vs-Rust: b300
support-matrix --compare-engine-step-backends green; b200 verified by
running both engine-step backends directly (identical tpot/ttft) because
the b200 support-matrix cell fails on a PRE-EXISTING K3 memory-fit issue
(tp8 w4a8 weights exceed the 180 GiB modeled capacity; unrelated to this
change, fails identically on the parent commit).

Human review wave (2026-08-04, Arsene12358, 2 BLOCKER + 6 MAJOR — all
landed):
- Truncated digest RECOVERED: the second kimi-k3 build is the 2026-07-29
  tag re-push of lmsysorg/sglang,
  sha256:6d9594a421be244f2af29d726158ebffe9c3c2b3f39b5b89affd8150a106e187
  (branch commit c6ad1f26, build tag kimi-k3-c6ad1f26-20260729;
  recovered via the Docker Hub tag manifest).
  Recorded in the b200 kda + b200 MegaMoE metas; the two bare metas
  (b300 kda sglang, b200 kda vllm) upgraded to the structured format
  with honest "pre-provenance-writer, hashes not captured" notes.
- sglang stock-lane situ-as-silu marked IN DATA (mirrors the vllm
  standard): the K3 w4a16_mxfp4 rows on h100 (541) / h200 (488) / rtx
  (551, marlin) and l40s (2997, triton) now carry the _situ_as_silu
  kernel_source suffix. kernel_source verified NOT a slice key for these
  rows (Python load_moe_data routes only exact-match names:
  min_latency split + two DeepSeek-V4 quant remaps; Rust twin
  identical); loader spot-check bit-identical
  (h100 K3 tp8ep1 power_law_1.01 query 0.5942489624023437 before ==
  after, at BOTH 0.5.14 exact and 0.5.16 nearest-version resolution).
  op_kernel_source_manifest.yaml regen DEFERRED (owner decision needed):
  a fresh tool run produces ~1400 lines of tree-wide staleness churn —
  the checked-in manifest predates many lanes including the PR's own
  vllm _situ_as_silu rows — and the manifest is a live shared-layer
  inheritance whitelist, so a wholesale regen is its own
  behavior-changing pass. Interim state is conservative and matches the
  vllm precedent: suffixed (approximated) rows are NOT cross-backend
  inheritable; own-backend queries are unaffected (verified above).
  K3's sglang_moe_backends map gained explicit 89/120 keys with the
  branch citation (overrides.py:479-504 @ c6ad1f26 fires on SM100/103
  only). NOTE for future stock re-collections: the suffix must be
  re-applied until the sglang collector learns the checkpoint's situ
  property (the vllm collector already probes it).
- SM>=90 fused-decode gate REMOVED from the sglang kda collector: the
  serving chain has NO JIT-failure fallback (model stashes fused args on
  every NVIDIA GPU, kimi_k3.py:1563-1614 @ c6ad1f26; backend calls the
  kernel wherever covered() accepts with no try/except,
  linear/kda_backend.py:426-476), so the old "same as serving after its
  own JIT failure" claim was an invented fallback. SM89 12-head decode
  cells now raise classified (the crash serving hits). DATA IMPLICATION
  (owner decision pending): the shipped l40s kda 12-head generation
  Triton rows have no serving-truth backing — serving on SM89 crashes
  rather than reaching the Triton pair; future l40s kda runs will fail
  generation strict-completeness on those cells. Keep as interpolation
  support or prune — flagged to owner.
- vllm int32 conv guard: FIXME(kernel-limit) landed with the honest
  finding — the pinned preview source is not publicly addressable
  (commit b6bbf29dd absent upstream) and the era file at v0.24.0 uses
  int64 token strides (causal_conv1d.py:39,47), i.e. mainline vLLM
  establishes NO int32 limit; the nt*proj bound stays as an unverified
  conservative guard (in-band cells pass on silicon) pending an on-image
  probe at the next bump. Guard expression now AST-pinned in the vllm
  contract test.
- K3 trtllm moe zero-expansion fixed: moe activation moved to
  framework_specific_op_cases.{sglang,vllm} (a K3-scoped trtllm run
  plans no moe at all), the yaml trtllm allowlist is an explicit
  tombstone (allowed_modes: []) so cross-model trtllm sweeps cannot
  plan K3 under ungated base modes, and the trtllm getter now logs
  fully-dropped models like the vllm getter. Population test pins all
  three. This resolves the parked "zero-expanding trtllm quant lane"
  fast-follow.
- SHARED MECHANISM CHANGE (declared per layer_permissions meta-rule 2):
  helper.log_perf's stale-lock breaker is now rename-based
  (os.rename to a pid-suffixed name, unlink the renamed file) — the
  unlink-based breaker had a two-waiter race where the loser could
  unlink a FRESH lock and interleave two writers. 30s wait window and
  60s stale threshold unchanged; unit tests pin the stale-break and the
  loser-never-unlinks contract.
- test_sglang_attention_0514.py is now module-level pytest.mark.unit
  (the K3 attention-population pins were invisible to CI's -m gate);
  the vllm kda dispatch contract test is AST-based (name references,
  not substring greps).
