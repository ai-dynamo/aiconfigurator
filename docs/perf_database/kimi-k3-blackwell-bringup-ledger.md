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
| moe (K3 shape, **w4a8_mxfp4_mxfp8**) | b300_sxm (SM103) | sglang 0.5.14 | **done (2026-07-29)** | 3078 | the Blackwell serving-truth precision lane: mxfp8 activations (`Mxfp4MoEMethod` default, mxfp4.py:1311-1330 @ kimi-k3 branch); merged into the shared b300 0.5.14 table (251,441 total). Same-shape delta vs the bf16-activation w4a16 lane at 8 tokens: 89.8 vs 102.5 µs/layer (~12%) — small-batch MoE is weight-bytes-bound, so the E2E dummy-vs-model MoE gap is dominated by the dummy routing collapse artifact, not precision. SDK: `resolve_kimi_k3_moe_arch_mode` routes K3+sglang+Blackwell onto this label; Hopper keeps W4A16 marlin |
| kda | b300_sxm (SM103) | sglang 0.5.16 | **done (2026-07-29)** | 976 | SM103 verified and lifted from the sglang kda `unverified_sms`; includes the NEW `kda_fused_decode` generation lane for the TP8 12-head shard (attempt-and-verify fused conv+recurrence+onorm, 6.6 µs/layer @ 8 tokens vs 15 µs for the Triton pair — matches the E2E nsys 0.40 ms/step); same known failures as B200 (52 int32 cells + the (256,8,96h) verify kernel-limit) |
| kda | b300_sxm vllm / gb200 / gb300 | — | open | — | vllm kda on Blackwell-Ultra not yet run (103 stays in the vllm `unverified_sms`); gb200/gb300 open |
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
