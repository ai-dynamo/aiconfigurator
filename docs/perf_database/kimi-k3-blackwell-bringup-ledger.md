# Kimi-K3 bring-up ledger (final-state record)

Campaign record for bringing Kimi-K3 (`KimiK3ForConditionalGeneration`)
perf data into the database. Scope evolved by owner decision: Hopper
bring-up first (data later removed), Blackwell-only packaging
(2026-07-28), widened to ada+hopper+blackwell sglang (2026-08-01), vllm
reopened to all eight systems (2026-08-02 addendum). This document is the
restructured FINAL STATE (2026-08-04); the wave-by-wave narrative lives in
git history of this file and in the branch's commit messages.

## Status summary

All planned lanes are DONE across eight systems (b200_sxm, b300_sxm,
gb200, gb300, h100_sxm, h200_sxm, l40s, rtx_pro_6000_server):

- **kda** — sglang 0.5.16 (kimi-k3 branch) + vllm 0.1.dev19262 (kimi-k3
  preview) on all eight systems. sglang `unverified_sms` = (80,); vllm =
  (80,) (SM89/103/120 lifted by probe/full runs).
- **moe (K3 shape, 3584/3072, 896x16)** — sglang 0.5.14 on all eight
  (Blackwell both precision lanes; Hopper/rtx marlin; l40s triton) +
  vllm 0.24.0 situ-as-silu Marlin lane on all eight.
- **mla_bmm 96-family (96/48/24/12 heads)** — sglang b200 (+424 rows) +
  NEW vllm mla_bmm tables on b200/b300/gb200/gb300/h100/h200 (636 rows
  each).
- **MegaMoE module (K3 shape)** — b200 sglang 0.5.16, 64 rows (the DEP
  scale-out serving path).
- Support matrix carries 228 exact K3 rows (full key union at the merged
  head).

## Data inventory

| lane | system | backend/version | date | rows | notes |
|---|---|---|---|---|---|
| kda | b200_sxm | sglang 0.5.16 | 2026-07-28 (+gen recollect 08-02) | 987→1064-class | context/verify from build 4b8a7542; GENERATION replaced wholesale from build 6d9594a4 (77 rows, zero failures: `kda_fused_decode` on the 12-head shard 6.75 µs @bs1, Triton pair on 24/48/96 — the per-key fused routing requires the Triton key ABSENT for the covered shard, so append was insufficient) |
| kda | b200_sxm | vllm 0.1.dev19262 | 2026-07-28 | 1203 | context 428+428 (conv_qkv3 + flashkda_fwd), generation 44+44+43, verify 108+108; dispatch probes verified on SM100 |
| kda | b300_sxm | sglang 0.5.16 | 2026-07-29 | 976 | SM103 lifted; first fused-decode-collector lane; same failure spectrum as b200 (52 int32 cells + the (256,8,96h) verify kernel-limit) |
| kda | gb200 / gb300 | sglang 0.5.16 | 2026-08-01 | 976 + 976 | identical dispatch + failure spectrum to b300 |
| kda | h100_sxm / h200_sxm | sglang 0.5.16 | 2026-08-01 | 1085 + 1085 | Hopper RECOLLECTION with the fixed int32 guard (396 context cells/kernel vs the old H20 220); `kda_fused_decode` engages on SM90 too |
| kda | rtx_pro_6000_server | sglang 0.5.16 | 2026-08-01 | 1085 | full grid clean; SM120 lifted (probe job 381312864) |
| kda | l40s | sglang 0.5.16 | 2026-08-01, pruned 08-04 | 1074 | SM89 lifted (probe job 381312863); 22 h12 generation rows PRUNED 2026-08-04 (no serving-truth backing — see owner decisions); verify h12 rows kept (dispatch-gated Triton pair is genuine SM89 serving) |
| kda | h100/h200/gb200/gb300/l40s/b300/rtx | vllm 0.1.dev19262 | 2026-08-02 | 1203 each (l40s 1145) | wave-1/2 artifacts (pipelines 60591225/60591349/60591439, 60709585); l40s = SM89 lane substitution (chunk_kda_with_fused_gate prefill, no fused_kda_decode); identical deterministic failure spectrum on all systems: 20 context int32-guard cells + 1 generation Triton bs=1024 grid limit |
| moe K3 w4a16_mxfp4 | b200_sxm | sglang 0.5.14 | 2026-07-28 | 3078 | `sglang_flashinfer_trtllm_moe`, TP 1-32 x EP 1-128 x 3 distributions, zero failures |
| moe K3 w4a8_mxfp4_mxfp8 | b300_sxm / b200_sxm | sglang 0.5.14 | 2026-07-29 / 08-01 | 3078 + 3078 | Blackwell serving-truth precision (mxfp8 activations, `Mxfp4MoEMethod` default, mxfp4.py:1311-1330 @ kimi-k3 branch); same-shape delta vs w4a16 at 8 tokens: 89.8 vs 102.5 µs/layer (~12%) |
| moe K3 (both lanes) | gb200 / gb300 | sglang 0.5.14 | 2026-08-01 | 6156 + 6156 | zero failures |
| moe K3 marlin | h100 / h200 / rtx | sglang 0.5.14 | 2026-08-01 / 08-03 | 541 / 488 / 551 | EP>1 IMA crash family: h100 2,537 / h200 2,587 / rtx 2,527 cells (see findings); h200 union of two runs after the lock-race fix; rtx via computelab pipeline 60790079 (70 min on 8 GPUs — lesson: high-failure grids go straight to 8-GPU); rows relabeled `sglang_marlin_moe_situ_as_silu` 2026-08-04 |
| moe K3 triton | l40s | sglang 0.5.14 | 2026-08-01 | 2997 | SM89 Triton default; 81 OOM cells classified (48 GB); relabeled `sglang_fused_moe_triton_situ_as_silu` 2026-08-04 |
| moe K3 w4a16_mxfp4 | all eight | vllm 0.24.0 (stock) | 2026-08-01..02 | 972 each | checkpoint-truth CompressedTensors path; b200 preview-native rows `vllm_..._marlinexperts`; the other systems the stock situ-as-silu Marlin lane `..._situ_as_silu` + EP-local shim (8555ac6a4/3004f56ca); 3 classified moe_tp=32 marlin tile-limit failures per system; Hopper ran FULL EP 1..128 clean (finding below) |
| mla_bmm 96-family | b200 sglang; b200/b300/gb200/gb300/h100/h200 vllm | 0.5.14 / 0.24.0 | 2026-08-02 | +424; 636 each | NEW vllm mla_bmm collector (bf16 torch.bmm per NVIDIA dispatch); clears the trtllm low-fidelity fallback for K3 absorb BMMs |
| megamoe module (K3) | b200_sxm | sglang 0.5.16 | 2026-08-01 | 64 | DeepGEMM fp8_fp4_mega_moe + fused symm-buffer a2a, SiTU, EP8 torchrun (image 6d9594a4); 24 context + 40 generation x 4 distributions; SiTU sentinel VERIFIED live (clamp 10.0 vs 0.03125 shifts latency -3.5% @64 tok/rank) |

Removed data: Hopper bring-up datasets (2026-07-28 scope cut; the sglang
context rows had the loose-guard coverage hole — the 2026-08-01
recollection is authoritative); the experimental `linear_attn_module_perf`
tables (2026-08-02 scope cut, below); 22 l40s h12 decode rows (2026-08-04
prune). All retrievable from branch history.

## Owner decisions (dated)

- **2026-07-28 — Blackwell-first packaging.** Hopper bring-up data removed
  from the PR; any Hopper return must RECOLLECT sglang kda context with the
  fixed int32 guard rather than restore (the old data silently lost the
  96-head shard above batch=2/seq=16384).
- **2026-08-01 — multi-arch widening, sglang-only** (ada+hopper+blackwell);
  vllm stayed b200-only until the 2026-08-02 addendum reopened it (all
  eight systems ingested at 88fbf9a9a; SM89/120 lifted, only SM80 remains
  — no probe hardware).
- **2026-08-01 — vllm pin stays at e90e2603**: the W4A16 Marlin lane is
  build-invariant; a pin bump triggers the full kda dispatch re-audit and
  is deferred to the MegaMoE lane work.
- **2026-08-01 — W4A8 scope via sglang trtllm-gen lane**; vllm W4A8 =
  future vllm MegaMoE module lane (needs a deep_gemm-bearing image +
  8-GPU EP).
- **2026-08-01 — b300 partial-status resolution (commit d4ac8532)**: the
  meta was wrong, not the loader — `partial` gates the whole version dir
  by design; the module table's in-scope phases were complete, so status
  became `complete` with scope documented in the meta.
- **2026-08-01 — do NOT activate `mla_*_module` for vllm**: the SDK's K3
  vllm path consumes the ATTENTION (GQA) tables + `MLABmm`, not MLA
  module tables; module rows would have no consumer.
- **2026-08-02 — k3_ar_fusion comm lane is a WON'T-DO**: the AR term's
  ~0.9 ms/step optimism at bs8 (-34% on the AR op) is accepted for this
  PR and stays documented as a known model characteristic.
- **2026-08-02 — experimental module-level KDA lane REMOVED from the PR**
  (collector, PerfFile.LINEAR_ATTN_MODULE, tests, design doc, both 435-row
  tables). Rationale: zero consumers (the SDK enum never included the
  file), the diagnostic value is banked in the kernel tables and in this
  ledger (prototype findings below), and the PR was too large. Follow-ups:
  kda_onorm double-count = issue #1463; W4A8 MegaMoE = issue #1462.
- **2026-08-03 — exact-first mla_bmm routing** (owner picked routing over
  a comment fix): KimiK3Model passes exact local heads (96/48/24/12) at
  count scale 1.0; the query layer routes exact-head-first with a
  data-presence fallback to the next-pow2 slice scaled by the head ratio
  (arithmetically the legacy count-ratio modeling). Python
  `MLABmm._query_mla_bmm_table` + Rust
  `operators/mla.rs::resolve_bmm_slice_heads`, twin-commented.
- **2026-08-04 — situ-as-silu marked IN DATA** (review item): sglang
  stock-lane K3 rows suffixed `_situ_as_silu` (h100 541 / h200 488 / rtx
  551 marlin; l40s 2997 triton), mirroring the vllm standard.
- **2026-08-04 — kda absence-dependence treated as a one-off; hardened
  tool exclusion** (not the SDK fix): `generate_perf_data_reuse_manifest.py` carries the
  documented ABSENCE_LOAD_BEARING denylist (see invariants).
- **2026-08-04 — l40s h12 decode rows PRUNED** (22 rows, 1096→1074): no
  serving-truth backing (see findings); verify h12 rows kept
  (dispatch-gated). Future l40s kda runs raise classified on those decode
  cells — the honest state. SDK side effect: l40s h12 decode queries take
  the nearest-shard fallback (h24, unscaled).
- **2026-08-04 — GDN cross-backend donor fill PINNED** rather than
  excluded: donors only extend grids (zero pre-existing values changed);
  `tests/unit/sdk/database/test_gdn_donor_fill_pins.py` freezes one
  donor-only and one own-row coordinate per audited combo.
- **2026-08-04 — ledger restructured to this final-state record.**
- **2026-08-05 — Python↔Rust parity anchor landed (review Blocker 1, the
  last open item)**: three committed Kimi-K3 engine-step parity cases
  (b300 sglang 0.5.16 nospec, b200 vllm 0.1.dev19262 nospec, b300 sglang
  DSPARK nextn=7 — a new `nextn` field on EngineStepParityCase) plus an
  11-point Python-generated `query_kda` oracle block in
  `state_space.rs::state_space_queries_match_python_v2_engine` (context
  chunk_kda / generation kda_fused_decode / fused CuTeDSL verify; exact,
  interpolated and beyond-range each, 1e-9, shared_layer=False view).
  Rationale (reviewer's, endorsed): the donor-injection regression class
  shifts BOTH engines through the shared serialized source chain, so only
  committed anchors that freeze the K3 numbers themselves fail CI on a
  silent shift — the compare gate cannot.

## Findings

### Serving dispatch (verified against framework source in-container)

sglang kimi-k3 branch (build c6ad1f26 = image 6d9594a4; first build
74968e5653 = image 4b8a7542; both report 0.5.16):

- **prefill**: Triton `chunk_kda` (default `--linear-attn-backend triton`;
  CuTeDSL prefill exists but is opt-in).
- **decode**: attempt-and-verify fused `kda_fused_decode` wherever its
  `covered()` accepts (compiled for the TP8 12-head/128-dim shard), else
  the Triton packed pair. Serving has NO SM gate and NO JIT-failure
  fallback in this chain: the model stashes fused args on every NVIDIA GPU
  (srt/models/kimi_k3.py:1563-1614 — HIP + weight-layout checks only) and
  the backend calls the kernel with no try/except around the lazy JIT
  (srt/layers/attention/linear/kda_backend.py:426-476;
  kernels/ops/attention/kda_fused_decode.py:42-51). On SM89 the JIT dies
  in ptxas (`mbarrier.try_wait.parity` requires sm_90) — serving CRASHES
  rather than falling back, which is why the collector's former SM>=90
  gate was an invented fallback (removed 2026-08-04) and the l40s h12
  decode rows were pruned.
- **verify (DSPARK)**: fused CuTeDSL `fused_kda_decode_mtp_dspark` on
  SM100/103 only (`_can_run_dspark_cutedsl_mtp`: capability==10, cutlass
  importable, draft width 2..8, 128-dim symmetric bf16 heads); everywhere
  else the Triton pair IS the genuine serving path (a dispatch decision,
  not a crash). Gated RMSNorm folds into the fused kernel only on the
  12-head shard (`_prepare_fused_decode`).
- **moe**: `_kimi_k3_moe_runner_overrides` (arg_groups/overrides.py:479-504
  @ c6ad1f26) selects flashinfer_mxfp4 (trtllm-gen SiTU) on SM100/103 only
  and otherwise leaves auto → marlin W4A16 on SM90/SM120, Triton default on
  SM89 — matching the collected kernel_sources exactly.

vllm kimi-k3 preview (0.1.dev19262+gb6bbf29dd, image e90e2603): FlashKDA
prefill (`is_flashkda_supported`), `fused_kda_decode` decode,
`fused_recurrent_kda` chain verify — the collector resolves all of it
through the framework's own probes. K3's checkpoint descriptor
(compressed-tensors mxfp4-pack) resolves to
`CompressedTensorsW4A4Mxfp4MoEMethod`; SiTU is not in the CUTLASS
activation whitelist, so single-node serving falls back to **MarlinExperts
weight-only = W4A16 + SiTU on every SM** (the block is the activation, not
the GPU) — the `w4a16_mxfp4` lane label is serving-true. The recommended
DEP scale-out (`deep_gemm_mega_moe`) is unrunnable at the pin (no
deep_gemm in-image); TRTLLM-Gen mxfp4 SiTU exists in-image but the
compressed-tensors method never selects it. The vllm DSPARK draft is the
Inferact checkpoint (64q/64kv MLA-style, qk_nope 128 + rope 64, v 128,
inter 14336, 5 layers) — different geometry from sglang's RadixArk GQA
draft; the SDK models both backend-conditionally (the initial WON'T-FIX
was superseded by the 2026-08-03 fix).

### Kernel and collector findings (upstream-relevant)

- **sglang 0.5.14 marlin MoE EP>1 IMA — FOUR system families**: h100
  2,537 / h200 2,587 / rtx(SM120) 2,527 crash cells (l40s routes to
  Triton, unaffected; Blackwell routes to flashinfer, unaffected).
  Upstream issue still unfiled. Notably vllm 0.24's `marlin_moe_wna16` ran
  FULL EP 1..128 clean on the same Hopper hardware — a kernel-stack
  difference and a useful upstream data point.
- **sglang Triton conv int32 offset limit (silicon-proven)**: the real
  bound is `total_tokens * conv_channels` (the per-block conv views stride
  the whole 3-block mixed_qkv buffer; causal_conv1d_triton.py:373-379) —
  the original per-block `* proj_size` bound was 3x loose and cells in
  `[2**31, 3*2**31)` IMA'd and poisoned the CUDA context. Every packaged
  dataset uses the fixed guard (52 context guard cells per sglang system,
  by design).
- **vllm int32 guard is UNVERIFIABLE at the pin**: the preview source is
  not publicly addressable (commit b6bbf29dd absent upstream) and the
  v0.24.0-era `causal_conv1d.py` uses int64 token strides (:39,:47) —
  mainline vLLM establishes NO int32 limit. The `nt * proj` bound stays as
  a documented unverified conservative guard (identical 20-context +
  1-generation guard spectrum on all eight systems, in-band cells pass on
  silicon, 2026-08-01); re-verify or delete at the next preview bump.
  Guard expression AST-pinned in the vllm contract test.
- **sglang fused verify kernel-limit**: single cell (batch=256, draft=8,
  96 heads) dies with cudaErrorIllegalAddress on every SM100/103 system;
  every smaller cell passes. Suspected per-SM resource growth in the
  persistent CuTe kernel. FIXME(kernel-limit) at the call site; not filed
  upstream.
- **vllm packed decode bs=1024 x 96 heads**: single-cell Triton grid-limit
  failure, identical signature on every platform since Hopper bring-up.
- **Module-vs-kernel gap (banked from the removed module lane)**: the B300
  prototype (2026-07-29, capture-and-replay through sglang's own
  ModelRunner; script archived in branch history) measured the 12-head
  shard at decode bs1 34.5 / bs8 36.6 / bs64 66.0 µs per layer vs
  kda_fused_decode kernel rows 6.7/6.7/17.0 µs — the delta is projection
  GEMMs + tiny GEMVs + boundary elementwise the kernel-sum misses, flat
  bs1→bs8 (weight-bound). Prefill must be CUDA-graphed (225 µs graphed vs
  742 µs eager, python-launch-bound). One unresolved fatal: the
  262144-token context band on the 12-head shard IMA'd below the conv
  int32 bound (culprit unidentified). E2E cross-check: 36.6 µs x 69 layers
  = 2.5 ms/step at bs8 vs ~2 ms KDA-attributed in nsys.
- **B200-vs-Hopper scaling sanity**: shared-cell latency ratio median
  1.47x (sglang) / 1.49x (vllm), consistent with the HBM bandwidth step;
  FlashKDA prefill up to ~2.7x.
- **log_perf lock-storm loss (mechanism change, declared per
  layer_permissions meta-rule 2)**: the 1s lock window dropped 47 measured
  H200 K3 moe rows during marlin IMA crash storms; widened to 30s with
  stale-lock breaking (60s). The breaker is rename-based (single-winner)
  since 2026-08-04 — the unlink-based breaker had a two-waiter race where
  the loser could unlink a FRESH lock and interleave two writers; unit
  tests pin both contracts.

### Modeling and E2E findings

- **Fused-dataset routing (Python + Rust twins)**: datasets with fused
  CuTeDSL verify rows and no Triton verify rows route the recurrence op
  onto the fused table and fold the conv op to 0; the fused SOL byte model
  equals conv + recurrence (asserted on both sides). Same per-key pattern
  for `kda_fused_decode` generation rows. Packaged b200 sanity: sglang
  verify recurrence 0.0640 ms silicon + conv 0.0; vllm 0.0504 + 0.0065.
- **Exact-first mla_bmm effect**: b200 sglang prices absorb BMMs from
  measured 96-family rows — the tp8 12-head slice is ~34% above the old
  16-head*0.75 linear scaling at small tokens (launch-overhead regime);
  agg tp8pp2 spec bs1 tpot 3.381 → 3.391 ms. Every other system
  bit-identical (b300 3.296000). The b200 support-matrix compare cell
  fails on a PRE-EXISTING memory-fit issue (tp8 w4a8 weights exceed the
  180 GiB modeled capacity), so b200 parity was verified by running both
  engine-step backends directly.
- **E2E anchors (8x B300 TP8, dummy weights; corrected 2026-08-03)**:
  no-spec step 13.94 ms; spec acc6 tpot 2.83 ms (step 19.8 ms); acc7
  404.3 tok/s/user. The earlier "17.7 ms implied step" was an arithmetic
  slip (tokens/step = 1 + accepted). Honest deltas: +18% vs the
  dummy-weight E2E step (expected-fast — random gate weights collapse
  routing to ~16 experts; forced-uniform routing validated the MoE model
  within 3%) and **+4.7% vs the sglang Day-0 blog's real-weight bs1 point**
  (19.8 vs implied 18.9 ms) — the blog is the meaningful external anchor.
  Known model characteristics: projection GEMMs +26% pessimistic, AR -34%
  optimistic (k3_ar_fusion WON'T-DO), kda_onorm ~0.2 ms/step double-count
  on the fused shard (#1463).
- **PP scale-out needs no collection**: ops.P2P is analytic; per-rank op
  shapes don't change with pp. 16/32-GPU probes priced from existing
  silicon (b200 sglang 452 tok/s/gpu @32 GPUs tp2 pp4 dp4 moe_ep4; h200
  132; h100 73.5; b200 vllm 157; post-wave-3 h100 vllm 69.5 / h200 117.8).
  gb200 fits at 8 GPUs once pp>1 is allowed (pp=1 OOMs by ~2 GiB) — the
  support-matrix FAILs there are the pp=1 search-space pin, not data. SDK
  follow-ups (not this PR): open pp search for capacity-bound models,
  worker>8 domain, 93%pp rounding.
- **GLM-5 sglang 0.5.9 rank-local flip (upstream-relevant)**: after
  upstream #1431 (rank-local num_heads) the 0.5.9-era b200-family GLM
  dsa-module slices are unreachable (b300's 0.5.9 lane still passes;
  #1460 does not restore them) — the regenerated support matrix records
  FAIL and the b200 pin was flipped with provenance. Upstream
  data-migration gap.

## Invariants

- **ABSENCE_LOAD_BEARING (manifest tool denylist + graduation criterion).**
  `kda_perf: causal_conv1d_update, fused_recurrent_kda_packed_decode` are
  emitted `tier: absence_load_bearing` (loader-inert): the SDK's per-key
  fused-decode routing REQUIRES those keys absent for the covered shard,
  and a cross-backend fill of vllm's genuine 12-head Triton rows defeated
  the reroute (measured: b300 K3 agg spec bs1 tpot 3.296 → 3.336 ms; the
  Python-vs-Rust compare gate cannot catch this class — Rust consumes
  Python's serialized source chain, both engines shift together). GDN was
  audited and left inheritable (nearest-shard fallbacks only, no
  absence-keyed routing). The donor-injection regression test
  (`tests/unit/sdk/operations/test_kda_donor_injection.py`) builds the
  manifest THROUGH the generator and fails if the exclusion is removed.
  **GRADUATION CRITERION**: this denylist is a one-off — a second
  absence-dependent kernel family means promoting the SDK to a
  primary-only routing view, not extending the list.
- **situ-as-silu standard.** A silu-epilogue row labelled situ is silently
  wrong data. Any lane that benchmarks the silu epilogue for a SiTU
  checkpoint carries the `_situ_as_silu` kernel_source suffix (vllm stock
  Marlin lane by collector probe + hard check; sglang stock lanes suffixed
  in data 2026-08-04). Suffixed rows are single-producer in the manifest
  (never cross-backend inheritable). kernel_source is NOT a slice key for
  these rows (verified Python+Rust; loader spot-check bit-identical,
  0.5942489624023437 at both 0.5.14-exact and 0.5.16-nearest resolution).
  Re-collections on stock images must re-apply the suffix until the sglang
  collector learns the checkpoint's situ property. Blackwell trtllm-gen
  lanes are unaffected (same kernel, epilogue constants; epilogue delta
  measured up to ~5.5% at small token counts on the vllm side).
- **Mixed-generation support-matrix lesson.** Matrix rows generated from
  different tree states cannot satisfy union-coverage tests: rows appended
  from a pre-merge tree introduced version keys other models' rows predate
  (and vice versa). Refresh affected models AT THE MERGED HEAD via
  per-(model,system) sharded runs into temp dirs (the tool's
  directory-save mode deletes all CSVs) and replace rows in place. The
  union is only as consistent as its most recently generated member.
- **Quant-distinct artifacts are separate rows.** A future W4A8 K3
  checkpoint gets its own `model_case_values` moe row + allowed_modes —
  never merged into the w4a16_mxfp4 row. K3 rows for another artifact
  MERGE into the shared 0.5.14 table; never overwrite.

## Open items

- **vllm MegaMoE W4A8 module lane (issue #1462)**: unblocked upstream —
  K3 merged to vllm main Jul 29-30 (#50089/#50000) with a SITU-capable
  DeepGEMM fork vendored into nightly images; no tagged release contains
  it. Running it means a pin bump → full kda dispatch re-audit (standing
  owner decision) + 8-GPU Blackwell EP. CUTLASS W4A4 SITU whitelist still
  not extended, so the single-node lane stays Marlin W4A16 (serving-true).
- **kda_onorm ~0.2 ms/step double-count** on the fused decode shard
  (issue #1463; fusion-boundary discussion).
- **Upstream filings**: sglang marlin EP>1 IMA (four families,
  reproducible); sglang fused-verify (256,8,96h) kernel-limit; GLM 0.5.9
  rank-local migration gap.
- **Parked review fast-follows**: SM89 vllm kda rows fall to SOL (no
  reverse alias); tp16/32 nearest-shard unscaled fallback; vllm mla_bmm
  dtype filter belongs in a YAML override;
  `resolve_kimi_k3_moe_arch_mode` should match config not path string;
  underived 232/128 literals in the gate GEMV chain; zero-expanding
  trtllm quant lane (RESOLVED 2026-08-04: per-framework activation +
  tombstone + logged drops); no-op tensors.clear (RESOLVED); dead Rust
  sol_latency_ms (RESOLVED in the parity work).
- **SM80**: remains in `unverified_sms` for kda both backends (no probe
  hardware).

## Appendix: provenance identifiers

Images / builds:
- sglang kimi-k3 build 1 (context/verify + most kda lanes; manifest pin):
  `lmsysorg/sglang@sha256:4b8a7542e00a4d82801d1276108438b024137c9a07f33215d03b04003e500f2b`
  (tag kimi-k3-cu12, 2026-07-27, git 74968e5653).
- sglang kimi-k3 build 2 (b200 kda generation recollect, MegaMoE,
  linear_attn_module): `lmsysorg/sglang@sha256:6d9594a421be244f2af29d726158ebffe9c3c2b3f39b5b89affd8150a106e187`
  (kimi-k3 tag re-push 2026-07-29, branch commit c6ad1f26, build tag
  kimi-k3-c6ad1f26-20260729; full digest recovered from the Docker Hub tag
  manifest after a truncated-digest review finding).
- vllm kimi-k3 preview (all vllm kda lanes; manifest families.kda pin):
  `vllm/vllm-openai@sha256:e90e2603b2781936651ba019804137714367c69e10a7b25a2e57b46995225616`
  (0.1.dev19262+gb6bbf29dd, CUDA 13; the once-suspected drift to
  sha256:d61e062d was the manifest list's arm64 entry — RETRACTED).
- vllm moe/mla_bmm lanes: stock v0.24.0 digest-pinned image (multi-arch
  verified).

Pipelines / jobs: sglang campaign kda 60591215 (+probes 60591341, jobs
381312863/381312864), moe 60591217, reruns 60603368/60603372; vllm wave-1
kda 60591225/60591349/60591439; wave-2 60709585 (gb300 kda), 60709588
(gb200/gb300 moe), 60709589 (b300/gb200/gb300 mla_bmm); wave-3 60730461
(h100/h200 moe), 60730462 (h100/h200 mla_bmm); rtx moe 60790079
(job 382547851).

Key commits: collector fixes 39a950b24 (SM89 gate — later removed —
+ lock hardening); wave ingests b8f9ef27f / 21a03ecd8 / b01050dc5 /
88fbf9a9a / 04036a56c; meta fix d4ac8532; module-lane import shim
e3f5440b; b300 meta collector_ref 72b6c0eeb (kda campaign;
case_plan a85d5a96).

Runbook gotchas that will recur on new nodes:
- NFS root-squash: run as the host uid and export HOME/USER/LOGNAME +
  TORCHINDUCTOR/TRITON/XDG/CUTE_DSL cache dirs to a writable tmpdir — an
  unmapped uid breaks `getpass.getuser()` in torch inductor instantly.
- The CuTe DSL verify kernel compiles per (heads, batch, draft) tuple
  (~25 s/shard-case); persist CUTE_DSL_CACHE_DIR between smoke and full.
- Delete stale `kda_perf.txt` between backend runs — `log_perf` appends
  and a mixed-backend staging file poisons the parquet.
