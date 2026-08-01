# Design: Linear-Attention Module Collection (KDA + GDN)

Status: PROPOSED (owner-requested 2026-07-29; B300 audit motivation recorded
in the Kimi-K3 bring-up ledger, pre-work item 6).

## Motivation

KDA and GDN are collected kernel-level (novel state kernels in
`kda_perf`/`gdn_perf`, projections via the shared gemm table, norms via
mem_op). The B300 E2E audit hit the failure mode the DSV4 attention-module
design predicted for kernel-level collection of complex modules: serving
fuses across the op boundaries —

- `kda_fused_decode` folds conv + recurrence + gated RMSNorm into one launch
  (TP8 12-head shard);
- `k3_ar_fusion` folds the next layer's norm and the residual into the
  all-reduce;
- the CuTeDSL DSPARK verify kernel folds the conv and (per-shard) the onorm —

so the sum-of-kernels drifts from the layer truth at small batch, and every
new fusion needs a collector dispatch branch plus consumer routing. The
per-key routing and covered()-driven dispatch landed in the bring-up are the
interim mitigation; the end state is module-level measurement, mirroring
`dsv4_*_module`.

## Shape of the design

One module family covers both linear-attention architectures — KDA (Kimi-K3 /
KimiLinear) and GDN (Qwen3.5 / NemotronH hybrids) — because they are the same
layer skeleton (input norm -> in_proj -> causal conv -> gated delta-rule
recurrence -> output norm -> out_proj [+ AR under TP]) with different gate
ranks and kernel families. `KDAKernel` already extends `GDNKernel` on the SDK
side; the module table keeps that symmetry.

- **Collector**: build the layer through the framework's own constructor
  (`RadixLinearAttention` + the KDA/GDN attention backend + MambaPool +
  ForwardBatch metadata), following the `collect_dsv4_attn.py` pattern.
  Measure `forward_extend` / `forward_decode` / `_forward_target_verify` per
  phase. Dispatch fidelity comes from construction — no shape or SM logic is
  replicated; whatever the backend picks (fused decode, CuTe verify, Triton
  fallbacks) is what the row records.
- **Boundary**: gate/topk-free by construction (linear attention has no
  router); include everything from the post-norm input projection to the
  out_proj output (the module owns its norms, conv, recurrence, onorm and
  projections). Under TP the AR stays OUTSIDE the module (it belongs to the
  k3_ar_fusion comm lane, collected separately) — document per row whether
  the framework folded norm/residual work into the AR so the comm lane and
  the module lane never double-count.
- **Table**: `linear_attn_module_perf` — phase (context/generation/verify),
  batch, seq (draft width for verify), the per-shard head geometry key
  (num_k_heads, head_k_dim, num_v_heads, head_v_dim, d_conv, gate rank),
  model key, kernel_source recording the backend's actually-selected path.
- **SDK/Rust**: one module op per phase replacing today's 9-op decomposition
  for models that have module rows (data-presence routing, same pattern as
  the fused-decode per-key routing); kernel-level tables remain for
  interpolation support and older datasets.

## Prefix cache (the "KV hit" question)

Linear attention has no paged KV; the per-request state is O(1) (conv window
+ SSM state). The serving branch implements radix/prefix caching via
**state checkpoints**: a prefix hit COW-copies the nearest valid checkpoint
(`MambaPool.copy_from`; Snapshot/Donate on the write side) and replays the
tail tokens from that boundary. A token-prefix match alone is NOT
recoverable — KDA state cannot be truncated backward, so the hit floors to
the nearest checkpoint at or before the match point, and the MLA KV beyond
that boundary does not save the replay (the tail forward must run all
layers to rebuild KDA state).

Checkpoint placement (per the Day-0 serving write-ups — sglang: prefill
CHUNK boundaries + a fixed decode interval + radix fork nodes, with a
per-path count budget, LRU eviction and an optional INT8-compressed cache
pool; vllm: `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` + prompt-end states +
Marconi-style cache-on-second-hit; overview:
https://mp.weixin.qq.com/s/Yxmt-Foq2D7b46sYOk7WAg) is therefore SPARSE and
adaptive, and the replay tail is bounded by the local checkpoint spacing —
up to `chunked_prefill_size` (8-16k tokens) for prefill-built prefixes,
`mamba_track_interval` for decode-built ones, and unbounded when eviction
removed the boundary. The Mooncake post's worked example is the first-order
effect: 15k shared / 10k spacing → only 10k reusable, recompute doubles.

Modeling consequence: a prefix hit runs the module context query at
`seq = isl - checkpoint_floor(prefix_hit)`, NOT `isl - prefix_hit`. The
module design still needs no extra table axis; it is a query-time seq
adjustment on the consumer side — but when the CLI `--prefix` input is
wired for K3, the effective prefix must floor to a checkpoint-spacing
parameter (backend-dependent as above) instead of using the raw match
length.

### Capacity side (memory model, `models/kimi_k3.py`)

One full-model KDA state copy is 96h x 128 x 128 x fp32 x 69 layers
~ 0.4 GiB (matches the kvcache.ai calculator figure cited by the Mooncake
post), TP-sharded. The current SDK model charges
`KDA_STATE_SLOTS_PER_REQUEST = 5` state slots per admitted request and
treats MLA KV + KDA state as one elastic byte budget. Owner decision
2026-07-31: keep both as-is and ANNOTATE the behavior in the code (the KDA
term is secondary — at TP8, 5 slots = 270 MB/request vs ~1.8 GB/request of
MLA KV at 128k context), rather than modeling the split. The two known
deviations, recorded at the charging sites in `models/kimi_k3.py`:

1. The flat per-request slot multiplier folds the radix checkpoint budget
   into admission cost. Serving actually splits an ACTIVE pool
   (1 running slot + snapshot double-buffer per request) from a global
   checkpoint cache pool with its own budget/LRU — and the optional INT8
   compressed pool stores inactive temporal states 4x denser. The flat 5x
   over-charges short-context high-concurrency mixes.
2. The single elastic budget matches sglang's opt-in
   `--enable-unified-memory` mode. The DEFAULT is two separately sized
   pools (MLA KV vs KDA state) with a boot-time ratio; under a mismatched
   workload the binding pool saturates while the other idles, so real
   capacity can be below the elastic estimate.

## Context parallelism

The linear-attention backends carry NO context-parallel path (the recurrence
is sequential over the sequence; CP would need cross-rank state handoff,
unimplemented on the kimi-k3 branch), and `KimiK3Model` raises for
`cp_size > 1` — the model and the framework agree. sglang's DCP
(decode context parallel) shards the paged MLA KV only; KDA layers keep
their O(1) state per rank and their cost is DCP-invariant. Supporting the
cookbook Balanced/DCP recipe is therefore MLA-side modeling work; the
linear-attention module needs no CP axis.

## Migration order

1. sglang KDA module collector (K3 shapes, B300/B200) — validates the
   pattern where the fusion drift is largest.
2. GDN module collector on the same harness (Qwen3.5/NemotronH shapes).
3. SDK/Rust module ops + data-presence routing; retire the per-key fused
   routing once module rows exist for the shipped systems.
4. vllm variants (vllm already serves KDA decode fused via its own probes;
   its kernel-level rows are closer to layer truth, so it migrates last).
