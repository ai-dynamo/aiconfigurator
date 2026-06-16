# Context-Parallel (CP) Modeling for DSA Prefill

How AIC estimates **one decoder layer of GLM-5 DSA (DeepSeek Sparse Attention)
prefill under Context Parallelism** — e.g. GLM-5-NVFP4, `attn_cp_size = 8`,
`attn_dp = 1`, `attn_tp = 1`, isl = 32768.

A CP layer is modeled as **three serial parts**:

```text
per_layer = DSA  +  MoE  +  Comm
```

each built from the single-card collected data plus mechanism-driven
corrections. This doc derives the three parts, pins the precision/base
selection (a real pitfall — see §4), validates against a captured nsys
timeline, and specifies the AIC implementation.

---

## 1. What CP does (SGLang mechanism)

For a prefill of `isl` tokens with `attn_cp_size = cp` (`attn_dp=1` ⇒
`attn_tp=1`, full CP / no head split):

* **Token split = round-robin.** Card `c` owns global positions
  `{c, c+cp, c+2cp, …}` — a **uniform 1/cp sample** of all `isl` positions
  (kernel `nsa_cp_round_robin_split_q_seqs`). Each card holds `M = isl/cp`
  query tokens.
* **KV all-gathered before attention** → every card sees the **full KV**; a
  query at global position `p` attends to / selects from its full causal
  context `[0,p]`.
* **MoE** runs on the **full token set** (TP-sharded), reusing the CP group as
  the MoE-TP group (`_MOE_DP = _ATTN_CP`). The attention output is
  all-gathered to full, MoE-TP computes, and the result is reduce-scattered
  back to the per-card tokens.

**One layer, one ReduceScatter.** Per-layer timeline (one RS→RS period):

```text
… MoE → RS │ AG_KV → mqa → topk → AG_LSE → fmha → AG_hidden │ MoE → RS │ …
            └──────────────── DSA ────────────────┘└ Comm-in┘└MoE┘└Comm-out┘
```

`RS` is the single per-layer collective (after MoE). It does **two jobs at
once**: TP-reduce the MoE partials **and** SP-scatter back to per-card tokens.

---

## 2. The model — three parts

```text
DSA  = dsa_context_module(isl/cp, prefix=0)        # base, per-card token count
     + mqa_correction
     + topk_correction
     + AG_KV + AG_LSE                              # the two SMALL all-gathers
#      ( fmha, projections, RMSNorm: NO correction )

MoE  = moe(isl, moe_tp)                            # full token set, MoE-TP

Comm = AG_hidden + RS                              # the BIG comms around MoE
```

`per_layer = DSA + MoE + Comm` (serial — see §7).

Why only mqa / topk move inside DSA:

| sub-op | context-scaling | CP treatment |
|--------|-----------------|--------------|
| q/kv proj, o_proj, RMSNorm | none (per-token) | base, unchanged |
| **indexer mqa (logits)** | ∝ context² (quadratic) | scale up (`×cp`) |
| **topk selection** | ∝ context¹·⁶ (sub-quad) + data-dependent | `full/cp`, flat→top_last |
| sparse FMHA | none (capped at `index_topk` keys) | base, unchanged |
| AG_KV, AG_LSE | n/a (CP attention comm) | add (small) |

---

## 3. Deriving the DSA corrections

### 3.1 mqa — quadratic ⇒ `× cp` on the small kernel

mqa cost ∝ Σ_card causal context = `(1/cp)·full_mqa`. Because mqa ∝ `isl²`:
`mqa(isl/cp,p0)·cp = (isl/cp)²/2·cp = isl²/(2cp) = full_mqa/cp`.

Use the **`×cp` form** (not `full/cp`): it carries the **small-kernel
(`isl/cp`-query) efficiency** that the per-card kernel actually has.

```text
mqa_correction = mqa(isl/cp, prefix=0) · (cp − 1)
```
*isl=32768,cp=8:* 45.1·7 = **+316us**; per-card mqa = 360 vs timeline 374 (−4%).
(`full/cp` = 2558/8 = 320 → −14%, worse — confirms the small-kernel point.)

### 3.2 topk — sub-quadratic ⇒ `full/cp`, then flat→top_last

topk is **per-token independent** ⇒ round-robin (uniform 1/cp of tokens) gives
the per-card cost **exactly** `topk_full(isl)/cp`. It does **not** obey the mqa
`×cp` identity (sub-quadratic) — do **not** use `topk(isl/cp)·cp`.

The `dsa_context_module` runs **dummy weights**, so its internal topk sees
**flat** scores (the degenerate worst-case anchor). Real runtime is the
**top_last** distribution. Correct flat→top_last, both at `full/cp`:

```text
topk_correction = − [ topk_full(flat) − topk_full(top_last) ] / cp
```
*isl=32768,cp=8:* −(3383 − 1520.6)/8 = **−233us**; per-card topk = `full(top_last)/cp` = 190.

> There is **no separate `isl/cp → full` context scale-up** for topk — `full/cp`
> already uses the full-context measurement. (Mixing the per-card-shape value
> 204 with the full/cp anchors is the source of the −14 vs −233 confusion;
> the correct decomposition is purely the flat→top_last delta at full/cp.)

### 3.3 fmha, projections — no correction

FMHA attends to ≤ `index_topk` (2048) keys/token ⇒ per-token cost is
context-independent (4096-ctx 0.477 us/tok vs 32768-ctx 0.492, +3%). The base
already has the right per-token fmha at `isl/cp` tokens. Projections/RMSNorm
process the per-card `isl/cp` tokens — unchanged by CP. **Leave both as the
base value.**

### 3.4 AG_KV, AG_LSE — the two small attention comms

| comm | tensor | size | timeline |
|------|--------|------|----------|
| AG_KV | gathered KV latent (fp8) | `isl·(kv_lora+rope)·1B` | ~54us |
| AG_LSE | softmax LSE (bf16) | `isl·kv_lora·2B` | ~100us |

Looked up in the NCCL all-gather table at `num_gpus = cp`. (AG_hidden and RS
are **not** here — they belong to Comm, §6.)

---

## 4. Precision / base-row selection — READ THIS

`dsa_context_module` is collected for `kv_cache_dtype ∈ {bfloat16, fp8}` (both
with `mla_dtype = bfloat16`). Picking the wrong row swings the DSA base by
~1.6× (4300 vs 2668 at isl=4096). The rule:

1. **mqa (indexer) is ALWAYS fp8** — `sm100_fp8_(paged_)mqa_logits`, intrinsic
   to DSA, independent of `kv_cache_dtype`. (`glm5_mqa_logits` perf only ever
   has the `fp8_e4m3` combo.)
2. **The fmha runs the `QkvBfloat16` kernel** — bf16 compute. Even though the
   GLM-5-NVFP4 KV cache is stored fp8 (`fp8_ds_mla`), FlashMLA **dequantizes to
   bf16** and runs the **bf16 fmha kernel**. So the matching base row is the
   one whose fmha kernel is bf16:
   * **`kv_cache_dtype = bfloat16` row** → fmha = bf16 `QkvBfloat16` kernel =
     **what the real run uses** → **USE THIS ROW**.
   * `kv_cache_dtype = fp8` row → a *different, faster* true-fp8 fmha kernel
     that **does not occur** in the real run → using it under-predicts DSA by
     ~900us/layer.
3. **Pitfall:** "KV cache stored fp8" ≠ "fmha kernel is fp8". Select the base
   row by the **fmha kernel dtype (bf16)**, not by the KV storage dtype.
4. With `attn_tp=1`, pick `num_heads = native(64)`, `tp_size = 1`, `batch_size
   = 1` (per request).

Net: base = `dsa_context_module(isl/cp, prefix=0, kv=bfloat16, heads=64, tp=1,
bs=1)` = **4300us** (isl/cp=4096).

---

## 5. MoE — comm pattern follows the parallel mode

MoE = `moe(isl, moe_tp=cp)` on the **full token set** (after AG_hidden). The
collector caps `num_tokens` at 16384, so isl=32768 ⇒ `2 × moe(16384, …)`.
*isl=32768, tp8, ep1, power_law_1.01:* 1023.4·2 = **2047us**.

**The MoE's collective is dictated by its parallel mode — model it accordingly:**

| MoE parallel | input comm | output comm |
|--------------|-----------|-------------|
| **TP + SP (this config: moe_tp=cp, ep=1)** | **AG_hidden** (SP-gather) | **RS** (TP-reduce **+** SP-scatter, one op) |
| EP | all-to-all dispatch | all-to-all combine |
| pure DP | none | none |

For TP+SP there is **no separate TP all-reduce** — the RS does the reduce. (The
`cross_device_reduce_1stage` allreduce seen in the trace is **decode/post-
processing**, NOT part of a prefill layer — do not count it.)

Pick the `(moe_tp, moe_ep, distribution)` from the deployment config and look up
`moe_perf` accordingly; the Comm part (§6) then uses the comm primitives of
that mode.

---

## 6. Comm — the big collectives around MoE

For the TP+SP MoE config:

| comm | tensor | size | timeline |
|------|--------|------|----------|
| AG_hidden | attn-out hidden (bf16) | `isl·hidden·2B` | ~628us |
| RS | hidden, TP-reduce+SP-scatter (bf16) | `isl·hidden·2B` | ~658us |

```text
Comm = AG_hidden + RS                    # one AG_hidden, one RS per layer
```

Looked up in the NCCL all-gather / reduce-scatter tables at `num_gpus = cp`.

---

## 7. Per-layer assembly — serial, no overlap

The per-layer wall-clock is the **serial sum** of one card's critical path
(round-robin is balanced, so all cards are equal):

```text
per_layer = DSA + MoE + Comm
          = [base + mqaΔ + topkΔ + AG_KV + AG_LSE] + moe(isl,tp) + [AG_hidden + RS]
```

Measured decomposition (device 0, RS→RS, isl=32768/cp8) confirms it is serial —
segment sum = layer period, no hidden overlap:

```text
DSA(attn, no big comm)  3808us
AG_hidden                622us
MoE                     2430us
RS                       610us
─────────────────────────────
RS→RS period            7470us   (≈ 7.46ms, stable across layers)
```

(An earlier "~870us overlap" claim was an artifact of including the decode
allreduce in the busy-sum and a wrong per-layer normalization — there is no
real comm/compute overlap at this granularity.)

---

## 8. End-to-end validation (isl=32768, cp=8, bf16 base)

```text
DSA  = 4300(base,kv=bf16) + 316(mqa) − 233(topk) + 54(AG_KV) + 100(AG_LSE) = 4537
MoE  = moe(32768,tp8,ep1,pl1.01)                                           = 2047
Comm = AG_hidden 628 + RS 658                                              = 1286
──────────────────────────────────────────────────────────────────────────────
per_layer (predicted)                                                ≈ 7870 us
```

vs measured **7470us** (RS→RS) → **+5.4%**.

Per-segment (predicted vs measured): mqa 360/374 ✓, topk 190/251, fmha & proj
inside base (base slightly heavy), AG_KV 54/55 ✓, AG_LSE 100/94 ✓, AG_hidden
628/622 ✓, RS 658/610 ✓, MoE 2047/2430 (moe_perf misses input-quant +
elementwise ~380us). Comm and mqa are accurate; residual error is base-fmha
slightly heavy (+) vs MoE-perf light (−), partly cancelling.

---

## 9. How to implement in AIC

### 9.1 Data

`systems/data/b200_sxm/sglang/0.5.12/`: `dsa_context_module_perf.parquet`
(isl→16384 after the piecewise-skip fix in `collect_mla_module.py`),
`glm5_mqa_logits_module_perf.parquet`, `glm5_topk_module_perf.parquet`
(`score_mode ∈ {flat, top_last}`), `glm5_dsa_attn_module_perf.parquet`,
`nccl/2.28.9/nccl_perf.parquet`.

> The `topk_full` / per-card-reference points at `isl=32768, prefix=0` need
> `chunked_prefill_size ≥ isl`; the standard sweep caps isl at the runtime
> chunk size (16384). Collect the full point with `AIC_CHUNKED_PREFILL_SIZE`
> when a single-forward full reference is required.

### 9.2 `sdk/operations/dsa.py :: ContextDSAModule` — DSA part

```python
def get_cp_dsa(self, b, isl, cp, db, dims):
    per_card = isl // cp
    # base: bf16-KV row (fmha kernel is bf16 QkvBfloat16), heads=native, tp=1, bs=b
    base = db.query_dsa_context_module(per_card, prefix=0,
                                       kv_cache_dtype="bfloat16",   # NOT fp8 — §4
                                       num_heads=dims["num_heads"], tp_size=1, bs=b)
    mqa_corr  = db.query_glm5_mqa(per_card, prefix=0) * (cp - 1)        # ×cp, quadratic
    tk_flat   = db.query_glm5_topk(isl, prefix=0, mode="flat")          # full/cp
    tk_last   = db.query_glm5_topk(isl, prefix=0, mode="top_last")
    topk_corr = -(tk_flat - tk_last) / cp                              # flat→top_last
    ag_kv  = db.ag_latency(isl * (dims["kv_lora"] + dims["rope"]) * 1, cp)
    ag_lse = db.ag_latency(isl * dims["kv_lora"] * 2, cp)
    return base + mqa_corr + topk_corr + ag_kv + ag_lse
```

### 9.3 Layer assembly (estimator)

```python
dsa  = ContextDSAModule.get_cp_dsa(b, isl, cp, db, dims)
moe  = MoEOp.get(num_tokens=isl, moe_tp=cp, moe_ep=ep, distribution=dist)   # full tokens
comm = db.ag_latency(isl*hidden*2, cp) + db.rs_latency(isl*hidden*2, cp)    # AG_hidden + RS
layer = dsa + moe + comm        # serial
prefill = layer * num_layers    # (first_k_dense_replace layers have no MoE/DSA-MoE comm)
```

### 9.4 Invariants (do not get these wrong)

1. **base row = bf16-KV** (fmha kernel is bf16; KV-stored-fp8 ≠ fp8-fmha). §4.
2. **mqa uses `×cp`** on `mqa(isl/cp)`; **topk uses `full/cp`** then flat→top_last.
   Different on purpose; don't unify. topk has **no** context scale-up term.
3. **fmha & projections: no correction.**
4. **MoE on full `isl`** (not per-card); **comm primitives follow the MoE
   parallel mode** (TP+SP → AG_hidden+RS; EP → all-to-all). §5.
5. **One RS per layer**, after MoE, doing TP-reduce+SP-scatter. The
   `cross_device_reduce` allreduce is decode/post-proc — exclude it.
6. **per_layer = DSA + MoE + Comm, serial** (no overlap). §7.
7. `cp` from `attn_cp_size`; `attn_tp = tp_size/(attn_dp·cp)` sets the base
   row's `num_heads/tp`.
