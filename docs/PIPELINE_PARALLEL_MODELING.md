# Pipeline-Parallel (PP) Modeling

How AIC estimates a step under **pipeline parallelism** — why the previous
"multiply throughput by `pp_size`" model over-predicts, and what
`PipelineSchedule` replaces it with.

---

## 1. What the model used to assume

PP had three touch points, and none of them changed a step's latency:

1. **Model graph** — each model appends a `P2P` op scaled by `pp_size - 1`.
2. **Memory** — `weights /= pp_size`. KV cache is deliberately *not* divided:
   each stage holds `layers/pp` layers for `b * pp` in-flight sequences, so the
   two `pp` factors cancel and `b * layers` per GPU is correct.
3. **Throughput** — `output_throughput *= pp_size`, `concurrency = b * pp_size`.

Crucially, `_num_layers` is **never** divided by `pp_size`, so a step's latency
is the whole model's latency `T_full`. Translated into pipeline terms, that is:

| Quantity | Old model | Implied assumption |
|---|---|---|
| Cycle time (one stage retires one microbatch) | `T_full / pp` | every stage costs the same |
| In-flight microbatches `M` | `= pp` | the pipe is always exactly full |
| TPOT | `T_full` | a request advances one token per rotation ✓ |
| Throughput | `pp * b / T_full` | zero bubble |

The TPOT row is right. The first two are not.

---

## 2. Why that over-predicts

### 2.1 A pipeline advances at `max_i(t_i)`, not the average

The un-sharded `lm_head` lives alone on the last stage. For a **decode** step
its cost is a large fraction of a single stage's layer work, so it sets the
cycle. Qwen3-32B (64 layers, vocab 151936) on H20 at `pp=8`:

```
stage times (ms): [3.153, 3.150, 3.150, 3.150, 3.150, 3.150, 3.150, 3.901]
                                                                    ^ lm_head
ideal cycle 3.244   realized cycle 3.901   ->  balance 83.2%  ->  6.65x, not 8x
```

For a **prefill** step the per-layer work dominates (the head only runs on one
token position), so the same config stays at 99.9% — the imbalance is a
decode-side effect.

### 2.2 Uneven layer splits were only warned about

`num_layers % pp_size != 0` used to log *"this will introduce additional
rounding error. Currently we're nothing to correct this."* The cycle is set by
the fattest stage, so 64 layers over 6 stages costs `11 / (64/6)` — a ~3% loss
that the old model did not charge.

### 2.3 A starved pipe still scaled linearly

With fewer in-flight microbatches than stages, some stages idle every cycle.
Nothing expressed that.

---

## 3. The model — two layers, deliberately separated

`aiconfigurator_core.sdk.pipeline` is split so that the piece other consumers
need is free of AIC's own scheduling assumptions.

### 3.1 `PipelineLayout` — where the work lives

Layer partition, op placement, per-stage compute times, per-hop link cost.
Pure geometry plus cost lookup, **no scheduling policy**.

```
stage_times(per_op_ms, num_layers) -> [t_0, t_1, ... t_{pp-1}]
per_hop_latency(per_op_ms)         -> P2P cost of one stage-to-stage hop
layer_partition(num_layers)        -> layers per stage
```

### 3.2 `PipelineSteadyState` — how the pipe runs

Wraps a layout and adds occupancy: microbatch count, P2P overlap, and the
closed-form factors AIC's mean-field step model collapses onto scalars.

```
cycle_time     = max_i(stage_time_i) + p2p_per_hop
balance_factor = (step_total / pp_size) / cycle_time      # per-microbatch latency effect
fill_factor    = min(1, num_microbatches / pp_size)       # throughput-only effect
efficiency     = balance_factor * fill_factor
```

### 3.3 Why the split

`PipelineLayout` owns the **rules** — which op belongs to which stage, and how
many layers each stage gets. `stage_times` is one derivation from those rules;
slicing the op list per stage would be another. Keeping the rules in one place
is what stops two derivations from drifting.

`balance_factor` and `fill_factor` are a different kind of thing: a
steady-state *collapse*, valid only because AIC evaluates one step shape and
assumes every stage sees it. Anything that models per-stage occupancy directly
would derive those effects itself and must not also apply the scalars.

Note what the current embedder contract actually is, because it constrains
where PP belongs. `ForwardPassPerfModel::estimate_forward_pass_time_ms` takes
per-**attention-DP**-rank `ForwardPassMetrics` and returns one iteration time,
reducing with `max` over ranks. There is no stage axis anywhere in it: a PP
worker reports a single forward pass, and the caller expects the estimate to
already include every intra-worker parallelism effect. **PP is AIC's
responsibility to model, not the caller's.**

That makes the placement obvious by symmetry:

```
Engine::forward_pass_time_ms:  max over attention-DP ranks   # DP runs in lockstep
pipeline cycle:                max over PP stages           # slowest stage sets the pace
```

Same reduction, different axis. PP belongs next to the DP `max` inside
`rank_latency_ms` — collapsed into the answer, not exported outward. See §6.

**Op placement** follows the naming contract the model classes already
require (see the `GPTModel` docstring: *"attn layer name needs to be
context_attention or generation_attention, exact match is required. Same for
logits_gemm"*):

| Match | Placement |
|---|---|
| `*embedding*` | first stage |
| `*logits_gemm*` | last stage |
| `*p2p*` | link — excluded from stage compute, charged once per hop |
| everything else | per-layer, follows the layer partition |

`warn_on_unclassified_ops` flags an op whose scale factor is far below
`num_layers` but which no marker matched — i.e. a new head-like op that would
otherwise be silently smeared across every stage.

**Layer partition** defaults to an even split with the remainder on the leading
stages, matching vLLM's `get_pp_indices` and TRT-LLM. `PipelineSchedule` also
accepts an explicit `partition` (some engines let you front-load layers to
compensate for the head).

**P2P** is charged once per hop rather than as a whole-step total: the op's
scale factor is `pp_size - 1`, so `per_hop = link_total / (pp_size - 1)`.
`p2p_overlap=True` drops it entirely for engines that hide the transfer behind
compute.

---

## 4. Where it plugs in

Two touch points in `BaseBackend.run_agg`, each with a single meaning:

```python
# 1. Inflate the step latency to the real pipeline traversal time (pp * cycle).
pp_pipe = self._pipeline_steady_state(model, **kwargs)
mix_step_latency_ms     /= pp_pipe.balance_factor(mix_per_ops, model._num_layers)
genonly_step_latency_ms /= pp_pipe.balance_factor(genonly_per_ops, model._num_layers)

# 2. Charge pipe starvation against throughput only.
output_throughput = output_throughput * scale_factor * pp_pipe.fill_factor()
```

Inflating the step latency is the load-bearing choice: TTFT, TPOT,
`_total_step_latency_ms`, `_step_throughput` and `_throughput_cap` are all
derived from it downstream, so they stay consistent without individual patches.

Two overridable backend hooks mirror the split:

| Hook | Returns | Override to change |
|---|---|---|
| `_pipeline_layout(model)` | `PipelineLayout` | the layer partition |
| `_pipeline_steady_state(model, **kwargs)` | `PipelineSteadyState` | the microbatch policy |

`_pipeline_steady_state` composes `_pipeline_layout` by default, so a backend
that overrides only the partition affects both AIC's own math and anything
else built on the layout.

### Why it lives in orchestration, not in the ops

This is deliberately **not** an op-layer change. Per
`.claude/rules/rust-core/parity.md`, changing op query math obligates a
mirrored Rust implementation plus oracle anchors. `PipelineSchedule` consumes
the per-op breakdown *after* the step is evaluated, so the Python and Rust
engine-step paths both benefit with no parity surface touched — the same
layering already used for AFD pipeline modeling.

---

## 5. Invariants

- **`pp_size == 1` returns exactly `1.0`** from both factors, so single-stage
  results are bit-identical to the previous model.
- `balance_factor` is clamped to `(0, 1]`; it can never manufacture speedup.
- `fill_factor` defaults to `1.0` (`num_microbatches=None` ⇒ `pp_size`),
  reproducing the historical "always exactly full" assumption unless a caller
  passes `pipeline_microbatches`.

---

## 6. Silicon validation

Qwen3-32B (bf16, dummy weights), 8× H20-3e, TRT-LLM 1.3.0rc20 against the
`h20e_sxm` perf DB collected on the same stack and version — so absolute
numbers are comparable, not just ratios. Decode-dominated workload
(isl=256, osl=512), 8 GPUs held constant across every `(tp, pp)`, chunked
prefill off, aiperf closed-loop.

### 6.1 Scope: this run cannot show PP winning, by construction

Measured output throughput, as a ratio to `tp8/pp1`:

| concurrency | tp4/pp2 | tp2/pp4 | tp1/pp8 |
|---|---|---|---|
| 1 | 0.62 | 0.43 | 0.28 |
| 8 | 0.60 | 0.42 | 0.28 |
| 32 | 0.62 | 0.45 | 0.33 |
| 64 | 0.53 | 0.56 | 0.36 |
| 128 | 0.38 | 0.54 | 0.40 |

No configuration beats `pp=1` — as expected. **PP's payoff is crossing nodes**,
and this run is single-node. Holding total GPUs fixed inside one NVLink domain,
per-GPU weight bytes are unchanged by PP (`tp*pp` constant), so pipelining buys
nothing while the P2P hops and the smaller per-stage batch cost real time.

The regime PP exists for is the one where TP would have to span the slow
fabric. On this spec (`num_gpus_per_node=8`, `inter_node_bw` 50 GB/s vs
`intra_node_bw` 450 GB/s), TP all-reduces every layer across that 9x cliff
while PP crosses it once per stage boundary with a single activation tensor.
AIC predicts the difference clearly (Qwen3-32B, isl=256/osl=512, C=64):

| GPUs | TP-only | best PP config | ratio |
|---|---|---|---|
| 16 (2 nodes) | tp16/pp1: 1714 tok/s | tp8/pp2: 7073 | **4.1x** |
| 32 (4 nodes) | tp32/pp1: 1735 tok/s | tp8/pp4: 9275 | **5.4x** |

So treat §6.2 and §6.3 as **error-mode stress tests of the model**, not as
evidence about whether PP is worth deploying. Single-node PP is the regime that
maximally exposes the model's failure modes precisely because there is no real
gain for them to hide behind.

The `pp=1` baseline predicts within −8% to +11% on throughput and within 8% on
TPOT at every concurrency, so the discrepancies below are PP-specific and not
a perf-database problem.

**Both remaining gaps over-predict**, which is the dangerous direction for the
cross-node case too: if the §6.3 saturation effect generalises, the 4–5x above
is optimistic and AIC would over-recommend pipeline depth. Neither gap has been
measured cross-node — that needs multi-node hardware this campaign did not
have.

### 6.2 The "always full pipe" default is the largest single error

With `C` concurrent requests and `pp` stages, at most `min(pp, C)` microbatches
can be in flight. The default assumes `pp`. Driving `num_microbatches` from the
real concurrency instead:

| C | tp/pp | measured | AIC full pipe | AIC `M=min(pp,C)` |
|---|---|---|---|---|
| 1 | 4/2 | 106.9 | 260.4 (+144%) | 130.2 (+22%) |
| 1 | 2/4 | 73.4 | 317.6 (+333%) | 79.4 (+8%) |
| 1 | 1/8 | 48.0 | 357.6 (**+645%**) | 44.7 (−7%) |

This is what `fill_factor` is for, and it works — but nothing sets it today.
See gap 1.

### 6.3 A concurrency-dependent PP gap remains unexplained

For `C >= pp`, where `min(pp, C) == pp` and the fill correction is inert, AIC
still over-predicts, and the error grows with concurrency:

| C | pp=2 | pp=4 | pp=8 |
|---|---|---|---|
| 8 | +44% | +26% | +7% |
| 32 | +64% | +56% | +25% |
| 64 | +100% | +45% | +42% |
| 128 | +173% | +82% | +73% |

Stage imbalance does not explain it: toggling the factors from §3 moves these
by about 1 point. The measured behaviour is also **non-monotonic in `pp`** —
at C=128 the pp=2 ITL is 49.8 ms against pp=4's 32.8 ms, and pp=2 throughput
*falls* between C=64 and C=128 (2576 → 2413 tok/s) while ITL doubles. That
saturation signature points at engine-side scheduling rather than a smooth
pipeline bubble, and needs its own investigation before any coefficient is
fitted to it.

## 7. Known gaps

- **Nothing sets `num_microbatches`, and the default is wrong.** `run_agg`
  treats `concurrency = batch_size * pp_size` as an *output*, so the pipe is
  full by construction and `fill_factor` is always 1.0. A caller that wants a
  concurrency below `pp_size` cannot express it — the reported concurrency
  silently rounds up to a multiple of `pp_size`. §6.2 measures the cost: up to
  +645% at C=1. `pipeline_microbatches` is the hook; wiring it means letting
  the sweep/SLA layer treat concurrency as a target rather than a derived
  value, which is a convention change worth deciding explicitly.
- **`fill_factor` is linear** (`min(1, M/pp)`), and §6.3 shows a
  concurrency-dependent gap it does not capture. Do not fit a coefficient to
  that gap until the non-monotonic saturation behaviour is understood.
- **This change covers `run_agg` only — the compiled-engine path is still
  ideal-PP.** The Dynamo planner and Mocker reach AIC through
  `ForwardPassPerfModel` → `Engine::forward_pass_time_ms` →
  `rank_latency_ms`, which never enters `run_agg`. On that path `pp_size`
  still reaches only the `P2P` op, so a PP worker's iteration time is the
  whole model's with no stage reduction. Fixing it means applying the same
  `max`-over-stages inside `rank_latency_ms`, which is a Rust change against a
  literal port (`SessionEstimator::rank_latency_ms`) and therefore its own PR
  under `.claude/rules/rust-core/parity.md`.

- **The FPM online correction silently absorbs the PP error.**
  `tune_with_fpms` learns `median(observed_ms / native_ms)` per region, where
  `native_ms` is the whole-model estimate above. The only guard on an
  observation is finite-and-positive — **the factor is unbounded in
  magnitude** — and the region key is built purely from scheduler counts
  (`sum_prefill_tokens`, `num_decode_requests`, `sum_decode_kv_tokens`), so
  `pp_size` is not a feature. Consequences depend on what the engine's
  telemetry reports as `wall_time`, which is emitter-side and not determined
  here:

  - whole-iteration wall time ⇒ `observed ≈ native`, correction ≈ 1, and PP's
    throughput benefit is simply absent from the model;
  - per-stage wall time ⇒ the correction learns ≈ `1/pp_size` and the model
    becomes right for the wrong reason: a fitted constant standing in for a
    structural effect, correct at the calibrated point and wrong as soon as
    `pp_size` or the stage imbalance changes.

  Either way a structural error is laundered through a fitted factor with no
  warning. This is a second reason to make `rank_latency_ms` pipeline-aware
  rather than leaving the correction layer to paper over it.
- **Chunked prefill × PP is not modeled.** Adjacent chunks of one request have
  a RAW dependency on the KV they write, so they cannot occupy the pipe
  simultaneously — a single request doing chunked prefill gets no PP benefit,
  and filling the pipe requires enough concurrent prefills. `fill_factor` is
  the intended hook; the coefficient needs measurement.
- **P2P always uses `inter_node_bw`.** `SystemSpec.get_p2p_bandwidth(num_gpus)`
  already implements the three-tier topology selection on both the Python and
  Rust sides, but the `P2P` op hardcodes the inter-node tier, which is ~9x
  pessimistic for single-node PP. The correct selector is `tp * pp` (the
  worker's GPU count), which the op is not constructed with — fixing it means
  touching every model's `P2P` construction plus the Rust mirror, so it is left
  to its own change. Measured impact on Qwen3-32B at `pp=8`: 0.67% of a decode
  step, 2.2% of a prefill step, in the conservative direction.
- **PP is excluded from the automatic search, and that is costly.**
  `build_disagg_parallel_lists` takes `should_enable_pp` (default `False`) and
  no caller passes `True`, so PP is reachable only via an explicit `--pp`. For
  any deployment that must span nodes this is not a conservative default but a
  wrong answer: at 16 GPUs the search can only offer `tp16/pp1` (1714 tok/s)
  when `tp8/pp2` is worth 7073 — a 4x miss on a real sizing decision (§6.1).
  The fix is gated on the gaps above, since enabling a search dimension the
  model over-predicts would make it pick pipelines that are too deep.
