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

`balance_factor` and `fill_factor` are **surrogates** for what an event-driven
simulator computes natively. A request-level simulator (the Dynamo Mocker,
which embeds the compiled engine for per-iteration cost estimates) models
stage occupancy directly: imbalance, starvation, and heterogeneous
prefill/decode microbatches blocking each other all *emerge* from the
simulation. Such a consumer must take `PipelineLayout` and derive its own
bubbles — applying `fill_factor` on top would charge for the same bubble
twice.

AIC needs the closed form only because it has no simulator. Keeping the shared
primitive free of that assumption is what lets both live on one source of
truth for partitioning and per-stage cost, instead of drifting apart.

There is a validation payoff too: a simulator's occupancy result and
`balance_factor` should agree in the steady-state limit (high concurrency,
homogeneous requests). Disagreement means one of the two is wrong, and unlike
a silicon comparison this check is not confounded by perf-database error.

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

## 6. Known gaps

- **`fill_factor` is linear** (`min(1, M/pp)`). The real curve is likely
  steeper once scheduler and synchronization overhead are included; it needs
  silicon calibration.
- **`PipelineLayout` is not exported to native embedders yet.** The compiled
  engine's per-iteration surface (`prefill_latency_ms`, `decode_latency_ms`,
  `mixed_step_latency`) returns whole-model forward-pass latency with no stage
  dimension, and `pp_size` reaches only the `P2P` op on the Rust side. A
  simulator that wants to model pipeline occupancy needs a stage-indexed cost
  oracle (`stage_latency_ms(stage_idx, ctx_tokens, decode_tokens)`) plus the
  partition and per-hop cost. The Python layout is shaped for that; the FFI is
  future work.
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
- **PP is still excluded from the automatic search.** `build_disagg_parallel_lists`
  takes `should_enable_pp` (default `False`) and no caller passes `True`, so PP
  is reachable only via an explicit `--pp`. Turning it on should wait until the
  gaps above are calibrated.
