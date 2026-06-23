<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Spica search space

A `SearchSpace` (the `search_space:` block of a `SmartSearchConfig` YAML) is the input
to one smart-sweep run: the knobs to **explore**, plus the **pinned** context (model,
hardware, GPU budget, and the engine/router/kv-manager scalars). The optimizer runs one
study per `(deployment_mode, backend)` branch and ranks candidates across branches.

This document is the reference for **what you can pin and how**.

## The pinning model

There are four kinds of knob:

| Kind | How to pin | How to search |
|---|---|---|
| **Atomic list knob** | a single-element list, `[x]` | a multi-element list, `[a, b, c]` |
| **Pinned scalar** | just set the value (always one value) | — (never searched) |
| **Composite knob** | one entry: a preset id **or** a dict | multiple entries (presets and/or dicts) |
| **`parallel_configs`** | one `dict` entry | multiple `dict` entries (a custom menu) |

Everything is pinnable **except** the parallel shape + replica count when you leave
`parallel_configs` empty — then it is *derived* by KV-feasible enumeration. Provide
`parallel_configs` to pin or to search a custom menu (see below).

### Atomic list knobs

A list whose entries must be a non-empty subset of the allowed choices. One element pins it.

- `deployment_mode` (`disagg` / `agg`), `backend` (`vllm` / `sglang` / `trtllm`)
- engine batching: `agg_max_num_batched_tokens` / `agg_max_num_seqs`,
  `prefill_max_num_batched_tokens` / `prefill_max_num_seqs`,
  `decode_max_num_batched_tokens` / `decode_max_num_seqs`
- router: `router_mode` (`kv_router` / `round_robin`) + the kv-router weights
  `overlap_score_credit`, `prefill_load_scale`, `router_temperature`.
  `host_cache_hit_weight` and `disk_cache_hit_weight` are **only swept when multi-tier
  KV offload is enabled** (`num_g2_blocks > 0`): in the router's scoring they weight the
  host/disk *extension* blocks, which are 0 with offload off (the default), so they can't
  affect a replay and are dropped from the search to avoid dead dimensions.

### Pinned scalars (one value, never searched)

`model_name`, `hardware_sku`, `gpu_budget`, `min_gpu_budget`, `min_endpoint`,
`context_length`, `startup_time`, `aic_nextn`; the per-role `*_block_size`,
`*_gpu_memory_utilization`, `*_enable_prefix_caching`; the kv-manager fields
(`num_g2_blocks`, `bandwidth_g1_to_g2_gbps`, `bandwidth_g2_to_g1_gbps`,
`offload_batch_size`); and admission control (`active_decode_blocks_threshold`,
`active_prefill_tokens_threshold`, `active_prefill_tokens_threshold_frac`,
`no_admission_control`).

## Composite knobs

A composite knob bundles several coupled *unrolled* fields behind a named **preset**.
Each list entry is **either** a preset id (string) **or** a `dict` that pins those
unrolled fields directly — the escape hatch for a value no preset offers. The list is a
candidate set: one entry = pin, several = search; presets and dicts can be mixed.

A dict entry is **self-contained**: it *replaces* the preset expansion (no partial / merge).
Required keys are enforced at config load — the three **planner** composites must provide
*all* of their unrolled fields; a **`load_predictor_candidates`** dict needs at least
`load_predictor` (the family), and its family params default per family. Unknown keys are
rejected. To tweak one field off a preset, copy that composite's fields into a dict and
change the one. The *legality* of the values (perfect-square fpm bucket, interval > 0,
scale-up > scale-down, …) is validated downstream by dynamo's `PlannerConfig`.

### `planner_scaling_policy`

| preset | `enable_throughput_scaling` | `enable_load_scaling` | `throughput_adjustment_interval_seconds` | `load_adjustment_interval_seconds` |
|---|---|---|---|---|
| `disabled` | False | False | — | — |
| `throughput_180_5` | True | False | 180 | 5 |
| `throughput_600_5` | True | False | 600 | 5 |
| `load_180_5` | False | True | 180 | 5 |
| `load_180_10` | False | True | 180 | 10 |
| `hybrid_180_5` | True | True | 180 | 5 |
| `hybrid_600_5` | True | True | 600 | 5 |

Dict keys: `enable_throughput_scaling`, `enable_load_scaling`,
`throughput_adjustment_interval_seconds`, `load_adjustment_interval_seconds`.
**Both flags `false` ⇒ planner off** (static replica count — same as `disabled`); no
intervals / fpm / sensitivity / predictor are emitted.

**The planner's `optimization_target` is derived from the sweep goal, not from this
policy** — `throughput`→`"throughput"`, `e2e_latency`→`"latency"`,
`goodput`/`goodput_per_gpu_hour`→`"sla"` (see `OptimizationTarget.planner_optimization_target`).
The policy only decides *which* scaling loops run + their intervals.

**Predictive throughput scaling needs an SLA**, so it only works under a goodput sweep
(`optimization_target="sla"`). For a `throughput`/`e2e_latency` sweep, the
throughput-scaling presets (`throughput_*`, `hybrid_*`, or any dict with
`enable_throughput_scaling: true`) are **automatically dropped** from the search (with a
log); only `disabled` / `load_*` survive. If you list *only* throughput-scaling policies
for a non-goodput sweep, the run errors (nothing left to search).

### `planner_fpm_sampling`

| preset | `max_num_fpm_samples` | `fpm_sample_bucket_size` |
|---|---|---|
| `small` | 32 | 4 |
| `default` | 64 | 16 |
| `large` | 128 | 16 |
| `fine` | 128 | 64 |

Dict keys: `max_num_fpm_samples`, `fpm_sample_bucket_size` (must be a perfect square).

### `planner_load_sensitivity`

| preset | `load_scaling_down_sensitivity` | `load_min_observations` |
|---|---|---|
| `aggressive` | 70 | 3 |
| `default` | 80 | 5 |
| `conservative` | 90 | 8 |

Dict keys: `load_scaling_down_sensitivity` (0–100), `load_min_observations`.

### `load_predictor_candidates`

The forecaster for predictive throughput scaling. It is **not** part of the main study:
a separate forecast-loss sub-sweep scores every entry against the trace and pins the
per-interval winner into the main sweep. A dict entry is therefore a *custom predictor*
the sub-sweep will also score.

| preset | `load_predictor` | `load_predictor_log1p` | extra family fields |
|---|---|---|---|
| `constant_last` | constant | False | — |
| `arima_raw` / `arima_log1p` | arima | False / True | — |
| `prophet_w20_*` | prophet | raw / log1p | `prophet_window_size`=20 |
| `prophet_w50_*` | prophet | raw / log1p | `prophet_window_size`=50 |
| `kalman_default_*` | kalman | raw / log1p | `kalman_q_level`=1.0, `kalman_q_trend`=0.1, `kalman_r`=10.0, `kalman_min_points`=5 |
| `kalman_reactive_*` | kalman | raw / log1p | `kalman_q_level`=10.0, `kalman_q_trend`=1.0, `kalman_r`=5.0, `kalman_min_points`=3 |

Dict keys: `load_predictor` (`constant` / `arima` / `prophet` / `kalman`),
`load_predictor_log1p`, `prophet_window_size` (prophet), and
`kalman_q_level` / `kalman_q_trend` / `kalman_r` / `kalman_min_points` (kalman). Omitted
family fields take the planner defaults.

## Branches & backend

The sweep runs **one Vizier study per `deployment_mode`** (agg / disagg — they have
structurally different parallel configs). **`backend` is a searched knob, not a branch**:
listing multiple backends searches them *together* within each mode's study (the optimizer
can shift budget toward the better backend), and `rank()` picks the global best across all.
The parallel-config menu is the **union** of every backend's KV-feasible configs; a sampled
`(backend, parallel_config)` pair the backend can't run is marked infeasible (no replay), so
the optimizer learns to avoid it. A backend with no perf DB / no viable config for a mode is
dropped from the search; a **mode** for which no backend is viable is **skipped with a
warning** (a viable mode still runs), and only if *no* mode is viable does the run error.
(A *pinned* config legal for no backend is a hard error — fail fast.)

## `parallel_configs` (the derived dimension you can also pin)

Left empty, the parallel shape + replica count are **enumerated**: for `(model, hardware,
gpu_budget)` and each backend, every KV-feasible per-worker shape × replica count that fits
the budget (unioned across backends). Provide a list of dicts to **pin** (one entry) or to
search a **custom menu** (several entries) instead of the full enumeration; a pinned config is
kept for whichever backends it's legal+feasible on (errors if none).

- **agg** entry — a flat shape dict: `tp` (required), `attention_dp`, `moe_tp`, `moe_ep`,
  `pp`, `replicas`. Omitted dims default to `1`; `replicas` defaults to `1`. Dense models
  can write just `{tp: N}`.
- **disagg** entry — nests two shape dicts: `{prefill: <shape>, decode: <shape>}`, each
  with its own `replicas`.
- **Derived, not settable:** `strategy` (computed from the shape — `tp` / `tep` / `dep`)
  and `used_gpus` (`gpus_per_worker × replicas`, summed across roles for disagg).

Each pinned shape is validated against the same rules the enumerator applies — MoE width
(`tp × attention_dp == moe_tp × moe_ep`), `gpus_per_worker ∈ {1,2,4,8,16}`, backend
filters (e.g. trtllm forbids `tp>1 & attention_dp>1`), KV-cache feasibility for the model's
max sequence, and `used_gpus ≤ gpu_budget`. An illegal pin is rejected (it never reaches
replay). **Pinning `parallel_configs` requires `deployment_mode` to list exactly one mode.**

## Examples

Pin a single deployment (one candidate — handy for a targeted re-evaluation):

```yaml
search_space:
  model_name: deepseek-ai/DeepSeek-V3
  hardware_sku: gb200
  deployment_mode: [agg]            # single mode (required to pin parallel_configs)
  backend: [trtllm]
  gpu_budget: 64
  parallel_configs:
    - {tp: 4, moe_ep: 4, replicas: 2}     # TEP, 8 GPUs
  agg_max_num_batched_tokens: [16384]
  agg_max_num_seqs: [512]
  router_mode: [round_robin]
  planner_scaling_policy: ["load_180_5"]
  planner_fpm_sampling: ["default"]
  planner_load_sensitivity: ["default"]
```

Search presets but pin one composite field with a dict (custom 240 s interval):

```yaml
  planner_scaling_policy:
    - "throughput_180_5"
    - {enable_throughput_scaling: true, enable_load_scaling: false,
       throughput_adjustment_interval_seconds: 240, load_adjustment_interval_seconds: 5}
```

Pin a disagg parallel config (prefill TEP, decode DEP):

```yaml
  deployment_mode: [disagg]
  parallel_configs:
    - prefill: {tp: 8, attention_dp: 1, moe_tp: 1, moe_ep: 8, replicas: 1}   # 8 GPUs
      decode:  {tp: 1, attention_dp: 8, moe_tp: 1, moe_ep: 8, replicas: 2}   # 16 GPUs
```

## Validation summary

- **List knob** — non-empty; string entries must be a listed choice.
- **Composite dict** — no unknown keys, and the required keys are present (all fields for
  the planner composites; at least `load_predictor` for `load_predictor_candidates`). Value
  legality is checked downstream by dynamo's `PlannerConfig`.
- **`parallel_configs`** — structural + single-mode at config load; full legality /
  KV-feasibility when branches are enumerated.
