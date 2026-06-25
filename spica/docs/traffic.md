<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Traffic (Workload)

The `workload:` block of a `SmartSearchConfig` YAML is the traffic **every candidate is
replayed against**. It is **pinned, never searched** (the one exception — a list-valued
`concurrency` under a `pareto` goal — is the swept Pareto dimension, not a candidate knob).
It maps to `Workload` in `src/spica/config.py`.

A workload is **exactly one of three load shapes**. The shape is inferred from which field
is set (`Workload._validate_workload`), and each shape is either **open-loop** (requests
arrive on a clock, independent of how fast the system drains them) or **closed-loop** (a
fixed number of requests are kept in flight; a new one starts as one finishes).

## The three load shapes

| # | Shape | Set | Loop | Driven by |
|---|---|---|---|---|
| 1 | **mooncake trace** | `trace_path` | open-loop (default) | the trace's arrival timestamps, scaled by `arrival_speedup_ratio` |
| 1c | **mooncake trace, capped** | `trace_path` + `replay_concurrency` | closed-loop | cap N in flight; **trace timestamps ignored** |
| 2 | **synthetic request-rate** | `request_rate` (+ `isl`/`osl`/`num_request_ratio`) | open-loop | a fixed QPS (`synthetic_arrival_interval_ms = 1000 / request_rate`) |
| 3 | **synthetic concurrency** | `concurrency` (+ `isl`/`osl`/`num_request_ratio`) | closed-loop | cap N in flight |

`is_trace_based` is `trace_path is not None`; `is_synthetic` is its complement. Setting
`trace_path` selects shape 1; otherwise it is synthetic and **exactly one** of
`request_rate` (shape 2) or `concurrency` (shape 3) selects the sub-shape.

The closed-loop in-flight cap is resolved by `effective_in_flight_cap()` (`None` = open-loop):

- trace -> `replay_concurrency` (so a trace is open-loop unless `replay_concurrency` is set);
- synthetic with a per-trial `concurrency_override` (pareto sweep) -> that override;
- synthetic with a scalar `concurrency` -> that value;
- otherwise (request-rate, or a list `concurrency` with no override) -> `None` (open-loop / awaiting override).

## Fields

Every `Workload` field:

| Field | Type | Default | Meaning |
|---|---|---|---|
| `isl` | `int \| None` | `None` | Synthetic input (prompt) sequence length, tokens. Required for synthetic. |
| `osl` | `int \| None` | `None` | Synthetic output sequence length, tokens. Required for synthetic. |
| `concurrency` | `int \| list[int] \| None` | `None` | Closed-loop in-flight cap (shape 3). A `list[int]` is the **swept Pareto dimension** and is allowed **only under a `pareto` goal**. Mutually exclusive with `request_rate`. |
| `request_rate` | `float \| None` | `None` | Open-loop QPS (shape 2). Mutually exclusive with `concurrency`. |
| `num_request_ratio` | `float \| None` | `None` | Synthetic request count relative to the load: `num_requests = round(num_request_ratio * load)`. Required for synthetic. See below. |
| `shared_prefix_ratio` | `float` | `0.0` | Fraction of shared prefix across requests (cache-locality / prefix sharing). |
| `num_prefix_groups` | `int` | `0` | Number of distinct shared-prefix groups. |
| `turns_per_session` | `int` | `1` | Turns per multi-turn session. |
| `inter_turn_delay_ms` | `float` | `0.0` | Think-time between turns in a multi-turn synthetic session, ms. |
| `trace_path` | `str \| None` | `None` | Path to a replay trace (shape 1). Its presence selects the trace shape and **forbids** all synthetic fields. |
| `trace_format` | `str` | `"mooncake"` | Replay-ready trace schema. Decoded but **not** forwarded by the evaluator — the trace path is read as mooncake regardless; effectively inert today. |
| `arrival_speedup_ratio` | `float` | `1.0` | Scales the trace's inter-arrival times (open-loop trace only). `>1` speeds arrivals up. |
| `replay_concurrency` | `int \| None` | `None` | Closed-loop in-flight cap **for a trace** (shape 1c); when set, trace timestamps are ignored. For synthetic closed-loop use `concurrency` instead. |

The synthetic fields are `isl`, `osl`, `request_rate`, `concurrency`, `num_request_ratio`;
`shared_prefix_ratio`, `num_prefix_groups`, `turns_per_session`, `inter_turn_delay_ms` are
shared synthetic knobs threaded into the replay (`ReplayEvaluator._synthetic_kwargs`).

## `num_request_ratio` (synthetic length scales with the load)

`resolved_request_count(concurrency_override=None)` computes the synthetic request count as

```
num_requests = max(1, round(num_request_ratio * load))
```

where `load` is, in precedence order: the per-trial `concurrency_override` (pareto sweep),
else a scalar `concurrency` (closed-loop), else `request_rate` (open-loop). A list-valued
`concurrency` with **no** override raises — the override is expected to supply the load.

So the synthetic trace length **scales with the swept load automatically**: with
`num_request_ratio = 10`, concurrency `256` yields `2560` requests, concurrency `512`
yields `5120`. Result is floored at `1`; `num_request_ratio` itself is treated as `0.0`
when unset (`max(1, …)` keeps at least one request).

## Validation (`Workload._validate_workload`)

- **Trace workload** (`trace_path` set): must **not** set any synthetic field
  (`isl`, `osl`, `request_rate`, `concurrency`, `num_request_ratio`) — error lists the
  offenders. `replay_concurrency`, if set, must be a positive int.
- **Synthetic workload** (no `trace_path`): **exactly one** of `request_rate` or
  `concurrency` (neither / both -> error); `isl`, `osl`, `num_request_ratio` are all
  **required**; `replay_concurrency` is rejected (it is trace-only — use `concurrency`).
  `concurrency` must be a positive int or a non-empty list of positive ints (bools
  rejected); `request_rate`, `isl`, `osl`, `num_request_ratio` must be positive.
- **List-`concurrency` only under pareto** — enforced one level up in
  `SmartSearchConfig._validate_concurrency_sweep`: a list-valued `workload.concurrency`
  (i.e. `concurrency_choices is not None`) is allowed **only** when `goal.target == "pareto"`;
  every other goal needs a single concurrency value. `concurrency_choices` returns the list
  (the sampler's per-trial discrete dimension) or `None`.

## Replay routing (from `evaluator.py`)

Each shape × deployment case routes to a dynamo replay entrypoint; all emit the same flat
`trace_report` dict. `ReplayEvaluator.evaluate` branches on `is_trace_based`, then on
`plan.is_static`:

| Load | static (no planner) | planner-in-the-loop |
|---|---|---|
| **mooncake trace** | `dynamo.replay.api.run_trace_replay` (arrival timestamps, or closed-loop when `replay_concurrency` set) | `dynamo.replay.main._run_planner_replay(trace_file=…)` |
| **synthetic** (rate or concurrency) | `PlannerReplayBridge.from_synthetic` / `from_synthetic_disagg`, driven to completion with no `apply_scaling` (fixed-fleet) | `_run_planner_replay(synthetic=SyntheticWorkload(…))` |

Notes:

- A synthetic **static** run reuses the planner bridge (`from_synthetic[_disagg]`) and
  drives it via `advance_to(_SIM_FOREVER_MS)` until `is_done`, never calling
  `apply_scaling` — a fixed-fleet replay. It uses the bridge (not the plain
  `run_synthetic_trace_replay`) because that constructor threads the **goodput SLA**; that
  is the one real differentiator (`gpu_hours` is part of the shared report every mocker
  runtime emits, so it is not bridge-exclusive).
- The closed-loop cap passed as `replay_concurrency=` on every path is
  `effective_in_flight_cap()` — `replay_concurrency` for a trace, `concurrency` (or the
  per-trial `concurrency_override`) for synthetic.
- The **goodput SLA** (`goal.sla`) is passed as `sla_ttft_ms` / `sla_itl_ms` / `sla_e2e_ms`
  on every path **only when an SLA is configured** — `_goodput_sla_kwargs` returns `{}` when
  `goal.sla is None`, so no `sla_*` kwargs are passed and no goodput is computed. It is
  independent of the planner's own scaling SLA.
- Under `kv_router` the searched router weights become a real `KvRouterConfig`;
  `round_robin` passes `router_config=None`.
