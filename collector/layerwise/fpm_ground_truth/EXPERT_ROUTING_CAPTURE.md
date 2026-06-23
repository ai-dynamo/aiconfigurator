# FPM Expert-Routing Capture (design + GPU runbook)

Design for instrumenting the FPM ground-truth deployment to log **per-(layer,
step, rank) expert routing** alongside latency. This is the "Path 2" of the MoE
skew-modeling effort. It must be executed on a GPU machine with the Dynamo vLLM
image; this doc is the runbook to do that.

## Why

The MoE skew-feature idea (Frontier-style continuous `CV/Gini/entropy` features
fed to a regressor) is only worth building if skew actually drives latency in
real serving. Two questions we currently **cannot** answer:

1. **Does skew matter end-to-end?** FPM is the ground truth, but its scheduler
   metrics record only token counts — **no routing**. We can't compute skew from
   the existing CSVs.
2. **What causes the 0.21 residual?** On qwen36 `tp4_ep4`, mixed steps with
   *identical aggregate inputs* still vary ±21% in latency (stable after dropping
   warmup). The driver is an unrecorded per-step factor — plausibly skew, but
   unprovable without per-expert counts.

So this instrumentation is a **de-risking measurement**, not yet the model: it
tells us whether the whole RF+skew bet is worth making.

Context: `[[../README.md]]` (FPM Ground Truth section), `fpm_golden_runs/KNOB_MAPPING.md`,
and the driver analysis in `collector/layerwise/diagnostics/plot_mixed_latency_drivers.py`.

## What FPM lacks

`fpm_metrics_phase.csv` columns are scheduler-level only
(`ctx_tokens, decode_requests, ctx_kv_tokens, mean_decode_kv_tokens, latency_ms, …`).
Routing happens inside `model.forward`, which the scheduler never sees. So we must
instrument the model, not the scheduler.

## vLLM 0.20.1 mechanism (verified against tag `v0.20.1`)

vLLM ships a purpose-built, **CUDA-graph-safe** hook for exactly this. Do **not**
monkeypatch topk, and do **not** read the EPLB counter.

- **Routing chokepoint** — `BaseRouter.select_experts`
  (`vllm/model_executor/layers/fused_moe/router/base_router.py`). Shared by all
  router subclasses / backends; called from `MoERunner.forward`
  (`fused_moe/runner/moe_runner.py:462`). `topk_ids` is `[num_tokens_local,
  top_k]` integer expert indices; at the hook point they are **logical** ids
  (pre-EPLB-remap) = exactly the per-rank routing we want.
- **Official hook** — `BaseRouter.set_capture_fn(fn)`; `fn(topk_ids)` is called
  inside `select_experts` before EPLB remap. Backing impl `RoutedExpertsCapturer`
  (`fused_moe/routed_experts_capturer.py`) writes raw ids into a preallocated GPU
  buffer; gated by `model_config.enable_return_routed_experts` (default False),
  bound in `gpu_model_runner._bind_routed_experts_capturer` **before**
  `capture_model()`.
- **Do NOT use EPLB `expert_load_view`** (`distributed/eplb/eplb_state.py`):
  physical-slot space, only recorded on scheduler-chosen steps, requires
  `enable_eplb=True`. Wrong tool.

### CUDA-graph constraint (the critical detail)

Decode runs under CUDA graphs (unless `enforce_eager`); `enable_return_routed_experts`
does **not** force eager. A Python `capture_fn` **body** runs only once at graph
**capture/trace** time, but the **GPU op it enqueues** (bincount/scatter into a
fixed buffer) **replays every decode step**. So:

> The hook MUST be bound **before `capture_model()`**. A hook installed after
> capture runs only on eager fallback and silently misses every graphed step.

TP/EP/DP correctness is automatic: each worker's `select_experts` sees only its
local tokens; ids are logical → a per-layer `[num_experts]` bincount is the local
logical routing distribution.

## Two-stage rollout

### Stage A — concept check (cheapest)

Launch with `enable_return_routed_experts=True`, read the official
`RoutedExpertsCapturer` buffer, bincount over the topk dim yourself per
(layer, step).
- ✅ minimal code; see the skew distribution immediately.
- ⚠️ the built-in path does a per-step D2H copy → **timing perturbation**, so
  latency in this run is NOT clean. Use Stage A only to confirm "we can read
  routing + what the skew looks like", not to attribute latency.

### Stage B — minimal-perturbation (the real attribution run)

At worker init, **before `capture_model()`**, iterate `static_forward_context`
FusedMoE modules and `module.router.set_capture_fn(fn)`, where `fn(topk_ids)`
does an **on-GPU** bincount/scatter_add into a preallocated `[num_layers,
num_experts]` int tensor indexed by `layer_id`. Keep it fully on-GPU (no
`.item()`/`.cpu()` in the hook). Do one async D2H of the whole counter every N
steps on the host loop (reuse the `vllm_step_marker` `execute_model` boundary).
- routing + **clean latency** captured in the same run → enables attribution of
  the 0.21 residual.

## Injecting into the FPM Docker stack

FPM serving is a real Dynamo vLLM Docker stack (`collect_fpm_metrics.sh` →
`python3 -m dynamo.vllm`); it does **not** mount the layerwise patches. Both stages
have an injection path in the existing launcher:

- **Stage A (passthrough flag):** `collect_fpm_metrics.sh … -- --enable-return-routed-experts`
  (collected into `WORKER_EXTRA_ARGS`, appended to the worker cmd — same path as
  `--enable-expert-parallel`). Or `python -m collector.layerwise.fpm.collect
  --extra-vllm-arg=--enable-return-routed-experts`.
- **Stage B (sitecustomize):** the worker `docker run` (collect_fpm_metrics.sh
  ~line 1352) has an extensible env array `WORKER_DOCKER_ENV` and `-v` mounts. Add
  `-v <host-inject>:/inject:ro` and `WORKER_DOCKER_ENV+=(-e PYTHONPATH=/inject)`,
  put a `sitecustomize.py` in `/inject` (mirror `collector/layerwise/vllm/sitecustomize.py`).
  The hook itself must bind at the model-load-after / capture-before point — mirror
  `collector/layerwise/vllm/vllm_layer_skip_patch.py` (`_looks_like_moe_mlp`
  locates MoE modules; it already patches MoE forward, so the binding site is known).

## Open items to verify on the GPU machine

1. Does `dynamo.vllm` **pass through** `--enable-return-routed-experts` to vLLM?
   (gates Stage A)
2. In the Dynamo Docker image, where is the model-load → `capture_model()` point to
   bind the hook? (gates Stage B; layerwise binds in its own worker, FPM uses
   dynamo.vllm so the site may differ.)

## Output format

Per worker process, per step:
- raw per-(layer) `[num_experts]` counts → sidecar (npz/parquet), keyed by
  `(counter_id/step, dp_rank)` to align with `fpm_metrics_phase.csv`.
- derived **per-rank** skew scalars (`cv, gini, entropy, expert_utilization`) →
  a companion CSV column set (deterministic post-process from raw counts).

Compute skew at **per-rank** granularity (grouped-GEMM is per-device); raw global
counts + the EP/DP mapping are sufficient to derive per-rank, except under EPLB
(dynamic slot mapping) where slot/rank assignment must also be logged.

## Success criteria

- Stage B run yields routing + clean latency in the same run.
- Group mixed steps by identical aggregate inputs; check whether per-step skew
  (CV/Gini) explains the **0.21 within-group residual**. If yes → skew is a real
  driver and the RF+skew model is justified. If the residual stays unexplained →
  the bet is weak; reconsider before investing in the regressor.

## Key references

- vLLM `v0.20.1`: `fused_moe/router/base_router.py` (`select_experts`,
  `set_capture_fn`), `fused_moe/routed_experts_capturer.py`,
  `fused_moe/runner/moe_runner.py:462`, `v1/worker/gpu_model_runner.py`
  (`_bind_routed_experts_capturer`, per-step clear/save), `config/model.py`
  (`enable_return_routed_experts`).
- In-repo: `collect_fpm_metrics.sh` (worker launch, `WORKER_EXTRA_ARGS`,
  `WORKER_DOCKER_ENV`), `collector/layerwise/vllm/{sitecustomize.py,
  vllm_layer_skip_patch.py,vllm_step_marker.py}` (injection + MoE-locating +
  per-step boundary patterns).
