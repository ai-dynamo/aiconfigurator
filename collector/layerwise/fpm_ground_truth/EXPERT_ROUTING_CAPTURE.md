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

## ⚠️ Reality check (measured on B300, vLLM 0.20.1, Qwen3.6-35B-A3B)

The documented `enable_return_routed_experts` mechanism below **does not fire on
this model/hardware**. On B300/Blackwell the unquantized bf16 MoE selects the
**FlashInfer-TRTLLM monolithic** kernel (`MoEPrepareAndFinalizeNoDPEPMonolithic`).
`MoERunner._apply_quant_method` then takes the `quant_method.is_monolithic`
branch → `apply_monolithic()` routes **inside** the kernel and
`BaseRouter.select_experts` is **never called**. The official `RoutedExpertsCapturer`
buffer therefore stays all-zero (verified: 100% of routing mass on expert 0, the
bincount of a zero buffer). `enable_return_routed_experts` is silently a no-op for
the monolithic backend.

**What we actually capture instead** (real APIs only): wrap
`MoERunner._apply_quant_method` (always receives `router_logits`, holds
`self.router` + the `layer` whose `.layer_id` is the global index) and recompute
the routing with the model's own
`self.router.select_experts(hidden_states, router_logits)` — a pure function of
the gate logits, identical to what the monolithic kernel routes internally. We
scatter_add the logical expert ids into a preallocated `[num_layers, num_experts]`
GPU counter; the op is enqueued inside the CUDA graph (the wrap is installed
before `capture_model()`), so it replays every graphed step. The monolithic
kernel still runs unchanged for latency. The only perturbation is the extra topk
recompute, which is ~constant per fixed token count and so does not bias
variance-based within-group attribution. See
`routing_capture/inject/fpm_routing_capture.py`.

**Padding note:** under FULL decode CUDA graphs the recompute runs on the padded
batch, so pure-decode counts are inflated (`Σexpert/(tok·top_k)` up to ~1.78).
**Mixed steps are exact** (ratio = 1.000) because the real prefill tokens dominate
— and mixed steps are the only ones the 0.21 residual lives on, so this is fine.

## vLLM 0.20.1 mechanism (the documented path — bypassed here, kept for reference)

vLLM ships a purpose-built, **CUDA-graph-safe** hook. It works only when the MoE
runs the **non-monolithic** path (Triton backend etc.); on the monolithic kernel
above it is silently skipped. Do **not** monkeypatch topk, and do **not** read the
EPLB counter.

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

## Running it in-image on SLURM (the path actually used)

The doc originally assumed a host that `docker run`s separate frontend/worker/
collector containers. On the SLURM GPU node we are **already inside** the Dynamo
vLLM 0.20.1 image, so there is no outer docker to orchestrate and no `-v` mount:
we run the pieces as **direct processes** and inject the hook with
`export PYTHONPATH=<inject_dir>:$PYTHONPATH`.

Launcher: `routing_capture/run_routing_stack.sh` (core logic extracted from
`collect_fpm_metrics.sh`, docker wrappers dropped). It starts:

- `python3 -m dynamo.frontend --http-port 8000 --discovery-backend file --request-plane tcp --event-plane zmq`
- the worker (golden tp4_ep4 engine args) with `PYTHONPATH=routing_capture/inject`
  and `FPM_ROUTING_STAGE=B`, see the exact command below;
- `python3 fpm_collect.py --port 20380 --output … --detail-output …` (ZMQ SUB);
- `send_requests.py` to drive traffic.

Worker command (reproduces the golden `tp4_ep4` engine args — note **no**
`--enable-return-routed-experts`, since the monolithic backend ignores it and our
hook is independent):

```bash
DYN_DISCOVERY_BACKEND=file DYN_REQUEST_PLANE=tcp DYN_EVENT_PLANE=zmq \
DYN_FILE_KV=$RUN_DIR/discovery DYN_NAMESPACE=dynamo \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
DYN_FORWARDPASS_METRIC_PORT=20380 DYN_SYSTEM_PORT=8081 \
PYTHONPATH=routing_capture/inject:$PYTHONPATH \
FPM_ROUTING_STAGE=B FPM_ROUTING_OUT=$RUN_DIR/routing FPM_ROUTING_FLUSH_EVERY=500 \
python3 -m dynamo.vllm \
  --model /workspace/models/Qwen3.6-35B-A3B \
  --gpu-memory-utilization 0.9 --tensor-parallel-size 4 \
  --skip-mm-profiling --limit-mm-per-prompt '{"image":0,"video":0}' \
  --generation-config vllm --enable-expert-parallel \
  --discovery-backend file --request-plane tcp --event-plane zmq \
  --kv-events-config '{"enable_kv_cache_events": false}'
```

`sitecustomize.py` (in `routing_capture/inject/`, on `PYTHONPATH`) imports
`fpm_routing_capture` when `FPM_ROUTING_STAGE` is set; Python loads it at
interpreter startup in **every** spawned TP worker. `enable_return_routed_experts`
is **not** passed (the official path is dead on the monolithic kernel).

Analysis: `python3 routing_capture/analyze_skew_residual.py --run-dir $RUN_DIR`.

Gotchas learned:
- Model load is fast (~20s for 26 shards from `/workspace`); the ~210s startup is
  `torch.compile` (~55s) + CUDA-graph capture of 102 sizes — compute, not I/O.
  A different model path does not help.
- The Dynamo frontend **rejects (503 `overload`)** rather than queues once the
  worker hits its in-flight cap (~22 reqs). Drive at concurrency **≤ ~16** to keep
  the pipe full without 503s.

## Open items — RESOLVED on the GPU machine

1. `dynamo.vllm` **does** pass `--enable-return-routed-experts` through to vLLM
   (`AsyncEngineArgs` flag; reaches `enable_return_routed_experts=True`). But it is
   a **no-op on the monolithic MoE backend** (see Reality check above) — so Stage A
   as documented captures nothing; we recompute routing via `select_experts`.
2. Bind point: the official capturer binds in `gpu_worker.py:init_routed_experts_capturer`
   (cache init) **before** `compile_or_warm_up_model → capture_model()`. Our hook
   wraps `GPUModelRunner.capture_model` (resolves topology + allocates the counter)
   and `MoERunner._apply_quant_method` (the recompute), both installed before graph
   capture.

## Output format

Per worker process, per step:
- raw per-(layer) `[num_experts]` counts → sidecar (npz/parquet), keyed by
  `(counter_id/step, dp_rank)` to align with `fpm_metrics_phase.csv`.
- derived **per-rank** skew scalars (`cv, gini, entropy, expert_utilization`) →
  a companion CSV column set (deterministic post-process from raw counts).

Compute skew at **per-rank** granularity (grouped-GEMM is per-device); raw global
counts + the EP/DP mapping are sufficient to derive per-rank, except under EPLB
(dynamic slot mapping) where slot/rank assignment must also be logged.

## Results & verdict (measured 2026-06-23, B300, qwen36 tp4_ep4)

Runs: `routing_capture/run_routing_stack.sh` → `routing_runs/stageB2` (fixed
ISL 2048/4096, conc 16) and `stageB3` (varied ISL 128-8000, conc 18, 320 reqs);
analysis `routing_capture/analyze_final.py`. 622 outlier-trimmed mixed steps.

1. **Routing is captured cleanly** on mixed steps (count invariant `Σexpert =
   tokens·top_k` exact = 1.000; all 256 experts populated).
2. **Per-rank skew is real but nearly constant**: busiest-rank overload
   `max/mean ≈ 1.35×` (so the heaviest EP rank does ~35% more expert work than
   average — the model's intrinsic expert-popularity imbalance), but it varies
   only **±1.8% step-to-step** (CV-across-steps = 0.018). It self-averages over
   the hundreds-to-thousands of tokens routed per step.
3. **The residual reproduces** the golden ~0.21: coarse-binned within-group
   clean-GPU-time CV = **0.228**. Regressing latency on the aggregate FPM
   features explains R²=0.77 (gpu) leaving a 22%-of-mean residual.
4. **Skew does NOT explain the residual**: corr(residual, skew) = **+0.14-0.16,
   R² ≈ 0.02** on clean GPU time, and ≈0 on scheduler wall-time. Skew accounts
   for **~2%** of the within-group residual.

> **Verdict: NEGATIVE — per-rank routing skew is not the driver of the 0.21
> mixed-step residual.** It is too stable step-to-step (±1.8%) to be a useful
> continuous feature, and explains only ~2% of the residual. **Reconsider the
> RF+skew bet before investing.** The residual is dominated by other unrecorded
> per-step factors (prefill-chunk boundaries, KV fragmentation, kernel
> wave-quantization/autotune, CUDA-graph replay jitter) and — on wall-time —
> scheduler/host noise (wall-time residual CV 0.71, far above the 0.23 compute
> residual). Caveat: single model/hardware (Qwen3.6-35B-A3B, B300, monolithic
> FlashInfer-TRTLLM MoE); a model whose MoE kernel is genuinely load-imbalance
> sensitive could differ.

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
