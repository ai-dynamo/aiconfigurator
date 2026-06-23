"""In-process FPM expert-routing capture for the Dynamo vLLM worker.

Captures per-(layer, step, rank) MoE expert routing alongside a per-step GPU-time
latency, so the within-group latency residual on mixed steps can be attributed to
per-rank routing skew. Runs inside every TP worker process via ``sitecustomize``
(PYTHONPATH injection); no code mount needed since we are already inside the
Dynamo vLLM image.

WHY NOT the documented ``enable_return_routed_experts`` / ``BaseRouter.set_capture_fn``
path: on B300/Blackwell this unquantized bf16 MoE selects the **FlashInfer-TRTLLM
monolithic** kernel (worker log: ``Using MoEPrepareAndFinalizeNoDPEPMonolithic``).
``MoERunner._apply_quant_method`` then takes the ``quant_method.is_monolithic``
branch -> ``apply_monolithic()`` routes *inside* the kernel and
``BaseRouter.select_experts`` is NEVER called, so the official capturer's GPU
buffer stays all-zero (verified: 100% of routing mass landed on expert 0). The
documented capture mechanism is silently bypassed by the monolithic backend.

WHAT WE DO INSTEAD (real APIs only): wrap ``MoERunner._apply_quant_method``,
which always receives ``router_logits`` and holds ``self.router``. We recompute
the routing with the model's own ``self.router.select_experts(hidden_states,
router_logits)`` (a pure function of the gate logits — identical routing the
monolithic kernel uses internally), and scatter_add the logical expert ids into
a preallocated ``[num_layers, num_experts]`` GPU counter keyed by
``layer.layer_id``. The op is enqueued INSIDE the CUDA graph (capture happens
before ``capture_model()``), so it replays on every graphed decode step. The
monolithic kernel still runs unchanged for latency; the only perturbation is the
extra (cheap, ~constant) topk recompute, which adds a near-constant offset and so
does not bias variance-based within-group attribution.

Per-rank skew: under TP (DP=1) the router runs redundantly on every rank, so the
logical ``topk_ids`` is the *global* routing on every rank. We capture the global
per-(layer) per-expert bincount and derive per-rank device load by slicing experts
by the contiguous EP map (expert e -> rank e // (num_experts // ep_size)). Raw
per-expert counts are persisted so the mapping can be re-derived offline.

Env:
  FPM_ROUTING_STAGE   A | B (required to activate)
  FPM_ROUTING_OUT     output dir for sidecars (default /tmp/fpm_routing)
  FPM_ROUTING_FLUSH_EVERY   steps between sidecar flushes (default 512)
  FPM_ROUTING_CAPTURE_RANKS  comma TP ranks to capture (default "0")
"""

from __future__ import annotations

import atexit
import logging
import os
import re
import sys

import numpy as np
import torch

logger = logging.getLogger("fpm_routing_capture")

STAGE = os.environ.get("FPM_ROUTING_STAGE", "").upper()
OUT_DIR = os.environ.get("FPM_ROUTING_OUT", "/tmp/fpm_routing")
FLUSH_EVERY = int(os.environ.get("FPM_ROUTING_FLUSH_EVERY", "512"))
_CAPTURE_RANKS = {
    int(x) for x in os.environ.get("FPM_ROUTING_CAPTURE_RANKS", "0").split(",") if x.strip()
}

_S = {
    "installed": False,
    "step": 0,
    "meta": [],
    "counts": [],
    "num_layers": None,
    "num_experts": None,
    "top_k": None,
    "tp_rank": None,
    "tp_size": None,
    "dp_rank": 0,
    "ep_size": None,
    "active": False,
    "gpu_counts": None,      # [num_layers, num_experts] int64 GPU per-step accumulator
    "flush_idx": 0,
    "hook_calls": 0,         # diagnostic: # of _apply_quant_method invocations seen
    "topology_ready": False,
}


def _log(msg: str) -> None:
    print(f"[fpm-routing][stage={STAGE}] {msg}", file=sys.stderr, flush=True)


def _layer_id_of(layer) -> int:
    lid = getattr(layer, "layer_id", None)
    if isinstance(lid, int):
        return lid
    name = getattr(layer, "layer_name", "") or getattr(layer, "prefix", "")
    m = re.search(r"layers\.(\d+)", str(name))
    return int(m.group(1)) if m else -1


def _extract_signature(scheduler_output) -> dict:
    """Replicate dynamo InstrumentedScheduler._extract_scheduled aggregates so the
    sidecar joins to fpm_metrics_phase.csv on identical aggregate inputs."""
    new_reqs = scheduler_output.scheduled_new_reqs
    cached = scheduler_output.scheduled_cached_reqs
    num_scheduled = scheduler_output.num_scheduled_tokens

    num_prefill = 0
    sum_prefill_tokens = 0
    sum_prefill_kv_tokens = 0
    decode_kv_vals = []

    for req in new_reqs:
        num_prefill += 1
        sum_prefill_tokens += int(num_scheduled.get(req.req_id, 0))
        sum_prefill_kv_tokens += int(getattr(req, "num_computed_tokens", 0))

    for i, req_id in enumerate(cached.req_ids):
        if cached.is_context_phase(req_id):
            num_prefill += 1
            sum_prefill_tokens += int(num_scheduled.get(req_id, 0))
            sum_prefill_kv_tokens += int(cached.num_computed_tokens[i])
        else:
            decode_kv_vals.append(int(cached.num_computed_tokens[i]))

    decode_requests = len(decode_kv_vals)
    mean_decode_kv = (sum(decode_kv_vals) / decode_requests) if decode_requests else 0.0
    total_tokens = int(sum(int(v) for v in num_scheduled.values()))
    return {
        "ctx_tokens": int(sum_prefill_tokens),
        "ctx_requests": int(num_prefill),
        "ctx_kv_tokens": int(sum_prefill_kv_tokens),
        "decode_requests": int(decode_requests),
        "mean_decode_kv_tokens": float(mean_decode_kv),
        "total_tokens": total_tokens,
    }


def _flush() -> None:
    if not _S["meta"]:
        return
    os.makedirs(OUT_DIR, exist_ok=True)
    rank = _S["tp_rank"]
    idx = _S["flush_idx"]
    counts = np.stack(_S["counts"], axis=0) if _S["counts"] else np.zeros((0, 0, 0), np.int32)
    meta = _S["meta"]
    path = os.path.join(OUT_DIR, f"routing_rank{rank}_part{idx:04d}.npz")
    np.savez_compressed(
        path,
        counts=counts,
        step=np.array([m["step"] for m in meta], dtype=np.int64),
        gpu_time_ms=np.array([m["gpu_time_ms"] for m in meta], dtype=np.float64),
        ctx_tokens=np.array([m["ctx_tokens"] for m in meta], dtype=np.int64),
        ctx_requests=np.array([m["ctx_requests"] for m in meta], dtype=np.int64),
        ctx_kv_tokens=np.array([m["ctx_kv_tokens"] for m in meta], dtype=np.int64),
        decode_requests=np.array([m["decode_requests"] for m in meta], dtype=np.int64),
        mean_decode_kv_tokens=np.array([m["mean_decode_kv_tokens"] for m in meta], dtype=np.float64),
        total_tokens=np.array([m["total_tokens"] for m in meta], dtype=np.int64),
        tp_rank=np.array([_S["tp_rank"]] * len(meta), dtype=np.int64),
        dp_rank=np.array([_S["dp_rank"]] * len(meta), dtype=np.int64),
    )
    _log(f"flushed {len(meta)} steps -> {path} (counts {counts.shape}, hook_calls={_S['hook_calls']})")
    _S["meta"] = []
    _S["counts"] = []
    _S["flush_idx"] += 1


def _write_manifest() -> None:
    if _S["tp_rank"] is None:
        return
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"manifest_rank{_S['tp_rank']}.txt")
    with open(path, "w") as f:
        for k in ("num_layers", "num_experts", "top_k", "tp_rank", "tp_size",
                  "dp_rank", "ep_size", "step", "hook_calls"):
            f.write(f"{k}={_S[k]}\n")


def _resolve_topology(runner) -> None:
    if _S["topology_ready"]:
        return
    from vllm.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )

    cfg = runner.vllm_config
    hf = cfg.model_config.hf_text_config
    _S["num_layers"] = int(hf.num_hidden_layers)
    _S["num_experts"] = int(getattr(hf, "num_experts", 0))
    _S["top_k"] = int(getattr(hf, "num_experts_per_tok", 0))
    _S["tp_rank"] = int(get_tensor_model_parallel_rank())
    _S["tp_size"] = int(get_tensor_model_parallel_world_size())
    _S["dp_rank"] = int(getattr(cfg.parallel_config, "data_parallel_rank", 0) or 0)
    _S["ep_size"] = int(
        _S["tp_size"] * max(1, int(getattr(cfg.parallel_config, "data_parallel_size", 1)))
    )
    _S["active"] = _S["tp_rank"] in _CAPTURE_RANKS
    _S["gpu_counts"] = torch.zeros(
        (_S["num_layers"], _S["num_experts"]), dtype=torch.int64, device="cuda"
    )
    _S["topology_ready"] = True
    _log(
        f"topology: layers={_S['num_layers']} experts={_S['num_experts']} top_k={_S['top_k']} "
        f"tp_rank={_S['tp_rank']}/{_S['tp_size']} dp_rank={_S['dp_rank']} ep_size={_S['ep_size']} "
        f"active={_S['active']}"
    )
    _write_manifest()


def _install():
    if _S["installed"] or STAGE not in ("A", "B"):
        return
    from vllm.v1.worker import gpu_model_runner as _gmr
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

    orig_capture = _gmr.GPUModelRunner.capture_model
    orig_execute = _gmr.GPUModelRunner.execute_model
    orig_aqm = MoERunner._apply_quant_method

    def patched_aqm(self, layer, hidden_states, router_logits, *args, **kwargs):
        # Recompute routing on the monolithic path (where select_experts is
        # otherwise skipped). In-graph scatter_add -> replays every decode step.
        if _S["active"] and _S["gpu_counts"] is not None:
            try:
                _S["hook_calls"] += 1
                input_ids = kwargs.get("input_ids", None)
                _w, topk_ids = self.router.select_experts(
                    hidden_states=hidden_states,
                    router_logits=router_logits,
                    input_ids=input_ids,
                )
                lid = _layer_id_of(layer)
                if 0 <= lid < _S["num_layers"]:
                    ids = topk_ids.reshape(-1).long().clamp_(0, _S["num_experts"] - 1)
                    _S["gpu_counts"][lid].scatter_add_(0, ids, torch.ones_like(ids))
            except Exception:
                if _S["hook_calls"] <= 2:
                    logger.exception("[fpm-routing] routing recompute failed")
        return orig_aqm(self, layer, hidden_states, router_logits, *args, **kwargs)

    def patched_capture(self):
        try:
            _resolve_topology(self)
        except Exception:
            logger.exception("[fpm-routing] topology resolution failed")
        return orig_capture(self)

    def patched_execute(self, scheduler_output, intermediate_tensors=None):
        if not (_S["active"] and _S["topology_ready"]):
            return orig_execute(self, scheduler_output, intermediate_tensors)
        total = int(sum(int(v) for v in scheduler_output.num_scheduled_tokens.values()))
        if total <= 0:
            return orig_execute(self, scheduler_output, intermediate_tensors)

        _S["gpu_counts"].zero_()  # per-step reset (host-enqueued before the forward)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        ret = orig_execute(self, scheduler_output, intermediate_tensors)
        end.record()
        end.synchronize()
        gpu_ms = float(start.elapsed_time(end))

        try:
            counts = _S["gpu_counts"].to("cpu").to(torch.int32).numpy()
            sig = _extract_signature(scheduler_output)
            _S["meta"].append({"step": _S["step"], "gpu_time_ms": gpu_ms, **sig})
            _S["counts"].append(counts)
        except Exception:
            logger.exception("[fpm-routing] count readout failed")
        _S["step"] += 1
        if len(_S["meta"]) >= FLUSH_EVERY:
            _flush()
            _write_manifest()
        return ret

    MoERunner._apply_quant_method = patched_aqm
    _gmr.GPUModelRunner.capture_model = patched_capture
    _gmr.GPUModelRunner.execute_model = patched_execute
    atexit.register(lambda: (_flush(), _write_manifest()))
    _S["installed"] = True
    _log(f"installed (MoERunner._apply_quant_method hook; capture_ranks={sorted(_CAPTURE_RANKS)}, "
         f"flush_every={FLUSH_EVERY}, out={OUT_DIR})")


try:
    _install()
except Exception:  # pragma: no cover
    logger.exception("[fpm-routing] install failed")
    _log("install FAILED (see traceback above)")
