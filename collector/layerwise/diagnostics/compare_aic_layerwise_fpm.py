#!/usr/bin/env python3
"""Compare AIC vLLM layerwise predictions against FPM phase rows.

This diagnostic is intentionally narrow: it uses an explicit layerwise CSV as
the layerwise database, reuses the repo's real communication/MoE tables, and
calls ``VLLMBackend`` phase estimators directly for FPM-comparable shapes.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from aiconfigurator.sdk import interpolation
from aiconfigurator.sdk.backends import vllm_backend
from aiconfigurator.sdk.backends.vllm_backend import VLLMBackend
from aiconfigurator.sdk.config import RuntimeConfig
from aiconfigurator.sdk.operations.layerwise import (
    _interpolate_metric_2d,
    _uniform_bool_metric,
    _uniform_float_metric,
    _uniform_str_metric,
    load_layerwise_data,
)
from aiconfigurator.sdk.perf_database import PerfDatabase


def _trimmed_mean(values: list[float]) -> float:
    """Return a trimmed mean, dropping one min and max when possible."""

    if len(values) < 3:
        return float(statistics.fmean(values))
    return float(statistics.fmean(sorted(values)[1:-1]))


def _aggregate(values: list[float], mode: str) -> float:
    """Aggregate latency samples."""

    if mode == "median":
        return float(statistics.median(values))
    if mode == "mean":
        return float(statistics.fmean(values))
    if mode == "trimmed_mean":
        return _trimmed_mean(values)
    raise ValueError(f"unsupported aggregation: {mode}")


def _load_fpm(path: Path) -> tuple[dict[tuple[int, int], list[float]], dict[tuple[int, float], list[float]]]:
    """Load context/decode bins from an FPM phase CSV."""

    context: dict[tuple[int, int], list[float]] = defaultdict(list)
    decode: dict[tuple[int, float], list[float]] = defaultdict(list)
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            latency = float(row["latency_ms"])
            if row["phase"] == "context":
                context[(int(row["ctx_requests"]), int(row["ctx_tokens"]))].append(latency)
            elif row["phase"] == "decode":
                decode[(int(row["decode_requests"]), float(row["mean_decode_kv_tokens"]))].append(latency)
    return context, decode


def _match_decode(
    decode: dict[tuple[int, float], list[float]],
    batch_size: int,
    past_kv: int,
    mode: str,
    max_distance: float,
    pool_forward_window: float,
) -> tuple[str, list[float], str] | None:
    """Return the requested decode bin/window for a batch and KV target."""

    exact = (batch_size, float(past_kv))
    if mode != "pooled" and exact in decode:
        return f"{float(past_kv):.3f}", decode[exact], "exact"
    if mode == "exact":
        return None
    if mode == "pooled":
        lower = float(past_kv)
        upper = lower + pool_forward_window
        pooled = [
            (kv, values)
            for (bs, kv), values in decode.items()
            if bs == batch_size and lower <= kv <= upper
        ]
        if not pooled:
            return None
        pooled.sort(key=lambda item: item[0])
        values = [latency for _, samples in pooled for latency in samples]
        first_kv = pooled[0][0]
        last_kv = pooled[-1][0]
        label = f"{first_kv:.3f}" if first_kv == last_kv else f"{first_kv:.3f}..{last_kv:.3f}"
        return label, values, "pooled"
    candidates = [
        (abs(kv - float(past_kv)), kv, values)
        for (bs, kv), values in decode.items()
        if bs == batch_size
    ]
    if not candidates:
        return None
    distance, kv, values = min(candidates, key=lambda item: (item[0], item[1]))
    if distance > max_distance:
        return None
    return f"{kv:.3f}", values, "nearest"


class _Config:
    """Minimal model config consumed by VLLMBackend phase estimators."""

    def __init__(self, *, tp_size: int, moe_tp_size: int, moe_ep_size: int):
        self.tp_size = tp_size
        self.pp_size = 1
        self.moe_tp_size = moe_tp_size
        self.moe_ep_size = moe_ep_size
        self.attention_dp_size = 1
        self.moe_quant_mode = None
        self.workload_distribution = "power_law"
        self.moe_backend = None
        self.enable_eplb = False


class _Model:
    """Minimal model object consumed by VLLMBackend phase estimators."""

    def __init__(
        self,
        *,
        model_path: str,
        tp_size: int,
        moe_tp_size: int,
        moe_ep_size: int,
        num_layers: int,
        hidden_size: int,
        topk: int = 0,
        num_experts: int = 0,
        moe_inter_size: int = 0,
    ):
        self.model_path = model_path
        self.config = _Config(tp_size=tp_size, moe_tp_size=moe_tp_size, moe_ep_size=moe_ep_size)
        self._num_layers = num_layers
        self._hidden_size = hidden_size
        self._nextn = 0
        self._topk = topk
        self._num_experts = num_experts
        self._moe_inter_size = moe_inter_size


class _LayerwiseDatabase:
    """Adapter that serves supplied layerwise rows plus real comm/MoE tables."""

    def __init__(self, layerwise_csv: Path, real_database: PerfDatabase):
        self.layerwise = load_layerwise_data(str(layerwise_csv))
        self.real_database = real_database
        self._extracted_metrics_cache: dict[Any, Any] = {}

    def query_layerwise_detail(
        self,
        model: str,
        phase: str,
        tp_size: int,
        batch_size: int,
        seq_len: int,
        seq_len_kv_cache: int = 0,
    ) -> dict[str, Any]:
        """Return an exact layerwise detail row."""

        model_data = self.layerwise[model.lower()][phase.upper()][tp_size]
        if phase.upper() == "CTX":
            if seq_len in model_data and seq_len_kv_cache in model_data[seq_len]:
                return self._normalize_detail(model_data[seq_len][seq_len_kv_cache])
            if len(model_data) < 2:
                raise KeyError((model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache))
            result = interpolation.interp_2d_linear(
                seq_len,
                seq_len_kv_cache,
                model_data,
                self._extracted_metrics_cache,
            )
            result["rms_latency"] = _interpolate_metric_2d(
                seq_len,
                seq_len_kv_cache,
                model_data,
                "rms_latency",
                self._extracted_metrics_cache,
            )
        elif batch_size in model_data and seq_len in model_data[batch_size]:
            return self._normalize_detail(model_data[batch_size][seq_len])
        else:
            result = interpolation.interp_2d_linear(
                batch_size,
                seq_len,
                model_data,
                self._extracted_metrics_cache,
            )
            result["rms_latency"] = _interpolate_metric_2d(
                batch_size,
                seq_len,
                model_data,
                "rms_latency",
                self._extracted_metrics_cache,
            )
        result["includes_moe"] = _uniform_bool_metric(model_data, "includes_moe")
        result["layer_type"] = _uniform_str_metric(model_data, "layer_type")
        result["layer_index"] = _uniform_float_metric(model_data, "layer_index")
        result["measured_layer_count"] = _uniform_float_metric(model_data, "measured_layer_count", 1.0)
        result["layer_multiplier"] = _uniform_float_metric(model_data, "layer_multiplier")
        return self._normalize_detail(result)

    def _normalize_detail(self, result: Any) -> dict[str, Any]:
        """Return layerwise detail fields matching ``PerfDatabase`` output."""

        if not isinstance(result, dict):
            result = {"latency": float(result), "energy": 0.0}
        out: dict[str, Any] = {
            "latency": float(result["latency"]),
            "energy": float(result.get("energy", 0.0)),
            "rms_latency": float(result.get("rms_latency", 0.0)),
            "rms_kernel_count": float(result.get("rms_kernel_count", 0.0)),
            "includes_moe": bool(result.get("includes_moe", False)),
        }
        if result.get("layer_type") not in (None, ""):
            out["layer_type"] = str(result["layer_type"])
        for metric in ("layer_index", "measured_layer_count", "layer_multiplier", "physical_gpus"):
            if result.get(metric) not in (None, ""):
                out[metric] = float(result[metric])
        for metric in ("latency_source", "measurement_mode", "attribution_target", "vllm_config_hash"):
            if result.get(metric) not in (None, ""):
                out[metric] = str(result[metric])
        if isinstance(result.get("components"), list):
            out["components"] = [dict(component) for component in result["components"] if isinstance(component, dict)]
        return out

    def query_custom_allreduce(self, *args: Any, **kwargs: Any) -> Any:
        """Proxy TP allreduce queries to the real database."""

        return self.real_database.query_custom_allreduce(*args, **kwargs)

    def query_allreduce_rms(self, *args: Any, **kwargs: Any) -> Any:
        """Proxy fused allreduce+RMS queries to the real database."""

        return self.real_database.query_allreduce_rms(*args, **kwargs)

    def query_nccl(self, *args: Any, **kwargs: Any) -> Any:
        """Proxy NCCL collective queries to the real database."""

        return self.real_database.query_nccl(*args, **kwargs)

    def query_moe(self, *args: Any, **kwargs: Any) -> Any:
        """Proxy MoE op queries to the real database."""

        return self.real_database.query_moe(*args, **kwargs)


def _model_defaults(model: str, tp: int, moe_tp: int, ep: int) -> _Model:
    """Return known model metadata for current diagnostics."""

    if model == "Qwen/Qwen3-32B":
        return _Model(
            model_path=model,
            tp_size=tp,
            moe_tp_size=moe_tp,
            moe_ep_size=ep,
            num_layers=64,
            hidden_size=5120,
        )
    if model == "Qwen/Qwen3.6-35B-A3B":
        return _Model(
            model_path=model,
            tp_size=tp,
            moe_tp_size=moe_tp,
            moe_ep_size=ep,
            num_layers=40,
            hidden_size=2048,
            topk=8,
            num_experts=256,
            moe_inter_size=256,
        )
    raise ValueError(f"unknown model defaults for {model!r}")


def compare(
    *,
    layerwise_csv: Path,
    fpm_csv: Path,
    model_name: str,
    tp: int,
    moe_tp: int,
    ep: int,
    output: Path,
    aggregation: str,
    decode_past_kv: int,
    decode_osl: int,
    decode_match: str,
    max_decode_kv_distance: float,
    decode_pool_forward_window: float,
    include_mixed: bool,
) -> list[dict[str, Any]]:
    """Write AIC-vs-FPM comparison rows."""

    real_db = PerfDatabase("b300_sxm", "vllm", "0.20.1", systems_root="src/aiconfigurator/systems")
    database = _LayerwiseDatabase(layerwise_csv, real_db)
    model = _model_defaults(model_name, tp, moe_tp, ep)
    backend = VLLMBackend()
    old_use_layerwise = vllm_backend._USE_LAYERWISE
    vllm_backend._USE_LAYERWISE = True
    context, decode = _load_fpm(fpm_csv)
    rows: list[dict[str, Any]] = []
    try:
        for (batch_size, ctx_tokens), samples in sorted(context.items()):
            if batch_size != 1:
                continue
            fpm_ms = _aggregate(samples, aggregation)
            try:
                latency, _, _ = backend._run_context_phase(
                    model,
                    database,
                    RuntimeConfig(),
                    batch_size=batch_size,
                    isl=ctx_tokens,
                    prefix=0,
                )
            except KeyError:
                continue
            aic_ms = float(sum(latency.values()))
            rows.append({
                "model": model_name,
                "tp": tp,
                "moe_tp": moe_tp,
                "ep": ep,
                "phase": "ctx",
                "shape": f"ctx{ctx_tokens}",
                "fpm_ms": fpm_ms,
                "aic_ms": aic_ms,
                "error_pct": ((aic_ms / fpm_ms) - 1.0) * 100.0,
                "fpm_samples": len(samples),
                "fpm_match": "exact",
            })
        for batch_size in sorted({bs for bs, _ in decode}):
            matched = _match_decode(
                decode,
                batch_size,
                decode_past_kv,
                decode_match,
                max_decode_kv_distance,
                decode_pool_forward_window,
            )
            if matched is None:
                continue
            kv_label, samples, match = matched
            fpm_ms = _aggregate(samples, aggregation)
            try:
                latency, _, _ = backend._run_generation_phase(
                    model,
                    database,
                    RuntimeConfig(),
                    batch_size=batch_size,
                    beam_width=1,
                    isl=decode_past_kv,
                    osl=decode_osl,
                    stride=32,
                )
            except KeyError:
                continue
            aic_ms = float(sum(latency.values()))
            rows.append({
                "model": model_name,
                "tp": tp,
                "moe_tp": moe_tp,
                "ep": ep,
                "phase": "gen",
                "shape": f"bs{batch_size}_past{decode_past_kv}",
                "fpm_ms": fpm_ms,
                "aic_ms": aic_ms,
                "error_pct": ((aic_ms / fpm_ms) - 1.0) * 100.0,
                "fpm_samples": len(samples),
                "fpm_match": f"{match}:{kv_label}",
            })
        if include_mixed:
            with fpm_csv.open(newline="") as f:
                for fpm_row in csv.DictReader(f):
                    if fpm_row["phase"] != "mixed":
                        continue
                    fpm_ms = float(fpm_row["latency_ms"])
                    ctx_tokens = int(fpm_row["ctx_tokens"])
                    gen_tokens = int(fpm_row["decode_requests"])
                    mean_decode_kv = float(fpm_row["mean_decode_kv_tokens"])
                    try:
                        aic_ms, _, _, _ = backend._get_mix_step_latency(
                            model,
                            database,
                            RuntimeConfig(),
                            ctx_tokens=ctx_tokens,
                            gen_tokens=gen_tokens,
                            isl=int(round(mean_decode_kv)),
                            osl=1,
                            prefix=0,
                        )
                    except (KeyError, ValueError):
                        continue
                    rows.append({
                        "model": model_name,
                        "tp": tp,
                        "moe_tp": moe_tp,
                        "ep": ep,
                        "phase": "mixed",
                        "shape": f"ctx{ctx_tokens}_gen{gen_tokens}_kv{mean_decode_kv:.3f}",
                        "fpm_ms": fpm_ms,
                        "aic_ms": float(aic_ms),
                        "error_pct": ((float(aic_ms) / fpm_ms) - 1.0) * 100.0,
                        "fpm_samples": 1,
                        "fpm_match": f"row:{fpm_row.get('counter_id', '')}",
                        "counter_id": fpm_row.get("counter_id", ""),
                        "ctx_tokens": ctx_tokens,
                        "ctx_requests": int(fpm_row["ctx_requests"]),
                        "decode_requests": gen_tokens,
                        "mean_decode_kv_tokens": mean_decode_kv,
                    })
    finally:
        vllm_backend._USE_LAYERWISE = old_use_layerwise

    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model",
        "tp",
        "moe_tp",
        "ep",
        "phase",
        "shape",
        "fpm_ms",
        "aic_ms",
        "error_pct",
        "fpm_samples",
        "fpm_match",
        "counter_id",
        "ctx_tokens",
        "ctx_requests",
        "decode_requests",
        "mean_decode_kv_tokens",
    ]
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layerwise", type=Path, required=True)
    parser.add_argument("--fpm", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--moe-tp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--aggregation", choices=("median", "mean", "trimmed_mean"), default="trimmed_mean")
    parser.add_argument("--decode-past-kv", type=int, default=4096)
    parser.add_argument("--decode-osl", type=int, default=2)
    parser.add_argument(
        "--decode-match",
        choices=("exact", "nearest", "pooled"),
        default="nearest",
        help="Decode KV matching mode. Pooled compares against a forward steady-state KV window.",
    )
    parser.add_argument("--max-decode-kv-distance", type=float, default=4.0)
    parser.add_argument("--decode-pool-forward-window", type=float, default=6.0)
    parser.add_argument(
        "--include-mixed",
        action="store_true",
        help="Also compare mixed prefill+decode FPM scheduler rows with _get_mix_step_latency.",
    )
    return parser


def main() -> None:
    """Run the diagnostic comparison."""

    args = _build_parser().parse_args()
    compare(
        layerwise_csv=args.layerwise,
        fpm_csv=args.fpm,
        model_name=args.model,
        tp=args.tp,
        moe_tp=args.moe_tp,
        ep=args.ep,
        output=args.output,
        aggregation=args.aggregation,
        decode_past_kv=args.decode_past_kv,
        decode_osl=args.decode_osl,
        decode_match=args.decode_match,
        max_decode_kv_distance=args.max_decode_kv_distance,
        decode_pool_forward_window=args.decode_pool_forward_window,
        include_mixed=args.include_mixed,
    )


if __name__ == "__main__":
    main()
