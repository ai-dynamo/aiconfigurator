#!/usr/bin/env python3
"""Analyze DSv4 MegaMoE inter-size sweep results into a comm-path perf table.

The input is produced by running the DeepGEMM MegaMoE collector repeatedly with
fixed routing and different intermediate sizes.  The analysis keeps the raw
curve and derives a conservative plateau estimate:

    fused_ms(intermediate) ~= fixed + max(comm_path, compute_path(intermediate))

The plateau is therefore a measured comm-plus-tail term for the fixed traffic
case.  DeepGEMM's own combine-reduction formula is reported separately as an
analytic tail estimate; it is not treated as a measured kernel sub-range.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


SUMMARY_NAME = "summary_samples.jsonl"
COMM_PATH_PERF_FILENAME = "dsv4_megamoe_comm_path_perf.txt"
ROUTE_CONSTANT_FIELDS = (
    "max_recv_edges",
    "max_remote_recv_edges",
    "max_deepgemm_nvlink_bytes",
    "max_remote_only_nvlink_bytes",
)
PERF_FIELDNAMES = [
    "framework",
    "version",
    "device",
    "op_name",
    "kernel_source",
    "hidden_size",
    "inter_size",
    "topk",
    "num_experts",
    "moe_ep_size",
    "routing_mode",
    "power_law_alpha",
    "num_tokens_per_rank",
    "comm_path_ms",
    "tail_ms",
    "comm_plus_tail_ms",
    "target_fused_ms",
    "plateau_num_points",
    "max_remote_only_nvlink_bytes",
    "max_deepgemm_nvlink_bytes",
    "num_samples",
    "source",
]


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _summary_path(intermediate_dir: Path) -> Path | None:
    candidates = sorted(intermediate_dir.glob(f"*/{SUMMARY_NAME}"))
    if candidates:
        return candidates[0]
    direct = intermediate_dir / SUMMARY_NAME
    return direct if direct.exists() else None


def _sample_payload_path(intermediate_dir: Path, tokens: int, sample_idx: int) -> Path | None:
    name = f"tokens_{tokens}_sample{sample_idx}.json"
    candidates = sorted(intermediate_dir.glob(f"*/{name}"))
    if candidates:
        return candidates[0]
    direct = intermediate_dir / name
    return direct if direct.exists() else None


def _parse_intermediate(path: Path) -> int:
    prefix = "intermediate_"
    if not path.name.startswith(prefix):
        raise ValueError(f"unexpected intermediate directory name: {path}")
    return int(path.name[len(prefix) :])


def _row_from_payload(intermediate_hidden: int, payload: dict[str, Any]) -> dict[str, Any]:
    config = payload["config"]
    aggregate = payload["aggregate"]
    return {
        "intermediate_hidden": intermediate_hidden,
        "tokens": int(config["num_tokens"]),
        "sample_idx": int(config.get("sample_idx", 0)),
        "max_t_fused_ms": float(aggregate["max_t_fused_ms"]),
        "max_recv_edges": int(aggregate["max_recv_edges"]),
        "max_remote_recv_edges": int(aggregate["max_remote_recv_edges"]),
        "max_deepgemm_nvlink_bytes": int(aggregate["max_deepgemm_nvlink_bytes"]),
        "max_remote_only_nvlink_bytes": int(aggregate["max_remote_only_nvlink_bytes"]),
        "reduction_us": float(aggregate["reduction_us"]),
        "reduction_tail_ms": float(aggregate["reduction_us"]) / 1000.0,
        "bottleneck_rank_by_time": int(aggregate["bottleneck_rank_by_time"]),
        "bottleneck_rank_by_deepgemm_bytes": int(aggregate["bottleneck_rank_by_deepgemm_bytes"]),
        "bottleneck_rank_by_remote_only_bytes": int(aggregate["bottleneck_rank_by_remote_only_bytes"]),
        "route_hash": _stable_hash(payload.get("routes")),
        "routing_hash": _stable_hash(payload.get("routing")),
    }


def _median(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(statistics.median(values))


def _load_rows(input_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for intermediate_dir in sorted(input_dir.glob("intermediate_*"), key=_parse_intermediate):
        intermediate_hidden = _parse_intermediate(intermediate_dir)
        summary = _summary_path(intermediate_dir)
        if summary is None:
            for payload_path in sorted(intermediate_dir.glob("*.json")):
                payload = json.loads(payload_path.read_text(encoding="utf-8"))
                if "config" in payload and "aggregate" in payload:
                    rows.append(_row_from_payload(intermediate_hidden, payload))
            continue
        for row in _read_jsonl(summary):
            tokens = int(row["tokens"])
            sample_idx = int(row["sample_idx"])
            sample_path = _sample_payload_path(intermediate_dir, tokens, sample_idx)
            route_hash = ""
            routing_hash = ""
            if sample_path is not None:
                try:
                    payload = json.loads(sample_path.read_text(encoding="utf-8"))
                    route_hash = _stable_hash(payload.get("routes"))
                    routing_hash = _stable_hash(payload.get("routing"))
                except json.JSONDecodeError:
                    route_hash = ""
                    routing_hash = ""
            rows.append(
                {
                    "intermediate_hidden": intermediate_hidden,
                    "tokens": tokens,
                    "sample_idx": sample_idx,
                    "max_t_fused_ms": float(row["max_t_fused_ms"]),
                    "max_recv_edges": int(row["max_recv_edges"]),
                    "max_remote_recv_edges": int(row["max_remote_recv_edges"]),
                    "max_deepgemm_nvlink_bytes": int(row["max_deepgemm_nvlink_bytes"]),
                    "max_remote_only_nvlink_bytes": int(row["max_remote_only_nvlink_bytes"]),
                    "reduction_us": float(row["reduction_us"]),
                    "reduction_tail_ms": float(row["reduction_us"]) / 1000.0,
                    "bottleneck_rank_by_time": int(row["bottleneck_rank_by_time"]),
                    "bottleneck_rank_by_deepgemm_bytes": int(row["bottleneck_rank_by_deepgemm_bytes"]),
                    "bottleneck_rank_by_remote_only_bytes": int(row["bottleneck_rank_by_remote_only_bytes"]),
                    "route_hash": route_hash,
                    "routing_hash": routing_hash,
                }
            )
    return rows


def _validate_fixed_traffic(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["tokens"], row["sample_idx"])].append(row)

    for (tokens, sample_idx), group_rows in sorted(grouped.items()):
        baseline = min(group_rows, key=lambda item: item["intermediate_hidden"])
        for row in group_rows:
            for field in ROUTE_CONSTANT_FIELDS:
                if row[field] != baseline[field]:
                    mismatches.append(
                        {
                            "tokens": tokens,
                            "sample_idx": sample_idx,
                            "intermediate_hidden": row["intermediate_hidden"],
                            "field": field,
                            "baseline": baseline[field],
                            "actual": row[field],
                        }
                    )
            for field in ("route_hash", "routing_hash"):
                if baseline[field] and row[field] and row[field] != baseline[field]:
                    mismatches.append(
                        {
                            "tokens": tokens,
                            "sample_idx": sample_idx,
                            "intermediate_hidden": row["intermediate_hidden"],
                            "field": field,
                            "baseline": baseline[field],
                            "actual": row[field],
                        }
                    )
    return mismatches


def _infer_plateaus(
    rows: list[dict[str, Any]],
    *,
    plateau_tol: float,
    target_intermediate_hidden: int,
) -> list[dict[str, Any]]:
    inferred: list[dict[str, Any]] = []
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["tokens"], row["sample_idx"])].append(row)

    for (tokens, sample_idx), group_rows in sorted(grouped.items()):
        ordered = sorted(group_rows, key=lambda item: item["intermediate_hidden"])
        target_rows = [row for row in ordered if row["intermediate_hidden"] == target_intermediate_hidden]
        min_fused_ms = min(float(row["max_t_fused_ms"]) for row in ordered)
        plateau_rows = [row for row in ordered if float(row["max_t_fused_ms"]) <= min_fused_ms * (1.0 + plateau_tol)]
        plateau_ms = _median([float(row["max_t_fused_ms"]) for row in plateau_rows])
        reduction_tail_ms = _median([float(row["reduction_tail_ms"]) for row in plateau_rows])
        inferred.append(
            {
                "tokens": tokens,
                "sample_idx": sample_idx,
                "plateau_tolerance_pct": plateau_tol * 100.0,
                "plateau_num_points": len(plateau_rows),
                "plateau_intermediate_hiddens": ",".join(str(row["intermediate_hidden"]) for row in plateau_rows),
                "comm_plus_tail_plateau_ms": plateau_ms,
                "analytic_reduction_tail_ms": reduction_tail_ms,
                "comm_path_minus_analytic_tail_ms": max(0.0, plateau_ms - reduction_tail_ms),
                "target_intermediate_hidden": target_intermediate_hidden,
                "target_deepgemm_latency_ms": _median([float(row["max_t_fused_ms"]) for row in target_rows]),
                "min_fused_ms": min_fused_ms,
                "max_remote_only_nvlink_bytes": int(ordered[0]["max_remote_only_nvlink_bytes"]),
                "max_deepgemm_nvlink_bytes": int(ordered[0]["max_deepgemm_nvlink_bytes"]),
                "route_hash": ordered[0]["route_hash"],
                "routing_hash": ordered[0]["routing_hash"],
            }
        )
    return inferred


def _aggregate_by_tokens(inferred_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in inferred_rows:
        grouped[int(row["tokens"])].append(row)

    aggregate_rows: list[dict[str, Any]] = []
    for tokens, rows in sorted(grouped.items()):
        aggregate_rows.append(
            {
                "tokens": tokens,
                "num_samples": len(rows),
                "comm_plus_tail_plateau_ms_median": _median(
                    [float(row["comm_plus_tail_plateau_ms"]) for row in rows]
                ),
                "analytic_reduction_tail_ms_median": _median(
                    [float(row["analytic_reduction_tail_ms"]) for row in rows]
                ),
                "comm_path_minus_analytic_tail_ms_median": _median(
                    [float(row["comm_path_minus_analytic_tail_ms"]) for row in rows]
                ),
                "target_intermediate_hidden": int(rows[0]["target_intermediate_hidden"]),
                "target_deepgemm_latency_ms_median": _median(
                    [float(row["target_deepgemm_latency_ms"]) for row in rows]
                ),
                "plateau_num_points_median": _median([float(row["plateau_num_points"]) for row in rows]),
                "max_remote_only_nvlink_bytes_median": _median(
                    [float(row["max_remote_only_nvlink_bytes"]) for row in rows]
                ),
                "max_deepgemm_nvlink_bytes_median": _median(
                    [float(row["max_deepgemm_nvlink_bytes"]) for row in rows]
                ),
            }
        )
    return aggregate_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_perf_csv(path: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PERF_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "framework": args.framework,
                    "version": args.backend_version,
                    "device": args.device_name,
                    "op_name": "dsv4_megamoe_comm_path",
                    "kernel_source": args.kernel_source,
                    "hidden_size": args.hidden_size,
                    "inter_size": row["target_intermediate_hidden"],
                    "topk": args.topk,
                    "num_experts": args.num_experts,
                    "moe_ep_size": args.moe_ep_size,
                    "routing_mode": args.routing_mode,
                    "power_law_alpha": args.power_law_alpha if args.routing_mode == "power-law" else "",
                    "num_tokens_per_rank": row["tokens"],
                    "comm_path_ms": row["comm_path_minus_analytic_tail_ms_median"],
                    "tail_ms": row["analytic_reduction_tail_ms_median"],
                    "comm_plus_tail_ms": row["comm_plus_tail_plateau_ms_median"],
                    "target_fused_ms": row["target_deepgemm_latency_ms_median"],
                    "plateau_num_points": row["plateau_num_points_median"],
                    "max_remote_only_nvlink_bytes": row["max_remote_only_nvlink_bytes_median"],
                    "max_deepgemm_nvlink_bytes": row["max_deepgemm_nvlink_bytes_median"],
                    "num_samples": row["num_samples"],
                    "source": args.source,
                }
            )


def analyze_comm_sweep(
    *,
    input_dir: Path,
    output_dir: Path,
    plateau_tolerance_pct: float = 5.0,
    target_intermediate_hidden: int = 3072,
    perf_output: Path | None = None,
    framework: str = "SGLang",
    backend_version: str = "unknown",
    device_name: str = "unknown",
    kernel_source: str = "DeepGEMM_fp8_fp4_mega_moe",
    hidden_size: int = 7168,
    topk: int = 6,
    num_experts: int = 384,
    moe_ep_size: int = 8,
    routing_mode: str = "random",
    power_law_alpha: float = 1.01,
    source: str = "",
) -> dict[str, Any]:
    rows = _load_rows(input_dir)
    if not rows:
        raise ValueError(f"no compute-sweep rows found under {input_dir}")

    mismatches = _validate_fixed_traffic(rows)
    inferred_rows = _infer_plateaus(
        rows,
        plateau_tol=plateau_tolerance_pct / 100.0,
        target_intermediate_hidden=target_intermediate_hidden,
    )
    aggregate_rows = _aggregate_by_tokens(inferred_rows)

    args = argparse.Namespace(
        framework=framework,
        backend_version=backend_version,
        device_name=device_name,
        kernel_source=kernel_source,
        hidden_size=hidden_size,
        topk=topk,
        num_experts=num_experts,
        moe_ep_size=moe_ep_size,
        routing_mode=routing_mode,
        power_law_alpha=power_law_alpha,
        source=source,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    curves_path = output_dir / "curves.csv"
    inferred_path = output_dir / "inferred_comm_tail_by_sample.csv"
    aggregate_path = output_dir / "inferred_comm_tail_by_tokens.csv"
    validation_path = output_dir / "validation.json"
    _write_csv(curves_path, rows)
    _write_csv(inferred_path, inferred_rows)
    _write_csv(aggregate_path, aggregate_rows)
    if perf_output is not None:
        _write_perf_csv(perf_output, aggregate_rows, args)
    validation_path.write_text(
        json.dumps(
            {
                "num_curve_rows": len(rows),
                "num_fixed_traffic_mismatches": len(mismatches),
                "fixed_traffic_mismatches": mismatches,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    if mismatches:
        raise ValueError(f"fixed-traffic validation failed with {len(mismatches)} mismatches")

    return {
        "curves_path": str(curves_path),
        "inferred_by_sample_path": str(inferred_path),
        "inferred_by_tokens_path": str(aggregate_path),
        "perf_output": str(perf_output) if perf_output is not None else "",
        "validation_path": str(validation_path),
        "num_curve_rows": len(rows),
        "num_perf_rows": len(aggregate_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--plateau-tolerance-pct",
        type=float,
        default=5.0,
        help="Rows within this percentage of the minimum fused latency are treated as plateau candidates.",
    )
    parser.add_argument(
        "--target-intermediate-hidden",
        type=int,
        default=3072,
        help="Report raw DeepGEMM latency at this production intermediate size.",
    )
    parser.add_argument(
        "--perf-output",
        type=Path,
        default=None,
        help=f"Optional AIC perf CSV output, usually named {COMM_PATH_PERF_FILENAME}.",
    )
    parser.add_argument("--framework", default="SGLang")
    parser.add_argument("--backend-version", default="unknown")
    parser.add_argument("--device-name", default="unknown")
    parser.add_argument("--kernel-source", default="DeepGEMM_fp8_fp4_mega_moe")
    parser.add_argument("--hidden-size", type=int, default=7168)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--num-experts", type=int, default=384)
    parser.add_argument("--moe-ep-size", type=int, default=8)
    parser.add_argument("--routing-mode", choices=("random", "power-law"), default="random")
    parser.add_argument("--power-law-alpha", type=float, default=1.01)
    parser.add_argument("--source", default="")
    args = parser.parse_args()

    try:
        result = analyze_comm_sweep(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            plateau_tolerance_pct=args.plateau_tolerance_pct,
            target_intermediate_hidden=args.target_intermediate_hidden,
            perf_output=args.perf_output,
            framework=args.framework,
            backend_version=args.backend_version,
            device_name=args.device_name,
            kernel_source=args.kernel_source,
            hidden_size=args.hidden_size,
            topk=args.topk,
            num_experts=args.num_experts,
            moe_ep_size=args.moe_ep_size,
            routing_mode=args.routing_mode,
            power_law_alpha=args.power_law_alpha,
            source=args.source,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Wrote {result['curves_path']}")
    print(f"Wrote {result['inferred_by_sample_path']}")
    print(f"Wrote {result['inferred_by_tokens_path']}")
    if args.perf_output is not None:
        print(f"Wrote {result['perf_output']}")
    print(f"Wrote {result['validation_path']}")


if __name__ == "__main__":
    main()
