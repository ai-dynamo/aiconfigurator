#!/usr/bin/env python3
import argparse
import csv
import sys
from contextlib import redirect_stdout
from pathlib import Path

from aiconfigurator.cli.api import cli_estimate

DEFAULT_INPUT = Path(__file__).resolve().parents[2] / "src/aiconfigurator/systems/silicon_sample.csv"
PARALLEL_FIELDS = ("tp_size", "pp_size", "attention_dp_size", "moe_tp_size", "moe_ep_size")
AGG_FIELDS = ("batch_size", *PARALLEL_FIELDS)
DISAGG_FIELDS = tuple(
    f"{stage}_{field}" for stage in ("prefill", "decode") for field in (*PARALLEL_FIELDS, "batch_size", "num_workers")
)


def _estimate(row: dict[str, str]) -> tuple[float, float]:
    fields = AGG_FIELDS if row["mode"] == "agg" else DISAGG_FIELDS
    args = dict(
        model_path=row["model_path"],
        system_name=row["system"],
        mode=row["mode"],
        backend_name=row["backend"],
        backend_version=row["backend_version"] or None,
        isl=int(row["isl"]),
        osl=int(row["osl"]),
        gemm_quant_mode=row["gemm_quant_mode"] or None,
        moe_quant_mode=row["moe_quant_mode"] or None,
        **{field: int(row[field]) for field in fields if row.get(field)},
    )
    with redirect_stdout(sys.stderr):
        result = cli_estimate(**args)
    return result.ttft, result.tpot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--aic-output-prefix", default="aic")
    args = parser.parse_args()
    ttft_col = f"{args.aic_output_prefix}_predicted_ttft_ms"
    tpot_col = f"{args.aic_output_prefix}_predicted_tpot_ms"
    with args.input.open(newline="") as f:
        reader = csv.DictReader(f)
        writer = csv.DictWriter(sys.stdout, fieldnames=[*(reader.fieldnames or []), ttft_col, tpot_col])
        writer.writeheader()
        for row in reader:
            try:
                row[ttft_col], row[tpot_col] = (f"{v:.6f}" for v in _estimate(row))
            except Exception as e:
                print(f"{row.get('id', '<unknown>')}: {e}", file=sys.stderr)
                row[ttft_col] = row[tpot_col] = ""
            writer.writerow(row)


if __name__ == "__main__":
    main()
