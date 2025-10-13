# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
from datetime import datetime

LOG = logging.getLogger(__name__)

def configure_parser(parser: argparse.ArgumentParser) -> None:
    from aiconfigurator.cli.main import configure_parser as reuse_cli_parser
    reuse_cli_parser(parser)

    g = parser.add_argument_group("Eval pipeline")
    g.add_argument("--service-mode", choices=["disagg", "agg"], default="disagg",
                   help="Which service to start. Default: disagg")
    g.add_argument("--service-dir", type=str, default="/workspace/components/backends/trtllm",
                   help="Where backend folders (disagg/agg) are copied and service is started.")
    g.add_argument("--start-script", type=str, default="",
                   help="Optional override of start script path (relative to service-dir).")
    g.add_argument("--health-timeout-s", type=int, default=600,
                   help="Max seconds to wait for service ready.")
    g.add_argument("--coldstart-wait-s", type=int, default=10,
                   help="Extra seconds to wait after process spawn.")
    g.add_argument("--no-generate", action="store_true",
                   help="Skip running `aiconfigurator cli`; use an existing save_dir run.")
    g.add_argument("--run-name", type=str, default="",
                   help="Optional run label (folder name suffix).")
    g.add_argument("--runs", type=int, default=1,
                   help="Number of pipeline cycles to execute (same service).")
    g.add_argument("--keep-running", action="store_true",
                   help="Do not stop service after evaluation.")
    g.add_argument("--gpu-monitor", action="store_true",
                   help="Enable GPU monitoring (NVML) and timeseries HTML output.")
    g.add_argument("--nvml-interval-s", type=float, default=1.0,
                   help="GPU sampling interval seconds (used only when --gpu-monitor is set).")
    g.add_argument(
        "--benchmark-concurrency",
        type=int,
        nargs="+",
        help=(
            "Benchmark concurrency list. If omitted -> auto service_mode: "
            "read max_batch_size from backend YAML "
            "(agg: agg/agg_config.yaml; disagg: disagg/decode_config.yaml), "
            "then pick 6 values from 1..max (incl), roughly even and preferring multiples of 4/8 "
            "(e.g., max=20 -> 1,4,8,12,16,20). If provided, use provided cc list."
        ),
    )
    g.add_argument("--artifact-root", type=str, default="",
                   help="Optional base folder for eval outputs (default under save_dir).")
    g.add_argument("--venv-dir",  type=str, default="/workspace/aic",
                   help="uv venv path for aiperf")

    parser.epilog = (parser.epilog or "") + (
        "\n\nEVAL NOTES:\n"
        "\n\nEVAL NOTES:\n"
        "  • `eval` reuses all `cli` args for config generation.\n"
        "  • Health URLs are derived from --port as http://0.0.0.0:<port>/health and /v1/models.\n"
        "  • Use --gpu-monitor to enable NVML sampling and timeseries HTML; otherwise no monitoring is performed.\n"
    )
    parser.formatter_class = argparse.RawDescriptionHelpFormatter


def main(args) -> int:
    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "debug", False) else logging.INFO,
        format="%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s"
    )

    if not getattr(args, "save_dir", None):
        LOG.error("--save_dir is required for eval")
        return 2

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = args.run_name or f"{args.model}_{args.system}_{ts}"
    LOG.info("Eval start: run=%s  service_mode=%s  runs=%d", run_name, args.service_mode, args.runs)

    from aiconfigurator.eval.pipeline import Pipeline, EvalConfig
    port = int(getattr(args, "port", 8000))

    cfg = EvalConfig(
        service_mode=args.service_mode,
        service_dir=args.service_dir,
        start_script=args.start_script,
        venv_dir=args.venv_dir,
        port=port,
        health_timeout_s=args.health_timeout_s,
        coldstart_wait_s=args.coldstart_wait_s,
        no_generate=args.no_generate,
        gpu_monitor=bool(getattr(args, "gpu_monitor", False)),
        nvml_interval_s=args.nvml_interval_s,
        bench_concurrency=list(args.benchmark_concurrency or []),
        runs=args.runs,
        artifact_root=args.artifact_root or "",
        cli_args=args,
    )

    pipe = Pipeline(cfg)
    rc = pipe.run(run_name)
    LOG.info("Eval done: run=%s rc=%s", run_name, rc)

    if rc == 0 and not args.keep_running:
        LOG.info("Stopping service...")
        try:
            pipe.stop_service()
        except Exception as e:
            LOG.warning("Stop failed: %s", e)

    return rc
