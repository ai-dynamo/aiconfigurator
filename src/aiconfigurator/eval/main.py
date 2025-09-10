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
    g.add_argument("--mode", choices=["disagg", "agg"], default="disagg",
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
            "Benchmark concurrency list. If omitted -> auto mode: "
            "read max_batch_size from backend YAML "
            "(agg: agg/agg_config.yaml; disagg: disagg/decode_config.yaml), "
            "then pick 6 values from 1..max (incl), roughly even and preferring multiples of 4/8 "
            "(e.g., max=20 -> 1,4,8,12,16,20). If provided, use provided cc list."
        ),
    )
    g.add_argument("--artifact-root", type=str, default="",
                   help="Optional base folder for eval outputs (default under save_dir).")
    g.add_argument("--tokenizer-path", dest="tokenizer_path", type=str, default="",
                   help=("Override tokenizer path used by genai-perf. "
                         "Recommended in k8s mode where the served model is remote but the tokenizer is local."))
 
    # --- Kubernetes args ---
    gk = parser.add_argument_group("Kubernetes")
    gk.add_argument("--k8s", action="store_true",
                    help="Enable Kubernetes deployment mode.")
    gk.add_argument("--k8s-namespace", type=str, default="ets-dynamo",
                    help="Kubernetes namespace. Default: ets-dynamo")
    gk.add_argument("--k8s-deploy-file", type=str, default="",
                    help="Override path to k8s_deploy.yaml; if empty, auto-detect under backend_configs/<mode>/k8s_deploy.yaml.")
    gk.add_argument("--k8s-engine-cm-name", type=str, default="engine-configs",
                    help="ConfigMap name to store engine YAML(s) (Method 2).")
    gk.add_argument("--k8s-frontend-selector", type=str,
                    default="dynamo.nvidia.com/componentType=frontend",
                    help="Label selector to find frontend pod for port-forward (e.g. 'app=my-frontend').")
    gk.add_argument("--k8s-context", type=str, default="",
                    help="kubectl context to use (optional).")
    gk.add_argument("--k8s-cr-name", type=str, default="",
                    help="Override CR name in the deploy yaml; if empty will be parsed from the yaml.")
    gk.add_argument("--k8s-frontend-name-regex", type=str, default="",
                    help="Fallback regex to match frontend pod name; if empty, will be inferred as '^{CR_NAME}-.*-frontend-.*$' (case-insensitive).")
    gk.add_argument("--k8s-delete-on-stop", action="store_true",
                    help="Delete the deployed graph on stop.")
    gk.add_argument("--k8s-pf-kind", choices=["pod", "svc"], default="pod",
                    help="Resource kind to port-forward: pod or svc. Default: pod")
    gk.add_argument("--k8s-pf-name", type=str, default="",
                    help="Explicit resource name to port-forward; if empty, first Ready frontend pod is used.")
    gk.add_argument("--k8s-wait-timeout-s", type=int, default=900,
                    help="Max seconds to wait for pods to be Ready in k8s mode. Default: 900")

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
    LOG.info("Eval start: run=%s  mode=%s  runs=%d", run_name, args.mode, args.runs)

    from aiconfigurator.eval.pipeline import Pipeline, EvalConfig
    port = int(getattr(args, "port", 8000))

    cfg = EvalConfig(
        mode=args.mode,
        service_dir=args.service_dir,
        start_script=args.start_script,
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
        # k8s
        k8s_enabled=bool(getattr(args, "k8s", False)),
        k8s_namespace=args.k8s_namespace,
        k8s_deploy_file=args.k8s_deploy_file,
        k8s_engine_cm_name=args.k8s_engine_cm_name,
        k8s_frontend_selector=args.k8s_frontend_selector,
        k8s_cr_name=args.k8s_cr_name,
        k8s_frontend_name_regex=args.k8s_frontend_name_regex,
        k8s_context=(args.k8s_context or ""),
        k8s_delete_on_stop=bool(getattr(args, "k8s_delete_on_stop", False)),
        k8s_pf_kind=args.k8s_pf_kind,
        k8s_pf_name=args.k8s_pf_name or "",
        k8s_wait_timeout_s=args.k8s_wait_timeout_s,
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

