# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
import re
import math
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any

import yaml
import pandas as pd

from aiconfigurator.eval.utils import run_stream, find_newest_subdir, mkdir_p, write_json, parse_disagg_start_script
from aiconfigurator.eval.service import ServiceManager
from aiconfigurator.eval.benchmarks import get as get_bench

from aiconfigurator.eval.gpu import GPUWatcher
from aiconfigurator.eval.plots.pareto import ParetoPlot
from aiconfigurator.eval.plots.timeseries import plot_gpu_timeseries

LOG = logging.getLogger(__name__)

@dataclass
class EvalConfig:
    mode: str
    service_dir: str
    start_script: str
    port: int
    health_timeout_s: int
    coldstart_wait_s: int
    no_generate: bool
    gpu_monitor: bool
    nvml_interval_s: float
    bench_concurrency: List[int]
    runs: int
    artifact_root: str
    cli_args: Any
    post_health_delay_s: int
    bench_runner: str
    bench_backend: Optional[str]
    # k8s
    k8s_image_pull_token: str
    k8s_enabled: bool
    k8s_namespace: str
    k8s_deploy_file: str
    k8s_engine_cm_name: str
    k8s_frontend_selector: str
    k8s_cr_name: str
    k8s_frontend_name_regex: str
    k8s_wait_workers_ready: bool
    k8s_context: str
    k8s_delete_on_stop: bool
    k8s_pf_kind: str
    k8s_pf_name: str
    k8s_wait_timeout_s: int
    k8s_bench_in_pod: bool


class Pipeline:
    def __init__(self, cfg: EvalConfig):
        self.cfg = cfg
        self.service: Optional[ServiceManager] = None
        self.last_config_dir: Optional[Path] = None
        self.art_root: Optional[Path] = None
        self._gpu_watcher = None
        self._gpu_csv: Optional[Path] = None

    def _generate_configs(self) -> Path:
        """
        Call existing CLI.
        """
        from aiconfigurator.cli import main as cli_runner

        args = self.cfg.cli_args
        save_dir = getattr(args, "save_dir", None)
        if not save_dir:
            raise ValueError("--save_dir is required for eval to pick artifacts.")

        pre_existing = set(p.name for p in Path(save_dir).glob("*") if p.is_dir())

        LOG.info("Generating configs via `cli`...")
        t0 = time.time()
        rc = cli_runner.main(args)
        if rc not in (None, 0):
            raise RuntimeError(f"`aiconfigurator cli` returned rc={rc}")

        base = Path(save_dir)
        new_dirs = [p for p in base.glob("*") if p.is_dir() and p.name not in pre_existing]
        result_dir = max(new_dirs, key=lambda p: p.stat().st_mtime) if new_dirs else find_newest_subdir(base)
        if not result_dir:
            raise FileNotFoundError("No new result folder found in save_dir.")
        LOG.info("Config generated in %.1fs: %s", time.time() - t0, result_dir)
        return result_dir

    def _copy_backend_configs(self, run_dir: Path, dest_root: Path) -> Dict[str, Path]:
        src_root = run_dir / "backend_configs"
        if not src_root.exists():
            raise FileNotFoundError(f"{src_root} does not exist")

        if self.cfg.k8s_enabled:
            mode = self.cfg.mode
            src_mode = src_root / mode
            if not src_mode.exists():
                raise FileNotFoundError(f"{src_mode} does not exist")
            cm_dir = run_dir / "k8s_engine_cm"
            if cm_dir.exists():
                shutil.rmtree(cm_dir)
            cm_dir.mkdir(parents=True, exist_ok=True)
            needed = []
            if mode == "agg":
                needed = ["agg_config.yaml"]
            else:
                needed = ["prefill_config.yaml", "decode_config.yaml"]
            copied = {}
            for name in needed:
                s = src_mode / name
                if not s.exists():
                    LOG.warning("Engine config missing: %s", s)
                    continue
                d = cm_dir / name
                shutil.copy2(s, d)
                copied[name] = d
            if not copied:
                raise FileNotFoundError(f"No engine YAML copied for mode={mode} under {src_mode}")

            # kubectl create configmap ... --from-file=<cm_dir> -o yaml --dry-run=client | kubectl apply -f -
            create_cmd = ["kubectl"]
            if self.cfg.k8s_context:
                create_cmd += ["--context", self.cfg.k8s_context]
            create_cmd += [
                "-n", self.cfg.k8s_namespace,
                "create", "configmap", self.cfg.k8s_engine_cm_name,
                "--from-file", str(cm_dir),
                "-o", "yaml", "--dry-run=client",
            ]
            LOG.info("Creating/Updating ConfigMap via dry-run+apply: %s", " ".join(create_cmd))
            dry = subprocess.run(create_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if dry.returncode != 0:
                LOG.error("ConfigMap dry-run failed: %s", dry.stderr)
                raise RuntimeError("kubectl create configmap dry-run failed")
            apply_cmd = ["kubectl"]
            if self.cfg.k8s_context:
                apply_cmd += ["--context", self.cfg.k8s_context]
            apply_cmd += ["apply", "-f", "-"]
            app = subprocess.run(apply_cmd, input=dry.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if app.returncode != 0:
                LOG.error("ConfigMap apply failed: %s", app.stderr)
                raise RuntimeError("kubectl apply -f - failed")
            LOG.info("ConfigMap '%s' updated with %d file(s).", self.cfg.k8s_engine_cm_name, len(copied))
            return {"k8s_engine_cm_dir": cm_dir, **copied}

        copied: Dict[str, Path] = {}
        for mode in ("disagg", "agg"):
            src = src_root / mode
            if src.exists():
                dst = dest_root / mode
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
                copied[mode] = dst
                LOG.info("Copied backend configs: %s -> %s", src, dst)
            else:
                LOG.info("No '%s' backend config in %s (skip)", mode, src_root)
        if not copied:
            raise FileNotFoundError("No backend config folders found to copy.")
        return copied

    def _ensure_art_root(self, save_dir: Path, run_name: str) -> Path:
        base = Path(self.cfg.artifact_root) if self.cfg.artifact_root else save_dir
        out = base / "eval_runs" / run_name
        mkdir_p(out)
        return out

    def _read_max_batch_size(self, service_dir: Path, mode: str) -> Optional[int]:
        cfg_path = service_dir / ("agg/agg_config.yaml" if mode == "agg" else "disagg/decode_config.yaml")
        if not cfg_path.exists():
            LOG.warning("YAML not found for auto concurrency: %s", cfg_path)
            return None
        try:
            with cfg_path.open() as f:
                y = yaml.safe_load(f) or {}
            mbs = y.get("max_batch_size")
            if isinstance(mbs, int) and mbs >= 1:
                return mbs
            LOG.warning("max_batch_size not found or invalid in %s", cfg_path)
            return None
        except Exception as e:
            LOG.warning("Failed to read %s: %s", cfg_path, e)
            return None

    def _snap_to_grid(self, target: float, g: int, lo: int, hi: int) -> int:
        if lo > hi:
            return hi

        down = int(math.floor(target / g) * g)
        up   = int(math.ceil(target / g) * g)

        cands = []
        if lo <= down <= hi and down > 0:
            cands.append(down)
        if lo <= up <= hi and up > 0 and up != down:
            cands.append(up)

        if cands:
            cands.sort(key=lambda x: (abs(x - target), -x))
            return cands[0]

        first_ge_lo = int(math.ceil(lo / g) * g)
        if lo <= first_ge_lo <= hi:
            return first_ge_lo

        return lo

    def _auto_concurrency_values(self, max_bs: int) -> List[int]:
        K = 6
        if max_bs <= 1:
            return [1]

        if max_bs >= 32:
            g = 8
        elif max_bs >= 8:
            g = 4
        elif max_bs >= 4:
            g = 2
        else:
            g = 1

        fracs = [1/8, 1/4, 1/2, 3/4]
        pts = [1]
        n_int = len(fracs)

        for i, f in enumerate(fracs):
            lo = pts[-1] + 1
            hi = max_bs - (n_int - (i + 1))
            if lo > hi:
                lo = hi
            target = f * max_bs
            v = self._snap_to_grid(target, g, lo, hi)
            if v <= pts[-1]:
                v = min(hi, pts[-1] + 1)
            pts.append(int(v))

        pts.append(max_bs)

        out = []
        for v in pts:
            if not out or v > out[-1]:
                out.append(int(v))

        if len(out) > K:
            mids = out[1:-1]
            need = K - 2
            idxs = [round(j * (len(mids) - 1) / (need - 1)) for j in range(need)]
            out = [out[0]] + [mids[i] for i in idxs] + [out[-1]]

        while len(out) < K:
            ins = min(out[-1] - 1, out[-2] + g if len(out) >= 2 else 2)
            if ins > out[-2]:
                out.insert(-1, ins)
            else:
                break

        return out[:K]

    def _start_service(self, service_dir: Path, log_file: Path) -> "ServiceManager":
        if self.cfg.k8s_enabled:
            from aiconfigurator.eval.service import K8sServiceManager
            deploy = self._find_k8s_deploy_yaml(self.last_config_dir or service_dir)
            cr_name_yaml, ns_yaml = self._parse_cr_meta(deploy)
            cr_name = self.cfg.k8s_cr_name or cr_name_yaml or ""
            namespace = self.cfg.k8s_namespace
            if ns_yaml and ns_yaml != namespace:
                LOG.warning("Namespace mismatch: CLI=%s, YAML=%s. Using YAML namespace.", namespace, ns_yaml)
                namespace = ns_yaml
            import re as _re
            fe_regex = self.cfg.k8s_frontend_name_regex or (rf"^{_re.escape(cr_name)}-.*-frontend-.*$" if cr_name else "frontend")
            expected_model_id = str(getattr(self.cfg.cli_args, "served_model_name", "") or "")
            sm = K8sServiceManager(
                namespace=namespace,
                deploy_yaml=deploy,
                port=self.cfg.port,
                frontend_selector=self.cfg.k8s_frontend_selector,
                frontend_name_regex=fe_regex,
                cr_name=cr_name or "",
                context=self.cfg.k8s_context,
                pf_kind=self.cfg.k8s_pf_kind,
                pf_name=self.cfg.k8s_pf_name,
                delete_on_stop=self.cfg.k8s_delete_on_stop,
                wait_timeout_s=self.cfg.k8s_wait_timeout_s,
                wait_workers_ready=self.cfg.k8s_wait_workers_ready,
                expected_model_id=expected_model_id,
                image_pull_token=self.cfg.k8s_image_pull_token,
            )
            sm.start(log_path=log_file, cold_wait_s=self.cfg.coldstart_wait_s)
            sm.wait_healthy(timeout_s=self.cfg.health_timeout_s)
            return sm
        else:
            from aiconfigurator.eval.service import ServiceManager
            start_rel = self.cfg.start_script.strip() or ("disagg/node_0_run.sh" if self.cfg.mode == "disagg" else "agg/node_0_run.sh")
            sm = ServiceManager(
                workdir=service_dir,
                start_cmd=["bash", start_rel],
                port=self.cfg.port,
            )
            sm.start(log_path=log_file, cold_wait_s=self.cfg.coldstart_wait_s)
            sm.wait_healthy(timeout_s=self.cfg.health_timeout_s)
            return sm

    def _find_k8s_deploy_yaml(self, root: Path) -> Path:
        # Priority: CLI override > backend_configs/<mode>/k8s_deploy.yaml > service_dir/<mode>/k8s_deploy.yaml
        if self.cfg.k8s_deploy_file:
            p = Path(self.cfg.k8s_deploy_file)
            if not p.exists():
                raise FileNotFoundError(f"--k8s-deploy-file not found: {p}")
            return p
        cands = [
            root / "backend_configs" / self.cfg.mode / "k8s_deploy.yaml",
            Path(self.cfg.service_dir) / self.cfg.mode / "k8s_deploy.yaml",
        ]
        for p in cands:
            if p.exists():
                return p
        raise FileNotFoundError(f"k8s_deploy.yaml not found in {cands}")

    def _parse_cr_meta(self, deploy_yaml: Path) -> tuple[Optional[str], Optional[str]]:
        try:
            with open(deploy_yaml) as f:
                docs = list(yaml.safe_load_all(f))
            for d in docs:
                if isinstance(d, dict) and d.get("kind") == "DynamoGraphDeployment":
                    meta = d.get("metadata", {}) or {}
                    return meta.get("name"), meta.get("namespace")
        except Exception as e:
            LOG.warning("Failed to parse CR meta from %s: %s", deploy_yaml, e)
        return None, None

    def _collect_gpu_once(self, where: Path) -> Dict[str, Any]:
        """Quick NVML snapshot before benchmark for worker sanity check."""
        # Lazy import since monitoring might be disabled
        from .gpu import quick_nvml_snapshot
        snap = quick_nvml_snapshot()
        write_json(where / "gpu_snapshot_prebench.json", snap)
        return snap

    def _load_optimal_configs(self, config_dir: Path, target_tpot: Optional[float] = None) -> Dict[str, pd.DataFrame]:
        """Load optimal configuration data from saved aiconfigurator results with TPOT filtering."""
        optimal_configs = {}

        # Try to load from CSV files first (more complete data)
        agg_pareto_path = config_dir / "agg_pareto.csv"
        disagg_pareto_path = config_dir / "disagg_pareto.csv"

        if agg_pareto_path.exists():
            try:
                agg_pareto = pd.read_csv(agg_pareto_path)
                if not agg_pareto.empty:
                    best_agg = self._get_best_config_under_tpot_constraint(agg_pareto, target_tpot)
                    if not best_agg.empty:
                        optimal_configs['agg'] = best_agg
                        LOG.info(f"Loaded optimal agg config: {best_agg['tokens/s/gpu'].iloc[0]:.2f} tokens/s/gpu, "
                                 f"TPOT: {best_agg.get('tpot', [None]).iloc[0]} ms")
                    else:
                        LOG.warning("No agg config found that meets TPOT constraint")
            except Exception as e:
                LOG.warning(f"Failed to load agg pareto data: {e}")

        if disagg_pareto_path.exists():
            try:
                disagg_pareto = pd.read_csv(disagg_pareto_path)
                if not disagg_pareto.empty:
                    best_disagg = self._get_best_config_under_tpot_constraint(disagg_pareto, target_tpot)
                    if not best_disagg.empty:
                        optimal_configs['disagg'] = best_disagg
                        LOG.info(f"Loaded optimal disagg config: {best_disagg['tokens/s/gpu'].iloc[0]:.2f} tokens/s/gpu, "
                                 f"TPOT: {best_disagg.get('tpot', [None]).iloc[0]} ms")
                    else:
                        LOG.warning("No disagg config found that meets TPOT constraint")
            except Exception as e:
                LOG.warning(f"Failed to load disagg pareto data: {e}")

        return optimal_configs

    def _get_best_config_under_tpot_constraint(self, pareto_df: pd.DataFrame, target_tpot: Optional[float]) -> pd.DataFrame:
        """Get the best configuration that meets TPOT constraint, similar to CLI logic."""
        if pareto_df.empty:
            return pd.DataFrame()

        # If no TPOT constraint, return the best overall configuration
        if target_tpot is None:
            best_config = pareto_df.loc[pareto_df['tokens/s/gpu'].idxmax()].to_frame().T
            LOG.info("No TPOT constraint specified, using best overall configuration")
            return best_config

        # Filter configurations that meet TPOT constraint
        if 'tpot' not in pareto_df.columns:
            LOG.warning("TPOT column not found in pareto data, using best overall configuration")
            return pareto_df.loc[pareto_df['tokens/s/gpu'].idxmax()].to_frame().T

        # Find configurations that meet the TPOT constraint
        candidate_configs = pareto_df[pareto_df['tpot'] <= target_tpot].copy()

        if not candidate_configs.empty:
            # Among valid candidates, pick the one with highest tokens/s/gpu
            best_config = candidate_configs.loc[candidate_configs['tokens/s/gpu'].idxmax()].to_frame().T
            LOG.info(f"Found {len(candidate_configs)} configs meeting TPOT <= {target_tpot}ms, "
                     f"selected best with {best_config['tokens/s/gpu'].iloc[0]:.2f} tokens/s/gpu")
            return best_config
        else:
            LOG.warning(f"No config found with TPOT <= {target_tpot}ms, using best overall configuration")
            return pareto_df.loc[pareto_df['tokens/s/gpu'].idxmax()].to_frame().T

    def _convert_optimal_config_to_plot_format(self, config_df: pd.DataFrame, config_type: str) -> pd.DataFrame:
        """Convert optimal configuration DataFrame to format expected by ParetoPlot."""
        try:
            plot_df = pd.DataFrame()

            if 'tokens/s/user' in config_df.columns:
                plot_df['output_token_throughput_per_user_avg'] = config_df['tokens/s/user']

            if 'tokens/s/gpu' in config_df.columns:
                plot_df['output_token_throughput_avg'] = config_df['tokens/s/gpu']

            if 'concurrency' in config_df.columns:
                plot_df['load_label'] = config_df['concurrency'].astype(str)
            elif 'bs' in config_df.columns:
                plot_df['load_label'] = config_df['bs'].astype(str)
            else:
                plot_df['load_label'] = f"{config_type}_optimal"

            LOG.debug(f"Converted optimal {config_type} config to plot format: {plot_df.to_dict()}")
            return plot_df

        except Exception as e:
            LOG.warning(f"Failed to convert optimal {config_type} config to plot format: {e}")
            return pd.DataFrame()

    def _run_benchmark(self, art_dir: Path, *, url: str, isl: int, osl: int, concurrency: List[int]) -> Path:
        """
        Dispatch benchmark execution based on cfg.bench_runner:
        - "genai-perf": use the default implementation
        - "bench-serving": call bench_serving_runner locally, or delegate to
            service.run_benchmark_in_pod when running inside a k8s Pod
        """
        runner = (self.cfg.bench_runner or "genai-perf").lower()

        # ---------- Resolve model / tokenizer from CLI args (consistent precedence) ----------
        args = self.cfg.cli_args
        model_path = str(getattr(args, "model_path", "") or "")
        served_model_name = str(getattr(args, "served_model_name", "") or "")
        model_cli = str(getattr(args, "model", "") or "")

        # Model identifier precedence:
        #   --model > served_model_name > basename(model_path) > model identifier
        model = served_model_name or (Path(model_path).name if model_path else "") or model_cli

        # Tokenizer precedence:
        #   --tokenizer-path > --model_path > model identifier
        tok_override = str(getattr(args, "tokenizer_path", "") or "")
        tokenizer = tok_override or model_path or model

        # Warn if tokenizer looks like a host-local path that does not exist
        try:
            looks_like_path = tokenizer and ("/" in tokenizer or tokenizer.startswith("."))
            if looks_like_path and not Path(tokenizer).exists():
                LOG.warning("Specified tokenizer looks like a local path but does not exist: %s", tokenizer)
        except Exception:
            pass

        bench_dir = art_dir / "bench"
        bench_dir.mkdir(parents=True, exist_ok=True)

        # K8s in-pod execution: delegate to Service and copy results back
        if self.cfg.k8s_enabled and self.cfg.k8s_bench_in_pod:
            LOG.info("Running benchmark in k8s Pod: runner=%s backend=%s", runner, self.cfg.bench_backend)
            return self.service.run_benchmark_in_pod(
                art_dir=art_dir,
                model=model,
                tokenizer=tokenizer,
                isl=int(isl),
                osl=int(osl),
                conc_list=list(map(int, concurrency or [])),
                runner=runner,
                bench_backend=self.cfg.bench_backend,
            )

        # Local execution (including k8s non in-pod): call the selected runner directly
        if runner == "bench-serving":
            run_fn = get_bench("bench_serving")["run"]
            cfg = {
                "base_folder": str(art_dir),
                "result_folder": "bench",
                "name": "bench",
                "url": url,
                "model": model,
                "tokenizer": tokenizer,
                "input_sequence_length": int(isl),
                "output_sequence_length": int(osl),
                "concurrency": list(map(int, concurrency or [])),
                "backend": (self.cfg.bench_backend or "sglang-oai-chat"),
                "dataset_name": "random",
                "seed": int(getattr(self.cfg.cli_args, "seed", 42)),
                "warmup_requests": 0,
            }
            LOG.info("Running local bench_serving: %s", cfg)
            run_fn(cfg)  # Output is written into bench_dir
            return bench_dir

        # Default: genai-perf
        run_fn = get_bench("genai_perf")["run"]
        cfg = {
            "base_folder": str(art_dir),
            "result_folder": "bench",
            "name": "bench",
            "url": url,
            "model": model,
            "tokenizer": tokenizer,
            "input_sequence_length": int(isl),
            "output_sequence_length": int(osl),
            "concurrency": list(map(int, concurrency or [])),
        }
        LOG.info("Running local genai-perf: %s", cfg)
        run_fn(cfg)
        return bench_dir

    def _analyze_and_plot(
        self,
        art_dir: Path,
        bench_dir: Path,
        workers_info: Dict[str, int],
        mode: str,
        gpu_monitor_enabled: bool,
        gpu_csv: Optional[Path],
        optimal_configs: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> None:
        """Parse benchmark outputs (genai-perf or bench-serving) and create plots."""
        # 1) Parser selection: honor cfg.bench_runner; on failure, fall back automatically
        runner = (self.cfg.bench_runner or "genai-perf").lower()
        parse_fn = None
        try:
            if runner == "bench-serving":
                parse_fn = get_bench("bench_serving")["parse"]
            else:
                parse_fn = get_bench("genai_perf")["parse"]
            df = parse_fn(bench_dir)
        except FileNotFoundError as e:
            LOG.warning("Primary parser (%s) failed: %s. Falling back to alternative parser...", runner, e)
            alt = "genai_perf" if runner == "bench-serving" else "bench_serving"
            df = get_bench(alt)["parse"](bench_dir)
            LOG.info("Fallback parser '%s' succeeded.", alt)

        # 2) Export rollup CSV
        out_csv = art_dir / "bench_summary.csv"
        df.to_csv(out_csv, index=False)
        LOG.info("Saved summary CSV to %s", out_csv)

        # Extract GPU count based on mode
        if mode == "disagg":
            p_workers = int(workers_info.get("PREFILL_WORKERS", 0) or 0)
            d_workers = int(workers_info.get("DECODE_WORKERS", 0) or 0)
            p_gpu = int(workers_info.get("PREFILL_GPU", 0) or 0)
            d_gpu = int(workers_info.get("DECODE_GPU", 0) or 0)
            total_gpus = p_workers * p_gpu + d_workers * d_gpu
            legend = f"disagg_{p_workers}p({p_gpu} gpu){d_workers}d({d_gpu} gpu)"
        else:
            # For agg mode, extract GPU count from config
            total_gpus = self._get_agg_gpu_count(workers_info)
            legend = f"agg_{total_gpus}gpu"

        # Validate GPU count
        if total_gpus <= 0:
            LOG.warning("Total GPUs computed as 0; skip per-GPU normalization.")
            total_gpus = None

        x_metric = "output_token_throughput_per_user::avg"
        y_metric = "output_token_throughput::avg"
        if f"{x_metric.split('::')[0]}_avg" not in df.columns:
            x_metric = "request_throughput::avg"
            LOG.warning("Per-user throughput missing; fallback to request_throughput::avg for X-axis.")

        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111)

        p = ParetoPlot(
            x_metric=x_metric,
            y_metric=y_metric,
            merge=True,
            num_gpus=total_gpus if total_gpus else None,
            plot_label=legend,
            show_cc_label=True,
            expand_x=True,
        )
        p.add_series("bench", df)

        # Add optimal configuration points if available
        if optimal_configs:
            for config_type, config_df in optimal_configs.items():
                if not config_df.empty:
                    optimal_point_df = self._convert_optimal_config_to_plot_format(config_df, config_type)
                    if not optimal_point_df.empty:
                        p.add_optimal_point(config_type.capitalize(), optimal_point_df)
                        LOG.info(f"Added optimal {config_type} point to plot")

        p.render(
            ax,
            title="Throughput(per gpu)/Throughput(per user)",
            x_label="Throughput (per user)",
            y_label="Throughput (per gpu)",
        )
        fig.savefig(art_dir / "pareto.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        LOG.info("Saved plot: %s", art_dir / "pareto.png")

        if gpu_monitor_enabled and gpu_csv and gpu_csv.exists():
            from .plots.gpu_timeseries import plot_gpu_timeseries_bokeh
            out_html = art_dir / "gpu_timeseries.html"
            plot_gpu_timeseries_bokeh(gpu_csv, out_html, title="GPU Utilization (%)")
            LOG.info("Saved plot: %s", out_html)

    def _get_agg_gpu_count(self, workers_info: Dict[str, int]) -> int:
        if self.cfg.k8s_enabled:
            try:
                base = (self.last_config_dir / "backend_configs" / "agg")
                cfg_path = base / "agg_config.yaml"
                if cfg_path.exists():
                    with cfg_path.open() as f:
                        y = yaml.safe_load(f) or {}
                    tp = int(y.get("tensor_parallel_size", 1) or 1)
                    pp = int(y.get("pipeline_parallel_size", 1) or 1)
                    if tp * pp > 0:
                        return tp * pp
            except Exception as e:
                LOG.warning("Failed to read TP/PP for agg from engine yaml: %s", e)
            try:
                deploy = self._find_k8s_deploy_yaml(Path(self.cfg.service_dir))
                with open(deploy) as f:
                    docs = list(yaml.safe_load_all(f))
                for d in docs:
                    services = (d or {}).get("spec", {}).get("services", {})
                    aw = services.get("TRTLLMWorker", {})
                    if aw:
                        rep = int(aw.get("replicas", 0) or 0)
                        gpu = int(str(((aw.get("resources", {}) or {}).get("limits", {}) or {}).get("gpu", "1")).strip('"') or 1)
                        if rep * gpu > 0:
                            return rep * gpu
            except Exception as e:
                LOG.warning("Failed to parse agg gpu from CR: %s", e)
        return workers_info.get("AGG_GPU_COUNT", 1)

    def _extract_workers_from_start_script(self, service_dir: Path) -> Dict[str, int]:
        """
        For disagg, read disagg/node_0_run.sh and extract worker/GPU info.
        For agg, extract GPU count from agg_config.yaml.
        """
        if self.cfg.k8s_enabled:
            # In k8s mode, try to infer from k8s_deploy.yaml; fallback to config YAMLs
            try:
                deploy = self._find_k8s_deploy_yaml(self.last_config_dir or service_dir)
                with open(deploy) as f:
                    docs = list(yaml.safe_load_all(f))
                # Heuristic: look for services named *PrefillWorker / *DecodeWorker
                vals = {
                    "PREFILL_GPU": 0, "PREFILL_WORKERS": 0,
                    "DECODE_GPU": 0,  "DECODE_WORKERS": 0,
                    "AGG_WORKERS": 0, "AGG_GPU_COUNT": 0,
                }
                for d in docs:
                    spec = (d or {}).get("spec", {})
                    services = spec.get("services", {})
                    # disagg workers
                    pw = services.get("TRTLLMPrefillWorker", {})
                    dw = services.get("TRTLLMDecodeWorker", {})
                    aw = services.get("TRTLLMWorker", {})
                    if pw:
                        vals["PREFILL_WORKERS"] = int(pw.get("replicas", 0) or 0)
                        vals["PREFILL_GPU"] = int(str(((pw.get("resources", {}) or {}).get("limits", {}) or {}).get("gpu", "0")).strip('"') or 0)
                    if dw:
                        vals["DECODE_WORKERS"] = int(dw.get("replicas", 0) or 0)
                        vals["DECODE_GPU"] = int(str(((dw.get("resources", {}) or {}).get("limits", {}) or {}).get("gpu", "0")).strip('"') or 0)
                    if aw:
                        vals["AGG_WORKERS"] = int(aw.get("replicas", 0) or 0)
                LOG.info("Parsed workers from k8s_deploy.yaml: %s", vals)
                if vals["PREFILL_WORKERS"] or vals["DECODE_WORKERS"] or vals["AGG_WORKERS"]:
                    return vals
            except Exception as e:
                LOG.warning("Failed to parse k8s deploy for workers: %s", e)
            # Fallback (agg only)
            if self.cfg.mode != "disagg":
                agg_gpu_count = self._extract_agg_gpu_count_from_config(Path(self.cfg.service_dir))
                return {
                    "PREFILL_GPU": -1, "PREFILL_WORKERS": 0,
                    "DECODE_GPU": -1,  "DECODE_WORKERS": 0,
                    "AGG_WORKERS": 1,  "AGG_GPU_COUNT": agg_gpu_count,
                }
            # Last fallback
            return {"PREFILL_GPU": 0, "PREFILL_WORKERS": 0, "DECODE_GPU": 0, "DECODE_WORKERS": 0}

        if self.cfg.mode == "disagg":
            start_rel = self.cfg.start_script.strip() or "disagg/node_0_run.sh"
            script = service_dir / start_rel
            vals = parse_disagg_start_script(script)
            LOG.info("Parsed workers from %s: %s", script, vals)
            return vals
        else:
            # For agg mode, extract GPU count from config
            agg_gpu_count = self._extract_agg_gpu_count_from_config(service_dir)
            return {
                "PREFILL_GPU": -1,
                "PREFILL_WORKERS": 0,
                "DECODE_GPU": -1,
                "DECODE_WORKERS": 0,
                "AGG_WORKERS": 1,
                "AGG_GPU_COUNT": agg_gpu_count,
            }

    def _extract_agg_gpu_count_from_config(self, service_dir: Path) -> int:
        """Extract GPU count from agg config file (TP * PP for TRT-LLM)."""
        try:
            config_path = service_dir / "agg" / "agg_config.yaml"
            if not config_path.exists():
                LOG.warning(f"Agg config not found: {config_path}")
                return 1

            with config_path.open() as f:
                config = yaml.safe_load(f) or {}

            # For TRT-LLM, GPU count is TP * PP (DP is handled through TP)
            tp = config.get("tensor_parallel_size", 1)
            pp = config.get("pipeline_parallel_size", 1)
            gpu_count = tp * pp

            LOG.info(f"Extracted agg GPU count (TRT-LLM): TP={tp} * PP={pp} = {gpu_count}")
            return gpu_count

        except Exception as e:
            LOG.warning(f"Failed to extract agg GPU count: {e}")
            return 1

    def stop_service(self):
        if self.service:
            self.service.stop()
            LOG.info("Service stopped.")

    def run(self, run_name: str) -> int:
        """
        Procedures:
          1) (optional) generate configs
          2) start service + health check
          3) (optional) NVML snapshot + GPU watcher
          4) run benchmark using args.isl/args.osl
          5) analyze + plots
        """
        args = self.cfg.cli_args
        save_dir = Path(getattr(args, "save_dir", None) or "")
        if not save_dir:
            raise ValueError("--save_dir is required")

        # 1) generate configs
        if not self.cfg.no_generate:
            self.last_config_dir = self._generate_configs()
        else:
            self.last_config_dir = find_newest_subdir(save_dir)
            if not self.last_config_dir:
                raise FileNotFoundError(f"No runs in {save_dir} and --no-generate was set.")

        # 2) copy configs to dynamo trtllm folder
        service_dir = Path(self.cfg.service_dir)
        _ = self._copy_backend_configs(self.last_config_dir, service_dir)

        # Load optimal configurations from the saved results with TPOT filtering
        target_tpot = getattr(args, "tpot", None)
        optimal_configs = self._load_optimal_configs(self.last_config_dir, target_tpot)

        # Determine concurrency
        if self.cfg.bench_concurrency and len(self.cfg.bench_concurrency) > 0:
            conc_list = list(self.cfg.bench_concurrency)
            LOG.info("Using provided benchmark concurrency: %s", conc_list)
        else:
            # In k8s mode read from backend_configs; local mode read from service_dir
            bs_base = (self.last_config_dir / "backend_configs") if self.cfg.k8s_enabled else service_dir
            mbs = self._read_max_batch_size(bs_base, self.cfg.mode)
            if not mbs:
                # Safe fallback if YAML missing
                conc_list = [1, 2, 4, 8, 16, 32]
                LOG.warning("Auto concurrency: max_batch_size not found -> fallback %s", conc_list)
            else:
                conc_list = self._auto_concurrency_values(int(mbs))
                LOG.info("Auto concurrency from max_batch_size=%s -> %s", mbs, conc_list)

        # Prepare log path
        log_dir = Path(getattr(args, "save_dir")).resolve() / "log"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{run_name}_{self.cfg.mode}_p{self.cfg.port}.log"

        # 3) start + health
        self.service = self._start_service(service_dir, log_file)

        # Artifacts root
        self.art_root = self._ensure_art_root(save_dir, run_name)

        # Worker info from script (disagg) or default (agg)
        workers_info = self._extract_workers_from_start_script(service_dir)
        write_json(self.art_root / "workers_extracted.json", workers_info)

        # GPU monitoring (conditional)
        if self.cfg.gpu_monitor and not self.cfg.k8s_enabled:
            LOG.info("GPU monitor enabled")
            from .gpu import GPUWatcher
            self._gpu_csv = self.art_root / "gpu_stats.csv"
            self._gpu_watcher = GPUWatcher(interval_s=self.cfg.nvml_interval_s, out_csv=self._gpu_csv)
            self._collect_gpu_once(self.art_root)
            self._gpu_watcher.start()
        else:
            if self.cfg.k8s_enabled and self.cfg.gpu_monitor:
                LOG.warning("K8s mode: local NVML monitor is disabled (it only monitors local GPUs).")
            else:
                LOG.info("GPU monitor disabled: skipping NVML sampling and timeseries.")
            self._gpu_csv = None

        # Benchmark tokens
        isl = int(getattr(args, "isl", 0) or 0) or 1024
        osl = int(getattr(args, "osl", 0) or 0) or 128
        LOG.info("Benchmark tokens: isl=%d osl=%d", isl, osl)

        # Extra wait after health OK, before running benchmarks
        LOG.info("Health OK. Waiting %ds before starting benchmark...", self.cfg.post_health_delay_s)
        import time as _time
        _time.sleep(max(0, int(self.cfg.post_health_delay_s)))

        # 4) benchmarking
        try:
            base_url = self.service.base_url()
            for i in range(self.cfg.runs):
                tag = f"{run_name}_r{i+1}"
                art_dir = self.art_root / tag
                mkdir_p(art_dir)
                LOG.info("Run %d/%d -> %s (concurrency=%s)", i + 1, self.cfg.runs, art_dir, conc_list)
                bench_dir = self._run_benchmark(art_dir, url=base_url, isl=isl, osl=osl, concurrency=conc_list)
                self._analyze_and_plot(
                    art_dir=art_dir,
                    bench_dir=bench_dir,
                    workers_info=workers_info,
                    mode=self.cfg.mode,
                    gpu_monitor_enabled=self.cfg.gpu_monitor,
                    gpu_csv=self._gpu_csv,
                    optimal_configs=optimal_configs,
                )
        finally:
            if self._gpu_watcher:
                self._gpu_watcher.stop()

        return 0
