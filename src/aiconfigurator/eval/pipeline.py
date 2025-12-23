# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import yaml

from aiconfigurator.eval.benchmarks.genai_perf_runner import parse as parse_genai_perf
from aiconfigurator.eval.benchmarks.genai_perf_runner import run as run_genai_perf
from aiconfigurator.eval.plots.pareto import ParetoPlot
from aiconfigurator.eval.service import ServiceManager
from aiconfigurator.eval.utils import (
    find_newest_subdir,
    mkdir_p,
    parse_disagg_start_script,
    write_json,
)

LOG = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    service_mode: str
    venv_dir: str
    service_dir: str
    start_script: str
    port: int
    health_timeout_s: int
    coldstart_wait_s: int
    no_generate: bool
    gpu_monitor: bool
    nvml_interval_s: float
    bench_concurrency: list[int]
    runs: int
    artifact_root: str
    cli_args: Any


class Pipeline:
    def __init__(self, cfg: EvalConfig):
        self.cfg = cfg
        self.service: ServiceManager | None = None
        self.last_config_dir: Path | None = None
        self.art_root: Path | None = None
        self._gpu_watcher = None
        self._gpu_csv: Path | None = None

    def _generate_configs(self) -> Path:
        """
        Call existing CLI.
        """
        from aiconfigurator.cli import main as cli_runner

        args = self.cfg.cli_args
        args.service_mode = "default"
        save_dir = getattr(args, "save_dir", None)
        if not save_dir:
            raise ValueError("--save_dir is required for eval to pick artifacts.")

        pre_existing = set(p.name for p in Path(save_dir).glob("*") if p.is_dir())

        LOG.info("Generating configs via `cli`...")
        t0 = time.time()
        rc = cli_runner.main(args)
        if rc not in (None, 0):
            raise RuntimeError(f"`aiconfigurator cli default` returned rc={rc}")

        base = Path(save_dir)
        new_dirs = [p for p in base.glob("*") if p.is_dir() and p.name not in pre_existing]
        result_dir = max(new_dirs, key=lambda p: p.stat().st_mtime) if new_dirs else find_newest_subdir(base)
        if not result_dir:
            raise FileNotFoundError("No new result folder found in save_dir.")
        LOG.info("Config generated in %.1fs: %s", time.time() - t0, result_dir)
        return result_dir

    def _copy_backend_configs(self, run_dir: Path, dest_root: Path) -> dict[str, Path]:
        """
        Copy backend configs
        """
        copied: dict[str, Path] = {}

        def pick_src(service_mode: str) -> Path:
            src = run_dir / service_mode / "top1"
            if not src.exists():
                raise FileNotFoundError(f"Expected path not found for service_mode '{service_mode}': {src}")

            if service_mode == "agg":
                if not (src / "agg_config.yaml").exists():
                    LOG.warning("agg_config.yaml not found under %s", src)
            else:
                if not ((src / "decode_config.yaml").exists() or (src / "prefill_config.yaml").exists()):
                    LOG.warning("Neither decode_config.yaml nor prefill_config.yaml found under %s", src)
            return src

        for service_mode in ("disagg", "agg"):
            src = pick_src(service_mode)
            dst = dest_root / service_mode
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            copied[service_mode] = dst
            LOG.info("Copied %s backend configs: %s -> %s", service_mode, src, dst)

        if not copied:
            raise FileNotFoundError(f"No backend config folders found to copy under {run_dir}")

        return copied

    def _ensure_art_root(self, save_dir: Path, run_name: str) -> Path:
        base = Path(self.cfg.artifact_root) if self.cfg.artifact_root else save_dir
        out = base / "eval_runs" / run_name
        mkdir_p(out)
        return out

    def _read_max_batch_size(self, service_dir: Path, service_mode: str) -> Optional[int]:
        cfg_path = service_dir / ("agg/agg_config.yaml" if service_mode == "agg" else "disagg/decode_config.yaml")
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
        up = int(math.ceil(target / g) * g)

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

    def _auto_concurrency_values(self, max_bs: int) -> list[int]:
        K = 6  # noqa: N806
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

        fracs = [1 / 8, 1 / 4, 1 / 2, 3 / 4]
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

    def _start_service(self, service_dir: Path, log_file: Path) -> ServiceManager:
        start_rel = self.cfg.start_script.strip() or (
            "disagg/node_0_run.sh" if self.cfg.service_mode == "disagg" else "agg/node_0_run.sh"
        )
        sm = ServiceManager(
            workdir=service_dir,
            start_cmd=["bash", start_rel],
            port=self.cfg.port,
        )
        sm.start(log_path=log_file, cold_wait_s=self.cfg.coldstart_wait_s)
        sm.wait_healthy(timeout_s=self.cfg.health_timeout_s)
        return sm

    def _collect_gpu_once(self, where: Path) -> dict[str, Any]:
        """Quick NVML snapshot before benchmark for worker sanity check."""
        # Lazy import since monitoring might be disabled
        from .gpu import quick_nvml_snapshot

        snap = quick_nvml_snapshot()
        write_json(where / "gpu_snapshot_prebench.json", snap)
        return snap

    def _load_optimal_configs(self, config_dir: Path, target_tpot: float | None = None) -> dict[str, pd.DataFrame]:
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
                        optimal_configs["agg"] = best_agg
                        LOG.info(
                            f"Loaded optimal agg config: "
                            f"{best_agg['tokens/s/gpu'].iloc[0]:.2f} tokens/s/gpu, "
                            f"TPOT: {best_agg.get('tpot', [None]).iloc[0]} ms"
                        )
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
                        optimal_configs["disagg"] = best_disagg
                        LOG.info(
                            f"Loaded optimal disagg config: "
                            f"{best_disagg['tokens/s/gpu'].iloc[0]:.2f} tokens/s/gpu, "
                            f"TPOT: {best_disagg.get('tpot', [None]).iloc[0]} ms"
                        )
                    else:
                        LOG.warning("No disagg config found that meets TPOT constraint")
            except Exception as e:
                LOG.warning(f"Failed to load disagg pareto data: {e}")

        return optimal_configs

    def _get_best_config_under_tpot_constraint(
        self, pareto_df: pd.DataFrame, target_tpot: float | None
    ) -> pd.DataFrame:
        """Get the best configuration that meets TPOT constraint, similar to CLI logic."""
        if pareto_df.empty:
            return pd.DataFrame()

        # If no TPOT constraint, return the best overall configuration
        if target_tpot is None:
            best_config = pareto_df.loc[pareto_df["tokens/s/gpu"].idxmax()].to_frame().T
            LOG.info("No TPOT constraint specified, using best overall configuration")
            return best_config

        # Filter configurations that meet TPOT constraint
        if "tpot" not in pareto_df.columns:
            LOG.warning("TPOT column not found in pareto data, using best overall configuration")
            return pareto_df.loc[pareto_df["tokens/s/gpu"].idxmax()].to_frame().T

        # Find configurations that meet the TPOT constraint
        candidate_configs = pareto_df[pareto_df["tpot"] <= target_tpot].copy()

        if not candidate_configs.empty:
            # Among valid candidates, pick the one with highest tokens/s/gpu
            best_config = candidate_configs.loc[candidate_configs["tokens/s/gpu"].idxmax()].to_frame().T
            LOG.info(
                f"Found {len(candidate_configs)} configs meeting TPOT <= {target_tpot}ms, "
                f"selected best with {best_config['tokens/s/gpu'].iloc[0]:.2f} tokens/s/gpu"
            )
            return best_config
        else:
            LOG.warning(f"No config found with TPOT <= {target_tpot}ms, using best overall configuration")
            return pareto_df.loc[pareto_df["tokens/s/gpu"].idxmax()].to_frame().T

    def _convert_optimal_config_to_plot_format(self, config_df: pd.DataFrame, config_type: str) -> pd.DataFrame:
        """Convert optimal configuration DataFrame to format expected by ParetoPlot."""
        try:
            # Create a DataFrame with the required columns for plotting
            plot_df = pd.DataFrame()

            # Map the columns from the optimal config to the expected plot format
            if "tokens/s/user" in config_df.columns:
                plot_df["output_token_throughput_per_user_avg"] = config_df["tokens/s/user"]

            if "tokens/s/gpu" in config_df.columns:
                plot_df["output_token_throughput_avg"] = config_df["tokens/s/gpu"]

            # Add concurrency information if available
            if "concurrency" in config_df.columns:
                plot_df["load_label"] = config_df["concurrency"].astype(str)
            elif "bs" in config_df.columns:
                plot_df["load_label"] = config_df["bs"].astype(str)
            else:
                plot_df["load_label"] = f"{config_type}_optimal"

            LOG.debug(f"Converted optimal {config_type} config to plot format: {plot_df.to_dict()}")
            return plot_df

        except Exception as e:
            LOG.warning(f"Failed to convert optimal {config_type} config to plot format: {e}")
            return pd.DataFrame()

    def _run_benchmark(self, art_dir: Path, url: str, isl: int, osl: int, concurrency: list[int]) -> Path:
        args = self.cfg.cli_args
        model_path = str(getattr(args, "model_path", ""))
        served_model_name = str(getattr(args, "served_model_name", ""))
        venv_dir = str(getattr(args, "venv_dir", "/workspace/aic"))

        cfg = {
            "name": "genai_perf_eval",
            "base_folder": str(art_dir),
            "result_folder": "bench",
            "model": served_model_name or "unknown-model",
            "tokenizer": model_path or "unknown-tokenizer",
            "url": url,
            "endpoint_type": "chat",
            "input_sequence_length": int(isl),
            "output_sequence_length": int(osl),
            "concurrency": list(concurrency),
            "venv_dir": venv_dir,
        }
        run_genai_perf(cfg)
        return art_dir / "bench"

    def _analyze_and_plot(
        self,
        art_dir: Path,
        bench_dir: Path,
        workers_info: dict[str, int],
        service_mode: str,
        gpu_monitor_enabled: bool,
        gpu_csv: Path | None,
        optimal_configs: dict[str, pd.DataFrame] | None = None,
    ) -> None:
        """Parse genai-perf JSON and create plots."""
        df = parse_genai_perf(bench_dir)
        out_csv = art_dir / "bench_summary.csv"
        df.to_csv(out_csv, index=False)
        LOG.info("Saved summary: %s", out_csv)

        # Extract GPU count based on service_mode
        if service_mode == "disagg":
            p_workers = int(workers_info.get("PREFILL_WORKERS", 0) or 0)
            d_workers = int(workers_info.get("DECODE_WORKERS", 0) or 0)
            p_gpu = int(workers_info.get("PREFILL_GPU", 0) or 0)
            d_gpu = int(workers_info.get("DECODE_GPU", 0) or 0)
            total_gpus = p_workers * p_gpu + d_workers * d_gpu
            legend = f"disagg_{p_workers}p({p_gpu} gpu){d_workers}d({d_gpu} gpu)"
        else:
            # For agg service_mode, extract GPU count from config
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
                    # Convert the optimal config data to match the expected format
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

    def _get_agg_gpu_count(self, workers_info: dict[str, int]) -> int:
        """Simple helper to get GPU count for agg service_mode."""
        return workers_info.get("AGG_GPU_COUNT", 1)

    def _extract_workers_from_start_script(self, service_dir: Path) -> dict[str, int]:
        """
        For disagg, read disagg/node_0_run.sh and extract worker/GPU info.
        For agg, extract GPU count from agg_config.yaml.
        """
        if self.cfg.service_mode == "disagg":
            start_rel = self.cfg.start_script.strip() or "disagg/node_0_run.sh"
            script = service_dir / start_rel
            vals = parse_disagg_start_script(script)
            LOG.info("Parsed workers from %s: %s", script, vals)
            return vals
        else:
            # For agg service_mode, extract GPU count from config
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

        # determine cc
        if self.cfg.bench_concurrency and len(self.cfg.bench_concurrency) > 0:
            conc_list = list(self.cfg.bench_concurrency)
            LOG.info("Using provided benchmark concurrency: %s", conc_list)
        else:
            mbs = self._read_max_batch_size(service_dir, self.cfg.service_mode)
            if not mbs:
                # safe fallback if YAML missing
                conc_list = [1, 2, 4, 8, 16, 32]
                LOG.warning("Auto concurrency: max_batch_size not found -> fallback %s", conc_list)
            else:
                conc_list = self._auto_concurrency_values(int(mbs))
                LOG.info("Auto concurrency from max_batch_size=%s -> %s", mbs, conc_list)

        # prepare log path
        log_dir = Path(args.save_dir).resolve() / "log"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{run_name}_{self.cfg.service_mode}_p{self.cfg.port}.log"

        # 3) start + health
        self.service = self._start_service(service_dir, log_file)

        # artifacts root
        self.art_root = self._ensure_art_root(save_dir, run_name)

        # Worker info from script (disagg) or default (agg)
        workers_info = self._extract_workers_from_start_script(service_dir)
        write_json(self.art_root / "workers_extracted.json", workers_info)

        # GPU monitoring (conditional)
        if self.cfg.gpu_monitor:
            LOG.info("GPU monitor enabled")
            # lazy import to avoid NVML dependency when disabled
            from .gpu import GPUWatcher

            self._gpu_csv = self.art_root / "gpu_stats.csv"
            self._gpu_watcher = GPUWatcher(interval_s=self.cfg.nvml_interval_s, out_csv=self._gpu_csv)
            # optional one-shot snapshot before benchmark
            self._collect_gpu_once(self.art_root)
            self._gpu_watcher.start()
        else:
            LOG.info("GPU monitor disabled: skipping NVML sampling and timeseries.")
            self._gpu_csv = None

        # benchmark tokens
        isl = int(getattr(args, "isl", 0) or 0) or 1024
        osl = int(getattr(args, "osl", 0) or 0) or 128
        LOG.info("Benchmark tokens: isl=%d osl=%d", isl, osl)

        # wait 30s after health OK, before running benchmarks
        LOG.info("Health OK. Waiting 30 seconds before starting benchmark...")
        import time as _time

        _time.sleep(30)

        try:
            base_url = f"http://0.0.0.0:{self.cfg.port}"
            for i in range(self.cfg.runs):
                tag = f"{run_name}_r{i + 1}"
                art_dir = self.art_root / tag
                mkdir_p(art_dir)
                LOG.info("Run %d/%d -> %s (concurrency=%s)", i + 1, self.cfg.runs, art_dir, conc_list)
                bench_dir = self._run_benchmark(art_dir, url=base_url, isl=isl, osl=osl, concurrency=conc_list)
                self._analyze_and_plot(
                    art_dir=art_dir,
                    bench_dir=bench_dir,
                    workers_info=workers_info,
                    service_mode=self.cfg.service_mode,
                    gpu_monitor_enabled=self.cfg.gpu_monitor,
                    gpu_csv=self._gpu_csv,
                    optimal_configs=optimal_configs,
                )
        finally:
            if self._gpu_watcher:
                self._gpu_watcher.stop()

        return 0
