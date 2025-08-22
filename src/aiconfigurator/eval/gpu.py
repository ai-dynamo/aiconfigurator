# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional

LOG = logging.getLogger(__name__)

def _init_nvml():
    import pynvml as nvml
    nvml.nvmlInit()
    return nvml

def _get_compute_procs(nvml, h):
    for fn in (
        "nvmlDeviceGetComputeRunningProcesses_v3",
        "nvmlDeviceGetComputeRunningProcesses_v2",
        "nvmlDeviceGetComputeRunningProcesses",
    ):
        try:
            procs = getattr(nvml, fn)(h) or []
            return procs
        except AttributeError:
            continue
        except Exception:
            continue
    return []

def quick_nvml_snapshot() -> Dict[str, Any]:
    nv = _init_nvml()
    gpus = []
    try:
        n = nv.nvmlDeviceGetCount()
        for i in range(n):
            h = nv.nvmlDeviceGetHandleByIndex(i)
            try:
                name_raw = nv.nvmlDeviceGetName(h)
                name = name_raw.decode() if isinstance(name_raw, (bytes, bytearray)) else str(name_raw)
            except Exception:
                name = f"GPU{i}"
            mem = nv.nvmlDeviceGetMemoryInfo(h)
            util = nv.nvmlDeviceGetUtilizationRates(h)
            procs = _get_compute_procs(nv, h)
            gpus.append({
                "index": i,
                "name": name,
                "mem_used_mb": int(mem.used / (1024*1024)),
                "mem_total_mb": int(mem.total / (1024*1024)),
                "utilization": int(getattr(util, "gpu", 0)),
                "pids": [getattr(p, "pid", -1) for p in procs],
            })
    finally:
        try: nv.nvmlShutdown()
        except Exception: pass
    return {"gpus": gpus, "timestamp": time.time()}

@dataclass
class GPUWatcher:
    """
    Background sampler that writes CSV rows:
      timestamp, gpu_index, util_percent, mem_used_mb
    Designed to support high-frequency sampling (e.g., 0.01s).
    """
    interval_s: float
    out_csv: Path

    def __post_init__(self):
        self._th: threading.Thread | None = None
        self._stop = threading.Event()
        self._fp = None
        self._writer: Optional[csv.DictWriter] = None
        self._last_flush = 0.0

    def start(self):
        self.out_csv.parent.mkdir(parents=True, exist_ok=True)
        LOG.info("GPU watcher start: interval=%.4fs -> %s", self.interval_s, self.out_csv)
        self._th = threading.Thread(target=self._run, name="GPUWatcher", daemon=True)
        self._th.start()

    def stop(self):
        LOG.info("GPU watcher stop...")
        self._stop.set()
        if self._th:
            self._th.join(timeout=2.0)
        # close file
        try:
            if self._fp:
                self._fp.flush()
                self._fp.close()
        except Exception:
            pass

    def _open_writer(self):
        existed = self.out_csv.exists()
        self._fp = open(self.out_csv, "a", newline="")
        self._writer = csv.DictWriter(self._fp, fieldnames=["timestamp", "gpu_index", "util_percent", "mem_used_mb"])
        if not existed or self.out_csv.stat().st_size == 0:
            self._writer.writeheader()
        self._last_flush = time.time()

    def _run(self):
        try:
            self._open_writer()
        except Exception as e:
            LOG.warning("GPU watcher cannot open CSV: %s", e)
            return

        # Initialize NVML once, cache handles
        try:
            nv = _init_nvml()
            count = nv.nvmlDeviceGetCount()
            handles = [nv.nvmlDeviceGetHandleByIndex(i) for i in range(count)]
        except Exception as e:
            LOG.warning("NVML init failed in watcher: %s", e)
            return

        try:
            while not self._stop.is_set():
                ts = time.time()
                try:
                    for idx, h in enumerate(handles):
                        mem = nv.nvmlDeviceGetMemoryInfo(h)
                        util = nv.nvmlDeviceGetUtilizationRates(h)
                        self._writer.writerow({
                            "timestamp": ts,
                            "gpu_index": idx,
                            "util_percent": int(getattr(util, "gpu", 0)),
                            "mem_used_mb": int(mem.used / (1024*1024)),
                        })
                except Exception as e:
                    LOG.warning("GPU watcher sample error: %s", e)

                # flush at ~1 Hz to bound buffering (tunable)
                if ts - self._last_flush >= 1.0:
                    try:
                        self._fp.flush()
                        self._last_flush = ts
                    except Exception:
                        pass

                # precise wait
                self._stop.wait(self.interval_s)
        finally:
            try:
                nv.nvmlShutdown()
            except Exception:
                pass
