# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import subprocess
import time
from pathlib import Path
from typing import List, Optional, TextIO

import requests

LOG = logging.getLogger(__name__)

def _is_models_ready(models_payload: dict) -> bool:
    data = models_payload.get("data")
    return isinstance(data, list) and len(data) > 0

def _is_health_ready(health_payload: dict) -> bool:
    status = (health_payload.get("status") or "").lower()
    eps = health_payload.get("endpoints")
    return status == "healthy" and isinstance(eps, list) and len(eps) > 0

class ServiceManager:
    def __init__(self, workdir: Path, start_cmd: List[str], port: int):
        self.workdir = Path(workdir)
        self.start_cmd = list(start_cmd)
        self.port = int(port)
        self._p: Optional[subprocess.Popen] = None
        self._log_fp: Optional[TextIO] = None
        self._log_path: Optional[Path] = None

    def _base(self) -> str:
        return f"http://0.0.0.0:{self.port}"

    def _url_health(self) -> str:
        return f"{self._base()}/health"

    def _url_models(self) -> str:
        return f"{self._base()}/v1/models"

    def start(self, *, log_path: Path, cold_wait_s: int = 10):
        """Start process and stream stdout/stderr into a log file."""
        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_fp = open(self._log_path, "a", encoding="utf-8", buffering=1)

        LOG.info("Starting service: %s (cwd=%s)  log=%s",
                 " ".join(self.start_cmd), self.workdir, self._log_path)
        self._p = subprocess.Popen(
            self.start_cmd,
            cwd=str(self.workdir),
            stdout=self._log_fp,
            stderr=self._log_fp,
            text=True,
            bufsize=1,
        )
        time.sleep(max(0, cold_wait_s))

    def wait_healthy(self, timeout_s: int = 600):
        LOG.info("Waiting for health at %s ...", self._base())
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            try:
                mh = requests.get(self._url_health(), timeout=2)
                mm = requests.get(self._url_models(), timeout=2)
                if mh.status_code == 200 and mm.status_code == 200:
                    try:
                        h_json = mh.json()
                        m_json = mm.json()
                    except json.JSONDecodeError:
                        time.sleep(2)
                        continue
                    if _is_health_ready(h_json) and _is_models_ready(m_json):
                        LOG.info("Health checks passed: status=healthy and models loaded.")
                        return
            except Exception:
                pass
            time.sleep(2)
        raise TimeoutError(f"Service did not become healthy within {timeout_s}s")

    def stop(self):
        if not self._p:
            # Close log fp if opened
            if self._log_fp:
                try: self._log_fp.close()
                except Exception: pass
            return
        LOG.info("Stopping service (SIGTERM)...")
        try:
            self._p.terminate()
        except Exception:
            pass
        try:
            self._p.wait(timeout=20)
            LOG.info("Service stopped (clean).")
        except Exception:
            LOG.warning("Terminate timed out; killing...")
            try:
                self._p.kill()
                self._p.wait(timeout=10)
            except Exception:
                pass
            LOG.info("Service killed.")
        finally:
            if self._log_fp:
                try: self._log_fp.close()
                except Exception: pass
