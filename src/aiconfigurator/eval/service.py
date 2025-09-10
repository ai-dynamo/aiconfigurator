# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import subprocess
import time
from pathlib import Path
from typing import List, Optional, TextIO, Tuple
import re

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

    def base_url(self) -> str:
        return self._base()

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
                try:
                    self._log_fp.close()
                except Exception:
                    pass
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
                try:
                    self._log_fp.close()
                except Exception:
                    pass


class K8sServiceManager(ServiceManager):
    """
    Start by applying a k8s manifest, wait for frontend Pod Ready,
    then port-forward <port>:<port> to localhost and reuse HTTP health checks.
    """

    def __init__(
        self,
        *,
        namespace: str,
        deploy_yaml: Path,
        port: int,
        frontend_selector: str,
        frontend_name_regex: str = "frontend",
        cr_name: str = "",
        context: str = "",
        pf_kind: str = "pod",
        pf_name: str = "",
        delete_on_stop: bool = False,
        wait_timeout_s: int = 900,
    ):
        super().__init__(workdir=deploy_yaml.parent, start_cmd=[], port=port)
        self.namespace = namespace
        self.deploy_yaml = Path(deploy_yaml)
        self.frontend_selector = frontend_selector
        self._re_frontend = re.compile(frontend_name_regex, re.IGNORECASE) if frontend_name_regex else None
        self.cr_name = cr_name
        self.context = context
        self.pf_kind = pf_kind
        self.pf_name = pf_name
        self.delete_on_stop = delete_on_stop
        self.wait_timeout_s = wait_timeout_s
        self._pf: Optional[subprocess.Popen] = None

    def _kubectl_base(self) -> List[str]:
        base = ["kubectl"]
        if self.context:
            base += ["--context", self.context]
        return base + ["-n", self.namespace]

    def _apply(self, log_fp: TextIO):
        cmd = self._kubectl_base() + ["apply", "-f", str(self.deploy_yaml)]
        LOG.info("kubectl apply: %s", " ".join(cmd))
        subprocess.run(cmd, stdout=log_fp, stderr=log_fp, check=True, text=True)

    def _get_frontend_pods(self) -> dict:
        cmd = self._kubectl_base() + ["get", "pods", "-l", self.frontend_selector, "-o", "json"]
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if r.returncode != 0:
            return {"items": []}
        try:
            return json.loads(r.stdout or "{}")
        except Exception:
            return {"items": []}

    def _get_pods_by_selector(self, selector: str) -> List[dict]:
        cmd = self._kubectl_base() + ["get", "pods", "-l", selector, "-o", "json"]
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if r.returncode != 0:
            return []
        try:
            obj = json.loads(r.stdout or "{}")
            return obj.get("items", []) or []
        except Exception:
            return []

    def _get_all_pods(self) -> List[dict]:
        cmd = self._kubectl_base() + ["get", "pods", "-o", "json"]
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if r.returncode != 0:
            return []
        try:
            obj = json.loads(r.stdout or "{}")
            return obj.get("items", []) or []
        except Exception:
            return []

    def _svc_exists(self, name: str) -> bool:
        cmd = self._kubectl_base() + ["get", "svc", name, "-o", "json"]
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return r.returncode == 0

    @staticmethod
    def _is_pod_ready(pod: dict) -> bool:
        for cond in (pod.get("status", {}).get("conditions") or []):
            if cond.get("type") == "Ready" and cond.get("status") == "True":
                return True
        return False

    def _pick_pf_target_kind_and_name(self) -> Optional[Tuple[str, str]]:
        # 0) explicit override
        if self.pf_name:
            return (self.pf_kind, self.pf_name)
        # 1) prefer svc by conventions: {CR}-frontend, {CR}
        if self.cr_name:
            for cand in (f"{self.cr_name}-frontend", f"{self.cr_name}"):
                if self._svc_exists(cand):
                    return ("svc", cand)
        # 2) label selectors
        for sel in [
            self.frontend_selector,
            "dynamo.nvidia.com/componentType in (frontend,Frontend)",
            "componentType in (frontend,Frontend)",
        ]:
            pods = self._get_pods_by_selector(sel)
            for p in pods:
                if self._is_pod_ready(p):
                    name = (p.get("metadata", {}) or {}).get("name", "")
                    if name:
                        return ("pod", name)
        # 3) name regex fallback
        if self._re_frontend:
            for p in self._get_all_pods():
                name = (p.get("metadata", {}) or {}).get("name", "")
                if name and self._re_frontend.search(name) and self._is_pod_ready(p):
                    return ("pod", name)
        return None

    def _start_port_forward(self, target_name: str, log_fp: TextIO):
        resource = f"{self.pf_kind}/{target_name}"
        cmd = self._kubectl_base() + ["port-forward", resource, f"{self.port}:{self.port}"]
        LOG.info("kubectl port-forward: %s", " ".join(cmd))
        # Keep running in background; logs to file
        self._pf = subprocess.Popen(cmd, stdout=log_fp, stderr=log_fp, text=True, bufsize=1)
        time.sleep(1.0)

    def _base(self) -> str:
        # Port-forward binds localhost by default
        return f"http://127.0.0.1:{self.port}"

    def start(self, *, log_path: Path, cold_wait_s: int = 10):
        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_fp = open(self._log_path, "a", encoding="utf-8", buffering=1)

        # 1) apply manifest
        self._apply(self._log_fp)

        # 2) wait for Ready pod / target and port-forward
        t0 = time.time()
        LOG.info(
            "Waiting for frontend target (ns=%s, selector='%s', cr='%s', name_regex='%s', timeout=%ss)...",
            self.namespace,
            self.frontend_selector,
            self.cr_name,
            getattr(self._re_frontend, "pattern", ""),
            self.wait_timeout_s,
        )
        last_log = 0.0
        while time.time() - t0 < max(self.wait_timeout_s, cold_wait_s):
            picked = self._pick_pf_target_kind_and_name()
            if picked:
                kind, name = picked
                self.pf_kind = kind  # record actual kind used
                self._start_port_forward(name, self._log_fp)
                LOG.info("k8s target ready: %s/%s (pf started)", kind, name)
                return
            now = time.time()
            if now - last_log >= 5.0:
                try:
                    pods = self._get_pods_by_selector(self.frontend_selector)
                    names = [(p.get("metadata", {}) or {}).get("name", "") for p in pods]
                    states = []
                    for p in pods:
                        nm = (p.get("metadata", {}) or {}).get("name", "")
                        states.append(f"{nm or '<noname>'}:{'Ready' if self._is_pod_ready(p) else 'NotReady'}")
                    LOG.info(
                        "Searching frontend... selector=%s, candidates=%s",
                        self.frontend_selector,
                        ", ".join(states) or "<none>",
                    )
                    if self.cr_name:
                        LOG.info("Also trying services: %s", f"{self.cr_name}-frontend / {self.cr_name}")
                except Exception:
                    pass
                last_log = now
            time.sleep(2.0)
        raise TimeoutError(
            f"Frontend target not Ready/Found within {self.wait_timeout_s}s (ns={self.namespace}, selector={self.frontend_selector}, cr={self.cr_name})"
        )

    def stop(self):
        LOG.info("Stopping k8s port-forward and (optionally) resources...")
        # 1) stop port-forward
        if self._pf:
            try:
                self._pf.terminate()
                self._pf.wait(timeout=5)
            except Exception:
                try:
                    self._pf.kill()
                except Exception:
                    pass
        # 2) delete resources if requested
        if self.delete_on_stop:
            try:
                cmd = self._kubectl_base() + ["delete", "-f", str(self.deploy_yaml), "--ignore-not-found=true"]
                subprocess.run(cmd, stdout=self._log_fp, stderr=self._log_fp, text=True)
            except Exception:
                pass
        if self._log_fp:
            try:
                self._log_fp.close()
            except Exception:
                pass
