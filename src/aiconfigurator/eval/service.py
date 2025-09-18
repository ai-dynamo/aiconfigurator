# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import subprocess
import time
import re
from pathlib import Path
from typing import List, Optional, TextIO, Tuple

import requests
import yaml

from textwrap import dedent
import shlex

def _shell_q(s: str) -> str:
    """Safe shell quoting."""
    return shlex.quote("" if s is None else str(s))

def _maybe_tok_arg(tokenizer: str) -> str:
    """Return CLI piece for --tokenizer only when non-empty."""
    tok = (tokenizer or "").strip()
    return f' --tokenizer {_shell_q(tok)}' if tok else ""

def _read_file_text(paths: List[Path]) -> str:
    for p in paths:
        try:
            if p.exists():
                return p.read_text(encoding="utf-8")
        except Exception:
            pass
    raise FileNotFoundError("bench_serving.py not found in any candidate paths")


LOG = logging.getLogger(__name__)

def _is_models_ready(models_payload: dict) -> bool:
    data = models_payload.get("data")
    return isinstance(data, list) and len(data) > 0

def _is_health_ready(health_payload: dict) -> bool:
    status = (health_payload.get("status") or "").lower()
    eps = health_payload.get("endpoints")
    return status == "healthy" and isinstance(eps, list) and len(eps) > 0

class ServiceManager:
    def __init__(self, workdir: Path, start_cmd: List[str], port: int, expected_model_id: str = ""):
        self.workdir = Path(workdir)
        self.start_cmd = list(start_cmd)
        self.port = int(port)
        self._expected_model_id = expected_model_id
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
                        if self._expected_model_id:
                            data = m_json.get("data", []) or []
                            ids = [str(x.get("id", "")) for x in data if isinstance(x, dict)]
                            if self._expected_model_id not in ids:
                                LOG.info("Models visible but expected id '%s' not present yet; have=%s",
                                         self._expected_model_id, ids)
                            else:
                                LOG.info("Health checks passed: healthy and model '%s' loaded.", self._expected_model_id)
                                return
                        else:
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


class K8sServiceManager(ServiceManager):
    """
    Apply k8s manifest, wait for frontend Pod/Service, (optionally) wait for workers Ready,
    then port-forward <port>:<port> to localhost and reuse HTTP health checks.
    """
    def __init__(self, *, namespace: str, deploy_yaml: Path, port: int,
                 frontend_selector: str, frontend_name_regex: str = "frontend", cr_name: str = "",
                 context: str = "",
                 pf_kind: str = "pod", pf_name: str = "",
                 delete_on_stop: bool = False, wait_timeout_s: int = 900,
                 wait_workers_ready: bool = True, expected_model_id: str = "", image_pull_token: str = ""):
        super().__init__(workdir=deploy_yaml.parent, start_cmd=[], port=port, expected_model_id=expected_model_id)
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
        self.wait_workers_ready = wait_workers_ready
        self._pf: Optional[subprocess.Popen] = None
        self.image_pull_token = image_pull_token

    def _apply_yaml_text(self, yaml_text: str) -> None:
        cmd = self._kubectl_base() + ["apply", "-f", "-"]
        LOG.info("kubectl apply (stdin yaml)")
        try:
            dry_cmd = self._kubectl_base() + ["apply", "--dry-run=client", "-f", "-"]
            dry = subprocess.run(
                dry_cmd, input=yaml_text, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            if dry.returncode != 0:
                LOG.error("kubectl apply --dry-run=client failed (rc=%s)\nSTDERR:\n%s\nSTDOUT:\n%s",
                        dry.returncode, dry.stderr.strip(), dry.stdout.strip())
                # Save the failed YAML for debugging
                try:
                    out_path = (self._log_path.parent if self._log_path else Path(".")) / "last_failed_apply.yaml"
                    out_path.write_text(yaml_text)
                    LOG.error("Saved failed YAML to: %s", out_path)
                except Exception:
                    pass
                raise subprocess.CalledProcessError(dry.returncode, dry_cmd, dry.stdout, dry.stderr)
        except Exception:
            pass

        res = subprocess.run(
            cmd,
            input=yaml_text,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if res.returncode != 0:
            LOG.error("kubectl apply failed (rc=%s)\nSTDERR:\n%s\nSTDOUT:\n%s",
                    res.returncode, (res.stderr or "").strip(), (res.stdout or "").strip())
            try:
                out_path = (self._log_path.parent if self._log_path else Path(".")) / "last_failed_apply.yaml"
                out_path.write_text(yaml_text)
                LOG.error("Saved failed YAML to: %s", out_path)
            except Exception:
                pass
            raise subprocess.CalledProcessError(res.returncode, cmd, res.stdout, res.stderr)
        else:
            if res.stdout:
                LOG.info("kubectl apply output:\n%s", res.stdout.strip())


    def _extract_image_and_secret_from_yaml(self) -> tuple[Optional[str], Optional[str]]:
        try:
            with open(self.deploy_yaml) as f:
                docs = list(yaml.safe_load_all(f))
        except Exception as e:
            LOG.warning("parse deploy yaml failed: %s", e)
            return None, None

        first_image = None
        first_secret = None

        for d in docs:
            if not isinstance(d, dict):
                continue
            spec = (d.get("spec") or {})
            services = (spec.get("services") or {})
            for _, svc in (services.items() if isinstance(services, dict) else []):
                eps = svc.get("extraPodSpec") or {}
                # image
                mc = eps.get("mainContainer") or {}
                img = mc.get("image")
                if img and not first_image:
                    first_image = str(img)
                # imagePullSecrets
                ips = eps.get("imagePullSecrets") or []
                if isinstance(ips, list):
                    for item in ips:
                        nm = (item or {}).get("name")
                        if nm:
                            first_secret = str(nm)
                            break
                if first_image and first_secret:
                    return first_image, first_secret
        return first_image, first_secret

    def _derive_docker_server(self, image: str) -> str:
        if not image:
            return "https://index.docker.io/v1/"
        head = image.split("/", 1)[0]
        if "." in head or ":" in head or head == "localhost":
            return f"https://{head}"
        return "https://index.docker.io/v1/"

    def _kubectl_base(self) -> List[str]:
        base = ["kubectl"]
        if self.context:
            base += ["--context", self.context]
        return base + ["-n", self.namespace]

    def _apply(self, log_fp: TextIO):
        cmd = self._kubectl_base() + ["apply", "-f", str(self.deploy_yaml)]
        LOG.info("kubectl apply: %s", " ".join(cmd))
        subprocess.run(cmd, stdout=log_fp, stderr=log_fp, check=True, text=True)

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
        # 1) prefer service by convention: {CR}-frontend or {CR}
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
            all_pods = self._get_all_pods()
            for p in all_pods:
                name = (p.get("metadata", {}) or {}).get("name", "")
                if name and self._re_frontend.search(name) and self._is_pod_ready(p):
                    return ("pod", name)
        return None

    def _start_port_forward(self, target_name: str, log_fp: TextIO):
        resource = f"{self.pf_kind}/{target_name}"
        cmd = self._kubectl_base() + ["port-forward", resource, f"{self.port}:{self.port}"]
        LOG.info("kubectl port-forward: %s", " ".join(cmd))
        self._pf = subprocess.Popen(cmd, stdout=log_fp, stderr=log_fp, text=True, bufsize=1)
        time.sleep(1.0)

    def _base(self) -> str:
        # Port-forward binds localhost by default
        return f"http://127.0.0.1:{self.port}"

    def start(self, *, log_path: Path, cold_wait_s: int = 10):
        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_fp = open(self._log_path, "a", encoding="utf-8", buffering=1)
        
        # Create imagePull secret based on token (if provided)
        try:
            if self.image_pull_token:
                image, secret_name = self._extract_image_and_secret_from_yaml()
                if not secret_name:
                    LOG.warning("k8s-image-pull-token is set but deploy yaml has no imagePullSecrets; "
                                "please add dynamo_config.k8s_image_pull_secret in template.")
                else:
                    server = self._derive_docker_server(image or "")
                    # Token parsing: if it contains a colon, treat as USERNAME:PASSWORD;
                    # otherwise use USERNAME=oauth2accesstoken and PASSWORD=<token>.
                    if ":" in self.image_pull_token:
                        user, pwd = self.image_pull_token.split(":", 1)
                    else:
                        user, pwd = "oauth2accesstoken", self.image_pull_token

                    create_cmd = self._kubectl_base() + [
                        "create", "secret", "docker-registry", secret_name,
                        f"--docker-server={server}",
                        f"--docker-username={user}",
                        f"--docker-password={pwd}",
                        "-o", "yaml", "--dry-run=client",
                    ]
                    LOG.info("Ensuring imagePull secret '%s' for registry '%s' ...", secret_name, server)
                    dry = subprocess.run(create_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    if dry.returncode != 0:
                        LOG.error("docker-registry secret dry-run failed: %s", dry.stderr)
                        # Continue flow but warn
                    else:
                        self._apply_yaml_text(dry.stdout)
        except Exception as e:
            LOG.warning("Ensure image pull secret by token failed: %s", e)

        # 1) apply manifest
        self._apply(self._log_fp)

        # 2) wait worker ready
        if self.wait_workers_ready:
            try:
                self._wait_workers_ready()
            except Exception as e:
                LOG.warning("wait_workers_ready failed: %s", e)

        # 3) wait frontend ready
        self._wait_frontend_ready()

        # 4) After Frontend and Workers are Ready, pick the port-forward target and start port-forward.
        t0 = time.time()
        LOG.info("Selecting port-forward target (ns=%s, selector='%s', cr='%s', timeout=%ss)...",
                self.namespace, self.frontend_selector, self.cr_name, self.wait_timeout_s)
        last_log = 0.0
        while time.time() - t0 < max(self.wait_timeout_s, cold_wait_s):
            picked = self._pick_pf_target_kind_and_name()
            if picked:
                kind, name = picked
                self.pf_kind = kind
                self._start_port_forward(name, self._log_fp)
                LOG.info("k8s target ready: %s/%s (pf started)", kind, name)
                return

            now = time.time()
            if now - last_log >= 5.0:
                try:
                    pods = self._get_pods_by_selector(self.frontend_selector)
                    states = []
                    for p in pods:
                        nm = (p.get('metadata',{}) or {}).get('name','')
                        states.append(f"{nm or '<noname>'}:{'Ready' if self._is_pod_ready(p) else 'NotReady'}")
                    LOG.info("Searching frontend... selector=%s, candidates=%s",
                            self.frontend_selector, ", ".join(states) or "<none>")
                    if self.cr_name:
                        LOG.info("Also trying services: %s / %s", f"{self.cr_name}-frontend", self.cr_name)
                except Exception:
                    pass
                last_log = now
            time.sleep(2.0)

        raise TimeoutError(
            f"Port-forward target not found within {self.wait_timeout_s}s "
            f"(ns={self.namespace}, selector={self.frontend_selector}, cr={self.cr_name})"
        )

    def _wait_pod_phase(self, pod_name: str, *, phases=("Succeeded",), fail_on=("Failed",), timeout_s: int = 7200):
        """
        For benchmarking pod.
        """
        t0 = time.time()
        last = 0.0
        while time.time() - t0 < timeout_s:
            cmd = self._kubectl_base() + ["get", "pod", pod_name, "-o", "json"]
            r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if r.returncode == 0:
                try:
                    obj = json.loads(r.stdout or "{}")
                    phase = ((obj.get("status") or {}).get("phase") or "")
                    if phase in phases:
                        LOG.info("Benchmark pod phase: %s", phase)
                        return phase
                    if phase in fail_on:
                        LOG.error("Benchmark pod failed (phase=%s).", phase)
                        return phase
                except Exception:
                    pass
            now = time.time()
            if now - last >= 10.0:
                LOG.info("Waiting benchmark pod '%s' ...", pod_name)
                last = now
            time.sleep(2.0)
        raise TimeoutError(f"Benchmark pod '{pod_name}' not finished within {timeout_s}s")

    def _kubectl_cp(self, src: str, dst: str, *, container: Optional[str] = None) -> int:
        cmd = self._kubectl_base() + ["cp", src, dst]
        if container:
            cmd = self._kubectl_base() + ["cp", src, dst, "-c", container]
        LOG.info("kubectl cp: %s", " ".join(cmd))
        r = subprocess.run(
            cmd,
            stdout=self._log_fp or subprocess.PIPE,
            stderr=self._log_fp or subprocess.PIPE,
            text=True,
        )
        if r.returncode != 0:
            LOG.warning("kubectl cp returned non-zero: %s\nSTDERR:\n%s\nSTDOUT:\n%s",
                        r.returncode, (r.stderr or "").strip(), (r.stdout or "").strip())
        else:
            if r.stdout:
                LOG.info("kubectl cp output:\n%s", r.stdout.strip())
        return r.returncode

    def _save_bench_pod_logs(self, pod_name: str, container: str, local_bench_dir: Path) -> None:
        """
        Save benchmark pod logs and a `kubectl describe` to local bench dir.
        """
        try:
            local_bench_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        # --- logs (binary) ---
        try:
            log_cmd = self._kubectl_base() + ["logs", pod_name, "-c", container, "--timestamps"]
            r = subprocess.run(log_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False)
            (local_bench_dir / "bench_pod.log").write_bytes(r.stdout or b"")
            if r.stderr:
                (local_bench_dir / "bench_pod.log.stderr").write_bytes(r.stderr or b"")
            try:
                (local_bench_dir / "bench_pod.log.utf8.txt").write_text(
                    (r.stdout or b"").decode("utf-8", errors="replace"),
                    encoding="utf-8"
                )
            except Exception:
                pass
            LOG.info("Saved bench pod logs to: %s", local_bench_dir / "bench_pod.log")
        except Exception as e:
            LOG.warning("Saving bench pod logs failed: %s", e)

        try:
            desc_cmd = self._kubectl_base() + ["describe", "pod", pod_name]
            d = subprocess.run(desc_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False)
            content = (d.stdout or b"").decode("utf-8", errors="replace")
            if d.stderr:
                content += "\n\n# STDERR\n" + (d.stderr or b"").decode("utf-8", errors="replace")
            (local_bench_dir / "bench_pod.describe.txt").write_text(content, encoding="utf-8")
            LOG.info("Saved bench pod describe to: %s", local_bench_dir / "bench_pod.describe.txt")
        except Exception as e:
            LOG.warning("Saving bench pod describe failed: %s", e)



    def _copy_bench_outputs_with_fallback(
        self,
        pod_name: str,
        container: str,
        artifact_dir_in_pod: str,
        local_bench_dir: Path,
    ) -> None:
        """
        Try `kubectl cp` (once + one retry). If it still fails,
        stream a tar.gz from inside the container using Python
        """
        local_bench_dir.mkdir(parents=True, exist_ok=True)

        # --- Attempt 1: kubectl cp (directory) ---
        rc = self._kubectl_cp(f"{pod_name}:{artifact_dir_in_pod}", str(local_bench_dir), container=container)
        if rc == 0:
            return

        # --- Attempt 2: one retry for kubectl cp ---
        time.sleep(1.5)
        LOG.info("Retrying kubectl cp ...")
        rc = self._kubectl_cp(f"{pod_name}:{artifact_dir_in_pod}", str(local_bench_dir), container=container)
        if rc == 0:
            return

        LOG.info("kubectl cp directory failed twice; fallback to Python streaming tar.gz over exec ...")

        # --- Fallback: stream tar.gz over stdout with Python (no tar dependency) ---
        py_cmd = (
            "python3",
            "-c",
            # Stream tar.gz of /tmp/genai_bench to stdout
            "import tarfile,sys,os; "
            "p='/tmp/genai_bench'; "
            "tf=tarfile.open(fileobj=sys.stdout.buffer, mode='w|gz'); "
            "tf.add(p, arcname=os.path.basename(p)); "
            "tf.close()"
        )
        cmd = self._kubectl_base() + ["exec", pod_name, "-c", container, "--", *py_cmd]
        LOG.info("kubectl exec streaming tar.gz: %s", " ".join(cmd))

        # Run and capture binary stdout
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False)
        if res.returncode != 0 or not res.stdout:
            LOG.error("kubectl exec streaming tar failed (rc=%s)\nSTDERR:\n%s",
                    res.returncode, (res.stderr or b"").decode(errors="ignore"))
            raise RuntimeError("Failed to stream bench tarball from pod via exec.")

        # Write to temp tar.gz then extract
        tmp_tar = local_bench_dir / "_bench_stream.tar.gz"
        try:
            tmp_tar.write_bytes(res.stdout)
            import tarfile
            with tarfile.open(tmp_tar, "r:gz") as tf:
                tf.extractall(local_bench_dir)
            LOG.info("Extracted streamed tarball to: %s", local_bench_dir)
        finally:
            try:
                tmp_tar.unlink(missing_ok=True)
            except Exception:
                pass

    def run_benchmark_in_pod(
        self, *,
        art_dir: Path,
        model: str,
        tokenizer: str,
        isl: int,
        osl: int,
        conc_list: List[int],
        runner: str = "genai-perf",
        bench_backend: Optional[str] = None,
    ) -> Path:
        """
        Launch a temporary Pod in Kubernetes to run genai-perf.
        Copy /tmp/genai_bench back to <art_dir>/bench while the container is still Running.
        If `kubectl cp` fails, fall back to streaming a tar.gz via `exec/cat`.
        Finally, delete the temp Pod.
        """
        # Resolve image & imagePullSecret from deploy YAML
        # image, pull_secret = self._extract_image_and_secret_from_yaml()
        image, pull_secret  = "python:3.11-slim", None
        # image, pull_secret  = "nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:0.4.0", None
        if not image:
            raise RuntimeError("Cannot resolve image from deploy yaml for benchmark pod.")

        # Resolve Frontend Service (Pod calls Service directly, no local port-forward)
        svc = self._resolve_frontend_service_name()
        if not svc:
            raise RuntimeError("Cannot resolve frontend service name for benchmark pod.")

        # Resolve Model Cache PVC + mountPath (reuse if available)
        pvc, mount_path = self._extract_model_cache_info()
        if not pvc or not mount_path:
            LOG.warning("Model cache PVC or mount path not found in CR; tokenizer may download from HF.")

        # Unique Pod name
        import uuid
        pod_name = f"bench-{self.cr_name or 'genai'}-{uuid.uuid4().hex[:6]}".lower()

        local_bench_dir = art_dir / "bench"
        local_bench_dir.mkdir(parents=True, exist_ok=True)

        # Workload script in container (dedented to avoid here-doc/indent issues)
        from textwrap import dedent
        conc_str = " ".join(str(int(c)) for c in conc_list if int(c) > 0)
        artifact_in_pod = "/tmp/genai_bench"

        runner = (runner or "genai-perf").strip().lower()
        if runner not in ("genai-perf", "bench-serving"):
            raise ValueError(f"Unsupported runner='{runner}'. Must be 'genai-perf' or 'bench-serving'.")

        # Common vars
        conc_str = " ".join(str(int(c)) for c in conc_list if int(c) > 0)
        artifact_in_pod = "/tmp/genai_bench"


        if runner == "genai-perf":
            run_script = dedent(f"""\
                set -euo pipefail
                ART="{artifact_in_pod}"
                mkdir -p "$ART"
                SVC="{svc}"
                PORT="{self.port}"
                URL="http://$SVC:$PORT"
                MODEL="{model}"
                TOKENIZER="{tokenizer}"
                ISL="{isl}"
                OSL="{osl}"
                CONCS="{conc_str}"
                echo "==> Using URL=$URL"
                echo "==> MODEL=$MODEL"
                echo "==> TOKENIZER=$TOKENIZER"
                for cc in $CONCS; do
                prof="profile_export_isl_${{ISL}}_osl_${{OSL}}_concurrency_${{cc}}.json"
                echo "==> Running cc=$cc -> $prof"
                genai-perf profile \\
                    -m "$MODEL" \\
                    --tokenizer "$TOKENIZER" \\
                    --endpoint-type chat \\
                    --url "$URL" \\
                    --streaming \\
                    --profile-export-file "$prof" \\
                    --artifact-dir "$ART" \\
                    --endpoint /v1/chat/completions \\
                    --synthetic-input-tokens-mean "$ISL" \\
                    --synthetic-input-tokens-stddev 0 \\
                    --output-tokens-mean "$OSL" \\
                    --output-tokens-stddev 0 \\
                    --extra-inputs max_tokens:"$OSL" \\
                    --extra-inputs min_tokens:"$OSL" \\
                    --extra-inputs ignore_eos:true \\
                    --extra-inputs "{{\\"nvext\\":{{\\"ignore_eos\\":true}}}}" \\
                    --concurrency "$cc" \\
                    --request-count "$((cc * 10))" \\
                    --warmup-request-count "$((cc * 2))" \\
                    --num-dataset-entries "$((cc * 12))" \\
                    -- -v --max-threads 256 \\
                        -H 'Authorization: Bearer NOT USED' \\
                        -H 'Accept: text/event-stream'
                echo "==> Done cc=$cc"
                done
                echo "==> All concs done."

                python3 -c "import tarfile,os; src='/tmp/genai_bench'; dst='/tmp/genai_bench_py.tar.gz'; \
                tf=tarfile.open(dst, 'w:gz'); tf.add(src, arcname=os.path.basename(src)); tf.close(); \
                print('PY_TAR_DONE')"

                # Keep container alive so we can copy while it's still Running
                echo "COPY_READY"
                sleep 1800
                """)
        else:
            # === bench-serving ===
            backend = (bench_backend or "dynamo-oai-chat").strip()
            bench_py_text = _read_file_text([
                Path(__file__).parent / "benchmarks" / "bench_serving.py",
                Path(__file__).parent / "bench_serving.py",
                Path.cwd() / "bench_serving.py",
            ])

            script_head = dedent(f"""\
                set -euo pipefail
                ART="{artifact_in_pod}"
                mkdir -p "$ART"
                SVC="{svc}"
                PORT="{self.port}"
                URL="http://$SVC:$PORT"
                MODEL={_shell_q(model)}
                TOKENIZER={_shell_q(tokenizer)}
                ISL="{isl}"
                OSL="{osl}"
                CONCS="{conc_str}"
                BACKEND={_shell_q(backend)}
                SEED="42"
                echo "==> Using URL=$URL"
                echo "==> MODEL=$MODEL"
                echo "==> TOKENIZER=$TOKENIZER"
                echo "==> BACKEND=$BACKEND"

                python3 -V
                python3 -m pip install --no-cache-dir --upgrade pip || true
                python3 -m pip install --no-cache-dir -U aiohttp numpy transformers tqdm requests huggingface_hub || \\
                python3 -m pip install --break-system-packages --no-cache-dir -U aiohttp numpy transformers tqdm requests huggingface_hub

                cat > /tmp/bench_serving.py <<'PY'
            """)
            script_tail = dedent("""\
            PY

                chmod +x /tmp/bench_serving.py
                for cc in $CONCS; do
                num_req="$((cc * 10))"
                out="$ART/bench_serving_${BACKEND}_isl_${ISL}_osl_${OSL}_cc_${cc}.jsonl"
                echo "==> Running cc=$cc, num-prompts=$num_req -> $out"
                python3 /tmp/bench_serving.py \
                    --backend "$BACKEND" \
                    --base-url "$URL" \
                    --model "$MODEL" \
                    --tokenizer "$TOKENIZER" \
                    --tokenize-prompt \
                    --dataset-name random \
                    --num-prompts "$num_req" \
                    --random-input-len "$ISL" \
                    --random-output-len "$OSL" \
                    --random-range-ratio 1 \
                    --max-concurrency "$cc" \
                    --throughput-denominator decode \
                    --seed "$SEED" \
                    --warmup-requests 0""" + _maybe_tok_arg(tokenizer) + """ \
                    --output-file "$out" || echo "WARN: bench_serving exit non-zero for cc=$cc"
                echo "==> Done cc=$cc"
            done
            echo "==> All concs done."
            
            python3 -c "import tarfile,os; src='/tmp/genai_bench'; dst='/tmp/genai_bench_py.tar.gz'; \
            tf=tarfile.open(dst, 'w:gz'); tf.add(src, arcname=os.path.basename(src)); tf.close(); \
            print('PY_TAR_DONE')"
            
            echo "COPY_READY"
            sleep 1800
            """)
            run_script = script_head + bench_py_text + "\n" + script_tail
        # Build Pod manifest
        pod_obj = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {"name": pod_name, "namespace": self.namespace},
            "spec": {
                "restartPolicy": "Never",
                "containers": [{
                    "name": "bench",
                    "image": image,
                    "imagePullPolicy": "IfNotPresent",
                    "workingDir": "/workspace",
                    "resources": {
                        "requests": {"cpu": "16", "memory": "64Gi", "ephemeral-storage": "40Gi"},
                        "limits":   {"cpu": "32", "memory": "128Gi", "ephemeral-storage": "80Gi", "nvidia.com/gpu": "1"},
                    },
                    "command": ["/bin/bash", "-lc"],
                    "args": [run_script],
                }],
            },
        }
        if pull_secret:
            pod_obj["spec"]["imagePullSecrets"] = [{"name": pull_secret}]
        if pvc and mount_path:
            pod_obj["spec"]["volumes"] = [{"name": "model-cache", "persistentVolumeClaim": {"claimName": pvc}}]
            pod_obj["spec"]["containers"][0]["volumeMounts"] = [{"name": "model-cache", "mountPath": mount_path}]

        yaml_text = yaml.safe_dump(pod_obj, sort_keys=False)

        # Create Pod
        LOG.info("Applying benchmark pod yaml ...")
        try:
            self._apply_yaml_text(yaml_text)
        except Exception as e:
            LOG.error("Apply benchmark pod failed: %s", e)
            raise

        # 1) Wait pod Running
        self._wait_pod_running(pod_name, timeout_s=7200)

        # 2) Wait for log marker that tarball is ready
        try:
            self._wait_pod_logs_contains(pod_name, container="bench", marker="PY_TAR_DONE",
                                        timeout_s=max(1800, 300 * max(1, len(conc_list))))
        except Exception:
            try:
                self._save_bench_pod_logs(pod_name, "bench", local_bench_dir)
            except Exception:
                pass
            self._wait_pod_logs_contains(pod_name, container="bench", marker="COPY_READY",
                                        timeout_s=300)

        try:
            self._save_bench_pod_logs(pod_name, "bench", local_bench_dir)
        except Exception:
            pass

        # 3) Copy artifacts while the container is still Running
        local_bench_dir = art_dir / "bench"
        try:
            self._copy_bench_outputs_with_fallback(
                pod_name=pod_name,
                container="bench",
                artifact_dir_in_pod=artifact_in_pod,
                local_bench_dir=local_bench_dir,
            )
        finally:
            try:
                self._save_bench_pod_logs(pod_name, "bench", local_bench_dir)
            except Exception:
                pass
            # Regardless of copy success, delete the temp pod
            try:
                del_cmd = self._kubectl_base() + ["delete", "pod", pod_name, "--ignore-not-found=true"]
                LOG.info("kubectl delete bench pod: %s", " ".join(del_cmd))
                subprocess.run(del_cmd, stdout=self._log_fp or subprocess.PIPE,
                            stderr=self._log_fp or subprocess.PIPE, text=True)
            except Exception:
                pass

        return local_bench_dir


    

    def _get_pod_phase(self, pod_name: str) -> str:
        cmd = self._kubectl_base() + ["get", "pod", pod_name, "-o", "json"]
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if r.returncode != 0:
            return ""
        try:
            obj = json.loads(r.stdout or "{}")
            return ((obj.get("status") or {}).get("phase") or "")
        except Exception:
            return ""

    def _wait_pod_running(self, pod_name: str, timeout_s: int = 600) -> None:
        t0 = time.time(); last = 0.0
        while time.time() - t0 < timeout_s:
            phase = self._get_pod_phase(pod_name)
            if phase == "Running":
                LOG.info("Benchmark pod is Running.")
                return
            if phase in ("Failed", "Succeeded"):
                try:
                    logs_cmd = self._kubectl_base() + ["logs", pod_name, "-c", "bench", "--tail=2000"]
                    res = subprocess.run(logs_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False)
                    tail_txt = ((res.stdout or b"").decode("utf-8", errors="ignore")).strip()[-2000:]
                    LOG.error("Bench pod exited early (phase=%s). Logs tail:\n%s", phase, tail_txt)
                except Exception:
                    pass
                raise RuntimeError(f"Benchmark pod reached phase={phase} before Running.")
            now = time.time()
            if now - last >= 5.0:
                LOG.info("Waiting benchmark pod to be Running (current phase=%s)...", phase or "<unknown>")
                last = now
            time.sleep(2.0)
        raise TimeoutError(f"Benchmark pod did not reach Running within {timeout_s}s")


    def _wait_pod_logs_contains(self, pod_name: str, container: str, marker: str, timeout_s: int = 1800) -> None:
        t0 = time.time(); last = 0.0
        while time.time() - t0 < timeout_s:
            cmd = self._kubectl_base() + ["logs", pod_name, "-c", container, "--tail=2000"]
            r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False)
            raw = r.stdout or b""
            text = raw.decode("utf-8", errors="ignore")   # Important: tolerant decode
            if marker in text:
                LOG.info("Detected log marker '%s' in benchmark pod logs.", marker)
                return
            phase = self._get_pod_phase(pod_name)
            if phase in ("Failed", "Succeeded"):
                tail = text[-2000:]
                LOG.error("Benchmark pod ended with phase=%s before marker '%s' appeared.\nLogs tail:\n%s",
                        phase, marker, tail)
                raise RuntimeError(f"Benchmark pod ended with phase={phase} before marker '{marker}'")
            now = time.time()
            if now - last >= 5.0:
                LOG.info("Waiting for benchmark pod to finish")
                last = now
            time.sleep(2.0)
        raise TimeoutError(f"Timeout waiting for log marker '{marker}' in bench pod logs.")




    def _wait_pods_gone(self, prefix: str, timeout_s: int = 180, poll_s: float = 2.0) -> None:
        if not prefix:
            return
        t0 = time.time()
        last_log = 0.0
        while time.time() - t0 < timeout_s:
            pods = []
            try:
                for p in self._get_all_pods():
                    name = (p.get("metadata", {}) or {}).get("name", "")
                    if name.startswith(prefix + "-"):
                        pods.append(name)
            except Exception:
                pass

            if not pods:
                LOG.info("All CR pods are gone for prefix '%s'.", prefix)
                return

            now = time.time()
            if now - last_log >= 5.0:
                LOG.info("Waiting CR pods to terminate (%ds/%ds): %s",
                         int(now - t0), timeout_s, ", ".join(pods))
                last_log = now
            time.sleep(poll_s)

        LOG.warning("Timeout while waiting for CR pods to terminate for prefix '%s'.", prefix)

    def stop(self):
        LOG.info("Stopping k8s service... (delete_on_stop=%s)", self.delete_on_stop)

        # 1) Stop port-forward
        if getattr(self, "_pf", None):
            try:
                self._pf.terminate()
                self._pf.wait(timeout=5)
                LOG.info("Port-forward terminated.")
            except Exception:
                try:
                    self._pf.kill()
                    self._pf.wait(timeout=5)
                    LOG.info("Port-forward killed.")
                except Exception:
                    LOG.warning("Failed to stop port-forward process cleanly.")
            finally:
                self._pf = None

        # 2) Delete CR
        if self.delete_on_stop:
            try:
                cmd = self._kubectl_base() + ["delete", "-f", str(self.deploy_yaml), "--ignore-not-found=true"]
                LOG.info("kubectl delete: %s", " ".join(cmd))
                stdout = self._log_fp if self._log_fp else subprocess.PIPE
                stderr = self._log_fp if self._log_fp else subprocess.PIPE
                res = subprocess.run(cmd, stdout=stdout, stderr=stderr, text=True)
                if res.returncode != 0:
                    LOG.warning("kubectl delete returned non-zero: %s", res.returncode)
            except Exception as e:
                LOG.warning("kubectl delete failed: %s", e)

            # 3) Wait CR-related pods to exit
            try:
                if self.cr_name:
                    self._wait_pods_gone(prefix=self.cr_name, timeout_s=180, poll_s=2.0)
            except Exception as e:
                LOG.warning("Wait for CR pods gone failed: %s", e)

        # 4) Close logs
        if self._log_fp:
            try:
                self._log_fp.close()
            except Exception:
                pass



    # ----- Worker readiness gate -----
    def _parse_cr_worker_replicas(self) -> Tuple[int, int, int]:
        """Return (prefill, decode, agg) replicas from CR deploy yaml."""
        with open(self.deploy_yaml) as f:
            docs = list(yaml.safe_load_all(f))
        pf = de = ag = 0
        for d in docs:
            if isinstance(d, dict) and d.get("kind") == "DynamoGraphDeployment":
                sv = (d.get("spec", {}) or {}).get("services", {}) or {}
                pf = int((sv.get("TRTLLMPrefillWorker", {}) or {}).get("replicas", 0) or 0)
                de = int((sv.get("TRTLLMDecodeWorker", {}) or {}).get("replicas", 0) or 0)
                ag = int((sv.get("TRTLLMWorker", {}) or {}).get("replicas", 0) or 0)
                break
        return pf, de, ag

    def _parse_cr_frontend_replicas(self) -> int:
        try:
            with open(self.deploy_yaml) as f:
                docs = list(yaml.safe_load_all(f))
            for d in docs:
                if isinstance(d, dict) and d.get("kind") == "DynamoGraphDeployment":
                    sv = (d.get("spec", {}) or {}).get("services", {}) or {}
                    fe = (sv.get("Frontend", {}) or {})
                    rep = int(fe.get("replicas", 0) or 0)
                    return rep if rep > 0 else 1
        except Exception as e:
            LOG.warning("Parse Frontend replicas failed: %s", e)
        return 1


    def _count_ready_by_name_regex(self, pattern: str) -> int:
        rx = re.compile(pattern, re.IGNORECASE)
        cnt = 0
        for p in self._get_all_pods():
            name = (p.get("metadata", {}) or {}).get("name", "")
            if name and rx.search(name) and self._is_pod_ready(p):
                cnt += 1
        return cnt

    def _wait_frontend_ready(self) -> None:
        need = self._parse_cr_frontend_replicas()
        t0 = time.time()
        last_log = 0.0
        LOG.info("Waiting Frontend pods Ready: need=%d (selector='%s')", need, self.frontend_selector)
        while time.time() - t0 < self.wait_timeout_s:
            have = 0
            names_states = []
            try:
                pods = self._get_pods_by_selector(self.frontend_selector)
                for p in pods:
                    nm = (p.get("metadata", {}) or {}).get("name", "")
                    ready = self._is_pod_ready(p)
                    names_states.append(f"{nm or '<noname>'}:{'Ready' if ready else 'NotReady'}")
                    if ready:
                        have += 1
            except Exception:
                pass

            if have < need and self._re_frontend:
                try:
                    have = max(have, self._count_ready_by_name_regex(self._re_frontend.pattern))
                except Exception:
                    pass

            if have >= need:
                LOG.info("Frontend pods Ready OK: %d/%d", have, need)
                return

            now = time.time()
            if now - last_log >= 5.0:
                LOG.info("Frontend readiness: %s  (%d/%d) ... waiting",
                        ", ".join(names_states) or "<none>", have, need)
                last_log = now
            time.sleep(2.0)

        raise TimeoutError(f"Frontend pods not Ready within {self.wait_timeout_s}s (need={need})")


    def _get_svcs_by_selector(self, selector: str) -> List[dict]:
        cmd = self._kubectl_base() + ["get", "svc", "-l", selector, "-o", "json"]
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if r.returncode != 0:
            return []
        try:
            obj = json.loads(r.stdout or "{}")
            return obj.get("items", []) or []
        except Exception:
            return []

    def _resolve_frontend_service_name(self) -> Optional[str]:
        if self.cr_name:
            for cand in (f"{self.cr_name}-frontend", f"{self.cr_name}"):
                if self._svc_exists(cand):
                    return cand
        for sel in [
            self.frontend_selector.replace("pod", "svc"),
            "dynamo.nvidia.com/componentType in (frontend,Frontend)",
            "componentType in (frontend,Frontend)",
        ]:
            for s in self._get_svcs_by_selector(sel):
                name = (s.get("metadata", {}) or {}).get("name", "")
                if name:
                    return name
        return None

    def _extract_model_cache_info(self) -> Tuple[Optional[str], Optional[str]]:
        try:
            with open(self.deploy_yaml) as f:
                docs = list(yaml.safe_load_all(f))
        except Exception as e:
            LOG.warning("parse deploy yaml for model cache failed: %s", e)
            return None, None

        for d in docs:
            if not isinstance(d, dict) or d.get("kind") != "DynamoGraphDeployment":
                continue
            services = ((d.get("spec") or {}).get("services") or {})
            fe = services.get("Frontend", {}) or {}
            eps = fe.get("extraPodSpec") or {}
            vols = eps.get("volumes") or []
            vol_pvc = {}
            for v in vols:
                nm = (v or {}).get("name")
                pvc = (((v or {}).get("persistentVolumeClaim") or {}).get("claimName"))
                if nm and pvc:
                    vol_pvc[nm] = pvc
            mc = (eps.get("mainContainer") or {})
            vms = mc.get("volumeMounts") or []
            for m in vms:
                vname = (m or {}).get("name")
                mpath = (m or {}).get("mountPath")
                if vname in vol_pvc and mpath:
                    return vol_pvc[vname], mpath
        return None, None



    def _wait_workers_ready(self):
        pf_rep, de_rep, ag_rep = self._parse_cr_worker_replicas()
        if (pf_rep, de_rep, ag_rep) == (0, 0, 0):
            LOG.info("No worker replicas declared in CR (maybe external); skipping worker readiness gate.")
            return
        base = re.escape(self.cr_name) if self.cr_name else ""
        patterns = []
        if pf_rep > 0:
            patterns.append(("prefill", pf_rep, rf"^{base}-\d+-trtllmprefillworker-"))
        if de_rep > 0:
            patterns.append(("decode", de_rep, rf"^{base}-\d+-trtllmdecodeworker-"))
        if ag_rep > 0:
            patterns.append(("agg",    ag_rep, rf"^{base}-\d+-trtllmworker-"))
        t0 = time.time()
        last = 0.0
        LOG.info("Waiting worker pods Ready per CR: %s", ", ".join([f"{k}={r}" for k,r,_ in patterns]) or "<none>")
        while time.time() - t0 < self.wait_timeout_s:
            ok = True
            statuses = []
            for tag, need, pat in patterns:
                have = self._count_ready_by_name_regex(pat)
                statuses.append(f"{tag}:{have}/{need}")
                if have < need:
                    ok = False
            if ok:
                LOG.info("All worker pods Ready: %s", ", ".join(statuses))
                return
            now = time.time()
            if now - last >= 5.0:
                LOG.info("Worker readiness: %s ... waiting", ", ".join(statuses))
                last = now
            time.sleep(2.0)
        raise TimeoutError(f"Worker pods not Ready within {self.wait_timeout_s}s")
