#!/bin/bash
# Rebuild probe images + the generator-CLI venv. ALL version pins come from
# targets.yaml (single source; bump versions there, never here). Re-run
# whenever docker prune eats images on this shared box.
set -euxo pipefail
cd "$(dirname "$0")/.."

readarray -t PINS < <(python3 - <<'PY'
import yaml
t = yaml.safe_load(open('targets.yaml'))
for be, cfg in t['backends'].items():
    for ver, img in cfg['images'].items():
        print(f"{be}|{ver}|{img}")
print("vllmbase|img|" + t['tooling']['vllm_probe_base_image'])
PY
)

VLLM_BASE=""
# the checkout itself declares which compiled core it needs — read, don't pin
CORE_WHEEL=$(python3 -c "
import re
print(re.search(r'aiconfigurator-core==([\w.]+)', open('aic/pyproject.toml').read()).group(1))")
for pin in "${PINS[@]}"; do
  IFS='|' read -r be ver img <<< "$pin"
  case "$be" in
    vllmbase) VLLM_BASE="$img" ;;
    vllm)     ;;  # built below from VLLM_BASE
    *)        docker pull "$img" ;;
  esac
done

# --- vllm-probe:<ver>-fix ----------------------------------------------------
# tilelang ships a broken libcudart stub (libcudart_stub.so) whose unresolved
# symbol poisons flashinfer.comm's ctypes load; tilelang is a HARD dep of DSV4
# mhc on vllm. Replace stubs with symlinks to the real libcudart.
docker pull "$VLLM_BASE"
cid=$(docker run -d --entrypoint bash "$VLLM_BASE" -c 'sleep infinity')
docker exec "$cid" bash -lc '
  set -e
  real=$(ldconfig -p | grep -m1 "libcudart.so.13\|libcudart.so.12" | awk "{print \$NF}")
  for stub in $(python3 - <<PY
import glob, tilelang, os
root = os.path.dirname(tilelang.__file__)
print("\n".join(glob.glob(root + "/**/libcudart*", recursive=True)))
PY
  ); do ln -sf "$real" "$stub"; done
  python3 -c "import flashinfer.comm" # must import cleanly now
'
VLLM_TAG=$(python3 -c "
import yaml
t = yaml.safe_load(open('targets.yaml'))
print(list(t['backends']['vllm']['images'].values())[0])")
docker commit "$cid" "$VLLM_TAG"
docker rm -f "$cid"

# --- generator CLI venv (golden pipeline) ------------------------------------
python3 -m venv venv_aic
./venv_aic/bin/pip install -q "aiconfigurator-core==$CORE_WHEEL" \
  jinja2 packaging 'numpy~=1.26.4' pandas plotext plotly prettytable pydantic pyarrow pyyaml tqdm matplotlib
./venv_aic/bin/pip install -q -e ./aic --no-deps
ln -sf "$(pwd)/venv_aic/lib/python3.10/site-packages/aiconfigurator_core/_aiconfigurator_core.abi3.so" \
       aic/aic-core/src/aiconfigurator_core/_aiconfigurator_core.abi3.so
