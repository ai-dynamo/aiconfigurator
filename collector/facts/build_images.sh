#!/bin/bash
# Rebuild the probe images. Bases are public; the fixes are one-liners applied
# via docker commit. Re-run whenever docker prune eats them (shared machine).
set -euxo pipefail

# --- vllm-probe:0.24.0-fix -------------------------------------------------
# tilelang ships a broken libcudart *stub* whose unresolved symbol poisons
# flashinfer.comm's ctypes load; tilelang is a HARD dep of DSV4 mhc on vllm,
# so it cannot just be removed. Replace the stub with a symlink to the real
# libcudart inside the image.
docker pull vllm/vllm-openai:v0.24.0
cid=$(docker run -d --entrypoint bash vllm/vllm-openai:v0.24.0 -c 'sleep infinity')
docker exec "$cid" bash -lc '
  set -e
  real=$(ldconfig -p | grep -m1 "libcudart.so.13\|libcudart.so.12" | awk "{print \$NF}")
  for stub in $(python3 - <<PY
import glob, tilelang, os
root = os.path.dirname(tilelang.__file__)
print("\n".join(glob.glob(root + "/**/libcudart.so*", recursive=True)))
PY
  ); do ln -sf "$real" "$stub"; done
  python3 -c "import flashinfer.comm" # must import cleanly now
'
docker commit "$cid" vllm-probe:0.24.0-fix
docker rm -f "$cid"

# --- sglang / trtllm --------------------------------------------------------
# Used as-is (no commit needed):
docker pull lmsysorg/sglang:v0.5.14
docker pull lmsysorg/sglang:v0.5.16
docker pull nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc23
# rc20 needed onnx pinned to 1.19.1 (trtllm-probe:1.3.0rc20-onnxfix); rc23
# does not need it — probe_trtllm's conditional cutlass stub covers both.
