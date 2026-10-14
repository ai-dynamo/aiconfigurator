#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
TRT_SOURCE_COMMIT=14efb6ac673c0cbe828e1206cc5c7d5748d05ffa
DEEPEP_COMMIT=5be51b228a7c82dbdb213ea58e77bffd12b38af8
NVSHMEM_VERSION=3.2.5-1
NVSHMEM_ARCHIVE_SHA256=eb2c8fb3b7084c2db86bd9fd905387909f1dfd483e7b45f7b3c3d5fcf5374b5a
IMAGE_INDEX_DIGEST=sha256:1532b38814b3faf2affdb5ef01ca91468685d314ffb7e8926a0567595355ed88

die() { echo "ERROR: $*" >&2; exit 1; }
safe_existing_path() {
    local resolved; resolved=$(realpath -e -- "$2") || die "$1 does not exist: $2"
    case "${resolved}" in /mnt/cifs|/mnt/cifs/*|/mnt/nvdl|/mnt/nvdl/*) die "$1 uses prohibited storage" ;; esac
    printf '%s\n' "${resolved}"
}
for name in SYSTEM GPUS_PER_NODE BACKEND RUN_KIND CAMPAIGN_ROOT REPO_DIR CONTAINER_IMAGE WHEEL_DIR IMAGE_ARCH SLURM_JOB_ID SLURM_NODELIST; do
    [[ -n "${!name:-}" ]] || die "missing ${name}"
done
case "${BACKEND}" in trtllm_deepep_ht|trtllm_deepep_ll) ;; *) die "unsupported backend" ;; esac
case "${RUN_KIND}" in canary|full) ;; *) die "unsupported run kind" ;; esac
case "${SYSTEM}" in
    gb200) expected_gpus=4; gpu_token=GB200; compute_cap=10.0; expected_cuda_arches=100a-real ;;
    gb300) expected_gpus=4; gpu_token=GB300; compute_cap=10.3; expected_cuda_arches=103a-real ;;
    b200_sxm) expected_gpus=8; gpu_token=B200; compute_cap=10.0; expected_cuda_arches=100a-real ;;
    b300_sxm) expected_gpus=8; gpu_token=B300; compute_cap=10.3; expected_cuda_arches=103a-real ;;
    h100_sxm) expected_gpus=8; gpu_token=H100; compute_cap=9.0; expected_cuda_arches=90-real ;;
    h200_sxm) expected_gpus=8; gpu_token=H200; compute_cap=9.0; expected_cuda_arches=90-real ;;
    *) die "unsupported system" ;;
esac
[[ "${GPUS_PER_NODE}" == "${expected_gpus}" && "${SLURM_NTASKS:-}" == "${expected_gpus}" ]] || die "invalid single-node rank layout"
mapfile -t nodes < <(scontrol show hostnames "${SLURM_NODELIST}" | sort -u)
[[ "${#nodes[@]}" == 1 ]] || die "formal TRT-LLM campaign is single-node only"
if [[ "${SYSTEM}" == b200_sxm || "${SYSTEM}" == b300_sxm ]]; then
    [[ -n "${AIC_APPROVED_NODELIST:-}" && -n "${AIC_FABRIC_APPROVAL_ID:-}" ]] || die "missing fabric approval"
    [[ "$(scontrol show hostnames "${AIC_APPROVED_NODELIST}" | sort -u)" == "${nodes[0]}" ]] || die "allocation differs from approved nodelist"
fi

repo_dir=$(safe_existing_path "repository" "${REPO_DIR}")
campaign_root=$(safe_existing_path "campaign root" "${CAMPAIGN_ROOT}")
container_image=$(safe_existing_path "container image" "${CONTAINER_IMAGE}")
wheel_dir=$(safe_existing_path "wheel directory" "${WHEEL_DIR}")
[[ -z "$(git -C "${repo_dir}" status --porcelain)" ]] || die "repository checkout is dirty"
collector_ref=$(git -C "${repo_dir}" rev-parse HEAD)

read -r child_digest image_variant wheel_path wheel_sha dependency_dir < <(python3 - "${container_image}" "${container_image}.meta.json" "${wheel_dir}" "${SYSTEM}" "${IMAGE_ARCH}" "${expected_cuda_arches}" <<'PY'
import hashlib, json, re, sys
from pathlib import Path
image, image_meta, wheel_dir, system, arch, cuda_arches = sys.argv[1:]
image, image_meta, wheel_dir = Path(image), Path(image_meta), Path(wheel_dir)
im = json.loads(image_meta.read_text()); wm = json.loads((wheel_dir / "build_meta.json").read_text())
checks = {"schema_version": 1, "system": system, "architecture": arch,
          "configured_image_digest": "sha256:1532b38814b3faf2affdb5ef01ca91468685d314ffb7e8926a0567595355ed88",
          "source_commit": "14efb6ac673c0cbe828e1206cc5c7d5748d05ffa",
          "deep_ep": "5be51b228a7c82dbdb213ea58e77bffd12b38af8", "nvshmem": "3.2.5-1"}
for key, expected in checks.items():
    if im.get(key) != expected or wm.get(key) != expected: raise SystemExit(f"runtime {key} mismatch")
if wm.get("cuda_arches") != cuda_arches: raise SystemExit("runtime CUDA architecture mismatch")
if im.get("seed_provenance") != wm.get("seed_provenance"):
    raise SystemExit("runtime seed provenance mismatch")
seed = im.get("seed_provenance")
if seed is not None:
    required = {"mode", "source_system", "source_image_sha256", "source_image_meta_sha256", "source_image_digest"}
    if not isinstance(seed, dict) or not required <= set(seed) or seed["mode"] not in {"image", "runtime"}:
        raise SystemExit("invalid runtime seed provenance")
    for key in ("source_image_sha256", "source_image_meta_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(seed[key])): raise SystemExit("invalid runtime seed checksum")
    system_arches = {"gb200": "arm64", "gb300": "arm64", "b200_sxm": "amd64", "b300_sxm": "amd64", "h100_sxm": "amd64", "h200_sxm": "amd64"}
    if system_arches.get(seed["source_system"]) != arch or seed["source_image_digest"] != im["observed_image_digest"]:
        raise SystemExit("seed image identity mismatch")
    if seed["source_image_sha256"] != im["sqsh_sha256"]:
        raise SystemExit("seed image checksum provenance mismatch")
    if seed["mode"] == "runtime":
        if seed.get("cuda_arches") != cuda_arches: raise SystemExit("seed wheel CUDA architecture mismatch")
        if seed.get("source_wheel_sha256") != wm.get("wheel_sha256"): raise SystemExit("seed wheel checksum provenance mismatch")
        if not re.fullmatch(r"[0-9a-f]{64}", str(seed.get("source_wheel_meta_sha256", ""))):
            raise SystemExit("invalid seed wheel metadata checksum")
if hashlib.sha256(image.read_bytes()).hexdigest() != im["sqsh_sha256"]: raise SystemExit("sqsh checksum mismatch")
wheel = (wheel_dir / wm["wheel"]).resolve(strict=True)
if wheel.parent != wheel_dir.resolve() or hashlib.sha256(wheel.read_bytes()).hexdigest() != wm["wheel_sha256"]: raise SystemExit("wheel checksum mismatch")
if wm.get("python_requirements") != ["transformers==4.57.3"]:
    raise SystemExit("runtime Python requirement mismatch")
dependency_dir = (wheel_dir / "dependencies").resolve(strict=True)
expected_dependencies = wm.get("dependency_wheels")
if not isinstance(expected_dependencies, dict) or not expected_dependencies:
    raise SystemExit("runtime dependency wheel manifest missing")
actual_dependencies = {path.name for path in dependency_dir.glob("*.whl")}
if actual_dependencies != set(expected_dependencies):
    raise SystemExit("runtime dependency wheel set mismatch")
for name, digest in expected_dependencies.items():
    if Path(name).name != name or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise SystemExit("invalid dependency wheel manifest entry")
    if hashlib.sha256((dependency_dir / name).read_bytes()).hexdigest() != digest:
        raise SystemExit(f"dependency wheel checksum mismatch: {name}")
print(im["observed_image_digest"], im["image_variant"], wheel, wm["wheel_sha256"], dependency_dir)
PY
) || die "staged runtime validation failed"

gpu_inventory=$(nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader,nounits)
[[ "$(wc -l <<< "${gpu_inventory}" | xargs)" == "${expected_gpus}" ]] || die "GPU count mismatch"
while IFS=',' read -r name driver capability; do
    name=$(xargs <<< "${name}"); capability=$(xargs <<< "${capability}")
    [[ "${name^^}" == *"${gpu_token}"* && "${capability}" == "${compute_cap}" ]] || die "GPU identity mismatch: ${name}/${capability}"
done <<< "${gpu_inventory}"

staging_root="/tmp/aic-trtllm-a2a-${SLURM_JOB_ID}"
mkdir -p "${staging_root}"
output_dir="${staging_root}/output"
mkdir -p "${output_dir}"
python3 - "${output_dir}/runtime_evidence.json" "${SYSTEM}" "${nodes[0]}" "${child_digest}" "${image_variant}" "${wheel_sha}" "${collector_ref}" "${container_image}.meta.json" "${expected_cuda_arches}" <<'PY'
import json, sys
from pathlib import Path
out, system, node, child, variant, wheel_sha, collector_ref, image_meta, cuda_arches = sys.argv[1:]
seed = json.loads(Path(image_meta).read_text(encoding="utf-8")).get("seed_provenance")
payload = {"system": system, "node": node, "configured_image_digest": "sha256:1532b38814b3faf2affdb5ef01ca91468685d314ffb7e8926a0567595355ed88", "observed_image_digest": child, "image_variant": variant, "wheel_sha256": wheel_sha, "cuda_arches": cuda_arches, "collector_ref": collector_ref, "slurm_topology_verified": True}
if seed is not None:
    payload["seed_provenance"] = seed
Path(out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

canary_flag=""; [[ "${RUN_KIND}" != canary ]] || canary_flag=--canary
export AIC_REPO_DIR="${repo_dir}" AIC_OUTPUT_DIR="${output_dir}" AIC_WHEEL="${wheel_path}" AIC_DEPENDENCY_DIR="${dependency_dir}"
export AIC_CACHE_ROOT="${staging_root}/rank-cache"
export AIC_SYSTEM="${SYSTEM}" AIC_GPUS_PER_NODE="${GPUS_PER_NODE}" AIC_BACKEND="${BACKEND}" AIC_CANARY_FLAG="${canary_flag}"
export AIC_SOURCE_COMMIT="${TRT_SOURCE_COMMIT}" AIC_DEEPEP_COMMIT="${DEEPEP_COMMIT}" AIC_NVSHMEM_VERSION="${NVSHMEM_VERSION}" AIC_NVSHMEM_SHA="${NVSHMEM_ARCHIVE_SHA256}" AIC_IMAGE_INDEX_DIGEST="${IMAGE_INDEX_DIGEST}"
command='set -euo pipefail; target="/tmp/aic-trtllm-wheel-${SLURM_JOB_ID}-${SLURM_PROCID}"; rank_cache="${AIC_CACHE_ROOT}/${SLURM_PROCID}"; mkdir -p "${target}" "${rank_cache}/home" "${rank_cache}/xdg" "${rank_cache}/huggingface" "${rank_cache}/flashinfer" "${rank_cache}/triton"; export HOME="${rank_cache}/home" XDG_CACHE_HOME="${rank_cache}/xdg" HF_HOME="${rank_cache}/huggingface" FLASHINFER_WORKSPACE_DIR="${rank_cache}/flashinfer" TRITON_CACHE_DIR="${rank_cache}/triton"; python3 -m pip install --no-index --find-links "${AIC_DEPENDENCY_DIR}" --target "${target}" "transformers==4.57.3" >/dev/null; python3 -m pip install --no-deps --target "${target}" "${AIC_WHEEL}" >/dev/null; export PYTHONPATH="${target}:${AIC_REPO_DIR}:${PYTHONPATH:-}"; python3 -m collector.wideep.trtllm.collect_moe_a2a --system "${AIC_SYSTEM}" --gpus-per-node "${AIC_GPUS_PER_NODE}" --modes "${AIC_BACKEND}" --output-path "${AIC_OUTPUT_DIR}" --source-commit "${AIC_SOURCE_COMMIT}" --image-digest "${AIC_IMAGE_INDEX_DIGEST}" --deep-ep-commit "${AIC_DEEPEP_COMMIT}" --nvshmem-version "${AIC_NVSHMEM_VERSION}" --nvshmem-archive-sha256 "${AIC_NVSHMEM_SHA}" ${AIC_CANARY_FLAG}'
set +e
srun --nodes=1 --ntasks="${expected_gpus}" --ntasks-per-node="${expected_gpus}" --mpi=pmix \
    --container-image="${container_image}" --container-mounts="${repo_dir}:${repo_dir},${wheel_dir}:${wheel_dir},${staging_root}:${staging_root}" \
    --container-workdir="${staging_root}" bash -lc "${command}"
benchmark_status=$?
set -e

kind_root="${campaign_root}/${SYSTEM}/trtllm/${RUN_KIND}/1n/${BACKEND}"
preserve_failure_evidence() {
    local status=$1
    local reason=$2
    destination="${campaign_root}/failure_evidence/${SYSTEM}/trtllm/${RUN_KIND}/1n/${BACKEND}/job_${SLURM_JOB_ID}"
    mkdir -p "${destination}"
    cp -a "${output_dir}/." "${destination}/"
    python3 - "${destination}/artifact_checksums.json" "${destination}" \
        "${status}" "${SLURM_JOB_ID}" "${BACKEND}" "${RUN_KIND}" "${reason}" <<'PY'
import hashlib, json, sys
from pathlib import Path
out, root = Path(sys.argv[1]), Path(sys.argv[2])
payload = {"benchmark_status": int(sys.argv[3]), "job_id": sys.argv[4], "backend": sys.argv[5],
           "run_kind": sys.argv[6], "reason": sys.argv[7]}
(root / "job_failure.json").write_text(json.dumps(payload, sort_keys=True) + "\n")
checks = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.is_file() and p != out}
out.write_text(json.dumps(checks, indent=2, sort_keys=True) + "\n")
PY
}
# ``finalize_perf_files`` intentionally leaves its flock inode in place while
# finalizers may still race.  At this point the job-unique srun has completed,
# so no finalizer remains; remove the empty control file before either failure
# evidence preservation or transactional formal publication.
merge_lock="${output_dir}/moe_a2a_perf.parquet.mergelock"
if [[ -e "${merge_lock}" ]]; then
    if [[ ! -f "${merge_lock}" || -L "${merge_lock}" || -s "${merge_lock}" ]]; then
        preserve_failure_evidence 92 "invalid parquet merge-lock artifact"
        die "invalid parquet merge-lock artifact preserved at ${destination}"
    fi
    rm -f -- "${merge_lock}"
fi
if [[ "${benchmark_status}" -ne 0 ]]; then
    preserve_failure_evidence "${benchmark_status}" "collector command failed"
    die "collector failed with status ${benchmark_status}; all partial rows and rank evidence preserved at ${destination}"
fi
if [[ ! -f "${output_dir}/moe_a2a_perf.parquet" || ! -f "${output_dir}/collection_meta.yaml" ]]; then
    preserve_failure_evidence 90 "collector omitted finalized artifacts"
    die "collector omitted finalized artifacts; evidence preserved at ${destination}"
fi
if ! python3 - "${output_dir}/artifact_checksums.json" "${output_dir}" <<'PY'
import hashlib, json, sys
from pathlib import Path
out, root = Path(sys.argv[1]), Path(sys.argv[2])
allowed = {"moe_a2a_perf.parquet", "collection_meta.yaml", "runtime_evidence.json"}
actual = {p.name for p in root.iterdir() if p.is_file() and p != out}
if actual != allowed:
    raise SystemExit(f"unexpected formal artifact set: {sorted(actual)}")
checks = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sorted(allowed)}
out.write_text(json.dumps(checks, indent=2, sort_keys=True) + "\n")
PY
then
    preserve_failure_evidence 91 "unexpected formal artifact set"
    die "formal artifact validation failed; evidence preserved at ${destination}"
fi
touch "${output_dir}/SUCCESS"
destination="${kind_root}/job_${SLURM_JOB_ID}"
[[ ! -e "${destination}" ]] || die "refusing to overwrite campaign job output ${destination}"
publish_stage="${destination}.publish-${SLURM_JOB_ID}"
[[ ! -e "${publish_stage}" ]] || die "stale transactional publish stage ${publish_stage}"
mkdir -p "${publish_stage}"
cp -a "${output_dir}/." "${publish_stage}/"
mv -- "${publish_stage}" "${destination}"
echo "Published validated ${SYSTEM} ${BACKEND} ${RUN_KIND} artifacts to ${destination}"
